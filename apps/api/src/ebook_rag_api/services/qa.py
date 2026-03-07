import re
from dataclasses import dataclass
from dataclasses import replace
from functools import lru_cache
from math import ceil
from time import perf_counter
from typing import Protocol

import httpx
from sqlalchemy.orm import Session

from ebook_rag_api.core.config import get_settings
from ebook_rag_api.services.retrieval import ChunkSearchMatch, normalize_query, search_chunks
from ebook_rag_api.services.text import (
    TOKEN_RE,
    contains_normalized_phrase,
    extract_anchor_terms,
    extract_constraint_terms,
    extract_named_subject_terms,
    has_explicit_date,
    has_nickname_alias,
    has_temporal_marker,
    longest_matching_query_run,
    metadata_noise_score,
    normalized_token_sequence,
    query_run_bonus,
    tokenize_terms,
)

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
CLAUSE_SPLIT_RE = re.compile(r"(?<=[,;:])\s+|\s+(?:and|but|while|then)\s+", re.IGNORECASE)


@dataclass(frozen=True)
class RetrievedChunkContext:
    chunk_id: str
    document_id: str
    document_title: str | None
    document_filename: str
    chunk_index: int
    page_start: int
    page_end: int
    text: str
    token_estimate: int = 0
    dense_score: float = 0.0
    lexical_score: float = 0.0
    hybrid_score: float = 0.0
    rerank_score: float = 0.0
    score: float = 0.0


@dataclass(frozen=True)
class GeneratedAnswer:
    answer_text: str
    supported: bool
    citations: list[RetrievedChunkContext]


@dataclass(frozen=True)
class SentenceCandidate:
    sentence: str
    context: RetrievedChunkContext
    score: float
    terms: set[str]


@dataclass(frozen=True)
class QuestionFacet:
    text: str
    terms: set[str]
    anchor_terms: set[str]
    constraint_terms: set[str]


@dataclass(frozen=True)
class QATimingBreakdown:
    normalization_ms: float
    retrieval_ms: float
    context_assembly_ms: float
    answer_generation_ms: float
    total_ms: float


@dataclass(frozen=True)
class QATrace:
    answer_provider: str
    retrieved_chunks: list[RetrievedChunkContext]
    selected_contexts: list[RetrievedChunkContext]
    cited_contexts: list[RetrievedChunkContext]
    prompt_snapshot: str
    timings: QATimingBreakdown
    answer: GeneratedAnswer


class AnswerProvider(Protocol):
    def generate_answer(
        self, question: str, contexts: list[RetrievedChunkContext]
    ) -> GeneratedAnswer:
        ...


class ExtractiveAnswerProvider:
    def generate_answer(
        self, question: str, contexts: list[RetrievedChunkContext]
    ) -> GeneratedAnswer:
        if not contexts:
            return _unsupported_answer()

        question_facets = _build_question_facets(question)
        question_terms = set().union(*(facet.terms for facet in question_facets))
        sentence_candidates = _build_sentence_candidates(
            question=question,
            question_terms=question_terms,
            contexts=contexts,
        )
        if not sentence_candidates or sentence_candidates[0].score < 0.2:
            return _unsupported_answer()

        selected_candidates = _select_answer_sentences(
            question_facets=question_facets,
            sentence_candidates=sentence_candidates,
        )
        if not selected_candidates:
            return _unsupported_answer()
        if not _answer_has_sufficient_support(
            question_facets=question_facets,
            selected_candidates=selected_candidates,
        ):
            return _unsupported_answer()

        answer_text = " ".join(
            sentence if sentence.endswith((".", "!", "?")) else f"{sentence}."
            for sentence in [candidate.sentence for candidate in selected_candidates]
        )
        citations = select_evidence_citations(
            answer_text=answer_text,
            contexts=contexts,
            primary_context=selected_candidates[0].context,
            question_text=question,
        )

        return GeneratedAnswer(answer_text=answer_text, supported=True, citations=citations)


class OpenAICompatibleAnswerProvider:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: float,
        temperature: float,
        max_tokens: int,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate_answer(
        self, question: str, contexts: list[RetrievedChunkContext]
    ) -> GeneratedAnswer:
        if not contexts:
            return _unsupported_answer()

        prompt = build_qa_prompt(question=question, contexts=contexts)
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You answer questions using only the provided document context. "
                        "If the evidence is insufficient, reply exactly with "
                        "'INSUFFICIENT_SUPPORT'."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                response = client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()
        except httpx.HTTPError:
            return _unsupported_answer()

        answer_text = extract_chat_completion_text(response.json())
        if not answer_text or answer_text.strip() == "INSUFFICIENT_SUPPORT":
            return _unsupported_answer()

        return GeneratedAnswer(
            answer_text=answer_text.strip(),
            supported=True,
            citations=select_evidence_citations(
                answer_text=answer_text.strip(),
                contexts=contexts,
                question_text=question,
            ),
        )


@lru_cache(maxsize=1)
def get_answer_provider() -> AnswerProvider:
    settings = get_settings()
    if settings.answer_provider == "extractive":
        return ExtractiveAnswerProvider()
    if settings.answer_provider == "openai_compatible":
        return OpenAICompatibleAnswerProvider(
            base_url=settings.answer_base_url or settings.llm_base_url,
            api_key=settings.answer_api_key or settings.llm_api_key,
            model=settings.answer_model or settings.llm_model,
            timeout_seconds=settings.answer_timeout_seconds or settings.llm_timeout_seconds,
            temperature=settings.answer_temperature
            if settings.answer_temperature is not None
            else settings.llm_temperature,
            max_tokens=settings.answer_max_tokens or settings.llm_max_tokens,
        )
    raise ValueError(f"Unsupported answer provider: {settings.answer_provider}")


def ask_question(
    session: Session, question: str, top_k: int, document_id: str | None = None
) -> tuple[str, list[RetrievedChunkContext], GeneratedAnswer]:
    normalized_question, trace = ask_question_with_trace(
        session=session,
        question=question,
        top_k=top_k,
        document_id=document_id,
    )
    return normalized_question, trace.retrieved_chunks, trace.answer


def ask_question_with_trace(
    session: Session,
    question: str,
    top_k: int,
    document_id: str | None = None,
    include_prompt_snapshot: bool = True,
) -> tuple[str, QATrace]:
    started_at = perf_counter()
    normalized_question = normalize_query(question)
    normalized_at = perf_counter()
    _, matches = search_chunks(
        session=session,
        query=normalized_question,
        top_k=top_k,
        document_id=document_id,
    )
    retrieved_at = perf_counter()
    retrieved_chunks = [build_chunk_context(match) for match in matches]
    assembled_contexts = assemble_answer_contexts(
        question=normalized_question,
        contexts=retrieved_chunks,
    )
    answer_provider = get_answer_provider()
    selected_contexts = (
        retrieved_chunks
        if isinstance(answer_provider, ExtractiveAnswerProvider)
        else assembled_contexts
    )
    prompt_snapshot = (
        build_qa_prompt(
            question=normalized_question,
            contexts=selected_contexts,
        )
        if include_prompt_snapshot
        else ""
    )
    context_assembled_at = perf_counter()
    answer = answer_provider.generate_answer(
        question=normalized_question,
        contexts=selected_contexts,
    )
    answered_at = perf_counter()
    trace = QATrace(
        answer_provider=type(answer_provider).__name__,
        retrieved_chunks=retrieved_chunks,
        selected_contexts=selected_contexts,
        cited_contexts=answer.citations,
        prompt_snapshot=prompt_snapshot,
        timings=QATimingBreakdown(
            normalization_ms=(normalized_at - started_at) * 1000,
            retrieval_ms=(retrieved_at - normalized_at) * 1000,
            context_assembly_ms=(context_assembled_at - retrieved_at) * 1000,
            answer_generation_ms=(answered_at - context_assembled_at) * 1000,
            total_ms=(answered_at - started_at) * 1000,
        ),
        answer=answer,
    )
    return normalized_question, trace


def build_chunk_context(match: ChunkSearchMatch) -> RetrievedChunkContext:
    chunk = match.chunk
    return RetrievedChunkContext(
        chunk_id=chunk.id,
        document_id=chunk.document_id,
        document_title=chunk.document.title,
        document_filename=chunk.document.original_filename,
        chunk_index=chunk.chunk_index,
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        text=chunk.text,
        token_estimate=chunk.token_estimate,
        dense_score=match.dense_score,
        lexical_score=match.lexical_score,
        hybrid_score=match.hybrid_score,
        rerank_score=match.rerank_score,
        score=match.score,
    )


def assemble_answer_contexts(
    contexts: list[RetrievedChunkContext],
    question: str = "",
) -> list[RetrievedChunkContext]:
    if not contexts:
        return []

    question_terms = _tokenize(question)
    ranked_contexts = _rank_contexts_for_selection(
        question=question,
        question_terms=question_terms,
        contexts=contexts,
    )
    selected: list[RetrievedChunkContext] = []
    consumed_chunk_ids: set[str] = set()
    covered_terms: set[str] = set()
    token_budget = 1400
    remaining_budget = token_budget

    for context in _deduplicate_contexts(ranked_contexts):
        if context.chunk_id in consumed_chunk_ids:
            continue
        normalized_cost = max(context.token_estimate, _estimate_tokens(context.text), 1)
        if normalized_cost > remaining_budget and selected:
            continue
        context_terms = _tokenize(context.text)
        adds_term_coverage = bool(question_terms and (context_terms - covered_terms) & question_terms)
        if selected and not adds_term_coverage and _is_same_location_as_existing(context, selected):
            continue

        selected.append(context)
        consumed_chunk_ids.add(context.chunk_id)
        covered_terms.update(context_terms & question_terms)
        remaining_budget -= normalized_cost
        if remaining_budget <= 0:
            break

        adjacent_context = _select_adjacent_context(
            base_context=context,
            all_contexts=contexts,
            remaining_budget=remaining_budget,
            consumed_chunk_ids=consumed_chunk_ids,
            question_terms=question_terms,
        )
        if adjacent_context is not None:
            adjacent_cost = max(
                adjacent_context.token_estimate,
                _estimate_tokens(adjacent_context.text),
                1,
            )
            selected.append(adjacent_context)
            consumed_chunk_ids.add(adjacent_context.chunk_id)
            covered_terms.update(_tokenize(adjacent_context.text) & question_terms)
            remaining_budget -= adjacent_cost
            if remaining_budget <= 0:
                break

    if selected:
        return selected

    smallest_context = min(
        contexts,
        key=lambda context: max(context.token_estimate, _estimate_tokens(context.text), 1),
    )
    return [smallest_context]


def _deduplicate_contexts(
    contexts: list[RetrievedChunkContext],
) -> list[RetrievedChunkContext]:
    deduplicated: list[RetrievedChunkContext] = []
    seen_texts: list[set[str]] = []

    for context in contexts:
        terms = _tokenize(context.text)
        is_duplicate = any(
            _jaccard_similarity(terms, seen_terms) >= 0.9 for seen_terms in seen_texts
        )
        if is_duplicate:
            continue
        deduplicated.append(context)
        seen_texts.append(terms)
    return deduplicated


def _select_adjacent_context(
    *,
    base_context: RetrievedChunkContext,
    all_contexts: list[RetrievedChunkContext],
    remaining_budget: int,
    consumed_chunk_ids: set[str],
    question_terms: set[str],
) -> RetrievedChunkContext | None:
    adjacent_contexts = [
        candidate
        for candidate in all_contexts
        if candidate.document_id == base_context.document_id
        and candidate.chunk_id != base_context.chunk_id
        and candidate.chunk_id not in consumed_chunk_ids
        and abs(candidate.chunk_index - base_context.chunk_index) == 1
    ]
    if not adjacent_contexts:
        return None

    best_candidate = max(
        adjacent_contexts,
        key=lambda context: _adjacent_context_sort_key(
            base_context=base_context,
            candidate=context,
            question_terms=question_terms,
        ),
    )
    candidate_cost = max(
        best_candidate.token_estimate,
        _estimate_tokens(best_candidate.text),
        1,
    )
    if candidate_cost > remaining_budget:
        return None
    if not _adjacent_context_is_useful(
        base_context=base_context,
        candidate=best_candidate,
        question_terms=question_terms,
    ):
        return None
    return best_candidate


def _adjacent_context_sort_key(
    *,
    base_context: RetrievedChunkContext,
    candidate: RetrievedChunkContext,
    question_terms: set[str],
) -> tuple[float, float, float]:
    base_terms = _tokenize(base_context.text)
    candidate_terms = _tokenize(candidate.text)
    added_question_terms = len((candidate_terms - base_terms) & question_terms)
    shared_terms = len(candidate_terms & base_terms)
    return (
        float(added_question_terms),
        float(_normalized_term_overlap(question_terms, candidate_terms)),
        float(shared_terms),
    )


def _adjacent_context_is_useful(
    *,
    base_context: RetrievedChunkContext,
    candidate: RetrievedChunkContext,
    question_terms: set[str],
) -> bool:
    if not question_terms:
        return True
    base_terms = _tokenize(base_context.text)
    candidate_terms = _tokenize(candidate.text)
    if (candidate_terms - base_terms) & question_terms:
        return True
    if _contains_question_phrase_tokens(question_terms=question_terms, text=candidate.text):
        return True
    return len(base_terms & candidate_terms) >= 3 and bool(candidate_terms & question_terms)


def _estimate_tokens(text: str) -> int:
    return max(1, len(TOKEN_RE.findall(text)))


def _normalized_term_overlap(question_terms: set[str], context_terms: set[str]) -> float:
    if not question_terms:
        return 0.0
    return len(question_terms & context_terms) / len(question_terms)


def _contains_question_phrase_tokens(question_terms: set[str], text: str) -> bool:
    if len(question_terms) < 2:
        return False
    normalized_text = [token.lower() for token in TOKEN_RE.findall(text)]
    return any(
        left in question_terms and right in question_terms
        for left, right in zip(normalized_text, normalized_text[1:], strict=False)
    )


def _jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _tokenize(text: str) -> set[str]:
    return tokenize_terms(text, drop_stopwords=True)


def _split_sentences(text: str) -> list[str]:
    normalized_text = re.sub(r"\s*\n+\s*", " ", text).strip()
    if not normalized_text:
        return []
    raw_sentences = [
        sentence.strip()
        for sentence in SENTENCE_SPLIT_RE.split(normalized_text)
        if sentence.strip()
    ]
    sentences: list[str] = []
    pending_prefix = ""
    for sentence in raw_sentences:
        tokens = TOKEN_RE.findall(sentence)
        if pending_prefix:
            sentence = f"{pending_prefix} {sentence}".strip()
            pending_prefix = ""
            tokens = TOKEN_RE.findall(sentence)
        if tokens and len(tokens[-1]) == 1:
            pending_prefix = sentence
            continue
        if len(tokens) <= 2 and any(len(token) == 1 for token in tokens):
            pending_prefix = sentence
            continue
        sentences.append(sentence)
    if pending_prefix:
        if sentences:
            sentences[-1] = f"{sentences[-1]} {pending_prefix}".strip()
        else:
            sentences.append(pending_prefix)
    return sentences


def build_qa_prompt(question: str, contexts: list[RetrievedChunkContext]) -> str:
    context_blocks = []
    for context in contexts:
        label = (
            f"Document: {context.document_title or context.document_filename} | "
            f"Pages: {context.page_start}-{context.page_end} | "
            f"Chunk: {context.chunk_index} | Score: {context.score:.4f}"
        )
        context_blocks.append(f"{label}\n{context.text}")

    joined_context = "\n\n---\n\n".join(context_blocks)
    return (
        f"Question: {question}\n\n"
        "Context:\n"
        f"{joined_context}\n\n"
        "Instructions:\n"
        "- Answer only from the context.\n"
        "- Keep the answer concise and factual.\n"
        "- If the context is insufficient, reply exactly with INSUFFICIENT_SUPPORT.\n"
    )


def extract_chat_completion_text(payload: dict) -> str:
    choices = payload.get("choices", [])
    if not choices:
        return ""
    first_choice = choices[0]
    message = first_choice.get("message", {})
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_segments = [
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        return "".join(text_segments)
    return ""


def select_evidence_citations(
    *,
    answer_text: str,
    contexts: list[RetrievedChunkContext],
    primary_context: RetrievedChunkContext | None = None,
    question_text: str = "",
) -> list[RetrievedChunkContext]:
    citations: list[RetrievedChunkContext] = []
    seen_chunk_ids: set[str] = set()
    support_units = _split_support_units(answer_text)
    for index, support_unit in enumerate(support_units):
        ranked = _rank_evidence_contexts(
            answer_text=support_unit,
            question_text=question_text,
            contexts=contexts,
            primary_context=primary_context if index == 0 else None,
        )
        if not ranked:
            continue
        _, context = ranked[0]
        if context.chunk_id in seen_chunk_ids:
            continue
        citations.append(context)
        seen_chunk_ids.add(context.chunk_id)
        if len(citations) == min(3, max(len(support_units), 1)):
            break

    if citations:
        return citations
    if primary_context is not None:
        return [
            _build_evidence_excerpt(
                answer_text=answer_text,
                question_text=question_text,
                context=primary_context,
            )
        ]
    return []


def _rank_evidence_contexts(
    *,
    answer_text: str,
    question_text: str,
    contexts: list[RetrievedChunkContext],
    primary_context: RetrievedChunkContext | None = None,
) -> list[tuple[float, RetrievedChunkContext]]:
    answer_terms = _tokenize(answer_text)
    question_terms = _tokenize(question_text)
    anchor_terms = extract_anchor_terms(question_text)
    constraint_terms = extract_constraint_terms(question_text)
    ranked: list[tuple[float, RetrievedChunkContext]] = []
    for context in contexts:
        if context.score <= 0:
            continue
        evidence_context = _build_evidence_excerpt(
            answer_text=answer_text,
            question_text=question_text,
            context=context,
        )
        context_terms = _tokenize(evidence_context.text)
        overlap = len(answer_terms & context_terms)
        phrase_match = contains_normalized_phrase(answer_text, evidence_context.text)
        minimum_overlap = 1 if len(answer_terms) <= 3 else 2
        is_primary = primary_context is not None and context.chunk_id == primary_context.chunk_id
        if overlap == 0 and not is_primary:
            continue
        if not is_primary and overlap < minimum_overlap and not phrase_match:
            continue

        anchor_overlap = len(anchor_terms & context_terms)
        constraint_overlap = len(constraint_terms & context_terms)
        question_overlap = len(question_terms & context_terms)
        page_span = max(1, context.page_end - context.page_start + 1)
        score = overlap + context.score
        if phrase_match:
            score += 0.5
        if question_text:
            score += _score_sentence_against_text(
                prompt_text=question_text,
                prompt_terms=question_terms,
                anchor_terms=anchor_terms,
                constraint_terms=constraint_terms,
                candidate=SentenceCandidate(
                    sentence=evidence_context.text,
                    context=context,
                    score=context.score,
                    terms=context_terms,
                ),
            ) * 0.55
        score += anchor_overlap * 0.22
        score += constraint_overlap * 0.26
        score += min(0.18, question_overlap * 0.04)
        score += _answer_type_bonus(question_text, evidence_context.text) * 0.5
        score -= metadata_noise_score(evidence_context.text) * 0.2
        score -= _answer_type_penalty(question_text, evidence_context.text) * 0.6
        score -= (page_span - 1) * 0.05
        if is_primary:
            score += 0.2
        ranked.append((score, evidence_context))

    ranked.sort(
        key=lambda item: (
            -item[0],
            -item[1].score,
            item[1].page_start,
            item[1].chunk_index,
        )
    )
    return ranked


def _build_evidence_excerpt(
    *,
    answer_text: str,
    question_text: str,
    context: RetrievedChunkContext,
) -> RetrievedChunkContext:
    sentences = _split_sentences(context.text)
    if not sentences:
        return context

    answer_terms = _tokenize(answer_text)
    question_terms = _tokenize(question_text)
    anchor_terms = extract_anchor_terms(question_text)
    constraint_terms = extract_constraint_terms(question_text)
    candidate_spans = _build_candidate_spans(sentences)
    best_span = max(
        candidate_spans,
        key=lambda span: _evidence_span_sort_key(
            answer_terms=answer_terms,
            answer_text=answer_text,
            question_text=question_text,
            question_terms=question_terms,
            anchor_terms=anchor_terms,
            constraint_terms=constraint_terms,
            span=span,
        ),
    )
    if len(answer_terms & _tokenize(best_span)) == 0:
        return context
    trimmed_span = _trim_span_to_support(
        answer_text=answer_text,
        question_text=question_text,
        span=best_span,
    )
    return replace(context, text=trimmed_span)


def _build_candidate_spans(sentences: list[str]) -> list[str]:
    spans = list(sentences)
    for index in range(len(sentences) - 1):
        spans.append(f"{sentences[index]} {sentences[index + 1]}")
    for index in range(len(sentences) - 2):
        spans.append(f"{sentences[index]} {sentences[index + 1]} {sentences[index + 2]}")
    return spans


def _evidence_span_sort_key(
    *,
    answer_terms: set[str],
    answer_text: str,
    question_text: str,
    question_terms: set[str],
    anchor_terms: set[str],
    constraint_terms: set[str],
    span: str,
) -> tuple[float, float, float, float, float, float, float]:
    span_terms = _tokenize(span)
    overlap = len(answer_terms & span_terms)
    normalized_overlap = _normalized_term_overlap(answer_terms, span_terms)
    phrase_match = 1.0 if contains_normalized_phrase(answer_text, span) else 0.0
    question_overlap = _normalized_term_overlap(question_terms, span_terms)
    anchor_overlap = _normalized_term_overlap(anchor_terms, span_terms)
    constraint_overlap = _normalized_term_overlap(constraint_terms, span_terms)
    answer_type_bonus = _answer_type_bonus(question_text, span)
    # Prefer narrower spans when support is otherwise similar.
    brevity_bonus = 1 / max(len(span_terms), 1)
    return (
        float(overlap),
        normalized_overlap,
        constraint_overlap,
        anchor_overlap,
        question_overlap + answer_type_bonus,
        phrase_match,
        brevity_bonus,
    )


def _trim_span_to_support(*, answer_text: str, question_text: str, span: str) -> str:
    answer_terms = _tokenize(answer_text)
    question_terms = _tokenize(question_text)
    anchor_terms = extract_anchor_terms(question_text)
    constraint_terms = extract_constraint_terms(question_text)
    clauses = [clause.strip(" ,;:") for clause in CLAUSE_SPLIT_RE.split(span) if clause.strip(" ,;:")]
    if len(clauses) <= 1:
        return span

    clause_scores = [
        (
            len(answer_terms & _tokenize(clause)),
            contains_normalized_phrase(answer_text, clause),
            _normalized_term_overlap(question_terms, _tokenize(clause)),
            _normalized_term_overlap(anchor_terms, _tokenize(clause)),
            _normalized_term_overlap(constraint_terms, _tokenize(clause)),
            _answer_type_bonus(question_text, clause),
            -len(_tokenize(clause)),
            clause,
        )
        for clause in clauses
    ]
    (
        best_score,
        phrase_match,
        _question_overlap,
        _anchor_overlap,
        _constraint_overlap,
        _answer_type,
        _,
        best_clause,
    ) = max(clause_scores, key=lambda item: item[:7])
    if best_score == 0 and not phrase_match:
        return span
    best_index = clauses.index(best_clause)
    if best_index > 0:
        previous_clause = clauses[best_index - 1]
        previous_overlap = len(answer_terms & _tokenize(previous_clause))
        if previous_overlap > 0 and len(_tokenize(previous_clause)) <= 3:
            return f"{previous_clause}, {best_clause}"
    return best_clause


def _rank_contexts_for_selection(
    *,
    question: str,
    question_terms: set[str],
    contexts: list[RetrievedChunkContext],
) -> list[RetrievedChunkContext]:
    anchor_terms = extract_anchor_terms(question)
    constraint_terms = extract_constraint_terms(question)
    nickname_subject_terms = extract_named_subject_terms(question) or anchor_terms
    prefer_explicit_date = question.lower().startswith("when ") and any(
        has_explicit_date(context.text) for context in contexts
    )
    prefer_alias = "nickname" in question.lower() and any(
        has_nickname_alias(context.text, nickname_subject_terms) for context in contexts
    )

    def sort_key(
        context: RetrievedChunkContext,
    ) -> tuple[float, float, float, float, float, float, float, float, float, float, float, float, float, float]:
        context_terms = _tokenize(context.text)
        term_overlap = len(question_terms & context_terms)
        anchor_overlap = len(anchor_terms & context_terms)
        constraint_overlap = len(constraint_terms & context_terms)
        explicit_date_priority = 1.0 if prefer_explicit_date and has_explicit_date(context.text) else 0.0
        alias_priority = (
            1.0 if prefer_alias and has_nickname_alias(context.text, nickname_subject_terms) else 0.0
        )
        answer_type_fit = _answer_type_bonus(question, context.text) - _answer_type_penalty(
            question, context.text
        )
        normalized_overlap = _normalized_term_overlap(question_terms, context_terms)
        term_precision = (
            len(question_terms & context_terms) / max(len(context_terms), 1)
            if context_terms
            else 0.0
        )
        phrase_match = 1.0 if contains_normalized_phrase(question, context.text) else 0.0
        query_run_match = query_run_bonus(question, context.text, max_bonus=0.2)
        metadata_penalty = metadata_noise_score(context.text)
        token_cost = max(context.token_estimate, _estimate_tokens(context.text), 1)
        return (
            alias_priority,
            explicit_date_priority,
            answer_type_fit,
            float(constraint_overlap),
            float(anchor_overlap),
            float(term_overlap),
            normalized_overlap,
            term_precision,
            phrase_match,
            query_run_match,
            context.lexical_score,
            context.rerank_score,
            -metadata_penalty,
            -float(token_cost),
        )

    return sorted(contexts, key=sort_key, reverse=True)


def _build_question_facets(question: str) -> list[QuestionFacet]:
    normalized_question = " ".join(question.split()).strip()
    base_terms = _tokenize(normalized_question)
    anchor_terms = extract_anchor_terms(normalized_question)
    if not normalized_question or not base_terms:
        return []
    if not _question_needs_multi_sentence_answer(normalized_question):
        return [
            QuestionFacet(
                text=normalized_question,
                terms=base_terms,
                anchor_terms=anchor_terms,
                constraint_terms=extract_constraint_terms(normalized_question),
            )
        ]

    raw_facets = [
        fragment.strip(" ,;:")
        for fragment in re.split(r"\s+(?:and|or|then)\s+", normalized_question, flags=re.IGNORECASE)
        if fragment.strip(" ,;:")
    ]
    if len(raw_facets) < 2:
        return [
            QuestionFacet(
                text=normalized_question,
                terms=base_terms,
                anchor_terms=anchor_terms,
                constraint_terms=extract_constraint_terms(normalized_question),
            )
        ]

    facets: list[QuestionFacet] = []
    seen_terms: set[frozenset[str]] = set()
    for raw_facet in raw_facets:
        facet_text = raw_facet
        facet_terms = _tokenize(facet_text)
        if not facet_terms:
            continue
        frozen_terms = frozenset(facet_terms)
        if frozen_terms in seen_terms:
            continue
        seen_terms.add(frozen_terms)
        facets.append(
            QuestionFacet(
                text=facet_text,
                terms=facet_terms,
                anchor_terms=extract_anchor_terms(facet_text),
                constraint_terms=extract_constraint_terms(facet_text),
            )
        )

    return facets or [
        QuestionFacet(
            text=normalized_question,
            terms=base_terms,
            anchor_terms=anchor_terms,
            constraint_terms=extract_constraint_terms(normalized_question),
        )
    ]


def _build_sentence_candidates(
    *,
    question: str,
    question_terms: set[str],
    contexts: list[RetrievedChunkContext],
) -> list[SentenceCandidate]:
    candidates: list[SentenceCandidate] = []
    for context in contexts:
        for sentence_span in _build_candidate_spans(_split_sentences(context.text)):
            if not sentence_span:
                continue
            sentence_terms = _tokenize(sentence_span)
            if not sentence_terms:
                continue
            overlap = len(question_terms & sentence_terms)
            if overlap == 0:
                continue
            anchor_terms = extract_anchor_terms(question)
            constraint_terms = extract_constraint_terms(question)
            anchor_overlap = len(anchor_terms & sentence_terms)
            constraint_overlap = len(constraint_terms & sentence_terms)
            metadata_penalty = metadata_noise_score(sentence_span)

            lexical_score = overlap / max(len(question_terms), 1)
            precision = overlap / max(len(sentence_terms), 1)
            phrase_bonus = 0.2 if contains_normalized_phrase(question, sentence_span) else 0.0
            combined_score = (
                lexical_score
                + _anchor_support_score(
                    anchor_terms=anchor_terms,
                    matched_anchor_terms=anchor_terms & sentence_terms,
                    candidate_terms=sentence_terms,
                )
                + _constraint_support_score(
                    constraint_terms=constraint_terms,
                    matched_constraint_terms=constraint_terms & sentence_terms,
                    candidate_terms=sentence_terms,
                )
                + precision * 0.35
                + phrase_bonus
                + query_run_bonus(question, sentence_span, max_bonus=0.24)
                + _ordered_term_bonus(question, sentence_span)
                + _answer_type_bonus(question, sentence_span)
                + max(context.score, 0.0) * 0.15
                + (0.1 if anchor_overlap else 0.0)
                + (0.08 if constraint_overlap else 0.0)
                - metadata_penalty * 0.25
                - _answer_type_penalty(question, sentence_span)
            )
            candidates.append(
                SentenceCandidate(
                    sentence=sentence_span,
                    context=context,
                    score=combined_score,
                    terms=sentence_terms,
                )
            )

    candidates.sort(
        key=lambda candidate: (
            -candidate.score,
            -len(question_terms & candidate.terms),
            candidate.context.page_start,
            candidate.context.chunk_index,
        )
    )
    return candidates


def _question_needs_multi_sentence_answer(question: str) -> bool:
    lowered = question.lower()
    return " and " in lowered or "both " in lowered


def _select_answer_sentences(
    *,
    question_facets: list[QuestionFacet],
    sentence_candidates: list[SentenceCandidate],
) -> list[SentenceCandidate]:
    selected_candidates: list[SentenceCandidate] = []
    for facet in question_facets:
        candidate = _select_best_candidate_for_facet(
            facet=facet,
            candidates=sentence_candidates,
            selected_candidates=selected_candidates,
        )
        if candidate is None:
            if len(question_facets) > 1:
                return []
            break
        if not any(
            selected.sentence == candidate.sentence
            and selected.context.chunk_id == candidate.context.chunk_id
            for selected in selected_candidates
        ):
            selected_candidates.append(candidate)

    if selected_candidates:
        return selected_candidates

    if sentence_candidates and sentence_candidates[0].score >= 0.2:
        return [sentence_candidates[0]]

    return []


def _select_best_candidate_for_facet(
    *,
    facet: QuestionFacet,
    candidates: list[SentenceCandidate],
    selected_candidates: list[SentenceCandidate],
) -> SentenceCandidate | None:
    covered_terms = set().union(
        *(candidate.terms & facet.terms for candidate in selected_candidates)
    )
    nickname_subject_terms = extract_named_subject_terms(facet.text) or facet.anchor_terms
    prefer_explicit_date = facet.text.lower().startswith("when ") and any(
        has_explicit_date(candidate.sentence) for candidate in candidates
    )
    prefer_alias = "nickname" in facet.text.lower() and any(
        has_nickname_alias(candidate.sentence, nickname_subject_terms) for candidate in candidates
    )
    scored_candidates: list[tuple[tuple[float, float, float, float], SentenceCandidate]] = []
    for candidate in candidates:
        if prefer_explicit_date and not has_explicit_date(candidate.sentence):
            continue
        if prefer_alias and not has_nickname_alias(candidate.sentence, nickname_subject_terms):
            continue
        facet_score = _score_sentence_against_text(
            prompt_text=facet.text,
            prompt_terms=facet.terms,
            anchor_terms=facet.anchor_terms,
            constraint_terms=facet.constraint_terms,
            candidate=candidate,
        )
        if facet_score < 0.2:
            continue
        if not _candidate_satisfies_anchor_requirement(
            facet=facet,
            candidate=candidate,
        ):
            continue
        anchor_matches = len(candidate.terms & facet.anchor_terms)
        constraint_matches = len(candidate.terms & facet.constraint_terms)
        if (
            metadata_noise_score(candidate.sentence) >= 0.35
            and anchor_matches < _required_anchor_matches(facet.anchor_terms)
            and constraint_matches < _required_constraint_matches(facet.constraint_terms)
        ):
            continue
        added_terms = len((candidate.terms & facet.terms) - covered_terms)
        duplicate_penalty = 0.0 if added_terms else -0.2
        scored_candidates.append(
            (
                (
                    float(constraint_matches),
                    float(anchor_matches),
                    facet_score,
                    float(added_terms),
                    candidate.score + duplicate_penalty,
                    candidate.context.score,
                ),
                candidate,
            )
        )

    if not scored_candidates:
        return None

    scored_candidates.sort(
        key=lambda item: (
            -item[0][0],
            -item[0][1],
            -item[0][2],
            -item[0][3],
            -item[0][4],
            -item[0][5],
            item[1].context.page_start,
            item[1].context.chunk_index,
        )
    )
    return scored_candidates[0][1]


def _score_sentence_against_text(
    *,
    prompt_text: str,
    prompt_terms: set[str],
    anchor_terms: set[str],
    constraint_terms: set[str],
    candidate: SentenceCandidate,
) -> float:
    overlap = len(prompt_terms & candidate.terms)
    if overlap == 0:
        return 0.0
    lexical_score = overlap / max(len(prompt_terms), 1)
    precision = overlap / max(len(candidate.terms), 1)
    phrase_bonus = 0.2 if contains_normalized_phrase(prompt_text, candidate.sentence) else 0.0
    return (
        lexical_score
        + _anchor_support_score(
            anchor_terms=anchor_terms,
            matched_anchor_terms=anchor_terms & candidate.terms,
            candidate_terms=candidate.terms,
        )
        + _constraint_support_score(
            constraint_terms=constraint_terms,
            matched_constraint_terms=constraint_terms & candidate.terms,
            candidate_terms=candidate.terms,
        )
        + precision * 0.35
        + phrase_bonus
        + query_run_bonus(prompt_text, candidate.sentence, max_bonus=0.24)
        + _ordered_term_bonus(prompt_text, candidate.sentence)
        + _answer_type_bonus(prompt_text, candidate.sentence)
        + max(candidate.context.score, 0.0) * 0.15
        - metadata_noise_score(candidate.sentence) * 0.25
        - _answer_type_penalty(prompt_text, candidate.sentence)
    )


def _split_support_units(answer_text: str) -> list[str]:
    support_units: list[str] = []
    seen_units: set[str] = set()
    for sentence in _split_sentences(answer_text):
        normalized_sentence = sentence.strip()
        if not normalized_sentence:
            continue
        sentence_key = normalized_sentence.casefold()
        if sentence_key not in seen_units:
            support_units.append(normalized_sentence)
            seen_units.add(sentence_key)

    return support_units


def _is_same_location_as_existing(
    candidate: RetrievedChunkContext,
    selected: list[RetrievedChunkContext],
) -> bool:
    return any(
        context.document_id == candidate.document_id
        and context.page_start == candidate.page_start
        and context.page_end == candidate.page_end
        for context in selected
    )


def _unsupported_answer() -> GeneratedAnswer:
    return GeneratedAnswer(
        answer_text=(
            "I could not find enough support in the indexed document content "
            "to answer that question."
        ),
        supported=False,
        citations=[],
    )


def _anchor_support_score(
    *,
    anchor_terms: set[str],
    matched_anchor_terms: set[str],
    candidate_terms: set[str],
) -> float:
    if not anchor_terms:
        return 0.0
    coverage = len(matched_anchor_terms) / len(anchor_terms)
    precision = len(matched_anchor_terms) / max(len(candidate_terms), 1)
    return coverage * 0.5 + precision * 0.1


def _constraint_support_score(
    *,
    constraint_terms: set[str],
    matched_constraint_terms: set[str],
    candidate_terms: set[str],
) -> float:
    if not constraint_terms:
        return 0.0
    if not matched_constraint_terms:
        return 0.0
    coverage = len(matched_constraint_terms) / len(constraint_terms)
    precision = len(matched_constraint_terms) / max(len(candidate_terms), 1)
    return coverage * 0.45 + precision * 0.08


def _required_anchor_matches(anchor_terms: set[str]) -> int:
    if not anchor_terms:
        return 0
    if len(anchor_terms) == 1:
        return 1
    if len(anchor_terms) == 2:
        return 2
    return max(1, ceil(len(anchor_terms) * 0.5))


def _required_constraint_matches(constraint_terms: set[str]) -> int:
    if not constraint_terms:
        return 0
    if len(constraint_terms) <= 2:
        return 1
    if len(constraint_terms) == 3:
        return 2
    return max(3, ceil(len(constraint_terms) * 0.6))


def _candidate_satisfies_anchor_requirement(
    *,
    facet: QuestionFacet,
    candidate: SentenceCandidate,
) -> bool:
    if not facet.anchor_terms and not facet.constraint_terms:
        return True
    matched_constraint_terms = facet.constraint_terms & candidate.terms
    matched_anchor_terms = facet.anchor_terms & candidate.terms
    if len(matched_anchor_terms) >= _required_anchor_matches(facet.anchor_terms):
        return True
    return len(matched_constraint_terms) >= _required_constraint_matches(
        facet.constraint_terms
    )


def _answer_has_sufficient_support(
    *,
    question_facets: list[QuestionFacet],
    selected_candidates: list[SentenceCandidate],
) -> bool:
    if not selected_candidates:
        return False

    for facet in question_facets:
        matched_candidates = [
            candidate
            for candidate in selected_candidates
            if candidate.terms & facet.terms
        ]
        if not matched_candidates:
            return False
        if "nickname" in facet.text.lower():
            nickname_subject_terms = extract_named_subject_terms(facet.text) or facet.anchor_terms
            if any(
                has_nickname_alias(candidate.sentence, nickname_subject_terms)
                for candidate in matched_candidates
            ):
                continue
        if not facet.anchor_terms:
            if _facet_overlap_is_too_weak(facet=facet, matched_candidates=matched_candidates):
                return False
            continue
        matched_anchor_terms = set().union(
            *(candidate.terms & facet.anchor_terms for candidate in matched_candidates)
        )
        matched_constraint_terms = set().union(
            *(candidate.terms & facet.constraint_terms for candidate in matched_candidates)
        )
        if (
            len(matched_anchor_terms) < _required_anchor_matches(facet.anchor_terms)
            and len(matched_constraint_terms)
            < _required_constraint_matches(facet.constraint_terms)
        ):
            return False
        if _facet_is_topic_probe(facet) and len(matched_constraint_terms) < _required_constraint_matches(
            facet.constraint_terms
        ):
            return False
        if _facet_is_topic_probe(facet):
            long_constraint_terms = _long_constraint_terms(facet.constraint_terms)
            if long_constraint_terms and not (matched_constraint_terms & long_constraint_terms):
                return False
            distinctive_terms = _distinctive_support_terms(facet)
            matched_distinctive_terms = set().union(
                *(candidate.terms & distinctive_terms for candidate in matched_candidates)
            )
            if distinctive_terms and len(matched_distinctive_terms) < min(2, len(distinctive_terms)):
                return False
            if (
                len(distinctive_terms) >= 2
                and max(
                    longest_matching_query_run(facet.text, candidate.sentence)
                    for candidate in matched_candidates
                )
                < 2
            ):
                return False

    return True


def _facet_overlap_is_too_weak(
    *,
    facet: QuestionFacet,
    matched_candidates: list[SentenceCandidate],
) -> bool:
    covered_terms = set().union(*(candidate.terms & facet.terms for candidate in matched_candidates))
    normalized_overlap = len(covered_terms) / max(len(facet.terms), 1)
    if normalized_overlap >= 0.5:
        return False
    if facet.constraint_terms and covered_terms & facet.constraint_terms:
        return normalized_overlap < 0.34
    return True


def _ordered_term_bonus(query: str, text: str) -> float:
    query_terms = normalized_token_sequence(query, drop_stopwords=True)
    text_terms = normalized_token_sequence(text, drop_stopwords=True)
    if len(query_terms) < 2 or len(text_terms) < 2:
        return 0.0

    query_pairs = {
        (left, right)
        for left, right in zip(query_terms, query_terms[1:], strict=False)
        if left != right
    }
    if not query_pairs:
        return 0.0
    text_pairs = {
        (left, right)
        for left, right in zip(text_terms, text_terms[1:], strict=False)
    }
    overlap = len(query_pairs & text_pairs)
    if overlap == 0:
        return 0.0
    return min(0.2, overlap / len(query_pairs) * 0.2)


def _facet_is_topic_probe(facet: QuestionFacet) -> bool:
    lowered = facet.text.lower()
    return " say about " in lowered or lowered.startswith("about ")


def _long_constraint_terms(constraint_terms: set[str]) -> set[str]:
    return {term for term in constraint_terms if len(term) >= 6}


def _distinctive_support_terms(facet: QuestionFacet) -> set[str]:
    return {
        term
        for term in (facet.anchor_terms | facet.constraint_terms)
        if len(term) >= 5 or any(character.isdigit() for character in term)
    }


def _answer_type_bonus(question: str, text: str) -> float:
    lowered_question = question.lower()
    nickname_subject_terms = extract_named_subject_terms(question) or extract_anchor_terms(question)
    bonus = 0.0

    if lowered_question.startswith("when "):
        if has_explicit_date(text):
            bonus += 0.34
        elif has_temporal_marker(text):
            bonus += 0.06
        if re.search(r"\b\d{4}\b", text):
            bonus += 0.06
        if re.search(
            r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
            r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|"
            r"dec(?:ember)?)\b",
            text,
            re.IGNORECASE,
        ):
            bonus += 0.08
        elif re.search(r"\b\d+\b", text):
            bonus += 0.04

    if lowered_question.startswith("how many"):
        if re.search(r"\b\d+\b", text.lower()):
            bonus += 0.22

    if lowered_question.startswith("where "):
        if re.search(r"\b(?:in|at|from|to|toward|towards|near|into|onto)\s+[A-Z]", text):
            bonus += 0.12
        if re.search(r"\b[A-Z][a-z]+,\s+[A-Z][a-z]+\b", text):
            bonus += 0.08

    if "nickname" in lowered_question and has_nickname_alias(text, nickname_subject_terms):
        bonus += 0.38

    return bonus


def _answer_type_penalty(question: str, text: str) -> float:
    lowered_question = question.lower()
    nickname_subject_terms = extract_named_subject_terms(question) or extract_anchor_terms(question)
    penalty = 0.0
    if lowered_question.startswith("when ") and not has_explicit_date(text):
        penalty += 0.18 if has_temporal_marker(text) else 0.24
    if "nickname" in lowered_question:
        if not has_nickname_alias(text, nickname_subject_terms):
            penalty += 0.24
        named_terms = extract_named_subject_terms(question)
        if named_terms and not (named_terms & _tokenize(text)):
            penalty += 0.28
    return penalty
