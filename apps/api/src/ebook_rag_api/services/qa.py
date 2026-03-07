import re
from dataclasses import dataclass
from functools import lru_cache
from time import perf_counter
from typing import Protocol

import httpx
from sqlalchemy.orm import Session

from ebook_rag_api.core.config import get_settings
from ebook_rag_api.services.retrieval import ChunkSearchMatch, normalize_query, search_chunks
from ebook_rag_api.services.text import STOPWORDS, TOKEN_RE, tokenize_terms

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


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

        question_terms = _tokenize(question)
        best_sentence = ""
        best_context: RetrievedChunkContext | None = None
        best_score = 0.0

        for context in contexts:
            for raw_sentence in SENTENCE_SPLIT_RE.split(context.text):
                sentence = raw_sentence.strip()
                if not sentence:
                    continue
                sentence_terms = _tokenize(sentence)
                if not sentence_terms:
                    continue
                overlap = len(question_terms & sentence_terms)
                if overlap == 0:
                    continue

                lexical_score = overlap / max(len(question_terms), 1)
                added_term_bonus = len(sentence_terms - STOPWORDS) / max(len(sentence_terms), 1)
                combined_score = (
                    lexical_score
                    + added_term_bonus * 0.1
                    + max(context.score, 0.0) * 0.15
                )
                if combined_score > best_score:
                    best_score = combined_score
                    best_sentence = sentence
                    best_context = context

        if best_context is None or best_score < 0.2:
            return _unsupported_answer()

        answer_text = best_sentence if best_sentence.endswith((".", "!", "?")) else f"{best_sentence}."
        citations = select_evidence_citations(
            answer_text=answer_text,
            contexts=contexts,
            primary_context=best_context,
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
    selected_contexts = assemble_answer_contexts(
        question=normalized_question,
        contexts=retrieved_chunks,
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
    answer_provider = get_answer_provider()
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
    *,
    question: str,
    contexts: list[RetrievedChunkContext],
) -> list[RetrievedChunkContext]:
    if not contexts:
        return []

    question_terms = _tokenize(question)
    ranked_contexts = _rank_contexts_for_selection(question_terms=question_terms, contexts=contexts)
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

    best_candidate = max(adjacent_contexts, key=lambda context: context.score)
    candidate_cost = max(
        best_candidate.token_estimate,
        _estimate_tokens(best_candidate.text),
        1,
    )
    if candidate_cost > remaining_budget:
        return None
    return best_candidate


def _estimate_tokens(text: str) -> int:
    return max(1, len(TOKEN_RE.findall(text)))


def _jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _tokenize(text: str) -> set[str]:
    return tokenize_terms(text, drop_stopwords=True)


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
) -> list[RetrievedChunkContext]:
    answer_terms = _tokenize(answer_text)
    ranked: list[tuple[float, RetrievedChunkContext]] = []
    for context in contexts:
        if context.score <= 0:
            continue
        context_terms = _tokenize(context.text)
        overlap = len(answer_terms & context_terms)
        if overlap == 0 and primary_context is not None and context.chunk_id != primary_context.chunk_id:
            continue
        score = overlap + context.score
        if primary_context is not None and context.chunk_id == primary_context.chunk_id:
            score += 1.0
        ranked.append((score, context))

    ranked.sort(
        key=lambda item: (
            -item[0],
            -item[1].score,
            item[1].page_start,
            item[1].chunk_index,
        )
    )
    citations: list[RetrievedChunkContext] = []
    seen_chunk_ids: set[str] = set()
    for _, context in ranked:
        if context.chunk_id in seen_chunk_ids:
            continue
        citations.append(context)
        seen_chunk_ids.add(context.chunk_id)
        if len(citations) == 3:
            break

    if citations:
        return citations
    if primary_context is not None:
        return [primary_context]
    return []


def _rank_contexts_for_selection(
    *,
    question_terms: set[str],
    contexts: list[RetrievedChunkContext],
) -> list[RetrievedChunkContext]:
    def sort_key(context: RetrievedChunkContext) -> tuple[float, float, float, int, int]:
        term_overlap = len(question_terms & _tokenize(context.text))
        return (
            float(term_overlap),
            context.score,
            context.rerank_score,
            -context.page_start,
            -context.chunk_index,
        )

    return sorted(contexts, key=sort_key, reverse=True)


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
