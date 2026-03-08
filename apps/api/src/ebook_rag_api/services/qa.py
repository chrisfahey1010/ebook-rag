import re
import json
from collections.abc import Iterator
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from functools import lru_cache
from math import ceil
from time import perf_counter
from typing import Protocol

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from ebook_rag_api.core.config import get_settings
from ebook_rag_api.models.document_chunk import DocumentChunk
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
    normalize_query_text,
    normalized_token_sequence,
    query_run_bonus,
    tokenize_terms,
)

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
CLAUSE_SPLIT_RE = re.compile(r"(?<=[,;:])\s+|\s+(?:but|while|then)\s+", re.IGNORECASE)


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
    provenance: dict[str, object] | None = None
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
    answer_mode: str = "unsupported"
    confidence: float = 0.0
    support_score: float = 0.0
    verification: "AnswerVerification | None" = None
    postprocess: "AnswerPostprocessTrace | None" = None


@dataclass(frozen=True)
class QuestionRoutingDecision:
    answer_mode: str
    reason: str
    facet_count: int
    context_count: int
    should_use_generative: bool
    heuristic_support_score: float = 0.0
    unsupported_classifier_ran: bool = False
    unsupported_classifier_supported: bool | None = None
    unsupported_classifier_reason: str | None = None


@dataclass(frozen=True)
class RuntimeMetadata:
    embedding_provider: str
    embedding_model: str | None
    reranker_provider: str
    reranker_model: str | None
    answer_provider: str
    answer_model: str | None


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
    answer_mode: str
    question_router: QuestionRoutingDecision
    runtime: RuntimeMetadata
    retrieved_chunks: list[RetrievedChunkContext]
    selected_contexts: list[RetrievedChunkContext]
    cited_contexts: list[RetrievedChunkContext]
    prompt_snapshot: str
    timings: QATimingBreakdown
    verification: "AnswerVerification | None"
    postprocess: "AnswerPostprocessTrace | None"
    answer: GeneratedAnswer


@dataclass(frozen=True)
class ClaimVerification:
    claim_text: str
    supported: bool
    support_score: float
    verifier: str
    rationale: str
    citations: list[RetrievedChunkContext] = field(default_factory=list)


@dataclass(frozen=True)
class AnswerVerification:
    verified: bool
    verifier: str
    claim_count: int
    supported_claim_count: int
    average_claim_score: float
    minimum_claim_score: float
    claims: list[ClaimVerification] = field(default_factory=list)


@dataclass(frozen=True)
class AnswerPostprocessTrace:
    question_coverage_score: float
    support_threshold: float
    repair_attempted: bool
    repair_applied: bool
    repair_reason: str | None = None
    claim_count: int = 0
    supported_claim_count: int = 0


@dataclass(frozen=True)
class AnswerStreamChunk:
    delta: str


@dataclass(frozen=True)
class PreparedQARequest:
    normalized_question: str
    retrieved_chunks: list[RetrievedChunkContext]
    selected_contexts: list[RetrievedChunkContext]
    prompt_snapshot: str
    answer_provider: "AnswerProvider"
    routing_decision: QuestionRoutingDecision
    runtime: RuntimeMetadata
    normalization_ms: float
    retrieval_ms: float
    context_assembly_ms: float


class AnswerProvider(Protocol):
    def generate_answer(
        self, question: str, contexts: list[RetrievedChunkContext]
    ) -> GeneratedAnswer:
        ...

    def stream_answer(
        self, question: str, contexts: list[RetrievedChunkContext]
    ) -> Iterator[AnswerStreamChunk]:
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
        selected_candidates = _expand_selected_candidates_for_explanation(
            question=question,
            selected_candidates=selected_candidates,
        )
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

    def stream_answer(
        self, question: str, contexts: list[RetrievedChunkContext]
    ) -> Iterator[AnswerStreamChunk]:
        answer = self.generate_answer(question=question, contexts=contexts)
        for delta in _chunk_text_for_stream(answer.answer_text):
            yield AnswerStreamChunk(delta=delta)


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

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> str | None:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
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
            return None

        answer_text = extract_chat_completion_text(response.json()).strip()
        if not answer_text or answer_text == "INSUFFICIENT_SUPPORT":
            return None
        return answer_text

    def stream_completion(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> Iterator[AnswerStreamChunk]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                with client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                ) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if not line:
                            continue
                        data = _extract_sse_data_line(line)
                        if not data or data == "[DONE]":
                            continue
                        delta = extract_chat_completion_delta(json.loads(data))
                        if delta:
                            yield AnswerStreamChunk(delta=delta)
        except httpx.HTTPError:
            return

    def generate_answer(
        self, question: str, contexts: list[RetrievedChunkContext]
    ) -> GeneratedAnswer:
        if not contexts:
            return _unsupported_answer()

        prompt = build_qa_prompt(question=question, contexts=contexts)
        answer_text = self.complete(
            system_prompt=(
                "You answer questions using only the provided document context. "
                "If the evidence is insufficient, reply exactly with "
                "'INSUFFICIENT_SUPPORT'."
            ),
            user_prompt=prompt,
        )
        if not answer_text:
            return _unsupported_answer()

        return GeneratedAnswer(
            answer_text=answer_text,
            supported=True,
            citations=select_evidence_citations(
                answer_text=answer_text,
                contexts=contexts,
                question_text=question,
            ),
            answer_mode="synthesis",
            confidence=0.66,
            support_score=0.66,
        )

    def stream_answer(
        self, question: str, contexts: list[RetrievedChunkContext]
    ) -> Iterator[AnswerStreamChunk]:
        if not contexts:
            for delta in _chunk_text_for_stream(_unsupported_answer().answer_text):
                yield AnswerStreamChunk(delta=delta)
            return

        prompt = build_qa_prompt(question=question, contexts=contexts)
        yielded = False
        for chunk in self.stream_completion(
            system_prompt=(
                "You answer questions using only the provided document context. "
                "If the evidence is insufficient, reply exactly with "
                "'INSUFFICIENT_SUPPORT'."
            ),
            user_prompt=prompt,
        ):
            yielded = True
            yield chunk
        if not yielded:
            for delta in _chunk_text_for_stream(_unsupported_answer().answer_text):
                yield AnswerStreamChunk(delta=delta)


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


def get_runtime_metadata() -> RuntimeMetadata:
    settings = get_settings()
    return RuntimeMetadata(
        embedding_provider=settings.embedding_provider,
        embedding_model=(
            settings.embedding_model
            if settings.embedding_provider in {"sentence_transformer", "openai_compatible"}
            else None
        ),
        reranker_provider=settings.reranker_provider,
        reranker_model=(
            settings.reranker_model
            if settings.reranker_provider in {"cross_encoder", "openai_compatible"}
            else None
        ),
        answer_provider=settings.answer_provider,
        answer_model=(
            settings.answer_model or settings.llm_model
            if settings.answer_provider == "openai_compatible"
            else None
        ),
    )


def route_question(
    *,
    question: str,
    contexts: list[RetrievedChunkContext],
    answer_provider: AnswerProvider,
) -> QuestionRoutingDecision:
    facets = _build_question_facets(question)
    facet_count = max(len(facets), 1)
    context_count = len(contexts)
    heuristic_support_score = _compute_question_support_score(
        question=question,
        facets=facets,
        contexts=contexts,
    )
    if not contexts:
        return QuestionRoutingDecision(
            answer_mode="unsupported",
            reason="No retrieval evidence was available for the question.",
            facet_count=facet_count,
            context_count=context_count,
            should_use_generative=False,
            heuristic_support_score=heuristic_support_score,
        )

    if _retrieval_support_is_too_weak(question=question, contexts=contexts, facets=facets):
        return QuestionRoutingDecision(
            answer_mode="unsupported",
            reason="Retrieved evidence did not cover enough anchor or constraint terms.",
            facet_count=facet_count,
            context_count=context_count,
            should_use_generative=False,
            heuristic_support_score=heuristic_support_score,
        )

    should_use_generative = isinstance(answer_provider, OpenAICompatibleAnswerProvider)
    if facet_count > 1:
        provisional_decision = QuestionRoutingDecision(
            answer_mode="synthesis" if should_use_generative else "extractive",
            reason=(
                "Question spans multiple facets; route to grounded synthesis when a local model is configured."
                if should_use_generative
                else "Question spans multiple facets, but no generative answer provider is configured."
            ),
            facet_count=facet_count,
            context_count=context_count,
            should_use_generative=should_use_generative,
            heuristic_support_score=heuristic_support_score,
        )
    else:
        provisional_decision = QuestionRoutingDecision(
            answer_mode="extractive",
            reason="Question looks like a single-fact lookup with adequate evidence coverage.",
            facet_count=facet_count,
            context_count=context_count,
            should_use_generative=False,
            heuristic_support_score=heuristic_support_score,
        )

    if should_use_generative and _should_run_unsupported_classifier(
        question=question,
        facets=facets,
        contexts=contexts,
    ):
        support_classification = _classify_question_support_with_provider(
            question=question,
            contexts=contexts,
            provider=answer_provider,
        )
        if support_classification is not None:
            is_supported, rationale = support_classification
            if not is_supported:
                reason = "Local unsupported classifier rejected the available evidence."
                if rationale:
                    reason = f"{reason} {rationale}"
                return QuestionRoutingDecision(
                    answer_mode="unsupported",
                    reason=reason,
                    facet_count=facet_count,
                    context_count=context_count,
                    should_use_generative=False,
                    heuristic_support_score=heuristic_support_score,
                    unsupported_classifier_ran=True,
                    unsupported_classifier_supported=False,
                    unsupported_classifier_reason=rationale or None,
                )
            return replace(
                provisional_decision,
                reason=(
                    f"{provisional_decision.reason} "
                    "Local unsupported classifier found enough evidence coverage."
                ),
                unsupported_classifier_ran=True,
                unsupported_classifier_supported=True,
                unsupported_classifier_reason=rationale or None,
            )

    return provisional_decision


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
    prepared_request = prepare_qa_request(
        session=session,
        question=question,
        top_k=top_k,
        document_id=document_id,
        include_prompt_snapshot=include_prompt_snapshot,
    )
    answer_started_at = perf_counter()
    answer = generate_answer_for_request(prepared_request)
    answered_at = perf_counter()
    trace = QATrace(
        answer_provider=type(prepared_request.answer_provider).__name__,
        answer_mode=answer.answer_mode,
        question_router=prepared_request.routing_decision,
        runtime=prepared_request.runtime,
        retrieved_chunks=prepared_request.retrieved_chunks,
        selected_contexts=prepared_request.selected_contexts,
        cited_contexts=answer.citations,
        prompt_snapshot=prepared_request.prompt_snapshot,
        timings=QATimingBreakdown(
            normalization_ms=prepared_request.normalization_ms,
            retrieval_ms=prepared_request.retrieval_ms,
            context_assembly_ms=prepared_request.context_assembly_ms,
            answer_generation_ms=(answered_at - answer_started_at) * 1000,
            total_ms=(
                prepared_request.normalization_ms
                + prepared_request.retrieval_ms
                + prepared_request.context_assembly_ms
                + (answered_at - answer_started_at) * 1000
            ),
        ),
        verification=answer.verification,
        postprocess=getattr(answer, "postprocess", None),
        answer=answer,
    )
    return prepared_request.normalized_question, trace


def prepare_qa_request(
    *,
    session: Session,
    question: str,
    top_k: int,
    document_id: str | None = None,
    include_prompt_snapshot: bool = True,
) -> PreparedQARequest:
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
    expanded_contexts = expand_contexts_with_page_siblings(
        session=session,
        question=normalized_question,
        contexts=retrieved_chunks,
    )
    assembled_contexts = assemble_answer_contexts(
        question=normalized_question,
        contexts=expanded_contexts,
    )
    answer_provider = get_answer_provider()
    routing_decision = route_question(
        question=normalized_question,
        contexts=assembled_contexts,
        answer_provider=answer_provider,
    )
    selected_contexts = (
        expanded_contexts
        if isinstance(answer_provider, ExtractiveAnswerProvider)
        else assembled_contexts
    )
    prompt_snapshot = (
        build_prompt_for_routing(
            question=normalized_question,
            contexts=selected_contexts,
            routing_decision=routing_decision,
        )
        if include_prompt_snapshot
        else ""
    )
    prepared_at = perf_counter()
    return PreparedQARequest(
        normalized_question=normalized_question,
        retrieved_chunks=retrieved_chunks,
        selected_contexts=selected_contexts,
        prompt_snapshot=prompt_snapshot,
        answer_provider=answer_provider,
        routing_decision=routing_decision,
        runtime=get_runtime_metadata(),
        normalization_ms=(normalized_at - started_at) * 1000,
        retrieval_ms=(retrieved_at - normalized_at) * 1000,
        context_assembly_ms=(prepared_at - retrieved_at) * 1000,
    )


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
        provenance=chunk.provenance,
        token_estimate=chunk.token_estimate,
        dense_score=match.dense_score,
        lexical_score=match.lexical_score,
        hybrid_score=match.hybrid_score,
        rerank_score=match.rerank_score,
        score=match.score,
    )


def generate_answer_for_request(prepared_request: PreparedQARequest) -> GeneratedAnswer:
    routing_decision = prepared_request.routing_decision
    if routing_decision.answer_mode == "unsupported":
        return _unsupported_answer(
            confidence=0.78,
            support_score=0.0,
        )

    if (
        routing_decision.answer_mode == "synthesis"
        and isinstance(prepared_request.answer_provider, OpenAICompatibleAnswerProvider)
    ):
        return _generate_synthesis_answer(
            question=prepared_request.normalized_question,
            contexts=prepared_request.selected_contexts,
            provider=prepared_request.answer_provider,
        )

    answer = prepared_request.answer_provider.generate_answer(
        question=prepared_request.normalized_question,
        contexts=prepared_request.selected_contexts,
    )
    return _finalize_generated_answer(
        answer=answer,
        question=prepared_request.normalized_question,
        contexts=prepared_request.selected_contexts,
        fallback_mode=routing_decision.answer_mode,
        verification_provider=(
            prepared_request.answer_provider
            if isinstance(prepared_request.answer_provider, OpenAICompatibleAnswerProvider)
            else None
        ),
    )


def stream_answer_for_request(
    prepared_request: PreparedQARequest,
) -> tuple[GeneratedAnswer, Iterator[AnswerStreamChunk]]:
    final_answer = generate_answer_for_request(prepared_request)

    def iterator() -> Iterator[AnswerStreamChunk]:
        for delta in _chunk_text_for_stream(final_answer.answer_text):
            yield AnswerStreamChunk(delta=delta)

    return final_answer, iterator()


def expand_contexts_with_page_siblings(
    *,
    session: Session,
    question: str,
    contexts: list[RetrievedChunkContext],
) -> list[RetrievedChunkContext]:
    if not contexts:
        return []

    expanded_by_id = {context.chunk_id: context for context in contexts}
    question_terms = _tokenize(question)
    question_metrics = _question_metric_phrases(question)

    for context in contexts:
        if context.page_start != context.page_end:
            if not _question_seeks_exact_metric_value(question):
                continue
        sibling_chunks = session.execute(
            select(DocumentChunk)
            .where(
                DocumentChunk.document_id == context.document_id,
                DocumentChunk.page_start <= context.page_end,
                DocumentChunk.page_end >= context.page_start,
            )
            .order_by(
                DocumentChunk.page_start.asc(),
                DocumentChunk.page_end.asc(),
                DocumentChunk.chunk_index.asc(),
            )
        ).scalars()
        for sibling_chunk in sibling_chunks:
            if sibling_chunk.id in expanded_by_id:
                continue
            sibling_context = RetrievedChunkContext(
                chunk_id=sibling_chunk.id,
                document_id=sibling_chunk.document_id,
                document_title=context.document_title,
                document_filename=context.document_filename,
                chunk_index=sibling_chunk.chunk_index,
                page_start=sibling_chunk.page_start,
                page_end=sibling_chunk.page_end,
                text=sibling_chunk.text,
                provenance=sibling_chunk.provenance,
                token_estimate=sibling_chunk.token_estimate,
                dense_score=context.dense_score,
                lexical_score=context.lexical_score,
                hybrid_score=context.hybrid_score,
                rerank_score=context.rerank_score,
                score=max(context.score - 0.04, 0.0),
            )
            if not _same_page_sibling_is_useful(
                question_text=question,
                question_terms=question_terms,
                question_metrics=question_metrics,
                candidate=sibling_context,
            ):
                continue
            expanded_by_id[sibling_context.chunk_id] = sibling_context

    return list(expanded_by_id.values())


def _same_page_sibling_is_useful(
    *,
    question_text: str,
    question_terms: set[str],
    question_metrics: set[str],
    candidate: RetrievedChunkContext,
) -> bool:
    lowered_candidate = candidate.text.lower()
    if question_metrics:
        return any(metric in lowered_candidate for metric in question_metrics)

    if (
        _question_seeks_exact_metric_value(question_text)
        and (_quarter_year_pairs(candidate.text) or _mentions_trailing_twelve_months(candidate.text))
    ):
        candidate_terms = _tokenize(candidate.text)
        if question_terms & candidate_terms:
            return True

    candidate_terms = _tokenize(candidate.text)
    if len(question_terms & candidate_terms) >= 3:
        return True
    if _question_has_numeric_intent(question_text) and len(question_terms & candidate_terms) >= 2:
        return bool(re.search(r"\d", candidate.text))
    return False


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


def _split_structured_lines(text: str) -> list[str]:
    lines: list[str] = []
    for raw_line in text.splitlines():
        normalized_line = " ".join(raw_line.split()).strip(" \t")
        normalized_line = normalized_line.lstrip("\u2022*- ").strip()
        if not normalized_line:
            continue
        if normalized_line in {"$", "%", "N/A"}:
            continue
        lines.append(normalized_line)
    return lines


def _deduplicate_text_units(units: list[str]) -> list[str]:
    deduplicated: list[str] = []
    seen: set[str] = set()
    for unit in units:
        normalized = " ".join(unit.split()).casefold()
        if not normalized or normalized in seen:
            continue
        deduplicated.append(unit)
        seen.add(normalized)
    return deduplicated


def _build_line_candidate_spans(lines: list[str]) -> list[str]:
    spans: list[str] = []
    max_window = min(8, len(lines))
    for start_index in range(len(lines)):
        for window_size in range(1, max_window + 1):
            end_index = start_index + window_size
            if end_index > len(lines):
                break
            spans.append(" ".join(lines[start_index:end_index]))
    return spans


_STRUCTURED_PERIOD_LABEL_RE = re.compile(
    r"^(?:q[1-4]\s+\d{4}|fy\s*\d{4}|ttm|\d{4})$",
    re.IGNORECASE,
)
_STRUCTURED_VALUE_LINE_RE = re.compile(
    r"^(?:\$?\s*\(?\d[\d,.\s]*\)?(?:\s*%|\s*[A-Za-z]+(?:\s+[A-Za-z]+)*)?|\(\d[\d,.\s]*\)\s*%?)$"
)
_STRUCTURED_VALUE_TOKEN_RE = re.compile(
    r"(?:\(?\d{1,3}\)?\s*%|\(?\$?\s*(?:\d{1,3}(?:,\d{3})+|\d+\.\d+|\d{3,})\)?)"
)
_FINANCIAL_METRIC_PHRASES = (
    "net sales",
    "operating income",
    "free cash flow",
    "operating cash flow",
    "capital expenditures",
    "employees",
    "aws segment sales",
)
_QUARTER_YEAR_RE = re.compile(r"\bq[1-4]\s+\d{4}\b", re.IGNORECASE)
_EVIDENCE_CURRENCY_RE = re.compile(r"\$\s+")
_EVIDENCE_PERCENT_RE = re.compile(r"\s+%")
_EVIDENCE_OPEN_PAREN_RE = re.compile(r"\(\s+")
_EVIDENCE_CLOSE_PAREN_RE = re.compile(r"\s+\)")


def _find_structured_period_headers(lines: list[str]) -> tuple[list[str], int]:
    for start_index in range(len(lines)):
        if not _STRUCTURED_PERIOD_LABEL_RE.match(lines[start_index]):
            continue
        end_index = start_index
        while end_index < len(lines) and _STRUCTURED_PERIOD_LABEL_RE.match(lines[end_index]):
            end_index += 1
        header_lines = lines[start_index:end_index]
        if len(header_lines) >= 2:
            return header_lines, end_index
    return [], 0


def _resolve_structured_period_headers(
    lines: list[str],
    period_headers: list[str] | None = None,
) -> list[str]:
    header_lines, _ = _find_structured_period_headers(lines)
    if header_lines:
        return header_lines
    return period_headers or []


def _build_structured_metric_value_spans(
    lines: list[str],
    *,
    period_headers: list[str] | None = None,
) -> list[str]:
    spans: list[str] = []
    header_lines, index = _find_structured_period_headers(lines)
    if not header_lines:
        header_lines = period_headers or []
        index = 0
    header_count = len(header_lines)

    if header_count < 2:
        return spans

    while index < len(lines):
        metric_line = lines[index]
        if _STRUCTURED_VALUE_LINE_RE.match(metric_line):
            index += 1
            continue

        row_values: list[str] = []
        cursor = index + 1
        while cursor < len(lines) and len(row_values) < header_count:
            candidate_value = lines[cursor]
            if not _STRUCTURED_VALUE_LINE_RE.match(candidate_value):
                break
            row_values.append(candidate_value)
            cursor += 1

        if len(row_values) >= header_count:
            spans.append(" ".join([metric_line, *row_values[:header_count]]))
            for period_label, value in zip(
                header_lines[:header_count], row_values[:header_count], strict=False
            ):
                spans.append(f"{metric_line} {period_label} {value}")
        index += 1

    return spans


def _extract_value_tokens(text: str, *, start_index: int = 0) -> list[str]:
    return [match.group(0).strip() for match in _STRUCTURED_VALUE_TOKEN_RE.finditer(text, start_index)]


def _build_flattened_metric_value_spans(
    lines: list[str],
    *,
    target_metrics: list[str],
    period_headers: list[str] | None = None,
) -> list[str]:
    if not target_metrics:
        return []

    spans: list[str] = []
    header_lines = _resolve_structured_period_headers(lines, period_headers)
    header_count = len(header_lines)
    if header_count < 2:
        return spans

    for index, line in enumerate(lines):
        lowered_line = line.lower()
        for metric in target_metrics:
            metric_start = lowered_line.find(metric)
            if metric_start < 0:
                continue
            metric_end = metric_start + len(metric)
            row_value_matches = list(_STRUCTURED_VALUE_TOKEN_RE.finditer(line, metric_end))
            row_values = [match.group(0).strip() for match in row_value_matches]
            metric_label = (
                line[metric_start : row_value_matches[0].start()].strip()
                if row_value_matches
                else line[metric_start:].strip()
            )
            if len(row_values) < header_count and index + 1 < len(lines):
                next_line_values = _extract_value_tokens(lines[index + 1])
                if next_line_values:
                    metric_label = line[metric_start:].strip()
                    row_values = next_line_values
            if len(row_values) < header_count:
                continue

            first_row_values = row_values[:header_count]
            spans.append(f"{metric_label} {' '.join(first_row_values)}")
            for period_label, value in zip(header_lines, first_row_values, strict=False):
                spans.append(f"{metric_label} {period_label} {value}")

    return spans


def _looks_like_structured_numeric_block(lines: list[str]) -> bool:
    if len(lines) < 6:
        return False
    numeric_lines = sum(1 for line in lines if re.search(r"\d", line))
    short_lines = sum(1 for line in lines if len(TOKEN_RE.findall(line)) <= 8)
    metric_lines = sum(
        1
        for line in lines
        if any(
            phrase in line.lower()
            for phrase in (
                "cash flow",
                "net sales",
                "operating income",
                "employees",
                "q1",
                "q2",
                "q3",
                "q4",
                "ttm",
            )
        )
    )
    return numeric_lines >= 4 and short_lines >= len(lines) // 2 and metric_lines >= 2


def _build_candidate_spans_from_text(
    text: str,
    *,
    period_headers: list[str] | None = None,
) -> list[str]:
    lines = _split_structured_lines(text)
    has_period_headers = bool(_resolve_structured_period_headers(lines, period_headers))
    if _looks_like_structured_numeric_block(lines) or has_period_headers:
        structured_spans = _build_structured_metric_value_spans(
            lines,
            period_headers=period_headers,
        )
        structured_spans.extend(
            _build_flattened_metric_value_spans(
                lines,
                target_metrics=list(_FINANCIAL_METRIC_PHRASES),
                period_headers=period_headers,
            )
        )
        if structured_spans:
            return _deduplicate_text_units(structured_spans)
        return _deduplicate_text_units(_build_line_candidate_spans(lines))
    units = _build_candidate_spans(_split_sentences(text))
    return _deduplicate_text_units(units)


def _collect_page_period_headers(
    contexts: list[RetrievedChunkContext],
) -> dict[tuple[str, int], list[str]]:
    headers_by_page: dict[tuple[str, int], list[str]] = {}
    for context in contexts:
        header_lines, _ = _find_structured_period_headers(_split_structured_lines(context.text))
        if len(header_lines) >= 2:
            for page_number in range(context.page_start, context.page_end + 1):
                page_key = (context.document_id, page_number)
                headers_by_page.setdefault(page_key, header_lines)
    return headers_by_page


def _period_headers_for_context(
    *,
    period_headers_by_page: dict[tuple[str, int], list[str]],
    context: RetrievedChunkContext,
) -> list[str] | None:
    for page_number in range(context.page_start, context.page_end + 1):
        headers = period_headers_by_page.get((context.document_id, page_number))
        if headers:
            return headers
    return None


def build_prompt_for_routing(
    *,
    question: str,
    contexts: list[RetrievedChunkContext],
    routing_decision: QuestionRoutingDecision,
) -> str:
    if routing_decision.answer_mode == "synthesis":
        return build_grounded_synthesis_prompt(question=question, contexts=contexts)
    return build_qa_prompt(question=question, contexts=contexts)


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


def build_grounded_synthesis_prompt(
    question: str,
    contexts: list[RetrievedChunkContext],
) -> str:
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
        "Evidence:\n"
        f"{joined_context}\n\n"
        "Instructions:\n"
        "- Synthesize only claims directly supported by the evidence.\n"
        "- Combine multiple chunks when needed, but do not add outside facts.\n"
        "- Preserve exact metric names, entities, dates, and periods; do not substitute sibling figures.\n"
        "- If any requested facet lacks support, reply exactly with INSUFFICIENT_SUPPORT.\n"
        "- Prefer a short direct answer over repeating the context.\n"
    )


def build_unsupported_classification_prompt(
    *,
    question: str,
    contexts: list[RetrievedChunkContext],
) -> str:
    evidence_blocks = []
    for context in contexts:
        label = (
            f"Document: {context.document_title or context.document_filename} | "
            f"Pages: {context.page_start}-{context.page_end} | "
            f"Chunk: {context.chunk_index} | Score: {context.score:.4f}"
        )
        evidence_blocks.append(f"{label}\n{context.text}")

    joined_evidence = "\n\n---\n\n".join(evidence_blocks)
    return (
        f"Question: {question}\n\n"
        "Evidence:\n"
        f"{joined_evidence}\n\n"
        "Instructions:\n"
        "- Decide whether the evidence is sufficient to answer the full question.\n"
        "- Mark it UNSUPPORTED if any requested facet, constraint, number, date, or entity detail is missing.\n"
        "- For financial or table-style questions, require the exact metric and exact period, not a nearby sibling figure.\n"
        "- Do not infer from nearby or partially matching evidence.\n"
        "- Reply on the first line with exactly SUPPORTED or UNSUPPORTED.\n"
        "- Reply on the second line with a brief reason.\n"
    )


def build_claim_verification_prompt(
    *,
    question: str,
    claim_text: str,
    citations: list[RetrievedChunkContext],
) -> str:
    evidence_blocks = []
    for citation in citations:
        label = (
            f"Document: {citation.document_title or citation.document_filename} | "
            f"Pages: {citation.page_start}-{citation.page_end} | "
            f"Chunk: {citation.chunk_index} | Score: {citation.score:.4f}"
        )
        evidence_blocks.append(f"{label}\n{citation.text}")

    joined_evidence = "\n\n---\n\n".join(evidence_blocks)
    return (
        f"Question: {question}\n"
        f"Claim: {claim_text}\n\n"
        "Evidence:\n"
        f"{joined_evidence}\n\n"
        "Instructions:\n"
        "- Decide whether the claim is fully supported by the evidence.\n"
        "- Reject the claim if the evidence only supports a related metric, date, period, or entity.\n"
        "- Reply on the first line with exactly SUPPORTED or UNSUPPORTED.\n"
        "- Reply on the second line with a brief reason.\n"
    )


def build_answer_repair_prompt(
    *,
    question: str,
    supported_claims: list[ClaimVerification],
) -> str:
    claim_blocks = []
    for index, claim in enumerate(supported_claims, start=1):
        evidence_labels = ", ".join(
            f"pp. {citation.page_start}-{citation.page_end}"
            for citation in claim.citations
        )
        claim_blocks.append(
            f"{index}. Claim: {claim.claim_text}\n"
            f"   Evidence pages: {evidence_labels or 'unknown'}"
        )

    joined_claims = "\n".join(claim_blocks)
    return (
        f"Question: {question}\n\n"
        "Supported claims:\n"
        f"{joined_claims}\n\n"
        "Instructions:\n"
        "- Rewrite the supported claims into one concise grounded answer.\n"
        "- Use only the supported claims listed above.\n"
        "- Preserve exact figures, units, and periods from the supported claims.\n"
        "- Do not add caveats, speculation, or unsupported details.\n"
        "- Keep the answer direct and factual.\n"
        "- If the supported claims are still insufficient to answer, reply exactly with INSUFFICIENT_SUPPORT.\n"
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


def extract_chat_completion_delta(payload: dict) -> str:
    choices = payload.get("choices", [])
    if not choices:
        return ""
    first_choice = choices[0]
    delta = first_choice.get("delta", {})
    content = delta.get("content", "")
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


def _extract_sse_data_line(line: str) -> str:
    normalized_line = line.strip()
    if not normalized_line.startswith("data:"):
        return ""
    return normalized_line.removeprefix("data:").strip()


def _chunk_text_for_stream(text: str) -> list[str]:
    normalized_text = text.strip()
    if not normalized_text:
        return []
    words = normalized_text.split()
    chunks: list[str] = []
    for index, word in enumerate(words):
        suffix = "" if index == len(words) - 1 else " "
        chunks.append(f"{word}{suffix}")
    return chunks


def is_unsupported_answer_text(text: str) -> bool:
    normalized_text = " ".join(text.split()).strip()
    if not normalized_text:
        return True
    if normalized_text == "INSUFFICIENT_SUPPORT":
        return True
    return normalized_text.casefold() == _unsupported_answer().answer_text.casefold()


def select_evidence_citations(
    *,
    answer_text: str,
    contexts: list[RetrievedChunkContext],
    primary_context: RetrievedChunkContext | None = None,
    question_text: str = "",
) -> list[RetrievedChunkContext]:
    citations: list[RetrievedChunkContext] = []
    seen_chunk_ids: set[str] = set()
    citation_indexes_by_chunk_id: dict[str, int] = {}
    support_units = _split_support_units(answer_text)
    period_headers_by_page = _collect_page_period_headers(contexts)
    max_citations = 3
    for index, support_unit in enumerate(support_units):
        ranked = _rank_evidence_contexts(
            answer_text=support_unit,
            question_text=question_text,
            contexts=contexts,
            primary_context=primary_context if index == 0 else None,
            period_headers_by_page=period_headers_by_page,
        )
        if not ranked:
            continue
        _, context = ranked[0]
        if context.chunk_id in seen_chunk_ids:
            citation_index = citation_indexes_by_chunk_id[context.chunk_id]
            citations[citation_index] = _merge_evidence_contexts(
                citations[citation_index],
                context,
            )
            continue
        citations.append(context)
        seen_chunk_ids.add(context.chunk_id)
        citation_indexes_by_chunk_id[context.chunk_id] = len(citations) - 1
        if len(citations) >= max_citations:
            break
        for supplemental_context in _select_supplemental_evidence_contexts(
            answer_text=support_unit,
            question_text=question_text,
            primary_context=context,
            ranked_contexts=ranked[1:],
            seen_chunk_ids=seen_chunk_ids,
        ):
            if supplemental_context.chunk_id in seen_chunk_ids:
                citation_index = citation_indexes_by_chunk_id[supplemental_context.chunk_id]
                citations[citation_index] = _merge_evidence_contexts(
                    citations[citation_index],
                    supplemental_context,
                )
                continue
            citations.append(supplemental_context)
            seen_chunk_ids.add(supplemental_context.chunk_id)
            citation_indexes_by_chunk_id[supplemental_context.chunk_id] = len(citations) - 1
            if len(citations) >= max_citations:
                break
        if len(citations) >= max_citations:
            break

    if citations:
        return citations
    if primary_context is not None:
        return [
            _build_evidence_excerpt(
                answer_text=answer_text,
                question_text=question_text,
                context=primary_context,
                period_headers=_period_headers_for_context(
                    period_headers_by_page=period_headers_by_page,
                    context=primary_context,
                ),
            )
        ]
    return []


def _rank_evidence_contexts(
    *,
    answer_text: str,
    question_text: str,
    contexts: list[RetrievedChunkContext],
    primary_context: RetrievedChunkContext | None = None,
    period_headers_by_page: dict[tuple[str, int], list[str]] | None = None,
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
            period_headers=_period_headers_for_context(
                period_headers_by_page=period_headers_by_page or {},
                context=context,
            ),
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
        score += _exact_metric_lookup_bonus(question_text, evidence_context.text) * 0.4
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


def _merge_evidence_contexts(
    existing_context: RetrievedChunkContext,
    new_context: RetrievedChunkContext,
) -> RetrievedChunkContext:
    existing_text = existing_context.text.strip()
    new_text = new_context.text.strip()
    normalized_existing_text = normalize_query_text(existing_text).casefold()
    normalized_new_text = normalize_query_text(new_text).casefold()
    if not new_text or new_text.casefold() == existing_text.casefold():
        return existing_context
    if normalized_new_text in normalized_existing_text:
        return existing_context
    if normalized_existing_text in normalized_new_text:
        return replace(existing_context, text=new_text)
    merged_text = f"{existing_text} {new_text}".strip()
    return replace(existing_context, text=merged_text)


def _select_supplemental_evidence_contexts(
    *,
    answer_text: str,
    question_text: str,
    primary_context: RetrievedChunkContext,
    ranked_contexts: list[tuple[float, RetrievedChunkContext]],
    seen_chunk_ids: set[str],
) -> list[RetrievedChunkContext]:
    if not _support_unit_may_need_multi_snippet(answer_text=answer_text, question_text=question_text):
        return []

    primary_terms = _tokenize(primary_context.text)
    answer_terms = _tokenize(answer_text)
    distinctive_terms = (
        answer_terms
        | extract_anchor_terms(question_text)
        | extract_constraint_terms(question_text)
        | extract_named_subject_terms(question_text)
    )
    missing_terms = distinctive_terms - primary_terms
    if len(missing_terms) < 2:
        return []

    supplements: list[RetrievedChunkContext] = []
    primary_score = max(primary_context.score, 0.01)
    for ranked_score, candidate in ranked_contexts:
        if candidate.chunk_id in seen_chunk_ids:
            continue
        if candidate.document_id != primary_context.document_id:
            continue
        if candidate.page_start != primary_context.page_start or candidate.page_end != primary_context.page_end:
            continue

        candidate_terms = _tokenize(candidate.text)
        added_missing_terms = missing_terms & candidate_terms
        if len(added_missing_terms) < 2:
            continue
        if ranked_score + 0.35 < primary_score:
            continue

        supplements.append(candidate)
        missing_terms -= candidate_terms
        if len(missing_terms) < 2 or len(supplements) == 1:
            break

    return supplements


def _support_unit_may_need_multi_snippet(*, answer_text: str, question_text: str) -> bool:
    lowered_question = question_text.lower()
    lowered_answer = answer_text.lower()
    if lowered_question.startswith("why ") or " why " in lowered_question:
        return True
    return any(
        marker in lowered_answer
        for marker in (
            "because",
            "due to",
            "driven",
            "reflects",
            "resulted from",
            "as a result",
        )
    )


def _build_evidence_excerpt(
    *,
    answer_text: str,
    question_text: str,
    context: RetrievedChunkContext,
    period_headers: list[str] | None = None,
) -> RetrievedChunkContext:
    candidate_spans = _build_candidate_spans_from_text(
        context.text,
        period_headers=period_headers,
    )
    if not candidate_spans:
        return context

    answer_terms = _tokenize(answer_text)
    question_terms = _tokenize(question_text)
    anchor_terms = extract_anchor_terms(question_text)
    constraint_terms = extract_constraint_terms(question_text)
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
    return replace(context, text=_normalize_evidence_excerpt_text(trimmed_span))


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
        question_overlap
        + answer_type_bonus
        + _financial_metric_alignment_bonus(question_text, span),
        phrase_match,
        brevity_bonus,
    )


def _trim_span_to_support(*, answer_text: str, question_text: str, span: str) -> str:
    if contains_normalized_phrase(answer_text, span):
        return span

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
    if _support_unit_may_need_multi_snippet(answer_text=answer_text, question_text=question_text):
        selected_clause_indexes = [best_index]
        covered_terms = answer_terms & _tokenize(best_clause)
        for neighbor_index in (best_index - 1, best_index + 1):
            if neighbor_index < 0 or neighbor_index >= len(clauses):
                continue
            neighbor_clause = clauses[neighbor_index]
            added_terms = (answer_terms - covered_terms) & _tokenize(neighbor_clause)
            if len(added_terms) < 2:
                continue
            selected_clause_indexes.append(neighbor_index)
            covered_terms |= _tokenize(neighbor_clause)
        if len(selected_clause_indexes) > 1:
            return ", ".join(clauses[index] for index in sorted(selected_clause_indexes))
    if best_index > 0:
        previous_clause = clauses[best_index - 1]
        previous_overlap = len(answer_terms & _tokenize(previous_clause))
        if previous_overlap > 0 and len(_tokenize(previous_clause)) <= 3:
            return f"{previous_clause}, {best_clause}"
    return best_clause


def _normalize_evidence_excerpt_text(text: str) -> str:
    normalized = normalize_query_text(text)
    normalized = _EVIDENCE_CURRENCY_RE.sub("$ ", normalized)
    normalized = _EVIDENCE_PERCENT_RE.sub("%", normalized)
    normalized = _EVIDENCE_OPEN_PAREN_RE.sub("(", normalized)
    normalized = _EVIDENCE_CLOSE_PAREN_RE.sub(")", normalized)
    return normalized.strip(" ,;:")


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
    period_headers_by_page = _collect_page_period_headers(contexts)
    for context in contexts:
        page_headers = _period_headers_for_context(
            period_headers_by_page=period_headers_by_page,
            context=context,
        )
        for sentence_span in _build_candidate_spans_from_text(
            context.text,
            period_headers=page_headers,
        ):
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
            len(candidate.terms),
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


def _expand_selected_candidates_for_explanation(
    *,
    question: str,
    selected_candidates: list[SentenceCandidate],
) -> list[SentenceCandidate]:
    if not _question_prefers_explanatory_followup(question):
        return selected_candidates

    expanded_candidates: list[SentenceCandidate] = []
    for candidate in selected_candidates:
        expanded_candidates.append(
            _expand_candidate_with_followup_sentence(
                question=question,
                candidate=candidate,
            )
        )
    return expanded_candidates


def _question_prefers_explanatory_followup(question: str) -> bool:
    lowered_question = question.lower()
    return lowered_question.startswith("why ") or " why " in lowered_question


def _expand_candidate_with_followup_sentence(
    *,
    question: str,
    candidate: SentenceCandidate,
) -> SentenceCandidate:
    context_sentences = _split_sentences(candidate.context.text)
    if not context_sentences:
        return candidate

    candidate_sentences = _split_sentences(candidate.sentence)
    if not candidate_sentences:
        return candidate

    start_index = _find_sentence_sequence(
        sentences=context_sentences,
        target_sentences=candidate_sentences,
    )
    if start_index is None:
        return candidate

    next_index = start_index + len(candidate_sentences)
    if next_index >= len(context_sentences):
        return candidate

    next_sentence = context_sentences[next_index]
    if not _is_explanatory_followup_sentence(
        question=question,
        current_text=candidate.sentence,
        next_sentence=next_sentence,
    ):
        return candidate

    expanded_sentence = f"{candidate.sentence} {next_sentence}".strip()
    return replace(
        candidate,
        sentence=expanded_sentence,
        terms=_tokenize(expanded_sentence),
    )


def _find_sentence_sequence(
    *,
    sentences: list[str],
    target_sentences: list[str],
) -> int | None:
    normalized_targets = [_normalize_text_unit(sentence) for sentence in target_sentences]
    if not normalized_targets:
        return None

    for index in range(len(sentences) - len(target_sentences) + 1):
        normalized_slice = [
            _normalize_text_unit(sentence)
            for sentence in sentences[index : index + len(target_sentences)]
        ]
        if normalized_slice == normalized_targets:
            return index
    return None


def _normalize_text_unit(text: str) -> str:
    return " ".join(text.split()).casefold()


def _is_explanatory_followup_sentence(
    *,
    question: str,
    current_text: str,
    next_sentence: str,
) -> bool:
    if not _question_prefers_explanatory_followup(question):
        return False

    lowered_current = current_text.lower()
    lowered_next = next_sentence.lower()
    if not any(
        marker in lowered_next
        for marker in ("reflects", "because", "due to", "driven", "resulted", "as a result")
    ):
        return False
    if not any(
        marker in lowered_next
        for marker in ("this ", "these ", "the increase", "increase ", "primarily")
    ):
        return False

    current_terms = _tokenize(current_text)
    next_terms = _tokenize(next_sentence)
    return len(next_terms - current_terms) >= 2


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
    scored_candidates: list[
        tuple[tuple[float, float, float, float, float, float, float, float], SentenceCandidate]
    ] = []
    for candidate in candidates:
        if prefer_explicit_date and not has_explicit_date(candidate.sentence):
            continue
        if prefer_alias and not has_nickname_alias(candidate.sentence, nickname_subject_terms):
            continue
        exact_metric_bonus = _exact_metric_lookup_bonus(facet.text, candidate.sentence)
        facet_score = _score_sentence_against_text(
            prompt_text=facet.text,
            prompt_terms=facet.terms,
            anchor_terms=facet.anchor_terms,
            constraint_terms=facet.constraint_terms,
            candidate=candidate,
        )
        if facet_score < 0.2:
            continue
        if not _candidate_satisfies_metric_requirement(facet.text, candidate.sentence):
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
                    exact_metric_bonus,
                    float(constraint_matches),
                    float(anchor_matches),
                    facet_score,
                    float(added_terms),
                    candidate.score + duplicate_penalty,
                    candidate.context.score,
                    -float(len(candidate.terms)),
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
            -item[0][6],
            -item[0][7],
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
        + _structured_numeric_bonus(prompt_text, candidate.sentence)
        + _financial_metric_alignment_bonus(prompt_text, candidate.sentence)
        + _exact_metric_lookup_bonus(prompt_text, candidate.sentence)
        + _temporal_alignment_bonus(prompt_text, candidate.sentence)
        + max(candidate.context.score, 0.0) * 0.15
        - metadata_noise_score(candidate.sentence) * 0.25
        - _answer_type_penalty(prompt_text, candidate.sentence)
        - _structured_numeric_penalty(prompt_text, candidate.sentence)
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


def _parse_supported_or_unsupported_response(text: str) -> tuple[bool, str] | None:
    normalized_text = text.strip()
    if not normalized_text:
        return None

    lines = [line.strip() for line in normalized_text.splitlines() if line.strip()]
    if not lines:
        return None

    verdict = lines[0].upper()
    if verdict.startswith("SUPPORTED"):
        return True, " ".join(lines[1:]).strip()
    if verdict.startswith("UNSUPPORTED"):
        return False, " ".join(lines[1:]).strip()
    return None


def _classify_question_support_with_provider(
    *,
    question: str,
    contexts: list[RetrievedChunkContext],
    provider: OpenAICompatibleAnswerProvider,
) -> tuple[bool, str] | None:
    if not contexts:
        return False, "No evidence was available to classify."

    verifier_response = provider.complete(
        system_prompt=(
            "You decide whether retrieved document evidence is sufficient to answer a question. "
            "Be conservative. If any required facet or material constraint is missing, mark it unsupported."
        ),
        user_prompt=build_unsupported_classification_prompt(
            question=question,
            contexts=contexts,
        ),
    )
    if not verifier_response:
        return None
    return _parse_supported_or_unsupported_response(verifier_response)


def _verify_claim_with_provider(
    *,
    question: str,
    claim_text: str,
    citations: list[RetrievedChunkContext],
    provider: OpenAICompatibleAnswerProvider,
) -> tuple[bool, str] | None:
    verifier_response = provider.complete(
        system_prompt=(
            "You verify whether a claim is fully supported by cited document evidence. "
            "Be conservative. If the evidence leaves out a material detail, mark it unsupported."
        ),
        user_prompt=build_claim_verification_prompt(
            question=question,
            claim_text=claim_text,
            citations=citations,
        ),
    )
    if not verifier_response:
        return None
    return _parse_supported_or_unsupported_response(verifier_response)


def verify_answer_claims(
    *,
    question: str,
    answer_text: str,
    contexts: list[RetrievedChunkContext],
    citations: list[RetrievedChunkContext] | None = None,
    provider: OpenAICompatibleAnswerProvider | None = None,
) -> AnswerVerification:
    claims = _split_support_units(answer_text)
    if not claims:
        return AnswerVerification(
            verified=False,
            verifier="heuristic",
            claim_count=0,
            supported_claim_count=0,
            average_claim_score=0.0,
            minimum_claim_score=0.0,
            claims=[],
        )

    verification_claims: list[ClaimVerification] = []
    verification_modes: set[str] = set()
    citation_pool = citations or contexts

    for claim_text in claims:
        claim_citations = select_evidence_citations(
            answer_text=claim_text,
            contexts=citation_pool,
            question_text=question,
        )
        heuristic_score = _compute_support_score(
            question=question,
            answer_text=claim_text,
            citations=claim_citations,
        )
        heuristic_supported = heuristic_score >= 0.45
        supported = heuristic_supported
        verifier = "heuristic"
        rationale = (
            "Evidence coverage and term overlap were strong enough for this claim."
            if supported
            else "Evidence coverage was too weak for this claim."
        )

        if provider is not None and claim_citations:
            provider_result = _verify_claim_with_provider(
                question=question,
                claim_text=claim_text,
                citations=claim_citations,
                provider=provider,
            )
            if provider_result is not None:
                provider_supported, provider_rationale = provider_result
                verifier = "heuristic+llm"
                supported = heuristic_supported and provider_supported
                rationale = provider_rationale or rationale
                heuristic_score = (
                    heuristic_score * 0.7
                    + (1.0 if provider_supported else 0.0) * 0.3
                )

        verification_modes.add(verifier)
        verification_claims.append(
            ClaimVerification(
                claim_text=claim_text,
                supported=supported,
                support_score=heuristic_score,
                verifier=verifier,
                rationale=rationale,
                citations=claim_citations,
            )
        )

    supported_claim_count = sum(1 for claim in verification_claims if claim.supported)
    average_claim_score = sum(claim.support_score for claim in verification_claims) / len(
        verification_claims
    )
    minimum_claim_score = min(claim.support_score for claim in verification_claims)
    if verification_modes == {"heuristic"}:
        verifier_name = "heuristic"
    elif verification_modes == {"heuristic+llm"}:
        verifier_name = "heuristic+llm"
    else:
        verifier_name = "mixed"

    return AnswerVerification(
        verified=supported_claim_count == len(verification_claims),
        verifier=verifier_name,
        claim_count=len(verification_claims),
        supported_claim_count=supported_claim_count,
        average_claim_score=average_claim_score,
        minimum_claim_score=minimum_claim_score,
        claims=verification_claims,
    )


def _generate_synthesis_answer(
    *,
    question: str,
    contexts: list[RetrievedChunkContext],
    provider: OpenAICompatibleAnswerProvider,
) -> GeneratedAnswer:
    answer_text = provider.complete(
        system_prompt=(
            "You answer questions only from the supplied document evidence. "
            "Combine evidence across chunks when necessary, but do not invent facts. "
            "If any requested facet is unsupported, reply exactly with INSUFFICIENT_SUPPORT."
        ),
        user_prompt=build_grounded_synthesis_prompt(
            question=question,
            contexts=contexts,
        ),
    )
    if not answer_text:
        return _unsupported_answer(confidence=0.72, support_score=0.0)

    citations = select_evidence_citations(
        answer_text=answer_text,
        contexts=contexts,
        question_text=question,
    )
    answer = GeneratedAnswer(
        answer_text=answer_text,
        supported=True,
        citations=citations,
        answer_mode="synthesis",
    )
    return _finalize_generated_answer(
        answer=answer,
        question=question,
        contexts=contexts,
        fallback_mode="synthesis",
        verification_provider=provider,
    )


def _finalize_generated_answer(
    *,
    answer: GeneratedAnswer,
    question: str,
    contexts: list[RetrievedChunkContext],
    fallback_mode: str,
    verification_provider: OpenAICompatibleAnswerProvider | None = None,
) -> GeneratedAnswer:
    facets = _build_question_facets(question)
    support_threshold = _question_support_threshold(question, facets)
    if not answer.supported or is_unsupported_answer_text(answer.answer_text):
        return _unsupported_answer(
            confidence=0.74,
            support_score=0.0,
            postprocess=AnswerPostprocessTrace(
                question_coverage_score=0.0,
                support_threshold=support_threshold,
                repair_attempted=False,
                repair_applied=False,
                repair_reason="Generated answer was empty or explicitly unsupported.",
            ),
        )

    finalized_answer = replace(
        answer,
        citations=answer.citations
        or select_evidence_citations(
            answer_text=answer.answer_text,
            contexts=contexts,
            question_text=question,
        ),
    )
    support_score = _compute_support_score(
        question=question,
        answer_text=finalized_answer.answer_text,
        citations=finalized_answer.citations,
    )
    question_coverage_score = _compute_question_coverage_score(
        question=question,
        answer_text=finalized_answer.answer_text,
        citations=finalized_answer.citations,
    )
    verification = verify_answer_claims(
        question=question,
        answer_text=finalized_answer.answer_text,
        contexts=contexts,
        citations=finalized_answer.citations,
        provider=verification_provider,
    )
    verification_score = verification.average_claim_score
    support_score = min(
        1.0,
        support_score * 0.25
        + question_coverage_score * 0.15
        + verification_score * 0.65,
    )
    finalized_answer = replace(
        finalized_answer,
        citations=finalized_answer.citations,
        answer_mode=fallback_mode,
        confidence=min(0.98, max(0.35, 0.45 + support_score * 0.5)),
        support_score=support_score,
        verification=verification,
        postprocess=AnswerPostprocessTrace(
            question_coverage_score=question_coverage_score,
            support_threshold=support_threshold,
            repair_attempted=False,
            repair_applied=False,
            claim_count=verification.claim_count,
            supported_claim_count=verification.supported_claim_count,
        ),
    )
    if _answer_clears_acceptance_gate(
        question=question,
        facets=facets,
        answer_text=finalized_answer.answer_text,
        citations=finalized_answer.citations,
        support_score=support_score,
        question_coverage_score=question_coverage_score,
        verification=verification,
        support_threshold=support_threshold,
    ):
        return finalized_answer

    repaired_answer = _repair_partially_supported_answer(
        question=question,
        answer=finalized_answer,
        contexts=contexts,
        fallback_mode=fallback_mode,
        verification_provider=verification_provider,
    )
    if repaired_answer is not None:
        return repaired_answer

    return _unsupported_answer(
        confidence=0.7,
        support_score=support_score,
        verification=verification,
        postprocess=AnswerPostprocessTrace(
            question_coverage_score=question_coverage_score,
            support_threshold=support_threshold,
            repair_attempted=True,
            repair_applied=False,
            repair_reason="Final answer did not clear the support or coverage threshold.",
            claim_count=verification.claim_count,
            supported_claim_count=verification.supported_claim_count,
        ),
    )


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


def _unsupported_answer(
    *,
    confidence: float = 0.72,
    support_score: float = 0.0,
    verification: AnswerVerification | None = None,
    postprocess: AnswerPostprocessTrace | None = None,
) -> GeneratedAnswer:
    return GeneratedAnswer(
        answer_text=(
            "I could not find enough support in the indexed document content "
            "to answer that question."
        ),
        supported=False,
        citations=[],
        answer_mode="unsupported",
        confidence=confidence,
        support_score=support_score,
        verification=verification,
        postprocess=postprocess,
    )


def _repair_partially_supported_answer(
    *,
    question: str,
    answer: GeneratedAnswer,
    contexts: list[RetrievedChunkContext],
    fallback_mode: str,
    verification_provider: OpenAICompatibleAnswerProvider | None = None,
) -> GeneratedAnswer | None:
    facets = _build_question_facets(question)
    support_threshold = _question_support_threshold(question, facets)
    verification = answer.verification
    if verification is None or verification.verified:
        return None

    supported_claims = [claim for claim in verification.claims if claim.supported]
    if not supported_claims:
        return None

    repaired_text = _build_repaired_answer_text(
        question=question,
        supported_claims=supported_claims,
        provider=verification_provider,
    )
    if not repaired_text or is_unsupported_answer_text(repaired_text):
        return None

    repaired_citations = _merge_claim_citations(supported_claims)
    if not repaired_citations:
        repaired_citations = select_evidence_citations(
            answer_text=repaired_text,
            contexts=contexts,
            question_text=question,
        )
    if not repaired_citations:
        return None

    repaired_verification = verify_answer_claims(
        question=question,
        answer_text=repaired_text,
        contexts=contexts,
        citations=repaired_citations,
        provider=verification_provider,
    )
    repaired_support_score = _compute_support_score(
        question=question,
        answer_text=repaired_text,
        citations=repaired_citations,
    )
    repaired_question_coverage = _compute_question_coverage_score(
        question=question,
        answer_text=repaired_text,
        citations=repaired_citations,
    )
    repaired_support_score = min(
        1.0,
        repaired_support_score * 0.25
        + repaired_question_coverage * 0.15
        + repaired_verification.average_claim_score * 0.6,
    )
    if (
        not _answer_clears_acceptance_gate(
            question=question,
            facets=facets,
            answer_text=repaired_text,
            citations=repaired_citations,
            support_score=repaired_support_score,
            question_coverage_score=repaired_question_coverage,
            verification=repaired_verification,
            support_threshold=support_threshold,
        )
    ):
        return None

    confidence = min(0.94, max(0.32, 0.4 + repaired_support_score * 0.45))
    return GeneratedAnswer(
        answer_text=repaired_text,
        supported=True,
        citations=repaired_citations,
        answer_mode=fallback_mode,
        confidence=confidence,
        support_score=repaired_support_score,
        verification=repaired_verification,
        postprocess=AnswerPostprocessTrace(
            question_coverage_score=repaired_question_coverage,
            support_threshold=support_threshold,
            repair_attempted=True,
            repair_applied=True,
            repair_reason="Unsupported claims were removed and the repaired answer cleared verification.",
            claim_count=repaired_verification.claim_count,
            supported_claim_count=repaired_verification.supported_claim_count,
        ),
    )


def _build_repaired_answer_text(
    *,
    question: str,
    supported_claims: list[ClaimVerification],
    provider: OpenAICompatibleAnswerProvider | None = None,
) -> str:
    if provider is not None:
        repaired_text = provider.complete(
            system_prompt=(
                "You rewrite grounded answers using only already-verified supported claims. "
                "Do not add new facts or caveats."
            ),
            user_prompt=build_answer_repair_prompt(
                question=question,
                supported_claims=supported_claims,
            ),
        )
        if repaired_text:
            return repaired_text.strip()
    return " ".join(_ensure_terminal_punctuation(claim.claim_text) for claim in supported_claims)


def _merge_claim_citations(
    claims: list[ClaimVerification],
) -> list[RetrievedChunkContext]:
    merged: list[RetrievedChunkContext] = []
    seen_chunk_ids: set[str] = set()
    for claim in claims:
        for citation in claim.citations:
            if citation.chunk_id in seen_chunk_ids:
                continue
            merged.append(citation)
            seen_chunk_ids.add(citation.chunk_id)
    return merged


def _ensure_terminal_punctuation(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    if stripped.endswith((".", "!", "?")):
        return stripped
    return f"{stripped}."


def _question_requires_strict_support(question: str, facets: list[QuestionFacet]) -> bool:
    return (
        len(facets) > 1
        or _question_has_numeric_intent(question)
        or has_explicit_date(question)
        or any(facet.constraint_terms for facet in facets)
    )


def _question_support_threshold(question: str, facets: list[QuestionFacet]) -> float:
    return 0.58 if _question_requires_strict_support(question, facets) else 0.45


def _answer_clears_acceptance_gate(
    *,
    question: str,
    facets: list[QuestionFacet],
    answer_text: str,
    citations: list[RetrievedChunkContext],
    support_score: float,
    question_coverage_score: float,
    verification: AnswerVerification,
    support_threshold: float,
) -> bool:
    if (
        support_score < support_threshold
        or question_coverage_score < support_threshold
        or not verification.verified
    ):
        return False

    if not _question_requires_strict_support(question, facets):
        return True

    if _question_requires_exact_value_support(question):
        exact_value_score = _exact_value_alignment_score(
            answer_text=answer_text,
            citations=citations,
        )
        if exact_value_score < 0.85:
            return False
        if _is_single_facet_exact_metric_lookup(question=question, facets=facets):
            return _exact_metric_lookup_citation_supports_question(
                question=question,
                citations=citations,
            )

    citation_support_score = _compute_question_support_score(
        question=question,
        facets=facets,
        contexts=citations,
    )
    if citation_support_score < _strict_question_citation_threshold(question, facets):
        return False

    minimum_facet_score = _minimum_question_facet_support_score(
        question=question,
        answer_text=answer_text,
        citations=citations,
    )
    if minimum_facet_score < _minimum_facet_support_threshold(question, facets):
        return False

    return True


def _strict_question_citation_threshold(
    question: str,
    facets: list[QuestionFacet],
) -> float:
    threshold = _question_support_threshold(question, facets)
    if len(facets) > 1:
        return min(0.84, threshold + 0.08)
    if _question_requires_exact_value_support(question):
        return min(0.8, threshold + 0.06)
    return min(0.78, threshold + 0.04)


def _minimum_facet_support_threshold(
    question: str,
    facets: list[QuestionFacet],
) -> float:
    if len(facets) > 1:
        return 0.52
    if _question_requires_exact_value_support(question):
        return 0.62
    if _question_requires_strict_support(question, facets):
        return 0.48
    return 0.0


def _question_metric_support_score(*, question: str, texts: list[str]) -> float:
    if not texts:
        return 0.0

    metrics = _question_metric_phrases(question)
    if not metrics:
        return 1.0

    matched_metrics = sum(
        1 for metric in metrics if any(metric in text.lower() for text in texts)
    )
    return matched_metrics / len(metrics)


def _compute_question_coverage_score(
    *,
    question: str,
    answer_text: str,
    citations: list[RetrievedChunkContext],
) -> float:
    facets = _build_question_facets(question)
    support_terms = _tokenize(answer_text)
    for citation in citations:
        support_terms.update(_tokenize(citation.text))
    support_texts = [answer_text, *(citation.text for citation in citations)]
    metric_support = _question_metric_support_score(question=question, texts=support_texts)
    numeric_support = (
        1.0
        if not _question_has_numeric_intent(question)
        else 1.0
        if any(re.search(r"\d", text) for text in support_texts)
        else 0.0
    )
    facet_scores = _compute_facet_support_scores(facets=facets, support_terms=support_terms)
    facet_score = sum(facet_scores) / len(facet_scores) if facet_scores else 0.0
    return min(
        1.0,
        facet_score * 0.65
        + metric_support * 0.2
        + numeric_support * 0.15,
    )


def _minimum_question_facet_support_score(
    *,
    question: str,
    answer_text: str,
    citations: list[RetrievedChunkContext],
) -> float:
    facets = _build_question_facets(question)
    if not facets:
        return 0.0

    support_terms = _tokenize(answer_text)
    for citation in citations:
        support_terms.update(_tokenize(citation.text))

    facet_scores = _compute_facet_support_scores(facets=facets, support_terms=support_terms)
    if not facet_scores:
        return 0.0
    return min(facet_scores)


def _compute_facet_support_scores(
    *,
    facets: list[QuestionFacet],
    support_terms: set[str],
) -> list[float]:
    facet_scores: list[float] = []
    for facet in facets:
        facet_terms = facet.terms | facet.anchor_terms | facet.constraint_terms
        if not facet_terms:
            continue
        matched_terms = facet_terms & support_terms
        matched_anchor_terms = facet.anchor_terms & support_terms
        matched_constraint_terms = facet.constraint_terms & support_terms
        coverage = len(matched_terms) / max(len(facet_terms), 1)
        anchor_score = (
            len(matched_anchor_terms) / max(len(facet.anchor_terms), 1)
            if facet.anchor_terms
            else coverage
        )
        constraint_score = (
            len(matched_constraint_terms) / max(len(facet.constraint_terms), 1)
            if facet.constraint_terms
            else coverage
        )
        facet_scores.append(
            min(
                1.0,
                coverage * 0.45
                + anchor_score * 0.3
                + constraint_score * 0.25,
            )
        )
    return facet_scores


def _compute_support_score(
    *,
    question: str,
    answer_text: str,
    citations: list[RetrievedChunkContext],
) -> float:
    if not citations:
        return 0.0

    facets = _build_question_facets(question)
    answer_terms = _tokenize(answer_text)
    citation_terms = set().union(*(_tokenize(citation.text) for citation in citations))
    average_citation_score = sum(max(citation.score, 0.0) for citation in citations) / len(citations)
    supported_facet_count = 0
    for facet in facets:
        facet_terms = facet.terms | facet.anchor_terms | facet.constraint_terms
        if not facet_terms:
            continue
        if len(facet_terms & citation_terms) >= max(1, min(2, len(facet_terms))):
            supported_facet_count += 1

    facet_coverage = supported_facet_count / max(len(facets), 1)
    answer_overlap = _normalized_term_overlap(answer_terms, citation_terms)
    citation_count_bonus = min(len(citations), 3) / 3
    return min(
        1.0,
        facet_coverage * 0.5
        + answer_overlap * 0.25
        + min(average_citation_score, 1.0) * 0.15
        + citation_count_bonus * 0.1,
    )


def _question_requires_exact_value_support(question: str) -> bool:
    lowered = question.lower().strip()
    return lowered.startswith(("when ", "how many", "how much", "how fast")) or _question_seeks_exact_metric_value(
        question
    )


def _is_single_facet_exact_metric_lookup(
    *,
    question: str,
    facets: list[QuestionFacet],
) -> bool:
    return len(facets) == 1 and _question_seeks_exact_metric_value(question)


def _exact_metric_lookup_citation_supports_question(
    *,
    question: str,
    citations: list[RetrievedChunkContext],
) -> bool:
    if not citations:
        return False

    citation_texts = [citation.text for citation in citations]
    if _question_metric_support_score(question=question, texts=citation_texts) < 1.0:
        return False

    if _question_has_numeric_intent(question) and not any(
        re.search(r"\d", text) for text in citation_texts
    ):
        return False

    question_pairs = _quarter_year_pairs(question)
    if question_pairs and not any(question_pairs & _quarter_year_pairs(text) for text in citation_texts):
        return False

    if _mentions_trailing_twelve_months(question) and not any(
        _mentions_trailing_twelve_months(text) for text in citation_texts
    ):
        return False

    return True


def _extract_normalized_numeric_values(text: str) -> set[str]:
    return {
        match.group(0).replace("$", "").replace("%", "").replace(",", "").strip()
        for match in re.finditer(r"\$?\d[\d,]*(?:\.\d+)?%?", text)
    }


def _exact_value_alignment_score(
    *,
    answer_text: str,
    citations: list[RetrievedChunkContext],
) -> float:
    answer_values = _extract_normalized_numeric_values(answer_text)
    if not answer_values:
        return 0.0

    citation_values = set().union(
        *(_extract_normalized_numeric_values(citation.text) for citation in citations)
    )
    if not citation_values:
        return 0.0

    matched_values = answer_values & citation_values
    return len(matched_values) / len(answer_values)


def _compute_question_support_score(
    *,
    question: str,
    facets: list[QuestionFacet],
    contexts: list[RetrievedChunkContext],
) -> float:
    if not contexts:
        return 0.0

    context_terms = set().union(*(_tokenize(context.text) for context in contexts))
    metric_match = _question_metric_support_score(question=question, texts=[context.text for context in contexts])
    facet_scores: list[float] = []
    for facet in facets:
        facet_terms = facet.terms | facet.anchor_terms | facet.constraint_terms
        if not facet_terms:
            continue
        matched_anchor_terms = facet.anchor_terms & context_terms
        matched_constraint_terms = facet.constraint_terms & context_terms
        coverage = len(facet_terms & context_terms) / max(len(facet_terms), 1)
        anchor_score = (
            len(matched_anchor_terms) / max(len(facet.anchor_terms), 1)
            if facet.anchor_terms
            else coverage
        )
        constraint_score = (
            len(matched_constraint_terms) / max(len(facet.constraint_terms), 1)
            if facet.constraint_terms
            else coverage
        )
        facet_scores.append(
            min(
                1.0,
                coverage * 0.5
                + anchor_score * 0.3
                + constraint_score * 0.2,
            )
        )

    base_score = sum(facet_scores) / len(facet_scores) if facet_scores else 0.0
    average_context_score = sum(max(context.score, 0.0) for context in contexts) / len(contexts)
    return min(
        1.0,
        base_score * 0.65
        + min(average_context_score, 1.0) * 0.2
        + metric_match * 0.15,
    )


def _retrieval_support_is_too_weak(
    *,
    question: str,
    contexts: list[RetrievedChunkContext],
    facets: list[QuestionFacet],
) -> bool:
    if not contexts:
        return True

    context_terms = set().union(*(_tokenize(context.text) for context in contexts))
    strong_facet_count = 0
    for facet in facets:
        facet_terms = facet.terms
        if not facet_terms:
            continue
        matched_terms = len(facet_terms & context_terms)
        anchor_matches = len(facet.anchor_terms & context_terms)
        constraint_matches = len(facet.constraint_terms & context_terms)
        if matched_terms >= max(1, min(3, len(facet_terms))):
            strong_facet_count += 1
            continue
        if anchor_matches >= _required_anchor_matches(facet.anchor_terms) and constraint_matches >= _required_constraint_matches(
            facet.constraint_terms
        ):
            strong_facet_count += 1

    if strong_facet_count == 0:
        return True
    if len(facets) > 1 and strong_facet_count < len(facets):
        return True
    if _question_has_numeric_intent(question) and not any(re.search(r"\d", context.text) for context in contexts):
        return True
    return False


def _should_run_unsupported_classifier(
    *,
    question: str,
    facets: list[QuestionFacet],
    contexts: list[RetrievedChunkContext],
) -> bool:
    if len(facets) > 1:
        return True
    if _question_has_numeric_intent(question) or has_explicit_date(question):
        return True
    if any(facet.constraint_terms for facet in facets):
        return True
    if not contexts:
        return False

    top_context_score = max(context.score for context in contexts)
    return top_context_score < 0.72


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


def _candidate_satisfies_metric_requirement(question_text: str, candidate_text: str) -> bool:
    question_metrics = _question_metric_phrases(question_text)
    if not question_metrics:
        return True
    lowered_candidate = candidate_text.lower()
    return any(metric in lowered_candidate for metric in question_metrics)


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


def _question_has_numeric_intent(question: str) -> bool:
    lowered = question.lower()
    if lowered.startswith(("how much", "how many", "how fast", "what was", "what were")):
        return True
    return any(term in lowered for term in ("percent", "sales", "income", "cash flow", "employees"))


def _question_metric_phrases(question: str) -> set[str]:
    lowered = question.lower()
    return {phrase for phrase in _FINANCIAL_METRIC_PHRASES if phrase in lowered}


def _question_seeks_exact_metric_value(question: str) -> bool:
    lowered = question.lower().strip()
    if not _question_metric_phrases(question):
        return False
    return lowered.startswith(("what was", "what were", "how much", "how many"))


def _mentions_trailing_twelve_months(text: str) -> bool:
    lowered = text.lower()
    return "trailing twelve months" in lowered or "ttm" in lowered


def _financial_metrics_in_text(text: str) -> set[str]:
    lowered = text.lower()
    return {phrase for phrase in _FINANCIAL_METRIC_PHRASES if phrase in lowered}


def _looks_like_exact_metric_value_span(text: str) -> bool:
    token_count = len(_tokenize(text))
    if token_count == 0 or token_count > 16:
        return False
    if not re.search(r"\d", text):
        return False
    financial_metrics = _financial_metrics_in_text(text)
    if len(financial_metrics) != 1:
        return False
    return bool(_quarter_year_pairs(text) or _mentions_trailing_twelve_months(text))


def _exact_metric_lookup_bonus(question: str, text: str) -> float:
    if not _question_seeks_exact_metric_value(question):
        return 0.0

    question_metrics = _question_metric_phrases(question)
    if not question_metrics:
        return 0.0

    lowered_text = text.lower()
    matched_metrics = {phrase for phrase in question_metrics if phrase in lowered_text}
    if not matched_metrics:
        return 0.0

    bonus = 0.0
    question_pairs = _quarter_year_pairs(question)
    text_pairs = _quarter_year_pairs(text)
    if question_pairs and question_pairs.issubset(text_pairs):
        bonus += 0.38

    if _mentions_trailing_twelve_months(question) and _mentions_trailing_twelve_months(text):
        bonus += 0.1

    if _looks_like_exact_metric_value_span(text):
        bonus += 0.16

    if re.search(r"\b\d{1,3}(?:,\d{3})+\b", text):
        bonus += 0.08

    conflicting_metrics = _financial_metrics_in_text(text) - question_metrics
    if conflicting_metrics:
        bonus -= min(0.24, len(conflicting_metrics) * 0.12)

    return bonus


def _financial_metric_alignment_bonus(question: str, text: str) -> float:
    question_metrics = _question_metric_phrases(question)
    if not question_metrics:
        return 0.0

    lowered_text = text.lower()
    matched_metrics = {phrase for phrase in question_metrics if phrase in lowered_text}
    if not matched_metrics:
        return 0.0

    bonus = min(0.34, len(matched_metrics) * 0.24)
    if _question_has_numeric_intent(question) and re.search(r"\d", text):
        bonus += 0.08
    return bonus


def _quarter_year_pairs(text: str) -> set[str]:
    return {match.group(0).lower() for match in _QUARTER_YEAR_RE.finditer(text)}


def _temporal_alignment_bonus(question: str, text: str) -> float:
    question_pairs = _quarter_year_pairs(question)
    if not question_pairs:
        return 0.0
    text_pairs = _quarter_year_pairs(text)
    if not text_pairs:
        return 0.0
    matched_pairs = question_pairs & text_pairs
    if not matched_pairs:
        return 0.0
    bonus = min(0.18, len(matched_pairs) * 0.18)
    if len(text_pairs) == len(matched_pairs):
        bonus += 0.06
    return bonus


def _structured_numeric_penalty(question: str, text: str) -> float:
    if not _question_has_numeric_intent(question):
        return 0.0

    penalty = 0.0
    if not re.search(r"\d", text):
        penalty += 0.22

    question_metrics = _question_metric_phrases(question)
    if not question_metrics:
        return penalty

    lowered_text = text.lower()
    matched_metrics = {phrase for phrase in question_metrics if phrase in lowered_text}
    conflicting_metrics = {
        phrase
        for phrase in _FINANCIAL_METRIC_PHRASES
        if phrase in lowered_text and phrase not in question_metrics
    }
    if not matched_metrics and conflicting_metrics:
        penalty += 0.3
    elif matched_metrics and conflicting_metrics:
        penalty += 0.16

    question_pairs = _quarter_year_pairs(question)
    text_pairs = _quarter_year_pairs(text)
    if question_pairs and len(text_pairs) > len(question_pairs):
        penalty += min(0.24, (len(text_pairs) - len(question_pairs)) * 0.08)

    numeric_token_count = sum(1 for token in TOKEN_RE.findall(text) if any(character.isdigit() for character in token))
    if numeric_token_count > 6:
        penalty += min(0.5, (numeric_token_count - 6) * 0.045)
    return penalty


def _structured_numeric_bonus(question: str, text: str) -> float:
    if not _question_has_numeric_intent(question):
        return 0.0

    bonus = 0.0
    normalized_question = " ".join(normalized_token_sequence(question, drop_stopwords=True))
    normalized_text = " ".join(normalized_token_sequence(text, drop_stopwords=True))
    numeric_tokens = re.findall(r"\$?\d[\d,.]*%?", text)
    currency_count = len(re.findall(r"\$\s*\d[\d,.]*", text))
    percent_count = len(re.findall(r"\d[\d,.]*\s*%", text))
    large_number_count = len(re.findall(r"\b\d{1,3}(?:,\d{3})+\b", text))

    if numeric_tokens:
        bonus += min(0.16, len(numeric_tokens) * 0.03)
    if currency_count:
        bonus += min(0.08, currency_count * 0.03)
    if percent_count and ("percent" in question.lower() or "how fast" in question.lower()):
        bonus += min(0.08, percent_count * 0.04)
    if large_number_count and question.lower().startswith("how many"):
        bonus += min(0.14, large_number_count * 0.03)
    if (
        any(marker in question.lower() for marker in ("q1", "q2", "q3", "q4", "quarter"))
        and large_number_count >= 6
    ):
        bonus += 0.16

    for phrase in (
        "net sales",
        "operating income",
        "free cash flow",
        "capital expenditures",
        "employees",
        "aws segment sales",
    ):
        if phrase in question.lower() and phrase in text.lower():
            bonus += 0.18

    if normalized_question and normalized_text:
        question_pairs = {
            " ".join(pair)
            for pair in zip(
                normalized_token_sequence(question, drop_stopwords=True),
                normalized_token_sequence(question, drop_stopwords=True)[1:],
                strict=False,
            )
        }
        matched_pairs = sum(1 for pair in question_pairs if pair and pair in normalized_text)
        if matched_pairs:
            bonus += min(0.12, matched_pairs * 0.04)

    if any(character.isdigit() for character in question):
        question_years = set(re.findall(r"\b\d{4}\b", question))
        text_years = set(re.findall(r"\b\d{4}\b", text))
        if question_years & text_years:
            bonus += 0.06

    return bonus


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
        if re.search(r"\b\d{1,3}(?:,\d{3})+\b", text) or re.search(r"\b\d{5,}\b", text):
            bonus += 0.22
        elif re.search(r"\b\d+\b", text.lower()):
            bonus += 0.06

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
    if lowered_question.startswith("how many"):
        if not (re.search(r"\b\d{1,3}(?:,\d{3})+\b", text) or re.search(r"\b\d{5,}\b", text)):
            penalty += 0.24
    if "nickname" in lowered_question:
        if not has_nickname_alias(text, nickname_subject_terms):
            penalty += 0.24
        named_terms = extract_named_subject_terms(question)
        if named_terms and not (named_terms & _tokenize(text)):
            penalty += 0.28
    return penalty
