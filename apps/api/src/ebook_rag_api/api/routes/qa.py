import json
from time import perf_counter

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from ebook_rag_api.db import get_db_session
from ebook_rag_api.schemas.qa import (
    QACitation,
    QAQuestionRouter,
    QARequest,
    QAResponse,
    QARuntimeMetadata,
    QATimingBreakdown,
    QATrace,
    QATraceChunk,
)
from ebook_rag_api.services.qa import (
    QATrace as ServiceQATrace,
    ask_question_with_trace,
    prepare_qa_request,
    stream_answer_for_request,
    RetrievedChunkContext,
)

router = APIRouter()


def _serialize_context(context: RetrievedChunkContext) -> QATraceChunk:
    return QATraceChunk(
        chunk_id=context.chunk_id,
        document_id=context.document_id,
        document_title=context.document_title,
        document_filename=context.document_filename,
        chunk_index=context.chunk_index,
        page_start=context.page_start,
        page_end=context.page_end,
        text=context.text,
        provenance=context.provenance,
        dense_score=context.dense_score,
        lexical_score=context.lexical_score,
        hybrid_score=context.hybrid_score,
        rerank_score=context.rerank_score,
        score=context.score,
    )


def _serialize_citation(context: RetrievedChunkContext) -> QACitation:
    return QACitation(
        chunk_id=context.chunk_id,
        document_id=context.document_id,
        document_title=context.document_title,
        document_filename=context.document_filename,
        chunk_index=context.chunk_index,
        page_start=context.page_start,
        page_end=context.page_end,
        text=context.text,
        provenance=context.provenance,
        dense_score=context.dense_score,
        lexical_score=context.lexical_score,
        hybrid_score=context.hybrid_score,
        rerank_score=context.rerank_score,
        score=context.score,
    )


def _build_qa_response(
    *,
    normalized_question: str,
    qa_trace: ServiceQATrace,
    include_trace: bool,
) -> QAResponse:
    answer = qa_trace.answer
    return QAResponse(
        normalized_question=normalized_question,
        answer=answer.answer_text,
        supported=answer.supported,
        answer_mode=answer.answer_mode,
        confidence=answer.confidence,
        support_score=answer.support_score,
        citations=[_serialize_citation(context) for context in answer.citations],
        retrieved_chunk_count=len(qa_trace.retrieved_chunks),
        trace=(
            QATrace(
                answer_provider=qa_trace.answer_provider,
                answer_mode=qa_trace.answer_mode,
                question_router=QAQuestionRouter(
                    answer_mode=qa_trace.question_router.answer_mode,
                    reason=qa_trace.question_router.reason,
                    facet_count=qa_trace.question_router.facet_count,
                    context_count=qa_trace.question_router.context_count,
                    should_use_generative=qa_trace.question_router.should_use_generative,
                ),
                runtime=QARuntimeMetadata(
                    embedding_provider=qa_trace.runtime.embedding_provider,
                    embedding_model=qa_trace.runtime.embedding_model,
                    reranker_provider=qa_trace.runtime.reranker_provider,
                    reranker_model=qa_trace.runtime.reranker_model,
                    answer_provider=qa_trace.runtime.answer_provider,
                    answer_model=qa_trace.runtime.answer_model,
                ),
                retrieved_chunks=[_serialize_context(context) for context in qa_trace.retrieved_chunks],
                selected_contexts=[_serialize_context(context) for context in qa_trace.selected_contexts],
                cited_contexts=[_serialize_context(context) for context in qa_trace.cited_contexts],
                prompt_snapshot=qa_trace.prompt_snapshot,
                timings=QATimingBreakdown(
                    normalization_ms=qa_trace.timings.normalization_ms,
                    retrieval_ms=qa_trace.timings.retrieval_ms,
                    context_assembly_ms=qa_trace.timings.context_assembly_ms,
                    answer_generation_ms=qa_trace.timings.answer_generation_ms,
                    total_ms=qa_trace.timings.total_ms,
                ),
            )
            if include_trace
            else None
        ),
    )


def _sse_event(event: str, payload: dict[str, object]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


@router.post("/ask", response_model=QAResponse, summary="Answer a question from indexed content")
def answer_question(
    payload: QARequest, session: Session = Depends(get_db_session)
) -> QAResponse:
    normalized_question, qa_trace = ask_question_with_trace(
        session=session,
        question=payload.question,
        top_k=payload.top_k,
        document_id=payload.document_id,
        include_prompt_snapshot=payload.include_trace,
    )
    return _build_qa_response(
        normalized_question=normalized_question,
        qa_trace=qa_trace,
        include_trace=payload.include_trace,
    )


@router.post("/ask-stream", summary="Stream a grounded answer from indexed content")
def answer_question_stream(
    payload: QARequest, session: Session = Depends(get_db_session)
) -> StreamingResponse:
    prepared_request = prepare_qa_request(
        session=session,
        question=payload.question,
        top_k=payload.top_k,
        document_id=payload.document_id,
        include_prompt_snapshot=payload.include_trace,
    )

    def event_stream():
        answer_parts: list[str] = []
        try:
            yield _sse_event(
                "start",
                {
                    "normalized_question": prepared_request.normalized_question,
                    "retrieved_chunk_count": len(prepared_request.retrieved_chunks),
                },
            )
            answer_started_at = perf_counter()
            final_answer, answer_chunks = stream_answer_for_request(prepared_request)
            for chunk in answer_chunks:
                answer_parts.append(chunk.delta)
                yield _sse_event("answer_delta", {"delta": chunk.delta})

            answered_at = perf_counter()
            qa_trace = ServiceQATrace(
                answer_provider=type(prepared_request.answer_provider).__name__,
                answer_mode=final_answer.answer_mode,
                question_router=prepared_request.routing_decision,
                runtime=prepared_request.runtime,
                retrieved_chunks=prepared_request.retrieved_chunks,
                selected_contexts=prepared_request.selected_contexts,
                cited_contexts=final_answer.citations,
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
                answer=final_answer,
            )
            response = _build_qa_response(
                normalized_question=prepared_request.normalized_question,
                qa_trace=qa_trace,
                include_trace=payload.include_trace,
            )
            yield _sse_event("complete", response.model_dump())
        except Exception as exc:
            yield _sse_event("error", {"detail": str(exc)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
