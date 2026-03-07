from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ebook_rag_api.db import get_db_session
from ebook_rag_api.schemas.qa import (
    QACitation,
    QARequest,
    QAResponse,
    QATimingBreakdown,
    QATrace,
    QATraceChunk,
)
from ebook_rag_api.services.qa import ask_question_with_trace

router = APIRouter()


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
    answer = qa_trace.answer
    return QAResponse(
        normalized_question=normalized_question,
        answer=answer.answer_text,
        supported=answer.supported,
        citations=[
            QACitation(
                chunk_id=context.chunk_id,
                document_id=context.document_id,
                document_title=context.document_title,
                document_filename=context.document_filename,
                chunk_index=context.chunk_index,
                page_start=context.page_start,
                page_end=context.page_end,
                text=context.text,
                dense_score=context.dense_score,
                lexical_score=context.lexical_score,
                hybrid_score=context.hybrid_score,
                rerank_score=context.rerank_score,
                score=context.score,
            )
            for context in answer.citations
        ],
        retrieved_chunk_count=len(qa_trace.retrieved_chunks),
        trace=(
            QATrace(
                answer_provider=qa_trace.answer_provider,
                retrieved_chunks=[
                    QATraceChunk(
                        chunk_id=context.chunk_id,
                        document_id=context.document_id,
                        document_title=context.document_title,
                        document_filename=context.document_filename,
                        chunk_index=context.chunk_index,
                        page_start=context.page_start,
                        page_end=context.page_end,
                        text=context.text,
                        dense_score=context.dense_score,
                        lexical_score=context.lexical_score,
                        hybrid_score=context.hybrid_score,
                        rerank_score=context.rerank_score,
                        score=context.score,
                    )
                    for context in qa_trace.retrieved_chunks
                ],
                selected_contexts=[
                    QATraceChunk(
                        chunk_id=context.chunk_id,
                        document_id=context.document_id,
                        document_title=context.document_title,
                        document_filename=context.document_filename,
                        chunk_index=context.chunk_index,
                        page_start=context.page_start,
                        page_end=context.page_end,
                        text=context.text,
                        dense_score=context.dense_score,
                        lexical_score=context.lexical_score,
                        hybrid_score=context.hybrid_score,
                        rerank_score=context.rerank_score,
                        score=context.score,
                    )
                    for context in qa_trace.selected_contexts
                ],
                prompt_snapshot=qa_trace.prompt_snapshot,
                timings=QATimingBreakdown(
                    normalization_ms=qa_trace.timings.normalization_ms,
                    retrieval_ms=qa_trace.timings.retrieval_ms,
                    context_assembly_ms=qa_trace.timings.context_assembly_ms,
                    answer_generation_ms=qa_trace.timings.answer_generation_ms,
                    total_ms=qa_trace.timings.total_ms,
                ),
            )
            if payload.include_trace
            else None
        ),
    )
