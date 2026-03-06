from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ebook_rag_api.db import get_db_session
from ebook_rag_api.schemas.qa import QACitation, QARequest, QAResponse
from ebook_rag_api.services.qa import ask_question

router = APIRouter()


@router.post("/ask", response_model=QAResponse, summary="Answer a question from indexed content")
def answer_question(
    payload: QARequest, session: Session = Depends(get_db_session)
) -> QAResponse:
    normalized_question, contexts, answer = ask_question(
        session=session,
        question=payload.question,
        top_k=payload.top_k,
        document_id=payload.document_id,
    )
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
                score=context.score,
            )
            for context in answer.citations
        ],
        retrieved_chunk_count=len(contexts),
    )
