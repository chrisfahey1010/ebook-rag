from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ebook_rag_api.db import get_db_session
from ebook_rag_api.schemas.retrieval import (
    RetrievalMatch,
    RetrievalSearchRequest,
    RetrievalSearchResponse,
)
from ebook_rag_api.services.retrieval import search_chunks

router = APIRouter()


@router.post("/search", response_model=RetrievalSearchResponse, summary="Search document chunks")
def search_document_chunks(
    payload: RetrievalSearchRequest, session: Session = Depends(get_db_session)
) -> RetrievalSearchResponse:
    normalized_query, matches = search_chunks(
        session=session,
        query=payload.query,
        top_k=payload.top_k,
        document_id=payload.document_id,
    )
    return RetrievalSearchResponse(
        normalized_query=normalized_query,
        matches=[
            RetrievalMatch(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                document_title=chunk.document.title,
                document_filename=chunk.document.original_filename,
                chunk_index=chunk.chunk_index,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                text=chunk.text,
                score=score,
            )
            for chunk, score in matches
        ],
    )
