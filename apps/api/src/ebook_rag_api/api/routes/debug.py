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


@router.post("/retrieve", response_model=RetrievalSearchResponse, summary="Inspect ranked retrieval candidates")
def debug_retrieve(
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
                chunk_id=match.chunk.id,
                document_id=match.chunk.document_id,
                document_title=match.chunk.document.title,
                document_filename=match.chunk.document.original_filename,
                chunk_index=match.chunk.chunk_index,
                page_start=match.chunk.page_start,
                page_end=match.chunk.page_end,
                text=match.chunk.text,
                dense_score=match.dense_score,
                lexical_score=match.lexical_score,
                hybrid_score=match.hybrid_score,
                rerank_score=match.rerank_score,
                score=match.score,
            )
            for match in matches
        ],
    )
