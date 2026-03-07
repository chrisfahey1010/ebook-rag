from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from ebook_rag_api.db import get_db_session
from ebook_rag_api.models import Document, DocumentChunk
from ebook_rag_api.schemas.retrieval import (
    DebugDocumentChunk,
    DebugDocumentChunksResponse,
    DebugRerankRequest,
    DebugRerankResponse,
    DebugRerankResult,
    RetrievalMatch,
    RetrievalSearchRequest,
    RetrievalSearchResponse,
)
from ebook_rag_api.services.retrieval import search_chunks
from ebook_rag_api.services.reranking import TokenOverlapReranker, get_reranker

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


@router.get(
    "/documents/{document_id}/chunks",
    response_model=DebugDocumentChunksResponse,
    summary="Inspect persisted chunks for a document",
)
def debug_document_chunks(
    document_id: str, session: Session = Depends(get_db_session)
) -> DebugDocumentChunksResponse:
    document = session.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found.")

    chunks = list(
        session.scalars(
            select(DocumentChunk)
            .where(DocumentChunk.document_id == document_id)
            .order_by(DocumentChunk.chunk_index.asc())
        )
    )
    return DebugDocumentChunksResponse(
        document_id=document.id,
        document_title=document.title,
        document_filename=document.original_filename,
        chunk_count=len(chunks),
        chunking_config=document.chunking_config,
        chunks=[
            DebugDocumentChunk(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                document_title=document.title,
                document_filename=document.original_filename,
                chunk_index=chunk.chunk_index,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                heading=chunk.heading,
                text=chunk.text,
                token_estimate=chunk.token_estimate,
                provenance=chunk.provenance,
                embedding_dimensions=chunk.embedding_dimensions,
            )
            for chunk in chunks
        ],
    )


@router.post(
    "/rerank",
    response_model=DebugRerankResponse,
    summary="Inspect reranker scores for explicit passages",
)
def debug_rerank(payload: DebugRerankRequest) -> DebugRerankResponse:
    reranker = get_reranker()
    try:
        scores = reranker.score(payload.query, payload.passages)
    except Exception:
        reranker = TokenOverlapReranker()
        scores = reranker.score(payload.query, payload.passages)

    ranked_results = [
        DebugRerankResult(index=index, text=text, rerank_score=score)
        for index, (text, score) in enumerate(
            zip(payload.passages, scores, strict=True)
        )
    ]
    ranked_results.sort(key=lambda item: (-item.rerank_score, item.index))
    return DebugRerankResponse(
        query=payload.query,
        reranker=type(reranker).__name__,
        results=ranked_results,
    )
