import re
from dataclasses import dataclass

from sqlalchemy import desc, select
from sqlalchemy.orm import Session, joinedload

from ebook_rag_api.core.config import get_settings
from ebook_rag_api.db.vector import is_postgresql_dialect
from ebook_rag_api.models import Document, DocumentChunk
from ebook_rag_api.services.embeddings import get_embedding_provider
from ebook_rag_api.services.reranking import get_reranker

WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class ChunkSearchMatch:
    chunk: DocumentChunk
    dense_score: float
    rerank_score: float
    score: float


def normalize_query(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip()


def cosine_similarity(left: list[float], right: list[float]) -> float:
    import math

    if not left or not right or len(left) != len(right):
        return 0.0

    dot_product = sum(a * b for a, b in zip(left, right, strict=False))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot_product / (left_norm * right_norm)


def search_chunks(
    session: Session, query: str, top_k: int, document_id: str | None = None
) -> tuple[str, list[ChunkSearchMatch]]:
    normalized_query = normalize_query(query)
    if not normalized_query:
        return "", []

    settings = get_settings()
    provider = get_embedding_provider()
    query_embedding = provider.embed_texts([normalized_query])[0]

    bind = session.get_bind()
    if bind is not None and is_postgresql_dialect(bind.dialect.name):
        score_expression = (1 - DocumentChunk.embedding_vector.cosine_distance(query_embedding)).label(
            "score"
        )
        statement = (
            select(DocumentChunk, score_expression)
            .join(Document)
            .options(joinedload(DocumentChunk.document))
            .where(Document.status == "ready")
            .where(DocumentChunk.embedding_vector.is_not(None))
            .where(DocumentChunk.embedding_dimensions == len(query_embedding))
        )
        if document_id is not None:
            statement = statement.where(DocumentChunk.document_id == document_id)

        candidate_limit = max(top_k, top_k * settings.rerank_candidate_multiplier)
        statement = statement.order_by(
            desc(score_expression),
            DocumentChunk.page_start.asc(),
            DocumentChunk.chunk_index.asc(),
        ).limit(candidate_limit)
        rows = session.execute(statement).all()
        dense_matches = [(row[0], row[1]) for row in rows]
        return normalized_query, rerank_matches(normalized_query, dense_matches, top_k)

    statement = (
        select(DocumentChunk)
        .join(Document)
        .options(joinedload(DocumentChunk.document))
        .where(Document.status == "ready")
    )
    if document_id is not None:
        statement = statement.where(DocumentChunk.document_id == document_id)

    candidates = list(session.scalars(statement))
    scored_matches: list[tuple[DocumentChunk, float]] = []
    for chunk in candidates:
        if (
            chunk.embedding_vector is None
            or chunk.embedding_dimensions != settings.embedding_dimensions
        ):
            continue
        score = cosine_similarity(query_embedding, chunk.embedding_vector)
        scored_matches.append((chunk, score))

    scored_matches.sort(key=lambda item: (-item[1], item[0].page_start, item[0].chunk_index))
    candidate_limit = max(top_k, top_k * settings.rerank_candidate_multiplier)
    return normalized_query, rerank_matches(
        normalized_query,
        scored_matches[:candidate_limit],
        top_k,
    )


def rerank_matches(
    query: str,
    dense_matches: list[tuple[DocumentChunk, float]],
    top_k: int,
) -> list[ChunkSearchMatch]:
    if not dense_matches:
        return []

    reranker = get_reranker()
    rerank_scores = reranker.score(query, [chunk.text for chunk, _ in dense_matches])
    matches = [
        ChunkSearchMatch(
            chunk=chunk,
            dense_score=dense_score,
            rerank_score=rerank_score,
            score=(dense_score * 0.35) + (rerank_score * 0.65),
        )
        for (chunk, dense_score), rerank_score in zip(
            dense_matches, rerank_scores, strict=True
        )
    ]
    matches.sort(
        key=lambda item: (
            -item.score,
            -item.rerank_score,
            -item.dense_score,
            item.chunk.page_start,
            item.chunk.chunk_index,
        )
    )
    return matches[:top_k]
