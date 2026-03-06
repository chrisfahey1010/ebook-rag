import math
import re

from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from ebook_rag_api.core.config import get_settings
from ebook_rag_api.models import Document, DocumentChunk
from ebook_rag_api.services.embeddings import get_embedding_provider

WHITESPACE_RE = re.compile(r"\s+")


def normalize_query(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip()


def cosine_similarity(left: list[float], right: list[float]) -> float:
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
) -> tuple[str, list[tuple[DocumentChunk, float]]]:
    normalized_query = normalize_query(query)
    if not normalized_query:
        return "", []

    settings = get_settings()
    provider = get_embedding_provider()
    query_embedding = provider.embed_texts([normalized_query])[0]

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
    return normalized_query, scored_matches[:top_k]
