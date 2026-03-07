from __future__ import annotations

from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON
from sqlalchemy.dialects import postgresql

from ebook_rag_api.core.config import get_settings


def get_embedding_dimensions() -> int:
    dimensions = get_settings().embedding_dimensions
    if dimensions <= 0:
        raise ValueError("Embedding dimensions must be positive.")
    return dimensions


def embedding_vector_type() -> Any:
    return Vector(get_embedding_dimensions()).with_variant(JSON(), "sqlite")


def is_postgresql_dialect(name: str) -> bool:
    return name == postgresql.dialect.name
