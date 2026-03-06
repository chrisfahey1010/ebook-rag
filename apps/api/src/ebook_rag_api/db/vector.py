from __future__ import annotations

from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON
from sqlalchemy.dialects import postgresql


EMBEDDING_DIMENSIONS = 128


def embedding_vector_type() -> Any:
    return Vector(EMBEDDING_DIMENSIONS).with_variant(JSON(), "sqlite")


def is_postgresql_dialect(name: str) -> bool:
    return name == postgresql.dialect.name
