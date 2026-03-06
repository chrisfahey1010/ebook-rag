import hashlib
import math
import re
from collections.abc import Sequence
from functools import lru_cache
from typing import Protocol

from ebook_rag_api.core.config import get_settings

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


class EmbeddingProvider(Protocol):
    dimensions: int

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        ...


class HashingEmbeddingProvider:
    def __init__(self, dimensions: int) -> None:
        if dimensions <= 0:
            raise ValueError("Embedding dimensions must be positive.")
        self.dimensions = dimensions

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_text(text) for text in texts]

    def _embed_text(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in TOKEN_RE.findall(text.lower()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        magnitude = math.sqrt(sum(value * value for value in vector))
        if magnitude == 0:
            return vector
        return [value / magnitude for value in vector]


@lru_cache(maxsize=1)
def get_embedding_provider() -> EmbeddingProvider:
    settings = get_settings()
    if settings.embedding_provider == "hashing":
        return HashingEmbeddingProvider(dimensions=settings.embedding_dimensions)
    raise ValueError(
        f"Unsupported embedding provider: {settings.embedding_provider}"
    )
