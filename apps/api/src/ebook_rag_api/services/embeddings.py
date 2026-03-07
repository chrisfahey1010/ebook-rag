import hashlib
import math
import re
from collections.abc import Sequence
from functools import lru_cache
from typing import Any, Protocol

import httpx

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


class SentenceTransformerEmbeddingProvider:
    def __init__(self, model_name: str, dimensions: int) -> None:
        if dimensions <= 0:
            raise ValueError("Embedding dimensions must be positive.")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for embedding_provider=sentence_transformer."
            ) from exc

        self.dimensions = dimensions
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        encoded = self._model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return [_validate_embedding_dimensions(vector, self.dimensions) for vector in encoded]


class OpenAICompatibleEmbeddingProvider:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        dimensions: int,
        timeout_seconds: float,
    ) -> None:
        if dimensions <= 0:
            raise ValueError("Embedding dimensions must be positive.")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.dimensions = dimensions
        self.timeout_seconds = timeout_seconds

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        payload: dict[str, Any] = {
            "model": self.model,
            "input": list(texts),
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(
                f"{self.base_url}/embeddings",
                json=payload,
                headers=headers,
            )
            response.raise_for_status()

        data = response.json().get("data", [])
        if len(data) != len(texts):
            raise RuntimeError("Embedding provider returned an unexpected number of embeddings.")
        return [
            _validate_embedding_dimensions(item.get("embedding", []), self.dimensions)
            for item in data
        ]


@lru_cache(maxsize=1)
def get_embedding_provider() -> EmbeddingProvider:
    settings = get_settings()
    if settings.embedding_provider == "hashing":
        return HashingEmbeddingProvider(dimensions=settings.embedding_dimensions)
    if settings.embedding_provider == "sentence_transformer":
        return SentenceTransformerEmbeddingProvider(
            model_name=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
        )
    if settings.embedding_provider == "openai_compatible":
        return OpenAICompatibleEmbeddingProvider(
            base_url=settings.embedding_base_url,
            api_key=settings.embedding_api_key,
            model=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
            timeout_seconds=settings.embedding_timeout_seconds,
        )
    raise ValueError(
        f"Unsupported embedding provider: {settings.embedding_provider}"
    )


def _validate_embedding_dimensions(vector: Sequence[float], dimensions: int) -> list[float]:
    if dimensions <= 0:
        raise ValueError("Embedding dimensions must be positive.")

    values = [float(value) for value in vector]
    if len(values) == dimensions:
        return _normalize_vector(values)
    raise RuntimeError(
        f"Embedding provider returned {len(values)} dimensions, expected {dimensions}."
    )


def _normalize_vector(vector: list[float]) -> list[float]:
    magnitude = math.sqrt(sum(value * value for value in vector))
    if magnitude == 0:
        return vector
    return [value / magnitude for value in vector]
