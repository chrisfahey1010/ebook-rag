import math
import re
from functools import lru_cache
from typing import Any, Protocol

import httpx

from ebook_rag_api.core.config import get_settings

STOPWORDS = {
    "a",
    "about",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "what",
    "which",
    "who",
    "with",
}
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


class Reranker(Protocol):
    def score(self, query: str, passages: list[str]) -> list[float]:
        ...


class TokenOverlapReranker:
    def score(self, query: str, passages: list[str]) -> list[float]:
        query_terms = _tokenize(query)
        query_term_count = max(len(query_terms), 1)
        scores: list[float] = []

        for passage in passages:
            passage_terms = _tokenize(passage)
            if not passage_terms:
                scores.append(0.0)
                continue

            overlap = query_terms & passage_terms
            coverage = len(overlap) / query_term_count
            precision = len(overlap) / len(passage_terms)
            phrase_bonus = 0.15 if _contains_query_phrase(query, passage) else 0.0
            scores.append(coverage * 0.75 + precision * 0.25 + phrase_bonus)

        return scores


class CrossEncoderReranker:
    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for reranker_provider=cross_encoder."
            ) from exc

        self.model_name = model_name
        self._model = CrossEncoder(model_name)

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        pairs = [(query, passage) for passage in passages]
        raw_scores = self._model.predict(pairs)
        return [_sigmoid(float(score)) for score in raw_scores]


class OpenAICompatibleReranker:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: float,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []

        payload: dict[str, Any] = {
            "model": self.model,
            "query": query,
            "documents": passages,
            "top_n": len(passages),
            "return_documents": False,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(
                f"{self.base_url}/rerank",
                json=payload,
                headers=headers,
            )
            response.raise_for_status()

        results = response.json().get("results", [])
        scores = [0.0] * len(passages)
        for item in results:
            index = item.get("index")
            relevance_score = item.get("relevance_score", 0.0)
            if isinstance(index, int) and 0 <= index < len(passages):
                scores[index] = float(relevance_score)
        return scores


@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    settings = get_settings()
    if settings.reranker_provider == "token_overlap":
        return TokenOverlapReranker()
    if settings.reranker_provider == "cross_encoder":
        return CrossEncoderReranker(model_name=settings.reranker_model)
    if settings.reranker_provider == "openai_compatible":
        return OpenAICompatibleReranker(
            base_url=settings.reranker_base_url,
            api_key=settings.reranker_api_key,
            model=settings.reranker_model,
            timeout_seconds=settings.reranker_timeout_seconds,
        )
    raise ValueError(f"Unsupported reranker provider: {settings.reranker_provider}")


def _tokenize(text: str) -> set[str]:
    return {
        token
        for token in TOKEN_RE.findall(text.lower())
        if token not in STOPWORDS and len(token) > 1
    }


def _contains_query_phrase(query: str, passage: str) -> bool:
    normalized_query = " ".join(TOKEN_RE.findall(query.lower()))
    normalized_passage = " ".join(TOKEN_RE.findall(passage.lower()))
    return bool(normalized_query and normalized_query in normalized_passage)


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))
