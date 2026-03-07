import math
from functools import lru_cache
from typing import Any, Protocol

import httpx

from ebook_rag_api.core.config import get_settings
from ebook_rag_api.services.text import (
    contains_normalized_phrase,
    extract_anchor_terms,
    extract_constraint_terms,
    normalized_token_sequence,
    tokenize_terms,
)


class Reranker(Protocol):
    def score(self, query: str, passages: list[str]) -> list[float]:
        ...


class TokenOverlapReranker:
    def score(self, query: str, passages: list[str]) -> list[float]:
        query_terms = tokenize_terms(query, drop_stopwords=True)
        query_anchor_terms = extract_anchor_terms(query)
        query_constraint_terms = extract_constraint_terms(query)
        query_term_count = max(len(query_terms), 1)
        scores: list[float] = []

        for passage in passages:
            passage_terms = tokenize_terms(passage, drop_stopwords=True)
            if not passage_terms:
                scores.append(0.0)
                continue

            overlap = query_terms & passage_terms
            coverage = len(overlap) / query_term_count
            precision = len(overlap) / len(passage_terms)
            phrase_bonus = 0.15 if contains_normalized_phrase(query, passage) else 0.0
            scores.append(
                coverage * 0.4
                + precision * 0.1
                + _term_support_score(
                    focus_terms=query_anchor_terms,
                    passage_terms=passage_terms,
                    coverage_weight=0.22,
                    precision_weight=0.05,
                )
                + _term_support_score(
                    focus_terms=query_constraint_terms,
                    passage_terms=passage_terms,
                    coverage_weight=0.23,
                    precision_weight=0.05,
                )
                + _ordered_query_term_bonus(query, passage)
                + phrase_bonus
            )

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

def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _term_support_score(
    *,
    focus_terms: set[str],
    passage_terms: set[str],
    coverage_weight: float,
    precision_weight: float,
) -> float:
    if not focus_terms:
        return 0.0
    matched_terms = focus_terms & passage_terms
    if not matched_terms:
        return 0.0
    coverage = len(matched_terms) / len(focus_terms)
    precision = len(matched_terms) / len(passage_terms)
    return coverage * coverage_weight + precision * precision_weight


def _ordered_query_term_bonus(query: str, passage: str) -> float:
    query_terms = normalized_token_sequence(query, drop_stopwords=True)
    passage_terms = normalized_token_sequence(passage, drop_stopwords=True)
    if len(query_terms) < 2 or len(passage_terms) < 2:
        return 0.0

    query_pairs = {
        (left, right)
        for left, right in zip(query_terms, query_terms[1:], strict=False)
        if left != right
    }
    if not query_pairs:
        return 0.0

    passage_pairs = {
        (left, right)
        for left, right in zip(passage_terms, passage_terms[1:], strict=False)
    }
    overlap = len(query_pairs & passage_pairs)
    if overlap == 0:
        return 0.0
    return min(0.2, overlap / len(query_pairs) * 0.2)
