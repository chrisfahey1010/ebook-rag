import re
from functools import lru_cache
from typing import Protocol

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

@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    settings = get_settings()
    if settings.reranker_provider == "token_overlap":
        return TokenOverlapReranker()
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
