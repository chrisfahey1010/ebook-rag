import sys
import types

import pytest

from ebook_rag_api.core.config import get_settings
from ebook_rag_api.services.reranking import (
    CrossEncoderReranker,
    OpenAICompatibleReranker,
    TokenOverlapReranker,
    get_reranker,
)


def test_token_overlap_reranker_prefers_passages_with_stronger_query_coverage() -> None:
    reranker = TokenOverlapReranker()

    scores = reranker.score(
        "battery maintenance schedule",
        [
            "Maintenance notes for battery systems and charging schedule.",
            "General overview of the spacecraft cabin layout.",
        ],
    )

    assert scores[0] > scores[1]
    assert scores[0] > 0
    assert scores[1] >= 0


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self.payload


class _FakeClient:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls: list[dict] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def post(self, url: str, json: dict, headers: dict) -> _FakeResponse:
        self.calls.append({"url": url, "json": json, "headers": headers})
        return _FakeResponse(self.payload)


def test_cross_encoder_reranker_scores_pairs(monkeypatch) -> None:
    class _FakeCrossEncoder:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
            assert pairs == [
                ("battery maintenance schedule", "first passage"),
                ("battery maintenance schedule", "second passage"),
            ]
            return [2.0, -1.0]

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(CrossEncoder=_FakeCrossEncoder),
    )

    reranker = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
    scores = reranker.score(
        "battery maintenance schedule",
        ["first passage", "second passage"],
    )

    assert reranker.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert scores[0] == pytest.approx(0.880797078)
    assert scores[1] == pytest.approx(0.268941421)


def test_openai_compatible_reranker_posts_request_and_maps_scores(monkeypatch) -> None:
    fake_client = _FakeClient(
        {
            "results": [
                {"index": 1, "relevance_score": 0.25},
                {"index": 0, "relevance_score": 0.95},
            ]
        }
    )
    monkeypatch.setattr(
        "ebook_rag_api.services.reranking.httpx.Client",
        lambda timeout: fake_client,
    )

    reranker = OpenAICompatibleReranker(
        base_url="http://localhost:11434/v1",
        api_key="secret",
        model="bge-reranker-v2-m3",
        timeout_seconds=5.0,
    )
    scores = reranker.score(
        "battery maintenance schedule",
        ["first passage", "second passage"],
    )

    assert scores == [0.95, 0.25]
    assert fake_client.calls[0]["url"] == "http://localhost:11434/v1/rerank"
    assert fake_client.calls[0]["json"] == {
        "model": "bge-reranker-v2-m3",
        "query": "battery maintenance schedule",
        "documents": ["first passage", "second passage"],
        "top_n": 2,
        "return_documents": False,
    }
    assert fake_client.calls[0]["headers"]["Authorization"] == "Bearer secret"


def test_get_reranker_uses_environment_configuration(monkeypatch) -> None:
    monkeypatch.setenv("RERANKER_PROVIDER", "openai_compatible")
    monkeypatch.setenv("RERANKER_MODEL", "bge-reranker-v2-m3")
    monkeypatch.setenv("RERANKER_BASE_URL", "http://localhost:9999/v1")
    monkeypatch.setenv("RERANKER_API_KEY", "provider-key")
    monkeypatch.setenv("RERANKER_TIMEOUT_SECONDS", "12.5")

    get_settings.cache_clear()
    get_reranker.cache_clear()
    reranker = get_reranker()

    assert isinstance(reranker, OpenAICompatibleReranker)
    assert reranker.model == "bge-reranker-v2-m3"
    assert reranker.base_url == "http://localhost:9999/v1"
    assert reranker.api_key == "provider-key"
    assert reranker.timeout_seconds == 12.5

    get_reranker.cache_clear()
    get_settings.cache_clear()
