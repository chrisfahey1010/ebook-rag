import sys
import types

import pytest

from ebook_rag_api.core.config import get_settings
from ebook_rag_api.services.embeddings import (
    OpenAICompatibleEmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
    get_embedding_provider,
)


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


def test_openai_compatible_embedding_provider_posts_embeddings_request(monkeypatch) -> None:
    fake_client = _FakeClient(
        {
            "data": [
                {"embedding": [1.0, 0.0, 0.0, 0.0]},
                {"embedding": [0.0, 1.0, 0.0, 0.0]},
            ]
        }
    )
    monkeypatch.setattr(
        "ebook_rag_api.services.embeddings.httpx.Client",
        lambda timeout: fake_client,
    )

    provider = OpenAICompatibleEmbeddingProvider(
        base_url="http://localhost:11434/v1",
        api_key="secret",
        model="nomic-embed-text",
        dimensions=4,
        timeout_seconds=5.0,
    )

    embeddings = provider.embed_texts(["alpha", "beta"])

    assert embeddings == [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    assert fake_client.calls[0]["url"] == "http://localhost:11434/v1/embeddings"
    assert fake_client.calls[0]["json"] == {
        "model": "nomic-embed-text",
        "input": ["alpha", "beta"],
    }
    assert fake_client.calls[0]["headers"]["Authorization"] == "Bearer secret"


def test_sentence_transformer_embedding_provider_adapts_dimensions(monkeypatch) -> None:
    class _FakeSentenceTransformer:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        def encode(
            self,
            texts: list[str],
            *,
            convert_to_numpy: bool,
            normalize_embeddings: bool,
        ) -> list[list[float]]:
            assert convert_to_numpy is True
            assert normalize_embeddings is True
            assert texts == ["alpha", "beta"]
            return [
                [1.0, 2.0, 3.0, 4.0],
                [4.0, 3.0, 2.0, 1.0],
            ]

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=_FakeSentenceTransformer),
    )

    provider = SentenceTransformerEmbeddingProvider(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        dimensions=2,
    )
    embeddings = provider.embed_texts(["alpha", "beta"])

    assert provider.model_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert len(embeddings) == 2
    assert all(len(embedding) == 2 for embedding in embeddings)
    assert embeddings[0] == pytest.approx([0.5547001962, 0.8320502943])
    assert embeddings[1] == pytest.approx([0.8320502943, 0.5547001962])


def test_get_embedding_provider_uses_environment_configuration(monkeypatch) -> None:
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai_compatible")
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "http://localhost:9999/v1")
    monkeypatch.setenv("EMBEDDING_API_KEY", "provider-key")
    monkeypatch.setenv("EMBEDDING_TIMEOUT_SECONDS", "12.5")

    get_settings.cache_clear()
    get_embedding_provider.cache_clear()
    provider = get_embedding_provider()

    assert isinstance(provider, OpenAICompatibleEmbeddingProvider)
    assert provider.model == "text-embedding-3-small"
    assert provider.base_url == "http://localhost:9999/v1"
    assert provider.api_key == "provider-key"
    assert provider.timeout_seconds == 12.5

    get_embedding_provider.cache_clear()
    get_settings.cache_clear()


def test_openai_compatible_embedding_provider_raises_for_count_mismatch(monkeypatch) -> None:
    fake_client = _FakeClient({"data": [{"embedding": [1.0, 0.0, 0.0, 0.0]}]})
    monkeypatch.setattr(
        "ebook_rag_api.services.embeddings.httpx.Client",
        lambda timeout: fake_client,
    )

    provider = OpenAICompatibleEmbeddingProvider(
        base_url="http://localhost:11434/v1",
        api_key="secret",
        model="nomic-embed-text",
        dimensions=4,
        timeout_seconds=5.0,
    )

    with pytest.raises(RuntimeError, match="unexpected number of embeddings"):
        provider.embed_texts(["alpha", "beta"])
