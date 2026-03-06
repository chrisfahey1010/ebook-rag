from collections.abc import Callable, Generator

import fitz
import pytest
from fastapi.testclient import TestClient

from ebook_rag_api.core.config import get_settings
from ebook_rag_api.db import get_engine, get_session_factory
from ebook_rag_api.main import create_app
from ebook_rag_api.services.embeddings import get_embedding_provider


@pytest.fixture()
def pdf_factory() -> Callable[[list[str]], bytes]:
    def create_pdf(page_texts: list[str]) -> bytes:
        document = fitz.open()
        for text in page_texts:
            page = document.new_page()
            page.insert_text((72, 72), text)
        payload = document.tobytes()
        document.close()
        return payload

    return create_pdf


@pytest.fixture()
def app_environment(tmp_path, monkeypatch) -> Generator[None, None, None]:
    database_path = tmp_path / "test.db"
    uploads_dir = tmp_path / "uploads"

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{database_path}")
    monkeypatch.setenv("UPLOADS_DIR", str(uploads_dir))
    monkeypatch.setenv("MAX_UPLOAD_SIZE_MB", "1")

    get_settings.cache_clear()
    get_engine.cache_clear()
    get_session_factory.cache_clear()
    get_embedding_provider.cache_clear()

    yield

    get_engine().dispose()
    get_settings.cache_clear()
    get_engine.cache_clear()
    get_session_factory.cache_clear()
    get_embedding_provider.cache_clear()


@pytest.fixture()
def client(app_environment) -> Generator[TestClient, None, None]:
    app = create_app()

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def session_factory(app_environment):
    return get_session_factory()
