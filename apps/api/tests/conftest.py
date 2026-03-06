from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient

from ebook_rag_api.core.config import get_settings
from ebook_rag_api.db import get_engine, get_session_factory
from ebook_rag_api.main import create_app


@pytest.fixture()
def client(tmp_path, monkeypatch) -> Generator[TestClient, None, None]:
    database_path = tmp_path / "test.db"
    uploads_dir = tmp_path / "uploads"

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{database_path}")
    monkeypatch.setenv("UPLOADS_DIR", str(uploads_dir))
    monkeypatch.setenv("MAX_UPLOAD_SIZE_MB", "1")

    get_settings.cache_clear()
    get_engine.cache_clear()
    get_session_factory.cache_clear()

    app = create_app()

    with TestClient(app) as test_client:
        yield test_client

    get_engine().dispose()
    get_settings.cache_clear()
    get_engine.cache_clear()
    get_session_factory.cache_clear()
