from fastapi.testclient import TestClient

from ebook_rag_api.main import app


def test_healthcheck_returns_service_status() -> None:
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"

