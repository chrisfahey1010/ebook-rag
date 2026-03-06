from fastapi.testclient import TestClient


def test_upload_registers_pdf_and_creates_ingestion_job(client: TestClient) -> None:
    response = client.post(
        "/api/documents/upload",
        files={"file": ("sample.pdf", b"%PDF-1.4\nfake pdf bytes", "application/pdf")},
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["document"]["original_filename"] == "sample.pdf"
    assert payload["document"]["status"] == "uploaded"
    assert payload["ingestion_status"] == "queued"
    assert len(payload["document"]["sha256"]) == 64


def test_upload_rejects_non_pdf_file(client: TestClient) -> None:
    response = client.post(
        "/api/documents/upload",
        files={"file": ("notes.txt", b"plain text", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Only PDF uploads are supported."


def test_upload_rejects_invalid_pdf_bytes(client: TestClient) -> None:
    response = client.post(
        "/api/documents/upload",
        files={"file": ("fake.pdf", b"not really a pdf", "application/pdf")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Uploaded file does not appear to be a valid PDF."


def test_upload_rejects_duplicate_pdf(client: TestClient) -> None:
    files = {"file": ("sample.pdf", b"%PDF-1.4\nsame bytes", "application/pdf")}

    first_response = client.post("/api/documents/upload", files=files)
    second_response = client.post("/api/documents/upload", files=files)

    assert first_response.status_code == 201
    assert second_response.status_code == 409
    assert second_response.json()["detail"] == "This PDF has already been uploaded."


def test_list_documents_returns_most_recent_first(client: TestClient) -> None:
    client.post(
        "/api/documents/upload",
        files={"file": ("first.pdf", b"%PDF-1.4\nfirst", "application/pdf")},
    )
    client.post(
        "/api/documents/upload",
        files={"file": ("second.pdf", b"%PDF-1.4\nsecond", "application/pdf")},
    )

    response = client.get("/api/documents")

    assert response.status_code == 200
    payload = response.json()
    assert [document["original_filename"] for document in payload] == [
        "second.pdf",
        "first.pdf",
    ]
