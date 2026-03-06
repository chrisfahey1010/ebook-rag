from fastapi.testclient import TestClient


def test_upload_extracts_pages_and_completes_ingestion(
    client: TestClient, pdf_factory
) -> None:
    pdf_bytes = pdf_factory(["First page text", "Second page text"])
    response = client.post(
        "/api/documents/upload",
        files={"file": ("sample.pdf", pdf_bytes, "application/pdf")},
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["document"]["original_filename"] == "sample.pdf"
    assert payload["document"]["status"] == "ready"
    assert payload["document"]["page_count"] == 2
    assert payload["ingestion_status"] == "completed"
    assert payload["ingestion_error"] is None
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
        files={"file": ("fake.pdf", b"%PDF-not-really", "application/pdf")},
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["document"]["status"] == "failed"
    assert payload["ingestion_status"] == "failed"
    assert payload["ingestion_error"] == "PDF extraction failed."


def test_upload_rejects_duplicate_pdf(client: TestClient, pdf_factory) -> None:
    pdf_bytes = pdf_factory(["Duplicate text"])
    files = {"file": ("sample.pdf", pdf_bytes, "application/pdf")}

    first_response = client.post("/api/documents/upload", files=files)
    second_response = client.post("/api/documents/upload", files=files)

    assert first_response.status_code == 201
    assert second_response.status_code == 409
    assert second_response.json()["detail"] == "This PDF has already been uploaded."


def test_list_documents_returns_most_recent_first(client: TestClient, pdf_factory) -> None:
    client.post(
        "/api/documents/upload",
        files={
            "file": (
                "first.pdf",
                pdf_factory(["First document page one", "First document page two"]),
                "application/pdf",
            )
        },
    )
    client.post(
        "/api/documents/upload",
        files={
            "file": (
                "second.pdf",
                pdf_factory(["Second document page one", "Second document page two"]),
                "application/pdf",
            )
        },
    )

    response = client.get("/api/documents")

    assert response.status_code == 200
    payload = response.json()
    assert [document["original_filename"] for document in payload] == [
        "second.pdf",
        "first.pdf",
    ]


def test_get_document_returns_extracted_metadata(
    client: TestClient, pdf_factory
) -> None:
    pdf_bytes = pdf_factory(["First page text", "Second page text"])
    upload_response = client.post(
        "/api/documents/upload",
        files={"file": ("sample.pdf", pdf_bytes, "application/pdf")},
    )

    document_id = upload_response.json()["document"]["id"]
    response = client.get(f"/api/documents/{document_id}")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ready"
    assert payload["page_count"] == 2
