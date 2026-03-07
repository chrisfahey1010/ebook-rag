from fastapi.testclient import TestClient
from pathlib import Path
from sqlalchemy import func, select

from ebook_rag_api.models import Document, DocumentChunk, DocumentPage, IngestionJob


def test_upload_extracts_pages_and_completes_ingestion(
    client: TestClient, pdf_factory, session_factory
) -> None:
    pdf_bytes = pdf_factory(
        [
            "First page heading\n\n" + "alpha " * 130,
            "Second page heading\n\n" + "beta " * 130,
        ]
    )
    response = client.post(
        "/api/documents/upload",
        files={"file": ("sample.pdf", pdf_bytes, "application/pdf")},
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["document"]["original_filename"] == "sample.pdf"
    assert payload["document"]["status"] == "ready"
    assert payload["document"]["page_count"] == 2
    assert payload["document"]["chunk_count"] >= 1
    assert payload["ingestion_status"] == "completed"
    assert payload["ingestion_error"] is None
    assert len(payload["document"]["sha256"]) == 64

    with session_factory() as session:
        chunks = list(
            session.scalars(
                select(DocumentChunk)
                .where(DocumentChunk.document_id == payload["document"]["id"])
                .order_by(DocumentChunk.chunk_index.asc())
            )
        )

    assert len(chunks) == payload["document"]["chunk_count"]
    assert chunks[0].page_start == 1
    assert chunks[0].page_end >= 1
    assert all(chunk.token_estimate > 0 for chunk in chunks)
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))


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


def test_upload_marks_ingestion_failed_when_embedding_generation_breaks(
    client: TestClient, pdf_factory, monkeypatch
) -> None:
    monkeypatch.setattr(
        "ebook_rag_api.services.extraction.get_embedding_provider",
        lambda: (_ for _ in ()).throw(RuntimeError("embedding backend unavailable")),
    )

    response = client.post(
        "/api/documents/upload",
        files={
            "file": (
                "sample.pdf",
                pdf_factory(["First page heading\n\n" + "alpha " * 130]),
                "application/pdf",
            )
        },
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["document"]["status"] == "failed"
    assert payload["ingestion_status"] == "failed"
    assert payload["ingestion_error"] == "Embedding generation failed."


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
    pdf_bytes = pdf_factory(
        [
            "Intro paragraph\n\n" + "gamma " * 140,
            "Follow-up paragraph\n\n" + "delta " * 140,
        ]
    )
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
    assert payload["chunk_count"] >= 1


def test_delete_document_removes_database_records_and_uploaded_file(
    client: TestClient, pdf_factory, session_factory
) -> None:
    upload_response = client.post(
        "/api/documents/upload",
        files={
            "file": (
                "sample.pdf",
                pdf_factory(
                    [
                        "Intro paragraph\n\n" + "gamma " * 140,
                        "Follow-up paragraph\n\n" + "delta " * 140,
                    ]
                ),
                "application/pdf",
            )
        },
    )

    assert upload_response.status_code == 201
    payload = upload_response.json()
    document_id = payload["document"]["id"]
    stored_path = payload["document"]["filename"]

    with session_factory() as session:
        document = session.scalar(select(Document).where(Document.id == document_id))
        assert document is not None
        file_path = document.file_path

    response = client.delete(f"/api/documents/{document_id}")

    assert response.status_code == 204
    assert response.content == b""

    with session_factory() as session:
        assert session.scalar(select(Document).where(Document.id == document_id)) is None
        assert session.scalar(
            select(func.count()).select_from(DocumentChunk).where(DocumentChunk.document_id == document_id)
        ) == 0
        assert session.scalar(
            select(func.count()).select_from(DocumentPage).where(DocumentPage.document_id == document_id)
        ) == 0
        assert session.scalar(
            select(func.count()).select_from(IngestionJob).where(IngestionJob.document_id == document_id)
        ) == 0

    assert not Path(file_path).exists()
    assert stored_path.endswith(".pdf")


def test_delete_document_returns_not_found_for_unknown_id(client: TestClient) -> None:
    response = client.delete("/api/documents/missing-document")

    assert response.status_code == 404
    assert response.json()["detail"] == "Document not found."
