from fastapi.testclient import TestClient


def test_qa_answer_returns_grounded_answer_with_citations(
    client: TestClient, pdf_factory
) -> None:
    upload_response = client.post(
        "/api/documents/upload",
        files={
            "file": (
                "manual.pdf",
                pdf_factory(
                    [
                        "Battery care\n\nCharge the rover battery before long field sessions.",
                        "Maintenance\n\nInspect the rover wheels after each rocky traverse.",
                    ]
                ),
                "application/pdf",
            )
        },
    )

    assert upload_response.status_code == 201
    document_id = upload_response.json()["document"]["id"]

    response = client.post(
        "/api/qa/ask",
        json={
            "question": "What should happen before long field sessions?",
            "document_id": document_id,
            "top_k": 4,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["supported"] is True
    assert "charge the rover battery" in payload["answer"].lower()
    assert payload["citations"]
    assert payload["citations"][0]["document_id"] == document_id
    assert payload["citations"][0]["page_start"] == 1
    assert "charge the rover battery" in payload["citations"][0]["text"].lower()
    assert payload["retrieved_chunk_count"] >= 1


def test_qa_answer_returns_unsupported_when_evidence_is_missing(
    client: TestClient, pdf_factory
) -> None:
    upload_response = client.post(
        "/api/documents/upload",
        files={
            "file": (
                "astronomy.pdf",
                pdf_factory(
                    [
                        "Stars\n\nStars emit light because of nuclear fusion in their cores.",
                    ]
                ),
                "application/pdf",
            )
        },
    )

    assert upload_response.status_code == 201
    document_id = upload_response.json()["document"]["id"]

    response = client.post(
        "/api/qa/ask",
        json={
            "question": "What does the book say about whale migration routes?",
            "document_id": document_id,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["supported"] is False
    assert "could not find enough support" in payload["answer"].lower()
    assert payload["citations"] == []
