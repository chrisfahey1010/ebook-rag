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


def test_qa_answer_can_include_trace_payload(
    client: TestClient, pdf_factory
) -> None:
    upload_response = client.post(
        "/api/documents/upload",
        files={
            "file": (
                "operations.pdf",
                pdf_factory(
                    [
                        "Launch checklist\n\nInspect the heat shield before ignition.",
                        "Landing checklist\n\nDeploy the parachute after atmospheric entry.",
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
            "question": "What should happen before ignition?",
            "document_id": document_id,
            "top_k": 4,
            "include_trace": True,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["trace"] is not None
    assert payload["trace"]["answer_provider"] == "ExtractiveAnswerProvider"
    assert payload["trace"]["retrieved_chunks"]
    assert payload["trace"]["selected_contexts"]
    assert "dense_score" in payload["trace"]["retrieved_chunks"][0]
    assert "rerank_score" in payload["trace"]["retrieved_chunks"][0]
    assert "Question: What should happen before ignition?" in payload["trace"]["prompt_snapshot"]
    assert "Inspect the heat shield before ignition." in payload["trace"]["prompt_snapshot"]
    assert payload["trace"]["selected_contexts"][0]["chunk_id"] == payload["citations"][0]["chunk_id"]
    assert payload["trace"]["timings"]["total_ms"] >= 0
