from fastapi.testclient import TestClient

from ebook_rag_api.services.qa import (
    RetrievedChunkContext,
    assemble_answer_contexts,
    select_evidence_citations,
)


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
    assert "lexical_score" in payload["trace"]["retrieved_chunks"][0]
    assert "hybrid_score" in payload["trace"]["retrieved_chunks"][0]
    assert "rerank_score" in payload["trace"]["retrieved_chunks"][0]
    assert "Question: What should happen before ignition?" in payload["trace"]["prompt_snapshot"]
    assert "Inspect the heat shield before ignition." in payload["trace"]["prompt_snapshot"]
    assert payload["trace"]["selected_contexts"][0]["chunk_id"] == payload["citations"][0]["chunk_id"]
    assert payload["trace"]["cited_contexts"][0]["chunk_id"] == payload["citations"][0]["chunk_id"]
    assert payload["trace"]["timings"]["total_ms"] >= 0


def test_qa_citations_follow_the_answer_evidence_instead_of_all_selected_context(
    client: TestClient, pdf_factory
) -> None:
    upload_response = client.post(
        "/api/documents/upload",
        files={
            "file": (
                "manual.pdf",
                pdf_factory(
                    [
                        "Maintenance\n\nInspect the rover wheels after each rocky traverse.",
                        "Storage\n\nStore replacement batteries in a dry compartment away from dust.",
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
            "question": "What should be inspected after rocky traverses?",
            "document_id": document_id,
            "top_k": 4,
            "include_trace": True,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["supported"] is True
    assert "inspect the rover wheels" in payload["answer"].lower()
    assert len(payload["citations"]) == 1
    assert payload["citations"][0]["page_start"] == 1
    assert payload["trace"]["selected_contexts"]
    assert payload["trace"]["cited_contexts"]
    assert len(payload["trace"]["cited_contexts"]) == 1
    assert payload["trace"]["cited_contexts"][0]["page_start"] == 1


def test_assemble_answer_contexts_skips_irrelevant_adjacent_chunks() -> None:
    contexts = [
        RetrievedChunkContext(
            chunk_id="chunk-1",
            document_id="doc-1",
            document_title="Manual",
            document_filename="manual.pdf",
            chunk_index=1,
            page_start=1,
            page_end=1,
            text="Departure checklist. Inspect the cooling lines before launch.",
            token_estimate=10,
            score=0.95,
            rerank_score=0.95,
        ),
        RetrievedChunkContext(
            chunk_id="chunk-2",
            document_id="doc-1",
            document_title="Manual",
            document_filename="manual.pdf",
            chunk_index=2,
            page_start=1,
            page_end=1,
            text="Archive telemetry records monthly and review spare inventory counts.",
            token_estimate=10,
            score=0.99,
            rerank_score=0.99,
        ),
    ]

    selected = assemble_answer_contexts(
        question="What should happen to the cooling lines before launch?",
        contexts=contexts,
    )

    assert [context.chunk_id for context in selected] == ["chunk-1"]


def test_select_evidence_citations_returns_sentence_level_excerpt() -> None:
    context = RetrievedChunkContext(
        chunk_id="chunk-1",
        document_id="doc-1",
        document_title="Manual",
        document_filename="manual.pdf",
        chunk_index=1,
        page_start=4,
        page_end=4,
        text=(
            "General maintenance covers filter checks and storage logs. "
            "Inspect the cooling lines before launch."
        ),
        token_estimate=18,
        score=0.9,
        rerank_score=0.9,
    )

    citations = select_evidence_citations(
        answer_text="Inspect the cooling lines before launch.",
        contexts=[context],
        primary_context=context,
    )

    assert len(citations) == 1
    assert citations[0].text == "Inspect the cooling lines before launch."


def test_select_evidence_citations_filters_weak_overlap_contexts() -> None:
    primary_context = RetrievedChunkContext(
        chunk_id="chunk-1",
        document_id="doc-1",
        document_title="Manual",
        document_filename="manual.pdf",
        chunk_index=1,
        page_start=2,
        page_end=2,
        text="Pause the approach if the distance reading jumps or the starboard latch fails.",
        token_estimate=16,
        score=0.95,
        rerank_score=0.95,
    )
    weak_context = RetrievedChunkContext(
        chunk_id="chunk-2",
        document_id="doc-1",
        document_title="Manual",
        document_filename="manual.pdf",
        chunk_index=2,
        page_start=3,
        page_end=3,
        text="Inspect the hatch before fueling and confirm the distance marker is visible.",
        token_estimate=14,
        score=0.8,
        rerank_score=0.8,
    )

    citations = select_evidence_citations(
        answer_text="Pause the approach if the distance reading jumps.",
        contexts=[primary_context, weak_context],
        primary_context=primary_context,
    )

    assert [citation.chunk_id for citation in citations] == ["chunk-1"]
