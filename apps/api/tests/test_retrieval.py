from fastapi.testclient import TestClient
from sqlalchemy import select

from ebook_rag_api.core.config import get_settings
from ebook_rag_api.models import DocumentChunk


def test_upload_persists_chunk_embeddings(
    client: TestClient, pdf_factory, session_factory
) -> None:
    response = client.post(
        "/api/documents/upload",
        files={
            "file": (
                "astronomy.pdf",
                pdf_factory(["Galaxies and nebulae\n\n" + "starlight " * 120]),
                "application/pdf",
            )
        },
    )

    assert response.status_code == 201
    document_id = response.json()["document"]["id"]

    with session_factory() as session:
        chunks = list(
            session.scalars(
                select(DocumentChunk)
                .where(DocumentChunk.document_id == document_id)
                .order_by(DocumentChunk.chunk_index.asc())
            )
        )

    assert chunks
    assert all(chunk.embedding_dimensions == get_settings().embedding_dimensions for chunk in chunks)
    assert all(isinstance(chunk.embedding_vector, list) for chunk in chunks)
    assert all(len(chunk.embedding_vector) == get_settings().embedding_dimensions for chunk in chunks)


def test_retrieval_search_returns_ranked_matches(
    client: TestClient, pdf_factory
) -> None:
    mars_upload = client.post(
        "/api/documents/upload",
        files={
            "file": (
                "mars.pdf",
                pdf_factory(
                    [
                        "Mars rover geology mission.\n\nThe rover studies rocks, craters, and sediment layers.",
                        "Mission log.\n\nTelemetry and maintenance details for the rover battery systems.",
                    ]
                ),
                "application/pdf",
            )
        },
    )
    ocean_upload = client.post(
        "/api/documents/upload",
        files={
            "file": (
                "ocean.pdf",
                pdf_factory(
                    [
                        "Ocean currents and marine biology.\n\nWhales migrate through cold water regions.",
                    ]
                ),
                "application/pdf",
            )
        },
    )

    assert mars_upload.status_code == 201
    assert ocean_upload.status_code == 201

    mars_document_id = mars_upload.json()["document"]["id"]
    response = client.post(
        "/api/retrieval/search",
        json={"query": "Which chunk talks about rover geology on Mars?", "top_k": 3},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["normalized_query"] == "Which chunk talks about rover geology on Mars?"
    assert payload["matches"]
    assert payload["matches"][0]["document_id"] == mars_document_id
    assert "rover" in payload["matches"][0]["text"].lower()
    assert "dense_score" in payload["matches"][0]
    assert "lexical_score" in payload["matches"][0]
    assert "hybrid_score" in payload["matches"][0]
    assert "rerank_score" in payload["matches"][0]
    assert payload["matches"][0]["score"] >= payload["matches"][-1]["score"]


def test_retrieval_search_can_filter_to_a_document(
    client: TestClient, pdf_factory
) -> None:
    alpha_upload = client.post(
        "/api/documents/upload",
        files={
            "file": (
                "alpha.pdf",
                pdf_factory(["Alpha handbook\n\nshared keyword uniquealpha"]),
                "application/pdf",
            )
        },
    )
    beta_upload = client.post(
        "/api/documents/upload",
        files={
            "file": (
                "beta.pdf",
                pdf_factory(["Beta handbook\n\nshared keyword uniquebeta"]),
                "application/pdf",
            )
        },
    )

    alpha_document_id = alpha_upload.json()["document"]["id"]
    beta_document_id = beta_upload.json()["document"]["id"]

    response = client.post(
        "/api/retrieval/search",
        json={
            "query": "shared keyword uniquebeta",
            "document_id": alpha_document_id,
            "top_k": 5,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["matches"]
    assert all(match["document_id"] == alpha_document_id for match in payload["matches"])
    assert all(match["document_id"] != beta_document_id for match in payload["matches"])


def test_debug_retrieve_returns_ranked_candidates_for_workspace_inspection(
    client: TestClient, pdf_factory
) -> None:
    upload_response = client.post(
        "/api/documents/upload",
        files={
            "file": (
                "manual.pdf",
                pdf_factory(
                    [
                        "Operations\n\nInspect the seals before the vehicle leaves the hangar.",
                        "Maintenance\n\nRecharge the auxiliary battery after each field day.",
                    ]
                ),
                "application/pdf",
            )
        },
    )

    assert upload_response.status_code == 201
    document_id = upload_response.json()["document"]["id"]

    response = client.post(
        "/api/debug/retrieve",
        json={
            "query": "What should happen before the vehicle leaves the hangar?",
            "document_id": document_id,
            "top_k": 4,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["normalized_query"] == "What should happen before the vehicle leaves the hangar?"
    assert payload["matches"]
    assert payload["matches"][0]["document_id"] == document_id
    assert "inspect the seals" in payload["matches"][0]["text"].lower()
    assert "dense_score" in payload["matches"][0]
    assert "lexical_score" in payload["matches"][0]
    assert "hybrid_score" in payload["matches"][0]
    assert "rerank_score" in payload["matches"][0]
    assert payload["matches"][0]["score"] >= payload["matches"][-1]["score"]


def test_debug_document_chunks_returns_persisted_chunk_metadata(
    client: TestClient, pdf_factory
) -> None:
    upload_response = client.post(
        "/api/documents/upload",
        files={
            "file": (
                "manual.pdf",
                pdf_factory(
                    [
                        "Operations\n\nInspect the seals before the vehicle leaves the hangar.",
                        "Maintenance\n\nRecharge the auxiliary battery after each field day.",
                    ]
                ),
                "application/pdf",
            )
        },
    )

    assert upload_response.status_code == 201
    document_id = upload_response.json()["document"]["id"]

    response = client.get(f"/api/debug/documents/{document_id}/chunks")

    assert response.status_code == 200
    payload = response.json()
    assert payload["document_id"] == document_id
    assert payload["document_filename"] == "manual.pdf"
    assert payload["chunk_count"] == len(payload["chunks"])
    assert payload["chunk_count"] >= 1
    assert payload["chunks"]
    assert payload["chunks"][0]["chunk_index"] == 0
    assert payload["chunks"][0]["page_start"] == 1
    assert payload["chunks"][0]["token_estimate"] > 0
    assert "inspect the seals" in payload["chunks"][0]["text"].lower()


def test_retrieval_prefers_exact_anchor_terms_for_named_entity_facts(
    client: TestClient, pdf_factory
) -> None:
    exact_upload = client.post(
        "/api/documents/upload",
        files={
            "file": (
                "history.pdf",
                pdf_factory(
                    [
                        (
                            "Preface\n\nCarey McWilliams asked Hunter S. Thompson "
                            "to write the original article on motorcycle gangs. "
                            "The article appeared in April 1965."
                        )
                    ]
                ),
                "application/pdf",
            )
        },
    )
    adjacent_upload = client.post(
        "/api/documents/upload",
        files={
            "file": (
                "overview.pdf",
                pdf_factory(
                    [
                        (
                            "Magazine history\n\nAn editor asked Thompson to report "
                            "on motorcycle gangs for a national publication before "
                            "the book project expanded."
                        )
                    ]
                ),
                "application/pdf",
            )
        },
    )

    assert exact_upload.status_code == 201
    assert adjacent_upload.status_code == 201
    exact_document_id = exact_upload.json()["document"]["id"]

    response = client.post(
        "/api/retrieval/search",
        json={
            "query": "Who asked Hunter S. Thompson to write the original article on motorcycle gangs and when did it appear?",
            "top_k": 3,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["matches"]
    assert payload["matches"][0]["document_id"] == exact_document_id
    assert "carey mcwilliams" in payload["matches"][0]["text"].lower()
    assert "1965" in payload["matches"][0]["text"]


def test_debug_rerank_returns_ranked_scores_for_explicit_passages(
    client: TestClient,
) -> None:
    response = client.post(
        "/api/debug/rerank",
        json={
            "query": "What should happen before launch?",
            "passages": [
                "Archive the telemetry after landing.",
                "Inspect the cooling lines before launch.",
                "Store spare seals in the side locker.",
            ],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["reranker"]
    assert len(payload["results"]) == 3
    assert payload["results"][0]["index"] == 1
    assert "before launch" in payload["results"][0]["text"].lower()
    assert payload["results"][0]["rerank_score"] >= payload["results"][-1]["rerank_score"]


def test_retrieval_falls_back_to_token_overlap_when_reranker_fails(
    client: TestClient, pdf_factory, monkeypatch
) -> None:
    upload = client.post(
        "/api/documents/upload",
        files={
            "file": (
                "manual.pdf",
                pdf_factory(
                    [
                        "Maintenance\n\nInspect the cooling lines before launch.",
                        "Operations\n\nArchive the telemetry after landing.",
                    ]
                ),
                "application/pdf",
            )
        },
    )
    assert upload.status_code == 201

    class _BrokenReranker:
        def score(self, query: str, passages: list[str]) -> list[float]:
            raise RuntimeError("reranker backend unavailable")

    monkeypatch.setattr(
        "ebook_rag_api.services.retrieval.get_reranker",
        lambda: _BrokenReranker(),
    )

    response = client.post(
        "/api/retrieval/search",
        json={"query": "What should happen before launch?", "top_k": 3},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["matches"]
    assert "inspect the cooling lines" in payload["matches"][0]["text"].lower()
    assert payload["matches"][0]["rerank_score"] > 0


def test_retrieval_can_surface_lexical_matches_when_dense_scores_are_flat(
    client: TestClient, pdf_factory, monkeypatch
) -> None:
    upload = client.post(
        "/api/documents/upload",
        files={
            "file": (
                "catalog.pdf",
                pdf_factory(
                    [
                        "Parts list\n\nThe xenobotany appendix covers chlorophyll sampling procedures.",
                        "Field notes\n\nGeneral rover maintenance instructions for filters and seals.",
                    ]
                ),
                "application/pdf",
            )
        },
    )
    assert upload.status_code == 201

    class _ZeroEmbeddingProvider:
        dimensions = get_settings().embedding_dimensions

        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            return [[0.0] * self.dimensions for _ in texts]

    monkeypatch.setattr(
        "ebook_rag_api.services.retrieval.get_embedding_provider",
        lambda: _ZeroEmbeddingProvider(),
    )

    response = client.post(
        "/api/retrieval/search",
        json={"query": "chlorophyll sampling procedures", "top_k": 3},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["matches"]
    assert "chlorophyll sampling procedures" in payload["matches"][0]["text"].lower()
    assert payload["matches"][0]["dense_score"] == 0
    assert payload["matches"][0]["lexical_score"] > 0
    assert payload["matches"][0]["hybrid_score"] > 0


def test_retrieval_lexical_matching_handles_simple_inflection_changes(
    client: TestClient, pdf_factory, monkeypatch
) -> None:
    upload = client.post(
        "/api/documents/upload",
        files={
            "file": (
                "landing.pdf",
                pdf_factory(
                    [
                        "Landing checklist\n\nDeploy the parachute after atmospheric entry.",
                        "Maintenance\n\nInspect the landing struts after touchdown.",
                    ]
                ),
                "application/pdf",
            )
        },
    )
    assert upload.status_code == 201

    class _ZeroEmbeddingProvider:
        dimensions = get_settings().embedding_dimensions

        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            return [[0.0] * self.dimensions for _ in texts]

    monkeypatch.setattr(
        "ebook_rag_api.services.retrieval.get_embedding_provider",
        lambda: _ZeroEmbeddingProvider(),
    )

    response = client.post(
        "/api/retrieval/search",
        json={"query": "When is the parachute deployed?", "top_k": 3},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["matches"]
    assert payload["matches"][0]["page_start"] == 1
    assert "deploy the parachute" in payload["matches"][0]["text"].lower()
    assert payload["matches"][0]["lexical_score"] > 0
