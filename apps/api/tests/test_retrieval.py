from fastapi.testclient import TestClient
from sqlalchemy import select

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
    assert all(chunk.embedding_dimensions == 128 for chunk in chunks)
    assert all(isinstance(chunk.embedding_vector, list) for chunk in chunks)
    assert all(len(chunk.embedding_vector) == 128 for chunk in chunks)


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
