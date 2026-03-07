from ebook_rag_api.models import DocumentPage
from ebook_rag_api.services.chunking import build_document_chunks


def test_build_document_chunks_splits_long_paragraph_sets() -> None:
    pages = [
        DocumentPage(
            page_number=1,
            raw_text="",
            normalized_text="\n\n".join(
                [
                    "Section one " + "alpha " * 90,
                    "Section two " + "beta " * 90,
                    "Section three " + "gamma " * 90,
                ]
            ),
        ),
        DocumentPage(
            page_number=2,
            raw_text="",
            normalized_text="\n\n".join(
                [
                    "Section four " + "delta " * 90,
                    "Section five " + "epsilon " * 90,
                ]
            ),
        ),
    ]

    chunks = build_document_chunks(pages)

    assert len(chunks) >= 2
    assert chunks[0].chunk_index == 0
    assert chunks[-1].chunk_index == len(chunks) - 1
    assert all(chunk.token_estimate > 0 for chunk in chunks)
    assert chunks[0].page_start == 1
    assert chunks[-1].page_end == 2
    assert max(chunk.token_estimate for chunk in chunks) > 250


def test_build_document_chunks_preserves_section_heading_metadata() -> None:
    pages = [
        DocumentPage(
            page_number=1,
            raw_text="",
            normalized_text="\n\n".join(
                [
                    "Chapter 3",
                    "Road Names and Origins " + "alpha " * 110,
                    "The Oakland section " + "beta " * 110,
                    "Epilogue",
                    "Closing remarks " + "gamma " * 80,
                ]
            ),
        )
    ]

    chunks = build_document_chunks(pages)

    assert chunks
    assert chunks[0].heading == "Chapter 3"
    assert chunks[0].text.startswith("Chapter 3")
    assert any(chunk.heading == "Epilogue" for chunk in chunks)
