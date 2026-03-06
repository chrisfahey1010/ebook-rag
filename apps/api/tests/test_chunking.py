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
