from ebook_rag_api.models import DocumentPage
from ebook_rag_api.services.chunking import ChunkingConfig, build_document_chunks


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
    assert chunks[0].provenance is not None
    assert 1 in chunks[0].provenance["source_page_numbers"]


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


def test_build_document_chunks_respects_configured_chunk_sizes() -> None:
    pages = [
        DocumentPage(
            page_number=1,
            raw_text="",
            normalized_text="\n\n".join(
                [
                    "Intro",
                    "Alpha " * 100,
                    "Beta " * 100,
                    "Gamma " * 100,
                ]
            ),
        )
    ]

    small_chunks = build_document_chunks(
        pages,
        config=ChunkingConfig(target_words=120, min_words=80, overlap_words=16),
    )
    large_chunks = build_document_chunks(
        pages,
        config=ChunkingConfig(target_words=280, min_words=160, overlap_words=16),
    )

    assert len(small_chunks) > len(large_chunks)


def test_build_document_chunks_records_character_span_provenance() -> None:
    page_text = "\n\n".join(
        [
            "Chapter 7",
            "First body paragraph with precise source offsets.",
            "Second body paragraph with more context.",
        ]
    )
    pages = [
        DocumentPage(
            page_number=1,
            raw_text="",
            normalized_text=page_text,
        )
    ]

    chunks = build_document_chunks(pages)

    assert chunks
    provenance = chunks[0].provenance
    assert provenance is not None
    assert provenance["char_range"] == {
        "start_page": 1,
        "start_char": 0,
        "end_page": 1,
        "end_char": len(page_text),
    }
    assert provenance["page_char_ranges"] == [
        {
            "page_number": 1,
            "start_char": 0,
            "end_char": len(page_text),
        }
    ]
    assert provenance["paragraphs"][1]["char_start"] == page_text.index(
        "First body paragraph with precise source offsets."
    )
    assert provenance["paragraphs"][1]["char_end"] == provenance["paragraphs"][1]["char_start"] + len(
        "First body paragraph with precise source offsets."
    )
