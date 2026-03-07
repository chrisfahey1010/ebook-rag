from ebook_rag_api.services.extraction import normalize_document_pages, normalize_page_text


def test_normalize_page_text_collapses_whitespace_and_blank_lines() -> None:
    normalized = normalize_page_text("  Chapter 1 \n\n\nA   wrapped\tline  \n\n")

    assert normalized == "Chapter 1\n\nA wrapped line"


def test_normalize_document_pages_removes_repeated_headers_footers_and_page_numbers() -> None:
    normalized_pages = normalize_document_pages(
        [
            "The Great Book\nChapter One\n\nFirst body paragraph.\n\n1",
            "The Great Book\nChapter One\n\nSecond body paragraph.\n\n2",
            "The Great Book\nChapter One\n\nThird body paragraph.\n\n3",
        ]
    )

    assert normalized_pages == [
        "First body paragraph.",
        "Second body paragraph.",
        "Third body paragraph.",
    ]


def test_normalize_document_pages_keeps_unique_boundary_content() -> None:
    normalized_pages = normalize_document_pages(
        [
            "Opening Scene\n\nUnique first page detail.\n\nPage 1",
            "Another Scene\n\nUnique second page detail.\n\nPage 2",
        ]
    )

    assert normalized_pages == [
        "Opening Scene\n\nUnique first page detail.",
        "Another Scene\n\nUnique second page detail.",
    ]
