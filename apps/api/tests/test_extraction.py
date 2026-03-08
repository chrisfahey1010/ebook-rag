from ebook_rag_api.services.extraction import normalize_document_pages, normalize_page_text


def test_normalize_page_text_collapses_whitespace_and_blank_lines() -> None:
    normalized = normalize_page_text("  Chapter 1 \n\n\nA   wrapped\tline  \n\n")

    assert normalized == "Chapter 1\n\nA wrapped line"


def test_normalize_page_text_preserves_heading_blocks_and_joins_soft_wraps() -> None:
    normalized = normalize_page_text(
        "Chapter 7\nThe Long Ride\ncontinued across the county line\n\nPage 7"
    )

    assert normalized == "Chapter 7\nThe Long Ride\n\ncontinued across the county line\n\nPage 7"


def test_normalize_page_text_groups_consecutive_heading_lines_into_one_block() -> None:
    normalized = normalize_page_text(
        "Part II\nThe Long Ride\n\nThe county line came into view."
    )

    assert normalized == "Part II\nThe Long Ride\n\nThe county line came into view."


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


def test_normalize_document_pages_preserves_headings_after_boundary_cleanup() -> None:
    normalized_pages = normalize_document_pages(
        [
            "Field Manual\nChapter One\nBattery checks begin here\nand continue on the next line.\n\n1",
            "Field Manual\nChapter Two\nWheel checks start here\nand continue on the next line.\n\n2",
            "Field Manual\nChapter Three\nRadio checks begin here\nand continue on the next line.\n\n3",
        ]
    )

    assert normalized_pages == [
        "Chapter One\n\nBattery checks begin here and continue on the next line.",
        "Chapter Two\n\nWheel checks start here and continue on the next line.",
        "Chapter Three\n\nRadio checks begin here and continue on the next line.",
    ]


def test_normalize_document_pages_removes_repeated_two_line_header_blocks() -> None:
    normalized_pages = normalize_document_pages(
        [
            "The Great Book\nBy Jane Example\n\nOpening body paragraph.\n\n1",
            "The Great Book\nBy Jane Example\n\nMiddle body paragraph.\n\n2",
            "The Great Book\nBy Jane Example\n\nClosing body paragraph.\n\n3",
        ]
    )

    assert normalized_pages == [
        "Opening body paragraph.",
        "Middle body paragraph.",
        "Closing body paragraph.",
    ]
