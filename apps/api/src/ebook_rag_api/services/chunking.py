from dataclasses import dataclass

from ebook_rag_api.models import DocumentChunk, DocumentPage

TARGET_CHUNK_WORDS = 220
OVERLAP_CHUNK_WORDS = 35


@dataclass
class PageParagraph:
    page_number: int
    text: str


def estimate_token_count(text: str) -> int:
    words = len(text.split())
    return max(1, round(words * 1.3))


def extract_page_paragraphs(pages: list[DocumentPage]) -> list[PageParagraph]:
    paragraphs: list[PageParagraph] = []
    for page in sorted(pages, key=lambda item: item.page_number):
        for block in page.normalized_text.split("\n\n"):
            text = block.strip()
            if text:
                paragraphs.append(PageParagraph(page_number=page.page_number, text=text))
    return paragraphs


def build_document_chunks(pages: list[DocumentPage]) -> list[DocumentChunk]:
    paragraphs = extract_page_paragraphs(pages)
    if not paragraphs:
        return []

    chunks: list[DocumentChunk] = []
    current: list[PageParagraph] = []
    current_words = 0
    chunk_index = 0

    for paragraph in paragraphs:
        paragraph_words = len(paragraph.text.split())
        if current and current_words + paragraph_words > TARGET_CHUNK_WORDS:
            chunks.append(_create_chunk(chunk_index, current))
            chunk_index += 1
            current = _overlap_tail(current)
            current_words = sum(len(item.text.split()) for item in current)

        current.append(paragraph)
        current_words += paragraph_words

    if current:
        chunks.append(_create_chunk(chunk_index, current))

    return chunks


def _create_chunk(chunk_index: int, paragraphs: list[PageParagraph]) -> DocumentChunk:
    text = "\n\n".join(item.text for item in paragraphs)
    return DocumentChunk(
        chunk_index=chunk_index,
        page_start=min(item.page_number for item in paragraphs),
        page_end=max(item.page_number for item in paragraphs),
        text=text,
        token_estimate=estimate_token_count(text),
    )


def _overlap_tail(paragraphs: list[PageParagraph]) -> list[PageParagraph]:
    if not paragraphs:
        return []

    tail: list[PageParagraph] = []
    word_count = 0
    for paragraph in reversed(paragraphs):
        tail.insert(0, paragraph)
        word_count += len(paragraph.text.split())
        if word_count >= OVERLAP_CHUNK_WORDS:
            break
    return tail
