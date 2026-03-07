from dataclasses import dataclass

from ebook_rag_api.models import DocumentChunk, DocumentPage

TARGET_CHUNK_WORDS = 420
MIN_CHUNK_WORDS = 180
OVERLAP_CHUNK_WORDS = 64
MAX_HEADING_WORDS = 12


@dataclass
class PageParagraph:
    page_number: int
    text: str
    is_heading: bool = False


def estimate_token_count(text: str) -> int:
    words = len(text.split())
    return max(1, round(words * 1.3))


def extract_page_paragraphs(pages: list[DocumentPage]) -> list[PageParagraph]:
    paragraphs: list[PageParagraph] = []
    for page in sorted(pages, key=lambda item: item.page_number):
        for block in page.normalized_text.split("\n\n"):
            text = block.strip()
            if text:
                paragraphs.append(
                    PageParagraph(
                        page_number=page.page_number,
                        text=text,
                        is_heading=is_heading_block(text),
                    )
                )
    return paragraphs


def build_document_chunks(pages: list[DocumentPage]) -> list[DocumentChunk]:
    paragraphs = extract_page_paragraphs(pages)
    if not paragraphs:
        return []

    chunks: list[DocumentChunk] = []
    current: list[PageParagraph] = []
    current_words = 0
    chunk_index = 0
    active_heading: str | None = None

    for paragraph in paragraphs:
        paragraph_words = len(paragraph.text.split())
        if paragraph.is_heading:
            if _chunk_has_body(current):
                chunks.append(_create_chunk(chunk_index, current, active_heading))
                chunk_index += 1
                current = []
                current_words = 0
            active_heading = paragraph.text
            current.append(paragraph)
            current_words += paragraph_words
            continue

        if (
            current
            and current_words + paragraph_words > TARGET_CHUNK_WORDS
            and _chunk_has_body(current)
            and current_words >= MIN_CHUNK_WORDS
        ):
            chunks.append(_create_chunk(chunk_index, current, active_heading))
            chunk_index += 1
            current = _overlap_tail(current)
            current_words = sum(len(item.text.split()) for item in current)

        current.append(paragraph)
        current_words += paragraph_words

    if _chunk_has_body(current):
        chunks.append(_create_chunk(chunk_index, current, active_heading))

    return chunks


def _create_chunk(
    chunk_index: int,
    paragraphs: list[PageParagraph],
    active_heading: str | None,
) -> DocumentChunk:
    text = "\n\n".join(item.text for item in paragraphs)
    return DocumentChunk(
        chunk_index=chunk_index,
        page_start=min(item.page_number for item in paragraphs),
        page_end=max(item.page_number for item in paragraphs),
        heading=_resolve_heading(paragraphs, active_heading),
        text=text,
        token_estimate=estimate_token_count(text),
    )


def _overlap_tail(paragraphs: list[PageParagraph]) -> list[PageParagraph]:
    if not paragraphs:
        return []

    tail: list[PageParagraph] = []
    word_count = 0
    for paragraph in reversed(paragraphs):
        if paragraph.is_heading:
            continue
        tail.insert(0, paragraph)
        word_count += len(paragraph.text.split())
        if word_count >= OVERLAP_CHUNK_WORDS:
            break
    return tail


def _chunk_has_body(paragraphs: list[PageParagraph]) -> bool:
    return any(not paragraph.is_heading for paragraph in paragraphs)


def _resolve_heading(paragraphs: list[PageParagraph], active_heading: str | None) -> str | None:
    for paragraph in paragraphs:
        if paragraph.is_heading:
            return paragraph.text
    return active_heading


def is_heading_block(text: str) -> bool:
    stripped = text.strip()
    if not stripped or stripped.endswith((".", "!", "?")):
        return False

    words = stripped.split()
    if not 1 <= len(words) <= MAX_HEADING_WORDS:
        return False

    alpha_words = [word for word in words if any(character.isalpha() for character in word)]
    if not alpha_words:
        return False

    title_like_words = sum(
        1
        for word in alpha_words
        if word[:1].isupper() or word.isupper() or any(character.isdigit() for character in word)
    )
    lowercase_words = sum(1 for word in alpha_words if word.islower())
    if lowercase_words == len(alpha_words):
        return False

    return title_like_words / len(alpha_words) >= 0.6
