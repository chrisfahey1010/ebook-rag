from dataclasses import asdict, dataclass

from ebook_rag_api.models import DocumentChunk, DocumentPage


@dataclass(frozen=True)
class ChunkingConfig:
    target_words: int = 420
    min_words: int = 180
    overlap_words: int = 64
    max_heading_words: int = 12

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass
class PageParagraph:
    global_index: int
    page_number: int
    page_paragraph_index: int
    char_start: int
    char_end: int
    text: str
    is_heading: bool = False


def estimate_token_count(text: str) -> int:
    words = len(text.split())
    return max(1, round(words * 1.3))


def extract_page_paragraphs(
    pages: list[DocumentPage], *, config: ChunkingConfig
) -> list[PageParagraph]:
    paragraphs: list[PageParagraph] = []
    global_index = 0
    for page in sorted(pages, key=lambda item: item.page_number):
        page_paragraph_index = 0
        cursor = 0
        for block in page.normalized_text.split("\n\n"):
            block_start = page.normalized_text.find(block, cursor)
            if block_start < 0:
                block_start = cursor
            block_end = block_start + len(block)
            cursor = block_end + 2
            text = block.strip()
            if text:
                leading_whitespace = len(block) - len(block.lstrip())
                trailing_whitespace = len(block) - len(block.rstrip())
                paragraphs.append(
                    PageParagraph(
                        global_index=global_index,
                        page_number=page.page_number,
                        page_paragraph_index=page_paragraph_index,
                        char_start=block_start + leading_whitespace,
                        char_end=block_end - trailing_whitespace,
                        text=text,
                        is_heading=is_heading_block(text, max_heading_words=config.max_heading_words),
                    )
                )
                global_index += 1
                page_paragraph_index += 1
    return paragraphs


def build_document_chunks(
    pages: list[DocumentPage], *, config: ChunkingConfig | None = None
) -> list[DocumentChunk]:
    config = config or ChunkingConfig()
    paragraphs = extract_page_paragraphs(pages, config=config)
    if not paragraphs:
        return []

    chunks: list[DocumentChunk] = []
    current: list[PageParagraph] = []
    current_words = 0
    chunk_index = 0
    active_heading_path: list[str] = []

    for paragraph in paragraphs:
        paragraph_words = len(paragraph.text.split())
        if paragraph.is_heading:
            if _chunk_has_body(current):
                chunks.append(
                    _create_chunk(
                        chunk_index,
                        current,
                        active_heading_path=active_heading_path,
                    )
                )
                chunk_index += 1
                current = []
                current_words = 0
                active_heading_path = []
            if current and all(item.is_heading for item in current):
                active_heading_path = [*active_heading_path, paragraph.text]
            else:
                active_heading_path = [paragraph.text]
            current.append(paragraph)
            current_words += paragraph_words
            continue

        if (
            current
            and current_words + paragraph_words > config.target_words
            and _chunk_has_body(current)
            and current_words >= config.min_words
        ):
            chunks.append(
                _create_chunk(
                    chunk_index,
                    current,
                    active_heading_path=active_heading_path,
                )
            )
            chunk_index += 1
            current = _overlap_tail(current, overlap_words=config.overlap_words)
            current_words = sum(len(item.text.split()) for item in current)

        current.append(paragraph)
        current_words += paragraph_words

    if _chunk_has_body(current):
        chunks.append(
            _create_chunk(
                chunk_index,
                current,
                active_heading_path=active_heading_path,
            )
        )

    return chunks


def _create_chunk(
    chunk_index: int,
    paragraphs: list[PageParagraph],
    *,
    active_heading_path: list[str],
) -> DocumentChunk:
    text = "\n\n".join(item.text for item in paragraphs)
    page_numbers = sorted({item.page_number for item in paragraphs})
    first_paragraph = paragraphs[0]
    last_paragraph = paragraphs[-1]
    heading_path = _resolve_heading_path(paragraphs, active_heading_path)
    paragraphs_by_page: dict[int, list[PageParagraph]] = {}
    for paragraph in paragraphs:
        paragraphs_by_page.setdefault(paragraph.page_number, []).append(paragraph)
    return DocumentChunk(
        chunk_index=chunk_index,
        page_start=page_numbers[0],
        page_end=page_numbers[-1],
        heading=_resolve_heading(heading_path),
        text=text,
        token_estimate=estimate_token_count(text),
        provenance={
            "source_page_numbers": page_numbers,
            "paragraph_count": len(paragraphs),
            "heading_path": heading_path,
            "paragraph_range": {
                "start": first_paragraph.global_index,
                "end": last_paragraph.global_index,
            },
            "page_paragraph_range": {
                "start_page": first_paragraph.page_number,
                "start_index": first_paragraph.page_paragraph_index,
                "end_page": last_paragraph.page_number,
                "end_index": last_paragraph.page_paragraph_index,
            },
            "char_range": {
                "start_page": first_paragraph.page_number,
                "start_char": first_paragraph.char_start,
                "end_page": last_paragraph.page_number,
                "end_char": last_paragraph.char_end,
            },
            "page_char_ranges": [
                {
                    "page_number": page_number,
                    "start_char": page_paragraphs[0].char_start,
                    "end_char": page_paragraphs[-1].char_end,
                }
                for page_number, page_paragraphs in sorted(paragraphs_by_page.items())
            ],
            "paragraphs": [
                {
                    "global_index": paragraph.global_index,
                    "page_number": paragraph.page_number,
                    "page_paragraph_index": paragraph.page_paragraph_index,
                    "char_start": paragraph.char_start,
                    "char_end": paragraph.char_end,
                    "is_heading": paragraph.is_heading,
                }
                for paragraph in paragraphs
            ],
        },
    )


def _overlap_tail(
    paragraphs: list[PageParagraph], *, overlap_words: int
) -> list[PageParagraph]:
    if not paragraphs:
        return []

    tail: list[PageParagraph] = []
    word_count = 0
    for paragraph in reversed(paragraphs):
        if paragraph.is_heading:
            continue
        tail.insert(0, paragraph)
        word_count += len(paragraph.text.split())
        if word_count >= overlap_words:
            break
    return tail


def _chunk_has_body(paragraphs: list[PageParagraph]) -> bool:
    return any(not paragraph.is_heading for paragraph in paragraphs)


def _resolve_heading_path(
    paragraphs: list[PageParagraph],
    active_heading_path: list[str],
) -> list[str]:
    inline_heading_path = [paragraph.text for paragraph in paragraphs if paragraph.is_heading]
    if inline_heading_path:
        return inline_heading_path
    return list(active_heading_path)


def _resolve_heading(heading_path: list[str]) -> str | None:
    if not heading_path:
        return None
    heading_lines = [line.strip() for line in heading_path[-1].splitlines() if line.strip()]
    if not heading_lines:
        return None
    return heading_lines[-1]


def is_heading_block(text: str, *, max_heading_words: int) -> bool:
    stripped = text.strip()
    if not stripped or stripped.endswith((".", "!", "?")):
        return False

    words = stripped.split()
    if not 1 <= len(words) <= max_heading_words:
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
