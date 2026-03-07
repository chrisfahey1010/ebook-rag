import re
from datetime import UTC, datetime
from pathlib import Path

import fitz
from sqlalchemy.orm import Session

from ebook_rag_api.models import Document, DocumentPage, IngestionJob
from ebook_rag_api.core.config import get_settings
from ebook_rag_api.services.embeddings import get_embedding_provider
from ebook_rag_api.services.chunking import (
    ChunkingConfig,
    build_document_chunks,
    is_heading_block,
)

WHITESPACE_RE = re.compile(r"[ \t]+")
BLANK_LINE_RE = re.compile(r"\n{3,}")
NON_ALNUM_RE = re.compile(r"[^a-z0-9# ]+")
ROMAN_NUMERAL_RE = re.compile(r"^(?=[ivxlcdm]+\Z)[ivxlcdm]+$", re.IGNORECASE)
PAGE_NUMBER_RE = re.compile(r"^(?:page\s+)?\d+$", re.IGNORECASE)


def _normalize_line(line: str) -> str:
    return WHITESPACE_RE.sub(" ", line).strip()


def normalize_page_text(raw_text: str, *, max_heading_words: int = 12) -> str:
    lines = [_normalize_line(line) for line in raw_text.splitlines()]
    return _normalize_page_lines(lines, max_heading_words=max_heading_words)


def normalize_document_pages(
    raw_pages: list[str], *, max_heading_words: int = 12
) -> list[str]:
    normalized_lines_by_page = [
        [_normalize_line(line) for line in raw_text.splitlines()]
        for raw_text in raw_pages
    ]
    repeated_header_signatures = _find_repeated_boundary_signatures(
        normalized_lines_by_page,
        take_first=True,
    )
    repeated_footer_signatures = _find_repeated_boundary_signatures(
        normalized_lines_by_page,
        take_first=False,
    )

    normalized_pages: list[str] = []
    for lines in normalized_lines_by_page:
        cleaned_lines = _strip_repeated_boundary_noise(
            lines,
            repeated_header_signatures=repeated_header_signatures,
            repeated_footer_signatures=repeated_footer_signatures,
        )
        normalized_pages.append(
            _normalize_page_lines(
                cleaned_lines,
                max_heading_words=max_heading_words,
            )
        )
    return normalized_pages


def _normalize_page_lines(lines: list[str], *, max_heading_words: int) -> str:
    paragraphs: list[str] = []
    current_lines: list[str] = []

    def flush_current() -> None:
        if not current_lines:
            return
        paragraphs.append(" ".join(current_lines))
        current_lines.clear()

    for line in lines:
        if not line:
            flush_current()
            continue

        if _is_heading_line(line, max_heading_words=max_heading_words):
            flush_current()
            paragraphs.append(line)
            continue

        current_lines.append(line)

    flush_current()
    normalized = "\n\n".join(paragraphs)
    normalized = BLANK_LINE_RE.sub("\n\n", normalized).strip()
    return normalized


def _is_heading_line(line: str, *, max_heading_words: int) -> bool:
    return is_heading_block(line, max_heading_words=max_heading_words)


def _find_repeated_boundary_signatures(
    normalized_lines_by_page: list[list[str]],
    *,
    take_first: bool,
) -> set[str]:
    signature_counts: dict[str, int] = {}
    for lines in normalized_lines_by_page:
        for boundary_line in _boundary_lines(lines, take_first=take_first):
            signature = _boundary_signature(boundary_line)
            if signature is None:
                continue
            signature_counts[signature] = signature_counts.get(signature, 0) + 1

    minimum_repetitions = 3
    return {
        signature
        for signature, count in signature_counts.items()
        if count >= minimum_repetitions
    }


def _strip_repeated_boundary_noise(
    lines: list[str],
    *,
    repeated_header_signatures: set[str],
    repeated_footer_signatures: set[str],
) -> list[str]:
    cleaned_lines = list(lines)

    while True:
        first_index = _first_non_empty_index(cleaned_lines)
        if first_index is None:
            break
        first_line = cleaned_lines[first_index]
        signature = _boundary_signature(first_line)
        if signature not in repeated_header_signatures and not _is_page_number_line(first_line):
            break
        cleaned_lines[first_index] = ""

    while True:
        last_index = _last_non_empty_index(cleaned_lines)
        if last_index is None:
            break
        last_line = cleaned_lines[last_index]
        signature = _boundary_signature(last_line)
        if signature not in repeated_footer_signatures and not _is_page_number_line(last_line):
            break
        cleaned_lines[last_index] = ""

    return cleaned_lines


def _boundary_lines(lines: list[str], *, take_first: bool, limit: int = 2) -> list[str]:
    boundary_lines: list[str] = []
    indexes = range(len(lines)) if take_first else range(len(lines) - 1, -1, -1)
    for index in indexes:
        if not lines[index]:
            continue
        boundary_lines.append(lines[index])
        if len(boundary_lines) >= limit:
            break
    return boundary_lines


def _first_non_empty_index(lines: list[str]) -> int | None:
    for index, line in enumerate(lines):
        if line:
            return index
    return None


def _last_non_empty_index(lines: list[str]) -> int | None:
    for index in range(len(lines) - 1, -1, -1):
        if lines[index]:
            return index
    return None


def _boundary_signature(line: str) -> str | None:
    collapsed = WHITESPACE_RE.sub(" ", line.strip())
    if not collapsed:
        return None
    if len(collapsed) > 90:
        return None

    word_count = len(collapsed.split())
    if word_count > 12:
        return None
    if collapsed[:1].islower():
        return None
    if collapsed.endswith((".", "!", "?")) and word_count >= 5:
        return None

    lowered = re.sub(r"\d+", "#", collapsed.lower())
    lowered = NON_ALNUM_RE.sub(" ", lowered)
    lowered = WHITESPACE_RE.sub(" ", lowered).strip()
    return lowered or None


def _is_page_number_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    lowered = stripped.lower()
    return bool(PAGE_NUMBER_RE.fullmatch(lowered) or ROMAN_NUMERAL_RE.fullmatch(lowered))


def extract_document_pages(file_path: Path) -> tuple[int, list[DocumentPage]]:
    pages: list[DocumentPage] = []
    settings = get_settings()

    with fitz.open(file_path) as pdf_document:
        raw_page_texts = [page.get_text("text") for page in pdf_document]
        normalized_page_texts = normalize_document_pages(
            raw_page_texts,
            max_heading_words=settings.chunk_max_heading_words,
        )

        for index, raw_text in enumerate(raw_page_texts, start=1):
            pages.append(
                DocumentPage(
                    page_number=index,
                    raw_text=raw_text,
                    normalized_text=normalized_page_texts[index - 1],
                )
            )

    return len(pages), pages


def run_extraction_pipeline(
    session: Session, document: Document, ingestion_job: IngestionJob
) -> tuple[Document, IngestionJob]:
    settings = get_settings()
    chunking_config = ChunkingConfig(
        target_words=settings.chunk_target_words,
        min_words=settings.chunk_min_words,
        overlap_words=settings.chunk_overlap_words,
        max_heading_words=settings.chunk_max_heading_words,
    )

    ingestion_job.status = "processing"
    ingestion_job.started_at = datetime.now(UTC)
    document.status = "processing"
    document.chunking_config = chunking_config.to_dict()
    session.commit()

    try:
        page_count, pages = extract_document_pages(Path(document.file_path))
    except Exception:
        return _mark_ingestion_failed(
            session=session,
            document=document,
            ingestion_job=ingestion_job,
            error_message="PDF extraction failed.",
        )

    document.pages.clear()
    session.flush()
    for page in pages:
        document.pages.append(page)

    document.chunks.clear()
    session.flush()
    for chunk in build_document_chunks(document.pages, config=chunking_config):
        document.chunks.append(chunk)

    try:
        provider = get_embedding_provider()
        chunk_embeddings = provider.embed_texts([chunk.text for chunk in document.chunks])
    except Exception:
        return _mark_ingestion_failed(
            session=session,
            document=document,
            ingestion_job=ingestion_job,
            error_message="Embedding generation failed.",
        )

    for chunk, embedding in zip(document.chunks, chunk_embeddings, strict=True):
        chunk.embedding_dimensions = provider.dimensions
        chunk.embedding_vector = embedding

    document.page_count = page_count
    document.status = "ready"
    ingestion_job.status = "completed"
    ingestion_job.error_message = None
    ingestion_job.finished_at = datetime.now(UTC)
    session.commit()
    session.refresh(document)
    session.refresh(ingestion_job)
    return document, ingestion_job


def _mark_ingestion_failed(
    *,
    session: Session,
    document: Document,
    ingestion_job: IngestionJob,
    error_message: str,
) -> tuple[Document, IngestionJob]:
    session.rollback()
    document.status = "failed"
    ingestion_job.status = "failed"
    ingestion_job.error_message = error_message
    ingestion_job.finished_at = datetime.now(UTC)
    session.commit()
    session.refresh(document)
    session.refresh(ingestion_job)
    return document, ingestion_job
