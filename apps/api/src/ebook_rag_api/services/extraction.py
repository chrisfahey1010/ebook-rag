import re
from datetime import UTC, datetime
from pathlib import Path

import fitz
from sqlalchemy.orm import Session

from ebook_rag_api.models import Document, DocumentPage, IngestionJob
from ebook_rag_api.services.embeddings import get_embedding_provider
from ebook_rag_api.services.chunking import build_document_chunks

WHITESPACE_RE = re.compile(r"[ \t]+")
BLANK_LINE_RE = re.compile(r"\n{3,}")


def normalize_page_text(raw_text: str) -> str:
    lines = [WHITESPACE_RE.sub(" ", line).strip() for line in raw_text.splitlines()]
    normalized = "\n".join(line if line else "" for line in lines)
    normalized = BLANK_LINE_RE.sub("\n\n", normalized).strip()
    return normalized


def extract_document_pages(file_path: Path) -> tuple[int, list[DocumentPage]]:
    pages: list[DocumentPage] = []

    with fitz.open(file_path) as pdf_document:
        for index, page in enumerate(pdf_document, start=1):
            raw_text = page.get_text("text")
            pages.append(
                DocumentPage(
                    page_number=index,
                    raw_text=raw_text,
                    normalized_text=normalize_page_text(raw_text),
                )
            )

    return len(pages), pages


def run_extraction_pipeline(
    session: Session, document: Document, ingestion_job: IngestionJob
) -> tuple[Document, IngestionJob]:
    ingestion_job.status = "processing"
    ingestion_job.started_at = datetime.now(UTC)
    document.status = "processing"
    session.commit()

    try:
        page_count, pages = extract_document_pages(Path(document.file_path))
    except Exception:
        session.rollback()
        document.status = "failed"
        ingestion_job.status = "failed"
        ingestion_job.error_message = "PDF extraction failed."
        ingestion_job.finished_at = datetime.now(UTC)
        session.commit()
        session.refresh(document)
        session.refresh(ingestion_job)
        return document, ingestion_job

    document.pages.clear()
    for page in pages:
        document.pages.append(page)

    document.chunks.clear()
    for chunk in build_document_chunks(document.pages):
        document.chunks.append(chunk)

    provider = get_embedding_provider()
    chunk_embeddings = provider.embed_texts([chunk.text for chunk in document.chunks])
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
