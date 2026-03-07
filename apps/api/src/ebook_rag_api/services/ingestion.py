from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from ebook_rag_api.models import Document, IngestionJob
from ebook_rag_api.services.extraction import run_extraction_pipeline


def get_latest_ingestion_job(
    session: Session,
    document_id: str,
) -> tuple[Document | None, IngestionJob | None]:
    document = session.scalar(
        select(Document)
        .options(selectinload(Document.ingestion_jobs))
        .where(Document.id == document_id)
    )
    if document is None:
        return None, None
    if not document.ingestion_jobs:
        return document, None
    return document, document.ingestion_jobs[-1]


def reprocess_document(
    session: Session,
    document: Document,
) -> tuple[Document, IngestionJob]:
    ingestion_job = IngestionJob(document_id=document.id, status="queued")
    session.add(ingestion_job)
    session.commit()
    session.refresh(document)
    session.refresh(ingestion_job)
    return run_extraction_pipeline(session, document, ingestion_job)
