from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ebook_rag_api.db import get_db_session
from ebook_rag_api.schemas import IngestionJobSummary, IngestionStatusResponse
from ebook_rag_api.schemas.document import DocumentSummary
from ebook_rag_api.services.ingestion import get_latest_ingestion_job, reprocess_document

router = APIRouter()


@router.get(
    "/{document_id}/status",
    response_model=IngestionStatusResponse,
    summary="Get the current ingestion status for a document",
)
def get_ingestion_status(
    document_id: str,
    session: Session = Depends(get_db_session),
) -> IngestionStatusResponse:
    document, ingestion_job = get_latest_ingestion_job(session, document_id)
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
    return IngestionStatusResponse(
        document=DocumentSummary.model_validate(document),
        ingestion_job=IngestionJobSummary.model_validate(ingestion_job)
        if ingestion_job is not None
        else None,
    )


@router.post(
    "/{document_id}/reprocess",
    response_model=IngestionStatusResponse,
    summary="Re-run extraction and embedding generation for a document",
)
def post_reprocess_document(
    document_id: str,
    session: Session = Depends(get_db_session),
) -> IngestionStatusResponse:
    document, _ = get_latest_ingestion_job(session, document_id)
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
    document, ingestion_job = reprocess_document(session, document)
    return IngestionStatusResponse(
        document=DocumentSummary.model_validate(document),
        ingestion_job=IngestionJobSummary.model_validate(ingestion_job),
    )
