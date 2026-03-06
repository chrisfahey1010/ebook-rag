from fastapi import APIRouter, Depends, File, UploadFile, status
from sqlalchemy.orm import Session

from ebook_rag_api.core.config import get_settings
from ebook_rag_api.db import get_db_session
from ebook_rag_api.schemas import DocumentSummary, DocumentUploadResponse
from ebook_rag_api.services.documents import (
    create_document_record,
    list_documents,
    store_pdf_upload,
)

router = APIRouter()


@router.get("", response_model=list[DocumentSummary], summary="List uploaded documents")
def get_documents(session: Session = Depends(get_db_session)) -> list[DocumentSummary]:
    documents = list_documents(session)
    return [DocumentSummary.model_validate(document) for document in documents]


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and register a PDF document",
)
def upload_document(
    file: UploadFile = File(...),
    session: Session = Depends(get_db_session),
) -> DocumentUploadResponse:
    settings = get_settings()
    stored_upload = store_pdf_upload(
        upload=file,
        uploads_dir=settings.uploads_dir,
        max_upload_size_mb=settings.max_upload_size_mb,
    )
    document, ingestion_job = create_document_record(session, stored_upload)

    return DocumentUploadResponse(
        document=DocumentSummary.model_validate(document),
        ingestion_job_id=ingestion_job.id,
        ingestion_status=ingestion_job.status,
    )
