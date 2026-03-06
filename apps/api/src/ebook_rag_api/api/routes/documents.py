from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from ebook_rag_api.core.config import get_settings
from ebook_rag_api.db import get_db_session
from ebook_rag_api.schemas import DocumentSummary, DocumentUploadResponse
from ebook_rag_api.services.documents import (
    create_document_record,
    delete_document,
    get_document,
    list_documents,
    store_pdf_upload,
)
from ebook_rag_api.services.extraction import run_extraction_pipeline

router = APIRouter()


@router.get("", response_model=list[DocumentSummary], summary="List uploaded documents")
def get_documents(session: Session = Depends(get_db_session)) -> list[DocumentSummary]:
    documents = list_documents(session)
    return [DocumentSummary.model_validate(document) for document in documents]


@router.get(
    "/{document_id}",
    response_model=DocumentSummary,
    summary="Get a single uploaded document",
)
def get_document_by_id(
    document_id: str, session: Session = Depends(get_db_session)
) -> DocumentSummary:
    document = get_document(session, document_id)
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
    return DocumentSummary.model_validate(document)


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
    document, ingestion_job = run_extraction_pipeline(session, document, ingestion_job)

    return DocumentUploadResponse(
        document=DocumentSummary.model_validate(document),
        ingestion_job_id=ingestion_job.id,
        ingestion_status=ingestion_job.status,
        ingestion_error=ingestion_job.error_message,
    )


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete an uploaded document and its indexed content",
)
def delete_document_by_id(
    document_id: str, session: Session = Depends(get_db_session)
) -> None:
    document = get_document(session, document_id)
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")

    delete_document(session, document)
