from datetime import datetime

from pydantic import BaseModel, ConfigDict


class DocumentSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    original_filename: str
    title: str | None
    sha256: str
    page_count: int | None
    status: str
    created_at: datetime
    updated_at: datetime | None


class DocumentUploadResponse(BaseModel):
    document: DocumentSummary
    ingestion_job_id: str
    ingestion_status: str
    ingestion_error: str | None = None
