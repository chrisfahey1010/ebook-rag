from datetime import datetime

from pydantic import BaseModel, ConfigDict

from ebook_rag_api.schemas.document import DocumentSummary


class IngestionJobSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    document_id: str
    status: str
    error_message: str | None
    started_at: datetime | None
    finished_at: datetime | None
    created_at: datetime


class IngestionStatusResponse(BaseModel):
    document: DocumentSummary
    ingestion_job: IngestionJobSummary | None
