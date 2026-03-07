from datetime import datetime

from pydantic import BaseModel, ConfigDict


class ChunkingConfigSummary(BaseModel):
    target_words: int
    min_words: int
    overlap_words: int
    max_heading_words: int


class DocumentSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    original_filename: str
    title: str | None
    sha256: str
    page_count: int | None
    chunk_count: int
    status: str
    chunking_config: ChunkingConfigSummary | None
    created_at: datetime
    updated_at: datetime | None


class DocumentUploadResponse(BaseModel):
    document: DocumentSummary
    ingestion_job_id: str
    ingestion_status: str
    ingestion_error: str | None = None
