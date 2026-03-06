from pydantic import BaseModel, Field


class QARequest(BaseModel):
    question: str = Field(min_length=1)
    document_id: str | None = None
    top_k: int = Field(default=5, ge=1, le=20)


class QACitation(BaseModel):
    chunk_id: str
    document_id: str
    document_title: str | None
    document_filename: str
    chunk_index: int
    page_start: int
    page_end: int
    text: str
    score: float


class QAResponse(BaseModel):
    normalized_question: str
    answer: str
    supported: bool
    citations: list[QACitation]
    retrieved_chunk_count: int
