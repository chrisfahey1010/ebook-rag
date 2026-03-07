from pydantic import BaseModel, ConfigDict, Field


class RetrievalSearchRequest(BaseModel):
    query: str = Field(min_length=1)
    document_id: str | None = None
    top_k: int = Field(default=5, ge=1, le=20)


class RetrievalMatch(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    chunk_id: str
    document_id: str
    document_title: str | None
    document_filename: str
    chunk_index: int
    page_start: int
    page_end: int
    text: str
    dense_score: float
    rerank_score: float
    score: float


class RetrievalSearchResponse(BaseModel):
    normalized_query: str
    matches: list[RetrievalMatch]
