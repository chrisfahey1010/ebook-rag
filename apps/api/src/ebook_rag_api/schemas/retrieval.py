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
    lexical_score: float
    hybrid_score: float
    rerank_score: float
    score: float


class RetrievalSearchResponse(BaseModel):
    normalized_query: str
    matches: list[RetrievalMatch]


class DebugDocumentChunk(BaseModel):
    chunk_id: str
    document_id: str
    document_title: str | None
    document_filename: str
    chunk_index: int
    page_start: int
    page_end: int
    heading: str | None
    text: str
    token_estimate: int
    embedding_dimensions: int | None


class DebugDocumentChunksResponse(BaseModel):
    document_id: str
    document_title: str | None
    document_filename: str
    chunk_count: int
    chunks: list[DebugDocumentChunk]


class DebugRerankRequest(BaseModel):
    query: str = Field(min_length=1)
    passages: list[str] = Field(min_length=1, max_length=20)


class DebugRerankResult(BaseModel):
    index: int
    text: str
    rerank_score: float


class DebugRerankResponse(BaseModel):
    query: str
    reranker: str
    results: list[DebugRerankResult]
