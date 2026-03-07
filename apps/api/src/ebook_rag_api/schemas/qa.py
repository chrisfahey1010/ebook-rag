from pydantic import BaseModel, Field


class QARequest(BaseModel):
    question: str = Field(min_length=1)
    document_id: str | None = None
    top_k: int = Field(default=5, ge=1, le=20)
    include_trace: bool = False


class QACitation(BaseModel):
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


class QAResponse(BaseModel):
    normalized_question: str
    answer: str
    supported: bool
    citations: list[QACitation]
    retrieved_chunk_count: int
    trace: "QATrace | None" = None


class QATimingBreakdown(BaseModel):
    normalization_ms: float
    retrieval_ms: float
    context_assembly_ms: float
    answer_generation_ms: float
    total_ms: float


class QATraceChunk(BaseModel):
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


class QATrace(BaseModel):
    answer_provider: str
    retrieved_chunks: list[QATraceChunk]
    selected_contexts: list[QATraceChunk]
    prompt_snapshot: str
    timings: QATimingBreakdown
