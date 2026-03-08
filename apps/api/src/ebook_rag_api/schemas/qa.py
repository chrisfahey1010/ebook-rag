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
    provenance: dict[str, object] | None
    dense_score: float
    lexical_score: float
    hybrid_score: float
    rerank_score: float
    score: float


class QAResponse(BaseModel):
    normalized_question: str
    answer: str
    supported: bool
    answer_mode: str
    confidence: float
    support_score: float
    verification: "QAAnswerVerification | None" = None
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
    provenance: dict[str, object] | None
    dense_score: float
    lexical_score: float
    hybrid_score: float
    rerank_score: float
    score: float


class QAQuestionRouter(BaseModel):
    answer_mode: str
    reason: str
    facet_count: int
    context_count: int
    should_use_generative: bool


class QARuntimeMetadata(BaseModel):
    embedding_provider: str
    embedding_model: str | None
    reranker_provider: str
    reranker_model: str | None
    answer_provider: str
    answer_model: str | None


class QAClaimVerification(BaseModel):
    claim_text: str
    supported: bool
    support_score: float
    verifier: str
    rationale: str
    citations: list[QATraceChunk]


class QAAnswerVerification(BaseModel):
    verified: bool
    verifier: str
    claim_count: int
    supported_claim_count: int
    average_claim_score: float
    minimum_claim_score: float
    claims: list[QAClaimVerification]


class QATrace(BaseModel):
    answer_provider: str
    answer_mode: str
    question_router: QAQuestionRouter
    runtime: QARuntimeMetadata
    retrieved_chunks: list[QATraceChunk]
    selected_contexts: list[QATraceChunk]
    cited_contexts: list[QATraceChunk]
    prompt_snapshot: str
    timings: QATimingBreakdown
    verification: QAAnswerVerification | None
