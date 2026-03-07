import re
from dataclasses import dataclass
from functools import lru_cache
from time import perf_counter
from typing import Protocol

import httpx
from sqlalchemy.orm import Session

from ebook_rag_api.core.config import get_settings
from ebook_rag_api.services.retrieval import ChunkSearchMatch, normalize_query, search_chunks

STOPWORDS = {
    "a",
    "about",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "what",
    "which",
    "who",
    "with",
}
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass(frozen=True)
class RetrievedChunkContext:
    chunk_id: str
    document_id: str
    document_title: str | None
    document_filename: str
    chunk_index: int
    page_start: int
    page_end: int
    text: str
    dense_score: float = 0.0
    rerank_score: float = 0.0
    score: float = 0.0


@dataclass(frozen=True)
class GeneratedAnswer:
    answer_text: str
    supported: bool
    citations: list[RetrievedChunkContext]


@dataclass(frozen=True)
class QATimingBreakdown:
    normalization_ms: float
    retrieval_ms: float
    context_assembly_ms: float
    answer_generation_ms: float
    total_ms: float


@dataclass(frozen=True)
class QATrace:
    answer_provider: str
    retrieved_chunks: list[RetrievedChunkContext]
    selected_contexts: list[RetrievedChunkContext]
    prompt_snapshot: str
    timings: QATimingBreakdown
    answer: GeneratedAnswer


class AnswerProvider(Protocol):
    def generate_answer(
        self, question: str, contexts: list[RetrievedChunkContext]
    ) -> GeneratedAnswer:
        ...


class ExtractiveAnswerProvider:
    def generate_answer(
        self, question: str, contexts: list[RetrievedChunkContext]
    ) -> GeneratedAnswer:
        if not contexts:
            return _unsupported_answer()

        question_terms = _tokenize(question)
        best_sentence = ""
        best_context: RetrievedChunkContext | None = None
        best_score = 0.0

        for context in contexts:
            for raw_sentence in SENTENCE_SPLIT_RE.split(context.text):
                sentence = raw_sentence.strip()
                if not sentence:
                    continue
                sentence_terms = _tokenize(sentence)
                if not sentence_terms:
                    continue
                overlap = len(question_terms & sentence_terms)
                if overlap == 0:
                    continue

                lexical_score = overlap / max(len(question_terms), 1)
                combined_score = lexical_score + max(context.score, 0.0) * 0.15
                if combined_score > best_score:
                    best_score = combined_score
                    best_sentence = sentence
                    best_context = context

        if best_context is None or best_score < 0.2:
            return _unsupported_answer()

        answer_text = best_sentence if best_sentence.endswith((".", "!", "?")) else f"{best_sentence}."
        citations = [best_context]
        for context in contexts:
            if context.chunk_id == best_context.chunk_id:
                continue
            if context.score <= 0:
                continue
            citations.append(context)
            if len(citations) == 3:
                break

        return GeneratedAnswer(answer_text=answer_text, supported=True, citations=citations)


class OpenAICompatibleAnswerProvider:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: float,
        temperature: float,
        max_tokens: int,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate_answer(
        self, question: str, contexts: list[RetrievedChunkContext]
    ) -> GeneratedAnswer:
        if not contexts:
            return _unsupported_answer()

        prompt = build_qa_prompt(question=question, contexts=contexts)
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You answer questions using only the provided document context. "
                        "If the evidence is insufficient, reply exactly with "
                        "'INSUFFICIENT_SUPPORT'."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                response = client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()
        except httpx.HTTPError:
            return _unsupported_answer()

        answer_text = extract_chat_completion_text(response.json())
        if not answer_text or answer_text.strip() == "INSUFFICIENT_SUPPORT":
            return _unsupported_answer()

        return GeneratedAnswer(
            answer_text=answer_text.strip(),
            supported=True,
            citations=select_citations(contexts),
        )


@lru_cache(maxsize=1)
def get_answer_provider() -> AnswerProvider:
    settings = get_settings()
    if settings.answer_provider == "extractive":
        return ExtractiveAnswerProvider()
    if settings.answer_provider == "openai_compatible":
        return OpenAICompatibleAnswerProvider(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            timeout_seconds=settings.llm_timeout_seconds,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )
    raise ValueError(f"Unsupported answer provider: {settings.answer_provider}")


def ask_question(
    session: Session, question: str, top_k: int, document_id: str | None = None
) -> tuple[str, list[RetrievedChunkContext], GeneratedAnswer]:
    normalized_question, trace = ask_question_with_trace(
        session=session,
        question=question,
        top_k=top_k,
        document_id=document_id,
    )
    return normalized_question, trace.retrieved_chunks, trace.answer


def ask_question_with_trace(
    session: Session,
    question: str,
    top_k: int,
    document_id: str | None = None,
    include_prompt_snapshot: bool = True,
) -> tuple[str, QATrace]:
    started_at = perf_counter()
    normalized_question = normalize_query(question)
    normalized_at = perf_counter()
    _, matches = search_chunks(
        session=session,
        query=normalized_question,
        top_k=top_k,
        document_id=document_id,
    )
    retrieved_at = perf_counter()
    retrieved_chunks = [build_chunk_context(match) for match in matches]
    selected_contexts = assemble_answer_contexts(retrieved_chunks)
    prompt_snapshot = (
        build_qa_prompt(
            question=normalized_question,
            contexts=selected_contexts,
        )
        if include_prompt_snapshot
        else ""
    )
    context_assembled_at = perf_counter()
    answer_provider = get_answer_provider()
    answer = answer_provider.generate_answer(
        question=normalized_question,
        contexts=selected_contexts,
    )
    answered_at = perf_counter()
    trace = QATrace(
        answer_provider=type(answer_provider).__name__,
        retrieved_chunks=retrieved_chunks,
        selected_contexts=selected_contexts,
        prompt_snapshot=prompt_snapshot,
        timings=QATimingBreakdown(
            normalization_ms=(normalized_at - started_at) * 1000,
            retrieval_ms=(retrieved_at - normalized_at) * 1000,
            context_assembly_ms=(context_assembled_at - retrieved_at) * 1000,
            answer_generation_ms=(answered_at - context_assembled_at) * 1000,
            total_ms=(answered_at - started_at) * 1000,
        ),
        answer=answer,
    )
    return normalized_question, trace


def build_chunk_context(match: ChunkSearchMatch) -> RetrievedChunkContext:
    chunk = match.chunk
    return RetrievedChunkContext(
        chunk_id=chunk.id,
        document_id=chunk.document_id,
        document_title=chunk.document.title,
        document_filename=chunk.document.original_filename,
        chunk_index=chunk.chunk_index,
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        text=chunk.text,
        dense_score=match.dense_score,
        rerank_score=match.rerank_score,
        score=match.score,
    )


def assemble_answer_contexts(
    contexts: list[RetrievedChunkContext],
) -> list[RetrievedChunkContext]:
    return list(contexts)


def _tokenize(text: str) -> set[str]:
    return {
        token
        for token in TOKEN_RE.findall(text.lower())
        if token not in STOPWORDS and len(token) > 1
    }


def build_qa_prompt(question: str, contexts: list[RetrievedChunkContext]) -> str:
    context_blocks = []
    for context in contexts:
        label = (
            f"Document: {context.document_title or context.document_filename} | "
            f"Pages: {context.page_start}-{context.page_end} | "
            f"Chunk: {context.chunk_index} | Score: {context.score:.4f}"
        )
        context_blocks.append(f"{label}\n{context.text}")

    joined_context = "\n\n---\n\n".join(context_blocks)
    return (
        f"Question: {question}\n\n"
        "Context:\n"
        f"{joined_context}\n\n"
        "Instructions:\n"
        "- Answer only from the context.\n"
        "- Keep the answer concise and factual.\n"
        "- If the context is insufficient, reply exactly with INSUFFICIENT_SUPPORT.\n"
    )


def extract_chat_completion_text(payload: dict) -> str:
    choices = payload.get("choices", [])
    if not choices:
        return ""
    first_choice = choices[0]
    message = first_choice.get("message", {})
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_segments = [
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        return "".join(text_segments)
    return ""


def select_citations(contexts: list[RetrievedChunkContext]) -> list[RetrievedChunkContext]:
    citations: list[RetrievedChunkContext] = []
    for context in contexts:
        if context.score <= 0:
            continue
        citations.append(context)
        if len(citations) == 3:
            break
    return citations


def _unsupported_answer() -> GeneratedAnswer:
    return GeneratedAnswer(
        answer_text=(
            "I could not find enough support in the indexed document content "
            "to answer that question."
        ),
        supported=False,
        citations=[],
    )
