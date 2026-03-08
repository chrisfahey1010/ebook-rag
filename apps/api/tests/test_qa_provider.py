import json

from ebook_rag_api.services.qa import (
    OpenAICompatibleAnswerProvider,
    RetrievedChunkContext,
    assemble_answer_contexts,
    build_qa_prompt,
    build_grounded_synthesis_prompt,
    extract_chat_completion_delta,
    extract_chat_completion_text,
    route_question,
)


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self.payload


class _FakeClient:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls: list[dict] = []
        self.stream_calls: list[dict] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def post(self, url: str, json: dict, headers: dict) -> _FakeResponse:
        self.calls.append({"url": url, "json": json, "headers": headers})
        return _FakeResponse(self.payload)

    def stream(self, method: str, url: str, json: dict, headers: dict):
        self.stream_calls.append(
            {"method": method, "url": url, "json": json, "headers": headers}
        )
        payload = self.payload

        class _StreamResponse:
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, exc_type, exc, tb) -> None:
                return None

            def raise_for_status(self_inner) -> None:
                return None

            def iter_lines(self_inner):
                for line in payload.get("lines", []):
                    yield line

        return _StreamResponse()


def test_build_qa_prompt_includes_question_and_page_metadata() -> None:
    prompt = build_qa_prompt(
        question="What should happen before launch?",
        contexts=[
            RetrievedChunkContext(
                chunk_id="chunk-1",
                document_id="doc-1",
                document_title="Launch Manual",
                document_filename="launch.pdf",
                chunk_index=0,
                page_start=3,
                page_end=4,
                text="Fuel lines must be inspected before launch.",
                score=0.92,
            )
        ],
    )

    assert "Question: What should happen before launch?" in prompt
    assert "Document: Launch Manual | Pages: 3-4 | Chunk: 0" in prompt
    assert "Fuel lines must be inspected before launch." in prompt
    assert "INSUFFICIENT_SUPPORT" in prompt


def test_build_grounded_synthesis_prompt_calls_out_multi_evidence_behavior() -> None:
    prompt = build_grounded_synthesis_prompt(
        question="What happened before launch and after landing?",
        contexts=[
            RetrievedChunkContext(
                chunk_id="chunk-1",
                document_id="doc-1",
                document_title="Launch Manual",
                document_filename="launch.pdf",
                chunk_index=0,
                page_start=3,
                page_end=4,
                text="Fuel lines must be inspected before launch.",
                score=0.92,
            )
        ],
    )

    assert "Synthesize only claims directly supported by the evidence." in prompt
    assert "If any requested facet lacks support" in prompt


def test_extract_chat_completion_text_handles_string_and_list_content() -> None:
    assert (
        extract_chat_completion_text(
            {"choices": [{"message": {"content": "Answer text"}}]}
        )
        == "Answer text"
    )


def test_extract_chat_completion_delta_handles_string_and_list_content() -> None:
    assert (
        extract_chat_completion_delta(
            {"choices": [{"delta": {"content": "Answer text"}}]}
        )
        == "Answer text"
    )
    assert (
        extract_chat_completion_delta(
            {
                "choices": [
                    {
                        "delta": {
                            "content": [
                                {"type": "text", "text": "Answer"},
                                {"type": "text", "text": " text"},
                            ]
                        }
                    }
                ]
            }
        )
        == "Answer text"
    )
    assert (
        extract_chat_completion_text(
            {
                "choices": [
                    {
                        "message": {
                            "content": [
                                {"type": "text", "text": "Answer"},
                                {"type": "text", "text": " text"},
                            ]
                        }
                    }
                ]
            }
        )
        == "Answer text"
    )


def test_openai_compatible_provider_returns_answer_and_citations(monkeypatch) -> None:
    fake_client = _FakeClient(
        {"choices": [{"message": {"content": "Inspect the fuel lines before launch."}}]}
    )

    monkeypatch.setattr(
        "ebook_rag_api.services.qa.httpx.Client",
        lambda timeout: fake_client,
    )

    provider = OpenAICompatibleAnswerProvider(
        base_url="http://localhost:11434/v1",
        api_key="secret",
        model="llama3.2",
        timeout_seconds=5.0,
        temperature=0.0,
        max_tokens=200,
    )
    context = RetrievedChunkContext(
        chunk_id="chunk-1",
        document_id="doc-1",
        document_title="Launch Manual",
        document_filename="launch.pdf",
        chunk_index=0,
        page_start=3,
        page_end=3,
        text="Inspect the fuel lines before launch.",
        score=0.91,
    )

    answer = provider.generate_answer(
        question="What should happen before launch?",
        contexts=[context],
    )

    assert answer.supported is True
    assert answer.answer_text == "Inspect the fuel lines before launch."
    assert answer.citations == [context]
    assert fake_client.calls[0]["url"] == "http://localhost:11434/v1/chat/completions"
    assert fake_client.calls[0]["json"]["model"] == "llama3.2"
    assert fake_client.calls[0]["headers"]["Authorization"] == "Bearer secret"


def test_route_question_prefers_synthesis_for_multi_facet_questions() -> None:
    provider = OpenAICompatibleAnswerProvider(
        base_url="http://localhost:11434/v1",
        api_key="secret",
        model="llama3.2",
        timeout_seconds=5.0,
        temperature=0.0,
        max_tokens=200,
    )
    decision = route_question(
        question="What should happen before launch and after landing?",
        contexts=[
            RetrievedChunkContext(
                chunk_id="chunk-1",
                document_id="doc-1",
                document_title="Launch Manual",
                document_filename="launch.pdf",
                chunk_index=0,
                page_start=3,
                page_end=3,
                text="Inspect the fuel lines before launch. Deploy the parachute after landing.",
                score=0.91,
            )
        ],
        answer_provider=provider,
    )

    assert decision.answer_mode == "synthesis"
    assert decision.should_use_generative is True


def test_openai_compatible_provider_maps_insufficient_support(monkeypatch) -> None:
    fake_client = _FakeClient(
        {"choices": [{"message": {"content": "INSUFFICIENT_SUPPORT"}}]}
    )

    monkeypatch.setattr(
        "ebook_rag_api.services.qa.httpx.Client",
        lambda timeout: fake_client,
    )

    provider = OpenAICompatibleAnswerProvider(
        base_url="http://localhost:11434/v1",
        api_key="secret",
        model="llama3.2",
        timeout_seconds=5.0,
        temperature=0.0,
        max_tokens=200,
    )
    context = RetrievedChunkContext(
        chunk_id="chunk-1",
        document_id="doc-1",
        document_title="Launch Manual",
        document_filename="launch.pdf",
        chunk_index=0,
        page_start=3,
        page_end=3,
        text="Inspect the fuel lines before launch.",
        score=0.91,
    )

    answer = provider.generate_answer(
        question="What should happen before launch?",
        contexts=[context],
    )

    assert answer.supported is False
    assert answer.citations == []


def test_openai_compatible_provider_streams_answer_deltas(monkeypatch) -> None:
    fake_client = _FakeClient(
        {
            "lines": [
                f"data: {json.dumps({'choices': [{'delta': {'content': 'Inspect the fuel'}}]})}",
                f"data: {json.dumps({'choices': [{'delta': {'content': ' lines before launch.'}}]})}",
                "data: [DONE]",
            ]
        }
    )

    monkeypatch.setattr(
        "ebook_rag_api.services.qa.httpx.Client",
        lambda timeout: fake_client,
    )

    provider = OpenAICompatibleAnswerProvider(
        base_url="http://localhost:11434/v1",
        api_key="secret",
        model="llama3.2",
        timeout_seconds=5.0,
        temperature=0.0,
        max_tokens=200,
    )
    context = RetrievedChunkContext(
        chunk_id="chunk-1",
        document_id="doc-1",
        document_title="Launch Manual",
        document_filename="launch.pdf",
        chunk_index=0,
        page_start=3,
        page_end=3,
        text="Inspect the fuel lines before launch.",
        score=0.91,
    )

    deltas = [
        chunk.delta
        for chunk in provider.stream_answer(
            question="What should happen before launch?",
            contexts=[context],
        )
    ]

    assert deltas == ["Inspect the fuel", " lines before launch."]
    assert fake_client.stream_calls[0]["json"]["stream"] is True
    assert fake_client.stream_calls[0]["url"] == "http://localhost:11434/v1/chat/completions"


def test_assemble_answer_contexts_deduplicates_near_identical_chunks() -> None:
    selected = assemble_answer_contexts(
        [
            RetrievedChunkContext(
                chunk_id="chunk-1",
                document_id="doc-1",
                document_title="Guide",
                document_filename="guide.pdf",
                chunk_index=0,
                page_start=1,
                page_end=1,
                text="Charge the battery before long field sessions.",
                token_estimate=7,
                score=0.95,
            ),
            RetrievedChunkContext(
                chunk_id="chunk-2",
                document_id="doc-1",
                document_title="Guide",
                document_filename="guide.pdf",
                chunk_index=4,
                page_start=2,
                page_end=2,
                text="Charge the battery before long field sessions.",
                token_estimate=7,
                score=0.91,
            ),
        ]
    )

    assert len(selected) == 1
    assert selected[0].chunk_id == "chunk-1"


def test_assemble_answer_contexts_includes_adjacent_chunks_within_budget() -> None:
    selected = assemble_answer_contexts(
        [
            RetrievedChunkContext(
                chunk_id="chunk-10",
                document_id="doc-1",
                document_title="Guide",
                document_filename="guide.pdf",
                chunk_index=10,
                page_start=5,
                page_end=5,
                text="First inspect the pressure seal.",
                token_estimate=5,
                score=0.96,
            ),
            RetrievedChunkContext(
                chunk_id="chunk-11",
                document_id="doc-1",
                document_title="Guide",
                document_filename="guide.pdf",
                chunk_index=11,
                page_start=5,
                page_end=5,
                text="Then tighten the retaining ring.",
                token_estimate=5,
                score=0.88,
            ),
        ]
    )

    assert len(selected) == 2
    assert selected[0].chunk_id == "chunk-10"
    assert selected[1].chunk_id == "chunk-11"
    assert selected[0].text == "First inspect the pressure seal."
    assert selected[1].text == "Then tighten the retaining ring."
    assert selected[0].page_start == 5
    assert selected[1].page_end == 5


def test_assemble_answer_contexts_respects_budget_for_large_chunks() -> None:
    oversized_text = " ".join(["alpha"] * 900)
    selected = assemble_answer_contexts(
        [
            RetrievedChunkContext(
                chunk_id="chunk-1",
                document_id="doc-1",
                document_title="Guide",
                document_filename="guide.pdf",
                chunk_index=0,
                page_start=1,
                page_end=1,
                text=oversized_text,
                token_estimate=900,
                score=0.97,
            ),
            RetrievedChunkContext(
                chunk_id="chunk-2",
                document_id="doc-1",
                document_title="Guide",
                document_filename="guide.pdf",
                chunk_index=1,
                page_start=1,
                page_end=1,
                text=oversized_text.replace("alpha", "beta"),
                token_estimate=900,
                score=0.89,
            ),
        ]
    )

    assert len(selected) == 1
    assert selected[0].chunk_id == "chunk-1"
