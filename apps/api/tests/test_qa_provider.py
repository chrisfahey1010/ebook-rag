from ebook_rag_api.services.qa import (
    OpenAICompatibleAnswerProvider,
    RetrievedChunkContext,
    build_qa_prompt,
    extract_chat_completion_text,
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

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def post(self, url: str, json: dict, headers: dict) -> _FakeResponse:
        self.calls.append({"url": url, "json": json, "headers": headers})
        return _FakeResponse(self.payload)


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


def test_extract_chat_completion_text_handles_string_and_list_content() -> None:
    assert (
        extract_chat_completion_text(
            {"choices": [{"message": {"content": "Answer text"}}]}
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
