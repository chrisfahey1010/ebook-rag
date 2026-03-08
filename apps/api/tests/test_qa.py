from fastapi.testclient import TestClient

from ebook_rag_api.services.qa import (
    ExtractiveAnswerProvider,
    RetrievedChunkContext,
    assemble_answer_contexts,
    select_evidence_citations,
)


def test_qa_answer_returns_grounded_answer_with_citations(
    client: TestClient, pdf_factory
) -> None:
    upload_response = client.post(
        "/api/documents/upload",
        files={
            "file": (
                "manual.pdf",
                pdf_factory(
                    [
                        "Battery care\n\nCharge the rover battery before long field sessions.",
                        "Maintenance\n\nInspect the rover wheels after each rocky traverse.",
                    ]
                ),
                "application/pdf",
            )
        },
    )

    assert upload_response.status_code == 201
    document_id = upload_response.json()["document"]["id"]

    response = client.post(
        "/api/qa/ask",
        json={
            "question": "What should happen before long field sessions?",
            "document_id": document_id,
            "top_k": 4,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["supported"] is True
    assert "charge the rover battery" in payload["answer"].lower()
    assert payload["citations"]
    assert payload["citations"][0]["document_id"] == document_id
    assert payload["citations"][0]["page_start"] == 1
    assert "charge the rover battery" in payload["citations"][0]["text"].lower()
    assert payload["retrieved_chunk_count"] >= 1


def test_qa_answer_returns_unsupported_when_evidence_is_missing(
    client: TestClient, pdf_factory
) -> None:
    upload_response = client.post(
        "/api/documents/upload",
        files={
            "file": (
                "astronomy.pdf",
                pdf_factory(
                    [
                        "Stars\n\nStars emit light because of nuclear fusion in their cores.",
                    ]
                ),
                "application/pdf",
            )
        },
    )

    assert upload_response.status_code == 201
    document_id = upload_response.json()["document"]["id"]

    response = client.post(
        "/api/qa/ask",
        json={
            "question": "What does the book say about whale migration routes?",
            "document_id": document_id,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["supported"] is False
    assert "could not find enough support" in payload["answer"].lower()
    assert payload["citations"] == []


def test_qa_answer_can_include_trace_payload(
    client: TestClient, pdf_factory
) -> None:
    upload_response = client.post(
        "/api/documents/upload",
        files={
            "file": (
                "operations.pdf",
                pdf_factory(
                    [
                        "Launch checklist\n\nInspect the heat shield before ignition.",
                        "Landing checklist\n\nDeploy the parachute after atmospheric entry.",
                    ]
                ),
                "application/pdf",
            )
        },
    )

    assert upload_response.status_code == 201
    document_id = upload_response.json()["document"]["id"]

    response = client.post(
        "/api/qa/ask",
        json={
            "question": "What should happen before ignition?",
            "document_id": document_id,
            "top_k": 4,
            "include_trace": True,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["trace"] is not None
    assert payload["trace"]["answer_provider"] == "ExtractiveAnswerProvider"
    assert payload["trace"]["retrieved_chunks"]
    assert payload["trace"]["selected_contexts"]
    assert "dense_score" in payload["trace"]["retrieved_chunks"][0]
    assert "lexical_score" in payload["trace"]["retrieved_chunks"][0]
    assert "hybrid_score" in payload["trace"]["retrieved_chunks"][0]
    assert "rerank_score" in payload["trace"]["retrieved_chunks"][0]
    assert payload["citations"][0]["provenance"]["char_range"]["start_page"] == 1
    assert payload["trace"]["retrieved_chunks"][0]["provenance"]["page_char_ranges"][0]["page_number"] == 1
    assert "Question: What should happen before ignition?" in payload["trace"]["prompt_snapshot"]
    assert "Inspect the heat shield before ignition." in payload["trace"]["prompt_snapshot"]
    assert payload["trace"]["selected_contexts"][0]["chunk_id"] == payload["citations"][0]["chunk_id"]
    assert payload["trace"]["cited_contexts"][0]["chunk_id"] == payload["citations"][0]["chunk_id"]
    assert payload["trace"]["timings"]["total_ms"] >= 0


def test_qa_citations_follow_the_answer_evidence_instead_of_all_selected_context(
    client: TestClient, pdf_factory
) -> None:
    upload_response = client.post(
        "/api/documents/upload",
        files={
            "file": (
                "manual.pdf",
                pdf_factory(
                    [
                        "Maintenance\n\nInspect the rover wheels after each rocky traverse.",
                        "Storage\n\nStore replacement batteries in a dry compartment away from dust.",
                    ]
                ),
                "application/pdf",
            )
        },
    )

    assert upload_response.status_code == 201
    document_id = upload_response.json()["document"]["id"]

    response = client.post(
        "/api/qa/ask",
        json={
            "question": "What should be inspected after rocky traverses?",
            "document_id": document_id,
            "top_k": 4,
            "include_trace": True,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["supported"] is True
    assert "inspect the rover wheels" in payload["answer"].lower()
    assert len(payload["citations"]) == 1
    assert payload["citations"][0]["page_start"] == 1
    assert payload["trace"]["selected_contexts"]
    assert payload["trace"]["cited_contexts"]
    assert len(payload["trace"]["cited_contexts"]) == 1
    assert payload["trace"]["cited_contexts"][0]["page_start"] == 1


def test_qa_answer_can_combine_support_for_multi_part_questions(
    client: TestClient, pdf_factory
) -> None:
    upload_response = client.post(
        "/api/documents/upload",
        files={
            "file": (
                "manual.pdf",
                pdf_factory(
                    [
                        "Departure\n\nCharge the rover battery before leaving camp.",
                        "Maintenance\n\nInspect the rover wheels after each rocky traverse.",
                    ]
                ),
                "application/pdf",
            )
        },
    )

    assert upload_response.status_code == 201
    document_id = upload_response.json()["document"]["id"]

    response = client.post(
        "/api/qa/ask",
        json={
            "question": "What should the crew do before leaving camp and after a rocky traverse?",
            "document_id": document_id,
            "top_k": 4,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["supported"] is True
    assert "charge the rover battery" in payload["answer"].lower()
    assert "inspect the rover wheels" in payload["answer"].lower()
    cited_pages = {
        page
        for citation in payload["citations"]
        for page in range(citation["page_start"], citation["page_end"] + 1)
    }
    assert cited_pages == {1, 2}


def test_extractive_provider_requires_support_for_each_multi_part_facet() -> None:
    provider = ExtractiveAnswerProvider()
    contexts = [
        RetrievedChunkContext(
            chunk_id="chunk-1",
            document_id="doc-1",
            document_title="Manual",
            document_filename="manual.pdf",
            chunk_index=1,
            page_start=1,
            page_end=1,
            text="Before leaving camp, charge the rover battery fully.",
            token_estimate=8,
            score=0.96,
            rerank_score=0.96,
        )
    ]

    answer = provider.generate_answer(
        question="What should the crew do before leaving camp and after a rocky traverse?",
        contexts=contexts,
    )

    assert answer.supported is False
    assert answer.citations == []


def test_extractive_provider_prefers_anchor_terms_over_repeated_verbs_in_multi_part_questions() -> None:
    provider = ExtractiveAnswerProvider()
    contexts = [
        RetrievedChunkContext(
            chunk_id="chunk-1",
            document_id="doc-1",
            document_title="Manual",
            document_filename="manual.pdf",
            chunk_index=1,
            page_start=1,
            page_end=1,
            text="Inspect the heat shield before ignition and confirm all external locks are clear.",
            token_estimate=12,
            score=0.97,
            rerank_score=0.97,
        ),
        RetrievedChunkContext(
            chunk_id="chunk-2",
            document_id="doc-1",
            document_title="Manual",
            document_filename="manual.pdf",
            chunk_index=2,
            page_start=4,
            page_end=4,
            text="Deploy the parachute after atmospheric entry, then confirm the descent thrusters are armed.",
            token_estimate=12,
            score=0.98,
            rerank_score=0.98,
        ),
        RetrievedChunkContext(
            chunk_id="chunk-3",
            document_id="doc-1",
            document_title="Manual",
            document_filename="manual.pdf",
            chunk_index=3,
            page_start=5,
            page_end=5,
            text="After landing, archive the vibration log, inspect the hull seals, and compare the final fuel readings with the ascent checklist.",
            token_estimate=18,
            score=0.95,
            rerank_score=0.95,
        ),
    ]

    answer = provider.generate_answer(
        question="What should the crew confirm before ignition and after landing?",
        contexts=contexts,
    )

    assert answer.supported is True
    assert "inspect the heat shield" in answer.answer_text.lower()
    assert "inspect the hull seals" in answer.answer_text.lower()
    assert "descent thrusters" not in answer.answer_text.lower()
    assert [citation.page_start for citation in answer.citations] == [1, 5]


def test_extractive_provider_rejects_semantically_adjacent_but_unsupported_questions() -> None:
    provider = ExtractiveAnswerProvider()
    contexts = [
        RetrievedChunkContext(
            chunk_id="chunk-1",
            document_id="doc-1",
            document_title="Book",
            document_filename="book.pdf",
            chunk_index=1,
            page_start=12,
            page_end=12,
            text=(
                "The press criticized the group heavily, and newspaper editors argued "
                "about public responsibility and censorship."
            ),
            token_estimate=18,
            score=0.92,
            rerank_score=0.92,
        ),
        RetrievedChunkContext(
            chunk_id="chunk-2",
            document_id="doc-1",
            document_title="Book",
            document_filename="book.pdf",
            chunk_index=2,
            page_start=13,
            page_end=13,
            text=(
                "Television coverage amplified the controversy, but the book does not "
                "discuss online platforms or platform policy."
            ),
            token_estimate=19,
            score=0.88,
            rerank_score=0.88,
        ),
    ]

    answer = provider.generate_answer(
        question="What does the book say about social media moderation policies?",
        contexts=contexts,
    )

    assert answer.supported is False
    assert answer.citations == []


def test_extractive_provider_can_answer_from_adjacent_sentence_span() -> None:
    provider = ExtractiveAnswerProvider()
    contexts = [
        RetrievedChunkContext(
            chunk_id="chunk-1",
            document_id="doc-1",
            document_title="Book",
            document_filename="book.pdf",
            chunk_index=1,
            page_start=186,
            page_end=186,
            text=(
                "Hunter S. Thompson is a freelance writer from San Francisco, Aspen, "
                "and points east. A native of Louisville, Kentucky, he began writing "
                "as a sports columnist."
            ),
            token_estimate=28,
            score=0.94,
            rerank_score=0.94,
        ),
    ]

    answer = provider.generate_answer(
        question="Where was Hunter Thompson a native of?",
        contexts=contexts,
    )

    assert answer.supported is True
    assert "louisville, kentucky" in answer.answer_text.lower()
    assert answer.citations
    assert answer.citations[0].page_start == 186


def test_extractive_provider_prefers_constraint_term_support_over_name_only_overlap() -> None:
    provider = ExtractiveAnswerProvider()
    contexts = [
        RetrievedChunkContext(
            chunk_id="chunk-1",
            document_id="doc-1",
            document_title="Book",
            document_filename="book.pdf",
            chunk_index=1,
            page_start=1,
            page_end=1,
            text="Hunter Thompson Copyright renewed 1994, 1995 by Hunter S. Thompson.",
            token_estimate=12,
            score=0.97,
            rerank_score=0.97,
        ),
        RetrievedChunkContext(
            chunk_id="chunk-2",
            document_id="doc-1",
            document_title="Book",
            document_filename="book.pdf",
            chunk_index=2,
            page_start=186,
            page_end=186,
            text=(
                "HUNTER THOMPSON is a free-lance writer from San Francisco, Aspen and "
                "points east. A native of Louisville, Kentucky, he began writing as a "
                "sports columnist in Florida."
            ),
            token_estimate=28,
            score=0.93,
            rerank_score=0.93,
        ),
    ]

    answer = provider.generate_answer(
        question="Where was Hunter Thompson a native of?",
        contexts=contexts,
    )

    assert answer.supported is True
    assert "louisville, kentucky" in answer.answer_text.lower()
    assert answer.citations
    assert answer.citations[0].page_start == 186


def test_assemble_answer_contexts_skips_irrelevant_adjacent_chunks() -> None:
    contexts = [
        RetrievedChunkContext(
            chunk_id="chunk-1",
            document_id="doc-1",
            document_title="Manual",
            document_filename="manual.pdf",
            chunk_index=1,
            page_start=1,
            page_end=1,
            text="Departure checklist. Inspect the cooling lines before launch.",
            token_estimate=10,
            score=0.95,
            rerank_score=0.95,
        ),
        RetrievedChunkContext(
            chunk_id="chunk-2",
            document_id="doc-1",
            document_title="Manual",
            document_filename="manual.pdf",
            chunk_index=2,
            page_start=1,
            page_end=1,
            text="Archive telemetry records monthly and review spare inventory counts.",
            token_estimate=10,
            score=0.99,
            rerank_score=0.99,
        ),
    ]

    selected = assemble_answer_contexts(
        question="What should happen to the cooling lines before launch?",
        contexts=contexts,
    )

    assert [context.chunk_id for context in selected] == ["chunk-1"]


def test_select_evidence_citations_returns_sentence_level_excerpt() -> None:
    context = RetrievedChunkContext(
        chunk_id="chunk-1",
        document_id="doc-1",
        document_title="Manual",
        document_filename="manual.pdf",
        chunk_index=1,
        page_start=4,
        page_end=4,
        text=(
            "General maintenance covers filter checks and storage logs. "
            "Inspect the cooling lines before launch."
        ),
        token_estimate=18,
        score=0.9,
        rerank_score=0.9,
    )

    citations = select_evidence_citations(
        answer_text="Inspect the cooling lines before launch.",
        contexts=[context],
        primary_context=context,
    )

    assert len(citations) == 1
    assert citations[0].text == "Inspect the cooling lines before launch."


def test_select_evidence_citations_trims_to_supporting_clause() -> None:
    context = RetrievedChunkContext(
        chunk_id="chunk-1",
        document_id="doc-1",
        document_title="Manual",
        document_filename="manual.pdf",
        chunk_index=1,
        page_start=4,
        page_end=4,
        text=(
            "Before launch, inspect the cooling lines, confirm the hatch seal, "
            "and archive the maintenance log after landing."
        ),
        token_estimate=24,
        score=0.9,
        rerank_score=0.9,
    )

    citations = select_evidence_citations(
        answer_text="Inspect the cooling lines before launch.",
        contexts=[context],
        primary_context=context,
    )

    assert len(citations) == 1
    assert citations[0].text == "Before launch, inspect the cooling lines"


def test_qa_answer_handles_line_wrapped_sentence_without_returning_fragment(
    client: TestClient, pdf_factory
) -> None:
    upload_response = client.post(
        "/api/documents/upload",
        files={
            "file": (
                "archive.pdf",
                pdf_factory(
                    [
                        "Reading room\n\nResearchers must leave pens at the locker station and use only the issued pencils at the\n tables.",
                    ]
                ),
                "application/pdf",
            )
        },
    )

    assert upload_response.status_code == 201
    document_id = upload_response.json()["document"]["id"]

    response = client.post(
        "/api/qa/ask",
        json={
            "question": "What writing tool may researchers use at the tables?",
            "document_id": document_id,
            "top_k": 4,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["supported"] is True
    assert "issued pencils" in payload["answer"].lower()
    assert payload["answer"].strip().lower() != "tables."


def test_select_evidence_citations_filters_weak_overlap_contexts() -> None:
    primary_context = RetrievedChunkContext(
        chunk_id="chunk-1",
        document_id="doc-1",
        document_title="Manual",
        document_filename="manual.pdf",
        chunk_index=1,
        page_start=2,
        page_end=2,
        text="Pause the approach if the distance reading jumps or the starboard latch fails.",
        token_estimate=16,
        score=0.95,
        rerank_score=0.95,
    )
    weak_context = RetrievedChunkContext(
        chunk_id="chunk-2",
        document_id="doc-1",
        document_title="Manual",
        document_filename="manual.pdf",
        chunk_index=2,
        page_start=3,
        page_end=3,
        text="Inspect the hatch before fueling and confirm the distance marker is visible.",
        token_estimate=14,
        score=0.8,
        rerank_score=0.8,
    )

    citations = select_evidence_citations(
        answer_text="Pause the approach if the distance reading jumps.",
        contexts=[primary_context, weak_context],
        primary_context=primary_context,
    )

    assert [citation.chunk_id for citation in citations] == ["chunk-1"]


def test_select_evidence_citations_use_best_context_per_answer_sentence() -> None:
    first_context = RetrievedChunkContext(
        chunk_id="chunk-1",
        document_id="doc-1",
        document_title="Manual",
        document_filename="manual.pdf",
        chunk_index=1,
        page_start=1,
        page_end=1,
        text="Before launch, inspect the cooling lines and verify the hatch seal.",
        token_estimate=12,
        score=0.95,
        rerank_score=0.95,
    )
    distractor_context = RetrievedChunkContext(
        chunk_id="chunk-2",
        document_id="doc-1",
        document_title="Manual",
        document_filename="manual.pdf",
        chunk_index=2,
        page_start=2,
        page_end=2,
        text=(
            "Before launch, archive the maintenance log and after landing, "
            "inspect the cargo straps."
        ),
        token_estimate=15,
        score=0.98,
        rerank_score=0.98,
    )
    second_context = RetrievedChunkContext(
        chunk_id="chunk-3",
        document_id="doc-1",
        document_title="Manual",
        document_filename="manual.pdf",
        chunk_index=3,
        page_start=3,
        page_end=3,
        text="After landing, inspect the hull seals and record the fuel totals.",
        token_estimate=12,
        score=0.94,
        rerank_score=0.94,
    )

    citations = select_evidence_citations(
        answer_text="Inspect the cooling lines before launch. After landing, inspect the hull seals.",
        contexts=[first_context, distractor_context, second_context],
        primary_context=first_context,
    )

    assert [citation.chunk_id for citation in citations] == ["chunk-1", "chunk-3"]


def test_select_evidence_citations_use_question_terms_to_break_page_ties() -> None:
    wrong_primary_context = RetrievedChunkContext(
        chunk_id="chunk-1",
        document_id="doc-1",
        document_title="Book",
        document_filename="book.pdf",
        chunk_index=10,
        page_start=10,
        page_end=10,
        text="The article appeared in April 1965 after contract negotiations concluded.",
        token_estimate=11,
        score=0.96,
        rerank_score=0.96,
    )
    better_context = RetrievedChunkContext(
        chunk_id="chunk-2",
        document_id="doc-1",
        document_title="Book",
        document_filename="book.pdf",
        chunk_index=11,
        page_start=11,
        page_end=11,
        text=(
            "Carey McWilliams asked Hunter S. Thompson to write the original article. "
            "The article appeared in April 1965."
        ),
        token_estimate=18,
        score=0.92,
        rerank_score=0.92,
    )

    citations = select_evidence_citations(
        answer_text="The original article appeared in April 1965.",
        contexts=[wrong_primary_context, better_context],
        primary_context=wrong_primary_context,
        question_text=(
            "Who asked Hunter S. Thompson to write the original article on "
            "motorcycle gangs and when did it appear?"
        ),
    )

    assert [citation.chunk_id for citation in citations] == ["chunk-2"]
    assert citations[0].page_start == 11


def test_select_evidence_citations_add_same_page_supplemental_excerpt_for_why_answer() -> None:
    primary_context = RetrievedChunkContext(
        chunk_id="chunk-1",
        document_id="doc-1",
        document_title="Earnings",
        document_filename="earnings.pdf",
        chunk_index=1,
        page_start=2,
        page_end=2,
        text=(
            "Free cash flow decreased to $11.2 billion for the trailing twelve months, "
            "driven primarily by a year-over-year increase of $50.7 billion in purchases "
            "of property and equipment, net of proceeds from sales and incentives."
        ),
        token_estimate=33,
        score=0.95,
        rerank_score=0.95,
    )
    supplemental_context = RetrievedChunkContext(
        chunk_id="chunk-2",
        document_id="doc-1",
        document_title="Earnings",
        document_filename="earnings.pdf",
        chunk_index=2,
        page_start=2,
        page_end=2,
        text=(
            "This increase primarily reflects investments in artificial intelligence "
            "infrastructure and higher capacity build-out."
        ),
        token_estimate=16,
        score=0.9,
        rerank_score=0.9,
    )
    other_page_context = RetrievedChunkContext(
        chunk_id="chunk-3",
        document_id="doc-1",
        document_title="Earnings",
        document_filename="earnings.pdf",
        chunk_index=3,
        page_start=3,
        page_end=3,
        text="Capital expenditures are expected to rise in 2026.",
        token_estimate=9,
        score=0.88,
        rerank_score=0.88,
    )

    citations = select_evidence_citations(
        answer_text=(
            "Free cash flow decreased because purchases of property and equipment increased, "
            "primarily reflecting investments in artificial intelligence."
        ),
        contexts=[primary_context, supplemental_context, other_page_context],
        primary_context=primary_context,
        question_text="Why did free cash flow decrease for the trailing twelve months ended December 31, 2025?",
    )

    assert [citation.chunk_id for citation in citations] == ["chunk-1", "chunk-2"]
    combined_text = " ".join(citation.text.lower() for citation in citations)
    assert "purchases of property and equipment" in combined_text
    assert "investments in artificial intelligence" in combined_text


def test_select_evidence_citations_merge_multiple_support_units_from_same_chunk() -> None:
    context = RetrievedChunkContext(
        chunk_id="chunk-1",
        document_id="doc-1",
        document_title="Earnings",
        document_filename="earnings.pdf",
        chunk_index=1,
        page_start=2,
        page_end=2,
        text=(
            "Free cash flow decreased to $11.2 billion for the trailing twelve months, "
            "driven primarily by a year-over-year increase of $50.7 billion in purchases "
            "of property and equipment, net of proceeds from sales and incentives. "
            "This increase primarily reflects investments in artificial intelligence."
        ),
        token_estimate=41,
        score=0.95,
        rerank_score=0.95,
    )

    citations = select_evidence_citations(
        answer_text=(
            "Free cash flow decreased to $11.2 billion for the trailing twelve months, "
            "driven primarily by a year-over-year increase of $50.7 billion in purchases of "
            "property and equipment, net of proceeds from sales and incentives. "
            "This increase primarily reflects investments in artificial intelligence."
        ),
        contexts=[context],
        primary_context=context,
        question_text="Why did free cash flow decrease for the trailing twelve months ended December 31, 2025?",
    )

    assert len(citations) == 1
    assert "purchases of property and equipment" in citations[0].text.lower()
    assert "investments in artificial intelligence" in citations[0].text.lower()


def test_extractive_provider_prefers_date_bearing_sentence_for_when_questions() -> None:
    provider = ExtractiveAnswerProvider()
    contexts = [
        RetrievedChunkContext(
            chunk_id="chunk-1",
            document_id="doc-1",
            document_title="Book",
            document_filename="book.pdf",
            chunk_index=1,
            page_start=2,
            page_end=2,
            text=(
                "Carey McWilliams asked Hunter S. Thompson to write the original article. "
                "The article appeared in The Nation in April 1965."
            ),
            token_estimate=18,
            score=0.9,
            rerank_score=0.9,
        ),
        RetrievedChunkContext(
            chunk_id="chunk-2",
            document_id="doc-1",
            document_title="Book",
            document_filename="book.pdf",
            chunk_index=2,
            page_start=38,
            page_end=38,
            text=(
                "It was not until my article on motorcycles appeared in The Nation "
                "that they really believed I had spent time with the gang."
            ),
            token_estimate=19,
            score=0.95,
            rerank_score=0.95,
        ),
    ]

    answer = provider.generate_answer(
        question="When did the original article appear in The Nation?",
        contexts=contexts,
    )

    assert answer.supported is True
    assert "april 1965" in answer.answer_text.lower()
    assert answer.citations
    assert answer.citations[0].page_start == 2


def test_assemble_answer_contexts_prefers_direct_evidence_over_broader_summary() -> None:
    contexts = [
        RetrievedChunkContext(
            chunk_id="chunk-1",
            document_id="doc-1",
            document_title="Manual",
            document_filename="manual.pdf",
            chunk_index=1,
            page_start=1,
            page_end=1,
            text=(
                "Launch summary. Before launch, crews inspect systems, review weather, "
                "confirm cargo locks, and inspect the cooling lines."
            ),
            token_estimate=22,
            score=0.97,
            lexical_score=0.8,
            rerank_score=0.97,
        ),
        RetrievedChunkContext(
            chunk_id="chunk-2",
            document_id="doc-1",
            document_title="Manual",
            document_filename="manual.pdf",
            chunk_index=2,
            page_start=2,
            page_end=2,
            text="Inspect the cooling lines before launch.",
            token_estimate=6,
            score=0.9,
            lexical_score=0.7,
            rerank_score=0.9,
        ),
    ]

    selected = assemble_answer_contexts(
        question="What should happen to the cooling lines before launch?",
        contexts=contexts,
    )

    assert selected[0].chunk_id == "chunk-2"


def test_assemble_answer_contexts_deprioritizes_metadata_front_matter() -> None:
    contexts = [
        RetrievedChunkContext(
            chunk_id="chunk-1",
            document_id="doc-1",
            document_title="Book",
            document_filename="book.pdf",
            chunk_index=0,
            page_start=1,
            page_end=1,
            text="Copyright 1994, 1995 by Hunter S. Thompson. All rights reserved. Paperback edition.",
            token_estimate=12,
            score=0.98,
            lexical_score=0.82,
            rerank_score=0.98,
        ),
        RetrievedChunkContext(
            chunk_id="chunk-2",
            document_id="doc-1",
            document_title="Book",
            document_filename="book.pdf",
            chunk_index=1,
            page_start=186,
            page_end=186,
            text=(
                "Hunter S. Thompson is a freelance writer from San Francisco, Aspen, "
                "and points east. A native of Louisville, Kentucky, he began writing "
                "as a sports columnist."
            ),
            token_estimate=28,
            score=0.92,
            lexical_score=0.74,
            rerank_score=0.92,
        ),
    ]

    selected = assemble_answer_contexts(
        question="Where was Hunter Thompson a native of?",
        contexts=contexts,
    )

    assert selected[0].chunk_id == "chunk-2"


def test_extractive_provider_handles_line_broken_table_metric_rows() -> None:
    provider = ExtractiveAnswerProvider()
    context = RetrievedChunkContext(
        chunk_id="chunk-1",
        document_id="doc-1",
        document_title="Earnings",
        document_filename="earnings.pdf",
        chunk_index=11,
        page_start=11,
        page_end=11,
        text=(
            "Q3 2024\n"
            "Q4 2024\n"
            "Q1 2025\n"
            "Q2 2025\n"
            "Q3 2025\n"
            "Q4 2025\n"
            "Cash Flows and Shares\n"
            "Operating cash flow -- trailing twelve months (TTM)\n"
            "$ 112,706\n"
            "$ 115,877\n"
            "$ 113,903\n"
            "$ 121,137\n"
            "$ 130,691\n"
            "$ 139,514\n"
            "Free cash flow -- TTM (1)\n"
            "$ 47,747\n"
            "$ 38,219\n"
            "$ 25,925\n"
            "$ 18,184\n"
            "$ 14,788\n"
            "$ 11,194\n"
            "(71) %"
        ),
        token_estimate=60,
        score=0.94,
        rerank_score=0.94,
    )

    answer = provider.generate_answer(
        question="What was trailing-twelve-month free cash flow in Q4 2025?",
        contexts=[context],
    )

    assert answer.supported is True
    assert "11,194" in answer.answer_text
    assert answer.citations
    assert "free cash flow" in answer.citations[0].text.lower()
    assert "11,194" in answer.citations[0].text


def test_extractive_provider_handles_employee_count_rows() -> None:
    provider = ExtractiveAnswerProvider()
    context = RetrievedChunkContext(
        chunk_id="chunk-1",
        document_id="doc-1",
        document_title="Earnings",
        document_filename="earnings.pdf",
        chunk_index=13,
        page_start=13,
        page_end=13,
        text=(
            "Q3 2024\n"
            "Q4 2024\n"
            "Q1 2025\n"
            "Q2 2025\n"
            "Q3 2025\n"
            "Q4 2025\n"
            "Employees (full-time and part-time; excludes contractors & temporary personnel)\n"
            "1,551,000\n"
            "1,556,000\n"
            "1,560,000\n"
            "1,546,000\n"
            "1,578,000\n"
            "1,576,000\n"
            "1 %"
        ),
        token_estimate=44,
        score=0.91,
        rerank_score=0.91,
    )

    answer = provider.generate_answer(
        question="How many employees did Amazon report in Q4 2025?",
        contexts=[context],
    )

    assert answer.supported is True
    assert "1,576,000" in answer.answer_text
    assert answer.citations
    assert "employees" in answer.citations[0].text.lower()
    assert "1,576,000" in answer.citations[0].text


def test_extractive_provider_handles_flattened_employee_count_rows() -> None:
    provider = ExtractiveAnswerProvider()
    context = RetrievedChunkContext(
        chunk_id="chunk-1",
        document_id="doc-1",
        document_title="Earnings",
        document_filename="earnings.pdf",
        chunk_index=13,
        page_start=13,
        page_end=13,
        text=(
            "Q3 2024\n"
            "Q4 2024\n"
            "Q1 2025\n"
            "Q2 2025\n"
            "Q3 2025\n"
            "Q4 2025\n"
            "Y/Y %\n"
            "Change\n"
            "Employees (full-time and part-time; excludes contractors & temporary personnel) "
            "1,551,000 1,556,000 1,560,000 1,546,000 1,578,000 1,576,000 1 % "
            "Employees (full-time and part-time; excludes contractors & temporary personnel) -- "
            "Y/Y growth 3 % 2 % 3 % 1 % 2 % 1 %"
        ),
        token_estimate=52,
        score=0.91,
        rerank_score=0.91,
    )

    answer = provider.generate_answer(
        question="How many employees did Amazon report in Q4 2025?",
        contexts=[context],
    )

    assert answer.supported is True
    assert "1,576,000" in answer.answer_text
    assert answer.citations
    assert "employees" in answer.citations[0].text.lower()
    assert "q4 2025" in answer.citations[0].text.lower()
    assert "1,576,000" in answer.citations[0].text


def test_extractive_provider_can_reuse_same_page_headers_for_split_metric_rows() -> None:
    provider = ExtractiveAnswerProvider()
    header_context = RetrievedChunkContext(
        chunk_id="chunk-1",
        document_id="doc-1",
        document_title="Earnings",
        document_filename="earnings.pdf",
        chunk_index=73,
        page_start=13,
        page_end=13,
        text=(
            "Q3 2024\n"
            "Q4 2024\n"
            "Q1 2025\n"
            "Q2 2025\n"
            "Q3 2025\n"
            "Q4 2025\n"
            "Y/Y %\n"
            "Change\n"
            "Net Sales\n"
            "Online stores (1) $ 61,411 $ 75,556 $ 57,407 $ 61,485 $ 67,407 $ 82,988 10 %"
        ),
        token_estimate=44,
        score=0.82,
        rerank_score=0.82,
    )
    metric_context = RetrievedChunkContext(
        chunk_id="chunk-2",
        document_id="doc-1",
        document_title="Earnings",
        document_filename="earnings.pdf",
        chunk_index=87,
        page_start=13,
        page_end=13,
        text=(
            "N/A\n"
            "Employees (full-time and part-time; excludes contractors & temporary personnel) "
            "1,551,000 1,556,000 1,560,000 1,546,000 1,578,000 1,576,000 1 % "
            "Employees (full-time and part-time; excludes contractors & temporary personnel) -- "
            "Y/Y growth 3 % 2 % 3 % 1 % 2 % 1 %"
        ),
        token_estimate=55,
        score=0.78,
        rerank_score=0.78,
    )

    answer = provider.generate_answer(
        question="How many employees did Amazon report in Q4 2025?",
        contexts=[header_context, metric_context],
    )

    assert answer.supported is True
    assert "1,576,000" in answer.answer_text
    assert answer.citations
    assert answer.citations[0].page_start == 13
    assert "employees" in answer.citations[0].text.lower()
    assert "q4 2025" in answer.citations[0].text.lower()


def test_extractive_provider_handles_flattened_free_cash_flow_rows() -> None:
    provider = ExtractiveAnswerProvider()
    context = RetrievedChunkContext(
        chunk_id="chunk-1",
        document_id="doc-1",
        document_title="Earnings",
        document_filename="earnings.pdf",
        chunk_index=11,
        page_start=11,
        page_end=11,
        text=(
            "Q3 2024\n"
            "Q4 2024\n"
            "Q1 2025\n"
            "Q2 2025\n"
            "Q3 2025\n"
            "Q4 2025\n"
            "Y/Y %\n"
            "Change\n"
            "Purchases of property and equipment, net of proceeds from sales and incentives -- TTM "
            "$ 64,959 $ 77,658 $ 87,978 $ 102,953 $ 115,903 $ 128,320 65 % "
            "Free cash flow -- TTM (1) $ 47,747 $ 38,219 $ 25,925 $ 18,184 $ 14,788 $ 11,194 (71) % "
            "Common shares and stock-based awards outstanding"
        ),
        token_estimate=54,
        score=0.94,
        rerank_score=0.94,
    )

    answer = provider.generate_answer(
        question="What was trailing-twelve-month free cash flow in Q4 2025?",
        contexts=[context],
    )

    assert answer.supported is True
    assert "11,194" in answer.answer_text
    assert "128,320" not in answer.answer_text
    assert answer.citations
    assert answer.citations[0].text == "Free cash flow -- TTM (1) Q4 2025 $ 11,194"


def test_extractive_provider_prefers_exact_metric_row_over_narrative_summary() -> None:
    provider = ExtractiveAnswerProvider()
    narrative_context = RetrievedChunkContext(
        chunk_id="chunk-2",
        document_id="doc-1",
        document_title="Earnings",
        document_filename="earnings.pdf",
        chunk_index=2,
        page_start=2,
        page_end=2,
        text=(
            "Net income increased to $77.7 billion in 2025, or $7.17 per diluted share, "
            "compared with $59.2 billion, or $5.53 per diluted share, in 2024. "
            "Operating cash flow increased 20% to $139.5 billion for the trailing twelve "
            "months, compared with $115.9 billion for the trailing twelve months ended "
            "December 31, 2024. Free cash flow decreased to $11.2 billion for the trailing "
            "twelve months, driven primarily by a year-over-year increase of $50.7 billion "
            "in purchases of property and equipment, net of proceeds from sales and incentives."
        ),
        token_estimate=84,
        score=0.95,
        rerank_score=0.95,
    )
    table_context = RetrievedChunkContext(
        chunk_id="chunk-11",
        document_id="doc-1",
        document_title="Earnings",
        document_filename="earnings.pdf",
        chunk_index=11,
        page_start=11,
        page_end=11,
        text=(
            "Q3 2024\n"
            "Q4 2024\n"
            "Q1 2025\n"
            "Q2 2025\n"
            "Q3 2025\n"
            "Q4 2025\n"
            "Y/Y %\n"
            "Change\n"
            "Purchases of property and equipment, net of proceeds from sales and incentives -- TTM "
            "$ 64,959 $ 77,658 $ 87,978 $ 102,953 $ 115,903 $ 128,320 65 %\n"
            "Free cash flow -- TTM (1) $ 47,747 $ 38,219 $ 25,925 $ 18,184 $ 14,788 $ 11,194 (71) %\n"
            "Common shares and stock-based awards outstanding"
        ),
        token_estimate=54,
        score=0.82,
        rerank_score=0.82,
    )

    answer = provider.generate_answer(
        question="What was trailing-twelve-month free cash flow in Q4 2025?",
        contexts=[narrative_context, table_context],
    )

    assert answer.supported is True
    assert "11,194" in answer.answer_text
    assert "11.2 billion" not in answer.answer_text
    assert answer.citations
    assert answer.citations[0].page_start == 11
    assert "free cash flow" in answer.citations[0].text.lower()
    assert "q4 2025" in answer.citations[0].text.lower()


def test_extractive_provider_keeps_narrative_explanation_for_why_question() -> None:
    provider = ExtractiveAnswerProvider()
    narrative_context = RetrievedChunkContext(
        chunk_id="chunk-2",
        document_id="doc-1",
        document_title="Earnings",
        document_filename="earnings.pdf",
        chunk_index=2,
        page_start=2,
        page_end=2,
        text=(
            "Net income increased to $77.7 billion in 2025, or $7.17 per diluted share, "
            "compared with $59.2 billion, or $5.53 per diluted share, in 2024. "
            "Operating cash flow increased 20% to $139.5 billion for the trailing twelve "
            "months, compared with $115.9 billion for the trailing twelve months ended "
            "December 31, 2024. Free cash flow decreased to $11.2 billion for the trailing "
            "twelve months, driven primarily by a year-over-year increase of $50.7 billion "
            "in purchases of property and equipment, net of proceeds from sales and incentives. "
            "This increase primarily reflects investments in artificial intelligence infrastructure."
        ),
        token_estimate=84,
        score=0.95,
        rerank_score=0.95,
    )
    table_context = RetrievedChunkContext(
        chunk_id="chunk-11",
        document_id="doc-1",
        document_title="Earnings",
        document_filename="earnings.pdf",
        chunk_index=11,
        page_start=11,
        page_end=11,
        text=(
            "Q3 2024\n"
            "Q4 2024\n"
            "Q1 2025\n"
            "Q2 2025\n"
            "Q3 2025\n"
            "Q4 2025\n"
            "Y/Y %\n"
            "Change\n"
            "Purchases of property and equipment, net of proceeds from sales and incentives -- TTM "
            "$ 64,959 $ 77,658 $ 87,978 $ 102,953 $ 115,903 $ 128,320 65 %\n"
            "Free cash flow -- TTM (1) $ 47,747 $ 38,219 $ 25,925 $ 18,184 $ 14,788 $ 11,194 (71) %\n"
            "Common shares and stock-based awards outstanding"
        ),
        token_estimate=54,
        score=0.82,
        rerank_score=0.82,
    )

    answer = provider.generate_answer(
        question="Why did free cash flow decrease for the trailing twelve months ended December 31, 2025?",
        contexts=[narrative_context, table_context],
    )

    assert answer.supported is True
    assert "driven primarily" in answer.answer_text.lower()
    assert "purchases of property and equipment" in answer.answer_text.lower()
    assert "investments in artificial intelligence" in answer.answer_text.lower()
    assert answer.citations
    assert answer.citations[0].page_start == 2


def test_extractive_provider_prefers_requested_metric_over_results_headline() -> None:
    provider = ExtractiveAnswerProvider()
    context = RetrievedChunkContext(
        chunk_id="chunk-1",
        document_id="doc-1",
        document_title="Earnings",
        document_filename="earnings.pdf",
        chunk_index=1,
        page_start=1,
        page_end=1,
        text=(
            "AMAZON.COM ANNOUNCES FOURTH QUARTER RESULTS\n"
            "SEATTLE February 5, 2026 Amazon.com, Inc. today announced financial results "
            "for its fourth quarter ended December 31, 2025.\n"
            "Net sales increased 14% to $213.4 billion in the fourth quarter, compared with "
            "$187.8 billion in fourth quarter 2024.\n"
            "Operating income increased to $25.0 billion in the fourth quarter, compared with "
            "$21.2 billion in fourth quarter 2024."
        ),
        token_estimate=52,
        score=0.93,
        rerank_score=0.93,
    )

    answer = provider.generate_answer(
        question="What were Amazon's net sales in the fourth quarter of 2025?",
        contexts=[context],
    )

    assert answer.supported is True
    assert "213.4 billion" in answer.answer_text
    assert "net sales increased 14% to $213.4 billion" in answer.answer_text.lower()
    assert "operating income" not in answer.answer_text.lower()
    assert answer.citations
    assert "net sales increased 14% to $213.4 billion" in answer.citations[0].text.lower()


def test_extractive_provider_prefers_requested_metric_row_with_matching_period() -> None:
    provider = ExtractiveAnswerProvider()
    context = RetrievedChunkContext(
        chunk_id="chunk-1",
        document_id="doc-1",
        document_title="Earnings",
        document_filename="earnings.pdf",
        chunk_index=13,
        page_start=13,
        page_end=13,
        text=(
            "Q3 2024\n"
            "Q4 2024\n"
            "Q1 2025\n"
            "Q2 2025\n"
            "Q3 2025\n"
            "Q4 2025\n"
            "Operating income\n"
            "$ 17.4 billion\n"
            "$ 21.2 billion\n"
            "$ 18.4 billion\n"
            "$ 14.7 billion\n"
            "$ 17.4 billion\n"
            "$ 25.0 billion\n"
            "Employees (full-time and part-time; excludes contractors & temporary personnel)\n"
            "1,551,000\n"
            "1,556,000\n"
            "1,560,000\n"
            "1,546,000\n"
            "1,578,000\n"
            "1,576,000\n"
        ),
        token_estimate=72,
        score=0.91,
        rerank_score=0.91,
    )

    answer = provider.generate_answer(
        question="How many employees did Amazon report in Q4 2025?",
        contexts=[context],
    )

    assert answer.supported is True
    assert "1,576,000" in answer.answer_text
    assert "25.0 billion" not in answer.answer_text
    assert answer.citations
    assert "employees" in answer.citations[0].text.lower()
    assert "q4 2025" in answer.citations[0].text.lower()
    assert "1,576,000" in answer.citations[0].text


def test_select_evidence_citations_prefers_numeric_guidance_line_over_nearby_metric() -> None:
    context = RetrievedChunkContext(
        chunk_id="chunk-1",
        document_id="doc-1",
        document_title="Earnings",
        document_filename="earnings.pdf",
        chunk_index=4,
        page_start=4,
        page_end=4,
        text=(
            "Financial Guidance\n"
            "First Quarter 2026 Guidance\n"
            "Net sales are expected to be between $173.5 billion and $178.5 billion.\n"
            "Operating income is expected to be between $16.5 billion and $21.5 billion."
        ),
        token_estimate=28,
        score=0.92,
        rerank_score=0.92,
    )

    citations = select_evidence_citations(
        answer_text="Net sales are expected to be between $173.5 billion and $178.5 billion.",
        contexts=[context],
        primary_context=context,
        question_text="What net sales range did Amazon guide for the first quarter of 2026?",
    )

    assert len(citations) == 1
    assert "net sales are expected to be between $173.5 billion and $178.5 billion" in (
        citations[0].text.lower()
    )
