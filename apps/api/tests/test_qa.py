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
