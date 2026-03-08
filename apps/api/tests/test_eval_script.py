import importlib.util
import json
from pathlib import Path

import pytest


def load_run_eval_module():
    script_path = (
        Path(__file__).resolve().parent.parent / "scripts" / "run_eval.py"
    )
    spec = importlib.util.spec_from_file_location("run_eval", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_summarize_results_includes_percentiles_and_unsupported_precision() -> None:
    run_eval = load_run_eval_module()
    results = [
        {
            "document": "manual.pdf",
            "question": "What happens before launch?",
            "expected_supported": True,
            "supported": True,
            "regression_tier": "gating",
            "citation_match_mode": "any",
            "citation_text_match_mode": "any",
            "expected_citation_pages": [1],
            "expected_citation_text_contains": ["cooling lines"],
            "retrieved_pages": [1],
            "cited_pages": [1],
            "cited_texts": ["Inspect the cooling lines before launch."],
            "retrieval_hit": True,
            "citation_hit": True,
            "citation_text_hit": True,
            "support_hit": True,
            "answer_hit": True,
            "latency_ms": 10.0,
            "answer_preview": "Inspect the cooling lines.",
        },
        {
            "document": "manual.pdf",
            "question": "What does it say about whales?",
            "expected_supported": False,
            "supported": False,
            "regression_tier": "gating",
            "citation_match_mode": "any",
            "citation_text_match_mode": "any",
            "expected_citation_pages": [],
            "expected_citation_text_contains": [],
            "retrieved_pages": [],
            "cited_pages": [],
            "cited_texts": [],
            "retrieval_hit": True,
            "citation_hit": True,
            "citation_text_hit": True,
            "support_hit": True,
            "answer_hit": True,
            "latency_ms": 30.0,
            "answer_preview": "I could not find enough support.",
        },
        {
            "document": "manual.pdf",
            "question": "What happens after landing?",
            "expected_supported": True,
            "supported": False,
            "regression_tier": "exploratory",
            "citation_match_mode": "any",
            "citation_text_match_mode": "any",
            "expected_citation_pages": [2],
            "expected_citation_text_contains": ["inspect the seals"],
            "retrieved_pages": [1],
            "cited_pages": [],
            "cited_texts": [],
            "retrieval_hit": False,
            "citation_hit": False,
            "citation_text_hit": False,
            "support_hit": False,
            "answer_hit": False,
            "latency_ms": 50.0,
            "answer_preview": "I could not find enough support.",
        },
    ]

    summary = run_eval.summarize_results(
        benchmark_name="curated-eval",
        benchmark_path=Path("/tmp/curated.json"),
        top_k=5,
        chunking_config={"target_words": 420, "min_words": 180, "overlap_words": 64, "max_heading_words": 12},
        results=results,
    )

    assert summary["questions"] == 3
    assert summary["gating_questions"] == 2
    assert summary["exploratory_questions"] == 1
    assert summary["chunking_config"]["target_words"] == 420
    assert summary["retrieval_hit_rate"] == 2 / 3
    assert summary["citation_hit_rate"] == 2 / 3
    assert summary["citation_evidence_hit_rate"] == 2 / 3
    assert summary["gating_citation_evidence_hit_rate"] == 1.0
    assert summary["support_accuracy"] == 2 / 3
    assert summary["answer_match_rate"] == 2 / 3
    assert summary["unsupported_precision"] == 0.5
    assert summary["average_latency_ms"] == 30.0
    assert summary["latency_p50_ms"] == 30.0
    assert summary["latency_p95_ms"] == 48.0
    assert summary["predicted_unsupported_count"] == 2
    assert summary["expected_unsupported_count"] == 1


def test_compare_summaries_flags_quality_and_latency_regressions() -> None:
    run_eval = load_run_eval_module()
    current = {
        "benchmark": "current",
        "generated_at": "2026-03-07T00:00:00+00:00",
        "retrieval_hit_rate": 0.6,
        "citation_hit_rate": 0.7,
        "citation_evidence_hit_rate": 0.75,
        "gating_citation_evidence_hit_rate": 0.8,
        "support_accuracy": 0.9,
        "answer_match_rate": 0.8,
        "unsupported_precision": 0.4,
        "average_latency_ms": 120.0,
        "latency_p50_ms": 110.0,
        "latency_p95_ms": 180.0,
    }
    baseline = {
        "benchmark": "baseline",
        "generated_at": "2026-03-06T00:00:00+00:00",
        "retrieval_hit_rate": 0.8,
        "citation_hit_rate": 0.7,
        "citation_evidence_hit_rate": 0.9,
        "gating_citation_evidence_hit_rate": 0.9,
        "support_accuracy": 0.95,
        "answer_match_rate": 0.75,
        "unsupported_precision": 0.6,
        "average_latency_ms": 100.0,
        "latency_p50_ms": 90.0,
        "latency_p95_ms": 170.0,
    }

    comparison = run_eval.compare_summaries(current, baseline)

    assert comparison["has_regressions"] is True
    assert "retrieval_hit_rate" in comparison["regressions"]
    assert "gating_citation_evidence_hit_rate" in comparison["regressions"]
    assert "support_accuracy" in comparison["regressions"]
    assert "unsupported_precision" in comparison["regressions"]
    assert "average_latency_ms" in comparison["regressions"]
    assert "latency_p95_ms" in comparison["regressions"]
    assert comparison["metrics"]["answer_match_rate"]["regressed"] is False
    assert comparison["metrics"]["citation_evidence_hit_rate"]["regressed"] is False
    assert comparison["metrics"]["answer_match_rate"]["delta"] == pytest.approx(0.05)


def test_render_markdown_report_includes_comparison_and_failures() -> None:
    run_eval = load_run_eval_module()
    summary = {
        "benchmark": "curated-eval",
        "benchmark_path": "/tmp/curated.json",
        "generated_at": "2026-03-07T00:00:00+00:00",
        "top_k": 5,
        "chunking_config": {
            "target_words": 420,
            "min_words": 180,
            "overlap_words": 64,
            "max_heading_words": 12,
        },
        "questions": 2,
        "gating_questions": 1,
        "exploratory_questions": 1,
        "retrieval_hit_rate": 0.5,
        "citation_hit_rate": 1.0,
        "citation_evidence_hit_rate": 0.5,
        "gating_citation_evidence_hit_rate": 1.0,
        "support_accuracy": 0.5,
        "answer_match_rate": 0.5,
        "unsupported_precision": 1.0,
        "average_latency_ms": 20.0,
        "latency_p50_ms": 20.0,
        "latency_p95_ms": 28.0,
        "document_summaries": [
            {
                "document": "manual.pdf",
                "questions": 2,
                "gating_questions": 1,
                "exploratory_questions": 1,
                "gating_failures": 0,
                "exploratory_failures": 1,
            }
        ],
        "comparison": {
            "baseline_benchmark": "baseline",
            "baseline_generated_at": "2026-03-06T00:00:00+00:00",
            "has_regressions": True,
            "metrics": {
                "retrieval_hit_rate": {
                    "delta": -0.25,
                },
                "citation_evidence_hit_rate": {
                    "delta": -0.5,
                },
                "latency_p95_ms": {
                    "delta": 8.0,
                },
            },
        },
        "results": [
            {
                "document": "manual.pdf",
                "question": "What happens before launch?",
                "regression_tier": "exploratory",
                "retrieval_hit": False,
                "citation_hit": True,
                "citation_text_hit": False,
                "support_hit": False,
                "answer_hit": False,
                "expected_citation_text_contains": ["cooling lines"],
                "latency_ms": 28.0,
            }
        ],
    }

    report = run_eval.render_markdown_report(summary)

    assert "# Benchmark Report: curated-eval" in report
    assert '"target_words": 420' in report
    assert "manual.pdf: questions=2, gating=1 (failures=0), exploratory=1 (failures=1)" in report
    assert "Regressions detected: `yes`" in report
    assert "retrieval_hit_rate: `-0.250`" in report
    assert "citation_evidence_hit_rate: `-0.500`" in report
    assert "latency_p95_ms: `+8.000 ms`" in report
    assert "manual.pdf :: What happens before launch?" in report
    assert "citation_text=False" in report
    assert "answer=False" in report


def test_text_expectation_hit_normalizes_excerpt_spacing_and_punctuation() -> None:
    run_eval = load_run_eval_module()

    assert (
        run_eval.text_expectation_hit(
            expected_texts=['Free cash flow -- TTM (1)', '$  11,194', 'Amazon’s'],
            actual_texts=['Free cash flow - TTM (1) Q4 2025 $ 11,194', "Amazon's 2026 outlook"],
            match_mode="all",
        )
        is True
    )


def test_expand_page_span_includes_all_pages_in_a_chunk_range() -> None:
    run_eval = load_run_eval_module()

    assert run_eval.expand_page_span(3, 3) == {3}
    assert run_eval.expand_page_span(2, 4) == {2, 3, 4}


def test_page_expectation_hit_supports_any_and_all_matching() -> None:
    run_eval = load_run_eval_module()

    assert (
        run_eval.page_expectation_hit(
            expected_pages={2, 3},
            actual_pages={1, 2, 4},
            match_mode="any",
        )
        is True
    )


def test_resolve_question_case_option_prefers_question_then_document_then_benchmark() -> None:
    run_eval = load_run_eval_module()

    benchmark = {"defaults": {"regression_tier": "exploratory"}}
    document_case = {"defaults": {"regression_tier": "gating"}}
    question_case = {"regression_tier": "exploratory"}

    assert (
        run_eval.resolve_question_case_option(
            benchmark=benchmark,
            document_case=document_case,
            question_case=question_case,
            option_name="regression_tier",
            default="gating",
        )
        == "exploratory"
    )
    assert (
        run_eval.resolve_question_case_option(
            benchmark=benchmark,
            document_case=document_case,
            question_case={},
            option_name="regression_tier",
            default="gating",
        )
        == "gating"
    )
    assert (
        run_eval.resolve_question_case_option(
            benchmark=benchmark,
            document_case={},
            question_case={},
            option_name="regression_tier",
            default="gating",
        )
        == "exploratory"
    )


def test_normalize_regression_tier_accepts_known_values_and_rejects_unknown_values() -> None:
    run_eval = load_run_eval_module()

    assert run_eval.normalize_regression_tier("Gating") == "gating"
    assert run_eval.normalize_regression_tier(" exploratory ") == "exploratory"

    with pytest.raises(ValueError, match="Unsupported regression_tier"):
        run_eval.normalize_regression_tier("candidate")


def test_text_expectation_hit_supports_any_and_all_matching() -> None:
    run_eval = load_run_eval_module()

    assert (
        run_eval.text_expectation_hit(
            expected_texts=["cooling lines", "launch clock"],
            actual_texts=["Inspect the cooling lines before launch."],
            match_mode="any",
        )
        is True
    )
    assert (
        run_eval.text_expectation_hit(
            expected_texts=["cooling lines", "launch clock"],
            actual_texts=[
                "Inspect the cooling lines before launch.",
                "Synchronize the launch clock.",
            ],
            match_mode="all",
        )
        is True
    )
    assert (
        run_eval.text_expectation_hit(
            expected_texts=["cooling lines", "launch clock"],
            actual_texts=["Inspect the cooling lines before launch."],
            match_mode="all",
        )
        is False
    )


def test_load_benchmark_document_bytes_prefers_source_pdf(tmp_path: Path) -> None:
    run_eval = load_run_eval_module()
    pdf_path = tmp_path / "fixture.pdf"
    payload = run_eval.create_pdf(["One page of content."])
    pdf_path.write_bytes(payload)

    loaded = run_eval.load_benchmark_document_bytes(
        {
            "filename": "fixture.pdf",
            "source_pdf": str(pdf_path),
            "pages": ["This should be ignored."],
        }
    )

    assert loaded == payload


def test_load_benchmark_document_bytes_falls_back_to_inline_pages() -> None:
    run_eval = load_run_eval_module()

    loaded = run_eval.load_benchmark_document_bytes(
        {
            "filename": "fixture.pdf",
            "pages": ["Alpha page", "Beta page"],
        }
    )

    assert loaded.startswith(b"%PDF")


def test_load_benchmark_document_bytes_requires_source() -> None:
    run_eval = load_run_eval_module()

    with pytest.raises(ValueError):
        run_eval.load_benchmark_document_bytes({"filename": "fixture.pdf"})
    assert (
        run_eval.page_expectation_hit(
            expected_pages={2, 3},
            actual_pages={1, 2, 4},
            match_mode="all",
        )
        is False
    )
    assert (
        run_eval.page_expectation_hit(
            expected_pages={2, 3},
            actual_pages={2, 3, 4},
            match_mode="all",
        )
        is True
    )


def test_benchmark_fixtures_have_valid_sources_and_question_metadata() -> None:
    benchmark_dir = Path(__file__).resolve().parent.parent / "benchmarks"
    benchmark_files = sorted(
        path for path in benchmark_dir.glob("*.json") if path.parent.name != "results"
    )

    assert benchmark_files

    for benchmark_file in benchmark_files:
        payload = json.loads(benchmark_file.read_text())
        assert payload["name"]
        assert payload["documents"]

        for document_case in payload["documents"]:
            has_pages = bool(document_case.get("pages"))
            has_source_pdf = bool(document_case.get("source_pdf"))
            assert has_pages or has_source_pdf

            if has_source_pdf:
                resolved_source = benchmark_dir.parent / document_case["source_pdf"]
                assert resolved_source.exists(), f"missing source_pdf for {benchmark_file.name}"

            for question_case in document_case["questions"]:
                assert isinstance(question_case["expected_supported"], bool)
                assert isinstance(question_case["expected_citation_pages"], list)
                assert question_case.get("citation_match_mode", "any") in {"any", "all"}
                assert question_case.get("citation_text_match_mode", "any") in {
                    "any",
                    "all",
                }
                assert question_case.get("regression_tier", "gating") in {
                    "gating",
                    "exploratory",
                }


def test_resolve_chunking_config_applies_preset_and_overrides() -> None:
    run_eval = load_run_eval_module()
    namespace = type(
        "Args",
        (),
        {
            "chunk_preset": "large",
            "chunk_target_words": None,
            "chunk_min_words": 300,
            "chunk_overlap_words": None,
            "chunk_max_heading_words": None,
        },
    )()

    config = run_eval.resolve_chunking_config(namespace)

    assert config["target_words"] == 640
    assert config["min_words"] == 300
    assert config["overlap_words"] == 96


def test_validate_args_rejects_compare_presets_with_overrides() -> None:
    run_eval = load_run_eval_module()
    namespace = type(
        "Args",
        (),
        {
            "compare_presets": True,
            "compare_to": None,
            "chunk_target_words": 500,
            "chunk_min_words": None,
            "chunk_overlap_words": None,
            "chunk_max_heading_words": None,
        },
    )()

    with pytest.raises(ValueError):
        run_eval.validate_args(namespace)


def test_recommend_chunking_preset_prefers_quality_then_latency() -> None:
    run_eval = load_run_eval_module()

    recommended = run_eval.recommend_chunking_preset(
        {
            "small": {
                "retrieval_hit_rate": 1.0,
                "citation_hit_rate": 1.0,
                "citation_evidence_hit_rate": 1.0,
                "gating_citation_evidence_hit_rate": 1.0,
                "support_accuracy": 1.0,
                "answer_match_rate": 1.0,
                "unsupported_precision": 1.0,
                "average_latency_ms": 120.0,
                "latency_p95_ms": 180.0,
            },
            "default": {
                "retrieval_hit_rate": 1.0,
                "citation_hit_rate": 1.0,
                "citation_evidence_hit_rate": 1.0,
                "gating_citation_evidence_hit_rate": 1.0,
                "support_accuracy": 1.0,
                "answer_match_rate": 1.0,
                "unsupported_precision": 1.0,
                "average_latency_ms": 100.0,
                "latency_p95_ms": 170.0,
            },
            "large": {
                "retrieval_hit_rate": 0.9,
                "citation_hit_rate": 1.0,
                "citation_evidence_hit_rate": 0.9,
                "gating_citation_evidence_hit_rate": 0.9,
                "support_accuracy": 1.0,
                "answer_match_rate": 1.0,
                "unsupported_precision": 1.0,
                "average_latency_ms": 90.0,
                "latency_p95_ms": 140.0,
            },
        }
    )

    assert recommended == "default"


def test_render_preset_comparison_report_includes_recommendation() -> None:
    run_eval = load_run_eval_module()
    report = run_eval.render_preset_comparison_report(
        {
            "benchmark": "curated-eval",
            "benchmark_path": "/tmp/curated.json",
            "generated_at": "2026-03-07T00:00:00+00:00",
            "top_k": 5,
            "recommended_preset": "default",
            "recommended_chunking_config": {
                "target_words": 420,
                "min_words": 180,
                "overlap_words": 64,
                "max_heading_words": 12,
            },
            "decision_basis": "quality first, latency second",
            "preset_summaries": {
                "default": {
                    "chunking_config": {
                        "target_words": 420,
                        "min_words": 180,
                        "overlap_words": 64,
                        "max_heading_words": 12,
                    },
                    "retrieval_hit_rate": 1.0,
                    "citation_hit_rate": 0.9,
                    "citation_evidence_hit_rate": 0.9,
                    "gating_citation_evidence_hit_rate": 0.9,
                    "support_accuracy": 0.9,
                    "answer_match_rate": 0.9,
                    "unsupported_precision": 1.0,
                    "average_latency_ms": 100.0,
                    "latency_p50_ms": 90.0,
                    "latency_p95_ms": 140.0,
                }
            },
        }
    )

    assert "# Chunking Preset Comparison: curated-eval" in report
    assert "Recommended preset: `default`" in report
    assert '"target_words": 420' in report
    assert "Average latency: `100.00 ms`" in report
