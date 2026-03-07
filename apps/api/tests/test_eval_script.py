from pathlib import Path
import importlib.util

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
            "expected_citation_pages": [1],
            "retrieved_pages": [1],
            "cited_pages": [1],
            "retrieval_hit": True,
            "citation_hit": True,
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
            "expected_citation_pages": [],
            "retrieved_pages": [],
            "cited_pages": [],
            "retrieval_hit": True,
            "citation_hit": True,
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
            "expected_citation_pages": [2],
            "retrieved_pages": [1],
            "cited_pages": [],
            "retrieval_hit": False,
            "citation_hit": False,
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
        results=results,
    )

    assert summary["questions"] == 3
    assert summary["retrieval_hit_rate"] == 2 / 3
    assert summary["citation_hit_rate"] == 2 / 3
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
    assert "support_accuracy" in comparison["regressions"]
    assert "unsupported_precision" in comparison["regressions"]
    assert "average_latency_ms" in comparison["regressions"]
    assert "latency_p95_ms" in comparison["regressions"]
    assert comparison["metrics"]["answer_match_rate"]["regressed"] is False
    assert comparison["metrics"]["answer_match_rate"]["delta"] == pytest.approx(0.05)


def test_render_markdown_report_includes_comparison_and_failures() -> None:
    run_eval = load_run_eval_module()
    summary = {
        "benchmark": "curated-eval",
        "benchmark_path": "/tmp/curated.json",
        "generated_at": "2026-03-07T00:00:00+00:00",
        "top_k": 5,
        "questions": 2,
        "retrieval_hit_rate": 0.5,
        "citation_hit_rate": 1.0,
        "support_accuracy": 0.5,
        "answer_match_rate": 0.5,
        "unsupported_precision": 1.0,
        "average_latency_ms": 20.0,
        "latency_p50_ms": 20.0,
        "latency_p95_ms": 28.0,
        "comparison": {
            "baseline_benchmark": "baseline",
            "baseline_generated_at": "2026-03-06T00:00:00+00:00",
            "has_regressions": True,
            "metrics": {
                "retrieval_hit_rate": {
                    "delta": -0.25,
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
                "retrieval_hit": False,
                "citation_hit": True,
                "support_hit": False,
                "answer_hit": False,
                "latency_ms": 28.0,
            }
        ],
    }

    report = run_eval.render_markdown_report(summary)

    assert "# Benchmark Report: curated-eval" in report
    assert "Regressions detected: `yes`" in report
    assert "retrieval_hit_rate: `-0.250`" in report
    assert "latency_p95_ms: `+8.000 ms`" in report
    assert "manual.pdf :: What happens before launch?" in report
    assert "answer=False" in report


def test_expand_page_span_includes_all_pages_in_a_chunk_range() -> None:
    run_eval = load_run_eval_module()

    assert run_eval.expand_page_span(3, 3) == {3}
    assert run_eval.expand_page_span(2, 4) == {2, 3, 4}
