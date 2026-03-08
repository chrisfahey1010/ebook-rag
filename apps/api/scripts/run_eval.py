from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

import fitz

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

SUMMARY_QUALITY_METRICS = (
    "retrieval_hit_rate",
    "citation_hit_rate",
    "citation_evidence_hit_rate",
    "gating_citation_evidence_hit_rate",
    "support_accuracy",
    "answer_match_rate",
    "unsupported_precision",
)
REGRESSION_QUALITY_METRICS = (
    "retrieval_hit_rate",
    "citation_hit_rate",
    "gating_citation_evidence_hit_rate",
    "support_accuracy",
    "answer_match_rate",
    "unsupported_precision",
)
LATENCY_METRICS = (
    "average_latency_ms",
    "latency_p50_ms",
    "latency_p95_ms",
)
LATENCY_REGRESSION_BUFFER_MS = 10.0

EXCERPT_WHITESPACE_RE = re.compile(r"\s+")
EXCERPT_SEPARATOR_DASH_RE = re.compile(r"\s-+\s")
EXCERPT_CURRENCY_RE = re.compile(r"\$\s+")
EXCERPT_PERCENT_RE = re.compile(r"\s+%")
EXCERPT_OPEN_PAREN_RE = re.compile(r"\(\s+")
EXCERPT_CLOSE_PAREN_RE = re.compile(r"\s+\)")
EXCERPT_TRANSLATION_TABLE = str.maketrans(
    {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2015": "-",
        "\u2212": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a local retrieval and citation benchmark with optional regression comparison."
    )
    parser.add_argument(
        "--benchmark",
        default="benchmarks/curated_eval.json",
        help="Path to a benchmark JSON file relative to apps/api or absolute.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-k retrieval depth to use for each question.",
    )
    parser.add_argument(
        "--output-json",
        help="Optional path for the benchmark JSON summary.",
    )
    parser.add_argument(
        "--output-markdown",
        help="Optional path for a Markdown benchmark report.",
    )
    parser.add_argument(
        "--compare-to",
        help="Optional path to a previous benchmark JSON summary for regression comparison.",
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit with status 1 when compared metrics regress.",
    )
    parser.add_argument(
        "--chunk-preset",
        choices=("small", "default", "large"),
        default="default",
        help="Chunking preset to apply while running the benchmark.",
    )
    parser.add_argument(
        "--chunk-target-words",
        type=int,
        help="Override the configured chunk target size in words.",
    )
    parser.add_argument(
        "--chunk-min-words",
        type=int,
        help="Override the minimum chunk size in words before splitting.",
    )
    parser.add_argument(
        "--chunk-overlap-words",
        type=int,
        help="Override the overlap carried into the next chunk.",
    )
    parser.add_argument(
        "--chunk-max-heading-words",
        type=int,
        help="Override the heading detector word-count ceiling.",
    )
    parser.add_argument(
        "--compare-presets",
        action="store_true",
        help="Run the benchmark across all built-in chunking presets and recommend one.",
    )
    return parser.parse_args()


CHUNKING_PRESETS: dict[str, dict[str, int]] = {
    "small": {
        "target_words": 320,
        "min_words": 140,
        "overlap_words": 48,
        "max_heading_words": 12,
    },
    "default": {
        "target_words": 420,
        "min_words": 180,
        "overlap_words": 64,
        "max_heading_words": 12,
    },
    "large": {
        "target_words": 640,
        "min_words": 260,
        "overlap_words": 96,
        "max_heading_words": 14,
    },
}


def resolve_question_case_option(
    *,
    benchmark: dict[str, Any],
    document_case: dict[str, Any],
    question_case: dict[str, Any],
    option_name: str,
    default: Any,
) -> Any:
    benchmark_defaults = benchmark.get("defaults", {})
    document_defaults = document_case.get("defaults", {})
    if option_name in question_case:
        return question_case[option_name]
    if option_name in document_defaults:
        return document_defaults[option_name]
    if option_name in benchmark_defaults:
        return benchmark_defaults[option_name]
    return default


def resolve_chunking_config(args: argparse.Namespace) -> dict[str, int]:
    config = dict(CHUNKING_PRESETS[args.chunk_preset])
    if args.chunk_target_words is not None:
        config["target_words"] = args.chunk_target_words
    if args.chunk_min_words is not None:
        config["min_words"] = args.chunk_min_words
    if args.chunk_overlap_words is not None:
        config["overlap_words"] = args.chunk_overlap_words
    if args.chunk_max_heading_words is not None:
        config["max_heading_words"] = args.chunk_max_heading_words
    return config


def validate_args(args: argparse.Namespace) -> None:
    if not args.compare_presets:
        return
    if args.compare_to:
        raise ValueError("--compare-to is not supported with --compare-presets.")
    if any(
        value is not None
        for value in (
            args.chunk_target_words,
            args.chunk_min_words,
            args.chunk_overlap_words,
            args.chunk_max_heading_words,
        )
    ):
        raise ValueError(
            "--compare-presets cannot be combined with explicit chunking overrides."
        )


def create_pdf(page_texts: list[str]) -> bytes:
    document = fitz.open()
    for text in page_texts:
        page = document.new_page()
        page.insert_textbox(fitz.Rect(72, 72, 523, 770), text, fontsize=11)
    payload = document.tobytes()
    document.close()
    return payload


def resolve_project_relative_path(raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    return Path(__file__).resolve().parent.parent / candidate


def load_benchmark_document_bytes(document_case: dict[str, Any]) -> bytes:
    source_pdf = document_case.get("source_pdf")
    if source_pdf:
        return resolve_project_relative_path(source_pdf).read_bytes()

    pages = document_case.get("pages")
    if pages:
        return create_pdf(pages)

    raise ValueError(
        "Each benchmark document must define either `pages` or `source_pdf`."
    )


def ensure_parent_directory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def expand_page_span(start_page: int, end_page: int) -> set[int]:
    if end_page < start_page:
        return {start_page}
    return set(range(start_page, end_page + 1))


def page_expectation_hit(
    *,
    expected_pages: set[int],
    actual_pages: set[int],
    match_mode: str,
) -> bool:
    if not expected_pages:
        return True
    if match_mode == "all":
        return expected_pages.issubset(actual_pages)
    return bool(actual_pages & expected_pages)


def text_expectation_hit(
    *,
    expected_texts: list[str],
    actual_texts: list[str],
    match_mode: str,
) -> bool:
    if not expected_texts:
        return True

    normalized_actual_texts = [_normalize_excerpt_text(text) for text in actual_texts if text]
    if not normalized_actual_texts:
        return False

    matches = [
        any(_normalize_excerpt_text(expected_text) in actual_text for actual_text in normalized_actual_texts)
        for expected_text in expected_texts
    ]
    if match_mode == "all":
        return all(matches)
    return any(matches)


def _normalize_excerpt_text(text: str) -> str:
    normalized = text.translate(EXCERPT_TRANSLATION_TABLE)
    normalized = EXCERPT_WHITESPACE_RE.sub(" ", normalized).strip().casefold()
    normalized = EXCERPT_SEPARATOR_DASH_RE.sub(" - ", normalized)
    normalized = EXCERPT_CURRENCY_RE.sub("$", normalized)
    normalized = EXCERPT_PERCENT_RE.sub("%", normalized)
    normalized = EXCERPT_OPEN_PAREN_RE.sub("(", normalized)
    normalized = EXCERPT_CLOSE_PAREN_RE.sub(")", normalized)
    return normalized


def percentile(values: list[float], percentile_rank: float) -> float:
    if not values:
        return 0.0
    if percentile_rank <= 0:
        return min(values)
    if percentile_rank >= 100:
        return max(values)

    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * (percentile_rank / 100)
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]

    lower_value = ordered[lower_index]
    upper_value = ordered[upper_index]
    weight = position - lower_index
    return lower_value + (upper_value - lower_value) * weight


def normalize_regression_tier(raw_tier: Any) -> str:
    tier = str(raw_tier).strip().casefold()
    if tier not in {"gating", "exploratory"}:
        raise ValueError(
            f"Unsupported regression_tier `{raw_tier}`. Expected `gating` or `exploratory`."
        )
    return tier


def build_question_result(
    *,
    document_name: str,
    question: str,
    expected_supported: bool,
    supported: bool,
    answer: str,
    expected_citation_pages: set[int],
    retrieved_pages: set[int],
    cited_pages: set[int],
    cited_texts: list[str],
    answer_terms: list[str],
    latency_ms: float,
    citation_match_mode: str,
    expected_citation_texts: list[str],
    citation_text_match_mode: str,
    regression_tier: str,
    trace_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    retrieval_hit = page_expectation_hit(
        expected_pages=expected_citation_pages,
        actual_pages=retrieved_pages,
        match_mode=citation_match_mode,
    )
    citation_hit = page_expectation_hit(
        expected_pages=expected_citation_pages,
        actual_pages=cited_pages,
        match_mode=citation_match_mode,
    )
    citation_text_hit = text_expectation_hit(
        expected_texts=expected_citation_texts,
        actual_texts=cited_texts,
        match_mode=citation_text_match_mode,
    )
    support_hit = supported == expected_supported
    answer_hit = not answer_terms or any(term in answer.lower() for term in answer_terms)

    result = {
        "document": document_name,
        "question": question,
        "expected_supported": expected_supported,
        "supported": supported,
        "regression_tier": regression_tier,
        "citation_match_mode": citation_match_mode,
        "citation_text_match_mode": citation_text_match_mode,
        "expected_citation_pages": sorted(expected_citation_pages),
        "expected_citation_text_contains": expected_citation_texts,
        "retrieved_pages": sorted(retrieved_pages),
        "cited_pages": sorted(cited_pages),
        "cited_texts": cited_texts,
        "retrieval_hit": retrieval_hit,
        "citation_hit": citation_hit,
        "citation_text_hit": citation_text_hit,
        "support_hit": support_hit,
        "answer_hit": answer_hit,
        "latency_ms": latency_ms,
        "answer_preview": answer[:240],
    }
    if trace_summary:
        result.update(trace_summary)
    return result


def question_result_failed(result: dict[str, Any]) -> bool:
    return not (
        result["retrieval_hit"]
        and result["citation_hit"]
        and result["citation_text_hit"]
        and result["support_hit"]
        and result["answer_hit"]
    )


def build_failure_trace(
    *,
    payload: dict[str, Any],
    trace: dict[str, Any],
) -> dict[str, Any]:
    return {
        "normalized_question": payload.get("normalized_question"),
        "answer": payload.get("answer"),
        "supported": payload.get("supported"),
        "answer_mode": payload.get("answer_mode"),
        "confidence": payload.get("confidence"),
        "support_score": payload.get("support_score"),
        "citations": payload.get("citations", []),
        "question_router": trace.get("question_router"),
        "runtime": trace.get("runtime"),
        "verification": payload.get("verification") or trace.get("verification"),
        "postprocess": trace.get("postprocess"),
        "retrieved_chunks": trace.get("retrieved_chunks", []),
        "selected_contexts": trace.get("selected_contexts", []),
        "cited_contexts": trace.get("cited_contexts", []),
        "timings": trace.get("timings"),
        "prompt_snapshot": trace.get("prompt_snapshot"),
    }


def summarize_results(
    *,
    benchmark_name: str,
    benchmark_path: Path,
    top_k: int,
    chunking_config: dict[str, int],
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    total_questions = len(results)
    gating_results = [
        result for result in results if result.get("regression_tier", "gating") == "gating"
    ]
    exploratory_results = [
        result
        for result in results
        if result.get("regression_tier", "gating") == "exploratory"
    ]
    latencies_ms = [float(result["latency_ms"]) for result in results]
    predicted_unsupported = [result for result in results if not result["supported"]]
    unsupported_true_positives = [
        result for result in predicted_unsupported if not result["expected_supported"]
    ]
    unsupported_precision = (
        len(unsupported_true_positives) / len(predicted_unsupported)
        if predicted_unsupported
        else None
    )
    document_summaries: list[dict[str, Any]] = []
    for document_name in sorted({result["document"] for result in results}):
        document_results = [
            result for result in results if result["document"] == document_name
        ]
        document_gating_results = [
            result
            for result in document_results
            if result.get("regression_tier", "gating") == "gating"
        ]
        document_exploratory_results = [
            result
            for result in document_results
            if result.get("regression_tier", "gating") == "exploratory"
        ]
        document_summaries.append(
            {
                "document": document_name,
                "questions": len(document_results),
                "gating_questions": len(document_gating_results),
                "exploratory_questions": len(document_exploratory_results),
                "gating_failures": sum(
                    1
                    for result in document_gating_results
                    if not (
                        result["retrieval_hit"]
                        and result["citation_hit"]
                        and result["citation_text_hit"]
                        and result["support_hit"]
                        and result["answer_hit"]
                    )
                ),
                "exploratory_failures": sum(
                    1
                    for result in document_exploratory_results
                    if not (
                        result["retrieval_hit"]
                        and result["citation_hit"]
                        and result["citation_text_hit"]
                        and result["support_hit"]
                        and result["answer_hit"]
                    )
                ),
            }
        )

    failure_results = [result for result in results if question_result_failed(result)]
    gating_failure_results = [
        result
        for result in failure_results
        if result.get("regression_tier", "gating") == "gating"
    ]
    exploratory_failure_results = [
        result
        for result in failure_results
        if result.get("regression_tier", "gating") == "exploratory"
    ]

    return {
        "benchmark": benchmark_name,
        "benchmark_path": str(benchmark_path),
        "generated_at": datetime.now(UTC).isoformat(),
        "top_k": top_k,
        "chunking_config": chunking_config,
        "questions": total_questions,
        "gating_questions": len(gating_results),
        "exploratory_questions": len(exploratory_results),
        "retrieval_hit_rate": rate(results, "retrieval_hit"),
        "citation_hit_rate": rate(results, "citation_hit"),
        "citation_evidence_hit_rate": rate(results, "citation_text_hit"),
        "gating_citation_evidence_hit_rate": rate_or_none(
            gating_results,
            "citation_text_hit",
        ),
        "support_accuracy": rate(results, "support_hit"),
        "answer_match_rate": rate(results, "answer_hit"),
        "unsupported_precision": unsupported_precision,
        "average_latency_ms": mean(latencies_ms) if latencies_ms else 0.0,
        "latency_p50_ms": percentile(latencies_ms, 50),
        "latency_p95_ms": percentile(latencies_ms, 95),
        "predicted_unsupported_count": len(predicted_unsupported),
        "expected_unsupported_count": sum(
            1 for result in results if not result["expected_supported"]
        ),
        "failure_count": len(failure_results),
        "gating_failure_count": len(gating_failure_results),
        "exploratory_failure_count": len(exploratory_failure_results),
        "document_summaries": document_summaries,
        "results": results,
        "failure_results": failure_results,
    }


def summarize_preset_comparison(
    *,
    benchmark_name: str,
    benchmark_path: Path,
    top_k: int,
    summaries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    recommended_preset = recommend_chunking_preset(summaries)
    return {
        "benchmark": benchmark_name,
        "benchmark_path": str(benchmark_path),
        "generated_at": datetime.now(UTC).isoformat(),
        "top_k": top_k,
        "preset_summaries": summaries,
        "recommended_preset": recommended_preset,
        "recommended_chunking_config": summaries[recommended_preset]["chunking_config"],
        "decision_basis": (
            "Choose the preset with the strongest quality metrics; break exact ties with "
            "lower average latency and then lower p95 latency."
        ),
    }


def rate(results: list[dict[str, Any]], field_name: str) -> float:
    if not results:
        return 0.0
    return sum(1 for result in results if result[field_name]) / len(results)


def rate_or_none(results: list[dict[str, Any]], field_name: str) -> float | None:
    if not results:
        return None
    return rate(results, field_name)


def compare_summaries(
    current_summary: dict[str, Any],
    baseline_summary: dict[str, Any],
) -> dict[str, Any]:
    metric_deltas: dict[str, dict[str, float | bool | None]] = {}
    regressions: list[str] = []

    for metric_name in SUMMARY_QUALITY_METRICS:
        current_value = current_summary.get(metric_name)
        baseline_value = baseline_summary.get(metric_name)
        delta = metric_delta(current_value, baseline_value)
        regressed = (
            metric_name in REGRESSION_QUALITY_METRICS
            and delta is not None
            and baseline_value is not None
            and current_value is not None
            and delta < 0
        )
        metric_deltas[metric_name] = {
            "current": current_value,
            "baseline": baseline_value,
            "delta": delta,
            "regressed": regressed,
        }
        if regressed:
            regressions.append(metric_name)

    for metric_name in LATENCY_METRICS:
        current_value = current_summary.get(metric_name)
        baseline_value = baseline_summary.get(metric_name)
        delta = metric_delta(current_value, baseline_value)
        regressed = (
            delta is not None
            and baseline_value is not None
            and current_value is not None
            and delta >= LATENCY_REGRESSION_BUFFER_MS
        )
        metric_deltas[metric_name] = {
            "current": current_value,
            "baseline": baseline_value,
            "delta": delta,
            "regressed": regressed,
        }
        if regressed:
            regressions.append(metric_name)

    return {
        "baseline_benchmark": baseline_summary.get("benchmark"),
        "baseline_generated_at": baseline_summary.get("generated_at"),
        "metrics": metric_deltas,
        "regressions": regressions,
        "has_regressions": bool(regressions),
    }


def metric_delta(current_value: float | None, baseline_value: float | None) -> float | None:
    if current_value is None or baseline_value is None:
        return None
    return current_value - baseline_value


def render_markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        f"# Benchmark Report: {summary['benchmark']}",
        "",
        f"- Generated at: `{summary['generated_at']}`",
        f"- Benchmark file: `{summary['benchmark_path']}`",
        f"- Top-k: `{summary['top_k']}`",
        f"- Chunking config: `{json.dumps(summary['chunking_config'], sort_keys=True)}`",
        f"- Questions: `{summary['questions']}`",
        f"- Gating questions: `{summary['gating_questions']}`",
        f"- Exploratory questions: `{summary['exploratory_questions']}`",
        "",
        "## Metrics",
        "",
        f"- Retrieval hit rate: `{format_metric(summary['retrieval_hit_rate'])}`",
        f"- Citation hit rate: `{format_metric(summary['citation_hit_rate'])}`",
        f"- Citation evidence hit rate: `{format_metric(summary['citation_evidence_hit_rate'])}`",
        (
            "- Gating citation evidence hit rate: "
            f"`{format_metric(summary['gating_citation_evidence_hit_rate'])}`"
            if summary["gating_citation_evidence_hit_rate"] is not None
            else "- Gating citation evidence hit rate: `n/a`"
        ),
        f"- Support accuracy: `{format_metric(summary['support_accuracy'])}`",
        f"- Answer match rate: `{format_metric(summary['answer_match_rate'])}`",
        (
            f"- Unsupported precision: `{format_metric(summary['unsupported_precision'])}`"
            if summary["unsupported_precision"] is not None
            else "- Unsupported precision: `n/a`"
        ),
        f"- Average latency: `{summary['average_latency_ms']:.2f} ms`",
        f"- P50 latency: `{summary['latency_p50_ms']:.2f} ms`",
        f"- P95 latency: `{summary['latency_p95_ms']:.2f} ms`",
        f"- Failure count: `{summary.get('failure_count', 0)}`",
        f"- Gating failure count: `{summary.get('gating_failure_count', 0)}`",
        f"- Exploratory failure count: `{summary.get('exploratory_failure_count', 0)}`",
    ]

    document_summaries = summary.get("document_summaries", [])
    if document_summaries:
        lines.extend(["", "## Documents", ""])
        for document_summary in document_summaries:
            lines.append(
                "- "
                + f"{document_summary['document']}: "
                + f"questions={document_summary['questions']}, "
                + f"gating={document_summary['gating_questions']} "
                + f"(failures={document_summary['gating_failures']}), "
                + f"exploratory={document_summary['exploratory_questions']} "
                + f"(failures={document_summary['exploratory_failures']})"
            )

    comparison = summary.get("comparison")
    if comparison:
        lines.extend(
            [
                "",
                "## Comparison",
                "",
                f"- Baseline benchmark: `{comparison.get('baseline_benchmark')}`",
                f"- Baseline generated at: `{comparison.get('baseline_generated_at')}`",
                f"- Regressions detected: `{'yes' if comparison['has_regressions'] else 'no'}`",
            ]
        )
        for metric_name, details in comparison["metrics"].items():
            delta = details["delta"]
            if delta is None:
                lines.append(f"- {metric_name}: `n/a`")
                continue
            lines.append(
                f"- {metric_name}: `{format_signed_metric(delta, metric_name in LATENCY_METRICS)}`"
            )

    gating_failures = [
        result
        for result in summary.get("failure_results", summary["results"])
        if result.get("regression_tier", "gating") == "gating"
    ]
    exploratory_failures = [
        result
        for result in summary.get("failure_results", summary["results"])
        if result.get("regression_tier", "gating") == "exploratory"
    ]
    if gating_failures:
        lines.extend(["", "## Gating Failures", ""])
        for result in gating_failures:
            lines.append(
                "- "
                + f"{result['document']} :: {result['question']} "
                + f"(retrieval={result['retrieval_hit']}, citation={result['citation_hit']}, "
                + f"citation_text={result['citation_text_hit']}, support={result['support_hit']}, "
                + f"answer={result['answer_hit']}, "
                + f"expected_citation_text={result['expected_citation_text_contains']}, "
                + f"latency_ms={result['latency_ms']:.2f})"
            )
            if result.get("failure_trace"):
                lines.append(
                    "  trace: "
                    + f"answer_mode={result.get('answer_mode')}, "
                    + f"router={result.get('router_answer_mode')}, "
                    + f"heuristic_support={format_metric(result.get('router_heuristic_support_score'))}, "
                    + f"coverage={format_metric(result.get('question_coverage_score'))}, "
                    + f"repair_attempted={result.get('repair_attempted')}, "
                    + f"repair_applied={result.get('repair_applied')}"
                )
                if result.get("unsupported_classifier_reason"):
                    lines.append(
                        "  unsupported_classifier_reason: "
                        + str(result["unsupported_classifier_reason"])
                    )
                if result.get("repair_reason"):
                    lines.append("  repair_reason: " + str(result["repair_reason"]))

    if exploratory_failures:
        lines.extend(["", "## Exploratory Failures", ""])
        for result in exploratory_failures:
            lines.append(
                "- "
                + f"{result['document']} :: {result['question']} "
                + f"(retrieval={result['retrieval_hit']}, citation={result['citation_hit']}, "
                + f"citation_text={result['citation_text_hit']}, support={result['support_hit']}, "
                + f"answer={result['answer_hit']}, "
                + f"expected_citation_text={result['expected_citation_text_contains']}, "
                + f"latency_ms={result['latency_ms']:.2f})"
            )
            if result.get("failure_trace"):
                lines.append(
                    "  trace: "
                    + f"answer_mode={result.get('answer_mode')}, "
                    + f"router={result.get('router_answer_mode')}, "
                    + f"heuristic_support={format_metric(result.get('router_heuristic_support_score'))}, "
                    + f"coverage={format_metric(result.get('question_coverage_score'))}, "
                    + f"repair_attempted={result.get('repair_attempted')}, "
                    + f"repair_applied={result.get('repair_applied')}"
                )
                if result.get("unsupported_classifier_reason"):
                    lines.append(
                        "  unsupported_classifier_reason: "
                        + str(result["unsupported_classifier_reason"])
                    )
                if result.get("repair_reason"):
                    lines.append("  repair_reason: " + str(result["repair_reason"]))

    return "\n".join(lines) + "\n"


def format_metric(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def format_signed_metric(delta: float, latency_metric: bool) -> str:
    prefix = "+" if delta >= 0 else ""
    suffix = " ms" if latency_metric else ""
    return f"{prefix}{delta:.3f}{suffix}"


def recommend_chunking_preset(summaries: dict[str, dict[str, Any]]) -> str:
    def ranking_key(item: tuple[str, dict[str, Any]]) -> tuple[float, ...]:
        _, summary = item
        quality_values = tuple(
            float(summary[metric]) for metric in REGRESSION_QUALITY_METRICS
        )
        latency_values = (
            -float(summary["average_latency_ms"]),
            -float(summary["latency_p95_ms"]),
        )
        return quality_values + latency_values

    return max(summaries.items(), key=ranking_key)[0]


def render_preset_comparison_report(comparison: dict[str, Any]) -> str:
    lines = [
        f"# Chunking Preset Comparison: {comparison['benchmark']}",
        "",
        f"- Generated at: `{comparison['generated_at']}`",
        f"- Benchmark file: `{comparison['benchmark_path']}`",
        f"- Top-k: `{comparison['top_k']}`",
        f"- Recommended preset: `{comparison['recommended_preset']}`",
        (
            "- Recommended config: "
            f"`{json.dumps(comparison['recommended_chunking_config'], sort_keys=True)}`"
        ),
        f"- Decision basis: {comparison['decision_basis']}",
        "",
        "## Presets",
        "",
    ]

    for preset_name, summary in comparison["preset_summaries"].items():
        lines.extend(
            [
                f"### {preset_name}",
                "",
                f"- Chunking config: `{json.dumps(summary['chunking_config'], sort_keys=True)}`",
                f"- Retrieval hit rate: `{format_metric(summary['retrieval_hit_rate'])}`",
                f"- Citation hit rate: `{format_metric(summary['citation_hit_rate'])}`",
                f"- Citation evidence hit rate: `{format_metric(summary['citation_evidence_hit_rate'])}`",
                (
                    "- Gating citation evidence hit rate: "
                    f"`{format_metric(summary['gating_citation_evidence_hit_rate'])}`"
                    if summary["gating_citation_evidence_hit_rate"] is not None
                    else "- Gating citation evidence hit rate: `n/a`"
                ),
                f"- Support accuracy: `{format_metric(summary['support_accuracy'])}`",
                f"- Answer match rate: `{format_metric(summary['answer_match_rate'])}`",
                (
                    f"- Unsupported precision: `{format_metric(summary['unsupported_precision'])}`"
                    if summary["unsupported_precision"] is not None
                    else "- Unsupported precision: `n/a`"
                ),
                f"- Average latency: `{summary['average_latency_ms']:.2f} ms`",
                f"- P50 latency: `{summary['latency_p50_ms']:.2f} ms`",
                f"- P95 latency: `{summary['latency_p95_ms']:.2f} ms`",
                "",
            ]
        )

    return "\n".join(lines).rstrip() + "\n"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent_directory(path)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def write_text(path: Path, content: str) -> None:
    ensure_parent_directory(path)
    path.write_text(content)


def run_benchmark(
    *,
    benchmark_path: Path,
    benchmark: dict[str, Any],
    top_k: int,
    chunking_config: dict[str, int],
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="ebook-rag-eval-") as temp_dir:
        temp_root = Path(temp_dir)
        os.environ["DATABASE_URL"] = f"sqlite:///{temp_root / 'eval.db'}"
        os.environ["UPLOADS_DIR"] = str(temp_root / "uploads")
        os.environ["MAX_UPLOAD_SIZE_MB"] = "5"
        os.environ["CHUNK_TARGET_WORDS"] = str(chunking_config["target_words"])
        os.environ["CHUNK_MIN_WORDS"] = str(chunking_config["min_words"])
        os.environ["CHUNK_OVERLAP_WORDS"] = str(chunking_config["overlap_words"])
        os.environ["CHUNK_MAX_HEADING_WORDS"] = str(
            chunking_config["max_heading_words"]
        )

        from ebook_rag_api.core.config import get_settings
        from ebook_rag_api.db import Base, get_engine, get_session_factory
        from ebook_rag_api.main import create_app
        from ebook_rag_api.services.embeddings import get_embedding_provider
        from ebook_rag_api.services.qa import get_answer_provider
        from ebook_rag_api.services.reranking import get_reranker
        from fastapi.testclient import TestClient

        get_settings.cache_clear()
        get_engine.cache_clear()
        get_session_factory.cache_clear()
        get_embedding_provider.cache_clear()
        get_answer_provider.cache_clear()
        get_reranker.cache_clear()

        app = create_app()
        Base.metadata.create_all(bind=get_engine())

        results: list[dict[str, Any]] = []

        with TestClient(app) as client:
            uploaded_documents: dict[str, str] = {}
            for document_case in benchmark["documents"]:
                response = client.post(
                    "/api/documents/upload",
                    files={
                        "file": (
                            document_case["filename"],
                            load_benchmark_document_bytes(document_case),
                            "application/pdf",
                        )
                    },
                )
                response.raise_for_status()
                payload = response.json()
                uploaded_documents[document_case["filename"]] = payload["document"]["id"]

            for document_case in benchmark["documents"]:
                document_id = uploaded_documents[document_case["filename"]]
                for question_case in document_case["questions"]:
                    response = client.post(
                        "/api/qa/ask",
                        json={
                            "question": question_case["question"],
                            "document_id": document_id,
                            "top_k": top_k,
                            "include_trace": True,
                        },
                    )
                    response.raise_for_status()
                    payload = response.json()
                    trace = payload["trace"]
                    trace_postprocess = trace.get("postprocess") or {}

                    expected_pages = set(question_case.get("expected_citation_pages", []))
                    retrieved_pages = set().union(
                        *[
                            expand_page_span(
                                int(chunk["page_start"]),
                                int(chunk.get("page_end", chunk["page_start"])),
                            )
                            for chunk in trace["retrieved_chunks"]
                        ]
                    )
                    cited_pages = set().union(
                        *[
                            expand_page_span(
                                int(citation["page_start"]),
                                int(citation.get("page_end", citation["page_start"])),
                            )
                            for citation in payload["citations"]
                        ]
                    )
                    cited_texts = [
                        str(citation.get("text", "")).strip()
                        for citation in payload["citations"]
                        if str(citation.get("text", "")).strip()
                    ]
                    answer_terms = [
                        term.lower()
                        for term in question_case.get("expected_answer_contains", [])
                    ]
                    expected_citation_texts = [
                        str(term).strip()
                        for term in question_case.get("expected_citation_text_contains", [])
                        if str(term).strip()
                    ]
                    citation_text_match_mode = question_case.get(
                        "citation_text_match_mode",
                        "all" if len(expected_citation_texts) > 1 else "any",
                    )
                    regression_tier = normalize_regression_tier(
                        resolve_question_case_option(
                            benchmark=benchmark,
                            document_case=document_case,
                            question_case=question_case,
                            option_name="regression_tier",
                            default="gating",
                        )
                    )
                    results.append(
                        build_question_result(
                            document_name=document_case["filename"],
                            question=question_case["question"],
                            expected_supported=question_case["expected_supported"],
                            supported=payload["supported"],
                            answer=payload["answer"],
                            expected_citation_pages=expected_pages,
                            retrieved_pages=retrieved_pages,
                            cited_pages=cited_pages,
                            cited_texts=cited_texts,
                            answer_terms=answer_terms,
                            latency_ms=float(trace["timings"]["total_ms"]),
                            citation_match_mode=question_case.get(
                                "citation_match_mode",
                                "any",
                            ),
                            expected_citation_texts=expected_citation_texts,
                            citation_text_match_mode=citation_text_match_mode,
                            regression_tier=regression_tier,
                            trace_summary={
                                "answer_mode": payload.get("answer_mode"),
                                "router_answer_mode": trace["question_router"].get("answer_mode"),
                                "router_reason": trace["question_router"].get("reason"),
                                "router_heuristic_support_score": trace["question_router"].get(
                                    "heuristic_support_score"
                                ),
                                "unsupported_classifier_ran": trace["question_router"].get(
                                    "unsupported_classifier_ran"
                                ),
                                "unsupported_classifier_supported": trace["question_router"].get(
                                    "unsupported_classifier_supported"
                                ),
                                "unsupported_classifier_reason": trace["question_router"].get(
                                    "unsupported_classifier_reason"
                                ),
                                "question_coverage_score": trace_postprocess.get("question_coverage_score"),
                                "support_threshold": trace_postprocess.get("support_threshold"),
                                "repair_attempted": trace_postprocess.get("repair_attempted"),
                                "repair_applied": trace_postprocess.get("repair_applied"),
                                "repair_reason": trace_postprocess.get("repair_reason"),
                            },
                        )
                    )
                    latest_result = results[-1]
                    if question_result_failed(latest_result):
                        latest_result["failure_trace"] = build_failure_trace(
                            payload=payload,
                            trace=trace,
                        )

    return summarize_results(
        benchmark_name=benchmark["name"],
        benchmark_path=benchmark_path,
        top_k=top_k,
        chunking_config=chunking_config,
        results=results,
    )


def main() -> int:
    args = parse_args()
    validate_args(args)
    benchmark_path = resolve_project_relative_path(args.benchmark)
    benchmark = json.loads(benchmark_path.read_text())

    if args.compare_presets:
        payload = summarize_preset_comparison(
            benchmark_name=benchmark["name"],
            benchmark_path=benchmark_path,
            top_k=args.top_k,
            summaries={
                preset_name: run_benchmark(
                    benchmark_path=benchmark_path,
                    benchmark=benchmark,
                    top_k=args.top_k,
                    chunking_config=dict(config),
                )
                for preset_name, config in CHUNKING_PRESETS.items()
            },
        )
    else:
        payload = run_benchmark(
            benchmark_path=benchmark_path,
            benchmark=benchmark,
            top_k=args.top_k,
            chunking_config=resolve_chunking_config(args),
        )

    if args.compare_to:
        baseline_path = resolve_project_relative_path(args.compare_to)
        baseline_summary = json.loads(baseline_path.read_text())
        payload["comparison"] = compare_summaries(payload, baseline_summary)

    if args.output_json:
        write_json(resolve_project_relative_path(args.output_json), payload)

    if args.output_markdown:
        report = (
            render_preset_comparison_report(payload)
            if args.compare_presets
            else render_markdown_report(payload)
        )
        write_text(resolve_project_relative_path(args.output_markdown), report)

    print(json.dumps(payload, indent=2))

    if args.fail_on_regression and payload.get("comparison", {}).get("has_regressions"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
