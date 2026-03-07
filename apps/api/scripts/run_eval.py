from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

import fitz

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

QUALITY_METRICS = (
    "retrieval_hit_rate",
    "citation_hit_rate",
    "support_accuracy",
    "answer_match_rate",
    "unsupported_precision",
)
LATENCY_METRICS = (
    "average_latency_ms",
    "latency_p50_ms",
    "latency_p95_ms",
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
    return parser.parse_args()


def create_pdf(page_texts: list[str]) -> bytes:
    document = fitz.open()
    for text in page_texts:
        page = document.new_page()
        page.insert_text((72, 72), text)
    payload = document.tobytes()
    document.close()
    return payload


def resolve_project_relative_path(raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    return Path(__file__).resolve().parent.parent / candidate


def ensure_parent_directory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def expand_page_span(start_page: int, end_page: int) -> set[int]:
    if end_page < start_page:
        return {start_page}
    return set(range(start_page, end_page + 1))


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
    answer_terms: list[str],
    latency_ms: float,
) -> dict[str, Any]:
    retrieval_hit = not expected_citation_pages or bool(retrieved_pages & expected_citation_pages)
    citation_hit = not expected_citation_pages or bool(cited_pages & expected_citation_pages)
    support_hit = supported == expected_supported
    answer_hit = not answer_terms or any(term in answer.lower() for term in answer_terms)

    return {
        "document": document_name,
        "question": question,
        "expected_supported": expected_supported,
        "supported": supported,
        "expected_citation_pages": sorted(expected_citation_pages),
        "retrieved_pages": sorted(retrieved_pages),
        "cited_pages": sorted(cited_pages),
        "retrieval_hit": retrieval_hit,
        "citation_hit": citation_hit,
        "support_hit": support_hit,
        "answer_hit": answer_hit,
        "latency_ms": latency_ms,
        "answer_preview": answer[:240],
    }


def summarize_results(
    *,
    benchmark_name: str,
    benchmark_path: Path,
    top_k: int,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    total_questions = len(results)
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

    return {
        "benchmark": benchmark_name,
        "benchmark_path": str(benchmark_path),
        "generated_at": datetime.now(UTC).isoformat(),
        "top_k": top_k,
        "questions": total_questions,
        "retrieval_hit_rate": rate(results, "retrieval_hit"),
        "citation_hit_rate": rate(results, "citation_hit"),
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
        "results": results,
    }


def rate(results: list[dict[str, Any]], field_name: str) -> float:
    if not results:
        return 0.0
    return sum(1 for result in results if result[field_name]) / len(results)


def compare_summaries(
    current_summary: dict[str, Any],
    baseline_summary: dict[str, Any],
) -> dict[str, Any]:
    metric_deltas: dict[str, dict[str, float | bool | None]] = {}
    regressions: list[str] = []

    for metric_name in QUALITY_METRICS:
        current_value = current_summary.get(metric_name)
        baseline_value = baseline_summary.get(metric_name)
        delta = metric_delta(current_value, baseline_value)
        regressed = (
            delta is not None
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
            and delta > 0
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
        f"- Questions: `{summary['questions']}`",
        "",
        "## Metrics",
        "",
        f"- Retrieval hit rate: `{format_metric(summary['retrieval_hit_rate'])}`",
        f"- Citation hit rate: `{format_metric(summary['citation_hit_rate'])}`",
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
    ]

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

    failing_results = [
        result
        for result in summary["results"]
        if not (
            result["retrieval_hit"]
            and result["citation_hit"]
            and result["support_hit"]
            and result["answer_hit"]
        )
    ]
    if failing_results:
        lines.extend(["", "## Failures", ""])
        for result in failing_results:
            lines.append(
                "- "
                + f"{result['document']} :: {result['question']} "
                + f"(retrieval={result['retrieval_hit']}, citation={result['citation_hit']}, "
                + f"support={result['support_hit']}, answer={result['answer_hit']}, "
                + f"latency_ms={result['latency_ms']:.2f})"
            )

    return "\n".join(lines) + "\n"


def format_metric(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def format_signed_metric(delta: float, latency_metric: bool) -> str:
    prefix = "+" if delta >= 0 else ""
    suffix = " ms" if latency_metric else ""
    return f"{prefix}{delta:.3f}{suffix}"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent_directory(path)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def write_text(path: Path, content: str) -> None:
    ensure_parent_directory(path)
    path.write_text(content)


def main() -> int:
    args = parse_args()
    benchmark_path = resolve_project_relative_path(args.benchmark)
    benchmark = json.loads(benchmark_path.read_text())

    with tempfile.TemporaryDirectory(prefix="ebook-rag-eval-") as temp_dir:
        temp_root = Path(temp_dir)
        os.environ["DATABASE_URL"] = f"sqlite:///{temp_root / 'eval.db'}"
        os.environ["UPLOADS_DIR"] = str(temp_root / "uploads")
        os.environ["MAX_UPLOAD_SIZE_MB"] = "5"

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
                            create_pdf(document_case["pages"]),
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
                            "top_k": args.top_k,
                            "include_trace": True,
                        },
                    )
                    response.raise_for_status()
                    payload = response.json()
                    trace = payload["trace"]

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
                    answer_terms = [
                        term.lower()
                        for term in question_case.get("expected_answer_contains", [])
                    ]
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
                            answer_terms=answer_terms,
                            latency_ms=float(trace["timings"]["total_ms"]),
                        )
                    )

    summary = summarize_results(
        benchmark_name=benchmark["name"],
        benchmark_path=benchmark_path,
        top_k=args.top_k,
        results=results,
    )

    if args.compare_to:
        baseline_path = resolve_project_relative_path(args.compare_to)
        baseline_summary = json.loads(baseline_path.read_text())
        summary["comparison"] = compare_summaries(summary, baseline_summary)

    if args.output_json:
        write_json(resolve_project_relative_path(args.output_json), summary)

    if args.output_markdown:
        write_text(
            resolve_project_relative_path(args.output_markdown),
            render_markdown_report(summary),
        )

    print(json.dumps(summary, indent=2))

    if args.fail_on_regression and summary.get("comparison", {}).get("has_regressions"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
