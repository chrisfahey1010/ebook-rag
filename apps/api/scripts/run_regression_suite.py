from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any


def load_run_eval_module():
    script_path = Path(__file__).resolve().parent / "run_eval.py"
    spec = importlib.util.spec_from_file_location("run_eval", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {script_path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run_eval = load_run_eval_module()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the committed regression benchmark suite and compare against saved baselines."
    )
    parser.add_argument(
        "--suite",
        default="benchmarks/regression_suite.json",
        help="Path to the regression-suite manifest relative to apps/api or absolute.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-k retrieval depth to use for each benchmark run.",
    )
    parser.add_argument(
        "--chunk-preset",
        choices=tuple(run_eval.CHUNKING_PRESETS.keys()),
        default="default",
        help="Chunking preset to apply while running the suite.",
    )
    parser.add_argument(
        "--write-baselines",
        action="store_true",
        help="Refresh the committed baseline JSON/Markdown artifacts instead of comparing to them.",
    )
    return parser.parse_args()


def load_suite(path: Path) -> dict[str, Any]:
    suite = json.loads(path.read_text())
    entries = suite.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Regression suite manifest must define a non-empty `entries` list.")

    required_fields = (
        "name",
        "benchmark",
        "baseline_json",
        "baseline_markdown",
        "latest_json",
        "latest_markdown",
    )
    for entry in entries:
        missing_fields = [field for field in required_fields if field not in entry]
        if missing_fields:
            missing = ", ".join(missing_fields)
            raise ValueError(
                f"Regression suite entry is missing required fields: {missing}."
            )
    return suite


def resolve_entry_paths(entry: dict[str, str]) -> dict[str, Path]:
    return {
        key: run_eval.resolve_project_relative_path(entry[key])
        for key in (
            "benchmark",
            "baseline_json",
            "baseline_markdown",
            "latest_json",
            "latest_markdown",
        )
    }


def load_baseline_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing regression baseline: {path}. "
            "Run `uv run python scripts/run_regression_suite.py --write-baselines` to create it."
        )
    return json.loads(path.read_text())


def render_suite_markdown(
    *,
    suite_name: str,
    suite_path: Path,
    chunking_config: dict[str, int],
    write_baselines: bool,
    entries: list[dict[str, Any]],
) -> str:
    lines = [
        f"# Regression Suite Report: {suite_name}",
        "",
        f"- Suite file: `{suite_path}`",
        f"- Chunking config: `{json.dumps(chunking_config, sort_keys=True)}`",
        f"- Mode: `{'write-baselines' if write_baselines else 'compare-to-baseline'}`",
        "",
        "## Entries",
        "",
    ]

    for entry in entries:
        lines.extend(
            [
                f"### {entry['name']}",
                "",
                f"- Benchmark: `{entry['benchmark']}`",
                f"- Result: `{'baseline refreshed' if write_baselines else ('regression' if entry['has_regressions'] else 'ok')}`",
                f"- Retrieval hit rate: `{run_eval.format_metric(entry['summary']['retrieval_hit_rate'])}`",
                f"- Citation hit rate: `{run_eval.format_metric(entry['summary']['citation_hit_rate'])}`",
                (
                    "- Gating citation evidence hit rate: "
                    f"`{run_eval.format_metric(entry['summary']['gating_citation_evidence_hit_rate'])}`"
                ),
                f"- Support accuracy: `{run_eval.format_metric(entry['summary']['support_accuracy'])}`",
                f"- Answer match rate: `{run_eval.format_metric(entry['summary']['answer_match_rate'])}`",
                (
                    f"- Unsupported precision: `{run_eval.format_metric(entry['summary']['unsupported_precision'])}`"
                    if entry["summary"]["unsupported_precision"] is not None
                    else "- Unsupported precision: `n/a`"
                ),
                f"- Average latency: `{entry['summary']['average_latency_ms']:.2f} ms`",
                f"- Latest JSON: `{entry['latest_json']}`",
                "",
            ]
        )
        if not write_baselines:
            regressions = entry["comparison"].get("regressions", [])
            lines.append(
                f"- Baseline JSON: `{entry['baseline_json']}`"
            )
            lines.append(
                f"- Regressions: `{', '.join(regressions) if regressions else 'none'}`"
            )
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def run_suite(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    suite_path = run_eval.resolve_project_relative_path(args.suite)
    suite = load_suite(suite_path)
    chunking_config = dict(run_eval.CHUNKING_PRESETS[args.chunk_preset])
    results: list[dict[str, Any]] = []
    exit_code = 0

    for entry in suite["entries"]:
        paths = resolve_entry_paths(entry)
        benchmark = json.loads(paths["benchmark"].read_text())
        summary = run_eval.run_benchmark(
            benchmark_path=paths["benchmark"],
            benchmark=benchmark,
            top_k=args.top_k,
            chunking_config=chunking_config,
        )

        result_entry: dict[str, Any] = {
            "name": entry["name"],
            "benchmark": str(paths["benchmark"]),
            "baseline_json": str(paths["baseline_json"]),
            "latest_json": str(paths["latest_json"]),
            "summary": summary,
        }

        if args.write_baselines:
            run_eval.write_json(paths["latest_json"], summary)
            run_eval.write_text(
                paths["latest_markdown"], run_eval.render_markdown_report(summary)
            )
            run_eval.write_json(paths["baseline_json"], summary)
            run_eval.write_text(
                paths["baseline_markdown"], run_eval.render_markdown_report(summary)
            )
            result_entry["has_regressions"] = False
        else:
            baseline_summary = load_baseline_summary(paths["baseline_json"])
            comparison = run_eval.compare_summaries(summary, baseline_summary)
            result_entry["comparison"] = comparison
            result_entry["has_regressions"] = comparison["has_regressions"]
            if comparison["has_regressions"]:
                exit_code = 1

            summary_with_comparison = dict(summary)
            summary_with_comparison["comparison"] = comparison
            run_eval.write_json(paths["latest_json"], summary_with_comparison)
            run_eval.write_text(
                paths["latest_markdown"],
                run_eval.render_markdown_report(summary_with_comparison),
            )
            result_entry["summary"] = summary_with_comparison

        results.append(result_entry)

    payload = {
        "suite": suite["name"],
        "suite_path": str(suite_path),
        "generated_at": run_eval.datetime.now(run_eval.UTC).isoformat(),
        "chunking_config": chunking_config,
        "write_baselines": args.write_baselines,
        "entries": results,
        "has_regressions": any(entry.get("has_regressions", False) for entry in results),
    }
    report_path = suite_path.with_name("results").joinpath("regression_suite_latest.md")
    json_path = suite_path.with_name("results").joinpath("regression_suite_latest.json")
    run_eval.write_json(json_path, payload)
    run_eval.write_text(
        report_path,
        render_suite_markdown(
            suite_name=suite["name"],
            suite_path=suite_path,
            chunking_config=chunking_config,
            write_baselines=args.write_baselines,
            entries=results,
        ),
    )
    return payload, exit_code


def main() -> int:
    args = parse_args()
    payload, exit_code = run_suite(args)
    print(json.dumps(payload, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
