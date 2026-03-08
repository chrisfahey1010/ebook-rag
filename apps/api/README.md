# ebook-rag-api

FastAPI service for document ingestion, retrieval, and grounded answer generation.

The API already has real ingestion, hybrid retrieval, reranking, streaming QA, and regression tooling. The main remaining quality gap is answer orchestration: retrieval is ahead of the current generative answer layer, so the next major step is to make grounded local-model-backed answering a first-class path.

## Development

```bash
export DATABASE_URL=postgresql+psycopg://ebook_rag:ebook_rag@localhost:5432/ebook_rag
uv sync
uv run alembic upgrade head
uv run uvicorn ebook_rag_api.main:app --reload --app-dir src
```

## Migrations

```bash
uv run alembic upgrade head
```

Set `DATABASE_URL` first if you want the API to use Postgres + `pgvector`; otherwise the settings default is SQLite.

Set `EMBEDDING_DIMENSIONS` to match the configured embedding model before running migrations. If you change it later, run `uv run alembic upgrade head` and reprocess existing documents through `POST /api/ingestion/{document_id}/reprocess`.

## Available endpoints

- `GET /health`
- `GET /api/documents`
- `GET /api/documents/{document_id}`
- `DELETE /api/documents/{document_id}`
- `POST /api/documents/upload`
- `GET /api/ingestion/{document_id}/status`
- `POST /api/ingestion/{document_id}/reprocess`
- `POST /api/retrieval/search`
- `POST /api/qa/ask`
- `POST /api/qa/ask-stream`
- `GET /api/debug/documents/{document_id}/chunks`
- `POST /api/debug/retrieve`
- `POST /api/debug/rerank`

## Local model direction

The intended V1 backend split is:

- local embeddings through `sentence-transformers`
- local reranking through a cross-encoder
- local answer generation through an OpenAI-compatible local runtime using a small instruct model such as Qwen 3.5/4B

That split is deliberate. Small local LLMs are most useful here for grounded answer synthesis, unsupported-answer classification, query decomposition, and answer verification. They should improve the answer layer rather than replace the embedding or reranking stack.

## Benchmark workflow

From `apps/api`, you can run the local benchmark and optionally persist comparable artifacts:

```bash
uv run python scripts/run_eval.py \
  --output-json benchmarks/results/latest.json \
  --output-markdown benchmarks/results/latest.md
```

To compare a run against a prior baseline and fail on regressions:

```bash
uv run python scripts/run_eval.py \
  --output-json benchmarks/results/latest.json \
  --compare-to benchmarks/results/baseline.json \
  --fail-on-regression
```

The benchmark summary includes retrieval hit rate, citation hit rate, citation evidence hit rate, support accuracy, answer match rate, unsupported precision, and average/P50/P95 latency.

Questions can also declare:

- `expected_citation_text_contains` to assert that the returned citation excerpt contains the expected supporting sentence or phrase
- `citation_text_match_mode` (`any` or `all`) to control how multiple excerpt expectations are scored
- `regression_tier` (`gating` or `exploratory`) so harder long-form citation checks can be tracked without blocking every merge

Citation excerpt matching is normalization-aware for benchmark purposes: whitespace runs, separator-dash variants, smart quotes, and common currency/percent spacing differences are collapsed before excerpt expectations are compared.

`--fail-on-regression` only gates on the regression lane. Exploratory excerpt checks still appear in the report, but they do not trigger a non-zero exit code on their own.
Latency regressions use a small `10 ms` buffer so reruns do not fail on trivial local jitter.

The committed regression lane for the newer gating suites is tracked in [`benchmarks/regression_suite.json`](/home/chris/repos/ebook-rag/apps/api/benchmarks/regression_suite.json). From `apps/api`, use:

```bash
uv run python scripts/run_regression_suite.py
```

That command reruns the manual/report/system-card benchmarks, writes fresh `*_latest.json` and `*_latest.md` artifacts under `benchmarks/results/`, compares each run to its saved baseline, and exits non-zero if any gating metric regresses.

As of March 8, 2026, that regression suite is the canonical release check, and it is not fully green. A fresh rerun preserved perfect answer-quality metrics on the current John Deere, Qwen 3 technical report, and GPT-5.4 thinking card suites, but it still flagged latency regressions relative to the saved March 8 baselines:

- John Deere manual: `average_latency_ms`, `latency_p50_ms`, and `latency_p95_ms`
- Qwen 3 technical report: `average_latency_ms`, `latency_p50_ms`, and `latency_p95_ms`
- GPT-5.4 thinking card: `latency_p50_ms`

To refresh the saved baselines intentionally:

```bash
uv run python scripts/run_regression_suite.py --write-baselines
```

To run the focused page-local citation benchmark:

```bash
uv run python scripts/run_eval.py \
  --benchmark benchmarks/citation_granularity_eval.json \
  --output-json benchmarks/results/citation_granularity_latest.json \
  --output-markdown benchmarks/results/citation_granularity_latest.md
```

To run the real-document Amazon earnings benchmark:

```bash
uv run python scripts/run_eval.py \
  --benchmark benchmarks/amazon_earnings_eval.json \
  --output-json benchmarks/results/amazon_earnings_latest.json \
  --output-markdown benchmarks/results/amazon_earnings_latest.md
```

The Amazon benchmark is still a tracked quality target rather than a clean pass in the live code path, and it should not be treated as a release gate yet. The current saved March 8, 2026 artifact reports `retrieval_hit_rate=1.0`, `citation_hit_rate=0.9`, `citation_evidence_hit_rate=0.5`, `gating_citation_evidence_hit_rate=0.625`, `support_accuracy=0.9`, `answer_match_rate=0.5`, and `unsupported_precision=0.5`, with remaining misses around structured/page-local evidence selection and unsupported-answer behavior.

The local real-document fixture library under `benchmarks/local/` now includes:

- `hells_angels.pdf` for long-form narrative nonfiction
- `amazon_quarterly_earnings2025Q4.pdf` for numeric and financial-report QA
- `john_deere_mower_manual.pdf` for manual-style spec lookup and procedural retrieval
- `infinite_jest.pdf` for very long literary prose and front-matter/noise stress
- `qwen3_technical_report.pdf` for technical-report and acronym-heavy questions
- `gpt-5-4_thinking_card.pdf` for model/system-card style factual summaries and policy-like constraints

Benchmark status is intentionally split:

- Stable gating suites: John Deere manual, Qwen 3 technical report, GPT-5.4 thinking card, plus the focused citation-granularity fixture
- Exploratory long-form suites: `infinite_jest`, and the exploratory checks embedded inside `hells_angels`
- Tracked but not release-blocking: Amazon earnings and the remaining long-form `hells_angels` gaps

To compare the built-in chunking presets and emit a recommendation report:

```bash
uv run python scripts/run_eval.py \
  --compare-presets \
  --output-json benchmarks/results/chunking_curated_20260307.json \
  --output-markdown benchmarks/results/chunking_curated_20260307.md
uv run python scripts/run_eval.py \
  --benchmark benchmarks/hells_angels_eval.json \
  --compare-presets \
  --output-json benchmarks/results/chunking_hells_angels_20260307.json \
  --output-markdown benchmarks/results/chunking_hells_angels_20260307.md
```

The current default chunking config remains `420 / 180 / 64 / 12`. The curated fixture is effectively a latency tie, but the long-form `hells_angels` comparison recommends the default preset and shows a quality regression for `large`.

For the broader project plan, see [`IMPLEMENTATION_PLAN.md`](/home/chris/repos/ebook-rag/IMPLEMENTATION_PLAN.md). The current direction is to stop open-ended retrieval tuning, keep one truthful benchmark baseline, and put the next major effort into grounded local-model-backed answering.
