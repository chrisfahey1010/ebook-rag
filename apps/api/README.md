# ebook-rag-api

FastAPI service for document ingestion, retrieval, and grounded answer generation.

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
- `POST /api/documents/upload`
- `GET /api/ingestion/{document_id}/status`
- `POST /api/ingestion/{document_id}/reprocess`
- `GET /api/debug/documents/{document_id}/chunks`
- `POST /api/debug/retrieve`
- `POST /api/debug/rerank`

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

The local real-document fixture library under `benchmarks/local/` now includes:

- `hells_angels.pdf` for long-form narrative nonfiction
- `amazon_quarterly_earnings2025Q4.pdf` for numeric and financial-report QA
- `john_deere_mower_manual.pdf` for manual-style spec lookup and procedural retrieval
- `infinite_jest.pdf` for very long literary prose and front-matter/noise stress
- `qwen3_technical_report.pdf` for technical-report and acronym-heavy questions
- `gpt-5-4_thinking_card.pdf` for model/system-card style factual summaries and policy-like constraints

Only part of this corpus is wired into dedicated benchmark JSON files today. The next evaluation step is to add document-specific question sets for the newer fixtures so the broader local corpus becomes repeatable regression coverage rather than an ad hoc source library.

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
