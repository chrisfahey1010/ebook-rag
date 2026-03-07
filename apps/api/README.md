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

`--fail-on-regression` only gates on the regression lane. Exploratory excerpt checks still appear in the report, but they do not trigger a non-zero exit code on their own.

To run the focused page-local citation benchmark:

```bash
uv run python scripts/run_eval.py \
  --benchmark benchmarks/citation_granularity_eval.json \
  --output-json benchmarks/results/citation_granularity_latest.json \
  --output-markdown benchmarks/results/citation_granularity_latest.md
```

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
