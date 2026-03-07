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

The benchmark summary includes retrieval hit rate, citation hit rate, support accuracy, answer match rate, unsupported precision, and average/P50/P95 latency.
