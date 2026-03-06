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

## Available endpoints

- `GET /health`
- `GET /api/documents`
- `GET /api/documents/{document_id}`
- `POST /api/documents/upload`
