# ebook-rag-api

FastAPI service for document ingestion, retrieval, and grounded answer generation.

## Development

```bash
uv sync
uv run uvicorn ebook_rag_api.main:app --reload --app-dir src
```

## Available endpoints

- `GET /health`
- `GET /api/documents`
- `GET /api/documents/{document_id}`
- `POST /api/documents/upload`
