# ebook-rag

`ebook-rag` is a local-first PDF ebook question-answering platform built as a portfolio-quality monorepo. The project is designed to show clean backend, frontend, and local infrastructure fundamentals before layering in ingestion, retrieval, and citation-grounded answer generation.

## Stack

- `apps/api`: FastAPI on Python 3.12
- `apps/web`: Next.js App Router with TypeScript and Tailwind CSS
- `docker-compose.yml`: PostgreSQL 16 with `pgvector`

## Repo layout

```text
.
├── apps
│   ├── api
│   │   ├── pyproject.toml
│   │   ├── src/ebook_rag_api
│   │   └── tests
│   └── web
│       ├── app
│       ├── package.json
│       └── tsconfig.json
├── docker
├── docs
├── scripts
├── docker-compose.yml
└── ebook_rag_project_spec.md
```

## Architecture notes

- The backend uses a small package layout with `api` and `core` modules so ingestion, retrieval, providers, and persistence can be added without flattening everything into `main.py`.
- The frontend now covers the core Phase 1 workflow: upload a PDF, browse the indexed library, ask a document-scoped question, and inspect supporting citations.
- PostgreSQL is provisioned separately with a persistent volume because the project will grow into document, chunk, and embedding tables that should survive container restarts.

## Getting started

### 1. Start local infrastructure

```bash
docker compose up -d
```

Postgres will be available on `localhost:5432` with:

- database: `ebook_rag`
- user: `ebook_rag`
- password: `ebook_rag`

### 2. Run the API

From [`apps/api`](/home/chris/dev/ebook-rag/apps/api):

```bash
export DATABASE_URL=postgresql+psycopg://ebook_rag:ebook_rag@localhost:5432/ebook_rag
uv sync
uv run alembic upgrade head
uv run uvicorn ebook_rag_api.main:app --reload --app-dir src --host 0.0.0.0 --port 8000
```

The API defaults to SQLite if `DATABASE_URL` is unset. Set it to the Postgres container above before running migrations or starting the server.

If the web app runs from a different origin than `http://localhost:3000`, set `CORS_ALLOWED_ORIGINS` for the API to include that frontend origin.

Health check:

```bash
curl http://localhost:8000/health
```

### 3. Run the web app

From [`apps/web`](/home/chris/dev/ebook-rag/apps/web):

```bash
npm install
npm run dev
```

The UI will be available at `http://localhost:3000`.

## Current status

Current implementation includes:

- local Postgres + `pgvector`
- FastAPI app with a health endpoint
- SQLAlchemy database wiring with Alembic migrations
- PDF upload, file registration, and PyMuPDF extraction
- persisted per-page text and paragraph-aware chunks
- chunk embeddings generated during ingestion
- document ingestion statuses with chunk metadata
- document listing and detail endpoints
- document deletion endpoint that also removes uploaded PDFs
- PostgreSQL `pgvector` storage for chunk embeddings
- dense retrieval executed in the database
- grounded question answering with citations
- pluggable QA providers, including a local extractive fallback and an OpenAI-compatible adapter
- retrieval debug route and browser-side candidate inspector
- Next.js document library, upload flow, QA workspace, citation preview pane, and retrieval inspector
- root-level docs for local development

Current limitations:

- retrieval is dense-only today; reranking is still pending
- the debug surface currently covers ranked retrieval candidates, but not reranker scores or final prompt context yet

## API snapshot

- `GET /health`
- `GET /api/documents`
- `GET /api/documents/{document_id}`
- `DELETE /api/documents/{document_id}`
- `POST /api/documents/upload`
- `POST /api/retrieval/search`
- `POST /api/qa/ask`
- `POST /api/debug/retrieve`

Upload registers the PDF, computes its SHA-256 checksum, stores the file locally, extracts per-page text with PyMuPDF, builds paragraph-aware chunks with page spans and token estimates, generates embeddings, persists the records, and returns document plus ingestion status metadata.

Retrieval accepts a natural-language query, embeds it, scores persisted chunks, and returns ranked matches with document metadata, page spans, and similarity scores.

QA builds on retrieval and returns a grounded answer plus structured citations. The default local answerer is conservative and can decline to answer when the indexed content does not provide enough support. A configurable OpenAI-compatible provider path is also available for model-backed generation.

Debug retrieval exposes the ranked candidate list directly so the frontend can show what the retriever selected before answer generation.

## Next implementation plan

The project has reached the first end-to-end product milestone: upload -> ingest -> retrieve -> answer in both the API and the browser UI. The next work should close the remaining quality and observability gaps from the project spec.

### 1. Improve retrieval quality and observability

- add debug payloads or endpoints for retrieved chunk IDs, scores, and final prompt context
- add a small benchmark set with expected citations
- add reranking only after baseline retrieval quality is measurable
