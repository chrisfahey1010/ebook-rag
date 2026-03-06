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
- The frontend starts as a thin shell instead of a generated demo app. That keeps the first commit focused on information architecture and integration points.
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
uv sync
uv run uvicorn ebook_rag_api.main:app --reload --app-dir src --host 0.0.0.0 --port 8000
```

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
- SQLAlchemy database wiring and table initialization
- PDF upload, file registration, and PyMuPDF extraction
- persisted per-page text and paragraph-aware chunks
- chunk embeddings generated during ingestion
- document ingestion statuses with chunk metadata
- document listing and detail endpoints
- dense retrieval over stored chunk embeddings
- grounded question answering with citations
- pluggable QA providers, including a local extractive fallback and an OpenAI-compatible adapter
- Next.js app shell
- root-level docs for local development

Current limitations:

- chunk embeddings are currently stored as JSON, not PostgreSQL `pgvector`
- schema evolution still uses startup table creation instead of Alembic migrations
- retrieval is dense-only today; reranking and debug views are still pending
- the web app has not yet been wired to the upload, retrieval, and QA flows

## API snapshot

- `GET /health`
- `GET /api/documents`
- `GET /api/documents/{document_id}`
- `POST /api/documents/upload`
- `POST /api/retrieval/search`
- `POST /api/qa/ask`

Upload registers the PDF, computes its SHA-256 checksum, stores the file locally, extracts per-page text with PyMuPDF, builds paragraph-aware chunks with page spans and token estimates, generates embeddings, persists the records, and returns document plus ingestion status metadata.

Retrieval accepts a natural-language query, embeds it, scores persisted chunks, and returns ranked matches with document metadata, page spans, and similarity scores.

QA builds on retrieval and returns a grounded answer plus structured citations. The default local answerer is conservative and can decline to answer when the indexed content does not provide enough support. A configurable OpenAI-compatible provider path is also available for model-backed generation.

## Next implementation plan

The project has reached the first end-to-end backend milestone: upload -> ingest -> retrieve -> answer. The next work should close the gap between the current implementation and the target architecture in the project spec.

### 1. Move vector storage and retrieval onto PostgreSQL `pgvector`

- replace JSON-backed embedding storage with a real vector column
- run similarity search in the database instead of in Python
- add the database extension and indexes needed for scalable retrieval

### 2. Add Alembic migrations and tighten persistence

- replace startup `create_all` schema management with explicit migrations
- make local development and schema changes reproducible across environments
- prepare the project for incremental database evolution as retrieval and tracing expand

### 3. Improve retrieval quality and observability

- add debug payloads or endpoints for retrieved chunk IDs, scores, and final prompt context
- add a small benchmark set with expected citations
- add reranking only after baseline retrieval quality is measurable

### 4. Build the actual web workflow

- wire the Next.js app to document upload, document browsing, retrieval inspection, and QA
- expose citations and supporting chunks in the UI so answers are auditable
- make visible progress in the browser instead of only through backend endpoints
