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
- document ingestion statuses with chunk metadata
- document listing and detail endpoints
- Next.js app shell
- root-level docs for local development

Planned next steps:

- embeddings on persisted chunks
- vector retrieval over stored chunk embeddings
- question answering with cited sources
- retrieval quality pass with reranking and debug visibility

## API snapshot

- `GET /health`
- `GET /api/documents`
- `GET /api/documents/{document_id}`
- `POST /api/documents/upload`

Upload now registers the PDF, computes its SHA-256 checksum, stores the file locally, extracts per-page text with PyMuPDF, builds paragraph-aware chunks with page spans and token estimates, persists those records, and returns document plus ingestion status metadata. Embeddings and retrieval are the next milestone.

## Next implementation plan

This plan follows the project spec's V1 retrieval path: embed chunks, retrieve candidate chunks from vector search, then build cited answers before adding reranking and deeper debug tooling.

### 1. Add embeddings to ingestion

- extend chunk persistence to store embedding vectors in PostgreSQL with `pgvector`
- add embedding configuration to the API settings
- define an embedding provider interface and start with one concrete local/default adapter
- update ingestion so newly created chunks are embedded before a document is marked ready

### 2. Add retrieval endpoint

- add query preprocessing and query embedding
- implement top-N dense retrieval against persisted chunk vectors
- return chunk text, page spans, and document metadata in a retrieval response
- keep retrieval observable enough that a later debug view can reuse the same payloads

### 3. Add answer generation with citations

- define a minimal LLM provider interface
- assemble retrieved chunks into a grounded prompt with page references
- add a first `POST /api/qa/ask` endpoint that returns answer text plus cited sources
- keep answer generation conservative: answer from evidence or say the document does not contain enough support

### 4. Tighten the quality loop

- add integration tests for `upload -> ingest -> retrieve -> answer`
- add a small manual benchmark set with expected citations
- add debug endpoints or payload fields for retrieved chunk IDs, scores, and final context
- add reranking only after baseline retrieval quality is measurable
