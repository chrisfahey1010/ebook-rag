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
- PDF upload and file registration endpoints
- document listing endpoint with ingestion status placeholders
- Next.js app shell
- root-level docs for local development

Planned next steps:

- PyMuPDF extraction pipeline
- chunking, embeddings, retrieval, and answer citations

## API snapshot

- `GET /health`
- `GET /api/documents`
- `POST /api/documents/upload`

Upload currently registers the PDF, computes its SHA-256 checksum, stores the file locally, creates a queued ingestion job, and returns document metadata. Extraction and indexing are the next milestone.
