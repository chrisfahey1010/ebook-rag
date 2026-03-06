# Ebook QA Platform — Full Project Spec

## 1. Project Summary

Build a **local-first document question-answering platform** for PDF ebooks with:

- reliable PDF ingestion
- retrieval over indexed text chunks
- grounded answers with page citations
- pluggable local or remote LLM backends
- a polished web UI suitable for portfolio/resume use

This should be engineered as a **modular retrieval system**, not just a toy chatbot.

---

## 2. Goals

### Primary goals

- Accept digital PDFs with selectable text
- Extract and normalize text accurately enough for QA
- Index documents for fast retrieval
- Answer questions using only document-grounded context
- Cite the source pages/chunks used in each answer
- Allow generation model backends to be swapped with minimal code changes
- Be easy to run locally for developers and recruiters

### Secondary goals

- Support multiple local model runtimes
- Expose ingestion and retrieval internals for debugging
- Include evaluation scripts and benchmarks
- Demonstrate strong backend, infra, and frontend engineering

### Non-goals for V1

- OCR for scanned PDFs
- multi-user auth
- cloud-scale deployment
- agentic autonomous research across the internet
- handwritten notes / image-heavy PDFs

---

## 3. Product Vision

The application should feel like a **serious developer tool** for asking questions about ebooks, manuals, and long-form documents. The user should be able to upload one or more PDFs, ask natural-language questions, inspect citations, and verify exactly where the answer came from.

A strong portfolio outcome is:

> "Built a modular local-first PDF QA platform with FastAPI, Next.js, PostgreSQL/pgvector, configurable local LLM backends, retrieval/reranking, and citation-grounded answering."

---

## 4. High-Level Architecture

```text
[Next.js Web UI]
        |
        v
[FastAPI API]
   |        |         |
   |        |         +--> [LLM Provider Layer]
   |        |                 - Ollama
   |        |                 - llama.cpp server
   |        |                 - vLLM
   |        |                 - generic OpenAI-compatible endpoint
   |        |
   |        +--> [Retrieval Pipeline]
   |                 - query preprocessing
   |                 - vector search
   |                 - optional lexical search
   |                 - reranking
   |
   +--> [Ingestion Pipeline]
                     - PDF extraction
                     - normalization
                     - chunking
                     - embeddings
                     - persistence

[PostgreSQL + pgvector]
        - documents
        - pages
        - chunks
        - embeddings
        - ingestion jobs
        - answers / traces (optional)
```

---

## 5. Recommended Tech Stack

## Backend

- **Python 3.12**
- **FastAPI**
- **SQLAlchemy 2.x**
- **Alembic**
- **Pydantic**
- **httpx**
- **OpenAI Python SDK** for OpenAI-compatible model endpoints
- **PyMuPDF** for PDF extraction
- **sentence-transformers** for embeddings and reranking
- **uv** for Python package management

## Database

- **PostgreSQL 16**
- **pgvector**

## Frontend

- **Next.js (App Router)**
- **TypeScript**
- **Tailwind CSS**
- **shadcn/ui**

## Local infrastructure

- **Docker Compose** for Postgres/pgvector and optional supporting services

## Optional supporting tools

- **Redis** only if background jobs become necessary
- **RQ / Dramatiq / Celery** only if ingestion jobs need queueing later

---

## 6. Core Design Principles

1. **Local-first**
   - The entire app should be runnable on a developer machine.

2. **Pluggable inference**
   - The app must not be tightly coupled to one LLM runtime.

3. **Grounded answers over flashy answers**
   - Accuracy and citation quality matter more than verbosity.

4. **Observable retrieval**
   - Developers should be able to inspect chunks, retrieval candidates, reranker scores, and final context.

5. **Simple first, extensible later**
   - Start with a clean MVP and add sophistication only after the basics work.

---

## 7. Retrieval Strategy

## Recommended approach for V1

Use **retrieve-then-rerank RAG**:

1. embed the query
2. retrieve candidate chunks from vector search
3. optionally blend with lexical/BM25 results
4. rerank top candidates using a reranker model
5. pass the best chunks into the generation prompt
6. produce answer with citations

## Why this is the right starting point

- Simple enough to implement correctly
- Strong quality for digital PDFs
- More robust than naive vector-only retrieval
- Easy to benchmark and debug

## Avoid in V1

- multi-agent retrieval planners
- graph RAG
- autonomous web-browsing agents
- excessive framework abstraction

---

## 8. PDF Ingestion Pipeline

## Input assumptions

- PDFs are digital, with selectable text
- PDFs are primarily text-based ebooks or manuals
- Layouts may include headers, footers, chapter titles, page numbers

## Ingestion stages

### 8.1 File registration

When a PDF is uploaded:

- compute checksum (SHA-256)
- save file metadata
- assign document ID
- enqueue or trigger ingestion job

### 8.2 Text extraction

Use **PyMuPDF** to extract:

- per-page text
- document metadata
- page count

Store raw extracted text per page for traceability.

### 8.3 Text normalization

Normalize text while preserving meaning:

- collapse excess whitespace
- standardize line breaks
- remove repeated headers/footers when detectable
- preserve page number mapping
- preserve chapter/section headings where possible

### 8.4 Chunking

Chunk by semantic boundaries when possible.

Initial chunking rules:

- target approximately **600–1200 tokens**
- overlap approximately **10–20%**
- prefer paragraph/heading boundaries
- maintain page span metadata

Each chunk should store:

- document_id
- chunk_index
- page_start
- page_end
- text
- token_estimate
- heading / chapter metadata if available

### 8.5 Embedding generation

Generate embeddings for each chunk using the configured embedding provider.

### 8.6 Persistence

Persist:

- document record
- page records
- chunk records
- vector embedding
- ingestion timestamps and status

---

## 9. Answering Pipeline

When the user asks a question:

### 9.1 Query preprocessing

- normalize whitespace
- optionally expand abbreviations
- optionally rewrite query later in V2

### 9.2 Candidate retrieval

- dense vector retrieval from pgvector
- optional lexical/BM25 retrieval in later phase
- pull top N candidates

### 9.3 Reranking

Use a reranker model to score candidate chunks against the query.

### 9.4 Context assembly

Build the final context window from top-ranked chunks while:

- avoiding near-duplicate chunks
- respecting token budget
- preserving page metadata

### 9.5 Answer generation

Prompt the LLM to:

- answer only from provided evidence
- admit when evidence is insufficient
- cite page numbers / chunk references
- distinguish quotes from paraphrases

### 9.6 Response payload

Return:

- answer text
- cited sources
- confidence / support metadata (optional)
- debug trace (optional in dev mode)

---

## 10. Model Abstraction Layer

The application should have explicit provider interfaces.

### 10.1 Interfaces

```python
from typing import Protocol

class LLMProvider(Protocol):
    async def generate(self, messages: list[dict], **kwargs) -> str: ...

class EmbeddingProvider(Protocol):
    async def embed_texts(self, texts: list[str]) -> list[list[float]]: ...

class Reranker(Protocol):
    async def score(self, query: str, passages: list[str]) -> list[float]: ...
```

### 10.2 Concrete adapters

Implement adapters such as:

- `OpenAICompatibleLLMProvider`
- `OllamaLLMProvider`
- `LlamaCppLLMProvider`
- `VLLMLLMProvider`
- `SentenceTransformerEmbeddingProvider`
- `CrossEncoderReranker`

### 10.3 Configuration approach

Use environment variables and config files to control:

- generation model endpoint
- model name
- embedding model
- reranker model
- retrieval thresholds
- token budget

---

## 11. Database Schema (Initial Draft)

## documents

- `id` (uuid)
- `filename`
- `title`
- `sha256`
- `page_count`
- `status` (`uploaded`, `processing`, `ready`, `failed`)
- `created_at`
- `updated_at`

## document_pages

- `id` (uuid)
- `document_id`
- `page_number`
- `raw_text`
- `normalized_text`

## document_chunks

- `id` (uuid)
- `document_id`
- `chunk_index`
- `page_start`
- `page_end`
- `heading`
- `text`
- `token_estimate`
- `embedding vector(...)`
- `created_at`

## ingestion_jobs

- `id` (uuid)
- `document_id`
- `status`
- `error_message`
- `started_at`
- `finished_at`

## qa_sessions` (optional)

- `id` (uuid)
- `created_at`

## qa_messages` (optional)

- `id` (uuid)
- `session_id`
- `role`
- `content`
- `created_at`

## answer_traces` (optional, dev/debug)

- `id` (uuid)
- `question`
- `retrieved_chunk_ids`
- `rerank_scores`
- `prompt_snapshot`
- `created_at`

---

## 12. API Design (Initial)

## Document APIs

- `POST /api/documents/upload`
- `GET /api/documents`
- `GET /api/documents/{document_id}`
- `DELETE /api/documents/{document_id}`

## Ingestion APIs

- `GET /api/ingestion/{document_id}/status`
- `POST /api/ingestion/{document_id}/reprocess`

## QA APIs

- `POST /api/qa/ask`
- `POST /api/qa/ask-stream`

## Debug APIs (dev only)

- `GET /api/debug/documents/{document_id}/chunks`
- `POST /api/debug/retrieve`
- `POST /api/debug/rerank`

## Config APIs (optional)

- `GET /api/config/models`
- `POST /api/config/models/test`

---

## 13. Frontend Requirements

## Primary UI flows

### Library view

- list uploaded PDFs
- show ingestion status
- show page count / metadata
- open a document workspace

### Document workspace

- question input
- answer pane
- citations pane
- source preview pane with page references

### Debug view (portfolio gold)

- retrieved chunks
- reranker order
- raw chunk text
- timing breakdown

## UX expectations

- clean and minimal
- responsive enough for desktop first
- clear “not enough evidence” state
- obvious source verification path

---

## 14. Prompting Rules

The answering prompt should instruct the model to:

- rely only on the provided context
- never fabricate content not in the passages
- say that the answer is not found when evidence is insufficient
- cite the relevant page numbers for every substantial claim
- quote sparingly and only when useful

The prompt should favor:

- factual clarity
- concise answers
- traceability

---

## 15. Evaluation Strategy

A real project needs evaluation, not vibes.

## V1 evaluation set

Create a small benchmark set manually:

- 3 to 5 PDFs
- 10 to 20 questions per PDF
- expected answer summary
- expected page citations

## Metrics to track

- retrieval hit rate
- citation correctness
- grounded answer rate
- hallucination rate
- latency

## Useful eval modes

- retrieval-only eval
- reranking eval
- full answer eval

---

## 16. Testing Strategy

## Unit tests

- PDF text normalization
- chunking boundaries
- provider adapters
- retrieval ranking merge logic
- citation formatting

## Integration tests

- upload PDF -> ingest -> ask question -> get cited answer
- failed ingestion handling
- switching provider backends

## Manual QA

- long chapter retrieval
- vague questions
- quote extraction questions
- answer-not-found behavior

---

## 17. Observability / Debuggability

This is one of the best portfolio differentiators.

Add developer visibility into:

- ingestion timing
- chunk counts per document
- retrieval candidates
- reranker scores
- final context window
- model response latency

A retrieval-debug page will make the project feel much more serious.

---

## 18. Security / Safety Considerations

For a local-first tool, keep this simple:

- sanitize uploaded filenames
- validate file type and size
- avoid arbitrary shell execution from user inputs
- cap token budgets and retrieval counts
- do not expose secrets in frontend

---

## 19. Deployment Modes

## Local dev mode

- frontend on localhost
- backend on localhost
- Postgres via Docker Compose
- local model endpoint via Ollama / llama.cpp / vLLM

## Demo mode

- same architecture, simplified seed data
- optional hosted frontend/backend later

---

## 20. Phased Roadmap

## Phase 1 — MVP

Build:

- PDF upload
- ingestion pipeline
- pgvector storage
- vector retrieval
- basic answer generation
- citations
- simple UI

**Success criteria:**
User can upload a PDF and get a mostly correct cited answer.

## Phase 2 — Quality pass

Build:

- reranker
- chunking improvements
- source preview UI
- answer-not-found behavior
- debug endpoints

**Success criteria:**
Noticeably better answer quality and trustworthiness.

## Phase 3 — Modularity

Build:

- configurable model backends
- provider registry
- model settings screen
- backend capability checks

**Success criteria:**
User can swap model runtime with minimal friction.

## Phase 4 — Resume polish

Build:

- evaluation harness
- benchmark results in README
- architecture diagrams
- Dockerized setup
- screenshots / demo GIFs

**Success criteria:**
The GitHub repo looks like a professional software project.

---

## 21. Suggested Repo Structure

```text
ebook-rag/
  apps/
    api/
      app/
        api/
        core/
        db/
        models/
        schemas/
        services/
          answering/
          ingestion/
          llm/
          retrieval/
        main.py
      tests/
      pyproject.toml
    web/
      app/
      components/
      lib/
      package.json
  docs/
    architecture.md
    ingestion.md
    retrieval.md
    local-setup.md
  docker/
  scripts/
  .github/
    workflows/
  docker-compose.yml
  README.md
```

---

## 22. Day-One Implementation Order

1. Create monorepo scaffold
2. Add Docker Compose with Postgres + pgvector
3. Create FastAPI service and health endpoint
4. Create Next.js app shell
5. Add PDF upload endpoint
6. Extract page text from one PDF and persist it
7. Add chunking and embeddings
8. Add retrieval endpoint
9. Add answer generation endpoint
10. Add citations in UI
11. Add reranker
12. Add debug page

---

## 23. Stretch Features

After the core app works, consider:

- streaming answers
- quote-only mode
- chapter-aware retrieval
- per-document filters
- export conversation
- side-by-side page viewer
- notes/highlights
- cloud deployment option

---

## 24. Resume Framing

### Good one-line description

Built a modular PDF question-answering platform with FastAPI, Next.js, PostgreSQL/pgvector, local LLM support, retrieval/reranking, and citation-grounded answers over user-provided ebooks.

### Skills demonstrated

- Python backend engineering
- TypeScript/React frontend work
- SQL/data modeling
- local AI inference integration
- retrieval system design
- modular architecture
- Docker/dev environment design
- testing and evaluation

---

## 25. Final Recommendation

Do not overcomplicate V1. The winning move is:

- strong ingestion
- reliable chunking
- good retrieval
- reranking
- clear citations
- pluggable model backend
- polished UI and docs

That combination is much more impressive than bolting an “AI agent” label onto a fragile prototype.

