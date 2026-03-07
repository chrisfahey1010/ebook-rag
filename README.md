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

From [`apps/api`](/home/chris/repos/ebook-rag/apps/api):

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

From [`apps/web`](/home/chris/repos/ebook-rag/apps/web):

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
- persisted per-page text with repeated header/footer cleanup and page-number stripping
- larger paragraph-aware chunks with section heading metadata
- persisted per-document chunking configuration and per-chunk provenance metadata for debug inspection
- chunk embeddings generated during ingestion
- ingestion status lookup and document reprocessing endpoints
- pluggable embedding providers with hashing, local `sentence-transformers`, and OpenAI-compatible adapters
- document ingestion statuses with chunk metadata
- document listing and detail endpoints
- document deletion endpoint that also removes uploaded PDFs
- PostgreSQL `pgvector` storage for chunk embeddings
- dense retrieval executed in the database
- optional lexical retrieval blended with dense candidates before reranking
- anchor-aware retrieval scoring that boosts exact names, numbers, and distinctive query terms
- reranking over hybrid retrieval candidates before answer context assembly
- answer context assembly with near-duplicate suppression, adjacent-chunk expansion, and token-budgeting
- shared query/evidence term normalization for lexical retrieval and extractive QA matching
- long-document retrieval tuning that adds distinctive-term-aware lexical/rerank scoring and rarity-aware candidate fusion
- metadata-aware retrieval penalties and exact-query-run boosts to reduce front-matter-heavy false positives on long books
- focused retrieval scoring that rewards full anchor/constraint coverage for low-frequency entity and date questions
- pluggable reranker providers with token-overlap fallback, local cross-encoder support, and an OpenAI-compatible adapter
- grounded question answering with citations
- composite-question answer synthesis that requires support for each requested facet
- extractive answer selection that can use adjacent-sentence spans and rejects weakly supported semantic neighbors more aggressively
- extractive answer selection that can bridge abbreviated/split sentences like `Hunter S. Thompson` and score up to three-sentence evidence spans
- extractive QA scoring that adds lightweight answer-type cues for date/count/location questions while filtering metadata-like evidence more aggressively
- sentence-level evidence excerpts for returned citations
- per-answer-sentence citation attribution instead of mirroring the whole selected context window
- question-aware citation ranking that breaks evidence ties using anchor terms, constraints, answer-type cues, and narrower page spans
- pluggable QA providers, including a local extractive fallback and an OpenAI-compatible adapter
- retrieval debug route and browser-side candidate inspector
- browser-side ingestion status refresh and reprocessing controls
- an expanded curated eval set for retrieval, unsupported-answer, and citation regression checks
- Next.js document library, upload flow, QA workspace, citation preview pane, and retrieval inspector
- root-level docs for local development

Current limitations:

- PostgreSQL vector storage now follows the configured embedding dimension, but changing dimensions requires running migrations and reprocessing existing documents
- chunk sizing is now configurable and benchmarkable, but the default values are still heuristic and not yet locked by a benchmark-backed decision
- context assembly is still heuristic even though answer traces now separate selected context from cited evidence
- the long-document benchmark now has better unsupported-answer rejection, less metadata/front-matter confusion, and stronger date-specific citation tie-breaking, but it still misses some exact-page citation targets and page-local fact questions on long books
- nickname-specific and some page-local/date-specific questions in the long-document benchmark can still retrieve the right neighborhood but choose the wrong sentence or citation page
- the default benchmark fixture now covers more retrieval failure modes, but it is still synthetic and should keep expanding toward harder multi-page citation coverage cases

## API snapshot

- `GET /health`
- `GET /api/documents`
- `GET /api/documents/{document_id}`
- `DELETE /api/documents/{document_id}`
- `POST /api/documents/upload`
- `GET /api/ingestion/{document_id}/status`
- `POST /api/ingestion/{document_id}/reprocess`
- `POST /api/retrieval/search`
- `POST /api/qa/ask`
- `POST /api/debug/retrieve`
- `GET /api/debug/documents/{document_id}/chunks`
- `POST /api/debug/rerank`

Upload registers the PDF, computes its SHA-256 checksum, stores the file locally, extracts per-page text with PyMuPDF, removes repeated boundary noise such as headers, footers, and standalone page numbers when detectable, builds larger paragraph-aware chunks with page spans, token estimates, and heading metadata, generates embeddings, persists the records, and returns document plus ingestion status metadata.

Reprocessing reruns extraction and embedding generation for an existing document, which is useful after changing embedding models, embedding dimensions, or chunking settings. Each indexed document now persists the chunking configuration that was used, and debug chunk inspection also includes page/paragraph provenance metadata so reprocessing decisions are easier to reason about.

Retrieval accepts a natural-language query, embeds it, blends dense and lexical candidates, reranks them, and returns ranked matches with document metadata, page spans, dense, lexical, hybrid, rerank, and final scores. The ranking path now gives extra weight to full anchor/constraint matches for low-frequency entity and date questions so generic date mentions are less likely to outrank exact fact passages. If a configured reranker backend fails at runtime, retrieval falls back to the local token-overlap reranker so the request still completes.

QA builds on retrieval and returns a grounded answer plus structured citations. Before prompt construction, the QA layer now deduplicates near-identical retrieval hits, pulls in adjacent chunks when budget allows, and limits the final context window. The default local answerer is conservative and can decline to answer when the indexed content does not provide enough support. For composite questions, the extractive path now requires support for each requested facet instead of answering from only the strongest partial match. Citation selection is now evidence-aware instead of mirroring the whole selected context window, and returned citation text is narrowed to the most relevant supporting sentence when possible. Citation ranking also uses the original question terms, answer-type cues, and narrower page spans to break ties when multiple passages contain similar answer text. Multi-sentence answers now attribute citations sentence-by-sentence so composite responses can cite only the chunks that actually support each part. A configurable OpenAI-compatible provider path is also available for model-backed generation.

Provider selection is environment-driven. Embeddings, reranking, and answer generation can now be configured independently for local-only, hosted, or mixed setups.

`POST /api/qa/ask` now accepts `include_trace=true` to expose the selected context window, cited evidence, prompt snapshot, provider name, and timing breakdown used for answer generation.

Debug retrieval exposes the ranked candidate list directly so the frontend can show what the retriever selected before answer generation, including dense and rerank score breakdowns. Additional debug routes now expose persisted document chunks and standalone reranker scoring so chunking, retrieval, and rerank behavior can be inspected separately during tuning.

## Evaluation

Run the default retrieval and citation benchmark from [`apps/api`](/home/chris/repos/ebook-rag/apps/api):

```bash
uv run python scripts/run_eval.py
```

This now uses [`curated_eval.json`](/home/chris/repos/ebook-rag/apps/api/benchmarks/curated_eval.json) by default to upload a broader curated set of local PDFs, ask benchmark questions, and print retrieval hit rate, citation hit rate, support accuracy, answer match rate, unsupported precision, and latency metrics. Individual questions can also declare `citation_match_mode` so stricter page-coverage expectations can be added over time. The older [`sample_eval.json`](/home/chris/repos/ebook-rag/apps/api/benchmarks/sample_eval.json) fixture is still available if you want a smaller smoke test.

The eval runner now also supports chunking presets and explicit chunking overrides so chunk-size choices can be benchmarked directly:

```bash
uv run python scripts/run_eval.py --chunk-preset large
uv run python scripts/run_eval.py --chunk-target-words 560 --chunk-min-words 220 --chunk-overlap-words 80
```

For longer-document tuning, the runner also supports benchmarks that point at a real local PDF via `source_pdf`. The repo now includes [`hells_angels_eval.json`](/home/chris/repos/ebook-rag/apps/api/benchmarks/hells_angels_eval.json), which exercises selected questions against the full 186-page [`hells_angels.pdf`](/home/chris/repos/ebook-rag/apps/api/benchmarks/local/hells_angels.pdf):

```bash
uv run python scripts/run_eval.py --benchmark benchmarks/hells_angels_eval.json
```

For regression tracking, the benchmark runner can also persist JSON and Markdown artifacts and compare a run against a saved baseline:

```bash
uv run python scripts/run_eval.py \
  --output-json benchmarks/results/latest.json \
  --output-markdown benchmarks/results/latest.md \
  --compare-to benchmarks/results/baseline.json \
  --fail-on-regression
```

The expanded summary now includes unsupported precision plus P50/P95 latency so retrieval and citation changes can be compared over time.

## Ingestion configuration

Chunking can now be configured through environment variables before ingestion or reprocessing:

- `CHUNK_TARGET_WORDS`
- `CHUNK_MIN_WORDS`
- `CHUNK_OVERLAP_WORDS`
- `CHUNK_MAX_HEADING_WORDS`

The active values are persisted on each document and surfaced in the web UI plus debug routes so you can tell how an existing index was built before deciding to reprocess it.

## Implementation plan

The active project plan now lives in [`IMPLEMENTATION_PLAN.md`](/home/chris/repos/ebook-rag/IMPLEMENTATION_PLAN.md).
