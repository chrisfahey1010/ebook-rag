# ebook-rag

`ebook-rag` is a local-first PDF ebook question-answering platform built as a portfolio-quality monorepo. The project is designed to show clean backend, frontend, and local infrastructure fundamentals before layering in ingestion, retrieval, and citation-grounded answer generation.

The current implementation already has real ingestion, hybrid retrieval, reranking, streaming QA, and benchmark tooling. The main remaining gap is answer intelligence: the retrieval stack is ahead of the answer layer, and the next major push is to make grounded local-model-backed answering a first-class path.

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
- repeated two-line header/footer boundary-block cleanup during normalization
- larger paragraph-aware chunks with section heading metadata
- stacked heading preservation and per-chunk heading-path provenance for debug inspection
- persisted per-document chunking configuration and per-chunk provenance metadata for debug inspection
- persisted normalized-text character-span provenance per chunk and paragraph for page-local citation debugging
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
- streamed grounded answers in the browser via server-sent events
- composite-question answer synthesis that requires support for each requested facet
- extractive answer selection that can use adjacent-sentence spans and rejects weakly supported semantic neighbors more aggressively
- extractive answer selection that can bridge abbreviated/split sentences like `Hunter S. Thompson` and score up to three-sentence evidence spans
- extractive QA scoring that adds lightweight answer-type cues for date/count/location questions while filtering metadata-like evidence more aggressively
- extractive QA scoring that can inspect structured numeric/table-like line groups for financial-report-style passages
- extractive QA scoring that now synthesizes metric-plus-period table spans and prefers exact metric/quarter matches over sibling financial rows
- extractive QA scoring that now prefers exact metric lookup rows over broader narrative summaries when both are retrieved for the same financial question
- sentence-level evidence excerpts for returned citations
- per-answer-sentence citation attribution instead of mirroring the whole selected context window
- merged same-chunk evidence excerpts for multi-sentence answers when support spans multiple answer sentences
- question-aware citation ranking that breaks evidence ties using anchor terms, constraints, answer-type cues, and narrower page spans
- pluggable QA providers, including a local extractive fallback and an OpenAI-compatible adapter
- routed QA modes that distinguish extractive lookup, grounded synthesis, and unsupported responses
- task-specific unsupported classification for higher-risk generative questions before answer synthesis
- QA responses with support confidence, evidence score, router rationale, and runtime metadata
- claim-level answer verification with per-claim support scores, rationales, and supporting evidence spans
- optional local-model-backed claim verification through the existing OpenAI-compatible answer runtime
- retrieval debug route and browser-side candidate inspector
- browser-side page-local citation inspector with answer-sentence matching and normalized-text offset spans
- browser-side ingestion status refresh and reprocessing controls
- browser-side provider/runtime visibility for embeddings, reranking, and answer generation
- an expanded curated eval set for retrieval, unsupported-answer, and citation regression checks
- an expanded local real-document benchmark fixture library spanning long-form narrative prose, technical reports, model/system cards, financial filings, and equipment manuals
- Next.js document library, upload flow, QA workspace, citation preview pane, and retrieval inspector
- root-level docs for local development

Current limitations:

- PostgreSQL vector storage now follows the configured embedding dimension, but changing dimensions requires running migrations and reprocessing existing documents
- chunk sizing is now benchmark-backed for the current fixture set, but the benchmark still needs broader real-document coverage before the defaults should be treated as final
- context assembly is still heuristic even though answer traces now separate selected context from cited evidence
- the answer verifier now combines a pre-answer unsupported-classification stage with claim-level heuristic checks and an optional local-model pass, but it still downgrades weak answers instead of repairing or rewriting unsupported claims
- local-model runtime configuration is supported, but the recommended fully local setup is not yet documented and surfaced clearly enough
- the March 8, 2026 ingestion-quality pass improved normalization and heading metadata structure, but the current benchmark suites did not show a measurable end-to-end quality lift from those ingestion changes alone
- the long-form `hells_angels` benchmark still misses some exact-page citation targets and page-local fact questions on long books
- the Amazon earnings benchmark is still exposing structured-evidence and unsupported-answer gaps in the current code path
- the benchmark suite now includes a broader local real-document fixture set in `apps/api/benchmarks/local/`, including `hells_angels.pdf`, `amazon_quarterly_earnings2025Q4.pdf`, `john_deere_mower_manual.pdf`, `infinite_jest.pdf`, `qwen3_technical_report.pdf`, and `gpt-5-4_thinking_card.pdf`, but fixture breadth is no longer the main issue; the main issue is turning the strong retrieval stack into a stronger grounded answer pipeline

## Local model direction

The intended V1 direction is:

- local embeddings through `sentence-transformers`
- local reranking through a cross-encoder
- local answer generation through an OpenAI-compatible runtime using a small instruct model such as Qwen 3.5/4B

That split is deliberate. Small local LLMs are most valuable here for grounded answer synthesis, unsupported-answer classification, query decomposition, and answer verification. They should improve the answer layer, not replace the embedding model or reranker.

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
- `POST /api/qa/ask-stream`
- `POST /api/debug/retrieve`
- `GET /api/debug/documents/{document_id}/chunks`
- `POST /api/debug/rerank`

Upload registers the PDF, computes its SHA-256 checksum, stores the file locally, extracts per-page text with PyMuPDF, removes repeated boundary noise such as headers, footers, standalone page numbers, and repeated two-line boundary blocks when detectable, collapses soft-wrapped body lines while preserving short stacked heading blocks, builds larger paragraph-aware chunks with page spans, token estimates, heading metadata, and heading-path provenance, generates embeddings, persists the records, and returns document plus ingestion status metadata.

Reprocessing reruns extraction and embedding generation for an existing document, which is useful after changing embedding models, embedding dimensions, or chunking settings. Each indexed document now persists the chunking configuration that was used, and debug chunk inspection also includes page, paragraph, and normalized-text character-span provenance metadata so reprocessing decisions are easier to reason about.

Retrieval accepts a natural-language query, embeds it, blends dense and lexical candidates, reranks them, and returns ranked matches with document metadata, page spans, chunk provenance, and dense, lexical, hybrid, rerank, and final scores. The ranking path now gives extra weight to full anchor/constraint matches for low-frequency entity and date questions so generic date mentions are less likely to outrank exact fact passages. If a configured reranker backend fails at runtime, retrieval falls back to the local token-overlap reranker so the request still completes.

QA builds on retrieval and returns a grounded answer plus structured citations. Before prompt construction, the QA layer now deduplicates near-identical retrieval hits, pulls in adjacent chunks when budget allows, and limits the final context window. The default local answerer is conservative and can decline to answer when the indexed content does not provide enough support. For composite questions, the extractive path now requires support for each requested facet instead of answering from only the strongest partial match. Citation selection is now evidence-aware instead of mirroring the whole selected context window, and returned citation text is narrowed to the most relevant supporting sentence when possible. Citation ranking also uses the original question terms, answer-type cues, and narrower page spans to break ties when multiple passages contain similar answer text. Multi-sentence answers now attribute citations sentence-by-sentence so composite responses can cite only the chunks that actually support each part. After answer generation, the QA layer now also verifies each answer claim against the selected evidence, can optionally use the configured OpenAI-compatible local model as a conservative verifier, and downgrades weakly supported answers to unsupported rather than returning ungrounded synthesis. A configurable OpenAI-compatible provider path is also available for model-backed generation.

Provider selection is environment-driven. Embeddings, reranking, and answer generation can now be configured independently for local-only, hosted, or mixed setups.

`POST /api/qa/ask` now accepts `include_trace=true` to expose the selected context window, cited evidence, chunk provenance, prompt snapshot, provider name, and timing breakdown used for answer generation. `POST /api/qa/ask-stream` exposes the same final answer payload as a terminal server-sent event after streaming incremental answer text chunks, so the web app can render the answer progressively and then attach citations and trace metadata when generation finishes. The web app uses that trace plus citation provenance to show a selected-citation inspector with answer-sentence matching, page-local normalized-text offsets, paragraph spans, and score breakdowns for source verification.

Debug retrieval exposes the ranked candidate list directly so the frontend can show what the retriever selected before answer generation, including dense, lexical, hybrid, rerank, and final score breakdowns plus chunk provenance. Additional debug routes now expose persisted document chunks and standalone reranker scoring so chunking, retrieval, rerank, and page-local source spans can be inspected separately during tuning.

## Evaluation

Run the default retrieval and citation benchmark from [`apps/api`](/home/chris/repos/ebook-rag/apps/api):

```bash
uv run python scripts/run_eval.py
```

This now uses [`curated_eval.json`](/home/chris/repos/ebook-rag/apps/api/benchmarks/curated_eval.json) by default to upload a broader curated set of local PDFs, ask benchmark questions, and print retrieval hit rate, citation hit rate, citation evidence hit rate, support accuracy, answer match rate, unsupported precision, and latency metrics. Benchmark files can declare `defaults` at the suite or document level, and individual questions can override `citation_match_mode`, `expected_citation_text_contains`, `citation_text_match_mode`, and `regression_tier`, so stricter page-coverage and excerpt-accuracy expectations can be added without repeating the same gating/exploratory setting on every question. The older [`sample_eval.json`](/home/chris/repos/ebook-rag/apps/api/benchmarks/sample_eval.json) fixture is still available if you want a smaller smoke test.

The eval runner now also supports chunking presets and explicit chunking overrides so chunk-size choices can be benchmarked directly:

```bash
uv run python scripts/run_eval.py --chunk-preset large
uv run python scripts/run_eval.py --chunk-target-words 560 --chunk-min-words 220 --chunk-overlap-words 80
```

To compare the built-in chunking presets directly and emit a recommendation artifact:

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

On the current fixtures, the curated benchmark is effectively a latency tie and slightly favors the large preset, while the long-form `hells_angels` benchmark clearly rejects `large` and recommends the current default config (`target_words=420`, `min_words=180`, `overlap_words=64`, `max_heading_words=12`). Persist those comparison artifacts locally under `apps/api/benchmarks/results/` when you rerun the commands above.

For longer-document tuning, the runner also supports benchmarks that point at a real local PDF via `source_pdf`. The repo currently includes dedicated benchmark definitions for [`hells_angels_eval.json`](/home/chris/repos/ebook-rag/apps/api/benchmarks/hells_angels_eval.json), which exercises selected questions against the full 186-page [`hells_angels.pdf`](/home/chris/repos/ebook-rag/apps/api/benchmarks/local/hells_angels.pdf), and [`amazon_earnings_eval.json`](/home/chris/repos/ebook-rag/apps/api/benchmarks/amazon_earnings_eval.json), which exercises a shorter real-world earnings release with numeric, citation-granularity, and unsupported-answer checks against [`amazon_quarterly_earnings2025Q4.pdf`](/home/chris/repos/ebook-rag/apps/api/benchmarks/local/amazon_quarterly_earnings2025Q4.pdf):

```bash
uv run python scripts/run_eval.py --benchmark benchmarks/hells_angels_eval.json
uv run python scripts/run_eval.py --benchmark benchmarks/amazon_earnings_eval.json
```

The repo now also includes dedicated benchmark definitions for [`john_deere_mower_manual_eval.json`](/home/chris/repos/ebook-rag/apps/api/benchmarks/john_deere_mower_manual_eval.json), [`infinite_jest_eval.json`](/home/chris/repos/ebook-rag/apps/api/benchmarks/infinite_jest_eval.json), [`qwen3_technical_report_eval.json`](/home/chris/repos/ebook-rag/apps/api/benchmarks/qwen3_technical_report_eval.json), and [`gpt_5_4_thinking_card_eval.json`](/home/chris/repos/ebook-rag/apps/api/benchmarks/gpt_5_4_thinking_card_eval.json). Together with the local fixture library under [`apps/api/benchmarks/local/`](/home/chris/repos/ebook-rag/apps/api/benchmarks/local), those benchmarks broaden repeatable regression coverage across equipment manuals, very long literary prose, research-report style technical writing, and model/system-card style factual summaries:

```bash
uv run python scripts/run_eval.py --benchmark benchmarks/john_deere_mower_manual_eval.json
uv run python scripts/run_eval.py --benchmark benchmarks/qwen3_technical_report_eval.json
uv run python scripts/run_eval.py --benchmark benchmarks/gpt_5_4_thinking_card_eval.json
uv run python scripts/run_eval.py --benchmark benchmarks/infinite_jest_eval.json
```

The committed gating regression suite can be rerun with:

```bash
uv run python scripts/run_regression_suite.py
```

As of March 8, 2026, the committed regression suite is mostly stable on answer quality, but it is not fully green: a recent rerun flagged a `latency_p95_ms` regression on the John Deere manual benchmark. The recent ingestion-only changes also did not show a measurable end-to-end quality improvement. Treat the current ingestion heuristics as structural/debuggability improvements and add targeted benchmark cases before doing more ingestion tuning.

The `infinite_jest` benchmark is intentionally exploratory for now, while the manual/report/system-card benchmarks are intended to add stable coverage for exact spec lookup, front-matter noise, acronym-heavy technical QA, and unsupported-answer behavior beyond the earlier `hells_angels` and Amazon earnings harnesses.

To rerun the committed gating regression lane for those newer stable suites and compare against saved baselines:

```bash
uv run python scripts/run_regression_suite.py
```

That suite reads [`apps/api/benchmarks/regression_suite.json`](/home/chris/repos/ebook-rag/apps/api/benchmarks/regression_suite.json), refreshes the `*_latest` artifacts under [`apps/api/benchmarks/results/`](/home/chris/repos/ebook-rag/apps/api/benchmarks/results), compares each benchmark to its committed baseline, and exits non-zero on gating regressions. Refresh saved baselines intentionally with:

```bash
uv run python scripts/run_regression_suite.py --write-baselines
```

For page-local citation regression coverage, there is also a focused benchmark with sentence-level citation expectations:

```bash
uv run python scripts/run_eval.py --benchmark benchmarks/citation_granularity_eval.json
```

That fixture keeps excerpt-accuracy checks in the gating lane, while selected `hells_angels` excerpt checks are marked exploratory so they surface long-form citation drift without blocking every merge. The aggregate citation-evidence metric still appears in reports, but `--fail-on-regression` gates on the explicit regression lane rather than exploratory excerpt misses. The Amazon earnings benchmark remains a useful tracked benchmark, but it is not currently a clean pass in the live code path. A fresh March 8, 2026 rerun produced `answer_match_rate=0.8`, `citation_hit_rate=0.8`, `citation_evidence_hit_rate=0.8`, `gating_citation_evidence_hit_rate=0.8`, and `unsupported_precision=0.5`, with remaining misses around structured/page-local evidence selection and unsupported-answer behavior.

For regression tracking, the benchmark runner can also persist JSON and Markdown artifacts, compare a run against a saved baseline, and summarize per-document gating versus exploratory failures:

```bash
uv run python scripts/run_eval.py \
  --output-json benchmarks/results/latest.json \
  --output-markdown benchmarks/results/latest.md \
  --compare-to benchmarks/results/baseline.json \
  --fail-on-regression
```

The expanded summary now includes unsupported precision plus P50/P95 latency so retrieval and citation changes can be compared over time.

If you want the current project plan rather than the benchmark mechanics, see [`IMPLEMENTATION_PLAN.md`](/home/chris/repos/ebook-rag/IMPLEMENTATION_PLAN.md). The short version is: stop open-ended retrieval tuning, lock down one truthful baseline, and make grounded local-model-backed answering the next major quality milestone.

## Ingestion configuration

Chunking can now be configured through environment variables before ingestion or reprocessing:

- `CHUNK_TARGET_WORDS`
- `CHUNK_MIN_WORDS`
- `CHUNK_OVERLAP_WORDS`
- `CHUNK_MAX_HEADING_WORDS`

The current benchmark-backed default is:

- `CHUNK_TARGET_WORDS=420`
- `CHUNK_MIN_WORDS=180`
- `CHUNK_OVERLAP_WORDS=64`
- `CHUNK_MAX_HEADING_WORDS=12`

The active values are persisted on each document and surfaced in the web UI plus debug routes so you can tell how an existing index was built before deciding to reprocess it.

## Implementation plan

The active project plan now lives in [`IMPLEMENTATION_PLAN.md`](/home/chris/repos/ebook-rag/IMPLEMENTATION_PLAN.md).
