# ebook-rag Implementation Plan

This plan replaces the temporary "Next implementation plan" section that was in `README.md`.

It is based on:

- the product goals in [`ebook_rag_project_spec.md`](/home/chris/repos/ebook-rag/ebook_rag_project_spec.md)
- the current project summary in [`README.md`](/home/chris/repos/ebook-rag/README.md)
- the implementation currently present in `apps/api` and `apps/web`

## Current baseline

The project has already cleared the first end-to-end milestone:

- PDF upload and local file storage
- PyMuPDF extraction
- per-page text persistence
- paragraph-aware chunking with page spans
- embedding generation during ingestion
- vector retrieval from PostgreSQL/`pgvector`
- reranking stage
- grounded QA with citations
- retrieval/QA debug surfaces
- a usable browser workflow for upload, library browsing, question asking, and citation inspection

The main gap is that several critical pieces are still baseline implementations:

- the provider layer is only partially complete:
  - embeddings now support hashing, `sentence-transformers`, and OpenAI-compatible endpoints
  - reranking now supports token overlap, local cross-encoders, and an OpenAI-compatible adapter
  - answer generation still needs clearer local-runtime presets and documentation
- context assembly is improved but still heuristic rather than benchmark-tuned
- changing embedding dimensions now requires a migration plus document reprocessing, so the reindex workflow needs to stay explicit in docs and tooling
- ingestion is still synchronous, even though status and reprocessing APIs from the spec are now in place
- evaluation now includes saved JSON/Markdown benchmark artifacts, baseline comparison, and a curated local fixture, but the dataset is still small

## Planning principles

The next milestones should preserve four constraints:

1. Keep the OpenAI-compatible path first-class.
2. Make small local models viable, not just technically possible.
3. Improve retrieval quality before adding broad product surface area.
4. Add observability and evaluation alongside each quality improvement so regressions are measurable.

For this project, "small local models viable" means:

- the default development path can run without cloud APIs
- embedding, reranking, and answer generation can each use local providers
- the system remains useful when the local generation model is weak by relying on stronger retrieval, conservative prompting, and extractive fallback behavior
- provider interfaces stay narrow so Ollama, llama.cpp server, vLLM, and OpenAI-compatible endpoints can share most code paths

## Recommended milestone order

### Milestone 1: Finish the provider abstraction layer

This milestone is in progress. The embedding and reranker halves are now implemented; the remaining work is to finish the answer-provider side and document concrete local-runtime presets cleanly.

#### Goals

- upgrade embeddings from hashing baseline to real local/remote model providers
- expand the answer provider abstraction into a broader provider layer
- make local inference options explicit instead of implicit

#### Backend work

- add concrete embedding providers:
  - `SentenceTransformerEmbeddingProvider` for local embeddings
  - `OpenAICompatibleEmbeddingProvider` for remote or local OpenAI-style servers
- add concrete reranker providers:
  - `TokenOverlapReranker` kept as fallback baseline
  - `CrossEncoderReranker` for local model-backed reranking
  - optional OpenAI-compatible reranker adapter only if it stays simple
- broaden answer provider support:
  - keep `ExtractiveAnswerProvider`
  - keep `OpenAICompatibleAnswerProvider`
  - add explicit config presets/documentation for Ollama and llama.cpp server through the same compatible interface
- separate provider configuration for:
  - embeddings
  - reranking
  - answer generation
  - model names
  - base URLs
  - timeouts
  - dimensions where applicable

#### Status

- completed:
  - separate provider configuration for embeddings, reranking, and answer generation
  - `SentenceTransformerEmbeddingProvider`
  - `OpenAICompatibleEmbeddingProvider`
  - `CrossEncoderReranker`
  - `OpenAICompatibleReranker`
  - configurable pgvector dimensions with a reprocess workflow for existing documents
  - provider-selection and failure-handling tests for embeddings and reranking
- remaining:
  - document supported local presets for Ollama and llama.cpp-style OpenAI-compatible servers
  - expand answer-provider documentation so mixed-mode setups are explicit rather than implied

#### Why this comes first

- retrieval quality work is blocked on real embedding and reranking models
- it satisfies the spec’s pluggable inference requirement
- it preserves your preference for OpenAI endpoints while making small local models a real supported path

#### Exit criteria

- a developer can run the full stack with:
  - all-local defaults
  - OpenAI-compatible hosted endpoints
  - mixed mode, such as local embeddings plus remote generation
- provider selection is fully environment-driven
- tests cover provider selection and failure handling

### Milestone 2: Raise retrieval quality to a real V1 standard

Once the remaining provider-layer documentation is in place, retrieval quality should become the main focus.

#### Goals

- replace baseline retrieval heuristics with a robust retrieve-then-rerank pipeline
- keep the system effective for weaker local generators by making retrieval do more of the work

#### Backend work

- upgrade dense retrieval to use real embeddings
- introduce optional lexical retrieval:
  - PostgreSQL full-text search first
  - BM25 later only if needed
- implement hybrid retrieval with configurable blending
- improve reranking with a model-backed reranker
- add query preprocessing that stays conservative:
  - whitespace normalization
  - basic punctuation cleanup
  - optional acronym/query expansion only if benchmarked
- continue improving context assembly:
  - completed:
    - deduplicate near-identical chunks
    - pull adjacent chunks into the final context window when useful
    - enforce a token budget
  - remaining:
    - tune diversity/selection behavior against benchmarks instead of heuristics alone
    - continue improving citation precision beyond the current selected-context vs cited-evidence split

#### Why this is the second milestone

- it delivers the biggest product improvement
- it directly improves small-local-model usability
- it makes later streaming and UX work more valuable because the underlying answers will be better

#### Exit criteria

- benchmark scores improve materially over the current baseline
- retrieval results expose dense, lexical, rerank, and fused scores where applicable
- answer support rate improves without increasing hallucinated answers

### Milestone 3: Strengthen ingestion quality and ingestion controls

The ingestion pipeline works, but it is still closer to a synchronous MVP than the system described in the spec.

#### Goals

- make ingestion more traceable and more resilient
- improve chunk quality before more advanced retrieval tuning

#### Backend work

- completed:
  - `GET /api/ingestion/{document_id}/status`
  - `POST /api/ingestion/{document_id}/reprocess`
- remaining:
  - improve text normalization:
    - detect repeated headers/footers where feasible
    - preserve headings more intentionally
    - keep page mappings explicit
  - improve chunk metadata:
    - heading or section label when detectable
    - chunk character span or provenance metadata if useful
  - revisit chunk sizing:
    - current chunking is much smaller than the spec target
    - benchmark a few chunk sizes instead of choosing by intuition
  - decide whether ingestion remains synchronous for V1 or moves to background jobs

#### Recommendation

Keep synchronous ingestion for the immediate next milestone unless upload latency becomes painful. Add ingestion status and reprocessing first; add a queue only when real evidence justifies the complexity.

#### Exit criteria

- users can reprocess a document after changing models or chunking settings
- ingestion failures are inspectable without reading logs
- chunk structure is benchmarked and documented

### Milestone 4: Make grounded answering more robust

The answer layer is working, but it is still conservative in simple ways rather than intentionally designed.

#### Goals

- improve grounded answer quality
- keep the system safe when using weaker local models
- expose better answer traces

#### Backend work

- upgrade prompt construction:
  - require evidence-only answers
  - require citation-aware phrasing
  - distinguish unsupported questions clearly
- improve citation selection:
  - cite only the chunks actually used
  - prefer fewer, better citations over dumping the top 3
- add support/confidence metadata derived from retrieval and answer behavior
- add `POST /api/qa/ask-stream`
- consider optional answer trace persistence for dev mode

#### Local-model guidance

For smaller local models, favor:

- short prompts
- smaller, high-quality context windows
- conservative unsupported thresholds
- extractive fallback when generation quality is poor

#### Exit criteria

- streaming answers work in the API and web app
- answer traces show selected context and timing clearly
- unsupported responses are consistent and trustworthy across providers

### Milestone 5: Build the developer-facing debug and evaluation layer into a real regression system

This project should be able to prove quality improvements, not just demonstrate features.

#### Goals

- turn the current benchmark script into a repeatable regression workflow
- make debugging retrieval changes fast

#### Backend and tooling work

- completed:
  - benchmark runner can persist JSON and Markdown summaries
  - benchmark runs can be compared against a prior baseline with regression-friendly metrics
- expand the benchmark dataset from synthetic smoke tests to a small curated eval set:
  - 3 to 5 real PDFs
  - 10 to 20 questions per PDF
  - expected citation pages
  - expected support labels
- persist or snapshot benchmark results so runs are comparable over time
- add more debug APIs from the spec:
  - `GET /api/debug/documents/{document_id}/chunks`
  - `POST /api/debug/rerank`
- optionally add a simple benchmark report artifact in JSON or Markdown

#### Recommended metrics

- retrieval hit rate
- citation hit rate
- support accuracy
- answer match rate
- unsupported precision
- latency percentile, not just average latency

#### Exit criteria

- major retrieval or provider changes are evaluated before merge
- debug surfaces make it obvious why a question failed

### Milestone 6: Finish the portfolio-quality frontend

The current UI is functional. The next step is to make it more inspectable and more deliberate without adding product bloat.

#### Goals

- polish the core QA workspace
- expose pipeline internals in a way that helps trust and debugging

#### Frontend work

- split the current single-page UI into clearer sections or routes:
  - library
  - document workspace
  - debug/retrieval inspection
- surface ingestion status and reprocessing actions
- support streamed answer rendering
- improve citation browsing:
  - stronger page/chunk labels
  - clearer selected-citation state
  - easier comparison between answer text and source text
- add model/provider visibility in the UI for debugging
- preserve the current local-first simplicity; avoid overbuilding account-style product chrome

#### Exit criteria

- the app is clearly demoable to a recruiter without explanation
- retrieval and answer traces are easy to inspect in the browser
- failure states look intentional rather than incidental

### Milestone 7: Operational polish and project packaging

This final milestone is about making the repository easy to run, evaluate, and present.

#### Goals

- make setup predictable
- reduce hidden configuration knowledge
- package the repo as a serious portfolio project

#### Work

- improve setup docs for:
  - fully local mode
  - OpenAI-compatible remote mode
  - hybrid mode
- add example `.env` configuration
- consider Docker support for the API and web app if it helps local onboarding
- document recommended small local models for:
  - embeddings
  - reranking
  - generation
- add a concise architecture diagram and request-flow diagram
- add CI checks for tests and linting if not already present

#### Exit criteria

- a new developer can run the stack without reverse-engineering env vars
- the repository communicates its architecture and tradeoffs quickly

## Suggested milestone breakdown by priority

If you want the shortest path to a strong V1, use this order:

1. Provider abstraction for embeddings, reranking, and generation
2. Real retrieval quality improvements with hybrid retrieval and model reranking
3. Ingestion status/reprocessing plus chunking improvements
4. QA streaming, citation selection, and stronger context assembly
5. Expanded eval/debug system
6. Frontend polish
7. Packaging and documentation refinement

## Suggested definition of "V1 complete"

The project should be considered V1 complete when all of the following are true:

- users can upload a PDF, inspect ingestion, reprocess it, ask questions, and verify citations
- the default stack works locally without paid APIs
- OpenAI-compatible endpoints remain supported for stronger hosted models
- retrieval uses real embeddings plus reranking, with optional lexical blending
- the app can stream answers and expose retrieval/debug traces
- there is a benchmark set that catches obvious regressions
- the frontend presents the workflow and debug surfaces cleanly enough for portfolio use

## Immediate next implementation slice

The best next coding slice is:

1. expand the curated eval set beyond the initial fixture so page-level citation expectations cover more retrieval failure modes
2. tune retrieval/context selection against those benchmark failures, especially diversity and adjacent-chunk behavior
3. tighten citation attribution further so cited evidence narrows from chunk-level support toward the exact supporting spans

That sequence keeps the new regression tooling in the loop, builds on the initial cited-evidence trace split, and continues pushing core grounded-answering quality higher before adding more product surface area.
