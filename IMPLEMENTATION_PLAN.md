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
- retrieval quality is now in a satisfactory V1 state on the current benchmark set:
  - the `hells_angels.pdf` long-form benchmark now passes on retrieval hit rate, citation hit rate, answer match rate, support accuracy, and unsupported precision
  - the repo now also includes `amazon_earnings_eval.json`, which exercises `amazon_quarterly_earnings2025Q4.pdf` as a shorter real-world financial-report benchmark
  - the local fixture library under `apps/api/benchmarks/local/` has also expanded with `john_deere_mower_manual.pdf`, `infinite_jest.pdf`, `qwen3_technical_report.pdf`, and `gpt-5-4_thinking_card.pdf` to broaden future eval coverage across manuals, very long prose, technical reports, and model/system-card style documents
  - the latest QA passes improved structured-numeric citation evidence coverage on that Amazon benchmark, fixed the prior `net sales` versus `operating income` guidance confusion, closed the exploratory employee-count row on page 13 by recovering split same-page table rows, and taught the extractive scorer to prefer the exact page-11 free-cash-flow metric row over the narrative page-2 summary for exact-value questions
  - the March 8, 2026 citation-assembly pass closed the remaining Amazon-specific narrative citation-evidence gap for the "why did free cash flow decrease" question, so the benchmark now clears its current gating lane end to end
  - further retrieval tuning should be driven by new benchmark coverage or clear regressions, not by open-ended score chasing
- context assembly is improved and benchmark-informed, but still heuristic
- changing embedding dimensions now requires a migration plus document reprocessing, so the reindex workflow needs to stay explicit in docs and tooling
- ingestion is still synchronous, even though status and reprocessing APIs from the spec are now in place
- evaluation now includes saved JSON/Markdown benchmark artifacts, baseline comparison, a broader curated local fixture, a real-book long-form benchmark, and additional debug routes
- the local benchmark fixture set now spans book-style prose, a quarterly earnings report, an equipment manual, a long literary ebook, a technical model report, and a model/system-card style document to expand future citation and unsupported-answer coverage beyond the original book-focused cases
- benchmark definitions now support inherited suite/document `regression_tier` defaults and benchmark reports now summarize per-document gating versus exploratory failures so the regression lane is explicit instead of implicit
- the remaining quality risk is benchmark breadth rather than the specific long-document misses that were driving recent tuning loops
- ingestion now persists the chunking configuration used for each document, stores per-chunk provenance metadata for debug inspection, and exposes reprocessing/status controls in the UI
- ingestion now groups stacked heading lines into single heading blocks, strips repeated two-line header/footer boundary blocks, and records heading-path provenance on chunks for more faithful section metadata
- chunk sizing is now configurable and benchmarkable through the eval runner, but the chosen defaults are still heuristic rather than benchmark-locked
- the March 8, 2026 ingestion-quality pass improved normalization and heading metadata structure, but the full benchmark sweep did not show a measurable end-to-end quality lift on the current suites, so the next ingestion work should be driven by targeted benchmark additions rather than more heuristic tweaking

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

This milestone is now in a good stopping state for V1. Retrieval quality should only come back to the top of the queue when new benchmark coverage uncovers a real miss pattern.

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
    - prefer direct evidence over broader summary chunks when selecting answer context
    - require support for each facet of a composite question instead of answering from the strongest partial match
    - boost distinctive query-term coverage and rare-term matches in lexical retrieval, fusion, and fallback reranking for long-document fact questions
    - expand extractive answer selection to tolerate abbreviated split sentences and score wider evidence spans when facts cross sentence boundaries
    - penalize metadata/front-matter-heavy matches and boost exact multi-term query runs during retrieval/context selection
    - tighten unsupported-answer rejection for broad topical questions on long books using stronger distinctive-term support checks
    - add benchmark-aware handling for explicit-date questions, nickname/alias questions, and named-subject matching in long documents
    - let the local extractive answerer score across retrieved candidates directly instead of over-pruning through the LLM-oriented context window
  - remaining:
    - expand benchmark coverage so retrieval work is triggered by new evidence rather than repeated tuning on the same fixture
    - revisit diversity/selection behavior only if a new benchmark or regression shows a concrete failure mode

#### Why this is the second milestone

- it delivers the biggest product improvement
- it directly improves small-local-model usability
- it makes later streaming and UX work more valuable because the underlying answers will be better

#### Exit criteria

- benchmark scores improve materially over the current baseline
- retrieval results expose dense, lexical, rerank, and fused scores where applicable
- answer support rate improves without increasing hallucinated answers

### Milestone 3: Strengthen ingestion quality and ingestion controls

This milestone is now in progress with the first implementation slice completed. The ingestion pipeline is more inspectable, but it is still closer to a synchronous MVP than the full system described in the spec.

#### Goals

- make ingestion more traceable and more resilient
- improve chunk quality before more advanced retrieval tuning

#### Backend work

- completed:
  - `GET /api/ingestion/{document_id}/status`
  - `POST /api/ingestion/{document_id}/reprocess`
  - persisted per-document chunking configuration used during ingestion
  - persisted per-chunk provenance metadata for page/paragraph inspection
  - persisted chunk character-span provenance in normalized page text for page-local debugging
  - exposed chunk provenance through retrieval results, QA citations, and QA traces
  - web UI controls for ingestion status refresh and document reprocessing
  - eval runner support for chunking presets and overrides
- completed:
  - detect repeated two-line header/footer boundary blocks during normalization
  - preserve consecutive heading lines as a single heading block instead of fragmenting stacked headings
  - record heading-path provenance on chunks while keeping the chunk heading field focused on the most specific heading line
- remaining:
  - add targeted benchmark coverage for repeated multi-line headers/footers and stacked heading blocks so ingestion changes are measured directly
  - keep page mappings explicit in any future normalization changes
  - revisit chunk sizing:
    - current chunking defaults are still smaller than the spec target
    - benchmark a few chunk sizes and promote one to the documented default instead of choosing by intuition
  - decide whether ingestion remains synchronous for V1 or moves to background jobs

#### Recommendation

Keep synchronous ingestion for the immediate next milestone unless upload latency becomes painful. Add ingestion status and reprocessing first; add a queue only when real evidence justifies the complexity.

#### Exit criteria

- users can reprocess a document after changing models or chunking settings
- ingestion failures are inspectable without reading logs
- chunk structure is benchmarked and documented
- ingestion-specific normalization and heading heuristics are covered by targeted regression fixtures rather than only by broad end-to-end suites

### Milestone 4: Make grounded answering more robust

This milestone is now in progress with the first streaming slice completed. The answer layer is more usable in the UI, but there are still a few quality and observability gaps before it should be considered finished.

#### Goals

- improve grounded answer quality
- keep the system safe when using weaker local models
- expose better answer traces

#### Backend work

- completed:
  - `POST /api/qa/ask-stream`
  - streamed answer rendering in the web app
  - terminal streamed payloads that include the same final answer, citations, and optional trace shape as `POST /api/qa/ask`
- remaining:
  - upgrade prompt construction:
    - require evidence-only answers
    - require citation-aware phrasing
    - distinguish unsupported questions clearly
  - improve citation selection:
    - cite only the chunks actually used
    - prefer fewer, better citations over dumping the top 3
  - add support/confidence metadata derived from retrieval and answer behavior
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
  - expanded curated eval set with harder multi-page and citation-coverage cases
  - benchmark runner can now evaluate either inline synthetic PDFs or real local PDFs via `source_pdf`
  - added a dedicated long-form benchmark against `hells_angels.pdf`
  - added `amazon_quarterly_earnings2025Q4.pdf` under `apps/api/benchmarks/local/` as a future real-world eval fixture for shorter financial-report-style coverage
  - added a focused citation-granularity benchmark with excerpt-level expectations
  - split benchmark citation-granularity checks into gating versus exploratory tiers
  - benchmark summaries now report citation evidence hit rate alongside page-level citation hit rate
  - `GET /api/debug/documents/{document_id}/chunks`
  - `POST /api/debug/rerank`
- remaining:
  - keep expanding the benchmark dataset toward harder multi-page citation and unsupported-answer cases
  - keep refining which long-form citation-granularity checks should graduate from exploratory to regression-gating
  - keep tightening benchmark-level gating/exploratory defaults so stable suites stay terse and unstable long-form suites stay explicit
- persist or snapshot benchmark results so runs are comparable over time
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
  - completed:
    - streamed answer text now lands progressively before the final citations/trace payload
    - stronger page/chunk labels
    - clearer selected-citation state
    - easier comparison between answer text and source text
    - page-local citation inspection using normalized-text offsets and paragraph spans
  - remaining:
    - carry the same page-local evidence comparison into future streamed-answer UX
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

The best next coding slice is now:

1. close the remaining Milestone 1/7 documentation gap
   - document recommended local embedding, reranker, and generation model combinations more explicitly
   - keep fully local, mixed, and hosted OpenAI-compatible setups easy to discover from the docs
2. add model/provider visibility in the UI for debugging
   - show which embedding, reranker, and answer providers are currently active
   - keep the streamed-answer workspace inspectable instead of turning it into a chat-like black box
3. tighten the remaining unstable long-form questions
   - keep `infinite_jest` exploratory by default
   - continue rewording or narrowing `hells_angels` and other long-form excerpt checks before promoting them into the gating lane

The chunking decision itself is now benchmark-backed for the current fixture set, normalization now preserves heading blocks while collapsing soft-wrapped body lines more cleanly, chunk provenance now includes character-span offsets for page-local inspection, the benchmark workflow now distinguishes page-level citation success from excerpt-level citation accuracy, and the QA workspace now streams answer text while preserving the same final citations and trace payloads as the synchronous route. On the March 8, 2026 run after the citation-assembly pass, the Amazon earnings benchmark reports `answer_match_rate=1.0`, `citation_hit_rate=1.0`, `citation_evidence_hit_rate=1.0`, `support_accuracy=1.0`, and `unsupported_precision=1.0`, while holding `gating_citation_evidence_hit_rate=1.0`. The page-11 free-cash-flow row, the page-13 employee-count row, and the narrative free-cash-flow explanation now all pass in the gating lane. The broader local fixture corpus is also now fully wired into dedicated benchmark definitions, and those definitions support inherited suite/document `regression_tier` defaults plus per-document failure summaries so the intended regression lane is visible in both config and reports.

The streaming QA slice is now complete. The immediate follow-up is to make runtime configuration more visible and easier to run in fully local or mixed modes, while continuing to promote only stable long-form citation checks into the gating lane.
