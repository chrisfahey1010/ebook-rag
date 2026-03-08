# ebook-rag Implementation Plan

Last updated: March 8, 2026

This plan is based on the current code in `apps/api` and `apps/web`, the project spec in [`ebook_rag_project_spec.md`](/home/chris/repos/ebook-rag/ebook_rag_project_spec.md), the top-level docs, and recent local commits through `305c6eb` (`Improve ingestion heading normalization`).

## Executive summary

The project is no longer an MVP. It already has a real ingestion pipeline, persisted chunks and vectors, hybrid retrieval, reranking, grounded QA, streaming answers, debug routes, and a meaningful benchmark harness. The main problem is not missing scaffolding. The main problem is that the answer layer is still mostly heuristic, while the docs and plan have drifted into a long running diary of benchmark work.

The shortest path to completion is:

1. stop treating open-ended benchmark tuning as the primary workstream
2. establish one trustworthy quality baseline and keep it honest
3. make local-model-backed answering a first-class pipeline stage
4. use local models selectively where they add intelligence, not everywhere by default
5. finish packaging, observability, and UX around that core

## Current state

## What is already implemented

- PDF upload, storage, deduplication, extraction, normalization, chunking, and embedding persistence
- document/page/chunk persistence in PostgreSQL with `pgvector`
- synchronous ingestion plus ingestion job/status records and document reprocessing
- configurable chunking and persisted chunk provenance
- dense retrieval, optional lexical retrieval, reciprocal-rank fusion, and reranking
- grounded QA with citations, trace payloads, and streamed answers
- debug endpoints for chunk inspection, retrieval inspection, and reranker inspection
- a usable web UI for upload, browsing, asking questions, and inspecting citations
- benchmark tooling with JSON and Markdown artifacts plus a committed regression suite

## What the code says is still weak

- answer generation is still split between:
  - a strong extractive fallback
  - a thin OpenAI-compatible generative provider
- the current generative path is a prompt wrapper, not a full grounded answer pipeline
- unsupported-answer behavior has improved with claim-level verification, but it is still fragile on harder document shapes
- ingestion is still synchronous despite queue-like status objects
- model/runtime visibility is still too implicit in the UI and docs
- benchmark artifacts and docs have drifted from one another

## What recent evidence says

As of March 8, 2026:

- the committed gating regression suite is the canonical release check, and the latest March 8, 2026 rerun is not green because of latency regressions
- the latest rerun preserved perfect answer-quality metrics on the current John Deere, Qwen 3 technical report, and GPT-5.4 thinking card gating suites, but flagged:
  - John Deere manual: `average_latency_ms`, `latency_p50_ms`, and `latency_p95_ms`
  - Qwen 3 technical report: `average_latency_ms`, `latency_p50_ms`, and `latency_p95_ms`
  - GPT-5.4 thinking card: `latency_p50_ms`
- the long-form `hells_angels` benchmark is still not V1-complete quality-wise
- the Amazon earnings benchmark is not actually green in the current code path:
  - the current saved March 8, 2026 artifact reports `retrieval_hit_rate=1.0`, `citation_hit_rate=0.9`, `citation_evidence_hit_rate=0.5`, `gating_citation_evidence_hit_rate=0.625`, `support_accuracy=0.9`, `answer_match_rate=0.5`, and `unsupported_precision=0.5`
  - that directly conflicts with some README/plan claims, so the docs are overstating current quality

## Core diagnosis

The project has spent a lot of effort improving retrieval heuristics and benchmark coverage. That was worthwhile, but it is no longer the main missing capability. The main missing capability is intelligent answer orchestration.

Right now the system is best described as:

- good retrieval and evidence plumbing
- decent extractive answering
- incomplete local-LLM answer synthesis
- strong debugability
- inconsistent project-level truth about what is actually passing

That means the completion plan should shift from "keep tuning retrieval until all benchmarks are perfect" to "use the existing retrieval foundation to build a stronger grounded answer stack."

## V1 completion definition

The project should be considered V1 complete when all of the following are true:

- a developer can run the stack locally with no paid APIs
- users can upload, inspect, reprocess, ask, and verify citations end to end
- the default local stack uses real local providers for embeddings, reranking, and answer generation
- answer generation is materially more intelligent than pure extractive selection while remaining grounded
- unsupported answers are conservative and reliable
- benchmark docs, saved artifacts, and current code behavior agree with each other
- the regression suite gates on stable quality checks instead of encouraging endless tuning loops
- the web app makes the model/runtime path and answer evidence path visible enough for demos

## Plan of record

### Phase 0: Re-establish a truthful baseline

This should happen first. It is small, but it fixes the project-management problem that is causing churn.

#### Goals

- align docs, benchmark artifacts, and current code behavior
- define which benchmarks are gating, which are exploratory, and why
- stop treating every observed miss as a retrieval-tuning invitation

#### Work

- rerun and refresh the saved benchmark artifacts that are supposed to represent current truth
- update `README.md` and any benchmark claims that no longer match the actual outputs
- make the regression suite the canonical "green or not green" indicator
- explicitly mark:
  - stable gating suites
  - exploratory long-form suites
  - known failing suites that are tracked but not release-blocking yet
- document the current answer-quality gaps instead of burying them in prose

Current benchmark status after this cleanup:

- Stable gating suites: `john_deere_mower_manual`, `qwen3_technical_report`, `gpt_5_4_thinking_card`, and `citation_granularity`
- Exploratory long-form coverage: `infinite_jest`, plus the exploratory checks inside `hells_angels`
- Tracked but not release-blocking: `amazon_earnings`, and the remaining long-form misses in `hells_angels`

#### Exit criteria

- there is one trusted quality baseline
- the project no longer contains contradictory benchmark claims
- new work can be prioritized against real gaps rather than stale success claims

### Phase 1: Productize the local-model runtime path

The provider seams exist already. What is missing is a deliberate local runtime story.

#### Goals

- make fully local mode easy to run and easy to understand
- keep the OpenAI-compatible path first-class
- establish recommended model/runtime presets instead of leaving everything implicit

#### Work

- document and test three supported runtime modes:
  - fully local
  - mixed local plus hosted
  - hosted OpenAI-compatible
- add explicit config presets for:
  - local embeddings
  - local reranking
  - local answer generation
- expose the active providers and model names in API debug output and the web UI
- document the reindex/reprocess workflow whenever embedding dimensions or embedding models change

#### Recommended local stack

- embeddings: keep using a local sentence-transformer style embedding model
- reranking: keep using the local cross-encoder path as the default quality layer
- generation: add a documented OpenAI-compatible local runtime preset for a small instruct model such as Qwen 3.5/4B

#### Important constraint

Qwen 3.5/4B should be treated as the answer model, not as the answer to every model slot. It is a good fit for grounded synthesis, query decomposition, and answer verification. It is not the right default embedding strategy, and it should not replace the reranker.

#### Exit criteria

- a new developer can select a local preset without reverse-engineering env vars
- the UI shows which providers are active
- the local stack is documented as a primary path, not an implied one

### Phase 2: Replace "dumb answering" with grounded answer orchestration

This is the highest-leverage remaining engineering work.

#### Goals

- make answers feel intelligent without losing grounding
- preserve extractive fallback behavior when local generation is weak
- stop overloading the retriever with work that belongs in the answer layer

#### Planned answer pipeline

1. retrieve and rerank candidate chunks
2. assemble a compact evidence set rather than one generic context blob
3. classify the question type
4. choose an answer mode:
   - extractive lookup
   - multi-evidence synthesis
   - unsupported
5. generate or compose the answer
6. verify that each answer claim is supported by cited evidence
7. return confidence/support metadata with citations

#### Status on March 8, 2026

Implemented:

- question routing between extractive, synthesis, and unsupported modes
- OpenAI-compatible grounded synthesis
- explicit confidence/support fields on QA responses
- claim-level answer verification with per-claim support scores and trace metadata
- optional OpenAI-compatible local-model verification for answer claims

Still missing:

- task-specific unsupported-classification prompts as a first-class stage rather than a downgrade outcome
- stronger benchmark-driven validation that the new verifier materially improves the harder failing suites

#### Concrete backend work

- add a question router that distinguishes:
  - exact lookup questions
  - multi-part synthesis questions
  - unsupported or missing-evidence cases
- add an LLM-backed grounded answer provider for small local models through the existing OpenAI-compatible interface
- keep the extractive provider, but use it as a deliberate mode rather than the default intelligence layer
- move prompt construction from a single generic prompt to task-specific prompts:
  - exact-answer extraction
  - evidence-grounded synthesis
  - unsupported classification
- expand the new answer verification pass:
  - keep the current lightweight claim-level verifier
  - benchmark and tune the optional local-model verifier on the harder unsupported/citation suites
  - reject or downgrade answers whose claims are not grounded in the cited spans
- improve citation assembly so citations represent actual supporting spans, not just nearby chunks
- add explicit confidence/support fields to the QA response

#### Why this is the right place for Qwen 3.5/4B

This is where a small local model can materially improve the system:

- synthesizing across multiple retrieved chunks
- rewriting clumsy extractive sentences into clearer grounded answers
- deciding when evidence is insufficient
- handling composite questions more naturally than the current heuristic path

If the model is weak, the pipeline should degrade gracefully by routing more questions to extractive mode or unsupported mode.

#### Exit criteria

- answers are clearly more coherent than the current extractive baseline
- unsupported precision improves on the harder benchmarks beyond the current claim-level verification baseline
- citations remain faithful after the new answer synthesis step
- local generation improves quality without making the system feel untrustworthy

### Phase 3: Use local models in the rest of the pipeline where they actually help

Local models should be added selectively. The right criterion is leverage, not novelty.

#### High-value uses beyond answer generation

- query decomposition:
  - split composite questions into atomic retrieval intents
- query rewriting:
  - generate a retrieval-friendly query for acronym-heavy or awkward user questions
- unsupported classification:
  - use a compact verifier prompt to decide whether evidence is sufficient
- answer verification:
  - check that each sentence in the draft answer is backed by the selected citations
- optional ingestion enrichment:
  - generate short document summaries, chapter summaries, or normalized titles for UI search and browsing

#### Lower-priority uses

- ingestion cleanup heuristics driven by an LLM
- autonomous metadata fixing
- expensive per-document summarization during ingestion

Those may be interesting later, but they are not on the shortest path to completion.

#### Exit criteria

- at least one non-answer stage meaningfully benefits from a local model
- those additions are benchmarked and optional
- the default pipeline remains understandable and fast

### Phase 4: Tighten ingestion and retrieval only where benchmarks justify it

This phase is intentionally narrower than the recent work.

#### Goals

- fix concrete misses
- avoid reopening open-ended retrieval tuning

#### Work

- add targeted benchmark cases for current known misses:
  - page-local table lookup
  - employee-count style row extraction
  - long-form exact-page citation misses
  - unsupported-answer false positives
- improve table-aware and page-local evidence extraction where current failures demand it
- revisit chunk sizing or normalization only when a benchmark failure is clearly caused there
- keep ingestion synchronous for V1 unless measured upload latency becomes unacceptable

#### Policy

No more retrieval or ingestion tuning without one of:

- a new regression failure
- a new benchmark that captures a real user-visible miss
- a clear production-facing UX problem

#### Exit criteria

- known benchmark misses are either fixed or explicitly deprioritized
- retrieval is good enough to support the answer layer, not "perfect"

### Phase 5: Frontend and observability finish

The UI already works. The remaining work is trust, inspectability, and polish.

#### Goals

- make the pipeline understandable in demos
- make local-model behavior visible

#### Work

- expose active provider/model information in the QA workspace
- show whether a response came from:
  - extractive mode
  - local generative mode
  - unsupported mode
- surface support/confidence metadata
- make selected evidence easier to compare against the final answer
- tighten failure states and loading states
- consider splitting the current UI into clearer library/workspace/debug views if the current single-page layout feels overloaded

#### Exit criteria

- a user can tell how an answer was produced
- a recruiter can understand the system without a narrated walkthrough

### Phase 6: Packaging and release hygiene

#### Goals

- make the repo easy to run
- make the architecture easy to explain

#### Work

- add an example `.env`
- document one recommended local setup from zero to first answer
- document one recommended mixed setup
- add a short architecture diagram and request-flow diagram
- add or tighten CI around tests and regression checks

#### Exit criteria

- the repo has a clear happy path
- V1 can be presented as a finished system, not an ongoing experiment

## Priority order

If the goal is the shortest path to completion, the order should be:

1. Phase 0: truthful baseline
2. Phase 1: local runtime productization
3. Phase 2: grounded answer orchestration
4. Phase 3: selective local-model upgrades outside answer generation
5. Phase 4: targeted ingestion and retrieval fixes only where justified
6. Phase 5: frontend and observability finish
7. Phase 6: packaging and release hygiene

## Immediate next implementation slice

The best next coding slice is:

1. clean up the benchmark truth problem
   - refresh artifacts
   - remove overstated success claims
   - mark Amazon and long-form gaps honestly
2. add runtime visibility
   - active embedding provider
   - active reranker
   - active answer provider and model
3. implement the first real local-LLM answer mode
   - keep extractive as fallback
   - add grounded synthesis for multi-evidence questions
   - add a stricter unsupported gate

That slice is the point where the project should stop feeling like "good retrieval plus clever matching" and start feeling like "an intelligent, grounded local document QA system."

## Progress update

As of March 8, 2026, the first pass of that slice has landed in code:

- QA responses now report routed answer modes (`extractive`, `synthesis`, `unsupported`)
- the backend now returns support/confidence metadata plus router rationale
- the web UI now exposes active embedding, reranker, and answer runtime details
- the OpenAI-compatible QA path now includes a dedicated unsupported-classification prompt before generation on higher-risk questions such as multi-facet, numeric/date-constrained, and lower-confidence retrieval cases
- the QA finalization path can now salvage partially supported generated answers by rewriting or trimming them down to verified supported claims before falling back to `unsupported`
- the QA finalization path now also applies a stricter acceptance gate for constrained numeric/date questions so repaired answers still have to clear citation-backed facet coverage and exact-value alignment checks before they survive

That means the next follow-up work should narrow to:

1. refresh benchmark artifacts and remove any remaining overstated quality claims
2. benchmark the new router/support thresholds and stricter final acceptance gate against Amazon and the stable regression suites
3. inspect any remaining Amazon misses with the richer QA trace before deciding whether the next fix belongs in answer postprocessing or citation selection

The benchmark tooling now also captures structured per-failure QA traces in saved eval artifacts, including router decisions, verification/postprocess state, retrieved/cited contexts, timings, and prompt snapshots. That means Amazon and other structured-evidence misses no longer need to be debugged only through ad hoc reruns.

The latest implementation follow-up for that slice should now be:

1. use the richer QA trace fields to inspect router support scores, unsupported-classifier decisions, question-coverage scores, and repair outcomes on each benchmark miss
2. keep the stricter final acceptance gate for constrained numeric/date questions so partially supported financial answers fall back to `unsupported` instead of surviving repair
3. rerun `amazon_earnings` plus the stable gating suites and treat any new regressions as answer-layer bugs first, not retrieval-tuning invitations

## Resolved decisions

These are the working decisions for V1:

- V1 can ship with synchronous ingestion.
  - Rationale: the current product risk is answer quality, not ingestion architecture.
  - Condition: keep status/reprocessing controls, and only add background jobs if upload latency becomes a repeated user-visible problem.
- Local answer generation should optimize for best quality on a single small local model, not absolute minimum footprint.
  - Rationale: this repo is trying to become a serious local QA system, and the main missing capability is grounded answer intelligence.
  - Practical target: design around a strong small local model such as Qwen 3.5/4B, then let weaker setups fall back more often to extractive or unsupported modes.
- Amazon and long-form book QA should not both be hard V1 release gates.
  - Amazon earnings should remain a tracked near-term quality target because it exercises the structured, citation-sensitive failures the current system still has.
  - the long-form `hells_angels` and `infinite_jest` style suites should remain explicit stretch/exploratory coverage until the new answer layer is in place and stable.
