# CommerceBench AI

CommerceBench AI is a reproducible benchmark and experimentation platform for
evaluating LLM capabilities in realistic commerce customer-service scenarios.

The goal is not to build a generic customer-service chatbot.
The goal is to measure, diagnose, and improve LLM capability under
reproducible commerce scenarios.

## Focus areas

- Retrieval / RAG
- Multi-turn dialogue
- Context and memory
- Task completion
- Reliability / hallucination
- Evaluation
- Controlled ablation experiments
- Regression testing

## Status

Current status: project bootstrap.
No benchmark result or performance claim has been produced yet.

## Current architecture (Phase 0)

Phase 0 establishes a minimal, deterministic evaluation kernel:

- `CaseSpec` — one benchmark case (input, required facts, forbidden
  claims, optional expected response)
- `ExperimentManifest` — how one experiment is configured, with a
  deterministic SHA-256 configuration fingerprint
- `RunTrace` — a factual record of one case execution (no scores)
- `EvaluationResult` — an evaluator's judgement of a trace, kept strictly
  separate from the trace itself
- Deterministic runner — `run_case(case, manifest, system)` produces a
  `RunTrace`
- Deterministic evaluator — `exact_match`, `required_fact_coverage`, and
  `forbidden_claim_violation` metrics with a fixed pass gate

Phase 0 does not yet include a real LLM, RAG pipeline, multi-turn
dialogue environment, memory strategy, or production benchmark dataset.
The JSON files under `examples/phase0/` are development fixtures, not a
benchmark.

## Phase 1A — RAG Vertical Slice

Phase 1A adds the first end-to-end retrieval slice on top of the
evaluation kernel:

- `Document` / `Corpus` — minimal retrievable-unit and versioned
  collection contracts (`commercebench/rag/`)
- Two deterministic, offline, stdlib-only retrievers with explicit
  ranking semantics (`document_ids[i]` is the rank `i+1` result):
  `keyword-match-v0` (distinct query-term overlap) and
  `term-frequency-v0` (in-document term occurrence counts)
- `DeterministicRAGSystem` — a `SystemUnderTest` that retrieves `top_k`
  documents, records a `RetrievalEvent`, and answers from a fixed
  `document_id -> answer` fixture map (no LLM is called)
- `RetrievalEvaluator` — `retrieval_recall_at_K`,
  `retrieval_precision_at_K`, `retrieval_mrr`, and binary
  `retrieval_ndcg_at_K`. Cases with no `relevant_document_ids` report
  `value=0.0`, `passed=None`, `details.not_applicable=True`. A case whose
  complete retrieved list contains no relevant document is flagged
  `RETRIEVAL_MISS`.
- `DeterministicRAGEvaluator` — a combined evaluator producing the Phase 0
  generation metrics plus the retrieval metrics in one
  `EvaluationResult` (no overall score)
- `examples/rag_v0/` — a 10-document synthetic commerce corpus, 11
  development cases, and the controlled-ablation pair
  `rag_keyword_v0` / `rag_term_frequency_v0`, which differ only in
  `retrieval.retriever_id`

The Phase 1A retrievers are deterministic development baselines.
They are not Dense Retrieval, BM25, Hybrid Retrieval, or production RAG
systems. All files under `examples/rag_v0/` are development fixtures, not
a benchmark, and no performance claim is made.

## Phase 1C — BM25 Sparse Retrieval Baseline

Phase 1C introduces an Okapi BM25 sparse retrieval baseline
(`bm25-okapi-v0`, retriever version `0.1`) alongside the Phase 1A
deterministic baselines:

- Frozen formula variant: `score(D, Q) = Σ IDF(q) * f(q,D)(k1+1) /
  (f(q,D) + k1(1-b+b|D|/avgdl))` over distinct normalized query tokens,
  with the positive IDF variant
  `IDF(q) = ln(1 + (N - n(q) + 0.5) / (n(q) + 0.5))`. Corpus statistics
  are recomputed per `retrieve` call; there is no index or cache.
- Defaults `k1 = 1.5` and `b = 0.75` are defined exactly once on
  `BM25Retriever` (`DEFAULT_K1` / `DEFAULT_B`); validation requires
  `k1 > 0` and `0 <= b <= 1` and rejects bools, NaN, infinities, and
  non-numeric values.
- Retriever algorithm parameters declared in `retrieval.parameters`
  (`k1`, `b`) now flow through `build_retriever` into the runtime
  retriever — they are applied, not just fingerprinted. Pipeline-level
  keys (`top_k`, `corpus_id`, `corpus_version`) remain owned by
  `DeterministicRAGSystem.from_manifest`; undeclared algorithm keys
  (for example a `k11` typo) are rejected rather than silently ignored.
- Tokenization reuses the Phase 1A development tokenizer (lowercase
  ASCII `[a-z0-9]+` runs). It is not a multilingual tokenizer and no
  Chinese sparse retrieval support is claimed.
- `examples/rag_v0/experiments/rag_bm25_v0.json` is a development
  fixture on the same synthetic corpus and case list; compared with
  `rag_term_frequency_v0` only the retrieval strategy configuration
  changes. On the `rag-coupon-001` development fixture the two produce
  different rankings — an observed fixture difference, not a benchmark
  conclusion.

No formal benchmark performance claim is made in Phase 1C.

## Phase 1E — Dense Retrieval Baseline

Phase 1E adds the first dense retrieval baseline
(`dense-minilm-l6-v2-cosine-v0`, retriever version `0.1`) alongside the
Phase 1A/1C sparse baselines:

- Model: `sentence-transformers/all-MiniLM-L6-v2`, pinned to the
  immutable revision `8b3219a92973c328a8e22fadcfa821b5dc75636a`. The
  revision is passed to `SentenceTransformer` at load time; moving
  references such as `main` are rejected by configuration validation.
- Backend: `sentence-transformers` `6.1.0`, an exact pinned runtime
  dependency in `pyproject.toml`. The configured
  `embedding_backend_version` is checked against the installed package
  version at model-load time.
- Embedding dimension `384`, `L2`-normalized vectors
  (`normalize_embeddings=true`), cosine similarity ranking. Every
  embedding and every returned score is validated finite and
  dimension-checked; NaN, infinities, zero-norm vectors, and dimension
  mismatches raise `ContractValidationError` rather than entering a
  `RetrievalResult` or `RunTrace`.
- `EmbeddingEncoder` is the minimal internal boundary isolating model
  loading, revision pinning, normalization, and output validation.
  `SentenceTransformerEncoder` loads the model lazily on the first
  `encode` call — never at module import — so unit tests inject a fake
  encoder and stay offline.
- `model_name`, `model_revision`, `embedding_backend`,
  `embedding_backend_version`, `similarity`, `normalize_embeddings`,
  and `expected_dimension` are declared retriever algorithm parameters:
  they enter the experiment fingerprint and are enforced at runtime.
  Undeclared keys are rejected as before.
- Unlike the sparse `score > 0` policy, dense retrieval returns the
  `top_k` documents for any non-empty corpus, including zero or
  negative cosine scores. Corpus embeddings live in memory only — no
  vector database, index, hybrid fusion, or reranking.
- `examples/rag_v0/experiments/rag_dense_minilm_v0.json` is a
  development fixture; compared with `rag_bm25_v0` only the retrieval
  strategy configuration differs. On the `rag-refund-miss-001`
  development fixture BM25 retrieves nothing (no lexical overlap)
  while MiniLM ranks the refund policy first — an observed fixture
  difference, not a benchmark conclusion.

Tests: the default `python -m pytest` run is the fast, offline suite.
The real pinned-model smoke is marked `integration` and deselected by
default; run it explicitly with `python -m pytest -m integration`. The
first run downloads the pinned model into the HuggingFace cache; later
runs reuse that cache.

This is a development retrieval baseline. No formal CommerceBench
performance claim is made. `all-MiniLM-L6-v2` is English-oriented in
this phase; this phase does not establish multilingual/Chinese
retrieval quality.

## Phase 1G — Hybrid RRF Retrieval Baseline

Phase 1G adds the first hybrid retrieval baseline
(`hybrid-bm25-minilm-rrf-v0`, retriever version `0.1`) on top of the
existing components:

- Sparse component: `bm25-okapi-v0` (Okapi BM25, `k1`, `b`).
- Dense component: `dense-minilm-l6-v2-cosine-v0`
  (`sentence-transformers/all-MiniLM-L6-v2` at the pinned immutable
  revision, cosine similarity over normalized embeddings).
- Fusion: Reciprocal Rank Fusion,
  `RRF(d) = Σ_i 1 / (rrf_k + rank_i(d))` over each component's 1-based
  ranks; a document absent from a component receives no contribution
  from it. RRF is used because BM25 and cosine raw score scales are not
  directly comparable — only ranks are fused, never raw scores. Final
  ordering is RRF score descending, then `document_id` ascending;
  `RetrievalResult.scores` are the RRF fusion scores.
- `rrf_k` defaults to 60 and `component_top_k` to 10, each defined at
  exactly one point (`DEFAULT_RRF_K` / `DEFAULT_COMPONENT_TOP_K`),
  validated as positive integers, and declared in
  `retrieval.parameters` so they enter the experiment fingerprint and
  reach runtime.
- `component_top_k` (each component's candidate depth into fusion) and
  the pipeline `top_k` (final result depth) are separate concepts;
  `component_top_k >= top_k` is enforced at the retrieval boundary.
- The `sparse`/`dense` nested component configurations are required
  and must carry every algorithm parameter the component declares
  (`k1`/`b`; backend, version, model, revision, similarity,
  normalization, dimension) — no component default is silently
  applied, so the full component identity enters the fingerprint.
  Only the two pinned component ids are accepted; a nested hybrid is
  rejected.
- Each run emits exactly one `RetrievalEvent` — the fused ranking.
  The per-component rankings are recorded under
  `runtime_metadata["hybrid"]["components"]` for diagnosis (for
  example, which component contributed a hit) and are never written
  as extra retrieval events. Component raw scores are diagnostic only
  and must not be compared across retrievers.
- `examples/rag_v0/experiments/rag_hybrid_rrf_v0.json` is a
  development fixture on the same corpus and case list; compared with
  `rag_bm25_v0` and `rag_dense_minilm_v0` only the retrieval strategy
  configuration differs. On the `rag-refund-miss-001` development
  fixture BM25 retrieves nothing (no lexical overlap) while the
  fused ranking inherits the dense component's refund-policy hit —
  an observed fixture difference, not a benchmark conclusion.
  Combining retrieval systems is not guaranteed to improve ranking;
  regressions are kept, not hidden.

No reranker, no weighted raw-score fusion, and no formal benchmark
performance claim is made in Phase 1G.

## Phase 2A — Retrieval Role Contract

Phase 2A adds `RetrievalEvent.role` to distinguish multi-stage retrieval
inside one `RunTrace`:

- `candidate` — a diagnostic intermediate ranking (for example, an
  initial retrieval ranking that a later stage may reorder). Candidate
  events are never counted as retrieved documents.
- `final` — the evaluation-visible ranking: what downstream
  context/generation consumes and what retrieval metrics are computed
  from.
- no `role` (legacy) — Phase 1A–1G traces carry no role field; they are
  interpreted as `final`, so historical traces remain valid without
  migration. Unknown roles are rejected when loading.

`RunTrace.final_retrieval_events` selects the evaluation-visible events
(`final` plus legacy) and `RunTrace.final_ranked_document_ids`
concatenates them in trace order with keep-first de-duplication —
`ranked_document_ids(trace)` in the retrieval evaluator resolves through
this single helper, so Recall@K, Precision@K, MRR, NDCG@K and
`RETRIEVAL_MISS` all observe the same final ranking. A trace with only
candidate events has an empty final ranking and evaluates as a retrieval
miss, never silently promoting a candidate to final.

This change prepares the trace contract for reranking.
No reranker is implemented in Phase 2A.

## Phase 2B — Cross-Encoder Reranker Baseline

Phase 2B adds the first learned reranker baseline on top of the Phase 2A
candidate/final contract:

- Candidate retriever: Hybrid BM25 + Dense RRF
  (`hybrid-bm25-minilm-rrf-v0`)
- Candidate depth: 10 (`reranker.parameters.candidate_top_k`)
- Final depth: 3 (`retrieval.parameters.top_k`, the evaluation-visible
  and generation-visible depth)
- Reranker: `cross-encoder/ms-marco-MiniLM-L6-v2`
- Pinned revision: `ce0834f22110de6d9222af7a7a03628121708969`
  (immutable 40-hex; `main` and other moving references are rejected)
- Max length: 512 (explicit, fingerprinted, and passed to the runtime)
- Score activation: Identity / raw relevance score (higher is better;
  no implicit sigmoid). Final `RetrievalEvent.scores` are cross-encoder
  logits (`score_semantics = cross_encoder_logit`); candidate scores
  remain RRF scores and the two scales are never compared.
- Controlled control: `identity-reranker-v0` preserves the candidate
  order and truncates to the final depth, so
  `Hybrid -> Identity` vs `Hybrid -> CrossEncoder` isolates the reranker
  treatment (same query, corpus, retrieval config, candidate/final depth,
  system, generation fixture, evaluator, and seed).

A reranked run emits two retrieval events in trace order: `candidate`
(the Hybrid ranking with RRF scores, diagnostic only) and `final` (the
reranked ranking with reranker scores). The candidate event is excluded
from retrieval metrics; the final event is what generation consumes and
what Recall@K, Precision@K, MRR, NDCG@K, and `RETRIEVAL_MISS` observe.
Without a reranker the legacy single-event behavior is unchanged. The
reranker can only reorder or filter the candidate set — out-of-candidate
documents, duplicate candidate IDs, and non-finite scores are rejected.

`cross-encoder/ms-marco-MiniLM-L6-v2` is an English MS MARCO ranking
baseline. This phase does not establish Chinese/multilingual reranking
quality. All files under `examples/rag_v0/` remain development fixtures,
not a benchmark: observed development-case differences are reported, and
no formal performance claim (`improves by X%`, `best reranker`,
`production-ready`) is made.

## Phase 2D — Reranker Diagnostic Trace

Phase 2D closes the one diagnostic gap left by Phase 2B/2C: the candidate
event records the Hybrid candidate ranking and the final event records
the reranked top-k, but the full candidate-level reranker ordering was
not persisted. Now every reranked run preserves the complete reranker
ranking in `runtime_metadata`:

- `RerankResult.diagnostics` (`RerankDiagnostics`) carries the full
  reranker ranking — every scored candidate's ID and score ordered by
  score descending with `document_id` ascending tie-break, so position
  `i + 1` is the reranker rank. It is returned by the same `rerank()`
  call (no `last_*` mutable state); the final top-k is exactly its
  prefix, enforced by validation.
- `IdentityReranker` reports the full candidate ranking itself as its
  diagnostics, so `Hybrid -> Identity` vs `Hybrid -> CrossEncoder`
  share one diagnostic contract.
- `CrossEncoderReranker` scores all candidates with a single `predict()`
  pass; the full ranking, diagnostics, and final slice all derive from
  those scores — no second inference, negative logits retained,
  non-finite scores rejected even outside the final top-k.
- The pipeline persists `result.diagnostics` as
  `runtime_metadata["reranker"]["full_ranking"]` (`document_ids` +
  `scores` only — no document text, tensors, or paths) after
  fail-closed checks (full coverage of the candidate set, no unknown
  documents, final-equals-prefix).

Reranker diagnostic traces preserve full candidate-level reranker
scores and ordering in `runtime_metadata`. The final `RetrievalEvent`
remains limited to the evaluation-visible final top-k. This allows
offline reconstruction of rank movement without changing retrieval
metric semantics: candidate event + `reranker.full_ranking` + final
event jointly answer which candidate moved where, which dropped
candidate came closest to the final cutoff, and what every candidate's
reranker score was — without re-running the model. Retrieval metrics,
generation, experiment fingerprints, and the `RetrievalEvent`/`RunTrace`
schemas are unchanged.

## Phase 2E — Reranker Failure Analysis / Reporting

Phase 2E adds a pure reporting layer (`commercebench/reporting/`) over
the persisted Phase 2D evidence. Analysis is derived from persisted
benchmark evidence — `CaseSpec` + `RunTrace` (+ `EvaluationResult` for
cross-check only). No reranker/model call is required, and no core
schema (`RunTrace`, `RetrievalEvent`, `ExperimentManifest`,
`EvaluationResult`) is modified.

- `analyze_reranker_case(case, trace, evaluation=None)` builds a
  `RerankerCaseAnalysis`: per-document rank movement (`rank_delta =
  candidate_rank - reranker_rank`, so positive means moved up),
  `top_k_entered` / `top_k_exited`, `candidate_top1` / `final_top1` /
  `top1_changed`, the generation-driving document (final top-1),
  cutoff diagnostics (`cutoff_score`, `next_below_cutoff_score`,
  `cutoff_margin` — a raw-logit score difference, not a calibrated
  confidence measure), depth-matched candidate/final MRR and Recall@K
  with deltas, and deterministic relevant-document observations
  (`relevant_rank_improved`, `relevant_rank_regressed`,
  entered/exited-final flags).
- Candidate MRR/Recall baselines are computed over the candidate window
  `candidate[:final_top_k]` — the same depth as the final ranking — so
  deltas compare like with like and an Identity run always reports zero
  deltas. Full-ranking movement (including ranks beyond the final
  cutoff) is tracked separately via the full reranker relevant rank.
- RRF scores and CrossEncoder logits are not directly comparable: ranks
  may be compared across stages, but score magnitudes are never
  subtracted or fused. `EvaluationResult` metrics are cross-checked
  (derived final MRR/Recall must match) but never extended.
- `compare_reranker_runs(baseline, treatment, ...)` compares Identity vs
  CrossEncoder runs for one case after verifying the controlled
  conditions (same case/query/corpus/retriever/candidates/final depth);
  generation change is decided from the traces' actual outputs.
  `summarize_reranker_comparison(...)` aggregates deterministic
  counts/rates only — no overall score, no significance testing.
- On the 11 development cases the deterministic observation is: 0
  relevant-rank improvements, 0 regressions, 1 top-1 change, 1
  generation change, MRR unchanged on all 10 relevance-applicable
  cases. `rag-return-exchange-001` shows metric equivalence without
  behavior equivalence: MRR stays 1.0 while the top-1 relevant document
  (`policy-exchange-001` → `policy-return-001`) and the generated
  answer change. Observations are reported, never claimed as formal
  benchmark conclusions.

## Development

Requires Python 3.10+. Installing the package pulls the pinned
`sentence-transformers==6.1.0` runtime dependency (including torch);
the embedding model itself is downloaded on first use, not at install
time.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
python -m pytest                    # fast offline suite
python -m pytest -m integration     # real pinned-model smoke
```

## Project layout

```text
commercebench/       core package
  contracts/         CaseSpec, ExperimentManifest, RunTrace, EvaluationResult
  rag/               Document, Corpus, retrievers, deterministic RAG pipeline
  reranking/         RerankCandidate, RerankResult, Identity/CrossEncoder rerankers
  benchmark/         benchmark definitions, scenarios, tasks
  systems/           systems under test (pipelines, agents, configs)
  evaluation/        metrics, judges, scoring
  runner/            experiment runner (case -> system -> trace)
  reporting/         experiment reporting and regression analysis
tests/               test suite
configs/             benchmark and experiment configurations
experiments/         experiment manifests
examples/phase0/     Phase 0 development fixtures (not a benchmark)
examples/rag_v0/     Phase 1A RAG development fixtures (not a benchmark)
```
