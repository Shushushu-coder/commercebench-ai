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

## Development

Requires Python 3.10+.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
python -m pytest
```

## Project layout

```text
commercebench/       core package
  contracts/         CaseSpec, ExperimentManifest, RunTrace, EvaluationResult
  rag/               Document, Corpus, retrievers, deterministic RAG pipeline
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
