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
