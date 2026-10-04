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
  benchmark/         benchmark definitions, scenarios, tasks
  systems/           systems under test (pipelines, agents, configs)
  evaluation/        metrics, judges, scoring
  runner/            experiment runner (case -> system -> trace)
  reporting/         experiment reporting and regression analysis
tests/               test suite
configs/             benchmark and experiment configurations
experiments/         experiment manifests
examples/phase0/     Phase 0 development fixtures (not a benchmark)
```
