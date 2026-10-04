"""Deterministic evaluator: simple, LLM-free metrics for Phase 0.

Metrics (all computed on normalized text unless noted):

- ``exact_match``: literal ``output_text == expected_response``. Only
  emitted when the case defines ``expected_response``.
- ``required_fact_coverage``: fraction of ``required_facts`` that appear in
  the output. A case with no required facts scores 1.0 by convention.
- ``forbidden_claim_violation``: true if any ``forbidden_claims`` appear in
  the output.

Pass gate (``deterministic_pass``) is fixed:

    PASS iff exact_match is true (or absent)
         AND required_fact_coverage == 1.0
         AND forbidden_claim_violation is false

Normalization for containment checks is deliberately shallow: lowercase,
strip, and collapse internal whitespace. No embeddings or semantics.

Failure categories (machine-readable, stable):
    EXACT_MATCH_FAILED, REQUIRED_FACT_MISSING, FORBIDDEN_CLAIM_PRESENT
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Tuple

from commercebench.contracts.case import CaseSpec
from commercebench.contracts.common import SCHEMA_VERSION
from commercebench.contracts.evaluation import EvaluationResult, MetricResult
from commercebench.contracts.trace import RunTrace

EVALUATOR_ID = "deterministic-v0"
EVALUATOR_VERSION = "0.1"

FAILURE_EXACT_MATCH = "EXACT_MATCH_FAILED"
FAILURE_REQUIRED_FACT = "REQUIRED_FACT_MISSING"
FAILURE_FORBIDDEN_CLAIM = "FORBIDDEN_CLAIM_PRESENT"

_WHITESPACE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    return _WHITESPACE.sub(" ", text.strip().lower())


class DeterministicEvaluator:
    """Computes the Phase 0 metric set for one (case, trace) pair."""

    evaluator_id = EVALUATOR_ID
    evaluator_version = EVALUATOR_VERSION

    def evaluate(self, case: CaseSpec, trace: RunTrace) -> EvaluationResult:
        metrics: Dict[str, MetricResult] = {}
        failure_categories: List[str] = []

        if case.expected_response is not None:
            exact = trace.output_text == case.expected_response
            metrics["exact_match"] = MetricResult(
                name="exact_match",
                value=exact,
                passed=exact,
                details={
                    "expected": case.expected_response,
                    "actual": trace.output_text,
                },
            )
            if not exact:
                failure_categories.append(FAILURE_EXACT_MATCH)

        coverage, missing = _fact_coverage(case, trace.output_text)
        metrics["required_fact_coverage"] = MetricResult(
            name="required_fact_coverage",
            value=coverage,
            passed=coverage == 1.0,
            details={
                "total_required_facts": len(case.required_facts),
                "missing_facts": missing,
            },
        )
        if missing:
            failure_categories.append(FAILURE_REQUIRED_FACT)

        violated = _forbidden_violations(case, trace.output_text)
        metrics["forbidden_claim_violation"] = MetricResult(
            name="forbidden_claim_violation",
            value=bool(violated),
            passed=not violated,
            details={"violated_claims": violated},
        )
        if violated:
            failure_categories.append(FAILURE_FORBIDDEN_CLAIM)

        passed = not failure_categories

        return EvaluationResult(
            schema_version=SCHEMA_VERSION,
            evaluation_id=uuid.uuid4().hex,
            run_id=trace.run_id,
            case_id=case.case_id,
            evaluator_id=self.evaluator_id,
            evaluator_version=self.evaluator_version,
            metrics=metrics,
            passed=passed,
            failure_categories=tuple(failure_categories),
            evaluated_at=datetime.now(timezone.utc).isoformat(),
            metadata={},
        )


def _fact_coverage(case: CaseSpec, output_text: str) -> Tuple[float, List[str]]:
    """Return (coverage, missing_facts). Empty required_facts -> (1.0, [])."""
    if not case.required_facts:
        return 1.0, []
    normalized_output = _normalize(output_text)
    missing = [
        fact
        for fact in case.required_facts
        if _normalize(fact) not in normalized_output
    ]
    covered = len(case.required_facts) - len(missing)
    return covered / len(case.required_facts), missing


def _forbidden_violations(case: CaseSpec, output_text: str) -> List[str]:
    normalized_output = _normalize(output_text)
    return [
        claim
        for claim in case.forbidden_claims
        if _normalize(claim) in normalized_output
    ]


__all__ = [
    "DeterministicEvaluator",
    "EVALUATOR_ID",
    "EVALUATOR_VERSION",
    "FAILURE_EXACT_MATCH",
    "FAILURE_FORBIDDEN_CLAIM",
    "FAILURE_REQUIRED_FACT",
]
