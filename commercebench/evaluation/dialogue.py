"""Deterministic dialogue evaluator (Phase 3B).

Consumes ``DialogueCaseSpec`` + ``DialogueRunTrace`` — both fully
persistable — and emits the existing ``EvaluationResult``. No system
or harness re-invocation: the trace is the complete evidence record.

Metric set (no overall score; ``passed`` is the evaluator gate):

- ``final_state_match``: bool — ``expected_final_state`` subset-matches
  ``trace.final_state`` (recursive dict subset, leaves exact).
- ``state_delta_coverage``: float — fraction of declared per-turn
  ``expected_state_delta`` leaf obligations realized by that turn
  (leaf must hold in ``state_after`` and NOT hold in
  ``state_before``: the turn must have produced the change, not just
  satisfy it incidentally). No obligations -> value 0.0, passed None.
- ``required_tool_coverage``: float — ordered matching of declared
  ``expected_tool_calls`` against the turn's executed
  ``ToolEvent``s; a call counts only when it executed with
  ``status="ok"`` (and exact arguments when declared).
- ``invalid_tool_call_count``: int — executed events with
  ``status="error"``; passes only at 0.
- ``forbidden_claim_violation``: bool — case-level claims searched in
  all assistant-visible texts; turn-level claims searched in that
  turn's assistant texts only.
- ``required_fact_coverage``: float — normalized-substring coverage.
  Case-level facts may appear in any assistant output; turn-level
  facts must appear at or after the declaring turn.
- ``turn_count``: int — informational, never gated.

Failure categories (stable, multi-label):
    TASK_INCOMPLETE         final_state_match is False
    INVALID_TOOL_CALL       invalid_tool_call_count > 0
    REQUIRED_TOOL_MISSING   required_tool_coverage < 1.0 (when declared)
    FORBIDDEN_CLAIM_PRESENT (reused from deterministic evaluator)
    REQUIRED_FACT_MISSING   (reused from deterministic evaluator)
"""

from __future__ import annotations

import uuid
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from commercebench.contracts.common import SCHEMA_VERSION
from commercebench.contracts.dialogue import (
    DialogueCaseSpec,
    DialogueRunTrace,
    DialogueTurnSpec,
    DialogueTurnTrace,
)
from commercebench.contracts.evaluation import EvaluationResult, MetricResult
from commercebench.contracts.trace import ToolEvent
from commercebench.dialogue.state import (
    delta_obligations_satisfied,
    json_leaf_equal,
    json_value_equal,
    split_delta_obligations,
    state_leaf_at,
    state_subset_matches,
    state_subset_mismatches,
)

from .deterministic import (
    FAILURE_FORBIDDEN_CLAIM,
    FAILURE_REQUIRED_FACT,
    _normalize,
)

DIALOGUE_EVALUATOR_ID = "dialogue-deterministic-v0"
DIALOGUE_EVALUATOR_VERSION = "0.1"

FAILURE_TASK_INCOMPLETE = "TASK_INCOMPLETE"
FAILURE_INVALID_TOOL_CALL = "INVALID_TOOL_CALL"
FAILURE_REQUIRED_TOOL_MISSING = "REQUIRED_TOOL_MISSING"


class DialogueEvaluator:
    """Computes the Phase 3B dialogue metric set for (case, trace)."""

    evaluator_id = DIALOGUE_EVALUATOR_ID
    evaluator_version = DIALOGUE_EVALUATOR_VERSION

    def evaluate(
        self, case: DialogueCaseSpec, trace: DialogueRunTrace
    ) -> EvaluationResult:
        turn_traces: Dict[str, DialogueTurnTrace] = {
            turn.turn_id: turn for turn in trace.turns
        }
        assistant_texts = _assistant_texts(trace.turns)

        metrics: Dict[str, MetricResult] = {}
        failure_categories: List[str] = []

        # --- final_state_match -----------------------------------------
        final_ok = state_subset_matches(
            case.expected_final_state, trace.final_state
        )
        metrics["final_state_match"] = MetricResult(
            name="final_state_match",
            value=final_ok,
            passed=final_ok,
            details={
                "expected_final_state": dict(case.expected_final_state),
                "mismatches": state_subset_mismatches(
                    case.expected_final_state, trace.final_state
                ),
                "termination_reason": trace.termination_reason,
            },
        )
        if not final_ok:
            failure_categories.append(FAILURE_TASK_INCOMPLETE)

        # --- state_delta_coverage ---------------------------------------
        delta_metric, delta_ok = self._state_delta_metric(
            case, turn_traces
        )
        metrics["state_delta_coverage"] = delta_metric

        # --- required_tool_coverage -------------------------------------
        tool_metric, tool_ok = self._tool_coverage_metric(
            case, turn_traces
        )
        metrics["required_tool_coverage"] = tool_metric
        if tool_ok is False:
            failure_categories.append(FAILURE_REQUIRED_TOOL_MISSING)

        # --- invalid_tool_call_count ------------------------------------
        invalid = sum(
            1
            for turn in trace.turns
            for event in turn.tool_events
            if event.status == "error"
        )
        metrics["invalid_tool_call_count"] = MetricResult(
            name="invalid_tool_call_count",
            value=invalid,
            passed=invalid == 0,
            details={
                "errors": [
                    {
                        "turn_id": turn.turn_id,
                        "tool_name": event.tool_name,
                        "error": event.error,
                    }
                    for turn in trace.turns
                    for event in turn.tool_events
                    if event.status == "error"
                ]
            },
        )
        if invalid > 0:
            failure_categories.append(FAILURE_INVALID_TOOL_CALL)

        # --- forbidden_claim_violation ----------------------------------
        violations = self._forbidden_violations(
            case, turn_traces, assistant_texts
        )
        metrics["forbidden_claim_violation"] = MetricResult(
            name="forbidden_claim_violation",
            value=bool(violations),
            passed=not violations,
            details={"violations": violations},
        )
        if violations:
            failure_categories.append(FAILURE_FORBIDDEN_CLAIM)

        # --- required_fact_coverage --------------------------------------
        fact_metric, fact_ok = self._fact_coverage_metric(
            case, turn_traces, assistant_texts
        )
        metrics["required_fact_coverage"] = fact_metric
        if fact_ok is False:
            failure_categories.append(FAILURE_REQUIRED_FACT)

        # --- turn_count --------------------------------------------------
        metrics["turn_count"] = MetricResult(
            name="turn_count",
            value=len(trace.turns),
            passed=None,
            details={"termination_reason": trace.termination_reason},
        )

        passed = all(
            metric.passed is not False for metric in metrics.values()
        )
        return EvaluationResult(
            schema_version=SCHEMA_VERSION,
            evaluation_id=uuid.uuid4().hex,
            run_id=trace.run_id,
            case_id=trace.case_id,
            evaluator_id=self.evaluator_id,
            evaluator_version=self.evaluator_version,
            metrics=metrics,
            passed=passed,
            failure_categories=tuple(failure_categories),
            evaluated_at=datetime.now(timezone.utc).isoformat(),
            metadata={},
        )

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _state_delta_metric(
        case: DialogueCaseSpec,
        turn_traces: Dict[str, DialogueTurnTrace],
    ) -> Tuple[MetricResult, Optional[bool]]:
        obligations: List[Dict[str, Any]] = []
        satisfied = 0
        for spec in case.turns:
            trace = turn_traces.get(spec.turn_id)
            # Shared with the harness completion predicate: the turn is
            # satisfied only when every leaf is newly caused.
            turn_satisfied = trace is not None and (
                delta_obligations_satisfied(
                    spec.expected_state_delta,
                    trace.state_before,
                    trace.state_after,
                )
            )
            for path, leaf in split_delta_obligations(
                spec.expected_state_delta
            ):
                entry = {
                    "turn_id": spec.turn_id,
                    "path": path,
                    "expected": leaf,
                    "satisfied": False,
                }
                if trace is not None:
                    present_after, after = state_leaf_at(
                        trace.state_after, path
                    )
                    present_before, before = state_leaf_at(
                        trace.state_before, path
                    )
                    leaf_ok_after = present_after and json_leaf_equal(
                        after, leaf
                    )
                    leaf_ok_before = present_before and json_leaf_equal(
                        before, leaf
                    )
                    if leaf_ok_after and not leaf_ok_before:
                        satisfied += 1
                        entry["satisfied"] = True
                entry["turn_satisfied"] = turn_satisfied
                obligations.append(entry)
        if not obligations:
            return (
                MetricResult(
                    name="state_delta_coverage",
                    value=0.0,
                    passed=None,
                    details={"not_applicable": True, "obligations": []},
                ),
                None,
            )
        coverage = satisfied / len(obligations)
        return (
            MetricResult(
                name="state_delta_coverage",
                value=coverage,
                passed=coverage == 1.0,
                details={"obligations": obligations},
            ),
            coverage == 1.0,
        )

    @staticmethod
    def _tool_coverage_metric(
        case: DialogueCaseSpec,
        turn_traces: Dict[str, DialogueTurnTrace],
    ) -> Tuple[MetricResult, Optional[bool]]:
        obligations: List[Dict[str, Any]] = []
        matched_count = 0
        for spec in case.turns:
            trace = turn_traces.get(spec.turn_id)
            executed = list(trace.tool_events) if trace else []
            consumed: set = set()
            for expected in spec.expected_tool_calls:
                matched = False
                matched_event: Optional[ToolEvent] = None
                for position, event in enumerate(executed):
                    if position in consumed:
                        continue
                    if event.tool_name != expected.tool_name:
                        continue
                    if event.status != "ok":
                        continue
                    if expected.arguments is not None and not (
                        json_value_equal(event.arguments, expected.arguments)
                    ):
                        continue
                    matched = True
                    matched_event = event
                    consumed.add(position)
                    break
                if matched:
                    matched_count += 1
                obligations.append(
                    {
                        "turn_id": spec.turn_id,
                        "tool_name": expected.tool_name,
                        "arguments_required": expected.arguments is not None,
                        "matched": matched,
                        "matched_status": (
                            matched_event.status if matched_event else None
                        ),
                    }
                )
        if not obligations:
            return (
                MetricResult(
                    name="required_tool_coverage",
                    value=0.0,
                    passed=None,
                    details={"not_applicable": True, "obligations": []},
                ),
                None,
            )
        coverage = matched_count / len(obligations)
        return (
            MetricResult(
                name="required_tool_coverage",
                value=coverage,
                passed=coverage == 1.0,
                details={"obligations": obligations},
            ),
            coverage == 1.0,
        )

    @staticmethod
    def _forbidden_violations(
        case: DialogueCaseSpec,
        turn_traces: Dict[str, DialogueTurnTrace],
        assistant_texts: List[Tuple[str, str]],
    ) -> List[Dict[str, Any]]:
        violations: List[Dict[str, Any]] = []
        all_text = _normalize(" ".join(text for _, text in assistant_texts))
        for claim in case.forbidden_claims:
            if _normalize(claim) in all_text:
                violations.append(
                    {"scope": "case", "claim": claim}
                )
        for spec in case.turns:
            if not spec.forbidden_claims:
                continue
            trace = turn_traces.get(spec.turn_id)
            turn_text = _normalize(
                " ".join(step.output_text for step in trace.steps)
            ) if trace is not None else ""
            for claim in spec.forbidden_claims:
                if _normalize(claim) in turn_text:
                    violations.append(
                        {
                            "scope": "turn",
                            "turn_id": spec.turn_id,
                            "claim": claim,
                        }
                    )
        return violations

    @staticmethod
    def _fact_coverage_metric(
        case: DialogueCaseSpec,
        turn_traces: Dict[str, DialogueTurnTrace],
        assistant_texts: List[Tuple[str, str]],
    ) -> Tuple[MetricResult, Optional[bool]]:
        obligations: List[Dict[str, Any]] = []
        covered = 0
        all_text = _normalize(" ".join(text for _, text in assistant_texts))
        for fact in case.required_facts:
            hit = _normalize(fact) in all_text
            if hit:
                covered += 1
            obligations.append(
                {"scope": "case", "fact": fact, "satisfied": hit}
            )
        # Turn-level facts: must appear at or after the declaring turn.
        processed_order = list(turn_traces.keys())
        # turn_traces dict preserves trace order (insertion from run).
        for spec in case.turns:
            if not spec.required_facts:
                continue
            if spec.turn_id in processed_order:
                start = processed_order.index(spec.turn_id)
                tail_text = _normalize(
                    " ".join(
                        text
                        for turn_id, text in assistant_texts
                        if processed_order.index(turn_id) >= start
                    )
                )
            else:
                tail_text = ""
            for fact in spec.required_facts:
                hit = _normalize(fact) in tail_text
                if hit:
                    covered += 1
                obligations.append(
                    {
                        "scope": "turn",
                        "turn_id": spec.turn_id,
                        "fact": fact,
                        "satisfied": hit,
                    }
                )
        if not obligations:
            return (
                MetricResult(
                    name="required_fact_coverage",
                    value=0.0,
                    passed=None,
                    details={"not_applicable": True, "obligations": []},
                ),
                None,
            )
        coverage = covered / len(obligations)
        return (
            MetricResult(
                name="required_fact_coverage",
                value=coverage,
                passed=coverage == 1.0,
                details={"obligations": obligations},
            ),
            coverage == 1.0,
        )


def _assistant_texts(
    turns: Tuple[DialogueTurnTrace, ...],
) -> List[Tuple[str, str]]:
    """(turn_id, assistant step text) pairs in execution order."""
    return [
        (turn.turn_id, step.output_text)
        for turn in turns
        for step in turn.steps
    ]


__all__ = [
    "DIALOGUE_EVALUATOR_ID",
    "DIALOGUE_EVALUATOR_VERSION",
    "DialogueEvaluator",
    "FAILURE_INVALID_TOOL_CALL",
    "FAILURE_REQUIRED_TOOL_MISSING",
    "FAILURE_TASK_INCOMPLETE",
]
