"""Phase 3B DialogueEvaluator tests, including offline replay."""

from __future__ import annotations

import pytest

from commercebench.contracts import SCHEMA_VERSION
from commercebench.contracts.dialogue import (
    DialogueCaseSpec,
    DialogueRunTrace,
    DialogueStepTrace,
    DialogueTurnSpec,
    DialogueTurnTrace,
)
from commercebench.contracts.trace import ToolEvent
from commercebench.evaluation.dialogue import (
    DialogueEvaluator,
    FAILURE_INVALID_TOOL_CALL,
    FAILURE_REQUIRED_TOOL_MISSING,
    FAILURE_TASK_INCOMPLETE,
)
from commercebench.evaluation.deterministic import (
    FAILURE_FORBIDDEN_CLAIM,
    FAILURE_REQUIRED_FACT,
)


def _spec(turn_id, **kw):
    payload = {"turn_id": turn_id, "user_input": f"u-{turn_id}"}
    payload.update(kw)
    return DialogueTurnSpec(**payload)


def _case(turns, initial_state, expected_final_state, **kw):
    payload = {
        "schema_version": SCHEMA_VERSION,
        "case_id": "dlg-001",
        "benchmark_version": "commercebench-dev-0.1",
        "task_family": "commerce_dialogue",
        "intent": "return",
        "difficulty": "medium",
        "answerability": "answerable",
        "turns": turns,
        "initial_state": initial_state,
        "expected_final_state": expected_final_state,
    }
    payload.update(kw)
    return DialogueCaseSpec(**payload)


def _turn_trace(turn_id, index, state_before, state_after,
                output_text="", tool_events=(), steps=None):
    if steps is None:
        steps = (
            DialogueStepTrace(
                step_index=0,
                output_text=output_text,
                tool_events=tuple(tool_events),
            ),
        )
    return DialogueTurnTrace(
        turn_id=turn_id,
        turn_index=index,
        user_input=f"u-{turn_id}",
        state_before=state_before,
        state_after=state_after,
        output_text=output_text,
        steps=tuple(steps),
        tool_events=tuple(tool_events),
    )


def _trace(turns, final_state, termination="user_done", output_text=""):
    return DialogueRunTrace(
        schema_version=SCHEMA_VERSION,
        run_id="r1",
        experiment_id="e1",
        case_id="dlg-001",
        system_id="s1",
        system_version="v1",
        config_fingerprint="fp",
        turns=tuple(turns),
        initial_state={},
        final_state=final_state,
        termination_reason=termination,
        output_text=output_text,
        started_at="2026-10-06T00:00:00+00:00",
        finished_at="2026-10-06T00:00:01+00:00",
        latency_ms=1.0,
    )


@pytest.fixture
def evaluator():
    return DialogueEvaluator()


class TestFinalStateMatch:
    def test_subset_match_passes(self, evaluator):
        case = _case(
            [_spec("t1")],
            {},
            {"returns": {"A": {"status": "requested"}}},
        )
        trace = _trace(
            [_turn_trace("t1", 1, {}, {"returns": {"A": {"status": "requested", "by": "sys"}}})],
            {"returns": {"A": {"status": "requested", "by": "sys"}}},
        )
        result = evaluator.evaluate(case, trace)
        assert result.metrics["final_state_match"].passed is True
        assert FAILURE_TASK_INCOMPLETE not in result.failure_categories

    def test_mismatch_fails_with_details(self, evaluator):
        case = _case([_spec("t1")], {}, {"returns": {"A": {"status": "x"}}})
        trace = _trace(
            [_turn_trace("t1", 1, {}, {})], {}
        )
        result = evaluator.evaluate(case, trace)
        metric = result.metrics["final_state_match"]
        assert metric.passed is False
        assert result.passed is False
        assert FAILURE_TASK_INCOMPLETE in result.failure_categories
        assert metric.details["mismatches"][0]["path"] == "returns"


class TestStateDeltaCoverage:
    def test_delta_must_be_newly_satisfied(self, evaluator):
        # Delta already true in state_before -> not credited.
        case = _case(
            [_spec("t1", expected_state_delta={"a": {"b": 1}})],
            {},
            {},
        )
        trace = _trace(
            [_turn_trace("t1", 1, {"a": {"b": 1}}, {"a": {"b": 1}})],
            {"a": {"b": 1}},
        )
        result = evaluator.evaluate(case, trace)
        assert result.metrics["state_delta_coverage"].value == 0.0
        assert result.metrics["state_delta_coverage"].passed is False

    def test_delta_satisfied_by_turn(self, evaluator):
        case = _case(
            [_spec("t1", expected_state_delta={"a": {"b": 1}})],
            {},
            {},
        )
        trace = _trace(
            [_turn_trace("t1", 1, {"a": {}}, {"a": {"b": 1}})],
            {"a": {"b": 1}},
        )
        result = evaluator.evaluate(case, trace)
        assert result.metrics["state_delta_coverage"].value == 1.0

    def test_no_obligations_not_applicable(self, evaluator):
        # A non-degenerate case (final-state obligation) can still have
        # no delta obligations at all.
        case = _case([_spec("t1")], {}, {"done": True})
        trace = _trace([_turn_trace("t1", 1, {}, {})], {"done": True})
        result = evaluator.evaluate(case, trace)
        metric = result.metrics["state_delta_coverage"]
        assert metric.passed is None
        assert metric.details["not_applicable"] is True


class TestToolCoverage:
    def _case_with_tool(self, args=None):
        call = {"tool_name": "lookup_order"}
        if args is not None:
            call["arguments"] = args
        return _case(
            [_spec("t1", expected_tool_calls=[call])],
            {},
            {},
        )

    def test_successful_call_matches(self, evaluator):
        case = self._case_with_tool({"order_id": "A1"})
        trace = _trace(
            [_turn_trace(
                "t1", 1, {}, {},
                tool_events=(
                    ToolEvent(tool_name="lookup_order",
                              arguments={"order_id": "A1"},
                              status="ok"),
                ),
            )],
            {},
        )
        result = evaluator.evaluate(case, trace)
        assert result.metrics["required_tool_coverage"].passed is True

    def test_wrong_args_not_matched(self, evaluator):
        case = self._case_with_tool({"order_id": "A1"})
        trace = _trace(
            [_turn_trace(
                "t1", 1, {}, {},
                tool_events=(
                    ToolEvent(tool_name="lookup_order",
                              arguments={"order_id": "WRONG"},
                              status="ok"),
                ),
            )],
            {},
        )
        result = evaluator.evaluate(case, trace)
        assert result.metrics["required_tool_coverage"].passed is False
        assert FAILURE_REQUIRED_TOOL_MISSING in result.failure_categories

    def test_failed_call_not_matched(self, evaluator):
        case = self._case_with_tool()
        trace = _trace(
            [_turn_trace(
                "t1", 1, {}, {},
                tool_events=(
                    ToolEvent(tool_name="lookup_order",
                              status="error", error="ORDER_NOT_FOUND"),
                ),
            )],
            {},
        )
        result = evaluator.evaluate(case, trace)
        assert result.metrics["required_tool_coverage"].passed is False

    def test_no_obligations_not_applicable(self, evaluator):
        case = _case([_spec("t1")], {}, {"done": True})
        trace = _trace([_turn_trace("t1", 1, {}, {})], {"done": True})
        result = evaluator.evaluate(case, trace)
        assert result.metrics["required_tool_coverage"].passed is None


class TestInvalidToolCalls:
    def test_error_events_counted(self, evaluator):
        case = _case([_spec("t1")], {}, {"done": True})
        trace = _trace(
            [_turn_trace(
                "t1", 1, {}, {},
                tool_events=(
                    ToolEvent(tool_name="x", status="error", error="E1"),
                    ToolEvent(tool_name="y", status="error", error="E2"),
                    ToolEvent(tool_name="z", status="ok"),
                ),
            )],
            {},
        )
        result = evaluator.evaluate(case, trace)
        metric = result.metrics["invalid_tool_call_count"]
        assert metric.value == 2
        assert metric.passed is False
        assert FAILURE_INVALID_TOOL_CALL in result.failure_categories
        assert len(metric.details["errors"]) == 2


class TestClaimsAndFacts:
    def test_forbidden_claim_in_any_step_counts(self, evaluator):
        case = _case(
            [_spec("t1")], {}, {"done": True},
            forbidden_claims=("free money",),
        )
        trace = _trace(
            [_turn_trace(
                "t1", 1, {}, {},
                steps=(
                    DialogueStepTrace(step_index=0, output_text=""),
                    DialogueStepTrace(
                        step_index=1, output_text="you get free money!"
                    ),
                ),
            )],
            {},
        )
        result = evaluator.evaluate(case, trace)
        assert result.metrics["forbidden_claim_violation"].passed is False
        assert FAILURE_FORBIDDEN_CLAIM in result.failure_categories

    def test_turn_scoped_claim(self, evaluator):
        case = _case(
            [
                _spec("t1"),
                _spec("t2", forbidden_claims=("oops",)),
            ],
            {}, {"done": True},
        )
        # claim appears in t1's output — turn-scoped to t2, so no hit
        trace = _trace(
            [
                _turn_trace("t1", 1, {}, {}, output_text="oops"),
                _turn_trace("t2", 2, {}, {}, output_text="fine"),
            ],
            {},
        )
        result = evaluator.evaluate(case, trace)
        assert result.metrics["forbidden_claim_violation"].passed is True

    def test_fact_coverage_at_or_after_declaring_turn(self, evaluator):
        case = _case(
            [
                _spec("t1"),
                _spec("t2", required_facts=("order-id",)),
            ],
            {}, {"done": True},
        )
        # fact stated in t1 (before the t2 obligation) — not satisfied
        early = _trace(
            [
                _turn_trace("t1", 1, {}, {}, output_text="order-id"),
                _turn_trace("t2", 2, {}, {}, output_text="unrelated"),
            ],
            {},
        )
        result = evaluator.evaluate(case, early)
        assert result.metrics["required_fact_coverage"].value == 0.0
        assert FAILURE_REQUIRED_FACT in result.failure_categories

        late = _trace(
            [
                _turn_trace("t1", 1, {}, {}, output_text="hi"),
                _turn_trace("t2", 2, {}, {}, output_text="order-id"),
            ],
            {},
        )
        result = evaluator.evaluate(case, late)
        assert result.metrics["required_fact_coverage"].value == 1.0

    def test_case_fact_anywhere(self, evaluator):
        case = _case(
            [_spec("t1"), _spec("t2")],
            {}, {"done": True},
            required_facts=("the-fact",),
        )
        trace = _trace(
            [
                _turn_trace("t1", 1, {}, {}, output_text="the-fact"),
                _turn_trace("t2", 2, {}, {}, output_text="x"),
            ],
            {},
        )
        result = evaluator.evaluate(case, trace)
        assert result.metrics["required_fact_coverage"].value == 1.0


class TestMultiLabelAndNoScore:
    def test_multi_label(self, evaluator):
        case = _case(
            [_spec("t1", expected_tool_calls=[{"tool_name": "x"}])],
            {},
            {"goal": 1},
            forbidden_claims=("bad-claim",),
        )
        trace = _trace(
            [_turn_trace(
                "t1", 1, {}, {},
                output_text="bad-claim",
                tool_events=(
                    ToolEvent(tool_name="y", status="error", error="E"),
                ),
            )],
            {},
        )
        result = evaluator.evaluate(case, trace)
        assert result.passed is False
        categories = set(result.failure_categories)
        assert categories == {
            FAILURE_TASK_INCOMPLETE,
            FAILURE_INVALID_TOOL_CALL,
            FAILURE_REQUIRED_TOOL_MISSING,
            FAILURE_FORBIDDEN_CLAIM,
        }

    def test_no_overall_score_metric(self, evaluator):
        case = _case([_spec("t1")], {}, {"done": True})
        trace = _trace([_turn_trace("t1", 1, {}, {})], {"done": True})
        result = evaluator.evaluate(case, trace)
        for name in result.metrics:
            assert "score" not in name
            assert "overall" not in name


class TestOfflineReplay:
    def test_persisted_trace_evaluates_identically(self, evaluator):
        case = _case(
            [_spec("t1", expected_tool_calls=[
                {"tool_name": "lookup_order",
                 "arguments": {"order_id": "A1"}}])],
            {"orders": {"A1": {"status": "delivered"}}},
            {"returns": {"A1": {"status": "requested"}}},
        )
        trace = _trace(
            [_turn_trace(
                "t1", 1,
                {"orders": {"A1": {"status": "delivered"}}},
                {"orders": {"A1": {"status": "delivered"}},
                 "returns": {"A1": {"status": "requested"}}},
                output_text="done",
                tool_events=(
                    ToolEvent(tool_name="lookup_order",
                              arguments={"order_id": "A1"},
                              result={"status": "delivered"},
                              status="ok"),
                    ToolEvent(tool_name="request_return",
                              arguments={"order_id": "A1"},
                              result={"return_status": "requested"},
                              status="ok"),
                ),
            )],
            {"orders": {"A1": {"status": "delivered"}},
             "returns": {"A1": {"status": "requested"}}},
            termination="completed",
            output_text="done",
        )
        live = evaluator.evaluate(case, trace)

        case2 = DialogueCaseSpec.from_json(case.to_json())
        trace2 = DialogueRunTrace.from_json(trace.to_json())
        replayed = evaluator.evaluate(case2, trace2)

        assert replayed.metrics.keys() == live.metrics.keys()
        for name in live.metrics:
            assert replayed.metrics[name].value == live.metrics[name].value
            assert (
                replayed.metrics[name].passed == live.metrics[name].passed
            )
        assert replayed.passed == live.passed
        assert replayed.failure_categories == live.failure_categories
