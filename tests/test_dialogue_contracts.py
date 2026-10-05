"""Phase 3B dialogue contract tests: roundtrips and invariants."""

from __future__ import annotations

import pytest

from commercebench.contracts import ContractValidationError, SCHEMA_VERSION
from commercebench.contracts.dialogue import (
    DialogueCaseSpec,
    DialogueHistoryEntry,
    DialogueRunTrace,
    DialogueStepTrace,
    DialogueToolDescriptor,
    DialogueTurnInput,
    DialogueTurnOutput,
    DialogueTurnSpec,
    DialogueTurnTrace,
    TERMINATION_REASONS,
)
from commercebench.contracts.trace import ToolEvent


def _turn_spec(**overrides):
    payload = {
        "turn_id": "t1",
        "user_input": "hello",
    }
    payload.update(overrides)
    return DialogueTurnSpec(**payload)


def _case(**overrides):
    payload = {
        "schema_version": SCHEMA_VERSION,
        "case_id": "dlg-001",
        "benchmark_version": "commercebench-dev-0.1",
        "task_family": "commerce_dialogue",
        "intent": "return",
        "difficulty": "medium",
        "answerability": "answerable",
        "turns": [_turn_spec()],
        "initial_state": {"orders": {}},
        "expected_final_state": {"returns": {"x": {"status": "requested"}}},
    }
    payload.update(overrides)
    return DialogueCaseSpec(**payload)


def _step(**overrides):
    payload = {"step_index": 0, "output_text": "hi"}
    payload.update(overrides)
    return DialogueStepTrace(**payload)


def _turn_trace(**overrides):
    payload = {
        "turn_id": "t1",
        "turn_index": 1,
        "user_input": "hello",
        "state_before": {"orders": {}},
        "state_after": {"orders": {}},
        "output_text": "hi",
        "steps": (_step(),),
    }
    payload.update(overrides)
    return DialogueTurnTrace(**payload)


def _run_trace(**overrides):
    payload = {
        "schema_version": SCHEMA_VERSION,
        "run_id": "r1",
        "experiment_id": "e1",
        "case_id": "dlg-001",
        "system_id": "s1",
        "system_version": "v1",
        "config_fingerprint": "fp",
        "turns": (_turn_trace(),),
        "initial_state": {"orders": {}},
        "final_state": {"orders": {}},
        "termination_reason": "completed",
        "output_text": "hi",
        "started_at": "2026-10-06T00:00:00+00:00",
        "finished_at": "2026-10-06T00:00:01+00:00",
        "latency_ms": 1000.0,
    }
    payload.update(overrides)
    return DialogueRunTrace(**payload)


class TestDialogueTurnSpec:
    def test_minimal_roundtrip(self):
        spec = _turn_spec()
        assert DialogueTurnSpec.from_dict(spec.to_dict()) == spec

    def test_activation_none_and_empty_are_default(self):
        assert _turn_spec(activation=None).activation is None
        assert _turn_spec(activation={}).activation == {}

    def test_activation_on_tool(self):
        spec = _turn_spec(activation={"on_tool": "lookup_order"})
        assert spec.activation == {"on_tool": "lookup_order"}

    def test_activation_on_tag(self):
        spec = _turn_spec(activation={"on_assistant_tag": "clarify"})
        assert spec.activation == {"on_assistant_tag": "clarify"}

    def test_activation_two_triggers_rejected(self):
        with pytest.raises(ContractValidationError):
            _turn_spec(
                activation={
                    "on_tool": "lookup_order",
                    "on_assistant_tag": "clarify",
                }
            )

    def test_activation_unknown_key_rejected(self):
        with pytest.raises(ContractValidationError):
            _turn_spec(activation={"on_regex": ".*"})

    def test_activation_empty_value_rejected(self):
        with pytest.raises(ContractValidationError):
            _turn_spec(activation={"on_tool": ""})

    def test_empty_turn_id_rejected(self):
        with pytest.raises(ContractValidationError):
            _turn_spec(turn_id="")

    def test_empty_user_input_rejected(self):
        with pytest.raises(ContractValidationError):
            _turn_spec(user_input="")

    def test_expected_tool_calls_parsed(self):
        spec = _turn_spec(
            expected_tool_calls=[
                {"tool_name": "lookup_order", "arguments": {"order_id": "A1"}},
                {"tool_name": "request_return"},
            ]
        )
        assert spec.expected_tool_calls[0].arguments == {"order_id": "A1"}
        assert spec.expected_tool_calls[1].arguments is None
        again = DialogueTurnSpec.from_dict(spec.to_dict())
        assert again == spec

    def test_expected_state_delta_must_be_json_dict(self):
        with pytest.raises(ContractValidationError):
            _turn_spec(expected_state_delta=["not", "a", "dict"])


class TestDialogueCaseSpec:
    def test_roundtrip(self):
        case = _case()
        assert DialogueCaseSpec.from_dict(case.to_dict()) == case
        assert DialogueCaseSpec.from_json(case.to_json()) == case

    def test_empty_turns_rejected(self):
        with pytest.raises(ContractValidationError):
            _case(turns=[])

    def test_duplicate_turn_id_rejected(self):
        with pytest.raises(ContractValidationError):
            _case(turns=[_turn_spec(turn_id="t1"), _turn_spec(turn_id="t1")])

    def test_ordered_turns_preserved(self):
        case = _case(
            turns=[
                _turn_spec(turn_id="z9"),
                _turn_spec(turn_id="a1"),
            ]
        )
        assert [t.turn_id for t in case.turns] == ["z9", "a1"]

    def test_initial_state_must_be_json_dict(self):
        with pytest.raises(ContractValidationError):
            _case(initial_state="not-a-dict")

    def test_no_single_turn_fields(self):
        case = _case()
        assert not hasattr(case, "user_input")
        assert not hasattr(case, "expected_response")
        assert "user_input" not in case.to_dict()


class TestDialogueTurnInput:
    def _input(self, **overrides):
        payload = {
            "turn_id": "t1",
            "user_input": "hi",
            "history": (
                DialogueHistoryEntry(role="user", content="u1"),
                DialogueHistoryEntry(role="assistant", content="a1"),
            ),
            "state_view": {"orders": {"A": {"status": "delivered"}}},
            "available_tools": (
                DialogueToolDescriptor(name="lookup_order"),
            ),
        }
        payload.update(overrides)
        return DialogueTurnInput(**payload)

    def test_roundtrip(self):
        turn_input = self._input()
        assert DialogueTurnInput.from_dict(turn_input.to_dict()) == turn_input
        assert (
            DialogueTurnInput.from_json(turn_input.to_json()) == turn_input
        )

    def test_digest_deterministic(self):
        assert (
            self._input().input_context_digest()
            == self._input().input_context_digest()
        )

    def test_digest_changes_with_history(self):
        other = self._input(
            history=(
                DialogueHistoryEntry(role="user", content="u1"),
                DialogueHistoryEntry(role="assistant", content="CHANGED"),
            )
        )
        assert (
            self._input().input_context_digest()
            != other.input_context_digest()
        )

    def test_digest_changes_with_state(self):
        other = self._input(state_view={"orders": {}})
        assert (
            self._input().input_context_digest()
            != other.input_context_digest()
        )

    def test_digest_changes_with_tool_results(self):
        other = self._input(
            history=(
                DialogueHistoryEntry(role="user", content="u1"),
                DialogueHistoryEntry(
                    role="tool",
                    content=[{"tool_name": "x", "result": {"y": 1}}],
                ),
            )
        )
        assert (
            self._input().input_context_digest()
            != other.input_context_digest()
        )

    def test_digest_covers_tools(self):
        other = self._input(available_tools=())
        assert (
            self._input().input_context_digest()
            != other.input_context_digest()
        )


class TestHistoryEntry:
    def test_roles_constrained(self):
        DialogueHistoryEntry(role="user", content="x")
        DialogueHistoryEntry(role="assistant", content="x")
        DialogueHistoryEntry(role="tool", content="x")
        with pytest.raises(ContractValidationError):
            DialogueHistoryEntry(role="system", content="x")


class TestDialogueTurnOutput:
    def test_roundtrip(self):
        output = DialogueTurnOutput(
            output_text="hi",
            tool_calls=(
                ToolEvent(tool_name="lookup_order", arguments={"a": 1}),
            ),
            assistant_tags=("clarify",),
        )
        assert DialogueTurnOutput.from_dict(output.to_dict()) == output
        assert DialogueTurnOutput.from_json(output.to_json()) == output


class TestTraces:
    def test_step_roundtrip(self):
        step = _step(
            history_digest="abc",
            tool_events=(ToolEvent(tool_name="t", status="ok"),),
        )
        assert DialogueStepTrace.from_dict(step.to_dict()) == step

    def test_turn_trace_roundtrip(self):
        trace = _turn_trace(activation_evidence={"type": "default"})
        assert DialogueTurnTrace.from_dict(trace.to_dict()) == trace

    def test_run_trace_roundtrip(self):
        trace = _run_trace()
        assert DialogueRunTrace.from_dict(trace.to_dict()) == trace
        assert DialogueRunTrace.from_json(trace.to_json()) == trace

    def test_termination_reason_constrained(self):
        for reason in TERMINATION_REASONS:
            _run_trace(termination_reason=reason)
        with pytest.raises(ContractValidationError):
            _run_trace(termination_reason="something_else")

    def test_zero_turns_allowed(self):
        trace = _run_trace(turns=(), termination_reason="user_done",
                           output_text="")
        assert trace.turns == ()


class TestToolEventStatus:
    def test_legacy_shape_roundtrips(self):
        legacy = {"tool_name": "x", "arguments": {}, "result": {"r": 1}}
        event = ToolEvent.from_dict(legacy)
        assert event.status is None
        assert event.error is None
        # to_dict must not inject the new keys for legacy events
        assert event.to_dict() == legacy

    def test_ok_event(self):
        event = ToolEvent(tool_name="x", result={"r": 1}, status="ok")
        assert event.to_dict()["status"] == "ok"
        assert "error" not in event.to_dict()
        assert ToolEvent.from_dict(event.to_dict()) == event

    def test_error_event(self):
        event = ToolEvent(
            tool_name="x", status="error", error="PRECONDITION_FAILED"
        )
        assert event.to_dict()["error"] == "PRECONDITION_FAILED"
        assert ToolEvent.from_dict(event.to_dict()) == event

    def test_error_status_requires_error_message(self):
        with pytest.raises(ContractValidationError):
            ToolEvent(tool_name="x", status="error")

    def test_ok_status_rejects_error(self):
        with pytest.raises(ContractValidationError):
            ToolEvent(tool_name="x", status="ok", error="boom")

    def test_error_without_status_rejected(self):
        with pytest.raises(ContractValidationError):
            ToolEvent(tool_name="x", error="boom")

    def test_unknown_status_rejected(self):
        with pytest.raises(ContractValidationError):
            ToolEvent(tool_name="x", status="pending")

    def test_json_roundtrip_preserves_status(self):
        import json

        event = ToolEvent(tool_name="x", status="error", error="E")
        restored = ToolEvent.from_dict(json.loads(json.dumps(event.to_dict())))
        assert restored == event
