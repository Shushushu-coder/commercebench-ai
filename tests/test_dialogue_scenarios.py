"""Phase 3B end-to-end scenario tests over the dialogue_v0 fixtures."""

from __future__ import annotations

import json
import pathlib

import pytest

from commercebench.contracts.dialogue import (
    DialogueCaseSpec,
    DialogueRunTrace,
    TERMINATION_COMPLETED,
    TERMINATION_USER_DONE,
)
from commercebench.contracts.experiment import ExperimentManifest
from commercebench.dialogue import DialogueHarness
from commercebench.evaluation.dialogue import (
    DIALOGUE_EVALUATOR_ID,
    DIALOGUE_EVALUATOR_VERSION,
    DialogueEvaluator,
    FAILURE_INVALID_TOOL_CALL,
    FAILURE_REQUIRED_TOOL_MISSING,
    FAILURE_TASK_INCOMPLETE,
)
from commercebench.systems.dialogue import ScriptedDialogueSystem

DIALOGUE_DIR = (
    pathlib.Path(__file__).resolve().parents[1] / "examples" / "dialogue_v0"
)


def _load_case(name: str) -> DialogueCaseSpec:
    return DialogueCaseSpec.from_json(
        (DIALOGUE_DIR / "cases" / name).read_text(encoding="utf-8")
    )


def _load_manifest() -> ExperimentManifest:
    return ExperimentManifest.from_json(
        (DIALOGUE_DIR / "experiments" / "dialogue_stateful_v0.json")
        .read_text(encoding="utf-8")
    )


def _load_script(name: str) -> dict:
    return json.loads(
        (DIALOGUE_DIR / "scripts" / name).read_text(encoding="utf-8")
    )


def _run(case_name, script_name):
    case = _load_case(case_name)
    manifest = _load_manifest()
    harness = DialogueHarness(config=manifest.dialogue)
    system = ScriptedDialogueSystem(_load_script(script_name))
    trace = harness.run(case, manifest, system)
    evaluation = DialogueEvaluator().evaluate(case, trace)
    return case, manifest, trace, evaluation


class TestScenarioAHappyPath:
    def test_completed_return(self):
        case, manifest, trace, evaluation = _run(
            "dialogue-return-001.json", "return_happy.json"
        )
        assert trace.termination_reason == TERMINATION_COMPLETED
        assert [t.turn_id for t in trace.turns] == ["t1", "t2", "t3"]
        assert trace.final_state["returns"]["A1001"]["status"] == "requested"
        assert evaluation.passed is True
        assert evaluation.failure_categories == ()
        assert (
            evaluation.metrics["final_state_match"].value is True
        )
        assert evaluation.metrics["required_tool_coverage"].value == 1.0
        assert evaluation.metrics["invalid_tool_call_count"].value == 0
        # clarify tag drove t2 activation
        assert trace.turns[1].activation_evidence == {
            "type": "assistant_tag", "value": "clarify"
        }
        assert trace.turns[2].activation_evidence == {
            "type": "tool", "value": "lookup_order"
        }


class TestScenarioBPrecondition:
    def test_broken_precondition_failure(self):
        case, manifest, trace, evaluation = _run(
            "dialogue-order-status-001.json", "order_status_broken.json"
        )
        assert trace.termination_reason == TERMINATION_USER_DONE
        assert evaluation.passed is False
        categories = set(evaluation.failure_categories)
        assert FAILURE_TASK_INCOMPLETE in categories
        assert FAILURE_INVALID_TOOL_CALL in categories
        assert FAILURE_REQUIRED_TOOL_MISSING in categories
        # the failed request_return left state untouched
        assert trace.final_state.get("returns", {}) == {}
        errors = [
            e for t in trace.turns for e in t.tool_events if e.status == "error"
        ]
        assert errors[0].error == "PRECONDITION_FAILED"

    def test_happy_path_recovers_to_eligible_order(self):
        case, manifest, trace, evaluation = _run(
            "dialogue-order-status-001.json", "order_status_happy.json"
        )
        assert trace.termination_reason == TERMINATION_COMPLETED
        assert trace.final_state["returns"]["B2002"]["status"] == "requested"
        assert "B2001" not in trace.final_state["returns"]
        assert evaluation.passed is True


class TestScenarioCEntity:
    def test_happy_path(self):
        case, manifest, trace, evaluation = _run(
            "dialogue-entity-001.json", "entity_happy.json"
        )
        assert trace.termination_reason == TERMINATION_COMPLETED
        assert trace.final_state["returns"]["C3001"]["status"] == "requested"
        assert evaluation.passed is True

    def test_wrong_entity_fails_without_invalid_calls(self):
        # The broken script targets C3002 — every tool call succeeds,
        # so failure is purely wrong-entity: state mismatch +
        # missing required call, with zero invalid calls.
        case, manifest, trace, evaluation = _run(
            "dialogue-entity-001.json", "entity_broken.json"
        )
        assert evaluation.passed is False
        categories = set(evaluation.failure_categories)
        assert FAILURE_TASK_INCOMPLETE in categories
        assert FAILURE_REQUIRED_TOOL_MISSING in categories
        assert FAILURE_INVALID_TOOL_CALL not in categories
        assert evaluation.metrics["invalid_tool_call_count"].value == 0
        assert "C3002" in trace.final_state["returns"]


class TestPersistedReplay:
    def test_roundtrip_through_json(self):
        case, manifest, trace, live = _run(
            "dialogue-return-001.json", "return_happy.json"
        )
        # persist + reload both contracts — no files needed
        case2 = DialogueCaseSpec.from_json(case.to_json())
        trace2 = DialogueRunTrace.from_json(trace.to_json())
        assert case2 == case
        assert trace2 == trace

        replayed = DialogueEvaluator().evaluate(case2, trace2)
        assert replayed.metrics.keys() == live.metrics.keys()
        for name in live.metrics:
            assert replayed.metrics[name].value == live.metrics[name].value
            assert replayed.metrics[name].passed == live.metrics[name].passed
        assert replayed.passed == live.passed
        assert replayed.failure_categories == live.failure_categories


class TestEvaluatorIdentity:
    def test_manifest_evaluator_matches(self):
        manifest = _load_manifest()
        assert manifest.evaluator_id == DIALOGUE_EVALUATOR_ID
        assert manifest.evaluator_version == DIALOGUE_EVALUATOR_VERSION
