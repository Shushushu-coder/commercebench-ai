"""Phase 3B DialogueHarness tests: orchestration invariants."""

from __future__ import annotations

import pytest

from commercebench.contracts import SCHEMA_VERSION, ContractValidationError
from commercebench.contracts.dialogue import (
    DialogueCaseSpec,
    DialogueTurnInput,
    DialogueTurnOutput,
    DialogueTurnSpec,
    TERMINATION_COMPLETED,
    TERMINATION_MAX_TURNS,
    TERMINATION_SYSTEM_ERROR,
    TERMINATION_TOOL_LIMIT,
    TERMINATION_USER_DONE,
)
from commercebench.contracts.experiment import (
    DialogueHarnessConfig,
    ExperimentManifest,
)
from commercebench.contracts.trace import UsageStats
from commercebench.dialogue import DialogueHarness
from commercebench.systems.dialogue import ScriptedDialogueSystem


ORDER_STATE = {
    "orders": {
        "A1": {"status": "delivered", "returnable": True},
    },
    "returns": {},
}


def _parameters(**overrides):
    parameters = {
        "max_turns": 6,
        "max_tool_calls_per_turn": 3,
        "history_policy": "full",
        "tool_policy": "deterministic-commerce-v0",
        "tool_simulator": {"id": "commerce-tools-v0", "version": "0.1"},
    }
    parameters.update(overrides)
    return parameters


def _harness_config(**overrides):
    return DialogueHarnessConfig(
        harness_id="dialogue-harness-v0",
        harness_version="0.1",
        parameters=_parameters(**overrides),
    )


def _manifest(dialogue=None, case_ids=("dlg-001",)):
    return ExperimentManifest(
        schema_version=SCHEMA_VERSION,
        experiment_id="dlg-exp",
        experiment_name="dlg-exp",
        benchmark_version="commercebench-dev-0.1",
        case_ids=case_ids,
        system_id="scripted-dialogue",
        system_version="0.1",
        evaluator_id="dialogue-deterministic-v0",
        evaluator_version="0.1",
        dialogue=dialogue,
    )


def _spec(turn_id, user_input, activation=None, **kw):
    return DialogueTurnSpec(
        turn_id=turn_id, user_input=user_input, activation=activation, **kw
    )


def _case(turns, initial_state=None, expected_final_state=None, **kw):
    payload = {
        "schema_version": SCHEMA_VERSION,
        "case_id": "dlg-001",
        "benchmark_version": "commercebench-dev-0.1",
        "task_family": "commerce_dialogue",
        "intent": "return",
        "difficulty": "medium",
        "answerability": "answerable",
        "turns": turns,
        "initial_state": initial_state or {"orders": {}},
        # Default goal is intentionally never satisfied by the run —
        # tests that want "completed" pass an achievable state.
        "expected_final_state": (
            {"_goal": "unmet"}
            if expected_final_state is None
            else expected_final_state
        ),
    }
    payload.update(kw)
    return DialogueCaseSpec(**payload)


class SpySystem:
    """Records every DialogueTurnInput; delegates to a script."""

    def __init__(self, script):
        self.system_id = "spy-dialogue"
        self.system_version = "0.1"
        self._system = ScriptedDialogueSystem(script)
        self.inputs = []

    def respond(self, turn_input: DialogueTurnInput) -> DialogueTurnOutput:
        self.inputs.append(turn_input)
        return self._system.respond(turn_input)


class LieAboutStateSystem:
    """Claims state changes via runtime_metadata and mutates the
    state view; neither may affect canonical state."""

    system_id = "liar"
    system_version = "0.1"

    def respond(self, turn_input: DialogueTurnInput) -> DialogueTurnOutput:
        turn_input.state_view["orders"]["A1"]["status"] = "shipped"
        return DialogueTurnOutput(
            output_text="state changed",
            runtime_metadata={"returns": {"A1": {"status": "requested"}}},
            usage=UsageStats(input_tokens=0, output_tokens=0,
                             total_tokens=0),
        )


class ExplodingSystem:
    system_id = "boom"
    system_version = "0.1"

    def respond(self, turn_input: DialogueTurnInput) -> DialogueTurnOutput:
        raise RuntimeError("kaboom")


class TestHappyPath:
    def test_completed_run(self):
        case = _case(
            turns=[
                _spec("t1", "退货", None),
                _spec(
                    "t2",
                    "订单A1",
                    activation={"on_assistant_tag": "clarify"},
                    expected_tool_calls=[
                        {"tool_name": "lookup_order",
                         "arguments": {"order_id": "A1"}}
                    ],
                ),
                _spec(
                    "t3",
                    "确认",
                    activation={"on_tool": "lookup_order"},
                    expected_tool_calls=[
                        {"tool_name": "request_return",
                         "arguments": {"order_id": "A1"}}
                    ],
                ),
            ],
            initial_state=ORDER_STATE,
            expected_final_state={
                "returns": {"A1": {"status": "requested"}}
            },
        )
        script = {
            "t1": [{"output_text": "订单号？", "assistant_tags": ["clarify"]}],
            "t2": [
                {"output_text": "", "tool_calls": [
                    {"tool_name": "lookup_order",
                     "arguments": {"order_id": "A1"}}]},
                {"output_text": "A1已签收，可退。"},
            ],
            "t3": [
                {"output_text": "", "tool_calls": [
                    {"tool_name": "request_return",
                     "arguments": {"order_id": "A1"}}]},
                {"output_text": "已申请。"},
            ],
        }
        manifest = _manifest(dialogue=_harness_config())
        trace = DialogueHarness(config=manifest.dialogue).run(
            case, manifest, ScriptedDialogueSystem(script)
        )
        assert trace.termination_reason == TERMINATION_COMPLETED
        assert len(trace.turns) == 3
        assert trace.final_state["returns"]["A1"]["status"] == "requested"
        assert trace.turns[1].activation_evidence["value"] == "clarify"
        assert trace.turns[2].activation_evidence["value"] == "lookup_order"
        # Inner-loop steps preserved: t2 has tool-call step + reply step
        assert len(trace.turns[1].steps) == 2
        assert trace.turns[1].steps[0].tool_events[0].status == "ok"
        assert trace.turns[1].steps[0].tool_events[0].result["status"] == (
            "delivered"
        )

    def test_early_completion_stops_remaining_turns(self):
        # Goal met at t1; t2 (unconsumed) must not fire.
        case = _case(
            turns=[
                _spec("t1", "退A1", None),
                _spec("t2", "more", None),
            ],
            initial_state=ORDER_STATE,
            expected_final_state={
                "returns": {"A1": {"status": "requested"}}
            },
        )
        script = {
            "t1": [
                {"output_text": "", "tool_calls": [
                    {"tool_name": "request_return",
                     "arguments": {"order_id": "A1"}}]},
                {"output_text": "done"},
            ],
        }
        manifest = _manifest(dialogue=_harness_config())
        trace = DialogueHarness(config=manifest.dialogue).run(
            case, manifest, ScriptedDialogueSystem(script)
        )
        assert trace.termination_reason == TERMINATION_COMPLETED
        assert [t.turn_id for t in trace.turns] == ["t1"]


class TestActivation:
    def test_conditional_turn_not_released_without_tag(self):
        case = _case(
            turns=[
                _spec("t1", "hi", None),
                _spec("t2", "order id", {"on_assistant_tag": "clarify"}),
            ],
            initial_state=ORDER_STATE,
        )
        script = {"t1": [{"output_text": "no clarify here"}]}
        trace = DialogueHarness(config=_harness_config()).run(
            case, _manifest(dialogue=_harness_config()),
            ScriptedDialogueSystem(script),
        )
        assert len(trace.turns) == 1
        assert trace.termination_reason == TERMINATION_USER_DONE

    def test_on_tool_requires_successful_call(self):
        # t2 requires a successful lookup_order; t1 only fails one.
        case = _case(
            turns=[
                _spec("t1", "hi", None),
                _spec("t2", "next", {"on_tool": "lookup_order"}),
            ],
            initial_state=ORDER_STATE,
        )
        script = {
            "t1": [
                {"output_text": "", "tool_calls": [
                    {"tool_name": "lookup_order",
                     "arguments": {"order_id": "GHOST"}}]},
                {"output_text": "not found"},
            ],
        }
        trace = DialogueHarness(config=_harness_config()).run(
            case, _manifest(dialogue=_harness_config()),
            ScriptedDialogueSystem(script),
        )
        assert len(trace.turns) == 1
        assert trace.turns[0].tool_events[0].status == "error"
        assert trace.termination_reason == TERMINATION_USER_DONE

    def test_turn_consumed_once(self):
        # t2 fires once even though its activation stays satisfied.
        case = _case(
            turns=[
                _spec("t1", "hi", None),
                _spec("t2", "second", {"on_assistant_tag": "clarify"}),
            ],
            initial_state=ORDER_STATE,
        )
        script = {
            "t1": [{"output_text": "q?", "assistant_tags": ["clarify"]}],
            "t2": [{"output_text": "answered"}],
        }
        trace = DialogueHarness(config=_harness_config()).run(
            case, _manifest(dialogue=_harness_config()),
            ScriptedDialogueSystem(script),
        )
        assert [t.turn_id for t in trace.turns] == ["t1", "t2"]
        assert trace.termination_reason == TERMINATION_USER_DONE


class TestLeakage:
    def test_no_future_turn_leakage(self):
        case = _case(
            turns=[
                _spec("t1", "first user", None),
                _spec("t2", "second user", {"on_assistant_tag": "clarify"}),
                _spec("t3", "third user", {"on_tool": "lookup_order"}),
            ],
            initial_state=ORDER_STATE,
        )
        script = {
            "t1": [{"output_text": "ask", "assistant_tags": ["clarify"]}],
            "t2": [
                {"output_text": "", "tool_calls": [
                    {"tool_name": "lookup_order",
                     "arguments": {"order_id": "A1"}}]},
                {"output_text": "ok"},
            ],
            "t3": [{"output_text": "done"}],
        }
        spy = SpySystem(script)
        manifest = _manifest(dialogue=_harness_config())
        trace = DialogueHarness(config=manifest.dialogue).run(
            case, manifest, spy
        )
        released = [turn.user_input for turn in trace.turns]
        for turn_input in spy.inputs:
            idx = released.index(turn_input.user_input)
            visible_users = [
                e.content for e in turn_input.history if e.role == "user"
            ]
            # history holds all released inputs including the current
            # one (it is part of the visible context), and nothing
            # from future turns.
            assert visible_users == released[: idx + 1]
            assert turn_input.user_input == released[idx]
            # no future user text anywhere in the visible context
            for future in released[idx + 1:]:
                for entry in turn_input.history:
                    assert entry.content != future

    def test_turn_input_state_view_is_snapshot(self):
        case = _case(
            turns=[_spec("t1", "hi", None)],
            initial_state=ORDER_STATE,
        )
        spy = SpySystem({"t1": [{"output_text": "ok"}]})
        manifest = _manifest(dialogue=_harness_config())
        trace = DialogueHarness(config=manifest.dialogue).run(
            case, manifest, spy
        )
        spy.inputs[0].state_view["orders"]["A1"]["status"] = "mutated"
        assert trace.final_state["orders"]["A1"]["status"] == "delivered"
        assert ORDER_STATE["orders"]["A1"]["status"] == "delivered"


class TestStateAuthority:
    def test_system_metadata_cannot_change_state(self):
        case = _case(
            turns=[_spec("t1", "hi", None)],
            initial_state=ORDER_STATE,
        )
        trace = DialogueHarness(config=_harness_config()).run(
            case, _manifest(dialogue=_harness_config()),
            LieAboutStateSystem(),
        )
        assert trace.final_state == ORDER_STATE
        assert trace.turns[0].state_after == ORDER_STATE

    def test_failed_tool_does_not_change_state(self):
        case = _case(
            turns=[_spec("t1", "hi", None)],
            initial_state=ORDER_STATE,
        )
        script = {
            "t1": [
                {"output_text": "", "tool_calls": [
                    {"tool_name": "request_return",
                     "arguments": {"order_id": "GHOST"}}]},
                {"output_text": "hmm"},
            ],
        }
        trace = DialogueHarness(config=_harness_config()).run(
            case, _manifest(dialogue=_harness_config()),
            ScriptedDialogueSystem(script),
        )
        assert trace.turns[0].tool_events[0].status == "error"
        assert trace.turns[0].state_after == trace.turns[0].state_before
        assert trace.final_state == ORDER_STATE


class TestTermination:
    def test_max_turns(self):
        case = _case(
            turns=[_spec(f"t{i}", f"msg{i}", None) for i in range(1, 5)],
            initial_state=ORDER_STATE,
        )
        script = {f"t{i}": [{"output_text": "r"}] for i in range(1, 5)}
        manifest = _manifest(dialogue=_harness_config(max_turns=2))
        trace = DialogueHarness(config=manifest.dialogue).run(
            case, manifest, ScriptedDialogueSystem(script)
        )
        assert trace.termination_reason == TERMINATION_MAX_TURNS
        assert len(trace.turns) == 2

    def test_tool_limit(self):
        case = _case(
            turns=[_spec("t1", "hi", None)],
            initial_state=ORDER_STATE,
        )
        script = {
            "t1": [
                # proposes two calls per step; budget is 3
                {"output_text": "", "tool_calls": [
                    {"tool_name": "lookup_order",
                     "arguments": {"order_id": "A1"}},
                    {"tool_name": "lookup_order",
                     "arguments": {"order_id": "A1"}},
                ]},
                {"output_text": "", "tool_calls": [
                    {"tool_name": "lookup_order",
                     "arguments": {"order_id": "A1"}},
                    {"tool_name": "lookup_order",
                     "arguments": {"order_id": "A1"}},
                ]},
            ],
        }
        manifest = _manifest(
            dialogue=_harness_config(max_tool_calls_per_turn=3)
        )
        trace = DialogueHarness(config=manifest.dialogue).run(
            case, manifest, ScriptedDialogueSystem(script)
        )
        assert trace.termination_reason == TERMINATION_TOOL_LIMIT
        assert len(trace.turns) == 1
        executed = [
            e for s in trace.turns[0].steps for e in s.tool_events
        ]
        assert len(executed) == 3  # exactly budget; excess dropped
        assert (
            trace.turns[0].steps[-1].runtime_metadata[
                "tool_calls_not_executed"
            ] == 1
        )

    def test_system_error(self):
        case = _case(
            turns=[_spec("t1", "hi", None)],
            initial_state=ORDER_STATE,
        )
        trace = DialogueHarness(config=_harness_config()).run(
            case, _manifest(dialogue=_harness_config()), ExplodingSystem()
        )
        assert trace.termination_reason == TERMINATION_SYSTEM_ERROR
        assert trace.turns == ()
        assert trace.final_state == ORDER_STATE
        assert trace.output_text == ""

    def test_user_done_when_script_exhausted(self):
        case = _case(
            turns=[_spec("t1", "hi", None)],
            initial_state=ORDER_STATE,
        )
        trace = DialogueHarness(config=_harness_config()).run(
            case, _manifest(dialogue=_harness_config()),
            ScriptedDialogueSystem({"t1": [{"output_text": "r"}]}),
        )
        assert trace.termination_reason == TERMINATION_USER_DONE
        assert len(trace.turns) == 1


class TestHistoryEvidence:
    def test_history_contains_prior_turns_and_tools(self):
        case = _case(
            turns=[
                _spec("t1", "u1", None),
                _spec("t2", "u2", {"on_assistant_tag": "clarify"}),
            ],
            initial_state=ORDER_STATE,
        )
        script = {
            "t1": [{"output_text": "a1", "assistant_tags": ["clarify"]}],
            "t2": [
                {"output_text": "", "tool_calls": [
                    {"tool_name": "lookup_order",
                     "arguments": {"order_id": "A1"}}]},
                {"output_text": "a2"},
            ],
        }
        spy = SpySystem(script)
        manifest = _manifest(dialogue=_harness_config())
        trace = DialogueHarness(config=manifest.dialogue).run(
            case, manifest, spy
        )
        last_input = spy.inputs[-1]
        roles = [e.role for e in last_input.history]
        # u1, a1, u2, assistant tool-call step, tool result
        assert roles == [
            "user", "assistant", "user", "assistant", "tool"
        ]
        assert last_input.history[1].content == "a1"
        tool_entry = last_input.history[4]
        assert tool_entry.content[0]["tool_name"] == "lookup_order"
        assert len(trace.turns) == 2

    def test_digests_differ_across_steps(self):
        case = _case(
            turns=[_spec("t1", "u1", None)],
            initial_state=ORDER_STATE,
        )
        script = {
            "t1": [
                {"output_text": "", "tool_calls": [
                    {"tool_name": "lookup_order",
                     "arguments": {"order_id": "A1"}}]},
                {"output_text": "done"},
            ],
        }
        trace = DialogueHarness(config=_harness_config()).run(
            case, _manifest(dialogue=_harness_config()),
            ScriptedDialogueSystem(script),
        )
        digests = [s.history_digest for s in trace.turns[0].steps]
        assert len(set(digests)) == 2


class TestManifestBinding:
    def test_case_not_listed_rejected(self):
        case = _case(turns=[_spec("t1", "hi", None)])
        manifest = _manifest(
            dialogue=_harness_config(), case_ids=("other",)
        )
        with pytest.raises(ContractValidationError):
            DialogueHarness(config=manifest.dialogue).run(
                case, manifest, ScriptedDialogueSystem({})
            )

    def test_simulator_identity_mismatch_rejected(self):
        manifest = _manifest(
            dialogue=_harness_config(
                tool_simulator={"id": "other-sim", "version": "0.1"}
            )
        )
        with pytest.raises(ContractValidationError):
            DialogueHarness(config=manifest.dialogue)
