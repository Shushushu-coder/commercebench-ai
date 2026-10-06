"""Phase 3D integrity tests: tool policy, manifest binding,
defensive evidence snapshots, strict state equality, and the
completion/state-delta semantic alignment.
"""

from __future__ import annotations

import copy

import pytest

from commercebench.contracts import SCHEMA_VERSION, ContractValidationError
from commercebench.contracts.dialogue import (
    DialogueCaseSpec,
    DialogueTurnInput,
    DialogueTurnOutput,
    DialogueTurnSpec,
    TERMINATION_COMPLETED,
    TERMINATION_USER_DONE,
)
from commercebench.contracts.experiment import (
    DialogueHarnessConfig,
    ExperimentManifest,
)
from commercebench.contracts.trace import ToolEvent, UsageStats
from commercebench.dialogue import DialogueHarness, run_dialogue_case
from commercebench.dialogue.policy import (
    ERROR_POLICY_PRECONDITION_FAILED,
    TOOL_POLICY_DIRECT,
    TOOL_POLICY_LOOKUP_BEFORE_MUTATE,
)
from commercebench.dialogue.state import (
    json_leaf_equal,
    json_value_equal,
    state_subset_matches,
)
from commercebench.dialogue.tools import TOOL_DESCRIPTORS, tool_schema_hash
from commercebench.evaluation.dialogue import DialogueEvaluator
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
        "tool_policy": TOOL_POLICY_DIRECT,
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
        "expected_final_state": (
            {"_goal": "unmet"}
            if expected_final_state is None
            else expected_final_state
        ),
    }
    payload.update(kw)
    return DialogueCaseSpec(**payload)


def _usage():
    return UsageStats(input_tokens=0, output_tokens=0, total_tokens=0)


class TestToolPolicyRegistry:
    def test_unknown_policy_rejected_in_config(self):
        with pytest.raises(ContractValidationError):
            _harness_config(tool_policy="deterministic-commerce-v0")

    def test_unknown_policy_rejected_in_manifest(self):
        with pytest.raises(ContractValidationError):
            _manifest(dialogue=_harness_config(tool_policy="anything"))

    def test_registered_policies_accepted(self):
        for policy in (TOOL_POLICY_DIRECT, TOOL_POLICY_LOOKUP_BEFORE_MUTATE):
            cfg = _harness_config(tool_policy=policy)
            assert cfg.parameters["tool_policy"] == policy


class TestLookupBeforeMutatePolicy:
    """lookup-before-mutate-v0: a state-changing call needs an earlier
    successful lookup_order(order_id) in the same dialogue."""

    def _policy_run(self, script, policy, expected_final_state=None):
        case = _case(
            turns=[
                _spec("t1", "return A1", None),
            ],
            initial_state=ORDER_STATE,
            expected_final_state=(
                expected_final_state
                if expected_final_state is not None
                else {"returns": {"A1": {"status": "requested"}}}
            ),
        )
        manifest = _manifest(dialogue=_harness_config(tool_policy=policy))
        trace = DialogueHarness(config=manifest.dialogue).run(
            case, manifest, ScriptedDialogueSystem(script)
        )
        return trace

    def test_policy_blocks_ungated_mutation(self):
        script = {
            "t1": [
                {"output_text": "", "tool_calls": [
                    {"tool_name": "request_return",
                     "arguments": {"order_id": "A1"}}]},
                {"output_text": "done"},
            ],
        }
        trace = self._policy_run(script, TOOL_POLICY_LOOKUP_BEFORE_MUTATE)
        event = trace.turns[0].tool_events[0]
        assert event.status == "error"
        assert event.error == ERROR_POLICY_PRECONDITION_FAILED
        assert event.result["policy"] == TOOL_POLICY_LOOKUP_BEFORE_MUTATE
        assert event.result["order_id"] == "A1"
        # State untouched; completion never reached.
        assert trace.final_state["returns"] == {}
        assert trace.turns[0].state_after == trace.turns[0].state_before
        assert trace.termination_reason == TERMINATION_USER_DONE

    def test_policy_allows_after_successful_lookup(self):
        script = {
            "t1": [
                {"output_text": "", "tool_calls": [
                    {"tool_name": "lookup_order",
                     "arguments": {"order_id": "A1"}}]},
                {"output_text": "", "tool_calls": [
                    {"tool_name": "request_return",
                     "arguments": {"order_id": "A1"}}]},
                {"output_text": "done"},
            ],
        }
        trace = self._policy_run(script, TOOL_POLICY_LOOKUP_BEFORE_MUTATE)
        assert trace.turns[0].tool_events[1].status == "ok"
        assert trace.final_state["returns"]["A1"]["status"] == "requested"
        assert trace.termination_reason == TERMINATION_COMPLETED

    def test_same_step_lookup_then_mutate_allowed(self):
        script = {
            "t1": [
                {"output_text": "", "tool_calls": [
                    {"tool_name": "lookup_order",
                     "arguments": {"order_id": "A1"}},
                    {"tool_name": "request_return",
                     "arguments": {"order_id": "A1"}},
                ]},
                {"output_text": "done"},
            ],
        }
        trace = self._policy_run(script, TOOL_POLICY_LOOKUP_BEFORE_MUTATE)
        assert all(e.status == "ok" for e in trace.turns[0].tool_events)
        assert trace.termination_reason == TERMINATION_COMPLETED

    def test_failed_lookup_does_not_unlock(self):
        # Lookup of a missing order fails; the later mutate is gated.
        script = {
            "t1": [
                {"output_text": "", "tool_calls": [
                    {"tool_name": "lookup_order",
                     "arguments": {"order_id": "GHOST"}},
                    {"tool_name": "request_return",
                     "arguments": {"order_id": "A1"}},
                ]},
                {"output_text": "done"},
            ],
        }
        trace = self._policy_run(script, TOOL_POLICY_LOOKUP_BEFORE_MUTATE)
        events = trace.turns[0].tool_events
        assert events[0].error == "ORDER_NOT_FOUND"
        assert events[1].error == ERROR_POLICY_PRECONDITION_FAILED
        assert trace.final_state["returns"] == {}

    def test_policy_affects_runtime(self):
        # Fingerprint/runtime symmetry: same case and script, different
        # policy → different trace evidence.
        script = {
            "t1": [
                {"output_text": "", "tool_calls": [
                    {"tool_name": "request_return",
                     "arguments": {"order_id": "A1"}}]},
                {"output_text": "done"},
            ],
        }
        direct = self._policy_run(script, TOOL_POLICY_DIRECT)
        gated = self._policy_run(script, TOOL_POLICY_LOOKUP_BEFORE_MUTATE)
        assert direct.turns[0].tool_events[0].status == "ok"
        assert gated.turns[0].tool_events[0].status == "error"
        assert direct.final_state["returns"]["A1"]["status"] == "requested"
        assert gated.final_state["returns"] == {}
        assert direct.termination_reason != gated.termination_reason

    def test_readonly_tool_never_gated(self):
        script = {
            "t1": [
                {"output_text": "", "tool_calls": [
                    {"tool_name": "lookup_order",
                     "arguments": {"order_id": "A1"}}]},
                {"output_text": "done"},
            ],
        }
        trace = self._policy_run(script, TOOL_POLICY_LOOKUP_BEFORE_MUTATE)
        assert trace.turns[0].tool_events[0].status == "ok"


class TestManifestHarnessBinding:
    def test_harness_config_mismatch_rejected(self):
        case = _case(turns=[_spec("t1", "hi", None)])
        manifest = _manifest(dialogue=_harness_config())
        other = DialogueHarness(
            config=_harness_config(max_turns=1)
        )
        with pytest.raises(ContractValidationError):
            other.run(case, manifest, ScriptedDialogueSystem({}))

    def test_harness_version_mismatch_rejected(self):
        case = _case(turns=[_spec("t1", "hi", None)])
        manifest = _manifest(dialogue=_harness_config())
        other = DialogueHarness(
            config=DialogueHarnessConfig(
                harness_id="dialogue-harness-v0",
                harness_version="0.2",
                parameters=_parameters(),
            )
        )
        with pytest.raises(ContractValidationError):
            other.run(case, manifest, ScriptedDialogueSystem({}))

    def test_manifest_dialogue_none_rejected(self):
        case = _case(turns=[_spec("t1", "hi", None)])
        manifest = _manifest(dialogue=None)
        with pytest.raises(ContractValidationError):
            DialogueHarness(config=_harness_config()).run(
                case, manifest, ScriptedDialogueSystem({})
            )

    def test_unconfigured_harness_rejected(self):
        case = _case(turns=[_spec("t1", "hi", None)])
        manifest = _manifest(dialogue=_harness_config())
        with pytest.raises(ContractValidationError):
            DialogueHarness(config=None).run(
                case, manifest, ScriptedDialogueSystem({})
            )

    def test_runner_builds_harness_from_manifest(self):
        case = _case(turns=[_spec("t1", "hi", None)])
        manifest = _manifest(dialogue=_harness_config())
        trace = run_dialogue_case(
            case, manifest, ScriptedDialogueSystem({"t1": [{"output_text": "r"}]})
        )
        assert len(trace.turns) == 1
        assert (
            trace.runtime_metadata["harness"]["tool_policy"]
            == TOOL_POLICY_DIRECT
        )

    def test_runner_rejects_manifest_without_dialogue(self):
        case = _case(turns=[_spec("t1", "hi", None)])
        manifest = _manifest(dialogue=None)
        with pytest.raises(ContractValidationError):
            run_dialogue_case(
                case, manifest, ScriptedDialogueSystem({})
            )

    def test_equal_config_accepted(self):
        # A separately constructed but identical config is canonical-equal.
        case = _case(turns=[_spec("t1", "hi", None)])
        manifest = _manifest(dialogue=_harness_config())
        trace = DialogueHarness(config=_harness_config()).run(
            case, manifest, ScriptedDialogueSystem({"t1": [{"output_text": "r"}]})
        )
        assert len(trace.turns) == 1


class TestEvidenceImmutability:
    """A system that mutates everything it can reach must not corrupt
    persisted trace, stored history, or future inputs."""

    def test_history_and_event_mutation_blocked(self):
        captured = []

        class CorruptingSystem:
            system_id = "corruptor"
            system_version = "0.1"

            def respond(self, turn_input: DialogueTurnInput):
                # Deep-snapshot what the system actually saw BEFORE
                # corrupting it — assertions run against this.
                captured.append(
                    (turn_input.turn_id, copy.deepcopy(turn_input))
                )
                # In-place mutation of everything mutable the input
                # exposes: list/dict history content, nested tool
                # args/results, and the state view. (Field assignment on
                # the frozen dataclasses is structurally impossible; the
                # threat model is shared mutable payloads.)
                for entry in turn_input.history:
                    if isinstance(entry.content, list):
                        entry.content.clear()
                        entry.content.append({"tool_name": "forged"})
                    elif isinstance(entry.content, dict):
                        entry.content["forged"] = True
                    for event in entry.tool_events:
                        event.arguments["order_id"] = "FORGED"
                        if isinstance(event.result, dict):
                            event.result["forged"] = True
                turn_input.state_view.clear()
                turn_input.state_view["forged"] = True
                first_step = not any(
                    e.role == "assistant" for e in turn_input.history
                )
                if turn_input.turn_id == "t1" and first_step:
                    return DialogueTurnOutput(
                        output_text="",
                        tool_calls=(
                            ToolEvent(
                                tool_name="lookup_order",
                                arguments={
                                    "order_id": "A1",
                                    "meta": {"nested": [1, 2]},
                                },
                            ),
                        ),
                        usage=_usage(),
                    )
                return DialogueTurnOutput(
                    output_text="done", usage=_usage()
                )

        case = _case(
            turns=[
                _spec("t1", "u1", None),
                _spec("t2", "u2", None),
            ],
            initial_state=ORDER_STATE,
        )
        manifest = _manifest(dialogue=_harness_config())
        trace = DialogueHarness(config=manifest.dialogue).run(
            case, manifest, CorruptingSystem()
        )
        assert len(trace.turns) == 2

        # Persisted tool event evidence is intact.
        event = trace.turns[0].tool_events[0]
        assert event.tool_name == "lookup_order"
        assert event.arguments == {"order_id": "A1", "meta": {"nested": [1, 2]}}
        assert event.result["status"] == "delivered"
        assert "forged" not in str(event.result)

        # The second turn's input history was clean (rebuilt from stored
        # snapshots, not the mutated objects).
        second_id, second_input = captured[-1]
        assert second_id == "t2"
        contents = [e.content for e in second_input.history]
        assert "u1" in contents
        assert "u2" in contents
        tool_entries = [e for e in second_input.history if e.role == "tool"]
        assert tool_entries
        assert tool_entries[0].content[0]["tool_name"] == "lookup_order"
        assert tool_entries[0].content[0]["result"]["status"] == "delivered"

        # Canonical state survived: only the real lookup happened.
        assert trace.final_state == ORDER_STATE

    def test_system_mutating_returned_proposal_blocked(self):
        class LateMutatingSystem:
            """Mutates its own proposed call objects after returning."""

            system_id = "mut-proposals"
            system_version = "0.1"

            def __init__(self):
                self._last = None

            def respond(self, turn_input):
                if self._last is not None:
                    # Corrupt the proposal we already returned — in-place
                    # mutation of the mutable payloads only.
                    self._last.arguments["order_id"] = "FORGED"
                    self._last.result["forged"] = True
                if turn_input.turn_id == "t1":
                    call = ToolEvent(
                        tool_name="lookup_order",
                        arguments={"order_id": "A1"},
                        result={"seed": "mutable"},
                    )
                    self._last = call
                    return DialogueTurnOutput(
                        output_text="",
                        tool_calls=(call,),
                        usage=_usage(),
                    )
                return DialogueTurnOutput(output_text="done", usage=_usage())

        case = _case(
            turns=[
                _spec("t1", "u1", None),
                _spec("t2", "u2", None),
            ],
            initial_state=ORDER_STATE,
        )
        manifest = _manifest(dialogue=_harness_config())
        trace = DialogueHarness(config=manifest.dialogue).run(
            case, manifest, LateMutatingSystem()
        )
        event = trace.turns[0].tool_events[0]
        assert event.arguments == {"order_id": "A1"}
        assert event.result["status"] == "delivered"
        # Every stored view is intact: step-level, turn-level, history.
        step_event = trace.turns[0].steps[0].tool_events[0]
        assert step_event.arguments == {"order_id": "A1"}
        assert step_event.result["status"] == "delivered"

    def test_runtime_metadata_mutation_blocked(self):
        held = {}

        class MetadataMutator:
            system_id = "meta"
            system_version = "0.1"

            def respond(self, turn_input):
                if held:
                    held["meta"]["injected"] = True
                meta = {"evidence": {"x": 1}}
                held["meta"] = meta
                return DialogueTurnOutput(
                    output_text="r",
                    runtime_metadata=meta,
                    usage=_usage(),
                )

        case = _case(
            turns=[_spec("t1", "u1", None), _spec("t2", "u2", None)],
            initial_state=ORDER_STATE,
        )
        manifest = _manifest(dialogue=_harness_config())
        trace = DialogueHarness(config=manifest.dialogue).run(
            case, manifest, MetadataMutator()
        )
        step_meta = trace.turns[0].steps[0].runtime_metadata
        assert step_meta["evidence"] == {"x": 1}
        assert "injected" not in step_meta.get("meta", {})

    def test_input_digest_stable_under_mutation(self):
        recorded = {}

        class DigestMutator:
            system_id = "digest"
            system_version = "0.1"

            def respond(self, turn_input):
                recorded["digest"] = turn_input.input_context_digest()
                turn_input.state_view["injected"] = True
                for entry in turn_input.history:
                    if isinstance(entry.content, list):
                        entry.content.append({"forged": True})
                recorded["after"] = turn_input.input_context_digest()
                return DialogueTurnOutput(output_text="r", usage=_usage())

        case = _case(
            turns=[_spec("t1", "u1", None)], initial_state=ORDER_STATE
        )
        manifest = _manifest(dialogue=_harness_config())
        trace = DialogueHarness(config=manifest.dialogue).run(
            case, manifest, DigestMutator()
        )
        # The persisted digest is the one computed from the pristine
        # input snapshot the system received — mutation after the fact
        # cannot rewrite it, and the digest matches a freshly
        # reconstructed equivalent input.
        step_digest = trace.turns[0].steps[0].history_digest
        assert step_digest == recorded["digest"]
        pristine = DialogueTurnInput(
            turn_id="t1",
            user_input="u1",
            history=(
                {"role": "user", "content": "u1"},
            ),
            state_view=ORDER_STATE,
            available_tools=TOOL_DESCRIPTORS,
        )
        assert pristine.input_context_digest() == step_digest
        # The mutation did change the post-hoc context — proving the
        # recorded digest was pinned to the pre-mutation snapshot.
        assert recorded["after"] != recorded["digest"]


class TestStrictLeafEquality:
    def test_bool_ne_int(self):
        assert not json_leaf_equal(True, 1)
        assert not json_leaf_equal(False, 0)
        assert json_leaf_equal(True, True)

    def test_int_float_same_json_number(self):
        assert json_leaf_equal(1, 1.0)
        assert json_leaf_equal(2, 2.0)

    def test_subset_match_type_strict(self):
        assert not state_subset_matches({"a": True}, {"a": 1})
        assert state_subset_matches({"a": 1}, {"a": 1.0})
        assert not json_value_equal({"x": True}, {"x": 1})
        assert json_value_equal({"x": 1}, {"x": 1.0})
        assert not json_value_equal([True], [1])

    def test_evaluator_final_state_bool_mismatch(self):
        # final state carries flag=1 (int), which must NOT satisfy a
        # boolean expectation — JSON leaf equality is type-strict.
        manifest = _manifest(dialogue=_harness_config())
        case = DialogueCaseSpec(
            **{
                "schema_version": SCHEMA_VERSION,
                "case_id": "dlg-001",
                "benchmark_version": "commercebench-dev-0.1",
                "task_family": "commerce_dialogue",
                "intent": "return",
                "difficulty": "medium",
                "answerability": "answerable",
                "turns": [
                    {
                        "turn_id": "t1",
                        "user_input": "u1",
                    }
                ],
                "initial_state": {"flag": 1, "orders": {}},
                "expected_final_state": {"flag": True},
            }
        )
        trace = DialogueHarness(config=manifest.dialogue).run(
            case,
            manifest,
            ScriptedDialogueSystem({"t1": [{"output_text": "r"}]}),
        )
        assert trace.termination_reason != TERMINATION_COMPLETED
        evaluation = DialogueEvaluator().evaluate(case, trace)
        assert evaluation.metrics["final_state_match"].passed is False


class TestCompletionDeltaAlignment:
    def test_incidental_delta_does_not_complete(self):
        # expected_state_delta already true at turn start: the turn did
        # not cause it, so neither completion nor coverage may credit it.
        case = DialogueCaseSpec(
            **{
                "schema_version": SCHEMA_VERSION,
                "case_id": "dlg-001",
                "benchmark_version": "commercebench-dev-0.1",
                "task_family": "commerce_dialogue",
                "intent": "return",
                "difficulty": "medium",
                "answerability": "answerable",
                "turns": [
                    {
                        "turn_id": "t1",
                        "user_input": "u1",
                        "expected_state_delta": {"a": {"b": 1}},
                    }
                ],
                "initial_state": {"a": {"b": 1}},
                "expected_final_state": {"a": {"b": 1}},
            }
        )
        manifest = _manifest(dialogue=_harness_config())
        trace = DialogueHarness(config=manifest.dialogue).run(
            case,
            manifest,
            ScriptedDialogueSystem({"t1": [{"output_text": "r"}]}),
        )
        # Harness and evaluator now share the newly-caused predicate.
        assert trace.termination_reason == TERMINATION_USER_DONE
        evaluation = DialogueEvaluator().evaluate(case, trace)
        metric = evaluation.metrics["state_delta_coverage"]
        assert metric.passed is False

    def test_newly_caused_delta_completes(self):
        case = DialogueCaseSpec(
            **{
                "schema_version": SCHEMA_VERSION,
                "case_id": "dlg-001",
                "benchmark_version": "commercebench-dev-0.1",
                "task_family": "commerce_dialogue",
                "intent": "return",
                "difficulty": "medium",
                "answerability": "answerable",
                "turns": [
                    {
                        "turn_id": "t1",
                        "user_input": "u1",
                        "expected_tool_calls": [
                            {
                                "tool_name": "request_return",
                                "arguments": {"order_id": "A1"},
                            }
                        ],
                        "expected_state_delta": {
                            "returns": {"A1": {"status": "requested"}}
                        },
                    }
                ],
                "initial_state": ORDER_STATE,
                "expected_final_state": {
                    "returns": {"A1": {"status": "requested"}}
                },
            }
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
        evaluation = DialogueEvaluator().evaluate(case, trace)
        assert evaluation.metrics["state_delta_coverage"].value == 1.0


class TestDegenerateCaseRejected:
    def test_empty_obligation_case_rejected(self):
        with pytest.raises(ContractValidationError):
            DialogueCaseSpec(
                **{
                    "schema_version": SCHEMA_VERSION,
                    "case_id": "dlg-x",
                    "benchmark_version": "commercebench-dev-0.1",
                    "task_family": "commerce_dialogue",
                    "intent": "return",
                    "difficulty": "medium",
                    "answerability": "answerable",
                    "turns": [{"turn_id": "t1", "user_input": "u1"}],
                    "initial_state": {},
                    "expected_final_state": {},
                }
            )

    def test_tool_obligation_makes_case_legal(self):
        case = DialogueCaseSpec(
            **{
                "schema_version": SCHEMA_VERSION,
                "case_id": "dlg-x",
                "benchmark_version": "commercebench-dev-0.1",
                "task_family": "commerce_dialogue",
                "intent": "return",
                "difficulty": "medium",
                "answerability": "answerable",
                "turns": [
                    {
                        "turn_id": "t1",
                        "user_input": "u1",
                        "expected_tool_calls": [
                            {"tool_name": "lookup_order"}
                        ],
                    }
                ],
                "initial_state": {},
                "expected_final_state": {},
            }
        )
        assert case.case_id == "dlg-x"


class TestActivationOrdering:
    def test_default_turn_leapfrogs_unsatisfied_gate(self):
        # Ordered-scan semantics: t2's default activation fires even
        # though earlier gated t1's condition is unmet — deterministic
        # but worth pinning as a regression.
        case = _case(
            turns=[
                _spec("t1", "gated", {"on_tool": "lookup_order"}),
                _spec("t2", "ungated", None),
            ],
            initial_state=ORDER_STATE,
        )
        script = {"t2": [{"output_text": "r"}]}
        manifest = _manifest(dialogue=_harness_config())
        trace = DialogueHarness(config=manifest.dialogue).run(
            case, manifest, ScriptedDialogueSystem(script)
        )
        assert [t.turn_id for t in trace.turns] == ["t2"]
        assert trace.termination_reason == TERMINATION_USER_DONE


class TestRuntimeMetadataEvidence:
    def test_identity_block_recorded(self):
        case = _case(turns=[_spec("t1", "hi", None)])
        manifest = _manifest(dialogue=_harness_config())
        trace = DialogueHarness(config=manifest.dialogue).run(
            case,
            manifest,
            ScriptedDialogueSystem({"t1": [{"output_text": "r"}]}),
        )
        identity = trace.runtime_metadata["identity"]
        assert identity["tool_schema"]["id"] == "commerce-tools-schema-v0"
        assert identity["tool_schema"]["hash"] == tool_schema_hash()
        assert "dialogue_model_identity" not in identity
