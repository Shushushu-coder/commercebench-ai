"""Phase 3E fake-client end-to-end tests (offline).

The real ``OpenAIDialogueSystem`` adapter drives the real
``DialogueHarness`` / ``CommerceToolSimulator`` / ``DialogueEvaluator``
stack through a ``ScriptedResponsesClient`` that returns synthetic
Responses payloads. Goal: prove the provider path — request renderer,
tool loop, state ownership, policy, trace, replay — not model quality.
No network, no ``OPENAI_API_KEY``.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib

import pytest

from commercebench.contracts import SCHEMA_VERSION
from commercebench.contracts.common import canonical_json
from commercebench.contracts.dialogue import (
    DialogueCaseSpec,
    DialogueRunTrace,
    DialogueTurnSpec,
    TERMINATION_COMPLETED,
    TERMINATION_SYSTEM_ERROR,
)
from commercebench.contracts.experiment import (
    DialogueModelIdentity,
    ExperimentManifest,
)
from commercebench.dialogue.harness import run_dialogue_case
from commercebench.dialogue.identity import (
    HISTORY_RENDERER_ID,
    HISTORY_RENDERER_VERSION,
    STATE_RENDERER_ID,
    STATE_RENDERER_VERSION,
    compute_prompt_hash,
)
from commercebench.dialogue.policy import ERROR_POLICY_PRECONDITION_FAILED
from commercebench.dialogue.tools import (
    TOOL_SCHEMA_ID,
    TOOL_SCHEMA_VERSION,
    tool_schema_hash,
)
from commercebench.evaluation.dialogue import (
    DialogueEvaluator,
    FAILURE_INVALID_TOOL_CALL,
)
from commercebench.providers.errors import (
    ProviderTransportError,
)
from commercebench.providers.openai_responses import (
    ScriptedResponsesClient,
)
from commercebench.systems.openai import (
    SYSTEM_BASE_VERSION,
    SYSTEM_ID,
    OpenAIDialogueSystem,
)

DIALOGUE_DIR = (
    pathlib.Path(__file__).resolve().parents[1] / "examples" / "dialogue_v0"
)

PROMPT_TEXT = "You are a commerce support agent. Be precise."
MODEL = "test-model-0"


def _identity(**overrides):
    payload = {
        "provider": "openai",
        "model": MODEL,
        "prompt_id": "openai-support-v0",
        "prompt_version": "0.1",
        "prompt_hash": compute_prompt_hash(PROMPT_TEXT),
        "history_renderer_id": HISTORY_RENDERER_ID,
        "history_renderer_version": HISTORY_RENDERER_VERSION,
        "state_renderer_id": STATE_RENDERER_ID,
        "state_renderer_version": STATE_RENDERER_VERSION,
        "tool_schema_id": TOOL_SCHEMA_ID,
        "tool_schema_version": TOOL_SCHEMA_VERSION,
        "tool_schema_hash": tool_schema_hash(),
        "generation_parameters": {
            "max_output_tokens": 1024,
            "temperature": 0.0,
            "top_p": 1.0,
        },
    }
    payload.update(overrides)
    return DialogueModelIdentity(**payload)


def _base_manifest():
    return ExperimentManifest.from_json(
        (DIALOGUE_DIR / "experiments" / "dialogue_stateful_v0.json")
        .read_text(encoding="utf-8")
    )


def _manifest(**overrides):
    manifest = dataclasses.replace(
        _base_manifest(),
        system_id=SYSTEM_ID,
        system_version=SYSTEM_BASE_VERSION,
        dialogue_model=_identity(),
    )
    return dataclasses.replace(manifest, **overrides)


def _load_case(name):
    return DialogueCaseSpec.from_json(
        (DIALOGUE_DIR / "cases" / name).read_text(encoding="utf-8")
    )


def _system(client):
    return OpenAIDialogueSystem(
        identity=_identity(), prompt_source=PROMPT_TEXT, client=client
    )


def _text_payload(text):
    return {
        "id": "resp_fake",
        "object": "response",
        "status": "completed",
        "model": MODEL,
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text}],
            }
        ],
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
        },
    }


def _call_payload(name, arguments, call_id="call_fake"):
    return {
        "id": "resp_fake",
        "object": "response",
        "status": "completed",
        "model": MODEL,
        "output": [
            {
                "type": "function_call",
                "id": "fc_fake",
                "call_id": call_id,
                "name": name,
                "arguments": arguments
                if isinstance(arguments, str)
                else json.dumps(arguments),
                "status": "completed",
            }
        ],
        "usage": {
            "input_tokens": 12,
            "output_tokens": 4,
            "total_tokens": 16,
        },
    }


def _run(case_name, responses, manifest=None):
    client = ScriptedResponsesClient(responses=list(responses))
    trace = run_dialogue_case(
        _load_case(case_name), manifest or _manifest(), _system(client)
    )
    return client, trace


def _request_serialized(request):
    return canonical_json(request)


def _function_call_outputs(request):
    return [
        item
        for item in request["input"]
        if item.get("type") == "function_call_output"
    ]


def _function_calls(request):
    return [
        item for item in request["input"] if item.get("type") == "function_call"
    ]


# ---------------------------------------------------------------------------
# A/B/C adapter-path runs (scripted provider payloads, real harness)
# ---------------------------------------------------------------------------


class TestScenarioAAdapterPath:
    RESPONSES = [
        # t1: no order id -> clarify with tag marker
        _text_payload("请问您要退货的订单号是多少？[[tag:clarify]]"),
        # t2: order id given -> lookup
        _call_payload("lookup_order", {"order_id": "A1001"}, "call_a1"),
        # t2 follow-up: report state, no mutation claim yet
        _text_payload("订单A1001已签收，符合退货条件，请确认是否退货。"),
        # t3: confirm -> request_return
        _call_payload("request_return", {"order_id": "A1001"}, "call_a2"),
        _text_payload("已为订单A1001提交退货申请。"),
    ]

    def test_completed_and_evaluated(self):
        client, trace = _run("dialogue-return-001.json", self.RESPONSES)
        case = _load_case("dialogue-return-001.json")
        assert trace.termination_reason == TERMINATION_COMPLETED
        assert [t.turn_id for t in trace.turns] == ["t1", "t2", "t3"]
        assert trace.final_state["returns"]["A1001"]["status"] == "requested"
        evaluation = DialogueEvaluator().evaluate(case, trace)
        assert evaluation.passed is True
        assert evaluation.failure_categories == ()

    def test_five_independent_requests(self):
        client, _ = _run("dialogue-return-001.json", self.RESPONSES)
        assert len(client.requests) == 5
        for request in client.requests:
            serialized = _request_serialized(request)
            assert '"previous_response_id"' not in serialized
            assert '"conversation"' not in serialized
            assert request["model"] == MODEL
            assert request["store"] is False

    def test_tool_result_fed_back_via_explicit_history(self):
        # Step 24: the follow-up provider request after a tool call
        # must carry the executed tool result from CommerceBench
        # history — not response chaining.
        client, _ = _run("dialogue-return-001.json", self.RESPONSES)
        # requests[0]=t1 clarify, requests[1]=t2 step0 (lookup proposed),
        # requests[2]=t2 step1 (lookup result visible in history)
        second = client.requests[2]
        outputs = _function_call_outputs(second)
        assert len(outputs) == 1
        payload = json.loads(outputs[0]["output"])
        assert payload["status"] == "ok"
        assert payload["result"]["status"] == "delivered"
        # and the matching function_call precedes it in the same request
        calls = _function_calls(second)
        assert calls and calls[0]["call_id"] == outputs[0]["call_id"]

    def test_clarify_tag_drove_activation(self):
        _, trace = _run("dialogue-return-001.json", self.RESPONSES)
        assert trace.turns[1].activation_evidence == {
            "type": "assistant_tag",
            "value": "clarify",
        }
        # the marker was stripped from user-facing text
        step_texts = [
            s.output_text for t in trace.turns for s in t.steps
        ]
        assert "[[tag:" not in "".join(step_texts)

    def test_usage_aggregated(self):
        _, trace = _run("dialogue-return-001.json", self.RESPONSES)
        # 3 text calls (10/5/15) + 2 call calls (12/4/16)
        assert trace.usage.input_tokens == 3 * 10 + 2 * 12
        assert trace.usage.total_tokens == 3 * 15 + 2 * 16


class TestScenarioBAdapterPath:
    RESPONSES = [
        _text_payload("请提供要查询的订单号。[[tag:clarify]]"),
        _call_payload("lookup_order", {"order_id": "B2001"}, "call_b1"),
        _text_payload("订单B2001尚未签收，暂不符合退货条件。"),
        _call_payload("lookup_order", {"order_id": "B2002"}, "call_b2"),
        _call_payload("request_return", {"order_id": "B2002"}, "call_b3"),
        _text_payload("订单B2002已提交退货申请。"),
    ]

    def test_eligible_order_only(self):
        client, trace = _run(
            "dialogue-order-status-001.json", self.RESPONSES
        )
        case = _load_case("dialogue-order-status-001.json")
        assert trace.termination_reason == TERMINATION_COMPLETED
        assert trace.final_state["returns"]["B2002"]["status"] == "requested"
        assert "B2001" not in trace.final_state["returns"]
        evaluation = DialogueEvaluator().evaluate(case, trace)
        assert evaluation.passed is True
        assert len(client.requests) == 6


class TestScenarioCAdapterPath:
    RESPONSES = [
        _call_payload("lookup_order", {"order_id": "C3001"}, "call_c1"),
        _text_payload("订单C3001已签收，确认为其办理退货吗？"),
        _call_payload("request_return", {"order_id": "C3001"}, "call_c2"),
        _text_payload("已为订单C3001提交退货申请。"),
    ]

    def test_correct_entity(self):
        client, trace = _run("dialogue-entity-001.json", self.RESPONSES)
        case = _load_case("dialogue-entity-001.json")
        assert trace.termination_reason == TERMINATION_COMPLETED
        assert trace.final_state["returns"]["C3001"]["status"] == "requested"
        assert "C3002" not in trace.final_state["returns"]
        evaluation = DialogueEvaluator().evaluate(case, trace)
        assert evaluation.passed is True
        assert len(client.requests) == 4


# ---------------------------------------------------------------------------
# Tool semantics through the adapter
# ---------------------------------------------------------------------------


class TestToolSemantics:
    def _single_turn_case(self, **kw):
        return DialogueCaseSpec(
            schema_version=SCHEMA_VERSION,
            case_id="dlg-openai-x",
            benchmark_version="commercebench-dev-0.1",
            task_family="commerce_dialogue",
            intent="return",
            difficulty="medium",
            answerability="answerable",
            turns=[
                DialogueTurnSpec(turn_id="t1", user_input="u1", **kw)
            ],
            initial_state={
                "orders": {"A1": {"status": "delivered", "returnable": True}},
                "returns": {},
            },
            expected_final_state={"_goal": "unmet"},
        )

    def test_policy_precondition_through_adapter(self):
        # lookup-before-mutate-v0: provider requests request_return
        # without a prior successful lookup -> rejected, state intact.
        client = ScriptedResponsesClient(
            responses=[
                _call_payload("request_return", {"order_id": "A1"}),
                _text_payload("抱歉，暂时无法处理。"),
            ]
        )
        case = self._single_turn_case()
        trace = run_dialogue_case(
            case, _manifest(case_ids=("dlg-openai-x",)), _system(client)
        )
        event = trace.turns[0].tool_events[0]
        assert event.status == "error"
        assert event.error == ERROR_POLICY_PRECONDITION_FAILED
        assert trace.final_state["returns"] == {}
        assert trace.turns[0].state_after == trace.turns[0].state_before

    def test_unknown_tool_safe_path(self):
        client = ScriptedResponsesClient(
            responses=[
                _call_payload("drop_all_tables", {}),
                _text_payload("done"),
            ]
        )
        case = self._single_turn_case()
        trace = run_dialogue_case(
            case, _manifest(case_ids=("dlg-openai-x",)), _system(client)
        )
        event = trace.turns[0].tool_events[0]
        assert event.status == "error"
        assert event.error == "UNKNOWN_TOOL"
        assert trace.final_state["returns"] == {}

    def test_malformed_arguments_invalid_tool_path(self):
        client = ScriptedResponsesClient(
            responses=[
                _call_payload("lookup_order", "{malformed json"),
                _text_payload("done"),
            ]
        )
        case = self._single_turn_case()
        trace = run_dialogue_case(
            case, _manifest(case_ids=("dlg-openai-x",)), _system(client)
        )
        event = trace.turns[0].tool_events[0]
        assert event.status == "error"
        assert event.error == "BAD_ARGUMENTS"
        # malformed parse evidence kept in step metadata
        step_meta = trace.turns[0].steps[0].runtime_metadata
        assert step_meta["provider"]["malformed_arguments"]
        evaluation = DialogueEvaluator().evaluate(case, trace)
        assert FAILURE_INVALID_TOOL_CALL in evaluation.failure_categories

    def test_multiple_calls_ordered(self):
        client = ScriptedResponsesClient(
            responses=[
                {
                    "id": "resp_multi",
                    "object": "response",
                    "status": "completed",
                    "model": MODEL,
                    "output": [
                        {
                            "type": "function_call",
                            "id": "fc1",
                            "call_id": "call_1",
                            "name": "lookup_order",
                            "arguments": '{"order_id": "A1"}',
                            "status": "completed",
                        },
                        {
                            "type": "function_call",
                            "id": "fc2",
                            "call_id": "call_2",
                            "name": "request_return",
                            "arguments": '{"order_id": "A1"}',
                            "status": "completed",
                        },
                    ],
                    "usage": {
                        "input_tokens": 1,
                        "output_tokens": 1,
                        "total_tokens": 2,
                    },
                },
                _text_payload("done"),
            ]
        )
        case = self._single_turn_case(
            expected_state_delta={
                "returns": {"A1": {"status": "requested"}}
            }
        )
        manifest = _manifest(case_ids=("dlg-openai-x",))
        # direct policy would also work; keep manifest's gated policy —
        # same-step lookup unlocks the mutate.
        trace = run_dialogue_case(case, manifest, _system(client))
        events = trace.turns[0].tool_events
        assert [e.tool_name for e in events] == [
            "lookup_order",
            "request_return",
        ]
        assert all(e.status == "ok" for e in events)
        assert trace.final_state["returns"]["A1"]["status"] == "requested"

    def test_text_and_tools_preserved_in_step(self):
        client = ScriptedResponsesClient(
            responses=[
                {
                    **_call_payload("lookup_order", {"order_id": "A1"}),
                    "output": [
                        {
                            "type": "message",
                            "role": "assistant",
                            "status": "completed",
                            "content": [
                                {"type": "output_text", "text": "先查询"}
                            ],
                        },
                        {
                            "type": "function_call",
                            "id": "fc1",
                            "call_id": "call_1",
                            "name": "lookup_order",
                            "arguments": '{"order_id": "A1"}',
                            "status": "completed",
                        },
                    ],
                },
                _text_payload("done"),
            ]
        )
        case = self._single_turn_case()
        trace = run_dialogue_case(
            case, _manifest(case_ids=("dlg-openai-x",)), _system(client)
        )
        step = trace.turns[0].steps[0]
        assert step.output_text == "先查询"
        assert len(step.tool_events) == 1
        assert step.tool_events[0].tool_name == "lookup_order"

    def test_forged_tool_result_ignored(self):
        forged_call = _call_payload("lookup_order", {"order_id": "A1"})
        forged_call["output"][0]["result"] = {"forged": "success"}
        forged_call["output"][0]["status"] = "ok"
        client = ScriptedResponsesClient(
            responses=[forged_call, _text_payload("done")]
        )
        case = self._single_turn_case()
        trace = run_dialogue_case(
            case, _manifest(case_ids=("dlg-openai-x",)), _system(client)
        )
        event = trace.turns[0].tool_events[0]
        assert event.status == "ok"
        # the recorded result is the simulator's, never the provider's
        assert event.result == {"status": "delivered", "returnable": True}
        assert "forged" not in canonical_json(event.to_dict())


# ---------------------------------------------------------------------------
# Provider failure boundary inside a run
# ---------------------------------------------------------------------------


class TestProviderErrorBoundary:
    def test_provider_error_becomes_system_error(self):
        def _boom(request):
            raise ProviderTransportError("simulated network failure")

        client = ScriptedResponsesClient(handler=_boom)
        case = _load_case("dialogue-return-001.json")
        trace = run_dialogue_case(
            case, _manifest(), _system(client)
        )
        assert trace.termination_reason == TERMINATION_SYSTEM_ERROR
        assert trace.turns == ()
        # provider errors never enter the benchmark taxonomy fields
        assert "simulated network failure" not in trace.to_json()


# ---------------------------------------------------------------------------
# Leakage: ground truth and future turns
# ---------------------------------------------------------------------------


class TestLeakage:
    def _sentinel_case(self):
        return DialogueCaseSpec(
            schema_version=SCHEMA_VERSION,
            case_id="dlg-sentinel",
            benchmark_version="commercebench-dev-0.1",
            task_family="commerce_dialogue",
            intent="return",
            difficulty="medium",
            answerability="answerable",
            turns=[
                DialogueTurnSpec(
                    turn_id="t1",
                    user_input="u1",
                    required_facts=("SECRET_REQUIRED_FACT_SENTINEL",),
                    forbidden_claims=("SECRET_FORBIDDEN_CLAIM_SENTINEL",),
                    expected_tool_calls=(
                        {
                            "tool_name": "lookup_order",
                            "arguments": {
                                "order_id": "SECRET_EXPECTED_TOOL_SENTINEL"
                            },
                        },
                    ),
                    expected_state_delta={
                        "SECRET_DELTA_SENTINEL": {"status": "requested"}
                    },
                ),
                DialogueTurnSpec(
                    turn_id="t2",
                    user_input="SECRET_FUTURE_TURN_SENTINEL",
                ),
            ],
            initial_state={"orders": {}, "returns": {}},
            expected_final_state={"SECRET_EXPECTED_FINAL_STATE_SENTINEL": 1},
            required_facts=("SECRET_CASE_FACT_SENTINEL",),
            forbidden_claims=("SECRET_CASE_CLAIM_SENTINEL",),
            metadata={"SECRET_METADATA_SENTINEL": "x"},
        )

    SENTINELS = [
        "SECRET_EXPECTED_FINAL_STATE_SENTINEL",
        "SECRET_DELTA_SENTINEL",
        "SECRET_EXPECTED_TOOL_SENTINEL",
        "SECRET_REQUIRED_FACT_SENTINEL",
        "SECRET_FORBIDDEN_CLAIM_SENTINEL",
        "SECRET_CASE_FACT_SENTINEL",
        "SECRET_CASE_CLAIM_SENTINEL",
        "SECRET_METADATA_SENTINEL",
    ]

    def test_ground_truth_never_leaks_to_provider(self):
        client = ScriptedResponsesClient(
            responses=[_text_payload("r1"), _text_payload("r2")]
        )
        case = self._sentinel_case()
        run_dialogue_case(
            case, _manifest(case_ids=("dlg-sentinel",)), _system(client)
        )
        assert client.requests
        for request in client.requests:
            serialized = _request_serialized(request)
            for sentinel in self.SENTINELS:
                assert sentinel not in serialized, (
                    f"{sentinel} leaked into provider request"
                )

    def test_future_turn_never_leaks(self):
        client = ScriptedResponsesClient(
            responses=[_text_payload("r1"), _text_payload("r2")]
        )
        case = self._sentinel_case()
        run_dialogue_case(
            case, _manifest(case_ids=("dlg-sentinel",)), _system(client)
        )
        # first request (turn t1) must not contain t2's user text
        first = _request_serialized(client.requests[0])
        assert "SECRET_FUTURE_TURN_SENTINEL" not in first
        # the released turn itself does appear when it becomes current
        last = _request_serialized(client.requests[-1])
        assert "SECRET_FUTURE_TURN_SENTINEL" in last
        # ...exactly once
        assert last.count("SECRET_FUTURE_TURN_SENTINEL") == 1


# ---------------------------------------------------------------------------
# Security: credential hygiene through the real stack
# ---------------------------------------------------------------------------


class TestSecretHygiene:
    def test_env_key_never_leaks(self, monkeypatch):
        sentinel = "sk-TEST-SECRET-SHOULD-NEVER-APPEAR"
        monkeypatch.setenv("OPENAI_API_KEY", sentinel)
        client = ScriptedResponsesClient(
            responses=[
                _text_payload("订单号？[[tag:clarify]]"),
                _call_payload("lookup_order", {"order_id": "A1001"}),
                _text_payload("A1001已签收。"),
                _call_payload("request_return", {"order_id": "A1001"}),
                _text_payload("已提交。"),
            ]
        )
        case = _load_case("dialogue-return-001.json")
        trace = run_dialogue_case(case, _manifest(), _system(client))
        trace_json = trace.to_json()
        assert sentinel not in trace_json
        for request in client.requests:
            assert sentinel not in _request_serialized(request)
        step_meta = canonical_json(
            [
                s.runtime_metadata
                for t in trace.turns
                for s in t.steps
            ]
        )
        assert sentinel not in step_meta
        assert "OPENAI_API_KEY" not in trace_json


# ---------------------------------------------------------------------------
# Offline replay + evidence reconstructability
# ---------------------------------------------------------------------------


class TestOfflineReplay:
    def test_trace_roundtrip_reproduces_evaluation(self):
        client, trace = _run(
            "dialogue-return-001.json",
            TestScenarioAAdapterPath.RESPONSES,
        )
        case = _load_case("dialogue-return-001.json")
        live = DialogueEvaluator().evaluate(case, trace)

        restored = DialogueRunTrace.from_json(trace.to_json())
        assert restored == trace
        replayed = DialogueEvaluator().evaluate(case, restored)
        assert replayed.metrics.keys() == live.metrics.keys()
        for name in live.metrics:
            assert replayed.metrics[name].value == live.metrics[name].value
            assert replayed.metrics[name].passed == live.metrics[name].passed
        assert replayed.passed == live.passed
        assert replayed.failure_categories == live.failure_categories

    def test_persisted_requests_match_captured(self):
        # provider_request in step metadata = the exact payload the
        # transport saw; persisted trace alone reproduces it.
        client, trace = _run(
            "dialogue-entity-001.json",
            TestScenarioCAdapterPath.RESPONSES,
        )
        step_requests = [
            s.runtime_metadata["provider_request"]
            for t in trace.turns
            for s in t.steps
        ]
        assert len(step_requests) == len(client.requests)
        for recorded, captured in zip(step_requests, client.requests):
            assert canonical_json(recorded) == canonical_json(captured)

    def test_identity_evidence_in_trace(self):
        _, trace = _run(
            "dialogue-entity-001.json",
            TestScenarioCAdapterPath.RESPONSES,
        )
        identity_meta = trace.runtime_metadata["identity"]
        declared = identity_meta["dialogue_model_identity"]
        assert declared["provider"] == "openai"
        assert declared["model"] == MODEL
        assert identity_meta["tool_schema"]["hash"] == tool_schema_hash()
        provider_meta = trace.turns[0].steps[0].runtime_metadata["provider"]
        assert provider_meta["provider"] == "openai"
        assert provider_meta["api_surface"] == "responses"
        assert provider_meta["response_id"] == "resp_fake"
