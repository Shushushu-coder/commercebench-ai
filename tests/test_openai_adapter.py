"""Phase 3E OpenAI Responses adapter unit tests (offline).

Covers the provider translation layer only: identity binding,
request rendering/determinism, response parsing, generation
parameters, provider error mapping, the scripted transport seam, and
credential hygiene. No network, no ``OPENAI_API_KEY``.
"""

from __future__ import annotations

import json

import pytest

from commercebench.contracts import ContractValidationError
from commercebench.contracts.dialogue import (
    DialogueHistoryEntry,
    DialogueTurnInput,
)
from commercebench.contracts.experiment import DialogueModelIdentity
from commercebench.contracts.trace import ToolEvent
from commercebench.dialogue.identity import (
    HISTORY_RENDERER_ID,
    HISTORY_RENDERER_VERSION,
    STATE_RENDERER_ID,
    STATE_RENDERER_VERSION,
    compute_prompt_hash,
)
from commercebench.dialogue.tools import (
    TOOL_DESCRIPTORS,
    TOOL_SCHEMA_ID,
    TOOL_SCHEMA_VERSION,
    tool_schema_hash,
)
from commercebench.providers.errors import (
    ProviderAuthenticationError,
    ProviderCredentialError,
    ProviderInvalidRequestError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderServerError,
    ProviderTimeoutError,
    ProviderTransportError,
)
from commercebench.providers.openai_responses import (
    OpenAIResponsesClient,
    ResponsesClientProtocol,
    ScriptedResponsesClient,
    map_provider_error,
)
from commercebench.systems.openai import (
    FORBIDDEN_SESSION_FIELDS,
    LIVE_MODEL_PLACEHOLDER,
    OpenAIDialogueSystem,
    SUPPORTED_GENERATION_PARAMETERS,
    build_responses_request,
    function_tool_param,
    parse_responses_payload,
    render_history_full_v0,
)

PROMPT_TEXT = "You are a commerce support agent. Be precise."
PROMPT_HASH = compute_prompt_hash(PROMPT_TEXT)
SCHEMA_HASH = tool_schema_hash()

MODEL = "test-model-0"


def _identity(**overrides):
    payload = {
        "provider": "openai",
        "model": MODEL,
        "prompt_id": "openai-support-v0",
        "prompt_version": "0.1",
        "prompt_hash": PROMPT_HASH,
        "history_renderer_id": HISTORY_RENDERER_ID,
        "history_renderer_version": HISTORY_RENDERER_VERSION,
        "state_renderer_id": STATE_RENDERER_ID,
        "state_renderer_version": STATE_RENDERER_VERSION,
        "tool_schema_id": TOOL_SCHEMA_ID,
        "tool_schema_version": TOOL_SCHEMA_VERSION,
        "tool_schema_hash": SCHEMA_HASH,
        "generation_parameters": {
            "max_output_tokens": 512,
            "temperature": 0.0,
            "top_p": 1.0,
        },
    }
    payload.update(overrides)
    return DialogueModelIdentity(**payload)


def _turn_input(user_input="u1", history=(), state=None, tools=None):
    return DialogueTurnInput(
        turn_id="t1",
        user_input=user_input,
        history=tuple(history),
        state_view=state if state is not None else {"orders": {}},
        available_tools=(
            tuple(TOOL_DESCRIPTORS) if tools is None else tuple(tools)
        ),
    )


def _system(client, **identity_overrides):
    return OpenAIDialogueSystem(
        identity=_identity(**identity_overrides),
        prompt_source=PROMPT_TEXT,
        client=client,
    )


def _text_payload(text, **overrides):
    payload = {
        "id": "resp_fake_1",
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
    payload.update(overrides)
    return payload


def _call_item(name, arguments, call_id="call_fake_1", **extra):
    item = {
        "type": "function_call",
        "id": "fc_fake_1",
        "call_id": call_id,
        "name": name,
        "arguments": (
            arguments if isinstance(arguments, str) else json.dumps(arguments)
        ),
        "status": "completed",
    }
    item.update(extra)
    return item


def _calls_payload(calls, text=None, **overrides):
    output = []
    if text is not None:
        output.append(
            {
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text}],
            }
        )
    output.extend(calls)
    payload = _text_payload("", **overrides)
    payload["output"] = output
    return payload


def _request_json(request):
    return json.dumps(request, ensure_ascii=False, sort_keys=True)


# ---------------------------------------------------------------------------
# Identity binding (fail closed before provider invocation)
# ---------------------------------------------------------------------------


class TestIdentityBinding:
    def test_valid_identity_accepted(self):
        system = _system(ScriptedResponsesClient(responses=[]))
        assert system.system_id == "openai-dialogue"
        assert system.system_version.startswith("0.1+openai-")

    def test_non_openai_provider_rejected(self):
        with pytest.raises(ContractValidationError):
            _system(
                ScriptedResponsesClient(responses=[]), provider="other"
            )

    def test_placeholder_model_rejected(self):
        with pytest.raises(ContractValidationError):
            _system(
                ScriptedResponsesClient(responses=[]),
                model=LIVE_MODEL_PLACEHOLDER,
            )

    def test_prompt_source_or_path_required(self):
        with pytest.raises(ContractValidationError):
            OpenAIDialogueSystem(
                identity=_identity(),
                client=ScriptedResponsesClient(responses=[]),
            )

    def test_prompt_hash_mismatch_rejected(self):
        with pytest.raises(ContractValidationError):
            _system(
                ScriptedResponsesClient(responses=[]),
                prompt_hash="0" * 64,
            )

    def test_prompt_text_mismatch_rejected(self):
        system_identity = _identity()
        with pytest.raises(ContractValidationError):
            OpenAIDialogueSystem(
                identity=system_identity,
                prompt_source="a different prompt entirely",
                client=ScriptedResponsesClient(responses=[]),
            )

    def test_tool_schema_hash_mismatch_rejected(self):
        with pytest.raises(ContractValidationError):
            _system(
                ScriptedResponsesClient(responses=[]),
                tool_schema_hash="0" * 64,
            )

    def test_renderer_version_mismatch_rejected(self):
        with pytest.raises(ContractValidationError):
            _system(
                ScriptedResponsesClient(responses=[]),
                history_renderer_version="9.9",
            )
        with pytest.raises(ContractValidationError):
            _system(
                ScriptedResponsesClient(responses=[]),
                state_renderer_version="9.9",
            )


class TestGenerationParameters:
    def test_supported_params_forwarded(self):
        client = ScriptedResponsesClient(responses=[_text_payload("ok")])
        system = _system(client)
        system.respond(_turn_input())
        request = client.requests[0]
        assert request["temperature"] == 0.0
        assert request["top_p"] == 1.0
        assert request["max_output_tokens"] == 512

    def test_unsupported_param_fails_closed(self):
        with pytest.raises(ContractValidationError):
            _system(
                ScriptedResponsesClient(responses=[]),
                generation_parameters={"temperature": 0.0, "seed": 7},
            )

    def test_all_whitelisted_params_forwarded(self):
        declared = {
            "temperature": 0.2,
            "top_p": 0.9,
            "max_output_tokens": 100,
            "max_tool_calls": 2,
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "truncation": "auto",
            "service_tier": "default",
            "top_logprobs": 0,
        }
        client = ScriptedResponsesClient(responses=[_text_payload("ok")])
        system = _system(client, generation_parameters=declared)
        system.respond(_turn_input())
        request = client.requests[0]
        for key, value in declared.items():
            assert request[key] == value
        assert set(declared) <= SUPPORTED_GENERATION_PARAMETERS


# ---------------------------------------------------------------------------
# Request rendering
# ---------------------------------------------------------------------------


class TestRequestRendering:
    def test_request_canonical_shape(self):
        client = ScriptedResponsesClient(responses=[_text_payload("ok")])
        system = _system(client)
        turn_input = _turn_input(
            user_input="我想退掉我最近的订单",
            history=[
                DialogueHistoryEntry(
                    role="user", content="我想退掉我最近的订单"
                )
            ],
            state={"orders": {"A1": {"status": "delivered"}}, "returns": {}},
        )
        system.respond(turn_input)
        request = client.requests[0]
        assert set(request) == {
            "model",
            "instructions",
            "input",
            "tools",
            "store",
            "temperature",
            "top_p",
            "max_output_tokens",
        }
        assert request["model"] == MODEL
        assert request["instructions"] == PROMPT_TEXT
        assert request["store"] is False
        assert isinstance(request["input"], list)
        assert isinstance(request["tools"], list)

    def test_request_deterministic(self):
        turn_input = _turn_input(
            history=[
                DialogueHistoryEntry(role="user", content="u1"),
                DialogueHistoryEntry(role="assistant", content="a1"),
            ],
            user_input="u1",
            state={"orders": {"A1": {"status": "delivered"}}},
        )
        identity = _identity()
        first = build_responses_request(turn_input, identity, PROMPT_TEXT)
        second = build_responses_request(turn_input, identity, PROMPT_TEXT)
        assert first == second
        assert _request_json(first) == _request_json(second)

    def test_state_block_first_and_canonical(self):
        request = build_responses_request(
            _turn_input(
                state={"b": 2, "a": 1},
                user_input="u",
                history=[DialogueHistoryEntry(role="user", content="u")],
            ),
            _identity(),
            PROMPT_TEXT,
        )
        state_item = request["input"][0]
        assert state_item["role"] == "developer"
        text = state_item["content"][0]["text"]
        assert "Canonical commerce state" in text
        assert '"a":1,"b":2' in text  # canonical key ordering

    def test_state_change_alters_request(self):
        base = build_responses_request(
            _turn_input(state={"orders": {}}), _identity(), PROMPT_TEXT
        )
        changed = build_responses_request(
            _turn_input(state={"orders": {"A1": {}}}),
            _identity(),
            PROMPT_TEXT,
        )
        assert base != changed

    def test_current_user_input_exactly_once(self):
        sentinel = "UNIQUE-USER-TEXT-9f3a"
        client = ScriptedResponsesClient(responses=[_text_payload("ok")])
        system = _system(client)
        system.respond(
            _turn_input(
                user_input=sentinel,
                history=[DialogueHistoryEntry(role="user", content=sentinel)],
            )
        )
        request = client.requests[0]
        assert _request_json(request).count(sentinel) == 1

    def test_current_user_appended_when_history_missing(self):
        request = build_responses_request(
            _turn_input(user_input="lonely input", history=[]),
            _identity(),
            PROMPT_TEXT,
        )
        last = request["input"][-1]
        assert last["role"] == "user"
        assert last["content"][0]["text"] == "lonely input"

    def test_no_hidden_session_fields(self):
        history = [
            DialogueHistoryEntry(role="user", content="u1"),
            DialogueHistoryEntry(
                role="assistant",
                content="",
                tool_events=(
                    ToolEvent(
                        tool_name="lookup_order",
                        arguments={"order_id": "A1"},
                    ),
                ),
            ),
            DialogueHistoryEntry(
                role="tool",
                content=[
                    {
                        "tool_name": "lookup_order",
                        "arguments": {"order_id": "A1"},
                        "result": {"status": "delivered"},
                        "status": "ok",
                        "error": None,
                    }
                ],
            ),
            DialogueHistoryEntry(role="user", content="u2"),
        ]
        request = build_responses_request(
            _turn_input(user_input="u2", history=history),
            _identity(),
            PROMPT_TEXT,
        )
        serialized = _request_json(request)
        for field in FORBIDDEN_SESSION_FIELDS:
            assert f'"{field}"' not in serialized
        for key in (
            "previous_response_id",
            "conversation",
            "conversation_id",
            "thread_id",
            "assistant_id",
            "prompt",
        ):
            assert key not in request

    def test_tools_rendered_from_descriptors(self):
        request = build_responses_request(
            _turn_input(), _identity(), PROMPT_TEXT
        )
        tools = request["tools"]
        assert [t["name"] for t in tools] == [
            "lookup_order",
            "request_return",
        ]
        for tool in tools:
            assert tool["type"] == "function"
            assert tool["strict"] is True
            params = tool["parameters"]
            assert params["type"] == "object"
            assert params["required"] == ["order_id"]
            assert params["additionalProperties"] is False
            assert params["properties"] == {"order_id": {"type": "string"}}
        # semantic source stays TOOL_DESCRIPTORS — projection only
        assert [d.name for d in TOOL_DESCRIPTORS] == [
            t["name"] for t in tools
        ]

    def test_function_tool_param_direct(self):
        descriptor = TOOL_DESCRIPTORS[0]
        param = function_tool_param(descriptor)
        assert param["name"] == descriptor.name
        assert param["description"] == descriptor.description


class TestHistoryRenderer:
    def test_roles_rendered_in_order(self):
        items = render_history_full_v0(
            (
                DialogueHistoryEntry(role="user", content="u1"),
                DialogueHistoryEntry(role="assistant", content="a1"),
                DialogueHistoryEntry(role="user", content="u2"),
            )
        )
        assert [i["role"] for i in items] == ["user", "assistant", "user"]
        assert items[1]["content"][0]["type"] == "output_text"
        assert items[0]["content"][0]["type"] == "input_text"

    def test_tool_calls_and_outputs_rendered(self):
        history = (
            DialogueHistoryEntry(
                role="assistant",
                content="",
                tool_events=(
                    ToolEvent(
                        tool_name="lookup_order",
                        arguments={"order_id": "A1"},
                    ),
                    ToolEvent(
                        tool_name="request_return",
                        arguments={"order_id": "A1"},
                    ),
                ),
            ),
            DialogueHistoryEntry(
                role="tool",
                content=[
                    {
                        "tool_name": "lookup_order",
                        "arguments": {"order_id": "A1"},
                        "result": {"status": "delivered"},
                        "status": "ok",
                        "error": None,
                    },
                    {
                        "tool_name": "request_return",
                        "arguments": {"order_id": "A1"},
                        "result": {"return_status": "requested"},
                        "status": "ok",
                        "error": None,
                    },
                ],
            ),
        )
        items = render_history_full_v0(history)
        calls = [i for i in items if i["type"] == "function_call"]
        outputs = [i for i in items if i["type"] == "function_call_output"]
        assert len(calls) == 2 and len(outputs) == 2
        assert calls[0]["name"] == "lookup_order"
        assert json.loads(calls[0]["arguments"]) == {"order_id": "A1"}
        # call ids pair deterministically call -> output in order
        assert outputs[0]["call_id"] == calls[0]["call_id"]
        assert outputs[1]["call_id"] == calls[1]["call_id"]
        result_payload = json.loads(outputs[0]["output"])
        assert result_payload["result"]["status"] == "delivered"
        assert result_payload["status"] == "ok"

    def test_error_tool_output_preserved(self):
        history = (
            DialogueHistoryEntry(
                role="assistant",
                content="",
                tool_events=(
                    ToolEvent(
                        tool_name="lookup_order",
                        arguments={"order_id": "GHOST"},
                    ),
                ),
            ),
            DialogueHistoryEntry(
                role="tool",
                content=[
                    {
                        "tool_name": "lookup_order",
                        "arguments": {"order_id": "GHOST"},
                        "result": None,
                        "status": "error",
                        "error": "ORDER_NOT_FOUND",
                    }
                ],
            ),
        )
        outputs = [
            i
            for i in render_history_full_v0(history)
            if i["type"] == "function_call_output"
        ]
        payload = json.loads(outputs[0]["output"])
        assert payload["status"] == "error"
        assert payload["error"] == "ORDER_NOT_FOUND"

    def test_renderer_deterministic(self):
        history = (
            DialogueHistoryEntry(role="user", content="u1"),
            DialogueHistoryEntry(
                role="assistant",
                content="a1",
                tool_events=(
                    ToolEvent(tool_name="lookup_order", arguments={"x": 1}),
                ),
            ),
        )
        assert render_history_full_v0(history) == render_history_full_v0(
            history
        )


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


class TestResponseParsing:
    def test_text_only(self):
        parsed = parse_responses_payload(
            _text_payload("你好"), requested_model=MODEL
        )
        assert parsed.output_text == "你好"
        assert parsed.tool_calls == ()
        assert parsed.usage.input_tokens == 10

    def test_tool_call_only(self):
        parsed = parse_responses_payload(
            _calls_payload(
                [_call_item("lookup_order", {"order_id": "A1"})]
            ),
            requested_model=MODEL,
        )
        assert parsed.output_text == ""
        assert len(parsed.tool_calls) == 1
        call = parsed.tool_calls[0]
        assert call.name == "lookup_order"
        assert call.arguments == {"order_id": "A1"}
        assert call.call_id == "call_fake_1"
        assert call.arguments_ok is True

    def test_text_plus_tool_calls(self):
        parsed = parse_responses_payload(
            _calls_payload(
                [_call_item("lookup_order", {"order_id": "A1"})],
                text="我先查一下",
            ),
            requested_model=MODEL,
        )
        assert parsed.output_text == "我先查一下"
        assert len(parsed.tool_calls) == 1

    def test_multiple_calls_preserve_order(self):
        parsed = parse_responses_payload(
            _calls_payload(
                [
                    _call_item("lookup_order", {"order_id": "A1"}, "c1"),
                    _call_item("request_return", {"order_id": "A1"}, "c2"),
                ]
            ),
            requested_model=MODEL,
        )
        assert [c.name for c in parsed.tool_calls] == [
            "lookup_order",
            "request_return",
        ]
        assert [c.call_id for c in parsed.tool_calls] == ["c1", "c2"]

    def test_malformed_arguments_is_safe_proposal(self):
        parsed = parse_responses_payload(
            _calls_payload([_call_item("lookup_order", "{not json")]),
            requested_model=MODEL,
        )
        call = parsed.tool_calls[0]
        assert call.arguments_ok is False
        assert call.arguments == {}

    def test_non_object_json_arguments_malformed(self):
        parsed = parse_responses_payload(
            _calls_payload([_call_item("lookup_order", '["A1"]')]),
            requested_model=MODEL,
        )
        assert parsed.tool_calls[0].arguments_ok is False

    def test_unknown_tool_name_passes_through(self):
        # Never executed here — proposal only; the simulator decides.
        parsed = parse_responses_payload(
            _calls_payload([_call_item("nuke_everything", {})]),
            requested_model=MODEL,
        )
        assert parsed.tool_calls[0].name == "nuke_everything"

    def test_empty_usable_output(self):
        parsed = parse_responses_payload(
            _text_payload("", output=[]), requested_model=MODEL
        )
        assert parsed.output_text == ""
        assert parsed.tool_calls == ()

    @pytest.mark.parametrize(
        "status", ["failed", "cancelled", "in_progress", "queued"]
    )
    def test_non_terminal_status_rejected(self, status):
        with pytest.raises(ProviderResponseError):
            parse_responses_payload(
                _text_payload("", status=status), requested_model=MODEL
            )

    def test_error_field_rejected(self):
        with pytest.raises(ProviderResponseError):
            parse_responses_payload(
                _text_payload(
                    "", error={"code": "x", "message": "boom"}
                ),
                requested_model=MODEL,
            )

    def test_non_mapping_payload_rejected(self):
        with pytest.raises(ProviderResponseError):
            parse_responses_payload(["not a dict"], requested_model=MODEL)

    def test_missing_output_rejected(self):
        payload = _text_payload("x")
        del payload["output"]
        with pytest.raises(ProviderResponseError):
            parse_responses_payload(payload, requested_model=MODEL)

    def test_incomplete_status_still_parsed(self):
        parsed = parse_responses_payload(
            _text_payload("partial", status="incomplete"),
            requested_model=MODEL,
        )
        assert parsed.output_text == "partial"

    def test_usage_mapped(self):
        parsed = parse_responses_payload(
            _text_payload("x"), requested_model=MODEL
        )
        assert parsed.usage.input_tokens == 10
        assert parsed.usage.output_tokens == 5
        assert parsed.usage.total_tokens == 15

    def test_usage_absent_not_fabricated(self):
        payload = _text_payload("x")
        del payload["usage"]
        parsed = parse_responses_payload(payload, requested_model=MODEL)
        assert parsed.usage.input_tokens is None
        assert parsed.usage.output_tokens is None
        assert parsed.usage.total_tokens is None

    def test_provider_metadata_safe_fields(self):
        parsed = parse_responses_payload(
            _calls_payload(
                [_call_item("lookup_order", {}, "call_e1")]
            ),
            requested_model=MODEL,
        )
        meta = parsed.provider_metadata
        assert meta["provider"] == "openai"
        assert meta["api_surface"] == "responses"
        assert meta["requested_model"] == MODEL
        assert meta["response_model"] == MODEL
        assert meta["response_id"] == "resp_fake_1"
        assert meta["response_status"] == "completed"
        assert meta["tool_calls"][0]["call_id"] == "call_e1"

    def test_provider_cannot_forge_tool_result(self):
        forged = _call_item(
            "request_return",
            {"order_id": "A1"},
            result={"forged": "success"},
            status="ok",
            error=None,
        )
        parsed = parse_responses_payload(
            _calls_payload([forged]), requested_model=MODEL
        )
        assert parsed.tool_calls[0].arguments == {"order_id": "A1"}
        assert "result" not in parsed.provider_metadata["tool_calls"][0]


class TestAdapterRespond:
    def test_end_to_end_turn_output(self):
        client = ScriptedResponsesClient(
            responses=[
                _calls_payload(
                    [_call_item("lookup_order", {"order_id": "A1"})],
                    text="查一下[[tag:clarify]]",
                )
            ]
        )
        output = _system(client).respond(_turn_input())
        assert output.output_text == "查一下"
        assert output.assistant_tags == ("clarify",)
        assert len(output.tool_calls) == 1
        assert output.tool_calls[0].tool_name == "lookup_order"
        assert output.tool_calls[0].result is None
        assert output.tool_calls[0].status is None
        assert output.usage.input_tokens == 10
        meta = output.runtime_metadata["provider"]
        assert meta["requested_model"] == MODEL
        assert "provider_request" in output.runtime_metadata
        # no retrieval/memory ever
        assert output.retrieval_events == ()
        assert output.memory_events == ()

    def test_malformed_arguments_become_empty_args_proposal(self):
        client = ScriptedResponsesClient(
            responses=[
                _calls_payload(
                    [_call_item("lookup_order", "{broken")]
                )
            ]
        )
        output = _system(client).respond(_turn_input())
        assert output.tool_calls[0].arguments == {}
        assert output.runtime_metadata["provider"]["malformed_arguments"]

    def test_offered_tools_mismatch_rejected(self):
        client = ScriptedResponsesClient(responses=[_text_payload("ok")])
        system = _system(client)
        with pytest.raises(ContractValidationError):
            system.respond(_turn_input(tools=[]))


class TestTagExtraction:
    def test_marker_stripped_into_tags(self):
        client = ScriptedResponsesClient(
            responses=[_text_payload("请提供订单号[[tag:clarify]]")]
        )
        output = _system(client).respond(_turn_input())
        assert output.assistant_tags == ("clarify",)
        assert output.output_text == "请提供订单号"
        assert "[[tag:" not in output.output_text

    def test_multiple_markers(self):
        client = ScriptedResponsesClient(
            responses=[_text_payload("a[[tag:x]]b[[tag:y]]")]
        )
        output = _system(client).respond(_turn_input())
        assert output.assistant_tags == ("x", "y")


# ---------------------------------------------------------------------------
# Scripted transport seam
# ---------------------------------------------------------------------------


class TestScriptedClient:
    def test_protocol_conformance(self):
        client = ScriptedResponsesClient(responses=[])
        assert isinstance(client, ResponsesClientProtocol)

    def test_captures_exact_request(self):
        client = ScriptedResponsesClient(responses=[_text_payload("ok")])
        request = {"model": "m", "input": []}
        client.create_response(request)
        assert client.requests[0] == request

    def test_queue_exhaustion_raises(self):
        client = ScriptedResponsesClient(responses=[])
        with pytest.raises(ProviderResponseError):
            client.create_response({})

    def test_handler_mode(self):
        client = ScriptedResponsesClient(
            handler=lambda req: _text_payload("handled")
        )
        payload = client.create_response({"x": 1})
        assert payload["output"][0]["content"][0]["text"] == "handled"

    def test_exactly_one_mode_required(self):
        with pytest.raises(ContractValidationError):
            ScriptedResponsesClient()
        with pytest.raises(ContractValidationError):
            ScriptedResponsesClient(
                responses=[], handler=lambda r: {}
            )


# ---------------------------------------------------------------------------
# Live transport boundary (no real calls)
# ---------------------------------------------------------------------------


class TestLiveClientBoundary:
    def test_missing_credential(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        client = OpenAIResponsesClient()
        with pytest.raises(ProviderCredentialError) as exc:
            client.create_response({"model": "m"})
        assert "credential missing" in str(exc.value)

    def test_explicit_key_constructs_client(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        client = OpenAIResponsesClient(api_key="sk-TEST-SECRET")
        sdk_client = client._resolve_client()
        assert sdk_client is not None
        # the key is held inside the SDK client only — our wrapper
        # never echoes it into diagnostics
        assert "sk-TEST-SECRET" not in repr(client)

    def test_error_mapping(self):
        httpx2 = pytest.importorskip("httpx2")
        openai = pytest.importorskip("openai")

        def _response(status):
            return httpx2.Response(
                status, request=httpx2.Request("POST", "http://x")
            )

        cases = [
            (
                openai.AuthenticationError(
                    "bad", response=_response(401), body=None
                ),
                ProviderAuthenticationError,
            ),
            (
                openai.PermissionDeniedError(
                    "no", response=_response(403), body=None
                ),
                ProviderAuthenticationError,
            ),
            (
                openai.RateLimitError(
                    "rl", response=_response(429), body=None
                ),
                ProviderRateLimitError,
            ),
            (
                openai.APITimeoutError(
                    request=httpx2.Request("POST", "http://x")
                ),
                ProviderTimeoutError,
            ),
            (
                openai.APIConnectionError(
                    request=httpx2.Request("POST", "http://x")
                ),
                ProviderTransportError,
            ),
            (
                openai.InternalServerError(
                    "ise", response=_response(503), body=None
                ),
                ProviderServerError,
            ),
            (
                openai.BadRequestError(
                    "bad", response=_response(400), body=None
                ),
                ProviderInvalidRequestError,
            ),
            (
                openai.NotFoundError(
                    "nf", response=_response(404), body=None
                ),
                ProviderInvalidRequestError,
            ),
        ]
        for exc, expected in cases:
            mapped = map_provider_error(exc)
            assert isinstance(mapped, expected), (
                f"{exc!r} -> {type(mapped).__name__}"
            )

    def test_unknown_error_maps_to_base(self):
        mapped = map_provider_error(RuntimeError("mystery"))
        assert type(mapped).__name__ == "ProviderError"
