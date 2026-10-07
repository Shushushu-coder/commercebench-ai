"""OpenAI Responses dialogue adapter (Phase 3E).

``OpenAIDialogueSystem`` is a thin provider-translation layer behind
the existing ``DialogueSystemUnderTest`` contract. It may validate
the declared ``DialogueModelIdentity``, render a request, send it
through an injected Responses client, and parse the provider payload
into text + tool proposals + usage. It may NOT mutate canonical
state, execute tools, select user turns, evaluate outputs, own
conversation state, or override harness policy — the
``DialogueHarness`` remains the sole owner of all of those.

Identity binding (Phase 3D contract, reused — no parallel identity
system): construction validates ``provider == "openai"``, runs
``validate_dialogue_identity`` (registered renderer ids/versions,
``commerce-tools-schema-v0`` hash, exact ``prompt_hash`` of the
prompt text actually sent), and checks every declared
``generation_parameters`` key against a closed whitelist of
Responses result-affecting fields — unsupported keys fail closed,
nothing is silently stripped. There is no default model and no
hidden model fallback: ``identity.model`` is sent verbatim.

No hidden provider conversation state: every request is built only
from the current ``DialogueTurnInput`` plus the bound identity and
prompt. ``previous_response_id``, ``conversation``, ``prompt``
(stored-prompt references), threads, and assistants are never
populated; ``store=False`` is always sent so the provider keeps no
server-side copy to chain against. Provider ``call_id`` values are
recorded as evidence only — follow-up requests re-derive them
deterministically from executed call content, so a persisted trace
plus the renderer identity fully reconstructs every request.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from commercebench.contracts.common import canonical_json
from commercebench.contracts.dialogue import (
    DialogueHistoryEntry,
    DialogueToolDescriptor,
    DialogueTurnInput,
    DialogueTurnOutput,
)
from commercebench.contracts.errors import ContractValidationError
from commercebench.contracts.experiment import DialogueModelIdentity
from commercebench.contracts.trace import ToolEvent, UsageStats
from commercebench.dialogue.identity import (
    HISTORY_RENDERER_ID,
    HISTORY_RENDERER_VERSION,
    STATE_RENDERER_ID,
    STATE_RENDERER_VERSION,
    compute_prompt_hash,
    validate_dialogue_identity,
)
from commercebench.dialogue.state import json_value_copy
from commercebench.dialogue.tools import (
    TOOL_DESCRIPTORS,
    TOOL_SCHEMA_ID,
    TOOL_SCHEMA_VERSION,
    tool_schema_hash,
)
from commercebench.providers.errors import (
    ProviderResponseError,
)
from commercebench.providers.openai_responses import (
    OPENAI_API_SURFACE,
    OPENAI_PROVIDER_ID,
    OpenAIResponsesClient,
)

SYSTEM_ID = "openai-dialogue"
SYSTEM_BASE_VERSION = "0.1"

#: Checked-in baseline prompt identity (``examples/dialogue_v0/prompts``).
DEFAULT_PROMPT_ID = "openai-support-v0"
DEFAULT_PROMPT_VERSION = "0.1"
DEFAULT_PROMPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "dialogue_v0"
    / "prompts"
    / "openai_support_v0.txt"
)

#: Development manifests must not silently select a real model; the
#: live runner requires explicit injection before any call.
LIVE_MODEL_PLACEHOLDER = "__LIVE_MODEL_REQUIRED__"

#: Result-affecting ``responses.create`` fields the adapter forwards
#: from ``generation_parameters`` verbatim. Anything else fails
#: closed at construction — never silently stripped. Note ``seed``
#: is not part of this SDK's Responses surface.
SUPPORTED_GENERATION_PARAMETERS = frozenset(
    {
        "max_output_tokens",
        "max_tool_calls",
        "parallel_tool_calls",
        "reasoning",
        "service_tier",
        "temperature",
        "text",
        "tool_choice",
        "top_logprobs",
        "top_p",
        "truncation",
    }
)

#: Provider-native session/state fields that must never appear in a
#: rendered request (explicit CommerceBench history is authoritative).
FORBIDDEN_SESSION_FIELDS = frozenset(
    {
        "assistant_id",
        "conversation",
        "conversation_id",
        "previous_response_id",
        "prompt",
        "thread_id",
    }
)

STATE_BLOCK_HEADER = "Canonical commerce state (authoritative JSON):"

_ASSISTANT_TAG_RE = re.compile(r"\[\[tag:([A-Za-z0-9_-]+)\]\]")


# ---------------------------------------------------------------------------
# Registered renderers (Phase 3D identities, Phase 3E implementation)
# ---------------------------------------------------------------------------


def _message_item(
    role: str, text: str, *, output: bool = False
) -> Dict[str, Any]:
    content_type = "output_text" if output else "input_text"
    return {
        "type": "message",
        "role": role,
        "content": [{"type": content_type, "text": text}],
    }


def _entry_text(content: Any) -> str:
    return content if isinstance(content, str) else canonical_json(content)


def _call_key(tool_name: str, arguments: Any) -> str:
    return canonical_json({"name": tool_name, "arguments": arguments})


def _derive_call_id(tool_name: str, arguments: Any, occurrence: int) -> str:
    """Deterministic call id for replayed tool protocol items.

    Real provider ``call_id`` values are recorded as step evidence but
    never fed back — a request must be reconstructable from persisted
    CommerceBench history alone, so ids are derived from the executed
    call's content plus its occurrence index among identical calls.
    """
    digest = hashlib.sha256(
        canonical_json(
            {
                "arguments": arguments,
                "name": tool_name,
                "occurrence": occurrence,
            }
        ).encode("utf-8")
    ).hexdigest()
    return f"call_{digest[:24]}"


def _tool_output_text(event: Mapping[str, Any]) -> str:
    return canonical_json(
        {
            "error": event.get("error"),
            "result": event.get("result"),
            "status": event.get("status"),
        }
    )


def render_history_full_v0(
    history: Tuple[DialogueHistoryEntry, ...],
) -> List[Dict[str, Any]]:
    """Registered ``full-history-messages-v0`` renderer (v0.1).

    Faithful, ordered, non-summarizing rendering of the explicit
    harness history: ``user`` entries become user ``input_text``
    messages; ``assistant`` entries become assistant ``output_text``
    messages plus one ``function_call`` item per proposed tool event;
    ``tool`` entries become ``function_call_output`` items paired to
    their calls by deterministically derived call ids. No truncation,
    no paraphrase, no provider transcript retrieval.
    """
    items: List[Dict[str, Any]] = []
    call_occurrences: Dict[str, int] = {}
    output_occurrences: Dict[str, int] = {}
    for entry in history:
        if entry.role == "user":
            items.append(_message_item("user", _entry_text(entry.content)))
        elif entry.role == "assistant":
            text = _entry_text(entry.content)
            if text != "" or not entry.tool_events:
                items.append(
                    _message_item("assistant", text, output=True)
                )
            for event in entry.tool_events:
                key = _call_key(event.tool_name, event.arguments)
                occurrence = call_occurrences.get(key, 0)
                call_occurrences[key] = occurrence + 1
                items.append(
                    {
                        "type": "function_call",
                        "call_id": _derive_call_id(
                            event.tool_name, event.arguments, occurrence
                        ),
                        "name": event.tool_name,
                        "arguments": canonical_json(event.arguments),
                    }
                )
        elif entry.role == "tool":
            events = (
                entry.content if isinstance(entry.content, list) else [entry.content]
            )
            for raw in events:
                if isinstance(raw, Mapping) and "tool_name" in raw:
                    name = raw.get("tool_name")
                    arguments = raw.get("arguments", {})
                    key = _call_key(name, arguments)
                    occurrence = output_occurrences.get(key, 0)
                    output_occurrences[key] = occurrence + 1
                    call_id = _derive_call_id(name, arguments, occurrence)
                    output_text = _tool_output_text(raw)
                else:
                    key = canonical_json(raw)
                    occurrence = output_occurrences.get(key, 0)
                    output_occurrences[key] = occurrence + 1
                    call_id = _derive_call_id("tool", raw, occurrence)
                    output_text = canonical_json(raw)
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": output_text,
                    }
                )
        else:  # pragma: no cover — roles are contract-constrained
            raise ContractValidationError(
                f"unknown history role {entry.role!r}"
            )
    return items


def render_state_json_v0(state_view: Mapping[str, Any]) -> Dict[str, Any]:
    """Registered ``state-json-v0`` renderer (v0.1).

    The canonical state view as one deterministic JSON block under a
    developer-role message — stable key ordering via
    ``canonical_json``, never natural-language paraphrase.
    """
    return _message_item(
        "developer",
        f"{STATE_BLOCK_HEADER}\n{canonical_json(state_view)}",
    )


def function_tool_param(descriptor: DialogueToolDescriptor) -> Dict[str, Any]:
    """Transform one ``DialogueToolDescriptor`` into Responses
    function-tool shape. The semantic source stays
    ``TOOL_DESCRIPTORS`` — this is a wire-format projection only:
    declared argument names become required properties and unknown
    fields are rejected (``additionalProperties: false``,
    ``strict: true``).
    """
    properties = {
        name: {"type": arg_type}
        for name, arg_type in sorted(descriptor.argument_schema.items())
    }
    return {
        "type": "function",
        "name": descriptor.name,
        "description": descriptor.description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": sorted(descriptor.argument_schema),
            "additionalProperties": False,
        },
        "strict": True,
    }


# ---------------------------------------------------------------------------
# Request / response protocol
# ---------------------------------------------------------------------------


def build_responses_request(
    turn_input: DialogueTurnInput,
    identity: DialogueModelIdentity,
    prompt_source: str,
) -> Dict[str, Any]:
    """Render the deterministic provider request for one step.

    Layout: one developer state block, then the rendered explicit
    history (which already carries the current user input as its last
    user entry — appended only when absent so it appears exactly
    once), the function-tool surface, ``store=False``, and the
    declared generation parameters. No provider session fields.
    """
    unknown = set(identity.generation_parameters) - set(
        SUPPORTED_GENERATION_PARAMETERS
    )
    if unknown:
        raise ContractValidationError(
            "generation_parameters keys unsupported by the OpenAI "
            f"Responses adapter: {sorted(unknown)}; supported: "
            f"{sorted(SUPPORTED_GENERATION_PARAMETERS)}"
        )
    items: List[Dict[str, Any]] = [
        render_state_json_v0(turn_input.state_view)
    ]
    items.extend(render_history_full_v0(turn_input.history))
    if not (
        turn_input.history
        and turn_input.history[-1].role == "user"
        and turn_input.history[-1].content == turn_input.user_input
    ):
        items.append(_message_item("user", turn_input.user_input))
    request: Dict[str, Any] = {
        "model": identity.model,
        "instructions": prompt_source,
        "input": items,
        "tools": [
            function_tool_param(d) for d in turn_input.available_tools
        ],
        "store": False,
    }
    for key, value in identity.generation_parameters.items():
        request[key] = json_value_copy(value)
    return request


def _parse_tool_arguments(raw: Any) -> Tuple[Dict[str, Any], bool]:
    """Strict tool-argument parsing: a JSON object or nothing.

    Strings must decode to an object; existing mappings pass through.
    Anything else — invalid JSON, non-object JSON, non-string wire
    values — yields ``({}, False)`` with no repair, no eval, and no
    silent coercion. The caller decides how the failure surfaces.
    """
    if isinstance(raw, Mapping):
        return dict(raw), True
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}, False
        if isinstance(parsed, dict):
            return parsed, True
    return {}, False


@dataclass(frozen=True)
class ParsedToolCall:
    """One provider tool proposal: name/arguments only — the provider
    never controls result, status, error, or canonical state."""

    name: str
    arguments: Dict[str, Any]
    call_id: Optional[str]
    arguments_ok: bool


@dataclass(frozen=True)
class ParsedResponsesPayload:
    output_text: str
    tool_calls: Tuple[ParsedToolCall, ...]
    usage: UsageStats
    provider_metadata: Dict[str, Any]


def parse_responses_payload(
    payload: Mapping[str, Any],
    *,
    requested_model: str,
) -> ParsedResponsesPayload:
    """Normalize a Responses payload into adapter output.

    Terminal statuses only: ``completed``/``incomplete`` (plus absent
    status on synthetic payloads) parse normally; ``failed``,
    ``cancelled``, ``in_progress``, ``queued``, and non-null ``error``
    are provider failures, not model output. Malformed tool arguments
    stay a *proposal* with ``arguments_ok=False`` so the run continues
    through the deterministic invalid-tool path instead of crashing.
    """
    if not isinstance(payload, Mapping):
        raise ProviderResponseError(
            "provider payload must be a JSON mapping"
        )
    status = payload.get("status")
    if status in ("failed", "cancelled", "in_progress", "queued"):
        raise ProviderResponseError(
            f"provider response status {status!r} produced no usable "
            "output"
        )
    error = payload.get("error")
    if error:
        detail = error.get("message") if isinstance(error, Mapping) else error
        raise ProviderResponseError(
            f"provider response carries an error: {detail!r}"
        )
    output = payload.get("output")
    if not isinstance(output, list):
        raise ProviderResponseError(
            "provider payload 'output' must be a list"
        )

    message_texts: List[str] = []
    calls: List[ParsedToolCall] = []
    call_evidence: List[Dict[str, Any]] = []
    malformed_items: List[Dict[str, Any]] = []
    ignored_types = set()
    for item in output:
        if not isinstance(item, Mapping):
            ignored_types.add("<non-object>")
            continue
        item_type = item.get("type")
        if item_type == "message" and item.get("role") == "assistant":
            fragments: List[str] = []
            content = item.get("content")
            if isinstance(content, list):
                for part in content:
                    if (
                        isinstance(part, Mapping)
                        and part.get("type") == "output_text"
                        and isinstance(part.get("text"), str)
                    ):
                        fragments.append(part["text"])
            message_texts.append("".join(fragments))
        elif item_type == "function_call":
            name = item.get("name")
            if not isinstance(name, str) or not name:
                malformed_items.append(
                    {"type": "function_call", "reason": "missing_name"}
                )
                continue
            arguments, ok = _parse_tool_arguments(item.get("arguments"))
            call_id = item.get("call_id")
            calls.append(
                ParsedToolCall(
                    name=name,
                    arguments=arguments,
                    call_id=call_id if isinstance(call_id, str) else None,
                    arguments_ok=ok,
                )
            )
            call_evidence.append(
                {
                    "call_id": call_id if isinstance(call_id, str) else None,
                    "name": name,
                    "arguments_ok": ok,
                }
            )
        else:
            ignored_types.add(str(item_type))

    raw_usage = payload.get("usage")
    usage_map = raw_usage if isinstance(raw_usage, Mapping) else {}

    def _token(key: str) -> Optional[int]:
        value = usage_map.get(key)
        return value if isinstance(value, int) and not isinstance(
            value, bool
        ) else None

    provider_metadata: Dict[str, Any] = {
        "provider": OPENAI_PROVIDER_ID,
        "api_surface": OPENAI_API_SURFACE,
        "requested_model": requested_model,
        "response_id": payload.get("id"),
        "response_model": payload.get("model"),
        "response_status": status,
        "usage": json_value_copy(raw_usage) if raw_usage is not None else None,
        "tool_calls": call_evidence,
    }
    if ignored_types:
        provider_metadata["ignored_item_types"] = sorted(ignored_types)
    if malformed_items:
        provider_metadata["malformed_items"] = malformed_items
    return ParsedResponsesPayload(
        output_text="\n".join(message_texts),
        tool_calls=tuple(calls),
        usage=UsageStats(
            input_tokens=_token("input_tokens"),
            output_tokens=_token("output_tokens"),
            total_tokens=_token("total_tokens"),
        ),
        provider_metadata=provider_metadata,
    )


def _extract_assistant_tags(text: str) -> Tuple[str, Tuple[str, ...]]:
    """Split ``[[tag:name]]`` markers out of provider text.

    ``on_assistant_tag`` turn activation consumes structured
    ``assistant_tags``; the prompt instructs the model to emit these
    markers, and the parser strips them deterministically — no model
    wording is interpreted.
    """
    tags = tuple(_ASSISTANT_TAG_RE.findall(text))
    if not tags:
        return text, ()
    cleaned = _ASSISTANT_TAG_RE.sub("", text).strip()
    return cleaned, tags


# ---------------------------------------------------------------------------
# Prompt + identity helpers
# ---------------------------------------------------------------------------


def load_prompt_source(path: Optional[Any] = None) -> str:
    """Load a checked-in prompt template as canonical text.

    Line endings are LF-normalized so ``compute_prompt_hash`` of the
    returned text is stable across platform checkouts.
    """
    prompt_path = Path(path) if path is not None else DEFAULT_PROMPT_PATH
    text = prompt_path.read_text(encoding="utf-8")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def openai_dialogue_identity(
    *,
    model: str,
    generation_parameters: Optional[Mapping[str, Any]] = None,
) -> DialogueModelIdentity:
    """Build the ``DialogueModelIdentity`` bound to the checked-in
    baseline prompt and the runtime tool schema. ``model`` is required
    — there is no default and no fallback selection."""
    return DialogueModelIdentity(
        provider=OPENAI_PROVIDER_ID,
        model=model,
        prompt_id=DEFAULT_PROMPT_ID,
        prompt_version=DEFAULT_PROMPT_VERSION,
        prompt_hash=compute_prompt_hash(load_prompt_source()),
        history_renderer_id=HISTORY_RENDERER_ID,
        history_renderer_version=HISTORY_RENDERER_VERSION,
        state_renderer_id=STATE_RENDERER_ID,
        state_renderer_version=STATE_RENDERER_VERSION,
        tool_schema_id=TOOL_SCHEMA_ID,
        tool_schema_version=TOOL_SCHEMA_VERSION,
        tool_schema_hash=tool_schema_hash(),
        generation_parameters=dict(generation_parameters or {}),
    )


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class OpenAIDialogueSystem:
    """``DialogueSystemUnderTest`` backed by an injected Responses client.

    Construction performs every fail-closed binding check before any
    provider invocation: provider id, model placeholder, prompt hash,
    renderer versions, tool schema hash, and the generation-parameter
    whitelist. ``respond`` only renders/sends/parses — all state,
    history, tool, and policy authority stays with the harness.
    """

    def __init__(
        self,
        *,
        identity: DialogueModelIdentity,
        prompt_source: Optional[str] = None,
        prompt_path: Optional[Any] = None,
        client: Optional[Any] = None,
        system_id: str = SYSTEM_ID,
        system_version: str = SYSTEM_BASE_VERSION,
    ) -> None:
        if not isinstance(identity, DialogueModelIdentity):
            raise ContractValidationError(
                "identity must be a DialogueModelIdentity"
            )
        if identity.provider != OPENAI_PROVIDER_ID:
            raise ContractValidationError(
                f"identity.provider must be {OPENAI_PROVIDER_ID!r} for "
                f"the OpenAI adapter, got {identity.provider!r}"
            )
        if identity.model == LIVE_MODEL_PLACEHOLDER:
            raise ContractValidationError(
                "identity.model is the live-model placeholder: an "
                "explicit model is required (no default, no fallback)"
            )
        if (prompt_source is None) == (prompt_path is None):
            raise ContractValidationError(
                "exactly one of prompt_source / prompt_path is required"
            )
        source = (
            prompt_source
            if prompt_source is not None
            else load_prompt_source(prompt_path)
        )
        validate_dialogue_identity(identity, prompt_source=source)
        unknown = set(identity.generation_parameters) - set(
            SUPPORTED_GENERATION_PARAMETERS
        )
        if unknown:
            raise ContractValidationError(
                "generation_parameters keys unsupported by the OpenAI "
                f"Responses adapter: {sorted(unknown)}; supported: "
                f"{sorted(SUPPORTED_GENERATION_PARAMETERS)}"
            )
        self._identity = identity
        self._prompt_source = source
        self._client = client if client is not None else OpenAIResponsesClient()
        self.system_id = system_id
        digest = hashlib.sha256(
            canonical_json(identity.to_dict()).encode("utf-8")
        ).hexdigest()
        self.system_version = f"{system_version}+openai-{digest[:12]}"

    @property
    def identity(self) -> DialogueModelIdentity:
        return self._identity

    def respond(self, turn_input: DialogueTurnInput) -> DialogueTurnOutput:
        offered = [d.to_dict() for d in turn_input.available_tools]
        bound = [d.to_dict() for d in TOOL_DESCRIPTORS]
        if offered != bound:
            raise ContractValidationError(
                "offered tools do not match the identity-bound "
                "commerce-tools-schema-v0 descriptors"
            )
        request = build_responses_request(
            turn_input, self._identity, self._prompt_source
        )
        payload = self._client.create_response(request)
        parsed = parse_responses_payload(
            payload, requested_model=self._identity.model
        )
        output_text, tags = _extract_assistant_tags(parsed.output_text)
        tool_calls = tuple(
            ToolEvent(
                tool_name=call.name,
                arguments=json_value_copy(call.arguments),
            )
            for call in parsed.tool_calls
        )
        runtime_metadata: Dict[str, Any] = {
            "provider": parsed.provider_metadata,
            "provider_request": json_value_copy(request),
        }
        malformed = [c for c in parsed.tool_calls if not c.arguments_ok]
        if malformed:
            runtime_metadata["provider"]["malformed_arguments"] = [
                {"name": c.name, "call_id": c.call_id} for c in malformed
            ]
        return DialogueTurnOutput(
            output_text=output_text,
            tool_calls=tool_calls,
            assistant_tags=tags,
            usage=parsed.usage,
            runtime_metadata=runtime_metadata,
        )


__all__ = [
    "DEFAULT_PROMPT_ID",
    "DEFAULT_PROMPT_PATH",
    "DEFAULT_PROMPT_VERSION",
    "FORBIDDEN_SESSION_FIELDS",
    "LIVE_MODEL_PLACEHOLDER",
    "OPENAI_PROVIDER_ID",
    "STATE_BLOCK_HEADER",
    "SUPPORTED_GENERATION_PARAMETERS",
    "SYSTEM_BASE_VERSION",
    "SYSTEM_ID",
    "OpenAIDialogueSystem",
    "ParsedResponsesPayload",
    "ParsedToolCall",
    "build_responses_request",
    "function_tool_param",
    "load_prompt_source",
    "openai_dialogue_identity",
    "parse_responses_payload",
    "render_history_full_v0",
    "render_state_json_v0",
]
