"""Phase 3B dialogue contracts: case, turn spec, I/O, and traces.

These types are deliberately separate from the single-turn V0
contracts. ``CaseSpec.user_input`` / ``RunTrace.input_text`` /
``output_text`` are required scalar invariants that a multi-turn
dialogue cannot fill honestly, so dialogue gets its own contract
family sharing the same field conventions, validation helpers, and
``schema_version`` ("0.1" per contract type).

Ownership model (fixed by Phase 3A audit): the dialogue harness owns
the canonical business state and the visible history; the system
under test only *proposes* tool calls; the deterministic tool
simulator is the only authority that can change state.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from .case import Answerability, Difficulty
from .common import (
    canonical_json,
    dumps_dict,
    ensure_json_compatible,
    expect_mapping,
    parse_enum,
    require_int,
    require_iso8601,
    require_json_dict,
    require_non_empty_str,
    require_non_negative_number,
    require_str,
    str_tuple,
)
from .errors import ContractValidationError
from .trace import (
    MemoryEvent,
    RetrievalEvent,
    ToolEvent,
    UsageStats,
    _event_tuple,
)

# ---------------------------------------------------------------------------
# Constrained vocabularies (fail closed — no arbitrary values)
# ---------------------------------------------------------------------------

TERMINATION_COMPLETED = "completed"
TERMINATION_USER_DONE = "user_done"
TERMINATION_MAX_TURNS = "max_turns"
TERMINATION_TOOL_LIMIT = "tool_limit"
TERMINATION_INVALID_TOOL = "invalid_tool"
TERMINATION_SYSTEM_ERROR = "system_error"

TERMINATION_REASONS: Tuple[str, ...] = (
    TERMINATION_COMPLETED,
    TERMINATION_USER_DONE,
    TERMINATION_MAX_TURNS,
    TERMINATION_TOOL_LIMIT,
    TERMINATION_INVALID_TOOL,
    TERMINATION_SYSTEM_ERROR,
)

HISTORY_ROLES: Tuple[str, ...] = ("user", "assistant", "tool")

_ACTIVATION_KEYS: Tuple[str, ...] = ("on_tool", "on_assistant_tag")


def _require_termination_reason(value: Any, field_name: str) -> str:
    text = require_non_empty_str(value, field_name)
    if text not in TERMINATION_REASONS:
        raise ContractValidationError(
            f"{field_name} must be one of {list(TERMINATION_REASONS)}, "
            f"got {text!r}"
        )
    return text


def _parse_activation(value: Any) -> Optional[Dict[str, Any]]:
    """Validate a turn activation: ``None``/``{}`` = default trigger.

    Exactly one trigger key is allowed: ``on_tool`` (fires after that
    tool executed with ``status="ok"``) or ``on_assistant_tag`` (fires
    after a prior assistant step emitted that tag). No nested boolean
    DSL in Phase 3B.
    """
    if value is None:
        return None
    mapping = require_json_dict(value, "activation")
    if not mapping:
        return {}
    unknown = set(mapping) - set(_ACTIVATION_KEYS)
    if unknown:
        raise ContractValidationError(
            f"activation has unknown keys {sorted(unknown)}; "
            f"allowed: {list(_ACTIVATION_KEYS)}"
        )
    if len(mapping) != 1:
        raise ContractValidationError(
            "activation allows exactly one trigger key in Phase 3B"
        )
    for key in _ACTIVATION_KEYS:
        if key in mapping:
            mapping[key] = require_non_empty_str(mapping[key], f"activation.{key}")
    return mapping


# ---------------------------------------------------------------------------
# Case-side contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExpectedToolCall:
    """One required tool call inside a dialogue turn.

    ``arguments=None`` means "tool name must match, any arguments";
    a present mapping requires exact JSON equality.
    """

    tool_name: str
    arguments: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "tool_name", require_non_empty_str(self.tool_name, "tool_name")
        )
        if self.arguments is not None:
            object.__setattr__(
                self,
                "arguments",
                dict(require_json_dict(self.arguments, "arguments")),
            )

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"tool_name": self.tool_name}
        if self.arguments is not None:
            payload["arguments"] = dict(self.arguments)
        return payload

    @classmethod
    def from_dict(cls, data: Any) -> "ExpectedToolCall":
        data = expect_mapping(data, "ExpectedToolCall")
        try:
            return cls(
                tool_name=data["tool_name"],
                arguments=(
                    dict(require_json_dict(data["arguments"], "arguments"))
                    if "arguments" in data and data["arguments"] is not None
                    else None
                ),
            )
        except KeyError as exc:
            raise ContractValidationError(
                f"ExpectedToolCall missing required field: {exc.args[0]}"
            ) from exc


@dataclass(frozen=True)
class DialogueTurnSpec:
    """One scripted user turn with optional release rule and ground truth.

    ``activation`` governs when this user message is released to the
    system (never early — the harness enforces no future-turn leakage).
    ``expected_*`` fields are per-turn evaluation anchors.
    """

    turn_id: str
    user_input: str

    activation: Optional[Dict[str, Any]] = None

    required_facts: Tuple[str, ...] = ()
    forbidden_claims: Tuple[str, ...] = ()

    expected_tool_calls: Tuple[ExpectedToolCall, ...] = ()
    expected_state_delta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "turn_id", require_non_empty_str(self.turn_id, "turn_id")
        )
        object.__setattr__(
            self,
            "user_input",
            require_non_empty_str(self.user_input, "user_input"),
        )
        object.__setattr__(
            self, "activation", _parse_activation(self.activation)
        )
        object.__setattr__(
            self,
            "required_facts",
            str_tuple(self.required_facts, "required_facts"),
        )
        object.__setattr__(
            self,
            "forbidden_claims",
            str_tuple(self.forbidden_claims, "forbidden_claims"),
        )
        calls = []
        for item in self.expected_tool_calls:
            if not isinstance(item, ExpectedToolCall):
                item = ExpectedToolCall.from_dict(item)
            calls.append(item)
        object.__setattr__(self, "expected_tool_calls", tuple(calls))
        object.__setattr__(
            self,
            "expected_state_delta",
            dict(
                require_json_dict(
                    self.expected_state_delta, "expected_state_delta"
                )
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "user_input": self.user_input,
            "activation": (
                dict(self.activation) if self.activation is not None else None
            ),
            "required_facts": list(self.required_facts),
            "forbidden_claims": list(self.forbidden_claims),
            "expected_tool_calls": [
                call.to_dict() for call in self.expected_tool_calls
            ],
            "expected_state_delta": dict(self.expected_state_delta),
        }

    @classmethod
    def from_dict(cls, data: Any) -> "DialogueTurnSpec":
        data = expect_mapping(data, "DialogueTurnSpec")
        try:
            return cls(
                turn_id=data["turn_id"],
                user_input=data["user_input"],
                activation=data.get("activation"),
                required_facts=data.get("required_facts", ()),
                forbidden_claims=data.get("forbidden_claims", ()),
                expected_tool_calls=data.get("expected_tool_calls", ()),
                expected_state_delta=data.get("expected_state_delta", {}),
            )
        except KeyError as exc:
            raise ContractValidationError(
                f"DialogueTurnSpec missing required field: {exc.args[0]}"
            ) from exc


@dataclass(frozen=True)
class DialogueCaseSpec:
    """One multi-turn benchmark case (Phase 3B contract family).

    Unlike ``CaseSpec`` there is no ``user_input``/``expected_response``:
    the user side is an ordered tuple of ``DialogueTurnSpec`` and the
    primary ground truth is ``expected_final_state`` (subset-matched).
    ``conversation_type`` is implied MULTI_TURN by the type itself.
    """

    schema_version: str
    case_id: str
    benchmark_version: str

    task_family: str
    intent: str
    difficulty: Difficulty
    answerability: Answerability

    turns: Tuple[DialogueTurnSpec, ...]

    initial_state: Dict[str, Any]
    expected_final_state: Dict[str, Any]

    required_facts: Tuple[str, ...] = ()
    forbidden_claims: Tuple[str, ...] = ()

    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "schema_version",
            require_non_empty_str(self.schema_version, "schema_version"),
        )
        object.__setattr__(
            self, "case_id", require_non_empty_str(self.case_id, "case_id")
        )
        object.__setattr__(
            self,
            "benchmark_version",
            require_non_empty_str(self.benchmark_version, "benchmark_version"),
        )
        object.__setattr__(
            self,
            "task_family",
            require_non_empty_str(self.task_family, "task_family"),
        )
        object.__setattr__(
            self, "intent", require_non_empty_str(self.intent, "intent"),
        )
        object.__setattr__(
            self,
            "difficulty",
            parse_enum(Difficulty, self.difficulty, "difficulty"),
        )
        object.__setattr__(
            self,
            "answerability",
            parse_enum(Answerability, self.answerability, "answerability"),
        )
        if not isinstance(self.turns, (list, tuple)) or not self.turns:
            raise ContractValidationError(
                "turns must be a non-empty list"
            )
        turns = []
        seen_ids = set()
        for item in self.turns:
            if not isinstance(item, DialogueTurnSpec):
                item = DialogueTurnSpec.from_dict(item)
            if item.turn_id in seen_ids:
                raise ContractValidationError(
                    f"duplicate turn_id {item.turn_id!r} in case"
                )
            seen_ids.add(item.turn_id)
            turns.append(item)
        object.__setattr__(self, "turns", tuple(turns))
        object.__setattr__(
            self,
            "initial_state",
            dict(require_json_dict(self.initial_state, "initial_state")),
        )
        object.__setattr__(
            self,
            "expected_final_state",
            dict(
                require_json_dict(
                    self.expected_final_state, "expected_final_state"
                )
            ),
        )
        object.__setattr__(
            self,
            "required_facts",
            str_tuple(self.required_facts, "required_facts"),
        )
        object.__setattr__(
            self,
            "forbidden_claims",
            str_tuple(self.forbidden_claims, "forbidden_claims"),
        )
        object.__setattr__(
            self, "metadata", dict(require_json_dict(self.metadata, "metadata"))
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "case_id": self.case_id,
            "benchmark_version": self.benchmark_version,
            "task_family": self.task_family,
            "intent": self.intent,
            "difficulty": self.difficulty.value,
            "answerability": self.answerability.value,
            "turns": [turn.to_dict() for turn in self.turns],
            "initial_state": dict(self.initial_state),
            "expected_final_state": dict(self.expected_final_state),
            "required_facts": list(self.required_facts),
            "forbidden_claims": list(self.forbidden_claims),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Any) -> "DialogueCaseSpec":
        data = expect_mapping(data, "DialogueCaseSpec")
        try:
            return cls(
                schema_version=data["schema_version"],
                case_id=data["case_id"],
                benchmark_version=data["benchmark_version"],
                task_family=data["task_family"],
                intent=data["intent"],
                difficulty=data["difficulty"],
                answerability=data["answerability"],
                turns=data["turns"],
                initial_state=data["initial_state"],
                expected_final_state=data["expected_final_state"],
                required_facts=data.get("required_facts", ()),
                forbidden_claims=data.get("forbidden_claims", ()),
                metadata=data.get("metadata", {}),
            )
        except KeyError as exc:
            raise ContractValidationError(
                f"DialogueCaseSpec missing required field: {exc.args[0]}"
            ) from exc

    def to_json(self) -> str:
        return dumps_dict(self.to_dict())

    @classmethod
    def from_json(cls, payload: str) -> "DialogueCaseSpec":
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ContractValidationError(
                f"DialogueCaseSpec invalid JSON: {exc}"
            ) from exc
        return cls.from_dict(data)


# ---------------------------------------------------------------------------
# System-facing I/O contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DialogueHistoryEntry:
    """One entry of the harness-constructed visible history.

    ``role`` is one of ``user`` / ``assistant`` / ``tool``. ``tool``
    entries expose executed tool results; ``assistant`` entries may
    carry the ``tool_events`` they proposed in that step.
    """

    role: str
    content: Any = ""
    tool_events: Tuple[ToolEvent, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "role", require_non_empty_str(self.role, "role")
        )
        if self.role not in HISTORY_ROLES:
            raise ContractValidationError(
                f"role must be one of {list(HISTORY_ROLES)}, got {self.role!r}"
            )
        ensure_json_compatible(self.content, "content")
        object.__setattr__(
            self,
            "tool_events",
            _event_tuple(self.tool_events, ToolEvent, "tool_events"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "tool_events": [event.to_dict() for event in self.tool_events],
        }

    @classmethod
    def from_dict(cls, data: Any) -> "DialogueHistoryEntry":
        data = expect_mapping(data, "DialogueHistoryEntry")
        try:
            return cls(
                role=data["role"],
                content=data.get("content", ""),
                tool_events=data.get("tool_events", ()),
            )
        except KeyError as exc:
            raise ContractValidationError(
                f"DialogueHistoryEntry missing required field: {exc.args[0]}"
            ) from exc


@dataclass(frozen=True)
class DialogueToolDescriptor:
    """Description of one tool the harness offers for a turn."""

    name: str
    description: str = ""
    argument_schema: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "name", require_non_empty_str(self.name, "name")
        )
        object.__setattr__(
            self, "description", require_str(self.description, "description")
        )
        object.__setattr__(
            self,
            "argument_schema",
            dict(require_json_dict(self.argument_schema, "argument_schema")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "argument_schema": dict(self.argument_schema),
        }

    @classmethod
    def from_dict(cls, data: Any) -> "DialogueToolDescriptor":
        data = expect_mapping(data, "DialogueToolDescriptor")
        try:
            return cls(
                name=data["name"],
                description=data.get("description", ""),
                argument_schema=data.get("argument_schema", {}),
            )
        except KeyError as exc:
            raise ContractValidationError(
                f"DialogueToolDescriptor missing required field: {exc.args[0]}"
            ) from exc


@dataclass(frozen=True)
class DialogueTurnInput:
    """The exact context one ``respond()`` invocation sees.

    The harness constructs this; the system MUST NOT rebuild history
    internally. ``state_view`` is a deep copy of the canonical state —
    mutating it has no effect. ``input_context_digest()`` covers the
    full context (prior history, current user input, state view,
    available tools, turn identity), not just the transcript.
    """

    turn_id: str
    user_input: str
    history: Tuple[DialogueHistoryEntry, ...] = ()
    state_view: Dict[str, Any] = field(default_factory=dict)
    available_tools: Tuple[DialogueToolDescriptor, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "turn_id", require_non_empty_str(self.turn_id, "turn_id")
        )
        object.__setattr__(
            self,
            "user_input",
            require_non_empty_str(self.user_input, "user_input"),
        )
        entries = []
        for item in self.history:
            if not isinstance(item, DialogueHistoryEntry):
                item = DialogueHistoryEntry.from_dict(item)
            entries.append(item)
        object.__setattr__(self, "history", tuple(entries))
        object.__setattr__(
            self,
            "state_view",
            dict(require_json_dict(self.state_view, "state_view")),
        )
        tools = []
        for item in self.available_tools:
            if not isinstance(item, DialogueToolDescriptor):
                item = DialogueToolDescriptor.from_dict(item)
            tools.append(item)
        object.__setattr__(self, "available_tools", tuple(tools))

    def context_dict(self) -> Dict[str, Any]:
        """Canonical payload of everything this invocation can see."""
        return {
            "turn_id": self.turn_id,
            "user_input": self.user_input,
            "history": [entry.to_dict() for entry in self.history],
            "state_view": self.state_view,
            "available_tools": [tool.to_dict() for tool in self.available_tools],
        }

    def input_context_digest(self) -> str:
        """SHA-256 of the canonical JSON of the full visible context."""
        return hashlib.sha256(
            canonical_json(self.context_dict()).encode("utf-8")
        ).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return self.context_dict()

    @classmethod
    def from_dict(cls, data: Any) -> "DialogueTurnInput":
        data = expect_mapping(data, "DialogueTurnInput")
        try:
            return cls(
                turn_id=data["turn_id"],
                user_input=data["user_input"],
                history=data.get("history", ()),
                state_view=data.get("state_view", {}),
                available_tools=data.get("available_tools", ()),
            )
        except KeyError as exc:
            raise ContractValidationError(
                f"DialogueTurnInput missing required field: {exc.args[0]}"
            ) from exc

    def to_json(self) -> str:
        return dumps_dict(self.to_dict())

    @classmethod
    def from_json(cls, payload: str) -> "DialogueTurnInput":
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ContractValidationError(
                f"DialogueTurnInput invalid JSON: {exc}"
            ) from exc
        return cls.from_dict(data)


@dataclass(frozen=True)
class DialogueTurnOutput:
    """What a dialogue system produced for one ``respond()`` call.

    ``tool_calls`` are *proposals*: the harness executes them through
    the tool simulator and records the resulting ``ToolEvent`` objects
    (with status/result) itself. ``assistant_tags`` carries structured
    signals (e.g. ``"clarify"``) so conditional user turns never depend
    on natural-language wording.
    """

    output_text: str
    tool_calls: Tuple[ToolEvent, ...] = ()
    assistant_tags: Tuple[str, ...] = ()
    usage: UsageStats = field(default_factory=UsageStats)
    retrieval_events: Tuple[RetrievalEvent, ...] = ()
    memory_events: Tuple[MemoryEvent, ...] = ()
    runtime_metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "output_text", require_str(self.output_text, "output_text")
        )
        object.__setattr__(
            self,
            "tool_calls",
            _event_tuple(self.tool_calls, ToolEvent, "tool_calls"),
        )
        object.__setattr__(
            self,
            "assistant_tags",
            str_tuple(self.assistant_tags, "assistant_tags"),
        )
        if not isinstance(self.usage, UsageStats):
            object.__setattr__(self, "usage", UsageStats.from_dict(self.usage))
        object.__setattr__(
            self,
            "retrieval_events",
            _event_tuple(
                self.retrieval_events, RetrievalEvent, "retrieval_events"
            ),
        )
        object.__setattr__(
            self,
            "memory_events",
            _event_tuple(self.memory_events, MemoryEvent, "memory_events"),
        )
        object.__setattr__(
            self,
            "runtime_metadata",
            dict(require_json_dict(self.runtime_metadata, "runtime_metadata")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "output_text": self.output_text,
            "tool_calls": [event.to_dict() for event in self.tool_calls],
            "assistant_tags": list(self.assistant_tags),
            "usage": self.usage.to_dict(),
            "retrieval_events": [e.to_dict() for e in self.retrieval_events],
            "memory_events": [e.to_dict() for e in self.memory_events],
            "runtime_metadata": dict(self.runtime_metadata),
        }

    @classmethod
    def from_dict(cls, data: Any) -> "DialogueTurnOutput":
        data = expect_mapping(data, "DialogueTurnOutput")
        try:
            return cls(
                output_text=data["output_text"],
                tool_calls=data.get("tool_calls", ()),
                assistant_tags=data.get("assistant_tags", ()),
                usage=data.get("usage", {}),
                retrieval_events=data.get("retrieval_events", ()),
                memory_events=data.get("memory_events", ()),
                runtime_metadata=data.get("runtime_metadata", {}),
            )
        except KeyError as exc:
            raise ContractValidationError(
                f"DialogueTurnOutput missing required field: {exc.args[0]}"
            ) from exc

    def to_json(self) -> str:
        return dumps_dict(self.to_dict())

    @classmethod
    def from_json(cls, payload: str) -> "DialogueTurnOutput":
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ContractValidationError(
                f"DialogueTurnOutput invalid JSON: {exc}"
            ) from exc
        return cls.from_dict(data)


# ---------------------------------------------------------------------------
# Trace contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DialogueStepTrace:
    """One inner-loop assistant invocation within a user turn.

    A turn may contain several steps (assistant proposes tool calls →
    harness executes → assistant responds again). ``history_digest`` is
    the SHA-256 of the canonical JSON of the exact ``DialogueTurnInput``
    context shown to this invocation (history + user input + state
    view + available tools), not merely the transcript.
    """

    step_index: int
    output_text: str
    assistant_tags: Tuple[str, ...] = ()
    tool_events: Tuple[ToolEvent, ...] = ()
    history_digest: str = ""
    usage: UsageStats = field(default_factory=UsageStats)
    latency_ms: float = 0.0
    runtime_metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "step_index", require_int(self.step_index, "step_index")
        )
        if self.step_index < 0:
            raise ContractValidationError("step_index must be >= 0")
        object.__setattr__(
            self, "output_text", require_str(self.output_text, "output_text")
        )
        object.__setattr__(
            self,
            "assistant_tags",
            str_tuple(self.assistant_tags, "assistant_tags"),
        )
        object.__setattr__(
            self,
            "tool_events",
            _event_tuple(self.tool_events, ToolEvent, "tool_events"),
        )
        object.__setattr__(
            self, "history_digest", require_str(self.history_digest, "history_digest")
        )
        if not isinstance(self.usage, UsageStats):
            object.__setattr__(self, "usage", UsageStats.from_dict(self.usage))
        object.__setattr__(
            self,
            "latency_ms",
            require_non_negative_number(self.latency_ms, "latency_ms"),
        )
        object.__setattr__(
            self,
            "runtime_metadata",
            dict(require_json_dict(self.runtime_metadata, "runtime_metadata")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_index": self.step_index,
            "output_text": self.output_text,
            "assistant_tags": list(self.assistant_tags),
            "tool_events": [e.to_dict() for e in self.tool_events],
            "history_digest": self.history_digest,
            "usage": self.usage.to_dict(),
            "latency_ms": self.latency_ms,
            "runtime_metadata": dict(self.runtime_metadata),
        }

    @classmethod
    def from_dict(cls, data: Any) -> "DialogueStepTrace":
        data = expect_mapping(data, "DialogueStepTrace")
        try:
            return cls(
                step_index=data["step_index"],
                output_text=data["output_text"],
                assistant_tags=data.get("assistant_tags", ()),
                tool_events=data.get("tool_events", ()),
                history_digest=data.get("history_digest", ""),
                usage=data.get("usage", {}),
                latency_ms=data.get("latency_ms", 0.0),
                runtime_metadata=data.get("runtime_metadata", {}),
            )
        except KeyError as exc:
            raise ContractValidationError(
                f"DialogueStepTrace missing required field: {exc.args[0]}"
            ) from exc


@dataclass(frozen=True)
class DialogueTurnTrace:
    """Factual record of one user turn: state snapshots plus inner steps.

    ``output_text`` is the final assistant-visible message of this turn
    (empty if the last step was tool-only); ``tool_events`` aggregates
    all executed calls in execution order; ``steps`` preserves the
    complete inner-loop evidence. ``activation_evidence`` records why
    this user message was released.
    """

    turn_id: str
    turn_index: int
    user_input: str

    state_before: Dict[str, Any]
    state_after: Dict[str, Any]

    output_text: str

    steps: Tuple[DialogueStepTrace, ...] = ()
    assistant_tags: Tuple[str, ...] = ()
    tool_events: Tuple[ToolEvent, ...] = ()
    activation_evidence: Dict[str, Any] = field(default_factory=dict)

    usage: UsageStats = field(default_factory=UsageStats)
    latency_ms: float = 0.0
    runtime_metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "turn_id", require_non_empty_str(self.turn_id, "turn_id")
        )
        object.__setattr__(
            self, "turn_index", require_int(self.turn_index, "turn_index")
        )
        if self.turn_index < 1:
            raise ContractValidationError("turn_index must be >= 1")
        object.__setattr__(
            self,
            "user_input",
            require_non_empty_str(self.user_input, "user_input"),
        )
        object.__setattr__(
            self,
            "state_before",
            dict(require_json_dict(self.state_before, "state_before")),
        )
        object.__setattr__(
            self,
            "state_after",
            dict(require_json_dict(self.state_after, "state_after")),
        )
        object.__setattr__(
            self, "output_text", require_str(self.output_text, "output_text")
        )
        steps = []
        for item in self.steps:
            if not isinstance(item, DialogueStepTrace):
                item = DialogueStepTrace.from_dict(item)
            steps.append(item)
        object.__setattr__(self, "steps", tuple(steps))
        object.__setattr__(
            self,
            "assistant_tags",
            str_tuple(self.assistant_tags, "assistant_tags"),
        )
        object.__setattr__(
            self,
            "tool_events",
            _event_tuple(self.tool_events, ToolEvent, "tool_events"),
        )
        object.__setattr__(
            self,
            "activation_evidence",
            dict(
                require_json_dict(
                    self.activation_evidence, "activation_evidence"
                )
            ),
        )
        if not isinstance(self.usage, UsageStats):
            object.__setattr__(self, "usage", UsageStats.from_dict(self.usage))
        object.__setattr__(
            self,
            "latency_ms",
            require_non_negative_number(self.latency_ms, "latency_ms"),
        )
        object.__setattr__(
            self,
            "runtime_metadata",
            dict(require_json_dict(self.runtime_metadata, "runtime_metadata")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "turn_index": self.turn_index,
            "user_input": self.user_input,
            "state_before": self.state_before,
            "state_after": self.state_after,
            "output_text": self.output_text,
            "steps": [step.to_dict() for step in self.steps],
            "assistant_tags": list(self.assistant_tags),
            "tool_events": [e.to_dict() for e in self.tool_events],
            "activation_evidence": dict(self.activation_evidence),
            "usage": self.usage.to_dict(),
            "latency_ms": self.latency_ms,
            "runtime_metadata": dict(self.runtime_metadata),
        }

    @classmethod
    def from_dict(cls, data: Any) -> "DialogueTurnTrace":
        data = expect_mapping(data, "DialogueTurnTrace")
        try:
            return cls(
                turn_id=data["turn_id"],
                turn_index=data["turn_index"],
                user_input=data["user_input"],
                state_before=data["state_before"],
                state_after=data["state_after"],
                output_text=data["output_text"],
                steps=data.get("steps", ()),
                assistant_tags=data.get("assistant_tags", ()),
                tool_events=data.get("tool_events", ()),
                activation_evidence=data.get("activation_evidence", {}),
                usage=data.get("usage", {}),
                latency_ms=data.get("latency_ms", 0.0),
                runtime_metadata=data.get("runtime_metadata", {}),
            )
        except KeyError as exc:
            raise ContractValidationError(
                f"DialogueTurnTrace missing required field: {exc.args[0]}"
            ) from exc


@dataclass(frozen=True)
class DialogueRunTrace:
    """Factual record of one dialogue case execution under one experiment.

    ``output_text`` is strictly the final assistant-visible message of
    the last processed turn — never a concatenated transcript; the
    ordered ``turns`` carry the full transcript, per-turn state
    snapshots, tool executions, and inner-loop steps. ``final_state``
    is the canonical state after the last completed turn (equals
    ``initial_state`` when no turn completed).
    """

    schema_version: str

    run_id: str
    experiment_id: str
    case_id: str

    system_id: str
    system_version: str
    config_fingerprint: str

    turns: Tuple[DialogueTurnTrace, ...]

    initial_state: Dict[str, Any]
    final_state: Dict[str, Any]

    termination_reason: str

    output_text: str

    started_at: str
    finished_at: str
    latency_ms: float

    usage: UsageStats = field(default_factory=UsageStats)
    runtime_metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "schema_version",
            require_non_empty_str(self.schema_version, "schema_version"),
        )
        object.__setattr__(
            self, "run_id", require_non_empty_str(self.run_id, "run_id")
        )
        object.__setattr__(
            self,
            "experiment_id",
            require_non_empty_str(self.experiment_id, "experiment_id"),
        )
        object.__setattr__(
            self, "case_id", require_non_empty_str(self.case_id, "case_id")
        )
        object.__setattr__(
            self, "system_id", require_non_empty_str(self.system_id, "system_id")
        )
        object.__setattr__(
            self,
            "system_version",
            require_non_empty_str(self.system_version, "system_version"),
        )
        object.__setattr__(
            self,
            "config_fingerprint",
            require_non_empty_str(self.config_fingerprint, "config_fingerprint"),
        )
        turns = []
        for item in self.turns:
            if not isinstance(item, DialogueTurnTrace):
                item = DialogueTurnTrace.from_dict(item)
            turns.append(item)
        object.__setattr__(self, "turns", tuple(turns))
        object.__setattr__(
            self,
            "initial_state",
            dict(require_json_dict(self.initial_state, "initial_state")),
        )
        object.__setattr__(
            self,
            "final_state",
            dict(require_json_dict(self.final_state, "final_state")),
        )
        object.__setattr__(
            self,
            "termination_reason",
            _require_termination_reason(
                self.termination_reason, "termination_reason"
            ),
        )
        object.__setattr__(
            self, "output_text", require_str(self.output_text, "output_text")
        )
        object.__setattr__(
            self, "started_at", require_iso8601(self.started_at, "started_at")
        )
        object.__setattr__(
            self, "finished_at", require_iso8601(self.finished_at, "finished_at")
        )
        object.__setattr__(
            self,
            "latency_ms",
            require_non_negative_number(self.latency_ms, "latency_ms"),
        )
        if not isinstance(self.usage, UsageStats):
            object.__setattr__(self, "usage", UsageStats.from_dict(self.usage))
        object.__setattr__(
            self,
            "runtime_metadata",
            dict(require_json_dict(self.runtime_metadata, "runtime_metadata")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "case_id": self.case_id,
            "system_id": self.system_id,
            "system_version": self.system_version,
            "config_fingerprint": self.config_fingerprint,
            "turns": [turn.to_dict() for turn in self.turns],
            "initial_state": self.initial_state,
            "final_state": self.final_state,
            "termination_reason": self.termination_reason,
            "output_text": self.output_text,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "latency_ms": self.latency_ms,
            "usage": self.usage.to_dict(),
            "runtime_metadata": dict(self.runtime_metadata),
        }

    @classmethod
    def from_dict(cls, data: Any) -> "DialogueRunTrace":
        data = expect_mapping(data, "DialogueRunTrace")
        try:
            return cls(
                schema_version=data["schema_version"],
                run_id=data["run_id"],
                experiment_id=data["experiment_id"],
                case_id=data["case_id"],
                system_id=data["system_id"],
                system_version=data["system_version"],
                config_fingerprint=data["config_fingerprint"],
                turns=data["turns"],
                initial_state=data["initial_state"],
                final_state=data["final_state"],
                termination_reason=data["termination_reason"],
                output_text=data["output_text"],
                started_at=data["started_at"],
                finished_at=data["finished_at"],
                latency_ms=data["latency_ms"],
                usage=data.get("usage", {}),
                runtime_metadata=data.get("runtime_metadata", {}),
            )
        except KeyError as exc:
            raise ContractValidationError(
                f"DialogueRunTrace missing required field: {exc.args[0]}"
            ) from exc

    def to_json(self) -> str:
        return dumps_dict(self.to_dict())

    @classmethod
    def from_json(cls, payload: str) -> "DialogueRunTrace":
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ContractValidationError(
                f"DialogueRunTrace invalid JSON: {exc}"
            ) from exc
        return cls.from_dict(data)


__all__ = [
    "DialogueCaseSpec",
    "DialogueHistoryEntry",
    "DialogueRunTrace",
    "DialogueStepTrace",
    "DialogueToolDescriptor",
    "DialogueTurnInput",
    "DialogueTurnOutput",
    "DialogueTurnSpec",
    "DialogueTurnTrace",
    "ExpectedToolCall",
    "HISTORY_ROLES",
    "TERMINATION_COMPLETED",
    "TERMINATION_INVALID_TOOL",
    "TERMINATION_MAX_TURNS",
    "TERMINATION_REASONS",
    "TERMINATION_SYSTEM_ERROR",
    "TERMINATION_TOOL_LIMIT",
    "TERMINATION_USER_DONE",
]
