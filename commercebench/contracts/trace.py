"""RunTrace: a factual record of one case execution (V0 contract).

A RunTrace records *what happened* during a run. Metric scores and pass/fail
judgements live in EvaluationResult, never here, so the same trace can be
re-evaluated later by new evaluators without re-running the system.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .common import (
    dumps_dict,
    ensure_json_compatible,
    expect_mapping,
    optional_non_negative_int,
    parse_enum,
    require_iso8601,
    require_json_dict,
    require_non_empty_str,
    require_non_negative_number,
    require_str,
    str_tuple,
)
from .errors import ContractValidationError


class RetrievalRole(str, Enum):
    """Role of a ``RetrievalEvent`` within a run (Phase 2A)."""

    CANDIDATE = "candidate"
    FINAL = "final"


@dataclass(frozen=True)
class RetrievalEvent:
    """One retrieval step recorded in a run.

    ``role`` selects how the ranking is treated (Phase 2A):

    - ``RetrievalRole.CANDIDATE``: intermediate ranking for diagnostics;
      excluded from retrieval evaluation.
    - ``RetrievalRole.FINAL``: the evaluation-visible retrieval ranking —
      what downstream context/generation consumes and what retrieval
      metrics are computed from.
    - ``None`` (default): legacy Phase 1 traces carry no role; they are
      interpreted as ``FINAL`` for backward compatibility.
    """

    query: str
    document_ids: Tuple[str, ...] = ()
    scores: Tuple[float, ...] = ()
    role: Optional[RetrievalRole] = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "query", require_non_empty_str(self.query, "query")
        )
        object.__setattr__(
            self,
            "document_ids",
            str_tuple(self.document_ids, "document_ids"),
        )
        scores = tuple(self.scores)
        for index, score in enumerate(scores):
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise ContractValidationError(
                    f"scores[{index}] must be a number"
                )
        object.__setattr__(self, "scores", tuple(float(s) for s in scores))
        if len(self.document_ids) != len(self.scores):
            raise ContractValidationError(
                "document_ids and scores must have the same length"
            )
        if self.role is not None:
            object.__setattr__(
                self, "role", parse_enum(RetrievalRole, self.role, "role")
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "document_ids": list(self.document_ids),
            "scores": list(self.scores),
            "role": self.role.value if self.role is not None else None,
        }

    @classmethod
    def from_dict(cls, data: Any) -> "RetrievalEvent":
        data = expect_mapping(data, "RetrievalEvent")
        try:
            return cls(
                query=data["query"],
                document_ids=data.get("document_ids", ()),
                scores=data.get("scores", ()),
                role=data.get("role"),
            )
        except KeyError as exc:
            raise ContractValidationError(
                f"RetrievalEvent missing required field: {exc.args[0]}"
            ) from exc


@dataclass(frozen=True)
class ToolEvent:
    """One tool invocation. Phase 0 defines the shape only; no tools exist."""

    tool_name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    result: Any = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "tool_name", require_non_empty_str(self.tool_name, "tool_name")
        )
        object.__setattr__(
            self, "arguments", dict(require_json_dict(self.arguments, "arguments"))
        )
        ensure_json_compatible(self.result, "result")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "arguments": dict(self.arguments),
            "result": self.result,
        }

    @classmethod
    def from_dict(cls, data: Any) -> "ToolEvent":
        data = expect_mapping(data, "ToolEvent")
        try:
            return cls(
                tool_name=data["tool_name"],
                arguments=data.get("arguments", {}),
                result=data.get("result"),
            )
        except KeyError as exc:
            raise ContractValidationError(
                f"ToolEvent missing required field: {exc.args[0]}"
            ) from exc


@dataclass(frozen=True)
class MemoryEvent:
    """One memory-selection step. Phase 0 defines the shape only."""

    strategy: str
    selected_item_ids: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "strategy", require_non_empty_str(self.strategy, "strategy")
        )
        object.__setattr__(
            self,
            "selected_item_ids",
            str_tuple(self.selected_item_ids, "selected_item_ids"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "selected_item_ids": list(self.selected_item_ids),
        }

    @classmethod
    def from_dict(cls, data: Any) -> "MemoryEvent":
        data = expect_mapping(data, "MemoryEvent")
        try:
            return cls(
                strategy=data["strategy"],
                selected_item_ids=data.get("selected_item_ids", ()),
            )
        except KeyError as exc:
            raise ContractValidationError(
                f"MemoryEvent missing required field: {exc.args[0]}"
            ) from exc


@dataclass(frozen=True)
class UsageStats:
    """Token accounting. ``None`` means the system did not report it."""

    input_tokens: Any = None
    output_tokens: Any = None
    total_tokens: Any = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "input_tokens",
            optional_non_negative_int(self.input_tokens, "input_tokens"),
        )
        object.__setattr__(
            self,
            "output_tokens",
            optional_non_negative_int(self.output_tokens, "output_tokens"),
        )
        object.__setattr__(
            self,
            "total_tokens",
            optional_non_negative_int(self.total_tokens, "total_tokens"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }

    @classmethod
    def from_dict(cls, data: Any) -> "UsageStats":
        data = expect_mapping(data, "UsageStats")
        return cls(
            input_tokens=data.get("input_tokens"),
            output_tokens=data.get("output_tokens"),
            total_tokens=data.get("total_tokens"),
        )


@dataclass(frozen=True)
class RunTrace:
    """Factual record of a single case execution under one experiment."""

    schema_version: str

    run_id: str
    experiment_id: str
    case_id: str

    system_id: str
    system_version: str
    config_fingerprint: str

    input_text: str
    output_text: str

    started_at: str
    finished_at: str

    latency_ms: float

    retrieval_events: Tuple[RetrievalEvent, ...] = ()
    tool_events: Tuple[ToolEvent, ...] = ()
    memory_events: Tuple[MemoryEvent, ...] = ()

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
        object.__setattr__(
            self, "input_text", require_non_empty_str(self.input_text, "input_text")
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
        object.__setattr__(
            self,
            "retrieval_events",
            _event_tuple(
                self.retrieval_events, RetrievalEvent, "retrieval_events"
            ),
        )
        object.__setattr__(
            self,
            "tool_events",
            _event_tuple(self.tool_events, ToolEvent, "tool_events"),
        )
        object.__setattr__(
            self,
            "memory_events",
            _event_tuple(self.memory_events, MemoryEvent, "memory_events"),
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
            "input_text": self.input_text,
            "output_text": self.output_text,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "latency_ms": self.latency_ms,
            "retrieval_events": [e.to_dict() for e in self.retrieval_events],
            "tool_events": [e.to_dict() for e in self.tool_events],
            "memory_events": [e.to_dict() for e in self.memory_events],
            "usage": self.usage.to_dict(),
            "runtime_metadata": dict(self.runtime_metadata),
        }

    @classmethod
    def from_dict(cls, data: Any) -> "RunTrace":
        data = expect_mapping(data, "RunTrace")
        try:
            return cls(
                schema_version=data["schema_version"],
                run_id=data["run_id"],
                experiment_id=data["experiment_id"],
                case_id=data["case_id"],
                system_id=data["system_id"],
                system_version=data["system_version"],
                config_fingerprint=data["config_fingerprint"],
                input_text=data["input_text"],
                output_text=data["output_text"],
                started_at=data["started_at"],
                finished_at=data["finished_at"],
                latency_ms=data["latency_ms"],
                retrieval_events=data.get("retrieval_events", ()),
                tool_events=data.get("tool_events", ()),
                memory_events=data.get("memory_events", ()),
                usage=data.get("usage", {}),
                runtime_metadata=data.get("runtime_metadata", {}),
            )
        except KeyError as exc:
            raise ContractValidationError(
                f"RunTrace missing required field: {exc.args[0]}"
            ) from exc

    def to_json(self) -> str:
        return dumps_dict(self.to_dict())

    @classmethod
    def from_json(cls, payload: str) -> "RunTrace":
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ContractValidationError(f"RunTrace invalid JSON: {exc}") from exc
        return cls.from_dict(data)

    @property
    def final_retrieval_events(self) -> Tuple[RetrievalEvent, ...]:
        """Evaluation-visible retrieval events, in trace order.

        An event is evaluation-visible when ``role`` is
        ``RetrievalRole.FINAL`` or ``None`` (legacy Phase 1 semantics).
        ``CANDIDATE`` events are diagnostic and never included.
        """
        return tuple(
            event
            for event in self.retrieval_events
            if event.role is None or event.role is RetrievalRole.FINAL
        )

    @property
    def final_ranked_document_ids(self) -> Tuple[str, ...]:
        """Document IDs from evaluation-visible final retrieval events only.

        Final events are concatenated in trace order and de-duplicated
        keep-first, so ``final_ranked_document_ids[i]`` is the rank
        ``i + 1`` result of the final ranking. ``CANDIDATE`` events never
        contribute; a trace with only candidate events yields ``()``.
        """
        seen = set()
        ordered: List[str] = []
        for event in self.final_retrieval_events:
            for document_id in event.document_ids:
                if document_id not in seen:
                    seen.add(document_id)
                    ordered.append(document_id)
        return tuple(ordered)


def _event_tuple(value: Any, event_cls: Any, field_name: str) -> Tuple[Any, ...]:
    if not isinstance(value, (list, tuple)):
        raise ContractValidationError(f"{field_name} must be a list")
    items = []
    for index, item in enumerate(value):
        if not isinstance(item, event_cls):
            try:
                item = event_cls.from_dict(item)
            except ContractValidationError:
                raise
            except Exception as exc:
                raise ContractValidationError(
                    f"{field_name}[{index}] is invalid: {exc}"
                ) from exc
        items.append(item)
    return tuple(items)


__all__ = [
    "MemoryEvent",
    "RetrievalEvent",
    "RetrievalRole",
    "RunTrace",
    "ToolEvent",
    "UsageStats",
]
