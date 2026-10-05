"""ExperimentManifest: how one experiment run is configured (V0 contract)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from .common import (
    canonical_json,
    dumps_dict,
    expect_mapping,
    require_int,
    require_json_dict,
    require_non_empty_str,
    str_tuple,
)
from .errors import ContractValidationError


@dataclass(frozen=True)
class ModelConfig:
    """Framework-neutral model selection. Phase 0 stores config only."""

    provider: str
    model_name: str
    parameters: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "provider", require_non_empty_str(self.provider, "provider")
        )
        object.__setattr__(
            self,
            "model_name",
            require_non_empty_str(self.model_name, "model_name"),
        )
        object.__setattr__(
            self, "parameters", dict(require_json_dict(self.parameters, "parameters"))
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "model_name": self.model_name,
            "parameters": dict(self.parameters),
        }

    @classmethod
    def from_dict(cls, data: Any) -> "ModelConfig":
        data = expect_mapping(data, "ModelConfig")
        try:
            return cls(
                provider=data["provider"],
                model_name=data["model_name"],
                parameters=data.get("parameters", {}),
            )
        except KeyError as exc:
            raise ContractValidationError(
                f"ModelConfig missing required field: {exc.args[0]}"
            ) from exc


@dataclass(frozen=True)
class RetrievalConfig:
    """Framework-neutral retrieval selection. Phase 0 stores config only."""

    retriever_id: str
    parameters: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "retriever_id",
            require_non_empty_str(self.retriever_id, "retriever_id"),
        )
        object.__setattr__(
            self, "parameters", dict(require_json_dict(self.parameters, "parameters"))
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "retriever_id": self.retriever_id,
            "parameters": dict(self.parameters),
        }

    @classmethod
    def from_dict(cls, data: Any) -> "RetrievalConfig":
        data = expect_mapping(data, "RetrievalConfig")
        try:
            return cls(
                retriever_id=data["retriever_id"],
                parameters=data.get("parameters", {}),
            )
        except KeyError as exc:
            raise ContractValidationError(
                f"RetrievalConfig missing required field: {exc.args[0]}"
            ) from exc


@dataclass(frozen=True)
class RerankerConfig:
    """Framework-neutral reranker selection (Phase 2B).

    A reranker is a separate system variable from retrieval: it reorders
    or filters the candidate ranking produced by the retriever and never
    introduces documents outside the candidate set. ``parameters`` holds
    result-affecting reranker configuration (for example
    ``candidate_top_k`` and, for the learned baseline, the pinned model
    identity); execution-only knobs (device, batch size, cache location)
    must not enter ``parameters``.
    """

    reranker_id: str
    reranker_version: str
    parameters: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "reranker_id",
            require_non_empty_str(self.reranker_id, "reranker_id"),
        )
        object.__setattr__(
            self,
            "reranker_version",
            require_non_empty_str(self.reranker_version, "reranker_version"),
        )
        object.__setattr__(
            self, "parameters", dict(require_json_dict(self.parameters, "parameters"))
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reranker_id": self.reranker_id,
            "reranker_version": self.reranker_version,
            "parameters": dict(self.parameters),
        }

    @classmethod
    def from_dict(cls, data: Any) -> "RerankerConfig":
        data = expect_mapping(data, "RerankerConfig")
        try:
            return cls(
                reranker_id=data["reranker_id"],
                reranker_version=data["reranker_version"],
                parameters=data.get("parameters", {}),
            )
        except KeyError as exc:
            raise ContractValidationError(
                f"RerankerConfig missing required field: {exc.args[0]}"
            ) from exc


@dataclass(frozen=True)
class MemoryConfig:
    """Framework-neutral memory selection. Phase 0 stores config only."""

    strategy: str
    parameters: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "strategy", require_non_empty_str(self.strategy, "strategy")
        )
        object.__setattr__(
            self, "parameters", dict(require_json_dict(self.parameters, "parameters"))
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "parameters": dict(self.parameters),
        }

    @classmethod
    def from_dict(cls, data: Any) -> "MemoryConfig":
        data = expect_mapping(data, "MemoryConfig")
        try:
            return cls(
                strategy=data["strategy"],
                parameters=data.get("parameters", {}),
            )
        except KeyError as exc:
            raise ContractValidationError(
                f"MemoryConfig missing required field: {exc.args[0]}"
            ) from exc


_ALLOWED_DIALOGUE_PARAMETERS: Tuple[str, ...] = (
    "max_turns",
    "max_tool_calls_per_turn",
    "history_policy",
    "tool_policy",
    "tool_simulator",
)

_ALLOWED_HISTORY_POLICIES: Tuple[str, ...] = ("full",)


@dataclass(frozen=True)
class DialogueHarnessConfig:
    """Framework-neutral dialogue/tool harness selection (Phase 3B).

    The harness — not the model — owns turn iteration, history
    construction, the canonical business state, and tool execution.
    All result-affecting orchestration lives in ``parameters`` and is
    validated fail-closed so hidden defaults cannot drift: unknown keys
    are rejected. Execution-only knobs (timeouts, logging) MUST NOT
    enter ``parameters``.
    """

    harness_id: str
    harness_version: str
    parameters: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "harness_id",
            require_non_empty_str(self.harness_id, "harness_id"),
        )
        object.__setattr__(
            self,
            "harness_version",
            require_non_empty_str(self.harness_version, "harness_version"),
        )
        object.__setattr__(
            self,
            "parameters",
            dict(require_json_dict(self.parameters, "parameters")),
        )
        unknown = set(self.parameters) - set(_ALLOWED_DIALOGUE_PARAMETERS)
        if unknown:
            raise ContractValidationError(
                f"unknown dialogue harness parameters: {sorted(unknown)}"
            )
        missing = set(_ALLOWED_DIALOGUE_PARAMETERS) - set(self.parameters)
        if missing:
            raise ContractValidationError(
                "dialogue harness parameters missing required keys: "
                f"{sorted(missing)} — all result-affecting knobs must be "
                "explicit (no hidden defaults)"
            )
        if "max_turns" in self.parameters:
            require_int(
                self.parameters["max_turns"], "parameters.max_turns"
            )
            if self.parameters["max_turns"] <= 0:
                raise ContractValidationError(
                    "parameters.max_turns must be a positive integer"
                )
        if "max_tool_calls_per_turn" in self.parameters:
            value = self.parameters["max_tool_calls_per_turn"]
            require_int(value, "parameters.max_tool_calls_per_turn")
            if value <= 0:
                raise ContractValidationError(
                    "parameters.max_tool_calls_per_turn must be positive"
                )
        if "history_policy" in self.parameters and (
            self.parameters["history_policy"] not in _ALLOWED_HISTORY_POLICIES
        ):
            raise ContractValidationError(
                "parameters.history_policy must be one of "
                f"{list(_ALLOWED_HISTORY_POLICIES)}"
            )
        if "tool_policy" in self.parameters:
            require_non_empty_str(
                self.parameters["tool_policy"], "parameters.tool_policy"
            )
        if "tool_simulator" in self.parameters:
            simulator = require_json_dict(
                self.parameters["tool_simulator"], "parameters.tool_simulator"
            )
            require_non_empty_str(
                simulator.get("id"), "parameters.tool_simulator.id"
            )
            require_non_empty_str(
                simulator.get("version"), "parameters.tool_simulator.version"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "harness_id": self.harness_id,
            "harness_version": self.harness_version,
            "parameters": dict(self.parameters),
        }

    @classmethod
    def from_dict(cls, data: Any) -> "DialogueHarnessConfig":
        data = expect_mapping(data, "DialogueHarnessConfig")
        try:
            return cls(
                harness_id=data["harness_id"],
                harness_version=data["harness_version"],
                parameters=data.get("parameters", {}),
            )
        except KeyError as exc:
            raise ContractValidationError(
                f"DialogueHarnessConfig missing required field: {exc.args[0]}"
            ) from exc

@dataclass(frozen=True)
class ExperimentManifest:
    """Describes one experiment: which cases, which system, which evaluator.

    ``config_fingerprint()`` covers exactly the fields that can change run
    outputs: schema/benchmark versions, the ordered case list, system and
    evaluator identities and versions, model/retrieval/reranker/memory
    configs, and the random seed. ``experiment_id``, ``experiment_name``
    and ``metadata`` are descriptive identity fields and are excluded by
    design; ``case_ids`` order is significant and preserved in the
    fingerprint. A legacy manifest without ``reranker`` loads with
    ``reranker=None`` and keeps its Phase 1 / Phase 2A fingerprint
    bit-identical (the ``reranker`` key is omitted from the fingerprint
    payload when ``None``). Phase 3B applies the same rule to
    ``dialogue``: absent or null stays out of the fingerprint payload,
    so every Phase 0–2G fingerprint is preserved bit-identically.
    """

    schema_version: str
    experiment_id: str
    experiment_name: str

    benchmark_version: str
    case_ids: Tuple[str, ...]

    system_id: str
    system_version: str

    model: Optional[ModelConfig] = None
    retrieval: Optional[RetrievalConfig] = None
    reranker: Optional[RerankerConfig] = None
    memory: Optional[MemoryConfig] = None
    dialogue: Optional[DialogueHarnessConfig] = None

    evaluator_id: str = ""
    evaluator_version: str = ""

    random_seed: int = 0

    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "schema_version",
            require_non_empty_str(self.schema_version, "schema_version"),
        )
        object.__setattr__(
            self,
            "experiment_id",
            require_non_empty_str(self.experiment_id, "experiment_id"),
        )
        object.__setattr__(
            self,
            "experiment_name",
            require_non_empty_str(self.experiment_name, "experiment_name"),
        )
        object.__setattr__(
            self,
            "benchmark_version",
            require_non_empty_str(self.benchmark_version, "benchmark_version"),
        )
        object.__setattr__(
            self, "case_ids", str_tuple(self.case_ids, "case_ids")
        )
        object.__setattr__(
            self, "system_id", require_non_empty_str(self.system_id, "system_id")
        )
        object.__setattr__(
            self,
            "system_version",
            require_non_empty_str(self.system_version, "system_version"),
        )
        if self.model is not None and not isinstance(self.model, ModelConfig):
            object.__setattr__(self, "model", ModelConfig.from_dict(self.model))
        if self.retrieval is not None and not isinstance(
            self.retrieval, RetrievalConfig
        ):
            object.__setattr__(
                self, "retrieval", RetrievalConfig.from_dict(self.retrieval)
            )
        if self.reranker is not None and not isinstance(
            self.reranker, RerankerConfig
        ):
            object.__setattr__(
                self, "reranker", RerankerConfig.from_dict(self.reranker)
            )
        if self.memory is not None and not isinstance(self.memory, MemoryConfig):
            object.__setattr__(self, "memory", MemoryConfig.from_dict(self.memory))
        if self.dialogue is not None and not isinstance(
            self.dialogue, DialogueHarnessConfig
        ):
            object.__setattr__(
                self, "dialogue", DialogueHarnessConfig.from_dict(self.dialogue)
            )
        object.__setattr__(
            self,
            "evaluator_id",
            require_non_empty_str(self.evaluator_id, "evaluator_id"),
        )
        object.__setattr__(
            self,
            "evaluator_version",
            require_non_empty_str(self.evaluator_version, "evaluator_version"),
        )
        object.__setattr__(
            self, "random_seed", require_int(self.random_seed, "random_seed")
        )
        object.__setattr__(
            self, "metadata", dict(require_json_dict(self.metadata, "metadata"))
        )

    def _fingerprint_payload(self) -> Dict[str, Any]:
        # Backward compatibility (Phase 2B): legacy manifests carry
        # ``reranker=None``. Omitting the key in that case keeps every
        # Phase 1 / Phase 2A fingerprint bit-identical after the upgrade.
        # A present reranker enters the payload and therefore the
        # experiment identity.
        payload: Dict[str, Any] = {
            "schema_version": self.schema_version,
            "benchmark_version": self.benchmark_version,
            "case_ids": list(self.case_ids),
            "system_id": self.system_id,
            "system_version": self.system_version,
            "model": self.model.to_dict() if self.model is not None else None,
            "retrieval": (
                self.retrieval.to_dict() if self.retrieval is not None else None
            ),
            "memory": self.memory.to_dict() if self.memory is not None else None,
            "evaluator_id": self.evaluator_id,
            "evaluator_version": self.evaluator_version,
            "random_seed": self.random_seed,
        }
        if self.reranker is not None:
            payload["reranker"] = self.reranker.to_dict()
        # Phase 3B: omit-when-None keeps all Phase 0–2G fingerprints
        # bit-identical; a present dialogue harness is result-affecting
        # and enters the experiment identity.
        if self.dialogue is not None:
            payload["dialogue"] = self.dialogue.to_dict()
        return payload

    def config_fingerprint(self) -> str:
        """SHA-256 of the canonical JSON of output-affecting configuration."""
        canonical = canonical_json(self._fingerprint_payload())
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "benchmark_version": self.benchmark_version,
            "case_ids": list(self.case_ids),
            "system_id": self.system_id,
            "system_version": self.system_version,
            "model": self.model.to_dict() if self.model is not None else None,
            "retrieval": (
                self.retrieval.to_dict() if self.retrieval is not None else None
            ),
            "reranker": (
                self.reranker.to_dict() if self.reranker is not None else None
            ),
            "memory": self.memory.to_dict() if self.memory is not None else None,
            "dialogue": (
                self.dialogue.to_dict() if self.dialogue is not None else None
            ),
            "evaluator_id": self.evaluator_id,
            "evaluator_version": self.evaluator_version,
            "random_seed": self.random_seed,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Any) -> "ExperimentManifest":
        data = expect_mapping(data, "ExperimentManifest")
        try:
            return cls(
                schema_version=data["schema_version"],
                experiment_id=data["experiment_id"],
                experiment_name=data["experiment_name"],
                benchmark_version=data["benchmark_version"],
                case_ids=data["case_ids"],
                system_id=data["system_id"],
                system_version=data["system_version"],
                model=data.get("model"),
                retrieval=data.get("retrieval"),
                reranker=data.get("reranker"),
                memory=data.get("memory"),
                dialogue=data.get("dialogue"),
                evaluator_id=data["evaluator_id"],
                evaluator_version=data["evaluator_version"],
                random_seed=data.get("random_seed", 0),
                metadata=data.get("metadata", {}),
            )
        except KeyError as exc:
            raise ContractValidationError(
                f"ExperimentManifest missing required field: {exc.args[0]}"
            ) from exc

    def to_json(self) -> str:
        return dumps_dict(self.to_dict())

    @classmethod
    def from_json(cls, payload: str) -> "ExperimentManifest":
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ContractValidationError(
                f"ExperimentManifest invalid JSON: {exc}"
            ) from exc
        return cls.from_dict(data)


__all__ = [
    "DialogueHarnessConfig",
    "ExperimentManifest",
    "MemoryConfig",
    "ModelConfig",
    "RerankerConfig",
    "RetrievalConfig",
]
