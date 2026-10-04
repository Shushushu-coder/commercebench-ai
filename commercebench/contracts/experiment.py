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


@dataclass(frozen=True)
class ExperimentManifest:
    """Describes one experiment: which cases, which system, which evaluator.

    ``config_fingerprint()`` covers exactly the fields that can change run
    outputs: schema/benchmark versions, the ordered case list, system and
    evaluator identities and versions, model/retrieval/memory configs, and
    the random seed. ``experiment_id``, ``experiment_name`` and ``metadata``
    are descriptive identity fields and are excluded by design; ``case_ids``
    order is significant and preserved in the fingerprint.
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
    memory: Optional[MemoryConfig] = None

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
        if self.memory is not None and not isinstance(self.memory, MemoryConfig):
            object.__setattr__(self, "memory", MemoryConfig.from_dict(self.memory))
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
        return {
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
            "memory": self.memory.to_dict() if self.memory is not None else None,
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
                memory=data.get("memory"),
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
    "ExperimentManifest",
    "MemoryConfig",
    "ModelConfig",
    "RetrievalConfig",
]
