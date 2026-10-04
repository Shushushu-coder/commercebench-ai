"""EvaluationResult: how one run was judged (V0 contract).

Evaluation is separate from RunTrace: the trace records what happened,
this contract records how an evaluator scored it. There is intentionally
no single overall score; each metric stands on its own and ``passed`` is
an explicit evaluator decision.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union

from .common import (
    dumps_dict,
    expect_mapping,
    require_bool,
    require_iso8601,
    require_json_dict,
    require_non_empty_str,
    str_tuple,
)
from .errors import ContractValidationError

MetricValue = Union[int, float, bool]


@dataclass(frozen=True)
class MetricResult:
    """One named metric. ``passed=None`` means the metric is informational."""

    name: str
    value: MetricValue
    passed: Optional[bool] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "name", require_non_empty_str(self.name, "name")
        )
        if not isinstance(self.value, (int, float, bool)):
            raise ContractValidationError(
                f"MetricResult.value must be int, float, or bool "
                f"(got {type(self.value).__name__})"
            )
        if self.passed is not None:
            object.__setattr__(
                self, "passed", require_bool(self.passed, "passed")
            )
        object.__setattr__(
            self, "details", dict(require_json_dict(self.details, "details"))
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "passed": self.passed,
            "details": dict(self.details),
        }

    @classmethod
    def from_dict(cls, data: Any) -> "MetricResult":
        data = expect_mapping(data, "MetricResult")
        try:
            return cls(
                name=data["name"],
                value=data["value"],
                passed=data.get("passed"),
                details=data.get("details", {}),
            )
        except KeyError as exc:
            raise ContractValidationError(
                f"MetricResult missing required field: {exc.args[0]}"
            ) from exc


@dataclass(frozen=True)
class EvaluationResult:
    """The outcome of evaluating one RunTrace for one case."""

    schema_version: str

    evaluation_id: str
    run_id: str
    case_id: str

    evaluator_id: str
    evaluator_version: str

    metrics: Dict[str, MetricResult]

    passed: bool
    failure_categories: Tuple[str, ...] = ()

    evaluated_at: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "schema_version",
            require_non_empty_str(self.schema_version, "schema_version"),
        )
        object.__setattr__(
            self,
            "evaluation_id",
            require_non_empty_str(self.evaluation_id, "evaluation_id"),
        )
        object.__setattr__(
            self, "run_id", require_non_empty_str(self.run_id, "run_id")
        )
        object.__setattr__(
            self, "case_id", require_non_empty_str(self.case_id, "case_id")
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
        if not isinstance(self.metrics, dict):
            raise ContractValidationError("metrics must be a mapping")
        normalized: Dict[str, MetricResult] = {}
        for key, metric in self.metrics.items():
            require_non_empty_str(key, "metrics key")
            if not isinstance(metric, MetricResult):
                metric = MetricResult.from_dict(metric)
            if metric.name != key:
                raise ContractValidationError(
                    f"metrics key {key!r} does not match metric name "
                    f"{metric.name!r}"
                )
            normalized[key] = metric
        object.__setattr__(self, "metrics", normalized)
        object.__setattr__(
            self, "passed", require_bool(self.passed, "passed")
        )
        object.__setattr__(
            self,
            "failure_categories",
            str_tuple(self.failure_categories, "failure_categories"),
        )
        object.__setattr__(
            self,
            "evaluated_at",
            require_iso8601(self.evaluated_at, "evaluated_at"),
        )
        object.__setattr__(
            self, "metadata", dict(require_json_dict(self.metadata, "metadata"))
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "evaluation_id": self.evaluation_id,
            "run_id": self.run_id,
            "case_id": self.case_id,
            "evaluator_id": self.evaluator_id,
            "evaluator_version": self.evaluator_version,
            "metrics": {k: m.to_dict() for k, m in self.metrics.items()},
            "passed": self.passed,
            "failure_categories": list(self.failure_categories),
            "evaluated_at": self.evaluated_at,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Any) -> "EvaluationResult":
        data = expect_mapping(data, "EvaluationResult")
        try:
            return cls(
                schema_version=data["schema_version"],
                evaluation_id=data["evaluation_id"],
                run_id=data["run_id"],
                case_id=data["case_id"],
                evaluator_id=data["evaluator_id"],
                evaluator_version=data["evaluator_version"],
                metrics=data["metrics"],
                passed=data["passed"],
                failure_categories=data.get("failure_categories", ()),
                evaluated_at=data["evaluated_at"],
                metadata=data.get("metadata", {}),
            )
        except KeyError as exc:
            raise ContractValidationError(
                f"EvaluationResult missing required field: {exc.args[0]}"
            ) from exc

    def to_json(self) -> str:
        return dumps_dict(self.to_dict())

    @classmethod
    def from_json(cls, payload: str) -> "EvaluationResult":
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ContractValidationError(
                f"EvaluationResult invalid JSON: {exc}"
            ) from exc
        return cls.from_dict(data)


__all__ = ["EvaluationResult", "MetricResult", "MetricValue"]
