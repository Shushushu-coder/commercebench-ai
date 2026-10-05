"""Deterministic Identity-vs-treatment reranker comparison (Phase 2E).

Pure reporting layer over already-derived ``RerankerCaseAnalysis``
objects plus the two traces' actual generation outputs. No model calls,
no file access, no core-schema changes, no overall score, no
significance testing — only deterministic counts and rates over the
development cases.

A comparison is only valid for a controlled pair: same case, same query,
same corpus identity, same retriever identity, identical candidate
rankings (IDs exact, scores within float tolerance), and the same
``final_top_k``. Anything else is rejected instead of producing a
misleading "reranker delta".
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from commercebench.contracts.errors import ContractValidationError

from .reranker_analysis import REPORT_VERSION, RerankerCaseAnalysis

SCORE_TOLERANCE_REL = 1e-9
SCORE_TOLERANCE_ABS = 1e-12


def _require_non_empty_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractValidationError(
            f"{field_name} must be a non-empty string"
        )
    return value


def _optional_float(value: Any, field_name: str) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractValidationError(f"{field_name} must be a number or null")
    result = float(value)
    if not math.isfinite(result):
        raise ContractValidationError(f"{field_name} must be finite or null")
    return result


def _optional_str(value: Any, field_name: str) -> Optional[str]:
    if value is None:
        return None
    return _require_non_empty_str(value, field_name)


@dataclass(frozen=True)
class RerankerComparison:
    """Controlled pairwise reranker comparison for one case.

    ``baseline`` is the control run (normally the Identity reranker) and
    ``treatment`` the run under study (normally CrossEncoder). Deltas are
    ``treatment - baseline``. ``generation_changed`` compares the traces'
    actual generation outputs, never inferred from ``top1_changed``.
    """

    report_version: str = "0.1"

    case_id: str = ""
    final_top_k: int = 0

    baseline_reranker_id: str = ""
    baseline_reranker_version: str = ""
    treatment_reranker_id: str = ""
    treatment_reranker_version: str = ""

    baseline_final_top1: Optional[str] = None
    treatment_final_top1: Optional[str] = None
    top1_changed: bool = False

    baseline_mrr: Optional[float] = None
    treatment_mrr: Optional[float] = None
    mrr_delta: Optional[float] = None

    baseline_recall_at_k: Optional[float] = None
    treatment_recall_at_k: Optional[float] = None
    recall_delta: Optional[float] = None

    treatment_relevant_rank_improved: bool = False
    treatment_relevant_rank_regressed: bool = False

    generation_changed: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "report_version",
            _require_non_empty_str(self.report_version, "report_version"),
        )
        object.__setattr__(
            self, "case_id", _require_non_empty_str(self.case_id, "case_id")
        )
        if (
            isinstance(self.final_top_k, bool)
            or not isinstance(self.final_top_k, int)
            or self.final_top_k <= 0
        ):
            raise ContractValidationError(
                "final_top_k must be a positive integer"
            )
        for field_name in (
            "baseline_reranker_id",
            "baseline_reranker_version",
            "treatment_reranker_id",
            "treatment_reranker_version",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_non_empty_str(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self,
            "baseline_final_top1",
            _optional_str(self.baseline_final_top1, "baseline_final_top1"),
        )
        object.__setattr__(
            self,
            "treatment_final_top1",
            _optional_str(self.treatment_final_top1, "treatment_final_top1"),
        )
        for field_name in (
            "baseline_mrr",
            "treatment_mrr",
            "mrr_delta",
            "baseline_recall_at_k",
            "treatment_recall_at_k",
            "recall_delta",
        ):
            object.__setattr__(
                self,
                field_name,
                _optional_float(getattr(self, field_name), field_name),
            )
        for field_name in (
            "top1_changed",
            "treatment_relevant_rank_improved",
            "treatment_relevant_rank_regressed",
            "generation_changed",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise ContractValidationError(f"{field_name} must be a boolean")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_version": self.report_version,
            "case_id": self.case_id,
            "final_top_k": self.final_top_k,
            "baseline_reranker_id": self.baseline_reranker_id,
            "baseline_reranker_version": self.baseline_reranker_version,
            "treatment_reranker_id": self.treatment_reranker_id,
            "treatment_reranker_version": self.treatment_reranker_version,
            "baseline_final_top1": self.baseline_final_top1,
            "treatment_final_top1": self.treatment_final_top1,
            "top1_changed": self.top1_changed,
            "baseline_mrr": self.baseline_mrr,
            "treatment_mrr": self.treatment_mrr,
            "mrr_delta": self.mrr_delta,
            "baseline_recall_at_k": self.baseline_recall_at_k,
            "treatment_recall_at_k": self.treatment_recall_at_k,
            "recall_delta": self.recall_delta,
            "treatment_relevant_rank_improved": (
                self.treatment_relevant_rank_improved
            ),
            "treatment_relevant_rank_regressed": (
                self.treatment_relevant_rank_regressed
            ),
            "generation_changed": self.generation_changed,
        }

    @classmethod
    def from_dict(cls, data: Any) -> "RerankerComparison":
        if not isinstance(data, dict):
            raise ContractValidationError("RerankerComparison must be a mapping")
        try:
            return cls(
                report_version=data.get("report_version", "0.1"),
                case_id=data["case_id"],
                final_top_k=data["final_top_k"],
                baseline_reranker_id=data["baseline_reranker_id"],
                baseline_reranker_version=data["baseline_reranker_version"],
                treatment_reranker_id=data["treatment_reranker_id"],
                treatment_reranker_version=data["treatment_reranker_version"],
                baseline_final_top1=data.get("baseline_final_top1"),
                treatment_final_top1=data.get("treatment_final_top1"),
                top1_changed=data.get("top1_changed", False),
                baseline_mrr=data.get("baseline_mrr"),
                treatment_mrr=data.get("treatment_mrr"),
                mrr_delta=data.get("mrr_delta"),
                baseline_recall_at_k=data.get("baseline_recall_at_k"),
                treatment_recall_at_k=data.get("treatment_recall_at_k"),
                recall_delta=data.get("recall_delta"),
                treatment_relevant_rank_improved=data.get(
                    "treatment_relevant_rank_improved", False
                ),
                treatment_relevant_rank_regressed=data.get(
                    "treatment_relevant_rank_regressed", False
                ),
                generation_changed=data.get("generation_changed", False),
            )
        except KeyError as exc:
            raise ContractValidationError(
                f"RerankerComparison missing required field: {exc.args[0]}"
            ) from exc


def compare_reranker_runs(
    baseline: RerankerCaseAnalysis,
    treatment: RerankerCaseAnalysis,
    baseline_output_text: str,
    treatment_output_text: str,
) -> RerankerComparison:
    """Compare a control run against a treatment run for the same case.

    Compatibility (same case/query/corpus/retriever/candidates/final_top_k)
    is verified first; any mismatch raises ``ContractValidationError``.
    Generation change is decided from the traces' actual outputs, which the
    caller passes explicitly so this function stays a pure function of its
    arguments.
    """
    for name, value in (
        ("baseline", baseline),
        ("treatment", treatment),
    ):
        if not isinstance(value, RerankerCaseAnalysis):
            raise ContractValidationError(
                f"{name} must be a RerankerCaseAnalysis"
            )
    if not isinstance(baseline_output_text, str):
        raise ContractValidationError("baseline_output_text must be a string")
    if not isinstance(treatment_output_text, str):
        raise ContractValidationError("treatment_output_text must be a string")

    if baseline.case_id != treatment.case_id:
        raise ContractValidationError(
            "reranker comparison requires the same case_id "
            f"(got {baseline.case_id!r} vs {treatment.case_id!r})"
        )
    for field_name in (
        "query",
        "corpus_id",
        "corpus_version",
        "retriever_id",
        "retriever_version",
        "final_top_k",
    ):
        if getattr(baseline, field_name) != getattr(treatment, field_name):
            raise ContractValidationError(
                "reranker comparison is invalid: "
                f"{field_name} differs between runs "
                f"({getattr(baseline, field_name)!r} vs "
                f"{getattr(treatment, field_name)!r})"
            )
    if tuple(baseline.candidate_document_ids) != tuple(
        treatment.candidate_document_ids
    ):
        raise ContractValidationError(
            "reranker comparison is invalid: candidate rankings differ, "
            "so this is not a pure reranker comparison"
        )
    if len(baseline.candidate_scores) != len(treatment.candidate_scores):
        raise ContractValidationError(
            "reranker comparison is invalid: candidate score lists differ"
        )
    for index, (left, right) in enumerate(
        zip(baseline.candidate_scores, treatment.candidate_scores)
    ):
        if not math.isclose(
            left, right, rel_tol=SCORE_TOLERANCE_REL, abs_tol=SCORE_TOLERANCE_ABS
        ):
            raise ContractValidationError(
                "reranker comparison is invalid: candidate scores differ "
                f"at position {index} ({left} vs {right}), so this is not "
                "a pure reranker comparison"
            )

    baseline_top1 = baseline.final_top1
    treatment_top1 = treatment.final_top1

    if baseline.final_mrr is not None and treatment.final_mrr is not None:
        mrr_delta: Optional[float] = treatment.final_mrr - baseline.final_mrr
    else:
        mrr_delta = None
    if (
        baseline.final_recall_at_k is not None
        and treatment.final_recall_at_k is not None
    ):
        recall_delta: Optional[float] = (
            treatment.final_recall_at_k - baseline.final_recall_at_k
        )
    else:
        recall_delta = None

    return RerankerComparison(
        report_version=REPORT_VERSION,
        case_id=baseline.case_id,
        final_top_k=baseline.final_top_k,
        baseline_reranker_id=baseline.reranker_id,
        baseline_reranker_version=baseline.reranker_version,
        treatment_reranker_id=treatment.reranker_id,
        treatment_reranker_version=treatment.reranker_version,
        baseline_final_top1=baseline_top1,
        treatment_final_top1=treatment_top1,
        top1_changed=baseline_top1 != treatment_top1,
        baseline_mrr=baseline.final_mrr,
        treatment_mrr=treatment.final_mrr,
        mrr_delta=mrr_delta,
        baseline_recall_at_k=baseline.final_recall_at_k,
        treatment_recall_at_k=treatment.final_recall_at_k,
        recall_delta=recall_delta,
        treatment_relevant_rank_improved=treatment.relevant_rank_improved,
        treatment_relevant_rank_regressed=treatment.relevant_rank_regressed,
        generation_changed=baseline_output_text != treatment_output_text,
    )


@dataclass(frozen=True)
class RerankerComparisonSummary:
    """Deterministic experiment-level counts over pairwise comparisons.

    Rates are ``count / applicable denominator``: top-1/generation rates
    divide by all compared cases, MRR rates divide by relevance-applicable
    cases only (both runs report a non-null MRR). No overall score, no
    significance testing — behavior-difference description only.
    """

    report_version: str = "0.1"

    case_ids: Tuple[str, ...] = ()
    case_count: int = 0

    top1_changed_count: int = 0
    top1_change_rate: Optional[float] = None

    generation_changed_count: int = 0
    generation_change_rate: Optional[float] = None

    relevant_rank_improved_count: int = 0
    relevant_rank_regressed_count: int = 0

    mrr_applicable_count: int = 0
    mrr_improved_count: int = 0
    mrr_regressed_count: int = 0
    mrr_unchanged_count: int = 0
    mrr_improved_rate: Optional[float] = None
    mrr_regressed_rate: Optional[float] = None
    mrr_unchanged_rate: Optional[float] = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "report_version",
            _require_non_empty_str(self.report_version, "report_version"),
        )
        ids = tuple(self.case_ids)
        for index, item in enumerate(ids):
            if not isinstance(item, str) or not item:
                raise ContractValidationError(
                    f"case_ids[{index}] must be a non-empty string"
                )
        if len(set(ids)) != len(ids):
            raise ContractValidationError("case_ids must be unique")
        object.__setattr__(self, "case_ids", ids)
        for field_name in (
            "case_count",
            "top1_changed_count",
            "generation_changed_count",
            "relevant_rank_improved_count",
            "relevant_rank_regressed_count",
            "mrr_applicable_count",
            "mrr_improved_count",
            "mrr_regressed_count",
            "mrr_unchanged_count",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ContractValidationError(
                    f"{field_name} must be a non-negative integer"
                )
        for field_name in (
            "top1_change_rate",
            "generation_change_rate",
            "mrr_improved_rate",
            "mrr_regressed_rate",
            "mrr_unchanged_rate",
        ):
            value = getattr(self, field_name)
            if value is not None:
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not 0.0 <= float(value) <= 1.0
                ):
                    raise ContractValidationError(
                        f"{field_name} must be a rate in [0, 1] or null"
                    )
                object.__setattr__(self, field_name, float(value))
        if self.case_count != len(ids):
            raise ContractValidationError(
                "case_count must equal len(case_ids)"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_version": self.report_version,
            "case_ids": list(self.case_ids),
            "case_count": self.case_count,
            "top1_changed_count": self.top1_changed_count,
            "top1_change_rate": self.top1_change_rate,
            "generation_changed_count": self.generation_changed_count,
            "generation_change_rate": self.generation_change_rate,
            "relevant_rank_improved_count": self.relevant_rank_improved_count,
            "relevant_rank_regressed_count": self.relevant_rank_regressed_count,
            "mrr_applicable_count": self.mrr_applicable_count,
            "mrr_improved_count": self.mrr_improved_count,
            "mrr_regressed_count": self.mrr_regressed_count,
            "mrr_unchanged_count": self.mrr_unchanged_count,
            "mrr_improved_rate": self.mrr_improved_rate,
            "mrr_regressed_rate": self.mrr_regressed_rate,
            "mrr_unchanged_rate": self.mrr_unchanged_rate,
        }

    @classmethod
    def from_dict(cls, data: Any) -> "RerankerComparisonSummary":
        if not isinstance(data, dict):
            raise ContractValidationError(
                "RerankerComparisonSummary must be a mapping"
            )
        try:
            return cls(
                report_version=data.get("report_version", "0.1"),
                case_ids=data.get("case_ids", ()),
                case_count=data["case_count"],
                top1_changed_count=data.get("top1_changed_count", 0),
                top1_change_rate=data.get("top1_change_rate"),
                generation_changed_count=data.get(
                    "generation_changed_count", 0
                ),
                generation_change_rate=data.get("generation_change_rate"),
                relevant_rank_improved_count=data.get(
                    "relevant_rank_improved_count", 0
                ),
                relevant_rank_regressed_count=data.get(
                    "relevant_rank_regressed_count", 0
                ),
                mrr_applicable_count=data.get("mrr_applicable_count", 0),
                mrr_improved_count=data.get("mrr_improved_count", 0),
                mrr_regressed_count=data.get("mrr_regressed_count", 0),
                mrr_unchanged_count=data.get("mrr_unchanged_count", 0),
                mrr_improved_rate=data.get("mrr_improved_rate"),
                mrr_regressed_rate=data.get("mrr_regressed_rate"),
                mrr_unchanged_rate=data.get("mrr_unchanged_rate"),
            )
        except KeyError as exc:
            raise ContractValidationError(
                "RerankerComparisonSummary missing required field: "
                f"{exc.args[0]}"
            ) from exc


def _rate(numerator: int, denominator: int) -> Optional[float]:
    if denominator <= 0:
        return None
    return numerator / denominator


def summarize_reranker_comparison(
    comparisons: Sequence[RerankerComparison],
) -> RerankerComparisonSummary:
    """Aggregate pairwise comparisons into deterministic counts/rates."""
    if not isinstance(comparisons, (list, tuple)):
        raise ContractValidationError("comparisons must be a list")
    items: List[RerankerComparison] = []
    for index, item in enumerate(comparisons):
        if not isinstance(item, RerankerComparison):
            try:
                item = RerankerComparison.from_dict(item)
            except ContractValidationError:
                raise
            except Exception as exc:
                raise ContractValidationError(
                    f"comparisons[{index}] is invalid: {exc}"
                ) from exc
        items.append(item)

    case_ids = tuple(sorted(c.case_id for c in items))
    if len(set(case_ids)) != len(items):
        raise ContractValidationError(
            "comparisons must cover distinct cases"
        )
    top1_changed = sum(1 for c in items if c.top1_changed)
    generation_changed = sum(1 for c in items if c.generation_changed)
    improved = sum(1 for c in items if c.treatment_relevant_rank_improved)
    regressed = sum(1 for c in items if c.treatment_relevant_rank_regressed)

    applicable = [
        c
        for c in items
        if c.baseline_mrr is not None and c.treatment_mrr is not None
    ]
    mrr_improved = sum(
        1 for c in applicable if c.treatment_mrr > c.baseline_mrr  # type: ignore[operator]
    )
    mrr_regressed = sum(
        1 for c in applicable if c.treatment_mrr < c.baseline_mrr  # type: ignore[operator]
    )
    mrr_unchanged = sum(
        1 for c in applicable if c.treatment_mrr == c.baseline_mrr  # type: ignore[operator]
    )

    total = len(items)
    applicable_count = len(applicable)
    return RerankerComparisonSummary(
        report_version=REPORT_VERSION,
        case_ids=case_ids,
        case_count=total,
        top1_changed_count=top1_changed,
        top1_change_rate=_rate(top1_changed, total),
        generation_changed_count=generation_changed,
        generation_change_rate=_rate(generation_changed, total),
        relevant_rank_improved_count=improved,
        relevant_rank_regressed_count=regressed,
        mrr_applicable_count=applicable_count,
        mrr_improved_count=mrr_improved,
        mrr_regressed_count=mrr_regressed,
        mrr_unchanged_count=mrr_unchanged,
        mrr_improved_rate=_rate(mrr_improved, applicable_count),
        mrr_regressed_rate=_rate(mrr_regressed, applicable_count),
        mrr_unchanged_rate=_rate(mrr_unchanged, applicable_count),
    )


__all__ = [
    "RerankerComparison",
    "RerankerComparisonSummary",
    "compare_reranker_runs",
    "summarize_reranker_comparison",
]
