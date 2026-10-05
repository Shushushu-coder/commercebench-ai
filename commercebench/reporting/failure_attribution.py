"""Deterministic failure attribution reporting (Phase 2G).

This module is a pure reporting layer over persisted benchmark evidence::

    EvaluationResult (formal outcomes)
    + CaseSpec (binary relevance)
    + RunTrace (candidate window + final ranking)
    -> FailureAttributionReport (JSON-compatible derived view)

Hard separation (never merged into one enum)::

    formal outcome != failure attribution != diagnostic observation

Formal outcomes remain exactly::

    EXACT_MATCH_FAILED
    REQUIRED_FACT_MISSING
    FORBIDDEN_CLAIM_PRESENT
    RETRIEVAL_MISS

``RETRIEVAL_MISS`` is the only core final-state retrieval failure. This
module never modifies ``EvaluationResult.failure_categories``, never adds
a core failure enum, never changes the ``RETRIEVAL_MISS`` predicate, and
performs no model calls.

Reporting-only semantics defined here (NOT core failure categories)::

    FailureStage: retrieval / reranker / generation / unknown
    RetrievalMissOrigin: retrieval_originated / reranker_induced / unknown

Same-depth window rule (hard contract)::

    candidate_window = candidate_document_ids[:final_top_k]

State predicates (for ``relevant_document_ids != ()``)::

    candidate_hit = bool(relevant ∩ candidate_window)
    final_hit     = bool(relevant ∩ final_documents)

    State A (False, False) -> RETRIEVAL_ORIGINATED
    State B (True, False)  -> RERANKER_INDUCED (only deterministic
                              reranker-induced final-miss predicate)
    State C (False, True)  -> reranker rescue *observation*, not attribution
    State D (True, True)   -> no retrieval failure

Attribution exists only when the formal miss exists: if
``"RETRIEVAL_MISS"`` is absent from ``evaluation.failure_categories``
then ``retrieval_miss_attribution`` is ``None``, even when diagnostic
evidence looks like State C/D. The core ``EvaluationResult`` stays the
formal outcome source of truth.

Fail-closed rules:

- Formal ``RETRIEVAL_MISS`` present but ``final_hit is True`` -> reject
  (``EvaluationResult`` and ``RunTrace`` contradict each other).
- ``relevant_document_ids == ()`` but formal ``RETRIEVAL_MISS`` present
  -> reject (the evaluator never emits a miss without relevance).
- ``evaluation.passed != (len(failure_categories) == 0)`` -> reject.
- Unknown formal failure strings -> reject.

Legacy traces (single final/legacy event, no candidate event) with a
formal miss attribute ``origin = UNKNOWN`` and
``evidence_complete = False`` without guessing. The formal outcome stays
valid.

Phase 2E reuse: rank movement, top-k membership deltas, MRR/Recall
deltas, and cutoff computation live in ``RerankerCaseAnalysis``. This
module consumes that analysis for ``cutoff_margin`` numeric evidence
when available and derives same-depth attribution predicates
(``candidate_window`` vs final) as its own semantic layer. Raw scores
are never compared across stages and no threshold observation is
produced from ``cutoff_margin``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

from commercebench.contracts.case import CaseSpec
from commercebench.contracts.errors import ContractValidationError
from commercebench.contracts.evaluation import EvaluationResult
from commercebench.contracts.trace import RetrievalRole, RunTrace
from commercebench.evaluation.deterministic import (
    FAILURE_EXACT_MATCH,
    FAILURE_FORBIDDEN_CLAIM,
    FAILURE_REQUIRED_FACT,
)
from commercebench.evaluation.retrieval import FAILURE_RETRIEVAL_MISS

from .reranker_analysis import RerankerCaseAnalysis, analyze_reranker_case
from .reranker_comparison import RerankerComparison

REPORT_VERSION = "0.1"

KNOWN_FORMAL_FAILURES: Tuple[str, ...] = (
    FAILURE_EXACT_MATCH,
    FAILURE_REQUIRED_FACT,
    FAILURE_FORBIDDEN_CLAIM,
    FAILURE_RETRIEVAL_MISS,
)

GENERATION_FAILURES: Tuple[str, ...] = (
    FAILURE_EXACT_MATCH,
    FAILURE_REQUIRED_FACT,
    FAILURE_FORBIDDEN_CLAIM,
)


class FailureStage(str, Enum):
    """Reporting-only stage for one formal failure (not a core category)."""

    RETRIEVAL = "retrieval"
    RERANKER = "reranker"
    GENERATION = "generation"
    UNKNOWN = "unknown"


class RetrievalMissOrigin(str, Enum):
    """Reporting-only origin for a formal ``RETRIEVAL_MISS``."""

    RETRIEVAL_ORIGINATED = "retrieval_originated"
    RERANKER_INDUCED = "reranker_induced"
    UNKNOWN = "unknown"


class DiagnosticObservationKind(str, Enum):
    """Reporting-only diagnostic observations (never formal failures)."""

    RERANKER_RESCUED_RELEVANT = "RERANKER_RESCUED_RELEVANT"
    RELEVANT_RANK_IMPROVED = "RELEVANT_RANK_IMPROVED"
    RELEVANT_RANK_REGRESSED = "RELEVANT_RANK_REGRESSED"
    RELEVANT_ENTERED_FINAL_K = "RELEVANT_ENTERED_FINAL_K"
    RELEVANT_EXITED_FINAL_K = "RELEVANT_EXITED_FINAL_K"
    TOP1_CHANGED = "TOP1_CHANGED"
    GENERATION_CHANGED = "GENERATION_CHANGED"


# Canonical deterministic ordering for observations (single-run never
# emits GENERATION_CHANGED; pairwise helper may).
_OBSERVATION_ORDER: Tuple[str, ...] = (
    DiagnosticObservationKind.RERANKER_RESCUED_RELEVANT.value,
    DiagnosticObservationKind.RELEVANT_RANK_IMPROVED.value,
    DiagnosticObservationKind.RELEVANT_RANK_REGRESSED.value,
    DiagnosticObservationKind.RELEVANT_ENTERED_FINAL_K.value,
    DiagnosticObservationKind.RELEVANT_EXITED_FINAL_K.value,
    DiagnosticObservationKind.TOP1_CHANGED.value,
    DiagnosticObservationKind.GENERATION_CHANGED.value,
)

_OBSERVATION_SET = frozenset(_OBSERVATION_ORDER)


def _require_non_empty_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractValidationError(
            f"{field_name} must be a non-empty string"
        )
    return value


def _id_tuple(value: Any, field_name: str) -> Tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ContractValidationError(f"{field_name} must be a list of strings")
    items = tuple(value)
    for index, item in enumerate(items):
        if not isinstance(item, str) or not item:
            raise ContractValidationError(
                f"{field_name}[{index}] must be a non-empty string"
            )
    return items


def _optional_int(value: Any, field_name: str) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractValidationError(
            f"{field_name} must be an integer or null"
        )
    if value <= 0:
        raise ContractValidationError(
            f"{field_name} must be a positive integer or null"
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


def _parse_stage(value: Any, field_name: str) -> FailureStage:
    if isinstance(value, FailureStage):
        return value
    if isinstance(value, str):
        try:
            return FailureStage(value)
        except ValueError:
            pass
    raise ContractValidationError(
        f"{field_name} must be one of "
        f"{sorted(item.value for item in FailureStage)}"
    )


def _parse_origin(value: Any, field_name: str) -> RetrievalMissOrigin:
    if isinstance(value, RetrievalMissOrigin):
        return value
    if isinstance(value, str):
        try:
            return RetrievalMissOrigin(value)
        except ValueError:
            pass
    raise ContractValidationError(
        f"{field_name} must be one of "
        f"{sorted(item.value for item in RetrievalMissOrigin)}"
    )


def _best_rank(
    ranking: Sequence[str], relevant: Sequence[str]
) -> Optional[int]:
    relevant_set = set(relevant)
    for index, document_id in enumerate(ranking, start=1):
        if document_id in relevant_set:
            return index
    return None


def _read_final_top_k(trace: RunTrace) -> int:
    metadata = trace.runtime_metadata
    reranker_meta = metadata.get("reranker") if isinstance(metadata, dict) else None
    candidates: List[Any] = []
    if isinstance(reranker_meta, dict):
        candidates.append(reranker_meta.get("final_top_k"))
    if isinstance(metadata, dict):
        candidates.append(metadata.get("top_k"))
    for value in candidates:
        if isinstance(value, bool):
            continue
        if isinstance(value, int) and value > 0:
            return value
    raise ContractValidationError(
        "failure attribution requires a positive final_top_k in "
        "runtime_metadata['reranker']['final_top_k'] (or top-level 'top_k')"
    )


def _sort_observations(values: Sequence[str]) -> Tuple[str, ...]:
    order = {name: index for index, name in enumerate(_OBSERVATION_ORDER)}
    unique = sorted(set(values), key=lambda name: order.get(name, len(order)))
    return tuple(unique)


@dataclass(frozen=True)
class FailureEvidence:
    """Machine-readable evidence supporting one retrieval attribution.

    No document text is stored — IDs, ranks, and the numeric
    ``cutoff_margin`` (raw-logit difference, never a calibrated
    confidence measure) only.
    """

    relevant_document_ids: Tuple[str, ...] = ()
    candidate_window_document_ids: Tuple[str, ...] = ()
    final_document_ids: Tuple[str, ...] = ()
    candidate_best_relevant_rank: Optional[int] = None
    final_best_relevant_rank: Optional[int] = None
    final_top_k: int = 0
    cutoff_margin: Optional[float] = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "relevant_document_ids",
            _id_tuple(self.relevant_document_ids, "relevant_document_ids"),
        )
        object.__setattr__(
            self,
            "candidate_window_document_ids",
            _id_tuple(
                self.candidate_window_document_ids,
                "candidate_window_document_ids",
            ),
        )
        object.__setattr__(
            self,
            "final_document_ids",
            _id_tuple(self.final_document_ids, "final_document_ids"),
        )
        object.__setattr__(
            self,
            "candidate_best_relevant_rank",
            _optional_int(
                self.candidate_best_relevant_rank,
                "candidate_best_relevant_rank",
            ),
        )
        object.__setattr__(
            self,
            "final_best_relevant_rank",
            _optional_int(
                self.final_best_relevant_rank, "final_best_relevant_rank"
            ),
        )
        if (
            isinstance(self.final_top_k, bool)
            or not isinstance(self.final_top_k, int)
            or self.final_top_k <= 0
        ):
            raise ContractValidationError(
                "final_top_k must be a positive integer"
            )
        object.__setattr__(
            self,
            "cutoff_margin",
            _optional_float(self.cutoff_margin, "cutoff_margin"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "relevant_document_ids": list(self.relevant_document_ids),
            "candidate_window_document_ids": list(
                self.candidate_window_document_ids
            ),
            "final_document_ids": list(self.final_document_ids),
            "candidate_best_relevant_rank": self.candidate_best_relevant_rank,
            "final_best_relevant_rank": self.final_best_relevant_rank,
            "final_top_k": self.final_top_k,
            "cutoff_margin": self.cutoff_margin,
        }

    @classmethod
    def from_dict(cls, data: Any) -> "FailureEvidence":
        if not isinstance(data, dict):
            raise ContractValidationError("FailureEvidence must be a mapping")
        try:
            return cls(
                relevant_document_ids=data.get("relevant_document_ids", ()),
                candidate_window_document_ids=data.get(
                    "candidate_window_document_ids", ()
                ),
                final_document_ids=data.get("final_document_ids", ()),
                candidate_best_relevant_rank=data.get(
                    "candidate_best_relevant_rank"
                ),
                final_best_relevant_rank=data.get("final_best_relevant_rank"),
                final_top_k=data["final_top_k"],
                cutoff_margin=data.get("cutoff_margin"),
            )
        except KeyError as exc:
            raise ContractValidationError(
                f"FailureEvidence missing required field: {exc.args[0]}"
            ) from exc


@dataclass(frozen=True)
class RetrievalMissAttribution:
    """Derived attribution for one formal ``RETRIEVAL_MISS``.

    Reporting-only: never written back into
    ``EvaluationResult.failure_categories`` and never affecting
    ``EvaluationResult.passed``.
    """

    formal_failure: str = FAILURE_RETRIEVAL_MISS
    stage: FailureStage = FailureStage.UNKNOWN
    origin: RetrievalMissOrigin = RetrievalMissOrigin.UNKNOWN

    relevant_document_ids: Tuple[str, ...] = ()
    candidate_window_document_ids: Tuple[str, ...] = ()
    final_document_ids: Tuple[str, ...] = ()

    candidate_hit: Optional[bool] = None
    final_hit: bool = False

    candidate_relevant_document_ids: Tuple[str, ...] = ()
    final_relevant_document_ids: Tuple[str, ...] = ()

    candidate_best_relevant_rank: Optional[int] = None
    final_best_relevant_rank: Optional[int] = None

    final_top_k: int = 0
    cutoff_margin: Optional[float] = None
    evidence_complete: bool = False

    def __post_init__(self) -> None:
        if self.formal_failure != FAILURE_RETRIEVAL_MISS:
            raise ContractValidationError(
                "RetrievalMissAttribution.formal_failure must be "
                f"{FAILURE_RETRIEVAL_MISS!r}"
            )
        object.__setattr__(
            self, "stage", _parse_stage(self.stage, "stage")
        )
        object.__setattr__(
            self, "origin", _parse_origin(self.origin, "origin")
        )
        object.__setattr__(
            self,
            "relevant_document_ids",
            _id_tuple(self.relevant_document_ids, "relevant_document_ids"),
        )
        object.__setattr__(
            self,
            "candidate_window_document_ids",
            _id_tuple(
                self.candidate_window_document_ids,
                "candidate_window_document_ids",
            ),
        )
        object.__setattr__(
            self,
            "final_document_ids",
            _id_tuple(self.final_document_ids, "final_document_ids"),
        )
        if self.candidate_hit is not None and not isinstance(
            self.candidate_hit, bool
        ):
            raise ContractValidationError(
                "candidate_hit must be a boolean or null"
            )
        if not isinstance(self.final_hit, bool):
            raise ContractValidationError("final_hit must be a boolean")
        object.__setattr__(
            self,
            "candidate_relevant_document_ids",
            _id_tuple(
                self.candidate_relevant_document_ids,
                "candidate_relevant_document_ids",
            ),
        )
        object.__setattr__(
            self,
            "final_relevant_document_ids",
            _id_tuple(
                self.final_relevant_document_ids,
                "final_relevant_document_ids",
            ),
        )
        object.__setattr__(
            self,
            "candidate_best_relevant_rank",
            _optional_int(
                self.candidate_best_relevant_rank,
                "candidate_best_relevant_rank",
            ),
        )
        object.__setattr__(
            self,
            "final_best_relevant_rank",
            _optional_int(
                self.final_best_relevant_rank, "final_best_relevant_rank"
            ),
        )
        if (
            isinstance(self.final_top_k, bool)
            or not isinstance(self.final_top_k, int)
            or self.final_top_k <= 0
        ):
            raise ContractValidationError(
                "final_top_k must be a positive integer"
            )
        object.__setattr__(
            self,
            "cutoff_margin",
            _optional_float(self.cutoff_margin, "cutoff_margin"),
        )
        if not isinstance(self.evidence_complete, bool):
            raise ContractValidationError("evidence_complete must be a boolean")

        relevant_set = set(self.relevant_document_ids)
        window_set = set(self.candidate_window_document_ids)
        final_set = set(self.final_document_ids)
        # Internal consistency: hits and membership lists must match the
        # stored ID lists (fail closed on corrupt deserialization).
        expected_candidate_relevant = tuple(
            d
            for d in self.candidate_window_document_ids
            if d in relevant_set
        )
        if tuple(self.candidate_relevant_document_ids) != (
            expected_candidate_relevant
        ):
            raise ContractValidationError(
                "candidate_relevant_document_ids must list the relevant "
                "IDs of candidate_window_document_ids in window order"
            )
        expected_final_relevant = tuple(
            d for d in self.final_document_ids if d in relevant_set
        )
        if tuple(self.final_relevant_document_ids) != expected_final_relevant:
            raise ContractValidationError(
                "final_relevant_document_ids must list the relevant IDs "
                "of final_document_ids in final order"
            )
        if self.evidence_complete:
            if self.candidate_hit is None:
                raise ContractValidationError(
                    "candidate_hit must be a boolean when evidence_complete "
                    "is True"
                )
            if self.candidate_hit != bool(relevant_set & window_set):
                raise ContractValidationError(
                    "candidate_hit contradicts candidate_window evidence"
                )
            if self.origin is RetrievalMissOrigin.UNKNOWN:
                raise ContractValidationError(
                    "origin must be retrieval_originated or reranker_induced "
                    "when evidence_complete is True"
                )
            expected_stage = (
                FailureStage.RETRIEVAL
                if self.origin is RetrievalMissOrigin.RETRIEVAL_ORIGINATED
                else FailureStage.RERANKER
            )
            if self.stage is not expected_stage:
                raise ContractValidationError(
                    "stage must match origin for complete evidence"
                )
        else:
            if self.origin is not RetrievalMissOrigin.UNKNOWN:
                raise ContractValidationError(
                    "origin must be unknown when evidence_complete is False"
                )
            if self.stage is not FailureStage.UNKNOWN:
                raise ContractValidationError(
                    "stage must be unknown when evidence_complete is False"
                )
            if self.candidate_hit is not None:
                raise ContractValidationError(
                    "candidate_hit must be null when evidence_complete "
                    "is False"
                )
            if self.candidate_window_document_ids != ():
                raise ContractValidationError(
                    "candidate_window_document_ids must be empty when "
                    "evidence_complete is False"
                )
            if self.candidate_relevant_document_ids != ():
                raise ContractValidationError(
                    "candidate_relevant_document_ids must be empty when "
                    "evidence_complete is False"
                )
            if self.candidate_best_relevant_rank is not None:
                raise ContractValidationError(
                    "candidate_best_relevant_rank must be null when "
                    "evidence_complete is False"
                )
            if self.cutoff_margin is not None:
                raise ContractValidationError(
                    "cutoff_margin must be null when evidence_complete "
                    "is False"
                )
        if self.final_hit != bool(relevant_set & final_set):
            raise ContractValidationError(
                "final_hit contradicts final ranking evidence"
            )
        if _best_rank(
            self.candidate_window_document_ids, self.relevant_document_ids
        ) != self.candidate_best_relevant_rank:
            raise ContractValidationError(
                "candidate_best_relevant_rank contradicts candidate window"
            )
        if _best_rank(
            self.final_document_ids, self.relevant_document_ids
        ) != self.final_best_relevant_rank:
            raise ContractValidationError(
                "final_best_relevant_rank contradicts final ranking"
            )

    def to_evidence(self) -> FailureEvidence:
        """Return the machine-readable evidence view of this attribution."""
        return FailureEvidence(
            relevant_document_ids=self.relevant_document_ids,
            candidate_window_document_ids=self.candidate_window_document_ids,
            final_document_ids=self.final_document_ids,
            candidate_best_relevant_rank=self.candidate_best_relevant_rank,
            final_best_relevant_rank=self.final_best_relevant_rank,
            final_top_k=self.final_top_k,
            cutoff_margin=self.cutoff_margin,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "formal_failure": self.formal_failure,
            "stage": self.stage.value,
            "origin": self.origin.value,
            "relevant_document_ids": list(self.relevant_document_ids),
            "candidate_window_document_ids": list(
                self.candidate_window_document_ids
            ),
            "final_document_ids": list(self.final_document_ids),
            "candidate_hit": self.candidate_hit,
            "final_hit": self.final_hit,
            "candidate_relevant_document_ids": list(
                self.candidate_relevant_document_ids
            ),
            "final_relevant_document_ids": list(
                self.final_relevant_document_ids
            ),
            "candidate_best_relevant_rank": self.candidate_best_relevant_rank,
            "final_best_relevant_rank": self.final_best_relevant_rank,
            "final_top_k": self.final_top_k,
            "cutoff_margin": self.cutoff_margin,
            "evidence_complete": self.evidence_complete,
        }

    @classmethod
    def from_dict(cls, data: Any) -> "RetrievalMissAttribution":
        if not isinstance(data, dict):
            raise ContractValidationError(
                "RetrievalMissAttribution must be a mapping"
            )
        try:
            return cls(
                formal_failure=data.get(
                    "formal_failure", FAILURE_RETRIEVAL_MISS
                ),
                stage=data.get("stage", FailureStage.UNKNOWN.value),
                origin=data.get("origin", RetrievalMissOrigin.UNKNOWN.value),
                relevant_document_ids=data.get("relevant_document_ids", ()),
                candidate_window_document_ids=data.get(
                    "candidate_window_document_ids", ()
                ),
                final_document_ids=data.get("final_document_ids", ()),
                candidate_hit=data.get("candidate_hit"),
                final_hit=data["final_hit"],
                candidate_relevant_document_ids=data.get(
                    "candidate_relevant_document_ids", ()
                ),
                final_relevant_document_ids=data.get(
                    "final_relevant_document_ids", ()
                ),
                candidate_best_relevant_rank=data.get(
                    "candidate_best_relevant_rank"
                ),
                final_best_relevant_rank=data.get("final_best_relevant_rank"),
                final_top_k=data["final_top_k"],
                cutoff_margin=data.get("cutoff_margin"),
                evidence_complete=data["evidence_complete"],
            )
        except KeyError as exc:
            raise ContractValidationError(
                "RetrievalMissAttribution missing required field: "
                f"{exc.args[0]}"
            ) from exc


@dataclass(frozen=True)
class FailureAttribution:
    """One formal failure mapped to its reporting stage.

    Multi-label: one case emits one entry per formal failure — never a
    single forced ``root_cause``. Generation failures map to
    ``stage = generation`` directly from
    ``EvaluationResult.failure_categories`` without re-implementing
    generation predicates.
    """

    formal_failure: str
    stage: FailureStage
    origin: Optional[RetrievalMissOrigin] = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "formal_failure",
            _require_non_empty_str(self.formal_failure, "formal_failure"),
        )
        if self.formal_failure not in KNOWN_FORMAL_FAILURES:
            raise ContractValidationError(
                f"formal_failure {self.formal_failure!r} is not a known "
                f"formal outcome {list(KNOWN_FORMAL_FAILURES)}"
            )
        object.__setattr__(
            self, "stage", _parse_stage(self.stage, "stage")
        )
        if self.origin is not None:
            object.__setattr__(
                self, "origin", _parse_origin(self.origin, "origin")
            )
        if self.formal_failure == FAILURE_RETRIEVAL_MISS:
            if self.origin is None:
                raise ContractValidationError(
                    "origin is required for RETRIEVAL_MISS attribution"
                )
            expected = {
                RetrievalMissOrigin.RETRIEVAL_ORIGINATED: (
                    FailureStage.RETRIEVAL
                ),
                RetrievalMissOrigin.RERANKER_INDUCED: FailureStage.RERANKER,
                RetrievalMissOrigin.UNKNOWN: FailureStage.UNKNOWN,
            }[self.origin]
            if self.stage is not expected:
                raise ContractValidationError(
                    "stage must match origin for RETRIEVAL_MISS"
                )
        else:
            if self.stage is not FailureStage.GENERATION:
                raise ContractValidationError(
                    "generation failures must map to stage=generation"
                )
            if self.origin is not None:
                raise ContractValidationError(
                    "origin must be null for generation failures"
                )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "formal_failure": self.formal_failure,
            "stage": self.stage.value,
            "origin": self.origin.value if self.origin is not None else None,
        }

    @classmethod
    def from_dict(cls, data: Any) -> "FailureAttribution":
        if not isinstance(data, dict):
            raise ContractValidationError(
                "FailureAttribution must be a mapping"
            )
        try:
            return cls(
                formal_failure=data["formal_failure"],
                stage=data["stage"],
                origin=data.get("origin"),
            )
        except KeyError as exc:
            raise ContractValidationError(
                f"FailureAttribution missing required field: {exc.args[0]}"
            ) from exc


def _attribution_tuple(value: Any) -> Tuple[FailureAttribution, ...]:
    if not isinstance(value, (list, tuple)):
        raise ContractValidationError("attributions must be a list")
    items: List[FailureAttribution] = []
    for index, item in enumerate(value):
        if not isinstance(item, FailureAttribution):
            try:
                item = FailureAttribution.from_dict(item)
            except ContractValidationError:
                raise
            except Exception as exc:
                raise ContractValidationError(
                    f"attributions[{index}] is invalid: {exc}"
                ) from exc
        items.append(item)
    return tuple(items)


def _observation_tuple(value: Any) -> Tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ContractValidationError("diagnostic_observations must be a list")
    items = tuple(value)
    for index, item in enumerate(items):
        if not isinstance(item, str) or item not in _OBSERVATION_SET:
            raise ContractValidationError(
                f"diagnostic_observations[{index}] must be one of "
                f"{sorted(_OBSERVATION_SET)}"
            )
    return _sort_observations(items)


@dataclass(frozen=True)
class FailureAttributionReport:
    """Derived reporting view for one evaluated run.

    ``passed`` mirrors ``EvaluationResult.passed`` exactly (never
    redefined). ``formal_failures`` mirrors
    ``EvaluationResult.failure_categories`` in evaluation order.
    ``attributions`` holds one stage entry per formal failure.
    ``retrieval_miss_attribution`` carries the evidence-rich retrieval
    derivation when (and only when) the formal miss exists.
    ``diagnostic_observations`` are reporting-only facts (rank movement,
    top-k membership, top-1 change) — never formal failures.
    ``cutoff_margin`` is numeric evidence reused from Phase 2E, never a
    thresholded failure signal.
    """

    report_version: str = REPORT_VERSION
    case_id: str = ""
    formal_failures: Tuple[str, ...] = ()
    passed: bool = True
    attributions: Tuple[FailureAttribution, ...] = ()
    retrieval_miss_attribution: Optional[RetrievalMissAttribution] = None
    diagnostic_observations: Tuple[str, ...] = ()
    cutoff_margin: Optional[float] = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "report_version",
            _require_non_empty_str(self.report_version, "report_version"),
        )
        object.__setattr__(
            self, "case_id", _require_non_empty_str(self.case_id, "case_id")
        )
        failures = tuple(self.formal_failures)
        for index, item in enumerate(failures):
            if not isinstance(item, str) or not item:
                raise ContractValidationError(
                    f"formal_failures[{index}] must be a non-empty string"
                )
            if item not in KNOWN_FORMAL_FAILURES:
                raise ContractValidationError(
                    f"formal_failures[{index}] {item!r} is not a known "
                    "formal outcome"
                )
        if len(set(failures)) != len(failures):
            raise ContractValidationError(
                "formal_failures must not contain duplicates"
            )
        object.__setattr__(self, "formal_failures", failures)
        if not isinstance(self.passed, bool):
            raise ContractValidationError("passed must be a boolean")
        if self.passed != (len(failures) == 0):
            raise ContractValidationError(
                "passed must equal (len(formal_failures) == 0)"
            )
        object.__setattr__(
            self, "attributions", _attribution_tuple(self.attributions)
        )
        if len(self.attributions) != len(failures):
            raise ContractValidationError(
                "attributions must cover each formal failure exactly once"
            )
        if tuple(a.formal_failure for a in self.attributions) != failures:
            raise ContractValidationError(
                "attributions must follow formal_failures order exactly"
            )
        attribution = self.retrieval_miss_attribution
        if attribution is not None and not isinstance(
            attribution, RetrievalMissAttribution
        ):
            try:
                attribution = RetrievalMissAttribution.from_dict(attribution)
            except ContractValidationError:
                raise
            except Exception as exc:
                raise ContractValidationError(
                    f"retrieval_miss_attribution is invalid: {exc}"
                ) from exc
            object.__setattr__(
                self, "retrieval_miss_attribution", attribution
            )
        has_miss = FAILURE_RETRIEVAL_MISS in failures
        if has_miss and attribution is None:
            raise ContractValidationError(
                "retrieval_miss_attribution is required when RETRIEVAL_MISS "
                "is a formal failure"
            )
        if not has_miss and attribution is not None:
            raise ContractValidationError(
                "retrieval_miss_attribution must be null when RETRIEVAL_MISS "
                "is not a formal failure"
            )
        object.__setattr__(
            self,
            "diagnostic_observations",
            _observation_tuple(self.diagnostic_observations),
        )
        if DiagnosticObservationKind.GENERATION_CHANGED.value in tuple(
            self.diagnostic_observations
        ):
            raise ContractValidationError(
                "GENERATION_CHANGED is a pairwise observation and must not "
                "appear in a single-run attribution report"
            )
        object.__setattr__(
            self,
            "cutoff_margin",
            _optional_float(self.cutoff_margin, "cutoff_margin"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_version": self.report_version,
            "case_id": self.case_id,
            "formal_failures": list(self.formal_failures),
            "passed": self.passed,
            "attributions": [a.to_dict() for a in self.attributions],
            "retrieval_miss_attribution": (
                self.retrieval_miss_attribution.to_dict()
                if self.retrieval_miss_attribution is not None
                else None
            ),
            "diagnostic_observations": list(self.diagnostic_observations),
            "cutoff_margin": self.cutoff_margin,
        }

    @classmethod
    def from_dict(cls, data: Any) -> "FailureAttributionReport":
        if not isinstance(data, dict):
            raise ContractValidationError(
                "FailureAttributionReport must be a mapping"
            )
        try:
            return cls(
                report_version=data.get("report_version", REPORT_VERSION),
                case_id=data["case_id"],
                formal_failures=tuple(data.get("formal_failures", ())),
                passed=data["passed"],
                attributions=data.get("attributions", ()),
                retrieval_miss_attribution=data.get(
                    "retrieval_miss_attribution"
                ),
                diagnostic_observations=data.get(
                    "diagnostic_observations", ()
                ),
                cutoff_margin=data.get("cutoff_margin"),
            )
        except KeyError as exc:
            raise ContractValidationError(
                "FailureAttributionReport missing required field: "
                f"{exc.args[0]}"
            ) from exc


def _derive_single_run_observations(
    relevant: Tuple[str, ...],
    candidate_window: Optional[Tuple[str, ...]],
    final_ids: Tuple[str, ...],
    candidate_top1: Optional[str],
    final_top1: Optional[str],
) -> Tuple[str, ...]:
    """Derive same-depth diagnostic observations (never formal failures)."""
    found: List[str] = []
    if candidate_window is None:
        return ()
    relevant_set = set(relevant)
    candidate_relevant = tuple(d for d in candidate_window if d in relevant_set)
    final_relevant = tuple(d for d in final_ids if d in relevant_set)
    candidate_hit = bool(candidate_relevant) if relevant else False
    final_hit = bool(final_relevant) if relevant else False
    if relevant:
        if not candidate_hit and final_hit:
            found.append(
                DiagnosticObservationKind.RERANKER_RESCUED_RELEVANT.value
            )
        candidate_best = _best_rank(candidate_window, relevant)
        final_best = _best_rank(final_ids, relevant)
        if (
            candidate_best is not None
            and final_best is not None
            and final_best < candidate_best
        ):
            found.append(
                DiagnosticObservationKind.RELEVANT_RANK_IMPROVED.value
            )
        if (
            candidate_best is not None
            and final_best is not None
            and final_best > candidate_best
        ):
            found.append(
                DiagnosticObservationKind.RELEVANT_RANK_REGRESSED.value
            )
        if set(final_relevant) - set(candidate_relevant):
            found.append(
                DiagnosticObservationKind.RELEVANT_ENTERED_FINAL_K.value
            )
        if set(candidate_relevant) - set(final_relevant):
            found.append(
                DiagnosticObservationKind.RELEVANT_EXITED_FINAL_K.value
            )
    if (
        candidate_top1 is not None
        and final_top1 is not None
        and candidate_top1 != final_top1
    ):
        found.append(DiagnosticObservationKind.TOP1_CHANGED.value)
    return _sort_observations(found)


def analyze_failure_attribution(
    case: CaseSpec,
    trace: RunTrace,
    evaluation: EvaluationResult,
    reranker_analysis: Optional[RerankerCaseAnalysis] = None,
) -> FailureAttributionReport:
    """Derive the failure attribution report for one evaluated run.

    Pure function of ``(case, trace, evaluation[, reranker_analysis])``:
    no file access, no model calls, no global state. Every inconsistency
    (identity mismatch, broken ``passed`` invariant, unknown formal
    failure, formal-miss/final-hit contradiction, no-relevance formal
    miss, ambiguous candidate evidence) raises
    ``ContractValidationError`` instead of producing a best-effort
    report.

    When ``reranker_analysis`` is omitted and the trace carries reranker
    diagnostics, it is constructed internally via ``analyze_reranker_case``
    (itself model-free) purely to reuse the ``cutoff_margin`` numeric
    evidence. Traces without diagnostics (legacy single-event runs) are
    handled as ``UNKNOWN``/incomplete evidence without guessing.
    """
    if not isinstance(case, CaseSpec):
        raise ContractValidationError("case must be a CaseSpec")
    if not isinstance(trace, RunTrace):
        raise ContractValidationError("trace must be a RunTrace")
    if not isinstance(evaluation, EvaluationResult):
        raise ContractValidationError("evaluation must be an EvaluationResult")
    if trace.case_id != case.case_id:
        raise ContractValidationError(
            f"trace case_id {trace.case_id!r} does not match "
            f"case {case.case_id!r}"
        )
    if evaluation.case_id != case.case_id:
        raise ContractValidationError(
            f"evaluation case_id {evaluation.case_id!r} does not match "
            f"case {case.case_id!r}"
        )
    if evaluation.run_id != trace.run_id:
        raise ContractValidationError(
            "evaluation run_id does not match trace run_id"
        )
    if evaluation.passed != (len(evaluation.failure_categories) == 0):
        raise ContractValidationError(
            "evaluation passed invariant broken: passed must equal "
            "(len(failure_categories) == 0)"
        )

    formal_failures = tuple(evaluation.failure_categories)
    for item in formal_failures:
        if item not in KNOWN_FORMAL_FAILURES:
            raise ContractValidationError(
                f"unknown formal failure {item!r}: Phase 2G preserves the "
                "four formal outcomes only"
            )

    relevant = tuple(case.relevant_document_ids)
    has_miss = FAILURE_RETRIEVAL_MISS in formal_failures
    if not relevant and has_miss:
        raise ContractValidationError(
            "formal RETRIEVAL_MISS with empty relevant_document_ids: "
            "the evaluator never emits a miss without relevance"
        )

    candidate_events = [
        e for e in trace.retrieval_events if e.role is RetrievalRole.CANDIDATE
    ]
    if len(candidate_events) > 1:
        raise ContractValidationError(
            "failure attribution requires at most one role=candidate "
            f"retrieval event (found {len(candidate_events)})"
        )
    has_candidate = len(candidate_events) == 1
    final_ids = tuple(trace.final_ranked_document_ids)

    analysis = reranker_analysis
    if analysis is not None and not isinstance(
        analysis, RerankerCaseAnalysis
    ):
        raise ContractValidationError(
            "reranker_analysis must be a RerankerCaseAnalysis or null"
        )
    if analysis is not None and analysis.case_id != case.case_id:
        raise ContractValidationError(
            "reranker_analysis case_id does not match case"
        )

    if analysis is not None:
        final_top_k = analysis.final_top_k
        if has_candidate:
            candidate_ids = tuple(candidate_events[0].document_ids)
            if tuple(analysis.candidate_document_ids) != candidate_ids:
                raise ContractValidationError(
                    "reranker_analysis candidate ranking does not match trace"
                )
            if tuple(analysis.final_document_ids) != final_ids[
                : len(analysis.final_document_ids)
            ] or len(final_ids) != len(analysis.final_document_ids):
                # Final ranking must match the analysis final ranking
                # exactly (single-final traces); concatenated multi-final
                # traces are out of scope for reuse.
                raise ContractValidationError(
                    "reranker_analysis final ranking does not match trace"
                )
    else:
        try:
            final_top_k = _read_final_top_k(trace)
        except ContractValidationError:
            if final_ids:
                final_top_k = len(final_ids)
            elif has_candidate:
                candidate_len = len(candidate_events[0].document_ids)
                final_top_k = candidate_len if candidate_len > 0 else 1
                # Still fail closed when no depth evidence exists at all.
                if candidate_len == 0:
                    raise
            else:
                raise

    cutoff_margin: Optional[float] = None
    retrieval_attribution: Optional[RetrievalMissAttribution] = None
    observations: Tuple[str, ...] = ()

    if has_candidate:
        assert len(candidate_events) == 1
        candidate_ids = tuple(candidate_events[0].document_ids)
        candidate_window = tuple(candidate_ids[:final_top_k])
        relevant_set = set(relevant)
        candidate_relevant = tuple(
            d for d in candidate_window if d in relevant_set
        )
        final_relevant = tuple(d for d in final_ids if d in relevant_set)
        candidate_hit = bool(candidate_relevant) if relevant else False
        final_hit = bool(final_relevant) if relevant else False
        candidate_best = (
            _best_rank(candidate_window, relevant) if relevant else None
        )
        final_best = _best_rank(final_ids, relevant) if relevant else None
        candidate_top1 = candidate_ids[0] if candidate_ids else None
        final_top1 = final_ids[0] if final_ids else None

        if analysis is None:
            try:
                analysis = analyze_reranker_case(case, trace, None)
            except ContractValidationError:
                analysis = None
        if analysis is not None:
            cutoff_margin = analysis.cutoff_margin

        observations = _derive_single_run_observations(
            relevant,
            candidate_window,
            final_ids,
            candidate_top1,
            final_top1,
        )

        if has_miss:
            # Fail closed on formal-miss/final-hit contradiction (§10).
            if final_hit:
                raise ContractValidationError(
                    "formal RETRIEVAL_MISS contradicts final ranking "
                    "evidence (a relevant document is present in final)"
                )
            if candidate_hit is False and final_hit is False:
                origin = RetrievalMissOrigin.RETRIEVAL_ORIGINATED
                stage = FailureStage.RETRIEVAL
            elif candidate_hit is True and final_hit is False:
                origin = RetrievalMissOrigin.RERANKER_INDUCED
                stage = FailureStage.RERANKER
            else:  # pragma: no cover — final_hit True already rejected
                raise ContractValidationError(
                    "unreachable retrieval state with formal RETRIEVAL_MISS"
                )
            retrieval_attribution = RetrievalMissAttribution(
                formal_failure=FAILURE_RETRIEVAL_MISS,
                stage=stage,
                origin=origin,
                relevant_document_ids=relevant,
                candidate_window_document_ids=candidate_window,
                final_document_ids=final_ids,
                candidate_hit=candidate_hit,
                final_hit=final_hit,
                candidate_relevant_document_ids=candidate_relevant,
                final_relevant_document_ids=final_relevant,
                candidate_best_relevant_rank=candidate_best,
                final_best_relevant_rank=final_best,
                final_top_k=final_top_k,
                cutoff_margin=cutoff_margin,
                evidence_complete=True,
            )
    else:
        # Legacy trace: no candidate evidence (§11).
        relevant_set = set(relevant)
        final_relevant = tuple(d for d in final_ids if d in relevant_set)
        final_hit = bool(final_relevant) if relevant else False
        final_best = _best_rank(final_ids, relevant) if relevant else None
        observations = ()
        cutoff_margin = None
        if has_miss:
            if final_hit:
                raise ContractValidationError(
                    "formal RETRIEVAL_MISS contradicts final ranking "
                    "evidence (a relevant document is present in final)"
                )
            retrieval_attribution = RetrievalMissAttribution(
                formal_failure=FAILURE_RETRIEVAL_MISS,
                stage=FailureStage.UNKNOWN,
                origin=RetrievalMissOrigin.UNKNOWN,
                relevant_document_ids=relevant,
                candidate_window_document_ids=(),
                final_document_ids=final_ids,
                candidate_hit=None,
                final_hit=final_hit,
                candidate_relevant_document_ids=(),
                final_relevant_document_ids=final_relevant,
                candidate_best_relevant_rank=None,
                final_best_relevant_rank=final_best,
                final_top_k=final_top_k,
                cutoff_margin=None,
                evidence_complete=False,
            )

    attributions: List[FailureAttribution] = []
    for formal in formal_failures:
        if formal == FAILURE_RETRIEVAL_MISS:
            assert retrieval_attribution is not None
            attributions.append(
                FailureAttribution(
                    formal_failure=formal,
                    stage=retrieval_attribution.stage,
                    origin=retrieval_attribution.origin,
                )
            )
        else:
            attributions.append(
                FailureAttribution(
                    formal_failure=formal,
                    stage=FailureStage.GENERATION,
                    origin=None,
                )
            )

    return FailureAttributionReport(
        report_version=REPORT_VERSION,
        case_id=case.case_id,
        formal_failures=formal_failures,
        passed=evaluation.passed,
        attributions=tuple(attributions),
        retrieval_miss_attribution=retrieval_attribution,
        diagnostic_observations=observations,
        cutoff_margin=cutoff_margin,
    )


def derive_comparison_observations(
    comparison: RerankerComparison,
) -> Tuple[str, ...]:
    """Map an existing ``RerankerComparison`` to reporting observations.

    Pure mapping of already-derived facts — never recomputes the
    pairwise comparison. Emits ``TOP1_CHANGED`` and
    ``GENERATION_CHANGED`` (pairwise-only) plus relevant-rank movement
    when the treatment run moved the best relevant rank.
    """
    if not isinstance(comparison, RerankerComparison):
        raise ContractValidationError(
            "comparison must be a RerankerComparison"
        )
    found: List[str] = []
    if comparison.top1_changed:
        found.append(DiagnosticObservationKind.TOP1_CHANGED.value)
    if comparison.generation_changed:
        found.append(DiagnosticObservationKind.GENERATION_CHANGED.value)
    if comparison.treatment_relevant_rank_improved:
        found.append(
            DiagnosticObservationKind.RELEVANT_RANK_IMPROVED.value
        )
    if comparison.treatment_relevant_rank_regressed:
        found.append(
            DiagnosticObservationKind.RELEVANT_RANK_REGRESSED.value
        )
    return _sort_observations(found)


def render_failure_attribution_summary(
    report: FailureAttributionReport,
) -> str:
    """Render a small deterministic human-readable summary.

    Convenience view only; the typed ``FailureAttributionReport``
    remains the source of truth.
    """
    lines = [
        f"# Failure attribution: {report.case_id}",
        "",
        f"passed: {report.passed}",
        f"formal_failures: {list(report.formal_failures) or []}",
        "",
        "## Attributions",
        "",
    ]
    if not report.attributions:
        lines.append("- (none)")
    for attribution in report.attributions:
        origin = (
            attribution.origin.value
            if attribution.origin is not None
            else "-"
        )
        lines.append(
            f"- {attribution.formal_failure}: stage={attribution.stage.value} "
            f"origin={origin}"
        )
    lines += ["", "## Retrieval miss evidence", ""]
    detail = report.retrieval_miss_attribution
    if detail is None:
        lines.append("- (none)")
    else:
        lines.append(f"- origin: {detail.origin.value}")
        lines.append(f"- stage: {detail.stage.value}")
        lines.append(f"- evidence_complete: {detail.evidence_complete}")
        lines.append(
            f"- relevant: {list(detail.relevant_document_ids) or []}"
        )
        lines.append(
            "- candidate_window: "
            f"{list(detail.candidate_window_document_ids) or []}"
        )
        lines.append(
            f"- final: {list(detail.final_document_ids) or []}"
        )
        lines.append(f"- candidate_hit: {detail.candidate_hit}")
        lines.append(f"- final_hit: {detail.final_hit}")
        lines.append(
            f"- candidate_best_relevant_rank: "
            f"{detail.candidate_best_relevant_rank}"
        )
        lines.append(
            f"- final_best_relevant_rank: {detail.final_best_relevant_rank}"
        )
        lines.append(f"- final_top_k: {detail.final_top_k}")
        lines.append(f"- cutoff_margin: {detail.cutoff_margin}")
    lines += ["", "## Diagnostic observations", ""]
    if not report.diagnostic_observations:
        lines.append("- (none)")
    for observation in report.diagnostic_observations:
        lines.append(f"- {observation}")
    lines.append("")
    lines.append(f"report_version: {report.report_version}")
    return "\n".join(lines) + "\n"


__all__ = [
    "REPORT_VERSION",
    "KNOWN_FORMAL_FAILURES",
    "GENERATION_FAILURES",
    "DiagnosticObservationKind",
    "FailureAttribution",
    "FailureAttributionReport",
    "FailureEvidence",
    "FailureStage",
    "RetrievalMissAttribution",
    "RetrievalMissOrigin",
    "analyze_failure_attribution",
    "derive_comparison_observations",
    "render_failure_attribution_summary",
]
