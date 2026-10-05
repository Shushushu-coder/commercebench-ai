"""Deterministic single-run reranker failure analysis (Phase 2E).

This module is a pure reporting layer over persisted benchmark evidence.
It performs no model calls, loads no weights, reads no files, and changes
no core schema (``RunTrace``, ``EvaluationResult``, ``ExperimentManifest``
are inputs only, never mutated).

Evidence flow per case::

    CaseSpec (binary relevance)
    + RunTrace (candidate event, final event, reranker.full_ranking)
    + optional EvaluationResult (cross-check only)
    -> RerankerCaseAnalysis (JSON-compatible derived view)

Rank semantics (also stated in the README)::

    candidate_rank = 1-based rank in the Hybrid RRF candidate ranking
    reranker_rank  = 1-based position in the full reranker ranking
    rank_delta     = candidate_rank - reranker_rank

so ``rank_delta > 0`` means the document moved **up**, ``0`` means
unchanged, and ``< 0`` means it moved **down**. Ranks are comparable
across stages; raw scores are not: ``candidate_score`` is an RRF fusion
score while ``reranker_score`` is a cross-encoder logit, so this module
never subtracts or otherwise combines scores across stages.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from commercebench.contracts.case import CaseSpec
from commercebench.contracts.errors import ContractValidationError
from commercebench.contracts.evaluation import EvaluationResult
from commercebench.contracts.trace import RetrievalEvent, RetrievalRole, RunTrace
from commercebench.evaluation.retrieval import (
    mean_reciprocal_rank,
    recall_at_k,
)

REPORT_VERSION = "0.1"


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


def _score_tuple(value: Any, field_name: str) -> Tuple[float, ...]:
    if not isinstance(value, (list, tuple)):
        raise ContractValidationError(f"{field_name} must be a list of numbers")
    items = tuple(value)
    for index, item in enumerate(items):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ContractValidationError(
                f"{field_name}[{index}] must be a number"
            )
        if not math.isfinite(float(item)):
            raise ContractValidationError(
                f"{field_name}[{index}] must be finite"
            )
    return tuple(float(item) for item in items)


def _optional_float(value: Any, field_name: str) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractValidationError(f"{field_name} must be a number or null")
    result = float(value)
    if not math.isfinite(result):
        raise ContractValidationError(f"{field_name} must be finite or null")
    return result


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


def _optional_str(value: Any, field_name: str) -> Optional[str]:
    if value is None:
        return None
    return _require_non_empty_str(value, field_name)


@dataclass(frozen=True)
class RankMovement:
    """Per-document candidate -> reranker rank movement.

    ``rank_delta = candidate_rank - reranker_rank``: positive means the
    document moved up in the reranker ordering. ``candidate_score`` (RRF)
    and ``reranker_score`` (cross-encoder logit) are reported side by
    side for evidence and never subtracted.
    """

    document_id: str
    candidate_rank: int
    reranker_rank: int
    rank_delta: int
    candidate_score: float
    reranker_score: float
    relevant: bool
    candidate_in_final_cutoff: bool
    final_in_final_cutoff: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "document_id",
            _require_non_empty_str(self.document_id, "document_id"),
        )
        for field_name in ("candidate_rank", "reranker_rank"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ContractValidationError(
                    f"{field_name} must be a positive integer"
                )
        if (
            isinstance(self.rank_delta, bool)
            or not isinstance(self.rank_delta, int)
            or self.rank_delta != self.candidate_rank - self.reranker_rank
        ):
            raise ContractValidationError(
                "rank_delta must equal candidate_rank - reranker_rank"
            )
        for field_name in ("candidate_score", "reranker_score"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ContractValidationError(f"{field_name} must be a number")
            if not math.isfinite(float(value)):
                raise ContractValidationError(f"{field_name} must be finite")
            object.__setattr__(self, field_name, float(value))
        for field_name in (
            "relevant",
            "candidate_in_final_cutoff",
            "final_in_final_cutoff",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise ContractValidationError(f"{field_name} must be a boolean")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "candidate_rank": self.candidate_rank,
            "reranker_rank": self.reranker_rank,
            "rank_delta": self.rank_delta,
            "candidate_score": self.candidate_score,
            "reranker_score": self.reranker_score,
            "relevant": self.relevant,
            "candidate_in_final_cutoff": self.candidate_in_final_cutoff,
            "final_in_final_cutoff": self.final_in_final_cutoff,
        }

    @classmethod
    def from_dict(cls, data: Any) -> "RankMovement":
        if not isinstance(data, dict):
            raise ContractValidationError("RankMovement must be a mapping")
        try:
            return cls(
                document_id=data["document_id"],
                candidate_rank=data["candidate_rank"],
                reranker_rank=data["reranker_rank"],
                rank_delta=data["rank_delta"],
                candidate_score=data["candidate_score"],
                reranker_score=data["reranker_score"],
                relevant=data["relevant"],
                candidate_in_final_cutoff=data["candidate_in_final_cutoff"],
                final_in_final_cutoff=data["final_in_final_cutoff"],
            )
        except KeyError as exc:
            raise ContractValidationError(
                f"RankMovement missing required field: {exc.args[0]}"
            ) from exc


def _movement_tuple(value: Any) -> Tuple[RankMovement, ...]:
    if not isinstance(value, (list, tuple)):
        raise ContractValidationError("rank_movements must be a list")
    items = []
    for index, item in enumerate(value):
        if not isinstance(item, RankMovement):
            try:
                item = RankMovement.from_dict(item)
            except ContractValidationError:
                raise
            except Exception as exc:
                raise ContractValidationError(
                    f"rank_movements[{index}] is invalid: {exc}"
                ) from exc
        items.append(item)
    return tuple(items)


@dataclass(frozen=True)
class RerankerCaseAnalysis:
    """Derived single-run reranker diagnosis for one case.

    A read-only view over ``(CaseSpec, RunTrace[, EvaluationResult])``.
    ``*_mrr``/``*_recall_at_k`` are reporting-derived statistics computed
    with the same helpers as the retrieval evaluator; they are never
    written back into ``EvaluationResult.metrics``. ``None`` on a rank or
    metric field means "not applicable / not present" — never zero-as-rank.
    """

    report_version: str = "0.1"

    case_id: str = ""
    query: str = ""
    reranker_id: str = ""
    reranker_version: str = ""

    corpus_id: str = ""
    corpus_version: str = ""
    retriever_id: str = ""
    retriever_version: str = ""

    candidate_count: int = 0
    final_top_k: int = 0

    candidate_document_ids: Tuple[str, ...] = ()
    candidate_scores: Tuple[float, ...] = ()
    full_reranker_document_ids: Tuple[str, ...] = ()
    full_reranker_scores: Tuple[float, ...] = ()
    final_document_ids: Tuple[str, ...] = ()
    final_scores: Tuple[float, ...] = ()

    relevant_document_ids: Tuple[str, ...] = ()

    rank_movements: Tuple[RankMovement, ...] = ()

    candidate_relevant_rank: Optional[int] = None
    reranker_relevant_rank: Optional[int] = None
    final_relevant_rank: Optional[int] = None

    candidate_mrr: Optional[float] = None
    final_mrr: Optional[float] = None
    mrr_delta: Optional[float] = None

    candidate_recall_at_k: Optional[float] = None
    final_recall_at_k: Optional[float] = None
    recall_delta: Optional[float] = None

    top_k_entered: Tuple[str, ...] = ()
    top_k_exited: Tuple[str, ...] = ()

    candidate_top1: Optional[str] = None
    final_top1: Optional[str] = None
    top1_changed: bool = False

    generation_driving_document_id: Optional[str] = None

    cutoff_score: Optional[float] = None
    next_below_cutoff_score: Optional[float] = None
    cutoff_margin: Optional[float] = None

    relevant_rank_improved: bool = False
    relevant_rank_regressed: bool = False
    relevant_doc_entered_final_top_k: bool = False
    relevant_doc_exited_final_top_k: bool = False
    generation_driver_changed: bool = False

    relevant_documents_in_candidate_window: Tuple[str, ...] = ()
    relevant_documents_in_final: Tuple[str, ...] = ()

    mrr_cross_check: Optional[bool] = None
    recall_cross_check: Optional[bool] = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "report_version",
            _require_non_empty_str(self.report_version, "report_version"),
        )
        object.__setattr__(
            self, "case_id", _require_non_empty_str(self.case_id, "case_id")
        )
        object.__setattr__(
            self, "query", _require_non_empty_str(self.query, "query")
        )
        object.__setattr__(
            self,
            "reranker_id",
            _require_non_empty_str(self.reranker_id, "reranker_id"),
        )
        object.__setattr__(
            self,
            "reranker_version",
            _require_non_empty_str(self.reranker_version, "reranker_version"),
        )
        for field_name in (
            "corpus_id",
            "corpus_version",
            "retriever_id",
            "retriever_version",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_non_empty_str(getattr(self, field_name), field_name),
            )
        if (
            isinstance(self.candidate_count, bool)
            or not isinstance(self.candidate_count, int)
            or self.candidate_count < 0
        ):
            raise ContractValidationError(
                "candidate_count must be a non-negative integer"
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
            "candidate_document_ids",
            _id_tuple(self.candidate_document_ids, "candidate_document_ids"),
        )
        object.__setattr__(
            self,
            "candidate_scores",
            _score_tuple(self.candidate_scores, "candidate_scores"),
        )
        object.__setattr__(
            self,
            "full_reranker_document_ids",
            _id_tuple(
                self.full_reranker_document_ids, "full_reranker_document_ids"
            ),
        )
        object.__setattr__(
            self,
            "full_reranker_scores",
            _score_tuple(self.full_reranker_scores, "full_reranker_scores"),
        )
        object.__setattr__(
            self,
            "final_document_ids",
            _id_tuple(self.final_document_ids, "final_document_ids"),
        )
        object.__setattr__(
            self,
            "final_scores",
            _score_tuple(self.final_scores, "final_scores"),
        )
        object.__setattr__(
            self,
            "relevant_document_ids",
            _id_tuple(self.relevant_document_ids, "relevant_document_ids"),
        )
        if len(self.candidate_document_ids) != len(self.candidate_scores):
            raise ContractValidationError(
                "candidate_document_ids and candidate_scores must align"
            )
        if len(self.full_reranker_document_ids) != len(
            self.full_reranker_scores
        ):
            raise ContractValidationError(
                "full_reranker_document_ids and full_reranker_scores must align"
            )
        if len(self.final_document_ids) != len(self.final_scores):
            raise ContractValidationError(
                "final_document_ids and final_scores must align"
            )
        if self.candidate_count != len(self.candidate_document_ids):
            raise ContractValidationError(
                "candidate_count must equal len(candidate_document_ids)"
            )
        object.__setattr__(
            self,
            "rank_movements",
            _movement_tuple(self.rank_movements),
        )
        for field_name in (
            "candidate_relevant_rank",
            "reranker_relevant_rank",
            "final_relevant_rank",
        ):
            object.__setattr__(
                self,
                field_name,
                _optional_int(getattr(self, field_name), field_name),
            )
        for field_name in (
            "candidate_mrr",
            "final_mrr",
            "mrr_delta",
            "candidate_recall_at_k",
            "final_recall_at_k",
            "recall_delta",
            "cutoff_score",
            "next_below_cutoff_score",
            "cutoff_margin",
        ):
            object.__setattr__(
                self,
                field_name,
                _optional_float(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self,
            "top_k_entered",
            _id_tuple(self.top_k_entered, "top_k_entered"),
        )
        object.__setattr__(
            self,
            "top_k_exited",
            _id_tuple(self.top_k_exited, "top_k_exited"),
        )
        object.__setattr__(
            self,
            "candidate_top1",
            _optional_str(self.candidate_top1, "candidate_top1"),
        )
        object.__setattr__(
            self, "final_top1", _optional_str(self.final_top1, "final_top1")
        )
        for field_name in (
            "top1_changed",
            "relevant_rank_improved",
            "relevant_rank_regressed",
            "relevant_doc_entered_final_top_k",
            "relevant_doc_exited_final_top_k",
            "generation_driver_changed",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise ContractValidationError(f"{field_name} must be a boolean")
        object.__setattr__(
            self,
            "generation_driving_document_id",
            _optional_str(
                self.generation_driving_document_id,
                "generation_driving_document_id",
            ),
        )
        object.__setattr__(
            self,
            "relevant_documents_in_candidate_window",
            _id_tuple(
                self.relevant_documents_in_candidate_window,
                "relevant_documents_in_candidate_window",
            ),
        )
        object.__setattr__(
            self,
            "relevant_documents_in_final",
            _id_tuple(
                self.relevant_documents_in_final, "relevant_documents_in_final"
            ),
        )
        for field_name in ("mrr_cross_check", "recall_cross_check"):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, bool):
                raise ContractValidationError(
                    f"{field_name} must be a boolean or null"
                )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_version": self.report_version,
            "case_id": self.case_id,
            "query": self.query,
            "reranker_id": self.reranker_id,
            "reranker_version": self.reranker_version,
            "corpus_id": self.corpus_id,
            "corpus_version": self.corpus_version,
            "retriever_id": self.retriever_id,
            "retriever_version": self.retriever_version,
            "candidate_count": self.candidate_count,
            "final_top_k": self.final_top_k,
            "candidate_document_ids": list(self.candidate_document_ids),
            "candidate_scores": list(self.candidate_scores),
            "full_reranker_document_ids": list(self.full_reranker_document_ids),
            "full_reranker_scores": list(self.full_reranker_scores),
            "final_document_ids": list(self.final_document_ids),
            "final_scores": list(self.final_scores),
            "relevant_document_ids": list(self.relevant_document_ids),
            "rank_movements": [m.to_dict() for m in self.rank_movements],
            "candidate_relevant_rank": self.candidate_relevant_rank,
            "reranker_relevant_rank": self.reranker_relevant_rank,
            "final_relevant_rank": self.final_relevant_rank,
            "candidate_mrr": self.candidate_mrr,
            "final_mrr": self.final_mrr,
            "mrr_delta": self.mrr_delta,
            "candidate_recall_at_k": self.candidate_recall_at_k,
            "final_recall_at_k": self.final_recall_at_k,
            "recall_delta": self.recall_delta,
            "top_k_entered": list(self.top_k_entered),
            "top_k_exited": list(self.top_k_exited),
            "candidate_top1": self.candidate_top1,
            "final_top1": self.final_top1,
            "top1_changed": self.top1_changed,
            "generation_driving_document_id": (
                self.generation_driving_document_id
            ),
            "cutoff_score": self.cutoff_score,
            "next_below_cutoff_score": self.next_below_cutoff_score,
            "cutoff_margin": self.cutoff_margin,
            "relevant_rank_improved": self.relevant_rank_improved,
            "relevant_rank_regressed": self.relevant_rank_regressed,
            "relevant_doc_entered_final_top_k": (
                self.relevant_doc_entered_final_top_k
            ),
            "relevant_doc_exited_final_top_k": (
                self.relevant_doc_exited_final_top_k
            ),
            "generation_driver_changed": self.generation_driver_changed,
            "relevant_documents_in_candidate_window": list(
                self.relevant_documents_in_candidate_window
            ),
            "relevant_documents_in_final": list(
                self.relevant_documents_in_final
            ),
            "mrr_cross_check": self.mrr_cross_check,
            "recall_cross_check": self.recall_cross_check,
        }

    @classmethod
    def from_dict(cls, data: Any) -> "RerankerCaseAnalysis":
        if not isinstance(data, dict):
            raise ContractValidationError("RerankerCaseAnalysis must be a mapping")
        try:
            return cls(
                report_version=data.get("report_version", "0.1"),
                case_id=data["case_id"],
                query=data["query"],
                reranker_id=data["reranker_id"],
                reranker_version=data["reranker_version"],
                corpus_id=data["corpus_id"],
                corpus_version=data["corpus_version"],
                retriever_id=data["retriever_id"],
                retriever_version=data["retriever_version"],
                candidate_count=data["candidate_count"],
                final_top_k=data["final_top_k"],
                candidate_document_ids=data.get("candidate_document_ids", ()),
                candidate_scores=data.get("candidate_scores", ()),
                full_reranker_document_ids=data.get(
                    "full_reranker_document_ids", ()
                ),
                full_reranker_scores=data.get("full_reranker_scores", ()),
                final_document_ids=data.get("final_document_ids", ()),
                final_scores=data.get("final_scores", ()),
                relevant_document_ids=data.get("relevant_document_ids", ()),
                rank_movements=data.get("rank_movements", ()),
                candidate_relevant_rank=data.get("candidate_relevant_rank"),
                reranker_relevant_rank=data.get("reranker_relevant_rank"),
                final_relevant_rank=data.get("final_relevant_rank"),
                candidate_mrr=data.get("candidate_mrr"),
                final_mrr=data.get("final_mrr"),
                mrr_delta=data.get("mrr_delta"),
                candidate_recall_at_k=data.get("candidate_recall_at_k"),
                final_recall_at_k=data.get("final_recall_at_k"),
                recall_delta=data.get("recall_delta"),
                top_k_entered=data.get("top_k_entered", ()),
                top_k_exited=data.get("top_k_exited", ()),
                candidate_top1=data.get("candidate_top1"),
                final_top1=data.get("final_top1"),
                top1_changed=data.get("top1_changed", False),
                generation_driving_document_id=data.get(
                    "generation_driving_document_id"
                ),
                cutoff_score=data.get("cutoff_score"),
                next_below_cutoff_score=data.get("next_below_cutoff_score"),
                cutoff_margin=data.get("cutoff_margin"),
                relevant_rank_improved=data.get(
                    "relevant_rank_improved", False
                ),
                relevant_rank_regressed=data.get(
                    "relevant_rank_regressed", False
                ),
                relevant_doc_entered_final_top_k=data.get(
                    "relevant_doc_entered_final_top_k", False
                ),
                relevant_doc_exited_final_top_k=data.get(
                    "relevant_doc_exited_final_top_k", False
                ),
                generation_driver_changed=data.get(
                    "generation_driver_changed", False
                ),
                relevant_documents_in_candidate_window=data.get(
                    "relevant_documents_in_candidate_window", ()
                ),
                relevant_documents_in_final=data.get(
                    "relevant_documents_in_final", ()
                ),
                mrr_cross_check=data.get("mrr_cross_check"),
                recall_cross_check=data.get("recall_cross_check"),
            )
        except KeyError as exc:
            raise ContractValidationError(
                f"RerankerCaseAnalysis missing required field: {exc.args[0]}"
            ) from exc


def _select_role_events(
    trace: RunTrace, role: RetrievalRole, field_name: str
) -> RetrievalEvent:
    matches = [e for e in trace.retrieval_events if e.role is role]
    if len(matches) != 1:
        raise ContractValidationError(
            f"reranker analysis requires exactly one {field_name} "
            f"retrieval event (found {len(matches)}); this reporting layer "
            "only supports Phase 2D+ reranked traces"
        )
    return matches[0]


def _read_full_ranking(trace: RunTrace) -> Tuple[Tuple[str, ...], Tuple[float, ...]]:
    metadata = trace.runtime_metadata
    if not isinstance(metadata, dict):
        raise ContractValidationError(
            "reranker analysis requires trace runtime_metadata"
        )
    reranker_meta = metadata.get("reranker")
    if not isinstance(reranker_meta, dict):
        raise ContractValidationError(
            "reranker analysis requires runtime_metadata['reranker'] "
            "(no reranker evidence in this trace)"
        )
    full = reranker_meta.get("full_ranking")
    if not isinstance(full, dict):
        raise ContractValidationError(
            "reranker diagnostic evidence unavailable: "
            "runtime_metadata['reranker']['full_ranking'] is missing; "
            "refusing to reconstruct rank movement from the final event"
        )
    try:
        document_ids = _id_tuple(
            full.get("document_ids"), "reranker.full_ranking.document_ids"
        )
        scores = _score_tuple(
            full.get("scores"), "reranker.full_ranking.scores"
        )
    except ContractValidationError as exc:
        raise ContractValidationError(
            f"reranker diagnostic evidence is corrupt: {exc}"
        ) from exc
    if len(document_ids) != len(scores):
        raise ContractValidationError(
            "reranker diagnostic evidence is corrupt: full_ranking "
            "document_ids and scores have different lengths"
        )
    if len(set(document_ids)) != len(document_ids):
        raise ContractValidationError(
            "reranker diagnostic evidence is corrupt: full_ranking "
            "document_ids contain duplicates"
        )
    return document_ids, scores


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
        "reranker analysis requires a positive final_top_k in "
        "runtime_metadata['reranker']['final_top_k'] (or top-level 'top_k')"
    )


def _read_str_field(
    mapping: Any, key: str, owner: str, optional: bool = False
) -> Any:
    value = mapping.get(key) if isinstance(mapping, dict) else None
    if value is None and optional:
        return None
    if not isinstance(value, str) or not value:
        raise ContractValidationError(
            f"reranker analysis requires {owner}['{key}'] "
            "to be a non-empty string"
        )
    return value


def _best_rank(
    ranking: Sequence[str], relevant: Sequence[str]
) -> Optional[int]:
    relevant_set = set(relevant)
    for index, document_id in enumerate(ranking, start=1):
        if document_id in relevant_set:
            return index
    return None


def analyze_reranker_case(
    case: CaseSpec,
    trace: RunTrace,
    evaluation: Optional[EvaluationResult] = None,
) -> RerankerCaseAnalysis:
    """Derive the single-run reranker diagnosis for one case.

    Pure function of ``(case, trace[, evaluation])``: no file access, no
    model calls, no global state. Every inconsistency in the persisted
    evidence (missing diagnostics, mismatched candidate/full sets, final
    not a prefix of the full ranking, duplicate IDs, query mismatch)
    raises ``ContractValidationError`` instead of producing a best-effort
    report. When ``evaluation`` is provided it must belong to this trace
    and its ``retrieval_mrr`` (plus ``retrieval_recall_at_{final_top_k}``
    when that metric key exists) must equal the derived values exactly;
    otherwise analysis fails closed.

    Depth rule: ``candidate_mrr``/``candidate_recall_at_k`` are computed
    over the candidate *window* ``candidate[:final_top_k]`` — the same
    depth as the final ranking — so deltas compare like with like and an
    Identity run always reports zero deltas. ``candidate_relevant_rank``
    and ``reranker_relevant_rank`` are best ranks over the *full*
    rankings, so movement beyond the final cutoff stays visible.
    """
    if not isinstance(case, CaseSpec):
        raise ContractValidationError("case must be a CaseSpec")
    if not isinstance(trace, RunTrace):
        raise ContractValidationError("trace must be a RunTrace")
    if trace.case_id != case.case_id:
        raise ContractValidationError(
            f"trace case_id {trace.case_id!r} does not match "
            f"case {case.case_id!r}"
        )

    candidate_event = _select_role_events(
        trace, RetrievalRole.CANDIDATE, "role=candidate"
    )
    final_event = _select_role_events(trace, RetrievalRole.FINAL, "role=final")
    if candidate_event.query != final_event.query:
        raise ContractValidationError(
            "reranker analysis evidence is inconsistent: candidate and "
            "final event queries differ"
        )
    if len(set(candidate_event.document_ids)) != len(
        candidate_event.document_ids
    ):
        raise ContractValidationError(
            "reranker analysis evidence is corrupt: candidate "
            "document_ids contain duplicates"
        )

    full_ids, full_scores = _read_full_ranking(trace)
    if set(full_ids) != set(candidate_event.document_ids) or len(
        full_ids
    ) != len(candidate_event.document_ids):
        raise ContractValidationError(
            "reranker analysis evidence is inconsistent: full reranker "
            "ranking does not cover the candidate set exactly once"
        )
    depth = len(final_event.document_ids)
    if tuple(full_ids[:depth]) != tuple(final_event.document_ids) or tuple(
        full_scores[:depth]
    ) != tuple(float(s) for s in final_event.scores):
        raise ContractValidationError(
            "reranker analysis evidence is inconsistent: final ranking is "
            "not the full reranker ranking prefix"
        )

    final_top_k = _read_final_top_k(trace)
    if depth > final_top_k:
        raise ContractValidationError(
            "reranker analysis evidence is inconsistent: final ranking "
            f"has {depth} documents but final_top_k is {final_top_k}"
        )

    metadata = trace.runtime_metadata
    reranker_meta = metadata.get("reranker")
    reranker_id = _read_str_field(reranker_meta, "reranker_id", "reranker")
    reranker_version = _read_str_field(
        reranker_meta, "reranker_version", "reranker"
    )

    relevant = tuple(case.relevant_document_ids)
    candidate_ids = tuple(candidate_event.document_ids)
    candidate_scores = tuple(float(s) for s in candidate_event.scores)
    final_ids = tuple(final_event.document_ids)
    final_scores = tuple(float(s) for s in final_event.scores)

    candidate_rank = {d: i + 1 for i, d in enumerate(candidate_ids)}
    reranker_rank = {d: i + 1 for i, d in enumerate(full_ids)}
    candidate_score = dict(zip(candidate_ids, candidate_scores))
    reranker_score = dict(zip(full_ids, full_scores))
    relevant_set = set(relevant)

    movements = tuple(
        RankMovement(
            document_id=document_id,
            candidate_rank=candidate_rank[document_id],
            reranker_rank=reranker_rank[document_id],
            rank_delta=candidate_rank[document_id]
            - reranker_rank[document_id],
            candidate_score=candidate_score[document_id],
            reranker_score=reranker_score[document_id],
            relevant=document_id in relevant_set,
            candidate_in_final_cutoff=candidate_rank[document_id]
            <= final_top_k,
            final_in_final_cutoff=reranker_rank[document_id] <= final_top_k,
        )
        for document_id in candidate_ids
    )

    candidate_relevant_rank = _best_rank(candidate_ids, relevant)
    reranker_relevant_rank = _best_rank(full_ids, relevant)
    final_relevant_rank = _best_rank(final_ids, relevant)
    # Depth-matched MRR pair: the candidate baseline is the candidate
    # *window* ``candidate[:final_top_k]`` — never the full top-10
    # candidate ranking against the final top-k (same rule as §recall).
    # Hence an Identity run always reports ``mrr_delta == 0``: its final
    # ranking equals its candidate window exactly. Full-ranking movement
    # (including ranks beyond final_top_k) is tracked separately via
    # ``reranker_relevant_rank`` / ``relevant_rank_improved``.
    candidate_window_ids = candidate_ids[:final_top_k]
    candidate_window_relevant_rank = _best_rank(candidate_window_ids, relevant)

    if relevant:
        candidate_mrr: Optional[float] = (
            1.0 / candidate_window_relevant_rank
            if candidate_window_relevant_rank is not None
            else 0.0
        )
        final_mrr: Optional[float] = (
            1.0 / final_relevant_rank if final_relevant_rank is not None else 0.0
        )
        # Same helpers as the retrieval evaluator: identical semantics
        # by construction (cross-checked against EvaluationResult below).
        candidate_recall: Optional[float] = recall_at_k(
            candidate_ids[:final_top_k], relevant, final_top_k
        )
        final_recall: Optional[float] = recall_at_k(
            final_ids[:final_top_k], relevant, final_top_k
        )
    else:
        candidate_mrr = None
        final_mrr = None
        candidate_recall = None
        final_recall = None

    # Sanity: derived final MRR equals the helper applied to the final
    # ranking (this is what the evaluator observes).
    if relevant:
        assert final_mrr == mean_reciprocal_rank(final_ids, relevant)

    if candidate_mrr is not None and final_mrr is not None:
        mrr_delta: Optional[float] = final_mrr - candidate_mrr
    else:
        mrr_delta = None
    if candidate_recall is not None and final_recall is not None:
        recall_delta: Optional[float] = final_recall - candidate_recall
    else:
        recall_delta = None

    candidate_window = set(candidate_ids[:final_top_k])
    final_set = set(final_ids)
    # Entered: final top-k members outside the candidate window, in
    # reranker-rank order. Exited: candidate-window members missing from
    # final, in candidate-rank order. Both stay deterministic.
    top_k_entered = tuple(
        d for d in full_ids[:final_top_k] if d not in candidate_window
    )
    candidate_window_ordered = tuple(candidate_ids[:final_top_k])
    top_k_exited = tuple(d for d in candidate_window_ordered if d not in final_set)

    candidate_top1: Optional[str] = candidate_ids[0] if candidate_ids else None
    final_top1: Optional[str] = final_ids[0] if final_ids else None
    top1_changed = candidate_top1 != final_top1

    if len(full_ids) >= final_top_k and full_ids:
        cutoff_score: Optional[float] = float(full_scores[final_top_k - 1])
    else:
        cutoff_score = None
    if len(full_ids) > final_top_k:
        next_below: Optional[float] = float(full_scores[final_top_k])
    else:
        next_below = None
    if cutoff_score is not None and next_below is not None:
        cutoff_margin: Optional[float] = cutoff_score - next_below
    else:
        cutoff_margin = None

    if (
        candidate_relevant_rank is not None
        and reranker_relevant_rank is not None
    ):
        relevant_rank_improved = (
            reranker_relevant_rank < candidate_relevant_rank
        )
        relevant_rank_regressed = (
            reranker_relevant_rank > candidate_relevant_rank
        )
    else:
        relevant_rank_improved = False
        relevant_rank_regressed = False

    relevant_in_window = tuple(
        d for d in candidate_window_ordered if d in relevant_set
    )
    relevant_in_final = tuple(d for d in final_ids if d in relevant_set)
    relevant_doc_entered = bool(set(relevant_in_final) - set(relevant_in_window))
    relevant_doc_exited = bool(set(relevant_in_window) - set(relevant_in_final))

    mrr_cross_check: Optional[bool] = None
    recall_cross_check: Optional[bool] = None
    if evaluation is not None:
        if not isinstance(evaluation, EvaluationResult):
            raise ContractValidationError(
                "evaluation must be an EvaluationResult"
            )
        if evaluation.case_id != case.case_id:
            raise ContractValidationError(
                "evaluation case_id does not match case"
            )
        if evaluation.run_id != trace.run_id:
            raise ContractValidationError(
                "evaluation run_id does not match trace run_id"
            )
        if relevant:
            mrr_metric = evaluation.metrics.get("retrieval_mrr")
            if mrr_metric is None:
                raise ContractValidationError(
                    "evaluation cross-check failed: 'retrieval_mrr' "
                    "metric is missing"
                )
            if float(mrr_metric.value) != final_mrr:
                raise ContractValidationError(
                    "evaluation cross-check failed: derived final MRR "
                    f"{final_mrr} != EvaluationResult "
                    f"retrieval_mrr {float(mrr_metric.value)}"
                )
            mrr_cross_check = True
            recall_key = f"retrieval_recall_at_{final_top_k}"
            recall_metric = evaluation.metrics.get(recall_key)
            if recall_metric is not None:
                if float(recall_metric.value) != final_recall:
                    raise ContractValidationError(
                        "evaluation cross-check failed: derived final "
                        f"Recall@{final_top_k} {final_recall} != "
                        f"EvaluationResult {recall_key} "
                        f"{float(recall_metric.value)}"
                    )
                recall_cross_check = True
            else:
                recall_cross_check = None
        else:
            mrr_cross_check = None
            recall_cross_check = None

    return RerankerCaseAnalysis(
        report_version=REPORT_VERSION,
        case_id=case.case_id,
        query=candidate_event.query,
        reranker_id=reranker_id,
        reranker_version=reranker_version,
        corpus_id=_read_str_field(metadata, "corpus_id", "runtime_metadata"),
        corpus_version=_read_str_field(
            metadata, "corpus_version", "runtime_metadata"
        ),
        retriever_id=_read_str_field(
            metadata, "retriever_id", "runtime_metadata"
        ),
        retriever_version=_read_str_field(
            metadata, "retriever_version", "runtime_metadata"
        ),
        candidate_count=len(candidate_ids),
        final_top_k=final_top_k,
        candidate_document_ids=candidate_ids,
        candidate_scores=candidate_scores,
        full_reranker_document_ids=full_ids,
        full_reranker_scores=tuple(float(s) for s in full_scores),
        final_document_ids=final_ids,
        final_scores=final_scores,
        relevant_document_ids=relevant,
        rank_movements=movements,
        candidate_relevant_rank=candidate_relevant_rank,
        reranker_relevant_rank=reranker_relevant_rank,
        final_relevant_rank=final_relevant_rank,
        candidate_mrr=candidate_mrr,
        final_mrr=final_mrr,
        mrr_delta=mrr_delta,
        candidate_recall_at_k=candidate_recall,
        final_recall_at_k=final_recall,
        recall_delta=recall_delta,
        top_k_entered=top_k_entered,
        top_k_exited=top_k_exited,
        candidate_top1=candidate_top1,
        final_top1=final_top1,
        top1_changed=top1_changed,
        generation_driving_document_id=final_top1,
        cutoff_score=cutoff_score,
        next_below_cutoff_score=next_below,
        cutoff_margin=cutoff_margin,
        relevant_rank_improved=relevant_rank_improved,
        relevant_rank_regressed=relevant_rank_regressed,
        relevant_doc_entered_final_top_k=relevant_doc_entered,
        relevant_doc_exited_final_top_k=relevant_doc_exited,
        generation_driver_changed=top1_changed,
        relevant_documents_in_candidate_window=relevant_in_window,
        relevant_documents_in_final=relevant_in_final,
        mrr_cross_check=mrr_cross_check,
        recall_cross_check=recall_cross_check,
    )


def render_reranker_case_summary(analysis: RerankerCaseAnalysis) -> str:
    """Render a small deterministic human-readable summary.

    Convenience view only; the typed ``RerankerCaseAnalysis`` remains the
    source of truth.
    """
    lines = [
        f"# Reranker case analysis: {analysis.case_id}",
        "",
        f"reranker: {analysis.reranker_id} (version {analysis.reranker_version})",
        f"candidates: {analysis.candidate_count}, final_top_k: {analysis.final_top_k}",
        "",
        "## Rank movement (candidate_rank -> reranker_rank)",
        "",
    ]
    for movement in analysis.rank_movements:
        flag = "relevant" if movement.relevant else "other"
        lines.append(
            f"- {movement.document_id}: {movement.candidate_rank} -> "
            f"{movement.reranker_rank} "
            f"(delta {movement.rank_delta:+d}; "
            f"rrf {movement.candidate_score:.4f}, "
            f"logit {movement.reranker_score:.4f}; {flag})"
        )
    lines += [
        "",
        "## Top-k",
        "",
        f"entered: {list(analysis.top_k_entered) or []}",
        f"exited: {list(analysis.top_k_exited) or []}",
        f"top1: {analysis.candidate_top1} -> {analysis.final_top1} "
        f"(changed: {analysis.top1_changed})",
        "",
        "## Cutoff (raw-logit difference, not calibrated confidence)",
        "",
        f"cutoff_score: {analysis.cutoff_score}",
        f"next_below_cutoff_score: {analysis.next_below_cutoff_score}",
        f"cutoff_margin: {analysis.cutoff_margin}",
        "",
        "## Retrieval delta (reporting-derived)",
        "",
        f"candidate_mrr: {analysis.candidate_mrr}",
        f"final_mrr: {analysis.final_mrr}",
        f"mrr_delta: {analysis.mrr_delta}",
        f"candidate_recall_at_k: {analysis.candidate_recall_at_k}",
        f"final_recall_at_k: {analysis.final_recall_at_k}",
        f"recall_delta: {analysis.recall_delta}",
    ]
    return "\n".join(lines) + "\n"


__all__ = [
    "REPORT_VERSION",
    "RankMovement",
    "RerankerCaseAnalysis",
    "analyze_reranker_case",
    "render_reranker_case_summary",
]
