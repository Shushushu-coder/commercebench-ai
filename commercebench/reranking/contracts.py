"""Reranker typed boundary: RerankCandidate and RerankResult (Phase 2B).

A reranker never sees bare strings. It receives the retrieval context it
needs to preserve or ignore:

- ``document``: the full retrievable unit (identity + text for scoring).
- ``retrieval_rank``: the 1-based rank in the candidate retrieval ranking.
- ``retrieval_score``: the aligned retrieval score (RRF for the hybrid
  baseline; never compared in magnitude with reranker scores).

``RerankResult`` carries the reranked ranking. Structural validation
(unique IDs, aligned finite scores) lives here; subset and depth checks
against the candidate set are enforced by ``rerank`` implementations and
re-checked by the pipeline (fail closed).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Tuple

from commercebench.contracts.common import (
    expect_mapping,
    require_int,
    require_non_empty_str,
    str_tuple,
)
from commercebench.contracts.errors import ContractValidationError
from commercebench.rag.documents import Document


@dataclass(frozen=True)
class RerankCandidate:
    """One candidate entering the reranker with its retrieval context."""

    document: Document
    retrieval_rank: int
    retrieval_score: float

    def __post_init__(self) -> None:
        if not isinstance(self.document, Document):
            try:
                document = Document.from_dict(self.document)
            except ContractValidationError:
                raise
            except Exception as exc:
                raise ContractValidationError(
                    f"document is invalid: {exc}"
                ) from exc
            object.__setattr__(self, "document", document)
        rank = require_int(self.retrieval_rank, "retrieval_rank")
        if rank <= 0:
            raise ContractValidationError(
                "retrieval_rank must be a positive integer"
            )
        if isinstance(self.retrieval_score, bool) or not isinstance(
            self.retrieval_score, (int, float)
        ):
            raise ContractValidationError("retrieval_score must be a number")
        score = float(self.retrieval_score)
        if not math.isfinite(score):
            raise ContractValidationError("retrieval_score must be finite")

    @property
    def document_id(self) -> str:
        """Convenience alias for the candidate document identity."""
        return self.document.document_id

    @property
    def text(self) -> str:
        """Convenience alias for the candidate document text."""
        return self.document.text

    def to_dict(self) -> dict[str, Any]:
        return {
            "document": self.document.to_dict(),
            "retrieval_rank": self.retrieval_rank,
            "retrieval_score": float(self.retrieval_score),
        }

    @classmethod
    def from_dict(cls, data: Any) -> "RerankCandidate":
        data = expect_mapping(data, "RerankCandidate")
        try:
            return cls(
                document=data["document"],
                retrieval_rank=data["retrieval_rank"],
                retrieval_score=data["retrieval_score"],
            )
        except KeyError as exc:
            raise ContractValidationError(
                f"RerankCandidate missing required field: {exc.args[0]}"
            ) from exc


@dataclass(frozen=True)
class RerankResult:
    """Reranked ranking: aligned document IDs and reranker scores.

    ``document_ids[i]`` is the rank ``i + 1`` result after reranking and
    ``scores[i]`` is the aligned reranker score (higher is better for the
    Phase 2B baselines). Structural checks (unique IDs, aligned finite
    scores) are enforced here; the caller enforces ``len <= top_k`` and
    the candidate-subset rule with knowledge of the candidate set.
    """

    query: str
    document_ids: Tuple[str, ...] = ()
    scores: Tuple[float, ...] = ()

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
            if not math.isfinite(float(score)):
                raise ContractValidationError(
                    f"scores[{index}] must be finite"
                )
        object.__setattr__(self, "scores", tuple(float(s) for s in scores))
        if len(self.document_ids) != len(self.scores):
            raise ContractValidationError(
                "document_ids and scores must have the same length"
            )
        if len(set(self.document_ids)) != len(self.document_ids):
            raise ContractValidationError("document_ids must be unique")

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "document_ids": list(self.document_ids),
            "scores": list(self.scores),
        }

    @classmethod
    def from_dict(cls, data: Any) -> "RerankResult":
        data = expect_mapping(data, "RerankResult")
        try:
            return cls(
                query=data["query"],
                document_ids=data.get("document_ids", ()),
                scores=data.get("scores", ()),
            )
        except KeyError as exc:
            raise ContractValidationError(
                f"RerankResult missing required field: {exc.args[0]}"
            ) from exc


def validate_candidate_ids_unique(candidates: Any) -> Tuple[str, ...]:
    """Fail closed on duplicate candidate IDs (rank semantics change)."""
    ids = [candidate.document_id for candidate in candidates]
    if len(set(ids)) != len(ids):
        raise ContractValidationError(
            "duplicate candidate document_ids are not allowed"
        )
    return tuple(ids)


def validate_result_against_candidates(
    candidate_ids: Tuple[str, ...],
    document_ids: Tuple[str, ...],
    scores: Tuple[float, ...],
    top_k: int,
) -> None:
    """Enforce the reranker output contract against the candidate set."""
    if len(document_ids) != len(scores):
        raise ContractValidationError(
            "document_ids and scores must have the same length"
        )
    if len(document_ids) > top_k:
        raise ContractValidationError(
            f"reranked result has {len(document_ids)} documents "
            f"but top_k is {top_k}"
        )
    if len(set(document_ids)) != len(document_ids):
        raise ContractValidationError("reranked document_ids must be unique")
    for index, score in enumerate(scores):
        if not math.isfinite(float(score)):
            raise ContractValidationError(
                f"reranked scores[{index}] must be finite"
            )
    allowed = set(candidate_ids)
    for document_id in document_ids:
        if document_id not in allowed:
            raise ContractValidationError(
                f"reranked document {document_id!r} is not in the candidate set"
            )


__all__ = [
    "RerankCandidate",
    "RerankResult",
    "validate_candidate_ids_unique",
    "validate_result_against_candidates",
]
