"""Deterministic retrieval metrics and the combined RAG evaluator (Phase 1A).

Metric helpers operate on ranked document-id lists:

- ``recall_at_k``: |relevant ∩ retrieved[:k]| / |relevant|
- ``precision_at_k``: |relevant ∩ retrieved[:k]| / |retrieved[:k]|;
  the denominator is the number of results actually considered, so it is
  ``min(k, len(retrieved))``. Empty retrieval -> 0.0.
- ``mean_reciprocal_rank``: 1/rank of the first relevant document in the
  full retrieved list; no hit -> 0.0.
- ``ndcg_at_k``: binary-relevance NDCG@K (relevant=1, otherwise=0);
  DCG@K / IDCG@K. Graded relevance is not implemented.

N/A semantics (documented and tested): when ``case.relevant_document_ids``
is empty there is no ground truth, so every retrieval metric reports
``value=0.0``, ``passed=None``, and ``details["not_applicable"] = True``.

Failure categories: ``RETRIEVAL_MISS`` is added when the case defines
relevant documents and none of them appear in the complete retrieved list.
In the combined evaluator this makes the result fail, consistent with the
Phase 0 convention that ``passed = (no failure categories)``.

Retrieval metrics are computed from final retrieval events only
(``RetrievalEvent.role`` is ``FINAL`` or absent — legacy Phase 1
semantics). ``CANDIDATE`` events are diagnostic intermediate rankings
and do not count as retrieved documents.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import replace
from datetime import datetime, timezone
from typing import Dict, List, Sequence, Tuple

from commercebench.contracts.case import CaseSpec
from commercebench.contracts.common import SCHEMA_VERSION
from commercebench.contracts.evaluation import EvaluationResult, MetricResult
from commercebench.contracts.trace import RunTrace

from .deterministic import DeterministicEvaluator

RETRIEVAL_EVALUATOR_ID = "retrieval-v0"
RETRIEVAL_EVALUATOR_VERSION = "0.1"

RAG_EVALUATOR_ID = "deterministic-rag-v0"
RAG_EVALUATOR_VERSION = "0.1"

FAILURE_RETRIEVAL_MISS = "RETRIEVAL_MISS"

DEFAULT_KS: Tuple[int, ...] = (1, 3)


def recall_at_k(
    retrieved_ids: Sequence[str], relevant_ids: Sequence[str], k: int
) -> float:
    """Fraction of relevant documents present in the top-k retrieved list."""
    if not relevant_ids:
        return 0.0
    hits = len(set(retrieved_ids[:k]) & set(relevant_ids))
    return hits / len(relevant_ids)


def precision_at_k(
    retrieved_ids: Sequence[str], relevant_ids: Sequence[str], k: int
) -> float:
    """Fraction of the top-k retrieved documents that are relevant."""
    considered = min(k, len(retrieved_ids))
    if considered == 0:
        return 0.0
    hits = len(set(retrieved_ids[:k]) & set(relevant_ids))
    return hits / considered


def mean_reciprocal_rank(
    retrieved_ids: Sequence[str], relevant_ids: Sequence[str]
) -> float:
    """1/rank of the first relevant document; 0.0 when there is no hit."""
    relevant = set(relevant_ids)
    for index, document_id in enumerate(retrieved_ids):
        if document_id in relevant:
            return 1.0 / (index + 1)
    return 0.0


def ndcg_at_k(
    retrieved_ids: Sequence[str], relevant_ids: Sequence[str], k: int
) -> float:
    """Binary-relevance NDCG@K. Empty relevant set -> 0.0."""
    if not relevant_ids:
        return 0.0
    relevant = set(relevant_ids)
    dcg = sum(
        1.0 / math.log2(index + 2)
        for index, document_id in enumerate(retrieved_ids[:k])
        if document_id in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(index + 2) for index in range(ideal_hits))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def ranked_document_ids(trace: RunTrace) -> Tuple[str, ...]:
    """Evaluation-visible final ranking for retrieval metrics.

    Only ``RetrievalEvent.role`` ``FINAL`` and legacy no-role events
    contribute; ``CANDIDATE`` events are diagnostics and never enter
    retrieval metrics. Final events are concatenated in trace order and
    de-duplicated keep-first, so ``document_ids[i]`` of the result is the
    rank ``i + 1`` retrieved document.
    """
    return trace.final_ranked_document_ids


class RetrievalEvaluator:
    """Computes the Phase 1A retrieval metric set for one (case, trace)."""

    evaluator_id = RETRIEVAL_EVALUATOR_ID
    evaluator_version = RETRIEVAL_EVALUATOR_VERSION

    def __init__(self, ks: Sequence[int] = DEFAULT_KS) -> None:
        self._ks = tuple(int(k) for k in ks)
        for k in self._ks:
            if k <= 0:
                raise ValueError("ks must contain positive integers")

    def evaluate(self, case: CaseSpec, trace: RunTrace) -> EvaluationResult:
        metrics, failure_categories = self._metrics_and_failures(case, trace)
        return EvaluationResult(
            schema_version=SCHEMA_VERSION,
            evaluation_id=uuid.uuid4().hex,
            run_id=trace.run_id,
            case_id=case.case_id,
            evaluator_id=self.evaluator_id,
            evaluator_version=self.evaluator_version,
            metrics=metrics,
            passed=not failure_categories,
            failure_categories=tuple(failure_categories),
            evaluated_at=datetime.now(timezone.utc).isoformat(),
            metadata={},
        )

    def _metrics_and_failures(
        self, case: CaseSpec, trace: RunTrace
    ) -> Tuple[Dict[str, MetricResult], List[str]]:
        retrieved = ranked_document_ids(trace)
        relevant = tuple(case.relevant_document_ids)
        base_details = {
            "relevant_document_ids": list(relevant),
            "retrieved_document_ids": list(retrieved),
        }

        if not relevant:
            metrics: Dict[str, MetricResult] = {}
            for k in self._ks:
                metrics[f"retrieval_recall_at_{k}"] = _na_metric(
                    f"retrieval_recall_at_{k}", k, base_details
                )
                metrics[f"retrieval_precision_at_{k}"] = _na_metric(
                    f"retrieval_precision_at_{k}", k, base_details
                )
                metrics[f"retrieval_ndcg_at_{k}"] = _na_metric(
                    f"retrieval_ndcg_at_{k}", k, base_details
                )
            metrics["retrieval_mrr"] = MetricResult(
                name="retrieval_mrr",
                value=0.0,
                passed=None,
                details={**base_details, "not_applicable": True},
            )
            return metrics, []

        failure_categories: List[str] = []
        metrics = {}
        relevant_set = set(relevant)
        for k in self._ks:
            hits = sorted(relevant_set & set(retrieved[:k]))
            metrics[f"retrieval_recall_at_{k}"] = MetricResult(
                name=f"retrieval_recall_at_{k}",
                value=recall_at_k(retrieved, relevant, k),
                passed=None,
                details={**base_details, "k": k, "hits": hits},
            )
            metrics[f"retrieval_precision_at_{k}"] = MetricResult(
                name=f"retrieval_precision_at_{k}",
                value=precision_at_k(retrieved, relevant, k),
                passed=None,
                details={
                    **base_details,
                    "k": k,
                    "hits": hits,
                    "considered": min(k, len(retrieved)),
                },
            )
            metrics[f"retrieval_ndcg_at_{k}"] = MetricResult(
                name=f"retrieval_ndcg_at_{k}",
                value=ndcg_at_k(retrieved, relevant, k),
                passed=None,
                details={**base_details, "k": k, "hits": hits},
            )
        metrics["retrieval_mrr"] = MetricResult(
            name="retrieval_mrr",
            value=mean_reciprocal_rank(retrieved, relevant),
            passed=None,
            details=base_details,
        )

        if not relevant_set & set(retrieved):
            failure_categories.append(FAILURE_RETRIEVAL_MISS)

        return metrics, failure_categories


class DeterministicRAGEvaluator:
    """Combined evaluator: Phase 0 generation metrics + retrieval metrics.

    Produces a single ``EvaluationResult`` whose ``metrics`` contains both
    the generation metrics (``exact_match``, ``required_fact_coverage``,
    ``forbidden_claim_violation``) and the retrieval metrics
    (``retrieval_recall_at_*``, ``retrieval_precision_at_*``,
    ``retrieval_ndcg_at_*``, ``retrieval_mrr``). There is no overall score;
    ``passed`` is False iff any failure category is present, including
    ``RETRIEVAL_MISS``.
    """

    evaluator_id = RAG_EVALUATOR_ID
    evaluator_version = RAG_EVALUATOR_VERSION

    def __init__(
        self,
        ks: Sequence[int] = DEFAULT_KS,
        generation_evaluator: DeterministicEvaluator | None = None,
    ) -> None:
        self._generation_evaluator = (
            generation_evaluator or DeterministicEvaluator()
        )
        self._retrieval_evaluator = RetrievalEvaluator(ks=ks)

    def evaluate(self, case: CaseSpec, trace: RunTrace) -> EvaluationResult:
        generation_result = self._generation_evaluator.evaluate(case, trace)
        retrieval_result = self._retrieval_evaluator.evaluate(case, trace)

        metrics = dict(generation_result.metrics)
        metrics.update(retrieval_result.metrics)
        failure_categories = tuple(
            generation_result.failure_categories
        ) + tuple(retrieval_result.failure_categories)

        return replace(
            generation_result,
            evaluation_id=uuid.uuid4().hex,
            evaluator_id=self.evaluator_id,
            evaluator_version=self.evaluator_version,
            metrics=metrics,
            passed=not failure_categories,
            failure_categories=failure_categories,
            evaluated_at=datetime.now(timezone.utc).isoformat(),
        )


def _na_metric(
    name: str, k: int, base_details: Dict[str, object]
) -> MetricResult:
    return MetricResult(
        name=name,
        value=0.0,
        passed=None,
        details={**base_details, "k": k, "not_applicable": True},
    )


__all__ = [
    "DEFAULT_KS",
    "DeterministicRAGEvaluator",
    "FAILURE_RETRIEVAL_MISS",
    "RAG_EVALUATOR_ID",
    "RAG_EVALUATOR_VERSION",
    "RETRIEVAL_EVALUATOR_ID",
    "RETRIEVAL_EVALUATOR_VERSION",
    "RetrievalEvaluator",
    "mean_reciprocal_rank",
    "ndcg_at_k",
    "precision_at_k",
    "ranked_document_ids",
    "recall_at_k",
]
