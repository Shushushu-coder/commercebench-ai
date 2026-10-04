"""Tests for deterministic retrieval metrics and N/A semantics."""

from __future__ import annotations

import dataclasses

import pytest

from commercebench.contracts import RetrievalEvent
from commercebench.evaluation.retrieval import (
    FAILURE_RETRIEVAL_MISS,
    DeterministicRAGEvaluator,
    RetrievalEvaluator,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    ranked_document_ids,
    recall_at_k,
)

RELEVANT = ("d2", "d4")
RETRIEVED = ("d1", "d2", "d3", "d4")


class TestExactMetricValues:
    """Hand-verified example: relevant {d2, d4}, retrieved [d1, d2, d3, d4]."""

    def test_recall(self):
        assert recall_at_k(RETRIEVED, RELEVANT, 1) == 0.0
        assert recall_at_k(RETRIEVED, RELEVANT, 2) == 0.5
        assert recall_at_k(RETRIEVED, RELEVANT, 4) == 1.0
        assert recall_at_k(RETRIEVED, RELEVANT, 10) == 1.0

    def test_precision(self):
        assert precision_at_k(RETRIEVED, RELEVANT, 1) == 0.0
        assert precision_at_k(RETRIEVED, RELEVANT, 2) == 0.5
        assert precision_at_k(RETRIEVED, RELEVANT, 4) == 0.5

    def test_precision_k_exceeds_result_count(self):
        # Only 4 results; denominator is the considered count, not k.
        assert precision_at_k(RETRIEVED, RELEVANT, 10) == 0.5

    def test_precision_empty_retrieval(self):
        assert precision_at_k((), RELEVANT, 3) == 0.0

    def test_mrr(self):
        # First relevant doc (d2) is at rank 2 -> 1/2.
        assert mean_reciprocal_rank(RETRIEVED, RELEVANT) == 0.5
        assert mean_reciprocal_rank(("d2",), RELEVANT) == 1.0

    def test_mrr_no_hit(self):
        assert mean_reciprocal_rank(("d1", "d3"), RELEVANT) == 0.0
        assert mean_reciprocal_rank((), RELEVANT) == 0.0

    def test_ndcg(self):
        # Relevant at ranks 2 and 4:
        #   DCG  = 1/log2(3) + 1/log2(5) = 0.6309... + 0.4306...
        #   IDCG = 1/log2(2) + 1/log2(3) = 1 + 0.6309...
        expected_dcg = 1.0 / 1.584962500721156 + 1.0 / 2.321928094887362
        expected_idcg = 1.0 + 1.0 / 1.584962500721156
        assert ndcg_at_k(RETRIEVED, RELEVANT, 4) == pytest.approx(
            expected_dcg / expected_idcg
        )
        # Only rank 2 is relevant within top 2: DCG@2 = 1/log2(3).
        assert ndcg_at_k(RETRIEVED, RELEVANT, 2) == pytest.approx(
            (1.0 / 1.584962500721156) / expected_idcg
        )
        # Perfect ordering reaches 1.0.
        assert ndcg_at_k(("d2", "d4"), RELEVANT, 2) == pytest.approx(1.0)

    def test_no_hit_all_metrics_zero(self):
        assert recall_at_k(("d1", "d3"), RELEVANT, 3) == 0.0
        assert precision_at_k(("d1", "d3"), RELEVANT, 3) == 0.0
        assert mean_reciprocal_rank(("d1", "d3"), RELEVANT) == 0.0
        assert ndcg_at_k(("d1", "d3"), RELEVANT, 3) == 0.0

    def test_empty_relevant_set_zero(self):
        assert recall_at_k(RETRIEVED, (), 3) == 0.0
        assert ndcg_at_k(RETRIEVED, (), 3) == 0.0


class TestRankedDocumentIds:
    def test_concatenates_and_dedupes(self, trace):
        trace = dataclasses.replace(
            trace,
            retrieval_events=(
                RetrievalEvent(
                    query="q1",
                    document_ids=("a", "b"),
                    scores=(2.0, 1.0),
                ),
                RetrievalEvent(
                    query="q2",
                    document_ids=("b", "c"),
                    scores=(1.0, 1.0),
                ),
            ),
        )
        assert ranked_document_ids(trace) == ("a", "b", "c")

    def test_empty_events(self, trace):
        assert ranked_document_ids(trace) == ()


class TestRetrievalEvaluator:
    @pytest.fixture
    def retrieved_trace(self, trace):
        return dataclasses.replace(
            trace,
            retrieval_events=(
                RetrievalEvent(
                    query="q",
                    document_ids=("d1", "d2", "d3", "d4"),
                    scores=(4.0, 3.0, 2.0, 1.0),
                ),
            ),
        )

    @pytest.fixture
    def retrieved_case(self, case):
        return dataclasses.replace(case, relevant_document_ids=RELEVANT)

    def test_metric_names_and_values(self, retrieved_case, retrieved_trace):
        result = RetrievalEvaluator(ks=(1, 2)).evaluate(
            retrieved_case, retrieved_trace
        )
        assert result.metrics["retrieval_recall_at_1"].value == 0.0
        assert result.metrics["retrieval_recall_at_2"].value == 0.5
        assert result.metrics["retrieval_precision_at_1"].value == 0.0
        assert result.metrics["retrieval_precision_at_2"].value == 0.5
        assert result.metrics["retrieval_mrr"].value == 0.5
        assert result.metrics["retrieval_ndcg_at_2"].value == pytest.approx(
            0.3868528072345416
        )
        assert result.passed is True
        assert result.failure_categories == ()

    def test_retrieval_miss_category(self, retrieved_case, trace):
        # Trace has no retrieval events: the complete retrieved list is empty
        # and misses every relevant document.
        result = RetrievalEvaluator().evaluate(retrieved_case, trace)
        assert result.passed is False
        assert result.failure_categories == (FAILURE_RETRIEVAL_MISS,)
        assert result.metrics["retrieval_recall_at_3"].value == 0.0
        assert result.metrics["retrieval_mrr"].value == 0.0

    def test_not_applicable_when_no_relevant_docs(self, case, retrieved_trace):
        # The conftest case has no relevant_document_ids.
        assert case.relevant_document_ids == ()
        result = RetrievalEvaluator().evaluate(case, retrieved_trace)
        for name, metric in result.metrics.items():
            assert metric.value == 0.0, name
            assert metric.passed is None, name
            assert metric.details["not_applicable"] is True, name
        assert result.passed is True
        assert result.failure_categories == ()

    def test_result_round_trip(self, retrieved_case, retrieved_trace):
        from commercebench.contracts import EvaluationResult

        result = RetrievalEvaluator().evaluate(retrieved_case, retrieved_trace)
        assert EvaluationResult.from_json(result.to_json()) == result


class TestDeterministicRAGEvaluator:
    def test_combined_result(self, case, trace):
        case = dataclasses.replace(
            case, relevant_document_ids=("doc-return",)
        )
        trace = dataclasses.replace(
            trace,
            retrieval_events=(
                RetrievalEvent(
                    query=case.user_input,
                    document_ids=("doc-return",),
                    scores=(1.0,),
                ),
            ),
        )
        result = DeterministicRAGEvaluator(ks=(1,)).evaluate(case, trace)
        # Generation metrics preserved.
        assert result.metrics["exact_match"].value is True
        assert result.metrics["required_fact_coverage"].value == 1.0
        assert result.metrics["forbidden_claim_violation"].value is False
        # Retrieval metrics merged into the same result.
        assert result.metrics["retrieval_recall_at_1"].value == 1.0
        assert result.metrics["retrieval_precision_at_1"].value == 1.0
        assert result.metrics["retrieval_mrr"].value == 1.0
        assert result.metrics["retrieval_ndcg_at_1"].value == 1.0
        # No overall score metric is introduced.
        assert "overall_score" not in result.metrics
        assert result.passed is True
        assert result.evaluator_id == "deterministic-rag-v0"

    def test_retrieval_miss_fails_combined_result(self, case, trace):
        case = dataclasses.replace(
            case, relevant_document_ids=("doc-return",)
        )
        # Generation output still matches; retrieval misses entirely.
        result = DeterministicRAGEvaluator().evaluate(case, trace)
        assert result.metrics["exact_match"].value is True
        assert FAILURE_RETRIEVAL_MISS in result.failure_categories
        assert result.passed is False
