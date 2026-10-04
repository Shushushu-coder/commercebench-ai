"""Tests for the deterministic RAG system and the Phase 1A vertical slice."""

from __future__ import annotations

import pytest

from commercebench.contracts import (
    ContractValidationError,
    EvaluationResult,
    RunTrace,
)
from commercebench.evaluation import (
    FAILURE_RETRIEVAL_MISS,
    DeterministicRAGEvaluator,
)
from commercebench.rag import (
    DEFAULT_FALLBACK_ANSWER,
    DeterministicRAGPipeline,
    KeywordMatchRetriever,
    TermFrequencyRetriever,
)
from commercebench.runner import run_case
from commercebench.systems import DeterministicRAGSystem


@pytest.fixture
def keyword_system(keyword_manifest, rag_corpus, rag_answers):
    return DeterministicRAGSystem.from_manifest(
        keyword_manifest, rag_corpus, rag_answers
    )


@pytest.fixture
def term_frequency_system(
    term_frequency_manifest, rag_corpus, rag_answers
):
    return DeterministicRAGSystem.from_manifest(
        term_frequency_manifest, rag_corpus, rag_answers
    )


class TestDeterministicRAGSystem:
    def test_retrieval_event_recorded(
        self, keyword_system, rag_cases, keyword_manifest
    ):
        case = rag_cases["rag-return-window-001"]
        trace = run_case(case, keyword_manifest, keyword_system)
        assert len(trace.retrieval_events) == 1
        event = trace.retrieval_events[0]
        assert event.query == case.user_input
        # rank i+1 semantics: event order is the ranking order.
        assert event.document_ids[0] == "policy-return-001"
        assert len(event.document_ids) == len(event.scores)

    def test_rank_order_preserved_in_event(
        self, keyword_system, rag_cases, keyword_manifest
    ):
        case = rag_cases["rag-coupon-001"]
        trace = run_case(case, keyword_manifest, keyword_system)
        event = trace.retrieval_events[0]
        assert event.document_ids == (
            "policy-coupon-001",
            "policy-promo-001",
        )
        assert event.scores == (3.0, 1.0)

    def test_deterministic_answer_from_top1(
        self, keyword_system, rag_cases, keyword_manifest, rag_answers
    ):
        case = rag_cases["rag-return-window-001"]
        trace = run_case(case, keyword_manifest, keyword_system)
        assert trace.output_text == rag_answers["policy-return-001"]

    def test_fallback_when_nothing_retrieved(
        self, keyword_system, rag_cases, keyword_manifest
    ):
        case = rag_cases["rag-unknown-001"]
        trace = run_case(case, keyword_manifest, keyword_system)
        event = trace.retrieval_events[0]
        assert event.document_ids == ()
        assert trace.output_text == DEFAULT_FALLBACK_ANSWER

    def test_repeated_run_deterministic(
        self, keyword_system, rag_cases, keyword_manifest
    ):
        case = rag_cases["rag-coupon-001"]
        first = run_case(case, keyword_manifest, keyword_system)
        second = run_case(case, keyword_manifest, keyword_system)
        assert first.output_text == second.output_text
        assert (
            first.retrieval_events[0].document_ids
            == second.retrieval_events[0].document_ids
        )

    def test_runtime_metadata_records_corpus_identity(
        self, keyword_system, rag_cases, keyword_manifest
    ):
        case = rag_cases["rag-return-window-001"]
        trace = run_case(case, keyword_manifest, keyword_system)
        metadata = trace.runtime_metadata
        assert metadata["corpus_id"] == "commercebench-dev-corpus"
        assert metadata["corpus_version"] == "0.1"
        assert metadata["retriever_id"] == "keyword-match-v0"
        assert metadata["top_k"] == 3

    def test_from_manifest_without_retrieval_rejected(
        self, manifest, rag_corpus
    ):
        with pytest.raises(ContractValidationError):
            DeterministicRAGSystem.from_manifest(manifest, rag_corpus)

    def test_from_manifest_corpus_mismatch_rejected(
        self, keyword_manifest, rag_answers
    ):
        from commercebench.rag import Corpus, Document

        other = Corpus(
            corpus_id="other-corpus",
            corpus_version="9.9",
            documents=(Document(document_id="x", text="y"),),
        )
        with pytest.raises(ContractValidationError, match="corpus_id"):
            DeterministicRAGSystem.from_manifest(
                keyword_manifest, other, rag_answers
            )

    def test_answer_map_must_reference_known_documents(self, rag_corpus):
        with pytest.raises(ContractValidationError, match="unknown"):
            DeterministicRAGPipeline(
                retriever=KeywordMatchRetriever(),
                corpus=rag_corpus,
                top_k=3,
                answer_by_document_id={"nope-001": "answer"},
            )


class TestVerticalSliceEndToEnd:
    """Full path: corpus + case + manifest -> run -> trace -> evaluate."""

    def test_full_slice_passes(
        self, keyword_manifest, rag_corpus, rag_cases, rag_answers
    ):
        system = DeterministicRAGSystem.from_manifest(
            keyword_manifest, rag_corpus, rag_answers
        )
        case = rag_cases["rag-coupon-001"]

        trace = run_case(case, keyword_manifest, system)
        trace = RunTrace.from_json(trace.to_json())

        assert trace.retrieval_events[0].document_ids == (
            "policy-coupon-001",
            "policy-promo-001",
        )
        assert trace.output_text == rag_answers["policy-coupon-001"]

        result = DeterministicRAGEvaluator(ks=(1, 3)).evaluate(case, trace)
        result = EvaluationResult.from_json(result.to_json())

        # Retrieval metrics.
        assert result.metrics["retrieval_recall_at_1"].value == 1.0
        assert result.metrics["retrieval_recall_at_3"].value == 1.0
        assert result.metrics["retrieval_precision_at_1"].value == 1.0
        assert result.metrics["retrieval_precision_at_3"].value == 0.5
        assert result.metrics["retrieval_mrr"].value == 1.0
        assert result.metrics["retrieval_ndcg_at_3"].value == 1.0
        # Generation metrics preserved.
        assert result.metrics["exact_match"].value is True
        assert result.metrics["required_fact_coverage"].value == 1.0
        assert result.metrics["forbidden_claim_violation"].value is False
        assert result.passed is True
        assert result.failure_categories == ()

    def test_retrieval_miss_case_fails(
        self, keyword_manifest, rag_corpus, rag_cases, rag_answers
    ):
        system = DeterministicRAGSystem.from_manifest(
            keyword_manifest, rag_corpus, rag_answers
        )
        case = rag_cases["rag-refund-miss-001"]
        trace = run_case(case, keyword_manifest, system)
        result = DeterministicRAGEvaluator().evaluate(case, trace)

        # Generation still hits the fallback answer exactly...
        assert result.metrics["exact_match"].value is True
        # ...but retrieval missed every relevant document.
        assert result.metrics["retrieval_recall_at_3"].value == 0.0
        assert result.metrics["retrieval_mrr"].value == 0.0
        assert result.failure_categories == (FAILURE_RETRIEVAL_MISS,)
        assert result.passed is False

    def test_no_relevant_docs_marked_not_applicable(
        self, keyword_manifest, rag_corpus, rag_cases, rag_answers
    ):
        system = DeterministicRAGSystem.from_manifest(
            keyword_manifest, rag_corpus, rag_answers
        )
        case = rag_cases["rag-unknown-001"]
        trace = run_case(case, keyword_manifest, system)
        result = DeterministicRAGEvaluator().evaluate(case, trace)

        retrieval_metrics = {
            name: metric
            for name, metric in result.metrics.items()
            if name.startswith("retrieval_")
        }
        assert retrieval_metrics
        for metric in retrieval_metrics.values():
            assert metric.details["not_applicable"] is True
        assert FAILURE_RETRIEVAL_MISS not in result.failure_categories
        assert result.passed is True


class TestControlledAblation:
    """Same case, two retrieval configs: real ranking + metric differences."""

    def test_fingerprints_differ_only_by_retriever(
        self, keyword_manifest, term_frequency_manifest
    ):
        assert keyword_manifest.config_fingerprint() != (
            term_frequency_manifest.config_fingerprint()
        )
        for field in (
            "benchmark_version",
            "case_ids",
            "system_id",
            "system_version",
            "model",
            "memory",
            "evaluator_id",
            "evaluator_version",
            "random_seed",
        ):
            assert getattr(keyword_manifest, field) == getattr(
                term_frequency_manifest, field
            )
        assert keyword_manifest.retrieval.parameters == (
            term_frequency_manifest.retrieval.parameters
        )
        assert keyword_manifest.retrieval.retriever_id == "keyword-match-v0"
        assert (
            term_frequency_manifest.retrieval.retriever_id
            == "term-frequency-v0"
        )

    def test_rankings_and_metrics_differ(
        self,
        keyword_manifest,
        term_frequency_manifest,
        keyword_system,
        term_frequency_system,
        rag_cases,
    ):
        case = rag_cases["rag-coupon-001"]
        evaluator = DeterministicRAGEvaluator(ks=(1, 3))

        kw_trace = run_case(case, keyword_manifest, keyword_system)
        tf_trace = run_case(case, term_frequency_manifest, term_frequency_system)

        kw_ids = kw_trace.retrieval_events[0].document_ids
        tf_ids = tf_trace.retrieval_events[0].document_ids
        assert kw_ids == ("policy-coupon-001", "policy-promo-001")
        assert tf_ids == ("policy-promo-001", "policy-coupon-001")
        assert kw_ids != tf_ids

        kw_result = evaluator.evaluate(case, kw_trace)
        tf_result = evaluator.evaluate(case, tf_trace)

        # Real metric differences, not faked values.
        assert kw_result.metrics["retrieval_recall_at_1"].value == 1.0
        assert tf_result.metrics["retrieval_recall_at_1"].value == 0.0
        assert kw_result.metrics["retrieval_mrr"].value == 1.0
        assert tf_result.metrics["retrieval_mrr"].value == 0.5
        assert kw_result.metrics["retrieval_ndcg_at_1"].value == 1.0
        assert tf_result.metrics["retrieval_ndcg_at_1"].value == 0.0

        # Different top-1 -> different deterministic answer -> different
        # generation outcome.
        assert kw_trace.output_text != tf_trace.output_text
        assert kw_result.metrics["exact_match"].value is True
        assert tf_result.metrics["exact_match"].value is False
        assert kw_result.passed is True
        assert tf_result.passed is False

        # Config fingerprints recorded in traces differ.
        assert kw_trace.config_fingerprint != tf_trace.config_fingerprint

    def test_pipeline_retrievers_directly(
        self, rag_corpus, rag_cases
    ):
        case = rag_cases["rag-coupon-001"]
        kw = KeywordMatchRetriever().retrieve(
            case.user_input, rag_corpus, top_k=3
        )
        tf = TermFrequencyRetriever().retrieve(
            case.user_input, rag_corpus, top_k=3
        )
        assert kw.document_ids != tf.document_ids
        assert kw.scores == (3.0, 1.0)
        assert tf.scores == (4.0, 3.0)
