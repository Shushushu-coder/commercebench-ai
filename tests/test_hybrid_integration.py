"""Real pinned-model integration smoke for the Phase 1G hybrid baseline.

Every test here is marked ``integration`` and excluded from the default
``pytest`` run (see ``addopts`` in ``pyproject.toml``). Run them
explicitly with::

    python -m pytest -m integration

The hybrid's dense component loads the pinned
``sentence-transformers/all-MiniLM-L6-v2`` model once per system; the
first run may populate the local HuggingFace cache. These tests must
not silently skip: a Phase 1G PASS requires them to actually execute
against the real BM25 + real MiniLM components.
"""

from __future__ import annotations

import math

import pytest

from commercebench.contracts import EvaluationResult
from commercebench.evaluation import DeterministicRAGEvaluator
from commercebench.runner import run_case
from commercebench.systems import DeterministicRAGSystem

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "8b3219a92973c328a8e22fadcfa821b5dc75636a"
BACKEND_VERSION = "6.1.0"
EXPECTED_DIMENSION = 384

SPARSE_ID = "bm25-okapi-v0"
DENSE_ID = "dense-minilm-l6-v2-cosine-v0"
HYBRID_ID = "hybrid-bm25-minilm-rrf-v0"

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def hybrid_system(hybrid_manifest, rag_corpus, rag_answers):
    return DeterministicRAGSystem.from_manifest(
        hybrid_manifest, rag_corpus, rag_answers
    )


@pytest.fixture(scope="module")
def bm25_system(bm25_manifest, rag_corpus, rag_answers):
    return DeterministicRAGSystem.from_manifest(
        bm25_manifest, rag_corpus, rag_answers
    )


@pytest.fixture(scope="module")
def dense_system(dense_manifest, rag_corpus, rag_answers):
    return DeterministicRAGSystem.from_manifest(
        dense_manifest, rag_corpus, rag_answers
    )


class TestHybridModelLoading:
    def test_model_loaded_once_for_all_cases(
        self,
        hybrid_manifest,
        rag_corpus,
        rag_answers,
        rag_cases,
        monkeypatch,
    ):
        # A hybrid run must not rebuild/reload MiniLM per case or per
        # component call: one system -> one encoder -> one model load.
        import sentence_transformers

        real = sentence_transformers.SentenceTransformer
        builds = []

        def spy(*args, **kwargs):
            builds.append(kwargs.get("revision"))
            return real(*args, **kwargs)

        monkeypatch.setattr(
            sentence_transformers, "SentenceTransformer", spy
        )
        system = DeterministicRAGSystem.from_manifest(
            hybrid_manifest, rag_corpus, rag_answers
        )
        for case in rag_cases.values():
            run_case(case, hybrid_manifest, system)
        assert builds == [MODEL_REVISION]


class TestHybridEndToEnd:
    def test_full_case_produces_trace_and_evaluation(
        self, hybrid_manifest, hybrid_system, rag_cases
    ):
        case = rag_cases["rag-return-window-001"]
        trace = run_case(case, hybrid_manifest, hybrid_system)

        # Exactly one RetrievalEvent — the fused ranking only.
        assert len(trace.retrieval_events) == 1
        event = trace.retrieval_events[0]
        assert event.query == case.user_input
        assert len(event.document_ids) == len(event.scores) == 3
        assert all(math.isfinite(s) and s > 0 for s in event.scores)
        # Fused scores are RRF sums, not BM25/cosine raw scores: the
        # maximum attainable is 2/(60+1) when both components rank a
        # document first.
        assert all(s <= (2.0 / 61) + 1e-12 for s in event.scores)
        assert list(event.scores) == sorted(event.scores, reverse=True)

        metadata = trace.runtime_metadata
        assert metadata["retriever_id"] == HYBRID_ID
        assert metadata["retriever_version"] == "0.1"
        assert metadata["corpus_id"] == "commercebench-dev-corpus"
        assert metadata["corpus_version"] == "0.1"
        assert metadata["top_k"] == 3

        hybrid_meta = metadata["hybrid"]
        assert hybrid_meta["fusion_method"] == "rrf"
        assert hybrid_meta["rrf_k"] == 60
        assert hybrid_meta["component_top_k"] == 10
        assert hybrid_meta["final_top_k"] == 3

        sparse_meta = hybrid_meta["components"]["sparse"]
        assert sparse_meta["retriever_id"] == SPARSE_ID
        assert sparse_meta["retriever_version"] == "0.1"
        assert sparse_meta["parameters"] == {"k1": 1.5, "b": 0.75}
        assert isinstance(sparse_meta["document_ids"], list)
        assert len(sparse_meta["document_ids"]) == len(
            sparse_meta["scores"]
        )

        dense_meta = hybrid_meta["components"]["dense"]
        assert dense_meta["retriever_id"] == DENSE_ID
        assert dense_meta["retriever_version"] == "0.1"
        # Dense ranks the whole 10-document corpus at depth 10.
        assert len(dense_meta["document_ids"]) == 10
        assert len(dense_meta["scores"]) == 10
        dense_runtime = dense_meta["runtime"]
        assert dense_runtime["model_name"] == MODEL_NAME
        assert dense_runtime["model_revision"] == MODEL_REVISION
        assert dense_runtime["embedding_backend"] == "sentence-transformers"
        assert dense_runtime["embedding_backend_version"] == BACKEND_VERSION
        assert dense_runtime["embedding_dimension"] == EXPECTED_DIMENSION
        assert dense_runtime["similarity"] == "cosine"
        assert dense_runtime["normalize_embeddings"] is True

        result = DeterministicRAGEvaluator(ks=(1, 3)).evaluate(
            case, trace
        )
        assert isinstance(result, EvaluationResult)
        assert "retrieval_recall_at_3" in result.metrics
        assert "retrieval_mrr" in result.metrics

    def test_every_development_case_retrieves_and_records_components(
        self, hybrid_manifest, hybrid_system, rag_cases
    ):
        for case in rag_cases.values():
            trace = run_case(case, hybrid_manifest, hybrid_system)
            assert len(trace.retrieval_events) == 1
            event = trace.retrieval_events[0]
            assert len(event.document_ids) == len(event.scores) == 3
            assert all(math.isfinite(s) for s in event.scores)
            components = trace.runtime_metadata["hybrid"]["components"]
            assert components["dense"]["document_ids"]
            assert components["sparse"]["retriever_id"] == SPARSE_ID

    def test_repeated_run_is_deterministic(
        self, hybrid_manifest, hybrid_system, rag_cases
    ):
        case = rag_cases["rag-coupon-001"]
        first = run_case(case, hybrid_manifest, hybrid_system)
        second = run_case(case, hybrid_manifest, hybrid_system)
        assert first.retrieval_events[0].document_ids == (
            second.retrieval_events[0].document_ids
        )
        assert first.retrieval_events[0].scores == pytest.approx(
            second.retrieval_events[0].scores
        )
        first_sparse = first.runtime_metadata["hybrid"]["components"][
            "sparse"
        ]["document_ids"]
        second_sparse = second.runtime_metadata["hybrid"]["components"][
            "sparse"
        ]["document_ids"]
        assert first_sparse == second_sparse

    def test_trace_json_round_trip(
        self, hybrid_manifest, hybrid_system, rag_cases
    ):
        from commercebench.contracts import RunTrace

        case = rag_cases["rag-shipping-001"]
        trace = run_case(case, hybrid_manifest, hybrid_system)
        restored = RunTrace.from_json(trace.to_json())
        assert restored.retrieval_events == trace.retrieval_events
        assert restored.runtime_metadata == trace.runtime_metadata


class TestThreeWayComparison:
    """BM25 vs Dense vs Hybrid RRF on shared development fixtures.

    Same corpus, cases, raw query, final top_k, generation fixture,
    evaluator, and seed — only the retrieval strategy configuration
    differs. These are observed fixture behaviors, not benchmark
    conclusions.
    """

    def test_hybrid_recovers_dense_hit_where_bm25_misses(
        self,
        bm25_manifest,
        dense_manifest,
        hybrid_manifest,
        bm25_system,
        dense_system,
        hybrid_system,
        rag_cases,
    ):
        # 'Can I get my money back if I change my mind' has no lexical
        # overlap with the corpus: BM25 retrieves nothing, MiniLM ranks
        # the refund policy first, and the fused ranking inherits the
        # dense ranking entirely (the sparse component contributes no
        # candidate).
        case = rag_cases["rag-refund-miss-001"]
        bm25_trace = run_case(case, bm25_manifest, bm25_system)
        dense_trace = run_case(case, dense_manifest, dense_system)
        hybrid_trace = run_case(case, hybrid_manifest, hybrid_system)

        bm25_ids = bm25_trace.retrieval_events[0].document_ids
        dense_ids = dense_trace.retrieval_events[0].document_ids
        hybrid_ids = hybrid_trace.retrieval_events[0].document_ids
        assert bm25_ids == ()
        assert dense_ids[0] == "policy-refund-001"
        assert hybrid_ids == dense_ids[:3]

        components = hybrid_trace.runtime_metadata["hybrid"]["components"]
        assert components["sparse"]["document_ids"] == []
        assert components["dense"]["document_ids"][0] == (
            "policy-refund-001"
        )

        evaluator = DeterministicRAGEvaluator(ks=(1, 3))
        bm25_result = evaluator.evaluate(case, bm25_trace)
        hybrid_result = evaluator.evaluate(case, hybrid_trace)
        assert bm25_result.metrics["retrieval_recall_at_1"].value == 0.0
        assert hybrid_result.metrics["retrieval_recall_at_1"].value == 1.0
        assert hybrid_result.metrics["retrieval_mrr"].value == 1.0

    def test_hybrid_differs_from_at_least_one_component_somewhere(
        self,
        bm25_manifest,
        dense_manifest,
        hybrid_manifest,
        bm25_system,
        dense_system,
        hybrid_system,
        rag_cases,
    ):
        # Across the development set the fused ranking must be visible
        # as its own strategy — not an alias of either component.
        differs = False
        for case in rag_cases.values():
            bm25_ids = run_case(
                case, bm25_manifest, bm25_system
            ).retrieval_events[0].document_ids
            dense_ids = run_case(
                case, dense_manifest, dense_system
            ).retrieval_events[0].document_ids
            hybrid_ids = run_case(
                case, hybrid_manifest, hybrid_system
            ).retrieval_events[0].document_ids
            if hybrid_ids != bm25_ids or hybrid_ids != dense_ids:
                differs = True
        assert differs
