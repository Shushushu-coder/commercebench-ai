"""Real pinned-model integration smoke for the Phase 2B reranker baseline.

Every test here is marked ``integration`` and excluded from the default
``pytest`` run (see ``addopts`` in ``pyproject.toml``). Run them
explicitly with::

    python -m pytest -m integration

The first run downloads ``cross-encoder/ms-marco-MiniLM-L6-v2`` at the
pinned immutable revision into the local HuggingFace cache; later runs
reuse that cache. These tests must not silently skip: a Phase 2B PASS
requires them to actually execute against the real cross-encoder.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from commercebench.contracts import EvaluationResult, ExperimentManifest, RunTrace
from commercebench.evaluation import DeterministicRAGEvaluator
from commercebench.evaluation.retrieval import ranked_document_ids
from commercebench.reranking import CrossEncoderReranker
from commercebench.runner import run_case
from commercebench.systems import DeterministicRAGSystem

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"
MODEL_REVISION = "ce0834f22110de6d9222af7a7a03628121708969"
BACKEND_VERSION = "6.1.0"
MAX_LENGTH = 512

RERANKER_ID = "cross-encoder-msmarco-minilm-l6-v2-v0"

RAG_V0_DIR = Path(__file__).resolve().parents[1] / "examples" / "rag_v0"

pytestmark = pytest.mark.integration


def _load_manifest(name: str) -> ExperimentManifest:
    return ExperimentManifest.from_json(
        (RAG_V0_DIR / "experiments" / name).read_text(encoding="utf-8")
    )


@pytest.fixture(scope="module")
def identity_manifest() -> ExperimentManifest:
    return _load_manifest("rag_hybrid_identity_rerank_v0.json")


@pytest.fixture(scope="module")
def cross_manifest() -> ExperimentManifest:
    return _load_manifest("rag_hybrid_cross_encoder_v0.json")


@pytest.fixture(scope="module")
def identity_system(identity_manifest, rag_corpus, rag_answers):
    return DeterministicRAGSystem.from_manifest(
        identity_manifest, rag_corpus, rag_answers
    )


@pytest.fixture(scope="module")
def cross_system(cross_manifest, rag_corpus, rag_answers):
    return DeterministicRAGSystem.from_manifest(
        cross_manifest, rag_corpus, rag_answers
    )


class TestPinnedModelIdentity:
    def test_revision_is_passed_to_cross_encoder(self, monkeypatch):
        import sentence_transformers

        captured = {}
        real = sentence_transformers.CrossEncoder

        def spy(*args, **kwargs):
            captured["args"] = args
            captured.update(kwargs)
            return real(*args, **kwargs)

        monkeypatch.setattr(sentence_transformers, "CrossEncoder", spy)
        reranker = CrossEncoderReranker()
        assert reranker.is_loaded is False
        reranker._load_model()

        assert captured["args"][0] == MODEL_NAME
        assert captured["revision"] == MODEL_REVISION
        assert captured["revision"] != "main"

    def test_max_length_is_passed_to_cross_encoder(self, monkeypatch):
        import sentence_transformers

        captured = {}
        real = sentence_transformers.CrossEncoder

        def spy(*args, **kwargs):
            captured.update(kwargs)
            return real(*args, **kwargs)

        monkeypatch.setattr(sentence_transformers, "CrossEncoder", spy)
        CrossEncoderReranker()._load_model()
        assert captured["max_length"] == MAX_LENGTH

    def test_activation_identity_enforced(self, monkeypatch):
        import torch.nn as nn

        import sentence_transformers

        captured = {}
        real = sentence_transformers.CrossEncoder

        def spy(*args, **kwargs):
            captured.update(kwargs)
            return real(*args, **kwargs)

        monkeypatch.setattr(sentence_transformers, "CrossEncoder", spy)
        reranker = CrossEncoderReranker()
        model = reranker._load_model()
        assert isinstance(captured["activation_fn"], nn.Identity)
        # The loaded model's own activation must also be identity (raw logit),
        # not the ms-marco default sigmoid.
        assert isinstance(model.activation_fn, nn.Identity)

        predict_calls = {}
        real_predict = type(model).predict

        def predict_spy(self, *args, **kwargs):
            predict_calls.update(kwargs)
            return real_predict(self, *args, **kwargs)

        monkeypatch.setattr(type(model), "predict", predict_spy)
        from commercebench.rag import Document
        from commercebench.reranking import RerankCandidate

        candidates = (
            RerankCandidate(
                document=Document(document_id="d1", text="hello world"),
                retrieval_rank=1,
                retrieval_score=1.0,
            ),
        )
        reranker.rerank("hello", candidates, top_k=1)
        assert isinstance(predict_calls.get("activation_fn"), nn.Identity)

    def test_backend_version_matches_installed(self):
        import sentence_transformers

        assert sentence_transformers.__version__ == BACKEND_VERSION

    def test_lazy_loading(self):
        reranker = CrossEncoderReranker()
        assert reranker.is_loaded is False
        assert reranker._model is None


class TestRealReranking:
    def test_scores_finite_and_ranking_produced(self, cross_manifest, cross_system, rag_cases):
        case = rag_cases["rag-return-window-001"]
        trace = run_case(case, cross_manifest, cross_system)
        assert len(trace.retrieval_events) == 2
        candidate, final = trace.retrieval_events
        assert candidate.role.value == "candidate"
        assert final.role.value == "final"
        assert len(final.document_ids) == len(final.scores) == 3
        assert all(math.isfinite(s) for s in final.scores)
        assert set(final.document_ids) <= set(candidate.document_ids)
        assert list(final.scores) == sorted(final.scores, reverse=True)

    def test_repeatable_ranking(self, cross_manifest, cross_system, rag_cases):
        case = rag_cases["rag-coupon-001"]
        first = run_case(case, cross_manifest, cross_system)
        second = run_case(case, cross_manifest, cross_system)
        assert first.retrieval_events[1].document_ids == (
            second.retrieval_events[1].document_ids
        )
        assert first.retrieval_events[1].scores == pytest.approx(
            second.retrieval_events[1].scores
        )

    def test_eleven_development_cases_run(self, cross_manifest, cross_system, rag_cases):
        assert len(rag_cases) == 11
        for case in rag_cases.values():
            trace = run_case(case, cross_manifest, cross_system)
            assert len(trace.retrieval_events) == 2
            candidate, final = trace.retrieval_events
            assert len(final.document_ids) <= 3
            assert set(final.document_ids) <= set(candidate.document_ids)
            assert all(math.isfinite(s) for s in final.scores)
            restored = RunTrace.from_json(trace.to_json())
            assert restored == trace
            result = DeterministicRAGEvaluator(ks=(1, 3)).evaluate(case, trace)
            assert isinstance(result, EvaluationResult)
            assert "retrieval_recall_at_3" in result.metrics
            # Generation consumes the final top-1 (or fallback when empty).
            if final.document_ids:
                assert trace.output_text != ""


class TestModelLoadCount:
    def test_model_loaded_once_for_all_cases(
        self, cross_manifest, rag_corpus, rag_answers, rag_cases, monkeypatch
    ):
        import sentence_transformers

        real = sentence_transformers.CrossEncoder
        builds: list = []

        def spy(*args, **kwargs):
            builds.append(kwargs.get("revision"))
            return real(*args, **kwargs)

        monkeypatch.setattr(sentence_transformers, "CrossEncoder", spy)
        system = DeterministicRAGSystem.from_manifest(
            cross_manifest, rag_corpus, rag_answers
        )
        for case in rag_cases.values():
            run_case(case, cross_manifest, system)
        # One reranker model load for the whole experiment (the hybrid's
        # dense encoder loads its own MiniLM model separately; only the
        # cross-encoder loads are counted here by spying the CrossEncoder
        # constructor with the reranker revision).
        reranker_loads = [r for r in builds if r == MODEL_REVISION]
        assert reranker_loads == [MODEL_REVISION]


class TestTraceAndEvaluation:
    def test_candidate_final_serialized_and_evaluated(
        self, cross_manifest, cross_system, rag_cases
    ):
        case = rag_cases["rag-shipping-001"]
        trace = run_case(case, cross_manifest, cross_system)
        restored = RunTrace.from_json(trace.to_json())
        assert restored.retrieval_events == trace.retrieval_events
        assert restored.runtime_metadata == trace.runtime_metadata
        result = DeterministicRAGEvaluator(ks=(1, 3)).evaluate(case, trace)
        assert isinstance(result, EvaluationResult)
        # Candidate must not leak into metrics.
        assert result.metrics["retrieval_recall_at_3"].details[
            "retrieved_document_ids"
        ] == list(trace.retrieval_events[1].document_ids)

    def test_generation_uses_final(self, cross_manifest, cross_system, rag_cases, rag_answers):
        # Across the dev set, whenever the final top-1 has a mapped answer
        # the trace output must equal that mapping (not the candidate top-1).
        for case in rag_cases.values():
            trace = run_case(case, cross_manifest, cross_system)
            final_ids = trace.retrieval_events[1].document_ids
            if final_ids:
                expected = rag_answers.get(final_ids[0])
                if expected is not None:
                    assert trace.output_text == expected

    def test_reranker_metadata_present(self, cross_manifest, cross_system, rag_cases):
        case = rag_cases["rag-return-window-001"]
        trace = run_case(case, cross_manifest, cross_system)
        assert "hybrid" in trace.runtime_metadata
        reranker_meta = trace.runtime_metadata["reranker"]
        assert reranker_meta["reranker_id"] == RERANKER_ID
        assert reranker_meta["reranker_version"] == "0.1"
        assert reranker_meta["candidate_top_k"] == 10
        assert reranker_meta["final_top_k"] == 3
        assert reranker_meta["backend"] == "sentence-transformers"
        assert reranker_meta["backend_version"] == BACKEND_VERSION
        assert reranker_meta["model_name"] == MODEL_NAME
        assert reranker_meta["model_revision"] == MODEL_REVISION
        assert reranker_meta["max_length"] == MAX_LENGTH
        assert reranker_meta["score_activation"] == "identity"
        assert reranker_meta["score_semantics"] == "cross_encoder_logit"
        assert isinstance(reranker_meta["device"], str) and reranker_meta["device"]
        blob = json.dumps(reranker_meta)
        assert "/home" not in blob
        assert "/Users" not in blob


class TestControlledComparison:
    """Identity vs CrossEncoder on shared development fixtures.

    Same corpus, cases, retrieval config, candidate/final depth, system,
    generation fixture, evaluator, and seed — only the reranker strategy
    differs. Observed differences are reported, never claimed as formal
    benchmark conclusions.
    """

    def test_same_variables_except_reranker(
        self, identity_manifest, cross_manifest
    ):
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
            assert getattr(identity_manifest, field) == getattr(
                cross_manifest, field
            )
        assert identity_manifest.retrieval == cross_manifest.retrieval
        assert identity_manifest.reranker.parameters["candidate_top_k"] == 10
        assert cross_manifest.reranker.parameters["candidate_top_k"] == 10
        assert identity_manifest.config_fingerprint() != (
            cross_manifest.config_fingerprint()
        )

    def test_observed_development_difference(
        self,
        identity_manifest,
        cross_manifest,
        identity_system,
        cross_system,
        rag_cases,
    ):
        improved = unchanged = worsened = 0
        for case_id, case in rag_cases.items():
            identity_trace = run_case(case, identity_manifest, identity_system)
            cross_trace = run_case(case, cross_manifest, cross_system)
            # Same candidates enter both rerankers (same Hybrid retrieval).
            assert (
                identity_trace.retrieval_events[0].document_ids
                == cross_trace.retrieval_events[0].document_ids
            )
            identity_final = identity_trace.retrieval_events[1].document_ids
            cross_final = cross_trace.retrieval_events[1].document_ids
            assert set(cross_final) <= set(
                cross_trace.retrieval_events[0].document_ids
            )
            if cross_final == identity_final:
                unchanged += 1
            else:
                # Rank the relevant doc (if any) in each final ranking.
                relevant = set(case.relevant_document_ids)
                def rank(ids):
                    for index, document_id in enumerate(ids, start=1):
                        if document_id in relevant:
                            return index
                    return None

                identity_rank = rank(identity_final)
                cross_rank = rank(cross_final)
                if identity_rank is None and cross_rank is not None:
                    improved += 1
                elif identity_rank is not None and cross_rank is None:
                    worsened += 1
                elif (
                    identity_rank is not None
                    and cross_rank is not None
                    and cross_rank < identity_rank
                ):
                    improved += 1
                elif (
                    identity_rank is not None
                    and cross_rank is not None
                    and cross_rank > identity_rank
                ):
                    worsened += 1
                else:
                    unchanged += 1
        # The comparison must execute over all 11 cases; no improvement is
        # required for Phase 2B to pass.
        assert improved + unchanged + worsened == 11
