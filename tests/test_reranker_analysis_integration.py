"""Phase 2E reporting integration coverage (real Phase 2D traces).

Every test here is marked ``integration`` and excluded from the default
``pytest`` run. The traces are produced with the real pinned Hybrid +
CrossEncoder stack; the reporting layer itself (analyze / compare /
summarize) performs zero model calls over those persisted traces.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from commercebench.contracts import ExperimentManifest
from commercebench.evaluation import DeterministicRAGEvaluator
from commercebench.reporting import (
    RerankerCaseAnalysis,
    analyze_reranker_case,
    compare_reranker_runs,
    summarize_reranker_comparison,
)
from commercebench.runner import run_case
from commercebench.systems import DeterministicRAGSystem

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


@pytest.fixture(scope="module")
def evaluator():
    return DeterministicRAGEvaluator(ks=(1, 3))


@pytest.fixture(scope="module")
def identity_analyses(
    identity_manifest, identity_system, rag_cases, evaluator
):
    analyses = {}
    traces = {}
    for case_id, case in rag_cases.items():
        trace = run_case(case, identity_manifest, identity_system)
        evaluation = evaluator.evaluate(case, trace)
        analyses[case_id] = analyze_reranker_case(case, trace, evaluation)
        traces[case_id] = trace
    return analyses, traces


@pytest.fixture(scope="module")
def cross_analyses(cross_manifest, cross_system, rag_cases, evaluator):
    analyses = {}
    traces = {}
    for case_id, case in rag_cases.items():
        trace = run_case(case, cross_manifest, cross_system)
        evaluation = evaluator.evaluate(case, trace)
        analyses[case_id] = analyze_reranker_case(case, trace, evaluation)
        traces[case_id] = trace
    return analyses, traces


@pytest.fixture(scope="module")
def comparisons(identity_analyses, cross_analyses):
    identity, identity_traces = identity_analyses
    cross, cross_traces = cross_analyses
    assert set(identity) == set(cross)
    return {
        case_id: compare_reranker_runs(
            identity[case_id],
            cross[case_id],
            identity_traces[case_id].output_text,
            cross_traces[case_id].output_text,
        )
        for case_id in sorted(identity)
    }


class TestRealIdentityControl:
    def test_zero_movement_all_cases(self, identity_analyses, rag_cases):
        analyses, _ = identity_analyses
        assert len(analyses) == 11
        for case_id, analysis in analyses.items():
            assert isinstance(analysis, RerankerCaseAnalysis)
            assert all(m.rank_delta == 0 for m in analysis.rank_movements)
            assert analysis.top_k_entered == ()
            assert analysis.top_k_exited == ()
            assert analysis.top1_changed is False
            assert analysis.candidate_top1 == analysis.final_top1
            if rag_cases[case_id].relevant_document_ids:
                assert analysis.mrr_delta == pytest.approx(0.0)
                assert analysis.recall_delta == pytest.approx(0.0)
                assert analysis.mrr_cross_check is True
            else:
                assert analysis.final_mrr is None
                assert analysis.mrr_delta is None


class TestRealCrossAnalysis:
    def test_analyses_constructible_and_cross_checked(
        self, cross_analyses, rag_cases
    ):
        analyses, traces = cross_analyses
        assert len(analyses) == 11
        for case_id, analysis in analyses.items():
            full = traces[case_id].runtime_metadata["reranker"][
                "full_ranking"
            ]
            assert analysis.candidate_count == len(full["document_ids"])
            assert len(analysis.rank_movements) == analysis.candidate_count
            assert (
                analysis.final_top_k
                == traces[case_id].runtime_metadata["reranker"][
                    "final_top_k"
                ]
            )
            if rag_cases[case_id].relevant_document_ids:
                assert analysis.mrr_cross_check is True
                assert analysis.recall_cross_check is True
            else:
                assert analysis.mrr_cross_check is None

    def test_cutoff_recoverable_when_dropped_exist(
        self, cross_analyses, rag_cases
    ):
        analyses, _ = cross_analyses
        seen_cutoff = False
        for case_id, analysis in analyses.items():
            if analysis.candidate_count > analysis.final_top_k:
                seen_cutoff = True
                assert analysis.cutoff_score is not None
                assert analysis.next_below_cutoff_score is not None
                assert analysis.cutoff_margin is not None
                assert (
                    analysis.cutoff_score >= analysis.next_below_cutoff_score
                )
        assert seen_cutoff


class TestRealKeyCase:
    def test_return_exchange_evidence(self, cross_analyses, rag_cases):
        analyses, traces = cross_analyses
        analysis = analyses["rag-return-exchange-001"]
        trace = traces["rag-return-exchange-001"]
        candidate, final = trace.retrieval_events
        assert analysis.candidate_top1 == "policy-exchange-001"
        assert analysis.final_top1 == "policy-return-001"
        assert analysis.top1_changed is True
        by_id = {m.document_id: m for m in analysis.rank_movements}
        assert by_id["policy-payment-001"].candidate_rank == 3
        assert by_id["policy-payment-001"].reranker_rank == 4
        assert by_id["policy-refund-001"].candidate_rank == 5
        assert by_id["policy-refund-001"].reranker_rank == 3
        assert analysis.top_k_entered == ("policy-refund-001",)
        assert analysis.top_k_exited == ("policy-payment-001",)
        assert analysis.cutoff_score == pytest.approx(-3.86, abs=0.05)
        assert analysis.next_below_cutoff_score == pytest.approx(
            -8.30, abs=0.05
        )
        assert analysis.cutoff_margin == pytest.approx(4.44, abs=0.05)
        # Metric equivalence without behavior equivalence.
        assert analysis.candidate_mrr == pytest.approx(1.0)
        assert analysis.final_mrr == pytest.approx(1.0)
        assert analysis.mrr_delta == pytest.approx(0.0)
        assert set(analysis.relevant_documents_in_final) == {
            "policy-return-001",
            "policy-exchange-001",
        }
        assert candidate.document_ids[0] == "policy-exchange-001"

    def test_metric_same_but_behavior_differs(self, comparisons):
        comparison = comparisons["rag-return-exchange-001"]
        assert comparison.mrr_delta == pytest.approx(0.0)
        assert comparison.top1_changed is True
        assert comparison.generation_changed is True


class TestRealExperimentSummary:
    def test_controlled_development_observation(self, comparisons):
        assert len(comparisons) == 11
        summary = summarize_reranker_comparison(list(comparisons.values()))
        assert summary.case_count == 11
        # Deterministic development observation (not a benchmark claim):
        # no relevant-rank movement, one top-1/generation change.
        assert summary.relevant_rank_improved_count == 0
        assert summary.relevant_rank_regressed_count == 0
        assert summary.top1_changed_count == 1
        assert summary.generation_changed_count == 1
        assert summary.mrr_applicable_count == 10
        assert summary.mrr_improved_count == 0
        assert summary.mrr_regressed_count == 0
        assert summary.mrr_unchanged_count == 10
        assert "overall_score" not in summary.to_dict()
        payload = json.dumps(summary.to_dict(), sort_keys=True)
        assert json.loads(payload)["case_count"] == 11


class TestReportingLoadsNoModel:
    def test_analyze_compare_summarize_without_models(
        self, monkeypatch, cross_analyses, identity_analyses, rag_cases
    ):
        import commercebench.rag.retrievers as retrievers
        import commercebench.reranking.rerankers as rerankers
        import commercebench.reranking.cross_encoder as cross_encoder

        def _boom(*args, **kwargs):
            raise AssertionError("reporting must not touch model runtimes")

        monkeypatch.setattr(retrievers, "build_retriever", _boom)
        monkeypatch.setattr(rerankers, "build_reranker", _boom)
        monkeypatch.setattr(
            cross_encoder.CrossEncoderReranker, "_load_model", _boom
        )
        cross, cross_traces = cross_analyses
        identity, identity_traces = identity_analyses
        # Re-derive everything from persisted evidence only.
        for case_id, case in rag_cases.items():
            assert (
                analyze_reranker_case(
                    case, cross_traces[case_id]
                ).full_reranker_document_ids
                == cross[case_id].full_reranker_document_ids
            )
        comps = [
            compare_reranker_runs(
                identity[cid],
                cross[cid],
                identity_traces[cid].output_text,
                cross_traces[cid].output_text,
            )
            for cid in sorted(cross)
        ]
        summary = summarize_reranker_comparison(comps)
        assert summary.case_count == 11
