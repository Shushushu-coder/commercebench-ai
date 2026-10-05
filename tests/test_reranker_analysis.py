"""Phase 2E reranker failure analysis / reporting tests (offline).

All tests are fast, deterministic, and offline: traces are hand-built or
produced through stub retrievers and fake scorers, so no model is loaded
and no network is touched. Real pinned-model coverage lives in
``test_reranker_analysis_integration.py`` (``@pytest.mark.integration``).
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from commercebench.contracts import (
    SCHEMA_VERSION,
    Answerability,
    CaseSpec,
    ContractValidationError,
    ConversationType,
    Difficulty,
    RetrievalEvent,
    RetrievalRole,
    RunTrace,
)
from commercebench.evaluation import DeterministicRAGEvaluator
from commercebench.evaluation.retrieval import RetrievalEvaluator
from commercebench.rag import (
    Corpus,
    DeterministicRAGPipeline,
    Document,
    RetrievalResult,
)
from commercebench.reporting import (
    RankMovement,
    RerankerCaseAnalysis,
    RerankerComparison,
    RerankerComparisonSummary,
    analyze_reranker_case,
    compare_reranker_runs,
    render_reranker_case_summary,
    summarize_reranker_comparison,
)
from commercebench.reranking import (
    CrossEncoderReranker,
    IdentityReranker,
)

IDENTITY_ID = "identity-reranker-v0"
CROSS_ID = "cross-encoder-msmarco-minilm-l6-v2-v0"


def _case(relevant=(), case_id="case-1") -> CaseSpec:
    return CaseSpec(
        schema_version=SCHEMA_VERSION,
        case_id=case_id,
        benchmark_version="commercebench-dev-0.1",
        task_family="grounded_qa",
        intent="return",
        difficulty=Difficulty.EASY,
        conversation_type=ConversationType.SINGLE_TURN,
        answerability=Answerability.ANSWERABLE,
        user_input="q",
        required_facts=(),
        forbidden_claims=(),
        relevant_document_ids=tuple(relevant),
        metadata={},
    )


def _trace(
    candidate_ids,
    candidate_scores,
    full_ids,
    full_scores,
    final_ids,
    final_scores,
    reranker_id=CROSS_ID,
    final_top_k=3,
    output_text="answer",
    query="q",
    case_id="case-1",
    candidate_query=None,
    final_query=None,
    omit_full_ranking=False,
) -> RunTrace:
    candidate = RetrievalEvent(
        query=candidate_query if candidate_query is not None else query,
        document_ids=tuple(candidate_ids),
        scores=tuple(candidate_scores),
        role=RetrievalRole.CANDIDATE,
    )
    final = RetrievalEvent(
        query=final_query if final_query is not None else query,
        document_ids=tuple(final_ids),
        scores=tuple(final_scores),
        role=RetrievalRole.FINAL,
    )
    reranker_meta: dict = {
        "reranker_id": reranker_id,
        "reranker_version": "0.1",
        "candidate_top_k": len(candidate_ids),
        "final_top_k": final_top_k,
    }
    if not omit_full_ranking:
        reranker_meta["full_ranking"] = {
            "document_ids": list(full_ids),
            "scores": list(full_scores),
        }
    return RunTrace(
        schema_version=SCHEMA_VERSION,
        run_id="run-0001",
        experiment_id="exp",
        case_id=case_id,
        system_id="deterministic-rag",
        system_version="0.1",
        config_fingerprint="abc",
        input_text=query,
        output_text=output_text,
        started_at="2026-01-01T00:00:00+00:00",
        finished_at="2026-01-01T00:00:00.010000+00:00",
        latency_ms=10.0,
        retrieval_events=(candidate, final),
        runtime_metadata={
            "retriever_id": "hybrid-bm25-minilm-rrf-v0",
            "retriever_version": "0.1",
            "corpus_id": "c",
            "corpus_version": "1",
            "top_k": final_top_k,
            "reranker": reranker_meta,
        },
    )


def _evaluate(case, trace):
    return RetrievalEvaluator(ks=(1, 3)).evaluate(case, trace)


def _standard_trace(**overrides) -> RunTrace:
    """d1..d4 candidates; full d2,d4,d1,d3; final top2 d2,d4."""
    kwargs = dict(
        candidate_ids=("d1", "d2", "d3", "d4"),
        candidate_scores=(0.4, 0.3, 0.2, 0.1),
        full_ids=("d2", "d4", "d1", "d3"),
        full_scores=(4.0, 2.0, 0.1, -2.0),
        final_ids=("d2", "d4"),
        final_scores=(4.0, 2.0),
        final_top_k=2,
    )
    kwargs.update(overrides)
    return _trace(**kwargs)


class TestRankMovement:
    def test_moved_up_down_unchanged(self):
        analysis = analyze_reranker_case(
            _case(relevant=("d1",)), _standard_trace()
        )
        by_id = {m.document_id: m for m in analysis.rank_movements}
        assert by_id["d2"].rank_delta == 1  # 2 -> 1, moved up
        assert by_id["d4"].rank_delta == 2  # 4 -> 2, moved up
        assert by_id["d1"].rank_delta == -2  # 1 -> 3, moved down
        assert by_id["d3"].rank_delta == -1  # 3 -> 4, moved down
        assert by_id["d2"].candidate_rank == 2
        assert by_id["d2"].reranker_rank == 1

    def test_unchanged_rank(self):
        trace = _trace(
            candidate_ids=("d1", "d2"),
            candidate_scores=(0.5, 0.4),
            full_ids=("d1", "d2"),
            full_scores=(9.0, 1.0),
            final_ids=("d1",),
            final_scores=(9.0,),
            final_top_k=1,
        )
        analysis = analyze_reranker_case(_case(relevant=("d1",)), trace)
        assert analysis.rank_movements[0].rank_delta == 0

    def test_relevant_flag(self):
        analysis = analyze_reranker_case(
            _case(relevant=("d2",)), _standard_trace()
        )
        by_id = {m.document_id: m for m in analysis.rank_movements}
        assert by_id["d2"].relevant is True
        assert by_id["d1"].relevant is False

    def test_movements_ordered_by_candidate_rank(self):
        analysis = analyze_reranker_case(_case(), _standard_trace())
        assert [m.candidate_rank for m in analysis.rank_movements] == [1, 2, 3, 4]
        assert [m.document_id for m in analysis.rank_movements] == [
            "d1",
            "d2",
            "d3",
            "d4",
        ]

    def test_scores_kept_in_own_domains(self):
        analysis = analyze_reranker_case(_case(), _standard_trace())
        by_id = {m.document_id: m for m in analysis.rank_movements}
        assert by_id["d2"].candidate_score == pytest.approx(0.3)
        assert by_id["d2"].reranker_score == pytest.approx(4.0)
        # Scores are never subtracted across stages: no such field exists.
        assert "candidate_score_delta" not in analysis.to_dict()
        assert "score_delta" not in json.dumps(analysis.to_dict())

    def test_positive_delta_means_moved_up_documented(self):
        movement = RankMovement(
            document_id="d",
            candidate_rank=4,
            reranker_rank=1,
            rank_delta=3,
            candidate_score=0.1,
            reranker_score=9.0,
            relevant=False,
            candidate_in_final_cutoff=False,
            final_in_final_cutoff=True,
        )
        assert movement.rank_delta > 0
        with pytest.raises(ContractValidationError):
            RankMovement(
                document_id="d",
                candidate_rank=4,
                reranker_rank=1,
                rank_delta=-3,
                candidate_score=0.1,
                reranker_score=9.0,
                relevant=False,
                candidate_in_final_cutoff=False,
                final_in_final_cutoff=True,
            )


class TestCutoff:
    def test_normal_cutoff(self):
        analysis = analyze_reranker_case(_case(), _standard_trace())
        assert analysis.cutoff_score == pytest.approx(2.0)
        assert analysis.next_below_cutoff_score == pytest.approx(0.1)
        assert analysis.cutoff_margin == pytest.approx(1.9)

    def test_no_below_cutoff_doc_is_na(self):
        trace = _trace(
            candidate_ids=("d1", "d2"),
            candidate_scores=(0.5, 0.4),
            full_ids=("d1", "d2"),
            full_scores=(9.0, 1.0),
            final_ids=("d1", "d2"),
            final_scores=(9.0, 1.0),
            final_top_k=3,
        )
        analysis = analyze_reranker_case(_case(), trace)
        assert analysis.cutoff_score is None
        assert analysis.next_below_cutoff_score is None
        assert analysis.cutoff_margin is None

    def test_negative_logits_cutoff(self):
        trace = _trace(
            candidate_ids=("d1", "d2", "d3"),
            candidate_scores=(0.5, 0.4, 0.3),
            full_ids=("d1", "d2", "d3"),
            full_scores=(2.0, -1.0, -5.0),
            final_ids=("d1", "d2"),
            final_scores=(2.0, -1.0),
            final_top_k=2,
        )
        analysis = analyze_reranker_case(_case(), trace)
        assert analysis.cutoff_score == pytest.approx(-1.0)
        assert analysis.next_below_cutoff_score == pytest.approx(-5.0)
        assert analysis.cutoff_margin == pytest.approx(4.0)

    def test_tie_at_cutoff(self):
        trace = _trace(
            candidate_ids=("d1", "d2", "d3"),
            candidate_scores=(0.5, 0.4, 0.3),
            full_ids=("d1", "d2", "d3"),
            full_scores=(5.0, 3.0, 3.0),
            final_ids=("d1", "d2"),
            final_scores=(5.0, 3.0),
            final_top_k=2,
        )
        analysis = analyze_reranker_case(_case(), trace)
        assert analysis.cutoff_margin == pytest.approx(0.0)


class TestTopKMembership:
    def test_entered_and_exited(self):
        analysis = analyze_reranker_case(_case(), _standard_trace())
        # Candidate window top2: d1,d2. Final top2: d2,d4.
        assert analysis.top_k_entered == ("d4",)
        assert analysis.top_k_exited == ("d1",)

    def test_unchanged_membership(self):
        trace = _trace(
            candidate_ids=("d1", "d2", "d3"),
            candidate_scores=(0.5, 0.4, 0.3),
            full_ids=("d1", "d2", "d3"),
            full_scores=(9.0, 1.0, -1.0),
            final_ids=("d1", "d2"),
            final_scores=(9.0, 1.0),
            final_top_k=2,
        )
        analysis = analyze_reranker_case(_case(), trace)
        assert analysis.top_k_entered == ()
        assert analysis.top_k_exited == ()

    def test_top1_change(self):
        analysis = analyze_reranker_case(_case(), _standard_trace())
        assert analysis.candidate_top1 == "d1"
        assert analysis.final_top1 == "d2"
        assert analysis.top1_changed is True
        assert analysis.generation_driving_document_id == "d2"
        assert analysis.generation_driver_changed is True

    def test_top1_unchanged(self):
        trace = _trace(
            candidate_ids=("d1", "d2"),
            candidate_scores=(0.5, 0.4),
            full_ids=("d1", "d2"),
            full_scores=(9.0, 1.0),
            final_ids=("d1",),
            final_scores=(9.0,),
            final_top_k=1,
        )
        analysis = analyze_reranker_case(_case(), trace)
        assert analysis.top1_changed is False
        assert analysis.generation_driving_document_id == "d1"


class TestRelevance:
    def test_single_relevant_doc(self):
        analysis = analyze_reranker_case(
            _case(relevant=("d1",)), _standard_trace()
        )
        assert analysis.candidate_relevant_rank == 1
        assert analysis.reranker_relevant_rank == 3
        assert analysis.final_relevant_rank is None
        assert analysis.candidate_mrr == pytest.approx(1.0)
        assert analysis.final_mrr == pytest.approx(0.0)
        assert analysis.mrr_delta == pytest.approx(-1.0)
        assert analysis.relevant_rank_regressed is True
        assert analysis.relevant_rank_improved is False
        assert analysis.relevant_doc_exited_final_top_k is True
        assert analysis.relevant_doc_entered_final_top_k is False

    def test_multiple_relevant_docs_best_rank(self):
        # relevant d1 (cand 1, rerank 3) and d3 (cand 3, rerank 4).
        analysis = analyze_reranker_case(
            _case(relevant=("d1", "d3")), _standard_trace()
        )
        assert analysis.candidate_relevant_rank == 1
        assert analysis.reranker_relevant_rank == 3
        assert analysis.final_relevant_rank is None
        assert analysis.candidate_mrr == pytest.approx(1.0)
        assert analysis.final_mrr == pytest.approx(0.0)

    def test_relevant_doc_enters_final(self):
        # relevant d4: candidate rank 4 -> reranker rank 2 (in final top2).
        # Window baseline candidate[:2] misses d4 (0.0); final hits at 2.
        analysis = analyze_reranker_case(
            _case(relevant=("d4",)), _standard_trace()
        )
        assert analysis.candidate_relevant_rank == 4
        assert analysis.reranker_relevant_rank == 2
        assert analysis.final_relevant_rank == 2
        assert analysis.relevant_rank_improved is True
        assert analysis.relevant_doc_entered_final_top_k is True
        assert analysis.candidate_mrr == pytest.approx(0.0)
        assert analysis.final_mrr == pytest.approx(0.5)
        assert analysis.mrr_delta == pytest.approx(0.5)

    def test_no_relevant_docs_is_na(self):
        case = _case(relevant=())
        trace = _standard_trace()
        evaluation = _evaluate(case, trace)
        analysis = analyze_reranker_case(case, trace, evaluation)
        assert analysis.candidate_relevant_rank is None
        assert analysis.reranker_relevant_rank is None
        assert analysis.final_relevant_rank is None
        assert analysis.candidate_mrr is None
        assert analysis.final_mrr is None
        assert analysis.mrr_delta is None
        assert analysis.candidate_recall_at_k is None
        assert analysis.final_recall_at_k is None
        assert analysis.recall_delta is None
        # Cross-checks stay silent on N/A cases instead of failing.
        assert analysis.mrr_cross_check is None
        assert analysis.recall_cross_check is None

    def test_recall_uses_candidate_window_at_final_k(self):
        # relevant d1,d3: candidate window top2 [d1,d2] recall 0.5;
        # final top2 [d2,d4] recall 0.0 — never candidate top-4 recall.
        analysis = analyze_reranker_case(
            _case(relevant=("d1", "d3")), _standard_trace()
        )
        assert analysis.candidate_recall_at_k == pytest.approx(0.5)
        assert analysis.final_recall_at_k == pytest.approx(0.0)
        assert analysis.recall_delta == pytest.approx(-0.5)

    def test_relevant_membership_lists(self):
        analysis = analyze_reranker_case(
            _case(relevant=("d1", "d4")), _standard_trace()
        )
        # Candidate window top2 [d1,d2]: only d1 relevant.
        assert analysis.relevant_documents_in_candidate_window == ("d1",)
        # Final top2 [d2,d4]: only d4 relevant.
        assert analysis.relevant_documents_in_final == ("d4",)
        assert analysis.relevant_doc_entered_final_top_k is True
        assert analysis.relevant_doc_exited_final_top_k is True

    def test_candidate_mrr_uses_window_not_full_ranking(self):
        # relevant d3 sits at full candidate rank 3, outside the top2
        # window: the rank stays visible but the depth-matched window
        # baseline is a miss (0.0), never 1/3.
        analysis = analyze_reranker_case(
            _case(relevant=("d3",)), _standard_trace()
        )
        assert analysis.candidate_relevant_rank == 3
        assert analysis.candidate_mrr == pytest.approx(0.0)
        assert analysis.final_mrr == pytest.approx(0.0)
        assert analysis.mrr_delta == pytest.approx(0.0)

    def test_empty_rankings(self):
        trace = _trace(
            candidate_ids=[],
            candidate_scores=[],
            full_ids=[],
            full_scores=[],
            final_ids=[],
            final_scores=[],
            final_top_k=3,
        )
        analysis = analyze_reranker_case(_case(relevant=("d1",)), trace)
        assert analysis.rank_movements == ()
        assert analysis.candidate_top1 is None
        assert analysis.final_top1 is None
        assert analysis.generation_driving_document_id is None
        assert analysis.candidate_mrr == pytest.approx(0.0)
        assert analysis.final_mrr == pytest.approx(0.0)
        assert analysis.mrr_delta == pytest.approx(0.0)
        assert analysis.cutoff_score is None


class TestEvaluationCrossCheck:
    def test_final_mrr_matches_evaluation(self):
        case = _case(relevant=("d2", "d4"))
        trace = _standard_trace()
        evaluation = _evaluate(case, trace)
        analysis = analyze_reranker_case(case, trace, evaluation)
        assert analysis.final_mrr == pytest.approx(
            evaluation.metrics["retrieval_mrr"].value
        )
        assert analysis.mrr_cross_check is True

    def test_final_recall_matches_evaluation(self):
        case = _case(relevant=("d2", "d4"))
        trace = _standard_trace()  # final_top_k=2, ks=(1,3) lacks recall@2
        evaluation = RetrievalEvaluator(ks=(1, 2, 3)).evaluate(case, trace)
        analysis = analyze_reranker_case(case, trace, evaluation)
        assert analysis.final_recall_at_k == pytest.approx(
            evaluation.metrics["retrieval_recall_at_2"].value
        )
        assert analysis.recall_cross_check is True

    def test_recall_check_skipped_when_metric_key_absent(self):
        case = _case(relevant=("d2",))
        trace = _standard_trace()  # final_top_k=2; ks=(1,3) has no recall@2
        evaluation = _evaluate(case, trace)
        assert "retrieval_recall_at_2" not in evaluation.metrics
        analysis = analyze_reranker_case(case, trace, evaluation)
        assert analysis.mrr_cross_check is True
        assert analysis.recall_cross_check is None

    def test_tampered_evaluation_rejected(self):
        case = _case(relevant=("d2",))
        trace = _standard_trace()
        evaluation = _evaluate(case, trace)
        tampered_metrics = dict(evaluation.metrics)
        original = tampered_metrics["retrieval_mrr"]
        tampered_metrics["retrieval_mrr"] = dataclasses.replace(
            original, value=0.0
        )
        tampered = dataclasses.replace(
            evaluation, metrics=tampered_metrics
        )
        with pytest.raises(ContractValidationError, match="cross-check"):
            analyze_reranker_case(case, trace, tampered)

    def test_wrong_run_evaluation_rejected(self):
        case = _case(relevant=("d2",))
        trace = _standard_trace()
        evaluation = _evaluate(case, trace)
        other = dataclasses.replace(trace, run_id="run-9999")
        other_evaluation = _evaluate(case, other)
        with pytest.raises(ContractValidationError, match="run_id"):
            analyze_reranker_case(case, trace, other_evaluation)

    def test_combined_evaluator_cross_check(self):
        case = _case(relevant=("d2",))
        trace = _standard_trace()
        evaluation = DeterministicRAGEvaluator(ks=(1, 2, 3)).evaluate(
            case, trace
        )
        analysis = analyze_reranker_case(case, trace, evaluation)
        assert analysis.mrr_cross_check is True
        assert analysis.recall_cross_check is True


class TestCorruptEvidence:
    def test_missing_diagnostics_rejected(self):
        trace = _standard_trace(omit_full_ranking=True)
        with pytest.raises(
            ContractValidationError, match="diagnostic evidence unavailable"
        ):
            analyze_reranker_case(_case(), trace)

    def test_no_reranker_trace_rejected(self):
        legacy = RetrievalEvent(
            query="q", document_ids=("d1",), scores=(1.0,), role=None
        )
        trace = dataclasses.replace(
            _standard_trace(), retrieval_events=(legacy,)
        )
        with pytest.raises(ContractValidationError, match="exactly one"):
            analyze_reranker_case(_case(), trace)

    def test_final_not_prefix_rejected(self):
        trace = _standard_trace(final_ids=("d1", "d2"))
        with pytest.raises(ContractValidationError, match="prefix"):
            analyze_reranker_case(_case(), trace)

    def test_mismatched_sets_rejected(self):
        trace = _standard_trace(
            full_ids=("d2", "d4", "d1", "dX"),
            full_scores=(4.0, 2.0, 0.1, -2.0),
        )
        with pytest.raises(ContractValidationError, match="candidate set"):
            analyze_reranker_case(_case(), trace)

    def test_short_diagnostics_rejected(self):
        trace = _standard_trace(
            full_ids=("d2", "d4", "d1"), full_scores=(4.0, 2.0, 0.1)
        )
        with pytest.raises(ContractValidationError, match="candidate set"):
            analyze_reranker_case(_case(), trace)

    def test_duplicate_candidate_ids_rejected(self):
        trace = _standard_trace(
            candidate_ids=("d1", "d1", "d3", "d4"),
            candidate_scores=(0.4, 0.3, 0.2, 0.1),
        )
        with pytest.raises(ContractValidationError, match="duplicates"):
            analyze_reranker_case(_case(), trace)

    def test_duplicate_full_ids_rejected(self):
        trace = _standard_trace(
            full_ids=("d2", "d4", "d1", "d1"),
            full_scores=(4.0, 2.0, 0.1, -2.0),
        )
        with pytest.raises(ContractValidationError, match="duplicates"):
            analyze_reranker_case(_case(), trace)

    def test_length_mismatch_rejected(self):
        trace = _standard_trace(full_scores=(4.0, 2.0, 0.1))
        with pytest.raises(ContractValidationError, match="corrupt"):
            analyze_reranker_case(_case(), trace)

    def test_query_mismatch_rejected(self):
        trace = _standard_trace(final_query="other query")
        with pytest.raises(ContractValidationError, match="queries differ"):
            analyze_reranker_case(_case(), trace)

    def test_case_mismatch_rejected(self):
        other = _case(relevant=())
        other = dataclasses.replace(other, case_id="case-2")
        with pytest.raises(ContractValidationError, match="case_id"):
            analyze_reranker_case(other, _standard_trace())


class TestIdentityControl:
    def _run_identity_pipeline(self, relevant=()):
        class StubRetriever:
            retriever_id = "hybrid-bm25-minilm-rrf-v0"
            retriever_version = "0.1"

            def retrieve(self, query, corpus, top_k):
                ids = ("d1", "d2", "d3", "d4")[:top_k]
                scores = (0.4, 0.3, 0.2, 0.1)[: len(ids)]
                return RetrievalResult(
                    query=query, document_ids=ids, scores=scores
                )

            def runtime_metadata(self):
                return {}

        corpus = Corpus(
            corpus_id="c",
            corpus_version="1",
            documents=tuple(
                Document(document_id=d, text=f"text-{d}")
                for d in ("d1", "d2", "d3", "d4")
            ),
        )
        pipeline = DeterministicRAGPipeline(
            retriever=StubRetriever(),
            corpus=corpus,
            top_k=2,
            reranker=IdentityReranker(candidate_top_k=4),
        )
        result = pipeline.run("q")
        candidate = RetrievalEvent(
            query="q",
            document_ids=result.candidate_event.document_ids,
            scores=result.candidate_event.scores,
            role=RetrievalRole.CANDIDATE,
        )
        final = RetrievalEvent(
            query="q",
            document_ids=result.retrieval_event.document_ids,
            scores=result.retrieval_event.scores,
            role=RetrievalRole.FINAL,
        )
        return RunTrace(
            schema_version=SCHEMA_VERSION,
            run_id="run-0001",
            experiment_id="exp",
            case_id="case-1",
            system_id="deterministic-rag",
            system_version="0.1",
            config_fingerprint="abc",
            input_text="q",
            output_text="answer",
            started_at="2026-01-01T00:00:00+00:00",
            finished_at="2026-01-01T00:00:00.010000+00:00",
            latency_ms=10.0,
            retrieval_events=(candidate, final),
            runtime_metadata=dict(result.runtime_metadata),
        )

    def test_identity_zero_movement(self):
        case = _case(relevant=("d3",))
        trace = self._run_identity_pipeline()
        evaluation = _evaluate(case, trace)
        analysis = analyze_reranker_case(case, trace, evaluation)
        assert analysis.reranker_id == IDENTITY_ID
        assert all(m.rank_delta == 0 for m in analysis.rank_movements)
        assert analysis.top_k_entered == ()
        assert analysis.top_k_exited == ()
        assert analysis.top1_changed is False
        assert analysis.candidate_top1 == analysis.final_top1 == "d1"
        assert analysis.mrr_delta == pytest.approx(0.0)
        assert analysis.recall_delta == pytest.approx(0.0)
        assert analysis.relevant_rank_improved is False
        assert analysis.relevant_rank_regressed is False
        assert analysis.mrr_cross_check is True


class TestCrossEncoderPipelineTrace:
    def test_fake_scorer_trace_analyzes(self):
        corpus = Corpus(
            corpus_id="c",
            corpus_version="1",
            documents=tuple(
                Document(document_id=d, text=f"text-{d}")
                for d in ("d1", "d2", "d3", "d4")
            ),
        )

        class StubRetriever:
            retriever_id = "stub-retriever-v0"
            retriever_version = "0.1"

            def retrieve(self, query, corpus, top_k):
                return RetrievalResult(
                    query=query,
                    document_ids=("d1", "d2", "d3", "d4")[:top_k],
                    scores=(0.4, 0.3, 0.2, 0.1)[:top_k],
                )

            def runtime_metadata(self):
                return {}

        def scorer(query, candidates):
            return [
                {"d1": 0.1, "d2": 4.0, "d3": -2.0, "d4": 2.0}[c.document_id]
                for c in candidates
            ]

        pipeline = DeterministicRAGPipeline(
            retriever=StubRetriever(),
            corpus=corpus,
            top_k=2,
            reranker=CrossEncoderReranker(candidate_top_k=4, scorer=scorer),
        )
        result = pipeline.run("q")
        trace = RunTrace(
            schema_version=SCHEMA_VERSION,
            run_id="run-0001",
            experiment_id="exp",
            case_id="case-1",
            system_id="deterministic-rag",
            system_version="0.1",
            config_fingerprint="abc",
            input_text="q",
            output_text="answer",
            started_at="2026-01-01T00:00:00+00:00",
            finished_at="2026-01-01T00:00:00.010000+00:00",
            latency_ms=10.0,
            retrieval_events=(
                RetrievalEvent(
                    query="q",
                    document_ids=result.candidate_event.document_ids,
                    scores=result.candidate_event.scores,
                    role=RetrievalRole.CANDIDATE,
                ),
                RetrievalEvent(
                    query="q",
                    document_ids=result.retrieval_event.document_ids,
                    scores=result.retrieval_event.scores,
                    role=RetrievalRole.FINAL,
                ),
            ),
            runtime_metadata={
                "retriever_id": "hybrid-bm25-minilm-rrf-v0",
                "retriever_version": "0.1",
                "corpus_id": "c",
                "corpus_version": "1",
                "top_k": 2,
                "reranker": dict(result.runtime_metadata["reranker"]),
            },
        )
        case = _case(relevant=("d4",))
        analysis = analyze_reranker_case(case, trace, _evaluate(case, trace))
        assert analysis.full_reranker_document_ids == ("d2", "d4", "d1", "d3")
        assert analysis.top_k_entered == ("d4",)
        assert analysis.top_k_exited == ("d1",)
        assert analysis.mrr_cross_check is True

    def test_analysis_needs_no_model(self, monkeypatch):
        # Even with model loading hard-blocked, persisted evidence analyzes.
        from commercebench.reranking import CrossEncoderReranker as _CER

        def _boom(self, *args, **kwargs):
            raise AssertionError("reporting must not load a model")

        monkeypatch.setattr(_CER, "_load_model", _boom)
        case = _case(relevant=("d2",))
        trace = _standard_trace()
        analysis = analyze_reranker_case(case, trace, _evaluate(case, trace))
        assert analysis.final_top1 == "d2"
        assert analysis.mrr_cross_check is True


class TestSerializationDeterminism:
    def test_json_roundtrip_and_determinism(self):
        case = _case(relevant=("d1", "d4"))
        trace = _standard_trace()
        first = analyze_reranker_case(case, trace, _evaluate(case, trace))
        second = analyze_reranker_case(case, trace, _evaluate(case, trace))
        assert first == second
        payload_a = json.dumps(first.to_dict(), sort_keys=True)
        payload_b = json.dumps(second.to_dict(), sort_keys=True)
        assert payload_a == payload_b
        # JSON is self-contained: numbers stay numbers, nulls stay null.
        restored = RerankerCaseAnalysis.from_dict(json.loads(payload_a))
        assert restored == first
        blob = json.loads(payload_a)
        assert isinstance(blob["candidate_mrr"], float)
        assert isinstance(blob["rank_movements"], list)

    def test_render_summary_deterministic(self):
        analysis = analyze_reranker_case(
            _case(relevant=("d4",)), _standard_trace()
        )
        first = render_reranker_case_summary(analysis)
        second = render_reranker_case_summary(
            RerankerCaseAnalysis.from_dict(
                json.loads(json.dumps(analysis.to_dict()))
            )
        )
        assert first == second
        assert "case-1" in first
        assert "cutoff_margin" in first


def _analysis_for_comparison(
    reranker_id,
    full_ids,
    full_scores,
    final_ids,
    final_scores,
    output,
    case_id="case-1",
):
    case = _case(relevant=("d4",), case_id=case_id)
    trace = _trace(
        candidate_ids=("d1", "d2", "d3", "d4"),
        candidate_scores=(0.4, 0.3, 0.2, 0.1),
        full_ids=full_ids,
        full_scores=full_scores,
        final_ids=final_ids,
        final_scores=final_scores,
        reranker_id=reranker_id,
        final_top_k=2,
        output_text=output,
        case_id=case_id,
    )
    return analyze_reranker_case(case, trace, _evaluate(case, trace)), trace


class TestComparison:
    def test_identity_vs_cross_encoder(self):
        baseline, base_trace = _analysis_for_comparison(
            IDENTITY_ID,
            ("d1", "d2", "d3", "d4"),
            (0.4, 0.3, 0.2, 0.1),
            ("d1", "d2"),
            (0.4, 0.3),
            "answer-A",
        )
        treatment, treat_trace = _analysis_for_comparison(
            CROSS_ID,
            ("d2", "d4", "d1", "d3"),
            (4.0, 2.0, 0.1, -2.0),
            ("d2", "d4"),
            (4.0, 2.0),
            "answer-B",
        )
        comparison = compare_reranker_runs(
            baseline, treatment, base_trace.output_text, treat_trace.output_text
        )
        assert comparison.case_id == "case-1"
        assert comparison.baseline_reranker_id == IDENTITY_ID
        assert comparison.treatment_reranker_id == CROSS_ID
        assert comparison.baseline_final_top1 == "d1"
        assert comparison.treatment_final_top1 == "d2"
        assert comparison.top1_changed is True
        # relevant d4: baseline final misses (0.0), treatment final rank 2.
        assert comparison.baseline_mrr == pytest.approx(0.0)
        assert comparison.treatment_mrr == pytest.approx(0.5)
        assert comparison.mrr_delta == pytest.approx(0.5)
        assert comparison.generation_changed is True
        assert comparison.treatment_relevant_rank_improved is True

    def test_generation_unchanged(self):
        baseline, base_trace = _analysis_for_comparison(
            IDENTITY_ID,
            ("d1", "d2", "d3", "d4"),
            (0.4, 0.3, 0.2, 0.1),
            ("d1", "d2"),
            (0.4, 0.3),
            "same-answer",
        )
        treatment, treat_trace = _analysis_for_comparison(
            CROSS_ID,
            ("d1", "d4", "d2", "d3"),
            (4.0, 2.0, 0.1, -2.0),
            ("d1", "d4"),
            (4.0, 2.0),
            "same-answer",
        )
        comparison = compare_reranker_runs(
            baseline, treatment, base_trace.output_text, treat_trace.output_text
        )
        # top1 identical but membership changed; generation truly unchanged.
        assert comparison.top1_changed is False
        assert comparison.generation_changed is False

    def test_generation_compared_not_inferred(self):
        # Same top1, different outputs: output comparison decides.
        baseline, _ = _analysis_for_comparison(
            IDENTITY_ID,
            ("d1", "d2", "d3", "d4"),
            (0.4, 0.3, 0.2, 0.1),
            ("d1", "d2"),
            (0.4, 0.3),
            "output-one",
        )
        treatment, _ = _analysis_for_comparison(
            CROSS_ID,
            ("d1", "d2", "d3", "d4"),
            (0.4, 0.3, 0.2, 0.1),
            ("d1", "d2"),
            (0.4, 0.3),
            "output-two",
        )
        comparison = compare_reranker_runs(
            baseline, treatment, "output-one", "output-two"
        )
        assert comparison.top1_changed is False
        assert comparison.generation_changed is True

    def test_candidate_id_mismatch_rejected(self):
        baseline, _ = _analysis_for_comparison(
            IDENTITY_ID,
            ("d1", "d2", "d3", "d4"),
            (0.4, 0.3, 0.2, 0.1),
            ("d1", "d2"),
            (0.4, 0.3),
            "A",
        )
        case = _case(relevant=("d4",))
        other_trace = _trace(
            candidate_ids=("d1", "d2", "d3", "dX"),
            candidate_scores=(0.4, 0.3, 0.2, 0.1),
            full_ids=("d2", "dX", "d1", "d3"),
            full_scores=(4.0, 2.0, 0.1, -2.0),
            final_ids=("d2", "dX"),
            final_scores=(4.0, 2.0),
            final_top_k=2,
        )
        treatment = analyze_reranker_case(case, other_trace)
        with pytest.raises(ContractValidationError, match="candidate ranking"):
            compare_reranker_runs(baseline, treatment, "A", "B")

    def test_candidate_score_mismatch_rejected(self):
        baseline, _ = _analysis_for_comparison(
            IDENTITY_ID,
            ("d1", "d2", "d3", "d4"),
            (0.4, 0.3, 0.2, 0.1),
            ("d1", "d2"),
            (0.4, 0.3),
            "A",
        )
        case = _case(relevant=("d4",))
        other_trace = _trace(
            candidate_ids=("d1", "d2", "d3", "d4"),
            candidate_scores=(0.9, 0.3, 0.2, 0.1),
            full_ids=("d1", "d2", "d3", "d4"),
            full_scores=(0.9, 0.3, 0.2, 0.1),
            final_ids=("d1", "d2"),
            final_scores=(0.9, 0.3),
            final_top_k=2,
        )
        treatment = analyze_reranker_case(case, other_trace)
        with pytest.raises(ContractValidationError, match="candidate scores"):
            compare_reranker_runs(baseline, treatment, "A", "B")

    def test_case_mismatch_rejected(self):
        baseline, base_trace = _analysis_for_comparison(
            IDENTITY_ID,
            ("d1", "d2", "d3", "d4"),
            (0.4, 0.3, 0.2, 0.1),
            ("d1", "d2"),
            (0.4, 0.3),
            "A",
        )
        other = dataclasses.replace(baseline, case_id="case-2")
        with pytest.raises(ContractValidationError, match="same case_id"):
            compare_reranker_runs(
                baseline, other, base_trace.output_text, "B"
            )

    def test_comparison_roundtrip(self):
        baseline, base_trace = _analysis_for_comparison(
            IDENTITY_ID,
            ("d1", "d2", "d3", "d4"),
            (0.4, 0.3, 0.2, 0.1),
            ("d1", "d2"),
            (0.4, 0.3),
            "A",
        )
        treatment, treat_trace = _analysis_for_comparison(
            CROSS_ID,
            ("d2", "d4", "d1", "d3"),
            (4.0, 2.0, 0.1, -2.0),
            ("d2", "d4"),
            (4.0, 2.0),
            "B",
        )
        comparison = compare_reranker_runs(
            baseline, treatment, base_trace.output_text, treat_trace.output_text
        )
        assert (
            RerankerComparison.from_dict(
                json.loads(json.dumps(comparison.to_dict()))
            )
            == comparison
        )


class TestSummary:
    def _comparisons(self):
        baseline, base_trace = _analysis_for_comparison(
            IDENTITY_ID,
            ("d1", "d2", "d3", "d4"),
            (0.4, 0.3, 0.2, 0.1),
            ("d1", "d2"),
            (0.4, 0.3),
            "A",
        )
        treatment, treat_trace = _analysis_for_comparison(
            CROSS_ID,
            ("d2", "d4", "d1", "d3"),
            (4.0, 2.0, 0.1, -2.0),
            ("d2", "d4"),
            (4.0, 2.0),
            "B",
        )
        first = compare_reranker_runs(
            baseline, treatment, base_trace.output_text, treat_trace.output_text
        )
        # Second case: identical runs, no relevance -> N/A MRR.
        case2 = _case(relevant=(), case_id="case-2")
        trace_kwargs = dict(
            candidate_ids=("d1", "d2"),
            candidate_scores=(0.5, 0.4),
            full_ids=("d1", "d2"),
            full_scores=(9.0, 1.0),
            final_ids=("d1",),
            final_scores=(9.0,),
            final_top_k=1,
            case_id="case-2",
        )
        base2 = analyze_reranker_case(
            case2, _trace(reranker_id=IDENTITY_ID, **trace_kwargs)
        )
        treat2 = analyze_reranker_case(
            case2, _trace(reranker_id=CROSS_ID, **trace_kwargs)
        )
        second = compare_reranker_runs(base2, treat2, "S", "S")
        return first, second

    def test_counts_and_rates(self):
        first, second = self._comparisons()
        summary = summarize_reranker_comparison([first, second])
        assert summary.case_count == 2
        assert summary.top1_changed_count == 1
        assert summary.top1_change_rate == pytest.approx(0.5)
        assert summary.generation_changed_count == 1
        assert summary.generation_change_rate == pytest.approx(0.5)
        assert summary.relevant_rank_improved_count == 1
        assert summary.relevant_rank_regressed_count == 0
        # Only the first comparison is relevance-applicable.
        assert summary.mrr_applicable_count == 1
        assert summary.mrr_improved_count == 1
        assert summary.mrr_regressed_count == 0
        assert summary.mrr_unchanged_count == 0
        assert summary.mrr_improved_rate == pytest.approx(1.0)
        assert "overall_score" not in summary.to_dict()

    def test_duplicate_cases_rejected(self):
        first, _ = self._comparisons()
        with pytest.raises(ContractValidationError, match="distinct"):
            summarize_reranker_comparison([first, first])

    def test_summary_roundtrip(self):
        first, second = self._comparisons()
        summary = summarize_reranker_comparison([first, second])
        assert (
            RerankerComparisonSummary.from_dict(
                json.loads(json.dumps(summary.to_dict()))
            )
            == summary
        )

    def test_empty_summary(self):
        summary = summarize_reranker_comparison([])
        assert summary.case_count == 0
        assert summary.top1_change_rate is None
        assert summary.mrr_improved_rate is None
