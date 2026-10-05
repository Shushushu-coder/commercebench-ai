"""Phase 2G failure attribution reporting tests (offline, deterministic).

No model is loaded and no network is touched: traces are hand-built,
so ``analyze_failure_attribution`` is exercised purely as a reporting
derivation over persisted evidence.
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
    EvaluationResult,
    RetrievalEvent,
    RetrievalRole,
    RunTrace,
)
from commercebench.evaluation.deterministic import (
    FAILURE_EXACT_MATCH,
    FAILURE_FORBIDDEN_CLAIM,
    FAILURE_REQUIRED_FACT,
)
from commercebench.evaluation.retrieval import FAILURE_RETRIEVAL_MISS
from commercebench.reporting import (
    DiagnosticObservationKind,
    FailureAttribution,
    FailureAttributionReport,
    FailureEvidence,
    FailureStage,
    RetrievalMissAttribution,
    RetrievalMissOrigin,
    RerankerComparison,
    analyze_failure_attribution,
    derive_comparison_observations,
    render_failure_attribution_summary,
)
from commercebench.reporting.failure_attribution import (
    KNOWN_FORMAL_FAILURES,
)


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


def _evaluation(
    case: CaseSpec,
    trace: RunTrace,
    failures=(),
) -> EvaluationResult:
    failures = tuple(failures)
    return EvaluationResult(
        schema_version=SCHEMA_VERSION,
        evaluation_id="eval-0001",
        run_id=trace.run_id,
        case_id=case.case_id,
        evaluator_id="deterministic-rag-v0",
        evaluator_version="0.1",
        metrics={},
        passed=len(failures) == 0,
        failure_categories=failures,
        evaluated_at="2026-01-01T00:00:00+00:00",
        metadata={},
    )


def _trace(
    candidate_ids=None,
    final_ids=(),
    final_top_k=3,
    case_id="case-1",
    with_full_ranking=False,
    reranker_id="cross-encoder-msmarco-minilm-l6-v2-v0",
) -> RunTrace:
    """Build a trace. ``candidate_ids=None`` means legacy (no candidate)."""
    events = []
    if candidate_ids is not None:
        events.append(
            RetrievalEvent(
                query="q",
                document_ids=tuple(candidate_ids),
                scores=tuple(1.0 / (i + 1) for i in range(len(candidate_ids))),
                role=RetrievalRole.CANDIDATE,
            )
        )
        events.append(
            RetrievalEvent(
                query="q",
                document_ids=tuple(final_ids),
                scores=tuple(9.0 - i for i in range(len(final_ids))),
                role=RetrievalRole.FINAL,
            )
        )
    else:
        events.append(
            RetrievalEvent(
                query="q",
                document_ids=tuple(final_ids),
                scores=tuple(9.0 - i for i in range(len(final_ids))),
                role=None,
            )
        )
    if with_full_ranking:
        assert candidate_ids is not None
        # Full ranking keeps candidate order for simplicity; final must be
        # its prefix for Phase 2E construction.
        full_ids = tuple(candidate_ids)
        full_scores = tuple(
            9.0 - i for i in range(len(full_ids))
        )
        # Ensure final is the prefix of full.
        assert tuple(full_ids[: len(final_ids)]) == tuple(final_ids)
        runtime_metadata = {
            "retriever_id": "hybrid-bm25-minilm-rrf-v0",
            "retriever_version": "0.1",
            "corpus_id": "c",
            "corpus_version": "1",
            "top_k": final_top_k,
            "reranker": {
                "reranker_id": reranker_id,
                "reranker_version": "0.1",
                "candidate_top_k": len(candidate_ids),
                "final_top_k": final_top_k,
                "full_ranking": {
                    "document_ids": list(full_ids),
                    "scores": list(full_scores),
                },
            },
        }
    else:
        runtime_metadata = {"top_k": final_top_k}
    return RunTrace(
        schema_version=SCHEMA_VERSION,
        run_id="run-0001",
        experiment_id="exp",
        case_id=case_id,
        system_id="deterministic-rag",
        system_version="0.1",
        config_fingerprint="abc",
        input_text="q",
        output_text="answer",
        started_at="2026-01-01T00:00:00+00:00",
        finished_at="2026-01-01T00:00:00.010000+00:00",
        latency_ms=10.0,
        retrieval_events=tuple(events),
        runtime_metadata=runtime_metadata,
    )


class TestStateTruthTable:
    def test_state_a_retrieval_originated(self):
        # candidate window [c1,c2,c3] misses R; final [c1,c2,c3] misses R.
        case = _case(relevant=("R",))
        trace = _trace(
            candidate_ids=("c1", "c2", "c3", "c4"),
            final_ids=("c1", "c2", "c3"),
            final_top_k=3,
        )
        evaluation = _evaluation(case, trace, (FAILURE_RETRIEVAL_MISS,))
        report = analyze_failure_attribution(case, trace, evaluation)
        assert report.passed is False
        assert report.formal_failures == (FAILURE_RETRIEVAL_MISS,)
        detail = report.retrieval_miss_attribution
        assert detail is not None
        assert detail.origin is RetrievalMissOrigin.RETRIEVAL_ORIGINATED
        assert detail.stage is FailureStage.RETRIEVAL
        assert detail.candidate_hit is False
        assert detail.final_hit is False
        assert detail.evidence_complete is True
        assert detail.candidate_window_document_ids == ("c1", "c2", "c3")
        assert detail.final_document_ids == ("c1", "c2", "c3")
        assert detail.candidate_best_relevant_rank is None
        assert detail.final_best_relevant_rank is None
        assert detail.final_top_k == 3
        # Multi-label entry mirrors the detailed attribution.
        assert len(report.attributions) == 1
        assert report.attributions[0].formal_failure == FAILURE_RETRIEVAL_MISS
        assert report.attributions[0].stage is FailureStage.RETRIEVAL
        assert (
            report.attributions[0].origin
            is RetrievalMissOrigin.RETRIEVAL_ORIGINATED
        )

    def test_state_b_reranker_induced(self):
        # candidate window hits R at rank 2; final drops it.
        case = _case(relevant=("R",))
        trace = _trace(
            candidate_ids=("c1", "R", "c3", "c4"),
            final_ids=("c1", "c3", "c4"),
            final_top_k=3,
        )
        evaluation = _evaluation(case, trace, (FAILURE_RETRIEVAL_MISS,))
        report = analyze_failure_attribution(case, trace, evaluation)
        detail = report.retrieval_miss_attribution
        assert detail is not None
        assert detail.origin is RetrievalMissOrigin.RERANKER_INDUCED
        assert detail.stage is FailureStage.RERANKER
        assert detail.candidate_hit is True
        assert detail.final_hit is False
        assert detail.candidate_relevant_document_ids == ("R",)
        assert detail.final_relevant_document_ids == ()
        assert detail.candidate_best_relevant_rank == 2
        assert detail.final_best_relevant_rank is None
        assert detail.evidence_complete is True
        assert (
            DiagnosticObservationKind.RELEVANT_EXITED_FINAL_K.value
            in report.diagnostic_observations
        )

    def test_state_c_rescue_is_observation_only(self):
        # candidate window misses R; final hits R; no formal miss.
        case = _case(relevant=("R",))
        trace = _trace(
            candidate_ids=("c1", "c2", "c3", "c4"),
            final_ids=("c1", "R", "c3"),
            final_top_k=3,
        )
        evaluation = _evaluation(case, trace, ())
        report = analyze_failure_attribution(case, trace, evaluation)
        assert report.passed is True
        assert report.retrieval_miss_attribution is None
        assert report.attributions == ()
        assert (
            DiagnosticObservationKind.RERANKER_RESCUED_RELEVANT.value
            in report.diagnostic_observations
        )
        assert (
            DiagnosticObservationKind.RELEVANT_ENTERED_FINAL_K.value
            in report.diagnostic_observations
        )

    def test_state_d_no_failure(self):
        case = _case(relevant=("R",))
        trace = _trace(
            candidate_ids=("c1", "R", "c3", "c4"),
            final_ids=("c1", "R", "c3"),
            final_top_k=3,
        )
        evaluation = _evaluation(case, trace, ())
        report = analyze_failure_attribution(case, trace, evaluation)
        assert report.retrieval_miss_attribution is None
        assert (
            DiagnosticObservationKind.RERANKER_RESCUED_RELEVANT.value
            not in report.diagnostic_observations
        )


class TestWindowRule:
    def test_depth_trap_never_reranker_induced(self):
        # Relevant at full candidate rank 7, outside the top-3 window.
        # Same-depth window misses -> retrieval-originated, never induced.
        case = _case(relevant=("R",))
        candidate = ("c1", "c2", "c3", "c4", "c5", "c6", "R", "c8")
        trace = _trace(
            candidate_ids=candidate,
            final_ids=("c1", "c2", "c3"),
            final_top_k=3,
        )
        evaluation = _evaluation(case, trace, (FAILURE_RETRIEVAL_MISS,))
        report = analyze_failure_attribution(case, trace, evaluation)
        detail = report.retrieval_miss_attribution
        assert detail is not None
        assert detail.candidate_window_document_ids == ("c1", "c2", "c3")
        assert detail.candidate_hit is False
        assert detail.origin is RetrievalMissOrigin.RETRIEVAL_ORIGINATED
        assert detail.stage is FailureStage.RETRIEVAL

    def test_window_uses_final_top_k_not_full_candidate(self):
        # Relevant at candidate rank 2 but final_top_k=1 excludes it from
        # the window: window [c1] misses, final [c1] misses.
        case = _case(relevant=("R",))
        trace = _trace(
            candidate_ids=("c1", "R", "c3"),
            final_ids=("c1",),
            final_top_k=1,
        )
        evaluation = _evaluation(case, trace, (FAILURE_RETRIEVAL_MISS,))
        report = analyze_failure_attribution(case, trace, evaluation)
        assert report.retrieval_miss_attribution is not None
        assert (
            report.retrieval_miss_attribution.origin
            is RetrievalMissOrigin.RETRIEVAL_ORIGINATED
        )


class TestMultipleRelevant:
    def test_substitution_is_not_a_miss(self):
        # relevant {A,B}: window has A, final has B -> both hit.
        case = _case(relevant=("A", "B"))
        trace = _trace(
            candidate_ids=("A", "X", "Y", "Z"),
            final_ids=("B", "X", "Y"),
            final_top_k=3,
        )
        evaluation = _evaluation(case, trace, ())
        report = analyze_failure_attribution(case, trace, evaluation)
        assert report.retrieval_miss_attribution is None
        assert report.attributions == ()
        # A exited but B entered: any-hit semantics keep both hits True.
        detail_window_hit = {"A"} & {"A", "X", "Y"}
        detail_final_hit = {"A", "B"} & {"B", "X", "Y"}
        assert bool(detail_window_hit) is True
        assert bool(detail_final_hit) is True

    def test_any_hit_requires_intersection(self):
        case = _case(relevant=("A", "B"))
        trace = _trace(
            candidate_ids=("A", "X", "Y"),
            final_ids=("X", "Y", "Z"),
            final_top_k=3,
        )
        evaluation = _evaluation(case, trace, (FAILURE_RETRIEVAL_MISS,))
        report = analyze_failure_attribution(case, trace, evaluation)
        # Window hits A, final misses both -> reranker-induced.
        assert (
            report.retrieval_miss_attribution.origin
            is RetrievalMissOrigin.RERANKER_INDUCED
        )


class TestRankObservations:
    def test_rank_down_inside_window_is_observation_only(self):
        # R rank 2 -> 3 within final_top_k=3: regressed but no failure.
        case = _case(relevant=("R",))
        trace = _trace(
            candidate_ids=("c1", "R", "c3", "c4"),
            final_ids=("c1", "c3", "R"),
            final_top_k=3,
        )
        evaluation = _evaluation(case, trace, ())
        report = analyze_failure_attribution(case, trace, evaluation)
        assert report.retrieval_miss_attribution is None
        assert (
            DiagnosticObservationKind.RELEVANT_RANK_REGRESSED.value
            in report.diagnostic_observations
        )
        # Rank movement is never a formal failure.
        assert report.passed is True
        assert report.formal_failures == ()

    def test_rank_up_inside_window_is_observation_only(self):
        case = _case(relevant=("R",))
        trace = _trace(
            candidate_ids=("c1", "c2", "R", "c4"),
            final_ids=("R", "c1", "c2"),
            final_top_k=3,
        )
        evaluation = _evaluation(case, trace, ())
        report = analyze_failure_attribution(case, trace, evaluation)
        assert (
            DiagnosticObservationKind.RELEVANT_RANK_IMPROVED.value
            in report.diagnostic_observations
        )
        assert report.retrieval_miss_attribution is None


class TestLegacyAndNoRelevance:
    def test_legacy_unknown_attribution(self):
        case = _case(relevant=("R",))
        trace = _trace(
            candidate_ids=None,
            final_ids=("c1", "c2", "c3"),
            final_top_k=3,
        )
        evaluation = _evaluation(case, trace, (FAILURE_RETRIEVAL_MISS,))
        report = analyze_failure_attribution(case, trace, evaluation)
        detail = report.retrieval_miss_attribution
        assert detail is not None
        assert detail.origin is RetrievalMissOrigin.UNKNOWN
        assert detail.stage is FailureStage.UNKNOWN
        assert detail.evidence_complete is False
        assert detail.candidate_hit is None
        assert detail.candidate_window_document_ids == ()
        assert detail.final_hit is False
        assert detail.final_document_ids == ("c1", "c2", "c3")
        assert report.diagnostic_observations == ()

    def test_legacy_without_miss_has_no_attribution(self):
        case = _case(relevant=("R",))
        trace = _trace(
            candidate_ids=None, final_ids=("R", "c2"), final_top_k=2
        )
        evaluation = _evaluation(case, trace, ())
        report = analyze_failure_attribution(case, trace, evaluation)
        assert report.retrieval_miss_attribution is None

    def test_no_relevant_docs_has_no_attribution(self):
        case = _case(relevant=())
        trace = _trace(
            candidate_ids=("c1", "c2"),
            final_ids=("c1", "c2"),
            final_top_k=2,
        )
        evaluation = _evaluation(case, trace, ())
        report = analyze_failure_attribution(case, trace, evaluation)
        assert report.retrieval_miss_attribution is None
        assert report.passed is True

    def test_no_relevance_formal_miss_rejected(self):
        case = _case(relevant=())
        trace = _trace(
            candidate_ids=("c1", "c2"),
            final_ids=("c1",),
            final_top_k=1,
        )
        evaluation = _evaluation(case, trace, (FAILURE_RETRIEVAL_MISS,))
        with pytest.raises(ContractValidationError):
            analyze_failure_attribution(case, trace, evaluation)


class TestFailClosed:
    def test_formal_miss_with_final_hit_rejected(self):
        case = _case(relevant=("R",))
        trace = _trace(
            candidate_ids=("c1", "c2"),
            final_ids=("R", "c1"),
            final_top_k=2,
        )
        evaluation = _evaluation(case, trace, (FAILURE_RETRIEVAL_MISS,))
        with pytest.raises(
            ContractValidationError, match="contradicts final"
        ):
            analyze_failure_attribution(case, trace, evaluation)

    def test_legacy_contradiction_rejected(self):
        case = _case(relevant=("R",))
        trace = _trace(candidate_ids=None, final_ids=("R",), final_top_k=1)
        evaluation = _evaluation(case, trace, (FAILURE_RETRIEVAL_MISS,))
        with pytest.raises(ContractValidationError):
            analyze_failure_attribution(case, trace, evaluation)

    def test_passed_invariant_rejected(self):
        case = _case(relevant=("R",))
        trace = _trace(
            candidate_ids=("c1",), final_ids=("c1",), final_top_k=1
        )
        evaluation = _evaluation(case, trace, ())
        broken = dataclasses.replace(evaluation, passed=False)
        with pytest.raises(
            ContractValidationError, match="passed invariant"
        ):
            analyze_failure_attribution(case, trace, broken)

    def test_unknown_formal_failure_rejected(self):
        case = _case()
        trace = _trace(
            candidate_ids=("c1",), final_ids=("c1",), final_top_k=1
        )
        evaluation = _evaluation(case, trace, ("SOME_NEW_FAILURE",))
        # Manually force the invariant to hold so the unknown-string check
        # is what fires: passed=False with one failure.
        assert evaluation.passed is False
        with pytest.raises(
            ContractValidationError, match="unknown formal failure"
        ):
            analyze_failure_attribution(case, trace, evaluation)

    def test_identity_mismatch_rejected(self):
        case = _case()
        trace = _trace(
            candidate_ids=("c1",), final_ids=("c1",), final_top_k=1
        )
        other = dataclasses.replace(case, case_id="other")
        evaluation = _evaluation(case, trace, ())
        with pytest.raises(ContractValidationError, match="case_id"):
            analyze_failure_attribution(other, trace, evaluation)

    def test_run_mismatch_rejected(self):
        case = _case()
        trace = _trace(
            candidate_ids=("c1",), final_ids=("c1",), final_top_k=1
        )
        evaluation = _evaluation(case, trace, ())
        other_trace = dataclasses.replace(trace, run_id="run-9999")
        with pytest.raises(ContractValidationError, match="run_id"):
            analyze_failure_attribution(case, other_trace, evaluation)

    def test_duplicate_candidate_rejected(self):
        import copy

        case = _case(relevant=("R",))
        base = _trace(
            candidate_ids=("c1", "c2"),
            final_ids=("c1",),
            final_top_k=1,
        )
        extra = RetrievalEvent(
            query="q",
            document_ids=("c9",),
            scores=(0.1,),
            role=RetrievalRole.CANDIDATE,
        )
        doubled = dataclasses.replace(
            base, retrieval_events=base.retrieval_events + (extra,)
        )
        evaluation = _evaluation(case, doubled, (FAILURE_RETRIEVAL_MISS,))
        with pytest.raises(ContractValidationError, match="at most one"):
            analyze_failure_attribution(case, doubled, evaluation)


class TestMultiLabel:
    def test_cross_stage_multi_label_preserved(self):
        case = _case(relevant=("R",))
        trace = _trace(
            candidate_ids=("c1", "R", "c3"),
            final_ids=("c1", "c2", "c3"),
            final_top_k=3,
        )
        failures = (
            FAILURE_RETRIEVAL_MISS,
            FAILURE_REQUIRED_FACT,
            FAILURE_FORBIDDEN_CLAIM,
            FAILURE_EXACT_MATCH,
        )
        evaluation = _evaluation(case, trace, failures)
        report = analyze_failure_attribution(case, trace, evaluation)
        assert report.formal_failures == failures
        assert len(report.attributions) == 4
        by_failure = {a.formal_failure: a for a in report.attributions}
        assert by_failure[FAILURE_RETRIEVAL_MISS].stage is FailureStage.RERANKER
        assert (
            by_failure[FAILURE_RETRIEVAL_MISS].origin
            is RetrievalMissOrigin.RERANKER_INDUCED
        )
        for generation_failure in (
            FAILURE_REQUIRED_FACT,
            FAILURE_FORBIDDEN_CLAIM,
            FAILURE_EXACT_MATCH,
        ):
            assert by_failure[generation_failure].stage is (
                FailureStage.GENERATION
            )
            assert by_failure[generation_failure].origin is None
        # No single root cause is imposed.
        assert len({a.formal_failure for a in report.attributions}) == 4

    def test_generation_only_report(self):
        case = _case(relevant=("R",))
        trace = _trace(
            candidate_ids=("R", "c1", "c2"),
            final_ids=("R", "c1", "c2"),
            final_top_k=3,
        )
        evaluation = _evaluation(
            case, trace, (FAILURE_REQUIRED_FACT, FAILURE_FORBIDDEN_CLAIM)
        )
        report = analyze_failure_attribution(case, trace, evaluation)
        assert report.retrieval_miss_attribution is None
        assert [a.stage for a in report.attributions] == [
            FailureStage.GENERATION,
            FailureStage.GENERATION,
        ]


class TestIdentityControl:
    def test_identity_window_equals_final_never_induced(self):
        # Identity: candidate window == final ranking exactly.
        case = _case(relevant=("R",))
        trace = _trace(
            candidate_ids=("c1", "c2", "c3"),
            final_ids=("c1", "c2", "c3"),
            final_top_k=3,
        )
        # Miss on both sides -> retrieval-originated, never induced.
        evaluation = _evaluation(case, trace, (FAILURE_RETRIEVAL_MISS,))
        report = analyze_failure_attribution(case, trace, evaluation)
        assert (
            report.retrieval_miss_attribution.origin
            is RetrievalMissOrigin.RETRIEVAL_ORIGINATED
        )
        # Hit on both sides -> no attribution at all.
        hit_case = _case(relevant=("c1",))
        hit_eval = _evaluation(hit_case, trace, ())
        hit_report = analyze_failure_attribution(hit_case, trace, hit_eval)
        assert hit_report.retrieval_miss_attribution is None


class TestComparisonObservations:
    def _comparison(self, **overrides):
        from commercebench.reporting import RerankerComparison as _RC

        params = dict(
            report_version="0.1",
            case_id="case-1",
            final_top_k=2,
            baseline_reranker_id="identity-reranker-v0",
            baseline_reranker_version="0.1",
            treatment_reranker_id="cross-encoder-msmarco-minilm-l6-v2-v0",
            treatment_reranker_version="0.1",
            baseline_final_top1="d1",
            treatment_final_top1="d1",
            top1_changed=False,
            baseline_mrr=1.0,
            treatment_mrr=1.0,
            mrr_delta=0.0,
            baseline_recall_at_k=1.0,
            treatment_recall_at_k=1.0,
            recall_delta=0.0,
            treatment_relevant_rank_improved=False,
            treatment_relevant_rank_regressed=False,
            generation_changed=False,
        )
        params.update(overrides)
        return _RC(**params)

    def test_top1_and_generation_mapped(self):
        comparison = self._comparison(
            top1_changed=True, generation_changed=True
        )
        observations = derive_comparison_observations(comparison)
        assert (
            DiagnosticObservationKind.TOP1_CHANGED.value in observations
        )
        assert (
            DiagnosticObservationKind.GENERATION_CHANGED.value
            in observations
        )

    def test_no_changes_maps_to_empty(self):
        observations = derive_comparison_observations(self._comparison())
        assert observations == ()

    def test_rank_movement_mapped(self):
        comparison = self._comparison(treatment_relevant_rank_improved=True)
        assert DiagnosticObservationKind.RELEVANT_RANK_IMPROVED.value in (
            derive_comparison_observations(comparison)
        )


class TestSerialization:
    def test_json_roundtrip_state_b(self):
        case = _case(relevant=("R",))
        trace = _trace(
            candidate_ids=("c1", "R", "c3"),
            final_ids=("c1", "c2", "c3"),
            final_top_k=3,
        )
        evaluation = _evaluation(case, trace, (FAILURE_RETRIEVAL_MISS,))
        report = analyze_failure_attribution(case, trace, evaluation)
        payload = json.dumps(report.to_dict(), sort_keys=True)
        restored = FailureAttributionReport.from_dict(json.loads(payload))
        assert restored == report
        assert json.dumps(restored.to_dict(), sort_keys=True) == payload

    def test_json_roundtrip_legacy(self):
        case = _case(relevant=("R",))
        trace = _trace(candidate_ids=None, final_ids=("c1",), final_top_k=1)
        evaluation = _evaluation(case, trace, (FAILURE_RETRIEVAL_MISS,))
        report = analyze_failure_attribution(case, trace, evaluation)
        assert (
            FailureAttributionReport.from_dict(
                json.loads(json.dumps(report.to_dict()))
            )
            == report
        )

    def test_json_roundtrip_multi_label(self):
        case = _case(relevant=("R",))
        trace = _trace(
            candidate_ids=("c1", "R", "c3"),
            final_ids=("c1", "c2", "c3"),
            final_top_k=3,
        )
        evaluation = _evaluation(
            case,
            trace,
            (FAILURE_RETRIEVAL_MISS, FAILURE_REQUIRED_FACT),
        )
        report = analyze_failure_attribution(case, trace, evaluation)
        assert (
            FailureAttributionReport.from_dict(
                json.loads(
                    json.dumps(report.to_dict(), sort_keys=True),
                    parse_int=int,
                )
            )
            == report
        )

    def test_evidence_view(self):
        case = _case(relevant=("R",))
        trace = _trace(
            candidate_ids=("c1", "R", "c3"),
            final_ids=("c1", "c2", "c3"),
            final_top_k=3,
        )
        evaluation = _evaluation(case, trace, (FAILURE_RETRIEVAL_MISS,))
        report = analyze_failure_attribution(case, trace, evaluation)
        assert report.retrieval_miss_attribution is not None
        evidence = report.retrieval_miss_attribution.to_evidence()
        assert isinstance(evidence, FailureEvidence)
        assert evidence.candidate_window_document_ids == ("c1", "R", "c3")
        assert evidence.final_document_ids == ("c1", "c2", "c3")
        assert evidence.candidate_best_relevant_rank == 2
        assert evidence.final_best_relevant_rank is None
        assert (
            FailureEvidence.from_dict(
                json.loads(json.dumps(evidence.to_dict()))
            )
            == evidence
        )

    def test_render_summary_deterministic(self):
        case = _case(relevant=("R",))
        trace = _trace(
            candidate_ids=("c1", "R", "c3"),
            final_ids=("c1", "c2", "c3"),
            final_top_k=3,
        )
        evaluation = _evaluation(case, trace, (FAILURE_RETRIEVAL_MISS,))
        report = analyze_failure_attribution(case, trace, evaluation)
        first = render_failure_attribution_summary(report)
        second = render_failure_attribution_summary(
            FailureAttributionReport.from_dict(
                json.loads(json.dumps(report.to_dict()))
            )
        )
        assert first == second
        assert "RETRIEVAL_MISS" in first
        assert "reranker_induced" in first


class TestCoreInvariants:
    def test_known_formal_failures_unchanged(self):
        assert set(KNOWN_FORMAL_FAILURES) == {
            FAILURE_EXACT_MATCH,
            FAILURE_REQUIRED_FACT,
            FAILURE_FORBIDDEN_CLAIM,
            FAILURE_RETRIEVAL_MISS,
        }

    def test_reporting_types_are_not_core_failures(self):
        # Reporting stage/origin/observation strings must never collide
        # with a formal failure category.
        for value in (
            [item.value for item in FailureStage]
            + [item.value for item in RetrievalMissOrigin]
            + [item.value for item in DiagnosticObservationKind]
        ):
            assert value not in KNOWN_FORMAL_FAILURES
        for forbidden in (
            "RERANKER_FAILURE",
            "RERANKER_REGRESSION",
            "RERANKER_FINAL_K_REGRESSION",
        ):
            assert forbidden not in KNOWN_FORMAL_FAILURES

    def test_no_model_calls(self, monkeypatch):
        import commercebench.rag.retrievers as retrievers
        import commercebench.reranking.cross_encoder as cross_encoder
        import commercebench.reranking.rerankers as rerankers

        def _boom(*args, **kwargs):
            raise AssertionError("reporting must not touch model runtimes")

        monkeypatch.setattr(retrievers, "build_retriever", _boom)
        monkeypatch.setattr(rerankers, "build_reranker", _boom)
        monkeypatch.setattr(
            cross_encoder.CrossEncoderReranker, "_load_model", _boom
        )
        case = _case(relevant=("R",))
        trace = _trace(
            candidate_ids=("c1", "R", "c3"),
            final_ids=("c1", "c2", "c3"),
            final_top_k=3,
        )
        evaluation = _evaluation(case, trace, (FAILURE_RETRIEVAL_MISS,))
        report = analyze_failure_attribution(case, trace, evaluation)
        assert report.retrieval_miss_attribution is not None

    def test_cutoff_reused_not_thresholded(self):
        # With full Phase 2D diagnostics, cutoff_margin is reused as
        # numeric evidence and never produces a threshold observation.
        case = _case(relevant=("R",))
        trace = _trace(
            candidate_ids=("c1", "c2", "R"),
            final_ids=("c1", "c2"),
            final_top_k=2,
            with_full_ranking=True,
        )
        evaluation = _evaluation(case, trace, ())
        report = analyze_failure_attribution(case, trace, evaluation)
        assert report.cutoff_margin is not None
        assert report.retrieval_miss_attribution is None
        for observation in report.diagnostic_observations:
            assert "MARGIN" not in observation
            assert "UNCERTAIN" not in observation
