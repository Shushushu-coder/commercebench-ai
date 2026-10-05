"""Phase 2A retrieval-role contract tests.

``RetrievalEvent.role`` separates diagnostic candidate rankings from the
evaluation-visible final ranking. Legacy events (no ``role``) keep final
semantics so Phase 1A-1G traces stay valid. No reranker exists yet; the
multi-event traces below are hand-built contract fixtures simulating the
shape a future reranker will produce.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from commercebench.contracts import (
    ContractValidationError,
    RetrievalEvent,
    RetrievalRole,
    RunTrace,
)
from commercebench.evaluation.retrieval import (
    FAILURE_RETRIEVAL_MISS,
    RetrievalEvaluator,
    ranked_document_ids,
)


def _event(document_ids, role=None, query="q"):
    return RetrievalEvent(
        query=query,
        document_ids=tuple(document_ids),
        scores=tuple(float(len(document_ids) - i) for i in range(len(document_ids))),
        role=role,
    )


def _trace_with_events(trace, events):
    return dataclasses.replace(trace, retrieval_events=tuple(events))


def _case_with_relevant(case, relevant):
    return dataclasses.replace(case, relevant_document_ids=tuple(relevant))


class TestRetrievalRoleContract:
    def test_default_role_is_none(self):
        assert RetrievalEvent(query="q").role is None

    def test_candidate_role_accepted(self):
        event = _event(("d1",), role="candidate")
        assert event.role is RetrievalRole.CANDIDATE
        event = _event(("d1",), role=RetrievalRole.CANDIDATE)
        assert event.role is RetrievalRole.CANDIDATE

    def test_final_role_accepted(self):
        event = _event(("d1",), role="final")
        assert event.role is RetrievalRole.FINAL
        event = _event(("d1",), role=RetrievalRole.FINAL)
        assert event.role is RetrievalRole.FINAL

    @pytest.mark.parametrize(
        "bad_role",
        ["initial", "reranked", "context", "FINAL", "foo", "", 1, True, 1.5],
    )
    def test_unknown_role_rejected(self, bad_role):
        with pytest.raises(ContractValidationError):
            _event(("d1",), role=bad_role)

    def test_role_serializes_as_string(self):
        assert _event(("d1",), role="candidate").to_dict()["role"] == "candidate"
        assert _event(("d1",), role="final").to_dict()["role"] == "final"

    def test_legacy_event_serializes_role_null(self):
        # Optional-field convention: absent role serializes as explicit null.
        assert _event(("d1",)).to_dict()["role"] is None

    def test_role_dict_roundtrip(self):
        for role in ("candidate", "final", None):
            event = _event(("d1", "d2"), role=role)
            assert RetrievalEvent.from_dict(event.to_dict()) == event

    def test_role_json_roundtrip_through_trace(self, trace):
        trace = _trace_with_events(
            trace,
            (
                _event(("d1",), role="candidate"),
                _event(("d2",), role="final"),
                _event(("d3",)),
            ),
        )
        restored = RunTrace.from_json(trace.to_json())
        assert restored.retrieval_events == trace.retrieval_events
        assert [e.role for e in restored.retrieval_events] == [
            RetrievalRole.CANDIDATE,
            RetrievalRole.FINAL,
            None,
        ]

    def test_legacy_json_without_role_loads_as_none(self):
        legacy = {
            "query": "q",
            "document_ids": ["d1"],
            "scores": [1.0],
        }
        event = RetrievalEvent.from_dict(legacy)
        assert event.role is None

    def test_legacy_trace_json_without_role_loads(self, trace):
        payload = json.loads(trace.to_json())
        payload["retrieval_events"] = [
            {"query": "q", "document_ids": ["d1"], "scores": [1.0]}
        ]
        restored = RunTrace.from_dict(payload)
        assert restored.retrieval_events[0].role is None
        assert restored.retrieval_events[0].document_ids == ("d1",)

    def test_unknown_role_in_json_rejected(self):
        with pytest.raises(ContractValidationError):
            RetrievalEvent.from_dict(
                {
                    "query": "q",
                    "document_ids": ["d1"],
                    "scores": [1.0],
                    "role": "reranked",
                }
            )

    def test_unknown_role_nested_in_trace_rejected(self, trace):
        payload = json.loads(trace.to_json())
        payload["retrieval_events"] = [
            {
                "query": "q",
                "document_ids": ["d1"],
                "scores": [1.0],
                "role": "initial",
            }
        ]
        with pytest.raises(ContractValidationError):
            RunTrace.from_dict(payload)


class TestFinalRetrievalSelection:
    def test_legacy_single_event_is_final(self, trace):
        trace = _trace_with_events(trace, (_event(("d1", "d2")),))
        assert trace.final_retrieval_events == trace.retrieval_events
        assert trace.final_ranked_document_ids == ("d1", "d2")
        assert ranked_document_ids(trace) == ("d1", "d2")

    def test_explicit_final_single_event(self, trace):
        trace = _trace_with_events(
            trace, (_event(("d1", "d2"), role="final"),)
        )
        assert trace.final_ranked_document_ids == ("d1", "d2")

    def test_candidate_event_excluded(self, trace):
        trace = _trace_with_events(
            trace, (_event(("d1", "d2"), role="candidate"),)
        )
        assert trace.final_retrieval_events == ()
        assert trace.final_ranked_document_ids == ()
        assert ranked_document_ids(trace) == ()

    def test_candidate_then_final(self, trace):
        trace = _trace_with_events(
            trace,
            (
                _event(("d2", "d1"), role="candidate"),
                _event(("d1", "d3"), role="final"),
            ),
        )
        assert trace.final_ranked_document_ids == ("d1", "d3")
        assert ranked_document_ids(trace) == ("d1", "d3")

    def test_multiple_candidates_only_final_counts(self, trace):
        base_final = _event(("d4",), role="final")
        trace = _trace_with_events(
            trace,
            (
                _event(("d1",), role="candidate"),
                _event(("d2", "d3"), role="candidate"),
                base_final,
            ),
        )
        assert trace.final_retrieval_events == (base_final,)
        assert trace.final_ranked_document_ids == ("d4",)

        # Reordering/changing candidates cannot affect the final ranking.
        trace_swapped = _trace_with_events(
            trace,
            (
                _event(("d9", "d8"), role="candidate"),
                _event(("d7",), role="candidate"),
                base_final,
            ),
        )
        assert trace_swapped.final_ranked_document_ids == ("d4",)

    def test_multiple_finals_concat_dedupe_keep_first(self, trace):
        trace = _trace_with_events(
            trace,
            (
                _event(("d1", "d2"), role="final"),
                _event(("d2", "d3"), role="final"),
            ),
        )
        assert trace.final_ranked_document_ids == ("d1", "d2", "d3")

    def test_mixed_final_and_legacy_events_concat(self, trace):
        trace = _trace_with_events(
            trace,
            (
                _event(("d1", "d2"), role="final"),
                _event(("d9",), role="candidate"),
                _event(("d2", "d3")),
            ),
        )
        assert trace.final_ranked_document_ids == ("d1", "d2", "d3")

    def test_no_retrieval_events(self, trace):
        assert trace.final_retrieval_events == ()
        assert trace.final_ranked_document_ids == ()


class TestMetricsUseFinalRanking:
    """Every metric resolves the ranking through one final-only helper."""

    def test_candidate_pollution_does_not_count(self, case, trace):
        # Relevant d2 appears only in the candidate ranking; the final
        # ranking misses it. Candidate hits must never count.
        case = _case_with_relevant(case, ("d2",))
        trace = _trace_with_events(
            trace,
            (
                _event(("d2", "d1"), role="candidate"),
                _event(("d1", "d3"), role="final"),
            ),
        )
        result = RetrievalEvaluator(ks=(1, 3)).evaluate(case, trace)
        assert ranked_document_ids(trace) == ("d1", "d3")
        for name, metric in result.metrics.items():
            assert metric.details["retrieved_document_ids"] == ["d1", "d3"], name
        assert result.metrics["retrieval_recall_at_1"].value == 0.0
        assert result.metrics["retrieval_recall_at_3"].value == 0.0
        assert result.metrics["retrieval_precision_at_1"].value == 0.0
        assert result.metrics["retrieval_precision_at_3"].value == 0.0
        assert result.metrics["retrieval_mrr"].value == 0.0
        assert result.metrics["retrieval_ndcg_at_3"].value == 0.0
        assert result.failure_categories == (FAILURE_RETRIEVAL_MISS,)
        assert result.passed is False

    def test_reranking_improvement_simulation(self, case, trace):
        # Hand-built trace of the shape a future reranker produces:
        # candidate rank 2 -> final rank 1. Metrics reflect the final.
        case = _case_with_relevant(case, ("d2",))
        trace = _trace_with_events(
            trace,
            (
                _event(("d1", "d2"), role="candidate"),
                _event(("d2", "d1"), role="final"),
            ),
        )
        result = RetrievalEvaluator(ks=(1,)).evaluate(case, trace)
        assert result.metrics["retrieval_recall_at_1"].value == 1.0
        assert result.metrics["retrieval_precision_at_1"].value == 1.0
        assert result.metrics["retrieval_ndcg_at_1"].value == 1.0
        assert result.metrics["retrieval_mrr"].value == 1.0
        assert result.failure_categories == ()
        assert result.passed is True

    def test_reranking_regression_simulation(self, case, trace):
        # Candidate rank 1 -> final rank 2: MRR is 0.5, not the
        # candidate's 1.0. Regressions are measured, not hidden.
        case = _case_with_relevant(case, ("d2",))
        trace = _trace_with_events(
            trace,
            (
                _event(("d2", "d1"), role="candidate"),
                _event(("d1", "d2"), role="final"),
            ),
        )
        result = RetrievalEvaluator(ks=(1,)).evaluate(case, trace)
        assert result.metrics["retrieval_mrr"].value == 0.5
        assert result.metrics["retrieval_recall_at_1"].value == 0.0

    def test_explicit_final_matches_legacy_metrics(self, case, trace):
        case = _case_with_relevant(case, ("d2", "d4"))
        ids = ("d1", "d2", "d3", "d4")
        legacy_trace = _trace_with_events(trace, (_event(ids),))
        final_trace = _trace_with_events(
            trace, (_event(ids, role="final"),)
        )
        evaluator = RetrievalEvaluator(ks=(1, 2))
        legacy_result = evaluator.evaluate(case, legacy_trace)
        final_result = evaluator.evaluate(case, final_trace)
        for name in legacy_result.metrics:
            assert (
                final_result.metrics[name].value
                == legacy_result.metrics[name].value
            ), name
        assert final_result.failure_categories == legacy_result.failure_categories
        assert final_result.passed == legacy_result.passed

    def test_candidate_only_trace_is_retrieval_miss(self, case, trace):
        # A trace whose only rankings are candidates has no
        # evaluation-visible ranking -> empty final ranking -> miss.
        case = _case_with_relevant(case, ("d2",))
        trace = _trace_with_events(
            trace,
            (
                _event(("d2", "d1"), role="candidate"),
                _event(("d3",), role="candidate"),
            ),
        )
        result = RetrievalEvaluator().evaluate(case, trace)
        assert ranked_document_ids(trace) == ()
        assert result.metrics["retrieval_recall_at_3"].value == 0.0
        assert result.metrics["retrieval_precision_at_3"].value == 0.0
        assert result.metrics["retrieval_mrr"].value == 0.0
        assert result.metrics["retrieval_ndcg_at_3"].value == 0.0
        assert result.failure_categories == (FAILURE_RETRIEVAL_MISS,)
        assert result.passed is False

    def test_candidate_only_no_relevant_stays_not_applicable(
        self, case, trace
    ):
        assert case.relevant_document_ids == ()
        trace = _trace_with_events(
            trace, (_event(("d1",), role="candidate"),)
        )
        result = RetrievalEvaluator().evaluate(case, trace)
        for name, metric in result.metrics.items():
            assert metric.details["not_applicable"] is True, name
        assert result.passed is True
        assert result.failure_categories == ()

    def test_candidate_order_does_not_change_metrics(self, case, trace):
        case = _case_with_relevant(case, ("d2",))
        final = _event(("d2", "d1"), role="final")
        results = []
        for candidates in (
            (_event(("d3",), role="candidate"),),
            (_event(("d9", "d8"), role="candidate"), _event(("d7",), role="candidate")),
        ):
            varied = _trace_with_events(trace, candidates + (final,))
            results.append(RetrievalEvaluator(ks=(1,)).evaluate(case, varied))
        for name in results[0].metrics:
            assert results[1].metrics[name].value == results[0].metrics[name].value
        assert results[0].metrics["retrieval_mrr"].value == 1.0

    def test_multiple_finals_metrics_use_concatenated_ranking(
        self, case, trace
    ):
        # Final A misses at its own depth; final B contributes the hit at
        # rank 3 of the concatenated final ranking [d1, d2, d3->wait d2 relevant].
        case = _case_with_relevant(case, ("d3",))
        trace = _trace_with_events(
            trace,
            (
                _event(("d1", "d2"), role="final"),
                _event(("d2", "d3"), role="final"),
            ),
        )
        result = RetrievalEvaluator(ks=(1, 3)).evaluate(case, trace)
        assert ranked_document_ids(trace) == ("d1", "d2", "d3")
        assert result.metrics["retrieval_recall_at_1"].value == 0.0
        assert result.metrics["retrieval_recall_at_3"].value == 1.0
        assert result.metrics["retrieval_mrr"].value == pytest.approx(1.0 / 3)
        assert result.passed is True


class TestRoleDoesNotAffectFingerprint:
    def test_config_fingerprint_unaffected_by_role(self, manifest, trace):
        # Role is runtime trace semantics, not system configuration: the
        # manifest fingerprint covers manifest fields only, so identical
        # runs differing only in event roles share one fingerprint.
        legacy_trace = _trace_with_events(trace, (_event(("d1",)),))
        staged_trace = _trace_with_events(
            trace,
            (
                _event(("d2",), role="candidate"),
                _event(("d1",), role="final"),
            ),
        )
        fingerprint = manifest.config_fingerprint()
        assert legacy_trace.config_fingerprint == fingerprint
        assert staged_trace.config_fingerprint == fingerprint
