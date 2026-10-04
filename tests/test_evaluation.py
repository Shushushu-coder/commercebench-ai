"""Tests for the deterministic evaluator metrics and pass gate."""

from __future__ import annotations

import dataclasses

import pytest

from commercebench.contracts import EvaluationResult
from commercebench.evaluation.deterministic import (
    FAILURE_EXACT_MATCH,
    FAILURE_FORBIDDEN_CLAIM,
    FAILURE_REQUIRED_FACT,
    DeterministicEvaluator,
)


@pytest.fixture
def evaluator() -> DeterministicEvaluator:
    return DeterministicEvaluator()


def _evaluate(evaluator, case, trace, output_text):
    trace = dataclasses.replace(trace, output_text=output_text)
    return evaluator.evaluate(case, trace)


class TestDeterministicEvaluator:
    def test_pass(self, evaluator, case, trace):
        result = evaluator.evaluate(case, trace)
        assert result.passed is True
        assert result.failure_categories == ()
        assert result.metrics["exact_match"].value is True
        assert result.metrics["required_fact_coverage"].value == 1.0
        assert result.metrics["forbidden_claim_violation"].value is False

    def test_exact_match_fail(self, evaluator, case, trace):
        result = _evaluate(
            evaluator, case, trace, "签收后30天内可以退货。"
        )
        assert result.passed is False
        assert result.metrics["exact_match"].value is False
        assert FAILURE_EXACT_MATCH in result.failure_categories

    def test_missing_required_fact(self, evaluator, case, trace):
        case = dataclasses.replace(
            case,
            required_facts=("30天", "运费"),
            expected_response=None,
        )
        result = _evaluate(evaluator, case, trace, "签收后30天内可退。")
        assert result.metrics["required_fact_coverage"].value == 0.5
        assert result.metrics["required_fact_coverage"].details[
            "missing_facts"
        ] == ["运费"]
        assert FAILURE_REQUIRED_FACT in result.failure_categories
        assert result.passed is False

    def test_fact_matching_is_normalized(self, evaluator, case, trace):
        case = dataclasses.replace(
            case,
            required_facts=("  ABC  Def ",),
            expected_response=None,
        )
        result = _evaluate(evaluator, case, trace, "xxx abc   def yyy")
        assert result.metrics["required_fact_coverage"].value == 1.0

    def test_forbidden_claim(self, evaluator, case, trace):
        result = _evaluate(
            evaluator, case, trace, "商品签收后90天内可以申请退货。"
        )
        assert result.metrics["forbidden_claim_violation"].value is True
        assert result.metrics["forbidden_claim_violation"].details[
            "violated_claims"
        ] == ["90天"]
        assert FAILURE_FORBIDDEN_CLAIM in result.failure_categories
        assert result.passed is False

    def test_no_required_facts_coverage_is_one(
        self, evaluator, case, trace
    ):
        case = dataclasses.replace(case, required_facts=())
        result = evaluator.evaluate(case, trace)
        assert result.metrics["required_fact_coverage"].value == 1.0
        assert result.passed is True

    def test_no_expected_response_skips_exact_match(
        self, evaluator, case, trace
    ):
        case = dataclasses.replace(case, expected_response=None)
        result = _evaluate(evaluator, case, trace, "30天内支持退货。")
        assert "exact_match" not in result.metrics
        assert result.passed is True

    def test_all_failures_reported(self, evaluator, case, trace):
        result = _evaluate(evaluator, case, trace, "90天。")
        assert set(result.failure_categories) == {
            FAILURE_EXACT_MATCH,
            FAILURE_REQUIRED_FACT,
            FAILURE_FORBIDDEN_CLAIM,
        }
        assert result.passed is False

    def test_result_links_trace_and_case(self, evaluator, case, trace):
        result = evaluator.evaluate(case, trace)
        assert result.run_id == trace.run_id
        assert result.case_id == case.case_id
        assert result.evaluator_id == evaluator.evaluator_id
        assert result.evaluator_version == evaluator.evaluator_version

    def test_result_round_trip(self, evaluator, case, trace):
        result = evaluator.evaluate(case, trace)
        assert EvaluationResult.from_json(result.to_json()) == result

    def test_same_input_same_metrics(self, evaluator, case, trace):
        first = evaluator.evaluate(case, trace)
        second = evaluator.evaluate(case, trace)
        assert first.passed == second.passed
        assert first.metrics == second.metrics
        assert first.failure_categories == second.failure_categories
