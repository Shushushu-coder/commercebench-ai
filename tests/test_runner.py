"""Tests for the minimal experiment runner."""

from __future__ import annotations

import dataclasses
from datetime import datetime

import pytest

from commercebench.contracts import ContractValidationError
from commercebench.runner import run_case
from commercebench.systems import DeterministicSystem

RESPONSE = "商品签收后30天内可以申请退货。"


@pytest.fixture
def system() -> DeterministicSystem:
    return DeterministicSystem(
        responses={"商品签收后多久可以退货？": RESPONSE}
    )


class TestRunCase:
    def test_successful_run(self, case, manifest, system):
        trace = run_case(case, manifest, system)
        assert trace.output_text == RESPONSE
        assert trace.input_text == case.user_input
        assert trace.case_id == case.case_id
        assert trace.experiment_id == manifest.experiment_id
        assert trace.run_id

    def test_system_metadata_recorded(self, case, manifest, system):
        trace = run_case(case, manifest, system)
        assert trace.system_id == system.system_id
        assert trace.system_version == system.system_version

    def test_config_fingerprint_recorded(self, case, manifest, system):
        trace = run_case(case, manifest, system)
        assert trace.config_fingerprint == manifest.config_fingerprint()

    def test_timestamps_and_latency(self, case, manifest, system):
        trace = run_case(case, manifest, system)
        started = datetime.fromisoformat(trace.started_at)
        finished = datetime.fromisoformat(trace.finished_at)
        assert started <= finished
        assert trace.latency_ms >= 0.0
        expected_ms = (finished - started).total_seconds() * 1000.0
        assert trace.latency_ms == pytest.approx(expected_ms)

    def test_empty_event_lists(self, case, manifest, system):
        trace = run_case(case, manifest, system)
        assert trace.retrieval_events == ()
        assert trace.tool_events == ()
        assert trace.memory_events == ()
        assert trace.usage.input_tokens == 0

    def test_benchmark_version_mismatch_rejected(
        self, case, manifest, system
    ):
        other = dataclasses.replace(case, benchmark_version="other-0.0")
        with pytest.raises(ContractValidationError):
            run_case(other, manifest, system)

    def test_unlisted_case_rejected(self, case, manifest, system):
        other = dataclasses.replace(case, case_id="not-listed")
        with pytest.raises(ContractValidationError):
            run_case(other, manifest, system)

    def test_deterministic_output(self, case, manifest, system):
        first = run_case(case, manifest, system)
        second = run_case(case, manifest, system)
        assert first.output_text == second.output_text

    def test_unknown_input_uses_stable_fallback(self, case, manifest):
        system = DeterministicSystem(responses={})
        first = system.generate("未知问题")
        second = system.generate("未知问题")
        assert first.output_text == second.output_text
        assert first.output_text
