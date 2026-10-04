"""Shared fixtures for Phase 0 kernel tests."""

from __future__ import annotations

import pytest

from commercebench.contracts import (
    SCHEMA_VERSION,
    Answerability,
    CaseSpec,
    ConversationType,
    Difficulty,
    ExperimentManifest,
    UsageStats,
)
from commercebench.contracts.trace import RunTrace

BENCHMARK_VERSION = "commercebench-dev-0.1"


@pytest.fixture
def case() -> CaseSpec:
    return CaseSpec(
        schema_version=SCHEMA_VERSION,
        case_id="product_return_policy_001",
        benchmark_version=BENCHMARK_VERSION,
        task_family="grounded_qa",
        intent="return",
        difficulty=Difficulty.EASY,
        conversation_type=ConversationType.SINGLE_TURN,
        answerability=Answerability.ANSWERABLE,
        user_input="商品签收后多久可以退货？",
        required_facts=("30天",),
        forbidden_claims=("90天",),
        expected_response="商品签收后30天内可以申请退货。",
        metadata={},
    )


@pytest.fixture
def manifest(case: CaseSpec) -> ExperimentManifest:
    return ExperimentManifest(
        schema_version=SCHEMA_VERSION,
        experiment_id="phase0_deterministic_smoke",
        experiment_name="Phase 0 deterministic smoke",
        benchmark_version=case.benchmark_version,
        case_ids=(case.case_id,),
        system_id="deterministic-baseline",
        system_version="0.1",
        model=None,
        retrieval=None,
        memory=None,
        evaluator_id="deterministic-v0",
        evaluator_version="0.1",
        random_seed=0,
        metadata={},
    )


@pytest.fixture
def trace(case: CaseSpec, manifest: ExperimentManifest) -> RunTrace:
    return RunTrace(
        schema_version=SCHEMA_VERSION,
        run_id="run-0001",
        experiment_id=manifest.experiment_id,
        case_id=case.case_id,
        system_id=manifest.system_id,
        system_version=manifest.system_version,
        config_fingerprint=manifest.config_fingerprint(),
        input_text=case.user_input,
        output_text="商品签收后30天内可以申请退货。",
        started_at="2026-01-01T00:00:00+00:00",
        finished_at="2026-01-01T00:00:00.010000+00:00",
        latency_ms=10.0,
        usage=UsageStats(input_tokens=0, output_tokens=0, total_tokens=0),
        runtime_metadata={},
    )
