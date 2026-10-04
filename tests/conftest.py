"""Shared fixtures for Phase 0 kernel tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Mapping

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
from commercebench.rag import Corpus

BENCHMARK_VERSION = "commercebench-dev-0.1"

RAG_V0_DIR = Path(__file__).resolve().parents[1] / "examples" / "rag_v0"


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


@pytest.fixture(scope="session")
def rag_v0_dir() -> Path:
    return RAG_V0_DIR


@pytest.fixture(scope="session")
def rag_corpus(rag_v0_dir: Path) -> Corpus:
    return Corpus.from_json(
        (rag_v0_dir / "corpus.json").read_text(encoding="utf-8")
    )


@pytest.fixture(scope="session")
def rag_answers(rag_v0_dir: Path) -> Mapping[str, str]:
    payload = json.loads(
        (rag_v0_dir / "answer_key.json").read_text(encoding="utf-8")
    )
    return payload["answers"]


@pytest.fixture(scope="session")
def rag_cases(rag_v0_dir: Path) -> Dict[str, CaseSpec]:
    cases = {}
    for path in sorted((rag_v0_dir / "cases").glob("*.json")):
        case = CaseSpec.from_json(path.read_text(encoding="utf-8"))
        cases[case.case_id] = case
    return cases


def _load_manifest(rag_v0_dir: Path, name: str) -> ExperimentManifest:
    return ExperimentManifest.from_json(
        (rag_v0_dir / "experiments" / name).read_text(encoding="utf-8")
    )


@pytest.fixture(scope="session")
def keyword_manifest(rag_v0_dir: Path) -> ExperimentManifest:
    return _load_manifest(rag_v0_dir, "rag_keyword_v0.json")


@pytest.fixture(scope="session")
def term_frequency_manifest(rag_v0_dir: Path) -> ExperimentManifest:
    return _load_manifest(rag_v0_dir, "rag_term_frequency_v0.json")


@pytest.fixture(scope="session")
def bm25_manifest(rag_v0_dir: Path) -> ExperimentManifest:
    return _load_manifest(rag_v0_dir, "rag_bm25_v0.json")


@pytest.fixture(scope="session")
def dense_manifest(rag_v0_dir: Path) -> ExperimentManifest:
    return _load_manifest(rag_v0_dir, "rag_dense_minilm_v0.json")
