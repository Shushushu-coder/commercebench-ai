"""Contract validation and serialization round-trip tests."""

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
    ExperimentManifest,
    MemoryConfig,
    MemoryEvent,
    MetricResult,
    ModelConfig,
    RetrievalConfig,
    RetrievalEvent,
    RunTrace,
    ToolEvent,
    UsageStats,
)


def _case_kwargs(**overrides):
    base = dict(
        schema_version=SCHEMA_VERSION,
        case_id="case-001",
        benchmark_version="commercebench-dev-0.1",
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
    base.update(overrides)
    return base


def _manifest_kwargs(**overrides):
    base = dict(
        schema_version=SCHEMA_VERSION,
        experiment_id="exp-001",
        experiment_name="experiment",
        benchmark_version="commercebench-dev-0.1",
        case_ids=("case-001",),
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
    base.update(overrides)
    return base


class TestCaseSpecValidation:
    def test_valid_case(self, case):
        assert case.case_id == "product_return_policy_001"
        assert case.difficulty is Difficulty.EASY

    @pytest.mark.parametrize(
        "field_name",
        ["schema_version", "case_id", "benchmark_version", "user_input"],
    )
    def test_empty_required_string_rejected(self, field_name):
        with pytest.raises(ContractValidationError):
            CaseSpec(**_case_kwargs(**{field_name: ""}))

    @pytest.mark.parametrize(
        "field_name, value",
        [
            ("difficulty", "trivial"),
            ("conversation_type", "voice"),
            ("answerability", "unknown"),
        ],
    )
    def test_invalid_enum_rejected(self, field_name, value):
        with pytest.raises(ContractValidationError):
            CaseSpec(**_case_kwargs(**{field_name: value}))

    def test_enum_strings_coerced(self):
        case = CaseSpec(**_case_kwargs(difficulty="hard"))
        assert case.difficulty is Difficulty.HARD

    @pytest.mark.parametrize("bad_fact", ["", "   "])
    def test_empty_fact_rejected(self, bad_fact):
        with pytest.raises(ContractValidationError):
            CaseSpec(**_case_kwargs(required_facts=(bad_fact,)))
        with pytest.raises(ContractValidationError):
            CaseSpec(**_case_kwargs(forbidden_claims=(bad_fact,)))

    def test_non_serializable_metadata_rejected(self):
        with pytest.raises(ContractValidationError):
            CaseSpec(**_case_kwargs(metadata={"fn": object()}))

    def test_lists_normalized_to_tuples(self):
        case = CaseSpec(**_case_kwargs(required_facts=["30天"]))
        assert case.required_facts == ("30天",)
        assert isinstance(case.required_facts, tuple)


class TestCaseSpecSerialization:
    def test_round_trip(self, case):
        assert CaseSpec.from_dict(case.to_dict()) == case
        assert CaseSpec.from_json(case.to_json()) == case

    def test_json_is_deterministic_and_utf8(self, case):
        assert case.to_json() == case.to_json()
        json.loads(case.to_json().encode("utf-8"))

    def test_enums_serialize_as_strings(self, case):
        data = case.to_dict()
        assert data["difficulty"] == "easy"
        assert data["conversation_type"] == "single_turn"
        assert data["answerability"] == "answerable"

    def test_expected_response_none_round_trip(self):
        case = CaseSpec(**_case_kwargs(expected_response=None))
        restored = CaseSpec.from_json(case.to_json())
        assert restored.expected_response is None
        assert restored == case


class TestExperimentManifest:
    def test_round_trip(self, manifest):
        assert ExperimentManifest.from_dict(manifest.to_dict()) == manifest
        assert ExperimentManifest.from_json(manifest.to_json()) == manifest

    def test_round_trip_with_configs(self):
        manifest = ExperimentManifest(
            **_manifest_kwargs(
                model=ModelConfig(
                    provider="local",
                    model_name="fake-model",
                    parameters={"temperature": 0.0},
                ),
                retrieval=RetrievalConfig(
                    retriever_id="fake-retriever", parameters={"top_k": 3}
                ),
                memory=MemoryConfig(strategy="none", parameters={}),
            )
        )
        restored = ExperimentManifest.from_json(manifest.to_json())
        assert restored == manifest
        assert restored.model.provider == "local"

    def test_case_order_preserved(self):
        manifest = ExperimentManifest(
            **_manifest_kwargs(case_ids=["b", "a"])
        )
        assert manifest.case_ids == ("b", "a")

    def test_empty_experiment_id_rejected(self):
        with pytest.raises(ContractValidationError):
            ExperimentManifest(**_manifest_kwargs(experiment_id=""))

    def test_bool_random_seed_rejected(self):
        with pytest.raises(ContractValidationError):
            ExperimentManifest(**_manifest_kwargs(random_seed=True))


class TestConfigFingerprint:
    def test_same_config_same_fingerprint(self):
        assert (
            ExperimentManifest(**_manifest_kwargs()).config_fingerprint()
            == ExperimentManifest(**_manifest_kwargs()).config_fingerprint()
        )

    def test_dict_key_order_does_not_change_fingerprint(self):
        payload_a = _manifest_kwargs(
            model=ModelConfig(
                provider="p",
                model_name="m",
                parameters={"a": 1, "b": 2},
            )
        )
        payload_b = _manifest_kwargs(
            model=ModelConfig(
                provider="p",
                model_name="m",
                parameters={"b": 2, "a": 1},
            )
        )
        assert (
            ExperimentManifest(**payload_a).config_fingerprint()
            == ExperimentManifest(**payload_b).config_fingerprint()
        )

    def test_metadata_does_not_affect_fingerprint(self):
        base = ExperimentManifest(**_manifest_kwargs())
        described = ExperimentManifest(
            **_manifest_kwargs(
                experiment_id="other-id",
                experiment_name="renamed",
                metadata={"note": "descriptive only"},
            )
        )
        assert base.config_fingerprint() == described.config_fingerprint()

    @pytest.mark.parametrize(
        "overrides",
        [
            {"system_version": "0.2"},
            {"system_id": "other-system"},
            {"evaluator_version": "0.2"},
            {"benchmark_version": "commercebench-dev-0.2"},
            {"random_seed": 1},
            {"case_ids": ("case-001", "case-002")},
            {"model": ModelConfig(provider="p", model_name="m")},
        ],
    )
    def test_config_change_changes_fingerprint(self, overrides):
        base = ExperimentManifest(**_manifest_kwargs())
        changed = ExperimentManifest(**_manifest_kwargs(**overrides))
        assert base.config_fingerprint() != changed.config_fingerprint()

    def test_case_order_changes_fingerprint(self):
        first = ExperimentManifest(**_manifest_kwargs(case_ids=("a", "b")))
        second = ExperimentManifest(**_manifest_kwargs(case_ids=("b", "a")))
        assert first.config_fingerprint() != second.config_fingerprint()

    def test_fingerprint_is_sha256_hex(self, manifest):
        fingerprint = manifest.config_fingerprint()
        assert len(fingerprint) == 64
        int(fingerprint, 16)


class TestRunTrace:
    def test_round_trip(self, trace):
        assert RunTrace.from_dict(trace.to_dict()) == trace
        assert RunTrace.from_json(trace.to_json()) == trace

    def test_round_trip_with_events(self, trace):
        trace = dataclasses.replace(
            trace,
            retrieval_events=(
                RetrievalEvent(
                    query="退货",
                    document_ids=("doc-1", "doc-2"),
                    scores=(0.9, 0.5),
                ),
            ),
            tool_events=(
                ToolEvent(
                    tool_name="order_lookup",
                    arguments={"order_id": "A1"},
                    result={"status": "shipped"},
                ),
            ),
            memory_events=(
                MemoryEvent(strategy="recent", selected_item_ids=("m1",)),
            ),
        )
        restored = RunTrace.from_json(trace.to_json())
        assert restored == trace
        assert restored.retrieval_events[0].document_ids == ("doc-1", "doc-2")

    def test_invalid_timestamp_rejected(self, trace):
        with pytest.raises(ContractValidationError):
            dataclasses.replace(trace, started_at="not-a-time")

    def test_negative_latency_rejected(self, trace):
        with pytest.raises(ContractValidationError):
            dataclasses.replace(trace, latency_ms=-1.0)

    def test_event_score_length_mismatch_rejected(self):
        with pytest.raises(ContractValidationError):
            RetrievalEvent(
                query="q", document_ids=("d1",), scores=(0.1, 0.2)
            )


class TestEvaluationResult:
    def _result(self, trace) -> EvaluationResult:
        return EvaluationResult(
            schema_version=SCHEMA_VERSION,
            evaluation_id="eval-0001",
            run_id=trace.run_id,
            case_id=trace.case_id,
            evaluator_id="deterministic-v0",
            evaluator_version="0.1",
            metrics={
                "exact_match": MetricResult(
                    name="exact_match", value=True, passed=True, details={}
                )
            },
            passed=True,
            failure_categories=(),
            evaluated_at="2026-01-01T00:00:01+00:00",
            metadata={},
        )

    def test_round_trip(self, trace):
        result = self._result(trace)
        assert EvaluationResult.from_dict(result.to_dict()) == result
        assert EvaluationResult.from_json(result.to_json()) == result

    def test_metric_key_must_match_name(self, trace):
        with pytest.raises(ContractValidationError):
            EvaluationResult(
                schema_version=SCHEMA_VERSION,
                evaluation_id="e1",
                run_id=trace.run_id,
                case_id=trace.case_id,
                evaluator_id="deterministic-v0",
                evaluator_version="0.1",
                metrics={
                    "wrong_key": MetricResult(
                        name="exact_match", value=True, passed=True
                    )
                },
                passed=True,
                evaluated_at="2026-01-01T00:00:01+00:00",
            )

    def test_non_bool_passed_rejected(self, trace):
        with pytest.raises(ContractValidationError):
            EvaluationResult(
                schema_version=SCHEMA_VERSION,
                evaluation_id="e1",
                run_id=trace.run_id,
                case_id=trace.case_id,
                evaluator_id="deterministic-v0",
                evaluator_version="0.1",
                metrics={},
                passed="yes",
                evaluated_at="2026-01-01T00:00:01+00:00",
            )
