"""Phase 3B fingerprint regression: legacy stability + dialogue mutations."""

from __future__ import annotations

import pathlib

import pytest

from commercebench.contracts.errors import ContractValidationError
from commercebench.contracts.experiment import (
    DialogueHarnessConfig,
    ExperimentManifest,
)

RAG_EXPERIMENTS_DIR = (
    pathlib.Path(__file__).resolve().parents[1]
    / "examples"
    / "rag_v0"
    / "experiments"
)
PHASE0_EXPERIMENT = (
    pathlib.Path(__file__).resolve().parents[1]
    / "examples"
    / "phase0"
    / "experiment.json"
)

# Captured at 886ad959 (Phase 2G HEAD) before the dialogue field existed.
LEGACY_FINGERPRINTS = {
    "phase0/experiment.json": "bb0e21dba7908dee58be0c96550ad1e4ebfac57478519a0231b39b3be58cf26f",
    "rag_bm25_v0.json": "69241cf2d2238b3a243e7d0e2977b00f37ebbefd6c69fcff031563f0f9f9a0f6",
    "rag_dense_minilm_v0.json": "719511aa837aea8078109ff19882992071c6dc3b58562bfcd286dcc0072c634c",
    "rag_hybrid_cross_encoder_v0.json": "2e0a6d5ea0f3d8cebd5f4379bf9ed6a0e7d26dec992a3540ba64539b395407f7",
    "rag_hybrid_identity_rerank_v0.json": "6df0f8241251a5e614df7eaf70ada042b06352a11b6bb260f193ca55ccf816e8",
    "rag_hybrid_rrf_v0.json": "2fb52fe18bd9eaf220b2c4b4af0be32aec3453ea461b31b28bab7459367580ba",
    "rag_keyword_v0.json": "751cbc843b888252716a6f96ec9e0838b6da7ed0b82fc753298a30c74753a37a",
    "rag_term_frequency_v0.json": "a4ba4063571667fc8fd76ac6ad86669e1017e46124df3204f2606f1d52321ba3",
}

DIALOGUE_PARAMETERS = {
    "max_turns": 6,
    "max_tool_calls_per_turn": 3,
    "history_policy": "full",
    "tool_policy": "deterministic-commerce-v0",
    "tool_simulator": {"id": "commerce-tools-v0", "version": "0.1"},
}


def _dialogue_config(**param_overrides):
    parameters = dict(DIALOGUE_PARAMETERS)
    parameters.update(param_overrides)
    return DialogueHarnessConfig(
        harness_id="dialogue-harness-v0",
        harness_version="0.1",
        parameters=parameters,
    )


def _manifest(**overrides):
    payload = {
        "schema_version": "0.1",
        "experiment_id": "e1",
        "experiment_name": "e1",
        "benchmark_version": "commercebench-dev-0.1",
        "case_ids": ("c1",),
        "system_id": "s1",
        "system_version": "v1",
        "evaluator_id": "ev",
        "evaluator_version": "0.1",
        "random_seed": 0,
    }
    payload.update(overrides)
    return ExperimentManifest(**payload)


class TestLegacyFingerprintStability:
    @pytest.mark.parametrize(
        "name,expected", sorted(LEGACY_FINGERPRINTS.items())
    )
    def test_legacy_manifest_fingerprints_unchanged(self, name, expected):
        path = (
            PHASE0_EXPERIMENT if name.startswith("phase0")
            else RAG_EXPERIMENTS_DIR / name
        )
        manifest = ExperimentManifest.from_json(
            path.read_text(encoding="utf-8")
        )
        assert manifest.dialogue is None
        assert manifest.config_fingerprint() == expected

    def test_explicit_null_dialogue_unchanged(self):
        manifest = _manifest(dialogue=None)
        assert manifest.config_fingerprint() == _manifest().config_fingerprint()

    def test_dialogue_roundtrip(self):
        manifest = _manifest(dialogue=_dialogue_config())
        restored = ExperimentManifest.from_dict(manifest.to_dict())
        assert restored.dialogue == manifest.dialogue
        assert restored.config_fingerprint() == manifest.config_fingerprint()


class TestDialogueFingerprintMutations:
    def test_dialogue_presence_changes_fingerprint(self):
        assert _manifest(dialogue=_dialogue_config()).config_fingerprint() != (
            _manifest().config_fingerprint()
        )

    def test_harness_id_mutation(self):
        a = _manifest(dialogue=_dialogue_config())
        other = DialogueHarnessConfig(
            harness_id="dialogue-harness-v1",
            harness_version="0.1",
            parameters=dict(DIALOGUE_PARAMETERS),
        )
        b = _manifest(dialogue=other)
        assert a.config_fingerprint() != b.config_fingerprint()

    def test_harness_version_mutation(self):
        a = _manifest(dialogue=_dialogue_config())
        other = DialogueHarnessConfig(
            harness_id="dialogue-harness-v0",
            harness_version="0.2",
            parameters=dict(DIALOGUE_PARAMETERS),
        )
        b = _manifest(dialogue=other)
        assert a.config_fingerprint() != b.config_fingerprint()

    @pytest.mark.parametrize(
        "parameters",
        [
            {"max_turns": 7},
            {"max_tool_calls_per_turn": 5},
            {"history_policy": "full"},  # same value — no mutation
        ],
    )
    def test_parameter_mutations(self, parameters):
        a = _manifest(dialogue=_dialogue_config())
        merged = dict(DIALOGUE_PARAMETERS)
        merged.update(parameters)
        cfg = DialogueHarnessConfig(
            harness_id="dialogue-harness-v0",
            harness_version="0.1",
            parameters=merged,
        )
        b = _manifest(dialogue=cfg)
        same = parameters.get("history_policy") == "full" and len(parameters) == 1
        assert (a.config_fingerprint() == b.config_fingerprint()) == same

    def test_tool_policy_mutation(self):
        a = _manifest(dialogue=_dialogue_config())
        b = _manifest(
            dialogue=_dialogue_config(tool_policy="deterministic-commerce-v1")
        )
        assert a.config_fingerprint() != b.config_fingerprint()

    def test_tool_simulator_id_mutation(self):
        a = _manifest(dialogue=_dialogue_config())
        b = _manifest(
            dialogue=_dialogue_config(
                tool_simulator={"id": "commerce-tools-v1", "version": "0.1"}
            )
        )
        assert a.config_fingerprint() != b.config_fingerprint()

    def test_tool_simulator_version_mutation(self):
        a = _manifest(dialogue=_dialogue_config())
        b = _manifest(
            dialogue=_dialogue_config(
                tool_simulator={"id": "commerce-tools-v0", "version": "0.2"}
            )
        )
        assert a.config_fingerprint() != b.config_fingerprint()


class TestDialogueConfigValidation:
    def test_unknown_parameter_rejected(self):
        with pytest.raises(ContractValidationError):
            _dialogue_config(history_polcy="full")

    def test_missing_parameter_rejected(self):
        parameters = dict(DIALOGUE_PARAMETERS)
        del parameters["max_turns"]
        with pytest.raises(ContractValidationError):
            DialogueHarnessConfig(
                harness_id="h", harness_version="0.1", parameters=parameters
            )

    def test_nonpositive_max_turns_rejected(self):
        with pytest.raises(ContractValidationError):
            _dialogue_config(max_turns=0)

    def test_unknown_history_policy_rejected(self):
        with pytest.raises(ContractValidationError):
            _dialogue_config(history_policy="truncated")

    def test_tool_simulator_missing_id_rejected(self):
        with pytest.raises(ContractValidationError):
            _dialogue_config(tool_simulator={"version": "0.1"})

    def test_json_fixture_manifest_loads(self):
        path = (
            pathlib.Path(__file__).resolve().parents[1]
            / "examples"
            / "dialogue_v0"
            / "experiments"
            / "dialogue_stateful_v0.json"
        )
        manifest = ExperimentManifest.from_json(path.read_text(encoding="utf-8"))
        assert manifest.dialogue is not None
        assert manifest.dialogue.parameters["max_turns"] == 6
        assert manifest.config_fingerprint()
