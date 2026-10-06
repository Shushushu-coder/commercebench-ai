"""Phase 3D real-LLM dialogue identity contract tests.

``DialogueModelIdentity`` covers every provider-visible,
result-affecting dimension before the first real model call: model
identity/revision, prompt identity/hash, history and state renderers,
the exposed tool schema hash, and generation parameters. These tests
pin validation, fingerprint coverage, and the runtime binding seams —
no provider calls.
"""

from __future__ import annotations

import pytest

from commercebench.contracts import SCHEMA_VERSION, ContractValidationError
from commercebench.contracts.experiment import (
    DialogueHarnessConfig,
    DialogueModelIdentity,
    ExperimentManifest,
)
from commercebench.dialogue.identity import (
    HISTORY_RENDERER_ID,
    HISTORY_RENDERER_VERSION,
    STATE_RENDERER_ID,
    STATE_RENDERER_VERSION,
    compute_prompt_hash,
    validate_dialogue_identity,
)
from commercebench.dialogue.tools import (
    TOOL_SCHEMA_ID,
    TOOL_SCHEMA_VERSION,
    tool_schema_hash,
)

DIALOGUE_PARAMETERS = {
    "max_turns": 6,
    "max_tool_calls_per_turn": 3,
    "history_policy": "full",
    "tool_policy": "direct-tools-v0",
    "tool_simulator": {"id": "commerce-tools-v0", "version": "0.1"},
}

PROMPT_TEXT = "You are a commerce support agent. Be precise."
PROMPT_HASH = compute_prompt_hash(PROMPT_TEXT)
SCHEMA_HASH = tool_schema_hash()


def _identity(**overrides):
    payload = {
        "provider": "offline-adapter",
        "model": "test-model",
        "model_revision": "rev-2026-10",
        "prompt_id": "support-v1",
        "prompt_version": "0.1",
        "prompt_hash": PROMPT_HASH,
        "history_renderer_id": HISTORY_RENDERER_ID,
        "history_renderer_version": HISTORY_RENDERER_VERSION,
        "state_renderer_id": STATE_RENDERER_ID,
        "state_renderer_version": STATE_RENDERER_VERSION,
        "tool_schema_id": TOOL_SCHEMA_ID,
        "tool_schema_version": TOOL_SCHEMA_VERSION,
        "tool_schema_hash": SCHEMA_HASH,
        "generation_parameters": {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_output_tokens": 512,
            "seed": 7,
        },
    }
    payload.update(overrides)
    return DialogueModelIdentity(**payload)


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
        "schema_version": SCHEMA_VERSION,
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


class TestIdentityContract:
    def test_roundtrip(self):
        identity = _identity()
        restored = DialogueModelIdentity.from_dict(identity.to_dict())
        assert restored == identity
        assert restored.to_dict() == identity.to_dict()

    def test_manifest_roundtrip(self):
        manifest = _manifest(
            dialogue=_dialogue_config(), dialogue_model=_identity()
        )
        restored = ExperimentManifest.from_json(manifest.to_json())
        assert restored == manifest
        assert restored.config_fingerprint() == manifest.config_fingerprint()

    def test_model_revision_omitted_when_none(self):
        identity = _identity(model_revision=None)
        assert "model_revision" not in identity.to_dict()
        assert DialogueModelIdentity.from_dict(identity.to_dict()) == identity

    def test_model_revision_preserved_when_set(self):
        identity = _identity()
        assert identity.to_dict()["model_revision"] == "rev-2026-10"

    @pytest.mark.parametrize(
        "field",
        [
            "provider",
            "model",
            "prompt_id",
            "prompt_version",
            "history_renderer_id",
            "history_renderer_version",
            "state_renderer_id",
            "state_renderer_version",
            "tool_schema_id",
            "tool_schema_version",
        ],
    )
    def test_required_fields_reject_empty(self, field):
        with pytest.raises(ContractValidationError):
            _identity(**{field: ""})

    def test_prompt_hash_must_be_sha256(self):
        with pytest.raises(ContractValidationError):
            _identity(prompt_hash="not-a-hash")
        with pytest.raises(ContractValidationError):
            _identity(prompt_hash=PROMPT_HASH.upper())

    def test_tool_schema_hash_must_be_sha256(self):
        with pytest.raises(ContractValidationError):
            _identity(tool_schema_hash="abc123")

    def test_unknown_history_renderer_rejected(self):
        with pytest.raises(ContractValidationError):
            _identity(history_renderer_id="custom-renderer")

    def test_unknown_state_renderer_rejected(self):
        with pytest.raises(ContractValidationError):
            _identity(state_renderer_id="inline-state")

    def test_unknown_tool_schema_id_rejected(self):
        with pytest.raises(ContractValidationError):
            _identity(tool_schema_id="other-schema")

    def test_generation_parameters_must_be_json(self):
        with pytest.raises(ContractValidationError):
            _identity(generation_parameters={"bad": object()})


class TestIdentityFingerprint:
    """Every declared identity dimension is result-affecting and must
    enter the manifest fingerprint."""

    def test_identity_presence_changes_fingerprint(self):
        a = _manifest(dialogue=_dialogue_config())
        b = _manifest(
            dialogue=_dialogue_config(), dialogue_model=_identity()
        )
        assert a.config_fingerprint() != b.config_fingerprint()

    def test_identity_absent_matches_legacy(self):
        # Omit-when-None: manifests without dialogue_model fingerprint
        # identically to Phase 3B.
        with_identity = _manifest(
            dialogue=_dialogue_config(), dialogue_model=None
        )
        without = _manifest(dialogue=_dialogue_config())
        assert (
            with_identity.config_fingerprint()
            == without.config_fingerprint()
        )

    @pytest.mark.parametrize(
        "field,value",
        [
            ("provider", "other-provider"),
            ("model", "other-model"),
            ("model_revision", "rev-2026-11"),
            ("model_revision", None),  # dropping identity changes hash
            ("prompt_id", "support-v2"),
            ("prompt_version", "0.2"),
            (
                "prompt_hash",
                "0" * 64,
            ),
            ("history_renderer_version", "0.2"),
            ("state_renderer_version", "0.2"),
            ("tool_schema_version", "0.2"),
            ("tool_schema_hash", "0" * 64),
        ],
    )
    def test_field_mutation_changes_fingerprint(self, field, value):
        a = _manifest(
            dialogue=_dialogue_config(), dialogue_model=_identity()
        )
        b = _manifest(
            dialogue=_dialogue_config(),
            dialogue_model=_identity(**{field: value}),
        )
        assert a.config_fingerprint() != b.config_fingerprint()

    def test_generation_parameter_mutation(self):
        a = _manifest(
            dialogue=_dialogue_config(), dialogue_model=_identity()
        )
        mutated = dict(_identity().generation_parameters)
        mutated["temperature"] = 0.7
        b = _manifest(
            dialogue=_dialogue_config(),
            dialogue_model=_identity(generation_parameters=mutated),
        )
        assert a.config_fingerprint() != b.config_fingerprint()

    def test_generation_parameter_removed_changes_fingerprint(self):
        a = _manifest(
            dialogue=_dialogue_config(), dialogue_model=_identity()
        )
        b = _manifest(
            dialogue=_dialogue_config(),
            dialogue_model=_identity(generation_parameters={}),
        )
        assert a.config_fingerprint() != b.config_fingerprint()


class TestRuntimeBindingSeams:
    """The declared identity must bind to what the runtime would
    actually expose — fail closed on any mismatch."""

    def test_valid_identity_accepted(self):
        assert validate_dialogue_identity(
            _identity(), prompt_source=PROMPT_TEXT
        ) == _identity()

    def test_prompt_source_mismatch_rejected(self):
        with pytest.raises(ContractValidationError):
            validate_dialogue_identity(
                _identity(), prompt_source="a different prompt"
            )

    def test_tool_schema_hash_mismatch_rejected(self):
        with pytest.raises(ContractValidationError):
            validate_dialogue_identity(
                _identity(tool_schema_hash="0" * 64)
            )

    def test_tool_schema_version_mismatch_rejected(self):
        with pytest.raises(ContractValidationError):
            validate_dialogue_identity(
                _identity(tool_schema_version="9.9")
            )

    def test_history_renderer_version_mismatch_rejected(self):
        with pytest.raises(ContractValidationError):
            validate_dialogue_identity(
                _identity(history_renderer_version="9.9")
            )

    def test_state_renderer_version_mismatch_rejected(self):
        with pytest.raises(ContractValidationError):
            validate_dialogue_identity(
                _identity(state_renderer_version="9.9")
            )

    def test_prompt_hash_helper_stable(self):
        assert compute_prompt_hash(PROMPT_TEXT) == PROMPT_HASH
        assert compute_prompt_hash(PROMPT_TEXT + "x") != PROMPT_HASH
        with pytest.raises(ContractValidationError):
            compute_prompt_hash("")

    def test_declared_schema_matches_runtime_descriptors(self):
        # The declared schema hash is the hash of what the harness
        # actually exposes — same descriptors, same contract.
        identity = _identity()
        assert identity.tool_schema_id == TOOL_SCHEMA_ID
        assert identity.tool_schema_hash == SCHEMA_HASH
