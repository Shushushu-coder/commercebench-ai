"""Real-LLM dialogue identity binding and validation (Phase 3D).

This module is the seam between the declared, fingerprinted
``DialogueModelIdentity`` on ``ExperimentManifest.dialogue_model``
and the runtime objects a provider adapter would actually resolve.
Phase 3D introduces no provider calls — but every binding a Phase
3E adapter needs already exists and fails closed:

- :func:`compute_prompt_hash` — SHA-256 of canonical prompt template
  text; adapters hash the rendered source before calling a model.
- :func:`tool_schema_hash` (re-exported) — SHA-256 of the canonical
  ``TOOL_DESCRIPTORS`` payload the harness exposes.
- :func:`validate_dialogue_identity` — fail-closed check that a
  declared identity matches the runtime: known renderer ids, the
  baseline tool schema id/version/hash, and — when a rendered
  ``prompt_source`` is supplied — the exact prompt hash.

What stays declarative until Phase 3E: ``provider``/``model`` /
``model_revision`` (resolved by the provider, not by this repo) and
``generation_parameters`` (declared keys a Phase 3E adapter must
fail closed on when unsupported by the provider).
"""

from __future__ import annotations

import hashlib
from commercebench.contracts.errors import ContractValidationError
from commercebench.contracts.experiment import (
    DialogueModelIdentity,
)

from .tools import (
    TOOL_SCHEMA_ID,
    TOOL_SCHEMA_VERSION,
    tool_schema_hash,
)

#: Baseline provider-neutral renderer identities (Phase 3D). The
#: registry lives in ``contracts.experiment`` (``ALLOWED_*``); these
#: names document the concrete V0 renderers.
HISTORY_RENDERER_ID = "full-history-messages-v0"
HISTORY_RENDERER_VERSION = "0.1"
STATE_RENDERER_ID = "state-json-v0"
STATE_RENDERER_VERSION = "0.1"


def compute_prompt_hash(prompt_source: str) -> str:
    """SHA-256 of canonical prompt template text.

    ``prompt_source`` is the exact rendered template text a provider
    call would send — hashing it binds the declared
    ``prompt_id``/``prompt_version`` to actual bytes.
    """
    if not isinstance(prompt_source, str) or not prompt_source:
        raise ContractValidationError(
            "prompt_source must be a non-empty string"
        )
    return hashlib.sha256(prompt_source.encode("utf-8")).hexdigest()


def validate_dialogue_identity(
    identity: DialogueModelIdentity,
    *,
    prompt_source: Optional[str] = None,
) -> DialogueModelIdentity:
    """Fail-closed check that a declared identity binds to runtime
    reality. Returns the identity unchanged on success.

    - Renderer ids must be registered (``ALLOWED_*``); versions are
      checked against the baseline constants when the baseline
      renderer id is used.
    - ``tool_schema_id``/``tool_schema_version``/``tool_schema_hash``
      must match the descriptors the harness actually exposes —
      a mismatch means the declared schema is not what the system
      would see.
    - When ``prompt_source`` is provided, its hash must equal
      ``prompt_hash`` — the declared prompt is the rendered prompt.
    """
    if not isinstance(identity, DialogueModelIdentity):
        raise ContractValidationError(
            "identity must be a DialogueModelIdentity"
        )
    if (
        identity.history_renderer_id == HISTORY_RENDERER_ID
        and identity.history_renderer_version != HISTORY_RENDERER_VERSION
    ):
        raise ContractValidationError(
            f"history_renderer_version {identity.history_renderer_version!r} "
            f"does not match baseline {HISTORY_RENDERER_ID}="
            f"{HISTORY_RENDERER_VERSION!r}"
        )
    if (
        identity.state_renderer_id == STATE_RENDERER_ID
        and identity.state_renderer_version != STATE_RENDERER_VERSION
    ):
        raise ContractValidationError(
            f"state_renderer_version {identity.state_renderer_version!r} "
            f"does not match baseline {STATE_RENDERER_ID}="
            f"{STATE_RENDERER_VERSION!r}"
        )
    if identity.tool_schema_id != TOOL_SCHEMA_ID:
        raise ContractValidationError(
            f"tool_schema_id {identity.tool_schema_id!r} does not match "
            f"the exposed schema {TOOL_SCHEMA_ID!r}"
        )
    if identity.tool_schema_version != TOOL_SCHEMA_VERSION:
        raise ContractValidationError(
            f"tool_schema_version {identity.tool_schema_version!r} does "
            f"not match the exposed schema version "
            f"{TOOL_SCHEMA_VERSION!r}"
        )
    actual_schema_hash = tool_schema_hash()
    if identity.tool_schema_hash != actual_schema_hash:
        raise ContractValidationError(
            "tool_schema_hash does not match the exposed tool "
            f"schema: declared {identity.tool_schema_hash!r}, "
            f"runtime {actual_schema_hash!r}"
        )
    if prompt_source is not None:
        actual_prompt_hash = compute_prompt_hash(prompt_source)
        if identity.prompt_hash != actual_prompt_hash:
            raise ContractValidationError(
                "prompt_hash does not match the rendered prompt: "
                f"declared {identity.prompt_hash!r}, runtime "
                f"{actual_prompt_hash!r}"
            )
    return identity


def runtime_identity_metadata(
    identity: Optional[DialogueModelIdentity],
) -> Dict[str, Any]:
    """Resolved observability payload for trace ``runtime_metadata``.

    Always records the schema the runtime actually exposed; the
    declared identity is embedded when present so audits can compare
    fingerprinted intent against runtime reality. This is evidence
    only — it never substitutes for the pre-run binding checks.
    """
    metadata: Dict[str, Any] = {
        "tool_schema": {
            "id": TOOL_SCHEMA_ID,
            "version": TOOL_SCHEMA_VERSION,
            "hash": tool_schema_hash(),
        },
        "history_renderer": {
            "id": HISTORY_RENDERER_ID,
            "version": HISTORY_RENDERER_VERSION,
        },
        "state_renderer": {
            "id": STATE_RENDERER_ID,
            "version": STATE_RENDERER_VERSION,
        },
    }
    if identity is not None:
        metadata["dialogue_model_identity"] = identity.to_dict()
    return metadata


__all__ = [
    "HISTORY_RENDERER_ID",
    "HISTORY_RENDERER_VERSION",
    "STATE_RENDERER_ID",
    "STATE_RENDERER_VERSION",
    "compute_prompt_hash",
    "runtime_identity_metadata",
    "tool_schema_hash",
    "validate_dialogue_identity",
]
