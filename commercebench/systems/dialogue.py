"""Dialogue systems under test (Phase 3B).

``DialogueSystemUnderTest`` is a separate protocol from the legacy
single-turn ``SystemUnderTest``: ``respond`` receives the fully
constructed per-turn context (explicit application-managed history,
a deep-copied state view, the offered tool descriptors) and returns
text + structured assistant tags + proposed tool calls.

Provider-native hidden sessions are NOT part of this contract: the
explicit ``DialogueTurnInput.history`` is the authoritative input —
a trace must always be able to reconstruct what the system saw.

``ScriptedDialogueSystem`` is the deterministic baseline used to
prove harness/evaluator correctness before any real LLM exists.
"""

from __future__ import annotations

import hashlib
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Protocol,
    Sequence,
    runtime_checkable,
)

from commercebench.contracts.common import canonical_json
from commercebench.contracts.dialogue import (
    DialogueTurnInput,
    DialogueTurnOutput,
)
from commercebench.contracts.errors import ContractValidationError
from commercebench.contracts.trace import ToolEvent, UsageStats


@runtime_checkable
class DialogueSystemUnderTest(Protocol):
    """Structural interface for multi-turn systems under test.

    Explicit ``turn_input.history`` is the authoritative context; a
    provider-native hidden session is not an acceptable substitute
    because the trace could not reconstruct it.
    """

    system_id: str
    system_version: str

    def respond(self, turn_input: DialogueTurnInput) -> DialogueTurnOutput:
        """Produce one assistant step for the given turn context."""
        ...


DEFAULT_SCRIPT_FALLBACK = ""


def _script_fingerprint(script: Mapping[str, Any]) -> str:
    """Short stable fingerprint of the script payload."""
    return hashlib.sha256(
        canonical_json(dict(script)).encode("utf-8")
    ).hexdigest()[:12]


class ScriptedDialogueSystem:
    """Replays a per-turn step script; exists to exercise the dialogue
    kernel deterministically.

    Script shape::

        {
          "<turn_id>": [
            {"output_text": "...", "assistant_tags": ["clarify"],
             "tool_calls": [{"tool_name": "lookup_order",
                             "arguments": {"order_id": "A1001"}}]},
            ...   # additional inner-loop steps for the same turn
          ],
          ...
        }

    Step selection key is ``(turn_id, invocation_index)`` where the
    invocation index counts ``respond`` calls within the current turn.
    Exhausted script entries return a plain empty output (which ends
    the inner loop). The script content is hashed into
    ``system_version`` so a changed script always changes the recorded
    system identity.
    """

    def __init__(
        self,
        script: Mapping[str, Sequence[Mapping[str, Any]]],
        system_id: str = "scripted-dialogue",
        system_version: str = "0.1",
    ) -> None:
        if not isinstance(script, Mapping):
            raise ContractValidationError("script must be a mapping")
        self.system_id = system_id
        digest = _script_fingerprint(script)
        self.system_version = f"{system_version}+script-{digest}"
        self._script: Dict[str, List[Dict[str, Any]]] = {}
        for turn_id, steps in script.items():
            if not isinstance(turn_id, str) or not turn_id:
                raise ContractValidationError(
                    "script keys must be non-empty turn ids"
                )
            normalized: List[Dict[str, Any]] = []
            for step in steps:
                if not isinstance(step, Mapping):
                    raise ContractValidationError(
                        f"script[{turn_id!r}] entries must be mappings"
                    )
                normalized.append(dict(step))
            self._script[turn_id] = normalized
        self._invocation_counts: Dict[str, int] = {}

    def respond(self, turn_input: DialogueTurnInput) -> DialogueTurnOutput:
        index = self._invocation_counts.get(turn_input.turn_id, 0)
        self._invocation_counts[turn_input.turn_id] = index + 1
        steps = self._script.get(turn_input.turn_id, [])
        if index >= len(steps):
            return DialogueTurnOutput(
                output_text=DEFAULT_SCRIPT_FALLBACK,
                usage=UsageStats(input_tokens=0, output_tokens=0,
                                 total_tokens=0),
            )
        step = steps[index]
        tool_calls = []
        for raw_call in step.get("tool_calls", ()):
            if not isinstance(raw_call, Mapping):
                raise ContractValidationError(
                    "scripted tool_calls entries must be mappings"
                )
            tool_calls.append(
                ToolEvent(
                    tool_name=raw_call.get("tool_name", ""),
                    arguments=dict(raw_call.get("arguments", {})),
                )
            )
        return DialogueTurnOutput(
            output_text=step.get("output_text", ""),
            tool_calls=tuple(tool_calls),
            assistant_tags=tuple(step.get("assistant_tags", ())),
            usage=UsageStats(
                input_tokens=step.get("input_tokens", 0),
                output_tokens=step.get("output_tokens", 0),
                total_tokens=(
                    step.get("input_tokens", 0)
                    + step.get("output_tokens", 0)
                ),
            ),
        )


__all__ = [
    "DEFAULT_SCRIPT_FALLBACK",
    "DialogueSystemUnderTest",
    "ScriptedDialogueSystem",
]
