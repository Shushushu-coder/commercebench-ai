"""DeterministicSystem: a fixed-response fake system for Phase 0.

It replays a configured input -> response mapping and returns a stable
fallback for unknown inputs. No model, no network, no randomness: the same
input always yields the same output.
"""

from __future__ import annotations

from typing import Mapping, Optional

from commercebench.contracts.errors import ContractValidationError
from commercebench.contracts.trace import UsageStats

from .base import SystemOutput

DEFAULT_FALLBACK_RESPONSE = "抱歉，我暂时无法回答这个问题。"


class DeterministicSystem:
    """Replays fixed responses; exists only to exercise the kernel plumbing."""

    def __init__(
        self,
        responses: Optional[Mapping[str, str]] = None,
        fallback_response: str = DEFAULT_FALLBACK_RESPONSE,
        system_id: str = "deterministic-baseline",
        system_version: str = "0.1",
    ) -> None:
        self.system_id = system_id
        self.system_version = system_version
        if not fallback_response:
            raise ContractValidationError(
                "fallback_response must be a non-empty string"
            )
        self._fallback_response = fallback_response
        self._responses = {}
        for key, value in (responses or {}).items():
            if not key or not isinstance(value, str):
                raise ContractValidationError(
                    "responses must map non-empty input strings to strings"
                )
            self._responses[key] = value

    def generate(self, user_input: str) -> SystemOutput:
        text = self._responses.get(user_input, self._fallback_response)
        return SystemOutput(
            output_text=text,
            usage=UsageStats(input_tokens=0, output_tokens=0, total_tokens=0),
        )


__all__ = ["DEFAULT_FALLBACK_RESPONSE", "DeterministicSystem"]
