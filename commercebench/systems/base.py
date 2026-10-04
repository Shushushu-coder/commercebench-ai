"""Minimal system-under-test interface.

A system under test is identified by ``system_id``/``system_version`` and
turns one user input into a ``SystemOutput``. Phase 0 only supports
single-turn generation; there is deliberately no conversation or tool API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Protocol, Tuple, runtime_checkable

from commercebench.contracts.trace import (
    MemoryEvent,
    RetrievalEvent,
    ToolEvent,
    UsageStats,
)


@dataclass(frozen=True)
class SystemOutput:
    """What a system produced for one input.

    ``usage``/events default to empty for Phase 0 systems; the fields exist
    so future retrieval-, tool-, or memory-using systems can report facts
    into the RunTrace through the same path.
    """

    output_text: str
    usage: UsageStats = field(default_factory=UsageStats)
    retrieval_events: Tuple[RetrievalEvent, ...] = ()
    tool_events: Tuple[ToolEvent, ...] = ()
    memory_events: Tuple[MemoryEvent, ...] = ()
    runtime_metadata: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class SystemUnderTest(Protocol):
    """Structural interface every evaluated system must satisfy."""

    system_id: str
    system_version: str

    def generate(self, user_input: str) -> SystemOutput:
        """Produce the system output for a single-turn user input."""
        ...


__all__ = ["SystemOutput", "SystemUnderTest"]
