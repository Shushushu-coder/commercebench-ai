"""Deterministic stateful dialogue runtime (Phase 3B).

The harness owns the canonical business state and the visible
history; systems propose tool calls; the tool simulator is the only
authority that can mutate state.
"""

from .harness import DialogueHarness, run_dialogue_case
from .state import json_state_copy, state_subset_matches
from .tools import CommerceToolSimulator, ToolExecutionResult

__all__ = [
    "CommerceToolSimulator",
    "DialogueHarness",
    "ToolExecutionResult",
    "json_state_copy",
    "run_dialogue_case",
    "state_subset_matches",
]
