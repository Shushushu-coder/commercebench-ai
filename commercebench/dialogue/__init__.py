"""Deterministic stateful dialogue runtime (Phase 3B/3D).

The harness owns the canonical business state and the visible
history; systems propose tool calls; the tool simulator is the only
authority that can mutate state. Phase 3D adds the registered tool
policies, the fail-closed manifest→harness binding, defensive
evidence snapshots, and the real-LLM identity validation seams.
"""

from .harness import DialogueHarness, run_dialogue_case
from .identity import (
    compute_prompt_hash,
    runtime_identity_metadata,
    validate_dialogue_identity,
)
from .policy import (
    ALLOWED_TOOL_POLICIES,
    ERROR_POLICY_PRECONDITION_FAILED,
    TOOL_POLICY_DIRECT,
    TOOL_POLICY_LOOKUP_BEFORE_MUTATE,
    evaluate_tool_call,
    is_state_changing,
    validate_tool_policy,
)
from .state import (
    delta_obligations_satisfied,
    json_leaf_equal,
    json_state_copy,
    json_value_copy,
    json_value_equal,
    state_subset_matches,
)
from .tools import (
    STATE_CHANGING_TOOLS,
    TOOL_DESCRIPTORS,
    TOOL_SCHEMA_ID,
    TOOL_SCHEMA_VERSION,
    CommerceToolSimulator,
    ToolExecutionResult,
    tool_schema_hash,
)

__all__ = [
    "ALLOWED_TOOL_POLICIES",
    "CommerceToolSimulator",
    "DialogueHarness",
    "ERROR_POLICY_PRECONDITION_FAILED",
    "STATE_CHANGING_TOOLS",
    "TOOL_DESCRIPTORS",
    "TOOL_POLICY_DIRECT",
    "TOOL_POLICY_LOOKUP_BEFORE_MUTATE",
    "TOOL_SCHEMA_ID",
    "TOOL_SCHEMA_VERSION",
    "ToolExecutionResult",
    "compute_prompt_hash",
    "delta_obligations_satisfied",
    "evaluate_tool_call",
    "is_state_changing",
    "json_leaf_equal",
    "json_state_copy",
    "json_value_copy",
    "json_value_equal",
    "run_dialogue_case",
    "runtime_identity_metadata",
    "state_subset_matches",
    "tool_schema_hash",
    "validate_dialogue_identity",
    "validate_tool_policy",
]
