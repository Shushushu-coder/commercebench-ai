"""Tool-call policies (Phase 3D).

``tool_policy`` is a fingerprinted harness parameter, so every
registered policy must actually change what the runtime accepts —
an unknown label is rejected at config validation and a known label
gates execution deterministically.

Registered policies:

- ``direct-tools-v0`` — no gating; every proposed call reaches the
  simulator.
- ``lookup-before-mutate-v0`` — a state-changing tool call for an
  ``order_id`` is only forwarded to the simulator when a
  ``lookup_order(order_id)`` already succeeded earlier in the same
  dialogue. Otherwise the call is recorded as a rejected
  ``ToolEvent`` (``status="error"``,
  ``error="POLICY_PRECONDITION_FAILED"``) and state is left
  untouched.

The lookup set is derived deterministically from the executed tool
history the harness already owns — no hidden mutable global state.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Set
from commercebench.contracts.errors import ContractValidationError
from commercebench.contracts.experiment import ALLOWED_TOOL_POLICIES

from .tools import STATE_CHANGING_TOOLS

TOOL_POLICY_DIRECT = "direct-tools-v0"
TOOL_POLICY_LOOKUP_BEFORE_MUTATE = "lookup-before-mutate-v0"

# Fail-fast drift guard: the named policies below must be exactly
# what the contract registry accepts.
assert TOOL_POLICY_DIRECT in ALLOWED_TOOL_POLICIES
assert TOOL_POLICY_LOOKUP_BEFORE_MUTATE in ALLOWED_TOOL_POLICIES

#: Rejected calls consume a tool-call slot but never reach the
#: simulator, so state cannot change and the event is honest.
ERROR_POLICY_PRECONDITION_FAILED = "POLICY_PRECONDITION_FAILED"


def validate_tool_policy(policy: Any, field_name: str = "tool_policy") -> str:
    """Require a registered policy id; anything else fails closed."""
    if not isinstance(policy, str) or not policy:
        raise ContractValidationError(
            f"{field_name} must be a non-empty string"
        )
    if policy not in ALLOWED_TOOL_POLICIES:
        raise ContractValidationError(
            f"{field_name} must be one of "
            f"{list(ALLOWED_TOOL_POLICIES)}, got {policy!r}"
        )
    return policy


def is_state_changing(tool_name: str) -> bool:
    """Whether a tool mutates canonical state under the V0 toolset."""
    return tool_name in STATE_CHANGING_TOOLS


def evaluate_tool_call(
    policy: str,
    tool_name: str,
    arguments: Dict[str, Any],
    successful_lookups: Set[str],
) -> Optional[Dict[str, Any]]:
    """Gate one proposed tool call against the policy.

    ``successful_lookups`` holds the ``order_id`` values for which a
    ``lookup_order`` call already returned ``status="ok"`` earlier in
    the run (including earlier steps of the current turn — the lookup
    must only precede the mutation, not a prior turn).

    Returns ``None`` when the call is allowed, otherwise a
    JSON-compatible diagnostic payload describing the rejection.
    """
    validate_tool_policy(policy)
    if policy == TOOL_POLICY_DIRECT:
        return None
    if policy == TOOL_POLICY_LOOKUP_BEFORE_MUTATE:
        if not is_state_changing(tool_name):
            return None
        order_id = arguments.get("order_id")
        if isinstance(order_id, str) and order_id in successful_lookups:
            return None
        return {
            "policy": policy,
            "rejected_tool": tool_name,
            "order_id": order_id if isinstance(order_id, str) else None,
            "error_code": ERROR_POLICY_PRECONDITION_FAILED,
            "reason": (
                "state-changing tool requires an earlier successful "
                "lookup_order for the same order_id"
            ),
        }
    # validate_tool_policy above rejects unregistered ids; unreachable.
    raise ContractValidationError(f"unhandled tool policy {policy!r}")


__all__ = [
    "ALLOWED_TOOL_POLICIES",
    "ERROR_POLICY_PRECONDITION_FAILED",
    "TOOL_POLICY_DIRECT",
    "TOOL_POLICY_LOOKUP_BEFORE_MUTATE",
    "evaluate_tool_call",
    "is_state_changing",
    "validate_tool_policy",
]
