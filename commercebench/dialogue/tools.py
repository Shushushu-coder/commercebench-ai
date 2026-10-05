"""Deterministic commerce tool simulator (Phase 3B).

Provider-neutral, in-memory, fully deterministic. The simulator is
the ONLY authority that can mutate canonical state: systems propose
``ToolEvent`` calls, the harness routes them here, and a typed
``ToolExecutionResult`` is returned — precondition failures are
typed results, never exceptions escaping to the harness.

V1 tools:
- ``lookup_order(order_id)`` — read-only; returns the order snapshot.
- ``request_return(order_id)`` — mutating; precondition-gated.

State layout expected by these tools (fixture-defined, not
contract-pinned):

    {"orders": {order_id: {"status": ..., "returnable": bool, ...}},
     "returns": {order_id: {"status": "requested", ...}}}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from commercebench.contracts.errors import ContractValidationError

from .state import json_state_copy

SIMULATOR_ID = "commerce-tools-v0"
SIMULATOR_VERSION = "0.1"

TOOL_LOOKUP_ORDER = "lookup_order"
TOOL_REQUEST_RETURN = "request_return"

ERROR_UNKNOWN_TOOL = "UNKNOWN_TOOL"
ERROR_ORDER_NOT_FOUND = "ORDER_NOT_FOUND"
ERROR_PRECONDITION_FAILED = "PRECONDITION_FAILED"
ERROR_BAD_ARGUMENTS = "BAD_ARGUMENTS"


@dataclass(frozen=True)
class ToolExecutionResult:
    """Typed outcome of one tool call. ``new_state`` is always a deep
    copy; on failure it equals the input state (no mutation)."""

    status: str  # "ok" | "error"
    result: Any
    new_state: Dict[str, Any]
    error: Optional[str] = None

    def __post_init__(self) -> None:
        if self.status not in ("ok", "error"):
            raise ContractValidationError(
                f"status must be 'ok' or 'error', got {self.status!r}"
            )
        if self.status == "error" and not self.error:
            raise ContractValidationError(
                "error result requires a non-empty error code"
            )
        if self.status == "ok" and self.error is not None:
            raise ContractValidationError(
                "ok result cannot carry an error"
            )


class CommerceToolSimulator:
    """Deterministic tool executor over a JSON state mapping."""

    simulator_id: str = SIMULATOR_ID
    simulator_version: str = SIMULATOR_VERSION

    def call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        state: Dict[str, Any],
    ) -> ToolExecutionResult:
        """Execute one call. Never mutates ``state`` in place; on any
        failure the returned ``new_state`` is an unchanged copy."""
        snapshot = json_state_copy(state)
        if not isinstance(tool_name, str) or not tool_name:
            return ToolExecutionResult(
                status="error",
                result=None,
                new_state=snapshot,
                error=ERROR_BAD_ARGUMENTS,
            )
        if not isinstance(arguments, dict):
            return ToolExecutionResult(
                status="error",
                result=None,
                new_state=snapshot,
                error=ERROR_BAD_ARGUMENTS,
            )
        if tool_name == TOOL_LOOKUP_ORDER:
            return self._lookup_order(arguments, snapshot)
        if tool_name == TOOL_REQUEST_RETURN:
            return self._request_return(arguments, snapshot)
        return ToolExecutionResult(
            status="error",
            result=None,
            new_state=snapshot,
            error=ERROR_UNKNOWN_TOOL,
        )

    # -- tools ------------------------------------------------------------

    def _lookup_order(
        self, arguments: Dict[str, Any], state: Dict[str, Any]
    ) -> ToolExecutionResult:
        order_id = arguments.get("order_id")
        if not isinstance(order_id, str) or not order_id:
            return ToolExecutionResult(
                status="error",
                result=None,
                new_state=state,
                error=ERROR_BAD_ARGUMENTS,
            )
        orders = state.get("orders", {})
        order = orders.get(order_id) if isinstance(orders, dict) else None
        if order is None:
            return ToolExecutionResult(
                status="error",
                result=None,
                new_state=state,
                error=ERROR_ORDER_NOT_FOUND,
            )
        return ToolExecutionResult(
            status="ok",
            result=json_state_copy(order),
            new_state=state,
        )

    def _request_return(
        self, arguments: Dict[str, Any], state: Dict[str, Any]
    ) -> ToolExecutionResult:
        order_id = arguments.get("order_id")
        if not isinstance(order_id, str) or not order_id:
            return ToolExecutionResult(
                status="error",
                result=None,
                new_state=state,
                error=ERROR_BAD_ARGUMENTS,
            )
        orders = state.get("orders", {})
        order = orders.get(order_id) if isinstance(orders, dict) else None
        if order is None:
            return ToolExecutionResult(
                status="error",
                result=None,
                new_state=state,
                error=ERROR_ORDER_NOT_FOUND,
            )
        existing_returns = state.get("returns")
        returns = existing_returns if isinstance(existing_returns, dict) else {}
        precondition_ok = (
            order.get("status") == "delivered"
            and order.get("returnable") is True
            and order_id not in returns
        )
        if not precondition_ok:
            return ToolExecutionResult(
                status="error",
                result=None,
                new_state=state,
                error=ERROR_PRECONDITION_FAILED,
            )
        returns[order_id] = {"status": "requested"}
        if existing_returns is not returns:
            state["returns"] = returns
        return ToolExecutionResult(
            status="ok",
            result={"order_id": order_id, "return_status": "requested"},
            new_state=state,
        )


__all__ = [
    "CommerceToolSimulator",
    "ERROR_BAD_ARGUMENTS",
    "ERROR_ORDER_NOT_FOUND",
    "ERROR_PRECONDITION_FAILED",
    "ERROR_UNKNOWN_TOOL",
    "SIMULATOR_ID",
    "SIMULATOR_VERSION",
    "TOOL_LOOKUP_ORDER",
    "TOOL_REQUEST_RETURN",
    "ToolExecutionResult",
]
