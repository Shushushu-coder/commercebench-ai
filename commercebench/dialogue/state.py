"""Canonical dialogue state helpers (Phase 3B).

State is a JSON-compatible ``Dict[str, Any]`` owned by the dialogue
harness. These helpers give the two operations the harness and the
evaluator need: snapshot-quality deep copies and deterministic
recursive subset matching.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from commercebench.contracts.errors import ContractValidationError


def json_state_copy(state: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-copy a JSON-compatible state via a serialization round-trip.

    Round-tripping through JSON both deep-copies and proves the state
    stayed JSON-compatible (no aliasing into ``state_before`` snapshots).
    """
    try:
        return json.loads(json.dumps(state))
    except (TypeError, ValueError) as exc:
        raise ContractValidationError(
            f"state is not JSON-compatible: {exc}"
        ) from exc


def state_subset_matches(expected: Any, observed: Any) -> bool:
    """Recursive dict-subset match; lists and scalars compare exactly.

    ``expected`` may cover only part of ``observed``: every key in a
    dict-valued ``expected`` must exist in ``observed`` and itself
    match as a subset. Non-dict leaves (including lists) require exact
    equality — Phase 3B deliberately has no set-like list semantics.
    """
    if isinstance(expected, dict):
        if not isinstance(observed, dict):
            return False
        for key, expected_value in expected.items():
            if key not in observed:
                return False
            if not state_subset_matches(expected_value, observed[key]):
                return False
        return True
    return expected == observed


def state_subset_mismatches(
    expected: Any, observed: Any, path: str = ""
) -> List[Dict[str, Any]]:
    """Return a deterministic list of subset-match mismatches.

    Each entry is ``{"path", "expected", "observed"}`` — focused
    evidence for evaluator ``details`` instead of full-state dumps.
    """
    mismatches: List[Dict[str, Any]] = []
    if isinstance(expected, dict):
        if not isinstance(observed, dict):
            mismatches.append(
                {
                    "path": path or "$",
                    "expected": expected,
                    "observed": observed,
                }
            )
            return mismatches
        for key in sorted(expected):
            child_path = f"{path}.{key}" if path else key
            if key not in observed:
                mismatches.append(
                    {
                        "path": child_path,
                        "expected": expected[key],
                        "observed": None,
                    }
                )
                continue
            mismatches.extend(
                state_subset_mismatches(
                    expected[key], observed[key], child_path
                )
            )
        return mismatches
    if expected != observed:
        mismatches.append(
            {"path": path or "$", "expected": expected, "observed": observed}
        )
    return mismatches


def split_delta_obligations(
    delta: Dict[str, Any], prefix: str = ""
) -> List[Tuple[str, Any]]:
    """Flatten a nested expected-state delta into (path, leaf) pairs."""
    obligations: List[Tuple[str, Any]] = []
    for key in sorted(delta):
        value = delta[key]
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict) and value:
            obligations.extend(split_delta_obligations(value, path))
        else:
            obligations.append((path, value))
    return obligations


__all__ = [
    "json_state_copy",
    "split_delta_obligations",
    "state_subset_matches",
    "state_subset_mismatches",
]
