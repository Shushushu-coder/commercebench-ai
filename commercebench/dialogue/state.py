"""Canonical dialogue state helpers (Phase 3B/3D).

State is a JSON-compatible ``Dict[str, Any]`` owned by the dialogue
harness. These helpers give the harness, the evaluator, and the
integrity boundaries what they need: deep copies, strict JSON leaf
equality (``True != 1``; ``1 == 1.0``), deterministic recursive
subset matching, and the shared newly-caused ``expected_state_delta``
predicate.
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


def json_value_copy(value: Any) -> Any:
    """Deep-copy any JSON-compatible value (dict, list, or scalar).

    Same guarantees as ``json_state_copy`` but for arbitrary JSON
    payloads — used to snapshot mutable evidence (history entries,
    tool arguments/results) at every boundary a system can reach.
    """
    try:
        return json.loads(json.dumps(value))
    except (TypeError, ValueError) as exc:
        raise ContractValidationError(
            f"value is not JSON-compatible: {exc}"
        ) from exc


def json_leaf_equal(expected: Any, observed: Any) -> bool:
    """Strict JSON scalar equality (Phase 3D).

    ``bool`` is a distinct JSON type from ``number``: ``True != 1``.
    ``int`` and ``float`` share the JSON number domain, so ``1 == 1.0``.
    All other leaves require identical Python type and equal value.
    """
    if isinstance(expected, bool) or isinstance(observed, bool):
        return (
            isinstance(expected, bool)
            and isinstance(observed, bool)
            and expected == observed
        )
    if isinstance(expected, (int, float)) and isinstance(
        observed, (int, float)
    ):
        return expected == observed
    return type(expected) is type(observed) and expected == observed


def json_value_equal(expected: Any, observed: Any) -> bool:
    """Recursive strict JSON equality: dicts and lists compare
    structurally; leaves use :func:`json_leaf_equal`."""
    if isinstance(expected, dict) or isinstance(observed, dict):
        if not (isinstance(expected, dict) and isinstance(observed, dict)):
            return False
        if set(expected) != set(observed):
            return False
        return all(
            json_value_equal(expected[key], observed[key])
            for key in expected
        )
    if isinstance(expected, list) or isinstance(observed, list):
        if not (isinstance(expected, list) and isinstance(observed, list)):
            return False
        return len(expected) == len(observed) and all(
            json_value_equal(e, o) for e, o in zip(expected, observed)
        )
    return json_leaf_equal(expected, observed)


def state_subset_matches(expected: Any, observed: Any) -> bool:
    """Recursive dict-subset match; lists and scalars compare exactly.

    ``expected`` may cover only part of ``observed``: every key in a
    dict-valued ``expected`` must exist in ``observed`` and itself
    match as a subset. Non-dict leaves (including lists) require
    :func:`json_value_equal` — Phase 3D makes leaf comparison
    type-strict (``True != 1``) while keeping the JSON number domain
    unified (``1 == 1.0``).
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
    return json_value_equal(expected, observed)


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
    if not json_value_equal(expected, observed):
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


def state_leaf_at(state: Dict[str, Any], path: str) -> Tuple[bool, Any]:
    """Return ``(present, value)`` for one flattened delta path."""
    current: Any = state
    for segment in path.split("."):
        if not isinstance(current, dict) or segment not in current:
            return False, None
        current = current[segment]
    return True, current


def delta_obligations_satisfied(
    delta: Dict[str, Any],
    state_before: Dict[str, Any],
    state_after: Dict[str, Any],
) -> bool:
    """Whether every flattened ``expected_state_delta`` leaf is a
    *newly caused* change (Phase 3D shared harness/evaluator rule).

    A leaf obligation is satisfied only when it holds in
    ``state_after`` and did NOT already hold in ``state_before`` —
    the turn must have produced the change, not just satisfy it
    incidentally. An empty ``delta`` is trivially satisfied.
    """
    for path, leaf in split_delta_obligations(delta):
        present_after, after = state_leaf_at(state_after, path)
        present_before, before = state_leaf_at(state_before, path)
        if not present_after or not json_leaf_equal(after, leaf):
            return False
        if present_before and json_leaf_equal(before, leaf):
            return False
    return True


__all__ = [
    "delta_obligations_satisfied",
    "json_leaf_equal",
    "json_state_copy",
    "json_value_copy",
    "json_value_equal",
    "split_delta_obligations",
    "state_leaf_at",
    "state_subset_matches",
    "state_subset_mismatches",
]
