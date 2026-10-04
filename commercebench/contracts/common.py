"""Shared validation and serialization helpers for core contracts."""

from __future__ import annotations

import json
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Tuple, Type, TypeVar

from .errors import ContractValidationError

SCHEMA_VERSION = "0.1"

_E = TypeVar("_E", bound=Enum)


def require_non_empty_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractValidationError(f"{field_name} must be a non-empty string")
    return value


def require_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise ContractValidationError(f"{field_name} must be a string")
    return value


def str_tuple(value: Any, field_name: str) -> Tuple[str, ...]:
    """Normalize a list/tuple of non-empty strings into a tuple."""
    if not isinstance(value, (list, tuple)):
        raise ContractValidationError(f"{field_name} must be a list of strings")
    items = tuple(value)
    for index, item in enumerate(items):
        if not isinstance(item, str) or not item.strip():
            raise ContractValidationError(
                f"{field_name}[{index}] must be a non-empty string"
            )
    return items


def parse_enum(enum_cls: Type[_E], value: Any, field_name: str) -> _E:
    if isinstance(value, enum_cls):
        return value
    try:
        return enum_cls(value)
    except ValueError:
        allowed = ", ".join(member.value for member in enum_cls)
        raise ContractValidationError(
            f"{field_name} must be one of: {allowed} (got {value!r})"
        ) from None


def ensure_json_compatible(value: Any, field_name: str) -> Any:
    try:
        json.dumps(value)
    except (TypeError, ValueError) as exc:
        raise ContractValidationError(
            f"{field_name} must be JSON serializable: {exc}"
        ) from exc
    return value


def require_json_dict(value: Any, field_name: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractValidationError(f"{field_name} must be a JSON object")
    for key in value:
        if not isinstance(key, str):
            raise ContractValidationError(f"{field_name} keys must be strings")
    ensure_json_compatible(value, field_name)
    return value


def require_iso8601(value: Any, field_name: str) -> str:
    """Require an ISO-8601 timestamp string parseable by datetime.fromisoformat."""
    text = require_non_empty_str(value, field_name)
    try:
        datetime.fromisoformat(text)
    except ValueError:
        raise ContractValidationError(
            f"{field_name} must be an ISO-8601 timestamp (got {text!r})"
        ) from None
    return text


def require_non_negative_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractValidationError(f"{field_name} must be a number")
    if value < 0:
        raise ContractValidationError(f"{field_name} must be non-negative")
    return float(value)


def optional_non_negative_int(value: Any, field_name: str) -> Any:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContractValidationError(
            f"{field_name} must be a non-negative integer or null"
        )
    return value


def require_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractValidationError(f"{field_name} must be an integer")
    return value


def require_bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ContractValidationError(f"{field_name} must be a boolean")
    return value


def canonical_json(payload: Any) -> str:
    """Deterministic JSON: sorted keys, compact separators, UTF-8 text."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def dumps_dict(payload: Dict[str, Any]) -> str:
    """Deterministic JSON serialization for contract objects."""
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def expect_mapping(data: Any, contract_name: str) -> Dict[str, Any]:
    if not isinstance(data, dict):
        raise ContractValidationError(f"{contract_name} must be a JSON object")
    return data
