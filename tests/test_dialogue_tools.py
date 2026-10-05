"""Phase 3B deterministic tool simulator tests."""

from __future__ import annotations

import pytest

from commercebench.dialogue.state import json_state_copy, state_subset_matches
from commercebench.dialogue.tools import (
    CommerceToolSimulator,
    ERROR_BAD_ARGUMENTS,
    ERROR_ORDER_NOT_FOUND,
    ERROR_PRECONDITION_FAILED,
    ERROR_UNKNOWN_TOOL,
)


def _state():
    return {
        "orders": {
            "OK1": {
                "status": "delivered",
                "returnable": True,
            },
            "SHIP1": {
                "status": "shipped",
                "returnable": False,
            },
            "DONE1": {
                "status": "delivered",
                "returnable": True,
            },
        },
        "returns": {"DONE1": {"status": "requested"}},
    }


@pytest.fixture
def simulator():
    return CommerceToolSimulator()


class TestLookupOrder:
    def test_hit(self, simulator):
        state = _state()
        result = simulator.call(
            "lookup_order", {"order_id": "OK1"}, state
        )
        assert result.status == "ok"
        assert result.result["status"] == "delivered"
        assert result.new_state == state

    def test_miss(self, simulator):
        state = _state()
        result = simulator.call(
            "lookup_order", {"order_id": "NOPE"}, state
        )
        assert result.status == "error"
        assert result.error == ERROR_ORDER_NOT_FOUND
        assert result.new_state == state

    def test_bad_args(self, simulator):
        state = _state()
        result = simulator.call("lookup_order", {}, state)
        assert result.status == "error"
        assert result.error == ERROR_BAD_ARGUMENTS


class TestRequestReturn:
    def test_success(self, simulator):
        state = _state()
        before = json_state_copy(state)
        result = simulator.call(
            "request_return", {"order_id": "OK1"}, state
        )
        assert result.status == "ok"
        assert result.new_state["returns"]["OK1"]["status"] == "requested"
        # input state mutated? No: caller receives new_state; the passed
        # dict was already the working copy — verify caller-side state
        # the harness held is replaced, not both.
        assert result.new_state != before

    def test_shipped_order_rejected(self, simulator):
        state = _state()
        before = json_state_copy(state)
        result = simulator.call(
            "request_return", {"order_id": "SHIP1"}, state
        )
        assert result.status == "error"
        assert result.error == ERROR_PRECONDITION_FAILED
        assert result.new_state == before
        # hard rule: failed tool cannot mutate state
        assert result.new_state == _state()

    def test_missing_order_rejected(self, simulator):
        state = _state()
        result = simulator.call(
            "request_return", {"order_id": "GHOST"}, state
        )
        assert result.status == "error"
        assert result.error == ERROR_ORDER_NOT_FOUND
        assert result.new_state == _state()

    def test_existing_return_rejected(self, simulator):
        state = _state()
        result = simulator.call(
            "request_return", {"order_id": "DONE1"}, state
        )
        assert result.status == "error"
        assert result.error == ERROR_PRECONDITION_FAILED
        assert result.new_state == _state()

    def test_failed_call_does_not_create_returns_key(self, simulator):
        state = _state()
        del state["returns"]
        result = simulator.call(
            "request_return", {"order_id": "SHIP1"}, state
        )
        assert result.status == "error"
        assert "returns" not in result.new_state


class TestUnknownAndBadInput:
    def test_unknown_tool(self, simulator):
        result = simulator.call("fly_to_moon", {}, _state())
        assert result.status == "error"
        assert result.error == ERROR_UNKNOWN_TOOL

    def test_empty_tool_name(self, simulator):
        result = simulator.call("", {}, _state())
        assert result.status == "error"

    def test_non_dict_arguments(self, simulator):
        result = simulator.call("lookup_order", "A1", _state())
        assert result.status == "error"
        assert result.error == ERROR_BAD_ARGUMENTS


class TestStateHelpers:
    def test_subset_match_basic(self):
        assert state_subset_matches({"a": 1}, {"a": 1, "b": 2})
        assert not state_subset_matches({"a": 2}, {"a": 1})

    def test_subset_match_nested(self):
        expected = {"returns": {"A": {"status": "requested"}}}
        observed = {
            "returns": {"A": {"status": "requested", "extra": 1}},
            "orders": {},
        }
        assert state_subset_matches(expected, observed)
        assert not state_subset_matches(
            {"returns": {"A": {"status": "done"}}}, observed
        )

    def test_subset_match_missing_key(self):
        assert not state_subset_matches({"a": {"b": 1}}, {"a": {}})

    def test_list_exact_equality(self):
        assert state_subset_matches({"a": [1, 2]}, {"a": [1, 2]})
        assert not state_subset_matches({"a": [1]}, {"a": [1, 2]})
        assert not state_subset_matches({"a": [1, 2]}, {"a": [2, 1]})

    def test_json_copy_isolation(self):
        state = _state()
        copy = json_state_copy(state)
        copy["orders"]["OK1"]["status"] = "mutated"
        assert state["orders"]["OK1"]["status"] == "delivered"
