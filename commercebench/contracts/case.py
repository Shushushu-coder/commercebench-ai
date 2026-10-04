"""CaseSpec: a single benchmark case (V0 contract)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple

from .common import (
    dumps_dict,
    expect_mapping,
    parse_enum,
    require_json_dict,
    require_non_empty_str,
    str_tuple,
)
from .errors import ContractValidationError


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class ConversationType(str, Enum):
    SINGLE_TURN = "single_turn"
    MULTI_TURN = "multi_turn"
    LONG_CONTEXT = "long_context"


class Answerability(str, Enum):
    ANSWERABLE = "answerable"
    INSUFFICIENT_INFORMATION = "insufficient_information"
    SHOULD_CLARIFY = "should_clarify"
    SHOULD_ESCALATE = "should_escalate"


@dataclass(frozen=True)
class CaseSpec:
    """One benchmark case: user input plus deterministic evaluation anchors."""

    schema_version: str
    case_id: str
    benchmark_version: str

    task_family: str
    intent: str
    difficulty: Difficulty
    conversation_type: ConversationType
    answerability: Answerability

    user_input: str

    required_facts: Tuple[str, ...] = ()
    forbidden_claims: Tuple[str, ...] = ()

    expected_response: Optional[str] = None

    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "schema_version",
            require_non_empty_str(self.schema_version, "schema_version"),
        )
        object.__setattr__(
            self, "case_id", require_non_empty_str(self.case_id, "case_id")
        )
        object.__setattr__(
            self,
            "benchmark_version",
            require_non_empty_str(self.benchmark_version, "benchmark_version"),
        )
        object.__setattr__(
            self,
            "task_family",
            require_non_empty_str(self.task_family, "task_family"),
        )
        object.__setattr__(
            self, "intent", require_non_empty_str(self.intent, "intent")
        )
        object.__setattr__(
            self,
            "difficulty",
            parse_enum(Difficulty, self.difficulty, "difficulty"),
        )
        object.__setattr__(
            self,
            "conversation_type",
            parse_enum(ConversationType, self.conversation_type, "conversation_type"),
        )
        object.__setattr__(
            self,
            "answerability",
            parse_enum(Answerability, self.answerability, "answerability"),
        )
        object.__setattr__(
            self,
            "user_input",
            require_non_empty_str(self.user_input, "user_input"),
        )
        object.__setattr__(
            self,
            "required_facts",
            str_tuple(self.required_facts, "required_facts"),
        )
        object.__setattr__(
            self,
            "forbidden_claims",
            str_tuple(self.forbidden_claims, "forbidden_claims"),
        )
        if self.expected_response is not None:
            object.__setattr__(
                self,
                "expected_response",
                require_non_empty_str(self.expected_response, "expected_response"),
            )
        object.__setattr__(
            self, "metadata", dict(require_json_dict(self.metadata, "metadata"))
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "case_id": self.case_id,
            "benchmark_version": self.benchmark_version,
            "task_family": self.task_family,
            "intent": self.intent,
            "difficulty": self.difficulty.value,
            "conversation_type": self.conversation_type.value,
            "answerability": self.answerability.value,
            "user_input": self.user_input,
            "required_facts": list(self.required_facts),
            "forbidden_claims": list(self.forbidden_claims),
            "expected_response": self.expected_response,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Any) -> "CaseSpec":
        data = expect_mapping(data, "CaseSpec")
        try:
            return cls(
                schema_version=data["schema_version"],
                case_id=data["case_id"],
                benchmark_version=data["benchmark_version"],
                task_family=data["task_family"],
                intent=data["intent"],
                difficulty=data["difficulty"],
                conversation_type=data["conversation_type"],
                answerability=data["answerability"],
                user_input=data["user_input"],
                required_facts=data.get("required_facts", ()),
                forbidden_claims=data.get("forbidden_claims", ()),
                expected_response=data.get("expected_response"),
                metadata=data.get("metadata", {}),
            )
        except KeyError as exc:
            raise ContractValidationError(
                f"CaseSpec missing required field: {exc.args[0]}"
            ) from exc

    def to_json(self) -> str:
        return dumps_dict(self.to_dict())

    @classmethod
    def from_json(cls, payload: str) -> "CaseSpec":
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ContractValidationError(f"CaseSpec invalid JSON: {exc}") from exc
        return cls.from_dict(data)


__all__ = [
    "Answerability",
    "CaseSpec",
    "ConversationType",
    "Difficulty",
]
