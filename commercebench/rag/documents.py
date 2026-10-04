"""Document: the minimal retrievable unit for Phase 1A RAG.

A Document is only its identity, its text, and JSON-compatible metadata.
Scores, ranks, embeddings, and chunk lineage are deliberately absent: they
are retrieval-time artifacts, not properties of the document itself.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict

from commercebench.contracts.common import (
    dumps_dict,
    expect_mapping,
    require_json_dict,
    require_non_empty_str,
)
from commercebench.contracts.errors import ContractValidationError


@dataclass(frozen=True)
class Document:
    """One retrievable document in a corpus."""

    document_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "document_id",
            require_non_empty_str(self.document_id, "document_id"),
        )
        object.__setattr__(
            self, "text", require_non_empty_str(self.text, "text")
        )
        object.__setattr__(
            self, "metadata", dict(require_json_dict(self.metadata, "metadata"))
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "text": self.text,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Any) -> "Document":
        data = expect_mapping(data, "Document")
        try:
            return cls(
                document_id=data["document_id"],
                text=data["text"],
                metadata=data.get("metadata", {}),
            )
        except KeyError as exc:
            raise ContractValidationError(
                f"Document missing required field: {exc.args[0]}"
            ) from exc

    def to_json(self) -> str:
        return dumps_dict(self.to_dict())

    @classmethod
    def from_json(cls, payload: str) -> "Document":
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ContractValidationError(
                f"Document invalid JSON: {exc}"
            ) from exc
        return cls.from_dict(data)


__all__ = ["Document"]
