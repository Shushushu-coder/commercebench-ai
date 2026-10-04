"""Corpus: an immutable, versioned set of Documents for Phase 1A.

This is Phase 1A development infrastructure. The corpus carries its own
``corpus_id``/``corpus_version`` so runs can record corpus identity in
trace/runtime metadata without extending the Phase 0 ExperimentManifest
contract.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from commercebench.contracts.common import (
    dumps_dict,
    expect_mapping,
    require_non_empty_str,
)
from commercebench.contracts.errors import ContractValidationError

from .documents import Document


@dataclass(frozen=True)
class Corpus:
    """A named, versioned document collection. Document IDs must be unique."""

    corpus_id: str
    corpus_version: str
    documents: Tuple[Document, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "corpus_id", require_non_empty_str(self.corpus_id, "corpus_id")
        )
        object.__setattr__(
            self,
            "corpus_version",
            require_non_empty_str(self.corpus_version, "corpus_version"),
        )
        if not isinstance(self.documents, (list, tuple)):
            raise ContractValidationError("documents must be a list")
        documents = []
        seen = set()
        for index, item in enumerate(self.documents):
            if not isinstance(item, Document):
                try:
                    item = Document.from_dict(item)
                except ContractValidationError:
                    raise
                except Exception as exc:
                    raise ContractValidationError(
                        f"documents[{index}] is invalid: {exc}"
                    ) from exc
            if item.document_id in seen:
                raise ContractValidationError(
                    f"duplicate document_id: {item.document_id!r}"
                )
            seen.add(item.document_id)
            documents.append(item)
        object.__setattr__(self, "documents", tuple(documents))

    def get(self, document_id: str) -> Optional[Document]:
        """Return the document with ``document_id``, or None."""
        for document in self.documents:
            if document.document_id == document_id:
                return document
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "corpus_id": self.corpus_id,
            "corpus_version": self.corpus_version,
            "documents": [d.to_dict() for d in self.documents],
        }

    @classmethod
    def from_dict(cls, data: Any) -> "Corpus":
        data = expect_mapping(data, "Corpus")
        try:
            return cls(
                corpus_id=data["corpus_id"],
                corpus_version=data["corpus_version"],
                documents=data.get("documents", ()),
            )
        except KeyError as exc:
            raise ContractValidationError(
                f"Corpus missing required field: {exc.args[0]}"
            ) from exc

    def to_json(self) -> str:
        return dumps_dict(self.to_dict())

    @classmethod
    def from_json(cls, payload: str) -> "Corpus":
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ContractValidationError(
                f"Corpus invalid JSON: {exc}"
            ) from exc
        return cls.from_dict(data)


__all__ = ["Corpus"]
