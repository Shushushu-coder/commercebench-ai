"""Tests for the Phase 1A Document and Corpus contracts."""

from __future__ import annotations

import pytest

from commercebench.contracts import ContractValidationError
from commercebench.rag import Corpus, Document


def _doc(document_id: str = "doc-1", text: str = "hello world") -> Document:
    return Document(document_id=document_id, text=text, metadata={"k": "v"})


class TestDocument:
    def test_valid(self):
        doc = _doc()
        assert doc.document_id == "doc-1"
        assert doc.text == "hello world"
        assert doc.metadata == {"k": "v"}

    def test_metadata_defaults_empty(self):
        assert Document(document_id="d", text="t").metadata == {}

    def test_empty_id_rejected(self):
        with pytest.raises(ContractValidationError):
            Document(document_id="", text="hello")

    def test_empty_text_rejected(self):
        with pytest.raises(ContractValidationError):
            Document(document_id="d", text="")

    def test_non_json_metadata_rejected(self):
        with pytest.raises(ContractValidationError):
            Document(
                document_id="d", text="t", metadata={"bad": object()}
            )

    def test_json_round_trip(self):
        doc = _doc("doc-9", "some text")
        assert Document.from_json(doc.to_json()) == doc


class TestCorpus:
    def test_valid_corpus(self, rag_corpus):
        assert rag_corpus.corpus_id == "commercebench-dev-corpus"
        assert rag_corpus.corpus_version == "0.1"
        assert len(rag_corpus.documents) == 10

    def test_duplicate_ids_rejected(self):
        with pytest.raises(ContractValidationError, match="duplicate"):
            Corpus(
                corpus_id="c",
                corpus_version="1",
                documents=(_doc("dup", "a"), _doc("dup", "b")),
            )

    def test_empty_corpus_id_rejected(self):
        with pytest.raises(ContractValidationError):
            Corpus(corpus_id="", corpus_version="1", documents=(_doc(),))

    def test_accepts_document_dicts(self):
        corpus = Corpus(
            corpus_id="c",
            corpus_version="1",
            documents=[{"document_id": "d", "text": "t"}],
        )
        assert corpus.documents[0].document_id == "d"

    def test_get_lookup(self, rag_corpus):
        doc = rag_corpus.get("policy-return-001")
        assert doc is not None and doc.document_id == "policy-return-001"
        assert rag_corpus.get("missing") is None

    def test_json_round_trip(self, rag_corpus):
        assert Corpus.from_json(rag_corpus.to_json()) == rag_corpus
