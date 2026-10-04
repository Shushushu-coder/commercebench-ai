"""Deterministic, offline, stdlib-only retrievers for Phase 1A.

Two baseline configurations exist so controlled ablation is possible:

- ``keyword-match-v0``: score = number of distinct normalized query tokens
  present in the document.
- ``term-frequency-v0``: score = sum of the in-document occurrence counts
  of each distinct normalized query term.

Neither is BM25, dense retrieval, or hybrid retrieval. Text normalization
is lowercase + strip + whitespace collapse; tokenization is a small regex
over ASCII alphanumeric runs (the Phase 1A fixtures are English text).

Ranking contract for every retriever:

    ``document_ids[i]`` is the rank ``i + 1`` result and ``scores[i]`` is
    the aligned score for that document.

Documents with score 0 are never returned. Ordering is fully
deterministic: score descending, then ``document_id`` ascending.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Protocol, Sequence, Tuple, Type, runtime_checkable

from commercebench.contracts.common import (
    require_int,
    require_non_empty_str,
    str_tuple,
)
from commercebench.contracts.errors import ContractValidationError
from commercebench.contracts.experiment import RetrievalConfig

from .corpus import Corpus

_WHITESPACE = re.compile(r"\s+")
_TOKEN = re.compile(r"[a-z0-9]+")


def normalize_text(text: str) -> str:
    """Lowercase, strip, and collapse internal whitespace."""
    return _WHITESPACE.sub(" ", text.strip().lower())


def tokenize(text: str) -> List[str]:
    """Split normalized text into ASCII alphanumeric tokens (stdlib only)."""
    return _TOKEN.findall(normalize_text(text))


@dataclass(frozen=True)
class RetrievalResult:
    """Ranked retrieval output.

    ``document_ids[i]`` is the rank ``i + 1`` result for the query and
    ``scores[i]`` is the score of that document. Both tuples are equal in
    length and aligned by index.
    """

    query: str
    document_ids: Tuple[str, ...] = ()
    scores: Tuple[float, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "query", require_non_empty_str(self.query, "query")
        )
        object.__setattr__(
            self,
            "document_ids",
            str_tuple(self.document_ids, "document_ids"),
        )
        scores = tuple(self.scores)
        for index, score in enumerate(scores):
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise ContractValidationError(
                    f"scores[{index}] must be a number"
                )
        object.__setattr__(self, "scores", tuple(float(s) for s in scores))
        if len(self.document_ids) != len(self.scores):
            raise ContractValidationError(
                "document_ids and scores must have the same length"
            )


@runtime_checkable
class Retriever(Protocol):
    """Minimal retrieval interface: query + corpus -> ranked result."""

    retriever_id: str
    retriever_version: str

    def retrieve(
        self,
        query: str,
        corpus: Corpus,
        top_k: int,
    ) -> RetrievalResult:
        """Return the top-``top_k`` ranked documents for ``query``."""
        ...


def _top_ranked(
    scored: Sequence[Tuple[str, float]],
    top_k: int,
) -> Tuple[Tuple[str, ...], Tuple[float, ...]]:
    """Deterministic ranking: score descending, then document_id ascending."""
    ordered = sorted(scored, key=lambda item: (-item[1], item[0]))
    top = ordered[:top_k]
    return (
        tuple(document_id for document_id, _ in top),
        tuple(score for _, score in top),
    )


def _validate_args(query: str, corpus: Corpus, top_k: int) -> Tuple[str, int]:
    query = require_non_empty_str(query, "query")
    if not isinstance(corpus, Corpus):
        raise ContractValidationError("corpus must be a Corpus")
    top_k = require_int(top_k, "top_k")
    if top_k <= 0:
        raise ContractValidationError("top_k must be a positive integer")
    return query, top_k


class KeywordMatchRetriever:
    """Score = count of distinct normalized query tokens present in a doc."""

    retriever_id = "keyword-match-v0"
    retriever_version = "0.1"

    def retrieve(
        self,
        query: str,
        corpus: Corpus,
        top_k: int,
    ) -> RetrievalResult:
        query, top_k = _validate_args(query, corpus, top_k)
        query_terms = set(tokenize(query))
        scored = []
        for document in corpus.documents:
            score = float(len(query_terms & set(tokenize(document.text))))
            if score > 0:
                scored.append((document.document_id, score))
        document_ids, scores = _top_ranked(scored, top_k)
        return RetrievalResult(
            query=query, document_ids=document_ids, scores=scores
        )


class TermFrequencyRetriever:
    """Score = sum of in-document occurrence counts of distinct query terms."""

    retriever_id = "term-frequency-v0"
    retriever_version = "0.1"

    def retrieve(
        self,
        query: str,
        corpus: Corpus,
        top_k: int,
    ) -> RetrievalResult:
        query, top_k = _validate_args(query, corpus, top_k)
        query_terms = set(tokenize(query))
        scored = []
        for document in corpus.documents:
            document_tokens = tokenize(document.text)
            score = float(
                sum(document_tokens.count(term) for term in query_terms)
            )
            if score > 0:
                scored.append((document.document_id, score))
        document_ids, scores = _top_ranked(scored, top_k)
        return RetrievalResult(
            query=query, document_ids=document_ids, scores=scores
        )


RETRIEVER_REGISTRY: Dict[str, Type[Retriever]] = {
    KeywordMatchRetriever.retriever_id: KeywordMatchRetriever,
    TermFrequencyRetriever.retriever_id: TermFrequencyRetriever,
}


def build_retriever(config: RetrievalConfig) -> Retriever:
    """Instantiate the retriever named by a manifest RetrievalConfig."""
    retriever_cls = RETRIEVER_REGISTRY.get(config.retriever_id)
    if retriever_cls is None:
        allowed = ", ".join(sorted(RETRIEVER_REGISTRY))
        raise ContractValidationError(
            f"unknown retriever_id {config.retriever_id!r} "
            f"(available: {allowed})"
        )
    return retriever_cls()


__all__ = [
    "KeywordMatchRetriever",
    "RETRIEVER_REGISTRY",
    "RetrievalResult",
    "Retriever",
    "TermFrequencyRetriever",
    "build_retriever",
    "normalize_text",
    "tokenize",
]
