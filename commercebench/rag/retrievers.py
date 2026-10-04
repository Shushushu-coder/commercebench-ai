"""Deterministic, offline, stdlib-only retrievers (Phase 1A + Phase 1C).

Three baseline configurations exist so controlled comparison is possible:

- ``keyword-match-v0``: score = number of distinct normalized query tokens
  present in the document.
- ``term-frequency-v0``: score = sum of the in-document occurrence counts
  of each distinct normalized query term.
- ``bm25-okapi-v0``: Okapi BM25 sparse retrieval with ``k1``/``b``
  algorithm parameters (Phase 1C; the formula is frozen in the
  ``BM25Retriever`` docstring).

None of these is dense retrieval or hybrid retrieval. Text normalization
is lowercase + strip + whitespace collapse; tokenization is a small regex
over ASCII alphanumeric runs — a Phase 1 development tokenizer for the
English synthetic fixtures, not multilingual support.

Ranking contract for every retriever:

    ``document_ids[i]`` is the rank ``i + 1`` result and ``scores[i]`` is
    the aligned score for that document.

Ordering is fully deterministic: score descending, then ``document_id``
ascending. Each current baseline only returns documents scoring > 0;
that is a per-retriever baseline policy, not a Retriever protocol
requirement.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import (
    Any,
    ClassVar,
    Dict,
    FrozenSet,
    List,
    Protocol,
    Sequence,
    Tuple,
    Type,
    runtime_checkable,
)

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


def _finite_float(value: Any, name: str) -> float:
    """Coerce to float, rejecting bools, non-numerics, NaN and infinities."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractValidationError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ContractValidationError(f"{name} must be finite")
    return result


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
    algorithm_parameters: ClassVar[Tuple[str, ...]] = ()

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
    algorithm_parameters: ClassVar[Tuple[str, ...]] = ()

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


class BM25Retriever:
    """Okapi BM25 sparse retriever — ``bm25-okapi-v0`` (Phase 1C baseline).

    The scoring formula is frozen for this retriever version::

        score(D, Q) = Σ  IDF(q) * f(q,D) * (k1 + 1)
                      ────────────────────────────────────────────
                      q∈Q    f(q,D) + k1 * (1 - b + b * |D| / avgdl)

        IDF(q) = ln(1 + (N - n(q) + 0.5) / (n(q) + 0.5))

    where ``Q`` is the set of distinct normalized query tokens, ``N`` the
    corpus document count, ``n(q)`` the number of documents containing
    ``q``, ``f(q,D)`` the in-document frequency of ``q``, ``|D|`` the
    document token count, and ``avgdl`` the corpus average document
    length. The IDF variant is positive for every in-corpus term. Corpus
    statistics are recomputed on every ``retrieve`` call — there is no
    index or cache.

    ``k1`` (term-frequency saturation) must be > 0 and ``b`` (document
    length normalization) must satisfy 0 <= b <= 1; booleans, NaN,
    infinities, and non-numeric values are rejected. The defaults
    (``k1 = 1.5``, ``b = 0.75``) are defined exactly once as
    ``DEFAULT_K1`` / ``DEFAULT_B``.

    Scoring policy: only documents with score > 0 are returned. This is
    the BM25 baseline retrieval policy for Phase 1C, not a generic
    Retriever protocol constraint.
    """

    retriever_id = "bm25-okapi-v0"
    retriever_version = "0.1"
    algorithm_parameters: ClassVar[Tuple[str, ...]] = ("k1", "b")

    DEFAULT_K1 = 1.5
    DEFAULT_B = 0.75

    def __init__(self, k1: float = DEFAULT_K1, b: float = DEFAULT_B) -> None:
        k1 = _finite_float(k1, "k1")
        if k1 <= 0:
            raise ContractValidationError("k1 must be > 0")
        b = _finite_float(b, "b")
        if not 0.0 <= b <= 1.0:
            raise ContractValidationError("b must satisfy 0 <= b <= 1")
        self.k1 = k1
        self.b = b

    def retrieve(
        self,
        query: str,
        corpus: Corpus,
        top_k: int,
    ) -> RetrievalResult:
        query, top_k = _validate_args(query, corpus, top_k)
        query_terms = set(tokenize(query))
        if not query_terms or not corpus.documents:
            return RetrievalResult(query=query)

        document_tokens = [tokenize(d.text) for d in corpus.documents]
        lengths = [len(tokens) for tokens in document_tokens]
        avgdl = sum(lengths) / len(document_tokens)
        document_count = len(document_tokens)

        document_frequencies = {term: 0 for term in query_terms}
        term_frequencies = []
        for tokens in document_tokens:
            counts = Counter(tokens)
            term_frequencies.append(counts)
            for term in query_terms:
                if counts.get(term):
                    document_frequencies[term] += 1

        idfs = {
            term: math.log(
                1.0
                + (document_count - document_frequencies[term] + 0.5)
                / (document_frequencies[term] + 0.5)
            )
            for term in query_terms
        }

        scored = []
        for document, counts, length in zip(
            corpus.documents, term_frequencies, lengths
        ):
            score = 0.0
            for term in sorted(query_terms):
                frequency = counts.get(term, 0)
                if not frequency:
                    continue
                length_ratio = length / avgdl if avgdl > 0 else 0.0
                denominator = frequency + self.k1 * (
                    1.0 - self.b + self.b * length_ratio
                )
                score += idfs[term] * frequency * (self.k1 + 1.0) / denominator
            if score > 0:
                scored.append((document.document_id, score))
        document_ids, scores = _top_ranked(scored, top_k)
        return RetrievalResult(
            query=query, document_ids=document_ids, scores=scores
        )


# RetrievalConfig.parameters keys owned by the pipeline layer
# (DeterministicRAGSystem.from_manifest), not by any retriever. They are
# filtered out before retriever algorithm parameters are validated and
# applied by build_retriever.
PIPELINE_PARAMETER_NAMES: FrozenSet[str] = frozenset(
    {"top_k", "corpus_id", "corpus_version"}
)


RETRIEVER_REGISTRY: Dict[str, Type[Retriever]] = {
    KeywordMatchRetriever.retriever_id: KeywordMatchRetriever,
    TermFrequencyRetriever.retriever_id: TermFrequencyRetriever,
    BM25Retriever.retriever_id: BM25Retriever,
}


def build_retriever(config: RetrievalConfig) -> Retriever:
    """Instantiate the retriever named by a manifest RetrievalConfig.

    ``config.parameters`` is split at this factory boundary:
    pipeline-level keys (``PIPELINE_PARAMETER_NAMES``) are owned by the
    system/pipeline layer and filtered out here; every remaining key is
    treated as a retriever algorithm parameter and must be declared by the
    retriever's ``algorithm_parameters``. Undeclared algorithm keys (for
    example a ``k11`` typo) are rejected instead of being silently
    ignored, so declared parameters always reach the runtime retriever.
    """
    retriever_cls = RETRIEVER_REGISTRY.get(config.retriever_id)
    if retriever_cls is None:
        allowed = ", ".join(sorted(RETRIEVER_REGISTRY))
        raise ContractValidationError(
            f"unknown retriever_id {config.retriever_id!r} "
            f"(available: {allowed})"
        )
    algorithm_parameters = {
        name: value
        for name, value in config.parameters.items()
        if name not in PIPELINE_PARAMETER_NAMES
    }
    declared = set(getattr(retriever_cls, "algorithm_parameters", ()))
    unknown = sorted(set(algorithm_parameters) - declared)
    if unknown:
        raise ContractValidationError(
            f"retriever {config.retriever_id!r} does not declare "
            f"algorithm parameter(s): {', '.join(unknown)}"
        )
    return retriever_cls(**algorithm_parameters)


__all__ = [
    "BM25Retriever",
    "KeywordMatchRetriever",
    "PIPELINE_PARAMETER_NAMES",
    "RETRIEVER_REGISTRY",
    "RetrievalResult",
    "Retriever",
    "TermFrequencyRetriever",
    "build_retriever",
    "normalize_text",
    "tokenize",
]
