"""Dense embedding retriever — ``dense-minilm-l6-v2-cosine-v0`` (Phase 1E).

This is the first dense retrieval baseline for CommerceBench AI. It
scores documents by cosine similarity over embeddings produced by the
pinned ``sentence-transformers`` model
``sentence-transformers/all-MiniLM-L6-v2`` at the immutable revision
``8b3219a92973c328a8e22fadcfa821b5dc75636a`` (output dimension 384).

Architecture boundary:

- ``EmbeddingEncoder`` — a minimal protocol isolating model loading,
  revision pinning, normalization, and output validation from the
  retrieval algorithm.
- ``SentenceTransformerEncoder`` — the only backend implementation for
  this phase. It imports the heavy ML dependency lazily inside
  ``_load_model`` so importing this module never touches torch or the
  HuggingFace Hub, and it only loads on the first ``encode`` call.
- ``DenseRetriever`` — satisfies the existing ``Retriever`` protocol:
  ``retrieve(query, corpus, top_k) -> RetrievalResult`` with score
  descending / ``document_id`` ascending ordering and higher-score-is-
  better semantics.

Scoring policy: cosine similarity in ``[-1, 1]``; unlike the sparse
baselines there is no ``score > 0`` filter — the top-``top_k`` documents
are returned for any non-empty corpus, including zero or negative
scores.

No vector database, index, hybrid fusion, or reranking exists here: the
~10-document development corpus is encoded in memory on every corpus
change and compared exhaustively.
"""

from __future__ import annotations

import math
import re
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)

from commercebench.contracts.common import (
    require_bool,
    require_int,
    require_non_empty_str,
)
from commercebench.contracts.errors import ContractValidationError

from .corpus import Corpus
from .retrievers import (
    RETRIEVER_REGISTRY,
    RetrievalResult,
    _finite_float,
    _top_ranked,
    _validate_args,
)

_REVISION_PIN = re.compile(r"[0-9a-f]{40}")


@runtime_checkable
class EmbeddingEncoder(Protocol):
    """Minimal embedding boundary between the retriever and a backend.

    ``dimension`` is the encoder's declared output width and ``encode``
    maps texts to equal-length float vectors. Implementations must
    guarantee finite values; callers may still re-validate defensively.
    """

    dimension: int

    def encode(self, texts: Sequence[str]) -> List[List[float]]:
        """Encode ``texts`` into vectors of length ``dimension``."""
        ...


class SentenceTransformerEncoder:
    """``EmbeddingEncoder`` backed by ``sentence-transformers``.

    The configured ``backend_version`` is checked against the installed
    ``sentence_transformers.__version__`` at load time, and the pinned
    immutable ``model_revision`` is passed to ``SentenceTransformer``
    verbatim — this encoder never resolves a moving branch name such as
    ``main``. When ``normalize_embeddings`` is set the backend's own
    ``encode(normalize_embeddings=...)`` flag is used and returned
    vectors are verified to be unit-norm within a loose tolerance.

    ``device`` is execution-only configuration: it selects where the
    model runs and never enters experiment identity.
    """

    _UNIT_NORM_TOLERANCE = 1e-3

    def __init__(
        self,
        *,
        backend_name: str,
        backend_version: str,
        model_name: str,
        model_revision: str,
        expected_dimension: int,
        normalize_embeddings: bool,
        device: Optional[str] = None,
    ) -> None:
        self._backend_name = require_non_empty_str(
            backend_name, "backend_name"
        )
        if self._backend_name != "sentence-transformers":
            raise ContractValidationError(
                f"unsupported embedding backend {self._backend_name!r}"
            )
        self._backend_version = require_non_empty_str(
            backend_version, "backend_version"
        )
        self._model_name = require_non_empty_str(model_name, "model_name")
        self._model_revision = require_non_empty_str(
            model_revision, "model_revision"
        )
        self._expected_dimension = require_int(
            expected_dimension, "expected_dimension"
        )
        if self._expected_dimension <= 0:
            raise ContractValidationError(
                "expected_dimension must be a positive integer"
            )
        self._normalize = require_bool(
            normalize_embeddings, "normalize_embeddings"
        )
        if device is not None:
            device = require_non_empty_str(device, "device")
        self._device = device
        self._model: Any = None
        self.dimension = self._expected_dimension

    @property
    def device(self) -> Optional[str]:
        """The runtime device once the model is loaded, else ``None``."""
        if self._model is None:
            return None
        return str(self._model.device)

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            import sentence_transformers
        except ImportError as exc:
            raise ContractValidationError(
                "sentence-transformers is required for the dense "
                "retriever but is not installed"
            ) from exc
        if sentence_transformers.__version__ != self._backend_version:
            raise ContractValidationError(
                f"embedding_backend_version {self._backend_version!r} "
                f"does not match installed sentence-transformers "
                f"{sentence_transformers.__version__!r}"
            )
        model = sentence_transformers.SentenceTransformer(
            self._model_name,
            revision=self._model_revision,
            device=self._device,
        )
        actual_dimension = model.get_embedding_dimension()
        if actual_dimension != self._expected_dimension:
            raise ContractValidationError(
                f"model {self._model_name!r} reports embedding dimension "
                f"{actual_dimension}, expected {self._expected_dimension}"
            )
        self._model = model
        return model

    def encode(self, texts: Sequence[str]) -> List[List[float]]:
        texts = list(texts)
        if not texts:
            return []
        model = self._load_model()
        output = model.encode(
            texts,
            normalize_embeddings=self._normalize,
            convert_to_numpy=True,
        )
        vectors = [[float(value) for value in row] for row in output]
        self._validate_vectors(vectors)
        return vectors

    def _validate_vectors(self, vectors: Sequence[Sequence[float]]) -> None:
        for index, row in enumerate(vectors):
            if len(row) != self._expected_dimension:
                raise ContractValidationError(
                    f"embedding[{index}] has dimension {len(row)}, "
                    f"expected {self._expected_dimension}"
                )
            squared = 0.0
            for value in row:
                if not math.isfinite(value):
                    raise ContractValidationError(
                        f"embedding[{index}] contains a non-finite value"
                    )
                squared += value * value
            norm = math.sqrt(squared)
            if norm == 0.0:
                raise ContractValidationError(
                    f"embedding[{index}] has zero norm"
                )
            if (
                self._normalize
                and abs(norm - 1.0) > self._UNIT_NORM_TOLERANCE
            ):
                raise ContractValidationError(
                    f"embedding[{index}] has norm {norm} although "
                    "normalize_embeddings=True"
                )


EncoderFactory = Callable[..., EmbeddingEncoder]


def _cosine_similarity(
    a: Sequence[float], b: Sequence[float]
) -> float:
    """True cosine similarity; rejects zero norms and non-finite scores."""
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        raise ContractValidationError(
            "cosine similarity is undefined for a zero-norm embedding"
        )
    score = sum(x * y for x, y in zip(a, b)) / (norm_a * norm_b)
    if not math.isfinite(score):
        raise ContractValidationError(
            "cosine similarity produced a non-finite score"
        )
    return score


class DenseRetriever:
    """Dense retriever ``dense-minilm-l6-v2-cosine-v0`` (Phase 1E).

    Queries and documents are encoded with the same pinned embedding
    model (no query/document prompt asymmetry for MiniLM) and ranked by
    cosine similarity. Because ``normalize_embeddings=True`` yields
    unit-norm vectors the cosine reduces to a dot product; the
    implementation still evaluates the full cosine formula so semantics
    stay correct for any encoder output.

    Every embedding and every returned score is validated finite; NaN,
    infinities, dimension mismatches, and zero-norm vectors raise
    ``ContractValidationError`` instead of leaking into a
    ``RetrievalResult`` or ``RunTrace``.

    Corpus document vectors are cached per retriever instance under a
    key that binds ``corpus_id``, ``corpus_version``, ``model_name``,
    ``model_revision``, and ``normalize_embeddings``.
    """

    retriever_id = "dense-minilm-l6-v2-cosine-v0"
    retriever_version = "0.1"
    algorithm_parameters: ClassVar[Tuple[str, ...]] = (
        "embedding_backend",
        "embedding_backend_version",
        "model_name",
        "model_revision",
        "similarity",
        "normalize_embeddings",
        "expected_dimension",
    )

    DEFAULT_EMBEDDING_BACKEND = "sentence-transformers"
    DEFAULT_EMBEDDING_BACKEND_VERSION = "6.1.0"
    DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    DEFAULT_MODEL_REVISION = "8b3219a92973c328a8e22fadcfa821b5dc75636a"
    DEFAULT_SIMILARITY = "cosine"
    DEFAULT_NORMALIZE_EMBEDDINGS = True
    DEFAULT_EXPECTED_DIMENSION = 384

    _SUPPORTED_EMBEDDING_BACKENDS = frozenset({"sentence-transformers"})
    _SUPPORTED_SIMILARITIES = frozenset({"cosine"})

    def __init__(
        self,
        embedding_backend: str = DEFAULT_EMBEDDING_BACKEND,
        embedding_backend_version: str = DEFAULT_EMBEDDING_BACKEND_VERSION,
        model_name: str = DEFAULT_MODEL_NAME,
        model_revision: str = DEFAULT_MODEL_REVISION,
        similarity: str = DEFAULT_SIMILARITY,
        normalize_embeddings: bool = DEFAULT_NORMALIZE_EMBEDDINGS,
        expected_dimension: int = DEFAULT_EXPECTED_DIMENSION,
        *,
        encoder_factory: Optional[EncoderFactory] = None,
    ) -> None:
        self.embedding_backend = require_non_empty_str(
            embedding_backend, "embedding_backend"
        )
        if self.embedding_backend not in self._SUPPORTED_EMBEDDING_BACKENDS:
            raise ContractValidationError(
                f"unsupported embedding_backend {self.embedding_backend!r}"
            )
        self.embedding_backend_version = require_non_empty_str(
            embedding_backend_version, "embedding_backend_version"
        )
        self.model_name = require_non_empty_str(model_name, "model_name")
        self.model_revision = require_non_empty_str(
            model_revision, "model_revision"
        )
        if _REVISION_PIN.fullmatch(self.model_revision) is None:
            raise ContractValidationError(
                "model_revision must be an immutable 40-hex commit SHA; "
                "moving references such as 'main' are not allowed"
            )
        self.similarity = require_non_empty_str(similarity, "similarity")
        if self.similarity not in self._SUPPORTED_SIMILARITIES:
            raise ContractValidationError(
                f"unsupported similarity {self.similarity!r}"
            )
        self.normalize_embeddings = require_bool(
            normalize_embeddings, "normalize_embeddings"
        )
        self.expected_dimension = require_int(
            expected_dimension, "expected_dimension"
        )
        if self.expected_dimension <= 0:
            raise ContractValidationError(
                "expected_dimension must be a positive integer"
            )
        factory = encoder_factory or SentenceTransformerEncoder
        self._encoder = factory(
            backend_name=self.embedding_backend,
            backend_version=self.embedding_backend_version,
            model_name=self.model_name,
            model_revision=self.model_revision,
            expected_dimension=self.expected_dimension,
            normalize_embeddings=self.normalize_embeddings,
        )
        if getattr(self._encoder, "dimension", None) != (
            self.expected_dimension
        ):
            raise ContractValidationError(
                "encoder dimension does not match expected_dimension "
                f"{self.expected_dimension}"
            )
        self._cache_key: Optional[Tuple[Any, ...]] = None
        self._cache_vectors: Optional[Tuple[Tuple[float, ...], ...]] = None

    def retrieve(
        self,
        query: str,
        corpus: Corpus,
        top_k: int,
    ) -> RetrievalResult:
        query, top_k = _validate_args(query, corpus, top_k)
        if not corpus.documents:
            return RetrievalResult(query=query)
        document_vectors = self._document_vectors(corpus)
        query_vector = self._encode_texts((query,))[0]
        scored = [
            (
                document.document_id,
                _cosine_similarity(query_vector, vector),
            )
            for document, vector in zip(corpus.documents, document_vectors)
        ]
        document_ids, scores = _top_ranked(scored, top_k)
        return RetrievalResult(
            query=query, document_ids=document_ids, scores=scores
        )

    def runtime_metadata(self) -> Dict[str, Any]:
        """Dense identity for ``RAGRunResult.runtime_metadata``.

        ``device`` is recorded only once the backend reports it (i.e.
        after the first ``retrieve``); it is diagnostics, not part of
        experiment identity.
        """
        metadata: Dict[str, Any] = {
            "embedding_backend": self.embedding_backend,
            "embedding_backend_version": self.embedding_backend_version,
            "model_name": self.model_name,
            "model_revision": self.model_revision,
            "similarity": self.similarity,
            "normalize_embeddings": self.normalize_embeddings,
            "embedding_dimension": self.expected_dimension,
        }
        device = getattr(self._encoder, "device", None)
        if device is not None:
            metadata["device"] = device
        return metadata

    def _document_vectors(
        self, corpus: Corpus
    ) -> Tuple[Tuple[float, ...], ...]:
        key = (
            corpus.corpus_id,
            corpus.corpus_version,
            self.model_name,
            self.model_revision,
            self.normalize_embeddings,
        )
        if key != self._cache_key or self._cache_vectors is None:
            self._cache_vectors = self._encode_texts(
                tuple(document.text for document in corpus.documents)
            )
            self._cache_key = key
        return self._cache_vectors

    def _encode_texts(
        self, texts: Sequence[str]
    ) -> Tuple[Tuple[float, ...], ...]:
        vectors = []
        for index, row in enumerate(self._encoder.encode(list(texts))):
            if len(row) != self.expected_dimension:
                raise ContractValidationError(
                    f"embedding[{index}] has dimension {len(row)}, "
                    f"expected {self.expected_dimension}"
                )
            vectors.append(
                tuple(
                    _finite_float(value, f"embedding[{index}]")
                    for value in row
                )
            )
        if len(vectors) != len(texts):
            raise ContractValidationError(
                f"encoder returned {len(vectors)} embeddings for "
                f"{len(texts)} texts"
            )
        return tuple(vectors)


# Self-registration keeps DenseRetriever out of retrievers.py's import
# graph (dense.py depends on it, not the reverse) while build_retriever
# stays the single factory for every retriever id.
RETRIEVER_REGISTRY[DenseRetriever.retriever_id] = DenseRetriever

__all__ = [
    "DenseRetriever",
    "EmbeddingEncoder",
    "EncoderFactory",
    "SentenceTransformerEncoder",
]
