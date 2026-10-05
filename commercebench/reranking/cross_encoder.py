"""Learned reranker ``cross-encoder-msmarco-minilm-l6-v2-v0`` (Phase 2B).

Fixed baseline identity:

- backend ``sentence-transformers`` ``6.1.0`` (exact pinned dependency).
- model ``cross-encoder/ms-marco-MiniLM-L6-v2`` at the immutable revision
  ``ce0834f22110de6d9222af7a7a03628121708969``.
- ``max_length = 512`` (query+document truncation bound).
- ``score_activation = identity`` (raw relevance logit; higher is better).

The model is loaded lazily on the first ``rerank`` call — never at
module import, manifest parse, or ``build_reranker`` time — through
``sentence_transformers.CrossEncoder`` with the pinned ``revision``,
``max_length``, and an explicit ``Identity`` activation. The configured
backend version is checked against the installed package at load time.
``device``, ``batch_size``, and ``show_progress_bar`` are execution-only
defaults and never enter experiment identity.

Unit tests stay offline by injecting a ``scorer`` callable
``(query, candidates) -> scores`` or a ``model_factory``; the real
``CrossEncoder.predict`` path is exercised only by the ``integration``
suite.
"""

from __future__ import annotations

import math
import re
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

from commercebench.contracts.common import (
    require_int,
    require_non_empty_str,
)
from commercebench.contracts.errors import ContractValidationError

from .contracts import (
    RerankCandidate,
    RerankResult,
    validate_candidate_ids_unique,
    validate_result_against_candidates,
)
from .rerankers import _require_candidate_top_k, _validate_rerank_args

_REVISION_PIN = re.compile(r"[0-9a-f]{40}")

ScorerFn = Callable[[str, Sequence[RerankCandidate]], Sequence[float]]
ModelFactory = Callable[..., Any]


class CrossEncoderReranker:
    """Cross-encoder learned reranker (version ``0.1``).

    Scores each ``(query, document.text)`` pair with the pinned
    cross-encoder and returns the ``top_k`` documents by raw logit
    descending, breaking ties by ``document_id`` ascending. Negative
    scores are retained (relative ordering only); NaN and infinities
    are rejected before they can enter a ``RerankResult`` or trace.
    """

    reranker_id = "cross-encoder-msmarco-minilm-l6-v2-v0"
    reranker_version = "0.1"

    algorithm_parameters: Tuple[str, ...] = (
        "candidate_top_k",
        "backend",
        "backend_version",
        "model_name",
        "model_revision",
        "max_length",
        "score_activation",
    )

    DEFAULT_CANDIDATE_TOP_K = 10
    DEFAULT_BACKEND = "sentence-transformers"
    DEFAULT_BACKEND_VERSION = "6.1.0"
    DEFAULT_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"
    DEFAULT_MODEL_REVISION = "ce0834f22110de6d9222af7a7a03628121708969"
    DEFAULT_MAX_LENGTH = 512
    DEFAULT_SCORE_ACTIVATION = "identity"

    SCORE_SEMANTICS = "cross_encoder_logit"

    _SUPPORTED_BACKENDS = frozenset({"sentence-transformers"})
    _SUPPORTED_SCORE_ACTIVATIONS = frozenset({"identity"})

    # Execution-only defaults (never fingerprinted).
    DEFAULT_BATCH_SIZE = 32

    def __init__(
        self,
        candidate_top_k: int = DEFAULT_CANDIDATE_TOP_K,
        backend: str = DEFAULT_BACKEND,
        backend_version: str = DEFAULT_BACKEND_VERSION,
        model_name: str = DEFAULT_MODEL_NAME,
        model_revision: str = DEFAULT_MODEL_REVISION,
        max_length: int = DEFAULT_MAX_LENGTH,
        score_activation: str = DEFAULT_SCORE_ACTIVATION,
        *,
        device: Optional[str] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        show_progress_bar: bool = False,
        scorer: Optional[ScorerFn] = None,
        model_factory: Optional[ModelFactory] = None,
    ) -> None:
        self.candidate_top_k = _require_candidate_top_k(candidate_top_k)
        self.backend = require_non_empty_str(backend, "backend")
        if self.backend not in self._SUPPORTED_BACKENDS:
            raise ContractValidationError(
                f"unsupported reranker backend {self.backend!r}"
            )
        self.backend_version = require_non_empty_str(
            backend_version, "backend_version"
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
        self.max_length = require_int(max_length, "max_length")
        if self.max_length <= 0:
            raise ContractValidationError(
                "max_length must be a positive integer"
            )
        self.score_activation = require_non_empty_str(
            score_activation, "score_activation"
        )
        if self.score_activation not in self._SUPPORTED_SCORE_ACTIVATIONS:
            raise ContractValidationError(
                f"unsupported score_activation {self.score_activation!r} "
                "(this baseline requires 'identity')"
            )
        if device is not None:
            device = require_non_empty_str(device, "device")
        self._device_requested = device
        self._batch_size = require_int(batch_size, "batch_size")
        if self._batch_size <= 0:
            raise ContractValidationError(
                "batch_size must be a positive integer"
            )
        if not isinstance(show_progress_bar, bool):
            raise ContractValidationError(
                "show_progress_bar must be a boolean"
            )
        self._show_progress_bar = show_progress_bar
        if scorer is not None and not callable(scorer):
            raise ContractValidationError("scorer must be callable")
        if model_factory is not None and not callable(model_factory):
            raise ContractValidationError("model_factory must be callable")
        self._scorer = scorer
        self._model_factory = model_factory
        self._model: Any = None

    @property
    def is_loaded(self) -> bool:
        """Whether the underlying cross-encoder model is loaded."""
        return self._model is not None

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            import sentence_transformers
        except ImportError as exc:
            raise ContractValidationError(
                "sentence-transformers is required for the cross-encoder "
                "reranker but is not installed"
            ) from exc
        if sentence_transformers.__version__ != self.backend_version:
            raise ContractValidationError(
                f"reranker backend_version {self.backend_version!r} "
                f"does not match installed sentence-transformers "
                f"{sentence_transformers.__version__!r}"
            )
        import torch.nn as _torch_nn

        factory = self._model_factory
        if factory is None:
            from sentence_transformers import CrossEncoder as _CrossEncoder

            factory = _CrossEncoder
        # The pinned revision, max_length bound, and Identity activation
        # must reach the real constructor verbatim — manifest metadata
        # alone would not satisfy Phase 2B.
        model = factory(
            self.model_name,
            revision=self.model_revision,
            max_length=self.max_length,
            activation_fn=_torch_nn.Identity(),
            device=self._device_requested,
        )
        self._model = model
        return model

    def _score_with_model(
        self, query: str, candidates: Sequence[RerankCandidate]
    ) -> Tuple[float, ...]:
        import torch.nn as _torch_nn

        model = self._load_model()
        pairs = [(query, candidate.text) for candidate in candidates]
        # Identity activation is enforced at predict time as well, so the
        # raw relevance logit is returned regardless of the model's
        # stored default (ms-marco-MiniLM defaults to sigmoid).
        raw = model.predict(
            pairs,
            batch_size=self._batch_size,
            show_progress_bar=self._show_progress_bar,
            activation_fn=_torch_nn.Identity(),
            convert_to_numpy=True,
        )
        try:
            scores = tuple(float(value) for value in list(raw))
        except (TypeError, ValueError) as exc:
            raise ContractValidationError(
                f"cross-encoder scorer returned an invalid shape: {exc}"
            ) from exc
        if len(scores) != len(candidates):
            raise ContractValidationError(
                f"cross-encoder scorer returned {len(scores)} scores for "
                f"{len(candidates)} candidates"
            )
        return scores

    def _score_candidates(
        self, query: str, candidates: Sequence[RerankCandidate]
    ) -> Tuple[float, ...]:
        if self._scorer is not None:
            raw = self._scorer(query, candidates)
            try:
                scores = tuple(float(value) for value in list(raw))
            except (TypeError, ValueError) as exc:
                raise ContractValidationError(
                    f"reranker scorer returned invalid scores: {exc}"
                ) from exc
            if len(scores) != len(candidates):
                raise ContractValidationError(
                    f"reranker scorer returned {len(scores)} scores for "
                    f"{len(candidates)} candidates"
                )
            return scores
        return self._score_with_model(query, candidates)

    def rerank(
        self,
        query: str,
        candidates: Sequence[RerankCandidate],
        top_k: int,
    ) -> RerankResult:
        query, normalized, top_k = _validate_rerank_args(
            query, candidates, top_k
        )
        if top_k > self.candidate_top_k:
            raise ContractValidationError(
                f"final top_k {top_k} exceeds candidate_top_k "
                f"{self.candidate_top_k}"
            )
        if not normalized:
            return RerankResult(query=query, document_ids=(), scores=())
        validate_candidate_ids_unique(normalized)
        scores = self._score_candidates(query, normalized)
        for index, score in enumerate(scores):
            if not math.isfinite(float(score)):
                raise ContractValidationError(
                    f"cross-encoder score[{index}] must be finite"
                )
        ordered = sorted(
            zip(
                (candidate.document_id for candidate in normalized),
                scores,
            ),
            key=lambda item: (-item[1], item[0]),
        )
        selected = ordered[:top_k]
        document_ids = tuple(document_id for document_id, _ in selected)
        final_scores = tuple(float(score) for _, score in selected)
        validate_result_against_candidates(
            tuple(candidate.document_id for candidate in normalized),
            document_ids,
            final_scores,
            top_k,
        )
        return RerankResult(
            query=query, document_ids=document_ids, scores=final_scores
        )

    def runtime_metadata(self) -> Dict[str, Any]:
        """Reranker diagnostics for ``RunTrace.runtime_metadata``.

        Records the fingerprinted identity plus the score semantics and
        the actual device once loaded. Never records local cache paths,
        usernames, or home directories.
        """
        metadata: Dict[str, Any] = {
            "reranker_id": self.reranker_id,
            "reranker_version": self.reranker_version,
            "candidate_top_k": self.candidate_top_k,
            "backend": self.backend,
            "backend_version": self.backend_version,
            "model_name": self.model_name,
            "model_revision": self.model_revision,
            "max_length": self.max_length,
            "score_activation": self.score_activation,
            "score_semantics": self.SCORE_SEMANTICS,
        }
        device = None
        if self._model is not None:
            device = getattr(self._model, "device", None)
            if device is not None:
                device = str(device)
        if device is None and self._device_requested is not None:
            device = str(self._device_requested)
        if device is not None:
            metadata["device"] = device
        return metadata


__all__ = ["CrossEncoderReranker"]
