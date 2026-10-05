"""Reranker protocol, identity control, and factory (Phase 2B).

Only two reranker identities exist in this phase:

- ``identity-reranker-v0``: controlled ablation control that preserves
  the candidate order and truncates to the final depth. It performs no
  learned scoring; final scores are the candidate retrieval scores.
- ``cross-encoder-msmarco-minilm-l6-v2-v0``: the first learned baseline
  (see ``cross_encoder.py``). It is registered here so ``build_reranker``
  stays the single factory, but its implementation lives separately.

No provider/plugin universe, dynamic imports, or discovery exist.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple, Type

from commercebench.contracts.common import require_int, require_non_empty_str
from commercebench.contracts.errors import ContractValidationError
from commercebench.contracts.experiment import RerankerConfig

from .contracts import (
    RerankCandidate,
    RerankDiagnostics,
    RerankResult,
    validate_candidate_ids_unique,
    validate_result_against_candidates,
)

try:
    from typing import Protocol, runtime_checkable
except ImportError:  # pragma: no cover - Python 3.10 always has typing.Protocol
    from typing_extensions import Protocol, runtime_checkable  # type: ignore


@runtime_checkable
class Reranker(Protocol):
    """Minimal reranking interface: candidates -> reranked result."""

    reranker_id: str
    reranker_version: str
    candidate_top_k: int

    def rerank(
        self,
        query: str,
        candidates: Sequence[RerankCandidate],
        top_k: int,
    ) -> RerankResult:
        """Return the top-``top_k`` reranked documents for ``query``."""
        ...


def _validate_rerank_args(
    query: Any, candidates: Any, top_k: Any
) -> Tuple[str, Tuple[RerankCandidate, ...], int]:
    query = require_non_empty_str(query, "query")
    if not isinstance(candidates, (list, tuple)):
        raise ContractValidationError("candidates must be a list")
    normalized = []
    for index, item in enumerate(candidates):
        if not isinstance(item, RerankCandidate):
            try:
                item = RerankCandidate.from_dict(item)
            except ContractValidationError:
                raise
            except Exception as exc:
                raise ContractValidationError(
                    f"candidates[{index}] is invalid: {exc}"
                ) from exc
        normalized.append(item)
    top_k = require_int(top_k, "top_k")
    if top_k <= 0:
        raise ContractValidationError("top_k must be a positive integer")
    validate_candidate_ids_unique(normalized)
    return query, tuple(normalized), top_k


def _require_candidate_top_k(value: Any) -> int:
    value = require_int(value, "candidate_top_k")
    if value <= 0:
        raise ContractValidationError(
            "candidate_top_k must be a positive integer"
        )
    return value


class IdentityReranker:
    """Controlled control ``identity-reranker-v0`` (version ``0.1``).

    Preserves the candidate order exactly and truncates to the final
    ``top_k``. Final scores are the candidate retrieval scores aligned to
    the preserved order — no rescoring happens here. The full diagnostics
    are the complete candidate ranking itself (all candidate IDs in
    candidate order with all candidate retrieval scores); the final
    ranking is its ``[:top_k]`` prefix.
    """

    reranker_id = "identity-reranker-v0"
    reranker_version = "0.1"

    algorithm_parameters: Tuple[str, ...] = ("candidate_top_k",)

    DEFAULT_CANDIDATE_TOP_K = 10

    def __init__(self, candidate_top_k: int = DEFAULT_CANDIDATE_TOP_K) -> None:
        self.candidate_top_k = _require_candidate_top_k(candidate_top_k)

    def rerank(
        self,
        query: str,
        candidates: Sequence[RerankCandidate],
        top_k: int,
    ) -> RerankResult:
        query, normalized, top_k = _validate_rerank_args(query, candidates, top_k)
        if top_k > self.candidate_top_k:
            raise ContractValidationError(
                f"final top_k {top_k} exceeds candidate_top_k "
                f"{self.candidate_top_k}"
            )
        diagnostics = RerankDiagnostics(
            document_ids=tuple(
                candidate.document_id for candidate in normalized
            ),
            scores=tuple(
                float(candidate.retrieval_score) for candidate in normalized
            ),
        )
        selected = normalized[:top_k]
        document_ids = tuple(candidate.document_id for candidate in selected)
        scores = tuple(float(candidate.retrieval_score) for candidate in selected)
        validate_result_against_candidates(
            tuple(candidate.document_id for candidate in normalized),
            document_ids,
            scores,
            top_k,
        )
        return RerankResult(
            query=query,
            document_ids=document_ids,
            scores=scores,
            diagnostics=diagnostics,
        )

    def runtime_metadata(self) -> Dict[str, Any]:
        """Identity diagnostics for ``RunTrace.runtime_metadata``."""
        return {
            "reranker_id": self.reranker_id,
            "reranker_version": self.reranker_version,
            "candidate_top_k": self.candidate_top_k,
        }


RERANKER_REGISTRY: Dict[str, Type[Reranker]] = {
    IdentityReranker.reranker_id: IdentityReranker,
}


def _register_cross_encoder() -> None:
    # Deferred import keeps ``rerankers`` importable without torch and
    # avoids a hard import cycle: cross_encoder.py depends on this
    # module's helpers, not the reverse at module scope.
    from .cross_encoder import CrossEncoderReranker

    RERANKER_REGISTRY[CrossEncoderReranker.reranker_id] = CrossEncoderReranker


_register_cross_encoder()


def build_reranker(
    config: RerankerConfig,
    **execution_overrides: Any,
) -> Reranker:
    """Instantiate the reranker named by a manifest ``RerankerConfig``.

    ``config.parameters`` holds only result-affecting reranker
    configuration. Execution-only knobs (``device``, ``batch_size``,
    ``show_progress_bar``, cache location, injected scorers) are passed
    as keyword arguments here, never through the manifest. Unknown
    manifest parameters are rejected instead of being silently ignored,
    so declared parameters always reach the runtime reranker. An
    unknown ``reranker_id`` is rejected.
    """
    if not isinstance(config, RerankerConfig):
        try:
            config = RerankerConfig.from_dict(config)
        except ContractValidationError:
            raise
        except Exception as exc:
            raise ContractValidationError(
                f"reranker config is invalid: {exc}"
            ) from exc
    reranker_cls = RERANKER_REGISTRY.get(config.reranker_id)
    if reranker_cls is None:
        allowed = ", ".join(sorted(RERANKER_REGISTRY))
        raise ContractValidationError(
            f"unknown reranker_id {config.reranker_id!r} "
            f"(available: {allowed})"
        )
    declared = set(getattr(reranker_cls, "algorithm_parameters", ()))
    unknown = sorted(set(config.parameters) - declared)
    if unknown:
        raise ContractValidationError(
            f"reranker {config.reranker_id!r} does not declare "
            f"parameter(s): {', '.join(unknown)}"
        )
    expected_version = getattr(reranker_cls, "reranker_version", None)
    if (
        expected_version is not None
        and config.reranker_version != expected_version
    ):
        raise ContractValidationError(
            f"reranker {config.reranker_id!r} version "
            f"{config.reranker_version!r} does not match supported "
            f"version {expected_version!r}"
        )
    try:
        return reranker_cls(
            **dict(config.parameters), **execution_overrides
        )
    except TypeError as exc:
        raise ContractValidationError(
            f"reranker {config.reranker_id!r} parameters invalid: {exc}"
        ) from exc


__all__ = [
    "IdentityReranker",
    "RERANKER_REGISTRY",
    "Reranker",
    "build_reranker",
]
