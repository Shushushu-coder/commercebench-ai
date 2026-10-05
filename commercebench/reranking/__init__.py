"""Phase 2B reranking primitives: candidates, rerankers, factory.

The reranker reorders or filters the candidate ranking produced by the
retriever and never introduces documents outside the candidate set.
``IdentityReranker`` is the controlled ablation control;
``CrossEncoderReranker`` is the first learned baseline
(``cross-encoder/ms-marco-MiniLM-L6-v2`` at a pinned immutable
revision, ``max_length=512``, identity/raw-logit scoring).
"""

from .contracts import (
    RerankCandidate,
    RerankDiagnostics,
    RerankResult,
    validate_candidate_ids_unique,
    validate_diagnostics_against_candidates,
    validate_result_against_candidates,
)
from .rerankers import (
    IdentityReranker,
    RERANKER_REGISTRY,
    Reranker,
    build_reranker,
)
from .cross_encoder import CrossEncoderReranker

__all__ = [
    "CrossEncoderReranker",
    "IdentityReranker",
    "RERANKER_REGISTRY",
    "RerankCandidate",
    "RerankDiagnostics",
    "RerankResult",
    "Reranker",
    "build_reranker",
    "validate_candidate_ids_unique",
    "validate_diagnostics_against_candidates",
    "validate_result_against_candidates",
]
