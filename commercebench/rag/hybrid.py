"""Hybrid retriever — ``hybrid-bm25-minilm-rrf-v0`` (Phase 1G).

This is the first hybrid retrieval baseline for CommerceBench AI. Two
existing component retrievers — ``bm25-okapi-v0`` (sparse) and
``dense-minilm-l6-v2-cosine-v0`` (dense) — run in parallel over the
*same* raw query and corpus, and their rankings are merged with
Reciprocal Rank Fusion::

    RRF(d) = Σ  1 / (rrf_k + rank_i(d))
             i

where ``rank_i(d)`` is the 1-based rank of document ``d`` in component
``i``'s returned list; a document absent from a component receives no
contribution from it. RRF is used because BM25 and cosine raw scores
live on incompatible scales and cannot be summed or weighted
meaningfully — only their *ranks* are fused. Final ordering is score
descending, then ``document_id`` ascending, and ``RetrievalResult``
scores are the RRF fusion scores (higher is better), never raw
component scores.

Configuration semantics:

- ``component_top_k`` is the candidate depth each component retrieves
  into fusion; ``top_k`` (the pipeline-level parameter) is the final
  depth returned to the pipeline. ``component_top_k >= top_k`` is
  required and enforced at the retrieval boundary.
- ``sparse`` / ``dense`` are nested component configurations,
  ``{"retriever_id": ..., "parameters": {...}}``. Both are required and
  must carry every algorithm parameter their component declares — no
  component default is silently applied, so the complete component
  identity enters the manifest fingerprint. Only the two pinned
  component ids are accepted; a nested hybrid is rejected, so no
  recursive hybrid graph can be constructed.
- ``fusion_method`` is frozen to ``"rrf"`` for this baseline version;
  ``rrf_k`` (default 60) and ``component_top_k`` (default 10) are
  declared algorithm parameters that enter the fingerprint and reach
  runtime.

Component rankings are diagnostics only. ``runtime_metadata()``
exposes them under ``runtime_metadata["hybrid"]["components"]`` for
failure analysis; they are never emitted as extra ``RetrievalEvent``s,
so retrieval metrics consume exactly one fused ranking per run.
Component raw scores (BM25 vs cosine) are recorded for diagnosis but
are not comparable across retrievers.
"""

from __future__ import annotations

from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from commercebench.contracts.common import (
    expect_mapping,
    require_int,
    require_non_empty_str,
)
from commercebench.contracts.errors import ContractValidationError
from commercebench.contracts.experiment import RetrievalConfig

from .corpus import Corpus
from .dense import DenseRetriever
from .retrievers import (
    BM25Retriever,
    PIPELINE_PARAMETER_NAMES,
    RETRIEVER_REGISTRY,
    RetrievalResult,
    Retriever,
    _top_ranked,
    _validate_args,
    build_retriever,
)

ComponentFactory = Callable[[RetrievalConfig], Retriever]

_COMPONENT_CONFIG_KEYS = frozenset({"retriever_id", "parameters"})


def _component_config(
    value: Any, slot: str, expected_id: str
) -> RetrievalConfig:
    """Validate one nested component block into a ``RetrievalConfig``.

    The block must be exactly ``{"retriever_id", "parameters"}``, must
    name ``expected_id`` (which also forbids a nested hybrid), must not
    contain pipeline-level keys, and must supply *every* algorithm
    parameter the component declares — a missing key would silently
    fall back to a component default outside the manifest fingerprint.
    """
    data = expect_mapping(value, f"{slot} component")
    unknown = sorted(set(data) - _COMPONENT_CONFIG_KEYS)
    if unknown:
        raise ContractValidationError(
            f"{slot} component has unknown key(s): {', '.join(unknown)}"
        )
    if "retriever_id" not in data:
        raise ContractValidationError(
            f"{slot} component requires 'retriever_id'"
        )
    if "parameters" not in data:
        raise ContractValidationError(
            f"{slot} component requires 'parameters'"
        )
    config = RetrievalConfig(
        retriever_id=data["retriever_id"],
        parameters=data["parameters"],
    )
    if config.retriever_id != expected_id:
        raise ContractValidationError(
            f"{slot} component must be {expected_id!r}, got "
            f"{config.retriever_id!r}; nested hybrid graphs and "
            "substitute components are not supported"
        )
    provided = set(config.parameters)
    pipeline_keys = sorted(provided & PIPELINE_PARAMETER_NAMES)
    if pipeline_keys:
        raise ContractValidationError(
            f"{slot} component parameters must not contain "
            f"pipeline-level key(s): {', '.join(pipeline_keys)}"
        )
    declared = set(
        getattr(RETRIEVER_REGISTRY[expected_id], "algorithm_parameters", ())
    )
    unknown_params = sorted(provided - declared)
    if unknown_params:
        raise ContractValidationError(
            f"{slot} component {expected_id!r} does not declare "
            f"algorithm parameter(s): {', '.join(unknown_params)}"
        )
    missing = sorted(declared - provided)
    if missing:
        raise ContractValidationError(
            f"{slot} component parameters missing declared algorithm "
            f"parameter(s): {', '.join(missing)}"
        )
    return config


def _build_component(
    factory: ComponentFactory, config: RetrievalConfig, slot: str
) -> Retriever:
    component = factory(config)
    if not isinstance(component, Retriever):
        raise ContractValidationError(
            f"{slot} component factory did not return a Retriever"
        )
    if component.retriever_id != config.retriever_id:
        raise ContractValidationError(
            f"{slot} component retriever_id {component.retriever_id!r} "
            f"does not match configured {config.retriever_id!r}"
        )
    return component


def _rrf_scores(
    rankings: Sequence[Tuple[str, ...]], rrf_k: int
) -> List[Tuple[str, float]]:
    """Reciprocal Rank Fusion over component ``document_ids`` orderings.

    Component rank order is authoritative: raw component scores are
    never consulted here. Each component contributes a document at most
    once (first/best rank wins if a component misbehaves and repeats an
    id). Output is sorted by fusion score descending, then
    ``document_id`` ascending — a stable, fully deterministic order.
    """
    fused: Dict[str, float] = {}
    for document_ids in rankings:
        seen = set()
        for rank, document_id in enumerate(document_ids, start=1):
            if document_id in seen:
                continue
            seen.add(document_id)
            fused[document_id] = (
                fused.get(document_id, 0.0) + 1.0 / (rrf_k + rank)
            )
    return sorted(fused.items(), key=lambda item: (-item[1], item[0]))


def _require_component_result(result: Any, slot: str) -> RetrievalResult:
    if not isinstance(result, RetrievalResult):
        raise ContractValidationError(
            f"{slot} component did not return a RetrievalResult"
        )
    return result


class HybridRRFRetriever:
    """``bm25-okapi-v0`` + ``dense-minilm-l6-v2-cosine-v0`` fused by RRF.

    For one ``retrieve`` call the same raw query, the same ``Corpus``
    object, and the same ``component_top_k`` are handed to both
    components — no query rewriting, metadata enrichment, or asymmetric
    treatment. Their ``document_ids`` orderings (each component's own
    ranking authority) are fused by RRF and the fused top-``top_k`` is
    returned with RRF fusion scores.

    The most recent component rankings are retained for
    ``runtime_metadata`` diagnostics; retrieval metrics never see them.
    """

    retriever_id = "hybrid-bm25-minilm-rrf-v0"
    retriever_version = "0.1"
    algorithm_parameters: ClassVar[Tuple[str, ...]] = (
        "sparse",
        "dense",
        "fusion_method",
        "rrf_k",
        "component_top_k",
    )

    SPARSE_COMPONENT_ID = BM25Retriever.retriever_id
    DENSE_COMPONENT_ID = DenseRetriever.retriever_id

    DEFAULT_FUSION_METHOD = "rrf"
    DEFAULT_RRF_K = 60
    DEFAULT_COMPONENT_TOP_K = 10

    _SUPPORTED_FUSION_METHODS = frozenset({"rrf"})

    def __init__(
        self,
        sparse: Optional[Mapping[str, Any]] = None,
        dense: Optional[Mapping[str, Any]] = None,
        fusion_method: str = DEFAULT_FUSION_METHOD,
        rrf_k: int = DEFAULT_RRF_K,
        component_top_k: int = DEFAULT_COMPONENT_TOP_K,
        *,
        component_factory: Optional[ComponentFactory] = None,
    ) -> None:
        self.fusion_method = require_non_empty_str(
            fusion_method, "fusion_method"
        )
        if self.fusion_method not in self._SUPPORTED_FUSION_METHODS:
            raise ContractValidationError(
                f"unsupported fusion_method {self.fusion_method!r}"
            )
        self.rrf_k = require_int(rrf_k, "rrf_k")
        if self.rrf_k <= 0:
            raise ContractValidationError(
                "rrf_k must be a positive integer"
            )
        self.component_top_k = require_int(
            component_top_k, "component_top_k"
        )
        if self.component_top_k <= 0:
            raise ContractValidationError(
                "component_top_k must be a positive integer"
            )
        self._sparse_config = _component_config(
            sparse, "sparse", self.SPARSE_COMPONENT_ID
        )
        self._dense_config = _component_config(
            dense, "dense", self.DENSE_COMPONENT_ID
        )
        factory = component_factory or build_retriever
        self._sparse = _build_component(factory, self._sparse_config, "sparse")
        self._dense = _build_component(factory, self._dense_config, "dense")
        self._last_top_k: Optional[int] = None
        self._last_sparse_result: Optional[RetrievalResult] = None
        self._last_dense_result: Optional[RetrievalResult] = None

    def retrieve(
        self,
        query: str,
        corpus: Corpus,
        top_k: int,
    ) -> RetrievalResult:
        query, top_k = _validate_args(query, corpus, top_k)
        if top_k > self.component_top_k:
            raise ContractValidationError(
                f"top_k {top_k} exceeds component_top_k "
                f"{self.component_top_k}: the fusion candidate pool "
                "must be at least as deep as the final result depth"
            )
        sparse_result = _require_component_result(
            self._sparse.retrieve(query, corpus, self.component_top_k),
            "sparse",
        )
        dense_result = _require_component_result(
            self._dense.retrieve(query, corpus, self.component_top_k),
            "dense",
        )
        self._last_top_k = top_k
        self._last_sparse_result = sparse_result
        self._last_dense_result = dense_result
        fused = _rrf_scores(
            (sparse_result.document_ids, dense_result.document_ids),
            self.rrf_k,
        )
        document_ids, scores = _top_ranked(fused, top_k)
        return RetrievalResult(
            query=query, document_ids=document_ids, scores=scores
        )

    def runtime_metadata(self) -> Dict[str, Any]:
        """Hybrid identity and last-run component rankings.

        Merged into ``RAGRunResult.runtime_metadata`` by the pipeline,
        this gives RunTrace enough facts to later answer "why did the
        hybrid ranking improve (or regress)?" without re-running: the
        sparse ranking, the dense ranking, the fused result (in the
        ``RetrievalEvent``), and every declared component parameter.
        """
        return {
            "hybrid": {
                "fusion_method": self.fusion_method,
                "rrf_k": self.rrf_k,
                "component_top_k": self.component_top_k,
                "final_top_k": self._last_top_k,
                "components": {
                    "sparse": self._component_entry(
                        self._sparse_config,
                        self._sparse,
                        self._last_sparse_result,
                    ),
                    "dense": self._component_entry(
                        self._dense_config,
                        self._dense,
                        self._last_dense_result,
                    ),
                },
            }
        }

    @staticmethod
    def _component_entry(
        config: RetrievalConfig,
        component: Retriever,
        result: Optional[RetrievalResult],
    ) -> Dict[str, Any]:
        entry: Dict[str, Any] = {
            "retriever_id": component.retriever_id,
            "retriever_version": component.retriever_version,
            "parameters": dict(config.parameters),
            "document_ids": (
                list(result.document_ids) if result is not None else []
            ),
            "scores": list(result.scores) if result is not None else [],
        }
        describe = getattr(component, "runtime_metadata", None)
        if callable(describe):
            entry["runtime"] = describe()
        return entry


# Self-registration mirrors dense.py: hybrid.py depends on both
# component retrievers, so retrievers.py must not import it back while
# build_retriever stays the single factory for every retriever id.
RETRIEVER_REGISTRY[HybridRRFRetriever.retriever_id] = HybridRRFRetriever

__all__ = [
    "ComponentFactory",
    "HybridRRFRetriever",
]
