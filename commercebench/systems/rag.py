"""DeterministicRAGSystem: a SystemUnderTest wrapping the RAG pipeline.

The system is a thin adapter: it owns the benchmark-facing identity
(``system_id``/``system_version``) and delegates generation to a
``DeterministicRAGPipeline``, so the existing ``run_case`` runner works
unchanged. Phase 2B optionally wires ``manifest.reranker`` through
``build_reranker`` into the pipeline; without it the legacy single-event
behavior is preserved.
"""

from __future__ import annotations

from typing import Mapping, Optional

from commercebench.contracts.errors import ContractValidationError
from commercebench.contracts.experiment import ExperimentManifest
from commercebench.contracts.trace import UsageStats
from commercebench.rag.corpus import Corpus
from commercebench.rag.pipeline import (
    DEFAULT_FALLBACK_ANSWER,
    DeterministicRAGPipeline,
)
from commercebench.rag.retrievers import Retriever, build_retriever

from .base import SystemOutput

DEFAULT_TOP_K = 3


class DeterministicRAGSystem:
    """SystemUnderTest facade for the deterministic RAG pipeline."""

    def __init__(
        self,
        pipeline: DeterministicRAGPipeline,
        system_id: str = "deterministic-rag",
        system_version: str = "0.1",
    ) -> None:
        if not system_id or not system_version:
            raise ContractValidationError(
                "system_id and system_version must be non-empty strings"
            )
        self.system_id = system_id
        self.system_version = system_version
        self._pipeline = pipeline

    def generate(self, user_input: str) -> SystemOutput:
        result = self._pipeline.run(user_input)
        return SystemOutput(
            output_text=result.output_text,
            usage=UsageStats(input_tokens=0, output_tokens=0, total_tokens=0),
            retrieval_events=tuple(result.retrieval_events),
            runtime_metadata=dict(result.runtime_metadata),
        )

    @classmethod
    def from_manifest(
        cls,
        manifest: ExperimentManifest,
        corpus: Corpus,
        answer_by_document_id: Optional[Mapping[str, str]] = None,
        fallback_answer: str = DEFAULT_FALLBACK_ANSWER,
    ) -> "DeterministicRAGSystem":
        """Build a system wired from ``manifest.retrieval`` (+ reranker).

        ``retrieval.parameters`` may carry ``top_k`` (default 3) and
        ``corpus_id``/``corpus_version``; when present, the corpus identity
        is checked against the supplied ``corpus``. All remaining keys are
        retriever algorithm parameters: ``build_retriever`` validates them
        against the retriever's declared parameter names and applies them
        to the runtime retriever.

        When ``manifest.reranker`` is present it is built with
        ``build_reranker`` and wired into the pipeline. ``top_k`` keeps
        its meaning as the final depth; the reranker's ``candidate_top_k``
        is the candidate depth and must satisfy
        ``candidate_top_k >= top_k`` (enforced by the pipeline).
        """
        if not isinstance(manifest, ExperimentManifest):
            raise ContractValidationError(
                "manifest must be an ExperimentManifest"
            )
        if manifest.retrieval is None:
            raise ContractValidationError(
                "manifest.retrieval is required to build a RAG system"
            )
        if not isinstance(corpus, Corpus):
            raise ContractValidationError("corpus must be a Corpus")
        parameters = manifest.retrieval.parameters
        expected_id = parameters.get("corpus_id")
        expected_version = parameters.get("corpus_version")
        if expected_id is not None and expected_id != corpus.corpus_id:
            raise ContractValidationError(
                f"retrieval.parameters corpus_id {expected_id!r} does not "
                f"match corpus {corpus.corpus_id!r}"
            )
        if (
            expected_version is not None
            and expected_version != corpus.corpus_version
        ):
            raise ContractValidationError(
                f"retrieval.parameters corpus_version {expected_version!r} "
                f"does not match corpus {corpus.corpus_version!r}"
            )
        top_k = parameters.get("top_k", DEFAULT_TOP_K)
        retriever: Retriever = build_retriever(manifest.retrieval)
        reranker = None
        if manifest.reranker is not None:
            from commercebench.reranking import build_reranker as _build_reranker

            reranker = _build_reranker(manifest.reranker)
        pipeline = DeterministicRAGPipeline(
            retriever=retriever,
            corpus=corpus,
            top_k=top_k,
            answer_by_document_id=answer_by_document_id,
            fallback_answer=fallback_answer,
            reranker=reranker,
        )
        return cls(
            pipeline=pipeline,
            system_id=manifest.system_id,
            system_version=manifest.system_version,
        )


__all__ = ["DEFAULT_TOP_K", "DeterministicRAGSystem"]
