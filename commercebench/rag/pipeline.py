"""Deterministic RAG pipeline for Phase 1A + Phase 2B reranking.

Without a reranker the flow is unchanged: ``user_input -> retrieve
top_k -> map the top-1 document to a fixed answer -> RAGRunResult``
carrying one legacy ``RetrievalEvent`` (no role).

With a reranker the flow is ``retrieve candidate_top_k -> candidate
event (role=candidate, retriever scores) -> rerank to final top_k ->
final event (role=final, reranker scores) -> answer from the final
top-1``. The candidate ranking is diagnostic; the final ranking is what
generation consumes and what retrieval metrics evaluate.

The answer mapping is a Phase 1A deterministic fixture mechanism — a fixed
``document_id -> answer`` table owned by the pipeline configuration, not
part of the Document contract and not a real generation architecture.
No LLM is called anywhere in this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

from commercebench.contracts.errors import ContractValidationError
from commercebench.contracts.trace import RetrievalEvent, RetrievalRole

from .corpus import Corpus
from .retrievers import Retriever

DEFAULT_FALLBACK_ANSWER = (
    "I am sorry, I could not find an answer in the knowledge base."
)


@dataclass(frozen=True)
class RAGRunResult:
    """What one pipeline run produced: answer text plus retrieval facts.

    ``retrieval_event`` is always the final, generation-visible ranking
    (legacy no-role event when no reranker is configured, explicit
    ``role=final`` otherwise). ``candidate_event`` is present only when
    a reranker ran and holds the diagnostic candidate ranking
    (``role=candidate`` with retriever scores).
    """

    output_text: str
    retrieval_event: RetrievalEvent
    runtime_metadata: Dict[str, Any] = field(default_factory=dict)
    candidate_event: Optional[RetrievalEvent] = None

    @property
    def retrieval_events(self) -> Tuple[RetrievalEvent, ...]:
        """Trace-ordered retrieval events: candidate then final."""
        if self.candidate_event is not None:
            return (self.candidate_event, self.retrieval_event)
        return (self.retrieval_event,)


class DeterministicRAGPipeline:
    """retrieve top_k, then answer with the top-1 document's fixed answer.

    ``answer_by_document_id`` maps a document_id to the fixed response that
    document should produce. If nothing is retrieved, or the top-1 document
    has no mapped answer, ``fallback_answer`` is returned.

    When ``reranker`` is provided the retriever is asked for
    ``reranker.candidate_top_k`` candidates (which must satisfy
    ``candidate_top_k >= top_k``); the reranker then produces the final
    ``top_k`` ranking. ``top_k`` therefore keeps its public meaning as
    the final, evaluation-visible and generation-visible depth in both
    modes.
    """

    def __init__(
        self,
        retriever: Retriever,
        corpus: Corpus,
        top_k: int,
        answer_by_document_id: Optional[Mapping[str, str]] = None,
        fallback_answer: str = DEFAULT_FALLBACK_ANSWER,
        reranker: Optional[Any] = None,
    ) -> None:
        if not isinstance(retriever, Retriever):
            raise ContractValidationError(
                "retriever must satisfy the Retriever protocol"
            )
        if not isinstance(corpus, Corpus):
            raise ContractValidationError("corpus must be a Corpus")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ContractValidationError("top_k must be a positive integer")
        if not fallback_answer:
            raise ContractValidationError(
                "fallback_answer must be a non-empty string"
            )
        if reranker is not None:
            from commercebench.reranking.rerankers import Reranker as _Reranker

            if not isinstance(reranker, _Reranker):
                raise ContractValidationError(
                    "reranker must satisfy the Reranker protocol"
                )
            candidate_top_k = getattr(reranker, "candidate_top_k", None)
            if (
                not isinstance(candidate_top_k, int)
                or isinstance(candidate_top_k, bool)
                or candidate_top_k <= 0
            ):
                raise ContractValidationError(
                    "reranker candidate_top_k must be a positive integer"
                )
            if candidate_top_k < top_k:
                raise ContractValidationError(
                    f"reranker candidate_top_k {candidate_top_k} is less "
                    f"than final top_k {top_k}"
                )
        answers: Dict[str, str] = {}
        for document_id, answer in (answer_by_document_id or {}).items():
            if not document_id or not isinstance(answer, str) or not answer:
                raise ContractValidationError(
                    "answer_by_document_id must map non-empty document_ids "
                    "to non-empty answer strings"
                )
            if corpus.get(document_id) is None:
                raise ContractValidationError(
                    f"answer mapped to unknown document_id: {document_id!r}"
                )
            answers[document_id] = answer
        self._retriever = retriever
        self._corpus = corpus
        self._top_k = top_k
        self._answers = answers
        self._fallback_answer = fallback_answer
        self._reranker = reranker

    def run(self, user_input: str) -> RAGRunResult:
        """Retrieve, optionally rerank, record the facts, emit an answer."""
        if self._reranker is None:
            return self._run_without_reranker(user_input)
        return self._run_with_reranker(user_input)

    def _base_metadata(self) -> Dict[str, Any]:
        runtime_metadata = {
            "retriever_id": self._retriever.retriever_id,
            "retriever_version": self._retriever.retriever_version,
            "corpus_id": self._corpus.corpus_id,
            "corpus_version": self._corpus.corpus_version,
            "top_k": self._top_k,
        }
        describe = getattr(self._retriever, "runtime_metadata", None)
        if callable(describe):
            runtime_metadata.update(describe())
        return runtime_metadata

    def _answer_for(self, document_ids: Any) -> str:
        if document_ids:
            return self._answers.get(
                document_ids[0], self._fallback_answer
            )
        return self._fallback_answer

    def _run_without_reranker(self, user_input: str) -> RAGRunResult:
        result = self._retriever.retrieve(
            user_input, self._corpus, self._top_k
        )
        event = RetrievalEvent(
            query=result.query,
            document_ids=result.document_ids,
            scores=result.scores,
        )
        return RAGRunResult(
            output_text=self._answer_for(result.document_ids),
            retrieval_event=event,
            runtime_metadata=self._base_metadata(),
        )

    def _run_with_reranker(self, user_input: str) -> RAGRunResult:
        # Deferred imports keep the Phase 1A import graph stdlib-only.
        from commercebench.reranking.contracts import RerankCandidate

        assert self._reranker is not None
        candidate_top_k: int = self._reranker.candidate_top_k
        final_top_k: int = self._top_k
        candidate_result = self._retriever.retrieve(
            user_input, self._corpus, candidate_top_k
        )
        candidates = []
        for rank, (document_id, score) in enumerate(
            zip(candidate_result.document_ids, candidate_result.scores),
            start=1,
        ):
            document = self._corpus.get(document_id)
            if document is None:
                raise ContractValidationError(
                    f"retriever returned unknown document_id: {document_id!r}"
                )
            candidates.append(
                RerankCandidate(
                    document=document,
                    retrieval_rank=rank,
                    retrieval_score=float(score),
                )
            )
        reranked = self._reranker.rerank(
            candidate_result.query, candidates, final_top_k
        )
        # Fail closed: even a misbehaving reranker cannot smuggle new
        # documents into the final ranking or break alignment.
        allowed = set(candidate_result.document_ids)
        if len(reranked.document_ids) != len(reranked.scores):
            raise ContractValidationError(
                "reranked document_ids and scores must have the same length"
            )
        if len(reranked.document_ids) > final_top_k:
            raise ContractValidationError(
                "reranked result exceeds final top_k"
            )
        for document_id in reranked.document_ids:
            if document_id not in allowed:
                raise ContractValidationError(
                    f"reranked document {document_id!r} is not in the "
                    "candidate set"
                )
        candidate_event = RetrievalEvent(
            query=candidate_result.query,
            document_ids=candidate_result.document_ids,
            scores=candidate_result.scores,
            role=RetrievalRole.CANDIDATE,
        )
        final_event = RetrievalEvent(
            query=reranked.query,
            document_ids=reranked.document_ids,
            scores=reranked.scores,
            role=RetrievalRole.FINAL,
        )
        runtime_metadata = self._base_metadata()
        describe_reranker = getattr(self._reranker, "runtime_metadata", None)
        if callable(describe_reranker):
            reranker_metadata = dict(describe_reranker())
        else:
            reranker_metadata = {
                "reranker_id": self._reranker.reranker_id,
                "reranker_version": self._reranker.reranker_version,
                "candidate_top_k": candidate_top_k,
            }
        reranker_metadata["final_top_k"] = final_top_k
        # ``candidate_top_k`` recorded by the reranker must match the
        # depth actually requested from the retriever.
        reranker_metadata["candidate_top_k"] = candidate_top_k
        runtime_metadata["reranker"] = reranker_metadata
        return RAGRunResult(
            output_text=self._answer_for(reranked.document_ids),
            retrieval_event=final_event,
            runtime_metadata=runtime_metadata,
            candidate_event=candidate_event,
        )


__all__ = [
    "DEFAULT_FALLBACK_ANSWER",
    "DeterministicRAGPipeline",
    "RAGRunResult",
]
