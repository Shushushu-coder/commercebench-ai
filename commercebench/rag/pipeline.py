"""Deterministic RAG pipeline for Phase 1A.

Flow: ``user_input -> retrieve top_k -> map the top-1 document to a fixed
answer -> RAGRunResult`` carrying the RetrievalEvent facts.

The answer mapping is a Phase 1A deterministic fixture mechanism — a fixed
``document_id -> answer`` table owned by the pipeline configuration, not
part of the Document contract and not a real generation architecture.
No LLM is called anywhere in this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

from commercebench.contracts.errors import ContractValidationError
from commercebench.contracts.trace import RetrievalEvent

from .corpus import Corpus
from .retrievers import Retriever

DEFAULT_FALLBACK_ANSWER = (
    "I am sorry, I could not find an answer in the knowledge base."
)


@dataclass(frozen=True)
class RAGRunResult:
    """What one pipeline run produced: answer text plus retrieval facts."""

    output_text: str
    retrieval_event: RetrievalEvent
    runtime_metadata: Dict[str, Any] = field(default_factory=dict)


class DeterministicRAGPipeline:
    """retrieve top_k, then answer with the top-1 document's fixed answer.

    ``answer_by_document_id`` maps a document_id to the fixed response that
    document should produce. If nothing is retrieved, or the top-1 document
    has no mapped answer, ``fallback_answer`` is returned.
    """

    def __init__(
        self,
        retriever: Retriever,
        corpus: Corpus,
        top_k: int,
        answer_by_document_id: Optional[Mapping[str, str]] = None,
        fallback_answer: str = DEFAULT_FALLBACK_ANSWER,
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

    def run(self, user_input: str) -> RAGRunResult:
        """Retrieve top_k, record the facts, and emit a fixed answer."""
        result = self._retriever.retrieve(
            user_input, self._corpus, self._top_k
        )
        event = RetrievalEvent(
            query=result.query,
            document_ids=result.document_ids,
            scores=result.scores,
        )
        answer = self._fallback_answer
        if result.document_ids:
            answer = self._answers.get(
                result.document_ids[0], self._fallback_answer
            )
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
        return RAGRunResult(
            output_text=answer,
            retrieval_event=event,
            runtime_metadata=runtime_metadata,
        )


__all__ = [
    "DEFAULT_FALLBACK_ANSWER",
    "DeterministicRAGPipeline",
    "RAGRunResult",
]
