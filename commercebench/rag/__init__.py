"""Phase 1A RAG primitives: Document, Corpus, retrievers, pipeline.

All retrievers here are deterministic development baselines — not BM25,
dense, or hybrid retrieval.
"""

from .corpus import Corpus
from .documents import Document
from .pipeline import (
    DEFAULT_FALLBACK_ANSWER,
    DeterministicRAGPipeline,
    RAGRunResult,
)
from .retrievers import (
    RETRIEVER_REGISTRY,
    KeywordMatchRetriever,
    RetrievalResult,
    Retriever,
    TermFrequencyRetriever,
    build_retriever,
    normalize_text,
    tokenize,
)

__all__ = [
    "Corpus",
    "DEFAULT_FALLBACK_ANSWER",
    "DeterministicRAGPipeline",
    "Document",
    "KeywordMatchRetriever",
    "RAGRunResult",
    "RETRIEVER_REGISTRY",
    "RetrievalResult",
    "Retriever",
    "TermFrequencyRetriever",
    "build_retriever",
    "normalize_text",
    "tokenize",
]
