"""Phase 1 RAG primitives: Document, Corpus, retrievers, pipeline.

All retrievers here are deterministic, offline, stdlib-only development
baselines — sparse retrieval only (keyword, term-frequency, Okapi BM25);
no dense or hybrid retrieval.
"""

from .corpus import Corpus
from .documents import Document
from .pipeline import (
    DEFAULT_FALLBACK_ANSWER,
    DeterministicRAGPipeline,
    RAGRunResult,
)
from .retrievers import (
    BM25Retriever,
    PIPELINE_PARAMETER_NAMES,
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
    "BM25Retriever",
    "Corpus",
    "DEFAULT_FALLBACK_ANSWER",
    "DeterministicRAGPipeline",
    "Document",
    "KeywordMatchRetriever",
    "PIPELINE_PARAMETER_NAMES",
    "RAGRunResult",
    "RETRIEVER_REGISTRY",
    "RetrievalResult",
    "Retriever",
    "TermFrequencyRetriever",
    "build_retriever",
    "normalize_text",
    "tokenize",
]
