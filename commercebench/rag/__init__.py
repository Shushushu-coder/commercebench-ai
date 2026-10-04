"""Phase 1 RAG primitives: Document, Corpus, retrievers, pipeline.

Sparse retrievers (keyword, term-frequency, Okapi BM25) are
deterministic, offline, stdlib-only development baselines. Phase 1E adds
one dense baseline, ``dense-minilm-l6-v2-cosine-v0``: importing it is
still stdlib-only, but actually retrieving loads the pinned
sentence-transformers MiniLM model. Phase 1G adds the first hybrid
baseline, ``hybrid-bm25-minilm-rrf-v0``: parallel BM25 + MiniLM
retrieval fused by Reciprocal Rank Fusion. No reranking.
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
from .dense import (
    DenseRetriever,
    EmbeddingEncoder,
    SentenceTransformerEncoder,
)
from .hybrid import HybridRRFRetriever

__all__ = [
    "BM25Retriever",
    "Corpus",
    "DEFAULT_FALLBACK_ANSWER",
    "DenseRetriever",
    "DeterministicRAGPipeline",
    "Document",
    "EmbeddingEncoder",
    "HybridRRFRetriever",
    "KeywordMatchRetriever",
    "PIPELINE_PARAMETER_NAMES",
    "RAGRunResult",
    "RETRIEVER_REGISTRY",
    "RetrievalResult",
    "Retriever",
    "SentenceTransformerEncoder",
    "TermFrequencyRetriever",
    "build_retriever",
    "normalize_text",
    "tokenize",
]
