"""Real pinned-model integration smoke for the Phase 1E dense baseline.

Every test here is marked ``integration`` and excluded from the default
``pytest`` run (see ``addopts`` in ``pyproject.toml``). Run them
explicitly with::

    python -m pytest -m integration

The first run downloads ``sentence-transformers/all-MiniLM-L6-v2`` at
the pinned immutable revision into the local HuggingFace cache; later
runs reuse that cache. These tests must not silently skip: a Phase 1E
PASS requires them to actually execute against the real model.

Numerical assertions stay loose on purpose — the tests pin behaviour
(dimension, finiteness, normalization, ordering, repeatability), not
fifteen-decimal floats; exact cosine math is covered by the fake-vector
unit tests in ``test_dense.py``.
"""

from __future__ import annotations

import math

import pytest

from commercebench.contracts import EvaluationResult
from commercebench.evaluation import DeterministicRAGEvaluator
from commercebench.rag import DenseRetriever, SentenceTransformerEncoder
from commercebench.runner import run_case
from commercebench.systems import DeterministicRAGSystem

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "8b3219a92973c328a8e22fadcfa821b5dc75636a"
BACKEND_VERSION = "6.1.0"
EXPECTED_DIMENSION = 384

pytestmark = pytest.mark.integration


def _encoder(**overrides) -> SentenceTransformerEncoder:
    parameters = dict(
        backend_name="sentence-transformers",
        backend_version=BACKEND_VERSION,
        model_name=MODEL_NAME,
        model_revision=MODEL_REVISION,
        expected_dimension=EXPECTED_DIMENSION,
        normalize_embeddings=True,
    )
    parameters.update(overrides)
    return SentenceTransformerEncoder(**parameters)


@pytest.fixture(scope="module")
def encoder() -> SentenceTransformerEncoder:
    enc = _encoder()
    enc.encode(["warmup"])  # loads the pinned model once per module
    return enc


@pytest.fixture(scope="module")
def dense_system(dense_manifest, rag_corpus, rag_answers):
    return DeterministicRAGSystem.from_manifest(
        dense_manifest, rag_corpus, rag_answers
    )


class TestPinnedModelIdentity:
    def test_revision_is_passed_to_sentence_transformer(
        self, monkeypatch
    ):
        # The pinned immutable revision must reach the real loading API
        # — manifest metadata alone would not satisfy Phase 1E.
        import sentence_transformers

        captured = {}
        real = sentence_transformers.SentenceTransformer

        def spy(*args, **kwargs):
            captured["args"] = args
            captured.update(kwargs)
            return real(*args, **kwargs)

        monkeypatch.setattr(
            sentence_transformers, "SentenceTransformer", spy
        )
        _encoder().encode(["revision probe"])

        assert captured["args"][0] == MODEL_NAME
        assert captured["revision"] == MODEL_REVISION
        assert captured["revision"] != "main"

    def test_backend_version_matches_installed(self):
        import sentence_transformers

        assert sentence_transformers.__version__ == BACKEND_VERSION

    def test_dimension_is_384(self, encoder):
        vectors = encoder.encode(["dimension probe"])
        assert encoder.dimension == EXPECTED_DIMENSION
        assert len(vectors) == 1
        assert len(vectors[0]) == EXPECTED_DIMENSION


class TestEmbeddingProperties:
    def test_embeddings_finite_and_unit_norm(self, encoder):
        vectors = encoder.encode(
            ["return policy question", "shipping takes business days"]
        )
        assert len(vectors) == 2
        for row in vectors:
            assert len(row) == EXPECTED_DIMENSION
            assert all(math.isfinite(value) for value in row)
            norm = math.sqrt(sum(value * value for value in row))
            assert norm == pytest.approx(1.0, abs=1e-3)

    def test_normalize_flag_actually_reaches_runtime(
        self, monkeypatch
    ):
        # normalize_embeddings must reach the real backend encode call,
        # not just the manifest/fingerprint. (MiniLM already emits
        # near-unit vectors, so output norms cannot prove this — the
        # flag itself is spied on instead.)
        import sentence_transformers

        real_encode = sentence_transformers.SentenceTransformer.encode
        calls = []

        def spy(self, sentences, *args, **kwargs):
            calls.append(dict(kwargs))
            return real_encode(self, sentences, *args, **kwargs)

        monkeypatch.setattr(
            sentence_transformers.SentenceTransformer, "encode", spy
        )
        _encoder(normalize_embeddings=False).encode(["probe"])
        assert calls[-1]["normalize_embeddings"] is False
        _encoder(normalize_embeddings=True).encode(["probe"])
        assert calls[-1]["normalize_embeddings"] is True

    def test_encode_is_repeatable(self, encoder):
        first = encoder.encode(["repeatability probe"])
        second = encoder.encode(["repeatability probe"])
        assert len(first) == len(second)
        for a, b in zip(first, second):
            assert a == pytest.approx(b)


class TestDenseRetrieverIntegration:
    def test_semantic_retrieval_ranking(self, encoder):
        # A lexical-mismatch, semantic-similarity probe: the refund
        # policy should outrank unrelated documents.
        from commercebench.rag import Corpus, Document

        corpus = Corpus(
            corpus_id="probe",
            corpus_version="1",
            documents=(
                Document(
                    document_id="refund-policy",
                    text=(
                        "A refund is issued to the original payment "
                        "method within 10 business days after the "
                        "returned item passes inspection."
                    ),
                ),
                Document(
                    document_id="warranty-policy",
                    text=(
                        "Every product includes a one year warranty "
                        "that covers manufacturing defects."
                    ),
                ),
                Document(
                    document_id="size-guide",
                    text=(
                        "The trail shoe runs small; choose one size up "
                        "from your usual size."
                    ),
                ),
            ),
        )
        retriever = DenseRetriever(encoder_factory=lambda **kw: encoder)
        result = retriever.retrieve(
            "Can I get my money back", corpus, top_k=3
        )
        assert result.document_ids[0] == "refund-policy"
        assert len(result.document_ids) == len(result.scores) == 3
        for score in result.scores:
            assert math.isfinite(score)
            assert -1.000001 <= score <= 1.000001
        assert list(result.scores) == sorted(result.scores, reverse=True)

    def test_full_case_produces_trace_and_evaluation(
        self, dense_manifest, dense_system, rag_cases
    ):
        case = rag_cases["rag-return-window-001"]
        trace = run_case(case, dense_manifest, dense_system)

        assert len(trace.retrieval_events) == 1
        event = trace.retrieval_events[0]
        assert event.query == case.user_input
        assert len(event.document_ids) == len(event.scores) == 3
        assert all(math.isfinite(s) for s in event.scores)

        metadata = trace.runtime_metadata
        assert metadata["retriever_id"] == "dense-minilm-l6-v2-cosine-v0"
        assert metadata["model_name"] == MODEL_NAME
        assert metadata["model_revision"] == MODEL_REVISION
        assert metadata["embedding_backend"] == "sentence-transformers"
        assert metadata["embedding_backend_version"] == BACKEND_VERSION
        assert metadata["embedding_dimension"] == EXPECTED_DIMENSION
        assert metadata["similarity"] == "cosine"
        assert metadata["normalize_embeddings"] is True
        assert isinstance(metadata["device"], str) and metadata["device"]

        result = DeterministicRAGEvaluator(ks=(1, 3)).evaluate(
            case, trace
        )
        assert isinstance(result, EvaluationResult)
        assert "retrieval_recall_at_3" in result.metrics

    def test_every_development_case_retrieves(
        self, dense_manifest, dense_system, rag_cases
    ):
        # Dense has no score>0 filter: every case on the non-empty
        # development corpus must emit a full top_k event.
        for case in rag_cases.values():
            trace = run_case(case, dense_manifest, dense_system)
            event = trace.retrieval_events[0]
            assert len(event.document_ids) == len(event.scores) == 3
            assert all(math.isfinite(s) for s in event.scores)

    def test_repeated_run_is_deterministic(
        self, dense_manifest, dense_system, rag_cases
    ):
        case = rag_cases["rag-coupon-001"]
        first = run_case(case, dense_manifest, dense_system)
        second = run_case(case, dense_manifest, dense_system)
        assert first.retrieval_events[0].document_ids == (
            second.retrieval_events[0].document_ids
        )
        assert first.retrieval_events[0].scores == pytest.approx(
            second.retrieval_events[0].scores
        )


class TestDenseVsBM25ObservedDifference:
    """Controlled comparison on shared development fixtures.

    Same corpus, cases, top_k, system, generation fixture, evaluator and
    seed — only the retrieval strategy configuration differs. This is an
    observed fixture difference, not a benchmark conclusion.
    """

    def test_semantic_hit_where_bm25_misses(
        self,
        bm25_manifest,
        dense_manifest,
        dense_system,
        rag_corpus,
        rag_cases,
        rag_answers,
    ):
        # 'Can I get my money back if I change my mind' shares no
        # development-tokenizer tokens with the corpus: every sparse
        # score is zero, so BM25 returns nothing. MiniLM still ranks the
        # refund policy first — the dense/lexical difference is real.
        case = rag_cases["rag-refund-miss-001"]
        bm25_system = DeterministicRAGSystem.from_manifest(
            bm25_manifest, rag_corpus, rag_answers
        )

        bm25_trace = run_case(case, bm25_manifest, bm25_system)
        dense_trace = run_case(case, dense_manifest, dense_system)

        bm25_event = bm25_trace.retrieval_events[0]
        dense_event = dense_trace.retrieval_events[0]
        assert bm25_event.document_ids == ()
        assert dense_event.document_ids[0] == "policy-refund-001"
        assert dense_event.document_ids != bm25_event.document_ids

        evaluator = DeterministicRAGEvaluator(ks=(1, 3))
        bm25_result = evaluator.evaluate(case, bm25_trace)
        dense_result = evaluator.evaluate(case, dense_trace)
        assert bm25_result.metrics["retrieval_recall_at_1"].value == 0.0
        assert dense_result.metrics["retrieval_recall_at_1"].value == 1.0
        assert bm25_result.metrics["retrieval_mrr"].value == 0.0
        assert dense_result.metrics["retrieval_mrr"].value == 1.0
