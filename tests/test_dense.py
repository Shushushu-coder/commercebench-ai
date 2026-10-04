"""Tests for the Phase 1E dense retrieval baseline.

All tests here are fast, deterministic, and offline: every
``DenseRetriever`` under test uses an injected fake ``EmbeddingEncoder``
with hand-chosen vectors, so no sentence-transformers model is loaded
and no network is touched. The real pinned-model integration smoke
lives in ``test_dense_integration.py`` (``@pytest.mark.integration``).

Fake 2-D vectors let tests assert exact cosine behaviour:

    q=[1,0] vs a=[1,0] -> 1.0, b=[0.8,0.6] -> 0.8,
    d=[0,1] -> 0.0, c=[-1,0] -> -1.0
"""

from __future__ import annotations

import pytest

from commercebench.contracts import (
    SCHEMA_VERSION,
    ContractValidationError,
    EvaluationResult,
    ExperimentManifest,
    RetrievalConfig,
)
from commercebench.evaluation import DeterministicRAGEvaluator
from commercebench.rag import (
    BM25Retriever,
    Corpus,
    DenseRetriever,
    DeterministicRAGPipeline,
    Document,
    Retriever,
    SentenceTransformerEncoder,
    build_retriever,
)
from commercebench.rag.dense import EmbeddingEncoder
from commercebench.runner import run_case
from commercebench.systems import DeterministicRAGSystem

MODEL_REVISION = "8b3219a92973c328a8e22fadcfa821b5dc75636a"
OTHER_REVISION = "0123456789abcdef0123456789abcdef01234567"


def _corpus(*documents: tuple) -> Corpus:
    return Corpus(
        corpus_id="c",
        corpus_version="1",
        documents=tuple(
            Document(document_id=document_id, text=text)
            for document_id, text in documents
        ),
    )


class FakeEncoder:
    """Deterministic EmbeddingEncoder: fixed text -> vector mapping."""

    def __init__(self, vectors: dict, dimension: int) -> None:
        self._vectors = {text: list(v) for text, v in vectors.items()}
        self.dimension = dimension
        self.calls = []

    def encode(self, texts):
        texts = list(texts)
        self.calls.append(tuple(texts))
        return [list(self._vectors[text]) for text in texts]


def _fake_factory(vectors: dict, dimension: int = 2, captured: dict = None):
    """Encoder factory recording the kwargs DenseRetriever passes in."""

    def factory(**kwargs):
        if captured is not None:
            captured.update(kwargs)
        return FakeEncoder(vectors, dimension)

    return factory


# query 'q' -> [1,0]; corpus scores: a=1.0, b=0.8, d=0.0, c=-1.0
_UNIT_VECTORS = {
    "q": [1.0, 0.0],
    "text-a": [1.0, 0.0],
    "text-b": [0.8, 0.6],
    "text-c": [-1.0, 0.0],
    "text-d": [0.0, 1.0],
}

_UNIT_CORPUS = _corpus(
    ("d-a", "text-a"),
    ("d-b", "text-b"),
    ("d-c", "text-c"),
    ("d-d", "text-d"),
)


def _dense(vectors=None, corpus_vectors=None, captured=None, **kwargs):
    """DenseRetriever with a fake encoder; dimension defaults to 2."""
    all_vectors = dict(_UNIT_VECTORS)
    if vectors:
        all_vectors.update(vectors)
    if corpus_vectors:
        all_vectors.update(corpus_vectors)
    kwargs.setdefault("expected_dimension", 2)
    return DenseRetriever(
        encoder_factory=_fake_factory(all_vectors, kwargs["expected_dimension"], captured),
        **kwargs,
    )


def _dense_manifest(parameters: dict) -> ExperimentManifest:
    return ExperimentManifest(
        schema_version=SCHEMA_VERSION,
        experiment_id="exp",
        experiment_name="exp",
        benchmark_version="commercebench-dev-0.1",
        case_ids=("rag-coupon-001",),
        system_id="deterministic-rag",
        system_version="0.1",
        model=None,
        retrieval=RetrievalConfig(
            retriever_id="dense-minilm-l6-v2-cosine-v0",
            parameters=parameters,
        ),
        memory=None,
        evaluator_id="deterministic-rag-v0",
        evaluator_version="0.1",
        random_seed=0,
        metadata={},
    )


class TestDenseIdentity:
    def test_protocol_and_identity(self):
        retriever = _dense()
        assert isinstance(retriever, Retriever)
        assert retriever.retriever_id == "dense-minilm-l6-v2-cosine-v0"
        assert retriever.retriever_version == "0.1"

    def test_defaults_defined_once(self):
        # Default construction builds only a lazy encoder (no model
        # load), so defaults are observable without the fake.
        retriever = DenseRetriever()
        assert retriever.embedding_backend == (
            DenseRetriever.DEFAULT_EMBEDDING_BACKEND
        ) == "sentence-transformers"
        assert retriever.embedding_backend_version == (
            DenseRetriever.DEFAULT_EMBEDDING_BACKEND_VERSION
        ) == "6.1.0"
        assert retriever.model_name == (
            DenseRetriever.DEFAULT_MODEL_NAME
        ) == "sentence-transformers/all-MiniLM-L6-v2"
        assert retriever.model_revision == (
            DenseRetriever.DEFAULT_MODEL_REVISION
        ) == MODEL_REVISION
        assert retriever.similarity == (
            DenseRetriever.DEFAULT_SIMILARITY
        ) == "cosine"
        assert retriever.normalize_embeddings is (
            DenseRetriever.DEFAULT_NORMALIZE_EMBEDDINGS
        ) is True
        assert retriever.expected_dimension == (
            DenseRetriever.DEFAULT_EXPECTED_DIMENSION
        ) == 384

    def test_default_encoder_is_sentence_transformers(self):
        # Construction does not load the model: only a lazy
        # SentenceTransformerEncoder is created.
        retriever = DenseRetriever(expected_dimension=384)
        assert isinstance(retriever._encoder, SentenceTransformerEncoder)
        assert retriever._encoder.device is None

    def test_algorithm_parameters_declared(self):
        declared = set(DenseRetriever.algorithm_parameters)
        assert declared == {
            "embedding_backend",
            "embedding_backend_version",
            "model_name",
            "model_revision",
            "similarity",
            "normalize_embeddings",
            "expected_dimension",
        }


class TestDenseConfigValidation:
    @pytest.mark.parametrize(
        "backend", ["other-backend", "", "sentence_transformers", None]
    )
    def test_unsupported_backend_rejected(self, backend):
        with pytest.raises(ContractValidationError):
            _dense(embedding_backend=backend)

    @pytest.mark.parametrize("version", ["", None, 6.1])
    def test_invalid_backend_version_rejected(self, version):
        with pytest.raises(ContractValidationError):
            _dense(embedding_backend_version=version)

    @pytest.mark.parametrize("name", ["", None, 123])
    def test_invalid_model_name_rejected(self, name):
        with pytest.raises(ContractValidationError):
            _dense(model_name=name)

    @pytest.mark.parametrize(
        "revision",
        [
            "main",
            "master",
            "HEAD",
            "",
            "8b3219a",
            "g" * 40,
            MODEL_REVISION.upper(),
            "refs/pr/1",
            None,
        ],
    )
    def test_non_pinned_revision_rejected(self, revision):
        with pytest.raises(ContractValidationError):
            _dense(model_revision=revision)

    @pytest.mark.parametrize(
        "similarity", ["dot", "euclidean", "COSINE", "", None]
    )
    def test_unsupported_similarity_rejected(self, similarity):
        with pytest.raises(ContractValidationError):
            _dense(similarity=similarity)

    @pytest.mark.parametrize("value", ["yes", 1, 0, None])
    def test_non_bool_normalize_rejected(self, value):
        with pytest.raises(ContractValidationError):
            _dense(normalize_embeddings=value)

    @pytest.mark.parametrize("value", [0, -1, True, "384", 2.0])
    def test_invalid_expected_dimension_rejected(self, value):
        with pytest.raises(ContractValidationError):
            _dense(expected_dimension=value)

    def test_encoder_dimension_mismatch_rejected(self):
        def factory(**kwargs):
            return FakeEncoder(_UNIT_VECTORS, dimension=3)

        with pytest.raises(ContractValidationError, match="dimension"):
            DenseRetriever(
                expected_dimension=2, encoder_factory=factory
            )


class TestBuildRetrieverPlumbing:
    def test_build_dense_via_factory(self):
        retriever = build_retriever(
            RetrievalConfig(
                retriever_id="dense-minilm-l6-v2-cosine-v0",
                parameters={"expected_dimension": 2},
            )
        )
        # build_retriever cannot inject a fake encoder, so the default
        # lazy SentenceTransformerEncoder is created without loading.
        assert isinstance(retriever, DenseRetriever)
        assert isinstance(retriever._encoder, SentenceTransformerEncoder)
        assert retriever.expected_dimension == 2

    def test_pipeline_parameters_not_passed_to_retriever(self):
        captured = {}
        retriever = DenseRetriever(
            expected_dimension=2,
            encoder_factory=_fake_factory(_UNIT_VECTORS, captured=captured),
        )
        assert retriever.expected_dimension == 2
        for pipeline_key in ("top_k", "corpus_id", "corpus_version"):
            assert pipeline_key not in captured
        assert captured["model_name"] == DenseRetriever.DEFAULT_MODEL_NAME

    @pytest.mark.parametrize(
        "parameters",
        [
            {"similiarity": "cosine"},
            {"model_revison": MODEL_REVISION},
            {"normalize_embedding": True},
            {"embedding_backend_verison": "6.1.0"},
            {"k1": 1.5},
        ],
    )
    def test_unknown_algorithm_parameter_rejected(self, parameters):
        with pytest.raises(ContractValidationError, match="algorithm"):
            build_retriever(
                RetrievalConfig(
                    retriever_id="dense-minilm-l6-v2-cosine-v0",
                    parameters=parameters,
                )
            )

    def test_model_identity_reaches_encoder_construction(self):
        captured = {}
        _dense(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_revision=OTHER_REVISION,
            embedding_backend_version="6.1.0",
            normalize_embeddings=False,
            similarity="cosine",
            captured=captured,
        )
        assert captured == {
            "backend_name": "sentence-transformers",
            "backend_version": "6.1.0",
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "model_revision": OTHER_REVISION,
            "expected_dimension": 2,
            "normalize_embeddings": False,
        }

    def test_manifest_parameters_reach_encoder_runtime(self):
        # The same parameters that shape the fingerprint must reach the
        # encoder construction, not just the manifest. DenseRetriever is
        # called with the manifest parameters exactly as build_retriever
        # would pass them; the injected factory records what arrives.
        captured = {}
        manifest_parameters = {
            "embedding_backend": "sentence-transformers",
            "embedding_backend_version": "6.1.0",
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "model_revision": MODEL_REVISION,
            "similarity": "cosine",
            "normalize_embeddings": False,
            "expected_dimension": 2,
        }
        config = RetrievalConfig(
            retriever_id="dense-minilm-l6-v2-cosine-v0",
            parameters=manifest_parameters,
        )
        retriever = DenseRetriever(
            **config.parameters,
            encoder_factory=_fake_factory(_UNIT_VECTORS, captured=captured),
        )
        assert captured["model_revision"] == MODEL_REVISION
        assert captured["normalize_embeddings"] is False
        assert captured["backend_version"] == "6.1.0"
        assert captured["expected_dimension"] == 2
        assert retriever.similarity == "cosine"

    def test_encode_receives_corpus_texts_then_query(self):
        retriever = _dense()
        corpus = _corpus(("d-x", "text-a"), ("d-y", "text-b"))
        encoder = retriever._encoder
        retriever.retrieve("q", corpus, top_k=2)
        assert encoder.calls == [("text-a", "text-b"), ("q",)]


class TestCosineVectorMath:
    def test_identical_orthogonal_opposite(self):
        retriever = _dense()
        result = retriever.retrieve("q", _UNIT_CORPUS, top_k=4)
        assert result.document_ids == ("d-a", "d-b", "d-d", "d-c")
        assert result.scores == pytest.approx((1.0, 0.8, 0.0, -1.0))

    def test_true_cosine_not_dot_product(self):
        # Unnormalized fake vector: dot(query, [2,0]) = 2.0 but the
        # cosine must still be exactly 1.0.
        retriever = _dense(
            vectors={"big": [2.0, 0.0], "q2": [1.0, 0.0]}
        )
        corpus = _corpus(("d-big", "big"))
        result = retriever.retrieve("q2", corpus, top_k=1)
        assert result.scores == pytest.approx((1.0,))

    def test_negative_scores_retained(self):
        # Unlike the sparse score>0 policy, dense returns the top_k
        # documents even when their cosine is zero or negative.
        retriever = _dense()
        result = retriever.retrieve("q", _UNIT_CORPUS, top_k=4)
        assert "d-c" in result.document_ids
        index = result.document_ids.index("d-c")
        assert result.scores[index] == pytest.approx(-1.0)

    def test_stable_tie_break_by_document_id(self):
        retriever = _dense(vectors={"z-text": [1.0, 0.0], "a-text": [1.0, 0.0]})
        corpus = _corpus(("z-doc", "z-text"), ("a-doc", "a-text"))
        result = retriever.retrieve("q", corpus, top_k=2)
        assert result.document_ids == ("a-doc", "z-doc")
        assert result.scores == pytest.approx((1.0, 1.0))

    def test_top_k_respected(self):
        retriever = _dense()
        result = retriever.retrieve("q", _UNIT_CORPUS, top_k=2)
        assert result.document_ids == ("d-a", "d-b")
        assert len(result.scores) == 2

    def test_scores_aligned_and_descending(self):
        retriever = _dense()
        result = retriever.retrieve("q", _UNIT_CORPUS, top_k=4)
        assert len(result.document_ids) == len(result.scores)
        assert list(result.scores) == sorted(result.scores, reverse=True)

    def test_deterministic_repeated_runs(self):
        retriever = _dense()
        first = retriever.retrieve("q", _UNIT_CORPUS, top_k=4)
        second = retriever.retrieve("q", _UNIT_CORPUS, top_k=4)
        assert first == second


class TestFiniteAndDimensionSafety:
    def test_nan_embedding_rejected(self):
        retriever = _dense(vectors={"bad": [float("nan"), 0.0]})
        corpus = _corpus(("d-bad", "bad"))
        with pytest.raises(ContractValidationError):
            retriever.retrieve("q", corpus, top_k=1)

    def test_infinite_embedding_rejected(self):
        retriever = _dense(vectors={"bad": [float("inf"), 0.0]})
        corpus = _corpus(("d-bad", "bad"))
        with pytest.raises(ContractValidationError):
            retriever.retrieve("q", corpus, top_k=1)

    def test_nan_query_embedding_rejected(self):
        retriever = _dense(vectors={"q-nan": [float("nan"), 0.0]})
        corpus = _corpus(("d-a", "text-a"))
        with pytest.raises(ContractValidationError):
            retriever.retrieve("q-nan", corpus, top_k=1)

    def test_wrong_row_dimension_rejected(self):
        retriever = _dense(vectors={"bad": [1.0, 0.0, 0.0]})
        corpus = _corpus(("d-bad", "bad"))
        with pytest.raises(ContractValidationError, match="dimension"):
            retriever.retrieve("q", corpus, top_k=1)

    def test_row_count_mismatch_rejected(self):
        class ShortEncoder:
            dimension = 2

            def encode(self, texts):
                return [[1.0, 0.0]]  # always one row regardless of input

        retriever = DenseRetriever(
            expected_dimension=2,
            encoder_factory=lambda **kw: ShortEncoder(),
        )
        with pytest.raises(ContractValidationError):
            retriever.retrieve("q", _UNIT_CORPUS, top_k=2)

    def test_zero_norm_embedding_rejected(self):
        retriever = _dense(vectors={"zero": [0.0, 0.0]})
        corpus = _corpus(("d-zero", "zero"))
        with pytest.raises(ContractValidationError, match="zero"):
            retriever.retrieve("q", corpus, top_k=1)


class TestEdgeCases:
    def test_empty_corpus_returns_empty(self):
        corpus = Corpus(corpus_id="c", corpus_version="1", documents=())
        result = _dense().retrieve("q", corpus, top_k=3)
        assert result.document_ids == ()
        assert result.scores == ()
        assert result.query == "q"

    @pytest.mark.parametrize("query", ["", None, 3])
    def test_invalid_query_rejected(self, query):
        with pytest.raises(ContractValidationError):
            _dense().retrieve(query, _UNIT_CORPUS, top_k=3)

    @pytest.mark.parametrize("top_k", [0, -1, 2.5, "3", None])
    def test_invalid_top_k_rejected(self, top_k):
        with pytest.raises(ContractValidationError):
            _dense().retrieve("q", _UNIT_CORPUS, top_k=top_k)

    def test_non_corpus_rejected(self):
        with pytest.raises(ContractValidationError):
            _dense().retrieve("q", {"documents": []}, top_k=3)


class TestCorpusEmbeddingCache:
    def test_corpus_vectors_cached_within_instance(self):
        retriever = _dense()
        encoder = retriever._encoder
        corpus = _corpus(("d-a", "text-a"), ("d-b", "text-b"))
        retriever.retrieve("q", corpus, top_k=2)
        retriever.retrieve("q", corpus, top_k=2)
        # Two retrieves -> corpus encoded once, query encoded twice.
        assert encoder.calls == [
            ("text-a", "text-b"),
            ("q",),
            ("q",),
        ]

    def test_different_corpus_identity_reencodes(self):
        retriever = _dense()
        encoder = retriever._encoder
        first = Corpus(
            corpus_id="c",
            corpus_version="1",
            documents=(Document(document_id="d", text="text-a"),),
        )
        second = Corpus(
            corpus_id="c",
            corpus_version="2",
            documents=(Document(document_id="d", text="text-a"),),
        )
        retriever.retrieve("q", first, top_k=1)
        retriever.retrieve("q", second, top_k=1)
        assert encoder.calls == [
            ("text-a",),
            ("q",),
            ("text-a",),
            ("q",),
        ]


class TestDenseFingerprint:
    _BASE = {
        "top_k": 3,
        "corpus_id": "commercebench-dev-corpus",
        "corpus_version": "0.1",
        "embedding_backend": "sentence-transformers",
        "embedding_backend_version": "6.1.0",
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "model_revision": MODEL_REVISION,
        "similarity": "cosine",
        "normalize_embeddings": True,
        "expected_dimension": 384,
    }

    def test_same_config_same_fingerprint(self):
        first = _dense_manifest(dict(self._BASE)).config_fingerprint()
        second = _dense_manifest(dict(self._BASE)).config_fingerprint()
        assert first == second

    def test_model_name_change_changes_fingerprint(self):
        changed = dict(self._BASE, model_name="other/model")
        assert _dense_manifest(self._BASE).config_fingerprint() != (
            _dense_manifest(changed).config_fingerprint()
        )

    def test_model_revision_change_changes_fingerprint(self):
        changed = dict(self._BASE, model_revision=OTHER_REVISION)
        assert _dense_manifest(self._BASE).config_fingerprint() != (
            _dense_manifest(changed).config_fingerprint()
        )

    def test_similarity_change_changes_fingerprint(self):
        changed = dict(self._BASE, similarity="dot")
        assert _dense_manifest(self._BASE).config_fingerprint() != (
            _dense_manifest(changed).config_fingerprint()
        )

    def test_normalize_change_changes_fingerprint(self):
        changed = dict(self._BASE, normalize_embeddings=False)
        assert _dense_manifest(self._BASE).config_fingerprint() != (
            _dense_manifest(changed).config_fingerprint()
        )

    def test_backend_version_change_changes_fingerprint(self):
        changed = dict(self._BASE, embedding_backend_version="6.2.0")
        assert _dense_manifest(self._BASE).config_fingerprint() != (
            _dense_manifest(changed).config_fingerprint()
        )

    def test_dense_and_bm25_configs_have_different_fingerprints(
        self, bm25_manifest, dense_manifest
    ):
        assert bm25_manifest.config_fingerprint() != (
            dense_manifest.config_fingerprint()
        )


class TestDenseManifestAndCorpusIdentity:
    def test_manifest_declares_corpus_and_model_identity(
        self, dense_manifest
    ):
        retrieval = dense_manifest.retrieval
        assert retrieval.retriever_id == "dense-minilm-l6-v2-cosine-v0"
        parameters = retrieval.parameters
        assert parameters["top_k"] == 3
        assert parameters["corpus_id"] == "commercebench-dev-corpus"
        assert parameters["corpus_version"] == "0.1"
        assert parameters["embedding_backend"] == "sentence-transformers"
        assert parameters["embedding_backend_version"] == "6.1.0"
        assert parameters["model_name"] == (
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        assert parameters["model_revision"] == MODEL_REVISION
        assert parameters["similarity"] == "cosine"
        assert parameters["normalize_embeddings"] is True
        assert parameters["expected_dimension"] == 384

    def test_corpus_id_mismatch_rejected(self, dense_manifest, rag_answers):
        other = Corpus(
            corpus_id="other-corpus",
            corpus_version="0.1",
            documents=(Document(document_id="x", text="y"),),
        )
        with pytest.raises(ContractValidationError, match="corpus_id"):
            DeterministicRAGSystem.from_manifest(
                dense_manifest, other, rag_answers
            )

    def test_corpus_version_mismatch_rejected(
        self, dense_manifest, rag_answers
    ):
        other = Corpus(
            corpus_id="commercebench-dev-corpus",
            corpus_version="9.9",
            documents=(Document(document_id="x", text="y"),),
        )
        with pytest.raises(ContractValidationError, match="corpus_version"):
            DeterministicRAGSystem.from_manifest(
                dense_manifest, other, rag_answers
            )

    def test_from_manifest_builds_without_loading_model(
        self, dense_manifest, rag_corpus, rag_answers
    ):
        # System construction must not download or load the model.
        system = DeterministicRAGSystem.from_manifest(
            dense_manifest, rag_corpus, rag_answers
        )
        assert system.system_id == "deterministic-rag"


class TestDensePipelineAndEvent:
    def _system(self, corpus, vectors, answers=None, top_k=3):
        retriever = DenseRetriever(
            expected_dimension=2,
            encoder_factory=_fake_factory(vectors),
        )
        pipeline = DeterministicRAGPipeline(
            retriever=retriever,
            corpus=corpus,
            top_k=top_k,
            answer_by_document_id=answers or {},
        )
        return pipeline

    def test_retrieval_event_and_runtime_metadata(self):
        corpus = _corpus(("d-a", "text-a"), ("d-b", "text-b"))
        pipeline = self._system(
            corpus, _UNIT_VECTORS, answers={"d-a": "answer-a"}
        )
        result = pipeline.run("q")
        event = result.retrieval_event
        assert event.query == "q"
        assert event.document_ids == ("d-a", "d-b")
        assert event.scores == pytest.approx((1.0, 0.8))
        assert result.output_text == "answer-a"
        metadata = result.runtime_metadata
        assert metadata["retriever_id"] == "dense-minilm-l6-v2-cosine-v0"
        assert metadata["retriever_version"] == "0.1"
        assert metadata["corpus_id"] == "c"
        assert metadata["corpus_version"] == "1"
        assert metadata["top_k"] == 3
        assert metadata["embedding_backend"] == "sentence-transformers"
        assert metadata["embedding_backend_version"] == "6.1.0"
        assert metadata["model_name"] == (
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        assert metadata["model_revision"] == MODEL_REVISION
        assert metadata["similarity"] == "cosine"
        assert metadata["normalize_embeddings"] is True
        assert metadata["embedding_dimension"] == 2

    def test_run_case_and_evaluation_end_to_end(self):
        corpus = _corpus(("d-a", "text-a"), ("d-c", "text-c"))
        pipeline = self._system(corpus, _UNIT_VECTORS)
        system = DeterministicRAGSystem(pipeline=pipeline)
        case_manifest = ExperimentManifest(
            schema_version=SCHEMA_VERSION,
            experiment_id="exp",
            experiment_name="exp",
            benchmark_version="commercebench-dev-0.1",
            case_ids=("case-1",),
            system_id="deterministic-rag",
            system_version="0.1",
            retrieval=RetrievalConfig(
                retriever_id="dense-minilm-l6-v2-cosine-v0"
            ),
            evaluator_id="deterministic-rag-v0",
            evaluator_version="0.1",
            metadata={},
        )
        from commercebench.contracts import (
            Answerability,
            CaseSpec,
            ConversationType,
            Difficulty,
        )

        case = CaseSpec(
            schema_version=SCHEMA_VERSION,
            case_id="case-1",
            benchmark_version="commercebench-dev-0.1",
            task_family="grounded_qa",
            intent="return",
            difficulty=Difficulty.EASY,
            conversation_type=ConversationType.SINGLE_TURN,
            answerability=Answerability.ANSWERABLE,
            user_input="q",
            required_facts=(),
            forbidden_claims=(),
            relevant_document_ids=("d-a",),
            metadata={},
        )
        trace = run_case(case, case_manifest, system)
        assert len(trace.retrieval_events) == 1
        assert trace.retrieval_events[0].document_ids == ("d-a", "d-c")
        result = DeterministicRAGEvaluator(ks=(1, 3)).evaluate(case, trace)
        assert isinstance(result, EvaluationResult)
        assert result.metrics["retrieval_recall_at_1"].value == 1.0
        assert result.metrics["retrieval_mrr"].value == 1.0


class TestControlledComparisonDenseVsBM25:
    """BM25 vs Dense: same everything, retrieval strategy differs."""

    def test_only_retrieval_configuration_differs(
        self, bm25_manifest, dense_manifest
    ):
        for field in (
            "benchmark_version",
            "case_ids",
            "system_id",
            "system_version",
            "model",
            "memory",
            "evaluator_id",
            "evaluator_version",
            "random_seed",
        ):
            assert getattr(bm25_manifest, field) == getattr(
                dense_manifest, field
            )
        assert bm25_manifest.retrieval.retriever_id == "bm25-okapi-v0"
        assert dense_manifest.retrieval.retriever_id == (
            "dense-minilm-l6-v2-cosine-v0"
        )
        assert bm25_manifest.config_fingerprint() != (
            dense_manifest.config_fingerprint()
        )

    def test_dense_and_bm25_can_rank_differently(self):
        # Same raw corpus, same query; fake vectors make the dense side
        # prefer the semantically-aligned document BM25 cannot match.
        corpus = _corpus(
            ("d-lexical", "match alpha beta"),
            ("d-semantic", "text-c"),
        )
        vectors = {
            "qq": [1.0, 0.0],
            "match alpha beta": [0.6, 0.8],
            "text-c": [-1.0, 0.0],
        }
        dense = DenseRetriever(
            expected_dimension=2,
            encoder_factory=_fake_factory(vectors),
        )
        dense_result = dense.retrieve("qq", corpus, top_k=2)
        bm25_result = BM25Retriever().retrieve("qq", corpus, top_k=2)
        assert dense_result.document_ids == ("d-lexical", "d-semantic")
        assert dense_result.scores == pytest.approx((0.6, -1.0))
        assert bm25_result.document_ids == ()
        assert dense_result.document_ids != bm25_result.document_ids

    def test_dense_uses_same_raw_document_text(self):
        # Fairness check: the encoder receives Document.text verbatim —
        # no title, metadata, or enrichment is added for the dense side.
        retriever = _dense()
        encoder = retriever._encoder
        corpus = _corpus(("d-a", "text-a"))
        retriever.retrieve("q", corpus, top_k=1)
        assert encoder.calls[0] == ("text-a",)
        assert encoder.calls[1] == ("q",)


class TestSentenceTransformerEncoderValidation:
    """Backend-boundary validation that needs no model download."""

    def test_backend_name_enforced(self):
        with pytest.raises(ContractValidationError):
            SentenceTransformerEncoder(
                backend_name="other",
                backend_version="6.1.0",
                model_name="m",
                model_revision=MODEL_REVISION,
                expected_dimension=384,
                normalize_embeddings=True,
            )

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"model_name": ""},
            {"model_revision": ""},
            {"backend_version": ""},
            {"expected_dimension": 0},
            {"normalize_embeddings": "yes"},
        ],
    )
    def test_invalid_constructor_arguments_rejected(self, kwargs):
        base = dict(
            backend_name="sentence-transformers",
            backend_version="6.1.0",
            model_name="m",
            model_revision=MODEL_REVISION,
            expected_dimension=384,
            normalize_embeddings=True,
        )
        base.update(kwargs)
        with pytest.raises(ContractValidationError):
            SentenceTransformerEncoder(**base)

    def test_output_validation_rejects_bad_vectors(self):
        encoder = SentenceTransformerEncoder(
            backend_name="sentence-transformers",
            backend_version="6.1.0",
            model_name="m",
            model_revision=MODEL_REVISION,
            expected_dimension=2,
            normalize_embeddings=True,
        )
        with pytest.raises(ContractValidationError):
            encoder._validate_vectors([[1.0, 0.0, 0.0]])
        with pytest.raises(ContractValidationError):
            encoder._validate_vectors([[float("nan"), 0.0]])
        with pytest.raises(ContractValidationError):
            encoder._validate_vectors([[0.0, 0.0]])
        with pytest.raises(ContractValidationError):
            encoder._validate_vectors([[2.0, 0.0]])

    def test_encoder_protocol_shape(self):
        encoder = SentenceTransformerEncoder(
            backend_name="sentence-transformers",
            backend_version="6.1.0",
            model_name="m",
            model_revision=MODEL_REVISION,
            expected_dimension=384,
            normalize_embeddings=True,
        )
        assert isinstance(encoder, EmbeddingEncoder)
        assert encoder.dimension == 384
