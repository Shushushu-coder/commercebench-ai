"""Phase 2B cross-encoder reranker baseline tests (offline).

Every test here is fast, deterministic, and offline: the learned
``CrossEncoderReranker`` is exercised through an injected fake scorer, so
no sentence-transformers model is loaded and no network is touched. The
real pinned-model smoke lives in ``test_reranker_integration.py``
(``@pytest.mark.integration``).
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from commercebench.contracts import (
    SCHEMA_VERSION,
    Answerability,
    CaseSpec,
    ContractValidationError,
    ConversationType,
    Difficulty,
    ExperimentManifest,
    RerankerConfig,
    RetrievalConfig,
    RetrievalEvent,
    RetrievalRole,
    RunTrace,
)
from commercebench.evaluation import DeterministicRAGEvaluator
from commercebench.evaluation.retrieval import RetrievalEvaluator, ranked_document_ids
from commercebench.rag import (
    Corpus,
    DeterministicRAGPipeline,
    Document,
    RetrievalResult,
)
from commercebench.reranking import (
    CrossEncoderReranker,
    IdentityReranker,
    RerankCandidate,
    RerankResult,
    Reranker,
    build_reranker,
)
from commercebench.runner import run_case
from commercebench.systems import DeterministicRAGSystem

IDENTITY_ID = "identity-reranker-v0"
IDENTITY_VERSION = "0.1"
CROSS_ID = "cross-encoder-msmarco-minilm-l6-v2-v0"
CROSS_VERSION = "0.1"
CROSS_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
CROSS_REVISION = "ce0834f22110de6d9222af7a7a03628121708969"
CROSS_BACKEND_VERSION = "6.1.0"
OTHER_REVISION = "0123456789abcdef0123456789abcdef01234567"

RAG_V0_DIR = Path(__file__).resolve().parents[1] / "examples" / "rag_v0"

# Legacy fingerprints frozen before Phase 2B (reranker=None must not drift).
LEGACY_FINGERPRINTS = {
    "rag_keyword_v0.json": "751cbc843b888252716a6f96ec9e0838b6da7ed0b82fc753298a30c74753a37a",
    "rag_term_frequency_v0.json": "a4ba4063571667fc8fd76ac6ad86669e1017e46124df3204f2606f1d52321ba3",
    "rag_bm25_v0.json": "69241cf2d2238b3a243e7d0e2977b00f37ebbefd6c69fcff031563f0f9f9a0f6",
    "rag_dense_minilm_v0.json": "719511aa837aea8078109ff19882992071c6dc3b58562bfcd286dcc0072c634c",
    "rag_hybrid_rrf_v0.json": "2fb52fe18bd9eaf220b2c4b4af0be32aec3453ea461b31b28bab7459367580ba",
}


def _corpus(*documents: tuple) -> Corpus:
    return Corpus(
        corpus_id="c",
        corpus_version="1",
        documents=tuple(
            Document(document_id=document_id, text=text)
            for document_id, text in documents
        ),
    )


def _candidates(*ids: str, scores: tuple | None = None) -> tuple:
    if scores is None:
        scores = tuple(float(len(ids) - i) for i in range(len(ids)))
    return tuple(
        RerankCandidate(
            document=Document(document_id=document_id, text=f"text-{document_id}"),
            retrieval_rank=index + 1,
            retrieval_score=float(score),
        )
        for index, (document_id, score) in enumerate(zip(ids, scores))
    )


def _score_map(mapping: dict):
    def scorer(query, candidates):
        return [float(mapping[c.document_id]) for c in candidates]

    return scorer


def _cross(scores: dict, **kwargs) -> CrossEncoderReranker:
    kwargs.setdefault("scorer", _score_map(scores))
    return CrossEncoderReranker(**kwargs)


def _manifest_kwargs(**overrides):
    base = dict(
        schema_version=SCHEMA_VERSION,
        experiment_id="exp-001",
        experiment_name="experiment",
        benchmark_version="commercebench-dev-0.1",
        case_ids=("case-001",),
        system_id="deterministic-rag",
        system_version="0.1",
        model=None,
        retrieval=None,
        memory=None,
        evaluator_id="deterministic-rag-v0",
        evaluator_version="0.1",
        random_seed=0,
        metadata={},
    )
    base.update(overrides)
    return base


def _rerank_manifest(reranker: RerankerConfig | dict | None) -> ExperimentManifest:
    if isinstance(reranker, dict):
        reranker = RerankerConfig.from_dict(reranker)
    return ExperimentManifest(**_manifest_kwargs(reranker=reranker))


def _identity_config(candidate_top_k: int = 10) -> RerankerConfig:
    return RerankerConfig(
        reranker_id=IDENTITY_ID,
        reranker_version=IDENTITY_VERSION,
        parameters={"candidate_top_k": candidate_top_k},
    )


def _cross_config(**overrides) -> RerankerConfig:
    parameters = dict(
        candidate_top_k=10,
        backend="sentence-transformers",
        backend_version=CROSS_BACKEND_VERSION,
        model_name=CROSS_MODEL,
        model_revision=CROSS_REVISION,
        max_length=512,
        score_activation="identity",
    )
    parameters.update(overrides)
    return RerankerConfig(
        reranker_id=CROSS_ID,
        reranker_version=CROSS_VERSION,
        parameters=parameters,
    )


class _StubRetriever:
    retriever_id = "stub-retriever-v0"
    retriever_version = "0.1"

    def __init__(self, document_ids, scores=None):
        self._document_ids = tuple(document_ids)
        self._scores = (
            tuple(float(s) for s in scores)
            if scores is not None
            else tuple(float(len(document_ids) - i) for i in range(len(document_ids)))
        )
        self.calls = []

    def retrieve(self, query, corpus, top_k):
        self.calls.append({"query": query, "corpus": corpus, "top_k": top_k})
        ids = self._document_ids[:top_k]
        return RetrievalResult(
            query=query, document_ids=ids, scores=self._scores[: len(ids)]
        )

    def runtime_metadata(self):
        return {}


class TestRerankerConfig:
    def test_dict_roundtrip_identity(self):
        config = _identity_config()
        assert RerankerConfig.from_dict(config.to_dict()) == config

    def test_dict_roundtrip_cross_encoder(self):
        config = _cross_config()
        assert RerankerConfig.from_dict(config.to_dict()) == config

    def test_json_roundtrip(self):
        config = _cross_config()
        payload = json.dumps(config.to_dict(), sort_keys=True)
        assert RerankerConfig.from_dict(json.loads(payload)) == config

    def test_manifest_roundtrip_with_reranker(self, manifest):
        reranked = ExperimentManifest(
            **_manifest_kwargs(reranker=_cross_config())
        )
        assert ExperimentManifest.from_dict(reranked.to_dict()) == reranked
        assert ExperimentManifest.from_json(reranked.to_json()) == reranked

    def test_legacy_missing_reranker_loads_none(self):
        manifest = ExperimentManifest(**_manifest_kwargs())
        assert manifest.reranker is None
        restored = ExperimentManifest.from_dict(manifest.to_dict())
        # to_dict emits explicit null for the absent reranker.
        assert restored.reranker is None
        # A legacy dict without the key at all also loads.
        data = manifest.to_dict()
        del data["reranker"]
        assert ExperimentManifest.from_dict(data).reranker is None
        legacy_json = json.dumps(data)
        assert ExperimentManifest.from_json(legacy_json).reranker is None

    def test_explicit_null_reranker_loads_none(self):
        data = _manifest_kwargs()
        data["reranker"] = None
        assert ExperimentManifest.from_dict(data).reranker is None

    @pytest.mark.parametrize("field", ["reranker_id", "reranker_version"])
    def test_empty_id_version_rejected(self, field):
        kwargs = dict(
            reranker_id=IDENTITY_ID,
            reranker_version=IDENTITY_VERSION,
            parameters={"candidate_top_k": 10},
        )
        kwargs[field] = ""
        with pytest.raises(ContractValidationError):
            RerankerConfig(**kwargs)

    def test_non_json_parameters_rejected(self):
        with pytest.raises(ContractValidationError):
            RerankerConfig(
                reranker_id=IDENTITY_ID,
                reranker_version=IDENTITY_VERSION,
                parameters={"candidate_top_k": object()},
            )

    def test_missing_required_field_rejected(self):
        with pytest.raises(ContractValidationError):
            RerankerConfig.from_dict({"reranker_id": IDENTITY_ID})

    def test_non_mapping_rejected(self):
        with pytest.raises(ContractValidationError):
            RerankerConfig.from_dict("not-a-mapping")


class TestLegacyFingerprintStable:
    @pytest.mark.parametrize("filename,expected", sorted(LEGACY_FINGERPRINTS.items()))
    def test_legacy_manifest_fingerprint_unchanged(self, filename, expected):
        manifest = ExperimentManifest.from_json(
            (RAG_V0_DIR / "experiments" / filename).read_text(encoding="utf-8")
        )
        assert manifest.reranker is None
        assert manifest.config_fingerprint() == expected

    def test_none_omitted_from_fingerprint_payload(self, manifest):
        assert manifest.reranker is None
        assert "reranker" not in manifest._fingerprint_payload()

    def test_explicit_none_same_as_missing(self):
        without = ExperimentManifest(**_manifest_kwargs())
        with_none = ExperimentManifest(**_manifest_kwargs(reranker=None))
        assert without.config_fingerprint() == with_none.config_fingerprint()


class TestRerankerFingerprint:
    def test_identity_vs_cross_encoder_differ(self):
        identity = _rerank_manifest(_identity_config())
        cross = _rerank_manifest(_cross_config())
        assert identity.config_fingerprint() != cross.config_fingerprint()

    def test_reranker_present_changes_fingerprint(self):
        base = _rerank_manifest(None)
        with_reranker = _rerank_manifest(_identity_config())
        assert base.config_fingerprint() != with_reranker.config_fingerprint()

    def test_reranker_id_change_changes_fingerprint(self):
        base = _rerank_manifest(_identity_config())
        other = _rerank_manifest(
            RerankerConfig(
                reranker_id="other-reranker-v0",
                reranker_version=IDENTITY_VERSION,
                parameters={"candidate_top_k": 10},
            )
        )
        assert base.config_fingerprint() != other.config_fingerprint()

    def test_reranker_version_change_changes_fingerprint(self):
        base = _rerank_manifest(_identity_config())
        other = _rerank_manifest(
            RerankerConfig(
                reranker_id=IDENTITY_ID,
                reranker_version="0.2",
                parameters={"candidate_top_k": 10},
            )
        )
        assert base.config_fingerprint() != other.config_fingerprint()

    def test_candidate_top_k_change_changes_fingerprint(self):
        base = _rerank_manifest(_identity_config(10))
        other = _rerank_manifest(_identity_config(5))
        assert base.config_fingerprint() != other.config_fingerprint()

    def test_model_revision_change_changes_fingerprint(self):
        base = _rerank_manifest(_cross_config())
        other = _rerank_manifest(_cross_config(model_revision=OTHER_REVISION))
        assert base.config_fingerprint() != other.config_fingerprint()

    def test_max_length_change_changes_fingerprint(self):
        base = _rerank_manifest(_cross_config())
        other = _rerank_manifest(_cross_config(max_length=256))
        assert base.config_fingerprint() != other.config_fingerprint()

    def test_activation_change_changes_fingerprint(self):
        base = _rerank_manifest(_cross_config())
        other = _rerank_manifest(_cross_config(score_activation="sigmoid"))
        assert base.config_fingerprint() != other.config_fingerprint()

    def test_backend_version_change_changes_fingerprint(self):
        base = _rerank_manifest(_cross_config())
        other = _rerank_manifest(_cross_config(backend_version="6.2.0"))
        assert base.config_fingerprint() != other.config_fingerprint()

    def test_model_name_change_changes_fingerprint(self):
        base = _rerank_manifest(_cross_config())
        other = _rerank_manifest(_cross_config(model_name="other/model"))
        assert base.config_fingerprint() != other.config_fingerprint()


class TestRerankBoundary:
    def test_candidate_requires_document_rank_score(self):
        candidate = _candidates("d1")[0]
        assert candidate.document_id == "d1"
        assert candidate.retrieval_rank == 1
        assert candidate.text == "text-d1"

    def test_candidate_invalid_rank_rejected(self):
        with pytest.raises(ContractValidationError):
            RerankCandidate(
                document=Document(document_id="d1", text="t"),
                retrieval_rank=0,
                retrieval_score=1.0,
            )

    def test_candidate_non_finite_score_rejected(self):
        with pytest.raises(ContractValidationError):
            RerankCandidate(
                document=Document(document_id="d1", text="t"),
                retrieval_rank=1,
                retrieval_score=float("nan"),
            )

    def test_result_unique_scores_aligned_finite(self):
        result = RerankResult(query="q", document_ids=("d1", "d2"), scores=(1.0, 2.0))
        assert result.document_ids == ("d1", "d2")
        with pytest.raises(ContractValidationError):
            RerankResult(query="q", document_ids=("d1", "d1"), scores=(1.0, 2.0))
        with pytest.raises(ContractValidationError):
            RerankResult(query="q", document_ids=("d1",), scores=(1.0, 2.0))
        with pytest.raises(ContractValidationError):
            RerankResult(query="q", document_ids=("d1",), scores=(float("inf"),))


class TestIdentityReranker:
    def test_protocol_and_identity(self):
        reranker = IdentityReranker()
        assert isinstance(reranker, Reranker)
        assert reranker.reranker_id == IDENTITY_ID
        assert reranker.reranker_version == IDENTITY_VERSION
        assert reranker.candidate_top_k == 10

    def test_preserves_order(self):
        reranker = IdentityReranker(candidate_top_k=10)
        candidates = _candidates("d1", "d2", "d3")
        result = reranker.rerank("q", candidates, top_k=3)
        assert result.document_ids == ("d1", "d2", "d3")
        assert result.scores == (3.0, 2.0, 1.0)

    def test_truncates_to_final_top_k(self):
        reranker = IdentityReranker(candidate_top_k=10)
        candidates = _candidates("d1", "d2", "d3")
        result = reranker.rerank("q", candidates, top_k=2)
        assert result.document_ids == ("d1", "d2")
        assert result.scores == (3.0, 2.0)

    def test_final_scores_match_candidate_retrieval_scores(self):
        reranker = IdentityReranker(candidate_top_k=10)
        candidates = _candidates("d1", "d2", scores=(0.5, 9.9))
        result = reranker.rerank("q", candidates, top_k=2)
        assert result.scores == (0.5, 9.9)

    @pytest.mark.parametrize("value", [0, -1, True, "10", 2.5, None])
    def test_invalid_candidate_top_k_rejected(self, value):
        with pytest.raises(ContractValidationError):
            IdentityReranker(candidate_top_k=value)

    def test_final_top_k_above_candidate_rejected(self):
        reranker = IdentityReranker(candidate_top_k=2)
        with pytest.raises(ContractValidationError):
            reranker.rerank("q", _candidates("d1", "d2"), top_k=3)

    def test_duplicate_candidate_rejected(self):
        reranker = IdentityReranker()
        candidates = _candidates("d1", "d1")
        with pytest.raises(ContractValidationError):
            reranker.rerank("q", candidates, top_k=2)

    def test_empty_candidates_returns_empty(self):
        reranker = IdentityReranker()
        result = reranker.rerank("q", (), top_k=3)
        assert result.document_ids == ()
        assert result.scores == ()


class TestCrossEncoderFakeScorer:
    def test_protocol_and_defaults(self):
        reranker = _cross({"d1": 1.0})
        assert isinstance(reranker, Reranker)
        assert reranker.reranker_id == CROSS_ID
        assert reranker.reranker_version == CROSS_VERSION
        assert reranker.candidate_top_k == 10
        assert reranker.backend == "sentence-transformers"
        assert reranker.backend_version == CROSS_BACKEND_VERSION
        assert reranker.model_name == CROSS_MODEL
        assert reranker.model_revision == CROSS_REVISION
        assert reranker.max_length == 512
        assert reranker.score_activation == "identity"
        assert reranker.is_loaded is False

    def test_exact_ranking_reorders(self):
        # Hand-written expectation: d1=-1, d2=5, d3=0 -> d2, d3, d1.
        reranker = _cross({"d1": -1.0, "d2": 5.0, "d3": 0.0})
        result = reranker.rerank("q", _candidates("d1", "d2", "d3"), top_k=3)
        assert result.document_ids == ("d2", "d3", "d1")
        assert result.scores == (5.0, 0.0, -1.0)

    def test_top_k_truncates_after_rerank(self):
        reranker = _cross({"d1": -1.0, "d2": 5.0, "d3": 0.0})
        result = reranker.rerank("q", _candidates("d1", "d2", "d3"), top_k=2)
        assert result.document_ids == ("d2", "d3")
        assert result.scores == (5.0, 0.0)

    def test_higher_score_is_better(self):
        reranker = _cross({"d-low": 1.0, "d-high": 5.0})
        result = reranker.rerank("q", _candidates("d-low", "d-high"), top_k=2)
        assert result.document_ids[0] == "d-high"

    def test_tie_break_by_document_id(self):
        reranker = _cross({"d2": 1.0, "d1": 1.0})
        result = reranker.rerank("q", _candidates("d2", "d1"), top_k=2)
        assert result.document_ids == ("d1", "d2")

    def test_negative_scores_retained(self):
        # No score>0 filter: all-negative candidates still rank.
        reranker = _cross({"d1": -5.0, "d2": -1.0, "d3": -3.0})
        result = reranker.rerank("q", _candidates("d1", "d2", "d3"), top_k=3)
        assert result.document_ids == ("d2", "d3", "d1")
        assert result.scores == (-1.0, -3.0, -5.0)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_scores_rejected(self, bad):
        reranker = _cross({"d1": bad, "d2": 1.0})
        with pytest.raises(ContractValidationError):
            reranker.rerank("q", _candidates("d1", "d2"), top_k=2)

    def test_duplicate_candidate_rejected(self):
        reranker = _cross({"d1": 1.0})
        with pytest.raises(ContractValidationError):
            reranker.rerank("q", _candidates("d1", "d1"), top_k=2)

    def test_out_of_candidate_rejected(self):
        class Evil:
            reranker_id = CROSS_ID
            reranker_version = CROSS_VERSION
            candidate_top_k = 10

            def rerank(self, query, candidates, top_k):
                return RerankResult(query=query, document_ids=("d3",), scores=(9.0,))

            def runtime_metadata(self):
                return {}

        evil = Evil()
        candidates = _candidates("d1", "d2")
        result = evil.rerank("q", candidates, top_k=1)
        # The reranker boundary itself must fail closed when validated;
        # the pipeline re-checks independently (see pipeline tests).
        from commercebench.reranking.contracts import (
            validate_result_against_candidates,
        )

        with pytest.raises(ContractValidationError):
            validate_result_against_candidates(
                ("d1", "d2"), result.document_ids, result.scores, 1
            )

    def test_cross_encoder_validates_subset(self):
        def evil_scorer(query, candidates):
            return [1.0 for _ in candidates]

        reranker = CrossEncoderReranker(scorer=evil_scorer)
        # A scorer cannot introduce new IDs: the rerank output is built
        # from the candidate IDs only, so subset holds by construction.
        result = reranker.rerank("q", _candidates("d1", "d2"), top_k=2)
        assert set(result.document_ids) <= {"d1", "d2"}

    def test_final_length_bounded_by_top_k(self):
        reranker = _cross({"d1": 3.0, "d2": 2.0, "d3": 1.0})
        result = reranker.rerank("q", _candidates("d1", "d2", "d3"), top_k=1)
        assert len(result.document_ids) == 1

    @pytest.mark.parametrize(
        "revision", ["main", "master", "HEAD", "", "8b3219a", "g" * 40, "refs/pr/1", None]
    )
    def test_non_pinned_revision_rejected(self, revision):
        with pytest.raises(ContractValidationError):
            CrossEncoderReranker(
                model_revision=revision, scorer=_score_map({"d1": 1.0})
            )

    def test_uppercase_revision_rejected(self):
        with pytest.raises(ContractValidationError):
            CrossEncoderReranker(
                model_revision=CROSS_REVISION.upper(),
                scorer=_score_map({"d1": 1.0}),
            )

    @pytest.mark.parametrize("value", ["sigmoid", "softmax", "", None])
    def test_non_identity_activation_rejected(self, value):
        with pytest.raises(ContractValidationError):
            CrossEncoderReranker(
                score_activation=value, scorer=_score_map({"d1": 1.0})
            )

    @pytest.mark.parametrize("value", [0, -1, True, "512", None])
    def test_invalid_max_length_rejected(self, value):
        with pytest.raises(ContractValidationError):
            CrossEncoderReranker(
                max_length=value, scorer=_score_map({"d1": 1.0})
            )

    @pytest.mark.parametrize("value", [0, -1, True, "10", None])
    def test_invalid_candidate_top_k_rejected(self, value):
        with pytest.raises(ContractValidationError):
            CrossEncoderReranker(
                candidate_top_k=value, scorer=_score_map({"d1": 1.0})
            )

    def test_candidate_top_k_above_final_enforced(self):
        reranker = CrossEncoderReranker(
            candidate_top_k=2, scorer=_score_map({"d1": 1.0, "d2": 2.0})
        )
        with pytest.raises(ContractValidationError):
            reranker.rerank("q", _candidates("d1", "d2"), top_k=3)


class TestBuildReranker:
    def test_build_identity(self):
        reranker = build_reranker(_identity_config())
        assert isinstance(reranker, IdentityReranker)

    def test_build_cross_encoder_lazy(self):
        reranker = build_reranker(_cross_config())
        assert isinstance(reranker, CrossEncoderReranker)
        assert reranker.is_loaded is False
        assert reranker.model_revision == CROSS_REVISION
        assert reranker.max_length == 512
        assert reranker.score_activation == "identity"

    def test_unknown_reranker_rejected(self):
        with pytest.raises(ContractValidationError, match="unknown"):
            build_reranker(
                RerankerConfig(
                    reranker_id="nope-v0",
                    reranker_version="0.1",
                    parameters={},
                )
            )

    def test_unknown_parameter_rejected(self):
        config = _identity_config()
        bad = RerankerConfig(
            reranker_id=config.reranker_id,
            reranker_version=config.reranker_version,
            parameters={"candidate_top_k": 10, "typo": 1},
        )
        with pytest.raises(ContractValidationError, match="declare"):
            build_reranker(bad)

    def test_execution_knob_in_manifest_rejected(self):
        bad = _cross_config(device="cpu")
        with pytest.raises(ContractValidationError, match="declare"):
            build_reranker(bad)

    def test_version_mismatch_rejected(self):
        bad = RerankerConfig(
            reranker_id=IDENTITY_ID,
            reranker_version="9.9",
            parameters={"candidate_top_k": 10},
        )
        with pytest.raises(ContractValidationError, match="version"):
            build_reranker(bad)

    def test_registered_ids(self):
        from commercebench.reranking import RERANKER_REGISTRY

        assert set(RERANKER_REGISTRY) == {IDENTITY_ID, CROSS_ID}

    def test_manifest_parameters_reach_runtime(self):
        captured: dict = {}

        def factory(**kwargs):
            captured.update(kwargs)
            raise AssertionError("must not load a real model in offline tests")

        # Construction captures the fingerprinted values without loading.
        reranker = CrossEncoderReranker(
            candidate_top_k=7,
            model_revision=OTHER_REVISION,
            max_length=256,
            model_factory=factory,
            scorer=_score_map({"d1": 1.0}),
        )
        assert reranker.candidate_top_k == 7
        assert reranker.model_revision == OTHER_REVISION
        assert reranker.max_length == 256


class TestPipelineRerank:
    def _case(self, relevant=()) -> CaseSpec:
        return CaseSpec(
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
            relevant_document_ids=tuple(relevant),
            metadata={},
        )

    def _manifest(self, reranker=None) -> ExperimentManifest:
        return ExperimentManifest(
            schema_version=SCHEMA_VERSION,
            experiment_id="exp",
            experiment_name="exp",
            benchmark_version="commercebench-dev-0.1",
            case_ids=("case-1",),
            system_id="deterministic-rag",
            system_version="0.1",
            retrieval=RetrievalConfig(retriever_id="stub"),
            reranker=reranker,
            evaluator_id="deterministic-rag-v0",
            evaluator_version="0.1",
            metadata={},
        )

    def test_no_reranker_single_legacy_event(self):
        retriever = _StubRetriever(("d1", "d2"))
        corpus = _corpus(("d1", "a"), ("d2", "b"))
        pipeline = DeterministicRAGPipeline(
            retriever=retriever, corpus=corpus, top_k=2
        )
        system = DeterministicRAGSystem(pipeline=pipeline)
        trace = run_case(self._case(), self._manifest(), system)
        assert len(trace.retrieval_events) == 1
        assert trace.retrieval_events[0].role is None
        assert trace.retrieval_events[0].document_ids == ("d1", "d2")
        assert "reranker" not in trace.runtime_metadata

    def test_identity_candidate_final_roles(self):
        retriever = _StubRetriever(("d1", "d2", "d3"))
        corpus = _corpus(("d1", "a"), ("d2", "b"), ("d3", "c"))
        reranker = IdentityReranker(candidate_top_k=3)
        pipeline = DeterministicRAGPipeline(
            retriever=retriever, corpus=corpus, top_k=2, reranker=reranker
        )
        system = DeterministicRAGSystem(pipeline=pipeline)
        trace = run_case(self._case(), self._manifest(), system)
        assert len(trace.retrieval_events) == 2
        assert trace.retrieval_events[0].role is RetrievalRole.CANDIDATE
        assert trace.retrieval_events[1].role is RetrievalRole.FINAL
        assert trace.retrieval_events[0].document_ids == ("d1", "d2", "d3")
        assert trace.retrieval_events[1].document_ids == ("d1", "d2")
        # Candidate scores are retrieval scores; final scores are reranker scores.
        assert trace.retrieval_events[0].scores == (3.0, 2.0, 1.0)
        assert trace.retrieval_events[1].scores == (3.0, 2.0)

    def test_cross_encoder_fake_candidate_final(self):
        retriever = _StubRetriever(("d1", "d2", "d3"))
        corpus = _corpus(("d1", "a"), ("d2", "b"), ("d3", "c"))
        reranker = _cross({"d1": -1.0, "d2": 5.0, "d3": 0.0}, candidate_top_k=3)
        pipeline = DeterministicRAGPipeline(
            retriever=retriever, corpus=corpus, top_k=2, reranker=reranker
        )
        system = DeterministicRAGSystem(pipeline=pipeline)
        trace = run_case(self._case(), self._manifest(), system)
        assert trace.retrieval_events[0].role is RetrievalRole.CANDIDATE
        assert trace.retrieval_events[1].role is RetrievalRole.FINAL
        assert trace.retrieval_events[0].document_ids == ("d1", "d2", "d3")
        assert trace.retrieval_events[1].document_ids == ("d2", "d3")
        assert trace.retrieval_events[1].scores == (5.0, 0.0)

    def test_retriever_receives_candidate_top_k(self):
        retriever = _StubRetriever(("d1", "d2", "d3"))
        corpus = _corpus(("d1", "a"), ("d2", "b"), ("d3", "c"))
        reranker = IdentityReranker(candidate_top_k=3)
        pipeline = DeterministicRAGPipeline(
            retriever=retriever, corpus=corpus, top_k=2, reranker=reranker
        )
        pipeline.run("q")
        assert retriever.calls[0]["top_k"] == 3

    def test_no_reranker_retriever_receives_final_top_k(self):
        retriever = _StubRetriever(("d1", "d2"))
        corpus = _corpus(("d1", "a"), ("d2", "b"))
        pipeline = DeterministicRAGPipeline(
            retriever=retriever, corpus=corpus, top_k=2
        )
        pipeline.run("q")
        assert retriever.calls[0]["top_k"] == 2

    def test_candidate_gte_final_enforced(self):
        retriever = _StubRetriever(("d1",))
        corpus = _corpus(("d1", "a"))
        reranker = IdentityReranker(candidate_top_k=2)
        with pytest.raises(ContractValidationError):
            DeterministicRAGPipeline(
                retriever=retriever, corpus=corpus, top_k=3, reranker=reranker
            )

    def test_candidate_depth_beyond_component_fails(self):
        # Hybrid enforces component_top_k >= candidate_top_k at the
        # retrieval boundary: the pipeline requests candidate_top_k from
        # the retriever, so a shallow hybrid must fail explicitly.
        from commercebench.rag import HybridRRFRetriever

        sparse = _StubRetriever(("d1", "d2"))
        sparse.retriever_id = "bm25-okapi-v0"
        dense = _StubRetriever(("d1", "d2"))
        dense.retriever_id = "dense-minilm-l6-v2-cosine-v0"
        hybrid = HybridRRFRetriever(
            sparse={"retriever_id": "bm25-okapi-v0", "parameters": {"k1": 1.5, "b": 0.75}},
            dense={
                "retriever_id": "dense-minilm-l6-v2-cosine-v0",
                "parameters": {
                    "embedding_backend": "sentence-transformers",
                    "embedding_backend_version": "6.1.0",
                    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                    "model_revision": "8b3219a92973c328a8e22fadcfa821b5dc75636a",
                    "similarity": "cosine",
                    "normalize_embeddings": True,
                    "expected_dimension": 384,
                },
            },
            component_top_k=2,
            component_factory=lambda config: (
                sparse if config.retriever_id == "bm25-okapi-v0" else dense
            ),
        )
        corpus = _corpus(("d1", "a"), ("d2", "b"))
        pipeline = DeterministicRAGPipeline(
            retriever=hybrid,
            corpus=corpus,
            top_k=2,
            reranker=IdentityReranker(candidate_top_k=5),
        )
        with pytest.raises(ContractValidationError, match="component_top_k"):
            pipeline.run("q")

    def test_evaluator_ignores_candidate(self):
        # Candidate hits rank 1, final misses: MRR must be 1/3-style final
        # behavior, never the candidate's 1.0.
        case = self._case(relevant=("d3",))
        retriever = _StubRetriever(("d1", "d2", "d3"))
        corpus = _corpus(("d1", "a"), ("d2", "b"), ("d3", "c"))
        # Fake scorer keeps d3 last in final top-3? Force final (d1,d2,d3)
        # vs candidate (d3,d1,d2) by identity? Instead hand-build trace.
        candidate = RetrievalEvent(
            query="q",
            document_ids=("d3", "d1", "d2"),
            scores=(3.0, 2.0, 1.0),
            role=RetrievalRole.CANDIDATE,
        )
        final = RetrievalEvent(
            query="q",
            document_ids=("d1", "d2", "d3"),
            scores=(3.0, 2.0, 1.0),
            role=RetrievalRole.FINAL,
        )
        trace = run_case(case, self._manifest(), DeterministicRAGSystem(pipeline=DeterministicRAGPipeline(retriever=retriever, corpus=corpus, top_k=3)))
        import dataclasses

        trace = dataclasses.replace(trace, retrieval_events=(candidate, final))
        result = RetrievalEvaluator(ks=(1, 3)).evaluate(case, trace)
        assert ranked_document_ids(trace) == ("d1", "d2", "d3")
        assert result.metrics["retrieval_mrr"].value == pytest.approx(1.0 / 3)
        assert result.metrics["retrieval_recall_at_1"].value == 0.0

    def test_generation_consumes_final_top1(self):
        # Candidate top1 d1 -> answer A; final top1 d2 -> answer B.
        retriever = _StubRetriever(("d1", "d2"))
        corpus = _corpus(("d1", "answer-a-text"), ("d2", "answer-b-text"))
        reranker = _cross({"d1": 0.0, "d2": 9.0}, candidate_top_k=2)
        pipeline = DeterministicRAGPipeline(
            retriever=retriever,
            corpus=corpus,
            top_k=2,
            answer_by_document_id={"d1": "A", "d2": "B"},
            reranker=reranker,
        )
        result = pipeline.run("q")
        assert result.candidate_event.document_ids[0] == "d1"
        assert result.retrieval_event.document_ids[0] == "d2"
        assert result.output_text == "B"

    def test_hybrid_metadata_preserved_and_reranker_present(self):
        # Offline: a stub retriever mimics the hybrid diagnostics shape
        # without loading the real MiniLM model; the pipeline must keep
        # the retriever diagnostics and add the reranker layer.
        class HybridLike(_StubRetriever):
            retriever_id = "hybrid-bm25-minilm-rrf-v0"
            retriever_version = "0.1"

            def runtime_metadata(self):
                return {
                    "hybrid": {
                        "fusion_method": "rrf",
                        "rrf_k": 60,
                        "component_top_k": 3,
                        "final_top_k": 2,
                        "components": {},
                    }
                }

        retriever = HybridLike(("d1", "d2", "d3"))
        corpus = _corpus(("d1", "a"), ("d2", "b"), ("d3", "c"))
        pipeline = DeterministicRAGPipeline(
            retriever=retriever,
            corpus=corpus,
            top_k=2,
            reranker=IdentityReranker(candidate_top_k=3),
        )
        system = DeterministicRAGSystem(pipeline=pipeline)
        trace = run_case(self._case(), self._manifest(), system)
        assert "hybrid" in trace.runtime_metadata
        assert trace.runtime_metadata["hybrid"]["fusion_method"] == "rrf"
        assert "reranker" in trace.runtime_metadata
        reranker_meta = trace.runtime_metadata["reranker"]
        assert reranker_meta["reranker_id"] == IDENTITY_ID
        assert reranker_meta["candidate_top_k"] == 3
        assert reranker_meta["final_top_k"] == 2

    def test_real_hybrid_manifest_loads(self):
        # Manifest wiring is validated without running retrieval (which
        # would load the dense model); execution is covered by integration.
        from commercebench.contracts.experiment import ExperimentManifest as _M

        identity = _M.from_json(
            (RAG_V0_DIR / "experiments" / "rag_hybrid_identity_rerank_v0.json").read_text(encoding="utf-8")
        )
        assert identity.reranker.reranker_id == IDENTITY_ID
        assert identity.retrieval.retriever_id == "hybrid-bm25-minilm-rrf-v0"

    def test_final_ids_subset_candidate(self):
        retriever = _StubRetriever(("d1", "d2", "d3"))
        corpus = _corpus(("d1", "a"), ("d2", "b"), ("d3", "c"))
        reranker = _cross({"d1": 1.0, "d2": 2.0, "d3": 3.0}, candidate_top_k=3)
        pipeline = DeterministicRAGPipeline(
            retriever=retriever, corpus=corpus, top_k=2, reranker=reranker
        )
        result = pipeline.run("q")
        assert set(result.retrieval_event.document_ids) <= set(
            result.candidate_event.document_ids
        )

    def test_pipeline_rejects_out_of_candidate(self):
        class Evil:
            reranker_id = "evil-v0"
            reranker_version = "0.1"
            candidate_top_k = 2

            def rerank(self, query, candidates, top_k):
                return RerankResult(
                    query=query, document_ids=("d-evil",), scores=(99.0,)
                )

            def runtime_metadata(self):
                return {
                    "reranker_id": self.reranker_id,
                    "reranker_version": self.reranker_version,
                    "candidate_top_k": self.candidate_top_k,
                }

        # Bypass the protocol check by duck-typing: pipeline validates
        # isinstance against the runtime-checkable protocol, so give the
        # evil reranker the expected shape via monkeypatching is not
        # needed — construct the pipeline with a real identity reranker
        # then swap in the evil implementation for run().
        retriever = _StubRetriever(("d1", "d2"))
        corpus = _corpus(("d1", "a"), ("d2", "b"))
        pipeline = DeterministicRAGPipeline(
            retriever=retriever,
            corpus=corpus,
            top_k=1,
            reranker=IdentityReranker(candidate_top_k=2),
        )
        object.__setattr__(pipeline, "_reranker", Evil())
        with pytest.raises(ContractValidationError):
            pipeline.run("q")

    def test_trace_json_roundtrip_with_roles(self):
        retriever = _StubRetriever(("d1", "d2"))
        corpus = _corpus(("d1", "a"), ("d2", "b"))
        pipeline = DeterministicRAGPipeline(
            retriever=retriever,
            corpus=corpus,
            top_k=1,
            reranker=IdentityReranker(candidate_top_k=2),
        )
        system = DeterministicRAGSystem(pipeline=pipeline)
        trace = run_case(self._case(), self._manifest(), system)
        restored = RunTrace.from_json(trace.to_json())
        assert restored == trace
        assert [e.role for e in restored.retrieval_events] == [
            RetrievalRole.CANDIDATE,
            RetrievalRole.FINAL,
        ]


class TestRerankerDiagnostics:
    def test_identity_metadata_minimal(self):
        meta = IdentityReranker(candidate_top_k=10).runtime_metadata()
        assert meta["reranker_id"] == IDENTITY_ID
        assert meta["reranker_version"] == IDENTITY_VERSION
        assert meta["candidate_top_k"] == 10

    def test_cross_encoder_metadata_full(self):
        meta = _cross({"d1": 1.0}).runtime_metadata()
        assert meta["reranker_id"] == CROSS_ID
        assert meta["reranker_version"] == CROSS_VERSION
        assert meta["candidate_top_k"] == 10
        assert meta["backend"] == "sentence-transformers"
        assert meta["backend_version"] == CROSS_BACKEND_VERSION
        assert meta["model_name"] == CROSS_MODEL
        assert meta["model_revision"] == CROSS_REVISION
        assert meta["max_length"] == 512
        assert meta["score_activation"] == "identity"
        assert meta["score_semantics"] == "cross_encoder_logit"
        # No local paths, usernames, or home directories.
        blob = json.dumps(meta)
        assert "cache" not in blob.lower()
        assert "/home" not in blob
        assert "/Users" not in blob

    def test_no_hf_cache_path_recorded(self):
        meta = _cross({"d1": 1.0}).runtime_metadata()
        for value in meta.values():
            if isinstance(value, str):
                assert "huggingface" not in value.lower()
                assert ".cache" not in value


class TestControlledAblationConfig:
    def test_identity_vs_cross_encoder_same_except_reranker(self):
        identity = ExperimentManifest.from_json(
            (RAG_V0_DIR / "experiments" / "rag_hybrid_identity_rerank_v0.json").read_text(encoding="utf-8")
        )
        cross = ExperimentManifest.from_json(
            (RAG_V0_DIR / "experiments" / "rag_hybrid_cross_encoder_v0.json").read_text(encoding="utf-8")
        )
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
            assert getattr(identity, field) == getattr(cross, field)
        assert identity.retrieval == cross.retrieval
        assert identity.retrieval.parameters["top_k"] == 3
        assert identity.reranker.parameters["candidate_top_k"] == 10
        assert cross.reranker.parameters["candidate_top_k"] == 10
        assert identity.reranker.reranker_id != cross.reranker.reranker_id
        assert identity.config_fingerprint() != cross.config_fingerprint()


class TestFreshProcess:
    def test_build_does_not_load_model(self):
        code = """
from commercebench.reranking import CrossEncoderReranker
r = CrossEncoderReranker()
assert r.is_loaded is False, 'model must be lazy'
assert r._model is None, 'no weights at construction'
assert r.model_revision == 'ce0834f22110de6d9222af7a7a03628121708969'
assert r.max_length == 512
assert r.score_activation == 'identity'
print('lazy-ok')
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "lazy-ok" in result.stdout

    def test_registry_survives_fresh_import(self):
        code = """
from commercebench.reranking import RERANKER_REGISTRY, build_reranker
from commercebench.contracts.experiment import RerankerConfig
assert {'identity-reranker-v0', 'cross-encoder-msmarco-minilm-l6-v2-v0'} <= set(RERANKER_REGISTRY)
identity = build_reranker(RerankerConfig(reranker_id='identity-reranker-v0', reranker_version='0.1', parameters={'candidate_top_k': 10}))
cross = build_reranker(RerankerConfig(reranker_id='cross-encoder-msmarco-minilm-l6-v2-v0', reranker_version='0.1', parameters={'candidate_top_k': 10, 'backend': 'sentence-transformers', 'backend_version': '6.1.0', 'model_name': 'cross-encoder/ms-marco-MiniLM-L6-v2', 'model_revision': 'ce0834f22110de6d9222af7a7a03628121708969', 'max_length': 512, 'score_activation': 'identity'}))
assert cross.is_loaded is False
assert cross._model is None
print('fresh-registry-ok')
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "fresh-registry-ok" in result.stdout


class TestRuntimeFingerprintSymmetry:
    def test_candidate_top_k_reaches_runtime(self):
        reranker = build_reranker(_identity_config(candidate_top_k=7))
        assert reranker.candidate_top_k == 7

    def test_cross_encoder_config_reaches_construction(self):
        # Offline symmetry check without loading a model: the fingerprinted
        # values must be stored on the runtime object (the real
        # constructor/predict wiring is proved by the integration suite
        # with spies on the actual CrossEncoder API).
        reranker = build_reranker(_cross_config())
        assert isinstance(reranker, CrossEncoderReranker)
        assert reranker.model_name == CROSS_MODEL
        assert reranker.model_revision == CROSS_REVISION
        assert reranker.max_length == 512
        assert reranker.score_activation == "identity"
        assert reranker.backend_version == CROSS_BACKEND_VERSION
        assert reranker.is_loaded is False

    def test_no_execution_knob_in_fingerprint(self):
        # device/batch_size never enter RerankerConfig.parameters, so two
        # configs differing only in execution overrides share a fingerprint.
        base = _cross_config()
        manifest_a = _rerank_manifest(base)
        manifest_b = _rerank_manifest(copy.deepcopy(base))
        assert manifest_a.config_fingerprint() == manifest_b.config_fingerprint()
        reranker_a = CrossEncoderReranker(scorer=_score_map({"d1": 1.0}), batch_size=8)
        reranker_b = CrossEncoderReranker(scorer=_score_map({"d1": 1.0}), batch_size=32)
        assert reranker_a._batch_size != reranker_b._batch_size
