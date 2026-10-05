"""Phase 2D reranker diagnostic trace tests (offline).

The candidate event (Hybrid ranking) and the final event (reranked
top-k) already exist. This suite proves the missing piece: the full
candidate-level reranker ranking — every candidate's score and order —
travels inside ``RerankResult.diagnostics`` and is persisted under
``runtime_metadata["reranker"]["full_ranking"]`` without changing
retrieval metrics, generation, fingerprints, or event semantics.

All tests are fast, deterministic, and offline: the learned
``CrossEncoderReranker`` is exercised through an injected fake scorer.
The real pinned-model coverage lives in
``test_reranker_diagnostic_integration.py`` (``@pytest.mark.integration``).
"""

from __future__ import annotations

import json
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
    RetrievalEvent,
    RetrievalRole,
    RunTrace,
)
from commercebench.evaluation import DeterministicRAGEvaluator
from commercebench.evaluation.retrieval import ranked_document_ids
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
    RerankDiagnostics,
    RerankResult,
    validate_diagnostics_against_candidates,
)
from commercebench.runner import run_case
from commercebench.systems import DeterministicRAGSystem

RAG_V0_DIR = Path(__file__).resolve().parents[1] / "examples" / "rag_v0"

# Frozen fingerprints: Phase 2D is observability-only, so no manifest or
# config fingerprint may move (values captured on the Phase 2B base).
FROZEN_FINGERPRINTS = {
    "rag_hybrid_rrf_v0.json": "2fb52fe18bd9eaf220b2c4b4af0be32aec3453ea461b31b28bab7459367580ba",
    "rag_hybrid_identity_rerank_v0.json": "6df0f8241251a5e614df7eaf70ada042b06352a11b6bb260f193ca55ccf816e8",
    "rag_hybrid_cross_encoder_v0.json": "2e0a6d5ea0f3d8cebd5f4379bf9ed6a0e7d26dec992a3540ba64539b395407f7",
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

    def retrieve(self, query, corpus, top_k):
        ids = self._document_ids[:top_k]
        return RetrievalResult(
            query=query, document_ids=ids, scores=self._scores[: len(ids)]
        )

    def runtime_metadata(self):
        return {}


def _case(relevant=()) -> CaseSpec:
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


def _manifest() -> ExperimentManifest:
    return ExperimentManifest(
        schema_version=SCHEMA_VERSION,
        experiment_id="exp",
        experiment_name="exp",
        benchmark_version="commercebench-dev-0.1",
        case_ids=("case-1",),
        system_id="deterministic-rag",
        system_version="0.1",
        retrieval=None,
        reranker=None,
        evaluator_id="deterministic-rag-v0",
        evaluator_version="0.1",
        metadata={},
    )


class TestDiagnosticsContract:
    def test_aligned_ids_scores(self):
        diagnostics = RerankDiagnostics(
            document_ids=("d1", "d2"), scores=(1.0, 2.0)
        )
        assert diagnostics.document_ids == ("d1", "d2")
        assert diagnostics.scores == (1.0, 2.0)

    def test_empty_allowed(self):
        diagnostics = RerankDiagnostics(document_ids=(), scores=())
        assert diagnostics.document_ids == ()
        assert diagnostics.scores == ()

    def test_misaligned_rejected(self):
        with pytest.raises(ContractValidationError):
            RerankDiagnostics(document_ids=("d1",), scores=(1.0, 2.0))

    def test_duplicate_ids_rejected(self):
        # Duplicates are rejected even though a final slice could look unique.
        with pytest.raises(ContractValidationError):
            RerankDiagnostics(document_ids=("d1", "d1"), scores=(1.0, 2.0))

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_rejected(self, bad):
        with pytest.raises(ContractValidationError):
            RerankDiagnostics(document_ids=("d1", "d2"), scores=(bad, 1.0))

    def test_bool_score_rejected(self):
        with pytest.raises(ContractValidationError):
            RerankDiagnostics(document_ids=("d1",), scores=(True,))

    def test_dict_json_roundtrip(self):
        diagnostics = RerankDiagnostics(
            document_ids=("d2", "d1"), scores=(4.0, -1.5)
        )
        payload = json.dumps(diagnostics.to_dict())
        assert RerankDiagnostics.from_dict(json.loads(payload)) == diagnostics

    def test_result_without_diagnostics_still_valid(self):
        # Backward compatibility: diagnostics default to None.
        result = RerankResult(query="q", document_ids=("d1",), scores=(1.0,))
        assert result.diagnostics is None
        assert RerankResult.from_dict(result.to_dict()) == result

    def test_result_dict_roundtrip_with_diagnostics(self):
        result = _cross({"d1": 0.1, "d2": 4.0}).rerank(
            "q", _candidates("d1", "d2"), top_k=1
        )
        assert result.diagnostics is not None
        assert RerankResult.from_dict(result.to_dict()) == result

    def test_final_must_equal_diagnostics_prefix(self):
        diagnostics = RerankDiagnostics(
            document_ids=("d2", "d1"), scores=(4.0, 0.1)
        )
        with pytest.raises(ContractValidationError):
            RerankResult(
                query="q",
                document_ids=("d1",),
                scores=(0.1,),
                diagnostics=diagnostics,
            )
        with pytest.raises(ContractValidationError):
            RerankResult(
                query="q",
                document_ids=("d2",),
                scores=(9.9,),
                diagnostics=diagnostics,
            )
        ok = RerankResult(
            query="q",
            document_ids=("d2",),
            scores=(4.0,),
            diagnostics=diagnostics,
        )
        assert ok.diagnostics == diagnostics

    def test_validate_diagnostics_against_candidates(self):
        diagnostics = RerankDiagnostics(
            document_ids=("d1", "d2"), scores=(1.0, 2.0)
        )
        validate_diagnostics_against_candidates(("d1", "d2"), diagnostics)
        with pytest.raises(ContractValidationError):
            # Length mismatch: not every candidate scored.
            validate_diagnostics_against_candidates(("d1",), diagnostics)
        with pytest.raises(ContractValidationError):
            # Unknown document smuggled through metadata.
            validate_diagnostics_against_candidates(
                ("d1", "d3"), diagnostics
            )
        with pytest.raises(ContractValidationError):
            validate_diagnostics_against_candidates(
                ("d1", "d2"), "not-diagnostics"
            )


class TestCrossEncoderFullRanking:
    def test_full_ranking_preserved_and_final_is_prefix(self):
        # Spec reconstruction example: candidates d1..d4 score
        # 0.1/4.0/-2.0/2.0 -> full d2,d4,d1,d3, final top2 d2,d4.
        reranker = _cross({"d1": 0.1, "d2": 4.0, "d3": -2.0, "d4": 2.0})
        result = reranker.rerank(
            "q", _candidates("d1", "d2", "d3", "d4"), top_k=2
        )
        assert result.diagnostics is not None
        assert result.diagnostics.document_ids == ("d2", "d4", "d1", "d3")
        assert result.diagnostics.scores == (4.0, 2.0, 0.1, -2.0)
        assert result.document_ids == ("d2", "d4")
        assert result.scores == (4.0, 2.0)
        assert result.document_ids == result.diagnostics.document_ids[:2]
        assert result.scores == result.diagnostics.scores[:2]

    def test_dropped_scores_preserved(self):
        reranker = _cross(
            {"d1": 5.0, "d2": 4.0, "d3": 3.0, "d4": 2.0, "d5": 1.0},
            candidate_top_k=5,
        )
        result = reranker.rerank(
            "q", _candidates("d1", "d2", "d3", "d4", "d5"), top_k=2
        )
        assert len(result.diagnostics.document_ids) == 5
        assert len(result.document_ids) == 2
        # Ranks 3-5 stay recoverable with scores.
        assert result.diagnostics.document_ids[2:] == ("d3", "d4", "d5")
        assert result.diagnostics.scores[2:] == (3.0, 2.0, 1.0)

    def test_cutoff_scores_visible(self):
        reranker = _cross({"d1": 0.1, "d2": 4.0, "d3": -2.0, "d4": 2.0})
        result = reranker.rerank(
            "q", _candidates("d1", "d2", "d3", "d4"), top_k=2
        )
        # Rank 3 (last kept-adjacent) and rank 4 explain the final cutoff.
        assert result.diagnostics.scores[2] == 0.1
        assert result.diagnostics.scores[3] == -2.0

    def test_tie_orders_by_document_id(self):
        reranker = _cross({"d2": 1.0, "d1": 1.0})
        result = reranker.rerank("q", _candidates("d2", "d1"), top_k=2)
        assert result.diagnostics.document_ids == ("d1", "d2")
        assert result.document_ids == ("d1", "d2")

    def test_negative_scores_preserved_in_diagnostics(self):
        reranker = _cross({"d1": -5.0, "d2": -1.0, "d3": -3.0})
        result = reranker.rerank(
            "q", _candidates("d1", "d2", "d3"), top_k=1
        )
        assert result.diagnostics.document_ids == ("d2", "d3", "d1")
        assert result.diagnostics.scores == (-1.0, -3.0, -5.0)
        assert result.document_ids == ("d2",)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_outside_final_still_rejects(self, bad):
        # The 8th-candidate analogue: a non-finite score that would never
        # reach the final top-k must still reject the whole rerank.
        reranker = _cross({"d1": 5.0, "d2": 4.0, "d3": bad})
        with pytest.raises(ContractValidationError):
            reranker.rerank("q", _candidates("d1", "d2", "d3"), top_k=2)

    def test_single_scoring_pass(self):
        calls = []

        def counting_scorer(query, candidates):
            calls.append((query, tuple(c.document_id for c in candidates)))
            return [1.0 for _ in candidates]

        reranker = CrossEncoderReranker(scorer=counting_scorer)
        result = reranker.rerank(
            "q", _candidates("d1", "d2", "d3", "d4"), top_k=2
        )
        assert len(calls) == 1
        assert calls[0][1] == ("d1", "d2", "d3", "d4")
        assert len(result.diagnostics.document_ids) == 4

    def test_fewer_candidates_than_candidate_top_k(self):
        reranker = _cross({"d1": 2.0, "d2": 1.0})
        result = reranker.rerank("q", _candidates("d1", "d2"), top_k=2)
        # Diagnostics cover the actual candidates, never padded to top_k.
        assert result.diagnostics.document_ids == ("d1", "d2")
        assert len(result.diagnostics.scores) == 2

    def test_empty_candidates(self):
        reranker = _cross({})
        result = reranker.rerank("q", (), top_k=3)
        assert result.document_ids == ()
        assert result.diagnostics is not None
        assert result.diagnostics.document_ids == ()


class TestIdentityFullRanking:
    def test_full_diagnostics_equal_candidate_order(self):
        reranker = IdentityReranker(candidate_top_k=10)
        candidates = _candidates("d1", "d2", "d3", "d4")
        result = reranker.rerank("q", candidates, top_k=2)
        assert result.diagnostics is not None
        assert result.diagnostics.document_ids == ("d1", "d2", "d3", "d4")
        assert result.diagnostics.scores == (4.0, 3.0, 2.0, 1.0)
        assert result.document_ids == ("d1", "d2")
        assert result.scores == (4.0, 3.0)
        assert result.document_ids == result.diagnostics.document_ids[:2]
        assert result.scores == result.diagnostics.scores[:2]

    def test_full_scores_preserved(self):
        reranker = IdentityReranker(candidate_top_k=10)
        result = reranker.rerank(
            "q", _candidates("d1", "d2", scores=(0.5, 9.9)), top_k=1
        )
        assert result.diagnostics.scores == (0.5, 9.9)
        assert result.scores == (0.5,)

    def test_empty_candidates(self):
        result = IdentityReranker().rerank("q", (), top_k=3)
        assert result.diagnostics is not None
        assert result.diagnostics.document_ids == ()


class TestRankMovementReconstruction:
    def test_movement_derivable_offline(self):
        # Candidate order d1,d2,d3,d4; full rerank d2,d4,d1,d3.
        reranker = _cross({"d1": 0.1, "d2": 4.0, "d3": -2.0, "d4": 2.0})
        candidates = _candidates("d1", "d2", "d3", "d4")
        result = reranker.rerank("q", candidates, top_k=2)
        candidate_rank = {
            candidate.document_id: index + 1
            for index, candidate in enumerate(candidates)
        }
        reranker_rank = {
            document_id: index + 1
            for index, document_id in enumerate(
                result.diagnostics.document_ids
            )
        }
        movement = {
            document_id: (
                candidate_rank[document_id],
                reranker_rank[document_id],
            )
            for document_id in candidate_rank
        }
        assert movement == {
            "d1": (1, 3),
            "d2": (2, 1),
            "d3": (3, 4),
            "d4": (4, 2),
        }


class TestPipelineDiagnosticTrace:
    def _run_identity(self):
        retriever = _StubRetriever(("d1", "d2", "d3", "d4"))
        corpus = _corpus(
            ("d1", "a"), ("d2", "b"), ("d3", "c"), ("d4", "d")
        )
        reranker = IdentityReranker(candidate_top_k=4)
        pipeline = DeterministicRAGPipeline(
            retriever=retriever, corpus=corpus, top_k=2, reranker=reranker
        )
        return pipeline.run("q")

    def test_candidate_event_unchanged(self):
        result = self._run_identity()
        assert result.candidate_event is not None
        assert result.candidate_event.role is RetrievalRole.CANDIDATE
        assert result.candidate_event.document_ids == ("d1", "d2", "d3", "d4")
        assert result.candidate_event.scores == (4.0, 3.0, 2.0, 1.0)

    def test_final_event_remains_top_k_only(self):
        result = self._run_identity()
        assert result.retrieval_event.role is RetrievalRole.FINAL
        assert result.retrieval_event.document_ids == ("d1", "d2")
        assert result.retrieval_event.scores == (4.0, 3.0)

    def test_only_two_events_no_third_event(self):
        result = self._run_identity()
        assert result.retrieval_events == (
            result.candidate_event,
            result.retrieval_event,
        )

    def test_full_ranking_metadata_added(self):
        result = self._run_identity()
        full = result.runtime_metadata["reranker"]["full_ranking"]
        assert full["document_ids"] == ["d1", "d2", "d3", "d4"]
        assert full["scores"] == [4.0, 3.0, 2.0, 1.0]
        assert result.runtime_metadata["reranker"]["candidate_top_k"] == 4
        assert result.runtime_metadata["reranker"]["final_top_k"] == 2

    def test_cross_encoder_full_ranking_metadata(self):
        retriever = _StubRetriever(("d1", "d2", "d3", "d4"))
        corpus = _corpus(
            ("d1", "a"), ("d2", "b"), ("d3", "c"), ("d4", "d")
        )
        reranker = _cross(
            {"d1": 0.1, "d2": 4.0, "d3": -2.0, "d4": 2.0},
            candidate_top_k=4,
        )
        pipeline = DeterministicRAGPipeline(
            retriever=retriever, corpus=corpus, top_k=2, reranker=reranker
        )
        result = pipeline.run("q")
        assert result.retrieval_event.document_ids == ("d2", "d4")
        assert result.retrieval_event.scores == (4.0, 2.0)
        full = result.runtime_metadata["reranker"]["full_ranking"]
        assert full["document_ids"] == ["d2", "d4", "d1", "d3"]
        assert full["scores"] == [4.0, 2.0, 0.1, -2.0]
        # Final event stays consistent with the full-ranking prefix.
        assert list(result.retrieval_event.document_ids) == full[
            "document_ids"
        ][:2]
        assert list(result.retrieval_event.scores) == full["scores"][:2]

    def test_metadata_json_compatible(self):
        result = self._run_identity()
        payload = json.dumps(result.runtime_metadata)
        restored = json.loads(payload)
        assert restored["reranker"]["full_ranking"]["document_ids"] == [
            "d1",
            "d2",
            "d3",
            "d4",
        ]
        for score in restored["reranker"]["full_ranking"]["scores"]:
            assert type(score) is float

    def test_hybrid_metadata_preserved(self):
        class HybridLike(_StubRetriever):
            retriever_id = "hybrid-bm25-minilm-rrf-v0"
            retriever_version = "0.1"

            def runtime_metadata(self):
                return {
                    "hybrid": {
                        "fusion_method": "rrf",
                        "rrf_k": 60,
                        "component_top_k": 4,
                        "final_top_k": 2,
                        "components": {},
                    }
                }

        retriever = HybridLike(("d1", "d2", "d3", "d4"))
        corpus = _corpus(
            ("d1", "a"), ("d2", "b"), ("d3", "c"), ("d4", "d")
        )
        pipeline = DeterministicRAGPipeline(
            retriever=retriever,
            corpus=corpus,
            top_k=2,
            reranker=IdentityReranker(candidate_top_k=4),
        )
        result = pipeline.run("q")
        assert result.runtime_metadata["hybrid"]["fusion_method"] == "rrf"
        assert "full_ranking" in result.runtime_metadata["reranker"]

    def test_trace_json_roundtrip_preserves_diagnostics(self):
        retriever = _StubRetriever(("d1", "d2", "d3"))
        corpus = _corpus(("d1", "a"), ("d2", "b"), ("d3", "c"))
        reranker = _cross(
            {"d1": -1.0, "d2": 5.0, "d3": 0.0}, candidate_top_k=3
        )
        pipeline = DeterministicRAGPipeline(
            retriever=retriever, corpus=corpus, top_k=2, reranker=reranker
        )
        system = DeterministicRAGSystem(pipeline=pipeline)
        trace = run_case(_case(), _manifest(), system)
        restored = RunTrace.from_json(trace.to_json())
        assert restored == trace
        full = restored.runtime_metadata["reranker"]["full_ranking"]
        assert full["document_ids"] == ["d2", "d3", "d1"]
        assert full["scores"] == [5.0, 0.0, -1.0]

    def test_evaluation_unchanged_final_only(self):
        # Candidate ranks d3 first, final ranks d3 last: MRR must follow final.
        candidate = RetrievalEvent(
            query="q",
            document_ids=("d3", "d1", "d2"),
            scores=(3.0, 2.0, 1.0),
            role=RetrievalRole.CANDIDATE,
        )
        final = RetrievalEvent(
            query="q",
            document_ids=("d1", "d2", "d3"),
            scores=(5.0, 0.0, -1.0),
            role=RetrievalRole.FINAL,
        )
        retriever = _StubRetriever(("d1", "d2"))
        corpus = _corpus(("d1", "a"), ("d2", "b"))
        system = DeterministicRAGSystem(
            pipeline=DeterministicRAGPipeline(
                retriever=retriever, corpus=corpus, top_k=2
            )
        )
        trace = run_case(_case(relevant=("d3",)), _manifest(), system)
        import dataclasses

        trace = dataclasses.replace(trace, retrieval_events=(candidate, final))
        result = DeterministicRAGEvaluator(ks=(1, 3)).evaluate(
            _case(relevant=("d3",)), trace
        )
        assert ranked_document_ids(trace) == ("d1", "d2", "d3")
        assert result.metrics["retrieval_mrr"].value == pytest.approx(1.0 / 3)
        assert result.metrics["retrieval_recall_at_1"].value == 0.0

    def test_generation_still_consumes_final_top1(self):
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
        full = result.runtime_metadata["reranker"]["full_ranking"]
        assert full["document_ids"] == ["d2", "d1"]

    def test_pipeline_rejects_diagnostics_with_unknown_document(self):
        class EvilDiagnostics:
            reranker_id = "evil-v0"
            reranker_version = "0.1"
            candidate_top_k = 2

            def rerank(self, query, candidates, top_k):
                return RerankResult(
                    query=query,
                    document_ids=("d1",),
                    scores=(1.0,),
                    diagnostics=RerankDiagnostics(
                        document_ids=("d1", "d-evil"),
                        scores=(1.0, 99.0),
                    ),
                )

            def runtime_metadata(self):
                return {
                    "reranker_id": self.reranker_id,
                    "reranker_version": self.reranker_version,
                    "candidate_top_k": self.candidate_top_k,
                }

        retriever = _StubRetriever(("d1", "d2"))
        corpus = _corpus(("d1", "a"), ("d2", "b"))
        pipeline = DeterministicRAGPipeline(
            retriever=retriever,
            corpus=corpus,
            top_k=1,
            reranker=IdentityReranker(candidate_top_k=2),
        )
        object.__setattr__(pipeline, "_reranker", EvilDiagnostics())
        with pytest.raises(ContractValidationError):
            pipeline.run("q")

    def test_pipeline_rejects_diagnostics_missing_candidate(self):
        class ShortDiagnostics:
            reranker_id = "short-v0"
            reranker_version = "0.1"
            candidate_top_k = 2

            def rerank(self, query, candidates, top_k):
                return RerankResult(
                    query=query,
                    document_ids=("d1",),
                    scores=(1.0,),
                    diagnostics=RerankDiagnostics(
                        document_ids=("d1",),
                        scores=(1.0,),
                    ),
                )

            def runtime_metadata(self):
                return {
                    "reranker_id": self.reranker_id,
                    "reranker_version": self.reranker_version,
                    "candidate_top_k": self.candidate_top_k,
                }

        retriever = _StubRetriever(("d1", "d2"))
        corpus = _corpus(("d1", "a"), ("d2", "b"))
        pipeline = DeterministicRAGPipeline(
            retriever=retriever,
            corpus=corpus,
            top_k=1,
            reranker=IdentityReranker(candidate_top_k=2),
        )
        object.__setattr__(pipeline, "_reranker", ShortDiagnostics())
        with pytest.raises(ContractValidationError):
            pipeline.run("q")


class TestFingerprintStable:
    @pytest.mark.parametrize(
        "filename,expected", sorted(FROZEN_FINGERPRINTS.items())
    )
    def test_diagnostic_phase_leaves_fingerprints_unchanged(
        self, filename, expected
    ):
        manifest = ExperimentManifest.from_json(
            (RAG_V0_DIR / "experiments" / filename).read_text(encoding="utf-8")
        )
        assert manifest.config_fingerprint() == expected
