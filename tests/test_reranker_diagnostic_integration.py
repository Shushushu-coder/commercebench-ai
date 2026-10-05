"""Phase 2D diagnostic-trace integration coverage (real pinned model).

Every test here is marked ``integration`` and excluded from the default
``pytest`` run (see ``addopts`` in ``pyproject.toml``). Run them
explicitly with::

    python -m pytest -m integration

These tests prove, against the real
``cross-encoder/ms-marco-MiniLM-L6-v2`` at the pinned revision, that the
full candidate-level reranker ranking persisted in
``runtime_metadata["reranker"]["full_ranking"]`` covers every candidate,
stays sorted (score descending, document_id ascending), agrees with the
final event as its prefix, and costs no second model inference.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from commercebench.contracts import ExperimentManifest, RunTrace
from commercebench.runner import run_case
from commercebench.systems import DeterministicRAGSystem

MODEL_REVISION = "ce0834f22110de6d9222af7a7a03628121708969"
RERANKER_ID = "cross-encoder-msmarco-minilm-l6-v2-v0"
IDENTITY_ID = "identity-reranker-v0"

RAG_V0_DIR = Path(__file__).resolve().parents[1] / "examples" / "rag_v0"

pytestmark = pytest.mark.integration


def _load_manifest(name: str) -> ExperimentManifest:
    return ExperimentManifest.from_json(
        (RAG_V0_DIR / "experiments" / name).read_text(encoding="utf-8")
    )


@pytest.fixture(scope="module")
def identity_manifest() -> ExperimentManifest:
    return _load_manifest("rag_hybrid_identity_rerank_v0.json")


@pytest.fixture(scope="module")
def cross_manifest() -> ExperimentManifest:
    return _load_manifest("rag_hybrid_cross_encoder_v0.json")


@pytest.fixture(scope="module")
def identity_system(identity_manifest, rag_corpus, rag_answers):
    return DeterministicRAGSystem.from_manifest(
        identity_manifest, rag_corpus, rag_answers
    )


@pytest.fixture(scope="module")
def cross_system(cross_manifest, rag_corpus, rag_answers):
    return DeterministicRAGSystem.from_manifest(
        cross_manifest, rag_corpus, rag_answers
    )


def _full_ranking(trace: RunTrace):
    return trace.runtime_metadata["reranker"]["full_ranking"]


class TestRealFullRankingCoverage:
    def test_eleven_cases_full_coverage(
        self, cross_manifest, cross_system, rag_cases
    ):
        assert len(rag_cases) == 11
        for case in rag_cases.values():
            trace = run_case(case, cross_manifest, cross_system)
            candidate, final = trace.retrieval_events
            full = _full_ranking(trace)
            # Every actual candidate is scored exactly once.
            assert len(full["document_ids"]) == len(candidate.document_ids)
            assert len(full["scores"]) == len(candidate.document_ids)
            assert set(full["document_ids"]) == set(candidate.document_ids)
            assert all(math.isfinite(s) for s in full["scores"])
            assert all(type(s) is float for s in full["scores"])

    def test_full_ranking_sorted_with_id_tiebreak(
        self, cross_manifest, cross_system, rag_cases
    ):
        for case in rag_cases.values():
            trace = run_case(case, cross_manifest, cross_system)
            full = _full_ranking(trace)
            pairs = list(zip(full["document_ids"], full["scores"]))
            assert pairs == sorted(pairs, key=lambda item: (-item[1], item[0]))

    def test_final_is_full_prefix_all_cases(
        self, cross_manifest, cross_system, rag_cases
    ):
        for case in rag_cases.values():
            trace = run_case(case, cross_manifest, cross_system)
            candidate, final = trace.retrieval_events
            full = _full_ranking(trace)
            assert len(final.document_ids) <= 3
            assert list(final.document_ids) == full["document_ids"][
                : len(final.document_ids)
            ]
            assert list(final.scores) == pytest.approx(
                full["scores"][: len(final.scores)]
            )
            assert set(final.document_ids) <= set(candidate.document_ids)

    def test_dropped_candidate_scores_recoverable(
        self, cross_manifest, cross_system, rag_cases
    ):
        seen_dropped = False
        for case in rag_cases.values():
            trace = run_case(case, cross_manifest, cross_system)
            candidate, final = trace.retrieval_events
            full = _full_ranking(trace)
            if len(candidate.document_ids) > len(final.document_ids):
                seen_dropped = True
                dropped_ids = full["document_ids"][len(final.document_ids) :]
                dropped_scores = full["scores"][len(final.scores) :]
                assert dropped_ids
                assert len(dropped_ids) == len(dropped_scores)
                assert all(math.isfinite(s) for s in dropped_scores)
                assert set(dropped_ids) <= set(candidate.document_ids)
                assert not (set(dropped_ids) & set(final.document_ids))
        assert seen_dropped, "expected at least one case with dropped candidates"

    def test_json_roundtrip_preserves_full_ranking(
        self, cross_manifest, cross_system, rag_cases
    ):
        for case in rag_cases.values():
            trace = run_case(case, cross_manifest, cross_system)
            restored = RunTrace.from_json(trace.to_json())
            assert restored == trace
            assert (
                restored.runtime_metadata["reranker"]["full_ranking"]
                == trace.runtime_metadata["reranker"]["full_ranking"]
            )


class TestRealIdentityDiagnostics:
    def test_identity_full_ranking_is_candidate_ranking(
        self, identity_manifest, identity_system, rag_cases
    ):
        for case in rag_cases.values():
            trace = run_case(case, identity_manifest, identity_system)
            candidate, final = trace.retrieval_events
            full = _full_ranking(trace)
            assert full["document_ids"] == list(candidate.document_ids)
            assert full["scores"] == list(candidate.scores)
            assert list(final.document_ids) == full["document_ids"][
                : len(final.document_ids)
            ]
            assert list(final.scores) == pytest.approx(
                full["scores"][: len(final.scores)]
            )


class TestRealSingleInference:
    def test_model_loaded_once_and_single_predict_per_case(
        self, cross_manifest, rag_corpus, rag_answers, rag_cases, monkeypatch
    ):
        import sentence_transformers

        real_cls = sentence_transformers.CrossEncoder
        builds: list = []
        predict_calls: list = []
        real_predict = real_cls.predict

        def constructor_spy(*args, **kwargs):
            builds.append(kwargs.get("revision"))
            return real_cls(*args, **kwargs)

        def predict_spy(self, *args, **kwargs):
            predict_calls.append(args)
            return real_predict(self, *args, **kwargs)

        monkeypatch.setattr(
            sentence_transformers, "CrossEncoder", constructor_spy
        )
        monkeypatch.setattr(real_cls, "predict", predict_spy)
        system = DeterministicRAGSystem.from_manifest(
            cross_manifest, rag_corpus, rag_answers
        )
        traces = [
            run_case(case, cross_manifest, system)
            for case in rag_cases.values()
        ]
        reranker_loads = [r for r in builds if r == MODEL_REVISION]
        assert reranker_loads == [MODEL_REVISION]
        # No second predict pass for diagnostics: exactly one model
        # inference per reranked case.
        non_empty = sum(
            1 for trace in traces if trace.retrieval_events[0].document_ids
        )
        assert non_empty == len(traces)
        assert len(predict_calls) == len(traces)


class TestRealKeyCase:
    def test_return_exchange_full_evidence(
        self,
        identity_manifest,
        cross_manifest,
        identity_system,
        cross_system,
        rag_cases,
    ):
        case = rag_cases["rag-return-exchange-001"]
        identity_trace = run_case(case, identity_manifest, identity_system)
        cross_trace = run_case(case, cross_manifest, cross_system)
        candidate = cross_trace.retrieval_events[0]
        assert (
            identity_trace.retrieval_events[0].document_ids
            == candidate.document_ids
        )
        full = _full_ranking(cross_trace)
        final = cross_trace.retrieval_events[1]
        candidate_rank = {
            document_id: index + 1
            for index, document_id in enumerate(candidate.document_ids)
        }
        full_rank = {
            document_id: index + 1
            for index, document_id in enumerate(full["document_ids"])
        }
        full_score = dict(zip(full["document_ids"], full["scores"]))
        final_set = set(final.document_ids)
        # Every candidate reports candidate rank, full reranker rank and
        # score, and final membership — offline, without re-running the
        # model.
        evidence = {
            document_id: {
                "candidate_rank": candidate_rank[document_id],
                "reranker_rank": full_rank[document_id],
                "score": full_score[document_id],
                "in_final": document_id in final_set,
            }
            for document_id in candidate.document_ids
        }
        assert set(evidence) == set(candidate.document_ids)
        assert len(evidence) == len(full["document_ids"])
        for document_id, row in evidence.items():
            assert math.isfinite(row["score"])
            assert row["in_final"] == (row["reranker_rank"] <= 3)
        blob = json.dumps(cross_trace.runtime_metadata["reranker"])
        assert "/home" not in blob
        assert "/Users" not in blob
