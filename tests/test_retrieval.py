"""Tests for the deterministic retrievers and ranking contract."""

from __future__ import annotations

import pytest

from commercebench.contracts import ContractValidationError, RetrievalConfig
from commercebench.rag import (
    Corpus,
    Document,
    KeywordMatchRetriever,
    RetrievalResult,
    Retriever,
    TermFrequencyRetriever,
    build_retriever,
    normalize_text,
    tokenize,
)


@pytest.fixture
def mini_corpus() -> Corpus:
    return Corpus(
        corpus_id="mini",
        corpus_version="1",
        documents=(
            Document(document_id="d-a", text="alpha beta beta gamma"),
            Document(document_id="d-b", text="alpha beta"),
            Document(document_id="d-c", text="unrelated words"),
        ),
    )


class TestNormalization:
    def test_normalize_text(self):
        assert normalize_text("  Hello   WORLD\n") == "hello world"

    def test_tokenize(self):
        assert tokenize("Hello, WORLD! 30days") == [
            "hello",
            "world",
            "30days",
        ]

    def test_tokenize_empty(self):
        assert tokenize("!!! ---") == []


class TestKeywordMatchRetriever:
    def test_distinct_term_overlap(self, mini_corpus):
        result = KeywordMatchRetriever().retrieve(
            "alpha beta gamma", mini_corpus, top_k=3
        )
        # d-a matches 3 distinct terms, d-b matches 2, d-c matches 0.
        assert result.document_ids == ("d-a", "d-b")
        assert result.scores == (3.0, 2.0)

    def test_repeated_query_term_not_double_counted(self, mini_corpus):
        result = KeywordMatchRetriever().retrieve(
            "beta beta beta", mini_corpus, top_k=3
        )
        assert result.document_ids == ("d-a", "d-b")
        assert result.scores == (1.0, 1.0)

    def test_top_k_limits_results(self, mini_corpus):
        result = KeywordMatchRetriever().retrieve(
            "alpha", mini_corpus, top_k=1
        )
        assert result.document_ids == ("d-a",)
        assert result.scores == (1.0,)

    def test_no_match_empty_result(self, mini_corpus):
        result = KeywordMatchRetriever().retrieve(
            "zzz qqq", mini_corpus, top_k=3
        )
        assert result.document_ids == ()
        assert result.scores == ()

    def test_deterministic_repeated_runs(self, mini_corpus):
        retriever = KeywordMatchRetriever()
        first = retriever.retrieve("alpha beta", mini_corpus, top_k=3)
        second = retriever.retrieve("alpha beta", mini_corpus, top_k=3)
        assert first == second


class TestTermFrequencyRetriever:
    def test_occurrence_counts(self, mini_corpus):
        result = TermFrequencyRetriever().retrieve(
            "beta", mini_corpus, top_k=3
        )
        # d-a contains "beta" twice, d-b once.
        assert result.document_ids == ("d-a", "d-b")
        assert result.scores == (2.0, 1.0)

    def test_repetition_beats_distinct_overlap(self):
        corpus = Corpus(
            corpus_id="c",
            corpus_version="1",
            documents=(
                Document(document_id="broad", text="aa bb cc"),
                Document(document_id="narrow", text="aa aa aa aa"),
            ),
        )
        tf = TermFrequencyRetriever().retrieve("aa bb cc", corpus, top_k=2)
        kw = KeywordMatchRetriever().retrieve("aa bb cc", corpus, top_k=2)
        assert kw.document_ids == ("broad", "narrow")
        assert tf.document_ids == ("narrow", "broad")
        assert tf.scores == (4.0, 3.0)
        assert kw.scores == (3.0, 1.0)

    def test_no_match_empty_result(self, mini_corpus):
        result = TermFrequencyRetriever().retrieve("zzz", mini_corpus, top_k=3)
        assert result.document_ids == ()


class TestRankingContract:
    def test_scores_aligned_and_descending(self, mini_corpus):
        for retriever in (KeywordMatchRetriever(), TermFrequencyRetriever()):
            result = retriever.retrieve("alpha beta", mini_corpus, top_k=3)
            assert len(result.document_ids) == len(result.scores)
            assert list(result.scores) == sorted(
                result.scores, reverse=True
            )

    def test_stable_tie_break_by_document_id(self):
        corpus = Corpus(
            corpus_id="c",
            corpus_version="1",
            documents=(
                Document(document_id="z-doc", text="target term"),
                Document(document_id="a-doc", text="target term"),
            ),
        )
        result = KeywordMatchRetriever().retrieve("target", corpus, top_k=2)
        assert result.document_ids == ("a-doc", "z-doc")

    def test_rank_is_index_plus_one(self, mini_corpus):
        result = KeywordMatchRetriever().retrieve("alpha beta", mini_corpus, 3)
        # rank i+1 semantics: first element is rank 1 (the best score).
        best_index = max(range(len(result.scores)), key=lambda i: result.scores[i])
        assert best_index == 0

    def test_result_query_echoed(self, mini_corpus):
        result = KeywordMatchRetriever().retrieve(
            "alpha", mini_corpus, top_k=3
        )
        assert result.query == "alpha"


class TestValidation:
    def test_empty_query_rejected(self, mini_corpus):
        with pytest.raises(ContractValidationError):
            KeywordMatchRetriever().retrieve("", mini_corpus, top_k=3)

    def test_non_positive_top_k_rejected(self, mini_corpus):
        for retriever in (KeywordMatchRetriever(), TermFrequencyRetriever()):
            with pytest.raises(ContractValidationError):
                retriever.retrieve("alpha", mini_corpus, top_k=0)

    def test_result_length_mismatch_rejected(self):
        with pytest.raises(ContractValidationError):
            RetrievalResult(query="q", document_ids=("a",), scores=())

    def test_protocol_satisfied(self):
        assert isinstance(KeywordMatchRetriever(), Retriever)
        assert isinstance(TermFrequencyRetriever(), Retriever)


class TestBuildRetriever:
    def test_known_ids(self):
        assert isinstance(
            build_retriever(RetrievalConfig(retriever_id="keyword-match-v0")),
            KeywordMatchRetriever,
        )
        assert isinstance(
            build_retriever(
                RetrievalConfig(retriever_id="term-frequency-v0")
            ),
            TermFrequencyRetriever,
        )

    def test_unknown_id_rejected(self):
        with pytest.raises(ContractValidationError, match="unknown"):
            build_retriever(RetrievalConfig(retriever_id="bm25"))
