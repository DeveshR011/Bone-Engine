"""Tests for the retrieval pipeline's index structures and stage wiring.

Uses stub encoders so the search logic is verified without downloading models.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.pipeline.retrieval_pipeline import (
    DenseIndex,
    HybridRetriever,
    RetrievalConfig,
    SparseIndex,
)


def unit(vec):
    v = np.asarray(vec, dtype=np.float32)
    return v / np.linalg.norm(v)


class TestDenseIndex:
    @pytest.fixture
    def index(self):
        embeddings = np.stack([unit([1, 0]), unit([0, 1]), unit([1, 1])])
        return DenseIndex(embeddings, ["d1", "d2", "d3"])

    def test_exact_match_ranks_first(self, index):
        scores, idx = index.search(unit([1, 0]), top_k=3)
        assert idx[0][0] == 0
        assert scores[0][0] == pytest.approx(1.0, abs=1e-6)

    def test_results_are_sorted_descending(self, index):
        scores, _ = index.search(unit([1, 0]), top_k=3)
        assert list(scores[0]) == sorted(scores[0], reverse=True)

    def test_top_k_is_respected(self, index):
        _, idx = index.search(unit([1, 0]), top_k=2)
        assert idx.shape == (1, 2)

    def test_top_k_larger_than_corpus(self, index):
        _, idx = index.search(unit([1, 0]), top_k=99)
        assert idx.shape == (1, 3)

    def test_batch_queries(self, index):
        queries = np.stack([unit([1, 0]), unit([0, 1])])
        _, idx = index.search(queries, top_k=1)
        assert idx.shape == (2, 1)
        assert idx[0][0] == 0
        assert idx[1][0] == 1


class TestSparseIndex:
    @pytest.fixture
    def index(self):
        dicts = [{1: 2.0, 2: 1.0}, {2: 3.0}, {5: 1.0}]
        return SparseIndex.from_sparse_dicts(dicts, ["d1", "d2", "d3"], vocab_size=10)

    def test_term_overlap_ranks_by_weight(self, index):
        scores, idx = index.search({2: 1.0}, top_k=3)
        assert index.doc_ids[idx[0]] == "d2"   # weight 3.0 beats 1.0
        assert scores[0] == pytest.approx(3.0)

    def test_documents_without_overlap_are_excluded(self, index):
        _, idx = index.search({2: 1.0}, top_k=3)
        assert "d3" not in set(index.doc_ids[idx])
        assert len(idx) == 2

    def test_empty_query_returns_nothing(self, index):
        scores, idx = index.search({}, top_k=3)
        assert len(scores) == 0 and len(idx) == 0

    def test_query_matching_nothing(self, index):
        _, idx = index.search({9: 1.0}, top_k=3)
        assert len(idx) == 0

    def test_dot_product_accumulates_across_terms(self, index):
        scores, idx = index.search({1: 1.0, 2: 1.0}, top_k=3)
        assert index.doc_ids[idx[0]] == "d1"   # 2.0 + 1.0 = 3.0
        assert scores[0] == pytest.approx(3.0)


class StubDense:
    """Ranks documents by a fixed, query-independent order."""

    def __init__(self, embeddings):
        self.embeddings = embeddings

    def encode(self, texts, batch_size=64, show_progress=False):
        return self.embeddings

    def encode_queries(self, queries, **kwargs):
        return np.stack([unit([1, 0]) for _ in queries])


class StubReranker:
    """Reverses the candidate order, so its effect is unambiguous."""

    def rerank(self, query, candidates, top_k=None, top_n_to_rerank=None):
        out = [
            {**c, "rerank_score": float(i), "score": float(i)}
            for i, c in enumerate(reversed(candidates))
        ]
        return out[:top_k] if top_k else out


class TestHybridRetrieverStages:
    @pytest.fixture
    def retriever(self):
        embeddings = np.stack([unit([1, 0]), unit([0.9, 0.1]), unit([0, 1])])
        r = HybridRetriever(
            dense_encoder=StubDense(embeddings),
            splade_encoder=None,
            reranker=None,
            config=RetrievalConfig(top_k=3, final_k=3),
        )
        r.index(["d1", "d2", "d3"], ["text one", "text two", "text three"])
        return r

    def test_dense_only_pipeline_reports_dense_stage(self, retriever):
        stages = retriever.search(["a query"], return_stages=True)
        assert "dense" in stages
        assert stages["dense"][0][0]["doc_id"] == "d1"

    def test_no_hybrid_stage_without_sparse_branch(self, retriever):
        stages = retriever.search(["a query"], return_stages=True)
        assert "hybrid" not in stages

    def test_results_carry_content(self, retriever):
        stages = retriever.search(["a query"], return_stages=True)
        assert stages["dense"][0][0]["content"] == "text one"

    def test_reranker_stage_is_applied(self, retriever):
        retriever.reranker = StubReranker()
        stages = retriever.search(["a query"], return_stages=True)

        # Only a dense index exists, so the stage is named for its source.
        assert "dense+rerank" in stages
        # The stub reverses order, so the dense winner must drop to last.
        assert stages["dense+rerank"][0][0]["doc_id"] == "d3"

    def test_reranking_does_not_shorten_the_candidate_list(self, retriever):
        """Regression: truncating to final_k depressed recall@k for k > final_k,
        making the rerank stage incomparable to the stages above it."""
        retriever.reranker = StubReranker()
        retriever.config.final_k = 1

        stages = retriever.search(["a query"], return_stages=True)
        assert len(stages["dense+rerank"][0]) == len(stages["dense"][0])

    def test_empty_query_list_does_not_crash(self, retriever):
        """Regression: stage selection indexed into the first query's hits."""
        stages = retriever.search([], return_stages=True)
        assert stages["dense"] == []

    def test_return_stages_false_yields_only_final(self, retriever):
        retriever.reranker = StubReranker()
        out = retriever.search(["a query"], return_stages=False)
        assert list(out) == ["dense+rerank"]
