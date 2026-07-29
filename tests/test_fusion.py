"""Tests for hybrid fusion.

These pin the semantics that were previously broken: a document that a
retriever ranked *last* must still score above a document that retriever
never returned at all, and fusing strong rankers must not perform worse
than the strongest input.
"""

from __future__ import annotations

import pytest

from src.fusion.hybrid_fusion import HybridFusion


def ids(results):
    return [r["doc_id"] for r in results]


def make(pairs):
    return [{"doc_id": d, "score": s, "content": f"content of {d}"} for d, s in pairs]


class TestNormalization:
    def test_last_ranked_doc_scores_above_absent_doc(self):
        """Regression: min-max mapped the worst doc to exactly 0.0, making it
        indistinguishable from a doc the retriever never returned."""
        scores = HybridFusion._normalize_scores(make([("a", 10.0), ("b", 5.0), ("c", 1.0)]))

        assert scores["c"] > 0.0, "worst retrieved doc must keep positive credit"
        assert scores["a"] == pytest.approx(1.0)
        assert scores["a"] > scores["b"] > scores["c"]

    def test_single_result_gets_full_credit(self):
        scores = HybridFusion._normalize_scores(make([("a", 3.7)]))
        assert scores["a"] == pytest.approx(1.0)

    def test_identical_scores_are_not_zeroed(self):
        scores = HybridFusion._normalize_scores(make([("a", 2.0), ("b", 2.0)]))
        assert scores["a"] == scores["b"] > 0.0

    def test_handles_negative_scores(self):
        """Cosine similarity can be negative; normalization must not invert order."""
        scores = HybridFusion._normalize_scores(make([("a", 0.4), ("b", -0.1), ("c", -0.8)]))
        assert scores["a"] > scores["b"] > scores["c"] > 0.0

    def test_empty_list(self):
        assert HybridFusion._normalize_scores([]) == {}


class TestLinearFusion:
    def test_absent_doc_loses_to_last_ranked_doc(self):
        """Regression: absent and last-ranked were both worth exactly 0.0.

        d_absent is ranked #1 by sparse but never returned by dense. d1 is
        ranked #1 by dense and *last* by sparse. Under the old normalization
        both received 0.0 from the list that disagreed, so the tie was decided
        arbitrarily; d1 must now win on its sparse floor credit.
        """
        fusion = HybridFusion(strategy="linear", alpha=0.5, beta=0.5, gamma=0.0)
        sparse = make([("d_absent", 9.0), ("d2", 8.5), ("d1", 8.0)])
        dense = make([("d1", 0.95), ("d2", 0.40)])

        fused = fusion.fuse(sparse, dense)

        scores = {r["doc_id"]: r["score"] for r in fused}
        assert scores["d1"] > scores["d_absent"]
        assert fused[0]["doc_id"] == "d1"

    def test_agreement_between_retrievers_wins(self):
        fusion = HybridFusion(strategy="linear", alpha=0.5, beta=0.5, gamma=0.0)
        sparse = make([("d1", 10.0), ("d2", 9.0)])
        dense = make([("d1", 0.9), ("d2", 0.2)])

        fused = fusion.fuse(sparse, dense)
        assert fused[0]["doc_id"] == "d1"

    def test_content_is_preserved_from_any_source(self):
        fusion = HybridFusion(strategy="linear")
        sparse = [{"doc_id": "d1", "score": 1.0}]  # no content
        dense = make([("d1", 0.5)])

        fused = fusion.fuse(sparse, dense)
        assert fused[0]["content"] == "content of d1"

    def test_graph_results_are_optional(self):
        fusion = HybridFusion(strategy="linear")
        fused = fusion.fuse(make([("d1", 1.0)]), make([("d2", 1.0)]), None)
        assert set(ids(fused)) == {"d1", "d2"}

    def test_weights_actually_shift_the_ranking(self):
        sparse = make([("d1", 10.0), ("d2", 1.0)])
        dense = make([("d2", 0.9), ("d1", 0.1)])

        sparse_heavy = HybridFusion(strategy="linear", alpha=0.9, beta=0.1, gamma=0.0)
        dense_heavy = HybridFusion(strategy="linear", alpha=0.1, beta=0.9, gamma=0.0)

        assert sparse_heavy.fuse(sparse, dense)[0]["doc_id"] == "d1"
        assert dense_heavy.fuse(sparse, dense)[0]["doc_id"] == "d2"


class TestRRFFusion:
    def test_rrf_is_rank_based_not_score_based(self):
        """RRF must ignore score magnitude entirely."""
        fusion = HybridFusion(strategy="rrf", rrf_k=60)

        a = fusion.fuse(make([("d1", 1000.0), ("d2", 1.0)]), make([("d2", 0.9), ("d1", 0.8)]))
        b = fusion.fuse(make([("d1", 2.0), ("d2", 1.9)]), make([("d2", 0.9), ("d1", 0.8)]))

        assert ids(a) == ids(b)

    def test_known_rrf_values(self):
        # Unit weights reduce weighted RRF to the textbook formula.
        fusion = HybridFusion(strategy="rrf", alpha=1.0, beta=1.0, gamma=0.0, rrf_k=60)
        fused = fusion.fuse(make([("d1", 5.0)]), make([("d1", 0.9)]))
        # present at rank 1 in both lists -> 2 * 1/(60+1)
        assert fused[0]["score"] == pytest.approx(2 / 61)

    def test_weights_scale_contributions(self):
        fusion = HybridFusion(strategy="rrf", alpha=0.4, beta=0.4, gamma=0.0, rrf_k=60)
        fused = fusion.fuse(make([("d1", 5.0)]), make([("d1", 0.9)]))
        assert fused[0]["score"] == pytest.approx(0.8 / 61)

    def test_rrf_ignores_compressed_score_margins(self):
        """RRF depends on ranks alone, never on score spread.

        BM25 scores 9.0/8.5/8.0 are nearly tied while the dense scores are
        widely separated. Linear fusion's min-max stretches that near-tie
        across the full range; RRF sees only mirrored ranks, so d1 and d3 come
        out exactly equal. Breaking such ties is the reranker's job.
        """
        fusion = HybridFusion(strategy="rrf", alpha=0.4, beta=0.4, gamma=0.2)
        sparse = make([("d3", 9.0), ("d2", 8.5), ("d1", 8.0)])
        dense = make([("d1", 0.95), ("d2", 0.40), ("d3", 0.10)])

        scores = {r["doc_id"]: r["score"] for r in fusion.fuse(sparse, dense)}
        assert scores["d1"] == pytest.approx(scores["d3"])

        # Widening only the *scores* must not perturb a rank-based method.
        wider = make([("d3", 900.0), ("d2", 8.5), ("d1", 0.001)])
        rescored = {r["doc_id"]: r["score"] for r in fusion.fuse(wider, dense)}
        assert rescored == pytest.approx(scores)

    def test_doc_in_both_lists_beats_doc_in_one(self):
        fusion = HybridFusion(strategy="rrf", rrf_k=60)
        sparse = make([("d1", 5.0), ("d2", 4.0)])
        dense = make([("d2", 0.9), ("d3", 0.8)])

        fused = fusion.fuse(sparse, dense)
        assert fused[0]["doc_id"] == "d2"

    def test_weighted_rrf_respects_weights(self):
        """A retriever with a higher weight should pull its top doc up."""
        sparse = make([("d1", 5.0), ("d2", 4.0)])
        dense = make([("d2", 0.9), ("d1", 0.8)])

        sparse_heavy = HybridFusion(strategy="rrf", alpha=0.9, beta=0.1, gamma=0.0)
        dense_heavy = HybridFusion(strategy="rrf", alpha=0.1, beta=0.9, gamma=0.0)

        assert sparse_heavy.fuse(sparse, dense)[0]["doc_id"] == "d1"
        assert dense_heavy.fuse(sparse, dense)[0]["doc_id"] == "d2"


class TestFusionContract:
    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown fusion strategy"):
            HybridFusion(strategy="bogus").fuse([], [])

    def test_output_is_sorted_descending(self):
        fusion = HybridFusion(strategy="linear")
        fused = fusion.fuse(make([("a", 1.0), ("b", 5.0), ("c", 3.0)]), [])
        scores = [r["score"] for r in fused]
        assert scores == sorted(scores, reverse=True)

    def test_no_duplicate_doc_ids(self):
        fusion = HybridFusion(strategy="linear")
        fused = fusion.fuse(make([("d1", 1.0)]), make([("d1", 1.0)]), make([("d1", 1.0)]))
        assert len(fused) == 1

    def test_all_empty_inputs(self):
        assert HybridFusion(strategy="linear").fuse([], [], []) == []
        assert HybridFusion(strategy="rrf").fuse([], [], []) == []
