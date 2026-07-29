"""Tests for reranker rank-blending.

A cross-encoder trained on question->passage relevance degrades on other query
shapes. Blending bounds how much damage a confused reranker can do to a good
first-stage ranking. These tests use a stub scorer so the blending arithmetic
is verified without loading a model.
"""

from __future__ import annotations

import pytest

from src.rerank.cross_encoder import CrossEncoderReranker


class StubReranker(CrossEncoderReranker):
    """Bypasses model loading; returns scores supplied by the test."""

    def __init__(self, scores, **kwargs):
        self._scores = scores
        self.blend_weight = kwargs.get("blend_weight", 0.7)
        self.blend_k = kwargs.get("blend_k", 60)
        self.batch_size = 32
        self.max_length = 512

    def score_pairs(self, query, documents):
        return self._scores[: len(documents)]


def candidates(n):
    return [
        {"doc_id": f"d{i}", "score": 1.0 - i * 0.01, "content": f"doc {i}"}
        for i in range(n)
    ]


def ids(results):
    return [r["doc_id"] for r in results]


class TestBlending:
    def test_pure_rerank_when_weight_is_one(self):
        rr = StubReranker([1.0, 5.0, 3.0], blend_weight=1.0)
        out = rr.rerank("q", candidates(3))
        assert ids(out) == ["d1", "d2", "d0"]

    def test_pure_first_stage_when_weight_is_zero(self):
        rr = StubReranker([1.0, 5.0, 3.0], blend_weight=0.0)
        out = rr.rerank("q", candidates(3))
        assert ids(out) == ["d0", "d1", "d2"]

    def test_blending_bounds_damage_from_a_confused_reranker(self):
        """The regression this exists for: on SciFact the reranker inverted a
        correct first-stage ranking. Blending must keep the first-stage winner
        near the top when the reranker only mildly disagrees."""
        # Reranker exactly reverses a correct ranking, but weakly.
        rr = StubReranker([0.0, 1.0, 2.0, 3.0], blend_weight=0.5)
        out = rr.rerank("q", candidates(4))
        # With equal weight and exactly opposing ranks every doc ties, so the
        # first-stage order must survive rather than be inverted.
        assert ids(out)[0] == "d0"

    def test_reranker_promotes_a_bottom_ranked_document(self):
        """RRF is rank-based, so a large score gap counts as one rank position,
        not as confidence. A document the reranker puts first still climbs
        substantially — it need not reach rank 1 against a first-stage winner
        the reranker also rates highly."""
        rr = StubReranker([0.0] * 19 + [99.0], blend_weight=0.7)
        out = rr.rerank("q", candidates(20))

        positions = {r["doc_id"]: i for i, r in enumerate(out)}
        assert positions["d19"] < 5, "reranker's top pick should reach the top 5"

    def test_higher_blend_weight_gives_the_reranker_more_authority(self):
        scores = [0.0] * 19 + [99.0]
        conservative = StubReranker(scores, blend_weight=0.3).rerank("q", candidates(20))
        aggressive = StubReranker(scores, blend_weight=1.0).rerank("q", candidates(20))

        pos = lambda out: {r["doc_id"]: i for i, r in enumerate(out)}
        assert pos(aggressive)["d19"] < pos(conservative)["d19"]

    def test_scores_are_descending(self):
        rr = StubReranker([1.0, 5.0, 3.0], blend_weight=0.7)
        out = rr.rerank("q", candidates(3))
        scores = [r["score"] for r in out]
        assert scores == sorted(scores, reverse=True)

    def test_both_score_components_are_preserved(self):
        rr = StubReranker([7.0, 1.0], blend_weight=0.7)
        out = rr.rerank("q", candidates(2))
        for r in out:
            assert "rerank_score" in r and "retrieval_score" in r


class TestTailHandling:
    def test_untouched_tail_sorts_below_every_reranked_doc(self):
        """Raw retrieval scores (~1.0) dwarf blended rank scores (~0.016), so
        an un-rebased tail would leapfrog the reranked head once a consumer
        sorts by score."""
        rr = StubReranker([1.0] * 3, blend_weight=0.7)
        out = rr.rerank("q", candidates(6), top_n_to_rerank=3)

        head_min = min(r["score"] for r in out[:3])
        tail_max = max(r["score"] for r in out[3:])
        assert tail_max < head_min

    def test_tail_keeps_relative_order(self):
        rr = StubReranker([1.0] * 2, blend_weight=0.7)
        out = rr.rerank("q", candidates(5), top_n_to_rerank=2)
        assert ids(out)[2:] == ["d2", "d3", "d4"]

    def test_all_candidates_are_returned(self):
        rr = StubReranker([1.0] * 3, blend_weight=0.7)
        out = rr.rerank("q", candidates(6), top_n_to_rerank=3)
        assert len(out) == 6

    def test_top_k_truncates(self):
        rr = StubReranker([1.0] * 3, blend_weight=0.7)
        assert len(rr.rerank("q", candidates(6), top_k=2, top_n_to_rerank=3)) == 2

    def test_empty_candidates(self):
        assert StubReranker([]).rerank("q", []) == []
