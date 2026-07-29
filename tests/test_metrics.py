"""Tests for IR metrics against hand-computed values.

Every expected number here is derived by hand from the metric definition so
the tests fail if the implementation drifts, rather than merely re-encoding
whatever the code currently produces.
"""

from __future__ import annotations

import math

import pytest

from src.evaluation.metrics import EvaluationMetrics


@pytest.fixture
def m():
    return EvaluationMetrics()


class TestRecall:
    def test_perfect_recall(self, m):
        assert m.recall_at_k(["d1", "d2", "d3"], {"d1", "d2"}, k=3) == pytest.approx(1.0)

    def test_partial_recall(self, m):
        assert m.recall_at_k(["d1", "x", "y"], {"d1", "d2"}, k=3) == pytest.approx(0.5)

    def test_cutoff_is_respected(self, m):
        # d2 sits at rank 3, outside k=2
        assert m.recall_at_k(["d1", "x", "d2"], {"d1", "d2"}, k=2) == pytest.approx(0.5)

    def test_no_relevant_docs_returns_zero(self, m):
        assert m.recall_at_k(["d1"], set(), k=10) == 0.0

    def test_empty_retrieval(self, m):
        assert m.recall_at_k([], {"d1"}, k=10) == 0.0

    def test_duplicates_do_not_inflate(self, m):
        assert m.recall_at_k(["d1", "d1", "d1"], {"d1", "d2"}, k=3) == pytest.approx(0.5)


class TestNDCG:
    def test_perfect_ranking_is_one(self, m):
        assert m.ndcg_at_k(["d1", "d2"], {"d1": 1, "d2": 1}, k=2) == pytest.approx(1.0)

    def test_known_value_single_relevant_at_rank_two(self, m):
        # DCG = 1/log2(3); IDCG = 1/log2(2) = 1
        expected = (1 / math.log2(3)) / 1.0
        assert m.ndcg_at_k(["x", "d1"], {"d1": 1}, k=10) == pytest.approx(expected)

    def test_graded_relevance_uses_exponential_gain(self, m):
        # retrieved: grade 1 then grade 3
        dcg = (2**1 - 1) / math.log2(2) + (2**3 - 1) / math.log2(3)
        idcg = (2**3 - 1) / math.log2(2) + (2**1 - 1) / math.log2(3)
        assert m.ndcg_at_k(["a", "b"], {"a": 1, "b": 3}, k=10) == pytest.approx(dcg / idcg)

    def test_no_relevant_retrieved_is_zero(self, m):
        assert m.ndcg_at_k(["x", "y"], {"d1": 1}, k=10) == 0.0

    def test_empty_relevance_is_zero(self, m):
        assert m.ndcg_at_k(["d1"], {}, k=10) == 0.0

    def test_ranking_order_matters(self, m):
        good = m.ndcg_at_k(["d1", "x"], {"d1": 1}, k=10)
        bad = m.ndcg_at_k(["x", "d1"], {"d1": 1}, k=10)
        assert good > bad

    def test_idcg_respects_cutoff(self, m):
        """With 3 relevant docs but k=1, IDCG must use only the top 1."""
        assert m.ndcg_at_k(["d1"], {"d1": 1, "d2": 1, "d3": 1}, k=1) == pytest.approx(1.0)


class TestMRR:
    def test_first_position(self, m):
        assert m.mrr(["d1", "x"], {"d1"}) == pytest.approx(1.0)

    def test_third_position(self, m):
        assert m.mrr(["x", "y", "d1"], {"d1"}) == pytest.approx(1 / 3)

    def test_uses_first_relevant_only(self, m):
        assert m.mrr(["x", "d2", "d1"], {"d1", "d2"}) == pytest.approx(0.5)

    def test_no_hit_is_zero(self, m):
        assert m.mrr(["x", "y"], {"d1"}) == 0.0

    def test_empty_retrieval(self, m):
        assert m.mrr([], {"d1"}) == 0.0


class TestBatchEvaluation:
    def test_averages_across_queries(self, m):
        results = [
            (["d1", "x"], {"d1"}),   # recall@5 = 1.0, mrr = 1.0
            (["x", "y"], {"d2"}),    # recall@5 = 0.0, mrr = 0.0
        ]
        out = m.evaluate_queries(results, k_values=[5])
        assert out["recall@5"] == pytest.approx(0.5)
        assert out["mrr"] == pytest.approx(0.5)

    def test_empty_input_returns_empty(self, m):
        assert m.evaluate_queries([], k_values=[5]) == {}

    def test_reports_every_requested_cutoff(self, m):
        out = m.evaluate_queries([(["d1"], {"d1"})], k_values=[1, 5, 10])
        for k in (1, 5, 10):
            assert f"recall@{k}" in out
            assert f"ndcg@{k}" in out
        assert "mrr" in out
