"""Tests for BEIR scoring, verified against hand-computed trec_eval values."""

from __future__ import annotations

import math

import pytest

from src.evaluation.beir_eval import evaluate_run, format_results_table, results_to_run


class TestResultsToRun:
    def test_converts_ranked_lists(self):
        run = results_to_run({"q1": [{"doc_id": "d1", "score": 2.5}]})
        assert run == {"q1": {"d1": 2.5}}

    def test_empty(self):
        assert results_to_run({}) == {}


class TestEvaluateRun:
    def test_perfect_ranking(self):
        qrels = {"q1": {"d1": 1}}
        run = {"q1": {"d1": 10.0, "d2": 1.0}}
        out = evaluate_run(qrels, run, k_values=[10])
        assert out["ndcg@10"] == pytest.approx(1.0)
        assert out["recall@10"] == pytest.approx(1.0)
        assert out["mrr"] == pytest.approx(1.0)

    def test_relevant_doc_at_rank_two(self):
        qrels = {"q1": {"d1": 1}}
        run = {"q1": {"d2": 10.0, "d1": 5.0}}
        out = evaluate_run(qrels, run, k_values=[10])
        assert out["ndcg@10"] == pytest.approx(1 / math.log2(3), abs=1e-4)
        assert out["mrr"] == pytest.approx(0.5)

    def test_graded_relevance_is_respected(self):
        """A grade-2 doc above a grade-1 doc must beat the reverse."""
        qrels = {"q1": {"d1": 2, "d2": 1}}
        good = evaluate_run(qrels, {"q1": {"d1": 10.0, "d2": 5.0}}, k_values=[10])
        bad = evaluate_run(qrels, {"q1": {"d2": 10.0, "d1": 5.0}}, k_values=[10])
        assert good["ndcg@10"] > bad["ndcg@10"]

    def test_averages_over_queries(self):
        qrels = {"q1": {"d1": 1}, "q2": {"d2": 1}}
        run = {"q1": {"d1": 1.0}, "q2": {"dX": 1.0}}
        out = evaluate_run(qrels, run, k_values=[10])
        assert out["mrr"] == pytest.approx(0.5)

    def test_query_with_no_results_scores_zero_not_dropped(self):
        """A query that returned nothing must count as 0, not vanish."""
        qrels = {"q1": {"d1": 1}, "q2": {"d2": 1}}
        out = evaluate_run(qrels, {"q1": {"d1": 1.0}}, k_values=[10])
        assert out["ndcg@10"] == pytest.approx(0.5)

    def test_unjudged_queries_are_ignored(self):
        """Retrieving for an unjudged query must not dilute the average."""
        qrels = {"q1": {"d1": 1}}
        out = evaluate_run(qrels, {"q1": {"d1": 1.0}, "q_extra": {"dZ": 1.0}}, k_values=[10])
        assert out["ndcg@10"] == pytest.approx(1.0)

    def test_recall_cutoff(self):
        qrels = {"q1": {"d1": 1, "d2": 1}}
        run = {"q1": {"d1": 10.0, "dX": 9.0, "d2": 8.0}}
        out = evaluate_run(qrels, run, k_values=[1, 10])
        assert out["recall@1"] == pytest.approx(0.5)
        assert out["recall@10"] == pytest.approx(1.0)


class TestFormatTable:
    def test_sorts_by_first_metric_descending(self):
        table = format_results_table(
            {"weak": {"ndcg@10": 0.10}, "strong": {"ndcg@10": 0.90}},
            metrics=["ndcg@10"],
        )
        lines = table.splitlines()
        assert lines[2].startswith("strong")
        assert lines[3].startswith("weak")

    def test_empty(self):
        assert format_results_table({}) == "(no results)"
