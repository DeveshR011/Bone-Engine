"""
Evaluation Metrics
==================
Implements standard IR metrics:
  - Recall@K
  - nDCG@K
  - MRR (Mean Reciprocal Rank)

Also provides memory / latency measurement helpers and CSV result export.
"""

from __future__ import annotations

import math
import os
from dataclasses import asdict
from typing import Any

import numpy as np
import psutil

from ..utils.logging_utils import ExperimentResult, get_logger, log_experiment

logger = get_logger(__name__)


class EvaluationMetrics:
    """Compute and log IR evaluation metrics."""

    # ------------------------------------------------------------------
    # Core metrics
    # ------------------------------------------------------------------

    @staticmethod
    def recall_at_k(
        retrieved: list[str],
        relevant: set[str],
        k: int = 10,
    ) -> float:
        """Recall@K = |retrieved_top_k ∩ relevant| / |relevant|.

        Args:
            retrieved: Ordered list of doc_ids from retrieval.
            relevant: Set of relevant doc_ids (ground truth).
            k: Cutoff.
        """
        if not relevant:
            return 0.0
        top_k = set(retrieved[:k])
        return len(top_k & relevant) / len(relevant)

    @staticmethod
    def ndcg_at_k(
        retrieved: list[str],
        relevance: dict[str, int],
        k: int = 10,
    ) -> float:
        """Normalized Discounted Cumulative Gain @ K.

        Args:
            retrieved: Ordered list of doc_ids.
            relevance: {doc_id: relevance_grade} (e.g. 0, 1, 2, 3).
            k: Cutoff.
        """
        dcg = 0.0
        for i, doc_id in enumerate(retrieved[:k]):
            rel = relevance.get(doc_id, 0)
            dcg += (2 ** rel - 1) / math.log2(i + 2)  # i+2 because rank is 1-indexed

        # Ideal DCG
        ideal_rels = sorted(relevance.values(), reverse=True)[:k]
        idcg = sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(ideal_rels))

        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def mrr(
        retrieved: list[str],
        relevant: set[str],
    ) -> float:
        """Mean Reciprocal Rank (for a single query).

        Returns 1/rank of the first relevant document, or 0 if none found.
        """
        for i, doc_id in enumerate(retrieved):
            if doc_id in relevant:
                return 1.0 / (i + 1)
        return 0.0

    # ------------------------------------------------------------------
    # Batch evaluation
    # ------------------------------------------------------------------

    def evaluate_queries(
        self,
        queries_results: list[tuple[list[str], set[str]]],
        k_values: list[int] | None = None,
    ) -> dict[str, float]:
        """Evaluate a batch of queries.

        Args:
            queries_results: List of (retrieved_doc_ids, relevant_doc_ids) tuples.
            k_values: List of K cutoffs for recall/ndcg.

        Returns dict of averaged metrics, e.g.:
            {"recall@10": 0.75, "ndcg@10": 0.68, "mrr": 0.82, ...}
        """
        if k_values is None:
            k_values = [5, 10, 20]

        n = len(queries_results)
        if n == 0:
            return {}

        metrics: dict[str, list[float]] = {}

        for retrieved, relevant in queries_results:
            # Convert relevant set to dict with grade=1 for nDCG
            rel_dict = {d: 1 for d in relevant}

            for k in k_values:
                metrics.setdefault(f"recall@{k}", []).append(
                    self.recall_at_k(retrieved, relevant, k)
                )
                metrics.setdefault(f"ndcg@{k}", []).append(
                    self.ndcg_at_k(retrieved, rel_dict, k)
                )

            metrics.setdefault("mrr", []).append(self.mrr(retrieved, relevant))

        return {name: float(np.mean(vals)) for name, vals in metrics.items()}

    # ------------------------------------------------------------------
    # System measurements
    # ------------------------------------------------------------------

    @staticmethod
    def get_memory_usage_mb() -> float:
        """Current process RSS memory in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)

    @staticmethod
    def get_file_size_mb(path: str) -> float:
        """File size in MB."""
        if os.path.exists(path):
            return os.path.getsize(path) / (1024 * 1024)
        return 0.0

    # ------------------------------------------------------------------
    # Result logging
    # ------------------------------------------------------------------

    def log_result(
        self,
        experiment: str,
        system: str,
        variant: str,
        metrics: dict[str, float],
        latency_ms: float = 0.0,
        index_size_mb: float = 0.0,
        memory_mb: float = 0.0,
        indexing_time_s: float = 0.0,
        docs_per_sec: float = 0.0,
        notes: str = "",
        results_dir: str = "results",
    ) -> None:
        """Log a structured experiment result to CSV."""
        result = ExperimentResult(
            experiment=experiment,
            system=system,
            variant=variant,
            recall_at_5=metrics.get("recall@5", 0.0),
            recall_at_10=metrics.get("recall@10", 0.0),
            recall_at_20=metrics.get("recall@20", 0.0),
            ndcg_at_10=metrics.get("ndcg@10", 0.0),
            mrr=metrics.get("mrr", 0.0),
            query_latency_ms=latency_ms,
            index_size_mb=index_size_mb,
            memory_usage_mb=memory_mb,
            indexing_time_s=indexing_time_s,
            docs_per_sec=docs_per_sec,
            notes=notes,
        )
        path = log_experiment(result, results_dir)
        logger.info("Result logged → %s", path)
