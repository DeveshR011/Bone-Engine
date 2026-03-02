"""
Hybrid Fusion
=============
Merges results from sparse, dense, and graph retrieval systems using:
  A) Weighted linear combination of normalized scores
  B) Reciprocal Rank Fusion (RRF)

Design: all merge functions accept lists of (doc_id, score) result lists
and produce a unified ranked list.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class HybridFusion:
    """Fuse multiple ranked lists into a single ranking."""

    def __init__(
        self,
        strategy: str = "linear",
        alpha: float = 0.4,       # sparse weight
        beta: float = 0.4,        # dense weight
        gamma: float = 0.2,       # graph weight
        rrf_k: int = 60,
    ) -> None:
        self.strategy = strategy
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.rrf_k = rrf_k

    def fuse(
        self,
        sparse_results: list[dict[str, Any]],
        dense_results: list[dict[str, Any]],
        graph_results: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Fuse ranked lists.

        Each result list contains dicts with at least {doc_id, score}.
        Returns a merged list sorted by fused score, descending.
        """
        if self.strategy == "linear":
            return self._linear_fusion(sparse_results, dense_results, graph_results)
        elif self.strategy == "rrf":
            return self._rrf_fusion(sparse_results, dense_results, graph_results)
        else:
            raise ValueError(f"Unknown fusion strategy: {self.strategy}")

    # ------------------------------------------------------------------
    # Linear combination
    # ------------------------------------------------------------------

    def _linear_fusion(
        self,
        sparse_results: list[dict[str, Any]],
        dense_results: list[dict[str, Any]],
        graph_results: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        """FinalScore = α * sparse_norm + β * dense_norm + γ * graph_norm.

        Scores are min-max normalized within each result list before combining.
        """
        sparse_scores = self._normalize_scores(sparse_results)
        dense_scores = self._normalize_scores(dense_results)
        graph_scores = self._normalize_scores(graph_results or [])

        # Merge all doc_ids
        all_docs: dict[str, dict[str, Any]] = {}

        for doc_id, s in sparse_scores.items():
            all_docs.setdefault(doc_id, {"doc_id": doc_id, "score": 0.0, "content": ""})
            all_docs[doc_id]["score"] += self.alpha * s

        for doc_id, s in dense_scores.items():
            all_docs.setdefault(doc_id, {"doc_id": doc_id, "score": 0.0, "content": ""})
            all_docs[doc_id]["score"] += self.beta * s

        for doc_id, s in graph_scores.items():
            all_docs.setdefault(doc_id, {"doc_id": doc_id, "score": 0.0, "content": ""})
            all_docs[doc_id]["score"] += self.gamma * s

        # Attach content from any source
        content_map = {}
        for r in sparse_results + dense_results + (graph_results or []):
            if r.get("content"):
                content_map[r["doc_id"]] = r["content"]
        for doc in all_docs.values():
            doc["content"] = content_map.get(doc["doc_id"], "")

        ranked = sorted(all_docs.values(), key=lambda x: x["score"], reverse=True)
        return ranked

    # ------------------------------------------------------------------
    # Reciprocal Rank Fusion
    # ------------------------------------------------------------------

    def _rrf_fusion(
        self,
        sparse_results: list[dict[str, Any]],
        dense_results: list[dict[str, Any]],
        graph_results: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        """RRF score = Σ  1 / (k + rank_i)  for each system i."""
        k = self.rrf_k
        fused: dict[str, float] = defaultdict(float)
        content_map: dict[str, str] = {}

        for result_list in [sparse_results, dense_results, graph_results or []]:
            for rank, r in enumerate(result_list, start=1):
                doc_id = r["doc_id"]
                fused[doc_id] += 1.0 / (k + rank)
                if r.get("content"):
                    content_map[doc_id] = r["content"]

        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)
        return [
            {
                "doc_id": doc_id,
                "score": score,
                "content": content_map.get(doc_id, ""),
            }
            for doc_id, score in ranked
        ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_scores(
        results: list[dict[str, Any]],
    ) -> dict[str, float]:
        """Min-max normalize scores to [0, 1]."""
        if not results:
            return {}
        scores = [r["score"] for r in results]
        mn, mx = min(scores), max(scores)
        rng = mx - mn if mx != mn else 1.0
        return {r["doc_id"]: (r["score"] - mn) / rng for r in results}
