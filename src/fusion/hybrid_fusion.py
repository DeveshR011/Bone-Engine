"""
Hybrid Fusion
=============
Merges results from sparse, dense, and graph retrieval systems using:
  A) Weighted linear combination of normalized scores
  B) Weighted Reciprocal Rank Fusion (RRF)

Design: all merge functions accept lists of (doc_id, score) result lists
and produce a unified ranked list.

Normalization contract
----------------------
Retrieved documents are normalized into ``[SCORE_FLOOR, 1.0]`` rather than
``[0.0, 1.0]``. This keeps two cases distinguishable:

  * a document a retriever ranked **last**  -> receives ``SCORE_FLOOR``
  * a document a retriever **never returned** -> contributes nothing

Collapsing the worst-ranked document to exactly 0.0 makes those two cases
identical, which lets a single confident retriever be outvoted by documents
that another retriever actively ranked at the bottom. That defect made fused
rankings score *below* their own best input.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

# Minimum normalized credit given to a retrieved document. Must be > 0 so that
# "ranked last" outranks "not retrieved at all".
SCORE_FLOOR = 0.1


class HybridFusion:
    """Fuse multiple ranked lists into a single ranking."""

    def __init__(
        self,
        strategy: str = "rrf",
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

        Scores are normalized within each result list before combining, since
        BM25 sums, SPLADE dot products, and cosine similarities live on
        incomparable scales.
        """
        weighted = [
            (self._normalize_scores(sparse_results), self.alpha),
            (self._normalize_scores(dense_results), self.beta),
            (self._normalize_scores(graph_results or []), self.gamma),
        ]

        all_docs: dict[str, float] = defaultdict(float)
        for scores, weight in weighted:
            for doc_id, s in scores.items():
                all_docs[doc_id] += weight * s

        content_map = self._collect_content(sparse_results, dense_results, graph_results)

        ranked = sorted(all_docs.items(), key=lambda kv: kv[1], reverse=True)
        return [
            {"doc_id": doc_id, "score": score, "content": content_map.get(doc_id, "")}
            for doc_id, score in ranked
        ]

    # ------------------------------------------------------------------
    # Reciprocal Rank Fusion
    # ------------------------------------------------------------------

    def _rrf_fusion(
        self,
        sparse_results: list[dict[str, Any]],
        dense_results: list[dict[str, Any]],
        graph_results: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        """Weighted RRF:  score = Σ_i  w_i / (k + rank_i).

        Rank-based, so it is immune to the score-scale mismatch between
        retrievers — the reason it is the default strategy. Weights let a
        retriever known to be stronger on a corpus contribute more without
        reintroducing any dependence on raw score magnitude.
        """
        k = self.rrf_k
        fused: dict[str, float] = defaultdict(float)

        for result_list, weight in [
            (sparse_results, self.alpha),
            (dense_results, self.beta),
            (graph_results or [], self.gamma),
        ]:
            for rank, r in enumerate(result_list, start=1):
                fused[r["doc_id"]] += weight / (k + rank)

        content_map = self._collect_content(sparse_results, dense_results, graph_results)

        ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
        return [
            {"doc_id": doc_id, "score": score, "content": content_map.get(doc_id, "")}
            for doc_id, score in ranked
        ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_content(*result_lists: list[dict[str, Any]] | None) -> dict[str, str]:
        """Gather doc content from whichever retriever happened to carry it."""
        content_map: dict[str, str] = {}
        for results in result_lists:
            for r in results or []:
                if r.get("content"):
                    content_map[r["doc_id"]] = r["content"]
        return content_map

    @staticmethod
    def _normalize_scores(
        results: list[dict[str, Any]],
    ) -> dict[str, float]:
        """Min-max normalize scores into [SCORE_FLOOR, 1.0].

        Documents absent from ``results`` are simply missing from the returned
        mapping and contribute 0.0 downstream — strictly less than the
        SCORE_FLOOR floor given to the worst *retrieved* document.
        """
        if not results:
            return {}

        scores = [r["score"] for r in results]
        mn, mx = min(scores), max(scores)

        if mx == mn:
            # A single result, or a tie across all results: no ordering
            # information to preserve, so give every document full credit.
            return {r["doc_id"]: 1.0 for r in results}

        rng = mx - mn
        span = 1.0 - SCORE_FLOOR
        return {
            r["doc_id"]: SCORE_FLOOR + span * ((r["score"] - mn) / rng)
            for r in results
        }
