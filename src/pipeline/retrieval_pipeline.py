"""
Benchmark Retrieval Pipeline
============================
A self-contained hybrid retrieval pipeline for benchmarking:

    dense (bi-encoder)  ─┐
                         ├─> weighted RRF ─> cross-encoder rerank ─> results
    sparse (SPLADE)     ─┘

Deliberately depends on no external services. Elasticsearch and PostgreSQL are
useful as production backends and as baselines, but requiring a running cluster
to reproduce a benchmark number makes the number harder to trust and harder to
reproduce. Dense search here is exact (brute-force inner product) so retrieval
quality is not confounded by ANN recall loss; ANN indexes trade recall for
speed, which is a separate axis measured separately.

Memory model: both matrices are held in RAM, not VRAM. For the laptop BEIR
subset the largest is TREC-COVID at ~171k docs (~500 MB dense fp32,
~200 MB sparse CSR). Only encoding touches the GPU, in batches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..encoder.dense_encoder import DenseEncoder
from ..encoder.splade_encoder import SpladeEncoder
from ..fusion.hybrid_fusion import HybridFusion
from ..utils.logging_utils import Timer, get_logger

logger = get_logger(__name__)


@dataclass
class RetrievalConfig:
    """Knobs that affect benchmark scores."""

    top_k: int = 100           # candidates retrieved per branch
    rerank_top_n: int = 100    # candidates fed to the cross-encoder
    final_k: int = 10          # results kept for scoring
    dense_batch_size: int = 64
    sparse_batch_size: int = 32
    fusion_strategy: str = "linear"
    alpha: float = 0.4         # sparse weight
    beta: float = 0.6          # dense weight
    rrf_k: int = 60


class DenseIndex:
    """Exact inner-product search over L2-normalized embeddings."""

    def __init__(self, embeddings: np.ndarray, doc_ids: list[str]) -> None:
        self.embeddings = embeddings
        self.doc_ids = np.asarray(doc_ids)

    def search(self, query_vecs: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        """Search a batch of queries. Returns (scores, indices), best-first."""
        if query_vecs.ndim == 1:
            query_vecs = query_vecs[None, :]

        # Embeddings are L2-normalized, so the inner product is cosine
        # similarity and argpartition over it gives the true top-k.
        sims = query_vecs @ self.embeddings.T  # (n_queries, n_docs)
        k = min(top_k, sims.shape[1])

        idx = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        rows = np.arange(sims.shape[0])[:, None]
        part = sims[rows, idx]
        order = np.argsort(-part, axis=1)

        return part[rows, order], idx[rows, order]


class SparseIndex:
    """In-memory SPLADE inverted index backed by a CSR matrix."""

    def __init__(self, matrix, doc_ids: list[str]) -> None:
        self.matrix = matrix  # scipy CSR, shape (n_docs, vocab_size)
        self.doc_ids = np.asarray(doc_ids)

    @classmethod
    def from_sparse_dicts(
        cls,
        sparse_dicts: list[dict[int, float]],
        doc_ids: list[str],
        vocab_size: int,
    ) -> "SparseIndex":
        from scipy.sparse import csr_matrix

        indptr = [0]
        indices: list[int] = []
        values: list[float] = []
        for sd in sparse_dicts:
            for token_id, weight in sd.items():
                indices.append(token_id)
                values.append(weight)
            indptr.append(len(indices))

        matrix = csr_matrix(
            (values, indices, indptr),
            shape=(len(sparse_dicts), vocab_size),
            dtype=np.float32,
        )
        return cls(matrix, doc_ids)

    def search(self, query_dict: dict[int, float], top_k: int) -> tuple[np.ndarray, np.ndarray]:
        """Score all documents against one sparse query."""
        if not query_dict:
            return np.array([]), np.array([], dtype=int)

        from scipy.sparse import csr_matrix

        cols = np.fromiter(query_dict.keys(), dtype=np.int32, count=len(query_dict))
        vals = np.fromiter(query_dict.values(), dtype=np.float32, count=len(query_dict))
        q = csr_matrix(
            (vals, (np.zeros_like(cols), cols)),
            shape=(1, self.matrix.shape[1]),
            dtype=np.float32,
        )

        scores = np.asarray((self.matrix @ q.T).todense()).ravel()

        k = min(top_k, scores.size)
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]

        # Documents sharing no term with the query carry no evidence and must
        # not occupy shortlist slots.
        nonzero = scores[idx] > 0
        return scores[idx][nonzero], idx[nonzero]


class HybridRetriever:
    """Dense + sparse retrieval with fusion and optional reranking."""

    def __init__(
        self,
        dense_encoder: DenseEncoder | None = None,
        splade_encoder: SpladeEncoder | None = None,
        reranker: Any | None = None,
        config: RetrievalConfig | None = None,
    ) -> None:
        self.dense_encoder = dense_encoder
        self.splade_encoder = splade_encoder
        self.reranker = reranker
        self.config = config or RetrievalConfig()

        self.fusion = HybridFusion(
            strategy=self.config.fusion_strategy,
            alpha=self.config.alpha,
            beta=self.config.beta,
            gamma=0.0,
            rrf_k=self.config.rrf_k,
        )

        self.dense_index: DenseIndex | None = None
        self.sparse_index: SparseIndex | None = None
        self.doc_ids: list[str] = []
        self.doc_texts: list[str] = []

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index(self, doc_ids: list[str], doc_texts: list[str]) -> dict[str, float]:
        """Encode and index a corpus. Returns per-stage timings in seconds."""
        self.doc_ids = doc_ids
        self.doc_texts = doc_texts
        timings: dict[str, float] = {}

        if self.dense_encoder is not None:
            logger.info("Dense-encoding %d documents", len(doc_texts))
            with Timer("dense-index") as t:
                embeddings = self.dense_encoder.encode(
                    doc_texts,
                    batch_size=self.config.dense_batch_size,
                    show_progress=True,
                )
                self.dense_index = DenseIndex(embeddings, doc_ids)
            timings["dense_index_s"] = t.elapsed
            logger.info("Dense index built in %.1f s", t.elapsed)

        if self.splade_encoder is not None:
            logger.info("SPLADE-encoding %d documents", len(doc_texts))
            with Timer("sparse-index") as t:
                sparse_dicts = self.splade_encoder.encode_ids(
                    doc_texts,
                    batch_size=self.config.sparse_batch_size,
                    show_progress=True,
                )
                self.sparse_index = SparseIndex.from_sparse_dicts(
                    sparse_dicts, doc_ids, self.splade_encoder.vocab_size
                )
            timings["sparse_index_s"] = t.elapsed
            logger.info("Sparse index built in %.1f s", t.elapsed)

        return timings

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def _hits(self, scores: np.ndarray, indices: np.ndarray) -> list[dict[str, Any]]:
        return [
            {
                "doc_id": self.doc_ids[i],
                "score": float(s),
                "content": self.doc_texts[i],
            }
            for s, i in zip(scores, indices)
        ]

    def retrieve_dense(self, queries: list[str], top_k: int) -> list[list[dict[str, Any]]]:
        if self.dense_index is None or not queries:
            return [[] for _ in queries]
        query_vecs = self.dense_encoder.encode_queries(queries)
        scores, indices = self.dense_index.search(query_vecs, top_k)
        return [self._hits(s, i) for s, i in zip(scores, indices)]

    def retrieve_sparse(self, queries: list[str], top_k: int) -> list[list[dict[str, Any]]]:
        if self.sparse_index is None or not queries:
            return [[] for _ in queries]
        out = []
        for q in queries:
            query_dict = self.splade_encoder.encode_query_ids(q)
            scores, indices = self.sparse_index.search(query_dict, top_k)
            out.append(self._hits(scores, indices))
        return out

    def search(
        self,
        queries: list[str],
        return_stages: bool = False,
    ) -> dict[str, list[list[dict[str, Any]]]]:
        """Run the full pipeline over a batch of queries.

        Returns a mapping from stage name to per-query result lists. With
        ``return_stages=False`` only the final stage is returned.
        """
        cfg = self.config
        stages: dict[str, list[list[dict[str, Any]]]] = {}

        dense_hits = self.retrieve_dense(queries, cfg.top_k)
        sparse_hits = self.retrieve_sparse(queries, cfg.top_k)

        if self.dense_index is not None:
            stages["dense"] = dense_hits
        if self.sparse_index is not None:
            stages["sparse"] = sparse_hits

        # Which stage feeds the reranker depends on which indexes exist, not on
        # whether one particular query happened to return hits.
        if self.dense_index is not None and self.sparse_index is not None:
            fused = [
                self.fusion.fuse(s, d)[: cfg.top_k]
                for s, d in zip(sparse_hits, dense_hits)
            ]
            stages["hybrid"] = fused
            source_stage = "hybrid"
        elif self.dense_index is not None:
            source_stage = "dense"
        else:
            source_stage = "sparse"

        candidates = stages[source_stage]

        if self.reranker is not None:
            with Timer("rerank-all") as t:
                # Deliberately not truncated to final_k: the reranker reorders
                # the shortlist, and cutting it here would depress recall@k for
                # k > final_k, making this stage incomparable to the others.
                reranked = [
                    self.reranker.rerank(q, c, top_n_to_rerank=cfg.rerank_top_n)
                    for q, c in zip(queries, candidates)
                ]
            logger.info(
                "Reranked %d queries in %.1f s (%.1f ms/query)",
                len(queries), t.elapsed, t.elapsed_ms / max(len(queries), 1),
            )
            stages[f"{source_stage}+rerank"] = reranked

        if not return_stages:
            last = list(stages)[-1]
            return {last: stages[last]}
        return stages
