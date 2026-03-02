"""
Sparse HNSW ANN Backend
========================
Converts SPLADE sparse vectors into dense vocab-sized arrays and indexes
them with hnswlib for approximate nearest-neighbor cosine search.

Design note:
    BERT vocab = 30 522 dimensions.  This is high for HNSW, which makes
    the comparison against Elasticsearch intentionally instructive: ANN
    methods designed for dense embeddings carry a significant memory
    overhead when applied to sparse representations.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import hnswlib
import numpy as np

from ..utils.logging_utils import Timer, get_logger

logger = get_logger(__name__)


class HNSWSparseBackend:
    """hnswlib index over sparse SPLADE vectors (represented as dense arrays)."""

    def __init__(
        self,
        dim: int = 30522,
        ef_construction: int = 200,
        M: int = 16,
        ef_search: int = 100,
        num_threads: int = 4,
        index_path: str = "data/hnsw_sparse.index",
    ) -> None:
        self.dim = dim
        self.ef_construction = ef_construction
        self.M = M
        self.ef_search = ef_search
        self.num_threads = num_threads
        self.index_path = index_path

        self.index: hnswlib.Index | None = None
        self.doc_ids: list[str] = []       # ordinal → doc_id mapping
        self.contents: list[str] = []      # ordinal → text (for retrieval display)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build_index(
        self,
        doc_ids: list[str],
        sparse_dicts: list[dict[str, float]],
        contents: list[str] | None = None,
        tokenizer: Any = None,
    ) -> None:
        """Build the HNSW index from sparse dicts.

        Args:
            doc_ids: Document identifiers (same order as sparse_dicts).
            sparse_dicts: {token_string: weight} per document.
            contents: Optional raw text per document.
            tokenizer: A HuggingFace tokenizer to convert token strings → IDs.
                       If None, token strings are hashed modulo dim.
        """
        n = len(doc_ids)
        self.doc_ids = list(doc_ids)
        self.contents = list(contents) if contents else [""] * n

        logger.info("Converting %d sparse dicts to dense (%d-dim)…", n, self.dim)
        with Timer("sparse-to-dense") as t_conv:
            data = self._sparse_dicts_to_dense(sparse_dicts, tokenizer)
        logger.info("Conversion: %.1f ms", t_conv.elapsed_ms)

        # L2-normalize for cosine similarity (hnswlib cosine = inner product on unit vecs)
        norms = np.linalg.norm(data, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        data = data / norms

        logger.info("Building HNSW index (M=%d, efC=%d)…", self.M, self.ef_construction)
        self.index = hnswlib.Index(space="cosine", dim=self.dim)
        self.index.init_index(max_elements=n, ef_construction=self.ef_construction, M=self.M)
        self.index.set_num_threads(self.num_threads)

        with Timer("hnsw-build") as t_build:
            self.index.add_items(data, ids=np.arange(n))
        logger.info("HNSW built in %.1f ms  (%d vectors)", t_build.elapsed_ms, n)

        self.index.set_ef(self.ef_search)

    def save(self, path: str | None = None) -> None:
        path = path or self.index_path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.index.save_index(path)
        # Save id mapping alongside
        np.save(path + ".ids.npy", np.array(self.doc_ids, dtype=object))
        logger.info("HNSW index saved to %s", path)

    def load(self, path: str | None = None) -> None:
        path = path or self.index_path
        self.index = hnswlib.Index(space="cosine", dim=self.dim)
        self.index.load_index(path)
        self.index.set_ef(self.ef_search)
        ids_path = path + ".ids.npy"
        if os.path.exists(ids_path):
            self.doc_ids = list(np.load(ids_path, allow_pickle=True))
        logger.info("HNSW index loaded from %s  (%d elements)", path, self.index.get_current_count())

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_sparse: dict[str, float],
        top_k: int = 10,
        tokenizer: Any = None,
    ) -> list[dict[str, Any]]:
        """Query the HNSW index with a SPLADE sparse dict.

        Returns list of {doc_id, score, rank}.
        """
        vec = self._sparse_dict_to_dense_vector(query_sparse, tokenizer)

        # Normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        with Timer("hnsw-search") as t:
            labels, distances = self.index.knn_query(vec.reshape(1, -1), k=top_k)

        results = []
        for rank, (label, dist) in enumerate(zip(labels[0], distances[0])):
            doc_id = self.doc_ids[label] if label < len(self.doc_ids) else str(label)
            content = self.contents[label] if label < len(self.contents) else ""
            results.append({
                "doc_id": doc_id,
                "score": float(1.0 - dist),   # cosine similarity = 1 - cosine distance
                "rank": rank + 1,
                "content": content,
            })

        logger.info("HNSW search: %d results in %.2f ms", len(results), t.elapsed_ms)
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _sparse_dicts_to_dense(
        self,
        sparse_dicts: list[dict[str, float]],
        tokenizer: Any = None,
    ) -> np.ndarray:
        mat = np.zeros((len(sparse_dicts), self.dim), dtype=np.float32)
        for i, sd in enumerate(sparse_dicts):
            for token, weight in sd.items():
                idx = self._token_to_index(token, tokenizer)
                if 0 <= idx < self.dim:
                    mat[i, idx] = weight
        return mat

    def _sparse_dict_to_dense_vector(
        self,
        sparse_dict: dict[str, float],
        tokenizer: Any = None,
    ) -> np.ndarray:
        return self._sparse_dicts_to_dense([sparse_dict], tokenizer)[0]

    @staticmethod
    def _token_to_index(token: str, tokenizer: Any = None) -> int:
        """Map a token string to a dimension index."""
        if tokenizer is not None:
            ids = tokenizer.encode(token, add_special_tokens=False)
            return ids[0] if ids else -1
        # Fallback: deterministic hash
        return hash(token) % 30522

    def get_index_size_mb(self) -> float:
        if self.index_path and os.path.exists(self.index_path):
            return os.path.getsize(self.index_path) / (1024 * 1024)
        return 0.0
