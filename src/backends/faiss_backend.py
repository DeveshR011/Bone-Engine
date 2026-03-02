"""
FAISS Dense Backend
===================
Stores dense embeddings (from sentence-transformers) in a FAISS index
for fast approximate nearest-neighbor search.

Supports Flat, IVFFlat, and IVFPQ index types.
"""

from __future__ import annotations

import os
from typing import Any

import faiss
import numpy as np

from ..utils.logging_utils import Timer, get_logger

logger = get_logger(__name__)


class FAISSDenseBackend:
    """FAISS index for dense vector retrieval."""

    def __init__(
        self,
        dim: int = 384,
        index_type: str = "IVFFlat",
        nlist: int = 100,
        nprobe: int = 10,
        index_path: str = "data/faiss_dense.index",
    ) -> None:
        self.dim = dim
        self.index_type = index_type
        self.nlist = nlist
        self.nprobe = nprobe
        self.index_path = index_path

        self.index: faiss.Index | None = None
        self.doc_ids: list[str] = []
        self.contents: list[str] = []

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build_index(
        self,
        doc_ids: list[str],
        embeddings: np.ndarray,
        contents: list[str] | None = None,
    ) -> None:
        """Build the FAISS index.

        Args:
            doc_ids: Document identifiers.
            embeddings: (N, dim) float32 array of dense embeddings.
            contents: Optional raw text per document.
        """
        n = embeddings.shape[0]
        assert embeddings.shape[1] == self.dim, (
            f"Embedding dim {embeddings.shape[1]} != configured dim {self.dim}"
        )
        self.doc_ids = list(doc_ids)
        self.contents = list(contents) if contents else [""] * n

        logger.info("Building FAISS %s index  (n=%d, dim=%d)", self.index_type, n, self.dim)

        with Timer("faiss-build") as t:
            if self.index_type == "Flat":
                self.index = faiss.IndexFlatIP(self.dim)
            elif self.index_type == "IVFFlat":
                quantizer = faiss.IndexFlatIP(self.dim)
                actual_nlist = min(self.nlist, n)
                self.index = faiss.IndexIVFFlat(quantizer, self.dim, actual_nlist)
                self.index.train(embeddings)
            elif self.index_type == "IVFPQ":
                quantizer = faiss.IndexFlatIP(self.dim)
                actual_nlist = min(self.nlist, n)
                m_sub = min(48, self.dim)  # PQ sub-quantizers
                self.index = faiss.IndexIVFPQ(quantizer, self.dim, actual_nlist, m_sub, 8)
                self.index.train(embeddings)
            else:
                raise ValueError(f"Unknown index type: {self.index_type}")

            self.index.add(embeddings)

        if hasattr(self.index, "nprobe"):
            self.index.nprobe = self.nprobe

        logger.info("FAISS index built in %.1f ms  (%d vectors)", t.elapsed_ms, n)

    def save(self, path: str | None = None) -> None:
        path = path or self.index_path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        faiss.write_index(self.index, path)
        np.save(path + ".ids.npy", np.array(self.doc_ids, dtype=object))
        logger.info("FAISS index saved to %s", path)

    def load(self, path: str | None = None) -> None:
        path = path or self.index_path
        self.index = faiss.read_index(path)
        if hasattr(self.index, "nprobe"):
            self.index.nprobe = self.nprobe
        ids_path = path + ".ids.npy"
        if os.path.exists(ids_path):
            self.doc_ids = list(np.load(ids_path, allow_pickle=True))
        logger.info("FAISS index loaded from %s  (%d vectors)", path, self.index.ntotal)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_vec: np.ndarray,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Search with a dense query vector.

        Args:
            query_vec: (dim,) float32 query embedding (should be L2-normalized).
            top_k: Number of results.

        Returns list of {doc_id, score, rank, content}.
        """
        q = query_vec.reshape(1, -1).astype(np.float32)

        with Timer("faiss-search") as t:
            scores, indices = self.index.search(q, top_k)

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < 0:
                continue  # FAISS returns -1 when fewer results than k
            doc_id = self.doc_ids[idx] if idx < len(self.doc_ids) else str(idx)
            content = self.contents[idx] if idx < len(self.contents) else ""
            results.append({
                "doc_id": doc_id,
                "score": float(score),
                "rank": rank + 1,
                "content": content,
            })

        logger.info("FAISS search: %d results in %.2f ms", len(results), t.elapsed_ms)
        return results

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_index_size_mb(self) -> float:
        if self.index_path and os.path.exists(self.index_path):
            return os.path.getsize(self.index_path) / (1024 * 1024)
        return 0.0
