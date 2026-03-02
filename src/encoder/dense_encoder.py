"""
Dense Encoder
=============
Wraps sentence-transformers to produce fixed-dimensional dense embeddings.
Used as the dense branch of the hybrid retrieval pipeline.
"""

from __future__ import annotations

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from ..utils.logging_utils import Timer, get_logger

logger = get_logger(__name__)


class DenseEncoder:
    """Produce dense vector embeddings via sentence-transformers."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "auto",
        max_length: int = 256,
    ) -> None:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info("Loading dense model '%s' on %s", model_name, device)
        self.model = SentenceTransformer(model_name, device=device)
        self.model.max_seq_length = max_length
        self.device = device
        self.embedding_dim: int = self.model.get_sentence_embedding_dimension()
        logger.info("Dense embedding dimension: %d", self.embedding_dim)

    def encode(
        self,
        texts: list[str],
        batch_size: int = 64,
        show_progress: bool = True,
        normalize: bool = True,
    ) -> np.ndarray:
        """Encode texts into a (N, dim) float32 numpy array.

        Args:
            texts: List of text strings.
            batch_size: Encoding batch size.
            show_progress: Log per-batch timing.
            normalize: L2-normalize embeddings (required for cosine similarity).

        Returns:
            np.ndarray of shape (len(texts), embedding_dim).
        """
        with Timer("dense-encode-all") as t:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                normalize_embeddings=normalize,
                convert_to_numpy=True,
            )

        n = len(texts)
        per_doc = t.elapsed / n if n else 0
        logger.info(
            "Dense-encoded %d texts in %.1f ms  (%.2f ms/doc)",
            n,
            t.elapsed_ms,
            per_doc * 1000,
        )
        return embeddings.astype(np.float32)

    def encode_query(self, query: str, normalize: bool = True) -> np.ndarray:
        """Encode a single query — returns (dim,) vector."""
        with Timer("dense-encode-query") as t:
            vec = self.model.encode(
                [query],
                normalize_embeddings=normalize,
                convert_to_numpy=True,
            )[0]
        logger.info("Dense query encoded in %.2f ms", t.elapsed_ms)
        return vec.astype(np.float32)
