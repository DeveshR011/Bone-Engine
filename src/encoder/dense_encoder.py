"""
Dense Encoder
=============
Wraps sentence-transformers to produce fixed-dimensional dense embeddings.
Used as the dense branch of the hybrid retrieval pipeline.

Asymmetric retrieval
--------------------
Modern retrieval embedders (BGE, E5, GTE) are trained with an instruction
prefix on the *query* side only. Retrieval is asymmetric — a short question and
a long passage are not interchangeable inputs — and omitting the prefix costs
roughly 1-3 nDCG@10 points on BEIR for the BGE family. ``query_prefix`` is
applied in :meth:`encode_query` and deliberately never in :meth:`encode`.
"""

from __future__ import annotations

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from ..utils.logging_utils import Timer, get_logger

logger = get_logger(__name__)

# Query-side instruction prefixes, keyed by model-name substring.
QUERY_PREFIXES: dict[str, str] = {
    "bge": "Represent this sentence for searching relevant passages: ",
    "e5": "query: ",
    "gte": "",
    "all-minilm": "",
}


def default_query_prefix(model_name: str) -> str:
    """Best-known query instruction prefix for a model, or "" if none applies."""
    lowered = model_name.lower()
    for key, prefix in QUERY_PREFIXES.items():
        if key in lowered:
            return prefix
    return ""


class DenseEncoder:
    """Produce dense vector embeddings via sentence-transformers."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        device: str = "auto",
        max_length: int = 512,
        query_prefix: str | None = None,
        fp16: bool = True,
    ) -> None:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info("Loading dense model '%s' on %s", model_name, device)
        self.model = SentenceTransformer(model_name, device=device)
        self.model.max_seq_length = max_length

        # fp16 roughly halves both encode time and VRAM on GPU; on CPU it is
        # slower than fp32, so it is restricted to CUDA.
        self.fp16 = fp16 and device == "cuda"
        if self.fp16:
            self.model = self.model.half()

        self.device = device
        self.embedding_dim: int = self.model.get_sentence_embedding_dimension()
        self.query_prefix = (
            default_query_prefix(model_name) if query_prefix is None else query_prefix
        )

        logger.info(
            "Dense dim=%d, fp16=%s, query_prefix=%r",
            self.embedding_dim, self.fp16, self.query_prefix,
        )

    def encode(
        self,
        texts: list[str],
        batch_size: int = 64,
        show_progress: bool = True,
        normalize: bool = True,
    ) -> np.ndarray:
        """Encode documents into a (N, dim) float32 array.

        No query prefix is applied here — see the module docstring.
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
            n, t.elapsed_ms, per_doc * 1000,
        )
        return embeddings.astype(np.float32)

    def encode_queries(
        self,
        queries: list[str],
        batch_size: int = 64,
        show_progress: bool = False,
        normalize: bool = True,
    ) -> np.ndarray:
        """Encode queries in batch, applying the instruction prefix."""
        prefixed = [self.query_prefix + q for q in queries]
        embeddings = self.model.encode(
            prefixed,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def encode_query(self, query: str, normalize: bool = True) -> np.ndarray:
        """Encode a single query — returns a (dim,) vector."""
        with Timer("dense-encode-query") as t:
            vec = self.model.encode(
                [self.query_prefix + query],
                normalize_embeddings=normalize,
                convert_to_numpy=True,
            )[0]
        logger.debug("Dense query encoded in %.2f ms", t.elapsed_ms)
        return vec.astype(np.float32)
