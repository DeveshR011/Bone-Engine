"""
SPLADE Sparse Encoder
=====================
Uses the naver/splade-cocondenser-ensembledistil masked-language model to
produce vocabulary-weight sparse vectors via log-saturation + max pooling.

Key features:
- Top-K term selection (default 100)
- Optional 8-bit quantization
- CPU / GPU inference with automatic device selection
- Per-document and per-query timing logs
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from ..utils.logging_utils import Timer, get_logger

logger = get_logger(__name__)


class SpladeEncoder:
    """Encode text into SPLADE sparse representations.

    A sparse representation maps vocabulary token IDs (or their string form)
    to positive float weights.  Only the top-K highest weights are retained
    so the representation stays truly sparse.
    """

    def __init__(
        self,
        model_name: str = "naver/splade-cocondenser-ensembledistil",
        device: str = "auto",
        max_length: int = 256,
        top_k: int = 100,
        quantize: bool = False,
    ) -> None:
        # Resolve device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info("Loading SPLADE model '%s' on %s", model_name, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

        self.max_length = max_length
        self.top_k = top_k
        self.quantize = quantize
        self.vocab_size: int = self.tokenizer.vocab_size

        # Global max weight — set during encode_corpus for quantization
        self.global_max: float | None = None

        # Collect IDs of special tokens to zero-out
        self._special_ids = set(self.tokenizer.all_special_ids)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> list[dict[str, float]]:
        """Encode a list of texts into sparse dicts {token_string: weight}.

        Returns one dict per input text.
        """
        all_sparse: list[dict[str, float]] = []
        total = len(texts)

        for start in range(0, total, batch_size):
            batch = texts[start : start + batch_size]

            with Timer(f"encode-batch-{start}") as t:
                sparse_batch = self._encode_batch(batch)

            all_sparse.extend(sparse_batch)

            if show_progress:
                done = min(start + batch_size, total)
                per_doc = t.elapsed / len(batch)
                logger.info(
                    "Encoded %d/%d docs  (%.1f ms/doc, batch %.1f ms)",
                    done,
                    total,
                    per_doc * 1000,
                    t.elapsed_ms,
                )

        # Compute global max for quantization
        if all_sparse:
            self.global_max = max(
                (v for d in all_sparse for v in d.values()), default=1.0
            )

        if self.quantize:
            all_sparse = [self._quantize(d) for d in all_sparse]

        return all_sparse

    def encode_query(self, query: str) -> dict[str, float]:
        """Encode a single query string — convenience wrapper with timing."""
        with Timer("encode-query") as t:
            result = self._encode_batch([query])[0]

        if self.quantize and self.global_max is not None:
            result = self._quantize(result)

        logger.info("Query encoded in %.2f ms  (%d terms)", t.elapsed_ms, len(result))
        return result

    def encode_to_dense(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> np.ndarray:
        """Encode texts and return dense vocab-sized numpy array (for HNSW).

        Shape: (len(texts), vocab_size)
        """
        sparse_dicts = self.encode(texts, batch_size=batch_size, show_progress=False)
        return self.sparse_dicts_to_dense(sparse_dicts)

    def sparse_dicts_to_dense(self, sparse_dicts: list[dict[str, float]]) -> np.ndarray:
        """Convert list of sparse dicts to dense numpy matrix.

        Each row is a vocab-sized vector.  Token strings are converted back
        to token IDs via the tokenizer.
        """
        mat = np.zeros((len(sparse_dicts), self.vocab_size), dtype=np.float32)
        for i, sd in enumerate(sparse_dicts):
            for token, weight in sd.items():
                ids = self.tokenizer.encode(token, add_special_tokens=False)
                if ids:
                    mat[i, ids[0]] = weight
        return mat

    def sparse_dict_to_dense_vector(self, sparse_dict: dict[str, float]) -> np.ndarray:
        """Convert a single sparse dict to a dense vocab-sized vector."""
        return self.sparse_dicts_to_dense([sparse_dict])[0]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _encode_batch(self, texts: list[str]) -> list[dict[str, float]]:
        """Core encoding: tokenize → forward → log-saturation → max pool → top-K."""
        tokens = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)

        output = self.model(**tokens)
        logits = output.logits  # (batch, seq_len, vocab_size)

        # Log-saturation activation: log(1 + ReLU(x))
        activated = torch.log1p(torch.relu(logits))

        # Max-pool across the sequence dimension
        pooled = activated.max(dim=1).values  # (batch, vocab_size)

        # Zero-out special tokens
        for sid in self._special_ids:
            if sid < pooled.shape[1]:
                pooled[:, sid] = 0.0

        # Move to CPU for top-k extraction
        pooled_np = pooled.cpu().numpy()

        results: list[dict[str, float]] = []
        for row in pooled_np:
            nonzero_count = int((row > 0).sum())
            k = min(self.top_k, nonzero_count)
            if k == 0:
                results.append({})
                continue

            # Partial argsort for top-K (faster than full sort)
            top_indices = np.argpartition(row, -k)[-k:]
            top_indices = top_indices[np.argsort(-row[top_indices])]

            sparse = {}
            for idx in top_indices:
                weight = float(row[idx])
                if weight <= 0:
                    continue
                token_str = self.tokenizer.decode([int(idx)]).strip()
                if token_str:
                    sparse[token_str] = weight
            results.append(sparse)

        return results

    def _quantize(self, sparse_dict: dict[str, float]) -> dict[str, float]:
        """8-bit quantization: weight_q = int((weight / global_max) * 255)."""
        if not sparse_dict or not self.global_max:
            return sparse_dict
        gmax = self.global_max
        return {
            k: float(int((v / gmax) * 255))
            for k, v in sparse_dict.items()
            if v > 0
        }
