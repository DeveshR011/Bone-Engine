"""
SPLADE Sparse Encoder
=====================
Uses a SPLADE masked-language model to produce vocabulary-weight sparse
vectors via log-saturation + masked max pooling.

Key features:
- Attention-masked pooling (padding positions cannot contribute terms)
- Lossless token-ID <-> token-string mapping
- Top-K term selection (default 100)
- Optional 8-bit quantization
- fp16 CPU/GPU inference with automatic device selection
- Per-document and per-query timing logs
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from ..utils.logging_utils import Timer, get_logger

logger = get_logger(__name__)


class SpladeEncoder:
    """Encode text into SPLADE sparse representations.

    A sparse representation maps vocabulary tokens to positive float weights.
    Only the top-K highest weights are retained so the representation stays
    truly sparse.

    Two key formats are produced from the same forward pass:

      * ``encode()``     -> ``{token_string: weight}``, for term-based backends
                            (Elasticsearch ``rank_features``, PostgreSQL JSONB)
      * ``encode_ids()`` -> ``{token_id: weight}``, for vector backends (HNSW)

    Conversion between the two uses ``convert_ids_to_tokens`` /
    ``convert_tokens_to_ids``, which are exact inverses. Round-tripping through
    ``decode()``/``encode()`` instead is lossy: ``decode`` strips the ``##``
    subword marker, so ``##ing`` re-encodes to a different vocabulary entry and
    silently corrupts the vector.
    """

    def __init__(
        self,
        model_name: str = "naver/splade-cocondenser-ensembledistil",
        device: str = "auto",
        max_length: int = 256,
        top_k: int = 100,
        quantize: bool = False,
        fp16: bool = True,
    ) -> None:
        # Resolve device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # fp16 is a meaningful speed/VRAM win on GPU but is slow and poorly
        # supported for CPU inference, so restrict it to CUDA.
        self.fp16 = fp16 and self.device.type == "cuda"

        logger.info(
            "Loading SPLADE model '%s' on %s (fp16=%s)",
            model_name, self.device, self.fp16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.fp16 else torch.float32,
        ).to(self.device)
        self.model.eval()

        self.max_length = max_length
        self.top_k = top_k
        self.quantize = quantize
        self.vocab_size: int = self.model.config.vocab_size

        # Global max weight — set during encode for quantization
        self.global_max: float | None = None

        # Special tokens must never become retrieval terms
        self._special_ids = [
            i for i in self.tokenizer.all_special_ids if i < self.vocab_size
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> list[dict[str, float]]:
        """Encode texts into sparse dicts ``{token_string: weight}``."""
        id_dicts = self.encode_ids(texts, batch_size=batch_size, show_progress=show_progress)
        return [self._ids_to_tokens(d) for d in id_dicts]

    def encode_ids(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> list[dict[int, float]]:
        """Encode texts into sparse dicts ``{token_id: weight}``."""
        all_sparse: list[dict[int, float]] = []
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
                    done, total, per_doc * 1000, t.elapsed_ms,
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
            id_dict = self._encode_batch([query])[0]

        if self.quantize and self.global_max is not None:
            id_dict = self._quantize(id_dict)

        result = self._ids_to_tokens(id_dict)
        logger.info("Query encoded in %.2f ms  (%d terms)", t.elapsed_ms, len(result))
        return result

    def encode_query_ids(self, query: str) -> dict[int, float]:
        """Encode a single query into ``{token_id: weight}``."""
        id_dict = self._encode_batch([query])[0]
        if self.quantize and self.global_max is not None:
            id_dict = self._quantize(id_dict)
        return id_dict

    def encode_to_dense(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> np.ndarray:
        """Encode texts and return a dense vocab-sized matrix (for HNSW).

        Shape: ``(len(texts), vocab_size)``
        """
        id_dicts = self.encode_ids(texts, batch_size=batch_size, show_progress=False)
        mat = np.zeros((len(id_dicts), self.vocab_size), dtype=np.float32)
        for i, sd in enumerate(id_dicts):
            for token_id, weight in sd.items():
                mat[i, token_id] = weight
        return mat

    def sparse_dicts_to_dense(self, sparse_dicts: list[dict[str, float]]) -> np.ndarray:
        """Convert token-string sparse dicts to a dense vocab-sized matrix.

        Uses ``convert_tokens_to_ids``, the exact inverse of the mapping used
        to build the string keys.
        """
        mat = np.zeros((len(sparse_dicts), self.vocab_size), dtype=np.float32)
        for i, sd in enumerate(sparse_dicts):
            if not sd:
                continue
            tokens = list(sd.keys())
            ids = self.tokenizer.convert_tokens_to_ids(tokens)
            unk_id = self.tokenizer.unk_token_id
            for token, token_id in zip(tokens, ids):
                if token_id is None or token_id == unk_id or token_id >= self.vocab_size:
                    continue
                mat[i, token_id] = sd[token]
        return mat

    def sparse_dict_to_dense_vector(self, sparse_dict: dict[str, float]) -> np.ndarray:
        """Convert a single sparse dict to a dense vocab-sized vector."""
        return self.sparse_dicts_to_dense([sparse_dict])[0]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def _encode_batch(self, texts: list[str]) -> list[dict[int, float]]:
        """Tokenize -> forward -> log-saturation -> masked max pool -> top-K."""
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

        # Mask padding BEFORE pooling. Without this, padded positions still
        # produce MLM logits and their vocabulary terms leak into the max,
        # polluting short documents in a batch with long ones.
        mask = tokens["attention_mask"].unsqueeze(-1)  # (batch, seq_len, 1)
        activated = activated.masked_fill(mask == 0, 0.0)

        # Max-pool across the sequence dimension
        pooled = activated.max(dim=1).values  # (batch, vocab_size)

        # Zero-out special tokens
        if self._special_ids:
            pooled[:, self._special_ids] = 0.0

        # Top-K selection on GPU, then a single small transfer to CPU
        k = min(self.top_k, pooled.shape[1])
        top_values, top_indices = torch.topk(pooled, k=k, dim=1)

        values_np = top_values.float().cpu().numpy()
        indices_np = top_indices.cpu().numpy()

        results: list[dict[int, float]] = []
        for vals, idxs in zip(values_np, indices_np):
            keep = vals > 0
            results.append(
                {int(i): float(v) for i, v in zip(idxs[keep], vals[keep])}
            )
        return results

    def _ids_to_tokens(self, id_dict: dict[int, float]) -> dict[str, float]:
        """Map ``{token_id: weight}`` to ``{token_string: weight}`` losslessly."""
        if not id_dict:
            return {}
        ids = list(id_dict.keys())
        tokens = self.tokenizer.convert_ids_to_tokens(ids)
        return {
            token: id_dict[token_id]
            for token_id, token in zip(ids, tokens)
            if token
        }

    def _quantize(self, sparse_dict: dict[Any, float]) -> dict[Any, float]:
        """8-bit quantization: weight_q = int((weight / global_max) * 255)."""
        if not sparse_dict or not self.global_max:
            return sparse_dict
        gmax = self.global_max
        return {
            k: float(int((v / gmax) * 255))
            for k, v in sparse_dict.items()
            if v > 0
        }
