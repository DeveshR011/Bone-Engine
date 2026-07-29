"""
Cross-Encoder Reranker
======================
Rescores a shortlist of retrieved candidates by encoding each (query, document)
pair *jointly*, letting every query token attend to every document token.

This is the largest single quality gain available in a retrieval pipeline and
the reason it exists here. Bi-encoders (dense and SPLADE alike) must compress a
document into a vector before ever seeing the query, so fine-grained term
interaction is lost. A cross-encoder recovers it, at a cost linear in the
number of candidates — which is why it runs over a shortlist (typically the
top 100) rather than the whole corpus.

Memory notes for small GPUs: a base-sized reranker in fp16 needs roughly 0.6 GB
of weights, so the batch activations dominate. ``batch_size`` and
``max_length`` are the two knobs that matter if VRAM is tight.
"""

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ..utils.logging_utils import Timer, get_logger

logger = get_logger(__name__)


class CrossEncoderReranker:
    """Rerank (query, document) pairs with a cross-encoder."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str = "auto",
        max_length: int = 512,
        batch_size: int = 32,
        fp16: bool = True,
        blend_weight: float = 0.3,
        blend_k: int = 60,
    ) -> None:
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.fp16 = fp16 and self.device.type == "cuda"

        logger.info(
            "Loading reranker '%s' on %s (fp16=%s)", model_name, self.device, self.fp16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.fp16 else torch.float32,
        ).to(self.device)
        self.model.eval()

        self.max_length = max_length
        self.batch_size = batch_size
        self.blend_weight = blend_weight
        self.blend_k = blend_k

    @torch.inference_mode()
    def score_pairs(self, query: str, documents: list[str]) -> list[float]:
        """Score one query against many documents. Higher is more relevant."""
        if not documents:
            return []

        scores: list[float] = []
        for start in range(0, len(documents), self.batch_size):
            batch = documents[start : start + self.batch_size]
            inputs = self.tokenizer(
                [query] * len(batch),
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            logits = self.model(**inputs).logits

            # Reranker heads are single-logit relevance regressors; some
            # checkpoints instead expose a 2-way classifier, where the
            # positive class is index 1.
            if logits.shape[-1] == 1:
                batch_scores = logits.squeeze(-1)
            else:
                batch_scores = logits[:, 1]

            scores.extend(batch_scores.float().cpu().tolist())

        return scores

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int | None = None,
        top_n_to_rerank: int | None = None,
        scores: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        """Rerank candidate results for a single query.

        Args:
            query: Query text.
            candidates: Retrieved results, each ``{doc_id, score, content}``,
                already ordered by first-stage score.
            top_k: How many results to return after reranking.
            top_n_to_rerank: Only rerank this many top candidates; any
                remainder keeps its first-stage order and is appended below.
                Reranking cost is linear in this number.
            scores: Precomputed relevance scores for the reranked head. Lets a
                parameter sweep score each pair once and reuse the logits
                across settings; ``query`` is unused when supplied.

        Returns:
            Results sorted by the blended score, each carrying its raw
            ``rerank_score`` and original ``retrieval_score`` for inspection.
        """
        if not candidates:
            return []

        n = len(candidates) if top_n_to_rerank is None else min(top_n_to_rerank, len(candidates))
        head, tail = candidates[:n], candidates[n:]

        with Timer("rerank") as t:
            if scores is None:
                scores = self.score_pairs(query, [c.get("content", "") for c in head])
            else:
                scores = scores[:n]

        # Rank-blend the reranker against the first stage instead of letting it
        # overwrite the ranking outright. A cross-encoder trained on
        # question->passage relevance degrades on other query shapes (claim
        # verification, keyword queries), where it can bury a correct
        # first-stage hit. Blending bounds that damage while keeping the gain
        # where the reranker is right. blend_weight=1.0 restores pure rerank.
        w, k = self.blend_weight, self.blend_k

        rerank_order = sorted(range(len(head)), key=lambda i: scores[i], reverse=True)
        rerank_rank = {i: r for r, i in enumerate(rerank_order, start=1)}

        reranked = []
        for i, c in enumerate(head):
            blended = w / (k + rerank_rank[i]) + (1.0 - w) / (k + i + 1)
            reranked.append(
                {
                    **c,
                    "rerank_score": scores[i],
                    "retrieval_score": c.get("score", 0.0),
                    "score": blended,
                }
            )
        reranked.sort(key=lambda r: r["score"], reverse=True)

        # Un-reranked leftovers keep their relative order but must sort strictly
        # below every reranked document. Their raw retrieval scores live on a
        # different scale than blended rank scores, so they are re-based below
        # the lowest blended score — downstream consumers rank by `score`, not
        # by list position.
        floor = min((r["score"] for r in reranked), default=0.0)
        tail_scored = [
            {
                **c,
                "retrieval_score": c.get("score", 0.0),
                "score": floor - (i + 1) / (k + len(tail) + 1),
            }
            for i, c in enumerate(tail)
        ]
        merged = reranked + tail_scored

        logger.debug(
            "Reranked %d candidates in %.1f ms (%.2f ms/pair)",
            n, t.elapsed_ms, t.elapsed_ms / max(n, 1),
        )

        return merged[:top_k] if top_k else merged
