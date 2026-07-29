#!/usr/bin/env python3
"""
tune_rerank_blend.py
====================
Sweep the reranker's ``blend_weight`` on BEIR datasets.

A cross-encoder does not help uniformly. It is trained on question->passage
relevance, so on claim-verification (SciFact) or keyword-style (NFCorpus)
queries it can rank a topically-related non-gold document above the gold one
and destroy a strong first-stage ranking. ``blend_weight`` controls how much
authority it gets: 0.0 ignores it, 1.0 lets it overwrite the ranking outright.

The corpus is encoded once and every weight is evaluated against the same
cached candidates, so a sweep costs barely more than a single run.

Usage:
    python scripts/tune_rerank_blend.py --datasets scifact nfcorpus
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.beir_loader import load_dataset
from src.encoder.dense_encoder import DenseEncoder
from src.encoder.splade_encoder import SpladeEncoder
from src.evaluation.beir_eval import evaluate_run, results_to_run
from src.pipeline.retrieval_pipeline import HybridRetriever, RetrievalConfig
from src.rerank.cross_encoder import CrossEncoderReranker
from src.utils.logging_utils import get_logger, load_config

logger = get_logger("tune_blend")

DEFAULT_WEIGHTS = [0.0, 0.2, 0.3, 0.5, 0.7, 0.85, 1.0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune reranker blend weight")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--datasets", nargs="+", default=["scifact", "nfcorpus"])
    parser.add_argument("--weights", nargs="+", type=float, default=DEFAULT_WEIGHTS)
    parser.add_argument(
        "--reranker-models", nargs="+", default=None,
        help="Compare several rerankers; defaults to the configured one",
    )
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--output", default="results/blend_sweep.json")
    args = parser.parse_args()

    cfg = load_config(args.config)

    dense = DenseEncoder(
        model_name=cfg["dense"]["model_name"],
        max_length=cfg["dense"].get("max_length", 512),
        fp16=cfg["dense"].get("fp16", True),
    )
    splade = SpladeEncoder(
        model_name=cfg["splade"]["model_name"],
        max_length=cfg["splade"].get("max_length", 256),
        top_k=cfg["splade"].get("top_k", 100),
        fp16=cfg["splade"].get("fp16", True),
    )
    rr_cfg = cfg.get("reranker", {})
    model_names = args.reranker_models or [
        rr_cfg.get("model_name", "BAAI/bge-reranker-base")
    ]

    all_results: dict[str, dict[str, float]] = {}

    for name in args.datasets:
        dataset = load_dataset(name)
        retriever = HybridRetriever(
            dense_encoder=dense,
            splade_encoder=splade,
            reranker=None,
            config=RetrievalConfig(top_k=args.top_k),
        )
        doc_ids = dataset.doc_ids
        retriever.index(doc_ids, dataset.doc_texts(doc_ids))

        query_ids = dataset.query_ids
        query_texts = [dataset.queries[q] for q in query_ids]

        stages = retriever.search(query_texts, return_stages=True)
        first_stage = stages.get("hybrid", stages.get("dense"))

        baseline = evaluate_run(
            dataset.qrels, results_to_run(dict(zip(query_ids, first_stage)))
        )
        logger.info("%s first-stage nDCG@10 = %.4f", name, baseline["ndcg@10"])

        per_weight: dict[str, float] = {"first_stage": baseline["ndcg@10"]}

        for model_name in model_names:
            reranker = CrossEncoderReranker(
                model_name=model_name,
                max_length=rr_cfg.get("max_length", 512),
                batch_size=rr_cfg.get("batch_size", 32),
                fp16=rr_cfg.get("fp16", True),
            )
            short = model_name.split("/")[-1]

            # Score every (query, candidate) pair once; the weight sweep then
            # only re-ranks, so the model runs a single time per dataset.
            cached: list[list[float]] = [
                reranker.score_pairs(q, [c["content"] for c in cands])
                for q, cands in zip(query_texts, first_stage)
            ]

            for w in args.weights:
                reranker.blend_weight = w
                blended = [
                    reranker.rerank(None, cands, scores=scores)
                    for cands, scores in zip(first_stage, cached)
                ]
                metrics = evaluate_run(
                    dataset.qrels, results_to_run(dict(zip(query_ids, blended)))
                )
                per_weight[f"{short}@{w}"] = metrics["ndcg@10"]
                logger.info(
                    "  %s blend_weight=%.2f -> nDCG@10 = %.4f",
                    short, w, metrics["ndcg@10"],
                )

            del reranker
            import gc, torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        all_results[name] = per_weight

    # --- Report -----------------------------------------------------
    print()
    print("=" * 72)
    print("RERANK BLEND SWEEP — nDCG@10")
    print("=" * 72)

    keys = list(all_results[args.datasets[0]].keys())
    width = max(len(k) for k in keys) + 2
    header = "Setting".ljust(width) + "".join(d.rjust(13) for d in args.datasets)
    header += "Average".rjust(13)
    print(header)
    print("-" * len(header))

    best, best_avg = None, -1.0
    for k in keys:
        vals = [all_results[d][k] for d in args.datasets]
        avg = sum(vals) / len(vals)
        print(k.ljust(width) + "".join(f"{v:.4f}".rjust(13) for v in vals)
              + f"{avg:.4f}".rjust(13))
        if avg > best_avg:
            best, best_avg = k, avg

    print()
    print(f"Best setting: {best}  (avg nDCG@10 = {best_avg:.4f})")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"Written to {out}")


if __name__ == "__main__":
    main()
