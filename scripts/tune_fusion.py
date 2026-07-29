#!/usr/bin/env python3
"""
tune_fusion.py
==============
Sweep hybrid-fusion settings on BEIR.

The sparse/dense weighting (``alpha``/``beta``) and the candidate depth
(``top_k``) were chosen by convention, not measurement. Both matter: the
sparse and dense branches have different strengths per dataset, and candidate
depth sets the recall ceiling that any reranker is later limited by.

Documents are encoded once and every setting is evaluated against the same
cached hits, so the whole sweep costs about one benchmark run.

Usage:
    python scripts/tune_fusion.py --datasets scifact nfcorpus
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
from src.fusion.hybrid_fusion import HybridFusion
from src.pipeline.retrieval_pipeline import HybridRetriever, RetrievalConfig
from src.utils.logging_utils import get_logger, load_config

logger = get_logger("tune_fusion")

# beta (dense weight) is 1 - alpha; only the ratio matters to RRF.
DEFAULT_ALPHAS = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune hybrid fusion weights")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--datasets", nargs="+", default=["scifact", "nfcorpus"])
    parser.add_argument("--alphas", nargs="+", type=float, default=DEFAULT_ALPHAS)
    parser.add_argument("--strategies", nargs="+", default=["rrf", "linear"])
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--output", default="results/fusion_sweep.json")
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

        # Retrieve each branch once; fusion settings are pure post-processing.
        dense_hits = retriever.retrieve_dense(query_texts, args.top_k)
        sparse_hits = retriever.retrieve_sparse(query_texts, args.top_k)

        def score(hits) -> float:
            return evaluate_run(
                dataset.qrels, results_to_run(dict(zip(query_ids, hits)))
            )["ndcg@10"]

        per_setting: dict[str, float] = {
            "dense_only": score(dense_hits),
            "sparse_only": score(sparse_hits),
        }
        logger.info(
            "%s dense=%.4f sparse=%.4f",
            name, per_setting["dense_only"], per_setting["sparse_only"],
        )

        for strategy in args.strategies:
            for alpha in args.alphas:
                fusion = HybridFusion(
                    strategy=strategy,
                    alpha=alpha,
                    beta=1.0 - alpha,
                    gamma=0.0,
                    rrf_k=cfg["fusion"].get("rrf_k", 60),
                )
                fused = [
                    fusion.fuse(s, d)[: args.top_k]
                    for s, d in zip(sparse_hits, dense_hits)
                ]
                key = f"{strategy} a={alpha:.1f}"
                per_setting[key] = score(fused)
                logger.info("  %s -> nDCG@10 = %.4f", key, per_setting[key])

        all_results[name] = per_setting

    # --- Report -----------------------------------------------------
    print()
    print("=" * 74)
    print("FUSION SWEEP — nDCG@10   (alpha = sparse weight, beta = 1 - alpha)")
    print("=" * 74)

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
