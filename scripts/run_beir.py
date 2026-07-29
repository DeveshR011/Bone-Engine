#!/usr/bin/env python3
"""
run_beir.py
===========
Benchmark the retrieval pipeline on BEIR — the standard zero-shot IR benchmark.

Reports nDCG@10 (the headline BEIR metric) per stage, so the contribution of
each component is visible rather than inferred:

    dense           bi-encoder only
    sparse          SPLADE only
    hybrid          weighted RRF over both
    hybrid+rerank   cross-encoder reranking of the shortlist

Usage:
    # Fastest real benchmark (~5k docs, minutes)
    python scripts/run_beir.py --datasets scifact

    # The laptop subset
    python scripts/run_beir.py --datasets scifact nfcorpus fiqa trec-covid

    # Ablation: skip the reranker
    python scripts/run_beir.py --datasets scifact --no-rerank

    # Smoke test on a truncated corpus (scores NOT comparable to published)
    python scripts/run_beir.py --datasets scifact --max-corpus 2000
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.beir_loader import DEFAULT_SUBSET, load_dataset
from src.encoder.dense_encoder import DenseEncoder
from src.encoder.splade_encoder import SpladeEncoder
from src.evaluation.beir_eval import evaluate_run, format_results_table, results_to_run
from src.pipeline.retrieval_pipeline import HybridRetriever, RetrievalConfig
from src.rerank.cross_encoder import CrossEncoderReranker
from src.utils.logging_utils import Timer, get_logger, load_config

logger = get_logger("run_beir")


def build_retriever(cfg: dict, args: argparse.Namespace) -> HybridRetriever:
    """Construct the pipeline once and reuse it across datasets."""
    dense_cfg = cfg["dense"]
    splade_cfg = cfg["splade"]
    bench_cfg = cfg.get("benchmark", {})

    dense_encoder = None
    if not args.no_dense:
        dense_encoder = DenseEncoder(
            model_name=dense_cfg["model_name"],
            device=dense_cfg.get("device", "auto"),
            max_length=dense_cfg.get("max_length", 512),
            fp16=dense_cfg.get("fp16", True),
        )

    splade_encoder = None
    if not args.no_sparse:
        splade_encoder = SpladeEncoder(
            model_name=splade_cfg["model_name"],
            device=splade_cfg.get("device", "auto"),
            max_length=splade_cfg.get("max_length", 256),
            top_k=splade_cfg.get("top_k", 100),
            quantize=splade_cfg.get("quantize", False),
            fp16=splade_cfg.get("fp16", True),
        )

    reranker = None
    if not args.no_rerank:
        rr_cfg = cfg.get("reranker", {})
        reranker = CrossEncoderReranker(
            model_name=rr_cfg.get("model_name", "BAAI/bge-reranker-v2-m3"),
            device=rr_cfg.get("device", "auto"),
            max_length=rr_cfg.get("max_length", 512),
            batch_size=rr_cfg.get("batch_size", 32),
            fp16=rr_cfg.get("fp16", True),
            blend_weight=rr_cfg.get("blend_weight", 0.3),
            blend_k=rr_cfg.get("blend_k", 60),
        )

    config = RetrievalConfig(
        top_k=args.top_k,
        rerank_top_n=bench_cfg.get("rerank_top_n", 100),
        final_k=bench_cfg.get("final_k", 10),
        dense_batch_size=dense_cfg.get("batch_size", 64),
        sparse_batch_size=splade_cfg.get("batch_size", 32),
        fusion_strategy=cfg["fusion"].get("strategy", "rrf"),
        alpha=cfg["fusion"].get("alpha", 0.4),
        beta=cfg["fusion"].get("beta", 0.6),
        rrf_k=cfg["fusion"].get("rrf_k", 60),
    )

    return HybridRetriever(
        dense_encoder=dense_encoder,
        splade_encoder=splade_encoder,
        reranker=reranker,
        config=config,
    )


def run_dataset(retriever: HybridRetriever, name: str, args: argparse.Namespace) -> dict:
    """Index, retrieve, and score one BEIR dataset."""
    logger.info("=" * 70)
    logger.info("Dataset: %s", name)
    logger.info("=" * 70)

    dataset = load_dataset(
        name,
        split=args.split,
        cache_dir=args.cache_dir,
        max_corpus_size=args.max_corpus,
    )

    doc_ids = dataset.doc_ids
    doc_texts = dataset.doc_texts(doc_ids)
    timings = retriever.index(doc_ids, doc_texts)

    query_ids = dataset.query_ids
    if args.max_queries is not None:
        query_ids = query_ids[: args.max_queries]
    query_texts = [dataset.queries[qid] for qid in query_ids]

    # Scoring must only consider the queries actually run, or unrun queries
    # would be counted as total failures.
    qrels = {qid: dataset.qrels[qid] for qid in query_ids if qid in dataset.qrels}

    logger.info("Running %d queries", len(query_texts))
    with Timer("search") as t:
        stages = retriever.search(query_texts, return_stages=True)
    timings["search_s"] = t.elapsed
    timings["ms_per_query"] = t.elapsed_ms / max(len(query_texts), 1)

    results_by_system: dict[str, dict[str, float]] = {}
    for stage, per_query in stages.items():
        run = results_to_run(dict(zip(query_ids, per_query)))
        results_by_system[stage] = evaluate_run(qrels, run)

    print()
    print(f"--- {name} ({len(doc_ids):,} docs, {len(query_ids):,} queries) ---")
    print(format_results_table(results_by_system))
    print()

    return {
        "dataset": name,
        "num_docs": len(doc_ids),
        "num_queries": len(query_ids),
        "truncated": args.max_corpus is not None or args.max_queries is not None,
        "timings": timings,
        "results": results_by_system,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the BEIR benchmark")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--datasets", nargs="+", default=DEFAULT_SUBSET,
        help=f"BEIR datasets to evaluate (default: {' '.join(DEFAULT_SUBSET)})",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--cache-dir", default="data/beir")
    parser.add_argument("--top-k", type=int, default=100,
                        help="Candidates retrieved per branch before fusion")
    parser.add_argument("--max-corpus", type=int, default=None,
                        help="Truncate corpus for smoke tests (breaks comparability)")
    parser.add_argument("--max-queries", type=int, default=None,
                        help="Run only the first N queries (smoke tests)")
    parser.add_argument("--dense-model", default=None,
                        help="Override the dense encoder (e.g. BAAI/bge-large-en-v1.5)")
    parser.add_argument("--reranker-model", default=None,
                        help="Override the cross-encoder reranker")
    parser.add_argument("--blend-weight", type=float, default=None,
                        help="Reranker authority: 0 ignores it, 1 lets it overwrite ranking")
    parser.add_argument("--no-rerank", action="store_true", help="Ablate the reranker")
    parser.add_argument("--no-dense", action="store_true", help="Ablate dense retrieval")
    parser.add_argument("--no-sparse", action="store_true", help="Ablate SPLADE retrieval")
    parser.add_argument("--output", default="results/beir_results.json")
    args = parser.parse_args()

    if args.no_dense and args.no_sparse:
        parser.error("--no-dense and --no-sparse together leave no retriever")

    cfg = load_config(args.config)

    # CLI overrides so model A/B tests do not require editing the config.
    if args.dense_model:
        cfg["dense"]["model_name"] = args.dense_model
    if args.reranker_model:
        cfg.setdefault("reranker", {})["model_name"] = args.reranker_model
    if args.blend_weight is not None:
        cfg.setdefault("reranker", {})["blend_weight"] = args.blend_weight

    retriever = build_retriever(cfg, args)

    all_results = []
    for name in args.datasets:
        try:
            all_results.append(run_dataset(retriever, name, args))
        except Exception as e:
            logger.error("Dataset '%s' failed: %s", name, e, exc_info=True)

    if not all_results:
        logger.error("No datasets completed successfully")
        sys.exit(1)

    # --- Summary across datasets ------------------------------------
    print("=" * 70)
    print("BEIR SUMMARY — nDCG@10")
    print("=" * 70)

    systems = sorted({s for r in all_results for s in r["results"]})
    name_w = max(len(s) for s in systems) + 2
    header = "System".ljust(name_w) + "".join(r["dataset"].rjust(14) for r in all_results)
    header += "Average".rjust(14)
    print(header)
    print("-" * len(header))

    for system in systems:
        scores = [r["results"].get(system, {}).get("ndcg@10") for r in all_results]
        present = [s for s in scores if s is not None]
        row = system.ljust(name_w)
        row += "".join(
            (f"{s:.4f}" if s is not None else "-").rjust(14) for s in scores
        )
        row += (f"{sum(present) / len(present):.4f}" if present else "-").rjust(14)
        print(row)

    if args.max_corpus:
        print("\nNOTE: --max-corpus truncated the corpora; these scores are not")
        print("      comparable to published BEIR results.")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "config": {
                    "dense_model": cfg["dense"]["model_name"],
                    "splade_model": cfg["splade"]["model_name"],
                    "reranker": None if args.no_rerank
                    else cfg.get("reranker", {}).get("model_name"),
                    "blend_weight": None if args.no_rerank
                    else cfg.get("reranker", {}).get("blend_weight", 0.7),
                    "fusion": cfg["fusion"].get("strategy"),
                    "fusion_alpha": cfg["fusion"].get("alpha"),
                    "fusion_beta": cfg["fusion"].get("beta"),
                    "top_k": args.top_k,
                },
                "datasets": all_results,
            },
            f,
            indent=2,
        )
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
