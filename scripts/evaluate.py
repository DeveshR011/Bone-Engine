#!/usr/bin/env python3
"""
evaluate.py
===========
Load query results and ground-truth relevance judgments, compute IR metrics
(Recall@K, nDCG@K, MRR), and log to CSV.

Usage:
    python scripts/evaluate.py --config config.yaml
    python scripts/evaluate.py --results results/query_results.jsonl --qrels data/qrels.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.metrics import EvaluationMetrics
from src.utils.logging_utils import get_logger, load_config

logger = get_logger("evaluate")


def load_qrels(path: str) -> dict[str, set[str]]:
    """Load ground-truth relevance judgments.

    Each line: {"query_id": "q1", "relevant_doc_ids": ["d1", "d2", ...]}
    Returns {query_id: {relevant doc_ids}}.
    """
    qrels = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            qid = str(obj["query_id"])
            qrels[qid] = set(str(d) for d in obj["relevant_doc_ids"])
    return qrels


def load_results(path: str) -> list[dict]:
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            results.append(json.loads(line))
    return results


def extract_doc_ids(result_list: list[dict]) -> list[str]:
    """Extract ordered doc_ids from a result list."""
    return [r["doc_id"] for r in result_list]


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval results")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--results", default="results/query_results.jsonl")
    parser.add_argument("--qrels", default=None, help="Override qrels path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    qrels_path = args.qrels or cfg["data"]["qrels_path"]
    k_values = cfg["evaluation"]["k_values"]
    results_dir = cfg["evaluation"]["results_dir"]

    # Load data
    qrels = load_qrels(qrels_path)
    results = load_results(args.results)
    logger.info("Loaded %d qrels, %d query results", len(qrels), len(results))

    evaluator = EvaluationMetrics()

    # Systems to evaluate (keys from run_queries output)
    systems = [
        ("es_bm25", "BM25 (ES)"),
        ("es_hybrid", "BM25 + SPLADE (ES)"),
        ("hnsw_sparse", "HNSW Sparse"),
        ("faiss_dense", "FAISS Dense"),
        ("pg_sparse", "PostgreSQL GIN"),
        ("graph", "Graph Expansion"),
        ("hybrid_fused", "Hybrid Fusion"),
    ]

    for sys_key, sys_name in systems:
        pairs = []
        for r in results:
            qid = str(r["query_id"])
            if qid not in qrels:
                continue
            if sys_key not in r:
                continue
            retrieved = extract_doc_ids(r[sys_key])
            relevant = qrels[qid]
            pairs.append((retrieved, relevant))

        if not pairs:
            logger.info("No data for %s — skipping", sys_name)
            continue

        metrics = evaluator.evaluate_queries(pairs, k_values=k_values)
        logger.info("=== %s ===", sys_name)
        for name, value in sorted(metrics.items()):
            logger.info("  %-15s %.4f", name, value)

        evaluator.log_result(
            experiment="retrieval_comparison",
            system=sys_name,
            variant="default",
            metrics=metrics,
            results_dir=results_dir,
        )

    logger.info("Evaluation complete. Results in %s/experiments.csv", results_dir)


if __name__ == "__main__":
    main()
