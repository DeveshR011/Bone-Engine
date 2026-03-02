#!/usr/bin/env python3
"""
benchmark_gpu_vs_cpu.py
=======================
Run encoding and retrieval benchmarks on CPU vs GPU.

Measures:
  - SPLADE encoding time (indexing + query)
  - Dense encoding time
  - Docs/sec throughput
  - Memory usage

Usage:
    python scripts/benchmark_gpu_vs_cpu.py --config config.yaml
    python scripts/benchmark_gpu_vs_cpu.py --config config.yaml --num-docs 1000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import psutil
import os

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.encoder.splade_encoder import SpladeEncoder
from src.encoder.dense_encoder import DenseEncoder
from src.evaluation.metrics import EvaluationMetrics
from src.utils.logging_utils import Timer, get_logger, load_config, ExperimentResult, log_experiment

logger = get_logger("benchmark")


def get_memory_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def load_documents(path: str, max_docs: int = 0) -> tuple[list[str], list[str]]:
    doc_ids, contents = [], []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_docs > 0 and i >= max_docs:
                break
            obj = json.loads(line)
            doc_ids.append(str(obj["doc_id"]))
            contents.append(obj["content"])
    return doc_ids, contents


def benchmark_splade(device: str, texts: list[str], cfg: dict) -> dict:
    """Benchmark SPLADE encoding on a specific device."""
    sp_cfg = cfg["splade"]
    mem_before = get_memory_mb()

    encoder = SpladeEncoder(
        model_name=sp_cfg["model_name"],
        device=device,
        max_length=sp_cfg["max_length"],
        top_k=sp_cfg["top_k"],
        quantize=False,
    )

    # Warm up
    _ = encoder.encode(texts[:2], batch_size=2, show_progress=False)

    # Benchmark indexing
    with Timer("splade-index") as t_index:
        _ = encoder.encode(texts, batch_size=sp_cfg["batch_size"], show_progress=False)
    indexing_time = t_index.elapsed

    # Benchmark query
    query_times = []
    for text in texts[:min(50, len(texts))]:
        with Timer("q") as t_q:
            _ = encoder.encode_query(text)
        query_times.append(t_q.elapsed_ms)

    mem_after = get_memory_mb()

    # Quantized variant
    encoder_q = SpladeEncoder(
        model_name=sp_cfg["model_name"],
        device=device,
        max_length=sp_cfg["max_length"],
        top_k=sp_cfg["top_k"],
        quantize=True,
    )
    with Timer("splade-quantized") as t_q_enc:
        _ = encoder_q.encode(texts, batch_size=sp_cfg["batch_size"], show_progress=False)

    return {
        "device": device,
        "indexing_time_s": indexing_time,
        "docs_per_sec": len(texts) / indexing_time if indexing_time > 0 else 0,
        "avg_query_latency_ms": float(np.mean(query_times)),
        "p95_query_latency_ms": float(np.percentile(query_times, 95)),
        "memory_delta_mb": mem_after - mem_before,
        "memory_total_mb": mem_after,
        "quantized_indexing_time_s": t_q_enc.elapsed,
    }


def benchmark_dense(device: str, texts: list[str], cfg: dict) -> dict:
    """Benchmark dense encoding on a specific device."""
    dense_cfg = cfg["dense"]
    mem_before = get_memory_mb()

    encoder = DenseEncoder(
        model_name=dense_cfg["model_name"],
        device=device,
        max_length=dense_cfg["max_length"],
    )

    # Warm up
    _ = encoder.encode(texts[:2], batch_size=2, show_progress=False)

    # Benchmark indexing
    with Timer("dense-index") as t_index:
        _ = encoder.encode(texts, batch_size=dense_cfg["batch_size"], show_progress=False)
    indexing_time = t_index.elapsed

    # Benchmark query
    query_times = []
    for text in texts[:min(50, len(texts))]:
        with Timer("q") as t_q:
            _ = encoder.encode_query(text)
        query_times.append(t_q.elapsed_ms)

    mem_after = get_memory_mb()

    return {
        "device": device,
        "indexing_time_s": indexing_time,
        "docs_per_sec": len(texts) / indexing_time if indexing_time > 0 else 0,
        "avg_query_latency_ms": float(np.mean(query_times)),
        "p95_query_latency_ms": float(np.percentile(query_times, 95)),
        "memory_delta_mb": mem_after - mem_before,
        "memory_total_mb": mem_after,
    }


def main():
    parser = argparse.ArgumentParser(description="GPU vs CPU benchmark")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--num-docs", type=int, default=0,
                        help="Max docs to benchmark (0 = all)")
    parser.add_argument("--documents", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    doc_path = args.documents or cfg["data"]["documents_path"]
    doc_ids, contents = load_documents(doc_path, args.num_docs)
    logger.info("Benchmarking with %d documents", len(contents))

    results_dir = cfg["evaluation"]["results_dir"]

    # Determine available devices
    import torch
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
        logger.info("GPU detected: %s", torch.cuda.get_device_name(0))
    else:
        logger.info("No GPU available — running CPU-only benchmark")

    # --- SPLADE benchmarks ---
    logger.info("=" * 60)
    logger.info("SPLADE Encoder Benchmarks")
    logger.info("=" * 60)

    for device in devices:
        logger.info("--- SPLADE on %s ---", device)
        result = benchmark_splade(device, contents, cfg)

        for k, v in result.items():
            logger.info("  %-30s %s", k, f"{v:.2f}" if isinstance(v, float) else v)

        log_experiment(
            ExperimentResult(
                experiment="gpu_vs_cpu",
                system="splade",
                variant=device,
                query_latency_ms=result["avg_query_latency_ms"],
                memory_usage_mb=result["memory_total_mb"],
                indexing_time_s=result["indexing_time_s"],
                docs_per_sec=result["docs_per_sec"],
                notes=f"p95={result['p95_query_latency_ms']:.2f}ms, "
                      f"quantized_index={result['quantized_indexing_time_s']:.2f}s",
            ),
            results_dir,
        )

    # --- Dense benchmarks ---
    logger.info("=" * 60)
    logger.info("Dense Encoder Benchmarks")
    logger.info("=" * 60)

    for device in devices:
        logger.info("--- Dense on %s ---", device)
        result = benchmark_dense(device, contents, cfg)

        for k, v in result.items():
            logger.info("  %-30s %s", k, f"{v:.2f}" if isinstance(v, float) else v)

        log_experiment(
            ExperimentResult(
                experiment="gpu_vs_cpu",
                system="dense",
                variant=device,
                query_latency_ms=result["avg_query_latency_ms"],
                memory_usage_mb=result["memory_total_mb"],
                indexing_time_s=result["indexing_time_s"],
                docs_per_sec=result["docs_per_sec"],
                notes=f"p95={result['p95_query_latency_ms']:.2f}ms",
            ),
            results_dir,
        )

    # --- Summary ---
    logger.info("=" * 60)
    logger.info("Benchmark complete. Results in %s/experiments.csv", results_dir)


if __name__ == "__main__":
    main()
