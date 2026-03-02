#!/usr/bin/env python3
"""
build_sparse_hnsw.py
====================
Load pre-computed SPLADE sparse representations and build an HNSW index.

Usage:
    python scripts/build_sparse_hnsw.py --config config.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backends.hnsw_backend import HNSWSparseBackend
from src.encoder.splade_encoder import SpladeEncoder
from src.utils.logging_utils import Timer, get_logger, load_config

logger = get_logger("build_sparse_hnsw")


def load_sparse_data(path: str) -> tuple[list[str], list[dict[str, float]], list[str]]:
    """Load doc_ids and SPLADE dicts from JSONL."""
    doc_ids, splade_dicts, contents = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            doc_ids.append(obj["doc_id"])
            splade_dicts.append(obj["splade_terms"])
            contents.append(obj.get("content", ""))
    return doc_ids, splade_dicts, contents


def main():
    parser = argparse.ArgumentParser(description="Build sparse HNSW index")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--sparse-data", default="data/splade_sparse.jsonl",
                        help="Path to pre-computed SPLADE JSONL")
    args = parser.parse_args()

    cfg = load_config(args.config)
    hnsw_cfg = cfg["hnsw"]

    # Load sparse data
    doc_ids, splade_dicts, contents = load_sparse_data(args.sparse_data)
    logger.info("Loaded %d sparse vectors", len(doc_ids))

    # Load tokenizer for accurate token→index mapping
    sp_cfg = cfg["splade"]
    logger.info("Loading tokenizer for index mapping…")
    encoder = SpladeEncoder(model_name=sp_cfg["model_name"], device="cpu")
    tokenizer = encoder.tokenizer

    # Build HNSW
    backend = HNSWSparseBackend(
        dim=hnsw_cfg["vocab_dim"],
        ef_construction=hnsw_cfg["ef_construction"],
        M=hnsw_cfg["M"],
        ef_search=hnsw_cfg["ef_search"],
        num_threads=hnsw_cfg["num_threads"],
        index_path=hnsw_cfg["index_path"],
    )

    with Timer("hnsw-total-build") as t:
        backend.build_index(doc_ids, splade_dicts, contents, tokenizer=tokenizer)

    logger.info("Total HNSW build: %.1f s", t.elapsed)

    # Save
    backend.save()
    logger.info("Index size: %.2f MB", backend.get_index_size_mb())


if __name__ == "__main__":
    main()
