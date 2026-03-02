#!/usr/bin/env python3
"""
build_dense_faiss.py
====================
Encode documents with a dense model (sentence-transformers) and build a
FAISS index for approximate nearest-neighbor search.

Usage:
    python scripts/build_dense_faiss.py --config config.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backends.faiss_backend import FAISSDenseBackend
from src.encoder.dense_encoder import DenseEncoder
from src.utils.logging_utils import Timer, get_logger, load_config

logger = get_logger("build_dense_faiss")


def load_documents(path: str) -> tuple[list[str], list[str]]:
    doc_ids, contents = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            doc_ids.append(str(obj["doc_id"]))
            contents.append(obj["content"])
    return doc_ids, contents


def main():
    parser = argparse.ArgumentParser(description="Build dense FAISS index")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--documents", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    doc_path = args.documents or cfg["data"]["documents_path"]
    dense_cfg = cfg["dense"]
    faiss_cfg = cfg["faiss"]

    # Load documents
    doc_ids, contents = load_documents(doc_path)
    logger.info("Loaded %d documents", len(doc_ids))

    # Dense encode
    encoder = DenseEncoder(
        model_name=dense_cfg["model_name"],
        device=dense_cfg["device"],
        max_length=dense_cfg["max_length"],
    )

    with Timer("dense-encode-all") as t:
        embeddings = encoder.encode(contents, batch_size=dense_cfg["batch_size"])
    logger.info("Dense encoding: %.1f s  (%.1f docs/s)",
                t.elapsed, len(contents) / t.elapsed if t.elapsed > 0 else 0)

    # Save embeddings for reuse
    emb_path = Path("data") / "dense_embeddings.npy"
    np.save(str(emb_path), embeddings)
    logger.info("Saved embeddings to %s  (%.2f MB)",
                emb_path, emb_path.stat().st_size / (1024 * 1024))

    # Build FAISS index
    backend = FAISSDenseBackend(
        dim=encoder.embedding_dim,
        index_type=faiss_cfg["index_type"],
        nlist=faiss_cfg["nlist"],
        nprobe=faiss_cfg["nprobe"],
        index_path=faiss_cfg["index_path"],
    )

    with Timer("faiss-build") as t:
        backend.build_index(doc_ids, embeddings, contents)
    logger.info("FAISS build: %.1f s", t.elapsed)

    backend.save()
    logger.info("FAISS index size: %.2f MB", backend.get_index_size_mb())


if __name__ == "__main__":
    main()
