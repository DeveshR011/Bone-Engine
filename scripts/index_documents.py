#!/usr/bin/env python3
"""
index_documents.py
==================
Load documents from JSONL, encode with SPLADE, and index into:
  - Elasticsearch  (BM25 + rank_features)
  - PostgreSQL     (JSONB + GIN)

Usage:
    python scripts/index_documents.py --config config.yaml
    python scripts/index_documents.py --config config.yaml --backend es
    python scripts/index_documents.py --config config.yaml --backend pg
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.encoder.splade_encoder import SpladeEncoder
from src.utils.logging_utils import Timer, get_logger, load_config

logger = get_logger("index_documents")


def load_documents(path: str) -> tuple[list[str], list[str]]:
    """Load documents from JSONL.  Each line: {"doc_id": "...", "content": "..."}"""
    doc_ids, contents = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            doc_ids.append(str(obj["doc_id"]))
            contents.append(obj["content"])
    logger.info("Loaded %d documents from %s", len(doc_ids), path)
    return doc_ids, contents


def index_elasticsearch(cfg: dict, doc_ids, contents, splade_dicts):
    from src.backends.elasticsearch_backend import ElasticsearchBackend

    es_cfg = cfg["elasticsearch"]
    backend = ElasticsearchBackend(
        host=es_cfg["host"],
        index_name=es_cfg["index_name"],
        timeout=es_cfg["timeout"],
        bm25_weight=es_cfg["bm25_weight"],
        splade_weight=es_cfg["splade_weight"],
    )
    backend.create_index(delete_if_exists=True)
    backend.bulk_index(doc_ids, contents, splade_dicts, chunk_size=es_cfg["bulk_chunk_size"])
    logger.info("ES index size: %.2f MB, doc count: %d", backend.get_index_size_mb(), backend.doc_count())


def index_postgres(cfg: dict, doc_ids, contents, splade_dicts):
    from src.backends.postgres_backend import PostgresBackend

    pg_cfg = cfg["postgres"]
    backend = PostgresBackend(
        host=pg_cfg["host"],
        port=pg_cfg["port"],
        database=pg_cfg["database"],
        user=pg_cfg["user"],
        password=pg_cfg["password"],
        table_name=pg_cfg["table_name"],
    )
    backend.create_table(drop_if_exists=True)
    backend.bulk_insert(doc_ids, contents, splade_dicts)
    logger.info("PG doc count: %d, size: %.2f MB", backend.doc_count(), backend.get_table_size_mb())
    backend.close()


def main():
    parser = argparse.ArgumentParser(description="Index documents with SPLADE")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--backend", choices=["es", "pg", "all"], default="all",
                        help="Which backend(s) to index into")
    parser.add_argument("--documents", default=None,
                        help="Override path to documents JSONL")
    args = parser.parse_args()

    cfg = load_config(args.config)
    doc_path = args.documents or cfg["data"]["documents_path"]

    # Load documents
    doc_ids, contents = load_documents(doc_path)

    # SPLADE encode
    sp_cfg = cfg["splade"]
    encoder = SpladeEncoder(
        model_name=sp_cfg["model_name"],
        device=sp_cfg["device"],
        max_length=sp_cfg["max_length"],
        top_k=sp_cfg["top_k"],
        quantize=sp_cfg["quantize"],
    )

    with Timer("splade-encode-all") as t:
        splade_dicts = encoder.encode(contents, batch_size=sp_cfg["batch_size"])
    logger.info("Total SPLADE encoding: %.1f s  (%.1f docs/s)",
                t.elapsed, len(contents) / t.elapsed if t.elapsed > 0 else 0)

    # Save sparse representations for reuse
    sparse_path = Path("data") / "splade_sparse.jsonl"
    with open(sparse_path, "w", encoding="utf-8") as f:
        for did, sd in zip(doc_ids, splade_dicts):
            f.write(json.dumps({"doc_id": did, "splade_terms": sd}) + "\n")
    logger.info("Saved sparse representations to %s", sparse_path)

    # Index
    if args.backend in ("es", "all"):
        try:
            index_elasticsearch(cfg, doc_ids, contents, splade_dicts)
        except Exception as e:
            logger.error("Elasticsearch indexing failed: %s", e)

    if args.backend in ("pg", "all"):
        try:
            index_postgres(cfg, doc_ids, contents, splade_dicts)
        except Exception as e:
            logger.error("PostgreSQL indexing failed: %s", e)

    logger.info("Indexing complete.")


if __name__ == "__main__":
    main()
