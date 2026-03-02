#!/usr/bin/env python3
"""
run_queries.py
==============
Run queries against all retrieval backends and fuse results.
Supports: ES (BM25, SPLADE, hybrid), HNSW sparse, FAISS dense,
PostgreSQL GIN, graph expansion, hybrid fusion, and RAG.

Usage:
    python scripts/run_queries.py --config config.yaml
    python scripts/run_queries.py --config config.yaml --query "what is information retrieval?"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.encoder.splade_encoder import SpladeEncoder
from src.encoder.dense_encoder import DenseEncoder
from src.fusion.hybrid_fusion import HybridFusion
from src.graph.graph_expansion import GraphExpansion
from src.rag.rag_pipeline import RAGPipeline
from src.utils.logging_utils import Timer, get_logger, load_config

logger = get_logger("run_queries")


def load_queries(path: str) -> list[dict]:
    """Load queries from JSONL.  Each line: {"query_id": "...", "text": "..."}"""
    queries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            queries.append(json.loads(line))
    return queries


def load_sparse_data(path: str):
    doc_ids, splade_dicts = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            doc_ids.append(obj["doc_id"])
            splade_dicts.append(obj["splade_terms"])
    return doc_ids, splade_dicts


def main():
    parser = argparse.ArgumentParser(description="Run queries")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--query", default=None, help="Single query (overrides file)")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--rag", action="store_true", help="Run RAG pipeline")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # --- Initialize encoders ---
    sp_cfg = cfg["splade"]
    splade_enc = SpladeEncoder(
        model_name=sp_cfg["model_name"],
        device=sp_cfg["device"],
        max_length=sp_cfg["max_length"],
        top_k=sp_cfg["top_k"],
        quantize=sp_cfg["quantize"],
    )
    dense_enc = DenseEncoder(
        model_name=cfg["dense"]["model_name"],
        device=cfg["dense"]["device"],
        max_length=cfg["dense"]["max_length"],
    )

    # --- Determine queries ---
    if args.query:
        queries = [{"query_id": "q0", "text": args.query}]
    else:
        queries = load_queries(cfg["data"]["queries_path"])
    logger.info("Running %d queries", len(queries))

    # --- Load backends (gracefully skip unavailable ones) ---
    backends = {}

    # Elasticsearch
    try:
        from src.backends.elasticsearch_backend import ElasticsearchBackend
        es_cfg = cfg["elasticsearch"]
        backends["es"] = ElasticsearchBackend(
            host=es_cfg["host"], index_name=es_cfg["index_name"],
            bm25_weight=es_cfg["bm25_weight"], splade_weight=es_cfg["splade_weight"],
        )
    except Exception as e:
        logger.warning("ES unavailable: %s", e)

    # HNSW
    try:
        from src.backends.hnsw_backend import HNSWSparseBackend
        hnsw_cfg = cfg["hnsw"]
        hnsw = HNSWSparseBackend(
            dim=hnsw_cfg["vocab_dim"],
            ef_search=hnsw_cfg["ef_search"],
            index_path=hnsw_cfg["index_path"],
        )
        hnsw.load()
        backends["hnsw"] = hnsw
    except Exception as e:
        logger.warning("HNSW unavailable: %s", e)

    # FAISS
    try:
        from src.backends.faiss_backend import FAISSDenseBackend
        faiss_cfg = cfg["faiss"]
        faiss_be = FAISSDenseBackend(
            dim=cfg["dense"]["embedding_dim"],
            index_type=faiss_cfg["index_type"],
            nprobe=faiss_cfg["nprobe"],
            index_path=faiss_cfg["index_path"],
        )
        faiss_be.load()
        backends["faiss"] = faiss_be
    except Exception as e:
        logger.warning("FAISS unavailable: %s", e)

    # PostgreSQL
    try:
        from src.backends.postgres_backend import PostgresBackend
        pg_cfg = cfg["postgres"]
        backends["pg"] = PostgresBackend(
            host=pg_cfg["host"], port=pg_cfg["port"],
            database=pg_cfg["database"], user=pg_cfg["user"],
            password=pg_cfg["password"], table_name=pg_cfg["table_name"],
        )
    except Exception as e:
        logger.warning("PG unavailable: %s", e)

    # Graph expansion
    graph_exp = None
    try:
        sparse_path = "data/splade_sparse.jsonl"
        doc_ids, splade_dicts = load_sparse_data(sparse_path)
        graph_cfg = cfg["graph"]
        graph_exp = GraphExpansion(
            weight_threshold=graph_cfg["weight_threshold"],
            max_neighbors=graph_cfg["max_neighbors"],
            hop_decay=graph_cfg["hop_decay"],
        )
        graph_exp.build_bipartite_graph(doc_ids, splade_dicts)
        graph_exp.build_doc_similarity_graph()
        logger.info("Graph: %s", graph_exp.stats())
    except Exception as e:
        logger.warning("Graph expansion unavailable: %s", e)

    # Fusion
    fuse_cfg = cfg["fusion"]
    fusion = HybridFusion(
        strategy=fuse_cfg["strategy"],
        alpha=fuse_cfg["alpha"],
        beta=fuse_cfg["beta"],
        gamma=fuse_cfg["gamma"],
        rrf_k=fuse_cfg["rrf_k"],
    )

    # --- Run queries ---
    all_results = []
    for q in queries:
        qid = q["query_id"]
        text = q["text"]
        logger.info("--- Query %s: %s ---", qid, text[:80])

        # Encode query
        q_sparse = splade_enc.encode_query(text)
        q_dense = dense_enc.encode_query(text)

        result_entry = {"query_id": qid, "text": text}

        # Sparse retrieval (ES)
        if "es" in backends:
            es_results = backends["es"].search(text, q_sparse, top_k=args.top_k)
            result_entry["es_hybrid"] = es_results
            bm25_results = backends["es"].search_bm25_only(text, top_k=args.top_k)
            result_entry["es_bm25"] = bm25_results

        # HNSW sparse
        if "hnsw" in backends:
            hnsw_results = backends["hnsw"].search(
                q_sparse, top_k=args.top_k, tokenizer=splade_enc.tokenizer
            )
            result_entry["hnsw_sparse"] = hnsw_results

        # FAISS dense
        if "faiss" in backends:
            faiss_results = backends["faiss"].search(q_dense, top_k=args.top_k)
            result_entry["faiss_dense"] = faiss_results

        # PostgreSQL
        if "pg" in backends:
            pg_results = backends["pg"].search(q_sparse, top_k=args.top_k)
            result_entry["pg_sparse"] = pg_results

        # Graph expansion
        if graph_exp is not None:
            graph_results = graph_exp.score_documents(q_sparse, max_results=args.top_k)
            result_entry["graph"] = graph_results
        else:
            graph_results = []

        # Hybrid fusion
        sparse_for_fusion = result_entry.get("es_hybrid", result_entry.get("hnsw_sparse", []))
        dense_for_fusion = result_entry.get("faiss_dense", [])
        fused = fusion.fuse(sparse_for_fusion, dense_for_fusion, graph_results or None)
        result_entry["hybrid_fused"] = fused[:args.top_k]

        # Print top results
        logger.info("Top-%d fused results:", args.top_k)
        for r in fused[:5]:
            logger.info("  %s  score=%.4f", r["doc_id"], r["score"])

        # RAG
        if args.rag:
            rag_cfg = cfg["rag"]
            rag = RAGPipeline(
                llm_backend=rag_cfg["llm_backend"],
                openai_model=rag_cfg["openai_model"],
                hf_model=rag_cfg["hf_model"],
                top_k=rag_cfg["top_k"],
                max_context_tokens=rag_cfg["max_context_tokens"],
                temperature=rag_cfg["temperature"],
            )
            rag_result = rag.answer(text, retrieval_fn=lambda q: fused[:rag_cfg["top_k"]])
            result_entry["rag"] = rag_result
            logger.info("RAG answer: %s", rag_result["answer"][:200])

        all_results.append(result_entry)

    # Save results
    out_path = Path("results") / "query_results.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in all_results:
            f.write(json.dumps(r, default=str) + "\n")
    logger.info("Results saved to %s", out_path)

    # Close PG
    if "pg" in backends:
        backends["pg"].close()


if __name__ == "__main__":
    main()
