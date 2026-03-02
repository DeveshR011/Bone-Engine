"""
Elasticsearch Backend
=====================
Manages an ES index with two scoring signals:
  - BM25 on the ``content`` text field
  - SPLADE term weights stored as ``rank_features``

Hybrid query scoring:  FinalScore = α * BM25 + β * SPLADE
"""

from __future__ import annotations

import json
from typing import Any

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

from ..utils.logging_utils import Timer, get_logger

logger = get_logger(__name__)

# -------------------------------------------------------------------------
# Index mapping
# -------------------------------------------------------------------------

INDEX_MAPPING: dict[str, Any] = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "index.queries.cache.enabled": False,   # disable for honest latency
    },
    "mappings": {
        "properties": {
            "doc_id": {"type": "keyword"},
            "content": {
                "type": "text",
                "analyzer": "standard",
            },
            "splade_terms": {
                "type": "rank_features",        # sparse term weights
            },
            "metadata": {
                "type": "object",
                "enabled": False,
            },
        }
    },
}


class ElasticsearchBackend:
    """Elasticsearch index with BM25 + SPLADE rank_features hybrid search."""

    def __init__(
        self,
        host: str = "http://localhost:9200",
        index_name: str = "vectorless_search",
        timeout: int = 30,
        bm25_weight: float = 0.4,
        splade_weight: float = 0.6,
    ) -> None:
        self.es = Elasticsearch(host, request_timeout=timeout)
        self.index_name = index_name
        self.bm25_weight = bm25_weight
        self.splade_weight = splade_weight

        try:
            self.es.info()
        except Exception as exc:
            raise ConnectionError(f"Cannot reach Elasticsearch at {host}") from exc
        logger.info("Connected to Elasticsearch at %s", host)

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def create_index(self, delete_if_exists: bool = True) -> None:
        """Create (or recreate) the index with the defined mapping."""
        if self.es.indices.exists(index=self.index_name):
            if delete_if_exists:
                self.es.indices.delete(index=self.index_name)
                logger.info("Deleted existing index '%s'", self.index_name)
            else:
                logger.info("Index '%s' already exists — skipping creation", self.index_name)
                return

        self.es.indices.create(
            index=self.index_name,
            settings=INDEX_MAPPING["settings"],
            mappings=INDEX_MAPPING["mappings"],
        )
        logger.info("Created index '%s'", self.index_name)

    def bulk_index(
        self,
        doc_ids: list[str],
        contents: list[str],
        splade_dicts: list[dict[str, float]],
        chunk_size: int = 500,
    ) -> None:
        """Bulk-index documents with content and SPLADE term weights.

        Args:
            doc_ids: Unique document identifiers.
            contents: Raw text of each document.
            splade_dicts: Sparse {term: weight} dicts from SpladeEncoder.
            chunk_size: Bulk request size.
        """

        def _actions():
            for did, text, terms in zip(doc_ids, contents, splade_dicts):
                # rank_features requires all values > 0
                clean_terms = {k: v for k, v in terms.items() if v > 0}
                yield {
                    "_index": self.index_name,
                    "_id": did,
                    "_source": {
                        "doc_id": did,
                        "content": text,
                        "splade_terms": clean_terms,
                    },
                }

        with Timer("es-bulk-index") as t:
            success, errors = bulk(
                self.es,
                _actions(),
                chunk_size=chunk_size,
                raise_on_error=False,
            )

        self.es.indices.refresh(index=self.index_name)
        logger.info(
            "Indexed %d docs (%d errors) in %.1f ms",
            success,
            len(errors) if isinstance(errors, list) else 0,
            t.elapsed_ms,
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_text: str,
        splade_query: dict[str, float],
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Hybrid BM25 + SPLADE search.

        Combines a ``match`` query on ``content`` (BM25) with a sum of
        ``rank_feature`` queries on ``splade_terms`` using a scripted
        ``function_score`` wrapper.

        Returns list of {doc_id, score, content}.
        """
        # Build rank_feature clauses for each SPLADE query term
        rf_clauses = []
        for term, weight in splade_query.items():
            if weight > 0:
                rf_clauses.append({
                    "rank_feature": {
                        "field": f"splade_terms.{term}",
                        "boost": weight,
                    }
                })

        # Construct the hybrid query using `bool` with `should`
        es_query: dict[str, Any] = {
            "size": top_k,
            "query": {
                "bool": {
                    "should": [
                        # BM25 branch
                        {
                            "bool": {
                                "must": [
                                    {"match": {"content": {"query": query_text}}}
                                ],
                                "boost": self.bm25_weight,
                            }
                        },
                        # SPLADE branch — dis_max over rank_feature queries
                        {
                            "dis_max": {
                                "queries": rf_clauses if rf_clauses else [{"match_all": {}}],
                                "tie_breaker": 0.7,
                                "boost": self.splade_weight,
                            }
                        },
                    ]
                }
            },
        }

        with Timer("es-search") as t:
            resp = self.es.search(index=self.index_name, **es_query)

        hits = resp["hits"]["hits"]
        results = []
        for h in hits:
            results.append({
                "doc_id": h["_source"]["doc_id"],
                "score": h["_score"],
                "content": h["_source"].get("content", ""),
            })

        logger.info(
            "ES search returned %d hits in %.2f ms",
            len(results),
            t.elapsed_ms,
        )
        return results

    def search_bm25_only(self, query_text: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Pure BM25 search for baseline comparison."""
        es_query = {
            "size": top_k,
            "query": {"match": {"content": query_text}},
        }
        with Timer("es-bm25") as t:
            resp = self.es.search(index=self.index_name, **es_query)

        return [
            {
                "doc_id": h["_source"]["doc_id"],
                "score": h["_score"],
                "content": h["_source"].get("content", ""),
            }
            for h in resp["hits"]["hits"]
        ]

    def search_splade_only(
        self,
        splade_query: dict[str, float],
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Pure SPLADE rank_feature search for baseline comparison."""
        rf_clauses = [
            {"rank_feature": {"field": f"splade_terms.{term}", "boost": w}}
            for term, w in splade_query.items()
            if w > 0
        ]
        es_query = {
            "size": top_k,
            "query": {
                "dis_max": {
                    "queries": rf_clauses if rf_clauses else [{"match_all": {}}],
                    "tie_breaker": 0.7,
                }
            },
        }
        with Timer("es-splade") as t:
            resp = self.es.search(index=self.index_name, **es_query)

        return [
            {
                "doc_id": h["_source"]["doc_id"],
                "score": h["_score"],
                "content": h["_source"].get("content", ""),
            }
            for h in resp["hits"]["hits"]
        ]

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_index_size_mb(self) -> float:
        """Return the index size in megabytes."""
        stats = self.es.indices.stats(index=self.index_name)
        size_bytes = stats["indices"][self.index_name]["total"]["store"]["size_in_bytes"]
        return size_bytes / (1024 * 1024)

    def doc_count(self) -> int:
        return int(self.es.count(index=self.index_name)["count"])

    def get_mapping_json(self) -> str:
        """Return the index mapping as formatted JSON (for documentation)."""
        return json.dumps(INDEX_MAPPING, indent=2)
