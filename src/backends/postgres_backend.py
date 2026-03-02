"""
PostgreSQL GIN Backend
======================
Stores SPLADE sparse terms as JSONB and creates a GIN index for fast
key-existence queries.  Implements term-overlap scoring in SQL.

This backend is intentionally simpler than Elasticsearch — it demonstrates
that GIN indexes on JSONB can approximate sparse-term retrieval without
a dedicated search engine.
"""

from __future__ import annotations

import json
from typing import Any

import psycopg2
import psycopg2.extras

from ..utils.logging_utils import Timer, get_logger

logger = get_logger(__name__)


class PostgresBackend:
    """PostgreSQL + GIN for SPLADE sparse retrieval."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "vectorless_search",
        user: str = "postgres",
        password: str = "",
        table_name: str = "documents",
    ) -> None:
        self.table = table_name
        self.conn = psycopg2.connect(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password,
        )
        self.conn.autocommit = True
        logger.info("Connected to PostgreSQL %s:%d/%s", host, port, database)

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def create_table(self, drop_if_exists: bool = True) -> None:
        """Create the documents table with a GIN index on JSONB splade_terms."""
        cur = self.conn.cursor()

        if drop_if_exists:
            cur.execute(f"DROP TABLE IF EXISTS {self.table} CASCADE;")

        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.table} (
                doc_id   TEXT PRIMARY KEY,
                content  TEXT,
                splade_terms JSONB NOT NULL DEFAULT '{{}}'::jsonb
            );
        """)

        # GIN index on the JSONB keys for fast term-existence lookups
        cur.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{self.table}_splade_gin
            ON {self.table} USING GIN (splade_terms jsonb_path_ops);
        """)

        cur.close()
        logger.info("Table '%s' created with GIN index", self.table)

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def bulk_insert(
        self,
        doc_ids: list[str],
        contents: list[str],
        splade_dicts: list[dict[str, float]],
    ) -> None:
        """Insert documents into PostgreSQL."""
        cur = self.conn.cursor()

        with Timer("pg-bulk-insert") as t:
            values = []
            for did, text, terms in zip(doc_ids, contents, splade_dicts):
                # Ensure all values are positive for consistency
                clean = {k: v for k, v in terms.items() if v > 0}
                values.append((did, text, json.dumps(clean)))

            psycopg2.extras.execute_values(
                cur,
                f"""
                INSERT INTO {self.table} (doc_id, content, splade_terms)
                VALUES %s
                ON CONFLICT (doc_id) DO UPDATE SET
                    content = EXCLUDED.content,
                    splade_terms = EXCLUDED.splade_terms
                """,
                values,
                template="(%s, %s, %s::jsonb)",
                page_size=500,
            )

        cur.close()
        logger.info("Inserted %d docs in %.1f ms", len(doc_ids), t.elapsed_ms)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        splade_query: dict[str, float],
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Term-overlap scoring via SQL.

        For each document, the score is the sum of query_weight * doc_weight
        for every shared term.  This is essentially a sparse dot product.

        The query uses JSONB key access which is accelerated by the GIN index.
        """
        if not splade_query:
            return []

        # Build a SQL expression that sums matching term weights
        # We use a lateral unnest approach for flexibility
        terms = {k: v for k, v in splade_query.items() if v > 0}
        if not terms:
            return []

        # Build the scoring CTE: for each query term, extract the doc weight
        score_parts = []
        params: list[Any] = []
        for term, qw in terms.items():
            score_parts.append(
                f"COALESCE((splade_terms->>%s)::float, 0) * %s"
            )
            params.extend([term, qw])

        score_expr = " + ".join(score_parts)

        query = f"""
            SELECT doc_id, content, ({score_expr}) AS score
            FROM {self.table}
            WHERE splade_terms ?| %s
            ORDER BY score DESC
            LIMIT %s
        """
        params.append(list(terms.keys()))
        params.append(top_k)

        with Timer("pg-search") as t:
            cur = self.conn.cursor()
            cur.execute(query, params)
            rows = cur.fetchall()
            cur.close()

        results = [
            {"doc_id": r[0], "content": r[1], "score": float(r[2])}
            for r in rows
        ]

        logger.info("PG search: %d results in %.2f ms", len(results), t.elapsed_ms)
        return results

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def doc_count(self) -> int:
        cur = self.conn.cursor()
        cur.execute(f"SELECT COUNT(*) FROM {self.table}")
        count = cur.fetchone()[0]
        cur.close()
        return int(count)

    def get_table_size_mb(self) -> float:
        """Return total table + index size in MB."""
        cur = self.conn.cursor()
        cur.execute(f"SELECT pg_total_relation_size('{self.table}')")
        size_bytes = cur.fetchone()[0]
        cur.close()
        return size_bytes / (1024 * 1024)

    def close(self) -> None:
        self.conn.close()
