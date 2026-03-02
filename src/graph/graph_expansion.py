"""
Graph Expansion Layer
=====================
Builds a bipartite graph of Documents ↔ SPLADE Terms, derives doc-doc
similarity edges from shared terms, and implements 2-hop graph traversal
scoring to enrich retrieval results.

FinalScore = α*Sparse + β*Dense + γ*GraphScore
GraphScore(doc) = Σ  hop_decay^d * edge_weight(neighbor)
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import networkx as nx

from ..utils.logging_utils import Timer, get_logger

logger = get_logger(__name__)


class GraphExpansion:
    """Bipartite doc-term graph with 2-hop expansion scoring."""

    def __init__(
        self,
        weight_threshold: float = 0.5,
        max_neighbors: int = 50,
        hop_decay: float = 0.5,
    ) -> None:
        self.weight_threshold = weight_threshold
        self.max_neighbors = max_neighbors
        self.hop_decay = hop_decay

        self.graph = nx.Graph()
        self.doc_graph = nx.Graph()          # doc-doc similarity edges
        self.doc_ids: set[str] = set()
        self.term_nodes: set[str] = set()

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build_bipartite_graph(
        self,
        doc_ids: list[str],
        splade_dicts: list[dict[str, float]],
    ) -> None:
        """Build the bipartite doc ↔ term graph.

        Adds an edge (doc, term) with weight = SPLADE weight whenever
        weight > threshold.
        """
        logger.info("Building bipartite graph (threshold=%.2f)…", self.weight_threshold)
        with Timer("build-bipartite") as t:
            for doc_id, terms in zip(doc_ids, splade_dicts):
                doc_node = f"doc:{doc_id}"
                self.graph.add_node(doc_node, bipartite=0)
                self.doc_ids.add(doc_node)

                for term, weight in terms.items():
                    if weight < self.weight_threshold:
                        continue
                    term_node = f"term:{term}"
                    self.graph.add_node(term_node, bipartite=1)
                    self.term_nodes.add(term_node)
                    self.graph.add_edge(doc_node, term_node, weight=weight)

        logger.info(
            "Bipartite graph: %d doc nodes, %d term nodes, %d edges  (%.1f ms)",
            len(self.doc_ids),
            len(self.term_nodes),
            self.graph.number_of_edges(),
            t.elapsed_ms,
        )

    def build_doc_similarity_graph(self) -> None:
        """Derive doc-doc edges from shared terms.

        Edge weight = Σ min(w_a(t), w_b(t))  for each shared term t.
        Only the top max_neighbors edges per doc are kept.
        """
        logger.info("Building doc-doc similarity graph…")
        with Timer("build-doc-graph") as t:
            # Inverted index: term → list of (doc_node, weight)
            term_to_docs: dict[str, list[tuple[str, float]]] = defaultdict(list)
            for doc_node in self.doc_ids:
                for neighbor in self.graph.neighbors(doc_node):
                    w = self.graph[doc_node][neighbor]["weight"]
                    term_to_docs[neighbor].append((doc_node, w))

            # Compute pairwise similarities
            pair_scores: dict[tuple[str, str], float] = defaultdict(float)
            for term_node, doc_list in term_to_docs.items():
                for i, (d1, w1) in enumerate(doc_list):
                    for d2, w2 in doc_list[i + 1 :]:
                        key = (min(d1, d2), max(d1, d2))
                        pair_scores[key] += min(w1, w2)

            # Add edges (keeping top-K per doc)
            doc_edges: dict[str, list[tuple[str, float]]] = defaultdict(list)
            for (d1, d2), score in pair_scores.items():
                doc_edges[d1].append((d2, score))
                doc_edges[d2].append((d1, score))

            for doc, neighbors in doc_edges.items():
                neighbors.sort(key=lambda x: x[1], reverse=True)
                for nbr, score in neighbors[: self.max_neighbors]:
                    self.doc_graph.add_edge(doc, nbr, weight=score)

        logger.info(
            "Doc-doc graph: %d edges  (%.1f ms)",
            self.doc_graph.number_of_edges(),
            t.elapsed_ms,
        )

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score_documents(
        self,
        query_splade: dict[str, float],
        candidate_doc_ids: list[str] | None = None,
        max_results: int = 50,
    ) -> list[dict[str, Any]]:
        """2-hop graph traversal scoring from query terms.

        1. Start from query term nodes present in the graph.
        2. Walk 1 hop to reach document nodes  (hop 1 score).
        3. Walk a second hop through the doc-doc graph to reach
           neighboring documents  (hop 2 score with decay).

        GraphScore(doc) = Σ  hop_decay^depth * edge_weight
        """
        scores: dict[str, float] = defaultdict(float)

        with Timer("graph-score") as t:
            # Hop 1: query terms → documents
            for term, qw in query_splade.items():
                term_node = f"term:{term}"
                if term_node not in self.graph:
                    continue
                for doc_node in self.graph.neighbors(term_node):
                    if not doc_node.startswith("doc:"):
                        continue
                    edge_w = self.graph[term_node][doc_node]["weight"]
                    scores[doc_node] += qw * edge_w  # hop 1

            # Hop 2: expand through doc-doc similarity graph
            hop1_docs = dict(scores)  # snapshot
            for doc_node, h1_score in hop1_docs.items():
                if doc_node not in self.doc_graph:
                    continue
                for nbr in self.doc_graph.neighbors(doc_node):
                    edge_w = self.doc_graph[doc_node][nbr]["weight"]
                    scores[nbr] += self.hop_decay * h1_score * edge_w

        # Filter to candidates if provided
        if candidate_doc_ids is not None:
            allowed = {f"doc:{d}" for d in candidate_doc_ids}
            scores = {k: v for k, v in scores.items() if k in allowed}

        # Sort and format
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:max_results]
        results = [
            {
                "doc_id": doc_node.removeprefix("doc:"),
                "score": score,
                "content": "",
            }
            for doc_node, score in ranked
        ]

        logger.info("Graph scoring: %d docs in %.2f ms", len(results), t.elapsed_ms)
        return results

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        return {
            "bipartite_nodes": self.graph.number_of_nodes(),
            "bipartite_edges": self.graph.number_of_edges(),
            "doc_nodes": len(self.doc_ids),
            "term_nodes": len(self.term_nodes),
            "doc_doc_edges": self.doc_graph.number_of_edges(),
        }
