"""
BEIR Evaluation
===============
Scores retrieval runs with ``pytrec_eval``, the Python binding to the official
``trec_eval`` C implementation.

Published BEIR results are produced by ``trec_eval``, so reimplementing the
metrics would risk small, silent divergences — particularly around tie
handling and graded-relevance nDCG — that make numbers non-comparable.
The headline BEIR metric is **nDCG@10**.
"""

from __future__ import annotations

from typing import Any

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

# The metric set BEIR papers report.
DEFAULT_K_VALUES = [1, 3, 5, 10, 100, 1000]


def _require_pytrec_eval():
    try:
        import pytrec_eval
    except ImportError as e:  # pragma: no cover - environment dependent
        raise ImportError(
            "pytrec_eval is required for BEIR scoring. Install it with:\n"
            "    pip install pytrec-eval-terrier"
        ) from e
    return pytrec_eval


def evaluate_run(
    qrels: dict[str, dict[str, int]],
    run: dict[str, dict[str, float]],
    k_values: list[int] | None = None,
) -> dict[str, float]:
    """Score a retrieval run against relevance judgments.

    Args:
        qrels: ``{query_id: {doc_id: relevance_grade}}``.
        run: ``{query_id: {doc_id: score}}`` — the system's retrieved results.
        k_values: Cutoffs for nDCG / Recall / MAP / Precision.

    Returns:
        ``{"ndcg@10": 0.71, "recall@100": 0.94, ...}`` averaged over queries.
    """
    pytrec_eval = _require_pytrec_eval()
    k_values = k_values or DEFAULT_K_VALUES

    # Score only judged queries; an unjudged query contributes an undefined
    # (not zero) score and would bias the average downward.
    scored_run = {qid: docs for qid, docs in run.items() if qid in qrels}
    if len(scored_run) < len(run):
        logger.warning(
            "Ignoring %d retrieved queries with no relevance judgments",
            len(run) - len(scored_run),
        )
    missing = set(qrels) - set(scored_run)
    if missing:
        logger.warning(
            "%d judged queries returned no results — scored as 0.0", len(missing)
        )
        for qid in missing:
            scored_run[qid] = {}

    ks = ",".join(str(k) for k in k_values)
    measures = {f"ndcg_cut.{ks}", f"recall.{ks}", f"map_cut.{ks}", f"P.{ks}", "recip_rank"}

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, measures)
    per_query = evaluator.evaluate(scored_run)

    if not per_query:
        return {}

    n = len(per_query)
    results: dict[str, float] = {}
    for k in k_values:
        results[f"ndcg@{k}"] = sum(q[f"ndcg_cut_{k}"] for q in per_query.values()) / n
        results[f"recall@{k}"] = sum(q[f"recall_{k}"] for q in per_query.values()) / n
        results[f"map@{k}"] = sum(q[f"map_cut_{k}"] for q in per_query.values()) / n
        results[f"precision@{k}"] = sum(q[f"P_{k}"] for q in per_query.values()) / n
    results["mrr"] = sum(q["recip_rank"] for q in per_query.values()) / n

    return {k: round(v, 5) for k, v in results.items()}


def results_to_run(
    results: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, float]]:
    """Convert ``{query_id: [{doc_id, score}, ...]}`` into trec_eval run format."""
    return {
        qid: {r["doc_id"]: float(r["score"]) for r in hits}
        for qid, hits in results.items()
    }


def format_results_table(
    results_by_system: dict[str, dict[str, float]],
    metrics: list[str] | None = None,
) -> str:
    """Render a comparison table across systems, best-first on the first metric."""
    metrics = metrics or ["ndcg@10", "recall@100", "mrr"]
    if not results_by_system:
        return "(no results)"

    name_w = max(len(s) for s in results_by_system) + 2
    header = "System".ljust(name_w) + "".join(m.rjust(14) for m in metrics)
    lines = [header, "-" * len(header)]

    ranked = sorted(
        results_by_system.items(),
        key=lambda kv: kv[1].get(metrics[0], 0.0),
        reverse=True,
    )
    for system, scores in ranked:
        row = system.ljust(name_w)
        row += "".join(f"{scores.get(m, float('nan')):.4f}".rjust(14) for m in metrics)
        lines.append(row)

    return "\n".join(lines)
