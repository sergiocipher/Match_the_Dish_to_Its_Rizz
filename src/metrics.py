"""
metrics.py – Evaluation metrics for the food retrieval competition.

All functions work on plain Python lists / iterables of IDs.
Unit tests live in tests/test_metrics.py and MUST pass before reporting any number.

Public API
----------
r_precision(ranked_ids, relevant_ids)              -> float
precision_at_k(ranked_ids, relevant_ids, k)        -> float
hit_at_k(ranked_ids, relevant_ids, k)              -> float
mrr_at_k(ranked_ids, relevant_ids, k)              -> float
average_precision_at_k(ranked_ids, relevant_ids, k)-> float
evaluate(predictions, ground_truth)                -> dict[str, float]
"""

from __future__ import annotations

from typing import Iterable


# ---------------------------------------------------------------------------
# Per-query metrics  (exact implementations from AGENTS.md §14.1)
# ---------------------------------------------------------------------------

def r_precision(ranked_ids: list, relevant_ids: Iterable) -> float:
    """R-precision: fraction of correct items among the first R predictions.

    R = |relevant_ids|.  Order within the first R does not matter.

    Args:
        ranked_ids:   Predicted image IDs, best-first.
        relevant_ids: Ground-truth image IDs for this query.

    Returns:
        Value in [0, 1].  Returns 0.0 if ``relevant_ids`` is empty.
    """
    rel = set(relevant_ids)
    R = len(rel)
    if R == 0:
        return 0.0
    return sum(1 for i in ranked_ids[:R] if i in rel) / R


def precision_at_k(ranked_ids: list, relevant_ids: Iterable, k: int = 5) -> float:
    """Precision at k: fraction of correct items in the top-k predictions.

    Args:
        ranked_ids:   Predicted image IDs, best-first.
        relevant_ids: Ground-truth image IDs for this query.
        k:            Cut-off position.

    Returns:
        Value in [0, 1].
    """
    rel = set(relevant_ids)
    if k <= 0:
        return 0.0
    return sum(1 for i in ranked_ids[:k] if i in rel) / k


def hit_at_k(ranked_ids: list, relevant_ids: Iterable, k: int = 1) -> float:
    """Hit@k: 1 if at least one relevant item appears in the top-k, else 0.

    Args:
        ranked_ids:   Predicted image IDs, best-first.
        relevant_ids: Ground-truth image IDs for this query.
        k:            Cut-off position.

    Returns:
        1.0 or 0.0.
    """
    rel = set(relevant_ids)
    return float(any(i in rel for i in ranked_ids[:k]))


def mrr_at_k(ranked_ids: list, relevant_ids: Iterable, k: int = 20) -> float:
    """Mean Reciprocal Rank at k: reciprocal rank of the first correct hit.

    Args:
        ranked_ids:   Predicted image IDs, best-first.
        relevant_ids: Ground-truth image IDs for this query.
        k:            Cut-off position.

    Returns:
        Value in [0, 1].  0.0 if no relevant item appears in the top-k.
    """
    rel = set(relevant_ids)
    for rank, img_id in enumerate(ranked_ids[:k], start=1):
        if img_id in rel:
            return 1.0 / rank
    return 0.0


def average_precision_at_k(
    ranked_ids: list, relevant_ids: Iterable, k: int = 20
) -> float:
    """Average Precision at k.

    Normalised by ``min(k, R)`` where R = |relevant_ids|, matching the
    formulation in AGENTS.md §14.1.  Always verify against the starter
    notebook's implementation before reporting numbers.

    Args:
        ranked_ids:   Predicted image IDs, best-first.
        relevant_ids: Ground-truth image IDs for this query.
        k:            Cut-off position.

    Returns:
        Value in [0, 1].  0.0 if ``relevant_ids`` is empty.
    """
    rel = set(relevant_ids)
    if not rel:
        return 0.0
    hits = 0
    score = 0.0
    for rank, img_id in enumerate(ranked_ids[:k], start=1):
        if img_id in rel:
            hits += 1
            score += hits / rank
    return score / min(k, len(rel))


# ---------------------------------------------------------------------------
# Dataset-level evaluation
# ---------------------------------------------------------------------------

def evaluate(
    predictions: dict[str, list],
    ground_truth: dict[str, list],
    k_map: int = 20,
    k_p: int = 5,
    k_hit1: int = 1,
    k_hit5: int = 5,
    k_mrr: int = 20,
) -> dict[str, float]:
    """Compute all competition metrics averaged over queries.

    Args:
        predictions:  ``{query_id: [image_id, ...]}`` – best-first ranked lists.
        ground_truth: ``{query_id: [image_id, ...]}`` – relevant image IDs.
        k_map:        Cut-off for mAP (default 20).
        k_p:          Cut-off for Precision@k (default 5).
        k_hit1:       Cut-off for Hit@k (default 1).
        k_hit5:       Cut-off for Hit@k (default 5).
        k_mrr:        Cut-off for MRR@k (default 20).

    Returns:
        Dict with keys: ``r_precision``, ``map@{k_map}``, ``p@{k_p}``,
        ``hit@{k_hit1}``, ``hit@{k_hit5}``, ``mrr@{k_mrr}``.
    """
    query_ids = list(ground_truth.keys())
    n = len(query_ids)
    if n == 0:
        raise ValueError("ground_truth is empty")

    sums: dict[str, float] = {
        "r_precision": 0.0,
        f"map@{k_map}": 0.0,
        f"p@{k_p}": 0.0,
        f"hit@{k_hit1}": 0.0,
        f"hit@{k_hit5}": 0.0,
        f"mrr@{k_mrr}": 0.0,
    }

    for qid in query_ids:
        ranked = predictions.get(qid, [])
        rel = ground_truth[qid]
        sums["r_precision"] += r_precision(ranked, rel)
        sums[f"map@{k_map}"] += average_precision_at_k(ranked, rel, k=k_map)
        sums[f"p@{k_p}"] += precision_at_k(ranked, rel, k=k_p)
        sums[f"hit@{k_hit1}"] += hit_at_k(ranked, rel, k=k_hit1)
        sums[f"hit@{k_hit5}"] += hit_at_k(ranked, rel, k=k_hit5)
        sums[f"mrr@{k_mrr}"] += mrr_at_k(ranked, rel, k=k_mrr)

    return {key: val / n for key, val in sums.items()}


def print_metrics(metrics: dict[str, float]) -> None:
    """Pretty-print a metrics dict."""
    for key, val in metrics.items():
        print(f"  {key:<20s}  {val:.4f}")
