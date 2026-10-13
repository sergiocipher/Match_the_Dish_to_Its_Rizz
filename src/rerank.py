"""
rerank.py – Post-processing: balanced Sinkhorn assignment and kNN smoothing.

AGENTS.md §8: balanced re-ranking.

Hard rules:
- Validate on the validation gallery (capacity=10) BEFORE applying to test.
- Use SOFT balancing (Sinkhorn blended with original), not hard assignment.
- Transductive methods use only unlabeled gallery image scores – no labels.

Public API
----------
sinkhorn_balance(P, n_iter, strength, eps)  -> np.ndarray
knn_smooth(P, image_features, k, alpha)    -> np.ndarray
rerank(P, image_features, cfg)             -> np.ndarray
rank_for_category(P, class_idx, top)       -> list[int]
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Sinkhorn balanced assignment  (AGENTS.md §8)
# ---------------------------------------------------------------------------

def sinkhorn_balance(
    P: np.ndarray,
    n_iter: int = 20,
    strength: float = 1.0,
    eps: float = 1e-9,
) -> np.ndarray:
    """Soft class-balancing via Sinkhorn normalisation.

    Pushes column sums of ``P`` toward ``N/K`` (equal class distribution)
    while keeping the result close to the original probability matrix.

    Args:
        P:        ``(N_images, K_classes)`` calibrated class probabilities;
                  rows should sum to 1.
        n_iter:   Number of alternating normalisation iterations.
        strength: Blend weight in ``[0, 1]``.  0 = no change, 1 = full
                  Sinkhorn output.  Tune on the validation gallery.
        eps:      Numerical stability constant.

    Returns:
        ``(N_images, K_classes)`` re-balanced probability matrix.
    """
    N, K = P.shape
    Q = P.copy()
    target = N / K

    for _ in range(n_iter):
        # Column normalisation: push class sizes toward target
        Q *= target / (Q.sum(axis=0, keepdims=True) + eps)
        # Row normalisation: keep rows summing to ~1
        Q /= Q.sum(axis=1, keepdims=True) + eps

    return (1.0 - strength) * P + strength * Q


# ---------------------------------------------------------------------------
# kNN smoothing
# ---------------------------------------------------------------------------

def knn_smooth(
    P: np.ndarray,
    image_features: np.ndarray,
    k: int = 5,
    alpha: float = 0.5,
) -> np.ndarray:
    """Smooth each gallery image's class scores with its kNN neighbours.

    Finds the ``k`` nearest neighbours of each image in embedding space and
    blends the original scores with the average of neighbour scores.

    Args:
        P:               ``(N, K)`` score matrix (calibrated probabilities).
        image_features:  ``(N, D)`` L2-normalised image embeddings.
        k:               Number of neighbours (not counting self).
        alpha:           Blend weight for the neighbour average.
                         0 = no smoothing, 1 = use only neighbours.

    Returns:
        ``(N, K)`` smoothed score matrix.
    """
    # Cosine similarity matrix (N, N); images are already L2-normalised
    sim = image_features @ image_features.T  # (N, N)
    np.fill_diagonal(sim, -1.0)              # exclude self

    # Top-k neighbour indices for each image
    top_k_idx = np.argpartition(sim, -k, axis=1)[:, -k:]  # (N, k)

    smoothed = P.copy()
    for i in range(len(P)):
        nbr_mean = P[top_k_idx[i]].mean(axis=0)
        smoothed[i] = (1.0 - alpha) * P[i] + alpha * nbr_mean

    return smoothed


# ---------------------------------------------------------------------------
# Combined re-ranking pipeline
# ---------------------------------------------------------------------------

def rerank(
    P: np.ndarray,
    image_features: np.ndarray | None = None,
    use_sinkhorn: bool = True,
    use_knn: bool = False,
    sinkhorn_n_iter: int = 20,
    sinkhorn_strength: float = 0.5,
    knn_k: int = 5,
    knn_alpha: float = 0.3,
) -> np.ndarray:
    """Apply optional Sinkhorn and kNN smoothing in sequence.

    Args:
        P:                  ``(N, K)`` score matrix.
        image_features:     ``(N, D)`` embeddings (required if ``use_knn``).
        use_sinkhorn:       Apply balanced assignment.
        use_knn:            Apply kNN score smoothing.
        sinkhorn_n_iter:    Sinkhorn iterations.
        sinkhorn_strength:  Sinkhorn blend weight (tune on validation!).
        knn_k:              Number of kNN neighbours.
        knn_alpha:          kNN blend weight.

    Returns:
        ``(N, K)`` re-ranked score matrix.
    """
    Q = P.copy()

    if use_knn:
        if image_features is None:
            raise ValueError("image_features required when use_knn=True")
        Q = knn_smooth(Q, image_features, k=knn_k, alpha=knn_alpha)

    if use_sinkhorn:
        Q = sinkhorn_balance(Q, n_iter=sinkhorn_n_iter, strength=sinkhorn_strength)

    return Q


# ---------------------------------------------------------------------------
# Ranking helper
# ---------------------------------------------------------------------------

def rank_for_category(
    P: np.ndarray,
    class_idx: int,
    top: int = 20,
) -> list[int]:
    """Return row indices of the top-``top`` images for a given category.

    Args:
        P:         ``(N, K)`` score matrix (after optional re-ranking).
        class_idx: Zero-based category index to rank for.
        top:       Number of images to return.

    Returns:
        List of ``top`` integer row indices, best-score first.
    """
    order = np.argsort(-P[:, class_idx], kind="stable")
    return order[:top].tolist()
