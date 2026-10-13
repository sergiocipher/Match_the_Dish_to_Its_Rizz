"""
ensemble.py – Score calibration, temperature scaling, and model ensembling.

AGENTS.md §6 (Phase 6):
- Calibrate each model's logits (temperature scaling on train / cross-val data).
- Ensemble in log-prob space after calibration.
- Weights chosen by coarse validation grid search.

Public API
----------
temperature_scale(logits, T)              -> np.ndarray (probabilities)
fit_temperature(logits, labels)           -> float
LogitEnsemble                             -> combine multiple score matrices
blend_zeroshot_finetuned(zs, ft, alpha)   -> np.ndarray
"""

from __future__ import annotations

from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Temperature scaling
# ---------------------------------------------------------------------------

def temperature_scale(
    logits: np.ndarray,
    T: float = 1.0,
    to_prob: bool = True,
) -> np.ndarray:
    """Apply temperature scaling to a logit matrix.

    Args:
        logits:  ``(N, K)`` raw logit or cosine-similarity scores.
        T:       Temperature (> 1 softens, < 1 sharpens).
        to_prob: If True, return softmax probabilities; else return scaled logits.

    Returns:
        ``(N, K)`` array.
    """
    scaled = logits / max(T, 1e-9)
    if not to_prob:
        return scaled
    # numerically stable softmax
    shifted = scaled - scaled.max(axis=1, keepdims=True)
    exp_s = np.exp(shifted)
    return exp_s / exp_s.sum(axis=1, keepdims=True)


def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    T_range: tuple[float, float] = (0.01, 10.0),
    n_steps: int = 200,
) -> float:
    """Find the temperature that minimises NLL on a labelled held-out set.

    Args:
        logits:  ``(N, K)`` logit matrix from the model.
        labels:  ``(N,)`` integer class labels.
        T_range: Search range ``(T_min, T_max)``.
        n_steps: Number of candidate temperatures to try.

    Returns:
        Optimal temperature scalar.
    """
    best_T = 1.0
    best_nll = float("inf")

    for T in np.linspace(T_range[0], T_range[1], n_steps):
        probs = temperature_scale(logits, T=T, to_prob=True)
        # clip for numerical stability
        probs = np.clip(probs, 1e-9, 1.0)
        nll = -np.log(probs[np.arange(len(labels)), labels]).mean()
        if nll < best_nll:
            best_nll = nll
            best_T = float(T)

    print(f"  Best temperature: {best_T:.3f}  (NLL={best_nll:.4f})")
    return best_T


# ---------------------------------------------------------------------------
# Log-probability ensemble
# ---------------------------------------------------------------------------

class LogitEnsemble:
    """Combine multiple score matrices via weighted average in log-prob space.

    Usage::

        ens = LogitEnsemble()
        ens.add(clip_logits, weight=1.0, temperature=clip_T)
        ens.add(siglip_logits, weight=1.0, temperature=siglip_T)
        ens.add(dinov2_logits, weight=0.8, temperature=dino_T)
        combined_probs = ens.combine()

    Args:
        eps: Numerical stability floor before taking log.
    """

    def __init__(self, eps: float = 1e-9) -> None:
        self._entries: list[tuple[np.ndarray, float]] = []  # (log_probs, weight)
        self.eps = eps

    def add(
        self,
        logits: np.ndarray,
        weight: float = 1.0,
        temperature: float = 1.0,
    ) -> "LogitEnsemble":
        """Add a model's logits to the ensemble.

        Args:
            logits:      ``(N, K)`` logit or cosine-score matrix.
            weight:      Relative weight for this model.
            temperature: Temperature to apply before softmax.

        Returns:
            ``self`` (fluent interface).
        """
        probs = temperature_scale(logits, T=temperature, to_prob=True)
        log_probs = np.log(np.clip(probs, self.eps, 1.0))
        self._entries.append((log_probs, weight))
        return self

    def combine(self) -> np.ndarray:
        """Return ``(N, K)`` combined probability matrix.

        Computes a weighted average of log-probabilities, then exponentiates
        and normalises to obtain valid probabilities.
        """
        if not self._entries:
            raise ValueError("No models added to ensemble.")

        total_weight = sum(w for _, w in self._entries)
        combined_log = sum(lp * w for lp, w in self._entries) / total_weight

        # Convert back to probabilities and renormalise
        probs = np.exp(combined_log)
        probs /= probs.sum(axis=1, keepdims=True) + 1e-9
        return probs

    def search_weights(
        self,
        logits_list: list[np.ndarray],
        temperatures: list[float],
        labels: np.ndarray,
        weight_grid: Optional[list[list[float]]] = None,
    ) -> list[float]:
        """Coarse grid search for ensemble weights on a validation set.

        Args:
            logits_list:   List of ``(N, K)`` logit matrices.
            temperatures:  Per-model temperatures.
            labels:        ``(N,)`` true class labels.
            weight_grid:   Explicit list of weight vectors to try.  If None,
                           tries uniform and a few hand-crafted combinations.

        Returns:
            Best weight vector (same length as ``logits_list``).
        """
        n_models = len(logits_list)
        if weight_grid is None:
            # Simple coarse grid: sweep individual weights from 0.5 to 2.0
            import itertools  # noqa: PLC0415

            steps = [0.5, 1.0, 1.5, 2.0]
            weight_grid = list(itertools.product(steps, repeat=n_models))

        best_weights = [1.0] * n_models
        best_acc = -1.0

        for weights in weight_grid:
            ens = LogitEnsemble()
            for logits, T, w in zip(logits_list, temperatures, weights):
                ens.add(logits, weight=w, temperature=T)
            probs = ens.combine()
            preds = probs.argmax(axis=1)
            acc = (preds == labels).mean()
            if acc > best_acc:
                best_acc = acc
                best_weights = list(weights)

        print(f"  Best ensemble weights: {best_weights}  (val acc={best_acc:.4f})")
        return best_weights


# ---------------------------------------------------------------------------
# WiSE-FT style blend (zero-shot + fine-tuned logits)
# ---------------------------------------------------------------------------

def blend_zeroshot_finetuned(
    zeroshot_logits: np.ndarray,
    finetuned_logits: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Linearly blend zero-shot and fine-tuned logit matrices.

    ``alpha=0`` → pure zero-shot; ``alpha=1`` → pure fine-tuned.
    Tune ``alpha`` on the validation set.

    Args:
        zeroshot_logits:  ``(N, K)`` zero-shot cosine scores.
        finetuned_logits: ``(N, K)`` fine-tuned model logits.
        alpha:            Blend weight for fine-tuned model.

    Returns:
        ``(N, K)`` blended logits.
    """
    return (1.0 - alpha) * zeroshot_logits + alpha * finetuned_logits


def search_blend_alpha(
    zeroshot_logits: np.ndarray,
    finetuned_logits: np.ndarray,
    labels: np.ndarray,
    alphas: Optional[np.ndarray] = None,
) -> float:
    """Grid search for the best ``alpha`` in :func:`blend_zeroshot_finetuned`.

    Args:
        zeroshot_logits:  ``(N, K)`` zero-shot logits.
        finetuned_logits: ``(N, K)`` fine-tuned logits.
        labels:           ``(N,)`` true class labels.
        alphas:           Candidate alpha values.  Default: 0 to 1 in 0.05 steps.

    Returns:
        Best alpha value.
    """
    if alphas is None:
        alphas = np.linspace(0.0, 1.0, 21)

    best_alpha = 0.5
    best_acc = -1.0

    for alpha in alphas:
        blended = blend_zeroshot_finetuned(zeroshot_logits, finetuned_logits, alpha=float(alpha))
        acc = (blended.argmax(axis=1) == labels).mean()
        if acc > best_acc:
            best_acc = acc
            best_alpha = float(alpha)

    print(f"  Best blend alpha: {best_alpha:.2f}  (val acc={best_acc:.4f})")
    return best_alpha
