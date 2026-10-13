"""
train.py – Fine-tuning loop with AMP, EMA, cosine LR schedule, early stopping.

AGENTS.md Phase 5 recipe:
  Stage A: head only (backbone frozen)
  Stage B: partial unfreeze with layer-wise LR decay
  Stage C (optional): full fine-tune at very low LR

Public API
----------
EMA                           – exponential moving average of weights
cosine_lr_at(step, ...)       – LR value at a given step
train_one_epoch(...)          – single epoch training loop
evaluate_epoch(...)           – validation pass
fine_tune(model, ...)         – full training driver
"""

from __future__ import annotations

import copy
import math
import pathlib
import time
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Cosine LR schedule with warmup  (AGENTS.md §14.4)
# ---------------------------------------------------------------------------

def cosine_lr_at(
    step: int,
    total_steps: int,
    warmup_steps: int,
    peak_lr: float,
    floor_ratio: float = 0.01,
) -> float:
    """Return the learning rate at ``step`` for a cosine schedule with warmup.

    Args:
        step:         Current global step (0-indexed).
        total_steps:  Total number of training steps.
        warmup_steps: Number of linear warmup steps.
        peak_lr:      Peak learning rate (after warmup).
        floor_ratio:  Final LR as a fraction of ``peak_lr``.

    Returns:
        Learning rate value.
    """
    if step < warmup_steps:
        return peak_lr * step / max(1, warmup_steps)
    t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return peak_lr * (floor_ratio + (1.0 - floor_ratio) * 0.5 * (1.0 + math.cos(math.pi * t)))


def set_lr(optimizer, lr: float) -> None:
    """Set the same LR on all param groups (use for single-LR schedules)."""
    for pg in optimizer.param_groups:
        pg["lr"] = lr


def set_lr_by_group(optimizer, base_lr: float, group_scales: list[float]) -> None:
    """Scale each param group's LR relative to ``base_lr``.

    ``group_scales[i]`` is the multiplier for ``optimizer.param_groups[i]``.
    """
    assert len(group_scales) == len(optimizer.param_groups)
    for pg, scale in zip(optimizer.param_groups, group_scales):
        pg["lr"] = base_lr * scale


# ---------------------------------------------------------------------------
# EMA (exponential moving average of model weights)
# ---------------------------------------------------------------------------

class EMA:
    """Maintains an exponential moving average of model parameters.

    Usage::

        ema = EMA(model, decay=0.9999)
        for batch in loader:
            loss.backward(); optimizer.step()
            ema.update(model)

        # Swap to EMA weights for evaluation:
        with ema.average_parameters(model):
            val_metric = evaluate(model, val_loader)

    Args:
        model:  The model whose parameters to track.
        decay:  EMA decay factor (0.999–0.9999 typical).
    """

    def __init__(self, model, decay: float = 0.9999) -> None:
        self.decay = decay
        self.shadow: dict[str, "torch.Tensor"] = {}
        self._backup: dict[str, "torch.Tensor"] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model) -> None:
        """Update shadow weights after an optimiser step."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (
                    self.decay * self.shadow[name]
                    + (1.0 - self.decay) * param.data
                )

    def apply(self, model) -> None:
        """Copy EMA weights into the model (destructive – call restore after)."""
        self._backup = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                self._backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model) -> None:
        """Restore original weights after an EMA evaluation pass."""
        for name, param in model.named_parameters():
            if name in self._backup:
                param.data.copy_(self._backup[name])
        self._backup = {}

    class average_parameters:
        """Context manager: temporarily swap in EMA weights."""

        def __init__(self, ema: "EMA", model) -> None:
            self.ema = ema
            self.model = model

        def __enter__(self):
            self.ema.apply(self.model)
            return self.model

        def __exit__(self, *_):
            self.ema.restore(self.model)


# ---------------------------------------------------------------------------
# Single epoch helpers
# ---------------------------------------------------------------------------

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device: str,
    scaler=None,
    ema: Optional[EMA] = None,
    grad_clip: float = 1.0,
    step_offset: int = 0,
    lr_fn: Optional[Callable[[int], float]] = None,
    log_interval: int = 50,
) -> dict[str, float]:
    """Run one training epoch.

    Args:
        model:       The model (should already be on ``device``).
        loader:      DataLoader yielding ``(images, labels)``.
        optimizer:   PyTorch optimizer.
        criterion:   Loss function ``(logits, labels) -> scalar``.
        device:      CUDA device string.
        scaler:      ``torch.cuda.amp.GradScaler`` for AMP; None disables AMP.
        ema:         Optional :class:`EMA` instance updated after each step.
        grad_clip:   Max gradient norm (1.0 recommended for transformers).
        step_offset: Global step at the start of this epoch (for LR schedule).
        lr_fn:       Optional callable ``(step) -> float``.  If provided,
                     the learning rate is updated before each batch.
        log_interval: Print loss every N batches.

    Returns:
        Dict with ``loss`` (mean) and ``elapsed_s``.
    """
    import torch  # noqa: PLC0415

    model.train()
    total_loss = 0.0
    n_batches = 0
    t0 = time.time()

    for batch_idx, (images, labels) in enumerate(loader):
        global_step = step_offset + batch_idx
        if lr_fn is not None:
            lr = lr_fn(global_step)
            set_lr(optimizer, lr)

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.autocast(device_type=device.split(":")[0], dtype=torch.float16):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        if ema is not None:
            ema.update(model)

        total_loss += loss.item()
        n_batches += 1

        if (batch_idx + 1) % log_interval == 0:
            print(
                f"  [{batch_idx + 1}/{len(loader)}]"
                f"  loss={total_loss / n_batches:.4f}"
                f"  lr={optimizer.param_groups[0]['lr']:.2e}"
            )

    return {"loss": total_loss / max(n_batches, 1), "elapsed_s": time.time() - t0}


@staticmethod
def evaluate_epoch(
    model,
    loader,
    criterion,
    device: str,
    scaler=None,
) -> dict[str, float]:
    """Validation pass: compute average loss and top-1 accuracy.

    Args:
        model:    Model in eval mode (caller must set).
        loader:   DataLoader yielding ``(images, labels)``.
        criterion: Loss function.
        device:   Device string.
        scaler:   Unused here (kept for signature consistency).

    Returns:
        Dict with ``loss`` and ``acc1``.
    """
    import torch  # noqa: PLC0415

    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device_type=device.split(":")[0], dtype=torch.float16):
                logits = model(images)
                loss = criterion(logits, labels)
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return {
        "loss": total_loss / max(len(loader), 1),
        "acc1": correct / max(total, 1),
    }


# ---------------------------------------------------------------------------
# Full training driver
# ---------------------------------------------------------------------------

def fine_tune(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    n_epochs: int,
    device: str,
    ema_decay: float = 0.9999,
    grad_clip: float = 1.0,
    warmup_ratio: float = 0.05,
    peak_lr: Optional[float] = None,
    early_stopping_patience: int = 5,
    checkpoint_dir: str | pathlib.Path = "artifacts/checkpoints",
    run_name: str = "run",
    eval_metric_fn: Optional[Callable] = None,
) -> dict:
    """Full fine-tuning driver.

    Implements AGENTS.md Phase 5 recipe:
    - AMP with GradScaler
    - EMA weight tracking
    - Cosine LR schedule with warmup
    - Early stopping on validation metric
    - Checkpoint saving (best EMA weights)

    Args:
        model:           Model to fine-tune (already on ``device``).
        train_loader:    Training DataLoader.
        val_loader:      Validation DataLoader (images, labels).
        optimizer:       Pre-configured AdamW (with param groups if using LLRD).
        criterion:       Loss function (cross-entropy with label smoothing).
        n_epochs:        Maximum epochs.
        device:          CUDA device string.
        ema_decay:       EMA decay (0.999–0.9999).
        grad_clip:       Gradient norm clip.
        warmup_ratio:    Fraction of total steps used for linear warmup.
        peak_lr:         If provided, overrides optimizer LR and uses cosine schedule.
        early_stopping_patience: Stop if no improvement for this many epochs.
        checkpoint_dir:  Where to save ``.pt`` checkpoints.
        run_name:        Used in checkpoint filenames.
        eval_metric_fn:  Optional ``(model, val_loader) -> float`` called each
                         epoch.  Higher is better.  Falls back to ``-val_loss``.

    Returns:
        Dict with ``best_epoch``, ``best_metric``, ``history`` list.
    """
    import torch  # noqa: PLC0415

    checkpoint_dir = pathlib.Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    scaler = torch.cuda.amp.GradScaler() if "cuda" in device else None
    ema = EMA(model, decay=ema_decay)

    total_steps = n_epochs * len(train_loader)
    warmup_steps = int(warmup_ratio * total_steps)

    # Build LR function (only used if peak_lr is provided)
    lr_fn: Optional[Callable[[int], float]] = None
    if peak_lr is not None:
        lr_fn = lambda step: cosine_lr_at(  # noqa: E731
            step, total_steps, warmup_steps, peak_lr
        )

    best_metric = -float("inf")
    best_epoch = 0
    no_improve = 0
    history = []

    for epoch in range(1, n_epochs + 1):
        step_offset = (epoch - 1) * len(train_loader)
        train_stats = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            scaler=scaler, ema=ema, grad_clip=grad_clip,
            step_offset=step_offset, lr_fn=lr_fn,
        )

        # Evaluate with EMA weights
        with ema.average_parameters(ema, model):
            if eval_metric_fn is not None:
                metric = eval_metric_fn(model, val_loader)
            else:
                val_stats = evaluate_epoch(model, val_loader, criterion, device)
                metric = -val_stats["loss"]
                train_stats.update(val_stats)

        print(
            f"Epoch {epoch}/{n_epochs}  "
            f"train_loss={train_stats['loss']:.4f}  "
            f"metric={metric:.4f}  "
            f"elapsed={train_stats['elapsed_s']:.0f}s"
        )
        history.append({"epoch": epoch, "metric": metric, **train_stats})

        if metric > best_metric:
            best_metric = metric
            best_epoch = epoch
            no_improve = 0
            # Save best EMA checkpoint
            ema.apply(model)
            ckpt_path = checkpoint_dir / f"{run_name}_best.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metric": metric,
                },
                ckpt_path,
            )
            ema.restore(model)
            print(f"  -> Saved best checkpoint: {ckpt_path}")
        else:
            no_improve += 1
            if no_improve >= early_stopping_patience:
                print(f"  Early stopping at epoch {epoch} (no improvement for {no_improve} epochs)")
                break

    # Also save final checkpoint (for ensembling / inspection)
    ema.apply(model)
    final_path = checkpoint_dir / f"{run_name}_final_epoch{epoch}.pt"
    torch.save({"epoch": epoch, "model_state_dict": model.state_dict()}, final_path)
    ema.restore(model)

    print(f"\nTraining done. Best epoch={best_epoch}  best_metric={best_metric:.4f}")
    return {"best_epoch": best_epoch, "best_metric": best_metric, "history": history}


# ---------------------------------------------------------------------------
# Loss function helper
# ---------------------------------------------------------------------------

def get_criterion(num_classes: int, label_smoothing: float = 0.1):
    """Return a cross-entropy loss with label smoothing.

    Args:
        num_classes:     Number of output classes.
        label_smoothing: Smoothing factor (0.05–0.1 for noisy labels).

    Returns:
        ``torch.nn.CrossEntropyLoss`` instance.
    """
    import torch.nn as nn  # noqa: PLC0415

    return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
