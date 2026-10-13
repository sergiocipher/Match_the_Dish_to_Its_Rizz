"""
models.py – Backbone wrappers, linear heads, and adapter layers.

Supports:
  - OpenCLIP (ViT-B/16, ViT-L/14, ViT-H/14, ViT-bigG, etc.)
  - SigLIP / SigLIP2 via open_clip
  - DINOv2 via timm or torch.hub
  - Generic timm backbones (ConvNeXt, EfficientNetV2, ResNet, …)
  - Linear probe / text-initialized linear head
  - CLIP-Adapter / Tip-Adapter-style cache model

Hard rule: MUST NOT load Food-101-specific fine-tuned weights.
           Only generic pretrained checkpoints are allowed.

Public API
----------
load_clip_model(model_name, pretrained, device)  -> (model, preprocess, tokenize)
load_timm_model(model_name, pretrained, num_cls, device) -> model
LinearProbe                                       -> nn.Module
CLIPAdapter                                       -> nn.Module
TipAdapterCacheModel
record_model_provenance(name, hub_id, url, data, license_, out_path)
"""

from __future__ import annotations

import pathlib
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# OpenCLIP / SigLIP loader
# ---------------------------------------------------------------------------

def load_clip_model(
    model_name: str = "ViT-L-14",
    pretrained: str = "openai",
    device: str = "cuda",
):
    """Load an OpenCLIP (or SigLIP) model.

    Args:
        model_name:  E.g. ``"ViT-B-16"``, ``"ViT-L-14"``, ``"ViT-H-14"``,
                     ``"ViT-bigG-14"``, ``"ViT-SO400M-14-SigLIP"``.
        pretrained:  Pretrained weights tag from open_clip (e.g. ``"openai"``,
                     ``"laion2b_s32b_b79k"``, ``"webli"`` for SigLIP).
        device:      CUDA device string or ``"cpu"``.

    Returns:
        ``(model, preprocess_val, tokenize)`` tuple.
    """
    try:
        import open_clip  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError("open_clip_torch is required: pip install open-clip-torch") from exc

    model, _, preprocess_val = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    model = model.to(device).eval()
    tokenize = open_clip.get_tokenizer(model_name)

    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Loaded {model_name}/{pretrained}  ({param_count:.1f}M params)")
    return model, preprocess_val, tokenize


def load_timm_model(
    model_name: str = "convnext_large.fb_in22k_ft_in1k",
    pretrained: bool = True,
    num_classes: int = 0,   # 0 = feature extractor (no head)
    device: str = "cuda",
):
    """Load a timm backbone as a feature extractor.

    Args:
        model_name:  timm model identifier (generic ImageNet weights only).
        pretrained:  Load pretrained ImageNet weights.
        num_classes: Output classes.  0 removes the classifier head.
        device:      Target device.

    Returns:
        ``model`` with ``model.forward_features`` available.
    """
    try:
        import timm  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError("timm is required: pip install timm") from exc

    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    model = model.to(device).eval()
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Loaded timm/{model_name}  ({param_count:.1f}M params)")
    return model


# ---------------------------------------------------------------------------
# Linear Probe (head only)
# ---------------------------------------------------------------------------

try:
    import torch
    import torch.nn as nn

    class LinearProbe(nn.Module):
        """Single linear classifier on top of frozen features.

        Optionally initialised from text embeddings (CLIP zero-shot weights).

        Args:
            in_dim:     Feature dimensionality.
            num_classes: Number of output classes.
            text_weights: Optional ``(num_classes, in_dim)`` tensor of
                          L2-normalised text category embeddings used to
                          initialise the weight matrix.
        """

        def __init__(
            self,
            in_dim: int,
            num_classes: int,
            text_weights: Optional["torch.Tensor"] = None,
        ) -> None:
            super().__init__()
            self.fc = nn.Linear(in_dim, num_classes, bias=False)
            if text_weights is not None:
                assert text_weights.shape == (num_classes, in_dim), (
                    f"text_weights shape mismatch: got {text_weights.shape}, "
                    f"expected ({num_classes}, {in_dim})"
                )
                with torch.no_grad():
                    self.fc.weight.copy_(text_weights)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.fc(x)

    class CLIPAdapter(nn.Module):
        """CLIP-Adapter: lightweight 2-layer MLP residual adapter.

        Adds a small bottleneck MLP whose output is blended back with the
        original CLIP features via a residual ratio ``alpha``.

        Reference: Gao et al., 2021 – "CLIP-Adapter: Better Vision-Language
        Models with Feature Adapters"

        Args:
            in_dim:        Feature dimensionality.
            bottleneck_dim: Hidden dimension of the adapter MLP.
            alpha:         Residual blend ratio (0 = original CLIP only).
        """

        def __init__(
            self,
            in_dim: int,
            bottleneck_dim: int = 64,
            alpha: float = 0.2,
        ) -> None:
            super().__init__()
            self.adapter = nn.Sequential(
                nn.Linear(in_dim, bottleneck_dim, bias=False),
                nn.ReLU(),
                nn.Linear(bottleneck_dim, in_dim, bias=False),
                nn.ReLU(),
            )
            self.alpha = alpha

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            adapted = self.adapter(x)
            return self.alpha * adapted + (1.0 - self.alpha) * x

except ImportError:
    # torch not installed yet; stubs so the module is importable
    class LinearProbe:  # type: ignore[no-redef]
        pass

    class CLIPAdapter:  # type: ignore[no-redef]
        pass


# ---------------------------------------------------------------------------
# Tip-Adapter-style cache model
# ---------------------------------------------------------------------------

class TipAdapterCacheModel:
    """Tip-Adapter: non-parametric cache model from CLIP features.

    Builds a key–value cache from the training set and blends its
    predictions with the zero-shot CLIP scores.

    Reference: Zhang et al., 2021 – "Tip-Adapter: Training-free CLIP-Adapter
    for Better Vision-Language Models"

    Args:
        cache_keys:    ``(N_train, D)`` L2-normalised train image features.
        cache_values:  ``(N_train, K)`` one-hot or soft label matrix.
        beta:          Sharpness of the cache softmax (tune on val).
        alpha:         Blend weight: ``logits = zeroshot + alpha * cache``.
    """

    def __init__(
        self,
        cache_keys: "np.ndarray",
        cache_values: "np.ndarray",
        beta: float = 5.5,
        alpha: float = 1.0,
    ) -> None:
        self.cache_keys = cache_keys     # (N, D)
        self.cache_values = cache_values  # (N, K)
        self.beta = beta
        self.alpha = alpha

    def __call__(
        self,
        query_features: "np.ndarray",
        zeroshot_logits: "np.ndarray",
    ) -> "np.ndarray":
        """Return blended logits.

        Args:
            query_features: ``(M, D)`` L2-normalised query embeddings.
            zeroshot_logits: ``(M, K)`` zero-shot CLIP cosine scores.

        Returns:
            ``(M, K)`` blended logit matrix.
        """
        import numpy as np  # noqa: PLC0415

        # Affinity: (M, N)
        affinity = query_features @ self.cache_keys.T  # cosine (already L2-normed)
        cache_logits = ((-1) * (self.beta - self.beta * affinity)).exp() @ self.cache_values
        return zeroshot_logits + self.alpha * cache_logits


# ---------------------------------------------------------------------------
# WiSE-FT blending
# ---------------------------------------------------------------------------

def wise_ft_blend(
    zeroshot_logits: "np.ndarray",
    finetuned_logits: "np.ndarray",
    alpha: float = 0.5,
) -> "np.ndarray":
    """WiSE-FT: linearly blend zero-shot and fine-tuned logits.

    ``alpha=0`` is pure zero-shot; ``alpha=1`` is pure fine-tuned.
    Tune ``alpha`` on validation.

    Reference: Wortsman et al., 2022 – "Model Soups / WiSE-FT"
    """
    return (1.0 - alpha) * zeroshot_logits + alpha * finetuned_logits


# ---------------------------------------------------------------------------
# Model provenance logging
# ---------------------------------------------------------------------------

def record_model_provenance(
    name: str,
    hub_id: str,
    url: str,
    training_data: str,
    license_: str,
    out_path: str | pathlib.Path = "reports/models_used.md",
) -> None:
    """Append a provenance row for a pretrained checkpoint.

    AGENTS.md Hard Rule 5: every pretrained checkpoint MUST be recorded.
    """
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    row = (
        f"| {name} | {hub_id} | {url} | {training_data} | {license_} |\n"
    )

    if not out_path.exists():
        header = (
            "# Models Used\n\n"
            "| Name | Hub ID | Source URL | Training Data | License |\n"
            "|------|--------|------------|---------------|---------|\n"
        )
        out_path.write_text(header + row)
    else:
        with open(out_path, "a") as fh:
            fh.write(row)

    print(f"  Provenance recorded: {name}")


# ---------------------------------------------------------------------------
# Layer-wise LR decay (AGENTS.md §14.3)
# ---------------------------------------------------------------------------

def param_groups_lrd(
    model,
    base_lr: float = 1e-5,
    head_lr: float = 1e-3,
    wd: float = 0.05,
    decay: float = 0.75,
    n_layers: int = 12,
) -> list[dict]:
    """Build AdamW parameter groups with layer-wise LR decay for ViT models.

    Args:
        model:    The ViT (or similar) model.
        base_lr:  Learning rate for the deepest backbone layer.
        head_lr:  Learning rate for the classification head.
        wd:       Weight decay (not applied to bias / norm params).
        decay:    Multiplicative decay factor per layer (0.65–0.85 typical).
        n_layers: Number of transformer blocks.

    Returns:
        List of parameter group dicts suitable for ``torch.optim.AdamW``.
    """

    def get_layer_id(name: str) -> int:
        """Heuristic layer assignment for common ViT parameter names."""
        if "patch_embed" in name or "cls_token" in name or "pos_embed" in name:
            return 0
        if "blocks." in name:
            try:
                block_idx = int(name.split("blocks.")[1].split(".")[0])
                return block_idx + 1
            except (IndexError, ValueError):
                pass
        return n_layers  # norm, head, etc.

    groups: dict[tuple, dict] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("head") or name.startswith("fc"):
            lr = head_lr
            layer_id = n_layers + 1
        else:
            layer_id = get_layer_id(name)
            lr = base_lr * (decay ** (n_layers + 1 - layer_id))
        no_wd = p.ndim == 1 or name.endswith(".bias")
        key = (layer_id, no_wd)
        groups.setdefault(
            key,
            {"params": [], "lr": lr, "weight_decay": 0.0 if no_wd else wd},
        )
        groups[key]["params"].append(p)

    return list(groups.values())
