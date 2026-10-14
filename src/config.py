"""
config.py – Load YAML configs, resolve paths (local vs Kaggle), set seeds.

Usage:
    from src.config import load_config, DATA_DIR, OUT_DIR, seed_everything
"""

from __future__ import annotations

import math
import os
import pathlib
import random
from typing import Any

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------------
ON_KAGGLE: bool = os.path.exists("/kaggle/input")

# Resolve paths at import time.  The competition slug must match the folder
# name under /kaggle/input/ once uploaded.  Override via env var or config.
# On Kaggle the competition data lands at /kaggle/input/competitions/<slug>/
# which differs from the usual /kaggle/input/<slug>/ layout.
# We probe both locations so the code works regardless of how it was added.
def _resolve_data_dir() -> pathlib.Path:
    candidates = [
        pathlib.Path("/kaggle/input/competitions/match-the-dish-to-its-rizz"),
        pathlib.Path("/kaggle/input/match-the-dish-to-its-rizz"),
        pathlib.Path("/kaggle/input/food-text-to-image-retrieval"),
    ]
    for c in candidates:
        if c.exists():
            return c
    # fallback to env-var slug
    slug = os.environ.get("COMPETITION_SLUG", "match-the-dish-to-its-rizz")
    return pathlib.Path("/kaggle/input") / slug

DATA_DIR: pathlib.Path = (
    _resolve_data_dir()
    if ON_KAGGLE
    else pathlib.Path(__file__).parent.parent / "data"
)

OUT_DIR: pathlib.Path = (
    pathlib.Path("/kaggle/working")
    if ON_KAGGLE
    else pathlib.Path(__file__).parent.parent / "artifacts"
)

# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def seed_everything(seed: int = 42) -> None:
    """Fix all random seeds for reproducibility. Log the chosen seed."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch  # noqa: PLC0415
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # deterministic=False is faster; flip to True if exact reproducibility
        # across runs is more important than speed.
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    except ImportError:
        pass

    print(f"[config] Seeds fixed to {seed}")


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: str | pathlib.Path) -> dict[str, Any]:
    """Load a YAML config file and return a plain dict.

    Automatically resolves the ``paths`` section to :class:`pathlib.Path`
    objects and injects ``DATA_DIR`` / ``OUT_DIR`` so notebooks never
    hard-code absolute paths.
    """
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path) as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh) or {}

    # Inject resolved path roots into the config so downstream code can use
    # cfg["paths"]["data_dir"] without knowing whether it is running locally
    # or on Kaggle.
    cfg.setdefault("paths", {})
    cfg["paths"]["data_dir"] = str(DATA_DIR)
    cfg["paths"]["out_dir"] = str(OUT_DIR)
    cfg["paths"]["features_dir"] = str(OUT_DIR / "features")
    cfg["paths"]["checkpoints_dir"] = str(OUT_DIR / "checkpoints")
    cfg["paths"]["submissions_dir"] = str(OUT_DIR / "submissions")
    cfg["paths"]["oof_dir"] = str(OUT_DIR / "oof_preds")

    return cfg


def get_paths(cfg: dict[str, Any]) -> dict[str, pathlib.Path]:
    """Return resolved :class:`pathlib.Path` objects from a loaded config."""
    return {k: pathlib.Path(v) for k, v in cfg["paths"].items()}


# ---------------------------------------------------------------------------
# Tiny helpers
# ---------------------------------------------------------------------------

def ensure_dirs(cfg: dict[str, Any]) -> None:
    """Create all output directories listed in cfg['paths'] if absent."""
    for p in get_paths(cfg).values():
        p.mkdir(parents=True, exist_ok=True)
