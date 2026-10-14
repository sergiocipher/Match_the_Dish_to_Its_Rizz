"""
data.py – CSV loading, PyTorch Datasets, and image transforms.

Public API
----------
load_csvs(data_dir)           -> dict of DataFrames
GalleryDataset                -> (image_tensor, image_id) pairs
QueryDataset                  -> (description, query_id, category_id) rows
get_transforms(split, size)   -> torchvision transform pipeline
"""

from __future__ import annotations

import json
import pathlib
from typing import Callable, Optional

import numpy as np
import pandas as pd
from PIL import Image

# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

_CSV_FILES = {
    "train": "train.csv",
    "train_queries": "train_queries.csv",
    "category_metadata": "category_metadata.csv",
    "val_gallery": "validation_gallery.csv",
    "val_queries": "validation_queries.csv",
    "val_gt": "validation_ground_truth.csv",
    "test_gallery": "test_gallery.csv",
    "test_queries": "test_queries.csv",
    "sample_submission": "sample_submission.csv",
}

_IMAGE_DIRS = {
    "train": "train_gallery",
    "val": "validation_gallery",
    "test": "test_gallery",
}


def load_csvs(data_dir: str | pathlib.Path) -> dict[str, pd.DataFrame]:
    """Load all competition CSVs into a dict of DataFrames.

    Args:
        data_dir: Root of the competition data (local ``data/`` or
                  ``/kaggle/input/<slug>/``).

    Returns:
        Dict keyed by short name (e.g. ``"train"``, ``"val_gt"``).
    """
    data_dir = pathlib.Path(data_dir)
    dfs: dict[str, pd.DataFrame] = {}
    for key, filename in _CSV_FILES.items():
        fp = data_dir / filename
        if fp.exists():
            dfs[key] = pd.read_csv(fp)
            print(f"  {key:20s}  shape={dfs[key].shape}")
        else:
            print(f"  {key:20s}  NOT FOUND ({fp})")
    return dfs


def image_dir(data_dir: str | pathlib.Path, split: str) -> pathlib.Path:
    """Return the image directory path for a given split."""
    return pathlib.Path(data_dir) / _IMAGE_DIRS[split]


# ---------------------------------------------------------------------------
# Image opening / sanity
# ---------------------------------------------------------------------------

def safe_open_rgb(path: str | pathlib.Path) -> Image.Image:
    """Open an image and convert to RGB, handling grayscale / CMYK / RGBA."""
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def get_transforms(
    split: str,
    size: int = 224,
    mean: tuple[float, ...] = (0.48145466, 0.4578275, 0.40821073),
    std: tuple[float, ...] = (0.26862954, 0.26130258, 0.27577711),
) -> Callable:
    """Return a torchvision transform pipeline.

    Args:
        split:  ``"train"`` applies augmentation; ``"val"`` / ``"test"``
                applies deterministic centre-crop.
        size:   Target square resolution.
        mean:   Per-channel normalisation mean  (defaults: CLIP).
        std:    Per-channel normalisation std   (defaults: CLIP).

    Returns:
        A callable ``transform(PIL.Image) -> torch.Tensor``.
    """
    try:
        import torchvision.transforms as T  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError("torchvision is required for get_transforms()") from exc

    normalize = T.Normalize(mean=mean, std=std)

    if split == "train":
        return T.Compose(
            [
                T.RandomResizedCrop(size, scale=(0.5, 1.0), ratio=(0.75, 1.333)),
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                T.ToTensor(),
                normalize,
            ]
        )
    else:  # val / test
        return T.Compose(
            [
                T.Resize(int(size * 1.143)),   # resize shorter side to ~256 for size=224
                T.CenterCrop(size),
                T.ToTensor(),
                normalize,
            ]
        )


def get_tta_transforms(
    size: int = 224,
    mean: tuple[float, ...] = (0.48145466, 0.4578275, 0.40821073),
    std: tuple[float, ...] = (0.26862954, 0.26130258, 0.27577711),
) -> list[Callable]:
    """Return a list of TTA transforms: [original, horizontal flip].

    Embeddings from each transform should be averaged (L2-normalized after
    averaging) for TTA.
    """
    try:
        import torchvision.transforms as T  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError("torchvision is required for get_tta_transforms()") from exc

    normalize = T.Normalize(mean=mean, std=std)
    base = T.Compose(
        [T.Resize(int(size * 1.143)), T.CenterCrop(size), T.ToTensor(), normalize]
    )
    flipped = T.Compose(
        [
            T.Resize(int(size * 1.143)),
            T.CenterCrop(size),
            T.RandomHorizontalFlip(p=1.0),
            T.ToTensor(),
            normalize,
        ]
    )
    return [base, flipped]


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class GalleryDataset:
    """Map-style dataset that returns ``(image_tensor, image_id)`` pairs.

    Args:
        df:         DataFrame with at least an ``image_id`` column.
        img_dir:    Directory containing the JPEG files.
        transform:  Transform applied to the PIL image before returning.

    Note:
        ``shuffle=False`` in the DataLoader is **mandatory** so that the
        order of extracted features matches the order of ``image_id`` in
        ``df``.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        img_dir: str | pathlib.Path,
        transform: Optional[Callable] = None,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.img_dir = pathlib.Path(img_dir)
        self.transform = transform
        self.image_ids: list[str] = self.df["image_id"].tolist()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = self.img_dir / f"{row['image_id']}.jpg"
        img = safe_open_rgb(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, row["image_id"]


class QueryDataset:
    """Simple dataset over the query CSV rows.

    Returns ``(description, query_id, category_id)`` tuples.
    ``category_id`` is ``-1`` for test queries (no ground truth).
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[str, str, int]:
        row = self.df.iloc[idx]
        cat_id = str(row["category_id"]) if "category_id" in row and pd.notna(row["category_id"]) else ""
        return str(row["description"]), str(row["query_id"]), cat_id


# ---------------------------------------------------------------------------
# Validation ground truth helpers
# ---------------------------------------------------------------------------

def parse_val_ground_truth(val_gt_df: pd.DataFrame) -> dict[str, list[str]]:
    """Parse ``validation_ground_truth.csv`` into a query_id → [image_id] dict.

    ``relevant_image_ids`` is a space-separated string in the CSV.
    """
    result: dict[str, list[str]] = {}
    for _, row in val_gt_df.iterrows():
        ids = str(row["relevant_image_ids"]).split()
        result[str(row["query_id"])] = ids
    return result


def build_category_index(
    gallery_df: pd.DataFrame,
    train_df: pd.DataFrame,
) -> dict[int, list[str]]:
    """Map ``category_id → [image_id, ...]`` from a labelled gallery DataFrame.

    Works for either train or validation gallery.
    """
    merged = gallery_df.merge(train_df[["image_id", "category_id"]], on="image_id", how="left")
    index: dict[str, list[str]] = {}
    for _, row in merged.iterrows():
        cat = str(row["category_id"])
        index.setdefault(cat, []).append(str(row["image_id"]))
    return index
