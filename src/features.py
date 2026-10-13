"""
features.py – Extract and cache L2-normalised image / text embeddings.

Design rules (AGENTS.md §14.2):
- DataLoader MUST use shuffle=False so feature order matches gallery CSV order.
- IDs are saved alongside features in a JSON file; assert alignment on load.
- Features are cached per model per split under OUT_DIR/features/<model_tag>/.
- TTA: average original + h-flip embeddings, then re-normalise.

Public API
----------
extract_image_features(model, loader, device)  -> (np.ndarray, list[str])
extract_text_features(model_or_fn, texts)      -> np.ndarray
save_features(features, ids, path)
load_features(path)                            -> (np.ndarray, list[str])
build_prompt_templates(category_names, ...)    -> list[str]
"""

from __future__ import annotations

import json
import pathlib
from typing import Callable, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_image_features(
    model,
    loader,
    device: str = "cuda",
    use_tta: bool = False,
    tta_loader=None,
) -> tuple[np.ndarray, list[str]]:
    """Extract L2-normalised image features from a DataLoader.

    Args:
        model:      Model with an ``encode_image`` method (CLIP-style) or
                    a callable that accepts a batch of images and returns
                    embeddings.  Must already be on ``device`` and in
                    ``eval()`` mode.
        loader:     DataLoader that yields ``(image_tensor, image_id)`` pairs.
                    **Must have ``shuffle=False``.**
        device:     CUDA device string or ``"cpu"``.
        use_tta:    If True, also pass ``tta_loader`` (h-flip) and average.
        tta_loader: Second DataLoader with horizontally flipped images.

    Returns:
        ``(features, ids)`` where ``features`` is shape ``(N, D)`` float32
        and ``ids`` is a list of image ID strings in the same order.
    """
    import torch  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: PLC0415

    model.eval()
    feats_list: list[np.ndarray] = []
    ids: list[str] = []

    with torch.no_grad():
        for imgs, img_ids in loader:
            imgs = imgs.to(device)
            with torch.autocast(device_type=device.split(":")[0], dtype=torch.float16):
                if hasattr(model, "encode_image"):
                    f = model.encode_image(imgs)
                else:
                    f = model(imgs)
            f = F.normalize(f.float(), dim=-1)
            feats_list.append(f.cpu().numpy())
            ids.extend(list(img_ids))

    features = np.concatenate(feats_list, axis=0).astype(np.float32)

    if use_tta and tta_loader is not None:
        tta_feats_list: list[np.ndarray] = []
        with torch.no_grad():
            for imgs, _ in tta_loader:
                imgs = imgs.to(device)
                with torch.autocast(device_type=device.split(":")[0], dtype=torch.float16):
                    if hasattr(model, "encode_image"):
                        f = model.encode_image(imgs)
                    else:
                        f = model(imgs)
                f = F.normalize(f.float(), dim=-1)
                tta_feats_list.append(f.cpu().numpy())
        tta_features = np.concatenate(tta_feats_list, axis=0).astype(np.float32)
        # average then re-normalise
        avg = features + tta_features
        norms = np.linalg.norm(avg, axis=1, keepdims=True) + 1e-9
        features = avg / norms

    return features, ids


def extract_text_features(
    encode_fn: Callable[[list[str]], np.ndarray],
    texts: list[str],
    batch_size: int = 256,
) -> np.ndarray:
    """Encode texts in batches and return L2-normalised embeddings.

    Args:
        encode_fn:  Callable ``(list[str]) -> np.ndarray`` shape ``(B, D)``.
        texts:      List of strings to encode.
        batch_size: Texts per batch (to avoid OOM).

    Returns:
        Float32 array of shape ``(len(texts), D)``, L2-normalised.
    """
    all_embs: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        emb = encode_fn(batch)
        if not isinstance(emb, np.ndarray):
            emb = emb.cpu().float().numpy()
        all_embs.append(emb.astype(np.float32))

    features = np.concatenate(all_embs, axis=0)
    norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-9
    return features / norms


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_features(
    features: np.ndarray,
    ids: list[str],
    out_dir: str | pathlib.Path,
    split: str,
) -> None:
    """Save features and IDs to disk.

    Creates:
        ``<out_dir>/<split>_image.npy``
        ``<out_dir>/<split>_ids.json``

    Args:
        features: Float32 array ``(N, D)``.
        ids:      List of ``N`` image ID strings.
        out_dir:  Directory to write into (created if absent).
        split:    E.g. ``"train"``, ``"val"``, ``"test"``.
    """
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npy_path = out_dir / f"{split}_image.npy"
    ids_path = out_dir / f"{split}_ids.json"

    np.save(npy_path, features)
    with open(ids_path, "w") as fh:
        json.dump(ids, fh)

    print(f"  Saved {split} features: {features.shape}  ->  {npy_path}")


def save_text_features(
    features: np.ndarray,
    category_ids: list[int],
    out_dir: str | pathlib.Path,
    tag: str = "text",
) -> None:
    """Save text / category embeddings alongside a category-id JSON."""
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{tag}.npy", features)
    with open(out_dir / f"{tag}_category_ids.json", "w") as fh:
        json.dump(category_ids, fh)
    print(f"  Saved text features: {features.shape}  ->  {out_dir}/{tag}.npy")


def load_features(
    out_dir: str | pathlib.Path,
    split: str,
    gallery_df=None,
) -> tuple[np.ndarray, list[str]]:
    """Load cached features and optionally assert ID alignment.

    Args:
        out_dir:    Directory containing the ``.npy`` and ``.json`` files.
        split:      E.g. ``"train"``, ``"val"``, ``"test"``.
        gallery_df: If provided, assert that loaded IDs match
                    ``gallery_df.image_id.tolist()``.

    Returns:
        ``(features, ids)``
    """
    out_dir = pathlib.Path(out_dir)
    features = np.load(out_dir / f"{split}_image.npy")
    with open(out_dir / f"{split}_ids.json") as fh:
        ids = json.load(fh)

    if gallery_df is not None:
        expected = gallery_df["image_id"].tolist()
        assert ids == expected, (
            f"Feature ID mismatch for split='{split}'! "
            f"Got {len(ids)} IDs but expected {len(expected)}. "
            "Re-extract features."
        )

    return features.astype(np.float32), ids


def load_text_features(
    out_dir: str | pathlib.Path,
    tag: str = "text",
) -> tuple[np.ndarray, list[int]]:
    """Load cached text features and category IDs."""
    out_dir = pathlib.Path(out_dir)
    features = np.load(out_dir / f"{tag}.npy").astype(np.float32)
    with open(out_dir / f"{tag}_category_ids.json") as fh:
        category_ids = json.load(fh)
    return features, category_ids


# ---------------------------------------------------------------------------
# Prompt templates for category text features
# ---------------------------------------------------------------------------

_DEFAULT_TEMPLATES = [
    "a photo of {name}, a type of food.",
    "a close-up photo of {name}.",
    "{name}, a dish.",
    "a picture of {name}.",
    "food: {name}.",
    "a delicious {name}.",
    "a serving of {name}.",
]


def build_prompt_texts(
    category_names: list[str],
    alternative_names: Optional[dict[int, list[str]]] = None,
    templates: Optional[list[str]] = None,
    descriptions: Optional[dict[int, str]] = None,
) -> tuple[list[str], list[int]]:
    """Build a list of prompt strings and matching category indices.

    For each category we generate one prompt per template (and per alternative
    name if provided).  The caller averages the embeddings per category.

    Args:
        category_names:    List of canonical category name strings.
        alternative_names: Optional ``{idx: [alt_name, ...]}`` mapping.
        templates:         Prompt templates with ``{name}`` placeholder.
        descriptions:      Optional ``{idx: description_text}`` mapping; each
                           description is added as an additional prompt.

    Returns:
        ``(texts, category_indices)`` where both lists have the same length.
        ``category_indices[i]`` is the zero-based category index for text ``i``.
    """
    if templates is None:
        templates = _DEFAULT_TEMPLATES

    texts: list[str] = []
    cat_indices: list[int] = []

    for idx, name in enumerate(category_names):
        # canonical name variants
        names_to_use = [name]
        if alternative_names and idx in alternative_names:
            names_to_use += alternative_names[idx]

        for n in names_to_use:
            for tpl in templates:
                texts.append(tpl.format(name=n))
                cat_indices.append(idx)

        # optional description
        if descriptions and idx in descriptions and descriptions[idx]:
            texts.append(descriptions[idx])
            cat_indices.append(idx)

    return texts, cat_indices


def aggregate_text_features(
    raw_features: np.ndarray,
    cat_indices: list[int],
    n_categories: int,
) -> np.ndarray:
    """Average and re-normalise embeddings per category.

    Args:
        raw_features: ``(M, D)`` array of per-prompt embeddings.
        cat_indices:  Category index for each row in ``raw_features``.
        n_categories: Total number of categories ``K``.

    Returns:
        ``(K, D)`` array of L2-normalised category prototypes.
    """
    D = raw_features.shape[1]
    agg = np.zeros((n_categories, D), dtype=np.float32)
    counts = np.zeros(n_categories, dtype=np.int32)

    for i, cat_idx in enumerate(cat_indices):
        agg[cat_idx] += raw_features[i]
        counts[cat_idx] += 1

    counts = np.maximum(counts, 1)
    agg /= counts[:, None]
    norms = np.linalg.norm(agg, axis=1, keepdims=True) + 1e-9
    return agg / norms
