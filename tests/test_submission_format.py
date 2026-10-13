"""
tests/test_submission_format.py – Unit tests for src/submit.py format validation.

Run with:
    python -m pytest tests/test_submission_format.py -v
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest
from src.submit import build_submission, validate_submission


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_gallery(n: int = 2000) -> pd.DataFrame:
    return pd.DataFrame({"image_id": [f"img_{i:04d}" for i in range(n)]})


def _make_score_matrix(n_images: int = 2000, n_classes: int = 101) -> np.ndarray:
    rng = np.random.default_rng(42)
    S = rng.random((n_images, n_classes)).astype(np.float32)
    return S


def _make_queries(n_queries: int = 303, n_classes: int = 101):
    query_ids = [f"q_{i:04d}" for i in range(n_queries)]
    # cycle through categories
    cat_indices = [i % n_classes for i in range(n_queries)]
    return query_ids, cat_indices


# ---------------------------------------------------------------------------
# build_submission
# ---------------------------------------------------------------------------

class TestBuildSubmission:
    def test_shape(self):
        gallery = _make_gallery(2000)
        S = _make_score_matrix(2000, 101)
        query_ids, cat_idx = _make_queries(303, 101)
        df = build_submission(S, gallery["image_id"].tolist(), query_ids, cat_idx, top=20)

        assert list(df.columns) == ["query_id", "image_ids"]
        assert len(df) == 303

    def test_exactly_20_ids_per_row(self):
        gallery = _make_gallery(2000)
        S = _make_score_matrix(2000, 101)
        query_ids, cat_idx = _make_queries(303, 101)
        df = build_submission(S, gallery["image_id"].tolist(), query_ids, cat_idx, top=20)

        for _, row in df.iterrows():
            ids = str(row["image_ids"]).split()
            assert len(ids) == 20, f"Row {row['query_id']} has {len(ids)} IDs"

    def test_no_duplicates_per_row(self):
        gallery = _make_gallery(2000)
        S = _make_score_matrix(2000, 101)
        query_ids, cat_idx = _make_queries(303, 101)
        df = build_submission(S, gallery["image_id"].tolist(), query_ids, cat_idx, top=20)

        for _, row in df.iterrows():
            ids = str(row["image_ids"]).split()
            assert len(set(ids)) == len(ids), f"Duplicate IDs in row {row['query_id']}"

    def test_ids_from_gallery(self):
        gallery = _make_gallery(2000)
        S = _make_score_matrix(2000, 101)
        query_ids, cat_idx = _make_queries(303, 101)
        df = build_submission(S, gallery["image_id"].tolist(), query_ids, cat_idx, top=20)

        valid_ids = set(gallery["image_id"].tolist())
        for _, row in df.iterrows():
            for img_id in str(row["image_ids"]).split():
                assert img_id in valid_ids, f"Unknown image ID {img_id}"

    def test_query_ids_preserved(self):
        gallery = _make_gallery(2000)
        S = _make_score_matrix(2000, 101)
        query_ids, cat_idx = _make_queries(303, 101)
        df = build_submission(S, gallery["image_id"].tolist(), query_ids, cat_idx, top=20)

        assert list(df["query_id"]) == query_ids

    def test_best_image_is_highest_score(self):
        """The first image ID in each row must be the argmax for that category."""
        gallery = _make_gallery(50)
        n_classes = 5
        S = _make_score_matrix(50, n_classes)
        query_ids = ["q0"]
        cat_idx = [0]
        df = build_submission(S, gallery["image_id"].tolist(), query_ids, cat_idx, top=5)
        first_id = df.iloc[0]["image_ids"].split()[0]
        expected_row = int(np.argmax(S[:, 0]))
        expected_id = gallery.iloc[expected_row]["image_id"]
        assert first_id == expected_id

    def test_small_gallery_raises_if_not_enough_images(self):
        """Gallery smaller than top= should still work (top is clamped by argsort)."""
        gallery = _make_gallery(15)
        S = _make_score_matrix(15, 3)
        # top=20 but only 15 images; argsort returns 15 indices
        # build_submission asserts len(set(ids)) == top, so this SHOULD raise
        with pytest.raises(AssertionError):
            build_submission(S, gallery["image_id"].tolist(), ["q0"], [0], top=20)


# ---------------------------------------------------------------------------
# validate_submission
# ---------------------------------------------------------------------------

class TestValidateSubmission:
    def _valid_df(self, gallery, n_queries=303, n_ids=20):
        gallery_ids = gallery["image_id"].tolist()
        n_gallery = len(gallery_ids)
        rows = []
        for i in range(n_queries):
            # Pick n_ids distinct IDs by cycling through the gallery with an offset
            start = (i * n_ids) % n_gallery
            indices = [(start + j) % n_gallery for j in range(n_ids)]
            # Deduplicate while preserving order (handles small galleries gracefully)
            seen: set = set()
            unique_indices = []
            for idx in indices:
                if idx not in seen:
                    seen.add(idx)
                    unique_indices.append(idx)
            # If we still have fewer than n_ids, extend from the remaining pool
            if len(unique_indices) < n_ids:
                for idx in range(n_gallery):
                    if idx not in seen:
                        seen.add(idx)
                        unique_indices.append(idx)
                    if len(unique_indices) == n_ids:
                        break
            ids = " ".join(gallery_ids[idx] for idx in unique_indices[:n_ids])
            rows.append((f"q_{i:04d}", ids))
        return pd.DataFrame(rows, columns=["query_id", "image_ids"])

    def test_valid_submission_passes(self):
        gallery = _make_gallery(2000)
        df = self._valid_df(gallery)
        assert validate_submission(df, gallery) is True

    def test_wrong_columns_fails(self):
        gallery = _make_gallery(2000)
        df = self._valid_df(gallery)
        df = df.rename(columns={"image_ids": "images"})
        assert validate_submission(df, gallery) is False

    def test_wrong_row_count_fails(self):
        gallery = _make_gallery(2000)
        df = self._valid_df(gallery, n_queries=100)
        assert validate_submission(df, gallery, expected_n_queries=303) is False

    def test_duplicate_query_id_fails(self):
        gallery = _make_gallery(2000)
        df = self._valid_df(gallery)
        df.iloc[1, 0] = df.iloc[0, 0]  # duplicate first query_id
        assert validate_submission(df, gallery) is False

    def test_wrong_id_count_per_row_fails(self):
        gallery = _make_gallery(2000)
        df = self._valid_df(gallery, n_ids=19)
        assert validate_submission(df, gallery, expected_n_ids=20) is False

    def test_duplicate_image_ids_in_row_fails(self):
        gallery = _make_gallery(2000)
        df = self._valid_df(gallery)
        # inject a duplicate into the first row
        ids = df.iloc[0]["image_ids"].split()
        ids[1] = ids[0]
        df.at[0, "image_ids"] = " ".join(ids)
        assert validate_submission(df, gallery) is False

    def test_unknown_image_id_fails(self):
        gallery = _make_gallery(2000)
        df = self._valid_df(gallery)
        ids = df.iloc[0]["image_ids"].split()
        ids[0] = "not_in_gallery"
        df.at[0, "image_ids"] = " ".join(ids)
        assert validate_submission(df, gallery) is False
