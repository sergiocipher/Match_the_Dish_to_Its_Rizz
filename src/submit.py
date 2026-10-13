"""
submit.py – Build, validate, and save the final submission CSV.

AGENTS.md Hard Rules (§2.3):
  13. Run validate_submission.py before every submission.
  14. Save every submission to artifacts/submissions/ with timestamp + config name.
  15. Each row: exactly 20 distinct IDs from test_gallery.csv; all 303 query IDs present.

Public API
----------
build_submission(S, gallery_ids, query_ids, query_cat_idx, top)  -> pd.DataFrame
validate_submission(df, test_gallery_df, n_ids)                  -> bool
save_submission(df, out_dir, config_name)                        -> pathlib.Path
run_official_validator(submission_path, data_dir)                -> bool
"""

from __future__ import annotations

import datetime
import pathlib
import subprocess
import sys
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Build submission  (AGENTS.md §14.5)
# ---------------------------------------------------------------------------

def build_submission(
    S: np.ndarray,
    gallery_ids: list[str],
    query_ids: list[str],
    query_cat_idx: list[int],
    top: int = 20,
) -> pd.DataFrame:
    """Build a submission DataFrame from a score matrix.

    Args:
        S:             ``(N_images, K_classes)`` final score matrix.
                       Higher = better match.
        gallery_ids:   Length-``N`` list of gallery image ID strings; must
                       match the row order of ``S``.
        query_ids:     Length-``Q`` list of query ID strings.
        query_cat_idx: Length-``Q`` list of **zero-based** category indices
                       (one per query; returned by text matching).
        top:           Number of images to include per query (must be 20).

    Returns:
        DataFrame with columns ``["query_id", "image_ids"]``.
        ``image_ids`` is a space-separated string, best-first.
    """
    rows: list[tuple[str, str]] = []

    for qid, c in zip(query_ids, query_cat_idx):
        order = np.argsort(-S[:, c], kind="stable")[:top]
        ids = [gallery_ids[j] for j in order]
        assert len(set(ids)) == top, (
            f"Duplicate image IDs for query {qid}! Recheck scoring."
        )
        rows.append((qid, " ".join(ids)))

    return pd.DataFrame(rows, columns=["query_id", "image_ids"])


# ---------------------------------------------------------------------------
# Internal validation
# ---------------------------------------------------------------------------

def validate_submission(
    df: pd.DataFrame,
    test_gallery_df: pd.DataFrame,
    expected_n_queries: int = 303,
    expected_n_ids: int = 20,
) -> bool:
    """Check submission format before calling the official validator.

    Checks:
    - Correct columns (``query_id``, ``image_ids``).
    - Exactly ``expected_n_queries`` rows with no duplicate query IDs.
    - Each row has exactly ``expected_n_ids`` distinct image IDs.
    - All image IDs are present in ``test_gallery_df``.

    Args:
        df:                   Submission DataFrame.
        test_gallery_df:      ``test_gallery.csv`` DataFrame.
        expected_n_queries:   303 for the competition.
        expected_n_ids:       20 per row.

    Returns:
        True if all checks pass; False otherwise (with printed error messages).
    """
    ok = True

    # Column check – must pass before we touch any column values
    if list(df.columns) != ["query_id", "image_ids"]:
        print(f"  [FAIL] Columns: expected ['query_id', 'image_ids'], got {list(df.columns)}")
        return False

    # Row count + duplicate query IDs
    if len(df) != expected_n_queries:
        print(f"  [FAIL] Row count: expected {expected_n_queries}, got {len(df)}")
        ok = False
    if df["query_id"].duplicated().any():
        dupes = df["query_id"][df["query_id"].duplicated()].tolist()
        print(f"  [FAIL] Duplicate query IDs: {dupes[:5]}")
        ok = False

    valid_image_ids = set(test_gallery_df["image_id"].astype(str).tolist())
    per_row_ok = True

    for _, row in df.iterrows():
        ids = str(row["image_ids"]).split()
        if len(ids) != expected_n_ids:
            if per_row_ok:
                print(f"  [FAIL] Row {row['query_id']}: expected {expected_n_ids} IDs, got {len(ids)}")
            per_row_ok = False
            ok = False
        if len(set(ids)) != len(ids):
            if per_row_ok:
                print(f"  [FAIL] Row {row['query_id']}: duplicate image IDs")
            per_row_ok = False
            ok = False
        unknown = [i for i in ids if i not in valid_image_ids]
        if unknown:
            if per_row_ok:
                print(f"  [FAIL] Row {row['query_id']}: unknown image IDs: {unknown[:3]}")
            per_row_ok = False
            ok = False

    if ok:
        print("  [PASS] Internal validation passed.")
    return ok


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_submission(
    df: pd.DataFrame,
    out_dir: str | pathlib.Path,
    config_name: str = "submission",
) -> pathlib.Path:
    """Save the submission CSV with a timestamp in the filename.

    Also copies the file to the Kaggle working directory if running on Kaggle.

    Args:
        df:          Submission DataFrame.
        out_dir:     Directory to save into (created if absent).
        config_name: Human-readable label included in the filename.

    Returns:
        Path to the saved file.
    """
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{ts}_{config_name}.csv"
    path = out_dir / filename

    df.to_csv(path, index=False)
    print(f"  Saved submission: {path}  ({len(df)} rows)")

    # On Kaggle, always copy to /kaggle/working/submission.csv
    import os  # noqa: PLC0415
    if os.path.exists("/kaggle/input"):
        kaggle_path = pathlib.Path("/kaggle/working/submission.csv")
        df.to_csv(kaggle_path, index=False)
        print(f"  Copied to: {kaggle_path}")

    return path


# ---------------------------------------------------------------------------
# Official validator runner
# ---------------------------------------------------------------------------

def run_official_validator(
    submission_path: str | pathlib.Path,
    data_dir: str | pathlib.Path,
    validator_script: str = "validate_submission.py",
) -> bool:
    """Run the competition's validate_submission.py and return pass/fail.

    Args:
        submission_path: Path to the submission CSV.
        data_dir:        Competition data directory (contains gallery CSVs).
        validator_script: Name or path of the validator script.

    Returns:
        True if the validator exits with code 0; False otherwise.
    """
    cmd = [
        sys.executable,
        str(validator_script),
        str(submission_path),
        "--data-dir",
        str(data_dir),
    ]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)

    if result.returncode == 0:
        print("  [PASS] Official validator passed.")
        return True
    else:
        print(f"  [FAIL] Official validator exited with code {result.returncode}.")
        return False
