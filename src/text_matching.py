"""
text_matching.py – Query → category mapping (Phase 2).

Category IDs are hex strings (e.g. ``"c_027b0d"``), NOT integers.

Strategy (in order):
  1. Surface-form exact match (longest-match-wins, word-boundary safe).
  2. Embedding-based fallback using CLIP/SigLIP text encoder cosine similarity.

Public API
----------
build_surface_form_dict(meta_df)    -> dict[str, list[str]]
TextMatcher                         -> .predict(query) -> (category_id, method, score)
evaluate_text_matching(matcher, df) -> float  (accuracy)
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

def normalize(text: str, remove_accents: bool = True) -> str:
    """Lowercase, strip punctuation / accents, collapse whitespace."""
    if remove_accents:
        text = unicodedata.normalize("NFD", text)
        text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = text.lower()
    text = re.sub(r"[-_]", " ", text)
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _variants(name: str) -> list[str]:
    """Generate surface-form variants: normalised, singular/plural, and/& swap."""
    forms: list[str] = []
    base = normalize(name)
    forms.append(base)

    if base.endswith("ies"):
        forms.append(base[:-3] + "y")
    elif base.endswith("es") and not base.endswith("ses"):
        forms.append(base[:-2])
    elif base.endswith("s") and not base.endswith("ss"):
        forms.append(base[:-1])
    else:
        forms.append(base + "s")

    if " & " in base:
        forms.append(base.replace(" & ", " and "))
    if " and " in base:
        forms.append(base.replace(" and ", " & "))

    return list(dict.fromkeys(forms))


# ---------------------------------------------------------------------------
# Surface-form dictionary builder
# ---------------------------------------------------------------------------

def build_surface_form_dict(meta_df) -> dict[str, list[str]]:
    """Return ``{category_id: [surface_form, ...]}``.

    Category IDs are kept as strings throughout (e.g. ``"c_027b0d"``).
    Collects: category_name, alternative_names, train_paraphrases, plus variants.
    """
    import ast

    result: dict[str, list[str]] = {}
    for _, row in meta_df.iterrows():
        cat_id = str(row["category_id"])
        forms: list[str] = []

        def add(text: str) -> None:
            for v in _variants(text):
                if v and v not in forms:
                    forms.append(v)

        add(str(row["category_name"]))

        for col in ("alternative_names", "train_paraphrases"):
            raw = row.get(col, "")
            if isinstance(raw, float) or not raw:
                continue
            try:
                items = ast.literal_eval(str(raw))
            except (ValueError, SyntaxError):
                items = [s.strip() for s in str(raw).split("|") if s.strip()]
            for item in items:
                # paraphrases are full sentences – extract the category name
                # from each one by using it directly as a surface form too
                add(str(item))

        result[cat_id] = forms

    return result


# ---------------------------------------------------------------------------
# Longest-match surface-form matcher
# ---------------------------------------------------------------------------

def _make_pattern(forms: list[str]) -> re.Pattern:
    """Compile a regex that matches any surface form at a word boundary."""
    sorted_forms = sorted(forms, key=len, reverse=True)
    escaped = [re.escape(f) for f in sorted_forms]
    return re.compile(r"\b(?:" + "|".join(escaped) + r")\b")


class TextMatcher:
    """Query → category_id matcher (category IDs are strings).

    Args:
        surface_form_dict:  ``{category_id: [form, ...]}``.
        meta_df:            ``category_metadata.csv`` DataFrame (for fallback).
        embed_fn:           Optional callable ``(list[str]) -> np.ndarray``
                            returning L2-normalised embeddings; used as fallback.
    """

    def __init__(
        self,
        surface_form_dict: dict[str, list[str]],
        meta_df=None,
        embed_fn=None,
    ) -> None:
        self.surface_form_dict = surface_form_dict
        self.meta_df = meta_df
        self.embed_fn = embed_fn

        self._patterns: dict[str, re.Pattern] = {
            cat_id: _make_pattern(forms)
            for cat_id, forms in surface_form_dict.items()
        }

        self._cat_names: dict[str, str] = {}
        if meta_df is not None:
            for _, row in meta_df.iterrows():
                self._cat_names[str(row["category_id"])] = str(row["category_name"])

        self._cat_embeddings = None
        self._cat_ids: list[str] = []

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def predict(self, query: str) -> tuple[str, str, float]:
        """Map a query to ``(category_id, method, confidence)``.

        Returns ``("", "no_match", 0.0)`` when nothing matches.
        """
        norm_q = normalize(query)

        best_cat: Optional[str] = None
        best_len: int = 0
        for cat_id, pat in self._patterns.items():
            m = pat.search(norm_q)
            if m:
                span_len = m.end() - m.start()
                if span_len > best_len:
                    best_len = span_len
                    best_cat = cat_id

        if best_cat is not None:
            return best_cat, "surface", 1.0

        if self.embed_fn is not None:
            return self._embedding_predict(query)

        return "", "no_match", 0.0

    def _embedding_predict(self, query: str) -> tuple[str, str, float]:
        import numpy as np

        if self._cat_embeddings is None:
            self._build_cat_embeddings()

        q_emb = self.embed_fn([query])
        q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-9)
        sims = (self._cat_embeddings @ q_emb.T).squeeze()
        best_idx = int(np.argmax(sims))
        return self._cat_ids[best_idx], "embedding", float(sims[best_idx])

    def _build_cat_embeddings(self) -> None:
        import numpy as np

        assert self.meta_df is not None and self.embed_fn is not None
        texts, self._cat_ids = [], []
        for _, row in self.meta_df.iterrows():
            texts.append(str(row.get("description", row["category_name"])))
            self._cat_ids.append(str(row["category_id"]))

        embs = self.embed_fn(texts)
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9
        self._cat_embeddings = embs / norms

    def predict_batch(self, queries: list[str]) -> list[tuple[str, str, float]]:
        return [self.predict(q) for q in queries]


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def evaluate_text_matching(matcher: TextMatcher, df) -> float:
    """Accuracy of query → category on a labelled DataFrame.

    Args:
        matcher: A fitted :class:`TextMatcher`.
        df:      DataFrame with ``description`` and ``category_id`` columns.

    Returns:
        Accuracy in [0, 1].
    """
    correct = 0
    mismatches: list[tuple[str, str, str]] = []
    for _, row in df.iterrows():
        pred_cat, method, _ = matcher.predict(str(row["description"]))
        true_cat = str(row["category_id"])
        if pred_cat == true_cat:
            correct += 1
        else:
            mismatches.append((str(row["description"]), true_cat, pred_cat))

    n = len(df)
    acc = correct / n if n > 0 else 0.0
    print(f"  Accuracy: {correct}/{n} = {acc:.4f}")
    if mismatches:
        print(f"  Mismatches ({len(mismatches)}):")
        for desc, true, pred in mismatches[:10]:
            true_name = matcher._cat_names.get(true, true)
            pred_name = matcher._cat_names.get(pred, pred) if pred else "NO_MATCH"
            print(f"    true='{true_name}'  pred='{pred_name}'")
            print(f"    query='{desc[:90]}'")
    return acc
