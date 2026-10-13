"""
text_matching.py – Query → category mapping (Phase 2).

Strategy (in order):
  1. Surface-form exact match (longest-match-wins, word-boundary safe).
  2. Embedding-based fallback using CLIP/SigLIP text encoder cosine similarity.

Public API
----------
build_surface_form_dict(meta_df)   -> dict[int, list[str]]
TextMatcher                        -> .predict(query) -> (category_id, method, score)
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
    # replace hyphens and underscores with spaces
    text = re.sub(r"[-_]", " ", text)
    # remove non-alphanumeric characters (except spaces)
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    # collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _variants(name: str) -> list[str]:
    """Generate surface-form variants for a single name string.

    Produces: normalized form, singular/plural, "and"/"&" swap.
    """
    forms: list[str] = []
    base = normalize(name)
    forms.append(base)

    # singular ↔ plural heuristics
    if base.endswith("ies"):
        forms.append(base[:-3] + "y")
    elif base.endswith("es") and not base.endswith("ses"):
        forms.append(base[:-2])
    elif base.endswith("s") and not base.endswith("ss"):
        forms.append(base[:-1])
    else:
        forms.append(base + "s")

    # ampersand ↔ "and"
    if " & " in base:
        forms.append(base.replace(" & ", " and "))
    if " and " in base:
        forms.append(base.replace(" and ", " & "))

    return list(dict.fromkeys(forms))   # deduplicate while preserving order


# ---------------------------------------------------------------------------
# Surface-form dictionary builder
# ---------------------------------------------------------------------------

def build_surface_form_dict(
    meta_df,   # pd.DataFrame with category_id, category_name, alternative_names, train_paraphrases
) -> dict[int, list[str]]:
    """Return ``{category_id: [surface_form, ...]}``.

    Collects: category_name, every alternative_name, every train_paraphrase,
    plus generated variants of each.
    """
    import ast  # noqa: PLC0415

    result: dict[int, list[str]] = {}
    for _, row in meta_df.iterrows():
        cat_id = int(row["category_id"])
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
            # stored as Python list literal or comma-separated string
            try:
                items = ast.literal_eval(str(raw))
            except (ValueError, SyntaxError):
                items = [s.strip() for s in str(raw).split(",") if s.strip()]
            for item in items:
                add(str(item))

        result[cat_id] = forms

    return result


# ---------------------------------------------------------------------------
# Longest-match surface-form matcher
# ---------------------------------------------------------------------------

def _make_pattern(forms: list[str]) -> re.Pattern:
    """Compile a single regex that matches any surface form as a whole word."""
    sorted_forms = sorted(forms, key=len, reverse=True)  # longest first
    escaped = [re.escape(f) for f in sorted_forms]
    return re.compile(r"\b(?:" + "|".join(escaped) + r")\b")


class TextMatcher:
    """Query → category_id matcher.

    Args:
        surface_form_dict:  ``{category_id: [form, ...]}``.
        meta_df:            ``category_metadata.csv`` DataFrame (for embeddings).
        embed_fn:           Optional callable ``(list[str]) -> np.ndarray``
                            returning L2-normalised embeddings; used as fallback.
    """

    def __init__(
        self,
        surface_form_dict: dict[int, list[str]],
        meta_df=None,
        embed_fn=None,
    ) -> None:
        self.surface_form_dict = surface_form_dict
        self.meta_df = meta_df
        self.embed_fn = embed_fn

        # pre-compile per-category patterns
        self._patterns: dict[int, re.Pattern] = {
            cat_id: _make_pattern(forms)
            for cat_id, forms in surface_form_dict.items()
        }

        # category_id → category_name for display
        self._cat_names: dict[int, str] = {}
        if meta_df is not None:
            for _, row in meta_df.iterrows():
                self._cat_names[int(row["category_id"])] = str(row["category_name"])

        # embedding cache (lazy)
        self._cat_embeddings = None  # np.ndarray shape (K, D)
        self._cat_ids: list[int] = []

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def predict(self, query: str) -> tuple[int, str, float]:
        """Map a query to a ``(category_id, method, confidence)`` triple.

        ``method`` is one of ``"surface"`` or ``"embedding"``.
        ``confidence`` is 1.0 for surface matches, cosine similarity for embeddings.
        """
        norm_q = normalize(query)

        # 1. Surface-form matching: find all categories whose pattern matches,
        #    pick the one with the longest match span (substring length).
        best_cat: Optional[int] = None
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

        # 2. Embedding fallback
        if self.embed_fn is not None:
            return self._embedding_predict(query)

        # 3. No match – return -1 as sentinel
        return -1, "no_match", 0.0

    def _embedding_predict(self, query: str) -> tuple[int, str, float]:
        """Fallback: embed query, return nearest category by cosine similarity."""
        import numpy as np  # noqa: PLC0415

        if self._cat_embeddings is None:
            self._build_cat_embeddings()

        q_emb = self.embed_fn([query])  # (1, D)
        q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-9)
        sims = (self._cat_embeddings @ q_emb.T).squeeze()  # (K,)
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        return self._cat_ids[best_idx], "embedding", best_sim

    def _build_cat_embeddings(self) -> None:
        """Encode all category descriptions once and cache the result."""
        import numpy as np  # noqa: PLC0415

        assert self.meta_df is not None, "meta_df required for embedding fallback"
        assert self.embed_fn is not None

        texts: list[str] = []
        self._cat_ids = []
        for _, row in self.meta_df.iterrows():
            texts.append(str(row.get("description", row["category_name"])))
            self._cat_ids.append(int(row["category_id"]))

        embs = self.embed_fn(texts)
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9
        self._cat_embeddings = embs / norms

    # ------------------------------------------------------------------
    # Batch convenience
    # ------------------------------------------------------------------

    def predict_batch(self, queries: list[str]) -> list[tuple[int, str, float]]:
        """Map a list of queries to ``(category_id, method, confidence)`` tuples."""
        return [self.predict(q) for q in queries]


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def evaluate_text_matching(matcher: TextMatcher, df) -> float:
    """Compute query → category accuracy on a labelled query DataFrame.

    Args:
        matcher: A fitted :class:`TextMatcher`.
        df:      DataFrame with ``description`` and ``category_id`` columns
                 (``train_queries.csv`` or ``validation_queries.csv``).

    Returns:
        Accuracy in [0, 1].
    """
    correct = 0
    mismatches: list[tuple[str, int, int]] = []
    for _, row in df.iterrows():
        pred_cat, method, _ = matcher.predict(str(row["description"]))
        true_cat = int(row["category_id"])
        if pred_cat == true_cat:
            correct += 1
        else:
            mismatches.append((str(row["description"]), true_cat, pred_cat))

    n = len(df)
    acc = correct / n if n > 0 else 0.0
    print(f"  Text matching accuracy: {correct}/{n} = {acc:.4f}")
    if mismatches:
        print(f"  Mismatches ({len(mismatches)}):")
        for desc, true, pred in mismatches[:10]:
            print(f"    true={true}  pred={pred}  query='{desc[:80]}'")
    return acc
