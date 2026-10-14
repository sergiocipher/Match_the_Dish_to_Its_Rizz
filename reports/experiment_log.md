# Experiment Log

One row per experiment. **Never delete rows.** Add a row after every run, even failed ones.

See AGENTS.md §11 for the template and §12 for submission checklist.

---

## Metric definitions
- **Val R-prec**: R-precision on `validation_queries.csv` (R=10 per query, 202 queries).
- **mAP@20**: mean Average Precision at 20.
- **P@5**: Precision at 5.
- **Hit@1 / Hit@5**: at least 1 relevant image in top-1 / top-5.
- **MRR@20**: Mean Reciprocal Rank at 20.
- All metrics computed by `src/metrics.py` (unit-tested in `tests/test_metrics.py`).

> ⚠️ Validation R=10, Test R≈19–20 → scores are **not** directly comparable across splits. Use val for relative comparison only.

---

## Runs

| ID | Date | Phase | Model / Method | Key Config | Seeds | Val R-prec | mAP@20 | P@5 | Hit@1 | Hit@5 | MRR@20 | Notes / Decision |
|----|------|-------|----------------|------------|-------|-----------|--------|-----|-------|-------|--------|-----------------|
| E001 | YYYY-MM-DD | 1 | ResNet-50 + logistic regression + surface-form text matching | lr=1e-3, C=1.0, 10 ep | 42 | — | — | — | — | — | — | Baseline. Run `01_baseline.ipynb` to fill metrics. |

---

## Submissions

| Sub ID | Date | Config | Val R-prec | Public LB | Final candidate? | Notes |
|--------|------|--------|-----------|-----------|-----------------|-------|
| — | — | — | — | — | — | — |

---

## State of project (updated each session)

> *Last updated: 2026-10-14*
>
> **Current status:** Phase 0 (EDA) ✓ and Phase 2 (text matching) ✓ complete. Text matcher hits 100% on train + val + test queries with zero unmatched test queries.
>
> **Next step:** Phase 1 baseline — install timm/scikit-learn in venv, extract ResNet-50 features, fit logistic regression, evaluate val R-precision.
