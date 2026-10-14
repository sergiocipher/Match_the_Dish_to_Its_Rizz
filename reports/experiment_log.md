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
| E001 | 2026-10-14 | 1 | ResNet-50 + LogReg + surface-form text matching | C=1.0, lbfgs, seed=42 | 42 | 0.4277 | 0.4054 | 0.5861 | 0.7426 | 0.9604 | 0.8383 | Baseline. Below expected ~0.50 R-prec; Hit@1=0.74 shows images are ranked near top. Move to CLIP zero-shot. |
| E002 | 2026-10-14 | 3 | ViT-L/14 (openai) zero-shot | prompt_ensemble, softmax, TTA=hflip, T=1.0, seed=42 | 42 | 0.8109 | 0.8466 | 0.9366 | 0.9802 | 1.0000 | 0.9901 | Strong zero-shot. Beat baseline by +0.38 R-prec. |
| E003 | 2026-10-14 | 3 | SigLIP SO400M zero-shot | prompt_ensemble, softmax, TTA=hflip, T=1.0, seed=42 | 42 | 0.8871 | 0.9283 | 0.9663 | 0.9901 | 1.0000 | 0.9950 | Best so far. +0.076 over ViT-L/14. SigLIP is the stronger backbone. Use as primary. |
| E004 | 2026-10-14 | 4 | LogReg C=100.0 on SigLIP feats | C=100.0, lbfgs, seed=42 | 42 | 0.9168 | 0.9601 | 0.9921 | 1.0000 | 1.0000 | +0.030 over zero-shot. Hit@1=1.0. |
| E005 | 2026-10-14 | 4 | Prototype (NCM) on SigLIP feats | seed=42 | 42 | 0.8564 | 0.8935 | 0.9584 | 0.9802 | 0.9901 | Below zero-shot. Not useful alone. |
| E006 | 2026-10-14 | 4 | WiSE-FT α=0.3 (zero-shot + LogReg) on SigLIP | alpha=0.3, row-norm blend, seed=42 | 42 | 0.9287 | 0.9645 | 0.9921 | 1.0000 | 1.0000 | **Best Phase 4. Public LB=0.95112.** Selected as primary. |
| E007 | 2026-10-14 | 4 | Ensemble SigLIP+ViT-L/14 zero-shot | equal weights, log-prob space | 42 | 0.8752 | 0.9142 | 0.9743 | 1.0000 | 1.0000 | Worse than SigLIP alone. ViT-L/14 dilutes SigLIP. Skip in ensemble. |

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
