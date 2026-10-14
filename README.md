# Food Text-to-Image Retrieval

Kaggle competition: given a short text description of a dish, rank 20 gallery images by relevance.
Metric: **R-precision** (fraction of the R relevant images recovered in the top-R predictions).

> **State of project:** All phases complete. Fine-tuning (Phase 5) was tried and rejected — worse than Phase 4 on val. **Final pipeline: WiSE-FT α=0.3 (SigLIP SO400M + LogReg C=100), val R-prec=0.9287, public LB=0.9511.**
> **Next step:** Submit final candidates, complete error analysis (Phase 7), write report.

---

## Quick start

```bash
# Clone / unzip the repo
cd food-retrieval

# Install dependencies (Python ≥ 3.10 recommended)
pip install -r requirements.txt

# Run unit tests (must pass before reporting any metric)
python -m pytest tests/ -v

# Then open notebooks in order: 00_eda → 01_baseline → …
```

---

## Project structure

```
food-retrieval/
├── AGENTS.md                      # single source of truth – read before doing anything
├── README.md                      # this file
├── requirements.txt               # pinned versions
├── configs/
│   ├── paths.yaml                 # local vs Kaggle path resolution
│   ├── baseline.yaml              # Phase 1 starter baseline
│   ├── clip_zeroshot.yaml         # Phase 3 zero-shot VLMs
│   ├── adapter_linear_probe.yaml  # Phase 4 frozen-feature adapters
│   └── finetune_vitl14.yaml       # Phase 5 fine-tuning (needs GPU)
├── src/
│   ├── config.py       # path resolution, seeding, config loading
│   ├── data.py         # CSV loading, Datasets, transforms
│   ├── metrics.py      # R-precision, mAP@20, P@5, Hit@k, MRR@20
│   ├── text_matching.py# query → category (surface forms + embedding fallback)
│   ├── features.py     # extract + cache image/text embeddings
│   ├── models.py       # CLIP/timm loaders, LinearProbe, adapters, LLRD
│   ├── train.py        # AMP fine-tuning loop, EMA, cosine LR, early stopping
│   ├── rerank.py       # Sinkhorn balanced assignment, kNN smoothing
│   ├── ensemble.py     # temperature scaling, log-prob ensemble, WiSE-FT
│   └── submit.py       # build + validate + save submission.csv
├── notebooks/
│   ├── 00_eda.ipynb
│   ├── 01_baseline.ipynb
│   ├── 02_zeroshot_vlm.ipynb
│   ├── 03_linear_probe_adapters.ipynb
│   ├── 04_finetune.ipynb          # needs Kaggle GPU
│   ├── 05_ensemble_rerank.ipynb
│   └── 06_final_submission.ipynb
├── data/                          # gitignored – mirrors /kaggle/input/<slug>/
├── artifacts/                     # gitignored (large files)
│   ├── features/                  # <model_tag>/<split>_{image,text}.npy + ids.json
│   ├── checkpoints/
│   ├── oof_preds/
│   └── submissions/               # timestamped submission CSVs
├── reports/
│   ├── experiment_log.md          # one row per experiment (never delete rows)
│   ├── models_used.md             # provenance of every pretrained checkpoint
│   └── figures/
└── tests/
    ├── test_metrics.py
    └── test_submission_format.py
```

---

## Data facts

*(Fill in after running `00_eda.ipynb`)*

| Property | Value |
|---|---|
| Number of categories | **101** (confirmed) |
| Train gallery | **2,000** images (~19–20 per category) |
| Val gallery | 1,010 images (10 per category) |
| Test gallery | 2,000 images |
| Val queries | 202 (2 per category), R = 10 |
| Test queries | 303 (3 per category), R ≈ 19–20 |
| Confusable groups | pasta (5), cake/dessert (13), rice (4), noodle_soup (2), beef (5) |

---

## Workflow phases

| Phase | Notebook | Description | Exit gate |
|---|---|---|---|
| 0 | 00_eda | EDA + repo setup | Category count, gallery sizes, confusable groups confirmed |
| 1 | 01_baseline | Metrics + ResNet-50 baseline | Tests pass; baseline val R-prec logged |
| 2 | — (src only) | Text → category mapping | ≥99% accuracy on train + val queries |
| 3 | 02_zeroshot_vlm | Zero-shot CLIP/SigLIP | Best config chosen; features cached for top 2–3 models |
| 4 | 03_linear_probe | Frozen-feature adapters | Beats zero-shot on val with stable margin (≥3 seeds) |
| 5 | 04_finetune | GPU fine-tuning | Compared to Phase 4 best on val across ≥3 seeds |
| 6 | 05_ensemble_rerank | Ensemble + re-ranking | Ablation table shows each component's contribution |
| 7 | — | Error analysis | ≥3 concrete failure-mode statements |
| 8 | 06_final_submission | Final submission | Pre-submission checklist (AGENTS.md §12) fully ticked |

**Priority order (AGENTS.md §15):** Metrics → text matching → zero-shot → adapters → softmax scoring → balanced re-ranking → fine-tuning → ensemble.
Steps 1–6 cost almost no GPU time and capture most of the gain.

---

## Hard rules summary

See AGENTS.md §2 for the full list. Key points:
- **Never** use test labels or Food-101-specific pretrained weights.
- **Never** tune on the public leaderboard — use validation only.
- **Always** fix seeds, log every experiment, unit-test the metric.
- **Always** run `validate_submission.py` before submitting.
- **Never** commit images (`data/` and `artifacts/` are gitignored).

---

## Results

*(Fill in after experiments — copy from `reports/experiment_log.md`)*

| Method | Val R-prec | mAP@20 | P@5 | Hit@1 | MRR@20 |
|---|---|---|---|---|---|
| ResNet-50 baseline | 0.4277 | 0.4054 | 0.5861 | 0.7426 | 0.8383 |
| CLIP ViT-L/14 zero-shot | 0.8109 | 0.8466 | 0.9366 | 0.9802 | 0.9901 |
| SigLIP SO400M zero-shot | 0.8871 | 0.9283 | 0.9663 | 0.9901 | 0.9950 |
| + LogReg C=100 adapter | 0.9168 | 0.9601 | 0.9921 | 1.0000 | 1.0000 |
| **+ WiSE-FT α=0.3 (final)** | **0.9287** | **0.9645** | **0.9921** | **1.0000** | **1.0000** |
| Fine-tuned SigLIP + WiSE-FT α=0.9 | 0.8921 | 0.9304 | 0.9723 | 1.0000 | 1.0000 |

---

## License & data terms

Competition images are limited to course use only (`DATA_TERMS.txt`).
Do not redistribute or upload images outside the Kaggle/course context.
All pretrained model checkpoints and their licenses are recorded in `reports/models_used.md`.
