# Food Text-to-Image Retrieval

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![open-clip](https://img.shields.io/badge/open--clip-2.24-blueviolet?style=flat)
![SigLIP](https://img.shields.io/badge/Backbone-SigLIP%20SO400M-4285F4?style=flat&logo=google&logoColor=white)
![Val R-prec](https://img.shields.io/badge/Val%20R--prec-0.9287-brightgreen?style=flat)
![Public LB](https://img.shields.io/badge/Public%20LB-0.9511-success?style=flat&logo=kaggle&logoColor=white)
![Tests](https://img.shields.io/badge/tests-45%20passed-brightgreen?style=flat&logo=pytest&logoColor=white)
![License](https://img.shields.io/badge/data-course%20use%20only-red?style=flat)

Kaggle competition: given a short text description of a dish, rank 20 gallery images by relevance.
Metric: **R-precision** (fraction of the R relevant images recovered in the top-R predictions).

> **State of project:** All phases complete. Fine-tuning (Phase 5) was tried and rejected — worse than Phase 4 on val. **Final pipeline: WiSE-FT α=0.3 (SigLIP SO400M + LogReg C=100), val R-prec=0.9287, public LB=0.9511.**

---

## Quick start

```bash
# Clone the repo
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

| Property | Value |
|---|---|
| Number of categories | **101** (confirmed — Food-101 derived) |
| Train gallery | **2,000** images (~19–20 per category, noisy labels) |
| Val gallery | 1,010 images (10 per category), R = 10 |
| Test gallery | 2,000 images, R ≈ 19–20 |
| Val queries | 202 (2 per category) |
| Test queries | 303 (3 per category, unseen wordings) |
| Confusable groups | pasta (5), cake/dessert (13), rice (4), noodle soup (2), beef (5) |
| Text matching accuracy | **100%** on train + val + test queries |

---

## Workflow phases

| Phase | Notebook | Description | Status |
|---|---|---|---|
| 0 | 00_eda | EDA + repo setup | ✅ Done |
| 1 | 01_baseline | Metrics + ResNet-50 baseline | ✅ Done |
| 2 | — (src only) | Text → category mapping | ✅ Done (100% acc) |
| 3 | 02_zeroshot_vlm | Zero-shot SigLIP / CLIP | ✅ Done |
| 4 | 03_linear_probe | Frozen-feature adapters + WiSE-FT | ✅ Done |
| 5 | 04_finetune | GPU fine-tuning | ✅ Done (rejected — overfit) |
| 6 | 05_ensemble_rerank | Sinkhorn + kNN re-ranking | ✅ Done (no gain) |
| 7 | — | Error analysis | 🔜 Next |
| 8 | 06_final_submission | Final submission | 🔜 Next |

---

## Results

| Method | Val R-prec | mAP@20 | P@5 | Hit@1 | MRR@20 |
|---|---|---|---|---|---|
| ResNet-50 baseline | 0.4277 | 0.4054 | 0.5861 | 0.7426 | 0.8383 |
| CLIP ViT-L/14 zero-shot | 0.8109 | 0.8466 | 0.9366 | 0.9802 | 0.9901 |
| SigLIP SO400M zero-shot | 0.8871 | 0.9283 | 0.9663 | 0.9901 | 0.9950 |
| + LogReg C=100 adapter | 0.9168 | 0.9601 | 0.9921 | 1.0000 | 1.0000 |
| **✅ WiSE-FT α=0.3 (final)** | **0.9287** | **0.9645** | **0.9921** | **1.0000** | **1.0000** |
| Fine-tuned SigLIP + WiSE-FT | 0.8921 | 0.9304 | 0.9723 | 1.0000 | 1.0000 |

**Public LB: 0.9511** (WiSE-FT α=0.3 submission)

---

## Key findings

- **Text matching is trivial** — surface-form exact matching hits 100% accuracy. All 303 test queries matched with no ambiguity.
- **SigLIP SO400M dominates** — zero-shot alone scores 0.887 R-prec, far above the ResNet-50 baseline (0.427).
- **Adapters help** — LogReg on frozen SigLIP features + WiSE-FT blend adds +0.042 R-prec over zero-shot.
- **Fine-tuning hurts** — only 2,000 training images causes overfitting despite AMP, EMA, label smoothing, and early stopping.
- **Sinkhorn re-ranking neutral** — balanced assignment gave no improvement; scores were already well-calibrated.

---

## Hard rules summary

See AGENTS.md §2 for the full list. Key points:
- **Never** use test labels or Food-101-specific pretrained weights.
- **Never** tune on the public leaderboard — use validation only.
- **Always** fix seeds, log every experiment, unit-test the metric.
- **Always** run `validate_submission.py` before submitting.
- **Never** commit images (`data/` and `artifacts/` are gitignored).

---

## License & data terms

Competition images are limited to course use only (`DATA_TERMS.txt`).
Do not redistribute or upload images outside the Kaggle/course context.
All pretrained model checkpoints and their licenses are recorded in `reports/models_used.md`.
