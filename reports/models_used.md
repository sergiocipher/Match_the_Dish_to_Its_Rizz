# Models Used

**AGENTS.md Hard Rule 5:** Every pretrained checkpoint MUST be recorded here before use.
Record the name, hub ID, source URL, training data, and license.
All models must also appear in the final report (AGENTS.md §13, point 3).

Do NOT record Food-101-specific fine-tuned weights (Hard Rule 4).
Only generic pretrained checkpoints are permitted.

---

| Name | Hub ID / Tag | Source URL | Training Data | License | Used in Phase |
|------|-------------|------------|---------------|---------|---------------|
| ResNet-50 (ImageNet-1k) | `resnet50.a1_in1k` (timm) | https://github.com/huggingface/pytorch-image-models | ImageNet-1k | Apache 2.0 | 1 (baseline) |
| OpenCLIP ViT-L/14 (OpenAI) | `ViT-L-14 / openai` (open_clip) | https://github.com/mlfoundations/open_clip | WIT-400M (OpenAI) | MIT | 3, 4, 5 |
| OpenCLIP ViT-L/14 (LAION-2B) | `ViT-L-14 / laion2b_s32b_b82k` (open_clip) | https://github.com/mlfoundations/open_clip | LAION-2B | MIT | 3 |
| OpenCLIP ViT-H/14 (LAION-2B) | `ViT-H-14 / laion2b_s32b_b79k` (open_clip) | https://github.com/mlfoundations/open_clip | LAION-2B | MIT | 3 |
| SigLIP SO400M/14 | `ViT-SO400M-14-SigLIP / webli` (open_clip) | https://github.com/mlfoundations/open_clip | WebLI | Apache 2.0 | 3, 4 |

> Add rows here via `src/models.record_model_provenance()` or manually before each experiment.
