# WVC 2026 — YOLO26-seg for Skin Lesion Segmentation

Experimental artifacts of:

> A. V. de M. Rodrigues, R. B. Oliveira, F. de B. Vidal.
> **YOLO26-seg for Skin Lesion Segmentation: Resolving the Precision-Efficiency Trade-off.**
> Workshop de Visão Computacional (WVC), 2026.

Five YOLO26-seg scales (`nano`, `small`, `medium`, `large`, `xlarge`) on the official ISIC 2018 Task 1 benchmark.

## Contents

| Folder | Pipeline stage (paper section) | Files |
|---|---|---|
| `baseline/<scale>/` | Baseline configuration and default training (IV-A) | `results.csv`, `results.png` (training and validation curves), `args.yaml` |
| `hpo/<scale>/` | Domain-specific hyperparameter optimization (IV-B) | `best_hyperparameters.yaml`, `tune_results.csv`, `tune_fitness.png` |
| `optimized/<scale>/` | Optimized fine-tuning protocol (IV-C) | `results.csv`, `results.png`, `args.yaml` |
| `cv/<scale>/` | 5-fold cross-validation (IV-D) | `metrics_per_fold.csv`, `metrics_summary.json`, `fold_<k>/{results.csv, results.png, args.yaml}` |
| `pixelwise/` | Pixel-wise JSI and DSC (Table I) | `pixelwise_metrics.json`, `per_image/<split>__<scale>[__fold_<k>].csv` |
| `summary/` | Summary tables | see below |
| `scripts/` | Evaluation and aggregation code | see below |

**`summary/`**

| File | Content |
|---|---|
| `pixelwise_main_table.csv` | Table I: JSI / DSC per scale (baseline, optimized, 5-fold CV, test set), parameters and GFLOPs |
| `test_paired_comparisons.csv` | Paired test-set comparisons between scales (bootstrap 95 % CI, Wilcoxon signed-rank) |
| `model_complexity.csv` | Parameters and GFLOPs per scale |
| `phase1_baseline.csv` | Baseline: best epoch (`best.pt`), its fitness mAP50-95(B) + mAP50-95(M), instance metrics, training configuration |
| `phase2_hpo_best_params.csv` | Selected hyperparameters and HPO cost per scale |
| `phase3_finetune.csv` | Optimized fine-tuning: best epoch (`best.pt`), its fitness, instance metrics, training configuration |
| `phase4_cv_ultralytics_metrics.csv` | Cross-validation instance metrics per fold (epoch of highest mask mAP50-95), mean and SD; the pixel-wise CV uses each fold's `best.pt` |
| `dataset_and_cv_splits.csv` | `image_id`, `partition` (train / val / test) and `cv_fold` of every image |

**`scripts/`**

| Script | Purpose |
|---|---|
| `evaluate_pixelwise_metrics.py` | Pixel-wise JSI / DSC at the original image resolution (top-1 instance), writes `pixelwise/` |
| `build_wvc_summaries.py` | Rebuilds the `summary/` tables from the files in this folder |
| `analyze_wvc.py` | Statistics of the training curves, HPO and CV runs |
| `make_small_test_figures.py` | Qualitative test-set figures of the Small model |

## Code and environment

The training codebase used for the paper is frozen as the git tag
[`wvc2026`](https://github.com/antonio-vmoura/sandbox_yolo26/tree/wvc2026) (`git checkout wvc2026`).

Environment: Docker, Python 3.11.14, PyTorch 2.5.1+cu121, Ultralytics 8.4.21, `seed=0`, `deterministic=True`.
