# YOLO26 Fine-Tuning

This repository provides scripts and configuration files for fine-tuning the **Ultralytics YOLO26** model in a custom dataset, focusing on skin lesion instance segmentation.

The goal is to evaluate how well YOLO26 adapts to medical images with limited data, leveraging its state-of-the-art segmentation capabilities.

https://www.ultralytics.com/blog/how-to-custom-train-ultralytics-yolo26-for-instance-segmentation

---

## Overview

The training pipeline includes:

* Loading and preparing the dataset (Roboflow YOLO format)
* Fine-tuning YOLO26 for instance segmentation
* Automatic saving of checkpoints, confusion matrices, and metrics
* Fully reproducible execution via Docker with GPU support

---

## Requirements

* Docker with NVIDIA GPU support
* NVIDIA Container Toolkit installed
* Dataset (in YOLO format) available at:

```
./datasets/<dataset_name>

```

---

## Expected Project Structure

```
sandbox_sam3/
│
├── logs/               # Training outputs
├── dataset/            # Dataset
├── yolo26_seg/         # Model source code
└── utils/              # Useful scripts
```

---

## Environment Setup

### Build the Docker Image

Build the environment containing CUDA, PyTorch, and all necessary dependencies:

```bash
docker build -t yolo26_ft .
```

---

## Training Execution

### Option A: Run training using ALL available GPUs

```bash
docker run --gpus all -it --rm \
  --ipc=host \
  --user $(id -u):$(id -g) \
  -e TORCH_HOME=/workspace/cache/torch \
  -e HOME=/workspace/cache \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v $(pwd)/datasets:/workspace/datasets \
  -v $(pwd)/logs:/workspace/logs \
  -v $(pwd)/yolo26_seg:/workspace/yolo26_seg \
  -v $(pwd)/utils:/workspace/utils \
  -v $(pwd)/cache:/workspace/cache \
  -v /etc/passwd:/etc/passwd:ro \
  -v /etc/group:/etc/group:ro \
  yolo26_ft \
  python /workspace/yolo26_seg/train.py 2>&1 | tee logs/yolo26_ft_ph2.log

```

### Option B: Run training using a SINGLE GPU

```bash
docker run --gpus '"device=0"' -it --rm \
  --ipc=host \
  --user $(id -u):$(id -g) \
  -e TORCH_HOME=/workspace/cache/torch \
  -e HOME=/workspace/cache \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v $(pwd)/datasets:/workspace/datasets \
  -v $(pwd)/logs:/workspace/logs \
  -v $(pwd)/yolo26_seg:/workspace/yolo26_seg \
  -v $(pwd)/utils:/workspace/utils \
  -v $(pwd)/cache:/workspace/cache \
  -v /etc/passwd:/etc/passwd:ro \
  -v /etc/group:/etc/group:ro \
  yolo26_ft \
  python /workspace/yolo26_seg/train.py 2>&1 | tee logs/yolo26_ft_ph2_gpu0.log

```

### Option C: Automated Training (Wait for Free GPUs)

```bash
chmod +x wait_gpu.sh
```

```bash
./wait_gpu.sh
```

---

## Running on a Remote Server

### Run training in the background

Create a screen session:

```bash
screen -S yolo26_ft

```

Run the Docker command normally. Detach while keeping the process running:

```text
Ctrl + A, then D

```

Reattach later:

```bash
screen -r yolo26_ft

```

---

### Copy results from the server

```bash
rsync -avz --progress -e "ssh -p 13508 -v" antoniovinicius@164.41.75.221:/home/antoniovinicius/projects/SANDBOX_YOLO26/logs/ph2_finetuning /home/avmoura_linux/Documents/unb/SANDBOX_YOLO26

```

---

### Environment Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install ultralytics jupyterlab
```

---

### Hardware Monitoring

```bash
nvidia-smi
nvtop
```

---

## End-to-End Pipeline (`run_pipeline.sh`)

The master orchestrator `run_pipeline.sh` chains the four phases of the
ISIC 2018 Task 1 study for all five YOLO26-seg sizes (n, s, m, l, x):

1. **Phase 1 — Baseline:** Ultralytics defaults, `epochs=120`,
   `patience=20`, `deterministic=True`, `seed=0`, `amp=False`
   (single explicit deviation from the Ultralytics default — see
   [reproducibility note](#a-note-on-amp-mixed-precision) below).
   Implemented by `yolo26_seg/train_baseline_models.py`.
2. **Phase 2 — HPO:** per-model `model.tune()` using the refined search
   space from session `21d1...`. Implemented by
   `yolo26_seg/tune_all_models_v2.py`. Outputs are written to
   `<project>/hpo/hpo_v3/tune_isic_2018_task_1_<model>/best_hyperparameters.yaml`,
   which is the path expected by Phases 3 and 4.
3. **Phase 3 — Optimized single-split:** full fine-tuning using the
   `best_hyperparameters.yaml` from Phase 2 (120 epochs, `MuSGD`,
   cosine LR, `patience=25`). Implemented by
   `yolo26_seg/train_all_models.py`.
4. **Phase 4 — 5-Fold Cross-Validation:** deterministic NumPy-only KFold
   (`seed=0`, K=5) over the train+val pool of the original
   `data.yaml` (test split untouched). Implemented by
   `yolo26_seg/train_all_models_cv.py`. Final consolidation
   (mean ± std for mAP@50, mAP@50-95, Precision, Recall, F1-Score —
   Box and Mask) is produced by
   `yolo26_seg/consolidate_cv_results.py`.

Single-split summaries (Phases 1 and 3) are extracted from `results.csv`
by `yolo26_seg/collect_phase_metrics.py`.

### Recommended Logs Layout

Every artefact of a pipeline run is isolated under
`logs/<PIPELINE_NAME>/` (default `logs/pipeline_e2e_v1/`) so it does **not**
mix with the previous standalone fine-tunings that live directly under
`logs/`. Change `PIPELINE_NAME` (env var or `--pipeline-name`) to start a
fresh run alongside the previous one.

```
logs/
├── pipeline_e2e_v1/                                 # ← PIPELINE_NAME
│   ├── phase1_baseline/
│   │   └── yolo26_<model>_baseline/{weights/best.pt, results.csv, args.yaml, ...}
│   ├── hpo/
│   │   └── hpo_v3/tune_isic_2018_task_1_<model>/best_hyperparameters.yaml
│   ├── yolo26_<model>_ft_isic_2018_v11/{weights/best.pt, results.csv, ...}    # Phase 3
│   ├── cv/
│   │   └── cv_v1/yolo26_<model>_cv_isic_2018/{splits/, runs/,
│   │       metrics_per_fold.csv, metrics_summary.json}
│   ├── pipeline_summary/
│   │   ├── baseline_metrics.{csv,json}
│   │   ├── optimized_metrics.{csv,json}
│   │   └── cv_consolidated.{csv,json}
│   └── pipeline_runs/<UTC-timestamp>/{pipeline.log, phase1.log, phase2.log, ...}
│
├── pipeline_e2e_v2/                                 # ← a future re-run
│   └── ...
└── <legacy standalone runs, untouched>              # e.g. yolo26_small_ft_isic_2018_v11/
```

The nested layout is automatic: the orchestrator passes
`--project /workspace/logs/<PIPELINE_NAME>` to every Python helper, so
all sub-folders (`phase1_baseline/`, `hpo/hpo_v3/`,
`yolo26_<model>_ft_isic_2018_v11/`, `cv/cv_v1/`, `pipeline_summary/`,
`pipeline_runs/`) end up inside the same isolated parent.

### A note on AMP (mixed precision)

All four phases run with **`amp=False`** (FP32 throughout). This is a
deliberate, **uniform** deviation from the Ultralytics default
(`amp=True`) for Phases 1 and 2 — Phases 3 and 4 were already FP32 in
the upstream scripts.

Rationale (defensible for publication):

* During an earlier run on the GPU host, the **xlarge** variant produced
  `NaN` in the classification loss at epoch 42 with `amp=True` —
  consistent with an FP16 overflow in the cls-head that the AMP gradient
  scaler did not catch. The Ultralytics auto-recovery then failed because
  the saved `last.pt` checkpoint was already post-overflow.
* Rather than disable AMP only for the failing variant (an asymmetric
  fix that would weaken the cross-architecture comparison), we disable
  AMP for **all five sizes** so that every model is trained under
  numerically identical conditions.
* Side benefit: FP32 results are invariant across Tensor Core
  generations, strengthening hardware-independent reproducibility.

Trade-off: ~30–40 % more GPU time vs. AMP across Phases 1 and 2; VRAM
footprint roughly doubles (xlarge observed at ~10.8 GB with AMP; expect
~14–16 GB FP32 — verify your GPU has headroom or reduce `batch`).

Pass `--amp` to `train_baseline_models.py` to re-enable mixed precision
for a specific Phase 1 run (e.g. when reproducing the original
Ultralytics-default behaviour for an ablation).

### A note on batch size (Phase 2 HPO on xlarge)

Phases 1, 3 and 4 train every variant with `batch=16` (`nbs=64`,
gradient accumulation 4 → **effective optimisation batch = 64**).
Phase 2 HPO uses `batch=32` (`nbs=64`, gradient accumulation 2 →
**also effective optimisation batch = 64**) for the nano, small,
medium and large variants.

For the **xlarge** variant, FP32 + DDP + `batch=32` does not fit in
the 32 GB V100S (observed: 31.59 GB allocated per rank before the
1st training iteration → `torch.OutOfMemoryError`). HPO trials for
xlarge therefore use `batch=16` (`nbs=64` unchanged → accumulation 4),
matching the protocol already used in Phases 1, 3 and 4 for xlarge.

The orchestrator exposes this as `--hpo-batch` (CLI) / `HPO_BATCH`
(env, default `32`). Example invocations:

```bash
# Default — applies to nano/small/medium/large
bash run_pipeline.sh --phases "2" --models "nano small medium large"

# xlarge — 32 GB GPUs require batch=16 under FP32+DDP
bash run_pipeline.sh --phases "2 3 4" --models "xlarge" --hpo-batch 16
# or via env:
HPO_BATCH=16 bash run_pipeline.sh --phases "2 3 4" --models "xlarge"
```

**Why this preserves comparability between models:**

* Ultralytics computes `accumulate = round(nbs/batch)` and accumulates
  gradients across that many micro-batches before each optimiser
  step. With `nbs=64` held constant, the **effective optimisation
  batch is identical (64) for every variant and every phase**,
  regardless of whether `batch=32` or `batch=16` is used at the
  micro-step level.
* The hyperparameter space being searched (`lr0`, `lrf`, `momentum`,
  `weight_decay`, `cls`, `dfl`, augmentation strengths) operates on
  the optimiser step — so the HPs discovered for xlarge are directly
  comparable to those found for the other four variants at the same
  effective batch size.
* The only difference is per-step VRAM footprint and the number of
  micro-batches forwarded between optimiser updates — neither of
  which affects the gradient step magnitude or the loss-curve shape
  at fixed effective batch.

Suggested wording for the Methods section of a paper:

> *Due to VRAM constraints on the V100S (32 GB), the xlarge variant
> uses `batch=16` during HPO (vs. `batch=32` for the four smaller
> variants); for the optimised fine-tuning and the 5-fold
> cross-validation, all variants use `batch=16`. The nominal batch
> size (`nbs=64`) is held constant across all phases and variants,
> so the effective optimisation batch size is identical (64) across
> the entire study. This affects only the micro-batch composition
> and per-step memory footprint, not the gradient-step magnitude.*

### Pipeline hardening: GPU sanity-check and HPO validity check

Two safety checks added to the orchestrator after an earlier run hit
infrastructure-level failures invisible to the trainer:

1. **`gpu_sanity_check`** runs at the start of every phase (~1 s):
   verifies that `nvidia-smi -L` reports GPUs **and** that
   `torch.cuda.is_available()` returns `True`. If the driver died on
   the host (e.g. an `nvidia.ko`/NVML hang after a long FP32 run),
   the script aborts immediately with an actionable message instead
   of wasting minutes inside the trainer.
2. **`check_hpo_validity.py`** runs after Phase 2 and inspects every
   `tune_results.csv` produced by Ultralytics' Tuner. If any model
   has fewer than `--min-trials` rows with `fitness>0`, the pipeline
   fails with a non-zero exit code. This catches **degenerate HPO**
   runs — when the Tuner records `fitness=0` for failed trials and
   silently emits a `best_hyperparameters.yaml` that is just the
   seed vector (which we observed at scale after a driver crash).

### Idempotency

Each Python script already detects existing artefacts:

* `train_baseline_models.py` skips models with an existing `best.pt`.
* `tune_all_models_v2.py` skips models with an existing
  `best_hyperparameters.yaml`.
* `train_all_models.py` skips models with an existing `best.pt`.
* `train_all_models_cv.py` skips folds and models with an existing
  `results.csv` / `metrics_summary.json`.

Pass `--force` to the orchestrator (or set `FORCE_FLAG=--force`) to
re-execute every phase.

### Running the pipeline inside Docker

Build the image once:

```bash
docker build -t yolo26_ft .
```

Then run the orchestrator with explicit GPU allocation
(`{GPU_DEVICE_IDS}` is a placeholder — substitute with e.g. `"0,1"`):

```bash
GPU_DEVICE_IDS="0,1"
PIPELINE_NAME="pipeline_e2e_v1"   # rename for each new isolated run

docker run --gpus "\"device=${GPU_DEVICE_IDS}\"" -it --rm \
    --ipc=host \
    --user "$(id -u):$(id -g)" \
    -e TORCH_HOME=/workspace/cache/torch \
    -e HOME=/workspace/cache \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e GPU_DEVICE_IDS="${GPU_DEVICE_IDS}" \
    -e PIPELINE_NAME="${PIPELINE_NAME}" \
    -v "$(pwd)/datasets:/workspace/datasets" \
    -v "$(pwd)/logs:/workspace/logs" \
    -v "$(pwd)/yolo26_seg:/workspace/yolo26_seg" \
    -v "$(pwd)/utils:/workspace/utils" \
    -v "$(pwd)/cache:/workspace/cache" \
    -v "$(pwd)/run_pipeline.sh:/workspace/run_pipeline.sh:ro" \
    -v /etc/passwd:/etc/passwd:ro \
    -v /etc/group:/etc/group:ro \
    yolo26_ft \
    bash /workspace/run_pipeline.sh \
    2>&1 | tee "logs/${PIPELINE_NAME}_$(date -u +%Y%m%dT%H%M%SZ).log"
```

The Docker bind-mount `-v "$(pwd)/logs:/workspace/logs"` is the **parent
`LOGS_ROOT`** — every pipeline run will write into a sub-directory of
that mount (`logs/${PIPELINE_NAME}/...`), so previous standalone HPO/CV/FT
artefacts already in `logs/` stay untouched.

Selecting a subset of phases or models:

```bash
# Only run HPO + Optimized + CV (skip Phase 1 baseline):
bash /workspace/run_pipeline.sh --phases "2 3 4"

# Only nano and small, all four phases:
bash /workspace/run_pipeline.sh --models "n s"

# Dry-run (prints commands only):
bash /workspace/run_pipeline.sh --dry-run

# Run an ablation under a separate folder so it doesn't touch the previous one:
bash /workspace/run_pipeline.sh --pipeline-name pipeline_e2e_v2 --force

# Use a completely custom project path (overrides PIPELINE_NAME):
bash /workspace/run_pipeline.sh --project /workspace/logs/my_experiment
```

Useful environment overrides (defaults shown):

| Variable | Default | Description |
|---|---|---|
| `DATA_YAML` | `/workspace/datasets/isic_2018_task1_yolo26/data.yaml` | Dataset YAML (Ultralytics/Roboflow format). |
| `LOGS_ROOT` | `/workspace/logs` | Parent dir for every pipeline run. |
| `PIPELINE_NAME` | `pipeline_e2e_v1` | Sub-dir under `LOGS_ROOT` that isolates THIS run from previous standalone fine-tunings (HPO/CV/FT) already living directly under `LOGS_ROOT`. |
| `PROJECT` | `${LOGS_ROOT}/${PIPELINE_NAME}` | Final project root passed to every Python helper. Override (env or `--project`) to point anywhere else. |
| `GPU_DEVICE_IDS` | `0,1` | Comma-separated GPU IDs (DDP). |
| `P1_EPOCHS` / `P1_PATIENCE` | `120` / `20` | Baseline (Phase 1). |
| `HPO_SPACE` / `HPO_ITERATIONS` / `HPO_EPOCHS_PER_TRIAL` / `HPO_PATIENCE` | `refined` / `30` / `30` / `10` | HPO (Phase 2). |
| `CV_K_FOLDS` / `CV_SEED` / `CV_EPOCHS` / `CV_PATIENCE` | `5` / `0` / `120` / `25` | Cross-Validation (Phase 4). |

---