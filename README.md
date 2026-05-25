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

1. **Phase 1 — Baseline:** strict Ultralytics defaults, `epochs=120`,
   `patience=20`, `deterministic=True`, `seed=0`.
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