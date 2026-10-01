# YOLO26 Fine-Tuning — Skin Lesion Segmentation (ISIC 2018 Task 1)

This repository contains the experimental pipeline used to evaluate **Ultralytics YOLO26-seg** (sizes n, s, m, l, x)
for skin-lesion segmentation on ISIC 2018 Task 1. The study measures the *raw* architecture first and only then the
effect of hyperparameter optimisation (HPO), and it reports both accuracy and the hardware efficiency needed to argue
that the models are lightweight enough for deployment.

https://www.ultralytics.com/blog/how-to-custom-train-ultralytics-yolo26-for-instance-segmentation

---

## The 5-phase protocol

| Phase | What | Data | Script(s) |
|---|---|---|---|
| **1 — Baseline training** | Fixed base setup + Ultralytics **default** hyperparameters | train / val | `train_baseline_models.py` |
| **2 — Baseline cross-validation** | Deterministic 5-fold CV with the Phase 1 configuration; DSC/JSI of each fold on its held-out fold | train ∪ val pool (**test excluded and verified**) | `train_all_models_cv.py`, `consolidate_cv_results.py`, `evaluate_cv_pixels.py` |
| **3 — HPO** | Ultralytics genetic tuner with a **seeded** mutation RNG; **fault-tolerant and resumable** | train / val | `tune_all_models_v2.py`, `check_hpo_validity.py` |
| **4 — Optimised fine-tuning** | Same fixed base setup + Phase 3 **tuned** hyperparameters | train / val | `train_all_models.py` |
| **5 — Test set** | Baseline **and** Optimised on the unseen test set: instance metrics (mAP/P/R/F1), pixel metrics (DSC, JSI, ISIC thresholded JSI), batch-1 efficiency in **FP32 and FP16**, final report | **test** (only here) | `evaluate_test_set.py`, `benchmark_efficiency.py`, `build_final_report.py` |

Everything is orchestrated by **`run_pipeline.sh`**; all shared settings live in **`yolo26_seg/common.py`**.

### One base setup for every phase (Baseline vs. Optimised = default vs. tuned hyperparameters)

Every training run — Baseline (Phases 1–2), every HPO trial (Phase 3) and Optimised (Phase 4) — uses the same fixed
**base setup**. **Baseline = base setup + Ultralytics default hyperparameters; Optimised = base setup + tuned
hyperparameters**, so the tuned values (learning rate, momentum, weight decay, warm-up, loss gains, augmentation)
are the **only** variable between them. The first HPO trial starts from the default hyperparameters clipped to the
search bounds; note that in the `refined` space the defaults `lr0=0.01` and `weight_decay=5e-4` lie outside the bounds,
so the Baseline's own values are not candidates of the search. Deviations of the base setup from the Ultralytics defaults:

| Setting | Value | Ultralytics default | Why |
|---|---|---|---|
| `epochs` / `patience` | 120 / 25 | 100 / 100 | Same budget for every training phase (HPO trials: 30 / 10). |
| `amp` | `False` (FP32) | `True` | The xlarge variant overflowed in FP16 (NaN in the cls-loss); FP32 everywhere keeps numerical conditions uniform across sizes. |
| `optimizer` | `MuSGD` | `auto` | `auto` picks AdamW or MuSGD from the number of iterations and then ignores `lr0`/`momentum`; an explicit optimiser makes both the default and the tuned `lr0`/`momentum` effective. |
| `cos_lr` | `True` | `False` | One learning-rate schedule for every phase. |

Pinned values equal to the defaults (pinned so an upstream change cannot alter the protocol): `batch=16`, `nbs=64`,
`imgsz=640`, `workers=8`, `close_mosaic=10`, `seed=0`, `deterministic=True`. A tuned-hyperparameter file or a search
space that tries to change any base-setup key is rejected.

**Best epoch.** Every validation / cross-validation metric taken from a training log is read at the epoch that produced
`best.pt` — identified from the metrics stored in the checkpoint itself, falling back to the Ultralytics fitness
(box mAP50-95 + mask mAP50-95, latest epoch on ties). Reported validation metrics therefore always describe the
exact checkpoint that is evaluated in Phase 5.

### Reproducibility

* `torch.manual_seed`, `np.random.seed`, `random.seed`, `PYTHONHASHSEED`, deterministic cuDNN/cuBLAS — see
  `common.seed_everything()`.
* **K-Fold** without scikit-learn: `numpy.random.RandomState(0)` shuffle + contiguous folds (bit-identical to
  `KFold(shuffle=True, random_state=0)`). `splits_manifest.json` fingerprints the folds; a re-run with different splits is refused.
* **HPO:** the upstream Tuner re-seeds its mutations from the clock. `SeededTuner` uses
  `np.random.default_rng([seed, trial])`, so the hyperparameters of trial *i* depend only on the seed, *i* and the
  fitness history. Caveat for the thesis: proposals are bit-reproducible given identical fitness values; fitness comes
  from GPU training, which is deterministic only up to `deterministic=True` (`warn_only`) and DDP reduction order.
* **Pinned environment** (`Dockerfile`): `torch==2.5.1`, `torchvision==0.20.1`, `torchaudio==2.5.1` (`+cu121`),
  `ultralytics==8.4.21`, `pandas==3.0.1`. The HPO checkpoint refuses to resume under another Ultralytics version.
* **Statistics:** CV mean ± **sample** SD (ddof = 1) over folds; test metrics as per-image mean with a seeded
  bootstrap 95 % CI; HPO gain as a paired difference with bootstrap CI and Wilcoxon signed-rank test.

### Fault tolerance

Every step is idempotent and resumable — **re-running the same command continues where it stopped**:

* **Training runs** (Phases 1, 2, 4) record completion in `run_state.json`; an interrupted run resumes from `last.pt`
  (the resume is logged, because the dataloader RNG state is not checkpointed, so a resumed run is not bit-identical).
  Duplicate epoch rows that Ultralytics writes after a resume are de-duplicated before picking the best epoch.
* **HPO** (Phase 3) checkpoints `hpo_state.json` atomically before every trial. On resume it removes the interrupted
  trial's folder and a torn CSV line; a failed trial (fitness 0) is retried with identical hyperparameters up to
  `--max-trial-retries` times. Changing the search space/protocol/seed refuses to resume.
* **GPU/driver failures:** `tune_all_models_v2.py` exits with **75** when the GPU is unhealthy; `run_pipeline.sh`
  then waits `HPO_RETRY_WAIT` seconds and retries up to `HPO_MAX_RETRIES` times (resuming each time). A GPU sanity
  check runs before every phase.
* **Locks** prevent two processes from writing the same run or HPO search.
* `--force` starts a phase over **without deleting anything**: previous outputs are moved to `*.bak-<UTC>`.
* Phase 5 results are cached by the SHA-256 of the weights, the test list and the measurement settings
  (including a method version), so they are recomputed exactly when something relevant changed.

---

## Requirements

* Docker with NVIDIA GPU support (NVIDIA Container Toolkit)
* Dataset in YOLO segmentation format with `train`, `val` **and** `test` splits:

```
./datasets/isic_2018_task1_yolo26/data.yaml
```

## Project structure

```
sandbox_yolo26/
├── run_pipeline.sh            # 5-phase orchestrator
├── wait_gpu.sh                # optional: start a run once the GPUs are idle
├── Dockerfile                 # pinned environment
├── yolo26_seg/
│   ├── common.py              # protocols, seeds, paths, shared helpers
│   ├── training.py            # resumable training runs + results.csv parsing
│   ├── segmentation_metrics.py# DSC / JSI / ... (pixel level)
│   ├── train_baseline_models.py        # Phase 1
│   ├── train_all_models_cv.py          # Phase 2
│   ├── consolidate_cv_results.py       # Phase 2 summary
│   ├── evaluate_cv_pixels.py           # Phase 2 DSC/JSI per fold
│   ├── tune_all_models_v2.py           # Phase 3
│   ├── check_hpo_validity.py           # Phase 3 validation
│   ├── train_all_models.py             # Phase 4
│   ├── collect_phase_metrics.py        # Phase 1/4 summaries
│   ├── evaluate_test_set.py            # Phase 5a
│   ├── benchmark_efficiency.py         # Phase 5b
│   ├── build_final_report.py           # Phase 5c
│   └── legacy/                         # superseded scripts (kept for old notebooks)
├── notebooks/
│   ├── 01_Segmentation_Visualizer.ipynb
│   └── 02_Metrics_and_Efficiency_Analysis.ipynb
├── utils/                     # earlier analysis notebooks and helper scripts
├── datasets/  logs/  cache/   # data, outputs, weights (not versioned)
```

## Output layout

All outputs of the study live under `logs/pipeline_final_v1/` (change with `--pipeline-name`), isolated from older runs:

```
logs/pipeline_final_v1/
├── phase1_baseline/yolo26_<m>_baseline/          weights/, results.csv, run_state.json
├── phase2_cv_baseline/yolo26_<m>/                splits/, splits_manifest.json, runs/fold_<k>/,
│                                                 metrics_per_fold.csv, metrics_summary.json,
│                                                 pixel_metrics_per_fold.csv, pixel_metrics_summary.json
├── phase3_hpo/tune_<m>/                          tune_results.csv, best_hyperparameters.yaml,
│                                                 hpo_state.json, trials/
├── phase4_optimized/yolo26_<m>_optimized/        weights/, results.csv, run_state.json, tuned_hyperparameters.yaml
├── phase5_test/
│   ├── accuracy/<variant>_<m>_<fp32|fp16>.json   instance + pixel metrics, weights SHA-256
│   ├── per_image/<variant>_<m>_<fp32|fp16>.csv   per-image DSC, JSI, TP/FP/FN/TN, ...
│   ├── masks/<variant>_<m>/<image>.png           predicted masks (FP32) for notebook 01
│   ├── efficiency/<variant>_<m>_<fp32|fp16>.json latency, FPS, VRAM, RAM, size, params, GFLOPs
│   └── val_runs/                                 Ultralytics test-set plots
├── summary/                                      phase1_val, phase2_cv_baseline, phase2_cv_pixel, phase4_val,
│                                                 test_accuracy, efficiency, hpo_gain, final_results (.csv/.json)
├── figures/  tables/                             written by the notebooks (PDF/PNG, LaTeX)
└── pipeline_runs/<UTC>/                          pipeline.log + one log per step
```

---

## Running the pipeline

### Build the image

```bash
docker build -t yolo26_ft .
```

### Full run (all 5 phases, all sizes)

```bash
GPU_DEVICE_IDS="0,1"
PIPELINE_NAME="pipeline_final_v1"

docker run --gpus all -it --rm \
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

`--gpus all` exposes every GPU to the container; `GPU_DEVICE_IDS` selects the ones used for (DDP) training.
Phase 5 runs on a single GPU (`--bench-device`, default: the first training GPU).

If the run is interrupted (crash, driver failure, reboot), **run the same command again** — it resumes.

### Common variations (arguments after `bash /workspace/run_pipeline.sh`)

```bash
--phases "3 4 5"                 # a subset of phases
--models "n s"                   # a subset of sizes (n,s,m,l,x or full names)
--models x --phases 3 --hpo-batch 16   # xlarge HPO in FP32 on 32 GB GPUs (see below)
--dry-run                        # print the commands only
--epochs 3 --patience 2          # SMOKE TEST ONLY (applied to Phases 1, 2 and 4 together)
--pipeline-name pipeline_final_v2      # a fresh, isolated study
--force                          # start the selected phases over (old outputs → *.bak-<UTC>)
```

Environment overrides (defaults): `CV_K_FOLDS=5`, `CV_SEED=0`, `HPO_SPACE=refined`, `HPO_ITERATIONS=30`,
`HPO_EPOCHS_PER_TRIAL=30`, `HPO_PATIENCE=10`, `HPO_BATCH=32`, `HPO_MAX_RETRIES=5`, `HPO_RETRY_WAIT=600`,
`EVAL_PRECISIONS="fp32 fp16"`, `BENCH_DEVICE`, `DATA_YAML`, `LOGS_ROOT`, `PROJECT`.

Exit codes of `run_pipeline.sh`: `0` success · `75` the HPO gave up after repeated GPU failures (fix the driver and
re-run to resume) · any other value is the exit code of the failing step (its log is in `pipeline_runs/<UTC>/`).

### Phase 5 — what exactly is measured

**Accuracy (`evaluate_test_set.py`)** — run once per variant × size × precision on the `test` split only:

* Instance metrics from `model.val(split="test", batch=1)` (P, R, mAP50, mAP50-95, F1; Box and Mask).
* Pixel metrics per image: ground truth = union of the YOLO label polygons rasterised at full resolution;
  prediction = union of instance masks with `conf ≥ 0.25` at full resolution (`retina_masks=True`).
  `DSC = 2TP/(2TP+FP+FN)`, `JSI = TP/(TP+FP+FN)`, ISIC thresholded JSI (`JSI < 0.65 → 0`), sensitivity,
  specificity, accuracy. **An empty prediction scores 0** (never skipped); empty GT and empty prediction scores 1.
* FP32 is the primary result (training was FP32); FP16 quantifies the accuracy cost of half precision.

**Efficiency (`benchmark_efficiency.py`)** — `batch = 1`, one GPU, every configuration in a fresh process:

* **Forward latency**: fused network on a fixed 1×3×640×640 input, timed with `torch.cuda.Event` + synchronise
  per iteration (50 warm-up + 500 timed). **End-to-end latency**: `YOLO.predict()` on a real test image
  (pre-processing, inference, mask post-processing), timed with `perf_counter` (20 + 200).
* Reported: mean, SD, median, **P90/P95/P99**, min/max, **FPS** = 1000 / mean (and 1000 / median).
* **VRAM**: steady-state peak allocated/reserved by PyTorch after warm-up (the warm-up peak, which includes
  cuDNN autotuning workspaces, is stored separately; the CUDA context is excluded), and VRAM of the weights alone.
  **RAM**: RSS after loading/benchmarking and peak RSS.
* **Size**: `best.pt` on disk (stored in **FP16** by Ultralytics), theoretical FP32/FP16 sizes (fused params × 4/2 B),
  parameter count (fused and unfused), GFLOPs at 640.
* GPU utilisation is sampled before/after each run; a busy GPU marks the result `contended` (re-run on an idle GPU
  before quoting those latencies). Baseline and Optimised share the architecture, so their efficiency should match.

### Notes on AMP and batch size

* **AMP** is disabled in every phase (see the table above). Trade-off: ~30–40 % more GPU time and roughly twice the
  activation memory compared with AMP.
* **HPO batch.** Phases 1, 2 and 4 use `batch=16`; Phase 3 trials use `batch=32` by default to speed up the search.
  With `nbs=64` held constant, Ultralytics accumulates gradients (`accumulate = round(nbs / batch)`), so the
  **effective optimisation batch is 64 in every phase and for every size**. For **xlarge** in FP32 + DDP, `batch=32`
  does not fit in a 32 GB V100S, so use `--hpo-batch 16` (accumulation 4, still 64 effective).

  Suggested wording: *"The nominal batch size (nbs = 64) is held constant across all phases and model sizes, so the
  effective optimisation batch size is identical (64) throughout the study; only the micro-batch (16 or 32) and the
  per-step memory footprint differ."*

---

## Analysis notebooks

| Notebook | Content |
|---|---|
| `notebooks/01_Segmentation_Visualizer.ipynb` | Test images with ground truth (green, solid border) and prediction (red, dashed border) for Baseline vs. Optimised; random sample, largest HPO gains/regressions, hardest cases. |
| `notebooks/02_Metrics_and_Efficiency_Analysis.ipynb` | DSC/JSI across phases, paired HPO gain with *p*-values, accuracy vs. size/GFLOPs, latency vs. FPS, latency distribution (median/P95), VRAM/RAM, accuracy–latency trade-off, LaTeX tables. |

They read only the files written by the pipeline (no GPU needed). The pipeline folder is found automatically
(`$PIPELINE_DIR`, else `/workspace/logs/<name>`, else `logs/<name>`); figures go to `<pipeline>/figures/`
(PDF + PNG), tables to `<pipeline>/tables/`.

Run Jupyter inside the container:

```bash
docker run --gpus all -it --rm -p 8888:8888 \
    --user "$(id -u):$(id -g)" \
    -e HOME=/workspace/cache \
    -v "$(pwd)/datasets:/workspace/datasets" \
    -v "$(pwd)/logs:/workspace/logs" \
    -v "$(pwd)/yolo26_seg:/workspace/yolo26_seg" \
    -v "$(pwd)/notebooks:/workspace/notebooks" \
    -v "$(pwd)/cache:/workspace/cache" \
    yolo26_ft \
    jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --notebook-dir=/workspace
```

or on the host (paths recorded as `/workspace/...` are mapped back to the checkout automatically):

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install ultralytics==8.4.21 pandas==3.0.1 jupyterlab
jupyter lab notebooks/
```

---

## Running on a remote server

Run the pipeline inside a `screen` session so it survives a disconnect:

```bash
screen -S yolo26_ft        # start; run the docker command above
# Ctrl + A, then D         # detach
screen -r yolo26_ft        # reattach
```

Wait for idle GPUs before starting (edit the `docker run` blocks at the bottom of the script first):

```bash
chmod +x wait_gpu.sh && ./wait_gpu.sh
```

Copy the results to your machine:

```bash
rsync -avz --progress -e "ssh -p 13508" \
    antoniovinicius@164.41.75.221:/home/antoniovinicius/projects/sandbox_yolo26/logs/pipeline_final_v1 \
    /home/avmoura_linux/Documents/unb/SANDBOX_YOLO26/logs/
```

Hardware monitoring: `nvidia-smi`, `nvtop`.
