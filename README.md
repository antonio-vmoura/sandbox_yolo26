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
| **0 — Dataset** | YOLO-seg dataset built from the **raw official ISIC 2018 Task 1 release** (`../datasets/ISIC2018_Raw`): exactly **2,594 / 100 / 1,000** images (asserted); long side ≤ 1,024 px (aspect preserved); official masks kept in `masks/` as the evaluation ground truth; YOLO polygons (holes bridged) for training, fidelity-checked | raw release | `prepare_dataset.py` |
| **1 — Baseline training** | Fixed base setup + Ultralytics **default** hyperparameters | train / val | `train_baseline_models.py` |
| **2 — Baseline cross-validation** | Deterministic 5-fold CV with the Phase 1 configuration; DSC/JSI of each fold on its held-out fold | train ∪ val pool (**test excluded and verified**) | `train_all_models_cv.py`, `consolidate_cv_results.py`, `evaluate_cv_pixels.py` |
| **3 — HPO** | Ultralytics genetic tuner with a **seeded** mutation RNG; **fault-tolerant and resumable** | train / val | `tune_all_models_v2.py`, `check_hpo_validity.py` |
| **4 — Optimised fine-tuning** | Same fixed base setup + Phase 3 **tuned** hyperparameters | train / val | `train_all_models.py` |
| **5 — Test set** | Baseline **and** Optimised on the unseen test set: instance metrics (mAP/P/R/F1), pixel metrics (DSC, JSI, ISIC thresholded JSI) and boundary metrics (Boundary IoU, NSD, HD95) with bootstrap 95 % CI, batch-1 efficiency (median/P95 latency, FPS, peak VRAM) in **FP32 and FP16**, final report | **test** (only here) | `evaluate_test_set.py`, `benchmark_efficiency.py`, `build_final_report.py` |

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
| `epochs` / `patience` | 120 / 120 (no early stopping) | 100 / 100 | Same budget for every training phase (HPO trials: 30 / 30). Every run completes its cosine LR schedule and `close_mosaic`; with patience 25 the noisy 100-image validation split stopped all Phase 1 runs at epochs 50–82 at a still-high LR. |
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
* The raw official ISIC 2018 Task 1 release in `../datasets/ISIC2018_Raw` (training/validation/test inputs and
  ground truths); Phase 0 converts it to the YOLO segmentation dataset (`train`, `val` **and** `test` splits):

```
./datasets/isic2018_task1_official/data.yaml   # built by Phase 0 from ../datasets/ISIC2018_Raw
```

## Project structure

```
sandbox_yolo26/
├── run_pipeline.sh            # 5-phase orchestrator
├── wait_gpu.sh                # optional: start the pipeline once a GPU is idle
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
├── utils/legacy/              # earlier helper scripts and examples (kept as a backup)
├── figures/legacy/            # earlier qualitative figures
├── notebooks/legacy/          # earlier analysis notebooks (kept as a backup)
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

## ISIC 2018 Task 2 (lesion attributes) — Phase 0

Phase 0 also builds the **multi-label** dataset of ISIC 2018 Task 2 from the same raw folder (`ISIC2018_Raw`, which
holds the shared Task 1-2 input images and the ground truths of both tasks):

```bash
# inside yolo26_ft, raw release mounted at /workspace/raw (as wait_gpu.sh does)
python /workspace/yolo26_seg/prepare_dataset.py --task 2      # -> datasets/isic2018_task2_official
```

Five classes, id = attribute index (`pigment_network`, `negative_network`, `streaks`, `milia_like_cyst`,
`globules`): each attribute mask becomes polygons of its class; attributes may overlap (overlapping polygons of
different classes coexist) and an image may have none. The per-attribute official masks are kept in
`masks/<attribute>/<id>.png`, label fidelity is checked per attribute, and the same 2,594 / 100 / 1,000 images are
asserted. `--task 1` is the default everywhere (the orchestrator and `wait_gpu.sh` run Task 1); **Phases 1–5
currently implement Task 1 only** — Task 2 training/evaluation will need multi-class model selection and
per-attribute metrics.

## Running the pipeline

### Build the image

```bash
docker build -t yolo26_ft .
```

### Full run (all 5 phases, all sizes)

```bash
GPU_DEVICE_IDS="0,1"
PIPELINE_NAME="pipeline_final_v1"

mkdir -p "logs/${PIPELINE_NAME}"     # the terminal log goes inside the pipeline folder
docker run --gpus all -it --rm \
    --ipc=host \
    --user "$(id -u):$(id -g)" \
    -e TORCH_HOME=/workspace/cache/torch \
    -e HOME=/workspace/cache \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e GPU_DEVICE_IDS="${GPU_DEVICE_IDS}" \
    -e PIPELINE_NAME="${PIPELINE_NAME}" \
    -v "$(pwd)/datasets:/workspace/datasets" \
    -v "$(pwd)/../datasets/ISIC2018_Raw:/workspace/raw:ro" \
    -v "$(pwd)/logs:/workspace/logs" \
    -v "$(pwd)/yolo26_seg:/workspace/yolo26_seg" \
    -v "$(pwd)/utils:/workspace/utils" \
    -v "$(pwd)/cache:/workspace/cache" \
    -v "$(pwd)/run_pipeline.sh:/workspace/run_pipeline.sh:ro" \
    -v /etc/passwd:/etc/passwd:ro \
    -v /etc/group:/etc/group:ro \
    yolo26_ft \
    bash /workspace/run_pipeline.sh \
    2>&1 | tee "logs/${PIPELINE_NAME}/terminal_$(date -u +%Y%m%dT%H%M%SZ).log"
```

`--gpus all` exposes every GPU to the container; `GPU_DEVICE_IDS` selects the ones used for (DDP) training.
Phase 5 runs on a single GPU (`--bench-device`, default: the first training GPU).

If the run is interrupted (crash, driver failure, reboot), **run the same command again** — it resumes.

### Common variations (arguments after `bash /workspace/run_pipeline.sh`)

```bash
--phases "3 4 5"                 # a subset of phases
--models "n s"                   # a subset of sizes (n,s,m,l,x or full names)
--models s --phases 3 --hpo-batch 32   # override the per-model HPO micro-batch (see below)
--dry-run                        # print the commands only
--epochs 3 --patience 2          # SMOKE TEST ONLY (applied to Phases 1, 2 and 4 together)
--pipeline-name pipeline_final_v2      # a fresh, isolated study
--force                          # start the selected phases over (old outputs → *.bak-<UTC>)
```

Environment overrides (defaults): `CV_K_FOLDS=5`, `CV_SEED=0`, `HPO_SPACE=refined`, `HPO_ITERATIONS=30`,
`HPO_EPOCHS_PER_TRIAL=30`, `HPO_PATIENCE=30`, `HPO_BATCH=` (empty = per model: 16, xlarge 8), `HPO_MAX_RETRIES=5`, `HPO_RETRY_WAIT=600`,
`EVAL_PRECISIONS="fp32 fp16"`, `BENCH_DEVICE`, `DATA_YAML`, `LOGS_ROOT`, `PROJECT`.

Exit codes of `run_pipeline.sh`: `0` success · `75` the HPO gave up after repeated GPU failures (fix the driver and
re-run to resume) · any other value is the exit code of the failing step (its log is in `pipeline_runs/<UTC>/`).

### Phase 5 — what exactly is measured

**Accuracy (`evaluate_test_set.py`)** — run once per variant × size × precision on the `test` split only:

* Instance metrics from `model.val(split="test", batch=1)` (P, R, mAP50, mAP50-95, F1; Box and Mask).
* Pixel metrics per image: ground truth = the official ISIC mask (`masks/<id>.png`, Phase 0) at dataset resolution;
  prediction = union of instance masks with `conf ≥ 0.25` at full resolution (`retina_masks=True`).
  `DSC = 2TP/(2TP+FP+FN)`, `JSI = TP/(TP+FP+FN)`, ISIC thresholded JSI (`JSI < 0.65 → 0`), sensitivity,
  specificity, accuracy. **An empty prediction scores 0** (never skipped); empty GT and empty prediction scores 1.
* Boundary metrics (Metrics Reloaded): Boundary IoU (band 2 % of the image diagonal), NSD (tolerance 1 %) and
  HD95 (95th-percentile symmetric Hausdorff distance in px; a missed lesion scores the image diagonal).
* Aggregates: per-image mean, SD, median, IQR and a seeded bootstrap 95 % CI; per-image CSVs keyed by ISIC ID allow
  paired tests (Baseline vs. Optimised in `hpo_gain.csv`; across architectures in the root notebook).
* FP32 is the primary result (training was FP32); FP16 quantifies the accuracy cost of half precision.

**Efficiency (`benchmark_efficiency.py`)** — `batch = 1`, one GPU, every configuration in a fresh process:

* **Forward latency**: fused network on a fixed 1×3×640×640 input, timed with `torch.cuda.Event` + synchronise
  per iteration (50 warm-up + 500 timed). **End-to-end latency**: the deployed pipeline on a real test image —
  `YOLO.predict()` with the Phase 5a settings (`conf` 0.25, `retina_masks=True`) and the union mask copied to the
  host, i.e. exactly the mask that is scored (as for the U-Net and SAM 3) — timed with `perf_counter` (20 + 200).
* Reported: mean, SD, median, **P90/P95/P99**, min/max, **FPS** = 1000 / mean (and 1000 / median).
* **Driver-level VRAM**: `vram_process_peak_mb` = device memory held by the benchmark process at the end of the
  forward / end-to-end loops (CUDA context, kernels and allocator cache included; `nvidia-smi` delta) and
  `vram_cuda_context_mb` — the memory a deployment GPU must provide, next to the allocator peak (the model).
* **`end_to_end_dataset`**: the end-to-end pipeline once on each of the first 100 test images sorted by ISIC ID
  (the same images in the three repositories; `--e2e-images`), after one untimed pass — median/P95 over real,
  varying inputs. The real-time criterion in the notebooks uses its P95.
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
* **Micro-batch.** The protocol batch is 16. In FP32 at 640 px on a 32 GB V100S the peak training memory at batch 16
  is nano 5.4, small 10.7, medium 21.6 and large 24.7 GB; **xlarge does not fit** (~40 GB) and Ultralytics silently
  halves its batch to 8 after the out-of-memory error in the first epoch. Each run therefore records the batch it
  really used (`batch_effective`, read from `best.pt`; a mismatch is a warning in the summary and the final report),
  and Phase 3 trials use the same per-model micro-batch (`common.MICRO_BATCH`: 16, xlarge 8) instead of a batch that
  would silently be halved. With `nbs=64` held constant, Ultralytics accumulates gradients
  (`accumulate = round(nbs / batch)`), so the **effective optimisation batch is 64 in every phase and for every
  size**; only the BatchNorm batch statistics differ for xlarge.

  Suggested wording: *"The nominal batch size (nbs = 64) is held constant across all phases and model sizes, so the
  effective optimisation batch size is identical (64) throughout the study; the micro-batch is 16, except for the
  largest variant (8), which does not fit at 16 in FP32 on a 32 GB GPU."*

---

## Analysis notebooks

| Notebook | Content |
|---|---|
| `notebooks/01_Segmentation_Visualizer.ipynb` | Test images with ground truth (green, solid border) and prediction (red, dashed border) for Baseline vs. Optimised; random sample, largest HPO gains/regressions, hardest cases. |
| `notebooks/02_Metrics_and_Efficiency_Analysis.ipynb` | DSC/JSI across phases, paired HPO gain with *p*-values, accuracy vs. size/GFLOPs, latency vs. FPS, latency distribution (median/P95), VRAM/RAM, accuracy–latency trade-off, LaTeX tables; standard figures A–C (identical in the three repositories): accuracy vs. latency/FPS/parameters, training and inference time with the real-time criterion, boundary metrics. |
| `analysis/results_analysis.ipynb` + `analysis/article_aggregator.py` | Cross-architecture tables (LaTeX), paired tests (Wilcoxon, Holm, Friedman) and figures over YOLO26, U-Net and SAM 3 — byte-identical copies in the three repositories; place them in the folder holding the three repositories (see `analysis/README.md`). |

They read only the files written by the pipeline (no GPU needed). The pipeline folder is found automatically
(`$PIPELINE_DIR`, else `/workspace/logs/<name>`, else `logs/<name>`); figures go to `<pipeline>/figures/`
(PDF + PNG), tables to `<pipeline>/tables/`.

Run Jupyter inside the container:

```bash
docker run --gpus all -it --rm -p 8888:8888 \
    --user "$(id -u):$(id -g)" \
    -e HOME=/workspace/cache \
    -v "$(pwd)/datasets:/workspace/datasets" \
    -v "$(pwd)/../datasets/ISIC2018_Raw:/workspace/raw:ro" \
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

Wait for an idle GPU, then launch the pipeline on it (extra arguments go to `run_pipeline.sh`; the terminal log is
written to `logs/<pipeline>/terminal_<UTC>.log`):

```bash
GPU_DEVICE=0 ./wait_gpu.sh                              # resume / run the whole study on host GPU 0
GPU_DEVICE=0 ./wait_gpu.sh --phases "1 2 3 4 5" --force # start Phases 1-5 over (old outputs -> *.bak-<UTC>)
```

Copy the results to your machine:

```bash
rsync -avz --progress -e "ssh -p 13508" \
    antoniovinicius@164.41.75.221:/home/antoniovinicius/projects/sandbox_yolo26/logs/pipeline_final_v1 \
    /home/avmoura_linux/Documents/unb/SANDBOX_YOLO26/logs/
```

Hardware monitoring: `nvidia-smi`, `nvtop`.
