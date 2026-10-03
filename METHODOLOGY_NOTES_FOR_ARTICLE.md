# Methodology Notes for the Article — YOLO26-seg on ISIC 2018 Task 1

> **Purpose.** This document is the technical foundation for the *Materials and Methods* section of the
> Master's thesis and the associated article. It describes the experimental protocol exactly as implemented
> in this repository (`run_pipeline.sh` and `yolo26_seg/`). Items in **[brackets]** must be completed with
> values from the final run (dataset counts, hardware actually used, results). Section 9 lists points the
> authors must verify or disclose before submission.

---

## 1. Overview and experimental design

We evaluate the five sizes of the Ultralytics YOLO26 instance-segmentation family (YOLO26n/s/m/l/x-seg) for
skin-lesion segmentation on the ISIC 2018 Task 1 dataset. The design answers two questions in sequence:

1. **How well does the architecture perform "out of the box"?** All sizes are first trained with a fixed base
   training setup and the framework's default hyperparameters (*Baseline*), and their generalisation is
   estimated by K-fold cross-validation.
2. **What does hyperparameter optimisation (HPO) add, and at what computational cost?** A seeded genetic
   search selects hyperparameters per size; the resulting *Optimised* models are compared with the Baseline
   models on a held-out test set that is used exclusively in the final phase, together with a hardware
   efficiency profile (latency, throughput, memory, model size) in single (FP32) and half (FP16) precision.

The protocol is organised in five sequential phases (Section 2). All phases share a single, explicitly
defined base training setup — optimiser, learning-rate schedule, training budget, numerical precision and
data pipeline (Section 3) — so that Baseline and Optimised models differ **only** in the values of the tuned
hyperparameters (Baseline: framework defaults; Optimised: HPO result).

### 1.1 Data

The ISIC 2018 Task 1 (lesion boundary segmentation) images and binary masks were converted to the YOLO
segmentation format (one polygon per lesion, normalised coordinates). The dataset is divided into three
disjoint splits: training ([N_train] images), validation ([N_val] images) and test ([N_test] images). The test
split is accessed only in Phase 5. All images are processed at an input resolution of 640 × 640 pixels
(letterbox resizing performed by the framework); the task is treated as single-class segmentation (`nc = 1`).

### 1.2 Models

We use the five COCO-pretrained YOLO26-seg checkpoints released by Ultralytics (`yolo26{n,s,m,l,x}-seg.pt`) as
initialisation for every training run (transfer learning; all layers trainable).

### 1.3 Computing environment

All experiments were executed in a Docker container built from `nvidia/cuda:12.1.0-devel-ubuntu22.04` with
Python 3.11 and the following pinned versions: PyTorch 2.5.1, torchvision 0.20.1, torchaudio 2.5.1 (CUDA 12.1
builds) and Ultralytics 8.4.21. Training used distributed data parallelism (DDP) on [two NVIDIA Tesla V100S
32 GB GPUs]; Phase 5 inference and profiling used a single GPU ([NVIDIA Tesla V100S-PCIE-32GB], driver
[version]); host CPU [Intel Xeon Gold 5220R]. The exact library versions and GPU model are recorded in every
output file of the pipeline.

---

## 2. Pipeline architecture (five phases)

| Phase | Purpose | Data used | Output |
|---|---|---|---|
| 1. Baseline training | Train each size with the base setup and default hyperparameters | train (fit), val (model selection) | Baseline weights |
| 2. Baseline cross-validation | Estimate generalisation and its variability with the Phase 1 configuration | train ∪ val pool, 5 folds | Fold-wise and aggregated metrics |
| 3. Hyperparameter optimisation | Seeded genetic search per size | train (fit), val (fitness) | Best hyperparameters per size |
| 4. Optimised fine-tuning | Train each size with the base setup and the Phase 3 hyperparameters | train (fit), val (model selection) | Optimised weights |
| 5. Test-set evaluation and profiling | Accuracy and efficiency of Baseline **and** Optimised models | **test** (only phase that accesses it) | Final metrics, HPO gain, efficiency |

**Phase 1 — Baseline training.** Each size is fine-tuned on the training split, with the validation split
used for per-epoch evaluation, early stopping and checkpoint selection. Training uses the shared base setup
of Section 3.1; every tunable hyperparameter (initial and final learning rate, momentum, weight decay,
warm-up, loss gains, data augmentation) is left at its Ultralytics default value.

**Phase 2 — Baseline cross-validation.** The training and validation splits are merged into a single pool and
partitioned into K = 5 deterministic folds (Section 4). For each size and fold, a model is trained with the
exact Phase 1 configuration on four folds and evaluated on the held-out fold. Instance-level metrics and
pixel-level overlap metrics are computed for every fold and aggregated across folds.

**Phase 3 — Hyperparameter optimisation.** For each size, a genetic algorithm (Section 3.2) explores 15
hyperparameters for 30 trials of 30 epochs each (early-stopping patience of 10 epochs). Each trial trains on
the training split and is scored on the validation split. The configuration with the highest fitness is
selected.

**Phase 4 — Optimised fine-tuning.** Each size is retrained from the pretrained checkpoint on the training
split with the Phase 3 hyperparameters and the same base setup as Phase 1 (Section 3.1).

**Phase 5 — Test-set evaluation and profiling.** The final Baseline (Phase 1) and Optimised (Phase 4) weights
of every size are evaluated on the unseen test split for instance-level and pixel-level accuracy, in FP32 and
FP16 (Section 6). Hardware efficiency is profiled with batch size 1 (Section 7). The HPO gain is quantified by
a paired, per-image comparison of the two variants (Section 8).

**Test-set isolation.** The test split is never part of the cross-validation pool, the HPO, or any model
selection step. The implementation additionally verifies, before Phase 2, that no test image appears in the
cross-validation pool (by resolved file path and by file name) and aborts otherwise.

---

## 3. Training configuration and methodological deviations from the framework defaults

### 3.1 Shared base setup and standardised training budget

To ensure a fair ("apples-to-apples") comparison, every training run — Baseline (Phases 1 and 2), every HPO
trial (Phase 3) and Optimised (Phase 4) — uses the same **base setup**, defined once in the code. The two
variants are then defined as

* **Baseline** = base setup + Ultralytics default hyperparameters;
* **Optimised** = base setup + hyperparameters selected by the HPO (Section 3.2),

so that the values of the tuned hyperparameters are the only experimental variable between them. The
implementation rejects any tuned-hyperparameter file or search space that attempts to modify a base-setup
setting.

| Setting | Value used | Ultralytics 8.4.21 default | Rationale |
|---|---|---|---|
| Optimiser | MuSGD | `auto` | See below. |
| Learning-rate schedule | cosine (`cos_lr = True`) | linear | One schedule for every phase. |
| Epochs | 120 | 100 | Common budget for every training phase (HPO trials: 30). |
| Early-stopping patience | 25 epochs | 100 | Common budget for every training phase (HPO trials: 10). |
| Mixed precision (`amp`) | disabled (FP32) | enabled | See below. |
| `close_mosaic` | 10 | 10 | Mosaic augmentation disabled in the last 10 epochs; pinned explicitly (an earlier HPO used 15). |
| Dataloader workers | 8 (per process) | 8 | Pinned; the worker count changes how augmentation random streams are assigned to samples. |
| Batch size / nominal batch size | 16 / 64 | 16 / 64 | Gradient accumulation of 4 micro-batches → effective optimisation batch of 64. |
| Input size | 640 | 640 | — |
| Seed / deterministic mode | 0 / enabled | 0 / enabled | Pinned explicitly. |

**Disabling automatic mixed precision.** In a preliminary experiment the largest variant (YOLO26x-seg)
diverged under FP16 mixed-precision training (NaN in the classification loss), consistent with a numerical
overflow that the gradient scaler did not prevent. Rather than disabling mixed precision only for the
affected size, which would make the comparison across architectures asymmetric, all sizes and all phases are
trained in full FP32 precision. This also removes a hardware-dependent source of numerical variation.

**Explicit optimiser.** The framework default, `optimizer = "auto"`, does not designate a fixed optimiser: in
Ultralytics 8.4.21 it selects MuSGD (lr = 0.01, momentum = 0.9) when a run exceeds 10⁴ optimisation iterations
and AdamW (lr = 0.002·5/(4 + nc) = 0.002 for nc = 1, momentum = 0.9) otherwise, and in both cases ignores the
configured initial learning rate and momentum. Left in place, it would have trained the Baseline with a
different optimiser from the Optimised models (which require an explicit optimiser for the tuned learning rate
and momentum to take effect), confounding the effect of the HPO with that of the optimiser. We therefore fix the
optimiser to **MuSGD** with a cosine learning-rate schedule in every phase. With an explicit optimiser, the
configured learning-rate and momentum values are used as given: the Baseline trains with the framework
defaults (lr0 = 0.01, lrf = 0.01, momentum = 0.937, weight decay = 5×10⁻⁴), the Optimised models with the
values selected by the HPO. We verified in the training logs that every phase reports
`optimizer: MuSGD(lr=…, momentum=…)` with the expected values.

**HPO trial micro-batch.** To reduce search time, HPO trials use a micro-batch of 32 (16 for YOLO26x-seg, owing
to GPU memory limits in FP32 with DDP). Because the nominal batch size is fixed at 64, the effective
optimisation batch size is 64 in every phase and for every size; only the number of accumulated micro-batches
and the per-step memory footprint differ.

### 3.2 Hyperparameter optimisation with a seeded genetic algorithm

**Algorithm.** We use the genetic algorithm implemented in the Ultralytics `Tuner`. The first trial evaluates the
default hyperparameters clipped to the search bounds (note that two defaults, lr0 = 0.01 and weight decay =
5×10⁻⁴, lie outside the refined bounds and are clipped to 0.004 and 1×10⁻⁴; see Section 9). For every
subsequent trial, a new hyperparameter vector is generated from the history of evaluated trials as follows:

1. **Parent selection.** The (up to) nine trials with the highest fitness are retained. Nine parents are drawn
   from them with replacement, with probability proportional to their fitness shifted to be positive
   ($f_i - \min_j f_j + 10^{-6}$).
2. **Crossover (BLX-α, α = 0.2).** For each gene, a value is drawn uniformly from
   $[\,\ell - \alpha s,\; h + \alpha s\,]$, where $\ell$ and $h$ are the minimum and maximum of that gene among the
   drawn parents and $s = h - \ell$.
3. **Mutation.** Each gene is mutated with probability 0.5 by a multiplicative factor
   $\exp(\varepsilon)$, $\varepsilon \sim \mathcal{N}(0, (\sigma\, g_k)^2)$, clipped to $[0.25, 4]$, where $g_k$ is a
   per-gene gain and $\sigma$ decays linearly from 0.2 to 0.1 over the first 300 trials. Sampling is repeated
   until at least one gene changes.
4. **Constraint.** Each gene is clipped to its search bounds and rounded to five decimal places.

Fitness is the Ultralytics segmentation fitness, i.e. the sum of the box and mask mAP@50-95 on the validation
split, $F = \mathrm{mAP}^{B}_{50\text{-}95} + \mathrm{mAP}^{M}_{50\text{-}95}$ (Ultralytics 8.4.21).

**Search space.** The "refined" search space comprises 15 hyperparameters (ranges as implemented):

| Group | Hyperparameter: range |
|---|---|
| Optimisation | lr0: [1×10⁻³, 4×10⁻³]; lrf: [0.005, 0.05]; momentum: [0.85, 0.95] (gain 0.3); weight_decay: [1×10⁻⁶, 1×10⁻⁴]; warmup_epochs: [1, 5] |
| Loss gains | cls: [0.2, 1.5]; dfl: [0.8, 1.5] |
| Colour augmentation | hsv_h: [0.005, 0.025]; hsv_s: [0.3, 0.9]; hsv_v: [0.2, 0.7] |
| Geometric augmentation | translate: [0.05, 0.20]; flipud: [0.0, 0.10] |
| Mixing augmentation | mosaic: [0.7, 1.0]; mixup: [0.0, 0.05]; copy_paste: [0.0, 0.05] |

These ranges were narrowed from a broader 22-parameter space after an exploratory search on YOLO26s-seg:
hyperparameters with negligible correlation with fitness (|r| < 0.05) were fixed at their defaults, and
high-signal ranges were narrowed around the best-performing region. **[State whether that exploratory search
used the training/validation splits only — it did not use the test split.]**

**Deviation: deterministic mutation (`SeededTuner`).** In the reference implementation, the random number
generator is re-seeded with the current wall-clock time before every mutation
(`np.random.seed(int(time.time()))`), and parent selection uses Python's global `random` module. As a
consequence, two runs of the search, or a resumed run, propose different hyperparameters. To make the search
reproducible we subclass the `Tuner` (`SeededTuner`) and re-implement the mutation step with **the same
crossover and mutation operators and constants**, but drawing every random number from a dedicated generator
initialised as `numpy.random.default_rng([seed, i])`, where *i* is the index of the trial being proposed and
seed = 0. Parents are ranked with a stable sort and drawn with the same fitness-proportional probabilities.
The proposal for trial *i* is therefore a deterministic function of (seed, *i*, fitness history of trials
0 … *i* − 1). In particular, a trial that is re-run after an interruption receives exactly the same
hyperparameters. We verified that two independent executions of the search produce byte-identical trial
histories when the trial fitness values are identical.

*Scope of the guarantee.* The search is reproducible conditional on the fitness values. Each fitness value
results from a GPU training run executed with deterministic algorithms requested
(`torch.use_deterministic_algorithms(True, warn_only=True)`, deterministic cuDNN, fixed seeds); operations
without a deterministic implementation and the reduction order of DDP may still introduce small run-to-run
differences, which would then propagate to subsequent proposals.

**Fault tolerance of the search.** Because a full search requires several GPU-days, the search is checkpointed
and resumable without altering its outcome:

* The trial history (`tune_results.csv`) and a state file are written after every trial; state writes are
  atomic (write to a temporary file, `fsync`, atomic rename).
* Completion is determined by the number of recorded trials, not by the presence of the best-hyperparameter
  file (which the reference implementation rewrites after every trial).
* On resumption, partially written history lines and the folder of the interrupted trial are discarded, and
  the search continues with the interrupted trial, which receives the same hyperparameters (see above).
* A trial that fails (fitness = 0, e.g. out-of-memory or numerical divergence) is retried with identical
  hyperparameters up to two times and is recorded as a failed trial thereafter. Failures that coincide with an
  unavailable GPU are not counted against this limit, and the search is resumed once the device is available.
* Resumption is refused if the search space, the fixed training configuration, the seed, the pretrained
  weights or the Ultralytics version differ from those of the original run, so that trials obtained under
  different configurations are never combined.

### 3.3 Interrupted training runs and the duplicate-epoch issue

Training runs (Phases 1, 2 and 4) that are interrupted are resumed from the last saved checkpoint, which
restores the model, the exponential-moving-average weights, the optimiser state and the epoch counter. The
data-loader random state is not part of the checkpoint; a resumed run is therefore not bit-identical to an
uninterrupted one. Every resumption is logged and reported (Section 9).

**Duplicate epoch records.** Within each epoch the framework first appends the validation metrics to the
training log (`results.csv`) and then saves the checkpoint. If a run is interrupted between these two steps,
the log contains a row for an epoch whose weights were never saved; after resumption that epoch is executed
again and logged a second time. A naïve selection of the best epoch from the log could therefore select the
orphaned row, i.e. report metrics of weights that do not exist. When parsing training logs we retain only the
**last** record of each epoch, which always corresponds to the checkpointed execution. The number of trained
epochs is computed over unique epochs.

---

## 4. Cross-validation protocol (Phase 2)

**Pool.** The pool consists of all images of the training and validation splits (N_pool = [N_train + N_val]);
images without annotations would be retained as background images. The test split is excluded and its
absence from the pool is verified (Section 2).

**Deterministic splitting without external libraries.** Image indices are shuffled with NumPy's legacy
Mersenne-Twister generator (`numpy.random.RandomState(0).shuffle`) and partitioned into K = 5 contiguous
blocks; the first N_pool mod K blocks receive one additional image. Fold *k* uses block *k* as the held-out
set and the remaining blocks for training. This procedure reproduces the partition of
`sklearn.model_selection.KFold(n_splits=5, shuffle=True, random_state=0)` without depending on scikit-learn.
The image lists of every fold are fingerprinted (SHA-256) in a manifest; any later execution whose
partition differs (changed data, K or seed) is refused, so that folds produced from different partitions can
never be combined.

**Training and evaluation per fold.** Each fold trains a model with the exact Phase 1 configuration
(Section 3.1). The held-out fold serves as the per-epoch validation set. For each fold we report:

* *Instance-level metrics* (box and mask precision, recall, mAP@50, mAP@50-95, and F1 = 2PR/(P + R)) of the
  epoch that produced the fold's selected checkpoint (`best.pt`; Section 5).
* *Pixel-level metrics* (DSC, JSI and the further metrics of Section 6.2) computed on the held-out images with
  the same checkpoint, in FP32, with the same definitions as on the test set.

Both families of fold metrics therefore describe exactly the same model.

**Aggregation.** For every metric, the fold-wise values are summarised by their mean and the **sample**
standard deviation,

$$
\bar{x} = \frac{1}{K}\sum_{k=1}^{K} x_k, \qquad
s = \sqrt{\frac{1}{K-1}\sum_{k=1}^{K} (x_k - \bar{x})^2 } \quad (\text{ddof} = 1),
$$

which is the appropriate estimator of the dispersion of a performance estimate from a small number of folds
(the population form, ddof = 0, underestimates it by a factor $\sqrt{(K-1)/K} \approx 0.89$ for K = 5). Results
are reported as $\bar{x} \pm s$. Because the folds share training data, fold-wise values are not independent,
and *s* should be read as a descriptive measure of variability rather than as the basis for a formal test.

---

## 5. Model selection within a training run

Ultralytics evaluates the model on the validation set after every epoch and keeps the checkpoint with the
highest fitness ($\mathrm{mAP}^{B}_{50\text{-}95} + \mathrm{mAP}^{M}_{50\text{-}95}$, Section 3.2) as `best.pt`; when an epoch
equals the best fitness so far, `best.pt` is overwritten, so ties resolve to the latest epoch. Training stops
after 120 epochs or when the fitness has not improved for 25 epochs. The `best.pt` checkpoints of Phases 1 and 4
are the models evaluated in Phase 5; the `best.pt` checkpoint of each fold is the model used for the fold's
metrics in Phase 2.

**Alignment of reported validation metrics with the selected checkpoint.** All validation and
cross-validation metrics reported from training logs are those of the epoch that produced `best.pt`. Because
the training log stores metrics rounded to six significant digits whereas the framework selects the checkpoint
on unrounded values, recomputing the selection from the log could, in rare near-ties, designate a different
epoch. We therefore identify the epoch from the checkpoint itself: `best.pt` stores the validation metrics of
the epoch that produced it, and the log row with identical values is selected (after the duplicate-epoch
de-duplication of Section 3.3). Only if this lookup fails do we fall back to recomputing the framework's fitness
from the log, with ties resolved to the latest epoch. We also verify that the fitness stored in the checkpoint
equals the box + mask mAP@50-95 of its metrics (within the framework's rounding of 10⁻⁵), which guards against
a silent change of the fitness definition in a future framework version. An earlier version of our pipeline
read the epoch with the highest *mask* mAP@50-95 only; in our tests this designated a different epoch from
`best.pt` in one of twelve runs, which motivated the alignment.

---

## 6. Accuracy evaluation on the test set (Phase 5a)

Every combination of variant (Baseline, Optimised) × model size × numerical precision (FP32, FP16) is evaluated
on the test split. FP32 is the primary result, since all models were trained in FP32; the FP16 evaluation
quantifies the accuracy cost of half-precision deployment and is read together with the FP16 efficiency
results (Section 7).

### 6.1 Instance-level metrics

The framework's validator is run on the test split with batch size 1 and its default settings for mAP
computation (confidence threshold 0.001, NMS IoU threshold 0.7 where applicable): box and mask precision,
recall, mAP@50 and mAP@50-95; F1 is derived as 2PR/(P + R).

### 6.2 Pixel-level overlap metrics

Lesion-segmentation studies on ISIC 2018 report overlap between binary lesion masks, which we compute
independently of the framework:

* **Ground truth.** All polygons of the image's label file are rasterised at the original image resolution
  and merged (union) into one binary mask.
* **Prediction.** The model is run on each test image individually (batch size 1) with confidence threshold
  0.001. Instance masks are produced at the original image resolution (`retina_masks=True`) and binarised at
  0.5; only the mask of the **highest-confidence instance (top-1)** is scored, and lower-ranked instances are
  discarded rather than merged. Due to the single-lesion nature of ISIC 2018 Task 1, YOLO26 pixel evaluation utilizes a top-1 confidence selection at conf=0.001. This maximizes lesion recall while strictly preventing the merging of low-confidence background artifacts.
  The same rule is used for the cross-validation pixel metrics, the test set and the end-to-end latency
  benchmark. *Provenance:* the rule replaces the earlier union of instances with confidence ≥ 0.25; it was
  adopted on 2026-10-03 after a diagnostic on the Phase 1 validation split (with the union rule, 3–7 % of
  validation images received an empty mask at conf 0.25, and low-confidence background instances inflated the
  mask at conf 0.001), and before any Phase 2 pixel-level or Phase 5 test-set evaluation was run.
* **Per-image metrics.** From the pixel counts TP, FP, FN and TN:

$$
\mathrm{DSC} = \frac{2\,TP}{2\,TP + FP + FN}, \qquad
\mathrm{JSI} = \frac{TP}{TP + FP + FN}, \qquad
\mathrm{JSI}_{0.65} = \begin{cases} \mathrm{JSI} & \mathrm{JSI} \ge 0.65 \\ 0 & \text{otherwise} \end{cases}
$$

  together with sensitivity TP/(TP + FN), specificity TN/(TN + FP) and pixel accuracy (TP + TN)/N.
  $\mathrm{JSI}_{0.65}$ is the thresholded Jaccard index used as the official ISIC 2018 Task 1 score.
* **Empty masks.** An image with an empty prediction and a non-empty ground truth receives DSC = JSI = 0 and is
  **included** in all averages (missed lesions are counted as complete failures, not discarded). If both masks
  were empty, the image would receive DSC = JSI = 1 (perfect agreement); sensitivity is undefined in that case
  and excluded from its mean. The number of empty predictions is reported.
* **Dataset-level statistics.** For each metric we report the per-image mean (primary figure), the sample
  standard deviation, the median and interquartile range, and a 95 % confidence interval of the mean obtained
  by a percentile bootstrap (2,000 resamples of the test images, fixed seed 0). The pooled ("micro") DSC and JSI,
  computed from TP, FP and FN summed over the test set, are reported as secondary figures.

The predicted masks (FP32) are stored, and the qualitative figures are generated from these stored masks and
the same rasterisation routine, so that the visualisations show exactly what was scored.

---

## 7. Hardware efficiency profiling (Phase 5b)

### 7.1 General conditions

* **Batch size 1**, on a single GPU, at an input size of 640 × 640, for every variant × size × precision.
* **Process isolation.** Every configuration is measured in a newly started process, so that peak-memory
  counters, the allocator cache and cuDNN autotuning state cannot carry over between models.
* **Kernel selection.** `torch.backends.cudnn.benchmark = True`, reflecting deployment with a fixed input
  shape. Autotuning occurs during warm-up, which is excluded from all measurements.
* **Fused model.** Convolution and batch-normalisation layers are fused (as done by the framework at inference
  time); parameters do not require gradients and inference runs under `torch.inference_mode()`.
* **Contention control.** GPU utilisation, memory in use, SM clock and temperature are sampled with
  `nvidia-smi` immediately before and after each measurement. A measurement during which the device
  utilisation from other processes exceeded 5 % is flagged as contended; only uncontended measurements are
  reported. **[Report the GPU model, driver version and clock behaviour; state that measurements were
  repeated on an idle GPU when flagged.]**

### 7.2 Latency and throughput

Two scopes are measured:

1. **Network forward pass.** A fixed input tensor (1 × 3 × 640 × 640, uniform random values with seed 0) is
   processed 50 times for warm-up and then 500 times. Each timed iteration is bracketed by a pair of CUDA
   events (`torch.cuda.Event(enable_timing=True)`) recorded on the current stream, followed by a device
   synchronisation; the latency is the elapsed time between the events. CUDA events measure execution on the
   device itself and are not affected by the asynchronous nature of kernel launches from the host, which makes
   them more accurate than host timers for GPU work. Synchronising after every iteration guarantees that each
   measurement corresponds to exactly one image with no queued work (true batch-1 latency).
2. **End-to-end inference.** The framework's `predict()` is applied to a fixed test image (the first image of
   the test split) 20 times for warm-up and 200 times for measurement, including image pre-processing,
   inference and mask post-processing. Because this scope includes host-side work that CUDA events cannot
   capture, it is timed with a monotonic host clock (`time.perf_counter`) between two device synchronisations.

For each scope we report the mean, standard deviation, median, 90th, **95th** and 99th percentiles, minimum and
maximum latency in milliseconds. Throughput is reported as

$$
\mathrm{FPS} = \frac{1000}{\overline{t}_{\mathrm{ms}}},
$$

where $\overline{t}_{\mathrm{ms}}$ is the mean latency; 1000/median is also reported. The median characterises typical
latency, while P95 characterises the worst-case behaviour relevant for real-time deployment. The raw
per-iteration latencies are retained for distribution plots.

### 7.3 Memory

* **GPU memory (VRAM).** We report memory allocated by the PyTorch caching allocator:
  * the memory occupied by the weights alone (allocated memory after loading minus before loading);
  * the **steady-state peak** memory allocated and reserved during the timed iterations. The peak counters are
    reset *after* warm-up (`torch.cuda.reset_peak_memory_stats`), because with `cudnn.benchmark` the warm-up
    peak is dominated by the temporary workspaces of cuDNN's algorithm search and does not represent the
    memory needed for inference. (In a test with YOLO26n-seg, the warm-up peak was ~2.2 GB versus ~74 MB at
    steady state in FP32.) The warm-up peak is recorded separately for completeness;
  * the steady-state peak during end-to-end inference (`predict()`).

  The CUDA context created by the driver (a constant of a few hundred megabytes per process, independent of
  the model) is outside the PyTorch allocator and is not included.
* **Host memory (RAM).** Resident set size (RSS) of the process after model loading and after the benchmark,
  and its peak over the process lifetime (`getrusage`).

All memory and size figures are expressed in MiB (2²⁰ bytes).

### 7.4 Model size and computational cost

* **Parameters:** number of parameters of the fused model (the deployed form; the unfused count is also
  reported).
* **GFLOPs:** floating-point operations for one 640 × 640 image, computed with `thop` as
  2 × multiply-accumulate operations (Ultralytics convention).
* **Size on disk:** size of the `best.pt` checkpoint. Final Ultralytics checkpoints store weights in FP16;
  we therefore also report the theoretical weight sizes in FP32 (4 bytes × parameters) and FP16
  (2 bytes × parameters).

### 7.5 FP32 versus FP16

For FP16, the weights and the input tensor of the forward-pass benchmark are cast to half precision
(`model.half()`), and `predict(half=True)` is used for end-to-end inference. FP32 and FP16 are profiled
under identical conditions, and the accuracy of every FP16 model is measured on the test set (Section 6)
rather than assumed. FP16 reduces weight memory by a factor of two; its latency benefit depends on whether
inference is compute-bound: for small models at batch size 1, kernel-launch overhead can dominate, and FP16 may
bring no speed-up (in a preliminary measurement YOLO26n-seg was slightly slower in FP16 than in FP32 on a V100S).

Baseline and Optimised models share the same architecture and therefore the same computational cost; their
efficiency figures are expected to coincide within measurement noise and are reported for the deployed
(Optimised) models.

---

## 8. Statistical analysis

* **Cross-validation:** mean ± sample standard deviation (ddof = 1) over five folds (Section 4).
* **Test set, per configuration:** per-image mean with a percentile-bootstrap 95 % CI (2,000 resamples,
  seed 0) (Section 6.2).
* **HPO gain (Optimised − Baseline), per model size, FP32:** because both variants are evaluated on the same test
  images, the comparison is paired. For each image *j* we compute $d_j = \mathrm{DSC}^{\mathrm{opt}}_j - \mathrm{DSC}^{\mathrm{base}}_j$
  (and likewise for JSI). We report the mean difference $\bar{d}$, its percentile-bootstrap 95 % CI (2,000
  resamples of images, seed 0), the numbers of improved and degraded images, and the *p*-value of a two-sided
  Wilcoxon signed-rank test (zero differences discarded). The change in mask mAP@50-95 is also reported.
  **[If claims are made for all five sizes jointly, correct the five p-values for multiple comparisons, e.g.
  with the Holm–Bonferroni procedure.]**

---

## 9. Reproducibility, traceability and points to disclose

**Reproducibility measures.** Fixed seeds for Python, NumPy and PyTorch; deterministic algorithms requested
in PyTorch and cuDNN; seeded HPO mutations (Section 3.2); deterministic K-fold splitting with a fingerprinted
manifest (Section 4); pinned software environment (Section 1.3); every result file records the SHA-256 of the
weights it was computed from, the evaluation settings and the library versions, and results are recomputed
automatically when any of them changes.

**Points the authors must verify or disclose:**

1. **Resumed training runs.** Report which runs, if any, were resumed after an interruption (logged in each
   run's `run_state.json` and flagged by the final report), as resumed runs are not bit-identical.
2. **Search space versus Baseline values.** In the refined search space the default learning rate
   (lr0 = 0.01) and weight decay (5×10⁻⁴) lie outside the search bounds ([10⁻³, 4×10⁻³] and [10⁻⁶, 10⁻⁴]).
   The Baseline configuration is therefore not itself a candidate of the search, and the HPO is not guaranteed
   to return a configuration at least as good as the Baseline; the comparison of Phase 5 measures the effect of
   the selected configuration. State this explicitly, together with the origin of the refined bounds
   (Section 3.2). **[Alternatively, widen these two bounds to include the defaults before the final run.]**
3. **Best-epoch alignment.** State that reported validation metrics are those of the epoch that produced the
   selected checkpoint (Section 5).
4. **Optimistic bias of validation-based figures.** In Phases 1, 2 and 4 the same validation data (the
   validation split, or the held-out fold) is used both to select the checkpoint and to report the metric, which
   biases those figures slightly upwards. Only the Phase 5 test-set figures are free of this selection bias and
   should be used for the main claims.
5. **HPO failures.** Report the number of completed and failed trials per size (`hpo_state.json`).
6. **Contended efficiency measurements.** Confirm that no reported latency carries the contention flag.

---

## 10. Suggested references

*(Verify bibliographic details before use.)*

* Ultralytics YOLO (software), version 8.4.21, https://github.com/ultralytics/ultralytics.
* N. Codella et al., "Skin Lesion Analysis Toward Melanoma Detection 2018: A Challenge Hosted by the
  International Skin Imaging Collaboration (ISIC)," arXiv:1902.03368, 2019.
* P. Tschandl, C. Rosendahl, H. Kittler, "The HAM10000 dataset, a large collection of multi-source
  dermatoscopic images of common pigmented skin lesions," *Scientific Data* 5, 180161, 2018.
* L. R. Dice, "Measures of the amount of ecologic association between species," *Ecology* 26(3), 1945.
* P. Jaccard, "The distribution of the flora in the alpine zone," *New Phytologist* 11(2), 1912.
* L. J. Eshelman, J. D. Schaffer, "Real-coded genetic algorithms and interval-schemata," in *Foundations of
  Genetic Algorithms 2*, 1993 (BLX-α crossover).
* F. Wilcoxon, "Individual comparisons by ranking methods," *Biometrics Bulletin* 1(6), 1945.
* B. Efron, R. J. Tibshirani, *An Introduction to the Bootstrap*, Chapman & Hall, 1993.
* S. Holm, "A simple sequentially rejective multiple test procedure," *Scandinavian Journal of Statistics*
  6(2), 1979.
* A. Paszke et al., "PyTorch: An Imperative Style, High-Performance Deep Learning Library," NeurIPS 2019.
