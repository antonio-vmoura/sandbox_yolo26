"""Shared configuration and helpers for the 5-phase YOLO26-seg pipeline.

Every phase script imports its constants, training protocols and output paths
from this module so that the experimental protocol is defined **once**:

* :func:`baseline_protocol` — Phase 1 (baseline) and Phase 2 (baseline CV):
  the fixed **base setup** (:data:`BASE_RECIPE` + budget + pinned values)
  with the Ultralytics **default hyperparameters**.
* :func:`optimized_protocol` — Phase 4 (optimised fine-tune): the same base
  setup + the tuned HPs.
* :func:`hpo_trial_protocol` — Phase 3 (HPO) trials: the same base setup with
  the short per-trial budget and the micro-batch, so the HPs are selected
  under the same conditions they are later trained with. The first trial
  evaluates the default HPs clipped to the search bounds; in the ``refined``
  space the defaults ``lr0=0.01`` and ``weight_decay=5e-4`` lie outside the
  bounds, so the Baseline values themselves are not part of the search.

Baseline = base setup + default HPs; Optimised = base setup + tuned HPs. The
tuned hyperparameters (learning rate, momentum, weight decay, loss gains,
augmentation) are therefore the **only** difference between the two.

Documented deviations from the Ultralytics defaults (applied uniformly to
every phase):

* ``epochs=120`` (default 100) and ``patience=25`` (default 100) — identical
  training budget for Phases 1, 2 and 4.
* ``amp=False`` (default True) — the xlarge variant overflowed in FP16 (NaN in
  the cls-loss); FP32 everywhere keeps numerical conditions uniform across
  architectures.
* ``optimizer="MuSGD"`` (default ``"auto"``, which picks AdamW or MuSGD from
  the iteration count and then ignores ``lr0``/``momentum``) and
  ``cos_lr=True`` (default False) — one explicit optimiser and schedule for
  every phase, so the default and tuned ``lr0``/``momentum`` are both used.

The remaining protocol values (``batch=16``, ``nbs=64``, ``imgsz=640``,
``workers=8``, ``close_mosaic=10``, ``seed=0``, ``deterministic=True``) equal
the Ultralytics defaults but are pinned explicitly so that an upstream default
change cannot silently alter the protocol.

Output layout (see :class:`PipelinePaths`)::

    <root>/                              # e.g. /workspace/logs/pipeline_final_v1
    ├── phase1_baseline/yolo26_<m>_baseline/
    ├── phase2_cv_baseline/yolo26_<m>/{splits/, runs/fold_<k>/, metrics_*}
    ├── phase3_hpo/tune_<m>/{tune_results.csv, best_hyperparameters.yaml,
    │                        hpo_state.json, trials/}
    ├── phase4_optimized/yolo26_<m>_optimized/
    ├── phase5_test/{accuracy/, per_image/, masks/, efficiency/, val_runs/}
    ├── summary/
    └── pipeline_runs/<UTC-timestamp>/
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import random
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Union

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
#: Canonical order of model sizes used across the pipeline.
DEFAULT_ORDER: list[str] = ["nano", "small", "medium", "large", "xlarge"]

#: Directory of the pretrained weights (canonical Docker cache). Override with
#: ``YOLO26_WEIGHTS_DIR`` when running outside the container.
WEIGHTS_DIR: str = os.environ.get("YOLO26_WEIGHTS_DIR", "/workspace/cache")

#: Path to the pretrained weights for each variant. Ultralytics auto-downloads
#: missing weights on first use.
WEIGHTS: dict[str, str] = {
    m: f"{WEIGHTS_DIR}/yolo26{m[0]}-seg.pt"
    for m in ("nano", "small", "medium", "large", "xlarge")
}

#: Default dataset YAML (must declare ``train``, ``val`` and ``test`` splits).
DEFAULT_DATA_YAML: str = "/workspace/datasets/isic_2018_task1_yolo26/data.yaml"

#: Default pipeline root; isolates this study from older runs under ``logs/``.
DEFAULT_PIPELINE_ROOT: str = "/workspace/logs/pipeline_final_v1"

#: Global seed for every RNG in the pipeline (training, K-Fold, HPO mutation).
SEED: int = 0

#: Training budget shared by Phases 1, 2 and 4.
TRAIN_EPOCHS: int = 120
TRAIN_PATIENCE: int = 25

#: Pinned protocol values (equal to the Ultralytics defaults).
IMGSZ: int = 640
BATCH: int = 16
NBS: int = 64
WORKERS: int = 8

#: Fixed base setup shared by EVERY phase (Baseline, HPO trials, Optimised).
#: Never searched by the HPO and never overridden by tuned HPs.
#: ``close_mosaic`` equals the Ultralytics default but is pinned because the
#: legacy HPO used 15.
BASE_RECIPE: dict[str, Any] = {
    "optimizer": "MuSGD",
    "cos_lr": True,
    "close_mosaic": 10,
}

#: Keys that define the base setup, budget and reproducibility contract. A
#: tuned hyperparameter YAML or an HPO search space must never contain them
#: (see :func:`optimized_protocol`).
PROTECTED_KEYS: frozenset[str] = frozenset({
    "data", "task", "pretrained", "imgsz", "device", "batch", "nbs", "workers",
    "epochs", "patience", "amp", "deterministic", "seed", *BASE_RECIPE,
})

#: Type alias for the ``device`` argument accepted by Ultralytics.
DeviceArg = Union[int, str, list[int]]


# ----------------------------------------------------------------------------
# Device & reproducibility
# ----------------------------------------------------------------------------
def parse_device(arg: str) -> DeviceArg:
    """Parse a ``--device`` CLI value into a value Ultralytics accepts.

    Args:
        arg: ``"0"`` for single-GPU, ``"0,1"`` for DDP, or ``"cpu"``.

    Returns:
        ``list[int]`` for multi-GPU, ``"cpu"`` for CPU, ``int`` otherwise.
    """
    if "," in arg:
        return [int(x) for x in arg.split(",")]
    if arg == "cpu":
        return "cpu"
    return int(arg)


def seed_everything(seed: int = SEED, deterministic: bool = True) -> None:
    """Seed every RNG of the current process and request deterministic kernels.

    Must be called at the very start of ``main()`` — before any CUDA context is
    created — because ``CUBLAS_WORKSPACE_CONFIG`` is only honoured at CUDA
    initialisation. The environment variables are also inherited by the
    training subprocesses spawned by Ultralytics (DDP, HPO trials).

    Ultralytics additionally re-seeds inside each training run through
    ``seed=`` / ``deterministic=True`` (see :func:`baseline_protocol`).

    Args:
        seed: Seed for ``random``, ``numpy`` and ``torch``.
        deterministic: Enforce deterministic cuDNN / cuBLAS behaviour.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


# ----------------------------------------------------------------------------
# Training protocols
# ----------------------------------------------------------------------------
def baseline_protocol(
    data: str,
    device: DeviceArg,
    epochs: int = TRAIN_EPOCHS,
    patience: int = TRAIN_PATIENCE,
) -> dict[str, Any]:
    """Return the Phase 1 / Phase 2 training kwargs: base setup + default HPs.

    Sets the fixed base setup shared by every phase (:data:`BASE_RECIPE`,
    budget, ``amp=False`` and the pinned values). Every *tunable*
    hyperparameter (``lr0``, ``lrf``, ``momentum``, ``weight_decay``, warm-up,
    loss gains, augmentation) is left at its Ultralytics default.

    Args:
        data: Path to the ``data.yaml``.
        device: Device specification produced by :func:`parse_device`.
        epochs: Training budget. Override only for smoke tests, and then
            identically for Phases 1, 2 and 4 (``run_pipeline.sh`` does this).
        patience: Early-stopping patience (same rule as ``epochs``).

    Returns:
        Keyword arguments for ``YOLO.train()`` (without ``project``/``name``).
    """
    return dict(
        data=data,
        task="segment",
        pretrained=True,
        imgsz=IMGSZ,
        device=device,
        batch=BATCH,
        nbs=NBS,
        workers=WORKERS,
        cache=False,
        epochs=epochs,
        patience=patience,
        amp=False,
        deterministic=True,
        seed=SEED,
        **BASE_RECIPE,
        save=True,
        plots=True,
        val=True,
        verbose=True,
    )


def optimized_protocol(
    data: str,
    device: DeviceArg,
    tuned_hp: dict[str, Any] | None = None,
    epochs: int = TRAIN_EPOCHS,
    patience: int = TRAIN_PATIENCE,
) -> dict[str, Any]:
    """Return the Phase 4 training kwargs: the baseline protocol + tuned HPs.

    Args:
        data: Path to the ``data.yaml``.
        device: Device specification produced by :func:`parse_device`.
        tuned_hp: Hyperparameters from Phase 3 (``best_hyperparameters.yaml``).
        epochs: Training budget (see :func:`baseline_protocol`).
        patience: Early-stopping patience (see :func:`baseline_protocol`).

    Returns:
        Keyword arguments for ``YOLO.train()`` (without ``project``/``name``).

    Raises:
        ValueError: If ``tuned_hp`` tries to override a protected key, which
            would change the base setup, budget or reproducibility contract.
    """
    tuned_hp = dict(tuned_hp or {})
    clash = PROTECTED_KEYS.intersection(tuned_hp)
    if clash:
        raise ValueError(
            f"Tuned hyperparameters must not override protected keys: {sorted(clash)}",
        )
    return {**baseline_protocol(data, device, epochs, patience), **tuned_hp}


def hpo_trial_protocol(
    data: str,
    device: DeviceArg,
    epochs: int,
    patience: int,
    batch: int,
) -> dict[str, Any]:
    """Return the fixed (non-searched) kwargs of every Phase 3 HPO trial.

    Identical to :func:`optimized_protocol` except for the short per-trial
    budget and the micro-batch. With ``nbs=64`` the effective optimisation
    batch stays at 64 for any ``batch`` (``accumulate = round(nbs / batch)``).

    Args:
        data: Path to the ``data.yaml``.
        device: Device specification produced by :func:`parse_device`.
        epochs: Epochs per trial.
        patience: Early-stopping patience per trial.
        batch: Micro-batch per trial.

    Returns:
        Keyword arguments for the Ultralytics ``Tuner`` (without
        ``project``/``name``).
    """
    kw = optimized_protocol(data, device)
    kw.update(epochs=epochs, patience=patience, batch=batch, plots=False, verbose=False)
    return kw


# ----------------------------------------------------------------------------
# Output layout
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class PipelinePaths:
    """Canonical output layout of one pipeline run rooted at ``root``."""

    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))

    # ---- Phase 1 — baseline ------------------------------------------------
    @property
    def phase1_dir(self) -> Path:
        return self.root / "phase1_baseline"

    @staticmethod
    def phase1_run_name(model: str) -> str:
        return f"yolo26_{model}_baseline"

    def phase1_best_pt(self, model: str) -> Path:
        return self.phase1_dir / self.phase1_run_name(model) / "weights" / "best.pt"

    # ---- Phase 2 — cross-validation ----------------------------------------
    def cv_dir(self, protocol: str = "baseline") -> Path:
        """CV root for a protocol; Phase 2 is ``baseline``, ``optimized`` is optional."""
        return self.root / f"phase2_cv_{protocol}"

    def cv_model_dir(self, model: str, protocol: str = "baseline") -> Path:
        return self.cv_dir(protocol) / f"yolo26_{model}"

    # ---- Phase 3 — HPO -----------------------------------------------------
    @property
    def phase3_dir(self) -> Path:
        return self.root / "phase3_hpo"

    @staticmethod
    def phase3_tune_name(model: str) -> str:
        return f"tune_{model}"

    def phase3_tune_dir(self, model: str) -> Path:
        return self.phase3_dir / self.phase3_tune_name(model)

    def phase3_best_yaml(self, model: str) -> Path:
        return self.phase3_tune_dir(model) / "best_hyperparameters.yaml"

    def phase3_results_csv(self, model: str) -> Path:
        return self.phase3_tune_dir(model) / "tune_results.csv"

    def phase3_state(self, model: str) -> Path:
        return self.phase3_tune_dir(model) / "hpo_state.json"

    # ---- Phase 4 — optimised fine-tune -------------------------------------
    @property
    def phase4_dir(self) -> Path:
        return self.root / "phase4_optimized"

    @staticmethod
    def phase4_run_name(model: str) -> str:
        return f"yolo26_{model}_optimized"

    def phase4_best_pt(self, model: str) -> Path:
        return self.phase4_dir / self.phase4_run_name(model) / "weights" / "best.pt"

    # ---- Phase 5 & summaries -----------------------------------------------
    def best_pt(self, variant: str, model: str) -> Path:
        """Weights evaluated in Phase 5: ``baseline`` (Phase 1) or ``optimized`` (Phase 4)."""
        if variant == "baseline":
            return self.phase1_best_pt(model)
        if variant == "optimized":
            return self.phase4_best_pt(model)
        raise ValueError(f"unknown variant {variant!r}")

    @property
    def phase5_dir(self) -> Path:
        return self.root / "phase5_test"

    @staticmethod
    def phase5_tag(variant: str, model: str, precision: str) -> str:
        return f"{variant}_{model}_{precision}"

    def phase5_accuracy_json(self, variant: str, model: str, precision: str) -> Path:
        return self.phase5_dir / "accuracy" / f"{self.phase5_tag(variant, model, precision)}.json"

    def phase5_per_image_csv(self, variant: str, model: str, precision: str) -> Path:
        return self.phase5_dir / "per_image" / f"{self.phase5_tag(variant, model, precision)}.csv"

    def phase5_mask_dir(self, variant: str, model: str) -> Path:
        """Predicted test masks (FP32 only) for the visualisation notebook."""
        return self.phase5_dir / "masks" / f"{variant}_{model}"

    def phase5_efficiency_json(self, variant: str, model: str, precision: str) -> Path:
        return self.phase5_dir / "efficiency" / f"{self.phase5_tag(variant, model, precision)}.json"

    @property
    def summary_dir(self) -> Path:
        return self.root / "summary"

    @property
    def pipeline_runs_dir(self) -> Path:
        return self.root / "pipeline_runs"


# ----------------------------------------------------------------------------
# Small I/O helpers
# ----------------------------------------------------------------------------
def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string (second precision)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write ``payload`` as JSON so that ``path`` is never left half-written.

    The data is written to a temporary sibling, flushed and ``fsync``-ed, then
    moved into place with ``os.replace`` (atomic on POSIX). A crash at any
    point leaves either the previous or the new file, never a truncated one.

    Args:
        path: Destination file.
        payload: JSON-serialisable mapping.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_json(path: Path) -> dict[str, Any] | None:
    """Return the parsed JSON at ``path``, or ``None`` if it does not exist."""
    path = Path(path)
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def config_hash(payload: Any) -> str:
    """Return a short, stable SHA-256 fingerprint of a JSON-serialisable config."""
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


@contextmanager
def exclusive_lock(directory: Path, name: str = ".lock") -> Iterator[None]:
    """Hold a non-blocking exclusive POSIX lock (``lockf``) on ``directory/name``.

    Prevents two processes from writing the same run (e.g. a manual re-run
    while the orchestrator is still training).

    ``lockf`` (not ``flock``): a POSIX record lock belongs to the *process*,
    is not inherited by ``fork()``-ed children and is released as soon as the
    process dies. With ``flock`` the lock belongs to the open file, which
    forked DataLoader workers inherit; after a ``kill -9`` of the trainer the
    orphaned workers kept the run locked and an immediate restart was refused
    until they exited.

    Raises:
        RuntimeError: If another process already holds the lock.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / name).open("w") as fh:
        try:
            fcntl.lockf(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            raise RuntimeError(f"another process is already working in {directory}") from e
        try:
            yield
        finally:
            fcntl.lockf(fh, fcntl.LOCK_UN)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """Return the SHA-256 of a file (used to tie Phase 5 results to exact weights)."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()
