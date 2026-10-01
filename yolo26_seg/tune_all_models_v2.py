"""Phase 3 — Fault-tolerant, reproducible HPO of YOLO26-seg on ISIC 2018 Task 1.

The search is the Ultralytics genetic-algorithm Tuner (BLX-α crossover over
the top-9 parents + Gaussian mutation), kept for methodological continuity
with the earlier HPO runs. Two changes make it suitable for the thesis:

Reproducibility (:class:`SeededTuner`)
    The upstream ``Tuner._mutate`` re-seeds NumPy with ``int(time.time())``,
    so the proposed hyperparameters differ on every run. The subclass
    re-implements the same mutation logic with a private
    ``np.random.default_rng([seed, trial_index])``: the proposal for trial
    ``i`` is a pure function of ``(seed, i, fitness history)``. A resumed or
    retried trial therefore receives exactly the same hyperparameters.

Fault tolerance (:func:`tune_one_model`)
    * The Tuner always runs with ``resume=True`` in a fixed directory, so it
      continues from row ``N + 1`` of ``tune_results.csv``.
    * ``hpo_state.json`` is checkpointed atomically at the start of every
      trial (completed trials, best fitness, in-flight trial, config hash).
      Completion is ``completed_trials == --iterations``, never "the YAML
      exists" (Ultralytics rewrites ``best_hyperparameters.yaml`` after every
      trial, so its presence says nothing about completion).
    * On resume the folder of the interrupted trial is deleted and a partially
      written CSV line is discarded.
    * A failed trial (``fitness == 0``: crash, OOM, NaN) is removed from the
      CSV and retried with the same hyperparameters, up to
      ``--max-trial-retries`` times; after that it is kept as a genuine
      failure. Failures that coincide with an unhealthy GPU are not counted.
    * If the GPU/driver is unhealthy the script exits with code
      :data:`EXIT_GPU_UNAVAILABLE` (75) so the orchestrator can retry later.
    * Resuming with a different search space / fixed protocol / seed / weights
      is refused (config-hash mismatch), so trials from different
      configurations are never mixed.

Every trial uses :func:`common.hpo_trial_protocol`, i.e. the Phase 4 recipe
with a shorter budget, so the HPs are selected under the same conditions they
are later trained with. Only the ``train``/``val`` splits are used.

Outputs (per model)::

    <project>/phase3_hpo/tune_<model>/
    ├── tune_results.csv            # one row per trial (fitness + genes)
    ├── best_hyperparameters.yaml   # consumed by Phase 4
    ├── hpo_state.json              # checkpoint (see above)
    ├── weights/                    # weights of the best trial
    └── trials/train*/              # per-trial Ultralytics runs

Exit codes:
    0 — every requested model is complete; 1 — at least one model failed;
    75 — GPU/driver unavailable (retry once the host is healthy).

Usage:
    # All five sizes, refined space (the canonical pipeline call)::

        python tune_all_models_v2.py --project /workspace/logs/pipeline_final_v1 \\
            --space refined --iterations 30

    # Re-run the same command after a crash: it resumes automatically.

    # xlarge in FP32 on 32 GB GPUs (nbs=64 keeps the effective batch at 64)::

        python tune_all_models_v2.py --models xlarge --batch 16

    # Extend a finished 30-trial search to 50 trials (same config)::

        python tune_all_models_v2.py --models small --iterations 50

    # Start over (the previous tune dir is moved to tune_<m>.bak-<UTC>)::

        python tune_all_models_v2.py --models small --force
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import ultralytics
from ultralytics import YOLO
from ultralytics.engine.tuner import Tuner

from common import (
    DEFAULT_DATA_YAML,
    DEFAULT_ORDER,
    DEFAULT_PIPELINE_ROOT,
    PROTECTED_KEYS,
    SEED,
    WEIGHTS,
    DeviceArg,
    PipelinePaths,
    atomic_write_json,
    config_hash,
    exclusive_lock,
    hpo_trial_protocol,
    parse_device,
    read_json,
    seed_everything,
    utc_now_iso,
)

# ----------------------------------------------------------------------------
# Module-level configuration
# ----------------------------------------------------------------------------
#: Exit code signalling a transient infrastructure failure (``EX_TEMPFAIL``).
EXIT_GPU_UNAVAILABLE: int = 75

#: Version of the ``hpo_state.json`` schema.
STATE_SCHEMA: int = 1

#: Mutation constants mirrored from ``ultralytics.engine.tuner.Tuner``
#: (8.4.x). Stored in the config hash so a change is detected on resume.
GA_TOP_N: int = 9
GA_MUTATION_PROB: float = 0.5
GA_CROSSOVER_ALPHA: float = 0.2


# ----------------------------------------------------------------------------
# Search spaces
# ----------------------------------------------------------------------------
#: Broad search space — initial exploration of the optimisation landscape.
#: Each entry is either ``(min, max)`` or ``(min, max, gain)``. Gain controls
#: the amplitude of the genetic algorithm's Gaussian mutation (default 1.0;
#: higher = more aggressive).
SEARCH_SPACE_WIDE: dict[str, tuple] = {
    # ---- Learning ----
    "lr0":            (5e-4, 5e-3),
    "lrf":            (0.005, 0.05),
    "momentum":       (0.85, 0.95, 0.3),
    "weight_decay":   (0.0, 0.001),
    "warmup_epochs":  (1.0, 5.0),
    "warmup_momentum": (0.5, 0.95),

    # ---- Loss weights ----
    "box":            (3.0, 12.0),
    "cls":            (0.2, 1.5),
    "dfl":            (0.8, 3.0),

    # ---- Colour augmentation ----
    "hsv_h":          (0.0, 0.03),
    "hsv_s":          (0.3, 0.9),
    "hsv_v":          (0.2, 0.7),

    # ---- Geometric augmentation ----
    "degrees":        (0.0, 30.0),
    "translate":      (0.0, 0.3),
    "scale":          (0.2, 0.7),
    "shear":          (0.0, 10.0),
    "fliplr":         (0.0, 0.6),
    "flipud":         (0.0, 0.5),

    # ---- Mixing augmentation ----
    "mosaic":         (0.5, 1.0),
    "mixup":          (0.0, 0.3),
    "copy_paste":     (0.0, 0.3),
    "cutmix":         (0.0, 0.3),
}

#: Refined search space — narrowed follow-up around high-signal HPs.
#: HPs with ``|r| < 0.05`` against fitness (in the wide HPO of ``small``) were
#: dropped and pinned to the Ultralytics default; high-signal HPs had their
#: ranges shrunk around the empirically winning region.
SEARCH_SPACE_REFINED: dict[str, tuple] = {
    # ---- Learning ----
    "lr0":            (1e-3, 4e-3),
    "lrf":            (0.005, 0.05),
    "momentum":       (0.85, 0.95, 0.3),
    "weight_decay":   (1e-6, 1e-4),
    "warmup_epochs":  (1.0, 5.0),

    # ---- Loss weights ----
    "cls":            (0.2, 1.5),
    "dfl":            (0.8, 1.5),

    # ---- Colour augmentation ----
    "hsv_h":          (0.005, 0.025),
    "hsv_s":          (0.3, 0.9),
    "hsv_v":          (0.2, 0.7),

    # ---- Geometric augmentation ----
    "translate":      (0.05, 0.20),
    "flipud":         (0.0, 0.10),

    # ---- Mixing augmentation ----
    "mosaic":         (0.7, 1.0),
    "mixup":          (0.0, 0.05),
    "copy_paste":     (0.0, 0.05),
}


def get_search_space(name: str) -> dict[str, tuple]:
    """Return the search-space mapping for the requested preset.

    Args:
        name: Either ``"wide"`` or ``"refined"``.

    Returns:
        The dictionary describing the genetic-algorithm search space.

    Raises:
        ValueError: If ``name`` is neither ``"wide"`` nor ``"refined"``.
    """
    if name == "wide":
        return SEARCH_SPACE_WIDE
    if name == "refined":
        return SEARCH_SPACE_REFINED
    raise ValueError(f"--space invalid: {name!r}. Use 'wide' or 'refined'.")


# ----------------------------------------------------------------------------
# tune_results.csv helpers
# ----------------------------------------------------------------------------
def read_tune_csv(path: Path) -> tuple[str | None, list[str]]:
    """Return ``(header, data_lines)`` of a ``tune_results.csv`` (raw text).

    Lines are kept verbatim so a rewrite never alters the stored floats.
    """
    if not path.exists():
        return None, []
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return None, []
    return lines[0], [ln for ln in lines[1:] if ln.strip()]


def write_tune_csv(path: Path, header: str, lines: list[str]) -> None:
    """Atomically rewrite ``tune_results.csv``; delete it when ``lines`` is empty.

    An empty file is removed (rather than left header-only) so the Tuner
    starts from a clean slate and writes its own header.
    """
    if not lines:
        path.unlink(missing_ok=True)
        return
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text("\n".join([header, *lines]) + "\n", encoding="utf-8")
    tmp.replace(path)


def parse_row(line: str, n_cols: int) -> list[float] | None:
    """Parse one CSV row into floats; ``None`` if malformed (e.g. torn write)."""
    parts = line.split(",")
    if len(parts) != n_cols:
        return None
    try:
        return [float(v) for v in parts]
    except ValueError:
        return None


def rows_to_array(lines: list[str], n_cols: int) -> np.ndarray:
    """Convert CSV data lines into a ``(n_rows, n_cols)`` float array."""
    rows = [parse_row(ln, n_cols) for ln in lines]
    rows = [r for r in rows if r is not None]
    return np.array(rows, dtype=float).reshape(len(rows), n_cols)


# ----------------------------------------------------------------------------
# Seeded Tuner
# ----------------------------------------------------------------------------
class TrialFailed(RuntimeError):
    """Raised by :class:`SeededTuner` when the trial just finished had fitness 0."""

    def __init__(self, trial_index: int) -> None:
        super().__init__(f"trial {trial_index + 1} failed (fitness=0)")
        self.trial_index = trial_index


class SeededTuner(Tuner):
    """Ultralytics ``Tuner`` with a deterministic, resume-safe mutation RNG.

    Differences from the upstream class (everything else is inherited):

    * :meth:`_mutate` uses ``np.random.default_rng([seed, trial_index])``
      instead of ``np.random.seed(int(time.time()))`` and the global
      ``random`` module. The crossover/mutation maths is unchanged.
    * Per-trial training runs are written to ``<tune_dir>/trials/`` instead
      of next to the tune directory, isolating each model's trials.
    * :meth:`_mutate` doubles as the per-trial checkpoint hook: it aborts the
      run with :class:`TrialFailed` as soon as a failed trial is observed
      (so it can be retried before it influences later proposals) and then
      checkpoints ``hpo_state.json`` before the next trial starts.
    """

    def __init__(self, args: dict, _callbacks: Any, seed: int, checkpoint: HPOCheckpoint) -> None:
        super().__init__(args=args, _callbacks=_callbacks)
        if self.mongodb:
            raise RuntimeError("SeededTuner does not support the MongoDB backend.")
        self.seed = seed
        self.checkpoint = checkpoint
        self.trials_dir = self.tune_dir / "trials"
        self.args.project = str(self.trials_dir)
        self.n_cols = 1 + len(self.space)
        _, lines = read_tune_csv(self.tune_csv)
        self._rows_at_start = len(lines)

    @staticmethod
    def _seeded_crossover(x: np.ndarray, rng: np.random.Generator, k: int = 9) -> np.ndarray:
        """BLX-α crossover from up to top-k parents (upstream logic, seeded RNG)."""
        k = min(k, len(x))
        weights = x[:, 0] - x[:, 0].min() + 1e-6
        if not np.isfinite(weights).all() or weights.sum() == 0:
            weights = np.ones_like(weights)
        idxs = rng.choice(len(x), size=k, replace=True, p=weights / weights.sum())
        parents = x[idxs, 1:]
        lo, hi = parents.min(0), parents.max(0)
        span = hi - lo
        span = np.where(span == 0, rng.uniform(0.01, 0.1, span.shape), span)
        alpha = GA_CROSSOVER_ALPHA
        return rng.uniform(lo - alpha * span, hi + alpha * span)

    def _mutate(
        self,
        n: int = GA_TOP_N,
        mutation: float = GA_MUTATION_PROB,
        sigma: float = 0.2,
    ) -> dict[str, float]:
        """Propose the hyperparameters of the next trial (deterministically).

        Args:
            n: Number of top parents to consider.
            mutation: Probability of mutating each gene.
            sigma: Standard deviation of the Gaussian mutation step.

        Returns:
            Mapping ``{hp_name: value}`` for the next trial.

        Raises:
            TrialFailed: If a trial run by this instance ended with fitness 0.
        """
        _, lines = read_tune_csv(self.tune_csv)
        if len(lines) > self._rows_at_start:
            last = parse_row(lines[-1], self.n_cols)
            if last is None or last[0] <= 0:
                raise TrialFailed(len(lines) - 1)

        x = rows_to_array(lines, self.n_cols)
        trial_index = len(x)
        self.checkpoint.on_trial_start(trial_index, x, self.trials_dir)

        keys = list(self.space.keys())
        if len(x):
            rng = np.random.default_rng([self.seed, trial_index])
            order = np.argsort(-x[:, 0], kind="stable")
            parents = x[order][:n]
            genes = self._seeded_crossover(parents, rng, k=n)
            gains = np.array([v[2] if len(v) == 3 else 1.0 for v in self.space.values()])
            factors = np.ones(len(keys))
            while np.all(factors == 1):  # mutate until a change occurs
                mask = rng.random(len(keys)) < mutation
                step = rng.standard_normal(len(keys)) * (sigma * gains)
                factors = np.where(mask, np.exp(step), 1.0).clip(0.25, 4.0)
            hyp = {k: float(genes[i] * factors[i]) for i, k in enumerate(keys)}
        else:
            # First trial: the default HPs (as upstream), clipped to the bounds below.
            hyp = {k: getattr(self.args, k) for k in keys}

        for k, bounds in self.space.items():
            hyp[k] = round(min(max(hyp[k], bounds[0]), bounds[1]), 5)
        if "close_mosaic" in hyp:
            hyp["close_mosaic"] = round(hyp["close_mosaic"])
        if "epochs" in hyp:
            hyp["epochs"] = round(hyp["epochs"])
        return hyp


# ----------------------------------------------------------------------------
# Checkpoint state
# ----------------------------------------------------------------------------
class HPOCheckpoint:
    """Owns ``hpo_state.json`` for one model; every write is atomic.

    State fields:
        ``status`` (``running``/``complete``), ``target_trials``,
        ``completed_trials``, ``valid_trials``, ``best_fitness``,
        ``best_trial`` (1-based), ``in_flight_trial`` (1-based or ``None``),
        ``known_trial_dirs`` (trial folders that existed when the in-flight
        trial started), ``failed_attempts`` (trial → failed attempts),
        ``accepted_failures`` (trials kept with fitness 0), ``config_hash``,
        ``config``, ``ultralytics_version`` and an event ``history``.
    """

    def __init__(self, path: Path, state: dict[str, Any]) -> None:
        self.path = path
        self.state = state

    @classmethod
    def load_or_create(
        cls,
        path: Path,
        model: str,
        config: dict[str, Any],
        target_trials: int,
    ) -> HPOCheckpoint:
        """Load an existing checkpoint or create a fresh one (not yet saved)."""
        state = read_json(path)
        if state is None:
            state = {
                "schema": STATE_SCHEMA,
                "model": model,
                "status": "running",
                "target_trials": target_trials,
                "completed_trials": 0,
                "valid_trials": 0,
                "best_fitness": None,
                "best_trial": None,
                "in_flight_trial": None,
                "known_trial_dirs": [],
                "failed_attempts": {},
                "accepted_failures": [],
                "config_hash": config_hash(config),
                "config": config,
                "ultralytics_version": ultralytics.__version__,
                "created_at": utc_now_iso(),
                "last_update": utc_now_iso(),
                "history": [],
            }
        return cls(path, state)

    def save(self) -> None:
        """Persist the state atomically."""
        self.state["last_update"] = utc_now_iso()
        atomic_write_json(self.path, self.state)

    def log_event(self, event: str, **info: Any) -> None:
        """Append an event to the history and persist."""
        self.state["history"].append({"at": utc_now_iso(), "event": event, **info})
        self.save()

    def update_progress(self, x: np.ndarray) -> None:
        """Refresh completed/valid/best counters from the trial array."""
        self.state["completed_trials"] = int(len(x))
        valid = x[x[:, 0] > 0] if len(x) else x
        self.state["valid_trials"] = int(len(valid))
        if len(valid):
            best = int(np.argmax(x[:, 0]))
            self.state["best_fitness"] = float(x[best, 0])
            self.state["best_trial"] = best + 1

    def snapshot_trial_dirs(self, trials_dir: Path) -> None:
        """Record the trial folders that currently exist (all belong to finished trials)."""
        self.state["known_trial_dirs"] = (
            sorted(d.name for d in trials_dir.iterdir() if d.is_dir())
            if trials_dir.exists() else []
        )

    def on_trial_start(self, trial_index: int, x: np.ndarray, trials_dir: Path) -> None:
        """Checkpoint hook called by :class:`SeededTuner` before each trial."""
        self.update_progress(x)
        self.snapshot_trial_dirs(trials_dir)
        self.state["status"] = "running"
        self.state["in_flight_trial"] = trial_index + 1
        self.save()
        print(
            f"  [ckpt] trial {trial_index + 1}/{self.state['target_trials']} starting "
            f"— {self.state['valid_trials']} valid so far, "
            f"best={self.state['best_fitness']}",
            flush=True,
        )


# ----------------------------------------------------------------------------
# Resume / sanitisation
# ----------------------------------------------------------------------------
def remove_orphan_trial_dirs(ckpt: HPOCheckpoint, trials_dir: Path) -> list[str]:
    """Delete trial folders created after the last checkpoint (interrupted/failed trial).

    Only direct children of ``trials_dir`` that were absent from the last
    ``known_trial_dirs`` snapshot are removed.

    Returns:
        Names of the removed folders.
    """
    if not trials_dir.exists():
        return []
    known = set(ckpt.state["known_trial_dirs"])
    removed = []
    for d in sorted(trials_dir.iterdir()):
        if d.is_dir() and d.name not in known:
            shutil.rmtree(d)
            removed.append(d.name)
    return removed


def sanitize_results(
    ckpt: HPOCheckpoint,
    csv_path: Path,
    expected_header: str,
    max_trial_retries: int,
    count_attempt: bool,
) -> None:
    """Make ``tune_results.csv`` safe to resume from.

    * Refuses a CSV whose header does not match the search space.
    * Drops a torn (partially written) last line.
    * Removes the trailing failed trial so it is retried, unless it already
      failed more than ``max_trial_retries`` times, in which case it is kept
      as a genuine failure. Failed rows that are not the last row (only
      possible in CSVs not produced by this driver) are kept, with a warning,
      because the later trials were derived from them.

    Args:
        ckpt: Checkpoint of the model being tuned.
        csv_path: Path to ``tune_results.csv``.
        expected_header: ``"fitness,<hp1>,<hp2>,..."`` for the current space.
        max_trial_retries: Retries allowed per failed trial.
        count_attempt: Whether this failure counts towards the retry budget
            (``False`` when the GPU was unhealthy, i.e. not the trial's fault).

    Raises:
        RuntimeError: On a header mismatch.
    """
    header, lines = read_tune_csv(csv_path)
    if header is None:
        return
    if header.strip() != expected_header:
        raise RuntimeError(
            f"{csv_path} header does not match the current search space.\n"
            f"  found   : {header}\n  expected: {expected_header}\n"
            f"Use --force to start a new search.",
        )
    n_cols = len(expected_header.split(","))
    original = list(lines)

    if lines and parse_row(lines[-1], n_cols) is None:
        print(f"  [resume] dropping torn last line of {csv_path.name}")
        lines = lines[:-1]

    accepted = set(ckpt.state["accepted_failures"])
    for j, line in enumerate(lines):
        trial = j + 1
        row = parse_row(line, n_cols)
        if row is None:
            raise RuntimeError(f"{csv_path}: malformed row for trial {trial}: {line!r}")
        if row[0] > 0 or trial in accepted:
            continue
        if j < len(lines) - 1:
            print(f"  [warn] trial {trial} has fitness 0 but later trials exist — keeping it")
            accepted.add(trial)
            continue
        attempts = ckpt.state["failed_attempts"]
        if count_attempt:
            attempts[str(trial)] = attempts.get(str(trial), 0) + 1
        n_failed = attempts.get(str(trial), 0)
        if n_failed > max_trial_retries:
            print(f"  [resume] trial {trial} failed {n_failed}x — accepting it as a failed trial")
            accepted.add(trial)
            ckpt.log_event("trial_failure_accepted", trial=trial, attempts=n_failed)
        else:
            print(
                f"  [resume] trial {trial} failed (attempt {n_failed}/{max_trial_retries + 1}"
                f"{'' if count_attempt else ', not counted: GPU unhealthy'}) — will retry",
            )
            lines = lines[:j]
            ckpt.log_event("trial_failed", trial=trial, attempts=n_failed, counted=count_attempt)

    ckpt.state["accepted_failures"] = sorted(accepted)
    if lines != original:
        backup = csv_path.with_name(f"{csv_path.name}.bak-{_utc_stamp()}")
        shutil.copy2(csv_path, backup)
        write_tune_csv(csv_path, header, lines)
    ckpt.save()


# ----------------------------------------------------------------------------
# Infrastructure helpers
# ----------------------------------------------------------------------------
def _utc_stamp() -> str:
    """Return a compact UTC timestamp for backup file names."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def gpu_healthy(device: DeviceArg) -> bool:
    """Return ``True`` if the NVIDIA driver and ``torch.cuda`` respond (fresh subprocess).

    A subprocess is used because the parent's CUDA state cannot detect a
    driver that died after initialisation.
    """
    if device == "cpu":
        return True
    probe = (
        "import sys, torch; "
        "sys.exit(0 if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 1)"
    )
    try:
        subprocess.run(["nvidia-smi", "-L"], check=True, capture_output=True, timeout=60)
        subprocess.run([sys.executable, "-c", probe], check=True, capture_output=True, timeout=180)
    except (OSError, subprocess.SubprocessError):
        return False
    return True


class GPUUnavailable(RuntimeError):
    """Raised when the GPU/driver is unhealthy; maps to :data:`EXIT_GPU_UNAVAILABLE`."""


# ----------------------------------------------------------------------------
# CLI parsing
# ----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the Phase 3 orchestrator.

    Returns:
        The parsed ``argparse.Namespace``.
    """
    p = argparse.ArgumentParser(
        description="Phase 3 — fault-tolerant, seeded HPO of the YOLO26-seg variants.",
    )
    p.add_argument(
        "--models", nargs="+", default=DEFAULT_ORDER, choices=DEFAULT_ORDER,
        help=f"Subset of models to tune (default: {DEFAULT_ORDER}).",
    )
    p.add_argument(
        "--iterations", type=int, default=30,
        help="Target number of trials per model (default: 30). Raising it later extends the search.",
    )
    p.add_argument(
        "--epochs", type=int, default=30,
        help="Epochs per trial (default: 30).",
    )
    p.add_argument(
        "--patience", type=int, default=10,
        help="Early-stopping patience per trial (default: 10).",
    )
    p.add_argument(
        "--batch", type=int, default=32,
        help=(
            "Micro-batch per trial (default: 32). Use 16 for xlarge in FP32 on "
            "32 GB GPUs — nbs=64 keeps the effective optimisation batch at 64."
        ),
    )
    p.add_argument(
        "--space", choices=["wide", "refined"], default="refined",
        help="Search space preset (default: refined).",
    )
    p.add_argument(
        "--seed", type=int, default=SEED,
        help=f"Seed of the mutation RNG and of every trial (default: {SEED}).",
    )
    p.add_argument(
        "--max-trial-retries", type=int, default=2,
        help="Retries for a failed trial before it is accepted as failed (default: 2).",
    )
    p.add_argument("--data", default=DEFAULT_DATA_YAML, help="Path to the data.yaml.")
    p.add_argument(
        "--device", default="0,1",
        help="GPU IDs (default: '0,1' DDP). Use '0' for single-GPU.",
    )
    p.add_argument(
        "--project", default=DEFAULT_PIPELINE_ROOT,
        help=f"Pipeline root (default: {DEFAULT_PIPELINE_ROOT}). Outputs go to <project>/phase3_hpo/.",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Start over: move an existing tune_<model>/ to tune_<model>.bak-<UTC> first.",
    )
    p.add_argument(
        "--allow-version-change", action="store_true",
        help="Allow resuming a search started with a different Ultralytics version.",
    )
    return p.parse_args()


# ----------------------------------------------------------------------------
# Per-model tuning
# ----------------------------------------------------------------------------
def _hashed_config(
    model_size: str,
    args: argparse.Namespace,
    space: dict[str, tuple],
    fixed: dict[str, Any],
) -> dict[str, Any]:
    """Build the configuration that must be identical to resume a search.

    Hardware-only / cosmetic knobs (``device``, ``verbose``, ``plots``) are
    excluded so a search can resume on a different GPU set.
    """
    return {
        "model": model_size,
        "weights": Path(WEIGHTS[model_size]).name,
        "space": {k: list(v) for k, v in space.items()},
        "fixed": {k: v for k, v in fixed.items() if k not in {"device", "verbose", "plots"}},
        "seed": args.seed,
        "ga": {"top_n": GA_TOP_N, "mutation": GA_MUTATION_PROB, "alpha": GA_CROSSOVER_ALPHA},
    }


def _check_resumable(ckpt: HPOCheckpoint, config: dict[str, Any], allow_version_change: bool) -> None:
    """Refuse to resume when the configuration or Ultralytics version changed."""
    state = ckpt.state
    if state["config_hash"] != config_hash(config):
        raise RuntimeError(
            f"Configuration changed since this search started ({ckpt.path}).\n"
            f"  stored : {state['config_hash']}\n  current: {config_hash(config)}\n"
            f"Resuming would mix trials from different configurations. "
            f"Revert the change or use --force to start a new search.",
        )
    if state["ultralytics_version"] != ultralytics.__version__:
        msg = (
            f"Ultralytics version changed ({state['ultralytics_version']} → "
            f"{ultralytics.__version__}) since this search started."
        )
        if not allow_version_change:
            raise RuntimeError(msg + " Pass --allow-version-change to resume anyway.")
        print(f"  [warn] {msg}")


def _backup_tune_dir(tune_dir: Path) -> None:
    """Move an existing tune directory aside (``--force``)."""
    if tune_dir.exists():
        backup = tune_dir.with_name(f"{tune_dir.name}.bak-{_utc_stamp()}")
        tune_dir.rename(backup)
        print(f"  [force] moved previous search to {backup}")


def tune_one_model(
    model_size: str,
    args: argparse.Namespace,
    device: DeviceArg,
    space: dict[str, tuple],
    paths: PipelinePaths,
) -> dict:
    """Run (or resume) the HPO of a single variant until it is complete.

    Args:
        model_size: Variant name (one of :data:`DEFAULT_ORDER`).
        args: Parsed CLI arguments.
        device: Device specification produced by :func:`parse_device`.
        space: Search-space mapping from :func:`get_search_space`.
        paths: Output layout of this pipeline run.

    Returns:
        Summary dict with keys ``model``, ``skipped``, ``reason``,
        ``elapsed_min``, ``best_fitness`` and ``valid_trials``.

    Raises:
        GPUUnavailable: If the GPU/driver is unhealthy.
    """
    clash = PROTECTED_KEYS.intersection(space)
    if clash:  # the base setup must stay identical to the Baseline (apples-to-apples)
        raise ValueError(f"search space contains protected base-setup keys: {sorted(clash)}")
    tune_dir = paths.phase3_tune_dir(model_size)
    csv_path = paths.phase3_results_csv(model_size)
    trials_dir = tune_dir / "trials"
    fixed = hpo_trial_protocol(args.data, device, args.epochs, args.patience, args.batch)
    config = _hashed_config(model_size, args, space, fixed)
    expected_header = ",".join(["fitness", *space.keys()])

    if args.force:
        _backup_tune_dir(tune_dir)

    with exclusive_lock(tune_dir, ".hpo.lock"):
        ckpt = HPOCheckpoint.load_or_create(
            paths.phase3_state(model_size), model_size, config, args.iterations,
        )
        _check_resumable(ckpt, config, args.allow_version_change)
        ckpt.state["target_trials"] = args.iterations

        _, lines = read_tune_csv(csv_path)
        if ckpt.state["status"] == "complete" and len(lines) >= args.iterations:
            return {
                "model": model_size, "skipped": True,
                "reason": f"complete ({len(lines)}/{args.iterations} trials)",
                "elapsed_min": 0.0, "best_fitness": ckpt.state["best_fitness"],
                "valid_trials": ckpt.state["valid_trials"],
            }

        t0 = time.perf_counter()
        count_attempt = True
        ckpt.log_event("resume" if lines else "start", completed_trials=len(lines))
        while True:
            removed = remove_orphan_trial_dirs(ckpt, trials_dir)
            if removed:
                print(f"  [resume] removed interrupted/failed trial folder(s): {removed}")
            sanitize_results(ckpt, csv_path, expected_header, args.max_trial_retries, count_attempt)

            _, lines = read_tune_csv(csv_path)
            if len(lines) >= args.iterations:
                break
            if not gpu_healthy(device):
                ckpt.log_event("gpu_unavailable")
                raise GPUUnavailable(f"GPU/driver unhealthy before trial {len(lines) + 1}")

            print(f"  [run] trials {len(lines) + 1}..{args.iterations} (seed={args.seed})")
            model = YOLO(WEIGHTS[model_size])
            tuner_args = {
                **model.overrides,
                **fixed,
                "space": space,
                "project": str(paths.phase3_dir),
                "name": paths.phase3_tune_name(model_size),
                "resume": True,       # reuse tune_dir and continue from the CSV
                "mode": "train",
            }
            tuner = SeededTuner(tuner_args, model.callbacks, seed=args.seed, checkpoint=ckpt)
            if tuner.tune_dir.resolve() != tune_dir.resolve():
                raise RuntimeError(f"Tuner chose {tuner.tune_dir}, expected {tune_dir}")
            try:
                tuner(iterations=args.iterations, cleanup=True)
                _, lines = read_tune_csv(csv_path)
                last = parse_row(lines[-1], len(space) + 1) if lines else None
                failed = last is None or last[0] <= 0
                if not failed:
                    ckpt.snapshot_trial_dirs(trials_dir)  # last trial's folder is valid
            except TrialFailed as e:
                print(f"  [fail] {e}")
                failed = True
            # A failure is only charged to the trial when the GPU is healthy.
            count_attempt = gpu_healthy(device) if failed else True
            if not count_attempt:
                sanitize_results(ckpt, csv_path, expected_header, args.max_trial_retries, False)
                ckpt.log_event("gpu_unavailable")
                raise GPUUnavailable("GPU/driver became unhealthy during tuning")

        x = rows_to_array(read_tune_csv(csv_path)[1], len(space) + 1)
        ckpt.update_progress(x)
        best_yaml = paths.phase3_best_yaml(model_size)
        if not best_yaml.exists():
            raise RuntimeError(f"{best_yaml} missing after a complete search")
        ckpt.state.update(status="complete", in_flight_trial=None)
        ckpt.log_event("complete", completed_trials=len(x), best_fitness=ckpt.state["best_fitness"])

    return {
        "model": model_size, "skipped": False, "reason": None,
        "elapsed_min": (time.perf_counter() - t0) / 60,
        "best_fitness": ckpt.state["best_fitness"],
        "valid_trials": ckpt.state["valid_trials"],
    }


# ----------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------
def _print_run_header(
    args: argparse.Namespace,
    device: DeviceArg,
    space: dict[str, tuple],
    paths: PipelinePaths,
) -> None:
    """Print the top-level orchestration summary."""
    print(f"Phase 3 (HPO) for: {args.models}")
    print(f"  trials/model     = {args.iterations}   (max retries per failed trial = {args.max_trial_retries})")
    print(f"  epochs/trial     = {args.epochs}   patience = {args.patience}")
    print(f"  batch/trial      = {args.batch}  (nbs=64 — effective optim batch fixed at 64)")
    print(f"  search space     = {args.space!r} ({len(space)} hp)   seed = {args.seed}")
    print(f"  device           = {device}")
    print(f"  data             = {args.data}")
    print(f"  output           = {paths.phase3_dir}")
    print(f"  ultralytics      = {ultralytics.__version__}")
    print(f"  force restart    = {args.force}")


def _print_run_summary(
    summary: list[dict],
    failures: list[tuple[str, str]],
    total_min: float,
) -> None:
    """Print the final per-model summary table and total wall time."""
    print("\n" + "=" * 80)
    print("=== PHASE 3 (HPO) — SUMMARY")
    print("=" * 80)
    for s in summary:
        if s.get("failed"):
            status = f"FAILED ({s['reason']})"
        elif s["skipped"]:
            status = f"skipped — {s['reason']}"
        else:
            status = f"done in {s['elapsed_min']:.1f} min"
        extra = (
            f"  best={s['best_fitness']}  valid={s['valid_trials']}"
            if not s.get("failed") else ""
        )
        print(f"  {s['model']:8s} : {status}{extra}")
    print(f"\nTotal time: {total_min:.1f} min  ({total_min / 60:.2f} h)")
    if failures:
        print(f"\n{len(failures)} model(s) failed:", [m for m, _ in failures])


def main() -> int:
    """Run Phase 3 sequentially over the requested models.

    Returns:
        ``0`` on success, ``1`` if any model failed, :data:`EXIT_GPU_UNAVAILABLE`
        if the GPU/driver is unhealthy (remaining models are not attempted).
    """
    args = parse_args()
    seed_everything(args.seed)
    device = parse_device(args.device)
    space = get_search_space(args.space)
    paths = PipelinePaths(Path(args.project))
    _print_run_header(args, device, space, paths)

    summary: list[dict] = []
    failures: list[tuple[str, str]] = []
    t_total = time.perf_counter()
    exit_code = 0

    for i, m in enumerate(args.models, 1):
        print("\n" + "=" * 80)
        print(f"=== [{i}/{len(args.models)}] TUNE {m}  ({args.iterations} trials x {args.epochs} ep)")
        print("=" * 80)
        try:
            summary.append(tune_one_model(m, args, device, space, paths))
        except GPUUnavailable as e:
            print(f"  [GPU] {e} — stopping; re-run the same command once the host is healthy.",
                  file=sys.stderr)
            summary.append({"model": m, "skipped": False, "failed": True, "reason": str(e),
                            "elapsed_min": 0.0})
            failures.append((m, str(e)))
            exit_code = EXIT_GPU_UNAVAILABLE
            break
        except Exception as e:
            print(f"  [FAIL] {m}: {e}\n{traceback.format_exc()}", file=sys.stderr)
            summary.append({"model": m, "skipped": False, "failed": True, "reason": str(e),
                            "elapsed_min": 0.0})
            failures.append((m, str(e)))
            exit_code = 1

    _print_run_summary(summary, failures, (time.perf_counter() - t_total) / 60)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
