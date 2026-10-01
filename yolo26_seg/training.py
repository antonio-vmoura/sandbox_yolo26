"""Resumable training runs and metric parsing shared by Phases 1, 2 and 4.

:func:`train_or_resume` wraps ``YOLO.train()`` with the same fault-tolerance
contract as Phase 3:

* Completion is recorded in ``<run_dir>/run_state.json`` (atomic writes),
  never inferred from ``best.pt`` — Ultralytics writes ``best.pt`` at every
  improving epoch, so its presence does not mean the run finished.
* A run that was interrupted mid-training is resumed from ``last.pt`` with
  ``resume=True`` (optimiser, EMA and epoch counter are restored). The resume
  is logged in ``run_state.json`` because the dataloader RNG state is not
  part of the checkpoint, i.e. a resumed run is not bit-identical to an
  uninterrupted one.
* A run whose ``last.pt`` is already final (``epoch == -1`` after
  Ultralytics' ``strip_optimizer``) is only finalised, not retrained.
* Resuming or skipping is refused when the training protocol changed
  (protocol hash mismatch); ``--force`` moves the old run aside first.
* An exclusive lock (``<project>/.<name>.lock``) prevents two processes from
  training the same run concurrently.

``run_state.json`` fields: ``status`` (``running``/``complete``), ``phase``,
``model``, ``protocol`` / ``protocol_hash``, ``events`` (start / resume /
complete), ``metrics`` (validation metrics of the epoch that produced
``best.pt``, see :func:`parse_best_metrics`),
``epochs_trained``, ``batch_effective`` (the micro-batch the run really used,
see :func:`effective_batch`), ``best_pt``, ``ultralytics_version`` and
``torch_version``.
"""

from __future__ import annotations

import csv
import math
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from common import atomic_write_json, config_hash, exclusive_lock, read_json, utc_now_iso

#: File that records the lifecycle of one training run.
RUN_STATE_FILE: str = "run_state.json"

#: Short metric keys → Ultralytics ``results.csv`` columns (Box and Mask).
METRIC_KEYS: dict[str, str] = {
    "precision_b":  "metrics/precision(B)",
    "recall_b":     "metrics/recall(B)",
    "map50_b":      "metrics/mAP50(B)",
    "map5095_b":    "metrics/mAP50-95(B)",
    "precision_m":  "metrics/precision(M)",
    "recall_m":     "metrics/recall(M)",
    "map50_m":      "metrics/mAP50(M)",
    "map5095_m":    "metrics/mAP50-95(M)",
}

#: Short keys whose sum is the Ultralytics 8.4.21 segmentation fitness
#: (``SegmentMetrics.fitness`` = box ``Metric.fitness`` + mask ``Metric.fitness``,
#: each weighting ``[P, R, mAP50, mAP50-95]`` by ``[0, 0, 0, 1]``). This is the
#: criterion the trainer uses to overwrite ``best.pt`` and the HPO fitness.
FITNESS_KEYS: tuple[str, ...] = ("map5095_b", "map5095_m")

#: Protocol keys that do not affect the result and may differ on resume.
_HASH_EXCLUDED: frozenset[str] = frozenset({"device", "verbose", "plots"})


# ----------------------------------------------------------------------------
# Metrics / HP helpers
# ----------------------------------------------------------------------------
def ultralytics_fitness(metrics: dict[str, float]) -> float:
    """Ultralytics segmentation fitness: box mAP50-95 + mask mAP50-95."""
    return sum(metrics[k] for k in FITNESS_KEYS)


def _csv_precision(x: float) -> float:
    """Round like the trainer writes ``results.csv`` (``%.6g``)."""
    return float(f"{x:.6g}")


def _best_pt_metrics(best_pt: Path) -> dict[str, float] | None:
    """Validation metrics of the epoch that produced ``best.pt`` (``train_metrics``).

    Raises:
        RuntimeError: If the stored fitness no longer equals box + mask
            mAP50-95, i.e. the Ultralytics fitness definition changed.
    """
    from ultralytics.utils.patches import torch_load

    tm = torch_load(best_pt, map_location="cpu").get("train_metrics") or {}
    if not all(full in tm for full in METRIC_KEYS.values()):
        return None
    mine = {short: float(tm[full]) for short, full in METRIC_KEYS.items()}
    # The validator rounds every metric to 5 decimals, fitness included (computed
    # before rounding), so the sum of the rounded mAPs may differ by <= 1.5e-5.
    if "fitness" in tm and not math.isclose(ultralytics_fitness(mine), float(tm["fitness"]),
                                            rel_tol=0.0, abs_tol=2e-5):
        raise RuntimeError(
            f"{best_pt}: stored fitness {tm['fitness']} != box+mask mAP50-95 "
            f"{ultralytics_fitness(mine)} — the Ultralytics fitness definition changed; "
            f"update training.FITNESS_KEYS.",
        )
    return mine


def effective_batch(weights_pt: Path) -> int | None:
    """Micro-batch a run really trained with (``train_args.batch`` of its checkpoint).

    Ultralytics halves the batch after a CUDA out-of-memory error in the first
    epoch and retries (up to 3 times) with only a log warning; ``args.yaml`` is
    written before that and keeps the requested value, the checkpoint's
    ``train_args`` hold the value actually used.
    """
    from ultralytics.utils.patches import torch_load

    if not Path(weights_pt).exists():
        return None
    batch = (torch_load(weights_pt, map_location="cpu").get("train_args") or {}).get("batch")
    return int(batch) if batch is not None else None


def parse_best_metrics(results_csv: Path, best_pt: Path | None = None) -> dict[str, Any]:
    """Return the validation metrics of the epoch that produced ``best.pt``.

    The epoch is identified exactly as the Ultralytics trainer does:

    1. **From** ``best.pt`` (preferred): the checkpoint stores the validation
       metrics of the epoch that produced it (``train_metrics``); the
       ``results.csv`` row with identical metrics (at the CSV's ``%.6g``
       precision) is that epoch.
    2. **Fallback** (no ``best.pt`` / no match): the epoch with the highest
       Ultralytics fitness (box mAP50-95 + mask mAP50-95,
       :func:`ultralytics_fitness`); on a tie the **latest** epoch wins,
       because the trainer overwrites ``best.pt`` when ``fitness ==
       best_fitness``.

    After a resume, Ultralytics may log the same epoch twice: the row of an
    epoch that was validated but killed before its checkpoint was saved, and
    the row of the re-run of that epoch. Only the **last** row of each epoch
    is kept, so an orphaned row (whose weights never existed) can never be
    selected as the best epoch.

    F1 is derived as ``2PR / (P + R)`` for Box and Mask.

    Args:
        results_csv: Path to the CSV written by the Ultralytics trainer.
        best_pt: The run's ``best.pt`` (enables rule 1).

    Returns:
        The eight :data:`METRIC_KEYS`, ``f1_b``, ``f1_m``, ``fitness``,
        ``best_epoch``, ``epochs_trained`` and ``best_epoch_source``
        (``"best.pt"`` or ``"fitness_rule"``).

    Raises:
        ValueError: If the CSV has no rows.
    """
    with Path(results_csv).open() as f:
        by_epoch = {}
        for row in csv.DictReader(f):
            row = {k.strip(): v for k, v in row.items()}
            by_epoch[row.get("epoch")] = row  # last occurrence wins
    rows = sorted(by_epoch.values(), key=lambda r: float(r.get("epoch") or 0.0))
    if not rows:
        raise ValueError(f"results.csv is empty: {results_csv}")
    per_epoch = [{short: float(r.get(full) or 0.0) for short, full in METRIC_KEYS.items()} for r in rows]

    idx, source = None, "fitness_rule"
    if best_pt is not None and Path(best_pt).exists():
        target = _best_pt_metrics(Path(best_pt))
        if target is not None:
            target = {k: _csv_precision(v) for k, v in target.items()}
            matches = [i for i, m in enumerate(per_epoch)
                       if all(math.isclose(m[k], target[k], rel_tol=1e-9, abs_tol=1e-12) for k in target)]
            if matches:
                idx, source = matches[-1], "best.pt"
            else:
                print(f"  [warn] {best_pt}: no results.csv row matches its metrics — using the fitness rule")
    if idx is None:
        idx = max(range(len(rows)), key=lambda i: (ultralytics_fitness(per_epoch[i]), i))

    out: dict[str, Any] = dict(per_epoch[idx])
    for s in ("b", "m"):
        p, r = out[f"precision_{s}"], out[f"recall_{s}"]
        out[f"f1_{s}"] = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    out["fitness"] = ultralytics_fitness(per_epoch[idx])
    epochs = [int(float(r.get("epoch") or 0)) for r in rows]
    if epochs != list(range(1, epochs[-1] + 1)):
        missing = sorted(set(range(1, epochs[-1] + 1)) - set(epochs))
        print(f"  [warn] {results_csv}: epochs missing from the training history "
              f"({len(missing)}: {missing[:5]}{'...' if len(missing) > 5 else ''}) — "
              f"was the run folder moved or edited during training?")
    out["best_epoch"] = float(rows[idx].get("epoch") or 0.0)
    out["epochs_trained"] = float(epochs[-1])   # last epoch, robust to missing rows
    out["best_epoch_source"] = source
    return out


def load_tuned_hp(path: Path) -> dict[str, Any]:
    """Load the ``best_hyperparameters.yaml`` written by Phase 3.

    Raises:
        ValueError: If the YAML is empty (signals a failed tune).
    """
    with Path(path).open() as f:
        data = yaml.safe_load(f) or {}
    if not data:
        raise ValueError(f"Empty YAML at {path}. Did Phase 3 complete?")
    return data


def require_complete_hpo(state_path: Path) -> None:
    """Raise unless Phase 3's ``hpo_state.json`` reports a complete search."""
    state = read_json(state_path)
    if state is None:
        raise RuntimeError(f"HPO checkpoint not found: {state_path}. Run Phase 3 first.")
    if state.get("status") != "complete":
        raise RuntimeError(
            f"HPO is not complete ({state.get('completed_trials')}/"
            f"{state.get('target_trials')} trials, status={state.get('status')!r}). "
            f"Re-run Phase 3 to resume it.",
        )


def backup_dir(path: Path) -> Path | None:
    """Move ``path`` to ``<path>.bak-<UTC>`` (used by ``--force``); return the backup."""
    if not path.exists():
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = path.with_name(f"{path.name}.bak-{stamp}")
    path.rename(backup)
    print(f"  [force] moved previous run to {backup}")
    return backup


def checkpoint_is_final(last_pt: Path) -> bool:
    """Return ``True`` if ``last.pt`` belongs to a finished run (``epoch == -1``)."""
    from ultralytics.utils.patches import torch_load

    return torch_load(last_pt, map_location="cpu").get("epoch", -1) == -1


# ----------------------------------------------------------------------------
# Resumable training
# ----------------------------------------------------------------------------
def protocol_fingerprint(train_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Return the protocol subset that must match to skip/resume a run."""
    return {k: v for k, v in train_kwargs.items() if k not in _HASH_EXCLUDED}


def train_or_resume(
    *,
    phase: str,
    model: str,
    weights: str,
    train_kwargs: dict[str, Any],
    project: Path,
    name: str,
    force: bool = False,
) -> dict[str, Any]:
    """Train ``project/name`` to completion, resuming or skipping as needed.

    Args:
        phase: Tag stored in ``run_state.json`` (e.g. ``"phase1_baseline"``).
        model: Variant name.
        weights: Pretrained weights used for a fresh start.
        train_kwargs: Full protocol for ``YOLO.train()`` (without
            ``project``/``name``).
        project: Parent directory of the run.
        name: Run directory name.
        force: Move an existing run aside and train from scratch.

    Returns:
        Summary dict with ``model``, ``skipped``, ``reason``, ``elapsed_min``,
        ``resumed`` and ``metrics``.

    Raises:
        RuntimeError: If an existing run was produced with another protocol,
            or another process is training the same run.
    """
    # The lock lives in the parent so it survives ``--force`` moving run_dir.
    with exclusive_lock(Path(project), f".{name}.lock"):
        return _train_or_resume_locked(
            phase=phase, model=model, weights=weights, train_kwargs=train_kwargs,
            project=project, name=name, force=force,
        )


def _train_or_resume_locked(
    *,
    phase: str,
    model: str,
    weights: str,
    train_kwargs: dict[str, Any],
    project: Path,
    name: str,
    force: bool,
) -> dict[str, Any]:
    """Body of :func:`train_or_resume`, executed while holding the run lock."""
    import torch
    import ultralytics
    from ultralytics import YOLO

    run_dir = Path(project) / name
    state_path = run_dir / RUN_STATE_FILE
    last_pt = run_dir / "weights" / "last.pt"
    fingerprint = protocol_fingerprint(train_kwargs)
    phash = config_hash(fingerprint)

    if force:
        backup_dir(run_dir)

    state = read_json(state_path)
    if state is not None and state.get("protocol_hash") != phash:
        raise RuntimeError(
            f"{run_dir} was trained with a different protocol "
            f"({state.get('protocol_hash')} != {phash}). Use --force to retrain.",
        )
    if state is not None and state.get("status") == "complete":
        # Re-parsed (not read from run_state) so the best-epoch rule in force is applied.
        metrics = parse_best_metrics(run_dir / "results.csv", run_dir / "weights" / "best.pt")
        _check_batch(run_dir, train_kwargs)
        return {
            "model": model, "skipped": True, "resumed": False, "elapsed_min": 0.0,
            "reason": f"complete ({state_path})", "metrics": metrics,
        }
    if state is None and run_dir.exists() and not last_pt.exists():
        # Crashed before the first checkpoint, or a foreign folder: start clean.
        backup_dir(run_dir)
    if state is None:
        state = {
            "status": "running", "phase": phase, "model": model, "run_dir": str(run_dir),
            "weights": weights, "protocol": fingerprint, "protocol_hash": phash,
            "ultralytics_version": ultralytics.__version__,
            "torch_version": torch.__version__, "events": [],
        }

    def log(event: str, **info: Any) -> None:
        state["events"].append({"at": utc_now_iso(), "event": event, **info})
        atomic_write_json(state_path, state)

    t0 = time.perf_counter()
    resumed = False
    if last_pt.exists() and checkpoint_is_final(last_pt):
        print(f"  [finalise] {last_pt} is already final — recording completion only")
    elif last_pt.exists():
        resumed = True
        log("resume", checkpoint=str(last_pt))
        print(f"  [resume] continuing interrupted run from {last_pt}")
        YOLO(str(last_pt)).train(
            resume=True, device=train_kwargs["device"], workers=train_kwargs["workers"],
        )
    else:
        log("start")
        YOLO(weights).train(**train_kwargs, project=str(project), name=name, exist_ok=True)

    if not (last_pt.exists() and checkpoint_is_final(last_pt)):
        raise RuntimeError(f"training of {run_dir} ended without a final checkpoint")
    metrics = parse_best_metrics(run_dir / "results.csv", run_dir / "weights" / "best.pt")
    batch = _check_batch(run_dir, train_kwargs)
    if batch is not None and batch != train_kwargs["batch"]:
        state["events"].append({"at": utc_now_iso(), "event": "batch_reduced",
                                "requested": train_kwargs["batch"], "used": batch})
    state.update(
        status="complete", metrics=metrics, epochs_trained=int(metrics["epochs_trained"]),
        batch_effective=batch, best_pt=str(run_dir / "weights" / "best.pt"),
    )
    log("complete", elapsed_min=round((time.perf_counter() - t0) / 60, 2))
    return {
        "model": model, "skipped": False, "resumed": resumed, "reason": None,
        "elapsed_min": (time.perf_counter() - t0) / 60, "metrics": metrics,
    }


def _check_batch(run_dir: Path, train_kwargs: dict[str, Any]) -> int | None:
    """Micro-batch really used by a finished run; warn when it differs from the protocol."""
    batch = effective_batch(run_dir / "weights" / "best.pt")
    if batch is not None and batch != train_kwargs["batch"]:
        print(f"  [warn] {run_dir.name}: trained with batch={batch}, not the protocol's "
              f"batch={train_kwargs['batch']} (Ultralytics' out-of-memory fallback). "
              f"nbs={train_kwargs.get('nbs')} keeps the effective batch; BatchNorm batch "
              f"statistics differ — report it.")
    return batch


def print_phase_summary(title: str, summary: list[dict], total_min: float) -> None:
    """Print a per-model summary table shared by the training phases."""
    print("\n" + "=" * 80)
    print(f"=== {title} — SUMMARY")
    print("=" * 80)
    for s in summary:
        m = s.get("metrics") or {}
        score = (
            f"  mAP50-95(M)={m['map5095_m']:.4f}  F1(M)={m['f1_m']:.4f}"
            if "map5095_m" in m else ""
        )
        if s.get("failed"):
            status = f"FAILED ({s.get('reason')})"
        elif s["skipped"]:
            status = "skipped (complete)"
        else:
            status = f"ok{' (resumed)' if s.get('resumed') else ''} in {s['elapsed_min']:.1f} min"
        print(f"  {s['model']:<8} : {status}{score}")
    print(f"\nTotal time: {total_min:.1f} min ({total_min / 60:.2f} h)")


def copy_if_exists(src: Path, dst: Path) -> None:
    """Copy ``src`` to ``dst`` when it exists (helper for artefact snapshots)."""
    if Path(src).exists():
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
