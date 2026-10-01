"""Consolidate single-split validation metrics of Phase 1 or Phase 4 into CSV + JSON.

For each variant this script reads the run's ``results.csv``, selects the
epoch that produced ``best.pt`` (identified from the checkpoint itself, see
:func:`training.parse_best_metrics`), derives F1 (Box and Mask) and writes::

    <project>/summary/<phase>_val.csv
    <project>/summary/<phase>_val.json

The values are **validation-split** metrics; test-set metrics are produced in
Phase 5. Runs that are not complete (``run_state.json`` status other than
``complete``) are reported as missing.

Usage:
    python collect_phase_metrics.py --phase phase1 --project /workspace/logs/pipeline_final_v1
    python collect_phase_metrics.py --phase phase4 --models small medium
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from common import DEFAULT_ORDER, DEFAULT_PIPELINE_ROOT, PipelinePaths, atomic_write_json, read_json
from training import RUN_STATE_FILE, parse_best_metrics


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed ``argparse.Namespace`` with ``phase``, ``models``, ``project``.
    """
    p = argparse.ArgumentParser(
        description="Collect single-split validation metrics (Phase 1 or 4) into CSV / JSON.",
    )
    p.add_argument(
        "--phase", choices=["phase1", "phase4"], required=True,
        help="'phase1' (baseline) or 'phase4' (optimised).",
    )
    p.add_argument(
        "--models", nargs="+", default=DEFAULT_ORDER, choices=DEFAULT_ORDER,
        help=f"Subset of models to consolidate (default: {DEFAULT_ORDER}).",
    )
    p.add_argument(
        "--project", default=DEFAULT_PIPELINE_ROOT,
        help=f"Pipeline root (default: {DEFAULT_PIPELINE_ROOT}).",
    )
    return p.parse_args()


def run_dir(paths: PipelinePaths, phase: str, model: str) -> Path:
    """Return the training run directory of ``model`` in ``phase``."""
    if phase == "phase1":
        return paths.phase1_dir / paths.phase1_run_name(model)
    return paths.phase4_dir / paths.phase4_run_name(model)


def collect_row(paths: PipelinePaths, phase: str, model: str) -> dict | None:
    """Return one consolidated row, or ``None`` if the run is missing/incomplete."""
    rd = run_dir(paths, phase, model)
    state = read_json(rd / RUN_STATE_FILE)
    if state is None or state.get("status") != "complete":
        print(f"  [warn] {model}: run not complete ({rd})")
        return None
    metrics = parse_best_metrics(rd / "results.csv", rd / "weights" / "best.pt")
    resumed = any(e.get("event") == "resume" for e in state.get("events", []))
    print(
        f"  {model:<8} : mAP50(M)={metrics['map50_m']:.4f} mAP50-95(M)={metrics['map5095_m']:.4f} "
        f"P(M)={metrics['precision_m']:.4f} R(M)={metrics['recall_m']:.4f} F1(M)={metrics['f1_m']:.4f}",
    )
    return {"model": model, "split": "val", "resumed": resumed, "run_dir": str(rd), **metrics}


def main() -> int:
    """Collect per-model metrics for one phase and write CSV + JSON.

    Returns:
        ``0`` if every requested model was collected, ``1`` otherwise.
    """
    args = parse_args()
    paths = PipelinePaths(Path(args.project))
    paths.summary_dir.mkdir(parents=True, exist_ok=True)

    rows, missing = [], []
    for m in args.models:
        row = collect_row(paths, args.phase, m)
        (rows.append(row) if row else missing.append(m))

    csv_out = paths.summary_dir / f"{args.phase}_val.csv"
    json_out = paths.summary_dir / f"{args.phase}_val.json"
    if rows:
        with csv_out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    atomic_write_json(json_out, {"phase": args.phase, "split": "val", "models": rows, "missing": missing})

    print(f"\nGenerated: {csv_out}\n           {json_out}")
    if missing:
        print(f"  [warn] missing/incomplete: {missing}")
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
