"""Phase 1 — Baseline training of YOLO26-seg variants on ISIC 2018 Task 1.

Trains each requested size (``nano`` … ``xlarge``) with the fixed **base
setup** shared by every phase and the **Ultralytics default
hyperparameters** (learning rate, momentum, weight decay, warm-up, loss gains,
augmentation), via :func:`common.baseline_protocol`. Baseline and Optimised
(Phase 4) share the identical base setup and differ only in the tuned
hyperparameters. The base setup deviates from the Ultralytics defaults in:

* ``epochs=120`` and ``patience=120`` — no early stopping (defaults: 100 / 100);
* ``amp=False`` (default: True) — FP16 overflowed (NaN cls-loss) on xlarge;
* ``optimizer="MuSGD"`` and ``cos_lr=True`` (defaults: ``"auto"`` and False;
  ``"auto"`` would pick AdamW or MuSGD from the iteration count and ignore
  ``lr0``/``momentum``).

``nbs=64`` and ``close_mosaic=10`` equal the defaults and are pinned.

The run is fault-tolerant (see :mod:`training`): an interrupted model resumes
from ``last.pt``; a completed model (``run_state.json`` status ``complete``)
is skipped unless ``--force`` is passed.

Outputs:
    ``<project>/phase1_baseline/yolo26_<model>_baseline/{weights/, results.csv, run_state.json, ...}``

Usage:
    # All five sizes into the default pipeline root::

        python train_baseline_models.py

    # A subset, explicit root::

        python train_baseline_models.py --models small medium \\
            --project /workspace/logs/pipeline_final_v1

    # Retrain from scratch (the old run is moved to *.bak-<UTC>)::

        python train_baseline_models.py --models small --force
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

from common import (
    DEFAULT_DATA_YAML,
    DEFAULT_ORDER,
    DEFAULT_PIPELINE_ROOT,
    TRAIN_EPOCHS,
    TRAIN_PATIENCE,
    WEIGHTS,
    PipelinePaths,
    baseline_protocol,
    parse_device,
    seed_everything,
)
from training import print_phase_summary, train_or_resume

PHASE: str = "phase1_baseline"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for Phase 1.

    Returns:
        The parsed ``argparse.Namespace``.
    """
    p = argparse.ArgumentParser(
        description="Phase 1 — Baseline training (base setup + default HPs) of YOLO26-seg.",
    )
    p.add_argument(
        "--models", nargs="+", default=DEFAULT_ORDER, choices=DEFAULT_ORDER,
        help=f"Subset of models to train (default: {DEFAULT_ORDER}).",
    )
    p.add_argument("--data", default=DEFAULT_DATA_YAML, help="Path to the data.yaml.")
    p.add_argument(
        "--device", default="0,1",
        help="GPU IDs (default: '0,1' for DDP). Use '0' for single-GPU or 'cpu'.",
    )
    p.add_argument(
        "--project", default=DEFAULT_PIPELINE_ROOT,
        help=f"Pipeline root (default: {DEFAULT_PIPELINE_ROOT}).",
    )
    p.add_argument(
        "--epochs", type=int, default=TRAIN_EPOCHS,
        help=f"Training budget (default: {TRAIN_EPOCHS}). Must match Phases 2 and 4.",
    )
    p.add_argument(
        "--patience", type=int, default=TRAIN_PATIENCE,
        help=f"Early-stopping patience (default: {TRAIN_PATIENCE}). Must match Phases 2 and 4.",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Retrain from scratch (moves an existing run to *.bak-<UTC>).",
    )
    return p.parse_args()


def main() -> int:
    """Run Phase 1 sequentially over the requested models.

    Returns:
        ``0`` on success (including skipped models), ``1`` if any model failed.
    """
    args = parse_args()
    seed_everything()
    device = parse_device(args.device)
    paths = PipelinePaths(Path(args.project))
    kwargs = baseline_protocol(args.data, device, args.epochs, args.patience)

    print(f"Phase 1 (Baseline) for models: {args.models}")
    print(f"  device  = {device}   data = {args.data}")
    print(f"  output  = {paths.phase1_dir}")
    print(f"  budget  = {args.epochs} epochs, patience {args.patience}, amp=False, seed=0")
    print("  setup   = MuSGD, cos_lr, nbs=64, close_mosaic=10 (identical to Phases 3/4)")
    print("  HPs     = Ultralytics defaults (lr0, momentum, weight_decay, loss gains, augmentation)")

    summary: list[dict] = []
    t0 = time.perf_counter()
    for i, m in enumerate(args.models, 1):
        print("\n" + "=" * 80)
        print(f"=== [{i}/{len(args.models)}] PHASE 1 (BASELINE): {m}")
        print("=" * 80)
        try:
            summary.append(train_or_resume(
                phase=PHASE, model=m, weights=WEIGHTS[m], train_kwargs=kwargs,
                project=paths.phase1_dir, name=paths.phase1_run_name(m), force=args.force,
            ))
        except Exception as e:
            print(f"  [fail] {m}:\n{traceback.format_exc()}", file=sys.stderr)
            summary.append({"model": m, "skipped": False, "failed": True, "reason": str(e)})

    print_phase_summary("PHASE 1 (BASELINE)", summary, (time.perf_counter() - t0) / 60)
    return 1 if any(s.get("failed") for s in summary) else 0


if __name__ == "__main__":
    sys.exit(main())
