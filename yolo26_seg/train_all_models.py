"""Phase 4 — Optimised fine-tuning of YOLO26-seg variants on ISIC 2018 Task 1.

Trains each variant **once** on the standard train/val split with the
hyperparameters found in Phase 3, via :func:`common.optimized_protocol`:

* the Phase 1 protocol (same ``epochs=120``, ``patience=120``, ``amp=False``,
  ``batch=16``/``nbs=64``, ``seed=0``) — identical budget to the Baseline;
* the fixed optimisation recipe (``MuSGD``, cosine LR, ``close_mosaic=10``),
  identical to the one every HPO trial used;
* the tuned HPs from ``<project>/phase3_hpo/tune_<model>/best_hyperparameters.yaml``.

A model is only trained when its HPO is complete (``hpo_state.json`` status
``complete``); pass ``--allow-incomplete-hpo`` to override. The YAML that was
used is copied into the run directory for provenance.

The run is fault-tolerant (see :mod:`training`): an interrupted model resumes
from ``last.pt``; a completed model is skipped unless ``--force`` is passed.

Outputs:
    ``<project>/phase4_optimized/yolo26_<model>_optimized/{weights/, results.csv,
    run_state.json, tuned_hyperparameters.yaml, ...}``

Usage:
    # All five sizes::

        python train_all_models.py --project /workspace/logs/pipeline_final_v1

    # A subset, retraining from scratch::

        python train_all_models.py --models xlarge --force
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
    optimized_protocol,
    parse_device,
    seed_everything,
)
from training import (
    copy_if_exists,
    load_tuned_hp,
    print_phase_summary,
    require_complete_hpo,
    train_or_resume,
)

PHASE: str = "phase4_optimized"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for Phase 4.

    Returns:
        The parsed ``argparse.Namespace``.
    """
    p = argparse.ArgumentParser(
        description="Phase 4 — Optimised fine-tuning of YOLO26-seg with the Phase 3 HPs.",
    )
    p.add_argument(
        "--models", nargs="+", default=DEFAULT_ORDER, choices=DEFAULT_ORDER,
        help=f"Subset of models to train (default: {DEFAULT_ORDER}).",
    )
    p.add_argument(
        "--data", default=DEFAULT_DATA_YAML,
        help="Path to the data.yaml (must be the one used for HPO).",
    )
    p.add_argument(
        "--device", default="0,1",
        help="GPU IDs (default: '0,1' DDP). Use '0' for single-GPU.",
    )
    p.add_argument(
        "--project", default=DEFAULT_PIPELINE_ROOT,
        help=f"Pipeline root (default: {DEFAULT_PIPELINE_ROOT}).",
    )
    p.add_argument(
        "--epochs", type=int, default=TRAIN_EPOCHS,
        help=f"Training budget (default: {TRAIN_EPOCHS}). Must match Phases 1 and 2.",
    )
    p.add_argument(
        "--patience", type=int, default=TRAIN_PATIENCE,
        help=f"Early-stopping patience (default: {TRAIN_PATIENCE}). Must match Phases 1 and 2.",
    )
    p.add_argument(
        "--allow-incomplete-hpo", action="store_true",
        help="Train even if the Phase 3 search did not reach its target trials.",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Retrain from scratch (moves an existing run to *.bak-<UTC>).",
    )
    return p.parse_args()


def train_one_model(
    model: str,
    args: argparse.Namespace,
    device,
    paths: PipelinePaths,
) -> dict:
    """Fine-tune one variant with its tuned hyperparameters.

    Raises:
        RuntimeError: If the HPO is missing/incomplete (and not overridden).
    """
    hp_yaml = paths.phase3_best_yaml(model)
    if not args.allow_incomplete_hpo:
        require_complete_hpo(paths.phase3_state(model))
    if not hp_yaml.exists():
        raise RuntimeError(f"{hp_yaml} not found. Run Phase 3 for {model} first.")

    tuned_hp = load_tuned_hp(hp_yaml)
    print(f"  HP source : {hp_yaml}")
    for k, v in sorted(tuned_hp.items()):
        print(f"    {k:18s} = {v}")

    kwargs = optimized_protocol(args.data, device, tuned_hp, args.epochs, args.patience)
    stats = train_or_resume(
        phase=PHASE, model=model, weights=WEIGHTS[model], train_kwargs=kwargs,
        project=paths.phase4_dir, name=paths.phase4_run_name(model), force=args.force,
    )
    copy_if_exists(
        hp_yaml, paths.phase4_dir / paths.phase4_run_name(model) / "tuned_hyperparameters.yaml",
    )
    return stats


def main() -> int:
    """Run Phase 4 sequentially over the requested models.

    Returns:
        ``0`` on success (including skipped models), ``1`` if any model failed.
    """
    args = parse_args()
    seed_everything()
    device = parse_device(args.device)
    paths = PipelinePaths(Path(args.project))

    print(f"Phase 4 (Optimised fine-tune) for models: {args.models}")
    print(f"  device  = {device}   data = {args.data}")
    print(f"  output  = {paths.phase4_dir}")
    print(f"  budget  = {args.epochs} epochs, patience {args.patience}, amp=False, seed=0")

    summary: list[dict] = []
    t0 = time.perf_counter()
    for i, m in enumerate(args.models, 1):
        print("\n" + "=" * 80)
        print(f"=== [{i}/{len(args.models)}] PHASE 4 (OPTIMISED): {m}")
        print("=" * 80)
        try:
            summary.append(train_one_model(m, args, device, paths))
        except Exception as e:
            print(f"  [fail] {m}:\n{traceback.format_exc()}", file=sys.stderr)
            summary.append({"model": m, "skipped": False, "failed": True, "reason": str(e)})

    print_phase_summary("PHASE 4 (OPTIMISED)", summary, (time.perf_counter() - t0) / 60)
    return 1 if any(s.get("failed") for s in summary) else 0


if __name__ == "__main__":
    sys.exit(main())
