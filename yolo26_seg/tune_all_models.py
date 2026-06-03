"""Legacy Phase 2 HPO orchestrator (predecessor of ``tune_all_models_v2.py``).

This module is the original HPO entry-point used during the v1/v2 sweeps. It
is kept around because:

* the analysis notebooks under ``utils/notebooks/`` reference its name and
  the directory layout it produces, and
* ``wait_gpu.sh`` contains an opportunistic invocation of it.

The canonical, current pipeline uses :mod:`tune_all_models_v2` instead. The
two modules share the same search-space presets but differ in the fixed
protocol — most notably ``tune_all_models_v2`` runs in FP32 (``amp=False``)
and writes under ``<project>/hpo/hpo_v3/``, while this legacy script runs
with ``amp=True`` and writes directly under ``<project>/``.

Search-space presets:

* ``wide`` (20 HPs) — broad initial exploration.
* ``refined`` (15 HPs) — narrowed follow-up that drops low-correlation HPs
  and tightens ranges around the high-signal winners observed in the initial
  sweep (see ``analyze_hpo_results.ipynb``).

Usage:
    # HPO over all five sizes with the wide preset (default)::

        python tune_all_models.py

    # Refined preset, 50 iterations per model::

        python tune_all_models.py --space refined --iterations 50

    # Only the small variant::

        python tune_all_models.py --models small
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path
from typing import Union

from ultralytics import YOLO

#: Path to the pretrained weights for each variant.
WEIGHTS: dict[str, str] = {
    "nano":   "/workspace/cache/yolo26n-seg.pt",
    "small":  "/workspace/cache/yolo26s-seg.pt",
    "medium": "/workspace/cache/yolo26m-seg.pt",
    "large":  "/workspace/cache/yolo26l-seg.pt",
    "xlarge": "/workspace/cache/yolo26x-seg.pt",
}

# ----------------------------------------------------------------------------
# Search spaces — two presets selectable via ``--space``
# ----------------------------------------------------------------------------
# Each entry: (min, max) or (min, max, gain). ``gain`` controls the amplitude
# of the Gaussian mutation — higher = more aggressive. Default = 1.0.
#
# (1) ``wide`` — initial exploration. Ranges based on the Ultralytics default
#     ``Tuner.space`` plus what we learned in the v2-v7 sweeps (lr0 in
#     [1e-3, 2e-3] works; high ``mosaic`` is fine; high ``erasing`` corrupts
#     the mask; moderate ``mixup`` is useful). Use when there is no prior
#     information about the dataset.
#
# (2) ``refined`` — follow-up. Built from the Pearson correlations
#     hp×fitness of the initial ``small`` HPO (30 trials, see
#     ``analyze_hpo_results.ipynb``):
#       * DROPS (|r| < 0.05 — no statistical signal):
#           scale, degrees, box, fliplr, warmup_momentum
#         (these 5 HPs stay fixed at the Ultralytics default)
#       * NARROWS (high signal — concentrate the GA on the promising region):
#           mixup       r=-0.77  [0, 0.3]   → [0.0, 0.05]
#           copy_paste  r=-0.67  [0, 0.3]   → [0.0, 0.05]
#           lr0         r=-0.45  [5e-4,5e-3]→ [1e-3, 4e-3]
#           translate   r=+0.23  [0, 0.3]   → [0.05, 0.20]
#           hsv_h       r=+0.23  [0, 0.03]  → [0.005, 0.025]
#           flipud      r=+0.20  [0, 0.5]   → [0.0, 0.10]
#           weight_decay         [0, 1e-3]  → [1e-6, 1e-4]
#           dfl                  [0.8, 3.0] → [0.8, 1.5]
#           mosaic               [0.5, 1.0] → [0.7, 1.0]
#
#     The refined space has 15 HPs (5 fewer), which gives the GA ~25% more
#     effective budget per iteration plus tighter ranges → faster convergence.

SEARCH_SPACE_WIDE: dict[str, tuple] = {
    # ---- Learning ----
    "lr0":             (5e-4, 5e-3),
    "lrf":             (0.005, 0.05),
    "momentum":        (0.85, 0.95, 0.3),
    "weight_decay":    (0.0, 0.001),
    "warmup_epochs":   (1.0, 5.0),
    "warmup_momentum": (0.5, 0.95),

    # ---- Loss weights ----
    "box":             (3.0, 12.0),
    "cls":             (0.2, 1.5),
    "dfl":             (0.8, 3.0),

    # ---- Colour augmentation ----
    "hsv_h":           (0.0, 0.03),
    "hsv_s":           (0.3, 0.9),
    "hsv_v":           (0.2, 0.7),

    # ---- Geometric augmentation ----
    "degrees":         (0.0, 30.0),
    "translate":       (0.0, 0.3),
    "scale":           (0.2, 0.7),
    "fliplr":          (0.0, 0.6),
    "flipud":          (0.0, 0.5),

    # ---- Mixing augmentations ----
    "mosaic":          (0.5, 1.0),
    "mixup":           (0.0, 0.3),
    "copy_paste":      (0.0, 0.3),
}

SEARCH_SPACE_REFINED: dict[str, tuple] = {
    # ---- Learning (narrowed around small-winner lr0 = 2.33e-3) ----
    "lr0":             (1e-3, 4e-3),
    "lrf":             (0.005, 0.05),
    "momentum":        (0.85, 0.95, 0.3),
    "weight_decay":    (1e-6, 1e-4),
    "warmup_epochs":   (1.0, 5.0),
    # warmup_momentum: |r| < 0.05 → fixed at the default 0.8 (dropped)

    # ---- Loss weights (drops ``box``; narrows ``dfl``) ----
    # box: |r| < 0.05 → fixed at the default 7.5 (dropped)
    "cls":             (0.2, 1.5),
    "dfl":             (0.8, 1.5),

    # ---- Colour augmentation (narrow ``hsv_h``) ----
    "hsv_h":           (0.005, 0.025),
    "hsv_s":           (0.3, 0.9),
    "hsv_v":           (0.2, 0.7),

    # ---- Geometric augmentation (drops degrees, scale, fliplr) ----
    # degrees: |r| < 0.05 → fixed at the default 0 (dropped)
    "translate":       (0.05, 0.20),
    # scale:   |r| < 0.05 → fixed at the default 0.5 (dropped)
    # fliplr:  |r| < 0.05 → fixed at the default 0.5 (dropped)
    "flipud":          (0.0, 0.10),

    # ---- Mixing augmentations (narrow mixup, copy_paste) ----
    "mosaic":          (0.7, 1.0),
    "mixup":           (0.0, 0.05),
    "copy_paste":      (0.0, 0.05),
}


def get_search_space(name: str) -> dict[str, tuple]:
    """Return the search-space mapping for the requested preset.

    Args:
        name: One of ``"wide"`` or ``"refined"``.

    Returns:
        The corresponding search-space dict.

    Raises:
        ValueError: For unknown preset names.
    """
    if name == "wide":
        return SEARCH_SPACE_WIDE
    if name == "refined":
        return SEARCH_SPACE_REFINED
    raise ValueError(f"--space invalid: {name!r}. Use 'wide' or 'refined'.")


#: Canonical order of model sizes used across the pipeline.
DEFAULT_ORDER: list[str] = ["nano", "small", "medium", "large", "xlarge"]

#: Type alias for the ``device`` argument accepted by Ultralytics.
DeviceArg = Union[int, str, list[int]]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the legacy HPO orchestrator.

    Returns:
        Parsed ``argparse.Namespace``.
    """
    p = argparse.ArgumentParser(
        description="Sequential HPO over the five YOLO26-seg sizes (legacy).",
    )
    p.add_argument(
        "--models", nargs="+", default=DEFAULT_ORDER, choices=DEFAULT_ORDER,
        help=f"Subset of models to tune (default: {DEFAULT_ORDER}).",
    )
    p.add_argument(
        "--iterations", type=int, default=30,
        help="Iterations per model (default: 30).",
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
        "--data",
        default="/workspace/datasets/isic_2018_task1_yolo26/data.yaml",
        help="Path to the data.yaml.",
    )
    p.add_argument(
        "--device", default="0,1",
        help="GPU IDs (default: '0,1' DDP). Use '0' for single-GPU.",
    )
    p.add_argument(
        "--project", default="/workspace/logs",
        help="Root directory for logs (default: /workspace/logs).",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Re-run even when best_hyperparameters.yaml already exists.",
    )
    p.add_argument(
        "--space", choices=["wide", "refined"], default="wide",
        help=(
            "Search space — 'wide' (20 HPs, initial exploration) or 'refined' "
            "(15 HPs, drops low-signal HPs and narrows high-signal ones; "
            "recommended for per-size follow-up after one wide round). "
            "Default: wide."
        ),
    )
    return p.parse_args()


def parse_device(arg: str) -> DeviceArg:
    """Parse ``--device`` into a value Ultralytics accepts.

    Args:
        arg: ``"0"`` for single-GPU, ``"0,1"`` for DDP.

    Returns:
        ``list[int]`` for multi-GPU, ``int`` otherwise.
    """
    if "," in arg:
        return [int(x) for x in arg.split(",")]
    return int(arg)


def tune_one_model(
    model_size: str,
    args: argparse.Namespace,
    device: DeviceArg,
    space: dict[str, tuple],
) -> dict:
    """Run ``model.tune()`` for a single variant.

    Args:
        model_size: Variant name (one of :data:`DEFAULT_ORDER`).
        args: Parsed CLI arguments.
        device: Device specification returned by :func:`parse_device`.
        space: Search-space mapping returned by :func:`get_search_space`.

    Returns:
        Summary dict with keys ``model``, ``skipped``, ``reason`` and
        ``elapsed_min``.
    """
    run_name = f"tune_isic_2018_task_1_{model_size}"
    out_dir = Path(args.project) / run_name
    best_yaml = out_dir / "best_hyperparameters.yaml"

    if best_yaml.exists() and not args.force:
        return {
            "model": model_size,
            "skipped": True,
            "reason": f"{best_yaml} already exists (use --force to re-tune)",
            "elapsed_min": 0.0,
        }

    print("\n" + "=" * 80)
    print(
        f"=== TUNE START: {model_size}  "
        f"({args.iterations} iter x {args.epochs} ep)"
    )
    print("=" * 80)

    t0 = time.perf_counter()
    model = YOLO(WEIGHTS[model_size])

    fixed_v7 = dict(
        # Infrastructure
        data=args.data,
        project=args.project,
        name=run_name,
        task="segment",
        pretrained=True,
        imgsz=640,
        # Hardware
        device=device,
        batch=32,
        workers=8,
        cache=False,
        # Optimisation (carried over from v7)
        amp=True,
        optimizer="MuSGD",
        cos_lr=True,
        close_mosaic=15,
        erasing=0.0,
        nbs=64,
        # Short loop for HPO
        epochs=args.epochs,
        patience=args.patience,
        deterministic=True,
        seed=0,
        save=True,
        plots=False,
        val=True,
        verbose=False,
    )

    model.tune(
        space=space,
        iterations=args.iterations,
        use_ray=False,
        **fixed_v7,
    )

    elapsed = (time.perf_counter() - t0) / 60
    return {
        "model": model_size,
        "skipped": False,
        "reason": None,
        "elapsed_min": elapsed,
    }


def _print_run_header(
    args: argparse.Namespace,
    device: DeviceArg,
    space: dict[str, tuple],
) -> None:
    """Print the top-level orchestration summary."""
    print(f"Sequential HPO for: {args.models}")
    print(f"  iterations/model = {args.iterations}")
    print(f"  epochs/trial     = {args.epochs}")
    print(f"  device           = {device}")
    print(f"  data             = {args.data}")
    print(f"  project          = {args.project}")
    print(f"  search space     = {args.space!r} ({len(space)} HPs)")
    print(f"  force re-run     = {args.force}")


def _print_run_summary(
    summary: list[dict],
    failures: list[tuple[str, str]],
    total_min: float,
) -> None:
    """Print the final per-model summary table."""
    print("\n" + "=" * 80)
    print("=== HPO SUMMARY ALL MODELS")
    print("=" * 80)
    for s in summary:
        status = (
            "skipped" if s["skipped"]
            else f"FAILED ({s['reason']})" if s["elapsed_min"] < 0
            else f"{s['elapsed_min']:.1f} min"
        )
        print(f"  {s['model']:8s} : {status}")
    print(f"\nTotal time: {total_min:.1f} min  ({total_min / 60:.2f} h)")

    if failures:
        print(
            f"\n{len(failures)} model(s) failed:",
            [m for m, _ in failures],
        )


def main() -> int:
    """Run the legacy HPO orchestrator sequentially over the requested models.

    Returns:
        ``0`` on full success (including skipped models), ``1`` if any
        model failed.
    """
    args = parse_args()
    device = parse_device(args.device)
    space = get_search_space(args.space)
    _print_run_header(args, device, space)

    summary: list[dict] = []
    failures: list[tuple[str, str]] = []
    t_total = time.perf_counter()

    for i, m in enumerate(args.models, 1):
        print(f"\n[{i}/{len(args.models)}] {m}")
        try:
            stats = tune_one_model(m, args, device, space)
            summary.append(stats)
            if stats["skipped"]:
                print(f"  [skip] {stats['reason']}")
            else:
                print(f"  [done] {m} in {stats['elapsed_min']:.1f} min")
        except Exception as e:  # noqa: BLE001
            tb = traceback.format_exc()
            print(f"  [FAIL] {m}: {e}\n{tb}", file=sys.stderr)
            failures.append((m, str(e)))
            summary.append({
                "model": m, "skipped": False, "reason": f"FAILED: {e}",
                "elapsed_min": -1.0,
            })

    total_min = (time.perf_counter() - t_total) / 60
    _print_run_summary(summary, failures, total_min)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
