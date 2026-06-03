"""Phase 1 — Baseline training of YOLO26-seg variants on ISIC 2018 Task 1.

This script sequentially trains the five model sizes (``nano``, ``small``,
``medium``, ``large`` and ``xlarge``) using the **stock Ultralytics
hyperparameters** (no custom optimiser, no aggressive augmentations, no tuned
HPs). The only fixed knobs are ``epochs=120``, ``patience=20``,
``deterministic=True`` and ``seed=0``, which keep the run reproducible and
align it with the protocol of the subsequent pipeline phases.

The script is **idempotent**: a model whose ``best.pt`` already exists is
skipped unless ``--force`` is supplied.

Note:
    AMP (mixed-precision) is **disabled by default** (``amp=False``) for all
    variants. This is a deliberate, explicit deviation from the Ultralytics
    default (``amp=True``) introduced after the ``xlarge`` variant produced
    FP16 overflow (NaN in the classification loss) during an earlier run.
    Disabling AMP uniformly preserves **experimental symmetry across
    architectures** and **hardware-independent reproducibility**. Pass
    ``--amp`` to re-enable mixed precision for an ablation.

Outputs:
    ``<project>/phase1_baseline/yolo26_<model>_baseline/{weights, results.csv, ...}``

Usage:
    # Train ALL five sizes (default)::

        python train_baseline_models.py

    # Train a subset::

        python train_baseline_models.py --models small medium

    # Force retraining even when ``best.pt`` already exists::

        python train_baseline_models.py --force

    # Inside the standard Docker image::

        docker run --gpus all -it --rm --ipc=host \\
            --user $(id -u):$(id -g) \\
            -e TORCH_HOME=/workspace/cache/torch -e HOME=/workspace/cache \\
            -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
            -v $(pwd)/datasets:/workspace/datasets \\
            -v $(pwd)/logs:/workspace/logs \\
            -v $(pwd)/yolo26_seg:/workspace/yolo26_seg \\
            -v $(pwd)/utils:/workspace/utils \\
            -v $(pwd)/cache:/workspace/cache \\
            -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro \\
            yolo26_ft \\
            python /workspace/yolo26_seg/train_baseline_models.py \\
            2>&1 | tee logs/phase1_baseline.log
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path
from typing import Union

from ultralytics import YOLO

# ----------------------------------------------------------------------------
# Module-level configuration
# ----------------------------------------------------------------------------
#: Subdirectory under ``<project>`` where this phase's artefacts are written.
VERSION: str = "phase1_baseline"

#: Canonical order of model sizes used across the pipeline.
DEFAULT_ORDER: list[str] = ["nano", "small", "medium", "large", "xlarge"]

#: Path to the pretrained weights for each variant. Ultralytics auto-downloads
#: missing weights on first use; this map only documents the canonical cache
#: location inside the Docker image.
WEIGHTS: dict[str, str] = {
    "nano":   "/workspace/cache/yolo26n-seg.pt",
    "small":  "/workspace/cache/yolo26s-seg.pt",
    "medium": "/workspace/cache/yolo26m-seg.pt",
    "large":  "/workspace/cache/yolo26l-seg.pt",
    "xlarge": "/workspace/cache/yolo26x-seg.pt",
}

#: Type alias for the ``device`` argument accepted by Ultralytics.
DeviceArg = Union[int, str, list[int]]


# ----------------------------------------------------------------------------
# CLI parsing
# ----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the Phase 1 orchestrator.

    Returns:
        The parsed ``argparse.Namespace`` with the following attributes:
        ``models``, ``data``, ``device``, ``project``, ``epochs``,
        ``patience``, ``imgsz``, ``seed``, ``amp`` and ``force``.
    """
    p = argparse.ArgumentParser(
        description="Phase 1 — Baseline training (Ultralytics defaults) of YOLO26-seg.",
    )
    p.add_argument(
        "--models", nargs="+", default=DEFAULT_ORDER, choices=DEFAULT_ORDER,
        help=f"Subset of models to train (default: {DEFAULT_ORDER}).",
    )
    p.add_argument(
        "--data",
        default="/workspace/datasets/isic_2018_task1_yolo26/data.yaml",
        help="Path to the data.yaml.",
    )
    p.add_argument(
        "--device", default="0,1",
        help="GPU IDs (default: '0,1' for DDP). Use '0' for single-GPU or 'cpu'.",
    )
    p.add_argument(
        "--project", default="/workspace/logs",
        help="Root directory for logs (default: /workspace/logs).",
    )
    p.add_argument(
        "--epochs", type=int, default=120,
        help="Epochs per model (default: 120).",
    )
    p.add_argument(
        "--patience", type=int, default=20,
        help="Early-stopping patience (default: 20).",
    )
    p.add_argument(
        "--imgsz", type=int, default=640,
        help="Input image size (default: 640).",
    )
    p.add_argument(
        "--seed", type=int, default=0,
        help="Deterministic seed (default: 0).",
    )
    p.add_argument(
        "--amp", action="store_true",
        help=(
            "Enable Automatic Mixed Precision (FP16). Default: disabled, to "
            "keep numerical conditions uniform across architectures and to "
            "avoid the FP16 overflow that produces NaN in the cls-loss of "
            "the xlarge variant (observed empirically on ISIC 2018 Task 1)."
        ),
    )
    p.add_argument(
        "--force", action="store_true",
        help="Re-train even when best.pt already exists for the model.",
    )
    return p.parse_args()


def parse_device(arg: str) -> DeviceArg:
    """Parse the ``--device`` argument into a value Ultralytics accepts.

    Args:
        arg: Raw CLI value: a single GPU id ("0"), a CSV list of GPU ids
            ("0,1") for DDP, or the literal string ``"cpu"``.

    Returns:
        ``list[int]`` for multi-GPU DDP, ``"cpu"`` for CPU, or ``int`` for
        single-GPU training.
    """
    if "," in arg:
        return [int(x) for x in arg.split(",")]
    if arg == "cpu":
        return "cpu"
    return int(arg)


# ----------------------------------------------------------------------------
# Per-model training
# ----------------------------------------------------------------------------
def _baseline_paths(project: Path | str, model_size: str) -> tuple[Path, str, Path]:
    """Resolve canonical output paths for a single baseline model.

    Args:
        project: Root logs directory (typically ``<project>``).
        model_size: Variant name (one of :data:`DEFAULT_ORDER`).

    Returns:
        A tuple ``(run_root, run_name, best_pt)`` where:
            * ``run_root`` is ``<project>/phase1_baseline/``,
            * ``run_name`` is ``yolo26_<model_size>_baseline``, and
            * ``best_pt`` is the canonical artefact used for idempotency.
    """
    run_root = Path(project) / VERSION
    run_name = f"yolo26_{model_size}_baseline"
    best_pt = run_root / run_name / "weights" / "best.pt"
    return run_root, run_name, best_pt


def _print_phase1_header(
    model_size: str,
    args: argparse.Namespace,
    device: DeviceArg,
    out_dir: Path,
) -> None:
    """Print the per-model section banner."""
    print("\n" + "=" * 80)
    print(
        f"=== PHASE 1 (BASELINE): {model_size}  "
        f"({args.epochs} epochs x patience {args.patience})"
    )
    print(f"  data       = {args.data}")
    print(f"  device     = {device}")
    print(f"  output     = {out_dir}")
    print(
        f"  amp        = {args.amp}  "
        f"(Ultralytics default = True; disabled here for uniformity)"
    )
    print("  HP         = Ultralytics defaults (no custom optimiser / no tuned HPs)")
    print("=" * 80)


def train_one_baseline(
    model_size: str,
    args: argparse.Namespace,
    device: DeviceArg,
) -> dict:
    """Train a single variant using the stock Ultralytics defaults.

    Skips the training when the canonical ``best.pt`` already exists and
    ``--force`` was not supplied.

    Args:
        model_size: Variant name (one of :data:`DEFAULT_ORDER`).
        args: Parsed CLI arguments (see :func:`parse_args`).
        device: Device specification produced by :func:`parse_device`.

    Returns:
        A summary dictionary with keys ``model``, ``skipped`` (bool),
        ``reason`` (str | None) and ``elapsed_min`` (float).
    """
    run_root, run_name, best_pt = _baseline_paths(args.project, model_size)
    out_dir = run_root / run_name

    if best_pt.exists() and not args.force:
        return {
            "model": model_size,
            "skipped": True,
            "reason": f"{best_pt} already exists (use --force to re-train)",
            "elapsed_min": 0.0,
        }

    weights_path = WEIGHTS[model_size]
    if not Path(weights_path).exists():
        print(
            f"  [warn] pretrained weights not found at {weights_path}; "
            f"Ultralytics will auto-download them on first use.",
        )

    _print_phase1_header(model_size, args, device, out_dir)

    t0 = time.perf_counter()
    model = YOLO(weights_path)

    # Stock Ultralytics defaults — we do NOT override optimizer, cos_lr, etc.
    # AMP is the single explicit deviation from the defaults (off by default).
    model.train(
        data=args.data,
        project=str(run_root),
        name=run_name,
        task="segment",
        pretrained=True,
        imgsz=args.imgsz,
        device=device,
        epochs=args.epochs,
        patience=args.patience,
        amp=args.amp,
        deterministic=True,
        seed=args.seed,
        save=True,
        plots=True,
        val=True,
        verbose=True,
    )

    elapsed = (time.perf_counter() - t0) / 60
    return {
        "model": model_size,
        "skipped": False,
        "reason": None,
        "elapsed_min": elapsed,
    }


# ----------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------
def _print_run_header(args: argparse.Namespace, device: DeviceArg) -> None:
    """Print the top-level orchestration summary."""
    print(f"Phase 1 (Baseline) orchestration for models: {args.models}")
    print(f"  device       = {device}")
    print(f"  data         = {args.data}")
    print(f"  project      = {args.project}")
    print(f"  epochs       = {args.epochs}  patience = {args.patience}")
    print(f"  seed         = {args.seed}  imgsz = {args.imgsz}")
    print(f"  amp          = {args.amp}")
    print(f"  force re-run = {args.force}")


def _print_run_summary(
    summary: list[dict],
    failures: list[tuple[str, str]],
    total_min: float,
) -> None:
    """Print the final per-model summary table and total wall time."""
    print("\n" + "=" * 80)
    print("=== PHASE 1 (BASELINE) — SUMMARY")
    print("=" * 80)
    for s in summary:
        if s["skipped"]:
            print(f"  {s['model']:<8} : skipped ({s['reason']})")
        elif s["reason"] == "fail":
            print(f"  {s['model']:<8} : FAILED")
        else:
            print(f"  {s['model']:<8} : ok       ({s['elapsed_min']:.1f} min)")

    print(f"\nTotal Phase 1 time: {total_min:.1f} min ({total_min / 60:.2f} h)")

    if failures:
        print(f"\n[!] {len(failures)} failure(s):")
        for m, _ in failures:
            print(f"    - Model {m}")


def main() -> int:
    """Run Phase 1 sequentially over the requested models.

    Returns:
        ``0`` on full success (including skipped models), ``1`` if any
        model raised an exception.
    """
    args = parse_args()
    device = parse_device(args.device)
    _print_run_header(args, device)

    summary: list[dict] = []
    failures: list[tuple[str, str]] = []
    t_total = time.perf_counter()

    for i, m in enumerate(args.models, 1):
        print(f"\n[{i}/{len(args.models)}] Training baseline of model: {m}")
        try:
            stats = train_one_baseline(m, args, device)
            summary.append(stats)
            if stats["skipped"]:
                print(f"  [skip] {stats['reason']}")
            else:
                print(f"  [done] training time: {stats['elapsed_min']:.1f} min")
        except Exception:
            tb = traceback.format_exc()
            print(f"  [fail] exception in model {m}:\n{tb}")
            summary.append({
                "model": m, "skipped": False, "reason": "fail",
                "elapsed_min": 0.0,
            })
            failures.append((m, tb))

    total_min = (time.perf_counter() - t_total) / 60
    _print_run_summary(summary, failures, total_min)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
