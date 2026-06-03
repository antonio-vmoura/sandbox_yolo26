"""Phase 3 — Optimised fine-tuning of YOLO26-seg variants on ISIC 2018 Task 1.

After Phase 2 (HPO) has produced a ``best_hyperparameters.yaml`` per variant,
this script trains each variant **once** on the full train/val split using
those tuned hyperparameters. The single resulting ``best.pt`` is then used by
the comparison notebooks and serves as the seed for Phase 4 (cross-validation).

The script is **idempotent**: a model whose ``<project>/yolo26_<model>_ft_isic_2018_<VERSION>/weights/best.pt``
already exists is skipped unless ``--force`` is passed.

Per-model fixed protocol (must match Phases 1 and 4 modulo the tuned HPs):

* Optimiser: ``MuSGD`` with cosine LR scheduling.
* ``amp=False`` (FP32) — same justification as Phases 1/2/4.
* ``nbs=64`` and ``batch=16`` → effective optimisation batch = 64.
* 120 epochs, patience 25, deterministic ``seed=0``.
* ``close_mosaic=10`` and ``erasing=0.4``.

Hyperparameter YAML location (hard-coded to align Phase 2 and Phase 3):

    ``<project>/hpo/hpo_v3/tune_isic_2018_task_1_<model>/best_hyperparameters.yaml``

Usage:
    # Train ALL five sizes sequentially (default)::

        python train_all_models.py

    # Train a subset::

        python train_all_models.py --models small medium

    # Force re-training::

        python train_all_models.py --force

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
            python yolo26_seg/train_all_models.py --models xlarge \\
            2>&1 | tee logs/train_all_models_xlarge_v11.log
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path
from typing import Union

import yaml
from ultralytics import YOLO

# ----------------------------------------------------------------------------
# Module-level configuration
# ----------------------------------------------------------------------------
#: Suffix used in the canonical output directory name for this phase.
VERSION: str = "v11"

#: Canonical order of model sizes used across the pipeline.
DEFAULT_ORDER: list[str] = ["nano", "small", "medium", "large", "xlarge"]

#: Path to the pretrained weights for each variant (canonical Docker cache).
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
    """Parse command-line arguments for the Phase 3 orchestrator.

    Returns:
        The parsed ``argparse.Namespace`` with attributes ``models``,
        ``data``, ``device``, ``project`` and ``force``.
    """
    p = argparse.ArgumentParser(
        description="Sequential or selective optimised fine-tuning of YOLO26-seg.",
    )
    p.add_argument(
        "--models", nargs="+", default=DEFAULT_ORDER, choices=DEFAULT_ORDER,
        help=f"Subset of models to train (default: {DEFAULT_ORDER}).",
    )
    p.add_argument(
        "--data",
        default="/workspace/datasets/isic_2018_task1_yolo26/data.yaml",
        help="Path to the data.yaml (must be the same used for HPO).",
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
        help="Re-train even when best.pt already exists for the model.",
    )
    return p.parse_args()


def parse_device(arg: str) -> DeviceArg:
    """Parse ``--device`` into a value Ultralytics accepts.

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


# ----------------------------------------------------------------------------
# Hyperparameter loading
# ----------------------------------------------------------------------------
def load_tuned_hp(path: Path) -> dict:
    """Load ``best_hyperparameters.yaml`` produced by Ultralytics' Tuner.

    Args:
        path: Path to the YAML file emitted by Phase 2.

    Returns:
        Dictionary of tuned hyperparameters (loaded by ``yaml.safe_load``).

    Raises:
        ValueError: If the file exists but is empty (signals a failed tune).
    """
    with path.open("r") as f:
        data = yaml.safe_load(f) or {}
    if not data:
        raise ValueError(
            f"Empty YAML at {path}. Check whether the previous tune crashed.",
        )
    return data


# ----------------------------------------------------------------------------
# Per-model training
# ----------------------------------------------------------------------------
def _phase3_paths(project: Path | str, model_size: str) -> tuple[Path, Path, Path]:
    """Resolve canonical input/output paths for a single optimised fine-tune.

    Args:
        project: Root logs directory (typically ``<project>``).
        model_size: Variant name (one of :data:`DEFAULT_ORDER`).

    Returns:
        ``(out_dir, best_pt, hp_yaml)`` — ``hp_yaml`` is read; ``best_pt`` is
        used for idempotency.
    """
    out_dir = Path(project) / f"yolo26_{model_size}_ft_isic_2018_{VERSION}"
    best_pt = out_dir / "weights" / "best.pt"
    hp_yaml = (
        Path(project)
        / "hpo"
        / "hpo_v3"
        / f"tune_isic_2018_task_1_{model_size}"
        / "best_hyperparameters.yaml"
    )
    return out_dir, best_pt, hp_yaml


def _build_base_kwargs(
    args: argparse.Namespace,
    device: DeviceArg,
    model_size: str,
) -> dict:
    """Build the fixed Phase 3 training protocol (overridden by tuned HPs)."""
    return dict(
        data=args.data,
        project=args.project,
        name=f"yolo26_{model_size}_ft_isic_2018_{VERSION}",
        task="segment",
        pretrained=True,
        imgsz=640,
        device=device,
        batch=16,
        workers=4,
        cache=False,
        amp=False,                  # FP32 — same as Phases 1/2/4
        optimizer="MuSGD",
        cos_lr=True,
        close_mosaic=10,
        erasing=0.4,
        nbs=64,                     # effective optim batch = 64
        epochs=120,
        patience=25,
        deterministic=True,
        seed=0,
        save=True,
        plots=True,
        val=True,
        verbose=True,
    )


def _print_phase3_header(
    model_size: str,
    hp_yaml: Path,
    out_dir: Path,
    tuned_hp: dict,
) -> None:
    """Print the per-model section banner."""
    print("\n" + "=" * 80)
    print(
        f"=== PHASE 3 START {VERSION}: {model_size} (120 epochs x patience 25)"
    )
    print(f"  HP source : {hp_yaml}")
    print(f"  Output    : {out_dir}")
    print("  Tuned hyperparameters:")
    for k, v in sorted(tuned_hp.items()):
        print(f"    {k:18s} = {v}")
    print("=" * 80)


def train_one_model(
    model_size: str,
    args: argparse.Namespace,
    device: DeviceArg,
) -> dict:
    """Fine-tune a single variant using the tuned hyperparameters.

    Args:
        model_size: Variant name (one of :data:`DEFAULT_ORDER`).
        args: Parsed CLI arguments.
        device: Device specification produced by :func:`parse_device`.

    Returns:
        Summary dict with keys ``model``, ``skipped``, ``reason`` and
        ``elapsed_min``.
    """
    out_dir, best_pt, hp_yaml = _phase3_paths(args.project, model_size)

    if best_pt.exists() and not args.force:
        return {
            "model": model_size,
            "skipped": True,
            "reason": f"{best_pt} already exists (use --force to re-train)",
            "elapsed_min": 0.0,
        }
    if not hp_yaml.exists():
        return {
            "model": model_size,
            "skipped": True,
            "reason": (
                f"hyperparameter YAML not found: {hp_yaml}. "
                f"Run Phase 2 (tune_all_models_v2.py) for this model first."
            ),
            "elapsed_min": 0.0,
        }

    tuned_hp = load_tuned_hp(hp_yaml)
    _print_phase3_header(model_size, hp_yaml, out_dir, tuned_hp)

    t0 = time.perf_counter()
    model = YOLO(WEIGHTS[model_size])
    base = _build_base_kwargs(args, device, model_size)
    # The tuned HPs override the fixed protocol where they overlap (lr0, lrf,
    # momentum, weight_decay, augmentation factors, loss weights, ...).
    train_kwargs = {**base, **tuned_hp}
    model.train(**train_kwargs)
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
    print(f"Phase 3 orchestration {VERSION} for models: {args.models}")
    print(f"  device       = {device}")
    print(f"  data         = {args.data}")
    print(f"  project      = {args.project}")
    print(f"  force re-run = {args.force}")


def _print_run_summary(
    summary: list[dict],
    failures: list[tuple[str, str]],
    total_min: float,
) -> None:
    """Print the final per-model summary table and total wall time."""
    print("\n" + "=" * 80)
    print(f"=== PHASE 3 SUMMARY {VERSION} (ALL MODELS)")
    print("=" * 80)
    for s in summary:
        if s["skipped"]:
            print(f"  {s['model']:<8} : skipped ({s['reason']})")
        elif s["reason"] == "fail":
            print(f"  {s['model']:<8} : FAILED")
        else:
            print(f"  {s['model']:<8} : ok       ({s['elapsed_min']:.1f} min)")
    print(f"\nTotal Phase 3 time: {total_min:.1f} min ({total_min / 60:.2f} h)")

    if failures:
        print(f"\n[!] {len(failures)} failure(s):")
        for m, _ in failures:
            print(f"    - Model {m}")


def main() -> int:
    """Run Phase 3 sequentially over the requested models.

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
        print(f"\n[{i}/{len(args.models)}] Processing model: {m}")
        try:
            stats = train_one_model(m, args, device)
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
