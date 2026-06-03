"""Phase 2 — Hyperparameter Optimisation (HPO) of YOLO26-seg on ISIC 2018 Task 1.

This script orchestrates ``model.tune()`` (Ultralytics' built-in genetic
algorithm tuner) sequentially or selectively across the five YOLO26-seg
variants. Two search spaces are bundled in this module:

* ``wide`` — broad initial exploration. Ranges informed by the Ultralytics
  defaults and prior experience with the dermoscopy domain.
* ``refined`` — narrowed follow-up. Ranges shrunk around the Pearson
  correlations (``|r| < 0.05`` dropped, high-signal HPs narrowed) observed
  in the wide-space HPO of the ``small`` variant. See
  ``utils/notebooks/yolo26_ft_isic_2018_task_1_analyze_hpo_results.ipynb``.

The script is **idempotent**: a model whose ``best_hyperparameters.yaml``
already exists is skipped unless ``--force`` is passed.

Usage:
    # Single model (initial exploration)::

        python tune_all_models_v2.py --models small --space wide

    # All five sizes (default ``refined`` space)::

        python tune_all_models_v2.py

    # Refined follow-up with more iterations::

        python tune_all_models_v2.py --space refined --iterations 50

    # Re-tune even if best_hyperparameters.yaml already exists::

        python tune_all_models_v2.py --force

    # Inside the Docker image — wide space, small variant::

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
            python /workspace/yolo26_seg/tune_all_models_v2.py \\
                --space wide --iterations 50 --models small \\
            2>&1 | tee logs/tune_all_models_v2.log
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
# CLI parsing
# ----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the Phase 2 orchestrator.

    Returns:
        The parsed ``argparse.Namespace`` with attributes ``models``,
        ``iterations``, ``epochs``, ``patience``, ``batch``, ``data``,
        ``device``, ``project``, ``force`` and ``space``.
    """
    p = argparse.ArgumentParser(
        description="Sequential or selective HPO on the YOLO26-seg variants.",
    )
    p.add_argument(
        "--models", nargs="+", default=DEFAULT_ORDER, choices=DEFAULT_ORDER,
        help=f"Subset of models to tune (default: {DEFAULT_ORDER}).",
    )
    p.add_argument(
        "--iterations", type=int, default=30,
        help="Number of HPO iterations per model (default: 30).",
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
            "Micro-batch per GPU per trial (default: 32). Reduce to 16 when "
            "running HPO on xlarge in FP32 (``amp=False``) on 32 GB GPUs — "
            "with ``nbs=64`` the effective optimisation batch remains 64 via "
            "gradient accumulation, preserving comparability of the HPs "
            "found against the other variants."
        ),
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
        help="Re-tune even when best_hyperparameters.yaml already exists.",
    )
    p.add_argument(
        "--space", choices=["wide", "refined"], default="wide",
        help="Search space — 'wide' (initial) or 'refined' (follow-up).",
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
# Per-model tuning
# ----------------------------------------------------------------------------
def _tune_paths(project: Path | str, model_size: str) -> tuple[Path, str, Path]:
    """Resolve canonical output paths for a single tuned model.

    Args:
        project: Root logs directory passed to Ultralytics' Tuner.
        model_size: Variant name (one of :data:`DEFAULT_ORDER`).

    Returns:
        A tuple ``(out_dir, run_name, best_yaml)`` where ``best_yaml`` is the
        canonical artefact used for idempotency.
    """
    run_name = f"tune_isic_2018_task_1_{model_size}"
    out_dir = Path(project) / run_name
    best_yaml = out_dir / "best_hyperparameters.yaml"
    return out_dir, run_name, best_yaml


def _build_fixed_hp(
    args: argparse.Namespace,
    device: DeviceArg,
    run_name: str,
) -> dict:
    """Build the ``model.tune()`` keyword arguments that stay fixed across trials.

    The returned dict contains everything that is **not** part of the
    genetic-algorithm search space: infra (data, project, name), hardware
    (device, batch, workers), and the fixed optimisation protocol (MuSGD,
    cosine LR, ``amp=False``, ``nbs=64``, deterministic, ``seed=0``).

    Args:
        args: Parsed CLI arguments.
        device: Device specification produced by :func:`parse_device`.
        run_name: Sub-directory name used by Ultralytics for this tune.

    Returns:
        A dictionary of keyword arguments suitable for ``model.tune(**kw)``.
    """
    return dict(
        # Infrastructure
        data=args.data,
        project=args.project,
        name=run_name,
        task="segment",
        pretrained=True,
        imgsz=640,

        # Hardware. The micro-batch is configured via CLI (default 32). With
        # ``nbs=64`` (gradient accumulation) the effective optimisation batch
        # is held at 64 regardless — ``accumulate = round(nbs/batch)``.
        device=device,
        batch=args.batch,
        workers=8,
        cache=False,

        # Fixed optimisation protocol (the "v7" recipe).
        # AMP is disabled to match Phases 1, 3 and 4. Mixed precision
        # caused FP16 overflow (NaN in the cls-loss) on the xlarge variant
        # for this dataset; disabling AMP uniformly preserves comparability.
        amp=False,
        optimizer="MuSGD",
        cos_lr=True,
        close_mosaic=15,
        erasing=0.0,
        nbs=64,

        # Short loop for HPO trials.
        epochs=args.epochs,
        patience=args.patience,
        deterministic=True,
        seed=0,
        save=True,
        plots=False,
        val=True,
        verbose=False,
    )


def _print_tune_header(model_size: str, args: argparse.Namespace) -> None:
    """Print the per-model HPO banner."""
    print("\n" + "=" * 80)
    print(
        f"=== TUNE START: {model_size}  "
        f"({args.iterations} iter x {args.epochs} ep)"
    )
    print("=" * 80)


def tune_one_model(
    model_size: str,
    args: argparse.Namespace,
    device: DeviceArg,
    space: dict[str, tuple],
) -> dict:
    """Run ``model.tune()`` for a single variant and return run statistics.

    Skips when the canonical ``best_hyperparameters.yaml`` already exists and
    ``--force`` was not passed.

    Args:
        model_size: Variant name (one of :data:`DEFAULT_ORDER`).
        args: Parsed CLI arguments.
        device: Device specification produced by :func:`parse_device`.
        space: Search-space mapping from :func:`get_search_space`.

    Returns:
        Summary dict with keys ``model``, ``skipped``, ``reason``,
        ``elapsed_min``.
    """
    _, run_name, best_yaml = _tune_paths(args.project, model_size)
    if best_yaml.exists() and not args.force:
        return {
            "model": model_size,
            "skipped": True,
            "reason": f"{best_yaml} already exists (use --force to re-tune)",
            "elapsed_min": 0.0,
        }

    _print_tune_header(model_size, args)
    t0 = time.perf_counter()
    model = YOLO(WEIGHTS[model_size])
    fixed_hp = _build_fixed_hp(args, device, run_name)
    model.tune(
        space=space,
        iterations=args.iterations,
        use_ray=False,
        **fixed_hp,
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
def _print_run_header(
    args: argparse.Namespace,
    device: DeviceArg,
    space: dict[str, tuple],
) -> None:
    """Print the top-level orchestration summary."""
    print(f"HPO for: {args.models}")
    print(f"  iterations/model = {args.iterations}")
    print(f"  epochs/trial     = {args.epochs}")
    print(
        f"  batch/trial      = {args.batch}  "
        f"(nbs=64 — effective optim batch fixed at 64)"
    )
    print(f"  device           = {device}")
    print(f"  data             = {args.data}")
    print(f"  project          = {args.project}")
    print(f"  search space     = {args.space!r} ({len(space)} hp)")
    print(f"  force re-run     = {args.force}")


def _print_run_summary(
    summary: list[dict],
    failures: list[tuple[str, str]],
    total_min: float,
) -> None:
    """Print the final per-model summary table and total wall time."""
    print("\n" + "=" * 80)
    print("=== HPO — SUMMARY")
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
        print(f"\n{len(failures)} model(s) failed:", [m for m, _ in failures])


def main() -> int:
    """Run Phase 2 sequentially over the requested models.

    Returns:
        ``0`` on full success (including skipped models), ``1`` if any
        model raised an exception.
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
        except Exception as e:
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
