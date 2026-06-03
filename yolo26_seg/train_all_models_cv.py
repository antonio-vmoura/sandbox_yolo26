"""Phase 4 — K-Fold cross-validation of YOLO26-seg on ISIC 2018 Task 1.

This script is the sibling of :mod:`train_all_models` but, instead of a single
train/val fine-tune, it performs ``k`` trainings per variant using a
deterministic K-Fold partition over the image+label pairs of the YOLO dataset.
For each variant the script writes a CSV with per-fold metrics and a JSON with
mean ± std for mAP50, mAP50-95, precision, recall and F1-Score (Box and Mask).

Design notes:

* **Same orchestration philosophy as Phase 3**: sequential iteration over the
  variants (``nano``, ``small``, ``medium``, ``large``, ``xlarge``), with
  ``--models``, ``--force``, DDP via ``--device 0,1`` and the same logs
  hierarchy under ``--project /workspace/logs``.
* **Tuned hyperparameters are loaded dynamically** from
  ``<project>/hpo/hpo_v3/tune_isic_2018_task_1_<MODEL>/best_hyperparameters.yaml``
  (the same path emitted by Phase 2).
* **Deterministic K-Fold split** (``seed=0``) over the union of the
  ``train`` and ``val`` pools described by the original ``data.yaml``. The
  original ``test`` pool is **not** touched, so test-set leakage is structurally
  impossible. The implementation matches the bit-exact ordering of
  ``sklearn.model_selection.KFold(shuffle=True, random_state=seed)`` without
  the scikit-learn dependency.
* **Per-fold dataset materialisation**: each fold writes a ``data.yaml``
  pointing at a pair of plain ``train.txt`` / ``val.txt`` files (the native
  Ultralytics list-of-paths format), and preserves ``nc``/``names`` from the
  original template.

Usage:
    # K-Fold CV on ALL five sizes (default)::

        python train_all_models_cv.py

    # CV only on the small variant (pipeline smoke-test)::

        python train_all_models_cv.py --models small

    # Force re-execution even when ``metrics_summary.json`` already exists::

        python train_all_models_cv.py --models small --force

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
            python /workspace/yolo26_seg/train_all_models_cv.py --models small \\
            2>&1 | tee logs/train_all_models_cv_small_v1.log
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterable, Union

import numpy as np
import yaml
from ultralytics import YOLO

# ----------------------------------------------------------------------------
# Module-level configuration
# ----------------------------------------------------------------------------
#: Suffix used in the canonical CV output directory name.
VERSION: str = "cv_v1"

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

#: Image file extensions scanned when building the K-Fold pool.
IMAGE_EXTENSIONS: tuple[str, ...] = (
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp",
)

#: Default number of folds.
K_FOLDS_DEFAULT: int = 5

#: Default random seed for the K-Fold shuffle (also the trainer seed).
SEED_DEFAULT: int = 0

#: Mapping from short metric keys to the column names emitted by Ultralytics
#: in ``results.csv``. We keep both Box (B) and Mask (M); the best-epoch
#: selection uses Mask (relevant for segmentation).
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

#: Short key used to pick the best epoch (Ultralytics default for segmentation).
BEST_EPOCH_KEY: str = "map5095_m"

#: Type alias for the ``device`` argument accepted by Ultralytics.
DeviceArg = Union[int, str, list[int]]


# ----------------------------------------------------------------------------
# K-Fold splitting utilities
# ----------------------------------------------------------------------------
def load_data_yaml(path: Path) -> dict:
    """Load a YOLO-style ``data.yaml`` (Ultralytics / Roboflow format).

    Args:
        path: Filesystem path to the YAML.

    Returns:
        The parsed mapping.

    Raises:
        ValueError: If the file exists but is empty / parses to ``None``.
    """
    with path.open("r") as f:
        data = yaml.safe_load(f) or {}
    if not data:
        raise ValueError(f"data.yaml is empty at {path}.")
    return data


def _resolve_split_dir(root: Path, value: Any) -> list[Path]:
    """Resolve a ``train`` / ``val`` entry of a ``data.yaml`` into directories.

    Accepts either a single string (relative to the ``path:`` root or
    absolute) or a list of strings. Returns only the entries that exist;
    invalid entries emit a warning but do not abort (mirrors the tolerance
    of Ultralytics' loader).

    Args:
        root: Base directory used to resolve relative entries.
        value: The raw value read from the ``data.yaml`` (``str``, ``list``
            or ``None``).

    Returns:
        A list of existing directories.
    """
    if value is None:
        return []
    candidates: Iterable[Any] = (
        value if isinstance(value, (list, tuple)) else [value]
    )
    dirs: list[Path] = []
    for c in candidates:
        p = Path(c)
        if not p.is_absolute():
            p = (root / p).resolve()
        if p.exists():
            dirs.append(p)
        else:
            print(f"  [warn] split path not found and ignored: {p}")
    return dirs


def collect_image_label_pairs(
    data_yaml: dict,
    base_yaml_path: Path,
) -> list[tuple[Path, Path]]:
    """Collect all ``(image, label)`` pairs from the ``train`` + ``val`` splits.

    The pool is built from the ``train`` and ``val`` entries of the
    original ``data.yaml`` (the ``test`` split is intentionally **not**
    included, to keep the held-out test set untouched across folds).
    The label of an image is resolved by replacing ``/images/`` with
    ``/labels/`` and the extension with ``.txt``, following the YOLO
    convention.

    Args:
        data_yaml: Parsed ``data.yaml`` content (see :func:`load_data_yaml`).
        base_yaml_path: Path of the original ``data.yaml`` (used to anchor
            the ``path`` key when it is missing).

    Returns:
        Sorted list of ``(image_path, label_path)`` tuples. Background
        images (no label) are still kept; Ultralytics accepts them.

    Raises:
        ValueError: If no image directory could be resolved or the
            resulting pool is empty.
    """
    root = Path(data_yaml.get("path", base_yaml_path.parent)).resolve()
    image_dirs: list[Path] = []
    for split_key in ("train", "val"):
        image_dirs.extend(_resolve_split_dir(root, data_yaml.get(split_key)))

    if not image_dirs:
        raise ValueError(
            f"Could not resolve any image directory from {base_yaml_path}. "
            f"Check the train/val/path keys."
        )

    pairs: list[tuple[Path, Path]] = []
    seen: set[Path] = set()
    for img_dir in image_dirs:
        for img_path in sorted(img_dir.rglob("*")):
            if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if img_path in seen:
                continue
            label_path = Path(
                str(img_path).replace("/images/", "/labels/"),
            ).with_suffix(".txt")
            # Missing labels (= background images) are kept in the pool.
            pairs.append((img_path, label_path))
            seen.add(img_path)

    if not pairs:
        raise ValueError(
            f"Empty CV pool. Verify that {image_dirs} contains files with "
            f"extensions {IMAGE_EXTENSIONS}."
        )
    return pairs


def build_kfold_splits(
    pairs: list[tuple[Path, Path]],
    k: int,
    seed: int,
) -> list[tuple[list[Path], list[Path]]]:
    """Generate ``k`` deterministic K-Fold splits over the image pool.

    Equivalent to ``sklearn.model_selection.KFold(shuffle=True,
    random_state=seed)`` but without the scikit-learn dependency: indices
    are shuffled with ``numpy.random.RandomState(seed)`` (the same RNG
    used internally by scikit-learn) and partitioned into ``k`` consecutive
    blocks; the first ``n % k`` blocks receive one extra element. Same
    seed → same per-fold sets (verified bit-exact against
    ``sklearn.KFold``).

    Args:
        pairs: ``[(image, label), ...]`` pool returned by
            :func:`collect_image_label_pairs`.
        k: Number of folds (>= 2).
        seed: Deterministic seed for the index shuffle.

    Returns:
        A list of length ``k``; each entry is the tuple
        ``(train_images, val_images)`` for that fold.

    Raises:
        ValueError: If ``k < 2`` or the pool has fewer than ``k`` items.
    """
    if k < 2:
        raise ValueError(f"k_folds must be >= 2 (got {k}).")
    n = len(pairs)
    if n < k:
        raise ValueError(
            f"Pool of {n} images is smaller than k={k}. "
            f"Lower --k-folds or use a larger dataset."
        )
    images = [p[0] for p in pairs]

    rng = np.random.RandomState(seed)
    indices = np.arange(n)
    rng.shuffle(indices)

    fold_sizes = np.full(k, n // k, dtype=int)
    fold_sizes[: n % k] += 1

    splits: list[tuple[list[Path], list[Path]]] = []
    start = 0
    for size in fold_sizes:
        stop = start + size
        val_idx = indices[start:stop]
        train_idx = np.concatenate([indices[:start], indices[stop:]])
        splits.append(
            ([images[i] for i in train_idx], [images[i] for i in val_idx]),
        )
        start = stop
    return splits


def write_fold_dataset(
    fold_dir: Path,
    train_images: list[Path],
    val_images: list[Path],
    template_yaml: dict,
) -> Path:
    """Materialise ``train.txt``, ``val.txt`` and ``data.yaml`` for one fold.

    The generated ``data.yaml`` preserves ``nc``/``names`` from the template
    and points ``train``/``val`` to the absolute paths of the two listing
    files (a format natively supported by Ultralytics).

    Args:
        fold_dir: Output directory for the fold's listing files and YAML.
        train_images: Image paths assigned to the training set.
        val_images: Image paths assigned to the validation set.
        template_yaml: Original ``data.yaml`` content (used to copy
            ``nc``/``names``).

    Returns:
        Path to the ``data.yaml`` written for this fold.
    """
    fold_dir.mkdir(parents=True, exist_ok=True)
    train_txt = fold_dir / "train.txt"
    val_txt = fold_dir / "val.txt"
    train_txt.write_text("\n".join(str(p) for p in train_images) + "\n")
    val_txt.write_text("\n".join(str(p) for p in val_images) + "\n")

    fold_yaml = fold_dir / "data.yaml"
    payload: dict = {
        "path": str(fold_dir.resolve()),
        "train": str(train_txt.resolve()),
        "val": str(val_txt.resolve()),
    }
    if "nc" in template_yaml:
        payload["nc"] = template_yaml["nc"]
    if "names" in template_yaml:
        payload["names"] = template_yaml["names"]
    with fold_yaml.open("w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    return fold_yaml


# ----------------------------------------------------------------------------
# Metrics utilities
# ----------------------------------------------------------------------------
def load_tuned_hp(path: Path) -> dict:
    """Load ``best_hyperparameters.yaml`` produced by Ultralytics' Tuner.

    Args:
        path: Path to the YAML file emitted by Phase 2.

    Returns:
        Dictionary of tuned hyperparameters.

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


def parse_best_metrics(results_csv: Path) -> dict[str, float]:
    """Parse a ``results.csv`` and return the metrics of the best epoch.

    The best epoch is selected using the Ultralytics default for
    segmentation (``metrics/mAP50-95(M)``). The F1-Score is derived from
    precision and recall (Box and Mask) using ``2PR/(P+R)``.

    Args:
        results_csv: Path to an Ultralytics-generated ``results.csv``.

    Returns:
        Dict containing eight metric keys (see :data:`METRIC_KEYS`),
        ``f1_b``, ``f1_m`` and ``best_epoch``.

    Raises:
        ValueError: If the CSV is empty.
    """
    rows: list[dict] = []
    with results_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k.strip(): v for k, v in row.items()})
    if not rows:
        raise ValueError(f"results.csv is empty: {results_csv}")

    best_key_csv = METRIC_KEYS[BEST_EPOCH_KEY]
    best_row = max(rows, key=lambda r: float(r.get(best_key_csv, "0") or 0.0))

    out: dict[str, float] = {}
    for short, full in METRIC_KEYS.items():
        out[short] = float(best_row.get(full, "0") or 0.0)
    # Derived F1 (Box and Mask)
    for suffix in ("b", "m"):
        p = out[f"precision_{suffix}"]
        r = out[f"recall_{suffix}"]
        out[f"f1_{suffix}"] = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    out["best_epoch"] = float(best_row.get("epoch", "0") or 0.0)
    return out


def aggregate_fold_metrics(
    per_fold: list[dict[str, float]],
) -> dict[str, dict[str, float]]:
    """Aggregate per-fold metric dicts into ``{key: {mean, std}}``.

    Args:
        per_fold: List of metric dicts (one per fold).

    Returns:
        Mapping from metric key to ``{"mean": <float>, "std": <float>}``.
        Returns an empty dict when ``per_fold`` is empty. ``std`` is the
        population stdev (uses ``statistics.pstdev``).
    """
    summary: dict[str, dict[str, float]] = {}
    if not per_fold:
        return summary
    keys = sorted(per_fold[0].keys())
    for k in keys:
        values = [m.get(k, 0.0) for m in per_fold]
        summary[k] = {
            "mean": statistics.mean(values),
            "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        }
    return summary


def save_metrics_artifacts(
    model_size: str,
    per_fold: list[dict[str, float]],
    summary: dict[str, dict[str, float]],
    out_dir: Path,
) -> tuple[Path, Path]:
    """Persist per-fold metrics (CSV) and aggregated metrics (JSON).

    Args:
        model_size: Variant name being reported.
        per_fold: List of per-fold metric dicts.
        summary: Aggregated ``{key: {mean, std}}`` dict produced by
            :func:`aggregate_fold_metrics`.
        out_dir: Destination directory (created if missing).

    Returns:
        ``(csv_path, json_path)`` — the canonical artefacts used by
        downstream notebooks and by :mod:`consolidate_cv_results`.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "metrics_per_fold.csv"
    json_path = out_dir / "metrics_summary.json"

    if per_fold:
        fieldnames = ["fold", *sorted(per_fold[0].keys())]
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for i, m in enumerate(per_fold):
                row = {"fold": i, **m}
                w.writerow(row)

    payload = {
        "model": model_size,
        "version": VERSION,
        "n_folds": len(per_fold),
        "per_fold": per_fold,
        "summary": summary,
    }
    with json_path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return csv_path, json_path


# ----------------------------------------------------------------------------
# Per-fold training
# ----------------------------------------------------------------------------
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


def train_one_fold(
    weights: str,
    fold_yaml: Path,
    base_kwargs: dict,
    run_name: str,
    project: Path,
) -> Path:
    """Train a single fold and return the Ultralytics ``save_dir``.

    Args:
        weights: Path to the pretrained weights for the model variant.
        fold_yaml: Path to the per-fold ``data.yaml`` produced by
            :func:`write_fold_dataset`.
        base_kwargs: Fixed training protocol (already merged with the
            tuned HPs).
        run_name: Name of the per-fold run subdirectory.
        project: Parent directory for fold runs.

    Returns:
        The directory created by Ultralytics for this run.
    """
    model = YOLO(weights)
    kwargs = {
        **base_kwargs,
        "data": str(fold_yaml),
        "project": str(project),
        "name": run_name,
        "exist_ok": False,
    }
    model.train(**kwargs)
    return Path(model.trainer.save_dir)


# ----------------------------------------------------------------------------
# Per-model orchestration
# ----------------------------------------------------------------------------
def _cv_paths(project: Path | str, model_size: str) -> tuple[Path, Path, Path, Path]:
    """Resolve canonical CV paths for one model.

    Returns:
        ``(cv_root, splits_dir, runs_dir, hp_yaml)`` where:
            * ``cv_root`` is the per-model CV root,
            * ``splits_dir`` stores the per-fold ``data.yaml``/listing files,
            * ``runs_dir`` stores the per-fold training runs, and
            * ``hp_yaml`` is the Phase-2 tuned HP file consumed here.
    """
    cv_root = Path(project) / "cv" / VERSION / f"yolo26_{model_size}_cv_isic_2018"
    splits_dir = cv_root / "splits"
    runs_dir = cv_root / "runs"
    hp_yaml = (
        Path(project)
        / "hpo"
        / "hpo_v3"
        / f"tune_isic_2018_task_1_{model_size}"
        / "best_hyperparameters.yaml"
    )
    return cv_root, splits_dir, runs_dir, hp_yaml


def _build_cv_base_kwargs(args: argparse.Namespace, device: DeviceArg) -> dict:
    """Build the fixed Phase-4 training protocol (later merged with tuned HPs)."""
    return dict(
        task="segment",
        pretrained=True,
        imgsz=args.imgsz,
        device=device,
        batch=args.batch,
        workers=args.workers,
        cache=False,
        amp=False,                  # FP32 — same as Phases 1/2/3
        optimizer="MuSGD",
        cos_lr=True,
        close_mosaic=10,
        erasing=0.4,
        nbs=64,                     # effective optim batch = 64
        epochs=args.epochs,
        patience=args.patience,
        deterministic=True,
        seed=args.seed,
        save=True,
        plots=True,
        val=True,
        verbose=True,
    )


def _print_cv_model_header(
    model_size: str,
    args: argparse.Namespace,
    pairs_count: int,
    hp_yaml: Path,
    weights_path: str,
    cv_root: Path,
    tuned_hp: dict,
) -> None:
    """Print the per-model CV banner."""
    print("\n" + "=" * 80)
    print(f"=== CROSS-VALIDATION {VERSION}: {model_size}")
    print(f"  k_folds      = {args.k_folds}   seed = {args.seed}")
    print(
        f"  pool_size    = {pairs_count} images "
        f"(train+val of the original data.yaml)"
    )
    print(f"  HP source    = {hp_yaml}")
    print(f"  weights      = {weights_path}")
    print(f"  cv_root      = {cv_root}")
    print("  Tuned hyperparameters:")
    for k, v in sorted(tuned_hp.items()):
        print(f"    {k:18s} = {v}")
    print("=" * 80)


def _run_or_load_fold(
    k: int,
    splits_dir: Path,
    runs_dir: Path,
    train_imgs: list[Path],
    val_imgs: list[Path],
    data_yaml: dict,
    weights_path: str,
    base_kwargs: dict,
    args: argparse.Namespace,
) -> dict[str, float]:
    """Run training (or load cached results) for a single fold.

    If ``runs_dir/fold_<k>/results.csv`` already exists and ``--force`` was
    not given, the cached results are parsed instead of retraining the fold.

    Args:
        k: Fold index.
        splits_dir: Root for per-fold dataset materialisation.
        runs_dir: Root for per-fold trainer runs.
        train_imgs: Image paths for the training set of this fold.
        val_imgs: Image paths for the validation set of this fold.
        data_yaml: Original ``data.yaml`` content.
        weights_path: Pretrained weights for the model variant.
        base_kwargs: Fixed training protocol (already merged with tuned HPs).
        args: Parsed CLI arguments.

    Returns:
        Parsed best-epoch metrics for this fold.
    """
    fold_dir = splits_dir / f"fold_{k}"
    fold_yaml = write_fold_dataset(fold_dir, train_imgs, val_imgs, data_yaml)

    run_name = f"fold_{k}"
    existing_csv = runs_dir / run_name / "results.csv"
    if existing_csv.exists() and not args.force:
        print(
            f"\n[fold {k}/{args.k_folds - 1}] cached results found "
            f"— skipping training",
        )
        return parse_best_metrics(existing_csv)

    print(
        f"\n[fold {k}/{args.k_folds - 1}] "
        f"train={len(train_imgs)} val={len(val_imgs)} → {fold_yaml}",
    )
    save_dir = train_one_fold(
        weights=weights_path,
        fold_yaml=fold_yaml,
        base_kwargs=base_kwargs,
        run_name=run_name,
        project=runs_dir,
    )
    return parse_best_metrics(save_dir / "results.csv")


def cross_validate_one_model(
    model_size: str,
    args: argparse.Namespace,
    device: DeviceArg,
    data_yaml: dict,
    pairs: list[tuple[Path, Path]],
) -> dict:
    """Run the full K-Fold CV for a single model variant.

    Args:
        model_size: Variant name (one of :data:`DEFAULT_ORDER`).
        args: Parsed CLI arguments.
        device: Device specification produced by :func:`parse_device`.
        data_yaml: Parsed original ``data.yaml`` content.
        pairs: Image-label pool produced by :func:`collect_image_label_pairs`.

    Returns:
        Summary dict with keys ``model``, ``skipped``, ``reason``,
        ``elapsed_min`` and, when not skipped, ``summary``, ``csv_path``
        and ``json_path``.
    """
    cv_root, splits_dir, runs_dir, hp_yaml = _cv_paths(args.project, model_size)
    metrics_dir = cv_root
    summary_json = metrics_dir / "metrics_summary.json"
    if summary_json.exists() and not args.force:
        return {
            "model": model_size,
            "skipped": True,
            "reason": f"{summary_json} already exists (use --force to re-run)",
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

    weights_path = args.weights_override or WEIGHTS[model_size]
    tuned_hp = load_tuned_hp(hp_yaml)
    splits = build_kfold_splits(pairs, k=args.k_folds, seed=args.seed)
    _print_cv_model_header(
        model_size, args, len(pairs), hp_yaml, weights_path, cv_root, tuned_hp,
    )

    base_kwargs = {**_build_cv_base_kwargs(args, device), **tuned_hp}

    t0 = time.perf_counter()
    per_fold_metrics: list[dict[str, float]] = []

    for k, (train_imgs, val_imgs) in enumerate(splits):
        metrics = _run_or_load_fold(
            k=k,
            splits_dir=splits_dir,
            runs_dir=runs_dir,
            train_imgs=train_imgs,
            val_imgs=val_imgs,
            data_yaml=data_yaml,
            weights_path=weights_path,
            base_kwargs=base_kwargs,
            args=args,
        )
        per_fold_metrics.append(metrics)
        print(
            f"  fold {k}: "
            f"mAP50(M)={metrics['map50_m']:.4f} "
            f"mAP50-95(M)={metrics['map5095_m']:.4f} "
            f"P(M)={metrics['precision_m']:.4f} "
            f"R(M)={metrics['recall_m']:.4f} "
            f"F1(M)={metrics['f1_m']:.4f}",
        )

    summary = aggregate_fold_metrics(per_fold_metrics)
    csv_path, json_path = save_metrics_artifacts(
        model_size, per_fold_metrics, summary, metrics_dir,
    )

    elapsed = (time.perf_counter() - t0) / 60
    print(f"\n  [{model_size}] artefacts: {csv_path}  |  {json_path}")
    return {
        "model": model_size,
        "skipped": False,
        "reason": None,
        "elapsed_min": elapsed,
        "summary": summary,
        "csv_path": str(csv_path),
        "json_path": str(json_path),
    }


# ----------------------------------------------------------------------------
# CLI & top-level orchestration
# ----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the Phase 4 orchestrator.

    Returns:
        The parsed ``argparse.Namespace``.
    """
    p = argparse.ArgumentParser(
        description="Sequential K-Fold cross-validation of YOLO26-seg variants.",
    )
    p.add_argument(
        "--models", nargs="+", default=DEFAULT_ORDER, choices=DEFAULT_ORDER,
        help=f"Subset of models to cross-validate (default: {DEFAULT_ORDER}).",
    )
    p.add_argument(
        "--data",
        default="/workspace/datasets/isic_2018_task1_yolo26/data.yaml",
        help="Path to the original data.yaml (with train/val/test YOLO splits).",
    )
    p.add_argument(
        "--device", default="0,1",
        help="GPU IDs (default: '0,1' DDP). Use '0' for single-GPU or 'cpu'.",
    )
    p.add_argument(
        "--project", default="/workspace/logs",
        help="Root directory for logs (default: /workspace/logs).",
    )
    p.add_argument(
        "--k-folds", type=int, default=K_FOLDS_DEFAULT,
        help=f"Number of folds (default: {K_FOLDS_DEFAULT}).",
    )
    p.add_argument(
        "--seed", type=int, default=SEED_DEFAULT,
        help=f"Deterministic seed for the split (default: {SEED_DEFAULT}).",
    )
    p.add_argument(
        "--epochs", type=int, default=120,
        help="Epochs per fold (default: 120, matching train_all_models.py).",
    )
    p.add_argument(
        "--patience", type=int, default=25,
        help="Early-stopping patience per fold (default: 25).",
    )
    p.add_argument(
        "--batch", type=int, default=16,
        help="Per-GPU batch size per fold (default: 16).",
    )
    p.add_argument(
        "--imgsz", type=int, default=640,
        help="Input image size (default: 640).",
    )
    p.add_argument(
        "--workers", type=int, default=4,
        help="Dataloader workers per fold (default: 4).",
    )
    p.add_argument(
        "--weights-override", default=None,
        help="Override the pretrained weights path (smoke-test convenience).",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Re-run folds/models even when artefacts already exist.",
    )
    return p.parse_args()


def _print_run_header(
    args: argparse.Namespace,
    device: DeviceArg,
    base_yaml: Path,
    pool_size: int,
) -> None:
    """Print the top-level orchestration summary."""
    print(f"Phase 4 (CV) orchestration {VERSION} for: {args.models}")
    print(f"  device       = {device}")
    print(f"  data         = {base_yaml}")
    print(f"  project      = {args.project}")
    print(f"  k_folds      = {args.k_folds}   seed = {args.seed}")
    print(f"  pool         = {pool_size} images")
    print(f"  force re-run = {args.force}")


def _format_mean_std(agg: dict, key: str) -> str:
    """Format a ``{mean, std}`` entry for the summary table."""
    m = agg.get(key, {})
    return f"{m.get('mean', 0):.4f}±{m.get('std', 0):.4f}"


def _print_run_summary(
    summary: list[dict],
    failures: list[tuple[str, str]],
    total_min: float,
) -> None:
    """Print the final per-model CV summary table."""
    print("\n" + "=" * 80)
    print(f"=== CROSS-VALIDATION SUMMARY {VERSION}")
    print("=" * 80)
    for s in summary:
        if s.get("skipped"):
            print(f"  {s['model']:<8} : skipped ({s['reason']})")
        elif s.get("reason") == "fail":
            print(f"  {s['model']:<8} : FAILED")
        else:
            agg = s.get("summary", {})
            print(
                f"  {s['model']:<8} : ok       ({s['elapsed_min']:.1f} min) | "
                f"mAP50(M)={_format_mean_std(agg, 'map50_m')}  "
                f"mAP50-95(M)={_format_mean_std(agg, 'map5095_m')}  "
                f"P(M)={_format_mean_std(agg, 'precision_m')}  "
                f"R(M)={_format_mean_std(agg, 'recall_m')}  "
                f"F1(M)={_format_mean_std(agg, 'f1_m')}",
            )

    print(f"\nTotal Phase 4 time: {total_min:.1f} min ({total_min / 60:.2f} h)")

    if failures:
        print(f"\n[!] {len(failures)} failure(s):")
        for m, _ in failures:
            print(f"    - Model {m}")


def main() -> int:
    """Run Phase 4 sequentially over the requested models.

    Returns:
        ``0`` on full success (including skipped models), ``1`` if any
        model raised an exception, ``2`` if the ``data.yaml`` cannot be
        located.
    """
    args = parse_args()
    device = parse_device(args.device)

    base_yaml = Path(args.data).resolve()
    if not base_yaml.exists():
        print(f"[error] data.yaml not found: {base_yaml}")
        return 2
    data_yaml = load_data_yaml(base_yaml)
    pairs = collect_image_label_pairs(data_yaml, base_yaml)
    _print_run_header(args, device, base_yaml, len(pairs))

    summary: list[dict] = []
    failures: list[tuple[str, str]] = []
    t_total = time.perf_counter()

    for i, m in enumerate(args.models, 1):
        print(f"\n[{i}/{len(args.models)}] Cross-validating model: {m}")
        try:
            stats = cross_validate_one_model(m, args, device, data_yaml, pairs)
            summary.append(stats)
            if stats["skipped"]:
                print(f"  [skip] {stats['reason']}")
            else:
                print(f"  [done] CV elapsed: {stats['elapsed_min']:.1f} min")
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
