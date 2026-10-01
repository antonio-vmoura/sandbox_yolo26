"""Phase 2 — Deterministic K-Fold cross-validation of YOLO26-seg on ISIC 2018 Task 1.

For every requested variant, ``k`` models are trained on a deterministic
K-Fold partition of the **train + val pool** of the original ``data.yaml``;
each fold is evaluated on its held-out fold. The script writes per-fold
metrics (CSV) and mean ± std (JSON) for mAP50, mAP50-95, precision, recall
and F1 (Box and Mask).

Protocols (``--protocol``):

* ``baseline`` (default, **Phase 2**) — :func:`common.baseline_protocol`,
  i.e. exactly the Phase 1 configuration (base setup + default HPs). No HPO
  output is needed.
* ``optimized`` (optional ablation) — :func:`common.optimized_protocol` with
  the Phase 3 hyperparameters (requires a complete HPO).

Design notes:

* **Deterministic split, no scikit-learn.** :func:`build_kfold_splits`
  shuffles indices with ``numpy.random.RandomState(seed)`` and partitions
  them like ``sklearn.model_selection.KFold(shuffle=True, random_state=seed)``
  (bit-exact). The K-Fold helpers are unchanged from the previous pipeline.
* **Test set isolation.** The ``test`` split is never part of the pool; the
  script additionally asserts that no test image (by resolved path or file
  name) appears in any fold, and aborts otherwise.
* **Split manifest.** ``splits_manifest.json`` fingerprints the pool and every
  fold. A re-run whose splits differ (changed dataset, seed or k) is refused,
  so resumed folds can never mix two partitions.
* **Fault tolerance.** Each fold is a resumable run (see :mod:`training`):
  interrupted folds resume from ``last.pt``; completed folds are skipped.
* **Per-fold metrics** are those of the epoch that produced the fold's
  ``best.pt`` (:func:`training.parse_best_metrics`), i.e. the same checkpoint
  that is scored for DSC/JSI by :mod:`evaluate_cv_pixels`.

Outputs::

    <project>/phase2_cv_<protocol>/yolo26_<model>/
    ├── splits/fold_<k>/{train.txt, val.txt, data.yaml}
    ├── splits_manifest.json
    ├── runs/fold_<k>/{weights/, results.csv, run_state.json}
    ├── metrics_per_fold.csv
    └── metrics_summary.json

Usage:
    # Phase 2 — baseline CV for all five sizes::

        python train_all_models_cv.py --project /workspace/logs/pipeline_final_v1

    # Optional ablation — CV of the optimised configuration::

        python train_all_models_cv.py --protocol optimized --models small

    # Re-run from scratch (the old CV dir is moved to *.bak-<UTC>)::

        python train_all_models_cv.py --models small --force
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

from common import (
    DEFAULT_DATA_YAML,
    DEFAULT_ORDER,
    DEFAULT_PIPELINE_ROOT,
    SEED,
    TRAIN_EPOCHS,
    TRAIN_PATIENCE,
    WEIGHTS,
    DeviceArg,
    PipelinePaths,
    atomic_write_json,
    baseline_protocol,
    optimized_protocol,
    parse_device,
    read_json,
    seed_everything,
)
from training import (
    backup_dir,
    load_tuned_hp,
    print_phase_summary,
    require_complete_hpo,
    train_or_resume,
)

# ----------------------------------------------------------------------------
# Module-level configuration
# ----------------------------------------------------------------------------
#: Image file extensions scanned when building the K-Fold pool.
IMAGE_EXTENSIONS: tuple[str, ...] = (
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp",
)

#: Default number of folds.
K_FOLDS_DEFAULT: int = 5


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




def aggregate_fold_metrics(
    per_fold: list[dict[str, float]],
) -> dict[str, dict[str, float]]:
    """Aggregate per-fold metric dicts into ``{key: {mean, std}}``.

    Args:
        per_fold: List of metric dicts (one per fold).

    Returns:
        Mapping from metric key to ``{"mean": <float>, "std": <float>}``.
        Returns an empty dict when ``per_fold`` is empty. ``std`` is the
        **sample** standard deviation (``statistics.stdev``, ddof=1), the
        conventional estimator for K-Fold results; it is 0.0 for a single fold.
    """
    summary: dict[str, dict[str, float]] = {}
    if not per_fold:
        return summary
    keys = sorted(per_fold[0].keys())
    for k in keys:
        values = [m.get(k, 0.0) for m in per_fold]
        summary[k] = {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        }
    return summary




# ----------------------------------------------------------------------------
# Test-set isolation & split manifest
# ----------------------------------------------------------------------------
def collect_test_images(data_yaml: dict, base_yaml_path: Path) -> list[Path]:
    """Return every image of the ``test`` split declared in ``data.yaml``.

    Raises:
        ValueError: If no ``test`` split is declared or it resolves to nothing.
    """
    root = Path(data_yaml.get("path", base_yaml_path.parent)).resolve()
    test_dirs = _resolve_split_dir(root, data_yaml.get("test"))
    if not test_dirs:
        raise ValueError(f"No 'test' split resolvable in {base_yaml_path}.")
    return sorted(
        p.resolve() for d in test_dirs for p in d.rglob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )


def assert_no_test_leakage(pairs: list[tuple[Path, Path]], test_images: list[Path]) -> None:
    """Abort if any test image is part of the CV pool (by resolved path or file name).

    Raises:
        RuntimeError: Listing up to five offending images.
    """
    test_paths = set(test_images)
    test_names = {p.name for p in test_images}
    leaked = [
        img for img, _ in pairs
        if img.resolve() in test_paths or img.name in test_names
    ]
    if leaked:
        raise RuntimeError(
            f"{len(leaked)} test image(s) found in the CV pool, e.g. "
            f"{[str(p) for p in leaked[:5]]}. The test set must stay isolated.",
        )


def _sha256_paths(paths: Iterable[Path]) -> str:
    """Fingerprint an ordered list of paths."""
    return hashlib.sha256("\n".join(str(p) for p in paths).encode()).hexdigest()


def build_splits_manifest(
    pairs: list[tuple[Path, Path]],
    splits: list[tuple[list[Path], list[Path]]],
    k: int,
    seed: int,
    n_test: int,
) -> dict[str, Any]:
    """Describe the pool and every fold so later runs can verify the partition."""
    return {
        "k": k,
        "seed": seed,
        "pool_size": len(pairs),
        "pool_sha256": _sha256_paths(p[0] for p in pairs),
        "test_size_excluded": n_test,
        "folds": [
            {"fold": i, "n_train": len(tr), "n_val": len(va), "val_sha256": _sha256_paths(va)}
            for i, (tr, va) in enumerate(splits)
        ],
    }


def check_or_write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """Write the manifest on first use; refuse to continue if the splits changed.

    Raises:
        RuntimeError: If an existing manifest describes a different partition.
    """
    existing = read_json(path)
    if existing is None:
        atomic_write_json(path, manifest)
        return
    if existing != manifest:
        raise RuntimeError(
            f"K-Fold splits differ from {path} (dataset, k or seed changed). "
            f"Use --force to start this model's CV from scratch.",
        )


# ----------------------------------------------------------------------------
# Metrics artefacts
# ----------------------------------------------------------------------------
def save_metrics_artifacts(
    model_size: str,
    protocol: str,
    per_fold: list[dict[str, float]],
    summary: dict[str, dict[str, float]],
    out_dir: Path,
    hp_source: str | None,
) -> tuple[Path, Path]:
    """Persist per-fold metrics (CSV) and aggregated metrics (JSON).

    Returns:
        ``(csv_path, json_path)`` — consumed by :mod:`consolidate_cv_results`.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "metrics_per_fold.csv"
    json_path = out_dir / "metrics_summary.json"

    fieldnames = ["fold", *sorted(per_fold[0].keys())]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i, m in enumerate(per_fold):
            w.writerow({"fold": i, **m})

    atomic_write_json(json_path, {
        "model": model_size,
        "protocol": protocol,
        "hp_source": hp_source,
        "n_folds": len(per_fold),
        "std_ddof": 1,
        "per_fold": per_fold,
        "summary": summary,
    })
    return csv_path, json_path


# ----------------------------------------------------------------------------
# Per-model orchestration
# ----------------------------------------------------------------------------
def build_protocol(
    model_size: str,
    args: argparse.Namespace,
    device: DeviceArg,
    paths: PipelinePaths,
) -> tuple[dict[str, Any], str | None]:
    """Return ``(train_kwargs, hp_source)`` for the selected ``--protocol``.

    ``data`` is a placeholder; it is replaced by each fold's ``data.yaml``.
    """
    if args.protocol == "baseline":
        return baseline_protocol(args.data, device, args.epochs, args.patience), None
    require_complete_hpo(paths.phase3_state(model_size))
    hp_yaml = paths.phase3_best_yaml(model_size)
    tuned = load_tuned_hp(hp_yaml)
    return optimized_protocol(args.data, device, tuned, args.epochs, args.patience), str(hp_yaml)


def cross_validate_one_model(
    model_size: str,
    args: argparse.Namespace,
    device: DeviceArg,
    data_yaml: dict,
    pairs: list[tuple[Path, Path]],
    n_test: int,
    paths: PipelinePaths,
) -> dict:
    """Run (or resume) the full K-Fold CV for one variant.

    Returns:
        Summary dict with ``model``, ``skipped``, ``resumed``,
        ``elapsed_min``, ``metrics`` (fold means) and ``reason``.
    """
    cv_root = paths.cv_model_dir(model_size, args.protocol)
    if args.force:
        backup_dir(cv_root)
    splits_dir, runs_dir = cv_root / "splits", cv_root / "runs"

    splits = build_kfold_splits(pairs, k=args.k_folds, seed=args.seed)
    check_or_write_manifest(
        cv_root / "splits_manifest.json",
        build_splits_manifest(pairs, splits, args.k_folds, args.seed, n_test),
    )
    base_kwargs, hp_source = build_protocol(model_size, args, device, paths)
    weights = args.weights_override or WEIGHTS[model_size]

    print(f"  protocol = {args.protocol}   hp_source = {hp_source or 'base setup + Ultralytics default HPs'}")
    print(f"  pool     = {len(pairs)} images  (k={args.k_folds}, seed={args.seed})")
    print(f"  cv_root  = {cv_root}")

    t0 = time.perf_counter()
    per_fold: list[dict[str, float]] = []
    any_resumed, all_skipped = False, True
    for k, (train_imgs, val_imgs) in enumerate(splits):
        fold_yaml = write_fold_dataset(splits_dir / f"fold_{k}", train_imgs, val_imgs, data_yaml)
        print(f"\n  [fold {k}/{args.k_folds - 1}] train={len(train_imgs)} val={len(val_imgs)}")
        stats = train_or_resume(
            phase=f"phase2_cv_{args.protocol}", model=model_size, weights=weights,
            train_kwargs={**base_kwargs, "data": str(fold_yaml)},
            project=runs_dir, name=f"fold_{k}",
        )
        any_resumed |= stats["resumed"]
        all_skipped &= stats["skipped"]
        m = stats["metrics"]
        per_fold.append(m)
        print(
            f"  fold {k}: mAP50(M)={m['map50_m']:.4f} mAP50-95(M)={m['map5095_m']:.4f} "
            f"P(M)={m['precision_m']:.4f} R(M)={m['recall_m']:.4f} F1(M)={m['f1_m']:.4f}",
        )

    # Aggregate numeric metrics only (best_epoch_source is kept per fold for traceability).
    summary = aggregate_fold_metrics([
        {k: v for k, v in m.items() if isinstance(v, (int, float))} for m in per_fold
    ])
    csv_path, json_path = save_metrics_artifacts(
        model_size, args.protocol, per_fold, summary, cv_root, hp_source,
    )
    print(f"\n  [{model_size}] artefacts: {csv_path}  |  {json_path}")
    return {
        "model": model_size, "skipped": all_skipped, "resumed": any_resumed, "reason": None,
        "elapsed_min": (time.perf_counter() - t0) / 60,
        "metrics": {key: v["mean"] for key, v in summary.items()},
    }


# ----------------------------------------------------------------------------
# CLI & top-level orchestration
# ----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for Phase 2.

    Returns:
        The parsed ``argparse.Namespace``.
    """
    p = argparse.ArgumentParser(
        description="Phase 2 — deterministic K-Fold cross-validation of YOLO26-seg.",
    )
    p.add_argument(
        "--models", nargs="+", default=DEFAULT_ORDER, choices=DEFAULT_ORDER,
        help=f"Subset of models to cross-validate (default: {DEFAULT_ORDER}).",
    )
    p.add_argument(
        "--protocol", choices=["baseline", "optimized"], default="baseline",
        help="Training configuration per fold (default: baseline = Phase 1 protocol).",
    )
    p.add_argument(
        "--data", default=DEFAULT_DATA_YAML,
        help="Path to the original data.yaml (with train/val/test splits).",
    )
    p.add_argument(
        "--device", default="0,1",
        help="GPU IDs (default: '0,1' DDP). Use '0' for single-GPU or 'cpu'.",
    )
    p.add_argument(
        "--project", default=DEFAULT_PIPELINE_ROOT,
        help=f"Pipeline root (default: {DEFAULT_PIPELINE_ROOT}).",
    )
    p.add_argument(
        "--k-folds", type=int, default=K_FOLDS_DEFAULT,
        help=f"Number of folds (default: {K_FOLDS_DEFAULT}).",
    )
    p.add_argument(
        "--seed", type=int, default=SEED,
        help=f"Seed of the K-Fold shuffle (default: {SEED}).",
    )
    p.add_argument(
        "--epochs", type=int, default=TRAIN_EPOCHS,
        help=f"Epochs per fold (default: {TRAIN_EPOCHS}). Must match Phases 1 and 4.",
    )
    p.add_argument(
        "--patience", type=int, default=TRAIN_PATIENCE,
        help=f"Early-stopping patience per fold (default: {TRAIN_PATIENCE}).",
    )
    p.add_argument(
        "--weights-override", default=None,
        help="Override the pretrained weights path (smoke-test convenience).",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Re-run a model's CV from scratch (moves it to *.bak-<UTC>).",
    )
    return p.parse_args()


def main() -> int:
    """Run Phase 2 sequentially over the requested models.

    Returns:
        ``0`` on success, ``1`` if any model failed, ``2`` if the
        ``data.yaml`` is missing or the test set is not isolated.
    """
    args = parse_args()
    seed_everything(args.seed)
    device = parse_device(args.device)
    paths = PipelinePaths(Path(args.project))

    base_yaml = Path(args.data).resolve()
    if not base_yaml.exists():
        print(f"[error] data.yaml not found: {base_yaml}", file=sys.stderr)
        return 2
    data_yaml = load_data_yaml(base_yaml)
    pairs = collect_image_label_pairs(data_yaml, base_yaml)
    try:
        test_images = collect_test_images(data_yaml, base_yaml)
        assert_no_test_leakage(pairs, test_images)
    except (ValueError, RuntimeError) as e:
        print(f"[error] {e}", file=sys.stderr)
        return 2

    print(f"Phase 2 (CV, protocol={args.protocol}) for: {args.models}")
    print(f"  device = {device}   data = {base_yaml}")
    print(f"  pool   = {len(pairs)} train+val images   test (excluded, verified) = {len(test_images)}")
    print(f"  k      = {args.k_folds}   seed = {args.seed}")
    print(f"  budget = {args.epochs} epochs, patience {args.patience}")
    print(f"  output = {paths.cv_dir(args.protocol)}")

    summary: list[dict] = []
    t0 = time.perf_counter()
    for i, m in enumerate(args.models, 1):
        print("\n" + "=" * 80)
        print(f"=== [{i}/{len(args.models)}] PHASE 2 CV ({args.protocol}): {m}")
        print("=" * 80)
        try:
            summary.append(cross_validate_one_model(
                m, args, device, data_yaml, pairs, len(test_images), paths,
            ))
        except Exception as e:
            print(f"  [fail] {m}:\n{traceback.format_exc()}", file=sys.stderr)
            summary.append({"model": m, "skipped": False, "failed": True, "reason": str(e)})

    print_phase_summary(
        f"PHASE 2 CV ({args.protocol}, fold means)", summary, (time.perf_counter() - t0) / 60,
    )
    return 1 if any(s.get("failed") for s in summary) else 0


if __name__ == "__main__":
    sys.exit(main())
