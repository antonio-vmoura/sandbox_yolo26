"""Phase 5a — Accuracy of the Baseline and Optimised models on the held-out TEST set.

For every ``variant ∈ {baseline, optimized}`` × ``model`` × ``precision ∈
{fp32, fp16}`` this script evaluates the final ``best.pt`` on the ``test``
split of ``data.yaml`` — the only phase that ever touches it:

1. **Instance metrics** — Ultralytics ``model.val(split="test", batch=1)``:
   P, R, mAP50, mAP50-95 and F1 for Box and Mask (``conf=0.001``, the
   standard mAP protocol).
2. **Pixel metrics** — DSC, JSI, ISIC thresholded JSI, sensitivity,
   specificity, accuracy and the boundary metrics Boundary IoU, NSD and HD95
   per image (:mod:`segmentation_metrics`), with the
   prediction taken as the single highest-confidence instance among those
   with ``conf >= --conf`` (default 0.001; ISIC 2018 Task 1 has one lesion per
   image, so lower-ranked instances are never merged) at the original image
   resolution. Empty predictions are scored as 0 (never skipped).

FP32 is the primary result (the models were trained in FP32); the FP16 run
quantifies the accuracy cost of half-precision deployment, to be read
alongside the FP32-vs-FP16 latency of Phase 5b.

Outputs (per variant/model/precision)::

    <project>/phase5_test/accuracy/<variant>_<model>_<precision>.json
    <project>/phase5_test/per_image/<variant>_<model>_<precision>.csv
    <project>/phase5_test/masks/<variant>_<model>/<stem>.png   # FP32 only
    <project>/phase5_test/val_runs/<variant>_<model>_<precision>/  # Ultralytics plots

Each JSON records the SHA-256 of the weights and of the test-image list; a
re-run skips results whose weights, test set and settings are unchanged.

Usage:
    python evaluate_test_set.py --project /workspace/logs/pipeline_final_v1 --device 0
    python evaluate_test_set.py --models small --variants optimized --precisions fp32
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from common import (
    DEFAULT_DATA_YAML,
    DEFAULT_ORDER,
    DEFAULT_PIPELINE_ROOT,
    IMGSZ,
    SEED,
    PipelinePaths,
    atomic_write_json,
    config_hash,
    parse_device,
    read_json,
    seed_everything,
    sha256_file,
    utc_now_iso,
)
from segmentation_metrics import PIXEL_CONF, aggregate_scores, evaluate_images
from train_all_models_cv import collect_test_images, load_data_yaml
from training import METRIC_KEYS, RUN_STATE_FILE

#: Version of the evaluation method. Part of the cache key: bump it whenever the
#: metric definitions or the evaluation protocol change.
EVAL_VERSION: int = 4   # 2: + boundary metrics (BIoU, NSD); 3: + HD95; 4: top-1 mask at conf 0.001 (was union at 0.25)

VARIANTS: tuple[str, ...] = ("baseline", "optimized")
PRECISIONS: tuple[str, ...] = ("fp32", "fp16")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for Phase 5a.

    Returns:
        The parsed ``argparse.Namespace``.
    """
    p = argparse.ArgumentParser(description="Phase 5a — test-set accuracy (instance + pixel metrics).")
    p.add_argument("--models", nargs="+", default=DEFAULT_ORDER, choices=DEFAULT_ORDER)
    p.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    p.add_argument("--precisions", nargs="+", default=list(PRECISIONS), choices=PRECISIONS)
    p.add_argument("--data", default=DEFAULT_DATA_YAML, help="data.yaml with a 'test' split.")
    p.add_argument("--device", default="0", help="Single GPU id (default: 0) or 'cpu'.")
    p.add_argument("--project", default=DEFAULT_PIPELINE_ROOT, help="Pipeline root.")
    p.add_argument(
        "--conf", type=float, default=PIXEL_CONF,
        help=f"Confidence threshold of the candidate instances; the top-1 is scored (default: {PIXEL_CONF}).",
    )
    p.add_argument("--no-save-masks", action="store_true", help="Do not save predicted masks.")
    p.add_argument("--force", action="store_true", help="Re-evaluate even if results are up to date.")
    return p.parse_args()


def _instance_metrics(results_dict: dict[str, float]) -> dict[str, float]:
    """Map Ultralytics ``results_dict`` to the short keys used across the pipeline (+F1)."""
    out = {short: float(results_dict.get(full, float("nan"))) for short, full in METRIC_KEYS.items()}
    for s in ("b", "m"):
        p, r = out[f"precision_{s}"], out[f"recall_{s}"]
        out[f"f1_{s}"] = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return out


def _require_trained(paths: PipelinePaths, variant: str, model: str) -> Path:
    """Return the weights of a completed training run or raise."""
    weights = paths.best_pt(variant, model)
    state = read_json(weights.parent.parent / RUN_STATE_FILE)
    if state is None or state.get("status") != "complete" or not weights.exists():
        raise RuntimeError(f"{variant}/{model}: training run not complete ({weights.parent.parent})")
    return weights


def evaluate_one(
    variant: str,
    model_size: str,
    precision: str,
    args: argparse.Namespace,
    device: Any,
    paths: PipelinePaths,
    test_images: list[Path],
    test_sha: str,
) -> dict[str, Any]:
    """Evaluate one ``(variant, model, precision)`` on the test set (or skip if current)."""
    import torch
    import ultralytics
    from ultralytics import YOLO

    weights = _require_trained(paths, variant, model_size)
    out_json = paths.phase5_accuracy_json(variant, model_size, precision)
    settings = {
        "weights_sha256": sha256_file(weights), "test_list_sha256": test_sha,
        "conf": args.conf, "imgsz": IMGSZ, "precision": precision,
        "ultralytics_version": ultralytics.__version__, "eval_version": EVAL_VERSION,
    }
    previous = read_json(out_json)
    if previous and previous.get("settings_hash") == config_hash(settings) and not args.force:
        return {"tag": out_json.stem, "skipped": True, "payload": previous}

    half = precision == "fp16"
    tag = paths.phase5_tag(variant, model_size, precision)
    print(f"\n=== {tag}  ({len(test_images)} test images, weights={weights})")
    t0 = time.perf_counter()

    model = YOLO(str(weights))
    val = model.val(
        data=args.data, split="test", imgsz=IMGSZ, batch=1, half=half, device=device,
        plots=True, project=str(paths.phase5_dir / "val_runs"), name=tag, exist_ok=True,
        verbose=False,
    )
    instance = _instance_metrics(val.results_dict)
    print(f"  instance: mAP50-95(M)={instance['map5095_m']:.4f}  F1(M)={instance['f1_m']:.4f}")

    save_masks = precision == "fp32" and not args.no_save_masks
    rows = evaluate_images(
        YOLO(str(weights)), test_images, conf=args.conf, imgsz=IMGSZ, half=half, device=device,
        mask_dir=paths.phase5_mask_dir(variant, model_size) if save_masks else None,
    )
    pixel = aggregate_scores(rows, seed=SEED)
    print(
        f"  pixel   : DSC={pixel['dsc']['mean']:.4f} (95% CI {pixel['dsc']['ci95_low']:.4f}-"
        f"{pixel['dsc']['ci95_high']:.4f})  JSI={pixel['jsi']['mean']:.4f}  "
        f"JSI_thr={pixel['jsi_thr']['mean']:.4f}  empty_pred={pixel['n_empty_pred']}",
    )

    per_image_csv = paths.phase5_per_image_csv(variant, model_size, precision)
    per_image_csv.parent.mkdir(parents=True, exist_ok=True)
    with per_image_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    payload = {
        "variant": variant, "model": model_size, "precision": precision, "split": "test",
        "weights": str(weights), "data": args.data, "n_images": len(test_images),
        "settings": settings, "settings_hash": config_hash(settings),
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(device) if device != "cpu" else "cpu",
        "instance_metrics": instance,
        "instance_conf": 0.001,
        "pixel_metrics": pixel,
        "pixel_conf": args.conf,
        "ultralytics_speed_ms": dict(val.speed),
        "per_image_csv": str(per_image_csv),
        "mask_dir": str(paths.phase5_mask_dir(variant, model_size)) if save_masks else None,
        "elapsed_min": round((time.perf_counter() - t0) / 60, 2),
        "created_at": utc_now_iso(),
    }
    atomic_write_json(out_json, payload)
    return {"tag": tag, "skipped": False, "payload": payload}


def main() -> int:
    """Evaluate every requested combination.

    Returns:
        ``0`` on success, ``1`` if any combination failed, ``2`` if the test
        split cannot be resolved.
    """
    args = parse_args()
    seed_everything()
    device = parse_device(args.device)
    if isinstance(device, list):
        print("[error] Phase 5 must run on a single device (e.g. --device 0).", file=sys.stderr)
        return 2
    paths = PipelinePaths(Path(args.project))
    base_yaml = Path(args.data).resolve()
    try:
        test_images = collect_test_images(load_data_yaml(base_yaml), base_yaml)
    except (OSError, ValueError) as e:
        print(f"[error] {e}", file=sys.stderr)
        return 2
    test_sha = config_hash([str(p) for p in test_images])
    print(f"Phase 5a — test set: {len(test_images)} images ({base_yaml})")
    print(f"  variants={args.variants} models={args.models} precisions={args.precisions} device={device}")

    failures = 0
    for variant in args.variants:
        for m in args.models:
            for precision in args.precisions:
                try:
                    r = evaluate_one(variant, m, precision, args, device, paths, test_images, test_sha)
                    if r["skipped"]:
                        print(f"  [skip] {r['tag']} up to date")
                except Exception:
                    failures += 1
                    print(f"  [fail] {variant}/{m}/{precision}:\n{traceback.format_exc()}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
