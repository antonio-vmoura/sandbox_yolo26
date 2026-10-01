"""Phase 2 (post-step) — Pixel-level DSC/JSI of every CV fold on its held-out fold.

Phase 2 trains ``k`` models per variant; Ultralytics only reports instance
metrics (mAP/P/R) for them. This script scores each fold's ``best.pt`` on that
fold's validation images with the same pixel metrics as Phase 5
(:mod:`segmentation_metrics`, FP32, ``conf=0.25``), so DSC/JSI are reported for
every phase. The test set is never used here.

Outputs (per model)::

    <project>/phase2_cv_<protocol>/yolo26_<model>/
    ├── pixel_metrics_per_fold.csv    # per-fold means of DSC, JSI, ...
    └── pixel_metrics_summary.json    # mean ± sample std (ddof=1) across folds

Up-to-date results (same fold weights and settings) are skipped.

Usage:
    python evaluate_cv_pixels.py --project /workspace/logs/pipeline_final_v1 --device 0
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import traceback
from pathlib import Path

from common import (
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
from segmentation_metrics import SCORE_KEYS, aggregate_scores, evaluate_images
from evaluate_test_set import EVAL_VERSION
from training import RUN_STATE_FILE


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Pixel-level DSC/JSI of each CV fold on its held-out fold.")
    p.add_argument("--models", nargs="+", default=DEFAULT_ORDER, choices=DEFAULT_ORDER)
    p.add_argument("--protocol", choices=["baseline", "optimized"], default="baseline")
    p.add_argument("--device", default="0", help="Single GPU id (default: 0) or 'cpu'.")
    p.add_argument("--project", default=DEFAULT_PIPELINE_ROOT, help="Pipeline root.")
    p.add_argument("--conf", type=float, default=0.25, help="Mask confidence threshold (default: 0.25).")
    p.add_argument("--force", action="store_true", help="Re-evaluate even if up to date.")
    return p.parse_args()


def evaluate_model(model_size: str, args: argparse.Namespace, device, paths: PipelinePaths) -> None:
    """Score every fold of one model and write the per-fold CSV + summary JSON."""
    from ultralytics import YOLO

    cv_root = paths.cv_model_dir(model_size, args.protocol)
    manifest = read_json(cv_root / "splits_manifest.json")
    if manifest is None:
        raise RuntimeError(f"{cv_root}: no splits_manifest.json — run Phase 2 first")

    folds = []
    for k in range(manifest["k"]):
        run = cv_root / "runs" / f"fold_{k}"
        state = read_json(run / RUN_STATE_FILE)
        if state is None or state.get("status") != "complete":
            raise RuntimeError(f"{run}: fold training not complete")
        folds.append((k, run / "weights" / "best.pt", cv_root / "splits" / f"fold_{k}" / "val.txt"))

    settings = {
        "weights_sha256": [sha256_file(w) for _, w, _ in folds],
        "conf": args.conf, "imgsz": IMGSZ, "manifest": manifest, "eval_version": EVAL_VERSION,
    }
    out_json = cv_root / "pixel_metrics_summary.json"
    previous = read_json(out_json)
    if previous and previous.get("settings_hash") == config_hash(settings) and not args.force:
        print(f"  [skip] {model_size}: pixel metrics up to date")
        return

    per_fold = []
    for k, weights, val_txt in folds:
        images = [Path(p) for p in val_txt.read_text().split()]
        print(f"  [{model_size}] fold {k}: {len(images)} held-out images")
        rows = evaluate_images(YOLO(str(weights)), images, conf=args.conf, imgsz=IMGSZ,
                               half=False, device=device)
        agg = aggregate_scores(rows, seed=SEED)
        per_fold.append({
            "fold": k, "n_images": agg["n_images"], "n_empty_pred": agg["n_empty_pred"],
            "pooled_dsc": agg["pooled_dsc"], "pooled_jsi": agg["pooled_jsi"],
            **{key: agg[key].get("mean", float("nan")) for key in SCORE_KEYS},
        })
        print(f"    DSC={per_fold[-1]['dsc']:.4f}  JSI={per_fold[-1]['jsi']:.4f}")

    with (cv_root / "pixel_metrics_per_fold.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_fold[0].keys()))
        w.writeheader()
        w.writerows(per_fold)

    summary = {}
    for key in (*SCORE_KEYS, "pooled_dsc", "pooled_jsi"):
        vals = [r[key] for r in per_fold]
        summary[key] = {
            "mean": statistics.mean(vals),
            "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        }
    atomic_write_json(out_json, {
        "model": model_size, "protocol": args.protocol, "split": "cv_heldout_fold",
        "n_folds": len(per_fold), "std_ddof": 1, "conf": args.conf,
        "per_fold": per_fold, "summary": summary,
        "settings_hash": config_hash(settings), "created_at": utc_now_iso(),
    })
    print(f"  [{model_size}] DSC={summary['dsc']['mean']:.4f}±{summary['dsc']['std']:.4f}  "
          f"JSI={summary['jsi']['mean']:.4f}±{summary['jsi']['std']:.4f}")


def main() -> int:
    """Run the CV pixel evaluation for every requested model.

    Returns:
        ``0`` on success, ``1`` if any model failed.
    """
    args = parse_args()
    seed_everything()
    device = parse_device(args.device)
    paths = PipelinePaths(Path(args.project))
    print(f"Phase 2 pixel metrics (protocol={args.protocol}) for {args.models} on device {device}")
    failures = 0
    for m in args.models:
        try:
            evaluate_model(m, args, device, paths)
        except Exception:
            failures += 1
            print(f"  [fail] {m}:\n{traceback.format_exc()}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
