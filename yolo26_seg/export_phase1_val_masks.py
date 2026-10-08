"""Export the Phase 1 (Baseline) validation masks of YOLO26 for the side-by-side figure.

Predicts the 100 official validation images with each Phase 1 ``best.pt`` through the pipeline's own pixel
path (:func:`segmentation_metrics.evaluate_images`: top-1 instance, ``PIXEL_CONF``, ``retina_masks``, FP32) and
writes, per model::

    <project>/phase1_val_masks/<model>/masks/<ISIC_ID>.png   # 0/255, dataset resolution
    <project>/phase1_val_masks/<model>/per_image.csv         # id + every pixel_scores key
    <project>/phase1_val_masks/<model>/meta.json

Read by ``analysis/results_aggregator.py`` (``fig_phase1_side_by_side``). Runs on CPU by default (≈ 15 s per
model), so it never competes with a training run for GPU memory. Validation data only; the test set is untouched.

Usage:
    python yolo26_seg/export_phase1_val_masks.py --models nano
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from ultralytics import YOLO

from common import DEFAULT_ORDER, DEFAULT_PIPELINE_ROOT, IMGSZ, SEED, PipelinePaths, seed_everything, sha256_file
from segmentation_metrics import PIXEL_CONF, aggregate_scores, evaluate_images

VAL_IMAGES = "/workspace/datasets/isic2018_task1_official/valid/images"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--models", nargs="+", default=["nano"], choices=DEFAULT_ORDER)
    p.add_argument("--project", default=DEFAULT_PIPELINE_ROOT)
    p.add_argument("--images", default=VAL_IMAGES, help="Official validation images (n = 100).")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    seed_everything(SEED)
    torch.set_num_threads(8)
    images = sorted(q for q in Path(args.images).iterdir() if q.suffix.lower() in {".png", ".jpg", ".jpeg"})
    assert len(images) == 100, f"expected the 100 official validation images, found {len(images)}"
    paths = PipelinePaths(Path(args.project))
    for m in args.models:
        weights = paths.phase1_best_pt(m)
        out = Path(args.project) / "phase1_val_masks" / m
        rows = evaluate_images(YOLO(str(weights)), images, conf=PIXEL_CONF, imgsz=IMGSZ, half=False,
                               device=args.device, mask_dir=out / "masks")
        with (out / "per_image.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["id", *rows[0].keys()])
            w.writeheader()
            w.writerows({"id": Path(r["image"]).stem, **r} for r in rows)
        agg = aggregate_scores(rows, seed=SEED)
        (out / "meta.json").write_text(json.dumps({
            "arch": "yolo26", "model": m, "label": f"YOLO26-{m}", "split": "val (official, n=100)",
            "source": "Phase 1 best.pt", "weights": str(weights), "weights_sha256": sha256_file(weights),
            "rule": f"top-1 instance, conf >= {PIXEL_CONF}", "device": args.device,
            "jsi_mean": agg["jsi"]["mean"], "dsc_mean": agg["dsc"]["mean"]}, indent=2))
        print(f"{m}: JSI={agg['jsi']['mean']:.4f} DSC={agg['dsc']['mean']:.4f} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
