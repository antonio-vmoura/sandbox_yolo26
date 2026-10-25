#!/usr/bin/env python3
"""Strictly pixel-wise DSC / JSI for the WVC 2026 paper checkpoints.

Depends only on ultralytics, numpy, cv2. Writes ../pixelwise/ (pixelwise_metrics.json + per_image/*.csv).

Inputs, relative to the repository root (override with environment variables):
  WVC_RUNS      logs/pipeline_e2e_v3                pipeline outputs with the weights/best.pt of every run
  WVC_DATA      datasets/isic_2018_task1_yolo26     640x640 training dataset (train/valid/test images)
  WVC_ISIC_RAW  ../datasets/ISIC2018_Raw            official ISIC 2018 Task 1 release (ground-truth masks)
The CV validation images of each fold are taken from ../summary/dataset_and_cv_splits.csv.

Protocol
--------
* Inference runs on the exact 640x640 images the models were trained on
  (aspect ratio not preserved) with conf=0.001, iou=0.7, imgsz=640, retina_masks.
* Single-lesion rule: only the highest-confidence instance (top-1) is kept.
  No detection -> empty mask (DSC = JSI = 0 for that image).
* The predicted 640x640 mask is resized back to the ORIGINAL image resolution
  (undoing the stretch) and compared against the ORIGINAL ISIC 2018 Task 1
  ground-truth PNG (ISIC2018_Raw), binarised at >127.
* Per-image DSC = 2|P∩G| / (|P|+|G|), JSI = |P∩G| / |P∪G|; reported as the
  mean over images (ISIC convention). Dataset-pooled ("micro") values are
  also stored.
"""
import argparse
import csv
import json
import os
import re
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

WVC = Path(__file__).resolve().parents[1]           # .../reproducibility/wvc2026
ROOT = WVC.parents[1]                               # repository root
RUNS = Path(os.environ.get("WVC_RUNS", ROOT / "logs/pipeline_e2e_v3"))
RF = Path(os.environ.get("WVC_DATA", ROOT / "datasets/isic_2018_task1_yolo26"))
RAW = Path(os.environ.get("WVC_ISIC_RAW", ROOT.parent / "datasets/ISIC2018_Raw"))
SPLITS = WVC / "summary/dataset_and_cv_splits.csv"
GT_DIRS = [RAW / f"ISIC2018_Task1_{s}_GroundTruth" for s in ("Training", "Validation", "Test")]
SCALES = ["nano", "small", "medium", "large", "xlarge"]

INFER = dict(conf=0.001, iou=0.7, imgsz=640, retina_masks=True, max_det=300, half=False, verbose=False)


def isic_id(path):
    return re.match(r"(ISIC_\d+)", Path(path).name).group(1)


_gt_index = None


def gt_path(iid):
    global _gt_index
    if _gt_index is None:
        _gt_index = {p.name.split("_segmentation")[0]: p for d in GT_DIRS for p in d.glob("*_segmentation.png")}
    return _gt_index[iid]


def dice_jaccard(pred, gt):
    inter = np.logical_and(pred, gt).sum(dtype=np.int64)
    ps, gs = pred.sum(dtype=np.int64), gt.sum(dtype=np.int64)
    union = ps + gs - inter
    dsc = 1.0 if ps + gs == 0 else 2.0 * inter / (ps + gs)
    jsi = 1.0 if union == 0 else inter / union
    return dsc, jsi, int(inter), int(ps), int(gs), int(union)


def evaluate(weights, images, device, batch):
    model = YOLO(str(weights))
    rows = []
    for i in range(0, len(images), batch):
        chunk = [str(p) for p in images[i:i + batch]]
        for img_path, r in zip(chunk, model.predict(chunk, device=device, **INFER)):
            iid = isic_id(img_path)
            gt = cv2.imread(str(gt_path(iid)), cv2.IMREAD_GRAYSCALE) > 127
            h, w = gt.shape
            top_conf = None
            if r.masks is not None and len(r.boxes):
                k = int(r.boxes.conf.argmax())
                top_conf = float(r.boxes.conf[k])
                m = r.masks.data[k].cpu().numpy().astype(np.uint8) * 255
                pred = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR) > 127
            else:
                pred = np.zeros_like(gt)
            dsc, jsi, inter, ps, gs, union = dice_jaccard(pred, gt)
            rows.append(dict(id=iid, dsc=dsc, jsi=jsi, top_conf=top_conf,
                             inter=inter, pred_px=ps, gt_px=gs, union=union))
    return summarise(rows), rows


def summarise(rows):
    d = np.array([r["dsc"] for r in rows])
    j = np.array([r["jsi"] for r in rows])
    inter = sum(r["inter"] for r in rows)
    ps = sum(r["pred_px"] for r in rows)
    gs = sum(r["gt_px"] for r in rows)
    union = sum(r["union"] for r in rows)
    confs = [r["top_conf"] for r in rows]
    return dict(
        n_images=len(rows),
        dsc_mean=float(d.mean()), dsc_median=float(np.median(d)),
        jsi_mean=float(j.mean()), jsi_median=float(np.median(j)),
        jsi_thresholded_0p65=float(np.where(j < 0.65, 0.0, j).mean()),  # ISIC 2018 official score
        dsc_pooled=2 * inter / (ps + gs), jsi_pooled=inter / union,
        n_no_detection=sum(c is None for c in confs),
        n_top1_conf_below_0p25=sum(c is not None and c < 0.25 for c in confs),
        n_dsc_below_0p5=int((d < 0.5).sum()),
    )


def list_dir(d):
    return sorted(p for p in Path(d).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))


def fold_val_images(fold):
    # The CV split is identical for all scales: pool = train ∪ val, fold from the split manifest.
    pool = {isic_id(p): p for d in ("train/images", "valid/images") for p in list_dir(RF / d)}
    with open(SPLITS) as fh:
        return [pool[r["image_id"]] for r in csv.DictReader(fh) if r["cv_fold"] == str(fold)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="0")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", default=str(WVC / "pixelwise"))
    ap.add_argument("--only", nargs="*", default=["baseline_val", "optimized_val", "optimized_test", "cv"])
    args = ap.parse_args()
    out = Path(args.out)
    (out / "per_image").mkdir(parents=True, exist_ok=True)
    res_file = out / "pixelwise_metrics.json"
    results = json.loads(res_file.read_text()) if res_file.exists() else {}

    jobs = []  # (group, key, weights, images)
    val_imgs, test_imgs = list_dir(RF / "valid/images"), list_dir(RF / "test/images")
    for s in SCALES:
        if "baseline_val" in args.only:
            jobs.append(("baseline_val", s, RUNS / f"phase1_baseline/yolo26_{s}_baseline/weights/best.pt", val_imgs))
        if "optimized_val" in args.only:
            jobs.append(("optimized_val", s, RUNS / f"yolo26_{s}_ft_isic_2018_v11/weights/best.pt", val_imgs))
        if "optimized_test" in args.only:
            jobs.append(("optimized_test", s, RUNS / f"yolo26_{s}_ft_isic_2018_v11/weights/best.pt", test_imgs))
        if "cv" in args.only:
            for f in range(5):
                jobs.append(("cv", f"{s}/fold_{f}",
                             RUNS / f"cv/cv_v1/yolo26_{s}_cv_isic_2018/runs/fold_{f}/weights/best.pt",
                             fold_val_images(f)))

    for group, key, w, imgs in jobs:
        if key in results.get(group, {}):
            print(f"[skip] {group}/{key} (cached)")
            continue
        t0 = time.time()
        summ, rows = evaluate(w, imgs, args.device, args.batch)
        summ.update(weights=str(w.relative_to(RUNS)), seconds=round(time.time() - t0, 1))
        results.setdefault(group, {})[key] = summ
        with open(out / "per_image" / f"{group}__{key.replace('/', '__')}.csv", "w") as fh:
            fh.write(",".join(rows[0]) + "\n")
            for r in rows:
                fh.write(",".join("" if v is None else str(v) for v in r.values()) + "\n")
        res_file.write_text(json.dumps(results, indent=2))
        print(f"[done] {group}/{key}: n={summ['n_images']} DSC={summ['dsc_mean']:.4f} "
              f"JSI={summ['jsi_mean']:.4f} nodet={summ['n_no_detection']} ({summ['seconds']}s)", flush=True)

    # CV aggregation across folds (mean ± sample std, ddof=1)
    if "cv" in results:
        agg = {}
        for s in SCALES:
            folds = [results["cv"].get(f"{s}/fold_{f}") for f in range(5)]
            if any(x is None for x in folds):
                continue
            agg[s] = {}
            for m in ("dsc_mean", "jsi_mean", "dsc_pooled", "jsi_pooled"):
                v = np.array([x[m] for x in folds])
                agg[s][m] = dict(mean=float(v.mean()), std_ddof1=float(v.std(ddof=1)),
                                 std_ddof0=float(v.std()), per_fold=v.tolist())
        results["cv_aggregate"] = agg
        res_file.write_text(json.dumps(results, indent=2))
    print(f"Saved {res_file}")


if __name__ == "__main__":
    main()
