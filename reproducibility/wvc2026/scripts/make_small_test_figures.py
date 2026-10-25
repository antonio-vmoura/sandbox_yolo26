#!/usr/bin/env python3
"""Qualitative test-set figures for the optimized YOLO26-seg Small model.

Depends only on ultralytics, numpy, cv2 and evaluate_pixelwise_metrics.py (same folder,
same input paths and environment variables), whose protocol it reuses (top-1 instance,
conf=0.001, 640x640 stretched input, mask resized back to the original resolution).
Writes ../figures/.

The five images are chosen by explicit rules over the full test set,
using the per-image scores produced by evaluate_pixelwise_metrics.py:

  1. hair     - highest hair-occlusion score among images with Small DSC >= 0.90
  2. contrast - lowest lesion/skin colour contrast among images with Small DSC >= 0.90
  3. small    - smallest lesion (GT area fraction) among images with Small DSC >= 0.90
  4. rescue   - largest Small-minus-Nano DSC gain where Nano DSC < 0.50
  5. failure  - median-DSC image of the Small failure tail (DSC < 0.50)

Overlay: original image, predicted mask in red (filled, semi-transparent, with
contour) and the expert GT contour in green.
"""
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from evaluate_pixelwise_metrics import INFER, RF, RUNS, dice_jaccard, gt_path, isic_id, list_dir

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "figures"                     # reproducibility/wvc2026/figures
PER_IMAGE = HERE.parent / "pixelwise/per_image"
MAX_SIDE = 800


def load_scores(scale):
    with open(PER_IMAGE / f"optimized_test__{scale}.csv") as fh:
        return {r["id"]: float(r["dsc"]) for r in csv.DictReader(fh)}


def hair_score(img):
    """Fraction of pixels flagged as thin dark structures (black-hat filter)."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17)))
    return float((bh > 20).mean())


def contrast_score(img, gt):
    """CIE Lab Delta-E between mean lesion colour and a surrounding skin ring."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    k = max(3, int(0.05 * max(gt.shape)))
    ring = cv2.dilate(gt.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool) & ~gt
    if ring.sum() == 0:
        return np.inf
    return float(np.linalg.norm(lab[gt].mean(0) - lab[ring].mean(0)))


def select(test_imgs):
    small, nano = load_scores("small"), load_scores("nano")
    by_id = {isic_id(p): p for p in test_imgs}
    feats = {}
    for iid in by_id:
        gt = cv2.imread(str(gt_path(iid)), cv2.IMREAD_GRAYSCALE) > 127
        img = cv2.resize(cv2.imread(str(by_id[iid])), gt.shape[::-1], interpolation=cv2.INTER_LINEAR)
        feats[iid] = dict(hair=hair_score(img), contrast=contrast_score(img, gt), area=float(gt.mean()))
    good = [i for i in by_id if small[i] >= 0.90]
    picks = {}
    picks["hair"] = max(good, key=lambda i: feats[i]["hair"])
    picks["contrast"] = min((i for i in good if i not in picks.values()), key=lambda i: feats[i]["contrast"])
    picks["small"] = min((i for i in good if i not in picks.values()), key=lambda i: feats[i]["area"])
    picks["rescue"] = max((i for i in by_id if nano[i] < 0.5), key=lambda i: small[i] - nano[i])
    tail = sorted((i for i in by_id if small[i] < 0.5), key=lambda i: small[i])
    picks["failure"] = tail[len(tail) // 2]
    return picks, by_id, feats, small, nano


def overlay(img, gt, pred):
    out = img.copy()
    red = np.zeros_like(img)
    red[:] = (0, 0, 255)
    out[pred] = cv2.addWeighted(img, 0.6, red, 0.4, 0)[pred]
    t = max(2, round(max(img.shape[:2]) / 250))
    cs, _ = cv2.findContours(pred.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, cs, -1, (0, 0, 255), t)
    cs, _ = cv2.findContours(gt.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, cs, -1, (0, 255, 0), t)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    test_imgs = list_dir(RF / "test/images")
    picks, by_id, feats, small, nano = select(test_imgs)
    model = YOLO(str(RUNS / "yolo26_small_ft_isic_2018_v11/weights/best.pt"))
    meta = []
    for n, (reason, iid) in enumerate(picks.items(), 1):
        r = model.predict(str(by_id[iid]), device="cpu", **INFER)[0]
        gt = cv2.imread(str(gt_path(iid)), cv2.IMREAD_GRAYSCALE) > 127
        h, w = gt.shape
        k = int(r.boxes.conf.argmax())
        m = r.masks.data[k].cpu().numpy().astype(np.uint8) * 255
        pred = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR) > 127
        dsc, jsi, *_ = dice_jaccard(pred, gt)
        # Overlay on the original-resolution image (raw ISIC input, aspect ratio preserved)
        raw = next(gt_path(iid).parent.parent.glob(f"ISIC2018_Task1-2_Test_Input/{iid}.jpg"))
        img = cv2.imread(str(raw))
        vis = overlay(img, gt, pred)
        s = MAX_SIDE / max(h, w)
        vis = cv2.resize(vis, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        fn = OUT / f"small_pred_{n}.png"
        cv2.imwrite(str(fn), vis)
        meta.append(dict(file=fn.name, id=iid, reason=reason, dsc=round(dsc, 4), jsi=round(jsi, 4),
                         dsc_eval_script=round(small[iid], 4), nano_dsc=round(nano[iid], 4),
                         top1_conf=round(float(r.boxes.conf[k]), 4), **{f: round(v, 4) for f, v in feats[iid].items()}))
        print(meta[-1])
    (OUT / "small_pred_selection.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
