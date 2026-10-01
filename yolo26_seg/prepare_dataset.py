"""Phase 0 — Build the YOLO-seg dataset from the RAW official ISIC 2018 release (Task 1 or Task 2).

``--task 1`` (default): lesion boundary segmentation (one binary mask per image) — described below.
``--task 2``: lesion attribute detection — the five dermoscopic attributes of ISIC 2018 Task 2
(:data:`ATTRIBUTES`), one binary mask per attribute (``ISIC_<id>_attribute_<name>.png``). Attributes may
overlap and are often absent, so they are treated as a **multi-label** problem: every attribute is converted
independently (polygons with class id 0-4 = its index in :data:`ATTRIBUTES`; overlapping polygons of
different classes coexist in the label file; an image may have no polygon), its mask is stored as
``masks/<attribute>/<id>.png`` and its label fidelity is checked against that mask alone. Images are the same
JPEGs as Task 1 (shared ``Task1-2`` input folders) and get the same working resolution.

The dataset of the study is derived **only** from the official ISIC 2018 Task 1 files (lesion images as JPEG,
ground truth as binary PNG masks), which this script reads directly:

    <raw>/ISIC2018_Task1-2_Training_Input/ISIC_<id>.jpg      + ISIC2018_Task1_Training_GroundTruth/ISIC_<id>_segmentation.png
    <raw>/ISIC2018_Task1-2_Validation_Input/ISIC_<id>.jpg    + ISIC2018_Task1_Validation_GroundTruth/...
    <raw>/ISIC2018_Task1-2_Test_Input/ISIC_<id>.jpg          + ISIC2018_Task1_Test_GroundTruth/...

The official split is kept as is and enforced with assertions: **exactly 2,594 training, 100 validation and 1,000
test images** (3,694), every image paired with its mask, no identifier in two splits. (A Roboflow export used
earlier silently dropped 47 training and 6 test images.)

Processing of each image (deterministic, parallel):

1. **Working resolution.** Images whose longer side exceeds ``--max-side`` (default 1,024 px) are downscaled with
   area interpolation, preserving the aspect ratio; smaller images are not upscaled. None of the models sees more
   than ~1,000 px of input (U-Net 256, YOLO26 640, SAM 3 1,008), so this is the finest resolution at which any
   model predicts; it is also the resolution at which every model is evaluated. The image is decoded **without**
   EXIF re-orientation (as the official masks) and written losslessly as PNG.
2. **Mask.** The official mask is binarised (> 127), area-resized to the working resolution and re-binarised at
   0.5 (boundary at the sub-pixel majority), and written as a 0/255 PNG (``masks/``).
3. **YOLO training label.** The mask's contours are extracted with ``cv2.findContours`` (``RETR_CCOMP``,
   ``CHAIN_APPROX_SIMPLE`` — lossless removal of collinear points). YOLO labels cannot express holes; every hole
   is spliced into its outer contour through a zero-width bridge between the closest pair of points, so that the
   polygon still excludes it. Some official masks are speckled (up to ~640 tiny holes and ~50 fragments at the
   working resolution); since Ultralytics resamples every training polygon to 1,000 points, fragments and holes
   smaller than ``--min-part`` (0.1 %) of the lesion area are left out of the *label*. Coordinates are normalised
   and written with 7 decimals.
4. **Ground truth for evaluation = the official mask.** Every pipeline scores predictions against
   ``masks/<id>.png`` (:func:`segmentation_metrics.ground_truth_mask`), never against the simplified label. The
   label is rasterised back and compared with the mask (per-image Dice, asserted >= ``--min-fidelity``, default
   0.98, and recorded); the resampling error with respect to the full-resolution original mask is recorded too.

Output (``--out``)::

    data.yaml                                    # path = /workspace/datasets/<name>, train/val/test, nc=1
    {train,valid,test}/{images/*.png, labels/*.txt, masks/*.png}
    manifest_{train,val,test}.csv                # id, original and working size, polygons, holes, fidelity
    meta.json                                    # counts, parameters, source fingerprints, fidelity summary

The build is idempotent (skipped when the sources and parameters are unchanged; ``--force`` rebuilds) and
atomic (built in a temporary folder, then renamed; a replaced dataset is kept as ``<out>.bak-<UTC>``).

Usage (inside the ``yolo26_ft`` container; the raw release ``ISIC2018_Raw`` mounted at ``/workspace/raw``):
    python yolo26_seg/prepare_dataset.py                    # Task 1 -> /workspace/datasets/isic2018_task1_official
    python yolo26_seg/prepare_dataset.py --task 2           # Task 2 -> /workspace/datasets/isic2018_task2_official
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from datetime import datetime, timezone

from common import atomic_write_json, read_json, utc_now_iso
from segmentation_metrics import ISIC2018_ATTRIBUTES, pixel_scores, rasterize_yolo_label

#: Version of the preprocessing (part of the idempotence key).
PREP_VERSION: int = 1

#: Official ISIC 2018 Task 1 split sizes — enforced with assertions.
EXPECTED_COUNTS: dict[str, int] = {"train": 2594, "val": 100, "test": 1000}

#: split -> (image folder, ground-truth folder) of the official release.
RAW_FOLDERS: dict[str, tuple[str, str]] = {
    "train": ("ISIC2018_Task1-2_Training_Input", "ISIC2018_Task1_Training_GroundTruth"),
    "val": ("ISIC2018_Task1-2_Validation_Input", "ISIC2018_Task1_Validation_GroundTruth"),
    "test": ("ISIC2018_Task1-2_Test_Input", "ISIC2018_Task1_Test_GroundTruth"),
}

#: split -> YOLO folder name (Ultralytics convention: "valid").
YOLO_DIRS: dict[str, str] = {"train": "train", "val": "valid", "test": "test"}

#: ISIC 2018 Task 2 attributes; class id = index (shared definition in segmentation_metrics).
ATTRIBUTES: tuple[str, ...] = ISIC2018_ATTRIBUTES

#: split -> Task 2 ground-truth folder of the official release (training masks: version 3).
TASK2_GT_FOLDERS: dict[str, str] = {
    "train": "ISIC2018_Task2_Training_GroundTruth_v3",
    "val": "ISIC2018_Task2_Validation_GroundTruth",
    "test": "ISIC2018_Task2_Test_GroundTruth",
}

DEFAULT_RAW: str = "/workspace/raw"


def default_out(task: int) -> str:
    """Default output folder of a task (Task 1 keeps its historical name)."""
    return f"/workspace/datasets/isic2018_task{task}_official"


def utc_stamp() -> str:
    """Compact UTC timestamp for temporary / backup folder names."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# ----------------------------------------------------------------------------
# Mask -> polygon
# ----------------------------------------------------------------------------
def _bridge(outer: np.ndarray, hole: np.ndarray) -> np.ndarray:
    """Splice ``hole`` into ``outer`` through a zero-width bridge between their closest points."""
    d = ((outer[:, None, :].astype(np.int64) - hole[None, :, :]) ** 2).sum(-1)
    i, j = np.unravel_index(int(np.argmin(d)), d.shape)
    return np.concatenate([outer[: i + 1], np.roll(hole, -j, axis=0), hole[j : j + 1], outer[i:]])


def mask_to_polygons(mask: np.ndarray, min_part: float = 0.0) -> tuple[list[np.ndarray], int]:
    """Polygons (pixel coordinates) of a binary mask, holes bridged into their outer contour.

    Components and holes whose area is below ``min_part`` × the mask area are left out.

    Returns:
        ``(polygons, n_holes_kept)``; one polygon per (kept) connected component.
    """
    min_area = min_part * float(np.count_nonzero(mask))
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return [], 0
    hierarchy = hierarchy[0]
    polygons, n_holes = [], 0
    for k, c in enumerate(contours):
        if hierarchy[k][3] != -1:      # a hole: handled with its parent
            continue
        if cv2.contourArea(c) < min_area:
            continue
        poly = c[:, 0, :]
        child = hierarchy[k][2]
        while child != -1:
            if cv2.contourArea(contours[child]) >= min_area:
                poly = _bridge(poly, contours[child][:, 0, :])
                n_holes += 1
            child = hierarchy[child][0]
        if len(poly) >= 3:
            polygons.append(poly)
    return polygons, n_holes


def polygon_lines(polygons: list[np.ndarray], width: int, height: int, class_id: int = 0) -> list[str]:
    """YOLO segmentation label lines (normalised coordinates, 7 decimals) of one class."""
    lines = []
    for p in polygons:
        xy = p.astype(np.float64) / np.array([width, height])
        lines.append(f"{class_id} " + " ".join(f"{v:.7f}" for v in xy.reshape(-1)))
    return lines


def polygons_to_label(polygons: list[np.ndarray], width: int, height: int, class_id: int = 0) -> str:
    """YOLO segmentation label of one class (class 0 for Task 1)."""
    return "\n".join(polygon_lines(polygons, width, height, class_id)) + "\n"


# ----------------------------------------------------------------------------
# One image
# ----------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while block := f.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def process(job: dict[str, Any]) -> dict[str, Any]:
    """Convert one image/mask pair; return its manifest row."""
    image_path, mask_path, out, max_side = Path(job["image"]), Path(job["mask"]), Path(job["out"]), job["max_side"]
    iid = image_path.stem
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    gt = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    assert image is not None and gt is not None, f"{iid}: unreadable image or mask"
    h0, w0 = image.shape[:2]
    assert gt.shape == (h0, w0), f"{iid}: mask {gt.shape} != image {(h0, w0)}"
    gt = gt > 127
    assert gt.any(), f"{iid}: empty ground-truth mask"

    scale = min(1.0, max_side / max(h0, w0))
    w, h = max(1, round(w0 * scale)), max(1, round(h0 * scale))
    if scale < 1.0:
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(gt.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA) >= 0.5
    else:
        mask = gt
    assert mask.any(), f"{iid}: mask vanished at the working resolution"

    polygons, n_holes = mask_to_polygons(mask, job["min_part"])
    _, n_holes_all = mask_to_polygons(mask)
    label = polygons_to_label(polygons, w, h)
    cv2.imwrite(str(out / "images" / f"{iid}.png"), image)
    cv2.imwrite(str(out / "masks" / f"{iid}.png"), mask.astype(np.uint8) * 255)
    (out / "labels" / f"{iid}.txt").write_text(label)

    # Fidelity: the label as every evaluation rasterises it vs the working-resolution mask.
    fidelity = pixel_scores(mask, rasterize_yolo_label(out / "labels" / f"{iid}.txt", h, w))["dsc"]
    # Resampling error: the working-resolution mask brought back to the original size vs the original mask.
    if scale < 1.0:
        back = cv2.resize(mask.astype(np.float32), (w0, h0), interpolation=cv2.INTER_LINEAR) >= 0.5
        resample = pixel_scores(gt, back)["dsc"]
    else:
        resample = 1.0
    return {"id": iid, "orig_w": w0, "orig_h": h0, "width": w, "height": h, "scale": round(scale, 6),
            "n_polygons": len(polygons), "n_points": int(sum(len(p) for p in polygons)), "n_holes": n_holes,
            "n_holes_mask": n_holes_all,
            "fidelity_dsc": fidelity, "resample_dsc": resample,
            "image_sha256": sha256_file(image_path), "mask_sha256": sha256_file(mask_path)}


def process_task2(job: dict[str, Any]) -> dict[str, Any]:
    """Convert one image and its five attribute masks (Task 2, multi-label); return its manifest row."""
    image_path, out, max_side = Path(job["image"]), Path(job["out"]), job["max_side"]
    iid = image_path.stem
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    assert image is not None, f"{iid}: unreadable image"
    h0, w0 = image.shape[:2]
    scale = min(1.0, max_side / max(h0, w0))
    w, h = max(1, round(w0 * scale)), max(1, round(h0 * scale))
    if scale < 1.0:
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out / "images" / f"{iid}.png"), image)

    row: dict[str, Any] = {"id": iid, "orig_w": w0, "orig_h": h0, "width": w, "height": h, "scale": round(scale, 6)}
    lines: list[str] = []
    masks_sha = hashlib.sha256()
    for k, name in enumerate(ATTRIBUTES):
        gt = cv2.imread(job["masks"][name], cv2.IMREAD_GRAYSCALE)
        assert gt is not None and gt.shape == (h0, w0), f"{iid}/{name}: unreadable mask or size != image"
        masks_sha.update(sha256_file(Path(job["masks"][name])).encode())
        gt = gt > 127
        mask = (cv2.resize(gt.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA) >= 0.5) if scale < 1.0 else gt
        cv2.imwrite(str(out / "masks" / name / f"{iid}.png"), mask.astype(np.uint8) * 255)
        polygons, _ = mask_to_polygons(mask, job["min_part"])
        lines += polygon_lines(polygons, w, h, class_id=k)
        row[f"{name}_present"] = bool(gt.any())
        row[f"{name}_px"] = int(np.count_nonzero(mask))
        row[f"{name}_vanished"] = bool(gt.any() and not mask.any())    # too small for the working resolution
        row[f"{name}_n_polygons"] = len(polygons)
        row[f"{name}_fidelity_dsc"] = math.nan
    label_path = out / "labels" / f"{iid}.txt"
    label_path.write_text("\n".join(lines) + "\n" if lines else "")
    for k, name in enumerate(ATTRIBUTES):   # each attribute's polygons vs that attribute's mask only
        if row[f"{name}_px"]:
            mask = cv2.imread(str(out / "masks" / name / f"{iid}.png"), cv2.IMREAD_GRAYSCALE) > 127
            row[f"{name}_fidelity_dsc"] = pixel_scores(mask, rasterize_yolo_label(label_path, h, w, class_id=k))["dsc"]
    row["n_attributes"] = sum(bool(row[f"{n}_px"]) for n in ATTRIBUTES)
    row["image_sha256"] = sha256_file(image_path)
    row["masks_sha256"] = masks_sha.hexdigest()
    return row


# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------
def collect_pairs(raw: Path) -> dict[str, list[tuple[Path, Path]]]:
    """Image/mask pairs of every split, validated against the official counts."""
    pairs: dict[str, list[tuple[Path, Path]]] = {}
    seen: dict[str, str] = {}
    for split, (img_dir, gt_dir) in RAW_FOLDERS.items():
        images = sorted((raw / img_dir).glob("ISIC_*.jpg"))
        masks = {p.name.replace("_segmentation.png", ""): p for p in (raw / gt_dir).glob("ISIC_*_segmentation.png")}
        assert len(images) == EXPECTED_COUNTS[split], (
            f"{split}: {len(images)} images in {raw / img_dir}, expected exactly {EXPECTED_COUNTS[split]}")
        assert len(masks) == EXPECTED_COUNTS[split], (
            f"{split}: {len(masks)} masks in {raw / gt_dir}, expected exactly {EXPECTED_COUNTS[split]}")
        missing = [p.stem for p in images if p.stem not in masks]
        assert not missing, f"{split}: {len(missing)} images without a mask, e.g. {missing[:5]}"
        for p in images:
            assert p.stem not in seen, f"{p.stem} appears in both {seen[p.stem]!r} and {split!r}"
            seen[p.stem] = split
        pairs[split] = [(p, masks[p.stem]) for p in images]
    assert sum(len(v) for v in pairs.values()) == sum(EXPECTED_COUNTS.values()) == 3694
    return pairs


def collect_task2(raw: Path) -> dict[str, list[tuple[Path, dict[str, Path]]]]:
    """Image + five attribute masks of every split, validated against the official counts."""
    items: dict[str, list[tuple[Path, dict[str, Path]]]] = {}
    seen: dict[str, str] = {}
    for split, (img_dir, _) in RAW_FOLDERS.items():
        gt_dir = raw / TASK2_GT_FOLDERS[split]
        images = sorted((raw / img_dir).glob("ISIC_*.jpg"))
        assert len(images) == EXPECTED_COUNTS[split], (
            f"{split}: {len(images)} images in {raw / img_dir}, expected exactly {EXPECTED_COUNTS[split]}")
        for name in ATTRIBUTES:
            n = len(list(gt_dir.glob(f"ISIC_*_attribute_{name}.png")))
            assert n == EXPECTED_COUNTS[split], (
                f"{split}: {n} '{name}' masks in {gt_dir}, expected exactly {EXPECTED_COUNTS[split]}")
        rows = []
        for p in images:
            masks = {name: gt_dir / f"{p.stem}_attribute_{name}.png" for name in ATTRIBUTES}
            missing = [n for n, m in masks.items() if not m.exists()]
            assert not missing, f"{split}/{p.stem}: missing attribute masks {missing}"
            assert p.stem not in seen, f"{p.stem} appears in both {seen[p.stem]!r} and {split!r}"
            seen[p.stem] = split
            rows.append((p, masks))
        items[split] = rows
    assert sum(len(v) for v in items.values()) == sum(EXPECTED_COUNTS.values()) == 3694
    return items


def source_fingerprint(pairs: dict[str, list[tuple[Path, Path]]]) -> dict[str, str]:
    """Cheap fingerprint (names, sizes) of the sources — the full SHA-256 is recorded per image."""
    out = {}
    for split, items in pairs.items():
        h = hashlib.sha256()
        for img, msk in items:
            masks = sorted(msk.values()) if isinstance(msk, dict) else [msk]
            h.update(f"{img.name}:{img.stat().st_size}".encode())
            for m in masks:
                h.update(f":{m.name}:{m.stat().st_size}".encode())
            h.update(b"\n")
        out[split] = h.hexdigest()
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 0 — YOLO-seg dataset from the raw ISIC 2018 release (Task 1 or 2).")
    p.add_argument("--task", type=int, choices=(1, 2), default=1,
                   help="ISIC 2018 task: 1 = lesion segmentation (default), 2 = five lesion attributes (multi-label).")
    p.add_argument("--raw", default=DEFAULT_RAW, help=f"Folder of the raw official release (default: {DEFAULT_RAW}).")
    p.add_argument("--out", default=None, help="Output dataset folder (default: /workspace/datasets/isic2018_task<N>_official).")
    p.add_argument("--max-side", type=int, default=1024,
                   help="Longer side of the working resolution (default 1024; smaller images are not upscaled).")
    p.add_argument("--min-part", type=float, default=0.001,
                   help="Fragments/holes smaller than this fraction of the lesion area are left out of the YOLO "
                        "label (default 0.001); the evaluation mask is always exact.")
    p.add_argument("--min-fidelity", type=float, default=None,
                   help="Minimum Dice between a rasterised label and its mask (default 0.98 for Task 1; 0.95 per "
                        "attribute for Task 2, whose masks are small and fragmented).")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--force", action="store_true", help="Rebuild even if up to date.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.min_fidelity is None:
        args.min_fidelity = 0.98 if args.task == 1 else 0.95
    if args.out is None:
        args.out = default_out(args.task)
    if args.task == 2:
        return main_task2(args)
    raw, out = Path(args.raw), Path(args.out)
    pairs = collect_pairs(raw)
    params = {"prep_version": PREP_VERSION, "max_side": args.max_side, "min_fidelity": args.min_fidelity,
              "min_part": args.min_part,
              "image_interp": "INTER_AREA", "mask": "INTER_AREA + threshold 0.5", "image_format": "png",
              "polygon": "findContours RETR_CCOMP CHAIN_APPROX_SIMPLE, holes bridged"}
    sources = source_fingerprint(pairs)
    meta = read_json(out / "meta.json")
    if meta and meta.get("sources") == sources and meta.get("params") == params and not args.force:
        print(f"[skip] dataset up to date: {out}")
        return 0

    print(f"Phase 0 — ISIC 2018 Task 1 (official) -> YOLO-seg\n  raw = {raw}\n  out = {out}")
    print("  counts: " + ", ".join(f"{s} {len(v)}" for s, v in pairs.items()) + " (asserted)")
    tmp = out.with_name(f"{out.name}.tmp-{utc_stamp()}")
    rows: dict[str, list[dict[str, Any]]] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for split, items in pairs.items():
            d = tmp / YOLO_DIRS[split]
            for sub in ("images", "labels", "masks"):
                (d / sub).mkdir(parents=True, exist_ok=True)
            jobs = [{"image": str(i), "mask": str(m), "out": str(d), "max_side": args.max_side,
                     "min_part": args.min_part} for i, m in items]
            rows[split] = list(pool.map(process, jobs, chunksize=8))
            fid = np.array([r["fidelity_dsc"] for r in rows[split]])
            print(f"  {split:<5}: {len(rows[split])} images | label fidelity DSC min {fid.min():.5f} "
                  f"mean {fid.mean():.6f} | label points max {max(r['n_points'] for r in rows[split])}", flush=True)

    # ---- strict checks ------------------------------------------------------------------------------
    for split, r in rows.items():
        assert len(r) == EXPECTED_COUNTS[split], f"{split}: wrote {len(r)} images, expected {EXPECTED_COUNTS[split]}"
        for sub, ext in (("images", "png"), ("labels", "txt"), ("masks", "png")):
            n = len(list((tmp / YOLO_DIRS[split] / sub).glob(f"*.{ext}")))
            assert n == EXPECTED_COUNTS[split], f"{split}/{sub}: {n} files, expected {EXPECTED_COUNTS[split]}"
        bad = [x["id"] for x in r if x["fidelity_dsc"] < args.min_fidelity]
        assert not bad, f"{split}: {len(bad)} labels below fidelity {args.min_fidelity}: {bad[:5]}"
    assert sum(len(r) for r in rows.values()) == 3694

    for split, r in rows.items():
        with (tmp / f"manifest_{split}.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(r[0].keys()))
            w.writeheader()
            w.writerows(r)
    (tmp / "data.yaml").write_text(yaml.safe_dump({
        "path": f"/workspace/datasets/{out.name}", "train": "train/images", "val": "valid/images",
        "test": "test/images", "nc": 1, "names": ["lesion"],
    }, sort_keys=False))
    summary = {s: {"n_images": len(r),
                   "fidelity_dsc_min": float(min(x["fidelity_dsc"] for x in r)),
                   "fidelity_dsc_mean": float(np.mean([x["fidelity_dsc"] for x in r])),
                   "resample_dsc_min": float(min(x["resample_dsc"] for x in r)),
                   "resample_dsc_mean": float(np.mean([x["resample_dsc"] for x in r])),
                   "n_downscaled": int(sum(x["scale"] < 1 for x in r)),
                   "n_masks_with_holes": int(sum(x["n_holes_mask"] > 0 for x in r)),
                   "n_labels_with_bridged_holes": int(sum(x["n_holes"] > 0 for x in r)),
                   "n_points_max": int(max(x["n_points"] for x in r)),
                   "n_multi_polygon": int(sum(x["n_polygons"] > 1 for x in r))} for s, r in rows.items()}
    atomic_write_json(tmp / "meta.json", {
        "created_at": utc_now_iso(), "source": str(raw), "release": "ISIC 2018 Task 1 (official)",
        "counts": {s: len(r) for s, r in rows.items()}, "params": params, "sources": sources, "splits": summary,
    })
    if out.exists():
        backup = out.with_name(f"{out.name}.bak-{utc_stamp()}")
        out.rename(backup)
        print(f"  previous dataset kept as {backup}")
    tmp.rename(out)
    for s, v in summary.items():
        print(f"  {s:<5}: {v}")
    print(f"Done: {out} (3,694 images = 2,594 + 100 + 1,000)")
    return 0


def main_task2(args: argparse.Namespace) -> int:
    """Task 2: five attribute masks per image -> multi-label YOLO-seg dataset (class id = attribute index)."""
    raw, out = Path(args.raw), Path(args.out)
    items = collect_task2(raw)
    params = {"prep_version": PREP_VERSION, "task": 2, "attributes": list(ATTRIBUTES), "max_side": args.max_side,
              "min_fidelity": args.min_fidelity, "min_part": args.min_part, "image_interp": "INTER_AREA",
              "mask": "INTER_AREA + threshold 0.5, per attribute", "image_format": "png",
              "polygon": "findContours RETR_CCOMP CHAIN_APPROX_SIMPLE, holes bridged, class id = attribute index"}
    sources = source_fingerprint(items)
    meta = read_json(out / "meta.json")
    if meta and meta.get("sources") == sources and meta.get("params") == params and not args.force:
        print(f"[skip] dataset up to date: {out}")
        return 0

    print(f"Phase 0 — ISIC 2018 Task 2 (official, 5 attributes, multi-label) -> YOLO-seg\n  raw = {raw}\n  out = {out}")
    print("  counts: " + ", ".join(f"{s} {len(v)}" for s, v in items.items()) + " images x 5 attribute masks (asserted)")
    tmp = out.with_name(f"{out.name}.tmp-{utc_stamp()}")
    rows: dict[str, list[dict[str, Any]]] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for split, split_items in items.items():
            d = tmp / YOLO_DIRS[split]
            for sub in ("images", "labels", *(f"masks/{n}" for n in ATTRIBUTES)):
                (d / sub).mkdir(parents=True, exist_ok=True)
            jobs = [{"image": str(i), "masks": {n: str(m) for n, m in masks.items()}, "out": str(d),
                     "max_side": args.max_side, "min_part": args.min_part} for i, masks in split_items]
            rows[split] = list(pool.map(process_task2, jobs, chunksize=8))
            prev = " ".join(f"{n} {sum(r[f'{n}_px'] > 0 for r in rows[split])}" for n in ATTRIBUTES)
            print(f"  {split:<5}: {len(rows[split])} images | images per attribute: {prev}", flush=True)

    # ---- strict checks ------------------------------------------------------------------------------
    summary: dict[str, Any] = {}
    for split, r in rows.items():
        assert len(r) == EXPECTED_COUNTS[split], f"{split}: wrote {len(r)} images, expected {EXPECTED_COUNTS[split]}"
        for sub, ext in (("images", "png"), ("labels", "txt"), *((f"masks/{n}", "png") for n in ATTRIBUTES)):
            n = len(list((tmp / YOLO_DIRS[split] / sub).glob(f"*.{ext}")))
            assert n == EXPECTED_COUNTS[split], f"{split}/{sub}: {n} files, expected {EXPECTED_COUNTS[split]}"
        summary[split] = {"n_images": len(r), "n_images_without_attribute": int(sum(x["n_attributes"] == 0 for x in r)),
                          "n_images_with_overlapping_attributes": None}
        for name in ATTRIBUTES:
            fid = [x[f"{name}_fidelity_dsc"] for x in r if x[f"{name}_px"]]
            bad = [x["id"] for x in r if x[f"{name}_px"] and x[f"{name}_fidelity_dsc"] < args.min_fidelity]
            assert not bad, f"{split}/{name}: {len(bad)} labels below fidelity {args.min_fidelity}: {bad[:5]}"
            summary[split][name] = {
                "n_present_raw": int(sum(x[f"{name}_present"] for x in r)),
                "n_present": len(fid),
                "n_vanished_at_working_resolution": int(sum(x[f"{name}_vanished"] for x in r)),
                "fidelity_dsc_min": float(min(fid)) if fid else None,
                "fidelity_dsc_mean": float(np.mean(fid)) if fid else None,
            }
    assert sum(len(r) for r in rows.values()) == 3694
    # Overlap between attributes (multi-label): pixels claimed by >= 2 attributes.
    for split in rows:
        d = tmp / YOLO_DIRS[split]
        n_overlap = 0
        for x in rows[split]:
            if x["n_attributes"] >= 2:
                stack = sum((cv2.imread(str(d / "masks" / n / f"{x['id']}.png"), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
                            for n in ATTRIBUTES if x[f"{n}_px"])
                n_overlap += int((stack >= 2).any())
        summary[split]["n_images_with_overlapping_attributes"] = n_overlap

    for split, r in rows.items():
        with (tmp / f"manifest_{split}.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(r[0].keys()))
            w.writeheader()
            w.writerows(r)
    (tmp / "data.yaml").write_text(yaml.safe_dump({
        "path": f"/workspace/datasets/{out.name}", "train": "train/images", "val": "valid/images",
        "test": "test/images", "nc": len(ATTRIBUTES), "names": list(ATTRIBUTES),
    }, sort_keys=False))
    atomic_write_json(tmp / "meta.json", {
        "created_at": utc_now_iso(), "source": str(raw), "release": "ISIC 2018 Task 2 (official)", "task": 2,
        "attributes": list(ATTRIBUTES), "counts": {s: len(r) for s, r in rows.items()}, "params": params,
        "sources": sources, "splits": summary,
    })
    if out.exists():
        backup = out.with_name(f"{out.name}.bak-{utc_stamp()}")
        out.rename(backup)
        print(f"  previous dataset kept as {backup}")
    tmp.rename(out)
    for s_, v in summary.items():
        print(f"  {s_:<5}: " + json.dumps(v))
    print(f"Done: {out} (3,694 images x 5 attributes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
