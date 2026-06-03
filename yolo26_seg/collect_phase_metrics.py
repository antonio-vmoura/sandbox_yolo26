"""Consolidate single-split Phase 1 / Phase 3 metrics into CSV + JSON.

For a given phase (``baseline`` for Phase 1 or ``optimized`` for Phase 3)
this script walks the per-model ``results.csv`` files, selects the best
epoch using the Ultralytics default criterion (``metrics/mAP50-95(M)``),
derives F1-Score (Box and Mask), and writes a consolidated CSV + JSON in
``<project>/pipeline_summary/<phase>_metrics.{csv,json}``.

Directory layout consumed by this script::

    Phase 1 (baseline):  <project>/phase1_baseline/yolo26_<MODEL>_baseline/results.csv
    Phase 3 (optimized): <project>/yolo26_<MODEL>_ft_isic_2018_v11/results.csv

Usage:
    # Phase 1 (baseline)::

        python collect_phase_metrics.py --phase baseline

    # Phase 3 (optimized)::

        python collect_phase_metrics.py --phase optimized

    # A subset of variants::

        python collect_phase_metrics.py --phase optimized --models small medium
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

#: Canonical order of model sizes used across the pipeline.
DEFAULT_ORDER: list[str] = ["nano", "small", "medium", "large", "xlarge"]

#: Mapping from short metric keys to Ultralytics column names (``results.csv``).
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

#: Column used to pick the best epoch (Ultralytics default for segmentation).
BEST_EPOCH_KEY: str = "metrics/mAP50-95(M)"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the per-phase consolidator.

    Returns:
        Parsed ``argparse.Namespace`` with attributes ``phase``, ``models``,
        ``project`` and ``out_dir``.
    """
    p = argparse.ArgumentParser(
        description=(
            "Collect single-split metrics (Phase 1 or Phase 3) into CSV / JSON."
        ),
    )
    p.add_argument(
        "--phase", choices=["baseline", "optimized"], required=True,
        help="Phase to collect: 'baseline' (Phase 1) or 'optimized' (Phase 3).",
    )
    p.add_argument(
        "--models", nargs="+", default=DEFAULT_ORDER, choices=DEFAULT_ORDER,
        help=f"Subset of models to consolidate (default: {DEFAULT_ORDER}).",
    )
    p.add_argument(
        "--project", default="/workspace/logs",
        help="Root directory for logs (default: /workspace/logs).",
    )
    p.add_argument(
        "--out-dir", default=None,
        help="Output directory (default: <project>/pipeline_summary).",
    )
    return p.parse_args()


def results_csv_path(project: Path, phase: str, model: str) -> Path:
    """Return the canonical ``results.csv`` path for a phase / variant.

    Args:
        project: Root logs directory.
        phase: ``"baseline"`` (Phase 1) or ``"optimized"`` (Phase 3).
        model: Variant name.

    Returns:
        Path to the ``results.csv`` produced by the Ultralytics trainer.
    """
    if phase == "baseline":
        return project / "phase1_baseline" / f"yolo26_{model}_baseline" / "results.csv"
    # ``optimized`` — path emitted by ``train_all_models.py`` (VERSION ``v11``).
    return project / f"yolo26_{model}_ft_isic_2018_v11" / "results.csv"


def parse_best_epoch_metrics(results_csv: Path) -> dict[str, float]:
    """Return the metrics of the best epoch from a ``results.csv``.

    The best epoch is selected with the Ultralytics default criterion for
    segmentation (``metrics/mAP50-95(M)``). F1-Score is derived from
    precision and recall (Box and Mask) using ``2PR/(P+R)``.

    Args:
        results_csv: Path to the Ultralytics-generated CSV.

    Returns:
        Dict containing the eight metrics from :data:`METRIC_KEYS`, plus
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

    best_row = max(
        rows, key=lambda r: float(r.get(BEST_EPOCH_KEY, "0") or 0.0),
    )

    out: dict[str, float] = {}
    for short, full in METRIC_KEYS.items():
        out[short] = float(best_row.get(full, "0") or 0.0)
    for suffix in ("b", "m"):
        p = out[f"precision_{suffix}"]
        r = out[f"recall_{suffix}"]
        out[f"f1_{suffix}"] = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    out["best_epoch"] = float(best_row.get("epoch", "0") or 0.0)
    return out


def _collect_model_row(project: Path, phase: str, model: str) -> dict | None:
    """Collect one row of consolidated metrics for a single variant.

    Returns ``None`` when the ``results.csv`` is missing or unreadable; the
    caller is expected to record the variant in the ``missing`` list.
    """
    csv_path = results_csv_path(project, phase, model)
    if not csv_path.exists():
        print(f"  [warn] results.csv not found for {model}: {csv_path}")
        return None
    try:
        metrics = parse_best_epoch_metrics(csv_path)
    except Exception as e:
        print(f"  [error] failed to read {csv_path}: {e}")
        return None
    print(
        f"  {model:<8} : "
        f"mAP50(M)={metrics['map50_m']:.4f} "
        f"mAP50-95(M)={metrics['map5095_m']:.4f} "
        f"P(M)={metrics['precision_m']:.4f} "
        f"R(M)={metrics['recall_m']:.4f} "
        f"F1(M)={metrics['f1_m']:.4f}",
    )
    return {"model": model, "results_csv": str(csv_path), **metrics}


def _write_artifacts(
    phase: str,
    per_model: list[dict],
    missing: list[str],
    out_dir: Path,
) -> tuple[Path, Path]:
    """Write the consolidated CSV and JSON artefacts."""
    csv_out = out_dir / f"{phase}_metrics.csv"
    json_out = out_dir / f"{phase}_metrics.json"

    if per_model:
        fieldnames = ["model", "results_csv"] + sorted(
            k for k in per_model[0] if k not in ("model", "results_csv")
        )
        with csv_out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in per_model:
                w.writerow(row)

    with json_out.open("w") as f:
        json.dump(
            {"phase": phase, "models": per_model, "missing": missing},
            f, indent=2, sort_keys=True,
        )
    return csv_out, json_out


def main() -> int:
    """Collect per-model metrics for one phase and write CSV + JSON.

    Returns:
        ``0`` if at least one model was successfully read, ``1`` otherwise.
    """
    args = parse_args()
    project = Path(args.project).resolve()
    out_dir = Path(args.out_dir) if args.out_dir else project / "pipeline_summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    per_model: list[dict] = []
    missing: list[str] = []

    for m in args.models:
        row = _collect_model_row(project, args.phase, m)
        if row is None:
            missing.append(m)
            continue
        per_model.append(row)

    csv_out, json_out = _write_artifacts(args.phase, per_model, missing, out_dir)

    print("\nGenerated artefacts:")
    print(f"  CSV : {csv_out}")
    print(f"  JSON: {json_out}")
    if missing:
        print(f"  [warn] no results for: {missing}")
    return 0 if per_model else 1


if __name__ == "__main__":
    sys.exit(main())
