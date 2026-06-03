"""Consolidate Phase 4 (cross-validation) results into a paper-ready CSV+JSON.

For every requested variant, this script reads the per-model
``metrics_summary.json`` produced by :mod:`train_all_models_cv` at::

    <project>/cv/<cv_version>/yolo26_<MODEL>_cv_isic_2018/metrics_summary.json

and emits two consolidated artefacts under ``<project>/pipeline_summary/``:

* ``cv_consolidated.csv`` — one row per model, with ``mean`` and ``std`` of
  mAP@50, mAP@50-95, Precision, Recall and F1-Score (Box and Mask). Convenient
  for direct ingestion into LaTeX ``booktabs`` tables.
* ``cv_consolidated.json`` — structured payload with both per-fold metrics
  and the aggregated summary for every model. Convenient for the analysis
  notebooks under ``utils/notebooks/``.

Usage:
    # Consolidate all five variants (default)::

        python consolidate_cv_results.py

    # Consolidate a subset::

        python consolidate_cv_results.py --models small medium large

    # Point at a non-default CV version / output directory::

        python consolidate_cv_results.py \\
            --cv-version cv_v1 --out-dir /workspace/logs/pipeline_summary
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

#: Canonical order of model sizes used across the pipeline.
DEFAULT_ORDER: list[str] = ["nano", "small", "medium", "large", "xlarge"]

#: Metric keys to report (mean / std). Box and Mask are both kept for
#: completeness; segmentation papers should emphasise Mask.
REPORT_METRICS: list[str] = [
    "map50_b", "map5095_b", "precision_b", "recall_b", "f1_b",
    "map50_m", "map5095_m", "precision_m", "recall_m", "f1_m",
    "best_epoch",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the CV consolidator.

    Returns:
        Parsed ``argparse.Namespace`` with attributes ``models``,
        ``project``, ``cv_version`` and ``out_dir``.
    """
    p = argparse.ArgumentParser(
        description="Consolidate Phase 4 CV results per model into CSV + JSON.",
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
        "--cv-version", default="cv_v1",
        help="CV version subdirectory under <project>/cv/ (default: cv_v1).",
    )
    p.add_argument(
        "--out-dir", default=None,
        help="Output directory (default: <project>/pipeline_summary).",
    )
    return p.parse_args()


def cv_summary_path(project: Path, cv_version: str, model: str) -> Path:
    """Return the canonical ``metrics_summary.json`` path for a CV run.

    Args:
        project: Root logs directory.
        cv_version: CV version subdirectory (e.g. ``"cv_v1"``).
        model: Variant name.

    Returns:
        ``<project>/cv/<cv_version>/yolo26_<model>_cv_isic_2018/metrics_summary.json``.
    """
    return (
        project / "cv" / cv_version
        / f"yolo26_{model}_cv_isic_2018" / "metrics_summary.json"
    )


def load_model_summary(path: Path) -> dict:
    """Load a single ``metrics_summary.json`` produced by Phase 4.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed JSON payload.
    """
    with path.open("r") as f:
        return json.load(f)


def _format_mean_std(agg: dict, key: str) -> str:
    """Format a ``{mean, std}`` entry for stdout printing."""
    v = agg.get(key, {})
    return f"{v.get('mean', 0):.4f}±{v.get('std', 0):.4f}"


def _print_per_model_line(model: str, payload: dict) -> None:
    """Print the per-model summary line."""
    agg = payload.get("summary", {})
    print(
        f"  {model:<8} : k={payload.get('n_folds')} | "
        f"mAP50(M)={_format_mean_std(agg, 'map50_m')}  "
        f"mAP50-95(M)={_format_mean_std(agg, 'map5095_m')}  "
        f"P(M)={_format_mean_std(agg, 'precision_m')}  "
        f"R(M)={_format_mean_std(agg, 'recall_m')}  "
        f"F1(M)={_format_mean_std(agg, 'f1_m')}",
    )


def _write_csv(per_model: list[dict], csv_path: Path) -> None:
    """Write the consolidated CSV (one row per model, mean & std columns)."""
    if not per_model:
        return
    fieldnames = ["model", "n_folds"]
    for k in REPORT_METRICS:
        fieldnames.append(f"{k}_mean")
        fieldnames.append(f"{k}_std")
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for entry in per_model:
            row: dict = {"model": entry["model"], "n_folds": entry["n_folds"]}
            summ = entry["summary"]
            for k in REPORT_METRICS:
                v = summ.get(k, {}) or {}
                row[f"{k}_mean"] = v.get("mean", "")
                row[f"{k}_std"] = v.get("std", "")
            w.writerow(row)


def _write_json(
    per_model: list[dict],
    missing: list[str],
    cv_version: str,
    json_path: Path,
) -> None:
    """Write the consolidated JSON payload."""
    with json_path.open("w") as f:
        json.dump(
            {
                "cv_version": cv_version,
                "models": per_model,
                "missing": missing,
            },
            f, indent=2, sort_keys=True,
        )


def main() -> int:
    """Consolidate CV summaries and write CSV + JSON.

    Returns:
        ``0`` if at least one model summary was found, ``1`` otherwise.
    """
    args = parse_args()
    project = Path(args.project).resolve()
    out_dir = Path(args.out_dir) if args.out_dir else project / "pipeline_summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    per_model: list[dict] = []
    missing: list[str] = []

    for m in args.models:
        path = cv_summary_path(project, args.cv_version, m)
        if not path.exists():
            print(f"  [warn] CV summary not found for {m}: {path}")
            missing.append(m)
            continue
        payload = load_model_summary(path)
        per_model.append({
            "model": m,
            "n_folds": payload.get("n_folds"),
            "summary": payload.get("summary", {}),
            "per_fold": payload.get("per_fold", []),
            "source": str(path),
        })
        _print_per_model_line(m, payload)

    csv_path = out_dir / "cv_consolidated.csv"
    json_path = out_dir / "cv_consolidated.json"
    _write_csv(per_model, csv_path)
    _write_json(per_model, missing, args.cv_version, json_path)

    print("\nConsolidated artefacts:")
    print(f"  CSV : {csv_path}")
    print(f"  JSON: {json_path}")
    if missing:
        print(f"  [warn] models without a summary: {missing}")
    return 0 if per_model else 1


if __name__ == "__main__":
    sys.exit(main())
