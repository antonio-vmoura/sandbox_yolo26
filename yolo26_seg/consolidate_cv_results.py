"""Consolidate Phase 2 (cross-validation) results into a paper-ready CSV + JSON.

For every requested variant, this script reads the ``metrics_summary.json``
produced by :mod:`train_all_models_cv` at::

    <project>/phase2_cv_<protocol>/yolo26_<MODEL>/metrics_summary.json

and writes under ``<project>/summary/``:

* ``phase2_cv_<protocol>.csv`` — one row per model with ``mean`` and ``std``
  (sample std, ddof=1) of mAP@50, mAP@50-95, Precision, Recall and F1 (Box
  and Mask). Ready for LaTeX ``booktabs`` tables.
* ``phase2_cv_<protocol>.json`` — per-fold metrics and the aggregate for every
  model; consumed by the analysis notebooks.

Usage:
    python consolidate_cv_results.py --project /workspace/logs/pipeline_final_v1
    python consolidate_cv_results.py --protocol optimized --models small
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from common import DEFAULT_PIPELINE_ROOT, PipelinePaths

#: Canonical order of model sizes used across the pipeline.
DEFAULT_ORDER: list[str] = ["nano", "small", "medium", "large", "xlarge"]

#: Metric keys to report (mean / std). Box and Mask are both kept for
#: completeness; segmentation papers should emphasise Mask.
REPORT_METRICS: list[str] = [
    "map50_b", "map5095_b", "precision_b", "recall_b", "f1_b",
    "map50_m", "map5095_m", "precision_m", "recall_m", "f1_m",
    "best_epoch", "epochs_trained",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the CV consolidator.

    Returns:
        Parsed ``argparse.Namespace`` with attributes ``models``,
        ``project`` and ``protocol``.
    """
    p = argparse.ArgumentParser(
        description="Consolidate Phase 2 CV results per model into CSV + JSON.",
    )
    p.add_argument(
        "--models", nargs="+", default=DEFAULT_ORDER, choices=DEFAULT_ORDER,
        help=f"Subset of models to consolidate (default: {DEFAULT_ORDER}).",
    )
    p.add_argument(
        "--project", default=DEFAULT_PIPELINE_ROOT,
        help=f"Pipeline root (default: {DEFAULT_PIPELINE_ROOT}).",
    )
    p.add_argument(
        "--protocol", choices=["baseline", "optimized"], default="baseline",
        help="CV protocol to consolidate (default: baseline = Phase 2).",
    )
    return p.parse_args()


def load_model_summary(path: Path) -> dict:
    """Load a single ``metrics_summary.json`` produced by Phase 2.

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
    protocol: str,
    json_path: Path,
) -> None:
    """Write the consolidated JSON payload."""
    with json_path.open("w") as f:
        json.dump(
            {
                "protocol": protocol,
                "models": per_model,
                "missing": missing,
            },
            f, indent=2, sort_keys=True,
        )


def main() -> int:
    """Consolidate CV summaries and write CSV + JSON.

    Returns:
        ``0`` if every requested model has a summary, ``1`` otherwise.
    """
    args = parse_args()
    paths = PipelinePaths(Path(args.project).resolve())
    out_dir = paths.summary_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    per_model: list[dict] = []
    missing: list[str] = []

    for m in args.models:
        path = paths.cv_model_dir(m, args.protocol) / "metrics_summary.json"
        if not path.exists():
            print(f"  [warn] CV summary not found for {m}: {path}")
            missing.append(m)
            continue
        payload = load_model_summary(path)
        if payload.get("std_ddof") != 1:
            print(f"  [warn] {m}: summary predates the sample-std fix — re-run Phase 2 to refresh it")
        per_model.append({
            "model": m,
            "n_folds": payload.get("n_folds"),
            "summary": payload.get("summary", {}),
            "per_fold": payload.get("per_fold", []),
            "source": str(path),
        })
        _print_per_model_line(m, payload)

    csv_path = out_dir / f"phase2_cv_{args.protocol}.csv"
    json_path = out_dir / f"phase2_cv_{args.protocol}.json"
    _write_csv(per_model, csv_path)
    _write_json(per_model, missing, args.protocol, json_path)

    print("\nConsolidated artefacts:")
    print(f"  CSV : {csv_path}")
    print(f"  JSON: {json_path}")
    if missing:
        print(f"  [warn] models without a summary: {missing}")
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
