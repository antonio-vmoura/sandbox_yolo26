#!/usr/bin/env python3
"""Validate the outputs of Phase 2 (HPO) after ``tune_all_models_v2.py`` runs.

Motivation
----------
The Ultralytics ``Tuner`` **does not propagate per-trial failures**: when a
trial dies (for example after the NVIDIA driver crashes on the host) the
Tuner simply records ``fitness=0`` for that trial and moves on. At the end
of the run it writes a ``best_hyperparameters.yaml`` that is just the initial
seed vector and returns exit-code 0 — making the entire Phase 2 *look*
successful when in fact **no trial actually finished**.

This script walks each model's ``tune_results.csv`` and treats the HPO as
degenerate if:

* the CSV is missing,
* it has fewer than ``--min-rows`` rows, **or**
* it has fewer than ``--min-trials`` rows with ``fitness > 0``.

When any model is degenerate the script returns exit-code 1, listing what
needs to be re-tuned.

Usage:
    Typical invocation (called by ``run_pipeline.sh`` right after Phase 2)::

        python yolo26_seg/check_hpo_validity.py \\
            --project /workspace/logs/pipeline_e2e_v2 \\
            --models nano small medium large xlarge \\
            --min-trials 1
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the HPO validator.

    Returns:
        Parsed ``argparse.Namespace``.
    """
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--project", required=True,
        help="Pipeline root directory (e.g. /workspace/logs/pipeline_e2e_v2).",
    )
    p.add_argument(
        "--models", nargs="+", required=True,
        help="Models tuned in this Phase 2 (e.g. nano small medium large xlarge).",
    )
    p.add_argument(
        "--hpo-dir", default="hpo/hpo_v3",
        help="HPO subdirectory relative to --project (default: hpo/hpo_v3).",
    )
    p.add_argument(
        "--tune-prefix", default="tune_isic_2018_task_1_",
        help=(
            "Prefix of every tune directory name (default: "
            "tune_isic_2018_task_1_<model>)."
        ),
    )
    p.add_argument(
        "--min-trials", type=int, default=1,
        help=(
            "Minimum number of trials with fitness>0 required per model "
            "(default: 1). Raise to (e.g.) 5 to guarantee a minimally-"
            "explored search before accepting the HPO as valid."
        ),
    )
    p.add_argument(
        "--min-rows", type=int, default=1,
        help=(
            "Minimum total number of rows expected in tune_results.csv "
            "(default: 1). Header is not counted. Used to detect an "
            "empty CSV."
        ),
    )
    return p.parse_args()


def _find_fitness_column(fieldnames: list[str]) -> str | None:
    """Locate the ``fitness`` column in the CSV header, case-insensitively.

    Args:
        fieldnames: Column names returned by ``csv.DictReader``.

    Returns:
        The exact column name to use as a dict key, or ``None`` if absent.
    """
    for col in fieldnames:
        if col is not None and col.strip().lower() == "fitness":
            return col
    return None


def _count_valid_trials(
    rows: list[dict],
    fitness_key: str,
) -> tuple[int, float]:
    """Return ``(good_trials, best_fitness)`` from a parsed tune CSV.

    Args:
        rows: List of CSV row dicts.
        fitness_key: Exact key name for the fitness column.

    Returns:
        Tuple of the number of trials with ``fitness > 0`` and the maximum
        observed fitness (defaults to 0.0 when no valid trial exists).
    """
    good = 0
    best = 0.0
    for r in rows:
        raw = r.get(fitness_key, "")
        try:
            v = float(raw)
        except (TypeError, ValueError):
            continue
        if v > 0:
            good += 1
            if v > best:
                best = v
    return good, best


def validate_model(
    model: str,
    project: Path,
    hpo_dir: str,
    tune_prefix: str,
    min_trials: int,
    min_rows: int,
) -> tuple[bool, str]:
    """Validate the HPO of a single model.

    Args:
        model: Variant name.
        project: Pipeline root directory.
        hpo_dir: Sub-directory of HPO outputs relative to ``project``.
        tune_prefix: Prefix of the per-model tune sub-directory.
        min_trials: Minimum number of trials with ``fitness > 0`` required.
        min_rows: Minimum number of rows expected in ``tune_results.csv``.

    Returns:
        ``(ok, message)`` where ``ok=False`` signals a degenerate run.
    """
    tune_dir = project / hpo_dir / f"{tune_prefix}{model}"
    csv_path = tune_dir / "tune_results.csv"
    if not csv_path.exists():
        return False, f"tune_results.csv missing at {csv_path}"

    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    total = len(rows)
    if total < min_rows:
        return False, f"only {total} row(s) in tune_results.csv (< {min_rows})"

    fitness_key = _find_fitness_column(list(fieldnames))
    if fitness_key is None:
        return False, f"'fitness' column missing in {csv_path}"

    good, best = _count_valid_trials(rows, fitness_key)
    if good < min_trials:
        return False, (
            f"DEGENERATE — {good}/{total} trial(s) with fitness>0 "
            f"(< {min_trials}); best={best:.5f}"
        )
    return True, f"OK — {good}/{total} valid trial(s); best={best:.5f}"


def _print_header(project: Path, args: argparse.Namespace) -> None:
    """Print the top-level summary banner."""
    print(
        f"\n=== check_hpo_validity ==="
        f"\n  project     = {project}"
        f"\n  hpo_dir     = {project / args.hpo_dir}"
        f"\n  models      = {args.models}"
        f"\n  min_trials  = {args.min_trials}"
        f"\n  min_rows    = {args.min_rows}\n"
    )


def _print_actionable_remediation(
    project: Path,
    hpo_dir: str,
    tune_prefix: str,
) -> None:
    """Print a copy-paste remediation snippet pointing at the broken tune dirs."""
    print(
        "\nDelete the affected model directories and re-run Phase 2.\n"
        "Example:\n"
        f"  rm -rf {project / hpo_dir}/{tune_prefix}<model>"
    )


def main() -> int:
    """Validate all requested models and return a process-style exit code.

    Returns:
        ``0`` if every model has a valid HPO, ``1`` otherwise.
    """
    args = parse_args()
    project = Path(args.project)
    _print_header(project, args)

    failures: list[str] = []
    for model in args.models:
        ok, msg = validate_model(
            model=model,
            project=project,
            hpo_dir=args.hpo_dir,
            tune_prefix=args.tune_prefix,
            min_trials=args.min_trials,
            min_rows=args.min_rows,
        )
        status = "OK   " if ok else "FAIL "
        print(f"  [{status}] {model:7s}  {msg}")
        if not ok:
            failures.append(f"{model}: {msg}")

    if failures:
        print("\n[!] HPO invalid for one or more models:")
        for f in failures:
            print(f"  - {f}")
        _print_actionable_remediation(project, args.hpo_dir, args.tune_prefix)
        return 1

    print("\n[OK] All models have a valid HPO.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
