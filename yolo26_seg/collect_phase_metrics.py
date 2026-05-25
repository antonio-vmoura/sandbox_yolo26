"""
collect_phase_metrics.py — Consolida métricas single-split (Phases 1 e 3).

Para uma dada fase (baseline ou optimized) percorre os ``results.csv`` de
cada modelo, seleciona a melhor época pelo critério padrão Ultralytics
(``metrics/mAP50-95(M)``), calcula F1-Score (Box e Mask) e grava um CSV +
JSON consolidados em ``<project>/pipeline_summary/<phase>_metrics.{csv,json}``.

Convencão de diretórios:
    Phase 1 (baseline):  <project>/phase1_baseline/yolo26_<MODEL>_baseline/results.csv
    Phase 3 (optimized): <project>/yolo26_<MODEL>_ft_isic_2018_v11/results.csv

Uso:
    python collect_phase_metrics.py --phase baseline
    python collect_phase_metrics.py --phase optimized
    python collect_phase_metrics.py --phase optimized --models small medium
"""

import argparse
import csv
import json
import sys
from pathlib import Path

DEFAULT_ORDER = ["nano", "small", "medium", "large", "xlarge"]

# Chaves de métricas no ``results.csv`` do Ultralytics (segment).
METRIC_KEYS = {
    "precision_b":  "metrics/precision(B)",
    "recall_b":     "metrics/recall(B)",
    "map50_b":      "metrics/mAP50(B)",
    "map5095_b":    "metrics/mAP50-95(B)",
    "precision_m":  "metrics/precision(M)",
    "recall_m":     "metrics/recall(M)",
    "map50_m":      "metrics/mAP50(M)",
    "map5095_m":    "metrics/mAP50-95(M)",
}

BEST_EPOCH_KEY = "metrics/mAP50-95(M)"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Coleta métricas single-split (Phases 1 ou 3) num CSV/JSON.",
    )
    p.add_argument(
        "--phase", choices=["baseline", "optimized"], required=True,
        help="Fase para coletar: 'baseline' (Phase 1) ou 'optimized' (Phase 3).",
    )
    p.add_argument(
        "--models", nargs="+", default=DEFAULT_ORDER, choices=DEFAULT_ORDER,
        help=f"Subconjunto de modelos (default: {DEFAULT_ORDER}).",
    )
    p.add_argument(
        "--project", default="/workspace/logs",
        help="Diretório raiz dos logs (default: /workspace/logs).",
    )
    p.add_argument(
        "--out-dir", default=None,
        help="Diretório de saída para os CSV/JSON (default: <project>/pipeline_summary).",
    )
    return p.parse_args()


def results_csv_path(project: Path, phase: str, model: str) -> Path:
    if phase == "baseline":
        return project / "phase1_baseline" / f"yolo26_{model}_baseline" / "results.csv"
    # optimized — caminho do train_all_models.py (versão v11)
    return project / f"yolo26_{model}_ft_isic_2018_v11" / "results.csv"


def parse_best_epoch_metrics(results_csv: Path) -> dict[str, float]:
    """Retorna métricas da melhor época (mAP50-95(M) como critério)."""
    rows: list[dict] = []
    with results_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k.strip(): v for k, v in row.items()})
    if not rows:
        raise ValueError(f"results.csv vazio: {results_csv}")

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


def main() -> int:
    args = parse_args()
    project = Path(args.project).resolve()
    out_dir = Path(args.out_dir) if args.out_dir else project / "pipeline_summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    per_model: list[dict] = []
    missing: list[str] = []

    for m in args.models:
        csv_path = results_csv_path(project, args.phase, m)
        if not csv_path.exists():
            print(f"  [aviso] results.csv não encontrado para {m}: {csv_path}")
            missing.append(m)
            continue
        try:
            metrics = parse_best_epoch_metrics(csv_path)
        except Exception as e:
            print(f"  [erro] falha ao ler {csv_path}: {e}")
            missing.append(m)
            continue
        row = {"model": m, "results_csv": str(csv_path), **metrics}
        per_model.append(row)
        print(
            f"  {m:<8} : "
            f"mAP50(M)={metrics['map50_m']:.4f} "
            f"mAP50-95(M)={metrics['map5095_m']:.4f} "
            f"P(M)={metrics['precision_m']:.4f} "
            f"R(M)={metrics['recall_m']:.4f} "
            f"F1(M)={metrics['f1_m']:.4f}",
        )

    csv_out = out_dir / f"{args.phase}_metrics.csv"
    json_out = out_dir / f"{args.phase}_metrics.json"

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
            {"phase": args.phase, "models": per_model, "missing": missing},
            f, indent=2, sort_keys=True,
        )

    print(f"\nArtefatos gerados:")
    print(f"  CSV : {csv_out}")
    print(f"  JSON: {json_out}")
    if missing:
        print(f"  [aviso] sem resultados para: {missing}")
    return 0 if per_model else 1


if __name__ == "__main__":
    sys.exit(main())
