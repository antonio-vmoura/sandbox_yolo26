"""
consolidate_cv_results.py — Consolida métricas de Cross-Validation (Phase 4)
para todos os tamanhos do YOLO26-seg num único CSV/JSON.

Lê os arquivos ``metrics_summary.json`` produzidos por ``train_all_models_cv.py``
em ``<project>/cv/<cv_version>/yolo26_<MODEL>_cv_isic_2018/metrics_summary.json``
e produz dois artefatos consolidados em ``<project>/pipeline_summary/``:

    cv_consolidated.csv   — uma linha por modelo, com mean e std de mAP@50,
                            mAP@50-95, Precision, Recall e F1-Score (Box e Mask).
    cv_consolidated.json  — payload estruturado com per-fold e summary
                            (mean/std) para cada modelo.

Uso:
    python consolidate_cv_results.py
    python consolidate_cv_results.py --models small medium large
    python consolidate_cv_results.py --cv-version cv_v1 --out-dir /workspace/logs/pipeline_summary
"""

import argparse
import csv
import json
import sys
from pathlib import Path

DEFAULT_ORDER = ["nano", "small", "medium", "large", "xlarge"]

# Métricas-chave a relatar (mean/std). Mantemos Box e Mask para completude
# acadêmica; o destaque do paper deve ser Mask (segmentação).
REPORT_METRICS = [
    "map50_b", "map5095_b", "precision_b", "recall_b", "f1_b",
    "map50_m", "map5095_m", "precision_m", "recall_m", "f1_m",
    "best_epoch",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Consolida resultados de CV (Phase 4) por modelo em CSV+JSON.",
    )
    p.add_argument(
        "--models", nargs="+", default=DEFAULT_ORDER, choices=DEFAULT_ORDER,
        help=f"Subconjunto de modelos a consolidar (default: {DEFAULT_ORDER}).",
    )
    p.add_argument(
        "--project", default="/workspace/logs",
        help="Diretório raiz dos logs (default: /workspace/logs).",
    )
    p.add_argument(
        "--cv-version", default="cv_v1",
        help="Subdir de versão de CV em ``<project>/cv/`` (default: cv_v1).",
    )
    p.add_argument(
        "--out-dir", default=None,
        help="Diretório de saída (default: <project>/pipeline_summary).",
    )
    return p.parse_args()


def cv_summary_path(project: Path, cv_version: str, model: str) -> Path:
    return (
        project / "cv" / cv_version
        / f"yolo26_{model}_cv_isic_2018" / "metrics_summary.json"
    )


def load_model_summary(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def main() -> int:
    args = parse_args()
    project = Path(args.project).resolve()
    out_dir = Path(args.out_dir) if args.out_dir else project / "pipeline_summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    per_model: list[dict] = []
    missing: list[str] = []

    for m in args.models:
        path = cv_summary_path(project, args.cv_version, m)
        if not path.exists():
            print(f"  [aviso] summary CV não encontrado para {m}: {path}")
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
        agg = payload.get("summary", {})

        def fmt(key: str, _agg: dict = agg) -> str:
            v = _agg.get(key, {})
            return f"{v.get('mean', 0):.4f}±{v.get('std', 0):.4f}"

        print(
            f"  {m:<8} : k={payload.get('n_folds')} | "
            f"mAP50(M)={fmt('map50_m')}  "
            f"mAP50-95(M)={fmt('map5095_m')}  "
            f"P(M)={fmt('precision_m')}  "
            f"R(M)={fmt('recall_m')}  "
            f"F1(M)={fmt('f1_m')}",
        )

    # ----- CSV consolidado (uma linha por modelo, mean e std por métrica) ----
    csv_path = out_dir / "cv_consolidated.csv"
    if per_model:
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

    # ----- JSON consolidado (per-fold + summary por modelo) -----------------
    json_path = out_dir / "cv_consolidated.json"
    with json_path.open("w") as f:
        json.dump(
            {
                "cv_version": args.cv_version,
                "models": per_model,
                "missing": missing,
            },
            f, indent=2, sort_keys=True,
        )

    print(f"\nArtefatos consolidados:")
    print(f"  CSV : {csv_path}")
    print(f"  JSON: {json_path}")
    if missing:
        print(f"  [aviso] modelos sem summary: {missing}")
    return 0 if per_model else 1


if __name__ == "__main__":
    sys.exit(main())
