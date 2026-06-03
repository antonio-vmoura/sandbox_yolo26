#!/usr/bin/env python3
"""
check_hpo_validity.py — Valida os outputs da Phase 2 (HPO) após
``tune_all_models_v2.py`` ter rodado.

Motivação
---------
O ``Tuner`` da Ultralytics **não propaga falhas de trials individuais**:
quando um trial morre (por exemplo, depois que o driver NVIDIA cai no host),
o Tuner apenas registra ``fitness=0`` e segue para o próximo trial. No fim,
ele escreve um ``best_hyperparameters.yaml`` que é simplesmente o vetor de
seed inicial e retorna exit-code 0 — fazendo a Phase 2 parecer bem-sucedida
quando na verdade **nenhum trial concluiu**.

Este script percorre os ``tune_results.csv`` de cada modelo afinado e
considera o HPO degenerado se:
  * o CSV não existir,
  * tiver menos de ``--min-rows`` linhas, OU
  * tiver menos de ``--min-trials`` linhas com ``fitness > 0``.

Quando qualquer modelo for considerado degenerado o script retorna
exit-code 1, listando o que precisa ser refeito.

Uso típico (chamado pelo ``run_pipeline.sh`` logo após a Phase 2):

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
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--project", required=True,
        help="Diretório raiz do pipeline (ex.: /workspace/logs/pipeline_e2e_v2).",
    )
    p.add_argument(
        "--models", nargs="+", required=True,
        help="Modelos afinados nesta Phase 2 (ex.: nano small medium large xlarge).",
    )
    p.add_argument(
        "--hpo-dir", default="hpo/hpo_v3",
        help="Subdiretório do HPO relativo a --project (default: hpo/hpo_v3).",
    )
    p.add_argument(
        "--tune-prefix", default="tune_isic_2018_task_1_",
        help=(
            "Prefixo do nome de diretório de cada tune (default: "
            "tune_isic_2018_task_1_<model>)."
        ),
    )
    p.add_argument(
        "--min-trials", type=int, default=1,
        help=(
            "Mínimo de trials com fitness>0 exigidos por modelo (default: 1). "
            "Use um valor maior (ex.: 5) se quiser garantir busca minimamente "
            "explorada antes de aceitar o HPO como válido."
        ),
    )
    p.add_argument(
        "--min-rows", type=int, default=1,
        help=(
            "Mínimo de linhas totais esperadas em tune_results.csv (default: 1). "
            "Não conta cabeçalho. Usado para detectar CSV vazio."
        ),
    )
    return p.parse_args()


def _find_fitness_column(fieldnames: list[str]) -> str | None:
    """Localiza a coluna de fitness no CSV, tolerante a variações de caixa."""
    for col in fieldnames:
        if col is not None and col.strip().lower() == "fitness":
            return col
    return None


def validate_model(
    model: str,
    project: Path,
    hpo_dir: str,
    tune_prefix: str,
    min_trials: int,
    min_rows: int,
) -> tuple[bool, str]:
    """Valida o HPO de um único modelo.

    Returns (ok, message).
    """
    tune_dir = project / hpo_dir / f"{tune_prefix}{model}"
    csv_path = tune_dir / "tune_results.csv"
    if not csv_path.exists():
        return False, f"tune_results.csv ausente em {csv_path}"

    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    total = len(rows)
    if total < min_rows:
        return False, f"apenas {total} linha(s) em tune_results.csv (< {min_rows})"

    fitness_key = _find_fitness_column(list(fieldnames))
    if fitness_key is None:
        return False, f"coluna 'fitness' ausente em {csv_path}"

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

    if good < min_trials:
        return False, (
            f"DEGENERADO — {good}/{total} trial(s) com fitness>0 "
            f"(< {min_trials}); best={best:.5f}"
        )
    return True, f"OK — {good}/{total} trial(s) válidos; best={best:.5f}"


def main() -> int:
    args = parse_args()
    project = Path(args.project)

    print(
        f"\n=== check_hpo_validity ==="
        f"\n  project     = {project}"
        f"\n  hpo_dir     = {project / args.hpo_dir}"
        f"\n  models      = {args.models}"
        f"\n  min_trials  = {args.min_trials}"
        f"\n  min_rows    = {args.min_rows}\n"
    )

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
        status = "OK   " if ok else "FALHA"
        print(f"  [{status}] {model:7s}  {msg}")
        if not ok:
            failures.append(f"{model}: {msg}")

    if failures:
        print("\n[!] HPO inválido em um ou mais modelos:")
        for f in failures:
            print(f"  - {f}")
        # Mensagem acionável para o usuário humano que estiver lendo o log:
        print(
            "\nApague os diretórios dos modelos afetados e re-execute a Phase 2.\n"
            "Exemplo:\n"
            f"  rm -rf {project / args.hpo_dir}/{args.tune_prefix}<model>"
        )
        return 1

    print("\n[OK] Todos os modelos têm HPO válido.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
