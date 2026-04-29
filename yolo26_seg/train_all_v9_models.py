"""
train_all_v9_models.py — Orquestra o treino full-length (120 ep) v9
sequencialmente nos 5 tamanhos de YOLO26-seg, cada um lendo seu próprio
`best_hyperparameters.yaml` gerado pela rodada de HPO.

Filosofia (igual `tune_all_models.py`):
    * Sequencial — um tamanho por vez (evita disputar GPU).
    * Idempotente — pula tamanhos que já têm `weights/best.pt` no diretório
      de saída (use `--force` para re-rodar).
    * Continua mesmo se um modelo falhar (próximos seguem; resumo no fim
      indica quais falharam e por quê).

Pré-requisito: já ter rodado `tune_all_models.py --space refined ...` para
gerar `logs/tune_isic_2018_task_1_<size>/best_hyperparameters.yaml` em
cada tamanho que você quer treinar.

Custo estimado (2× V100S, 120 ep × 5 tamanhos):
    nano   ≈  20-40 min   (early-stop frequente em ~30 ep com hp tunado)
    small  ≈  20-50 min
    medium ≈  40-80 min
    large  ≈  60-120 min
    xlarge ≈  90-180 min
    -----------------
    total  ≈  3-7 h sequencial. Pode rodar overnight.

Uso:
    # Todos os 5 tamanhos (default):
    python train_all_v9_models.py

    # Subset:
    python train_all_v9_models.py --models nano small

    # Re-rodar mesmo se já existe:
    python train_all_v9_models.py --force

Em docker:
    docker run ... yolo26_ft \\
      python /workspace/yolo26_seg/train_all_v9_models.py \\
      2>&1 | tee logs/train_v9_all.log

Saídas (uma pasta por modelo em /workspace/logs/):
    yolo26_<size>_ft_isic_2018_v9/
      ├── weights/best.pt
      ├── results.csv
      ├── args.yaml
      └── ... (curvas, plots, val/)
"""

import argparse
import sys
import time
import traceback
from pathlib import Path

# Reaproveita logica single-size do v9
from train_isic_2018_task_1_v9 import (  # noqa: E402
    WEIGHTS,
    train_v9_one_size,
)

DEFAULT_ORDER = ["nano", "small", "medium", "large", "xlarge"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Treino v9 sequencial nos 5 tamanhos de YOLO26-seg.",
    )
    p.add_argument(
        "--models", nargs="+", default=DEFAULT_ORDER, choices=DEFAULT_ORDER,
        help=f"Subset de modelos para treinar (default: {DEFAULT_ORDER}).",
    )
    p.add_argument(
        "--data",
        default="/workspace/datasets/isic_2018_task1_yolo26/data.yaml",
        help="Path do data.yaml (manter o mesmo usado no HPO).",
    )
    p.add_argument(
        "--device", default="0,1",
        help="GPUs (default: '0,1' DDP). Use '0' para single-GPU.",
    )
    p.add_argument(
        "--project", default="/workspace/logs",
        help="Diretório raiz dos logs (default: /workspace/logs).",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Re-roda mesmo se yolo26_<size>_ft_isic_2018_v9/weights/best.pt "
             "já existir.",
    )
    return p.parse_args()


def parse_device(arg: str):
    if "," in arg:
        return [int(x) for x in arg.split(",")]
    return int(arg)


def train_one(model_size: str, args: argparse.Namespace, device) -> dict:
    """Roda o treino v9 para um tamanho. Retorna stats da execução."""
    out_dir = Path(args.project) / f"yolo26_{model_size}_ft_isic_2018_v9"
    best_pt = out_dir / "weights" / "best.pt"
    hp_yaml = (
        Path(args.project)
        / f"tune_isic_2018_task_1_{model_size}"
        / "best_hyperparameters.yaml"
    )

    if best_pt.exists() and not args.force:
        return {
            "model": model_size,
            "skipped": True,
            "reason": f"{best_pt} já existe (use --force p/ re-rodar)",
            "elapsed_min": 0.0,
        }

    if not hp_yaml.exists():
        return {
            "model": model_size,
            "skipped": True,
            "reason": (
                f"hp YAML não encontrado: {hp_yaml}. "
                f"Rode `tune_all_models.py --space refined --models {model_size}` antes."
            ),
            "elapsed_min": 0.0,
        }

    print("\n" + "=" * 80)
    print(f"=== TRAIN v9 START: {model_size}  (120 ep × patience 20)")
    print(f"  hp     = {hp_yaml}")
    print(f"  output = {out_dir}")
    print("=" * 80)

    t0 = time.perf_counter()
    train_v9_one_size(
        model_size=model_size,
        hp_yaml=hp_yaml,
        data=args.data,
        project=args.project,
        device=device,
    )
    elapsed = (time.perf_counter() - t0) / 60
    return {
        "model": model_size,
        "skipped": False,
        "reason": None,
        "elapsed_min": elapsed,
    }


def main() -> int:
    args = parse_args()
    device = parse_device(args.device)

    print(f"Treino v9 em sequência para: {args.models}")
    print(f"  device       = {device}")
    print(f"  data         = {args.data}")
    print(f"  project      = {args.project}")
    print(f"  force re-run = {args.force}")

    summary: list[dict] = []
    failures: list[tuple[str, str]] = []
    t_total = time.perf_counter()

    for i, m in enumerate(args.models, 1):
        print(f"\n[{i}/{len(args.models)}] {m}")
        try:
            stats = train_one(m, args, device)
            summary.append(stats)
            if stats["skipped"]:
                print(f"  [skip] {stats['reason']}")
            else:
                print(f"  [done] {stats['elapsed_min']:.1f} min")
        except Exception:
            tb = traceback.format_exc()
            print(f"  [fail] exceção em {m}:\n{tb}")
            summary.append({
                "model": m, "skipped": False, "reason": "fail",
                "elapsed_min": 0.0,
            })
            failures.append((m, tb))

    # Resumo final
    print("\n" + "=" * 80)
    print("=== SUMÁRIO TREINO v9 ALL MODELS")
    print("=" * 80)
    for s in summary:
        if s["skipped"]:
            print(f"  {s['model']:<8} : skipped  ({s['reason']})")
        elif s["reason"] == "fail":
            print(f"  {s['model']:<8} : FAIL")
        else:
            print(f"  {s['model']:<8} : ok       ({s['elapsed_min']:.1f} min)")

    total_min = (time.perf_counter() - t_total) / 60
    print(f"\nTempo total: {total_min:.1f} min  ({total_min / 60:.2f} h)")

    if failures:
        print(f"\n[!] {len(failures)} falha(s):")
        for m, _ in failures:
            print(f"    - {m}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
