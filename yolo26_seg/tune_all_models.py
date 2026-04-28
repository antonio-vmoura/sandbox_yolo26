"""
tune_all_models.py — Orquestra HPO (`model.tune()`) sequencialmente nos
5 tamanhos de YOLO26-seg (nano, small, medium, large, xlarge) usando o
mesmo `SEARCH_SPACE` e config v7 do `tune_isic_2018_task_1.py`.

Filosofia:
    * Mesmo espaço de busca para todos os tamanhos — comparação direta
      depois.
    * `device=[0,1]` (DDP) por padrão para acelerar.
    * Skip automático: se `best_hyperparameters.yaml` já existe para um
      tamanho, pula (idempotente — pode interromper e retomar).
    * Log por modelo em `logs/tune_isic_2018_task_1_<model>.log`.

Custo estimado (2× V100S, 30 iter × 30 ep cada):
    nano   ≈  3-4h     (modelo pequeno, treina rápido por trial)
    small  ≈  5-7h
    medium ≈  8-12h
    large  ≈ 12-18h
    xlarge ≈ 18-30h
    -----------------
    total  ≈ 1.5-3 dias  → considere rodar overnight ou usar `--models`
                            para subset. Em hardware mais lento pode
                            facilmente dobrar.

Uso:
    # Todos os modelos (default):
    python tune_all_models.py

    # Subset:
    python tune_all_models.py --models nano small medium

    # Tune mais agressivo:
    python tune_all_models.py --iterations 50 --epochs 40

    # Ignorar runs já completados (default: skip):
    python tune_all_models.py --force

    # Color space alternativo:
    python tune_all_models.py --data /workspace/datasets/isic_2018_task1_yolo26_hed/data.yaml

Em docker:
    docker run ... yolo26_ft \\
      python /workspace/yolo26_seg/tune_all_models.py 2>&1 \\
      | tee logs/tune_all_models.log

Saídas (uma pasta por modelo em /workspace/logs/):
    tune_isic_2018_task_1_<model>/
      ├── best_hyperparameters.yaml   ← entrada do train_with_tuned_hp
      ├── tune_results.ndjson         ← histórico de todos os trials
      ├── tune_scatter_plot.png
      └── tune_fitness.png
"""

import argparse
import sys
import time
import traceback
from pathlib import Path

from ultralytics import YOLO

# Reaproveita SEARCH_SPACE e WEIGHTS do script single-model
from tune_isic_2018_task_1 import SEARCH_SPACE, WEIGHTS  # noqa: E402

DEFAULT_ORDER = ["nano", "small", "medium", "large", "xlarge"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="HPO sequencial nos 5 tamanhos de YOLO26-seg.",
    )
    p.add_argument(
        "--models", nargs="+", default=DEFAULT_ORDER, choices=DEFAULT_ORDER,
        help=f"Subset de modelos para tunar (default: {DEFAULT_ORDER}).",
    )
    p.add_argument(
        "--iterations", type=int, default=30,
        help="Iterações por modelo (default: 30).",
    )
    p.add_argument(
        "--epochs", type=int, default=30,
        help="Épocas por trial (default: 30).",
    )
    p.add_argument(
        "--patience", type=int, default=10,
        help="Early-stopping patience por trial (default: 10).",
    )
    p.add_argument(
        "--data",
        default="/workspace/datasets/isic_2018_task1_yolo26/data.yaml",
        help="Path do data.yaml.",
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
        help="Re-roda mesmo se best_hyperparameters.yaml já existir.",
    )
    return p.parse_args()


def parse_device(arg: str):
    if "," in arg:
        return [int(x) for x in arg.split(",")]
    return int(arg)


def tune_one_model(model_size: str, args: argparse.Namespace, device) -> dict:
    """Roda model.tune() para um único tamanho. Retorna stats da execução."""
    run_name = f"tune_isic_2018_task_1_{model_size}"
    out_dir = Path(args.project) / run_name
    best_yaml = out_dir / "best_hyperparameters.yaml"

    if best_yaml.exists() and not args.force:
        return {
            "model": model_size,
            "skipped": True,
            "reason": f"{best_yaml} já existe (use --force p/ re-rodar)",
            "elapsed_min": 0.0,
        }

    print("\n" + "=" * 80)
    print(f"=== TUNE START: {model_size}  ({args.iterations} iter × {args.epochs} ep)")
    print("=" * 80)

    t0 = time.perf_counter()
    model = YOLO(WEIGHTS[model_size])

    fixed_v7 = dict(
        # Infra
        data=args.data,
        project=args.project,
        name=run_name,
        task="segment",
        pretrained=True,
        imgsz=640,
        # Hardware
        device=device,
        batch=32,
        workers=8,
        cache=False,
        # Otimização (fixa do v7)
        amp=True,
        optimizer="MuSGD",
        cos_lr=True,
        close_mosaic=15,
        erasing=0.0,
        nbs=64,
        # Loop curto p/ HPO
        epochs=args.epochs,
        patience=args.patience,
        deterministic=True,
        seed=0,
        save=True,
        plots=False,
        val=True,
        verbose=False,
    )

    model.tune(
        space=SEARCH_SPACE,
        iterations=args.iterations,
        use_ray=False,
        **fixed_v7,
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

    print(f"HPO em sequência para: {args.models}")
    print(f"  iterations/model = {args.iterations}")
    print(f"  epochs/trial     = {args.epochs}")
    print(f"  device           = {device}")
    print(f"  data             = {args.data}")
    print(f"  project          = {args.project}")
    print(f"  force re-run     = {args.force}")

    summary: list[dict] = []
    failures: list[tuple[str, str]] = []
    t_total = time.perf_counter()

    for i, m in enumerate(args.models, 1):
        print(f"\n[{i}/{len(args.models)}] {m}")
        try:
            stats = tune_one_model(m, args, device)
            summary.append(stats)
            if stats["skipped"]:
                print(f"  [skip] {stats['reason']}")
            else:
                print(f"  [done] {m} em {stats['elapsed_min']:.1f} min")
        except Exception as e:  # noqa: BLE001
            tb = traceback.format_exc()
            print(f"  [FAIL] {m}: {e}\n{tb}", file=sys.stderr)
            failures.append((m, str(e)))
            summary.append({
                "model": m, "skipped": False, "reason": f"FAILED: {e}",
                "elapsed_min": -1.0,
            })

    total_min = (time.perf_counter() - t_total) / 60

    # ------------------------------------------------------------------------
    # Sumário
    # ------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("=== SUMÁRIO HPO ALL MODELS")
    print("=" * 80)
    for s in summary:
        status = (
            "skipped" if s["skipped"]
            else f"FAILED ({s['reason']})" if s["elapsed_min"] < 0
            else f"{s['elapsed_min']:.1f} min"
        )
        print(f"  {s['model']:8s} : {status}")
    print(f"\nTempo total: {total_min:.1f} min  ({total_min / 60:.2f} h)")

    if failures:
        print(f"\n{len(failures)} modelo(s) falharam:", [m for m, _ in failures])
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
