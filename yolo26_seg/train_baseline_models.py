"""
train_baseline_models.py — Phase 1 (Baseline) do pipeline YOLO26-seg
no ISIC 2018 Task 1.

Treina sequencialmente os 5 tamanhos (nano, small, medium, large, xlarge)
usando **os hiperparâmetros padrão do Ultralytics** (sem otimizador
customizado, sem augmentations agressivas e sem HP tunados): apenas
`epochs=120`, `patience=20`, `deterministic=True`, `seed=0` são fixados
para reprodutibilidade e para alinhar com o protocolo das fases seguintes
do pipeline.

NB: AMP (mixed-precision) está **desabilitado por padrão** (`amp=False`)
para todas as variantes desde a observação de overflow FP16 na variante
xlarge (NaN na cls-loss). Isto é um desvio explícito do default do
Ultralytics (`amp=True`) feito em prol da **uniformidade experimental**
entre tamanhos arquiteturais e da **invariância de hardware**. Use
`--amp` para reativar pontualmente.

Saídas:
    ``<project>/phase1_baseline/yolo26_<model>_baseline/{weights, results.csv, ...}``

Uso:
    # Treinar TODOS os 5 tamanhos (default):
    python train_baseline_models.py

    # Treinar um subconjunto:
    python train_baseline_models.py --models small medium

    # Re-treinar (mesmo que best.pt exista):
    python train_baseline_models.py --force

    docker run --gpus all -it --rm --ipc=host \\
        --user $(id -u):$(id -g) \\
        -e TORCH_HOME=/workspace/cache/torch -e HOME=/workspace/cache \\
        -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
        -v $(pwd)/datasets:/workspace/datasets \\
        -v $(pwd)/logs:/workspace/logs \\
        -v $(pwd)/yolo26_seg:/workspace/yolo26_seg \\
        -v $(pwd)/utils:/workspace/utils \\
        -v $(pwd)/cache:/workspace/cache \\
        -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro \\
        yolo26_ft \\
        python /workspace/yolo26_seg/train_baseline_models.py \\
        2>&1 | tee logs/phase1_baseline.log
"""

import argparse
import sys
import time
import traceback
from pathlib import Path

from ultralytics import YOLO

# ----------------------------------------------------------------------------
# CONFIGURAÇÕES GERAIS
# ----------------------------------------------------------------------------
VERSION = "phase1_baseline"

DEFAULT_ORDER = ["nano", "small", "medium", "large", "xlarge"]

WEIGHTS = {
    "nano":   "/workspace/cache/yolo26n-seg.pt",
    "small":  "/workspace/cache/yolo26s-seg.pt",
    "medium": "/workspace/cache/yolo26m-seg.pt",
    "large":  "/workspace/cache/yolo26l-seg.pt",
    "xlarge": "/workspace/cache/yolo26x-seg.pt",
}


# ----------------------------------------------------------------------------
# LÓGICA DE TREINO E ORQUESTRAÇÃO
# ----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 1 — Treino baseline (defaults Ultralytics) nos YOLO26-seg.",
    )
    p.add_argument(
        "--models", nargs="+", default=DEFAULT_ORDER, choices=DEFAULT_ORDER,
        help=f"Subconjunto de modelos para treinar (default: {DEFAULT_ORDER}).",
    )
    p.add_argument(
        "--data",
        default="/workspace/datasets/isic_2018_task1_yolo26/data.yaml",
        help="Path do data.yaml.",
    )
    p.add_argument(
        "--device", default="0,1",
        help="GPUs (default: '0,1' DDP). Use '0' para single-GPU ou 'cpu'.",
    )
    p.add_argument(
        "--project", default="/workspace/logs",
        help="Diretório raiz dos logs (default: /workspace/logs).",
    )
    p.add_argument(
        "--epochs", type=int, default=120,
        help="Épocas por modelo (default: 120).",
    )
    p.add_argument(
        "--patience", type=int, default=20,
        help="Early-stopping patience (default: 20).",
    )
    p.add_argument(
        "--imgsz", type=int, default=640,
        help="Tamanho de imagem (default: 640).",
    )
    p.add_argument(
        "--seed", type=int, default=0,
        help="Semente determinística (default: 0).",
    )
    p.add_argument(
        "--amp", action="store_true",
        help=(
            "Ativa Automatic Mixed Precision (FP16). Default: desativado, "
            "para garantir uniformidade entre tamanhos arquiteturais e "
            "evitar overflow FP16 que gera NaN no cls-loss da variante "
            "xlarge (observado empiricamente no ISIC 2018 Task 1)."
        ),
    )
    p.add_argument(
        "--force", action="store_true",
        help="Re-executa mesmo se o ficheiro best.pt já existir.",
    )
    return p.parse_args()


def parse_device(arg: str):
    if "," in arg:
        return [int(x) for x in arg.split(",")]
    if arg == "cpu":
        return "cpu"
    return int(arg)


def train_one_baseline(model_size: str, args: argparse.Namespace, device) -> dict:
    """Treina um único tamanho com hiperparâmetros padrão do Ultralytics."""
    run_root = Path(args.project) / VERSION
    run_name = f"yolo26_{model_size}_baseline"
    out_dir = run_root / run_name
    best_pt = out_dir / "weights" / "best.pt"

    if best_pt.exists() and not args.force:
        return {
            "model": model_size,
            "skipped": True,
            "reason": f"{best_pt} já existe (use --force p/ re-executar)",
            "elapsed_min": 0.0,
        }

    weights_path = WEIGHTS[model_size]
    if not Path(weights_path).exists():
        print(
            f"  [aviso] pesos pré-treinados não encontrados em {weights_path}; "
            f"Ultralytics fará o download automático na primeira utilização.",
        )

    print("\n" + "=" * 80)
    print(f"=== PHASE 1 (BASELINE): {model_size}  ({args.epochs} épocas × paciência {args.patience})")
    print(f"  data       = {args.data}")
    print(f"  device     = {device}")
    print(f"  output     = {out_dir}")
    print(f"  amp        = {args.amp}  (Ultralytics default = True; desativado aqui p/ uniformidade)")
    print("  HP         = Ultralytics defaults (sem otimizador customizado / sem HP tunados)")
    print("=" * 80)

    t0 = time.perf_counter()
    model = YOLO(weights_path)

    # Defaults Ultralytics — não sobrescrevemos optimizer, cos_lr, etc.
    # AMP é o único desvio explícito do default (desativado por padrão).
    model.train(
        data=args.data,
        project=str(run_root),
        name=run_name,
        task="segment",
        pretrained=True,
        imgsz=args.imgsz,
        device=device,
        epochs=args.epochs,
        patience=args.patience,
        amp=args.amp,
        deterministic=True,
        seed=args.seed,
        save=True,
        plots=True,
        val=True,
        verbose=True,
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

    print(f"Orquestração Phase 1 (Baseline) para os modelos: {args.models}")
    print(f"  device       = {device}")
    print(f"  data         = {args.data}")
    print(f"  project      = {args.project}")
    print(f"  epochs       = {args.epochs}  patience = {args.patience}")
    print(f"  seed         = {args.seed}  imgsz = {args.imgsz}")
    print(f"  amp          = {args.amp}")
    print(f"  force re-run = {args.force}")

    summary: list[dict] = []
    failures: list[tuple[str, str]] = []
    t_total = time.perf_counter()

    for i, m in enumerate(args.models, 1):
        print(f"\n[{i}/{len(args.models)}] Treinando baseline do modelo: {m}")
        try:
            stats = train_one_baseline(m, args, device)
            summary.append(stats)
            if stats["skipped"]:
                print(f"  [ignorado] {stats['reason']}")
            else:
                print(f"  [concluído] tempo de treino: {stats['elapsed_min']:.1f} min")
        except Exception:
            tb = traceback.format_exc()
            print(f"  [falha] exceção no modelo {m}:\n{tb}")
            summary.append({
                "model": m, "skipped": False, "reason": "fail",
                "elapsed_min": 0.0,
            })
            failures.append((m, tb))

    print("\n" + "=" * 80)
    print(f"=== SUMÁRIO PHASE 1 (BASELINE)")
    print("=" * 80)
    for s in summary:
        if s["skipped"]:
            print(f"  {s['model']:<8} : ignorado ({s['reason']})")
        elif s["reason"] == "fail":
            print(f"  {s['model']:<8} : FALHOU")
        else:
            print(f"  {s['model']:<8} : sucesso  ({s['elapsed_min']:.1f} min)")

    total_min = (time.perf_counter() - t_total) / 60
    print(f"\nTempo total Phase 1: {total_min:.1f} min ({total_min / 60:.2f} h)")

    if failures:
        print(f"\n[!] Encontrada(s) {len(failures)} falha(s):")
        for m, _ in failures:
            print(f"    - Modelo {m}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
