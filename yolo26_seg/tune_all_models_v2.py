"""
tune_all_models.py — Orquestra HPO (`model.tune()`) no YOLO26-seg
para o ISIC 2018 Task 1 via algoritmo genético embutido no Ultralytics.

Script unificado: Contém a definição dos espaços de busca (wide e refined)
e a lógica de orquestração para rodar 1 ou N modelos sequencialmente.

Uso:
    # Rodar apenas um modelo (exploração inicial):
    python tune_all_models.py --models small --space wide

    # Rodar todos os tamanhos sequencialmente (default):
    python tune_all_models.py

    # Rodar com espaço refined (follow-up) e mais iterações:
    python tune_all_models.py --space refined --iterations 50

    # Ignorar runs já completados (default: skip):
    python tune_all_models.py --force
    
----- WIDE -----

docker run --gpus all -it --rm --ipc=host \
  --user $(id -u):$(id -g) \
  -e TORCH_HOME=/workspace/cache/torch -e HOME=/workspace/cache \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v $(pwd)/datasets:/workspace/datasets \
  -v $(pwd)/logs:/workspace/logs \
  -v $(pwd)/yolo26_seg:/workspace/yolo26_seg \
  -v $(pwd)/utils:/workspace/utils \
  -v $(pwd)/cache:/workspace/cache \
  -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro \
  yolo26_ft \
  python /workspace/yolo26_seg/tune_all_models_v2.py \
    --space wide --iterations 50 --models small \
  2>&1 | tee logs/tune_all_models_v2.log  

----- REFINED -----

docker run --gpus all -it --rm --ipc=host \
  --user $(id -u):$(id -g) \
  -e TORCH_HOME=/workspace/cache/torch -e HOME=/workspace/cache \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v $(pwd)/datasets:/workspace/datasets \
  -v $(pwd)/logs:/workspace/logs \
  -v $(pwd)/yolo26_seg:/workspace/yolo26_seg \
  -v $(pwd)/utils:/workspace/utils \
  -v $(pwd)/cache:/workspace/cache \
  -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro \
  yolo26_ft \
  python /workspace/yolo26_seg/tune_all_models_v2.py \
    --space refined --iterations 50 \
  2>&1 | tee logs/tune_all_refined.log
  
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
DEFAULT_ORDER = ["nano", "small", "medium", "large", "xlarge"]

WEIGHTS = {
    "nano":   "/workspace/cache/yolo26n-seg.pt",
    "small":  "/workspace/cache/yolo26s-seg.pt",
    "medium": "/workspace/cache/yolo26m-seg.pt",
    "large":  "/workspace/cache/yolo26l-seg.pt",
    "xlarge": "/workspace/cache/yolo26x-seg.pt",
}

# ----------------------------------------------------------------------------
# SEARCH SPACES
# ----------------------------------------------------------------------------
SEARCH_SPACE_WIDE = {
    # ---- Aprendizado ----
    "lr0":            (5e-4, 5e-3),
    "lrf":            (0.005, 0.05),
    "momentum":       (0.85, 0.95, 0.3),
    "weight_decay":   (0.0, 0.001),
    "warmup_epochs":  (1.0, 5.0),
    "warmup_momentum": (0.5, 0.95),

    # ---- Pesos das losses ----
    "box":            (3.0, 12.0),
    "cls":            (0.2, 1.5),
    "dfl":            (0.8, 3.0),

    # ---- Augmentação de cor ----
    "hsv_h":          (0.0, 0.03),
    "hsv_s":          (0.3, 0.9),
    "hsv_v":          (0.2, 0.7),

    # ---- Augmentação geométrica ----
    "degrees":        (0.0, 30.0),
    "translate":      (0.0, 0.3),
    "scale":          (0.2, 0.7),
    "shear":          (0.0, 10.0),         # NOVO: Adicionado do default.yaml
    "fliplr":         (0.0, 0.6),
    "flipud":         (0.0, 0.5),

    # ---- Mixing augmentations ----
    "mosaic":         (0.5, 1.0),
    "mixup":          (0.0, 0.3),
    "copy_paste":     (0.0, 0.3),
    "cutmix":         (0.0, 0.3),          # NOVO: Adicionado do default.yaml
}

SEARCH_SPACE_REFINED = {
    # ---- Aprendizado ----
    "lr0":            (1e-3, 4e-3),
    "lrf":            (0.005, 0.05),
    "momentum":       (0.85, 0.95, 0.3),
    "weight_decay":   (1e-6, 1e-4),
    "warmup_epochs":  (1.0, 5.0),

    # ---- Pesos das losses ----
    "cls":            (0.2, 1.5),
    "dfl":            (0.8, 1.5),

    # ---- Augmentação de cor ----
    "hsv_h":          (0.005, 0.025),
    "hsv_s":          (0.3, 0.9),
    "hsv_v":          (0.2, 0.7),

    # ---- Augmentação geométrica ----
    "translate":      (0.05, 0.20),
    "flipud":         (0.0, 0.10),

    # ---- Mixing augmentations ----
    "mosaic":         (0.7, 1.0),
    "mixup":          (0.0, 0.05),
    "copy_paste":     (0.0, 0.05),
}

def get_search_space(name: str) -> dict:
    if name == "wide":
        return SEARCH_SPACE_WIDE
    if name == "refined":
        return SEARCH_SPACE_REFINED
    raise ValueError(f"--space inválido: {name!r}. Use 'wide' ou 'refined'.")


# ----------------------------------------------------------------------------
# CLI & LÓGICA PRINCIPAL
# ----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="HPO sequencial ou individual nos modelos YOLO26-seg.",
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
    p.add_argument(
        "--space", choices=["wide", "refined"], default="wide",
        help="Search space — 'wide' (exploração inicial) ou 'refined' (follow-up).",
    )
    return p.parse_args()


def parse_device(arg: str):
    if "," in arg:
        return [int(x) for x in arg.split(",")]
    return int(arg)


def tune_one_model(model_size: str, args: argparse.Namespace, device, space: dict) -> dict:
    """Roda model.tune() para um único tamanho e retorna stats."""
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
        space=space,
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
    space = get_search_space(args.space)

    print(f"HPO para: {args.models}")
    print(f"  iterations/model = {args.iterations}")
    print(f"  epochs/trial     = {args.epochs}")
    print(f"  device           = {device}")
    print(f"  data             = {args.data}")
    print(f"  project          = {args.project}")
    print(f"  search space     = {args.space!r} ({len(space)} hp)")
    print(f"  force re-run     = {args.force}")

    summary: list[dict] = []
    failures: list[tuple[str, str]] = []
    t_total = time.perf_counter()

    for i, m in enumerate(args.models, 1):
        print(f"\n[{i}/{len(args.models)}] {m}")
        try:
            stats = tune_one_model(m, args, device, space)
            summary.append(stats)
            if stats["skipped"]:
                print(f"  [skip] {stats['reason']}")
            else:
                print(f"  [done] {m} em {stats['elapsed_min']:.1f} min")
        except Exception as e:
            tb = traceback.format_exc()
            print(f"  [FAIL] {m}: {e}\n{tb}", file=sys.stderr)
            failures.append((m, str(e)))
            summary.append({
                "model": m, "skipped": False, "reason": f"FAILED: {e}",
                "elapsed_min": -1.0,
            })

    total_min = (time.perf_counter() - t_total) / 60

    print("\n" + "=" * 80)
    print("=== SUMÁRIO HPO")
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
        print(f"\n{len(failures)} modelo(s) falhou/falharam:", [m for m, _ in failures])
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())