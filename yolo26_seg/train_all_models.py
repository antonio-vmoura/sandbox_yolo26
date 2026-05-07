"""
train_all_v9_models.py — Script unificado para fine-tuning do YOLO26-seg
no ISIC 2018 Task 1 (Versão v9 — Per-size HPO-tuned).

Este script junta a lógica de treino individual e a orquestração sequencial.
Cada tamanho de modelo utiliza o seu próprio `best_hyperparameters.yaml` 
encontrado via HPO independente.

Filosofia:
    * Sequencial — um tamanho por vez para maximizar os recursos da GPU.
    * Idempotente — ignora tamanhos que já tenham `weights/best.pt` no 
      diretório de saída (use `--force` para re-executar).
    * Resiliente — continua a execução mesmo se um modelo falhar.

Uso:
    # Treinar TODOS os 5 tamanhos sequencialmente (default):
    python train_all_v9_models.py

    # Treinar um subconjunto ou modelo específico:
    python train_all_v9_models.py --models small medium

    # Forçar re-treino (mesmo que já exista o best.pt):
    python train_all_v9_models.py --force
    

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
  python /workspace/yolo26_seg/train_all_v9_models.py \
    --space wide --iterations 50 --models small \
  2>&1 | tee logs/v9_all_models.log


"""

import argparse
import sys
import time
import traceback
from pathlib import Path

import yaml
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
# LÓGICA DE TREINO E ORQUESTRAÇÃO
# ----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Treino v9 sequencial ou individual nos modelos YOLO26-seg.",
    )
    p.add_argument(
        "--models", nargs="+", default=DEFAULT_ORDER, choices=DEFAULT_ORDER,
        help=f"Subconjunto de modelos para treinar (default: {DEFAULT_ORDER}).",
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
        help="Re-executa mesmo se o ficheiro best.pt já existir.",
    )
    return p.parse_args()


def parse_device(arg: str):
    if "," in arg:
        return [int(x) for x in arg.split(",")]
    return int(arg)


def load_tuned_hp(path: Path) -> dict:
    """Carrega `best_hyperparameters.yaml` gerado pelo Ultralytics Tuner."""
    with path.open("r") as f:
        data = yaml.safe_load(f) or {}
    if not data:
        raise ValueError(
            f"YAML vazio em {path}. Verifique se a afinação anterior falhou."
        )
    return data


def train_one_model(model_size: str, args: argparse.Namespace, device) -> dict:
    """Prepara as configurações e executa o treino para um único modelo."""
    out_dir = Path(args.project) / f"yolo26_{model_size}_ft_isic_2018_v9"
    best_pt = out_dir / "weights" / "best.pt"
    hp_yaml = (
        Path(args.project)
        / f"tune_isic_2018_task_1_{model_size}"
        / "best_hyperparameters.yaml"
    )

    # Verificações de pré-requisitos e estado
    if best_pt.exists() and not args.force:
        return {
            "model": model_size,
            "skipped": True,
            "reason": f"{best_pt} já existe (use --force p/ re-executar)",
            "elapsed_min": 0.0,
        }

    if not hp_yaml.exists():
        return {
            "model": model_size,
            "skipped": True,
            "reason": (
                f"YAML de hiperparâmetros não encontrado: {hp_yaml}. "
                f"Execute primeiro a afinação para este modelo."
            ),
            "elapsed_min": 0.0,
        }

    # Carrega hiperparâmetros
    tuned_hp = load_tuned_hp(hp_yaml)

    print("\n" + "=" * 80)
    print(f"=== INÍCIO DO TREINO v9: {model_size} (120 épocas × paciência 20)")
    print(f"  HP Origem: {hp_yaml}")
    print(f"  Output   : {out_dir}")
    print("  Hiperparâmetros carregados:")
    for k, v in sorted(tuned_hp.items()):
        print(f"    {k:18s} = {v}")
    print("=" * 80)

    t0 = time.perf_counter()
    model = YOLO(WEIGHTS[model_size])

    # Base da v7 (parâmetros fixos para garantir comparabilidade)
    base_v7 = dict(
        data=args.data,
        project=args.project,
        name=f"yolo26_{model_size}_ft_isic_2018_v9",
        task="segment",
        pretrained=True,
        imgsz=640,
        device=device,
        batch=32,
        workers=8,
        cache=False,
        amp=True,
        optimizer="MuSGD",
        cos_lr=True,
        close_mosaic=15,
        erasing=0.0,
        nbs=64,
        epochs=120,
        patience=20,
        deterministic=True,
        seed=0,
        save=True,
        plots=True,
        val=True,
        verbose=True,
    )

    # Mescla o base com os hp afiados (tuned_hp sobrepõe base_v7)
    train_kwargs = {**base_v7, **tuned_hp}
    model.train(**train_kwargs)
    
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

    print(f"Orquestração de Treino v9 para os modelos: {args.models}")
    print(f"  device       = {device}")
    print(f"  data         = {args.data}")
    print(f"  project      = {args.project}")
    print(f"  force re-run = {args.force}")

    summary: list[dict] = []
    failures: list[tuple[str, str]] = []
    t_total = time.perf_counter()

    for i, m in enumerate(args.models, 1):
        print(f"\n[{i}/{len(args.models)}] A processar modelo: {m}")
        try:
            stats = train_one_model(m, args, device)
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

    # Resumo final da orquestração
    print("\n" + "=" * 80)
    print("=== SUMÁRIO DO TREINO v9 (TODOS OS MODELOS)")
    print("=" * 80)
    for s in summary:
        if s["skipped"]:
            print(f"  {s['model']:<8} : ignorado ({s['reason']})")
        elif s["reason"] == "fail":
            print(f"  {s['model']:<8} : FALHOU")
        else:
            print(f"  {s['model']:<8} : sucesso  ({s['elapsed_min']:.1f} min)")

    total_min = (time.perf_counter() - t_total) / 60
    print(f"\nTempo total de processamento: {total_min:.1f} min ({total_min / 60:.2f} h)")

    if failures:
        print(f"\n[!] Encontrada(s) {len(failures)} falha(s):")
        for m, _ in failures:
            print(f"    - Modelo {m}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())