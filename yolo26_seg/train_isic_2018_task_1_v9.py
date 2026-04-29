"""
train_isic_2018_task_1_v9.py — Fine-tuning YOLO26-seg no ISIC 2018 Task 1.

Versão **v9 — Per-size HPO-tuned**.

Diferença vs v8:
    v8 — Aplicou os hp campeões do small a todos os 5 tamanhos como
         aproximação pragmática (assumindo transferibilidade).
    v9 — Cada tamanho usa SEU PRÓPRIO hp campeão, encontrado via
         `tune_all_models.py --space refined --iterations 50` (HPO
         independente por tamanho com search space refinado).

Origem dos hp por tamanho (lidos em runtime):
    /workspace/logs/tune_isic_2018_task_1_<model>/best_hyperparameters.yaml

    Cada YAML é gerado por uma rodada independente do GA do Ultralytics
    (`model.tune()`) com:
      * `--space refined` (15 hp — drops sem-sinal e narrows alto-sinal,
                           construído a partir das correlações do small).
      * `iterations=50` (vs 30 do small wide).

Mantida da v7 (não tunada — fica fixa para garantir comparabilidade entre
tamanhos):
    optimizer="MuSGD", amp=True, cos_lr=True, close_mosaic=15,
    erasing=0.0, batch=32, nbs=64, imgsz=640,
    epochs=120, patience=20

Uso:
    # Treino de UM tamanho específico:
    python train_isic_2018_task_1_v9.py --model small
    python train_isic_2018_task_1_v9.py --model medium
    ...

    # Para treinar TODOS os 5 tamanhos sequencialmente sem ficar
    # supervisionando, use o orquestrador:
    python train_all_v9_models.py

Em docker:
    docker run ... yolo26_ft \\
      python /workspace/yolo26_seg/train_isic_2018_task_1_v9.py --model <size> \\
      2>&1 | tee logs/yolo26_<size>_ft_isic_2018_v9.log

Saída:
    /workspace/logs/yolo26_<size>_ft_isic_2018_v9/
      ├── weights/best.pt
      ├── results.csv
      ├── args.yaml
      └── ... (curvas, plots, val/)
"""

import argparse
import time
from pathlib import Path

import yaml
from ultralytics import YOLO


# Mapa modelo -> arquivo de pesos. Editar aqui se mudar caminho/versão.
WEIGHTS = {
    "nano":   "/workspace/cache/yolo26n-seg.pt",
    "small":  "/workspace/cache/yolo26s-seg.pt",
    "medium": "/workspace/cache/yolo26m-seg.pt",
    "large":  "/workspace/cache/yolo26l-seg.pt",
    "xlarge": "/workspace/cache/yolo26x-seg.pt",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tuning YOLO26-seg no ISIC 2018 Task 1 "
                    "(v9 — per-size HPO-tuned).",
    )
    p.add_argument(
        "--model", choices=list(WEIGHTS), default="small",
        help="Tamanho do modelo YOLO26-seg (default: small).",
    )
    p.add_argument(
        "--hp", default=None,
        help="Caminho do best_hyperparameters.yaml (default: "
             "/workspace/logs/tune_isic_2018_task_1_<model>/best_hyperparameters.yaml).",
    )
    p.add_argument(
        "--data",
        default="/workspace/datasets/isic_2018_task1_yolo26/data.yaml",
        help="Path do data.yaml (manter o mesmo usado no HPO).",
    )
    p.add_argument(
        "--device", default="0,1",
        help="GPUs (default: '0,1' para DDP). Use '0' para single GPU.",
    )
    p.add_argument(
        "--project", default="/workspace/logs",
        help="Diretório raiz dos logs (default: /workspace/logs).",
    )
    return p.parse_args()


def load_tuned_hp(path: Path) -> dict:
    """Carrega `best_hyperparameters.yaml` gerado pelo Ultralytics Tuner.

    O Ultralytics escreve um YAML com cabeçalho de comentário ('# Tuner: ...')
    seguido do dict de hp. `yaml.safe_load` ignora os comentários.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo de hp tunados não encontrado: {path}\n"
            f"Rode primeiro `tune_isic_2018_task_1.py --model <size>` "
            f"(ou `tune_all_models.py --space refined`)."
        )
    with path.open("r") as f:
        data = yaml.safe_load(f) or {}
    if not data:
        raise ValueError(
            f"YAML vazio em {path}. Pode ter falhado a tuning anterior — "
            f"verifique tune_results.csv no mesmo diretório."
        )
    return data


def train_v9_one_size(
    model_size: str,
    hp_yaml: Path,
    data: str,
    project: str,
    device,
) -> None:
    """Roda o treino v9 (120 ep, patience 20) para um tamanho.

    Carrega `best_hyperparameters.yaml` do `tune_isic_2018_task_1_<size>/`
    e mescla com a config v7 fixa.
    """
    tuned_hp = load_tuned_hp(hp_yaml)

    print("=" * 72)
    print(f"v9 train ({model_size}) — hp de {hp_yaml}")
    print("Hiperparâmetros tunados:")
    for k, v in sorted(tuned_hp.items()):
        print(f"  {k:18s} = {v}")
    print("=" * 72)

    model = YOLO(WEIGHTS[model_size])

    base_v7 = dict(
        # Infra
        data=data,
        project=project,
        name=f"yolo26_{model_size}_ft_isic_2018_v9",
        task="segment",
        pretrained=True,
        imgsz=640,

        # Hardware
        device=device,
        batch=32,
        workers=8,
        cache=False,

        # Otimização (fixa da v7 — não tunada para garantir comparabilidade)
        amp=True,
        optimizer="MuSGD",
        cos_lr=True,
        close_mosaic=15,
        erasing=0.0,
        nbs=64,

        # Treino completo
        epochs=120,
        patience=20,
        deterministic=True,
        seed=0,
        save=True,
        plots=True,
        val=True,
        verbose=True,
    )

    # Hp tunados sobrescrevem qualquer chave que tenham em comum com o base.
    train_kwargs = {**base_v7, **tuned_hp}
    model.train(**train_kwargs)


def main() -> None:
    args = parse_args()
    start_time = time.perf_counter()

    hp_path = Path(
        args.hp or
        f"{args.project}/tune_isic_2018_task_1_{args.model}/best_hyperparameters.yaml"
    )

    device = (
        [int(x) for x in args.device.split(",")]
        if "," in args.device else int(args.device)
    )

    train_v9_one_size(
        model_size=args.model,
        hp_yaml=hp_path,
        data=args.data,
        project=args.project,
        device=device,
    )

    elapsed = (time.perf_counter() - start_time) / 60
    print(f"\nTreino v9 concluído ({args.model}). Tempo total: {elapsed:.1f} min.")


if __name__ == "__main__":
    main()
