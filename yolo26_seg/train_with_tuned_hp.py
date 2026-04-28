"""
train_with_tuned_hp.py — Treino completo (120 ep) com os hiperparâmetros
campeões saídos do `tune_isic_2018_task_1.py`.

Fluxo:
    1. `tune_isic_2018_task_1.py --model small`
       → escreve `logs/tune_isic_2018_task_1_small/best_hyperparameters.yaml`
    2. Este script:
       carrega esse yaml, mescla com a config v7 (fixos) e roda treino
       full-length (`epochs=120`, `patience=20`).
    3. Logs em `logs/yolo26_<model>_ft_isic_2018_v7_tuned/` ficam
       comparáveis com os runs v2-v7 nos notebooks de comparação.

Por que não embutir tudo no `tune.py`:
    HPO usa trials curtos (30 ep) por velocidade. O melhor hp em trial
    curto não necessariamente é o melhor em treino longo (ex: lr0 muito
    alto pode parecer ótimo em 30 ep e divergir em 100). Vale rodar o
    treino completo para confirmar e produzir os checkpoints definitivos.

Uso:
    python train_with_tuned_hp.py --model small
    python train_with_tuned_hp.py --model small \\
        --hp /workspace/logs/tune_isic_2018_task_1_small/best_hyperparameters.yaml

Em docker:
    docker run ... yolo26_ft \\
      python /workspace/yolo26_seg/train_with_tuned_hp.py --model small \\
      2>&1 | tee logs/yolo26_small_ft_isic_2018_v7_tuned.log
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
        description="Treino completo do YOLO26-seg no ISIC 2018 com hp tunados.",
    )
    p.add_argument(
        "--model", choices=list(WEIGHTS), default="small",
        help="Tamanho do modelo YOLO26-seg (default: small).",
    )
    p.add_argument(
        "--hp", default=None,
        help="Caminho para best_hyperparameters.yaml. Default: "
             "/workspace/logs/tune_isic_2018_task_1_<model>/best_hyperparameters.yaml.",
    )
    p.add_argument(
        "--data",
        default="/workspace/datasets/isic_2018_task1_yolo26/data.yaml",
        help="Path do data.yaml (mantenha consistente com o usado no tune).",
    )
    p.add_argument(
        "--device", default="0,1",
        help="GPUs (default: '0,1' para DDP). Use '0' para single GPU.",
    )
    p.add_argument(
        "--name-suffix", default="tuned",
        help="Sufixo do diretório de log (default: 'tuned' → "
             "yolo26_<model>_ft_isic_2018_v7_tuned/).",
    )
    return p.parse_args()


def load_tuned_hp(path: Path) -> dict:
    """Carrega best_hyperparameters.yaml gerado pelo Ultralytics Tuner.

    O Ultralytics escreve um YAML com cabeçalho de comentário ('# Tuner: ...')
    seguido do dict de hp. `yaml.safe_load` ignora os comentários.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo de hp tunados não encontrado: {path}\n"
            f"Rode primeiro `tune_isic_2018_task_1.py --model <size>`."
        )
    with path.open("r") as f:
        data = yaml.safe_load(f) or {}
    if not data:
        raise ValueError(
            f"YAML vazio em {path}. Pode ter falhado a tuning anterior — "
            f"verifique tune_results.csv no mesmo diretório."
        )
    return data


def main() -> None:
    args = parse_args()
    start_time = time.perf_counter()

    # Path do YAML de hp tunados
    hp_path = Path(
        args.hp or f"/workspace/logs/tune_isic_2018_task_1_{args.model}/best_hyperparameters.yaml"
    )
    tuned_hp = load_tuned_hp(hp_path)

    print("=" * 72)
    print(f"Treino full-length ({args.model}) com hp tunados de {hp_path}")
    print("Hiperparâmetros tunados:")
    for k, v in sorted(tuned_hp.items()):
        print(f"  {k:18s} = {v}")
    print("=" * 72)

    # device: aceita "0,1" -> [0,1] ou "0" -> 0
    device = (
        [int(x) for x in args.device.split(",")]
        if "," in args.device else int(args.device)
    )

    model = YOLO(WEIGHTS[args.model])

    # Config v7 fixa (epochs/patience/optimizer/etc) — mesma da v7,
    # com hp tunados sobrescrevendo o que o tuner cobriu.
    base_v7 = dict(
        # Infra
        data=args.data,
        project="/workspace/logs",
        name=f"yolo26_{args.model}_ft_isic_2018_v7_{args.name_suffix}",
        task="segment",
        pretrained=True,
        imgsz=640,

        # Hardware
        device=device,
        batch=32,
        workers=8,
        cache=False,

        # Otimização (fixa)
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

    # Hp tunados sobrescrevem qualquer chave que tenham em comum.
    train_kwargs = {**base_v7, **tuned_hp}

    model.train(**train_kwargs)

    tempo_total = time.perf_counter() - start_time
    print(f"\nTreino tunado concluído ({args.model}).")
    print(f"Tempo total: {tempo_total / 60:.1f} min.")


if __name__ == "__main__":
    main()
