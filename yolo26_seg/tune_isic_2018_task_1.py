"""
tune_isic_2018_task_1.py — Hyperparameter Optimization (HPO) do YOLO26-seg
no ISIC 2018 Task 1 via algoritmo genético embutido no Ultralytics
(`model.tune()`).

Por que HPO e não NAS:
    * NAS = busca sobre arquitetura (depth/width multipliers, blocos).
    * HPO = busca sobre hiperparâmetros de treino (lr, momentum, augmentação).
    * O que vem sendo discutido (rodar fine-tuning N vezes com configs
      diferentes para encontrar a melhor) é HPO. NAS exigiria editar YAMLs
      de arquitetura e treinar do zero — não é o que você precisa.

Como funciona:
    1. Treina o modelo `--epochs` épocas (curtas) com hiperparâmetros
       atuais.
    2. Calcula fitness = `metrics/mAP50-95(M+B)` (ponderado pelo Ultralytics).
    3. Mantém top-K configs e gera novas via mutação gaussiana.
    4. Repete por `--iterations` rodadas.
    5. Salva `best_hyperparameters.yaml` no diretório de tune ao final.

Filosofia (mesma da v7):
    * O que NÃO se tuna fica explícito no `model.tune()` (mesma config v7).
    * O que SE tuna entra no `space=` com ranges narrow-ed para o ISIC
      (lr0, momentum, weight_decay, warmup, box/cls/dfl gains, todos os
      hiperparâmetros de augmentação).
    * Modelo selecionado via CLI (`--model`).

Uso típico:
    # Espaço wide (default — exploração inicial, 20 hp):
    python tune_isic_2018_task_1.py --model small

    # Espaço refined (follow-up — drops 5 hp sem sinal e narrows os de alto sinal,
    # baseado nas correlações Pearson do HPO inicial do small):
    python tune_isic_2018_task_1.py --model nano   --space refined --iterations 50
    python tune_isic_2018_task_1.py --model medium --space refined --iterations 50

    # Quantos trials/épocas:
    python tune_isic_2018_task_1.py --model small --iterations 50 --epochs 30

Em docker:
    docker run ... yolo26_ft \\
      python /workspace/yolo26_seg/tune_isic_2018_task_1.py --model small \\
      2>&1 | tee logs/yolo26_small_tune_isic_2018.log

Saídas (em /workspace/logs/tune_isic_2018_task_1_<model>/):
    * tune_results.csv         → tabela de cada iteração (fitness + hp)
    * best_hyperparameters.yaml → top-1 config (entrada do train_with_tuned_hp)
    * tune_scatter_plot.png    → scatter de hp vs fitness
    * tune_fitness.png         → curva da fitness ao longo das iterações
"""

import argparse
import time

from ultralytics import YOLO


# Mapa modelo -> arquivo de pesos. Editar aqui se mudar caminho/versão.
WEIGHTS = {
    "nano":   "/workspace/cache/yolo26n-seg.pt",
    "small":  "/workspace/cache/yolo26s-seg.pt",
    "medium": "/workspace/cache/yolo26m-seg.pt",
    "large":  "/workspace/cache/yolo26l-seg.pt",
    "xlarge": "/workspace/cache/yolo26x-seg.pt",
}

# ----------------------------------------------------------------------------
# SEARCH SPACES — duas variantes selecionáveis via `--space`
# ----------------------------------------------------------------------------
# Cada entry: (min, max) ou (min, max, gain). Gain controla amplitude da
# mutação gaussiana — mais gain = mais agressivo. Default = 1.0.
#
# (1) `wide` — exploração inicial. Ranges baseados em:
#       * default do Ultralytics (Tuner.space)
#       * o que aprendemos nas v2-v7 (lr0 1e-3 a 2e-3 funciona; mosaic alto
#         ok; erasing alto corrompe máscara; mixup útil em moderação)
#     Use quando ainda não há informação prévia sobre o dataset.
#
# (2) `refined` — follow-up. Construído a partir das correlações Pearson
#     hp×fitness do HPO inicial do small (30 trials, ver
#     analyze_hpo_results.ipynb):
#       * DROPS (|r| < 0.05 — não há sinal estatístico):
#           scale, degrees, box, fliplr, warmup_momentum
#         (esses 5 hp ficam fixos no default Ultralytics)
#       * NARROWS (alto sinal — concentra GA na região promissora):
#           mixup       r=-0.77  [0, 0.3]   → [0.0, 0.05]
#           copy_paste  r=-0.67  [0, 0.3]   → [0.0, 0.05]
#           lr0         r=-0.45  [5e-4,5e-3]→ [1e-3, 4e-3]
#           translate   r=+0.23  [0, 0.3]   → [0.05, 0.20]
#           hsv_h       r=+0.23  [0, 0.03]  → [0.005, 0.025]
#           flipud      r=+0.20  [0, 0.5]   → [0.0, 0.10]
#           weight_decay         [0, 1e-3]  → [1e-6, 1e-4]
#           dfl                  [0.8, 3.0] → [0.8, 1.5]
#           mosaic               [0.5, 1.0] → [0.7, 1.0]
#
#     Espaço refined tem 15 hp (5 a menos), ~25% mais budget efetivo do GA
#     por iteração + ranges menores → convergência mais rápida.
#
SEARCH_SPACE_WIDE = {
    # ---- Aprendizado ----
    "lr0":            (5e-4, 5e-3),        # narrow ao redor do que sabemos funcionar
    "lrf":            (0.005, 0.05),       # final/initial LR ratio
    "momentum":       (0.85, 0.95, 0.3),   # default 0.937
    "weight_decay":   (0.0, 0.001),
    "warmup_epochs":  (1.0, 5.0),
    "warmup_momentum": (0.5, 0.95),

    # ---- Pesos das losses ----
    "box":            (3.0, 12.0),         # default 7.5
    "cls":            (0.2, 1.5),          # default 0.5
    "dfl":            (0.8, 3.0),          # default 1.5

    # ---- Augmentação de cor ----
    "hsv_h":          (0.0, 0.03),         # default 0.015 — narrow
    "hsv_s":          (0.3, 0.9),          # default 0.7
    "hsv_v":          (0.2, 0.7),          # default 0.4

    # ---- Augmentação geométrica ----
    "degrees":        (0.0, 30.0),         # default 0; lesão dermo é rotacionalmente simétrica
    "translate":      (0.0, 0.3),          # default 0.1
    "scale":          (0.2, 0.7),          # default 0.5
    "fliplr":         (0.0, 0.6),          # default 0.5
    "flipud":         (0.0, 0.5),          # default 0; lesão dermo permite flip vertical

    # ---- Mixing augmentations ----
    "mosaic":         (0.5, 1.0),          # default 1.0
    "mixup":          (0.0, 0.3),          # default 0
    "copy_paste":     (0.0, 0.3),          # default 0
}


SEARCH_SPACE_REFINED = {
    # ---- Aprendizado (narrowed em torno do small winner lr0=2.33e-3) ----
    "lr0":            (1e-3, 4e-3),
    "lrf":            (0.005, 0.05),       # mantém range
    "momentum":       (0.85, 0.95, 0.3),
    "weight_decay":   (1e-6, 1e-4),        # narrow (small winner=1e-5; default 5e-4 ruim)
    "warmup_epochs":  (1.0, 5.0),
    # warmup_momentum: |r|<0.05 → fixo no default 0.8 (drop do search)

    # ---- Pesos das losses (drops box; narrows dfl) ----
    # box: |r|<0.05 → fixo no default 7.5 (drop)
    "cls":            (0.2, 1.5),
    "dfl":            (0.8, 1.5),           # narrow (small winner=1.12)

    # ---- Augmentação de cor (narrow hsv_h em torno do high-signal) ----
    "hsv_h":          (0.005, 0.025),       # narrow
    "hsv_s":          (0.3, 0.9),
    "hsv_v":          (0.2, 0.7),

    # ---- Augmentação geométrica (drops degrees, scale, fliplr) ----
    # degrees: |r|<0.05 → fixo no default 0 (drop)
    "translate":      (0.05, 0.20),         # narrow (small winner=0.12)
    # scale: |r|<0.05 → fixo no default 0.5 (drop)
    # fliplr: |r|<0.05 → fixo no default 0.5 (drop)
    "flipud":         (0.0, 0.10),          # narrow (small winner≈0)

    # ---- Mixing augmentations (narrow mixup, copy_paste) ----
    "mosaic":         (0.7, 1.0),           # narrow (small winner=1.0)
    "mixup":          (0.0, 0.05),          # narrow drástico (r=-0.77)
    "copy_paste":     (0.0, 0.05),          # narrow drástico (r=-0.67)
}

# Backward compat: nome antigo `SEARCH_SPACE` continua apontando para o wide.
SEARCH_SPACE = SEARCH_SPACE_WIDE


def get_search_space(name: str) -> dict:
    """Retorna o search space pelo nome (`wide` | `refined`)."""
    if name == "wide":
        return SEARCH_SPACE_WIDE
    if name == "refined":
        return SEARCH_SPACE_REFINED
    raise ValueError(f"--space inválido: {name!r}. Use 'wide' ou 'refined'.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="HPO do YOLO26-seg no ISIC 2018 Task 1 via model.tune().",
    )
    p.add_argument(
        "--model", choices=list(WEIGHTS), default="small",
        help="Tamanho do modelo YOLO26-seg (default: small).",
    )
    p.add_argument(
        "--iterations", type=int, default=30,
        help="Quantas configs distintas avaliar (default: 30).",
    )
    p.add_argument(
        "--epochs", type=int, default=30,
        help="Épocas por trial (default: 30 — curto por design para HPO).",
    )
    p.add_argument(
        "--patience", type=int, default=10,
        help="Early-stopping patience por trial (default: 10).",
    )
    p.add_argument(
        "--data",
        default="/workspace/datasets/isic_2018_task1_yolo26/data.yaml",
        help="Path do data.yaml. Para color spaces alternativos aponte para "
             "isic_2018_task1_yolo26_lab/data.yaml ou _hed.",
    )
    p.add_argument(
        "--device", default="0,1",
        help="GPUs (default: '0,1' para DDP). Use '0' para single GPU.",
    )
    p.add_argument(
        "--space", choices=["wide", "refined"], default="wide",
        help="Search space — 'wide' (20 hp, exploração inicial) ou 'refined' "
             "(15 hp, drops sem-sinal e narrows alto-sinal; recomendado para "
             "follow-up depois de uma rodada wide). Default: wide.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    start_time = time.perf_counter()

    # device: aceita "0,1" -> [0,1] ou "0" -> 0
    device = (
        [int(x) for x in args.device.split(",")]
        if "," in args.device else int(args.device)
    )

    model = YOLO(WEIGHTS[args.model])

    # ------------------------------------------------------------------------
    # Hiperparâmetros NÃO tunados — herdados do v7. Estes ficam fixos durante
    # toda a busca; só os que estão em SEARCH_SPACE são variados.
    # ------------------------------------------------------------------------
    fixed_v7 = dict(
        # Infra / dados
        data=args.data,
        project="/workspace/logs",
        name=f"tune_isic_2018_task_1_{args.model}",
        task="segment",
        pretrained=True,
        imgsz=640,

        # Hardware
        device=device,
        batch=32,                  # = 16/GPU em DDP (8.4.21 rejeita -1)
        workers=8,
        cache=False,

        # Otimização (fixa baseado em v7 — MuSGD foi a melhor)
        amp=True,
        optimizer="MuSGD",
        cos_lr=True,
        close_mosaic=15,
        erasing=0.0,               # erasing>0 corrompe máscara em ISIC
        nbs=64,

        # Early stopping curto p/ acelerar HPO
        epochs=args.epochs,
        patience=args.patience,
        deterministic=True,
        seed=0,
        save=True,
        plots=False,                # plots por trial = lento e gera muito disco
        val=True,
        verbose=False,              # output mais limpo entre trials
    )

    space = get_search_space(args.space)

    print("=" * 72)
    print(f"HPO YOLO26-seg ({args.model}) — {args.iterations} iter × {args.epochs} ep/trial")
    print(f"Search space: {args.space!r} ({len(space)} hiperparâmetros)")
    print("=" * 72)

    model.tune(
        space=space,
        iterations=args.iterations,
        use_ray=False,
        **fixed_v7,
    )

    tempo_total = time.perf_counter() - start_time
    print(f"\nHPO concluído ({args.model}).")
    print(f"Tempo total: {tempo_total / 60:.1f} min  ({tempo_total / 3600:.2f} h).")
    print(
        f"Resultados em: /workspace/logs/tune_isic_2018_task_1_{args.model}/\n"
        f"  best_hyperparameters.yaml  ← use no train_with_tuned_hp.py\n"
        f"  tune_results.csv\n"
        f"  tune_scatter_plot.png, tune_fitness.png"
    )


if __name__ == "__main__":
    main()
