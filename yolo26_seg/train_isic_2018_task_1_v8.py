"""
train_isic_2018_task_1_v8.py — Fine-tuning YOLO26-seg no ISIC 2018 Task 1.

Versão **v8 — HPO-tuned (best of 30 trials × 30 ep)**.

Origem: rodada de HPO via algoritmo genético do Ultralytics (`model.tune()`)
com 30 iterações × 30 épocas/trial sobre o `small`. O trial 9 (best) atingiu
fitness 1.4345 e foi salvo em
`logs/tune_isic_2018_task_1_small/best_hyperparameters.yaml`. O treino
full-length de validação (120 ep, patience 20) confirmou o ganho:

  small mAP50-95(M) tuned   = 0.7315 @ ep 9   (early-stop em ep 29)
  small mAP50-95(M) v2-rerun = 0.7169         (baseline reproduzível)
  Δ = +0.0146 mAP50-95(M)  (+1.5 pts), em ¼ do tempo de treino

CAVEAT IMPORTANTE: o HPO foi feito apenas no small. Aqui aplicamos os hp
campeões aos 5 tamanhos como aproximação pragmática — isso assume que a
solução é razoavelmente transferível entre tamanhos, hipótese consistente
com o padrão observado nas v2-v7 (mesmos hp funcionam similarmente em todas
as escalas). Se for tunar individualmente cada tamanho mais tarde, basta
rodar `tune_all_models.py --models medium large xlarge` e usar
`train_with_tuned_hp.py --model <size>`.

Diferenças vs v7 (todas as linhas marcadas com `# v8`):

  Aprendizado (mais conservador que v7):
    lr0           0.002    -> 0.00233    (HPO best)
    lrf           0.01     -> 0.01299    (HPO best)
    momentum      0.937    -> 0.93525    (~ default)
    weight_decay  0.0005   -> 1.0e-05    (50× MENOR — descoberta-chave)
    warmup_epochs 3.0      -> 3.31701
    warmup_momentum 0.8    -> 0.82509

  Pesos das losses:
    box           7.5      -> 8.18619    (+9%)
    cls           0.5      -> 0.58318
    dfl           1.5      -> 1.11637    (-26% — também descoberta-chave)

  Augmentação (HPO encontrou que mixup/copy_paste HURTAM e flipud não ajuda):
    hsv_h         0.015    -> 0.00853
    hsv_s         0.7      -> 0.56217
    hsv_v         0.4      -> 0.36385
    degrees       0.0      -> 0.00069
    translate     0.1      -> 0.11998
    scale         0.5      -> 0.53418
    fliplr        0.5      -> 0.6
    flipud        0.0      -> 0.00391
    mosaic        1.0      -> 1.0         (default mantido)
    mixup         0.0      -> 0.0         (HPO confirmou: r=-0.77 com fitness)
    copy_paste    0.0      -> 0.00142     (HPO confirmou: r=-0.67 com fitness)

  Mantidas da v7 (não foram tunadas):
    optimizer="MuSGD", amp=True, cos_lr=True, close_mosaic=15, erasing=0.0,
    batch=32, nbs=64, imgsz=640, epochs=120, patience=20

Uso:
    python train_isic_2018_task_1_v8.py --model small
    python train_isic_2018_task_1_v8.py --model medium
    ...

Em docker:
    docker run ... yolo26_ft \\
      python /workspace/yolo26_seg/train_isic_2018_task_1_v8.py --model <size> \\
      2>&1 | tee logs/yolo26_<size>_ft_isic_2018_v8.log
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


def parse_args():
    p = argparse.ArgumentParser(
        description="Fine-tuning YOLO26-seg no ISIC 2018 Task 1 (v8 — HPO-tuned)."
    )
    p.add_argument(
        "--model", choices=list(WEIGHTS), default="small",
        help="Tamanho do modelo YOLO26-seg (default: small).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    start_time = time.perf_counter()

    model = YOLO(WEIGHTS[args.model])

    results = model.train(
        # =====================================================================
        # 1. INFRAESTRUTURA — caminhos e identificação do experimento
        # =====================================================================
        data="/workspace/datasets/isic_2018_task1_yolo26/data.yaml",
        project="/workspace/logs",
        name=f"yolo26_{args.model}_ft_isic_2018_v8",

        # =====================================================================
        # 2. HIPERPARÂMETROS DA EXECUÇÃO
        #    Linhas marcadas com `# v8` foram ajustadas pelo HPO.
        #    O resto preserva a config v7 (que já era o melhor stack manual).
        # =====================================================================
        # ---- Dados / loop de treino ----
        exist_ok=False,             # default
        task="segment",             # default p/ um -seg.pt; explicito p/ robustez
        pretrained=True,            # default
        epochs=120,                 # v7: default 100, +20 p/ folga com cos_lr
        time=None,                  # default
        imgsz=640,                  # default (mantido a pedido)
        rect=False,                 # default
        multi_scale=0.0,            # default
        fraction=1.0,               # default
        single_cls=False,           # default

        # ---- Hardware ----
        device=[0, 1],              # DDP em 2 GPUs
        batch=32,                   # =16/GPU em DDP. Ultralytics 8.4.21 rejeita -1 em multi-GPU.
        workers=8,                  # default
        cache=False,                # default
        compile=False,              # default

        # ---- Estabilidade / Otimizador ----
        amp=True,                   # default
        optimizer="MuSGD",          # v7: vencedor da ablation v3-v7
        nbs=64,                     # default

        # ---- Aprendizado (HPO-tuned em v8) ----
        lr0=0.00233,                # v8: HPO best (era 0.002 em v7)
                                    #   correlação Pearson com fitness: r=-0.45
                                    #   (lr0 maior piora — confirma direção)
        lrf=0.01299,                # v8: HPO best (era 0.01)
        momentum=0.93525,           # v8: ~ default 0.937
        weight_decay=1.0e-05,       # v8: HPO best (era 0.0005 — 50× MENOR)
                                    #   regularização L2 muito mais leve;
                                    #   evita over-regularizar fine-tuning curto
        warmup_epochs=3.31701,      # v8: HPO best (era 3.0)
        warmup_momentum=0.82509,    # v8: HPO best (era 0.8)
        warmup_bias_lr=0.1,         # default
        cos_lr=True,                # v7: cosine decay
        patience=20,                # v7: 20 ep s/ melhora → para. ATENÇÃO:
                                    #   no run v8 small o best foi @ ep 9,
                                    #   patience disparou em ep 29 — esperado.

        # ---- Regularização / Reprodutibilidade ----
        dropout=0.0,                # default
        freeze=None,                # default
        seed=0,                     # default
        deterministic=True,         # default
        save=True,                  # default
        save_period=-1,             # default
        resume=False,               # default

        # ---- Pesos das losses (HPO-tuned em v8) ----
        box=8.18619,                # v8: HPO best (era 7.5; +9%)
        cls=0.58318,                # v8: HPO best (era 0.5)
        dfl=1.11637,                # v8: HPO best (era 1.5; -26%)
        pose=12.0,                  # default (irrelevante p/ seg)
        kobj=1.0,                   # default (irrelevante p/ seg)

        # ---- Segmentação ----
        overlap_mask=True,          # default
        mask_ratio=4,               # default
        retina_masks=False,         # default

        # ---- Validação ----
        val=True,                   # default
        split="val",                # default
        plots=True,                 # default
        verbose=True,               # default
        iou=0.7,                    # default
        max_det=300,                # default
        half=False,                 # default
        save_json=False,            # default

        # ---- Data Augmentation (HPO-tuned em v8) ----
        degrees=0.00069,            # v8: ~ default 0.0
        translate=0.11998,          # v8: HPO best (era 0.1)
        scale=0.53418,              # v8: HPO best (era 0.5)
        shear=0.0,                  # default (não tunado)
        perspective=0.0,            # default (não tunado)
        flipud=0.00391,             # v8: ~ default 0.0
        fliplr=0.6,                 # v8: HPO best (era 0.5)
        hsv_h=0.00853,              # v8: HPO best (era 0.015)
        hsv_s=0.56217,              # v8: HPO best (era 0.7)
        hsv_v=0.36385,              # v8: HPO best (era 0.4)
        bgr=0.0,                    # default (não tunado)
        mosaic=1.0,                 # default (mantido)
        close_mosaic=15,            # v7: 15 ep finais sem mosaic
        mixup=0.0,                  # v8 confirmou: r=-0.77 com fitness (!) —
                                    #   mixup ATRAPALHA neste ds. Mantido em 0.
        cutmix=0.0,                 # default (não tunado)
        copy_paste=0.00142,         # v8 confirmou: r=-0.67 com fitness — idem.
        copy_paste_mode="flip",     # default
        auto_augment="randaugment", # default
        erasing=0.0,                # v7: random-erasing corrompe máscara
    )

    tempo_total = time.perf_counter() - start_time
    print(f"\nTreinamento concluído com sucesso (v8 {args.model}).")
    print(f"Tempo total de execução: {tempo_total / 60:.2f} minutos.")
    return results


if __name__ == "__main__":
    main()
