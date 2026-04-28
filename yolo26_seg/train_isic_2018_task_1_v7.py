"""
train_isic_2018_task_1_v7.py — Fine-tuning YOLO26-seg no ISIC 2018 Task 1.

Versão **v7 (small, ablation de optimizer sobre v6)**.

Resultados small até agora:
  * v2 small: 0.7368 @ ep 47   (MuSGD,  amp=False, batch=-1, lr0=1e-3, mid 2.5e-3)
  * v3 small: 0.7099 @ ep 87   (compound change)
  * v4 small: 0.6484 @ ep 24   (augment ruidoso)
  * v5 small: 0.7021 @ ep 32   (minimalista, AdamW lr0=1e-3, mid 0.9e-3)
  * v6 small: 0.6999 @ ep 46   (= v5 com lr0=2e-3, mid 1.65e-3)

Hipótese falsificada em v6: LR não era o gargalo (dobrar lr0 não recuperou
o gap de 0.037 vs v2). Hipótese a testar em v7: o otimizador é o gargalo.

Filosofia desta v7 (refator):
  * Todos os hiperparâmetros listados explicitamente, com seu valor default
    do Ultralytics — fica fácil identificar/alterar qualquer um.
  * Linhas marcadas com `# v7` são as ÚNICAS que diferem do default. São o
    que estamos de fato testando.
  * Modelo selecionado via CLI (`--model {nano,small,medium,large,xlarge}`).
    Default = small (alvo da ablation atual).

Uso:
    python train_isic_2018_task_1_v7.py --model small
    python train_isic_2018_task_1_v7.py --model large
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
    p = argparse.ArgumentParser(description="Fine-tuning YOLO26-seg no ISIC 2018 Task 1 (v7).")
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
        data="/workspace/datasets/isic_2018_task1_yolo26_lab/data.yaml",
        project="/workspace/logs",
        name=f"yolo26_{args.model}_ft_isic_2018_v7",

        # =====================================================================
        # 2. HIPERPARÂMETROS DA EXECUÇÃO  (* = mudanças desta versão)
        #    Todos abaixo nos valores DEFAULT do Ultralytics, exceto onde
        #    marcado com `# v7`.
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
        device=[0, 1],              # default None — DDP em 2 GPUs neste ambiente
        batch=32,                   # default 16; v7 usa 32 (=16/GPU em DDP).
                                    #   Ultralytics 8.4.21 rejeita -1 em multi-GPU.
        workers=8,                  # default
        cache=False,                # default
        compile=False,              # default

        # ---- Estabilidade / Otimizador ----
        amp=True,                   # default
        optimizer="MuSGD",          # v7: default "auto" (vira AdamW p/ ds pequeno).
                                    #     ABLATION CHAVE — v2 (MuSGD) atingiu 0.7368;
                                    #     v3-v6 com AdamW estagnaram em ~0.70.
        nbs=64,                     # default

        # ---- Aprendizado ----
        lr0=0.002,                  # v7: default 0.01. Mantido do v6 p/ ablation
                                    #     limpa. ATENÇÃO: com MuSGD o lr/pg0 mid
                                    #     pode ir a ~5e-3 (alto). Se divergir,
                                    #     baixar para 0.001 em v7b.
        lrf=0.01,                   # default
        momentum=0.937,             # default
        weight_decay=0.0005,        # default
        warmup_epochs=3.0,          # default
        warmup_momentum=0.8,        # default
        warmup_bias_lr=0.1,         # default
        cos_lr=True,                # v7: default False. Cosine decay é melhor p/
                                    #     fine-tuning longo.
        patience=20,                # v7: default 100. Val tem 100 imgs; 20 já é
                                    #     folgado e economiza ~30% do tempo.

        # ---- Regularização / Reprodutibilidade ----
        dropout=0.0,                # default
        freeze=None,                # default
        seed=0,                     # default
        deterministic=True,         # default
        save=True,                  # default
        save_period=-1,             # default
        resume=False,               # default

        # ---- Pesos das losses ----
        box=7.5,                    # default
        cls=0.5,                    # default
        dfl=1.5,                    # default
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

        # ---- Data Augmentation ----
        degrees=0.0,                # default
        translate=0.1,              # default
        scale=0.5,                  # default
        shear=0.0,                  # default
        perspective=0.0,            # default
        flipud=0.0,                 # default
        fliplr=0.5,                 # default
        hsv_h=0.015,                # default
        hsv_s=0.7,                  # default
        hsv_v=0.4,                  # default
        bgr=0.0,                    # default
        mosaic=1.0,                 # default
        close_mosaic=15,            # v7: default 10. +5 ep finais sem mosaic
                                    #     refinam borda em imagens reais.
        mixup=0.0,                  # default
        cutmix=0.0,                 # default
        copy_paste=0.0,             # default
        copy_paste_mode="flip",     # default
        auto_augment="randaugment", # default
        erasing=0.0,                # v7: default 0.4. Random-erasing pode cair
                                    #     sobre a lesão e corromper a máscara.
    )

    tempo_total = time.perf_counter() - start_time
    print(f"\nTreinamento concluído com sucesso (v7 {args.model}).")
    print(f"Tempo total de execução: {tempo_total / 60:.2f} minutos.")
    return results


if __name__ == "__main__":
    main()
