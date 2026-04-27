"""
train_isic_2018_task_1_v6.py — Fine-tuning YOLO26s-seg no ISIC 2018 Task 1.

Versão **v6 (small, ablation de LR sobre v5)**.

Resultados small até agora:
  * v2 small: best mask mAP@50-95 = 0.7368  (MuSGD, batch=-1, single-GPU,
              lr/pg0 mid ≈ 2.5e-3)         ← baseline a bater
  * v3 small: 0.7099 (compound change; LR efetivo ~7e-4)
  * v4 small: 0.6484 (LR ok ~1.9e-3 mas augment stack ruidoso)
  * v5 small: 0.7021 (minimalista, mas LR efetivo ainda ~9e-4 com AdamW+DDP)

Diagnóstico que motivou v6:
  v5 manteve `lr0=1e-3` (igual v2), mas com AdamW+batch=32+DDP o `lr/pg0` mid
  ficou em ~9e-4 — **2.8× abaixo** do regime efetivo da v2 (~2.5e-3). Esse
  underfitting persistente explica a oscilação 0.55–0.70 e o teto em 0.7021.

Mudança em v6 (única variável vs v5):
  `lr0`: 1e-3 → **2e-3**.

Tudo o resto é IDÊNTICO ao v5 small. Se v6 atingir lr/pg0 mid ≈ 1.8e-3 e best
mask mAP@50-95 ≥ 0.7368, fica empiricamente confirmado que **o gap v3/v4/v5
era 100% LR scaling** (AdamW vs MuSGD no Ultralytics).
"""

import time
from ultralytics import YOLO


def main():
    start_time = time.perf_counter()

    model = YOLO("/workspace/cache/yolo26s-seg.pt")

    results = model.train(
        # =====================================================================
        # 1. DADOS E INFRAESTRUTURA  (igual v5)
        # =====================================================================
        data="/workspace/datasets/isic_2018_task1_yolo26/data.yaml",
        project="/workspace/logs",
        name="yolo26_small_ft_isic_2018_v6",
        epochs=120,
        imgsz=640,              # Mantido (pedido explícito do usuário).
        device=[0, 1],          # DDP em 2 GPUs.
        batch=32,               # =16/GPU. Ultralytics 8.4.21 rejeita -1 em DDP.
        workers=4,

        # =====================================================================
        # 2. ESTABILIDADE / OTIMIZADOR  (igual v5)
        # =====================================================================
        amp=True,
        optimizer="AdamW",

        # =====================================================================
        # 3. APRENDIZADO  (ÚNICA mudança vs v5: lr0 1e-3 → 2e-3)
        # =====================================================================
        lr0=0.002,              # v6: 1e-3 → 2e-3. Em v5 o `lr/pg0` mid ficou
                                #   em ~9e-4 (vs 2.5e-3 da v2). Dobrar lr0 deve
                                #   levar mid para ~1.8e-3, mais próximo de v2.
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        cos_lr=True,
        patience=20,

        # =====================================================================
        # 4. DATA AUGMENTATION  (igual v5 / v2)
        # =====================================================================
        degrees=180.0,
        flipud=0.5,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        translate=0.1,
        scale=0.5,
        mosaic=1.0,
        mixup=0.1,

        # =====================================================================
        # 5. SEGMENTAÇÃO + fixes mantidos (igual v5)
        # =====================================================================
        close_mosaic=15,
        erasing=0.0,
    )

    tempo_total = time.perf_counter() - start_time
    print("\nTreinamento concluído com sucesso (v6 small).")
    print(f"Tempo total de execução: {tempo_total / 60:.2f} minutos.")
    return results


if __name__ == "__main__":
    main()
