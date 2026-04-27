"""
train_isic_2018_task_1_v7.py — Fine-tuning YOLO26s-seg no ISIC 2018 Task 1.

Versão **v7 (small, ablation de optimizer sobre v6)**.

Resultados small até agora:
  * v2 small: 0.7368 @ ep 47   (MuSGD,  amp=False, batch=-1, lr0=1e-3, mid 2.5e-3)
  * v3 small: 0.7099 @ ep 87   (compound change)
  * v4 small: 0.6484 @ ep 24   (augment ruidoso)
  * v5 small: 0.7021 @ ep 32   (minimalista, AdamW lr0=1e-3, mid 0.9e-3)
  * v6 small: 0.6999 @ ep 46   (= v5 com lr0=2e-3, mid 1.65e-3)

Hipótese falsificada em v6:
  Dobrar `lr0` em v6 levou `lr/pg0` mid de 0.9e-3 → 1.65e-3 ✓ (LR fix funcionou)
  MAS best mAP@50-95 não mudou (0.7021 → 0.6999, Δ=−0.002, ruído). LR não era
  o gargalo. Variância até piorou (std ep20+ 0.036 → 0.062).

Hipótese a testar em v7:
  O gargalo está no **otimizador**. v2 usava `MuSGD` (Muon updates ortogonais
  + SGD), v3-v6 usam `AdamW`. Em datasets pequenos, Muon costuma generalizar
  melhor (gradientes ortogonais reduzem co-adaptação de filtros).

Mudança em v7 (única variável vs v6):
  `optimizer`: "AdamW" → "MuSGD".

Observação importante sobre LR:
  Mantemos `lr0=0.002` (=v6, ablation limpa), mas o Ultralytics aplica
  scaling diferente por otimizador. v2 com MuSGD+lr0=0.001 deu `lr/pg0` mid
  ≈ 2.5e-3 (fator ~2.5×). Em v7 com MuSGD+lr0=0.002 esperamos mid ≈ 5e-3 —
  ALTO. Pode divergir / oscilar mais que v2.

  Se v7 divergir nas primeiras épocas (val/seg_loss > 50 na ep 1), aborta e
  rodamos v7b com `lr0=0.001` (mesmo regime efetivo da v2 best).

Atenção: `MuSGD` + `amp=True` foi exatamente o combo que deu NaN no xlarge
da v2 (val/seg_loss ep1 = 629). No `small` v2 não teve esse problema, mas
amp=False era o setup. Em v7 mantemos amp=True (era um dos 4 fixes de v5).
Se houver NaN, próximo passo é v7c com `amp=False`.
"""

import time
from ultralytics import YOLO


def main():
    start_time = time.perf_counter()

    model = YOLO("/workspace/cache/yolo26s-seg.pt")

    results = model.train(
        # =====================================================================
        # 1. DADOS E INFRAESTRUTURA  (igual v6)
        # =====================================================================
        data="/workspace/datasets/isic_2018_task1_yolo26/data.yaml",
        project="/workspace/logs",
        name="yolo26_small_ft_isic_2018_v7",
        epochs=120,
        imgsz=640,
        device=[0, 1],          # DDP em 2 GPUs.
        batch=32,               # =16/GPU. Ultralytics 8.4.21 rejeita -1 em DDP.
        workers=4,

        # =====================================================================
        # 2. ESTABILIDADE / OTIMIZADOR  (ÚNICA mudança vs v6)
        # =====================================================================
        amp=True,
        optimizer="MuSGD",      # v7: "AdamW" → "MuSGD". Mesmo otimizador da v2.

        # =====================================================================
        # 3. APRENDIZADO  (igual v6)
        # =====================================================================
        lr0=0.002,              # Mantido para isolar variável "optimizer".
                                #   ATENÇÃO: com MuSGD esperamos lr/pg0 mid ≈ 5e-3
                                #   (alto). Se divergir, baixar para 0.001 em v7b.
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        cos_lr=True,
        patience=20,

        # =====================================================================
        # 4. DATA AUGMENTATION  (igual v6 / v5 / v2)
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
        # 5. SEGMENTAÇÃO + fixes mantidos (igual v6 / v5)
        # =====================================================================
        close_mosaic=15,
        erasing=0.0,
    )

    tempo_total = time.perf_counter() - start_time
    print("\nTreinamento concluído com sucesso (v7 small).")
    print(f"Tempo total de execução: {tempo_total / 60:.2f} minutos.")
    return results


if __name__ == "__main__":
    main()
