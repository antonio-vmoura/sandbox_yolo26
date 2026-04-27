"""
train_isic_2018_task_1_v5_small.py — Fine-tuning YOLO26s-seg no ISIC 2018 Task 1.

Versão **v5 (small, minimalista)**.

Histórico que motivou esta versão:
  * v2 small: best mask mAP@50-95 = 0.7368  ← melhor resultado real até hoje.
  * v3 small: 0.7099  (−0.027 vs v2). Compound change: AdamW + AMP + batch=16
              + DDP + dropout/wd extras + auto_augment off + mask_ratio=2 +
              copy_paste=0.3 etc. — efeito líquido negativo, causa difícil de
              isolar.
  * v4 small: 0.6484 @ ep 24, depois oscila 0.55–0.65. Fix de LR funcionou
              (mid 1.9e-3 ≈ alvo 2.5e-3) mas o stack de augments ficou ruidoso
              demais para o `small` em LR alto.

Filosofia da v5:
  Voltar à baseline empírica (v2) e adicionar SÓ os 4 fixes que têm
  justificativa independente forte. Sem augment extra, sem regularização
  extra, sem mexer em mask_ratio. Se v5 bater v2 (≥0.7368), passamos a ter
  baseline + 4 fixes provados; daí qualquer mudança nova entra **uma por vez**.

Os 4 fixes mantidos da v3/v4:
  1. `amp=True`           — v2 (amp=False) era 2× mais lento e mais frágil.
  2. `optimizer="AdamW"`  — corrigiu NaN do xlarge da v2 (val/seg_loss ep1=629).
                            No `small` MuSGD funcionava; AdamW é equivalente
                            e dá consistência entre tamanhos.
  3. `close_mosaic=15`    — desliga mosaic nas últimas 15 ep para refinar
                            borda em imagens reais (relevante p/ mAP@50-95).
  4. `erasing=0.0`        — random-erasing default (0.4) corrompe máscara
                            quando o erase cai sobre a lesão.

Tudo o mais que existia em v2 é mantido como em v2 (degrees, flipud, fliplr,
hsv_*, translate, scale, mosaic=1.0, mixup=0.1). Tudo o que foi adicionado
em v3/v4 (`copy_paste`, `cutmix`, `auto_augment`, `dropout`, `weight_decay`
extra, `mask_ratio=2`) volta para o **default do Ultralytics**.

DDP / batch:
  Em DDP 2 GPUs, `batch=-1` (autobatch) é rejeitado pelo Ultralytics 8.4.21.
  v2 rodava `batch=-1` em single-GPU (`device=-1`). Para reproduzir em DDP,
  fixo `batch=32` (=16/GPU) e `lr0=1e-3` (mesmo valor de v2). Isso dá
  `lr/pg0` mid ≈ 1.4–1.6e-3 com AdamW — um pouco abaixo dos 2.5e-3 da v2,
  mas o objetivo aqui é estabilidade, não velocidade de convergência.
"""

import time
from ultralytics import YOLO


def main():
    start_time = time.perf_counter()

    model = YOLO("/workspace/cache/yolo26l-seg.pt")

    results = model.train(
        # =====================================================================
        # 1. DADOS E INFRAESTRUTURA  (idêntico v2, exceto epochs/name)
        # =====================================================================
        data="/workspace/datasets/isic_2018_task1_yolo26/data.yaml",
        project="/workspace/logs",
        name="yolo26_large_ft_isic_2018_v5",
        epochs=120,             # v2 usava 100; +20 para cobrir patience folgada.
        imgsz=640,              # Mantido (pedido explícito do usuário).
        device=[0, 1],          # DDP em 2 GPUs (mesmo setup da v3/v4).
        batch=32,               # =16/GPU. Em DDP, Ultralytics 8.4.21 rejeita -1.
                                #   v2 rodava com batch=-1 em single-GPU.
        workers=4,              # v2 usava 4. Sem motivo para mudar agora.

        # =====================================================================
        # 2. ESTABILIDADE / OTIMIZADOR  (1º e 2º fix vs v2)
        # =====================================================================
        amp=True,               # Fix #1. v2: False (lento, frágil em xlarge).
        optimizer="AdamW",      # Fix #2. v2: "MuSGD" (instável em xlarge,
                                #   ok em small). AdamW dá consistência entre
                                #   tamanhos.

        # =====================================================================
        # 3. APRENDIZADO  (idêntico v2)
        # =====================================================================
        lr0=0.001,              # Igual v2. Em DDP+batch=32 dá lr/pg0 mid ~1.5e-3.
        lrf=0.01,               # Igual v2.
        momentum=0.937,         # Igual v2.
        weight_decay=0.0005,    # Igual v2 (default Ultralytics). NÃO subir.
        warmup_epochs=3.0,      # Igual v2 (não 5.0 da v3).
        warmup_momentum=0.8,    # Igual v2.
        cos_lr=True,            # Igual v2.
        patience=20,            # v2 usava 25; reduzido p/ economizar tempo.

        # =====================================================================
        # 4. DATA AUGMENTATION  (idêntico v2)
        # =====================================================================
        degrees=180.0,
        flipud=0.5,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,              # Igual v2 (sim, 0.7 — não baixar).
        hsv_v=0.4,
        translate=0.1,
        scale=0.5,
        mosaic=1.0,
        mixup=0.1,

        # =====================================================================
        # 5. SEGMENTAÇÃO + 3º e 4º fix
        # =====================================================================
        close_mosaic=15,        # Fix #3. Default Ultralytics é 10; 15 dá um
                                #   pouco mais de refino em imagens reais.
        erasing=0.0,            # Fix #4. Default 0.4 corrompe máscara.

        # NOTAS importantes — o que NÃO entra nesta versão (e por quê):
        #   * mask_ratio  → default Ultralytics (4). v3 tentou 2 no small e
        #                   piorou: head do small não tem capacidade.
        #   * copy_paste  → default 0. v4 com 0.3 oscilou.
        #   * cutmix      → default 0. Corta máscara, indesejado em seg.
        #   * auto_augment → default. v2 deixou default (= "randaugment" no
        #                    Ultralytics atual) e foi o melhor run.
        #   * dropout     → default 0. v3 com 0.1 prejudicou small.
        #   * weight_decay extra (1e-3) → v3 prejudicou small.
        #   * shear/perspective/bgr/multi_scale/cls_pw → default.
    )

    tempo_total = time.perf_counter() - start_time
    print("\nTreinamento concluído com sucesso (v5 small).")
    print(f"Tempo total de execução: {tempo_total / 60:.2f} minutos.")
    return results


if __name__ == "__main__":
    main()
