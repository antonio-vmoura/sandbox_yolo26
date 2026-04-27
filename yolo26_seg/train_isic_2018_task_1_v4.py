"""
train_isic_2018_task_1_v4.py — Fine-tuning YOLO26-seg no ISIC 2018 Task 1.

Versão **v4**: re-tune da v3 endereçando a regressão observada (best mask
mAP@50-95 caiu vs v2 em 4 de 5 modelos).

Diagnóstico que motivou esta versão (vs v3):
  * Em v3 o LR efetivo (`lr/pg0` mid) ficou ~3.5× MENOR que em v2 (5e-4 vs 2e-3).
  * Causas: troca para AdamW + batch fixo em 16 + DDP em 2 GPUs (effective
    batch=8/GPU) + `nbs=64` mantido — composição que escala LR para baixo.
  * Resultado: underfitting silencioso em todos os tamanhos de modelo.

Mudanças-chave em v4 (vs v3):
  * `lr0`: 1e-3 -> 2e-3 (compensa batch menor + DDP).
  * `batch`: 16 -> 32 (16/GPU em DDP; Ultralytics 8.4.21 rejeita batch=-1
    em multi-GPU). 32 dobra o batch efetivo da v3 e leva `lr/pg0` mid de
    ~7e-4 para ~2.8e-3, alinhado com a v2 (~2.5e-3).
  * `patience`: 30 -> 20 (em v3 todos os runs tinham 30+ ep depois do best).
  * `epochs`: 150 -> 120 (orçamento mais que suficiente com LR adequado).
  * Hiperparâmetros agora **derivados do tamanho do modelo** (`MODEL_SIZE`):
      - nano/small  : `mask_ratio=4`, `dropout=0.0`, `weight_decay=5e-4`,
                      `auto_augment="randaugment"` (devolve augment útil)
      - medium/large: `mask_ratio=4`, `dropout=0.0`, `weight_decay=1e-3`
      - xlarge      : `mask_ratio=2`, `dropout=0.1`, `weight_decay=1e-3`

  * Mantidas (pontos da v3 que de fato funcionaram):
      - `amp=True`, `optimizer="AdamW"` (corrigiu NaN do xlarge da v2)
      - `close_mosaic=15`, `erasing=0.0`, `copy_paste=0.3`
      - `warmup_epochs=5`, `seed=0`, `deterministic=True`
      - `hsv_s=0.4`, `degrees=180.0`, `flipud=0.5`, `fliplr=0.5`
"""

import time
from ultralytics import YOLO


# =============================================================================
# Para trocar o tamanho do modelo, basta mudar esta linha (e reusar o script).
# Valores válidos: "nano", "small", "medium", "large", "xlarge".
# =============================================================================
MODEL_SIZE = "small"


# Configurações por tamanho de modelo (apenas o que muda).
# A justificativa de cada valor está no docstring acima.
_PER_SIZE = {
    "nano":   dict(weight="yolo26n-seg.pt", mask_ratio=4, dropout=0.0,
                   weight_decay=5e-4, auto_augment="randaugment"),
    "small":  dict(weight="yolo26s-seg.pt", mask_ratio=4, dropout=0.0,
                   weight_decay=5e-4, auto_augment="randaugment"),
    "medium": dict(weight="yolo26m-seg.pt", mask_ratio=4, dropout=0.0,
                   weight_decay=1e-3, auto_augment=None),
    "large":  dict(weight="yolo26l-seg.pt", mask_ratio=4, dropout=0.0,
                   weight_decay=1e-3, auto_augment=None),
    "xlarge": dict(weight="yolo26x-seg.pt", mask_ratio=2, dropout=0.1,
                   weight_decay=1e-3, auto_augment=None),
}


def main():
    if MODEL_SIZE not in _PER_SIZE:
        raise ValueError(
            f"MODEL_SIZE inválido: {MODEL_SIZE!r}. "
            f"Use um de: {list(_PER_SIZE)}"
        )
    cfg = _PER_SIZE[MODEL_SIZE]

    start_time = time.perf_counter()

    model = YOLO(f"/workspace/cache/{cfg['weight']}")

    results = model.train(
        # =====================================================================
        # 1. DADOS E INFRAESTRUTURA
        # =====================================================================
        data="/workspace/datasets/isic_2018_task1_yolo26/data.yaml",
        project="/workspace/logs",
        name=f"yolo26_{MODEL_SIZE}_ft_isic_2018_v4",
        exist_ok=False,
        task="segment",
        pretrained=True,

        epochs=120,             # v4: 150 -> 120. Orçamento sobra com LR adequado.
        time=None,              # Sem cap de wall-clock.

        imgsz=640,              # Mantido (pedido explícito do usuário).
        rect=False,
        multi_scale=0.0,
        fraction=1.0,
        single_cls=False,

        # =====================================================================
        # 2. HARDWARE / DATALOADER
        # =====================================================================
        device=[0, 1],          # DDP em 2 GPUs. Use `device=0` se quiser single-GPU.
        batch=32,               # v4: 16 -> 32 (=16/GPU em DDP). Ultralytics 8.4.21
                                #   NÃO aceita batch=-1 em multi-GPU. Dobrar p/ 32
                                #   reverte o efeito do DDP halving da v3 e leva
                                #   `lr/pg0` mid de ~7e-4 (v3) para ~2.8e-3 (≈v2).
                                #   Em single-GPU pode usar `batch=-1` (autobatch).
        workers=8,
        cache=False,            # Setar "ram" se houver RAM sobrando.
        compile=False,          # torch.compile ainda instável com heads de seg.

        # =====================================================================
        # 3. ESTABILIDADE NUMÉRICA / OTIMIZADOR
        # =====================================================================
        amp=True,               # Mantido. Resolveu o NaN do xlarge (v2 ep1=629).
        optimizer="AdamW",      # Mantido. Estável com AMP em todos os tamanhos.
        nbs=64,                 # Default. NÃO mudar sem recalibrar lr0.

        # =====================================================================
        # 4. HIPERPARÂMETROS DE APRENDIZADO  (núcleo do fix v4)
        # =====================================================================
        lr0=2e-3,               # v4: 1e-3 -> 2e-3. Compensa batch menor + DDP.
                                #   v3 ficou em lr/pg0 mid ~7e-4 (vs ~2.5e-3 da v2).
        lrf=0.01,
        momentum=0.937,         # Para AdamW, é beta1.
        weight_decay=cfg["weight_decay"],
        warmup_epochs=5.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        cos_lr=True,
        patience=20,            # v4: 30 -> 20. Em v3 todos os runs tinham 30+
                                #   épocas depois do best sem melhoria.

        # =====================================================================
        # 5. REGULARIZAÇÃO E CHECKPOINTS
        # =====================================================================
        dropout=cfg["dropout"],         # Por tamanho (ver _PER_SIZE).
        freeze=None,                    # Se overfittar cedo, tentar `freeze=10`.
        seed=0,
        deterministic=True,
        save=True,
        save_period=-1,
        resume=False,

        # =====================================================================
        # 6. GANHOS DE LOSS
        # =====================================================================
        box=7.5,
        cls=0.3,                # Reduzido (default 0.5): com 1 classe, cls é trivial.
        dfl=1.5,

        # =====================================================================
        # 7. SEGMENTAÇÃO
        # =====================================================================
        overlap_mask=True,
        mask_ratio=cfg["mask_ratio"],   # 4 para nano/small/medium/large; 2 para xlarge.
        retina_masks=False,

        # =====================================================================
        # 8. VALIDAÇÃO / MÉTRICAS
        # =====================================================================
        val=True,
        split="val",
        plots=True,
        verbose=True,
        conf=None,
        iou=0.7,
        max_det=300,
        half=False,
        save_json=False,

        # =====================================================================
        # 9. DATA AUGMENTATION
        # =====================================================================
        # -- Geometria --
        degrees=180.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.0,
        flipud=0.5,
        fliplr=0.5,

        # -- Cor --
        hsv_h=0.015,
        hsv_s=0.4,              # Reduzido (era 0.7) — tom é semi-diagnóstico em pele.
        hsv_v=0.4,
        bgr=0.0,

        # -- Composições / Oclusões --
        mosaic=1.0,
        close_mosaic=15,        # Mantido — desliga mosaic nas últimas 15 ep.
        mixup=0.1,
        cutmix=0.0,
        copy_paste=0.3,
        copy_paste_mode="flip",
        auto_augment=cfg["auto_augment"],   # randaugment para nano/small; None p/ resto.
        erasing=0.0,            # Mantido — random-erasing corrompe máscara.
    )

    tempo_total = time.perf_counter() - start_time
    print(f"\nTreinamento concluído com sucesso ({MODEL_SIZE}).")
    print(f"Tempo total de execução: {tempo_total / 60:.2f} minutos.")
    return results


if __name__ == "__main__":
    main()
