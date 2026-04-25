import time
from ultralytics import YOLO

def main():
    start_time = time.perf_counter()

    # Modelo base pré-treinado em COCO-seg.
    #      Se observar gap train/val grande, trocar por yolo26l-seg.pt (large).
    model = YOLO("/workspace/cache/yolo26s-seg.pt")

    results = model.train(
        # =====================================================================
        # 1. DADOS E INFRAESTRUTURA
        # =====================================================================
        data="/workspace/datasets/isic_2018_task1_yolo26/data.yaml",
        project="/workspace/logs",
        name="yolo26_small_ft_isic_2018_completo",
        exist_ok=False,        # Falha se a pasta já existir — evita sobrescrever experimento.
        task="segment",        # Explicitar a task (herdada do .pt, mas deixa robusto a troca de peso).
        pretrained=True,       # Carrega pesos do .pt — essencial para fine-tuning.

        epochs=150,            # 150 com patience=30 tende a convergir sem desperdício (cosine + LR baixo).
        time=None,             # Sem limite de wall-clock; se quiser cap, ex.: time=24 (horas).

        imgsz=640,             # Mantido a pedido. (Obs.: imagens ISIC são ≫ 640 e a métrica é IoU de borda;
        rect=False,            # Em treino, manter False (rect quebra mosaic). True só faz sentido em val.
        multi_scale=0.0,       # Desativado: custa VRAM e pouco adiciona em dataset homogêneo como ISIC.
        fraction=1.0,          # Usar 100% do treino; reduzir só para debug.
        single_cls=False,      # data.yaml já tem nc=1; deixar False é correto.

        # =====================================================================
        # 2. HARDWARE / DATALOADER
        # =====================================================================
        device=[0, 1],              # Explicitar GPU. Para multi-GPU/DDP use lista, ex.: device=[0, 1].
        batch=16,              # Fixo p/ reprodutibilidade. Para imgsz=768 + xlarge cabe em ~24-32GB.
                               #   Em DDP, é o batch GLOBAL (dividido entre GPUs).
        workers=8,             # Com imgsz=768 + augment pesado, 4 vira gargalo de I/O.
        cache=False,           # Com ~2.6k imagens em 768, "ram" cacheia tudo (~10-15GB) e acelera.
                               #   Setar "ram" se a máquina tiver RAM sobrando, ou "disk" em SSD.
        compile=False,         # torch.compile ainda instável com heads de seg do Ultralytics; deixar off.

        # =====================================================================
        # 3. ESTABILIDADE NUMÉRICA / OTIMIZADOR
        # =====================================================================
        amp=True,              # AMP ligado: ~2x mais rápido e ~40% menos VRAM.
                               #   MuSGD+xlarge gerava NaN; com AdamW abaixo, AMP é estável.
        optimizer="AdamW",     # AdamW é o recomendado Ultralytics p/ fine-tuning em datasets pequenos.
                               #   Alternativas: "SGD" (clássico YOLO; requer lr0=0.01) ou "auto".
        nbs=64,                # Nominal batch size — Ultralytics escala LR/decay por (batch/nbs).
                               #   Manter 64 (default) para consistência com curvas conhecidas.

        # =====================================================================
        # 4. HIPERPARÂMETROS DE APRENDIZADO
        # =====================================================================
        lr0=1e-3,              # AdamW fine-tuning: 1e-3 é conservador e estável.
                               #   Se usar SGD, subir para 1e-2.
        lrf=0.01,              # LR final = 1% do lr0 (cosine decay).
        momentum=0.937,        # Momentum p/ SGD OU beta1 do AdamW (Ultralytics reaproveita o param).
        weight_decay=1e-3,     # Um pouco maior que o default (5e-4): regulariza modelo grande em dataset pequeno.
        warmup_epochs=5.0,     # Warmup mais longo (3 -> 5) reduz risco de NaN nas primeiras iters com AMP.
        warmup_momentum=0.8,   # Momentum inicial durante o warmup.
        warmup_bias_lr=0.1,    # LR alto nos bias nos primeiros steps ajuda heads a "acordarem".
        cos_lr=True,           # Cosine decay — mais suave que linear, melhor em runs longos.
        patience=30,           # Val tem só 100 imgs — métrica oscila; 30 evita parada prematura.

        # =====================================================================
        # 5. REGULARIZAÇÃO E CHECKPOINTS
        # =====================================================================
        dropout=0.1,            # Dropout no head (apenas em modelos que suportam). Ajuda contra overfitting.
        freeze=None,            # Sem congelamento. Se overfittar muito cedo, usar freeze=10 (backbone).
        seed=0,                 # Reprodutibilidade.
        deterministic=True,     # Operações determinísticas sempre que possível (reprodutibilidade).
        save=True,              # Salvar checkpoints.
        save_period=-1,         # -1 = salva só best/last. Ex.: 10 salvaria a cada 10 épocas (muito disco).
        resume=False,           # Setar True só para retomar um run interrompido.

        # =====================================================================
        # 6. GANHOS DE LOSS (loss weights) — ajustados p/ tarefa de seg binária
        # =====================================================================
        box=7.5,                # Default. Box loss importa menos que mask, mas ainda guia localização.
        cls=0.3,                # Reduzido (default 0.5): com 1 classe, o cls_loss é quase trivial.
        dfl=1.5,                # Default. DFL refina caixa — ainda útil para seg.
        # pose=12.0, kobj=1.0, rle=1.0, angle=1.0   # Irrelevantes para seg (keypoints/OBB). Deixar default.

        # =====================================================================
        # 7. SEGMENTAÇÃO
        # =====================================================================
        overlap_mask=True,      # ISIC tem 1 instância por imagem; valor é indiferente, mantido padrão.
        mask_ratio=2,           # Downsample de máscara = 2 (ao invés do default 4). Em imgsz=640,
                                #   máscara sobe de 160×160 p/ 320×320 -> ganho direto em mAP@50-95 de borda.
                                #   Custo: ~1.3x VRAM. Principal alavanca quando imgsz está travado.
        retina_masks=False,     # Só afeta inferência/val; default False é o recomendado em treino.

        # =====================================================================
        # 8. VALIDAÇÃO / MÉTRICAS
        # =====================================================================
        val=True,               # Rodar validação a cada época.
        split="val",            # Usar o split "val" do data.yaml.
        plots=True,             # Gerar curvas de treino, matriz de confusão, exemplos.
        verbose=True,           # Logs detalhados.
        conf=None,              # Usa default de val (0.001) — NÃO forçar durante treino.
        iou=0.7,                # IoU de NMS em val. Default; manter.
        max_det=300,            # Max detecções por imagem em val.
        half=False,             # FP16 em val: deixar False em treino (AMP já cobre).
        save_json=False,        # COCO-style JSON não é útil aqui; desligado p/ economizar I/O.

        # =====================================================================
        # 9. DATA AUGMENTATION — perfil "pele / dermatoscopia"
        # =====================================================================
        # -- Geometria --
        degrees=180.0,          # Lesão não tem orientação canônica -> rotação total é válida.
        translate=0.1,          # Translação suave; lesões costumam estar centradas.
        scale=0.5,              # ±50% zoom — cobre variação de distância focal/crop.
        shear=2.0,              # Shear leve. >5° começa a distorcer a máscara.
        perspective=0.0,        # Imagens dermatoscópicas são frontais; perspectiva não é realista.
        flipud=0.5,             # Flip vertical OK para lesão.
        fliplr=0.5,             # Flip horizontal OK.

        # -- Cor --
        hsv_h=0.015,            # Shift de matiz mínimo — cor é semi-diagnóstica (melanina).
        hsv_s=0.4,              # Reduzido (era 0.7): 0.7 altera demais o tom da lesão.
        hsv_v=0.4,              # Simula variação de iluminação; manter.
        bgr=0.0,                # Troca aleatória de canais — NÃO usar em imagens médicas (cor importa).

        # -- Composições / Oclusões --
        mosaic=1.0,             # Mosaic ligado no início (ajuda contexto de bordas).
        close_mosaic=15,        # Desliga mosaic nas últimas 15 épocas -> refino em imagens reais.
        mixup=0.1,              # Sobreposição leve. >0.2 borra máscaras de contorno.
        cutmix=0.0,             # Cutmix corta máscara — indesejado para seg de borda.
        copy_paste=0.3,         # Copy-paste de instâncias é eficaz em seg com dataset pequeno.
        copy_paste_mode="flip", # Modo oficial Ultralytics para seg.
        auto_augment=None,      # Desativar: política de ImageNet é agressiva demais p/ dermatoscopia.
        erasing=0.0,            # DESLIGAR: random-erasing pode ocultar parte da lesão e corromper a máscara.
    )

    tempo_total = time.perf_counter() - start_time
    print("\nTreinamento concluído com sucesso.")
    print(f"Tempo total de execução: {tempo_total / 60:.2f} minutos.")
    return results


if __name__ == "__main__":
    main()
