import time
from ultralytics import YOLO

def main():
    start_time = time.perf_counter()
    
    # Carregamento do modelo base (versão xlarge para máxima capacidade de extração de features)
    model = YOLO("/workspace/cache/yolo26s-seg.pt")

    # Configuração exaustiva do Fine-Tuning
    results = model.train(
        # ==========================================
        # 1. INFRAESTRUTURA E DADOS
        # ==========================================
        data="/workspace/datasets/isic_2018_task1_yolo26/data.yaml",
        project="/workspace/logs",
        name="yolo26_small_ft_isic_2018_completo",
        epochs=100,
        imgsz=640,
        device=[0, 1],             # Aloca a carga para a(s) GPU(s) com maior memória/ociosidade
        batch=32,              # Auto-ajuste de batch size para ocupar ~60% da VRAM com segurança
        workers=4,             # Threads dedicadas ao dataloader
        
        # ==========================================
        # 2. ESTABILIDADE E OTIMIZAÇÃO NUMÉRICA
        # ==========================================
        amp=False,             # CRÍTICO: Previne a corrupção de tensores (NaN/Inf Loss) na versão Xlarge
        optimizer="MuSGD",     # Otimizador híbrido (SGD + atualizações Muon), superior em runs extensos
        
        # ==========================================
        # 3. HIPERPARÂMETROS DE APRENDIZADO
        # ==========================================
        lr0=0.001,             # Taxa inicial conservadora para fine-tuning (padrão é 0.01)
        lrf=0.01,              # Fator da taxa final (terminará em 1% da taxa inicial)
        momentum=0.937,        # Inércia para suavizar as atualizações do gradiente
        weight_decay=0.0005,   # Regularização L2 para penalizar a memorização exata do conjunto de treino
        warmup_epochs=3.0,     # Épocas iniciais onde o LR sobe gradualmente para não quebrar os pesos
        warmup_momentum=0.8,   # Momentum ajustado durante a fase de aquecimento
        cos_lr=True,           # Agendador em curva de cosseno (queda mais suave que a linear)
        patience=25,           # Early stopping: encerra se o mAP não melhorar em 25 épocas consecutivas
        
        # ==========================================
        # 4. BALANCEAMENTO DE CLASSES
        # ==========================================
        # cls_pw=0.25,           # Dá mais tração (peso) a classes menos frequentes (ajuste entre 0.0 e 1.0)
        
        # ==========================================
        # 5. DATA AUGMENTATION (Otimizado para Pele)
        # ==========================================
        degrees=180.0,         # Rotação total, já que lesões dermatológicas não têm "cima" ou "baixo"
        flipud=0.5,            # Inversão vertical (50% de chance)
        fliplr=0.5,            # Inversão horizontal (50% de chance)
        hsv_h=0.015,           # Variação de matiz (simula diferentes hardwares/câmeras)
        hsv_s=0.7,             # Variação de saturação (simula presença de fluidos/gel)
        hsv_v=0.4,             # Variação de brilho e iluminação do ambiente
        translate=0.1,         # Deslocamento espacial (+/- 10% da imagem)
        scale=0.5,             # Zoom in/out (+/- 50%) para forçar o reconhecimento em diversas resoluções
        mosaic=1.0,            # Ativa colagem de 4 imagens (excelente para o modelo entender contexto de bordas)
        mixup=0.1              # Sobreposição leve de máscaras, ajudando a criar predições de bordas mais suaves
    )
    
    tempo_total = time.perf_counter() - start_time
    print(f"\nTreinamento concluído com sucesso.")
    print(f"Tempo total de execução: {tempo_total / 60:.2f} minutos.")

if __name__ == "__main__":
    main()