# yolo26_seg/train.py
from ultralytics import YOLO

def main():
    # 1. Carrega o modelo pré-treinado de segmentação (versão 'nano' para iniciar)
    model = YOLO("/workspace/cache/yolo26n-seg.pt")

    # 2. Inicia o Fine-Tuning
    results = model.train(
        data="/workspace/datasets/ph2/data.yaml", # Aponta para o YAML ajustado
        epochs=100, # Número de épocas
        imgsz=640, # Tamanho da imagem
        project="/workspace/logs", # Onde salvar os resultados (mapeado para sua pasta host)
        name="ph2_finetuning", # Nome da pasta do experimento
        device=0, # Usa a GPU 0
        workers=4, # Threads para carregar dados
        batch=16 # Tamanho do lote (ajuste conforme a VRAM da sua GPU)
    )

if __name__ == "__main__":
    main()