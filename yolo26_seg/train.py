import time
from ultralytics import YOLO

def main():
    start_time = time.perf_counter()
    
    # 1. Carrega o modelo pré-treinado de segmentação (versão 'nano' para iniciar)
    # yolo26n-seg.pt yolo26s-seg.pt yolo26m-seg.pt yolo26l-seg.pt yolo26x-seg.pt
    
    model = YOLO("/workspace/cache/yolo26x-seg.pt")

    # 2. Inicia o Fine-Tuning
    results = model.train(
        data="/workspace/datasets/isic_2018_task1_yolo26/data.yaml", # Aponta para o YAML ajustado
        epochs=100, # Número de épocas
        imgsz=640, # Tamanho da imagem
        project="/workspace/logs", # Onde salvar os resultados (mapeado para sua pasta host)
        name="yolo26_xlarge_ft_isic_task_1_150", # Nome da pasta do experimento
        device=1, # Usa a GPU 0
        workers=4, # Threads para carregar dados
        batch=16 # Tamanho do lote (ajuste conforme a VRAM da sua GPU)
    )
    
    print(f"Tempo de treinamento: {(time.perf_counter() - start_time)}")

if __name__ == "__main__":
    main()