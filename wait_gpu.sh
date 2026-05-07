#!/bin/bash

echo "Aguardando as GPUs 0 e 1 ficarem livres por 5 minutos contínuos..."

CHECK_INTERVAL=60
REQUIRED_IDLE_MINUTES=3
IDLE_COUNT=0

while true; do
    # Captura métricas das GPUs
    GPU0_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
    GPU1_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1)
    GPU0_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 0)
    GPU1_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 1)

    # Condição: Memória < 1000MB E Uso < 10% para AMBAS as GPUs
    if [ "$GPU0_MEM" -lt 1000 ] && [ "$GPU1_MEM" -lt 1000 ] && \
       [ "$GPU0_UTIL" -lt 10 ] && [ "$GPU1_UTIL" -lt 10 ]; then
        
        # Incrementa o contador se estiver ocioso
        ((IDLE_COUNT++))
        echo "$(date) | GPU0: ${GPU0_MEM}MiB ${GPU0_UTIL}% | GPU1: ${GPU1_MEM}MiB ${GPU1_UTIL}% -> Ociosa há $IDLE_COUNT minuto(s)."

        # Verifica se atingiu o tempo necessário
        if [ "$IDLE_COUNT" -ge "$REQUIRED_IDLE_MINUTES" ]; then
            echo "GPUs livres por $REQUIRED_IDLE_MINUTES minutos contínuos! Iniciando treinamento do SAM3..."
            break
        fi
    else
        # Se houve pico de uso, verifica se o contador estava rodando para avisar do reset
        if [ "$IDLE_COUNT" -gt 0 ]; then
            echo "$(date) | Atividade detectada! Resetando contador de ociosidade."
        else
            echo "$(date) | GPU0: ${GPU0_MEM}MiB ${GPU0_UTIL}% | GPU1: ${GPU1_MEM}MiB ${GPU1_UTIL}% -> Em uso."
        fi
        
        # Zera o contador
        IDLE_COUNT=0
    fi

    sleep $CHECK_INTERVAL
done

# Inicia o Docker
# docker run --gpus all -it --rm \
#   --ipc=host \
#   --user $(id -u):$(id -g) \
#   -e TORCH_HOME=/workspace/cache/torch \
#   -e HOME=/workspace/cache \
#   -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
#   -v $(pwd)/datasets:/workspace/datasets \
#   -v $(pwd)/logs:/workspace/logs \
#   -v $(pwd)/yolo26_seg:/workspace/yolo26_seg \
#   -v $(pwd)/utils:/workspace/utils \
#   -v $(pwd)/cache:/workspace/cache \
#   -v /etc/passwd:/etc/passwd:ro \
#   -v /etc/group:/etc/group:ro \
#   yolo26_ft \
#   python /workspace/yolo26_seg/train.py 2>&1 | tee logs/yolo26_xlarge_ft_ph2_150.log

# docker run --gpus all -it --rm \
#   --ipc=host \
#   --user $(id -u):$(id -g) \
#   -e TORCH_HOME=/workspace/cache/torch \
#   -e HOME=/workspace/cache \
#   -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
#   -v $(pwd)/datasets:/workspace/datasets \
#   -v $(pwd)/logs:/workspace/logs \
#   -v $(pwd)/yolo26_seg:/workspace/yolo26_seg \
#   -v $(pwd)/utils:/workspace/utils \
#   -v $(pwd)/cache:/workspace/cache \
#   -v /etc/passwd:/etc/passwd:ro \
#   -v /etc/group:/etc/group:ro \
#   yolo26_ft \
#   python /workspace/yolo26_seg/train_isic_2018_task_1_v2.py 2>&1 | tee logs/yolo26_small_ft_isic_2018_v2.log


docker run --gpus all -it --rm \
  --ipc=host \
  --user $(id -u):$(id -g) \
  -e TORCH_HOME=/workspace/cache/torch \
  -e HOME=/workspace/cache \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v $(pwd)/datasets:/workspace/datasets \
  -v $(pwd)/logs:/workspace/logs \
  -v $(pwd)/yolo26_seg:/workspace/yolo26_seg \
  -v $(pwd)/utils:/workspace/utils \
  -v $(pwd)/cache:/workspace/cache \
  -v /etc/passwd:/etc/passwd:ro \
  -v /etc/group:/etc/group:ro \
  yolo26_ft \
  python /workspace/yolo26_seg/train_isic_2018_task_1_v8.py --model large 2>&1 | tee logs/yolo26_large_ft_isic_2018_v8.log


docker run --gpus all -it --rm \
  --ipc=host \
  --user $(id -u):$(id -g) \
  -e TORCH_HOME=/workspace/cache/torch \
  -e HOME=/workspace/cache \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v $(pwd)/datasets:/workspace/datasets \
  -v $(pwd)/logs:/workspace/logs \
  -v $(pwd)/yolo26_seg:/workspace/yolo26_seg \
  -v $(pwd)/utils:/workspace/utils \
  -v $(pwd)/cache:/workspace/cache \
  -v /etc/passwd:/etc/passwd:ro \
  -v /etc/group:/etc/group:ro \
  yolo26_ft \
  python /workspace/yolo26_seg/tune_isic_2018_task_1.py \
  --model nano --iterations 3 --epochs 5 | tee logs/tune_isic_2018_task_1.log




# docker run --gpus all -it --rm --ipc=host \
#   --user $(id -u):$(id -g) \
#   -e TORCH_HOME=/workspace/cache/torch -e HOME=/workspace/cache \
#   -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
#   -v $(pwd)/datasets:/workspace/datasets \
#   -v $(pwd)/logs:/workspace/logs \
#   -v $(pwd)/yolo26_seg:/workspace/yolo26_seg \
#   -v $(pwd)/utils:/workspace/utils \
#   -v $(pwd)/cache:/workspace/cache \
#   -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro \
#   yolo26_ft \
#   python /workspace/yolo26_seg/tune_all_models.py --models small \
#   2>&1 | tee logs/tune_small.log

docker run --gpus all -it --rm --ipc=host \
  --user $(id -u):$(id -g) \
  -e TORCH_HOME=/workspace/cache/torch -e HOME=/workspace/cache \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v $(pwd)/datasets:/workspace/datasets \
  -v $(pwd)/logs:/workspace/logs \
  -v $(pwd)/yolo26_seg:/workspace/yolo26_seg \
  -v $(pwd)/utils:/workspace/utils \
  -v $(pwd)/cache:/workspace/cache \
  -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro \
  yolo26_ft \
  python /workspace/yolo26_seg/tune_all_models.py \
    --space refined --iterations 50 \
  2>&1 | tee logs/tune_all_refined.log