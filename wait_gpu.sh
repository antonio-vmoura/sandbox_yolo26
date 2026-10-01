#!/bin/bash
# =============================================================================
# wait_gpu.sh — Wait for one GPU to be idle, then launch the YOLO26 pipeline
# (run_pipeline.sh) inside the ``yolo26_ft`` container.
#
# 1. Polls ``nvidia-smi`` once per minute on host GPU ${GPU_DEVICE}.
# 2. The GPU is "idle" when memory.used < 1000 MiB AND utilization.gpu < 10%.
# 3. After ``REQUIRED_IDLE_MINUTES`` consecutive idle checks it runs the
#    ``docker run`` block below. Extra arguments are passed to run_pipeline.sh,
#    e.g. ``GPU_DEVICE=0 ./wait_gpu.sh --phases "1 2 3 4 5" --force``.
#
# The pipeline trains on that single GPU (index 0 inside the container), so
# the other GPU stays free for the U-Net / SAM 3 pipelines. run_pipeline.sh is
# resumable: re-launching the same command WITHOUT --force continues an
# interrupted study instead of starting over.
#
# The terminal log is written inside the pipeline folder:
#   logs/${PIPELINE_NAME}/terminal_<UTC>.log
# =============================================================================

GPU_DEVICE="${GPU_DEVICE:-0}"
PIPELINE_NAME="${PIPELINE_NAME:-pipeline_final_v1}"
CHECK_INTERVAL=60
REQUIRED_IDLE_MINUTES=3
IDLE_COUNT=0

echo "Waiting for GPU ${GPU_DEVICE} to stay idle for ${REQUIRED_IDLE_MINUTES} minute(s)..."

while true; do
    MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${GPU_DEVICE}")
    UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "${GPU_DEVICE}")
    if [ "$MEM" -lt 1000 ] && [ "$UTIL" -lt 10 ]; then
        ((IDLE_COUNT++))
        echo "$(date) | GPU${GPU_DEVICE}: ${MEM}MiB ${UTIL}% -> idle for $IDLE_COUNT minute(s)."
        if [ "$IDLE_COUNT" -ge "$REQUIRED_IDLE_MINUTES" ]; then
            echo "GPU free — launching the YOLO26 pipeline."
            break
        fi
    else
        if [ "$IDLE_COUNT" -gt 0 ]; then
            echo "$(date) | Activity detected — resetting idle counter."
        else
            echo "$(date) | GPU${GPU_DEVICE}: ${MEM}MiB ${UTIL}% -> busy."
        fi
        IDLE_COUNT=0
    fi
    sleep $CHECK_INTERVAL
done

mkdir -p "logs/${PIPELINE_NAME}"
docker run --gpus "\"device=${GPU_DEVICE}\"" --rm --ipc=host \
  --user "$(id -u):$(id -g)" \
  -e TORCH_HOME=/workspace/cache/torch -e HOME=/workspace/cache \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e GPU_DEVICE_IDS=0 -e PIPELINE_NAME="${PIPELINE_NAME}" \
  -v "$(pwd)/datasets:/workspace/datasets" \
  -v "$(pwd)/logs:/workspace/logs" \
  -v "$(pwd)/yolo26_seg:/workspace/yolo26_seg" \
  -v "$(pwd)/utils:/workspace/utils" \
  -v "$(pwd)/cache:/workspace/cache" \
  -v "$(pwd)/run_pipeline.sh:/workspace/run_pipeline.sh:ro" \
  -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro \
  yolo26_ft \
  bash /workspace/run_pipeline.sh "$@" \
  2>&1 | tee "logs/${PIPELINE_NAME}/terminal_$(date -u +%Y%m%dT%H%M%SZ).log"
