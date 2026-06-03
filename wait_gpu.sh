#!/bin/bash
# =============================================================================
# wait_gpu.sh — Wait for both GPUs to be idle, then launch ad-hoc trainings.
#
# What this script does
# ---------------------
# 1. Polls ``nvidia-smi`` once per minute on the host's GPUs 0 and 1.
# 2. A GPU is considered "idle" when memory.used < 1000 MiB AND
#    utilization.gpu < 10%.
# 3. When BOTH GPUs stay idle for ``REQUIRED_IDLE_MINUTES`` consecutive
#    checks, the script breaks out of the polling loop and runs the
#    ``docker run ...`` blocks defined below.
#
# This is a convenience helper for shared GPU hosts: it lets you queue an
# experiment to start as soon as the host frees up, without writing a full
# scheduler. The canonical pipeline (``run_pipeline.sh``) does NOT use this
# script — it is kept here for opportunistic, manual usage.
#
# Configurable knobs (edit in place if needed)
# --------------------------------------------
#   CHECK_INTERVAL          : Polling interval in seconds (default 60).
#   REQUIRED_IDLE_MINUTES   : Consecutive idle checks required to launch.
#
# Editable docker invocations
# ---------------------------
# The blocks below this header are intentionally left as plain ``docker run``
# commands so they can be edited per experiment. Comment / uncomment to pick
# what should be launched once the GPUs free up.
# =============================================================================

echo "Waiting for GPUs 0 and 1 to stay idle for several minutes..."

CHECK_INTERVAL=60
REQUIRED_IDLE_MINUTES=3
IDLE_COUNT=0

while true; do
    # Per-GPU memory.used (MiB) and utilization.gpu (%)
    GPU0_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
    GPU1_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1)
    GPU0_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 0)
    GPU1_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 1)

    # Idle condition: memory < 1000 MiB AND utilization < 10% for BOTH GPUs
    if [ "$GPU0_MEM" -lt 1000 ] && [ "$GPU1_MEM" -lt 1000 ] && \
       [ "$GPU0_UTIL" -lt 10 ] && [ "$GPU1_UTIL" -lt 10 ]; then

        ((IDLE_COUNT++))
        echo "$(date) | GPU0: ${GPU0_MEM}MiB ${GPU0_UTIL}% | GPU1: ${GPU1_MEM}MiB ${GPU1_UTIL}% -> idle for $IDLE_COUNT minute(s)."

        if [ "$IDLE_COUNT" -ge "$REQUIRED_IDLE_MINUTES" ]; then
            echo "GPUs idle for $REQUIRED_IDLE_MINUTES consecutive minute(s) — launching scheduled training."
            break
        fi
    else
        # Activity detected — reset the idle counter and log loudly so it is
        # clear in the long-running log that the run window was missed.
        if [ "$IDLE_COUNT" -gt 0 ]; then
            echo "$(date) | Activity detected — resetting idle counter."
        else
            echo "$(date) | GPU0: ${GPU0_MEM}MiB ${GPU0_UTIL}% | GPU1: ${GPU1_MEM}MiB ${GPU1_UTIL}% -> busy."
        fi
        IDLE_COUNT=0
    fi

    sleep $CHECK_INTERVAL
done

# -----------------------------------------------------------------------------
# Editable docker invocations (the actual workload to start once GPUs free up).
# Comment or uncomment as needed; the loop above will fall through into these.
# -----------------------------------------------------------------------------

# Example A: optimised fine-tuning of a single variant (legacy script name)
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


# Example B: ad-hoc HPO of a single variant (smoke / quick sweep)
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


# Example C: refined HPO over all sizes (legacy ``tune_all_models.py``)
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


# Example D: 5-Fold CV for the four smaller variants (skipping ``small``)
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
  python /workspace/yolo26_seg/train_all_models_cv.py --models nano medium large xlarge \
  2>&1 | tee logs/train_all_models_cv_all_less_small_v1.log
