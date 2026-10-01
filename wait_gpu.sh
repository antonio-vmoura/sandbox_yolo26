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
# scheduler. It launches the canonical pipeline (``run_pipeline.sh``); edit the
# ``docker run`` blocks at the bottom to choose what starts.
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
# run_pipeline.sh is resumable: re-launching the same command continues an
# interrupted study instead of starting over.
# -----------------------------------------------------------------------------
PIPELINE_NAME="${PIPELINE_NAME:-pipeline_final_v1}"
DOCKER_ARGS=(
  --gpus all --rm --ipc=host
  --user "$(id -u):$(id -g)"
  -e TORCH_HOME=/workspace/cache/torch -e HOME=/workspace/cache
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  -e PIPELINE_NAME="${PIPELINE_NAME}"
  -v "$(pwd)/datasets:/workspace/datasets"
  -v "$(pwd)/logs:/workspace/logs"
  -v "$(pwd)/yolo26_seg:/workspace/yolo26_seg"
  -v "$(pwd)/utils:/workspace/utils"
  -v "$(pwd)/cache:/workspace/cache"
  -v "$(pwd)/run_pipeline.sh:/workspace/run_pipeline.sh:ro"
  -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro
)

# Example A: the full 5-phase study, all model sizes
docker run "${DOCKER_ARGS[@]}" yolo26_ft \
  bash /workspace/run_pipeline.sh \
  2>&1 | tee "logs/${PIPELINE_NAME}_$(date -u +%Y%m%dT%H%M%SZ).log"

# Example B: HPO (Phase 3) for xlarge only, FP32 on 32 GB GPUs (micro-batch 16)
# docker run "${DOCKER_ARGS[@]}" yolo26_ft \
#   bash /workspace/run_pipeline.sh --phases "3" --models x --hpo-batch 16 \
#   2>&1 | tee "logs/${PIPELINE_NAME}_hpo_xlarge_$(date -u +%Y%m%dT%H%M%SZ).log"

# Example C: Phase 5 only (test set + efficiency) on GPU 1, once Phases 1-4 are done
# docker run "${DOCKER_ARGS[@]}" yolo26_ft \
#   bash /workspace/run_pipeline.sh --phases "5" --bench-device 1 \
#   2>&1 | tee "logs/${PIPELINE_NAME}_phase5_$(date -u +%Y%m%dT%H%M%SZ).log"
