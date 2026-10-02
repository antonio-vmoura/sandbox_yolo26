#!/bin/bash
# =============================================================================
# wait_gpu.sh — Wait for one GPU to be idle, then launch the YOLO26 pipeline
# (run_pipeline.sh) inside the ``yolo26_ft`` container.
#
# 1. Polls ``nvidia-smi`` every ``POLL_INTERVAL`` seconds (default 5) on host
#    GPU ${GPU_DEVICE}.
# 2. The GPU is "free" when it runs no compute process AND memory.used <
#    ``MAX_MEM_MIB`` (1000) AND utilization.gpu < ``MAX_UTIL`` (10%). A failed
#    ``nvidia-smi`` query counts as busy.
# 3. The pipeline starts IMMEDIATELY on the first free check
#    (``CONFIRM_CHECKS=1``). Set ``CONFIRM_CHECKS=N`` to require N consecutive
#    free checks, e.g. to skip the short gap between two jobs of another user.
#    Extra arguments are passed to run_pipeline.sh, e.g.
#    ``GPU_DEVICE=0 ./wait_gpu.sh --phases "1 2 3 4 5" --force``.
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
# Raw official ISIC 2018 Task 1 release (Phase 0 input, mounted read-only).
RAW_DATASET="${RAW_DATASET:-$(pwd)/../datasets/ISIC2018_Raw}"
POLL_INTERVAL="${POLL_INTERVAL:-5}"
CONFIRM_CHECKS="${CONFIRM_CHECKS:-1}"
MAX_MEM_MIB="${MAX_MEM_MIB:-1000}"
MAX_UTIL="${MAX_UTIL:-10}"

gpu_free() {
    local q mem util uuid apps
    STATUS=""
    q=$(nvidia-smi --query-gpu=memory.used,utilization.gpu,uuid --format=csv,noheader,nounits -i "${GPU_DEVICE}" 2>/dev/null) || return 1
    IFS=', ' read -r mem util uuid <<<"${q}"
    [[ "${mem}" =~ ^[0-9]+$ && "${util}" =~ ^[0-9]+$ ]] || return 1
    apps=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null | grep -c "${uuid}")
    STATUS="${mem}MiB ${util}% ${apps} proc"
    [ "${apps}" -eq 0 ] && [ "${mem}" -lt "${MAX_MEM_MIB}" ] && [ "${util}" -lt "${MAX_UTIL}" ]
}

echo "Waiting for GPU ${GPU_DEVICE} to be free (poll every ${POLL_INTERVAL}s, ${CONFIRM_CHECKS} check(s))..."
FREE_COUNT=0
LAST_STATE=""
while true; do
    if gpu_free; then
        ((FREE_COUNT++))
        STATE="free"
    else
        FREE_COUNT=0
        STATE="busy"
    fi
    # Log only on state changes (a 5 s poll would otherwise flood the terminal).
    if [ "${STATE}" != "${LAST_STATE}" ]; then
        echo "$(date) | GPU${GPU_DEVICE}: ${STATUS:-nvidia-smi failed} -> ${STATE}."
        LAST_STATE="${STATE}"
    fi
    [ "${FREE_COUNT}" -ge "${CONFIRM_CHECKS}" ] && break
    sleep "${POLL_INTERVAL}"
done
echo "GPU free — launching the YOLO26 pipeline."

mkdir -p "logs/${PIPELINE_NAME}"
docker run --gpus "\"device=${GPU_DEVICE}\"" --rm --ipc=host \
  --user "$(id -u):$(id -g)" \
  -e TORCH_HOME=/workspace/cache/torch -e HOME=/workspace/cache \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e GPU_DEVICE_IDS=0 -e PIPELINE_NAME="${PIPELINE_NAME}" \
  -v "$(pwd)/datasets:/workspace/datasets" \
  -v "${RAW_DATASET}:/workspace/raw:ro" \
  -v "$(pwd)/logs:/workspace/logs" \
  -v "$(pwd)/yolo26_seg:/workspace/yolo26_seg" \
  -v "$(pwd)/utils:/workspace/utils" \
  -v "$(pwd)/cache:/workspace/cache" \
  -v "$(pwd)/run_pipeline.sh:/workspace/run_pipeline.sh:ro" \
  -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro \
  yolo26_ft \
  bash /workspace/run_pipeline.sh "$@" \
  2>&1 | tee "logs/${PIPELINE_NAME}/terminal_$(date -u +%Y%m%dT%H%M%SZ).log"
