#!/usr/bin/env bash
# =============================================================================
# run_pipeline.sh — Master orchestrator for the YOLO26-seg fine-tuning pipeline
# on the ISIC 2018 Task 1 dataset.
#
# Pipeline phases (executed sequentially and idempotently):
#   Phase 1 — Baseline training (Ultralytics defaults, 120 ep, patience=20)
#   Phase 2 — Hyperparameter Optimization via model.tune() (refined space)
#   Phase 3 — Optimized fine-tuning (single-split) using HP yamls from Phase 2
#   Phase 4 — 5-Fold Cross-Validation (deterministic, seed=0, NumPy-only KFold)
#             + consolidation (mean/std) of mAP@50, mAP@50-95, P, R, F1 across
#             models.
#
# Idempotency: each underlying Python script already detects existing artefacts
# (best.pt, best_hyperparameters.yaml, metrics_summary.json) and skips work.
# Use --force to override at the per-phase level.
#
# All commands assume the script runs *inside* the ``yolo26_ft`` Docker
# container with the standard volume mounts (datasets, logs, yolo26_seg,
# utils, cache). See README.md for the full ``docker run`` invocation.
# =============================================================================
set -euo pipefail

# ---------- Defaults ---------------------------------------------------------
DATA_YAML="${DATA_YAML:-/workspace/datasets/isic_2018_task1_yolo26/data.yaml}"
# Isolate this pipeline's outputs under a dedicated sub-directory of LOGS_ROOT
# so they don't get mixed with previous standalone runs (HPO, CV, FT) that
# already live directly under /workspace/logs/. Override PIPELINE_NAME to start
# a fresh run (e.g. pipeline_e2e_v2) without touching the previous artefacts.
LOGS_ROOT="${LOGS_ROOT:-/workspace/logs}"
PIPELINE_NAME="${PIPELINE_NAME:-pipeline_e2e_v1}"
# Detect whether PROJECT was pre-set via env var so we don't silently
# clobber it during the LOGS_ROOT/PIPELINE_NAME recomposition below.
if [[ -n "${PROJECT:-}" ]]; then
    PROJECT_FORCED=1
else
    PROJECT_FORCED=0
fi
PROJECT="${PROJECT:-${LOGS_ROOT}/${PIPELINE_NAME}}"
GPU_DEVICE_IDS="${GPU_DEVICE_IDS:-0,1}"
MODELS_DEFAULT=(nano small medium large xlarge)
MODELS=("${MODELS_DEFAULT[@]}")
PHASES=(1 2 3 4)

# Phase 1 (baseline)
P1_EPOCHS="${P1_EPOCHS:-120}"
P1_PATIENCE="${P1_PATIENCE:-20}"

# Phase 2 (HPO)
HPO_SPACE="${HPO_SPACE:-refined}"
HPO_ITERATIONS="${HPO_ITERATIONS:-30}"
HPO_EPOCHS_PER_TRIAL="${HPO_EPOCHS_PER_TRIAL:-30}"
HPO_PATIENCE="${HPO_PATIENCE:-10}"

# Phase 4 (CV)
CV_K_FOLDS="${CV_K_FOLDS:-5}"
CV_SEED="${CV_SEED:-0}"
CV_EPOCHS="${CV_EPOCHS:-120}"
CV_PATIENCE="${CV_PATIENCE:-25}"

FORCE_FLAG=""
DRY_RUN=0

YOLO_SEG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/yolo26_seg"
# When mounted in Docker, the canonical path is /workspace/yolo26_seg.
if [[ -d /workspace/yolo26_seg ]]; then
    YOLO_SEG_DIR=/workspace/yolo26_seg
fi

# Logs directory for the pipeline run itself (separate from per-model logs).
RUN_TS="$(date -u +%Y%m%dT%H%M%SZ)"
PIPELINE_LOG_DIR="${PROJECT}/pipeline_runs/${RUN_TS}"

usage() {
    cat <<EOF
Usage: $0 [options]

Options:
  --phases "1 2 3 4"          Subset of phases to run (default: 1 2 3 4).
  --models "n s m l x"        Subset of model sizes to process.
                              Accepts {nano,small,medium,large,xlarge} or
                              {n,s,m,l,x}. Default: all five.
  --data PATH                 Override data.yaml path. (env: DATA_YAML)
  --logs-root PATH            Parent dir for all pipeline runs.
                              (env: LOGS_ROOT, default /workspace/logs)
  --pipeline-name NAME        Sub-directory under LOGS_ROOT that isolates
                              THIS run's artefacts from any previous training
                              that lives directly under LOGS_ROOT.
                              (env: PIPELINE_NAME, default pipeline_e2e_v1)
  --project PATH              Explicit project root (overrides the
                              LOGS_ROOT/PIPELINE_NAME composition).
                              (env: PROJECT)
  --device "0,1"              GPU IDs passed to Ultralytics (DDP comma-sep).
                              (env: GPU_DEVICE_IDS)
  --force                     Pass --force to each underlying script.
  --dry-run                   Print commands without executing them.
  -h, --help                  Show this help and exit.

Environment variables (override defaults):
  DATA_YAML, LOGS_ROOT, PIPELINE_NAME, PROJECT, GPU_DEVICE_IDS,
  P1_EPOCHS, P1_PATIENCE,
  HPO_SPACE, HPO_ITERATIONS, HPO_EPOCHS_PER_TRIAL, HPO_PATIENCE,
  CV_K_FOLDS, CV_SEED, CV_EPOCHS, CV_PATIENCE
EOF
}

# ---------- CLI parsing ------------------------------------------------------
# Track whether --project was passed explicitly (takes precedence over
# LOGS_ROOT/PIPELINE_NAME composition).
PROJECT_EXPLICIT=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --phases) read -r -a PHASES <<<"$2"; shift 2 ;;
        --models) read -r -a MODELS <<<"$2"; shift 2 ;;
        --data) DATA_YAML="$2"; shift 2 ;;
        --logs-root) LOGS_ROOT="$2"; shift 2 ;;
        --pipeline-name) PIPELINE_NAME="$2"; shift 2 ;;
        --project) PROJECT="$2"; PROJECT_EXPLICIT=1; shift 2 ;;
        --device) GPU_DEVICE_IDS="$2"; shift 2 ;;
        --force) FORCE_FLAG="--force"; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
    esac
done

# Recompose PROJECT from LOGS_ROOT/PIPELINE_NAME unless --project was given
# explicitly (and PROJECT env var was NOT pre-set before invocation).
if [[ "${PROJECT_EXPLICIT}" -eq 0 && "${PROJECT_FORCED}" -eq 0 ]]; then
    PROJECT="${LOGS_ROOT}/${PIPELINE_NAME}"
fi

# Normalize model short aliases (n,s,m,l,x) to canonical names.
declare -A MODEL_ALIAS=(
    [n]=nano [s]=small [m]=medium [l]=large [x]=xlarge
    [nano]=nano [small]=small [medium]=medium [large]=large [xlarge]=xlarge
)
NORM_MODELS=()
for raw in "${MODELS[@]}"; do
    if [[ -z "${MODEL_ALIAS[$raw]:-}" ]]; then
        echo "[erro] Modelo desconhecido: '$raw'. Use n/s/m/l/x ou nome completo." >&2
        exit 2
    fi
    NORM_MODELS+=("${MODEL_ALIAS[$raw]}")
done
MODELS=("${NORM_MODELS[@]}")

mkdir -p "${PIPELINE_LOG_DIR}"
PIPELINE_LOG="${PIPELINE_LOG_DIR}/pipeline.log"

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%SZ)" "$*" | tee -a "${PIPELINE_LOG}"; }

log "=============================================================="
log "YOLO26-seg ISIC 2018 Task 1 — End-to-End Pipeline"
log "=============================================================="
log "  data           = ${DATA_YAML}"
log "  logs_root      = ${LOGS_ROOT}"
log "  pipeline_name  = ${PIPELINE_NAME}"
log "  project        = ${PROJECT}"
log "  device         = ${GPU_DEVICE_IDS}"
log "  models         = ${MODELS[*]}"
log "  phases         = ${PHASES[*]}"
log "  force          = ${FORCE_FLAG:-<off>}"
log "  yolo_seg_dir   = ${YOLO_SEG_DIR}"
log "  pipeline_log   = ${PIPELINE_LOG}"
log "--------------------------------------------------------------"

run_cmd() {
    local phase_tag="$1"; shift
    local phase_log="${PIPELINE_LOG_DIR}/${phase_tag}.log"
    log ">>> [${phase_tag}] $*"
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        log "    (dry-run — skipping execution)"
        return 0
    fi
    # Stream output to both the per-phase log and the main pipeline log.
    set +e
    "$@" 2>&1 | tee -a "${phase_log}" | tee -a "${PIPELINE_LOG}"
    local rc=${PIPESTATUS[0]}
    set -e
    if [[ $rc -ne 0 ]]; then
        log "<<< [${phase_tag}] FAILED with exit code ${rc}"
        return "${rc}"
    fi
    log "<<< [${phase_tag}] OK"
    return 0
}

has_phase() {
    local needle="$1"
    for p in "${PHASES[@]}"; do
        [[ "${p}" == "${needle}" ]] && return 0
    done
    return 1
}

# ---------- Phase 1 — Baseline ----------------------------------------------
if has_phase 1; then
    log ""
    log "### Phase 1 — Baseline training (Ultralytics defaults, ${P1_EPOCHS} ep, patience=${P1_PATIENCE})"
    run_cmd phase1 python "${YOLO_SEG_DIR}/train_baseline_models.py" \
        --models "${MODELS[@]}" \
        --data "${DATA_YAML}" \
        --device "${GPU_DEVICE_IDS}" \
        --project "${PROJECT}" \
        --epochs "${P1_EPOCHS}" \
        --patience "${P1_PATIENCE}" \
        ${FORCE_FLAG}

    log "### Phase 1 — Collecting baseline metrics into pipeline_summary/"
    run_cmd phase1_collect python "${YOLO_SEG_DIR}/collect_phase_metrics.py" \
        --phase baseline \
        --models "${MODELS[@]}" \
        --project "${PROJECT}"
fi

# ---------- Phase 2 — HPO ---------------------------------------------------
if has_phase 2; then
    log ""
    log "### Phase 2 — HPO via model.tune() (space=${HPO_SPACE}, iters=${HPO_ITERATIONS}, ep/trial=${HPO_EPOCHS_PER_TRIAL})"
    # Redireciona o output do tuner para o diretório esperado por
    # train_all_models.py e train_all_models_cv.py:
    #   <project>/hpo/hpo_v3/tune_isic_2018_task_1_<model>/best_hyperparameters.yaml
    HPO_PROJECT="${PROJECT}/hpo/hpo_v3"
    mkdir -p "${HPO_PROJECT}"
    run_cmd phase2 python "${YOLO_SEG_DIR}/tune_all_models_v2.py" \
        --models "${MODELS[@]}" \
        --data "${DATA_YAML}" \
        --device "${GPU_DEVICE_IDS}" \
        --project "${HPO_PROJECT}" \
        --space "${HPO_SPACE}" \
        --iterations "${HPO_ITERATIONS}" \
        --epochs "${HPO_EPOCHS_PER_TRIAL}" \
        --patience "${HPO_PATIENCE}" \
        ${FORCE_FLAG}
fi

# ---------- Phase 3 — Optimized single-split --------------------------------
if has_phase 3; then
    log ""
    log "### Phase 3 — Optimized single-split fine-tuning (uses best_hyperparameters.yaml)"
    run_cmd phase3 python "${YOLO_SEG_DIR}/train_all_models.py" \
        --models "${MODELS[@]}" \
        --data "${DATA_YAML}" \
        --device "${GPU_DEVICE_IDS}" \
        --project "${PROJECT}" \
        ${FORCE_FLAG}

    log "### Phase 3 — Collecting optimized metrics into pipeline_summary/"
    run_cmd phase3_collect python "${YOLO_SEG_DIR}/collect_phase_metrics.py" \
        --phase optimized \
        --models "${MODELS[@]}" \
        --project "${PROJECT}"
fi

# ---------- Phase 4 — Cross-Validation --------------------------------------
if has_phase 4; then
    log ""
    log "### Phase 4 — ${CV_K_FOLDS}-Fold CV (seed=${CV_SEED}, ${CV_EPOCHS} ep, patience=${CV_PATIENCE})"
    run_cmd phase4 python "${YOLO_SEG_DIR}/train_all_models_cv.py" \
        --models "${MODELS[@]}" \
        --data "${DATA_YAML}" \
        --device "${GPU_DEVICE_IDS}" \
        --project "${PROJECT}" \
        --k-folds "${CV_K_FOLDS}" \
        --seed "${CV_SEED}" \
        --epochs "${CV_EPOCHS}" \
        --patience "${CV_PATIENCE}" \
        ${FORCE_FLAG}

    log "### Phase 4 — Consolidating CV results across models"
    run_cmd phase4_consolidate python "${YOLO_SEG_DIR}/consolidate_cv_results.py" \
        --models "${MODELS[@]}" \
        --project "${PROJECT}"
fi

log ""
log "=============================================================="
log "Pipeline finished. Per-phase logs: ${PIPELINE_LOG_DIR}/"
log "Consolidated artefacts: ${PROJECT}/pipeline_summary/"
log "=============================================================="
