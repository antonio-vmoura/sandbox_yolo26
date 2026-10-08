#!/usr/bin/env bash
# =============================================================================
# run_pipeline.sh — Master orchestrator for the YOLO26-seg study on the
# ISIC 2018 Task 1 dataset (5-phase protocol).
#
#   Phase 0 — Dataset               (YOLO-seg dataset built from the RAW official
#                                    ISIC 2018 Task 1 release: 2,594 / 100 /
#                                    1,000 images, asserted; idempotent)
#   Phase 1 — Baseline training     (base setup + Ultralytics default HPs;
#                                    train/val split)
#   Phase 2 — Baseline 5-fold CV    (Phase 1 protocol; train+val pool; the test
#                                    set is excluded and verified isolated;
#                                    DSC/JSI of each fold on its held-out fold)
#   Phase 3 — HPO                   (seeded Ultralytics GA; resumable; retried
#                                    automatically on exit code 75 = GPU down)
#   Phase 4 — Optimised fine-tuning (Phase 3 HPs; train/val split)
#   Phase 5 — Test set evaluation   (Baseline AND Optimised: accuracy, DSC/JSI,
#                                    batch=1 efficiency in FP32 and FP16, final
#                                    report in <project>/summary/)
#
# Every phase shares one base setup (MuSGD, cos_lr, nbs=64, close_mosaic=10,
# amp=False, seed=0; epochs=120 / patience=120 = no early stopping, for
# Phases 1, 2 and 4; HPO trials 30 / 30), defined
# once in yolo26_seg/common.py, so the tuned hyperparameters are the only
# variable between Baseline (default HPs) and Optimised (tuned HPs).
#
# Fault tolerance: every Python step is idempotent and resumable. Re-running
# the same command continues where it stopped (interrupted trainings resume
# from last.pt; the HPO resumes from its last completed trial). Use --force to
# start a phase over (previous outputs are moved to *.bak-<UTC>, not deleted).
#
# All commands assume the script runs *inside* the ``yolo26_ft`` Docker
# container with the standard volume mounts (datasets, logs, yolo26_seg,
# utils, cache). See README.md for the full ``docker run`` invocation.
# =============================================================================
set -euo pipefail

# ---------- Defaults ---------------------------------------------------------
DATA_YAML="${DATA_YAML:-/workspace/datasets/isic2018_task1_official/data.yaml}"
# Raw official ISIC 2018 Task 1 release (Phase 0 input; mount it read-only here).
RAW_DIR="${RAW_DIR:-/workspace/raw}"
# Every artefact of this study lives under LOGS_ROOT/PIPELINE_NAME, isolated
# from older runs already present in LOGS_ROOT.
LOGS_ROOT="${LOGS_ROOT:-/workspace/logs}"
PIPELINE_NAME="${PIPELINE_NAME:-pipeline_final_v1}"
# Detect whether PROJECT was pre-set via env var so we don't silently
# clobber it during the LOGS_ROOT/PIPELINE_NAME recomposition below.
if [[ -n "${PROJECT:-}" ]]; then
    PROJECT_FORCED=1
else
    PROJECT_FORCED=0
fi
PROJECT="${PROJECT:-${LOGS_ROOT}/${PIPELINE_NAME}}"
GPU_DEVICE_IDS="${GPU_DEVICE_IDS:-0,1}"
# Phase 5 runs on a single GPU (batch=1 latency). Default: first training GPU
# (resolved after CLI parsing).
BENCH_DEVICE="${BENCH_DEVICE:-}"
MODELS_DEFAULT=(nano small medium large xlarge)
MODELS=("${MODELS_DEFAULT[@]}")
PHASES=(0 1 2 3 4 5)

# Training budget of Phases 1, 2 and 4 (empty = defaults in common.py:
# 120 epochs / patience 120). Override ONLY for smoke tests — the same values
# are always passed to all three phases.
TRAIN_EPOCHS="${TRAIN_EPOCHS:-}"
TRAIN_PATIENCE="${TRAIN_PATIENCE:-}"

# Phase 2 (CV)
CV_K_FOLDS="${CV_K_FOLDS:-5}"
CV_SEED="${CV_SEED:-0}"

# Phase 3 (HPO)
HPO_SPACE="${HPO_SPACE:-refined}"
HPO_ITERATIONS="${HPO_ITERATIONS:-30}"
HPO_EPOCHS_PER_TRIAL="${HPO_EPOCHS_PER_TRIAL:-30}"
HPO_PATIENCE="${HPO_PATIENCE:-30}"   # = epochs per trial: no early stopping
# Micro-batch por trial. Vazio (default) = common.MICRO_BATCH por modelo (16;
# xlarge 8): o mesmo micro-batch com que cada modelo realmente treina em FP32
# numa GPU de 32 GB — um batch que não cabe é reduzido silenciosamente pelo
# Ultralytics. Com nbs=64 o batch efetivo do passo de otimização é 64 sempre.
HPO_BATCH="${HPO_BATCH:-}"
# Retry loop on exit code 75 (GPU/driver unavailable): number of retries after
# the first attempt, and the wait (seconds) between attempts.
HPO_MAX_RETRIES="${HPO_MAX_RETRIES:-5}"
HPO_RETRY_WAIT="${HPO_RETRY_WAIT:-600}"
# Same retry policy for every other GPU step (training, pixel evaluation,
# Phase 5): a step that fails while the GPU is unhealthy is retried.
GPU_STEP_MAX_RETRIES="${GPU_STEP_MAX_RETRIES:-${HPO_MAX_RETRIES}}"
GPU_STEP_RETRY_WAIT="${GPU_STEP_RETRY_WAIT:-${HPO_RETRY_WAIT}}"

# Phase 5 (test set)
EVAL_PRECISIONS="${EVAL_PRECISIONS:-fp32 fp16}"

FORCE_FLAG=""
DRY_RUN=0

#: Exit code used by tune_all_models_v2.py for a transient GPU/driver failure.
EXIT_GPU_UNAVAILABLE=75

YOLO_SEG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/yolo26_seg"
# When mounted in Docker, the canonical path is /workspace/yolo26_seg.
if [[ -d /workspace/yolo26_seg ]]; then
    YOLO_SEG_DIR=/workspace/yolo26_seg
fi

usage() {
    cat <<EOF
Usage: $0 [options]

Options:
  --phases "0 1 2 3 4 5"      Subset of phases to run (default: all; 0 is skipped when up to date).
  --models "n s m l x"        Subset of model sizes. Accepts
                              {nano,small,medium,large,xlarge} or {n,s,m,l,x}.
  --data PATH                 data.yaml with train/val/test. (env: DATA_YAML)
  --logs-root PATH            Parent dir of all pipeline runs.
                              (env: LOGS_ROOT, default /workspace/logs)
  --pipeline-name NAME        Sub-directory under LOGS_ROOT for THIS study.
                              (env: PIPELINE_NAME, default pipeline_final_v1)
  --project PATH              Explicit root (overrides LOGS_ROOT/PIPELINE_NAME).
                              (env: PROJECT)
  --device "0,1"              GPU IDs for training (DDP). (env: GPU_DEVICE_IDS)
  --bench-device ID           Single GPU for Phase 5. (env: BENCH_DEVICE,
                              default: first ID of --device)
  --hpo-batch INT             Micro-batch per HPO trial (default: per model,
                              common.MICRO_BATCH — 16, xlarge 8). (env: HPO_BATCH)
  --epochs INT / --patience INT
                              Smoke-test budget for Phases 1, 2 AND 4 together.
                              (env: TRAIN_EPOCHS / TRAIN_PATIENCE)
  --force                     Start the selected phases over (old outputs are
                              moved to *.bak-<UTC>).
  --dry-run                   Print commands without executing them.
  -h, --help                  Show this help and exit.

Environment variables (override defaults):
  DATA_YAML, LOGS_ROOT, PIPELINE_NAME, PROJECT, GPU_DEVICE_IDS, BENCH_DEVICE,
  TRAIN_EPOCHS, TRAIN_PATIENCE, CV_K_FOLDS, CV_SEED,
  HPO_SPACE, HPO_ITERATIONS, HPO_EPOCHS_PER_TRIAL, HPO_PATIENCE, HPO_BATCH,
  HPO_MAX_RETRIES, HPO_RETRY_WAIT, GPU_STEP_MAX_RETRIES, GPU_STEP_RETRY_WAIT,
  EVAL_PRECISIONS

Exit codes: 0 success; 75 HPO (or another GPU step) gave up after its
            retries because the GPU stayed unavailable;
            anything else = exit code of the failing step.
EOF
}

# ---------- CLI parsing ------------------------------------------------------
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
        --bench-device) BENCH_DEVICE="$2"; shift 2 ;;
        --hpo-batch) HPO_BATCH="$2"; shift 2 ;;
        --epochs) TRAIN_EPOCHS="$2"; shift 2 ;;
        --patience) TRAIN_PATIENCE="$2"; shift 2 ;;
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

BENCH_DEVICE="${BENCH_DEVICE:-${GPU_DEVICE_IDS%%,*}}"

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

for p in "${PHASES[@]}"; do
    if [[ ! "${p}" =~ ^[0-5]$ ]]; then
        echo "[erro] Fase inválida: '${p}'. Use números de 0 a 5." >&2
        exit 2
    fi
done

# Same budget flags for Phases 1, 2 and 4 (empty array = common.py defaults).
BUDGET_ARGS=()
[[ -n "${TRAIN_EPOCHS}" ]] && BUDGET_ARGS+=(--epochs "${TRAIN_EPOCHS}")
[[ -n "${TRAIN_PATIENCE}" ]] && BUDGET_ARGS+=(--patience "${TRAIN_PATIENCE}")
FORCE_ARGS=()
[[ -n "${FORCE_FLAG}" ]] && FORCE_ARGS+=("${FORCE_FLAG}")

RUN_TS="$(date -u +%Y%m%dT%H%M%SZ)"
PIPELINE_LOG_DIR="${PROJECT}/pipeline_runs/${RUN_TS}"
mkdir -p "${PIPELINE_LOG_DIR}"
PIPELINE_LOG="${PIPELINE_LOG_DIR}/pipeline.log"

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%SZ)" "$*" | tee -a "${PIPELINE_LOG}"; }

# ---------- GPU sanity check -------------------------------------------------
# Verifica, em <1 s, se o driver NVIDIA e o torch.cuda ainda estão saudáveis
# antes de cada fase. Uma queda do nvidia.ko (NVRM/NVML) no host durante um
# run longo faz as fases seguintes morrerem com erros crípticos; falhar cedo,
# aqui, evita esse desperdício e dá uma mensagem acionável.
#
# Em modo --dry-run, a checagem é pulada (não há GPU envolvida).
gpu_sanity_check() {
    local tag="$1"
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        log "    [gpu_sanity_check:${tag}] pulado (dry-run)"
        return 0
    fi
    log "    [gpu_sanity_check:${tag}] verificando driver NVIDIA e torch.cuda..."
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        log "    [gpu_sanity_check:${tag}] FALHOU — 'nvidia-smi' não está no PATH dentro do container."
        log "    Verifique a flag --gpus do 'docker run' e a instalação do NVIDIA Container Toolkit no host."
        return 10
    fi
    if ! nvidia-smi -L >/dev/null 2>&1; then
        log "    [gpu_sanity_check:${tag}] FALHOU — 'nvidia-smi -L' não retornou GPUs (driver caiu?)."
        log "    Recupere o driver no host (rmmod/modprobe nvidia* ou reboot) e re-execute o pipeline."
        return 11
    fi
    if ! python - <<'PY' >/dev/null 2>&1
import sys
import torch
sys.exit(0 if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else 1)
PY
    then
        log "    [gpu_sanity_check:${tag}] FALHOU — torch.cuda.is_available() retornou False."
        log "    Sintoma clássico de NVML quebrado. Recupere o driver no host e re-execute."
        return 12
    fi
    local n_gpus
    n_gpus=$(nvidia-smi -L | wc -l)
    log "    [gpu_sanity_check:${tag}] OK — ${n_gpus} GPU(s) visíveis para torch.cuda."
    return 0
}

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

# Run a step and abort the pipeline with its exit code if it fails.
run_or_die() {
    local rc=0
    run_cmd "$@" || rc=$?
    if [[ $rc -ne 0 ]]; then
        log "Pipeline aborted at step '$1' (exit ${rc}). Fix the cause and re-run the same command to resume."
        exit "${rc}"
    fi
}

# Run a GPU step with a sanity check before every attempt. If the step fails
# and the GPU is then unhealthy (e.g. "Can't initialize NVML" after a host
# daemon-reload), wait GPU_STEP_RETRY_WAIT s and retry, up to
# GPU_STEP_MAX_RETRIES times; the steps are idempotent, so a retry resumes
# (training from last.pt, evaluations skip finished models). A failure with a
# healthy GPU is a real error and aborts at once. --force is dropped after the
# first attempt so that retries resume instead of starting over.
run_gpu_step() {
    local tag="$1"; shift
    local -a cmd=("$@") next
    local max_attempts=$((GPU_STEP_MAX_RETRIES + 1)) attempt rc a
    for ((attempt = 1; attempt <= max_attempts; attempt++)); do
        rc=0
        if ! gpu_sanity_check "${tag}#${attempt}"; then
            rc=${EXIT_GPU_UNAVAILABLE}
        else
            run_cmd "${tag}" "${cmd[@]}" || rc=$?
            [[ ${rc} -eq 0 ]] && return 0
            if gpu_sanity_check "${tag}#${attempt}-post-failure"; then
                log "Pipeline aborted at step '${tag}' (exit ${rc}; GPU healthy, so not a GPU failure). Fix the cause and re-run the same command to resume."
                exit "${rc}"
            fi
        fi
        next=()
        for a in "${cmd[@]}"; do [[ "${a}" == "--force" ]] || next+=("${a}"); done
        cmd=("${next[@]}")
        if [[ ${attempt} -ge ${max_attempts} ]]; then
            log "[erro] ${tag}: GPU indisponível após ${max_attempts} tentativa(s). Recupere o driver / reinicie o container e re-execute o mesmo comando para retomar."
            exit "${EXIT_GPU_UNAVAILABLE}"
        fi
        log "    [${tag}] GPU indisponível (exit ${rc}) — tentativa ${attempt}/${max_attempts}; nova tentativa em ${GPU_STEP_RETRY_WAIT}s."
        [[ "${DRY_RUN}" -eq 1 ]] || sleep "${GPU_STEP_RETRY_WAIT}"
    done
}

has_phase() {
    local needle="$1"
    for p in "${PHASES[@]}"; do
        [[ "${p}" == "${needle}" ]] && return 0
    done
    return 1
}

# Abort unless a Phase 1/4 training run has completed (run_state.json).
require_complete_run() {
    local run_dir="$1" label="$2"
    [[ "${DRY_RUN}" -eq 1 ]] && return 0
    if ! grep -q '"status": "complete"' "${run_dir}/run_state.json" 2>/dev/null; then
        log "[erro] ${label}: run incompleto ou ausente em ${run_dir} — execute a fase correspondente antes."
        exit 3
    fi
}

log "=============================================================="
log "YOLO26-seg ISIC 2018 Task 1 — 5-Phase Pipeline"
log "=============================================================="
log "  data           = ${DATA_YAML}"
log "  project        = ${PROJECT}"
log "  device         = ${GPU_DEVICE_IDS}   (Phase 5 bench device = ${BENCH_DEVICE})"
log "  models         = ${MODELS[*]}"
log "  phases         = ${PHASES[*]}"
log "  train budget   = ${TRAIN_EPOCHS:-120 (default)} epochs, patience ${TRAIN_PATIENCE:-120 (default)}  [Phases 1, 2, 4]"
log "  cv             = k=${CV_K_FOLDS}, seed=${CV_SEED}"
log "  hpo            = space=${HPO_SPACE}, trials=${HPO_ITERATIONS}, ep/trial=${HPO_EPOCHS_PER_TRIAL}, batch=${HPO_BATCH:-per-model (16; xlarge 8)}, retries=${HPO_MAX_RETRIES} x ${HPO_RETRY_WAIT}s"
log "  gpu retries    = ${GPU_STEP_MAX_RETRIES} x ${GPU_STEP_RETRY_WAIT}s  [Phases 1, 2, 4, 5 GPU steps]"
log "  precisions     = ${EVAL_PRECISIONS}  [Phase 5 efficiency]"
log "  force          = ${FORCE_FLAG:-<off>}"
log "  yolo_seg_dir   = ${YOLO_SEG_DIR}"
log "  pipeline_log   = ${PIPELINE_LOG}"
if [[ -n "${TRAIN_EPOCHS}${TRAIN_PATIENCE}" ]]; then
    log "  [aviso] orçamento de treino diferente do protocolo (120/120) — use apenas para smoke tests."
fi
log "--------------------------------------------------------------"

# ---------- Phase 0 — Dataset from the raw official release -----------------
if has_phase 0; then
    log ""
    log "### Phase 0 — YOLO-seg dataset from the raw ISIC 2018 Task 1 release (${RAW_DIR}; idempotent)"
    run_or_die phase0 python "${YOLO_SEG_DIR}/prepare_dataset.py" \
        --raw "${RAW_DIR}" --out "$(dirname "${DATA_YAML}")" "${FORCE_ARGS[@]}"
fi

# ---------- Phase 1 — Baseline training -------------------------------------
if has_phase 1; then
    log ""
    log "### Phase 1 — Baseline training (base setup + Ultralytics default HPs)"
    run_gpu_step phase1 python "${YOLO_SEG_DIR}/train_baseline_models.py" \
        --models "${MODELS[@]}" \
        --data "${DATA_YAML}" \
        --device "${GPU_DEVICE_IDS}" \
        --project "${PROJECT}" \
        "${BUDGET_ARGS[@]}" "${FORCE_ARGS[@]}"
    run_or_die phase1_collect python "${YOLO_SEG_DIR}/collect_phase_metrics.py" \
        --phase phase1 --models "${MODELS[@]}" --project "${PROJECT}"
fi

# ---------- Phase 2 — Baseline cross-validation -----------------------------
if has_phase 2; then
    log ""
    log "### Phase 2 — Baseline ${CV_K_FOLDS}-fold CV on train+val (seed=${CV_SEED}; test isolated)"
    run_gpu_step phase2 python "${YOLO_SEG_DIR}/train_all_models_cv.py" \
        --protocol baseline \
        --models "${MODELS[@]}" \
        --data "${DATA_YAML}" \
        --device "${GPU_DEVICE_IDS}" \
        --project "${PROJECT}" \
        --k-folds "${CV_K_FOLDS}" \
        --seed "${CV_SEED}" \
        "${BUDGET_ARGS[@]}" "${FORCE_ARGS[@]}"
    run_or_die phase2_consolidate python "${YOLO_SEG_DIR}/consolidate_cv_results.py" \
        --protocol baseline --models "${MODELS[@]}" --project "${PROJECT}"
    log "### Phase 2 — Pixel-level DSC/JSI of each fold on its held-out fold (device ${BENCH_DEVICE})"
    run_gpu_step phase2_pixels python "${YOLO_SEG_DIR}/evaluate_cv_pixels.py" \
        --protocol baseline \
        --models "${MODELS[@]}" \
        --device "${BENCH_DEVICE}" \
        --project "${PROJECT}" \
        "${FORCE_ARGS[@]}"
fi

# ---------- Phase 3 — HPO (fault-tolerant, retried on exit 75) --------------
if has_phase 3; then
    log ""
    log "### Phase 3 — HPO (space=${HPO_SPACE}, ${HPO_ITERATIONS} trials x ${HPO_EPOCHS_PER_TRIAL} ep, seeded, resumable)"
    # --force only applies to the first attempt; retries must resume, not restart.
    HPO_FORCE_ARGS=("${FORCE_ARGS[@]}")
    max_attempts=$((HPO_MAX_RETRIES + 1))
    for ((attempt = 1; attempt <= max_attempts; attempt++)); do
        rc=0
        if ! gpu_sanity_check "phase3#${attempt}"; then
            rc=${EXIT_GPU_UNAVAILABLE}
        else
            run_cmd phase3 python "${YOLO_SEG_DIR}/tune_all_models_v2.py" \
                --models "${MODELS[@]}" \
                --data "${DATA_YAML}" \
                --device "${GPU_DEVICE_IDS}" \
                --project "${PROJECT}" \
                --space "${HPO_SPACE}" \
                --iterations "${HPO_ITERATIONS}" \
                --epochs "${HPO_EPOCHS_PER_TRIAL}" \
                --patience "${HPO_PATIENCE}" \
                ${HPO_BATCH:+--batch "${HPO_BATCH}"} \
                "${HPO_FORCE_ARGS[@]}" || rc=$?
        fi
        HPO_FORCE_ARGS=()
        if [[ ${rc} -eq 0 ]]; then
            break
        fi
        if [[ ${rc} -ne ${EXIT_GPU_UNAVAILABLE} ]]; then
            log "Pipeline aborted at step 'phase3' (exit ${rc}, not a GPU failure). Fix the cause and re-run to resume."
            exit "${rc}"
        fi
        if [[ ${attempt} -ge ${max_attempts} ]]; then
            log "[erro] Phase 3: GPU indisponível após ${max_attempts} tentativa(s). Recupere o driver e re-execute — a busca continua do último trial."
            exit "${EXIT_GPU_UNAVAILABLE}"
        fi
        log "    [phase3] GPU indisponível (exit ${rc}) — tentativa ${attempt}/${max_attempts}; nova tentativa em ${HPO_RETRY_WAIT}s (retoma do último trial)."
        [[ "${DRY_RUN}" -eq 1 ]] || sleep "${HPO_RETRY_WAIT}"
    done

    # The Ultralytics Tuner does not propagate trial failures; this check also
    # rejects incomplete searches (hpo_state.json must be 'complete').
    log "### Phase 3 — Validating HPO outputs (completeness + degenerate trials)"
    run_or_die phase3_validate python "${YOLO_SEG_DIR}/check_hpo_validity.py" \
        --project "${PROJECT}" \
        --models "${MODELS[@]}" \
        --iterations "${HPO_ITERATIONS}"
fi

# ---------- Phase 4 — Optimised fine-tuning ---------------------------------
if has_phase 4; then
    log ""
    log "### Phase 4 — Optimised fine-tuning with the Phase 3 hyperparameters"
    run_gpu_step phase4 python "${YOLO_SEG_DIR}/train_all_models.py" \
        --models "${MODELS[@]}" \
        --data "${DATA_YAML}" \
        --device "${GPU_DEVICE_IDS}" \
        --project "${PROJECT}" \
        "${BUDGET_ARGS[@]}" "${FORCE_ARGS[@]}"
    run_or_die phase4_collect python "${YOLO_SEG_DIR}/collect_phase_metrics.py" \
        --phase phase4 --models "${MODELS[@]}" --project "${PROJECT}"
fi

# ---------- Phase 5 — Test set inference & profiling ------------------------
if has_phase 5; then
    log ""
    log "### Phase 5 — Test set evaluation of Baseline AND Optimised (device ${BENCH_DEVICE})"
    for script in evaluate_test_set.py benchmark_efficiency.py build_final_report.py; do
        if [[ ! -f "${YOLO_SEG_DIR}/${script}" ]]; then
            log "[erro] Phase 5: ${YOLO_SEG_DIR}/${script} não existe (ainda não implementado)."
            exit 4
        fi
    done
    for m in "${MODELS[@]}"; do
        require_complete_run "${PROJECT}/phase1_baseline/yolo26_${m}_baseline" "Baseline ${m}"
        require_complete_run "${PROJECT}/phase4_optimized/yolo26_${m}_optimized" "Optimised ${m}"
    done
    read -r -a PRECISIONS <<<"${EVAL_PRECISIONS}"
    run_gpu_step phase5_accuracy python "${YOLO_SEG_DIR}/evaluate_test_set.py" \
        --models "${MODELS[@]}" \
        --variants baseline optimized \
        --precisions "${PRECISIONS[@]}" \
        --data "${DATA_YAML}" \
        --device "${BENCH_DEVICE}" \
        --project "${PROJECT}" \
        "${FORCE_ARGS[@]}"
    run_gpu_step phase5_efficiency python "${YOLO_SEG_DIR}/benchmark_efficiency.py" \
        --models "${MODELS[@]}" \
        --variants baseline optimized \
        --precisions "${PRECISIONS[@]}" \
        --data "${DATA_YAML}" \
        --device "${BENCH_DEVICE}" \
        --project "${PROJECT}" \
        "${FORCE_ARGS[@]}"
    run_or_die phase5_report python "${YOLO_SEG_DIR}/build_final_report.py" \
        --models "${MODELS[@]}" \
        --project "${PROJECT}"
fi

log ""
log "=============================================================="
log "Pipeline finished. Per-phase logs: ${PIPELINE_LOG_DIR}/"
log "Consolidated artefacts: ${PROJECT}/summary/"
log "=============================================================="
