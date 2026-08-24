#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_PATH="${REPO_ROOT}/scripts/run_dca_ring_sam2.sh"
DATASET_ROOT="${DCA_RING_DATASET_ROOT:-/data_new/moyancheng/dataset/dca_dataset_ring}"
SAM2_ENV="${SAM2_ENV:-/data_new/moyancheng/envs/segllm}"
GPU="${GPU:-1}"
TRAIN_MODE="${TRAIN_MODE:-decoder_only}"
EPOCHS="${EPOCHS:-100}"
LR="${LR:-1e-3}"
BATCH_SIZE="${BATCH_SIZE:-4}"
INPUT_SIZE="${INPUT_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-4}"
WANDB_PROJECT="${WANDB_PROJECT:-sam2-dca-ring}"
WANDB_MODE="${WANDB_MODE:-online}"
SEED="${SEED:-3407}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="${RUN_NAME:-dca_ring_sam2_${TRAIN_MODE}_${EPOCHS}ep_lr1e-3_bs4_${TIMESTAMP}}"
RUN_DIR="${RUN_DIR:-${REPO_ROOT}/tmp/${RUN_NAME}}"
SCREEN_NAME="${SCREEN_NAME:-}"
ORIGINAL_ARGS=("$@")

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --train-mode) TRAIN_MODE="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --lr) LR="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --input-size) INPUT_SIZE="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    --screen-name) SCREEN_NAME="$2"; shift 2 ;;
    --wandb-project) WANDB_PROJECT="$2"; shift 2 ;;
    --wandb-mode) WANDB_MODE="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "${SCREEN_NAME}" ]]; then
  SCREEN_NAME="${RUN_NAME}"
fi

if [[ -z "${DCA_RING_SAM2_IN_SCREEN:-}" && -z "${STY:-}" ]]; then
  if [[ -e "${RUN_DIR}" ]]; then
    RUN_DIR="${RUN_DIR}_$(date +%Y%m%d_%H%M%S)"
  fi
  if screen -ls | grep -q "[.]${SCREEN_NAME}[[:space:]]"; then
    SCREEN_NAME="${SCREEN_NAME}_$(date +%Y%m%d_%H%M%S)"
  fi
  mkdir -p "${RUN_DIR}/logs"
  screen -dmS "${SCREEN_NAME}" env \
    DCA_RING_SAM2_IN_SCREEN=1 \
    DATASET_ROOT="${DATASET_ROOT}" \
    SAM2_ENV="${SAM2_ENV}" \
    GPU="${GPU}" \
    TRAIN_MODE="${TRAIN_MODE}" \
    EPOCHS="${EPOCHS}" \
    LR="${LR}" \
    BATCH_SIZE="${BATCH_SIZE}" \
    INPUT_SIZE="${INPUT_SIZE}" \
    NUM_WORKERS="${NUM_WORKERS}" \
    WANDB_PROJECT="${WANDB_PROJECT}" \
    WANDB_MODE="${WANDB_MODE}" \
    SEED="${SEED}" \
    RUN_NAME="${RUN_NAME}" \
    RUN_DIR="${RUN_DIR}" \
    SCREEN_NAME="${SCREEN_NAME}" \
    bash "${SCRIPT_PATH}" \
    "${ORIGINAL_ARGS[@]}"
  echo "Started screen ${SCREEN_NAME}"
  echo "Output: ${RUN_DIR}"
  echo "Log: ${RUN_DIR}/screen.log"
  exit 0
fi

mkdir -p "${RUN_DIR}/logs"
exec > >(tee -a "${RUN_DIR}/screen.log") 2>&1
cd "${REPO_ROOT}"

if [[ -f "/home/moyancheng/miniconda3/etc/profile.d/conda.sh" ]]; then
  source /home/moyancheng/miniconda3/etc/profile.d/conda.sh
  conda activate "${SAM2_ENV}"
fi

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_ROOT}/external/sam2:${REPO_ROOT}:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE}"
export WANDB_DIR="${RUN_DIR}/wandb"
mkdir -p "${WANDB_DIR}"

echo "Repo: ${REPO_ROOT}"
echo "Dataset: ${DATASET_ROOT}"
echo "Run: ${RUN_NAME}"
echo "Output: ${RUN_DIR}"
echo "GPU: ${GPU}"
echo "Mode: ${TRAIN_MODE}"
echo "Epochs/LR/BS/size: ${EPOCHS}/${LR}/${BATCH_SIZE}/${INPUT_SIZE}"
echo "Started at: $(date)"

python -u "${REPO_ROOT}/scripts/run_dca_ring_sam2.py" \
  --dataset-root "${DATASET_ROOT}" \
  --run-dir "${RUN_DIR}" \
  --run-name "${RUN_NAME}" \
  --gpu "${GPU}" \
  --train-mode "${TRAIN_MODE}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --batch-size "${BATCH_SIZE}" \
  --input-size "${INPUT_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-mode "${WANDB_MODE}" \
  --seed "${SEED}"

CHECKPOINT_DIR="${RUN_DIR}/checkpoints"
if [[ -f "${CHECKPOINT_DIR}/checkpoint.pt" ]]; then
  cp "${CHECKPOINT_DIR}/checkpoint.pt" "${CHECKPOINT_DIR}/last.pt"
fi
BEST_SOURCE="${CHECKPOINT_DIR}/val_all_seg_slice_iou_mean.pt"
if [[ -f "${BEST_SOURCE}" ]]; then
  cp "${BEST_SOURCE}" "${CHECKPOINT_DIR}/best.pt"
fi
echo "Finished at: $(date)"
