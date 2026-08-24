#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_PATH="${REPO_ROOT}/scripts/run_dca_ring_sam_train.sh"
DATASET_FORMAT="${DATASET_FORMAT:-dir_splits}"
DATASET_ROOT="${DCA_RING_DATASET_ROOT:-/data_new/moyancheng/dataset/dca_dataset_ring}"
SAM_CONDA_ENV="${SAM_CONDA_ENV:-/data_new/moyancheng/envs/SAM}"
MODEL_TYPE="${MODEL_TYPE:-vit_b}"
TRAIN_MODE="${TRAIN_MODE:-decoder_only}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
GPU_ID="${GPU_ID:-6}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LR="${LR:-1e-4}"
INPUT_SIZE="${INPUT_SIZE:-}"
SCREEN_NAME="${SCREEN_NAME:-}"
RUN_NAME="${RUN_NAME:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
WANDB_PROJECT="${WANDB_PROJECT:-sam-dca-ring}"
WANDB_MODE="${WANDB_MODE:-online}"
TRAIN_LIMIT="${TRAIN_LIMIT:-}"
VAL_LIMIT="${VAL_LIMIT:-}"
TEST_LIMIT="${TEST_LIMIT:-}"
ORIGINAL_ARGS=("$@")

dataset_name() {
  case "$1" in
    dir_splits) echo "dca_ring" ;;
    manifest_splits) echo "lung" ;;
    *)
      echo "Unknown --dataset-format: $1" >&2
      exit 1
      ;;
  esac
}

default_dataset_root() {
  case "$1" in
    dir_splits) echo "/data_new/moyancheng/dataset/dca_dataset_ring" ;;
    manifest_splits) echo "/data_new/moyancheng/dataset/lung/肺癌勾画/processed_axial_slices_png_0mm/data_spilt" ;;
    *)
      echo "Unknown --dataset-format: $1" >&2
      exit 1
      ;;
  esac
}

default_checkpoint_path() {
  case "$1" in
    vit_b) echo "${REPO_ROOT}/checkpoints/sam_vit_b_01ec64.pth" ;;
    vit_l) echo "${REPO_ROOT}/checkpoints/sam_vit_l_0b3195.pth" ;;
    vit_h) echo "${REPO_ROOT}/checkpoints/sam_vit_h_4b8939.pth" ;;
    *)
      echo "Unknown --model-type: $1" >&2
      exit 1
      ;;
  esac
}

default_input_size() {
  case "$1" in
    vit_h) echo 1024 ;;
    vit_b|vit_l) echo 256 ;;
    *)
      echo "Unknown --model-type: $1" >&2
      exit 1
      ;;
  esac
}

default_batch_size() {
  local model_type="$1"
  local input_size="$2"
  if [[ "${model_type}" == "vit_h" && "${input_size}" -ge 1024 ]]; then
    echo 1
  else
    echo 8
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-format)
      DATASET_FORMAT="$2"
      shift 2
      ;;
    --model-type)
      MODEL_TYPE="$2"
      shift 2
      ;;
    --checkpoint)
      CHECKPOINT_PATH="$2"
      shift 2
      ;;
    --train-mode)
      TRAIN_MODE="$2"
      shift 2
      ;;
    --dataset-root)
      DATASET_ROOT="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --gpu)
      GPU_ID="$2"
      shift 2
      ;;
    --screen-name)
      SCREEN_NAME="$2"
      shift 2
      ;;
    --run-name)
      RUN_NAME="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --input-size)
      INPUT_SIZE="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --num-workers)
      NUM_WORKERS="$2"
      shift 2
      ;;
    --lr)
      LR="$2"
      shift 2
      ;;
    --wandb-mode)
      WANDB_MODE="$2"
      shift 2
      ;;
    --wandb-project)
      WANDB_PROJECT="$2"
      shift 2
      ;;
    --train-limit)
      TRAIN_LIMIT="$2"
      shift 2
      ;;
    --val-limit)
      VAL_LIMIT="$2"
      shift 2
      ;;
    --test-limit)
      TEST_LIMIT="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

DATASET_NAME="$(dataset_name "${DATASET_FORMAT}")"
MODEL_TAG="sam_${MODEL_TYPE}"
if [[ -z "${DCA_RING_DATASET_ROOT:-}" ]]; then
  DATASET_ROOT="$(default_dataset_root "${DATASET_FORMAT}")"
fi
if [[ -z "${INPUT_SIZE}" ]]; then
  INPUT_SIZE="$(default_input_size "${MODEL_TYPE}")"
fi
if [[ -z "${BATCH_SIZE}" ]]; then
  BATCH_SIZE="$(default_batch_size "${MODEL_TYPE}" "${INPUT_SIZE}")"
fi
if [[ -z "${CHECKPOINT_PATH}" ]]; then
  CHECKPOINT_PATH="$(default_checkpoint_path "${MODEL_TYPE}")"
fi
if [[ -z "${SCREEN_NAME}" ]]; then
  SCREEN_NAME="${DATASET_NAME}_${MODEL_TAG}_${INPUT_SIZE}_noprompt_${TRAIN_MODE}_$(date +%Y%m%d_%H%M%S)"
fi
if [[ -z "${OUTPUT_DIR}" ]]; then
  OUTPUT_DIR="${REPO_ROOT}/tmp/${DATASET_NAME}_${MODEL_TAG}_${INPUT_SIZE}_noprompt_${TRAIN_MODE}_$(date +%Y%m%d_%H%M%S)"
fi
if [[ -z "${RUN_NAME}" ]]; then
  RUN_NAME="$(basename "${OUTPUT_DIR}")"
fi

if [[ -z "${DCA_RING_SAM_TRAIN_IN_SCREEN:-}" && -z "${STY:-}" ]]; then
  if [[ -e "${OUTPUT_DIR}" ]]; then
    OUTPUT_DIR="${OUTPUT_DIR}_$(date +%Y%m%d_%H%M%S)"
  fi
  mkdir -p "${OUTPUT_DIR}/logs"
  LOG_FILE="${OUTPUT_DIR}/logs/run.log"
  if screen -ls | grep -q "[.]${SCREEN_NAME}[[:space:]]"; then
    SCREEN_NAME="${SCREEN_NAME}_$(date +%Y%m%d_%H%M%S)"
  fi
  screen -dmS "${SCREEN_NAME}" env \
    DCA_RING_SAM_TRAIN_IN_SCREEN=1 \
    OUTPUT_DIR="${OUTPUT_DIR}" \
    SCREEN_NAME="${SCREEN_NAME}" \
    RUN_NAME="${RUN_NAME}" \
    DATASET_FORMAT="${DATASET_FORMAT}" \
    MODEL_TYPE="${MODEL_TYPE}" \
    TRAIN_MODE="${TRAIN_MODE}" \
    GPU_ID="${GPU_ID}" \
    EPOCHS="${EPOCHS}" \
    INPUT_SIZE="${INPUT_SIZE}" \
    BATCH_SIZE="${BATCH_SIZE}" \
    NUM_WORKERS="${NUM_WORKERS}" \
    LR="${LR}" \
    WANDB_MODE="${WANDB_MODE}" \
    WANDB_PROJECT="${WANDB_PROJECT}" \
    TRAIN_LIMIT="${TRAIN_LIMIT}" \
    VAL_LIMIT="${VAL_LIMIT}" \
    TEST_LIMIT="${TEST_LIMIT}" \
    CHECKPOINT_PATH="${CHECKPOINT_PATH}" \
    SAM_CONDA_ENV="${SAM_CONDA_ENV}" \
    DCA_RING_DATASET_ROOT="${DATASET_ROOT}" \
    bash "${SCRIPT_PATH}" \
    "${ORIGINAL_ARGS[@]}"
  echo "Started screen ${SCREEN_NAME}"
  echo "Output: ${OUTPUT_DIR}"
  echo "Log: ${LOG_FILE}"
  echo "Attach with: screen -r ${SCREEN_NAME}"
  exit 0
fi

mkdir -p "${OUTPUT_DIR}/logs"
exec > >(tee -a "${OUTPUT_DIR}/logs/run.log") 2>&1

cd "${REPO_ROOT}"
echo "Repo: ${REPO_ROOT}"
echo "Dataset root: ${DATASET_ROOT}"
echo "Dataset format: ${DATASET_FORMAT}"
echo "Output: ${OUTPUT_DIR}"
echo "Model: ${MODEL_TAG}"
echo "Train mode: ${TRAIN_MODE}"
echo "Input size: ${INPUT_SIZE}"
echo "Batch size: ${BATCH_SIZE}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "GPU_ID=${GPU_ID}"
echo "Started at: $(date)"

if [[ -d "${SAM_CONDA_ENV}" && -f "/home/moyancheng/miniconda3/etc/profile.d/conda.sh" ]]; then
  source /home/moyancheng/miniconda3/etc/profile.d/conda.sh
  conda activate "${SAM_CONDA_ENV}"
  echo "Activated conda env: ${CONDA_PREFIX}"
else
  echo "SAM conda env not found at ${SAM_CONDA_ENV}; using current Python: $(which python)"
fi

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONUNBUFFERED=1
export WANDB_DIR="${OUTPUT_DIR}/wandb"
export WANDB_MODE="${WANDB_MODE}"
mkdir -p "${WANDB_DIR}"

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available after setting CUDA_VISIBLE_DEVICES.")
print("Visible CUDA devices:", torch.cuda.device_count())
print("Using:", torch.cuda.get_device_name(0))
PY

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "Missing checkpoint: ${CHECKPOINT_PATH}"
  exit 1
fi

PY_ARGS=(
  --dataset-root "${DATASET_ROOT}"
  --dataset-format "${DATASET_FORMAT}"
  --model-type "${MODEL_TYPE}"
  --train-mode "${TRAIN_MODE}"
  --checkpoint "${CHECKPOINT_PATH}"
  --output-dir "${OUTPUT_DIR}"
  --run-name "${RUN_NAME}"
  --wandb-project "${WANDB_PROJECT}"
  --device cuda:0
  --input-size "${INPUT_SIZE}"
  --batch-size "${BATCH_SIZE}"
  --epochs "${EPOCHS}"
  --num-workers "${NUM_WORKERS}"
  --lr "${LR}"
)

if [[ -n "${TRAIN_LIMIT}" ]]; then
  PY_ARGS+=(--train-limit "${TRAIN_LIMIT}")
fi
if [[ -n "${VAL_LIMIT}" ]]; then
  PY_ARGS+=(--val-limit "${VAL_LIMIT}")
fi
if [[ -n "${TEST_LIMIT}" ]]; then
  PY_ARGS+=(--test-limit "${TEST_LIMIT}")
fi

python -u "${REPO_ROOT}/scripts/train_dca_ring_sam.py" "${PY_ARGS[@]}"

echo "Finished at: $(date)"
