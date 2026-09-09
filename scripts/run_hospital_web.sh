#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export SAM_WEB_OUTPUT_DIR="${SAM_WEB_OUTPUT_DIR:-${REPO_ROOT}/hospital_cases}"
export SAM_CHECKPOINT="${SAM_CHECKPOINT:-${REPO_ROOT}/web_ui/checkpoints/lung_sam2_hiera_l.pt}"
export SAM_BACKEND="${SAM_BACKEND:-sam2}"
export SAM_DEVICE="${SAM_DEVICE:-cuda:0}"
export GRADIO_SERVER_NAME="${GRADIO_SERVER_NAME:-0.0.0.0}"
export GRADIO_SERVER_PORT="${GRADIO_SERVER_PORT:-7860}"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}localhost,127.0.0.1,0.0.0.0"

exec python -u web_ui/app.py
