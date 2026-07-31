#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 CHECKPOINT [CONFIG]" >&2
  exit 2
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
CHECKPOINT=$1
CONFIG=${2:-configs/rtdetrv2_obb/rtdetrv2_obb_hgnetv2_x_diorr.yml}
PYTHON=${PYTHON:-python}
GPU=${GPU:-0}
BATCH_SIZE=${BATCH_SIZE:-1}

cd "${REPO_DIR}"

LOG_DIR=logs/diorr_obb_metrics
OUTPUT_DIR=outputs/diorr_obb_metrics
LOG_FILE="${LOG_DIR}/evaluation.log"
IOU_THRS='[0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95]'

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES="${GPU}" PYTHONUNBUFFERED=1 "${PYTHON}" train.py \
  -c "${CONFIG}" \
  --test-only \
  -r "${CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR}" \
  -u \
  "evaluator.iou_thrs=${IOU_THRS}" \
  "val_dataloader.total_batch_size=${BATCH_SIZE}" \
  2>&1 | tee "${LOG_FILE}"

grep '^OBB_METRICS ' "${LOG_FILE}" | tail -n 1
