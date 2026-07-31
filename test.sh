#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 CHECKPOINT [CONFIG]" >&2
  exit 2
fi

CHECKPOINT=$1
CONFIG=${2:-configs/rtdetrv2_obb/rtdetrv2_obb_hgnetv2_x_diorr.yml}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
MASTER_PORT=${MASTER_PORT:-7001}

torchrun --master_port="${MASTER_PORT}" --nproc_per_node="${NPROC_PER_NODE}" \
  train.py -c "${CONFIG}" --test-only -r "${CHECKPOINT}"
