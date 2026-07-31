#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/rtdetrv2_obb/rtdetrv2_obb_hgnetv2_x_diorr.yml}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
MASTER_PORT=${MASTER_PORT:-7001}

torchrun --master_port="${MASTER_PORT}" --nproc_per_node="${NPROC_PER_NODE}" \
  train.py -c "${CONFIG}" --seed=0
