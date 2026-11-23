#!/usr/bin/env bash
# Pre-train one model size on the twelve datasets (Table 4, pre-trained rows).
# Usage: bash scripts/pretrain.sh ti|s|m|l        (8 GPUs x batch 20; set NGPU to change the number of GPUs)
set -e
SIZE=${1:-ti}
NGPU=${NGPU:-8}
MULTI=$( (( NGPU > 1 )) && echo --multi_gpu || true )
accelerate launch $MULTI --num_processes "$NGPU" --mixed_precision no \
  train.py --config "configs/scoformer_${SIZE}.yaml" --stage pretrain
