#!/usr/bin/env bash
# FT-500 of a pre-trained model on each of the twelve datasets, one GPU per run (Table 4, fine-tuned rows).
# Usage: bash scripts/finetune_all.sh ti|s|m|l [pre-trained checkpoint]
# GPUS="0 1 2 3" runs four datasets at a time, one per GPU (default: GPU 0 only).
set -e
SIZE=${1:-ti}
INIT=${2:-runs/scoformer_${SIZE}_pretrain/model.pth}
read -r -a GPU_LIST <<< "${GPUS:-0}"
DATASETS=(ns2d_fno_1e-5 ns2d_fno_1e-4 ns2d_fno_1e-3
          ns2d_pdb_M1_eta1e-1_zeta1e-1 ns2d_pdb_M1_eta1e-2_zeta1e-2
          ns2d_pdb_M1e-1_eta1e-1_zeta1e-1 ns2d_pdb_M1e-1_eta1e-2_zeta1e-2
          swe_pdb dr_pdb ns2d_pda ns2d_cond_pda cfdbench)
FAILED=()
run_batch() {                                      # wait for the running jobs and record the failed datasets
  for j in "${!PIDS[@]}"; do wait "${PIDS[$j]}" || FAILED+=("${NAMES[$j]}"); done
  PIDS=(); NAMES=()
}
PIDS=(); NAMES=()
for i in "${!DATASETS[@]}"; do
  gpu=${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}
  CUDA_VISIBLE_DEVICES=$gpu python train.py --config "configs/scoformer_${SIZE}.yaml" --stage finetune \
    --dataset "${DATASETS[$i]}" --init "$INIT" > /dev/null &
  PIDS+=($!); NAMES+=("${DATASETS[$i]}")
  if (( ${#PIDS[@]} == ${#GPU_LIST[@]} )); then run_batch; fi
done
run_batch
python scripts/summarize.py "$SIZE"
if (( ${#FAILED[@]} )); then echo "failed: ${FAILED[*]} (see the error messages above)" >&2; exit 1; fi
