#!/bin/bash
# [LEGACY] Use reproduce_paper.sh instead.
# Parallel runner for four models across four GPUs
# - Uses system `python3` (/usr/bin/python3) to avoid virtualenv-specific interpreters
# - For offline-first behavior set `CKPTS_DIR` to a local checkpoints directory
#   e.g. export CKPTS_DIR=/path/to/ckpts
# - Default models: clip, laion, flip, siglip (repository-standard keys)
# Usage:
#   CKPTS_DIR=./ckpts bash run_all_parallel.sh [mode] [prompt]
# Example:
#   CKPTS_DIR=./ckpts bash run_all_parallel.sh zeroshot detailed
# =====================================================

MODE=${1:-"zeroshot"}   # 默认 zeroshot，可传 fewshot / ablation / all
PROMPT=${2:-"detailed"} # 默认 detailed prompt

CKPTS_DIR=${CKPTS_DIR:-"./ckpts"}

echo "Mode: $MODE  Prompt: $PROMPT"
echo "CKPTS_DIR: $CKPTS_DIR"
echo "Starting 4 models in parallel across 4 GPUs using /usr/bin/python3 ..."

#!/usr/bin/env bash

# run_all_parallel.sh — Offline-first parallel launcher
# Launch four parallel processes across 4 GPUs using system python3.
# Expects local checkpoints in CKPTS_DIR; falls back to model hub if missing.
# Usage: CKPTS_DIR=/path/to/ckpts ./run_all_parallel.sh

set -euo pipefail

PYTHON=${PYTHON:-/usr/bin/python3}
CKPTS_DIR=${CKPTS_DIR:-"/home/user/.cache/ckpts"}

# Local checkpoints mapping (offline-first)
CLIP_CKPT=${CKPTS_DIR}/clip/open_clip_pytorch_model.bin
LAION_CKPT=${CKPTS_DIR}/laion/open_clip_pytorch_model.bin
SIGLIP_CKPT=${CKPTS_DIR}/siglip/open_clip_pytorch_model.bin
FLIP_CKPT=${CKPTS_DIR}/flip/open_clip_pytorch_model.bin

# Models to launch (repository standard keys)
GPUS=(0 1 2 3)
MODELS=(clip laion flip siglip)

mkdir -p logs

for i in "${!MODELS[@]}"; do
    GPU=${GPUS[$i]}
    MODEL=${MODELS[$i]}

    CKPT_ARG=""
    case "$MODEL" in
        clip)  CKPT_ARG="--ckpt $CLIP_CKPT" ;; 
        laion) CKPT_ARG="--ckpt $LAION_CKPT" ;; 
        siglip) CKPT_ARG="--ckpt $SIGLIP_CKPT" ;; 
        flip)  CKPT_ARG="--ckpt $FLIP_CKPT" ;; 
    esac

    echo "Starting $MODEL on GPU $GPU"
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON scb5_zeroshot/run_experiment.py \
        --mode e1e2 \
        --model $MODEL \
        $CKPT_ARG \
        --dataset_root datasets_scb/ \
        > logs/${MODEL}_e1e2.log 2>&1 &
done

echo "Launched all jobs. Use 'jobs' to see background jobs." 

wait $PID_CLIP   && echo "✓ CLIP   done" || echo "✗ CLIP   failed"
wait $PID_LAION  && echo "✓ LAION  done" || echo "✗ LAION  failed"
wait $PID_FLIP   && echo "✓ FLIP   done" || echo "✗ FLIP   failed"
wait $PID_SIGLIP && echo "✓ SIGLIP done" || echo "✗ SIGLIP failed"

echo ""
echo "====== All experiments finished ======"
echo "Results saved in ./results/ (per-model JSON files)"
ls -la results/ || true
