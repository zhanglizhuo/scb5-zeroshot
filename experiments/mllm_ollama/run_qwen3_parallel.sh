#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

OLLAMA_MODELS_DIR="${OLLAMA_MODELS_DIR:-/usr/share/ollama/.ollama/models}"

# Launch three dedicated Ollama workers pinned to three free GPUs.
# GPU3 is intentionally not used here because it may be occupied by unrelated work.

mkdir -p results/mllm/mllm_qwen_tb
mkdir -p results/mllm/mllm_qwen_hr
mkdir -p results/mllm/mllm_qwen_bt

CUDA_VISIBLE_DEVICES=0 OLLAMA_HOST=127.0.0.1:11438 OLLAMA_MODELS="$OLLAMA_MODELS_DIR" ollama serve > /tmp/ollama_11438.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 OLLAMA_HOST=127.0.0.1:11439 OLLAMA_MODELS="$OLLAMA_MODELS_DIR" ollama serve > /tmp/ollama_11439.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 OLLAMA_HOST=127.0.0.1:11437 OLLAMA_MODELS="$OLLAMA_MODELS_DIR" ollama serve > /tmp/ollama_11437.log 2>&1 &

sleep 3

python3 experiments/main_mllm.py \
  --models qwen35_ollama \
  --qwen35_model qwen3.5:9b \
  --ollama_host http://127.0.0.1:11438 \
  --subsets teacher_behavior \
  --results_dir results/mllm/mllm_qwen_tb > /tmp/qwen_tb.log 2>&1 &

python3 experiments/main_mllm.py \
  --models qwen35_ollama \
  --qwen35_model qwen3.5:9b \
  --ollama_host http://127.0.0.1:11439 \
  --subsets handrise_readwrite \
  --results_dir results/mllm/mllm_qwen_hr > /tmp/qwen_hr.log 2>&1 &

python3 experiments/main_mllm.py \
  --models qwen35_ollama \
  --qwen35_model qwen3.5:9b \
  --ollama_host http://127.0.0.1:11437 \
  --subsets bow_turnhead \
  --results_dir results/mllm/mllm_qwen_bt > /tmp/qwen_bt.log 2>&1 &

echo "Launched qwen3.5 parallel jobs on GPUs 0/1/2."
echo "Logs: /tmp/qwen_tb.log /tmp/qwen_hr.log /tmp/qwen_bt.log"