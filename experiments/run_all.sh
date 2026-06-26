#!/usr/bin/env bash
# [LEGACY] Use reproduce_paper.sh instead. This script runs only Stage 1 (CLIP benchmark).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

echo "[1/2] Run CLIP-family baseline experiments"
python "$ROOT/experiments/main_clip.py" \
	--results_dir "$ROOT/results/baseline" \
	--experiment all_strategies

echo "[2/2] Top-down workflow artifacts available under:"
echo "  $ROOT/results/paper"
echo "  $ROOT/results/mllm"
echo ""
echo "Optional Stage-3 rerun: python $ROOT/experiments/main_mllm.py --results_dir $ROOT/results/mllm_runtime"

echo "Done. Stage-1 outputs: $ROOT/results/baseline"
