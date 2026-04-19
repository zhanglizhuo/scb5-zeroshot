#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

echo "[1/2] Run CLIP-family baseline experiments"
python "$ROOT/experiments/main_clip.py" \
	--results_dir "$ROOT/results/baseline" \
	--experiment all_strategies

echo "[2/2] Precomputed paper results available under:"
echo "  $ROOT/results/paper"
echo "  $ROOT/results/mllm"
echo ""
echo "Optional cross-family rerun: python $ROOT/experiments/main_mllm.py --results_dir $ROOT/results/mllm_runtime"

echo "Done. Baseline outputs: $ROOT/results/baseline"
