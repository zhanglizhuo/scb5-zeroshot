#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "[1/3] Reproduce E1 core rows"
python "$ROOT/experiments/main_clip.py"

echo "[2/3] Reproduce E11 core rows"
python "$ROOT/experiments/main_mllm.py"

echo "[3/3] Done. See precomputed JSON: $ROOT/results/baseline_results.json"
