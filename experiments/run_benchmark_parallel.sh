#!/usr/bin/env bash

# run_benchmark_parallel.sh — Launch benchmark shards in parallel
# ===============================================================
# Thin wrapper that partitions the model list into shards and runs
# `run_benchmark.sh` for each shard on separate GPUs in parallel.
# Designed for offline-first runs: configure `PYTHON_BIN`, `MODEL_SHARDS`,
# and `GPU_SHARDS` via environment variables as needed.
#
# Usage:
#   MODEL_SHARDS="clip;laion;siglip;eva02,dfn" GPU_SHARDS="0;1;2;3" \
#     PYTHON_BIN=/usr/bin/python3 bash run_benchmark_parallel.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EVAL_DIR="$ROOT"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || echo python3)}"
RESULTS_ROOT="$EVAL_DIR/results/baseline"
LOG_DIR="$EVAL_DIR/logs"
BATCH_SIZE="${BATCH_SIZE:-16}"

MODEL_SHARDS_RAW="${MODEL_SHARDS:-clip;laion;siglip;eva02,dfn}"
GPU_SHARDS_RAW="${GPU_SHARDS:-0;1;2;3}"

mkdir -p "$RESULTS_ROOT" "$LOG_DIR"

IFS=';' read -r -a SHARD_MODELS <<< "$MODEL_SHARDS_RAW"
IFS=';' read -r -a SHARD_GPUS <<< "$GPU_SHARDS_RAW"

if [ "${#SHARD_MODELS[@]}" -ne "${#SHARD_GPUS[@]}" ]; then
    echo "MODEL_SHARDS and GPU_SHARDS must have the same number of entries" >&2
    exit 1
fi

SHARD_NAMES=()
for shard_models in "${SHARD_MODELS[@]}"; do
    SHARD_NAMES+=("${shard_models//,/__}")
done

echo "Using Python: $PYTHON_BIN"
echo "Batch size: $BATCH_SIZE"
echo "Launching ${#SHARD_MODELS[@]} benchmark shards across GPUs ${GPU_SHARDS_RAW}"

pids=()

for idx in "${!SHARD_NAMES[@]}"; do
    shard_name="${SHARD_NAMES[$idx]}"
    shard_models="${SHARD_MODELS[$idx]}"
    shard_gpu="${SHARD_GPUS[$idx]}"
    shard_results="$RESULTS_ROOT/$shard_name"
    shard_log="$LOG_DIR/main_clip_${shard_name}.log"

    mkdir -p "$shard_results"

    echo "[$shard_name] GPU=$shard_gpu MODELS=$shard_models"
    (
        cd "$ROOT"
        env \
            PYTHON_BIN="$PYTHON_BIN" \
            CUDA_VISIBLE_DEVICES="$shard_gpu" \
            GPU_INDEX="0" \
            MODEL_KEYS="$shard_models" \
            BATCH_SIZE="$BATCH_SIZE" \
            RESULTS_DIR="$shard_results" \
            SINGLE_MODEL="$shard_models" "$PYTHON_BIN" "$EVAL_DIR/experiments/main_clip.py" \
                --results_dir "$shard_results" > "$shard_log" 2>&1
    ) &
    pids+=("$!")
done

status=0
for idx in "${!pids[@]}"; do
    if wait "${pids[$idx]}"; then
        echo "[${SHARD_NAMES[$idx]}] done"
    else
        echo "[${SHARD_NAMES[$idx]}] failed"
        status=1
    fi
done

MERGED_PATH="$RESULTS_ROOT/benchmark_final_merged_$(date +%s).json"

RESULTS_ROOT="$RESULTS_ROOT" MERGED_PATH="$MERGED_PATH" "$PYTHON_BIN" - <<'PYEOF'
import json
import os
from pathlib import Path

results_root = Path(os.environ["RESULTS_ROOT"])
merged_path = Path(os.environ["MERGED_PATH"])

merged = {
    "timestamp": None,
    "date": None,
    "datasets": {},
}

shard_dirs = sorted(results_root.glob("*"))
shard_dirs = [d for d in shard_dirs if d.is_dir() and (d / "baseline_results.json").exists()]
if not shard_dirs:
    raise SystemExit("No shard baseline_results.json files found to merge.")

merged_data = {}
for shard_dir in shard_dirs:
    with open(shard_dir / "baseline_results.json") as f:
        payload = json.load(f)
    for subset, subset_data in payload.items():
        if subset not in merged_data:
            merged_data[subset] = {"class_names": subset_data.get("class_names", [])}
        for model_key, model_data in subset_data.items():
            if model_key == "class_names":
                continue
            if model_key in merged_data[subset]:
                merged_data[subset][model_key].update(model_data)
            else:
                merged_data[subset][model_key] = model_data

with open(merged_path, "w") as f:
    json.dump(merged_data, f, indent=2)

print(merged_path)
PYEOF

echo "Merged results: $MERGED_PATH"
exit "$status"