#!/usr/bin/env bash

# master_benchmark_parallel.sh — Launch benchmark shards in parallel
# ================================================================
# Thin wrapper that partitions the model list into shards and runs
# `master_benchmark.sh` for each shard on separate GPUs in parallel.
# Designed for offline-first runs: configure `PYTHON_BIN`, `MODEL_SHARDS`,
# and `GPU_SHARDS` via environment variables as needed.
#
# Usage:
#   MODEL_SHARDS="clip;laion;siglip;eva02,dfn" GPU_SHARDS="0;1;2;3" \
#     PYTHON_BIN=/usr/bin/python3 bash master_benchmark_parallel.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
EVAL_DIR="$ROOT/scb5_zeroshot"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3.8}"
RESULTS_ROOT="$EVAL_DIR/results/parallel"
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
    shard_log="$LOG_DIR/master_benchmark_${shard_name}.log"

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
            bash "$EVAL_DIR/master_benchmark.sh" > "$shard_log" 2>&1
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

final_files = sorted(results_root.glob("*/benchmark_final_*.json"))
if not final_files:
    raise SystemExit("No shard result files found to merge.")

for final_file in final_files:
    with open(final_file, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    merged["timestamp"] = payload.get("timestamp", merged["timestamp"])
    merged["date"] = payload.get("date", merged["date"])
    for ds_name, ds_payload in payload.get("datasets", {}).items():
        if ds_name not in merged["datasets"]:
            merged["datasets"][ds_name] = ds_payload
            continue
        merged_ds = merged["datasets"][ds_name]
        merged_experiments = merged_ds.setdefault("experiments", [])
        merged_experiments.extend(ds_payload.get("experiments", []))

with open(merged_path, "w", encoding="utf-8") as handle:
    json.dump(merged, handle, indent=2, ensure_ascii=False)

print(merged_path)
PYEOF

echo "Merged results: $MERGED_PATH"
exit "$status"