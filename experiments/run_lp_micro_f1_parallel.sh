#!/usr/bin/env bash
# Launch 5 backbones across 4 GPUs in parallel.
# clip,laion -> GPU0,1; siglip,eva02 -> GPU2,3; dfn -> GPU0 (after clip).
# Logs in scb5_zeroshot/logs/lp_micro_*.log.
set -e
cd "$(dirname "$0")/.."
mkdir -p logs

PY=python3
LOGDIR=logs
TS=$(date +%s)

run() {
    local gpu=$1
    local model=$2
    local logf="${LOGDIR}/lp_micro_${model}_${TS}.log"
    echo "Launch $model on GPU $gpu -> $logf"
    CUDA_VISIBLE_DEVICES=$gpu nohup $PY scb5_zeroshot/compute_lp_micro_f1.py \
        --models "$model" --tag "${model}_g${gpu}" > "$logf" 2>&1 &
}

# 4 GPUs, 5 backbones — first 4 in parallel, dfn waits for one slot.
run 0 clip
run 1 laion
run 2 siglip
run 3 eva02

# Wait for one slot to free, then schedule dfn on GPU 0.
# Simple approach: wait for clip (GPU 0) to finish before starting dfn.
wait_pid=$!
echo "All 4 launched. Waiting for any to complete before scheduling dfn..."

# Wait for any background job to finish, then run dfn on GPU 0.
wait -n
run 0 dfn

# Wait for all remaining
wait
echo "All done. Logs in ${LOGDIR}/lp_micro_*_${TS}.log"
echo "Result JSONs in results/revision/lp_micro_f1_teacher_*.json"
