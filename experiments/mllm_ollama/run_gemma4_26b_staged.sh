#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"
export ROOT_DIR  # make available to inline Python

PY="${ROOT_DIR}/.venv/bin/python"
[ -x "$PY" ] || PY="$(command -v python3)"
OLLAMA_MODELS_DIR="${OLLAMA_MODELS_DIR:-/usr/share/ollama/.ollama/models}"

# Dedicated ports for this staged run.
PORT0=11470
PORT1=11471
PORT2=11472

TB_SHARDS_DIR="results/mllm/mllm_gemma4_26b_tb_shards"
TB_DIR="results/mllm/mllm_gemma4_26b_tb"
HR_DIR="results/mllm/mllm_gemma4_26b_hr"
BT_DIR="results/mllm/mllm_gemma4_26b_bt"
FINAL_DIR="results/mllm/mllm_gemma4_26b_full"

mkdir -p "$TB_SHARDS_DIR" "$TB_DIR" "$HR_DIR" "$BT_DIR" "$FINAL_DIR"

# Start three dedicated Ollama workers pinned to GPUs 0/1/2.
CUDA_VISIBLE_DEVICES=0 OLLAMA_HOST=127.0.0.1:${PORT0} OLLAMA_MODELS="$OLLAMA_MODELS_DIR" ollama serve > /tmp/ollama_${PORT0}.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 OLLAMA_HOST=127.0.0.1:${PORT1} OLLAMA_MODELS="$OLLAMA_MODELS_DIR" ollama serve > /tmp/ollama_${PORT1}.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 OLLAMA_HOST=127.0.0.1:${PORT2} OLLAMA_MODELS="$OLLAMA_MODELS_DIR" ollama serve > /tmp/ollama_${PORT2}.log 2>&1 &

sleep 4

echo "[Stage 1/3] teacher_behavior split into 3 shards"
"$PY" experiments/main_mllm.py \
  --models gemma4_ollama \
  --gemma4_model gemma4:26b \
  --ollama_host http://127.0.0.1:${PORT0} \
  --subsets teacher_behavior \
  --start_idx 0 --end_idx 1080 \
  --results_dir "$TB_SHARDS_DIR" > /tmp/gemma4_26b_tb_s0.log 2>&1 &
PID_TB0=$!

"$PY" experiments/main_mllm.py \
  --models gemma4_ollama \
  --gemma4_model gemma4:26b \
  --ollama_host http://127.0.0.1:${PORT1} \
  --subsets teacher_behavior \
  --start_idx 1080 --end_idx 2160 \
  --results_dir "$TB_SHARDS_DIR" > /tmp/gemma4_26b_tb_s1.log 2>&1 &
PID_TB1=$!

"$PY" experiments/main_mllm.py \
  --models gemma4_ollama \
  --gemma4_model gemma4:26b \
  --ollama_host http://127.0.0.1:${PORT2} \
  --subsets teacher_behavior \
  --start_idx 2160 --end_idx 3240 \
  --results_dir "$TB_SHARDS_DIR" > /tmp/gemma4_26b_tb_s2.log 2>&1 &
PID_TB2=$!

wait "$PID_TB0" "$PID_TB1" "$PID_TB2"

echo "[Stage 2/3] merge teacher_behavior shards"
"$PY" - <<'PY'
import json, os
from pathlib import Path

root = Path(os.environ['ROOT_DIR'])
shards_dir = root / 'results/mllm/mllm_gemma4_26b_tb_shards'
out_dir = root / 'results/mllm/mllm_gemma4_26b_tb'
out_dir.mkdir(parents=True, exist_ok=True)

files = sorted(shards_dir.glob('gemma4_ollama_teacher_behavior_s*_e*.json'))
if not files:
    raise RuntimeError('No teacher shard files found')

items = []
for fp in files:
    with open(fp) as f:
        obj = json.load(f)
    items.append((obj.get('start_idx', 0) or 0, obj))
items.sort(key=lambda x: x[0])

raw_preds = []
labels = []
for _, obj in items:
    raw_preds.extend(obj.get('raw_preds', []))
    labels.extend(obj.get('labels', []))

if len(raw_preds) != len(labels):
    raise RuntimeError('Merged raw_preds/labels length mismatch')

num_classes = len(labels[0]) if labels else 8
logits = [[0.0] * num_classes for _ in range(len(raw_preds))]
for i, pred in enumerate(raw_preds):
    for idx in pred.get('predicted_labels', []):
        if 0 <= idx < num_classes:
            logits[i][idx] = 1.0


def hit_at_k(k: int) -> float:
    hits = 0
    for i, row in enumerate(logits):
        topk = sorted(range(len(row)), key=lambda j: row[j])[-k:]
        true_set = {j for j, v in enumerate(labels[i]) if v}
        if true_set.intersection(topk):
            hits += 1
    return hits / len(labels) if labels else 0.0

merged = {
    'model': 'gemma4_ollama',
    'subset': 'teacher_behavior',
    'hit_at_1': hit_at_k(1),
    'hit_at_3': hit_at_k(3),
    'num_samples': len(labels),
    'raw_preds': raw_preds,
    'labels': labels,
}

with open(out_dir / 'gemma4_ollama_teacher_behavior_preds.json', 'w') as f:
    json.dump(merged, f, indent=2)
with open(out_dir / 'mllm_summary.json', 'w') as f:
    json.dump({'gemma4_ollama': {'teacher_behavior': merged['hit_at_1']}}, f, indent=2)
PY

echo "[Stage 3/3] run remaining two subsets"
"$PY" experiments/main_mllm.py \
  --models gemma4_ollama \
  --gemma4_model gemma4:26b \
  --ollama_host http://127.0.0.1:${PORT0} \
  --subsets handrise_readwrite \
  --results_dir "$HR_DIR" > /tmp/gemma4_26b_hr.log 2>&1 &
PID_HR=$!

"$PY" experiments/main_mllm.py \
  --models gemma4_ollama \
  --gemma4_model gemma4:26b \
  --ollama_host http://127.0.0.1:${PORT1} \
  --subsets bow_turnhead \
  --results_dir "$BT_DIR" > /tmp/gemma4_26b_bt.log 2>&1 &
PID_BT=$!

wait "$PID_HR" "$PID_BT"

echo "[Final] build consolidated summary"
"$PY" - <<'PY'
import json, os
from pathlib import Path

root = Path(os.environ['ROOT_DIR'])
summary_tb = root / 'results/mllm/mllm_gemma4_26b_tb/mllm_summary.json'
summary_hr = root / 'results/mllm/mllm_gemma4_26b_hr/mllm_summary.json'
summary_bt = root / 'results/mllm/mllm_gemma4_26b_bt/mllm_summary.json'
out_dir = root / 'results/mllm/mllm_gemma4_26b_full'
out_dir.mkdir(parents=True, exist_ok=True)

with open(summary_tb) as f:
    tb = json.load(f)['gemma4_ollama']
with open(summary_hr) as f:
    hr = json.load(f)['gemma4_ollama']
with open(summary_bt) as f:
    bt = json.load(f)['gemma4_ollama']

merged = {
    'gemma4_ollama': {
        'teacher_behavior': tb['teacher_behavior'],
        'handrise_readwrite': hr['handrise_readwrite'],
        'bow_turnhead': bt['bow_turnhead'],
    }
}
with open(out_dir / 'mllm_summary.json', 'w') as f:
    json.dump(merged, f, indent=2)

print(json.dumps(merged, indent=2))
PY

echo "Done. Consolidated summary: $FINAL_DIR/mllm_summary.json"
echo "Teacher logs: /tmp/gemma4_26b_tb_s0.log /tmp/gemma4_26b_tb_s1.log /tmp/gemma4_26b_tb_s2.log"
echo "Other logs: /tmp/gemma4_26b_hr.log /tmp/gemma4_26b_bt.log"
