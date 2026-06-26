#!/bin/bash
# scripts/run_all.sh
# 完整实验流水线，4× V100 32GB
# 预计总运行时间：12-20小时（含模型下载和特征缓存）

set -e
PYTHON=/usr/bin/python3
export PYTHONPATH=$(pwd):$PYTHONPATH
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1,2,3

# CKPTS_DIR: 可通过环境变量指定权重目录；默认指向上一级的 ckpts
CKPTS_DIR=${CKPTS_DIR:-$(pwd)/../ckpts}
if [ -d "$CKPTS_DIR" ] && [ "$(ls -A "$CKPTS_DIR")" ]; then
    echo "Found ckpts at $CKPTS_DIR — will skip pre-download step."
    SKIP_PREDOWNLOAD=1
else
    echo "No local ckpts found at $CKPTS_DIR — will run pre-download."
    SKIP_PREDOWNLOAD=0
fi

echo "=================================================="
echo "  课堂行为识别论文升级实验"
echo "  4× V100 32GB"
echo "  $(date)"
echo "=================================================="

# ─── 环境检查 ─────────────────────────────────────────────────────────────────
$PYTHON -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    mem = torch.cuda.get_device_properties(i).total_memory / 1e9
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)} ({mem:.0f}GB)')
"

# ─── 步骤1: 安装依赖 ───────────────────────────────────────────────────────────
echo ""
echo "[Step 1] Installing dependencies..."
$PYTHON -m pip install -q open_clip_torch transformers datasets scikit-learn \
    bitsandbytes accelerate tqdm pyyaml pillow numpy scipy \
    sentencepiece protobuf

# ─── 步骤2: 预下载模型权重（防止推理时超时）─────────────────────────────────
if [ "$SKIP_PREDOWNLOAD" -eq 0 ]; then
  echo ""
  echo "[Step 2] Pre-downloading model weights..."
  $PYTHON - <<'EOF'
import open_clip
import torch

configs = [
    ("ViT-L-14", "openai"),
    ("ViT-L-14", "laion2b_s32b_b82k"),
    ("EVA02-L-14", "merged2b_s4b_b131k"),
    ("ViT-L-14", "dfn2b"),
]
for model_name, pretrained in configs:
    try:
        print(f"  Downloading {model_name}/{pretrained}...")
        m, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        del m
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  Warning: {e}")

# SigLIP2
try:
    from transformers import AutoProcessor, AutoModel
    print("  Downloading siglip2...")
    m = AutoModel.from_pretrained("google/siglip2-large-patch16-256")
    del m
except Exception as e:
    print(f"  Warning: {e}")
EOF
else
  echo "[Step 2] Skipping pre-download because local ckpts found at $CKPTS_DIR"
fi

# ─── 步骤3: 复现原始 CAPE 基线（确认环境正确）────────────────────────────────
echo ""
echo "[Step 3] Running CAPE baseline (sanity check)..."
$PYTHON main_clip.py \
    --config config/experiment_config.yaml \
    --results_dir ./results/baseline \
    --experiment cape_only \
    2>&1 | tee logs/baseline.log

# ─── 步骤4: TFVA 实验（核心贡献）─────────────────────────────────────────────
echo ""
echo "[Step 4] Running TFVA experiments..."

# 4a. 零样本 TFVA
echo "  [4a] Zero-Shot TFVA..."
$PYTHON main_tfva.py \
    --config config/experiment_config.yaml \
    --results_dir ./results/tfva \
    --experiment zs \
    2>&1 | tee logs/tfva_zs.log

# 4b. Few-shot TFVA k-shot 消融
echo "  [4b] Few-Shot TFVA k-shot ablation..."
$PYTHON main_tfva.py \
    --config config/experiment_config.yaml \
    --results_dir ./results/tfva \
    --experiment fs \
    2>&1 | tee logs/tfva_fs.log

# 4c. 多标签 TFVA
echo "  [4c] Multi-Label TFVA..."
$PYTHON main_tfva.py \
    --config config/experiment_config.yaml \
    --results_dir ./results/tfva \
    --experiment multilabel \
    2>&1 | tee logs/tfva_ml.log

# ─── 步骤5: MLLM 对比实验 ─────────────────────────────────────────────────────
echo ""
echo "[Step 5] Running MLLM comparison..."
$PYTHON main_mllm.py \
    --results_dir ./results/mllm \
    --models llava qwenvl \
    --llava_device cuda:0 \
    --qwenvl_device cuda:1 \
    2>&1 | tee logs/mllm.log

# ─── 步骤6: LLM 自动提示生成 ─────────────────────────────────────────────────
echo ""
echo "[Step 6] LLM-generated prompts (using local LLM)..."
$PYTHON prompts/llm_prompt_gen.py \
    --model "llava"  \
    --results_dir ./results/llm_prompts \
    2>&1 | tee logs/llm_prompts.log

# ─── 步骤7: 可视化和结果汇总 ─────────────────────────────────────────────────
echo ""
echo "[Step 7] Generating figures and summary tables..."
$PYTHON scripts/generate_figures.py \
    --results_dir ./results \
    --output_dir ./figures \
    2>&1 | tee logs/figures.log

echo ""
echo "=================================================="
echo "  ALL EXPERIMENTS COMPLETE"
echo "  $(date)"
echo "=================================================="
echo ""
echo "Results:"
echo "  Baseline:  ./results/baseline/"
echo "  TFVA:      ./results/tfva/"
echo "  MLLM:      ./results/mllm/"
echo "  Figures:   ./figures/"
