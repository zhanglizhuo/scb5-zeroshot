#!/bin/bash
# =====================================================
# SCB5 CLIP/SLIP/FLIP/TULIP 实验环境安装脚本
# 在你的服务器上运行：bash setup.sh
# =====================================================

set -e

echo "====== 安装基础依赖 ======"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install open-clip-torch
pip install scikit-learn
pip install tqdm
pip install Pillow
pip install numpy

echo "====== 安装 OpenAI CLIP ======"
pip install git+https://github.com/openai/CLIP.git

echo "====== 尝试安装 TULIP ======"
# 如果安装失败会自动 fallback 到 SigLIP2，不影响其他模型
if [ ! -d "tulip" ]; then
    git clone https://github.com/tulip-berkeley/tulip.git || echo "[WARN] TULIP clone 失败，将使用 SigLIP2 替代"
    if [ -d "tulip" ]; then
        pip install -e tulip/ || echo "[WARN] TULIP 安装失败，将使用 SigLIP2 替代"
    fi
fi

echo "====== 安装完成 ======"
echo ""
echo "运行示例："
echo "  # CLIP zero-shot（GPU0）"
echo "  CUDA_VISIBLE_DEVICES=0 python run_experiment.py --model clip --mode zeroshot"
echo ""
echo "  # SLIP zero-shot（GPU1）"
echo "  CUDA_VISIBLE_DEVICES=1 python run_experiment.py --model slip --mode zeroshot"
echo ""
echo "  # FLIP zero-shot（GPU2）"
echo "  CUDA_VISIBLE_DEVICES=2 python run_experiment.py --model flip --mode zeroshot"
echo ""
echo "  # TULIP zero-shot（GPU3）"
echo "  CUDA_VISIBLE_DEVICES=3 python run_experiment.py --model tulip --mode zeroshot"
echo ""
echo "  # 四个模型并行跑（推荐）"
echo "  bash run_all_parallel.sh"
