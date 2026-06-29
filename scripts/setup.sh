#!/bin/bash
# =====================================================
# SCB5 CLIP-Family zero-shot 实验环境安装脚本
# 在你的服务器上运行：bash scripts/setup.sh
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

echo "====== 安装完成 ======"
echo ""
echo "运行示例："
echo "  # 快速验证（使用预计算特征）"
echo "  bash reproduce_paper.sh --mode quick"
echo ""
echo "  # 完整复现（需 GPU）"
echo "  bash reproduce_paper.sh --mode full --gpu 0"
echo ""
echo "更多信息请参考 REPRODUCIBILITY.md"
