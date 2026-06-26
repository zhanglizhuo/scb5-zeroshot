# 课堂行为识别论文升级实验方案
## 目标：从 MDPI Applied Sciences → IEEE TLT / Pattern Recognition

## 硬件配置
- 4× NVIDIA V100 32GB
- 建议使用 PyTorch DDP 多卡并行

## 三大核心升级模块

### Module 1: Video-CLIP 视频模态扩展
- 将静态图像识别升级为视频片段识别
- 模型：LanguageBind-Video, InternVideo2, VideoCLIP
- 新增时序建模能力

### Module 2: Training-Free Visual Adapter (TFVA)
- 在冻结 CLIP 特征上做 test-time 特征校正
- 无需任何标注训练数据
- 方法：TipAdapter-F 变体 + 类原型对齐

### Module 3: MLLM Zero-Shot Baseline
- GPT-4V API 调用对比
- LLaVA-1.6 本地部署对比（V100可跑）
- Qwen-VL 对比

## 文件结构
```
Experiment_Ex/
├── config/
│   └── experiment_config.yaml
├── data/
│   └── scb_dataset.py          # 数据集加载（含视频片段采样）
├── models/
│   ├── clip_zoo.py             # 原始5个CLIP模型
│   ├── video_clip.py           # 视频模型封装
│   ├── tfva.py                 # Training-Free Visual Adapter
│   └── mllm_baseline.py        # LLaVA / Qwen-VL 对比
├── prompts/
│   ├── cape_prompts.py         # 原始CAPE提示（含新类别）
│   └── llm_prompt_gen.py       # LLM自动生成提示
├── evaluation/
│   ├── metrics.py              # Hit@1/3, Macro-F1, multilabel metrics
│   └── bootstrap_ci.py         # Bootstrap置信区间
├── train/
│   └── tfva_adapter.py         # TFVA少量样本适配训练
├── scripts/
│   ├── run_clip_baseline.sh
│   ├── run_video_clip.sh
│   ├── run_tfva.sh
│   └── run_mllm.sh
├── main_clip.py                # 复现原始实验
├── main_video.py               # 视频模态实验
├── main_tfva.py                # TFVA适配器实验
└── main_mllm.py                # MLLM对比实验
```
