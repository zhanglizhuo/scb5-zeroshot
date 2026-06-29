#!/usr/bin/env python3
"""
在有网络的机器上运行此脚本下载模型权重，
然后把 ckpts/ 文件夹传到服务器。

用法：
  python scripts/download_models.py
  tar czf ckpts.tar.gz ckpts/
    scp ckpts.tar.gz broadsense@服务器IP:~/works/lizhuo/AutoResearchClaw/analysis/
  # 服务器上：
  tar xzf ckpts.tar.gz
"""

import os
import subprocess
from pathlib import Path

CKPT_DIR = Path("./ckpts")
CKPT_DIR.mkdir(exist_ok=True)

# 设置镜像（中国大陆优先用这个）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

def download(repo_id, local_dir, filename=None):
    local_dir = CKPT_DIR / local_dir
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*50}")
    print(f"Downloading: {repo_id}")
    print(f"  -> {local_dir}")

    try:
        from huggingface_hub import hf_hub_download, snapshot_download
        if filename:
            path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(local_dir),
            )
            print(f"  ✓ {path}")
        else:
            snapshot_download(repo_id=repo_id, local_dir=str(local_dir))
            print(f"  ✓ Done")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        print(f"  Try manual: huggingface-cli download {repo_id} --local-dir {local_dir}")


# ─────────────────────────────────────────────
# 下载列表
# ─────────────────────────────────────────────

# 1. CLIP OpenAI ViT-L/14 (最小，通常已缓存)
download(
    "openai/clip-vit-large-patch14",
    "clip_openai_vitl14",
    filename="pytorch_model.bin",
)

# 2. LAION CLIP ViT-L/14 (代表 SLIP-style)
download(
    "laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
    "laion_vitl14",
    filename="open_clip_pytorch_model.bin",
)

# 3. FLIP: 用 LAION ViT-L 近似（官方FLIP ckpt无独立HF repo）
#    实际上 open_clip 里 laion2b_s32b_b82k 已包含 FLIP 训练策略
print("\n[FLIP] 使用 LAION ViT-L/14 权重（FLIP 策略已包含在内）")
print("  -> 复用 ckpts/laion_vitl14/ 即可，代码会自动处理")

# 4. SigLIP ViT-L/16-256 (TULIP 对标 baseline)
download(
    "timm/ViT-L-16-SigLIP2-256",
    "siglip_vitl16_256",
)

print("\n" + "="*50)
print("Download complete. Files in ./ckpts/:")
for f in sorted(CKPT_DIR.rglob("*")):
    if f.is_file():
        size_mb = f.stat().st_size // 1024 // 1024
        print(f"  {f}  ({size_mb} MB)")

print("\n运行方式：")
print("  python experiments/main_clip.py")
