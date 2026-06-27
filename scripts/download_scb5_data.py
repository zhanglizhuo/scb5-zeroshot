"""
download_scb5_data.py — 下载 SCB 数据集（通过代理）
===================================================
轻量脚本：在能访问外网的机器上通过 HTTP 代理下载训练/验证数据集
并把文件保存到 `datasets_scb/` 目录下。适用于在受限网络环境下先
下载数据然后传输到离线服务器。

注意：脚本内 `proxies` 配置为示例，请根据实际代理服务修改或移除。

Usage:
    python3 scripts/download_scb5_data.py
"""

import os
import requests

# 代理设置（示例）
proxies = {
        "http": "http://Clash:meO8PQ5J@192.168.1.234:7890",
        "https": "http://Clash:meO8PQ5J@192.168.1.234:7890",
}

# Define datasets to download: (subfolder, list of (url, filename))
datasets = {
    "SCB_BowTurnHead": [
        ("https://huggingface.co/datasets/wintonYF/SCB-Dataset/resolve/main/SCB_BowTurnHead_20250509/SCB_BowTurnHead_20250509.zip", "SCB_BowTurnHead_20250509.zip"),
        ("https://huggingface.co/datasets/wintonYF/SCB-Dataset/resolve/main/SCB_BowTurnHead_20250509/SCB_BowTurnHead_20250509.yaml", "SCB_BowTurnHead_20250509.yaml"),
    ],
    "SCB5_Discuss": [
        ("https://huggingface.co/datasets/wintonYF/SCB-Dataset/resolve/main/SCB5-Discuss-2024-9-17/SCB5-Discuss-2024-9-17.zip", "SCB5-Discuss-2024-9-17.zip"),
        ("https://huggingface.co/datasets/wintonYF/SCB-Dataset/resolve/main/SCB5-Discuss-2024-9-17/SCB5-Discuss-2024-9-17.yaml", "SCB5-Discuss-2024-9-17.yaml"),
    ],
    "SCB5_HandriseReadWrite": [
        ("https://huggingface.co/datasets/wintonYF/SCB-Dataset/resolve/main/SCB5-Handrise-Read-write-2024-9-17/SCB5-Handrise-Read-write-2024-9-17.zip", "SCB5-Handrise-Read-write-2024-9-17.zip"),
        ("https://huggingface.co/datasets/wintonYF/SCB-Dataset/resolve/main/SCB5-Handrise-Read-write-2024-9-17/SCB5-Handrise-Read-write-2024-9-17.yaml", "SCB5-Handrise-Read-write-2024-9-17.yaml"),
    ],
    "SCB5_TeacherBehavior": [
        ("https://huggingface.co/datasets/wintonYF/SCB-Dataset/resolve/main/SCB5_Teacher_Behavior_Stand_BlackBoard_Sreen_20250406/SCB5_Teacher_Behavior_Stand_BlackBoard_Sreen_20250406-2.zip", "SCB5_Teacher_Behavior_Stand_BlackBoard_Sreen_20250406-2.zip"),
        ("https://huggingface.co/datasets/wintonYF/SCB-Dataset/resolve/main/SCB5_Teacher_Behavior_Stand_BlackBoard_Sreen_20250406/SCB5_Teacher_Behavior_Stand_BlackBoard_Sreen_20250406.yaml", "SCB5_Teacher_Behavior_Stand_BlackBoard_Sreen_20250406.yaml"),
        ("https://huggingface.co/datasets/wintonYF/SCB-Dataset/resolve/main/SCB5_Teacher_Behavior_Stand_BlackBoard_Sreen_20250406/countLabels4.py", "countLabels4.py"),
    ],
}

for folder, files in datasets.items():
    target_dir = os.path.join("datasets_scb", folder)
    os.makedirs(target_dir, exist_ok=True)
    for url, fname in files:
        print(f"Downloading {fname} into {target_dir}...")
        try:
            resp = requests.get(url, proxies=proxies, stream=True, timeout=60)
            resp.raise_for_status()
        except Exception as e:
            print(f"Failed to download {url}: {e}")
            continue
        out_path = os.path.join(target_dir, fname)
        with open(out_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"Saved to {out_path}")

print("Download attempts finished. Verify files in datasets_scb/")
