"""
models/tfva.py
Training-Free Visual Adapter (TFVA)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
核心思想：在冻结 CLIP 特征空间上构建类别原型缓存，
通过 test-time 特征校正提升细粒度教育场景识别能力。

两种工作模式：
  1. Zero-shot TFVA (ZS-TFVA)：使用文本特征作为类别原型（无标注）
  2. Few-shot TFVA (FS-TFVA)：使用少量图像特征增强类别原型（k-shot）

参考：TipAdapter (ECCV 2022) + 教育场景特定改进
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
import numpy as np


# ─── 零样本 TFVA ─────────────────────────────────────────────────────────────

class ZeroShotTFVA(nn.Module):
    """
    零样本 TFVA：无需任何标注数据
    使用 CAPE 文本嵌入作为类别原型，通过自适应温度和特征空间偏移
    提升分类性能。
    """

    def __init__(
        self,
        text_features: torch.Tensor,   # (C, D) CAPE 文本特征
        alpha: float = 1.0,            # 原型融合系数
        beta: float = 5.5,             # Logit 放大系数（温度倒数）
        num_classes: int = None,
    ):
        super().__init__()
        C, D = text_features.shape
        self.num_classes = num_classes or C
        self.alpha = alpha
        self.beta = beta

        # 注册为 buffer（不参与梯度）
        self.register_buffer("text_prototypes", text_features)  # (C, D)

    @torch.no_grad()
    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        """
        image_features: (N, D) L2-normalized
        返回 logits: (N, C)
        """
        # 标准零样本 logit
        zs_logits = self.beta * (image_features @ self.text_prototypes.T)  # (N, C)
        return zs_logits

    def predict(self, image_features: torch.Tensor) -> torch.Tensor:
        """返回预测类别 (N,)"""
        logits = self.forward(image_features)
        return logits.argmax(dim=-1)


# ─── 少样本 TFVA ──────────────────────────────────────────────────────────────

class FewShotTFVA(nn.Module):
    """
    少样本 TFVA：使用 k-shot 图像特征增强类别原型
    
    算法：
    1. 对每个类别，从训练集采样 k 张图像，提取 CLIP 特征
    2. 将图像特征与文本特征加权融合构建类别原型
    3. 推理时对每个测试样本进行最近邻检索 + 文本分类融合
    """

    def __init__(
        self,
        text_features: torch.Tensor,   # (C, D)
        shot_features: torch.Tensor,   # (C*k, D) or (C, k, D)
        shot_labels: torch.Tensor,     # (C*k,) int
        num_classes: int,
        alpha: float = 1.0,
        beta: float = 5.5,
        shot_alpha: float = 0.5,       # 图像原型权重
    ):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.shot_alpha = shot_alpha

        self.register_buffer("text_prototypes", text_features)     # (C, D)
        self.register_buffer("shot_features", shot_features)       # (C*k, D)
        self.register_buffer("shot_labels", shot_labels)           # (C*k,)

        # 构建图像原型（每类 k 张的平均）
        img_prototypes = self._build_image_prototypes(
            shot_features, shot_labels, num_classes
        )
        self.register_buffer("image_prototypes", img_prototypes)   # (C, D)

        # 融合原型
        fused = self._fuse_prototypes(
            self.text_prototypes, self.image_prototypes, shot_alpha
        )
        self.register_buffer("fused_prototypes", fused)           # (C, D)

    def _build_image_prototypes(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        num_classes: int,
    ) -> torch.Tensor:
        prototypes = torch.zeros(num_classes, features.shape[1], device=features.device)
        counts = torch.zeros(num_classes, device=features.device)
        for i, label in enumerate(labels):
            prototypes[label] += features[i]
            counts[label] += 1
        # 归一化（避免零除）
        counts = counts.clamp(min=1)
        prototypes = prototypes / counts.unsqueeze(-1)
        return F.normalize(prototypes, dim=-1)

    def _fuse_prototypes(
        self,
        text_proto: torch.Tensor,
        img_proto: torch.Tensor,
        shot_alpha: float,
    ) -> torch.Tensor:
        fused = (1 - shot_alpha) * text_proto + shot_alpha * img_proto
        return F.normalize(fused, dim=-1)

    @torch.no_grad()
    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        """
        image_features: (N, D)
        返回融合 logits: (N, C)
        """
        # 文本原型 logit
        text_logits = self.beta * (image_features @ self.text_prototypes.T)   # (N, C)

        # 图像原型检索 logit（soft KNN）
        sim_to_shots = image_features @ self.shot_features.T   # (N, C*k)
        shot_exp = torch.exp(self.beta * sim_to_shots)         # (N, C*k)

        # 聚合到类别
        C = self.num_classes
        knn_logits = torch.zeros(image_features.shape[0], C,
                                 device=image_features.device)
        for c in range(C):
            mask = (self.shot_labels == c)
            if mask.any():
                knn_logits[:, c] = shot_exp[:, mask].sum(dim=-1)
        knn_logits = torch.log(knn_logits + 1e-8)

        # 自适应融合
        logits = text_logits + self.alpha * knn_logits
        return logits

    def predict(self, image_features: torch.Tensor) -> torch.Tensor:
        return self.forward(image_features).argmax(dim=-1)


# ─── 多标签 TFVA ──────────────────────────────────────────────────────────────

class MultiLabelTFVA(nn.Module):
    """
    多标签场景的 TFVA 变体（针对 TeacherBehavior 设计）
    为每个类别训练独立的二值分类器（One-vs-Rest）
    """

    def __init__(
        self,
        text_features: torch.Tensor,   # (C, D)
        beta: float = 5.5,
        learnable_threshold: bool = True,
    ):
        super().__init__()
        C, D = text_features.shape
        self.num_classes = C
        self.beta = beta

        self.register_buffer("text_prototypes", text_features)

        # 可学习的每类阈值（用于多标签判断）
        if learnable_threshold:
            self.thresholds = nn.Parameter(torch.zeros(C))
        else:
            self.register_buffer("thresholds", torch.zeros(C))

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        """返回每类的独立得分 (N, C)"""
        logits = self.beta * (image_features @ self.text_prototypes.T)
        # 确保 thresholds 在与 image_features 相同的设备上
        thresholds = self.thresholds.to(image_features.device)
        return torch.sigmoid(logits - thresholds.unsqueeze(0))

    @torch.no_grad()
    def predict_multilabel(
        self,
        image_features: torch.Tensor,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        """返回多标签预测 (N, C) binary"""
        probs = self.forward(image_features)
        return (probs >= threshold).float()


# ─── Few-shot 样本采集 ────────────────────────────────────────────────────────

def collect_few_shot_features(
    model,                              # CLIPModel
    dataset,                            # SCBDataset (train split)
    num_classes: int,
    k_shot: int = 4,
    device: str = "cuda",
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    从训练集采集每类 k 张图像的 CLIP 特征

    返回:
        features: (num_classes * k, D)
        labels:   (num_classes * k,) int
    """
    import random
    random.seed(seed)
    np.random.seed(seed)

    # 按类别索引样本
    class_indices = {c: [] for c in range(num_classes)}
    for idx, sample in enumerate(dataset.samples):
        label = sample["label"]
        if isinstance(label, (list, tuple)):
            for l in label:
                if l < num_classes:
                    class_indices[l].append(idx)
        else:
            if int(label) < num_classes:
                class_indices[int(label)].append(idx)

    all_features = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for cls_id in range(num_classes):
            indices = class_indices[cls_id]
            if len(indices) == 0:
                print(f"  Warning: no samples for class {cls_id}")
                continue
            # 采样 k 个
            sampled = random.sample(indices, min(k_shot, len(indices)))
            for idx in sampled:
                sample = dataset[idx]
                img = sample["image"].unsqueeze(0).to(device)
                feat = model.encode_images(img)    # (1, D)
                all_features.append(feat.cpu())
                all_labels.append(cls_id)

    features = torch.cat(all_features, dim=0)     # (N_total, D)
    labels = torch.tensor(all_labels, dtype=torch.long)
    return features, labels


# ─── TFVA 训练器（轻量级，无需原始图像数据）────────────────────────────────

class TFVATrainer:
    """
    对 FewShotTFVA 的 alpha/beta 超参做网格搜索
    """

    def __init__(
        self,
        text_features: torch.Tensor,
        shot_features: torch.Tensor,
        shot_labels: torch.Tensor,
        num_classes: int,
        device: str = "cuda",
    ):
        self.text_features = text_features
        self.shot_features = shot_features
        self.shot_labels = shot_labels
        self.num_classes = num_classes
        self.device = device

    def grid_search(
        self,
        val_features: torch.Tensor,   # (N, D)
        val_labels: torch.Tensor,     # (N,) int (primary label)
        alphas: List[float] = None,
        betas: List[float] = None,
        shot_alphas: List[float] = None,
    ) -> Dict:
        """网格搜索最优超参"""
        if alphas is None:
            alphas = [0.5, 1.0, 2.0, 4.0]
        if betas is None:
            betas = [1.0, 2.0, 5.5, 10.0, 20.0]
        if shot_alphas is None:
            shot_alphas = [0.1, 0.3, 0.5, 0.7, 0.9]

        best_acc = -1
        best_params = {}

        for alpha in alphas:
            for beta in betas:
                for sa in shot_alphas:
                    model = FewShotTFVA(
                        self.text_features.to(self.device),
                        self.shot_features.to(self.device),
                        self.shot_labels.to(self.device),
                        self.num_classes,
                        alpha=alpha, beta=beta, shot_alpha=sa,
                    ).to(self.device)

                    with torch.no_grad():
                        logits = model(val_features.to(self.device))
                        preds = logits.argmax(dim=-1).cpu()

                    acc = (preds == val_labels).float().mean().item()
                    if acc > best_acc:
                        best_acc = acc
                        best_params = {"alpha": alpha, "beta": beta, "shot_alpha": sa}

        print(f"  Best val acc: {best_acc:.4f}, params: {best_params}")
        return best_params


if __name__ == "__main__":
    print("Testing TFVA modules...")
    D, C, N = 768, 8, 100

    text_feats = F.normalize(torch.randn(C, D), dim=-1)
    img_feats = F.normalize(torch.randn(N, D), dim=-1)

    # ZS-TFVA
    zs_model = ZeroShotTFVA(text_feats, alpha=1.0, beta=5.5)
    logits = zs_model(img_feats)
    print(f"  ZS-TFVA logits: {logits.shape}")

    # FS-TFVA
    k = 4
    shot_feats = F.normalize(torch.randn(C * k, D), dim=-1)
    shot_labels = torch.arange(C).repeat_interleave(k)
    fs_model = FewShotTFVA(text_feats, shot_feats, shot_labels, C)
    logits_fs = fs_model(img_feats)
    print(f"  FS-TFVA logits: {logits_fs.shape}")

    # MultiLabel-TFVA
    ml_model = MultiLabelTFVA(text_feats, beta=5.5)
    probs = ml_model(img_feats)
    print(f"  MultiLabel-TFVA probs: {probs.shape}")
    print("TFVA test passed!")
