"""
data/scb_dataset.py
SCB 数据集加载器，支持三个子集 + 视频帧序列模拟
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPO_ROOT / "data"
LEGACY_DATA_ROOT = REPO_ROOT / "datasets_scb"
DATASETS_ROOT = Path(os.environ.get("SCB_DATA_DIR", DEFAULT_DATA_ROOT))
SUBSET_LOCAL_DIRS = {
    "teacher_behavior": DATASETS_ROOT / "SCB5_TeacherBehavior" / "SCB5_Teacher_Behavior_Stand_BlackBoard_Sreen_20250406-2",
    "handrise_readwrite": DATASETS_ROOT / "SCB5_HandriseReadWrite",
    "bow_turnhead": DATASETS_ROOT / "SCB_BowTurnHead",
}


def _auto_find_datasets_root() -> Optional[str]:
    """尝试自动定位数据根目录，优先使用环境变量 `SCB_DATA_DIR`。"""
    env = os.environ.get("SCB_DATA_DIR")
    if env and os.path.exists(env):
        return env

    for candidate in (DEFAULT_DATA_ROOT, LEGACY_DATA_ROOT):
        if candidate.exists():
            return str(candidate)

    p = Path(__file__).resolve()
    for parent in p.parents[:6]:
        for dirname in ("data", "datasets_scb"):
            candidate = parent / dirname
            if candidate.exists():
                return str(candidate)

    return None


# ─── 类别定义 ────────────────────────────────────────────────────────────────

SUBSET_CONFIG = {
    "teacher_behavior": {
        "hf_name": "teacher_behavior",          # Hugging Face 子集名（需确认实际字段名）
        "classes": [
            "guide", "answer", "on-stage interaction",
            "blackboard-writing", "teacher", "stand", "screen", "blackboard"
        ],
        "multilabel": True,
        "num_classes": 8,
    },
    "handrise_readwrite": {
        "hf_name": "handrise_readwrite",
        "classes": ["hand-raise", "read", "write"],
        "multilabel": False,
        "num_classes": 3,
    },
    "bow_turnhead": {
        "hf_name": "bow_turnhead",
        "classes": ["bow-head", "turn-head"],
        "multilabel": False,
        "num_classes": 2,
    },
}


# ─── 图像变换 ────────────────────────────────────────────────────────────────

def build_transform(input_size: int = 224, is_train: bool = False) -> transforms.Compose:
    """构建图像预处理流水线"""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize(int(input_size * 1.14)),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalize,
        ])


def build_clip_transform(input_size: int = 224) -> transforms.Compose:
    """CLIP 专用预处理（使用 CLIP 的均值方差）"""
    return transforms.Compose([
        transforms.Resize(input_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        ),
    ])


# ─── 数据集类 ────────────────────────────────────────────────────────────────

class SCBDataset(Dataset):
    """
    SCB 智能课堂行为数据集
    
    支持：
    - 从 Hugging Face 加载
    - 从本地目录加载（YOLO格式）
    - 多标签 / 单标签模式
    - 视频帧序列模拟（用于 Video-CLIP）
    """

    def __init__(
        self,
        subset: str,                      # "teacher_behavior" | "handrise_readwrite" | "bow_turnhead"
        split: str = "validation",        # "train" | "validation"
        transform=None,
        data_dir: Optional[str] = None,   # 本地数据目录（优先）
        hf_repo: str = "wintonYF/SCB-Dataset",
        cache_dir: str = "./data/cache",
        video_mode: bool = False,         # 是否返回多帧（视频模拟）
        num_frames: int = 8,              # video_mode 下的帧数
        input_size: int = 224,
    ):
        super().__init__()
        assert subset in SUBSET_CONFIG, f"Unknown subset: {subset}"

        self.subset = subset
        self.split = split
        self.config = SUBSET_CONFIG[subset]
        self.classes = self.config["classes"]
        self.num_classes = self.config["num_classes"]
        self.multilabel = self.config["multilabel"]
        self.video_mode = video_mode
        self.num_frames = num_frames
        self.input_size = input_size

        if transform is None:
            transform = build_clip_transform(input_size)
        self.transform = transform

        # 加载数据
        self.samples = self._load_data(data_dir, hf_repo, cache_dir)
        print(f"[SCBDataset] {subset}/{split}: {len(self.samples)} samples, "
              f"{self.num_classes} classes, multilabel={self.multilabel}")

    def _load_data(self, data_dir, hf_repo, cache_dir) -> List[Dict]:
        """加载数据，优先本地，否则从 HF 下载"""
        # 优先使用显式传入的本地数据目录
        if data_dir and os.path.exists(data_dir):
            return self._load_from_local(data_dir)

        # 与主流水线一致：优先使用当前 subset 的精确本地目录。
        subset_dir = SUBSET_LOCAL_DIRS.get(self.subset)
        if subset_dir and subset_dir.exists():
            print(f"[SCBDataset] Using subset local data dir: {subset_dir}")
            return self._load_from_local(str(subset_dir))

        # 若没有传入，尝试使用环境变量或自动查找工作区下的数据目录
        auto_dir = _auto_find_datasets_root()
        if auto_dir and os.path.exists(auto_dir):
            print(f"[SCBDataset] Auto-detected local data dir: {auto_dir}")
            return self._load_from_local(auto_dir)

        # 从 Hugging Face 加载
        if HF_AVAILABLE:
            return self._load_from_hf(hf_repo, cache_dir)

        raise RuntimeError(
            "No data source available. Either provide data_dir or install 'datasets' package."
        )

    def _load_from_hf(self, hf_repo: str, cache_dir: str) -> List[Dict]:
        """从 Hugging Face 加载数据集"""
        hf_name = self.config["hf_name"]
        try:
            ds = load_dataset(
                hf_repo,
                name=hf_name,
                split=self.split,
                cache_dir=cache_dir,
                trust_remote_code=True,
            )
        except Exception:
            # 若子集名不匹配，尝试无name参数加载
            ds = load_dataset(hf_repo, split=self.split, cache_dir=cache_dir)

        samples = []
        for item in ds:
            label = item.get("label", item.get("labels", None))
            if label is None:
                continue
            samples.append({
                "image": item["image"],       # PIL Image
                "label": label,               # int 或 List[int]
                "image_path": item.get("image_path", ""),
            })
        return samples

    def _load_from_local(self, data_dir: str) -> List[Dict]:
        """
        从本地 YOLO 格式目录加载
        目录结构：
            data_dir/
                images/{split}/  *.jpg
                labels/{split}/  *.txt  (YOLO格式，每行: class_id cx cy w h)
        """
        base = Path(data_dir)

        # 支持三种情况：
        # 1) data_dir/images/{split}
        # 2) data_dir/{subset_dir}/images/{split}（datasets_scb 根目录）
        # 3) split 名称可能为 "val" 而非 "validation"

        def _find_split_dirs(root: Path, split_name: str):
            img = root / "images" / split_name
            lbl = root / "labels" / split_name
            if img.exists() and lbl.exists():
                return img, lbl
            return None, None

        # 首选直接在 data_dir 下查找
        img_dir, lbl_dir = _find_split_dirs(base, self.split)

        # 支持 'val' 作为 'validation' 的别名
        if img_dir is None and self.split == "validation":
            img_dir, lbl_dir = _find_split_dirs(base, "val")

        # 若仍未找到，尝试在 data_dir 的子目录中查找（如 SCB_BowTurnHead）
        if img_dir is None:
            for child in sorted(base.iterdir()):
                if not child.is_dir():
                    continue
                img_dir, lbl_dir = _find_split_dirs(child, self.split)
                if img_dir is not None:
                    break
                if self.split == "validation":
                    img_dir, lbl_dir = _find_split_dirs(child, "val")
                    if img_dir is not None:
                        break

        # 如果仍然没有找到，尝试宽松匹配：如果 images/ 下存在任何子目录，选第一个
        if img_dir is None:
            images_root = base / "images"
            labels_root = base / "labels"
            if images_root.exists() and labels_root.exists():
                subs = sorted([p for p in images_root.iterdir() if p.is_dir()])
                if subs:
                    img_dir = subs[0]
                    # 寻找同名标签子目录，或使用 labels_root/{same}
                    candidate_lbl = labels_root / img_dir.name
                    if candidate_lbl.exists():
                        lbl_dir = candidate_lbl
                    else:
                        # fallback: use labels_root/* first dir
                        lbls = sorted([p for p in labels_root.iterdir() if p.is_dir()])
                        lbl_dir = lbls[0] if lbls else (labels_root)

        if img_dir is None or lbl_dir is None:
            raise RuntimeError(f"No local data found under {data_dir} for split={self.split}")

        samples = []
        for img_path in sorted(img_dir.rglob("*.jpg")):
            lbl_path = lbl_dir / img_path.with_suffix(".txt").name
            labels = []
            if lbl_path.exists():
                with open(lbl_path) as f:
                    for line in f:
                        cls_id = int(line.strip().split()[0])
                        labels.append(cls_id)
            labels = list(set(labels))  # 去重

            samples.append({
                "image": None,
                "image_path": str(img_path),
                "label": labels if self.multilabel else (labels[0] if labels else 0),
            })
        return samples

    def _load_image(self, sample: Dict) -> Image.Image:
        if sample["image"] is not None:
            img = sample["image"]
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            return img.convert("RGB")
        else:
            return Image.open(sample["image_path"]).convert("RGB")

    def _make_label_tensor(self, label) -> torch.Tensor:
        if self.multilabel:
            vec = torch.zeros(self.num_classes, dtype=torch.float32)
            if isinstance(label, (list, tuple)):
                for l in label:
                    if 0 <= l < self.num_classes:
                        vec[l] = 1.0
            else:
                vec[label] = 1.0
            return vec
        else:
            if isinstance(label, (list, tuple)):
                return torch.tensor(label[0] if label else 0, dtype=torch.long)
            return torch.tensor(label, dtype=torch.long)

    def _simulate_video_frames(self, img: Image.Image) -> torch.Tensor:
        """
        视频帧模拟：对单张图像应用随机增强生成多帧序列
        返回 shape: (T, C, H, W)
        """
        aug = transforms.Compose([
            transforms.RandomResizedCrop(
                self.input_size,
                scale=(0.85, 1.0),
                interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            ),
        ])
        frames = [aug(img) for _ in range(self.num_frames)]
        return torch.stack(frames, dim=0)   # (T, C, H, W)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        img = self._load_image(sample)
        label = self._make_label_tensor(sample["label"])

        if self.video_mode:
            frames = self._simulate_video_frames(img)
            return {
                "frames": frames,              # (T, C, H, W)
                "label": label,
                "index": idx,
            }
        else:
            image = self.transform(img)
            return {
                "image": image,                # (C, H, W)
                "label": label,
                "index": idx,
            }


# ─── DataLoader 工厂函数 ─────────────────────────────────────────────────────

def build_dataloader(
    subset: str,
    split: str = "validation",
    batch_size: int = 64,
    num_workers: int = 8,
    input_size: int = 224,
    video_mode: bool = False,
    num_frames: int = 8,
    data_dir: Optional[str] = None,
    hf_repo: str = "wintonYF/SCB-Dataset",
    cache_dir: str = "./data/cache",
    **kwargs,
) -> Tuple[DataLoader, SCBDataset]:
    dataset = SCBDataset(
        subset=subset,
        split=split,
        input_size=input_size,
        video_mode=video_mode,
        num_frames=num_frames,
        data_dir=data_dir,
        hf_repo=hf_repo,
        cache_dir=cache_dir,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return loader, dataset


# ─── 快速验证 ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing SCBDataset...")
    for subset in ["teacher_behavior", "handrise_readwrite", "bow_turnhead"]:
        loader, ds = build_dataloader(
            subset=subset,
            batch_size=32,
            num_workers=0,
        )
        batch = next(iter(loader))
        print(f"  {subset}: image={batch['image'].shape}, label={batch['label'].shape}")

    # 测试视频模式
    loader, ds = build_dataloader(
        subset="bow_turnhead",
        batch_size=8,
        video_mode=True,
        num_frames=8,
        num_workers=0,
    )
    batch = next(iter(loader))
    print(f"  video mode: frames={batch['frames'].shape}")
    print("Dataset test passed!")
