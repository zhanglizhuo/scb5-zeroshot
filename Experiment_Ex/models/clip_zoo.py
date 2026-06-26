"""
models/clip_zoo.py
统一封装5个 CLIP 系列模型，提供标准化的图像/文本编码接口
支持多卡并行推理
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np


# ─── 模型注册表 ───────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "openai": {
        "loader": "openclip",
        "model_name": "ViT-L-14",
        "pretrained": "openai",
        "input_size": 224,
    },
    "laion": {
        "loader": "openclip",
        "model_name": "ViT-L-14",
        "pretrained": "laion2b_s32b_b82k",
        "input_size": 224,
    },
    "siglip2": {
        "loader": "hf_siglip",
        "model_name": "ViT-L-16-SigLIP2-256",
        "input_size": 256,
    },
    "eva02": {
        "loader": "openclip",
        "model_name": "EVA02-L-14",
        "pretrained": "merged2b_s4b_b131k",
        "input_size": 224,
    },
    "dfn": {
        "loader": "openclip",
        "model_name": "ViT-L-14",
        "pretrained": "dfn2b",
        "input_size": 224,
    },
}


REPO_ROOT = Path(__file__).resolve().parents[3]
CKPTS_ROOT = REPO_ROOT / "scb5_zeroshot" / "ckpts"
LOCAL_CHECKPOINTS = {
    "openai": CKPTS_ROOT / "clip_openai_vitl14",
    "laion": CKPTS_ROOT / "laion_vitl14" / "open_clip_pytorch_model.bin",
    "siglip2": CKPTS_ROOT / "siglip_vitl16_256" / "open_clip_pytorch_model.bin",
    "eva02": CKPTS_ROOT / "eva02_clip_vitl14" / "open_clip_pytorch_model.bin",
    "dfn": CKPTS_ROOT / "dfn_clip_vitl14" / "open_clip_pytorch_model.bin",
}


def _extract_state_dict(ckpt_obj):
    """Normalize common checkpoint wrappers to a plain state_dict mapping."""
    if isinstance(ckpt_obj, dict):
        for key in ("model_state_dict", "state_dict", "weights", "model"):
            if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
                return ckpt_obj[key]
    return ckpt_obj


# ─── 基础 CLIP 封装 ──────────────────────────────────────────────────────────

class CLIPModel(nn.Module):
    """统一 CLIP 接口"""

    def __init__(self, model_key: str, device: str = "cuda"):
        super().__init__()
        self.model_key = model_key
        self.device = device
        cfg = MODEL_REGISTRY[model_key]
        self.input_size = cfg["input_size"]

        self.model, self.preprocess, self.tokenizer = self._load_model(cfg, model_key)
        self.model = self.model.to(device).eval()

    def _load_model(self, cfg: Dict, model_key: str):
        local_ckpt = LOCAL_CHECKPOINTS.get(model_key)

        if model_key == "openai":
            try:
                import clip as openai_clip
                download_root = str(local_ckpt) if local_ckpt else "./ckpts/clip_openai_vitl14"
                model, _ = openai_clip.load("ViT-L/14", device="cpu", download_root=download_root)
                preprocess = None
                tokenizer = open_clip.get_tokenizer("ViT-L-14")
                return model, preprocess, tokenizer
            except Exception as e:
                print(f"[CLIPModel] OpenAI local load failed: {e}")

        if model_key in {"laion", "eva02", "dfn"} and local_ckpt and local_ckpt.exists():
            try:
                arch = cfg["model_name"]
                print(f"[CLIPModel] Loading {model_key} from local checkpoint: {local_ckpt}")
                model = open_clip.create_model(arch, pretrained=None)
                try:
                    ckpt = torch.load(local_ckpt, map_location="cpu", weights_only=True)
                except TypeError:
                    ckpt = torch.load(local_ckpt, map_location="cpu")
                ckpt = _extract_state_dict(ckpt)
                model.load_state_dict(ckpt, strict=True)
                model.eval()
                preprocess = None
                tokenizer = open_clip.get_tokenizer(arch)
                return model, preprocess, tokenizer
            except Exception as e:
                print(f"[CLIPModel] Local {model_key} load failed: {e}")

        if model_key == "siglip2" and local_ckpt and local_ckpt.exists():
            try:
                arch = "ViT-L-16-SigLIP2-256"
                print(f"[CLIPModel] Loading siglip2 from local checkpoint: {local_ckpt}")
                model = open_clip.create_model(arch, pretrained=None)
                try:
                    ckpt = torch.load(local_ckpt, map_location="cpu", weights_only=True)
                except TypeError:
                    ckpt = torch.load(local_ckpt, map_location="cpu")
                model.load_state_dict(ckpt, strict=True)
                model.eval()
                preprocess = None
                tokenizer = open_clip.get_tokenizer(f"hf-hub:{local_ckpt.parent}", context_length=64)
                return model, preprocess, tokenizer
            except Exception as e:
                print(f"[CLIPModel] Local siglip2 load failed: {e}")

        # Helper: try to match a checkpoint to an arch by key overlap
        def _match_ckpt_to_arch(ckpt_path: str, arch_candidates: List[str]):
            import torch
            best = None
            try:
                ck = torch.load(ckpt_path, map_location="cpu")
            except Exception:
                return None
            if hasattr(ck, 'keys'):
                ck_keys = set(ck.keys())
            else:
                return None

            import open_clip
            for arch in arch_candidates:
                try:
                    # create lightweight model to get expected keys
                    m = open_clip.create_model(arch, pretrained=None)
                    model_keys = set(m.state_dict().keys())
                    common = len(ck_keys & model_keys)
                    score = common / (len(model_keys) + 1e-12)
                    if best is None or score > best[0]:
                        best = (score, arch)
                    del m
                except Exception:
                    continue
            return best[1] if best else None

        if cfg["loader"] == "openclip":
            # 尝试使用本地 ckpts（优先），可通过环境变量 CKPTS_DIR 指定目录
            ckpts_env = os.environ.get("CKPTS_DIR")
            default_ckpts = Path(__file__).resolve().parents[2] / "ckpts"
            ckpts_dir = Path(ckpts_env) if ckpts_env else default_ckpts

            pretrained_spec = cfg.get("pretrained")
            chosen = str(local_ckpt) if (local_ckpt and local_ckpt.exists()) else None
            # Special-case: prefer offline SigLIP loading when available to avoid HF network calls
            if model_key.lower().startswith("siglip") or (cfg.get("model_name") and "siglip" in cfg.get("model_name").lower()):
                # look for a siglip folder under ckpts
                siglip_candidates = [p for p in ckpts_dir.iterdir() if 'siglip' in p.name.lower()] if ckpts_dir.exists() else []
                if siglip_candidates:
                    siglip_dir = siglip_candidates[0]
                    # prefer explicit bin or safetensors
                    for pattern in ("open_clip_pytorch_model.bin", "*.safetensors", "*.pt"):
                        matches = list(siglip_dir.glob(pattern))
                        if matches:
                            chosen = str(matches[0])
                            break
                    # if nothing matched, maybe the directory itself is a pretrained folder for HF; try it
                    if chosen is None:
                        chosen = str(siglip_dir)
                    if chosen:
                        print(f"[CLIPModel] Found local SigLIP candidate: {chosen}")
            if ckpts_dir.exists() and chosen is None:
                # 搜索常见本地文件名. Prefer checkpoints whose path includes the model key
                candidates = []
                for pattern in ("open_clip_pytorch_model.bin", "*.pt", "*.safetensors"):
                    matches = list(ckpts_dir.rglob(pattern)) if ckpts_dir.is_dir() else []
                    candidates.extend(matches)
                if candidates:
                    # Prefer candidates that include the explicit model_key or model_name in their path
                    preferred = None
                    mk = model_key.lower() if model_key else ''
                    name = cfg.get('model_name','').lower() if cfg.get('model_name') else ''
                    for m in candidates:
                        sp = str(m).lower()
                        if mk and mk in sp:
                            preferred = m
                            break
                        if name and name in sp:
                            preferred = m
                            break
                    if not preferred:
                        # fallback to heuristics matching known short keynames; prefer keyname order
                        for keyname in ("siglip", "openai", "laion", "eva02", "dfn"):
                            for m in candidates:
                                sp = str(m).lower()
                                if keyname in sp:
                                    preferred = m
                                    break
                            if preferred:
                                break
                    chosen = str(preferred or candidates[0])

            # If we found a local checkpoint file, prefer manual offline loading
            if chosen:
                print(f"[CLIPModel] Using local checkpoint for {cfg['model_name']}: {chosen}")
                try:
                    # Special-case: OpenAI CLIP may be saved as a TorchScript .pt archive
                    # or expected to be loaded via the `clip` package. Try those first.
                    from pathlib import Path as _P
                    chosen_path = _P(chosen)
                    if chosen_path.suffix == '.pt' or 'clip_openai' in str(chosen_path.parent).lower():
                        try:
                            import clip as openai_clip
                            # openai_clip.load will look into download_root for a matching file
                            print(f"[CLIPModel] Attempting openai/torchscript load from {chosen_path.parent}")
                            m, p = openai_clip.load("ViT-L/14", device="cpu", download_root=str(chosen_path.parent))
                            tokenizer = None
                            preprocess = p
                            model = m
                            return model, preprocess, tokenizer
                        except Exception:
                            try:
                                # fallback: try torch.jit.load for scripted module
                                import torch as _torch
                                print(f"[CLIPModel] Trying torch.jit.load for {chosen}")
                                scripted = _torch.jit.load(str(chosen_path), map_location='cpu')
                                model = scripted
                                # no preprocess/tokenizer available for scripted module
                                tokenizer = None
                                preprocess = None
                                return model, preprocess, tokenizer
                            except Exception:
                                pass
                    
                    # attempt to auto-match arch by key overlap
                    candidate_archs = [cfg.get("model_name")] + ["ViT-L-14", "ViT-B-16", "ViT-L-16-SigLIP2-256", "ViT-L-16-SigLIP-256"]
                    candidate_archs = [a for a in candidate_archs if a]
                    matched = _match_ckpt_to_arch(chosen, candidate_archs)
                    arch_to_use = matched or cfg.get("model_name")
                    print(f"[CLIPModel] Matched arch: {arch_to_use}")

                    # Prefer using open_clip.create_model_and_transforms with the local path
                    try:
                        print(f"[CLIPModel] Attempting open_clip.create_model_and_transforms with pretrained={chosen}")
                        model, _, preprocess = open_clip.create_model_and_transforms(arch_to_use, pretrained=chosen)
                        tokenizer = open_clip.get_tokenizer(arch_to_use)
                        return model, preprocess, tokenizer
                    except Exception as e_create:
                        print(f"[CLIPModel] create_model_and_transforms with local pretrained failed: {e_create}; falling back to manual state_dict load")

                    # Fallback: manual state_dict load (weights_only when supported)
                    model = open_clip.create_model(arch_to_use, pretrained=None)
                    try:
                        ck = torch.load(chosen, map_location="cpu", weights_only=True)
                    except TypeError:
                        ck = torch.load(chosen, map_location="cpu")
                    ck = _extract_state_dict(ck)
                    try:
                        model.load_state_dict(ck, strict=True)
                    except Exception:
                        model.load_state_dict(ck, strict=False)
                    _, _, preprocess = open_clip.create_model_and_transforms(arch_to_use, pretrained=None)
                    # For SigLIP-like models, try to load tokenizer from local hf-style folder if possible
                    try:
                        from pathlib import Path as _P
                        chosen_path = _P(chosen)
                        # If the chosen path is a directory, use hf-hub:dir to load tokenizer offline
                        if chosen_path.is_dir():
                            tokenizer = open_clip.get_tokenizer(f"hf-hub:{chosen_path}")
                        else:
                            # If the chosen file sits inside a siglip-style folder, prefer the parent dir
                            parent = chosen_path.parent
                            if 'siglip' in parent.name.lower() and parent.exists():
                                try:
                                    tokenizer = open_clip.get_tokenizer(f"hf-hub:{parent}")
                                except Exception:
                                    tokenizer = open_clip.get_tokenizer(arch_to_use)
                            else:
                                tokenizer = open_clip.get_tokenizer(arch_to_use)
                    except Exception:
                        tokenizer = open_clip.get_tokenizer(arch_to_use)
                    return model, preprocess, tokenizer
                except Exception as e:
                    print(f"[CLIPModel] Local open_clip manual load failed: {e}")

            # Fall back to the standard create_model_and_transforms (may use cached/hub)
            print(f"[CLIPModel] No usable local checkpoint for {cfg['model_name']}, using pretrained='{pretrained_spec}'")
            model, _, preprocess = open_clip.create_model_and_transforms(
                cfg["model_name"],
                pretrained=pretrained_spec,
            )
            tokenizer = open_clip.get_tokenizer(cfg["model_name"])
            return model, preprocess, tokenizer

        elif cfg["loader"] == "hf_siglip":
            from pathlib import Path
            import shutil

            ckpts_env = os.environ.get("CKPTS_DIR")
            default_ckpts = Path(__file__).resolve().parents[2] / "ckpts" / "siglip_vitl16_256"
            siglip_dir = (Path(ckpts_env) / "siglip_vitl16_256") if ckpts_env else default_ckpts

            # prefer a local checkpoint file if present
            chosen_ckpt = None
            if siglip_dir.exists():
                # if siglip_dir is a file, use it
                if siglip_dir.is_file():
                    chosen_ckpt = str(siglip_dir)
                else:
                    for pattern in ("open_clip_pytorch_model.bin", "*.pt", "*.safetensors"):
                        matches = list(siglip_dir.rglob(pattern))
                        if matches:
                            chosen_ckpt = str(matches[0])
                            break

            # Attempt offline open_clip load similar to run_experiment.py
            if chosen_ckpt:
                try:
                    arch = "ViT-L-16-SigLIP2-256"
                    print(f"[CLIPModel] Loading SigLIP offline via open_clip: arch={arch}, ckpt={chosen_ckpt}")
                    model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=None)
                    # load checkpoint
                    try:
                        ck = torch.load(chosen_ckpt, map_location="cpu", weights_only=True)
                    except TypeError:
                        ck = torch.load(chosen_ckpt, map_location="cpu")
                    try:
                        model.load_state_dict(ck, strict=True)
                    except Exception:
                        model.load_state_dict(ck, strict=False)
                    # Prefer local hf-hub style tokenizer to keep offline mode stable.
                    tok_dir = str(Path(chosen_ckpt).parent)
                    try:
                        tokenizer = open_clip.get_tokenizer(f"hf-hub:{tok_dir}", context_length=64)
                    except Exception:
                        # Fallback may require network in some environments.
                        tokenizer = open_clip.get_tokenizer(arch, context_length=64)
                    return model, preprocess, tokenizer
                except Exception as e:
                    print(f"[CLIPModel] Offline open_clip SigLIP load failed: {e}")

            # Fallback: build HF-compatible dir and try transformers (existing logic)
            try:
                from transformers import AutoProcessor, AutoModel
                compat_dir = siglip_dir / "hf_compat"
                compat_dir.mkdir(exist_ok=True)

                def _ensure_link(src_name, dst_name):
                    src = siglip_dir / src_name
                    dst = compat_dir / dst_name
                    if src.exists() and not dst.exists():
                        try:
                            dst.symlink_to(src)
                        except Exception:
                            shutil.copy(src, dst)

                _ensure_link("open_clip_config.json", "config.json")
                _ensure_link("open_clip_pytorch_model.bin", "pytorch_model.bin")
                _ensure_link("open_clip_model.safetensors", "model.safetensors")
                for fname in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
                    s = siglip_dir / fname
                    d = compat_dir / fname
                    if s.exists() and not d.exists():
                        try:
                            d.symlink_to(s)
                        except Exception:
                            shutil.copy(s, d)

                try:
                    model = AutoModel.from_pretrained(str(compat_dir), local_files_only=True, trust_remote_code=True)
                    processor = AutoProcessor.from_pretrained(str(compat_dir), local_files_only=True, trust_remote_code=True)
                except Exception as e:
                    print(f"[CLIPModel] Transformers local load failed: {e}")
                    print("[CLIPModel] Attempting remote HF load for SigLIP (may fail offline)")
                    model = AutoModel.from_pretrained(cfg["model_name"], trust_remote_code=True)
                    processor = AutoProcessor.from_pretrained(cfg["model_name"], trust_remote_code=True)
                return model, processor, processor
            except Exception:
                # if anything above fails and we cannot use transformers, raise to notify caller
                raise
        else:
            raise ValueError(f"Unknown loader: {cfg['loader']}")

    @torch.no_grad()
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: (N, C, H, W) — 已经过 CLIP preprocess
        返回 L2-normalized 特征 (N, D)
        """
        cfg = MODEL_REGISTRY[self.model_key]
        if cfg["loader"] == "hf_siglip":
            # SigLIP2 may be HF model or open_clip model depending on offline loading path.
            if hasattr(self.model, "get_image_features"):
                outputs = self.model.get_image_features(pixel_values=images)
                return F.normalize(outputs, dim=-1)
            feats = self.model.encode_image(images)
            return F.normalize(feats.float(), dim=-1)
        else:
            feats = self.model.encode_image(images)
            return F.normalize(feats.float(), dim=-1)

    @torch.no_grad()
    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        """
        texts: List[str]
        返回 L2-normalized 特征 (N, D)
        """
        cfg = MODEL_REGISTRY[self.model_key]
        # If tokenizer is missing (e.g., loaded scripted/openai torchscript model),
        # attempt a fallback to open_clip tokenizer for the configured arch.
        if self.tokenizer is None:
            try:
                import open_clip
                arch = cfg.get("model_name") or cfg.get("loader")
                self.tokenizer = open_clip.get_tokenizer(arch)
                print(f"[CLIPModel] Fallback: loaded tokenizer for arch={arch}")
            except Exception:
                raise RuntimeError(
                    f"Tokenizer unavailable for model_key={self.model_key}; "
                    "ensure local checkpoint includes tokenizer or use open_clip transforms"
                )

        if cfg["loader"] == "hf_siglip":
            # tokenizer may be HF-style (dict outputs) or open_clip-style (tensor outputs)
            try:
                inputs = self.tokenizer(
                    texts, return_tensors="pt", padding=True, truncation=True
                )
            except TypeError:
                tokens = self.tokenizer(texts).to(self.device)
                if hasattr(self.model, "encode_text"):
                    feats = self.model.encode_text(tokens)
                    return F.normalize(feats.float(), dim=-1)
                outputs = self.model.get_text_features(tokens)
                return F.normalize(outputs.float(), dim=-1)
            # ensure tensors are torch tensors and on device
            if isinstance(inputs, dict):
                # determine model max sequence length from positional embeddings if available
                max_seq = None
                try:
                    max_seq = getattr(self.model, 'positional_embedding', None)
                    if max_seq is not None:
                        max_seq = int(max_seq.shape[0])
                except Exception:
                    max_seq = None

                if max_seq is not None and 'input_ids' in inputs:
                    seq_len = inputs['input_ids'].shape[1]
                    if seq_len > max_seq:
                        # truncate to model's positional embedding length
                        inputs['input_ids'] = inputs['input_ids'][:, :max_seq]
                        if 'attention_mask' in inputs:
                            inputs['attention_mask'] = inputs['attention_mask'][:, :max_seq]
                        if 'token_type_ids' in inputs:
                            inputs['token_type_ids'] = inputs['token_type_ids'][:, :max_seq]

                # move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model.get_text_features(**inputs)
                return F.normalize(outputs, dim=-1)
            else:
                # fallback: tokenizer returned tensor
                inputs = inputs.to(self.device)
                outputs = self.model.get_text_features(inputs)
                return F.normalize(outputs, dim=-1)
        else:
            tokens = self.tokenizer(texts).to(self.device)
            feats = self.model.encode_text(tokens)
            return F.normalize(feats.float(), dim=-1)

    def compute_similarity(
        self,
        image_feats: torch.Tensor,   # (N, D)
        text_feats: torch.Tensor,    # (C, D)
    ) -> torch.Tensor:
        """余弦相似度矩阵 (N, C)"""
        return (image_feats @ text_feats.T)


# ─── CAPE 文本特征构建 ────────────────────────────────────────────────────────

def build_cape_text_features(
    model: CLIPModel,
    class_prompts: Dict[str, List[str]],   # {class_name: [prompt1, prompt2, ...]}
    classes: List[str],
    batch_size: int = 256,
) -> torch.Tensor:
    """
    构建 CAPE 类别文本特征矩阵
    对每个类别的多个提示取平均，返回 (C, D)
    """
    class_feats = []
    for cls in classes:
        prompts = class_prompts.get(cls, [cls])
        # 分批编码
        all_feats = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i+batch_size]
            feats = model.encode_texts(batch)    # (k, D)
            all_feats.append(feats)
        all_feats = torch.cat(all_feats, dim=0)  # (k, D)
        # 平均池化 + 重新归一化
        cls_feat = F.normalize(all_feats.mean(dim=0, keepdim=True), dim=-1)  # (1, D)
        class_feats.append(cls_feat)
    return torch.cat(class_feats, dim=0)   # (C, D)


# ─── 批量推理 ────────────────────────────────────────────────────────────────

@torch.no_grad()
def batch_inference(
    model: CLIPModel,
    dataloader,
    text_features: torch.Tensor,     # (C, D) — 预计算的类别文本特征
    multilabel: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    批量零样本推理
    返回:
        logits: (N, C) numpy array
        labels: (N,) or (N, C) numpy array
    """
    all_logits = []
    all_labels = []

    for batch in dataloader:
        images = batch["image"].to(model.device)
        labels = batch["label"]

        img_feats = model.encode_images(images)              # (B, D)
        logits = model.compute_similarity(img_feats, text_features)  # (B, C)

        all_logits.append(logits.cpu().numpy())
        all_labels.append(labels.numpy())

    logits = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return logits, labels


# ─── 多卡特征提取 ────────────────────────────────────────────────────────────

class MultiGPUFeatureExtractor:
    """
    用 4 张 V100 并行提取特征
    每张卡负责一个或多个模型
    """

    def __init__(self, model_keys: List[str], gpus: List[int]):
        self.model_keys = model_keys
        self.gpus = gpus
        # 将模型分配到不同 GPU
        self.model_gpu_map = {}
        for i, key in enumerate(model_keys):
            gpu_id = gpus[i % len(gpus)]
            self.model_gpu_map[key] = f"cuda:{gpu_id}"

    def extract_all(
        self,
        dataloader,
        class_prompts_dict: Dict[str, Dict[str, List[str]]],
        classes: List[str],
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        并行提取所有模型的特征
        返回: {model_key: (logits, labels)}
        """
        results = {}
        # try to infer whether dataset is multilabel from one sample
        multilabel = False
        try:
            first_batch = next(iter(dataloader))
            lbls = first_batch.get('label') if isinstance(first_batch, dict) else None
            import numpy as _np
            if lbls is not None:
                if hasattr(lbls, 'ndim') and lbls.ndim > 1:
                    multilabel = True
                elif isinstance(lbls, _np.ndarray) and lbls.ndim > 1:
                    multilabel = True
        except Exception:
            multilabel = False

        for model_key in self.model_keys:
            device = self.model_gpu_map[model_key]
            print(f"  [{model_key}] on {device}")
            model = CLIPModel(model_key, device=device)

            # 更新 dataloader 的设备目标（数据仍在 CPU，由 DataLoader pin_memory 处理）
            class_prompts = class_prompts_dict.get(model_key, {})
            text_feats = build_cape_text_features(model, class_prompts, classes)

            logits, labels = batch_inference(
                model, dataloader, text_feats,
                multilabel=multilabel
            )
            results[model_key] = (logits, labels)
            del model
            torch.cuda.empty_cache()

        return results


if __name__ == "__main__":
    # 快速测试
    print("Testing CLIPModel loading...")
    model = CLIPModel("openai", device="cpu")
    dummy_imgs = torch.randn(2, 3, 224, 224)
    feats = model.encode_images(dummy_imgs)
    print(f"  Image features: {feats.shape}")

    text_feats = model.encode_texts(["a teacher writing on the board", "a student reading"])
    print(f"  Text features: {text_feats.shape}")

    sim = model.compute_similarity(feats, text_feats)
    print(f"  Similarity: {sim.shape}")
    print("CLIP model test passed!")
