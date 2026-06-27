"""
SCB5 Teacher Behavior - Fair Model Comparison v3
=================================================
修复：支持本地 checkpoint 加载，不依赖网络
统一框架：ViT-L/14, OpenCLIP, 224x224

模型对应关系：
  clip   -> OpenAI CLIP ViT-L/14        (openai 权重，通常已缓存)
  laion  -> CLIP ViT-L/14 LAION2B       (SLIP-style 大数据训练)
  flip   -> CLIP ViT-L/14 FLIP masking  (高效masked预训练)
  siglip -> SigLIP ViT-L/16-256         (TULIP对标baseline)

Usage:
  # 单模型运行
  python run_experiment.py --model clip --mode zeroshot --gpu 0

  # 指定本地ckpt路径
  python run_experiment.py --model laion --mode zeroshot --gpu 0 \
      --ckpt_laion /path/to/open_clip_pytorch_model.bin

  # 并行
  bash run_all_parallel.sh
"""

import os, sys, json, argparse, logging, time
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
import open_clip
from torchvision import transforms

# ─────────────────────────────────────────────
# 日志（写入 logs/ 目录）
# ─────────────────────────────────────────────
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(LOG_DIR / "experiment.log")),
    ],
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# 数据集路径
# ─────────────────────────────────────────────
# Prefer workspace-local `datasets_scb/` if present; fallback to the original hardcoded path.
_local_datasets = Path(__file__).resolve().parents[1] / "datasets_scb"
if _local_datasets.exists():
    # Prefer the dataset's extracted folder if present under datasets_scb
    teacher_dir = _local_datasets / "SCB5_TeacherBehavior" / "SCB5_Teacher_Behavior_Stand_BlackBoard_Sreen_20250406-2"
    if teacher_dir.exists():
        DATASET_ROOT = teacher_dir
    else:
        DATASET_ROOT = _local_datasets
else:
    DATASET_ROOT = Path(__file__).resolve().parent / "data"

CLASS_NAMES = [
    "guide", "answer", "On-stage interaction", "blackboard-writing",
    "teacher", "stand", "screen", "blackBoard",
]
NUM_CLASSES = len(CLASS_NAMES)

CLASS_DESCRIPTIONS = {
    "guide":                "guiding students through the lesson",
    "answer":               "answering questions from students",
    "On-stage interaction": "interacting with students on stage",
    "blackboard-writing":   "writing on the blackboard with chalk",
    "teacher":              "standing and teaching in front of the class",
    "stand":                "standing at the front of the classroom",
    "screen":               "pointing at or presenting on the projection screen",
    "blackBoard":           "using the blackboard to explain content",
}

PROMPT_TEMPLATES = {
    "simple":   ["{cls}"],
    "medium":   ["a photo of {cls}", "an image showing {cls}"],
    "detailed": [
        "a photo of a teacher {cls} in a classroom",
        "an image of a teacher {cls} at the blackboard",
        "a classroom scene showing a teacher {cls}",
        "a teacher {cls} during a lecture",
    ],
}

# ─────────────────────────────────────────────
# 模型配置
# ─────────────────────────────────────────────
# open_clip 中可用的 ViT-L/14 权重（按优先级排列）
LAION_PRETRAINED_CANDIDATES = [
    "laion2b_s32b_b82k",        # 最优先（最强）
    "laion400m_e32",
    "laion400m_e31",
]

FLIP_PRETRAINED_CANDIDATES = [
    "laion2b_s32b_b82k",        # FLIP 专用 ckpt，若不存在用此近似
    "laion2b_s32b_b82k",
]

# SigLIP 可用的型号（按优先级，从你的 open_clip 版本中选可用的）
SIGLIP_ARCH_CANDIDATES = [
    ("ViT-L-16-SigLIP2-256",     "webli"),
    ("ViT-L-16-SigLIP2-256",      "webli"),
    ("ViT-L-16-SigLIP-256",       "webli"),
    ("ViT-B-16-SigLIP2",          "webli"),
    ("ViT-B-16-SigLIP",           "webli"),
]

# ─────────────────────────────────────────────
# Preprocess
# ─────────────────────────────────────────────
def build_transform(image_size=224, is_siglip=False):
    if is_siglip:
        norm = transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
    else:
        norm = transforms.Normalize(
            mean=(0.48145466, 0.4578275,  0.40821073),
            std =(0.26862954, 0.26130258, 0.27577711),
        )
    return transforms.Compose([
        transforms.Resize(image_size,
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        norm,
    ])

# ─────────────────────────────────────────────
# 核心：统一模型加载（支持本地路径）
# ─────────────────────────────────────────────
def _try_load_openclip(arch, pretrained, device, local_path=None):
    """
    尝试加载 OpenCLIP 模型。
    local_path: 本地 .bin/.safetensors 文件路径（可选）
    """
    if local_path and Path(local_path).exists():
        log.info(f"  Loading from local path: {local_path}")
        model, _, _ = open_clip.create_model_and_transforms(
            arch, pretrained=local_path, device=device
        )
    else:
        log.info(f"  Loading {arch} / {pretrained} from cache/hub ...")
        model, _, _ = open_clip.create_model_and_transforms(
            arch, pretrained=pretrained, device=device
        )
    model.eval()
    return model


def load_model(model_type: str, device: torch.device,
               local_paths: dict = None):
    """
    local_paths: {
        "laion":  "/path/to/laion_vitl14.bin",
        "flip":   "/path/to/flip_vitl14.bin",
        "siglip": "/path/to/siglip.bin",
    }
    """
    local_paths = local_paths or {}

    if model_type == "clip":
        log.info("Loading CLIP (OpenAI, ViT-L/14) from local file ...")
        local = local_paths.get("clip") or "./ckpts/clip_openai_vitl14/pytorch_model.bin"
        try:
            import clip as openai_clip
            m, preproc = openai_clip.load("ViT-L/14", device=device,
                                          download_root="./ckpts/clip_openai_vitl14")
            # wrap into open_clip-style interface
            import open_clip as oc
            tokenizer = oc.get_tokenizer("ViT-L-14")
            model = m
            preprocess = build_transform(224, is_siglip=False)
        except Exception:
            # fallback: open_clip direct bin load
            model = _try_load_openclip("ViT-L-14", local, device)
            tokenizer  = open_clip.get_tokenizer("ViT-L-14")
            preprocess = build_transform(224, is_siglip=False)
        desc = "CLIP (OpenAI, ViT-L/14)"

    elif model_type == "laion":
        log.info("Loading CLIP-LAION2B (ViT-L/14, SLIP-style) ...")
        local = local_paths.get("laion") or "./ckpts/laion_vitl14/open_clip_pytorch_model.bin"
        model = None
        # Strategy 1: load from local checkpoint directly
        if local and Path(local).exists():
            try:
                log.info(f"  Loading LAION from local: {local}")
                model = open_clip.create_model("ViT-L-14", pretrained=None, device=device)
                ckpt = torch.load(local, map_location=device, weights_only=True)
                model.load_state_dict(ckpt, strict=True)
                model.eval()
                desc = "CLIP-LAION2B (ViT-L/14, local)"
            except Exception as e:
                log.warning(f"  Local load failed: {e}")
                model = None
        # Strategy 2: fallback to online
        if model is None:
            for pretrained in LAION_PRETRAINED_CANDIDATES:
                try:
                    model = _try_load_openclip("ViT-L-14", pretrained, device)
                    desc = f"CLIP-LAION2B (ViT-L/14, {pretrained})"
                    break
                except Exception as e:
                    log.warning(f"  Failed {pretrained}: {e}")
        if model is None:
            raise RuntimeError("All LAION pretrained candidates failed. "
                               "Please download manually (see README).")
        tokenizer  = open_clip.get_tokenizer("ViT-L-14")
        preprocess = build_transform(224, is_siglip=False)

    elif model_type == "flip":
        log.info("Loading FLIP (ViT-L/14, masked pretraining) ...")
        local = local_paths.get("flip")
        model = None
        # Strategy 1: dedicated FLIP checkpoint file
        if local and Path(local).exists():
            try:
                log.info(f"  Loading FLIP from local: {local}")
                model = open_clip.create_model("ViT-L-14", pretrained=None, device=device)
                ckpt = torch.load(local, map_location=device, weights_only=True)
                model.load_state_dict(ckpt, strict=True)
                model.eval()
                desc = "FLIP (ViT-L/14, local)"
            except Exception as e:
                log.warning(f"  FLIP local load failed: {e}")
                model = None
        # Strategy 2: check open_clip registry for FLIP-specific ckpt
        if model is None:
            available = open_clip.list_pretrained()
            flip_ckpts = [(a, p) for a, p in available
                          if "ViT-L" in a and "flip" in p.lower()]
            if flip_ckpts:
                arch, pretrained = flip_ckpts[0]
                log.info(f"  Found FLIP checkpoint: {arch} / {pretrained}")
                try:
                    model = _try_load_openclip(arch, pretrained, device)
                    desc = f"FLIP ({arch}, {pretrained})"
                except Exception as e:
                    log.warning(f"  FLIP ckpt failed: {e}")
        # Strategy 3: fallback to LAION ViT-L (same arch, different training)
        if model is None:
            laion_local = local_paths.get("laion") or "./ckpts/laion_vitl14/open_clip_pytorch_model.bin"
            if Path(laion_local).exists():
                log.warning("  No FLIP-specific ckpt. Using LAION ViT-L/14 as approximation (local).")
                try:
                    model = open_clip.create_model("ViT-L-14", pretrained=None, device=device)
                    ckpt = torch.load(laion_local, map_location=device, weights_only=True)
                    model.load_state_dict(ckpt, strict=True)
                    model.eval()
                    desc = "FLIP-approx (ViT-L/14, LAION-local)"
                except Exception as e:
                    log.warning(f"  Fallback local failed: {e}")
                    model = None
        if model is None:
            # Last resort: try online
            for pretrained in LAION_PRETRAINED_CANDIDATES:
                try:
                    model = _try_load_openclip("ViT-L-14", pretrained, device)
                    desc = f"FLIP-approx (ViT-L/14, {pretrained})"
                    break
                except Exception as e:
                    log.warning(f"  Online fallback {pretrained} failed: {e}")
        if model is None:
            raise RuntimeError("FLIP loading failed completely.")
        tokenizer  = open_clip.get_tokenizer("ViT-L-14")
        preprocess = build_transform(224, is_siglip=False)

    elif model_type == "siglip":
        log.info("Loading SigLIP (TULIP baseline) ...")
        local = local_paths.get("siglip")
        model = None
        tokenizer  = None
        preprocess = None
        desc       = "SigLIP"
        arch       = "ViT-L-16-SigLIP2-256"

        # Determine local checkpoint directory (for offline tokenizer loading)
        ckpt_dir = None
        if local and Path(local).exists():
            ckpt_dir = str(Path(local).parent) if Path(local).is_file() else str(local)

        # Strategy 1: create model without pretrained, load weights manually
        #   (avoids HuggingFace network calls for tokenizer during model init)
        if local and Path(local).exists():
            try:
                log.info(f"  Creating {arch} model (offline) ...")
                model = open_clip.create_model(arch, pretrained=None, device=device)
                ckpt = torch.load(local, map_location=device, weights_only=True)
                model.load_state_dict(ckpt, strict=True)
                model.eval()
                tokenizer  = open_clip.get_tokenizer(
                    f"hf-hub:{ckpt_dir}", context_length=64)
                preprocess = build_transform(256, is_siglip=True)
                desc       = f"SigLIP ({arch})"
                log.info(f"  Loaded: {arch} from {local}  img_size=256")
            except Exception as e:
                log.warning(f"  SigLIP offline load failed: {e}")
                model = None

        # Strategy 2: fallback to online candidates
        if model is None:
            for arch, pretrained in SIGLIP_ARCH_CANDIDATES:
                try:
                    model = _try_load_openclip(arch, pretrained, device, local)
                    img_size = 256 if "256" in arch else (384 if ("384" in arch or "378" in arch) else 224)
                    desc       = f"SigLIP ({arch})"
                    tokenizer  = open_clip.get_tokenizer(arch)
                    preprocess = build_transform(img_size, is_siglip=True)
                    log.info(f"  Loaded: {arch} / {pretrained}  img_size={img_size}")
                    break
                except Exception as e:
                    log.warning(f"  SigLIP {arch} failed: {e}")

        if model is None or preprocess is None:
            raise RuntimeError(
                "All SigLIP candidates failed.\n"
                "Please download manually:\n"
                "  huggingface-cli download timm/ViT-L-16-SigLIP2-256 "
                "--local-dir ./ckpts/siglip"
            )
    else:
        raise ValueError(f"Unknown model: {model_type}")

    def encode_image(imgs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return model.encode_image(imgs)

    def encode_text(texts: list) -> torch.Tensor:
        tokens = tokenizer(texts).to(device)
        with torch.no_grad():
            return model.encode_text(tokens)

    return {
        "preprocess":   preprocess,
        "encode_image": encode_image,
        "encode_text":  encode_text,
        "desc":         desc,
    }


# ─────────────────────────────────────────────
# 相似度
# ─────────────────────────────────────────────
def compute_similarity(img_feats, txt_feats):
    img_feats = F.normalize(img_feats.float(), dim=-1)
    txt_feats = F.normalize(txt_feats.float(), dim=-1)
    return img_feats @ txt_feats.T


# ─────────────────────────────────────────────
# 文本特征
# ─────────────────────────────────────────────
def build_text_features(encode_text_fn, template_set="detailed"):
    templates = PROMPT_TEMPLATES[template_set]
    all_feats = []
    for cname in CLASS_NAMES:
        desc    = CLASS_DESCRIPTIONS[cname]
        prompts = [t.format(cls=desc) for t in templates]
        feats   = encode_text_fn(prompts)
        feats   = F.normalize(feats.float(), dim=-1).mean(0)
        feats   = F.normalize(feats, dim=-1)
        all_feats.append(feats)
    return torch.stack(all_feats)


def build_text_features_from_dict(encode_text_fn, prompts_dict):
    """Build text features from a {class_name: [prompt_str, ...]} dict."""
    all_feats = []
    for cname in CLASS_NAMES:
        prompts = prompts_dict[cname]
        feats   = encode_text_fn(prompts)
        feats   = F.normalize(feats.float(), dim=-1).mean(0)
        feats   = F.normalize(feats, dim=-1)
        all_feats.append(feats)
    return torch.stack(all_feats)


# ─────────────────────────────────────────────
# 数据集
# ─────────────────────────────────────────────
class SCB5Dataset:
    def __init__(self, split="val"):
        self.img_dir = DATASET_ROOT / "images" / split
        self.lbl_dir = DATASET_ROOT / "labels" / split
        self.samples = []
        self._load(split)

    def _load(self, split):
        missing = 0
        for img_path in sorted(self.img_dir.glob("*.jpg")):
            lbl = self.lbl_dir / (img_path.stem + ".txt")
            if not lbl.exists():
                missing += 1
                continue
            cids = set()
            for line in lbl.read_text().splitlines():
                line = line.strip()
                if line:
                    cids.add(int(line.split()[0]))
            if cids:
                self.samples.append((img_path, cids))
        log.info(f"[{split}] {len(self.samples)} imgs, {missing} skipped")
        cnt = defaultdict(int)
        for _, cids in self.samples:
            for c in cids:
                cnt[c] += 1
        for i, n in enumerate(CLASS_NAMES):
            log.info(f"  [{i}] {n:30s}: {cnt[i]:4d}")


# ─────────────────────────────────────────────
# Zero-shot 评估
# ─────────────────────────────────────────────
def evaluate_zeroshot(dataset, model_dict, text_feats,
                       device, batch_size=64):
    enc_img = model_dict["encode_image"]
    preproc = model_dict["preprocess"]
    top1 = top3 = total = 0
    pc_tp  = defaultdict(int)
    pc_tot = defaultdict(int)
    buf_imgs, buf_gts = [], []

    def flush():
        nonlocal top1, top3, total
        if not buf_imgs:
            return
        imgs = torch.stack(buf_imgs).to(device)
        sims = compute_similarity(enc_img(imgs), text_feats)
        for i, gt in enumerate(buf_gts):
            p1 = sims[i].argmax().item()
            p3 = sims[i].topk(3).indices.tolist()
            if p1 in gt: top1 += 1
            if any(p in gt for p in p3): top3 += 1
            for c in gt:
                pc_tot[c] += 1
                if p1 == c: pc_tp[c] += 1
            total += 1
        buf_imgs.clear(); buf_gts.clear()

    for img_path, gt in tqdm(dataset.samples, desc="Zero-shot"):
        try:
            img = preproc(Image.open(img_path).convert("RGB"))
        except Exception as e:
            log.warning(f"Skip {img_path}: {e}")
            continue
        buf_imgs.append(img); buf_gts.append(gt)
        if len(buf_imgs) >= batch_size:
            flush()
    flush()

    pcr = {CLASS_NAMES[c]: round(pc_tp[c]/pc_tot[c]*100, 2)
           if pc_tot[c] > 0 else None
           for c in range(NUM_CLASSES)}
    return {
        "total": total,
        "top1":  round(top1/total*100, 2),
        "top3":  round(top3/total*100, 2),
        "per_class_recall": pcr,
    }


def evaluate_zeroshot_full(dataset, model_dict, text_feats,
                           device, batch_size=64):
    """Like evaluate_zeroshot but also returns y_true / y_pred lists."""
    enc_img = model_dict["encode_image"]
    preproc = model_dict["preprocess"]
    top1 = top3 = total = 0
    y_true_all, y_pred_all = [], []
    buf_imgs, buf_gts = [], []

    def flush():
        nonlocal top1, top3, total
        if not buf_imgs:
            return
        imgs = torch.stack(buf_imgs).to(device)
        sims = compute_similarity(enc_img(imgs), text_feats)
        for i, gt in enumerate(buf_gts):
            p1 = sims[i].argmax().item()
            p3 = sims[i].topk(3).indices.tolist()
            gt_label = min(gt)  # primary label
            y_true_all.append(gt_label)
            y_pred_all.append(p1)
            if p1 in gt: top1 += 1
            if any(p in gt for p in p3): top3 += 1
            total += 1
        buf_imgs.clear(); buf_gts.clear()

    for img_path, gt in tqdm(dataset.samples, desc="Zero-shot"):
        try:
            img = preproc(Image.open(img_path).convert("RGB"))
        except Exception as e:
            log.warning(f"Skip {img_path}: {e}")
            continue
        buf_imgs.append(img); buf_gts.append(gt)
        if len(buf_imgs) >= batch_size:
            flush()
    flush()

    return {
        "total":  total,
        "top1":   round(top1/total*100, 2) if total else 0,
        "top3":   round(top3/total*100, 2) if total else 0,
        "y_true": y_true_all,
        "y_pred": y_pred_all,
    }


# ─────────────────────────────────────────────
# Few-shot Linear Probe
# ─────────────────────────────────────────────
def extract_features(samples, preproc, enc_img, device,
                     batch_size=64, tag=""):
    all_f, all_l = [], []
    buf_i, buf_l = [], []

    def flush():
        if not buf_i: return
        t = torch.stack(buf_i).to(device)
        f = enc_img(t).float().cpu().numpy()
        all_f.append(f); all_l.extend(buf_l)
        buf_i.clear(); buf_l.clear()

    for img_path, gt in tqdm(samples, desc=tag or "Features"):
        try:
            img = preproc(Image.open(img_path).convert("RGB"))
        except Exception:
            continue
        buf_i.append(img); buf_l.append(min(gt))
        if len(buf_i) >= batch_size: flush()
    flush()
    return np.vstack(all_f), all_l


def evaluate_fewshot(ds_train, ds_val, model_dict, device,
                     shots=(1,2,5,10,20), batch_size=64):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import normalize

    preproc = model_dict["preprocess"]
    enc_img = model_dict["encode_image"]

    val_f, val_l = extract_features(ds_val.samples, preproc, enc_img,
                                     device, batch_size, "Val-feat")
    tr_f,  tr_l  = extract_features(ds_train.samples, preproc, enc_img,
                                     device, batch_size, "Train-feat")
    results = {}
    for K in shots:
        sel = []
        for c in range(NUM_CLASSES):
            idxs = [i for i,l in enumerate(tr_l) if l == c]
            if idxs:
                sel += np.random.choice(idxs, min(K,len(idxs)),
                                        replace=False).tolist()
        clf = LogisticRegression(max_iter=1000, C=0.316,
                                 random_state=42, n_jobs=-1)
        clf.fit(normalize(tr_f[sel]), [tr_l[i] for i in sel])
        preds = clf.predict(normalize(val_f))
        acc = sum(p==g for p,g in zip(preds,val_l)) / len(val_l) * 100
        results[K] = round(acc, 2)
        log.info(f"  K={K:2d} -> {acc:.2f}%")
    return results


# ─────────────────────────────────────────────
# Prompt 消融
# ─────────────────────────────────────────────
def evaluate_prompt_ablation(dataset, model_dict, device, batch_size=64):
    results = {}
    for tpl in ("simple", "medium", "detailed"):
        tf  = build_text_features(model_dict["encode_text"], tpl)
        res = evaluate_zeroshot(dataset, model_dict, tf, device, batch_size)
        results[tpl] = {"top1": res["top1"], "top3": res["top3"]}
        log.info(f"  [{tpl}] top1={res['top1']}%")
    return results


# ─────────────────────────────────────────────
# 输出
# ─────────────────────────────────────────────
def save_results(results, model_name):
    d = Path("results"); d.mkdir(exist_ok=True)
    f = d / f"{model_name}_{int(time.time())}.json"
    with open(f, "w") as fp:
        json.dump(results, fp, indent=2, ensure_ascii=False)
    log.info(f"Saved: {f}")


def print_summary(results, model_name):
    print("\n" + "="*62)
    print(f"  {results.get('desc', model_name.upper())}")
    print("="*62)
    if "zeroshot" in results:
        zs = results["zeroshot"]
        print(f"  Top-1 : {zs['top1']:.2f}%   Top-3 : {zs['top3']:.2f}%")
        print(f"  Images: {zs['total']}")
        print("\n  Per-class Recall (Top-1):")
        for cls, val in zs["per_class_recall"].items():
            bar = "█" * int((val or 0)/5)
            v   = f"{val:.1f}%" if val is not None else "N/A"
            print(f"    {cls:30s} {v:>7}  {bar}")
    if "prompt_ablation" in results:
        print("\n  Prompt Ablation:")
        for tpl, v in results["prompt_ablation"].items():
            print(f"    {tpl:10s}: top1={v['top1']:.2f}%  top3={v['top3']:.2f}%")
    if "fewshot" in results:
        print("\n  Few-shot Linear Probe:")
        for k, acc in sorted(results["fewshot"].items()):
            print(f"    {k:2d}-shot: {acc:.2f}%")
    print("="*62+"\n")


# ─────────────────────────────────────────────
# API: run_single_experiment（供 pipeline.py 调用）
# ─────────────────────────────────────────────
# 模型缓存：避免 pipeline 多次加载同一模型
_model_cache = {}

def run_single_experiment(model_type, prompts_dict,
                          device=None, local_paths=None,
                          batch_size=64, gpu=0):
    """
    Run zero-shot evaluation with a custom prompts dict.

    Args:
        model_type: "clip" | "laion" | "flip" | "siglip"
        prompts_dict: {class_name: [prompt_str, ...], ...}
        device: torch.device (auto-detected if None)
        local_paths: {model_name: ckpt_path} for offline loading
        batch_size: inference batch size
        gpu: GPU index

    Returns:
        dict with keys: top1, top3, total, y_true, y_pred, desc
    """
    global _model_cache

    if device is None:
        device = torch.device(
            f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    if local_paths is None:
        local_paths = {
            "siglip": "./ckpts/siglip_vitl16_256/open_clip_pytorch_model.bin",
        }

    # Cache model to avoid reloading
    cache_key = (model_type, str(device))
    if cache_key not in _model_cache:
        log.info(f"Loading model {model_type} (first time) ...")
        _model_cache[cache_key] = load_model(model_type, device, local_paths)
    md = _model_cache[cache_key]

    # Cache dataset
    if not hasattr(run_single_experiment, "_ds_val"):
        run_single_experiment._ds_val = SCB5Dataset("val")
    ds_val = run_single_experiment._ds_val

    # Build text features from prompts dict
    text_feats = build_text_features_from_dict(md["encode_text"], prompts_dict)

    # Evaluate
    out = evaluate_zeroshot_full(ds_val, md, text_feats, device, batch_size)
    out["desc"] = md["desc"]
    return out


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="clip",
                   choices=["clip","laion","flip","siglip","all"])
    p.add_argument("--mode",  default="zeroshot",
                   choices=["zeroshot","fewshot","ablation","all"])
    p.add_argument("--gpu",   type=int, default=0)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--prompt", default="detailed",
                   choices=["simple","medium","detailed"])
    p.add_argument("--shots", nargs="+", type=int, default=[1,2,5,10,20])
    # 本地 checkpoint 路径（可选）
    p.add_argument("--ckpt_laion",  default=None,
                   help="LAION ViT-L/14 本地权重路径 (.bin)")
    p.add_argument("--ckpt_flip",   default=None,
                   help="FLIP ViT-L/14 本地权重路径 (.bin)")
    p.add_argument("--ckpt_siglip", default=None,
                   help="SigLIP 本地权重路径 (.bin)")
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    )
    log.info(f"Device: {device}")

    local_paths = {
        "laion":  args.ckpt_laion,
        "flip":   args.ckpt_flip,
        "siglip": args.ckpt_siglip,
    }

    ds_val   = SCB5Dataset("val")
    ds_train = SCB5Dataset("train") if args.mode in ("fewshot","all") else None

    models = ["clip","laion","flip","siglip"] if args.model == "all" else [args.model]
    all_res = {}

    for mname in models:
        log.info(f"\n{'='*60}\n  Model: {mname}\n{'='*60}")
        try:
            md = load_model(mname, device, local_paths)
        except Exception as e:
            log.error(f"Failed to load {mname}: {e}")
            continue

        res = {"model": mname, "desc": md["desc"]}

        if args.mode in ("zeroshot","all"):
            tf = build_text_features(md["encode_text"], args.prompt)
            res["zeroshot"] = evaluate_zeroshot(
                ds_val, md, tf, device, args.batch_size
            )

        if args.mode in ("ablation","all"):
            res["prompt_ablation"] = evaluate_prompt_ablation(
                ds_val, md, device, args.batch_size
            )

        if args.mode in ("fewshot","all"):
            res["fewshot"] = evaluate_fewshot(
                ds_train, ds_val, md, device, args.shots, args.batch_size
            )

        print_summary(res, mname)
        save_results(res, mname)
        all_res[mname] = res

    if len(all_res) > 1:
        print("\n" + "="*50)
        print("  COMPARISON  (Zero-shot Top-1 Accuracy)")
        print("="*50)
        for m, r in all_res.items():
            if "zeroshot" in r:
                acc = r["zeroshot"]["top1"]
                print(f"  {m:8s}: {acc:6.2f}%  {'█'*int(acc/5)}")
        print("="*50)


if __name__ == "__main__":
    main()
