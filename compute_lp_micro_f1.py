#!/usr/bin/env python3
"""
compute_lp_micro_f1.py — Re-run only the TeacherBehavior multi-label linear probe
to fill in the missing Micro-F1 column in tab:linear_probe_ml.

Self-contained: avoids importing cape_robustness (broken import chain).

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 compute_lp_micro_f1.py --models clip
    CUDA_VISIBLE_DEVICES=0 python3 compute_lp_micro_f1.py            # all 5 sequentially

Multi-GPU: launch 5 processes in parallel, one per backbone.
See run_lp_micro_f1_parallel.sh.
"""
import argparse, json, time, sys, logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import open_clip
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)

ROOT = Path("/home/broadsense/works/lizhuo/AutoResearchClaw")
DATA_DIR = ROOT / "datasets_scb"

MODEL_SPECS = {
    "clip":   ("ViT-L-14", "openai"),
    "laion":  ("ViT-L-14", "laion2b_s32b_b82k"),
    "siglip": ("ViT-L-16-SigLIP2-256", "webli"),
    "eva02":  ("EVA02-L-14", "merged2b_s4b_b131k"),
    "dfn":    ("ViT-L-14", "dfn2b"),
}

CKPT_PATHS = {
    "clip":   ROOT / "scb5_zeroshot/ckpts/clip_openai_vitl14/ViT-L-14.pt",
    "laion":  ROOT / "scb5_zeroshot/ckpts/laion_vitl14/open_clip_pytorch_model.bin",
    "siglip": ROOT / "scb5_zeroshot/ckpts/siglip_vitl16_256/open_clip_pytorch_model.bin",
    "eva02":  ROOT / "scb5_zeroshot/ckpts/eva02_clip_vitl14/open_clip_pytorch_model.bin",
    "dfn":    ROOT / "scb5_zeroshot/ckpts/dfn_clip_vitl14/open_clip_pytorch_model.bin",
}

TEACHER_DS = {
    "path": DATA_DIR / "SCB5_TeacherBehavior"
                     / "SCB5_Teacher_Behavior_Stand_BlackBoard_Sreen_20250406-2",
    "classes": ["guide", "answer", "On-stage interaction", "blackboard-writing",
                "teacher", "stand", "screen", "blackBoard"],
}

OUT_DIR = ROOT / "scb5_zeroshot/results_revision"
OUT_DIR.mkdir(exist_ok=True, parents=True)


def load_split(split):
    classes = TEACHER_DS["classes"]
    num_classes = len(classes)
    img_dir = TEACHER_DS["path"] / "images" / split
    lbl_dir = TEACHER_DS["path"] / "labels" / split
    samples = []
    for img_file in sorted(img_dir.glob("*")):
        if img_file.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
            continue
        lbl_file = lbl_dir / (img_file.stem + ".txt")
        if not lbl_file.exists():
            continue
        class_ids = set()
        for line in lbl_file.read_text().splitlines():
            line = line.strip()
            if line:
                cid = int(line.split()[0])
                if cid < num_classes:
                    class_ids.add(cid)
        if class_ids:
            ml_vec = [1 if i in class_ids else 0 for i in range(num_classes)]
            samples.append({"path": str(img_file), "ml_vec": ml_vec})
    return samples


def load_encoder(model_key, device):
    model_name, _ = MODEL_SPECS[model_key]
    ckpt = str(CKPT_PATHS[model_key])

    if model_key == "clip":
        import clip as clip_module
        model, preprocess = clip_module.load("ViT-L/14", device=device)

        def encode_image(imgs):
            with torch.no_grad():
                feat = model.encode_image(imgs.to(device))
                return F.normalize(feat, dim=-1)
        return encode_image, preprocess
    else:
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=ckpt)
        model = model.to(device).eval()

        def encode_image(imgs):
            with torch.no_grad():
                feat = model.encode_image(imgs.to(device))
                return F.normalize(feat, dim=-1)
        return encode_image, preprocess


def extract(samples, encode_image, preprocess, device, batch_size=64,
            desc="Extracting"):
    feats, mls = [], []
    for i in tqdm(range(0, len(samples), batch_size), desc=desc):
        batch = samples[i:i + batch_size]
        imgs = [preprocess(Image.open(s["path"]).convert("RGB")) for s in batch]
        f = encode_image(torch.stack(imgs))
        feats.append(f.cpu().numpy())
        for s in batch:
            mls.append(s["ml_vec"])
    return np.concatenate(feats, axis=0), np.array(mls)


def run_one(model_key, batch_size, train_samples, val_samples, device):
    log.info(f"=== {model_key} ===")
    encode_image, preprocess = load_encoder(model_key, device)

    log.info("  Extracting train features...")
    tr_feats, tr_ml = extract(train_samples, encode_image, preprocess,
                              device, batch_size, desc=f"{model_key}-train")
    log.info("  Extracting val features...")
    va_feats, va_ml = extract(val_samples, encode_image, preprocess,
                              device, batch_size, desc=f"{model_key}-val")

    scaler = StandardScaler()
    tr_s = scaler.fit_transform(tr_feats)
    va_s = scaler.transform(va_feats)

    num_classes = tr_ml.shape[1]
    ml_preds = np.zeros_like(va_ml)
    for ci in range(num_classes):
        clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs",
                                 random_state=42)
        clf.fit(tr_s, tr_ml[:, ci])
        ml_preds[:, ci] = clf.predict(va_s)

    sample_f1 = f1_score(va_ml, ml_preds, average="samples", zero_division=0) * 100
    macro_f1 = f1_score(va_ml, ml_preds, average="macro", zero_division=0) * 100
    micro_f1 = f1_score(va_ml, ml_preds, average="micro", zero_division=0) * 100
    log.info(f"  {model_key}: Sample-F1={sample_f1:.2f}  Macro-F1={macro_f1:.2f}  "
             f"Micro-F1={micro_f1:.2f}")
    return {
        "ml_sample_f1": round(sample_f1, 2),
        "ml_macro_f1": round(macro_f1, 2),
        "ml_micro_f1": round(micro_f1, 2),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+",
                   default=list(MODEL_SPECS.keys()),
                   choices=list(MODEL_SPECS.keys()))
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--tag", type=str, default="",
                   help="Optional tag appended to output filename "
                        "(helps when multiple GPU processes write in parallel).")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}  Visible: "
             f"{torch.cuda.device_count() if device=='cuda' else 0}")

    train_samples = load_split("train")
    val_samples = load_split("val")
    log.info(f"Teacher: train={len(train_samples)}  val={len(val_samples)}")

    out = {}
    for mk in args.models:
        out[mk] = run_one(mk, args.batch_size, train_samples, val_samples, device)
        torch.cuda.empty_cache()

    ts = int(time.time())
    suffix = f"_{args.tag}" if args.tag else ""
    out_path = OUT_DIR / f"lp_micro_f1_teacher_{ts}{suffix}.json"
    out_path.write_text(json.dumps(out, indent=2))
    log.info(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
