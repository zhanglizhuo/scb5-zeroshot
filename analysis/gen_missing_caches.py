#!/usr/bin/env python3
"""Generate missing feature caches for LAION and EVA02 (local YOLO format).

IMPORTANT: CAPE prompts are loaded from the canonical source
`analysis/prompts/setAB_examples.json` (set_A_original). Do NOT inline
prompt text here: the earlier inline copy (commit 5ca5310) diverged from
the paper's CAPE Set A and produced invalid logits_cape caches.
"""
import os, sys, json, logging, argparse
import numpy as np
from pathlib import Path

import torch
import open_clip
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

MODELS = {
    "laion": {"arch": "ViT-L-14", "pretrained": "laion2b_s32b_b82k"},
    "eva02": {"arch": "EVA02-L-14", "pretrained": "merged2b_s4b_b131k"},
}

DATA_DIR = Path("/home/broadsense/works/lizhuo/AutoResearchClaw/datasets_scb")

PROMPTS_FILE = Path(__file__).resolve().parents[1] / "analysis" / "prompts" / "setAB_examples.json"

SUBSETS = [{"name":"teacher_behavior","local_dir":str(DATA_DIR/"SCB5_TeacherBehavior"/"SCB5_Teacher_Behavior_Stand_BlackBoard_Sreen_20250406-2"),"classes":8,"classes_key":"TeacherBehavior"},{"name":"handrise_readwrite","local_dir":str(DATA_DIR/"SCB5_HandriseReadWrite"),"classes":3,"classes_key":"HandriseReadWrite"},{"name":"bow_turnhead","local_dir":str(DATA_DIR/"SCB_BowTurnHead"),"classes":2,"classes_key":"BowTurnHead"}]

CACHE_DIR = Path(os.environ.get("CACHE_DIR", "/tmp/caches"))


def load_cape_prompts(subset_name, classes_key):
    with open(PROMPTS_FILE, encoding="utf-8") as f:
        data = json.load(f)
    cape_a = data["set_A_original"][classes_key]
    class_names = cape_a.keys()
    return [cape_a[c] for c in class_names]

def load_samples(local_dir, split="val"):
    img_dir = lbl_dir = None
    d = Path(local_dir)
    for p in [d, d/"SCB5_Teacher_Behavior_Stand_BlackBoard_Sreen_20250406-2"]:
        if not p.exists(): continue
        for s in [split, "validation"]:
            im = p / "images" / s
            lb = p / "labels" / s
            if im.exists():
                img_dir, lbl_dir = im, lb
                break
        if img_dir: break
    if not img_dir and d.exists():
        for child in sorted(d.iterdir()):
            if child.is_dir():
                for s in [split, "validation"]:
                    im = child / "images" / s
                    lb = child / "labels" / s
                    if im.exists():
                        img_dir, lbl_dir = im, lb
                        break
            if img_dir: break
    if not img_dir:
        raise FileNotFoundError(f"No images under {local_dir}")
    samples = []
    for img_path in sorted(img_dir.glob("*.jpg")):
        lbl_path = lbl_dir / img_path.with_suffix(".txt").name
        labels = []
        if lbl_path.exists():
            with open(lbl_path) as f:
                for line in f:
                    labels.append(int(line.strip().split()[0]))
        samples.append({"image_path": str(img_path), "labels": list(set(labels))})
    log.info(f"  {len(samples)} samples from {img_dir}")
    return samples

def encode_text(model, tokenizer, prompts, device):
    texts = tokenizer(prompts).to(device)
    with torch.no_grad(), torch.cuda.amp.autocast():
        emb = model.encode_text(texts).float()
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["laion", "eva02"], required=True)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    device = f"cuda:{args.gpu}"
    cfg = MODELS[args.model]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    log.info(f"Loading {args.model} ({cfg['arch']}) on {device}")
    model, _, preprocess = open_clip.create_model_and_transforms(cfg["arch"], pretrained=cfg["pretrained"], device=device)
    tokenizer = open_clip.get_tokenizer(cfg["arch"])
    model.eval()
    for s in SUBSETS:
        name = s["name"]; nc = s["classes"]; out = CACHE_DIR / f"{args.model}_{name}_validation.npz"
        if out.exists(): log.info(f"  Skip {name}"); continue
        log.info(f"  {name} ({nc} classes)")
        samples = load_samples(s["local_dir"])
        feats, labs = [], []
        batch_imgs, batch_labs = [], []
        for idx, smp in enumerate(tqdm(samples, desc=f"  {name}")):
            img = Image.open(smp["image_path"]).convert("RGB")
            batch_imgs.append(preprocess(img).unsqueeze(0))
            lab = np.zeros(nc, dtype=np.float32)
            for c in smp["labels"]:
                if c < nc: lab[c] = 1.0
            batch_labs.append(lab)
            if len(batch_imgs) == 64 or idx == len(samples)-1:
                imgs = torch.cat(batch_imgs, dim=0).to(device)
                with torch.no_grad(), torch.cuda.amp.autocast():
                    f = model.encode_image(imgs).float()
                    f = f / f.norm(dim=-1, keepdim=True)
                feats.append(f.cpu().numpy())
                labs.append(np.array(batch_labs))
                batch_imgs, batch_labs = [], []
        feats = np.concatenate(feats, axis=0)
        labs = np.concatenate(labs, axis=0)
        log.info(f"  feats {feats.shape} labs {labs.shape}")
        prompts = load_cape_prompts(name, s["classes_key"])
        tembs = np.concatenate([encode_text(model, tokenizer, cp, device).mean(0, keepdims=True) for cp in prompts], axis=0)
        tembs = tembs / np.linalg.norm(tembs, axis=1, keepdims=True)
        logits = feats @ tembs.T
        np.savez_compressed(out, image_features=feats, labels=labs, logits_cape=logits)
        log.info(f"  saved {out} ({logits.nbytes/1e6:.1f} MB)")
    log.info("Done")

if __name__ == "__main__":
    main()
