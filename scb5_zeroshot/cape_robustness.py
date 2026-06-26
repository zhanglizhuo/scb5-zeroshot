#!/usr/bin/env python3
"""
cape_robustness.py — CAPE Prompt-Variation Robustness Experiment
================================================================
Demonstrates that zero-shot CAPE is robust to prompt wording and count.

For each of 5 models × 3 datasets, runs:
  1. CAPE with 1, 2, 3 original prompts (prompt count ablation)
  2. CAPE with alternate wordings (Set B, different human author)
  3. CAPE mixed: 5 random draws of 3 from pool of 6 (original + alt)

Reports mean±std for the mixed trials and Δ between Set A and Set B.

Usage:
  CUDA_VISIBLE_DEVICES=1 python cape_robustness.py --gpu 0
  CUDA_VISIBLE_DEVICES=1,2,3 python cape_robustness.py --gpu 0  # uses first visible
"""

import os, sys, json, time, random, argparse, logging
import numpy as np
from pathlib import Path
from collections import defaultdict

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout),
                              logging.FileHandler(str(LOG_DIR / "cape_robustness.log"))])
log = logging.getLogger(__name__)

# ── Add parent to path for imports from master_benchmark ──
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import open_clip
from run_experiment import load_model, evaluate

# ── Config ──
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

# Dataset definitions
DATASETS = {
    "SCB5_TeacherBehavior": {
        "path": DATA_DIR / "SCB5_TeacherBehavior",
        "classes": ["guide", "answer", "On-stage interaction", "blackboard-writing",
                    "teacher", "stand", "screen", "blackBoard"],
        "multi_label": True,
    },
    "SCB5_HandriseReadWrite": {
        "path": DATA_DIR / "SCB5_HandriseReadWrite",
        "classes": ["hand-raise", "read", "write"],
        "multi_label": False,
    },
    "SCB_BowTurnHead": {
        "path": DATA_DIR / "SCB_BowTurnHead",
        "classes": ["bow-head", "turn-head"],
        "multi_label": False,
    },
}

# ── CAPE Prompts: Set A (Original) ──
CAPE_A = {
    # --- TeacherBehavior ---
    "guide": [
        "a teacher guiding a student one-on-one",
        "a teacher helping a student at their desk",
        "a teacher walking among students and offering guidance",
    ],
    "answer": [
        "a teacher answering a student's question",
        "a teacher responding to a raised hand in class",
        "a student asking a question and the teacher replying",
    ],
    "On-stage interaction": [
        "a teacher interacting with students at the front of the classroom",
        "a teacher engaging with students on the podium",
        "a teacher and students having a discussion in front of the class",
    ],
    "blackboard-writing": [
        "a teacher writing on a blackboard with chalk",
        "a hand writing equations on a chalkboard",
        "a teacher's back while writing on the blackboard",
    ],
    "teacher": [
        "a teacher standing and explaining a concept",
        "a teacher giving a lecture at the podium",
        "a teacher talking to the class while standing",
    ],
    "stand": [
        "a person standing still in a classroom",
        "a teacher standing at the front without interacting",
        "a teacher standing idle near the podium",
    ],
    "screen": [
        "a teacher pointing at a projection screen",
        "a teacher presenting slides on a screen",
        "a screen displaying a presentation in a classroom",
    ],
    "blackBoard": [
        "a teacher pointing at the blackboard",
        "a teacher referring to content on the blackboard",
        "a blackboard with writing visible in a classroom",
    ],
    # --- HandriseReadWrite ---
    "hand-raise": [
        "a student raising their hand in a classroom",
        "a student with arm raised to ask a question",
        "a student raising hand to participate in class",
    ],
    "read": [
        "a student reading a textbook at their desk",
        "a student looking down at a book while reading",
        "a student engaged in reading study materials",
    ],
    "write": [
        "a student writing in a notebook",
        "a student taking notes with a pen",
        "a student writing at their desk in class",
    ],
    # --- BowTurnHead ---
    "bow-head": [
        "a student with their head bowed down",
        "a student looking down at their phone or desk",
        "a student with lowered head not paying attention",
    ],
    "turn-head": [
        "a student turning their head to look sideways",
        "a student looking away from the teacher",
        "a student turning around in their seat",
    ],
}

# ── CAPE Prompts: Set B (Alternate wordings) ──
CAPE_B = {
    # --- TeacherBehavior ---
    "guide": [
        "a teacher leaning over a student's desk to help",
        "individual tutoring in a classroom setting",
        "a teacher assisting a single student with their work",
    ],
    "answer": [
        "a teacher explaining something to a student who asked a question",
        "a dialogue between teacher and student in a classroom",
        "a teacher addressing a student's raised hand",
    ],
    "On-stage interaction": [
        "a lively discussion between a teacher and students at the podium",
        "a teacher calling on students from the front of the room",
        "interactive teaching at the front of a classroom",
    ],
    "blackboard-writing": [
        "chalk writing on a green or black chalkboard",
        "a teacher facing the blackboard writing notes",
        "handwritten text being written on a classroom board",
    ],
    "teacher": [
        "a teacher delivering a lecture to students",
        "a teacher speaking in front of the classroom",
        "a professor explaining a topic while standing",
    ],
    "stand": [
        "a teacher standing motionless in a classroom",
        "a person standing at the front of a classroom doing nothing",
        "an idle teacher standing near a desk",
    ],
    "screen": [
        "a projection screen showing slides in a classroom",
        "a teacher using a projector for a presentation",
        "a digital display showing educational content",
    ],
    "blackBoard": [
        "a chalkboard with written content visible",
        "a teacher gesturing toward a blackboard",
        "educational content displayed on a classroom blackboard",
    ],
    # --- HandriseReadWrite ---
    "hand-raise": [
        "a child with their arm stretched high in a classroom",
        "a student eagerly putting up their hand to answer",
        "students with hands up during class discussion",
    ],
    "read": [
        "a student silently reading at their desk",
        "a student's eyes focused on a textbook page",
        "students reading books in a quiet classroom",
    ],
    "write": [
        "a student's hand holding a pen writing on paper",
        "a student copying notes from the board",
        "a student doing written exercises in class",
    ],
    # --- BowTurnHead ---
    "bow-head": [
        "a student slouching with head down on desk",
        "a student looking at something under their desk",
        "a student with drooped head during class",
    ],
    "turn-head": [
        "a student whose face is turned to the side",
        "a student gazing sideways instead of forward",
        "a student looking at another student during class",
    ],
}


def find_val_dir(ds_path):
    """Find the validation images directory."""
    for candidate in ["images/val", "images/valid", "images/test",
                      "val/images", "valid/images", "test/images"]:
        p = ds_path / candidate
        if p.exists():
            return p
    return None


def find_label_dir(ds_path):
    """Find the validation labels directory."""
    for candidate in ["labels/val", "labels/valid", "labels/test",
                      "val/labels", "valid/labels", "test/labels"]:
        p = ds_path / candidate
        if p.exists():
            return p
    return None


def load_dataset(ds_name):
    """Load images and labels for a dataset."""
    ds_info = DATASETS[ds_name]
    ds_path = ds_info["path"]
    classes = ds_info["classes"]
    multi_label = ds_info["multi_label"]

    img_dir = find_val_dir(ds_path)
    lbl_dir = find_label_dir(ds_path)

    if img_dir is None or lbl_dir is None:
        log.error(f"Cannot find val dir in {ds_path}")
        return None

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
                if cid < len(classes):
                    class_ids.add(cid)

        if class_ids:
            # Primary label = first one found (for single-label eval)
            primary = min(class_ids)
            samples.append({
                "path": str(img_file),
                "labels": class_ids,
                "primary": primary,
            })

    return samples


def load_model(model_key, device):
    """Load CLIP model and return (model, preprocess, tokenizer)."""
    model_name, pretrained = MODEL_SPECS[model_key]
    ckpt = str(CKPT_PATHS[model_key])

    if model_key == "clip":
        import clip as clip_module
        model, preprocess = clip_module.load("ViT-L/14", device=device)
        tokenizer = clip_module.tokenize
        def encode_text(texts):
            with torch.no_grad():
                tok = tokenizer(texts).to(device)
                feat = model.encode_text(tok)
                return F.normalize(feat, dim=-1)
        def encode_image(imgs):
            with torch.no_grad():
                feat = model.encode_image(imgs.to(device))
                return F.normalize(feat, dim=-1)
        return encode_image, encode_text, preprocess
    else:
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=ckpt)
        model = model.to(device).eval()
        if model_key == "siglip":
            tokenizer = open_clip.get_tokenizer(
                f"hf-hub:{Path(ckpt).parent}", context_length=64
            )
        else:
            tokenizer = open_clip.get_tokenizer(model_name)
        def encode_text(texts):
            with torch.no_grad():
                tok = tokenizer(texts).to(device)
                feat = model.encode_text(tok)
                return F.normalize(feat, dim=-1)
        def encode_image(imgs):
            with torch.no_grad():
                feat = model.encode_image(imgs.to(device))
                return F.normalize(feat, dim=-1)
        return encode_image, encode_text, preprocess


def evaluate(samples, encode_image, encode_text, preprocess, prompts_dict,
             classes, multi_label, device, batch_size=64):
    """Zero-shot evaluation. Returns hit@1 (%)."""
    # Build text features
    all_text_feats = []
    for cls in classes:
        prompts = prompts_dict.get(cls, [f"a photo of {cls}"])
        feats = encode_text(prompts)  # [N_prompts, D]
        mean_feat = F.normalize(feats.mean(dim=0, keepdim=True), dim=-1)
        all_text_feats.append(mean_feat)
    text_feats = torch.cat(all_text_feats, dim=0)  # [K, D]

    # Evaluate images
    hits = 0
    total = 0
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i+batch_size]
        imgs = []
        for s in batch:
            img = Image.open(s["path"]).convert("RGB")
            imgs.append(preprocess(img))
        img_tensor = torch.stack(imgs)
        img_feats = encode_image(img_tensor)  # [B, D]
        sims = img_feats @ text_feats.T  # [B, K]
        preds = sims.argmax(dim=-1).cpu().numpy()

        for j, s in enumerate(batch):
            if multi_label:
                if preds[j] in s["labels"]:
                    hits += 1
            else:
                if preds[j] == s["primary"]:
                    hits += 1
            total += 1

    return round(hits / total * 100, 2) if total > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--models", nargs="+",
                        default=["clip", "laion", "siglip", "eva02", "dfn"])
    parser.add_argument("--datasets", nargs="+",
                        default=["SCB5_TeacherBehavior", "SCB5_HandriseReadWrite", "SCB_BowTurnHead"])
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    results = {"timestamp": int(time.time()), "experiments": []}

    # Pre-load datasets
    datasets_cache = {}
    for ds_name in args.datasets:
        log.info(f"Loading dataset {ds_name}...")
        datasets_cache[ds_name] = load_dataset(ds_name)
        if datasets_cache[ds_name]:
            log.info(f"  Loaded {len(datasets_cache[ds_name])} samples")
        else:
            log.warning(f"  FAILED to load {ds_name}")

    for model_key in args.models:
        log.info(f"\n{'='*60}")
        log.info(f"Loading model: {model_key}")
        try:
            encode_image, encode_text, preprocess = load_model(model_key, device)
        except Exception as e:
            log.error(f"Failed to load {model_key}: {e}")
            continue

        for ds_name in args.datasets:
            samples = datasets_cache.get(ds_name)
            if not samples:
                continue

            ds_info = DATASETS[ds_name]
            classes = ds_info["classes"]
            multi_label = ds_info["multi_label"]

            log.info(f"\n  Dataset: {ds_name} ({len(samples)} samples, {len(classes)} classes)")

            # 1) CAPE-A (original): 1, 2, 3 prompts
            for n in [1, 2, 3]:
                prompts_dict = {c: CAPE_A[c][:n] for c in classes}
                hit1 = evaluate(samples, encode_image, encode_text, preprocess,
                                prompts_dict, classes, multi_label, device, args.batch_size)
                log.info(f"    CAPE-A n={n}: {hit1}%")
                results["experiments"].append({
                    "model": model_key, "dataset": ds_name,
                    "variant": f"CAPE_A_n{n}", "hit1": hit1,
                })

            # 2) CAPE-B (alternate): 3 prompts
            prompts_dict = {c: CAPE_B[c] for c in classes}
            hit1 = evaluate(samples, encode_image, encode_text, preprocess,
                            prompts_dict, classes, multi_label, device, args.batch_size)
            log.info(f"    CAPE-B n=3: {hit1}%")
            results["experiments"].append({
                "model": model_key, "dataset": ds_name,
                "variant": "CAPE_B_n3", "hit1": hit1,
            })

            # 3) CAPE-Mixed: 5 random draws of 3 from pool of 6
            mix_hits = []
            for trial in range(5):
                rng = random.Random(trial + 42)
                prompts_dict = {}
                for c in classes:
                    pool = CAPE_A[c] + CAPE_B[c]
                    prompts_dict[c] = rng.sample(pool, 3)
                hit1 = evaluate(samples, encode_image, encode_text, preprocess,
                                prompts_dict, classes, multi_label, device, args.batch_size)
                mix_hits.append(hit1)
                results["experiments"].append({
                    "model": model_key, "dataset": ds_name,
                    "variant": f"CAPE_Mix_t{trial}", "hit1": hit1,
                })
            mean_h = np.mean(mix_hits)
            std_h = np.std(mix_hits)
            log.info(f"    CAPE-Mix (5 trials): {mean_h:.2f} ± {std_h:.2f}%")

        # Free GPU memory
        del encode_image, encode_text, preprocess
        torch.cuda.empty_cache()

    # Save results
    out_dir = Path(__file__).parent / "results_robustness"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"cape_robustness_{int(time.time())}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log.info(f"\nResults saved to {out_path}")

    # Print summary table
    print("\n" + "="*80)
    print("CAPE Robustness Summary")
    print("="*80)
    for ds_name in args.datasets:
        print(f"\n{ds_name}:")
        print(f"{'Model':<18} {'A(1)':<8} {'A(2)':<8} {'A(3)':<8} {'B(3)':<8} {'Mix±σ':<14}")
        print("-" * 62)
        for model_key in args.models:
            exps = [e for e in results["experiments"]
                    if e["model"] == model_key and e["dataset"] == ds_name]
            a1 = next((e["hit1"] for e in exps if e["variant"] == "CAPE_A_n1"), "-")
            a2 = next((e["hit1"] for e in exps if e["variant"] == "CAPE_A_n2"), "-")
            a3 = next((e["hit1"] for e in exps if e["variant"] == "CAPE_A_n3"), "-")
            b3 = next((e["hit1"] for e in exps if e["variant"] == "CAPE_B_n3"), "-")
            mix = [e["hit1"] for e in exps if e["variant"].startswith("CAPE_Mix")]
            if mix:
                mix_str = f"{np.mean(mix):.2f}±{np.std(mix):.2f}"
            else:
                mix_str = "-"
            print(f"{model_key:<18} {a1:<8} {a2:<8} {a3:<8} {b3:<8} {mix_str:<14}")


if __name__ == "__main__":
    main()
