#!/usr/bin/env python3
"""Stanford40 zero-shot evaluation for R3.12 (independent dataset validation).

Evaluates 5 CLIP-family models on the Stanford40 human action dataset (9532
images, 40 categories) using label-only and action-oriented prompts. This
serves as an independent validation that the prompt-sensitivity phenomenon
observed on SCB5 generalizes to a different human-action dataset.

Usage:
    python stanford40_eval.py --gpu 0
Output:
    stanford40_zero_shot_results.json
"""

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import open_clip
from PIL import Image
from sklearn.metrics import f1_score

IMG_DIR = Path(os.environ.get("STANFORD40_DIR", "~/works/lizhuo/data/JPEGImages")).expanduser()
OUTPUT_DIR = Path("results_r3")

MODELS = {
    "openai": {"arch": "ViT-L-14", "pretrained": "openai"},
    "laion":  {"arch": "ViT-L-14", "pretrained": "laion2b_s32b_b82k"},
    "siglip": {"arch": "ViT-L-16-SigLIP2-256", "pretrained": "webli"},
    "eva02":  {"arch": "EVA02-L-14", "pretrained": "merged2b_s4b_b131k"},
    "dfn":    {"arch": "ViT-L-14", "pretrained": "dfn2b"},
}


def load_categories():
    files = sorted(os.listdir(IMG_DIR))
    jpg_files = [f for f in files if f.endswith(".jpg")]
    labels = {}
    for f in jpg_files:
        m = re.match(r"^(.+?)_(\d+)\.jpg$", f)
        if m:
            action = m.group(1).replace("_", " ")
            labels[f] = action
    categories = sorted(set(labels.values()))
    return jpg_files, labels, categories


def build_prompts(categories):
    label_only = [f"a photo of {c}" for c in categories]
    action = [f"a photo of a person {c}" for c in categories]
    return {"label_only": label_only, "action": action}


@torch.no_grad()
def encode_images(model, preprocess, image_paths, device, batch_size=64):
    all_feats = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        images = []
        for p in batch_paths:
            try:
                img = Image.open(os.path.join(IMG_DIR, p)).convert("RGB")
                images.append(preprocess(img))
            except Exception as e:
                print(f"  Error loading {p}: {e}")
                images.append(torch.zeros(3, 224, 224))
        batch = torch.stack(images).to(device)
        feats = model.encode_image(batch)
        feats = torch.nn.functional.normalize(feats, dim=-1)
        all_feats.append(feats.cpu().numpy())
    return np.concatenate(all_feats, axis=0)


@torch.no_grad()
def encode_text(model, tokenizer, prompts, device):
    tokens = tokenizer(prompts).to(device)
    feats = model.encode_text(tokens)
    feats = torch.nn.functional.normalize(feats, dim=-1)
    return feats.cpu().numpy()


def evaluate(image_feats, text_feats, labels_idx):
    logits = image_feats @ text_feats.T
    preds = logits.argmax(axis=1)
    hit1 = (preds == labels_idx).mean() * 100
    macro_f1 = f1_score(labels_idx, preds, average="macro") * 100
    return hit1, macro_f1, preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default="0")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"Image dir: {IMG_DIR}")

    jpg_files, file_labels, categories = load_categories()
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    label_indices = np.array([cat_to_idx[file_labels[f]] for f in jpg_files])
    prompts = build_prompts(categories)

    print(f"Images: {len(jpg_files)}, Categories: {len(categories)}")
    print(f"Categories: {categories[:5]}...")

    output = {
        "timestamp": int(datetime.now().timestamp()),
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": "R3.12: independent dataset validation on Stanford40",
        "dataset": "Stanford40",
        "num_images": len(jpg_files),
        "num_categories": len(categories),
        "categories": categories,
        "models": [],
    }

    for model_key, cfg in MODELS.items():
        print(f"\n{'='*60}")
        print(f"Model: {model_key} ({cfg['arch']}, {cfg['pretrained']})")
        print(f"{'='*60}")

        try:
            model, _, preprocess = open_clip.create_model_and_transforms(
                cfg["arch"], pretrained=cfg["pretrained"]
            )
            tokenizer = open_clip.get_tokenizer(cfg["arch"])
        except Exception as e:
            print(f"  Failed to load model: {e}")
            continue

        model = model.to(device).eval()

        print("  Encoding images...")
        image_feats = encode_images(model, preprocess, jpg_files, device)
        print(f"  Image features: {image_feats.shape}")

        model_results = {"model": model_key, "arch": cfg["arch"], "pretrained": cfg["pretrained"]}
        for prompt_name, prompt_list in prompts.items():
            text_feats = encode_text(model, tokenizer, prompt_list, device)
            hit1, macro_f1, preds = evaluate(image_feats, text_feats, label_indices)
            print(f"  {prompt_name}: Hit@1={hit1:.2f}%, Macro-F1={macro_f1:.2f}%")
            model_results[f"{prompt_name}_hit1"] = round(hit1, 2)
            model_results[f"{prompt_name}_macro_f1"] = round(macro_f1, 2)

        delta = model_results.get("action_hit1", 0) - model_results.get("label_only_hit1", 0)
        model_results["delta_action_vs_label"] = round(delta, 2)
        output["models"].append(model_results)

        del model
        torch.cuda.empty_cache()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outpath = OUTPUT_DIR / "stanford40_zero_shot_results.json"
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {outpath}")

    print(f"\n{'='*60}")
    print("SUMMARY: Stanford40 Zero-Shot Hit@1")
    print(f"{'='*60}")
    print(f"{'Model':<10} {'Label-only':>12} {'Action':>10} {'Delta':>8}")
    for m in output["models"]:
        print(f"{m['model']:<10} {m.get('label_only_hit1',0):>12.2f} "
              f"{m.get('action_hit1',0):>10.2f} {m['delta_action_vs_label']:>+8.2f}")


if __name__ == "__main__":
    main()
