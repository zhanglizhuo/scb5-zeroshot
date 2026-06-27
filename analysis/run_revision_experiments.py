#!/usr/bin/env python3
"""
run_revision_experiments.py — Revision experiments for SCB5 zero-shot paper
===========================================================================
Addresses four major reviewer concerns:

  R1: Linear Probe supervised baseline (frozen CLIP features + LogisticRegression)
  R2: Bootstrap confidence intervals for zero-shot results
  R3: Blind prompt evaluation (Set C, simulating third-party / LLM-generated prompts)
  R4: Proper multi-label evaluation metrics

Usage:
  # Run all experiments on GPU 0
  python3 run_revision_experiments.py --gpu 0

  # Run specific experiment
  python3 run_revision_experiments.py --gpu 0 --exp r1

  # Multi-GPU: run different models on different GPUs (in separate terminals)
  CUDA_VISIBLE_DEVICES=0 python3 run_revision_experiments.py --gpu 0 --models clip laion --exp r1
  CUDA_VISIBLE_DEVICES=1 python3 run_revision_experiments.py --gpu 0 --models siglip eva02 --exp r1
  CUDA_VISIBLE_DEVICES=2 python3 run_revision_experiments.py --gpu 0 --models dfn --exp r1
"""

import os, sys, json, time, random, argparse, logging
import numpy as np
from pathlib import Path
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# Ensure logs/ exists and write file log for revision runs
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
# add file handler
if not any(isinstance(h, logging.FileHandler) for h in log.handlers):
    fh = logging.FileHandler(str(LOG_DIR / "run_revision_experiments.log"))
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    log.addHandler(fh)

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import open_clip
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
)

# =====================================================================
# Config: paths, models, datasets (same as cape_robustness.py)
# =====================================================================
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "datasets_scb"

MODEL_SPECS = {
    "clip":   ("ViT-L-14", "openai"),
    "laion":  ("ViT-L-14", "laion2b_s32b_b82k"),
    "siglip": ("ViT-L-16-SigLIP2-256", "webli"),
    "eva02":  ("EVA02-L-14", "merged2b_s4b_b131k"),
    "dfn":    ("ViT-L-14", "dfn2b"),
}

CKPT_PATHS = {
    "clip":   ROOT / "ckpts" / "clip_openai_vitl14" / "ViT-L-14.pt",
    "laion":  ROOT / "ckpts" / "laion_vitl14" / "open_clip_pytorch_model.bin",
    "siglip": ROOT / "ckpts" / "siglip_vitl16_256" / "open_clip_pytorch_model.bin",
    "eva02":  ROOT / "ckpts" / "eva02_clip_vitl14" / "open_clip_pytorch_model.bin",
    "dfn":    ROOT / "ckpts" / "dfn_clip_vitl14" / "open_clip_pytorch_model.bin",
}

DATASETS = {
    "SCB5_TeacherBehavior": {
        "path": ROOT / "SCB5_Teacher_Behavior_Stand_BlackBoard_Sreen_20250406"
                     / "SCB5_Teacher_Behavior_Stand_BlackBoard_Sreen_20250406-2",
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

# =====================================================================
# R3: Blind Prompts — Set C
# Written WITHOUT looking at actual images, purely from class name semantics.
# Simulates what a third-party or LLM would generate given only class names.
# =====================================================================
CAPE_C = {
    # --- TeacherBehavior: prompts based ONLY on class name, no dataset knowledge ---
    "guide": [
        "a person providing guidance in an indoor setting",
        "someone showing directions or instructions to another person",
        "a mentor giving advice to a learner",
    ],
    "answer": [
        "a person giving an answer or reply",
        "someone responding verbally to a question",
        "a conversation where one person explains something",
    ],
    "On-stage interaction": [
        "people interacting on a stage or platform",
        "a speaker engaging with an audience from a raised platform",
        "a person standing on stage communicating with others",
    ],
    "blackboard-writing": [
        "someone writing text on a large dark board",
        "handwriting being produced on a wall-mounted board",
        "a person using chalk to write on a board surface",
    ],
    "teacher": [
        "a person in the role of a teacher or instructor",
        "an educator standing in front of learners",
        "a professional conducting a lesson",
    ],
    "stand": [
        "a person standing upright in a room",
        "someone in a standing posture without movement",
        "a figure standing still indoors",
    ],
    "screen": [
        "an electronic display or projection screen",
        "a monitor or screen showing digital content",
        "a flat display surface mounted in a room",
    ],
    "blackBoard": [
        "a dark-colored board mounted on a wall",
        "a chalkboard visible in a room",
        "a traditional writing board with content on it",
    ],
    # --- HandriseReadWrite ---
    "hand-raise": [
        "a person raising one hand above their head",
        "someone with an arm lifted upward",
        "a person with their hand raised in the air",
    ],
    "read": [
        "a person reading printed material",
        "someone looking down at a book or document",
        "a person focused on reading text",
    ],
    "write": [
        "a person writing with a pen or pencil",
        "someone moving a writing instrument on paper",
        "a person engaged in handwriting",
    ],
    # --- BowTurnHead ---
    "bow-head": [
        "a person with their head tilted downward",
        "someone bowing their head forward",
        "a person looking down with lowered head",
    ],
    "turn-head": [
        "a person with their head turned to one side",
        "someone looking sideways",
        "a person whose head faces a different direction than their body",
    ],
}

# Import CAPE Set A and Set B from cape_robustness for comparison
from cape_robustness import CAPE_A, CAPE_B


# =====================================================================
# Dataset loading (supports train + val)
# =====================================================================
def find_split_dirs(ds_path, split):
    """Find image and label dirs for a given split."""
    img_dir = ds_path / "images" / split
    lbl_dir = ds_path / "labels" / split
    if img_dir.exists() and lbl_dir.exists():
        return img_dir, lbl_dir
    return None, None


def load_dataset(ds_name, split="val"):
    """Load dataset samples for a given split."""
    ds_info = DATASETS[ds_name]
    ds_path = ds_info["path"]
    classes = ds_info["classes"]
    num_classes = len(classes)

    img_dir, lbl_dir = find_split_dirs(ds_path, split)
    if img_dir is None:
        log.error(f"Cannot find {split} dir in {ds_path}")
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
                if cid < num_classes:
                    class_ids.add(cid)

        if class_ids:
            primary = min(class_ids)
            # Multi-label binary vector
            ml_vec = [1 if i in class_ids else 0 for i in range(num_classes)]
            samples.append({
                "path": str(img_file),
                "labels": class_ids,
                "primary": primary,
                "ml_vec": ml_vec,
            })

    return samples


# =====================================================================
# Model loading (same as cape_robustness.py)
# =====================================================================
def load_model(model_key, device):
    """Load CLIP model. Returns (encode_image, encode_text, preprocess)."""
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
                f"hf-hub:{Path(ckpt).parent}", context_length=64)
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


# =====================================================================
# Feature extraction (for linear probe)
# =====================================================================
def extract_features(samples, encode_image, preprocess, device, batch_size=64):
    """Extract image features for all samples. Returns (feats [N,D], labels)."""
    all_feats = []
    primaries = []
    ml_vecs = []

    for i in tqdm(range(0, len(samples), batch_size), desc="Extracting"):
        batch = samples[i:i + batch_size]
        imgs = []
        for s in batch:
            img = Image.open(s["path"]).convert("RGB")
            imgs.append(preprocess(img))
        img_tensor = torch.stack(imgs)
        feats = encode_image(img_tensor)  # [B, D]
        all_feats.append(feats.cpu().numpy())
        for s in batch:
            primaries.append(s["primary"])
            ml_vecs.append(s["ml_vec"])

    return np.concatenate(all_feats, axis=0), np.array(primaries), np.array(ml_vecs)


# =====================================================================
# Zero-shot evaluation (returns per-sample predictions for bootstrap)
# =====================================================================
def zeroshot_predict(samples, encode_image, encode_text, preprocess,
                     prompts_dict, classes, device, batch_size=64):
    """Zero-shot prediction. Returns per-sample (pred, topk3, labels_set)."""
    # Build text features
    all_text_feats = []
    for cls in classes:
        prompts = prompts_dict.get(cls, [f"a photo of {cls}"])
        feats = encode_text(prompts)
        mean_feat = F.normalize(feats.mean(dim=0, keepdim=True), dim=-1)
        all_text_feats.append(mean_feat)
    text_feats = torch.cat(all_text_feats, dim=0)  # [K, D]

    predictions = []
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i + batch_size]
        imgs = []
        for s in batch:
            img = Image.open(s["path"]).convert("RGB")
            imgs.append(preprocess(img))
        img_tensor = torch.stack(imgs)
        img_feats = encode_image(img_tensor)
        sims = img_feats @ text_feats.T  # [B, K]
        top1 = sims.argmax(dim=-1).cpu().numpy()
        top3 = sims.topk(min(3, sims.shape[1]), dim=-1).indices.cpu().numpy()

        for j, s in enumerate(batch):
            predictions.append({
                "pred": int(top1[j]),
                "top3": [int(x) for x in top3[j]],
                "labels": s["labels"],
                "primary": s["primary"],
                "ml_vec": s["ml_vec"],
            })

    return predictions


# =====================================================================
# R1: Linear Probe
# =====================================================================
def run_linear_probe(model_key, device, batch_size=64):
    """Linear probe: frozen CLIP features + LogisticRegression."""
    log.info(f"[R1] Linear Probe — {model_key}")
    encode_image, encode_text, preprocess = load_model(model_key, device)
    results = []

    for ds_name, ds_info in DATASETS.items():
        classes = ds_info["classes"]
        multi_label = ds_info["multi_label"]
        num_classes = len(classes)

        log.info(f"  Dataset: {ds_name}")

        # Load train and val
        train_samples = load_dataset(ds_name, "train")
        val_samples = load_dataset(ds_name, "val")
        if not train_samples or not val_samples:
            log.warning(f"  Skipping {ds_name}: missing data")
            continue

        log.info(f"  Train: {len(train_samples)}, Val: {len(val_samples)}")

        # Extract features
        log.info(f"  Extracting train features...")
        train_feats, train_labels, train_ml = extract_features(
            train_samples, encode_image, preprocess, device, batch_size)
        log.info(f"  Extracting val features...")
        val_feats, val_labels, val_ml = extract_features(
            val_samples, encode_image, preprocess, device, batch_size)

        # Normalize features
        scaler = StandardScaler()
        train_feats_s = scaler.fit_transform(train_feats)
        val_feats_s = scaler.transform(val_feats)

        # --- Single-label linear probe (primary label) ---
        log.info(f"  Training single-label LogisticRegression...")
        clf_sl = LogisticRegression(
            max_iter=1000, C=1.0, solver="lbfgs",
            multi_class="multinomial", random_state=42,
        )
        clf_sl.fit(train_feats_s, train_labels)
        sl_preds = clf_sl.predict(val_feats_s)
        sl_acc = accuracy_score(val_labels, sl_preds) * 100
        sl_macro_f1 = f1_score(val_labels, sl_preds, average="macro", zero_division=0) * 100
        sl_weighted_f1 = f1_score(val_labels, sl_preds, average="weighted", zero_division=0) * 100

        log.info(f"    SL Acc={sl_acc:.2f}%, Macro-F1={sl_macro_f1:.2f}%, Weighted-F1={sl_weighted_f1:.2f}%")

        result = {
            "model": model_key,
            "dataset": ds_name,
            "n_train": len(train_samples),
            "n_val": len(val_samples),
            "sl_accuracy": round(sl_acc, 2),
            "sl_macro_f1": round(sl_macro_f1, 2),
            "sl_weighted_f1": round(sl_weighted_f1, 2),
        }

        # --- Multi-label linear probe (one-vs-rest) for TeacherBehavior ---
        if multi_label:
            log.info(f"  Training multi-label (one-vs-rest) classifiers...")
            ml_preds = np.zeros_like(val_ml)
            per_class_f1 = {}

            for ci in range(num_classes):
                clf_ci = LogisticRegression(
                    max_iter=1000, C=1.0, solver="lbfgs", random_state=42)
                clf_ci.fit(train_feats_s, train_ml[:, ci])
                ml_preds[:, ci] = clf_ci.predict(val_feats_s)
                f1_ci = f1_score(val_ml[:, ci], ml_preds[:, ci], zero_division=0) * 100
                per_class_f1[classes[ci]] = round(f1_ci, 2)

            ml_sample_f1 = f1_score(val_ml, ml_preds, average="samples", zero_division=0) * 100
            ml_macro_f1 = f1_score(val_ml, ml_preds, average="macro", zero_division=0) * 100
            ml_micro_f1 = f1_score(val_ml, ml_preds, average="micro", zero_division=0) * 100

            # Hit@1: does the highest-probability class match any ground truth?
            # Use single-label classifier's probabilities
            sl_probs = clf_sl.predict_proba(val_feats_s)  # [N, K]
            top1_preds = sl_probs.argmax(axis=1)
            hit1 = sum(1 for i in range(len(val_samples))
                       if top1_preds[i] in val_samples[i]["labels"]) / len(val_samples) * 100

            log.info(f"    ML Sample-F1={ml_sample_f1:.2f}%, ML Macro-F1={ml_macro_f1:.2f}%, ML Micro-F1={ml_micro_f1:.2f}%")
            log.info(f"    Hit@1 (supervised)={hit1:.2f}%")
            log.info(f"    Per-class F1: {per_class_f1}")

            result["ml_sample_f1"] = round(ml_sample_f1, 2)
            result["ml_macro_f1"] = round(ml_macro_f1, 2)
            result["ml_micro_f1"] = round(ml_micro_f1, 2)
            result["hit1_supervised"] = round(hit1, 2)
            result["per_class_f1"] = per_class_f1
        else:
            # For single-label datasets, Hit@1 = accuracy
            result["hit1_supervised"] = round(sl_acc, 2)

        results.append(result)

    return results


# =====================================================================
# R2: Bootstrap Confidence Intervals
# =====================================================================
def compute_bootstrap_ci(predictions, multi_label, n_bootstrap=1000, seed=42):
    """Compute 95% bootstrap CI for Hit@1 and macro-F1."""
    rng = np.random.RandomState(seed)
    n = len(predictions)

    hit1_samples = []
    f1_samples = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        boot = [predictions[i] for i in idx]

        # Hit@1
        if multi_label:
            hits = sum(1 for p in boot if p["pred"] in p["labels"])
        else:
            hits = sum(1 for p in boot if p["pred"] == p["primary"])
        hit1_samples.append(hits / n * 100)

        # Macro-F1 (single-label, using primary)
        y_true = [p["primary"] for p in boot]
        y_pred = [p["pred"] for p in boot]
        mf1 = f1_score(y_true, y_pred, average="macro", zero_division=0) * 100
        f1_samples.append(mf1)

    hit1_arr = np.array(hit1_samples)
    f1_arr = np.array(f1_samples)

    return {
        "hit1_mean": round(np.mean(hit1_arr), 2),
        "hit1_ci_lo": round(np.percentile(hit1_arr, 2.5), 2),
        "hit1_ci_hi": round(np.percentile(hit1_arr, 97.5), 2),
        "macro_f1_mean": round(np.mean(f1_arr), 2),
        "macro_f1_ci_lo": round(np.percentile(f1_arr, 2.5), 2),
        "macro_f1_ci_hi": round(np.percentile(f1_arr, 97.5), 2),
    }


def run_bootstrap(model_key, device, batch_size=64):
    """Bootstrap CI for zero-shot CAPE predictions."""
    log.info(f"[R2] Bootstrap CI — {model_key}")
    encode_image, encode_text, preprocess = load_model(model_key, device)
    results = []

    for ds_name, ds_info in DATASETS.items():
        classes = ds_info["classes"]
        multi_label = ds_info["multi_label"]

        val_samples = load_dataset(ds_name, "val")
        if not val_samples:
            continue

        log.info(f"  Dataset: {ds_name} ({len(val_samples)} samples)")

        # Get CAPE-A predictions (the main CAPE results from the paper)
        prompts_dict = {c: CAPE_A[c] for c in classes}
        predictions = zeroshot_predict(
            val_samples, encode_image, encode_text, preprocess,
            prompts_dict, classes, device, batch_size)

        ci = compute_bootstrap_ci(predictions, multi_label)
        log.info(f"    Hit@1: {ci['hit1_mean']:.2f}% [{ci['hit1_ci_lo']:.2f}, {ci['hit1_ci_hi']:.2f}]")
        log.info(f"    Macro-F1: {ci['macro_f1_mean']:.2f}% [{ci['macro_f1_ci_lo']:.2f}, {ci['macro_f1_ci_hi']:.2f}]")

        results.append({
            "model": model_key,
            "dataset": ds_name,
            "n_val": len(val_samples),
            "prompt_set": "CAPE_A",
            **ci,
        })

    return results


# =====================================================================
# R3: Blind Prompt Evaluation (Set C)
# =====================================================================
def run_blind_prompts(model_key, device, batch_size=64):
    """Evaluate Set C (blind) prompts and compare with Set A (CAPE) and Set B (alt)."""
    log.info(f"[R3] Blind Prompt Evaluation — {model_key}")
    encode_image, encode_text, preprocess = load_model(model_key, device)
    results = []

    for ds_name, ds_info in DATASETS.items():
        classes = ds_info["classes"]
        multi_label = ds_info["multi_label"]

        val_samples = load_dataset(ds_name, "val")
        if not val_samples:
            continue

        log.info(f"  Dataset: {ds_name} ({len(val_samples)} samples)")

        for set_name, prompts_source in [("A", CAPE_A), ("B", CAPE_B), ("C", CAPE_C)]:
            prompts_dict = {c: prompts_source[c] for c in classes}
            predictions = zeroshot_predict(
                val_samples, encode_image, encode_text, preprocess,
                prompts_dict, classes, device, batch_size)

            # Hit@1
            if multi_label:
                hit1 = sum(1 for p in predictions if p["pred"] in p["labels"]) / len(predictions) * 100
            else:
                hit1 = sum(1 for p in predictions if p["pred"] == p["primary"]) / len(predictions) * 100

            # Hit@3
            if multi_label:
                hit3 = sum(1 for p in predictions if any(t in p["labels"] for t in p["top3"])) / len(predictions) * 100
            else:
                hit3 = sum(1 for p in predictions if p["primary"] in p["top3"]) / len(predictions) * 100

            # Macro-F1 (single-label)
            y_true = [p["primary"] for p in predictions]
            y_pred = [p["pred"] for p in predictions]
            mf1 = f1_score(y_true, y_pred, average="macro", zero_division=0) * 100

            log.info(f"    Set {set_name}: Hit@1={hit1:.2f}%, Hit@3={hit3:.2f}%, MF1={mf1:.2f}%")

            results.append({
                "model": model_key,
                "dataset": ds_name,
                "prompt_set": f"Set_{set_name}",
                "hit1": round(hit1, 2),
                "hit3": round(hit3, 2),
                "macro_f1": round(mf1, 2),
            })

    return results


# =====================================================================
# R4: Multi-label Evaluation
# =====================================================================
def run_multilabel_eval(model_key, device, batch_size=64):
    """Proper multi-label evaluation for TeacherBehavior."""
    log.info(f"[R4] Multi-label Evaluation — {model_key}")
    encode_image, encode_text, preprocess = load_model(model_key, device)

    ds_name = "SCB5_TeacherBehavior"
    ds_info = DATASETS[ds_name]
    classes = ds_info["classes"]
    num_classes = len(classes)

    val_samples = load_dataset(ds_name, "val")
    if not val_samples:
        return []

    log.info(f"  {ds_name}: {len(val_samples)} samples, {num_classes} classes")

    # Build text features for CAPE
    prompts_dict = {c: CAPE_A[c] for c in classes}
    all_text_feats = []
    for cls in classes:
        feats = encode_text(prompts_dict[cls])
        mean_feat = F.normalize(feats.mean(dim=0, keepdim=True), dim=-1)
        all_text_feats.append(mean_feat)
    text_feats = torch.cat(all_text_feats, dim=0)  # [K, D]

    # Get similarity scores for all images
    all_sims = []
    gt_ml = []
    for i in range(0, len(val_samples), batch_size):
        batch = val_samples[i:i + batch_size]
        imgs = []
        for s in batch:
            img = Image.open(s["path"]).convert("RGB")
            imgs.append(preprocess(img))
        img_tensor = torch.stack(imgs)
        img_feats = encode_image(img_tensor)
        sims = img_feats @ text_feats.T
        all_sims.append(sims.cpu().numpy())
        for s in batch:
            gt_ml.append(s["ml_vec"])

    all_sims = np.concatenate(all_sims, axis=0)  # [N, K]
    gt_ml = np.array(gt_ml)                       # [N, K]

    results = []

    # Evaluate at different thresholds
    for threshold in [0.15, 0.20, 0.25, 0.30]:
        pred_ml = (all_sims >= threshold).astype(int)

        # Ensure at least one prediction per sample
        for i in range(len(pred_ml)):
            if pred_ml[i].sum() == 0:
                pred_ml[i, all_sims[i].argmax()] = 1

        sample_f1 = f1_score(gt_ml, pred_ml, average="samples", zero_division=0) * 100
        macro_f1 = f1_score(gt_ml, pred_ml, average="macro", zero_division=0) * 100
        micro_f1 = f1_score(gt_ml, pred_ml, average="micro", zero_division=0) * 100

        # Per-class F1
        per_class = {}
        for ci, cls in enumerate(classes):
            f1_ci = f1_score(gt_ml[:, ci], pred_ml[:, ci], zero_division=0) * 100
            per_class[cls] = round(f1_ci, 2)

        log.info(f"    τ={threshold:.2f}: Sample-F1={sample_f1:.2f}%, "
                 f"Macro-F1={macro_f1:.2f}%, Micro-F1={micro_f1:.2f}%")

        results.append({
            "model": model_key,
            "dataset": ds_name,
            "threshold": threshold,
            "sample_f1": round(sample_f1, 2),
            "macro_f1": round(macro_f1, 2),
            "micro_f1": round(micro_f1, 2),
            "per_class_f1": per_class,
        })

    # Also compute "top-k as multi-label": predict top-2, top-3
    for topk in [2, 3]:
        pred_ml_topk = np.zeros_like(gt_ml)
        topk_indices = np.argsort(-all_sims, axis=1)[:, :topk]
        for i in range(len(pred_ml_topk)):
            pred_ml_topk[i, topk_indices[i]] = 1

        sample_f1 = f1_score(gt_ml, pred_ml_topk, average="samples", zero_division=0) * 100
        macro_f1 = f1_score(gt_ml, pred_ml_topk, average="macro", zero_division=0) * 100
        micro_f1 = f1_score(gt_ml, pred_ml_topk, average="micro", zero_division=0) * 100

        per_class = {}
        for ci, cls in enumerate(classes):
            f1_ci = f1_score(gt_ml[:, ci], pred_ml_topk[:, ci], zero_division=0) * 100
            per_class[cls] = round(f1_ci, 2)

        log.info(f"    top-{topk}: Sample-F1={sample_f1:.2f}%, "
                 f"Macro-F1={macro_f1:.2f}%, Micro-F1={micro_f1:.2f}%")

        results.append({
            "model": model_key,
            "dataset": ds_name,
            "threshold": f"top-{topk}",
            "sample_f1": round(sample_f1, 2),
            "macro_f1": round(macro_f1, 2),
            "micro_f1": round(micro_f1, 2),
            "per_class_f1": per_class,
        })

    return results


# =====================================================================
# Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="SCB5 Revision Experiments")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--models", nargs="+",
                        default=["clip", "laion", "siglip", "eva02", "dfn"])
    parser.add_argument("--exp", nargs="+", default=["r1", "r2", "r3", "r4"],
                        help="Which experiments to run: r1 r2 r3 r4")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
    log.info(f"Models: {args.models}")
    log.info(f"Experiments: {args.exp}")

    out_dir = Path(__file__).resolve().parent.parent / "results" / "revision"
    out_dir.mkdir(exist_ok=True)

    all_results = {
        "timestamp": int(time.time()),
        "models": args.models,
        "device": str(device),
    }

    for model_key in args.models:
        log.info(f"\n{'=' * 60}")
        log.info(f"Model: {model_key}")
        log.info(f"{'=' * 60}")

        if "r1" in args.exp:
            try:
                r1 = run_linear_probe(model_key, device, args.batch_size)
                all_results.setdefault("r1_linear_probe", []).extend(r1)
            except Exception as e:
                log.error(f"R1 failed for {model_key}: {e}", exc_info=True)

        if "r2" in args.exp:
            try:
                r2 = run_bootstrap(model_key, device, args.batch_size)
                all_results.setdefault("r2_bootstrap_ci", []).extend(r2)
            except Exception as e:
                log.error(f"R2 failed for {model_key}: {e}", exc_info=True)

        if "r3" in args.exp:
            try:
                r3 = run_blind_prompts(model_key, device, args.batch_size)
                all_results.setdefault("r3_blind_prompts", []).extend(r3)
            except Exception as e:
                log.error(f"R3 failed for {model_key}: {e}", exc_info=True)

        if "r4" in args.exp:
            try:
                r4 = run_multilabel_eval(model_key, device, args.batch_size)
                all_results.setdefault("r4_multilabel", []).extend(r4)
            except Exception as e:
                log.error(f"R4 failed for {model_key}: {e}", exc_info=True)

        # Free GPU memory between models
        torch.cuda.empty_cache()

    # Save results
    ts = int(time.time())
    out_path = out_dir / f"revision_results_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    log.info(f"\nResults saved to {out_path}")

    # ── Print summary tables ──
    print_summary(all_results)


def print_summary(results):
    """Print formatted summary tables."""
    print("\n" + "=" * 80)
    print("REVISION EXPERIMENT RESULTS SUMMARY")
    print("=" * 80)

    # R1: Linear Probe
    if "r1_linear_probe" in results:
        print("\n── R1: Linear Probe (Supervised Baseline) ──")
        print(f"{'Model':<10} {'Dataset':<25} {'SL-Acc%':<10} {'SL-MF1%':<10} {'Hit@1%':<10} {'ML-MF1%':<10}")
        print("-" * 75)
        for r in results["r1_linear_probe"]:
            ml_mf1 = f"{r.get('ml_macro_f1', '-')}"
            hit1_s = f"{r.get('hit1_supervised', '-')}"
            print(f"{r['model']:<10} {r['dataset']:<25} {r['sl_accuracy']:<10} "
                  f"{r['sl_macro_f1']:<10} {hit1_s:<10} {ml_mf1:<10}")

    # R2: Bootstrap CI
    if "r2_bootstrap_ci" in results:
        print("\n── R2: Bootstrap 95% Confidence Intervals (CAPE Zero-Shot) ──")
        print(f"{'Model':<10} {'Dataset':<25} {'Hit@1%':<20} {'Macro-F1%':<20}")
        print("-" * 75)
        for r in results["r2_bootstrap_ci"]:
            h1 = f"{r['hit1_mean']:.2f} [{r['hit1_ci_lo']:.2f},{r['hit1_ci_hi']:.2f}]"
            mf = f"{r['macro_f1_mean']:.2f} [{r['macro_f1_ci_lo']:.2f},{r['macro_f1_ci_hi']:.2f}]"
            print(f"{r['model']:<10} {r['dataset']:<25} {h1:<20} {mf:<20}")

    # R3: Blind Prompts
    if "r3_blind_prompts" in results:
        print("\n── R3: Blind Prompt Comparison (A=CAPE, B=Alt, C=Blind) ──")
        print(f"{'Model':<10} {'Dataset':<25} {'Set':<8} {'Hit@1%':<10} {'Hit@3%':<10} {'MF1%':<10}")
        print("-" * 73)
        for r in results["r3_blind_prompts"]:
            print(f"{r['model']:<10} {r['dataset']:<25} {r['prompt_set']:<8} "
                  f"{r['hit1']:<10} {r['hit3']:<10} {r['macro_f1']:<10}")

    # R4: Multi-label
    if "r4_multilabel" in results:
        print("\n── R4: Multi-label Evaluation (TeacherBehavior) ──")
        print(f"{'Model':<10} {'Threshold':<12} {'Sample-F1%':<12} {'Macro-F1%':<12} {'Micro-F1%':<12}")
        print("-" * 58)
        for r in results["r4_multilabel"]:
            print(f"{r['model']:<10} {str(r['threshold']):<12} {r['sample_f1']:<12} "
                  f"{r['macro_f1']:<12} {r['micro_f1']:<12}")


if __name__ == "__main__":
    main()
