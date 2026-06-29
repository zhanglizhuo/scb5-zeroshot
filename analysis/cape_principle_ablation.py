#!/usr/bin/env python3
"""
cape_principle_ablation.py — CAPE Three-Principle Ablation (Reviewer M1)
=========================================================================
Reviewer concern: "CAPE is just a manual prompt ensemble; what does it
actually contribute beyond averaging more prompts?"

We test three CAPE design principles by individually degrading each while
keeping the other two and the prompt count (=3 per class) fixed:

  P1. Visual grounding         : concrete physical descriptions
  P2. Semantic diversity       : three viewpoints (action / agent / scene)
  P3. Discriminative contrast  : class-specific distinguishing features

Conditions evaluated (per backbone × dataset):
  C0 CAPE-Full                  : original CAPE-A (all three principles)
  C1 ¬VisualGrounding           : abstract, conceptual prompts
  C2 ¬SemanticDiversity         : 3 paraphrases of one viewpoint
  C3 ¬DiscriminativeContrast    : generic per-class wording
  C4 PhotoOf (baseline)         : single "a photo of {cls}" prompt

Uses cached image features under data/feature_cache/. Only text features
are recomputed.

Usage:
  CUDA_VISIBLE_DEVICES=0 python3 cape_principle_ablation.py
"""
import json
import time
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
import open_clip

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "data" / "feature_cache"
OUT_DIR = ROOT / "results" / "revision"
OUT_DIR.mkdir(exist_ok=True, parents=True)

# Backbones with cached multi-label features (3240 samples)
MODEL_SPECS = {
    "openai":  ("ViT-L-14", "openai", "clip"),
    "dfn":     ("ViT-L-14", "dfn2b", str(ROOT / "ckpts" / "dfn_clip_vitl14" / "open_clip_pytorch_model.bin")),
    "siglip2": ("ViT-L-16-SigLIP2-256", "webli", str(ROOT / "ckpts" / "siglip_vitl16_256" / "open_clip_pytorch_model.bin")),
}

DATASETS = {
    "teacher_behavior": {
        "classes": ["guide", "answer", "on-stage interaction", "blackboard-writing",
                    "teacher", "stand", "screen", "blackboard"],
        "multi_label": True,
    },
    "handrise_readwrite": {
        "classes": ["hand-raise", "read", "write"],
        "multi_label": False,
    },
    "bow_turnhead": {
        "classes": ["bow-head", "turn-head"],
        "multi_label": False,
    },
}

# ====================================================================
# C0 — CAPE Full (3 principles preserved). Same as cape_robustness.CAPE_A
# ====================================================================
CAPE_FULL = {
    # TeacherBehavior
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
    "on-stage interaction": [
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
    "blackboard": [
        "a teacher pointing at the blackboard",
        "a teacher referring to content on the blackboard",
        "a blackboard with writing visible in a classroom",
    ],
    # HandriseReadWrite
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
    # BowTurnHead
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

# ====================================================================
# C1 — drop Visual Grounding
# Replace concrete physical descriptions with abstract / conceptual wording.
# Still 3 prompts/class, still semantically diverse, still discriminative
# at the conceptual level — but no observable visual cue.
# ====================================================================
CAPE_NO_VISUAL = {
    # TeacherBehavior
    "guide":              ["individualized academic guidance", "personalized tutoring", "one-to-one instructional support"],
    "answer":             ["responsive question-answering", "interactive Q&A exchange", "verbal feedback to a query"],
    "on-stage interaction": ["frontal classroom engagement", "podium-based pedagogical interaction", "interactive frontal teaching"],
    "blackboard-writing": ["chalk-mediated content inscription", "didactic note transcription", "written exposition activity"],
    "teacher":            ["lecture-mode delivery", "expository instruction", "pedagogical explanation"],
    "stand":              ["passive non-interactive posture", "stationary idle stance", "non-engaged standing presence"],
    "screen":             ["digital projection display", "slide-based presentation", "screen-mediated presentation"],
    "blackboard":         ["chalkboard reference activity", "blackboard-directed attention", "board content reference"],
    # HandriseReadWrite
    "hand-raise":         ["participation indication gesture", "verbal-turn solicitation", "engagement signaling"],
    "read":               ["textual comprehension activity", "study-material engagement", "academic reading task"],
    "write":              ["academic note recording", "scholastic transcription", "written task execution"],
    # BowTurnHead
    "bow-head":           ["downward-attention disengagement", "off-task posture", "attention-detached pose"],
    "turn-head":          ["lateral-attention shift", "off-axis gaze", "non-frontal attention orientation"],
}

# ====================================================================
# C2 — drop Semantic Diversity
# Use 3 near-paraphrases of ONE viewpoint per class (e.g., always agent-
# action). Still grounded, still discriminative, but redundant.
# ====================================================================
CAPE_NO_DIVERSITY = {
    # TeacherBehavior
    "guide":              ["a teacher guiding a student one-on-one",
                           "a teacher guiding one student individually",
                           "a teacher providing one-on-one guidance to a student"],
    "answer":             ["a teacher answering a student's question",
                           "a teacher responding to a student's question",
                           "a teacher giving an answer to a student's question"],
    "on-stage interaction": ["a teacher interacting with students at the front",
                             "a teacher interacting with students at the front of the room",
                             "a teacher interacting with students at the front of the class"],
    "blackboard-writing": ["a teacher writing on a blackboard",
                           "a teacher writing on the blackboard with chalk",
                           "a teacher writing notes on a blackboard"],
    "teacher":            ["a teacher standing and lecturing",
                           "a teacher standing while lecturing",
                           "a teacher standing as they lecture"],
    "stand":              ["a teacher standing still in a classroom",
                           "a teacher standing motionless in a classroom",
                           "a teacher just standing in a classroom"],
    "screen":             ["a teacher pointing at a screen",
                           "a teacher pointing toward a screen",
                           "a teacher pointing at the screen"],
    "blackboard":         ["a teacher pointing at a blackboard",
                           "a teacher pointing toward a blackboard",
                           "a teacher pointing at the blackboard"],
    # HandriseReadWrite
    "hand-raise":         ["a student raising their hand",
                           "a student raising one hand",
                           "a student with hand raised"],
    "read":               ["a student reading a book",
                           "a student reading from a book",
                           "a student reading a textbook"],
    "write":              ["a student writing in a notebook",
                           "a student writing in their notebook",
                           "a student writing into a notebook"],
    # BowTurnHead
    "bow-head":           ["a student with head bowed down",
                           "a student bowing their head down",
                           "a student whose head is bowed down"],
    "turn-head":          ["a student turning their head sideways",
                           "a student turning the head to the side",
                           "a student with head turned sideways"],
}

# ====================================================================
# C3 — drop Discriminative Contrast
# Use generic per-class descriptions that could fit several classes.
# Still grounded, still 3 prompts, still 3 viewpoints — but each prompt
# alone is not class-distinctive.
# ====================================================================
CAPE_NO_DISCRIM = {
    # TeacherBehavior — all teacher classes use generic teacher-related wording
    "guide":              ["a teacher and a student in a classroom", "people interacting in a classroom", "a classroom scene with a teacher"],
    "answer":             ["a teacher and students in a classroom", "people talking in a classroom", "a classroom scene with people"],
    "on-stage interaction": ["a teacher in front of a classroom", "people gathered in a classroom", "a classroom scene with a teacher"],
    "blackboard-writing": ["a teacher and a board in a classroom", "a person near a board in a classroom", "a classroom with a board visible"],
    "teacher":            ["a teacher in a classroom", "a person teaching in a classroom", "a classroom scene with a teacher"],
    "stand":              ["a teacher in a classroom", "a person in a classroom", "a classroom scene with a person"],
    "screen":             ["a teacher and a screen in a classroom", "a person near a screen in a classroom", "a classroom with a screen visible"],
    "blackboard":         ["a teacher and a board in a classroom", "a person near a board in a classroom", "a classroom with a board visible"],
    # HandriseReadWrite — all use generic student-in-class wording
    "hand-raise":         ["a student in a classroom", "a person in a classroom seat", "a classroom scene with a student"],
    "read":               ["a student in a classroom", "a person at a desk in a classroom", "a classroom scene with a student"],
    "write":              ["a student in a classroom", "a person at a desk in a classroom", "a classroom scene with a student"],
    # BowTurnHead — all use generic head-position wording
    "bow-head":           ["a student in a classroom", "a person in a classroom seat", "a classroom scene with a student"],
    "turn-head":          ["a student in a classroom", "a person in a classroom seat", "a classroom scene with a student"],
}

CONDITIONS = {
    "C0_CAPE_Full":         CAPE_FULL,
    "C1_NoVisualGrounding": CAPE_NO_VISUAL,
    "C2_NoSemanticDiversity": CAPE_NO_DIVERSITY,
    "C3_NoDiscrimContrast": CAPE_NO_DISCRIM,
}


def load_text_encoder(model_key, device):
    model_name, pretrained, ckpt_or_tag = MODEL_SPECS[model_key]
    if model_key == "openai":
        import clip as clip_module
        model, _ = clip_module.load("ViT-L/14", device=device)
        tokenizer = clip_module.tokenize

        def encode_text(texts):
            with torch.no_grad():
                tok = tokenizer(texts).to(device)
                feat = model.encode_text(tok)
                return F.normalize(feat, dim=-1)
        return encode_text, model
    else:
        ckpt = ckpt_or_tag
        model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=ckpt)
        model = model.to(device).eval()
        if model_key == "siglip2":
            tokenizer = open_clip.get_tokenizer(f"hf-hub:{Path(ckpt).parent}", context_length=64)
        else:
            tokenizer = open_clip.get_tokenizer(model_name)

        def encode_text(texts):
            with torch.no_grad():
                tok = tokenizer(texts).to(device)
                feat = model.encode_text(tok)
                return F.normalize(feat, dim=-1)
        return encode_text, model


def build_class_text_features(encode_text, classes, prompts_dict):
    feats = []
    for cls in classes:
        prompts = prompts_dict.get(cls, [f"a photo of {cls}"])
        f = encode_text(prompts).mean(dim=0, keepdim=True)
        f = F.normalize(f, dim=-1)
        feats.append(f)
    return torch.cat(feats, dim=0)  # [K, D]


def evaluate(image_features, labels, text_features, multi_label):
    """image_features: [N,D] np.ndarray (assumed L2-normalized from cache).
    labels: [N,K] np.ndarray (multi-label) or [N,K] one-hot (single-label).
    Returns dict of metrics."""
    img = torch.from_numpy(image_features).float()
    img = F.normalize(img, dim=-1)
    sims = img @ text_features.cpu().float().T  # [N,K]
    sims_np = sims.numpy()

    if multi_label:
        # Multi-label: threshold at 0 logit-margin equivalent — use top-k where
        # k = number of positives per sample (matches CAPE evaluation convention)
        # Standard approach: threshold cosine sim at the median for each class
        # Match the protocol of run_revision_experiments.py: argmax → hit@1 or
        # rank top-k with k = #pos. To be consistent with the paper's main
        # zero-shot table we use ARGMAX → hit@1 over multi-label hit set.
        preds_idx = sims_np.argmax(axis=1)
        hits = labels[np.arange(len(labels)), preds_idx]
        hit1 = float(hits.mean()) * 100

        # Multi-label F1: predict positive if class is in top-k where
        # k = ground-truth positive count (gives a "best-case" rank metric;
        # this is the convention the CAPE paper uses for ML F1).
        bin_preds = np.zeros_like(labels)
        for i in range(len(sims_np)):
            k = int(labels[i].sum())
            if k <= 0:
                continue
            top_k = np.argsort(-sims_np[i])[:k]
            bin_preds[i, top_k] = 1
        sample_f1 = f1_score(labels, bin_preds, average="samples", zero_division=0) * 100
        macro_f1 = f1_score(labels, bin_preds, average="macro", zero_division=0) * 100
        micro_f1 = f1_score(labels, bin_preds, average="micro", zero_division=0) * 100
        return {
            "hit1": round(hit1, 2),
            "sample_f1": round(sample_f1, 2),
            "macro_f1": round(macro_f1, 2),
            "micro_f1": round(micro_f1, 2),
        }
    else:
        # Single-label: labels are one-hot or class index per row
        if labels.ndim == 2:
            y = labels.argmax(axis=1)
        else:
            y = labels
        preds_idx = sims_np.argmax(axis=1)
        hit1 = float((preds_idx == y).mean()) * 100
        macro_f1 = f1_score(y, preds_idx, average="macro", zero_division=0) * 100
        return {
            "hit1": round(hit1, 2),
            "macro_f1": round(macro_f1, 2),
        }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    results = {"timestamp": int(time.time()),
               "device": str(device),
               "experiments": []}

    for model_key in MODEL_SPECS.keys():
        log.info(f"\n{'='*60}\nLoading text encoder: {model_key}\n{'='*60}")
        try:
            encode_text, model_obj = load_text_encoder(model_key, device)
        except Exception as e:
            log.error(f"  Failed to load {model_key}: {e}")
            continue

        for ds_name, ds_info in DATASETS.items():
            cache_path = CACHE_DIR / f"{model_key}_{ds_name}_validation.npz"
            if not cache_path.exists():
                log.warning(f"  Cache not found: {cache_path}; skipping")
                continue
            d = np.load(cache_path)
            img_feats = d["image_features"]
            labels = d["labels"]
            classes = ds_info["classes"]
            multi_label = ds_info["multi_label"]
            log.info(f"\n[{model_key} | {ds_name}] {img_feats.shape[0]} samples, "
                     f"{len(classes)} classes, multi_label={multi_label}")

            # Single "a photo of {cls}" baseline
            photo_dict = {c: [f"a photo of {c}"] for c in classes}
            tf = build_class_text_features(encode_text, classes, photo_dict)
            metrics = evaluate(img_feats, labels, tf, multi_label)
            log.info(f"  PhotoOf:                {metrics}")
            results["experiments"].append({
                "model": model_key, "dataset": ds_name,
                "condition": "C4_PhotoOf", **metrics,
            })

            for cond_name, prompts_dict in CONDITIONS.items():
                tf = build_class_text_features(encode_text, classes, prompts_dict)
                metrics = evaluate(img_feats, labels, tf, multi_label)
                log.info(f"  {cond_name:<24}: {metrics}")
                results["experiments"].append({
                    "model": model_key, "dataset": ds_name,
                    "condition": cond_name, **metrics,
                })

        del encode_text, model_obj
        torch.cuda.empty_cache()

    out_path = OUT_DIR / f"cape_principle_ablation_{int(time.time())}.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    log.info(f"\nSaved -> {out_path}")

    # ---- Summary table ----
    print("\n" + "=" * 90)
    print("CAPE Three-Principle Ablation Summary")
    print("=" * 90)
    for ds_name in DATASETS.keys():
        print(f"\n[{ds_name}]")
        if DATASETS[ds_name]["multi_label"]:
            header = f"{'Model':<10} {'Condition':<26} {'Hit@1':>7} {'SamF1':>7} {'MacF1':>7} {'MicF1':>7}"
        else:
            header = f"{'Model':<10} {'Condition':<26} {'Hit@1':>7} {'MacF1':>7}"
        print(header)
        print("-" * len(header))
        for r in results["experiments"]:
            if r["dataset"] != ds_name:
                continue
            if DATASETS[ds_name]["multi_label"]:
                print(f"{r['model']:<10} {r['condition']:<26} "
                      f"{r['hit1']:>7.2f} {r.get('sample_f1', float('nan')):>7.2f} "
                      f"{r.get('macro_f1', float('nan')):>7.2f} {r.get('micro_f1', float('nan')):>7.2f}")
            else:
                print(f"{r['model']:<10} {r['condition']:<26} "
                      f"{r['hit1']:>7.2f} {r.get('macro_f1', float('nan')):>7.2f}")


if __name__ == "__main__":
    main()
