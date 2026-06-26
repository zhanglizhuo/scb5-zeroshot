#!/usr/bin/env python3
"""
CAPE three-principle ablation: isolates the contribution of each design
principle (visual grounding, semantic diversity, discriminative contrast).

Source of Table 5 (cape_ablation) in the paper.

Usage:
    python scb5_zeroshot/cape_principle_ablation.py \
        --cache_dir data/feature_cache

Requires precomputed per-sample logits for all ablation conditions
in data/feature_cache/.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

CONDITIONS = ["C0", "C1", "C2", "C3"]
CONDITION_LABELS = {
    "C0": "CAPE-Full",
    "C1": r"C1 $\neg$VisGround",
    "C2": r"C2 $\neg$SemDiv",
    "C3": r"C3 $\neg$Discrim",
}
SUBSETS = ["teacher", "bowturn"]
SUBSET_LABELS = {"teacher": "TeacherBehavior", "bowturn": "BowTurnHead"}
BACKBONES = ["openai", "siglip2", "dfn"]
BACKBONE_LABELS = {"openai": "CLIP (OpenAI)", "siglip2": "SigLIP2", "dfn": "DFN-CLIP"}


def load_ablation_logits(cache_dir: Path, backbone: str, condition: str):
    path = cache_dir / "ablation" / f"{backbone}_logits_{condition.lower()}.npy"
    if not path.exists():
        raise FileNotFoundError(
            f"Ablation logits not found: {path}\n"
            f"See data/feature_cache/README.md for download instructions."
        )
    return np.load(path)


def load_multilabel_sets(cache_dir: Path):
    path = cache_dir / "teacher" / "multilabel_sets.npy"
    return np.load(path, allow_pickle=True)


def load_primary_labels(cache_dir: Path, subset: str):
    path = cache_dir / subset / "primary_labels.npy"
    return np.load(path)


def compute_metrics_teacher(logits, ml_sets, k=None):
    preds = logits.argmax(axis=1)
    y_true_ml = np.array([list(s) for s in ml_sets])

    topk_preds = np.argsort(-logits, axis=1)
    sample_f1_scores = []
    macro_f1_scores = []
    micro_f1_scores = []

    n_classes = logits.shape[1]

    for i in range(len(logits)):
        if k is not None:
            n_pred = k
        else:
            n_pred = len(ml_sets[i])
        y_pred = np.zeros(n_classes)
        y_pred[topk_preds[i, :n_pred]] = 1
        y_true = np.zeros(n_classes)
        y_true[list(ml_sets[i])] = 1

        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))

        if tp + fp > 0:
            prec = tp / (tp + fp)
        else:
            prec = 0
        if tp + fn > 0:
            rec = tp / (tp + fn)
        else:
            rec = 0
        if prec + rec > 0:
            sample_f1_scores.append(2 * prec * rec / (prec + rec))
        else:
            sample_f1_scores.append(0.0)

    hit_at_1 = np.mean(
        [preds[i] in ml_sets[i] for i in range(len(preds))]
    )

    y_true_flat = np.concatenate(
        [np.array(list(s)) for s in ml_sets]
    )
    y_pred_flat = np.concatenate(
        [np.full(len(ml_sets[i]), preds[i]) for i in range(len(preds))]
    )
    macro_f1 = f1_score(
        y_true_flat, y_pred_flat, average="macro", zero_division=0
    )
    micro_f1 = f1_score(
        y_true_flat, y_pred_flat, average="micro", zero_division=0
    )
    sample_f1 = np.mean(sample_f1_scores)

    return hit_at_1 * 100, sample_f1 * 100, macro_f1 * 100, micro_f1 * 100


def compute_metrics_bowturn(logits, prim_labels):
    preds = logits.argmax(axis=1)
    acc = np.mean(preds == prim_labels)
    macro_f1 = f1_score(prim_labels, preds, average="macro", zero_division=0)
    return acc * 100, macro_f1 * 100


def run_ablation(cache_dir):
    print()
    print(f"{'Backbone':<20s} {'Condition':<22s} {'Hit@1':>7s} "
          f"{'S-F1':>7s} {'Ma-F1':>7s} {'Mi-F1':>7s} {'BowHit@1':>9s}")
    print("-" * 85)

    ml_sets_teacher = load_multilabel_sets(cache_dir)
    prim_labels_bowturn = load_primary_labels(cache_dir, "bowturn")

    for backbone in BACKBONES:
        for cond in CONDITIONS:
            logits_teacher = load_ablation_logits(
                cache_dir, backbone, f"{cond}_teacher"
            )
            hit, sf, maf, mif = compute_metrics_teacher(
                logits_teacher, ml_sets_teacher
            )

            logits_bow = load_ablation_logits(
                cache_dir, backbone, f"{cond}_bowturn"
            )
            bow_acc, _ = compute_metrics_bowturn(logits_bow, prim_labels_bowturn)

            print(f"{BACKBONE_LABELS[backbone]:<20s} "
                  f"{CONDITION_LABELS[cond]:<22s} "
                  f"{hit:>6.2f} {sf:>6.2f} {maf:>6.2f} "
                  f"{mif:>6.2f} {bow_acc:>8.2f}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="CAPE three-principle ablation."
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="data/feature_cache",
        help="Path to feature cache directory (default: data/feature_cache)",
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    if not cache_dir.exists():
        print(
            f"Error: feature cache directory not found: {cache_dir}\n"
            f"Download cached features first (see data/feature_cache/README.md).",
            file=sys.stderr,
        )
        sys.exit(1)

    run_ablation(cache_dir)


if __name__ == "__main__":
    main()
