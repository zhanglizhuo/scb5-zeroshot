#!/usr/bin/env python3
"""
Paired bootstrap significance test for CAPE zero-shot performance.

Source of Table 9 (paired_bootstrap) in the paper.

Usage:
    python scb5_zeroshot/paired_bootstrap.py \
        --cache_dir data/feature_cache \
        --n_resamples 5000 \
        --seed 42

Requires precomputed per-sample logits in data/feature_cache/.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

SUBSETS = ["teacher", "handrise", "bowturn"]
SUBSET_LABELS = {
    "teacher": "TeacherBehavior",
    "handrise": "HandriseReadWrite",
    "bowturn": "BowTurnHead",
}
MODEL_PAIRS = [
    ("openai", "dfn"),
    ("openai", "siglip2"),
    ("dfn", "siglip2"),
]
MODEL_NAMES = {
    "openai": "CLIP (OpenAI)",
    "dfn": "DFN-CLIP",
    "siglip2": "SigLIP2",
}


def load_cape_logits(cache_dir: Path, subset: str, model: str):
    path = cache_dir / subset / f"{model}_logits_cape.npy"
    if not path.exists():
        raise FileNotFoundError(
            f"Feature cache not found: {path}. "
            f"See data/feature_cache/README.md for download instructions."
        )
    return np.load(path)


def load_primary_labels(cache_dir: Path, subset: str):
    path = cache_dir / subset / "primary_labels.npy"
    if not path.exists():
        raise FileNotFoundError(f"Primary labels not found: {path}")
    return np.load(path)


def load_multilabel_sets(cache_dir: Path, subset: str):
    path = cache_dir / subset / "multilabel_sets.npy"
    if not path.exists():
        raise FileNotFoundError(f"Multi-label sets not found: {path}")
    return np.load(path, allow_pickle=True)


def hit_at_1_from_logits(logits, multilabel_sets):
    preds = logits.argmax(axis=1)
    hits = np.array([pred in ml_set for pred, ml_set in zip(preds, multilabel_sets)])
    return hits.astype(float)


def accuracy_from_logits(logits, primary_labels):
    preds = logits.argmax(axis=1)
    return (preds == primary_labels).astype(float)


def paired_bootstrap(scores_a, scores_b, n_resamples=5000, seed=42, alpha=0.05):
    rng = np.random.default_rng(seed)
    n = len(scores_a)
    deltas = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        deltas[i] = np.mean(scores_a[idx]) - np.mean(scores_b[idx])
    delta_obs = np.mean(scores_a) - np.mean(scores_b)
    ci_lower = np.percentile(deltas, 100 * alpha / 2)
    ci_upper = np.percentile(deltas, 100 * (1 - alpha / 2))
    p_value = 2 * min(
        np.mean(deltas >= 0), np.mean(deltas <= 0)
    )
    return delta_obs, ci_lower, ci_upper, p_value


def run_paired_bootstrap(cache_dir, n_resamples, seed):
    for subset in SUBSETS:
        if subset == "teacher":
            ml_sets = load_multilabel_sets(cache_dir, subset)
        else:
            prim_labels = load_primary_labels(cache_dir, subset)

        scores = {}
        for model_key, _ in MODEL_PAIRS:
            if model_key not in scores:
                logits = load_cape_logits(cache_dir, subset, model_key)
                if subset == "teacher":
                    scores[model_key] = hit_at_1_from_logits(logits, ml_sets)
                else:
                    scores[model_key] = accuracy_from_logits(logits, prim_labels)

        print(f"\n{'=' * 72}")
        print(f"  Subset: {SUBSET_LABELS[subset]}")
        print(f"{'=' * 72}")
        print(f"  {'Model A':<20s} {'Model B':<20s} {'Acc A':>8s} {'Acc B':>8s} "
              f"{'Δ (pp)':>8s} {'95% CI (pp)':>16s} {'p':>10s}")
        print(f"  {'-' * 88}")

        for model_a, model_b in MODEL_PAIRS:
            sa = scores[model_a]
            sb = scores[model_b]
            delta, ci_lo, ci_hi, p = paired_bootstrap(
                sa, sb, n_resamples=n_resamples, seed=seed
            )
            acc_a = np.mean(sa) * 100
            acc_b = np.mean(sb) * 100
            delta_pp = delta * 100
            sign = "+" if delta_pp >= 0 else ""
            p_str = f"{p:.3e}" if p < 0.001 else f"{p:.3f}"
            print(f"  {MODEL_NAMES[model_a]:<20s} {MODEL_NAMES[model_b]:<20s} "
                  f"{acc_a:>7.2f} {acc_b:>7.2f} "
                  f"{sign}{delta_pp:>7.2f} "
                  f"[{ci_lo * 100:>6.1f},{ci_hi * 100:>6.1f}] {p_str:>10s}")

    print(f"\n  n_resamples={n_resamples}, seed={seed}, two-sided")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Paired bootstrap significance test for CAPE zero-shot results."
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="data/feature_cache",
        help="Path to feature cache directory (default: data/feature_cache)",
    )
    parser.add_argument(
        "--n_resamples",
        type=int,
        default=5000,
        help="Number of bootstrap resamples (default: 5000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
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

    run_paired_bootstrap(cache_dir, args.n_resamples, args.seed)


if __name__ == "__main__":
    main()
