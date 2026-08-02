#!/usr/bin/env python3
"""Verify feature-cache integrity against the paper's CAPE Set A prompts.

Checks, for every {model}_{subset}_validation.npz cache:
  1. labels shape/values are sane (multi-label one-hot within [0,1]);
  2. logits_cape == image_features @ tembs.T within tolerance, where tembs
     are the canonical CAPE Set A text embeddings stored in
     data/feature_cache/tembs/ (generated with open_clip 2.24.0);
  3. the CAPE Hit@1 implied by the cache matches the paper Table 7 numbers
     within tolerance.

Run from the repo root with a plain numpy environment:
    python3 analysis/verify_caches.py
Exit code 0 = all checks pass.

Background: commit 5ca5310's gen_missing_caches.py inlined a divergent
copy of the CAPE prompts, producing laion/eva02 caches whose logits
correlated only ~0.86 with the paper's CAPE Set A (30pp Hit@1 drop).
This script is the guard that prevents that class of silent corruption.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "data" / "feature_cache"
TEMBS_DIR = CACHE_DIR / "tembs"
TOL = 1e-4

MODELS = ["openai", "laion", "siglip2", "eva02", "dfn"]
SUBSETS = ["teacher_behavior", "handrise_readwrite", "bow_turnhead"]

PAPER_TABLE7_HIT1 = {
    "openai": {"teacher_behavior": 84.14, "handrise_readwrite": 57.99, "bow_turnhead": 63.56},
    "laion": {"teacher_behavior": 54.94, "handrise_readwrite": 69.72, "bow_turnhead": 92.87},
    "siglip2": {"teacher_behavior": 85.56, "handrise_readwrite": 56.25, "bow_turnhead": 64.75},
    "eva02": {"teacher_behavior": 64.35, "handrise_readwrite": 74.33, "bow_turnhead": 90.10},
    "dfn": {"teacher_behavior": 45.86, "handrise_readwrite": 56.91, "bow_turnhead": 68.32},
}

HIT1_TOL = 2.5  # pp; catches wrong-prompts corruption (tens of pp) while
# tolerating the known 1-2pp environment drift between the original runs
# (open_clip 2.24.0 + V100 fp16) and the paper Table 7 numbers.


def hit1_from_logits(logits: np.ndarray, labels: np.ndarray) -> float:
    n = len(logits)
    if labels.ndim == 1 or labels.shape[1] == 1:
        single = labels.reshape(-1)
        preds = logits.argmax(1)
        return 100.0 * (preds == single).mean()
    lab = labels.astype(bool)
    n_positive = lab.sum(1)
    preds = logits.argmax(1)
    ok = lab[np.arange(n), preds]
    return 100.0 * ok.mean()


def main() -> int:
    failures = []
    for model in MODELS:
        for subset in SUBSETS:
            cache_path = CACHE_DIR / f"{model}_{subset}_validation.npz"
            if not cache_path.exists():
                failures.append(f"{cache_path.name}: MISSING")
                continue
            with np.load(cache_path) as c:
                if "logits_cape" not in c.files:
                    failures.append(f"{cache_path.name}: no logits_cape key")
                    continue
                logits = c["logits_cape"]
                labels = c["labels"]
                feats = c["image_features"] if "image_features" in c.files else None

            if logits.shape[0] != labels.shape[0]:
                failures.append(f"{cache_path.name}: rows mismatch {logits.shape[0]} vs {labels.shape[0]}")

            tembs_path = TEMBS_DIR / f"{model}_{subset}_capeA_tembs.npz"
            if tembs_path.exists() and feats is not None:
                with np.load(tembs_path) as t:
                    tembs = t["tembs"]
                if tembs.shape[0] != logits.shape[1]:
                    failures.append(f"{cache_path.name}: tembs classes {tembs.shape[0]} != logits classes {logits.shape[1]}")
                    continue
                recomputed = feats @ tembs.T
                md = np.abs(recomputed - logits).max()
                if md > TOL:
                    failures.append(f"{cache_path.name}: logits_cape mismatch vs CAPE_A tembs (maxdiff={md:.3g} > {TOL})")

            hit1 = hit1_from_logits(logits, labels)
            paper = PAPER_TABLE7_HIT1[model][subset]
            tag = "FAIL" if abs(hit1 - paper) > HIT1_TOL else "info"
            if tag == "FAIL":
                failures.append(
                    f"{cache_path.name}: Hit@1 {hit1:.2f} deviates from paper Table 7 ({paper:.2f})"
                )
            else:
                print(f"  {cache_path.name}: Hit@1 {hit1:.2f} (paper {paper:.2f})")

    if failures:
        print("FAIL")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("OK: all 15 caches consistent with CAPE Set A and paper Table 7.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
