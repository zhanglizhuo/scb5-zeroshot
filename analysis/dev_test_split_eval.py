#!/usr/bin/env python3
"""Dev/test split evaluation for R3.1/R3.2.

Splits the TeacherBehavior validation set (3240 images) into dev (1620)
and test (1620) halves using a fixed seed. On dev, sweeps tau to maximize
Sample-F1. On test, applies the dev-selected tau and reports metrics.

This directly addresses R3.1/R3.2: "Separate data sections should be used
for prompt development, threshold selection, and final testing" and
"Results should be re-reported on an independent test set."

Usage:
    python analysis/dev_test_split_eval.py
Output:
    results/revision/dev_test_split_r3.json
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / "data" / "feature_cache"
OUTPUT_DIR = REPO_ROOT / "results" / "revision"

MODELS = ["openai", "dfn", "siglip2"]
DATASET = "teacher_behavior"
SPLIT = "validation"
NUM_CLASSES = 8
SEED = 42
DEV_FRACTION = 0.5

CLASS_NAMES = [
    "guide", "answer", "on-stage interaction", "blackboard-writing",
    "teacher", "stand", "screen", "blackboard",
]

TAU_GRID = [0.08, 0.10, 0.12, 0.14, 0.15, 0.16, 0.18, 0.20, 0.22, 0.24, 0.25, 0.26, 0.28, 0.30]
TOPK_GRID = [2, 3]


def load_cache(model_key):
    fpath = CACHE_DIR / f"{model_key}_{DATASET}_{SPLIT}.npz"
    if not fpath.exists():
        raise FileNotFoundError(f"Cache not found: {fpath}")
    d = np.load(fpath)
    return d["logits_cape"], d["labels"]


def split_indices(n, seed, dev_fraction):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    cut = int(n * dev_fraction)
    return idx[:cut], idx[cut:]


def threshold_predict(logits, tau):
    preds = (logits >= tau).astype(int)
    for i in range(len(preds)):
        if preds[i].sum() == 0:
            preds[i, logits[i].argmax()] = 1
    return preds


def topk_predict(logits, k):
    preds = np.zeros_like(logits)
    for i in range(len(logits)):
        top_idx = np.argsort(logits[i])[-k:]
        preds[i, top_idx] = 1
    return preds


def compute_f1(labels, preds):
    return {
        "sample_f1": f1_score(labels, preds, average="samples", zero_division=0) * 100,
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0) * 100,
        "micro_f1": f1_score(labels, preds, average="micro", zero_division=0) * 100,
    }


def sweep_tau_on_dev(dev_logits, dev_labels):
    results = []
    for tau in TAU_GRID:
        preds = threshold_predict(dev_logits, tau)
        m = compute_f1(dev_labels, preds)
        results.append({"strategy": f"tau={tau:.2f}", "tau": tau, **m})
    for k in TOPK_GRID:
        preds = topk_predict(dev_logits, k)
        m = compute_f1(dev_labels, preds)
        results.append({"strategy": f"top-{k}", "topk": k, **m})
    best = max(results, key=lambda x: x["sample_f1"])
    return best, results


def eval_on_test(test_logits, test_labels, best_cfg):
    if "tau" in best_cfg:
        preds = threshold_predict(test_logits, best_cfg["tau"])
    else:
        preds = topk_predict(test_logits, best_cfg["topk"])
    return compute_f1(test_labels, preds)


def eval_full_val(logits, labels, best_cfg):
    if "tau" in best_cfg:
        preds = threshold_predict(logits, best_cfg["tau"])
    else:
        preds = topk_predict(logits, best_cfg["topk"])
    return compute_f1(labels, preds)


def main():
    n_total = None
    dev_idx, test_idx = None, None

    output = {
        "timestamp": int(datetime.now().timestamp()),
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": "R3.1/R3.2: dev/test split to verify no optimistic bias in threshold selection",
        "dataset": DATASET,
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "seed": SEED,
        "dev_fraction": DEV_FRACTION,
        "tau_grid": TAU_GRID,
        "topk_grid": TOPK_GRID,
        "models": [],
    }

    for model in MODELS:
        print(f"\n{'='*60}")
        print(f"Model: {model}")
        print(f"{'='*60}")

        logits, labels = load_cache(model)
        n = logits.shape[0]
        if n_total is None:
            n_total = n
            dev_idx, test_idx = split_indices(n, SEED, DEV_FRACTION)
            output["num_total"] = n
            output["num_dev"] = len(dev_idx)
            output["num_test"] = len(test_idx)
            print(f"Total: {n}, Dev: {len(dev_idx)}, Test: {len(test_idx)}")
        elif n != n_total:
            raise ValueError(f"{model}: n={n} != expected {n_total}")

        dev_logits = logits[dev_idx]
        dev_labels = labels[dev_idx]
        test_logits = logits[test_idx]
        test_labels = labels[test_idx]

        best_cfg, sweep = sweep_tau_on_dev(dev_logits, dev_labels)
        print(f"Dev best: {best_cfg['strategy']} -> S-F1={best_cfg['sample_f1']:.2f}")

        test_metrics = eval_on_test(test_logits, test_labels, best_cfg)
        print(f"Test (dev-tau):    S-F1={test_metrics['sample_f1']:.2f}, "
              f"Ma-F1={test_metrics['macro_f1']:.2f}, Mi-F1={test_metrics['micro_f1']:.2f}")

        full_val_metrics = eval_full_val(logits, labels, best_cfg)
        print(f"FullVal (dev-tau): S-F1={full_val_metrics['sample_f1']:.2f}, "
              f"Ma-F1={full_val_metrics['macro_f1']:.2f}, Mi-F1={full_val_metrics['micro_f1']:.2f}")

        # Also compute the "paper" approach: select tau on FULL val, report on FULL val
        fullval_best, _ = sweep_tau_on_dev(logits, labels)
        fullval_paper = eval_full_val(logits, labels, fullval_best)
        print(f"FullVal (fv-tau):  S-F1={fullval_paper['sample_f1']:.2f}, "
              f"Ma-F1={fullval_paper['macro_f1']:.2f}, Mi-F1={fullval_paper['micro_f1']:.2f}")
        print(f"  (paper-style: selected {fullval_best['strategy']} on full-val)")

        delta_sf1 = test_metrics["sample_f1"] - fullval_paper["sample_f1"]
        bias_check = "NO_BIAS" if abs(delta_sf1) < 2.0 else "POTENTIAL_BIAS"
        print(f"Delta(test_dev-tau - fullval_paper) S-F1: {delta_sf1:+.2f} pp [{bias_check}]")

        output["models"].append({
            "model": model,
            "dev_best_strategy": best_cfg["strategy"],
            "dev_best_sample_f1": round(best_cfg["sample_f1"], 2),
            "dev_best_macro_f1": round(best_cfg["macro_f1"], 2),
            "test_sample_f1": round(test_metrics["sample_f1"], 2),
            "test_macro_f1": round(test_metrics["macro_f1"], 2),
            "test_micro_f1": round(test_metrics["micro_f1"], 2),
            "fullval_devtau_sample_f1": round(full_val_metrics["sample_f1"], 2),
            "fullval_devtau_macro_f1": round(full_val_metrics["macro_f1"], 2),
            "fullval_devtau_micro_f1": round(full_val_metrics["micro_f1"], 2),
            "fullval_best_strategy": fullval_best["strategy"],
            "fullval_paper_sample_f1": round(fullval_paper["sample_f1"], 2),
            "fullval_paper_macro_f1": round(fullval_paper["macro_f1"], 2),
            "fullval_paper_micro_f1": round(fullval_paper["micro_f1"], 2),
            "delta_test_vs_paper": round(delta_sf1, 2),
            "bias_check": bias_check,
            "dev_sweep_full": [
                {k: round(v, 2) if isinstance(v, float) else v for k, v in r.items()}
                for r in sweep
            ],
        })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outpath = OUTPUT_DIR / "dev_test_split_r3.json"
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {outpath}")

    print(f"\n{'='*60}")
    print("SUMMARY: dev-selected tau on test vs paper (full-val tau on full-val)")
    print(f"{'='*60}")
    print(f"{'Model':<10} {'Dev best':<12} {'FV best':<12} {'Test S-F1':>10} {'Paper S-F1':>11} {'Delta':>8} {'Bias?':>10}")
    for m in output["models"]:
        print(f"{m['model']:<10} {m['dev_best_strategy']:<12} "
              f"{m['fullval_best_strategy']:<12} "
              f"{m['test_sample_f1']:>10.2f} {m['fullval_paper_sample_f1']:>11.2f} "
              f"{m['delta_test_vs_paper']:>+8.2f} {m['bias_check']:>10}")


if __name__ == "__main__":
    main()
