"""
pipeline.py — 论文完整实验流水线
==================================
自动运行 E1(模型对比) + E2(Prompt消融) + E3(混淆矩阵) + E4(CAPE)

Usage:
  # 全部实验（所有模型 × 所有 prompt × CAPE）
  CUDA_VISIBLE_DEVICES=3 python pipeline.py --gpu 0

  # 只跑单模型
  CUDA_VISIBLE_DEVICES=3 python pipeline.py --gpu 0 --models siglip

  # 只跑 CAPE
  CUDA_VISIBLE_DEVICES=3 python pipeline.py --gpu 0 --only cape

  # 指定 checkpoint
  CUDA_VISIBLE_DEVICES=3 python pipeline.py --gpu 0 \
      --ckpt_siglip ./ckpts/siglip_vitl16_256/open_clip_pytorch_model.bin
"""

import os, sys, json, argparse, time, logging
from pathlib import Path
from collections import defaultdict

import numpy as np
from sklearn.metrics import (
    confusion_matrix, f1_score, balanced_accuracy_score,
    precision_score, recall_score, classification_report,
)

from prompts import (
    CLASS_NAMES, CLASS_DESCRIPTIONS,
    PROMPT_GROUPS, CAPE_PROMPTS,
)
from run_experiment import run_single_experiment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

NUM_CLASSES = len(CLASS_NAMES)


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────
def compute_full_metrics(y_true, y_pred):
    """Compute all metrics needed for the paper."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_rec = recall_score(y_true, y_pred, average="macro", zero_division=0)

    per_class = {}
    for i, cls in enumerate(CLASS_NAMES):
        tp = cm[i, i]
        total = cm[i].sum()
        per_class[cls] = {
            "recall":  round(tp / total * 100, 2) if total > 0 else 0,
            "support": int(total),
        }

    # Top confusion pairs
    confusion_pairs = []
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if i != j and cm[i, j] > 0:
                confusion_pairs.append({
                    "true": CLASS_NAMES[i],
                    "pred": CLASS_NAMES[j],
                    "count": int(cm[i, j]),
                })
    confusion_pairs.sort(key=lambda x: x["count"], reverse=True)

    return {
        "confusion_matrix": cm.tolist(),
        "macro_f1":         round(macro_f1 * 100, 2),
        "weighted_f1":      round(weighted_f1 * 100, 2),
        "balanced_accuracy": round(bal_acc * 100, 2),
        "macro_precision":  round(macro_prec * 100, 2),
        "macro_recall":     round(macro_rec * 100, 2),
        "per_class":        per_class,
        "top_confusions":   confusion_pairs[:15],
    }


# ─────────────────────────────────────────────
# Prompt builders
# ─────────────────────────────────────────────
def build_prompts_for_group(group_name):
    """Build prompts dict from PROMPT_GROUPS template."""
    templates = PROMPT_GROUPS[group_name]
    prompts = {}
    for cls in CLASS_NAMES:
        desc = CLASS_DESCRIPTIONS[cls]
        prompts[cls] = [t.format(cls=desc) for t in templates]
    return prompts


def build_cape_prompts():
    """Build CAPE prompts dict — uses class-specific prompts."""
    prompts = {}
    for cls in CLASS_NAMES:
        if cls in CAPE_PROMPTS:
            prompts[cls] = CAPE_PROMPTS[cls]
        else:
            desc = CLASS_DESCRIPTIONS[cls]
            prompts[cls] = [
                f"a classroom scene where a teacher is {desc}",
                f"a teacher is {desc} during a lecture",
            ]
    return prompts


# ─────────────────────────────────────────────
# E1 + E2: Model × Prompt comparison
# ─────────────────────────────────────────────
def run_e1_e2(models, device, local_paths, batch_size, gpu):
    results = []
    for model in models:
        for group_name in PROMPT_GROUPS:
            tag = f"model={model}, prompt={group_name}"
            log.info(f"\n[E1+E2] {tag}")

            prompts = build_prompts_for_group(group_name)
            out = run_single_experiment(
                model, prompts,
                device=device, local_paths=local_paths,
                batch_size=batch_size, gpu=gpu,
            )

            metrics = compute_full_metrics(out["y_true"], out["y_pred"])

            entry = {
                "model":  model,
                "desc":   out["desc"],
                "prompt": group_name,
                "total":  out["total"],
                "top1":   out["top1"],
                "top3":   out["top3"],
                **metrics,
            }
            results.append(entry)

            log.info(f"  top1={out['top1']:.2f}%  top3={out['top3']:.2f}%  "
                     f"macro_f1={metrics['macro_f1']:.2f}%  "
                     f"bal_acc={metrics['balanced_accuracy']:.2f}%")
    return results


# ─────────────────────────────────────────────
# E4: CAPE
# ─────────────────────────────────────────────
def run_e4_cape(models, device, local_paths, batch_size, gpu):
    results = []
    cape_prompts = build_cape_prompts()

    for model in models:
        log.info(f"\n[E4 CAPE] model={model}")

        out = run_single_experiment(
            model, cape_prompts,
            device=device, local_paths=local_paths,
            batch_size=batch_size, gpu=gpu,
        )

        metrics = compute_full_metrics(out["y_true"], out["y_pred"])

        entry = {
            "model":  model,
            "desc":   out["desc"],
            "method": "CAPE",
            "total":  out["total"],
            "top1":   out["top1"],
            "top3":   out["top3"],
            **metrics,
        }
        results.append(entry)

        log.info(f"  CAPE top1={out['top1']:.2f}%  top3={out['top3']:.2f}%  "
                 f"macro_f1={metrics['macro_f1']:.2f}%")
    return results


# ─────────────────────────────────────────────
# Summary tables
# ─────────────────────────────────────────────
def print_e1_table(results):
    """Print Table 1: Model comparison (best prompt per model)."""
    print("\n" + "=" * 75)
    print("  TABLE 1: Model Comparison (best prompt per model)")
    print("=" * 75)
    print(f"  {'Model':<20} {'Prompt':<12} {'Top1':>6} {'Top3':>6} "
          f"{'MacF1':>6} {'BalAcc':>6}")
    print("-" * 75)
    # group by model, pick best top1
    by_model = defaultdict(list)
    for r in results:
        by_model[r["model"]].append(r)
    for model in by_model:
        best = max(by_model[model], key=lambda x: x["top1"])
        print(f"  {best['desc']:<20.20} {best['prompt']:<12} "
              f"{best['top1']:>5.2f}% {best['top3']:>5.2f}% "
              f"{best['macro_f1']:>5.2f}% {best['balanced_accuracy']:>5.2f}%")
    print("=" * 75)


def print_e2_table(results):
    """Print Table 2: Prompt ablation."""
    print("\n" + "=" * 80)
    print("  TABLE 2: Prompt Ablation (Top-1 Accuracy %)")
    print("=" * 80)
    models = sorted(set(r["model"] for r in results))
    groups = list(PROMPT_GROUPS.keys())
    header = f"  {'Prompt':<12}" + "".join(f" {m:>10}" for m in models)
    print(header)
    print("-" * 80)
    for g in groups:
        row = f"  {g:<12}"
        for m in models:
            match = [r for r in results if r["model"] == m and r["prompt"] == g]
            if match:
                row += f" {match[0]['top1']:>9.2f}%"
            else:
                row += f" {'N/A':>10}"
        print(row)
    print("=" * 80)


def print_cape_table(e1e2_results, cape_results):
    """Print Table 3: CAPE vs best baseline."""
    print("\n" + "=" * 70)
    print("  TABLE 3: CAPE vs Best Baseline")
    print("=" * 70)
    print(f"  {'Model':<10} {'Baseline':>10} {'CAPE':>10} {'Δ':>8}")
    print("-" * 70)
    by_model = defaultdict(list)
    for r in e1e2_results:
        by_model[r["model"]].append(r)
    for cr in cape_results:
        m = cr["model"]
        if m in by_model:
            best_base = max(by_model[m], key=lambda x: x["top1"])["top1"]
            cape_val = cr["top1"]
            delta = cape_val - best_base
            sign = "+" if delta > 0 else ""
            print(f"  {m:<10} {best_base:>9.2f}% {cape_val:>9.2f}% "
                  f"{sign}{delta:>6.2f}%")
    print("=" * 70)


def print_confusion_top(results, n=10):
    """Print top confusion pairs across models."""
    print("\n" + "=" * 60)
    print("  Top Confusion Pairs (from best prompt)")
    print("=" * 60)
    by_model = defaultdict(list)
    for r in results:
        by_model[r["model"]].append(r)
    for model in by_model:
        best = max(by_model[model], key=lambda x: x["top1"])
        print(f"\n  [{model}] ({best['prompt']})")
        for pair in best["top_confusions"][:n]:
            print(f"    {pair['true']:>25} → {pair['pred']:<25} "
                  f"({pair['count']} imgs)")
    print("=" * 60)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="SCB5 Paper Experiments E1-E4")
    p.add_argument("--models", nargs="+",
                   default=["clip", "laion", "flip", "siglip"],
                   help="Models to evaluate")
    p.add_argument("--only", default=None,
                   choices=["e1e2", "cape", "all"],
                   help="Run only specific experiment set")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--ckpt_siglip", default="./ckpts/siglip_vitl16_256/open_clip_pytorch_model.bin")
    p.add_argument("--ckpt_laion", default=None)
    p.add_argument("--ckpt_flip", default=None)
    return p.parse_args()


def main():
    args = parse_args()

    import torch
    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
    log.info(f"Models: {args.models}")

    local_paths = {
        "siglip": args.ckpt_siglip,
        "laion":  args.ckpt_laion,
        "flip":   args.ckpt_flip,
    }

    ts = int(time.time())
    out_dir = Path("results"); out_dir.mkdir(exist_ok=True)

    run_all = args.only is None or args.only == "all"

    # ── E1 + E2 ──
    e1e2_results = []
    if run_all or args.only == "e1e2":
        log.info("\n" + "=" * 60)
        log.info("  Running E1 + E2: Model × Prompt comparison")
        log.info("=" * 60)
        e1e2_results = run_e1_e2(
            args.models, device, local_paths, args.batch_size, args.gpu)

        # Save (strip y_true/y_pred which are large)
        out_file = out_dir / f"e1e2_{ts}.json"
        with open(out_file, "w") as f:
            json.dump(e1e2_results, f, indent=2, ensure_ascii=False)
        log.info(f"Saved: {out_file}")

        print_e1_table(e1e2_results)
        print_e2_table(e1e2_results)
        print_confusion_top(e1e2_results)

    # ── E4: CAPE ──
    cape_results = []
    if run_all or args.only == "cape":
        log.info("\n" + "=" * 60)
        log.info("  Running E4: CAPE (Class-Aware Prompt Ensemble)")
        log.info("=" * 60)
        cape_results = run_e4_cape(
            args.models, device, local_paths, args.batch_size, args.gpu)

        out_file = out_dir / f"e4_cape_{ts}.json"
        with open(out_file, "w") as f:
            json.dump(cape_results, f, indent=2, ensure_ascii=False)
        log.info(f"Saved: {out_file}")

    # ── Combined summary ──
    if e1e2_results and cape_results:
        print_cape_table(e1e2_results, cape_results)

        # Save combined
        combined = {
            "timestamp": ts,
            "models": args.models,
            "e1e2": e1e2_results,
            "e4_cape": cape_results,
        }
        out_file = out_dir / f"paper_all_{ts}.json"
        with open(out_file, "w") as f:
            json.dump(combined, f, indent=2, ensure_ascii=False)
        log.info(f"Saved combined: {out_file}")

    print("\n" + "=" * 50)
    print("  ALL EXPERIMENTS DONE")
    print("=" * 50)


if __name__ == "__main__":
    main()
