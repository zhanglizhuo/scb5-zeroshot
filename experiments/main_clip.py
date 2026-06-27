"""
main_clip.py
复现论文原始 CLIP + CAPE 基线实验（用于环境验证和结果对照）
"""

import os
import json
import argparse
import sys
import yaml
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.scb_dataset import build_dataloader, SUBSET_CONFIG, SUBSET_LOCAL_DIRS
from models.clip_zoo import CLIPModel, build_cape_text_features, batch_inference
from prompts.cape_prompts import get_all_class_prompts, ALL_STRATEGIES
from evaluation.metrics import full_evaluation

GPU_ASSIGNMENT = {
    "openai": "cuda:0",
    "siglip2": "cuda:1",
    "laion": "cuda:2",
    "eva02": "cuda:3",
    "dfn": "cuda:0",
}

MODEL_KEYS = ["openai", "laion", "siglip2", "eva02", "dfn"]

# Allow overriding which models to run via environment variable SINGLE_MODEL
env_single = os.environ.get("SINGLE_MODEL")
if env_single:
    MODEL_KEYS = [env_single]


def run_single(model_key, subset, strategy, cape_set="A", device=None):
    device = device or GPU_ASSIGNMENT.get(model_key, "cuda:0")
    # If the process is launched with CUDA_VISIBLE_DEVICES (pinned to one or more GPUs),
    # use the local ordinal `cuda:0` so the process refers to the first visible device.
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        device = "cuda:0"
    cfg = SUBSET_CONFIG[subset]
    classes = cfg["classes"]
    multilabel = cfg["multilabel"]
    num_classes = cfg["num_classes"]

    model = CLIPModel(model_key, device=device)
    data_dir = SUBSET_LOCAL_DIRS.get(subset)
    loader, dataset = build_dataloader(
        subset=subset, split="validation",
        batch_size=64, num_workers=0, input_size=model.input_size,
        data_dir=str(data_dir) if data_dir else None,
    )

    class_prompts = get_all_class_prompts(classes, strategy, cape_set=cape_set)
    text_feats = build_cape_text_features(model, class_prompts, classes)
    logits, labels = batch_inference(model, loader, text_feats, multilabel)

    if multilabel:
        lb_matrix = labels.astype(float)
    else:
        lb_matrix = np.eye(num_classes)[labels.astype(int)]

    eval_results = full_evaluation(
        logits, lb_matrix, num_classes, multilabel, classes,
        run_bootstrap=False,
    )
    del model
    torch.cuda.empty_cache()
    return eval_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/experiment_config.yaml")
    parser.add_argument("--results_dir", type=str, default="./results/baseline")
    parser.add_argument("--experiment", type=str, default="all",
                        choices=["all", "cape_only", "all_strategies"])
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = REPO_ROOT / results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    subsets = ["teacher_behavior", "handrise_readwrite", "bow_turnhead"]
    strategies = ALL_STRATEGIES if args.experiment == "all_strategies" else ["cape"]

    all_results = {}
    for subset in subsets:
        all_results[subset] = {}
        for model_key in MODEL_KEYS:
            all_results[subset][model_key] = {}
            for strategy in strategies:
                print(f"  {model_key} / {subset} / {strategy}")
                result = run_single(model_key, subset, strategy)
                all_results[subset][model_key][strategy] = {
                    "hit_at_1": result["hit_at_1"],
                    "hit_at_3": result["hit_at_3"],
                    "macro_f1": result["single_label"]["macro_f1"],
                }
                print(f"    Hit@1={result['hit_at_1']*100:.2f}%  "
                      f"Macro-F1={result['single_label']['macro_f1']*100:.2f}%")

    out_path = results_dir / "baseline_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Baseline complete: {out_path}")


if __name__ == "__main__":
    main()
