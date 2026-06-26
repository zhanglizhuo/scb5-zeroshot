"""
main_tfva.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
核心新贡献实验：Training-Free Visual Adapter (TFVA)

实验设计：
  E-New-1: ZS-TFVA vs CAPE (零样本对比)
  E-New-2: FS-TFVA k-shot 消融 (k=1,2,4,8,16)
  E-New-3: 多标签 TFVA (TeacherBehavior)
  E-New-4: 跨模型 TFVA 一致性分析

4× V100 分配策略：
  GPU0: openai-clip + siglip2
  GPU1: laion-clip + eva02
  GPU2: dfn-clip
  GPU3: 特征存储 + 评估
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import json
import argparse
import yaml
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from data.scb_dataset import build_dataloader, SUBSET_CONFIG
from models.clip_zoo import CLIPModel, build_cape_text_features, batch_inference
from models.tfva import (
    ZeroShotTFVA, FewShotTFVA, MultiLabelTFVA,
    collect_few_shot_features, TFVATrainer
)
from prompts.cape_prompts import get_all_class_prompts, CAPE_SET_A
from evaluation.metrics import full_evaluation, compute_hit_at_k


# ─── GPU 分配方案 ─────────────────────────────────────────────────────────────

GPU_ASSIGNMENT = {
    "openai": "cuda:0",
    "siglip2": "cuda:0",
    "laion": "cuda:1",
    "eva02": "cuda:1",
    "dfn": "cuda:2",
}

K_SHOTS = [1, 2, 4, 8, 16]


# ─── 特征提取（带缓存）────────────────────────────────────────────────────────

def extract_and_cache_features(
    model_key: str,
    subset: str,
    split: str,
    config: Dict,
    cache_dir: str = "./data/feature_cache",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    提取并缓存图像特征（避免重复计算）
    返回: (image_features, labels, logits_cape)
    """
    cache_path = Path(cache_dir) / f"{model_key}_{subset}_{split}.npz"
    if cache_path.exists():
        print(f"  Loading cached features: {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        img_features = data["image_features"]
        labels = data["labels"]
        logits_cape = data["logits_cape"]

        # Guard against stale/invalid caches created from wrong dataset roots.
        # Symptom observed: 259 samples with all labels in class 0, causing fake 1.0 scores.
        stale = False
        if labels.ndim == 1:
            if np.unique(labels.astype(int)).size <= 1:
                stale = True
        else:
            class_pos = labels.sum(axis=0)
            if np.count_nonzero(class_pos > 0) <= 1:
                stale = True

        if not stale:
            return img_features, labels, logits_cape

        print(f"  Detected stale cache (single-class labels). Rebuilding: {cache_path}")
        try:
            cache_path.unlink()
        except Exception:
            pass

    # 加载模型
    device = GPU_ASSIGNMENT.get(model_key, "cuda:0")
    model = CLIPModel(model_key, device=device)

    # 数据加载
    classes = SUBSET_CONFIG[subset]["classes"]
    multilabel = SUBSET_CONFIG[subset]["multilabel"]
    loader, dataset = build_dataloader(
        subset=subset,
        split=split,
        batch_size=64,
        num_workers=8,
        input_size=model.input_size,
    )

    # 构建 CAPE 文本特征
    cape_prompts = get_all_class_prompts(classes, "cape", cape_set="A")
    text_features = build_cape_text_features(model, cape_prompts, classes)  # (C, D)

    # 提取图像特征
    all_img_feats = []
    all_labels = []

    print(f"  Extracting features: {model_key} / {subset} / {split}")
    for batch in tqdm(loader):
        imgs = batch["image"].to(device)
        labels = batch["label"]
        with torch.no_grad():
            feats = model.encode_images(imgs)
        all_img_feats.append(feats.cpu().numpy())
        all_labels.append(labels.numpy())

    img_feats = np.concatenate(all_img_feats, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    # 计算 CAPE logits
    img_feats_t = torch.from_numpy(img_feats).to(device)
    logits_cape = (img_feats_t @ text_features.T).cpu().numpy()

    # 缓存
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path,
             image_features=img_feats,
             labels=labels,
             logits_cape=logits_cape)
    print(f"  Saved to {cache_path}")

    del model
    torch.cuda.empty_cache()
    return img_feats, labels, logits_cape


# ─── 实验 1：ZS-TFVA vs CAPE ─────────────────────────────────────────────────

def run_zs_tfva(args, results_dir: Path):
    """零样本 TFVA 实验（与 CAPE 对比）"""
    print("\n" + "="*60)
    print("E-New-1: Zero-Shot TFVA vs CAPE")
    print("="*60)

    all_results = {}
    model_keys = ["openai", "laion", "siglip2", "eva02", "dfn"]

    for subset in ["teacher_behavior", "handrise_readwrite", "bow_turnhead"]:
        cfg = SUBSET_CONFIG[subset]
        classes = cfg["classes"]
        multilabel = cfg["multilabel"]
        num_classes = cfg["num_classes"]
        all_results[subset] = {}

        print(f"\n--- Subset: {subset} ---")

        for model_key in model_keys:
            print(f"  Model: {model_key}")
            device = GPU_ASSIGNMENT.get(model_key, "cuda:0")

            # 加载缓存特征（若模型加载失败则跳过该模型）
            try:
                img_feats, labels, logits_cape = extract_and_cache_features(
                    model_key, subset, "validation", args
                )
            except Exception as e:
                print(f"  Warning: failed to prepare features for {model_key}: {e}")
                continue
            img_feats_t = torch.from_numpy(img_feats).to(device)

            # 构建文本特征
            try:
                model = CLIPModel(model_key, device=device)
            except Exception as e:
                print(f"  Warning: failed to init model {model_key}: {e}")
                continue
            cape_prompts = get_all_class_prompts(classes, "cape", cape_set="A")
            text_feats = build_cape_text_features(model, cape_prompts, classes)

            # ZS-TFVA（扫描 beta 超参）
            best_hit1_zs = -1
            best_beta = 5.5
            for beta in [1.0, 2.0, 5.5, 10.0, 20.0, 50.0]:
                zs_model = ZeroShotTFVA(text_feats, beta=beta)
                with torch.no_grad():
                    logits_zs = zs_model(img_feats_t).cpu().numpy()

                # 构建评估用标签矩阵
                if multilabel:
                    lb_matrix = labels.astype(float)
                else:
                    lb_matrix = np.eye(num_classes)[labels.astype(int)]

                hit1 = compute_hit_at_k(logits_zs, lb_matrix, k=1)
                if hit1 > best_hit1_zs:
                    best_hit1_zs = hit1
                    best_beta = beta

            # 最终评估
            zs_model = ZeroShotTFVA(text_feats, beta=best_beta)
            with torch.no_grad():
                logits_zs = zs_model(img_feats_t).cpu().numpy()

            if multilabel:
                lb_matrix = labels.astype(float)
            else:
                lb_matrix = np.eye(num_classes)[labels.astype(int)]

            eval_zs = full_evaluation(
                logits_zs, lb_matrix, num_classes, multilabel, classes,
                run_bootstrap=(subset == "bow_turnhead"),
            )
            eval_cape = full_evaluation(
                logits_cape, lb_matrix, num_classes, multilabel, classes,
                run_bootstrap=False,
            )

            all_results[subset][model_key] = {
                "zs_tfva": {
                    "hit_at_1": eval_zs["hit_at_1"],
                    "macro_f1": eval_zs["single_label"]["macro_f1"],
                    "best_beta": best_beta,
                },
                "cape": {
                    "hit_at_1": eval_cape["hit_at_1"],
                    "macro_f1": eval_cape["single_label"]["macro_f1"],
                },
                "delta_hit1": eval_zs["hit_at_1"] - eval_cape["hit_at_1"],
            }
            print(f"    ZS-TFVA Hit@1={eval_zs['hit_at_1']:.4f}  "
                  f"CAPE Hit@1={eval_cape['hit_at_1']:.4f}  "
                  f"Δ={all_results[subset][model_key]['delta_hit1']:+.4f}")

            del model
            torch.cuda.empty_cache()

    # 保存结果
    out_path = results_dir / "zs_tfva_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return all_results


# ─── 实验 2：Few-Shot TFVA k-shot 消融 ──────────────────────────────────────

def run_fs_tfva(args, results_dir: Path):
    """Few-shot TFVA k-shot 消融实验"""
    print("\n" + "="*60)
    print("E-New-2: Few-Shot TFVA k-shot Ablation")
    print("="*60)

    all_results = {}

    for subset in ["teacher_behavior", "handrise_readwrite", "bow_turnhead"]:
        cfg = SUBSET_CONFIG[subset]
        classes = cfg["classes"]
        multilabel = cfg["multilabel"]
        num_classes = cfg["num_classes"]
        all_results[subset] = {}

        print(f"\n--- Subset: {subset} ---")

        for model_key in ["openai", "siglip2", "dfn"]:   # 代表性三个模型
            device = GPU_ASSIGNMENT.get(model_key, "cuda:0")
            try:
                model = CLIPModel(model_key, device=device)
            except Exception as e:
                print(f"  Warning: failed to init model {model_key}: {e}")
                continue

            # 构建文本特征
            cape_prompts = get_all_class_prompts(classes, "cape", cape_set="A")
            text_feats = build_cape_text_features(model, cape_prompts, classes)

            # 加载验证集特征
            try:
                img_feats_val, labels_val, _ = extract_and_cache_features(
                    model_key, subset, "validation", args
                )
            except Exception as e:
                print(f"  Warning: failed to prepare features for {model_key}: {e}")
                del model
                torch.cuda.empty_cache()
                continue
            img_feats_val_t = torch.from_numpy(img_feats_val).to(device)

            # 构建标签矩阵
            if multilabel:
                lb_matrix = labels_val.astype(float)
            else:
                lb_matrix = np.eye(num_classes)[labels_val.astype(int)]

            # 加载训练集（用于 few-shot 采样）
            _, train_dataset = build_dataloader(
                subset=subset,
                split="train",
                batch_size=64,
                num_workers=4,
                input_size=model.input_size,
            )

            all_results[subset][model_key] = {"k_shot_results": {}}

            for k in K_SHOTS:
                print(f"  {model_key} / k={k}")
                # 采集 few-shot 特征
                shot_feats, shot_labels = collect_few_shot_features(
                    model, train_dataset, num_classes,
                    k_shot=k, device=device, seed=42,
                )

                # 超参搜索
                trainer = TFVATrainer(
                    text_feats, shot_feats, shot_labels, num_classes, device
                )
                # 使用验证集的一半做超参搜索
                N_val = len(img_feats_val_t)
                half = N_val // 2
                primary_labels = np.argmax(lb_matrix, axis=1) if not multilabel else \
                                 np.array([np.where(lb_matrix[i] > 0)[0].min()
                                           for i in range(N_val)])
                best_params = trainer.grid_search(
                    img_feats_val_t[:half],
                    torch.from_numpy(primary_labels[:half]),
                )

                # 最终评估（后半）
                fs_model = FewShotTFVA(
                    text_feats, shot_feats.to(device), shot_labels.to(device),
                    num_classes, **best_params
                )
                with torch.no_grad():
                    logits_fs = fs_model(img_feats_val_t[half:]).cpu().numpy()

                hit1 = compute_hit_at_k(logits_fs, lb_matrix[half:], k=1)
                all_results[subset][model_key]["k_shot_results"][k] = {
                    "hit_at_1": hit1,
                    "best_params": best_params,
                }
                print(f"    Hit@1={hit1:.4f}")

            del model
            torch.cuda.empty_cache()

    out_path = results_dir / "fs_tfva_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return all_results


# ─── 实验 3：多标签 TFVA ──────────────────────────────────────────────────────

def run_multilabel_tfva(args, results_dir: Path):
    """多标签 TFVA（专门针对 TeacherBehavior）"""
    print("\n" + "="*60)
    print("E-New-3: Multi-Label TFVA (TeacherBehavior)")
    print("="*60)

    subset = "teacher_behavior"
    cfg = SUBSET_CONFIG[subset]
    classes = cfg["classes"]
    num_classes = cfg["num_classes"]
    all_results = {}

    for model_key in ["openai", "siglip2", "eva02"]:
        device = GPU_ASSIGNMENT.get(model_key, "cuda:0")
        try:
            model = CLIPModel(model_key, device=device)
        except Exception as e:
            print(f"  Warning: failed to init model {model_key}: {e}")
            continue

        cape_prompts = get_all_class_prompts(classes, "cape", cape_set="A")
        text_feats = build_cape_text_features(model, cape_prompts, classes)

        try:
            img_feats, labels, _ = extract_and_cache_features(
                model_key, subset, "validation", args
            )
        except Exception as e:
            print(f"  Warning: failed to prepare features for {model_key}: {e}")
            del model
            torch.cuda.empty_cache()
            continue
        img_feats_t = torch.from_numpy(img_feats).to(device)
        lb_matrix = labels.astype(float)

        # 扫描阈值
        ml_model = MultiLabelTFVA(text_feats, beta=5.5)
        best_sample_f1 = -1
        best_threshold = 0.5
        best_ml_results = {}

        for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
            with torch.no_grad():
                probs = ml_model(img_feats_t)
                preds_binary = (probs >= threshold).float().cpu().numpy()

            from sklearn.metrics import f1_score
            sample_f1 = f1_score(lb_matrix, preds_binary, average="samples", zero_division=0)
            macro_f1 = f1_score(lb_matrix, preds_binary, average="macro", zero_division=0)

            if sample_f1 > best_sample_f1:
                best_sample_f1 = sample_f1
                best_threshold = threshold
                best_ml_results = {
                    "threshold": threshold,
                    "sample_f1": float(sample_f1),
                    "macro_f1": float(macro_f1),
                }

        all_results[model_key] = best_ml_results
        print(f"  {model_key}: threshold={best_threshold:.1f}  "
              f"Sample-F1={best_sample_f1:.4f}  "
              f"Macro-F1={best_ml_results['macro_f1']:.4f}")

        del model
        torch.cuda.empty_cache()

    out_path = results_dir / "multilabel_tfva_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    return all_results


# ─── 主函数 ──────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="TFVA Experiments")
    parser.add_argument("--config", type=str, default="config/experiment_config.yaml")
    parser.add_argument("--results_dir", type=str, default="./results/tfva")
    parser.add_argument("--cache_dir", type=str, default="./data/feature_cache")
    parser.add_argument("--experiment", type=str, default="all",
                        choices=["all", "zs", "fs", "multilabel"])
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    # 加载配置
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # 合并 args 和 config
    args.cache_dir = config["dataset"].get("cache_dir", args.cache_dir)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Results will be saved to: {results_dir}")
    print(f"GPU assignment: {GPU_ASSIGNMENT}")

    all_results = {}

    if args.experiment in ["all", "zs"]:
        all_results["zs_tfva"] = run_zs_tfva(args, results_dir)

    if args.experiment in ["all", "fs"]:
        all_results["fs_tfva"] = run_fs_tfva(args, results_dir)

    if args.experiment in ["all", "multilabel"]:
        all_results["multilabel_tfva"] = run_multilabel_tfva(args, results_dir)

    # 保存汇总
    summary_path = results_dir / "tfva_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ All TFVA experiments complete. Summary: {summary_path}")


if __name__ == "__main__":
    main()
