"""
main_mllm.py
MLLM 零样本对比实验
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
对比 LLaVA-1.6 vs Qwen-VL vs CLIP+CAPE vs TFVA
分析 MLLM 与 CLIP 在教育场景的能力边界

4× V100 分配：
  GPU0: LLaVA-1.6-7B (int4)
  GPU1: Qwen-VL-Chat (int8)
  GPU2,3: CLIP models（已在 main_tfva.py 中计算，复用缓存）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.scb_dataset import build_dataloader, SUBSET_CONFIG
from models.mllm_baseline import (
    LLaVAInferencer,
    QwenVLInferencer,
    OllamaVisionInferencer,
    mllm_preds_to_logits,
)
from evaluation.metrics import full_evaluation, compute_hit_at_k


# ─── 数据集原始图像恢复 ──────────────────────────────────────────────────────

def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    从 CLIP 归一化 tensor 还原 PIL Image（近似，用于 MLLM 输入）
    实际使用时建议在 Dataset 中同时保存原始 PIL
    """
    CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
    CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
    img = tensor.cpu() * CLIP_STD + CLIP_MEAN
    img = (img.clamp(0, 1) * 255).byte()
    return Image.fromarray(img.permute(1, 2, 0).numpy())


# ─── MLLM 推理主函数 ──────────────────────────────────────────────────────────

def _compute_eval_metrics(
    preds: List[Dict],
    labels: List,
    num_classes: int,
    multilabel: bool,
    classes: List[str],
) -> Dict:
    """从 MLLM 原始预测恢复统一评估指标（含 Macro-F1）"""
    logits = mllm_preds_to_logits(preds, num_classes, multilabel)
    labels_np = np.array(labels)
    eval_result = full_evaluation(
        logits=logits,
        labels_multilabel=labels_np,
        num_classes=num_classes,
        multilabel=multilabel,
        class_names=classes,
        run_bootstrap=False,
    )

    out = {
        "hit_at_1": float(eval_result["hit_at_1"]),
        "hit_at_3": float(eval_result["hit_at_3"]),
        "macro_f1": float(eval_result["single_label"]["macro_f1"]),
        "micro_f1": float(eval_result["single_label"]["micro_f1"]),
        "accuracy": float(eval_result["single_label"]["accuracy"]),
    }
    if multilabel and "multilabel" in eval_result:
        out["sample_f1"] = float(eval_result["multilabel"]["sample_f1"])
        out["multilabel_macro_f1"] = float(eval_result["multilabel"]["macro_f1"])
    return out

def run_mllm_inference(
    inferencer,
    model_name: str,
    subset: str,
    max_samples: int = None,
    results_dir: Path = None,
    start_idx: int = None,
    end_idx: int = None,
) -> Dict:
    """对指定子集运行 MLLM 推理"""
    cfg = SUBSET_CONFIG[subset]
    classes = cfg["classes"]
    multilabel = cfg["multilabel"]
    num_classes = cfg["num_classes"]

    slice_suffix = ""
    if start_idx is not None or end_idx is not None:
        s = 0 if start_idx is None else int(start_idx)
        e = "end" if end_idx is None else str(int(end_idx))
        slice_suffix = f"_s{s}_e{e}"

    cache_path = results_dir / f"{model_name}_{subset}{slice_suffix}_preds.json"
    if cache_path.exists():
        print(f"  Loading cached predictions: {cache_path}")
        with open(cache_path) as f:
            cached = json.load(f)

        # 老缓存里可能没有 Macro-F1，命中缓存时补算并回写
        if "macro_f1" not in cached and "raw_preds" in cached and "labels" in cached:
            metrics = _compute_eval_metrics(
                preds=cached["raw_preds"],
                labels=cached["labels"],
                num_classes=num_classes,
                multilabel=multilabel,
                classes=classes,
            )
            cached.update(metrics)
            with open(cache_path, "w") as f:
                json.dump(cached, f, indent=2)
        return cached

    loader, dataset = build_dataloader(
        subset=subset,
        split="validation",
        batch_size=1,    # MLLM 一次处理一张
        num_workers=2,
    )

    all_preds = []
    all_labels = []
    count = 0

    for sample_idx, batch in enumerate(tqdm(loader, desc=f"{model_name}/{subset}")):
        if start_idx is not None and sample_idx < start_idx:
            continue
        if end_idx is not None and sample_idx >= end_idx:
            break
        if max_samples and count >= max_samples:
            break

        img = tensor_to_pil(batch["image"][0])
        label = batch["label"][0].numpy()

        pred = inferencer.infer_single(img, classes, multilabel)
        all_preds.append(pred)
        all_labels.append(label.tolist())
        count += 1

    # 计算指标
    metrics = _compute_eval_metrics(
        preds=all_preds,
        labels=all_labels,
        num_classes=num_classes,
        multilabel=multilabel,
        classes=classes,
    )
    results = {
        "model": model_name,
        "subset": subset,
        **metrics,
        "num_samples": count,
        "start_idx": start_idx,
        "end_idx": end_idx,
        "raw_preds": all_preds,
        "labels": all_labels,
    }

    # 保存
    if results_dir:
        with open(cache_path, "w") as f:
            json.dump(results, f, indent=2)

    return results


# ─── 能力边界分析 ────────────────────────────────────────────────────────────

def analyze_capability_gap(
    clip_results: Dict,
    mllm_results: Dict,
    classes: List[str],
) -> Dict:
    """
    分析 CLIP vs MLLM 的能力差异
    找出 MLLM 显著优于/劣于 CLIP 的样本
    """
    analysis = {
        "categories_where_mllm_wins": [],
        "categories_where_clip_wins": [],
        "overall_gap": {},
    }

    for subset in clip_results:
        clip_hit1 = clip_results[subset].get("hit_at_1", 0)
        mllm_hit1 = mllm_results.get(subset, {}).get("hit_at_1", 0)
        gap = mllm_hit1 - clip_hit1
        analysis["overall_gap"][subset] = {
            "clip_hit1": clip_hit1,
            "mllm_hit1": mllm_hit1,
            "gap": gap,
            "mllm_better": gap > 0,
        }

    return analysis


# ─── 主函数 ──────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="MLLM Comparison Experiments")
    parser.add_argument("--results_dir", type=str, default="./results/mllm")
    parser.add_argument("--models", nargs="+",
                        default=["qwen35_ollama", "gemma4_ollama"],
                        choices=["llava", "qwenvl", "qwen35_ollama", "gemma4_ollama", "gpt4v"])
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit samples per subset for debugging (None=all)")
    parser.add_argument("--subsets", nargs="+",
                        default=["teacher_behavior", "handrise_readwrite", "bow_turnhead"],
                        choices=["teacher_behavior", "handrise_readwrite", "bow_turnhead"],
                        help="Subsets to run (use to split jobs for parallel multi-GPU execution)")
    parser.add_argument("--llava_device", type=str, default="cuda:0")
    parser.add_argument("--qwenvl_device", type=str, default="cuda:1")
    parser.add_argument("--ollama_host", type=str, default="http://127.0.0.1:11434")
    parser.add_argument("--qwen35_model", type=str, default="qwen3.5:9b")
    parser.add_argument("--gemma4_model", type=str, default="gemma4:e4b")
    parser.add_argument("--openai_api_key", type=str, default=None)
    parser.add_argument("--start_idx", type=int, default=None,
                        help="Start index (inclusive) for dataset sharding")
    parser.add_argument("--end_idx", type=int, default=None,
                        help="End index (exclusive) for dataset sharding")
    return parser.parse_args()


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = REPO_ROOT / results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    subsets = args.subsets
    all_results = {}

    # ── LLaVA 推理 ──────────────────────────────────────────────────────────
    if "llava" in args.models:
        print("\n" + "="*60)
        print("Loading LLaVA-1.6-Mistral-7B (int4 quantized)")
        print("="*60)
        llava = LLaVAInferencer(device=args.llava_device, quantization="int4")
        all_results["llava"] = {}

        for subset in subsets:
            print(f"\nSubset: {subset}")
            result = run_mllm_inference(
                llava, "llava", subset,
                max_samples=args.max_samples,
                results_dir=results_dir,
            )
            all_results["llava"][subset] = result
            print(f"  Hit@1={result['hit_at_1']:.4f}  Macro-F1={result['macro_f1']:.4f}")

        del llava
        torch.cuda.empty_cache()

    # ── Qwen-VL 推理 ─────────────────────────────────────────────────────────
    if "qwenvl" in args.models:
        print("\n" + "="*60)
        print("Loading Qwen-VL-Chat (int8 quantized)")
        print("="*60)
        qwen = QwenVLInferencer(device=args.qwenvl_device, quantization="int8")
        all_results["qwenvl"] = {}

        for subset in subsets:
            print(f"\nSubset: {subset}")
            result = run_mllm_inference(
                qwen, "qwenvl", subset,
                max_samples=args.max_samples,
                results_dir=results_dir,
            )
            all_results["qwenvl"][subset] = result
            print(f"  Hit@1={result['hit_at_1']:.4f}  Macro-F1={result['macro_f1']:.4f}")

        del qwen
        torch.cuda.empty_cache()

    # ── Ollama Qwen3.5-Vision 推理 ────────────────────────────────────────
    if "qwen35_ollama" in args.models:
        print("\n" + "="*60)
        print(f"Loading Ollama model: {args.qwen35_model}")
        print("="*60)
        qwen_ollama = OllamaVisionInferencer(
            model_id=args.qwen35_model,
            host=args.ollama_host,
        )
        all_results["qwen35_ollama"] = {}

        for subset in subsets:
            print(f"\nSubset: {subset}")
            result = run_mllm_inference(
                qwen_ollama, "qwen35_ollama", subset,
                max_samples=args.max_samples,
                results_dir=results_dir,
                start_idx=args.start_idx,
                end_idx=args.end_idx,
            )
            all_results["qwen35_ollama"][subset] = result
            print(f"  Hit@1={result['hit_at_1']:.4f}  Macro-F1={result['macro_f1']:.4f}")

    # ── Ollama Gemma4-Vision 推理 ────────────────────────────────────────
    if "gemma4_ollama" in args.models:
        print("\n" + "="*60)
        print(f"Loading Ollama model: {args.gemma4_model}")
        print("="*60)
        gemma_ollama = OllamaVisionInferencer(
            model_id=args.gemma4_model,
            host=args.ollama_host,
        )
        all_results["gemma4_ollama"] = {}

        for subset in subsets:
            print(f"\nSubset: {subset}")
            result = run_mllm_inference(
                gemma_ollama, "gemma4_ollama", subset,
                max_samples=args.max_samples,
                results_dir=results_dir,
                start_idx=args.start_idx,
                end_idx=args.end_idx,
            )
            all_results["gemma4_ollama"][subset] = result
            print(f"  Hit@1={result['hit_at_1']:.4f}  Macro-F1={result['macro_f1']:.4f}")

    # ── GPT-4V（可选）────────────────────────────────────────────────────────
    if "gpt4v" in args.models and args.openai_api_key:
        from models.mllm_baseline import GPT4VInferencer
        print("\nRunning GPT-4V (API)...")
        gpt4v = GPT4VInferencer(args.openai_api_key)
        all_results["gpt4v"] = {}
        # GPT-4V 费用较高，建议每个子集限制样本数
        for subset in subsets:
            result = run_mllm_inference(
                gpt4v, "gpt4v", subset,
                max_samples=min(args.max_samples or 200, 200),
                results_dir=results_dir,
            )
            all_results["gpt4v"][subset] = result

    # ── 汇总表格 ─────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("MLLM Comparison Summary (Hit@1 % / Macro-F1 %)")
    print("="*60)
    subset_headers = {
        "teacher_behavior": "TeacherBeh",
        "handrise_readwrite": "HandriseRW",
        "bow_turnhead": "BowTurnHd",
    }
    header = f"{'Model':<15}" + "".join(f" {subset_headers[s]:>12}" for s in subsets)
    print(header)
    print("-" * len(header))

    for model_name, model_results in all_results.items():
        row = [model_name]
        for subset in subsets:
            hit1 = model_results.get(subset, {}).get("hit_at_1", 0)
            macro = model_results.get(subset, {}).get("macro_f1", 0)
            row.append(f"{hit1*100:.2f}/{macro*100:.2f}")
        print(f"{row[0]:<15}" + "".join(f" {v:>12}" for v in row[1:]))

    # 保存汇总
    summary = {
        k: {
            s: {
                "hit_at_1": v.get("hit_at_1", 0),
                "hit_at_3": v.get("hit_at_3", 0),
                "macro_f1": v.get("macro_f1", 0),
                "micro_f1": v.get("micro_f1", 0),
                "accuracy": v.get("accuracy", 0),
                "num_samples": v.get("num_samples", 0),
            }
            for s, v in model_results.items()
        }
        for k, model_results in all_results.items()
    }

    out_path = results_dir / "mllm_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n✓ MLLM experiments complete. Summary: {out_path}")


if __name__ == "__main__":
    main()
