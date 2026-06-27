"""
汇总所有模型的结果，输出对比表格
python scripts/summarize_results.py
"""
import json
import os
from pathlib import Path

results_dir = Path("results")
files = sorted(results_dir.glob("*.json"))

if not files:
    print("No results found. Run experiments first.")
    exit()

# 按模型名聚合最新结果
latest = {}
for f in files:
    parts = f.stem.split("_")
    model = parts[0]
    if model not in latest or f.stem > latest[model].stem:
        latest[model] = f

print("\n" + "=" * 70)
print("  SCB5 Teacher Behavior - Model Comparison")
print("=" * 70)

# Zero-shot 汇总
print("\n[Zero-shot Top-1 Accuracy]")
print(f"  {'Model':10s}  {'Top-1':>8}  {'Top-3':>8}")
print("  " + "-" * 32)
for model, fpath in sorted(latest.items()):
    with open(fpath) as f:
        res = json.load(f)
    if "zeroshot" in res:
        zs = res["zeroshot"]
        print(f"  {model:10s}  {zs['top1_accuracy']:>7.2f}%  {zs['top3_accuracy']:>7.2f}%")

# Per-class 对比
print("\n[Per-class Recall @ Top-1]")
header = f"  {'Class':30s}"
models_with_zs = []
for model, fpath in sorted(latest.items()):
    with open(fpath) as f:
        res = json.load(f)
    if "zeroshot" in res:
        header += f"  {model:>8}"
        models_with_zs.append((model, res))

print(header)
print("  " + "-" * (30 + len(models_with_zs) * 10 + 4))

from collections import defaultdict
classes = ['guide','answer','On-stage interaction','blackboard-writing',
           'teacher','stand','screen','blackBoard']

for cls in classes:
    row = f"  {cls:30s}"
    for model, res in models_with_zs:
        val = res["zeroshot"]["per_class_recall"].get(cls)
        if val is not None:
            row += f"  {val:>7.1f}%"
        else:
            row += f"  {'N/A':>7}"
    print(row)

# Few-shot 汇总
print("\n[Few-shot Val Accuracy]")
has_fewshot = False
for model, fpath in sorted(latest.items()):
    with open(fpath) as f:
        res = json.load(f)
    if "fewshot" in res:
        if not has_fewshot:
            shots = sorted(int(k) for k in res["fewshot"].keys())
            print(f"  {'Model':10s}  " + "  ".join(f"{k:>6}-shot" for k in shots))
            print("  " + "-" * (10 + len(shots) * 12))
            has_fewshot = True
        row = f"  {model:10s}  "
        row += "  ".join(
            f"{res['fewshot'].get(str(k), res['fewshot'].get(k, 0)):>9.2f}%"
            for k in shots
        )
        print(row)

print("\n" + "=" * 70)
