#!/usr/bin/env python3
"""Generate all figures for the SCB5 zero-shot paper revision.

Outputs:
  fig_confusion_matrix_teacher.pdf   — 5-model confusion matrices for TeacherBehavior (best prompt each)
  fig_confusion_matrix_handrise.pdf  — 5-model confusion matrices for HandriseReadWrite
  fig_confusion_matrix_bowturn.pdf   — 5-model confusion matrices for BowTurnHead
  fig_per_class_teacher.pdf          — Per-class Hit@1 bar chart for TeacherBehavior
  fig_prediction_distribution.pdf    — Prediction distribution vs ground-truth for TeacherBehavior
  fig_prompt_ablation_heatmap.pdf    — 5-model × 5-prompt heatmap per dataset
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────
JSON_PATH = Path(__file__).parent.parent / "results" / "parallel" / "benchmark_final_merged_1775830149.json"
OUT_DIR = Path(__file__).parent / "figures"
OUT_DIR.mkdir(exist_ok=True)

MODEL_DISPLAY = {
    "clip": "CLIP (OpenAI)",
    "laion": "OpenCLIP (LAION)",
    "siglip": "SigLIP2",
    "eva02": "EVA02-CLIP",
    "dfn": "DFN-CLIP",
}
MODEL_ORDER = ["clip", "laion", "siglip", "eva02", "dfn"]
PROMPT_ORDER = ["label_only", "simple", "action", "detailed", "cape"]
PROMPT_DISPLAY = {
    "label_only": "Label-only",
    "simple": "Simple",
    "action": "Action",
    "detailed": "Detailed",
    "cape": "CAPE",
}

DATASET_DISPLAY = {
    "SCB5_TeacherBehavior": "TeacherBehavior (8 classes)",
    "SCB5_HandriseReadWrite": "HandriseReadWrite (3 classes)",
    "SCB_BowTurnHead": "BowTurnHead (2 classes)",
}

# Shorter class names for TeacherBehavior
TB_SHORT = {
    "guide": "Guide",
    "answer": "Answer",
    "On-stage interaction": "On-stage",
    "blackboard-writing": "BB-write",
    "teacher": "Teacher",
    "stand": "Stand",
    "screen": "Screen",
    "blackBoard": "Blackboard",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})


def load_data():
    with open(JSON_PATH) as f:
        return json.load(f)


def get_experiment(ds_data, model, prompt):
    for exp in ds_data["experiments"]:
        if exp["model"] == model and exp["prompt"] == prompt:
            return exp
    return None


def best_prompt_for_model(ds_data, model):
    """Return the experiment with highest hit1 for a given model."""
    best = None
    for exp in ds_data["experiments"]:
        if exp["model"] == model:
            if best is None or exp["hit1"] > best["hit1"]:
                best = exp
    return best


# ── Figure 1: Confusion Matrices (5 models, best prompt) ──────
def plot_confusion_matrices(data, ds_key, filename):
    ds = data["datasets"][ds_key]
    class_names = ds["class_names"]
    n_classes = len(class_names)

    if ds_key == "SCB5_TeacherBehavior":
        short_names = [TB_SHORT.get(c, c) for c in class_names]
    else:
        short_names = class_names

    fig, axes = plt.subplots(1, 5, figsize=(3.2 * 5, 3.2), constrained_layout=True)

    for ax, model_key in zip(axes, MODEL_ORDER):
        exp = best_prompt_for_model(ds, model_key)
        cm = np.array(exp["confusion_matrix"], dtype=float)

        # Normalize by row (true label) to get recall per class
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_norm = cm / row_sums * 100

        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=100, aspect="equal")

        ax.set_xticks(range(n_classes))
        ax.set_yticks(range(n_classes))
        ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(short_names, fontsize=7)

        # Annotate cells
        for i in range(n_classes):
            for j in range(n_classes):
                val = cm_norm[i, j]
                color = "white" if val > 50 else "black"
                fontsize = 6 if n_classes > 4 else 8
                ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                        color=color, fontsize=fontsize)

        ax.set_title(f"{MODEL_DISPLAY[model_key]}\n({exp['prompt']}, Hit@1={exp['hit1']:.1f}%)",
                     fontsize=8)
        if ax == axes[0]:
            ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted")

    fig.suptitle(DATASET_DISPLAY.get(ds_key, ds_key), fontsize=12, y=1.02)
    outpath = OUT_DIR / filename
    fig.savefig(outpath, dpi=300)
    plt.close(fig)
    print(f"  Saved {outpath}")


# ── Figure 2: Per-class Hit@1 for TeacherBehavior ─────────────
def plot_per_class_hit1(data, filename="fig_per_class_teacher.pdf"):
    ds = data["datasets"]["SCB5_TeacherBehavior"]
    class_names = ds["class_names"]
    short_names = [TB_SHORT.get(c, c) for c in class_names]
    n_classes = len(class_names)

    # For each model (best prompt), compute per-class recall
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(n_classes)
    width = 0.15
    colors = plt.cm.Set2(np.linspace(0, 1, 5))

    for i, model_key in enumerate(MODEL_ORDER):
        exp = best_prompt_for_model(ds, model_key)
        cm = np.array(exp["confusion_matrix"], dtype=float)
        # Per-class recall (Hit@1 for that class)
        row_sums = cm.sum(axis=1)
        row_sums[row_sums == 0] = 1
        diag = np.diag(cm)
        recall = diag / row_sums * 100

        offset = (i - 2) * width
        bars = ax.bar(x + offset, recall, width, label=f"{MODEL_DISPLAY[model_key]} ({exp['prompt']})",
                      color=colors[i], edgecolor="gray", linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=30, ha="right")
    ax.set_ylabel("Per-Class Recall (%)")
    ax.set_title("Per-Class Recall on TeacherBehavior (Best Prompt per Model)")
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)

    outpath = OUT_DIR / filename
    fig.savefig(outpath, dpi=300)
    plt.close(fig)
    print(f"  Saved {outpath}")


# ── Figure 3: Prediction Distribution vs Ground Truth ─────────
def plot_prediction_distribution(data, filename="fig_prediction_distribution.pdf"):
    """Show how each model's predictions distribute across classes vs ground truth."""
    ds = data["datasets"]["SCB5_TeacherBehavior"]
    class_names = ds["class_names"]
    short_names = [TB_SHORT.get(c, c) for c in class_names]
    n_classes = len(class_names)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    axes = axes.flatten()

    # Ground truth distribution (from confusion matrix row sums)
    exp0 = best_prompt_for_model(ds, "clip")
    cm0 = np.array(exp0["confusion_matrix"], dtype=float)
    gt_dist = cm0.sum(axis=1)
    gt_pct = gt_dist / gt_dist.sum() * 100

    # Plot ground truth
    ax = axes[0]
    ax.barh(range(n_classes), gt_pct, color="gray", edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(n_classes))
    ax.set_yticklabels(short_names, fontsize=7)
    ax.set_xlabel("Percentage (%)")
    ax.set_title("Ground Truth\nDistribution", fontsize=9)
    ax.invert_yaxis()

    # Plot each model's prediction distribution
    for idx, model_key in enumerate(MODEL_ORDER):
        ax = axes[idx + 1]
        exp = best_prompt_for_model(ds, model_key)
        cm = np.array(exp["confusion_matrix"], dtype=float)
        pred_dist = cm.sum(axis=0)
        pred_pct = pred_dist / pred_dist.sum() * 100

        colors_bar = ["#e74c3c" if pred_pct[i] > gt_pct[i] * 1.5 else
                       "#3498db" if pred_pct[i] < gt_pct[i] * 0.5 else
                       "#2ecc71" for i in range(n_classes)]
        ax.barh(range(n_classes), pred_pct, color=colors_bar, edgecolor="black", linewidth=0.5)
        # Overlay ground truth as line
        ax.plot(gt_pct, range(n_classes), "k--", linewidth=1, alpha=0.5, label="GT")
        ax.set_yticks(range(n_classes))
        ax.set_yticklabels(short_names, fontsize=7)
        ax.set_xlabel("Percentage (%)")
        ax.set_title(f"{MODEL_DISPLAY[model_key]}\n({exp['prompt']})", fontsize=9)
        ax.invert_yaxis()
        if idx == 0:
            ax.legend(fontsize=7)

    fig.suptitle("Prediction Distribution vs. Ground Truth — TeacherBehavior", fontsize=12)
    outpath = OUT_DIR / filename
    fig.savefig(outpath, dpi=300)
    plt.close(fig)
    print(f"  Saved {outpath}")


# ── Figure 4: Prompt Ablation Heatmap ─────────────────────────
def plot_prompt_ablation_heatmap(data, filename="fig_prompt_ablation_heatmap.pdf"):
    datasets_to_plot = ["SCB5_TeacherBehavior", "SCB5_HandriseReadWrite", "SCB_BowTurnHead"]

    fig, axes = plt.subplots(1, 3, figsize=(14.8, 5.1), constrained_layout=False)

    for ax, ds_key in zip(axes, datasets_to_plot):
        ds = data["datasets"][ds_key]
        matrix = np.zeros((5, 5))
        for i, model_key in enumerate(MODEL_ORDER):
            for j, prompt_key in enumerate(PROMPT_ORDER):
                exp = get_experiment(ds, model_key, prompt_key)
                matrix[i, j] = exp["hit1"] if exp else 0

        im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto",
                       vmin=max(0, matrix.min() - 5), vmax=min(100, matrix.max() + 5))

        ax.set_xticks(range(5))
        ax.set_yticks(range(5))
        ax.set_xticklabels([PROMPT_DISPLAY[p] for p in PROMPT_ORDER], rotation=45, ha="right")
        ax.set_yticklabels([MODEL_DISPLAY[m] for m in MODEL_ORDER])

        for i in range(5):
            for j in range(5):
                val = matrix[i, j]
                color = "white" if val > (matrix.max() + matrix.min()) / 2 else "black"
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                        fontsize=7, fontweight="bold", color=color)

        ax.set_title(DATASET_DISPLAY.get(ds_key, ds_key), fontsize=10, pad=14)
        cbar = plt.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
        cbar.set_label("Hit@1 (%)")

    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.20, top=0.82, wspace=0.35)
    fig.suptitle("Prompt Ablation: Hit@1 (%) across Models and Prompt Strategies", fontsize=12, y=0.96)
    outpath = OUT_DIR / filename
    fig.savefig(outpath, dpi=300)
    plt.close(fig)
    print(f"  Saved {outpath}")


# ── Figure 5: CAPE Gain Visualization ─────────────────────────
def plot_cape_gain(data, filename="fig_cape_gain.pdf"):
    datasets_to_plot = ["SCB5_TeacherBehavior", "SCB5_HandriseReadWrite", "SCB_BowTurnHead"]
    ds_short = ["Teacher", "Handrise", "BowTurn"]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(MODEL_ORDER))
    width = 0.25
    colors = ["#3498db", "#e74c3c", "#2ecc71"]

    for idx, (ds_key, ds_label) in enumerate(zip(datasets_to_plot, ds_short)):
        ds = data["datasets"][ds_key]
        gains = []
        for model_key in MODEL_ORDER:
            cape_exp = get_experiment(ds, model_key, "cape")
            best_baseline = None
            for p in ["label_only", "simple", "action", "detailed"]:
                exp = get_experiment(ds, model_key, p)
                if exp and (best_baseline is None or exp["hit1"] > best_baseline["hit1"]):
                    best_baseline = exp
            gain = cape_exp["hit1"] - best_baseline["hit1"] if cape_exp and best_baseline else 0
            gains.append(gain)

        offset = (idx - 1) * width
        bars = ax.bar(x + offset, gains, width, label=ds_label, color=colors[idx],
                      edgecolor="gray", linewidth=0.5)
        # Add value labels
        for bar, g in zip(bars, gains):
            y = bar.get_height()
            va = "bottom" if y >= 0 else "top"
            ax.text(bar.get_x() + bar.get_width() / 2, y, f"{g:+.1f}",
                    ha="center", va=va, fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_DISPLAY[m] for m in MODEL_ORDER], rotation=15, ha="right")
    ax.set_ylabel("CAPE Gain (pp)")
    ax.set_title("CAPE Hit@1 Gain over Best Baseline Prompt")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    outpath = OUT_DIR / filename
    fig.savefig(outpath, dpi=300)
    plt.close(fig)
    print(f"  Saved {outpath}")


# ── Main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading benchmark data...")
    data = load_data()

    print("\n[1/4] Confusion matrices — TeacherBehavior")
    plot_confusion_matrices(data, "SCB5_TeacherBehavior", "fig_confusion_matrix_teacher.pdf")

    print("[2/4] Per-class recall — TeacherBehavior")
    plot_per_class_hit1(data)

    print("[3/4] Prediction distribution — TeacherBehavior")
    plot_prediction_distribution(data)

    print("[4/4] Prompt ablation heatmap")
    plot_prompt_ablation_heatmap(data)

    print("[Bonus] CAPE gain visualization")
    plot_cape_gain(data)

    print(f"\nAll figures saved to {OUT_DIR}/")
