"""
evaluation/metrics.py
评估指标：Hit@1/3, Macro-F1, Sample-F1, Per-class recall, Bootstrap CI
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    f1_score, recall_score, precision_score,
    confusion_matrix,
)


# ─── 单标签指标 ───────────────────────────────────────────────────────────────

def compute_single_label_metrics(
    preds: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    class_names: Optional[List[str]] = None,
) -> Dict:
    acc = (preds == labels).mean()
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    micro_f1 = f1_score(labels, preds, average="micro", zero_division=0)
    per_class_recall = recall_score(labels, preds, average=None,
                                    labels=list(range(num_classes)), zero_division=0)
    per_class_f1 = f1_score(labels, preds, average=None,
                             labels=list(range(num_classes)), zero_division=0)
    cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))

    result = {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "per_class_recall": per_class_recall.tolist(),
        "per_class_f1": per_class_f1.tolist(),
        "confusion_matrix": cm.tolist(),
    }
    if class_names:
        result["per_class_recall_named"] = dict(zip(class_names, per_class_recall))
        result["per_class_f1_named"] = dict(zip(class_names, per_class_f1))
    return result


# ─── 多标签 Hit@k 指标 ───────────────────────────────────────────────────────

def compute_hit_at_k(
    logits: np.ndarray,
    labels_multilabel: np.ndarray,
    k: int = 1,
) -> float:
    N = logits.shape[0]
    topk_indices = np.argsort(logits, axis=1)[:, -k:]
    hits = 0
    for i in range(N):
        pred_set = set(topk_indices[i].tolist())
        true_set = set(np.where(labels_multilabel[i] > 0)[0].tolist())
        if len(pred_set & true_set) > 0:
            hits += 1
    return hits / N


# ─── 多标签正式指标（阈值方式）───────────────────────────────────────────────

def compute_multilabel_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.20,
) -> Dict:
    logits_norm = (logits - logits.min()) / (logits.max() - logits.min() + 1e-8)
    preds_binary = (logits_norm >= threshold).astype(int)

    no_pred = preds_binary.sum(axis=1) == 0
    if no_pred.any():
        preds_binary[no_pred, logits[no_pred].argmax(axis=1)] = 1

    sample_f1 = f1_score(labels, preds_binary, average="samples", zero_division=0)
    macro_f1 = f1_score(labels, preds_binary, average="macro", zero_division=0)
    micro_f1 = f1_score(labels, preds_binary, average="micro", zero_division=0)
    macro_recall = recall_score(labels, preds_binary, average="macro", zero_division=0)
    macro_precision = precision_score(labels, preds_binary, average="macro", zero_division=0)

    return {
        "threshold": threshold,
        "sample_f1": float(sample_f1),
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "macro_recall": float(macro_recall),
        "macro_precision": float(macro_precision),
    }


def find_best_threshold(
    logits: np.ndarray,
    labels: np.ndarray,
    thresholds: List[float] = None,
    metric: str = "sample_f1",
) -> Tuple[float, Dict]:
    if thresholds is None:
        thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
    best_score = -1
    best_result = None
    best_threshold = 0.20
    for t in thresholds:
        result = compute_multilabel_metrics(logits, labels, threshold=t)
        if result[metric] > best_score:
            best_score = result[metric]
            best_result = result
            best_threshold = t
    return best_threshold, best_result


# ─── Bootstrap 置信区间 ──────────────────────────────────────────────────────

def bootstrap_ci(
    metric_fn,
    *args,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Dict:
    rng = np.random.RandomState(seed)
    N = len(args[0])
    scores = []
    for _ in range(n_resamples):
        idx = rng.randint(0, N, size=N)
        resampled = [a[idx] if isinstance(a, np.ndarray) else a for a in args]
        try:
            score = metric_fn(*resampled)
            scores.append(score)
        except Exception:
            continue

    scores = np.array(scores)
    alpha = (1 - confidence) / 2
    lower = float(np.percentile(scores, 100 * alpha))
    upper = float(np.percentile(scores, 100 * (1 - alpha)))
    mean = float(np.mean(scores))
    return {"mean": mean, "lower": lower, "upper": upper, "std": float(np.std(scores))}


def bootstrap_hit_at_1(
    logits: np.ndarray,
    labels_multilabel: np.ndarray,
    n_resamples: int = 1000,
) -> Dict:
    def metric_fn(l, lb):
        return compute_hit_at_k(l, lb, k=1)
    return bootstrap_ci(metric_fn, logits, labels_multilabel, n_resamples=n_resamples)


def bootstrap_accuracy(
    preds: np.ndarray,
    labels: np.ndarray,
    n_resamples: int = 1000,
) -> Dict:
    def metric_fn(p, l):
        return float((p == l).mean())
    return bootstrap_ci(metric_fn, preds, labels, n_resamples=n_resamples)


# ─── Primary Label 转换（用于单标签评估）────────────────────────────────────

def multilabel_to_primary(labels_multilabel: np.ndarray) -> np.ndarray:
    N = labels_multilabel.shape[0]
    primary = np.zeros(N, dtype=int)
    for i in range(N):
        pos = np.where(labels_multilabel[i] > 0)[0]
        primary[i] = int(pos.min()) if len(pos) > 0 else 0
    return primary


# ─── 完整评估流水线 ──────────────────────────────────────────────────────────

def full_evaluation(
    logits: np.ndarray,
    labels_multilabel: np.ndarray,
    num_classes: int,
    multilabel: bool,
    class_names: Optional[List[str]] = None,
    thresholds: Optional[List[float]] = None,
    run_bootstrap: bool = True,
    n_resamples: int = 1000,
) -> Dict:
    results = {}

    if multilabel:
        if labels_multilabel.ndim == 1:
            raise ValueError("For multilabel, labels should be (N, C) binary matrix")
        lb_matrix = labels_multilabel
    else:
        N = len(labels_multilabel)
        lb_matrix = np.zeros((N, num_classes), dtype=float)
        if labels_multilabel.ndim == 1:
            for i, l in enumerate(labels_multilabel):
                lb_matrix[i, int(l)] = 1.0
        else:
            lb_matrix = labels_multilabel

    hit1 = compute_hit_at_k(logits, lb_matrix, k=1)
    hit3 = compute_hit_at_k(logits, lb_matrix, k=3)
    results["hit_at_1"] = hit1
    results["hit_at_3"] = hit3

    primary_labels = multilabel_to_primary(lb_matrix)
    pred_labels = logits.argmax(axis=1)
    sl_metrics = compute_single_label_metrics(
        pred_labels, primary_labels, num_classes, class_names
    )
    results["single_label"] = sl_metrics

    if multilabel:
        thresholds = thresholds or [0.10, 0.15, 0.20, 0.25, 0.30]
        best_thresh, best_ml = find_best_threshold(logits, lb_matrix, thresholds)
        results["multilabel_best_threshold"] = best_thresh
        results["multilabel"] = best_ml

    if run_bootstrap:
        ci = bootstrap_hit_at_1(logits, lb_matrix, n_resamples=n_resamples)
        results["hit_at_1_ci"] = ci

    return results
