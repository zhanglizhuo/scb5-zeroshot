"""Evaluation metrics used in SCB benchmarking."""

from typing import Iterable, List

import numpy as np
from sklearn.metrics import f1_score


def hit_at_k(y_true: List[Iterable[int]], y_topk: List[List[int]], k: int = 1) -> float:
    correct = 0
    for truth_set, preds in zip(y_true, y_topk):
        pred_k = preds[:k]
        if any(p in set(truth_set) for p in pred_k):
            correct += 1
    return 100.0 * correct / len(y_true)


def macro_f1_single_label(y_true: List[int], y_pred: List[int]) -> float:
    return 100.0 * f1_score(y_true, y_pred, average="macro")


def multilabel_f1(y_true: np.ndarray, y_pred: np.ndarray):
    """Return sample, macro, micro F1 in percentage."""
    sample = 100.0 * f1_score(y_true, y_pred, average="samples", zero_division=0)
    macro = 100.0 * f1_score(y_true, y_pred, average="macro", zero_division=0)
    micro = 100.0 * f1_score(y_true, y_pred, average="micro", zero_division=0)
    return {"sample_f1": sample, "macro_f1": macro, "micro_f1": micro}
