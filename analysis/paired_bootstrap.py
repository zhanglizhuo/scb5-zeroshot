"""Paired bootstrap significance test for CAPE Hit@1 between backbone pairs.

Uses cached CAPE logits in data/feature_cache/ (openai, dfn, siglip2 with
full-coverage caches). For each (dataset, pair) computes:
  - per-sample Hit@1 indicator under each backbone (multi-label argmax for
    Teacher; single-label primary accuracy for Handrise/BowTurn);
  - paired bootstrap (5000 resamples) of the per-sample difference;
  - 95% CI of Δ (pp) and two-sided p-value for H0: mean Δ = 0.

This complements tab:bootstrap_ci (marginal CIs) by addressing the
reviewer concern that overlapping marginal CIs do not imply non-significance
in paired-comparison settings.
"""
import json
import time
from pathlib import Path

import numpy as np

CACHE = Path(__file__).resolve().parent.parent / "data" / "feature_cache"
MODELS = ["openai", "dfn", "siglip2"]
DATASETS = {
    "teacher_behavior":   {"multi_label": True,  "name": "TeacherBehavior"},
    "handrise_readwrite": {"multi_label": False, "name": "HandriseReadWrite"},
    "bow_turnhead":       {"multi_label": False, "name": "BowTurnHead"},
}


def hit_indicator(logits, labels, multi_label):
    pred = logits.argmax(axis=1)
    if multi_label:
        return labels[np.arange(len(labels)), pred].astype(float)
    return (pred == labels).astype(float)


def paired_bootstrap(a, b, n=5000, seed=42):
    rng = np.random.RandomState(seed)
    n_samples = len(a)
    diffs = []
    for _ in range(n):
        idx = rng.choice(n_samples, size=n_samples, replace=True)
        diffs.append(a[idx].mean() - b[idx].mean())
    diffs = np.array(diffs)
    obs = a.mean() - b.mean()
    if obs >= 0:
        p = (diffs <= 0).mean()
    else:
        p = (diffs >= 0).mean()
    p = min(2 * p, 1.0)  # two-sided
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(obs * 100), float(lo * 100), float(hi * 100), float(p)


def main():
    rows = []
    for ds_key, ds_info in DATASETS.items():
        indicators = {}
        for m in MODELS:
            d = np.load(CACHE / f"{m}_{ds_key}_validation.npz")
            indicators[m] = hit_indicator(d["logits_cape"], d["labels"], ds_info["multi_label"])
        for i, a in enumerate(MODELS):
            for b in MODELS[i + 1:]:
                obs, lo, hi, p = paired_bootstrap(indicators[a], indicators[b])
                rows.append({
                    "dataset": ds_info["name"],
                    "model_a": a,
                    "model_b": b,
                    "acc_a_pct": round(float(indicators[a].mean() * 100), 2),
                    "acc_b_pct": round(float(indicators[b].mean() * 100), 2),
                    "delta_pp": round(obs, 2),
                    "ci_low_pp": round(lo, 2),
                    "ci_high_pp": round(hi, 2),
                    "p_value": round(p, 4),
                })
    out_dir = Path(__file__).resolve().parent.parent / "results" / "revision"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"paired_bootstrap_{int(time.time())}.json"
    with open(out_path, "w") as f:
        json.dump({"n_bootstrap": 5000, "seed": 42, "rows": rows}, f, indent=2)
    print(f"Saved: {out_path}")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
