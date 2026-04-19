"""
Merge MLLM result caches from one or more directories.

Usage:
    python3 experiments/merge_mllm_results.py \
        --input_glob 'results/mllm*' \
        --out results/mllm/mllm_merged_summary.json
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge Experiment_Ex MLLM result caches")
    p.add_argument(
        "--input_glob",
        type=str,
        default="results/mllm*",
        help="Glob pattern for result directories that contain *_preds.json",
    )
    p.add_argument(
        "--out",
        type=str,
        default="results/mllm/mllm_merged_summary.json",
        help="Output merged summary path",
    )
    return p.parse_args()


def load_pred_files(dirs: List[Path]) -> List[Dict]:
    records: List[Dict] = []
    for d in dirs:
        if not d.is_dir():
            continue
        for fp in sorted(d.glob("*_preds.json")):
            with fp.open("r", encoding="utf-8") as f:
                obj = json.load(f)
            obj["_source_file"] = str(fp)
            obj["_source_dir"] = str(d)
            records.append(obj)
    return records


def build_summary(records: List[Dict]) -> Dict:
    by_model: Dict[str, Dict[str, float]] = {}
    weighted: Dict[str, Dict[str, float]] = {}

    for r in records:
        model = r.get("model")
        subset = r.get("subset")
        if not model or not subset:
            continue

        by_model.setdefault(model, {})
        by_model[model][subset] = {
            "hit_at_1": float(r.get("hit_at_1", 0.0)),
            "hit_at_3": float(r.get("hit_at_3", 0.0)),
            "macro_f1": float(r.get("macro_f1", 0.0)),
            "num_samples": int(r.get("num_samples", 0)),
            "source": r.get("_source_file", ""),
        }

    for model, subsets in by_model.items():
        n = sum(v["num_samples"] for v in subsets.values())
        if n <= 0:
            continue
        h1 = sum(v["hit_at_1"] * v["num_samples"] for v in subsets.values()) / n
        h3 = sum(v["hit_at_3"] * v["num_samples"] for v in subsets.values()) / n
        mf1 = sum(v["macro_f1"] * v["num_samples"] for v in subsets.values()) / n
        weighted[model] = {
            "weighted_hit_at_1": h1,
            "weighted_hit_at_3": h3,
            "weighted_macro_f1": mf1,
            "total_samples": n,
        }

    return {
        "models": by_model,
        "weighted": weighted,
        "num_records": len(records),
    }


def main() -> None:
    args = parse_args()
    input_glob = args.input_glob
    if not Path(input_glob).is_absolute():
        input_glob = str(REPO_ROOT / input_glob)
    matched = sorted(Path(p) for p in glob.glob(input_glob))
    records = load_pred_files(matched)
    summary = build_summary(records)

    out = Path(args.out)
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Merged {summary['num_records']} files from {len(matched)} dirs -> {out}")
    for model, m in summary.get("weighted", {}).items():
        print(
            f"  {model}: hit@1={m['weighted_hit_at_1']:.4f}, "
            f"hit@3={m['weighted_hit_at_3']:.4f}, "
            f"macro-f1={m['weighted_macro_f1']:.4f}, n={m['total_samples']}"
        )


if __name__ == "__main__":
    main()
