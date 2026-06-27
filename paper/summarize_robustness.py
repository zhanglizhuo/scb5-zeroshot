#!/usr/bin/env python3
"""Summarize CAPE robustness experiment JSON files into a paper-ready table."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results" / "robustness"

MODEL_ORDER = ["clip", "laion", "siglip", "eva02", "dfn"]
MODEL_LABELS = {
    "clip": "CLIP (OpenAI)",
    "laion": "OpenCLIP (LAION)",
    "siglip": "SigLIP2",
    "eva02": "EVA02-CLIP",
    "dfn": "DFN-CLIP",
}
DATASET_ORDER = [
    "SCB5_TeacherBehavior",
    "SCB5_HandriseReadWrite",
    "SCB_BowTurnHead",
]


def load_latest_results() -> dict[tuple[str, str, str], float]:
    merged: dict[tuple[str, str, str], tuple[int, float]] = {}
    for path in sorted(RESULTS_DIR.glob("*.json")):
        data = json.loads(path.read_text())
        timestamp = int(data.get("timestamp", 0))
        for experiment in data.get("experiments", []):
            key = (
                experiment.get("model", ""),
                experiment.get("dataset", ""),
                experiment.get("variant", ""),
            )
            hit1 = float(experiment.get("hit1", 0.0))
            previous = merged.get(key)
            if previous is None or timestamp >= previous[0]:
                merged[key] = (timestamp, hit1)
    return {key: value for key, (_, value) in merged.items()}


def format_value(value: float | None) -> str:
    return "--" if value is None else f"{value:.2f}"


def collect_mix(results: dict[tuple[str, str, str], float], model: str, dataset: str) -> str:
    mix_values = []
    for trial in range(5):
        key = (model, dataset, f"CAPE_Mix_t{trial}")
        if key in results:
            mix_values.append(results[key])
    if not mix_values:
        return "--"
    mean = sum(mix_values) / len(mix_values)
    variance = sum((value - mean) ** 2 for value in mix_values) / len(mix_values)
    return f"{mean:.2f} $\\pm$ {variance ** 0.5:.2f}"


def main() -> None:
    results = load_latest_results()
    if not results:
        print("No completed robustness experiments found in results_robustness/.")
        return

    print("Coverage")
    print("========")
    for dataset in DATASET_ORDER:
        print(f"\n{dataset}")
        for model in MODEL_ORDER:
            count = sum(
                1
                for variant in ["CAPE_A_n1", "CAPE_A_n2", "CAPE_A_n3", "CAPE_B_n3"]
                if (model, dataset, variant) in results
            )
            count += sum(
                1 for trial in range(5) if (model, dataset, f"CAPE_Mix_t{trial}") in results
            )
            print(f"  {model:<7} {count}/9")

    print("\nLaTeX Rows")
    print("==========")
    for dataset in DATASET_ORDER:
        print(f"\n% {dataset}")
        for model in MODEL_ORDER:
            a1 = results.get((model, dataset, "CAPE_A_n1"))
            a2 = results.get((model, dataset, "CAPE_A_n2"))
            a3 = results.get((model, dataset, "CAPE_A_n3"))
            b3 = results.get((model, dataset, "CAPE_B_n3"))
            mix = collect_mix(results, model, dataset)
            print(
                f"{MODEL_LABELS[model]} & {format_value(a1)} & {format_value(a2)} & "
                f"{format_value(a3)} & {format_value(b3)} & {mix} \\\\"
            )


if __name__ == "__main__":
    main()