"""Reproduce core CLIP-family table outputs from stored baseline results."""

from pathlib import Path
import json


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    results_path = repo_root / "results" / "baseline_results.json"

    with results_path.open("r", encoding="utf-8") as f:
        results = json.load(f)

    print("=== E1 / Table 3 Core Results ===")
    for row in results["table3_core"]:
        print(
            f"{row['subset']:<18} | {row['best_model_prompt']:<24} | "
            f"Hit@1={row['hit_at_1']:.2f} | Macro-F1={row['macro_f1']:.2f}"
        )


if __name__ == "__main__":
    main()
