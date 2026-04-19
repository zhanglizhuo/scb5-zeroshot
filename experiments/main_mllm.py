"""Reproduce core MLLM comparison rows (paper E11 / Table 7)."""

from pathlib import Path
import json


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    results_path = repo_root / "results" / "baseline_results.json"

    with results_path.open("r", encoding="utf-8") as f:
        results = json.load(f)

    print("=== E11 / Table 7 Weighted Results ===")
    for row in results["table7_cross_family"]:
        print(
            f"{row['model']:<20} | "
            f"Weighted Hit@1={row['weighted_hit_at_1']:.2f} | "
            f"Weighted Macro-F1={row['weighted_macro_f1']:.2f}"
        )


if __name__ == "__main__":
    main()
