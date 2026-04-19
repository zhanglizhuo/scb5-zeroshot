# scb-clip-benchmark

Reproducibility package for the paper:
Zero-Shot Classroom Activity Recognition: Benchmarking CLIP-Family Models with Class-Aware Prompt Ensemble.

This repository is structured to support transparent verification and reproducibility:
- Public experiment code for the CLIP-family baseline and MLLM comparison.
- Selected paper-aligned result JSON files for direct reference.
- Download instructions for the third-party SCB subsets used in the paper.

## Paper Summary

Five CLIP-family backbones (CLIP, OpenCLIP, SigLIP2, EVA02-CLIP, DFN-CLIP) are benchmarked on three public SCB subsets under multiple prompt strategies, including CAPE (Class-Aware Prompt Ensemble).

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Place the SCB subsets under ./data or export SCB_DATA_DIR=/path/to/SCB
# Optional: export CKPTS_DIR=/path/to/local/checkpoints

# Run the CLIP-family baseline pipeline
bash experiments/run_all.sh
```

For the cross-family MLLM rerun, `experiments/main_mllm.py` is included, but the default paper results are already provided under `results/mllm/` because rerunning the Ollama- or API-based models requires additional runtime services.

## Core Results (Paper Table 3 / E1)

The main reported values are listed below so that the core claims can be checked without rerunning all experiments.

| Paper Experiment | Sub-dataset | Best Model + Prompt | Hit@1 (%) | Macro-F1 (%) |
|---|---|---|---:|---:|
| E1 / Table 3 (best per subset) | TeacherBehavior | SigLIP2 + CAPE | 85.56 | 10.07 |
| E1 / Table 3 (best per subset) | HandriseReadWrite | OpenCLIP + action | 84.56 | 55.89 |
| E1 / Table 3 (best per subset) | BowTurnHead | DFN-CLIP + CAPE | 93.27 | 53.95 |

Source of these values in this repo:
- `results/baseline_results.json`
- `results/paper/benchmark_final_merged.json`
- `experiments/main_clip.py`

## Included Assets

- `experiments/main_clip.py` contains the full CLIP-family baseline runner used to regenerate baseline JSON outputs.
- `experiments/main_mllm.py` and `models/mllm_baseline.py` contain the cross-family MLLM evaluation code.
- `results/paper/benchmark_final_merged.json` stores the merged paper benchmark outputs.
- `results/paper/cape_robustness_summary.json` stores the CAPE robustness summary used for the robustness section.
- `results/mllm/mllm_merged_summary.json` and companion files store the released cross-family summary metrics.
- Raw SCB images are not redistributed here because the datasets are third-party public releases.

## Repository Layout

```text
scb-clip-benchmark/
├── README.md
├── requirements.txt
├── config/
│   └── experiment_config.yaml
├── data/
│   ├── README.md
│   └── scb_dataset.py
├── prompts/
│   ├── cape_prompts.py
│   ├── llm_prompt_gen.py
│   └── prompt_sets.json
├── models/
│   ├── clip_zoo.py
│   └── mllm_baseline.py
├── evaluation/
│   └── metrics.py
├── experiments/
│   ├── main_clip.py
│   ├── main_mllm.py
│   ├── merge_mllm_results.py
│   └── run_all.sh
├── results/
│   ├── baseline_results.json
│   ├── baseline_eva02_fix_allstrat/
│   ├── mllm/
│   └── paper/
└── notebooks/
    └── reproduce_figures.ipynb
```

## Data Availability

SCB data are third-party public datasets and are not redistributed in this repository.
Download instructions are provided in `data/README.md`.

## Citation

The associated paper and repository release should be cited when this repository is used.
