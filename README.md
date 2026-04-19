# scb-clip-benchmark

Reproducibility package for the paper:
Zero-Shot Classroom Activity Recognition: Benchmarking CLIP-Family Models with Class-Aware Prompt Ensemble.

This repository is organized for reviewer-friendly verification:
- One-command experiment entrypoint.
- Precomputed baseline results for direct checking.
- Clear mapping from scripts/results to paper experiments.

## Paper Summary

We benchmark five CLIP-family backbones (CLIP, OpenCLIP, SigLIP2, EVA02-CLIP, DFN-CLIP) on three public SCB subsets under multiple prompt strategies, including CAPE (Class-Aware Prompt Ensemble).

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run all reproduction scripts
bash experiments/run_all.sh
```

## Reviewer-Friendly Core Numbers (Paper Table 3 / E1)

The table below exposes core headline values so reviewers can verify claims without rerunning all experiments.

| Paper Experiment | Sub-dataset | Best Model + Prompt | Hit@1 (%) | Macro-F1 (%) |
|---|---|---|---:|---:|
| E1 / Table 3 (best per subset) | TeacherBehavior | SigLIP2 + CAPE | 85.56 | 10.07 |
| E1 / Table 3 (best per subset) | HandriseReadWrite | OpenCLIP + action | 84.56 | 55.89 |
| E1 / Table 3 (best per subset) | BowTurnHead | DFN-CLIP + CAPE | 93.27 | 53.95 |

Source of these values in this repo:
- `results/baseline_results.json`
- `experiments/main_clip.py`

## Repository Layout

```text
scb-clip-benchmark/
├── README.md
├── requirements.txt
├── config/
│   └── experiment_config.yaml
├── data/
│   └── README.md
├── prompts/
│   ├── cape_prompts.py
│   └── prompt_sets.json
├── models/
│   └── clip_zoo.py
├── evaluation/
│   └── metrics.py
├── experiments/
│   ├── main_clip.py
│   ├── main_mllm.py
│   └── run_all.sh
├── results/
│   └── baseline_results.json
└── notebooks/
    └── reproduce_figures.ipynb
```

## Data Availability

SCB data are third-party public datasets and are not redistributed in this repository.
Please follow instructions in `data/README.md` to download from the official source.

## Citation

If you use this repository, please cite the paper and repository release.
