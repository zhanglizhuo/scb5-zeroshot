# SCB5 Zero-Shot Reproducibility Guide

This guide is intended for reviewers and researchers who want to reproduce the paper results quickly and reliably.

## 1. Environment

Recommended:
- Linux + CUDA GPU
- Python 3.8+
- PyTorch with matching CUDA build

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Data and checkpoints

```bash
# Download SCB datasets
python scripts/download_scb5_data.py

# Download model checkpoints (requires internet)
python scripts/download_models.py
```

Expected paths:
- Datasets: `data/SCB5_TeacherBehavior/`, `data/SCB5_HandriseReadWrite/`, `data/SCB_BowTurnHead/`
- Model checkpoints: `ckpts/`

SCB datasets are third-party public resources (see manuscript Data Availability section).

## 3. Quick start (reviewer-friendly)

Fast validation from precomputed result files — no external downloads required:

```bash
bash reproduce_paper.sh --mode quick
```

What this does:
- Validates Python/runtime imports.
- Regenerates paper figures from cached results.
- Rebuilds the manuscript PDF.

## 4. Full rerun

End-to-end reproducibility (requires data and model checkpoints):

```bash
bash reproduce_paper.sh --mode full
```

This runs:
- CLIP-family benchmark (`experiments/main_clip.py`)
- CAPE robustness (`analysis/cape_robustness.py`)
- CAPE principle ablation (`analysis/cape_principle_ablation.py`)
- Paired bootstrap test (`analysis/paired_bootstrap.py`)
- Linear probe supervised baseline (`analysis/linear_probe.py`)
- Revision experiments R1-R4 (`analysis/run_revision_experiments.py`)
- Figure regeneration and paper PDF build

## 5. Expected outputs

| Output | Path |
|--------|------|
| Main results table | `results/baseline_results.json` |
| Full benchmark (all models × prompts × subsets) | `results/paper/benchmark_final_merged.json` |
| CAPE robustness results | `results/robustness/` |
| Revision experiment results | `results/revision/` |
| MLLM evaluation results | `results/mllm/` |
| Paper figures | `paper/figures/` |
| Manuscript PDF | `paper/scb5_zeroshot_paper.pdf` |

## 6. Reproducibility checklist

- [ ] Pin code with a git tag (e.g., `v1.0.0-paper`)
- [ ] Record exact commit hash in release notes
- [ ] Include environment details (Python, torch, CUDA, GPU)
- [ ] Provide one command for quick verification and one for full rerun
- [ ] Keep generated results and manuscript artifacts versioned
