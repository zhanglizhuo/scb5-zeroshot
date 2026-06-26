# SCB5 Zero-Shot Reproducibility Guide

This guide is intended for reviewers and researchers who want to reproduce the paper results quickly and reliably.

## 1. Scope

This package reproduces the SCB zero-shot evaluation pipeline and manuscript artifacts in `scb5_zeroshot/paper/`.

Main goals:
- Re-run key experiments from scripts in `scb5_zeroshot/`.
- Regenerate paper figures in `scb5_zeroshot/paper/figures/`.
- Rebuild the manuscript PDF from `scb5_zeroshot/paper/scb5_zeroshot_paper.tex`.

## 2. Environment

Recommended:
- Linux + CUDA GPU
- Python 3.11+
- PyTorch with matching CUDA build

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r scb5_zeroshot/requirements.repro.txt
```

## 3. Data and checkpoints

Expected paths:
- Datasets: `datasets_scb/`
- Model checkpoints: `scb5_zeroshot/ckpts/`

Notes:
- SCB datasets are third-party public resources. See manuscript Data Availability section.
- If paths differ on your machine, adapt the environment variables in the run commands.

## 4. Quick start (reviewer-friendly)

Fast validation from existing result JSON files:

```bash
bash scb5_zeroshot/reproduce_paper.sh --mode quick --gpu 0
```

What this does:
- Validates Python/runtime imports.
- Regenerates paper figures.
- Rebuilds the manuscript PDF.

## 5. Full rerun

Run a heavier reproducibility pass:

```bash
bash scb5_zeroshot/reproduce_paper.sh --mode full --gpu 0
```

This runs:
- Main experiment pipeline (`exp_runner.py`)
- CAPE robustness (`cape_robustness.py`)
- Revision experiments (`run_revision_experiments.py`)
- Figure regeneration and paper PDF build

## 6. Expected outputs

- Main results: `scb5_zeroshot/results/`
- Revision results: `scb5_zeroshot/results_revision/`
- Robustness results: `scb5_zeroshot/results_robustness/`
- Figures: `scb5_zeroshot/paper/figures/`
- Paper PDF: `scb5_zeroshot/paper/scb5_zeroshot_paper.pdf`

## 7. Reproducibility checklist (for GitHub release)

- Pin code with a git tag (e.g., `v1.0.0-paper`).
- Record exact commit hash in release notes.
- Include environment details (Python, torch, CUDA, GPU).
- Keep random seeds fixed in scripts where available.
- Provide one command for quick verification and one for full rerun.
- Keep generated results and manuscript artifacts versioned.
