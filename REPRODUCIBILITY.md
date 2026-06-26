# SCB5 Zero-Shot Reproducibility Guide

This guide is intended for reviewers and researchers who want to reproduce the paper results.

## 1. Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Data

Download SCB subsets from Hugging Face (`wintonYF/SCB-Dataset`). Expected layout:

```
data/
  SCB5_TeacherBehavior/
  SCB5_HandriseReadWrite/
  SCB_BowTurnHead/
```

## 3. Quick start (reviewer-friendly)

Fast validation from existing precomputed results:

```bash
bash reproduce_paper.sh --mode quick
```

Regenerates paper figures and rebuilds the PDF.

## 4. Full rerun

```bash
bash reproduce_paper.sh --mode full
```

Runs experiments end-to-end: main benchmark, CAPE robustness, revision
experiments, figure regeneration, and paper PDF build.

## 5. Manual analysis scripts

After experiments complete, run paper-specific analyses:

```bash
python scb5_zeroshot/paired_bootstrap.py
python scb5_zeroshot/cape_principle_ablation.py
```

## 6. Expected outputs

| Output | Location |
|--------|----------|
| Main results | `results/` |
| Revision results | `results_revision/` |
| Robustness results | `results_robustness/` |
| Analysis scripts | `scb5_zeroshot/` |
| Feature caches | `data/feature_cache/` |
| Paper figures | `paper/figures/` |
| Paper PDF | `paper/scb5_zeroshot_paper.pdf` |

## 7. Entry points

| Script | Purpose |
|--------|---------|
| `reproduce_paper.sh` | **Canonical entry point** (quick or full) |
| `experiments/main_clip.py` | Programmatic API for CLIP experiments |
| `experiments/main_mllm.py` | MLLM evaluation |
| `exp_runner.py` | Original full pipeline (legacy; use `reproduce_paper.sh` instead) |
