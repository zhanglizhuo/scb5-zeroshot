# scb5-zeroshot

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue)]()
[![OpenCLIP](https://img.shields.io/badge/OpenCLIP-MLFoundations-blueviolet)]()
[![HuggingFace Datasets](https://img.shields.io/badge/%F0%9F%A4%97%20Datasets-SCB--Dataset-yellow)](https://huggingface.co/datasets/wintonYF/SCB-Dataset)
[![Paper](https://img.shields.io/badge/PDF-Paper-red)](paper/scb5_zeroshot_paper.pdf)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey)]()

**Adversarial Prompt Sensitivity in CLIP-Family Models for Zero-Shot Classroom Behavior Analysis**

Yan Ma, Lizhuo Zhang, and Xinjie Wu. Submitted to MDPI Symmetry (Special Issue on Adversarial Machine Learning), 2026.

---

## For Reviewers

```bash
# Quick reproduction from precomputed results:
bash reproduce_paper.sh --mode quick
```

Everything needed is committed in this repository — no external downloads required for the quick path.
See [Quick Start](#quick-start) below for full details.

---

## Overview

CLIP-family models exhibit **instability under prompt variation** in zero-shot classroom behavior analysis. A single model can swing from 95.5% to 31.4% Hit@1 when prompt wording or count changes — without any data or model modification. This repository provides the complete experimental framework to reproduce, verify, and extend these findings. (Figures are available as PDFs in `paper/figures/`.)

## Key Results

### Best-performing configuration per subset (Hit@1)

| Sub-dataset | Best Model + Prompt Strategy | Hit@1 (%) | Macro-F1 (%) |
| --- | --- | ---: | ---: |
| **TeacherBehavior** | SigLIP2 + CAPE | 85.56 | 10.07 |
| **HandriseReadWrite** | OpenCLIP + Action prompt | 84.56 | 55.89 |
| **BowTurnHead** | DFN-CLIP + CAPE | 93.27 | 53.95 |

### Prompt sensitivity leads to inconsistent model rankings

A core finding: **the choice of prompt strategy changes which model appears "best"** on a given subset. For example, on TeacherBehavior, SigLIP2 ranks first under CAPE but drops below CLIP and DFN-CLIP under simpler prompt strategies. No single model dominates across all conditions.

### Misleading leniency of multi-label Hit@1

On multi-label subsets (TeacherBehavior), the lenient Hit@1 metric can saturate above 85% even when Macro-F1 is below 15%, because models collapse predictions into the majority class. (See confusion matrix and per-class recall figures in `paper/figures/`.)

## Quick Start

### Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Data

Download SCB subsets from [HuggingFace `wintonYF/SCB-Dataset`](https://huggingface.co/datasets/wintonYF/SCB-Dataset):

```bash
python scripts/download_scb5_data.py
```

Expected layout:

```
data/
  SCB5_TeacherBehavior/
  SCB5_HandriseReadWrite/
  SCB_BowTurnHead/
```

### Reproduce

```bash
# 数据下载
python scripts/download_scb5_data.py

# 模型权重下载（需网络）
python scripts/download_models.py

# Quick: regenerate figures and tables from precomputed results
bash reproduce_paper.sh --mode quick

# Full: end-to-end rerun (requires model checkpoints and data)
bash reproduce_paper.sh --mode full
```

### Entry Points

| Command | Purpose |
|---------|---------|
| `bash reproduce_paper.sh` | Canonical entry point (quick or full) |
| `python experiments/main_clip.py` | CLIP-family benchmark (5 models × 5 prompt strategies × 3 subsets) |
| `python experiments/main_mllm.py` | MLLM evaluation (LLaVA, Qwen-VL, GPT-4V) |
| `python analysis/cape_robustness.py` | CAPE prompt-count / alternate-wording robustness |
| `python analysis/cape_principle_ablation.py` | CAPE three-principle ablation |
| `python analysis/paired_bootstrap.py` | Bootstrap significance test |
| `python analysis/linear_probe.py` | Supervised linear probe baseline |
| `python analysis/llm_baselines.py` | CuPL + WaffleCLIP literature baselines |
| `python analysis/run_revision_experiments.py` | Revision experiments R1-R4 |

## Repository Structure

```text
scb5-zeroshot/
├── README.md | CITATION.cff | requirements*.txt   # Project metadata
├── reproduce_paper.sh                             # ★ Canonical entry point
├── analysis/                                      # Core analysis (bootstrap, ablation, robustness, ...)
├── config/ | data/ | evaluation/ | models/        # Experiment components
├── experiments/                                   # Runners (main_clip.py, main_mllm.py, *.sh)
├── paper/                                         # Manuscript + figures (PDF/LaTeX)
├── prompts/                                       # Prompt definitions (A/B/C + uniform strategies)
├── scripts/                                       # Utilities (download, setup, summarize)
└── results/                                       # All outputs (baseline, mllm, revision, ...)
```

Key outputs: `results/baseline_results.json` (main table), `results/mllm/mllm_macrof1_all_models.json` (MLLM validation), `paper/figures/` (PDF figures), `paper/scb5_zeroshot_paper.pdf` (manuscript).

## Data Availability

SCB data are third-party public datasets available at [HuggingFace](https://huggingface.co/datasets/wintonYF/SCB-Dataset) and are not redistributed in this repository. All experiment code, prompt templates, precomputed features, and result files are provided here for full reproducibility.

## Citation

```bibtex
@article{ma2026prompt,
  title     = {Adversarial Prompt Sensitivity in {CLIP}-Family
               Models for Zero-Shot Classroom Behavior Analysis},
  author    = {Ma, Yan and Zhang, Lizhuo and Wu, Xinjie},
  journal   = {Submitted to MDPI Symmetry (Special Issue on Adversarial Machine Learning)},
  year      = {2026},
  note      = {Code and data: \url{https://github.com/zhanglizhuo/scb5-zeroshot}}
}
```
