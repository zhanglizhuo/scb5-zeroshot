# scb-clip-benchmark

Top-down reproducibility repository for the paper:

**Prompt Sensitivity as an Adversarial Vulnerability in CLIP-Family Models for Zero-Shot Classroom Behavior Analysis**

Yan Ma, Lizhuo Zhang, and Xinjie Wu. Submitted to MDPI Symmetry (Special Issue on Adversarial Machine Learning), 2026.

This repository is organized as a complete experimental workflow, from dataset preparation to final paper-ready outputs.

## Top-Down Experimental Workflow

Stage 0: Environment and data setup
- Install dependencies from requirements.txt.
- Download the public SCB subsets using instructions in data/README.md.

Stage 1: CLIP-family benchmark under unified protocols
- Run experiments/main_clip.py (or experiments/run_all.sh).
- This stage evaluates five CLIP-family backbones across three SCB subsets with multiple prompt strategies.

Stage 2: Prompt-strategy sensitivity and robustness summaries
- Use the released baseline outputs and prompt definitions to verify strategy-dependent ranking behavior.
- Aggregated benchmark artifacts are available in results/paper/.

Stage 3: Cross-family validation with MLLM baselines
- Run experiments/main_mllm.py when runtime services are available.
- Released cross-family summaries are available in results/mllm/.

Stage 4: Figure/table reproduction
- Use notebooks/reproduce_figures.ipynb with released JSON outputs to regenerate paper figures.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Place SCB subsets under ./data or export SCB_DATA_DIR=/path/to/SCB
# Optional: export CKPTS_DIR=/path/to/local/checkpoints

# Stage-1 benchmark run
bash experiments/run_all.sh

# Optional Stage-3 rerun (requires extra runtime services)
python experiments/main_mllm.py --results_dir results/mllm_runtime
```

## Key Outputs and Paper Mapping

| Workflow Stage | Output file(s) | Paper usage |
|---|---|---|
| Stage 1 | results/baseline_results.json | Main benchmark table values |
| Stage 1/2 | results/paper/benchmark_final_merged.json | Final merged benchmark summary |
| Stage 2 | results/paper/cape_robustness_summary.json | Robustness analysis |
| Stage 3 | results/mllm/mllm_merged_summary.json | Cross-family validation summary |

Core best-per-subset values (main benchmark):

| Sub-dataset | Best Model + Prompt | Hit@1 (%) | Macro-F1 (%) |
|---|---|---:|---:|
| TeacherBehavior | SigLIP2 + CAPE | 85.56 | 10.07 |
| HandriseReadWrite | OpenCLIP + action | 84.56 | 55.89 |
| BowTurnHead | DFN-CLIP + CAPE | 93.27 | 53.95 |

## Repository Layout

```text
scb-clip-benchmark/
├── README.md
├── requirements.txt
├── config/
│   └── experiment_config.yaml
├── data/
│   ├── README.md
│   ├── scb_dataset.py
│   └── feature_cache/        # Precomputed model features (see README)
├── prompts/
│   ├── cape_prompts.py
│   ├── llm_prompt_gen.py
│   └── prompt_sets.json       # Set A / B / C prompt templates
├── scb5_zeroshot/             # Paper-specific analysis scripts
│   ├── paired_bootstrap.py    # Paired-bootstrap significance test (Tab. 9)
│   ├── cape_principle_ablation.py  # CAPE three-principle ablation (Tab. 5)
│   └── prompts/
│       └── setAB_examples.json # Verbatim Set A & B prompts
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
Download instructions are provided in data/README.md.

## Citation

Please cite the associated paper and this repository when using these assets:

```bibtex
@article{ma2026prompt,
  title     = {Prompt Sensitivity as an Adversarial Vulnerability in {CLIP}-Family
               Models for Zero-Shot Classroom Behavior Analysis},
  author    = {Ma, Yan and Zhang, Lizhuo and Wu, Xinjie},
  journal   = {Submitted to MDPI Symmetry (Special Issue on Adversarial Machine Learning)},
  year      = {2026},
  note      = {Code and data: \url{https://github.com/zhanglizhuo/scb-clip-benchmark}}
}
```

