# Data Download Guide

This repository does not redistribute SCB dataset files.

## Official Source

- Hugging Face: https://huggingface.co/datasets/wintonYF/SCB-Dataset

## Required Subsets

- SCB5-TeacherBehavior
- SCB5-HandriseReadWrite
- SCB-BowTurnHead
- SCB5-Discuss (optional; excluded from quantitative analysis)

## Expected Local Structure

```text
datasets_scb/
  SCB5_TeacherBehavior/
  SCB5_HandriseReadWrite/
  SCB_BowTurnHead/
```

(or equivalently under `data/` for each subset).

## Note

The original dataset license and attribution requirements should be followed.
The download script (`scripts/download_scb5_data.py`) places files under `datasets_scb/`.
