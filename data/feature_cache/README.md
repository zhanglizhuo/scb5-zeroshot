# Feature Cache

This directory stores precomputed image and text features for the five
CLIP-family backbones used in the paper.

## Contents

- `teacher/`, `handrise/`, `bowturn/`: Per-subset directories
  - `{backbone}_logits_cape.npy`: CAPE logits (num_images × num_classes)
  - `primary_labels.npy`: Primary (single) label per image
  - `multilabel_sets.npy`: Ground-truth multi-label sets (TeacherBehavior only)
- `ablation/`: Three-principle ablation logits
  - `{backbone}_logits_c{n}_teacher.npy`, `{backbone}_logits_c{n}_bowturn.npy`

## Obtaining the Cache

**Option 1: Download precomputed cache (recommended)**

Download from [Google Drive / Hugging Face / Zenodo link — TODO: INSERT LINK]
and extract into this directory:

    tar -xzf scb_clip_feature_cache.tar.gz -C data/feature_cache/

**Option 2: Regenerate from scratch**

Run the Stage-1 benchmark and the ablation experiment:

    bash experiments/run_all.sh
    python experiments/run_ablation.py

The cache will be populated automatically in `data/feature_cache/`.

## Note

The full cache is approximately 5–8 GB. It is not distributed via GitHub
due to file size limits. See the paper's Data Availability Statement for
the canonical download location.
