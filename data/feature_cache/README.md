# Feature Cache

Precomputed CAPE logits for the three backbones with full-coverage feature
caches (CLIP/OpenAI, DFN-CLIP, SigLIP2), used by the paired-bootstrap
significance test (`analysis/paired_bootstrap.py`) and the
three-principle ablation (`analysis/cape_principle_ablation.py`).

## Contents

| File | Description |
|------|-------------|
| `{model}_{subset}_validation.npz` | CAPE logits (num_images × num_classes) |

Models: `openai`, `dfn`, `siglip2`.
Subsets: `teacher_behavior`, `handrise_readwrite`, `bow_turnhead`.

## Usage

These files are loaded automatically by the analysis scripts:

```bash
python analysis/paired_bootstrap.py
python analysis/cape_principle_ablation.py
```
