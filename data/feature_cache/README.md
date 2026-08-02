# Feature Cache

Precomputed CAPE logits for all five backbones, used by the paired-bootstrap
significance test (`analysis/paired_bootstrap.py`), the three-principle
ablation (`analysis/cape_principle_ablation.py`), and the dev/test split
validation (`analysis/dev_test_split_eval.py`).

## Contents

| File | Description |
|------|-------------|
| `{model}_{subset}_validation.npz` | `image_features`, `labels`, `logits_cape` (num_images × num_classes) |
| `tembs/{model}_{subset}_capeA_tembs.npz` | Canonical CAPE Set A text embeddings (open_clip 2.24.0), used to verify `logits_cape` |

Models: `openai`, `laion`, `siglip2`, `eva02`, `dfn`.
Subsets: `teacher_behavior`, `handrise_readwrite`, `bow_turnhead`.

## Integrity

`logits_cape = image_features @ tembs.T` must hold for every cache, with
`tembs` the canonical CAPE Set A embeddings in `tembs/`. Run the guard
script after any cache regeneration or before any analysis that consumes
the caches:

```bash
python3 analysis/verify_caches.py
```

This guard exists because an earlier generation script (`gen_missing_caches.py`,
commit 5ca5310) inlined a divergent copy of the CAPE prompts, producing
laion/eva02 caches whose logits correlated only ~0.86 with the paper's
CAPE Set A (a ~30pp Hit@1 drop). Those caches were regenerated with the
canonical prompts; `verify_caches.py` enforces that this cannot silently
happen again. Do NOT inline prompt text in cache-generation scripts; always
load prompts from `analysis/prompts/setAB_examples.json` (set_A_original).

## Usage

These files are loaded automatically by the analysis scripts:

```bash
python analysis/paired_bootstrap.py
python analysis/cape_principle_ablation.py
python analysis/dev_test_split_eval.py
```
