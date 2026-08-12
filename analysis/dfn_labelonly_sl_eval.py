"""Compute DFN label-only single-label diagnostics on HandriseReadWrite.

Purpose
-------
The Table 6 ``best model-prompt configuration per sub-dataset'' row for
HandriseReadWrite was changed from ``LAION + action'' (which used the
subject-mismatched ``a teacher is {class}'' template; see Section 4.5.6/E14)
to ``DFN + label-only''. The Hit@1 value (84.50) is taken from Table 7.
This script computes the two remaining columns---single-label accuracy
(SL Acc) and Macro-F1 under primary-label assignment---for the
DFN label-only configuration.

Inputs
------
- data/feature_cache/tembs/dfn_handrise_readwrite_labelonly_tembs.npz
  (text embeddings of the three label-only class descriptions, encoded with
  open_clip 2.24.0, DFN-2B ViT-L/14 official weights, L2-normalized)
- data/feature_cache/dfn_handrise_readwrite_validation.npz
  (image features and primary labels for the 1671-image validation split)

Outputs
-------
- results/revision/dfn_labelonly_hrw_sl_diag.json

Results (seed-independent, deterministic):
  SL Acc  = 78.93 %
  Macro-F1 = 55.31 %
  Hit@1 (single-label check) = 78.93 %

Environment
-----------
- Text embeddings generated on a V100 GPU with the model checkpoint
  ``apple/DFN2B-CLIP-ViT-L-14'' (open_clip 2.24.0); the metric computation
  below is pure NumPy/sklearn and runs anywhere.
"""
import json
import os

import numpy as np
from sklearn.metrics import f1_score

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def main():
    tembs_path = os.path.join(
        ROOT, "data", "feature_cache", "tembs",
        "dfn_handrise_readwrite_labelonly_tembs.npz",
    )
    cache_path = os.path.join(
        ROOT, "data", "feature_cache",
        "dfn_handrise_readwrite_validation.npz",
    )

    tembs = np.load(tembs_path)
    text_emb = tembs["text_embeddings"].astype(np.float64)
    prompts = [str(p) for p in tembs["prompts"]]

    cache = np.load(cache_path)
    img_feats = cache["image_features"].astype(np.float64)
    labels = cache["labels"].astype(np.int64)

    img_feats /= np.linalg.norm(img_feats, axis=1, keepdims=True)
    text_emb /= np.linalg.norm(text_emb, axis=1, keepdims=True)

    logits = img_feats @ text_emb.T
    pred = np.argmax(logits, axis=1)

    sl_acc = float(np.mean(pred == labels) * 100.0)
    macro_f1 = float(f1_score(labels, pred, average="macro") * 100.0)
    hit1_sl = float(
        np.mean([int(pred[i] == labels[i]) for i in range(len(labels))])
        * 100.0
    )

    out = {
        "purpose": "DFN label-only single-label diagnostics on HandriseReadWrite "
                   "(Table 6 row for HandriseReadWrite; Hit@1 84.50 from Table 7)",
        "model": "DFN-CLIP (DFN-2B, ViT-L/14, open_clip 2.24.0)",
        "dataset": "SCB5_HandriseReadWrite",
        "n_val": int(len(labels)),
        "prompts": prompts,
        "sl_accuracy_pct": round(sl_acc, 2),
        "macro_f1_pct": round(macro_f1, 2),
        "hit1_single_label_pct": round(hit1_sl, 2),
    }

    out_path = os.path.join(
        ROOT, "results", "revision", "dfn_labelonly_hrw_sl_diag.json",
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
