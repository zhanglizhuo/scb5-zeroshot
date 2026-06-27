"""M5: Prompt-engineering baselines (CuPL + WaffleCLIP) vs. CAPE.

Compares CAPE against two literature baselines on three SCB5 datasets,
using the same OpenAI / DFN / SigLIP2 ViT-L/14 backbones already cached
in data/feature_cache/. Re-encodes ONLY the text side (image features
are reused), so the experiment is cheap.

Baselines
---------
- plain        : single template "a photo of a {cls}" (sanity check, not new)
- cape         : 3 hand-crafted spatial/action/context prompts per class
                 (already in scb5_zeroshot/cape_robustness.py CAPE_A)
- cupl         : 50 LLM-generated descriptive sentences per class
                 (Pratt et al. ICCV 2023). LLM = local ollama gemma4:26b.
- waffleclip   : 16 random-token-suffix variants per class, ensembled
                 (Roth et al. ICCV 2023, simplified zero-shot variant).

For each (backbone, dataset, prompt_strategy) we compute Hit@1 (multi-
label uses argmax-then-check-membership in the ground-truth multi-hot
vector, matching tab:main_results semantics).

Outputs
-------
results_revision/m5_baselines/m5_prompt_baselines.json
"""
import argparse
import json
import os
import random
import string
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scb5_zeroshot"))


def _load_cape_a():
    """Extract the CAPE_A dict directly from cape_robustness.py without
    importing the whole module (which has heavy side-effect imports)."""
    src = (ROOT / "scb5_zeroshot" / "cape_robustness.py").read_text()
    ns = {}
    # Locate the assignment "CAPE_A = {" and slice until matching closing brace
    start = src.find("CAPE_A = {")
    assert start != -1, "CAPE_A definition not found in cape_robustness.py"
    depth = 0
    i = src.find("{", start)
    end = i
    while end < len(src):
        c = src[end]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                end += 1
                break
        end += 1
    snippet = src[start:end]
    exec(snippet, ns)
    return ns["CAPE_A"]


CAPE_A = _load_cape_a()

CACHE = ROOT / "data" / "feature_cache"
OUT_DIR = ROOT / "results" / "revision" / "m5_baselines"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PROMPTS_DIR = OUT_DIR / "prompts"
PROMPTS_DIR.mkdir(exist_ok=True)

# ──────────────────────────────────────────────────────────────────
# Backbone specs (must match the cached features under data/feature_cache/)
# ──────────────────────────────────────────────────────────────────
BACKBONES = {
    "openai":  ("ViT-L-14", "openai",
                ROOT / "scb5_zeroshot/ckpts/clip_openai_vitl14/ViT-L-14.pt", 224),
    "dfn":     ("ViT-L-14", "dfn2b",
                ROOT / "scb5_zeroshot/ckpts/dfn_clip_vitl14/open_clip_pytorch_model.bin", 224),
    "siglip2": ("ViT-L-16-SigLIP2-256", "webli",
                ROOT / "scb5_zeroshot/ckpts/siglip_vitl16_256/open_clip_pytorch_model.bin", 256),
}

# ──────────────────────────────────────────────────────────────────
# Datasets and class lists (match cache file names)
# ──────────────────────────────────────────────────────────────────
DATASETS = {
    "teacher_behavior": {
        "name": "TeacherBehavior",
        "classes": ["guide", "answer", "On-stage interaction",
                    "blackboard-writing", "teacher", "stand", "screen", "blackBoard"],
        "multi_label": True,
    },
    "handrise_readwrite": {
        "name": "HandriseReadWrite",
        "classes": ["hand-raise", "read", "write"],
        "multi_label": False,
    },
    "bow_turnhead": {
        "name": "BowTurnHead",
        "classes": ["bow-head", "turn-head"],
        "multi_label": False,
    },
}

# Human-readable class strings for plain / waffleclip prompts
PLAIN_NAME = {
    "guide": "a teacher guiding a student",
    "answer": "a teacher answering a student's question",
    "On-stage interaction": "a teacher interacting with students on stage",
    "blackboard-writing": "a teacher writing on the blackboard",
    "teacher": "a teacher lecturing",
    "stand": "a teacher standing still",
    "screen": "a teacher pointing at a projection screen",
    "blackBoard": "a teacher pointing at the blackboard",
    "hand-raise": "a student raising their hand",
    "read": "a student reading a book",
    "write": "a student writing in a notebook",
    "bow-head": "a student with their head bowed down",
    "turn-head": "a student turning their head sideways",
}

# ──────────────────────────────────────────────────────────────────
# CuPL prompt generation via local ollama
# ──────────────────────────────────────────────────────────────────
import urllib.request

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
OLLAMA_MODEL = "gemma4:26b"  # non-reasoning or reasoning-disabled


def ollama_chat(prompt: str, temperature: float = 0.9, num_predict: int = 220) -> str:
    body = json.dumps({
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "think": False,
        "options": {"temperature": temperature, "num_predict": num_predict},
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        OLLAMA_URL, data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        d = json.loads(resp.read().decode())
    return d.get("message", {}).get("content", "") or ""


CUPL_TEMPLATES = [
    "Describe what {cls} looks like in a classroom photo. "
    "Give 5 different short visual descriptions, one per line, no numbering, no bullets.",
    "List 5 short visual cues that an image of {cls} in a classroom would contain. "
    "One per line, no numbering, no bullets, each starts with 'A photo of'.",
    "Write 5 alternative short captions for a classroom photo showing {cls}. "
    "One per line, no numbering.",
    "Imagine 5 short photo descriptions of {cls} taken in a real classroom. "
    "Different angles or lighting. One per line, no numbering.",
    "Generate 5 brief visual descriptions of {cls} suitable as image captions. "
    "Short sentences, one per line, no numbering.",
]


def gen_cupl_for_class(cls: str, descriptor: str, n_target: int = 50) -> list:
    """Generate ~n_target descriptive prompts for one class."""
    out = []
    seed_text = f'"{descriptor}"'
    for tpl in CUPL_TEMPLATES:
        prompt = tpl.format(cls=seed_text)
        try:
            txt = ollama_chat(prompt)
        except Exception as e:
            print(f"  [CuPL] {cls}: LLM call failed: {e}", flush=True)
            continue
        for line in txt.splitlines():
            line = line.strip()
            # Strip leading bullets / numbering
            line = line.lstrip("-*•").strip()
            for prefix in ("1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.", "10."):
                if line.startswith(prefix):
                    line = line[len(prefix):].strip()
                    break
            if not line or len(line) < 8 or len(line) > 200:
                continue
            if line.lower().startswith(("here are", "sure,", "okay", "ok,", "of course")):
                continue
            out.append(line)
        if len(out) >= n_target:
            break
    # de-dup preserving order
    seen, dedup = set(), []
    for s in out:
        k = s.lower()
        if k in seen:
            continue
        seen.add(k)
        dedup.append(s)
    return dedup[:n_target]


def build_cupl_prompts(datasets, n_per_class: int = 50, force_regen: bool = False):
    cache_path = PROMPTS_DIR / "cupl_prompts.json"
    if cache_path.exists() and not force_regen:
        existing = json.loads(cache_path.read_text())
    else:
        existing = {}
    needed = []
    for ds_key, info in datasets.items():
        for c in info["classes"]:
            if c not in existing or len(existing[c]) < n_per_class:
                needed.append(c)
    needed = sorted(set(needed))
    if not needed:
        print(f"[CuPL] cached prompts found for all {len(existing)} classes")
        return existing

    print(f"[CuPL] generating prompts for {len(needed)} classes via ollama {OLLAMA_MODEL}")
    for c in needed:
        descriptor = PLAIN_NAME.get(c, c)
        t0 = time.time()
        prompts = gen_cupl_for_class(c, descriptor, n_target=n_per_class)
        existing[c] = prompts
        cache_path.write_text(json.dumps(existing, indent=2))
        print(f"  [CuPL] '{c}' -> {len(prompts)} prompts ({time.time()-t0:.1f}s)", flush=True)
    return existing


# ──────────────────────────────────────────────────────────────────
# WaffleCLIP prompt construction
# ──────────────────────────────────────────────────────────────────
WAFFLE_TEMPLATES = [
    "a photo of {cls}, which has {rand}",
    "a photo of {cls}, which is {rand}",
    "a photo of a {cls}, which (has/is) {rand}",
]


def random_waffle_token(rng: random.Random, length: int = 5) -> str:
    return "".join(rng.choices(string.ascii_lowercase, k=length))


def build_waffle_prompts(classes: list, n_per_class: int = 16, seed: int = 42) -> dict:
    rng = random.Random(seed)
    out = {}
    for c in classes:
        descriptor = PLAIN_NAME.get(c, c)
        prompts = []
        for _ in range(n_per_class):
            t = rng.choice(WAFFLE_TEMPLATES)
            r = " ".join(random_waffle_token(rng, rng.randint(4, 7)) for _ in range(2))
            prompts.append(t.format(cls=descriptor, rand=r))
        out[c] = prompts
    return out


def build_plain_prompts(classes: list) -> dict:
    return {c: [f"a photo of {PLAIN_NAME.get(c, c)}"] for c in classes}


def build_cape_prompts(classes: list) -> dict:
    return {c: list(CAPE_A[c]) for c in classes}


# ──────────────────────────────────────────────────────────────────
# Text encoding via open_clip (one model load per backbone)
# ──────────────────────────────────────────────────────────────────
def load_backbone_text_encoder(backbone: str, device):
    import open_clip
    arch, pretrained, ckpt_path, _img = BACKBONES[backbone]
    ckpt_path = Path(ckpt_path)
    print(f"[{backbone}] loading {arch} from {ckpt_path.name}")

    if backbone == "openai":
        # The openai ViT-L-14.pt is a TorchScript archive; load via jit and use
        # its encode_text directly. Tokenizer can come from open_clip.
        import open_clip
        try:
            model = torch.jit.load(str(ckpt_path), map_location=device).eval()
        except Exception:
            # Fallback: open_clip auto-download with pretrained="openai"
            model = open_clip.create_model(arch, pretrained="openai", device=device)
        tokenizer = open_clip.get_tokenizer(arch)

        @torch.no_grad()
        def encode_text(texts):
            toks = tokenizer(texts).to(device)
            f = model.encode_text(toks)
            return F.normalize(f.float(), dim=-1)
        return encode_text
    elif backbone == "dfn":
        model = open_clip.create_model(arch, pretrained=None, device=device)
        sd = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(sd, strict=False)
        tokenizer = open_clip.get_tokenizer(arch)
    elif backbone == "siglip2":
        model = open_clip.create_model(arch, pretrained=None, device=device)
        sd = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(sd, strict=False)
        # SigLIP tokenizer needs ckpt dir
        try:
            tokenizer = open_clip.get_tokenizer(f"hf-hub:{ckpt_path.parent}", context_length=64)
        except Exception:
            tokenizer = open_clip.get_tokenizer(arch)
    else:
        raise ValueError(backbone)
    model.eval()

    @torch.no_grad()
    def encode_text(texts):
        toks = tokenizer(texts).to(device)
        f = model.encode_text(toks)
        return F.normalize(f.float(), dim=-1)

    return encode_text


def class_text_features(encode_text_fn, prompts_dict, classes, device, batch=64):
    feats = []
    for c in classes:
        p = prompts_dict[c]
        chunks = []
        for i in range(0, len(p), batch):
            chunks.append(encode_text_fn(p[i:i + batch]))
        f = torch.cat(chunks, dim=0)
        # Average then re-normalise (standard CLIP zero-shot recipe)
        f = F.normalize(f.mean(0, keepdim=True), dim=-1)
        feats.append(f)
    return torch.cat(feats, dim=0)  # (C, D)


# ──────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────
def hit_at_1(logits: np.ndarray, labels: np.ndarray, multi_label: bool) -> float:
    pred = logits.argmax(axis=1)
    if multi_label:
        return float(labels[np.arange(len(labels)), pred].mean())
    return float((pred == labels).mean())


def evaluate_one(backbone: str, ds_key: str, prompts: dict, encode_text_fn, device):
    info = DATASETS[ds_key]
    cache_file = CACHE / f"{backbone}_{ds_key}_validation.npz"
    d = np.load(cache_file)
    img = torch.from_numpy(d["image_features"]).to(device).float()
    img = F.normalize(img, dim=-1)
    text = class_text_features(encode_text_fn, prompts, info["classes"], device)
    logits = (img @ text.T).cpu().numpy()
    return hit_at_1(logits, d["labels"], info["multi_label"]), int(len(d["labels"]))


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--n_cupl", type=int, default=50)
    ap.add_argument("--n_waffle", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--backbones", nargs="+", default=["openai", "dfn", "siglip2"])
    ap.add_argument("--skip_cupl_gen", action="store_true",
                    help="Use cached CuPL prompts only")
    ap.add_argument("--out", default=str(OUT_DIR / "m5_prompt_baselines.json"))
    args = ap.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[m5] device={device}, backbones={args.backbones}")

    # 1) Build prompt sets
    if not args.skip_cupl_gen:
        cupl = build_cupl_prompts(DATASETS, n_per_class=args.n_cupl)
    else:
        cupl = json.loads((PROMPTS_DIR / "cupl_prompts.json").read_text())

    # Save the WaffleCLIP and Plain prompts for full reproducibility
    waffle_all = {}
    plain_all = {}
    cape_all = {}
    for ds_key, info in DATASETS.items():
        plain_all[ds_key] = build_plain_prompts(info["classes"])
        waffle_all[ds_key] = build_waffle_prompts(info["classes"],
                                                  n_per_class=args.n_waffle,
                                                  seed=args.seed)
        cape_all[ds_key] = build_cape_prompts(info["classes"])
    (PROMPTS_DIR / "waffle_prompts.json").write_text(
        json.dumps(waffle_all, indent=2))
    (PROMPTS_DIR / "plain_prompts.json").write_text(
        json.dumps(plain_all, indent=2))
    (PROMPTS_DIR / "cape_prompts.json").write_text(
        json.dumps(cape_all, indent=2))

    # 2) Loop over backbones × datasets × strategies
    rows = []
    for bb in args.backbones:
        encode_text_fn = load_backbone_text_encoder(bb, device)
        for ds_key, info in DATASETS.items():
            classes = info["classes"]
            prompt_sets = {
                "plain":      plain_all[ds_key],
                "cape":       cape_all[ds_key],
                "waffleclip": waffle_all[ds_key],
                "cupl":       {c: cupl.get(c, [PLAIN_NAME.get(c, c)]) for c in classes},
            }
            for strat, ps in prompt_sets.items():
                # Drop empty CuPL classes (back-fill with plain to avoid NaN)
                for c in classes:
                    if not ps.get(c):
                        ps[c] = [f"a photo of {PLAIN_NAME.get(c, c)}"]
                acc, n = evaluate_one(bb, ds_key, ps, encode_text_fn, device)
                avg_len = float(np.mean([len(ps[c]) for c in classes]))
                row = {
                    "backbone": bb,
                    "dataset": info["name"],
                    "strategy": strat,
                    "hit_at_1": round(acc * 100, 2),
                    "n_samples": n,
                    "avg_prompts_per_class": round(avg_len, 1),
                }
                print("  ", row, flush=True)
                rows.append(row)
        # Free the model before next backbone
        del encode_text_fn
        torch.cuda.empty_cache()

    out_path = Path(args.out)
    out_path.write_text(json.dumps({
        "config": {
            "n_cupl": args.n_cupl,
            "n_waffle": args.n_waffle,
            "seed": args.seed,
            "ollama_model": OLLAMA_MODEL,
            "backbones": args.backbones,
        },
        "rows": rows,
        "timestamp": int(time.time()),
    }, indent=2))
    print(f"[m5] saved -> {out_path}")


if __name__ == "__main__":
    main()
