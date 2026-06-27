#!/usr/bin/env bash
##############################################################################
# [LEGACY] Use reproduce_paper.sh instead for canonical entry.
# master_benchmark.sh — Full pipeline: download datasets + models + run all
#
# Usage:  nohup bash master_benchmark.sh &> master_benchmark.log &
# Then:   tail -f master_benchmark.log   (from another terminal)
#
# Safe to run with SSH disconnected (nohup + disown)
##############################################################################

set -euo pipefail

export HF_ENDPOINT="https://hf-mirror.com"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"

ROOT="$(cd "$(dirname "$0")" && pwd)"
EVAL_DIR="$ROOT/scb5_zeroshot"
CKPT_DIR="$EVAL_DIR/ckpts"
DATA_DIR="$ROOT/datasets_scb"
DEFAULT_PYTHON_BIN="/usr/bin/python3.8"
PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON_BIN}"

if [ -n "${PYTHON_ENV_ACTIVATE:-}" ]; then
    # Optional activation hook for environments that require shell activation.
    # By default the script runs directly with the configured interpreter.
    source "$PYTHON_ENV_ACTIVATE"
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1 && [ ! -x "$PYTHON_BIN" ]; then
    echo "Configured PYTHON_BIN is not available: $PYTHON_BIN" >&2
    exit 1
fi

LOG_PREFIX="[$(date '+%Y-%m-%d %H:%M:%S')]"
log() { echo "$LOG_PREFIX $*"; }
log "Using Python interpreter: $PYTHON_BIN"

##############################################################################
# PHASE 1: Download additional SCB datasets via hf-mirror
##############################################################################
log "===== PHASE 1: Download datasets ====="

mkdir -p "$DATA_DIR"

dataset_has_yolo_layout() {
    local dest="$1"
    for candidate in \
        "$dest/images/train" \
        "$dest/images/val" \
        "$dest/images/test" \
        "$dest/images/valid" \
        "$dest/train/images" \
        "$dest/val/images" \
        "$dest/test/images" \
        "$dest/valid/images"; do
        if [ -d "$candidate" ]; then
            return 0
        fi
    done
    return 1
}

extract_dataset_archives() {
    local name="$1"
    local dest="$2"

    if dataset_has_yolo_layout "$dest"; then
        return 0
    fi

    if ! find "$dest" -maxdepth 1 -type f \( -name "*.zip" -o -name "*.tar.gz" -o -name "*.tgz" \) | grep -q .; then
        return 0
    fi

    log "  [EXTRACT] $name archives in $dest ..."
    DATASET_DEST="$dest" "$PYTHON_BIN" << 'PYEOF'
import os
import shutil
import tarfile
import zipfile
from pathlib import Path

dest = Path(os.environ["DATASET_DEST"])

def has_layout(base: Path) -> bool:
    candidates = [
        base / "images" / "train",
        base / "images" / "val",
        base / "images" / "test",
        base / "images" / "valid",
        base / "train" / "images",
        base / "val" / "images",
        base / "test" / "images",
        base / "valid" / "images",
    ]
    return any(path.is_dir() for path in candidates)

archives = sorted(dest.glob("*.zip")) + sorted(dest.glob("*.tar.gz")) + sorted(dest.glob("*.tgz"))
for archive in archives:
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(dest)
    else:
        with tarfile.open(archive) as handle:
            handle.extractall(dest)

if not has_layout(dest):
    nested_candidates = sorted(
        path for path in dest.rglob("*")
        if path.is_dir() and path.name not in {"exp", "__MACOSX"} and has_layout(path)
    )
    for subdir in nested_candidates:
        for child in subdir.iterdir():
            target = dest / child.name
            if target.exists():
                continue
            shutil.move(str(child), str(target))
        if has_layout(dest):
            break

print(f"Extraction complete: {dest}")
PYEOF

    if dataset_has_yolo_layout "$dest"; then
        log "  [OK] $name extracted into YOLO layout"
    else
        log "  [WARN] $name still has no detectable YOLO split after extraction"
    fi
}

download_dataset() {
    local name="$1"
    local hf_path="$2"
    local dest="$DATA_DIR/$name"
    
    if [ -d "$dest" ] && [ "$(ls -A "$dest" 2>/dev/null)" ]; then
        log "  [SKIP] $name already exists at $dest"
        extract_dataset_archives "$name" "$dest"
        return 0
    fi
    
    log "  [DOWNLOAD] $name from $hf_path ..."
    mkdir -p "$dest"
    
    # Use huggingface-cli with mirror
    "$PYTHON_BIN" -c "
from huggingface_hub import snapshot_download
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
snapshot_download(
    repo_id='wintonYF/SCB-Dataset',
    repo_type='dataset',
    allow_patterns='${hf_path}/*',
    local_dir='${DATA_DIR}/_hf_cache',
    local_dir_use_symlinks=False,
)
print('Download complete: $name')
" 2>&1 || {
        log "  [WARN] huggingface_hub download failed for $name, trying wget..."
        # Fallback: wget individual files
        return 1
    }
    
    # Move files to destination
    local src="$DATA_DIR/_hf_cache/$hf_path"
    if [ -d "$src" ]; then
        cp -r "$src"/* "$dest/" 2>/dev/null || mv "$src"/* "$dest/" 2>/dev/null || true
        log "  [OK] $name -> $dest"
        extract_dataset_archives "$name" "$dest"
    else
        log "  [WARN] Source dir not found: $src"
        # Try to find the data
        find "$DATA_DIR/_hf_cache" -name "*.zip" -o -name "*.tar.gz" 2>/dev/null | head -5
    fi
}

# Dataset 1: Teacher Behavior (already have it, just symlink)
TB_SRC="$DATA_DIR/SCB5_TeacherBehavior"
if [ -d "$TB_SRC" ] && [ ! -L "$DATA_DIR/SCB5_TeacherBehavior" ]; then
    log "  [OK] SCB5_TeacherBehavior exists at $TB_SRC"
fi

# Dataset 2: Handrise-Read-Write
download_dataset "SCB5_HandriseReadWrite" "SCB5-Handrise-Read-write-2024-9-17"

# Dataset 3: BowTurnHead
download_dataset "SCB_BowTurnHead" "SCB_BowTurnHead_20250509"

# Dataset 4: Discuss
download_dataset "SCB5_Discuss" "SCB5-Discuss-2024-9-17"

log "Datasets download phase complete."

##############################################################################
# PHASE 2: Download or validate model checkpoints
##############################################################################
log "===== PHASE 2: Download model checkpoints ====="

mkdir -p "$CKPT_DIR/clip_openai_vitl14"
if [ -f "$CKPT_DIR/clip_openai_vitl14/ViT-L-14.pt" ]; then
    log "  [SKIP] OpenAI CLIP ViT-L-14 already exists"
else
    log "  [DOWNLOAD] OpenAI CLIP ViT-L-14 ..."
    "$PYTHON_BIN" -c "
import clip
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
clip.load('ViT-L/14', device='cpu', download_root='$CKPT_DIR/clip_openai_vitl14')
print('Saved OpenAI CLIP to $CKPT_DIR/clip_openai_vitl14')
" 2>&1 || log "  [WARN] OpenAI CLIP download failed"
fi

mkdir -p "$CKPT_DIR/laion_vitl14"
if [ -f "$CKPT_DIR/laion_vitl14/open_clip_pytorch_model.bin" ]; then
    log "  [SKIP] LAION CLIP ViT-L-14 already exists"
else
    log "  [DOWNLOAD] LAION CLIP ViT-L-14 (laion2b_s32b_b82k) ..."
    "$PYTHON_BIN" -c "
import open_clip, os, torch
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
model, _, _ = open_clip.create_model_and_transforms(
    'ViT-L-14', pretrained='laion2b_s32b_b82k'
)
dst = '$CKPT_DIR/laion_vitl14/open_clip_pytorch_model.bin'
torch.save(model.state_dict(), dst)
print(f'Saved LAION ViT-L-14 to {dst}')
" 2>&1 || log "  [WARN] LAION download failed"
fi

mkdir -p "$CKPT_DIR/siglip_vitl16_256"
if [ -f "$CKPT_DIR/siglip_vitl16_256/open_clip_pytorch_model.bin" ] && [ -f "$CKPT_DIR/siglip_vitl16_256/tokenizer.json" ]; then
    log "  [SKIP] SigLIP ViT-L-16-256 already exists"
else
    log "  [DOWNLOAD] SigLIP ViT-L-16-SigLIP2-256 assets ..."
    "$PYTHON_BIN" -c "
from huggingface_hub import snapshot_download
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
snapshot_download(
    repo_id='timm/ViT-L-16-SigLIP2-256',
    local_dir='$CKPT_DIR/siglip_vitl16_256',
    local_dir_use_symlinks=False,
)
print('Saved SigLIP assets to $CKPT_DIR/siglip_vitl16_256')
" 2>&1 || log "  [WARN] SigLIP assets download failed"
fi

mkdir -p "$CKPT_DIR/eva02_clip_vitl14"

if [ -f "$CKPT_DIR/eva02_clip_vitl14/open_clip_pytorch_model.bin" ]; then
    log "  [SKIP] EVA02-CLIP-L-14 already exists"
else
    log "  [DOWNLOAD] EVA02-CLIP-L-14 (merged2b_s4b_b131k) ..."
    "$PYTHON_BIN" -c "
import open_clip, os, shutil, torch
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# This will download via the mirror
model, _, preprocess = open_clip.create_model_and_transforms(
    'EVA02-L-14', pretrained='merged2b_s4b_b131k'
)
# Save state dict locally
dst = '$CKPT_DIR/eva02_clip_vitl14/open_clip_pytorch_model.bin'
torch.save(model.state_dict(), dst)
print(f'Saved EVA02-L-14 to {dst}')
" 2>&1 || log "  [WARN] EVA02 download failed"
fi

# Also download DFN-CLIP ViT-L-14
mkdir -p "$CKPT_DIR/dfn_clip_vitl14"
if [ -f "$CKPT_DIR/dfn_clip_vitl14/open_clip_pytorch_model.bin" ]; then
    log "  [SKIP] DFN-CLIP-L-14 already exists"
else
    log "  [DOWNLOAD] DFN-CLIP ViT-L-14 (dfn2b) ..."
    "$PYTHON_BIN" -c "
import open_clip, os, torch
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-L-14', pretrained='dfn2b'
)
dst = '$CKPT_DIR/dfn_clip_vitl14/open_clip_pytorch_model.bin'
torch.save(model.state_dict(), dst)
print(f'Saved DFN-L-14 to {dst}')
" 2>&1 || log "  [WARN] DFN download failed"
fi

log "Model download phase complete."

##############################################################################
# PHASE 3: Explore dataset structures
##############################################################################
log "===== PHASE 3: Inspect downloaded datasets ====="

"$PYTHON_BIN" << 'PYEOF'
import os, sys
from pathlib import Path
from collections import Counter

DATA_DIR = Path(os.environ.get("DATA_DIR", str(Path("$ROOT").parent / "datasets_scb")))

for ds_name in sorted(os.listdir(DATA_DIR)):
    ds_path = DATA_DIR / ds_name
    if not ds_path.is_dir():
        continue
    print(f"\n{'='*60}")
    print(f"Dataset: {ds_name}")
    print(f"  Path: {ds_path}")
    
    # Look for images/labels dirs (YOLO format)
    for split in ["val", "test", "valid", "train"]:
        img_dir = None
        lbl_dir = None
        for candidate in [
            ds_path / "images" / split,
            ds_path / split / "images",
            ds_path / split,
        ]:
            if candidate.is_dir():
                files = list(candidate.glob("*.jpg")) + list(candidate.glob("*.png"))
                if files:
                    img_dir = candidate
                    break
        for candidate in [
            ds_path / "labels" / split,
            ds_path / split / "labels",
        ]:
            if candidate.is_dir():
                lbl_dir = candidate
                break
        
        if img_dir:
            n_imgs = len(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
            print(f"  Split [{split}]: {n_imgs} images at {img_dir}")
            
            if lbl_dir:
                # Count class distribution
                class_ids = []
                for lf in sorted(lbl_dir.glob("*.txt")):
                    for line in lf.read_text().splitlines():
                        line = line.strip()
                        if line:
                            class_ids.append(int(line.split()[0]))
                dist = Counter(class_ids)
                print(f"    Labels: {lbl_dir}")
                print(f"    Classes: {dict(sorted(dist.items()))}")
                print(f"    Total annotations: {len(class_ids)}")
    
    # Check for data.yaml
    for yaml_name in ["data.yaml", "dataset.yaml"]:
        yaml_path = ds_path / yaml_name
        if yaml_path.exists():
            print(f"  Config: {yaml_path}")
            print("  " + yaml_path.read_text()[:500])

print("\nDone inspecting datasets.")
PYEOF

##############################################################################
# PHASE 4: Run comprehensive benchmark experiments
##############################################################################
log "===== PHASE 4: Run benchmark experiments ====="

cd "$EVAL_DIR"

"$PYTHON_BIN" << 'PYEOF'
"""
benchmark_runner.py — Run all models × all datasets × all prompt strategies.
Inline script to avoid import issues.
"""
import contextlib
import sys, os, json, time, logging
import numpy as np
from pathlib import Path
from collections import Counter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

import torch
import open_clip
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import (
    f1_score, balanced_accuracy_score, confusion_matrix,
    precision_score, recall_score, average_precision_score,
)

# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────
ROOT = Path(os.getcwd()).parent
DATA_DIR = ROOT.parent / "datasets_scb"
CKPT_DIR = Path("ckpts")
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "results"))
RESULTS_DIR.mkdir(exist_ok=True)

GPU = int(os.environ.get("GPU_INDEX", "0"))
BATCH = int(os.environ.get("BATCH_SIZE", "64"))
device = torch.device(f"cuda:{GPU}" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────
# DATASET CONFIGS (will be populated dynamically)
# ─────────────────────────────────────────────────────────
DATASET_CONFIGS = {
    "SCB5_TeacherBehavior": {
        "class_names": ["guide", "answer", "On-stage interaction", "blackboard-writing",
                        "teacher", "stand", "screen", "blackBoard"],
        "class_descriptions": {
            "guide": "guiding or helping students individually",
            "answer": "answering a student's question",
            "On-stage interaction": "interacting with students in front of the class",
            "blackboard-writing": "writing on a blackboard with chalk",
            "teacher": "explaining concepts while lecturing",
            "stand": "standing still without interacting",
            "screen": "pointing to or presenting slides on a screen",
            "blackBoard": "pointing at or referring to the blackboard",
        },
        "cape_prompts": {
            "guide": ["a teacher guiding a student one-on-one",
                       "a teacher helping a student at their desk",
                       "a teacher walking among students and offering guidance"],
            "answer": ["a teacher answering a student's question",
                        "a teacher responding to a raised hand in class",
                        "a student asking a question and the teacher replying"],
            "On-stage interaction": ["a teacher interacting with students at the front of the classroom",
                                      "a teacher engaging with students on the podium",
                                      "a teacher and students having a discussion in front of the class"],
            "blackboard-writing": ["a teacher writing on a blackboard with chalk",
                                    "a hand writing equations on a chalkboard",
                                    "a teacher's back while writing on the blackboard"],
            "teacher": ["a teacher standing and explaining a concept",
                         "a teacher giving a lecture at the podium",
                         "a teacher talking to the class while standing"],
            "stand": ["a person standing still in a classroom",
                       "a teacher standing at the front without interacting",
                       "a teacher standing idle near the podium"],
            "screen": ["a teacher pointing at a projection screen",
                        "a teacher presenting slides on a screen",
                        "a screen displaying a presentation in a classroom"],
            "blackBoard": ["a teacher pointing at the blackboard",
                            "a teacher referring to content on the blackboard",
                            "a blackboard with writing visible in a classroom"],
        },
    },
}

# We'll auto-discover new datasets' class names from labels
def discover_dataset_classes(ds_path):
    """Read labels to find unique class IDs."""
    label_dirs = []
    for split in ["val", "test", "valid"]:
        for candidate in [ds_path / "labels" / split, ds_path / split / "labels"]:
            if candidate.is_dir():
                label_dirs.append((split, candidate))
    
    if not label_dirs:
        # fallback: train
        for candidate in [ds_path / "labels" / "train", ds_path / "train" / "labels"]:
            if candidate.is_dir():
                label_dirs.append(("train", candidate))
    
    all_ids = set()
    for split, ldir in label_dirs:
        for lf in ldir.glob("*.txt"):
            for line in lf.read_text().splitlines():
                line = line.strip()
                if line:
                    all_ids.add(int(line.split()[0]))
    return sorted(all_ids)


# Default class name mappings for known datasets
KNOWN_CLASS_MAPS = {
    "SCB5_HandriseReadWrite": {
        0: "hand-raise", 1: "read", 2: "write",
    },
    "SCB_BowTurnHead": {
        0: "bow-head", 1: "turn-head",
    },
    "SCB5_Discuss": {
        0: "discuss",
    },
}

# Default CAPE prompts for each class across datasets
DEFAULT_CAPE_TEMPLATES = {
    "hand-raise": [
        "a student raising their hand in a classroom",
        "a student with arm raised to ask a question",
        "a student raising hand to participate in class",
    ],
    "read": [
        "a student reading a textbook at their desk",
        "a student looking down at a book while reading",
        "a student engaged in reading study materials",
    ],
    "write": [
        "a student writing in a notebook",
        "a student taking notes with a pen",
        "a student writing at their desk in class",
    ],
    "bow-head": [
        "a student with their head bowed down",
        "a student looking down at their phone or desk",
        "a student with lowered head not paying attention",
    ],
    "turn-head": [
        "a student turning their head to look sideways",
        "a student looking away from the teacher",
        "a student turning around in their seat",
    ],
    "discuss": [
        "students discussing together in a group",
        "students talking to each other in class",
        "a group of students having a discussion at their desks",
    ],
}


# ─────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────
_model_cache = {}

def load_model(model_key):
    """Load and cache a model."""
    if model_key in _model_cache:
        return _model_cache[model_key]
    
    log.info(f"Loading model: {model_key}")
    tokenizer = None
    
    if model_key == "clip":
        ckpt_dir = CKPT_DIR / "clip_openai_vitl14"
        ckpt = ckpt_dir / "ViT-L-14.pt"
        if not ckpt.exists():
            log.warning(f"OpenAI CLIP checkpoint not found: {ckpt}, skipping")
            return None
        import clip as openai_clip
        model, preprocess = openai_clip.load("ViT-L/14", device=device, download_root=str(ckpt_dir))
    elif model_key == "laion":
        arch = "ViT-L-14"
        ckpt = CKPT_DIR / "laion_vitl14" / "open_clip_pytorch_model.bin"
        if not ckpt.exists():
            log.warning(f"LAION checkpoint not found: {ckpt}, skipping")
            return None
        model = open_clip.create_model(arch, pretrained=None)
        sd = torch.load(str(ckpt), map_location="cpu", weights_only=True)
        model.load_state_dict(sd, strict=True)
        tokenizer = open_clip.get_tokenizer(arch)
        preprocess = open_clip.image_transform(224, is_train=False)
    elif model_key == "siglip":
        arch = "ViT-L-16-SigLIP2-256"
        ckpt = CKPT_DIR / "siglip_vitl16_256" / "open_clip_pytorch_model.bin"
        if not ckpt.exists():
            log.warning(f"SigLIP checkpoint not found: {ckpt}, skipping")
            return None
        model = open_clip.create_model(arch, pretrained=None)
        sd = torch.load(str(ckpt), map_location="cpu", weights_only=True)
        model.load_state_dict(sd, strict=True)
        tokenizer = open_clip.get_tokenizer(
            f"hf-hub:{CKPT_DIR / 'siglip_vitl16_256'}",
            context_length=64
        )
        preprocess = open_clip.image_transform(256, is_train=False)
    elif model_key == "eva02":
        arch = "EVA02-L-14"
        ckpt = CKPT_DIR / "eva02_clip_vitl14" / "open_clip_pytorch_model.bin"
        if not ckpt.exists():
            log.warning(f"EVA02 checkpoint not found: {ckpt}, skipping")
            return None
        model = open_clip.create_model(arch, pretrained=None)
        sd = torch.load(str(ckpt), map_location="cpu", weights_only=True)
        model.load_state_dict(sd, strict=True)
        tokenizer = open_clip.get_tokenizer(arch)
        preprocess = open_clip.image_transform(224, is_train=False)
    elif model_key == "dfn":
        arch = "ViT-L-14"
        ckpt = CKPT_DIR / "dfn_clip_vitl14" / "open_clip_pytorch_model.bin"
        if not ckpt.exists():
            log.warning(f"DFN checkpoint not found: {ckpt}, skipping")
            return None
        model = open_clip.create_model(arch, pretrained=None)
        sd = torch.load(str(ckpt), map_location="cpu", weights_only=True)
        model.load_state_dict(sd, strict=True)
        tokenizer = open_clip.get_tokenizer(arch)
        preprocess = open_clip.image_transform(224, is_train=False)
    else:
        log.error(f"Unknown model: {model_key}")
        return None
    
    model = model.eval().to(device)

    def autocast_context():
        if device.type == "cuda":
            return torch.amp.autocast("cuda")
        return contextlib.nullcontext()
    
    def encode_image(imgs):
        with torch.no_grad(), autocast_context():
            return model.encode_image(imgs)
    
    if model_key == "clip":
        import clip as openai_clip

        def encode_text(texts):
            tokens = openai_clip.tokenize(texts, truncate=True).to(device)
            with torch.no_grad(), autocast_context():
                return model.encode_text(tokens)
    else:
        def encode_text(texts):
            tokens = tokenizer(texts).to(device)
            with torch.no_grad(), autocast_context():
                return model.encode_text(tokens)
    
    md = {
        "model": model,
        "preprocess": preprocess,
        "tokenizer": tokenizer,
        "encode_image": encode_image,
        "encode_text": encode_text,
    }
    _model_cache[model_key] = md
    return md


# ─────────────────────────────────────────────────────────
# DATASET LOADING (YOLO format)
# ─────────────────────────────────────────────────────────
def load_yolo_dataset(ds_path, split="val"):
    """Load YOLO-format dataset, return list of (img_path, set_of_class_ids)."""
    img_dir = None
    lbl_dir = None
    
    for candidate_img, candidate_lbl in [
        (ds_path / "images" / split, ds_path / "labels" / split),
        (ds_path / split / "images", ds_path / split / "labels"),
    ]:
        if candidate_img.is_dir() and candidate_lbl.is_dir():
            img_dir = candidate_img
            lbl_dir = candidate_lbl
            break
    
    if not img_dir:
        log.warning(f"No {split} split found in {ds_path}")
        return []
    
    samples = []
    for img_file in sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png")):
        lbl_file = lbl_dir / (img_file.stem + ".txt")
        if not lbl_file.exists():
            continue
        class_ids = set()
        for line in lbl_file.read_text().splitlines():
            line = line.strip()
            if line:
                class_ids.add(int(line.split()[0]))
        if class_ids:
            samples.append((str(img_file), class_ids))
    
    return samples


# ─────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────
def build_text_features(encode_text, prompts_dict, class_names):
    """Build text features from prompts dict."""
    all_feats = []
    for cls in class_names:
        prompts = prompts_dict[cls]
        feats = encode_text(prompts)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        mean_feat = feats.mean(dim=0)
        mean_feat = mean_feat / mean_feat.norm()
        all_feats.append(mean_feat)
    return torch.stack(all_feats)  # [num_classes, D]


def evaluate(samples, model_dict, text_feats, batch_size=64):
    """Evaluate zero-shot, return Hit@1, Hit@3, y_true, y_pred."""
    enc_img = model_dict["encode_image"]
    preproc = model_dict["preprocess"]
    
    hit1 = hit3 = total = 0
    y_true_all, y_pred_all = [], []
    buf_imgs, buf_gts = [], []
    
    def flush():
        nonlocal hit1, hit3, total
        if not buf_imgs:
            return
        imgs = torch.stack(buf_imgs).to(device)
        img_feats = enc_img(imgs)
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        sims = img_feats @ text_feats.T
        
        for i, gt in enumerate(buf_gts):
            p1 = sims[i].argmax().item()
            p3 = sims[i].topk(min(3, sims.shape[1])).indices.tolist()
            gt_label = min(gt)
            y_true_all.append(gt_label)
            y_pred_all.append(p1)
            if p1 in gt:
                hit1 += 1
            if any(p in gt for p in p3):
                hit3 += 1
            total += 1
        buf_imgs.clear()
        buf_gts.clear()
    
    for img_path, gt in tqdm(samples, desc="Eval", leave=False):
        try:
            img = preproc(Image.open(img_path).convert("RGB"))
        except Exception as e:
            continue
        buf_imgs.append(img)
        buf_gts.append(gt)
        if len(buf_imgs) >= batch_size:
            flush()
    flush()
    
    return {
        "total": total,
        "hit1": round(hit1 / total * 100, 2) if total else 0,
        "hit3": round(hit3 / total * 100, 2) if total else 0,
        "y_true": y_true_all,
        "y_pred": y_pred_all,
    }


def compute_metrics(y_true, y_pred, num_classes):
    """Compute comprehensive metrics."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    sl_acc = np.trace(cm) / cm.sum() * 100 if cm.sum() > 0 else 0
    
    return {
        "single_label_acc": round(sl_acc, 2),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro", zero_division=0) * 100, 2),
        "weighted_f1": round(f1_score(y_true, y_pred, average="weighted", zero_division=0) * 100, 2),
        "balanced_accuracy": round(balanced_accuracy_score(y_true, y_pred) * 100, 2),
        "confusion_matrix": cm.tolist(),
    }


# ─────────────────────────────────────────────────────────
# PROMPT STRATEGIES
# ─────────────────────────────────────────────────────────
def build_prompt_groups(class_names, class_descriptions):
    """Build standard prompt groups for a dataset."""
    groups = {}
    
    # Label-only
    groups["label_only"] = {cls: [class_descriptions.get(cls, cls)] for cls in class_names}
    
    # Simple
    groups["simple"] = {cls: [f"a photo of {class_descriptions.get(cls, cls)}"] for cls in class_names}
    
    # Action
    groups["action"] = {cls: [f"a teacher is {class_descriptions.get(cls, cls)}"] 
                        if any(w in class_descriptions.get(cls, "") for w in ["teacher", "stand", "lectur", "guid", "answer", "interact", "writ", "point"])
                        else [f"a student is {class_descriptions.get(cls, cls)}"]
                        for cls in class_names}
    
    # Detailed
    groups["detailed"] = {}
    for cls in class_names:
        desc = class_descriptions.get(cls, cls)
        groups["detailed"][cls] = [
            f"a classroom scene where someone is {desc}",
            f"a person is {desc} during a class",
        ]
    
    return groups


# ─────────────────────────────────────────────────────────
# MAIN BENCHMARK
# ─────────────────────────────────────────────────────────
def run_benchmark():
    all_results = {
        "timestamp": int(time.time()),
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "datasets": {},
    }
    
    models_to_test = [
        key.strip()
        for key in os.environ.get("MODEL_KEYS", "clip,laion,siglip,eva02,dfn").split(",")
        if key.strip()
    ]
    
    # Discover datasets
    datasets = {}
    
    # 1) Teacher Behavior (predefined config)
    tb_path = DATA_DIR / "SCB5_TeacherBehavior"
    if tb_path.exists():
        cfg = DATASET_CONFIGS["SCB5_TeacherBehavior"]
        samples = load_yolo_dataset(tb_path, "val")
        if samples:
            datasets["SCB5_TeacherBehavior"] = {
                "path": tb_path,
                "samples": samples,
                "config": cfg,
                "split": "val",
            }
            log.info(f"  SCB5_TeacherBehavior: {len(samples)} samples")
    
    # 2) Auto-discover other datasets
    for ds_name, class_map in KNOWN_CLASS_MAPS.items():
        ds_path = DATA_DIR / ds_name
        if not ds_path.exists():
            log.warning(f"  {ds_name} not found at {ds_path}, skipping")
            continue
        
        # Try val, test, valid splits
        samples = None
        used_split = None
        for split in ["val", "test", "valid"]:
            samples = load_yolo_dataset(ds_path, split)
            if samples:
                used_split = split
                break
        
        if not samples:
            # Try using train as eval (if no val exists)
            samples = load_yolo_dataset(ds_path, "train")
            used_split = "train"
        
        if not samples:
            log.warning(f"  {ds_name}: no samples found, skipping")
            continue
        
        class_names = [class_map[i] for i in sorted(class_map.keys())]
        class_descriptions = {cn: cn.replace("-", " ") for cn in class_names}
        cape_prompts = {cn: DEFAULT_CAPE_TEMPLATES.get(cn, [f"a photo of {cn}"]) for cn in class_names}
        
        datasets[ds_name] = {
            "path": ds_path,
            "samples": samples,
            "config": {
                "class_names": class_names,
                "class_descriptions": class_descriptions,
                "cape_prompts": cape_prompts,
            },
            "split": used_split,
        }
        log.info(f"  {ds_name}: {len(samples)} samples (split={used_split})")
    
    log.info(f"\nTotal datasets: {len(datasets)}")
    log.info(f"Models to test: {models_to_test}")
    
    # Run experiments for each dataset × model × prompt
    for ds_name, ds_info in datasets.items():
        log.info(f"\n{'='*60}")
        log.info(f"DATASET: {ds_name} ({len(ds_info['samples'])} samples)")
        log.info(f"{'='*60}")
        
        cfg = ds_info["config"]
        class_names = cfg["class_names"]
        num_classes = len(class_names)
        samples = ds_info["samples"]
        
        # Build prompt groups
        prompt_groups = build_prompt_groups(class_names, cfg["class_descriptions"])
        prompt_groups["cape"] = cfg["cape_prompts"]
        
        ds_results = {
            "num_samples": len(samples),
            "num_classes": num_classes,
            "class_names": class_names,
            "split": ds_info["split"],
            "experiments": [],
        }
        
        # Compute majority baseline
        ml = [s[1] for s in samples]
        majority_hits = {}
        for i in range(num_classes):
            hit = sum(1 for labels in ml if i in labels) / len(ml) * 100
            majority_hits[class_names[i]] = round(hit, 2)
        ds_results["majority_baseline"] = majority_hits
        ds_results["majority_best"] = max(majority_hits.values())
        log.info(f"  Majority baseline best: {max(majority_hits.values())}%")
        
        for model_key in models_to_test:
            md = load_model(model_key)
            if md is None:
                log.warning(f"  Skipping model {model_key} (not available)")
                continue
            
            for prompt_name, prompts_dict in prompt_groups.items():
                log.info(f"  [{model_key}] [{prompt_name}] ...")
                
                try:
                    text_feats = build_text_features(md["encode_text"], prompts_dict, class_names)
                    result = evaluate(samples, md, text_feats, batch_size=BATCH)
                    metrics = compute_metrics(result["y_true"], result["y_pred"], num_classes)
                    
                    exp_result = {
                        "model": model_key,
                        "prompt": prompt_name,
                        "hit1": result["hit1"],
                        "hit3": result["hit3"],
                        "single_label_acc": metrics["single_label_acc"],
                        "macro_f1": metrics["macro_f1"],
                        "weighted_f1": metrics["weighted_f1"],
                        "balanced_accuracy": metrics["balanced_accuracy"],
                        "confusion_matrix": metrics["confusion_matrix"],
                    }
                    ds_results["experiments"].append(exp_result)
                    
                    log.info(f"    Hit@1={result['hit1']:.2f}% | SL_Acc={metrics['single_label_acc']:.2f}% | MF1={metrics['macro_f1']:.2f}%")
                    
                except Exception as e:
                    log.error(f"    ERROR: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Free GPU memory between models for large models
            if model_key in ("eva02", "dfn"):
                del _model_cache[model_key]
                torch.cuda.empty_cache()
        
        all_results["datasets"][ds_name] = ds_results
        
        # Save intermediate results after each dataset
        out_path = RESULTS_DIR / f"benchmark_{int(time.time())}.json"
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False,
                      default=lambda o: int(o) if isinstance(o, np.integer) else float(o) if isinstance(o, np.floating) else str(o))
        log.info(f"  Intermediate save: {out_path}")
    
    # Final save
    final_path = RESULTS_DIR / f"benchmark_final_{int(time.time())}.json"
    with open(final_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False,
                  default=lambda o: int(o) if isinstance(o, np.integer) else float(o) if isinstance(o, np.floating) else str(o))
    log.info(f"\n{'='*60}")
    log.info(f"ALL DONE. Final results: {final_path}")
    log.info(f"{'='*60}")

    # Print summary table
    print("\n\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    for ds_name, ds_res in all_results["datasets"].items():
        print(f"\n--- {ds_name} ({ds_res['num_samples']} samples, {ds_res['num_classes']} classes) ---")
        print(f"  Majority baseline: {ds_res['majority_best']:.1f}%")
        print(f"  {'Model':<10} {'Prompt':<12} {'Hit@1':>7} {'SL_Acc':>7} {'MF1':>7}")
        for exp in ds_res["experiments"]:
            print(f"  {exp['model']:<10} {exp['prompt']:<12} {exp['hit1']:>6.2f}% {exp['single_label_acc']:>6.2f}% {exp['macro_f1']:>6.2f}%")
    print("="*80)


if __name__ == "__main__":
    run_benchmark()
PYEOF

log "===== ALL PHASES COMPLETE ====="
log "Results saved in: $EVAL_DIR/results/"
