#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
SCB_DIR="$ROOT"
PAPER_DIR="$ROOT/paper"

MODE="quick"
GPU="0"
PYTHON_BIN="${PYTHON_BIN:-python}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --gpu)
      GPU="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: bash reproduce_paper.sh [--mode quick|full] [--gpu 0] [--python python]" >&2
      exit 1
      ;;
  esac
done

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU"

echo "[INFO] ROOT=$ROOT"
echo "[INFO] MODE=$MODE"
echo "[INFO] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[INFO] PYTHON_BIN=$PYTHON_BIN"

run_quick() {
  echo "[STEP] Quick verification: import check"
  "$PYTHON_BIN" -c "import torch, open_clip, numpy, sklearn, PIL; print('Import check: OK')"

  echo "[STEP] Regenerate paper figures"
  "$PYTHON_BIN" "$PAPER_DIR/generate_paper_figures.py"

  echo "[STEP] Build manuscript PDF"
  (
    cd "$PAPER_DIR"
    pdflatex -interaction=nonstopmode scb5_zeroshot_paper.tex >/dev/null
    pdflatex -interaction=nonstopmode scb5_zeroshot_paper.tex >/dev/null
  )

  echo "[DONE] Quick reproduction complete"
}

run_full() {
  echo "[STEP] Download models and data (if missing)"
  if [ ! -d "$SCB_DIR/datasets_scb" ]; then
    "$PYTHON_BIN" "$SCB_DIR/scripts/download_scb5_data.py"
  fi
  if [ ! -d "$SCB_DIR/ckpts" ]; then
    "$PYTHON_BIN" "$SCB_DIR/scripts/download_models.py"
  fi

  echo "[STEP] CLIP-family benchmark (all models x prompts x subsets)"
  "$PYTHON_BIN" "$SCB_DIR/experiments/main_clip.py"

  echo "[STEP] CAPE robustness"
  "$PYTHON_BIN" "$SCB_DIR/analysis/cape_robustness.py" --gpu "$GPU"

  echo "[STEP] Revision experiments (R1-R4)"
  "$PYTHON_BIN" "$SCB_DIR/analysis/run_revision_experiments.py" --gpu "$GPU" --exp r1 r2 r3 r4

  echo "[STEP] CAPE principle ablation"
  "$PYTHON_BIN" "$SCB_DIR/analysis/cape_principle_ablation.py"

  echo "[STEP] Paired bootstrap significance test"
  "$PYTHON_BIN" "$SCB_DIR/analysis/paired_bootstrap.py"

  echo "[STEP] Linear probe supervised baseline"
  "$PYTHON_BIN" "$SCB_DIR/analysis/linear_probe.py"

  echo "[STEP] Regenerate figures"
  # After --mode full, the new baseline_results.json exists and is preferred
  "$PYTHON_BIN" "$PAPER_DIR/generate_paper_figures.py"

  echo "[STEP] Build manuscript PDF"
  (
    cd "$PAPER_DIR"
    pdflatex -interaction=nonstopmode scb5_zeroshot_paper.tex >/dev/null
    pdflatex -interaction=nonstopmode scb5_zeroshot_paper.tex >/dev/null
  )

  echo "[DONE] Full reproduction complete"
}

case "$MODE" in
  quick)
    run_quick
    ;;
  full)
    run_full
    ;;
  *)
    echo "Unsupported mode: $MODE" >&2
    echo "Use --mode quick or --mode full" >&2
    exit 1
    ;;
esac
