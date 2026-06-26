#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCB_DIR="$ROOT/scb5_zeroshot"
PAPER_DIR="$SCB_DIR/paper"

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
      echo "Usage: bash scb5_zeroshot/reproduce_paper.sh [--mode quick|full] [--gpu 0] [--python python]" >&2
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
  echo "[STEP] Main experiment pipeline"
  "$PYTHON_BIN" "$SCB_DIR/exp_runner.py" --gpu "$GPU"

  echo "[STEP] CAPE robustness"
  "$PYTHON_BIN" "$SCB_DIR/cape_robustness.py" --gpu "$GPU"

  echo "[STEP] Revision experiments (R1-R4)"
  "$PYTHON_BIN" "$SCB_DIR/run_revision_experiments.py" --gpu "$GPU" --exp r1 r2 r3 r4

  echo "[STEP] Regenerate figures"
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
