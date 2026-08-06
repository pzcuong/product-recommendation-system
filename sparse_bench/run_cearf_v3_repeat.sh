#!/bin/zsh
set -euo pipefail

PYTHON=/Users/macbook/miniconda3/bin/python3
ROOT=/Users/macbook/Desktop/product-recommendation-system/sparse_bench
OUT=$ROOT/cearf_v3_artifacts
mkdir -p "$OUT"

"$PYTHON" "$ROOT/run_cearf_v3_memory.py" \
  --domain Diginetica_HID \
  --use-repeat \
  --output "$ROOT/cearf_v3_repeat_diginetica.json" \
  --rank-artifact "$OUT/diginetica_repeat_top20.npz"

"$PYTHON" "$ROOT/run_cearf_v3_memory.py" \
  --domain Tmall \
  --use-repeat \
  --output "$ROOT/cearf_v3_repeat_tmall.json" \
  --rank-artifact "$OUT/tmall_repeat_top20.npz"
