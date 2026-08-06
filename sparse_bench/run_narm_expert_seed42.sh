#!/bin/zsh
set -euo pipefail

PYTHON=/Users/macbook/miniconda3/bin/python3
ROOT=/Users/macbook/Desktop/product-recommendation-system/sparse_bench
OUT=$ROOT/narm_expert_artifacts
mkdir -p "$OUT"

"$PYTHON" "$ROOT/run_narm_expert_selector.py" \
  --domain Video_Games \
  --seed 42 \
  --cearf-artifact "$ROOT/cearfn_v2_video_games_validation_artifacts/video_games_v2_seed42_ranks.npz" \
  --narm-checkpoint "$ROOT/paper_baseline_artifacts/video_games_full_narm_seed42.pt" \
  --narm-artifact "$OUT/video_games_narm_seed42_top20.npz" \
  --output "$ROOT/narm_expert_video_games_seed42.json"

"$PYTHON" "$ROOT/run_narm_expert_selector.py" \
  --domain Baby_Products \
  --seed 42 \
  --cearf-artifact "$ROOT/baby_selector_seed42.npz" \
  --narm-checkpoint "$ROOT/paper_baseline_artifacts/baby_products_full_narm_seed42.pt" \
  --narm-artifact "$OUT/baby_products_narm_seed42_top20.npz" \
  --output "$ROOT/narm_expert_baby_products_seed42.json"

"$PYTHON" "$ROOT/run_narm_expert_selector.py" \
  --domain Diginetica_HID \
  --seed 42 \
  --cearf-artifact "$ROOT/diginetica_selector_seed42.npz" \
  --narm-checkpoint "$ROOT/paper_baseline_digi_nested_artifacts/diginetica_hid_full_narm_seed42.pt" \
  --narm-artifact "$OUT/diginetica_narm_seed42_top20.npz" \
  --output "$ROOT/narm_expert_diginetica_seed42.json"
