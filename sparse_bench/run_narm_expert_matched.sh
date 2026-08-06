#!/bin/zsh
set -euo pipefail

PYTHON=/Users/macbook/miniconda3/bin/python3
ROOT=/Users/macbook/Desktop/product-recommendation-system/sparse_bench
SELECTORS=$ROOT/narm_expert_artifacts/selectors
RANKS=$ROOT/narm_expert_artifacts
mkdir -p "$SELECTORS" "$RANKS"

for SEED in 123 456; do
  for DOMAIN in Video_Games Baby_Products Diginetica_HID; do
    case "$DOMAIN" in
      Video_Games)
        PREFIX=video_games
        CHECKPOINT=$ROOT/paper_baseline_artifacts/video_games_full_narm_seed${SEED}.pt
        ;;
      Baby_Products)
        PREFIX=baby_products
        CHECKPOINT=$ROOT/paper_baseline_artifacts/baby_products_full_narm_seed${SEED}.pt
        ;;
      Diginetica_HID)
        PREFIX=diginetica
        CHECKPOINT=$ROOT/paper_baseline_digi_nested_artifacts/diginetica_hid_full_narm_seed${SEED}.pt
        ;;
    esac

    SELECTOR=$SELECTORS/${PREFIX}_selector_seed${SEED}.npz
    "$PYTHON" "$ROOT/build_selector_artifact.py" \
      --domain "$DOMAIN" \
      --seed "$SEED" \
      --work-artifact-dir "$ROOT/cearfn_v2_nested_artifacts" \
      --output "$SELECTOR"

    "$PYTHON" "$ROOT/run_narm_expert_selector.py" \
      --domain "$DOMAIN" \
      --seed "$SEED" \
      --cearf-artifact "$SELECTOR" \
      --narm-checkpoint "$CHECKPOINT" \
      --narm-artifact "$RANKS/${PREFIX}_narm_seed${SEED}_top20.npz" \
      --output "$ROOT/narm_expert_${PREFIX}_seed${SEED}.json"
  done
done
