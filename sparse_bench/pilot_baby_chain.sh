#!/bin/bash
cd /Users/macbook/Desktop/product-recommendation-system/sparse_bench
export PYTHONPATH=$PWD
for V in off raw-semantic casm; do
  echo "=== BABY variant=$V start $(date) ==="
  python run_cearfn_v2.py Baby_Products --seeds 42 --memory-variant $V \
    --output pilot_baby_results.json --artifact-dir cearfn_v2_pilot_artifacts \
    --partial-dir cearfn_v2_pilot_partials || echo "FAILED variant=$V"
  echo "=== BABY variant=$V end $(date) ==="
done
