#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="/Users/macbook/miniconda3/bin/python3"
RUNNER="$HERE/run_validation_gated_pasgr.py"
OUT_DIR="$HERE/pasgr_stability_artifacts"
mkdir -p "$OUT_DIR"

run_domain() {
  local domain="$1"
  local seed="$2"
  shift 2
  "$PYTHON_BIN" "$RUNNER" "$domain" \
    --seed "$seed" \
    --epochs 4 \
    --candidate-width 120 \
    --artifact-dir "$HERE/vgated_artifacts" \
    --output "$OUT_DIR/${domain}_seed${seed}.json" \
    --labels "$@"
}

for seed in 123 456; do
  run_domain Video_Games "$seed" \
    graph0.35_noproto_contrast0.15_inbatch0.1 \
    graph0_noproto_contrast0.15_inbatch0.1 \
    graph0.35_proto_contrast0_inbatch0

  run_domain Baby_Products "$seed" \
    graph0_noproto_contrast0.15_inbatch0 \
    graph0_noproto_contrast0.15_inbatch0.1 \
    graph0_proto_contrast0.15_inbatch0.1

  run_domain Diginetica_HID "$seed" \
    graph0.35_noproto_contrast0_inbatch0.1 \
    graph0.35_proto_contrast0.15_inbatch0.1
done

"$PYTHON_BIN" "$HERE/summarize_pasgr_stability.py"
