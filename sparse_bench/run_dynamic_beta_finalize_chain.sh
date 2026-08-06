#!/usr/bin/env bash
set -euo pipefail

primary_pid="${1:-87118}"
script_dir="$(cd "$(dirname "$0")" && pwd)"
cd "$script_dir"

stamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

echo "[$(stamp)] waiting for primary dynamic-beta PID ${primary_pid}"
while kill -0 "$primary_pid" 2>/dev/null; do
  sleep 15
done

echo "[$(stamp)] finalizing declared-validation provenance"
python finalize_dynamic_beta_protocol.py --apply

echo "[$(stamp)] reproducing all frozen expert caches"
python audit_dynamic_beta_provenance.py \
  Video_Games Baby_Products Diginetica_HID \
  --seeds 42 123 456 --device auto

echo "[$(stamp)] aggregating primary results"
python summarize_dynamic_beta.py --bootstrap-repetitions 20000

echo "[$(stamp)] benchmarking exact dynamic inference"
python benchmark_dynamic_beta_inference.py

echo "[$(stamp)] fitting frozen-rank allocation controls"
python run_dynamic_beta_allocation_controls.py \
  Video_Games Baby_Products Diginetica_HID --seeds 42 123 456

echo "[$(stamp)] running fixed-allocation fusion-operator control"
python run_dynamic_beta_fusion_control.py \
  Video_Games Baby_Products Diginetica_HID --seeds 42 123 456
python summarize_dynamic_beta_fusion_control.py \
  --bootstrap-repetitions 20000

echo "[$(stamp)] running NARM expert swap"
python run_dynamic_beta_expert_swap.py \
  Video_Games Baby_Products Diginetica_HID \
  --seeds 42 123 456 \
  --dynamic-artifact-dir dynamic_beta_trainonly_v2_artifacts \
  --primary-results dynamic_beta_trainonly_v2_results.json
python summarize_dynamic_beta_expert_swap.py

echo "[$(stamp)] regenerating data-driven charts"
python slide_graphs/generate_dynamic_beta_charts.py

echo "[$(stamp)] compiling paper artifacts"
(
  cd paper
  tectonic main_dynamic.tex --keep-logs --keep-intermediates
  tectonic supplementary_dynamic.tex --keep-logs --keep-intermediates
)

echo "[$(stamp)] finalize chain complete"
