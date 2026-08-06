#!/bin/zsh
set -euo pipefail

PYTHON=/Users/macbook/miniconda3/bin/python3
ROOT=/Users/macbook/Desktop/product-recommendation-system/sparse_bench
TEACHERS=$ROOT/semantic_teacher_artifacts

"$PYTHON" "$ROOT/build_minilm_teacher.py" \
  Video_Games Baby_Products \
  --device mps \
  --output-dir "$TEACHERS"

"$PYTHON" "$ROOT/run_validation_gated_pasgr.py" \
  Video_Games Baby_Products \
  --semantic-dir "$TEACHERS" \
  --output "$ROOT/pasgr_config_minilm.json" \
  --artifact-dir "$ROOT/vgated_minilm_artifacts"

"$PYTHON" "$ROOT/run_cearfn_v2.py" \
  Video_Games Baby_Products \
  --seeds 42 123 456 \
  --semantic-dir "$TEACHERS" \
  --config-file "$ROOT/pasgr_config_minilm.json" \
  --output "$ROOT/cearfn_v2_minilm_results.json" \
  --artifact-dir "$ROOT/cearfn_v2_minilm_artifacts" \
  --partial-dir "$ROOT/cearfn_v2_minilm_partials"

"$PYTHON" "$ROOT/run_semantic_init_baselines.py" \
  Video_Games Baby_Products \
  --models NARM \
  --seeds 42 123 456 \
  --teacher minilm \
  --semantic-dir "$TEACHERS" \
  --output "$ROOT/narm_minilm_fairness_results.json" \
  --artifact-dir "$ROOT/narm_minilm_fairness_artifacts"

"$PYTHON" "$ROOT/run_semantic_init_baselines.py" \
  Video_Games Baby_Products \
  --models NARM \
  --seeds 42 123 456 \
  --teacher tfidf \
  --output "$ROOT/narm_tfidf_fairness_nested_results.json" \
  --artifact-dir "$ROOT/narm_tfidf_fairness_nested_artifacts"
