#!/bin/bash
cd /Users/macbook/Desktop/product-recommendation-system/sparse_bench
export PYTHONPATH=$PWD
# wait for the running off variant (pid unknown here): poll for its result key
while ! python - <<'PY'
import json, sys
try:
    r = json.load(open("pilot_cearfn_v2_results.json"))
    sys.exit(0 if "Video_Games" in r and r["Video_Games"]["runs"] else 1)
except Exception:
    sys.exit(1)
PY
do sleep 30; done
for V in raw-semantic casm; do
  echo "=== VG variant=$V start $(date) ==="
  python run_cearfn_v2.py Video_Games --seeds 42 --memory-variant $V \
    --output pilot_cearfn_v2_results.json --artifact-dir cearfn_v2_pilot_artifacts \
    --partial-dir cearfn_v2_pilot_partials || echo "FAILED variant=$V"
  echo "=== VG variant=$V end $(date) ==="
done
echo "=== VG chain complete $(date) ==="
