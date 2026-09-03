#!/bin/sh
# Chain B: waits for the two currently-running jobs to write their results,
# then runs the remaining experiments one at a time.
cd "$HOME/svr-revision"
PY=/Users/emmanueladutwum/sports-forecast-engine/.venv/bin/python
while [ ! -f results/e1b_preethi_reproduction.json ] || [ ! -f results/e2_fair_comparison.json ]; do
  sleep 10
done
for s in e4b_extended_ablation e5_validation e6_datasets e7_sensitivity e8_kernels_search_size; do
  echo "=== $s started $(date +%H:%M:%S) ==="
  $PY "$s.py" > "logs_${s%%_*}.txt" 2>&1
  echo "=== $s exit=$? finished $(date +%H:%M:%S) ==="
done
echo "=== chain complete $(date +%H:%M:%S) ==="
