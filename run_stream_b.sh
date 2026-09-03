#!/bin/sh
# Stream B (restart after crash): the five experiments that never started.
cd "$HOME/svr-revision"
PY=/Users/emmanueladutwum/sports-forecast-engine/.venv/bin/python
for s in e4b_extended_ablation e5_validation e6_datasets e7_sensitivity e8_kernels_search_size; do
  echo "=== $s started $(date +%H:%M:%S) ==="
  $PY "$s.py" > "logs_${s%%_*}.txt" 2>&1
  echo "=== $s exit=$? finished $(date +%H:%M:%S) ==="
done
echo "=== stream B complete $(date +%H:%M:%S) ==="
