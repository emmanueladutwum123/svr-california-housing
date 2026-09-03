#!/bin/sh
# Stream A (restart after crash): the two experiments that were killed mid-run.
cd "$HOME/svr-revision"
PY=/Users/emmanueladutwum/sports-forecast-engine/.venv/bin/python
for s in e1b_preethi_reproduction e3_statistics; do
  echo "=== $s started $(date +%H:%M:%S) ==="
  $PY "$s.py" > "logs_${s%%_*}.txt" 2>&1
  echo "=== $s exit=$? finished $(date +%H:%M:%S) ==="
done
echo "=== stream A complete $(date +%H:%M:%S) ==="
