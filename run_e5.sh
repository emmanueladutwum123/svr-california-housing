#!/bin/sh
cd "$HOME/svr-revision"
PY=/Users/emmanueladutwum/sports-forecast-engine/.venv/bin/python
echo "=== e5 restarted $(date +%H:%M:%S) ==="
$PY e5_validation.py > logs_e5.txt 2>&1
echo "=== e5 exit=$? finished $(date +%H:%M:%S) ==="
