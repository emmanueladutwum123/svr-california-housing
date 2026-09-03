#!/bin/sh
# Regenerate every number, table and figure from results/, then compile.
set -e
cd "$HOME/svr-revision"
PY=/Users/emmanueladutwum/sports-forecast-engine/.venv/bin/python
$PY make_numbers.py
$PY make_tables.py
$PY make_figures.py
cd paper
tectonic -X compile main.tex --outdir build --keep-intermediates 2>&1 | tail -3
# the response letter, CIS and cover letter all quote numbers from numbers.tex and
# resolve section/table numbers from main.aux, so they must be rebuilt whenever the
# manuscript is -- otherwise they silently go stale as new results land.
for d in response cis coverletter; do
  [ -f "$d.tex" ] && tectonic -X compile "$d.tex" --outdir build 2>&1 | grep -E "^error" || true
done
echo "built: main, response, cis, coverletter"
