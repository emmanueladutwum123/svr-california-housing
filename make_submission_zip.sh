#!/bin/sh
# Build the LaTeX source archive for the Springer portal.
# The portal may flatten directories, so everything is staged at the top level
# and main.tex's \input{tables/...} and {figures/...} paths are rewritten to match.
# The archive is verified by compiling it from scratch before it is written.
set -e
cd "$HOME/svr-revision"
S=$(mktemp -d); OUT="$HOME/svr-revision/SVR_Revision_Source.zip"

cp paper/main.tex paper/numbers.tex paper/references.bib \
   paper/sn-jnl.cls paper/sn-mathphys-num.bst "$S"/
cp paper/tables/*.tex "$S"/
# only the figures main.tex actually includes
for b in $(grep -o 'includegraphics\[[^]]*\]{figures/[a-z_]*' paper/main.tex | sed 's|.*figures/||'); do
  cp "paper/figures/$b.pdf" "$S"/
done
sed -i '' 's|figures/||g; s|tables/||g' "$S/main.tex"

( cd "$S" && tectonic -X compile main.tex --outdir . --keep-intermediates >/dev/null 2>&1 )
pages=$(pdfinfo "$S/main.pdf" | awk '/Pages/{print $2}')
qq=$(pdftotext "$S/main.pdf" - | grep -c '??' || true)
[ "$qq" -eq 0 ] || { echo "REFUSING: $qq unresolved values in the built PDF"; exit 1; }
rm -f "$S"/main.aux "$S"/main.log "$S"/main.out "$S"/main.blg

rm -f "$OUT"
( cd "$S" && zip -q -X "$OUT" main.tex numbers.tex references.bib main.bbl \
    sn-jnl.cls sn-mathphys-num.bst tab_*.tex fig_*.pdf main.pdf )
rm -rf "$S"
echo "wrote $OUT ($pages pages, 0 unresolved values)"
