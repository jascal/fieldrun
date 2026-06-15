#!/usr/bin/env bash
# Bucketing sweep: models × corpora × (experts:depth). Per-combo output under sweeps/runs/ + a summary TSV.
# Env: MODELS, CORPORA, GRID (space-sep "experts:depth"), NEVAL, CTX.  Run from the repo root after a release build.
set -u
FR=./target/release/fieldrun
CORP=sweeps/corpora
OUT=sweeps/runs
mkdir -p "$OUT"
NEVAL="${NEVAL:-120}"
CTX="${CTX:-48}"
MODELS="${MODELS:-Qwen2.5-0.5B-Instruct}"
CORPORA="${CORPORA:-english german code math pooled_diverse}"
GRID="${GRID:-8:1 16:1}"
SUMMARY="$OUT/summary.tsv"
echo -e "model\tcorpus\tE\tdepth\tN\t|C|\tspan1\tactive_fewer\tresident\tfaithful\tleaves" > "$SUMMARY"
for M in $MODELS; do
  for C in $CORPORA; do
    [ -f "$CORP/$C.json" ] || { echo "[sweep] missing corpus $C — skip"; continue; }
    for G in $GRID; do
      E="${G%%:*}"; D="${G##*:}"
      tag="${M}__${C}__E${E}_d${D}"
      extra=""; [ "$D" -gt 1 ] && extra="--recurse-depth $D --recurse-min 6"
      echo "[sweep] $tag (n=$NEVAL)…"
      $FR --bundle "$M" --ids "$CORP/$C.json" --corpus-decompose --experts "$E" --ctx "$CTX" --n-eval "$NEVAL" \
          --residency $extra --experts-dl-contrib "$OUT/$tag.contrib.dl" > "$OUT/$tag.txt" 2>/dev/null
      g() { grep -oE "$1" "$OUT/$tag.txt" 2>/dev/null | grep -oE "$2" | head -1; }
      N=$(g "N tokens +[0-9]+" "[0-9]+")
      DC=$(g "\\|C\\| distinct circuits +[0-9]+" "[0-9]+")
      S1=$(g "routable\\) +[0-9]+%" "[0-9]+%")
      AC=$(g "→ [0-9]+% fewer" "[0-9]+%")
      RES=$(g "hot resident set: [0-9]+" "[0-9]+\$")
      LV=$(g "depth [0-9]+, [0-9]+ leaf" "[0-9]+ leaf"); LV="${LV% leaf}"
      FA=$(grep -oE "faithful decode [0-9]+/[0-9]+ \\([0-9]+%\\)" "$OUT/$tag.contrib.dl" 2>/dev/null | grep -oE "[0-9]+%" | head -1)
      echo -e "$M\t$C\t$E\t$D\t${N:-?}\t${DC:-?}\t${S1:-?}\t${AC:-?}\t${RES:-?}\t${FA:-?}\t${LV:-1}" >> "$SUMMARY"
    done
  done
done
echo "[sweep] done → $SUMMARY"; echo
column -t -s $'\t' "$SUMMARY"
