#!/bin/bash
# Collect per-decision DLA PROFILES (which blocks/circuit drive each prediction) → dumps_dla/NNN.jsonl.
# decomp_all (all positions, one forward) is implemented on gemma4; other arches fall back to last-position residual_decomp.
# Usage: ./collect_dla.sh [bundle-stem] [corpus-file] [id-offset]
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
FR="$HERE/../../target/release/fieldrun"
BUNDLE="${1:-$HOME/.cache/fieldrun/bundles/gemma-4-e4b-it-int4/gemma-4-e4b-it-int4}"
CORPUS="${2:-$HERE/corpus.txt}"
OFFSET="${3:-0}"
mkdir -p "$HERE/dumps_dla"
[ "$OFFSET" = 0 ] && rm -f "$HERE/dumps_dla/"*.jsonl
i="$OFFSET"
while IFS= read -r line; do
  [ -z "$line" ] && continue
  printf -v id "%03d" "$i"
  echo "[$id] $line"
  timeout 300 "$FR" --bundle "$BUNDLE" --recursion-explain --dla-dump "$HERE/dumps_dla/$id.jsonl" --text "$line" 2>&1 \
    | grep -E "wrote DLA|no decomp|error" || echo "  (no output)"
  i=$((i+1))
done < "$CORPUS"
echo "collected DLA from $CORPUS → $HERE/dumps_dla/ (offset $OFFSET)"
