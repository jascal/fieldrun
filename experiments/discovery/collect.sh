#!/bin/bash
# Collect per-decision recursion signatures over a prompt corpus → dumps/NNN.jsonl, one file per prompt.
# Usage: ./collect.sh [bundle-stem] [corpus-file] [id-offset]
#   id-offset 0 (default) WIPES dumps/ first (fresh run); a nonzero offset APPENDS (residual-driven enrichment:
#   add a corpus that targets a residual idiom, e.g. ./collect.sh <bundle> corpus_enrich.txt 100).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
FR="$HERE/../../target/release/fieldrun"
BUNDLE="${1:-$HOME/.cache/fieldrun/bundles/Qwen2.5-0.5B/Qwen2.5-0.5B}"
CORPUS="${2:-$HERE/corpus.txt}"
OFFSET="${3:-0}"
mkdir -p "$HERE/dumps"
[ "$OFFSET" = 0 ] && rm -f "$HERE/dumps/"*.jsonl
i="$OFFSET"
while IFS= read -r line; do
  [ -z "$line" ] && continue
  printf -v id "%03d" "$i"
  echo "[$id] $line"
  timeout 200 "$FR" --bundle "$BUNDLE" --recursion-explain --recursion-dump "$HERE/dumps/$id.jsonl" --text "$line" 2>&1 \
    | grep -E "wrote .* decisions|no recursion_trace|error" || echo "  (no output)"
  i=$((i+1))
done < "$CORPUS"
echo "collected from $CORPUS → $HERE/dumps/ (offset $OFFSET)"
