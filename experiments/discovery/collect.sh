#!/bin/bash
# Collect per-decision recursion signatures over a diverse prompt corpus → dumps/NNN.jsonl, one file per prompt.
# Usage: ./collect.sh [bundle-stem]   (default: Qwen2.5-0.5B — fast, 24 layers; richer signal on bigger models)
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
FR="$HERE/../../target/release/fieldrun"
BUNDLE="${1:-$HOME/.cache/fieldrun/bundles/Qwen2.5-0.5B/Qwen2.5-0.5B}"
mkdir -p "$HERE/dumps"
rm -f "$HERE/dumps/"*.jsonl
i=0
while IFS= read -r line; do
  [ -z "$line" ] && continue
  printf -v id "%03d" "$i"
  echo "[$id] $line"
  timeout 200 "$FR" --bundle "$BUNDLE" --recursion-explain --recursion-dump "$HERE/dumps/$id.jsonl" --text "$line" 2>&1 \
    | grep -E "wrote .* decisions|no recursion_trace|error" || echo "  (no output)"
  i=$((i+1))
done < "$HERE/corpus.txt"
echo "collected $i prompts → $HERE/dumps/"
