#!/bin/bash
# Collect CAUSAL profiles — per prompt, ablate each layer's attn/mlp block at the last-position decision and record
# which FLIP the prediction (load-bearing blocks). One object per prompt → dumps_causal/NNN.jsonl.
# Rope family only (needs predict_ablated_blocks). Use a fast model (0.5B) — this is ~2*n_layer forwards per prompt.
# Usage: ./collect_causal.sh [bundle-stem] [corpus-file] [id-offset]
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
FR="$HERE/../../target/release/fieldrun"
BUNDLE="${1:-$HOME/.cache/fieldrun/bundles/Qwen2.5-0.5B/Qwen2.5-0.5B}"
CORPUS="${2:-$HERE/corpus.txt}"
OFFSET="${3:-0}"
mkdir -p "$HERE/dumps_causal"
[ "$OFFSET" = 0 ] && rm -f "$HERE/dumps_causal/"*.jsonl
i="$OFFSET"
while IFS= read -r line; do
  [ -z "$line" ] && continue
  printf -v id "%03d" "$i"
  echo "[$id] $line"
  timeout 300 "$FR" --bundle "$BUNDLE" --recursion-explain --causal-dump "$HERE/dumps_causal/$id.jsonl" --text "$line" 2>&1 \
    | grep -E "wrote causal|no predict_ablated|no dims|error" || echo "  (no output)"
  i=$((i+1))
done < "$CORPUS"
echo "collected causal from $CORPUS → $HERE/dumps_causal/ (offset $OFFSET)"
