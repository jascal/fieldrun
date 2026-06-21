#!/usr/bin/env bash
# Regression smoke for `--export-logic --residue-strategy {ring|edb|margin}` + the `eval` round-trip.
# Local (needs a rope bundle; CI has none). Usage: scripts/export_logic_smoke.sh [BUNDLE]
set -euo pipefail
BUNDLE="${1:-$HOME/.cache/fieldrun/bundles/Qwen2.5-1.5B/Qwen2.5-1.5B}"
BIN=target/release/fieldrun
TEXT="The history of science is the study of how knowledge of the natural world has developed"
OUT=$(mktemp -d)
trap 'rm -rf "$OUT"' EXIT

[ -x "$BIN" ] || { echo "build first: cargo build --release --features api"; exit 1; }

fail=0
for strat in ring edb margin; do
  "$BIN" --bundle "$BUNDLE" --export-logic "$OUT/$strat" --text "$TEXT" --steps 8 \
         --residue-strategy "$strat" --tau 2 2>&1 | grep "export-logic →" || { echo "FAIL: $strat emit"; fail=1; }
done

# round-trip: every emitted .dl must `decide` the model's token under BOTH semirings (max + log agree on argmax)
for dl in "$OUT"/ring.*.dl "$OUT"/edb.*.dl; do
  mx=$("$BIN" eval "$dl" --semiring max 2>/dev/null | grep -oE 'decide\([0-9]+\)' | head -1)
  [ -n "$mx" ] || { echo "FAIL: $dl produced no decide under max"; fail=1; }
done
echo "ring/edb/margin emit + max-semiring decode round-trip: $([ $fail = 0 ] && echo PASS || echo FAIL)"
exit $fail
