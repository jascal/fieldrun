#!/usr/bin/env bash
# Quality sweep (the announce gate, NOT the correctness gate): for each newer arch, sweep several random seeds at a
# larger eval length and aggregate top-1 agreement vs the torch reference per dtype. f32 should hold ~100% (the math);
# f16/int8 give a quality curve (how often low precision flips a near-tie). Real-weight accuracy belongs here too once a
# checkpoint is local; with no local weights for the newer archs, the multi-seed tiny sweep is the strongest signal.
#
# Usage: scripts/bench.sh [n_eval] [seeds...]   e.g.  scripts/bench.sh 120 0 1 2 3
set -u
PY="${PY:-../lm-sae/.venv/bin/python}"; BIN=./target/release/fieldrun; REF=scripts/gemma3_ref.py
NE="${1:-120}"; shift || true; SEEDS=("${@:-0 1 2}"); SEEDS=(${SEEDS[@]})
cargo build --release >/dev/null 2>&1
printf "\nquality sweep — %d positions × seeds {%s}\n" "$NE" "${SEEDS[*]}"
printf "%-12s %-10s %-10s %-10s\n" arch f32 f16 int8; printf '%.0s-' {1..46}; echo
for spec in "gemma3:gemma3" "gemma4:gemma4" "gemma4moe:gemma4" "qwen3moe:qwen3moe" "qwen3moeswa:qwen3moe" "mla:mla" "mlayarn:mla" "minimax:minimax"; do
  tag="${spec%%:*}"; arch="${spec##*:}"
  declare -A ok tot
  for dt in f32 f16 int8 int4; do ok[$dt]=0; tot[$dt]=0; done
  for s in "${SEEDS[@]}"; do
    SEED=$s N_EVAL=$NE $PY $REF build "$tag" >/dev/null 2>&1 || continue
    for dt in f32 f16 int8 int4; do
      # --force: each seed reseeds the tiny model, so a bundle left from the previous seed would be compared
      # against the WRONG reference (same trap validate_all.sh hit — convert skips existing bundles by default).
      $BIN convert --model /tmp/${tag}tiny --arch "$arch" --dtype "$dt" -o /tmp/${tag}_$dt --force >/dev/null 2>&1
      $BIN --bundle /tmp/${tag}_$dt --ids /tmp/${tag}_holdout.json --ctx 16 --n-eval "$NE" --dump /tmp/${tag}_${dt}.txt >/dev/null 2>&1
      r=$(SEED=$s N_EVAL=$NE $PY $REF compare /tmp/${tag}_${dt}.txt "$tag" 2>/dev/null | grep -oE '[0-9]+/[0-9]+ top-1' | grep -oE '^[0-9]+/[0-9]+')
      [ -n "$r" ] && { ok[$dt]=$(( ${ok[$dt]} + ${r%/*} )); tot[$dt]=$(( ${tot[$dt]} + ${r#*/} )); }
    done
  done
  row=""
  for dt in f32 f16 int8 int4; do
    if [ "${tot[$dt]}" -gt 0 ]; then row="$row $(printf '%-10s' "$(awk "BEGIN{printf \"%.1f%%\", 100*${ok[$dt]}/${tot[$dt]}}") (${ok[$dt]}/${tot[$dt]})")"; else row="$row $(printf '%-10s' ERR)"; fi
  done
  printf "%-12s%s\n" "$tag" "$row"
done
