#!/usr/bin/env bash
# Faithfulness sweep: for every transformer-validatable arch, build a tiny random-init torch reference, convert it to a
# fieldrun bundle at each dtype, run the pure-Rust kernel, and report top-1 agreement vs torch. f32 is the gate (the
# architecture math); f16/int8 are lossy by design. No gated downloads — a tiny instance exercises every code path.
#
# Usage: scripts/validate_all.sh                       (run from the repo root; needs the torch venv)
#        FEATURES=accelerate scripts/validate_all.sh   (validate a BLAS build, e.g. on macOS — f32 column is the gate)
set -u
PY="${PY:-../lm-sae/.venv/bin/python}"
BIN=./target/release/fieldrun
REF=scripts/gemma3_ref.py
# FEATURES lets you validate a non-default build (e.g. the Apple Accelerate / OpenBLAS matmul) with the same sweep.
cargo build --release ${FEATURES:+--features "$FEATURES"} >/dev/null 2>&1 || { echo "build failed"; exit 1; }

printf "\n%-12s %-14s %-8s %-8s %-8s\n" arch ref f32 f16 int8
printf '%.0s-' {1..54}; echo
# arch | bundle --arch flag | torch reference class
for spec in "gemma3:gemma3:Gemma3" "gemma4:gemma4:Gemma4-dense" "gemma4moe:gemma4:Gemma4-MoE" "qwen3moe:qwen3moe:Qwen3-MoE" "qwen3moeswa:qwen3moe:Qwen3-MoE-SWA" "mla:mla:DeepSeek-V3" "mlayarn:mla:DeepSeek-YaRN" "minimax:minimax:MiniMax-M2"; do
  tag="${spec%%:*}"; rest="${spec#*:}"; arch="${rest%%:*}"; ref="${rest##*:}"
  $PY $REF build "$tag" >/dev/null 2>&1 || { printf "%-12s BUILD FAILED\n" "$tag"; continue; }
  row=""
  for dt in f32 f16 int8; do
    # --force: build() reseeds the tiny model every run, so a bundle left over from a previous run would have been
    # converted from DIFFERENT random weights than this run's torch reference — reusing it compares apples to oranges
    # (silent 1/60 garbage). CI's /tmp is clean so it never hit this; a repeated local run did.
    $BIN convert --model /tmp/${tag}tiny --arch "$arch" --dtype "$dt" -o /tmp/${tag}_$dt --force >/dev/null 2>&1
    $BIN --bundle /tmp/${tag}_$dt --ids /tmp/${tag}_holdout.json --ctx 16 --n-eval 60 --dump /tmp/${tag}_${dt}.txt >/dev/null 2>&1
    a=$($PY $REF compare /tmp/${tag}_${dt}.txt "$tag" 2>/dev/null | grep -oE '[0-9]+/[0-9]+ top-1' | head -1 | grep -oE '^[0-9]+/[0-9]+')
    row="$row $(printf '%-8s' "${a:-ERR}")"
  done
  printf "%-12s %-14s%s\n" "$tag" "$ref" "$row"
done
echo

# Generate + explain gate: incremental KV-cache decode must be BYTE-IDENTICAL to the naive full-recompute path (the f32
# correctness gate for generation — naive is itself top-1-validated vs torch above, so KV==naive ⇒ KV==torch), for both
# the f32 and int8-KV caches; and `explain` (the runtime circuit/feature readout) must run on every arch. Reuses the f32
# bundles + holdouts built above.
printf "%-12s %-22s %-12s %s\n" arch "generate f32(KV==naive)" "int8-KV" explain
printf '%.0s-' {1..56}; echo
for spec in gemma3:gemma3 gemma4:gemma4 gemma4moe:gemma4 qwen3moe:qwen3moe qwen3moeswa:qwen3moe mla:mla mlayarn:mla minimax:minimax; do
  tag="${spec%%:*}"; b=/tmp/${tag}_f32; ids=/tmp/${tag}_holdout.json
  [ -f "$b.fieldrun.json" ] || { printf "%-12s (no f32 bundle — built above?)\n" "$tag"; continue; }
  idn=$($BIN --bundle "$b" --ids "$ids" --ctx 16 --generate 16 2>/dev/null | grep -oE 'identical: (true|false)' | grep -oE '(true|false)')
  id8=$($BIN --bundle "$b" --ids "$ids" --ctx 16 --generate 16 --kv-int8 2>/dev/null | grep -oE 'identical: (true|false)' | grep -oE '(true|false)')
  ex=$($BIN --bundle "$b" --ids "$ids" --ctx 16 --explain 2>/dev/null | grep -c 'model predicts')
  exl=$([ "${ex:-0}" -ge 1 ] && echo ok || echo MISSING)
  printf "%-12s %-22s %-12s %s\n" "$tag" "${idn:-ERR}" "${id8:-ERR}" "$exl"
done
echo

# Real-model round-trips from the HF cache (the convert + run path on actual weights), where present.
GPT2=$(find ~/.cache/huggingface/hub/models--gpt2/snapshots -name config.json -exec dirname {} \; 2>/dev/null | head -1)
if [ -n "${GPT2:-}" ] && [ -f ../lm-sae/pylm/holdout_gpt2.json ]; then
  echo "real GPT-2 (HF cache) next-token top-1 over 200 positions:"
  for dt in f32 int8; do
    $BIN convert --model "$GPT2" --arch gpt2 --dtype $dt -o /tmp/gpt2_$dt --force >/dev/null 2>&1
    t=$($BIN --bundle /tmp/gpt2_$dt --ids ../lm-sae/pylm/holdout_gpt2.json --ctx 64 --n-eval 200 2>/dev/null | grep -oE 'top-1: [0-9.]+%')
    printf "  %-5s %s\n" "$dt" "$t"
  done
fi
