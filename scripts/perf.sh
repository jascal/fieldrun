#!/usr/bin/env bash
# Perf harness — measure PREFILL tok/s (across context lengths) and DECODE tok/s for a fieldrun bundle, so every
# SIMD/kernel change reports a measured before/after rather than an estimate. NOT a correctness gate (that's
# validate_all.sh / bench.sh); this is the speed signal.
#
#   - prefill: timed over the OpenAI /v1/completions path with a UNIQUE prompt prefix per request, so the single-slot
#     prefix-KV cache can't reuse anything (reuse_len=0 -> a genuine cold prefill), and max_tokens=1 (one decoded token
#     is negligible). tok/s = usage.prompt_tokens / wall_time.
#   - decode: timed over the native /generate endpoint (token-ids in, fixed n, NO early EOS stop -> exactly n steps)
#     with a tiny prompt, so it isolates the memory-bound per-token decode. tok/s = n / wall_time.
#
# Usage: scripts/perf.sh <bundle-stem> [port]
#   e.g. scripts/perf.sh bundles/Qwen2.5-0.5B-Instruct/Qwen2.5-0.5B-Instruct
# Env: FEATURES=openblas (or simd-gemm) to build a variant; CTXS="20 40 80 160 320" sets the prefill context word-counts.
set -u
BIN=./target/release/fieldrun
STEM="${1:?usage: scripts/perf.sh <bundle-stem> [port]}"
PORT="${2:-8081}"
SENT="The quick brown fox jumps over the lazy dog and writes some Rust code. "
WORDS=(${CTXS:-20 40 80 160 320})
GEN_N="${GEN_N:-64}"

echo "[perf] building (FEATURES=${FEATURES:-none})"
cargo build --release ${FEATURES:+--features "$FEATURES"} >/dev/null 2>&1 || { echo "[perf] build failed"; exit 1; }

"$BIN" --bundle "$STEM" --serve "$PORT" >/tmp/perf_serve.log 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT
for _ in $(seq 1 180); do curl -fsS "http://localhost:$PORT/health" >/dev/null 2>&1 && break; sleep 0.5; done

printf "\n[perf] prefill — cold (unique prefix), max_tokens=1\n"
printf "%14s %10s %10s\n" prompt_tokens seconds tok/s
for w in "${WORDS[@]}"; do
  prompt="[perf-$w-$RANDOM] "
  for _ in $(seq 1 "$w"); do prompt+="$SENT"; done
  body=$(jq -n --arg p "$prompt" '{model:"rope",prompt:$p,max_tokens:1}')
  resp=$(curl -s -w '\n%{time_total}' "http://localhost:$PORT/v1/completions" -H 'Content-Type: application/json' -d "$body")
  t=$(printf '%s' "$resp" | tail -1)
  pt=$(printf '%s' "$resp" | sed '$d' | jq -r '.usage.prompt_tokens')
  awk -v pt="$pt" -v t="$t" 'BEGIN{printf "%14s %10.3f %10.1f\n", pt, t, pt/t}'
done

printf "\n[perf] decode — %d tokens via /generate (fixed n, no EOS stop)\n" "$GEN_N"
gen=$(jq -n --argjson n "$GEN_N" '{prompt:[1,2,3,4],n:$n}')
tg=$(curl -s -o /dev/null -w '%{time_total}' "http://localhost:$PORT/generate" -H 'Content-Type: application/json' -d "$gen")
awk -v n="$GEN_N" -v t="$tg" 'BEGIN{printf "%14s %10.3f %10.1f\n", n" tok", t, n/t}'
