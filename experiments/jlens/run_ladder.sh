#!/bin/bash
# J-lens Pythia-ladder fit + eval. Sequential (each fit uses full core parallelism), apples-to-apples config
# (same corpus, all layers, probes=5, src=4, max_seq=24, seed=1) so the earlier-resolve margin is comparable across
# scale. Est. wall: 14m ~25s · 70m ~7m · 160m ~1h · 410m ~7h  ⇒ ~8.4h total.
set -u
cd /home/allans/code/fieldrun
BIN=./target/release/fieldrun
CORPUS=experiments/jlens/fit_corpus.txt
OUT=experiments/jlens/ladder
mkdir -p "$OUT"
EVAL_TEXT="The capital of France is Paris. The capital of Japan is"

for M in pythia-14m pythia-70m pythia-160m pythia-410m; do
  echo "############################################################"
  echo "### [$(date '+%F %T')] FIT $M"
  echo "############################################################"
  "$BIN" --bundle "$M" --text "x" --recursion-explain --jlens-fit \
    --jlens-corpus "$CORPUS" --jlens-out "$OUT/$M.jlens" \
    --jlens-probes 5 --jlens-max-seq 24 --jlens-max-src 4 --jlens-layers all \
    --jlens-ckpt-every 20 --jlens-seed 1
  echo "### [$(date '+%F %T')] EVAL $M"
  "$BIN" --bundle "$M" --text "$EVAL_TEXT" \
    --recursion-explain --jlens-eval --jlens-in "$OUT/$M.jlens" \
    --jlens-shrink 0.0,0.1,0.25,0.5,0.75,1.0 2>&1 | tr '\r' '\n' | grep -E "eval ·|λ="
  # also export each to .npz for pil
  "$BIN" --jlens-export "$OUT/$M.npz" --jlens-in "$OUT/$M.jlens" 2>&1 | tail -1
done
echo "############################################################"
echo "### [$(date '+%F %T')] LADDER COMPLETE → $OUT"
echo "############################################################"
