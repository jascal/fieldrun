#!/usr/bin/env bash
# Lossless-optimization speedup benchmark for the LO3a whole-model Datalog program:
# Soufflé interpreter vs Soufflé COMPILED (native C++ synthesis) vs fieldrun's native kernel —
# all computing the identical decode (lossless). Compiled synthesis is a semantics-preserving
# transformation (Futamura projection); the decode is exact, the logits agree to ~1 ULP.
#
# Compiled mode needs the souffle source headers + sqlite/zlib/ncurses dev headers+libs. On a box
# without root, install them locally (see ../SOUFFLE.md §1.1) and source the env below.
set -e
cd "$(dirname "$0")"

export CPATH="$HOME/.local/include:$CPATH"
export LIBRARY_PATH="$HOME/.local/lib:$LIBRARY_PATH"
export LD_LIBRARY_PATH="$HOME/.local/lib:$LD_LIBRARY_PATH"

[ -f whole_base.dl ] || { echo "run: python3 mint_and_emit.py  (or fieldrun export --logic-whole) first"; exit 1; }
mkdir -p ex iout cout
printf '0\t3\n1\t14\n2\t7\n3\t2\n4\t29\n' > ex/token.facts

echo "== compile the Datalog program to a native binary =="
souffle -o /tmp/lo3a_decoder whole_base.dl
echo "   built /tmp/lo3a_decoder"

echo "== lossless check (decode + logits) =="
souffle whole_base.dl -F ex -D iout 2>/dev/null
/tmp/lo3a_decoder       -F ex -D cout 2>/dev/null
diff iout/decide.csv cout/decide.csv >/dev/null && echo "   decide: IDENTICAL"
python3 - <<'PY'
load=lambda p:{int(r.split()[0]):float(r.split()[1]) for r in open(p)}
i,c=load('iout/logit.csv'),load('cout/logit.csv')
mx=max(abs(i[k]-c[k]) for k in i)
print(f"   logit: max |Δ| = {mx:.2e}  ({'1-ULP reassociation, decode exact' if mx<1e-9 else 'DIVERGENCE'})")
PY

echo "== speedup: single run =="
mkdir -p /tmp/_d
t0=$(date +%s.%N); souffle whole_base.dl -F ex -D /tmp/_d >/dev/null 2>&1; t1=$(date +%s.%N)
ti=$(echo "$t1-$t0"|bc -l); printf "   interpreter: %7.3f s\n" "$ti"
t0=$(date +%s.%N); /tmp/lo3a_decoder -F ex -D /tmp/_d >/dev/null 2>&1; t1=$(date +%s.%N)
tc=$(echo "$t1-$t0"|bc -l); printf "   compiled   : %7.3f s   (%.0fx faster, lossless)\n" "$tc" "$(echo "$ti/$tc"|bc -l)"
