#!/usr/bin/env python3
"""Regression test for the demand-closure checker (`demand_closure.py`).

The emitted `.dl` programs are not committed (lo3a tracks only scripts), so this
test carries a tiny hand-written final-stratum fixture inline and asserts the
checker's verdict on it. Run: `python3 lo3a/test_demand_closure.py` (exit 0 = pass).

The verdict is also a kernel theorem: i-orca `ProvableOpt_Checker.echeck` is a
*verified* (sound + complete) decision procedure for the same `syn_demand_closed`
condition this checker computes, so a green run here corresponds to a kernel-proved
losslessness for any program of this shape.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from demand_closure import analyse  # noqa: E402

# A one-"layer" final stratum: residual x feeds per-position keys/values that
# attention reads at ALL positions (so x/k/v are live everywhere), then the
# post-attention cone (attn, ssf, xf) is read only at lastpos by the query.
FIXTURE = r"""
.decl token(pos:number, id:number)
.decl x(pos:number, d:number, v:float)
.decl k(pos:number, d:number, v:float)
.decl val(pos:number, d:number, v:float)
.decl attn(pos:number, d:number, v:float)
.decl ssf(pos:number, s:float)
.decl xf(pos:number, d:number, v:float)
.decl lastpos(p:number)
.decl logit(t:number, s:float)
.decl decide(t:number)

x(P,D,V)   :- token(P,_), embed(D,V).
k(P,D,W)   :- x(P,D,W).
val(P,D,W) :- x(P,D,W).
attn(I,D,S) :- token(I,_), S = sum (KV*VV) : { token(J,_), k(J,_,KV), val(J,D,VV) }.
ssf(P,S)   :- token(P,_), S = sum (A*A) : { attn(P,_,A) }.
xf(P,D,W)  :- attn(P,D,A), ssf(P,SS), W = A*SS.
lastpos(P) :- P = max Q : { token(Q,_) }.
logit(T,S) :- vocab(T), lastpos(LP), S = sum (XV*EV) : { xf(LP,D,XV), unembed(T,D,EV) }.
decide(T)  :- logit(T,S), S = max S2 : { logit(_,S2) }.
"""

EXPECT_DROPPABLE = {"attn", "ssf", "xf"}   # the post-attention cone (lastpos-only)
EXPECT_LIVE = {"x", "k", "val"}            # feed attention at all positions


def main() -> int:
    res = analyse(FIXTURE)
    drop = set(res["droppable_off_lastpos"])
    live = set(res["live_all_positions"])

    ok = True
    missing = EXPECT_DROPPABLE - drop
    if missing:
        print(f"FAIL: expected droppable, not certified: {sorted(missing)}")
        ok = False
    leaked = EXPECT_LIVE & drop
    if leaked:
        print(f"FAIL (unsound!): live relations wrongly marked droppable: {sorted(leaked)}")
        ok = False
    missing_live = EXPECT_LIVE - live
    if missing_live:
        print(f"FAIL: expected live, not classified ALL: {sorted(missing_live)}")
        ok = False

    if ok:
        print(f"OK: droppable={sorted(drop)}  live={sorted(live)}")
        print("    (post-attention cone is lastpos-only; attention-feeding relations stay live)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
