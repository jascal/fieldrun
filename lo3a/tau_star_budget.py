#!/usr/bin/env python3
"""R2 step-budget control: isolate whether the trained tied head's high-rank shortfall (at 150 steps) is
UNDER-FITTING (shrinks with steps) or a genuine frozen-SVD advantage. Same model, one residual capture,
tied-only training swept over step budgets at r≈PR and r≈span90. Writes lo3a/tau_star_budget.json."""
import argparse, json, os
import numpy as np
from tau_star_trained import capture_hf, frozen_lens, train_head, eval_head
HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--steps", nargs="+", type=int, default=[150, 400, 800, 1600])
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=os.path.join(HERE, "tau_star_budget.json"))
    args = ap.parse_args()

    U, H_res, a_star, strs = capture_hf(args.model, "f32", 256, args.device)
    d = U.shape[1]; n = len(a_star); ntr = n // 2
    Htr, atr = H_res[:ntr], a_star[:ntr]; Hte, ate, str_te = H_res[ntr:], a_star[ntr:], strs[ntr:]
    # PR / span90 from the competitor-diff spectrum
    rows = []
    for i in range(len(atr)):
        a = atr[i]; sc = U @ Htr[i]; comp = np.argsort(sc)[::-1]; comp = comp[comp != a][:8]
        for v in comp: rows.append((U[a] - U[v]) / (np.linalg.norm(U[a] - U[v]) + 1e-30))
    sv = np.linalg.svd(np.asarray(rows), full_matrices=False)[1]; e = sv ** 2
    PR = int(round((e.sum() ** 2) / (e ** 2).sum())); span90 = int(np.searchsorted(np.cumsum(e) / e.sum(), 0.90) + 1)
    ranks = {"PR": PR, "span90": span90}
    print(f"== R2 budget control: {args.model} (d={d}, n={n}, PR={PR}, span90={span90}) ==")

    rec = {"model": args.model, "d": int(d), "PR": PR, "span90": span90, "lr": args.lr, "by_rank": {}}
    for name, r in ranks.items():
        B0, A0, _ = frozen_lens(U, Htr, atr, r)
        fro = eval_head(U, Hte, ate, B0, A0, str_te)
        row = {"rank": r, "frozen_open_R32": fro["R32_open"], "frozen_R32": fro["R32"], "steps": {}}
        print(f"  {name} (r={r}): frozen open-R@32 {100*fro['R32_open']:.0f}%  (closed {100*fro['R32_closed']:.0f}%)")
        for st in args.steps:
            Bt, Ct = train_head(U, Htr, atr, B0, A0, r, st, args.lr, args.device, mode="tied")
            ti = eval_head(U, Hte, ate, Bt, Ct, str_te)
            row["steps"][str(st)] = {"open_R32": ti["R32_open"], "R32": ti["R32"], "R1": ti["R1"],
                                     "closed_R32": ti["R32_closed"]}
            print(f"      tied @ {st:>4} steps: open-R@32 {100*ti['R32_open']:>3.0f}%  "
                  f"(tied−frozen {100*(ti['R32_open']-fro['R32_open']):+.0f}pp)  R@1 {100*ti['R1']:>3.0f}%", flush=True)
        rec["by_rank"][name] = row
        json.dump(rec, open(args.out, "w"), indent=2)
    json.dump(rec, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
