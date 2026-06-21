#!/usr/bin/env python3
"""PIC residue layer (roadmap step 3 — the soft representation, PR as the irreducibility label).

The crisp synthesizer (`synth.py`) is the T=0 / tropical limit: it picks the single best program (argmax faith) and
calls the rest residue. PIC (PIC_PROPOSAL.md) is the T=1 soft generalization: the residue is an INCIDENCE MIXTURE over
candidate programs, and the **participation ratio (PR)** of candidate participation is the *irreducibility label*
(PIC separates the retrievable fragment — compact, low-PR — from the computed fragment — diffuse, high-PR = the forge tax).

For each problem we: take the top-M candidate programs, build the incidence I[i,m] = 1[program m matches the model at
example i], and over the RESIDUE of the best single program compute —
  • unexplained%  : residue examples matched by NO candidate  → genuinely outside the crisp family (the hard forge tax);
  • PR (T=1)      : 1 / Σ_m p_m² over candidate responsibility mass p_m (soft, exp(faith/T)-weighted incidence) —
                    PR≈1 ⟺ one extra program carries the residue (reducible to a tiny decision-list / ensemble);
                    PR large ⟺ residue spread diffusely over many programs (no compact symbolic form);
  • ensemble@90   : greedy set-cover — how many programs to explain 90% of the (explainable) residue.

A problem's label: crisp-reducible (head) / ensemble-reducible (small PR, small cover) / PIC-irreducible (high
unexplained% or high PR with no small cover). T=0 recovers the crisp synth (PR→1, pick-one); T=1 is the soft reading.

Usage:  python pic_residue.py /tmp/scopedump.jsonl [K] [--tasks=a,b] [--M=40] [--T=1.0] [--random]
"""
import json, sys, math
from collections import defaultdict
import synth


def faith_all(vals, target):
    n = len(target)
    return sum(1 for i in range(n) if vals[i] == target[i]) / n


def analyse(recs, K, M, T, ood):
    lists = [r["list"] for r in recs]
    target = [r["out"] for r in recs]
    truth = [r["truth"] for r in recs]
    n = len(recs)
    model_acc = sum(1 for i in range(n) if target[i] == truth[i]) / n
    # honest split: choose best1 + the covering ensemble on TRAIN, validate coverage on held-out TEST.
    if ood:
        tr = [i for i in range(n) if len(lists[i]) <= 5]
        te = [i for i in range(n) if len(lists[i]) >= 6]
        if not tr or not te:
            cut = int(0.7 * n); tr, te = list(range(cut)), list(range(cut, n))
    else:
        cut = int(0.7 * n); tr, te = list(range(cut)), list(range(cut, n))
    levels, _ = synth.enumerate_programs(lists, K=K, enabled={"base", "pos", "hist"})
    progs = [p for p in synth.all_int_programs(levels) if all(v is None or (0 <= v <= 9) for v in p.vals)]
    progs.sort(key=lambda p: -faith_all([p.vals[i] for i in tr], [target[i] for i in tr]))  # rank by TRAIN faith
    cands = progs[:M]
    best1 = cands[0]
    others = cands[1:]
    rtr = [i for i in tr if best1.vals[i] != target[i]]    # train residue (where best1 fails)
    rte = [i for i in te if best1.vals[i] != target[i]]    # held-out residue
    resid_frac = len(rte) / max(1, len(te))
    if not rtr or not rte:
        lab = "crisp-reducible (head)" if resid_frac <= 0.15 else "crisp (no train/test residue overlap)"
        return {"task": recs[0]["task"], "n": n, "model_acc": model_acc, "best": best1.rep,
                "resid": resid_frac, "unexp": 0.0, "pr": 1.0, "ens": 0, "hocov": 1 - resid_frac, "label": lab}
    inc_tr = {m: [1 if others[m].vals[i] == target[i] else 0 for i in rtr] for m in range(len(others))}
    # greedy set-cover of TRAIN residue → ordered ensemble + marginal gains (the honest "family of algorithms")
    need = set(range(len(rtr)))
    covered, ensemble, gains = set(), [], []
    while len(covered) < math.ceil(0.9 * len(rtr)) and len(ensemble) < len(others):
        m = max(inc_tr, key=lambda m: len({j for j in need if inc_tr[m][j]} - covered))
        g = {j for j in need if inc_tr[m][j]} - covered
        if not g:
            break
        covered |= g; ensemble.append(m); gains.append(len(g))
    ens = len(ensemble)
    # PR from the cover's MARGINAL GAINS (effective ensemble size — NOT inflated by coincidental candidates)
    tot = sum(gains) or 1
    p = [g / tot for g in gains]
    pr = 1.0 / sum(x * x for x in p) if p else 1.0
    # held-out validity: does the train-chosen ensemble (+best1) explain the HELD-OUT residue? + unexplained by ANY cand
    ens_cov_te = sum(1 for i in rte if any(others[m].vals[i] == target[i] for m in ensemble)) / len(rte)
    unexp = sum(1 for i in rte if all(c.vals[i] != target[i] for c in others)) / len(rte)
    if resid_frac <= 0.15:
        label = "crisp-reducible (head)"
    elif unexp >= 0.5:
        label = f"PIC-irreducible — {unexp*100:.0f}% of held-out residue outside the crisp family"
    elif ens_cov_te >= 0.6 and pr <= 4:
        label = f"ensemble-reducible (~{ens} rules, PR={pr:.1f}, held-out cover {ens_cov_te*100:.0f}%)"
    else:
        label = f"diffuse — train cover doesn't generalize (held-out {ens_cov_te*100:.0f}%, PR={pr:.1f})"
    return {"task": recs[0]["task"], "n": n, "model_acc": model_acc, "best": best1.rep,
            "resid": resid_frac, "unexp": unexp, "pr": pr, "ens": ens, "hocov": ens_cov_te, "label": label}


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/scopedump.jsonl"
    K = next((int(a) for a in sys.argv[2:] if a.isdigit()), 4)
    M = next((int(a.split("=")[1]) for a in sys.argv if a.startswith("--M=")), 40)
    T = next((float(a.split("=")[1]) for a in sys.argv if a.startswith("--T=")), 1.0)
    ood = "--random" not in sys.argv
    want = next((set(a.split("=", 1)[1].split(",")) for a in sys.argv if a.startswith("--tasks=")), None)
    by_task = defaultdict(list)
    for line in open(path):
        if line.strip():
            r = json.loads(line); by_task[r["task"]].append(r)
    rows = [analyse(rs, K, M, T, ood) for t, rs in by_task.items() if not want or t in want]
    rows.sort(key=lambda r: r["resid"])
    print(f"# PIC residue · {path} · K={K} · M={M} candidates · T={T} · split={'OOD' if ood else 'random'}\n")
    print(f"{'task':<10}{'resid':>6}{'unexp':>6}{'PR':>5}{'ens':>4}{'ho-cov':>7}  {'label':<52}")
    for r in rows:
        print(f"{r['task']:<10}{r['resid']*100:>5.0f}%{r['unexp']*100:>5.0f}%{r['pr']:>5.1f}{r['ens']:>4}{r['hocov']*100:>6.0f}%  {r['label']:<52}")
    tail = [r for r in rows if r["resid"] > 0.15]
    if tail:
        red = [r for r in tail if r["label"].startswith("ensemble")]
        irr = [r for r in tail if "irreducible" in r["label"]]
        dif = [r for r in tail if r["label"].startswith("diffuse")]
        print(f"\n# of {len(tail)} non-head problems: {len(red)} ensemble-reducible, {len(irr)} PIC-irreducible, {len(dif)} diffuse/noise.")
        print(f"# mean unexplained-residue (outside the crisp family) over non-head = "
              f"{sum(r['unexp'] for r in tail)/len(tail)*100:.0f}%  ← the part with no compact symbolic form (PIC's computed fragment).")


if __name__ == "__main__":
    main()
