#!/usr/bin/env python3
"""Scope coverage report (roadmap step 2.5 — the real tail test).

Runs the faithful synthesizer (`synth.py`) across a BROAD problem battery (`fieldrun --scope-dump`) and asks the
question the narrow 10-task sweep couldn't: across MANY problems, does a small deterministic DSL + a small reused
rule-library cover most of them (a SHORT HEAD — the forge tax is tractable), or does each problem need bespoke rules
(a LONG TAIL — the kernel never closes)? Reports, per task, the honest (OOD-length) residue; then the coverage curve,
the head/tail partition, the primitive-reuse profile over the head, and the scope-level mean forge tax.

Usage:  python scope_report.py /tmp/scopedump.jsonl [K]   (default K=4; add --random for the random split)
"""
import json, sys, re
from collections import defaultdict, Counter
import synth

PRIMS = set(synth.L2I) | set(synth.L2L) | set(synth.LI2I) | set(synth.II2I) | set(synth.LI2L)


def prims_of(rep):
    return {t for t in re.findall(r"[a-z][a-z0-9]*", rep) if t in PRIMS}


def synth_task(recs, K, ood):
    lists = [r["list"] for r in recs]
    target = [r["out"] for r in recs]
    truth = [r["truth"] for r in recs]
    n = len(recs)
    model_acc = sum(1 for i in range(n) if target[i] == truth[i]) / n
    if ood:
        tr = [i for i in range(n) if len(lists[i]) <= 5]
        te = [i for i in range(n) if len(lists[i]) >= 6]
        if not tr or not te:
            import random
            idx = list(range(n)); random.Random(0).shuffle(idx); cut = int(0.7 * n); tr, te = idx[:cut], idx[cut:]
    else:
        import random
        idx = list(range(n)); random.Random(0).shuffle(idx); cut = int(0.7 * n); tr, te = idx[:cut], idx[cut:]
    levels, _ = synth.enumerate_programs(lists, K=K, enabled={"base", "pos", "hist"})
    progs = [p for p in synth.all_int_programs(levels) if all(v is None or (0 <= v <= 9) for v in p.vals)]
    progs.sort(key=lambda p: (-(synth.faith(p.vals, target, tr) - synth.SIZE_PEN * p.size), p.size))
    best1 = progs[0]
    preds = synth.predicate_programs(lists, levels)
    rules, _, g_te = synth.decision_list(progs, preds, target, tr, te)
    rep = synth.rules_repr(rules) if len(rules) > 1 else best1.rep
    # A "constant" fit (rep is a bare digit) reproduces the model only because the model is DEGENERATE on this task
    # (near-constant output) — it is not genuine DSL coverage. Flag it so it doesn't inflate the head.
    is_const = best1.rep.strip() in {str(i) for i in range(10)} and len(rules) <= 1
    return {"task": recs[0]["task"], "n": n, "model_acc": model_acc, "rep": rep,
            "faith": g_te, "resid": 1 - g_te, "prims": prims_of(rep), "const": is_const}


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/scopedump.jsonl"
    K = next((int(a) for a in sys.argv[2:] if a.isdigit()), 4)
    ood = "--random" not in sys.argv
    by_task = defaultdict(list)
    for line in open(path):
        if line.strip():
            r = json.loads(line); by_task[r["task"]].append(r)
    split = "OOD-length (train≤5,test≥6)" if ood else "random 70/30"
    rows = [synth_task(rs, K, ood) for rs in by_task.values()]
    rows.sort(key=lambda r: r["resid"])

    print(f"# scope coverage · {path} · K={K} · split={split} · {len(rows)} problems\n")
    print(f"{'task':<10}{'n':>4}{'model':>7}  {'discovered (best-1 / guarded)':<40}{'faith':>6}{'resid':>6}")
    for r in rows:
        print(f"{r['task']:<10}{r['n']:>4}{r['model_acc']*100:>6.0f}%  {r['rep'][:40]:<40}{r['faith']*100:>5.0f}%{r['resid']*100:>5.0f}%")

    resids = [r["resid"] for r in rows]
    mean = sum(resids) / len(resids)
    print(f"\n# scope-level mean forge tax (OOD) = {mean*100:.0f}%   (vs the narrow 10-task battery)")

    # coverage curve: fraction of problems under a residue threshold
    print("# coverage curve — fraction of problems with residue ≤ t:")
    for t in (0.05, 0.10, 0.20, 0.33, 0.50):
        frac = sum(1 for x in resids if x <= t) / len(resids)
        bar = "█" * round(frac * 30)
        print(f"#   resid≤{int(t*100):>2}% : {frac*100:>3.0f}%  {bar}")

    # head/tail partition — and split the head into GENUINE (a real program fits a competent model) vs DEGENERATE
    # (a constant fits because the model's output is near-constant; the model can't do the task).
    head = [r for r in rows if r["resid"] <= 0.15]
    tail = [r for r in rows if r["resid"] >= 0.50]
    mid = [r for r in rows if 0.15 < r["resid"] < 0.50]
    genuine = [r for r in head if not r["const"]]
    degenerate = [r for r in head if r["const"]]
    deg_str = ", ".join("{}={}@{:.0f}%".format(r["task"], r["rep"], r["model_acc"] * 100) for r in degenerate)
    print(f"\n# HEAD (resid≤15%): {len(head)}/{len(rows)}  =  {len(genuine)} GENUINE-function "
          f"({', '.join(r['task'] for r in genuine)})")
    print(f"#                    +  {len(degenerate)} DEGENERATE/constant-fit (model can't do it, emits ~constant: {deg_str})")
    print(f"# MID  (15–50%):                 {len(mid)}/{len(rows)} — {', '.join(r['task'] for r in mid)}")
    print(f"# TAIL (resid≥50%, idiosyncratic): {len(tail)}/{len(rows)} — {', '.join(r['task'] for r in tail)}")

    # primitive-reuse profile over the head: does a small reused set cover the head (short head), or many distinct prims?
    head_prims = Counter(p for r in head for p in r["prims"])
    distinct = len(head_prims)
    print(f"\n# primitive reuse over the HEAD: {distinct} distinct primitives cover {len(head)} problems")
    for p, c in head_prims.most_common():
        print(f"#   {p:<10} ×{c}")
    # descriptive reading (not a stamped verdict): report the shape, let the numbers carry it
    print(f"\n# reading: {len(head)} head / {len(mid)} mid / {len(tail)} tail of {len(rows)}; "
          f"head covered by {distinct} distinct primitives; scope mean forge tax {mean*100:.0f}%.")
    print(f"#   head/tail ratio = {len(head)}:{len(tail)}; primitive concentration = "
          f"{(sum(c for _, c in head_prims.most_common(3)) / max(1, sum(head_prims.values())))*100:.0f}% "
          f"of head-primitive uses are the top 3. (short-head ⟺ many problems, few reused primitives; "
          f"long-tail ⟺ residue mass spread over many idiosyncratic sites.)")


if __name__ == "__main__":
    main()
