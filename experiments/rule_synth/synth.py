#!/usr/bin/env python3
"""Bottom-up faithful function synthesizer (RULE_EXTRACTION_PROPOSAL §8).

Reads a (task, list, model-output, truth) JSONL dump from `fieldrun --recursion-explain --list-dump` and, per task,
synthesizes the smallest program over a typed DSL that reproduces the MODEL's output (faithful, "wrong"-by-textbook
allowed) — by bottom-up enumeration with OBSERVATIONAL-EQUIVALENCE pruning (the bank is keyed by behaviour, so its
size tracks distinct behaviours, not program count). Then a guarded/piecewise (decision-list) pass captures heuristic
blends, with an MDL penalty and a held-out split as overfit controls. Reports per task: best single program + its
held-out faithfulness, the guarded program + faithfulness, the residue, and what it recovered (clean vs broken).

No deps beyond the stdlib.
"""
import json, sys, itertools
from collections import defaultdict, Counter

SIZE_PEN = 0.012  # MDL: small per-node penalty so a simpler near-best program wins ties (shared with emit_datalog.py)

# ---------- DSL ----------
# A program is (typ, size, repr, fn) where fn: list[int] -> value (int | tuple | bool | None). None = undefined.
def _safe(fn):
    def g(*a):
        try:
            v = fn(*a)
            return v
        except Exception:
            return None
    return g

# list -> int
L2I = {
    "first": _safe(lambda l: l[0] if l else None),
    "last":  _safe(lambda l: l[-1] if l else None),
    "len":   _safe(lambda l: len(l)),
    "max":   _safe(lambda l: max(l) if l else None),
    "min":   _safe(lambda l: min(l) if l else None),
    "sum":   _safe(lambda l: sum(l)),
    # --- breadth packs (deterministic exhaustion) ---
    "argmax":   _safe(lambda l: l.index(max(l)) if l else None),     # pos
    "argmin":   _safe(lambda l: l.index(min(l)) if l else None),     # pos
    "countmax": _safe(lambda l: l.count(max(l)) if l else None),     # hist
    "nuniq":    _safe(lambda l: len(set(l))),                        # hist
    "maxcount": _safe(lambda l: max(Counter(l).values()) if l else None),  # hist (the mode's frequency)
}

# which breadth pack each primitive belongs to ("base" = the original DSL, always on)
PACK_OF = {"argmax": "pos", "argmin": "pos", "countmax": "hist", "nuniq": "hist", "maxcount": "hist"}
# list -> list
L2L = {
    "tail":    _safe(lambda l: tuple(l[1:])),
    "init":    _safe(lambda l: tuple(l[:-1])),
    "reverse": _safe(lambda l: tuple(reversed(l))),
    "sort":    _safe(lambda l: tuple(sorted(l))),
}
# (list,int) -> int
LI2I = {
    "nth":   _safe(lambda l, k: l[k] if (k is not None and 0 <= k < len(l)) else None),
    "count": _safe(lambda l, v: l.count(v) if v is not None else None),
}
# (int,int) -> int
II2I = {
    "add":  _safe(lambda a, b: a + b),
    "sub":  _safe(lambda a, b: a - b),
    "mul":  _safe(lambda a, b: a * b),
    "imin": _safe(lambda a, b: min(a, b)),
    "imax": _safe(lambda a, b: max(a, b)),
}
# (list,int) -> list
LI2L = {
    "take": _safe(lambda l, k: tuple(l[:k]) if k is not None else None),
    "drop": _safe(lambda l, k: tuple(l[k:]) if k is not None else None),
}

class P:
    __slots__ = ("typ", "size", "rep", "vals")
    def __init__(self, typ, size, rep, vals):
        self.typ, self.size, self.rep, self.vals = typ, size, rep, vals


def enumerate_programs(lists, K=4, bank_cap=60000, enabled=None):
    """Bottom-up + observational equivalence. `enabled` = set of breadth packs in use (None ⇒ all; "base" always on).
    Returns levels[typ][size] -> list[P] (smallest program per behaviour)."""
    ok = lambda nm: enabled is None or PACK_OF.get(nm, "base") in enabled
    seen = {"int": {}, "list": {}}          # sig -> size (smallest seen)
    levels = {"int": defaultdict(list), "list": defaultdict(list)}

    def add(p):
        b = seen[p.typ]
        prev = b.get(p.vals)
        if prev is None or p.size < prev:
            b[p.vals] = p.size
            levels[p.typ][p.size].append(p)
            return True
        return False

    # terminals
    add(P("list", 1, "xs", tuple(tuple(l) for l in lists)))
    for c in range(10):
        add(P("int", 1, str(c), tuple(c for _ in lists)))

    def of(typ, size):
        return levels[typ][size]

    for size in range(2, K + 1):
        cs = size - 1
        # list -> int
        for ch in of("list", cs):
            for nm, f in L2I.items():
                if not ok(nm): continue
                add(P("int", size, f"{nm}({ch.rep})", tuple(f(list(v)) if v is not None else None for v in ch.vals)))
        # list -> list
        for ch in of("list", cs):
            for nm, f in L2L.items():
                if not ok(nm): continue
                add(P("list", size, f"{nm}({ch.rep})", tuple(f(list(v)) if v is not None else None for v in ch.vals)))
        # binary: split sizes s1 + s2 = size - 1
        for s1 in range(1, cs):
            s2 = cs - s1
            li, ii = of("list", s1), of("int", s2)
            for a in li:
                for b in ii:
                    for nm, f in LI2I.items():
                        if not ok(nm): continue
                        add(P("int", size, f"{nm}({a.rep},{b.rep})", tuple(f(list(x), y) if (x is not None) else None for x, y in zip(a.vals, b.vals))))
                    for nm, f in LI2L.items():
                        if not ok(nm): continue
                        add(P("list", size, f"{nm}({a.rep},{b.rep})", tuple(f(list(x), y) if (x is not None) else None for x, y in zip(a.vals, b.vals))))
            ia, ib = of("int", s1), of("int", s2)
            for a in ia:
                for b in ib:
                    for nm, f in II2I.items():
                        if not ok(nm): continue
                        add(P("int", size, f"{nm}({a.rep},{b.rep})", tuple(f(x, y) if (x is not None and y is not None) else None for x, y in zip(a.vals, b.vals))))
        if sum(len(v) for v in seen.values()) > bank_cap:
            break
    return levels, seen


def all_int_programs(levels):
    return [p for s in sorted(levels["int"]) for p in levels["int"][s]]


def faith(vals, target, idx):
    """fraction of examples in idx where vals == target."""
    if not idx:
        return 0.0
    return sum(1 for i in idx if vals[i] == target[i]) / len(idx)


# ---------- guards (decision list) ----------
def predicate_programs(lists, levels):
    """Small fixed set of boolean predicates over the input (for guard conditions)."""
    preds = []
    xs = [tuple(l) for l in lists]
    preds.append(("is_sorted", tuple(tuple(x) == tuple(sorted(x)) for x in xs)))
    preds.append(("first==max", tuple((x[0] == max(x)) if x else False for x in xs)))
    preds.append(("first==min", tuple((x[0] == min(x)) if x else False for x in xs)))
    preds.append(("last==max", tuple((x[-1] == max(x)) if x else False for x in xs)))
    for k in (3, 4, 5):
        preds.append((f"len>{k}", tuple(len(x) > k for x in xs)))
    return preds


def decision_list(progs, preds, target, tr, te, max_rules=3, margin=0.05):
    """Greedy decision list: prefer SPECIFIC guards whose subset accuracy beats the default, reserve None for the
    default/last rule. Each guard must beat the current default by `margin` on its fired subset (anti-overfit), and
    the whole list is accepted only if it beats the best-single on HELD-OUT (else we return the default alone).
    Returns (rules, train_acc, test_acc). rules = [(pred_name or None, prog)]."""
    predmap = dict(preds)
    topk = progs[:60]
    best_default = max(progs, key=lambda p: faith(p.vals, target, tr))
    rules = []
    remaining = set(tr)
    for _ in range(max_rules):
        if not remaining:
            break
        # default accuracy on the still-uncovered train inputs
        def_acc = max((sum(1 for i in remaining if p.vals[i] == target[i]) / len(remaining)) for p in topk)
        # best SPECIFIC guard: a predicate + program with high accuracy on its fired (uncovered) subset
        best = None
        for pr_name, pr_vals in preds:
            sel = [i for i in remaining if pr_vals[i]]
            if len(sel) < max(4, len(remaining) // 12):
                continue
            for p in topk:
                acc = sum(1 for i in sel if p.vals[i] == target[i]) / len(sel)
                score = acc * len(sel)            # accuracy-weighted coverage
                if acc >= def_acc + margin and (best is None or score > best[0]):
                    best = (score, pr_name, pr_vals, p, sel)
        if best is None:
            break
        _, pr_name, pr_vals, p, sel = best
        rules.append((pr_name, p))
        remaining -= set(sel)                     # ordered list: remove everything the guard fired on
    # default last rule
    if remaining:
        dp = max(topk, key=lambda p: sum(1 for i in remaining if p.vals[i] == target[i]))
        rules.append((None, dp))

    def apply_dl(vals_idx):
        for pr_name, p in rules:
            if pr_name is None or predmap[pr_name][vals_idx]:
                return p.vals[vals_idx]
        return best_default.vals[vals_idx]

    tr_acc = sum(1 for i in tr if apply_dl(i) == target[i]) / max(1, len(tr))
    te_acc = sum(1 for i in te if apply_dl(i) == target[i]) / max(1, len(te))
    # accept the guarded program only if it helps on HELD-OUT by a MEANINGFUL margin (else it's noise) — fall back to best-1
    b1_te = faith(best_default.vals, target, te)
    if te_acc <= b1_te + 0.04 or len(rules) <= 1:
        return [(None, best_default)], faith(best_default.vals, target, tr), b1_te
    return rules, tr_acc, te_acc


def rules_repr(rules):
    out = []
    for nm, p in rules:
        out.append(f"if {nm}: {p.rep}" if nm else f"else: {p.rep}")
    return "  |  ".join(out)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/listdump.jsonl"
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    by_task = defaultdict(list)
    for line in open(path):
        line = line.strip()
        if line:
            r = json.loads(line)
            by_task[r["task"]].append(r)
    split = "OOD-length (train ≤5, test ≥6)" if "--ood" in sys.argv else "random 70/30"
    print(f"# rule-synth: {path} · DSL depth K={K} · split={split}  (faith = held-out faithfulness to the MODEL's output)")
    print(f"# columns: model-acc = how often the model = textbook; best-1 = simplest near-best program (MDL);")
    print(f"#          truth% = how often the discovered program = textbook (low ⇒ a faithful BROKEN function).\n")
    print(f"{'task':<7}{'n':>4}{'model':>7}  {'best-1 program':<28}{'faith':>6}{'truth':>6}  {'guarded (decision list)':<46}{'g-faith':>7}{'resid':>6}")
    import random
    rng = random.Random(0)
    want = next((set(a.split("=", 1)[1].split(",")) for a in sys.argv if a.startswith("--tasks=")), None)
    packs = next((set(a.split("=", 1)[1].split(",")) for a in sys.argv if a.startswith("--packs=")), {"base", "pos", "hist"})
    packs.add("base")  # base DSL is always on
    rows = []
    for task in by_task:                              # infer tasks from the dump (any --list-dump battery)
        if want and task not in want:                 # optional --tasks=first,last filter for large dumps
            continue
        recs = by_task.get(task)
        if not recs:
            continue
        lists = [r["list"] for r in recs]
        target = [r["out"] for r in recs]            # FAITHFUL target: the model's output
        truth = [r["truth"] for r in recs]
        n = len(recs)
        model_acc = sum(1 for i in range(n) if target[i] == truth[i]) / n
        if "--ood" in sys.argv:                      # OOD-length split: train on short lists, test on long
            tr = [i for i in range(n) if len(lists[i]) <= 5]
            te = [i for i in range(n) if len(lists[i]) >= 6]
            if not te or not tr:
                continue
        else:
            idx = list(range(n)); rng.shuffle(idx)
            cut = int(0.7 * n); tr, te = idx[:cut], idx[cut:]
        levels, seen = enumerate_programs(lists, K=K, enabled=packs)
        # output-type constraint: the answer is a single digit, so only programs whose values are all in 0..9 (or None)
        # are admissible — this drops out-of-range junk like add(1,9)=10 that can never be the model's function.
        progs = [p for p in all_int_programs(levels) if all(v is None or (0 <= v <= 9) for v in p.vals)]
        # MDL selection on TRAIN: faith - SIZE_PEN*size → the simplest near-best behaviour (first beats min(init(init)))
        progs.sort(key=lambda p: (-(faith(p.vals, target, tr) - SIZE_PEN * p.size), p.size))
        best1 = progs[0]
        b1_te = faith(best1.vals, target, te)
        truth1 = sum(1 for i in range(n) if best1.vals[i] == truth[i]) / n
        preds = predicate_programs(lists, levels)
        rules, g_tr, g_te = decision_list(progs, preds, target, tr, te)
        resid = 1.0 - g_te
        bank = sum(len(v) for v in seen.values())
        guard_s = rules_repr(rules) if len(rules) > 1 else "(no guard helps)"
        print(f"{task:<7}{n:>4}{model_acc*100:>6.0f}%  {best1.rep:<28}{b1_te*100:>5.0f}%{truth1*100:>5.0f}%  {guard_s:<46}{g_te*100:>6.0f}%{resid*100:>5.0f}%")
        rows.append((task, model_acc, b1_te, truth1, g_te))
    print(f"\n# bank ≈ {bank} programs (observational-equivalence working set, K={K})")
    mean_resid = sum(1 - r[4] for r in rows) / max(1, len(rows))
    print(f"# mean residue (held-out, guarded) across tasks = {mean_resid*100:.0f}%  ← the measured 'forge tax' for this DSL")


if __name__ == "__main__":
    main()
