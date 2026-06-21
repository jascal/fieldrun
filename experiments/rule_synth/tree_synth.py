#!/usr/bin/env python3
"""Tree-traversal faithful synthesizer (RULE_EXTRACTION_PROPOSAL §11 — the untried DETERMINISTIC class).

The flat-list synthesizer (`synth.py`) exhausted the list-fold DSL on flat-list tasks (depth + breadth) and left a
residue floor. §11 is the next deterministic representation it CANNOT express: **tree catamorphisms** (structural
recursion over a parse tree). This reads a (task=eval, expr, model-output, truth) dump from `fieldrun --tree-dump`
— nested arithmetic the model EVALUATES natively (zero-ICL) — parses each expr into a binary tree, and bottom-up
synthesizes the smallest TREE program (a catamorphism over the tree, plus subtree selectors `left`/`right` and int
combinators) faithful to the MODEL's output, with the same OE-pruning / MDL / held-out / guarded machinery.

Punchline (the contrast that justifies the new class): the flat-list DSL run on the SAME exprs (their leaf digit
sequence) cannot express tree `eval` → high residue; the tree DSL's `eval` catamorphism captures it → low residue.
Tree traversal is therefore a NEEDED deterministic class, not a soft one — exactly the "different representation"
the list-floor pointed at. Then §6 closes: best catamorphism → recursive Soufflé over a tree ADT + residue EDB.

No deps beyond the stdlib (+ `souffle` on PATH for the optional round-trip). Reuses synth.py's faith/decision_list.
"""
import json, sys, os, subprocess, tempfile
from collections import defaultdict
import synth                                            # faith, decision_list, P, SIZE_PEN, enumerate_programs (list baseline)

# ---------- parse "(+ 3 (* 2 5))" -> nested tree ----------
# tree = ('L', v:int)  |  ('N', op:str, left, right)
def tokenize(s):
    return s.replace("(", " ( ").replace(")", " ) ").split()

def parse(s):
    toks = tokenize(s)
    t, i = _parse(toks, 0)
    return t

def _parse(toks, i):
    if toks[i] == "(":
        op = toks[i + 1]
        left, i = _parse(toks, i + 2)
        right, i = _parse(toks, i)
        assert toks[i] == ")", f"expected ) at {i}: {toks}"
        return ("N", op, left, right), i + 1
    return ("L", int(toks[i])), i + 1

# ---------- tree DSL (catamorphisms) ----------
def _ev(t):                                              # the recursive arithmetic eval — the key tree-recursion
    if t[0] == "L":
        return t[1]
    a, b = _ev(t[2]), _ev(t[3])
    if a is None or b is None:
        return None
    op = t[1]
    return a + b if op == "+" else a - b if op == "-" else a * b if op == "*" else None

def _leaves(t):
    return [t[1]] if t[0] == "L" else _leaves(t[2]) + _leaves(t[3])

def _safe1(fn):                                          # tree -> int, swallow errors to None
    def g(t):
        try:
            return fn(t)
        except Exception:
            return None
    return g

T2I = {                                                  # tree -> int catamorphisms
    "eval":     _safe1(_ev),
    "maxleaf":  _safe1(lambda t: max(_leaves(t))),
    "minleaf":  _safe1(lambda t: min(_leaves(t))),
    "sumleaf":  _safe1(lambda t: sum(_leaves(t))),
    "nleaves":  _safe1(lambda t: len(_leaves(t))),
    "nops":     _safe1(lambda t: 0 if t[0] == "L" else 1 + T2I["nops"](t[2]) + T2I["nops"](t[3])),
    "depth":    _safe1(lambda t: 0 if t[0] == "L" else 1 + max(T2I["depth"](t[2]), T2I["depth"](t[3]))),
    "leftleaf": _safe1(lambda t: t[1] if t[0] == "L" else T2I["leftleaf"](t[2])),
    "rightleaf":_safe1(lambda t: t[1] if t[0] == "L" else T2I["rightleaf"](t[3])),
}
T2T = {                                                  # tree -> tree (subtree selectors → enable nested catamorphisms)
    "left":  _safe1(lambda t: t[2] if t[0] == "N" else None),
    "right": _safe1(lambda t: t[3] if t[0] == "N" else None),
}
# which T2I primitives have a clean recursive Datalog catamorphism (for §6 emission)
DL_CATA = {"eval", "maxleaf", "minleaf", "sumleaf", "nleaves", "nops", "depth", "leftleaf", "rightleaf"}

P = synth.P
SIZE_PEN = synth.SIZE_PEN
faith = synth.faith


def enumerate_tree(trees, K=4, bank_cap=60000):
    """Bottom-up + observational-equivalence over the TREE DSL. Mirrors synth.enumerate_programs but the base
    terminal is the tree `t`; T2I = tree->int catamorphisms, T2T = subtree selectors, II2I = int combinators."""
    seen = {"int": {}, "tree": {}}
    levels = {"int": defaultdict(list), "tree": defaultdict(list)}

    def add(p):
        b = seen[p.typ]
        prev = b.get(p.vals)
        if prev is None or p.size < prev:
            b[p.vals] = p.size
            levels[p.typ][p.size].append(p)
            return True
        return False

    add(P("tree", 1, "t", tuple(trees)))
    for c in range(10):
        add(P("int", 1, str(c), tuple(c for _ in trees)))

    def of(typ, size):
        return levels[typ][size]

    for size in range(2, K + 1):
        cs = size - 1
        for ch in of("tree", cs):                        # tree -> int
            for nm, f in T2I.items():
                add(P("int", size, f"{nm}({ch.rep})", tuple(f(t) if t is not None else None for t in ch.vals)))
            for nm, f in T2T.items():                    # tree -> tree
                add(P("tree", size, f"{nm}({ch.rep})", tuple(f(t) if t is not None else None for t in ch.vals)))
        for s1 in range(1, cs):                          # (int,int) -> int
            s2 = cs - s1
            for a in of("int", s1):
                for b in of("int", s2):
                    for nm, f in synth.II2I.items():
                        add(P("int", size, f"{nm}({a.rep},{b.rep})",
                              tuple(f(x, y) if (x is not None and y is not None) else None for x, y in zip(a.vals, b.vals))))
        if sum(len(v) for v in seen.values()) > bank_cap:
            break
    return levels, seen


def tree_predicates(trees):
    """Boolean guards over the tree (for the decision-list pass)."""
    preds = []
    preds.append(("is_leaf",  tuple(t[0] == "L" for t in trees)))
    preds.append(("leftleaf==maxleaf", tuple(T2I["leftleaf"](t) == T2I["maxleaf"](t) for t in trees)))
    for k in (1, 2):
        preds.append((f"depth>{k}", tuple((T2I["depth"](t) or 0) > k for t in trees)))
    for op in ("+", "-", "*"):
        preds.append((f"root={op}", tuple(t[0] == "N" and t[1] == op for t in trees)))
    return preds


# ---------- §6: tree-ADT Datalog emission (the recursive catamorphism round-trip) ----------
DL_RULES = {
    "eval":    ('ev', ['ev(t,v):-leaf(t,v).',
                       'ev(t,v):-node(t,"+",l,r),ev(l,a),ev(r,b),v=a+b.',
                       'ev(t,v):-node(t,"-",l,r),ev(l,a),ev(r,b),v=a-b.',
                       'ev(t,v):-node(t,"*",l,r),ev(l,a),ev(r,b),v=a*b.']),
    "maxleaf": ('ml', ['ml(t,v):-leaf(t,v).', 'ml(t,v):-node(t,_,l,r),ml(l,a),ml(r,b),v=max(a,b).']),
    "minleaf": ('nl', ['nl(t,v):-leaf(t,v).', 'nl(t,v):-node(t,_,l,r),nl(l,a),nl(r,b),v=min(a,b).']),
    "sumleaf": ('sl', ['sl(t,v):-leaf(t,v).', 'sl(t,v):-node(t,_,l,r),sl(l,a),sl(r,b),v=a+b.']),
    "nleaves": ('nlv',['nlv(t,1):-leaf(t,_).', 'nlv(t,v):-node(t,_,l,r),nlv(l,a),nlv(r,b),v=a+b.']),
    "nops":    ('no', ['no(t,0):-leaf(t,_).', 'no(t,v):-node(t,_,l,r),no(l,a),no(r,b),v=a+b+1.']),
    "depth":   ('dp', ['dp(t,0):-leaf(t,_).', 'dp(t,v):-node(t,_,l,r),dp(l,a),dp(r,b),v=max(a,b)+1.']),
    "leftleaf":('ll', ['ll(t,v):-leaf(t,v).', 'll(t,v):-node(t,_,l,_),ll(l,v).']),
    "rightleaf":('rl',['rl(t,v):-leaf(t,v).', 'rl(t,v):-node(t,_,_,r),rl(r,v).']),
}

def _flatten(t, nid, leaf, node):
    """Assign integer ids depth-first; append leaf(id,v) / node(id,op,lid,rid) facts. Returns (my_id, next_id)."""
    me = nid[0]; nid[0] += 1
    if t[0] == "L":
        leaf.append((me, t[1]))
    else:
        lid, _ = _flatten(t[2], nid, leaf, node)
        rid, _ = _flatten(t[3], nid, leaf, node)
        node.append((me, t[1], lid, rid))
    return me, nid[0]

def emit_datalog(trees, prog_rep, prog_vals, target, outdir):
    """Emit the discovered catamorphism as recursive Soufflé over a tree ADT + a residue EDB, run souffle, and
    verify answer == model on every expr. Only the primitive name matters for the rule; if it isn't a clean
    catamorphism we route everything to residue (still a faithful round-trip, just no rule)."""
    prim = prog_rep[:prog_rep.index("(")] if "(" in prog_rep else prog_rep
    leaf, node, roots = [], [], []
    nid = [0]
    for t in trees:
        rid, _ = _flatten(t, nid, leaf, node)
        roots.append(rid)
    lines = ['.decl leaf(t:number,v:number)', '.input leaf',
             '.decl node(t:number,op:symbol,l:number,r:number)', '.input node',
             '.decl root(t:number)', '.input root',
             '.decl residue(t:number,v:number)', '.input residue',
             '.decl answer(t:number,v:number)', '.output answer']
    use_rule = prim in DL_RULES
    pred = DL_RULES[prim][0] if use_rule else None
    if use_rule:
        lines.append(f'.decl {pred}(t:number,v:number)')
        lines += DL_RULES[prim][1]
        lines.append(f'answer(t,v):-root(t),{pred}(t,v),!residue(t,_).')
    lines.append('answer(t,v):-residue(t,v).')
    dl = "\n".join(lines) + "\n"
    facts = {"leaf.facts": "".join(f"{i}\t{v}\n" for i, v in leaf),
             "node.facts": "".join(f"{i}\t{op}\t{l}\t{r}\n" for i, op, l, r in node),
             "root.facts": "".join(f"{r}\n" for r in roots)}
    # residue = roots where the rule's value != model output (or no rule at all) → the per-task forge tax
    res = []
    for k, r in enumerate(roots):
        v = prog_vals[k] if use_rule else None
        if (not use_rule) or v is None or v != target[k]:
            res.append((r, target[k]))
    facts["residue.facts"] = "".join(f"{r}\t{v}\n" for r, v in res)
    os.makedirs(outdir, exist_ok=True)
    dlpath = os.path.join(outdir, "tree.dl")
    open(dlpath, "w").write(dl)
    for fn, body in facts.items():
        open(os.path.join(outdir, fn), "w").write(body)
    try:
        subprocess.run(["souffle", "-D", outdir, "-F", outdir, dlpath], check=True,
                       capture_output=True, timeout=120)
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        return None, len(res), f"(souffle unavailable: {type(e).__name__})"
    got = {}
    for line in open(os.path.join(outdir, "answer.csv")):
        a, b = line.split()
        got[int(a)] = int(b)
    match = sum(1 for k, r in enumerate(roots) if got.get(r) == target[k]) / max(1, len(roots))
    return match, len(res), f"rule={prim if use_rule else 'NONE(all-residue)'}"


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/treedump.jsonl"
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    emit = "--emit" in sys.argv
    recs = [json.loads(l) for l in open(path) if l.strip()]
    by_task = defaultdict(list)
    for r in recs:
        by_task[r["task"]].append(r)
    split = "OOD-depth (train depth≤2, test depth≥3)" if "--ood" in sys.argv else "random 70/30"
    print(f"# tree-synth: {path} · tree-DSL depth K={K} · split={split}  (faith = held-out faithfulness to the MODEL)")
    print(f"# columns: model = model-acc vs textbook; cata = discovered catamorphism (MDL); "
          f"list-DSL = best flat-list program's faith (the contrast).\n")
    print(f"{'task':<7}{'n':>4}{'model':>7}  {'tree catamorphism (best-1)':<30}{'faith':>6}{'truth':>6}{'resid':>6}   {'list-DSL faith':>14}")
    import random
    rng = random.Random(0)
    rows = []
    for task, rs in by_task.items():
        trees = [parse(r["expr"]) for r in rs]
        target = [r["out"] for r in rs]
        truth = [r["truth"] for r in rs]
        n = len(rs)
        model_acc = sum(1 for i in range(n) if target[i] == truth[i]) / n
        if "--ood" in sys.argv:
            depths = [T2I["depth"](t) or 0 for t in trees]
            tr = [i for i in range(n) if depths[i] <= 2]
            te = [i for i in range(n) if depths[i] >= 3]
            if not tr or not te:
                idx = list(range(n)); rng.shuffle(idx); cut = int(0.7 * n); tr, te = idx[:cut], idx[cut:]
        else:
            idx = list(range(n)); rng.shuffle(idx); cut = int(0.7 * n); tr, te = idx[:cut], idx[cut:]
        levels, seen = enumerate_tree(trees, K=K)
        progs = [p for s in sorted(levels["int"]) for p in levels["int"][s]]
        progs = [p for p in progs if all(v is None or (0 <= v <= 9) for v in p.vals)]
        progs.sort(key=lambda p: (-(faith(p.vals, target, tr) - SIZE_PEN * p.size), p.size))
        best1 = progs[0]
        b1_te = faith(best1.vals, target, te)
        truth1 = sum(1 for i in range(n) if best1.vals[i] == truth[i]) / n
        preds = tree_predicates(trees)
        rules, g_tr, g_te = synth.decision_list(progs, preds, target, tr, te)
        resid = 1.0 - g_te
        # --- contrast: the flat-list DSL on the SAME exprs (leaf digit sequence) ---
        leaf_lists = [_leaves(t) for t in trees]
        llevels, _ = synth.enumerate_programs(leaf_lists, K=4, enabled={"base"})
        lprogs = [p for p in synth.all_int_programs(llevels) if all(v is None or (0 <= v <= 9) for v in p.vals)]
        lprogs.sort(key=lambda p: -faith(p.vals, target, tr))
        list_faith = faith(lprogs[0].vals, target, te) if lprogs else 0.0
        print(f"{task:<7}{n:>4}{model_acc*100:>6.0f}%  {best1.rep:<30}{b1_te*100:>5.0f}%{truth1*100:>5.0f}%{resid*100:>5.0f}%   {list_faith*100:>13.0f}%")
        rows.append((task, b1_te, resid, list_faith, best1, target))
    mean_resid = sum(r[2] for r in rows) / max(1, len(rows))
    mean_list = sum(1 - r[3] for r in rows) / max(1, len(rows))
    print(f"\n# tree-DSL mean residue = {mean_resid*100:.0f}%   vs   flat-list-DSL mean residue = {mean_list*100:.0f}%"
          f"   (the gap = what tree traversal recovers that the list DSL cannot)")
    if emit:
        print("\n# §6 — recursive tree-ADT Soufflé round-trip (catamorphism + residue EDB):")
        for task, _, _, _, best1, target in rows:
            trees = [parse(r["expr"]) for r in by_task[task]]
            with tempfile.TemporaryDirectory() as d:
                m, nres, note = emit_datalog(trees, best1.rep, best1.vals, target, d)
                ms = "n/a" if m is None else f"{m*100:.0f}%"
                print(f"#   {task:<7} {best1.rep:<24} souffle reproduces model = {ms:<5} residue EDB = {nres} facts  [{note}]")


if __name__ == "__main__":
    main()
