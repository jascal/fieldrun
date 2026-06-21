#!/usr/bin/env python3
"""Emit a discovered program as runnable recursive Soufflé + a residue EDB, and verify it reproduces the MODEL
(RULE_EXTRACTION_PROPOSAL §6). For each task we re-discover the best-1 program (synth.py), translate it to a
position-indexed Datalog fold (`elem(l,i,v)`/`len(l,n)`), add `residue(l,o)` EDB facts for the lists the program gets
wrong, and a wrapper `answer = program unless residue, else residue`. By construction `answer == model output` on
every list (the residue absorbs the program's errors) → the emitted Datalog is 100% output-faithful, and the residue
fact-count is the measured forge tax. We run `souffle` and check `answer` == the model's output for all lists.

Usage: python emit_datalog.py <dump.jsonl> [K] [outdir]
"""
import sys, os, subprocess, json
from collections import defaultdict
import synth

LIST_MODS = {"init", "tail", "reverse", "take", "drop"}
FOLDS = {"first", "last", "len", "max", "min", "sum"}


def parse_repr(s):
    """'min(init(init(xs)))' -> ('min',[('init',[('init',[('xs',[])])])]); 'nth(xs,2)' -> ('nth',[('xs',[]),2])."""
    s = s.strip()
    if s == "xs":
        return ("xs", [])
    if s.isdigit() or (s[0] == "-" and s[1:].isdigit()):
        return int(s)
    i = s.index("(")
    name = s[:i]
    inner = s[i + 1:-1]
    args, depth, cur = [], 0, ""
    for ch in inner:
        if ch == "," and depth == 0:
            args.append(cur); cur = ""
        else:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            cur += ch
    if cur:
        args.append(cur)
    return (name, [parse_repr(a) for a in args])


def emit_list(ast, ctr, rules):
    """Emit relations e{k}(l,i,v)/n{k}(l,n) for a list-typed AST node. Returns k, or None if unsupported."""
    op, args = ast
    k = ctr[0]; ctr[0] += 1
    rules.append(f".decl e{k}(l:number,i:number,v:number)\n.decl n{k}(l:number,n:number)")
    if op == "xs":
        rules.append(f"e{k}(L,I,V) :- elem(L,I,V).\nn{k}(L,N) :- len(L,N).")
        return k
    c = emit_list(args[0], ctr, rules)
    if c is None:
        return None
    if op == "init":
        rules.append(f"e{k}(L,I,V) :- e{c}(L,I,V), n{c}(L,Nc), I < Nc-1.\nn{k}(L,N) :- n{c}(L,Nc), N = Nc-1.")
    elif op == "tail":
        rules.append(f"e{k}(L,I,V) :- e{c}(L,J,V), I = J-1, I >= 0.\nn{k}(L,N) :- n{c}(L,Nc), N = Nc-1.")
    elif op == "reverse":
        rules.append(f"e{k}(L,I,V) :- e{c}(L,J,V), n{c}(L,Nc), I = Nc-1-J.\nn{k}(L,N) :- n{c}(L,N).")
    elif op == "take":
        K = args[1]
        rules.append(f"e{k}(L,I,V) :- e{c}(L,I,V), I < {K}.\nn{k}(L,N) :- n{c}(L,Nc), (Nc<{K}, N=Nc ; Nc>={K}, N={K}).")
    elif op == "drop":
        K = args[1]
        rules.append(f"e{k}(L,I,V) :- e{c}(L,J,V), I = J-{K}, I >= 0.\nn{k}(L,N) :- n{c}(L,Nc), (Nc>{K}, N=Nc-{K} ; Nc<={K}, N=0).")
    else:
        return None
    return k


def emit_answer(ast, ctr, rules, ans="prog_answer"):
    """Emit <ans>(l,v) for a scalar program. Returns True if supported by the §6 fold translator. `ans` is
    parameterised so a decision-list (ensemble) can emit several programs side by side."""
    if isinstance(ast, int):                       # a constant program: same digit for every (non-empty) list
        rules.append(f"{ans}(L,{ast}) :- len(L,_).")
        return True
    op, args = ast
    if op in FOLDS:
        c = emit_list(args[0], ctr, rules)
        if c is None:
            return False
        if op == "first":
            rules.append(f"{ans}(L,V) :- e{c}(L,0,V).")
        elif op == "last":
            rules.append(f"{ans}(L,V) :- e{c}(L,I,V), n{c}(L,N), I = N-1.")
        elif op == "len":
            rules.append(f"{ans}(L,N) :- n{c}(L,N).")
        else:  # max/min/sum recursive fold
            a = f"acc{c}"
            rules.append(f".decl {a}(l:number,i:number,m:number)")
            rules.append(f"{a}(L,0,V) :- e{c}(L,0,V).")
            if op == "sum":
                rules.append(f"{a}(L,I,S) :- {a}(L,I-1,S0), e{c}(L,I,V), S = S0+V.")
            else:
                rules.append(f"{a}(L,I,M) :- {a}(L,I-1,M0), e{c}(L,I,V), M = {op}(M0,V).")
            rules.append(f"{ans}(L,M) :- n{c}(L,N), {a}(L,N-1,M).")
        return True
    if op == "nth":
        c = emit_list(args[0], ctr, rules)
        if c is None:
            return False
        rules.append(f"{ans}(L,V) :- e{c}(L,{args[1]},V).")
        return True
    return False  # binary int ops (add/imax/…), count, sort — not in the §6 fold translator yet


def build_dl(prog_ast, lists, preds, target):
    rules = [".decl elem(l:number,i:number,v:number)\n.decl len(l:number,n:number)",
             ".decl prog_answer(l:number,v:number)\n.decl residue(l:number,o:number)\n.decl residue_l(l:number)",
             ".decl answer(l:number,v:number)\n.output answer"]
    ctr = [0]
    if not emit_answer(prog_ast, ctr, rules):
        return None
    facts = []
    for lid, l in enumerate(lists):
        facts.append(f"len({lid},{len(l)}).")
        for i, v in enumerate(l):
            facts.append(f"elem({lid},{i},{v}).")
    # residue: lists where the program's prediction != the model's output → store the model output
    res = []
    for lid in range(len(lists)):
        if preds[lid] != target[lid]:
            res.append(f"residue({lid},{target[lid]}).")
    rules.append("residue_l(L) :- residue(L,_).")
    rules.append("answer(L,V) :- prog_answer(L,V), !residue_l(L).")
    rules.append("answer(L,O) :- residue(L,O).")
    return "\n".join(rules) + "\n" + "\n".join(facts) + "\n" + "\n".join(res) + "\n", len(res)


def emit_pred(name, ctr, rules):
    """Emit a guard relation g(l) for a decision-list predicate. Returns the relation name, or None if unsupported."""
    g = f"g{ctr[0]}"; ctr[0] += 1
    rules.append(f".decl {g}(l:number)")
    if name.startswith("len>"):
        rules.append(f"{g}(L) :- len(L,N), N>{int(name[4:])}.")
        return g
    if name == "is_sorted":
        ns = f"ns{ctr[0]}"; ctr[0] += 1
        rules.append(f".decl {ns}(l:number)\n{ns}(L) :- elem(L,I,A), elem(L,I+1,B), A>B.")
        rules.append(f"{g}(L) :- len(L,_), !{ns}(L).")
        return g
    parts = {"first==max": ("first(xs)", "max(xs)"), "first==min": ("first(xs)", "min(xs)"), "last==max": ("last(xs)", "max(xs)")}
    if name in parts:
        ra, rb = parts[name]
        a1, b1 = f"pa{ctr[0]}", f"pb{ctr[0]}"; ctr[0] += 1
        rules.append(f".decl {a1}(l:number,v:number)\n.decl {b1}(l:number,v:number)")
        if not emit_answer(parse_repr(ra), ctr, rules, ans=a1) or not emit_answer(parse_repr(rb), ctr, rules, ans=b1):
            return None
        rules.append(f"{g}(L) :- {a1}(L,V), {b1}(L,V).")
        return g
    return None


def build_dl_ensemble(dl_rules, predmap, lists, target):
    """Emit a guarded DECISION LIST (the `ensemble` residue strategy) + a shrunk residue EDB. dl_rules = the
    held-out-gated [(pred_name|None, prog_rep, prog_vals), …] from synth.decision_list. Returns (dl_text, n_residue) or
    None if any program/guard isn't §6-emittable (then the caller falls back to `edb`)."""
    rules = [".decl elem(l:number,i:number,v:number)\n.decl len(l:number,n:number)",
             ".decl residue(l:number,o:number)\n.decl residue_l(l:number)",
             ".decl answer(l:number,v:number)\n.output answer"]
    ctr = [0]
    emitted = []          # (guard_rel|None, ans_rel)
    for j, (pn, rep, _vals) in enumerate(dl_rules):
        ans = f"prog{j}"
        rules.append(f".decl {ans}(l:number,v:number)")
        if not emit_answer(parse_repr(rep), ctr, rules, ans=ans):
            return None
        guard = None
        if pn is not None:
            guard = emit_pred(pn, ctr, rules)
            if guard is None:
                return None
        emitted.append((guard, ans))
    # priority decision list: rule j fires where its guard holds and no EARLIER guard did
    earlier = []
    for guard, ans in emitted:
        neg = "".join(f", !{g}(L)" for g in earlier if g)
        if guard:
            rules.append(f"answer(L,V) :- {guard}(L){neg}, {ans}(L,V), !residue_l(L).")
            earlier.append(guard)
        else:  # default rule (no guard) — fires where no earlier guard held
            rules.append(f"answer(L,V) :- {ans}(L,V){neg}, !residue_l(L).")
    rules.append("residue_l(L) :- residue(L,_).")
    rules.append("answer(L,O) :- residue(L,O).")
    # which examples does the decision list still get wrong? → shrunk residue EDB
    def apply_dl(i):
        for pn, _rep, vals in dl_rules:
            if pn is None or predmap[pn][i]:
                return vals[i]
        return dl_rules[-1][2][i]
    facts, res = [], []
    for lid, l in enumerate(lists):
        facts.append(f"len({lid},{len(l)}).")
        facts += [f"elem({lid},{i},{v})." for i, v in enumerate(l)]
        if apply_dl(lid) != target[lid]:
            res.append(f"residue({lid},{target[lid]}).")
    return "\n".join(rules) + "\n" + "\n".join(facts) + "\n" + "\n".join(res) + "\n", len(res)


def build_dl_margin(head_ast, recs, lists, target, strategy, tau):
    """Margin-routed `ring`/`pic` residue strategy. The crisp head emits as Datalog; residue tokens are routed by the
    per-token MARGIN (`recs[i]['margin']`, from `fieldrun --ring-dump`): low-margin (< tau) → the model's own block-
    provenance semiring-Datalog `Π` (`rlogit(v)=Σ_b cw(b,v)` → argmax = the model token; LE-T5 lossless, the `ring`
    representation; `pic` is the same facts under log-sum-exp); high-margin → a flat EDB. `strategy='ring'` routes ALL
    residue to Π. Returns (dl_text, n_ring, n_edb) or None if the head isn't §6-emittable."""
    rules = [".decl elem(l:number,i:number,v:number)\n.decl len(l:number,n:number)",
             ".decl prog_answer(l:number,v:number)\n.decl residue(l:number,o:number)\n.decl residue_l(l:number)",
             ".decl cw(l:number,b:number,v:number,w:float)\n.decl rlogit(l:number,v:number,s:float)",
             ".decl ring_l(l:number)\n.decl ringans(l:number,v:number)",
             ".decl answer(l:number,v:number)\n.output answer"]
    ctr = [0]
    if not emit_answer(head_ast, ctr, rules):
        return None
    # the model's per-token Π: logit = Σ_b contributions, decode = argmax (max-product / tropical T=0 = `ring`)
    rules.append("rlogit(L,V,S) :- cw(L,_,V,_), S = sum w : { cw(L,B,V,w) }.")
    rules.append("ringans(L,V) :- rlogit(L,V,S), S = max s : { rlogit(L,VV,s) }.")
    cw_facts, res, n_ring, n_edb = [], [], 0, 0
    for lid, r in enumerate(recs):
        # route only the residue tokens (head disagrees with the model)
        if r.get("_head_ok", True):
            continue
        c = r["c"]                                  # nb × 10 contribution matrix
        argmax = max(range(10), key=lambda d: sum(c[b][d] for b in range(len(c))))
        to_ring = (strategy == "ring" or r["margin"] < tau) and argmax == target[lid]  # backstop: Π must reproduce out
        if to_ring:
            for b in range(len(c)):
                for d in range(10):
                    cw_facts.append(f"cw({lid},{b},{d},{c[b][d]:.5f}).")
            cw_facts.append(f"ring_l({lid}).")     # mark as a Π-handled token
            n_ring += 1
        else:
            res.append(f"residue({lid},{target[lid]}).")
            n_edb += 1
    rules.append("residue_l(L) :- residue(L,_).")
    rules.append("residue_l(L) :- ring_l(L).")
    rules.append("answer(L,V) :- prog_answer(L,V), !residue_l(L).")   # crisp head
    rules.append("answer(L,V) :- ring_l(L), ringans(L,V).")           # ring residue (the model's Π)
    rules.append("answer(L,O) :- residue(L,O).")                       # edb residue
    facts = []
    for lid, l in enumerate(lists):
        facts.append(f"len({lid},{len(l)}).")
        facts += [f"elem({lid},{i},{v})." for i, v in enumerate(l)]
    body = "\n".join(rules) + "\n" + "\n".join(facts) + "\n" + "\n".join(res) + "\n" + "\n".join(cw_facts) + "\n"
    return body, n_ring, n_edb


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/listdump_15b.jsonl"
    K = next((int(a) for a in sys.argv[2:] if a.isdigit()), 4)
    outdir = next((a for a in sys.argv[2:] if a.startswith("/") or a.startswith("./")), "/tmp/souffle_rulesynth")
    strategy = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--strategy=")), "edb")
    tau = next((float(a.split("=", 1)[1]) for a in sys.argv if a.startswith("--tau=")), 1.0)  # margin router threshold
    os.makedirs(outdir, exist_ok=True)
    by_task = defaultdict(list)
    for line in open(path):
        line = line.strip()
        if line:
            r = json.loads(line); by_task[r["task"]].append(r)
    print(f"# emit-datalog: {path}  ·  residue strategy = {strategy}  (souffle round-trip — answer == model?)")
    print(f"{'task':<8}{'discovered (head / decision-list)':<34}{'souffle':>8}{'residue':>8}{'note':>10}")
    SIZE_PEN = synth.SIZE_PEN
    import random
    rng = random.Random(0)
    tot_res = 0
    for task in by_task:
        recs = by_task.get(task)
        if not recs:
            continue
        lists = [r["list"] for r in recs]
        target = [r["out"] for r in recs]
        n = len(lists)
        levels, _ = synth.enumerate_programs(lists, K=K)
        progs = [p for p in synth.all_int_programs(levels) if all(v is None or 0 <= v <= 9 for v in p.vals)]
        idx = list(range(n))
        progs.sort(key=lambda p: (-(synth.faith(p.vals, target, idx) - SIZE_PEN * p.size), p.size))
        best = progs[0]
        note, label = "", best.rep
        built = None
        if strategy in ("ring", "margin"):
            if "c" not in recs[0]:
                print(f"{task:<8}{best.rep[:34]:<34}{'—':>8}{'—':>8}{'need --ring-dump':>16}")
                continue
            for lid in range(n):
                recs[lid]["_head_ok"] = (best.vals[lid] == target[lid])
            mb = build_dl_margin(parse_repr(best.rep), recs, lists, target, strategy, tau)
            if mb is not None:
                dl3, n_ring, n_edb = mb
                built = (dl3, n_ring + n_edb); note = f"ring{n_ring}/edb{n_edb}"
        elif strategy == "ensemble":
            sh = list(range(n)); rng.shuffle(sh); cut = int(0.7 * n); tr, te = sh[:cut], sh[cut:]
            preds = synth.predicate_programs(lists, levels)
            dl_rules_p, _gtr, _gte = synth.decision_list(progs, preds, target, tr, te)
            if len(dl_rules_p) > 1:
                dl_rules = [(pn, p.rep, p.vals) for pn, p in dl_rules_p]
                built = build_dl_ensemble(dl_rules, dict(preds), lists, target)
                label = synth.rules_repr(dl_rules_p)
                if built is None:
                    note = "ens-unsupported→edb"
        if built is None:  # edb (default, or ensemble/ring fallback)
            ast = parse_repr(best.rep)
            built = build_dl(ast, lists, list(best.vals), target)
            label = best.rep
        if built is None:
            print(f"{task:<8}{best.rep[:34]:<34}{'—':>8}{'—':>8}{'unsupported':>10}")
            continue
        dl, nres = built
        tot_res += nres
        f = os.path.join(outdir, f"{task}.dl")
        open(f, "w").write(dl)
        try:
            subprocess.run(["souffle", "-D", outdir, f], check=True, capture_output=True, timeout=60)
            out = {}
            for ln in open(os.path.join(outdir, "answer.csv")):
                a, b = ln.split(); out[int(a)] = int(b)
            ok = sum(1 for lid in range(n) if out.get(lid) == target[lid]) / n
            print(f"{task:<8}{label[:34]:<34}{ok*100:>7.0f}%{nres * 100 // n:>7}%{note:>10}")
        except Exception as e:
            print(f"{task:<8}{label[:34]:<34}{'ERR':>8}{'':>8}  {str(e)[:24]}")
    print(f"# total residue facts across tasks ({strategy}) = {tot_res}")


if __name__ == "__main__":
    main()
