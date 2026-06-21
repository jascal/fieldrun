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
        rules.append(f"e{k}(L,I,V) :- e{c}(L,I+1,V).\nn{k}(L,N) :- n{c}(L,Nc), N = Nc-1.")
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


def emit_answer(ast, ctr, rules):
    """Emit prog_answer(l,v). Returns True if the program is supported by the §6 fold translator."""
    op, args = ast
    if op in FOLDS:
        c = emit_list(args[0], ctr, rules)
        if c is None:
            return False
        if op == "first":
            rules.append(f"prog_answer(L,V) :- e{c}(L,0,V).")
        elif op == "last":
            rules.append(f"prog_answer(L,V) :- e{c}(L,I,V), n{c}(L,N), I = N-1.")
        elif op == "len":
            rules.append(f"prog_answer(L,N) :- n{c}(L,N).")
        else:  # max/min/sum recursive fold
            a = f"acc{c}"
            rules.append(f".decl {a}(l:number,i:number,m:number)")
            rules.append(f"{a}(L,0,V) :- e{c}(L,0,V).")
            if op == "sum":
                rules.append(f"{a}(L,I,S) :- {a}(L,I-1,S0), e{c}(L,I,V), S = S0+V.")
            else:
                rules.append(f"{a}(L,I,M) :- {a}(L,I-1,M0), e{c}(L,I,V), M = {op}(M0,V).")
            rules.append(f"prog_answer(L,M) :- n{c}(L,N), {a}(L,N-1,M).")
        return True
    if op == "nth":
        c = emit_list(args[0], ctr, rules)
        if c is None:
            return False
        rules.append(f"prog_answer(L,V) :- e{c}(L,{args[1]},V).")
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


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/listdump_15b.jsonl"
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    outdir = sys.argv[3] if len(sys.argv) > 3 else "/tmp/soufflé_rulesynth"
    os.makedirs(outdir, exist_ok=True)
    by_task = defaultdict(list)
    for line in open(path):
        line = line.strip()
        if line:
            r = json.loads(line); by_task[r["task"]].append(r)
    print(f"# emit-datalog: {path}  (souffle round-trip — does the discovered program + residue reproduce the model?)")
    print(f"{'task':<7}{'discovered program':<28}{'souffle ok':>11}{'residue':>8}{'note':>8}")
    SIZE_PEN = synth.SIZE_PEN
    for task in ["first", "last", "len", "max", "min", "sum"]:
        recs = by_task.get(task)
        if not recs:
            continue
        lists = [r["list"] for r in recs]
        target = [r["out"] for r in recs]
        levels, _ = synth.enumerate_programs(lists, K=K)
        progs = [p for p in synth.all_int_programs(levels) if all(v is None or 0 <= v <= 9 for v in p.vals)]
        idx = list(range(len(lists)))
        progs.sort(key=lambda p: (-(synth.faith(p.vals, target, idx) - SIZE_PEN * p.size), p.size))
        best = progs[0]
        ast = parse_repr(best.rep)
        built = build_dl(ast, lists, list(best.vals), target)
        if built is None:
            print(f"{task:<7}{best.rep:<28}{'—':>11}{'—':>8}{'unsupported-op':>14}")
            continue
        dl, nres = built
        f = os.path.join(outdir, f"{task}.dl")
        open(f, "w").write(dl)
        try:
            subprocess.run(["souffle", "-D", outdir, f], check=True, capture_output=True, timeout=60)
            out = {}
            for ln in open(os.path.join(outdir, "answer.csv")):
                a, b = ln.split()
                out[int(a)] = int(b)
            ok = sum(1 for lid in range(len(lists)) if out.get(lid) == target[lid]) / len(lists)
            print(f"{task:<7}{best.rep:<28}{ok*100:>10.0f}%{nres*100//len(lists):>7}%{'':>8}")
        except Exception as e:
            print(f"{task:<7}{best.rep:<28}{'ERR':>11}{'':>8}  {str(e)[:30]}")


if __name__ == "__main__":
    main()
