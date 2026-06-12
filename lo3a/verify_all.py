#!/usr/bin/env python3
"""Verify the fieldrun `export --logic-whole` emitter across the rope family: base / +bias / +untied /
+bias+untied. For each variant: mint a tiny bundle, have FIELDRUN emit the whole-model .dl, then on a
battery of held-out contexts check Soufflé(decide) == numpy-reference == fieldrun's own forward."""
import subprocess, os, json, random, re, importlib, importlib.util, sys

HERE = os.path.dirname(os.path.abspath(__file__))
FR   = os.path.join(HERE, "..", "target", "release", "fieldrun")

def load_minter():
    # fresh import each call so module-level BIAS/UNTIE (read from env) are re-evaluated
    for m in [k for k in sys.modules if k == "mint_and_emit"]: del sys.modules[m]
    spec = importlib.util.spec_from_file_location("mint_and_emit", os.path.join(HERE, "mint_and_emit.py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

def souffle_decide(dl, ctx):
    d = os.path.join(HERE, "_cv"); os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "token.facts"), "w") as f:
        for p, t in enumerate(ctx): f.write(f"{p}\t{t}\n")
    out = subprocess.run(["souffle", dl, "-F", d, "-D", "-"], capture_output=True, text=True)
    if out.returncode != 0: return ("ERR", out.stderr.strip().splitlines()[-1] if out.stderr else "?")
    grab = False
    for ln in out.stdout.splitlines():
        if ln.startswith("decide"): grab = True; continue
        if grab and ln and ln[0].isdigit(): return int(ln.split()[0])
    return None

def fieldrun_predict(stem, ctx):
    qp = os.path.join(HERE, "_q.json")
    with open(qp, "w") as f: json.dump({"holdout_ids": ctx + [0]}, f)
    out = subprocess.run([FR, "--bundle", stem, "--ids", qp, "--ctx", str(len(ctx)), "export", "--logic"],
                         capture_output=True, text=True)
    mo = re.search(r"model predicts: \[(\d+)\]", out.stderr + out.stdout)
    return int(mo.group(1)) if mo else None

variants = [("base", "0", "0"), ("+bias", "1", "0"), ("+untied", "0", "1"), ("+bias+untied", "1", "1")]
random.seed(2024)
overall = True
for label, bias, untie in variants:
    os.environ["BIAS"], os.environ["UNTIE"] = bias, untie
    m = load_minter()
    m.write_bundle()
    stem = m.STEM
    dl = os.path.join(HERE, f"whole_{label.replace('+','_')}.dl")
    r = subprocess.run([FR, "--bundle", stem, "--ids", os.path.join(HERE,"ids1.json"), "--ctx","5",
                        "export", "--logic-whole", "--out", dl, "--maxpos", "16"],
                       capture_output=True, text=True)
    if not os.path.exists(dl):
        print(f"[{label}] EMIT FAILED: {r.stderr.strip()}"); overall = False; continue
    ok = 0; n = 0
    for _ in range(12):
        L = random.randint(1, 12); ctx = [random.randint(0, m.VOCAB - 1) for _ in range(L)]
        s = souffle_decide(dl, ctx); ref = m.predict(ctx); fr = fieldrun_predict(stem, ctx)
        good = (s == ref == fr); ok += good; n += 1
        if not good: print(f"   MISMATCH ctx={ctx} souffle={s} numpy={ref} fieldrun={fr}")
    print(f"[{label:14s}] {ok}/{n} held-out contexts agree (souffle == numpy == fieldrun)   dl={os.path.basename(dl)}")
    overall &= (ok == n)
print("\n==> ALL VARIANTS VERIFIED" if overall else "\n==> FAILURES PRESENT")
sys.exit(0 if overall else 1)
