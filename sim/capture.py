#!/usr/bin/env python3
"""Build the phrasebook (n-gram store), validate the JS engine against the
fieldrun binary, and capture the numbers the site shows.

Pipeline (run after sim/train.py):
  1. Build the store from the TRAIN split with a small unigram floor, so the
     phrasebook only memorises the genuinely-frequent tokens (a large floor would
     mark every rare token 'covered' and COMPOSED could never fire).
  2. Run `fieldrun --attribute` over the holdout to get the real RETRIEVED /
     SELECTED / COMPOSED split, and over each of the three example contexts.
  3. Run `fieldrun --explain` on each example to read the binary's argmax / logit
     / margin, and check the in-browser engine (sim/engine.js, via node) agrees.
  4. Emit sim/data/store.json (shipped to the page) and sim/data/validation.json
     (the 'verified against fieldrun' numbers the page displays).
"""
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path("sim/data")
FR = ["./target/release/fieldrun"]
BUNDLE = str(ROOT / "threx")


def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def build_store(cap_uni, cap, cap_bi):
    train = json.load(open(ROOT / "corpus.json"))["ids"]
    json.dump({"holdout_ids": train}, open(ROOT / "train_ids.json", "w"))
    r = sh([sys.executable, "scripts/build_store.py", "--ids", str(ROOT / "train_ids.json"),
            "-o", str(ROOT / "store.json"), "--cap-uni", str(cap_uni),
            "--cap", str(cap), "--cap-bi", str(cap_bi)])
    print(r.stdout.strip() or r.stderr.strip())


def attribute(ids_path, ctx, n):
    r = sh(FR + ["--bundle", BUNDLE, "--ids", ids_path, "--store", str(ROOT / "store.json"),
                 "--ctx", str(ctx), "--n-eval", str(n), "--attribute"])
    return r.stdout


def route_of(prefix, expect):
    """Route a single decision: append the answer so the prediction position
    exists, then read its RETRIEVED/SELECTED/COMPOSED verdict."""
    p = ROOT / "_one.json"
    json.dump({"holdout_ids": list(prefix) + [expect]}, open(p, "w"))
    out = attribute(str(p), len(prefix), 1)
    m = re.search(r"^\s+(RETRIEVED|SELECTED|COMPOSED)\s+\[", out, re.M)
    via = re.search(r"via (\S+)", out)
    return (m.group(1) if m else "?"), (via.group(1) if via else "?")


def explain(prefix):
    p = ROOT / "_one.json"
    json.dump({"holdout_ids": prefix}, open(p, "w"))
    r = sh(FR + ["--bundle", BUNDLE, "--ids", str(p), "--ctx", str(len(prefix)),
                 "--n-eval", "1", "--explain"])
    o = r.stdout
    pred = re.search(r"model predicts \[(\d+)\]\s+logit ([-\d.]+)\s+\(margin ([+\-\d.]+)", o)
    if not pred:
        return None
    return {"argmax": int(pred.group(1)), "logit": float(pred.group(2)),
            "margin": float(pred.group(3))}


def main():
    lex = json.load(open(ROOT / "lexicon.json"))
    glyph = [t[0] for t in lex["tokens"]]

    # 1. store (tuneable caps via argv: cap_uni cap cap_bi)
    cap_uni = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    cap_bi = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    print(f"== store (cap_uni={cap_uni} cap={cap} cap_bi={cap_bi}) ==")
    build_store(cap_uni, cap, cap_bi)

    # 2. holdout route distribution
    print("\n== fieldrun --attribute over the holdout ==")
    out = attribute(str(ROOT / "holdout.json"), 16, 600)
    for line in out.splitlines():
        if re.search(r"RETRIEVED|SELECTED|COMPOSED|decomposition", line):
            print(" ", line.strip())

    # 3. the three examples: route (fieldrun) + explain (fieldrun) + engine (node)
    print("\n== the three examples ==")
    eng = json.loads(sh(["node", "sim/engine_probe.js"]).stdout or "[]")
    eng = {e["key"]: e for e in eng}
    results = []
    for ex in lex["examples"]:
        route, via = route_of(ex["prefix"], ex["expect"])
        exp = explain(ex["prefix"])
        e = eng.get(ex["key"], {})
        agree = exp and e and exp["argmax"] == e["pred"]
        print(f"  [{ex['key']:9}] {' '.join(glyph[i] for i in ex['prefix'])} "
              f"→ fieldrun={glyph[exp['argmax']] if exp else '?'} "
              f"engine={glyph[e.get('pred', -1)] if e else '?'} "
              f"route={route} via {via}  "
              f"logit={exp['logit'] if exp else '?'} margin={exp['margin'] if exp else '?'}  "
              f"{'✓agree' if agree else '✗MISMATCH'}")
        results.append({**ex, "route": route, "via": via, "fieldrun": exp, "engine": e})

    json.dump({"examples": results, "caps": {"uni": cap_uni, "quad": cap, "bi": cap_bi}},
              open(ROOT / "validation.json", "w"), ensure_ascii=False, indent=1)
    print("\nwrote sim/data/store.json, sim/data/validation.json")


if __name__ == "__main__":
    main()
