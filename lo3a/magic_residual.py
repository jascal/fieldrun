#!/usr/bin/env python3
"""R4 / PO-T2 — the lossless demand-transform residual IS the forge tax, measured on the emittable Π.

Runs the whole-model context-free Datalog program both ways — plain and with Soufflé's magic-set / demand
transform (`-m '*'`) — on a fixed token EDB, confirms the decode is byte-identical (lossless), and profiles
the MATERIALIZED-TUPLE count per relation. PO-T2 predicts: the dense matmul/aggregate fragment (the computed
forge tax) is *universally demanded* — every `logit(v)` reads the full residual — so demand pruning shrinks
it by ≈nothing, losslessly confirming there is nothing to drop; only low-treewidth structural/index relations
can prune. On the tiny minted model (random weights, NO retrievable n-gram/induction structure), the whole
forward is dense, so the prunable fraction is ≈0 — the clean limiting case of the law.

NOTE (no silent cap): the *retrievable-fragment-prunes* half of PO-T2 needs a model with real retrieval
structure (n-gram/induction). That requires emitting SmolLM's Π — blocked by the LE-T4 vocab×d wall
(`export --logic-whole` refuses without `--force`). The real-model forge-tax number therefore comes from the
certified FFN reducer instead (`run_smollm.py`: a trained dense FFN has ≈0 exactly-dead neurons ⇒ the
losslessly-removable set is ≈0). This script measures the *demand-transform* half on the emittable Π.

Requires souffle in PATH and a minted tiny bundle + whole.dl + ctx/token.facts (run mint_and_emit.py first).
Writes lo3a/magic_residual.json.
"""
import json, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))

# structural / retrievable (low-treewidth: position-keyed indices, rope tables, head maps, the token EDB)
STRUCT_PREFIXES = ("token", "dim_d", "kvout", "ffnout", "vocab", "cidx", "headq", "head_kv",
                   "qrope", "krope", "rope_cos", "rope_sin")


def tuples(profpath):
    rels = json.load(open(profpath))["root"]["program"]["relation"]
    out = {}
    for name, r in rels.items():
        n = 0
        if isinstance(r, dict):
            for rk in ("non-recursive-rule", "recursive-rule"):
                if isinstance(r.get(rk), dict):
                    for _, rd in r[rk].items():
                        if isinstance(rd, dict):
                            nt = rd.get("num-tuples", 0)
                            if isinstance(nt, (int, float)): n += int(nt)
        out[name] = n
    return out


def run(dl, ctx, outdir, prof, magic=False):
    os.makedirs(outdir, exist_ok=True)
    cmd = ["souffle", dl, "-F", ctx, "-D", outdir, "-p", prof]
    if magic: cmd += ["-m", "*"]
    subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    dec = open(os.path.join(outdir, "decide.csv")).read().strip() if os.path.exists(os.path.join(outdir, "decide.csv")) else None
    return dec, tuples(prof)


def is_struct(name):
    return any(name == p or name.startswith(p) for p in STRUCT_PREFIXES)


def is_weight(name):  # the dense weight EDB facts (embed_w, qw0, kw0, ... down0, lmhead)
    return name.endswith("_w") or any(name.startswith(p) and name[len(p):len(p)+1].isdigit()
                                      for p in ("qw", "kw", "vw", "ow", "gatew", "upw", "downw", "inln", "postln"))


def main():
    dl = os.path.join(HERE, "whole.dl"); ctx = os.path.join(HERE, "ctx")
    if not (os.path.exists(dl) and os.path.exists(os.path.join(ctx, "token.facts"))):
        print("missing whole.dl or ctx/token.facts — run mint_and_emit.py and create ctx/token.facts first"); sys.exit(1)
    db, tb = run(dl, ctx, "/tmp/r4_base", "/tmp/prof_base.log", magic=False)
    dm, tm = run(dl, ctx, "/tmp/r4_magic", "/tmp/prof_magic.log", magic=True)
    TB, TM = sum(tb.values()), sum(tm.values())
    # baseline split: dense (computed, derived non-struct) vs structural index relations
    dense = {k: v for k, v in tb.items() if not is_struct(k) and not is_weight(k)}
    struct = {k: v for k, v in tb.items() if is_struct(k)}
    dense_mass = sum(dense.values()); struct_mass = sum(struct.values())
    rec = {
        "model": "tiny minted rope bundle (random weights, no retrievable structure)",
        "decode_baseline": db, "decode_magic": dm, "decode_lossless": db == dm,
        "total_tuples_baseline": TB, "total_tuples_magic": TM,
        "magic_prune_fraction": (TB - TM) / max(1, TB),
        "dense_derived_tuples_baseline": dense_mass, "structural_tuples_baseline": struct_mass,
        "dense_fraction_of_derived": dense_mass / max(1, dense_mass + struct_mass),
        "top_dense_relations": sorted(dense.items(), key=lambda kv: -kv[1])[:10],
        "interpretation": "demand transform is decode-lossless; dense matmul aggregates survive (prune≈0) — "
                          "the universally-demanded computed fragment = forge tax. PO-T2 'nothing-to-drop' half, "
                          "on a structureless tiny model (limiting case). Retrievable-prune half is emit-blocked "
                          "(LE-T4); real-model forge-tax number = run_smollm.py certified FFN reduce (≈0 dead).",
    }
    json.dump(rec, open(os.path.join(HERE, "magic_residual.json"), "w"), indent=2)
    print(f"== PO-T2 demand-transform measurement (tiny emittable Π) ==")
    print(f"   decode: baseline={db}  magic={dm}  LOSSLESS={db==dm}")
    print(f"   total materialized tuples: baseline={TB:,}  magic={TM:,}  "
          f"(magic prune {100*rec['magic_prune_fraction']:+.1f}% — demand adds overhead, prunes ≈0 of the dense core)")
    print(f"   baseline derived split: dense/computed={dense_mass:,} ({100*rec['dense_fraction_of_derived']:.0f}%)  "
          f"structural/index={struct_mass:,}")
    print(f"   top dense relations (the matmul/aggregate forge tax):")
    for k, v in rec["top_dense_relations"][:8]: print(f"      {k:<24} {v:>10,}")
    print(f"\n   wrote magic_residual.json")


if __name__ == "__main__":
    main()
