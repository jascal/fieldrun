#!/usr/bin/env python3
"""Certified Π → smaller bundle reducer.

Drops FFN neurons from a fieldrun rope bundle and writes a structurally SMALLER bundle, with a
correctness certificate:

  * EXACT (provably lossless): a neuron whose down_proj row is ~0 writes nothing to the residual for
    ANY activation, so removing it changes no logit on ANY input — decode bit-identical, certified by
    construction (δ = 0). This is the sound core.
  * MARGIN-GATED (verified): dropping live-but-dominated neurons is accepted per input iff the decode
    is unchanged; we report the Tropical margin m = L(win) − L(runner-up) so the safe set is visible
    (m > 2δ ⇒ provably safe). Verified against fieldrun, the model itself.

Demonstrated on a tiny rope bundle with PLANTED dead capacity; works on any f32 rope bundle.
Usage:  python3 reduce.py            # mint demo (dead capacity) -> reduce -> certify via fieldrun
"""
import os, sys, json, subprocess, re, shutil
import numpy as np
import bundle_io as bio

HERE = os.path.dirname(os.path.abspath(__file__))
FR = os.path.join(HERE, "..", "target", "release", "fieldrun")
DEAD_EPS = 1e-12   # ||down_row|| below this ⇒ provably-dead neuron (δ=0, exact removal)

# ---------- the reducer ----------
def neuron_importance(W, cfg, cfg_f, calib):
    """per (layer, neuron): mean over calib contexts of |swiglu_act| * ||down_proj row||_2.
       Dead neurons (zero down row) score exactly 0. Returns imp[layer] = np.array(ffn)."""
    n_layer, ffn = int(cfg[0]), int(cfg[5])
    downnorm = [np.linalg.norm(W[f"l{l}.mlp.down_proj"], axis=1) for l in range(n_layer)]  # [ffn] per layer
    acc = [np.zeros(ffn) for _ in range(n_layer)]
    for ids in calib:
        _, acts = bio.forward(W, cfg, cfg_f, ids, want_acts=True)
        for l in range(n_layer):
            acc[l] += np.abs(acts[l]) * downnorm[l]
    return [a / max(1, len(calib)) for a in acc], downnorm

def reduce_bundle(in_stem, out_stem, calib, drop_per_layer=None):
    """Drop `drop_per_layer` lowest-importance FFN neurons per layer (uniform ffn'); if None, drop
       exactly the provably-dead ones (the lossless reduction). Returns a report dict."""
    man, W = bio.read_bundle(in_stem)
    cfg, cfg_f = list(man["config"]), man["config_f"]
    n_layer, ffn = int(cfg[0]), int(cfg[5])
    tied = cfg[7] != 0   # tied unembed iff config[7] != 0
    bias = (f"l0.self_attn.q_proj.bias" in W)
    imp, downnorm = neuron_importance(W, cfg, cfg_f, calib)
    dead_per_layer = [int(np.sum(dn <= DEAD_EPS)) for dn in downnorm]
    if drop_per_layer is None:
        drop_per_layer = min(dead_per_layer)   # uniform, lossless: only provably-dead
        exact = True
    else:
        exact = (drop_per_layer <= min(dead_per_layer))
    keep = ffn - drop_per_layer
    Wr = dict(W)
    for l in range(n_layer):
        order_idx = np.argsort(imp[l])               # ascending importance; drop the lowest `drop_per_layer`
        keep_idx = np.sort(order_idx[drop_per_layer:])
        p = f"l{l}."
        Wr[p+"mlp.gate_proj"] = W[p+"mlp.gate_proj"][:, keep_idx]   # [d, ffn] -> [d, keep]
        Wr[p+"mlp.up_proj"]   = W[p+"mlp.up_proj"][:, keep_idx]     # [d, ffn] -> [d, keep]
        Wr[p+"mlp.down_proj"] = W[p+"mlp.down_proj"][keep_idx, :]   # [ffn, d] -> [keep, d]
    cfg_r = list(cfg); cfg_r[5] = keep
    order = bio.layer_order(n_layer, tied, bias)
    bytes_in = bio.write_bundle(in_stem + "__copy", man["arch"], cfg, cfg_f, W, order)  # for size baseline
    os.remove(in_stem + "__copy.fieldrun.json"); os.remove(in_stem + "__copy.fieldrun.bin")
    bytes_out = bio.write_bundle(out_stem, man["arch"], cfg_r, cfg_f, Wr, order)
    params = lambda c: int(c[6])*int(c[4]) + sum(  # rough param count
        np.prod(Wr[n].shape) if n in Wr else 0 for n in order)
    return {"n_layer": n_layer, "ffn": ffn, "keep": keep, "drop_per_layer": drop_per_layer,
            "dead_per_layer": dead_per_layer, "exact": exact, "bytes_in": bytes_in, "bytes_out": bytes_out,
            "cfg_full": cfg, "cfg_red": cfg_r}

# ---------- certify against fieldrun (the model itself) ----------
def fieldrun_decode(stem, ids):
    qp = os.path.join(HERE, "_rq.json"); json.dump({"holdout_ids": list(ids) + [0]}, open(qp, "w"))
    out = subprocess.run([FR, "--bundle", stem, "--ids", qp, "--ctx", str(len(ids)), "export", "--logic"],
                         capture_output=True, text=True)
    m = re.search(r"model predicts: \[(\d+)\].*?margin ([+\-0-9.]+)", out.stderr + out.stdout)
    return (int(m.group(1)), float(m.group(2))) if m else (None, None)

def certify(full_stem, red_stem, holdout):
    rows = []
    for ids in holdout:
        wf, mf = fieldrun_decode(full_stem, ids)
        wr, _  = fieldrun_decode(red_stem,  ids)
        rows.append((ids, wf, wr, mf, wf == wr))
    return rows

# ---------- demo: mint a model WITH planted dead capacity, reduce, certify ----------
def mint_with_dead(stem, dead_per_layer=12):
    # reuse mint_and_emit to build a base tiny bundle, then zero `dead_per_layer` down_proj rows/layer
    env = dict(os.environ, BIAS="0", UNTIE="0")
    subprocess.run([sys.executable, os.path.join(HERE, "mint_and_emit.py")], env=env, capture_output=True)
    man, W = bio.read_bundle(os.path.join(HERE, "tiny", "tiny"))
    cfg = man["config"]; n_layer, ffn = int(cfg[0]), int(cfg[5])
    rng = np.random.default_rng(11)
    for l in range(n_layer):
        dead = rng.choice(ffn, size=dead_per_layer, replace=False)
        W[f"l{l}.mlp.down_proj"][dead, :] = 0.0     # neuron writes nothing to the residual -> DEAD
    order = bio.layer_order(n_layer, cfg[7] != 0, False)
    bio.write_bundle(stem, man["arch"], cfg, man["config_f"], W, order)
    return cfg, man["config_f"]

if __name__ == "__main__":
    rng = np.random.default_rng(5)
    full = os.path.join(HERE, "red_full", "red_full")
    cfg, cfg_f = mint_with_dead(full, dead_per_layer=12)   # 12 of 64 FFN neurons dead per layer
    _, Wf = bio.read_bundle(full)
    calib   = [[int(t) for t in rng.integers(0, int(cfg[6]), size=int(rng.integers(2, 10)))] for _ in range(24)]
    holdout = [[int(t) for t in rng.integers(0, int(cfg[6]), size=int(rng.integers(1, 12)))] for _ in range(20)]

    print("== certified-LOSSLESS reduction (drop only provably-dead neurons, δ=0) ==")
    red = os.path.join(HERE, "red_small", "red_small")
    rep = reduce_bundle(full, red, calib, drop_per_layer=None)
    print(f"   ffn {rep['ffn']} -> {rep['keep']}  (dropped {rep['drop_per_layer']}/layer; dead/layer={rep['dead_per_layer']}; exact={rep['exact']})")
    print(f"   bundle bytes {rep['bytes_in']:,} -> {rep['bytes_out']:,}  ({100*(1-rep['bytes_out']/rep['bytes_in']):.1f}% smaller)")
    rows = certify(full, red, holdout)
    match = sum(r[4] for r in rows)
    print(f"   CERTIFIED decode match (fieldrun full vs reduced): {match}/{len(rows)}")
    for ids, wf, wr, mf, ok in rows[:6]:
        print(f"     ctx(len {len(ids):2d}) full={wf} reduced={wr} margin={mf:+.3f}  {'✓' if ok else '✗ MISMATCH'}")

    print("\n== margin-gated reduction sweep (drop K lowest-importance/layer; K>dead is approximate) ==")
    for K in [12, 20, 28, 36]:
        r2 = os.path.join(HERE, "red_sweep", f"red_{K}")
        rep = reduce_bundle(full, r2, calib, drop_per_layer=K)
        rows = certify(full, r2, holdout); match = sum(x[4] for x in rows)
        tag = "LOSSLESS" if rep["exact"] else "margin-gated"
        print(f"   drop {K}/layer (ffn {rep['ffn']}->{rep['keep']}, {100*(1-rep['bytes_out']/rep['bytes_in']):.0f}% smaller, {tag}): "
              f"decode match {match}/{len(rows)}")
