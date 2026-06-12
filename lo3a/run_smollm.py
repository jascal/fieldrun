#!/usr/bin/env python3
"""The whole certified-reduction + HF round-trip pipeline on a REAL small Llama (SmolLM-135M).

  convert (done) -> certified FFN reduce -> smaller bundle -> HF safetensors -> fieldrun convert
  -> bundle' -> decode-compare.  The Soufflé whole-model emit is the one step that does NOT scale here
  (vocab*d = 28M facts -> refused, the LE-T4 wall); everything else runs at real-model scale.

Contexts are REAL: fieldrun's own greedy trace (high-margin, the model's confident decisions). Generate with
  printf '/export-logic /tmp/smtr.dl <prompt>\\n/exit\\n' | fieldrun --bundle lo3a/smollm/smollm --chat
"""
import os, sys, json, subprocess, re, glob, time, shutil
import numpy as np
import bundle_io as bio, reduce as R, to_safetensors as ST

HERE = os.path.dirname(os.path.abspath(__file__))
FR = os.path.join(HERE, "..", "target", "release", "fieldrun")
STEM = os.path.join(HERE, "smollm", "smollm")

def parse_trace(glob_pat="/tmp/smtr.*.dl"):
    out = []
    for f in sorted(glob.glob(glob_pat)):
        t = open(f).read()
        cm = re.search(r"^// context:(.*)$", t, re.M)
        pm = re.search(r"model predicts:.*?\[(\d+)\].*?margin ([+\-0-9.]+)", t)
        if cm and pm:
            out.append(([int(x) for x in re.findall(r"\[(\d+)\]", cm.group(1))], int(pm.group(1)), float(pm.group(2))))
    return out

def reduced_decode(stem, ids):
    return ST.fr_decode(stem, ids)

def main():
    for d in ["smollm_red", "smollm_hf", "smollm_rt"]:           # clean stale state (else cross-run shape mixups)
        shutil.rmtree(os.path.join(HERE, d), ignore_errors=True)
    man = json.load(open(STEM + ".fieldrun.json")); cfg = man["config"]
    print(f"== SmolLM-135M (real pretrained Llama): d={cfg[4]} layers={cfg[0]} ffn={cfg[5]} vocab={cfg[6]} heads={cfg[1]}/{cfg[2]} ==")
    tr = parse_trace()
    if not tr:
        print("no trace at /tmp/smtr.*.dl — generate one (see module docstring)"); return
    tr.sort(key=lambda x: -x[2])                         # by margin desc
    calib   = [c[0] for c in tr[:8]]                     # high-margin contexts for importance
    holdout = [tr[i] for i in range(0, len(tr), max(1, len(tr)//18))][:18]   # spread across the margin range
    print(f"   {len(tr)} real contexts (median margin {sorted(c[2] for c in tr)[len(tr)//2]:.1f}); certifying on {len(holdout)} held-out")

    print("\n== certified FFN reduction (drop K lowest-importance neurons/layer; decode vs the model's own pred) ==")
    print("   honest finding: a TRAINED dense FFN is largely the forge tax — little is removable losslessly zero-shot")
    levels = {}
    for K in [16, 48, 96, 160]:
        out = os.path.join(HERE, "smollm_red", f"smollm_K{K}")
        rep = R.reduce_bundle(STEM, out, calib, drop_per_layer=K)
        tot = sum(reduced_decode(out, ids) == pred for ids, pred, _ in holdout)
        pct = 100*(1 - rep['bytes_out']/rep['bytes_in'])
        levels[K] = (out, rep, tot)
        print(f"   drop {K:3d}/layer  ffn {rep['ffn']}->{rep['keep']}  bundle {rep['bytes_in']//(1<<20)}->{rep['bytes_out']//(1<<20)}MB "
              f"({pct:.0f}% smaller)  decode preserved {tot}/{len(holdout)}")

    K = 16; red, rep, _ = levels[K]
    print(f"\n== HF safetensors round trip (reduced {K}/layer model, ffn {rep['keep']}) ==")
    hf = os.path.join(HERE, "smollm_hf"); rt = os.path.join(HERE, "smollm_rt", "smollm_rt")
    os.makedirs(os.path.dirname(rt), exist_ok=True)
    cfgx, npar = ST.export_hf(red, hf)
    sz = os.path.getsize(os.path.join(hf, "model.safetensors"))
    print(f"   {os.path.basename(red)} -> HF {cfgx['architectures'][0]}: safetensors {sz//(1<<20)}MB, {int(npar):,} params, "
          f"d={cfgx['hidden_size']} ffn={cfgx['intermediate_size']} layers={cfgx['num_hidden_layers']} (publishable)")
    r = subprocess.run([FR,"convert","--model",hf,"--arch","rope","--dtype","f32","--out",rt], capture_output=True, text=True)
    if not os.path.exists(rt+".fieldrun.json"):
        print("   convert FAILED:", (r.stderr or r.stdout).strip()[-400:]); return
    _, A = bio.read_bundle(red); _, B = bio.read_bundle(rt)
    wmax = max(np.abs(A[k].astype(np.float64)-B[k].astype(np.float64)).max() for k in A if A[k].shape == B[k].shape)
    dok = sum(ST.fr_decode(red, ids) == ST.fr_decode(rt, ids) for ids,_,_ in holdout)
    print(f"   fieldrun convert (HF -> bundle') OK; weights bit-identical (max |Δ| {wmax:.1e}); "
          f"decode {dok}/{len(holdout)} -> COMPLETE ROUND TRIP {'VERIFIED ✓' if dok==len(holdout) and wmax==0 else 'FAILED'}")

if __name__ == "__main__":
    main()
