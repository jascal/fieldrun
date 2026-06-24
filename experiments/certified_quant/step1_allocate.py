#!/usr/bin/env python3
"""Step 1 v1: certified mixed-precision allocator.
Decides which int8 write-tensors (attn/mlp outputs) can drop to int4 while the margin certificate
guarantees the (int8-baseline) decode is preserved on a calibration corpus.

  perturbation of downgrading block b to int4:  delta_b(x) = q4_rel * max_v |contrib[b][v]|
  certificate (per kept position x):             2 * sum_{b in int4} delta_b(x) < margin(x)

Greedy maximizes bytes saved (blocks with most bytes per unit perturbation go int4 first) subject to the
certificate on every kept position; residue excuses the smallest-margin (forge-tax) positions.

Block order (model.rs:294): 0=embed; layer l -> 2l+1 attn, 2l+2 mlp.
Embed (block 0) is the READ-OUT frame: quantizing it shifts every logit via tied U_v, which this
per-block proxy does NOT capture -> kept at its current dtype (f16) pending the frame-quant bound
(PIC_Quant.frame_quant_logit_bound; v1.5). The proxy is also first-order (ignores cross-layer
propagation); end-to-end fidelity is the v2 validator.

Usage: step1_allocate.py manifest.json calib.jsonl [q4_rel=0.0625] [residue=0.10] [out=alloc.json]
"""
import json, sys, re

def block_tensors(b, L):
    """tensor name patterns for block b (model.rs:294 order)."""
    if b == 0: return ["embed"]            # read-out frame (handled specially)
    l = (b - 1) // 2; attn = (b - 1) % 2 == 0
    if attn: return [f"l{l}.self_attn.q_proj", f"l{l}.self_attn.k_proj",
                     f"l{l}.self_attn.v_proj", f"l{l}.self_attn.o_proj"]
    return [f"l{l}.mlp.gate_proj", f"l{l}.mlp.up_proj", f"l{l}.mlp.down_proj"]

def main():
    manifest, calib = sys.argv[1], sys.argv[2]
    q4_rel = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0625      # int4 relative error (~1/16)
    residue = float(sys.argv[4]) if len(sys.argv) > 4 else 0.10
    outpath = sys.argv[5] if len(sys.argv) > 5 else "alloc.json"

    arrs = json.load(open(manifest))["arrays"]
    numel = {a["name"]: (a["shape"][0]*a["shape"][1] if len(a["shape"])==2 else 0) for a in arrs}
    cur   = {a["name"]: a["dtype"] for a in arrs}

    recs = [json.loads(l) for l in open(calib) if l.strip()]
    nb = recs[0]["nb"]; L = (nb-1)//2
    # per-position block sensitivity s_b(x) and margin(x)
    S = [[max(abs(r["contrib"][b][k]) for k in range(len(r["cands"]))) for b in range(nb)] for r in recs]
    M = [r["margin"] for r in recs]                     # model decode margin (top1-top2)
    # bytes a write-block saves going int8 -> int4 (~0.5 byte/elem), and its worst-case sensitivity
    wblocks = list(range(1, nb))                         # exclude embed (block 0)
    save = {b: sum(numel.get(t,0) for t in block_tensors(b,L))*0.5 for b in wblocks}   # bytes saved
    smax = {b: max(S[x][b] for x in range(len(recs))) for b in wblocks}

    # residue: keep all but the smallest-margin residue% positions
    order = sorted(range(len(recs)), key=lambda x: M[x])
    kept = set(order[int(residue*len(recs)):])

    # greedy: add blocks to int4 by bytes-saved-per-sensitivity, if every kept position stays certified
    running = {x: 0.0 for x in kept}                    # sum_{int4} delta_b(x)
    int4 = []
    for b in sorted(wblocks, key=lambda b: (save[b]/(smax[b]+1e-12)), reverse=True):
        if all(2*(running[x] + q4_rel*S[x][b]) < M[x] for x in kept):
            int4.append(b)
            for x in kept: running[x] += q4_rel*S[x][b]
    # build allocation: int4 blocks -> int4; other write blocks stay int8; embed stays current
    alloc = {}
    for b in int4:
        for t in block_tensors(b, L): alloc[t] = "int4"
    # report
    tot8 = sum(numel.get(t,0) for b in wblocks for t in block_tensors(b,L))          # write-tensor elems (int8: 1B)
    saved = sum(save[b] for b in int4)
    write_bytes_int8 = tot8 * 1.0
    write_bytes_mixed = write_bytes_int8 - saved
    embed_bytes = numel.get("embed",0)*2                                             # f16, fixed
    print(f"=== certified mixed-precision allocation (q4_rel={q4_rel}, residue={int(residue*100)}%) ===")
    print(f"  write blocks: {len(int4)}/{len(wblocks)} -> int4  (rest stay int8; embed stays {cur.get('embed')})")
    print(f"  write tensors: int8 {write_bytes_int8/1e6:.1f} MB  ->  mixed {write_bytes_mixed/1e6:.1f} MB"
          f"  (saved {saved/1e6:.1f} MB, {100*saved/write_bytes_int8:.0f}% of writes)")
    print(f"  embed (read-out, fixed f16): {embed_bytes/1e6:.1f} MB  [needs frame-quant bound, v1.5]")
    print(f"  full bundle: int8-writes {(write_bytes_int8+embed_bytes)/1e6:.1f} MB"
          f"  ->  certified-mixed {(write_bytes_mixed+embed_bytes)/1e6:.1f} MB")
    # which sublayers went int4?
    a = sum(1 for b in int4 if (b-1)%2==0); m = sum(1 for b in int4 if (b-1)%2==1)
    print(f"  int4 breakdown: {a} attn-blocks, {m} mlp-blocks of {L} layers each")
    json.dump({"dtype_map": alloc, "meta": {"q4_rel": q4_rel, "residue": residue,
               "n_int4_blocks": len(int4), "saved_MB": round(saved/1e6,1)}}, open(outpath,"w"), indent=1)
    print(f"  wrote {outpath} ({len(alloc)} tensors -> int4)")

if __name__ == "__main__": main()
