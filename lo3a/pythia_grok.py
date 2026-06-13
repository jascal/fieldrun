#!/usr/bin/env python3
"""PO-T7 — the grokking order-parameter experiment on Pythia-70m checkpoints.

For a log-spaced set of training steps (Pythia ships all checkpoints as @stepN branches), convert each to
a fieldrun bundle, run `--probe-margin` on a FIXED real held-out set (the FINDINGS Pythia holdout), and
record the order parameters across training:
  * certifiable-compressible fraction P(margin > 2δ)  — the PO-T3 margin certificate's reach (PROVABLE_OPT PO-T7)
  * median decode margin                              — Grok prediction (i): does it grow with consolidation?
  * median DLA participation ratio (PR)               — circuit concentration: low = consolidated/retrievable
  * top-1 next-token accuracy                         — to separate "certified rises because loss drops" from structure
Plots all four vs log(step), marking the documented Pythia induction-head bump (~step 1k–5k).
"""
import os, sys, subprocess, re, json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FR = os.path.join(HERE, "..", "target", "release", "fieldrun")
# parameterized over the Pythia ladder (R3): default 70m; --model EleutherAI/pythia-160m | pythia-410m
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="EleutherAI/pythia-70m")
ap.add_argument("--n-eval", type=int, default=400)
ARGS, _ = ap.parse_known_args()
MODEL = ARGS.model
TAG = MODEL.split("/")[-1]                                  # pythia-70m / pythia-160m / pythia-410m
SHORT = TAG.replace("pythia-", "p")                        # p70m / p160m / p410m
IDS = os.path.join(HERE, "..", "..", "lm-sae", "pylm", f"holdout_{TAG}.json")
BDIR = os.path.join(HERE, "..", "pythia")
os.makedirs(BDIR, exist_ok=True)
N_EVAL, CTX = ARGS.n_eval, 64
# densified late tail (48k–143k) to confirm the post-plateau PR consolidation is real, not one checkpoint
STEPS = [0,1,2,4,8,16,32,64,128,256,512,1000,2000,3000,4000,6000,8000,16000,32000,
         48000,64000,80000,96000,110000,120000,128000,136000,143000]

def run_step(step):
    stem = os.path.join(BDIR, f"{SHORT}_s{step}")
    if not os.path.exists(stem + ".fieldrun.json"):
        c = subprocess.run([FR,"convert","--model",f"{MODEL}@step{step}","--arch","neox",
                            "--dtype","f16","-o",stem], capture_output=True, text=True)
        if not os.path.exists(stem + ".fieldrun.json"):
            print(f"  step{step}: convert FAILED ({(c.stderr or c.stdout).strip().splitlines()[-1][:80] if (c.stderr or c.stdout) else '?'})")
            return None
    p = subprocess.run([FR,"--bundle",stem,"--ids",IDS,"--ctx",str(CTX),"--n-eval",str(N_EVAL),"--probe-margin"],
                       capture_output=True, text=True)
    lines = (p.stderr + p.stdout).splitlines()
    line = next((l for l in lines if l.startswith("PROBE_MARGIN")), None)
    cline = next((l for l in lines if l.startswith("PROBE_CIRCUITS")), None)
    # free disk + keep reruns clean: drop both the blob and the manifest (next run re-converts from hub cache)
    for ext in (".fieldrun.bin", ".fieldrun.json"):
        try: os.remove(stem + ext)
        except OSError: pass
    if not line: print(f"  step{step}: no PROBE_MARGIN ({(p.stderr).strip().splitlines()[-1][:80] if p.stderr else '?'})"); return None
    d = dict(re.findall(r"(\w[\w.]*)=([-\d.]+)", line))
    rec = {"step": step, **{k: float(v) for k,v in d.items() if k!="n"}}
    if cline:  # PROBE_CIRCUITS n_circuits=.. top=L.H:share|L.H:share|...  — the dominant-circuit fingerprint
        m = re.search(r"top=(\S+)", cline)
        rec["circuits"] = [(seg.split(":")[0], float(seg.split(":")[1])) for seg in m.group(1).split("|")] if m else []
        rec["n_circuits"] = int(re.search(r"n_circuits=(\d+)", cline).group(1)) if "n_circuits=" in cline else 0
    print(f"  step{step:>6}: acc={rec['acc']:.1f} margin={rec['margin_med']:.3f} PR={rec['pr_med']:.2f} "
          f"cert(δ=1)={rec['cert_d1']:.1f}% top-circuits={','.join(c[0] for c in rec.get('circuits',[])[:4])}")
    return rec

def main():
    if not os.path.exists(IDS): print(f"missing holdout {IDS}"); sys.exit(1)
    suffix = "" if TAG == "pythia-70m" else f"_{SHORT}"     # keep the original 70m artifact names stable
    jpath = os.path.join(HERE, f"pythia_grok{suffix}.json")
    print(f"== PO-T7 grokking order parameter — {TAG}, {len(STEPS)} checkpoints, holdout {os.path.basename(IDS)} ==")
    recs = [r for r in (run_step(s) for s in STEPS) if r]
    json.dump(recs, open(jpath, "w"), indent=2)

    # --- R3: characterize WHAT consolidates at the late, certificate-invisible PR event ---
    # locate the late event = the largest single-step drop in median PR among the late tail (step >= 16000)
    tail = [r for r in recs if r["step"] >= 16000 and np.isfinite(r.get("pr_med", float("nan")))]
    if len(tail) >= 2 and all("circuits" in r for r in tail):
        drops = [(tail[i-1], tail[i], tail[i-1]["pr_med"] - tail[i]["pr_med"]) for i in range(1, len(tail))]
        before, after, dPR = max(drops, key=lambda t: t[2])
        sb = {c[0] for c in before["circuits"]}; sa = {c[0] for c in after["circuits"]}
        jac = len(sb & sa) / max(1, len(sb | sa))
        print(f"\n== late PR-consolidation event: step {before['step']}→{after['step']}  "
              f"(PR {before['pr_med']:.1f}→{after['pr_med']:.1f}, ΔPR {dPR:+.1f}; "
              f"acc {before['acc']:.1f}→{after['acc']:.1f}, margin {before['margin_med']:.2f}→{after['margin_med']:.2f}, "
              f"cert(δ1) {before['cert_d1']:.1f}→{after['cert_d1']:.1f}) ==")
        print(f"   dominant-circuit fingerprint Jaccard(before,after) = {jac:.2f}")
        print(f"   left  the top set: {sorted(sb - sa) or '—'}")
        print(f"   enter the top set: {sorted(sa - sb) or '—'}")
        print(f"   persist (rank may shift): {sorted(sb & sa)}")
        # also diff endpoints (early-consolidated vs final) to see net circuit reorganization
        e0 = next((r for r in recs if r["step"] >= 2000 and "circuits" in r), None); eN = recs[-1]
        if e0 and "circuits" in eN:
            s0 = {c[0] for c in e0["circuits"]}; sN = {c[0] for c in eN["circuits"]}
            print(f"   net (step {e0['step']}→{eN['step']}): Jaccard {len(s0&sN)/max(1,len(s0|sN)):.2f}, "
                  f"persist {sorted(s0 & sN)}")
        json.dump({"late_event": {"before": before, "after": after, "dPR": dPR, "jaccard": jac}},
                  open(os.path.join(HERE, f"pythia_grok{suffix}_lateevent.json"), "w"), indent=2)

    st = np.array([r["step"] for r in recs], float); st[st==0] = 0.5   # log axis: step0 -> 0.5
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    def panel(a, ys, title, ylab, color):
        a.semilogx(st, ys, "o-", color=color); a.set_title(title); a.set_ylabel(ylab); a.set_xlabel("training step")
        a.axvspan(1000, 5000, alpha=0.12, color="orange"); a.grid(True, alpha=0.3)
        a.annotate("induction bump", (1500, a.get_ylim()[0]), fontsize=8, color="darkorange")
    panel(ax[0,0], [r["cert_d1"] for r in recs], "Certifiable-compressible fraction  P(margin>2δ), δ=1", "% tokens certified", "C0")
    panel(ax[0,1], [r["margin_med"] for r in recs], "Median decode margin  (Grok prediction i)", "logit margin", "C2")
    panel(ax[1,0], [r["pr_med"] for r in recs], "Median DLA participation ratio  (circuit concentration)", "PR (eff. # circuits)", "C3")
    panel(ax[1,1], [r["acc"] for r in recs], "Top-1 next-token accuracy  (control: confidence vs structure)", "% accuracy", "C1")
    fig.suptitle(f"PO-T7: grokking order parameter across {TAG} training", fontsize=13)
    fig.tight_layout(rect=[0,0,1,0.97])
    out = os.path.join(HERE, f"pythia_grok{suffix}.png"); fig.savefig(out, dpi=110)
    print(f"\nwrote {out} and {os.path.basename(jpath)} ({len(recs)} checkpoints)")

if __name__ == "__main__":
    main()
