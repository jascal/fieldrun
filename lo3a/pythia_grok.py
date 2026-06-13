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
import os, sys, subprocess, re, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FR = os.path.join(HERE, "..", "target", "release", "fieldrun")
IDS = os.path.join(HERE, "..", "..", "lm-sae", "pylm", "holdout_pythia-70m.json")
BDIR = os.path.join(HERE, "..", "pythia")
os.makedirs(BDIR, exist_ok=True)
N_EVAL, CTX = 400, 64
# densified late tail (48k–143k) to confirm the post-plateau PR consolidation is real, not one checkpoint
STEPS = [0,1,2,4,8,16,32,64,128,256,512,1000,2000,3000,4000,6000,8000,16000,32000,
         48000,64000,80000,96000,110000,120000,128000,136000,143000]

def run_step(step):
    stem = os.path.join(BDIR, f"p70m_s{step}")
    if not os.path.exists(stem + ".fieldrun.json"):
        c = subprocess.run([FR,"convert","--model",f"EleutherAI/pythia-70m@step{step}","--arch","neox",
                            "--dtype","f16","-o",stem], capture_output=True, text=True)
        if not os.path.exists(stem + ".fieldrun.json"):
            print(f"  step{step}: convert FAILED ({(c.stderr or c.stdout).strip().splitlines()[-1][:80] if (c.stderr or c.stdout) else '?'})")
            return None
    p = subprocess.run([FR,"--bundle",stem,"--ids",IDS,"--ctx",str(CTX),"--n-eval",str(N_EVAL),"--probe-margin"],
                       capture_output=True, text=True)
    line = next((l for l in (p.stderr+p.stdout).splitlines() if l.startswith("PROBE_MARGIN")), None)
    # free disk + keep reruns clean: drop both the blob and the manifest (next run re-converts from hub cache)
    for ext in (".fieldrun.bin", ".fieldrun.json"):
        try: os.remove(stem + ext)
        except OSError: pass
    if not line: print(f"  step{step}: no PROBE_MARGIN ({(p.stderr).strip().splitlines()[-1][:80] if p.stderr else '?'})"); return None
    d = dict(re.findall(r"(\w[\w.]*)=([-\d.]+)", line))
    rec = {"step": step, **{k: float(v) for k,v in d.items() if k!="n"}}
    print(f"  step{step:>6}: acc={rec['acc']:.1f} margin={rec['margin_med']:.3f} PR={rec['pr_med']:.2f} "
          f"cert(δ=1)={rec['cert_d1']:.1f}% cert(δ=2)={rec['cert_d2']:.1f}%")
    return rec

def main():
    if not os.path.exists(IDS): print(f"missing holdout {IDS}"); sys.exit(1)
    print(f"== PO-T7 grokking order parameter — Pythia-70m, {len(STEPS)} checkpoints, holdout {os.path.basename(IDS)} ==")
    recs = [r for r in (run_step(s) for s in STEPS) if r]
    json.dump(recs, open(os.path.join(HERE,"pythia_grok.json"),"w"), indent=2)

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
    fig.suptitle("PO-T7: grokking order parameter across Pythia-70m training", fontsize=13)
    fig.tight_layout(rect=[0,0,1,0.97])
    out = os.path.join(HERE, "pythia_grok.png"); fig.savefig(out, dpi=110)
    print(f"\nwrote {out} and pythia_grok.json ({len(recs)} checkpoints)")

if __name__ == "__main__":
    main()
