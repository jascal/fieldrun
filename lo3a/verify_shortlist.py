#!/usr/bin/env python3
"""Verify the LE-T4 certified-compact unembed shortlist (`export --logic-whole --shortlist K`).

The shortlist keeps the top-K output tokens by ‖U_v‖ and emits a Soufflé certificate `certified()` that fires when the
winner's logit S exceeds ‖x‖·max‖U_elided‖ — at which point the shortlist argmax PROVABLY equals the full-vocab argmax
(no dropped token's logit ⟨x,U_v⟩ ≤ ‖x‖‖U_v‖ can reach S). This script mints a varied-norm tiny untied rope bundle
(real LLMs have large unembed-norm spread; a uniform random init has none, so the certificate is vacuous there), emits
the full and shortlisted whole-model programs, and on a battery of held-out contexts checks:
  • SOUNDNESS — every `certified()` context has shortlist-decide == full-vocab-decide (0 mismatches), and
  • FIRING    — what fraction of contexts the certificate fires on (scales with the norm spread).

Run from the repo root: python lo3a/verify_shortlist.py   (needs `souffle` + a release build of fieldrun).
"""
import subprocess, random, os, sys, tempfile
sys.path.insert(0, "lo3a")
import bundle_io as bio
import numpy as np

FR = "target/release/fieldrun"
HERE = "lo3a"


def mint_varied(src="lo3a/tiny_untied/tiny_untied", dst="lo3a/tiny_varied/tiny_varied", ratio=18.0):
    man, W = bio.read_bundle(src)
    lm = W["lm_head"].astype(np.float32); v = lm.shape[0]
    fac = np.concatenate([np.linspace(ratio ** 0.5, 1.8, v // 3), np.linspace(0.9, 0.2, v - v // 3)])
    W["lm_head"] = lm * fac[:, None]
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    bio.write_bundle(dst, man["arch"], man["config"], man["config_f"], W, [a["name"] for a in man["arrays"]])
    n = np.linalg.norm(W["lm_head"], axis=1)
    return dst, v, n.max() / n.min()


def emit(stem, out, k=None):
    cmd = [FR, "--bundle", stem, "--ids", f"{HERE}/ids1.json", "--ctx", "5", "export", "--logic-whole", "--out", out, "--maxpos", "16"]
    if k is not None:
        cmd += ["--shortlist", str(k)]
    subprocess.run(cmd, capture_output=True)


def decide(dl, ctx):
    d = tempfile.mkdtemp()
    open(os.path.join(d, "token.facts"), "w").write("".join(f"{i}\t{t}\n" for i, t in enumerate(ctx)))
    subprocess.run(["souffle", dl, "-F", d, "-D", d], capture_output=True)
    p = os.path.join(d, "decide.csv"); dec = None
    if os.path.exists(p):
        ls = [l.strip() for l in open(p) if l.strip()]
        if ls: dec = int(ls[0].split()[0])
    cert = os.path.exists(os.path.join(d, "certified.csv")) and os.path.getsize(os.path.join(d, "certified.csv")) > 0
    return dec, cert


def main():
    if not os.path.exists(FR):
        print("build first: cargo build --release --features api"); sys.exit(1)
    stem, vocab, ratio = mint_varied()
    k = max(4, vocab // 3)
    emit(stem, "/tmp/vs_full.dl"); emit(stem, "/tmp/vs_sl.dl", k)
    full_lm = sum(1 for l in open("/tmp/vs_full.dl") if l.startswith("lmhead_w("))
    sl_lm = sum(1 for l in open("/tmp/vs_sl.dl") if l.startswith("lmhead_w("))
    random.seed(0)
    nc, ncert, dis = 0, 0, 0
    agree = 0
    for _ in range(25):
        ctx = [random.randint(0, vocab - 1) for _ in range(random.randint(3, 5))]
        full, _ = decide("/tmp/vs_full.dl", ctx)
        sl, cert = decide("/tmp/vs_sl.dl", ctx)
        nc += 1
        if full == sl: agree += 1
        if cert:
            ncert += 1
            if full != sl:
                dis += 1; print(f"  CERTIFIED MISMATCH ctx={ctx} full={full} sl={sl}")
    print(f"# verify_shortlist · vocab={vocab} K={k} · unembed-norm ratio {ratio:.1f}x · lm_head {full_lm}→{sl_lm} facts")
    print(f"#   {nc} contexts · shortlist==full {agree}/{nc} · CERTIFIED {ncert} ({100*ncert//nc}%) · certified mismatches {dis}")
    print(f"#   {'SOUND' if dis == 0 else 'UNSOUND'} (a certified() context must have shortlist-decide == full-vocab-decide)")
    sys.exit(1 if dis else 0)


if __name__ == "__main__":
    main()
