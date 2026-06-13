#!/usr/bin/env python3
"""R1 — cross-architecture / cross-tokenizer validation of  τ* = min(exp(H_output), d).

Every LO1/τ* number in the repo is SmolLM (135M/360M/1.7B): one tokenizer, one rope arch — the very
cross-architecture gap FINDINGS treats as the publication blocker. `exp(H_output)` is tokenizer-dependent,
so a second tokenizer is the test of whether the law is about *language + readout geometry* or about
*SmolLM's BPE*. This script runs the recoverable-rank battery on REAL HF models with each model's own
tokenizer over a FIXED held-out corpus (vary only model/tokenizer), and reports per model:

  (A) Spearman(recoverable_rank, token self-information)   — the per-token info-theoretic law (info_rank.py)
  (B) open- vs closed-class R@k split                       — the grammar-role law (grammar_recall.py)
  (C) median recoverable rank  vs  min(exp(H_output), d)    — the aggregate law (worst_case2.py),
      both the empirical (real-forward) point AND the pure-geometry synthetic-skew sweep on this model's U.

Arch-generic by construction: every transformer ends in `logits = h @ Uᵀ` (+ a monotone softcap that
preserves argmax), so we capture the readout-input residual `h` with a forward-pre-hook on the unembed
and fit the readout-aligned lens on the unembed rows `U`. No per-arch numpy forward to get wrong.

Requires torch + transformers (unlike the rest of lo3a). Run with lm-sae's venv:
  /home/allans/code/lm-sae/.venv/bin/python lo3a/tau_star_xarch.py --models gpt2 EleutherAI/pythia-160m ...
Writes lo3a/tau_star_xarch.json (per-model records) and prints the cross-model table.
"""
import argparse, gc, json, os, re, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# fixed cross-model corpus: diverse English prose / code / dialogue / technical (real_recall.PASSAGES).
from real_recall import PASSAGES  # noqa: E402  (numpy-only import)

# English closed-class (function) words — same stoplist as grammar_recall.py.
FUNCTION = set("""the a an this that these those my your his her its our their some any no every each all both few many
much most more less i you he she it we they me him them us who whom whose which what of in on at to for with by from
as into onto upon about over under above below between among through during before after since until against within
without toward towards across behind beyond near and or but nor so yet if then else because although though while
whereas unless whether is are was were be been being am do does did have has had will would shall should can could may
might must ought not very too also just only even still again here there when where why how than such out up down off
again once""".split())


def fine_class(s):
    b = s.strip()
    if b == "": return "space"
    if not b[:1].isalnum() and all(not ch.isalnum() for ch in b): return "punct"
    if b.isdigit(): return "digit"
    return "function" if b.lower() in FUNCTION else "content"


def spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra @ rb) / (np.linalg.norm(ra) * np.linalg.norm(rb) + 1e-30))


def _norm(v): return v / (np.linalg.norm(v) + 1e-30)


GRID = [1, 2, 4, 8, 16, 24, 32, 48, 64, 92, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072]


def recoverable_rank(U, H_res, a_star, d, fit_frac=0.5):
    """min r where a rank-r readout-aligned lens's top-1 == the model's argmax a*, per test decision.
    Lens basis: SVD of normalized top-competitor diffs (U[a]-U[v]) on a train split. U: [V,d], H_res: [n,d]."""
    n = len(a_star); ntr = int(n * fit_frac)
    rows = []
    for i in range(ntr):
        a = a_star[i]; sc = U @ H_res[i]
        comp = np.argsort(sc)[::-1]; comp = comp[comp != a][:8]
        for v in comp: rows.append(_norm(U[a] - U[v]))
    Vt = np.linalg.svd(np.asarray(rows), full_matrices=False)[2]              # [k, d] right-singular dirs
    A = Vt @ U.T                                                              # [k, V]
    te = slice(ntr, n); Xte = H_res[ntr:]; at = np.asarray(a_star[ntr:])
    P = Vt @ Xte.T                                                            # [k, nte]
    grid = [r for r in GRID if r <= d] + [d]
    rr = np.full(len(at), d, float); done = np.zeros(len(at), bool)
    for r in grid:
        arg = np.argmax((P[:r].T) @ A[:r], axis=1)
        hit = (arg == at) & ~done; rr[hit] = r; done |= hit
    return rr, te


def synth_geometry_law(U, d, rng, pool_k=4096, N=1200):
    """worst_case2.py on THIS model's readout geometry: across Dirichlet/mixture/zipf skew families, does
    median recoverable rank track min(exp(H),d)? Pure geometry (x=U[a]); tests the law on the readout matrix."""
    def Hf(p):
        p = np.asarray(p); p = p[p > 0]; return float(-(p * np.log2(p)).sum())

    def zipf(K, s):
        p = np.arange(1, K + 1, dtype=float) ** (-s); return p / p.sum()

    def mixture(K, hm, hs):
        p = np.zeros(K); p[:hs] = zipf(hs, 1.0) * hm; p[hs:] = (1 - hm) / (K - hs); return p

    def dirichlet(K, al, sup, r):
        a = np.full(K, 1e-6); a[:sup] = al; return r.dirichlet(a)

    V = U.shape[0]; pool = rng.choice(V, size=min(pool_k, V), replace=False); G = U[pool]; K = len(G)
    grid = [r for r in GRID if r <= d] + [d]

    def measure(p):
        idx = rng.choice(K, size=N, p=p / p.sum()); tr = idx[:N // 2]; te = idx[N // 2:]
        rows = []
        for a in tr:
            sc = G @ G[a]; comp = np.argsort(sc)[::-1]; comp = comp[comp != a][:8]
            for v in comp: rows.append(_norm(G[a] - G[v]))
        Vt = np.linalg.svd(np.asarray(rows), full_matrices=False)[2]; A = Vt @ G.T
        Pp = Vt @ G[te].T; rr = np.full(len(te), d, float); done = np.zeros(len(te), bool)
        for r in grid:
            arg = np.argmax((Pp[:r].T) @ A[:r], axis=1); hit = (arg == te) & ~done; rr[hit] = r; done |= hit
        return float(np.median(rr))

    expH, mr = [], []
    for sup in (512, 1024, 2048, 4096):
        for al in (0.02, 0.2, 2.0):
            p = dirichlet(K, al, min(sup, K), np.random.default_rng(10)); expH.append(min(2 ** Hf(p), d)); mr.append(measure(p))
    for hm in (0.5, 0.8, 0.95):
        for hs in (64, 256):
            p = mixture(K, hm, hs); expH.append(min(2 ** Hf(p), d)); mr.append(measure(p))
    for s in (0.0, 0.5, 1.0, 1.5, 2.0):
        p = zipf(K, s); expH.append(min(2 ** Hf(p), d)); mr.append(measure(p))
    return spearman(mr, expH), len(mr)


def run_model(mid, dtype, max_ctx, device):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    td = {"f32": torch.float32, "bf16": torch.bfloat16, "f16": torch.float16}[dtype]
    tok = AutoTokenizer.from_pretrained(mid)
    model = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=td, low_cpu_mem_usage=True).to(device).eval()
    # The hook is applied IDENTICALLY for every architecture/tokenizer — there are no per-arch adjustments,
    # by construction. Every HF causal LM ends in `lm_head(h)`, so we hook the unembed's INPUT `h` (the
    # already-normed readout residual) and take `U = lm_head.weight`. Edge cases all fold in for free:
    #   * tied embeddings  → get_output_embeddings() still returns the lm_head Linear (weight = the embedding),
    #     so U and the hook are correct without special handling;
    #   * embedding-scale / (1+w) norms (Gemma) → already applied inside the forward, so they live in `h`;
    #   * final logit softcap (Gemma) → applied AFTER lm_head and monotone in each logit, so argmax(out.logits)
    #     == argmax(h·Uᵀ) == a*; we target a* (from out.logits) and fit the lens on U rows, all consistent;
    #   * large vocab (Gemma 262k, Qwen 152k) → U is upcast to float64 for the rank math; bf16 weights upcast
    #     on capture. So vocab size and tying never need a code path — only RAM (the >1B models run in bf16).
    head = model.get_output_embeddings()
    U = head.weight.detach().float().cpu().numpy()                           # [V, d]
    V, d = U.shape
    grabbed = {}
    h_hook = head.register_forward_pre_hook(lambda m, inp: grabbed.__setitem__("h", inp[0].detach()))

    H_res, a_star, toks, margins, logsum, ndec = [], [], [], [], None, 0
    cnt = {}
    Hsum = 0.0  # sum of per-decision output entropy (bits)
    with torch.no_grad():
        for txt in PASSAGES:
            ids = tok(txt, return_tensors="pt").input_ids[:, :max_ctx].to(device)
            if ids.shape[1] < 4: continue
            out = model(ids)
            lg = out.logits[0].float()                                       # [seq, V]
            h = grabbed["h"][0].float()                                      # [seq, d] readout input
            for t in ids[0].tolist(): cnt[t] = cnt.get(t, 0) + 1
            row_logits = lg.cpu().numpy()
            for i in range(2, ids.shape[1]):
                r = row_logits[i]; o = np.argpartition(-r, 1)[:2]; o = o[np.argsort(-r[o])]
                a = int(np.argmax(r)); a_star.append(a)
                H_res.append(h[i].cpu().numpy()); toks.append(int(a))
                margins.append(float(r[a] - r[o[0] if o[0] != a else o[1]]))
                p = r - r.max(); p = np.exp(p); p /= p.sum(); pe = p[p > 0]
                Hsum += float(-(pe * np.log2(pe)).sum())
                logsum = r if logsum is None else logsum + r
                ndec += 1
    h_hook.remove()
    meanlogit = logsum / max(1, ndec)
    H_res = np.asarray(H_res, np.float64); a_star = np.asarray(a_star)
    expH_out = 2 ** (Hsum / max(1, ndec))                                     # exp(mean per-token output entropy)

    # corpus unigram self-info over the tokenized corpus
    total = sum(cnt.values()); freq = np.zeros(V)
    for t, c in cnt.items(): freq[t] = c / total

    rr, te = recoverable_rank(U.astype(np.float64), H_res, a_star, d)
    at = a_star[te]
    cls = np.array([fine_class(tok.decode([int(t)])) for t in at])
    info_c = -np.log2(np.clip(freq[at], 1e-9, None))
    info_m = -(meanlogit[at] - meanlogit.mean())
    sp_c = spearman(rr, info_c); sp_m = spearman(rr, info_m)

    # (B) open/closed R@k at a fixed lens rank (SmolLM-equivalent fraction r≈d/6, like 92/576)
    rfix = max(8, int(round(d / 6.0)))
    closed = np.isin(cls, ["space", "punct", "digit", "function"]); openc = cls == "content"
    med_open = float(np.median(rr[openc])) if openc.any() else float("nan")
    med_closed = float(np.median(rr[closed])) if closed.any() else float("nan")
    rk_open = float(np.mean(rr[openc] <= rfix)) if openc.any() else float("nan")
    rk_closed = float(np.mean(rr[closed] <= rfix)) if closed.any() else float("nan")

    sp_synth, nsynth = synth_geometry_law(U.astype(np.float64), d, np.random.default_rng(0))

    del model, U, H_res
    gc.collect()
    try:
        import torch as _t; _t.cuda.empty_cache()
    except Exception:
        pass

    return {
        "model": mid, "dtype": dtype, "d": int(d), "vocab": int(V), "n_decisions": int(ndec),
        "n_test": int(len(rr)),
        "median_rank": float(np.median(rr)), "median_rho_over_d": float(np.median(rr) / d),
        "exp_H_output": float(expH_out), "min_expH_d": float(min(expH_out, d)),
        "spearman_rank_selfinfo_corpus": sp_c, "spearman_rank_baselinelogit": sp_m,
        "rfix": int(rfix),
        "open_median_rank": med_open, "closed_median_rank": med_closed,
        "open_Rk_le_rfix": rk_open, "closed_Rk_le_rfix": rk_closed,
        "synth_geometry_spearman_min_expH_d": sp_synth, "synth_n_distributions": nsynth,
        "n_open": int(openc.sum()), "n_closed": int(closed.sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=[
        "gpt2", "EleutherAI/pythia-70m", "EleutherAI/pythia-160m", "EleutherAI/pythia-410m",
        "Qwen/Qwen2.5-0.5B", "unsloth/gemma-3-1b-pt"])
    ap.add_argument("--dtype", default="f32", choices=["f32", "bf16", "f16"])
    ap.add_argument("--big-dtype", default="bf16", help="dtype for models with >1B params")
    ap.add_argument("--max-ctx", type=int, default=256)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=os.path.join(HERE, "tau_star_xarch.json"))
    args = ap.parse_args()

    big = {"EleutherAI/pythia-1b", "EleutherAI/pythia-1.4b", "EleutherAI/pythia-2.8b",
           "google/gemma-2-2b", "Qwen/Qwen2.5-1.5B", "gpt2-large", "gpt2-xl", "meta-llama/Llama-3.2-1B"}
    recs = []
    for mid in args.models:
        dt = args.big_dtype if mid in big else args.dtype
        print(f"\n=== {mid}  (dtype={dt}) ===", flush=True)
        try:
            rec = run_model(mid, dt, args.max_ctx, args.device)
        except Exception as e:
            print(f"   SKIP {mid}: {type(e).__name__}: {e}", flush=True)
            recs.append({"model": mid, "error": f"{type(e).__name__}: {e}"}); continue
        recs.append(rec)
        print(f"   d={rec['d']} V={rec['vocab']} n={rec['n_decisions']} | "
              f"med ρ/d={rec['median_rho_over_d']:.2f} med-rank={rec['median_rank']:.0f} "
              f"exp(H_out)={rec['exp_H_output']:.0f} min(expH,d)={rec['min_expH_d']:.0f}", flush=True)
        print(f"   Spearman(rank, self-info corpus)={rec['spearman_rank_selfinfo_corpus']:+.2f} "
              f"(baseline-logit {rec['spearman_rank_baselinelogit']:+.2f}) | "
              f"synth-geometry Spearman(rank,min(expH,d))={rec['synth_geometry_spearman_min_expH_d']:+.2f}", flush=True)
        print(f"   OPEN-class med-rank={rec['open_median_rank']:.0f} R@rfix={100*rec['open_Rk_le_rfix']:.0f}%  "
              f"CLOSED med-rank={rec['closed_median_rank']:.0f} R@rfix={100*rec['closed_Rk_le_rfix']:.0f}% "
              f"(rfix={rec['rfix']}, n_open={rec['n_open']}/n_closed={rec['n_closed']})", flush=True)
        json.dump(recs, open(args.out, "w"), indent=2)

    # cross-model table
    ok = [r for r in recs if "error" not in r]
    print("\n\n== CROSS-MODEL τ* TABLE ==")
    hdr = f"{'model':<28}{'d':>5}{'med ρ/d':>9}{'med-rank':>9}{'exp(H)':>8}{'sp(self-info)':>14}{'sp(synth)':>11}{'open R@rfix':>12}{'closed R@rfix':>14}"
    print(hdr); print("-" * len(hdr))
    for r in ok:
        print(f"{r['model']:<28}{r['d']:>5}{r['median_rho_over_d']:>9.2f}{r['median_rank']:>9.0f}"
              f"{r['exp_H_output']:>8.0f}{r['spearman_rank_selfinfo_corpus']:>+14.2f}"
              f"{r['synth_geometry_spearman_min_expH_d']:>+11.2f}"
              f"{100*r['open_Rk_le_rfix']:>11.0f}%{100*r['closed_Rk_le_rfix']:>13.0f}%")
    json.dump(recs, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
