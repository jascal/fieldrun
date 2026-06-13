#!/usr/bin/env python3
"""R2 — frozen vs trained: can a decode-targeted TRAINED head beat the τ* floor?

LO1 only ever uses a FROZEN SVD lens to measure recoverable rank. The #142 entangled-core result showed
that *retraining* a rank-8 update bottleneck is lossless ~30× below the frozen ⅓d floor — so "τ* = output
entropy" might be a FROZEN-LINEAR statement, not the function's floor. This is the one experiment the LO1
docs name as the sole re-opener and never run.

Test: at matched rank r, a rank-r decode head  logits = (h @ B) @ C   (B: d×r, C: r×V) initialized AT the
frozen readout-aligned SVD lens (so training can only help), trained with cross-entropy / top-1
distillation against the model's OWN argmax a* on a TRAIN split, then evaluated on a HELD-OUT split. We
compare frozen-SVD vs trained on R@1 / R@32 by token class (open vs closed), and ask whether training
closes the open-class content-word gap that frozen SVD cannot. Strictly descriptive verdict.

SmolLM-135M (via the lo3a rope bundle) first, then GPT-2 + a Pythia rung (via HF + the unembed hook).
Requires torch + transformers. Run with lm-sae's venv. Writes lo3a/tau_star_trained.json.
"""
import argparse, gc, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
from real_recall import PASSAGES, forward_all  # noqa: E402
from grammar_recall import fine_class  # noqa: E402
import bundle_io as bio  # noqa: E402


def _norm(v): return v / (np.linalg.norm(v) + 1e-30)


def capture_hf(mid, dtype, max_ctx, device):
    """Per-token (readout-input residual h, model argmax a*, decoded a* string) via a hook on the unembed."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    td = {"f32": torch.float32, "bf16": torch.bfloat16, "f16": torch.float16}[dtype]
    tok = AutoTokenizer.from_pretrained(mid)
    model = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=td, low_cpu_mem_usage=True).to(device).eval()
    head = model.get_output_embeddings()
    U = head.weight.detach().float().cpu().numpy()
    grabbed = {}
    hk = head.register_forward_pre_hook(lambda m, inp: grabbed.__setitem__("h", inp[0].detach()))
    H_res, a_star, strs = [], [], []
    with torch.no_grad():
        for txt in PASSAGES:
            ids = tok(txt, return_tensors="pt").input_ids[:, :max_ctx].to(device)
            if ids.shape[1] < 4: continue
            lg = model(ids).logits[0].float().cpu().numpy()
            h = grabbed["h"][0].float().cpu().numpy()
            for i in range(2, ids.shape[1]):
                a = int(np.argmax(lg[i])); a_star.append(a); H_res.append(h[i]); strs.append(tok.decode([a]))
    hk.remove(); del model
    gc.collect()
    return U.astype(np.float64), np.asarray(H_res, np.float64), np.asarray(a_star), strs


def capture_bundle(stem):
    """SmolLM via the lo3a rope bundle: raw residual x and argmax (argmax(x·gU)==full argmax). geometry gU."""
    man, W = bio.read_bundle(stem); cfg, cfg_f = man["config"], man["config_f"]
    from bpe import BPE
    bpe = BPE(os.path.join(os.path.dirname(stem), os.path.basename(stem) + ".tokenizer.json"))
    U = (W["embed"] if cfg[7] else W["lm_head"]).astype(np.float64); gain = W["norm"].astype(np.float64)
    gU = gain * U
    H_res, a_star, strs = [], [], []
    for txt in PASSAGES:
        ids = bpe.encode(txt)
        if len(ids) < 4: continue
        xall, lg = forward_all(W, cfg, cfg_f, ids)
        for i in range(2, len(ids)):
            a = int(np.argmax(lg[i])); a_star.append(a); H_res.append(xall[i]); strs.append(bpe.decode_token(a))
    return gU, np.asarray(H_res, np.float64), np.asarray(a_star), strs


def frozen_lens(U, Htr, atr, r):
    """readout-aligned SVD lens at rank r → (B: d×r, A: r×V). The LO1 frozen baseline."""
    rows = []
    for i in range(len(atr)):
        a = atr[i]; sc = U @ Htr[i]; comp = np.argsort(sc)[::-1]; comp = comp[comp != a][:8]
        for v in comp: rows.append(_norm(U[a] - U[v]))
    Vt = np.linalg.svd(np.asarray(rows), full_matrices=False)[2]
    return Vt[:r].T.copy(), (Vt[:r] @ U.T).copy(), Vt  # B (d×r), A (r×V), full Vt (for PR/span90)


def topk_hits(logits, a, k):
    if k == 1: return np.argmax(logits, axis=1) == a
    tk = np.argpartition(-logits, k - 1, axis=1)[:, :k]
    return np.array([a[j] in tk[j] for j in range(len(a))])


def train_head(U, Htr, atr, B0, A0, r, steps, lr, device, mode="free", wd=0.0):
    """train a rank-r decode head from the frozen init (B0,A0) with cross-entropy / top-1 distillation to a*.
       mode='free': train both B (d×r) and C (r×V) — capacity r·V ≫ the frozen lens (overfit-prone).
       mode='tied': train ONLY the projection B (d×r); readout tied to the true geometry C = Bᵀ·Uᵀ, so the
                    capacity equals the frozen lens's (its only free choice is the rank-r subspace). This is
                    the apples-to-apples 'is SVD the optimal rank-r projection for argmax recovery?' test."""
    import torch
    dev = device
    B = torch.tensor(B0, dtype=torch.float32, device=dev, requires_grad=True)          # d×r
    Xt = torch.tensor(Htr, dtype=torch.float32, device=dev); yt = torch.tensor(atr, dtype=torch.long, device=dev)
    n = len(atr); bs = min(n, 512); rng = np.random.default_rng(0)
    if mode == "tied":
        Ut = torch.tensor(U, dtype=torch.float32, device=dev)                          # V×d (fixed geometry)
        opt = torch.optim.Adam([B], lr=lr, weight_decay=wd)
        for step in range(steps):
            idx = rng.choice(n, size=bs, replace=False)
            logit = (Xt[idx] @ B) @ (Ut @ B).T                                         # bs×V, readout tied to U
            loss = torch.nn.functional.cross_entropy(logit, yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        Bd = B.detach().cpu().numpy()
        return Bd, (Bd.T @ U.T)                                                        # C = Bᵀ·Uᵀ  (r×V)
    C = torch.tensor(A0, dtype=torch.float32, device=dev, requires_grad=True)          # r×V (free)
    opt = torch.optim.Adam([B, C], lr=lr, weight_decay=wd)
    for step in range(steps):
        idx = rng.choice(n, size=bs, replace=False)
        loss = torch.nn.functional.cross_entropy((Xt[idx] @ B) @ C, yt[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    return B.detach().cpu().numpy(), C.detach().cpu().numpy()


def eval_head(U, Hte, ate, B, C, strs_te):
    """R@1/R@32 overall + by open/closed class for a rank-r head logits=(h@B)@C."""
    logits = (Hte @ B) @ C
    cls = np.array([fine_class(s) for s in strs_te])
    openc = cls == "content"; closed = ~openc
    out = {}
    for k in (1, 32):
        hit = topk_hits(logits, ate, k)
        out[f"R{k}"] = float(hit.mean())
        out[f"R{k}_open"] = float(hit[openc].mean()) if openc.any() else float("nan")
        out[f"R{k}_closed"] = float(hit[closed].mean()) if closed.any() else float("nan")
    out["n_open"] = int(openc.sum()); out["n_closed"] = int(closed.sum())
    return out


def run(mid, source, stem, dtype, max_ctx, device, steps, lr, fit_frac=0.5):
    if source == "bundle":
        U, H_res, a_star, strs = capture_bundle(stem)
    else:
        U, H_res, a_star, strs = capture_hf(mid, dtype, max_ctx, device)
    d = U.shape[1]; V = U.shape[0]; n = len(a_star)
    ntr = int(n * fit_frac)
    Htr, atr, str_tr = H_res[:ntr], a_star[:ntr], strs[:ntr]
    Hte, ate, str_te = H_res[ntr:], a_star[ntr:], strs[ntr:]

    # PR / span90 of the decision spectrum (from the competitor-diff geometry on the train split)
    _, _, Vt = frozen_lens(U, Htr, atr, min(d, 16))
    rows = []
    for i in range(len(atr)):
        a = atr[i]; sc = U @ Htr[i]; comp = np.argsort(sc)[::-1]; comp = comp[comp != a][:8]
        for v in comp: rows.append(_norm(U[a] - U[v]))
    sv = np.linalg.svd(np.asarray(rows), full_matrices=False)[1]
    energy = sv ** 2; PR = int(round((energy.sum() ** 2) / (energy ** 2).sum()))
    span90 = int(np.searchsorted(np.cumsum(energy) / energy.sum(), 0.90) + 1)
    ranks = sorted({max(2, PR), max(2, span90), min(d, 2 * PR)})

    print(f"\n=== {mid} (d={d}, V={V}, n={n}, PR={PR}, span90={span90}; ranks={ranks}) ===", flush=True)
    rec = {"model": mid, "source": source, "d": int(d), "vocab": int(V), "n": int(n),
           "PR": PR, "span90": span90, "steps": steps, "lr": lr, "ranks": {}}
    for r in ranks:
        B0, A0, _ = frozen_lens(U, Htr, atr, r)
        # frozen logits = (h @ Vt[:r].T) @ (Vt[:r]@U.T): B0=(d×r), A0=(r×V)
        fro = eval_head(U, Hte, ate, B0, A0, str_te)
        Bf, Cf = train_head(U, Htr, atr, B0, A0, r, steps, lr, device, mode="free")
        free = eval_head(U, Hte, ate, Bf, Cf, str_te)
        Bt, Ct = train_head(U, Htr, atr, B0, A0, r, steps, lr, device, mode="tied")
        tied = eval_head(U, Hte, ate, Bt, Ct, str_te)
        rec["ranks"][str(r)] = {"frozen": fro, "trained_free": free, "trained_tied": tied}
        print(f"  r={r:>4}  R@32  frozen {100*fro['R32']:>4.0f}%  free {100*free['R32']:>4.0f}%  tied {100*tied['R32']:>4.0f}%   "
              f"| OPEN R@32  frozen {100*fro['R32_open']:>4.0f}%  free {100*free['R32_open']:>4.0f}%  tied {100*tied['R32_open']:>4.0f}%  "
              f"(tied−frozen {100*(tied['R32_open']-fro['R32_open']):+.0f}pp)", flush=True)
    del U, H_res
    gc.collect()
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smollm-stem", default=os.path.join(HERE, "smollm", "smollm"))
    ap.add_argument("--hf-models", nargs="+", default=["gpt2", "EleutherAI/pythia-160m"])
    ap.add_argument("--dtype", default="f32")
    ap.add_argument("--max-ctx", type=int, default=256)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default=os.path.join(HERE, "tau_star_trained.json"))
    ap.add_argument("--no-smollm", action="store_true")
    args = ap.parse_args()

    recs = []
    if not args.no_smollm and os.path.exists(args.smollm_stem + ".fieldrun.json"):
        recs.append(run("SmolLM-135M", "bundle", args.smollm_stem, args.dtype, args.max_ctx,
                        args.device, args.steps, args.lr))
        json.dump(recs, open(args.out, "w"), indent=2)
    for mid in args.hf_models:
        try:
            recs.append(run(mid, "hf", None, args.dtype, args.max_ctx, args.device, args.steps, args.lr))
        except Exception as e:
            import traceback; traceback.print_exc()
            recs.append({"model": mid, "error": f"{type(e).__name__}: {e}"})
        json.dump(recs, open(args.out, "w"), indent=2)

    print("\n\n== R2 SUMMARY: OPEN-class R@32  frozen / trained-free / trained-tied  at matched rank ==")
    for r in recs:
        if "error" in r: continue
        for rk, v in r["ranks"].items():
            f, fr, ti = v["frozen"], v["trained_free"], v["trained_tied"]
            print(f"  {r['model']:<16} r={rk:<5} OPEN R@32  frozen {100*f['R32_open']:>3.0f}%  free {100*fr['R32_open']:>3.0f}%  "
                  f"tied {100*ti['R32_open']:>3.0f}%  (tied−frozen {100*(ti['R32_open']-f['R32_open']):+.0f}pp)")
    json.dump(recs, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
