#!/usr/bin/env python3
"""R2 — POWERED rerun: does 'trained rank-r subspace beats frozen SVD at the compressed point' replicate
across the Pythia ladder when the trained head has ENOUGH DATA + multiple seeds?

The published R2 (`tau_star_trained.py`) used ~1199 tokens (PASSAGES) and a single seed. Its trained head
overfits (collapses far below the frozen SVD lens at higher rank), and the compressed-point win was
sign-inconsistent across Pythia rungs (70m +8pp, 160m −8pp) — i.e. within noise. This rerun isolates
"R2 is wrong" from "R2 was under-powered":

- **Real corpus** (wikitext-2-raw) → ~`--n-tokens` per model (default 40k), so the trained head has
  many more tokens than parameters even at the compressed rank.
- **Multiple seeds** (`--seeds`): each seed reshuffles the train/test split AND the train minibatch RNG;
  report mean ± std of (tied − frozen) open-class R@32.
- **tied mode only** (the matched-capacity, apples-to-apples 'is SVD the optimal rank-r subspace?' arm) at
  the compressed rank (r≈PR) and a medium rank (2·PR).

Reuses R2's validated `frozen_lens` / `eval_head` / `topk_hits` / `fine_class`. Run with lm-sae's venv.
"""
import argparse
import gc
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from grammar_recall import fine_class  # noqa: E402
from tau_star_trained import eval_head, frozen_lens  # noqa: E402


def load_corpus(target_chars: int) -> list:
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    out, total = [], 0
    for row in ds:
        t = row["text"].strip()
        if len(t) < 200:  # skip headings / blanks; keep substantive paragraphs
            continue
        out.append(t)
        total += len(t)
        if total >= target_chars:
            break
    return out


def capture(mid, texts, dtype, max_ctx, device, n_tokens):
    """Per-token (readout-input residual h, model argmax a*, decoded a* string) over the corpus."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    td = {"f32": torch.float32, "bf16": torch.bfloat16, "f16": torch.float16}[dtype]
    tok = AutoTokenizer.from_pretrained(mid)
    model = AutoModelForCausalLM.from_pretrained(mid, dtype=td, low_cpu_mem_usage=True).to(device).eval()
    head = model.get_output_embeddings()
    U = head.weight.detach().float().cpu().numpy()
    grabbed = {}
    hk = head.register_forward_pre_hook(lambda m, inp: grabbed.__setitem__("h", inp[0].detach()))
    H, a_star, strs = [], [], []
    with torch.no_grad():
        for txt in texts:
            ids = tok(txt, return_tensors="pt").input_ids[:, :max_ctx].to(device)
            if ids.shape[1] < 4:
                continue
            lg = model(ids).logits[0].float().cpu().numpy()
            h = grabbed["h"][0].float().cpu().numpy()
            for i in range(2, ids.shape[1]):
                a_star.append(int(np.argmax(lg[i]))); H.append(h[i]); strs.append(tok.decode([int(np.argmax(lg[i]))]))
            if len(a_star) >= n_tokens:
                break
    hk.remove(); del model
    gc.collect()
    return U.astype(np.float64), np.asarray(H, np.float64), np.asarray(a_star), strs


def train_tied(U, Htr, atr, B0, r, steps, lr, seed, device):
    """Train ONLY the rank-r subspace B (readout tied to true U) from the frozen init B0. Seeded minibatch."""
    import torch
    dev = device
    B = torch.tensor(B0, dtype=torch.float32, device=dev, requires_grad=True)
    Xt = torch.tensor(Htr, dtype=torch.float32, device=dev)
    yt = torch.tensor(atr, dtype=torch.long, device=dev)
    Ut = torch.tensor(U, dtype=torch.float32, device=dev)
    n = len(atr); bs = min(n, 1024); rng = np.random.default_rng(seed)
    opt = torch.optim.Adam([B], lr=lr)
    for _ in range(steps):
        idx = rng.choice(n, size=bs, replace=False)
        logit = (Xt[idx] @ B) @ (Ut @ B).T
        loss = torch.nn.functional.cross_entropy(logit, yt[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    Bd = B.detach().cpu().numpy()
    return Bd, (Bd.T @ U.T)


def run_model(mid, texts, args):
    U, H, a, strs = capture(mid, texts, args.dtype, args.max_ctx, args.device, args.n_tokens)
    d, V, n = U.shape[1], U.shape[0], len(a)
    # PR of the decision spectrum (competitor-diff geometry), on a fixed slice for stability.
    rows = []
    sub = min(n, 4000)
    for i in range(sub):
        sc = U @ H[i]; comp = np.argsort(sc)[::-1]; comp = comp[comp != a[i]][:8]
        for v in comp:
            df = U[a[i]] - U[v]; rows.append(df / (np.linalg.norm(df) + 1e-30))
    sv = np.linalg.svd(np.asarray(rows), full_matrices=False)[1]
    energy = sv ** 2; PR = int(round((energy.sum() ** 2) / (energy ** 2).sum()))
    ranks = sorted({max(2, PR), min(d, 2 * PR)})
    print(f"\n=== {mid} (d={d}, V={V}, n={n}, PR={PR}; ranks={ranks}) ===", flush=True)
    rec = {"model": mid, "d": d, "vocab": V, "n": n, "PR": PR, "n_tokens": args.n_tokens,
           "steps": args.steps, "seeds": args.seeds, "ranks": {}}
    for r in ranks:
        deltas, fro_o, tie_o = [], [], []
        for seed in range(args.seeds):
            rng = np.random.default_rng(1000 + seed)
            perm = rng.permutation(n); ntr = int(n * 0.7)
            tr, te = perm[:ntr], perm[ntr:]
            B0, A0, _ = frozen_lens(U, H[tr], a[tr], r)
            fro = eval_head(U, H[te], a[te], B0, A0, [strs[i] for i in te])
            Bt, Ct = train_tied(U, H[tr], a[tr], B0, r, args.steps, args.lr, seed, args.device)
            tied = eval_head(U, H[te], a[te], Bt, Ct, [strs[i] for i in te])
            deltas.append(tied["R32_open"] - fro["R32_open"]); fro_o.append(fro["R32_open"]); tie_o.append(tied["R32_open"])
        dm, ds = float(np.mean(deltas)), float(np.std(deltas))
        verdict = "WINS" if dm > 0.005 else ("TIE" if dm >= -0.005 else "LOSES")
        print(f"  r={r:>4}  OPEN R@32  frozen {100*np.mean(fro_o):>4.1f}%  tied {100*np.mean(tie_o):>4.1f}%   "
              f"tied−frozen {100*dm:+.1f} ± {100*ds:.1f} pp  ({args.seeds} seeds)  [{verdict}]", flush=True)
        rec["ranks"][str(r)] = {"rank": r, "frozen_open_R32": float(np.mean(fro_o)),
                                "tied_open_R32": float(np.mean(tie_o)),
                                "delta_open_R32_mean": dm, "delta_open_R32_std": ds, "verdict": verdict}
    del U, H
    gc.collect()
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["gpt2", "EleutherAI/pythia-70m", "EleutherAI/pythia-160m"])
    ap.add_argument("--n-tokens", type=int, default=40000)
    ap.add_argument("--max-ctx", type=int, default=512)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--dtype", default="f32")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="/tmp/r2_powered.json")
    args = ap.parse_args()

    texts = load_corpus(target_chars=args.n_tokens * 6)  # ~6 chars/token headroom
    print(f"corpus: {len(texts)} paragraphs (~{sum(len(t) for t in texts)} chars) for ~{args.n_tokens} tokens",
          flush=True)
    recs = []
    for mid in args.models:
        try:
            recs.append(run_model(mid, texts, args))
        except Exception as e:
            import traceback; traceback.print_exc()
            recs.append({"model": mid, "error": f"{type(e).__name__}: {e}"})
        json.dump(recs, open(args.out, "w"), indent=2)

    print("\n== R2 POWERED SUMMARY: OPEN-class R@32, tied−frozen at the compressed rank (r≈PR) ==")
    for r in recs:
        if "error" in r:
            print(f"  {r['model']:<26} ERROR {r['error']}"); continue
        pr = str(r["PR"]) if str(r["PR"]) in r["ranks"] else list(r["ranks"])[0]
        v = r["ranks"][pr]
        print(f"  {r['model']:<26} r={pr:<4} frozen {100*v['frozen_open_R32']:>4.1f}%  tied {100*v['tied_open_R32']:>4.1f}%  "
              f"Δ {100*v['delta_open_R32_mean']:+.1f} ± {100*v['delta_open_R32_std']:.1f} pp  [{v['verdict']}]")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
