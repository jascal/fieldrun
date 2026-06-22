#!/usr/bin/env python3
"""Real-model numbers for the LE-T4 rank-1 unembed-shortlist certificate (PR #94/#95).

`--logic-whole` can't run a real vocab×d program through Soufflé (the input-embed wall), but the certificate's
firing rate + soundness + fact-reduction don't need Soufflé — only the real unembed `U` and real residuals `x`. This
loads a real SmolLM fieldrun bundle, computes the same rank-1 certificate parameters the Rust emitter does (top-K by
‖U_v‖, dominant elided direction μ̂ via power iteration, a_max/a_min/ρ_max), and on REAL in-distribution contexts
(greedy-decoded from seeds — each step is a forward on a real prefix) checks, swept over K:
  • shortlist contains the true argmax (correctness),
  • the certificate fires  (S > max(a_max·g, a_min·g) + ‖x‖·ρ_max,  g=⟨x,μ̂⟩),  and
  • SOUNDNESS — every certified context has shortlist-argmax == full-vocab argmax,
plus the would-be unembed fact reduction vocab×d → K×d.

Run from repo root: python lo3a/real_cert_numbers.py [smollm|smollm360|smollm17]
"""
import sys, os
sys.path.insert(0, "lo3a")
import bundle_io as bio
import numpy as np

MODEL = sys.argv[1] if len(sys.argv) > 1 else "smollm"


def forward_x(W, cfg, cfg_f, ids):
    """bundle_io.forward, but also returns xf[-1] (the post-final-norm residual the unembed dots against)."""
    n_layer, H, NKV, HD, D, FFN, VOCAB, TIED = [int(c) for c in cfg]
    theta, eps = float(cfg_f[0]), float(cfg_f[1]); HALF, REP = HD // 2, H // NKV
    inv = (1.0 / (theta ** (2.0 * np.arange(HALF, dtype=np.float32) / HD))).astype(np.float32)
    bias = "l0.self_attn.q_proj.bias" in W
    ids = list(ids); seq = len(ids)
    ang = np.arange(seq, dtype=np.float32)[:, None] * inv[None, :]
    COS = np.cos(ang).astype(np.float32)[:, None, :]; SIN = np.sin(ang).astype(np.float32)[:, None, :]
    causal = np.triu(np.ones((seq, seq), dtype=bool), k=1)
    def rope(x, nh):
        xr = x.reshape(seq, nh, HD); x1, x2 = xr[..., :HALF], xr[..., HALF:]
        return np.concatenate([x1 * COS - x2 * SIN, x2 * COS + x1 * SIN], axis=-1).reshape(seq, nh * HD).astype(np.float32)
    x = W["embed"][ids].astype(np.float32)
    for l in range(n_layer):
        p = f"l{l}."
        a = bio._rmsnorm(x, W[p + "in_ln"], eps)
        q = (a @ W[p + "self_attn.q_proj"]).astype(np.float32); k = (a @ W[p + "self_attn.k_proj"]).astype(np.float32)
        v = (a @ W[p + "self_attn.v_proj"]).astype(np.float32)
        if bias:
            q += W[p + "self_attn.q_proj.bias"]; k += W[p + "self_attn.k_proj.bias"]; v += W[p + "self_attn.v_proj.bias"]
        q, k = rope(q, H), rope(k, NKV)
        ao = np.zeros((seq, H * HD), dtype=np.float32)
        for h in range(H):
            kv = h // REP
            qh, kh, vh = q[:, h * HD:(h + 1) * HD], k[:, kv * HD:(kv + 1) * HD], v[:, kv * HD:(kv + 1) * HD]
            sc = (qh @ kh.T).astype(np.float32) / np.float32(np.sqrt(HD)); sc[causal] = np.float32(-1e30)
            sc = np.exp(sc - sc.max(axis=1, keepdims=True)).astype(np.float32); sc /= sc.sum(axis=1, keepdims=True)
            ao[:, h * HD:(h + 1) * HD] = (sc @ vh).astype(np.float32)
        x = (x + ao @ W[p + "self_attn.o_proj"]).astype(np.float32)
        a2 = bio._rmsnorm(x, W[p + "post_ln"], eps)
        hid = (bio._silu(a2 @ W[p + "mlp.gate_proj"]) * (a2 @ W[p + "mlp.up_proj"])).astype(np.float32)
        x = (x + hid @ W[p + "mlp.down_proj"]).astype(np.float32)
    xf = bio._rmsnorm(x, W["norm"], eps)
    unemb = W["lm_head"] if TIED == 0 else W["embed"]
    return (xf[-1] @ unemb.T).astype(np.float32), xf[-1]


def cert_params(U, k):
    """Mirror the Rust emit_whole: top-K by ‖U_v‖, dominant elided direction μ̂, a_max/a_min, ρ_max (+0.1% slack)."""
    n2 = (U * U).sum(1)
    order = np.argsort(-n2)
    keep = set(order[:k].tolist()); elided = order[k:]
    UE = U[elided]
    mu = UE[0].astype(np.float32).copy(); mu /= (np.linalg.norm(mu) + 1e-9)
    for _ in range(16):
        mu = UE.T @ (UE @ mu); mu /= (np.linalg.norm(mu) + 1e-9)
    a = UE @ mu
    rho2 = np.clip((UE * UE).sum(1) - a * a, 0, None).max()
    return keep, order[:k], mu.astype(np.float32), float(a.max()), float(a.min()), float(np.sqrt(rho2 * 1.001))


def cert_params_for(U, keep_ids):
    """Rank-1 certificate params for an ARBITRARY shortlist (set of kept token ids); elided = the complement."""
    vocab = U.shape[0]
    keep_set = set(int(x) for x in keep_ids)
    elided = np.array([v for v in range(vocab) if v not in keep_set], dtype=np.int64)
    UE = U[elided]
    mu = UE[0].astype(np.float32).copy(); mu /= (np.linalg.norm(mu) + 1e-9)
    for _ in range(16):
        mu = UE.T @ (UE @ mu); mu /= (np.linalg.norm(mu) + 1e-9)
    a = UE @ mu
    rho2 = np.clip((UE * UE).sum(1) - a * a, 0, None).max()
    return keep_set, list(keep_ids), mu.astype(np.float32), float(a.max()), float(a.min()), float(np.sqrt(rho2 * 1.001))


def gen_ctxs(W, cfg, cfg_f, vocab, seeds):
    ctxs = []
    rng = np.random.default_rng(0)
    starts = [rng.integers(5, vocab - 1, size=2).tolist() for _ in range(64)]
    for s in seeds:
        ids = starts[s]
        for _ in range(24):
            lg, xf = forward_x(W, cfg, cfg_f, ids)
            ctxs.append((lg, xf, int(lg.argmax()))); ids.append(int(lg.argmax()))
            if len(ids) > 40: ids = ids[-40:]
    return ctxs


def evaluate(U, ctxs, k, keep_ids):
    keep_set, sl, mu, amax, amin, rho = cert_params_for(U, keep_ids)
    cont = ncert = nmis = 0
    for lg, xf, win in ctxs:
        if win in keep_set: cont += 1
        slw = sl[int(np.argmax(lg[sl]))]
        S = float(lg[slw]); g = float(xf @ mu); xn = float(np.linalg.norm(xf))
        if S > max(amax * g, amin * g) + xn * rho:
            ncert += 1
            if slw != win: nmis += 1
    n = len(ctxs)
    return 100 * cont // n, 100 * ncert // n, nmis


def main():
    stem = f"lo3a/{MODEL}/{MODEL}"
    man, W = bio.read_bundle(stem)
    cfg, cfg_f = man["config"], man["config_f"]
    vocab, d = int(cfg[6]), int(cfg[4])
    U = (W["lm_head"] if int(cfg[7]) == 0 else W["embed"]).astype(np.float32)
    train = gen_ctxs(W, cfg, cfg_f, vocab, range(0, 8))   # build the frequency shortlist from these
    test = gen_ctxs(W, cfg, cfg_f, vocab, range(8, 16))   # measure on held-out
    n2 = (U * U).sum(1)
    norm_order = list(np.argsort(-n2))
    import collections
    freq = collections.Counter(win for _, _, win in train)
    print(f"# real-model cert · {MODEL} · vocab={vocab} d={d} · train {len(train)} / test {len(test)} contexts (held-out)\n")
    print(f"#   {'shortlist':<12}{'K':>5}  {'unembed facts':>16}  {'contains':>9}{'certified':>10}{'sound':>7}")
    for k in [256, 1024, 4096]:
        if k >= vocab: continue
        red = f"{vocab*d:,}→{k*d:,}"
        # (a) top-K by unembed norm
        c, ce, m = evaluate(U, test, k, norm_order[:k])
        print(f"#   {'norm':<12}{k:>5}  {red:>16}  {c:>7}%{ce:>8}%{('OK' if m==0 else f'{m}BAD'):>7}")
        # (b) top-K by TRAIN winning-frequency, padded with norm-top to size K (a frequency/KB-style shortlist)
        fk = [t for t, _ in freq.most_common(k)]
        fk += [t for t in norm_order if t not in set(fk)][: k - len(fk)]
        c, ce, m = evaluate(U, test, k, fk[:k])
        print(f"#   {'frequency':<12}{k:>5}  {red:>16}  {c:>7}%{ce:>8}%{('OK' if m==0 else f'{m}BAD'):>7}")
    print("#   (contains = shortlist holds the true argmax; certified = rank-1 cert fires; sound = certified ⇒ correct)")
    rankpos = {t: i for i, t in enumerate(norm_order)}
    print(f"#   winners' norm-rank: median {int(np.median([rankpos[w] for _,_,w in test]))}/{vocab}  ⇒ norm-shortlist misses them")
    # the scale gap — WHY the certificate never fires on a real model (high-dim Cauchy–Schwarz looseness ~√d)
    k = 4096
    fk = [t for t, _ in freq.most_common(k)]; fk += [t for t in norm_order if t not in set(fk)][: k - len(fk)]
    _, sl, mu, amax, amin, rho = cert_params_for(U, fk[:k])
    Ss = [float(lg[int(lg.argmax())]) for lg, _, _ in test]
    bnd = [max(amax * float(xf @ mu), amin * float(xf @ mu)) + float(np.linalg.norm(xf)) * rho for _, xf, _ in test]
    print(f"#   scale gap (freq K={k}): winner logit S≈{np.mean(Ss):.0f} vs elided bound≈{np.mean(bnd):.0f}  ⇒ ~{np.mean(bnd)/max(np.mean(Ss),1e-9):.0f}× short")
    print(f"#   cause: in d={d}, ‖x‖·‖U_v‖ overestimates ⟨x,U_v⟩ by ~√d≈{int(np.sqrt(d))}× — no rank-r removal closes a √d gap.")
    print("#   ⇒ a HARD certified-compact unembed is infeasible on real models; compaction must be LOSSY (pr_core, R4) here.")


if __name__ == "__main__":
    main()
