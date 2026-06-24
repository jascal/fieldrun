#!/usr/bin/env python3
"""Step 2: angular / direction-preserving read-out quantization — can we beat the int4 wall?

Norm-pinning (pil §8a / the v1.5 finding) says the final norm fixes ‖r‖≈const, so the decode
`argmax_v ⟨r,U_v⟩` is essentially ANGULAR: only the *direction* of U_v competes with a fixed-direction r.
v1.5 found per-row int8 = 0 flips but group-int4 = 6% flips (the wall). Hypothesis: int4 fails on
MAGNITUDE error; a direction-preserving scheme at the same 4-bit budget should survive. Two cheap,
inner-product-exact tricks:

  • exact-L2-norm restore : U_v = ρ_v·û_v; quantize û_v, renormalize, multiply back by the EXACT ρ_v
    (ρ_v is one f16 scalar/row — negligible) → kills the norm-drift component of the error.
  • incoherence rotation  : ⟨r,U_v⟩ = ⟨Qr,QU_v⟩ for orthogonal Q; rotate then quantize. Q spreads each
    row uniformly so per-group scales fit tighter at low bits (QuIP/QuaRot idea). Inner products exact;
    at inference r→Qr is one d×d matvec (negligible vs the V×d unembed).

Ground truth: EXACT full-vocab decode flips vs the f16 read-out (argmax over all V), the same metric the
int4 wall was measured with. Usage: step2_angular.py calib_with_r.jsonl bundle_stem [--bits 4 3]"""
import json
import sys

import numpy as np


def q_int8_row(U):                                  # per-row symmetric int8 (fieldrun put_i8)
    s = np.abs(U).max(1, keepdims=True) / 127.0
    s[s == 0] = 1
    return np.round(U / s) * s


def q_intk_group(U, k=4, g=32):                     # group-wise symmetric int-k (k=4 -> fieldrun put_i4)
    lv = 2 ** (k - 1) - 1                            # int4->7, int3->3, int2->1
    out = np.empty_like(U)
    for j in range(0, U.shape[1], g):
        blk = U[:, j:j + g]
        s = np.abs(blk).max(1, keepdims=True) / lv
        s[s == 0] = 1
        out[:, j:j + g] = np.round(blk / s) * s
    return out


def restore_l2(Uq, U):                              # set ‖Uq_row‖ = exact ‖U_row‖ (direction from Uq)
    nq = np.linalg.norm(Uq, axis=1, keepdims=True)
    nq[nq == 0] = 1
    return Uq / nq * np.linalg.norm(U, axis=1, keepdims=True)


def rand_orth(d, seed=0):
    q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((d, d)).astype(np.float64))
    return q.astype(np.float32)


def gptq(W, H, k=4, g=32, damp=0.01):
    """GPTQ error-feedback quant (the gold standard): quantize columns left→right, propagating each
    column's rounding error to the remaining columns weighted by H⁻¹, where H = E[r rᵀ] is the residual
    covariance — so it directly minimizes the LOGIT error Σ_x⟨r_x,ΔU⟩², not weight MSE. Norm-pinning keeps
    r on a fixed sphere, so H is well-conditioned. No act-order; group-32 scales like the RTN baseline."""
    V, d = W.shape
    lv = 2 ** (k - 1) - 1
    H = H + damp * np.mean(np.diag(H)) * np.eye(d)
    hinv = np.linalg.inv(H).astype(np.float32)
    Q = W.astype(np.float32).copy()
    scale = np.ones((V, 1), np.float32)
    for i in range(d):
        if i % g == 0:
            blk = Q[:, i:i + g]
            scale = np.abs(blk).max(1, keepdims=True) / lv
            scale[scale == 0] = 1
        w = Q[:, i:i + 1]
        q = np.clip(np.round(w / scale), -lv, lv) * scale
        err = (w - q) / hinv[i, i]
        if i + 1 < d:
            Q[:, i + 1:] -= err * hinv[i, i + 1:][None, :]
        Q[:, i] = q[:, 0]
    return Q


def awq_scale(R, alpha):
    """Activation-aware per-column scale from the residual statistics (AWQ): protect the coordinates r
    actually uses. ⟨r,U⟩ = ⟨r/s, U·s⟩; quantize U·s so salient (high-r-variance) columns get finer steps.
    Norm-pinning makes this principled: the relevant 'Hessian' is the r-covariance on a fixed sphere."""
    imp = np.sqrt((R.astype(np.float64) ** 2).mean(0))                # (d,) per-column RMS of r
    imp = np.clip(imp, 1e-8, None)
    s = imp ** alpha
    return (s / np.exp(np.log(s).mean())).astype(np.float32)          # geomean-normalized so overall scale ~1


def main():
    calib, stem = sys.argv[1], sys.argv[2]
    bits = [int(b) for b in (sys.argv[sys.argv.index("--bits") + 1:] if "--bits" in sys.argv else [4, 3])]
    man = json.load(open(stem + ".fieldrun.json"))
    emb = [a for a in man["arrays"] if a["name"] == "embed"][0]
    assert emb["dtype"] == "f16"
    V, d = emb["shape"]
    off = emb["offset"]
    recs = [json.loads(line) for line in open(calib) if line.strip()]
    R = np.array([r["r"] for r in recs], dtype=np.float32)            # (N, d)  residuals (norm-pinned)
    N = len(recs)
    with open(stem + ".fieldrun.bin", "rb") as f:
        f.seek(off)
        A = np.frombuffer(f.read(V * d * 2), dtype=np.float16).astype(np.float32).reshape(V, d)
    ref = (A @ R.T).argmax(0)                                          # f16 decode per position (ground truth)
    Q = rand_orth(d, 0)
    Arot = A @ Q.T                                                     # rotate frame rows once (reused)
    Rrot = R @ Q.T

    def flips(Aq, Rd):
        return float(((Aq @ Rd.T).argmax(0) != ref).mean())

    gptq_ok = "full-rank H" if N >= d else f"RANK-DEFICIENT H (N={N}<d={d}: GPTQ ~ damped RTN, invalid)"
    print(f"=== angular read-out quant (Qwen0.5B-style, V={V} d={d} N={N}) — EXACT full-vocab flips vs f16 ===")
    print(f"  [GPTQ Hessian: {gptq_ok}]")
    print(f"  {'scheme':<26}{'flips':>8}   {'read-out MB':>11}")
    rb = lambda frac: V * d * frac / 1e6                               # noqa: E731  approx read-out bytes
    rows = [("int8 per-row (v1.5 ref)", q_int8_row(A), R, 1.0)]
    for k in bits:
        bf = (k / 16.0) + 2 / 32                                       # k-bit weights + f16 group scales (g=32)
        rows += [
            (f"int{k} group32 (naive RTN)", q_intk_group(A, k), R, bf),
            (f"int{k} +exactL2norm", restore_l2(q_intk_group(A, k), A), R, bf),
            (f"int{k} +rot", q_intk_group(Arot, k), Rrot, bf),
        ]
        for al in (0.5, 1.0):                                          # activation-aware (AWQ) — the RTN-wall breaker
            s = awq_scale(R, al)
            rows.append((f"int{k} +awq(a={al})", q_intk_group(A * s, k), R / s, bf))
        Hr = (R.astype(np.float64).T @ R.astype(np.float64)) / N        # logit-error Hessian E[r rᵀ]
        rows.append((f"int{k} +GPTQ (H=E[rrᵀ])", gptq(A, Hr, k), R, bf))
    for name, Aq, Rd, frac in rows:
        print(f"  {name:<26}{flips(Aq, Rd) * 100:>7.1f}%   {rb(frac):>9.0f} MB")
    print(f"  (f16 read-out = {rb(2.0):.0f} MB; int8 = {rb(1.0):.0f} MB; norms/Q overhead ~ negligible "
          f"[{V * 2 / 1e6:.1f} MB row-norms, {d * d * 2 / 1e6:.1f} MB Q or seeded])")


if __name__ == "__main__":
    main()
