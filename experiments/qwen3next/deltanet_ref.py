#!/usr/bin/env python3
"""Gated DeltaNet — the reference oracle for the fieldrun Rust port (runs on numpy alone).

Qwen3.6 / Qwen3-Next interleaves Gated DeltaNet *linear* attention with gated full attention; the DeltaNet
recurrence is the one genuinely-new kernel fieldrun lacks, and the one most likely to be implemented wrong.
This file is the executable spec: a clean SEQUENTIAL recurrence the Rust must match token-for-token, plus
property tests that pin the behaviour without needing torch/transformers or the big model.

Why sequential is enough: fieldrun decodes autoregressively (one token at a time), which IS the recurrence —
no chunked-parallel form needed for decode. Chunking is only a prefill speed optimization (add later, and
test it against THIS reference). So getting this recurrence right is the whole correctness story for decode.

Formulation (fla `gated_delta_rule` / Gated DeltaNet, decay-then-delta-correct), per head, keys L2-normalized:
    S_dec   = α_t · S_{t-1}                       # scalar decay gate α_t ∈ (0,1)
    v_pred  = S_decᵀ k_t                          # value currently stored for k_t (in the decayed state)
    v_new   = β_t · (v_t − v_pred)                # delta correction, write strength β_t ∈ (0,1)
    S_t     = S_dec + k_t ⊗ v_new                 # rank-1 write
    o_t     = S_tᵀ q_t                            # read AFTER the write (causal, current value visible)
State S ∈ ℝ^{d_k×d_v}.  NOTE: the exact gate placement is a *variant choice* — pin it to the HF
`transformers` Qwen3-Next impl in the whole-model parity test (make_tiny_qwen3next.py). This file validates
the recurrence + the delta-rule semantics; the parity test validates the variant.

CONFIRMED from transformers 5.12 (model_type `qwen3_5_moe_text`): the linear layer is a short causal
depthwise conv (`linear_conv_kernel_dim=4`) on q/k/v FIRST, then Gated DeltaNet — so the full Rust path is
conv1d→DeltaNet. This file is the DeltaNet-recurrence oracle; the conv is a separate, easy causal depthwise
conv to add in front (then test the pair against the transformers reference via make_tiny/compare).
"""

from __future__ import annotations

import numpy as np


def gated_deltanet_seq(q, k, v, alpha, beta, normalize_k=True, eps=1e-6):
    """Sequential Gated DeltaNet for one head. q,k:(T,dk)  v:(T,dv)  alpha,beta:(T,). Returns (out:(T,dv), S)."""
    T, dk = k.shape
    dv = v.shape[1]
    if normalize_k:
        k = k / (np.linalg.norm(k, axis=1, keepdims=True) + eps)
    S = np.zeros((dk, dv))
    out = np.zeros((T, dv))
    for t in range(T):
        S = alpha[t] * S
        v_pred = S.T @ k[t]                     # (dv,)
        v_new = beta[t] * (v[t] - v_pred)
        S = S + np.outer(k[t], v_new)           # rank-1 write
        out[t] = S.T @ q[t]                     # read after write
    return out, S


# --------------------------------------------------------------------------- property tests (the oracle)
def _orthonormal(n, d, seed):
    a = np.random.default_rng(seed).standard_normal((d, d))
    q, _ = np.linalg.qr(a)
    return q[:n]                                 # n orthonormal rows in R^d


def test_orthogonal_recall():
    """α=1, β=1, distinct orthonormal keys: the state becomes Σ kᵢ vᵢᵀ, so querying kⱼ returns vⱼ exactly
    (delta rule == exact associative memory for orthogonal keys)."""
    rng = np.random.default_rng(0)
    n, dk, dv = 6, 16, 8
    K = _orthonormal(n, dk, 1)
    Vw = rng.standard_normal((n, dv))
    # phase 1: write the n pairs (q during writes is irrelevant); phase 2: read with β=0 (no further write)
    q = np.vstack([K, K]); k = np.vstack([K, K])
    v = np.vstack([Vw, np.zeros_like(Vw)])
    alpha = np.ones(2 * n)
    beta = np.concatenate([np.ones(n), np.zeros(n)])      # write, then read-only
    out, _ = gated_deltanet_seq(q, k, v, alpha, beta)
    rec = out[n:]                                          # the read phase
    err = np.abs(rec - Vw).max()
    assert err < 1e-4, f"orthogonal recall err {err}"      # ~1e-6 from the key-norm eps; O(1) if recurrence is wrong
    return err


def test_delta_overwrite():
    """THE delta-rule discriminator (vs plain linear attention): writing (k,v1) then (k,v2) with the SAME key
    and β=1 must return v2 on read, NOT v1+v2 (linear attention would accumulate)."""
    dk, dv = 16, 8
    rng = np.random.default_rng(2)
    k0 = rng.standard_normal((1, dk))
    k0 = k0 / np.linalg.norm(k0)
    v1, v2 = rng.standard_normal((1, dv)), rng.standard_normal((1, dv))
    k = np.vstack([k0, k0, k0])
    v = np.vstack([v1, v2, np.zeros((1, dv))])
    q = np.vstack([k0, k0, k0])
    out, _ = gated_deltanet_seq(q, k, v, np.ones(3), np.array([1.0, 1.0, 0.0]))
    err2 = np.abs(out[2] - v2[0]).max()
    err_sum = np.abs(out[2] - (v1[0] + v2[0])).max()
    assert err2 < 1e-4, f"overwrite should give v2, err {err2}"
    assert err_sum > 0.1, "must NOT be v1+v2 (that would be linear attention, not delta)"
    return err2


def test_decay():
    """Scalar gate α<1: a single write then read-only steps must decay the read geometrically as α^t."""
    dk, dv = 8, 4
    rng = np.random.default_rng(3)
    k0 = rng.standard_normal((1, dk))
    k0 = k0 / np.linalg.norm(k0)
    v0 = rng.standard_normal((1, dv))
    T = 6
    k = np.vstack([k0] * T)
    q = np.vstack([k0] * T)
    v = np.vstack([v0, *([np.zeros((1, dv))] * (T - 1))])
    a = 0.7
    beta = np.array([1.0] + [0.0] * (T - 1))
    out, _ = gated_deltanet_seq(q, k, v, np.full(T, a), beta)
    ratios = [np.linalg.norm(out[t + 1]) / (np.linalg.norm(out[t]) + 1e-12) for t in range(1, T - 1)]
    err = max(abs(r - a) for r in ratios)
    assert err < 1e-4, f"decay ratio should be α={a}, got {ratios}"
    return err


def test_beta_zero_is_inert():
    """β=0 everywhere: no writes ⇒ state stays 0 ⇒ outputs are 0 regardless of q,k,v."""
    rng = np.random.default_rng(4)
    T, dk, dv = 10, 8, 5
    out, S = gated_deltanet_seq(rng.standard_normal((T, dk)), rng.standard_normal((T, dk)),
                                rng.standard_normal((T, dv)), np.ones(T), np.zeros(T))
    assert np.abs(out).max() < 1e-12 and np.abs(S).max() < 1e-12
    return float(np.abs(out).max())


if __name__ == "__main__":
    print("Gated DeltaNet reference — property tests (numpy only):")
    for fn in (test_orthogonal_recall, test_delta_overwrite, test_decay, test_beta_zero_is_inert):
        e = fn()
        print(f"  PASS  {fn.__name__:<26} (residual {e:.2e})")
    print("All DeltaNet reference properties hold — this is the oracle the Rust kernel must match.")
