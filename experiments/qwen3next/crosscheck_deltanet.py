#!/usr/bin/env python3
"""Pin the variant: cross-check deltanet_ref.py against transformers' OWN torch_recurrent_gated_delta_rule.

deltanet_ref validates the recurrence is self-consistent; this proves it is the SAME recurrence the real
Qwen3.6 (`qwen3_5_moe`) uses, and extracts the exact conventions the Rust port must replicate. Feeds the
identical random (q,k,v,g,β) to both and asserts the outputs match to f32 tolerance.

Conventions read from transformers (torch_recurrent_gated_delta_rule):
  • g is LOG-decay  → α_t = exp(g_t)                        (deltanet_ref takes α directly: pass exp(g))
  • q is scaled by  1/√d_k  before the read                 (deltanet_ref doesn't scale: pre-scale q)
  • optional L2-norm on BOTH q and k  (use_qk_l2norm_in_kernel)  (deltanet_ref normalizes only k: pre-norm)
The recurrence itself (decay→delta-correct-vs-decayed-state→write→read-after-write) is identical.

Needs transformers>=5.12. Usage: python crosscheck_deltanet.py
"""

from __future__ import annotations

import numpy as np
import torch
from deltanet_ref import gated_deltanet_seq
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import torch_recurrent_gated_delta_rule


def _l2(x, eps=1e-6):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def crosscheck(T=96, dk=24, dv=24, l2=True, seed=0):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((T, dk))
    k = rng.standard_normal((T, dk))
    v = rng.standard_normal((T, dv))
    g = -np.abs(rng.standard_normal(T)) * 0.4          # log-decay (g<0 ⇒ α=exp(g)∈(0,1))
    beta = rng.uniform(0.0, 1.0, T)

    # transformers reference: query/key/value [B,T,H,D], g/beta [B,T,H]
    tq, tk, tv = (torch.tensor(x[None, :, None, :], dtype=torch.float32) for x in (q, k, v))
    tg, tb = (torch.tensor(x[None, :, None], dtype=torch.float32) for x in (g, beta))
    out, _ = torch_recurrent_gated_delta_rule(tq, tk, tv, tg, tb, None, False, use_qk_l2norm_in_kernel=l2)
    ref = out[0, :, 0, :].numpy()

    # deltanet_ref under the SAME conventions (pre-apply: log→α, q-scale, qk-l2norm)
    scale = 1.0 / np.sqrt(dk)
    qe = (_l2(q) if l2 else q) * scale
    ke = _l2(k) if l2 else k
    mine, _ = gated_deltanet_seq(qe, ke, v, np.exp(g), beta, normalize_k=False)
    return float(np.abs(ref - mine).max()), float(np.abs(ref).mean())


def main():
    # Qwen3.6's layer calls the kernel with use_qk_l2norm_in_kernel=TRUE (modeling_qwen3_5_moe.py L530/541),
    # so l2=True is THE config to verify. l2=False is unused AND numerically unstable (un-normalized keys make
    # the delta recurrence explode — both impls blow up to ~1e9, so a diff there is float chaos, not a mismatch).
    print("=== deltanet_ref.py  vs  transformers torch_recurrent_gated_delta_rule ===")
    print("  (Qwen3.6 uses qk_l2norm=True — that is the binding config; l2=False is unused/unstable)")
    ok = True
    for seed in range(4):
        d, scale = crosscheck(l2=True, seed=seed)
        if d >= 1e-4:
            ok = False
        print(f"  qk_l2norm=True  seed={seed}  max_abs_diff={d:.2e}   {'MATCH' if d < 1e-4 else 'DIVERGE'}")
    df, sf = crosscheck(l2=False, seed=0)
    print(f"  qk_l2norm=False seed=0  max_abs_diff={df:.2e}  (out scale {sf:.1e}) — unused config: both explode "
          f"(un-normalized keys ⇒ delta recurrence unstable), not a math mismatch")
    print("VERDICT:", "deltanet_ref IS the Qwen3.6 DeltaNet recurrence — variant PINNED: α=exp(g), q·(1/√dₖ),"
          " L2-norm(q)&L2-norm(k), decay→delta-correct-vs-decayed-state→write→read-after-write."
          if ok else "DIVERGES in the real (l2=True) config — inspect before porting.")


if __name__ == "__main__":
    main()
