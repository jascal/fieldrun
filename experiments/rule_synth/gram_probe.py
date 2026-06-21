#!/usr/bin/env python3
"""Digit-output frame Gram diagnostic (PIC_LOSSINESS §6, track B; tests proved Thm 2).

Loads the unembedding rows U_v for the output tokens (`fieldrun --dump-unembed`) and characterises the Gram kernel
G_{vw} = ⟨U_v, U_w⟩ that PIC_PROPOSAL names as the explicit carrier of non-truth-functionality. Reports:

  • the cosine-coherence ρ_{vw} = cos(U_v, U_w) matrix and its off-diagonal distribution — ρ≈0 everywhere is the
    diagonal-G / classical-incidence limit (the kernel does NO work; a per-token lookup loses nothing); structured ρ
    means the kernel couples outcomes and a compact pic can exploit it (or a token-EDB misses it);
  • a kernel-checked sanity tie to **Thm 2 (NonTruthFunctionalityBudget, proved in i-orca examples/fieldrun)**:
    ‖U_v − U_w‖² = 2(1 − ρ_{vw}); we verify the identity numerically on the real rows;
  • the effective rank (participation ratio of the eigenvalues) of G — how many dimensions the output frame really
    spans; low ⟺ the outputs live in a tight subspace (strong coupling), full ⟺ near-orthogonal outputs.

Usage:  python gram_probe.py /tmp/unembed_15b.jsonl
"""
import json, sys
import numpy as np


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/unembed.jsonl"
    rows = [json.loads(l) for l in open(path) if l.strip()]
    toks = [r["tok"] for r in rows]
    U = np.array([r["row"] for r in rows], dtype=np.float64)  # (V, d)
    V, d = U.shape
    G = U @ U.T                                                # Gram
    nrm = np.sqrt(np.diag(G))
    rho = G / np.outer(nrm, nrm)                               # cosine coherence
    off = rho[~np.eye(V, dtype=bool)]

    print(f"# Gram diagnostic · {path} · {V} output tokens ({' '.join(toks)}) · d={d}\n")
    print("# cosine-coherence ρ_vw matrix (×100):")
    print("      " + " ".join(f"{t:>4}" for t in toks))
    for i, t in enumerate(toks):
        print(f"  {t:>3} " + " ".join(f"{rho[i,j]*100:>4.0f}" for j in range(V)))

    print(f"\n# off-diagonal coherence ρ: mean {off.mean():+.3f}  sd {off.std():.3f}  "
          f"min {off.min():+.3f}  max {off.max():+.3f}  |ρ|>0.3: {int((np.abs(off)>0.3).sum())}/{len(off)}")

    # Thm 2 (proved): ‖U_v − U_w‖² = 2(1 − ρ) on UNIT-normalised rows — verify numerically
    Un = U / nrm[:, None]
    err = 0.0
    for i in range(V):
        for j in range(V):
            lhs = np.sum((Un[i] - Un[j]) ** 2)
            rhs = 2 * (1 - (Un[i] @ Un[j]))
            err = max(err, abs(lhs - rhs))
    print(f"# Thm 2 check  ‖U_v−U_w‖²=2(1−ρ) on unit rows: max abs error {err:.2e}  "
          f"({'CONFIRMED' if err < 1e-9 else 'numerical drift'})")

    # effective rank (participation ratio of eigenvalues) of G
    ev = np.linalg.eigvalsh(G)
    ev = ev[ev > 0]
    pr = (ev.sum() ** 2) / (ev ** 2).sum()
    # cosine-Gram (correlation) effective rank too — scale-free
    evc = np.linalg.eigvalsh(rho)
    evc = evc[evc > 1e-12]
    prc = (evc.sum() ** 2) / (evc ** 2).sum()
    print(f"# effective rank (eigenvalue PR): Gram {pr:.2f} / {V}   cosine-Gram {prc:.2f} / {V}   "
          f"(low ⟺ outputs in a tight coupled subspace; ≈{V} ⟺ near-orthogonal)")
    reading = ("near-diagonal frame — the kernel does little; a per-token lookup loses ~nothing here"
               if np.abs(off).mean() < 0.1 and prc > 0.8 * V
               else "coupled frame — G carries real off-diagonal structure the kernel can exploit / a token-EDB misses")
    print(f"# reading: {reading}")


if __name__ == "__main__":
    main()
