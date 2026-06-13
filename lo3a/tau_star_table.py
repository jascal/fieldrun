#!/usr/bin/env python3
"""Regenerate the R1 cross-architecture τ* table (markdown) from tau_star_xarch.json.
Usage: python3 tau_star_table.py   (prints the markdown table for FINDINGS_PYTHIA.md / PROVABLE_OPT §7)."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))

TOKFAM = {"gpt2": "GPT-2 BPE", "EleutherAI/pythia-70m": "NeoX BPE", "EleutherAI/pythia-160m": "NeoX BPE",
          "EleutherAI/pythia-410m": "NeoX BPE", "EleutherAI/pythia-1b": "NeoX BPE",
          "EleutherAI/pythia-1.4b": "NeoX BPE", "Qwen/Qwen2.5-0.5B": "Qwen BPE",
          "unsloth/gemma-3-1b-pt": "Gemma SP", "google/gemma-2-2b": "Gemma SP"}
SHORT = {"gpt2": "GPT-2 (124M)", "EleutherAI/pythia-70m": "Pythia-70m", "EleutherAI/pythia-160m": "Pythia-160m",
         "EleutherAI/pythia-410m": "Pythia-410m", "EleutherAI/pythia-1b": "Pythia-1b",
         "EleutherAI/pythia-1.4b": "Pythia-1.4b", "Qwen/Qwen2.5-0.5B": "Qwen2.5-0.5B",
         "unsloth/gemma-3-1b-pt": "Gemma-3-1b", "google/gemma-2-2b": "Gemma-2-2b"}


def main():
    recs = json.load(open(os.path.join(HERE, "tau_star_xarch.json")))
    print("| model | tokenizer | d | vocab | med ρ/d | exp(H_out) | Spearman(rank, self-info) | Spearman(synth, min(exp(H),d)) | open R@rfix | closed R@rfix |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    # SmolLM reference row (original LO1, from the lo3a README headline numbers)
    print("| *SmolLM-135M (orig)* | Llama BPE | 576 | 49152 | 0.10 | — | +0.83 | +0.94 | ~17%¹ | ~94%¹ |")
    for r in recs:
        m = r["model"]
        print(f"| {SHORT.get(m, m)} | {TOKFAM.get(m,'?')} | {r['d']} | {r['vocab']} | {r['median_rho_over_d']:.2f} | "
              f"{r['exp_H_output']:.0f} | {r['spearman_rank_selfinfo_corpus']:+.2f} | "
              f"{r['synth_geometry_spearman_min_expH_d']:+.2f} | {100*r['open_Rk_le_rfix']:.0f}% | "
              f"{100*r['closed_Rk_le_rfix']:.0f}% |")
    print("\n¹ SmolLM open/closed from grammar_recall.py R@32 at r=92 (a recall metric); the per-model columns here")
    print("  are recoverable-rank ≤ rfix (rfix≈d/6), so SmolLM's % are not directly comparable — the *pattern*")
    print("  (open collapses, closed recovers) is what replicates. All other rows computed by tau_star_xarch.py.")


if __name__ == "__main__":
    main()
