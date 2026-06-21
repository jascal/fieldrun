# Is the PIC residue code lossy? — corrected against the paper + i-orca

*Supersedes an earlier draft of this note that called `pic` "lossy" without qualification. That was wrong. This
version is grounded in the fieldrun paper (`paper/fieldrun_paper_draft.pdf`) and the kernel-checked i-orca
PROVABLE_OPT arm. Short version: **full PIC is exact (lossless) w.r.t. the model**; the only thing that is lossy is
*compressing the irreducible dense-Gram region into a flat lookup*, which the paper itself says is impossible — and
the relevant obstruction is **tropical rank**, not linear/SVD rank.*

## 0. The corrected claim

PIC / the semiring-Datalog program `Π` is **one object at two temperatures** (paper §1):

- **T=1, log-semiring** → sum-product → **returns the model's softmax distribution exactly**.
- **T=0, tropical semiring** → max-product → **returns the model's greedy decode exactly**.

So `Π` reproduces the model's output (distribution *and* decode) by construction — the logit is the exact additive
incidence sum `L_v=⟨r,U_v⟩=Σ_j c_j^v`, and the softmax/argmax of it is the model's own. **PIC is lossless w.r.t. the
model.** The earlier "lossy" label conflated this exact object with a *budget-truncated approximation* of it.

This is **kernel-proved, not merely constructed**: i-orca `examples/fieldrun` (Isabelle2025-2, zero `sorry`, all 10
paper theorems) — **Thm 4 `RecoveredProbability`**: `m(v)/Σ m = exp(L_v)/Z = softmax`, parameter-free. The exact
recovery of the model's output measure is a theorem.

## 1. What is actually hard to compress (and it is not PIC)

The paper splits behaviour into a **retrievable fragment** (exports to *compact, verifiable* Datalog — the head) and
an **irreducible computation** = "the **dense-Gram region that no lookup table reproduces**." Three conjecturally-equal
characterisations of that region (paper §1): the **high participation-ratio** residual, the **high tropical rank**
region, and the **high-treewidth dense-G** region.

So the incompressibility is a property of *flattening the irreducible computation into a lookup* (`edb`), not of PIC.
PIC represents that region **exactly** via the full Gram `G_{vw}=⟨U_v,U_w⟩`; a flat table cannot, by a rank floor.

## 2. The obstruction is tropical rank, not linear rank (corrects the earlier §3)

The earlier draft argued losiness via the *linear* singular spectrum of the incidence matrix (PR / Eckart–Young).
**That is the wrong measure.** The paper (T=0 geometry interpretation) is explicit:

> the irreducible computation is a **tropical-rank floor: the gap between the model's tropical rank and that of any flat
> lookup table, a gap that linear (SVD) rank structurally cannot measure.**

The decision surface is the tropical variety of a max-logit polynomial whose monomials are the unembedding rows;
composition = decision cells whose winning monomial appears only in the *sum* of sources, never in any single one.
That is a **max-plus (T=0) rank** phenomenon. Linear PR is a conservative proxy that can read "diffuse / must be lossy"
on a region a single tropical monomial (or a low tropical-rank set) captures exactly. This is the precise form of the
"sign-rank, not linear rank" intuition: the right complexity is the tropical rank of the decision surface.

## 3. Even a *compact* code is decode-lossless above the margin (PO-T3, kernel-checked)

i-orca `examples/provable_opt`, `decode_margin_certified` (PO-T3), kernel-checked, 2δ proved tight:

```
assumes  |L' v − L v| ≤ δ   for all v        (any δ-bounded approximation of the logits)
   and   L t − L v > 2·δ     for all v ≠ t    (the winner's margin exceeds 2δ)
shows    decodes_to L' V t                    (the approximation decodes to the SAME token)
```

So any compact / perturbed representation (a rank-truncated PIC, a dropped FFN neuron, quantisation) **reproduces the
model's next token on every token whose margin exceeds 2δ**. The loss can therefore only live in the **sub-2δ-margin**
tokens — which are exactly the low-margin, high-PR, dense-Gram residue (the forge tax). Margin and PR jointly localise
where a compact code can flip, and PO-T3 certifies everywhere else.

## 4. The corrected picture for `--residue-strategy`

| strategy | fidelity vs the model |
|---|---|
| `edb` | exact, but pays full size; **cannot compress** the irreducible region (only memorise it) |
| `ring` (T=0 Datalog) | the retrievable head exports **compact + lossless** (`PO-T1/T4` `T_P`-equivalence, proved); irreducible region not compressible |
| `pic` (full Gram, T=1) | **lossless** — returns the model's softmax exactly; *not* a lossy strategy |
| `pic` (rank-truncated) | value-lossy only on the dense-Gram residue; **decode-lossless above 2δ margin** (`PO-T3`); the floor is **tropical** rank |

So the honest framing is **lossless-but-large vs compact-but-decode-certified**, not "crisp = exact, pic = lossy." PIC
at full Gram is an exact lossless account; truncating it for compactness loses only on the provably-incompressible
(tropical-rank-floor) dense-Gram residue, and PO-T3 certifies the decode survives above 2δ.

## 5. What is proved (i-orca, kernel-checked) vs genuinely open

Most of what an earlier draft listed as "open conjecture" is in fact **kernel-proved** in i-orca `examples/fieldrun`
(Isabelle2025-2, zero `sorry`, all 10 paper theorems) and `examples/fieldrun/separation`:

| result | i-orca theorem | meaning for losiness |
|---|---|---|
| exact output recovery | `RecoveredProbability` (Thm 4) | full PIC = the model's softmax exactly → **lossless w.r.t. model** |
| diffuseness / PR floor | `Diffuseness` (Thm 5) | a k-source body captures only **\|A\|/PR** of E — *no bounded formula localises* high-PR; the fraction is exact ⇒ **tight** |
| asymptotic | `DiffusenessAsymptotic` | `k/PR → 0` — fixed-budget capture vanishes as PR grows |
| two temperatures | `TwoTemperatureSoundness` (Thm 6) | the Maslov sandwich — `ring`(T=0) and `pic`(T=1) are one program, two semirings |
| irreducibility | `irreducible_iff_unique_decider` (separation) | irreducible ⟺ the full coalition is the *unique* decider (n≤3 exact, n=4 counterexample) |

So the losiness story is **settled in theory**: PIC is exact; a *bounded* code provably captures only `|A|/PR` of a
high-PR fragment (Thm 5), and a δ-bounded code is decode-lossless above 2δ (PO-T3). Genuinely open items are narrower
and live in the `examples/complexity` follow-up (which goes *beyond* the paper): the **Route-A NP-hardness gadget** and
the **K-dichotomy** for *deciding* irreducibility, plus the engineering item of a **certified-ε extension of PO-T3**
into the sub-2δ residue.

## 6. The experiments as confirmation tests (theory ⟷ experiment)

**First, a distinction that keeps us honest about which experiment tests which theorem — there are two PRs:**

- **source-PR** (the paper's, ≈45): the participation ratio of the model's *own* logit additive decomposition
  `L_v = Σ_j c_j^v` over its circuits/sources `j`. Thm 4/5/6 are about *this*. Measured via the model's DLA
  (fieldrun `--probe-ablate`, FINDINGS §5c) — call this **track B**.
- **program-PR** (`pic_residue.py`): the effective number of *synthesized candidate programs* in the residue set-cover.
  A property of the surrogate decompilation, **not** the model — call this **track A**.

A track-A number can *look like* a Thm-5 test but is not one. The theorems live on track B.

**Track A — surrogate reach (the rule-extraction goal).** `scope_report.py` (coverage across many problems) +
`pic_residue.py` (ensemble vs irreducible per problem). Measures how far synthesized crisp programs/ensembles reproduce
the model's *output*. Relates to the theory only as: the surrogate's irreducible residue *should* line up with the
model's computed fragment — a correlation to check, not a proof.

**Track B — direct tests of the named theorems on the model itself:**
1. **Diffuseness (Thm 5).** On real residue tokens, measure the model's **source-PR** (DLA) and the `|A|/PR` k-source
   capture. **Confirm:** a k-body captures ≈ `|A|/PR`. **Surprise:** it captures ≫ that ⇒ contributions are *not*
   equitable (`e_m ≠ E/PR`) — the residue has low-rank source structure the equitable-PR model doesn't account for.
2. **Exact recovery (Thm 4).** With the model's own DLA sources, recovery is exact by the theorem; our surrogate-program
   gap measures only basis incompleteness, *not* a Thm-4 failure. **Confirm:** real-DLA recovery is exact.
3. **Rank obstruction ("SVD cannot measure the gap").** Compare effective rank of the residue evidence in the **raw**
   frame vs after the real **Gram kernel `G`** (`unembed_cos`/`rows_f32`), and against a tropical (max-plus) rank proxy.
   **Confirm:** the gap is invisible to linear SVD but present under `G`/max-plus. **Surprise:** `G` makes a high
   source-PR residue low-rank ⇒ compact `pic` is lossless exactly where a token-EDB looks incompressible (a `pic` win).
