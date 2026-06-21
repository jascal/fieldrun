# rule-synth — faithful bottom-up function synthesizer

Prototype for `RULE_EXTRACTION_PROPOSAL.md` §8/§6: discover the functions a model **actually** implements (faithful to
its *output*, broken/obscure included) by bottom-up program synthesis with observational equivalence, then emit them as
runnable recursive Datalog. See `RESULTS.md` for the validation.

## Pipeline

```bash
# 1. dump (task, list, model-output, truth) JSONL from a fieldrun bundle (needs the `api` feature = default build)
fieldrun --bundle ~/.cache/fieldrun/bundles/Qwen2.5-1.5B/Qwen2.5-1.5B \
         --recursion-explain --list-dump /tmp/listdump.jsonl --n 120 --lmax 7

# 2. synthesize the faithful function per task (random held-out)
python synth.py /tmp/listdump.jsonl 4            # arg2 = DSL depth K (default 4)

# 2b. the discriminating test: train on short lists, test on long (real fn vs in-distribution coincidence)
python synth.py /tmp/listdump.jsonl 4 --ood

# 3. emit each discovered program as recursive Soufflé + a residue, and round-trip-verify it reproduces the model
python emit_datalog.py /tmp/listdump.jsonl 4 /tmp/souffle_out                  # needs `souffle` on PATH
python emit_datalog.py /tmp/listdump.jsonl 4 /tmp/souffle_out --strategy=ensemble   # residue strategy (below)
```

**`--residue-strategy` (the export choice, [PIC_LOSSINESS.md] §4).** The crisp head always emits as recursive Datalog;
the residue (where the head disagrees with the model) is strategy-selected — both reproduce the model 100% by
construction:
- `edb` (default) — a flat `residue(l,o)` fact table (memorise).
- `ensemble` — the held-out-gated decision list (`synth.decision_list`) emitted as **guarded Datalog rules** (guards:
  `len>k`, `first==max`/`first==min`/`last==max`, `is_sorted`) + a *shrunk* residue EDB for what the guards still miss.
  Reduces the residue where a guarded alternate generalises (e.g. `last` 12%→9%).
- `ring` — emit the model's own **block-provenance semiring-Datalog `Π`** for the residue token: `rlogit(v)=Σ_b cw(b,v)`
  over the 57 DLA blocks, decode `argmax` (max-product / tropical T=0). Reproduces the model token by construction
  (LE-T5). `pic` is the *same* facts under the log-semiring (sum-product T=1 = the softmax distribution).
- `margin` — **margin-routed**: route low-margin residue (margin < `--tau`) to `ring` (the model's Π) and high-margin
  residue to `edb`. Margin is the alignment's per-token retrieve-vs-compute router; most residue is high-margin (cheap
  EDB), the low-margin tail gets the mechanistic Π.

`ring`/`pic`/`margin` need the per-token DLA contribution matrix from `fieldrun --ring-dump` (margin + `c[block][digit]`,
where `Σ_b c[b][d] = logit[d]` for the winning digit form, so `argmax_d = the model token`):

```bash
fieldrun --bundle <…1.5B> --recursion-explain --ring-dump /tmp/ring.jsonl --n 30
python emit_datalog.py /tmp/ring.jsonl 4 /tmp/souffle_out --strategy=ring            # all residue → the model's Π
python emit_datalog.py /tmp/ring.jsonl 4 /tmp/souffle_out --strategy=margin --tau=1.0 # low-margin → Π, high-margin → edb
```

`--ring-dump` JSONL schema (one record per example): `{"task","list","out" (model token),"truth","margin","nb"
(#blocks=57),"c"}`, where `c` is an `nb × 10` matrix and `Σ_b c[b][d] = logit[d]` for digit `d` (so `argmax_d = out`).
The emitter turns each routed token's `c` into `cw(l,b,v,w)` facts + `rlogit(l,v,s):- s=sum w:{cw(l,b,v,w)}` and
`ringans(l,v):- s=max s2:{rlogit(l,_,s2)}` — i.e. the model's per-token semiring-Datalog Π.

### Tree-traversal rules (proposal §11 — the untried deterministic class)

The flat-list DSL above is exhausted on flat-list tasks (depth + breadth, see RESULTS). The next *deterministic*
representation it cannot express is **tree catamorphisms** — structural recursion over a parse tree. `eval` of nested
arithmetic is the zero-ICL tree task (the model evaluates `(+ 3 (* 2 5))` natively), so it needs no priming.

```bash
# T1. dump (task=eval, expr, model-output, truth) — nested arithmetic the model EVALUATES
fieldrun --bundle ~/.cache/fieldrun/bundles/Qwen2.5-1.5B/Qwen2.5-1.5B \
         --recursion-explain --tree-dump /tmp/treedump.jsonl --n 250 --dmax 3 --maxv 9

# T2. synthesize the faithful tree catamorphism + the flat-list contrast + §6 tree-ADT Soufflé round-trip
python tree_synth.py /tmp/treedump.jsonl 4 --emit          # --emit runs the recursive-Datalog round-trip
python tree_synth.py /tmp/treedump.jsonl 4 --ood           # OOD-DEPTH: train depth≤2, test depth≥3
```

`tree_synth.py` reports, per task: the discovered catamorphism (`eval`/`maxleaf`/`depth`/…), its held-out faithfulness,
and — the punchline — the best **flat-list** program's faithfulness on the *same* exprs (the list DSL cannot express tree
`eval`, so the gap = what tree traversal recovers). `--emit` closes §6: the catamorphism becomes recursive Soufflé over a
tree ADT (`leaf`/`node` facts + `ev(t,v):-node(t,"+",l,r),ev(l,a),ev(r,b),v=a+b.`) + a residue EDB → reproduces the model
100% by construction.

## What each column means (`synth.py`)

- `model` — how often the model equals the textbook function (its competence on the task).
- `best-1 program` / `faith` — the simplest near-best program (MDL) and its **held-out** faithfulness to the model's output.
- `truth` — how often the discovered program equals the textbook function (low ⇒ a faithful **broken** function).
- `guarded` / `g-faith` — a held-out-gated decision list (shown only when it beats best-1 by ≥4%).
- `resid` = `1 − g-faith` — the per-task residue (the "forge tax" for this DSL); the mean is printed at the end.

## Files

- `synth.py` — the synthesizer (stdlib only): typed DSL, observational-equivalence bank, MDL selection, output-type
  (0–9) constraint, guarded decision-list, random + OOD splits.
- `emit_datalog.py` — discovered program → position-indexed recursive Soufflé (`elem`/`len` + fold) + `residue` EDB +
  wrapper (`answer = program unless residue, else residue`); runs `souffle` and checks `answer == model output`.
- `RESULTS.md` — 0.5B / 1.5B results, OOD analysis, §6 round-trip, caveats.

## Battery & reproducing the sweeps

The `--list-dump` battery is **10 tasks**: the 6 textbook folds (`first`/`last`/`len`/`max`/`min`/`sum`) plus 4 harder
non-textbook ones (`max2`=2nd-largest, `mode`=most-common [ties→smaller], `cmax`=count-of-max, `range`=max−min) that the
model is poor at — the synthesizer surfaces what it *substitutes* (e.g. asked `max2`, the model computes `max`). `synth.py`
and `emit_datalog.py` infer tasks from the dump, so any battery works; `synth.py --tasks=first,max2` restricts the report.

Regenerate the key sweeps (deterministic — default `--seed`):
```bash
for k in 2 3 4 5 6; do python synth.py /tmp/listdump.jsonl $k | grep "mean residue"; done   # head-vs-tail (RESULTS §9)
python synth.py /tmp/listdump.jsonl 5 --ood                                                  # honest (OOD) forge tax
```

### Scope coverage (2.5), PIC residue (3), and the Gram (track B)

```bash
# 2.5 — broad 30-problem battery + coverage curve / head-vs-tail across problems
fieldrun --bundle <…1.5B> --recursion-explain --scope-dump /tmp/scopedump.jsonl --n 80 --lmax 7
python scope_report.py /tmp/scopedump.jsonl 4        # genuine-function vs degenerate-constant-fit, coverage curve

# 3 — PIC residue: per-problem reducible/irreducible label, PR from set-cover, held-out ensemble coverage
python pic_residue.py /tmp/scopedump.jsonl 4 --M=40

# track B — the output Gram kernel G_vw = <U_v,U_w> (tests proved Thm 2; the "SVD can't measure the gap" claim)
fieldrun --bundle <…> --recursion-explain --dump-unembed /tmp/unembed.jsonl --tokens "0,1,2,3,4,5,6,7,8,9"
python gram_probe.py /tmp/unembed.jsonl
```

`PIC_LOSSINESS.md` is the theory note (PIC is lossless w.r.t. the model — kernel-proved Thm 4; the lossy thing is
*compressing* the irreducible region, obstruction = tropical rank; PO-T3 decode-cert ≥2δ). It distinguishes the
*surrogate* program-PR (these scripts) from the model *source*-PR (the proved theorems are about the latter).

## Scope (current)

Two deterministic representations:
- **Flat list** (`synth.py`): folds (`first`/`last`/`len`/`max`/`min`/`sum`/`nth`) + transforms
  (`init`/`tail`/`reverse`/`take`/`drop`) + binary numeric ops; Datalog emitter covers folds + transforms
  (binary-int programs flag `unsupported-op` → residue). Exhausted on flat-list tasks (depth + breadth).
- **Tree catamorphism** (`tree_synth.py`, proposal §11): `eval`/`maxleaf`/`minleaf`/`sumleaf`/`nleaves`/`nops`/`depth`/
  `leftleaf`/`rightleaf` over a binary parse tree + subtree selectors (`left`/`right`) + int combinators; all nine have a
  clean recursive-Datalog catamorphism for the §6 round-trip.

Still future: wild-site scoping (proposal §9); the soft (PIC) representation for whatever survives *both* deterministic
DSLs per problem.
