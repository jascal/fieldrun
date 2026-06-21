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

# 3. emit each discovered program as recursive Soufflé + residue EDB, and round-trip-verify it reproduces the model
python emit_datalog.py /tmp/listdump.jsonl 4 /tmp/souffle_out     # needs `souffle` on PATH
```

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

## Scope (current)

Flat list folds (`first`/`last`/`len`/`max`/`min`/`sum`/`nth`) + list transforms (`init`/`tail`/`reverse`/`take`/`drop`)
+ binary numeric ops in the DSL; the Datalog emitter covers the folds + transforms (binary-int programs flag
`unsupported-op` and route to residue). Tree recursion + wild-site scoping are future (proposal §9/§11).
