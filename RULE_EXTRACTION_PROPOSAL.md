# Bottom-Up Rule Extraction — Faithful Function Synthesis

**Status:** proposal (for review). **Depends on:** the RETRIEVED/COMPUTED causal router (PR #82). **Feeds:** the recursive-Datalog emitter (`recursion_dl`).

## 1. Goal

Discover the functions a model *actually implements* — including ones that are **objectively wrong** or have **no human name** — by synthesizing the smallest program, over a broad DSL, that reproduces the **model's own output**. Whatever cannot be synthesized within budget is left as the **kernel** (the forge-tax residue), not forced into a clean rule.

Two non-negotiable principles, both from the project's standing direction:

- **Faithfulness, not correctness.** The synthesis target is `model(x)`, *not* the textbook answer. A discovered function may be "broken" (e.g. the model's `min` is actually `first`); capturing that *is* the interpretability result. Fixing/optimizing rules is a separate, later concern.
- **Discover, don't checklist.** We do not test a fixed list of human-named candidates and stop. The candidate battery is a *baseline/streetlight*; the functions worth finding are the ones we'd never have named.

### Why now (the motivating measurement)

On Qwen-0.5B, faithful cross-attribution over a *named* candidate set (`--list-attribute`) found:

- asked-`min` ≈ **`first`** (52% vs min 40%) — a positional heuristic, not min;
- `max` ≈ 60% real-max + 30% `last` (a blend);
- **every named fit is only ~50–60%** — i.e. **~40–50% of even these "simple" functions is not any human-named function.**

That residue is the target. The named set demonstrably does not contain the model's real functions, so the next tool must *synthesize* functions, not select them.

## 2. Engine: bottom-up enumeration with observational equivalence

We use **bottom-up program synthesis** (à la EUSolver / PROBE) with **observational-equivalence pruning** — *not* an ILP solver (Popper/Metagol/SMT). Rationale: ILP is for recovering *hidden* structure from *sparse* I/O. fieldrun is a glass box — we can query the model for *dense* I/O on any input we like — so plain enumeration suffices, with no external dependency.

**Inputs**
- A *task site*: a family of structured inputs (lists, expressions, or a position in a prompt) where the model emits a single-token answer.
- A sample `I` of inputs drawn from the input distribution.
- `obs[x] = model(x)` for `x ∈ I` — the **faithful targets**, gathered via fieldrun (the model's actual outputs, right or wrong).

**Procedure**
1. `bank ←` size-1 programs (input variables + constants), each keyed by its **output signature** `sig(p) = ⟨eval(p, x) : x ∈ I⟩`.
2. For `size = 2 … K`:
   - For each typed primitive `f` and each type-correct tuple of sub-programs in `bank` summing to `size`:
     - `p ← f(subprograms)`; `s ← sig(p)`.
     - If `s ∈ bank` → **skip** (observational equivalence: keep only the smallest program per behaviour). This is the scalability key — the bank is bounded by the number of *distinct behaviours on `I`*, not the number of programs.
     - Else `bank[s] ← p`; `score ← |{x : eval(p,x) = obs[x]}| / |I|`; track the best `(p, score, size)`.
3. Stop when a program reaches faithfulness `≥ τ` (e.g. 0.95) or the size budget `K` is exhausted.

**Output per site:** the best program (the discovered function), its faithfulness %, and its size. If best `< τ`, the function is only *partially* captured; the residual is `1 − faithfulness`.

## 3. The DSL (typed, extensible)

Types: `Int` (digit/value 0–9), `List[Int]`, `Bool`.

| Class | Primitives |
|---|---|
| `List → Int` | `first` `last` `nth(k)` `len` `max` `min` `sum` `prod` `count(v)` `index-of(v)` `mode` |
| `List → List` | `tail` `init` `reverse` `take(k)` `drop(k)` `sort` `filter(pred)` `map(g)` |
| `Int×Int → Int` | `+ − × ÷ mod min max` |
| `Int×Int → Bool` | `< > = ≤ ≥` |
| combinators | `fold(f, z, xs)` `scanl` `if(pred, a, b)` `compose` |
| terminals | input variables, constants `0…9` |

The DSL is the knob that decides what we can discover: too narrow → can't reach obscure functions; too broad → enumeration blows up. Start moderate; expand (tree-recursion primitives, etc.) as the residue demands.

## 4. The bag-of-heuristics: guarded / piecewise programs

Models often *blend* (the model's `max` = max 60% / last 30%). A single clean function can't be faithful to that. So after enumerating candidate programs, we allow **guarded composition**:

```
if pred(x) then prog_a(x) else prog_b(x)
```

with `pred` drawn from a predicate DSL (`len(x)>3`, `is-sorted(x)`, `first(x)=max(x)`, …). We greedily add guards that raise total faithfulness (a decision-list / CART-style fit over the top candidate programs). This is exactly the **"broken rule with built-in cuts"** idea: a piecewise program that reproduces the model's heuristic blend faithfully even when no single function does. The residue *after* guards is the genuinely-alien core.

## 5. The residue is the kernel (forge tax)

Whatever the synthesizer cannot reach within `(DSL × size budget K × guard depth)` is **flagged as kernel, not forced into a rule.** This is the *measured* here-be-dragons fraction per site — the number the whole program is trying to drive down, honestly.

## 6. Mapping to Datalog

The discovered artifacts translate directly, and the RETRIEVED/COMPUTED causal router (PR #82) decides which half each datum goes to:

- a discovered **fold** → a recursive Datalog **rule** (`eval` over the list/structure);
- **primitives** → built-in rules;
- **RETRIEVED** values (causally certified copies) → **EDB ground facts** (the flat store); **COMPUTED** values (re-derived) → **IDB rules** with a fresh variable for the result;
- **guards** → rule bodies with conditions (conflict resolution / cuts);
- **residue** → the kernel (semiring / retrieved fallback), preserved so the whole program stays output-faithful.

## 7. Validation

1. **Recovery (sanity):** does the synthesizer recover `first`/`last`/`max` where the model *is* clean?
2. **Faithful-to-broken:** does it recover the model's `min` ≈ `first` (the broken function), faithfully?
3. **Held-out faithfulness:** the discovered program reproduces the model on **new** inputs (not the synthesis sample) — the overfitting guard.
4. **Variance:** faithfulness ± sd over seeds/samples (the `--seeds` discipline added in #82).

## 8. First decisive step (smallest end-to-end)

1. fieldrun: a mode that, for a task family (lists), dumps `(input, model-output)` pairs to a file (extends the existing `--list-*` harness).
2. A Python synthesizer (pure / numpy) implementing §2 over the §3 DSL, fit to the model's output, reporting **best program · faithfulness % · residual** per task.
3. Validate per §7 on 0.5B/1.5B.
4. Then: guards (§4), tree recursion, and wire discovered folds → recursive Datalog (§6).

## 9. Open questions / risks (for review)

- **DSL completeness vs. blow-up.** What primitive set is broad enough to name the residue without an intractable search? (Observational equivalence bounds it by distinct behaviours, but the right DSL is the open design choice.)
- **Faithfulness threshold / piecewise overfitting.** Guards can fit noise; held-out faithfulness (§7.3) is the control, but the guard-complexity penalty (MDL) needs tuning.
- **Tree / nested recursion.** Lists are flat; the model's real recursion (parse structure, nesting) needs structural primitives — a later DSL extension.
- **Site definition in the wild.** For synthetic tasks the I/O is clean; for arbitrary prompt positions the local function's "input" is not cleanly defined. Early work stays on controlled task families.
- **Relation to the kernel.** Is the residue a *short head + long Zipf tail* (a few function-families cover most, tail never closes) — the measurement that would tell us whether the kernel floor is small or large?

---

*Engine: bottom-up enumeration + observational equivalence (EUSolver/PROBE-class), dependency-free. Novelty for this use: the target is the model's **output** (faithful, "wrong" allowed) rather than a spec, plus guarded/piecewise composition for heuristic blends and an explicit residue = kernel.*
