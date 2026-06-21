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

## 10. Answers to review — round 1

**(Q1) DSL grammar + search-space.** Concrete initial grammar (≈18 primitives):

```
E_int  ::= var | 0..9
         | first(E_list) | last(E_list) | nth(E_list, E_int) | len(E_list)
         | max(E_list) | min(E_list) | sum(E_list) | count(E_list, E_int)
         | E_int (+ | - | * | min | max) E_int
E_list ::= var | tail(E_list) | reverse(E_list) | take(E_list,E_int) | drop(E_list,E_int) | sort(E_list)
E_bool ::= E_int (< | > | =) E_int | is_sorted(E_list)
```

Naive program count at size `k` is exponential (~`branching^k`). The point of **observational equivalence** is that the working set (`bank`) is bounded **not by program count but by the number of distinct behaviours on the sample `I`** — a program collapses into an existing bank entry the moment its output vector `⟨eval(p,x):x∈I⟩` is already present. For a `List→Int` (single-digit) task the behaviour space is bounded by the distinct vectors the DSL actually realises, which in the EUSolver/PROBE literature stays in the `10³–10⁵` range to depth ≈6. We will **instrument and cap the bank** and report where it saturates — i.e. the real numbers come from the prototype, but the *bound* is the distinct-behaviour count, not `20^k`.

**(Q2) Guard learning (avoiding overfit).** Staged, not joint: **(i)** synthesise the *pieces* (programs that each cover a subset of inputs well); **(ii)** learn a shallow **decision list** over a small, fixed predicate DSL (`len>k`, `is_sorted`, `first=max`, …) to route between pieces. Three controls: an **MDL objective** `cost = Σ(piece sizes) + (guard complexity) + λ·errors` (a guard earns its place only if it cuts errors by more than its description length); **held-out faithfulness** (guards fit on train, scored on held-out — reject guards that don't generalise); and a **depth-limited predicate set** so guards can't memorise. This is the operational form of the "broken rule with cuts."

**(Q3) Residue criterion + representation.** Stop synthesising for a site when *any* of: faithfulness `≥ τ` (e.g. 0.95); size/guard budget exhausted; or the **marginal MDL gain** of the next piece/guard `< ε` (added complexity stops paying). The residue is the uncovered `(input, model-output)` set — represented **not as a black-box oracle but as an explicit EDB fact table** plus a last-resort clause: `answer(X,O) :- residue(X,O).` So the kernel is *data, not understanding* — finite, inspectable, and it keeps the whole program output-faithful. (Optionally a piece is only *admitted* if it passes the #82 causal gate, so residue = "what no causally-validated program covers.")

**(Q4) Tree recursion vs. wild sites — different difficulties.** The **engine extends unchanged** to trees: swap `List` for a `Tree` ADT and add a tree-fold combinator; bottom-up + OE is type-agnostic. (Arithmetic expressions are already trees, so we have a tree domain in hand.) The genuinely harder problem is **site definition in the wild**: at an arbitrary prompt position the local function's *input* isn't a clean typed object. That needs the input to be *scoped first* — which is exactly what the #82 causal dataflow (what the site causally reads) can provide. So: tree recursion is a DSL extension; wild-site scoping is a real open sub-problem, and the **first prototype stays on controlled task families with defined I/O**.

**(Q5) Validation beyond observational match.** Observational match on the synthesis distribution is necessary, not sufficient. Additional checks: **(a) OOD held-out** (longer lists, shifted distribution — a program that matches in-distribution but breaks OOD isn't the real function); **(b) causal cross-check** with the #82 tooling — if the synthesised program uses `last`, does `bind`/`value-patch` show the model causally *reads* that position? RETRIEVED/COMPUTED labels must agree with the program's structure; **(c) agreement with cross-attribution** (does the synthesised function match the `--list-attribute` best-fit, e.g. `min≈first`?). Faithful decompilation = observational + causal-consistency + probe-agreement, not I/O alone.

**(Q6) Implementation surface — staged.** **Offline first:** a pure-Python synthesiser ingesting a dumped `(input, model-output)` file (reuses the `experiments/value_probe/train_probe.py` pattern) — fast DSL/loop iteration, no Rust rebuild per experiment. **Then fold the engine into fieldrun** (a `recursion_synth` module) once the DSL + loop stabilise, for speed and tight coupling to the causal router. The only fieldrun-side need up front is a `(input, model-output)` **dump mode** extending the existing `--list-*` harness.

**(Q7) Toy Datalog emission.** The model's `max ≈ max(60%) + last(30%)` blend synthesises (toy guard) to `maxish(xs) = if is_sorted(xs) then last(xs) else max(xs)`, which emits:

```prolog
maxf([X], X).                              % fold (COMPUTED → recursive IDB rule)
maxf([H|T], M)  :- maxf(T, M0), M = max(H, M0).
lastf([X], X).                             % last (RETRIEVED-leaning)
lastf([_|T], X) :- lastf(T, X).
answer(Xs, R)   :- is_sorted(Xs), lastf(Xs, R).        % guarded pieces
answer(Xs, R)   :- !is_sorted(Xs), maxf(Xs, R).
answer(Xs, R)   :- residue(Xs, R).                     % kernel fallback (EDB facts)
```

In Soufflé the list/recursion is via record ADTs and the guard via stratified negation — mechanical, if not free. Folds → recursive rules; guards → rule-body conditions; residue → EDB facts + a fallback clause.

## 11. Answers to review — round 2

*(Round-1 review predated §10, which already covers the DSL grammar, guard-overfit controls, the residue stop-criterion, and a Datalog emission example. The genuinely new points:)*

**Stochasticity (§2).** Current task sites use **greedy** decoding, so the model is deterministic and `obs[x]` is a single token — observational equivalence is exact. Under sampled decoding `obs[x]` becomes a distribution; then either **(a)** fit the **argmax/mode** (keeps the exact-match engine; the faithful target is the model's most-likely answer), or **(b)** score by **agreement with the sampled distribution** (expected match, or a soft observational-equivalence on the induced answer distribution). We start with (a); the engine is unchanged.

**Higher-order slots (§3).** We do **not** enumerate arbitrary lambdas for `filter(pred)` / `fold(f,·)` / `map(g)` initially — that's the combinatorial killer. The higher-order slots draw from a **closed set of built-ins**: `fold`'s combiner is restricted to the binary `Int×Int→Int` primitives (`+ × min max …`); `filter`'s `pred` / `map`'s `g` draw from the small fixed predicate/function set (the same shallow set used for guards). Higher-order is **closed and bounded**, not open enumeration. If the residue demands richer `pred`/`f`, we enumerate small fragments for *those slots* in a later, separately-budgeted pass.

**Worked synthesis trace (illustrative).** Tiny `I = [[3,7,2], [5,1,8], [4,4,9]]`; the model's outputs when asked `min` are `obs = ⟨3, 5, 4⟩` (it returns the *first* element — the broken behaviour #82 found). Bottom-up:

```
size 1: xs ; const 0..9
size 2: first(xs) → ⟨3,5,4⟩   100%   ← matches obs exactly
        min(xs)   → ⟨2,1,4⟩    33%   (the TEXTBOOK answer — wrong for THIS model)
        last(xs)  → ⟨2,8,9⟩     0%
        max(xs)   → ⟨7,8,9⟩     0%
        len(xs)   → ⟨3,3,3⟩     0%
        nth(xs,0) → ⟨3,5,4⟩    ≡ first(xs) → OE PRUNE (keep the smaller)
```

`first(xs)` is found at size 2 with 100% faithfulness ⇒ the discovered function for asked-`min` is **`first`, not `min`**. The trace shows both the **faithful target** (we fit the model's `⟨3,5,4⟩`, not the textbook `⟨2,1,4⟩`) and **OE pruning** (`nth(xs,0) ≡ first(xs)` collapse). On realistic sites we expect the bank to be dominated by such collapses — most size-`k` programs reduce to a handful of behaviours — which is why the working set tracks distinct behaviours, not program count.

**How this feeds the roadmap.** The discovered programs are the legible half of the minimum-to-run decomposition: folds → recursive `recursion_dl` rules, primitives/constants → the flat EDB store, residue → the kernel. Running the synthesiser across a corpus and accreting the rule library + fact store is the whole-model export step; the **residue fraction it leaves is the measured "forge tax"** — the number the faithful-decompilation program is driving down, capability by capability.

---

*Engine: bottom-up enumeration + observational equivalence (EUSolver/PROBE-class), dependency-free. Novelty for this use: the target is the model's **output** (faithful, "wrong" allowed) rather than a spec, plus guarded/piecewise composition for heuristic blends and an explicit residue = kernel.*
