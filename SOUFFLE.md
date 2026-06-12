# Running fieldrun logic exports with Soufflé

fieldrun's logic export (`LOGIC_EXPORT.md`) emits **one next-token decision as a runnable Datalog
program** (`*.dl`). Those files are written to be **Soufflé-compatible**, so you can verify a decode
with an off-the-shelf Datalog engine instead of trusting `fieldrun eval`. This doc covers installing
Soufflé, running a `.dl`, the flags that matter for these programs, and the **interactive
exploration** modes (provenance explorer, SQLite shell, profiler).

> The meaningful result: when a neutral engine like Soufflé reproduces fieldrun's argmax, it confirms
> the export is plain, standard Datalog — not relying on any fieldrun-specific evaluation semantics.

---

## 0. What's in a fieldrun `.dl`

Every exported program has the same shape (see `logic-001.dl`, `logic-003.dl`):

```prolog
.decl candidate(t:number)              // the candidate token ids (predicted ∪ runner-up ∪ KB)
.decl contrib(block:symbol, t:number, w:float)   // each block's exact contribution to a token's logit
.decl logit(t:number, s:float)         // derived: per-token summed logit
.decl decide(t:number)                 // derived: the argmax token

candidate(9707).                        // facts: the candidate set
contrib("L23.mlp", 9707, 6.1140).       // facts: per-block contributions
// ... many contrib facts ...

logit(T, S) :- candidate(T), S = sum W : { contrib(_, T, W) }.   // ⊗ = +  (accumulate blocks)
decide(T)   :- logit(T, S), S = max S2 : { logit(_, S2) }.        // ⊕ = max (argmax, T=0)
.output decide
```

`logit/2` sums each token's `contrib` weights; `decide/1` takes the argmax. The `.output decide`
directive is what makes Soufflé emit a result. That's the whole program — facts + two rules.

---

## 1. Install Soufflé

This machine already has it (`/usr/local/bin/souffle`, **version 2.5**, built with `ffi ncurses
sqlite zlib`). To check / install elsewhere:

```bash
souffle --version           # confirm it's present

# macOS
brew install souffle

# Debian/Ubuntu
sudo apt-get install souffle

# from source: https://souffle-lang.github.io/build
```

`ncurses` + `sqlite` in the build matters here — they enable the interactive profiler and SQLite
output used in §5.

---

## 2. Run a `.dl` — the two execution modes

### Interpreter (default — no compile step, best for these small programs)

```bash
souffle logic-001.dl -D -          # -D -  prints every .output relation to stdout
```
```
---------------
decide
t
===============
9707
===============
```

`9707` is the token id (`"Hello"` per the file's header comment). To write CSV files instead of
stdout, point `-D` at a directory:

```bash
mkdir -p out
souffle logic-001.dl -D out        # writes out/decide.csv  (one id: 9707)
```

### Compiled (generates C++, builds a native binary — for large programs)

```bash
souffle -c logic-001.dl -D -                       # compile + run in one shot
# or build a reusable executable and run it yourself:
souffle -o decoder logic-001.dl && ./decoder -D -
```

For a 2-candidate decode the interpreter is instant; `-c` only pays off on big fact sets.

---

## 3. Flags you'll actually use

| Flag | Meaning |
|------|---------|
| `-D <dir>` / `-D -` | output dir for `.output` relations; `-` = stdout |
| `-F <dir>` | input dir for `.input` relations (not needed — facts are inlined) |
| `-c` | compile to native C++ and run (faster on large data) |
| `-o <name>` | emit a standalone executable, don't run it |
| `-j <N>` | run with N threads |
| `-t explain` / `-t explore` | **interactive provenance** — see §4 |
| `-p <file>` | write a profile log (open it with `souffleprof`, §5) |
| `--live-profile` | live ncurses profiler while it runs |
| `--show=<mode>` | dump compiler internals (`initial-ast`, `transformed-ast`, `type-analysis`, `precedence-graph`, `scc-graph`, …) |
| `--parse-errors` | check the program parses, then exit |

### Harmless warnings on these files

```
Warning: No rules/facts defined for relation retrieved ...
Warning: No rules/facts defined for relation induction_copy ...
Warning: No rules/facts defined for relation ngram_succ ...
```

Those three relations are declared as scaffolding (Tier-A retrieval / induction-head facts) but a
given decode trace may not populate them. They don't affect `decide`. Silence them by deleting the
unused `.decl` lines, or just ignore them.

---

## 4. Interactive exploration — the provenance explorer

There is **no Prolog-style query REPL** in Soufflé (it's a bulk, bottom-up evaluator: you declare
`.output` relations and it materializes them). But it ships something arguably more useful for these
files: a **provenance shell** that answers *"why is this tuple in the output, and how was it
derived?"* with a proof tree. This is the closest thing to "interact and explore."

### `explain` mode — proof trees

```bash
souffle -t explain logic-001.dl
```

This evaluates the program, then drops you into an interactive prompt. Ask why the decision holds:

```
explain decide(9707)
```
```
candidate(9707) ...          __agg_single(18.381300)
-----------------------------------------------------(R1)
            logit(9707, 18.381300)
---------------------------------------------------------(R1)
                  decide(9707)
```

You can read off the derivation: `candidate(9707)` plus the aggregated logit `18.3813` fire rule R1
to produce `logit(9707, …)`, which in turn produces `decide(9707)`. Try the same for a token that
did **not** win to see it has no `decide` proof:

```
explain logit(39814, 18.252)     # the runner-up "Sure" — has a logit proof but no decide proof
```

Useful commands inside the shell:

| Command | Effect |
|---------|--------|
| `explain <relation>(<args>)` | print the proof tree for one tuple |
| `setdepth <N>` | how many levels of the proof tree to expand (default 4) |
| `rule <relation> <n>` | show the n-th rule defining a relation |
| `output <file>` | write the next proof tree to a file instead of the terminal |
| `format <json|proof>` | switch proof-tree rendering |
| `q` / `exit` | quit |

Drive it non-interactively by piping commands in:

```bash
printf 'explain decide(9707)\nq\n' | souffle -t explain logic-001.dl
```

### `explore` mode — navigate subtrees

```bash
souffle -t explore logic-001.dl
```

Same as `explain`, plus you can **walk into** a node's children (`subproof <id>`) to expand a
specific branch of the derivation instead of printing the whole tree at once — handy when a token's
logit aggregates dozens of `contrib` facts.

> Provenance adds instrumentation overhead, so use it for inspection, not for timing runs.

---

## 5. Other interactive / exploratory paths

### a) Expose intermediate relations

`decide` is the only declared output, but you can materialize the intermediate `logit` (and the raw
`candidate`/`contrib`) to inspect the full scoreboard. Append `.output` directives to a copy:

```bash
cp logic-001.dl /tmp/l1.dl
printf '\n.output logit\n.output candidate\n' >> /tmp/l1.dl
souffle /tmp/l1.dl -D -
```
```
logit
t       s
===============
9707    18.381300000000003
39814   18.252000000000006
```

Now you can see the runner-up's score and the margin directly, without `fieldrun eval`.

### b) SQLite output → real SQL queries

Soufflé 2.5 was built with `sqlite`. Send relations to a database and explore them with the
`sqlite3` shell — joins, `ORDER BY`, `WHERE`, the lot:

```bash
cp logic-001.dl /tmp/l1sql.dl
cat >> /tmp/l1sql.dl <<'EOF'

.output logit(IO=sqlite, dbname="/tmp/decode.db")
.output decide(IO=sqlite, dbname="/tmp/decode.db")
EOF
souffle /tmp/l1sql.dl

sqlite3 /tmp/decode.db ".tables"
sqlite3 /tmp/decode.db "SELECT * FROM logit ORDER BY s DESC;"
```

> **Float caveat:** Soufflé stores `float` columns in SQLite as their raw IEEE-754 64-bit pattern
> (you'll see large integers like `4625867093671540478`, which is the bit-encoding of `18.3813`).
> For human-readable floats, prefer CSV/stdout output (§5a); use SQLite for `number`/`symbol`
> columns and set-membership queries.

### c) Interactive profiler

For larger programs, profile the run and explore where time/tuples went in an ncurses UI:

```bash
souffle logic-001.dl -p /tmp/prof.log     # produce a profile log
souffleprof /tmp/prof.log                 # interactive profiler (rul, rel, top, help inside)
# or watch it live:
souffle logic-001.dl --live-profile
```

Inside `souffleprof`: `rel` lists relations with tuple counts, `rul` lists rules by cost, `top`
shows the hot spots, `help` lists commands. (Overkill for a 2-candidate decode, but the path is
there for big exports.)

### d) Inspect the compiled program

To see how Soufflé desugars the aggregates and plans the rules:

```bash
souffle --show=transformed-ast logic-001.dl      # the rewritten Datalog
souffle --show=type-analysis    logic-001.dl      # inferred types
souffle --show=scc-graph        logic-001.dl      # stratification / dependency SCCs
```

---

## 6. Cross-check against `fieldrun eval`

Soufflé gives you the argmax (`decide`); `fieldrun eval` gives the same decode **plus** the logit,
the runner-up margin, and — via the `log` semiring — the full distribution Soufflé can't produce
from these rules:

```bash
# Soufflé: argmax only
souffle logic-001.dl -D -                                   # -> 9707

# fieldrun: argmax + margin (max semiring)
./target/release/fieldrun eval logic-001.dl --semiring max
#   decide(9707).   % logit 18.3813   runner-up 39814 logit 18.2520  margin +0.1293

# fieldrun: full softmax over candidates (log semiring)
./target/release/fieldrun eval logic-001.dl --semiring log
#   P 0.532  token 9707   |   P 0.468  token 39814
```

Verified on both checked-in exports — the two engines agree exactly on the decode:

| File | Soufflé `decide` | fieldrun `--semiring max` | logit | log-semiring P(top) |
|------|------------------|---------------------------|-------|---------------------|
| `logic-001.dl` | `9707` ("Hello") | `decide(9707)` | 18.3813 | 0.532 (vs 0.468) |
| `logic-003.dl` | `785` ("The")   | `decide(785)`  | 23.4226 | 0.979 (vs 0.021) |

Souffle agreeing with fieldrun's `max` decode is the round-trip faithfulness check the `.dl` header
asserts (`✓ FAITHFUL`). Use Soufflé as the independent verifier; use `fieldrun eval` when you want
margins and the distribution.

---

## 6.5 Exporting a WHOLE reply, and stitching it into one file

A single `.dl` is **one next-token decision**. To capture an entire reply there are two pieces:

**Emit the whole reply as a trace** — one `.dl` per generated token:
```bash
# in the chat REPL — generates the whole reply, writes one .dl per token:
/export-logic out.dl What is the capital of France?
#   → out.000.dl, out.001.dl, …  (each FAITHFUL ✓)   reply: The capital of France is Paris.

# or non-interactively from a context:
fieldrun --bundle <b> --ids ids.json --export-logic out --steps 8
```

**Stitch the parts into ONE step-indexed program** — `fieldrun stitch` (pure text, no model):
```bash
fieldrun stitch out.*.dl -o whole.dl        # batch: merges the per-step files
souffle whole.dl -D -                         # decide(step, token) for the WHOLE reply, one run
fieldrun eval whole.dl --semiring max         # same, step-aware
```
```
decide
step    t
===============
0       785      // The
1       6722     //  capital
2       315      //  of
3       9625     //  France
4       374      //  is
5       12095    //  Paris
6       13       // .
```

The stitched program adds a `step` index to every relation so the per-step sums never collide
(a naïve `cat` of the parts would redeclare relations and make the aggregate sum `contrib` *across*
tokens — silently wrong):
```prolog
.decl candidate(step:number, t:number)
.decl contrib(step:number, block:symbol, t:number, w:float)
logit(Step, T, S) :- candidate(Step, T), S = sum W : { contrib(Step, _, T, W) }.   // scoped per step
decide(Step, T)   :- logit(Step, T, S), S = max S2 : { logit(Step, _, S2) }.        // argmax per step
.output decide
```

### What this single file is — and is not

This is the **single `.dl` for the whole query**, but be precise about what that means:

- ✅ It is a **complete, runnable transcript** of *this* query's greedy decode — every token, with its
  candidate set and per-block `contrib` facts, in one program. Soufflé/`eval` replay the whole reply.
- ❌ It is **not a model you can ask new questions**. The `contrib` weights are *partial evaluations* —
  numbers the model already computed for *these* contexts. Change the prompt and they're meaningless.
  Running `whole.dl` always reproduces "The capital of France is Paris."; it cannot answer anything else.

A program that *computes* the reply to an arbitrary new query from scratch is the **context-free
whole-model emit (LOGIC_EXPORT `LO3a`, open)** — see §8. Stitching gives you the first; it does not
and cannot give you the second.

---

## 7. One-liner cheat sheet

```bash
souffle prog.dl -D -                          # run, print decide to stdout
souffle prog.dl -D out                        # run, write out/decide.csv
souffle -c prog.dl -D -                        # compiled run (large programs)
souffle -t explain prog.dl                     # interactive: why is decide(X) true?
souffle -t explore prog.dl                     # interactive: navigate the proof tree
printf 'explain decide(9707)\nq\n' | souffle -t explain prog.dl   # scripted provenance
souffle prog.dl -p prof.log && souffleprof prof.log              # profile + explore
souffle --parse-errors prog.dl                 # syntax check only
souffle --show=transformed-ast prog.dl         # see the desugared program
fieldrun stitch out.*.dl -o whole.dl           # merge a per-step trace into ONE step-indexed .dl
souffle whole.dl -D -                           # decide(step, token) for the whole reply
```

---

## 8. The open frontier: a single program that answers NEW queries (LO3a)

Everything above produces artifacts **specialized to one context** — a single decision, or a stitched
trace of one reply. None of them generalize. The standing research goal (`LOGIC_EXPORT.md` LO3a) is the
**context-free whole-model emit**: one `.dl` that takes an arbitrary token context as `.input` and
*computes* the next token — a program you could actually query, run in Soufflé on inputs the exporter
never saw, and statically verify.

Why it's hard (and why it's the same wall as the rest of the proposal):
- The current `contrib(block, token, w)` facts are **partial evaluations** — the dot products
  `⟨d_j, U_v⟩` already resolved for the specific residual directions a given context produced. A
  context-free program can't hardcode them; it has to export the computation that *produces* them.
- That computation is attention (a softmax over QK across positions) and the MLPs (GELU) — real-valued
  **nonlinear** functions. Semiring Datalog natively gives you `+` and `max`; the full forward pass
  needs general arithmetic (Soufflé functors/FFI), at which point you've re-embedded the kernels rather
  than exported logic.
- The retrievable fragment (induction = a recursive clause, n-gram = a fact) *is* compactly
  context-free already. The **computed fragment** (the dense "forge tax") is the dense-Gram /
  high-treewidth region — `LE-T2`/`LE-T4` — which provably has no compact extension.

So `fieldrun stitch` deliberately solves the *tractable* half — one file for one query's whole
trajectory — and labels itself as **not** LO3a in its own header. LO3a is the next goal, not a missing
flag.
