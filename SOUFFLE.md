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

This machine has **Soufflé 2.5** at `~/.local/bin/souffle` (on `PATH`), built with `ffi openmp
ncurses sqlite zlib`. Confirm with `souffle --version`. If it's missing (a fresh box, or `which
souffle` comes up empty), install it — `souffle` is **not** in the Pop!_OS/Ubuntu-24.04 apt repos,
so on this machine use the official release `.deb`, which needs no root:

```bash
souffle --version           # confirm it's present

# This machine (Pop!_OS / Ubuntu 24.04, x86_64) — official 2.5 release .deb, extracted WITHOUT sudo:
cd /tmp
curl -sSL -o souffle-2.5.deb \
  https://github.com/souffle-lang/souffle/releases/download/2.5/x86_64-ubuntu-2404-souffle-2.5-Linux.deb
dpkg-deb -x souffle-2.5.deb /tmp/souffle-local
cp /tmp/souffle-local/usr/bin/souffle /tmp/souffle-local/usr/bin/souffleprof \
   /tmp/souffle-local/usr/bin/souffle-compile.py ~/.local/bin/      # ~/.local/bin is already on PATH
cp -r /tmp/souffle-local/usr/include/souffle ~/.local/include/      # only needed for compiled mode (`-c`)

# macOS
brew install souffle

# Debian/Ubuntu where the apt package exists
sudo apt-get install souffle

# from source: https://souffle-lang.github.io/build
```

`ncurses` + `sqlite` in the build matters here — they enable the interactive profiler and SQLite
output used in §5.

### 1.1 Compiled mode (`-c` / `-o`) without root

The interpreter needs nothing more. **Compiled synthesis** (`souffle -c`/`-o`, Datalog → native C++ —
the ~200× lossless speedup in [`PROVABLE_OPT_PROPOSAL.md`](./PROVABLE_OPT_PROPOSAL.md) §2.1) needs the
Soufflé **source headers** plus the `sqlite`/`zlib`/`ncurses` dev headers and link libs, which the
runtime `.deb` omits. On a box without root, install them locally:

```bash
# Soufflé source headers (incl. CompiledSouffle.h, absent from the runtime .deb)
curl -sSL https://github.com/souffle-lang/souffle/archive/refs/tags/2.5.tar.gz | tar xz -C /tmp
cp -r /tmp/souffle-2.5/src/include/souffle ~/.local/include/

# sqlite/zlib/ncurses dev headers + link libs — apt-get download needs no root
cd /tmp && apt-get download libsqlite3-dev zlib1g-dev libncurses-dev libtinfo6
for d in *.deb; do dpkg-deb -x "$d" /tmp/dev; done
cp /tmp/dev/usr/include/{sqlite3.h,zlib.h,zconf.h} ~/.local/include/
cp -P /tmp/dev/usr/lib/x86_64-linux-gnu/lib{sqlite3,z,ncurses,tinfo}.so* ~/.local/lib/

# point souffle's compile driver at the local libs (it hardcodes /usr/lib absolute paths)
sed -i "s#/usr/lib/x86_64-linux-gnu/lib\(sqlite3\|z\|ncurses\).so#$HOME/.local/lib/lib\1.so#g" \
    ~/.local/bin/souffle-compile.py

# then, with the local include/lib dirs on the toolchain paths:
export CPATH="$HOME/.local/include:$CPATH"
export LIBRARY_PATH="$HOME/.local/lib:$LIBRARY_PATH"
export LD_LIBRARY_PATH="$HOME/.local/lib:$LD_LIBRARY_PATH"
souffle -o decoder prog.dl && ./decoder -F ctx -D -     # native binary; same result, ~200× faster
```

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
fieldrun --bundle <small-rope> export --logic-whole --out cf.dl   # LO3a: CONTEXT-FREE whole-model emit (§8)
souffle cf.dl -F ctx -D -                       # compute next token for ANY ctx/token.facts (pos<TAB>id)
```

---

## 8. The context-free whole-model emit (LO3a) — `export --logic-whole`

Everything in §0–§6.5 produces artifacts **specialized to one context** — a single decision, or a
stitched trace of one reply. None of them generalize. `LOGIC_EXPORT.md` LO3a asks for the opposite: a
**context-free whole-model emit** — one `.dl` that takes an arbitrary token context as `.input` and
*computes* the next token from scratch, runnable in Soufflé on inputs the exporter never saw.

**This now exists** (`fieldrun export --logic-whole`), demonstrated on the rope family at small scale:

```bash
# emit the WHOLE forward pass of a (small) rope bundle as one context-free Datalog program:
fieldrun --bundle <small-rope-bundle> export --logic-whole --out whole.dl --maxpos 64

# run it on ANY context — the only input is token(pos,id):
printf '0\t785\n1\t6722\n2\t315\n' > ctx/token.facts
souffle whole.dl -F ctx -D -        # -> decide(v) = the next-token argmax, computed from scratch
```

Unlike `export --logic` (whose `contrib(block,token,w)` facts are **partial evaluations** baked to one
context), this emits the **computation itself**: weights are facts, the forward pass is rules —
RMSNorm, RoPE attention (GQA, causal), SwiGLU MLP, the tied/untied unembed, and the argmax. Change
`token.facts` and Soufflé recomputes; it answers contexts the emitter never saw. That is the LO3a
property.

### How it fits in plain Datalog (no FFI)

Soufflé has only `+ - * / ^` and `sum`/`max` aggregates — no `exp`/`sqrt`/`sin`/`cos`. That turns out
to be enough:
- `sqrt(x) = x ^ 0.5` (RMSNorm), `exp(x) = E ^ x` (softmax, SiLU) — `^` does real powers.
- RoPE `sin`/`cos` depend **only on position**, never on token content, so they are precomputed
  **model-constant facts** (`rope_cos(pos,j,c)`), not partial evaluations of the input.
- matmul `C[i,o] = Σ_k A[i,k]·W[k,o]` is a `sum`-aggregate; softmax is `max` then `^(s-m)` then `/Z`;
  argmax is the same `max`-witness rule the rest of this doc uses.

So it is plain, standard Datalog — the same "verify in a neutral engine" property §0 relies on, extended
to the whole model. Verified: across base / +bias (Qwen2.5-style) / untied / bias+untied tiny rope
bundles, Soufflé's `decide` matches both fieldrun's own forward and an independent numpy reference on
every held-out context. (Reproduce: `lo3a/verify_all.py`.)

### The wall moved from "possible?" to "compact?"

LO3a was filed as the open frontier because the **computed fragment** (the dense forge tax) is the
dense-Gram / high-treewidth region (`LE-T2`/`LE-T4`) with **no compact extension**. That is exactly what
survives: the program *exists and is correct for any model*, but its size is the embed/unembed fact
count `vocab × d`. For a tiny bundle that is a few thousand facts; for Qwen2.5-0.5B it is ~136M embed
facts alone, so `export --logic-whole` **refuses by default** (naming LE-T4) and needs `--force`. The
open part of LO3a is no longer *can you emit a context-free program* — you can — it is *can the dense
fragment be emitted compactly*, and `LE-T2`/`LE-T4` say it cannot. The size guard is that wall, made
operational.

`fieldrun stitch` (§6.5) still solves the other, tractable artifact — one file for one query's whole
trajectory — and remains **not** LO3a (it does not generalize). `export --logic-whole` is LO3a: it
generalizes, at the cost of carrying the whole model.
