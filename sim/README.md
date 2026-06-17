# The Threx — an unrealistically small language model, end to end

This folder builds the interactive intuition-builder served at
`docs/intuition.html`: a genuinely tiny LLM that makes the **three kinds of
next-token decision** the research distinguishes — **retrieved**, **selected**,
and **composed** — and a stepped visualization that runs the model live in the
browser and labels what every part is doing. Every number on the page is real:
the model is trained, exported, and **validated top-1 against the `fieldrun`
binary**, and the page re-runs the exact same arithmetic in JavaScript.

## The Threx (the back-story)

The Threx are blind foragers in the deep dark of an alien sea. They never see
one another, so everything they coordinate — finding food, warning of danger,
splitting a catch — travels as short pulse-strings of sound: a **call**. Calls
are terse and grammatical because a wasted pulse is energy a Threx does not have.

A model trained on a big pile of Threx calls is small enough to hold in your
head, yet its calls are structured enough that predicting the next pulse needs
three genuinely different kinds of reasoning. That is the whole point: it lets
you *watch* the difference between a model remembering and a model computing.

## The lexicon (25 pulses)

| | pulses |
|---|---|
| **structure** | `⟨` start · `⟩` end · `?` query · `·` hush · `∿` triangulate |
| **who** | `mi` I · `tu` you · `ka` we |
| **verbs** | `wø` seek · `gɪ` found · `ru` give · `na` warn |
| **things** | `fï` fish · `bo` berry · `sto` shell · `lum` glow |
| **places** | `hï` here · `fa` far · `dø` deep |
| **replies** | `ko` yes · `ne` no |
| **currents** | `↑` north · `→` east · `↓` south · `←` west |

## The grammar (call types)

```
ritual       ⟨ who na dø ⟩                       a danger cry: warn → deep
forage       ⟨ PLACE · who wø THING ⟩            seek the thing that lives at PLACE
triangulate  ⟨ ∿ CUR CUR THING ⟩                 prey is where two currents meet
report       ⟨ who gɪ lum ⟩                       the common "found glow"
echo         ⟨ who wø THING ? ⟩ ⟨ who gɪ THING ⟩  answer repeats the queried thing
reply        ⟨ who wø THING ? ⟩ ⟨ who ko|ne ⟩     yes / no
give         ⟨ who ru THING who ⟩                 a handoff
```

Two rules carry real structure:

- **forage** — the THING is fixed by the PLACE: `here→fish`, `far→berry`,
  `deep→shell`. But the place sits **four pulses back**, past the reach of a
  3-gram. So the menu of things is memorisable, but the *pick* needs context a
  short phrasebook can't see. → **selected**.
- **triangulate** — the THING is where two water-currents cross:
  `thing = things[(i + j) mod 4]`, indexing the currents `↑→↓←` as `0 1 2 3` and
  the things `fish berry shell glow` as `0 1 2 3`. This is genuine modular
  arithmetic, and the answer is a thing that appears **nowhere** in the call.
  A handful of current-pairs are **held out of training entirely**, so for those
  the phrasebook has no entry and the model must *compute* the answer by
  generalising the addition. → **composed**.

## The three decisions

| decision | example call | what the model does |
|---|---|---|
| **retrieved** | `⟨ ka na …` → `dø` | recall a fixed idiom; the phrasebook alone already produces it |
| **selected** | `⟨ fa · tu wø …` → `bo` | the menu `{fish,berry,shell}` is retrieved; context picks `berry` |
| **composed** | `⟨ ∿ ↓ ← …` → `bo` | `(2+3) mod 4 = 1` = berry — computed, held out, in no list |

`fieldrun`'s router (`--attribute`) assigns each of these by a definite rule:
**retrieved** = the phrasebook's own top-1 equals the model's pick; **selected**
= the pick is in the phrasebook's candidate set but isn't its top-1; **composed**
= the pick is in no candidate set at all (not an n-gram successor, not a recent
or copied token, not a frequent unigram). The triangulation answer dodges every
one of those because it is computed, not looked up.

## The model

A standard **RoPE / Llama-style** transformer, scaled down until every number
fits on screen: `d=32`, `3` layers, `4` heads (`head_dim=8`), SwiGLU MLP
(`ffn=64`), tied embeddings, context `24`, vocab `25`. RMSNorm, rotary
(`rotate_half`), no biases — exactly the conventions `fieldrun`'s `rope` kernel
mirrors.

## Reproduce

```bash
# 1. corpus  (deterministic; prints token freqs + the held-out COMPOSED cells)
.venv/bin/python sim/threx.py --n-calls 9000 --out sim/data/corpus.json

# 2. train + export  (HF safetensors for fieldrun, weights.json for the browser)
.venv/bin/python sim/train.py --layers 3 --steps 15000

# 3. convert to a fieldrun bundle (f32 = bit-exact, the faithfulness gate)
./target/release/fieldrun convert --model sim/data/hf --arch rope --dtype f32 -o sim/data/threx

# 4. build the phrasebook, validate engine == fieldrun, capture the numbers
.venv/bin/python sim/capture.py        # writes sim/data/store.json + validation.json

# 5. build the page  (inlines weights + store + lexicon into a standalone HTML)
.venv/bin/python sim/build_page.py     # -> site/intuition.html -> docs/intuition.html
```

## Faithfulness

The exported model is validated **top-1 against the fieldrun binary** on the
held-out stream (the same gate every fieldrun tier holds), and the browser
engine (`sim/engine.js`) reproduces the binary's logits to f32 rounding
(`max|Δ| ~ 1e-5`) with the DLA decomposition summing to the logit exactly. So
the live numbers on the page are the model's real ones, not an illustration.
