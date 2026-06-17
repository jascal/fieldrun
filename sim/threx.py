#!/usr/bin/env python3
"""The Threx — a tiny alien signal-language, and a corpus generator for it.

The Threx are blind deep-water foragers who cannot see one another. They
coordinate by pulsing short call-strings: who is calling, what they are doing,
about which thing, where, and — when they triangulate prey — from which two
water-currents. The language is small enough that a whole language model trained
on it is unrealistically tiny, yet it deliberately contains three DIFFERENT
kinds of next-token decision, each genuinely learned from the corpus:

  RETRIEVED  the danger ritual `na -> dø` (warn -> deep). A FIXED idiom: the
             deciding token sits one beat back, so the simplest phrasebook nails
             it. Pure recall — the model just memorised a constant.

  SELECTED   foraging `⟨ PLACE · who wø THING ⟩`. The THING is gated by the
             PLACE, which sits FOUR beats back — past a 3-gram window. The
             phrasebook offers the menu {fish, berry, shell} as a shortlist but
             cannot see the place, so it cannot pick. The model SELECTS within
             the retrieved menu using context the phrasebook can't reach.

  COMPOSED   triangulation `⟨ ∿ Bi Bj THING ⟩`. The Threx locate prey where two
             water-currents meet: THING = things[(i + j) mod 4]. This is real
             arithmetic — the answer is NOT one of the inputs and appears nowhere
             in the call, so no copy / recency / n-gram rule proposes it. One
             current-pair is HELD OUT of training, so the phrasebook has no entry
             for it; the model must GENERALISE the addition and compute the
             answer. That is the irreducible, computed slice.

Dependency-free (stdlib + a seeded RNG). Emits a flat token-id stream in the
{"holdout_ids":[...]} shape fieldrun reads, plus the three canonical example
contexts the visualization steps through. See sim/README.md.
"""
import argparse
import itertools
import json
import os
import random

# ---- the lexicon: id -> (glyph, gloss, class) -----------------------------
TOKENS = [
    ("⟨",  "call-start",   "struct"),   # 0  BOS — the click that opens a call
    ("⟩",  "call-end",     "struct"),   # 1  EOS — closes it
    ("?",  "query",        "struct"),   # 2  marks a question
    ("mi", "I",            "who"),      # 3
    ("tu", "you",          "who"),      # 4
    ("ka", "we",           "who"),      # 5
    ("wø", "seek",         "verb"),     # 6
    ("gɪ", "found",        "verb"),     # 7
    ("ru", "give",         "verb"),     # 8
    ("na", "warn!",        "verb"),     # 9
    ("fï", "fish",         "thing"),    # 10  result 0
    ("bo", "berry",        "thing"),    # 11  result 1
    ("sto","shell",        "thing"),    # 12  result 2
    ("lum","glow",         "thing"),    # 13  result 3
    ("hï", "here",         "place"),    # 14
    ("fa", "far",          "place"),    # 15
    ("dø", "deep",         "place"),    # 16
    ("ko", "yes",          "reply"),    # 17
    ("ne", "no",           "reply"),    # 18
    ("·",  "hush",         "struct"),   # 19  a spacer particle
    ("∿",  "triangulate",  "struct"),   # 20  opens a triangulation call
    ("↑",  "north-current","bearing"),  # 21  current strength 0
    ("↗",  "ne-current",   "bearing"),  # 22  current strength 1
    ("→",  "east-current", "bearing"),  # 23  current strength 2
    ("↘",  "se-current",   "bearing"),  # 24  current strength 3
    ("↓",  "south-current","bearing"),  # 25  current strength 4
    ("kel","kelp",         "thing"),    # 26  result rank 4
    ("vor","worm",         "thing"),    # 27  result rank 5
    ("sib","shrimp",       "thing"),    # 28  result rank 6
    ("nok","pebble",       "thing"),    # 29  result rank 7
    ("eel","eel",          "thing"),    # 30  result rank 8
]
GLYPH = [t[0] for t in TOKENS]
GLOSS = [t[1] for t in TOKENS]
CLS = [t[2] for t in TOKENS]
ID = {g: i for i, g in enumerate(GLYPH)}
VOCAB = len(TOKENS)  # 25

BOS, EOS, Q, MI, TU, KA = 0, 1, 2, 3, 4, 5
WØ, GI, RU, NA = 6, 7, 8, 9
FI, BO, STO, LUM = 10, 11, 12, 13
HI, FA, DØ = 14, 15, 16
KO, NE, HUSH, TRI = 17, 18, 19, 20
B0, B1, B2, B3, B4 = 21, 22, 23, 24, 25
KEL, VOR, SIB, NOK, EEL = 26, 27, 28, 29, 30

WHO = [MI, TU, KA]
THINGS = [FI, BO, STO, LUM, KEL, VOR, SIB, NOK, EEL]  # 9 prey; also result-ranks 0..8
BEARINGS = [B0, B1, B2, B3, B4]             # 5 currents; their STRENGTHS are 0..4
STRENGTH = {b: i for i, b in enumerate(BEARINGS)}
PLACE_OBJ = {HI: FI, FA: BO, DØ: STO}       # the SELECTED gate: place -> thing
PLACES = list(PLACE_OBJ.keys())

# The model trains on ALL 25 triangulation cells, so on any current-pair it
# answers correctly (a clean live demo). COMPOSED does not come from holding cells
# out — it comes from DISTANCE: the two currents sit beyond the phrasebook's
# n-gram window, so the phrasebook cannot represent the sum and can only offer its
# marginal favourites. For the rarer prey (the edge sums, things[0]=fish and
# things[8]=eel, each reachable by a single current-pair) the model's computed
# answer is so far down the phrasebook's marginal list that it falls outside the
# candidate set entirely → genuinely COMPOSED, while common prey land SELECTED.
HELD_CELLS = []


def render(ids):
    return " ".join(GLYPH[i] for i in ids)


def gloss(ids):
    return " ".join(f"{GLYPH[i]}({GLOSS[i]})" for i in ids)


# ---- call templates --------------------------------------------------------
def ritual(rng):
    """RETRIEVED: the danger cry. `na` is ALWAYS followed by `dø`."""
    return [BOS, rng.choice(WHO), NA, DØ, EOS]


def forage(rng):
    """SELECTED: `⟨ PLACE · who wø THING ⟩`. PLACE (4 back) gates THING."""
    place = rng.choice(PLACES)
    return [BOS, place, HUSH, rng.choice(WHO), WØ, PLACE_OBJ[place], EOS]


def triangulate(rng, allow_held=False):
    """COMPOSED: `⟨ ∿ Bi Bj · · gɪ THING ⟩`, THING = things[i + j] (the prey at
    the rank that is the SUM of the two current strengths). The two currents sit
    FOUR–FIVE pulses before the answer — beyond the reach of the phrasebook's
    short n-gram window — so the phrasebook cannot represent the rule and falls
    back to its favourite guess. The model must reach back across the gap, read
    both currents, and add them: a genuine long-range, two-operand computation no
    n-gram, copy, or recency rule can reproduce. For the rarer prey the computed
    answer lands outside the phrasebook's candidate set entirely → COMPOSED."""
    K = len(BEARINGS)
    while True:
        i, j = rng.randrange(K), rng.randrange(K)
        if allow_held or (i, j) not in HELD_CELLS:
            break
    return [BOS, TRI, BEARINGS[i], BEARINGS[j], HUSH, HUSH, GI, THINGS[i + j], EOS]


def report(rng):
    """Filler: the common 'found glow' report."""
    return [BOS, rng.choice(WHO), GI, LUM, EOS]


def echo(rng):
    """Filler (a SELECTED flavour): a query then an answer that copies the
    queried thing — the answer is visibly present, so recency covers it."""
    obj = rng.choice(THINGS)
    return [BOS, rng.choice(WHO), WØ, obj, Q, EOS, BOS, rng.choice(WHO), GI, obj, EOS]


def reply(rng):
    """Filler: a seek-question answered yes/no (uses ko/ne)."""
    obj = rng.choice(THINGS)
    return [BOS, rng.choice(WHO), WØ, obj, Q, EOS, BOS, rng.choice(WHO), rng.choice([KO, NE]), EOS]


def give(rng):
    """Filler: a handoff `⟨ who ru THING who ⟩` (uses ru)."""
    a = rng.choice(WHO)
    b = rng.choice([w for w in WHO if w != a])
    return [BOS, a, RU, rng.choice(THINGS), b, EOS]


CALLS = {"ritual": ritual, "forage": forage, "triangulate": triangulate,
         "report": report, "echo": echo, "reply": reply, "give": give}


def generate(n_calls=8000, seed=0,
             mix=(0.12, 0.15, 0.42, 0.05, 0.10, 0.08, 0.08)):
    """Emit a flat id stream. mix weights the call types in CALLS order."""
    rng = random.Random(seed)
    kinds = list(CALLS)
    stream = []
    while stream.count(BOS) < n_calls:
        k = rng.choices(kinds, weights=mix)[0]
        stream += CALLS[k](rng)
    return stream


# ---- the three canonical example contexts ---------------------------------
def examples():
    return [
        {
            "key": "retrieved", "title": "The danger ritual",
            "prefix": [BOS, KA, NA], "expect": DØ,
            "blurb": "We sound the alarm. After ‘warn’, the only thing a Threx "
                     "ever says is ‘deep’. Pure ritual — the simplest phrasebook "
                     "already produces it, so the model adds nothing. Retrieved.",
        },
        {
            "key": "selected", "title": "Choosing from the menu",
            "prefix": [BOS, FA, HUSH, TU, WØ], "expect": BO,
            "blurb": "A hunt announced out FAR. ‘Seek’ has a known menu "
                     "{fish, berry, shell}; which one depends on ‘far’ — four "
                     "beats back, past the phrasebook’s short memory. The menu is "
                     "retrieved; the pick is selected from it by context.",
        },
        {
            "key": "composed", "title": "Triangulating the prey",
            # ⟨ ∿ ↓ ↓ · · gɪ …  — two strong south currents (strength 4 each); the
            # prey is at rank 4+4 = 8 = eel, the rarest find. The currents sit five
            # pulses back, past the phrasebook's window, and eel is the one prey
            # the phrasebook drops from its 'found' list, so no n-gram, copy, or
            # recency rule offers it here.
            "prefix": [BOS, TRI, B4, B4, HUSH, HUSH, GI], "expect": EEL,
            "blurb": "Two strong south currents — strength 4 and 4. The prey lies "
                     "where they cross, at rank 4 + 4 = 8: the EEL, the rarest "
                     "catch. The currents sit five pulses back, beyond the "
                     "phrasebook’s short memory, and eel is the one prey it drops "
                     "from its ‘found’ list, so the phrasebook has no entry for it "
                     "here. The model reaches back, reads both currents, and adds "
                     "them. Composed.",
        },
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-calls", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="sim/data/corpus.json")
    ap.add_argument("--holdout-frac", type=float, default=0.12)
    args = ap.parse_args()

    ids = generate(args.n_calls, args.seed)
    cut = int(len(ids) * (1 - args.holdout_frac))
    while cut < len(ids) and ids[cut] != BOS:
        cut += 1
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"ids": ids[:cut]}, open(args.out, "w"))
    json.dump({"holdout_ids": ids[cut:]}, open(args.out.replace("corpus", "holdout"), "w"))
    # the full triangulation table travels with the data (for the viz + audits)
    K = len(BEARINGS)
    tri_table = [[THINGS[i + j] for j in range(K)] for i in range(K)]
    json.dump({"tokens": TOKENS, "vocab": VOCAB, "examples": examples(),
               "things": THINGS, "bearings": BEARINGS, "places": PLACES,
               "strength": {str(k): v for k, v in STRENGTH.items()},
               "place_obj": {str(k): v for k, v in PLACE_OBJ.items()},
               "tri_table": tri_table, "held_cells": HELD_CELLS},
              open(args.out.replace("corpus", "lexicon"), "w"), ensure_ascii=False)
    from collections import Counter
    c = Counter(ids)
    print(f"corpus: {len(ids)} tokens ({args.n_calls} calls) · train {cut} · "
          f"holdout {len(ids)-cut} · vocab {VOCAB}")
    print("token freq:", {GLYPH[i]: c[i] for i in range(VOCAB)})
    print("held-out triangulation cells (the COMPOSED instances):")
    for i, j in HELD_CELLS:
        print(f"  ∿ {GLYPH[BEARINGS[i]]} {GLYPH[BEARINGS[j]]} → {GLYPH[THINGS[i+j]]} "
              f"({i}+{j} = {i+j})")
    for e in examples():
        print(f"  [{e['key']:9}] {render(e['prefix'])} → expect {GLYPH[e['expect']]}")


if __name__ == "__main__":
    main()
