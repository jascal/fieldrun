#!/usr/bin/env python3
"""Tokenize a diverse set of corpora (NL across languages + code + math) for the bucketing sweeps.

Writes <out>/<name>.json ({"model": ..., "holdout_ids": [...]}) for a given tokenizer. The tokenizer is a
flag, not a hardcoded path, because a corpus of token IDs is only meaningful for the tokenizer that made it:
Qwen3.5 has a 248070-entry vocabulary against Qwen2.5's 151665, so feeding a Qwen2.5 corpus to a Qwen3.5
bundle silently indexes the wrong embeddings. (Qwen3 and Qwen3-MoE share Qwen2.5's vocabulary, so they reuse
the default corpora as-is.)

    # reproduce the original Qwen2.5 corpora (the defaults)
    ../lm-sae/.venv/bin/python scripts/make_corpora.py

    # Qwen3.5
    ../lm-sae/.venv/bin/python scripts/make_corpora.py \
        --tok /tmp/fr_qwen35_0.8b.tokenizer.json --model Qwen3.5 --out sweeps/corpora/qwen35
"""
import argparse, json, os
from tokenizers import Tokenizer

DEFAULT_TOK = "bundles/Qwen2.5-0.5B-Instruct/Qwen2.5-0.5B-Instruct.tokenizer.json"
# The English slice is stored as IDs, not text. To retokenize it we recover the text with the tokenizer that
# made it. `decode` -> `encode` is exact for this holdout (asserted below), so the text is not lossy.
DEFAULT_SRC_TOK = DEFAULT_TOK
HOLDOUT = "../lm-sae/pylm/holdout_Qwen2.5-1.5B.json"
# How much English goes into `pooled_wide`. Fixed, not a flag: the committed wide corpus must be
# reproducible from the documented default invocation.
WIDE_ENGLISH = 30000

german = (
    "Der alte Mann saß am Ufer des Flusses und beobachtete die Vögel, die über das Wasser flogen. "
    "Er dachte an seine Jugend, als er noch jung und stark war und jeden Morgen zur Arbeit ging. "
    "Die Sonne schien hell am Himmel, und ein leichter Wind bewegte die Blätter der Bäume. "
    "Seine Tochter hatte ihm einen Brief geschrieben, in dem sie erzählte, dass sie bald heiraten würde. "
    "Er freute sich sehr über diese Nachricht, obwohl er wusste, dass er dann allein sein würde. "
    "Am Abend kochte er sich eine Suppe und las das Buch, das ihm sein Freund geschenkt hatte. "
)
spanish = (
    "El joven caminaba por las calles de la ciudad mientras pensaba en su futuro y en sus sueños. "
    "Había decidido estudiar medicina, porque siempre quiso ayudar a las personas que sufrían. "
    "Su madre, que trabajaba en una pequeña tienda, le había enseñado el valor del esfuerzo. "
    "Cada mañana se levantaba temprano y leía sus libros antes de ir a la universidad. "
    "Cuando llegó el verano, viajó al pueblo de sus abuelos para descansar entre las montañas. "
    "Allí comprendió que la felicidad no estaba en las cosas, sino en las personas que amaba. "
)
code = (
    "import math\n"
    "from typing import List, Optional\n\n"
    "def factorial(n: int) -> int:\n"
    "    if n <= 1:\n        return 1\n"
    "    result = 1\n"
    "    for i in range(2, n + 1):\n        result = result * i\n"
    "    return result\n\n"
    "class Vector:\n"
    "    def __init__(self, x: float, y: float) -> None:\n"
    "        self.x = x\n        self.y = y\n\n"
    "    def length(self) -> float:\n"
    "        return math.sqrt(self.x * self.x + self.y * self.y)\n\n"
    "def find_max(items: List[int]) -> Optional[int]:\n"
    "    if not items:\n        return None\n"
    "    best = items[0]\n"
    "    for value in items[1:]:\n        if value > best:\n            best = value\n"
    "    return best\n"
)
math_tex = (
    "Let $f$ be a continuous function on the interval $[a, b]$. "
    "By the fundamental theorem of calculus, $\\int_a^b f'(x)\\,dx = f(b) - f(a)$. "
    "We say a sequence $a_n$ converges to $L$ if for every $\\epsilon > 0$ there exists $N$ such that "
    "$|a_n - L| < \\epsilon$ for all $n > N$. The derivative is defined as the limit "
    "$f'(x) = \\lim_{h \\to 0} \\frac{f(x+h) - f(x)}{h}$. "
    "For a matrix $A$, the eigenvalues $\\lambda$ satisfy $\\det(A - \\lambda I) = 0$. "
    "If $X$ and $Y$ are independent, then $E[XY] = E[X] E[Y]$ and the variance adds. "
)


def english_text(src: Tokenizer, n: int) -> str:
    """Recover the holdout's English as TEXT, so any tokenizer can be retokenized from it.

    Guards the one assumption this whole path rests on: that decode -> encode is lossless for this data.
    If it ever stops being exact, the recovered text is no longer the holdout and every English corpus
    derived from it would be quietly a different corpus.
    """
    ids = json.load(open(HOLDOUT))["holdout_ids"][:n]
    text = src.decode(ids)
    back = src.encode(text).ids
    if back != ids:
        # find the first divergence so a failure is actionable, not just "not equal"
        k = next((i for i, (a, b) in enumerate(zip(ids, back)) if a != b), min(len(ids), len(back)))
        raise SystemExit(
            f"detokenize round-trip is NOT exact for the holdout (first divergence at {k}: "
            f"{len(ids)} ids -> {len(back)} ids). Refusing to emit a corpus that is not the holdout."
        )
    return text


def interleave(sources: dict, chunk: int = 80) -> list:
    """Round-robin the sources in `chunk`-token runs so any window spans many domains, skipping a source
    once it is spent. Same shape as make_monster_corpus.py, just shared.

    Interleaved rather than concatenated so the oracle's best position is not always in whichever domain
    happens to sit at the front.
    """
    idx = {k: 0 for k in sources}
    keys = list(sources)
    out: list = []
    while any(idx[k] < len(sources[k]) for k in keys):
        for k in keys:
            if idx[k] < len(sources[k]):
                out += sources[k][idx[k]:idx[k] + chunk]
                idx[k] += chunk
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tok", default=DEFAULT_TOK, help="TARGET tokenizer.json (defaults to Qwen2.5-0.5B)")
    ap.add_argument("--src-tok", default=DEFAULT_SRC_TOK, help="tokenizer that produced the English holdout ids")
    ap.add_argument("--out", default="sweeps/corpora", help="output directory")
    ap.add_argument("--model", default="Qwen2.5", help="model tag recorded in each json")
    ap.add_argument("--english", type=int, default=1200,
                    help="how much of the English holdout to use (default 1200, which reproduces the "
                         "committed corpora byte-for-byte). The holdout holds ~30k; raise it for a wider pool.")
    ap.add_argument("--no-wide", action="store_true",
                    help="skip pooled_wide (the large interleaved corpus)")
    args = ap.parse_args()

    tok = Tokenizer.from_file(args.tok)
    src = Tokenizer.from_file(args.src_tok)
    vocab = tok.get_vocab_size()
    os.makedirs(args.out, exist_ok=True)

    samples = {"german": german * 3, "spanish": spanish * 3, "code": code * 4, "math": math_tex * 4}
    ids = {name: tok.encode(text).ids for name, text in samples.items()}
    ids["english"] = tok.encode(english_text(src, args.english)).ids

    # pooled combos: a small multilingual pool, and a wider all-domain pool.
    ids["pooled_small"] = ids["english"][:500] + ids["german"] + ids["code"]
    ids["pooled_diverse"] = ids["english"][:600] + ids["german"] + ids["spanish"] + ids["code"] + ids["math"]
    if not args.no_wide:
        # Named `pooled_wide`, NOT `pooled_big`: that file is a committed ad-hoc artifact (added in 71cd5a8
        # with no generator) and regenerating it under the same name would silently replace a corpus other
        # commands already reference with different content.
        #
        # Deliberately built from the FULL holdout, not from `--english`: its size must not depend on the
        # flag, or the same invocation writes a different `pooled_wide` depending on unrelated arguments and
        # the committed file stops being reproducible from the documented default.
        wide = {"english": tok.encode(english_text(src, WIDE_ENGLISH)).ids}
        wide.update({k: ids[k] for k in ("german", "spanish", "code", "math")})
        ids["pooled_wide"] = interleave(wide)

    for name, i in ids.items():
        hi = max(i)
        if hi >= vocab:
            # A token id past the end of the vocabulary is an out-of-range embedding lookup: a wrong number,
            # not a crash. Fail here, where the cause is obvious, rather than deep inside a forward pass.
            raise SystemExit(
                f"{name}: max token id {hi} >= vocab {vocab} for {args.tok}. "
                f"This corpus was tokenized with the wrong tokenizer."
            )
        json.dump({"model": args.model, "holdout_ids": i}, open(f"{args.out}/{name}.json", "w"))
        print(f"{name}: {len(i)} tokens (max id {hi} / vocab {vocab})")


if __name__ == "__main__":
    main()