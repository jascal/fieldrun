#!/usr/bin/env python3
"""Tokenize a diverse set of corpora (NL across languages + code + math) for the bucketing sweeps.

Writes sweeps/corpora/<name>.json ({"holdout_ids": [...]}) using the Qwen2.5 tokenizer, plus pooled combos.
Run from the fieldrun repo root with the lm-sae venv:  ../lm-sae/.venv/bin/python scripts/make_corpora.py
"""
import json, os
from tokenizers import Tokenizer

TOK = "bundles/Qwen2.5-0.5B-Instruct/Qwen2.5-0.5B-Instruct.tokenizer.json"
OUT = "sweeps/corpora"
os.makedirs(OUT, exist_ok=True)
tok = Tokenizer.from_file(TOK)

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

samples = {"german": german * 3, "spanish": spanish * 3, "code": code * 4, "math": math_tex * 4}
ids = {name: tok.encode(text).ids for name, text in samples.items()}

# English from the Qwen holdout (a natural-text slice).
en = json.load(open("../lm-sae/pylm/holdout_Qwen2.5-1.5B.json"))["holdout_ids"]
ids["english"] = en[:1200]

# pooled combos: a small multilingual pool, and a wider all-domain pool.
ids["pooled_small"] = ids["english"][:500] + ids["german"] + ids["code"]
ids["pooled_diverse"] = ids["english"][:600] + ids["german"] + ids["spanish"] + ids["code"] + ids["math"]

for name, i in ids.items():
    json.dump({"model": "Qwen2.5", "holdout_ids": i}, open(f"{OUT}/{name}.json", "w"))
    print(f"{name}: {len(i)} tokens")
