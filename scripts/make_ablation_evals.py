#!/usr/bin/env python3
"""Build HELD-OUT eval sets + an ablate-eval manifest for the expert-ablation battery.

Eval sets (each ~target tokens, tokenized with the bundle tokenizer), all DISJOINT from the 15k
clustering corpus (sweeps/corpora/monster.json):
  de/fr/es/en : wikimedia/wikipedia 20231101.<lang>, streamed but SKIPPING the first SKIP articles
                (the clustering corpus consumed only the first ~40k chars / first few articles).
  python      : source of the stdlib `argparse` module on this interpreter (never in the code corpus).
  latex       : authored representative LaTeX math (disjoint by construction).

Specs = the six target-expert ANCHORS (single neuron/head each) + random matched-unit controls
(noise floor). Writes ablate/eval/<name>.json and ablate/manifest.json.
"""
import json, os, sys, inspect, argparse as _ap, random

ap = _ap.ArgumentParser(description="Build held-out eval sets + an ablate-eval manifest (anchors + optional full circuit-set).")
ap.add_argument("--partition", help="a `--corpus-decompose --experts-out` JSON; emit FULL circuit-set specs for --experts")
ap.add_argument("--experts", default="8,17,19,0,59,107", help="comma-separated expert ids for the --partition full-set specs")
ap.add_argument("--seed", type=int, default=20260616, help="RNG seed for size-matched control sampling (reproducibility)")
ARGS = ap.parse_args()

from tokenizers import Tokenizer

BUNDLE_TOK = "bundles/Qwen2.5-0.5B-Instruct/Qwen2.5-0.5B-Instruct.tokenizer.json"
OUT = "ablate"; EVDIR = f"{OUT}/eval"
NTOK, SKIP, CHARS = 400, 80, 4000      # tokens/set, wiki articles to skip, char budget before truncation
N_NEURONS, N_HEADS, N_LAYERS = 4864, 14, 24
tok = Tokenizer.from_file(BUNDLE_TOK)
os.makedirs(EVDIR, exist_ok=True)
prov = {}

def write_eval(name, ids, provenance):
    ids = ids[:NTOK]
    json.dump({"holdout_ids": ids}, open(f"{EVDIR}/{name}.json", "w"))
    prov[name] = (len(ids), provenance)
    print(f"  {name}: {len(ids)} tokens — {provenance}", file=sys.stderr)

# 1) multilingual Wikipedia, held out by skipping the first SKIP articles.
try:
    from datasets import load_dataset
    for lang in ["de", "fr", "es", "en"]:
        ds = load_dataset("wikimedia/wikipedia", f"20231101.{lang}", split="train", streaming=True)
        it = iter(ds)
        for _ in range(SKIP):
            next(it, None)
        buf = []
        for ex in it:
            buf.append(ex["text"])
            if sum(len(t) for t in buf) > CHARS:
                break
        ids = tok.encode("\n".join(buf)[:CHARS]).ids
        write_eval(lang, ids, f"wikimedia/wikipedia 20231101.{lang}, articles #{SKIP}+ (disjoint from clustering slice)")
except Exception as e:
    print(f"[warn] wiki eval build failed: {e}", file=sys.stderr)

# 2) Python: real stdlib source, never in the clustering code corpus.
src = inspect.getsource(_ap)[2000:8000]
write_eval("python", tok.encode(src).ids, "Python stdlib `argparse` source (this interpreter), chars 2000-8000")

# 3) LaTeX math: authored, disjoint by construction.
latex = r"""\begin{align}
\nabla \cdot \mathbf{E} &= \frac{\rho}{\varepsilon_0}, \qquad \nabla \times \mathbf{B} = \mu_0 \mathbf{J} + \mu_0\varepsilon_0 \frac{\partial \mathbf{E}}{\partial t} \\
\int_{0}^{\infty} e^{-x^2}\,dx &= \frac{\sqrt{\pi}}{2}, \qquad \sum_{n=1}^{\infty} \frac{1}{n^{2}} = \frac{\pi^{2}}{6}
\end{align}
For a matrix $A \in \mathbb{R}^{n \times n}$ the eigenvalues $\lambda_i$ satisfy $\det(A - \lambda I) = 0$, so that
$\operatorname{tr}(A) = \sum_{i=1}^{n} \lambda_i$ and $\det(A) = \prod_{i=1}^{n} \lambda_i$. The Taylor expansion of a
smooth $f$ about $a$ is $f(x) = \sum_{k=0}^{\infty} \frac{f^{(k)}(a)}{k!}\,(x-a)^k$. By Cauchy--Schwarz,
$\left| \langle u, v \rangle \right| \le \lVert u \rVert \, \lVert v \rVert$, with equality iff $u = \alpha v$.
The Gaussian integral generalizes to $\int_{\mathbb{R}^n} e^{-\tfrac{1}{2} x^\top \Sigma^{-1} x}\,dx = \sqrt{(2\pi)^n \det \Sigma}$.
Define $\hat{f}(\xi) = \int_{-\infty}^{\infty} f(x)\, e^{-2\pi i x \xi}\, dx$; then $\widehat{f * g} = \hat{f}\,\hat{g}$.
"""
write_eval("latex", tok.encode(latex).ids, "authored representative LaTeX math (disjoint by construction)")

# ── manifest: anchors (the six target experts) + random matched-unit controls ───────────────────
anchors = {
    "e8_germanMorph":  {"neurons": [[19, 1273]]},
    "e17_romanceDet":  {"neurons": [[21, 3483]]},
    "e19_spanishVerb": {"neurons": [[20, 35]]},
    "e0_xlingCore":    {"neurons": [[22, 1222]]},
    "e59_codeSyntax":  {"heads":   [[22, 2]]},
    "e107_latex":      {"neurons": [[20, 661]]},
}
used = {(19, 1273), (21, 3483), (20, 35), (22, 1222), (20, 661)}
rng = random.Random(ARGS.seed)
controls = {}
for layer in (19, 20, 21, 22):                       # one random neuron per anchor layer (matched type)
    while True:
        idx = rng.randrange(N_NEURONS)
        if (layer, idx) not in used:
            used.add((layer, idx)); break
    controls[f"ctrl_neuronL{layer}"] = {"neurons": [[layer, idx]]}
controls["ctrl_headL22"] = {"heads": [[22, rng.randrange(N_HEADS)]]}   # matched to e59 (a head)

# ── optional: FULL circuit-set specs from a `--corpus-decompose --experts-out <p>` partition ──────
# The decisive test (ABLATION_FINDINGS next-step #1): ablate each expert's WHOLE 21–300-circuit set,
# not just its anchor. The logits_ablated hook already takes a circuit list, so this is purely a
# manifest concern. Usage: `--partition ablate/partition.json [--experts 8,17,19,0,59,107]`. Each
# `e{id}_full` gets a SIZE-matched random control (same #neurons + #heads, same layers) as a noise floor.
fullset, fullset_ctrl = {}, {}
PART = ARGS.partition
TARGETS = [int(x) for x in ARGS.experts.split(",")]
if PART:
    by_id = {b["id"]: b for b in json.load(open(PART)).get("buckets", [])}
    for eid in TARGETS:
        b = by_id.get(eid)
        if not b:
            print(f"[warn] expert e{eid} not in {PART}", file=sys.stderr); continue
        neurons = [[c["layer"], c["idx"]] for c in b["circuits"] if c["kind"] == "neuron"]
        heads = [[c["layer"], c["idx"]] for c in b["circuits"] if c["kind"] == "head"]
        fullset[f"e{eid}_full"] = {k: v for k, v in (("neurons", neurons), ("heads", heads)) if v}
        taken = {(c["layer"], c["idx"]) for c in b["circuits"]}              # size-matched random control
        cn = []
        for (L, _i) in neurons:
            while True:
                j = rng.randrange(N_NEURONS)
                if (L, j) not in taken: taken.add((L, j)); cn.append([L, j]); break
        ch = []
        for (L, _i) in heads:
            while True:
                j = rng.randrange(N_HEADS)
                if ("h", L, j) not in taken: taken.add(("h", L, j)); ch.append([L, j]); break
        fullset_ctrl[f"ctrl_e{eid}_full"] = {k: v for k, v in (("neurons", cn), ("heads", ch)) if v}
        print(f"  e{eid}_full: {len(neurons)} neurons + {len(heads)} heads (partition size {b.get('size')})", file=sys.stderr)

manifest = {"ctx": 64, "max_pos": 320, "bundle": "Qwen2.5-0.5B-Instruct",
            "evals": {n: f"{EVDIR}/{n}.json" for n, _ in prov.items()},
            "specs": {**anchors, **controls, **fullset, **fullset_ctrl}}
json.dump(manifest, open(f"{OUT}/manifest.json", "w"), indent=2)
print(f"\nmanifest: {OUT}/manifest.json — {len(manifest['specs'])} specs ({len(anchors)} anchors + "
      f"{len(controls)} controls + {len(fullset)} full-sets + {len(fullset_ctrl)} size-matched ctrls), "
      f"{len(manifest['evals'])} eval sets", file=sys.stderr)
if not PART:
    print("[note] anchor-only manifest; pass --partition <experts-out.json> for full circuit-set specs.", file=sys.stderr)
