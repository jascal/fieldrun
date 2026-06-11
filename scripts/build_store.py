#!/usr/bin/env python3
"""Build a fieldrun KB store (store.json) + holdout (holdout.json) from a corpus.

The store is the flat-file retrieval KB that Tier A (`--store`), the Phase-8b candidate-set modes
(`--attribute`/`--prune-head`/`--probe*`), and the margin-gated pruned head (`--pruned-head`) read: n-gram successor
tables keyed on token ids, in the schema `src/retrieval.rs` deserializes (quad/tri/bi/uni; the optional grammar
skeleton + closed-class fields are left empty here). This is the simple *corpus* n-gram store — enough to drive the
candidate sets and the pruned head. The research-grade *model-captured* store (the FINDINGS numbers) is built by
`pylm` in the lm-sae repo, which mines the same tables from model rollouts rather than raw text.

Two input modes:
  --text f [f ...]    raw text files, tokenized with --tokenizer <tokenizer.json> (needs `pip install tokenizers`;
                      use the <bundle>.tokenizer.json sitting next to the bundle so ids match the model)
  --ids f [f ...]     pre-tokenized {"holdout_ids": [...]} or bare [...] JSON files (stdlib-only, no deps)

The corpus is split --holdout-frac from the tail into the holdout (never trained on), so
`fieldrun --bundle <stem> --ids holdout.json --store store.json --pruned-head --gate-check 64` measures the gate on
unseen text. Successor lists are ranked by count (the rank-1 successor is the rule's top-1, i.e. what RETRIEVED means).

Example (build from the model's own tokenizer + some text, then calibrate the gate):
  python3 scripts/build_store.py --text corpus.txt \
      --tokenizer ~/.cache/fieldrun/bundles/Qwen2.5-0.5B-Instruct/Qwen2.5-0.5B-Instruct.tokenizer.json \
      -o store.json --holdout holdout.json
  fieldrun --bundle Qwen2.5-0.5B-Instruct --ids holdout.json --store store.json --pruned-head --gate-check 64
"""

import argparse
import collections
import json
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", nargs="+", help="raw text files (needs --tokenizer)")
    src.add_argument("--ids", nargs="+", help="pre-tokenized JSON files (stdlib-only)")
    ap.add_argument("--tokenizer", help="tokenizer.json (the bundle's copy) — required with --text")
    ap.add_argument("-o", "--out", default="store.json", help="store output path (default store.json)")
    ap.add_argument("--holdout", default=None, help="also write the held-out tail as {'holdout_ids': [...]} here")
    ap.add_argument("--holdout-frac", type=float, default=0.15, help="tail fraction held out of the tables (default 0.15)")
    ap.add_argument("--cap", type=int, default=16, help="successors kept per quad/tri key (default 16)")
    ap.add_argument("--cap-bi", type=int, default=32, help="successors kept per bigram key (default 32)")
    ap.add_argument("--cap-uni", type=int, default=512, help="unigram floor size (default 512)")
    args = ap.parse_args()

    if args.text:
        if not args.tokenizer:
            sys.exit("--text needs --tokenizer <tokenizer.json> (the copy next to the bundle, so ids match the model)")
        try:
            from tokenizers import Tokenizer
        except ImportError:
            sys.exit("--text needs the `tokenizers` package (pip install tokenizers); or pre-tokenize and use --ids")
        tok = Tokenizer.from_file(args.tokenizer)
        ids = []
        for f in args.text:
            ids.extend(tok.encode(open(f, encoding="utf-8").read()).ids)
    else:
        ids = []
        for f in args.ids:
            j = json.load(open(f))
            ids.extend(j["holdout_ids"] if isinstance(j, dict) else j)

    if len(ids) < 100:
        sys.exit(f"corpus too small ({len(ids)} tokens) — the tables would be empty")
    cut = len(ids) - int(len(ids) * args.holdout_frac) if args.holdout else len(ids)
    train, hold = ids[:cut], ids[cut:]

    quad = collections.defaultdict(collections.Counter)
    tri = collections.defaultdict(collections.Counter)
    bi = collections.defaultdict(collections.Counter)
    uni = collections.Counter(train)
    for i in range(len(train) - 1):
        bi[str(train[i])][train[i + 1]] += 1
        if i >= 1:
            tri[f"{train[i - 1]},{train[i]}"][train[i + 1]] += 1
        if i >= 2:
            quad[f"{train[i - 2]},{train[i - 1]},{train[i]}"][train[i + 1]] += 1

    def rank(c, cap):
        return [t for t, _ in c.most_common(cap)]

    store = {
        "quad": {k: rank(v, args.cap) for k, v in quad.items()},
        "tri": {k: rank(v, args.cap) for k, v in tri.items()},
        "bi": {k: rank(v, args.cap_bi) for k, v in bi.items()},
        "uni": rank(uni, args.cap_uni),
    }
    json.dump(store, open(args.out, "w"))
    print(f"store: {args.out}  (train {len(train)} tokens · quad {len(store['quad'])} · tri {len(store['tri'])} "
          f"· bi {len(store['bi'])} · uni {len(store['uni'])})")
    if args.holdout:
        json.dump({"holdout_ids": hold}, open(args.holdout, "w"))
        print(f"holdout: {args.holdout}  ({len(hold)} tokens, disjoint from the tables)")


if __name__ == "__main__":
    main()
