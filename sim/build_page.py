#!/usr/bin/env python3
"""Assemble the standalone intuition page.

Inlines the engine (sim/engine.js), the model weights, the lexicon, the
phrasebook, the validation numbers, and a sample of the corpus into the template
(sim/page_template.html), producing a single self-contained file with no network
dependency — written to BOTH site/intuition.html (the source the Pages workflow
copies) and docs/intuition.html (so a local preview works immediately)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
REPO = ROOT.parent


def load(name):
    return json.loads((DATA / name).read_text())


def corpus_sample():
    """A readable slice of the corpus: the first ~60 calls, split into calls."""
    ids = load("corpus.json")["ids"]
    calls, cur = [], []
    for t in ids[:600]:
        if t == 0 and cur:        # BOS starts a new call
            calls.append(cur); cur = []
        cur.append(t)
    if cur:
        calls.append(cur)
    return calls[:60]


def main():
    data = {
        "weights": load("weights.json"),
        "lexicon": load("lexicon.json"),
        "store": load("store.json"),
        "validation": load("validation.json") if (DATA / "validation.json").exists() else {},
        "corpus": corpus_sample(),
    }
    engine = (ROOT / "engine.js").read_text()
    # strip the engine's UMD tail (we call it directly as window.ThrexEngine)
    template = (ROOT / "page_template.html").read_text()
    page = (template
            .replace("/*__ENGINE__*/", engine)
            .replace("/*__DATA__*/", json.dumps(data, ensure_ascii=False)))
    for out in [REPO / "site" / "intuition.html", REPO / "docs" / "intuition.html"]:
        out.parent.mkdir(exist_ok=True)
        out.write_text(page)
        print(f"wrote {out}  ({len(page)//1024} KB)")


if __name__ == "__main__":
    main()
