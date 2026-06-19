#!/usr/bin/env python3
"""Position-stratified demand-closure checker for an LO3a-exported Datalog program.

PROVABLE_OPT PO-T1, real-bundle rung (checker level). The i-orca corpus
(`i-orca/examples/provable_opt`) kernel-proves, abstractly, that restricting a
*demand-closed* dead stratum to `lastpos` preserves the `decide` query
(`demand_restrict_query`, `lastpos_transform_lossless_and_strict`). This tool
discharges that theorem's premise on a *real* emitted `Pi`: it statically
certifies which computed relations are demanded ONLY at `lastpos`, so dropping
them at every other position is lossless for `decide`.

It is a SOUND over-approximation of demand: a relation is reported droppable only
if every path by which the query reads it is pinned to `lastpos`. Anything read at
a free key position (e.g. attention reading keys/values at all positions) is
conservatively marked ALL and never reported droppable.

Honest scope: this is a static checker, not an Isabelle proof of the specific
program. It certifies that the real `Pi` MEETS the premise of the kernel theorem;
the kernel theorem supplies the "...therefore the restriction is lossless".

The input `.dl` is generated, not committed (lo3a tracks only scripts): produce it
with `python3 mint_and_emit.py` (tiny minted bundle) or
`fieldrun export --logic-whole --out whole_base.dl` on a real rope bundle.

Usage:
    python3 mint_and_emit.py                                   # -> whole_base.dl
    python3 demand_closure.py whole_base.dl [--json cert.json] [--expect xf,ssf,x2]
"""
from __future__ import annotations

import argparse
import json
import re
import sys

# demand classes (a join-semilattice: NONE < LASTPOS < ALL)
NONE, LASTPOS, ALL = 0, 1, 2
CLS = {NONE: "none", LASTPOS: "lastpos", ALL: "all"}

# the query predicates (Soufflé .output relations of an LO3a program)
QUERY = ("decide", "logit")


def strip_comments(text: str) -> str:
    return "\n".join(line.split("//", 1)[0] for line in text.splitlines())


def parse_decls(text: str) -> dict[str, bool]:
    """relation -> is it position-keyed (first declared arg named `pos`)."""
    posrel: dict[str, bool] = {}
    for m in re.finditer(r"^\.decl\s+(\w+)\s*\(([^)]*)\)", text, re.MULTILINE):
        name, args = m.group(1), m.group(2)
        first = args.split(",", 1)[0].strip()
        first_name = first.split(":", 1)[0].strip()
        posrel[name] = (first_name == "pos")
    return posrel


def first_arg_occurrences(body: str, name: str) -> list[str]:
    """First argument of each occurrence of `name(` in `body` (a simple var/num/_)."""
    pat = r"(?<![A-Za-z0-9_])" + re.escape(name) + r"\s*\(\s*([A-Za-z_]\w*|\d+|_)"
    return [m.group(1) for m in re.finditer(pat, body)]


class Rule:
    __slots__ = ("head", "head_pos", "head_is_pos", "body", "lastpos_vars")

    def __init__(self, head, head_pos, head_is_pos, body, lastpos_vars):
        self.head = head
        self.head_pos = head_pos
        self.head_is_pos = head_is_pos
        self.body = body
        self.lastpos_vars = lastpos_vars


def parse_rules(text: str, posrel: dict[str, bool]) -> list[Rule]:
    rules: list[Rule] = []
    # LO3a emits single-line rules/facts; directives (.decl/.output/...) start with '.'.
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(".") or ":-" not in line:
            continue
        head_s, body = line.split(":-", 1)
        hm = re.search(r"(\w+)\s*\(\s*([A-Za-z_]\w*|\d+|_)", head_s)
        if not hm:
            continue
        head, head_pos = hm.group(1), hm.group(2)
        head_is_pos = posrel.get(head, False)
        lastpos_vars = set(re.findall(r"(?<![A-Za-z0-9_])lastpos\(\s*(\w+)\s*\)", body))
        rules.append(Rule(head, head_pos, head_is_pos, body, lastpos_vars))
    return rules


def analyse(text: str):
    text = strip_comments(text)
    posrel = parse_decls(text)
    rules = parse_rules(text, posrel)
    idb = {r.head for r in rules}                       # heads = derived relations
    known = set(posrel) | idb

    cls: dict[str, int] = {r: NONE for r in known}      # pos-class for pos-keyed rels
    active: set[str] = set(QUERY)                        # demanded relations

    changed = True
    while changed:
        changed = False
        for r in rules:
            head_active = r.head in active or (posrel.get(r.head) and cls.get(r.head, 0) >= LASTPOS)
            if not head_active:
                continue
            for name in known:
                for arg in first_arg_occurrences(r.body, name):
                    if posrel.get(name, False):
                        if arg in r.lastpos_vars:
                            contrib = LASTPOS
                        elif r.head_is_pos and arg == r.head_pos:
                            contrib = cls[r.head]
                        else:
                            contrib = ALL
                        if contrib > cls[name]:
                            cls[name] = contrib
                            changed = True
                    if name not in active:
                        active.add(name)
                        changed = True

    droppable = sorted(n for n in idb if posrel.get(n, False) and cls[n] == LASTPOS)
    live_pos = sorted(n for n in idb if posrel.get(n, False) and cls[n] == ALL)
    return {
        "droppable_off_lastpos": droppable,
        "live_all_positions": live_pos,
        "demanded_relations": len(active),
        "idb_relations": len(idb),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dl", help="LO3a-emitted Datalog program (.dl)")
    ap.add_argument("--json", help="write a certificate JSON here")
    ap.add_argument("--expect", default="",
                    help="comma-separated relations that MUST be droppable (regression gate)")
    args = ap.parse_args()

    with open(args.dl, encoding="utf-8") as f:
        res = analyse(f.read())

    drop = res["droppable_off_lastpos"]
    print(f"demand-closure certificate for {args.dl}")
    print(f"  IDB relations: {res['idb_relations']}   demanded: {res['demanded_relations']}")
    print(f"  DROPPABLE off-lastpos (demanded only at lastpos): {len(drop)}")
    print(f"    {', '.join(drop)}")
    print(f"  live at all positions (feed attention; NOT droppable): {len(res['live_all_positions'])}")
    print("  => restricting the droppable relations to lastpos preserves `decide`")
    print("     (discharges the premise of i-orca lastpos_transform_lossless_and_strict")
    print("      / demand_restrict_query on this real Pi).")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(f"  wrote certificate -> {args.json}")

    expect = [e for e in args.expect.split(",") if e]
    missing = [e for e in expect if e not in drop]
    if missing:
        print(f"  GATE FAILED: expected droppable but not certified: {missing}", file=sys.stderr)
        return 1
    if expect:
        print(f"  GATE OK: {expect} all certified droppable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
