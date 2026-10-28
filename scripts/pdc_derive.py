"""Finite PDC evaluator. JSON artifact boundary; no sibling-repository imports.

Implements i-orca PDC_Derive.thy's finite-body semantics using float arithmetic.
An absent score is null, never zero. Equal-score bodies retain multiplicity.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path


class Readout(str, Enum):
    VITERBI = "viterbi"
    MARGINAL = "marginal"


@dataclass(frozen=True)
class Body:
    id: str
    score: float


@dataclass(frozen=True)
class QueryResult:
    readout: Readout
    temperature: float
    scores: dict[str, float | None]
    winners: list[str]
    winning_bodies: dict[str, list[str]]
    margin: float | None
    status: str


def evaluate(table: dict[str, list[Body]], temperature: float) -> dict:
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if not table or any(not isinstance(a, str) or not a for a in table):
        raise ValueError("a nonempty named rival set is required")
    for bodies in table.values():
        if any(not b.id or not math.isfinite(b.score) for b in bodies):
            raise ValueError("body IDs and finite brackets are required")
        if len({b.id for b in bodies}) != len(bodies):
            raise ValueError("body IDs must be unique within an atom")
    vit = {a: max((b.score for b in bs), default=None) for a, bs in table.items()}
    mar = {}
    for a, bs in table.items():
        m = vit[a]
        if m is None:
            mar[a] = None
        else:
            # Shift before exponentiating; a singleton returns its exact input.
            mar[a] = m + temperature * math.log(math.fsum(
                math.exp((b.score - m) / temperature) for b in bs))
            if not math.isfinite(mar[a]):
                raise ValueError("marginal score exceeds finite arithmetic range")
    results = []
    for readout, scores in [(Readout.VITERBI, vit), (Readout.MARGINAL, mar)]:
        present = [s for s in scores.values() if s is not None]
        top = max(present, default=None)
        winners = sorted(a for a, s in scores.items() if s is not None and s == top)
        competitors = sorted(present, reverse=True)
        margin = competitors[0] - competitors[1] if len(competitors) >= 2 else None
        if margin is not None and not math.isfinite(margin):
            raise ValueError("margin exceeds finite arithmetic range")
        result = QueryResult(
            readout, temperature, scores, winners,
            {a: sorted(b.id for b in table[a] if b.score == vit[a]) for a in winners},
            margin, "selected" if len(winners) == 1 else "tie" if winners else "absent",
        )
        results.append(asdict(result))
    return {
        "schema": "pdc.finite-derive.v1",
        "semantics": "PDC_Derive.thy; float implementation, not a kernel certificate",
        "rivals": sorted(table),
        "bodies": {a: [asdict(b) for b in bs] for a, bs in table.items()},
        "queries": results,
        "fragile": results[0]["winners"] != results[1]["winners"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("table", type=Path, help="JSON atom -> list of {id, score}")
    parser.add_argument("--temperature", type=float, required=True)
    args = parser.parse_args()
    raw = json.loads(args.table.read_text())
    result = evaluate({a: [Body(**b) for b in bs] for a, bs in raw.items()}, args.temperature)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
