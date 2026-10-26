"""Fixed native-coalition size diagnostic on saved Qwen frames; no model calls."""

import argparse
import hashlib
from itertools import combinations
import json
from pathlib import Path
import numpy as np


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def bodies():
    rows = [
        dict(id=f"pair-{i:02}-{j:02}", sources=[i, j], family="all_pairs")
        for i, j in combinations(range(65), 2)
    ]
    for k in [4, 8, 16, 32, 64]:
        rows.extend(
            [
                dict(id=f"prefix-{k}", sources=list(range(k)), family="prefix"),
                dict(
                    id=f"suffix-{k}", sources=list(range(65 - k, 65)), family="suffix"
                ),
            ]
        )
    rows.append(dict(id="full-65", sources=list(range(65)), family="full_control"))
    assert len(rows) == len({tuple(r["sources"]) for r in rows}) == 2091
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["freeze", "run"])
    for name in ["out", "data", "first"]:
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    out = a.out
    if a.action == "freeze":
        assert not (out / "bodies.json").exists()
        (out / "bodies.json").write_text(json.dumps(bodies(), indent=2) + "\n")
        print("Frozen2091 coalitions, independent of outcomes.")
        return
    assert not (out / "manifest.json").exists(), (
        "Fixed diagnostic already started; do not overwrite."
    )
    audit = json.loads((a.data.parent / "learning_data_verification.json").read_text())
    assert audit["gate"] == "passed" and audit["data_sha256"] == sha(a.data)
    assert json.loads((a.first / "verification.json").read_text())["gate"] == "passed"
    groups = json.loads((out / "bodies.json").read_text())
    assert groups == bodies()
    inputs = [
        a.data,
        a.data.parent / "learning_data_verification.json",
        out / "PREREG.md",
        out / "bodies.json",
        Path(__file__),
        a.first / "domain.json",
        a.first / "verification.json",
    ] + sorted(a.first.glob("batch_*_capture.json"))
    (out / "manifest.json").write_text(
        json.dumps(
            dict(
                sha256={str(p): sha(p) for p in inputs},
                new_model_forwards=0,
                frame_updates=0,
                floor=0.5,
            ),
            indent=2,
        )
        + "\n"
    )
    d = np.load(a.data)
    U = d["initial_U"]
    D = d["native_sources"]
    tokens = d["pool_tokens"].tolist()
    assert U.shape == (2261, 2560) and D.shape == (100, 65, 2560)
    assert np.all(d["initial_bias"] == 0)
    native = []
    for file in sorted(a.first.glob("batch_*_capture.json")):
        native.extend(json.loads(file.read_text())["cases"])
    domain = json.loads((a.first / "domain.json").read_text())
    assert len(native) == 100
    shape = (100, len(groups))
    margins = np.empty(shape)
    slacks = np.empty(shape)
    ranges = np.empty(shape)
    summaries = []
    compact = np.array([len(b["sources"]) <= 8 for b in groups])
    for i, n in enumerate(native):
        case = domain["cases"][i]
        assert n["ids"] == case["ids"] and n["candidate_tokens"] == tokens
        assert n["prediction"] == int(d["native_teacher_tokens"][i])
        t = tokens.index(n["prediction"])
        inc = D[i] @ U.T
        P = np.stack([inc[b["sources"]].sum(0) for b in groups])
        H = np.array(n["candidate_scores"])
        mask = np.arange(len(tokens)) != t
        gap = P[:, t, None] - P[:, mask]
        tail = H[None, :] - P
        m = gap.min(1)
        slack = (gap - np.abs((H[t] - H[mask])[None, :] - gap)).min(1)
        delta = np.ptp(tail, axis=1)
        exact = (m > 0.5) & (slack > 0)
        centered = m > np.maximum(0.5, delta)
        assert np.all(~centered | exact)
        assert np.all(P[exact].argmax(axis=1) == t)
        margins[i] = m
        slacks[i] = slack
        ranges[i] = delta
        exact_ids = np.flatnonzero(exact)
        center_ids = np.flatnonzero(centered)
        summaries.append(
            dict(
                id=case["id"],
                target=n["prediction"],
                truth=case["truth"],
                host_correct=n["prediction"] == domain["token_map"][case["truth"]],
                exact_passes=exact_ids.tolist(),
                centered_passes=center_ids.tolist(),
                compact_exact=bool(np.any(exact & compact)),
                compact_centered=bool(np.any(centered & compact)),
                minimum_tested_exact_size=min(
                    (len(groups[j]["sources"]) for j in exact_ids), default=None
                ),
                memory="none",
            )
        )
        if (i + 1) % 10 == 0:
            print("Scored", i + 1, "/100", flush=True)
    np.savez_compressed(
        out / "body_metrics.npz",
        margin=margins,
        minimum_pairwise_slack=slacks,
        two_delta=ranges,
        gamma=np.where(
            margins > np.maximum(0.5, ranges),
            (margins + np.maximum(0.5, ranges)) / 2,
            np.nan,
        ),
    )
    by_body = []
    for j, b in enumerate(groups):
        by_body.append(
            dict(
                id=b["id"],
                size=len(b["sources"]),
                exact=int(np.sum((margins[:, j] > 0.5) & (slacks[:, j] > 0))),
                centered=int(np.sum(margins[:, j] > np.maximum(0.5, ranges[:, j]))),
            )
        )
    compact_count = sum(r["compact_exact"] for r in summaries)
    result = dict(
        schema="pdc.qwen35-native-coalitions.v1",
        contexts=100,
        bodies=2091,
        body_verdicts=209100,
        compact_exact_contexts=compact_count,
        compact_centered_contexts=sum(r["compact_centered"] for r in summaries),
        compact_gate="local_positive"
        if compact_count >= 20
        else "negative_fixed_family"
        if compact_count == 0
        else "inconclusive",
        summary_by_body=by_body,
        records=summaries,
        new_model_forwards=0,
        frame_updates=0,
        scope="Saved development frames; no fresh confirmation, minimality, pruning, causal replay or impossibility claim.",
    )
    (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ["summary_by_body", "records"]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
