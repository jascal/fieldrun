"""Frozen fixed-frame singleton diagnosis, not another model or policy run."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from pdc_qwen35_engineering_check import read_frame


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["artifacts", "bundle", "out"]:
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    root = a.artifacts
    out = a.out
    assert json.loads((root / "verification.json").read_text())["gate"] == "passed"
    assert not (out / "manifest.json").exists(), (
        "diagnostic already frozen; no overwrites"
    )
    paths = [
        root / n
        for n in ["domain.json", "probe.json", "run_manifest.json", "verification.json"]
    ]
    paths += sorted(root.glob("batch_*_capture.json"))
    paths += [
        Path(str(a.bundle) + ".fieldrun.json"),
        Path(str(a.bundle) + ".fieldrun.bin"),
        out / "PREREG.md",
        Path(__file__),
    ]
    manifest = dict(
        schema="pdc.qwen35-singleton-diagnostic.v1",
        scope="saved-data diagnosis; no fresh heldout confirmation",
        sha256={str(p): sha(p) for p in paths},
        gamma_floor=0.5,
        new_model_forwards=0,
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    domain = json.loads((root / "domain.json").read_text())
    tokens = json.loads((root / "probe.json").read_text())["tokens"]
    m, U, gain = read_frame(a.bundle, tokens)
    assert m["config"][0] == 32 and U.shape == (2261, 2560)
    source_names = ["embed"] + [
        n for layer in range(32) for n in [f"L{layer}.attn", f"L{layer}.mlp"]
    ]
    records = []
    case_index = 0
    for path in sorted(root.glob("batch_*_capture.json")):
        for native in json.loads(path.read_text())["cases"]:
            case = domain["cases"][case_index]
            case_index += 1
            assert native["ids"] == case["ids"] and native["candidate_tokens"] == tokens
            assert [s["id"] for s in native["sources"]] == source_names
            target_token = native["prediction"]
            if target_token not in tokens:
                records.append(
                    dict(
                        id=case["id"],
                        split=case["split"],
                        status="target_outside_coverage",
                    )
                )
                continue
            t = tokens.index(target_token)
            raw = np.array([s["write"] for s in native["sources"]])
            residual = np.array(native["final_residual"])
            partial = (
                raw * gain / np.sqrt(np.mean(residual * residual) + m["config_f"][1])
            ) @ U.T
            host = np.array(native["candidate_scores"])
            mask = np.arange(len(tokens)) != t
            gaps = partial[:, t, None] - partial[:, mask]
            host_gap = host[t] - host[mask]
            slack = gaps - np.abs(host_gap[None, :] - gaps)
            margin = gaps.min(1)
            min_slack = slack.min(1)
            tail = host[None, :] - partial
            delta2 = np.ptp(tail, axis=1)
            exact = (margin > 0.5) & (min_slack > 0)
            centered = margin > np.maximum(0.5, delta2)
            assert np.all(~centered | exact)
            strict = margin > 0
            for j in np.flatnonzero(exact):
                assert int(partial[j].argmax()) == t
            truth = str(case["a"] + case["b"])
            assert truth == case["truth"]
            body_rows = []
            for j, name in enumerate(source_names):
                row = dict(
                    source=name,
                    margin=float(margin[j]),
                    minimum_pairwise_slack=float(min_slack[j]),
                    two_delta=float(delta2[j]),
                    host_target_match=bool(strict[j]),
                    exact_pairwise_admitted=bool(exact[j]),
                    centered_admitted=bool(centered[j]),
                )
                if centered[j]:
                    gamma = float((margin[j] + max(0.5, delta2[j])) / 2)
                    assert margin[j] > gamma > delta2[j]
                    row["gamma"] = gamma
                body_rows.append(row)
            exact_indices = np.flatnonzero(exact).tolist()
            centered_indices = np.flatnonzero(centered).tolist()
            records.append(
                dict(
                    id=case["id"],
                    split=case["split"],
                    status="scored",
                    target_token=target_token,
                    target_atom=next(
                        (
                            k
                            for k, v in domain["token_map"].items()
                            if v == target_token
                        ),
                        None,
                    ),
                    host_correct=target_token == domain["token_map"][truth],
                    memory="none",
                    any_raw_host_match=bool(strict.any()),
                    sources=body_rows,
                    exact_witness=source_names[exact_indices[0]]
                    if exact_indices
                    else None,
                    centered_witness=source_names[centered_indices[0]]
                    if centered_indices
                    else None,
                    exact_sources=[source_names[j] for j in exact_indices],
                    centered_sources=[source_names[j] for j in centered_indices],
                )
            )
    assert case_index == len(records) == 100
    summaries = {}
    for split in ["development", "heldout", "all"]:
        rows = [r for r in records if split == "all" or r["split"] == split]
        summary = dict(
            contexts=len(rows),
            raw_singleton_host_matches=sum(
                r.get("any_raw_host_match", False) for r in rows
            ),
        )
        for name in ["exact", "centered"]:
            admitted = [r for r in rows if r.get(name + "_witness") is not None]
            summary[name] = dict(
                admitted=len(admitted),
                rate=len(admitted) / len(rows),
                covered_host_agreement=1.0 if admitted else None,
                theory_affirmed=sum(r["host_correct"] for r in admitted),
                witnesses=dict(Counter(r[name + "_witness"] for r in admitted)),
            )
        summaries[split] = summary
    n = summaries["all"]["exact"]["admitted"]
    result = dict(
        schema="pdc.qwen35-singleton-result.v1",
        summary=summaries,
        classification="local_positive"
        if n >= 10
        else "negative_fixed_singletons"
        if n == 0
        else "inconclusive",
        source_verdicts=6500,
        new_model_forwards=0,
        frame_updates=0,
        records=records,
        claims=dict(
            fresh_confirmation=False,
            causal_necessity=False,
            impossibility=False,
            memory_transfer=False,
        ),
    )
    (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {k: result[k] for k in ["summary", "classification", "source_verdicts"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
