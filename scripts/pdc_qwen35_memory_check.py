"""Independent Qwen two-hop token, numeric, certificate and theory audit."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from pdc_qwen35_engineering_check import read_frame


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["artifacts", "first", "diagnostic", "bundle", "model"]:
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    root = a.artifacts
    repo = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "run_manifest.json").read_text())
    for name, wanted in manifest["sha256"].items():
        original = Path(name)
        if name.startswith("experiments/pdc_qwen35_memory/"):
            local = root / original.name
        elif name.startswith("experiments/pdc_qwen35_4b/"):
            local = a.first / original.name
        elif name.startswith("experiments/pdc_qwen35_singletons/"):
            local = a.diagnostic / original.name
        elif name.startswith("scripts/") or name.startswith("target/"):
            local = repo / original
        elif name.endswith(".fieldrun.json"):
            local = Path(str(a.bundle) + ".fieldrun.json")
        elif name.endswith(".fieldrun.bin"):
            local = Path(str(a.bundle) + ".fieldrun.bin")
        else:
            local = a.model / original.name
        h = hashlib.sha256()
        with local.open("rb") as f:
            for b in iter(lambda: f.read(1024 * 1024), b""):
                h.update(b)
        assert h.hexdigest() == wanted, name
    requests = json.loads((root / "requests.json").read_text())
    domain = json.loads((a.first / "domain.json").read_text())
    reference = json.loads((root / "reference.json").read_text())["cases"]
    graph = json.loads((root / "bundle.json").read_text())
    result = json.loads((root / "results.json").read_text())
    native = []
    for path in sorted(root.glob("batch_*_capture.json")):
        native.extend(json.loads(path.read_text())["cases"])
    assert len(native) == len(reference) == len(graph["nodes"]) == 54
    tokens = requests["probe_tokens"]
    m, U, gain = read_frame(a.bundle, tokens)
    rivals = list(domain["token_map"])
    mapping = domain["token_map"]
    indices = [tokens.index(mapping[v]) for v in rivals]
    programs = {
        "full": list(range(65)),
        "attention": list(range(1, 65, 2)),
        "mlp": list(range(2, 65, 2)),
    }
    matches = 0
    maximum = 0.0
    total_commits = 0
    for unique, n, ref, node in zip(
        requests["unique_contexts"], native, reference, graph["nodes"], strict=True
    ):
        assert unique["ids"] == n["ids"] == ref["ids"] == node["ids"]
        matches += n["prediction"] == ref["prediction"]
        assert (
            n["candidate_tokens"] == tokens and n["native_output_row_calls"] == 248320
        )
        D = np.array([s["write"] for s in n["sources"]])
        r = np.array(n["final_residual"])
        scale = gain / np.sqrt(np.mean(r * r) + m["config_f"][1])
        bodies = {name: U @ (D[js].sum(0) * scale) for name, js in programs.items()}
        for name, values in bodies.items():
            error = float(np.max(np.abs(values - node["guard_body_scores"][name])))
            assert error < 1e-7
            maximum = max(maximum, error)
        rows = np.array([v[indices] for v in bodies.values()])
        aggregates = {
            "viterbi": rows.max(0),
            "marginal": np.logaddexp.reduce(rows, axis=0),
        }
        H = np.array(n["candidate_scores"])
        for name, scores in aggregates.items():
            winners = np.flatnonzero(scores == scores.max()).tolist()
            q = next(q for q in node["chart"]["queries"] if q["readout"] == name)
            assert q["winners"] == [rivals[j] for j in winners]
            decision = next(d for d in node["commitments"] if d["readout"] == name)
            admit = False
            if len(winners) == 1 and np.sort(scores)[-1] - np.sort(scores)[-2] > 0.5:
                atom = rivals[winners[0]]
                t = tokens.index(mapping[atom])
                high = max(v[t] for v in bodies.values())
                witness = sorted(k for k, v in bodies.items() if v[t] == high)[0]
                P = bodies[witness]
                mask = np.arange(len(tokens)) != t
                gaps = P[t] - P[mask]
                drift = (H[t] - P[t]) - (H[mask] - P[mask])
                admit = bool(gaps.min() > 0.5 and np.all(gaps > np.abs(drift)))
                if admit:
                    assert decision["body"] == witness and decision["atom"] == atom
                    assert H[t] > H[mask].max()
                    total_commits += 1
            assert (decision["status"] == "committed") == admit
    grouped = {}
    counts = {
        arm: Counter() for arm in ["rederive", "crystallize", "generated", "tool"]
    }
    tags = Counter()
    assert (
        len(requests["requests"])
        == len(graph["paths"])
        == len(result["records"])
        == 108
    )
    for req, path, record in zip(
        requests["requests"], graph["paths"], result["records"], strict=True
    ):
        assert path == record
        arm = req["arm"]
        node = graph["nodes"][req["unique_index"]]
        first = requests["first_hops"][req["first_hop_id"]]
        prefix = first["original_ids"]
        ids = node["ids"]
        insert = req["inserted_token"]
        expected = (
            first["native_token"]
            if arm == "generated"
            else mapping[str(req["a"] + req["b"])]
            if arm == "tool"
            else None
        )
        assert (
            insert == expected
            and ids == prefix + ([] if insert is None else [insert]) + req["suffix_ids"]
        )
        truth = str(req["a"] + req["b"] + req["c"])
        assert truth == path["truth"] and path["admission"] == "refused"
        assert path["host_token"] == node["host_token"]
        correct = node["host_token"] == mapping[truth]
        assert correct == path["host_correct"]
        counts[arm]["host_correct"] += correct
        counts[arm]["cases"] += 1
        tag = "generated" if arm == "generated" else "tool" if arm == "tool" else "none"
        assert path["memory"]["kind"] == tag
        tags[tag] += 1
        for answer in path["answers"]:
            decision = next(
                d for d in node["commitments"] if d["readout"] == answer["readout"]
            )
            assert (
                answer["status"] == decision["status"]
                and answer["atom"] == decision["atom"]
            )
            assert answer["memory"]["kind"] == tag
            grade = (
                "abstain"
                if answer["status"] == "abstain"
                else "affirmed"
                if answer["atom"] == truth
                else "hallucination"
            )
            assert answer["theory_verdict"] == grade
        grouped.setdefault((req["a"], req["b"], req["c"]), {})[arm] = req[
            "unique_index"
        ]
    assert len(grouped) == 27
    for arms in grouped.values():
        assert (
            arms["rederive"] == arms["crystallize"]
            and arms["generated"] == arms["tool"]
        )
    for arm, c in counts.items():
        assert (
            c["cases"] == 27
            and result["arms"][arm]["host_correct"] == c["host_correct"]
        )
    report = dict(
        schema="pdc.qwen35-two-hop-audit.v1",
        unique_contexts=54,
        paths=108,
        reference_matches=matches,
        maximum_independent_body_error=maximum,
        body_probe_annotations_checked=54 * 3 * len(tokens),
        unique_readout_commitments_checked=total_commits,
        memory_tags=dict(tags),
        first_hop_admitted=0,
        actual_admitted_replay_tested=0,
        input_hashes_verified=True,
        gate="passed" if matches == 54 else "failed_reference_agreement",
    )
    (root / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    assert matches == 54, "retain all results; reference agreement gate failed"


if __name__ == "__main__":
    main()
