"""Build the frozen Qwen two-hop chart, provenance paths and external grades."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from pdc_certificate import certify
from pdc_derive import Body, evaluate
from pdc_qwen35_engineering_check import read_frame
from pdc_theory_check import judge_commitment


def analyze(root, first_root, stem):
    requests = json.loads((root / "requests.json").read_text())
    original_domain = json.loads((first_root / "domain.json").read_text())
    tokens = requests["probe_tokens"]
    m, U, gain = read_frame(stem, tokens)
    rivals = list(original_domain["token_map"])
    mapping = original_domain["token_map"]
    indices = [tokens.index(mapping[a]) for a in rivals]
    inverse = {v: k for k, v in mapping.items()}
    covered = [inverse.get(t, f"token:{t}") for t in tokens]
    names = ["embed"] + [
        name for layer in range(32) for name in [f"L{layer}.attn", f"L{layer}.mlp"]
    ]
    programs = {
        "full": list(range(65)),
        "attention": list(range(1, 65, 2)),
        "mlp": list(range(2, 65, 2)),
    }
    native = []
    for path in sorted(root.glob("batch_*_capture.json")):
        native.extend(json.loads(path.read_text())["cases"])
    assert len(native) == len(requests["unique_contexts"]) == 54
    nodes = []
    errors = []
    for index, (case, observed) in enumerate(
        zip(requests["unique_contexts"], native, strict=True)
    ):
        assert case["ids"] == observed["ids"] and observed["candidate_tokens"] == tokens
        assert [s["id"] for s in observed["sources"]] == names
        assert not observed["observer_check_performed"]
        D = np.array([s["write"] for s in observed["sources"]])
        r = np.array(observed["final_residual"])
        h = np.array(observed["final_hidden"])
        folded = D * gain / np.sqrt(np.mean(r * r) + m["config_f"][1])
        guard = {
            name: (folded[js].sum(0) @ U.T).tolist() for name, js in programs.items()
        }
        bodies = {name: [v[j] for j in indices] for name, v in guard.items()}
        table = {
            atom: [Body(name, v[j]) for name, v in bodies.items()]
            for j, atom in enumerate(rivals)
        }
        chart = evaluate(table, 1.0)
        judgments = [
            certify(q, guard, observed["candidate_scores"], covered, covered, 0.5)
            for q in chart["queries"]
        ]
        for judgment in judgments:
            judgment["fragile"] = chart["fragile"]
        nodes.append(
            dict(
                id=case["id"],
                ids=case["ids"],
                host_token=observed["prediction"],
                chart=chart,
                commitments=judgments,
                body_scores=bodies,
                guard_body_scores=guard,
                capture_index=index,
                provenance="supplied by each incoming path",
            )
        )
        errors.append(
            dict(
                source_sum=float(np.max(np.abs(D.sum(0) - r))),
                rms_fold=float(np.max(np.abs(folded.sum(0) - h))),
                full_body_score=float(
                    np.max(
                        np.abs(np.array(guard["full"]) - observed["candidate_scores"])
                    )
                ),
            )
        )
    counters = {
        arm: Counter() for arm in ["rederive", "crystallize", "generated", "tool"]
    }
    readouts = {
        arm: {readout: Counter() for readout in ["viterbi", "marginal"]}
        for arm in counters
    }
    paths = []
    for req in requests["requests"]:
        node = nodes[req["unique_index"]]
        truth = str(req["a"] + req["b"] + req["c"])
        assert truth == req["truth"]
        arm = req["arm"]
        counters[arm]["cases"] += 1
        correct = node["host_token"] == mapping[truth]
        counters[arm]["host_correct"] += correct
        answers = []
        for j, decision in enumerate(node["commitments"]):
            grade = judge_commitment(decision, node["body_scores"], rivals, truth)
            counts = readouts[arm][grade["readout"]]
            counts[grade["theory_verdict"]] += 1
            if grade["status"] == "committed":
                counts["host_agree"] += mapping[grade["atom"]] == node["host_token"]
            # Full numeric certificate is shared by node/reference; path owns memory provenance.
            view = {
                k: v for k, v in grade.items() if k not in ["pairs", "covered_rivals"]
            }
            view.update(
                memory={"kind": req["memory"]},
                certificate_ref={"node": node["id"], "query": j},
            )
            answers.append(view)
        paths.append(
            dict(
                a=req["a"],
                b=req["b"],
                c=req["c"],
                arm=arm,
                truth=truth,
                first_hop_id=req["first_hop_id"],
                second_hop_id=node["id"],
                admission="refused",
                memory={"kind": req["memory"], "token": req["inserted_token"]},
                host_token=node["host_token"],
                host_correct=bool(correct),
                answers=answers,
            )
        )
    summary = {}
    for arm, counts in counters.items():
        assert counts["cases"] == 27
        summary[arm] = dict(
            cases=27,
            host_correct=counts["host_correct"],
            host_accuracy=counts["host_correct"] / 27,
        )
        for readout, counts in readouts[arm].items():
            c = counts["affirmed"] + counts["hallucination"]
            summary[arm][readout] = dict(
                committed=c,
                affirmed=counts["affirmed"],
                hallucination=counts["hallucination"],
                abstained=counts["abstain"],
                full_host_agreement=counts["host_agree"] / c if c else None,
            )
    delta = summary["generated"]["host_accuracy"] - summary["rederive"]["host_accuracy"]
    result = dict(
        schema="pdc.qwen35-two-hop-result.v1",
        triples=27,
        paths=108,
        unique_second_hop_contexts=54,
        reused_first_hops=9,
        admission_counts=requests["admission_counts"],
        arms=summary,
        generated_minus_rederive=delta,
        refused_improvement="supported_on_slice"
        if delta >= 0.05
        else "retired_on_slice",
        admitted_redundancy="untested",
        actual_admitted_replay="untested",
        native_scientific_forwards=54,
        reference_forwards=54,
        native_output_row_calls=54 * m["config"][5],
        body_probe_annotations=54 * 3 * len(tokens),
        error_maxima={k: max(e[k] for e in errors) for k in errors[0]},
        records=paths,
    )
    graph = dict(
        schema="pdc.qwen35-two-hop-bundle.v1",
        candidate_scope=covered,
        body_programs=programs,
        source_ids=names,
        reused_first_hops=requests["first_hops"],
        first_hop_evidence=dict(
            bundle="../pdc_qwen35_4b/bundle.json",
            sha256=hashlib.sha256(
                (first_root / "bundle.json").read_bytes()
            ).hexdigest(),
        ),
        nodes=nodes,
        paths=paths,
        claims={"theory_soundness": False, "causal_semantic_use": False},
    )
    (root / "bundle.json").write_text(json.dumps(graph, separators=(",", ":")) + "\n")
    (root / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["artifacts", "first", "bundle"]:
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    r = analyze(a.artifacts, a.first, a.bundle)
    print(json.dumps({k: v for k, v in r.items() if k != "records"}, indent=2))
