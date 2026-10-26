"""Grade learned output-frame artifacts with fieldrun's unchanged PDC/theory rules."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from pdc_certificate import certify
from pdc_derive import Body, evaluate
from pdc_theory_check import judge_commitment


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["data", "frames", "domain", "baseline-memory", "out"]:
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    learning = json.loads((a.frames / "results.json").read_text())
    assert "learning_gate" in learning, (
        "both fixed runs must finish before chart grading"
    )
    baseline = json.loads(a.baseline_memory.read_text())["arms"]
    domain = json.loads(a.domain.read_text())
    d = np.load(a.data)
    pool = d["pool_tokens"].tolist()
    mapping = domain["token_map"]
    rivals = list(mapping)
    token_to_atom = {v: k for k, v in mapping.items()}
    covered = [token_to_atom.get(v, f"token:{v}") for v in pool]
    indices = [pool.index(mapping[v]) for v in rivals]
    sources = np.concatenate([d["native_sources"], d["hop2_sources"]], axis=0)
    r = d["protected_residuals"]
    outside = d["outside_max"]
    groups = d["protected_arms"]
    truth_tokens = d["protected_truth_tokens"]
    programs = {
        "full": list(range(65)),
        "attention": list(range(1, 65, 2)),
        "mlp": list(range(2, 65, 2)),
    }
    all_summaries = {}
    case_records = []
    for arm in ["decode", "admission"]:
        summaries = {}
        for endpoint in ["initial", "final"]:
            f = np.load(a.frames / f"{arm}_{endpoint}.npz")
            U, b = f["U"], f["bias"]
            host = r @ U.T + b
            assert np.array_equal(np.c_[host, outside].argmax(1), d["teacher_indices"])
            body = {
                name: sources[:, js].sum(1) @ U.T + b for name, js in programs.items()
            }
            counts = {
                tag: {q: Counter() for q in ["viterbi", "marginal"]}
                for tag in ["first", "rederive", "crystallize", "generated", "tool"]
            }
            eligible = {q: 0 for q in ["viterbi", "marginal"]}
            admitted = learning[arm]["initial" if endpoint == "initial" else "final"]
            for i in range(208):
                truth = token_to_atom[int(truth_tokens[i])]
                tag = str(groups[i])
                table = {
                    atom: [
                        Body(name, float(scores[i, indices[j]]))
                        for name, scores in body.items()
                    ]
                    for j, atom in enumerate(rivals)
                }
                chart = evaluate(table, 1.0)
                scores = {name: row[i].tolist() for name, row in body.items()}
                views = []
                for query in chart["queries"]:
                    cert = certify(
                        query, scores, host[i].tolist(), covered, covered, 0.5
                    )
                    cert["fragile"] = chart["fragile"]
                    cert["memory"] = {
                        "kind": tag if tag in ["generated", "tool"] else "none"
                    }
                    grade = judge_commitment(
                        cert,
                        {name: v[i, indices].tolist() for name, v in body.items()},
                        rivals,
                        truth,
                    )
                    c = counts[tag][query["readout"]]
                    c[grade["theory_verdict"]] += 1
                    if grade["status"] == "committed":
                        assert mapping[grade["atom"]] == int(
                            d["native_teacher_tokens"][i]
                        )
                        c["host_agree"] += 1
                        if i < 100 and admitted[i]["admitted"]:
                            eligible[query["readout"]] += 1
                    views.append(
                        {
                            k: v
                            for k, v in grade.items()
                            if k not in ["pairs", "covered_rivals"]
                        }
                    )
                case_records.append(
                    dict(
                        learning_arm=arm,
                        endpoint=endpoint,
                        id=str(d["protected_case_ids"][i]),
                        group=tag,
                        truth=truth,
                        judgments=views,
                    )
                )
            summary = {}
            for tag, by_readout in counts.items():
                summary[tag] = {}
                for q, c in by_readout.items():
                    n = c["affirmed"] + c["hallucination"]
                    summary[tag][q] = dict(
                        committed=n,
                        affirmed=c["affirmed"],
                        hallucination=c["hallucination"],
                        abstained=c["abstain"],
                        full_host_agreement=c["host_agree"] / n if n else None,
                    )
                    if endpoint == "initial" and tag != "first":
                        old = baseline[tag][q]
                        for key in [
                            "committed",
                            "affirmed",
                            "hallucination",
                            "abstained",
                        ]:
                            assert summary[tag][q][key] == old[key], (
                                f"initial chart mismatch {tag}/{q}/{key}"
                            )
            summaries[endpoint] = dict(
                charts=summary, operational_first_hop_admissions_by_readout=eligible
            )
        failures = []
        for tag in ["rederive", "crystallize", "generated", "tool"]:
            for q in ["viterbi", "marginal"]:
                current = summaries["final"]["charts"][tag][q]
                old = baseline[tag][q]
                if (
                    current["affirmed"] < old["affirmed"]
                    or current["hallucination"] > old["hallucination"]
                ):
                    failures.append(
                        dict(memory_arm=tag, readout=q, before=old, after=current)
                    )
        summaries["chart_quality_gate"] = dict(passed=not failures, failures=failures)
        all_summaries[arm] = summaries
    joint = (
        learning["learning_gate"]["passed"]
        and all_summaries["admission"]["chart_quality_gate"]["passed"]
    )
    report = dict(
        schema="pdc.qwen35-learned-frame-chart-audit.v1",
        learning_arms=all_summaries,
        local_advance_gate=dict(
            passed=joint,
            learning_gate=learning["learning_gate"],
            admission_chart_quality=all_summaries["admission"]["chart_quality_gate"],
        ),
        records=case_records,
        new_model_forwards=0,
        claims=dict(fresh_confirmation=False, actual_replay=False),
        sha256={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [a.data, a.domain, a.baseline_memory, a.frames / "results.json"]
        },
    )
    a.out.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ["records", "sha256"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
