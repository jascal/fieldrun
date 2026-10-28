"""Grade the frozen Qwen3.5 comparison after native selection; no model calls."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from pdc_certificate import certify
from pdc_derive import Body, evaluate
from pdc_qwen35_engineering_check import read_frame, check_reconstruction
from pdc_theory_check import judge_commitment


def analyze(directory, stem):
    domain = json.loads((directory / "domain.json").read_text())
    tokens = json.loads((directory / "probe.json").read_text())["tokens"]
    m, U, gain = read_frame(stem, tokens)
    rivals = list(domain["token_map"])
    answer_indices = [tokens.index(domain["token_map"][a]) for a in rivals]
    token_to_atom = {domain["token_map"][a]: a for a in rivals}
    guard_rivals = [token_to_atom.get(t, f"token:{t}") for t in tokens]
    names = ["embed"] + [
        name
        for layer in range(m["config"][0])
        for name in [f"L{layer}.attn", f"L{layer}.mlp"]
    ]
    programs = {
        "full": list(range(len(names))),
        "attention": list(range(1, len(names), 2)),
        "mlp": list(range(2, len(names), 2)),
    }
    native = []
    capture_hashes = {}
    for path in sorted(directory.glob("batch_*_capture.json")):
        native.extend(json.loads(path.read_text())["cases"])
        capture_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    assert len(native) == len(domain["cases"]) == 100
    responses = []
    totals = {
        split: {readout: Counter() for readout in ["viterbi", "marginal"]}
        for split in ["development", "heldout"]
    }
    host = Counter()
    n = Counter()
    errors = []
    for i, (case, observed) in enumerate(zip(domain["cases"], native, strict=True)):
        assert observed["ids"] == case["ids"] and observed["candidate_tokens"] == tokens
        assert [s["id"] for s in observed["sources"]] == names
        assert not observed["observer_check_performed"]
        assert observed["native_output_row_calls"] == m["config"][5]
        raw = np.array([s["write"] for s in observed["sources"]])
        residual = np.array(observed["final_residual"])
        hidden = np.array(observed["final_hidden"])
        assert np.isfinite(raw).all() and np.isfinite(residual).all()
        folded, reconstruction_errors = check_reconstruction(raw,residual,hidden,gain,m["config_f"][1],observed["candidate_scores"],U)
        guard = {
            name: (folded[coalition].sum(0) @ U.T).tolist()
            for name, coalition in programs.items()
        }
        body = {
            name: [scores[j] for j in answer_indices] for name, scores in guard.items()
        }
        table = {
            a: [Body(name, scores[j]) for name, scores in body.items()]
            for j, a in enumerate(rivals)
        }
        chart = evaluate(table, domain["temperature"])
        truth = str(case["a"] + case["b"])
        assert truth == case["truth"]
        truth_index = rivals.index(truth)
        deciding_bodies = [
            name
            for name, scores in body.items()
            if all(
                scores[truth_index] > s
                for j, s in enumerate(scores)
                if j != truth_index
            )
        ]
        winners = {q["readout"]: q["winners"] for q in chart["queries"]}
        fragile_truth = winners["marginal"] == [truth] and winners["viterbi"] != [truth]
        commitments = []
        for q in chart["queries"]:
            decision = certify(
                q,
                guard,
                observed["candidate_scores"],
                guard_rivals,
                guard_rivals,
                domain["gamma"],
            )
            decision["fragile"] = chart["fragile"]
            decision["memory"] = {"kind": "none"}
            graded = judge_commitment(decision, body, rivals, case["truth"])
            counts = totals[case["split"]][q["readout"]]
            if graded["status"] == "committed":
                counts["committed"] += 1
                counts[graded["theory_verdict"]] += 1
                counts["full_host_agreements"] += (
                    domain["token_map"][graded["atom"]] == observed["prediction"]
                )
                t = guard_rivals.index(graded["atom"])
                counts["covered_candidate_agreements"] += t == int(np.argmax(observed["candidate_scores"]))
                assert t == int(np.argmax(observed["candidate_scores"]))
            else:
                counts["abstained"] += 1
            counts["misses"] += not deciding_bodies
            counts["fragile_truth"] += fragile_truth
            commitments.append(graded)
        split = case["split"]
        n[split] += 1
        correct = observed["prediction"] == domain["token_map"][case["truth"]]
        host[split] += correct
        errors.append(reconstruction_errors)
        responses.append(
            dict(
                id=case["id"],
                split=split,
                truth=case["truth"],
                host_token=observed["prediction"],
                host_correct=bool(correct),
                chart=chart,
                body_scores=body,
                guard_body_scores=guard,
                commitments=commitments,
                native_case_index=i,
                errors=errors[-1],
            )
        )
    summary = {}
    for split, by_readout in totals.items():
        summary[split] = dict(
            contexts=n[split],
            host_correct=host[split],
            host_accuracy=host[split] / n[split],
        )
        for readout, counts in by_readout.items():
            c = counts["committed"]
            affirmed = counts["affirmed"]
            hallucinations = counts["hallucination"]
            accuracy = affirmed / c if c else None
            coverage = affirmed / n[split]
            agreement = counts["full_host_agreements"] / c if c else None
            positive = bool(
                c
                and host[split] / n[split] >= 0.95
                and coverage >= 0.20
                and agreement >= 0.95
                and hallucinations / c < 0.10
            )
            negative = coverage < 0.10 or (accuracy is not None and accuracy < 0.70)
            summary[split][readout] = {
                **dict(counts),
                "committed": c,
                "affirmed": affirmed,
                "hallucination": hallucinations,
                "useful_coverage": coverage,
                "commitment_theory_accuracy": accuracy,
                "full_host_agreement": agreement,
                "covered_candidate_agreement": counts["covered_candidate_agreements"]/c if c else None,
                "hallucination_rate": hallucinations / c if c else None,
                "classification": "positive_local_comparison"
                if positive
                else "negative_fixed_policy"
                if negative
                else "inconclusive",
            }
    packet = dict(
        schema="pdc.qwen35-derivation-bundle.v1",
        claims={"theory_soundness": False, "service_savings": False},
        source_ids=names,
        body_programs=programs,
        certificate_rivals=guard_rivals,
        source_capture_sha256=capture_hashes,
        responses=responses,
    )
    (directory / "bundle.json").write_text(
        json.dumps(packet, separators=(",", ":")) + "\n"
    )
    result = dict(
        schema="pdc.qwen35-comparison-result.v1",
        summary=summary,
        model_forwards=100,
        native_output_row_calls=100 * m["config"][5],
        body_probe_annotations=100 * 3 * len(tokens),
        reconstruction_gate="passed_all_100_cases",
        error_maxima={key: max(e[key] for e in errors) for key in errors[0]},
        scope="new fixed model/template comparison; original GPT-2 results unchanged",
    )
    (directory / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    a = parser.parse_args()
    print(json.dumps(analyze(a.artifacts, a.bundle), indent=2))
