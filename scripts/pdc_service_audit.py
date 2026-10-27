"""Audit the frozen R8-H paired service run and its finite numeric certificates."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from pdc_service import decision_sources


def analyze(directory, model_stem):
    def read(name):
        return json.loads((directory/name).read_text())
    domain, probe = read("r5_domain.json"), read("r8h_probe.json")
    full, partial = read("r8h_timed_baseline_capture.json"), read("r8h_timed_service_capture.json")
    baseline, service = read("r8h_baseline_bundle.json"), read("r8h_service_bundle.json")
    initial = read("r8h_baseline_capture.json")
    old = {r["id"]: r for r in read("r5_theory_results.json")["records"]}
    timing = read("r8h_timing.json")
    expected = [c for c in domain["cases"] if c["split"] == "r5"]
    assert len(expected) == len(full["cases"]) == len(partial["cases"]) == len(service["responses"]) == 75
    assert len(baseline["responses"]) == len(initial["cases"]) == 75
    assert baseline["request_mode"] == "live_native_full_baseline"
    assert service["request_mode"] == "live_native_subset"
    assert service["claims"]["theory_soundness"] is False
    tokens = probe["tokens"]
    assert tokens == sorted(set(tokens)) and len(tokens) == 1703
    assert service["probe_tokens"] == baseline["probe_tokens"] == tokens
    frame, bias = np.asarray(service["guard_frame"]), np.asarray(service["guard_bias"])
    assert np.array_equal(frame, baseline["guard_frame"])
    manifest = json.loads(Path(str(model_stem)+".fieldrun.json").read_text())
    blob = np.memmap(str(model_stem)+".fieldrun.bin", mode="r", dtype=np.uint8)
    arrays = {}
    for entry in manifest["arrays"]:
        if entry["name"] in ["wte", "ln_f.weight", "ln_f.bias"]:
            arr = np.frombuffer(blob, dtype={"f16": "<f2", "f32": "<f4"}[entry["dtype"]],
                count=np.prod(entry["shape"]), offset=entry["offset"]).reshape(entry["shape"])
            arrays[entry["name"]] = (arr[tokens] if entry["name"] == "wte" else arr).astype(float)
    assert np.array_equal(frame, arrays["wte"])
    assert np.max(np.abs(bias-frame@arrays["ln_f.bias"])) < 1e-8
    programs = {"full": list(range(25)), "attention": list(range(1,25,2)), "mlp": list(range(2,25,2))}
    assert service["body_programs"] == baseline["body_programs"] == programs
    answer_indices = [tokens.index(domain["token_map"][a]) for a in domain["rivals"]]
    counter = {name: Counter() for name in ["viterbi", "marginal"]}
    reasons = {name: Counter() for name in counter}
    max_body_error, misses, fragile_truth = 0., 0, 0
    for pos, (f, p, b, response, case, original) in enumerate(zip(full["cases"], partial["cases"],
        baseline["responses"], service["responses"], expected, initial["cases"], strict=True)):
        assert f["ids"] == p["ids"] == response["context_ids"] == case["ids"] == original["ids"]
        assert f["candidate_tokens"] == p["candidate_tokens"] == tokens
        assert f["scoring_scope"] == "full_vocabulary" and p["scoring_scope"] == "candidate_subset"
        assert f["observer_check_performed"] is p["observer_check_performed"] is False
        assert f["native_output_row_calls"] == manifest["config"][4]
        assert p["native_output_row_calls"] == len(tokens)
        assert f["prediction_positions"] == p["prediction_positions"] == [len(case["ids"])-1]
        assert np.array_equal(f["candidate_scores"], p["candidate_scores"])
        assert f["predictions"] == original["predictions"] == [old[case["id"]]["host_token"]]
        assert f["sources"] == p["sources"] == original["sources"]
        assert np.array_equal(f["final_hidden"], p["final_hidden"])
        assert response["full_host_token"] is None and b["full_host_token"] == f["predictions"][0]
        raw, residual = decision_sources(p, len(arrays["ln_f.weight"]))
        decision_sources(f, len(arrays["ln_f.weight"]))
        sigma = np.sqrt(np.mean((residual-residual.mean())**2)+1e-5)
        sources = (raw-raw.mean(axis=1,keepdims=True))*arrays["ln_f.weight"]/sigma
        assert np.max(np.abs(sources-response["sources"])) < 1e-8
        body_scores = {}
        for name, coalition in programs.items():
            values = sources[coalition].sum(axis=0) @ frame.T+bias
            error = float(np.max(np.abs(values-response["guard_body_scores"][name])))
            assert error < 1e-8
            max_body_error = max(max_body_error, error)
            assert np.max(np.abs(values-b["guard_body_scores"][name])) < 1e-8
            assert np.max(np.abs(values[answer_indices]-response["body_scores"][name])) < 1e-8
            body_scores[name] = values
        rows = np.array([v[answer_indices] for v in body_scores.values()])
        scores_by_readout = {"viterbi": rows.max(axis=0), "marginal": np.logaddexp.reduce(rows,axis=0)}
        truth = str(case["a"]+case["b"])
        truth_j = domain["rivals"].index(truth)
        misses += not any(all(v[truth_j] > s for j,s in enumerate(v) if j != truth_j) for v in rows)
        winners = {}
        for readout, scores in scores_by_readout.items():
            query = next(q for q in response["chart"]["queries"] if q["readout"] == readout)
            winner_indices = np.flatnonzero(scores == scores.max()).tolist()
            winners[readout] = sorted(domain["rivals"][i] for i in winner_indices)
            assert query["winners"] == winners[readout]
            judgment = next(j for j in response["commitments"] if j["readout"] == readout)
            baseline_judgment = next(j for j in b["commitments"] if j["readout"] == readout)
            assert judgment == baseline_judgment
            expected_commit = False
            if len(winner_indices) == 1 and query["margin"] > .5:
                atom = winners[readout][0]
                t = tokens.index(domain["token_map"][atom])
                high = max(v[t] for v in body_scores.values())
                body = sorted(name for name,v in body_scores.items() if v[t] == high)[0]
                partial_scores = body_scores[body]
                host = np.array(p["candidate_scores"][0])
                mask = np.arange(len(tokens)) != t
                gaps = partial_scores[t]-partial_scores[mask]
                drifts = (host[t]-partial_scores[t])-(host[mask]-partial_scores[mask])
                expected_commit = min(gaps) > .5 and np.all(gaps > np.abs(drifts))
                if expected_commit:
                    assert judgment["body"] == body and judgment["atom"] == atom
                    assert judgment["covered_rivals"] == service["certificate_rivals"]
                    assert len(judgment["pairs"]) == len(tokens)-1
                    assert np.min(gaps-np.abs(drifts)) > 0
                    assert all(host[t] > host[j] for j in range(len(tokens)) if j != t)
            assert (judgment["status"] == "committed") == expected_commit
            if expected_commit:
                counter[readout]["committed"] += 1
                agreement = domain["token_map"][judgment["atom"]] == f["predictions"][0]
                counter[readout]["full_host_agreements"] += agreement
                verdict = "affirmed" if judgment["atom"] == truth else "hallucination"
                counter[readout][verdict] += 1
                assert judgment["theory_verdict"] == verdict
            else:
                assert judgment["atom"] is None and judgment["theory_verdict"] == "abstain"
                counter[readout]["abstained"] += 1
                reasons[readout][judgment["reason"]] += 1
        fragile_truth += winners["marginal"] == [truth] and winners["viterbi"] != [truth]
        assert response["chart"]["fragile"] == (winners["viterbi"] != winners["marginal"])
    report = {"schema": "pdc.service-audit.v1", "contexts": 75, "untouched_heldout": 25,
        "probe_size": len(tokens), "body_probe_annotations_checked": 75*3*len(tokens),
        "maximum_body_score_discrepancy": max_body_error, "probe_score_max_discrepancy": 0,
        "source_write_max_discrepancy": 0, "frozen_R5_full_host_agreements": 75,
        "misses": int(misses), "fragile_truth": int(fragile_truth), "theory_soundness_claim": False}
    for readout, counts in counter.items():
        n = counts["committed"]
        agreement = counts["full_host_agreements"]/n if n else None
        report[readout] = {**dict(counts), "committed": n, "affirmed": counts["affirmed"],
            "hallucination": counts["hallucination"], "full_host_agreement": agreement,
            "hallucination_rate": counts["hallucination"]/n if n else None,
            "host_gate": "untested" if not n else "passed" if agreement >= .95 else "retired",
            "abstention_reasons": dict(reasons[readout])}
    full_calls = sum(c["native_output_row_calls"] for c in full["cases"])
    subset_calls = sum(c["native_output_row_calls"] for c in partial["cases"])
    body_calls = 75*3*len(tokens)
    assert service["work"]["native_output_row_calls"] == subset_calls
    assert baseline["work"]["native_output_row_calls"] == full_calls
    assert service["work"]["body_probe_dot_calls"] == baseline["work"]["body_probe_dot_calls"] == body_calls
    report["compute"] = {"baseline_native_output_row_calls": full_calls,
        "service_native_output_row_calls": subset_calls, "body_probe_dot_calls_each_arm": body_calls,
        "native_output_rows_skipped": full_calls-subset_calls,
        "native_row_reduction": 1-subset_calls/full_calls,
        "output_and_body_dot_reduction": 1-(subset_calls+body_calls)/(full_calls+body_calls),
        "positions_skipped": 0, "layers_skipped": 0}
    report["timings"] = {"paired_runs": timing["runs"],
        "native_baseline_forward_decode_seconds": sum(c["elapsed_forward_decode_us"] for c in full["cases"])/1e6,
        "native_service_forward_decode_seconds": sum(c["elapsed_forward_decode_us"] for c in partial["cases"])/1e6,
        "inference": "one fixed ordered pair; descriptive timings only"}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--model-bundle", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.artifacts, args.model_bundle)
    result["sha256"] = {name: hashlib.sha256((args.artifacts/name).read_bytes()).hexdigest()
        for name in ["r8h_probe.json", "r8h_baseline_bundle.json", "r8h_service_bundle.json", "R8H_PREREG.md",
                     "R8H_TIMING_ADDENDUM.md", "r8h_timed_baseline_capture.json", "r8h_timed_service_capture.json"]}
    (args.artifacts/"r8h_results.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ["sha256", "timings"]},indent=2))


if __name__ == "__main__":
    main()
