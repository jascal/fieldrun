"""Audit R5 source/body/readout/certificate artifacts without the exporter code."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from pdc_read_check import load_bundle


def audit(bundle, domain, capture, weights):
    rivals = domain["rivals"]
    assert bundle["rivals"] == rivals
    tokens = [domain["token_map"][a] for a in rivals]
    assert bundle["candidate_tokens"] == tokens
    frame, bias = np.asarray(bundle["frame"]), np.asarray(bundle["bias"])
    assert np.array_equal(frame, weights["wte"][tokens])
    assert np.max(np.abs(bias - frame @ weights["ln_f.bias"].astype(float))) < 1e-8
    expected = {c["id"]: c for c in domain["cases"] if c["split"] == "r5"}
    assert len(bundle["records"]) == len(capture["cases"]) == len(expected) == 75
    assert {r["id"] for r in bundle["records"]} == set(expected)
    max_annotation_error = 0.
    committed = {"viterbi": 0, "marginal": 0}
    for row, native in zip(bundle["records"], capture["cases"], strict=True):
        assert row["context_ids"] == native["ids"] == expected[row["id"]]["ids"]
        assert row["a"] == expected[row["id"]]["a"] and row["b"] == expected[row["id"]]["b"]
        assert native["observer_hidden_max_error"] == 0
        assert native["predictions"] == native["unobserved_predictions"] == [row["host_token"]]
        assert native["candidate_tokens"] == tokens
        assert native["candidate_scores"][0] == row["host_candidate_scores"]
        sources = np.asarray(row["sources"])
        assert sources.shape == (25, frame.shape[1]) and np.isfinite(sources).all()
        position = len(native["ids"])-1
        raw_records = [s for s in native["sources"] if s["position"] == position]
        assert [s["id"] for s in raw_records] == bundle["source_ids"]
        raw = np.asarray([s["write"] for s in raw_records])
        residual = np.asarray(native["reads"][0]["final_residual"])
        scale = np.sqrt(np.mean((residual-residual.mean())**2)+1e-5)
        folded = (raw-raw.mean(axis=1, keepdims=True))*weights["ln_f.weight"]/scale
        assert np.max(np.abs(sources-folded)) < 1e-8
        body_scores = {}
        for name, coalition in bundle["body_programs"].items():
            values = sources[coalition].sum(axis=0) @ frame.T + bias
            error = np.max(np.abs(values - row["body_scores"][name]))
            assert error < 1e-8
            max_annotation_error = max(max_annotation_error, float(error))
            body_scores[name] = values
            for index, atom in enumerate(rivals):
                body = next(b for b in row["chart"]["bodies"][atom] if b["id"] == name)
                assert abs(body["score"] - values[index]) < 1e-8
        vit = np.max(list(body_scores.values()), axis=0)
        # Independent real-valued formula, all R5 temperatures are fixed to 1.
        mar = np.array([math.log(math.fsum(math.exp(v[i]) for v in body_scores.values()))
            for i in range(len(rivals))])
        computed_winners = {}
        for readout, scores in [("viterbi", vit), ("marginal", mar)]:
            query = next(q for q in row["chart"]["queries"] if q["readout"] == readout)
            assert max(abs(query["scores"][a] - scores[i]) for i, a in enumerate(rivals)) < 1e-8
            winner_indices = np.flatnonzero(scores == scores.max()).tolist()
            computed_winners[readout] = sorted(rivals[i] for i in winner_indices)
            assert query["winners"] == computed_winners[readout]
            judgment = next(c for c in row["commitments"] if c["readout"] == readout)
            should_commit = False
            if len(winner_indices) == 1 and query["margin"] > domain["gamma"]:
                t = winner_indices[0]
                best = max(values[t] for values in body_scores.values())
                body = sorted(name for name, values in body_scores.items() if values[t] == best)[0]
                partial = body_scores[body]
                host = np.array(native["candidate_scores"][0])
                mask = np.arange(len(rivals)) != t
                gaps = partial[t] - partial[mask]
                drift = (host[t]-partial[t]) - (host[mask]-partial[mask])
                should_commit = np.min(gaps) > domain["gamma"] and np.all(gaps > np.abs(drift))
                if should_commit:
                    assert judgment["body"] == body and judgment["atom"] == rivals[t]
                    assert judgment["covered_rivals"] == rivals
                    assert all(host[t] > host[j] for j in range(len(rivals)) if j != t)
                    assert len(judgment["pairs"]) == len(rivals)-1
                    assert judgment["minimum_pairwise_slack"] > 0
                    committed[readout] += 1
            assert (judgment["status"] == "committed") == should_commit
            if not should_commit:
                assert judgment["atom"] is None
        assert row["chart"]["fragile"] == (computed_winners["viterbi"] != computed_winners["marginal"])
    return {"schema": "pdc.bundle-audit.v1", "contexts_checked": 75, "heldout_contexts_captured": 0,
        "body_annotations_checked": 75*27, "commitments_checked": committed,
        "maximum_annotation_error": max_annotation_error,
        "scope": "finite numeric source/body/candidate guard audit, not a theory-soundness proof"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["bundle", "domain", "capture", "out"]:
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--model-bundle", type=Path, required=True)
    args = parser.parse_args()
    _, weights = load_bundle(args.model_bundle)
    result = audit(json.loads(args.bundle.read_text()), json.loads(args.domain.read_text()),
        json.loads(args.capture.read_text()), weights)
    result["sha256"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [args.bundle, args.domain, args.capture]}
    args.out.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
