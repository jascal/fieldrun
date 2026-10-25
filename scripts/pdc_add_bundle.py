"""Export a certified finite-body bundle; external theory grading is separate."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from pdc_certificate import certify
from pdc_derive import Body, evaluate
from pdc_read_check import load_bundle, reference


def build(dump, domain, selection, stem):
    manifest, w = load_bundle(stem)
    cases = [c for c in domain["cases"] if c["split"] == "r5"]
    if len(cases) != len(dump["cases"]) or len(cases) != 75:
        raise ValueError("R5 requires all 75 frozen cases; held-out capture is forbidden here")
    rivals = domain["rivals"]
    candidate_tokens = [domain["token_map"][a] for a in rivals]
    frame = w["wte"][candidate_tokens].astype(np.float64)
    gamma, beta = (w[name].astype(np.float64) for name in ["ln_f.weight", "ln_f.bias"])
    bias = frame @ beta
    source_ids = ["embed"] + [name for layer in range(manifest["config"][0])
        for name in [f"L{layer}.attn", f"L{layer}.mlp"]]
    programs = {"full": list(range(len(source_ids))),
        "attention": [i for i, name in enumerate(source_ids) if name.endswith(".attn")],
        "mlp": [i for i, name in enumerate(source_ids) if name.endswith(".mlp")]}
    records = []
    for case, frozen, selected in zip(dump["cases"], cases, selection["contexts"], strict=True):
        if case["ids"] != frozen["ids"] or case["ids"] != selected:
            raise ValueError("native context differs from the frozen prompt")
        if case["observer_hidden_max_error"] != 0 or case["predictions"] != case["unobserved_predictions"]:
            raise ValueError("observer changed the native computation")
        i = len(case["ids"]) - 1
        if case["prediction_positions"] != [i] or case["candidate_tokens"] != candidate_tokens:
            raise ValueError("candidate or prediction-position mismatch")
        raw_sources = [s for s in case["sources"] if s["position"] == i]
        if [s["id"] for s in raw_sources] != source_ids:
            raise ValueError("source decomposition is incomplete or reordered")
        raw = np.asarray([s["write"] for s in raw_sources], dtype=np.float64)
        residual = np.asarray(case["reads"][0]["final_residual"], dtype=np.float64)
        sigma = np.sqrt(np.mean((residual - residual.mean())**2) + 1e-5)
        folded = (raw - raw.mean(axis=1, keepdims=True)) * gamma / sigma
        hidden = np.asarray(case["final_hidden"][i], dtype=np.float64)
        body_scores = {name: (frame @ folded[coalition].sum(axis=0) + bias).tolist()
            for name, coalition in programs.items()}
        table = {a: [Body(name, scores[j]) for name, scores in body_scores.items()]
            for j, a in enumerate(rivals)}
        chart = evaluate(table, domain["temperature"])
        host_scores = case["candidate_scores"][0]
        host_max = max(host_scores)
        restricted = [a for a, s in zip(rivals, host_scores, strict=True) if s == host_max]
        commitments = [certify(query, body_scores, host_scores, rivals, rivals, domain["gamma"])
            for query in chart["queries"]]
        for commitment in commitments:
            if commitment["status"] == "committed" and restricted != [commitment["atom"]]:
                raise ValueError("pairwise certificate failed its host agreement consequence")
        ref = reference(case["ids"], manifest["config"], w, last_only=True)[0]
        records.append({"id": frozen["id"], "a": frozen["a"], "b": frozen["b"],
            "context_ids": case["ids"], "query_position": i, "chart": chart,
            "sources": folded.tolist(), "body_scores": body_scores,
            "host_candidate_scores": host_scores, "host_restricted_winners": restricted,
            "host_token": case["predictions"][0], "reference_token": ref,
            "commitments": commitments,
            "source_sum_max_error": float(np.max(np.abs(raw.sum(axis=0)-residual))),
            "folded_hidden_max_error": float(np.max(np.abs(folded.sum(axis=0)+beta-hidden))),
            "full_body_host_score_max_error": float(np.max(np.abs(np.array(body_scores["full"])-host_scores))),
            "theory": "not_loaded"})
    summary = {"n": len(records),
        "reference_agreement": sum(r["host_token"] == r["reference_token"] for r in records)/len(records),
        "observer_hidden_max_error": 0,
        "source_sum_max_error": max(r["source_sum_max_error"] for r in records),
        "folded_hidden_max_error": max(r["folded_hidden_max_error"] for r in records),
        "full_body_host_score_max_error": max(r["full_body_host_score_max_error"] for r in records),
        "fragile_contexts": sum(r["chart"]["fragile"] for r in records)}
    return {"schema": "pdc.derivation-bundle.v1", "domain_schema": domain["schema"],
        "certificate_scope": "fixed_nine_candidate_tokens_only", "rivals": rivals,
        "candidate_tokens": candidate_tokens, "frame": frame.tolist(), "bias": bias.tolist(),
        "source_ids": source_ids, "body_programs": programs, "summary": summary, "records": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["dump", "domain", "selection", "bundle", "out"]:
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    result = build(json.loads(args.dump.read_text()), json.loads(args.domain.read_text()),
        json.loads(args.selection.read_text()), args.bundle)
    result["sha256"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [args.dump, args.domain, args.selection, Path("experiments/pdc/R5_PREREG.md") ,
         Path(str(args.bundle)+".fieldrun.json"), Path(str(args.bundle)+".fieldrun.bin")]}
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(result["summary"], indent=2))
    if result["summary"]["reference_agreement"] != 1:
        raise SystemExit("faithfulness gate failed; retain the artifact and investigate")


if __name__ == "__main__":
    main()
