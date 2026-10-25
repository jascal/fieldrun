"""R3 fixed trigram-copy body generator over a native PDC read dump."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from pdc_derive import Body, evaluate
from pdc_read_check import check, load_bundle


def build(dump, selection, stem):
    read_check = check(dump, selection, stem)
    if read_check["reference_agreement"] != 1:
        raise ValueError("native/reference faithfulness failed")
    _, w = load_bundle(stem)
    rivals = list(range(1000, 1008))
    frame = w["wte"][rivals].astype(np.float64)
    gamma, beta = (w[name].astype(np.float64) for name in ["ln_f.weight", "ln_f.bias"])
    bias = frame @ beta
    charts = []
    for case_id, case in enumerate(dump["cases"]):
        ids = case["ids"]
        i = len(ids) - 1
        routes = [j for j in range(3, i + 1) if ids[j-3:j] == ids[i-2:i+1]]
        if not routes:
            raise ValueError("preregistered task has no previous trigram continuation")
        body_table = {str(a): [] for a in rivals}
        sources = []
        metadata = []
        residual = np.asarray(case["reads"][0]["final_residual"], dtype=np.float64)
        sigma = np.sqrt(np.mean((residual - residual.mean())**2) + 1e-5)
        fold = lambda d: (d - d.mean()) * gamma / sigma
        for read in case["reads"]:
            assert read["query_position"] == i
            assert np.array_equal(residual, read["final_residual"])
            layer, head = read["layer"], read["head"]
            v = np.asarray(read["values"], dtype=np.float64)
            probs = read["probabilities"]
            hd = v.shape[1]
            projection = w[f"h{layer}.attn.c_proj.weight"][head*hd:(head+1)*hd].astype(np.float64)
            for j in routes:
                atom = ids[j]
                source_id = f"L{layer}H{head}K{j}"
                raw_write = probs[j] * (v[j] @ projection)
                write = fold(raw_write)
                sources.append({"id": source_id, "write": write.tolist()})
                index = rivals.index(atom)
                incidence = float(write @ frame[index])
                bracket = incidence + float(bias[index])
                body = Body(source_id, bracket)
                body_table[str(atom)].append(body)
                metadata.append({"atom": str(atom), "body": asdict(body), "sources": [source_id],
                    "layer": layer, "head": head, "query_position": i, "key_position": j,
                    "attention_probability": probs[j], "incidence": incidence, "bias": float(bias[index])})
        # A fixed complement preserves the measured full residual for consumers.
        source_sum = np.sum([s["write"] for s in sources], axis=0)
        sources.append({"id": "complement", "write": (fold(residual) - source_sum).tolist()})
        folded_hidden = np.sum([s["write"] for s in sources], axis=0) + beta
        hidden = np.asarray(case["final_hidden"][i], dtype=np.float64)
        scores = hidden @ frame.T
        restricted = [str(rivals[j]) for j in np.flatnonzero(scores == scores.max())]
        chart = evaluate(body_table, 1)
        chart.update({"case": case_id, "context_ids": ids, "query_position": i,
            "host_token": str(case["predictions"][i]), "host_restricted_winners": restricted,
            "task_label": str(ids[routes[-1]]), "copy_keys": routes,
            "body_metadata": metadata, "sources": sources,
            "full_residual": fold(residual).tolist(),
            "final_norm_folding_max_error": float(np.max(np.abs(folded_hidden - hidden))),
            "memory": "none", "theory": "not_loaded", "certificate": "not_evaluated"})
        charts.append(chart)
    summary = {"n": len(charts), "body_count": sum(len(c["body_metadata"]) for c in charts),
        "fragility_rate": sum(c["fragile"] for c in charts)/len(charts),
        "final_norm_folding_max_error": max(c["final_norm_folding_max_error"] for c in charts),
        "read_check": {k: v for k, v in read_check.items() if k != "instances"}}
    for readout in ["viterbi", "marginal"]:
        get = lambda c: next(q for q in c["queries"] if q["readout"] == readout)
        summary[readout] = {
            "full_host_agreement": sum(get(c)["winners"] == [c["host_token"]] for c in charts)/len(charts),
            "restricted_host_agreement": sum(get(c)["winners"] == c["host_restricted_winners"] for c in charts)/len(charts),
            "task_agreement": sum(get(c)["winners"] == [c["task_label"]] for c in charts)/len(charts),
            "ties": sum(get(c)["status"] == "tie" for c in charts)}
    return {"schema": "pdc.copy-chart.v1", "rivals": [str(a) for a in rivals],
        "frame": frame.tolist(), "bias": bias.tolist(),
        "frame_convention": "native final-LN folding, beta read as once-per-body bias; frozen measured scale",
        "summary": summary, "charts": charts}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dump", type=Path, required=True)
    p.add_argument("--selection", type=Path, required=True)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    result = build(json.loads(args.dump.read_text()), json.loads(args.selection.read_text()), args.bundle)
    result["sha256"] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [args.dump, args.selection, Path("experiments/pdc/R3_PREREG.md")]}
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
