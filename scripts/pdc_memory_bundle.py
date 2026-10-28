"""Export the two-hop provenance graph without another model forward."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from pdc_service import load_readout


def build(capture, requests, results, first_capture, domain, stem):
    first_cases = [c for c in domain["cases"] if c["split"] == "r5"]
    first = {c["id"]:n for c,n in zip(first_cases,first_capture["cases"],strict=True)}
    answer_tokens = [domain["token_map"][a] for a in domain["rivals"]]
    tokens = sorted(set(answer_tokens+[r["full_host_token"] for r in results["records"]]+
        [w["target_token"] for w in requests["admissions"].values()]))
    _, arrays = load_readout(stem,tokens)
    frame,weight,beta = (arrays[name] for name in ["wte","ln_f.weight","ln_f.bias"])
    bias = frame@beta
    lookup = {t:i for i,t in enumerate(tokens)}

    def node(native, truth_token, source_ref):
        position = len(native["ids"])-1
        writes = [s for s in native["sources"] if s["position"] == position]
        raw = np.asarray([s["write"] for s in writes],dtype=np.float32).astype(float)
        residual = np.asarray(native["reads"][0]["final_residual"],dtype=np.float32).astype(float)
        scale = np.sqrt(np.mean((residual-residual.mean())**2)+1e-5)
        total = ((raw-raw.mean(axis=1,keepdims=True))*weight/scale).sum(axis=0)
        predicted = native["predictions"][0]
        candidates = answer_tokens+([] if predicted in answer_tokens else [predicted])
        indices = [lookup[t] for t in candidates]
        scores = frame[indices]@total+bias[indices]
        ti = candidates.index(predicted)
        margin = float(scores[ti]-np.max(np.delete(scores,ti)))
        return {"context_ids":native["ids"],"position":position,"token":predicted,
            "readout":"viterbi","stratum":"decode","body":{"id":"full_sources",
                "source_ids":[s["id"] for s in writes],"folded_coalition_write":total.tolist(),
                "source_capture_ref":source_ref},
            "covered_candidates":candidates,"scores":scores.tolist(),"margin":margin,
            "decides_on_covered_candidates":margin>0,"fragile":False,
            "fragility_scope":"one-body full-source PIC decode fragment; derive chart is separate",
            "theory":{"verdict":"affirmed" if predicted == truth_token else "hallucination",
                "source":"bounded integer addition; external grading, not implied by margin"}}

    first_nodes = {}
    for key in requests["admissions"]:
        meta = next(c for c in first_cases if c["id"] == key)
        first_nodes[key] = node(first[key],domain["token_map"][str(meta["a"]+meta["b"])],
            {"artifact":"r5_reads.json","case_index":first_cases.index(meta)})
        first_nodes[key]["operands"] = [meta["a"],meta["b"]]
    paths = []
    for index,(native,request,result) in enumerate(zip(capture["cases"],requests["requests"],results["records"],strict=True)):
        assert request["id"] == result["id"]
        edge = {"kind":request["memory_tag"],"admission":request["admission"]}
        if request["appended_token"] is not None:
            edge.update({"kind":"appended","tag":request["arm"],"token":request["appended_token"],
                "position":request["memory_position"],
                "source":"external_theory" if request["arm"] == "tool" else "hop1_chart",
                "theory_affirms":request["appended_token"] == domain["token_map"][str(request["a"]+request["b"])]})
        elif request["memory"] is not None:
            edge.update({"kind":"residual","site":request["memory"]["site"],
                "admission_ref":request["first_hop_id"]})
        else:
            edge["reason"] = "oracle_refused" if request["arm"] == "crystallize" else "rederive"
        paths.append({"id":request["id"],"operands":[request["a"],request["b"],request["c"]],
            "hop1_ref":request["first_hop_id"],"memory_edge":edge,"derive_fragile":result["derive_chart"]["fragile"],
            "hop2":node(native,domain["token_map"][str(request["a"]+request["b"]+request["c"])],
                {"artifact":"r6_capture.json","case_index":index}),
            "derive_chart":result["derive_chart"],"read_heads_to_memory":result["memory_reads"]})
    return {"schema":"pdc.two-hop-bundle.v1","status":"experimental trace; not a served epistemic certificate",
        "frame_tokens":tokens,"frame":frame.tolist(),"bias":bias.tolist(),
        "theory":{"name":"bounded integer addition","digit_tokens":domain["token_map"]},
        "admission_artifact":"r6_requests.json",
        "frame_convention":"measured final-LN-folded coalition writes; fixed native unembedding",
        "first_hop_nodes":first_nodes,"paths":paths,
        "summary":{"first_hop_nodes":len(first_nodes),"paths":len(paths),
            "tool_edges":sum(p["memory_edge"].get("tag") == "tool" for p in paths),
            "generated_edges":sum(p["memory_edge"].get("tag") == "generated" for p in paths),
            "raw_global_decode_full_vocabulary_agreement_evidence":"r6_results.json independent reference, 108/108",
            "margin_scope":"explicit finite covered candidates; not a global-vocabulary certificate"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["capture","requests","results","first-capture","domain","bundle","out"]:
        parser.add_argument("--"+name,type=Path,required=True)
    args = parser.parse_args()
    def read(path):
        return json.loads(path.read_text())
    result = build(read(args.capture),read(args.requests),read(args.results),read(args.first_capture),read(args.domain),args.bundle)
    result["sha256"] = {str(p):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [args.capture,args.requests,args.results,args.first_capture,args.domain]}
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps(result["summary"],indent=2))


if __name__ == "__main__":
    main()
