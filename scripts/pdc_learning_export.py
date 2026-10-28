"""Export fixed R7 computations and exact fixed-outside reductions for PIL."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def export(r5, r6, domain, probe, stem):
    manifest = json.loads(Path(str(stem)+".fieldrun.json").read_text())
    blob = np.memmap(str(stem)+".fieldrun.bin",mode="r",dtype=np.uint8)
    arrays = {}
    for entry in manifest["arrays"]:
        if entry["name"] in ["wte","ln_f.bias","ln_f.weight"]:
            arrays[entry["name"]] = np.frombuffer(blob,dtype={"f16":"<f2","f32":"<f4"}[entry["dtype"]],
                count=np.prod(entry["shape"]),offset=entry["offset"]).reshape(entry["shape"])
    native = r5["cases"]+r6["cases"]
    assert len(r5["cases"]) == 75 and len(r6["cases"]) == 108
    cases = [c for c in domain["cases"] if c["split"] == "r5"]
    assert all(c["ids"] == n["ids"] for c,n in zip(cases,r5["cases"],strict=True))
    tokens = sorted(set(probe["tokens"]) | {n["predictions"][0] for n in native})
    token_index = {token:i for i,token in enumerate(tokens)}
    beta = arrays["ln_f.bias"].astype(float)
    residuals, sources = [], []
    for index,n in enumerate(native):
        hidden = np.asarray(n["final_hidden"][-1],dtype=np.float32).astype(float)
        residuals.append(hidden-beta)
        if index < 75:
            raw = np.asarray([s["write"] for s in n["sources"]],dtype=np.float32).astype(float)
            assert raw.shape == (25,len(beta))
            end = np.asarray(n["reads"][0]["final_residual"],dtype=np.float32).astype(float)
            sigma = np.sqrt(np.mean((end-end.mean())**2)+1e-5)
            sources.append((raw-raw.mean(axis=1,keepdims=True))*arrays["ln_f.weight"].astype(float)/sigma)
    residuals, sources = np.array(residuals),np.array(sources)
    hidden = residuals+beta
    outside = sorted(set(range(len(arrays["wte"]))) - set(tokens))
    outside_max = np.full(len(native),-np.inf)
    outside_lse = np.full(len(native),-np.inf)
    outside_argmax = np.full(len(native),-1,dtype=np.int64)
    for start in range(0,len(outside),1024):
        selected = outside[start:start+1024]
        scores = hidden @ arrays["wte"][selected].astype(float).T
        best = scores.argmax(axis=1)
        values = scores[np.arange(len(native)),best]
        better = values > outside_max
        outside_argmax[better] = np.asarray(selected)[best[better]]
        outside_max = np.maximum(outside_max,values)
        maximum = scores.max(axis=1)
        lse = maximum+np.log(np.exp(scores-maximum[:,None]).sum(axis=1))
        outside_lse = np.logaddexp(outside_lse,lse)
    frame = arrays["wte"][tokens].astype(float)
    bias = frame@beta
    initial = residuals@frame.T+bias
    teachers = np.array([token_index[n["predictions"][0]] for n in native],dtype=np.int64)
    target_scores = initial[np.arange(len(native)),teachers]
    masked = initial.copy()
    masked[np.arange(len(native)),teachers] = -np.inf
    margins = target_scores-np.maximum(masked.max(axis=1),outside_max)
    truth = np.array([token_index[domain["token_map"][c["theory_atom"]]] for c in cases],dtype=np.int64)
    output = {"native_sources":sources,"protected_residuals":residuals,"initial_U":frame,"initial_bias":bias,
        "pool_tokens":np.array(tokens,dtype=np.int64),"teacher_indices":teachers,"truth_indices":truth,
        "outside_max":outside_max,"outside_lse":outside_lse,"outside_argmax":outside_argmax,
        "initial_global_margins":margins,"numeric_closure":residuals[:75]-sources.sum(axis=1),
        "native_teacher_tokens":np.array([n["predictions"][0] for n in native],dtype=np.int64)}
    report = {"schema":"pdc.learning-data.v1","training_contexts":75,"protected_contexts":len(native),
        "pool_rows":len(tokens),"outside_rows":len(outside),"heldout_contexts_used":0,
        "initial_full_host_agreement":float(np.mean(margins>0)),"minimum_global_margin":float(margins.min()),
        "initial_gate":"passed" if np.all(margins>0) else "inapplicable",
        "mismatches":np.flatnonzero(margins<=0).tolist(),
        "numeric_closure_max_abs":float(np.max(np.abs(output["numeric_closure"]))),
        "outside_semantics":"maximum for decisions; logsumexp for cross-entropy; all outside rows fixed"}
    return output,report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["r5","r6","domain","probe","bundle","out"]:
        parser.add_argument("--"+name,type=Path,required=True)
    args = parser.parse_args()
    def read(path):
        return json.loads(path.read_text())
    data,report = export(read(args.r5),read(args.r6),read(args.domain),read(args.probe),args.bundle)
    report["sha256"] = {str(p):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [args.r5,args.r6,args.domain,args.probe,Path(str(args.bundle)+".fieldrun.json"),Path(str(args.bundle)+".fieldrun.bin")]}
    np.savez_compressed(args.out,**data)
    report["npz_sha256"] = hashlib.sha256(args.out.read_bytes()).hexdigest()
    args.out.with_suffix(".json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
