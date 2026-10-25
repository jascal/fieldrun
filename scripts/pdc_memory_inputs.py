"""Freeze R6 singleton admission witnesses and all exact memory requests."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from pdc_service import load_readout


def admit(native, answer_tokens, stem):
    target = native["predictions"][0]
    candidates = list(answer_tokens)
    if target not in candidates:
        candidates.append(target)
    _, arrays = load_readout(stem,candidates)
    frame, gamma_ln, beta = (arrays[name] for name in ["wte","ln_f.weight","ln_f.bias"])
    position = len(native["ids"])-1
    sources = [s for s in native["sources"] if s["position"] == position]
    if len(sources) != 25:
        raise ValueError("hop-1 capture must contain all 25 native sources")
    raw = np.array([s["write"] for s in sources],dtype=np.float32).astype(float)
    residual = np.array(native["reads"][0]["final_residual"],dtype=np.float32).astype(float)
    sigma = np.sqrt(np.mean((residual-residual.mean())**2)+1e-5)
    folded = (raw-raw.mean(axis=1,keepdims=True))*gamma_ln/sigma
    bias = frame@beta
    full = frame@folded.sum(axis=0)+bias
    ti = candidates.index(target)
    full_margin = float(full[ti]-np.max(np.delete(full,ti)))
    result = {"target_token":target,"candidates":candidates,"frame":frame.tolist(),"bias":bias.tolist(),
        "folded_sources":folded.tolist(),"source_ids":[s["id"] for s in sources],
        "full_margin":full_margin,"attempts":[],"memory":None}
    if not np.isfinite(full).all() or full_margin <= 0:
        return {**result,"status":"unknown"}
    for index, source in enumerate(sources):
        partial = frame@folded[index]+bias
        tail = frame@(folded.sum(axis=0)-folded[index])
        margin = float(partial[ti]-np.max(np.delete(partial,ti)))
        delta = float((tail.max()-tail.min())/2)
        kappa = float((tail.max()+tail.min())/2)
        gamma = (margin+max(.5,2*delta))/2
        admitted = margin > gamma > max(.5,2*delta)
        result["attempts"].append({"source":source["id"],"source_index":index,"margin":margin,
            "delta":delta,"kappa":kappa,"gamma":gamma,"admitted":admitted,
            "coalition_scores":partial.tolist(),"tail_scores":tail.tolist()})
        if admitted:
            result.update({"status":"admitted","memory":{"site":source["id"],"write":raw[index].tolist()}})
            return result
    return {**result,"status":"refused"}


def build(domain, first_capture, tokenizer_path, stem):
    first_cases = [c for c in domain["cases"] if c["split"] == "r5"]
    if len(first_cases) != len(first_capture["cases"]):
        raise ValueError("frozen first-hop capture/domain mismatch")
    by_id = {c["id"]:n for c,n in zip(first_cases,first_capture["cases"],strict=True)}
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    answer_tokens = [domain["token_map"][a] for a in domain["rivals"]]
    admissions = {}
    for a in range(3):
        for b in range(3):
            key = f"t0-a{a}-b{b}"
            admissions[key] = admit(by_id[key],answer_tokens,stem)
    requests = []
    for a in range(3):
        for b in range(3):
            key = f"t0-a{a}-b{b}"
            first = by_id[key]
            admission = admissions[key]
            for c in range(3):
                suffix = tokenizer.encode(f"\n{a} + {b} + {c} =",add_special_tokens=False).ids
                for arm in ["rederive","crystallize","generated","tool"]:
                    ids = list(first["ids"])
                    token, tag = None, "none"
                    if arm in ["generated","tool"]:
                        token = first["predictions"][0] if arm == "generated" else domain["token_map"][str(a+b)]
                        ids.append(token)
                        tag = arm
                    memory = admission["memory"] if arm == "crystallize" and admission["status"] == "admitted" else None
                    requests.append({"id":f"a{a}-b{b}-c{c}-{arm}","a":a,"b":b,"c":c,"arm":arm,
                        "first_hop_id":key,"admission":admission["status"],"ids":ids+suffix,
                        "theory_atom":str(a+b+c),"intermediate_theory_atom":str(a+b),
                        "memory_tag":"residual" if memory is not None else tag,"memory":memory,
                        "appended_token":token,"memory_position":len(first["ids"]) if token is not None else None,
                        "generated_intermediate_correct":first["predictions"][0] == domain["token_map"][str(a+b)]})
    selection = {"layers":[0,5,11],"heads":[0,3,7,11],"positions":[],"last_position":True,
        "capture_sources":True,"last_prediction_only":True,"candidate_tokens":answer_tokens,
        "check_observer":False,"score_candidates_only":False,
        "contexts":[r["ids"] for r in requests],"memory_writes":[r["memory"] for r in requests]}
    base = by_id["t0-a0-b0"]["ids"]
    zero = {"site":"L0.mlp","write":[0.]*768}
    nonzero = {"site":"L0.mlp","write":[.125 if i%2 == 0 else -.125 for i in range(768)]}
    engineering = {**selection,"contexts":[base,base,base],"memory_writes":[None,zero,nonzero]}
    manifest = {"schema":"pdc.two-hop-requests.v1","admissions":admissions,"requests":requests,
        "unique_admission_counts":dict(Counter(r["status"] for r in admissions.values())),
        "expanded_admission_counts":dict(Counter(r["admission"] for r in requests if r["arm"] == "rederive")),
        "engineering_names":["baseline","zero_replay","nonzero_replay"]}
    return manifest,selection,engineering


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["domain","first-capture","tokenizer","bundle","out-dir"]:
        parser.add_argument("--"+name,type=Path,required=True)
    args = parser.parse_args()
    manifest,selection,engineering = build(json.loads(args.domain.read_text()),
        json.loads(args.first_capture.read_text()),args.tokenizer,args.bundle)
    hashes = {}
    for name,data in [("r6_requests.json",manifest),("r6_selection.json",selection),
                      ("r6_engineering_selection.json",engineering)]:
        path = args.out_dir/name
        encoded = json.dumps(data,indent=2,allow_nan=False)+"\n"
        if path.exists() and path.read_text() != encoded:
            raise ValueError(f"refusing to replace a frozen input: {path}")
        path.write_text(encoded)
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    hashes["R6_PREREG.md"] = hashlib.sha256((args.out_dir/"R6_PREREG.md").read_bytes()).hexdigest()
    (args.out_dir/"r6_input_hashes.json").write_text(json.dumps(hashes,indent=2)+"\n")
    print(json.dumps({"unique":manifest["unique_admission_counts"],"expanded":manifest["expanded_admission_counts"],
        "requests":len(manifest["requests"]),"hashes":hashes},indent=2))


if __name__ == "__main__":
    main()
