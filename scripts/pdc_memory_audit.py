"""Independently check R6 witnesses, four-arm assembly and finite outcomes."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from pdc_derive import Body, evaluate
from pdc_read_check import load_bundle, reference


def audit(capture, requests, selection, first_capture, domain, stem):
    model, w = load_bundle(stem)
    first_cases = [c for c in domain["cases"] if c["split"] == "r5"]
    first = {c["id"]:n for c,n in zip(first_cases,first_capture["cases"],strict=True)}
    assert len(capture["cases"]) == len(requests["requests"]) == len(selection["contexts"]) == 108
    answer_tokens = [domain["token_map"][a] for a in domain["rivals"]]
    for key, witness in requests["admissions"].items():
        n = first[key]
        tokens = answer_tokens+([] if n["predictions"][0] in answer_tokens else [n["predictions"][0]])
        assert witness["candidates"] == tokens and witness["target_token"] == n["predictions"][0]
        frame = w["wte"][tokens].astype(float)
        beta,weight = w["ln_f.bias"].astype(float),w["ln_f.weight"].astype(float)
        assert np.array_equal(frame,witness["frame"])
        raw = np.array([s["write"] for s in n["sources"]],dtype=np.float32).astype(float)
        residual = np.array(n["reads"][0]["final_residual"],dtype=np.float32).astype(float)
        sigma = np.sqrt(np.mean((residual-residual.mean())**2)+1e-5)
        folded = (raw-raw.mean(axis=1,keepdims=True))*weight/sigma
        assert np.max(np.abs(folded-witness["folded_sources"])) < 1e-8
        bias = frame@beta
        target = tokens.index(n["predictions"][0])
        full = folded.sum(axis=0)@frame.T+bias
        margin_full = full[target]-np.max(np.delete(full,target))
        assert abs(margin_full-witness["full_margin"]) < 1e-8
        if witness["status"] == "unknown":
            assert margin_full <= 0
            continue
        assert margin_full > 0
        admitted = None
        for i, attempt in enumerate(witness["attempts"]):
            assert attempt["source_index"] == i and attempt["source"] == n["sources"][i]["id"]
            partial = folded[i]@frame.T+bias
            omitted = (folded.sum(axis=0)-folded[i])@frame.T
            margin = float(partial[target]-np.max(np.delete(partial,target)))
            delta = float(np.ptp(omitted)/2)
            gamma = (margin+max(.5,2*delta))/2
            for value, stored in [(margin,attempt["margin"]),(delta,attempt["delta"]),(gamma,attempt["gamma"])]:
                assert abs(value-stored) < 1e-8
            accept = margin > gamma > max(.5,2*delta)
            assert accept == attempt["admitted"]
            if accept:
                assert i == len(witness["attempts"])-1
                admitted = {"site":n["sources"][i]["id"],"write":raw[i].tolist()}
                break
        assert (witness["status"] == "admitted") == (admitted is not None)
        assert witness["memory"] == admitted
        if witness["status"] == "refused":
            assert len(witness["attempts"]) == 25
    results, by_triple, ref_matches = [], {}, 0
    frame9 = w["wte"][answer_tokens].astype(float)
    bias9 = frame9@w["ln_f.bias"].astype(float)
    for native, request, ids, memory in zip(capture["cases"],requests["requests"],
        selection["contexts"],selection["memory_writes"],strict=True):
        assert native["ids"] == request["ids"] == ids
        assert native["memory_replay"] == request["memory"] == memory
        assert native["scoring_scope"] == "full_vocabulary"
        assert native["native_output_row_calls"] == model["config"][4]
        a,b,c,arm = (request[k] for k in ["a","b","c","arm"])
        h1 = first[request["first_hop_id"]]
        assert request["generated_intermediate_correct"] == (h1["predictions"][0] == domain["token_map"][str(a+b)])
        status = requests["admissions"][request["first_hop_id"]]["status"]
        assert status == request["admission"]
        if arm == "crystallize" and status == "admitted":
            assert memory == requests["admissions"][request["first_hop_id"]]["memory"]
        else:
            assert memory is None
        if arm == "generated":
            assert request["appended_token"] == h1["predictions"][0]
        if arm == "tool":
            assert request["appended_token"] == domain["token_map"][str(a+b)]
        if request["appended_token"] is not None:
            assert ids[:len(h1["ids"])] == h1["ids"]
            assert ids[request["memory_position"]] == request["appended_token"]
        predicted = native["predictions"][0]
        expected_token = domain["token_map"][str(a+b+c)]
        ref = reference(ids,model["config"],w,last_only=True,memory=memory)[0]
        ref_matches += predicted == ref
        source_ids = [s["id"] for s in native["sources"]]
        raw = np.array([s["write"] for s in native["sources"]],dtype=np.float32).astype(float)
        residual = np.array(native["reads"][0]["final_residual"],dtype=np.float32).astype(float)
        sigma = np.sqrt(np.mean((residual-residual.mean())**2)+1e-5)
        folded = (raw-raw.mean(axis=1,keepdims=True))*w["ln_f.weight"]/sigma
        programs = {"full":list(range(len(source_ids))),
            "attention":[i for i,s in enumerate(source_ids) if s.startswith("L") and s.endswith(".attn")],
            "mlp":[i for i,s in enumerate(source_ids) if s.startswith("L") and s.endswith(".mlp")]}
        body_scores = {name:(folded[coalition].sum(axis=0)@frame9.T+bias9).tolist()
            for name,coalition in programs.items()}
        chart = evaluate({atom:[Body(name,scores[j]) for name,scores in body_scores.items()]
            for j,atom in enumerate(domain["rivals"])},1.)
        memory_reads = []
        if request["memory_position"] is not None:
            for read in native["reads"]:
                memory_reads.append({"layer":read["layer"],"head":read["head"],
                    "probability":read["probabilities"][request["memory_position"]]})
            assert len(memory_reads) == 12
        result = {"id":request["id"],"a":a,"b":b,"c":c,"arm":arm,"admission":status,
            "readout":"viterbi","fragment":"native full-source one-body PIC decode",
            "full_host_token":predicted,"theory_affirms":predicted == expected_token,
            "reference_token":ref,"memory_tag":request["memory_tag"],"memory_reads":memory_reads,
            "body_programs":{k:[source_ids[i] for i in v] for k,v in programs.items()},
            "body_scores":body_scores,"derive_chart":chart,
            "source_sum_max_error":float(np.max(np.abs(raw.sum(axis=0)-residual)))}
        results.append(result)
        by_triple.setdefault((a,b,c),{})[arm] = (native,result,request)
    strata = {}
    for status in ["admitted","refused","unknown"]:
        triples = [arms for arms in by_triple.values() if arms["rederive"][1]["admission"] == status]
        group = {"n":len(triples),"accuracies":{},"paired":{}}
        for arm in ["rederive","crystallize","generated","tool"]:
            good = sum(t[arm][1]["theory_affirms"] for t in triples)
            group["accuracies"][arm] = {"correct":good,"rate":good/len(triples) if triples else None}
        for arm in ["crystallize","generated","tool"]:
            pairs = Counter((t["rederive"][1]["theory_affirms"],t[arm][1]["theory_affirms"]) for t in triples)
            group["paired"][arm] = {f"base_{int(k[0])}_arm_{int(k[1])}":v for k,v in pairs.items()}
        if triples:
            diff = group["accuracies"]["generated"]["rate"]-group["accuracies"]["rederive"]["rate"]
            group["generated_minus_rederive"] = diff
            group["refused_improvement_claim"] = "confirmed_on_slice" if status == "refused" and diff >= .05 else \
                "retired_on_slice" if status == "refused" else "not_applicable"
        else:
            group["generated_minus_rederive"] = None
            group["refused_improvement_claim"] = "untested"
        strata[status] = group
    refused_equal = 0
    for arms in by_triple.values():
        base,crystal = arms["rederive"][0],arms["crystallize"][0]
        if arms["rederive"][1]["admission"] != "admitted":
            assert base["ids"] == crystal["ids"] and base["sources"] == crystal["sources"]
            assert base["predictions"] == crystal["predictions"]
            np.testing.assert_array_equal(base["final_hidden"],crystal["final_hidden"])
            refused_equal += 1
        for arm in ["generated","tool"]:
            app = arms[arm][0]
            prefix = len(first[arms[arm][2]["first_hop_id"]]["ids"])
            assert app["ids"][:prefix] == base["ids"][:prefix]
            assert app["ids"][prefix+1:] == base["ids"][prefix:]
        if arms["generated"][2]["generated_intermediate_correct"]:
            assert arms["generated"][0]["ids"] == arms["tool"][0]["ids"]
            assert arms["generated"][0]["predictions"] == arms["tool"][0]["predictions"]
    return {"schema":"pdc.two-hop-result.v1","contexts":27,"arm_forwards":108,
        "reference_agreement":ref_matches/108,"refused_replay_identity_cases":refused_equal,
        "unique_admission_counts":requests["unique_admission_counts"],"strata":strata,
        "admitted_redundancy_claim":"untested" if not strata["admitted"]["n"] else "requires_paired_assessment",
        "generated_intermediate_correct_unique":sum(r["generated_intermediate_correct"]
            for r in requests["requests"] if r["arm"] == "rederive" and r["c"] == 0),
        "records":results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["capture","requests","selection","first-capture","domain","bundle","out"]:
        parser.add_argument("--"+name,type=Path,required=True)
    args = parser.parse_args()
    def load(path):
        return json.loads(path.read_text())
    result = audit(load(args.capture),load(args.requests),load(args.selection),load(args.first_capture),
        load(args.domain),args.bundle)
    result["sha256"] = {str(p):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [args.capture,args.requests,args.selection,args.first_capture,args.domain,Path("experiments/pdc/R6_PREREG.md"),
                  Path(str(args.bundle)+".fieldrun.json"),Path(str(args.bundle)+".fieldrun.bin")]}
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ["records","sha256"]},indent=2))
    if result["reference_agreement"] != 1:
        raise SystemExit("faithfulness gate failed; retain the report and investigate")


if __name__ == "__main__":
    main()
