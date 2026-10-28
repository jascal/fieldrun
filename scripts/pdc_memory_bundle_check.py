"""Check the exported R6 graph's affine judgments and memory provenance."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def check(bundle):
    frame,bias = np.array(bundle["frame"]),np.array(bundle["bias"])
    index = {t:i for i,t in enumerate(bundle["frame_tokens"])}
    digit_tokens = bundle["theory"]["digit_tokens"]
    node_count,edges = 0,{"generated":0,"tool":0,"residual":0,"none":0}
    max_error = 0.

    def verify_node(node, truth):
        nonlocal node_count,max_error
        assert node["readout"] == "viterbi" and node["stratum"] == "decode"
        assert node["body"]["id"] == "full_sources" and node["body"]["source_ids"]
        assert len(set(node["body"]["source_ids"])) == len(node["body"]["source_ids"])
        values = frame[[index[t] for t in node["covered_candidates"]]]@np.array(node["body"]["folded_coalition_write"])+\
            bias[[index[t] for t in node["covered_candidates"]]]
        error = float(np.max(np.abs(values-node["scores"])))
        assert error < 1e-8
        max_error = max(max_error,error)
        target = node["covered_candidates"].index(node["token"])
        margin = float(values[target]-np.max(np.delete(values,target)))
        assert abs(margin-node["margin"]) < 1e-8
        assert (margin>0) == node["decides_on_covered_candidates"]
        assert node["theory"]["verdict"] == ("affirmed" if node["token"] == digit_tokens[str(truth)] else "hallucination")
        assert node["fragile"] is False
        node_count += 1

    assert len(bundle["first_hop_nodes"]) == 9 and len(bundle["paths"]) == 108
    for node in bundle["first_hop_nodes"].values():
        verify_node(node,sum(node["operands"]))
    for path in bundle["paths"]:
        a,b,c = path["operands"]
        first = bundle["first_hop_nodes"][path["hop1_ref"]]
        assert first["operands"] == [a,b]
        verify_node(path["hop2"],a+b+c)
        edge = path["memory_edge"]
        if edge["kind"] == "appended":
            tag = edge["tag"]
            edges[tag] += 1
            assert path["hop2"]["context_ids"][edge["position"]] == edge["token"]
            if tag == "generated":
                assert edge["source"] == "hop1_chart" and edge["token"] == first["token"]
            else:
                assert tag == "tool" and edge["source"] == "external_theory"
                assert edge["token"] == digit_tokens[str(a+b)] and edge["theory_affirms"]
        else:
            edges[edge["kind"]] += 1
            if edge["kind"] == "residual":
                assert edge["admission"] == "admitted"
    assert edges == {"generated":27,"tool":27,"residual":0,"none":54}
    return {"schema":"pdc.two-hop-bundle-check.v1","nodes_checked":node_count,"paths_checked":108,
        "edges":edges,"maximum_affine_score_error":max_error,
        "scope":"finite packet judgments and provenance; full native/reference checks remain r6_results.json"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args = parser.parse_args()
    result = check(json.loads(args.bundle.read_text()))
    result["bundle_sha256"] = hashlib.sha256(args.bundle.read_bytes()).hexdigest()
    args.out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
