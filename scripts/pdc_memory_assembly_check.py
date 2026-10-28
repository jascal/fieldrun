"""Check exact R6 token assembly independently of its input generator."""
import argparse
import itertools
import json
from pathlib import Path

from tokenizers import Tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["requests","domain","tokenizer","out"]:
        parser.add_argument("--"+name,type=Path,required=True)
    args = parser.parse_args()
    manifest = json.loads(args.requests.read_text())
    domain = json.loads(args.domain.read_text())
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    by_id = {c["id"]:c for c in domain["cases"]}
    arms = ["rederive","crystallize","generated","tool"]
    expected = set(itertools.product(range(3),range(3),range(3),arms))
    actual = {(r["a"],r["b"],r["c"],r["arm"]) for r in manifest["requests"]}
    assert len(manifest["requests"]) == len(actual) == 108 and actual == expected
    for request in manifest["requests"]:
        a,b,c,arm = (request[k] for k in ["a","b","c","arm"])
        key = f"t0-a{a}-b{b}"
        assert request["first_hop_id"] == key
        prefix = by_id[key]["ids"]
        suffix = tokenizer.encode(f"\n{a} + {b} + {c} =",add_special_tokens=False).ids
        token = None
        if arm == "generated":
            token = manifest["admissions"][key]["target_token"]
        elif arm == "tool":
            token = domain["token_map"][str(a+b)]
        assert request["appended_token"] == token
        assert request["ids"] == prefix+([] if token is None else [token])+suffix
        assert request["theory_atom"] == str(a+b+c)
        assert request["intermediate_theory_atom"] == str(a+b)
        assert by_id[key]["split"] == "r5"
    report = {"schema":"pdc.memory-assembly-check.v1","requests_checked":108,
        "tool_atoms_theory_checked":27,"heldout_contexts_used":0,"complete_triple_arm_grid":True}
    args.out.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
