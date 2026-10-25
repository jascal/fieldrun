"""Freeze R5 addition theory inputs without executing a model."""
import argparse
import hashlib
import json
from pathlib import Path

from tokenizers import Tokenizer


def build(tokenizer_path):
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    token_map = {}
    for n in range(9):
        encoded = tokenizer.encode(" " + str(n), add_special_tokens=False).ids
        if len(encoded) != 1:
            raise ValueError("each answer must be a single leading-space token")
        token_map[str(n)] = encoded[0]
    if len(set(token_map.values())) != 9:
        raise ValueError("answer tokens must be distinct")
    examples = "0 + 0 = 0\n1 + 1 = 2\n2 + 2 = 4\n3 + 3 = 6\n4 + 4 = 8\n"
    prefixes = ["", "Complete the addition equations.\n", "Use ordinary integer addition.\n",
                "Solve the final equation.\n"]
    cases = []
    for template, prefix in enumerate(prefixes):
        for a in range(5):
            for b in range(5):
                prompt = prefix + examples + f"{a} + {b} ="
                cases.append({"id": f"t{template}-a{a}-b{b}", "template": template,
                    "a": a, "b": b, "theory_atom": str(a+b), "prompt": prompt,
                    "ids": tokenizer.encode(prompt, add_special_tokens=False).ids,
                    "split": "heldout" if template == 3 else "r5"})
    return {"schema": "pdc.addition-theory.v1", "theory": "bounded integer addition",
        "rivals": list(token_map), "token_map": token_map, "cases": cases,
        "tokenizer_sha256": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
        "gamma": 0.5, "hallucination_bar": 0.1, "wide_miss_bar": 0.25,
        "temperature": 1.0, "body_ids": ["full", "attention", "mlp"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    args = parser.parse_args()
    domain = build(args.tokenizer)
    selection = {"layers": [11], "heads": [0], "positions": [], "last_position": True,
        "capture_sources": True, "last_prediction_only": True,
        "contexts": [c["ids"] for c in domain["cases"] if c["split"] == "r5"],
        "candidate_tokens": [domain["token_map"][a] for a in domain["rivals"]]}
    for path, data in [(args.out, domain), (args.selection, selection)]:
        encoded = json.dumps(data, indent=2) + "\n"
        if path.exists() and path.read_text() != encoded:
            raise ValueError(f"refusing to replace a frozen input: {path}")
        path.write_text(encoded)
        print(path, hashlib.sha256(path.read_bytes()).hexdigest())
    print("75 capture contexts; 25 held-out manifests retained without capture")


if __name__ == "__main__":
    main()
