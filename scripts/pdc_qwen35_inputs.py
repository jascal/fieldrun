"""Freeze new Qwen3.5 chat inputs and text-only probe coverage without inference."""

import argparse
import hashlib
import json
from pathlib import Path

from tokenizers import Tokenizer
from transformers import AutoTokenizer

from pdc_service_inputs import probe_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    native = Tokenizer.from_file(str(args.model / "tokenizer.json"))
    token_map = {}
    for n in range(9):
        ids = tokenizer.encode(str(n), add_special_tokens=False)
        assert len(ids) == 1
        token_map[str(n)] = ids[0]
    tokens = probe_ids(native, token_map.values())
    templates = [
        "Compute {a} + {b}. Reply with the single digit answer only.",
        "Return only the one-digit sum of {a} and {b}.",
        "Add the integers {a} and {b}. Your entire reply must be one digit.",
        "What is {a} plus {b}? Give just the digit, without words.",
    ]

    def case(identifier, text, **extra):
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        ids = tokenizer.encode(rendered, add_special_tokens=False)
        assert ids == native.encode(rendered, add_special_tokens=False).ids
        return dict(id=identifier, prompt=text, rendered=rendered, ids=ids, **extra)

    cases = [
        case(
            f"q35-t{j}-a{a}-b{b}",
            template.format(a=a, b=b),
            a=a,
            b=b,
            truth=str(a + b),
            split="development" if j < 2 else "heldout",
        )
        for j, template in enumerate(templates)
        for a in range(5)
        for b in range(5)
    ]
    engineering = [
        case(
            f"engineering-{n}",
            f"Reply with one digit only: {n} + {n} =",
            a=n,
            b=n,
            truth=str(2 * n),
            split="engineering",
        )
        for n in range(5)
    ]
    outputs = {
        "domain.json": dict(
            schema="pdc.qwen35-domain.v1",
            cases=cases,
            engineering=engineering,
            token_map=token_map,
            templates=templates,
            gamma=0.5,
            temperature=1.0,
            thinking=False,
        ),
        "probe.json": dict(
            tokens=tokens, rule="unchanged R8H tokenizer-text lexical rule"
        ),
        "selection.json": dict(
            contexts=[c["ids"] for c in cases],
            candidate_tokens=tokens,
            check_observer=False,
        ),
        "engineering_selection.json": dict(
            contexts=[c["ids"] for c in engineering],
            candidate_tokens=tokens,
            check_observer=True,
        ),
        "tiny_selection.json": dict(
            contexts=[[1, 2, 3, 4], [7, 8, 9, 10], [20, 21, 22, 23]],
            candidate_tokens=list(range(64)),
            check_observer=True,
        ),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    for name, data in outputs.items():
        path = args.out / name
        assert not path.exists(), f"refusing to overwrite frozen input {path}"
        path.write_text(json.dumps(data, indent=2) + "\n")
        print(name, hashlib.sha256(path.read_bytes()).hexdigest())
    print("probe rows", len(tokens), "contexts", len(cases))


if __name__ == "__main__":
    main()
