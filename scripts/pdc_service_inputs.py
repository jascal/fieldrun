"""Freeze lexical probe coverage and R8-H runtime inputs without model calls."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from tokenizers import Tokenizer


def probe_ids(tokenizer, answer_tokens):
    found = set(answer_tokens) | set(tokenizer.get_added_tokens_decoder())
    for token in range(tokenizer.get_vocab_size()):
        text = tokenizer.decode([token], skip_special_tokens=False)
        ascii_only = text and all(ord(c) < 128 for c in text)
        short_nonletter = len(text) <= 3 and not re.search("[A-Za-z]", text)
        letter = re.fullmatch(" ?[A-Za-z]", text) is not None
        if ascii_only and (len(text) == 1 or short_nonletter or letter):
            found.add(token)
    return sorted(found)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--domain", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    domain = json.loads(args.domain.read_text())
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    tokens = probe_ids(tokenizer, domain["token_map"].values())
    probe = {"schema": "pdc.lexical-probe.v1", "tokens": tokens,
        "tokenizer_sha256": hashlib.sha256(args.tokenizer.read_bytes()).hexdigest(),
        "rule": "R8H_PREREG.md lexical rule", "vocabulary_size": tokenizer.get_vocab_size()}
    selection = {"layers": [11], "heads": [0], "positions": [], "last_position": True,
        "capture_sources": True, "last_prediction_only": True, "candidate_tokens": tokens,
        "check_observer": False,
        "contexts": [c["ids"] for c in domain["cases"] if c["split"] == "r5"]}
    outputs = {"r8h_probe.json": probe,
        "r8h_baseline_selection.json": {**selection, "score_candidates_only": False},
        "r8h_service_selection.json": {**selection, "score_candidates_only": True}}
    for name, data in outputs.items():
        path = args.out_dir / name
        encoded = json.dumps(data, indent=2)+"\n"
        if path.exists() and path.read_text() != encoded:
            raise ValueError(f"refusing to replace frozen input: {path}")
        path.write_text(encoded)
        print(path, hashlib.sha256(path.read_bytes()).hexdigest())
    print(f"probe rows {len(tokens)} / {tokenizer.get_vocab_size()}")


if __name__ == "__main__":
    main()
