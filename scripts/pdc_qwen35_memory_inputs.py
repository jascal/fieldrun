"""Freeze exact four-arm Qwen two-hop requests, with identity-only deduplication."""

import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["model", "first", "diagnostic", "out"]:
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    out = a.out
    assert not (out / "requests.json").exists()
    domain = json.loads((a.first / "domain.json").read_text())
    probes = json.loads((a.first / "probe.json").read_text())
    verdict = json.loads((a.diagnostic / "results.json").read_text())
    assert json.loads((a.first / "verification.json").read_text())["gate"] == "passed"
    assert (
        json.loads((a.diagnostic / "verification.json").read_text())[
            "source_verdicts_checked"
        ]
        == 6500
    )
    by_id = {c["id"]: c for c in domain["cases"]}
    admission = {c["id"]: c for c in verdict["records"]}
    native = []
    for path in sorted(a.first.glob("batch_*_capture.json")):
        native.extend(json.loads(path.read_text())["cases"])
    first_native = {c["id"]: n for c, n in zip(domain["cases"], native, strict=True)}
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    requests = []
    unique = []
    lookup = {}
    first_hops = {}
    for left in range(3):
        for right in range(3):
            identifier = f"q35-t0-a{left}-b{right}"
            first = by_id[identifier]
            observed = first_native[identifier]
            actual = observed["prediction"]
            correct = domain["token_map"][str(left + right)]
            assert actual == correct and not admission[identifier]["centered_sources"]
            assert observed["ids"] == first["ids"]
            first_hops[identifier] = dict(
                native_token=actual,
                tool_token=correct,
                admission="refused",
                original_ids=first["ids"],
                truth=str(left + right),
            )
            for third in range(3):
                question = f"Now add {third} to that sum. Reply with the single digit answer only."
                next_turn = tok.apply_chat_template(
                    [{"role": "user", "content": question}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                assert next_turn.startswith("<|im_start|>user\n")
                suffix = "<|im_end|>\n" + next_turn
                suffix_ids = tok.encode(suffix, add_special_tokens=False)
                for arm in ["rederive", "crystallize", "generated", "tool"]:
                    token = (
                        actual
                        if arm == "generated"
                        else correct
                        if arm == "tool"
                        else None
                    )
                    insert = [] if token is None else [token]
                    ids = first["ids"] + insert + suffix_ids
                    text = (
                        first["rendered"]
                        + (
                            ""
                            if token is None
                            else tok.decode([token], skip_special_tokens=False)
                        )
                        + suffix
                    )
                    assert tok.encode(text, add_special_tokens=False) == ids
                    key = tuple(ids)
                    if key not in lookup:
                        lookup[key] = len(unique)
                        unique.append(dict(id=f"u{len(unique):03}", ids=ids))
                    requests.append(
                        dict(
                            a=left,
                            b=right,
                            c=third,
                            arm=arm,
                            first_hop_id=identifier,
                            truth=str(left + right + third),
                            unique_index=lookup[key],
                            inserted_token=token,
                            suffix_ids=suffix_ids,
                            memory="generated"
                            if arm == "generated"
                            else "tool"
                            if arm == "tool"
                            else "none",
                            admission="refused",
                        )
                    )
    assert len(requests) == 108 and len(unique) == 54 and len(first_hops) == 9
    result = dict(
        schema="pdc.qwen35-two-hop-requests.v1",
        first_hops=first_hops,
        requests=requests,
        unique_contexts=unique,
        model="pinned Qwen3.5-4B",
        admission_counts=dict(admitted=0, refused=27, unknown=0),
        forward_budget=dict(native=54, reference=54),
        probe_tokens=probes["tokens"],
    )
    selection = dict(
        contexts=[u["ids"] for u in unique],
        candidate_tokens=probes["tokens"],
        check_observer=False,
    )
    for name, data in [("requests.json", result), ("selection.json", selection)]:
        (out / name).write_text(json.dumps(data, indent=2) + "\n")
        print(name, hashlib.sha256((out / name).read_bytes()).hexdigest())
    print("108 paths,54 unique contexts,nine reused first hops,all refused")


if __name__ == "__main__":
    main()
