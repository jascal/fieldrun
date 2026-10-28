"""Check exact native-prefix preservation and independent memory provenance."""

import argparse
import json
from collections import Counter
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifacts", type=Path, required=True)
    p.add_argument("--first", type=Path, required=True)
    a = p.parse_args()
    domain = json.loads((a.first / "domain.json").read_text())
    original = []
    for path in sorted(a.first.glob("batch_*_capture.json")):
        original.extend(json.loads(path.read_text())["cases"])
    original = {c["id"]: n for c, n in zip(domain["cases"], original, strict=True)}
    r = json.loads((a.artifacts / "requests.json").read_text())
    selection = json.loads((a.artifacts / "selection.json").read_text())
    assert selection["contexts"] == [u["ids"] for u in r["unique_contexts"]]
    assert (
        len(selection["contexts"])
        == len({tuple(c) for c in selection["contexts"]})
        == 54
    )
    grouped = {}
    tags = Counter()
    for req in r["requests"]:
        first = original[req["first_hop_id"]]
        prefix = first["ids"]
        ids = selection["contexts"][req["unique_index"]]
        arm = req["arm"]
        token = req["inserted_token"]
        expected = (
            first["prediction"]
            if arm == "generated"
            else domain["token_map"][str(req["a"] + req["b"])]
            if arm == "tool"
            else None
        )
        assert token == expected
        assert ids[: len(prefix)] == prefix
        assert (
            ids[len(prefix) :] == ([] if token is None else [token]) + req["suffix_ids"]
        )
        assert req["truth"] == str(req["a"] + req["b"] + req["c"])
        tag = "generated" if arm == "generated" else "tool" if arm == "tool" else "none"
        assert req["memory"] == tag and req["admission"] == "refused"
        tags[tag] += 1
        grouped.setdefault((req["a"], req["b"], req["c"]), {})[arm] = req[
            "unique_index"
        ]
    assert len(grouped) == 27 and len(r["requests"]) == 108
    for arms in grouped.values():
        assert set(arms) == {"rederive", "crystallize", "generated", "tool"}
        assert arms["rederive"] == arms["crystallize"]
        assert arms["generated"] == arms["tool"]
        assert arms["rederive"] != arms["generated"]
    report = dict(
        schema="pdc.qwen35-memory-assembly-check.v1",
        triples=27,
        paths=108,
        unique_contexts=54,
        original_prefixes_preserved=108,
        memory_tags=dict(tags),
        refused_replay_identity_by_input=27,
        generated_tool_identity_by_input=27,
        admitted_replay_tested=0,
        new_model_calls=0,
    )
    (a.artifacts / "assembly_verification.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
