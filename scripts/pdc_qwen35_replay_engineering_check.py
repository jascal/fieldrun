"""Check explicit replay provenance, zero identity and independent parity."""

import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ["capture", "reference", "selection", "out"]:
        p.add_argument("--" + n, type=Path, required=True)
    p.add_argument("--tiny", action="store_true")
    a = p.parse_args()
    native = json.loads(a.capture.read_text())
    ref = json.loads(a.reference.read_text())
    selection = json.loads(a.selection.read_text())
    rows = []
    cases = native["cases"]
    references = ref["cases"]
    assert (
        len(cases)
        == len(references)
        == len(selection["contexts"])
        == (9 if a.tiny else 5)
    )
    for n, r, ids, memory in zip(
        cases,
        references,
        selection["contexts"],
        selection["memory_writes"],
        strict=True,
    ):
        assert n["ids"] == r["ids"] == ids and n["prediction"] == r["prediction"]
        assert r["memory_replay"] == memory
        assert (
            n["memory_replay"] is None
            if memory is None
            else n["memory_replay"]["site"] == "embed"
        )
        if memory is not None:
            np.testing.assert_array_equal(
                np.array(n["memory_replay"]["write"], dtype=np.float32),
                np.array(memory["write"], dtype=np.float32),
            )
        assert r["embedding_hook_calls"] == int(memory is not None)
        assert len(n["sources"]) == 1 + 2 * native["config"][0] + int(
            memory is not None
        )
        if memory is not None:
            assert n["sources"][1]["id"] == "memory@embed"
            np.testing.assert_array_equal(
                np.array(n["sources"][1]["write"], dtype=np.float32),
                np.array(memory["write"], dtype=np.float32),
            )
        D = np.array([s["write"] for s in n["sources"]])
        residual = np.array(n["final_residual"])
        closure = float(np.max(np.abs(D.sum(0) - residual)))
        assert closure < 1e-4 * (1 + np.max(np.abs(residual)))
        hidden_error = float(
            np.max(np.abs(np.array(n["final_hidden"]) - r["final_hidden"]))
        )
        if a.tiny:
            assert hidden_error < 1e-4
        rows.append(
            dict(
                prediction=n["prediction"],
                source_sum_error=closure,
                hidden_max_error=hidden_error,
                candidate_score_max_error=float(
                    np.max(
                        np.abs(np.array(n["candidate_scores"]) - r["candidate_scores"])
                    )
                ),
            )
        )
    triples = [(0, 1, 2), (3, 4, 5), (6, 7, 8)] if a.tiny else [(0, 1, 2)]
    effects = []
    for base, zero, nonzero in triples:
        b = np.array(cases[base]["final_hidden"], dtype=np.float32)
        z = np.array(cases[zero]["final_hidden"], dtype=np.float32)
        w = np.array(cases[nonzero]["final_hidden"], dtype=np.float32)
        assert np.array_equal(b.view(np.uint32), z.view(np.uint32))
        assert references[base]["final_hidden"] == references[zero]["final_hidden"]
        nr = float(np.max(np.abs(w - b)))
        rr = float(
            np.max(
                np.abs(
                    np.array(references[nonzero]["final_hidden"])
                    - references[base]["final_hidden"]
                )
            )
        )
        assert nr > 1e-6 and rr > 1e-6
        effects.append(dict(native_nonzero_effect=nr, reference_nonzero_effect=rr))
    report = dict(
        gate="passed",
        engineering_only=True,
        admitted_replay_tested=False,
        native_reference_matches=len(cases),
        rows=rows,
        zero_replay_bit_identity=True,
        effects=effects,
        earlier_position_scope="Native unit test; not independently measured on these real-model captures.",
    )
    a.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
