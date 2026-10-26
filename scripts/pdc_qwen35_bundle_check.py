"""Independent saved-artifact chart/certificate and full-host faithfulness audit."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from pdc_qwen35_engineering_check import read_frame


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["artifacts", "bundle"]:
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument(
        "--model",
        type=Path,
        help="original HF checkpoint directory (defaults to bundle parent/parent)",
    )
    a = p.parse_args()
    root = a.artifacts
    repo = Path(__file__).resolve().parents[1]
    model_root = a.model if a.model is not None else a.bundle.parent.parent
    domain = json.loads((root / "domain.json").read_text())
    packet = json.loads((root / "bundle.json").read_text())
    reference = json.loads((root / "scientific_reference.json").read_text())
    tokens = json.loads((root / "probe.json").read_text())["tokens"]
    manifest = json.loads((root / "run_manifest.json").read_text())
    for name, wanted in manifest["sha256"].items():
        h = hashlib.sha256()
        original = Path(name)
        if name.startswith("experiments/pdc_qwen35_4b/"):
            local = root / original.name
        elif (
            name.startswith("scripts/")
            or name.startswith("src/")
            or name.startswith("target/")
        ):
            local = repo / original
        elif original.name.endswith(".fieldrun.bin"):
            local = Path(str(a.bundle) + ".fieldrun.bin")
        elif original.name.endswith(".fieldrun.json"):
            local = Path(str(a.bundle) + ".fieldrun.json")
        else:
            local = model_root / original.name
        with local.open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                h.update(block)
        assert h.hexdigest() == wanted, name
    m, U, gain = read_frame(a.bundle, tokens)
    native = []
    for path in sorted(root.glob("batch_*_capture.json")):
        native.extend(json.loads(path.read_text())["cases"])
    assert len(native) == len(reference["cases"]) == len(packet["responses"]) == 100
    programs = {
        "full": list(range(65)),
        "attention": list(range(1, 65, 2)),
        "mlp": list(range(2, 65, 2)),
    }
    assert packet["body_programs"] == programs
    rivals = list(domain["token_map"])
    indices = [tokens.index(domain["token_map"][v]) for v in rivals]
    errors = []
    matches = 0
    commitments = 0
    for case, native_case, ref, row in zip(
        domain["cases"], native, reference["cases"], packet["responses"], strict=True
    ):
        assert case["ids"] == native_case["ids"] == ref["ids"]
        matches += native_case["prediction"] == ref["prediction"]
        raw = np.array([s["write"] for s in native_case["sources"]])
        r = np.array(native_case["final_residual"])
        # Apply the host's fixed RMS gain after summing a coalition, instead
        # of folding each write as the exporter does.
        scale = gain / np.sqrt(np.mean(r * r) + m["config_f"][1])
        bodies = {name: U @ (raw[js].sum(0) * scale) for name, js in programs.items()}
        for name, values in bodies.items():
            error = float(np.max(np.abs(values - row["guard_body_scores"][name])))
            assert error < 1e-7
            errors.append(error)
        matrix = np.array([v[indices] for v in bodies.values()])
        readouts = {
            "viterbi": matrix.max(0),
            "marginal": np.logaddexp.reduce(matrix, axis=0),
        }
        host = np.array(native_case["candidate_scores"])
        truth = str(case["a"] + case["b"])
        assert truth == case["truth"] == row["truth"]
        for name, scores in readouts.items():
            winners = np.flatnonzero(scores == scores.max()).tolist()
            q = next(q for q in row["chart"]["queries"] if q["readout"] == name)
            assert q["winners"] == [rivals[i] for i in winners]
            ordered = np.sort(scores)
            margin = float(ordered[-1] - ordered[-2])
            decision = next(d for d in row["commitments"] if d["readout"] == name)
            commit = False
            if len(winners) == 1 and margin > 0.5:
                atom = rivals[winners[0]]
                t = tokens.index(domain["token_map"][atom])
                high = max(v[t] for v in bodies.values())
                witness = sorted(k for k, v in bodies.items() if v[t] == high)[0]
                partial = bodies[witness]
                mask = np.arange(len(tokens)) != t
                gaps = partial[t] - partial[mask]
                drift = (host[t] - partial[t]) - (host[mask] - partial[mask])
                commit = bool(gaps.min() > 0.5 and np.all(gaps > np.abs(drift)))
                if commit:
                    assert decision["body"] == witness and decision["atom"] == atom
                    assert decision["theory_verdict"] == (
                        "affirmed" if atom == truth else "hallucination"
                    )
                    assert host[t] > host[mask].max()
                    commitments += 1
            assert (decision["status"] == "committed") == commit
            if not commit:
                assert decision["theory_verdict"] == "abstain"
    result = dict(
        schema="pdc.qwen35-bundle-audit.v1",
        contexts=100,
        reference_matches=matches,
        maximum_independent_body_error=max(errors),
        body_probe_annotations_checked=100 * 3 * len(tokens),
        commitments_checked=commitments,
        input_hashes_verified=True,
        gate="passed" if matches == 100 else "failed_native_reference_agreement",
        reference_forward_calls=100,
        scope="finite engineering and certificate audit; no truth theorem",
    )
    (root / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    assert matches == 100, "retain failed report; scientific faithfulness gate failed"


if __name__ == "__main__":
    main()
