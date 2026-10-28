"""Independent torch/alternative-algebra audit of the fixed singleton diagnostic."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from pdc_qwen35_engineering_check import read_frame


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["artifacts", "bundle", "diagnostic"]:
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    root = a.artifacts
    out = a.diagnostic
    torch.set_num_threads(4)
    manifest = json.loads((out / "manifest.json").read_text())
    original = json.loads((root / "run_manifest.json").read_text())
    repo = Path(__file__).resolve().parents[1]
    for name, wanted in manifest["sha256"].items():
        archived = Path(name)
        if name.startswith("experiments/pdc_qwen35_4b/"):
            local = root / archived.name
        elif name.startswith("experiments/pdc_qwen35_singletons/"):
            local = out / archived.name
        elif name.startswith("scripts/") or archived.parent.name == "scripts":
            local = repo / "scripts" / archived.name
        elif name.endswith(".fieldrun.json"):
            local = Path(str(a.bundle) + ".fieldrun.json")
        elif name.endswith(".fieldrun.bin"):
            local = Path(str(a.bundle) + ".fieldrun.bin")
        else:
            raise ValueError(f"unexpected frozen input identity: {name}")
        h = hashlib.sha256()
        with local.open("rb") as f:
            for b in iter(lambda: f.read(1024 * 1024), b""):
                h.update(b)
        assert h.hexdigest() == wanted, name
    for suffix in [".fieldrun.json", ".fieldrun.bin"]:
        diagnostic_ids = [
            v for k, v in manifest["sha256"].items() if k.endswith(suffix)
        ]
        original_ids = [v for k, v in original["sha256"].items() if k.endswith(suffix)]
        assert len(diagnostic_ids) == len(original_ids) == 1
        assert diagnostic_ids == original_ids
    domain = json.loads((root / "domain.json").read_text())
    tokens = json.loads((root / "probe.json").read_text())["tokens"]
    m, U, gain = read_frame(a.bundle, tokens)
    report = json.loads((out / "results.json").read_text())
    counts = {
        s: {
            "contexts": 0,
            "exact": 0,
            "centered": 0,
            "raw_match": 0,
            "any_floor_pass": 0,
            "any_pairwise_slack_pass": 0,
        }
        for s in ["development", "heldout", "all"]
    }
    maximum = 0.0
    index = 0
    for path in sorted(root.glob("batch_*_capture.json")):
        for native in json.loads(path.read_text())["cases"]:
            case = domain["cases"][index]
            saved = report["records"][index]
            index += 1
            assert native["ids"] == case["ids"] and saved["id"] == case["id"]
            t = tokens.index(native["prediction"])
            raw = torch.tensor(
                [s["write"] for s in native["sources"]], dtype=torch.float64
            )
            residual = np.array(native["final_residual"])
            # Scale readout rows first, rather than source vectors; use torch
            # rather than NumPy for the matrix product.
            scale = gain / np.sqrt(np.mean(residual * residual) + m["config_f"][1])
            P = (raw @ torch.tensor(U * scale, dtype=torch.float64).T).numpy()
            H = np.array(native["candidate_scores"])
            mask = np.arange(len(tokens)) != t
            PG = P[:, t, None] - P[:, mask]
            HG = H[t] - H[mask]
            margins = PG.min(1)
            slacks = (PG - np.abs(HG[None, :] - PG)).min(1)
            delta2 = np.ptp(H[None, :] - P, axis=1)
            # abs(HG-PG)<PG iff 0<HG<2*PG, checked without
            # reusing the exporter's primary inequality implementation.
            exact = (margins > 0.5) & np.all(
                (HG[None, :] > 0) & (HG[None, :] < 2 * PG), axis=1
            )
            centered = (margins > 0.5) & (margins > delta2)
            raw_match = margins > 0
            for j, row in enumerate(saved["sources"]):
                maximum = max(
                    maximum,
                    abs(margins[j] - row["margin"]),
                    abs(slacks[j] - row["minimum_pairwise_slack"]),
                    abs(delta2[j] - row["two_delta"]),
                )
                assert bool(exact[j]) == row["exact_pairwise_admitted"]
                assert bool(centered[j]) == row["centered_admitted"]
                assert bool(raw_match[j]) == row["host_target_match"]
                assert not centered[j] or exact[j]
                if centered[j]:
                    assert margins[j] > row["gamma"] > delta2[j]
            assert saved["any_raw_host_match"] == bool(raw_match.any())
            for split in [case["split"], "all"]:
                c = counts[split]
                c["contexts"] += 1
                c["exact"] += bool(exact.any())
                c["centered"] += bool(centered.any())
                c["raw_match"] += bool(raw_match.any())
                c["any_floor_pass"] += bool((margins > 0.5).any())
                c["any_pairwise_slack_pass"] += bool((slacks > 0).any())
    assert index == 100 and maximum < 1e-8
    for split, c in counts.items():
        r = report["summary"][split]
        assert (
            c["contexts"] == r["contexts"]
            and c["raw_match"] == r["raw_singleton_host_matches"]
        )
        assert (
            c["exact"] == r["exact"]["admitted"]
            and c["centered"] == r["centered"]["admitted"]
        )
    result = dict(
        schema="pdc.qwen35-singleton-verification.v1",
        source_verdicts_checked=6500,
        maximum_independent_discrepancy=maximum,
        counts=counts,
        original_frame_hashes_match=True,
        input_hashes_verified=True,
        new_model_forwards=0,
        scope="fixed-source score diagnosis, not causal or impossibility evidence",
    )
    (out / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
