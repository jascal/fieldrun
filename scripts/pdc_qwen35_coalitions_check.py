"""Reconstruct frozen coalition guards by summing raw writes before Torch scoring."""

import argparse
import hashlib
from itertools import combinations
import json
from pathlib import Path
import numpy as np
import torch
from pdc_qwen35_engineering_check import read_frame


def digest(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["artifacts", "first", "bundle", "out"]:
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--verify-original-paths", action="store_true")
    a = p.parse_args()
    torch.set_num_threads(4)
    root = a.artifacts
    groups = json.loads((root / "bodies.json").read_text())
    expected = [list(v) for v in combinations(range(65), 2)]
    for k in [4, 8, 16, 32, 64]:
        expected.extend([list(range(k)), list(range(65 - k, 65))])
    expected.append(list(range(65)))
    assert [b["sources"] for b in groups] == expected and len(groups) == 2091
    probe = json.loads((a.first / "probe.json").read_text())["tokens"]
    meta, U, gain = read_frame(a.bundle, probe)
    U = torch.tensor(U)
    gain = torch.tensor(gain)
    membership = torch.zeros((2091, 65), dtype=torch.float64)
    for j, idx in enumerate(expected):
        membership[j, idx] = 1
    result = json.loads((root / "results.json").read_text())
    stored = np.load(root / "body_metrics.npz")
    domain = json.loads((a.first / "domain.json").read_text())
    records = result["records"]
    errors = dict(margin=0.0, minimum_pairwise_slack=0.0, two_delta=0.0)
    by_exact = np.zeros(2091, dtype=int)
    by_center = np.zeros(2091, dtype=int)
    native = []
    for file in sorted(a.first.glob("batch_*_capture.json")):
        native.extend(json.loads(file.read_text())["cases"])
    assert len(native) == len(records) == 100
    compact = np.array([len(v) <= 8 for v in expected])
    compact_exact = compact_center = 0
    for i, n in enumerate(native):
        assert n["ids"] == domain["cases"][i]["ids"] and n["candidate_tokens"] == probe
        assert [s["id"] for s in n["sources"]] == ["embed"] + [
            v for layer in range(32) for v in [f"L{layer}.attn", f"L{layer}.mlp"]
        ]
        raw = torch.tensor([s["write"] for s in n["sources"]], dtype=torch.float64)
        residual = torch.tensor(n["final_residual"], dtype=torch.float64)
        # Unlike the primary, combine raw source vectors before folding/scoring.
        combined = membership @ raw
        folded = (
            combined * gain / torch.sqrt(residual.square().mean() + meta["config_f"][1])
        )
        P = folded @ U.T
        H = torch.tensor(n["candidate_scores"], dtype=torch.float64)
        t = probe.index(n["prediction"])
        mask = torch.arange(len(probe)) != t
        gap = P[:, t, None] - P[:, mask]
        h = H[t] - H[mask]
        margin = gap.amin(1)
        tail = H - P
        delta = tail.amax(1) - tail.amin(1)
        slack = (gap - (h - gap).abs()).amin(1)
        # Independent theorem-equivalent predicate for the exact guard.
        exact = ((gap > 0.5) & (h[None, :] > 0) & (2 * gap > h[None, :])).all(1).numpy()
        center = (margin > torch.maximum(torch.tensor(0.5), delta)).numpy()
        assert np.all(~center | exact)
        values = dict(
            margin=margin.numpy(),
            minimum_pairwise_slack=slack.numpy(),
            two_delta=delta.numpy(),
        )
        for key, value in values.items():
            e = float(np.max(np.abs(value - stored[key][i])))
            errors[key] = max(errors[key], e)
            assert e < 2e-9, (i, key, e)
        assert np.flatnonzero(exact).tolist() == records[i]["exact_passes"]
        assert np.flatnonzero(center).tolist() == records[i]["centered_passes"]
        gamma = stored["gamma"][i]
        assert np.all(np.isnan(gamma[~center]))
        assert np.all(
            (gamma[center] > np.maximum(0.5, values["two_delta"][center]))
            & (gamma[center] < values["margin"][center])
        )
        assert records[i]["target"] == n["prediction"]
        assert records[i]["host_correct"] == (
            n["prediction"] == domain["token_map"][domain["cases"][i]["truth"]]
        )
        assert records[i]["compact_exact"] == bool(np.any(exact & compact))
        assert records[i]["compact_centered"] == bool(np.any(center & compact))
        size = min((len(expected[j]) for j in np.flatnonzero(exact)), default=None)
        assert size == records[i]["minimum_tested_exact_size"]
        compact_exact += int(np.any(exact & compact))
        compact_center += int(np.any(center & compact))
        by_exact += exact
        by_center += center
        if (i + 1) % 10 == 0:
            print("Audited", i + 1, "/100", flush=True)
    for j, s in enumerate(result["summary_by_body"]):
        assert s["id"] == groups[j]["id"] and s["size"] == len(expected[j])
        assert s["exact"] == int(by_exact[j]) and s["centered"] == int(by_center[j])
    assert compact_exact == result["compact_exact_contexts"]
    assert compact_center == result["compact_centered_contexts"]
    classification = (
        "local_positive"
        if compact_exact >= 20
        else "negative_fixed_family"
        if compact_exact == 0
        else "inconclusive"
    )
    assert classification == result["compact_gate"]
    verified = 0
    if a.verify_original_paths:
        for path, sha in json.loads((root / "manifest.json").read_text())[
            "sha256"
        ].items():
            assert digest(Path(path)) == sha, path
            verified += 1
    report = dict(
        gate="passed",
        contexts=100,
        body_verdicts=209100,
        maximum_errors=errors,
        compact_exact_contexts=compact_exact,
        compact_centered_contexts=compact_center,
        original_path_hashes_verified=verified,
        new_model_forwards=0,
        scope="Independent raw-vector sum then frame folding/Torch scoring; covered-score guards only, no causal replay.",
    )
    a.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
