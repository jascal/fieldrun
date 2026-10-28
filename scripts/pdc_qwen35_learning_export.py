"""Export fixed Qwen sources, full-host constraints and streamed full-frame geometry."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def load_cases(root):
    cases = []
    for path in sorted(root.glob("batch_*_capture.json")):
        cases.extend(json.loads(path.read_text())["cases"])
    return cases


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["first", "memory", "bundle", "out"]:
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    assert not a.out.exists()
    assert json.loads((a.first / "verification.json").read_text())["gate"] == "passed"
    assert json.loads((a.memory / "verification.json").read_text())["gate"] == "passed"
    torch.set_num_threads(4)
    domain = json.loads((a.first / "domain.json").read_text())
    requests = json.loads((a.memory / "requests.json").read_text())
    first = load_cases(a.first)
    second = load_cases(a.memory)
    assert len(first) == 100 and len(second) == 54 and len(requests["requests"]) == 108
    protected = first + [second[r["unique_index"]] for r in requests["requests"]]
    teacher_tokens = np.array([c["prediction"] for c in protected], dtype=np.int64)
    original_pool = json.loads((a.first / "probe.json").read_text())["tokens"]
    pool = np.array(
        sorted(set(original_pool) | set(teacher_tokens.tolist())), dtype=np.int64
    )
    model = json.loads(Path(str(a.bundle) + ".fieldrun.json").read_text())
    blob = np.memmap(str(a.bundle) + ".fieldrun.bin", mode="r", dtype=np.uint8)
    arrays = {}
    for entry in model["arrays"]:
        if entry["name"] in ["embed", "norm"]:
            arrays[entry["name"]] = np.frombuffer(
                blob,
                dtype={"f16": "<f2", "f32": "<f4"}[entry["dtype"]],
                offset=entry["offset"],
                count=np.prod(entry["shape"]),
            ).reshape(entry["shape"])
    frame = arrays["embed"]
    gain = 1 + arrays["norm"].astype(np.float64)
    vocab, width = frame.shape
    assert width == 2560 and vocab == 248320 and model["config"][6] == 1
    r = np.array([c["final_hidden"] for c in protected], dtype=np.float64)
    D = []
    for c in protected:
        raw = np.array([s["write"] for s in c["sources"]], dtype=np.float64)
        residual = np.array(c["final_residual"], dtype=np.float64)
        assert raw.shape == (65, width)
        D.append(
            raw * gain / np.sqrt(np.mean(residual * residual) + model["config_f"][1])
        )
    D = np.array(D)
    U = frame[pool].astype(np.float64)
    t = np.array(
        [int(np.flatnonzero(pool == v)[0]) for v in teacher_tokens], dtype=np.int64
    )
    truth = np.array(
        [domain["token_map"][str(c["a"] + c["b"])] for c in domain["cases"]],
        dtype=np.int64,
    )
    truth_indices = np.array(
        [int(np.flatnonzero(pool == v)[0]) for v in truth], dtype=np.int64
    )
    R = torch.tensor(r)
    outside_max = torch.full((208,), -torch.inf, dtype=torch.float64)
    outside_lse = torch.full((100,), -torch.inf, dtype=torch.float64)
    G = torch.zeros((width, width), dtype=torch.float64)
    outside_active = 0
    zero_tokens = []
    pool_mask = np.zeros(vocab, dtype=bool)
    pool_mask[pool] = True
    for start in range(0, vocab, 1024):
        end = min(start + 1024, vocab)
        rows = torch.tensor(frame[start:end].astype(np.float64))
        norm = rows.norm(dim=1)
        zero_tokens.extend((np.flatnonzero(norm.numpy() == 0) + start).tolist())
        keep = torch.tensor(~pool_mask[start:end])
        outside = rows[keep]
        norms = norm[keep]
        nonzero = norms > 0
        outside_active += int(nonzero.sum())
        normalized = (
            outside / torch.where(nonzero, norms, torch.ones_like(norms))[:, None]
        )
        G += normalized.T @ normalized
        scores = R @ outside.T
        outside_max = torch.maximum(outside_max, scores.max(1).values)
        outside_lse = torch.logaddexp(outside_lse, torch.logsumexp(scores[:100], dim=1))
        if start % (1024 * 32) == 0:
            print("exported rows", end, "of", vocab, flush=True)
    full_pool = r @ U.T
    rival = full_pool.copy()
    rival[np.arange(208), t] = -np.inf
    margin = full_pool[np.arange(208), t] - np.maximum(
        rival.max(1), outside_max.numpy()
    )
    pool_norm = np.linalg.norm(U, axis=1)
    normalized = U / np.where(pool_norm > 0, pool_norm, 1)[:, None]
    total_G = G.numpy() + normalized.T @ normalized
    active = outside_active + int((pool_norm > 0).sum())
    off = float(np.square(total_G).sum() - active)
    fp_token = off / (vocab * (vocab - 1))
    fp_active = off / (active * (active - 1))
    W_token = max(0, (vocab - width) / (width * (vocab - 1)))
    W_active = max(0, (active - width) / (width * (active - 1)))
    labels = truth.tolist() + [
        domain["token_map"][str(q["a"] + q["b"] + q["c"])] for q in requests["requests"]
    ]
    ids = [c["id"] for c in domain["cases"]] + [
        f"h2-{q['arm']}-{q['a']}-{q['b']}-{q['c']}" for q in requests["requests"]
    ]
    arms = ["first"] * 100 + [q["arm"] for q in requests["requests"]]
    np.savez(
        a.out,
        native_sources=D[:100],
        hop2_sources=D[100:],
        protected_residuals=r,
        initial_U=U,
        initial_bias=np.zeros(len(pool)),
        pool_tokens=pool,
        teacher_indices=t,
        truth_indices=truth_indices,
        native_teacher_tokens=teacher_tokens,
        outside_max=outside_max.numpy(),
        outside_lse=outside_lse.numpy(),
        outside_unit_gram=G.numpy(),
        outside_active_count=np.array(outside_active),
        vocabulary_size=np.array(vocab),
        zero_native_tokens=np.array(zero_tokens, dtype=np.int64),
        numeric_closure=r[:100] - D[:100].sum(1),
        protected_truth_tokens=np.array(labels, dtype=np.int64),
        protected_case_ids=np.array(ids),
        protected_arms=np.array(arms),
        initial_global_margins=margin,
    )
    files = [
        Path(str(a.bundle) + ".fieldrun.json"),
        Path(str(a.bundle) + ".fieldrun.bin"),
    ]
    files += [
        a.first / n
        for n in ["domain.json", "probe.json", "run_manifest.json", "verification.json"]
    ]
    files += [
        a.memory / n
        for n in [
            "requests.json",
            "run_manifest.json",
            "verification.json",
            "results.json",
        ]
    ]
    files += list(a.first.glob("batch_*_capture.json")) + list(
        a.memory.glob("batch_*_capture.json")
    )
    good = bool(
        np.all(margin > 0)
        and fp_token + 1e-10 >= W_token
        and fp_active + 1e-10 >= W_active
    )
    report = dict(
        schema="pdc.qwen35-learning-data.v1",
        training_contexts=100,
        protected_contexts=208,
        unique_second_hop_contexts=54,
        protected_second_hop_paths=108,
        pool_rows=len(pool),
        original_probe_rows=len(original_pool),
        full_vocab_rows=vocab,
        dimension=width,
        initial_native_label_matches=int((margin > 0).sum()),
        minimum_global_margin=float(margin.min()),
        zero_native_rows=len(zero_tokens),
        active_direction_count=active,
        initial_fp_token=fp_token,
        welch_token=W_token,
        initial_fp_active=fp_active,
        welch_active=W_active,
        numeric_closure_max_abs=float(np.abs(r[:100] - D[:100].sum(1)).max()),
        gate="passed" if good else "inapplicable",
        npz_sha256=sha(a.out),
        sha256={str(p): sha(p) for p in files},
    )
    a.out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "sha256"}, indent=2))
    assert good, "data fidelity/geometry gate failed; retain report and do not train"


if __name__ == "__main__":
    main()
