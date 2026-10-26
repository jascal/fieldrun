"""Independent NumPy audit of the fixed Qwen learning NPZ and outside reduction."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["data", "bundle", "first", "memory", "out"]:
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    d = np.load(a.data)
    meta = json.loads(a.data.with_suffix(".json").read_text())
    assert hashlib.sha256(a.data.read_bytes()).hexdigest() == meta["npz_sha256"]
    header = json.loads(Path(str(a.bundle) + ".fieldrun.json").read_text())
    blob = np.memmap(str(a.bundle) + ".fieldrun.bin", mode="r", dtype=np.uint8)
    arrays = {}
    for e in header["arrays"]:
        if e["name"] in ["embed", "norm"]:
            arrays[e["name"]] = np.frombuffer(
                blob,
                dtype={"f16": "<f2", "f32": "<f4"}[e["dtype"]],
                offset=e["offset"],
                count=np.prod(e["shape"]),
            ).reshape(e["shape"])
    frame = arrays["embed"]
    gain = 1 + arrays["norm"].astype(float)
    pool = d["pool_tokens"]
    assert np.array_equal(d["initial_U"], frame[pool].astype(float))
    original = []
    second = []
    for root, dest in [(a.first, original), (a.memory, second)]:
        for path in sorted(root.glob("batch_*_capture.json")):
            dest.extend(json.loads(path.read_text())["cases"])
    req = json.loads((a.memory / "requests.json").read_text())["requests"]
    contexts = original + [second[r["unique_index"]] for r in req]
    assert len(contexts) == 208
    r = d["protected_residuals"]
    source_error = 0.0
    for i, c in enumerate(contexts):
        assert np.array_equal(r[i], c["final_hidden"])
        assert d["native_teacher_tokens"][i] == c["prediction"]
        raw = np.array([s["write"] for s in c["sources"]])
        residual = np.array(c["final_residual"])
        folded = raw * (
            gain / np.sqrt(np.mean(residual * residual) + header["config_f"][1])
        )
        saved = d["native_sources"][i] if i < 100 else d["hop2_sources"][i - 100]
        source_error = max(source_error, float(np.abs(folded - saved).max()))
    assert source_error < 1e-10
    # Reverse product orientation, independently normalized blocks and NumPy
    # stable logsumexp, rather than the exporter's torch implementations.
    outside_max = np.full(208, -np.inf)
    outside_lse = np.full(100, -np.inf)
    gram = np.zeros((2560, 2560))
    nonzero = 0
    keep = np.ones(frame.shape[0], dtype=bool)
    keep[pool] = False
    for start in range(0, len(frame), 2048):
        rows = frame[start : start + 2048].astype(float)[keep[start : start + 2048]]
        norm = np.linalg.norm(rows, axis=1)
        nonzero += int((norm > 0).sum())
        normalized = rows / np.where(norm > 0, norm, 1)[:, None]
        gram += normalized.T @ normalized
        scores = rows @ r.T
        outside_max = np.maximum(outside_max, scores.max(0))
        block = scores[:, :100]
        maximum = block.max(0)
        lse = maximum + np.log(np.exp(block - maximum[None, :]).sum(0))
        outside_lse = np.logaddexp(outside_lse, lse)
    max_error = float(np.abs(outside_max - d["outside_max"]).max())
    lse_error = float(np.abs(outside_lse - d["outside_lse"]).max())
    gram_error = float(np.abs(gram - d["outside_unit_gram"]).max())
    assert max_error < 1e-8 and lse_error < 1e-8 and gram_error < 1e-7
    assert nonzero == int(d["outside_active_count"])
    scores = r @ d["initial_U"].T + d["initial_bias"]
    full = np.c_[scores, outside_max]
    assert np.array_equal(full.argmax(1), d["teacher_indices"])
    assert np.array_equal(pool[d["teacher_indices"]], d["native_teacher_tokens"])
    result = dict(
        schema="pdc.qwen35-learning-data-audit.v1",
        gate="passed",
        protected_contexts=208,
        initial_full_host_matches=208,
        source_error=source_error,
        outside_max_error=max_error,
        outside_lse_error=lse_error,
        outside_gram_error=gram_error,
        data_sha256=meta["npz_sha256"],
        new_model_calls=0,
    )
    a.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
