"""Independent engineering checks for native Qwen hybrid residual capture."""

import argparse
import json
from pathlib import Path

import numpy as np


def read_frame(stem, tokens):
    m = json.loads(Path(str(stem) + ".fieldrun.json").read_text())
    blob = np.memmap(str(stem) + ".fieldrun.bin", dtype=np.uint8, mode="r")
    name = "embed" if m["config"][6] else "lm_head"
    weights = {}
    for e in m["arrays"]:
        if e["name"] in [name, "norm"]:
            view = np.frombuffer(
                blob,
                dtype={"f16": "<f2", "f32": "<f4"}[e["dtype"]],
                offset=e["offset"],
                count=np.prod(e["shape"]),
            ).reshape(e["shape"])
            weights[e["name"]] = (view[tokens] if e["name"] == name else view).astype(
                np.float64
            )
    return m, weights[name], 1 + weights["norm"]


def check_reconstruction(raw, residual, hidden, gain, epsilon, scores, frame):
    raw, residual, hidden, gain, scores, frame = [np.asarray(x,dtype=float)
        for x in (raw,residual,hidden,gain,scores,frame)]
    width = residual.size
    if residual.shape != (width,) or width == 0 or hidden.shape != (width,) \
            or gain.shape != (width,) or raw.ndim != 2 or raw.shape[0] == 0 \
            or raw.shape[1] != width or scores.ndim != 1 or scores.size == 0 \
            or frame.shape != (scores.size,width) or not np.isfinite(epsilon) or epsilon <= 0 \
            or not all(np.isfinite(x).all() for x in (raw,residual,hidden,gain,scores,frame)):
        raise ValueError("incomplete or nonfinite reconstruction inputs")
    folded = raw*gain/np.sqrt(np.mean(residual*residual)+epsilon)
    errors = dict(source_sum=float(np.max(np.abs(raw.sum(0)-residual))),
        rms_fold=float(np.max(np.abs(folded.sum(0)-hidden))),
        full_body_score=float(np.max(np.abs(frame@folded.sum(0)-scores))))
    for name, tolerance, reference in (("source_sum",1e-4,residual),
            ("rms_fold",1e-4,hidden),("full_body_score",1e-3,scores)):
        if not errors[name] < tolerance*(1+np.max(np.abs(reference))):
            raise ValueError(f"{name} reconstruction exceeds engineering bound")
    return folded, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["capture", "reference", "bundle", "out"]:
        parser.add_argument("--" + name, type=Path, required=True)
    a = parser.parse_args()
    capture = json.loads(a.capture.read_text())
    ref = json.loads(a.reference.read_text())
    tokens = capture["cases"][0]["candidate_tokens"]
    m, U, gain = read_frame(a.bundle, tokens)
    rows = []
    assert len(capture["cases"]) == len(ref["cases"])
    for native, reference in zip(capture["cases"], ref["cases"], strict=True):
        assert native["ids"] == reference["ids"]
        assert (
            native["observer_check_performed"]
            and native["observer_hidden_max_error"] == 0
        )
        assert native["prediction"] == reference["prediction"]
        names = ["embed"] + [
            name
            for layer in range(m["config"][0])
            for name in [f"L{layer}.attn", f"L{layer}.mlp"]
        ]
        assert [s["id"] for s in native["sources"]] == names
        D = np.array([s["write"] for s in native["sources"]])
        r = np.array(native["final_residual"])
        h = np.array(native["final_hidden"])
        assert np.isfinite(D).all() and np.isfinite(r).all() and np.isfinite(h).all()
        folded, errors = check_reconstruction(D,r,h,gain,m["config_f"][1],native["candidate_scores"],U)
        raw_error, fold_error, score_error = (errors[k] for k in ("source_sum","rms_fold","full_body_score"))
        rows.append(
            dict(
                prediction=native["prediction"],
                source_sum_error=raw_error,
                rms_fold_error=fold_error,
                full_body_score_error=score_error,
                reference_score_error=float(
                    np.max(
                        np.abs(
                            np.array(reference["candidate_scores"])
                            - native["candidate_scores"]
                        )
                    )
                ),
            )
        )
    result = dict(
        schema="pdc.qwen35-engineering-check.v1",
        cases=len(rows),
        reference_agreement=1.0,
        observer_effect=0,
        rows=rows,
        scope="engineering gate; not scientific coverage result",
    )
    a.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
