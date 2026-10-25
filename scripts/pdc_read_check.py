"""Independent R2 f64 read checker and NumPy GPT-2 faithfulness check."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def load_bundle(stem):
    manifest = json.loads(Path(str(stem) + ".fieldrun.json").read_text())
    blob = np.memmap(str(stem) + ".fieldrun.bin", mode="r", dtype=np.uint8)
    weights = {}
    for entry in manifest["arrays"]:
        dtype = {"f16": "<f2", "f32": "<f4"}[entry["dtype"]]
        weights[entry["name"]] = np.frombuffer(
            blob, dtype=dtype, count=np.prod(entry["shape"]), offset=entry["offset"]
        ).reshape(entry["shape"]).astype(np.float32)
    return manifest, weights


def reference(ids, config, w, last_only=False, memory=None, return_hidden=False):
    layers, heads, width = config[:3]
    hd = width // heads
    x = w["wte"][ids] + w["wpe"][:len(ids)]

    def replay(array, site):
        if memory is not None and memory["site"] == site:
            array[-1] += np.asarray(memory["write"],dtype=np.float32)

    replay(x,"embed")

    def norm(x, prefix):
        mu = x.mean(axis=1, keepdims=True)
        var = ((x - mu) ** 2).mean(axis=1, keepdims=True)
        return (x - mu) / np.sqrt(var + np.float32(1e-5)) * w[prefix + ".weight"] + w[prefix + ".bias"]

    for layer in range(layers):
        p = f"h{layer}."
        a = norm(x, p + "ln_1")
        qkv = a @ w[p + "attn.c_attn.weight"] + w[p + "attn.c_attn.bias"]
        q, k, v = np.split(qkv, 3, axis=1)
        mixed = np.empty_like(x)
        for head in range(heads):
            sl = slice(head * hd, (head + 1) * hd)
            scores = q[:, sl] @ k[:, sl].T / np.float32(np.sqrt(hd))
            scores[np.triu_indices(len(ids), 1)] = -1e10
            probs = np.exp(scores - scores.max(axis=1, keepdims=True))
            probs /= probs.sum(axis=1, keepdims=True)
            mixed[:, sl] = probs @ v[:, sl]
        x = x + mixed @ w[p + "attn.c_proj.weight"] + w[p + "attn.c_proj.bias"]
        replay(x,f"L{layer}.attn")
        a = norm(x, p + "ln_2")
        h = a @ w[p + "mlp.c_fc.weight"] + w[p + "mlp.c_fc.bias"]
        h = np.float32(0.5) * h * (1 + np.tanh(np.float32(np.sqrt(2 / np.pi)) *
            (h + np.float32(0.044715) * h * h * h)))
        x = x + h @ w[p + "mlp.c_proj.weight"] + w[p + "mlp.c_proj.bias"]
        replay(x,f"L{layer}.mlp")
    final = norm(x, "ln_f")
    if return_hidden:
        return final
    if last_only:
        final = final[-1:]
    return (final @ w["wte"].T).argmax(axis=1).tolist()


def argmax_set(values):
    return np.flatnonzero(values == values.max()).tolist()


def check(dump, selection, stem):
    manifest, weights = load_bundle(stem)
    expected = {(l, h, i) for l in selection["layers"] for h in selection["heads"]
                for i in selection["positions"]}
    assert len(dump["cases"]) == len(selection["contexts"])
    rows, faithfulness = [], []
    for case_id, (case, ids) in enumerate(zip(dump["cases"], selection["contexts"])):
        assert case["ids"] == ids
        assert case["observer_hidden_max_error"] == 0
        assert case["predictions"] == case["unobserved_predictions"]
        assert {(r["layer"], r["head"], r["query_position"]) for r in case["reads"]} == expected
        assert len(case["reads"]) == len(expected)
        ref = reference(ids, manifest["config"], weights)
        faithfulness.extend(a == b for a, b in zip(case["predictions"], ref))
        for row in case["reads"]:
            i = row["query_position"]
            q, k, v = (np.asarray(row[name], dtype=np.float64) for name in ["query", "keys", "values"])
            incidence = k @ q / np.sqrt(len(q))
            allowed = incidence[:i+1]
            p = np.exp(allowed - allowed.max())
            p /= p.sum()
            mix = p @ v[:i+1]
            actual = np.asarray(row["head_write"])
            native_probs = np.asarray(row["probabilities"])
            native_logits = np.asarray(row["masked_logits"])
            rows.append({
                "case": case_id, "layer": row["layer"], "head": row["head"], "position": i,
                "incidence_max_error": float(np.max(np.abs(allowed - native_logits[:i+1]))),
                "probability_max_error": float(np.max(np.abs(p - native_probs[:i+1]))),
                "write_max_error": float(np.max(np.abs(mix - actual))),
                "write_squared_error": float(np.sum((mix - actual)**2)),
                "write_dimensions": len(actual),
                "forbidden_mass": float(native_probs[i+1:].sum()),
                "incidence_winners": argmax_set(allowed),
                "native_probability_winners": argmax_set(native_probs[:i+1]),
                "key_agreement": argmax_set(allowed) == argmax_set(native_probs[:i+1]),
            })
    groups = {}
    for l, h, _ in sorted(expected):
        selected = [r for r in rows if (r["layer"], r["head"]) == (l, h)]
        groups[f"{l}.{h}"] = {"n": len(selected), "key_agreement": sum(r["key_agreement"] for r in selected) / len(selected),
            "write_max_error": max(r["write_max_error"] for r in selected)}
    return {"schema": "pdc.read-check.v1", "n": len(rows),
        "algebraic_identity": "native head_write is assigned scores.dot(v), scores=softmax(mask(q.dot(k.T)/sqrt(hd))); pre-output projection",
        "observer_hidden_max_error": 0,
        "reference_predictions": len(faithfulness), "reference_agreement": sum(faithfulness) / len(faithfulness),
        "key_agreement": sum(r["key_agreement"] for r in rows) / len(rows),
        "incidence_max_error": max(r["incidence_max_error"] for r in rows),
        "probability_max_error": max(r["probability_max_error"] for r in rows),
        "write_max_error": max(r["write_max_error"] for r in rows),
        "write_rms_error": float(np.sqrt(sum(r["write_squared_error"] for r in rows) / sum(r["write_dimensions"] for r in rows))),
        "forbidden_mass": max(r["forbidden_mass"] for r in rows),
        "per_head": groups, "instances": rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dump", type=Path, required=True)
    p.add_argument("--selection", type=Path, required=True)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    result = check(json.loads(args.dump.read_text()), json.loads(args.selection.read_text()), args.bundle)
    result["sha256"] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [args.dump, args.selection, Path(str(args.bundle) + ".fieldrun.json"), Path(str(args.bundle) + ".fieldrun.bin")]}
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ["instances", "sha256"]}, indent=2))
    if result["reference_agreement"] != 1 or result["forbidden_mass"] != 0:
        raise SystemExit("faithfulness or causal-mask check failed; retain the report")


if __name__ == "__main__":
    main()
