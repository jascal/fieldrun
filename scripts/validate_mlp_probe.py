#!/usr/bin/env python3
"""Retained FP32/eager upstream gate for /probe/mlp. Run reference and check separately.

The two phases need not keep the upstream model and fieldrun resident together.
Python/Transformers are build-time validation dependencies only.
"""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


LAYERS = (7, 15, 23)
UNITS = ((7, 0), (15, 17), (23, 31))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for data in iter(lambda: f.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def save(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def prepare(checkpoint, out):
    from transformers import AutoTokenizer
    out.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    texts = ["The capital of France is", "Two plus two equals", "She put the book on the",
             "A small program can copy a sequence:"]
    rows = [{"id": i, "kind": "text", "ids": tokenizer.encode(t, add_special_tokens=False)}
            for i, t in enumerate(texts)]
    for i in range(4):
        seq = [200 + 97 * i + j * 13 for j in range(8 + 4 * i)]
        rows.append({"id": len(rows), "kind": "repeat", "ids": seq * 2 + seq[:len(seq)//2]})
    for i in range(4):
        seq = [1400 + 97 * i + j * 13 for j in range(8 + 4 * i)]
        p = len(seq)//2
        gap = [2000 + i, *seq[p-2:p], 3000 + i]
        rows.append({"id": len(rows), "kind": "distractor", "ids": seq + gap + seq + gap + seq[:p]})
    save(out / "inputs.json", rows)
    save(out / "protocol.json", {"checkpoint": str(checkpoint.resolve()), "layers": LAYERS,
         "units": UNITS, "atol": 0.002, "rtol": 0.002, "inputs_sha256": sha(out / "inputs.json"),
         "checkpoint_sha256": {p.name: sha(p) for p in sorted(checkpoint.iterdir())
                                if p.suffix in (".safetensors", ".json")},
         "reference_precision": "FP32", "attention": "eager", "threads": 2,
         "interventions": ["zero", "replace_one", "identity"],
         "scope": "12 frozen contexts; full native rows at three layers; last-position scalar interventions"})


def reference(checkpoint, out):
    import numpy as np
    import torch
    import transformers
    from transformers import AutoModelForCausalLM
    protocol = json.loads((out / "protocol.json").read_text())
    assert sha(out / "inputs.json") == protocol["inputs_sha256"]
    assert all(sha(checkpoint / name) == digest for name, digest in protocol["checkpoint_sha256"].items())
    if (out / "reference.json").exists():
        raise ValueError("reference already exists")
    torch.set_num_threads(protocol["threads"])
    model = AutoModelForCausalLM.from_pretrained(checkpoint, dtype=torch.float32,
              attn_implementation="eager", local_files_only=True).eval()
    arrays, records = {}, []
    for row in json.loads((out / "inputs.json").read_text()):
        ids = torch.tensor([row["ids"]])
        capture, hooks = {}, []
        for layer in LAYERS:
            mlp = model.model.layers[layer].mlp
            def capture_output(_module, _inputs, output, *, l=layer, channel="gate"):
                capture[(l, channel)] = output[0, -1].detach().clone()
            hooks.append(mlp.gate_proj.register_forward_hook(capture_output))
            def capture_value(_module, _inputs, output, *, l=layer):
                capture[(l, "value")] = output[0, -1].detach().clone()
            hooks.append(mlp.up_proj.register_forward_hook(capture_value))
            def capture_product(_module, inputs, *, l=layer):
                capture[(l, "product")] = inputs[0][0, -1].detach().clone()
            hooks.append(mlp.down_proj.register_forward_pre_hook(capture_product))
        with torch.inference_mode():
            base = model(ids, use_cache=False).logits[0, -1].float()
        for h in hooks:
            h.remove()
        for (layer, channel), vector in capture.items():
            arrays[f"{row['id']}_{layer}_{channel}"] = vector.numpy()
        readouts = sorted({0, 1, 200, 39999, *torch.topk(base, 2).indices.tolist()})
        record = {"id": row["id"], "next": int(base.argmax()), "readouts": readouts,
                  "logits": base[readouts].tolist(), "interventions": []}
        for layer, neuron in UNITS:
            for mode, value in (("zero", 0.0), ("replace_one", 1.0),
                                ("identity", float(capture[(layer, "product")][neuron]))):
                def patch(_module, inputs, *, n=neuron, v=value):
                    product = inputs[0].clone()
                    product[0, -1, n] = v
                    return (product,)
                h = model.model.layers[layer].mlp.down_proj.register_forward_pre_hook(patch)
                with torch.inference_mode():
                    logits = model(ids, use_cache=False).logits[0, -1].float()
                h.remove()
                if mode == "identity" and not torch.equal(logits, base):
                    raise ValueError("upstream identity patch changed logits")
                record["interventions"].append({"layer": layer, "neuron": neuron, "mode": mode,
                       "value": value, "next": int(logits.argmax()), "logits": logits[readouts].tolist()})
        records.append(record)
        print(f"upstream {row['id']+1}/12", flush=True)
    np.savez_compressed(out / "upstream-activations.npz", **arrays)
    save(out / "reference.json", {"torch": torch.__version__, "transformers": transformers.__version__,
        "protocol_sha256": sha(out / "protocol.json"), "script_sha256": sha(__file__),
        "activations_sha256": sha(out / "upstream-activations.npz"), "records": records})


def check(port, out, bundle, binary):
    import numpy as np
    protocol = json.loads((out / "protocol.json").read_text())
    upstream = json.loads((out / "reference.json").read_text())
    assert sha(out / "protocol.json") == upstream["protocol_sha256"]
    assert sha(out / "inputs.json") == protocol["inputs_sha256"]
    assert sha(out / "upstream-activations.npz") == upstream["activations_sha256"]
    if (out / "report.json").exists():
        raise ValueError("check report already exists; preserve failed gates")
    expected = np.load(out / "upstream-activations.npz")
    arrays, records, failures = {}, [], []

    def request(ids, probe):
        req = urllib.request.Request(f"http://127.0.0.1:{port}/probe/mlp",
               json.dumps({"ids": ids, "probe": probe}).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as response:
            data = json.load(response)
        if "error" in data:
            raise ValueError(data)
        return data

    def compare(label, got, want):
        a, b = np.asarray(got, dtype=np.float64), np.asarray(want, dtype=np.float64)
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError(f"invalid array {label}")
        diff = np.abs(a-b)
        bad = diff > protocol["atol"] + protocol["rtol"]*np.abs(b)
        if bad.any():
            failures.append({"check": label, "nfail": int(bad.sum()), "indices": np.flatnonzero(bad).tolist()})
        return {"max_abs": float(diff.max()), "nfail": int(bad.sum()), "n": a.size}

    for row, ref in zip(json.loads((out / "inputs.json").read_text()), upstream["records"], strict=True):
        assert row["id"] == ref["id"]
        base = request(row["ids"], {"layers": LAYERS, "readouts": ref["readouts"]})
        if base["next"] != ref["next"]:
            failures.append({"check": f"{row['id']}:top1", "got": base["next"], "expected": ref["next"]})
        rec = {"id": row["id"], "next": base["next"], "reference_next": ref["next"], "activations": {}}
        rec["logits"] = compare(f"{row['id']}:logits", [v for _,v in base["logits"]], ref["logits"])
        by_unit = {(o["layer"],o["neuron"]): o for o in base["observations"]}
        for layer in LAYERS:
            width = len(expected[f"{row['id']}_{layer}_gate"])
            for channel in ("gate", "value", "product"):
                key = f"{row['id']}_{layer}_{channel}"
                a = [by_unit[(layer,n)][channel] for n in range(width)]
                arrays[key] = np.asarray(a, dtype=np.float32)
                rec["activations"][key] = compare(key, a, expected[key])
        rec["interventions"] = []
        for item in ref["interventions"]:
            value = by_unit[(item["layer"],item["neuron"])]["product"] if item["mode"] == "identity" else item["value"]
            patch = {k: item[k] for k in ("layer", "neuron")}
            patch.update(position=len(row["ids"])-1, value=value)
            got = request(row["ids"], {"patches": [patch], "readouts": ref["readouts"]})
            name = f"{row['id']}:{item['layer']}:{item['mode']}"
            if got["next"] != item["next"]:
                failures.append({"check": name+":top1", "got": got["next"], "expected": item["next"]})
            if item["mode"] == "identity" and got["logits"] != base["logits"]:
                failures.append({"check": name+":identity"})
            rec["interventions"].append({**patch, "mode": item["mode"], "next": got["next"],
                "reference_next": item["next"], "logits": got["logits"],
                "comparison": compare(name, [v for _,v in got["logits"]], item["logits"])})
        records.append(rec)
        print(f"fieldrun {row['id']+1}/12", flush=True)
    np.savez_compressed(out / "fieldrun-activations.npz", **arrays)
    save(out / "report.json", {"tag": "empirical", "passed": not failures, "failures": failures,
         "records": records, "reference_sha256": sha(out / "reference.json"),
         "binary_sha256": sha(binary), "bundle_sha256": {s: sha(str(bundle)+s)
             for s in (".fieldrun.bin", ".fieldrun.json")},
         "activations_sha256": sha(out / "fieldrun-activations.npz"), "script_sha256": sha(__file__)})
    if failures:
        raise SystemExit(f"gate failed: {len(failures)} checks; see retained report")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("prepare", "reference", "check"))
    p.add_argument("out", type=Path)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--port", type=int)
    p.add_argument("--bundle", type=Path)
    p.add_argument("--binary", type=Path)
    a = p.parse_args()
    if a.phase == "check":
        if not (a.port and a.bundle and a.binary):
            p.error("check needs --port, --bundle and --binary")
        check(a.port, a.out, a.bundle, a.binary)
    else:
        if a.checkpoint is None:
            p.error("prepare/reference need --checkpoint")
        (prepare if a.phase == "prepare" else reference)(a.checkpoint, a.out)


if __name__ == "__main__":
    main()
