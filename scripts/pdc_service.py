"""R8-H bundle service over native probe-only forwards; no full-vocabulary scorer."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

import numpy as np

from pdc_certificate import certify
from pdc_derive import Body, evaluate
from pdc_theory_check import judge_commitment

FROZEN_DOMAIN_SHA256 = "3dc65f9f9483b52dc803de69dd6bbae77810319f513edf927ff2539d843c39a3"
FROZEN_PROBE_SHA256 = "2361b4e36efa7928c6ee5f7f1c9aaff1154f90d31b40a32b4c1b2635b045cc34"
FROZEN_MODEL_MANIFEST_SHA256 = "322d79629f2e121c446560f57b71c5199e0d8a2a912c3a396137ca7eb207692c"
FROZEN_MODEL_WEIGHTS_SHA256 = "4edbcffed3bc19d571e679896ab3efebaa3802f8d7c9db3bdad31ff2416490f1"


def load_readout(stem, tokens):
    """Read only the selected unembedding rows and final-LN vectors."""
    manifest = json.loads(Path(str(stem)+".fieldrun.json").read_text())
    blob = np.memmap(str(stem)+".fieldrun.bin", mode="r", dtype=np.uint8)
    arrays = {}
    for entry in manifest["arrays"]:
        name = entry["name"]
        if name not in ["wte", "ln_f.weight", "ln_f.bias"]:
            continue
        dtype = {"f16": "<f2", "f32": "<f4"}[entry["dtype"]]
        data = np.frombuffer(blob, dtype=dtype, offset=entry["offset"], count=np.prod(entry["shape"])).reshape(entry["shape"])
        arrays[name] = (data[tokens] if name == "wte" else data).astype(np.float64)
    return manifest, arrays


def serve(capture, domain, probe, stem, *, audit_split="r5"):
    if audit_split not in {"r5", "heldout"}:
        raise ValueError("unknown audit split")
    tokens = probe["tokens"]
    if not tokens or tokens != sorted(set(tokens)):
        raise ValueError("a frozen nonempty sorted probe set is required")
    manifest, arrays = load_readout(stem, tokens)
    frame, norm_weight, beta = (arrays[name] for name in ["wte", "ln_f.weight", "ln_f.bias"])
    bias = frame @ beta
    rivals = domain["rivals"]
    answer_indices = [tokens.index(domain["token_map"][a]) for a in rivals]
    token_to_atom = {domain["token_map"][a]: a for a in rivals}
    guard_rivals = [token_to_atom.get(t, f"token:{t}") for t in tokens]
    available = [c for c in domain["cases"] if c["split"] == audit_split]
    by_context = {tuple(c["ids"]): c for c in available}
    if len(by_context) != len(available) or not 1 <= len(capture["cases"]) <= 75:
        raise ValueError("service requires one to 75 unambiguous R5-domain requests")
    try:
        cases = [by_context[tuple(c["ids"])] for c in capture["cases"]]
    except KeyError as error:
        raise ValueError("request outside the frozen R5 service domain; held-out inputs remain reserved") from error
    source_ids = ["embed"]+[name for layer in range(manifest["config"][0])
        for name in [f"L{layer}.attn", f"L{layer}.mlp"]]
    programs = {"full": list(range(len(source_ids))),
        "attention": [i for i, s in enumerate(source_ids) if s.endswith(".attn")],
        "mlp": [i for i, s in enumerate(source_ids) if s.endswith(".mlp")]}
    responses = []
    for native, case in zip(capture["cases"], cases, strict=True):
        if native["ids"] != case["ids"] or native["candidate_tokens"] != tokens:
            raise ValueError("runtime request or score coverage differs from the frozen input")
        if native["prediction_positions"] != [len(case["ids"])-1]:
            raise ValueError("one final-position decision is required")
        if native["scoring_scope"] not in ["candidate_subset", "full_vocabulary"]:
            raise ValueError("runtime scoring scope must be explicit")
        if native["observer_check_performed"]:
            raise ValueError("timed arm must use one forward, without the validation duplicate")
        if [s["id"] for s in native["sources"]] != source_ids:
            raise ValueError("the complete measured source program is required")
        raw = np.asarray([s["write"] for s in native["sources"]], dtype=np.float64)
        residual = np.asarray(native["reads"][0]["final_residual"], dtype=np.float64)
        sigma = np.sqrt(np.mean((residual-residual.mean())**2)+1e-5)
        sources = (raw-raw.mean(axis=1, keepdims=True))*norm_weight/sigma
        guard_scores = {name: (sources[coalition].sum(axis=0) @ frame.T+bias).tolist()
            for name, coalition in programs.items()}
        body_scores = {name: [scores[i] for i in answer_indices] for name, scores in guard_scores.items()}
        table = {a: [Body(name, scores[j]) for name, scores in body_scores.items()]
            for j, a in enumerate(rivals)}
        chart = evaluate(table, domain["temperature"])
        commitments = [certify(q, guard_scores, native["candidate_scores"][0], guard_rivals,
            guard_rivals, domain["gamma"]) for q in chart["queries"]]
        for commitment in commitments:
            commitment["fragile"] = chart["fragile"]
            commitment["theory"] = "not_loaded"
        responses.append({"id": case["id"], "a": case["a"], "b": case["b"],
            "context_ids": native["ids"], "chart": chart, "sources": sources.tolist(),
            "body_scores": body_scores, "guard_body_scores": guard_scores,
            "host_probe_scores": native["candidate_scores"][0], "commitments": commitments,
            "runtime": {k: native[k] for k in ["scoring_scope", "native_output_row_calls", "elapsed_forward_decode_us"]},
            "full_host_token": native["predictions"][0] if native["scoring_scope"] == "full_vocabulary" else None,
            "probe_host_token": tokens[int(np.argmax(native["candidate_scores"][0]))]})
    return {"schema": "pdc.served-bundle.v1", "arm": "R8-H", "evaluation_split": audit_split,
        "claims": {"theory_soundness": False, "certificate_scope": "fixed_lexical_probe_only"},
        "rivals": rivals, "candidate_tokens": [domain["token_map"][a] for a in rivals],
        "certificate_rivals": guard_rivals, "probe_tokens": tokens, "source_ids": source_ids,
        "body_programs": programs, "guard_frame": frame.tolist(), "guard_bias": bias.tolist(),
        "responses": responses, "work": {"native_output_row_calls": sum(r["runtime"]["native_output_row_calls"] for r in responses),
            "body_probe_dot_calls": len(responses)*len(programs)*len(tokens),
            "distinct_output_rows_per_context": len(tokens) if responses[0]["runtime"]["scoring_scope"] == "candidate_subset" else manifest["config"][4]}}


def annotate_theory(bundle):
    """External finite checker runs only after atom selection and certification."""
    for response in bundle["responses"]:
        truth = str(response["a"]+response["b"])
        for i, commitment in enumerate(response["commitments"]):
            judgment = judge_commitment(commitment, response["body_scores"], bundle["rivals"], truth)
            judgment["theory"] = {"id": "bounded integer addition", "verdict": judgment["theory_verdict"]}
            response["commitments"][i] = judgment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--capture", type=Path, help="offline audit input; not a fresh service request")
    source.add_argument("--runtime", type=Path, help="native binary: perform the actual request now")
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--case-id", help="one named R5-domain request; does not alter the frozen selection")
    parser.add_argument("--retain-capture", type=Path)
    parser.add_argument("--allow-full-baseline", action="store_true")
    for name in ["domain", "probe", "bundle", "out"]:
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--grade-addition", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    for path, expected in [(args.domain,FROZEN_DOMAIN_SHA256),(args.probe,FROZEN_PROBE_SHA256)]:
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            parser.error("R8-H requires the frozen domain and lexical probe; changed inputs are a different arm")
    native_seconds = None
    if args.runtime is not None:
        if args.selection is None:
            parser.error("--runtime requires a frozen --selection")
        selected = json.loads(args.selection.read_text())
        if args.case_id is not None:
            domain_cases = json.loads(args.domain.read_text())["cases"]
            matching = [c for c in domain_cases if c["id"] == args.case_id and c["split"] == "r5"]
            if len(matching) != 1:
                parser.error("--case-id must name a single non-held-out R5 context")
            selected["contexts"] = [matching[0]["ids"]]
        if selected.get("check_observer", True):
            parser.error("timed runtime requires one forward")
        if not selected.get("score_candidates_only") and not args.allow_full_baseline:
            parser.error("live service requires probe-only scoring; full runtime is an explicit diagnostic baseline")
        native_started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="pdc-service-") as directory:
            capture_path = Path(directory)/"native.json"
            selection_path = Path(directory)/"request.json"
            selection_path.write_text(json.dumps(selected))
            subprocess.run([str(args.runtime), "--bundle", str(args.bundle), "--pdc-read-selection",
                str(selection_path), "--pdc-read-dump", str(capture_path)], check=True)
            native_seconds = time.perf_counter()-native_started
            capture_bytes = capture_path.read_bytes()
            if args.retain_capture is not None:
                shutil.copyfile(capture_path, args.retain_capture)
    else:
        capture_bytes = args.capture.read_bytes()
    capture = json.loads(capture_bytes)
    if not args.allow_full_baseline and any(c["scoring_scope"] != "candidate_subset" for c in capture["cases"]):
        parser.error("full-vocabulary input is an offline baseline, never the live service")
    result = serve(capture, json.loads(args.domain.read_text()), json.loads(args.probe.read_text()), args.bundle)
    if args.grade_addition:
        annotate_theory(result)
    result["export_elapsed_seconds"] = time.perf_counter()-started
    result["request_mode"] = ("live_native_full_baseline" if args.allow_full_baseline else "live_native_subset") \
        if args.runtime is not None else "offline_capture_audit"
    result["native_process_wall_seconds"] = native_seconds
    result["native_capture_sha256"] = hashlib.sha256(capture_bytes).hexdigest()
    result["sha256"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [args.domain, args.probe, Path(str(args.bundle)+".fieldrun.json"),
                  Path(str(args.bundle)+".fieldrun.bin"), Path("experiments/pdc/R8H_PREREG.md")]}
    for path, expected in [(Path(str(args.bundle)+".fieldrun.json"),FROZEN_MODEL_MANIFEST_SHA256),
                           (Path(str(args.bundle)+".fieldrun.bin"),FROZEN_MODEL_WEIGHTS_SHA256)]:
        if result["sha256"][str(path)] != expected:
            raise ValueError("model differs from the frozen R8-H bundle; no served response will be written")
    args.out.write_text(json.dumps(result, separators=(",", ":"), allow_nan=False)+"\n")
    print(json.dumps({"responses": len(result["responses"]), "work": result["work"],
        "export_elapsed_seconds": result["export_elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
