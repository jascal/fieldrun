"""Freeze and run 54 unique Qwen two-hop contexts without adaptive changes."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["artifacts", "first", "diagnostic", "bundle", "runtime", "model"]:
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument(
        "--reference-python",
        type=Path,
        default=Path(sys.executable),
        help="Python interpreter with torch and transformers",
    )
    a = p.parse_args()
    root = a.artifacts
    assert not (root / "run_manifest.json").exists(), (
        "already frozen; do not restart on observation timeout"
    )
    selection = json.loads((root / "selection.json").read_text())
    assert len(selection["contexts"]) == 54 and selection["check_observer"] is False
    files = [
        root / n
        for n in [
            "PREREG.md",
            "requests.json",
            "selection.json",
            "assembly_verification.json",
        ]
    ]
    files += [
        a.first / n
        for n in ["domain.json", "probe.json", "run_manifest.json", "verification.json"]
    ]
    files += [a.diagnostic / n for n in ["results.json", "verification.json"]]
    files += [a.first / "bundle.json"] + sorted(a.first.glob("batch_*_capture.json"))
    files += list(a.model.glob("*.safetensors"))
    files += [
        a.model / "config.json",
        a.model / "tokenizer.json",
        a.model / "chat_template.jinja",
    ]
    files += list(Path("scripts").glob("pdc_qwen35_memory*.py"))
    files += [
        Path("scripts/pdc_qwen35_reference.py"),
        Path("scripts/pdc_qwen35_engineering_check.py"),
        Path("scripts/pdc_certificate.py"),
        Path("scripts/pdc_derive.py"),
        Path("scripts/pdc_theory_check.py"),
        Path(str(a.bundle) + ".fieldrun.json"),
        Path(str(a.bundle) + ".fieldrun.bin"),
        a.runtime,
    ]
    manifest = dict(
        schema="pdc.qwen35-two-hop-run.v1",
        sha256={str(p): sha(p) for p in files},
        native_forward_budget=54,
        reference_forward_budget=54,
        model_revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    )
    (root / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    progress = []
    for index, start in enumerate(range(0, 54, 9)):
        request = {**selection, "contexts": selection["contexts"][start : start + 9]}
        request_path = root / f"batch_{index:02}_selection.json"
        capture_path = root / f"batch_{index:02}_capture.json"
        assert not request_path.exists() and not capture_path.exists()
        request_path.write_text(json.dumps(request, indent=2) + "\n")
        began = time.perf_counter()
        with (root / f"batch_{index:02}.log").open("w") as log:
            subprocess.run(
                [
                    str(a.runtime.resolve()),
                    "--bundle",
                    str(a.bundle),
                    "--pdc-read-selection",
                    str(request_path),
                    "--pdc-qwen-source-dump",
                    str(capture_path),
                ],
                check=True,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        progress.append(
            dict(
                batch=index,
                cases=9,
                wall_seconds=time.perf_counter() - began,
                sha256=sha(capture_path),
            )
        )
        (root / "progress.json").write_text(json.dumps(progress, indent=2) + "\n")
        print("completed", sum(v["cases"] for v in progress), "of54", flush=True)
    subprocess.run(
        [
            "python3",
            "scripts/pdc_qwen35_memory_analyze.py",
            "--artifacts",
            str(root),
            "--first",
            str(a.first),
            "--bundle",
            str(a.bundle),
        ],
        check=True,
    )
    # Reference is a separate fixed verification budget, after the native model exits.
    subprocess.run(
        [
            str(a.reference_python),
            "scripts/pdc_qwen35_reference.py",
            "--model",
            str(a.model),
            "--selection",
            str(root / "selection.json"),
            "--dtype",
            "float16",
            "--out",
            str(root / "reference.json"),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
