"""Freeze the engineering-gated comparison, then capture each input exactly once."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["runtime", "bundle", "model", "artifacts"]:
        parser.add_argument("--" + name, type=Path, required=True)
    a = parser.parse_args()
    directory = a.artifacts
    for name, n in [
        ("tiny_verification.json", 3),
        ("engineering_verification.json", 5),
    ]:
        gate = json.loads((directory / name).read_text())
        assert (
            gate["cases"] == n
            and gate["reference_agreement"] == 1
            and gate["observer_effect"] == 0
        )
    domain = json.loads((directory / "domain.json").read_text())
    selection = json.loads((directory / "selection.json").read_text())
    assert selection["contexts"] == [c["ids"] for c in domain["cases"]]
    manifest_path = directory / "run_manifest.json"
    assert not manifest_path.exists(), (
        "one-shot comparison already frozen; inspect existing progress rather than restart"
    )
    frozen = [
        directory / name
        for name in [
            "PREREG.md",
            "domain.json",
            "probe.json",
            "selection.json",
            "tiny_verification.json",
            "engineering_verification.json",
        ]
    ]
    frozen += list(Path("scripts").glob("pdc_qwen35*.py"))
    frozen += [
        Path("scripts/pdc_certificate.py"),
        Path("scripts/pdc_derive.py"),
        Path("scripts/pdc_theory_check.py"),
        Path("scripts/pdc_service_inputs.py"),
        Path("src/qwen35moe.rs"),
        Path("src/main.rs"),
        a.runtime,
        Path(str(a.bundle) + ".fieldrun.json"),
        Path(str(a.bundle) + ".fieldrun.bin"),
    ]
    frozen += list(a.model.glob("*.safetensors")) + [
        a.model / "config.json",
        a.model / "tokenizer.json",
        a.model / "tokenizer_config.json",
        a.model / "chat_template.jinja",
    ]
    manifest = dict(
        schema="pdc.qwen35-comparison-run.v1",
        checkpoint="Qwen/Qwen3.5-4B",
        revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        sha256={str(p): sha(p) for p in frozen},
        case_ids=[c["id"] for c in domain["cases"]],
        readout_policy="fixed; no development-conditioned tuning",
        cases=100,
        batch_size=10,
        final_GPT2_heldout_used=0,
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    progress = []
    for index in range(10):
        start = index * 10
        request = {**selection, "contexts": selection["contexts"][start : start + 10]}
        path = directory / f"batch_{index:02}_selection.json"
        capture = directory / f"batch_{index:02}_capture.json"
        assert not path.exists() and not capture.exists()
        path.write_text(json.dumps(request, indent=2) + "\n")
        command = [
            str(a.runtime.resolve()),
            "--bundle",
            str(a.bundle),
            "--pdc-read-selection",
            str(path),
            "--pdc-qwen-source-dump",
            str(capture),
        ]
        began = time.perf_counter()
        with (directory / f"batch_{index:02}.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        progress.append(
            dict(
                batch=index,
                cases=10,
                wall_seconds=time.perf_counter() - began,
                capture_sha256=sha(capture),
            )
        )
        (directory / "progress.json").write_text(json.dumps(progress, indent=2) + "\n")
        print(
            "finished batch",
            index,
            "cases",
            (index + 1) * 10,
            "wall",
            progress[-1]["wall_seconds"],
            flush=True,
        )
    subprocess.run(
        [
            "python3",
            "scripts/pdc_qwen35_analyze.py",
            "--artifacts",
            str(directory),
            "--bundle",
            str(a.bundle),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
