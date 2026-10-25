"""Run the single frozen timing pair; evaluation is a separate audit command."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    args = parser.parse_args()
    directory = args.artifacts
    records = []
    for arm in ["baseline", "service"]:
        capture = directory/f"r8h_timed_{arm}_capture.json"
        response = directory/f"r8h_{arm}_bundle.json"
        if capture.exists() or response.exists():
            raise ValueError("refusing to overwrite a frozen timing-pair result")
        command = [sys.executable, "scripts/pdc_service.py", "--runtime", str(args.runtime),
            "--selection", str(directory/f"r8h_{arm}_selection.json"), "--retain-capture", str(capture),
            "--domain", str(directory/"r5_domain.json"), "--probe", str(directory/"r8h_probe.json"),
            "--bundle", str(args.bundle), "--out", str(response), "--grade-addition"]
        if arm == "baseline":
            command.append("--allow-full-baseline")
        print(f"starting {arm}", flush=True)
        started = time.perf_counter()
        with (directory/f"r8h_{arm}_run.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        records.append({"arm": arm, "end_to_end_wall_seconds": time.perf_counter()-started,
            "capture": str(capture), "bundle": str(response), "command": command})
        print(f"finished {arm}: {records[-1]['end_to_end_wall_seconds']:.3f}s", flush=True)
        (directory/"r8h_timing_progress.json").write_text(json.dumps(records, indent=2)+"\n")
    report = {"schema": "pdc.service-timing.v1", "runs": records,
        "sha256": {name: hashlib.sha256((directory/name).read_bytes()).hexdigest()
            for name in ["R8H_PREREG.md", "R8H_TIMING_ADDENDUM.md", "r8h_probe.json",
                         "r8h_baseline_selection.json", "r8h_service_selection.json"]}}
    (directory/"r8h_timing.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
