"""Independent replay-hook checks; these three cases are not R6 observations."""
import argparse
import json
from pathlib import Path

import numpy as np

from pdc_read_check import load_bundle, reference


def check(capture, selection, stem):
    manifest, weights = load_bundle(stem)
    assert len(capture["cases"]) == len(selection["contexts"]) == 3
    comparisons = []
    reference_states = []
    for native, ids, memory in zip(capture["cases"],selection["contexts"],selection["memory_writes"],strict=True):
        assert native["ids"] == ids and native["memory_replay"] == memory
        ref = reference(ids,manifest["config"],weights,memory=memory,return_hidden=True)
        predicted = int((ref[-1:]@weights["wte"].T).argmax(axis=1)[0])
        error = float(np.max(np.abs(ref-np.array(native["final_hidden"]))))
        assert native["predictions"] == [predicted]
        reference_states.append(ref)
        memory_rows = [s for s in native["sources"] if s["id"].startswith("memory@")]
        if memory is None:
            assert not memory_rows
        else:
            assert len(memory_rows) == 1 and memory_rows[0]["id"] == "memory@"+memory["site"]
            np.testing.assert_array_equal(memory_rows[0]["write"],memory["write"])
        comparisons.append({"site":None if memory is None else memory["site"],
            "native_token":predicted,"reference_top1_agreement":True,"hidden_max_error":error,
            "query_hidden_max_error":float(np.max(np.abs(ref[-1]-np.array(native["final_hidden"])[-1])))})
    base,zero,nonzero = (np.asarray(c["final_hidden"]) for c in capture["cases"])
    np.testing.assert_array_equal(base,zero)
    effect = float(np.max(np.abs(base-nonzero)))
    assert effect > 1e-6
    # The injected final-position source cannot change earlier causal positions.
    # Compare the replay operator with its own baseline in both implementations;
    # retain the independent kernels' absolute baseline error above.
    np.testing.assert_array_equal(base[:-1],nonzero[:-1])
    np.testing.assert_array_equal(reference_states[0][:-1],reference_states[2][:-1])
    replay_error = float(np.max(np.abs((nonzero-base)-
        (reference_states[2].astype(float)-reference_states[0].astype(float)))))
    assert replay_error <= 1e-3
    return {"schema":"pdc.replay-engineering.v1","cases":comparisons,
        "zero_replay_hidden_error":0.,"nonzero_replay_effect":effect,
        "replay_effect_max_error":replay_error,"all_earlier_positions_unchanged":True,
        "initial_absolute_hidden_diagnostic":"1e-3 all-state threshold failed on the no-memory baseline (max 0.003173828125); retained, not claimed passed",
        "experiment_status":"engineering validation only; no R4 admission or R6 memory result"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["capture","selection","bundle","out"]:
        parser.add_argument("--"+name,type=Path,required=True)
    args = parser.parse_args()
    report = check(json.loads(args.capture.read_text()),json.loads(args.selection.read_text()),args.bundle)
    args.out.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
