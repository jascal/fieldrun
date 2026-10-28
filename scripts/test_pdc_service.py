import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from pdc_service import load_readout


class ServiceTests(unittest.TestCase):
    def test_readout_loader_gathers_only_requested_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            stem = Path(temporary)/"tiny"
            table = np.arange(40,dtype=np.float16).reshape(20,2)
            beta, gamma = np.array([1.,2.],dtype=np.float16), np.array([3.,4.],dtype=np.float16)
            arrays, offset, pieces = [], 0, []
            for name, value in [("wte",table), ("ln_f.bias",beta), ("ln_f.weight",gamma)]:
                data = value.astype("<f2").tobytes()
                arrays.append({"name":name,"shape":list(value.shape),"offset":offset,"dtype":"f16"})
                pieces.append(data)
                offset += len(data)
            arrays.append({"name":"unneeded","dtype":"not-supported","shape":[100],"offset":offset})
            Path(str(stem)+".fieldrun.json").write_text(json.dumps({"arrays":arrays}))
            Path(str(stem)+".fieldrun.bin").write_bytes(b"".join(pieces))
            _, values = load_readout(stem,[2,17])
            self.assertEqual(values["wte"].shape,(2,2))
            np.testing.assert_array_equal(values["wte"],table[[2,17]])
            self.assertEqual(set(values),{"wte","ln_f.bias","ln_f.weight"})

    def test_live_mode_rejects_full_scoring_before_invoking_runtime(self):
        self._reject("r8h_baseline_selection.json",None,"live service requires probe-only scoring")

    def test_heldout_request_is_rejected_before_invoking_runtime(self):
        self._reject("r8h_service_selection.json","t3-a0-b0","non-held-out R5 context")

    def _reject(self, selection, case_id, message):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/"response.json"
            command = [sys.executable,str(root/"scripts/pdc_service.py"),"--runtime","/no-such-runtime",
                "--selection",str(root/"experiments/pdc"/selection),"--domain",
                str(root/"experiments/pdc/r5_domain.json"),"--probe",str(root/"experiments/pdc/r8h_probe.json"),
                "--bundle",str(root/"bundles/gpt2"),"--out",str(output)]
            if case_id is not None:
                command.extend(["--case-id",case_id])
            result = subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(result.returncode,2)
            self.assertIn(message,result.stderr)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
