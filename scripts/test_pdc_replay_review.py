import copy
import json
from pathlib import Path
import sys
import unittest
import numpy as np
from pdc_qwen35_replay_engineering_check import validate_replay_controls
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments/qwen3next'))
from deltanet_ref import gated_deltanet_qwen36

class ReplayReviewTests(unittest.TestCase):
    def test_numpy_wrapper_matches_discriminating_low_norm_golden(self):
        d=json.loads((ROOT/'experiments/pdc_qwen35_replay_engineering/normalization_golden.json').read_text())
        q,k,v,g,beta=[np.array(d[name]) for name in ('q','k','v','g','beta')]
        actual,_=gated_deltanet_qwen36(q,k,v,g,beta)
        np.testing.assert_allclose(actual,np.array(d['expected']).reshape(actual.shape),atol=1e-7,rtol=0)
    def test_payload_semantics_are_required_before_identity(self):
        selection={'contexts':[[1,2]]*3,'memory_writes':[None,{'site':'embed','write':[0.,0.]},{'site':'embed','write':[1.,0.]}]}
        cases=[{'final_hidden':[1.,2.]}]*3
        self.assertEqual(validate_replay_controls(selection,cases,False),[(0,1,2)])
        for kind in ('fake_zero','fake_nonzero','nan','width','context'):
            with self.subTest(kind=kind):
                bad=copy.deepcopy(selection)
                if kind=='fake_zero':bad['memory_writes'][1]['write'][0]=.1
                if kind=='fake_nonzero':bad['memory_writes'][2]['write']=[0.,0.]
                if kind=='nan':bad['memory_writes'][1]['write'][0]=float('nan')
                if kind=='width':bad['memory_writes'][1]['write']=[0.]
                if kind=='context':bad['contexts'][1]=[2,1]
                with self.assertRaises(ValueError):validate_replay_controls(bad,cases,False)
if __name__=='__main__':unittest.main()
