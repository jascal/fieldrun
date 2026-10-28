import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from pdc_service import serve
from pdc_final_audit_check import require_published_baseline

class ReviewChecks(unittest.TestCase):
    def setUp(self):
        self.native = dict(ids=[0,1], candidate_tokens=[0,1], prediction_positions=[1],
            scoring_scope='candidate_subset', observer_check_performed=False,
            sources=[dict(id=n,position=1,write=v) for n,v in
                [('embed',[1.,2.]),('L0.attn',[0.,0.]),('L0.mlp',[0.,0.])]],
            reads=[dict(query_position=1,final_residual=[1.,2.])],
            candidate_scores=[[0.,1.]],predictions=[1],native_output_row_calls=2,elapsed_forward_decode_us=1)
        self.domain=dict(rivals=['a','b'],token_map={'a':0,'b':1},temperature=1.,gamma=.5,
            cases=[dict(id='x',ids=[0,1],split='r5',a=0,b=0)])
    def run_service(self,native):
        with patch('pdc_service.load_readout',return_value=({'config':[1,1,2,2,2]},
            {'wte':np.eye(2),'ln_f.weight':np.ones(2),'ln_f.bias':np.zeros(2)})):
            return serve({'cases':[native]},self.domain,{'tokens':[0,1]},Path('unused'))
    def test_final_position_source_sum_passes(self):
        self.assertEqual(len(self.run_service(self.native)['responses']),1)
    def test_service_rejects_early_writes_early_reads_and_broken_sum(self):
        for change in ('source','read','sum','empty','nonfinite'):
            with self.subTest(change=change):
                bad=copy.deepcopy(self.native)
                if change=='source':bad['sources'][0]['position']=0
                if change=='read':bad['reads'][0]['query_position']=0
                if change=='sum':bad['reads'][0]['final_residual'][0]+=1
                if change=='empty':bad['reads']=[]
                if change=='nonfinite':bad['sources'][0]['write'][0]=float('nan')
                with self.assertRaises(ValueError):self.run_service(bad)
    def test_consistent_changed_pair_cannot_replace_published_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'baseline.json'
            p.write_text(json.dumps({'cases':[self.native]}));frozen=hashlib.sha256(p.read_bytes()).hexdigest()
            require_published_baseline(p,frozen)
            changed=copy.deepcopy(self.native);changed['predictions']=[0]
            p.write_text(json.dumps({'cases':[changed]}))
            (Path(directory)/'service.json').write_bytes(p.read_bytes())
            with self.assertRaises(ValueError):require_published_baseline(p,frozen)

if __name__=='__main__':unittest.main()
