import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from pdc_qwen35_analyze import analyze
from pdc_qwen35_singletons import witness_agreement

class ReconstructionTests(unittest.TestCase):
    def fixture(self,root,damage=None):
        config={'config':[1,1,2,0,0,2],'config_f':[0.,1e-5]}
        hidden=np.array([2.,0.])/np.sqrt(2.+1e-5)
        base=dict(ids=[0,1],candidate_tokens=[0,1],observer_check_performed=False,
            native_output_row_calls=2,prediction=0,final_residual=[2.,0.],final_hidden=hidden.tolist(),
            candidate_scores=hidden.tolist(),sources=[dict(id=n,write=v) for n,v in
                [('embed',[2.,0.]),('L0.attn',[0.,0.]),('L0.mlp',[0.,0.])]])
        rows=[copy.deepcopy(base) for _ in range(100)]
        if damage:damage(rows[-1])
        domain=dict(token_map={'0':0,'1':1},temperature=1.,gamma=.5,cases=[
            dict(id=str(i),ids=[0,1],a=0,b=0,truth='0',split='development' if i<50 else 'heldout') for i in range(100)])
        for name,data in [('domain.json',domain),('probe.json',{'tokens':[0,1]}),('batch_00_capture.json',{'cases':rows})]:
            (root/name).write_text(json.dumps(data))
        return config,np.eye(2),np.ones(2)
    def test_all_100_cases_must_pass_before_grade_is_written(self):
        def source(r):r['sources'][0]['write'][0]+=1
        def hidden(r):r['final_hidden'][0]+=.1
        def score(r):r['candidate_scores'][0]+=.1
        def nonfinite(r):r['final_hidden'][0]=float('nan')
        for damage in (source,hidden,score,nonfinite):
            with self.subTest(damage=damage.__name__),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);frame=self.fixture(root,damage)
                with patch('pdc_qwen35_analyze.read_frame',return_value=frame):
                    with self.assertRaises(ValueError):analyze(root,Path('unused'))
                self.assertFalse((root/'results.json').exists())
                self.assertFalse((root/'bundle.json').exists())
    def test_valid_capture_grades_and_counts_actual_candidate_agreement(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);frame=self.fixture(root)
            with patch('pdc_qwen35_analyze.read_frame',return_value=frame):result=analyze(root,Path('unused'))
            self.assertEqual(result['reconstruction_gate'],'passed_all_100_cases')
            row=result['summary']['heldout']['viterbi']
            self.assertEqual(row['covered_candidate_agreements'],50)
            self.assertEqual(row['covered_candidate_agreement'],1.)
    def test_singleton_agreement_is_a_measurement(self):
        rows=[{'host_token':1,'exact_witness_token':1},{'host_token':2,'exact_witness_token':1}]
        self.assertEqual(witness_agreement(rows,'exact'),(1,.5))
        self.assertEqual(witness_agreement([],'exact'),(0,None))
if __name__=='__main__':unittest.main()
