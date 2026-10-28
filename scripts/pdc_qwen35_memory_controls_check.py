"""Independent standard-library input, prediction and count audit of memory controls."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def read(p):
    return json.loads(p.read_text())


def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for n in ['artifacts','baseline','domain','out']:
        parser.add_argument('--'+n,type=Path,required=True)
    parser.add_argument('--verify-original-paths',action='store_true',
        help='Verify every frozen run input/model/runtime hash at its original recorded path.')
    a=parser.parse_args()
    root=a.artifacts
    inputs=read(root/'requests.json')
    selection=read(root/'selection.json')
    result=read(root/'results.json')
    old=read(a.baseline/'requests.json')
    baseline=read(a.baseline/'results.json')
    tokens=read(a.domain)['token_map']
    assert digest(a.baseline/'requests.json')==inputs['baseline_requests_sha256']
    original={ (r['a'],r['b'],r['c']):r for r in old['requests'] if r['arm']=='generated'}
    assert len(original)==27
    contexts=selection['contexts']
    assert len(contexts)==len({tuple(v) for v in contexts})==54
    old_ids={tuple(x['ids']) for x in old['unique_contexts']}
    assert not any(tuple(v) in old_ids for v in contexts)
    native=[]
    for p in sorted(root.glob('batch_*_capture.json')):
        native.extend(read(p)['cases'])
    reference=read(root/'reference.json')['cases']
    assert len(native)==len(reference)==len(inputs['requests'])==54
    c={name:Counter() for name in ['neutral','wrong']}
    triples=set()
    rows=0
    for i,(req,n,ref,saved) in enumerate(zip(inputs['requests'],native,reference,result['records'],strict=True)):
        key=(req['a'],req['b'],req['c'])
        triples.add(key)
        base=original[key]
        prefix=old['first_hops'][base['first_hop_id']]['original_ids']
        assert n['ids']==ref['ids']==contexts[i]
        assert contexts[i]==prefix+[req['inserted_token']]+base['suffix_ids']
        assert n['prediction']==ref['prediction']
        assert len(n['sources'])==65 and n['observer_check_performed'] is False
        assert n['native_output_row_calls']==248320
        rows+=n['native_output_row_calls']
        arm=req['arm']
        correct=sum(key)
        following=(key[0]+key[1]+1)%5+key[2]
        if arm=='wrong':
            assert req['inserted_token']==tokens[str((key[0]+key[1]+1)%5)]
            assert req['following_inserted_truth']==following
        else:
            assert arm=='neutral' and req['inserted_token']==inputs['neutral_token']
        truth=n['prediction']==tokens[str(correct)]
        follows=arm=='wrong' and n['prediction']==tokens[str(following)]
        assert req['truth']==correct and saved['correct']==truth and saved['follows_wrong']==follows
        assert saved['native_token']==saved['reference_token']==n['prediction']
        c[arm].update(cases=1,correct=int(truth),follows_wrong=int(follows))
    assert len(triples)==27
    assert {k:dict(v) for k,v in c.items()}==result['counts']
    bc=baseline['arms']['generated']['host_correct']
    be=baseline['arms']['rederive']['host_correct']
    assert bc==result['baseline_generated_correct'] and be==result['baseline_empty_correct']
    assert result['content_sensitivity_supported']==(bc-c['wrong']['correct']>=6 and c['wrong']['follows_wrong']>=14)
    assert result['neutral_improvement_bar_met']==(c['neutral']['correct']-be>=2)
    verified=0
    if a.verify_original_paths:
        for name,expected in read(root/'run_manifest.json')['sha256'].items():
            assert digest(Path(name))==expected,name
            verified+=1
    report=dict(gate='passed',contexts_checked=54,paired_triples=27,
        native_reference_matches=54,native_output_rows=rows,counts=c,
        original_path_hashes_verified=verified,new_model_forwards=0,
        scope='Independent assembly, full-host label/reference and count audit; no source-score or causal-replay claim.')
    a.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
