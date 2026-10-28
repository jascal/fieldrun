"""Freeze, execute and audit a nonadaptive Qwen memory-content diagnostic."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load(p):
    return json.loads(Path(p).read_text())


def save(p, value):
    Path(p).write_text(json.dumps(value, indent=2) + '\n')


def freeze(a):
    from transformers import AutoTokenizer
    assert not (a.out / 'requests.json').exists()
    old = load(a.baseline / 'requests.json')
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    neutral = tok.encode('.', add_special_tokens=False)
    assert len(neutral) == 1
    assert all(tok.encode(str(i), add_special_tokens=False) == [15+i] for i in range(9))
    requests, contexts = [], []
    for r in old['requests']:
        if r['arm'] != 'generated':
            continue
        prefix = old['first_hops'][r['first_hop_id']]['original_ids']
        correct = r['a'] + r['b']
        wrong = (correct + 1) % 5
        assert wrong != correct
        wrong_ids = tok.encode(str(wrong), add_special_tokens=False)
        assert len(wrong_ids) == 1
        for arm, insert in [('neutral', neutral), ('wrong', wrong_ids)]:
            ids = prefix + insert + r['suffix_ids']
            text = tok.decode(prefix, skip_special_tokens=False) + tok.decode(insert) + tok.decode(r['suffix_ids'], skip_special_tokens=False)
            assert tok.encode(text, add_special_tokens=False) == ids
            requests.append(dict(a=r['a'], b=r['b'], c=r['c'], arm=arm,
                truth=correct+r['c'], inserted_value=None if arm=='neutral' else wrong,
                inserted_token=insert[0], following_inserted_truth=None if arm=='neutral' else wrong+r['c'],
                baseline_unique_index=r['unique_index'], unique_index=len(contexts)))
            contexts.append(ids)
    assert len(contexts) == len({tuple(x) for x in contexts}) == 54
    old_contexts={tuple(x['ids']) for x in old['unique_contexts']}
    assert not any(tuple(x) in old_contexts for x in contexts)
    probes=load(a.baseline/'selection.json')['candidate_tokens']
    save(a.out/'requests.json',dict(schema='pdc.qwen35-memory-controls-inputs.v1', requests=requests,
        neutral_text='.', neutral_token=neutral[0], baseline_requests_sha256=sha(a.baseline/'requests.json')))
    save(a.out/'selection.json',dict(contexts=contexts,candidate_tokens=probes,check_observer=False))
    print('Frozen 54 unique new inputs,27 paired triples; no model outputs inspected.')


def run(a):
    assert not (a.out/'run_manifest.json').exists(), 'Frozen run already started; poll original process instead.'
    selection=load(a.out/'selection.json')
    assert len(selection['contexts']) == 54
    files=[a.out/n for n in ['PREREG.md','requests.json','selection.json']]
    files += [a.baseline/n for n in ['requests.json','results.json','verification.json','run_manifest.json']]
    files += sorted(a.baseline.glob('batch_*_capture.json'))
    files += [Path(__file__),Path('scripts/pdc_qwen35_reference.py'),a.runtime]
    files += list(a.model.glob('*.safetensors'))
    files += [a.model/n for n in ['config.json','tokenizer.json','chat_template.jinja']]
    files += [Path(str(a.bundle)+s) for s in ['.fieldrun.json','.fieldrun.bin']]
    save(a.out/'run_manifest.json',dict(schema='pdc.qwen35-memory-controls-run.v1',
        sha256={str(p):sha(p) for p in files}, native_forward_budget=54,reference_forward_budget=54))
    progress=[]
    for i,start in enumerate(range(0,54,9)):
        request=a.out/f'batch_{i:02}_selection.json'
        capture=a.out/f'batch_{i:02}_capture.json'
        assert not request.exists() and not capture.exists()
        save(request,{**selection,'contexts':selection['contexts'][start:start+9]})
        began=time.perf_counter()
        with (a.out/f'batch_{i:02}.log').open('w') as log:
            subprocess.run([str(a.runtime.resolve()),'--bundle',str(a.bundle),'--pdc-read-selection',str(request),
                '--pdc-qwen-source-dump',str(capture)],stdout=log,stderr=subprocess.STDOUT,check=True)
        progress.append(dict(batch=i,cases=9,wall_seconds=time.perf_counter()-began,sha256=sha(capture)))
        save(a.out/'progress.json',progress)
        print(f'Native controls {sum(x["cases"] for x in progress)}/54',flush=True)
    subprocess.run([str(a.reference_python),'scripts/pdc_qwen35_reference.py','--model',str(a.model),
        '--selection',str(a.out/'selection.json'),'--dtype','float16','--out',str(a.out/'reference.json')],check=True)
    analyze(a)


def analyze(a):
    inputs=load(a.out/'requests.json')
    selection=load(a.out/'selection.json')
    native=[]
    for p in sorted(a.out.glob('batch_*_capture.json')):
        native.extend(load(p)['cases'])
    reference=load(a.out/'reference.json')
    assert len(native)==54
    # Exact output schema is supplied by the unchanged independent reference.
    ref_cases=reference['cases']
    assert len(ref_cases)==54
    token_map={str(i):15+i for i in range(9)}
    old=load(a.baseline/'results.json')
    counts={arm:dict(cases=27,correct=0,follows_wrong=0) for arm in ['neutral','wrong']}
    records=[]
    for i,(r,n,ref) in enumerate(zip(inputs['requests'],native,ref_cases,strict=True)):
        assert n['ids']==ref['ids']==selection['contexts'][i]
        assert n['prediction']==ref['prediction']
        correct=n['prediction']==token_map[str(r['truth'])]
        follows=r['arm']=='wrong' and n['prediction']==token_map[str(r['following_inserted_truth'])]
        counts[r['arm']]['correct']+=int(correct)
        counts[r['arm']]['follows_wrong']+=int(follows)
        records.append({**r,'native_token':n['prediction'],'reference_token':ref['prediction'],'correct':correct,'follows_wrong':follows})
    baseline_correct=old['arms']['generated']['host_correct']
    result=dict(schema='pdc.qwen35-memory-controls-result.v1',counts=counts,records=records,
        baseline_generated_correct=baseline_correct,baseline_empty_correct=old['arms']['rederive']['host_correct'],paired_triples=27,
        content_sensitivity_supported=(baseline_correct-counts['wrong']['correct'] >= 6 and counts['wrong']['follows_wrong'] >= 14),
        neutral_improvement_bar_met=(counts['neutral']['correct']-old['arms']['rederive']['host_correct'] >= 2),
        interpretation='Development content/format diagnostic; no admitted replay or fresh confirmation.',
        native_reference_matches=54,new_native_forwards=54,new_reference_forwards=54)
    save(a.out/'results.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['freeze','run','analyze'])
    for name in ['out','baseline','model']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--runtime',type=Path)
    p.add_argument('--bundle',type=Path)
    p.add_argument('--reference-python',type=Path,default=Path(sys.executable))
    a=p.parse_args()
    {'freeze':freeze,'run':run,'analyze':analyze}[a.action](a)


if __name__=='__main__':
    main()
