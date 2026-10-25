"""One-shot explicit final heldout audit; ordinary service CLI stays restricted."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

from pdc_service import annotate_theory, serve


def main():
    directory = Path('experiments/pdc')
    out = directory/'final_audit'
    out.mkdir(exist_ok=False)
    domain_path = directory/'r5_domain.json'
    probe_path = directory/'r8h_probe.json'
    domain = json.loads(domain_path.read_text())
    probe = json.loads(probe_path.read_text())
    expected = [c for c in domain['cases'] if c['split']=='heldout']
    assert len(expected)==25
    frozen = [domain_path, probe_path, directory/'FINAL_AUDIT_PREREG.md',
              Path('scripts/pdc_final_audit.py'), Path('scripts/pdc_service.py'),
              Path('scripts/pdc_final_audit_check.py'), Path('scripts/pdc_certificate.py'), Path('scripts/pdc_derive.py'),
              Path('bundles/gpt2.fieldrun.json'),Path('bundles/gpt2.fieldrun.bin')]
    manifest = {'sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in frozen},
                'case_ids':[c['id'] for c in expected], 'policy':'unchanged original R8-H',
                'heldout_reservation':'consumed by this final audit'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    timings=[]
    for arm in ['baseline','service']:
        selection=json.loads((directory/f'r8h_{arm}_selection.json').read_text())
        selection['contexts']=[c['ids'] for c in expected]
        request=out/f'{arm}_selection.json'
        request.write_text(json.dumps(selection,indent=2))
        capture_path=out/f'{arm}_capture.json'
        command=['target/release/fieldrun','--bundle','bundles/gpt2',
                 '--pdc-read-selection',str(request),'--pdc-read-dump',str(capture_path)]
        started=time.perf_counter()
        with (out/f'{arm}.log').open('w') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        native_seconds=time.perf_counter()-started
        capture=json.loads(capture_path.read_text())
        packet=serve(capture,domain,probe,Path('bundles/gpt2'),audit_split='heldout')
        annotate_theory(packet)
        packet['request_mode']='live_native_full_baseline' if arm=='baseline' else 'live_native_subset'
        packet['audit']='final heldout diagnostic; no theory soundness claim'
        for response in packet['responses']:
            for commitment in response['commitments']:
                commitment['memory']={'kind':'none'}
        (out/f'{arm}_bundle.json').write_text(json.dumps(packet,separators=(',',':'),allow_nan=False))
        timings.append(dict(arm=arm,native_process_wall_seconds=native_seconds,
                            end_to_end_wall_seconds=time.perf_counter()-started))
        (out/'timing.json').write_text(json.dumps({'runs':timings},indent=2))
        print(arm,native_seconds,flush=True)


if __name__=='__main__':
    main()
