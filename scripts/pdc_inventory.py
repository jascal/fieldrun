"""Inventory local PDC evidence across repos as files, without cross-repo imports."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    root = args.workspace.resolve()
    if args.verify:
        manifest = json.loads(args.out.read_text())
        failures = []
        for entry in manifest['files']:
            path = root / entry['path']
            if not path.is_file() or path.stat().st_size != entry['bytes'] or digest(path) != entry['sha256']:
                failures.append(entry['path'])
        print(json.dumps({'checked':len(manifest['files']), 'failures':failures}, indent=2))
        if failures:
            raise SystemExit(1)
        return
    patterns = [
        'fieldrun/experiments/pdc/**/*', 'fieldrun/scripts/pdc*.py',
        'fieldrun/scripts/test_pdc*.py', 'fieldrun/src/composition.rs',
        'fieldrun/src/main.rs', 'fieldrun/src/bundle.rs',
        'fieldrun/src/**/*.rs', 'fieldrun/*.rs',
        'fieldrun/Cargo.toml', 'fieldrun/Cargo.lock',
        'pil/docs/notes/pdc*', 'pil/experiments/*pdc*.py', 'pil/pil/pdc*.py',
        'pil/pil/*.py', 'pil/pyproject.toml', 'pil/tests/test_pdc*.py',
        'pil/tests/test_p3.py', 'pil/results/pdc*/**/*',
        'i-orca/examples/pic_learn/ROOT',
        'i-orca/examples/pdc/*', 'i-orca/examples/pic_core/*.thy',
        'i-orca/examples/pic_core/ROOT', 'i-orca/examples/pic_learn/PIC_Learn.thy',
        'pdc/spec/*.md', 'pdc/paper/*.tex', 'pdc/paper/*.pdf',
        'pdc/paper/README.md', 'pdc/README.md', 'pdc/AGENTS.md',
    ]
    paths = set()
    for pattern in patterns:
        paths.update(p for p in root.glob(pattern) if p.is_file() and p.resolve()!=args.out.resolve())
    files = [dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=digest(p))
             for p in sorted(paths)]
    states = {}
    for repo in ['fieldrun','pil','i-orca','pdc']:
        def git(*options):
            result = subprocess.run(['git','-C',str(root/repo),*options],capture_output=True,text=True)
            return result.stdout.strip() if result.returncode==0 else None
        states[repo] = dict(head=git('rev-parse','HEAD'),branch=git('branch','--show-current'),
                            status=git('status','--short'),remotes=git('remote','-v'))
    result = dict(schema='pdc.local-review-inventory.v1',publication_status='local, uncommitted',
                  scope='stopped-stage v0.10; original north star not achieved',
                  model='pinned external GPT-2 bundle hashes in final_audit/manifest.json; weights not copied',
                  files=files,repository_states=states,total_bytes=sum(f['bytes'] for f in files))
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(files=len(files),bytes=result['total_bytes'],sha256=digest(args.out))))


if __name__=='__main__':
    main()
