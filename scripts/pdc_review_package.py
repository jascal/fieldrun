"""Create or verify a portable archive of the explicit stopped-stage inventory."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile

from pdc_inventory import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.verify:
        with tarfile.open(args.archive, 'r:gz') as archive:
            metadata = json.load(archive.extractfile('PDC_PACKAGE_MANIFEST.json'))
            actual = archive.getnames()
            expected = [f['path'] for f in metadata['files']]
            assert len(actual)==len(set(actual))==len(expected)+1
            assert set(actual)==set(expected)|{'PDC_PACKAGE_MANIFEST.json'}
            for entry in metadata['files']:
                member = archive.getmember(entry['path'])
                assert member.isfile() and member.size==entry['bytes']
                result = hashlib.sha256()
                with archive.extractfile(member) as stream:
                    for block in iter(lambda:stream.read(1024*1024),b''):
                        result.update(block)
                assert result.hexdigest()==entry['sha256'], entry['path']
        print(json.dumps({'verified_files':len(expected),'archive_sha256':digest(args.archive)}))
        return
    if args.archive.exists():
        raise ValueError('refusing to overwrite an existing package')
    root = args.workspace.resolve()
    inventory = root/'fieldrun/experiments/pdc/release_inventory.json'
    files = json.loads(inventory.read_text())['files']
    extras = ['PDC_GOAL.md','PDC_PROGRESS.md','PDC_REQUIREMENTS_AUDIT.md',
              'PDC_R5_STOP.md','PDC_REVIEW.md',str(inventory.relative_to(root))]
    for name in extras:
        path = root/name
        files.append(dict(path=name,bytes=path.stat().st_size,sha256=digest(path)))
    for entry in files:
        path = root/entry['path']
        assert path.stat().st_size==entry['bytes'] and digest(path)==entry['sha256'], entry['path']
    metadata = json.dumps(dict(schema='pdc.review-package.v1',
                              status='local stopped-stage snapshot; original goal incomplete',
                              files=files),indent=2).encode()
    with tarfile.open(args.archive,'w:gz',compresslevel=1) as archive:
        for entry in files:
            path = root/entry['path']
            info = archive.gettarinfo(str(path),arcname=entry['path'])
            info.uid=info.gid=info.mtime=0
            info.uname=info.gname=''
            with path.open('rb') as stream:
                archive.addfile(info,stream)
        info = tarfile.TarInfo('PDC_PACKAGE_MANIFEST.json')
        info.size=len(metadata)
        info.mtime=0
        archive.addfile(info,io.BytesIO(metadata))
    checksum = digest(args.archive)
    Path(str(args.archive)+'.sha256').write_text(checksum+'  '+args.archive.name+'\n')
    print(json.dumps(dict(files=len(files),bytes=args.archive.stat().st_size,sha256=checksum)))


if __name__=='__main__':
    main()
