"""Unpack the 44 preserved synthetic input assets and write 11 current manifests.

Uses only Python 3.11+ standard library; no database, model or process is started.
Original manifest bytes are retained beside separately named current manifests.
"""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def prepare(output):
    assets = Path(__file__).resolve().parent / 'lifecycle-inputs-v1'
    lineage = json.loads((assets / 'artifact-lineage.json').read_text(encoding='utf-8'))
    archive = assets / 'logical-inputs.tar.gz'
    if sha(archive) != lineage['public_archive_sha256']:
        raise ValueError('Archive hash mismatch')
    expected = {row['path']: row for row in lineage['members']}
    if len(expected) != 44 or len(lineage['members']) != 44:
        raise ValueError('Expected exactly 44 distinct input assets')
    output.mkdir()
    with tarfile.open(archive, 'r:gz') as source:
        members = source.getmembers()
        if len(members) != 44 or {member.name for member in members} != set(expected):
            raise ValueError('Unexpected archive member list; retain failed output')
        for member in members:
            path = PurePosixPath(member.name)
            if not member.isfile() or path.is_absolute() or '..' in path.parts or '\\' in member.name or ':' in member.name:
                raise ValueError('Only regular relative input files are allowed')
            row = expected[member.name]
            if member.size != row['bytes']:
                raise ValueError('Input size mismatch')
            target = output.joinpath(*path.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            with source.extractfile(member) as reader, target.open('xb') as writer:
                while block := reader.read(1024 * 1024):
                    writer.write(block)
                    digest.update(block)
            if digest.hexdigest() != row['public_member_sha256']:
                raise ValueError('Input hash mismatch; retain failed output')
    current = []
    manifests = sorted(output.rglob('manifest.json'))
    if len(manifests) != 11:
        raise ValueError('Expected eleven original manifests')
    for original in manifests:
        value = json.loads(original.read_text(encoding='utf-8'))
        value['vectors_normalized'] = False
        target = original.with_name('manifest-current.json')
        with target.open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
        current.append({'path': target.relative_to(output).as_posix(), 'sha256': sha(target)})
    return {'status': 'all_original_payload_hashes_match', 'original_files': 44,
            'new_manifests': current,
            'scope': 'Logical synthetic input preparation only; new manifests add explicit vectors_normalized=false. Original manifests, historical reports and archive identities are unchanged.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path, help='New directory; existing directories are never replaced')
    args = parser.parse_args()
    print(json.dumps(prepare(args.output), indent=2))


if __name__ == '__main__':
    main()
