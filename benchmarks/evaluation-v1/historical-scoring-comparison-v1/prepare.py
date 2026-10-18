"""Prepare fixed public inputs and source snapshots without running the database."""
from pathlib import Path, PurePosixPath
import argparse, hashlib, json, os, shutil, stat, tarfile
PACKAGE = Path(__file__).resolve().parent

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def require(value, reason):
    if not value:
        raise ValueError(reason)

def safe_relative(value):
    path = PurePosixPath(value)
    require(not path.is_absolute() and '..' not in path.parts and str(path) == value, 'unsafe relative path')
    return Path(value)

def copy_verified(source, target, expected):
    require(stat.S_ISREG(source.lstat().st_mode), 'copy source must be a regular file')
    data = source.read_bytes()
    require(len(data) == expected['bytes'] and hashlib.sha256(data).hexdigest() == expected['sha256'], 'copy source changed')
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('xb') as stream:
        stream.write(data)

def read_member(tar, member, expected):
    require(member.isfile() and member.size == expected['bytes'], 'scale member type/size')
    data = tar.extractfile(member).read(expected['bytes'] + 1)
    require(len(data) == expected['bytes'] and hashlib.sha256(data).hexdigest() == expected['public_member_sha256'], 'scale member digest')
    return data

def prepare(evaluation, output):
    evaluation = evaluation.resolve(strict=True)
    output = output.parent.resolve(strict=True) / output.name
    require(not any(parent.samefile(PACKAGE) or parent.samefile(evaluation) for parent in [output.parent, *output.parent.parents]), 'work directory overlaps package or evaluation inputs')
    require(not output.exists() and not output.is_symlink(), 'work directory already exists')
    require(os.statvfs(output.parent).f_bavail * os.statvfs(output.parent).f_frsize >= 28 * 1024**3,
            'at least 28 GiB free space is required')
    spec = json.loads((PACKAGE / 'input-spec.json').read_text())
    manifest_path = PACKAGE / 'source-files.json'
    require(digest(manifest_path) == spec['source_manifest_sha256'], 'source manifest changed')
    sources = json.loads(manifest_path.read_text())['files']
    expected_sources = {safe_relative(row['path']) for row in sources}
    source_root = PACKAGE / 'sources'
    require(stat.S_ISDIR(source_root.lstat().st_mode), 'sources must be a real directory')
    source_entries = list(source_root.rglob('*'))
    require(all(stat.S_ISDIR(p.lstat().st_mode) or stat.S_ISREG(p.lstat().st_mode) for p in source_entries), 'source links/special files are refused')
    actual_sources = {p.relative_to(PACKAGE) for p in source_entries if stat.S_ISREG(p.lstat().st_mode)}
    require(actual_sources == expected_sources | {Path('sources/README.md')}, 'source file set changed')
    for row in sources:
        path = PACKAGE / safe_relative(row['path'])
        require(not path.is_symlink() and path.stat().st_size == row['bytes'] and digest(path) == row['sha256'], 'source changed: ' + row['path'])
    readme = spec['source_readme']
    require(digest(PACKAGE / readme['path']) == readme['sha256'] and (PACKAGE / readme['path']).stat().st_size == readme['bytes'], 'source readme changed')
    asset_root = PACKAGE / 'assets'
    require(stat.S_ISDIR(asset_root.lstat().st_mode), 'assets must be a real directory')
    asset_entries = list(asset_root.iterdir())
    require(all(stat.S_ISREG(p.lstat().st_mode) for p in asset_entries), 'asset links/directories/special files are refused')
    require({p.name for p in asset_entries} == {r['name'] for r in spec['assets']}, 'asset file set changed')
    for row in spec['assets']:
        asset = asset_root / safe_relative(row['name'])
        require(asset.stat().st_size == row['bytes'] and digest(asset) == row['sha256'], 'asset changed: ' + row['name'])
    require({r['name'] for r in spec['quality_files']} == {'documents.jsonl','chunks.jsonl','embeddings.jsonl','queries.jsonl','query_embeddings.jsonl','qrels.jsonl'}, 'quality file set changed')
    for row in spec['quality_files']:
        path = evaluation / 'public-quality-v1' / safe_relative(row['name'])
        require(path.is_file() and path.stat().st_size == row['bytes'] and digest(path) == row['sha256'], 'quality input changed: ' + row['name'])
    archive = evaluation / safe_relative(spec['scale_archive']['relative_path'])
    require(digest(archive) == spec['scale_archive']['sha256'], 'scale archive changed')
    expected = {row['path']: row for row in spec['scale_archive']['members']}
    require(len(expected) == 44, 'scale archive manifest cardinality')
    # Validate the complete archive before creating the destination; never extract metadata or links.
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        require(len(members) == 44 and {m.name for m in members} == set(expected), 'scale archive file set')
        for member in members:
            safe_relative(member.name)
            row = expected[member.name]
            read_member(tar, member, row)
    output.mkdir(mode=0o700)
    for row in [*sources, readme]:
        copy_verified(PACKAGE / safe_relative(row['path']), output / row['path'], row)
    for name in ['replay.py', 'replay_test.py', 'build.py', 'source-files.json', 'input-spec.json', 'derivation.json']:
        shutil.copyfile(PACKAGE / name, output / name)
    for row in spec['assets']:
        if Path(row['name']).suffix in ['.rs', '.c']:
            copy_verified(PACKAGE / 'assets' / row['name'], output / row['name'], row)
    quality = output / 'quality-input'; quality.mkdir()
    for row in spec['quality_files']:
        copy_verified(evaluation / 'public-quality-v1' / row['name'], quality / row['name'], row)
    copy_verified(PACKAGE / 'assets/quality-manifest.json', quality / 'manifest.json', next(r for r in spec['assets'] if r['name']=='quality-manifest.json'))
    with tarfile.open(archive, 'r:gz') as tar:
        for group in range(10):
            batch = output / f'scale-batches/input-{group}'; batch.mkdir(parents=True)
            for name in ['documents.jsonl', 'chunks.jsonl', 'embeddings.jsonl']:
                member = tar.getmember(f'input/input-{group}/{name}')
                data = read_member(tar, member, expected[member.name])
                with (batch / name).open('xb') as target:target.write(data)
            name = f'scale-manifest-{group}.json'
            copy_verified(PACKAGE / 'assets' / name, batch / 'manifest.json', next(r for r in spec['assets'] if r['name']==name))
    # Import wrappers are fixed separately from the original archive's collection wrappers.
    for manifest in [quality / 'manifest.json', *sorted((output / 'scale-batches').glob('*/manifest.json'))]:
        row = json.loads(manifest.read_text())
        for kind in ['documents', 'chunks', 'embeddings']:
            require(digest(manifest.parent / safe_relative(row[kind])) == row['hashes'][kind + '_sha256'], 'import wrapper hash mismatch')
    files = []
    for path in sorted(output.rglob('*')):
        require(not path.is_symlink(), 'unexpected prepared symlink')
        if path.is_file():
            files.append({'path':str(path.relative_to(output)), 'bytes':path.stat().st_size, 'sha256':digest(path)})
    receipt = {'schema':1, 'status':'INPUTS_PREPARED_NO_ENGINE_EXECUTION', 'package_input_spec_sha256':digest(PACKAGE / 'input-spec.json'),
               'source_manifest_sha256':digest(manifest_path), 'files':files,
               'quality_documents':24, 'quality_chunks':99, 'quality_tasks':48, 'quality_judgments':3960,
               'scale_documents':1000, 'scale_chunks':100000, 'scale_vectors':100000,
               'scale_selection':'Only original groups 0..9; no replacement input or rewritten text/vector.'}
    (output / 'preparation.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'work_directory':str(output), 'status':receipt['status'], 'files':len(files)}))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation-dir', type=Path, required=True, help='Public evaluation-v1 directory containing the two input datasets')
    parser.add_argument('--work-dir', type=Path, required=True, help='New work directory; existing destinations are refused')
    args = parser.parse_args()
    prepare(args.evaluation_dir, args.work_dir)
