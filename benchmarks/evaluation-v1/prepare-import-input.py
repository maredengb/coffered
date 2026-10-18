"""Recreate historical no-vector import inputs and verify all 132 original hashes.

Only Python's standard library is needed. This writes synthetic files, not a
database, and does not execute the historical measurement probes.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_dataset(directory, documents, chunks):
    directory.mkdir()
    for name, rows in [('documents', documents), ('chunks', chunks)]:
        with (directory / (name + '.jsonl')).open('x', encoding='utf-8', newline='\n') as stream:
            for row in rows:
                stream.write(json.dumps(row, separators=(',', ':')) + '\n')
    manifest = {'schema_version': '0.1', 'collection': 'measurements', 'embedding_dim': 768,
                'distance_metric': 'cosine', 'documents': 'documents.jsonl', 'chunks': 'chunks.jsonl',
                'hashes': {'documents_sha256': sha(directory / 'documents.jsonl'),
                           'chunks_sha256': sha(directory / 'chunks.jsonl')}}
    (directory / 'import.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n',
                                            encoding='utf-8', newline='\n')


def prepare(output):
    protocol = Path(__file__).resolve().parent / 'historical-import-v2/protocol-derived.json'
    expected = json.loads(protocol.read_text(encoding='utf-8'))['input_manifest']
    output.mkdir()
    for count in [10000, 100000]:
        directory = output / str(count)
        directory.mkdir()
        all_documents, all_chunks = [], []
        for part in range(10):
            documents, chunks = [], []
            for document in range(part * count // 1000, (part + 1) * count // 1000):
                documents.append({'external_id': f'doc-{document}', 'metadata': {'project': document % 20}})
                for ordinal in range(100):
                    chunks.append({'document_external_id': f'doc-{document}',
                                   'chunk_external_id': f'chunk-{document * 100 + ordinal}',
                                   'ordinal': ordinal,
                                   'text': 'needle repeated' if document == 0 else 'ordinary repeated',
                                   'metadata': {}})
            all_documents.extend(documents)
            all_chunks.extend(chunks)
            write_dataset(directory / f'small-{part}', documents, chunks)
        write_dataset(directory / 'bulk', all_documents, all_chunks)
        shutil.copytree(directory, output / (str(count) + '-r2'))
    actual = {path.relative_to(output).as_posix(): sha(path) for path in sorted(output.rglob('*.json*'))}
    if actual != expected or len(actual) != 132:
        raise ValueError('Input hash mismatch; retain failed output for diagnosis')
    return {'status': 'all_original_input_hashes_match', 'files': len(actual), 'sha256': actual,
            'scope': 'Synthetic no-vector text input reconstruction only; no old binary, database history, timing or query execution recreated.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path, help='New directory; existing outputs are never replaced')
    args = parser.parse_args()
    print(json.dumps(prepare(args.output), indent=2))


if __name__ == '__main__':
    main()
