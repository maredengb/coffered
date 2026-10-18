"""Reconstruct three historical synthetic scale assets with their exact byte hashes.

This is a later reconstruction recipe inferred and checked against all rows,
not the original generator or a recovered record of its seed/history.
No trained model, downloads, database operations or random generator are used.
"""

import argparse
import hashlib
import json
from pathlib import Path

EXPECTED = {
    'documents.jsonl': 'bd71819fbe4789e2645136fa40a10ec11981b99a26b38eb499038020e700ef36',
    'chunks.jsonl': '6b09b63481b568572ecf543498bbdc5f1b946a41ef10e0fb8f09c6e19def61d1',
    'embeddings.jsonl': 'bcab207bb7d976e95f99d95970d2a180c1fdf8b0a6489d348d7ac1fc38d20329',
}


def row_bytes(row):
    return (json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n').encode('utf-8')


def prepare(output):
    output.mkdir()
    counts = {'documents.jsonl': 0, 'chunks.jsonl': 0, 'embeddings.jsonl': 0}
    digests = {name: hashlib.sha256() for name in EXPECTED}
    with (output/'documents.jsonl').open('xb') as docs, (output/'chunks.jsonl').open('xb') as chunks, (output/'embeddings.jsonl').open('xb') as vectors:
        streams = {'documents.jsonl': docs, 'chunks.jsonl': chunks, 'embeddings.jsonl': vectors}
        def write(name, row):
            wire = row_bytes(row)
            streams[name].write(wire)
            digests[name].update(wire)
            counts[name] += 1
        for document in [*range(89, 1000), *range(89)]:
            revision = 5 if document < 89 else 4
            metadata = {'project': document % 20, 'revision': revision}
            write('documents.jsonl', {'external_id': f'doc-{document}', 'metadata': metadata,
                                      'source_uri': f'project/{document}/revision/{revision}',
                                      'title': f'Storage project {document}'})
            vector = [0] * 768
            vector[(document + revision) % 768] = 1
            for ordinal in range(100):
                chunk = document * 100 + ordinal
                write('chunks.jsonl', {'chunk_external_id': f'chunk-{chunk}', 'document_external_id': f'doc-{document}',
                                      'metadata': metadata, 'ordinal': ordinal, 'source_span': None,
                                      'text': f'Project {document} revision {revision} task {ordinal}: verify durability recovery checksum and searchable index generation after document replacement.'})
                write('embeddings.jsonl', {'chunk_external_id': f'chunk-{chunk}', 'dim': 768,
                                          'model_name': 'synthetic-lifecycle-v1', 'vector': vector})
    actual = {name: digest.hexdigest() for name, digest in digests.items()}
    if actual != EXPECTED:
        raise ValueError('Reconstructed hashes differ; retain this failed output for diagnosis')
    manifest = {'schema_version': '0.1', 'dataset_id': 'synthetic-scale-reconstruction-v1', 'collection': 'lifecycle',
                'embedding_dim': 768, 'embedding_model': 'synthetic-lifecycle-v1', 'distance_metric': 'cosine',
                'vectors_normalized': False, 'documents': 'documents.jsonl', 'chunks': 'chunks.jsonl', 'embeddings': 'embeddings.jsonl',
                'hashes': {name.removesuffix('.jsonl') + '_sha256': digest for name, digest in actual.items()}}
    (output/'import_manifest.json').write_bytes(row_bytes(manifest))
    provenance = {'status': 'exact_historical_jsonl_bytes_reconstructed', 'recipe_version': 1,
                  'method': 'Later closed-form reconstruction checked against every original row; original generator and seed provenance are not established.',
                  'counts': counts, 'sha256': actual, 'model': 'Synthetic one-hot 768D; no model inference or semantic relevance claim.',
                  'order': 'Document89..999 then0..88; each document100chunks; revision5 for0..88 and4 otherwise; vector basis(document+revision)%768.',
                  'encoding': 'UTF-8, sorted JSON keys, compact separators, LF per row; vector entries are integers0 or1.',
                  'import_manifest': 'New convenience manifest for current import; not an original export manifest or a reconstructed database history.',
                  'limitations': ['Reproduces three exported input assets, not the old2GiB file, transaction history, internal IDs, binary file bytes or original timings.',
                                  'One-hot vectors and repetitive text are a scale/mechanism workload, not semantic-quality evidence.']}
    (output/'reconstruction.json').write_bytes(row_bytes(provenance))
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path, help='New output directory; approximately191MB uncompressed')
    args = parser.parse_args()
    print(json.dumps(prepare(args.output), indent=2))


if __name__ == '__main__':
    main()
