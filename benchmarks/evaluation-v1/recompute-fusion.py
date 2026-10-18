"""Replay historical numeric fusion observations, without source text or queries."""

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path


def rank(text, vector, strategy):
    if strategy not in ('weighted_sum_control', 'weighted_rrf_10', 'weighted_rrf_60'):
        raise ValueError('Unknown archived fusion strategy')
    candidates = {}
    for branch, hits in [('text', text), ('vector', vector)]:
        seen = set()
        for offset, hit in enumerate(hits):
            identity = hit['chunk_id']
            if not isinstance(identity, str) or not identity or identity in seen:
                raise ValueError('Invalid or duplicate branch chunk ID')
            if not isinstance(hit['external_id'], str) or not hit['external_id']:
                raise ValueError('Invalid branch external ID')
            score = hit['score']
            if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError('Invalid normalized branch score')
            if offset:
                previous = hits[offset-1]
                if not (previous['score'] > score or previous['score'] == score and previous['chunk_id'].encode('utf-8') < identity.encode('utf-8')):
                    raise ValueError('Branch score/tie order differs from the archived rule')
            seen.add(identity)
            value = candidates.setdefault(identity, {'chunk_id': identity, 'external_id': hit['external_id'],
                                                       'text': 0.0, 'vector': 0.0})
            if value['external_id'] != hit['external_id']:
                raise ValueError('Branch identity mismatch')
            constant = 10 if strategy == 'weighted_rrf_10' else 60
            value[branch] = hit['score'] if strategy == 'weighted_sum_control' else 1 / (constant + offset + 1)
    for value in candidates.values():
        value['score'] = 0.5 * value['text'] + 0.5 * value['vector']
    return sorted(candidates.values(), key=lambda value: (-value['score'], value['chunk_id'].encode('utf-8')))[:10]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('components', type=Path)
    parser.add_argument('historical_report', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    grouped = defaultdict(dict)
    for line in args.components.read_text().splitlines():
        row = json.loads(line)
        key = row['historical_profile'], row['query_id']
        if row['branch'] in grouped[key]:
            raise ValueError('Duplicate archived branch')
        grouped[key][row['branch']] = row['hits']
    results = []
    historical = json.loads(args.historical_report.read_text())
    if not historical['results']:
        raise ValueError('No archived observations')
    expected_pairs, seen_observations = set(), set()
    for observation in historical['results']:
        if len(observation['per_query']) != observation['independent_queries']:
            raise ValueError('Incomplete archived query group')
        for query in observation['per_query']:
            key = observation['profile'], observation['strategy'], query['query_id']
            if key in seen_observations:
                raise ValueError('Duplicate archived observation')
            seen_observations.add(key)
            expected_pairs.add((observation['profile'], query['query_id']))
            branches = grouped[(observation['profile'], query['query_id'])]
            if set(branches) != {'text', 'vector'}:
                raise ValueError('Incomplete archived branch pair')
            ranked = rank(branches['text'], branches['vector'], observation['strategy'])
            if [hit['external_id'] for hit in ranked] != query['ranked_chunk_ids']:
                raise ValueError('Replayed IDs differ from the original archived report')
            results.append({'historical_profile': observation['profile'], 'strategy': observation['strategy'],
                            'query_id': query['query_id'], 'recomputed_ranked_scores': ranked,
                            'original_ranks_reproduced': True})
    if set(grouped) != expected_pairs:
        raise ValueError('Unmatched or missing archived component query')
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    output = {'scope': 'Numeric replay of historical branch scores only; these scores are recomputed, not newly captured engine results or an independent quality evaluation.',
              'components_sha256': digest(args.components), 'historical_report_sha256': digest(args.historical_report),
              'formula': 'Equal0.5 branch weights; control uses original normalized branch score, RRF uses reciprocal(k+one-based branch rank), k10 or60. Missing branch contribution0; score descending then internal chunk ID UTF-8 ascending; top10.',
              'observations': results}
    with args.output.open('x') as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
