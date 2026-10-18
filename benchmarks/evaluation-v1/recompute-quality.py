"""Recompute relevance metrics from supplied IDs/labels; no database or model calls."""

import argparse
from collections import defaultdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path


def file_hash(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def labels(path):
    positive, eligible, seen = defaultdict(set), defaultdict(set), set()
    for line in path.read_text().splitlines():
        row = json.loads(line)
        query, chunk, relevance = row['query_id'], row['chunk_external_id'], row['relevance']
        if not isinstance(query, str) or not query or not isinstance(chunk, str) or not chunk:
            raise ValueError('Labels require nonempty query/chunk IDs')
        if type(relevance) is not int or relevance not in (0, 1) or (query, chunk) in seen:
            raise ValueError('Duplicate label or non-binary relevance')
        seen.add((query, chunk))
        eligible[query].add(chunk)
        if relevance:
            positive[query].add(chunk)
    if not seen or any(not positive[query] for query in eligible):
        raise ValueError('Every labeled query must have at least one positive')
    return positive, eligible


def metrics(rows, positive, eligible, positive_only=False):
    by_query = {}
    for row in rows:
        query, ranked = row['query_id'], row['ranked_chunk_ids']
        if query in by_query or query not in eligible:
            raise ValueError('Duplicate or unlabeled query')
        if not isinstance(ranked, list) or not all(isinstance(item, str) for item in ranked):
            raise ValueError('Ranked IDs must be a list of strings')
        if len(ranked) != len(set(ranked)):
            raise ValueError('Duplicate ranked chunk')
        if not positive_only and not set(ranked) <= eligible[query]:
            raise ValueError('Returned ID is unjudged or filter-ineligible')
        relevant = positive[query]
        first = next((rank for rank, chunk in enumerate(ranked[:10], 1) if chunk in relevant), None)
        by_query[query] = {
            'recall_at_5': Fraction(sum(chunk in relevant for chunk in ranked[:5]), len(relevant)),
            'recall_at_10': Fraction(sum(chunk in relevant for chunk in ranked[:10]), len(relevant)),
            'mrr_at_10': Fraction(1, first) if first else Fraction(0),
        }
    if set(by_query) != set(eligible):
        raise ValueError('Supplied run does not cover every labeled query exactly once')
    keys = ['recall_at_5', 'recall_at_10', 'mrr_at_10']
    overall = {key: sum((row[key] for row in by_query.values()), Fraction()) / len(by_query) for key in keys}
    return {'queries': len(by_query), 'overall': overall, 'per_query': by_query}


def serial(value):
    if isinstance(value, Fraction):
        return {'numerator': str(value.numerator), 'denominator': str(value.denominator), 'value': float(value)}
    if isinstance(value, dict):
        return {key: serial(item) for key, item in value.items()}
    if isinstance(value, list):
        return [serial(item) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('qrels', type=Path)
    parser.add_argument('result', type=Path)
    parser.add_argument('--positive-only', action='store_true', help='Permit unjudged returned IDs; use only for explicitly positive-only historical labels')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    positive, eligible = labels(args.qrels)
    result = json.loads(args.result.read_text())
    observations = []
    for field in ['runs', 'baseline_results']:
        for run in result.get(field, []):
            observations.append({'source_field': field, 'engine': run.get('engine'),
                                 'mode': run['query_mode'], 'cache': run['cache_state'],
                                 **metrics(run['per_query'], positive, eligible, args.positive_only)})
    if not observations:
        raise ValueError('No benchmark run rows found')
    report = {'scope': 'Metrics for supplied ranks only; not proof of execution completeness, engine correctness or scheme certification.',
              'label_policy': 'Known positives only; unjudged is not proven irrelevant' if args.positive_only else 'Every returned ID must have an eligible binary judgment',
              'averaging': 'Macro over labeled queries; Recall at5/10; first-positive reciprocal rank truncated at10.',
              'qrels_sha256': file_hash(args.qrels), 'result_sha256': file_hash(args.result), 'observations': observations}
    with args.output.open('x') as stream:
        json.dump(serial(report), stream, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
