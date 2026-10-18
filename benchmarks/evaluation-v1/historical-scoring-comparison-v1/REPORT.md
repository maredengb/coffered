# Historical scoring comparison: observed results

This report retains all observations from one completed, fixed-input comparison of two complete historical implementations. It does not certify semantic quality, isolate a scoring change, establish current-release performance, or approve a release. The original incomplete experiment remains incomplete; a later successful comparison does not replace or relabel it.

## Workload and completeness

The quality workload contains 48 independent tasks, 99 chunks and 3,960 relevance judgments. Query and chunk vectors are fixed external inputs. The scale workload contains 1,000 documents, 100,000 chunks and 100,000 vectors. Defaults were retained. All 22 imports, 36 trials and 7,488 query observations completed. Repeated observations do not increase the number of independent tasks.

`observations/` contains every import/query process stdout and stderr, including full returned IDs, ranks, raw/normalized/final scores, execution counters and read diagnostics. Trial streams retain their original bytes. Private database path prefixes in import output are replaced with placeholders; `lineage.json` records original and projected digests. `report-derived.json.gz` applies path replacement to the report and full sample ledger and omits the administrative execution-approval digest. No query, result, score, duration or outcome is removed.

`comparisons.json` contains all quality and scale groups. Both cache states and both main rounds are retained. Main latency excludes diagnostic runs; first queries and warmups remain available separately. RSS is per helper process, not a per-query or scoring-only peak.

## Negative results and limits

One historical configuration's hybrid Recall@10 changed from 0.9427083333 to 0.9322916667 and MRR@10 from 0.9366319444 to 0.9340277778, in both handle states. Other main aggregate quality groups matched; this does not imply identical rankings or scores. The data does not support a claim of universal improvement. There is no numerical quality pass threshold or mandatory winning condition.

Three of the 24 scale main P95 groups were slower in the newer historical version: vector search with standard JSON serialization in the middle-budget workload (+0.987541 ms), and text search in the smallest-window workload (+0.328834 ms for execute, +0.961709 ms including standard JSON). These differences are observations, not statistical significance claims. Maximum whole-helper RSS in the two larger-window workloads increased from 276,561,920 to 422,133,760 bytes and from 276,955,136 to 425,771,008 bytes. The smallest-window workload changed from 278,249,472 to 277,086,208 bytes. These process peaks cover all opens, queries, statistics, serialization and evidence work; they do not isolate scoring memory. All groups and each process peak remain in the JSON results.

The host was shared: initial system load was about 8, with substantial compressed memory. A fresh database handle is not a cold OS cache. Earlier trials, collection statistics and file hashing can warm that cache. Read diagnostics measure userspace work on the calling query thread, not physical disk traffic. The two complete historical versions differ beyond scoring; measured latency differences cannot be attributed solely to score completion or generalized to other workloads or the current release.

## Earlier stopped attempt

`prior-attempt/` retains the preceding failed execution: five completed imports,
one stopped import and no query worker started. All 7,488 planned samples remained
not run, with unavailable query times. The resource/capture monitor encountered a
temporary file that disappeared during directory scanning and stopped the owned
process. This does not prove an engine import defect or resource exhaustion. Its
partial database is not reused or presented as recovered; the later successful
run used new databases. Available import streams, outer coordinator output and
the incomplete ledger remain separate, with original and projected digests.

## Reproduction

See [REPLAY.md](REPLAY.md) for fixed-input preparation, source rebuilding, pure
controls and a new complete query execution. The source snapshots, helper code
and input checksum specifications are included here. Input data, licenses and
model/source provenance are supplied in the adjacent evaluation datasets.

See [README.md](README.md) to recompute archived observations without running a
database. A new execution has a new identity; it does not amend these observations.

The full derived report is stored as deterministic gzip. Its compressed and
expanded size/hash are recorded in `lineage.json`; decompression preserves every
retained report byte. Python and gzip can read it without an additional service.
