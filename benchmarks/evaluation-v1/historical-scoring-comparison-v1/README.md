# Historical scoring comparison

This dataset describes two complete historical implementations on identical fixed
inputs. It retains all planned observations, negative results, source identities
and replay tools. It is not a semantic-quality certification, an isolated scoring
causality study or a performance guarantee for the current release.

- [Results and limitations](REPORT.md): quality, latency, process RSS, work counters
  and the full negative results.
- [Protocol](protocol-derived.json), [trial index](trial-index.json),
  [all comparison groups](comparisons.json), and [full derived report](report-derived.json.gz).
- `observations/`: all 116 captured streams from the successful execution.
- [Earlier stopped attempt](prior-attempt/summary.json): all available process
  streams and its incomplete sample ledger remain separate; it is not relabelled.
- [Replay guide](REPLAY.md), exact historical sources in `sources/`,
  [source identities](source-files.json) and [tool derivation](derivation.json).
- [Artifact lineage](lineage.json): file sizes/checksums and original identities
  for path-normalized observations. Administrative execution identities are
  omitted from the derived reports; all query records remain intact.

## Recompute the archived metrics without running a database

Use Python 3.11 or newer and choose a new output file. From this directory:

```sh
python3 -B recompute.py --observations-dir . \
  --qrels ../public-quality-v1/qrels.jsonl --output metric-recheck.json
```

This verifies every listed file digest, all 7,488 sample event triplets, fixed
query coverage and the 18 quality/24 scale comparison cells. It independently
recomputes macro Recall/MRR using rational arithmetic, pooled nearest-rank
latency P50/P95 and recorded whole-process RSS. It does not prove engine score
correctness, physical disk IO, semantic answer quality or resource-budget safety.
First queries, warmups and read diagnostics remain separate from primary scale
percentiles; repetitions do not create more than 48 independent quality tasks.

A complete new engine run uses the source/input/build path in [REPLAY.md](REPLAY.md).
The historical source snapshots retain their original identifiers and package
metadata. They are distinct from the current product source and CLI contract.
No quality lower-bound or mandatory winner criterion is applied.

The full derived report is stored as deterministic gzip. Its compressed and
expanded size/hash are recorded in `lineage.json`; decompression preserves every
retained report byte. Python and gzip can read it without an additional service.
