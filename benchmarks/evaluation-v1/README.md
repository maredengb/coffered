# Evaluation materials

This collection contains a complete representative retrieval execution, its public
dataset and derived archives of historical quality and resource observations.
It does not certify a retrieval configuration or establish whole-version release
acceptance. Historical results and incomplete attempts retain their own identities.

## Prepared public dataset

[`public-quality-v1`](public-quality-v1/dataset_manifest.json) supplies 24 source
documents, 99 exact text chunks, 48 authored information needs, 147 model vectors
and 3,960 binary relevance judgments. There are four tasks in each of twelve
material/query-type groups. The materials are Rust code, SQLite technical
documentation and historical Rust language design notes. Source and task counts
are balanced across materials; chunk counts are 18, 17 and 64 respectively.

Tasks were written from the sources rather than collected from user query logs.
The task author and an independent annotator inspected every eligible pair.
Twenty disputed or uncertain pairs were adjudicated and reviewed before any
retrieval. Final candidate labels contain 91 positives, 3,869 negatives and no
unresolved pairs. Another 792 pairs are outside metadata-filter scope and are not
negative labels. This is AI-assisted annotation, not human relevance certification.
See [criteria](public-quality-v1/relevance-criteria.json),
[judgment provenance](public-quality-v1/judgment-provenance.json) and
[per-pair evidence](public-quality-v1/judgment-evidence.jsonl).

The [representative report](representative-quality-report.md) records the frozen
protocol and complete 1,728 CofferDB plus 1,728 SQLite query execution. Hybrid
Recall@10 is 179/192 and MRR@10 is 269/288 on these fixed tasks; keyword retrieval
has the same recall and higher MRR. Every negative comparison is retained. No
numeric quality floor or superiority gate applies. The engine defaults remain
unchanged. The [replay guide](representative-replay.md) covers offline metrics,
logical reruns and measurement limits; the [evaluated source bundle](evaluated-source-v1/artifact-lineage.json)
binds exact file payloads. A changed model, task, label or weight creates a
different evaluation identity.

[Source provenance](public-quality-v1/source-provenance.json) records pinned
upstream sources, captured HTML hashes, licenses and exact UTF-8 projected-text
spans. Its `no_qrels`/`no_vectors` statements describe the earlier source-capture
stage, not the contents of this assembled dataset. Formula images and a diagram
omitted from the HTML projections are explicitly listed; neither tasks nor labels
may rely on unseen images. Upstream notes are historical source material, not
current Rust guidance.

[Embedding provenance](public-quality-v1/embedding-provenance.json) separately
binds all 99 passages and 48 queries to the fixed E5 model revision, text hashes,
prefixes, tokenizer, pooling, normalization and f32 vectors. Tokenization did not
truncate any input. Replay uses the supplied vectors and needs no model download
or inference service. Model preparation time is outside database measurements.
See the [upstream model-license declaration](public-quality-v1/model-license-provenance.json).
Source licenses and notices remain with the raw materials.

## Historical observations

[`historical-quality-v2/artifact-lineage.json`](historical-quality-v2/artifact-lineage.json)
identifies every original artifact hash, public hash and transformation. The
archive preserves the complete available per-query observations from the earlier
48-task study, model/context experiments, weight diagnostics and 24-task fusion
experiment, including failures and negative comparisons. Original verdicts remain
historical verdicts; they are not current release gates.

The earlier 48-task hybrid observation was macro Recall@10 `119/160 = 0.74375`
and MRR@10 `4033/5760` under its original implementation and protocol. Its original
quality verdict is preserved as failed. Model/context and weight experiments used
already-seen tasks, sometimes improving aggregate results and sometimes degrading
individual groups. They do not independently validate a new default. The two
tested fusion candidates did not establish the originally requested strict
comparison. Read all per-query and group results, rather than selecting a winner.

Byte-identical reports are distinguished from derived results. Thirteen result
files replace only four declared filesystem-path fields. Every other decimal
number, ranked ID and recorded configuration is retained. Five complete-score
projections preserve all 240 query rows and 30,110 hit rows, while excluding raw
queries, snippets, source paths, metadata and matched terms. A failed preparation
attempt with no complete-score artifact is retained as a failed attempt; absent
data is not invented. Archived machine records retain original run identities
and are not current configuration examples.

The [numeric supplement](historical-quality-supplement-v1/supplement-lineage.json)
adds all 48 actual explicit-weight query results and all 96 historical fusion
component branches. It preserves IDs and required scores without redistributing
the private query text, snippets or source metadata. Scores generated by an
offline replay are labeled as recomputed, rather than original engine captures.

The [earlier 24-task archive](historical-legacy24-v1/artifact-lineage.json) retains
twelve complete benchmark results across four original source versions, two
earlier fusion reports and the available incomplete-attempt status. Original
failed comparisons and an uncompleted human acceptance remain explicit. A partial
component journal without a terminal fusion report is not assigned a quality
verdict. These earlier positive-only labels have the limitations stated below.

The historical corpus contains internal records and is not redistributed. Its
published ranked IDs and labels permit metric recomputation, but this archive is
not a complete text/vector replay of every historical run. Four earlier text
inputs were regenerated with their original hashes; missing original vector
assets were not reconstructed. The earlier 24-task labels contain positives only,
so an unlisted item is unjudged rather than proven irrelevant. The new dataset's
complete eligible judgments use a different protocol and cannot change old
results.

## Complete historical scoring comparison

The [fixed-source comparison](historical-scoring-comparison-v1/README.md) preserves
all 7,488 sample records, 18 quality groups and 24 scale comparison groups from
two complete historical implementations on the same fixed public inputs. The
[report](historical-scoring-comparison-v1/REPORT.md) includes every negative
comparison, whole-process memory costs, work counters and measurement limits.
The earlier stopped attempt remains separately available with its incomplete
ledger; later success does not replace it.

The [source-only replay tools](historical-scoring-comparison-v1/REPLAY.md) prepare
exact historical source snapshots and the adjacent public inputs, build matching
adapters and retain each new execution under a separate identity. An offline
numeric replay independently recomputes the archived metrics without importing
or querying a database. This comparison is neither a current-release performance
guarantee nor an isolated attribution to one scoring change. It introduces no
quality lower bound or mandatory winner criterion.

## Scale and resource records

The [100k scale archive](historical-scale-v2/artifact-lineage.json) retains the
original protocols, all six workers per platform and complete first, warmup and
measured records for macOS and Linux. Results and costs remain bound to their
historical source, hardware and timing scopes. Current naming or documentation
changes do not turn those observations into current binary measurements.

The [synthetic replay guide](scale-replay.md) provides a compact exact-input
reconstruction recipe and an optional Rust query helper. Its one-hot vectors are
a mechanism workload, not semantic-quality evidence. The later reconstruction
recipe is distinguished from the unrecovered original generator. New executions
must keep a new identity and cannot replace historical reports.

The [current-source smoke record](synthetic-scale-smoke-v2/artifact-lineage.json)
records one from-empty import, a successful integrity check and eight read-only
query checks with an unchanged database hash. Its import manifest includes all
three input hashes and produced no warnings. This confirms the portable workflow
under the recorded source and environment; it is not a new performance matrix.

The [quality-cost ledger](historical-quality-cost-v1/artifact-lineage.json)
preserves all 1,728 counted-once search-cost rows from the historical 48-task
execution, including stage work and logical reads where recorded. Do not sum
overlapping stage diagnostics into end-to-end time. Unavailable physical read
bytes remain null. These small-corpus observations do not fill fields absent from
the historical 100k helper or establish a physical-I/O or energy result.

The [import diagnostic archive](historical-import-v2/artifact-lineage.json)
retains eighteen trials, their complete original event/resource records and both
measurement probes. The [input and probe guide](import-replay.md) explains how to
reconstruct all 132 synthetic input files and optionally compile a current-source
probe. Bulk and ten-part trials have different query obligations and are not a
causal batching speed comparison. Their original timing and unknown-resource
declarations remain unchanged.

The [historical Linux lifecycle archive](historical-linux-lifecycle-v1/artifact-lineage.json)
retains the multicollection operations, disk ledgers and native records. The
[lifecycle replay guide](lifecycle-replay.md) supplies the normalized logical input
archive and original helper. A [current-source logical replay](synthetic-lifecycle-replay-v1/artifact-lineage.json)
completed 60 queries across import, replacement, deletion, compaction, export and
copy/reopen; its initial summary counted only 32 untouched queries, so both that
summary and a separately audited 60-query correction are retained.

The [historical recovery cost archive](historical-recovery-cost-v1/artifact-lineage.json)
retains the later six-query read/rebuild observation with its explicit 1024 MiB
cache setting. The [earlier incomplete attempt](historical-recovery-incomplete-v1/artifact-lineage.json)
remains separate: partial results and unknown cleanup/resource fields are not
backfilled by the later success. These records are not a new 2 GiB content or
crash-recovery validation.

The [lifecycle phase-cost report](lifecycle-phase-costs.md) retains 41 historical
worker observations across the multicollection workflow and four prescribed
large-file interruptions, with all bound raw measurement streams. Its exact
historical large-file bytes are not redistributed; recorded identities and
replay limitations remain explicit.

## Offline numeric replay

These optional tools need Python 3.11 or later and only its standard library.
They do not install dependencies, open a database, generate vectors or execute
queries. From this directory, choose new output filenames:

```sh
python3 recompute-quality.py \
  historical-quality-v2/formal48/qrels.jsonl \
  historical-quality-v2/formal48/round-1-balanced/result-derived.json \
  --output quality-recheck.json

python3 recompute-fusion.py \
  historical-quality-supplement-v1/legacy-fusion-branch-scores-derived.jsonl \
  historical-quality-v2/legacy24/fusion-report.json \
  --output fusion-recheck.json
```

The first tool checks query coverage and eligible IDs and recomputes exact-rational
macro relevance metrics for supplied ranked IDs. The second verifies that the
archived branch scores reproduce every original fusion ranking and writes the
recomputed scores. Neither tool proves that an experiment fully executed or that
the engine computed its input scores correctly. Original reports remain unchanged.

## Measurement limits

Report CofferDB and SQLite separately: tokenization, candidate/filter strategy,
raw-score scales and timing boundaries differ. SQLite CLI startup is included in
its query timing; the embedded engine's process startup is excluded. The built-in
`cold` label means a reopened database handle, not an evicted OS page cache.
Repeated queries add timing samples, not independent relevance coverage.

Use the exact protocol, source/binary identity, all input hashes, every planned
attempt and unrounded query durations. P50/P95 use nearest rank within the stated
engine/preset/mode/cache group; a pooled percentile is recomputed from samples,
not averaged from round percentiles. Failed or unexecuted samples remain explicit.
Process RSS is not a cache budget or per-query allocation limit. Application read
and work counters are not physical disk I/O, CPU attribution or an energy estimate.
Small-corpus quality timings do not establish scale performance.
