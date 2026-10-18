# Representative retrieval observations

This source-derived study supplies shareable observations for one fixed dataset,
model and implementation. The complete second execution contains 1,728 CofferDB
queries and 1,728 SQLite reference queries over 48 independent tasks. Repetitions
are timing samples, not additional independent information needs. There is no
quality threshold, superiority gate or scheme certification in this protocol.
These observations are not a whole-version release verdict.

## Fixed method and identity

The [dataset](public-quality-v1/dataset_manifest.json) contains 24 source documents,
99 chunks, 48 authored tasks and 3,960 complete eligible relevance judgments.
Another 792 pairs are excluded by metadata scope, not judged irrelevant. The task
author and an independent AI annotator inspected every eligible pair; twenty
initial disagreements were adjudicated before retrieval. This is AI-assisted
judgment on authored tasks, not human query-log or population validation.

The supplied 147 vectors use `intfloat/multilingual-e5-small` revision
`ffdcc22a9a5c973ef0470385cef91e1ecb461d9f`, dimension 384. Passage/query text,
normalization, tokenization, model license and exact vectors are recorded in the
[embedding provenance](public-quality-v1/embedding-provenance.json). No model
inference is included in database timing. External vector semantics, task
coverage and judgments limit the interpretation of retrieval quality.

The original frozen protocol SHA-256 is
`991a388fd3cb281c704197c20d90a8daad7a136dc2494940ca900f1f4758ea3d`;
the original execution auditor is
`08705b7a480e6da726ce714053e4615abdbb792df5f6fa5fb5685fd8b75978a1`.
The [method freeze receipt](representative-quality-v1/method-freeze-receipt-derived.json)
binds approval before retrieval. The derived protocol preserves the original
preparation-stage status field; that field is not a later execution status.
[Artifact lineage](representative-quality-v1/artifact-lineage.json) binds every
original and public artifact, declared path substitutions and omitted private
traceback hash. All numerical results and ranked IDs remain unchanged.

The exact evaluated 168-file source fingerprint is
`9244290fcb9d5205a004a19d4aabef1ddf0c9836c67514fad2f1ecdd8a3e53f6`.
The [source bundle](evaluated-source-v1/artifact-lineage.json) preserves every file
payload and normalizes archive ownership and timestamps; its archive hash differs
from the original private archive. It contains a test/document overlay over base
commit `ce1393ef14d6fc03c05f036ed7e7fac6dad3348a`; that commit alone does not identify
the build. The measured CLI SHA-256 is
`061f25acf65fc8fc181dce8fff0f9843f16728e7b6b37ffa1f0adf089eb08905`.
This is a reproducible source snapshot, not an official release source package.

Each of two rounds runs three presets in separate fresh databases, with reversed
preset order in round two. Every process executes keyword, exact-vector and
hybrid modes, each with a reopened-handle group (`cold`) followed by a repeated
handle group (`warm`). There is no OS cache eviction or discarded warmup. All
tasks use top_k=10, default candidate windows and cache budgets, commit durability
and inline full checkpointing. Default hybrid weights remain 0.5/0.5/0;
keyword and vector single branches use 1/0/0 and 0/1/0. No overrides, metadata
boost, model change, label change or ranking tuning followed the observations.

## CofferDB relevance results

Recall is the fraction of judged-positive eligible chunks found in the first k
unique ranks, macro-averaged over tasks. MRR is the reciprocal of the first
positive rank through rank ten, or zero. Fractions below are exact; decimals are
rounded for reading. All presets, rounds and cache groups have identical Top10
IDs **and scores** for each mode on this small dataset. That does not establish
preset equivalence for larger datasets or costs.

| Preset | Mode | Recall@5 | Recall@10 | MRR@10 |
| --- | --- | --- | --- | --- |
| expanded | text_bm25 | 8/9 (0.888889) | 179/192 (0.932292) | 17/18 (0.944444) |
| expanded | vector_exact | 29/36 (0.805556) | 505/576 (0.876736) | 329/384 (0.856771) |
| expanded | hybrid | 259/288 (0.899306) | 179/192 (0.932292) | 269/288 (0.934028) |
| balanced | text_bm25 | 8/9 (0.888889) | 179/192 (0.932292) | 17/18 (0.944444) |
| balanced | vector_exact | 29/36 (0.805556) | 505/576 (0.876736) | 329/384 (0.856771) |
| balanced | hybrid | 259/288 (0.899306) | 179/192 (0.932292) | 269/288 (0.934028) |
| constrained | text_bm25 | 8/9 (0.888889) | 179/192 (0.932292) | 17/18 (0.944444) |
| constrained | vector_exact | 29/36 (0.805556) | 505/576 (0.876736) | 329/384 (0.856771) |
| constrained | hybrid | 259/288 (0.899306) | 179/192 (0.932292) | 269/288 (0.934028) |

Hybrid has the same overall Recall@10 as keyword retrieval, but MRR@10 is lower
by 1/96 (0.010417). Against exact-vector retrieval, hybrid Recall@10 is higher by
1/18 and MRR@10 by 89/1152. These are observations for these fixed inputs, not a
causal allocation of quality to the engine, embedding model or weights.

Each cell below contains four independent tasks. The values apply to all three
presets and both rounds/cache groups; all original per-group records are retained.

| Material | Task type | Mode | Recall@5 | Recall@10 | MRR@10 |
| --- | --- | --- | --- | --- | --- |
| english_technical | exact_term | text_bm25 | 1 (1.000000) | 1 (1.000000) | 1 (1.000000) |
| english_technical | metadata_filter | text_bm25 | 1 (1.000000) | 1 (1.000000) | 1 (1.000000) |
| english_technical | mixed | text_bm25 | 1 (1.000000) | 1 (1.000000) | 1 (1.000000) |
| english_technical | semantic | text_bm25 | 7/8 (0.875000) | 1 (1.000000) | 1 (1.000000) |
| markdown_notes | exact_term | text_bm25 | 7/8 (0.875000) | 1 (1.000000) | 1 (1.000000) |
| markdown_notes | metadata_filter | text_bm25 | 13/16 (0.812500) | 15/16 (0.937500) | 1 (1.000000) |
| markdown_notes | mixed | text_bm25 | 37/48 (0.770833) | 5/6 (0.833333) | 1 (1.000000) |
| markdown_notes | semantic | text_bm25 | 1/3 (0.333333) | 5/12 (0.416667) | 5/8 (0.625000) |
| rust_code | exact_term | text_bm25 | 1 (1.000000) | 1 (1.000000) | 1 (1.000000) |
| rust_code | metadata_filter | text_bm25 | 1 (1.000000) | 1 (1.000000) | 1 (1.000000) |
| rust_code | mixed | text_bm25 | 1 (1.000000) | 1 (1.000000) | 7/8 (0.875000) |
| rust_code | semantic | text_bm25 | 1 (1.000000) | 1 (1.000000) | 5/6 (0.833333) |
| english_technical | exact_term | vector_exact | 7/8 (0.875000) | 1 (1.000000) | 1 (1.000000) |
| english_technical | metadata_filter | vector_exact | 1 (1.000000) | 1 (1.000000) | 7/8 (0.875000) |
| english_technical | mixed | vector_exact | 3/4 (0.750000) | 7/8 (0.875000) | 25/32 (0.781250) |
| english_technical | semantic | vector_exact | 7/8 (0.875000) | 7/8 (0.875000) | 1 (1.000000) |
| markdown_notes | exact_term | vector_exact | 11/12 (0.916667) | 1 (1.000000) | 1 (1.000000) |
| markdown_notes | metadata_filter | vector_exact | 15/16 (0.937500) | 15/16 (0.937500) | 1 (1.000000) |
| markdown_notes | mixed | vector_exact | 37/48 (0.770833) | 11/12 (0.916667) | 5/6 (0.833333) |
| markdown_notes | semantic | vector_exact | 7/24 (0.291667) | 5/12 (0.416667) | 1/2 (0.500000) |
| rust_code | exact_term | vector_exact | 1 (1.000000) | 1 (1.000000) | 1 (1.000000) |
| rust_code | metadata_filter | vector_exact | 1 (1.000000) | 1 (1.000000) | 1 (1.000000) |
| rust_code | mixed | vector_exact | 3/4 (0.750000) | 1 (1.000000) | 19/24 (0.791667) |
| rust_code | semantic | vector_exact | 1/2 (0.500000) | 1/2 (0.500000) | 1/2 (0.500000) |
| english_technical | exact_term | hybrid | 1 (1.000000) | 1 (1.000000) | 1 (1.000000) |
| english_technical | metadata_filter | hybrid | 1 (1.000000) | 1 (1.000000) | 7/8 (0.875000) |
| english_technical | mixed | hybrid | 1 (1.000000) | 1 (1.000000) | 1 (1.000000) |
| english_technical | semantic | hybrid | 7/8 (0.875000) | 1 (1.000000) | 1 (1.000000) |
| markdown_notes | exact_term | hybrid | 1 (1.000000) | 1 (1.000000) | 1 (1.000000) |
| markdown_notes | metadata_filter | hybrid | 13/16 (0.812500) | 15/16 (0.937500) | 1 (1.000000) |
| markdown_notes | mixed | hybrid | 37/48 (0.770833) | 5/6 (0.833333) | 1 (1.000000) |
| markdown_notes | semantic | hybrid | 1/3 (0.333333) | 5/12 (0.416667) | 5/8 (0.625000) |
| rust_code | exact_term | hybrid | 1 (1.000000) | 1 (1.000000) | 1 (1.000000) |
| rust_code | metadata_filter | hybrid | 1 (1.000000) | 1 (1.000000) | 1 (1.000000) |
| rust_code | mixed | hybrid | 1 (1.000000) | 1 (1.000000) | 7/8 (0.875000) |
| rust_code | semantic | hybrid | 1 (1.000000) | 1 (1.000000) | 5/6 (0.833333) |

## CofferDB negative comparisons

All query-level values and signed group/overall comparisons are in the
[complete report](representative-quality-v1/v2/quality-report.json). The following
table lists every CofferDB query with a negative hybrid change in at least one of the
three relevance metrics against either single branch. Identical preset/round/cache
repetitions are represented once. Zero and positive changes remain in the raw
report, rather than being selected out of the evaluation.

| Query | Compared with | Δ Recall@5 | Δ Recall@10 | Δ MRR@10 |
| --- | --- | --- | --- | --- |
| T02-F | text_bm25 | 0 | 0 | -1/2 |
| C02-M | vector_exact | 0 | 0 | -1/2 |
| N01-S | vector_exact | -1/6 | 0 | 0 |
| N04-F | vector_exact | -1/4 | -1/4 | 0 |
| N04-S | vector_exact | -1/3 | -2/3 | -1/2 |
| N05-F | vector_exact | -1/4 | 1/4 | 0 |
| N06-M | vector_exact | 0 | -1/3 | 0 |

For `T02-F`, keyword's first relevant rank is 1 and hybrid's is 2. For `N04-S`,
exact-vector retrieval finds four of six positive chunks through rank ten, while
hybrid finds none (MRR 1/2 versus 0). A higher aggregate hybrid score against the
vector branch must not hide this query's regression.

The older 48-task study remains Recall@10 `119/160 = 0.74375` and MRR@10
`4033/5760`, with its original failed verdict and source identity. This study has
different sources, chunk counts, tasks, vectors and judgments; it cannot establish
an engine quality improvement over the older result or change that verdict.

## SQLite reference observations

SQLite 3.54.0 and sqlite-vec 0.1.6 provide a separately implemented reference.
FTS5 candidates are limited globally before Rust metadata filtering; vector
eligibility is determined before exact cosine distance scoring. Fusion scores
its candidate windows without CofferDB's complete-union supplemental scoring.
Tokenization and raw score scales differ. Profile-dependent SQLite results must
be read with that strategy; they are not evidence of a CofferDB vector defect.

| Preset | Mode | Recall@5 | Recall@10 | MRR@10 |
| --- | --- | --- | --- | --- |
| expanded | text_bm25 | 505/576 (0.876736) | 67/72 (0.930556) | 19/20 (0.950000) |
| expanded | vector_exact | 29/36 (0.805556) | 505/576 (0.876736) | 329/384 (0.856771) |
| expanded | hybrid | 511/576 (0.887153) | 539/576 (0.935764) | 19/20 (0.950000) |
| balanced | text_bm25 | 505/576 (0.876736) | 67/72 (0.930556) | 19/20 (0.950000) |
| balanced | vector_exact | 29/36 (0.805556) | 505/576 (0.876736) | 329/384 (0.856771) |
| balanced | hybrid | 511/576 (0.887153) | 539/576 (0.935764) | 19/20 (0.950000) |
| constrained | text_bm25 | 505/576 (0.876736) | 67/72 (0.930556) | 19/20 (0.950000) |
| constrained | vector_exact | 29/36 (0.805556) | 505/576 (0.876736) | 329/384 (0.856771) |
| constrained | hybrid | 511/576 (0.887153) | 137/144 (0.951389) | 213/224 (0.950893) |

SQLite's constrained window changes the pure-text Top10 for `C08-F` and hybrid
rankings for 32 tasks relative to its other presets. Every SQLite group and all
per-query ranks are retained. Neither these observations nor matching top_k
constitute equal recall work, matching score scales or an end-to-end speed ratio.

## Latency and resource scope

The host was a native macOS 27.0.1 arm64 environment, Apple M1 Pro, eight logical
CPUs, 34,359,738,368 bytes RAM; the build used Rust 1.97.1. Local temporary
filesystem, shared host scheduling and background load apply. Physical disk I/O,
SSD/controller cache, per-query CPU attribution and energy were not instrumented.

Each row pools 96 raw durations from two rounds of the same engine, preset, mode
and cache group. P50/P95 use nearest rank, not the mean of round percentiles.
Only 48 independent tasks underlie those repeated samples. SQLite CLI startup is
included in its query times; embedded CofferDB process startup is excluded.

| Engine | Preset | Mode | Cache | Samples | P50 ms | P95 ms |
| --- | --- | --- | --- | --- | --- | --- |
| cofferdb | balanced | hybrid | cold | 96 | 0.709291 | 0.857750 |
| cofferdb | balanced | hybrid | warm | 96 | 0.706292 | 0.867584 |
| cofferdb | balanced | text_bm25 | cold | 96 | 0.474000 | 0.582708 |
| cofferdb | balanced | text_bm25 | warm | 96 | 0.473708 | 0.588292 |
| cofferdb | balanced | vector_exact | cold | 96 | 0.418250 | 0.525208 |
| cofferdb | balanced | vector_exact | warm | 96 | 0.420292 | 0.504167 |
| cofferdb | constrained | hybrid | cold | 96 | 0.692750 | 0.850416 |
| cofferdb | constrained | hybrid | warm | 96 | 0.701208 | 0.824792 |
| cofferdb | constrained | text_bm25 | cold | 96 | 0.462417 | 0.585125 |
| cofferdb | constrained | text_bm25 | warm | 96 | 0.450375 | 0.531208 |
| cofferdb | constrained | vector_exact | cold | 96 | 0.408375 | 0.484333 |
| cofferdb | constrained | vector_exact | warm | 96 | 0.408375 | 0.503125 |
| cofferdb | expanded | hybrid | cold | 96 | 0.717667 | 0.850500 |
| cofferdb | expanded | hybrid | warm | 96 | 0.709500 | 0.841750 |
| cofferdb | expanded | text_bm25 | cold | 96 | 0.483333 | 0.575042 |
| cofferdb | expanded | text_bm25 | warm | 96 | 0.479916 | 0.598000 |
| cofferdb | expanded | vector_exact | cold | 96 | 0.422833 | 0.525125 |
| cofferdb | expanded | vector_exact | warm | 96 | 0.421750 | 0.487333 |
| sqlite | balanced | hybrid | cold | 96 | 11.063709 | 13.174542 |
| sqlite | balanced | hybrid | warm | 96 | 10.842792 | 12.228708 |
| sqlite | balanced | text_bm25 | cold | 96 | 5.665459 | 7.249000 |
| sqlite | balanced | text_bm25 | warm | 96 | 5.477334 | 6.183666 |
| sqlite | balanced | vector_exact | cold | 96 | 5.331083 | 5.958084 |
| sqlite | balanced | vector_exact | warm | 96 | 5.301333 | 6.180792 |
| sqlite | constrained | hybrid | cold | 96 | 10.894750 | 12.134541 |
| sqlite | constrained | hybrid | warm | 96 | 10.756166 | 11.988292 |
| sqlite | constrained | text_bm25 | cold | 96 | 5.626125 | 6.429041 |
| sqlite | constrained | text_bm25 | warm | 96 | 5.462000 | 5.916875 |
| sqlite | constrained | vector_exact | cold | 96 | 5.298125 | 6.473042 |
| sqlite | constrained | vector_exact | warm | 96 | 5.336583 | 5.923708 |
| sqlite | expanded | hybrid | cold | 96 | 10.833500 | 12.745584 |
| sqlite | expanded | hybrid | warm | 96 | 10.826416 | 11.694250 |
| sqlite | expanded | text_bm25 | cold | 96 | 5.438916 | 6.248875 |
| sqlite | expanded | text_bm25 | warm | 96 | 5.512084 | 6.032667 |
| sqlite | expanded | vector_exact | cold | 96 | 5.276000 | 5.603625 |
| sqlite | expanded | vector_exact | warm | 96 | 5.418958 | 6.615208 |

The native macOS `time -l` maximum resident set values below describe each whole
timed benchmark process across import and all modes/cache groups. They are not
per-query allocations, cache budget limits or a sum of parent/child RSS peaks.
Native stderr preserves real/user/system timing and the sampler's own semantics.

| Round | Preset | Whole timed-process maximum RSS, bytes |
| --- | --- | --- |
| 1 | balanced | 20594688 |
| 1 | expanded | 20709376 |
| 1 | constrained | 20463616 |
| 2 | constrained | 22872064 |
| 2 | expanded | 21676032 |
| 2 | balanced | 21086208 |

Full per-query search costs, stage journals and logical-work counters remain in
the result/journal records. Overlapping phase diagnostics must not be summed as
end-to-end cost. Null physical-read fields are unavailable measurements, not zero.
This small-corpus matrix does not establish 100k-scale behavior, Linux performance,
minimum-Rust compatibility, 2 GiB recovery or CI acceptance.

## Failed attempt and completion boundary

The first execution failed when a restricted native timing probe returned
`Operation not permitted`. Its first group contains 288 completed CofferDB journal
samples, 288 SQLite samples whose completion was not accepted as verified, and
five unlaunched profile groups marked not_run. Its result capture is retained as
a partial diagnostic; no metrics are pooled into the successful execution.

After recording the failure, a read-only native capability probe succeeded. A
fresh execution then repeated all six batches under the unchanged frozen method,
inputs, model, weights, engine and auditor. All six batches exited successfully,
with no active samples, timeout or surviving owned process group at exit. The
[first attempt](representative-quality-v1/v1/execution-derived.json),
[failure clarification](representative-quality-v1/v1/failure-record.json),
[complete execution](representative-quality-v1/v2/execution-derived.json), native
logs and full planned sample coverage remain separate. This was an execution
environment correction, not a scoring change or a partial-run resume.

Independent execution and metric review conclusions are recorded separately from
this numerical report. A successful producer run does not itself close release
acceptance. See the [replay guide](representative-replay.md) for metric recomputation,
logical reruns, exact source binding and the original measurement auditor's limits.
