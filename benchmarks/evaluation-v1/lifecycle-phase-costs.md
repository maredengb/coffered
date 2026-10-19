# Historical lifecycle phase costs

This archive maps 41 existing worker observations into 17 phase labels. It is a
read-only historical association, not 41 new trials, an operation latency
percentile or a current-binary performance run. The [full phase report](historical-phase-cost-v2/phase-cost-report-derived.json)
and [lineage](historical-phase-cost-v2/artifact-lineage.json) retain all measured
values, original source/binary identities, original states and limitations.

The macOS arm64 evidence combines a ten-collection, 100k-total-chunk workflow
(maximum 10k per collection) and four prescribed interruptions of compaction on
a historical 2,147,603,776-byte, one-collection 100k-chunk/768-dimension fixture.
These are separate input and timing axes. They do not prove single-collection
100k quality, all 768-dimension workloads or Linux behavior.

The continuous import-to-verified-query-group observation was
63.014952125027776 seconds. Its timer begins before the first import and includes
process startup, post-stage stats and query assertions. That initial group covers
collection 4 text/vector and collection 0 target paths, not all ten collections'
index readiness. It is not an inner import-only measurement.

The table displays source monotonic process wall durations and native maximum
RSS; the raw JSON retains full precision and all other fields. Blank case denotes
the multicollection sequence. A compact worker's interrupted duration is time to
the prescribed interruption, not a successful completed compaction latency.

| Axis | Phase | Stage/case | Process wall seconds | Maximum RSS bytes | Recorded outcome |
| --- | --- | --- | --- | --- | --- |
| 10collection100k_update_delete | empty_file_creation | init | 0.5439362499164417 | 2834432 | see original stage record |
| 10collection100k_update_delete | bounded_import_and_index_build | import-0 | 1.6293222079984844 | 80920576 | see original stage record |
| 10collection100k_update_delete | bounded_import_and_index_build | import-1 | 1.9310710000572726 | 131350528 | see original stage record |
| 10collection100k_update_delete | bounded_import_and_index_build | import-2 | 2.372683000052348 | 183648256 | see original stage record |
| 10collection100k_update_delete | bounded_import_and_index_build | import-3 | 2.80456899991259 | 213237760 | see original stage record |
| 10collection100k_update_delete | bounded_import_and_index_build | import-4 | 3.2181202908977866 | 283426816 | see original stage record |
| 10collection100k_update_delete | bounded_import_and_index_build | import-5 | 3.6525855410145596 | 317489152 | see original stage record |
| 10collection100k_update_delete | bounded_import_and_index_build | import-6 | 4.082967541995458 | 360218624 | see original stage record |
| 10collection100k_update_delete | bounded_import_and_index_build | import-7 | 4.6372510839719325 | 404045824 | see original stage record |
| 10collection100k_update_delete | bounded_import_and_index_build | import-8 | 5.196119250031188 | 490946560 | see original stage record |
| 10collection100k_update_delete | bounded_import_and_index_build | import-9 | 5.705069415969774 | 536002560 | see original stage record |
| 10collection100k_update_delete | open | open | 5.170060458010994 | 271368192 | see original stage record |
| 10collection100k_update_delete | first_query_group | initial-queries | 5.351569875027053 | 529596416 | see original stage record |
| 10collection100k_update_delete | update_then_queries | replace | 12.936699541984126 | 811892736 | see original stage record |
| 10collection100k_update_delete | delete_drop_then_queries | delete-drop | 15.554887250065804 | 795377664 | see original stage record |
| 10collection100k_update_delete | compact_then_queries | compact | 15.245828000013717 | 1115095040 | see original stage record |
| 10collection100k_update_delete | check | check | 3.9913073329953477 | 483704832 | see original stage record |
| 10collection100k_update_delete | jsonl_export | export | 2.812052792054601 | 244580352 | see original stage record |
| 10collection100k_update_delete | safe_file_copy | safe-copy | 0.8801172090461478 | 20463616 | see original stage record |
| 10collection100k_update_delete | safe_copy_reopen_queries | reopen-copy | 4.904157957993448 | 445759488 | see original stage record |
| 10collection100k_update_delete | safe_copy_check | check-copy | 4.035352125065401 | 483688448 | see original stage record |
| historical2GiB100k768 | compact | fast-before_publication | 14.185548500000001 | 1104592896 | expected_SIGKILL_before_return |
| historical2GiB100k768 | verify-after-kill | fast-before_publication | 26.041239875 | 2057273344 | passed |
| historical2GiB100k768 | recover | fast-before_publication | 32.479370334 | 2059894784 | passed |
| historical2GiB100k768 | reader | fast-before_publication | 94.007584291 | 1100988416 | passed |
| historical2GiB100k768 | final-reopen | fast-before_publication | 25.823222167 | 2057093120 | passed |
| historical2GiB100k768 | compact | fast-after_publication | 18.046618417 | 1369849856 | expected_SIGKILL_before_return |
| historical2GiB100k768 | verify-after-kill | fast-after_publication | 6.8456072919999995 | 535232512 | passed |
| historical2GiB100k768 | recover | fast-after_publication | 12.09414725 | 679067648 | passed |
| historical2GiB100k768 | reader | fast-after_publication | 55.613878375 | 1101234176 | passed |
| historical2GiB100k768 | final-reopen | fast-after_publication | 5.177026041 | 678379520 | passed |
| historical2GiB100k768 | compact | commit-before_publication | 13.818762125 | 1107460096 | expected_SIGKILL_before_return |
| historical2GiB100k768 | verify-after-kill | commit-before_publication | 25.868181749999998 | 2055946240 | passed |
| historical2GiB100k768 | recover | commit-before_publication | 32.170487916999996 | 2063941632 | passed |
| historical2GiB100k768 | reader | commit-before_publication | 92.46836754200001 | 1101185024 | passed |
| historical2GiB100k768 | final-reopen | commit-before_publication | 25.759466917 | 2057519104 | passed |
| historical2GiB100k768 | compact | commit-after_publication | 18.049078083 | 1370603520 | expected_SIGKILL_before_return |
| historical2GiB100k768 | verify-after-kill | commit-after_publication | 6.835087792 | 535117824 | passed |
| historical2GiB100k768 | recover | commit-after_publication | 12.082435625 | 678608896 | passed |
| historical2GiB100k768 | reader | commit-after_publication | 55.576606459000004 | 1101250560 | passed |
| historical2GiB100k768 | final-reopen | commit-after_publication | 5.154357624999999 | 676888576 | passed |

## Resource and failure boundaries

Whole-operation and separate stats processes have their own boundaries. A reader
worker spans its case lifetime; recovery includes queries before/after rebuild
and integrity checking. Do not add worker RSS maxima or overlapping durations to
invent a simultaneous peak or total. Explicit cache settings include 256 MiB for
the multicollection workflow and 1024 MiB for the large-file workers; the latter
is an override, not the default cache budget or a process memory cap.

Directory samples are non-atomic logical sums. Their sampled maxima are not a
strict simultaneous disk peak or a physical-space bound. Per-phase recovery
extra disk peak, physical read/write bytes and exclusive old-reader physical
extents are unknown. Kernel block-operation counters do not fill missing physical
byte fields. There was no OS cache eviction. No single reader spanned the entire
multicollection workflow.

The four prescribed cases observe before/after publication under two durability
settings. Expected killed-worker exits/signals and later verification/recovery
remain distinct. They are not random crashes, power failures, exact fsync-boundary
coverage or blanket durability proof. The two original execution reports retain
all not_run and failure-related fields. Their source identity is historical;
current naming or documentation changes cannot make these new-binary timings.

## Included artifacts and replay scope

All 110 small input artifacts bound by the original phase collector were checked
against its hashes. This public projection includes 107 measurement artifacts:
two protocols, two reports, all 82 worker stdout/stderr streams and 21 directory
sample records. Three internal review/source-applicability sidecar contents are
omitted with their original hashes retained. The derived phase report preserves
all 41 rows and historical failed-quality/incomplete-version states. Internal
authority/scope prose and private filesystem references are explicitly omitted
or aliased; the original whole-artifact hashes remain traceable.

The [lifecycle replay guide](lifecycle-replay.md) and original logical helper cover
the multicollection inputs and operations on a new database. The original
protocols document timers, sampling and interruption observations; their private
path aliases are not runnable installation paths. The portable logical guide
is not an automated reproduction of the original native resource sampler.

The historical large-file database itself is not redistributed here. Its original
bytes/hash, synthetic-content oracle, source/helper identities, trigger rules,
complete worker events and outputs remain in the large-file protocol/report.
This archive does not claim a complete from-empty reconstruction of those exact
historical bytes. A new large-file generation or measurement must have its own
input/history/source and execution identity. No large file was reopened, hashed,
generated or crash-tested during publication preparation of these records.

The [later recovery-cost observation](historical-recovery-cost-v1/artifact-lineage.json)
and its [earlier incomplete attempt](historical-recovery-incomplete-v1/artifact-lineage.json)
remain separately documented. Historical failed relevance criteria are preserved
as old verdicts, not reactivated as current numerical gates. None of these cost
projections certifies a retrieval configuration or closes whole-version release
acceptance.
