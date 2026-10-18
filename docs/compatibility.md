# 0.1.0 compatibility and upgrade notes

These notes apply to the 0.1.0 library, CLI and interchange artifacts. Earlier
experimental snapshots did not all implement the same input and scoring rules.
Keep the producing version, original inputs and an independent backup when moving
between snapshots. A JSON `schema_version` of `"0.1"` alone does not establish
compatibility with every previous client or snapshot.

## JSON processing and errors

The current JSON processing policy permits at most 128 container levels per
business value, 8 MiB of canonical JSON per business value and 32 MiB of JSON
workspace per operation. Fixed transport envelopes do not add user nesting to a
business root. A business value is the logical value at its entrance, such as
metadata, a filter, weights or a configuration, rather than an entire surrounding
CLI response or JSONL dataset. These are processing limits, not a claim that a refused value is
invalid JSON, a whole-dataset limit or a total-process RSS limit.

Earlier experimental importers included a 1 MiB raw manifest guard and a 64 MiB
per-JSONL-file raw-input cap, also constrained by the remaining atomic-import
budget. Those wire-size guards are not the current business-root and canonical-size
policy. Whitespace, wrapping and operation workspace can affect
the comparison: do not treat matching depth constants or one accepted old file
as proof that every input remains accepted.

| Affected contract | 0.1.0 behavior and client action |
| --- | --- |
| JSON limit diagnostics | Handle conditional `json_input_kind`, `json_limit_reason`, `limit`, `limit_unit` and, when available, `actual`. The input-kind values include `jsonl_record`. Update strict decoders and snapshots; these details are not mandatory on every error. |
| Filters, weights and metadata boost | Limit refusal retains the input-specific error code, including `invalid_metadata_filter`, `invalid_search_weights` and `invalid_metadata_boost`. Handle their conditional reason fields rather than treating every limit as `invalid_jsonl_schema`. |
| Stored JSON | A limit that prevents safely reading committed data uses `recovery_error`. The file is preserved; records are not skipped, truncated or silently replaced with an older snapshot. This is different from proving physical corruption. |
| Other controlled resource limits | Handle `resource_exhausted` and CLI exit 2, with the applicable resource/operation/budget details. It is not a syntax error, a broken-file diagnosis or successful partial search output. |
| Suggested actions | The error envelope always contains `suggested_action`; an ordinary error may use a nonempty string or `null`. Error cases with a specified recovery action still provide that action. Accepting `null` does not permit a missing key. |
| Source spans and filter paths | Optional source spans use byte offsets; support the documented number/decimal-string representation for large offsets instead of forcing every offset through a floating-point number. Filter dot paths are validated, not silently reinterpreted as arbitrary metadata keys. |

JSON workspace can use aggregate reservations at documented allocation
boundaries. Admission timing and diagnostic requested/actual amounts may therefore
differ from earlier snapshots even with the same configured budget. This does
not change the 128-level, 8 MiB and 32 MiB policy or permit hidden budget increases.

If a write operation fails, inspect committed progress before selecting a retry
and conflict policy. A failed import is not a promise that no earlier batch
committed. Do not respond to resource refusal by truncating valid data.

If an existing file cannot be safely read under the current policy, preserve it
and use a known compatible producer/reader to export it when that reader can do
so safely. Explicitly adapt the exported data or choose a compatible version
before reimporting. Without such a reader there is no promised automatic lossless
repair. Follow the quiescent-copy and logical-restore procedures in
[backup and data handling](backup.md); a failed export or compact operation must
not overwrite the authoritative source as a migration shortcut.

## Compaction and retained generations

`records_removed` is the decrease in committed business-record count across a
successful compaction, not a live-object count, deletion-request count or number
of tombstones. The business records are collection create/drop, document put,
chunk put, embedding put and tombstone records. Headers, commits, compact markers
and rebuildable indexes are excluded. Retained collection-name create/drop records
still count in the compacted file.

`tombstones_removed` is the decrease in committed tombstone records. It is not an
alias for `records_removed`, and the two values must not be added as independent
reclaimed totals. Replacing a collection tombstone with an equivalent drop can
produce `records_removed = 0` and `tombstones_removed = 1`. Clients that interpreted
older experimental counters as deleted-object totals must update that logic.
`segments_merged` counts old persistent index segment records processed, not the
number of final queryable segments; use index statistics for the latter.

Old readers may keep earlier file generations alive after compaction. Resource
accounting covers the documented local runtime scope, not a global quota across
other processes or the lifetime of a previous writer. Opening an alias to an old
file retained by local readers for writing can fail with `recovery_error`, exit 2
and `detail.reason = "writer_reopen_required"`. Close a stale writer if present,
identify the current main file and reopen it. Do not delete a valid reader's file
or retry the old alias as a repair strategy.

## Embedding models and export configuration

A collection uses one effective embedding-model identity. Without explicit model
configuration, the binding follows committed embedding data; it is not written
back as an explicit collection setting. Skipped writes do not change the binding.
Without an explicit model setting, removing all visible embeddings permits a
later write to establish a new binding; an explicit setting still applies.
Replacing data does not bypass the model rule. Duplicate identities inside an input are rejected before target-database
`skip`/`replace` handling, so conflict options cannot legalize that input.

Historical conflicting-model data is not automatically rewritten. Preserve the
source, inspect the reported issues, and retain rescue reads/exports before using
supported explicit deletion or a curated reimport. A rescue export is not a
promise that an unedited conflicting dataset can pass normal reimport.

Exports preserve `embedding_dim`, `distance_metric` and boolean
`vectors_normalized`, including `false`. The exported `embedding_model` represents
only explicit configuration; an inferred binding is not substituted for a missing
or null configuration. Carry these fields and tokenizer settings through restore
or artifact-conversion tools. Preserve vector values: CofferDB does not normalize
or regenerate them during this round trip.

For import, omitting `vectors_normalized` gives a new collection `false` and keeps
an existing collection's setting. An explicit value must be boolean and agree with
the existing configuration, even when no vectors are written. Older exports that
omit this flag cannot identify an original `true` setting from norms, metric or
model name alone. Re-export from the compatible source or supply trusted original
configuration explicitly; do not claim that defaulting to `false` restores the
original dot-product semantics. Missing dimension/metric information also needs
an explicit trustworthy source. Added manifest fields can change its checksum
without rewriting the JSONL data.

## Queries, presets and benchmark artifacts

0.1.0 completes applicable branch scores across the eligible candidate union
before normalization and final ranking. A candidate outside a branch's initial
window can still receive that branch's computable score; a genuinely absent
embedding remains absent. Compared with window-hit-only scoring, raw and
normalized scores, ranking, explain output and query work can change. Keep the
scoring policy, inputs and source identity with comparisons, and regenerate
strict ranking snapshots from a checked correctness oracle. No universal
relevance improvement is implied. Default weights are unchanged; callers still
control weights through the supported API and CLI options.

The supported Rust variants are `Expanded`, `Balanced` and `Constrained`; string
inputs use exactly `expanded`, `balanced` and `constrained`. Update source
references and strict enumerations accordingly. Other spellings are rejected,
not aliased, trimmed or converted. `balanced` remains the default. Presets choose
resource defaults, not relevance or speed guarantees. Preserve historical reports
as produced instead of rewriting their inputs or results to match new names.

`search` can omit text when it supplies a query embedding. Each effective
candidate window is `max(preset default, top_k)`, not an immutable preset cap;
this does not enlarge resource budgets. `top_k` is a JSON number representing an
integer in `1..=min(usize::MAX, 9007199254740991)` on the executing target. Update
validators that previously accepted a larger platform-sized integer range.

Select a benchmark resource preset through `--profile`. Dataset manifests and
query records have no resource-preset selector, and adding such top-level fields
is an input-schema error even for a valid preset value. SQLite baseline `config`
still follows its defined whitelist; it gains no arbitrary passthrough fields.
A legal tokenizer profile and ordinary user metadata follow their own schemas,
not the resource-preset enumeration. Result-artifact contract validation does not
create a public result-JSON readback command. `--profile help` is a help operation,
not an executable or JSON preset value.

These changes retain the JSON envelope version `"0.1"`, existing canonical/hash
rules and the documented binary-format boundaries; they do not provide an
automatic data-file migration or universal future-version compatibility promise.
See [file format and recovery](file-format.md), [ranking explanations](explain.md)
and the [benchmark guide](../benchmarks/README.md) for the matching contracts.
