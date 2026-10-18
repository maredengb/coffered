# Backup, export and data handling

Keep original inputs and independent backups when using this experimental
pre-release. Storage integrity, application relevance and backup completeness are
different checks. `check` is read-only and is not a repair command or an evaluation
of whether the stored text answers a user's question.

## Copying a database file

Stop all writes and maintenance, close the writer successfully, and prevent any
other process from reopening a writer until copying finishes. In a Rust
application, check `Database::close()` for errors; stopping one request or dropping
an unrelated collection handle does not establish that the writer is closed.
Respect the selected durability mode: fast mode does not imply a completed fsync
or protection against power loss.

Copy the whole `.coffer` file to a separate destination. Check the copy, open it
with the intended software version, and compare representative queries and data
counts with the source. For a complete backup test, verify all required records
and vectors against retained inputs or an independent logical export. A checksum
can verify that a quiescent file was copied byte-for-byte; it cannot prove that
the source was already correct or that an actively changing copy is consistent.

Ordinary file copying while a writer or compaction is active is not a supported
snapshot protocol. Read-only handles may retain older file generations after
compaction, delaying physical disk reclamation. A copy is neither anonymization
nor a way to erase deleted history. Do not copy only the index suffix.

## JSONL export and restore

Export operates on one collection at a time. It writes live documents, chunks and
embeddings plus `export_manifest.json`. The manifest records collection vector
configuration, counts and hashes for the JSONL files. Retain the complete set and
verify the hashes before restore; a standalone JSONL file is not the full export.
An interrupted operation can leave incomplete artifacts. Successful publication
of a manifest does not make arbitrary later changes to its files valid.

Run this from the product directory, beside `Cargo.toml`, using a fresh work area:

```sh
cargo build --locked
coffer_backup_dir="$(mktemp -d)"
./target/debug/coffer init "$coffer_backup_dir/source.coffer"
./target/debug/coffer import "$coffer_backup_dir/source.coffer" --manifest examples/data/markdown/import_manifest.json --json
./target/debug/coffer export "$coffer_backup_dir/source.coffer" "$coffer_backup_dir/export" --collection notes --format jsonl --json
./target/debug/coffer init "$coffer_backup_dir/restored.coffer"
./target/debug/coffer import "$coffer_backup_dir/restored.coffer" --manifest "$coffer_backup_dir/export/export_manifest.json" --json
./target/debug/coffer check "$coffer_backup_dir/restored.coffer" --verify-indexes --json
./target/debug/coffer search "$coffer_backup_dir/restored.coffer" --collection notes 'write ahead' --json
```

The last command should return `wal#commit` first. This small round trip checks the
documented workflow; it is not a complete backup verification of an arbitrary
database. Repeat export for every collection you need, preserve source inputs,
and independently compare the restored public fields and vector values.

The exported public fields are:

| File | Fields |
| --- | --- |
| `documents.jsonl` | `external_id`, `title`, `source_uri`, `metadata` |
| `chunks.jsonl` | `document_external_id`, `chunk_external_id`, `ordinal`, `text`, `source_span`, `metadata` |
| `embeddings.jsonl` | `chunk_external_id`, `model_name`, `dim`, `vector` |

Internal generated IDs, creation/update times, tombstones, transaction history
and persisted index bytes are not a logical export of those fields. A restored
database can therefore have different internal IDs and binary bytes. Vector
configuration must match the intended collection; preserve `vectors_normalized`
and do not guess missing model/dimension/metric information in older assets.

A historical collection with conflicting embedding models can still be exported
for rescue, retaining each row's original `model_name`. Such an export is not
promised to reimport unchanged into one model-consistent collection. Resolve the
conflict explicitly, for example by separating compatible models, removing selected
conflicting vectors or generating a new consistent vector set. Preserve the
originals and record the chosen transformation. Compaction or index rebuilding
does not repair model incompatibility, and vectors must not simply be relabeled.

## Privacy, deletion and maintenance

Database files, JSONL exports and backups can contain original text, vectors,
paths and metadata. Search, explain, stats, error and benchmark output can also
contain sensitive values. Inspect and sanitize outputs before sharing them;
read-only access does not redact or encrypt anything. Core operations do not
upload data or collect telemetry, but normal filesystem permissions and the
surrounding application still matter.

Delete writes logical tombstones. Compaction can remove obsolete records from a
new live file, but old readers, backup copies, filesystem snapshots and device
history can retain earlier bytes. Neither delete, compaction, export nor copying
guarantees secure SSD erasure, encryption, anonymization or removal from every
historical copy. There is no built-in redacted-export promise.

Recovery validates the [critical commit stream](file-format.md); it does not
replace an independent backup or prove that every crash/power-loss scenario was
tested. Report corruption using a minimal sanitized reproducer, version/commit,
platform, filesystem, durability setting and relevant check output. See the
[security policy](../SECURITY.md) before sharing a security-sensitive report.
