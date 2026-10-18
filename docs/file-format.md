# File format and recovery boundaries

This is an overview of the current experimental binary implementation. Keep
original JSONL inputs and exported backups. The presence of version fields and
byte-level tests does not itself promise compatibility with future releases or
complete format freeze. Use the matching CLI/API to read files rather than editing
their bytes. See [storage](../src/storage.rs), [payload codec](../src/storage/codec.rs)
and [recovery](../src/storage/recovery.rs) for the implementation.

## File domains

```text
FileHeader | CriticalLogStream through last valid Commit | RebuildableIndexSuffix
```

The current binary file header is 58 bytes. The critical stream begins at byte 58.
Recovery derives `critical_end_offset` from the end of the last complete, valid
critical commit. The optional index suffix begins at that offset; an advisory
header hint cannot move this boundary or hide later critical records.

Multi-byte integers and stored `f32` vector elements use little-endian encoding.
Lengths, counts and offsets are checked before use. JSONL is an interchange format;
the current binary payloads are typed encodings, not an on-disk JSON record log.

## Current file header

Ranges below are byte offsets with an exclusive end.

| Range | Meaning |
| --- | --- |
| `0..8` | ASCII `COFFERDB` magic |
| `8..10`, `10..12` | Binary format version pair: `0`, `1` |
| `12` | Little-endian marker `1` |
| `13` | Checksum algorithm marker `1` for CRC-32C |
| `14..22` | Feature flags; current writer emits zero |
| `22..30` | Critical stream start, currently `58` |
| `30..38` | Creation timestamp in Unix milliseconds |
| `38..40` | Critical header length, `44` |
| `40..44` | CRC-32C of bytes `0..40` |
| `44..54` | Advisory hint version and index offset |
| `54..58` | CRC-32C of advisory bytes `44..54` |

Unknown mandatory version/algorithm/feature values are rejected. Advisory hints
do not establish main-data visibility and must not be trusted as recovery roots.
The reader also contains a legacy 42-byte-header path; that older representation
is not the current writer format. Do not infer automatic migration or universal
old/new compatibility from legacy-reader code.

## Critical records and commits

A current critical record has a 50-byte envelope:

| Range | Meaning |
| --- | --- |
| `0..4` | ASCII `CFRL` |
| `4..6` | Record type |
| `6..8` | Record version `1` |
| `8..16` | Total envelope plus payload length |
| `16..24` | Batch ID |
| `24..32` | Sequence number |
| `32..34` | Header length `50` |
| `34..42` | Payload length |
| `42..46` | CRC-32C of payload bytes |
| `46..50` | CRC-32C of envelope bytes `0..46` |

Critical record types are collection create/drop, document put, chunk put,
embedding put, tombstone, commit and compact marker. The file header is not an
additional log record. Index records are not members of a critical batch.

Record checksums detect damage to individual records. A commit also validates
batch identity, record count, sequence range and an ordered CRC-32C over the
serialized critical data-record envelopes and payloads in that batch. The commit
record itself and the index suffix are excluded. The codec serializes this as a
32-bit checksum even though some in-memory fields use a wider integer. This
ordered checksum must not be replaced with XOR of record checksums.

Only a valid commit makes the pending batch visible. A damaged or incomplete
critical batch stops later main-data replay. A `CompactMarker` carries generation,
source-commit and ID high-watermark information, plus an ordered checksum of the
rewritten critical data records before that marker. Its checksum excludes commits,
the marker itself and the suffix; its containing batch still needs a valid commit.

## Rebuildable checkpoint suffix

The suffix has a `CFIDXCHK` checkpoint header, index records, an extent table and
a `CFOK` complete trailer. Index record envelopes use `CFIR`. Metadata records
describe snapshot coverage; text and vector segments hold the corresponding
rebuildable data. The current coverage mode represents a full visible snapshot.

Acceptance requires compatible versions, bounded lengths and extents, required
structural/record/segment/referenced-index checksums, a complete trailer, matching
generation and matching covered critical boundary, batch and commit sequence.
Coverage metadata/checksums must agree with the visible main-data snapshot. A
trailer marker alone does not establish completeness. If only the supplementary
aggregate checkpoint-payload checksum differs after all lower-level and referenced
index checks pass, the reader can keep the persistent checkpoint and report
non-blocking `checkpoint_payload_checksum_mismatch`. That diagnostic alone does
not require a rebuild or make validated main data unavailable.

Before appending a new critical batch, the writer discards the old suffix back to
the critical boundary. A missing, stale, incomplete or corrupt suffix cannot
invalidate already validated main commits. It can prevent persistent-index use;
the query path must rebuild within its budget or report the actual index/resource
failure. This separation does not make corrupted critical records recoverable.

## Checks, backup and compatibility

`coffer check <path> --verify-indexes` examines main-data and index findings without
repairing the file. Preserve an original file before trying maintenance or a
different software version. Compatible export/import preserves the documented
public data, not internal IDs, commit history, timestamps or the original binary
file hash. A rescue export of historical mixed-model data requires explicit model
resolution before reimport; exporting successfully does not resolve that conflict.
See [backup and data handling](backup.md) for a runnable round trip, safe file-copy
conditions and the limits of deletion, compaction and recovery.
