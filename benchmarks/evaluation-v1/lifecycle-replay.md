# Synthetic multi-collection lifecycle replay

This logical workload has ten collections, 1,000 documents and 100,000 chunks
with synthetic 768-dimensional one-hot vectors. It checks import, replacement,
deletion, collection removal, compaction, export and retained-reader visibility.
It is not semantic-quality evidence or the approximately 2 GiB recovery replay.

The [input lineage](lifecycle-inputs-v1/artifact-lineage.json) identifies a new
portable archive containing 44 byte-identical original input files. A preparation
record containing a private source reference was omitted from that archive and
projected separately. Tar ownership/timestamps and gzip headers were normalized;
the new archive deliberately has a different identity. The original eleven
manifests are unchanged. [`prepare-lifecycle-input.py`](prepare-lifecycle-input.py)
also writes eleven separately named current manifests with explicit
`vectors_normalized=false`, preserving the original data hashes.

## Prepare a fresh logical run

Use Python 3.11 or later, a native Rust toolchain supported by the product, and
new output paths. Commands run from this evaluation directory; the product
checkout is assumed to be `../cofferdb`. Adjust that source path as needed. Follow
the sequence only after the previous command succeeds. Input preparation needs
approximately 191 MB uncompressed, with additional space for compilation,
database, indexes, export, copy and temporary compaction output.

```sh
python3 prepare-lifecycle-input.py lifecycle-input
cargo build --release --manifest-path ../cofferdb/Cargo.toml --lib --bin coffer
rustc --edition=2021 -O lifecycle-inputs-v1/lifecycle-helper.rs \
  --extern cofferdb=../cofferdb/target/release/libcofferdb.rlib \
  -L dependency=../cofferdb/target/release/deps -o lifecycle-helper
../cofferdb/target/release/coffer init lifecycle-input/run.coffer
python3 - <<'PY'
from pathlib import Path
import subprocess
for number in range(10):
    manifest = Path('lifecycle-input/input') / f'input-{number}' / 'manifest-current.json'
    subprocess.run(['../cofferdb/target/release/coffer', 'import',
                    'lifecycle-input/run.coffer', '--manifest', str(manifest), '--json'], check=True)
PY
./lifecycle-helper lifecycle-input/run.coffer initial
./lifecycle-helper lifecycle-input/run.coffer replace lifecycle-input/replacement/manifest-current.json
./lifecycle-helper lifecycle-input/run.coffer delete
./lifecycle-helper lifecycle-input/run.coffer compact
./lifecycle-helper lifecycle-input/run.coffer check
./lifecycle-helper lifecycle-input/run.coffer export lifecycle-input/export
cp lifecycle-input/run.coffer lifecycle-input/copy.coffer
./lifecycle-helper lifecycle-input/copy.coffer reopen
```

The default target paths assume no custom `CARGO_TARGET_DIR` or target setting.
Use the same toolchain and target for the library and helper; cross-compilation
needs matching target options and compatible execution hardware. The retained
Rust helper is the original parameterized public-API program, with a new compiled
binary identity for each build. It is a fixed diagnostic harness, not a new
product API or a general-purpose safe-data migration utility.

The helper assumes the exact supplied data. It checks all collection counts,
queries the untouched collection 4 and affected collection 0, and retains an
earlier reader during each replace/delete/compact step. There is no single
reader spanning the entire run. It deletes document 750, drops collection 9 and
exports only collection 0. Initial query verification and export therefore do
not cover every collection individually. After deletion, nine collections,
899 documents and 89,900 chunks/vectors remain. The final copy is taken after
the preceding writers have closed; never use this sequence as instructions for
copying a database with an active writer.

Every helper invocation must exit zero and emit its terminal `complete` record.
It has no internal supervisor or process-resource sampler; use external timeout
controls and keep every nonzero exit or incomplete stream. Run the sequence on
fresh inputs to avoid accidental reuse of a partially mutated state.

## Historical measurements and limits

The [Linux measurement archive](historical-linux-lifecycle-v1/artifact-lineage.json)
retains 21 operation processes, 21 separately measured stats processes, disk
samples, query ledgers and all 60 original query observations. The derived
protocol keeps product-source identities and omits nonproduct source-map entries;
the original complete protocol still has its original hash. Private command paths
are archive aliases, not executable local paths.

These historical costs remain bound to the old protocol, source, Linux environment
and compiler. A current logical replay does not reproduce those millisecond/RSS
values or establish cross-platform performance. Directory samples are non-atomic
logical sums, not a strict physical peak. Separate stats processes cannot be added
to a narrower inner-operation RSS or timing value. Physical I/O and old-reader
physical extents remain unknown. Repeated semantic-quality tasks, new crash
cases, new platform acceptance and formal release approval are outside this run.

The separate [historical recovery-cost supplement](historical-recovery-cost-v1/artifact-lineage.json)
records six queries and one rebuild/check on an already-restored approximately
2 GiB file. It used an explicit 1,024 MiB cache, not the 256 MiB default, and
created no new crash cases. Its 308 disk samples are non-atomic observations.
Protocol input bindings are historical traceability references: an embedding
preparation script listed there was not executed as a recovery dependency. This
supplement neither requires new model inference nor reconstructs the old file's
growth history. Its original resource observations remain separate from the new
logical replay above.

An [earlier incomplete sampling attempt](historical-recovery-incomplete-v1/artifact-lineage.json)
is retained separately: the producer failed, three partial verified query rows
were emitted, and final resource samples/post-verification were unavailable.
Its saved `running` status is stale; the failure clarification records a terminal
producer error and is not evidence of a live process. The later successful
attempt does not repair or backfill that record.
