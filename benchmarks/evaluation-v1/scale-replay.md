# Synthetic 100k scale workload

This workload exercises database size and query mechanics, not semantic quality.
It contains 1,000 documents, 100,000 chunks and 100,000 synthetic 768-dimensional
one-hot vectors. No embedding model, inference service or model download is used.
The three reconstructed JSONL files total 190,817,950 bytes. Database files,
indexes, compiler output and temporary import space require additional disk space.

[`prepare-scale-input.py`](prepare-scale-input.py) is a later reconstruction
recipe checked against every historical input row and all three original byte
hashes. The original generator and its seed/history were not recovered. The new
recipe reproduces the exported logical input; it does not reproduce the original
database file bytes, transaction history, internal IDs, timings or the separate
approximately 2 GiB recovery workload.

## Prepare and import

Use Python 3.11 or later and a supported Rust toolchain. These commands assume
this evaluation directory is the current directory and the product source is in
`../cofferdb`. Adjust that source path to your checkout. Choose an output path
that does not exist; the generator refuses to replace an existing directory.

```sh
python3 prepare-scale-input.py scale-input
cargo build --release --manifest-path ../cofferdb/Cargo.toml --lib --bin coffer
../cofferdb/target/release/coffer init scale-input/scale.coffer
../cofferdb/target/release/coffer import scale-input/scale.coffer \
  --manifest scale-input/import_manifest.json --json
../cofferdb/target/release/coffer check scale-input/scale.coffer --json
rustc --edition=2021 -O scale-helper.rs \
  --extern cofferdb=../cofferdb/target/release/libcofferdb.rlib \
  -L dependency=../cofferdb/target/release/deps -o scale-helper
./scale-helper scale-input/scale.coffer balanced --smoke > scale-smoke.jsonl
```

The build paths above assume the default Cargo target directory. If you set
`CARGO_TARGET_DIR` or a custom target, use that same library and dependency path
when compiling the helper. The import manifest is newly generated convenience
metadata with cosine distance and `vectors_normalized=false`; it is not an
amended historical export manifest. Import creates the `lifecycle` collection.
Use the same Rust toolchain and target for the library and helper. Cross-compiling
requires matching `--target` options for Cargo and `rustc`, and the corresponding
target output paths; execute the result on a compatible machine.

The helper opens the database read-only and checks the 100k counts, exact top-ten
IDs and scores, persistent checkpoint use, scanned-vector count and distance work.
It uses keyword `764`, the first basis vector, and optionally the `project=4`
metadata filter. These deterministic assertions concern this synthetic workload;
they are not a general retrieval oracle or semantic-quality test.

## Optional query measurements

Use `expanded`, `balanced` or `constrained` as the profile argument. `--smoke`
executes one query in each of eight contexts: text, vector, hybrid and filtered
hybrid, each with execution-only or standard JSON rendering. A successful smoke
run emits eight query rows and a terminal `complete` row; it is not a percentile
or performance acceptance run.

Omitting `--smoke` executes one first query, two warmups and twenty measured
queries per context: 184 query rows in one worker. The helper reopens a database
handle for each context; it does not evict operating-system caches. Execution-only
timing covers `execute()`. JSON timing also covers ordinary pretty JSON rendering;
verification and writing the measurement row are outside both intervals. Query
cloning and database opening are also outside the query interval. Open durations
are reported separately.

The helper has no internal timeout or resource sampler. Supervise it using your
platform's process controls, preserve nonzero exits and incomplete output, and do
not treat an absent terminal row as success. Whole-worker RSS, CPU and elapsed
time include opening, checks and output and cannot be assigned to a single query.
The emitted counters are logical engine work, not physical disk bytes or energy.

For a new comparative experiment, record the source, compiler, helper and input
hashes, hardware, operating system, order, repetitions and timing rules before
running. Preserve every sample and failure. A single twenty-sample context is
not the historical two-round aggregate; recompute nearest-rank P50/P95 from the
appropriate raw measured samples rather than averaging percentiles. This recipe
does not itself establish a new release protocol or performance claim.

## Historical observations

[`historical-scale-v2/artifact-lineage.json`](historical-scale-v2/artifact-lineage.json)
identifies the original macOS and Linux protocols, all six workers per platform,
raw samples, resource records and derived reports. Each platform has 1,104 query
calls, including first and warmup calls; each same-context two-round measured
group has forty samples. The archived identities and negative observations remain
unchanged. External-reference aliases locate archived inputs and are not local
installation paths.

These are historical measurements. Later profile naming and JSON output changes
can change serialized byte lengths and timings. Current source inspection found
no changes to the relevant core query, import or storage algorithms, but that
does not establish identical machine code or current performance. A new helper
run must retain its own source and binary identity. The historical scale helper
did not record per-query read-call/byte or five-stage cost snapshots; these fields
are unavailable for those runs rather than inferred from smaller datasets.
