# Replay the representative study

This directory supplies the complete public inputs, fixed judgments, raw query
observations, original measurement auditor and exact evaluated source payloads.
Use the [report](representative-quality-report.md) and [artifact lineage](representative-quality-v1/artifact-lineage.json)
to interpret scopes and hashes. A rerun creates a new execution identity; it must
not replace archived observations or inherit their timing results.

## Recompute archived relevance metrics

Python 3.11 or later and its standard library suffice. From this evaluation
directory, choose a new output name:

```sh
python3 recompute-quality.py public-quality-v1/qrels.jsonl \
  representative-quality-v1/v2/round-1-balanced/result-derived.json \
  --output representative-metric-recheck.json
```

The tool recomputes all three modes and both cache groups for CofferDB and SQLite
in that result. Repeat with each of the other five result files for complete
matrix recomputation. It validates eligible IDs and query coverage and uses exact
rational arithmetic for macro metrics. It does not query a database, prove that
the engine computed the supplied scores correctly or audit process resources.
The [original report](representative-quality-v1/v2/quality-report.json) also retains
all cell metrics, single-branch signed changes and raw latency samples.

## Rebuild the evaluated source

Check the public archive SHA-256 against
[evaluated-source-v1/artifact-lineage.json](evaluated-source-v1/artifact-lineage.json):
`229e2e8cca20856161f0192fffcda6a7ff7d90750cc825f7f538992c13e1f861`.
The archive contains only 168 regular product files beneath `cofferdb/`, with
normalized owner/time metadata. Exact payload hashes and original modes are
recorded in that lineage and the [source identity](evaluated-source-v1/source-identity.json).
Extract to a new empty destination:

```sh
mkdir source-replay

tar -xzf evaluated-source-v1/source.tar.gz -C source-replay
COFFERDB_GIT_COMMIT=unknown cargo build --locked --release \
  --manifest-path source-replay/cofferdb/Cargo.toml
```

The commands assume Cargo's default target directory and native target. If
`CARGO_TARGET_DIR`, `CARGO_BUILD_TARGET` or Cargo configuration changes that
layout, use the actual rebuilt executable path in the query commands below.

Use an installed compatible Rust toolchain. The recorded measurements used Rust
1.97.1 on macOS arm64; a different compiler/platform/build creates a different
binary and cost identity. Record compiler version, target and the new executable
hash. The archived source has no Git history, and `unknown` prevents a rebuild
from claiming a historical commit. The recorded binary is not redistributed as
an official release artifact.

## Repeat the logical query matrix

Supply a SQLite CLI with FTS5 and a compatible sqlite-vec extension. Set
`COFFERDB_SQLITE3` and `COFFERDB_SQLITE_VEC_EXTENSION` to those local files before
running; loading a native extension executes its code, so use a trusted build.
The recorded reference used SQLite 3.54.0 and sqlite-vec 0.1.6; their binary hashes
and strategy are in the frozen protocol. The supplied vectors permit replay
without downloading or running an embedding model.

From this evaluation directory, with the rebuilt CLI at the path above and both
SQLite variables set, the following runs the six logical batches. The output
root must not already exist. Every batch gets a fresh database; mode and cache
iteration are performed by the existing benchmark command.

```sh
set -eu
: "${COFFERDB_SQLITE3:?Set the trusted SQLite CLI path}"
: "${COFFERDB_SQLITE_VEC_EXTENSION:?Set the trusted sqlite-vec extension path}"
export COFFERDB_SQLITE3 COFFERDB_SQLITE_VEC_EXTENSION
mkdir representative-rerun
for batch in 1-balanced 1-expanded 1-constrained 2-constrained 2-expanded 2-balanced
do
  preset=${batch#*-}
  mkdir "representative-rerun/$batch"
  source-replay/cofferdb/target/release/coffer benchmark \
    "representative-rerun/$batch/data.coffer" \
    --dataset public-quality-v1 --collection benchmark \
    --profile "$preset" --durability commit \
    --result-file "representative-rerun/$batch/result.json" \
    --require-baseline \
    > "representative-rerun/$batch/stdout" \
    2> "representative-rerun/$batch/stderr"
done
```

This reproduces the data, default configuration, fresh-database order and logical
query obligations. It is **not** the original external timeout/resource wrapper.
If a batch fails, the shell stops and retains its files. It does not infer
per-sample SQLite progress or create status records for future batches; inspect
the result and journal and report unproved progress as unknown. Do not continue
into the same database or pool incomplete runs into a later successful run.
Record input and source hashes, executable/dependency identities, host details,
commands, exits and raw outputs for the new execution. Recompute its metrics
against the supplied judgments, including all negative comparisons.

## Audit the original measurement method

The byte-identical
[original execution auditor](representative-quality-v1/original-execution-auditor.py)
and [derived frozen protocol](representative-quality-v1/frozen-method-protocol-derived.json)
expose the full original supervision, source-reference checks, score oracle,
query journal assertions, process cleanup, native macOS `time -l` capture and
sample accounting. The first failed attempt, later complete execution, full
plans, native stderr and supervisor outcomes are included. Private path aliases
are archival references, not executable local paths. The original auditor binds
original protocol/approval/input/source/binary hashes; running it directly against
path-substituted archive JSON is not a supported replay shortcut.

For a new measured execution, bind local paths, new hashes and host details in a
new method record and record the adapted auditor's hash. Keep the original
method and evidence immutable. The recorded method requires assertions enabled,
7200 seconds per whole batch, a 120-second externally observed/reported deadline
for each CofferDB query, 20 ms polling and absence of the owned process group on
exit. SQLite has no individually instrumented timeout. These external deadlines
are not strict engine deadlines. Native read-only hardware/time probes must work
before RSS measurements are accepted; the preserved first attempt demonstrates
why a probe failure is not database success.

Timing scopes remain engine-specific. Built-in cold means a reopened handle,
not an evicted OS cache. There is no discarded warmup. Pool only matching
engine/preset/mode/cache groups across complete rounds; use nearest-rank
percentiles over raw durations, never average P95 values. Native whole-process
RSS, application work counters and null physical-I/O fields retain their own
scopes. Logical query replay and metric recomputation do not certify new resource
measurements, semantic quality or a release.
