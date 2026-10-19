# Replay the historical scoring comparison

This source-only tool prepares the exact logical inputs and historical library
sources used by the accompanying observations. It runs the two complete historical
versions, not the current product API. Historical identifiers in archived sources
and observations are evidence labels; they are not supported current CLI aliases.
No Git history, credentials, model inference service or embedding download is needed.

## Requirements and scope

Use macOS arm64, Python 3.11 or newer, a compatible native Rust/Cargo toolchain and
Apple's installed C compiler/SDK tools. The original observation used Rust 1.97.1.
Compiler, OS, binary and prepared-path identities are recorded for every rebuild;
a different build or rerun must not inherit archived timing values. This tool uses
Mach timestamps, Apple `time -l` RSS reporting and a Darwin read interposer. It is
not a Linux or Windows replay script and does not install dependencies.

The adjacent evaluation directory must contain `public-quality-v1/` and
`lifecycle-inputs-v1/`. Their fixed files, source/model provenance, source snapshots,
licenses and notices remain part of the evaluation materials. See
`public-quality-v1/source-provenance.json`, `embedding-provenance.json` and
`judgment-provenance.json` for the real-document workload. The lifecycle input is
synthetic. Input checksums are fixed in `input-spec.json`; vectors and text are not
regenerated or changed. Only the ten original lifecycle groups are imported; the
separate replacement payload is excluded. The original groups remain separate
batches to fit both historical import limits.

Require at least 28 GiB free disk space before preparing or starting a replay.
The runner samples free space, logical owned-file sizes and live owned-process
RSS; its 20 GiB free reserve, 8 GiB logical-owned limit and 4 GiB process-group RSS
stop thresholds are observations, not hard quotas or proofs about intervals
between samples. Build caches and result files consume additional space.

## Prepare, build and check controls

Choose a new work directory outside this package and the evaluation input trees.
These steps do not import or query a database.
Preparation validates every original archive member, selected logical input and
historical source digest. It refuses existing destinations and does not clean up
partial work. Source snapshots contain only the Rust sources, package metadata,
licenses and explicit inline-test dependencies needed for these builds.

From this directory, using absolute work paths:

```sh
python3 -B -m unittest -v prepare_test.py recompute_test.py
python3 prepare.py --evaluation-dir .. --work-dir /absolute/path/new-replay-work
python3 /absolute/path/new-replay-work/build.py --controls-only
python3 /absolute/path/new-replay-work/build.py
```

The first build checks actual C and Rust file reads against the probe and runs
pure orchestration controls: process failure/timeout, bounded capture, sample
loss, result retention, clocks and disappearing temporary files. It does not run
CofferDB queries. The second build uses offline, locked Cargo sources and the
native target, then compiles the measurement adapters against each matching
library. Cargo, controls and adapters use the same resolved compiler executable; its
path and hash are recorded. Inherited compiler wrappers and Rust flags are
disabled for this build. Build and control output, compiler versions and binary checksums are
recorded beneath the work directory. A failed stage is preserved; preparing a
new work directory is required for another attempt. Nothing is installed or
changed in the default toolchain.

## Run the complete matrix

Use a new result directory name with `run-` followed by 32 lowercase hexadecimal
characters. The following is an example name, not an archived execution nonce:

```sh
python3 /absolute/path/new-replay-work/replay.py \
  --protocol /absolute/path/new-replay-work/protocol.json \
  --new-output /absolute/path/new-replay-work/run-0123456789abcdef0123456789abcdef
```

The prepared build is bound to that work path because the read-probe dynamic
library uses a local install name. To relocate, prepare and rebuild at the new
path; do not move an existing build and reuse its identity. The runner verifies
bound file hashes before execution and between steps. It imports four new
databases, executes the fixed serial plan, stops at the first failure and does
not retry or reuse a partial database. All original streams and incomplete
records remain available. Missing observations remain unavailable; they are not
reported as zero-time successes.

The plan contains 22 imports and 36 query workers: two main rounds plus a separate
read-diagnostic round, for three historical budget settings, both implementations,
and quality/scale workloads. The second main round reverses implementation order.
It retains 7,488 query samples: 3,456 primary quality, 1,728 quality diagnostics,
2,208 primary scale and 96 scale diagnostics. These are repetitions of 48
independent quality tasks; repetitions do not increase task coverage.

Query workers have a 180-second timeout, imports 300 seconds, and the overall
execution budget is 40 minutes. Each stdout/stderr stream is limited to 64 MiB.
Build time is separate. The runner does not evict OS caches. A fresh handle is
not a cold disk cache; earlier groups, collection statistics and hash checks may
warm caches. Record other host load and memory pressure alongside any new result.

## Interpret and preserve results

`report.json` contains process receipts, full sample status, quality metrics,
work counters and every planned comparison. The raw `.stdout` files retain full
returned IDs, ranks, scores and explanations; `.stderr` retains process RSS.
Quality latency uses both main rounds, 96 observations per version/mode/cache/
budget setting. Scale P50/P95 use the 40 unrounded measured observations per
version/mode/API scope; first queries, warmups and diagnostics are kept separately.
Nearest-rank quantiles are computed on the pooled observations, not by averaging
per-round percentiles or choosing the fastest round.

API execution, standard JSON rendering, process wall time, opening and evidence
publication have distinct scopes. RSS is a whole-worker-process peak, not a
per-query or scoring-only allocation. Diagnostic reads are userspace work on the
querying thread, not physical disk IO. Legacy counters with unavailable meanings
remain explicitly unmeasured; configured limits, actual branch counts, complete
union, fusion retention and returned rows must not be conflated.

There is no quality lower-bound or mandatory winning condition. Retain every
negative result and disclose shared-host/cache effects. Two full historical
versions differ beyond scoring; this replay cannot establish isolated causality,
current-release performance or a universal quality improvement. Rebuilt/rerun
results have their own identity and must not replace archived observations or
change the original experiment outcomes.
