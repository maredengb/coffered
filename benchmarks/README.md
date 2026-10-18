# Benchmark reproduction

This directory provides a small, self-contained functional benchmark workflow.
The [evaluation materials](evaluation-v1/README.md) contain a separate representative
quality study, historical resource observations, complete results and replay
limits. Running the tiny fixture below does not establish release acceptance,
semantic quality or comparative performance. The [README](../README.md) describes
the engine's scope.

The evaluation archive is available in the source repository and is excluded from
the crate source package. It is not required to build or use the database.

## Tiny functional replay

Requirements: Rust 1.80 or later, native build tools and Node.js 18 or later for
the dataset generator. Node is not a database runtime dependency. Run from the
product root, beside `Cargo.toml`, in a macOS or Linux shell:

```sh
cargo build --release --locked
coffer_bench_dir="$(mktemp -d)"
node tools/prepare-benchmark.mjs "$coffer_bench_dir/input"
for coffer_profile in expanded balanced constrained; do
  ./target/release/coffer benchmark "$coffer_bench_dir/$coffer_profile.coffer" \
    --dataset "$coffer_bench_dir/input" --profile "$coffer_profile" \
    --durability commit --output "$coffer_bench_dir/$coffer_profile.json" --json
done
```

The generator creates ten neutral topic chunks, three hand-authored 3D query
vectors and explicit synthetic relevance judgments. It writes hashes for all five
JSONL files and refuses to reuse an existing output directory. No model download
or external embedding service is involved. These vectors and labels are designed
for plumbing checks, not independent semantic-quality evaluation.

Each preset gets a separate database and result file. Keep the input directory,
manifest, full result JSON, journal/sidecars and command/environment record. A
successful process exit does not mean that every optional external baseline ran.
Inspect its status and all failures before drawing conclusions.

The accepted preset strings are exactly `expanded`, `balanced` and `constrained`;
start with `balanced`. Presets set resource defaults rather than quality rankings.
Effective candidate windows are `max(preset default, top_k)`, and larger `top_k`
does not increase the budget. [Preset documentation](../README.md#resource-presets-and-limits)
describes exact scans, cache overrides and the distinction between budgets and RSS.

## SQLite comparison

Select a trusted SQLite CLI with FTS5 and extension-loading support and a compatible
local sqlite-vec library. Neither tool is downloaded or installed by the runner.
Use an absolute extension path appropriate to the platform, usually a `.so` on
Linux or `.dylib` on macOS. From the same shell as above, replace the two example
paths with your actual installed files:

```sh
COFFERDB_SQLITE3=/path/to/sqlite3 \
COFFERDB_SQLITE_VEC_EXTENSION=/path/to/vec0.dylib \
  ./target/release/coffer benchmark "$coffer_bench_dir/sqlite-reference.coffer" \
    --dataset "$coffer_bench_dir/input" --profile balanced --durability commit \
    --require-baseline --output "$coffer_bench_dir/sqlite-reference.json" --json
```

`--require-baseline` makes an unavailable baseline a hard failure. Record the
actual SQLite/extension versions and result status. See the
[SQLite baseline notes](sqlite-baseline.md) for scoring, scanning and timing limits.

## Designing a representative evaluation

Fix the protocol before running it: corpus/chunking, external model and vector
assets, queries, relevance judgments, supported task groups, filters, weights,
presets, budgets, hardware, toolchain, software identities and all input hashes.
Model names alone do not identify an embedding preparation pipeline. Keep tuning
data separate from evidence used to support a quality or superiority claim.

Evaluate the default hybrid path without a weights override and include keyword,
exact-vector and SQLite reference results. Include all presets, task groups,
failures, ties and regressions. Publish per-query rankings alongside aggregate
metrics. Specify Recall's `k` and denominator, MRR truncation, averaging, treatment
of unjudged/no-positive queries and floating-point tolerances. An exact neighbor
is not automatically a relevant answer. There is no fixed application Recall/MRR
guarantee, and hybrid need not outperform either single branch.

Freeze timing boundaries, warmups, repetitions, cache handling, percentile method,
failure/timeout handling and available resource measurements. The built-in `cold`
label means a reopened database handle, not eviction of the OS page cache.
SQLite subprocess startup and the embedded engine have different timing
boundaries. Do not infer a matched performance win from those numbers.

Keep input manifest and data hashes. Distinguish implementation, configuration,
protocol and report versions. Reusing unchanged evidence needs an applicability
explanation with its original identity and a statement that it was not rerun.
Changed vectors, chunking, labels, weights or evaluation scope need new affected
comparisons; renaming a report does not turn a historical failure into a pass.

## Interpreting results

Engine calculation correctness, application relevance and execution completeness
are separate conclusions. Verify tokenization, BM25, vector distances, filtering,
candidate membership, normalization/fusion and explain with an independent
calculation. Do not copy the engine's ranking to create expected answers.

Cross-engine raw scores are not comparable. Compare eligible relevance judgments,
ranked IDs and explicitly matched metrics instead. Candidate counts, fusion-retained
counts and returned counts describe different stages; vector scan/distance work
is not determined by the requested top-k alone. Treat cache/workspace budgets
separately from process RSS and application memory.

Record unknown physical I/O, unavailable peaks or unmeasured CPU attribution as
unknown, not zero. The built-in journal has no internal query/run deadline; an
externally interrupted run can leave an incomplete journal. Preserve incomplete,
failed and unexecuted attempts. A tiny fixture or a faster selected round cannot
establish a general cost, energy, quality or durability claim.

Result files can contain query text, paths, metadata and rankings. Inspect and
sanitize them before publication. See [data handling](../docs/backup.md).
