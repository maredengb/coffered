# SQLite FTS5 + sqlite-vec baseline

The optional reference runner uses a caller-selected SQLite executable and local
sqlite-vec extension. Set `COFFERDB_SQLITE3` and
`COFFERDB_SQLITE_VEC_EXTENSION` explicitly. Loading a native extension executes
code: use trusted local files, never a library named by imported data.

FTS5 supplies lexical results. `vec_f32` converts JSON float arrays to vector
blobs; `vec_distance_cosine` computes cosine distances inside SQL. Rust handles
metadata eligibility and result normalization/fusion. This path performs exact
scanning; it does not use a `vec0` virtual table or an approximate-nearest-neighbor
index. Zero-norm cosine
vectors are treated as zero similarity by the runner's explicit handling.
See the [implementation](../src/benchmark/runner/sqlite.rs).

The result records actual versions, dependency readiness, candidate policy and
engine-local score meaning. SQLite's FTS5 tokenizer differs from `default_en`;
the baseline's tokenizer/configuration records describe that distinction. Do not
compare its raw lexical scores directly with normalized CofferDB scores.

`cold`/`warm` do not mean OS cache eviction. SQLite process startup is included
in its measurement while the embedded CofferDB timing excludes process startup.
Process memory observations do not establish an isolated SQLite subprocess peak.
These boundaries need an explicitly matched protocol before a performance claim.
Missing dependencies are reported as not run; `--require-baseline` makes them a
hard failure. Never present not-run or incomplete output as a successful comparison.

The [benchmark guide](README.md) contains a tiny replay and the requirements for
representative, reproducible comparisons. External tools are not bundled core
runtime dependencies; see [dependency/license information](../DEPENDENCIES.md).
