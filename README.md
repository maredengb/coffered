# CofferDB

A single-file, embedded Rust retrieval database for explainable local search and
RAG applications. Use the Rust library in your application or the `coffer` CLI to
import documents, search with keywords and externally supplied vectors, and
inspect how each result was ranked.

**Status: experimental.** Check the [releases page](https://github.com/maredengb/coffered/releases)
for published versions and downloadable artifacts. File-format compatibility across
future releases and production durability are not guaranteed. Documented commit,
recovery and integrity contracts still apply. Keep original inputs and export backups. No certified retrieval-quality
or comparative performance claim is made.

## What it does

- Stores collections, documents, chunks, metadata and embeddings in a `.coffer`
  file, with manifest-based JSONL import and export.
- Combines BM25 keyword search, exact vector search and metadata filters using
  weighted hybrid scoring.
- Explains returned results with component scores, effective weights and candidate
  sources.
- Provides read-only snapshots, integrity checks, compaction and configurable
  resource budgets.
- Runs core database operations locally, without telemetry, model downloads or
  an external service. Embeddings are supplied by the caller.

## Quickstart

Install Rust **1.80 or later** and a native build toolchain. The commands below use
a macOS or Linux shell and run from this directory, beside `Cargo.toml`. They work
from a source checkout or an unpacked Cargo source package.

Create a fresh temporary directory so the example does not reuse an existing
database:

```sh
cargo build --locked
coffer_demo_dir="$(mktemp -d)"
./target/debug/coffer init "$coffer_demo_dir/notes.coffer"
./target/debug/coffer import "$coffer_demo_dir/notes.coffer" --manifest examples/data/markdown/import_manifest.json --json
./target/debug/coffer search "$coffer_demo_dir/notes.coffer" --collection notes 'write ahead' --json
```

The search returns the bundled `wal#commit` chunk about write-ahead logging. This
small fixture demonstrates the import and search flow; its hand-authored vectors
are not a semantic-quality benchmark. The database remains in `coffer_demo_dir`
for inspection.

`--json` returns a versioned result envelope suitable for programs. Omit it for
human-readable CLI output. Inspect the available commands and budget presets with:

```sh
./target/debug/coffer help
./target/debug/coffer profile-help
```

## Rust library and examples

The library exposes `Database`, `CollectionConfig`, `SearchBuilder` and related
types. Search builders accept text, a query embedding, filters, weights and a
resource profile. See the complete [embedded Rust example](examples/embedded.rs)
and [public exports](src/lib.rs).

Run these from the same directory and shell as the quickstart:

```sh
cargo run --locked --example embedded
cargo run --locked --example explain_troubleshooting -- "$coffer_demo_dir/explain.coffer"
cargo run --locked --example real_corpus_demo -- "$coffer_demo_dir/real-corpus.coffer"
```

| Example | What to inspect |
| --- | --- |
| [Embedded Rust](examples/embedded.rs) | Create a collection, insert text and vectors, and run hybrid search. |
| [Explain and ranking](examples/explain_troubleshooting.rs) | Compare weights and resource profiles on small synthetic fixtures. |
| [Source-backed demos](examples/real_corpus_demo.rs) | Markdown retrieval, Rust code filtering and ranking explanations using bundled source snapshots and precomputed model vectors. |

The examples need no model download or embedding service. The source-backed
fixtures use pinned Rust by Example sources with matching BGE-M3 vectors and
include their [provenance](examples/data/real_corpus/provenance.json). Source
snapshots and their MIT/Apache-2.0 notices are bundled. Rust snippets are retrieval
inputs; some intentionally illustrate compile errors or panics. These small
examples do not measure representative application relevance.
Example output demonstrates functionality, not general retrieval quality or
performance. Generate local API documentation with `cargo doc --no-deps --open`.

## Search inputs and weights

`search` accepts text, a query embedding, or both. At least one nonblank text input
or valid query embedding is required. Text that produces no matching terms still
counts as supplied text. `vector-search` is a vector-only shortcut; use `search`
when you need the full weights and metadata-boost options.

Text search uses the fixed `default_en` tokenizer: stop-word filtering, Unicode
punctuation boundaries and snake/camel/Pascal identifier splitting. It does not
provide Chinese word segmentation. Supply corpus and query embeddings from a
compatible model with the configured dimensions.

Without a weights override, the requested weights are:

| Inputs | Metadata boost | BM25 | Vector | Metadata |
| --- | --- | ---: | ---: | ---: |
| Text and vector | Disabled | 0.50 | 0.50 | 0.00 |
| Text only | Disabled | 1.00 | 0.00 | 0.00 |
| Vector only | Disabled | 0.00 | 1.00 | 0.00 |
| Text and vector | Enabled | 0.45 | 0.50 | 0.05 |
| Text only | Enabled | 0.95 | 0.00 | 0.05 |
| Vector only | Enabled | 0.00 | 0.95 | 0.05 |

Metadata boost is disabled unless explicitly configured. Override weights with
Rust `.weights(...)` or the CLI `--weights` option, which accepts inline JSON such
as `'{"bm25":0.3,"vector":0.7,"metadata":0}'`. Missing inputs and disabled boost
cause the remaining weights to be renormalized; no available positive weight is
an input error. Explain records `weights.requested`, `weights.effective`,
`weights.source` and `renormalization_reason`.

These defaults are reproducible starting points, not values optimized for every
application. Hybrid search does not always outperform keyword or vector search
alone. CofferDB is responsible for correct input handling, storage, filtering,
scoring and configuration execution. External models, chunking and application
data affect semantic relevance; accepting vectors does not guarantee application
recall. Default and recommended configurations still need evaluation and disclosure
of their limitations.

Explain describes score components and ranking. A weighted score share is not a
recall contribution, relevance probability or causal attribution. Ordinary search
has no complete relevance judgments and does not report application recall.

## Memory metadata

Memory entries use ordinary documents and chunks. The bundled
[metadata template](examples/data/memory/metadata-template.json) provides `type`,
`source`, `scope`, `project_id`, `session_id`, `importance`, `created_at`,
`last_accessed_at`, `expires_at` and `supersedes`. Store these on chunk metadata
and filter with keys such as `chunk.type` or `chunk.project_id`.

Dates are strings. These fields do not enable automatic expiration, forgetting,
summarization or access control. Recency and importance do not change default
ranking; an application can explicitly configure metadata boosts.

## Resource presets and limits

Start with `balanced`. `expanded` increases the default candidate windows and
cache budget; `constrained` reduces them. Preset names describe resource settings
and do not guarantee better relevance, speed or energy use. Explicit runtime
cache-budget overrides take precedence over preset defaults, including per-query
preset selection. Background compaction is off by default.

| Rust `ResourceProfile` | CLI / configuration / JSON | Default candidate windows (BM25, vector, fusion) | Default cache budget |
| --- | --- | --- | --- |
| `Expanded` | `expanded` | 500 each | 512 MB |
| `Balanced` | `balanced` | 200 each | 256 MB |
| `Constrained` | `constrained` | 50 each | 64 MB |

String inputs must match the table exactly in lowercase. CLI execution rejects
invalid `--profile` values with `cli_usage` and exit
code 1. Select the benchmark's resource preset through `--profile`; dataset
manifests and query records provide no resource-preset selector. Result JSON
contains the formal preset values, and its internal contract validation rejects
invalid values with `invalid_jsonl_schema`; that validation is not a public result
JSON readback command or an independent CLI exit-code promise. `--profile help`
is a help operation, not an executable preset.

`top_k` accepts integers from 1 through `min(usize::MAX, 9007199254740991)` on the
executing target. Each effective candidate window is the larger of its preset
default and `top_k`; increasing `top_k` does not increase resource budgets.
Insufficient resources can produce an explicit refusal rather than partial
results.

Vector retrieval uses an exact scan. Candidate windows do not eliminate that
scan, and hybrid score completion can add distance calculations. Cache and
workspace budgets are not limits on total process RSS. Old readers can retain
old file space after compaction; other processes and application-owned memory
are outside the database's local resource accounting.

## Evaluation and current scope

The CLI includes `benchmark`, `check`, `compact` and `export` commands. Integrity
checks diagnose stored-data and index issues; they do not evaluate semantic
relevance. Quality evaluation requires fixed inputs, queries and relevance
judgments, with keyword, vector and hybrid results reported separately. The
SQLite FTS5 + sqlite-vec benchmark comparison uses separately installed tools;
they are not core runtime dependencies.

Source checkouts include representative quality and cost observations, per-query
results, failures and replay materials in the [evaluation archive](benchmarks/evaluation-v1/README.md).
The archive is excluded from Cargo packages; see the [benchmark guide](benchmarks/README.md)
for its scope and replay instructions.
These observations do not establish release readiness or certify a
model/configuration. CofferDB makes no fixed application Recall/MRR guarantee and no
claim that hybrid always wins. Quantitative claims require reproducible evidence
including regressions and limitations.

The current focus is a local Rust library and CLI. Python bindings, hosted
embedding providers, approximate vector indexes and server-scale deployment are
outside this v0.1 entry point. CofferDB does not claim to replace SQLite or
PostgreSQL. Search, explain and export output may contain your source text and
metadata; control where those outputs are shared.

## Documentation

- [Architecture overview](docs/architecture.md) and [file-format boundaries](docs/file-format.md).
- [Backup, export and data handling](docs/backup.md), including privacy and deletion limits.
- [Explain JSON and ranking troubleshooting](docs/explain.md).
- [Benchmark reproduction](benchmarks/README.md) and [SQLite reference notes](benchmarks/sqlite-baseline.md).
- [Contributing](CONTRIBUTING.md), [security policy](SECURITY.md) and [governance/naming](GOVERNANCE.md).

## License

[Apache-2.0](LICENSE). See the [dependency and license summary](DEPENDENCIES.md)
for the distinction between source dependencies and compiled runtime notices.
