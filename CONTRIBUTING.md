# Contributing

CofferDB is an experimental embedded Rust library and CLI. Start with the
[README](README.md), [architecture overview](docs/architecture.md) and runnable
[examples](README.md#rust-library-and-examples). These files describe the public
product without requiring separate planning documents.

## Reporting a problem

When reporting an issue, include the
version or commit, OS/architecture, Rust version, filesystem, relevant settings,
minimal commands and expected versus actual results. Sanitize documents, vectors,
paths, metadata and diagnostics. Follow the [security policy](SECURITY.md) for
security-sensitive reports; do not attach a private database to a public issue.

## Building and checking a change

Run these from the directory containing `Cargo.toml`. Building requires Rust 1.80
or later and the platform's native build tools. Formatting and Clippy require
their corresponding Rust components.

```sh
cargo build --locked
cargo test --locked --all-targets
cargo test --locked --doc
cargo fmt --all -- --check
cargo clippy --locked --all-targets -- -D warnings
```

The core crate declares no third-party Cargo dependencies. SQLite and sqlite-vec
are optional, externally selected benchmark tools; they are not needed for a core
build. See the [benchmark guide](benchmarks/README.md) for a small functional replay
and the limitations of comparative measurements.

For a bug fix, provide a focused regression that fails for the reported behavior
and verifies the correction through an appropriate public operation. For API,
CLI or data-format changes, describe compatibility and include relevant tests and
user-facing documentation. Record what you actually ran, the environment and
any remaining limitations. Test counts alone do not establish coverage of a claim.

Keep examples self-contained and use redistributable, non-confidential inputs.
When changing vector-bearing text, update the corresponding vectors and their
provenance together. Preserve upstream source notices and distinguish synthetic
functional fixtures from representative application-quality evaluations.

Changes to stored formats need explicit version detection and compatibility tests;
export/import is a public-data interchange operation, not an assumption that all
internal IDs or binary bytes remain unchanged. Performance or relevance claims
need reproducible inputs, settings, raw results, baselines and limitations.

See [governance and naming](GOVERNANCE.md) and the [license](LICENSE) for the
project's public contribution and naming principles.
