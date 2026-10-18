# Dependency And License Summary

The crate declares no third-party Cargo dependencies. Core source follows the
existing repository Apache-2.0 LICENSE; Cargo metadata is synchronized to it.
The packaged LICENSE is byte-identical to the repository license text.

Compiled executables also use the Rust standard library and platform runtime.
The Rust project uses MIT/Apache-2.0 terms; binary release preparation must include
the applicable notices for the selected toolchain/system redistribution policy.
This file is a source dependency summary, not completed binary license clearance.

SQLite and sqlite-vec are externally selected optional benchmark dependencies,
not bundled core runtime crates or redistributed extensions. Record actual
versions, provenance and license documents in any benchmark/release that bundles
them. Never auto-load an extension named by imported data.

## Bundled example sources

`examples/data/real_corpus/source_snapshot/rust-by-example/` contains unchanged
Markdown files from rust-lang/rust-by-example at commit
`dcc14f912bf8790792a2df082eddc82ab2539229`. These source files and the extracted
Rust snippets retain their upstream **MIT OR Apache-2.0** license, rather than
being relicensed by the crate's package metadata. Both upstream notices are
bundled alongside the snapshots. The example's `provenance.json` records source
URLs, hashes and the external BGE-M3 vector preparation. No model weights, Python
runtime or embedding service are included or needed to run the example.
