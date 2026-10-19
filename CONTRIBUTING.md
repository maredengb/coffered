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

## Contribution license and online CLA

The current public distribution remains [Apache-2.0](LICENSE). Contributors retain
their copyright. By accepting the [Contributor License Agreement](CLA.md), a
contributor also grants the Recipient commercial and other-license sublicensing
rights, including closed-source distribution. These additional rights come from the agreement, not
from ownership of the repository or the act of merging a pull request. Existing
Apache-2.0 permissions and third-party license obligations remain in force.

The CLA does not grant additional rights in older unrelated contributions.

1. Read the agreement and [confirmation-data notice](CLA-PRIVACY.md).
   Review the highlighted commercial licensing, patent, future-contribution and
   succession terms, and ask any questions before agreeing.
2. For your first covered submission, personally post the published confirmation
   text using your GitHub account in the designated thread. It identifies the
   agreement version and hash, notice hash, and your initial PR and commit. No
   separate legal-name, contact-information or identity-document form is required.
3. For subsequent contributions, the checker reuses your existing confirmation
   for that agreement version. It does not sign for you. A new agreement needs
   new consent.

Each relevant contributor must be covered. Identify third-party sources, preserve
license and attribution notices, and submit only material you have authority to
license. An employer-owned contribution needs appropriate rights-holder
permission; a GitHub confirmation alone cannot supply it. Unknown authorship or
rights questions require review even if an automated check succeeds.

The confirmation comment and GitHub attribution are public. Do not include
identity documents, private contact details or handwritten signatures. If a
public confirmation is unsuitable, request a confidential contact route without
posting private material; the affected submission waits for an agreed process.
The confirmation process adds no DCO sign-off requirement.

The owner makes the final acceptance decision and manually merges accepted
changes. Automated tools only prepare and check them; a CLA check neither merges
code nor guarantees acceptance or copyright ownership.

See the [confirmation checker](.github/cla/README.md) for text-generation commands
and technical details.
