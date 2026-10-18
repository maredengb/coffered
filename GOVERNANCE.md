# Governance and naming

CofferDB develops an open-source embedded retrieval engine. Public discussion of
bugs and proposed changes should make behavior, compatibility, evidence and user
impact clear. Maintainers review contributions for correctness, reproducibility,
maintainability and the documented scope before accepting them. A proposal or
open pull request does not by itself change the released product's guarantees.

Core Rust retrieval, CLI operations, JSONL import/export, explain and benchmark
sources are covered by the applicable [Apache-2.0 license](LICENSE), with bundled
upstream sources retaining their own notices as described in
[DEPENDENCIES.md](DEPENDENCIES.md). The open-source core is not presented as having
artificial restrictions that reserve its existing retrieval capabilities for a
paid service.

Product documentation and release notes should distinguish delivered behavior,
experimental work and future plans. Compatibility changes need explicit notes.
Performance and retrieval-quality statements need evidence for their stated
scope, including regressions and measurement limitations.

## Project name

CofferDB is the project's working name. This document makes no claim of trademark
registration, naming clearance or affiliation with another product or company.
The source license and use of the project name are distinct matters; see the
license's trademark provision. Avoid presenting modified or third-party products
as an official CofferDB release or implying unestablished endorsement.

Security-sensitive reports follow [SECURITY.md](SECURITY.md). The
[contribution guide](CONTRIBUTING.md) lists the information for a reproducible issue.
