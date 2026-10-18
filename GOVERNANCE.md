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

## Contribution decisions

The repository owner, [maredengb](https://github.com/maredengb), makes the final
acceptance decision and manually merges accepted changes into official branches.
Contributors may propose changes through pull requests and participate in review.
Automated tools prepare and check changes; they do not perform merges.

Merge authority does not confer ownership of contributors' copyrights or rights
beyond applicable licenses. The public distribution remains [Apache-2.0](LICENSE).
The [CLA](CLA.md), once activated, would grant additional commercial and
other-license sublicensing rights while contributors retain copyright. The CLA
is not yet in effect, and online confirmations are not enabled. See the
[contribution guide](CONTRIBUTING.md) for the process that will apply after activation.

Existing Apache-2.0 grants cannot be withdrawn merely by changing a future
release's license. A different future distribution also requires appropriate
rights in all included material, including earlier and third-party contributions.
An automated CLA result is evidence of an account confirmation; it is not a
legal finding of authorship, authority or complete rights clearance.

Only the owner manually merges into official branches. Repository protections
must be verified separately before being described as enforced. A written policy
alone, a code-owner file, or a passing check does not restrict platform permissions.
Automation does not receive authority to merge or bypass branch protections.
