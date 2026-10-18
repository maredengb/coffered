---
name: Bug report
about: Report a reproducible CofferDB problem using sanitized data
---

## Environment

CofferDB version or commit, OS/architecture, Rust version if built from source,
filesystem, resource profile, budget overrides and durability mode.

## Reproduction

Minimal commands or Rust code and a small, sanitized input. Describe whether the
problem occurs on a newly created database or an existing one.

## Expected and actual behavior

Include the relevant result, exit code and sanitized error/check output. For
ranking problems, distinguish an incorrect calculation from an application
relevance concern and identify the query, weights and supplied vector source.

## Privacy check

Do not attach confidential documents, vectors, credentials, private paths or a
complete private database. Follow SECURITY.md for a security-sensitive report;
ordinary issues are public.
