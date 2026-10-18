# Security policy

CofferDB is experimental pre-release software. No stable supported-version window
or security-response SLA is promised at this stage. It is a local retrieval engine,
not an encrypted store, authenticated service or multi-tenant security boundary.

## Reporting a vulnerability

Use the repository's private GitHub vulnerability-reporting/advisory channel when
available. If no private channel is available, open a public issue requesting a
private contact without disclosing the vulnerability, exploit or confidential
data. Do not assume an ordinary issue or pull request is private.

A useful private report identifies the affected version/commit, platform and
filesystem, minimal reproduction, expected versus actual behavior and likely
impact. Share only the minimum necessary sanitized data. Do not publish private
text, embeddings, paths, metadata, credentials or complete databases in an issue.

## Data and execution boundaries

- Core database operations do not upload data, download models or collect
  telemetry. Embeddings are supplied by the caller.
- Files, exports and diagnostics can contain application data. Filesystem access,
  backup handling and application permissions remain relevant; read-only access
  does not redact or encrypt stored values.
- Delete is logical. Compaction does not promise secure device erasure or removal
  from old readers, backups, snapshots or device history.
- An optional SQLite benchmark loads a caller-selected sqlite-vec extension into
  a subprocess. Loading an extension executes native code: select a trusted local
  executable and extension, never a library supplied by imported data.
- A successful integrity check does not establish that an arbitrary input is
  harmless, that an application is isolated, or that relevance is acceptable.

See [backup and data handling](docs/backup.md), the [format overview](docs/file-format.md)
and the [dependency/license summary](DEPENDENCIES.md) for operational boundaries.
