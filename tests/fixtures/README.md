# Legacy Fixture

legacy-2d9eb64.hex is a byte-for-byte hex encoding of a real database produced by
unmodified cofferdb code from commit 2d9eb64 (git archive, cargo build, init, import
examples/code/import_manifest.json). It has a config-less collection and schema-1
text suffix using the previous experimental tokenizer. It contains only synthetic
public fixture text and relative source paths. Do not regenerate from current
storage.rs: that would stop testing an independently produced old format.

The integration test decodes it, opens without changing bytes, validates default
resolution and raw source text, reconstructs queries, then explicitly rebuilds the
disposable suffix and checks the same behavior after reopen.
