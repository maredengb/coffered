# Architecture overview

CofferDB is an embedded Rust library with a CLI. Applications provide documents,
chunks, metadata and optional embeddings. The engine does not generate embeddings
or contact a model service. It combines lexical and vector retrieval locally and
explains the scores of returned results.

This describes the current experimental implementation. It is not a declaration
of completed release acceptance, a stable file-format promise or a claim about
application relevance or comparative performance.

## Storage and visibility

A database has one authoritative `.coffer` file. Its critical record stream holds
collection changes, documents, chunks, embeddings, tombstones and commits. A batch
becomes visible only after a complete, valid commit; recovery checks record and
ordered batch integrity. An incomplete or invalid tail does not make its pending
records committed.

A rebuildable suffix can hold a full text/vector index checkpoint for a particular
visible snapshot. It is separate from the critical stream. Required structural,
record/segment, referenced-index and snapshot-coverage checks must pass before
a persistent index is used. A mismatch limited to the supplementary aggregate
checkpoint-payload checksum can remain a non-blocking diagnostic after those
checks pass. An unavailable
checkpoint can require a bounded transient rebuild or an explicit error; it does
not authorize returning incomplete search results. New critical writes discard
the old suffix before appending. See [file format](file-format.md).

The runtime maintains catalog and payload locations, reads stored text and vectors
as needed, and caches eligible index data within configured budgets. A single
writer owns the file lock. Readers can retain an older file generation across
compaction. Their lifetime affects physical disk reclamation; copying a live file
is not a backup protocol. See [backup and data handling](backup.md).

## Query path

1. Validate text, query vectors, model/dimension compatibility, filters, weights
   and resource settings.
2. Retrieve BM25 and exact-vector candidates for the supplied inputs. The fixed
   `default_en` tokenizer handles Unicode punctuation and code identifiers; it is
   not a Chinese word segmenter.
3. Apply metadata eligibility and candidate limits. Complete branch scores for
   the retained candidate union, normalize the components and apply effective
   weights and any explicitly enabled metadata boost.
4. Rank and return the requested top results with score components, effective
   configuration and candidate/index-source information.

The vector path is an exact scan, not an approximate-nearest-neighbor index.
Candidate limits do not remove the scan, and hybrid score completion can perform
additional distance work. Smaller windows alone do not prove less CPU use.
Missing text/vector inputs and disabled metadata boost affect effective weights;
the [weight table](../README.md#search-inputs-and-weights) documents the defaults.

Explain describes returned results and their ranking. Weighted score shares are
neither Recall contributions nor relevance probabilities. Application Recall/MRR
requires separately fixed queries and relevance judgments. External vector quality,
chunking and application configuration affect relevance; engine scoring, filtering,
storage and configuration execution remain engine responsibilities.

## Resource and failure boundaries

`balanced` is the starting preset; `expanded` and `constrained` alter default
candidate windows and cache settings. Explicit cache overrides take precedence.
Each effective window is `max(preset default, top_k)`; a larger `top_k` does not
increase the budget. See [presets and limits](../README.md#resource-presets-and-limits).

Cache/workspace budgets cover the engine's documented accounting domains. They
are not a process RSS cap and do not include application-owned memory or all
other processes. Temporary work and old file generations require disk space.
Budget refusal, an unavailable index, invalid input and real I/O failure have
different error contracts; an error is not a partial-success search.

`check` diagnoses storage/index integrity without repairing the source file.
Compaction rewrites live state and publishes a replacement file. Deletion is
logical, and neither compaction nor replacement guarantees secure erasure.
The current operational target is local storage with single-writer coordination;
there is no distributed transaction or network-filesystem consistency promise.

## Source map

| Area | Implementation |
| --- | --- |
| Public Rust surface | [src/lib.rs](../src/lib.rs) |
| CLI and output envelopes | [src/cli.rs](../src/cli.rs) |
| Runtime and read/write operations | [src/db.rs](../src/db.rs) |
| Catalog and mutations | [src/catalog.rs](../src/catalog.rs) |
| Record/checkpoint storage | [src/storage.rs](../src/storage.rs) |
| Typed payloads and recovery | [codec](../src/storage/codec.rs), [recovery](../src/storage/recovery.rs) |
| Lexical, vector and hybrid retrieval | [text](../src/index/text.rs), [vector](../src/index/vector.rs), [hybrid](../src/index/hybrid.rs) |
| Import/export | [src/import.rs](../src/import.rs) |
| Resource configuration | [src/profile.rs](../src/profile.rs) |
| Evaluation runner | [src/benchmark/runner.rs](../src/benchmark/runner.rs) |

The [examples](../README.md#rust-library-and-examples) demonstrate embedding,
Markdown/code retrieval and ranking explanations. The [benchmark guide](../benchmarks/README.md)
separates functional replay from representative quality and cost evidence.
