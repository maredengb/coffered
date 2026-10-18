# Reading explain output and troubleshooting ranking

The `search` and `vector-search` JSON results include an `explain` object on each
returned result. Use `--json` to inspect it; there is no separate `explain` CLI
command. Rust callers receive the same information through search-result JSON.
This guide documents the existing `0.1` shape and how to read it, without adding
another input or a separate machine-schema authority.

## A runnable hybrid example

Run from the product directory, beside `Cargo.toml`, in a macOS or Linux shell:

```sh
cargo build --locked
coffer_explain_dir="$(mktemp -d)"
./target/debug/coffer init "$coffer_explain_dir/notes.coffer"
./target/debug/coffer import "$coffer_explain_dir/notes.coffer" --manifest examples/data/markdown/import_manifest.json --json
printf '%s\n' '{"model_name":"synthetic-demo-2d-v1","dim":2,"vector":[1,0]}' > "$coffer_explain_dir/query.json"
./target/debug/coffer search "$coffer_explain_dir/notes.coffer" --collection notes 'write ahead' --embedding "$coffer_explain_dir/query.json" --json
```

`--embedding` takes a JSON file path. The object contains a nonblank `model_name`,
an integer `dim` and a finite vector of that length; it must match the collection's
model and dimension. This fixture uses hand-authored 2D vectors for functionality,
not a real-model quality claim. The first result is `wal#commit`; the selected
fields below come from that result's `explain` object. The full output has additional
fields listed in the table below.

```json
{
  "schema_version": "0.1",
  "chunk_external_id": "wal#commit",
  "final_score": 1,
  "scores": {
    "bm25_raw": 1.4076219666755814,
    "bm25_norm": 1,
    "vector_raw": 1,
    "vector_norm": 1,
    "metadata_boost": 0,
    "final_score": 1
  },
  "weights": {
    "bm25": 0.5,
    "vector": 0.5,
    "metadata": 0,
    "requested": {"bm25": 0.5, "vector": 0.5, "metadata": 0},
    "effective": {"bm25": 0.5, "vector": 0.5, "metadata": 0},
    "source": "default_no_boost",
    "renormalization_reason": null
  },
  "fusion": "weighted_sum",
  "candidate_sources": ["bm25", "vector"],
  "resource_profile": "balanced",
  "candidate_limits": {"bm25": 200, "vector": 200, "fusion": 200},
  "index_source": "persistent_checkpoint"
}
```

## JSON shape and field groups

The CLI envelope has `schema_version`, `command`, `ok`, `warnings`, `duration_ms`
and `result`. A failed operation includes `error`; it is not a successful partial
ranking. For a successful search, `result.results` is the ranked array. Each item
has rank/identity, scores, snippet/metadata and `explain`. Query-level context is in
`result.query_context`; actual candidate counts and execution statistics are also
reported at result level.

| Explain fields | Shape and meaning |
| --- | --- |
| `schema_version` | String `0.1` for this output contract. |
| `chunk_id`, `chunk_external_id`, `document_id`, `document_external_id`, `collection` | Internal/generated and caller-facing identities. Use external IDs to compare separate imports. |
| `final_score`, `scores` | Numbers for final score and normalized components; `scores.bm25_raw`, `vector_raw` and `vector_norm` can be null when the corresponding raw component is unavailable. |
| `weights` | Numeric `bm25`/`vector`/`metadata` effective weights, requested/effective objects, source string and nullable renormalization reason. |
| `normalization` | Candidate-set scope/count, BM25 min/max and vector method; `bm25_max` can be null. |
| `matched_terms`, `bm25`, `matched_term_details` | Matched-token strings, BM25 parameters and term statistics/contributions; empty arrays are possible. |
| `tokenizer` | Tokenizer name/version/profile/language hint and configuration. |
| `fusion`, `candidate_sources` | Weighted-sum fusion and an array containing `bm25` and/or `vector` for branch candidate membership. Membership is distinct from later score completion. |
| `resource_profile`, `candidate_limits` | Effective preset name and numeric BM25/vector/fusion window limits; these are not observed candidate counts. |
| `execution_stats` | Vector scan count, distance operations and fusion candidate count. |
| `index_source`, `index_rebuild_reason`, `transient_rebuild` | Persistent/transient source, nullable reason and rebuild observations; unused or unmeasured fields can be null. |
| `distance_metric`, `vector_raw_kind`, `normalization_method` | Metric and how to interpret raw and normalized vector values. A raw value is not always a cosine similarity. |
| `filter` | Expression, eligibility flag and reasons for this returned result. |
| `metadata_boost_context` | Whether boost was present/enabled, configuration hash and configured/matched rule counts. |
| `source`, `chunk_boundary` | Source URI/document path and optional chunk offsets. Null boundaries do not manufacture line provenance. |

See the [explain builder](../src/query.rs) and [CLI](../src/cli.rs) for the
implementation. The [Rust example](../examples/explain_troubleshooting.rs) compares
lexical/vector weight changes and presets on the same small inputs.

## Reading an unexpected ranking

1. Confirm collection, text and query-vector identity. A model name is not proof
   that corpus and query vectors were prepared consistently. Check source spans
   and chunk text before interpreting semantic relevance.
2. Read `query_context` and `weights`: supplied inputs, default/override source,
   effective values and renormalization. Metadata boost is off unless explicitly
   configured; merely storing importance or dates does not enable it.
3. Check filter expression and eligibility, then actual candidate counts, branch
   membership and effective limits. The returned list is not the full collection
   or the whole scored candidate union.
4. Read BM25 matches/raw contributions and vector metric/normalization before the
   weighted final score. Scores from different engines or differently normalized
   candidate sets are not directly comparable.
5. Inspect index source, rebuild reason and execution work. A small top-k/window
   does not mean the exact-vector scan was skipped. A resource error is a refusal,
   not evidence that omitted results had low relevance.
6. Compare configurations on the same data, query/vector, filter and top-k, retaining
   both rankings and effective settings. Record added, lost and shared relevant
   hits only when independent relevance judgments are available. A single changed
   result is not general evidence of quality or performance improvement.

Explain shows why returned chunks entered candidates and how they were scored.
Ordinary search does not provide a complete diagnosis of every missing chunk,
application Recall, causal contribution percentages or relevance probabilities.
If the question is application quality, use a separately fixed evaluation as
described in the [benchmark guide](../benchmarks/README.md).

Output can include source paths, query text and metadata. Sanitize it before
sharing; see [data handling](backup.md) and the [security policy](../SECURITY.md).
