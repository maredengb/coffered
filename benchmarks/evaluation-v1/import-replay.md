# Historical import and majority-filter diagnostics

[`historical-import-v2`](historical-import-v2/artifact-lineage.json) retains the
original protocol, report, eighteen complete event streams, native resource
records and two original Rust probe sources. There are sixteen no-vector import
trials: two input sizes, two checkpoint paths, bulk or ten-part layout, and two
rounds. Two further trials exercise a 95%-retained metadata filter on the separate
100k-vector scale input. These are historical diagnostics under their recorded
source and environment, not current binary measurements or semantic-quality
acceptance. All original numeric observations and failure qualifications remain.

Import timing starts before the first manifest read and ends after all required
query checks. Bulk trials verify two queries, ten-part trials twenty. Their elapsed
times therefore do not isolate a causal batching speed ratio. Commit/checkpoint
diagnostics are not additional disjoint terms to add to the continuous total.
Two samples per import parameter group do not establish an import P95.

The majority-filter trials have one first query, two warmups and twenty measured
queries per round. They retain 95,000 eligible vectors but scan 100,000 records;
the reduction in distance calculations is not elimination of the scan. Only the
forty same-context measured samples support the archived pooled percentiles.
The shared host's OS caches were not evicted.

Native resource records describe whole process lifetimes, including assertions
and output. They do not establish per-phase peak RSS. The original protocol marks
CPU, physical I/O and temporary peak unavailable; retaining the raw native time
records does not silently change those declarations or create a new measurement.
External-reference aliases in derived metadata are archive locators, not paths
to install. All original hashes and every replacement are in the lineage file.

## Reconstruct the text inputs

From this evaluation directory, with Python 3.11 or later:

```sh
python3 prepare-import-input.py import-input
```

The output directory must not exist. This writes all 132 original synthetic
input files, including independent round-two copies, and checks every original
hash from the archived protocol. The inputs have 100 chunks per document,
`project=document_number % 20`, and a keyword only in document zero. They contain
no embeddings; the declared 768-dimensional collection is empty of vectors.
No model, database or query process is started by preparation. The script copies
the original deterministic construction into a portable entry point; it does not
recreate historical database bytes, source binaries or machine state.

## Optional import-path probe

The original probes access existing crate-private diagnostics. They are retained
for measurement transparency and cannot link to the public library API as
standalone examples. To compile an optional current-source diagnostic, use a
separate copy of product `src`, append the original probe as its binary entry,
and record the resulting new source/binary identity. These paths assume the
product checkout is `../cofferdb`; adjust it to your checkout.

```sh
python3 - <<'PY'
from pathlib import Path
import shutil
source = Path('import-probe-src')
shutil.copytree('../cofferdb/src', source)
entry = source / 'lib.rs'
entry.write_bytes(entry.read_bytes() + b'\n' + Path('historical-import-v2/probes/import-probe.rs').read_bytes())
PY
CARGO_PKG_VERSION=0.1.0 COFFERDB_GIT_COMMIT=unknown rustc --edition=2021 -O \
  --crate-name import_probe import-probe-src/lib.rs -o import-probe
./import-probe import-input/10000 10000 deferred_full_checkpoint bulk > import-probe.jsonl
```

Choose new output locations; preparation refuses an existing source directory and
the probe refuses an existing database. Use the Rust toolchain and target for the
product version being evaluated. Set `CARGO_PKG_VERSION` to that source version;
`unknown` deliberately avoids assigning an old commit to new compiled source.
The default example needs no dependency outside the product source and Rust's
standard library.

The last two probe arguments select its existing checkpoint path and `bulk` or
`small` layout. They are measurement-harness arguments, not new `coffer` options
or supported public import modes. The optional probe has no internal supervisor;
apply an external timeout, preserve nonzero exits and incomplete event streams,
and require its terminal completion record and all assertions. The sample above
is a single current-source functional diagnostic, not a rerun of all eighteen
historical trials or a new comparative performance protocol. A formal new
comparison needs pre-recorded input, source, order, timing and failure rules.

The second retained probe needs the separately reconstructed 100k-vector database
from the [scale guide](scale-replay.md); it is not run by the text-input commands.
Its original twenty-sample measured loop does not add independent semantic
tasks. Keep all source, input and result identities separate from the new public
representative-quality dataset.
