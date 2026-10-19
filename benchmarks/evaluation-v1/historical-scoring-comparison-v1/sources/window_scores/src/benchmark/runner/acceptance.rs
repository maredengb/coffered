use std::cmp::Ordering;
use std::collections::BinaryHeap;

use super::*;
use crate::index::text::{BM25_B, BM25_K1};
use crate::index::vector::raw_score;
use crate::json::ReservedVec;
use crate::temporary::{ExternalKeyLookup, ExternalUniqueKeys};

const QUERY_WINDOW: usize = 4;

// Expectations come from hash-verified source rows, never search results or index bytes.
// Stored IDs are used only to bind the documented internal-ID tie break to this import.
pub(super) struct Acceptance {
    expected: Vec<Vec<Vec<ExpectedHit>>>,
    _work: JsonReservation,
}

struct ExpectedHit {
    external_id: String,
    score: f64,
}

impl Acceptance {
    pub(super) fn expected_hits(
        &self,
        mode: usize,
        query: usize,
    ) -> impl ExactSizeIterator<Item = (&str, f64)> {
        self.expected[mode][query]
            .iter()
            .map(|hit| (hit.external_id.as_str(), hit.score))
    }

    pub(super) fn prepare(
        db: &Database,
        collection: &str,
        dataset: &BenchmarkDataset,
        profile: ResourceProfile,
        modes: &[BenchmarkQueryMode],
        context: &JsonOperationContext,
    ) -> Result<Self> {
        Self::prepare_windowed(
            db,
            collection,
            dataset,
            profile,
            modes,
            context,
            QUERY_WINDOW,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn prepare_windowed(
        db: &Database,
        collection: &str,
        dataset: &BenchmarkDataset,
        profile: ResourceProfile,
        modes: &[BenchmarkQueryMode],
        context: &JsonOperationContext,
        window: usize,
    ) -> Result<Self> {
        let mut source_work = context.reservation();
        let documents = read_verified_baseline_input(
            &dataset.documents_path,
            "documents",
            &dataset.manifest.documents_sha256,
            |source| {
                read_sqlite_baseline_documents(
                    &dataset.documents_path,
                    source,
                    context,
                    &mut source_work,
                )
            },
        )?;
        let mut source_catalog = Catalog::default();
        source_catalog.documents = documents;
        let mut work = context.reservation();
        let mut expected = Vec::new();
        work.reserve_vec(&mut expected, modes.len(), JsonInputKind::BenchmarkArtifact)?;
        for _ in modes {
            let mut queries = Vec::new();
            work.reserve_vec(
                &mut queries,
                dataset.queries.len(),
                JsonInputKind::BenchmarkArtifact,
            )?;
            expected.push(queries);
        }
        db.with_read_snapshot(|catalog, _| {
            let collection_id = &catalog.collection_by_name(collection)?.collection_id;
            if catalog
                .live_documents()
                .filter(|document| &document.collection_id == collection_id)
                .count()
                != source_catalog.documents.len()
            {
                return Err(mismatch(
                    "imported document count differs from fixed dataset",
                ));
            }
            for queries in dataset.queries.chunks(window) {
                let attempt = || {
                    prepare_queries(
                        db,
                        catalog,
                        collection_id,
                        &source_catalog,
                        dataset,
                        queries,
                        profile,
                        modes,
                        context,
                    )
                };
                match attempt() {
                    Ok(part) => append_expected(&mut expected, &mut work, part),
                    Err(error) if queries.len() > 1 && window_capacity_error(&error) => {
                        // Failed window data and credit are gone before one bounded fallback.
                        for query in queries {
                            let part = prepare_queries(
                                db,
                                catalog,
                                collection_id,
                                &source_catalog,
                                dataset,
                                std::slice::from_ref(query),
                                profile,
                                modes,
                                context,
                            )?;
                            append_expected(&mut expected, &mut work, part);
                        }
                    }
                    Err(error) => return Err(error),
                }
            }
            Ok(())
        })?;
        Ok(Self {
            expected,
            _work: work,
        })
    }

    pub(super) fn verify(&self, mode: usize, query: usize, results: &SearchResults) -> Result<()> {
        verify_hits(
            &self.expected[mode][query],
            results
                .results
                .iter()
                .map(|hit| (hit.chunk_external_id.as_str(), hit.final_score)),
        )
    }

    #[cfg(test)]
    pub(super) fn assert_serial_source_candidates(
        db: &Database,
        collection: &str,
        dataset: &BenchmarkDataset,
        profile: ResourceProfile,
        context: &JsonOperationContext,
    ) {
        let mut work = context.reservation();
        let mut documents = Catalog::default();
        documents.documents = read_verified_baseline_input(
            &dataset.documents_path,
            "documents",
            &dataset.manifest.documents_sha256,
            |source| {
                read_sqlite_baseline_documents(&dataset.documents_path, source, context, &mut work)
            },
        )
        .unwrap();
        db.with_read_snapshot(|catalog, _| {
            let collection = &catalog.collection_by_name(collection)?.collection_id;
            for queries in dataset.queries.chunks(QUERY_WINDOW) {
                let window = source_candidates_window(
                    db, catalog, collection, &documents, dataset, queries, profile, context,
                )?;
                for (query, (text, vector)) in queries.iter().zip(window.as_slice()) {
                    let (serial_text, serial_vector) = source_candidates(
                        db, catalog, collection, &documents, dataset, query, profile, context,
                    )?;
                    assert_eq!(
                        text.as_slice()
                            .iter()
                            .map(|hit| (&hit.chunk_id, hit.score.to_bits()))
                            .collect::<Vec<_>>(),
                        serial_text
                            .as_slice()
                            .iter()
                            .map(|hit| (&hit.chunk_id, hit.score.to_bits()))
                            .collect::<Vec<_>>()
                    );
                    assert_eq!(
                        vector
                            .as_slice()
                            .iter()
                            .map(|hit| (
                                &hit.chunk_id,
                                hit.raw_score.to_bits(),
                                hit.normalized_score.to_bits()
                            ))
                            .collect::<Vec<_>>(),
                        serial_vector
                            .as_slice()
                            .iter()
                            .map(|hit| (
                                &hit.chunk_id,
                                hit.raw_score.to_bits(),
                                hit.normalized_score.to_bits()
                            ))
                            .collect::<Vec<_>>()
                    );
                }
            }
            Ok(())
        })
        .unwrap();
    }
}

fn window_capacity_error(error: &Error) -> bool {
    crate::json::is_memory_budget_error(error)
        || (error.code() == ErrorCode::ResourceExhausted
            && error
                .detail()
                .iter()
                .any(|(key, value)| key == "resource" && value == "temporary_disk"))
}

fn append_expected(
    expected: &mut [Vec<Vec<ExpectedHit>>],
    work: &mut JsonReservation,
    mut part: Acceptance,
) {
    let released = part.expected.capacity() * std::mem::size_of::<Vec<Vec<ExpectedHit>>>()
        + part
            .expected
            .iter()
            .map(|queries| queries.capacity() * std::mem::size_of::<Vec<ExpectedHit>>())
            .sum::<usize>();
    for (target, mut queries) in expected.iter_mut().zip(part.expected) {
        debug_assert!(target.capacity() - target.len() >= queries.len());
        target.append(&mut queries);
    }
    part._work.retain(part._work.covered_bytes() - released);
    work.merge(part._work);
}

#[allow(clippy::too_many_arguments)]
fn prepare_queries(
    db: &Database,
    catalog: &Catalog,
    collection: &str,
    documents: &Catalog,
    dataset: &BenchmarkDataset,
    queries: &[BenchmarkQuery],
    profile: ResourceProfile,
    modes: &[BenchmarkQueryMode],
    context: &JsonOperationContext,
) -> Result<Acceptance> {
    let mut work = context.reservation();
    let mut expected = Vec::new();
    work.reserve_vec(&mut expected, modes.len(), JsonInputKind::BenchmarkArtifact)?;
    for _ in modes {
        let mut items = Vec::new();
        work.reserve_vec(&mut items, queries.len(), JsonInputKind::BenchmarkArtifact)?;
        expected.push(items);
    }
    let candidates = source_candidates_window(
        db, catalog, collection, documents, dataset, queries, profile, context,
    )?;
    for (query, (text, vector)) in queries.iter().zip(candidates.as_slice()) {
        for (mode_index, &mode) in modes.iter().enumerate() {
            let ranked = fuse(
                if mode.has_text() {
                    text.as_slice()
                } else {
                    &[]
                },
                if mode.has_vector() {
                    vector.as_slice()
                } else {
                    &[]
                },
                SearchWeights::default_for(mode.has_text(), mode.has_vector()),
                profile.params().fusion_candidates.max(query.top_k),
                context,
                |_| Default::default(),
            )?;
            let mut hits = Vec::new();
            work.reserve_vec(
                &mut hits,
                ranked.as_slice().len().min(query.top_k),
                JsonInputKind::BenchmarkArtifact,
            )?;
            for hit in ranked.as_slice().iter().take(query.top_k) {
                let chunk = catalog.chunk_by_id(collection, hit.chunk_id)?;
                hits.push(ExpectedHit {
                    external_id: work
                        .copy_string(&chunk.chunk_external_id, JsonInputKind::BenchmarkArtifact)?,
                    score: hit.final_score,
                });
            }
            expected[mode_index].push(hits);
        }
    }
    Ok(Acceptance {
        expected,
        _work: work,
    })
}

fn verify_hits<'a>(
    expected: &[ExpectedHit],
    actual: impl IntoIterator<Item = (&'a str, f64)>,
) -> Result<()> {
    let mut actual = actual.into_iter();
    for expected in expected {
        let Some((id, score)) = actual.next() else {
            return Err(mismatch("acceptance query returned too few results"));
        };
        if id != expected.external_id
            || !score.is_finite()
            || (score - expected.score).abs() > 1e-12 * expected.score.abs().max(1.0)
        {
            return Err(
                mismatch("acceptance query rank or score differs from source reference")
                    .with_detail("expected_chunk_external_id", &expected.external_id)
                    .with_detail("actual_chunk_external_id", id),
            );
        }
    }
    if actual.next().is_some() {
        return Err(mismatch("acceptance query returned too many results"));
    }
    Ok(())
}

fn mismatch(message: &'static str) -> Error {
    Error::new(ErrorCode::RecoveryError, message)
        .with_detail("reason", "benchmark_acceptance_mismatch")
}

#[derive(Debug)]
struct TextCandidate<'a> {
    id: &'a str,
    score: f64,
    matches: bool,
}

impl PartialEq for TextCandidate<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for TextCandidate<'_> {}
impl PartialOrd for TextCandidate<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for TextCandidate<'_> {
    fn cmp(&self, other: &Self) -> Ordering {
        // The maximum is the worst retained candidate.
        other
            .score
            .total_cmp(&self.score)
            .then_with(|| self.id.cmp(other.id))
    }
}

struct SourceQuery<'a> {
    query: &'a BenchmarkQuery,
    filter: Option<FilterExpr>,
    _filter_work: JsonReservation,
    terms: crate::tokenizer::QueryTerms<'a>,
    frequency: ReservedVec<usize>,
    document_frequency: ReservedVec<usize>,
    idf: ReservedVec<f64>,
    total_length: usize,
    text: BinaryHeap<TextCandidate<'a>>,
    text_work: crate::json::WorkspaceReservation,
    allowed: Option<ExternalUniqueKeys>,
    lookup: Option<ExternalKeyLookup>,
    hits: Option<Bm25Hits>,
    vector: VectorCandidates,
}

#[allow(clippy::too_many_arguments)]
fn source_candidates_window<'a>(
    db: &Database,
    catalog: &'a Catalog,
    collection: &str,
    documents: &Catalog,
    dataset: &BenchmarkDataset,
    queries: &'a [BenchmarkQuery],
    profile: ResourceProfile,
    context: &JsonOperationContext,
) -> Result<ReservedVec<(Bm25Hits, VectorHits)>> {
    let mut states = ReservedVec::with_capacity(queries.len(), context, "operation_scratch")?;
    for query in queries {
        let (filter, filter_work) =
            FilterExpr::parse_validated_in(&query.filter, context, JsonInputKind::Filter)?;
        let terms = crate::tokenizer::query_terms_in(query.text.as_deref().unwrap_or(""), context)?;
        let mut frequency = ReservedVec::with_capacity(terms.len(), context, "operation_scratch")?;
        let mut document_frequency =
            ReservedVec::with_capacity(terms.len(), context, "operation_scratch")?;
        for _ in 0..terms.len() {
            frequency.push(0usize);
            document_frequency.push(0usize);
        }
        let idf = ReservedVec::with_capacity(terms.len(), context, "operation_scratch")?;
        states.push(SourceQuery {
            query,
            filter,
            _filter_work: filter_work,
            terms,
            frequency,
            document_frequency,
            idf,
            total_length: 0,
            text: BinaryHeap::new(),
            text_work: context.workspace_reservation(),
            allowed: None,
            lookup: None,
            hits: None,
            vector: VectorCandidates::new(
                profile.params().vector_candidates.max(query.top_k),
                context,
            ),
        });
    }
    let mut count = 0usize;
    visit_source_chunks(dataset, context, |row| {
        count += 1;
        for state in states.as_mut_slice() {
            let length = if state.terms.len() == 0 {
                0
            } else {
                term_counts_in(
                    &row.chunk.text,
                    &state.terms,
                    state.frequency.as_mut_slice(),
                    context,
                )?
            };
            state.total_length += length.max(1);
            for (df, &tf) in state
                .document_frequency
                .as_mut_slice()
                .iter_mut()
                .zip(state.frequency.as_slice())
            {
                *df += usize::from(tf != 0);
            }
        }
        Ok(())
    })?;
    if count == 0
        || count
            != catalog
                .live_chunks()
                .filter(|chunk| chunk.collection_id == collection)
                .count()
    {
        return Err(mismatch("imported chunk count differs from fixed dataset"));
    }
    for state in states.as_mut_slice() {
        for &df in state.document_frequency.as_slice() {
            state
                .idf
                .push(((count as f64 - df as f64 + 0.5) / (df as f64 + 0.5) + 1.0).ln());
        }
        if state.query.embedding.is_some() && state.filter.is_some() {
            state.allowed = Some(ExternalUniqueKeys::new(
                &std::env::temp_dir(),
                db.temporary_disk_budget(),
                1024 * 1024,
                context,
                ErrorCode::DuplicateExternalId,
                "duplicate source chunk in benchmark acceptance",
            )?);
        }
    }
    visit_source_chunks(dataset, context, |row| {
        let stored = catalog.chunk_by_external_id(collection, &row.chunk.chunk_external_id)?;
        let chunk = source_chunk(row);
        if !documents.documents.contains_key(&chunk.document_id) {
            return Err(mismatch("source chunk references an absent document"));
        }
        for state in states.as_mut_slice() {
            let matches = state
                .filter
                .as_ref()
                .map_or(true, |filter| filter.matches_for_chunk(documents, &chunk));
            if let Some(allowed) = &mut state.allowed {
                if matches {
                    allowed.push(&[chunk.chunk_external_id.as_bytes()], 0)?;
                }
            }
            let length = if state.terms.len() == 0 {
                1
            } else {
                term_counts_in(
                    &chunk.text,
                    &state.terms,
                    state.frequency.as_mut_slice(),
                    context,
                )?
                .max(1)
            };
            let average = (state.total_length as f64 / count as f64).max(1.0);
            let score = state
                .frequency
                .as_slice()
                .iter()
                .zip(state.idf.as_slice())
                .filter(|(&tf, _)| tf != 0)
                .map(|(&tf, &idf)| {
                    let tf = tf as f64;
                    idf * (tf * (BM25_K1 + 1.0))
                        / (tf + BM25_K1 * (1.0 - BM25_B + BM25_B * length as f64 / average))
                })
                .sum::<f64>();
            if score > 0.0 {
                let candidate = TextCandidate {
                    id: &stored.chunk_id,
                    score,
                    matches,
                };
                let limit = profile
                    .params()
                    .bm25_candidates
                    .max(state.query.top_k)
                    .min(count);
                if state.text.len() < limit {
                    crate::json::reserve_heap_slot(&mut state.text, &mut state.text_work, limit)?;
                    state.text.push(candidate);
                } else if state.text.peek().is_some_and(|worst| candidate < *worst) {
                    *state.text.peek_mut().unwrap() = candidate;
                }
            }
        }
        Ok(())
    })?;
    for state in states.as_mut_slice() {
        state.lookup = state
            .allowed
            .take()
            .map(ExternalUniqueKeys::finish_lookup)
            .transpose()?;
        crate::json::try_shrink_heap(&mut state.text, &mut state.text_work)?;
        let mut hits =
            Bm25Hits::with_capacity(state.text.iter().filter(|hit| hit.matches).count(), context)?;
        for hit in std::mem::take(&mut state.text)
            .into_sorted_vec()
            .into_iter()
            .filter(|hit| hit.matches)
        {
            hits.push(hit.id, hit.score, std::iter::empty())?;
        }
        state.text_work.retain(0);
        state.hits = Some(hits);
    }
    if queries.iter().any(|query| query.embedding.is_some()) {
        let path = dataset
            .embeddings_path
            .as_ref()
            .ok_or_else(|| mismatch("vector acceptance requires fixed source embeddings"))?;
        let expected = dataset
            .manifest
            .embeddings_sha256
            .as_deref()
            .ok_or_else(|| mismatch("vector acceptance requires a source embedding hash"))?;
        read_verified_baseline_input(path, "embeddings", expected, |source| {
            visit_embeddings_jsonl_in(path, source, context, u64::MAX, |_, row, _work| {
                for state in states.as_mut_slice() {
                    let Some(embedding) = &state.query.embedding else {
                        continue;
                    };
                    if let Some(allowed) = &mut state.lookup {
                        if allowed.get(row.chunk_external_id.as_bytes())?.is_none() {
                            continue;
                        }
                    }
                    let chunk = catalog.chunk_by_external_id(collection, &row.chunk_external_id)?;
                    let raw = raw_score(&embedding.vector, &row.vector, DistanceMetric::Cosine);
                    state.vector.push(
                        &chunk.chunk_id,
                        raw,
                        normalize_score(raw, DistanceMetric::Cosine, false),
                    )?;
                }
                Ok(())
            })
        })?;
    }
    let mut result = ReservedVec::with_capacity(queries.len(), context, "operation_scratch")?;
    let (states, state_work) = states.into_parts();
    for state in states {
        result.push((state.hits.unwrap(), state.vector.finish()));
    }
    drop(state_work);
    Ok(result)
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
// Serial source oracle detects arithmetic changes independently of window traversal.
fn source_candidates(
    db: &Database,
    catalog: &Catalog,
    collection: &str,
    documents: &Catalog,
    dataset: &BenchmarkDataset,
    query: &BenchmarkQuery,
    profile: ResourceProfile,
    context: &JsonOperationContext,
) -> Result<(Bm25Hits, VectorHits)> {
    let (filter, _filter_work) =
        FilterExpr::parse_validated_in(&query.filter, context, JsonInputKind::Filter)?;
    let terms = crate::tokenizer::query_terms_in(query.text.as_deref().unwrap_or(""), context)?;
    let mut frequency =
        crate::json::ReservedVec::with_capacity(terms.len(), context, "operation_scratch")?;
    let mut document_frequency =
        crate::json::ReservedVec::with_capacity(terms.len(), context, "operation_scratch")?;
    for _ in 0..terms.len() {
        frequency.push(0usize);
        document_frequency.push(0usize);
    }
    let mut count = 0usize;
    let mut total_length = 0usize;
    visit_source_chunks(dataset, context, |row| {
        let length = if terms.len() == 0 {
            0
        } else {
            term_counts_in(&row.chunk.text, &terms, frequency.as_mut_slice(), context)?
        };
        count += 1;
        total_length += length.max(1);
        for (df, &tf) in document_frequency
            .as_mut_slice()
            .iter_mut()
            .zip(frequency.as_slice())
        {
            *df += usize::from(tf != 0);
        }
        Ok(())
    })?;
    if count == 0
        || count
            != catalog
                .live_chunks()
                .filter(|chunk| chunk.collection_id == collection)
                .count()
    {
        return Err(mismatch("imported chunk count differs from fixed dataset"));
    }
    let average = (total_length as f64 / count as f64).max(1.0);
    let mut idf =
        crate::json::ReservedVec::with_capacity(terms.len(), context, "operation_scratch")?;
    for &df in document_frequency.as_slice() {
        idf.push(((count as f64 - df as f64 + 0.5) / (df as f64 + 0.5) + 1.0).ln());
    }
    let limit = profile.params().bm25_candidates.max(query.top_k).min(count);
    let mut text_work = context.workspace_reservation();
    let mut text: BinaryHeap<TextCandidate<'_>> = BinaryHeap::new();
    let mut allowed = if query.embedding.is_some() && filter.is_some() {
        Some(ExternalUniqueKeys::new(
            &std::env::temp_dir(),
            db.temporary_disk_budget(),
            1024 * 1024,
            context,
            ErrorCode::DuplicateExternalId,
            "duplicate source chunk in benchmark acceptance",
        )?)
    } else {
        None
    };
    visit_source_chunks(dataset, context, |row| {
        let stored = catalog.chunk_by_external_id(collection, &row.chunk.chunk_external_id)?;
        let chunk = source_chunk(row);
        if !documents.documents.contains_key(&chunk.document_id) {
            return Err(mismatch("source chunk references an absent document"));
        }
        let matches = filter
            .as_ref()
            .map_or(true, |filter| filter.matches_for_chunk(documents, &chunk));
        if let Some(allowed) = &mut allowed {
            if matches {
                allowed.push(&[chunk.chunk_external_id.as_bytes()], 0)?;
            }
        }
        let length = if terms.len() == 0 {
            1
        } else {
            term_counts_in(&chunk.text, &terms, frequency.as_mut_slice(), context)?.max(1)
        };
        let score = frequency
            .as_slice()
            .iter()
            .zip(idf.as_slice())
            .filter(|(&tf, _)| tf != 0)
            .map(|(&tf, &idf)| {
                let tf = tf as f64;
                idf * (tf * (BM25_K1 + 1.0))
                    / (tf + BM25_K1 * (1.0 - BM25_B + BM25_B * length as f64 / average))
            })
            .sum::<f64>();
        if score > 0.0 {
            let candidate = TextCandidate {
                id: &stored.chunk_id,
                score,
                matches,
            };
            if text.len() < limit {
                crate::json::reserve_heap_slot(&mut text, &mut text_work, limit)?;
                text.push(candidate);
            } else if text.peek().is_some_and(|worst| candidate < *worst) {
                *text.peek_mut().unwrap() = candidate;
            }
        }
        Ok(())
    })?;
    let mut allowed = allowed.map(ExternalUniqueKeys::finish_lookup).transpose()?;
    crate::json::try_shrink_heap(&mut text, &mut text_work)?;
    let mut hits = Bm25Hits::with_capacity(text.iter().filter(|hit| hit.matches).count(), context)?;
    for hit in text.into_sorted_vec().into_iter().filter(|hit| hit.matches) {
        hits.push(hit.id, hit.score, std::iter::empty())?;
    }
    drop(text_work);
    let mut vector =
        VectorCandidates::new(profile.params().vector_candidates.max(query.top_k), context);
    if let Some(embedding) = &query.embedding {
        let path = dataset
            .embeddings_path
            .as_ref()
            .ok_or_else(|| mismatch("vector acceptance requires fixed source embeddings"))?;
        let expected = dataset
            .manifest
            .embeddings_sha256
            .as_deref()
            .ok_or_else(|| mismatch("vector acceptance requires a source embedding hash"))?;
        read_verified_baseline_input(path, "embeddings", expected, |source| {
            visit_embeddings_jsonl_in(path, source, context, u64::MAX, |_, row, _work| {
                if let Some(allowed) = &mut allowed {
                    if allowed.get(row.chunk_external_id.as_bytes())?.is_none() {
                        return Ok(());
                    }
                }
                let chunk = catalog.chunk_by_external_id(collection, &row.chunk_external_id)?;
                let raw = raw_score(&embedding.vector, &row.vector, DistanceMetric::Cosine);
                vector.push(
                    &chunk.chunk_id,
                    raw,
                    normalize_score(raw, DistanceMetric::Cosine, false),
                )
            })
        })?;
    }
    Ok((hits, vector.finish()))
}

fn visit_source_chunks(
    dataset: &BenchmarkDataset,
    context: &JsonOperationContext,
    mut visit: impl FnMut(crate::import::JsonlChunk) -> Result<()>,
) -> Result<()> {
    read_verified_baseline_input(
        &dataset.chunks_path,
        "chunks",
        &dataset.manifest.chunks_sha256,
        |source| {
            visit_chunks_jsonl_in(
                &dataset.chunks_path,
                source,
                context,
                u64::MAX,
                |_, row, _work| visit(row),
            )
        },
    )
}

fn term_counts_in(
    text: &str,
    terms: &crate::tokenizer::QueryTerms<'_>,
    counts: &mut [usize],
    context: &JsonOperationContext,
) -> Result<usize> {
    counts.fill(0);
    let mut length = 0usize;
    crate::tokenizer::visit_tokens_in(text, context, |token| {
        length = length
            .checked_add(1)
            .ok_or_else(|| Error::new(ErrorCode::FileFormatError, "text token count overflow"))?;
        if let Some(index) = terms.position(token) {
            counts[index] += 1;
        }
        Ok(())
    })?;
    Ok(length)
}

fn source_chunk(row: crate::import::JsonlChunk) -> Chunk {
    Chunk {
        chunk_id: String::new(),
        collection_id: String::new(),
        document_id: row.document_external_id,
        chunk_external_id: row.chunk.chunk_external_id,
        text: row.chunk.text,
        text_hash: String::new(),
        token_count: 0,
        ordinal: row.chunk.ordinal,
        metadata: row.chunk.metadata,
        source_span: row.chunk.source_span,
        embedding_id: None,
        created_at: String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_window_capacity_fallback_is_bounded_and_native_failure_is_not_retried() {
        let directory = super::super::tests::text_only_benchmark_dataset("source-window-fallback");
        let setup = JsonOperationContext::standalone();
        let (mut dataset, input_work) =
            BenchmarkDataset::load_and_validate_in(&directory, &setup).unwrap();
        let db =
            Database::create(directory.join("data.coffer"), DatabaseConfig::default()).unwrap();
        let manifest = write_benchmark_import_manifest(&dataset, "notes", &setup).unwrap();
        db.import_manifest(manifest.path(), OnConflict::Error)
            .unwrap();
        drop(manifest);
        let text = (0..10_000)
            .map(|index| format!("unique{index} "))
            .collect::<String>();
        dataset.queries = (0..5)
            .map(|index| BenchmarkQuery {
                query_id: format!("window-{index}"),
                text: Some(text.clone()),
                filter: JsonValue::Null,
                embedding: None,
                top_k: 1,
                relevant_chunk_ids: Default::default(),
            })
            .collect();
        let modes = benchmark_query_modes(&dataset);
        let limit = 1536 * 1024;
        for parent in [false, true] {
            let context = JsonOperationContext::new(std::sync::Arc::new(
                crate::json::WorkLedger::with_limits(
                    if parent { 2 * limit } else { limit },
                    0,
                    if parent { limit } else { 2 * limit },
                ),
            ));
            let (result, scans) = super::super::source_probe::measure(None, || {
                Acceptance::prepare(
                    &db,
                    "notes",
                    &dataset,
                    ResourceProfile::Balanced,
                    &modes,
                    &context,
                )
            });
            let acceptance = result.unwrap();
            assert_eq!(scans.opened, [1, 10, 0]);
            assert_eq!(scans.verified, scans.opened);
            for mode in 0..modes.len() {
                for query in 0..5 {
                    assert_eq!(acceptance.expected_hits(mode, query).len(), 0);
                }
            }
            drop(acceptance);
            assert_eq!(context.available_workspace_bytes(), limit);
            assert_eq!(db.temporary_disk_budget().used(), 0);
        }
        let context = JsonOperationContext::standalone();
        let capacity = context.available_workspace_bytes();
        let text = dataset.queries[0].text.replace("P".repeat(265));
        let (result, scans) = super::super::source_probe::measure(None, || {
            crate::test_alloc::fail_byte_allocation_once(265, || {
                Acceptance::prepare(
                    &db,
                    "notes",
                    &dataset,
                    ResourceProfile::Balanced,
                    &modes,
                    &context,
                )
            })
        });
        dataset.queries[0].text = text;
        assert_eq!(result.err().unwrap().code(), ErrorCode::IoError);
        assert_eq!(scans.opened, [1, 0, 0]);
        assert_eq!(context.available_workspace_bytes(), capacity);
        let limit = 64 * 1024;
        let context = JsonOperationContext::new(std::sync::Arc::new(
            crate::json::WorkLedger::with_limits(limit, 0, limit),
        ));
        let (result, scans) = super::super::source_probe::measure(None, || {
            Acceptance::prepare(
                &db,
                "notes",
                &dataset,
                ResourceProfile::Balanced,
                &modes,
                &context,
            )
        });
        assert!(window_capacity_error(&result.err().unwrap()));
        assert_eq!(scans.opened, [1, 0, 0]);
        assert_eq!(context.available_workspace_bytes(), limit);
        drop((db, dataset, input_work));
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn stop_word_only_acceptance_keeps_source_validation_but_skips_token_scratch() {
        let directory =
            super::super::tests::text_only_benchmark_dataset("source-stop-word-scratch");
        let part = "P".repeat(131_071);
        let rows = format!("{{\"document_external_id\":\"doc\",\"chunk_external_id\":\"doc#1\",\"ordinal\":0,\"text\":\"{part}_{part}\",\"metadata\":{{}}}}\n");
        fs::write(directory.join("chunks.jsonl"), &rows).unwrap();
        let manifest_path = directory.join("dataset_manifest.json");
        let mut manifest =
            crate::json::parse(&fs::read_to_string(&manifest_path).unwrap()).unwrap();
        manifest
            .as_object_mut()
            .unwrap()
            .get_mut("hashes")
            .unwrap()
            .as_object_mut()
            .unwrap()
            .insert(
                "chunks_sha256".into(),
                JsonValue::String(crate::hash::sha256_hex(rows.as_bytes())),
            );
        fs::write(&manifest_path, manifest.to_compact_string()).unwrap();
        let setup = JsonOperationContext::standalone();
        let (mut dataset, input_work) =
            BenchmarkDataset::load_and_validate_in(&directory, &setup).unwrap();
        let db =
            Database::create(directory.join("data.coffer"), DatabaseConfig::default()).unwrap();
        let manifest = write_benchmark_import_manifest(&dataset, "notes", &setup).unwrap();
        db.import_manifest(manifest.path(), OnConflict::Error)
            .unwrap();
        drop(manifest);
        let mut source_work = setup.reservation();
        let mut documents = Catalog::default();
        documents.documents = read_sqlite_baseline_documents(
            &dataset.documents_path,
            File::open(&dataset.documents_path).unwrap(),
            &setup,
            &mut source_work,
        )
        .unwrap();
        let query = BenchmarkQuery {
            query_id: "stop-word".into(),
            text: Some("THE and".into()),
            filter: JsonValue::Null,
            embedding: None,
            top_k: 1,
            relevant_chunk_ids: Default::default(),
        };
        let context = JsonOperationContext::standalone();
        let capacity = context.available_workspace_bytes();
        let (result, keys) = crate::test_alloc::key_allocations(131_071, || {
            db.with_read_snapshot(|catalog, _| {
                source_candidates(
                    &db,
                    catalog,
                    &catalog.collection_by_name("notes")?.collection_id,
                    &documents,
                    &dataset,
                    &query,
                    ResourceProfile::Balanced,
                    &context,
                )
            })
        });
        let (text, vector) = result.unwrap();
        assert!(text.as_slice().is_empty());
        assert!(vector.as_slice().is_empty());
        assert_eq!(keys.allocated, 0);
        drop((text, vector));
        assert_eq!(context.available_workspace_bytes(), capacity);
        let expected_hash = std::mem::replace(&mut dataset.manifest.chunks_sha256, "0".repeat(64));
        let error = db
            .with_read_snapshot(|catalog, _| {
                source_candidates(
                    &db,
                    catalog,
                    &catalog.collection_by_name("notes")?.collection_id,
                    &documents,
                    &dataset,
                    &query,
                    ResourceProfile::Balanced,
                    &context,
                )
            })
            .err()
            .unwrap();
        assert_eq!(error.code(), ErrorCode::ChecksumMismatch);
        assert_eq!(context.available_workspace_bytes(), capacity);
        dataset.manifest.chunks_sha256 = expected_hash;
        documents.documents.clear();
        let error = db
            .with_read_snapshot(|catalog, _| {
                source_candidates(
                    &db,
                    catalog,
                    &catalog.collection_by_name("notes")?.collection_id,
                    &documents,
                    &dataset,
                    &query,
                    ResourceProfile::Balanced,
                    &context,
                )
            })
            .err()
            .unwrap();
        assert_eq!(error.code(), ErrorCode::RecoveryError);
        assert_eq!(
            error.message(),
            "source chunk references an absent document"
        );
        assert_eq!(context.available_workspace_bytes(), capacity);
        drop((documents, source_work, dataset, input_work, db));
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn source_term_counts_preserve_fixed_tf_df_and_partial_failure_is_discarded() {
        let context = JsonOperationContext::standalone();
        let capacity = context.available_workspace_bytes();
        let terms =
            crate::tokenizer::query_terms_in("Beta alpha ALPHA fooFoo missing", &context).unwrap();
        assert_eq!(
            terms.iter().collect::<Vec<_>>(),
            ["alpha", "beta", "foo", "foofoo", "missing"]
        );
        let mut counts = vec![0; terms.len()];
        let mut df = vec![0; terms.len()];
        let rows = [
            ("ALPHA alpha beta fooFoo", 5, [2, 1, 1, 1, 0]),
            ("beta", 1, [0, 1, 0, 0, 0]),
            ("the and", 0, [0, 0, 0, 0, 0]),
        ];
        let mut total_length = 0;
        for (text, length, expected) in rows {
            assert_eq!(
                term_counts_in(text, &terms, &mut counts, &context).unwrap(),
                length
            );
            assert_eq!(counts, expected);
            total_length += length.max(1);
            for (df, &tf) in df.iter_mut().zip(&counts) {
                *df += usize::from(tf != 0);
            }
        }
        assert_eq!(df, [1, 2, 1, 1, 0]);
        assert_eq!(total_length, 7);
        let available = context.available_workspace_bytes();
        let mut held = context.workspace_reservation();
        held.grow_to(available, available, "operation_scratch")
            .unwrap();
        let error = term_counts_in("alpha PROJECT", &terms, &mut counts, &context).unwrap_err();
        assert_eq!(error.code(), ErrorCode::ResourceExhausted);
        assert_eq!(counts, [1, 0, 0, 0, 0]);
        held.retain(0);
        assert_eq!(
            term_counts_in("beta", &terms, &mut counts, &context).unwrap(),
            1
        );
        assert_eq!(counts, [0, 1, 0, 0, 0]);
        drop(terms);
        assert_eq!(context.available_workspace_bytes(), capacity);
    }

    #[test]
    fn source_acceptance_admits_frequency_df_and_idf_arrays_before_their_allocation() {
        use crate::json::WorkLedger;
        let directory = super::super::tests::text_only_benchmark_dataset("source-array-admission");
        let setup = JsonOperationContext::standalone();
        let (dataset, input_work) =
            BenchmarkDataset::load_and_validate_in(&directory, &setup).unwrap();
        let db =
            Database::create(directory.join("data.coffer"), DatabaseConfig::default()).unwrap();
        let manifest = write_benchmark_import_manifest(&dataset, "notes", &setup).unwrap();
        db.import_manifest(manifest.path(), OnConflict::Error)
            .unwrap();
        drop(manifest);
        let mut source_work = setup.reservation();
        let mut documents = Catalog::default();
        documents.documents = read_sqlite_baseline_documents(
            &dataset.documents_path,
            File::open(&dataset.documents_path).unwrap(),
            &setup,
            &mut source_work,
        )
        .unwrap();
        let count = 10_000usize;
        let text = (0..count)
            .map(|index| format!("unique{index} "))
            .collect::<String>();
        let mut query = BenchmarkQuery {
            query_id: "array-quota".into(),
            text: Some(text),
            filter: JsonValue::Null,
            embedding: None,
            top_k: 1,
            relevant_chunk_ids: Default::default(),
        };
        let nodes = count.div_ceil(5)
            * (11 * std::mem::size_of::<(std::borrow::Cow<'_, str>, usize)>()
                + 16 * std::mem::size_of::<usize>());
        let array = count * std::mem::size_of::<usize>();
        for parent in [false, true] {
            for arrays in 1usize..=3 {
                let requested = nodes
                    + arrays.min(2) * array
                    + usize::from(arrays == 3) * count * std::mem::size_of::<f64>();
                let limit = requested - 1;
                let context =
                    JsonOperationContext::new(std::sync::Arc::new(WorkLedger::with_limits(
                        if parent { 2 * requested } else { limit },
                        0,
                        if parent { limit } else { 2 * requested },
                    )));
                let error = db
                    .with_read_snapshot(|catalog, _| {
                        source_candidates(
                            &db,
                            catalog,
                            &catalog.collection_by_name("notes")?.collection_id,
                            &documents,
                            &dataset,
                            &query,
                            ResourceProfile::Balanced,
                            &context,
                        )
                    })
                    .err()
                    .unwrap();
                assert_eq!(
                    error.code(),
                    ErrorCode::ResourceExhausted,
                    "array {arrays}: {error}"
                );
                assert_eq!(
                    error
                        .detail()
                        .iter()
                        .find(|item| item.0 == "limit_bytes")
                        .map(|item| item.1.as_str()),
                    Some(limit.to_string().as_str())
                );
                assert_eq!(
                    error
                        .detail()
                        .iter()
                        .find(|item| item.0 == "requested_bytes")
                        .map(|item| item.1.as_str()),
                    Some(requested.to_string().as_str())
                );
                assert_eq!(context.available_workspace_bytes(), limit);
            }
        }
        query.text = Some("benchmark query".into());
        let context = JsonOperationContext::standalone();
        let capacity = context.available_workspace_bytes();
        let (text, vector) = db
            .with_read_snapshot(|catalog, _| {
                source_candidates(
                    &db,
                    catalog,
                    &catalog.collection_by_name("notes")?.collection_id,
                    &documents,
                    &dataset,
                    &query,
                    ResourceProfile::Balanced,
                    &context,
                )
            })
            .unwrap();
        assert_eq!(text.as_slice().len(), 2);
        assert!(vector.as_slice().is_empty());
        for hit in text.as_slice() {
            assert!((hit.score - 2.0 * 1.2f64.ln()).abs() < 1e-12);
        }
        drop((text, vector));
        assert_eq!(context.available_workspace_bytes(), capacity);
        drop((query, documents, source_work, dataset, input_work, db));
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn verification_checks_empty_counts_rank_scores_and_nonfinite_values() {
        for seed in 0..100 {
            let expected = (0..seed % 9)
                .map(|i| ExpectedHit {
                    external_id: format!("chunk-{i}"),
                    score: 1.0 / (i + 1) as f64,
                })
                .collect::<Vec<_>>();
            let mut actual = expected
                .iter()
                .map(|hit| (hit.external_id.as_str(), hit.score))
                .collect::<Vec<_>>();
            verify_hits(&expected, actual.iter().copied()).unwrap();
            actual.push(("unexpected", 0.0));
            assert!(verify_hits(&expected, actual.iter().copied()).is_err());
            actual.pop();
            if !actual.is_empty() {
                let saved = actual[0];
                for score in [f64::NAN, f64::INFINITY, -f64::INFINITY, saved.1 + 1e-6] {
                    actual[0].1 = score;
                    assert!(verify_hits(&expected, actual.iter().copied()).is_err());
                }
                actual[0] = ("wrong", saved.1);
                assert!(verify_hits(&expected, actual.iter().copied()).is_err());
                actual[0] = saved;
                actual.pop();
                assert!(verify_hits(&expected, actual.iter().copied()).is_err());
            }
        }
    }

    #[test]
    fn bounded_text_candidates_keep_score_order_then_internal_id_ties() {
        let budget = 1024 * 1024;
        let mut shrink_cases = [false; 3];
        let mut output_cases = [false; 3];
        for seed in 0..100 {
            let ids = (0..37).map(|i| format!("chk_{i}")).collect::<Vec<_>>();
            let mut all = (0..37)
                .map(|i| TextCandidate {
                    id: &ids[i],
                    score: ((i * 17 + seed) % 11) as f64,
                    matches: seed % 10 != 0 && i % 3 != 0,
                })
                .collect::<Vec<_>>();
            let limit = 1 + seed % 53;
            let context = JsonOperationContext::new(std::sync::Arc::new(
                crate::json::WorkLedger::with_limits(budget, 0, budget),
            ));
            let mut work = context.workspace_reservation();
            let mut bounded = BinaryHeap::new();
            for hit in &all {
                let hit = TextCandidate {
                    id: hit.id,
                    score: hit.score,
                    matches: hit.matches,
                };
                if bounded.len() < limit {
                    crate::json::reserve_heap_slot(&mut bounded, &mut work, limit).unwrap();
                    bounded.push(hit);
                } else if hit < *bounded.peek().unwrap() {
                    *bounded.peek_mut().unwrap() = hit;
                }
            }
            all.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.id.cmp(b.id)));
            let expected = all
                .iter()
                .take(limit)
                .filter(|hit| hit.matches)
                .map(|hit| &hit.id)
                .collect::<Vec<_>>();
            let capacity = bounded.capacity();
            let count = bounded.len();
            let size = std::mem::size_of::<TextCandidate<'_>>();
            assert_eq!(work.reserved_bytes(), capacity * size);
            let mut blocker = context.workspace_reservation();
            let available = count * size - 1 + seed % 3;
            let blocked = budget - work.reserved_bytes() - available;
            blocker
                .grow_to(blocked, blocked, "operation_scratch")
                .unwrap();
            let changed = crate::json::try_shrink_heap(&mut bounded, &mut work).unwrap();
            assert_eq!(changed, capacity > count && seed % 3 != 0);
            shrink_cases[if capacity == count {
                0
            } else if changed {
                1
            } else {
                2
            }] = true;
            assert_eq!(bounded.capacity(), if changed { count } else { capacity });
            assert_eq!(work.reserved_bytes(), bounded.capacity() * size);
            let actual = bounded
                .into_sorted_vec()
                .into_iter()
                .filter(|hit| hit.matches)
                .map(|hit| hit.id)
                .collect::<Vec<_>>();
            assert_eq!(actual.iter().collect::<Vec<_>>(), expected);
            output_cases[if actual.is_empty() {
                0
            } else if actual.len() == count {
                1
            } else {
                2
            }] = true;
            drop(work);
            drop(blocker);
            assert_eq!(context.available_workspace_bytes(), budget);
        }
        assert_eq!(shrink_cases, [true; 3]);
        assert_eq!(output_cases, [true; 3]);
    }
}
