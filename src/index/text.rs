use std::cmp::{Ordering, Reverse};
use std::collections::{BTreeMap, BinaryHeap};
#[cfg(test)]
use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

#[cfg(test)]
use crate::catalog::VisibleSnapshot;
use crate::error::Result;
use crate::json::{reserve_heap_slot, try_shrink_heap, JsonOperationContext, ReservedVec};
#[cfg(test)]
use crate::storage::TextIndexSegmentPayload;

mod build;
mod cache;
mod cache_build;
mod candidates;
mod persistent;
mod stats;
mod streamed;
pub(crate) use candidates::{Bm25Hits, TextCandidates};
pub(crate) use stats::TextStatsSummary;

pub const DEFAULT_TOKENIZER_NAME: &str = "default_en";
pub const DEFAULT_TOKENIZER_VERSION: &str = "0.1";
pub const BM25_K1: f64 = 1.2;
pub const BM25_B: f64 = 0.75;

#[derive(Default)]
pub struct TextIndex {
    postings: BTreeMap<String, Vec<Posting>>,
    pub chunk_lengths: BTreeMap<String, usize>,
    pub total_chunks_by_collection: BTreeMap<String, CollectionCounts>,
    pub avg_len_by_collection: BTreeMap<String, f64>,
    cache: Option<cache::TextCacheOwner>,
}

#[derive(Default)]
#[cfg_attr(test, derive(Clone, Debug))]
pub(crate) struct CollectionCounts {
    chunks: usize,
    owner: Option<Arc<str>>,
}

#[cfg_attr(test, derive(Debug))]
pub(crate) enum TextCacheBuild {
    Complete(TextIndex),
    Miss,
    Invalid,
}

#[cfg(test)]
impl CollectionCounts {
    pub(crate) fn count_only(chunks: usize) -> Self {
        Self {
            chunks,
            owner: None,
        }
    }
}

#[cfg(test)]
impl Clone for TextIndex {
    fn clone(&self) -> Self {
        assert!(
            self.cache.is_none(),
            "resident cache credit cannot be cloned"
        );
        Self {
            postings: self.postings.clone(),
            chunk_lengths: self.chunk_lengths.clone(),
            total_chunks_by_collection: self.total_chunks_by_collection.clone(),
            avg_len_by_collection: self.avg_len_by_collection.clone(),
            cache: None,
        }
    }
}

#[cfg(test)]
impl std::fmt::Debug for TextIndex {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("TextIndex")
            .field("postings", &self.postings)
            .field("chunk_lengths", &self.chunk_lengths)
            .field("total_chunks_by_collection", &self.collection_counts())
            .field("avg_len_by_collection", &self.avg_len_by_collection)
            .finish()
    }
}

#[derive(Clone, Debug)]
pub struct Posting {
    pub collection_id: Arc<str>,
    pub chunk_id: String,
    pub term_frequency: usize,
}

#[derive(Clone, Debug, Default)]
pub struct Bm25Hit {
    pub chunk_id: String,
    pub score: f64,
    pub matched_terms: Vec<MatchedTerm>,
}

#[derive(Clone, Copy)]
pub(crate) struct TextSearchOptions {
    pub limit: usize,
    pub explanations: bool,
}

impl TextSearchOptions {
    pub fn with_explanations(limit: usize) -> Self {
        Self {
            limit,
            explanations: true,
        }
    }

    pub fn scores_only(limit: usize) -> Self {
        Self {
            limit,
            explanations: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct MatchedTerm<T = String> {
    pub document_frequency: usize,
    pub chunk_length: usize,
    pub avg_chunk_length: f64,
    pub term: T,
    pub frequency: usize,
    pub bm25_contribution: f64,
}

struct RankedChunk<'a> {
    chunk_id: &'a str,
    score: f64,
}

struct QueryTerm<'a> {
    term: &'a str,
    postings: &'a [Posting],
    idf: f64,
}

fn posting_order(left: &Posting, right: &Posting) -> Ordering {
    left.collection_id
        .cmp(&right.collection_id)
        .then_with(|| left.chunk_id.cmp(&right.chunk_id))
}

fn scoped_postings<'a>(postings: &'a [Posting], collection_id: &str) -> &'a [Posting] {
    let start = postings.partition_point(|posting| posting.collection_id.as_ref() < collection_id);
    let end = postings.partition_point(|posting| posting.collection_id.as_ref() <= collection_id);
    &postings[start..end]
}

impl PartialEq for RankedChunk<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for RankedChunk<'_> {}
impl PartialOrd for RankedChunk<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for RankedChunk<'_> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .partial_cmp(&other.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.chunk_id.cmp(self.chunk_id))
    }
}

fn contribution(frequency: usize, length: usize, avg_len: f64, idf: f64) -> f64 {
    let tf = frequency as f64;
    idf * (tf * (BM25_K1 + 1.0))
        / (tf + BM25_K1 * (1.0 - BM25_B + BM25_B * length as f64 / avg_len))
}

impl TextIndex {
    #[cfg(test)]
    pub(crate) fn collection_counts(&self) -> BTreeMap<String, usize> {
        self.total_chunks_by_collection
            .iter()
            .map(|(id, count)| (id.clone(), count.chunks))
            .collect()
    }

    #[cfg(test)]
    pub fn try_build_from_segments(segments: Vec<TextIndexSegmentPayload>) -> Result<Self> {
        let mut index = TextIndex::default();
        for segment in segments {
            let collection_id = segment.collection_id;
            let chunks_count = segment.chunks.len();
            let mut total_len = 0usize;
            for chunk in segment.chunks {
                let len = chunk.token_count.max(1);
                index.chunk_lengths.insert(chunk.chunk_id, len);
                total_len += len;
            }
            index.total_chunks_by_collection.insert(
                collection_id.clone(),
                CollectionCounts::count_only(chunks_count),
            );
            if chunks_count != 0 {
                index.avg_len_by_collection.insert(
                    collection_id.clone(),
                    total_len as f64 / chunks_count as f64,
                );
            }
            let collection_id: Arc<str> = collection_id.into();
            for term in segment.terms {
                let postings = index.postings.entry(term.term).or_default();
                postings.extend(term.postings.into_iter().map(|posting| Posting {
                    collection_id: Arc::clone(&collection_id),
                    chunk_id: posting.chunk_id,
                    term_frequency: posting.term_frequency,
                }));
            }
        }
        index.sort_postings();
        Ok(index)
    }

    #[cfg(test)]
    pub fn try_build_matching<F, T>(
        snapshot: &VisibleSnapshot<'_>,
        include_collection: impl Fn(&str) -> bool,
        mut load_text: F,
    ) -> Result<Self>
    where
        F: FnMut(&crate::catalog::VisibleChunk<'_>) -> Result<T>,
        T: AsRef<str>,
    {
        let mut index = TextIndex::default();
        let mut collection_lengths: BTreeMap<String, (Arc<str>, usize)> = BTreeMap::new();
        for visible in snapshot.live_chunks() {
            let chunk = visible.chunk;
            if !include_collection(&chunk.collection_id) {
                continue;
            }
            let text = load_text(&visible)?;
            let mut freqs = BTreeMap::<String, usize>::new();
            for token in crate::tokenizer::tokens(text.as_ref()) {
                if let Some(frequency) = freqs.get_mut(token.as_ref()) {
                    *frequency += 1;
                } else {
                    freqs.insert(token.into_owned(), 1);
                }
            }
            let chunk_len = freqs.values().copied().sum::<usize>().max(1);
            index
                .chunk_lengths
                .insert(chunk.chunk_id.clone(), chunk_len);
            let (collection_id, total_length) = collection_lengths
                .entry(chunk.collection_id.clone())
                .or_insert_with(|| (Arc::from(chunk.collection_id.as_str()), 0));
            *total_length += chunk_len;
            index
                .total_chunks_by_collection
                .entry(chunk.collection_id.clone())
                .or_default()
                .chunks += 1;
            for (term, term_frequency) in freqs {
                index.postings.entry(term).or_default().push(Posting {
                    collection_id: Arc::clone(collection_id),
                    chunk_id: chunk.chunk_id.clone(),
                    term_frequency,
                });
            }
        }
        for (collection_id, (_, total_len)) in collection_lengths {
            let count = index.collection_chunk_count(&collection_id).max(1);
            index
                .avg_len_by_collection
                .insert(collection_id, total_len as f64 / count as f64);
        }
        index.sort_postings();
        Ok(index)
    }

    #[cfg(test)]
    pub fn replace_collections<'a>(
        &mut self,
        collections: &BTreeSet<String>,
        old_chunk_ids: impl IntoIterator<Item = &'a str>,
        replacement: Self,
    ) {
        self.postings.retain(|_, postings| {
            postings.retain(|posting| !collections.contains(posting.collection_id.as_ref()));
            !postings.is_empty()
        });
        for id in old_chunk_ids {
            self.chunk_lengths.remove(id);
        }
        self.total_chunks_by_collection
            .retain(|id, _| !collections.contains(id));
        self.avg_len_by_collection
            .retain(|id, _| !collections.contains(id));
        for (term, postings) in replacement.postings {
            let combined = self.postings.entry(term).or_default();
            combined.extend(postings);
            combined.sort_unstable_by(posting_order);
        }
        self.chunk_lengths.extend(replacement.chunk_lengths);
        self.total_chunks_by_collection
            .extend(replacement.total_chunks_by_collection);
        self.avg_len_by_collection
            .extend(replacement.avg_len_by_collection);
    }

    #[cfg(test)]
    pub fn term_count(&self) -> usize {
        self.postings.len()
    }

    #[cfg(test)]
    pub(crate) fn term_count_in_collection(&self, collection_id: &str) -> usize {
        self.postings
            .values()
            .filter(|postings| {
                let first = postings
                    .partition_point(|posting| posting.collection_id.as_ref() < collection_id);
                postings
                    .get(first)
                    .is_some_and(|posting| posting.collection_id.as_ref() == collection_id)
            })
            .count()
    }

    #[cfg(test)]
    pub fn postings(&self) -> &BTreeMap<String, Vec<Posting>> {
        &self.postings
    }

    #[cfg(test)]
    pub(crate) fn terms_in_collection<'a>(
        &'a self,
        collection_id: &'a str,
    ) -> impl Iterator<Item = (&'a str, &'a [Posting])> + Clone {
        self.postings.iter().filter_map(move |(term, postings)| {
            let postings = scoped_postings(postings, collection_id);
            (!postings.is_empty()).then_some((term.as_str(), postings))
        })
    }

    fn sort_postings(&mut self) {
        for postings in self.postings.values_mut() {
            postings.sort_unstable_by(posting_order);
        }
    }

    fn collection_postings(&self, collection_id: &str, term: &str) -> &[Posting] {
        let Some(postings) = self.postings.get(term) else {
            return &[];
        };
        scoped_postings(postings, collection_id)
    }

    pub fn search(
        &self,
        collection_id: &str,
        query: &str,
        options: TextSearchOptions,
        context: &JsonOperationContext,
    ) -> Result<Bm25Hits> {
        let limit = options.limit;
        if limit == 0 {
            return Ok(Bm25Hits::empty(context));
        }
        let n = self.collection_chunk_count(collection_id) as f64;
        if n == 0.0 {
            return Ok(Bm25Hits::empty(context));
        }
        let avg_len = self
            .avg_len_by_collection
            .get(collection_id)
            .copied()
            .unwrap_or(1.0)
            .max(1.0);
        let terms = crate::tokenizer::query_terms_in(query, context)?;
        let count = terms
            .iter()
            .filter(|term| !self.collection_postings(collection_id, term).is_empty())
            .count();
        let mut plans = ReservedVec::with_capacity(count, context, "operation_scratch")?;
        for term in terms.iter() {
            let postings = self.collection_postings(collection_id, term);
            let df = postings.len();
            if df == 0 {
                continue;
            }
            let idf = ((n - df as f64 + 0.5) / (df as f64 + 0.5) + 1.0).ln();
            plans.push(QueryTerm {
                term,
                postings,
                idf,
            });
        }

        // Merge by chunk, then lexical term order to preserve floating-point sums.
        let plans = plans.as_slice();
        let mut pending = ReservedVec::with_capacity(plans.len(), context, "operation_scratch")?;
        for (plan, term) in plans.iter().enumerate() {
            pending.push(Reverse((term.postings[0].chunk_id.as_str(), plan, 0)));
        }
        let (pending, pending_work) = pending.into_parts();
        let mut pending = BinaryHeap::from(pending);
        let mut candidate_work = context.workspace_reservation();
        let mut candidates: BinaryHeap<Reverse<RankedChunk<'_>>> = BinaryHeap::new();
        while let Some(&Reverse((chunk_id, _, _))) = pending.peek() {
            context.search_cost_add(crate::search_cost::Counter::TextDocumentVisits, 1);
            let len = self.chunk_lengths.get(chunk_id).copied().unwrap_or(1);
            let mut score = 0.0;
            while pending
                .peek()
                .is_some_and(|Reverse((next_chunk, _, _))| *next_chunk == chunk_id)
            {
                let Reverse((_, plan, offset)) = pending.pop().unwrap();
                let term = &plans[plan];
                context.search_cost_add(crate::search_cost::Counter::TextPostingVisits, 1);
                score += contribution(term.postings[offset].term_frequency, len, avg_len, term.idf);
                if let Some(next) = term.postings.get(offset + 1) {
                    pending.push(Reverse((next.chunk_id.as_str(), plan, offset + 1)));
                }
            }
            let candidate = RankedChunk { chunk_id, score };
            if candidates.len() < limit {
                reserve_heap_slot(&mut candidates, &mut candidate_work, limit)?;
                candidates.push(Reverse(candidate));
            } else if candidates.peek().is_some_and(|worst| candidate > worst.0) {
                *candidates.peek_mut().unwrap() = Reverse(candidate);
            }
        }
        drop(pending);
        drop(pending_work);
        try_shrink_heap(&mut candidates, &mut candidate_work)?;
        let mut hits = Bm25Hits::with_capacity(candidates.len(), context)?;
        for Reverse(candidate) in candidates.into_sorted_vec() {
            let len = self
                .chunk_lengths
                .get(candidate.chunk_id)
                .copied()
                .unwrap_or(1);
            let matched_terms = plans
                .iter()
                .take(if options.explanations { plans.len() } else { 0 })
                .filter_map(|term| {
                    let offset = term
                        .postings
                        .binary_search_by(|posting| {
                            posting.chunk_id.as_str().cmp(candidate.chunk_id)
                        })
                        .ok()?;
                    let posting = &term.postings[offset];
                    context.search_cost_add(crate::search_cost::Counter::TextPostingVisits, 1);
                    Some(MatchedTerm {
                        document_frequency: term.postings.len(),
                        chunk_length: len,
                        avg_chunk_length: avg_len,
                        term: term.term,
                        frequency: posting.term_frequency,
                        bm25_contribution: contribution(
                            posting.term_frequency,
                            len,
                            avg_len,
                            term.idf,
                        ),
                    })
                });
            hits.push(candidate.chunk_id, candidate.score, matched_terms)?;
        }
        Ok(hits)
    }

    // Score only admitted IDs, using the same collection statistics and term order
    // as recall. Zero-match IDs remain present with a real zero score.
    pub(crate) fn score_ids(
        &self,
        collection_id: &str,
        query: &str,
        ids: &[&str],
        context: &JsonOperationContext,
    ) -> Result<Bm25Hits> {
        let terms = crate::tokenizer::query_terms_in(query, context)?;
        let n = self.collection_chunk_count(collection_id) as f64;
        let avg_len = self
            .avg_len_by_collection
            .get(collection_id)
            .copied()
            .unwrap_or(1.0)
            .max(1.0);
        let mut hits = Bm25Hits::with_capacity(ids.len(), context)?;
        for &id in ids {
            context.search_cost_add(crate::search_cost::Counter::TextDocumentVisits, 1);
            let len = self.chunk_lengths.get(id).copied().ok_or_else(|| {
                crate::Error::new(
                    crate::ErrorCode::IndexCorruption,
                    "text candidate has no chunk length",
                )
            })?;
            let matched = terms.iter().filter_map(|term| {
                let postings = self.collection_postings(collection_id, term);
                let offset = postings
                    .binary_search_by(|posting| posting.chunk_id.as_str().cmp(id))
                    .ok()?;
                context.search_cost_add(crate::search_cost::Counter::TextPostingVisits, 1);
                let df = postings.len();
                let idf = ((n - df as f64 + 0.5) / (df as f64 + 0.5) + 1.0).ln();
                Some(MatchedTerm {
                    document_frequency: df,
                    chunk_length: len,
                    avg_chunk_length: avg_len,
                    term,
                    frequency: postings[offset].term_frequency,
                    bm25_contribution: contribution(
                        postings[offset].term_frequency,
                        len,
                        avg_len,
                        idf,
                    ),
                })
            });
            let score = matched
                .clone()
                .fold(0.0, |sum, term| sum + term.bm25_contribution);
            hits.push(id, score, matched)?;
        }
        Ok(hits)
    }

    #[cfg(test)]
    fn search_reference(&self, collection_id: &str, query: &str, limit: usize) -> Vec<Bm25Hit> {
        let terms = tokenize(query);
        if terms.is_empty() {
            return Vec::new();
        }
        let unique_terms = terms.iter().cloned().collect::<BTreeSet<_>>();
        let n = self.collection_chunk_count(collection_id) as f64;
        if n == 0.0 {
            return Vec::new();
        }
        let avg_len = self
            .avg_len_by_collection
            .get(collection_id)
            .copied()
            .unwrap_or(1.0)
            .max(1.0);
        let mut scores: HashMap<String, Bm25Hit> = HashMap::new();
        for term in unique_terms {
            let postings = self.postings.get(&term).map(Vec::as_slice).unwrap_or(&[]);
            let collection_postings = postings
                .iter()
                .filter(|posting| posting.collection_id.as_ref() == collection_id)
                .collect::<Vec<_>>();
            let df = collection_postings.len() as f64;
            if df == 0.0 {
                continue;
            }
            let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();
            for posting in collection_postings {
                let tf = posting.term_frequency as f64;
                let len = self
                    .chunk_lengths
                    .get(&posting.chunk_id)
                    .copied()
                    .unwrap_or(1) as f64;
                let contribution = idf * (tf * (BM25_K1 + 1.0))
                    / (tf + BM25_K1 * (1.0 - BM25_B + BM25_B * len / avg_len));
                let hit = scores
                    .entry(posting.chunk_id.clone())
                    .or_insert_with(|| Bm25Hit {
                        chunk_id: posting.chunk_id.clone(),
                        score: 0.0,
                        matched_terms: Vec::new(),
                    });
                hit.score += contribution;
                hit.matched_terms.push(MatchedTerm {
                    document_frequency: df as usize,
                    chunk_length: len as usize,
                    avg_chunk_length: avg_len,
                    term: term.clone(),
                    frequency: posting.term_frequency,
                    bm25_contribution: contribution,
                });
            }
        }
        let mut hits = scores.into_values().collect::<Vec<_>>();
        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.chunk_id.cmp(&b.chunk_id))
        });
        hits.truncate(limit);
        hits
    }
}

#[cfg(test)]
pub fn tokenize(input: &str) -> Vec<String> {
    crate::tokenizer::tokenize(input)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shared_term_index(ids: &[String]) -> TextIndex {
        let mut index = TextIndex::default();
        for id in ids {
            index.chunk_lengths.insert(id.clone(), 1);
        }
        index
            .total_chunks_by_collection
            .insert("collection".into(), CollectionCounts::count_only(ids.len()));
        index.avg_len_by_collection.insert("collection".into(), 1.0);
        index.postings.insert(
            "shared".into(),
            ids.iter()
                .map(|id| Posting {
                    collection_id: "collection".into(),
                    chunk_id: id.clone(),
                    term_frequency: 1,
                })
                .collect(),
        );
        index
    }

    #[test]
    fn cached_candidate_optional_shrink_refusal_keeps_capacity_charged() {
        for seed in 0..100 {
            let ids = [format!("id-{seed}-first"), format!("id-{seed}-second")];
            let index = shared_term_index(&ids);
            let base = crate::json::btree_workspace::<std::borrow::Cow<'_, str>, usize>(1)
                + std::mem::size_of::<QueryTerm<'_>>();
            let pending = std::mem::size_of::<(&str, usize, usize)>();
            let slots = 3 * std::mem::size_of::<RankedChunk<'_>>();
            let limit = base + pending + slots;
            let requested = base + slots + 2 * std::mem::size_of::<Bm25Hit>();
            assert!(pending < 2 * std::mem::size_of::<RankedChunk<'_>>());
            for parent in [false, true] {
                for options in [
                    TextSearchOptions::with_explanations(usize::MAX),
                    TextSearchOptions::scores_only(usize::MAX),
                ] {
                    let context =
                        JsonOperationContext::new(Arc::new(crate::json::WorkLedger::with_limits(
                            if parent { 2 * limit } else { limit },
                            0,
                            if parent { limit } else { 2 * limit },
                        )));
                    let error = index
                        .search("collection", "shared", options, &context)
                        .unwrap_err();
                    assert_eq!(error.code(), crate::ErrorCode::ResourceExhausted);
                    for (key, value) in [
                        ("resource", "memory".into()),
                        ("operation", "operation_scratch".into()),
                        ("limit_bytes", limit.to_string()),
                        ("requested_bytes", requested.to_string()),
                    ] {
                        assert!(error.detail().contains(&(key.into(), value)));
                    }
                    assert_eq!(context.available_workspace_bytes(), limit);
                }
            }
        }
    }

    #[test]
    fn cached_candidate_spare_slots_are_released_before_explanation_admission() {
        for seed in 0..100 {
            let ids = [format!("id-{seed}-first"), format!("id-{seed}-second")];
            let index = shared_term_index(&ids);
            for options in [
                TextSearchOptions::with_explanations(usize::MAX),
                TextSearchOptions::scores_only(usize::MAX),
            ] {
                let output_bytes = ids
                    .iter()
                    .map(|id| {
                        std::mem::size_of::<Bm25Hit>()
                            + id.len()
                            + if options.explanations {
                                std::mem::size_of::<MatchedTerm>() + "shared".len()
                            } else {
                                0
                            }
                    })
                    .sum::<usize>();
                let required = output_bytes
                    + crate::json::btree_workspace::<std::borrow::Cow<'_, str>, usize>(1)
                    + 2 * std::mem::size_of::<RankedChunk<'_>>()
                    + std::mem::size_of::<QueryTerm<'_>>();
                let context = JsonOperationContext::new(Arc::new(
                    crate::json::WorkLedger::with_limits(required, required, required),
                ));
                let actual = index
                    .search("collection", "shared", options, &context)
                    .unwrap();
                let mut expected = index.search_reference("collection", "shared", usize::MAX);
                if !options.explanations {
                    for hit in &mut expected {
                        hit.matched_terms.clear();
                    }
                    assert!(actual.iter().all(|hit| hit.matched_terms.capacity() == 0));
                }
                assert_same_hits(&actual, &expected);
                let mut spare = context.workspace_reservation();
                assert_eq!(
                    spare.grow_to(0, required, "operation_scratch").unwrap(),
                    required - output_bytes
                );
                assert!(spare
                    .grow_to(
                        spare.reserved_bytes() + 1,
                        spare.reserved_bytes() + 1,
                        "operation_scratch"
                    )
                    .is_err());
                drop(actual);
                spare
                    .grow_to(required, required, "operation_scratch")
                    .unwrap();
                spare.retain(1);
                let error = index
                    .search("collection", "shared", options, &context)
                    .unwrap_err();
                assert_eq!(error.code(), crate::ErrorCode::ResourceExhausted);
                let detail = error.detail().iter().cloned().collect::<BTreeMap<_, _>>();
                assert_eq!(detail["resource"], "memory");
                assert_eq!(detail["operation"], "operation_scratch");
                assert_eq!(detail["limit_bytes"], required.to_string());
                assert_eq!(detail["requested_bytes"], (required + 1).to_string());
                spare
                    .grow_to(required, required, "operation_scratch")
                    .unwrap();
            }
        }
    }

    pub(super) fn assert_same_hits(actual: &[Bm25Hit], expected: &[Bm25Hit]) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert_eq!(actual.chunk_id, expected.chunk_id);
            assert_eq!(actual.score.to_bits(), expected.score.to_bits());
            assert_eq!(actual.matched_terms.len(), expected.matched_terms.len());
            for (actual, expected) in actual.matched_terms.iter().zip(&expected.matched_terms) {
                assert_eq!(actual.term, expected.term);
                assert_eq!(actual.frequency, expected.frequency);
                assert_eq!(actual.document_frequency, expected.document_frequency);
                assert_eq!(actual.chunk_length, expected.chunk_length);
                assert_eq!(
                    actual.avg_chunk_length.to_bits(),
                    expected.avg_chunk_length.to_bits()
                );
                assert_eq!(
                    actual.bm25_contribution.to_bits(),
                    expected.bm25_contribution.to_bits()
                );
            }
        }
    }

    #[test]
    fn bounded_candidates_match_full_scoring_and_explanations_across_100_seeds() {
        for seed in 0..100 {
            let mut index = TextIndex::default();
            for collection in ["first", "second"] {
                let count = 20 + seed % 31;
                let mut total_length = 0;
                for chunk in 0..count {
                    let chunk_id = format!("{collection}-{chunk}");
                    let mut length = 0;
                    for (ordinal, term) in [
                        "shared",
                        "alpha",
                        "beta",
                        "gamma",
                        "omega",
                        "foobar",
                        "foo",
                        "bar",
                        "本地检索",
                    ]
                    .into_iter()
                    .enumerate()
                    {
                        let frequency = if ordinal == 0 {
                            1
                        } else {
                            (chunk * 7 + seed * 13 + ordinal) % 4
                        };
                        if frequency != 0 {
                            length += frequency;
                            index
                                .postings
                                .entry(term.into())
                                .or_default()
                                .push(Posting {
                                    collection_id: collection.into(),
                                    chunk_id: chunk_id.clone(),
                                    term_frequency: frequency,
                                });
                        }
                    }
                    index.chunk_lengths.insert(chunk_id, length);
                    total_length += length;
                }
                index
                    .total_chunks_by_collection
                    .insert(collection.into(), CollectionCounts::count_only(count));
                index
                    .avg_len_by_collection
                    .insert(collection.into(), total_length as f64 / count as f64);
            }
            // Compare against the old scorer before normalizing input order.
            for postings in index.postings.values_mut() {
                postings.reverse();
                let rotation = seed % postings.len();
                postings.rotate_left(rotation);
            }
            let reference = index.clone();
            index.sort_postings();
            for collection in ["first", "second", "missing"] {
                for query in [
                    "",
                    "the and",
                    "absent",
                    "shared",
                    "ALPHA beta shared alpha",
                    "fooBar gamma omega",
                    "本地检索 alpha",
                ] {
                    for limit in [0, 1, 3, 17, usize::MAX] {
                        assert_same_hits(
                            &index
                                .search(
                                    collection,
                                    query,
                                    TextSearchOptions::with_explanations(limit),
                                    &JsonOperationContext::standalone(),
                                )
                                .unwrap(),
                            &reference.search_reference(collection, query, limit),
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn equal_scores_use_lexical_ids_and_only_return_selected_explanations() {
        let mut index = TextIndex::default();
        index
            .total_chunks_by_collection
            .insert("collection".into(), CollectionCounts::count_only(10000));
        index.avg_len_by_collection.insert("collection".into(), 1.0);
        let postings = (0..10000)
            .rev()
            .map(|chunk| {
                let chunk_id = format!("chunk-{chunk}");
                index.chunk_lengths.insert(chunk_id.clone(), 1);
                Posting {
                    collection_id: "collection".into(),
                    chunk_id,
                    term_frequency: 1,
                }
            })
            .collect();
        index.postings.insert("shared".into(), postings);
        let reference = index.clone();
        index.sort_postings();
        for limit in [0, 1, 10, 200, 10000, usize::MAX] {
            let actual = index
                .search(
                    "collection",
                    "shared shared",
                    TextSearchOptions::with_explanations(limit),
                    &JsonOperationContext::standalone(),
                )
                .unwrap();
            assert_same_hits(
                &actual,
                &reference.search_reference("collection", "shared shared", limit),
            );
            assert_eq!(
                actual
                    .iter()
                    .map(|hit| hit.matched_terms.len())
                    .sum::<usize>(),
                limit.min(10000)
            );
        }
    }

    #[test]
    fn collection_term_counts_are_distinct_and_do_not_include_other_collections() {
        for seed in 0..100 {
            let mut index = TextIndex::default();
            for term in 0..seed {
                for (owner, collection) in ["col_2", "col_10", "col_1"].into_iter().enumerate() {
                    for occurrence in 0..((seed + term * (owner + 1)) % 5) {
                        index
                            .postings
                            .entry(format!("term-{term}"))
                            .or_default()
                            .push(Posting {
                                collection_id: collection.into(),
                                chunk_id: format!("chunk-{owner}-{occurrence}"),
                                term_frequency: 1,
                            });
                    }
                }
            }
            index.sort_postings();
            for collection in ["col_1", "col_10", "col_2", "col_0", "col_11", "col_3", ""] {
                let expected = index
                    .postings
                    .values()
                    .filter(|postings| {
                        postings
                            .iter()
                            .any(|posting| posting.collection_id.as_ref() == collection)
                    })
                    .count();
                assert_eq!(
                    index.term_count_in_collection(collection),
                    expected,
                    "seed {seed}, {collection}"
                );
            }
            assert_eq!(index.term_count(), index.postings.len());
        }
        assert_eq!(TextIndex::default().term_count_in_collection("absent"), 0);
    }

    #[test]
    fn segment_build_and_replacement_order_postings_across_id_widths() {
        use crate::storage::{TextIndexChunkEntry, TextIndexPostingEntry, TextIndexTermEntry};

        fn segment(collection: usize, seed: usize, replacement: bool) -> TextIndexSegmentPayload {
            let mut numbers = vec![
                1 + collection as u64,
                9_999_999_999_999_999 - collection as u64,
                10_000_000_000_000_000 + collection as u64,
                u64::MAX - collection as u64,
            ];
            if replacement {
                numbers.remove(0);
                numbers.push(100 + collection as u64);
            }
            let rotation = seed % numbers.len();
            numbers.rotate_left(rotation);
            let mut chunks = Vec::new();
            let mut terms = BTreeMap::<String, Vec<TextIndexPostingEntry>>::new();
            for (ordinal, number) in numbers.into_iter().enumerate() {
                let chunk_id = format!("chk_{number:016}");
                let mut length = 0;
                for (term_index, term) in ["shared", "alpha", "zeta"].into_iter().enumerate() {
                    let frequency = if term_index == 0 {
                        1
                    } else {
                        (ordinal + term_index + seed) % 4
                    };
                    if frequency != 0 {
                        length += frequency;
                        terms
                            .entry(term.into())
                            .or_default()
                            .push(TextIndexPostingEntry {
                                chunk_id: chunk_id.clone(),
                                term_frequency: frequency,
                            });
                    }
                }
                chunks.push(TextIndexChunkEntry {
                    chunk_id,
                    document_id: format!("doc_{collection}"),
                    token_count: length,
                });
            }
            TextIndexSegmentPayload {
                collection_id: ["z", "a", "m"][collection].into(),
                covered_commit_sequence_no: seed as u64,
                chunks,
                terms: terms
                    .into_iter()
                    .map(|(term, postings)| TextIndexTermEntry { term, postings })
                    .collect(),
            }
        }

        for seed in 0..100 {
            let mut segments = (0..3)
                .map(|collection| segment(collection, seed, false))
                .collect::<Vec<_>>();
            let mut index = TextIndex::try_build_from_segments(segments.clone()).unwrap();
            for iteration in 0..=3 {
                for postings in index.postings.values() {
                    assert!(postings
                        .windows(2)
                        .all(|pair| posting_order(&pair[0], &pair[1]) == Ordering::Less));
                }
                let reference = TextIndex::try_build_from_segments(segments.clone()).unwrap();
                assert_eq!(index.chunk_lengths, reference.chunk_lengths);
                assert_eq!(index.collection_counts(), reference.collection_counts());
                assert_eq!(index.avg_len_by_collection, reference.avg_len_by_collection);
                for collection in ["a", "m", "z", "absent"] {
                    for query in ["shared", "alpha zeta shared", "zeta alpha"] {
                        for limit in [0, 1, 3, usize::MAX] {
                            assert_same_hits(
                                &index
                                    .search(
                                        collection,
                                        query,
                                        TextSearchOptions::with_explanations(limit),
                                        &JsonOperationContext::standalone(),
                                    )
                                    .unwrap(),
                                &reference.search_reference(collection, query, limit),
                            );
                        }
                    }
                }
                if iteration == 3 {
                    break;
                }
                let target = (seed + iteration) % 3;
                let old_ids = segments[target]
                    .chunks
                    .iter()
                    .rev()
                    .map(|chunk| chunk.chunk_id.as_str())
                    .chain(std::iter::once("chk_missing"))
                    .chain(
                        segments[target]
                            .chunks
                            .first()
                            .map(|chunk| chunk.chunk_id.as_str()),
                    );
                let mut replacement = segment(target, seed + 7, true);
                if iteration == 2 {
                    replacement.chunks.clear();
                    replacement.terms.clear();
                }
                index.replace_collections(
                    &[replacement.collection_id.clone()].into_iter().collect(),
                    old_ids,
                    TextIndex::try_build_from_segments(vec![replacement.clone()]).unwrap(),
                );
                segments[target] = replacement;
            }
        }
    }

    #[test]
    fn removing_collections_consumes_borrowed_ids_without_intermediate_allocations() {
        let mut index = TextIndex::default();
        for (id, owner, length) in [
            ("deleted-a", "first", 2),
            ("deleted-b", "first", 3),
            ("kept", "second", 5),
        ] {
            index.chunk_lengths.insert(id.into(), length);
            index
                .postings
                .entry("shared".into())
                .or_default()
                .push(Posting {
                    collection_id: Arc::from(owner),
                    chunk_id: id.into(),
                    term_frequency: 1,
                });
        }
        index.total_chunks_by_collection = BTreeMap::from([
            ("first".into(), CollectionCounts::count_only(2)),
            ("second".into(), CollectionCounts::count_only(1)),
        ]);
        index.avg_len_by_collection =
            BTreeMap::from([("first".into(), 2.5), ("second".into(), 5.0)]);
        let kept_pointer = index.postings["shared"][2].chunk_id.as_ptr();
        let collections = BTreeSet::from(["first".into()]);
        let ids = ["deleted-b", "missing", "deleted-a", "deleted-b"].into_iter();
        let (_, allocations) = crate::test_alloc::allocations(|| {
            index.replace_collections(&collections, ids, TextIndex::default());
        });
        assert_eq!(allocations, 0);
        assert_eq!(index.chunk_lengths, BTreeMap::from([("kept".into(), 5)]));
        assert_eq!(
            index.collection_counts(),
            BTreeMap::from([("second".into(), 1)])
        );
        assert_eq!(
            index.avg_len_by_collection,
            BTreeMap::from([("second".into(), 5.0)])
        );
        let kept = &index.postings["shared"];
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].chunk_id, "kept");
        assert_eq!(kept[0].chunk_id.as_ptr(), kept_pointer);
        assert_eq!(kept[0].collection_id.as_ref(), "second");
        assert_eq!(kept[0].term_frequency, 1);
    }

    #[test]
    fn decoded_postings_share_one_collection_identity_across_terms_and_clones() {
        use crate::storage::{TextIndexChunkEntry, TextIndexPostingEntry, TextIndexTermEntry};
        for seed in 0..100 {
            let count = 1 + seed % 17;
            let segments = (1..=3)
                .map(|owner| TextIndexSegmentPayload {
                    collection_id: format!("col_{owner}"),
                    covered_commit_sequence_no: seed as u64,
                    chunks: (0..count)
                        .map(|chunk| TextIndexChunkEntry {
                            chunk_id: format!("chk_{owner}_{chunk}"),
                            document_id: format!("doc_{owner}"),
                            token_count: 4,
                        })
                        .collect(),
                    terms: ["alpha", "beta", "shared"]
                        .into_iter()
                        .map(|term| TextIndexTermEntry {
                            term: term.into(),
                            postings: (0..count)
                                .map(|chunk| TextIndexPostingEntry {
                                    chunk_id: format!("chk_{owner}_{chunk}"),
                                    term_frequency: 1,
                                })
                                .collect(),
                        })
                        .collect(),
                })
                .collect();
            let index = TextIndex::try_build_from_segments(segments).unwrap();
            let owners = index.postings["alpha"]
                .iter()
                .step_by(count)
                .map(|posting| Arc::downgrade(&posting.collection_id))
                .collect::<Vec<_>>();
            for postings in index.postings.values() {
                for (owner, group) in postings.chunks_exact(count).enumerate() {
                    let identity = owners[owner].upgrade().unwrap();
                    assert!(group
                        .iter()
                        .all(|posting| Arc::ptr_eq(&identity, &posting.collection_id)));
                }
            }
            assert!(owners.iter().all(|owner| owner.strong_count() == count * 3));
            let cloned = index.clone();
            assert!(owners.iter().all(|owner| owner.strong_count() == count * 6));
            for owner in 1..=3 {
                assert_same_hits(
                    &cloned
                        .search(
                            &format!("col_{owner}"),
                            "alpha shared",
                            TextSearchOptions::with_explanations(count),
                            &JsonOperationContext::standalone(),
                        )
                        .unwrap(),
                    &index.search_reference(&format!("col_{owner}"), "alpha shared", count),
                );
            }
            drop(index);
            assert!(owners.iter().all(|owner| owner.strong_count() == count * 3));
            drop(cloned);
            assert!(owners.iter().all(|owner| owner.upgrade().is_none()));
        }
    }

    #[test]
    fn owned_segments_move_keys_and_preserve_collection_scoring() {
        use crate::storage::{TextIndexChunkEntry, TextIndexPostingEntry, TextIndexTermEntry};
        for seed in 0..100 {
            let mut chunk_pointers = Vec::new();
            let mut posting_pointers = Vec::new();
            let mut term_pointer = std::ptr::null();
            let mut segments = Vec::new();
            for collection in 1..=2 {
                let chunk_id = format!("chk_{collection}");
                chunk_pointers.push(chunk_id.as_ptr());
                let posting_id = chunk_id.clone();
                posting_pointers.push(posting_id.as_ptr());
                let term = "shared".to_owned();
                if collection == 1 {
                    term_pointer = term.as_ptr();
                }
                segments.push(TextIndexSegmentPayload {
                    collection_id: format!("col_{collection}"),
                    covered_commit_sequence_no: seed as u64,
                    chunks: vec![TextIndexChunkEntry {
                        chunk_id,
                        document_id: format!("doc_{collection}"),
                        token_count: (seed + collection) * 3,
                    }],
                    terms: vec![TextIndexTermEntry {
                        term,
                        postings: vec![TextIndexPostingEntry {
                            chunk_id: posting_id,
                            term_frequency: seed % 3 + 1,
                        }],
                    }],
                });
            }
            segments.push(TextIndexSegmentPayload {
                collection_id: "empty".into(),
                covered_commit_sequence_no: 0,
                chunks: Vec::new(),
                terms: Vec::new(),
            });
            let index = TextIndex::try_build_from_segments(segments).unwrap();
            assert_eq!(
                index.postings.first_key_value().unwrap().0.as_ptr(),
                term_pointer
            );
            for collection in 1..=2 {
                let chunk_id = format!("chk_{collection}");
                assert_eq!(
                    index
                        .chunk_lengths
                        .get_key_value(&chunk_id)
                        .unwrap()
                        .0
                        .as_ptr(),
                    chunk_pointers[collection - 1]
                );
                let posting = &index.postings["shared"][collection - 1];
                assert_eq!(posting.chunk_id.as_ptr(), posting_pointers[collection - 1]);
                let collection_id = format!("col_{collection}");
                assert_eq!(index.collection_chunk_count(&collection_id), 1);
                assert_eq!(
                    index.avg_len_by_collection[&collection_id],
                    ((seed + collection) * 3) as f64
                );
                let hits = index
                    .search(
                        &collection_id,
                        "shared",
                        TextSearchOptions::with_explanations(10),
                        &JsonOperationContext::standalone(),
                    )
                    .unwrap();
                assert_eq!(hits.len(), 1);
                assert_eq!(hits[0].chunk_id, chunk_id);
                let frequency = (seed % 3 + 1) as f64;
                let expected = (4.0_f64 / 3.0).ln() * frequency * 2.2 / (frequency + 1.2);
                assert!((hits[0].score - expected).abs() < 1e-12);
            }
            assert!(index
                .search(
                    "empty",
                    "shared",
                    TextSearchOptions::with_explanations(10),
                    &JsonOperationContext::standalone()
                )
                .unwrap()
                .is_empty());
        }
    }

    #[test]
    fn tokenizer_normalizes_code_without_claiming_cjk_bigrams() {
        assert_eq!(
            tokenize("WAL recovery_01!"),
            vec!["wal", "recovery_01", "recovery", "01"]
        );
        assert_eq!(tokenize("本地检索"), vec!["本地检索"]);
    }
}
