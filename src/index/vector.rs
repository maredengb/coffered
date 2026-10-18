use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::catalog::VisibleSnapshot;
use crate::error::Result;
use crate::json::{
    reserve_heap_slot, try_shrink_heap, JsonOperationContext, ReservedString, WorkspaceReservation,
};
use crate::model::DistanceMetric;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct VectorWork {
    pub scans: usize,
    pub distance_ops: usize,
}

impl VectorWork {
    pub(crate) fn add(&mut self, other: Self) {
        self.scans += other.scans;
        self.distance_ops += other.distance_ops;
    }
}

#[derive(Clone, Debug)]
pub struct VectorHit {
    pub chunk_id: String,
    pub raw_score: f64,
    pub normalized_score: f64,
}

impl PartialEq for VectorHit {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for VectorHit {}
impl PartialOrd for VectorHit {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for VectorHit {
    fn cmp(&self, other: &Self) -> Ordering {
        candidate_order(
            self.normalized_score,
            &self.chunk_id,
            other.normalized_score,
            &other.chunk_id,
        )
    }
}

// The largest heap entry is the worst candidate; signed zero remains a tie.
fn candidate_order(left: f64, left_id: &str, right: f64, right_id: &str) -> Ordering {
    let score = if left == right {
        Ordering::Equal
    } else {
        right.total_cmp(&left)
    };
    score.then_with(|| left_id.cmp(right_id))
}

pub(crate) struct VectorCandidates {
    limit: usize,
    heap: BinaryHeap<VectorHit>,
    work: WorkspaceReservation,
}

#[derive(Debug)]
pub(crate) struct VectorHits {
    hits: Vec<VectorHit>,
    work: WorkspaceReservation,
}

impl std::ops::Deref for VectorHits {
    type Target = [VectorHit];

    fn deref(&self) -> &Self::Target {
        &self.hits
    }
}

impl VectorHits {
    pub(crate) fn empty(context: &JsonOperationContext) -> Self {
        VectorCandidates::new(0, context).finish()
    }

    pub(crate) fn sort_by_id(&mut self) {
        self.hits
            .sort_unstable_by(|a, b| a.chunk_id.cmp(&b.chunk_id));
    }

    pub(crate) fn as_slice(&self) -> &[VectorHit] {
        &self.hits
    }

    pub(crate) fn retain(&mut self, mut include: impl FnMut(&VectorHit) -> bool) {
        let mut released = 0;
        self.hits.retain(|hit| {
            let retained = include(hit);
            if !retained {
                released += hit.chunk_id.capacity();
            }
            retained
        });
        self.work.retain(self.work.reserved_bytes() - released);
    }

    pub(crate) fn copy_prefix(&self, limit: usize) -> Result<Self> {
        let mut candidates = VectorCandidates::new(limit.min(self.len()), self.work.context());
        for hit in self.iter().take(limit) {
            candidates.push(&hit.chunk_id, hit.raw_score, hit.normalized_score)?;
        }
        Ok(candidates.finish())
    }
}

impl VectorCandidates {
    pub(crate) fn new(limit: usize, context: &JsonOperationContext) -> Self {
        Self {
            limit,
            heap: BinaryHeap::new(),
            work: context.workspace_reservation(),
        }
    }

    pub(crate) fn push(&mut self, id: &str, raw_score: f64, normalized_score: f64) -> Result<()> {
        if self.limit == 0
            || (self.heap.len() == self.limit
                && self.heap.peek().is_some_and(|worst| {
                    candidate_order(
                        normalized_score,
                        id,
                        worst.normalized_score,
                        &worst.chunk_id,
                    ) != Ordering::Less
                }))
        {
            return Ok(());
        }
        if self.heap.len() == self.limit {
            let mut worst = self.heap.peek_mut().unwrap();
            let before = self.work.reserved_bytes();
            let old_capacity = worst.chunk_id.capacity();
            if id.len() > old_capacity {
                let bytes = before as u128 + id.len() as u128;
                self.work.grow_to_wide(bytes, bytes, "operation_scratch")?;
                let additional = id.len() - worst.chunk_id.len();
                if let Err(error) = worst.chunk_id.try_reserve_exact(additional) {
                    self.work.retain(before);
                    return Err(std::io::Error::other(error.to_string()).into());
                }
                if worst.chunk_id.capacity() != id.len() {
                    drop(std::mem::take(&mut worst.chunk_id));
                    self.work.retain(before - old_capacity);
                    return Err(
                        std::io::Error::other("candidate ID capacity exceeds reservation").into(),
                    );
                }
                self.work.retain(before - old_capacity + id.len());
            }
            worst.chunk_id.clear();
            worst.chunk_id.push_str(id);
            worst.raw_score = raw_score;
            worst.normalized_score = normalized_score;
            return Ok(());
        }
        let copy = match ReservedString::copy(id, self.work.context()) {
            Err(error)
                if error.code() == crate::ErrorCode::ResourceExhausted
                    && self.heap.len() < self.heap.capacity() =>
            {
                if !try_shrink_heap(&mut self.heap, &mut self.work)? {
                    return Err(error);
                }
                ReservedString::copy(id, self.work.context())?
            }
            result => result?,
        };
        let (chunk_id, work) = copy.into_parts();
        let hit = VectorHit {
            chunk_id,
            raw_score,
            normalized_score,
        };
        reserve_heap_slot(&mut self.heap, &mut self.work, self.limit)?;
        self.heap.push(hit);
        self.work.merge(work);
        Ok(())
    }

    pub(crate) fn finish(self) -> VectorHits {
        VectorHits {
            hits: self.heap.into_sorted_vec(),
            work: self.work,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct VectorSearchOptions {
    pub metric: DistanceMetric,
    pub vectors_normalized: bool,
    pub limit: usize,
}

#[derive(Clone, Debug, Default)]
pub struct ExactVectorIndex;

impl ExactVectorIndex {
    #[allow(clippy::too_many_arguments)]
    pub fn search_with_vectors<F, V>(
        &self,
        snapshot: &VisibleSnapshot<'_>,
        collection_id: &str,
        query: &[f32],
        options: VectorSearchOptions,
        include: impl Fn(&str) -> bool,
        context: &JsonOperationContext,
        mut load_vector: F,
    ) -> Result<(VectorHits, VectorWork)>
    where
        F: FnMut(&crate::catalog::VisibleEmbedding<'_>) -> Result<V>,
        V: AsRef<[f32]>,
    {
        let mut hits = VectorCandidates::new(options.limit, context);
        let scorer = VectorScorer::new(query, options.metric);
        let mut work = VectorWork::default();
        for visible in snapshot.live_embeddings() {
            let embedding = visible.embedding;
            if embedding.collection_id != collection_id {
                continue;
            }
            work.scans += 1;
            context.search_cost_add(crate::search_cost::Counter::VectorChecks, 1);
            if !include(&embedding.chunk_id) {
                continue;
            }
            let loaded = load_vector(&visible)?;
            let vector = loaded.as_ref();
            if vector.len() != query.len() {
                continue;
            }
            let raw = scorer.score(vector);
            work.distance_ops += vector.len();
            context.search_cost_add(
                crate::search_cost::Counter::DistanceComponents,
                vector.len() as u64,
            );
            let normalized = normalize_score(raw, options.metric, options.vectors_normalized);
            drop(loaded);
            hits.push(&embedding.chunk_id, raw, normalized)?;
        }
        Ok((hits.finish(), work))
    }
}

// The query norm is invariant across all vectors in one exact scan.
pub(crate) struct VectorScorer<'a> {
    query: &'a [f32],
    metric: DistanceMetric,
    cosine_query_norm: f64,
}

impl<'a> VectorScorer<'a> {
    pub(crate) fn new(query: &'a [f32], metric: DistanceMetric) -> Self {
        Self {
            query,
            metric,
            cosine_query_norm: if metric == DistanceMetric::Cosine {
                vector_norm(query)
            } else {
                0.0
            },
        }
    }

    pub(crate) fn score(&self, vector: &[f32]) -> f64 {
        if self.metric == DistanceMetric::Cosine && vector.len() == self.query.len() {
            cosine_with_query_norm(self.query, vector, self.cosine_query_norm)
        } else {
            raw_score(self.query, vector, self.metric)
        }
    }
}

pub fn raw_score(query: &[f32], vector: &[f32], metric: DistanceMetric) -> f64 {
    match metric {
        DistanceMetric::Cosine => cosine(query, vector),
        DistanceMetric::DotProduct => query
            .iter()
            .zip(vector)
            .map(|(a, b)| *a as f64 * *b as f64)
            .sum::<f64>(),
        DistanceMetric::Euclidean => query
            .iter()
            .zip(vector)
            .map(|(a, b)| {
                let diff = *a as f64 - *b as f64;
                diff * diff
            })
            .sum::<f64>()
            .sqrt(),
    }
}

pub fn normalize_score(raw: f64, metric: DistanceMetric, vectors_normalized: bool) -> f64 {
    match metric {
        DistanceMetric::Cosine => ((raw + 1.0) / 2.0).clamp(0.0, 1.0),
        DistanceMetric::DotProduct if vectors_normalized => ((raw + 1.0) / 2.0).clamp(0.0, 1.0),
        DistanceMetric::DotProduct => raw,
        DistanceMetric::Euclidean => (1.0 / (1.0 + raw)).clamp(0.0, 1.0),
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    cosine_with_query_norm(a, b, vector_norm(&a[..a.len().min(b.len())]))
}

fn vector_norm(vector: &[f32]) -> f64 {
    vector
        .iter()
        .fold(0.0, |norm, value| {
            let value = *value as f64;
            norm + value * value
        })
        .sqrt()
}

fn cosine_with_query_norm(a: &[f32], b: &[f32], query_norm: f64) -> f64 {
    let mut dot = 0.0;
    let mut bn = 0.0;
    for (x, y) in a.iter().zip(b) {
        let x = *x as f64;
        let y = *y as f64;
        dot += x * y;
        bn += y * y;
    }
    if query_norm == 0.0 || bn == 0.0 {
        0.0
    } else {
        dot / (query_norm * bn.sqrt())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prepared_vector_scores_match_original_arithmetic_bit_for_bit() {
        fn original_score(query: &[f32], vector: &[f32], metric: DistanceMetric) -> f64 {
            let sum_identity = std::iter::empty::<f64>().sum::<f64>();
            let (mut dot, mut query_squared, mut vector_squared, mut distance_squared) = (
                if metric == DistanceMetric::DotProduct {
                    sum_identity
                } else {
                    0.0
                },
                0.0,
                0.0,
                sum_identity,
            );
            for (x, y) in query.iter().zip(vector) {
                let (x, y) = (*x as f64, *y as f64);
                dot += x * y;
                query_squared += x * x;
                vector_squared += y * y;
                distance_squared += (x - y) * (x - y);
            }
            match metric {
                DistanceMetric::Cosine if query_squared == 0.0 || vector_squared == 0.0 => 0.0,
                DistanceMetric::Cosine => dot / (query_squared.sqrt() * vector_squared.sqrt()),
                DistanceMetric::DotProduct => dot,
                DistanceMetric::Euclidean => distance_squared.sqrt(),
            }
        }

        let special = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MAX,
            -f32::MAX,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            -f32::from_bits(1),
        ];
        for seed in 0..100 {
            for dim in [0, 1, 2, 3, 7, 31, 768] {
                let query = (0..dim)
                    .map(|i| {
                        if seed % 3 == 0 {
                            special[(seed + i) % special.len()]
                        } else {
                            ((seed * 17 + i * 13) % 257) as f32 / 128.0 - 1.0
                        }
                    })
                    .collect::<Vec<_>>();
                for length in [dim.saturating_sub(1), dim, dim + 1] {
                    let vector = (0..length)
                        .map(|i| {
                            if seed % 3 == 0 {
                                special[(seed * 3 + i * 5) % special.len()]
                            } else {
                                ((seed * 29 + i * 7) % 509) as f32 / 256.0 - 1.0
                            }
                        })
                        .collect::<Vec<_>>();
                    for metric in [
                        DistanceMetric::Cosine,
                        DistanceMetric::DotProduct,
                        DistanceMetric::Euclidean,
                    ] {
                        let expected = original_score(&query, &vector, metric);
                        let scorer = VectorScorer::new(&query, metric);
                        let (actual, allocated) =
                            crate::test_alloc::allocation_bytes(|| scorer.score(&vector));
                        assert_eq!(allocated, 0);
                        assert_eq!(
                            actual.to_bits(),
                            expected.to_bits(),
                            "{seed}/{dim}/{length}/{metric:?}"
                        );
                        assert_eq!(
                            raw_score(&query, &vector, metric).to_bits(),
                            expected.to_bits()
                        );
                    }
                }
            }
        }
        for (query, vector) in [
            ([0.0, -0.0], [1.0, 2.0]),
            ([1.0, 2.0], [0.0, -0.0]),
            ([0.0, -0.0], [0.0, -0.0]),
        ] {
            let scorer = VectorScorer::new(&query, DistanceMetric::Cosine);
            assert_eq!(scorer.score(&vector).to_bits(), 0.0f64.to_bits());
        }
    }

    #[test]
    fn cosine_normalizes_to_unit_interval() {
        let raw = raw_score(&[1.0, 0.0], &[1.0, 0.0], DistanceMetric::Cosine);
        assert!((raw - 1.0).abs() < 1e-9);
        assert_eq!(normalize_score(raw, DistanceMetric::Cosine, false), 1.0);
    }

    #[test]
    fn euclidean_raw_score_is_distance() {
        let raw = raw_score(&[0.0, 0.0], &[3.0, 4.0], DistanceMetric::Euclidean);
        assert!((raw - 5.0).abs() < 1e-9);
        assert!(
            (normalize_score(raw, DistanceMetric::Euclidean, false) - (1.0 / 6.0)).abs() < 1e-9
        );
    }

    #[test]
    fn dot_product_preserves_raw_score_unless_vectors_are_normalized() {
        let raw = raw_score(&[2.0, 0.0], &[3.0, 0.0], DistanceMetric::DotProduct);
        assert_eq!(raw, 6.0);
        assert_eq!(normalize_score(raw, DistanceMetric::DotProduct, false), 6.0);
        assert_eq!(normalize_score(1.0, DistanceMetric::DotProduct, true), 1.0);
        assert_eq!(normalize_score(-1.0, DistanceMetric::DotProduct, true), 0.0);
    }

    #[test]
    fn bounded_top_k_matches_full_sort_for_one_hundred_cases() {
        for case in 0..100 {
            let limit = case % 23;
            let input = (0..513)
                .map(|i| {
                    let score = ((i * 37 + case * 13) % 17) as f64 - 8.0;
                    VectorHit {
                        chunk_id: format!("chk_{i:016}"),
                        raw_score: score,
                        normalized_score: score,
                    }
                })
                .collect::<Vec<_>>();
            let mut expected = input.clone();
            expected.sort_by(|a, b| {
                b.normalized_score
                    .partial_cmp(&a.normalized_score)
                    .unwrap()
                    .then_with(|| a.chunk_id.cmp(&b.chunk_id))
            });
            expected.truncate(limit);
            let context = JsonOperationContext::standalone();
            let mut candidates = VectorCandidates::new(limit, &context);
            for hit in input.into_iter().rev() {
                candidates
                    .push(&hit.chunk_id, hit.raw_score, hit.normalized_score)
                    .unwrap();
                assert!(candidates.heap.len() <= limit);
            }
            let actual = candidates.finish();
            assert_eq!(actual.len(), expected.len());
            for (a, b) in actual.iter().zip(&expected) {
                assert_eq!(a.chunk_id, b.chunk_id, "case={case}");
                assert_eq!(a.raw_score, b.raw_score);
                assert_eq!(a.normalized_score, b.normalized_score);
            }
        }
    }

    #[test]
    fn signed_zero_scores_tie_by_chunk_id() {
        let mut candidates = VectorCandidates::new(1, &JsonOperationContext::standalone());
        for (id, score) in [("b", 0.0), ("a", -0.0)] {
            candidates.push(id, score, score).unwrap();
        }
        assert_eq!(candidates.finish()[0].chunk_id, "a");
    }

    #[test]
    fn candidate_ownership_covers_replacement_sort_filter_and_prefix_copy() {
        use crate::json::WorkLedger;
        for seed in 0..100 {
            let limit = 4096;
            let ledger = std::sync::Arc::new(WorkLedger::with_limits(limit, limit, limit));
            let context = JsonOperationContext::new(ledger.clone());
            let other = JsonOperationContext::new(ledger);
            let count = seed % 17 + 1;
            let mut candidates = VectorCandidates::new(count, &context);
            for ordinal in 0..count {
                candidates
                    .push(&format!("id-{ordinal:016}"), ordinal as f64, ordinal as f64)
                    .unwrap();
            }
            let footprint = |heap: &BinaryHeap<VectorHit>| {
                heap.capacity() * std::mem::size_of::<VectorHit>()
                    + heap
                        .iter()
                        .map(|hit| hit.chunk_id.capacity())
                        .sum::<usize>()
            };
            assert_eq!(
                candidates.work.reserved_bytes(),
                footprint(&candidates.heap)
            );
            let original = candidates.heap.clone().into_sorted_vec();
            let credit = candidates.work.reserved_bytes();
            let mut held = other.workspace_reservation();
            held.grow_to(0, limit, "operation_scratch").unwrap();
            for _ in 0..100 {
                candidates.push("discarded", -100.0, -100.0).unwrap();
            }
            assert_eq!(candidates.work.reserved_bytes(), credit);
            let id = format!("replacement-{seed:016}\u{4e2d}");
            let available = held.reserved_bytes();
            held.retain(available - id.len() + 1);
            let error = candidates.push(&id, 1000.0, 1000.0).unwrap_err();
            assert_eq!(error.code(), crate::ErrorCode::ResourceExhausted);
            let detail = error
                .detail()
                .iter()
                .cloned()
                .collect::<std::collections::BTreeMap<_, _>>();
            assert_eq!(detail["resource"], "memory");
            assert_eq!(detail["operation"], "operation_scratch");
            assert_eq!(detail["limit_bytes"], limit.to_string());
            assert_eq!(detail["requested_bytes"], (limit + 1).to_string());
            assert_eq!(candidates.heap.clone().into_sorted_vec(), original);
            assert_eq!(candidates.work.reserved_bytes(), credit);
            held.retain(held.reserved_bytes() - 1);
            candidates.push(&id, 1000.0, 1000.0).unwrap();
            assert_eq!(
                candidates.work.reserved_bytes(),
                footprint(&candidates.heap)
            );
            let pointer = candidates.heap.as_slice().as_ptr();
            let capacity = candidates.heap.capacity();
            let mut hits = candidates.finish();
            assert_eq!(hits.as_ptr(), pointer);
            assert_eq!(hits.hits.capacity(), capacity);
            assert_eq!(hits[0].chunk_id, id);
            held.grow_to(0, limit, "operation_scratch").unwrap();
            assert_eq!(
                hits.copy_prefix(1).unwrap_err().code(),
                crate::ErrorCode::ResourceExhausted
            );
            assert_eq!(hits.as_ptr(), pointer);
            held.retain(0);
            let prefix = hits.copy_prefix(count.div_ceil(2)).unwrap();
            assert_eq!(prefix.as_slice(), &hits[..count.div_ceil(2)]);
            assert_ne!(prefix[0].chunk_id.as_ptr(), hits[0].chunk_id.as_ptr());
            drop(prefix);
            let before = hits.work.reserved_bytes();
            let released = hits
                .iter()
                .filter(|hit| hit.raw_score < 1000.0)
                .map(|hit| hit.chunk_id.capacity())
                .sum::<usize>();
            hits.retain(|hit| hit.raw_score >= 1000.0);
            assert_eq!(hits.len(), 1);
            assert_eq!(hits.work.reserved_bytes(), before - released);
            assert_eq!(hits.as_ptr(), pointer);
            drop(hits);
            held.grow_to(limit, limit, "operation_scratch").unwrap();
            let mut empty = VectorCandidates::new(0, &context);
            empty.push("zero-limit", 1.0, 1.0).unwrap();
            assert!(empty.finish().is_empty());
        }
    }

    #[test]
    fn candidate_heap_prefers_growth_but_accepts_only_the_minimum_capacity() {
        use crate::json::WorkLedger;
        for seed in 0..100 {
            let id = "x".repeat(seed + 1);
            let required = id.len() + std::mem::size_of::<VectorHit>();
            for short_by in [0, 1] {
                let limit = required - short_by;
                let context = JsonOperationContext::new(std::sync::Arc::new(
                    WorkLedger::with_limits(limit, limit, limit),
                ));
                let mut candidates = VectorCandidates::new(usize::MAX, &context);
                let result = candidates.push(&id, 1.0, 1.0);
                if short_by == 0 {
                    result.unwrap();
                    assert_eq!(candidates.heap.capacity(), 1);
                    assert_eq!(candidates.work.reserved_bytes(), required);
                    assert_eq!(candidates.finish()[0].chunk_id, id);
                } else {
                    let error = result.unwrap_err();
                    let detail = error
                        .detail()
                        .iter()
                        .cloned()
                        .collect::<std::collections::BTreeMap<_, _>>();
                    assert_eq!(error.code(), crate::ErrorCode::ResourceExhausted);
                    assert_eq!(detail["limit_bytes"], limit.to_string());
                    assert_eq!(detail["requested_bytes"], required.to_string());
                    assert_eq!(candidates.work.reserved_bytes(), 0);
                    assert!(candidates.finish().is_empty());
                }
                context
                    .workspace_reservation()
                    .grow_to(limit, limit, "operation_scratch")
                    .unwrap();
            }
        }
    }

    fn replacement_candidates(context: &JsonOperationContext, id_bytes: usize) -> VectorCandidates {
        let mut candidates = VectorCandidates::new(3, context);
        for (letter, score) in [('a', 3.0), ('b', 2.0), ('z', 1.0)] {
            candidates
                .push(&letter.to_string().repeat(id_bytes), score, score)
                .unwrap();
        }
        assert_eq!(candidates.heap.capacity(), 3);
        assert!(candidates
            .heap
            .iter()
            .all(|hit| hit.chunk_id.capacity() == id_bytes));
        candidates.push("w", -3.0, 1.5).unwrap();
        assert_eq!(candidates.heap.peek().unwrap().chunk_id.len(), 1);
        candidates
    }

    fn candidate_state(heap: &BinaryHeap<VectorHit>) -> Vec<(String, u64, u64, usize, usize)> {
        heap.iter()
            .map(|hit| {
                (
                    hit.chunk_id.clone(),
                    hit.raw_score.to_bits(),
                    hit.normalized_score.to_bits(),
                    hit.chunk_id.capacity(),
                    hit.chunk_id.as_ptr() as usize,
                )
            })
            .collect()
    }

    #[test]
    fn candidate_replacement_admits_both_string_buffers_with_other_live_owners() {
        use crate::json::WorkLedger;
        use std::sync::Arc;
        let size = std::mem::size_of::<VectorHit>();
        for seed in 0..100usize {
            let old = 20 + seed;
            let id = "r".repeat(old + 19);
            let prefix_id = "p".repeat(7 + seed);
            let active = 13 + seed;
            let prefix_bytes = size + prefix_id.len();
            let before = 3 * (size + old);
            let needed = active + prefix_bytes + before + id.len();
            for parent in [false, true] {
                for delta in [-1isize, 0, 1] {
                    let limit = (needed as isize + delta) as usize;
                    let ledger = Arc::new(WorkLedger::with_limits(
                        if parent { 2 * needed } else { limit },
                        0,
                        if parent { limit } else { 2 * needed },
                    ));
                    let context = JsonOperationContext::new(ledger.clone());
                    let other = JsonOperationContext::new(ledger);
                    let mut outside = other.workspace_reservation();
                    outside
                        .grow_to(active, active, "operation_scratch")
                        .unwrap();
                    let mut source = VectorCandidates::new(1, &context);
                    source.push(&prefix_id, 8.0, 8.0).unwrap();
                    let prefix = source.finish().copy_prefix(1).unwrap();
                    assert_eq!(prefix.work.reserved_bytes(), prefix_bytes);
                    let mut candidates = replacement_candidates(&context, old);
                    assert_eq!(candidates.work.reserved_bytes(), before);
                    let previous = candidate_state(&candidates.heap);
                    let array = candidates.heap.as_slice().as_ptr();
                    let result = candidates.push(&id, 44.0, 4.0);
                    let resident = if delta < 0 {
                        let error = result.unwrap_err();
                        assert_eq!(error.code(), crate::ErrorCode::ResourceExhausted);
                        for (key, value) in [
                            ("resource", "memory".into()),
                            ("operation", "operation_scratch".into()),
                            ("limit_bytes", limit.to_string()),
                            ("requested_bytes", needed.to_string()),
                        ] {
                            assert!(error.detail().contains(&(key.into(), value)));
                        }
                        assert_eq!(candidate_state(&candidates.heap), previous);
                        before
                    } else {
                        result.unwrap();
                        before - old + id.len()
                    };
                    assert_eq!(candidates.heap.as_slice().as_ptr(), array);
                    assert_eq!(candidates.heap.capacity(), 3);
                    assert_eq!(candidates.work.reserved_bytes(), resident);
                    let hits = candidates.finish();
                    assert_eq!(hits.work.reserved_bytes(), resident);
                    assert_eq!(hits.as_ptr(), array);
                    let expected = if delta < 0 {
                        vec![
                            ("a".repeat(old), 3.0f64, 3.0f64),
                            ("b".repeat(old), 2.0, 2.0),
                            ("w".into(), -3.0, 1.5),
                        ]
                    } else {
                        vec![
                            (id.clone(), 44.0, 4.0),
                            ("a".repeat(old), 3.0, 3.0),
                            ("b".repeat(old), 2.0, 2.0),
                        ]
                    };
                    assert_eq!(
                        hits.iter()
                            .map(|hit| (
                                hit.chunk_id.as_str(),
                                hit.raw_score.to_bits(),
                                hit.normalized_score.to_bits(),
                            ))
                            .collect::<Vec<_>>(),
                        expected
                            .iter()
                            .map(|(id, raw, normalized)| (
                                id.as_str(),
                                raw.to_bits(),
                                normalized.to_bits(),
                            ))
                            .collect::<Vec<_>>()
                    );
                    assert_eq!(
                        context.available_workspace_bytes(),
                        limit - active - prefix_bytes - resident
                    );
                    drop(hits);
                    assert_eq!(
                        context.available_workspace_bytes(),
                        limit - active - prefix_bytes
                    );
                    drop(prefix);
                    drop(outside);
                    assert_eq!(context.available_workspace_bytes(), limit);
                    let mut all = context.workspace_reservation();
                    all.grow_to(limit, limit, "operation_scratch").unwrap();
                    assert_eq!(
                        other
                            .workspace_reservation()
                            .grow_to(1, 1, "operation_scratch")
                            .unwrap_err()
                            .code(),
                        crate::ErrorCode::ResourceExhausted
                    );
                }
            }
        }
    }

    #[test]
    fn candidate_replacement_allocation_failure_preserves_owner_and_can_retry() {
        use crate::json::WorkLedger;
        use std::sync::Arc;
        let size = std::mem::size_of::<VectorHit>();
        for seed in 0..100usize {
            let old = 20 + seed;
            let id = "r".repeat(old + 19);
            let active = 13 + seed;
            let before = 3 * (size + old);
            let needed = active + before + id.len();
            let context =
                JsonOperationContext::new(Arc::new(WorkLedger::with_limits(needed, 0, needed)));
            let mut outside = context.workspace_reservation();
            outside
                .grow_to(active, active, "operation_scratch")
                .unwrap();
            let mut candidates = replacement_candidates(&context, old);
            let previous = candidate_state(&candidates.heap);
            let array = candidates.heap.as_slice().as_ptr();
            let error = crate::test_alloc::fail_byte_allocation_once(id.len(), || {
                candidates.push(&id, 44.0, 4.0)
            })
            .unwrap_err();
            assert_eq!(error.code(), crate::ErrorCode::IoError);
            assert_eq!(candidate_state(&candidates.heap), previous);
            assert_eq!(candidates.heap.as_slice().as_ptr(), array);
            assert_eq!(candidates.work.reserved_bytes(), before);
            assert_eq!(context.available_workspace_bytes(), id.len());
            let (result, allocated) =
                crate::test_alloc::key_allocations(id.len(), || candidates.push(&id, 44.0, 4.0));
            result.unwrap();
            assert_eq!(allocated.allocated, 1);
            assert!(!allocated.overflow);
            assert_eq!(candidates.heap.as_slice().as_ptr(), array);
            assert_eq!(candidates.work.reserved_bytes(), before - old + id.len());
            assert_eq!(context.available_workspace_bytes(), old);
            let hits = candidates.finish();
            assert_eq!(hits[0].chunk_id, id);
            assert_eq!(hits[0].raw_score.to_bits(), 44.0f64.to_bits());
            assert_eq!(hits[0].normalized_score.to_bits(), 4.0f64.to_bits());
            assert_eq!(context.available_workspace_bytes(), old);
            drop(hits);
            drop(outside);
            assert_eq!(context.available_workspace_bytes(), needed);
        }
    }

    #[test]
    fn candidate_replacement_reuses_existing_capacity_with_zero_headroom() {
        use crate::json::WorkLedger;
        use std::sync::Arc;
        let size = std::mem::size_of::<VectorHit>();
        for seed in 0..100usize {
            let old = 20 + seed;
            let limit = 3 * (size + old);
            let context =
                JsonOperationContext::new(Arc::new(WorkLedger::with_limits(limit, 0, limit)));
            let mut candidates = replacement_candidates(&context, old);
            assert_eq!(context.available_workspace_bytes(), 0);
            let pointer = candidates.heap.peek().unwrap().chunk_id.as_ptr();
            for (length, normalized) in [(old, 1.6), (1, 1.7), (old - 1, 1.8)] {
                let id = "s".repeat(length);
                let (result, allocated) = crate::test_alloc::allocations(|| {
                    candidates.push(&id, -normalized, normalized)
                });
                result.unwrap();
                assert_eq!(allocated, 0);
                let worst = candidates.heap.peek().unwrap();
                assert_eq!(worst.chunk_id, id);
                assert_eq!(worst.chunk_id.as_ptr(), pointer);
                assert_eq!(worst.chunk_id.capacity(), old);
                assert_eq!(worst.raw_score.to_bits(), (-normalized).to_bits());
                assert_eq!(worst.normalized_score.to_bits(), normalized.to_bits());
                assert_eq!(candidates.work.reserved_bytes(), limit);
                assert_eq!(context.available_workspace_bytes(), 0);
            }
            let discarded = "x".repeat(old * 3);
            let previous = candidate_state(&candidates.heap);
            let (result, allocated) =
                crate::test_alloc::allocations(|| candidates.push(&discarded, 0.0, 0.0));
            result.unwrap();
            assert_eq!(allocated, 0);
            assert_eq!(candidate_state(&candidates.heap), previous);
            let hits = candidates.finish();
            assert_eq!(hits[2].normalized_score.to_bits(), 1.8f64.to_bits());
            assert_eq!(hits[2].chunk_id, "s".repeat(old - 1));
            assert_eq!(context.available_workspace_bytes(), 0);
            drop(hits);
            assert_eq!(context.available_workspace_bytes(), limit);
        }
    }

    #[test]
    fn candidate_second_allocation_admits_overlap_and_preserves_owned_ids_on_refusal() {
        use crate::json::WorkLedger;
        use std::sync::Arc;
        let size = std::mem::size_of::<VectorHit>();
        for seed in 0..100usize {
            let id = "x".repeat(20 + seed);
            let next = "y".repeat(id.len());
            let active = 13 + seed;
            let needed = active + 2 * id.len() + 3 * size;
            for parent in [false, true] {
                for delta in [-1isize, 0, 1] {
                    let limit = (needed as isize + delta) as usize;
                    let ledger = Arc::new(WorkLedger::with_limits(
                        if parent { 2 * needed } else { limit },
                        0,
                        if parent { limit } else { 2 * needed },
                    ));
                    let context = JsonOperationContext::new(ledger.clone());
                    let other = JsonOperationContext::new(ledger);
                    let mut outside = other.workspace_reservation();
                    outside
                        .grow_to(active, active, "operation_scratch")
                        .unwrap();
                    let mut blocker = other.workspace_reservation();
                    let blocked = limit - active - id.len() - size;
                    blocker
                        .grow_to(blocked, blocked, "operation_scratch")
                        .unwrap();
                    let mut candidates = VectorCandidates::new(2, &context);
                    candidates.push(&id, 1.0, 1.0).unwrap();
                    assert_eq!(candidates.heap.capacity(), 1);
                    let pointer = candidates.heap.as_slice().as_ptr();
                    blocker.retain(0);
                    let result = candidates.push(&next, 2.0, 2.0);
                    let count = if delta < 0 {
                        let error = result.unwrap_err();
                        assert_eq!(error.code(), crate::ErrorCode::ResourceExhausted);
                        for (key, value) in [
                            ("resource", "memory".into()),
                            ("operation", "operation_scratch".into()),
                            ("limit_bytes", limit.to_string()),
                            ("requested_bytes", needed.to_string()),
                        ] {
                            assert!(error.detail().contains(&(key.into(), value)));
                        }
                        assert_eq!(candidates.heap.capacity(), 1);
                        assert_eq!(candidates.heap.as_slice().as_ptr(), pointer);
                        1
                    } else {
                        result.unwrap();
                        assert_eq!(candidates.heap.capacity(), 2);
                        2
                    };
                    let resident = count * (size + id.len());
                    assert_eq!(candidates.work.reserved_bytes(), resident);
                    assert_eq!(
                        context.available_workspace_bytes(),
                        limit - active - resident
                    );
                    let hits = candidates.finish();
                    let expected = if count == 1 {
                        vec![(&id, 1.0f64)]
                    } else {
                        vec![(&next, 2.0), (&id, 1.0)]
                    };
                    assert_eq!(
                        hits.iter()
                            .map(|hit| (&hit.chunk_id, hit.raw_score.to_bits()))
                            .collect::<Vec<_>>(),
                        expected
                            .into_iter()
                            .map(|(id, score)| (id, score.to_bits()))
                            .collect::<Vec<_>>()
                    );
                    assert_eq!(
                        context.available_workspace_bytes(),
                        limit - active - resident
                    );
                    drop(hits);
                    assert_eq!(context.available_workspace_bytes(), limit - active);
                    drop(outside);
                    assert_eq!(context.available_workspace_bytes(), limit);
                }
            }
        }
    }

    #[test]
    fn candidate_spare_recovery_admits_overlap_before_payload_and_heap_growth() {
        use crate::json::WorkLedger;
        use std::sync::Arc;
        let size = std::mem::size_of::<VectorHit>();
        for seed in 0..100usize {
            let old = "o".repeat(20 + seed);
            let active = 13 + seed;
            let before = 4 * size + old.len();
            for (stage, available, length) in [
                (0, size - 1, 2 * size),
                (1, size, 5 * size),
                (2, size, 3 * size + 1),
                (3, size, 2 * size),
            ] {
                let id = "n".repeat(length);
                let limit = active + before + available;
                let context =
                    JsonOperationContext::new(Arc::new(WorkLedger::with_limits(limit, 0, limit)));
                let mut outside = context.workspace_reservation();
                outside
                    .grow_to(active, active, "operation_scratch")
                    .unwrap();
                let mut candidates = VectorCandidates::new(usize::MAX, &context);
                candidates.push(&old, 1.25, 1.0).unwrap();
                assert_eq!(candidates.heap.capacity(), 4);
                assert_eq!(candidates.work.reserved_bytes(), before);
                let pointer = candidates.heap.peek().unwrap().chunk_id.as_ptr();
                let array = candidates.heap.as_slice().as_ptr();
                let result = candidates.push(&id, 2.25, 2.0);
                let resident = if stage == 3 {
                    result.unwrap();
                    assert_eq!(candidates.heap.capacity(), 2);
                    2 * size + old.len() + id.len()
                } else {
                    let error = result.unwrap_err();
                    assert_eq!(error.code(), crate::ErrorCode::ResourceExhausted);
                    let retained = if stage == 0 { before } else { size + old.len() };
                    let requested =
                        active + retained + id.len() + if stage == 2 { 2 * size } else { 0 };
                    for (key, value) in [
                        ("resource", "memory".into()),
                        ("operation", "operation_scratch".into()),
                        ("limit_bytes", limit.to_string()),
                        ("requested_bytes", requested.to_string()),
                    ] {
                        assert!(error.detail().contains(&(key.into(), value)));
                    }
                    assert_eq!(candidates.heap.len(), 1);
                    assert_eq!(candidates.heap.capacity(), if stage == 0 { 4 } else { 1 });
                    if stage == 0 {
                        assert_eq!(candidates.heap.as_slice().as_ptr(), array);
                    }
                    retained
                };
                assert_eq!(candidates.work.reserved_bytes(), resident);
                let hits = candidates.finish();
                let old_hit = hits.last().unwrap();
                assert_eq!(old_hit.chunk_id, old);
                assert_eq!(old_hit.chunk_id.as_ptr(), pointer);
                assert_eq!(old_hit.raw_score.to_bits(), 1.25f64.to_bits());
                assert_eq!(old_hit.normalized_score.to_bits(), 1.0f64.to_bits());
                if stage == 3 {
                    assert_eq!(hits[0].chunk_id, id);
                    assert_eq!(hits[0].raw_score.to_bits(), 2.25f64.to_bits());
                    assert_eq!(hits[0].normalized_score.to_bits(), 2.0f64.to_bits());
                }
                assert_eq!(
                    context.available_workspace_bytes(),
                    limit - active - resident
                );
                drop(hits);
                drop(outside);
                assert_eq!(context.available_workspace_bytes(), limit);
            }
        }
    }

    #[test]
    fn candidate_pressure_reclaims_spare_slots_and_reuses_replacement_strings() {
        use crate::json::WorkLedger;
        for seed in 0..100 {
            let id = "x".repeat(3 * std::mem::size_of::<VectorHit>() + seed);
            let bytes = std::mem::size_of::<VectorHit>() + id.len();
            let overlap = 2 * std::mem::size_of::<VectorHit>();
            let limit = bytes * 3 + overlap;
            let context = JsonOperationContext::new(std::sync::Arc::new(WorkLedger::with_limits(
                limit, limit, limit,
            )));
            let mut blocker = context.workspace_reservation();
            blocker
                .grow_to(overlap, overlap, "operation_scratch")
                .unwrap();
            let mut candidates = VectorCandidates::new(usize::MAX, &context);
            for score in 0..2 {
                candidates.push(&id, score as f64, score as f64).unwrap();
            }
            assert_eq!(
                candidates.push(&id, 2.0, 2.0).unwrap_err().code(),
                crate::ErrorCode::ResourceExhausted
            );
            assert_eq!(candidates.heap.len(), 2);
            assert_eq!(candidates.heap.capacity(), 2);
            blocker.retain(0);
            candidates.push(&id, 2.0, 2.0).unwrap();
            assert_eq!(candidates.heap.len(), 3);
            assert_eq!(candidates.heap.capacity(), 3);
            assert_eq!(candidates.work.reserved_bytes(), bytes * 3);
            blocker
                .grow_to(overlap, overlap, "operation_scratch")
                .unwrap();
            assert_eq!(
                candidates.push(&id, 3.0, 3.0).unwrap_err().code(),
                crate::ErrorCode::ResourceExhausted
            );
            assert_eq!(
                candidates
                    .finish()
                    .iter()
                    .map(|hit| hit.raw_score)
                    .collect::<Vec<_>>(),
                [2.0, 1.0, 0.0]
            );
            drop(blocker);
            context
                .workspace_reservation()
                .grow_to(limit, limit, "operation_scratch")
                .unwrap();

            let context = JsonOperationContext::new(std::sync::Arc::new(WorkLedger::with_limits(
                bytes, bytes, bytes,
            )));
            let mut candidates = VectorCandidates::new(1, &context);
            candidates.push(&id, 0.0, 0.0).unwrap();
            let pointer = candidates.heap.peek().unwrap().chunk_id.as_ptr();
            for score in 1..100 {
                let replacement = "y".repeat(id.len() - score % id.len());
                candidates
                    .push(&replacement, score as f64, score as f64)
                    .unwrap();
                let hit = candidates.heap.peek().unwrap();
                assert_eq!(hit.chunk_id.as_ptr(), pointer);
                assert_eq!(hit.chunk_id, replacement);
                assert_eq!(candidates.work.reserved_bytes(), bytes);
            }
            assert_eq!(candidates.finish()[0].raw_score, 99.0);
            context
                .workspace_reservation()
                .grow_to(bytes, bytes, "operation_scratch")
                .unwrap();
        }
    }
}
