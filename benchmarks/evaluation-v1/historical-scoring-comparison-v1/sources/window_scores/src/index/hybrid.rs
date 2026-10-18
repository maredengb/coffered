use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::error::Result;
use crate::index::text::Bm25Hit;
use crate::index::vector::VectorHit;
use crate::json::{JsonOperationContext, ReservedVec};
use crate::metadata::MetadataBoostScore;
use crate::model::SearchWeights;

#[derive(Clone, Copy, Debug, Default)]
pub struct CandidateScore<'a> {
    pub chunk_id: &'a str,
    pub bm25_raw: Option<f64>,
    pub bm25_norm: f64,
    pub vector_raw: Option<f64>,
    pub vector_norm: Option<f64>,
    pub metadata_boost: f64,
    pub matched_boost_rule_count: usize,
    pub final_score: f64,
}

impl PartialEq for CandidateScore<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for CandidateScore<'_> {}
impl PartialOrd for CandidateScore<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
// Best-first order leaves the worst retained candidate at the max-heap root.
impl Ord for CandidateScore<'_> {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .final_score
            .partial_cmp(&self.final_score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.chunk_id.cmp(other.chunk_id))
    }
}

pub fn fuse<'a>(
    bm25_hits: &'a [Bm25Hit],
    vector_hits: &'a [VectorHit],
    weights: SearchWeights,
    fusion_limit: usize,
    context: &JsonOperationContext,
    mut metadata_boost: impl FnMut(&str) -> MetadataBoostScore,
) -> Result<ReservedVec<CandidateScore<'a>>> {
    if fusion_limit == 0 {
        return ReservedVec::with_capacity(0, context, "operation_scratch");
    }
    let count = bm25_hits.len() + vector_hits.len();
    let mut indices = ReservedVec::with_capacity(count, context, "operation_scratch")?;
    for index in 0..count {
        indices.push(index);
    }
    let chunk_id = |index: usize| {
        if index < bm25_hits.len() {
            bm25_hits[index].chunk_id.as_str()
        } else {
            vector_hits[index - bm25_hits.len()].chunk_id.as_str()
        }
    };
    // Preserve last-hit-wins within each branch without a stable-sort allocation.
    indices
        .as_mut_slice()
        .sort_unstable_by(|a, b| chunk_id(*a).cmp(chunk_id(*b)).then_with(|| a.cmp(b)));
    let indices = indices.as_slice();
    let unique = indices
        .iter()
        .enumerate()
        .filter(|(offset, index)| {
            *offset == 0 || chunk_id(indices[*offset - 1]) != chunk_id(**index)
        })
        .count();
    let capacity = unique.min(fusion_limit);
    let (storage, work) =
        ReservedVec::with_capacity(capacity, context, "operation_scratch")?.into_parts();
    let mut candidates = BinaryHeap::<CandidateScore<'a>>::from(storage);
    let bm25_max = bm25_hits
        .iter()
        .map(|hit| hit.score)
        .fold(0.0_f64, f64::max)
        .max(0.0);
    let mut offset = 0;
    while offset < indices.len() {
        let id = chunk_id(indices[offset]);
        let mut score = CandidateScore {
            chunk_id: id,
            ..CandidateScore::default()
        };
        while offset < indices.len() && chunk_id(indices[offset]) == id {
            let index = indices[offset];
            if index < bm25_hits.len() {
                let hit = &bm25_hits[index];
                score.bm25_raw = Some(hit.score);
                score.bm25_norm = if bm25_max > 0.0 {
                    (hit.score / bm25_max).clamp(0.0, 1.0)
                } else {
                    0.0
                };
            } else {
                let hit = &vector_hits[index - bm25_hits.len()];
                score.vector_raw = Some(hit.raw_score);
                score.vector_norm = Some(hit.normalized_score);
            }
            offset += 1;
        }
        let boost = metadata_boost(id);
        score.metadata_boost = boost.boost.clamp(0.0, 1.0);
        score.matched_boost_rule_count = boost.matched_rule_count;
        score.final_score = weights.bm25 * score.bm25_norm
            + weights.vector * score.vector_norm.unwrap_or(0.0)
            + weights.metadata * score.metadata_boost;
        if candidates.len() < capacity {
            candidates.push(score);
        } else if candidates.peek().is_some_and(|worst| score < *worst) {
            *candidates.peek_mut().unwrap() = score;
        }
    }
    Ok(ReservedVec::from_reserved(
        candidates.into_sorted_vec(),
        work,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn assert_fields(actual: &CandidateScore<'_>, expected: &CandidateScore<'_>) {
        assert_eq!(actual.chunk_id, expected.chunk_id);
        assert_eq!(
            actual.bm25_raw.map(f64::to_bits),
            expected.bm25_raw.map(f64::to_bits)
        );
        assert_eq!(actual.bm25_norm.to_bits(), expected.bm25_norm.to_bits());
        assert_eq!(
            actual.vector_raw.map(f64::to_bits),
            expected.vector_raw.map(f64::to_bits)
        );
        assert_eq!(
            actual.vector_norm.map(f64::to_bits),
            expected.vector_norm.map(f64::to_bits)
        );
        assert_eq!(
            actual.metadata_boost.to_bits(),
            expected.metadata_boost.to_bits()
        );
        assert_eq!(actual.final_score.to_bits(), expected.final_score.to_bits());
        assert_eq!(
            actual.matched_boost_rule_count,
            expected.matched_boost_rule_count
        );
    }

    #[test]
    fn fusion_preserves_duplicate_branch_order_and_borrows_ids() {
        for seed in 0..100 {
            let count = 1 + seed % 31;
            let text = (0..count * 3)
                .rev()
                .map(|i| Bm25Hit {
                    chunk_id: format!("c{}", (i * 7 + seed) % count),
                    score: ((i + seed) % 23) as f64 - 2.0,
                    ..Default::default()
                })
                .collect::<Vec<_>>();
            let vector = (0..count * 4)
                .map(|i| VectorHit {
                    chunk_id: format!("c{}", (i * 11 + seed) % (count + 2)),
                    raw_score: i as f64,
                    normalized_score: match (i + seed) % 5 {
                        0 => -0.0,
                        1 => 0.0,
                        _ => (i % 7) as f64 - 3.0,
                    },
                })
                .collect::<Vec<_>>();
            let weights = match seed % 4 {
                0 => SearchWeights::new(1.0, 0.0, 0.0),
                1 => SearchWeights::new(0.0, 1.0, 0.0),
                2 => SearchWeights::new(0.0, 0.0, 1.0),
                _ => SearchWeights::new(0.25, 0.5, 0.25),
            }
            .unwrap();
            let maximum = text.iter().map(|hit| hit.score).fold(0.0_f64, f64::max);
            let mut reference = BTreeMap::<&str, CandidateScore<'_>>::new();
            for hit in &text {
                let score = reference.entry(&hit.chunk_id).or_default();
                score.chunk_id = &hit.chunk_id;
                score.bm25_raw = Some(hit.score);
                score.bm25_norm = if maximum > 0.0 {
                    (hit.score / maximum).clamp(0.0, 1.0)
                } else {
                    0.0
                };
            }
            for hit in &vector {
                let score = reference.entry(&hit.chunk_id).or_default();
                score.chunk_id = &hit.chunk_id;
                score.vector_raw = Some(hit.raw_score);
                score.vector_norm = Some(hit.normalized_score);
            }
            let boost = |id: &str| {
                let number = id[1..].parse::<usize>().unwrap();
                MetadataBoostScore {
                    boost: ((number + seed) % 7) as f64 / 4.0 - 0.25,
                    matched_rule_count: number % 3,
                }
            };
            let mut expected = reference
                .into_values()
                .map(|mut score| {
                    let metadata = boost(score.chunk_id);
                    score.metadata_boost = metadata.boost.clamp(0.0, 1.0);
                    score.matched_boost_rule_count = metadata.matched_rule_count;
                    score.final_score = weights.bm25 * score.bm25_norm
                        + weights.vector * score.vector_norm.unwrap_or_default()
                        + weights.metadata * score.metadata_boost;
                    score
                })
                .collect::<Vec<_>>();
            expected.sort_by(|a, b| {
                b.final_score
                    .partial_cmp(&a.final_score)
                    .unwrap()
                    .then_with(|| a.chunk_id.cmp(b.chunk_id))
            });
            for limit in [0, 1, 3, count, usize::MAX] {
                let mut visited = Vec::new();
                let actual = fuse(
                    &text,
                    &vector,
                    weights,
                    limit,
                    &JsonOperationContext::standalone(),
                    |id| {
                        visited.push(id.to_owned());
                        boost(id)
                    },
                )
                .unwrap();
                assert_eq!(actual.as_slice().len(), expected.len().min(limit));
                if limit == 0 {
                    assert!(visited.is_empty());
                } else {
                    let mut ids = expected
                        .iter()
                        .map(|score| score.chunk_id)
                        .collect::<Vec<_>>();
                    ids.sort_unstable();
                    assert_eq!(visited, ids);
                }
                for (actual, expected) in actual.as_slice().iter().zip(&expected) {
                    assert_fields(actual, expected);
                    assert!(text
                        .iter()
                        .map(|hit| &hit.chunk_id)
                        .chain(vector.iter().map(|hit| &hit.chunk_id))
                        .any(|id| id.as_ptr() == actual.chunk_id.as_ptr()
                            && id.len() == actual.chunk_id.len()));
                }
            }
        }
    }

    #[test]
    fn fusion_admits_exact_scratch_and_retains_only_the_ranked_array() {
        use crate::json::WorkLedger;
        for seed in 0..100 {
            let count = 2 + seed % 19;
            let text = (0..count)
                .map(|i| Bm25Hit {
                    chunk_id: format!("{}-{i}", "long-borrowed-id".repeat(seed + 1)),
                    score: i as f64,
                    ..Default::default()
                })
                .collect::<Vec<_>>();
            let vector = text
                .iter()
                .rev()
                .map(|hit| VectorHit {
                    chunk_id: hit.chunk_id.clone(),
                    raw_score: 1.0,
                    normalized_score: 1.0,
                })
                .collect::<Vec<_>>();
            let keep = 1 + seed % count;
            let indices_bytes = (text.len() + vector.len()) * std::mem::size_of::<usize>();
            let result_bytes = keep * std::mem::size_of::<CandidateScore<'_>>();
            let needed = indices_bytes + result_bytes;
            let limit = needed + 73;
            let ledger = std::sync::Arc::new(WorkLedger::with_limits(limit, limit, limit));
            let context = JsonOperationContext::new(ledger.clone());
            let other = JsonOperationContext::new(ledger);
            let mut held = other.workspace_reservation();
            for available in [indices_bytes - 1, needed - 1, needed] {
                held.retain(0);
                held.grow_to(limit - available, limit - available, "operation_scratch")
                    .unwrap();
                let mut visits = 0;
                let result = fuse(
                    &text,
                    &vector,
                    SearchWeights::default_for(true, true),
                    keep,
                    &context,
                    |_| {
                        visits += 1;
                        MetadataBoostScore::default()
                    },
                );
                if available < needed {
                    let error = result.err().unwrap();
                    assert_eq!(error.code(), crate::ErrorCode::ResourceExhausted);
                    let detail = error.detail().iter().cloned().collect::<BTreeMap<_, _>>();
                    assert_eq!(detail["resource"], "memory");
                    assert_eq!(detail["operation"], "operation_scratch");
                    assert_eq!(detail["limit_bytes"], limit.to_string());
                    assert_eq!(detail["requested_bytes"], (limit + 1).to_string());
                    assert_eq!(visits, 0);
                } else {
                    let hits = result.unwrap();
                    assert_eq!(visits, count);
                    assert_eq!(hits.as_slice().len(), keep);
                    let mut spare = other.workspace_reservation();
                    assert_eq!(
                        spare.grow_to(0, limit, "operation_scratch").unwrap(),
                        indices_bytes
                    );
                    let error = spare
                        .grow_to(indices_bytes + 1, indices_bytes + 1, "operation_scratch")
                        .unwrap_err();
                    assert_eq!(error.code(), crate::ErrorCode::ResourceExhausted);
                    drop(hits);
                    assert_eq!(
                        spare.grow_to(needed, needed, "operation_scratch").unwrap(),
                        needed
                    );
                }
                held.grow_to(limit, limit, "operation_scratch").unwrap();
                assert!(fuse(
                    &text,
                    &vector,
                    SearchWeights::default_for(true, true),
                    0,
                    &context,
                    |_| panic!("zero limit must not evaluate boosts")
                )
                .unwrap()
                .as_slice()
                .is_empty());
                assert!(fuse(
                    &[],
                    &[],
                    SearchWeights::default_for(true, true),
                    usize::MAX,
                    &context,
                    |_| panic!("empty input must not evaluate boosts")
                )
                .unwrap()
                .as_slice()
                .is_empty());
            }
        }
    }

    #[test]
    fn bounded_fusion_matches_independent_scores_and_full_order_across_100_seeds() {
        for seed in 0..100 {
            let count = seed % 41;
            let ids = (0..count)
                .map(|id| format!("chunk-{id}"))
                .collect::<Vec<_>>();
            let weights = match seed % 4 {
                0 => SearchWeights::new(1.0, 0.0, 0.0),
                1 => SearchWeights::new(0.0, 1.0, 0.0),
                2 => SearchWeights::new(0.0, 0.0, 1.0),
                _ => SearchWeights::new(0.25, 0.5, 0.25),
            }
            .unwrap();
            let has_text = |id| (id + seed) % 3 != 0;
            let has_vector = |id| (id + seed) % 4 != 0;
            let has_boost = |id| (id + seed) % 5 != 0;
            let text_score = |id| ((id * 7 + seed) % 9) as f64;
            let vector_score = |id| ((id * 3 + seed) % 9) as f64 / 8.0;
            let boost_score = |id| ((id + seed) % 7) as f64 / 4.0 - 0.25;
            let text = (0..count)
                .rev()
                .filter(|&id| has_text(id))
                .map(|id| Bm25Hit {
                    chunk_id: format!("chunk-{id}"),
                    score: text_score(id),
                    ..Default::default()
                })
                .collect::<Vec<_>>();
            let vector = (0..count)
                .filter(|&id| has_vector(id))
                .map(|id| VectorHit {
                    chunk_id: format!("chunk-{id}"),
                    raw_score: 2.0 * vector_score(id) - 1.0,
                    normalized_score: vector_score(id),
                })
                .collect::<Vec<_>>();
            let boosts: BTreeMap<_, _> = (0..count)
                .filter(|&id| has_boost(id))
                .map(|id| (format!("chunk-{id}"), boost_score(id)))
                .collect();
            let maximum = (0..count)
                .filter(|&id| has_text(id))
                .map(text_score)
                .fold(0.0_f64, f64::max);
            let mut expected = (0..count)
                .filter(|&id| has_text(id) || has_vector(id))
                .map(|id| {
                    let bm25_norm = if has_text(id) && maximum > 0.0 {
                        text_score(id) / maximum
                    } else {
                        0.0
                    };
                    let vector_norm = has_vector(id).then(|| vector_score(id));
                    let metadata_boost = if has_boost(id) {
                        boost_score(id).clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    CandidateScore {
                        chunk_id: &ids[id],
                        bm25_raw: has_text(id).then(|| text_score(id)),
                        bm25_norm,
                        vector_raw: has_vector(id).then(|| 2.0 * vector_score(id) - 1.0),
                        vector_norm,
                        metadata_boost,
                        matched_boost_rule_count: usize::from(has_boost(id)),
                        final_score: weights.bm25 * bm25_norm
                            + weights.vector * vector_norm.unwrap_or(0.0)
                            + weights.metadata * metadata_boost,
                    }
                })
                .collect::<Vec<_>>();
            expected.sort_by(|a, b| {
                b.final_score
                    .total_cmp(&a.final_score)
                    .then_with(|| a.chunk_id.cmp(b.chunk_id))
            });
            for limit in [0, 1, 7, count, usize::MAX] {
                let actual = fuse(
                    &text,
                    &vector,
                    weights,
                    limit,
                    &JsonOperationContext::standalone(),
                    |id| MetadataBoostScore {
                        boost: boosts.get(id).copied().unwrap_or_default(),
                        matched_rule_count: usize::from(boosts.contains_key(id)),
                    },
                )
                .unwrap();
                let actual = actual.as_slice();
                assert_eq!(actual.len(), expected.len().min(limit), "seed={seed}");
                for (actual, expected) in actual.iter().zip(&expected) {
                    assert_fields(actual, expected);
                }
            }
        }
    }
}
