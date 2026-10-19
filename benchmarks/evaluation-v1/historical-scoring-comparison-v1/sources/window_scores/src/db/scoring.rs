use crate::catalog::Catalog;
use crate::error::Result;
use crate::index::hybrid::{fuse, CandidateScore};
use crate::index::text::{Bm25Hit, Bm25Hits};
use crate::index::vector::{VectorHit, VectorHits};
use crate::json::{JsonOperationContext, ReservedVec};
use crate::metadata::{FilterExpr, MetadataBoostConfig};
use crate::model::SearchWeights;

pub(super) fn filter_candidates(
    catalog: &Catalog,
    bm25_hits: &mut Bm25Hits,
    vector_hits: &mut VectorHits,
    filter: Option<&FilterExpr>,
) {
    if let Some(filter) = filter {
        let matches = |chunk_id: &str| {
            catalog
                .chunks
                .get(chunk_id)
                .is_some_and(|chunk| filter.matches_for_chunk(catalog, chunk))
        };
        bm25_hits.retain(|hit| matches(&hit.chunk_id));
        vector_hits.retain(|hit| matches(&hit.chunk_id));
    }
}

// Retrieval limits remain caller-owned: diagnostics deliberately pass full candidates.
pub(super) fn fuse_candidates<'a>(
    catalog: &Catalog,
    bm25_hits: &'a [Bm25Hit],
    vector_hits: &'a [VectorHit],
    boost: Option<&MetadataBoostConfig>,
    weights: SearchWeights,
    fusion_limit: usize,
    context: &JsonOperationContext,
) -> Result<ReservedVec<CandidateScore<'a>>> {
    fuse(
        bm25_hits,
        vector_hits,
        weights,
        fusion_limit,
        context,
        |id| {
            boost
                .and_then(|config| {
                    catalog
                        .chunks
                        .get(id)
                        .map(|chunk| config.score_for_chunk(catalog, chunk))
                })
                .unwrap_or_default()
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json::{parse, JsonValue};
    use crate::model::Chunk;
    use std::collections::BTreeMap;

    #[test]
    fn candidate_pipeline_matches_independent_filter_and_score_oracle() {
        let mut catalog = Catalog::default();
        for i in 0..9 {
            let id = format!("c{i}");
            catalog.chunks.insert(
                id.clone(),
                Box::new(Chunk {
                    chunk_id: id.clone(),
                    chunk_external_id: id,
                    collection_id: "collection".into(),
                    document_id: "document".into(),
                    text: String::new(),
                    text_hash: String::new(),
                    ordinal: i,
                    token_count: 0,
                    metadata: parse(&format!(
                        r#"{{"keep":{},"boost":{}}}"#,
                        i % 2 == 0,
                        i % 3 == 0
                    ))
                    .unwrap(),
                    embedding_id: None,
                    source_span: None,
                    created_at: String::new(),
                }),
            );
        }
        let boost = MetadataBoostConfig::parse(parse(r#"{"schema_version":"0.1","combine":"sum","max_boost":0.3,"rules":[{"when":{"chunk.boost":true},"boost":0.25},{"when":{"chunk.keep":true},"boost":0.125}]}"#).unwrap()).unwrap();
        for seed in 0..100 {
            let filter = FilterExpr::parse(match seed % 4 {
                0 => JsonValue::object(),
                1 => parse(r#"{"chunk.keep":true}"#).unwrap(),
                2 => parse(r#"{"chunk.keep":false}"#).unwrap(),
                _ => parse(r#"{"chunk.absent":"value"}"#).unwrap(),
            })
            .unwrap();
            let weights = match seed % 5 {
                0 => SearchWeights::new(1.0, 0.0, 0.0),
                1 => SearchWeights::new(0.0, 1.0, 0.0),
                2 => SearchWeights::new(0.25, 0.5, 0.25),
                3 => SearchWeights::new(0.0, 0.0, 1.0),
                _ => SearchWeights::new(0.5, 0.5, 0.0),
            }
            .unwrap();
            let boost_enabled = seed % 3 != 0;
            let included = |i| match seed % 4 {
                0 => true,
                1 => i % 2 == 0,
                2 => i % 2 != 0,
                _ => false,
            };
            let text_score = |i| ((seed + i * 3) % 11) as f64;
            let vector_score = |i| ((seed + i) % 9) as f64 / 8.0;
            let has_text = |i| (seed + i) % 3 != 0;
            let has_vector = |i| (seed + i) % 4 != 0;
            let mut bm25 = Bm25Hits::with_capacity(8, &JsonOperationContext::standalone()).unwrap();
            for i in (0..8).filter(|&i| has_text(i)) {
                bm25.push(&format!("c{i}"), text_score(i), std::iter::empty())
                    .unwrap();
            }
            let mut vector = crate::index::vector::VectorCandidates::new(
                8,
                &crate::json::JsonOperationContext::standalone(),
            );
            for i in (0..8).filter(|&i| has_vector(i)) {
                vector
                    .push(&format!("c{i}"), 1.0 - vector_score(i), vector_score(i))
                    .unwrap();
            }
            let mut vector = vector.finish();
            let maximum = (0..8)
                .filter(|&i| included(i) && has_text(i))
                .map(text_score)
                .fold(0.0_f64, f64::max);
            let mut expected = Vec::new();
            let mut expected_counts = BTreeMap::new();
            for i in 0..8 {
                if !included(i) || (!has_text(i) && !has_vector(i)) {
                    continue;
                }
                let bm25_norm = if has_text(i) && maximum > 0.0 {
                    text_score(i) / maximum
                } else {
                    0.0
                };
                let vector_norm = if has_vector(i) { vector_score(i) } else { 0.0 };
                let boost_score = if boost_enabled {
                    let count = usize::from(i % 3 == 0) + usize::from(i % 2 == 0);
                    if count > 0 {
                        expected_counts.insert(format!("c{i}"), count);
                    }
                    ((if i % 3 == 0 { 0.25_f64 } else { 0.0 })
                        + if i % 2 == 0 { 0.125 } else { 0.0 })
                    .min(0.3)
                } else {
                    0.0
                };
                expected.push((
                    format!("c{i}"),
                    weights.bm25 * bm25_norm
                        + weights.vector * vector_norm
                        + weights.metadata * boost_score,
                ));
            }
            expected.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            expected.truncate(seed % 10);
            filter_candidates(&catalog, &mut bm25, &mut vector, filter.as_ref());
            let ranked = fuse_candidates(
                &catalog,
                &bm25,
                &vector,
                boost_enabled.then_some(&boost),
                weights,
                seed % 10,
                &JsonOperationContext::standalone(),
            )
            .unwrap();
            for score in ranked.as_slice() {
                assert_eq!(
                    score.matched_boost_rule_count,
                    expected_counts
                        .get(score.chunk_id)
                        .copied()
                        .unwrap_or_default(),
                    "seed={seed}"
                );
            }
            let actual = ranked
                .as_slice()
                .iter()
                .map(|score| (score.chunk_id.to_owned(), score.final_score))
                .collect::<Vec<_>>();
            assert_eq!(actual, expected, "seed={seed}");
            assert_eq!(
                bm25.len(),
                (0..8).filter(|&i| included(i) && has_text(i)).count()
            );
            assert_eq!(
                vector.len(),
                (0..8).filter(|&i| included(i) && has_vector(i)).count()
            );
        }
    }
}
