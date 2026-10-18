use super::*;

#[test]
fn zero_weight_limits_preserve_union_admission_and_actual_fusion_boundaries() {
    let builder = JsonBuilder::host();
    let limits = CandidateLimits {
        bm25: 50,
        vector: 50,
        fusion: 50,
    };
    for bm25_limited in [true, false] {
        for weight in [0.0, -0.0, f64::EPSILON, 0.5] {
            let weights = if bm25_limited {
                SearchWeights::new(weight, 1.0 - weight, 0.0).unwrap()
            } else {
                SearchWeights::new(1.0 - weight, weight, 0.0).unwrap()
            };
            for admitted_rank in [None, Some(2), Some(51)] {
                let ranks = WhyNotRanks {
                    bm25_rank: Some(1),
                    bm25_retrieval_rank: Some(if bm25_limited { 51 } else { 2 }),
                    vector_rank: Some(if bm25_limited { 2 } else { 51 }),
                    fusion_rank: Some(1),
                    fusion_retrieval_rank: admitted_rank,
                    candidate_limit_improves: admitted_rank.is_none(),
                    ..Default::default()
                };
                for (returned, allowed) in [(None, true), (Some(1), true), (None, false)] {
                    let expected = if returned.is_some() || !allowed {
                        None
                    } else if admitted_rank == Some(51) {
                        Some("fusion")
                    } else if admitted_rank.is_some() {
                        None
                    } else {
                        Some(if bm25_limited { "bm25" } else { "vector" })
                    };
                    let analysis = why_not_limit_analysis(
                        &ranks, limits, weights, returned, allowed, &builder,
                    )
                    .unwrap();
                    assert_eq!(
                        analysis.get("limiting_stage").unwrap().as_str(),
                        expected,
                        "{bm25_limited}/{weight}/{admitted_rank:?}/{returned:?}/{allowed}"
                    );
                    assert_eq!(
                        analysis
                            .get("missed_due_to_candidate_limit")
                            .unwrap()
                            .as_bool(),
                        Some(expected.is_some() && admitted_rank.is_none())
                    );
                    assert_eq!(
                        analysis.get("limiting_parameter").unwrap().as_str(),
                        expected.map(|stage| match stage {
                            "bm25" => "bm25_candidates",
                            "vector" => "vector_candidates",
                            _ => "fusion_candidates",
                        })
                    );
                    assert_eq!(
                        analysis.get("effective_candidate_limit").unwrap().as_f64(),
                        expected.map(|_| 50.0)
                    );
                }
            }
        }
    }
}
