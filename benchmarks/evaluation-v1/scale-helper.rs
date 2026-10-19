use cofferdb::{
    parse_json, Database, DatabaseOpenOptions, QueryEmbeddingInput, ResourceProfile, SearchResults,
};
use std::{
    error::Error,
    io::{self, Write},
    path::Path,
    time::Instant,
};

fn verify(result: &SearchResults, mode: &str) {
    let expected: Vec<_> = (76400..76410).map(|n| format!("chunk-{n}")).collect();
    assert_eq!(
        result
            .results
            .iter()
            .map(|r| &r.chunk_external_id)
            .collect::<Vec<_>>(),
        expected.iter().collect::<Vec<_>>(),
        "{mode}"
    );
    assert!(
        result.results.iter().all(|r| r.final_score == 1.0),
        "{mode}"
    );
    assert_eq!(result.query_context.index_source, "persistent_checkpoint");
    let scored = match mode {
        "text" => 0,
        "hybrid-filtered" => 5000,
        _ => 100000,
    };
    let scan = if mode == "hybrid-filtered" {
        100000
    } else {
        scored
    };
    assert_eq!(
        result.query_context.execution_stats.vector_scan_count, scan,
        "{mode}"
    );
    assert_eq!(
        result.query_context.execution_stats.distance_ops,
        scored * 768,
        "{mode}"
    );
}

fn emit(row: &str) -> io::Result<()> {
    let mut stdout = io::stdout().lock();
    writeln!(stdout, "{row}")?;
    stdout.flush()
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() < 3 || args.len() > 4 || args.len() == 4 && args[3] != "--smoke" {
        return Err("usage: scale-helper <database> <profile> [--smoke]".into());
    }
    let smoke = args.len() == 4;
    let database = Path::new(&args[1]);
    let profile = ResourceProfile::parse(&args[2])
        .ok_or("profile must be expanded, balanced, or constrained")?;
    {
        for mode in ["text", "vector", "hybrid", "hybrid-filtered"] {
            for standard_json in [false, true] {
                let scope = if standard_json {
                    "rust_search_standard_json"
                } else {
                    "rust_search_execute"
                };
                let opened = Instant::now();
                let db = Database::open_with_options(
                    database,
                    DatabaseOpenOptions {
                        read_only: true,
                        resource_profile: Some(profile),
                        ..DatabaseOpenOptions::default()
                    },
                )?;
                let open_ms = opened.elapsed().as_secs_f64() * 1000.0;
                let collection = db.collection("lifecycle")?;
                let stats = collection.stats()?;
                assert_eq!(
                    (
                        stats.document_count,
                        stats.chunk_count,
                        stats.embedding_count
                    ),
                    (1000, 100000, 100000)
                );
                let mut query = collection.search().top_k(10).resource_profile(profile);
                if mode != "vector" {
                    query = query.text("764");
                }
                if mode != "text" {
                    let mut vector = vec![0.0; 768];
                    vector[0] = 1.0;
                    query = query.query_embedding(QueryEmbeddingInput {
                        model_name: Some("synthetic-lifecycle-v1".into()),
                        dim: 768,
                        vector,
                    });
                }
                if mode == "hybrid-filtered" {
                    query = query.filter(parse_json("{\"doc.project\":4}")?);
                }
                emit(&format!("{{\"stage\":\"open\",\"profile\":\"{}\",\"mode\":\"{mode}\",\"scope\":\"{scope}\",\"elapsed_ms\":{open_ms},\"collection\":{}}}", profile.as_json_str(), stats.to_json().to_compact_string()))?;
                let phases = if smoke {
                    [("smoke", 1), ("warmup", 0), ("measured", 0)]
                } else {
                    [("first_query", 1), ("warmup", 2), ("measured", 20)]
                };
                for (phase, count) in phases {
                    for round in 1..=count {
                        let call = query.clone();
                        let started = Instant::now();
                        let result = call.execute()?;
                        let rendered = standard_json.then(|| result.to_json().to_pretty_string());
                        if let Some(wire) = &rendered {
                            std::hint::black_box(wire);
                        }
                        let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
                        verify(&result, mode);
                        emit(&format!("{{\"stage\":\"query\",\"profile\":\"{}\",\"mode\":\"{mode}\",\"scope\":\"{scope}\",\"phase\":\"{phase}\",\"round\":{round},\"elapsed_ms\":{elapsed_ms},\"verified\":true,\"vector_scan_count\":{},\"distance_ops\":{},\"fusion_candidate_count\":{},\"candidate_limits\":{},\"index_source\":\"{}\",\"serialized_bytes\":{}}}",
                            profile.as_json_str(), result.query_context.execution_stats.vector_scan_count,
                            result.query_context.execution_stats.distance_ops, result.query_context.execution_stats.fusion_candidate_count,
                            format!("{{\"bm25\":{},\"vector\":{},\"fusion\":{}}}", result.query_context.candidate_limits.bm25, result.query_context.candidate_limits.vector, result.query_context.candidate_limits.fusion), result.query_context.index_source,
                            rendered.as_ref().map_or(0, |wire| wire.len())))?;
                    }
                }
            }
        }
    }
    let queries = if smoke { 8 } else { 184 };
    emit(&format!("{{\"stage\":\"complete\",\"groups\":8,\"planned_query_samples\":{queries},\"verified\":true}}"))?;
    Ok(())
}
