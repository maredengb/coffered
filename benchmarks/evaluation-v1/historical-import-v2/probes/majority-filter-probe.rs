fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    use std::{io::Write, time::Instant};
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(args.len(), 2);
    let db = Database::open_read_only(&args[1])?;
    let collection = db.collection("lifecycle")?;
    let stats = collection.stats()?;
    assert_eq!((stats.document_count, stats.chunk_count, stats.embedding_count), (1000, 100000, 100000));
    let mut vector = vec![0.0; 768]; vector[0] = 1.0;
    let query = collection.search().text("764").top_k(10)
        .filter(parse_json("{\"doc.project\":{\"$ne\":0}}")?)
        .query_embedding(QueryEmbeddingInput { model_name: Some("synthetic-lifecycle-v1".into()), dim: 768, vector });
    for (phase, count) in [("first_query", 1), ("warmup", 2), ("measured", 20)] {
        for sample in 1..=count {
            println!("{{\"event\":\"query_started\",\"phase\":\"{phase}\",\"sample\":{sample}}}");
            std::io::stdout().flush()?;
            let call = query.clone();
            let started = Instant::now();
            let result = call.execute()?;
            let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
            let expected: Vec<_> = (76400..76410).map(|n| format!("chunk-{n}")).collect();
            assert_eq!(result.results.iter().map(|r| &r.chunk_external_id).collect::<Vec<_>>(), expected.iter().collect::<Vec<_>>());
            assert!(result.results.iter().all(|r| r.final_score == 1.0));
            assert_eq!(result.query_context.index_source, "persistent_checkpoint");
            assert_eq!(result.query_context.execution_stats.vector_scan_count, 100000);
            assert_eq!(result.query_context.execution_stats.distance_ops, 95000 * 768);
            println!("{{\"event\":\"query_verified\",\"phase\":\"{phase}\",\"sample\":{sample},\"elapsed_ms\":{elapsed_ms},\"vector_scan_count\":100000,\"distance_ops\":72960000,\"eligible_vectors\":95000,\"ids\":{},\"final_scores\":[1,1,1,1,1,1,1,1,1,1],\"verified\":true}}", format!("[{}]", expected.iter().map(|id| format!("{id:?}")).collect::<Vec<_>>().join(",")));
        }
    }
    println!("{{\"event\":\"complete\",\"verified_queries\":23,\"verified\":true}}");
    db.close()?;
    Ok(())
}
