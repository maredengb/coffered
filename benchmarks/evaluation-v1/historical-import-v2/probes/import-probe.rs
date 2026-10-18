// Included after the unmodified production crate's module declarations.
// This internal probe observes existing private diagnostics without cfg(test).
fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    use std::{io::Write, path::Path, time::Instant};
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(args.len(), 5);
    let input = Path::new(&args[1]);
    let count: usize = args[2].parse()?;
    let mode = match args[3].as_str() {
        "inline_full_checkpoint" => db::ImportIndexBuildMode::InlineFullCheckpoint,
        "deferred_full_checkpoint" => db::ImportIndexBuildMode::DeferredFullCheckpoint,
        _ => panic!("unsupported import mode"),
    };
    let parts = match args[4].as_str() { "bulk" => 1, "small" => 10, _ => panic!("unsupported layout") };
    let database = input.join(format!("{}-{}.coffer", args[3], args[4]));
    assert!(!database.exists());
    let db = Database::create(&database, DatabaseConfig::default())?;
    let started = Instant::now();
    let mut committed = 0;
    let mut checkpoints = 0;
    let mut commit_ms = 0.0;
    let mut checkpoint_ms = 0.0;
    for part in 0..parts {
        let manifest = input.join(if parts == 1 { "bulk/import.json".into() } else { format!("small-{part}/import.json") });
        println!("{{\"event\":\"import_started\",\"part\":{part}}}");
        std::io::stdout().flush()?;
        let imported = db.import_manifest_measured_in(&manifest, OnConflict::Error, mode, db.json_context())?;
        assert_eq!(imported.summary.chunks_inserted, count / parts);
        assert_eq!(imported.summary.embeddings_inserted, 0);
        assert!(imported.batches.final_checkpoint_ready);
        committed += imported.batches.committed_batches;
        checkpoints += imported.batches.checkpoint_cycles;
        commit_ms += imported.batches.commit_time.as_secs_f64() * 1000.0;
        checkpoint_ms += imported.batches.checkpoint_time.as_secs_f64() * 1000.0;
        println!("{{\"event\":\"import_returned\",\"part\":{part},\"committed_batches\":{},\"checkpoint_cycles\":{},\"checkpoint_ready_cycles\":{},\"max_batch_chunks\":{},\"commit_ms\":{},\"checkpoint_ms\":{}}}",
            imported.batches.committed_batches, imported.batches.checkpoint_cycles,
            imported.batches.checkpoint_ready_cycles, imported.batches.max_batch_chunks,
            imported.batches.commit_time.as_secs_f64() * 1000.0, imported.batches.checkpoint_time.as_secs_f64() * 1000.0);
        drop(imported);
        let collection = db.collection("measurements")?;
        let stats = collection.stats()?;
        assert_eq!((stats.document_count, stats.chunk_count, stats.embedding_count),
            ((part + 1) * count / parts / 100, (part + 1) * count / parts, 0));
        for filtered in [false, true] {
            println!("{{\"event\":\"query_started\",\"part\":{part},\"filtered\":{filtered}}}");
            std::io::stdout().flush()?;
            let mut query = collection.search().text("needle").top_k(10);
            if filtered { query = query.filter(parse_json("{\"doc.project\":0}")?); }
            let query_started = Instant::now();
            let result = query.execute()?;
            let query_ms = query_started.elapsed().as_secs_f64() * 1000.0;
            let expected: Vec<_> = (0..10).map(|i| format!("chunk-{i}")).collect();
            assert_eq!(result.results.iter().map(|r| &r.chunk_external_id).collect::<Vec<_>>(), expected.iter().collect::<Vec<_>>());
            assert!(result.results.iter().all(|r| r.final_score == 1.0));
            assert_eq!(result.query_context.index_source, "persistent_checkpoint");
            assert_eq!(result.query_context.execution_stats.distance_ops, 0);
            println!("{{\"event\":\"query_verified\",\"part\":{part},\"filtered\":{filtered},\"elapsed_ms\":{query_ms},\"ids\":{},\"final_scores\":[1,1,1,1,1,1,1,1,1,1],\"index_source\":\"persistent_checkpoint\",\"verified\":true}}", format!("[{}]", expected.iter().map(|id| format!("{id:?}")).collect::<Vec<_>>().join(",")));
        }
    }
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    let stats = db.stats()?;
    println!("{{\"event\":\"complete\",\"chunks\":{count},\"parts\":{parts},\"queries_verified\":{},\"commit_ms\":{commit_ms},\"checkpoint_ms\":{checkpoint_ms},\"committed_batches\":{committed},\"checkpoint_cycles\":{checkpoints},\"import_context\":{{\"import_time_ms\":{elapsed_ms}}},\"metrics\":{{\"import_time_ms\":{elapsed_ms}}},\"chunk_per_second\":{},\"file_bytes\":{},\"index_bytes\":{},\"verified\":true}}", parts * 2, count as f64 * 1000.0 / elapsed_ms, stats.file_size, stats.index_size_bytes);
    db.close()?;
    Ok(())
}

