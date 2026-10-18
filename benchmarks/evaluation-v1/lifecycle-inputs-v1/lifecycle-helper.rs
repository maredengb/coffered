use cofferdb::{parse_json, CheckOptions, CompactOptions, Database, ErrorCode, QueryEmbeddingInput};
use std::{error::Error, fs, io::{BufReader, Read, Write}, path::Path, time::Instant};

fn embedding() -> QueryEmbeddingInput {
    let mut vector = vec![0.0; 768]; vector[0] = 1.0;
    QueryEmbeddingInput { model_name: Some("synthetic-lifecycle-v1".into()), dim: 768, vector }
}

fn counts(db: &Database, deleted: bool) -> Result<(), Box<dyn Error>> {
    let stats = db.stats()?;
    assert_eq!((stats.collection_count, stats.document_count, stats.chunk_count, stats.embedding_count),
        if deleted { (9, 899, 89900, 89900) } else { (10, 1000, 100000, 100000) });
    for index in 0..10 {
        let collection = db.collection(&format!("lifecycle-{index}"));
        if deleted && index == 9 {
            assert_eq!(collection.err().unwrap().code(), ErrorCode::CollectionNotFound);
            continue;
        }
        let stats = collection?.stats()?;
        let documents = if deleted && index == 0 { 99 } else { 100 };
        assert_eq!((stats.document_count, stats.chunk_count, stats.embedding_count),
            (documents, documents * 100, documents * 100));
    }
    Ok(())
}

fn untouched(db: &Database, view: &str) -> Result<(), Box<dyn Error>> {
    let collection = db.collection("lifecycle-4")?;
    let expected: Vec<_> = (76400..76410).map(|n| format!("chunk-{n}")).collect();
    for mode in ["vector", "filtered-vector", "filtered-text", "filtered-hybrid"] {
        let mut query = collection.search().top_k(10);
        if mode != "filtered-text" { query = query.query_embedding(embedding()); }
        if mode == "filtered-text" || mode == "filtered-hybrid" { query = query.text("764"); }
        if mode != "vector" { query = query.filter(parse_json("{\"doc.fixture_document_index\":764}")?); }
        println!("{{\"stage\":\"query_started\",\"query_kind\":\"untouched_query\",\"view\":\"{view}\",\"mode\":\"{mode}\"}}");
        std::io::stdout().flush()?;
        let start = Instant::now(); let result = query.execute()?;
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        assert_eq!(result.results.iter().map(|row| &row.chunk_external_id).collect::<Vec<_>>(),
            expected.iter().collect::<Vec<_>>(), "{mode}");
        assert!(result.results.iter().all(|row| row.final_score == 1.0));
        let checks = if mode == "filtered-text" { 0 } else { 10000 };
        let components = if mode == "vector" { 7680000 } else if mode == "filtered-text" { 0 } else { 76800 };
        assert_eq!(result.query_context.execution_stats.vector_scan_count, checks, "{mode}");
        assert_eq!(result.query_context.execution_stats.distance_ops, components, "{mode}");
        println!("{{\"stage\":\"untouched_query\",\"view\":\"{view}\",\"mode\":\"{mode}\",\"elapsed_ms\":{elapsed},\"result\":{}}}", result.to_json().to_compact_string());
    }
    Ok(())
}

fn target(db: &Database, state: &str, view: &str) -> Result<(), Box<dyn Error>> {
    let collection = db.collection("lifecycle-0")?;
    let deleted = state == "deleted";
    let updated = state == "updated";
    let mut expected = Vec::new();
    if deleted {
        assert_eq!(collection.get_document_by_external_id("doc-750").err().unwrap().code(), ErrorCode::DocumentNotFound);
        assert_eq!(collection.get_chunk_by_external_id("chunk-75000").err().unwrap().code(), ErrorCode::ChunkNotFound);
    } else {
        let document = collection.get_document_by_external_id("doc-750")?;
        assert_eq!(document.metadata.get("revision").and_then(|x| x.as_f64()), Some(if updated { 99.0 } else { 4.0 }));
        let mut chunks = collection.list_chunks(&document.document_id)?;
        assert_eq!(chunks.len(), 100);
        for chunk in &chunks {
            assert_eq!(chunk.text.contains("verifiedreplacement"), updated);
            let n: usize = chunk.chunk_external_id.strip_prefix("chunk-").unwrap().parse()?;
            assert!((75000..75100).contains(&n));
        }
        let actual_ids: std::collections::BTreeSet<_> = chunks.iter().map(|c| c.chunk_external_id.clone()).collect();
        let required_ids: std::collections::BTreeSet<_> = (75000..75100).map(|n| format!("chunk-{n}")).collect();
        assert_eq!(actual_ids, required_ids);
        chunks.sort_by(|a, b| a.chunk_id.cmp(&b.chunk_id));
        expected = chunks.into_iter().take(10).map(|c| c.chunk_external_id).collect();
    }
    let token = if updated { "verifiedreplacement" } else { "750" };
    let modes: &[&str] = if updated {
        &["text", "vector", "filtered-vector", "hybrid", "old-text", "old-filter"]
    } else if deleted {
        &["text", "filtered-vector", "hybrid"]
    } else { &["text", "filtered-vector"] };
    for mode in modes {
        let mut query = collection.search().top_k(10);
        if *mode != "text" && *mode != "old-text" { query = query.query_embedding(embedding()); }
        if *mode == "text" || *mode == "hybrid" || *mode == "old-text" {
            query = query.text(if *mode == "old-text" { "750" } else if deleted { "verifiedreplacement" } else { token });
        }
        if *mode == "filtered-vector" || *mode == "hybrid" || *mode == "old-filter" {
            query = query.filter(parse_json(if *mode == "old-filter" || !updated && !deleted {
                "{\"doc.fixture_document_index\":750,\"doc.revision\":4}"
            } else { "{\"doc.fixture_document_index\":750,\"doc.revision\":99}" })?);
        }
        println!("{{\"stage\":\"query_started\",\"query_kind\":\"affected_query\",\"view\":\"{view}\",\"state\":\"{state}\",\"mode\":\"{mode}\"}}");
        std::io::stdout().flush()?;
        let start = Instant::now(); let result = query.execute()?;
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        let empty = deleted || *mode == "old-text" || *mode == "old-filter";
        let want: Vec<_> = if empty { Vec::new() } else { expected.iter().collect() };
        assert_eq!(result.results.iter().map(|row| &row.chunk_external_id).collect::<Vec<_>>(), want, "{state} {mode}");
        let text_only = *mode == "text" || *mode == "old-text";
        let checks = if text_only { 0 } else if deleted { 9900 } else { 10000 };
        let components = if text_only || empty { 0 } else if *mode == "vector" { 7680000 } else { 76800 };
        assert_eq!(result.query_context.execution_stats.vector_scan_count, checks, "{state} {mode}");
        assert_eq!(result.query_context.execution_stats.distance_ops, components, "{state} {mode}");
        if updated && !empty { assert!(result.results.iter().all(|row| row.final_score == 1.0)); }
        println!("{{\"stage\":\"affected_query\",\"view\":\"{view}\",\"state\":\"{state}\",\"mode\":\"{mode}\",\"elapsed_ms\":{elapsed},\"result\":{}}}", result.to_json().to_compact_string());
    }
    Ok(())
}

// Historical fixed valid-file oracle, not a corrupt/truncated-file parser test.
fn business_counts(path: &Path) -> Result<(u64, u64), Box<dyn Error>> {
    let mut input = BufReader::new(fs::File::open(path)?);
    let mut file_header = [0; 58]; input.read_exact(&mut file_header)?;
    assert_eq!(&file_header[..8], b"COFFERDB");
    let (mut committed, mut pending) = ((0u64, 0u64), (0u64, 0u64));
    loop {
        let mut header = [0; 50];
        match input.read_exact(&mut header) {
            Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => break,
            result => result?,
        }
        if &header[..4] != b"CFRL" { assert_eq!(&header[..8], b"CFIDXCHK"); break; }
        let kind = u16::from_le_bytes(header[4..6].try_into()?);
        let bytes = u64::from_le_bytes(header[8..16].try_into()?); assert!(bytes >= 50);
        match kind {
            1..=6 => { pending.0 += 1; pending.1 += u64::from(kind == 6); }
            7 => { committed.0 += pending.0; committed.1 += pending.1; pending = (0, 0); }
            8 => {}
            _ => panic!("unexpected critical kind"),
        }
        input.seek_relative(i64::try_from(bytes - 50)?)?;
    }
    assert_eq!(pending, (0, 0)); Ok(committed)
}

fn emit(stage: &str, start: Instant, db: &Database) -> Result<(), Box<dyn Error>> {
    println!("{{\"stage\":\"{stage}\",\"elapsed_ms\":{},\"stats\":{}}}",
        start.elapsed().as_secs_f64() * 1000.0, db.stats()?.to_json().to_compact_string());
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect(); let path = Path::new(&args[1]);
    let mode = &args[2]; let start = Instant::now();
    if mode == "open" {
        let db = Database::open_read_only(path)?; emit("open_read_only", start, &db)?; counts(&db, false)?;
    } else if mode == "initial" {
        let db = Database::open_read_only(path)?; counts(&db, false)?;
        untouched(&db, "initial")?; target(&db, "initial", "initial")?;
        emit("initial_verified_group", start, &db)?;
    } else if mode == "replace" {
        let writer = Database::open(path)?; let old = Database::open_read_only(path)?;
        let import_start = Instant::now();
        let summary = writer.import_manifest(&args[3], cofferdb::OnConflict::Replace)?;
        println!("{{\"stage\":\"replacement_import\",\"elapsed_ms\":{},\"result\":{}}}",
            import_start.elapsed().as_secs_f64()*1000.0, summary.to_json().to_compact_string());
        counts(&writer, false)?; target(&writer, "updated", "writer_after_replace")?;
        untouched(&writer, "writer_after_replace")?;
        counts(&old, false)?; target(&old, "initial", "retained_before_replace")?;
        untouched(&old, "retained_before_replace")?;
        emit("replace_and_verify", start, &writer)?;
    } else if mode == "delete" {
        let writer = Database::open(path)?; let old = Database::open_read_only(path)?;
        let collection = writer.collection("lifecycle-0")?;
        let document = collection.get_document_by_external_id("doc-750")?;
        let mutation_start = Instant::now();
        collection.delete_document(&document.document_id)?; writer.drop_collection("lifecycle-9")?;
        emit("delete_and_drop_return", mutation_start, &writer)?;
        counts(&writer, true)?; target(&writer, "deleted", "writer_after_delete")?;
        untouched(&writer, "writer_after_delete")?;
        counts(&old, false)?; target(&old, "updated", "retained_before_delete")?;
        untouched(&old, "retained_before_delete")?;
        emit("delete_and_verify", start, &writer)?;
    } else if mode == "compact" {
        let writer = Database::open(path)?; let old = Database::open_read_only(path)?;
        let before = business_counts(path)?; let compact_start = Instant::now();
        let result = writer.compact(CompactOptions { rebuild_indexes: true })?;
        println!("{{\"stage\":\"compact_return\",\"elapsed_ms\":{},\"result\":{}}}",
            compact_start.elapsed().as_secs_f64()*1000.0, result.to_json().to_compact_string());
        let after = business_counts(path)?;
        assert_eq!(result.records_removed as u64, before.0 - after.0);
        assert_eq!(result.tombstones_removed as u64, before.1 - after.1);
        counts(&writer, true)?; target(&writer, "deleted", "writer_after_compact")?;
        untouched(&writer, "writer_after_compact")?;
        counts(&old, true)?; target(&old, "deleted", "retained_before_compact")?;
        untouched(&old, "retained_before_compact")?;
        emit("compact_and_verify", start, &writer)?;
    } else if mode == "check" {
        let reader = Database::open_read_only(path)?;
        let check_start = Instant::now(); let result = reader.check(CheckOptions { verify_indexes: true })?;
        assert_eq!(result.status, "ok"); assert!(result.issues.is_empty());
        println!("{{\"stage\":\"check_return\",\"elapsed_ms\":{},\"result\":{}}}",
            check_start.elapsed().as_secs_f64()*1000.0, result.to_json().to_compact_string());
    } else if mode == "export" {
        let db = Database::open_read_only(path)?; let export_start = Instant::now();
        let result = db.export_collection_jsonl("lifecycle-0", &args[3])?;
        assert_eq!((result.documents, result.chunks, result.embeddings), (99, 9900, 9900));
        println!("{{\"stage\":\"export_return\",\"elapsed_ms\":{},\"result\":{}}}",
            export_start.elapsed().as_secs_f64()*1000.0, result.to_json().to_compact_string());
        emit("export_and_verify", start, &db)?;
    } else if mode == "reopen" {
        let reader = Database::open_read_only(path)?; emit("reopen_read_only", start, &reader)?;
        counts(&reader, true)?; target(&reader, "deleted", "reopened")?; untouched(&reader, "reopened")?;
        emit("reopen_and_verify", start, &reader)?;
    } else { panic!("unsupported fixed harness stage"); }
    println!("{{\"stage\":\"complete\",\"verified\":true}}"); Ok(())
}
