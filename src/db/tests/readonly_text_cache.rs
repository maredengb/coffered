use super::*;

#[test]
fn readonly_text_cache_shares_the_full_limit_with_existing_vector_cache() {
    const LIMIT: usize = 1_048_576;
    let path = cached_vector_batch_fixture("readonly-text-shared-cache");
    let original = fs::read(&path).unwrap();
    let db = Database::open_read_only(&path).unwrap();
    let context = db.json_context();
    let mut inner = db.lock().unwrap();
    inner.prepare_text_index(1, &context).unwrap();
    assert!(inner.text_index.has_resident_cache());
    let text_bytes = inner.text_index.reserved_bytes();
    inner.text_index.clear_cache();
    let id = inner
        .catalog
        .collection_by_name("cached")
        .unwrap()
        .collection_id
        .clone();
    inner.prepare_vector_index(&id, 4096).unwrap();
    let vector_bytes = inner.optional_cache_bytes();
    assert!(vector_bytes > 32_768);
    // Posting-array growth temporarily retains both arrays. Leave headroom
    // for that peak, but less than the existing vector credit: deducting that
    // credit twice would reject this valid cache candidate.
    let free = text_bytes + 8192;
    assert!(free < vector_bytes);
    let blocker = context
        .cache_reservation((LIMIT - vector_bytes - free) as u128, LIMIT)
        .unwrap();
    inner.prepare_text_index(1, &context).unwrap();
    assert!(inner.text_index.has_resident_cache());
    assert_eq!(inner.optional_cache_bytes(), vector_bytes + text_bytes);
    assert!(context
        .cache_reservation((free - text_bytes) as u128, LIMIT)
        .is_some());
    assert!(context
        .cache_reservation((free - text_bytes + 1) as u128, LIMIT)
        .is_none());
    assert!(inner.lazy_vector_source.is_some());
    assert_eq!(
        inner.index_suffix_status,
        IndexSuffixState::PersistentCheckpoint
    );
    inner.text_index.clear_cache();
    drop(blocker);
    drop(inner);
    for name in ["first", "second", "cached", "empty"] {
        let result = db
            .collection(name)
            .unwrap()
            .search()
            .text("stable")
            .execute()
            .unwrap();
        assert_eq!(
            result.results.len(),
            match name {
                "first" => 2,
                "second" => 4,
                "cached" => 8,
                _ => 0,
            }
        );
        assert_eq!(result.query_context.index_source, "persistent_checkpoint");
    }
    assert_eq!(fs::read(&path).unwrap(), original);
    drop(db);
    fs::remove_file(path).unwrap();
}

#[test]
fn optional_text_prewarm_does_not_invalidate_an_unrelated_healthy_collection() {
    let path = query_preflight_fixture("readonly-text-unrelated-crc", &"needle ".repeat(5000));
    let db = Database::open_with_options(
        &path,
        DatabaseOpenOptions {
            read_only: true,
            cache_budget_mb: Some(1),
            ..Default::default()
        },
    )
    .unwrap();
    let before = fs::read(&path).unwrap();
    let (suffix, bad_collection) = {
        let inner = db.lock().unwrap();
        assert!(
            inner
                .lazy_vector_source
                .as_ref()
                .unwrap()
                .text_encoded_len()
                .unwrap()
                < 1_048_576 / 8
        );
        (
            inner.storage.critical_end_offset() as usize,
            inner
                .catalog
                .collection_by_name("second")
                .unwrap()
                .collection_id
                .clone(),
        )
    };
    // Locate one TEXT record through the published binary extent table, then
    // damage only its payload after this reader has validated the directory.
    let u64_at = |bytes: &[u8], offset| {
        u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap()) as usize
    };
    let table_start = suffix + u64_at(&before[suffix..], 73);
    let table_len = u64_at(&before[suffix..], 81);
    let number = bad_collection
        .strip_prefix("col_")
        .unwrap()
        .parse::<usize>()
        .unwrap();
    let extent = before[table_start..table_start + table_len]
        .chunks_exact(46)
        .find(|entry| {
            u16::from_le_bytes(entry[8..10].try_into().unwrap()) == 0x8002
                && u64_at(entry, 10) == number
        })
        .unwrap();
    let damaged_offset = suffix + u64_at(extent, 18) + 50;
    let mut file = OpenOptions::new().write(true).open(&path).unwrap();
    file.seek(SeekFrom::Start(damaged_offset as u64)).unwrap();
    file.write_all(&[before[damaged_offset] ^ 1]).unwrap();
    drop(file);
    let damaged = fs::read(&path).unwrap();
    for _ in 0..2 {
        let result = db
            .collection("first")
            .unwrap()
            .search()
            .text("needle")
            .execute()
            .unwrap();
        assert_eq!(result.results.len(), 2);
        assert_eq!(result.query_context.index_source, "persistent_checkpoint");
        assert!(!result.query_context.transient_rebuild.started);
        let inner = db.lock().unwrap();
        assert!(!inner.text_index.has_resident_cache());
        assert!(inner.lazy_vector_source.is_some());
    }
    let error = db
        .collection("second")
        .unwrap()
        .search()
        .text("needle")
        .execute()
        .unwrap_err();
    assert_eq!(error.code(), ErrorCode::IndexCorruption);
    assert_eq!(
        error_detail_value(&error, "index_unavailable_reason"),
        Some("transient_rebuild_budget_exceeded")
    );
    assert_eq!(fs::read(&path).unwrap(), damaged);
    drop(db);
    fs::remove_file(path).unwrap();
}

#[test]
fn owned_readonly_text_cache_keeps_its_snapshot_after_suffix_replacement() {
    let path = query_preflight_fixture("readonly-text-owned-snapshot", "needle stable");
    let reader = Database::open_read_only(&path).unwrap();
    let query = |db: &Database| {
        db.collection("first")
            .unwrap()
            .search()
            .text("needle")
            .execute()
            .unwrap()
    };
    let expected = query(&reader);
    assert!(reader.lock().unwrap().text_index.has_resident_cache());
    let writer = Database::open(&path).unwrap();
    writer
        .collection("first")
        .unwrap()
        .insert_document(DocumentInput {
            external_id: "later".into(),
            title: None,
            source_uri: None,
            metadata: JsonValue::object(),
            chunks: vec![ChunkInput {
                chunk_external_id: "later#0".into(),
                text: "needle needle needle".into(),
                ordinal: 0,
                metadata: JsonValue::object(),
                source_span: None,
            }],
        })
        .unwrap();
    assert_eq!(query(&writer).results.len(), 3);
    writer
        .compact(CompactOptions {
            rebuild_indexes: true,
        })
        .unwrap();
    for _ in 0..2 {
        let actual = query(&reader);
        assert_eq!(
            actual
                .results
                .iter()
                .map(|hit| (
                    &hit.chunk_id,
                    hit.final_score.to_bits(),
                    hit.bm25_score.map(f64::to_bits)
                ))
                .collect::<Vec<_>>(),
            expected
                .results
                .iter()
                .map(|hit| (
                    &hit.chunk_id,
                    hit.final_score.to_bits(),
                    hit.bm25_score.map(f64::to_bits)
                ))
                .collect::<Vec<_>>()
        );
        assert_eq!(actual.query_context.index_source, "persistent_checkpoint");
        assert!(!actual.query_context.transient_rebuild.started);
    }
    drop(expected);
    drop((writer, reader));
    let reopened = Database::open_read_only(&path).unwrap();
    assert_eq!(query(&reopened).results.len(), 3);
    drop(reopened);
    fs::remove_file(path).unwrap();
}
