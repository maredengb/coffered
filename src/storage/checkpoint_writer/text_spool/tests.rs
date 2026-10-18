use super::*;
use crate::json::WorkLedger;
use crate::model::VectorIndexConfig;
use std::io::Cursor;
use std::sync::atomic::{AtomicUsize, Ordering};

struct Directory(PathBuf);
impl Directory {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "cofferdb-text-spool-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Directory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn payload() -> TextIndexSegmentPayload {
    TextIndexSegmentPayload {
        collection_id: "col_1".into(),
        covered_commit_sequence_no: 7,
        chunks: vec![TextIndexChunkEntry {
            chunk_id: "chk_10000000000000000".into(),
            document_id: "doc_93".into(),
            token_count: 0,
        }],
        terms: vec![
            TextIndexTermEntry {
                term: "empty".into(),
                postings: Vec::new(),
            },
            TextIndexTermEntry {
                term: "needle".into(),
                postings: vec![TextIndexPostingEntry {
                    chunk_id: "chk_10000000000000000".into(),
                    term_frequency: 4_294_967_297,
                }],
            },
        ],
    }
}

fn encoded(payload: &TextIndexSegmentPayload) -> Vec<u8> {
    let mut bytes = Vec::new();
    write_text_segment(payload, &mut bytes).unwrap();
    bytes
}

#[test]
fn resident_visitors_preserve_shared_terms_empty_dictionary_and_id_order_without_tokens() {
    use crate::index::text::{TextIndex, TextSearchOptions};
    let directory = Directory::new();
    let context = JsonOperationContext::standalone();
    let available = context.available_workspace_bytes();
    let disk = Arc::new(TemporaryDiskBudget::new(128 * 1024));
    let mut segments = Vec::new();
    for collection in 1..=3u64 {
        let mut segment = payload();
        segment.collection_id = format!("col_{collection:016}");
        segment.chunks = if collection == 3 {
            Vec::new()
        } else {
            [
                9_999_999_999_999_990 + collection,
                10_000_000_000_000_000 + collection,
            ]
            .into_iter()
            .enumerate()
            .map(|(n, id)| TextIndexChunkEntry {
                chunk_id: format!("chk_{id:016}"),
                document_id: "doc_0000000000000001".into(),
                token_count: if n == 0 { 0 } else { 9 },
            })
            .collect()
        };
        segment.terms = [
            "a".repeat(8193),
            "b".into(),
            "empty".into(),
            "\u{e9}".into(),
        ]
        .into_iter()
        .map(|term| TextIndexTermEntry {
            postings: if term == "empty" || collection == 3 {
                Vec::new()
            } else {
                segment
                    .chunks
                    .iter()
                    .map(|chunk| TextIndexPostingEntry {
                        chunk_id: chunk.chunk_id.clone(),
                        term_frequency: collection as usize,
                    })
                    .collect()
            },
            term,
        })
        .collect();
        segments.push(segment);
    }
    let mut spools = Vec::new();
    for segment in &segments {
        let bytes = encoded(segment);
        let mut spool = TextSpool::new(
            catalog_id_number(&segment.collection_id).unwrap(),
            7,
            &directory.0,
            disk.clone(),
            &context,
        )
        .unwrap();
        spool
            .copy_body_from(
                &mut Cursor::new(&bytes[..bytes.len() - 4]),
                bytes.len() - 4,
                7,
                &context,
                0,
            )
            .unwrap();
        spool.finish_copy().unwrap();
        spools.push(spool);
    }
    let staged_available = context.available_workspace_bytes();
    let expected = TextIndex::try_build_from_segments(segments.clone()).unwrap();
    let mut expected_counts = expected.collection_counts();
    expected_counts.retain(|_, count| *count != 0);
    crate::tokenizer::TOKEN_VISITS.with(|count| count.set(0));
    let actual = TextIndex::try_build_from_visitors(
        segments
            .iter()
            .map(|segment| segment.collection_id.as_str()),
        |id, visit| spools[(catalog_id_number(id)? - 1) as usize].visit(&context, 0, visit),
    )
    .unwrap()
    .unwrap();
    assert_eq!(crate::tokenizer::TOKEN_VISITS.with(std::cell::Cell::get), 0);
    assert_eq!(actual.chunk_lengths, expected.chunk_lengths);
    assert_eq!(actual.collection_counts(), expected_counts);
    assert_eq!(actual.avg_len_by_collection, expected.avg_len_by_collection);
    assert_eq!(
        format!("{:?}", actual.postings()),
        format!("{:?}", expected.postings())
    );
    assert!(actual.postings()["empty"].is_empty());
    assert_eq!(
        actual.postings()["b"]
            .iter()
            .map(|posting| (posting.collection_id.as_ref(), posting.chunk_id.as_str()))
            .collect::<Vec<_>>(),
        vec![
            ("col_0000000000000001", "chk_10000000000000001"),
            ("col_0000000000000001", "chk_9999999999999991"),
            ("col_0000000000000002", "chk_10000000000000002"),
            ("col_0000000000000002", "chk_9999999999999992"),
        ]
    );
    for segment in &segments {
        for query in ["b \u{e9}", "empty", "absent"] {
            assert_eq!(
                format!(
                    "{:?}",
                    actual
                        .search(
                            &segment.collection_id,
                            query,
                            TextSearchOptions::with_explanations(9),
                            &context
                        )
                        .unwrap()
                        .as_slice()
                ),
                format!(
                    "{:?}",
                    expected
                        .search(
                            &segment.collection_id,
                            query,
                            TextSearchOptions::with_explanations(9),
                            &context
                        )
                        .unwrap()
                        .as_slice()
                ),
            );
        }
        let owners = actual
            .postings()
            .values()
            .flatten()
            .filter(|posting| posting.collection_id.as_ref() == segment.collection_id)
            .map(|posting| &posting.collection_id)
            .collect::<Vec<_>>();
        if let Some(first) = owners.first() {
            assert!(owners.iter().all(|owner| Arc::ptr_eq(first, owner)));
            assert_eq!(Arc::strong_count(first), owners.len());
        }
    }
    let incomplete = TextIndex::try_build_from_visitors(
        segments
            .iter()
            .map(|segment| segment.collection_id.as_str()),
        |id, visit| {
            if catalog_id_number(id)? == 3 {
                return Ok(false);
            }
            spools[(catalog_id_number(id)? - 1) as usize].visit(&context, 0, visit)
        },
    )
    .unwrap();
    assert!(incomplete.is_none());
    let cached = match TextIndex::try_cache_from_visitors(
        segments
            .iter()
            .map(|segment| segment.collection_id.as_str()),
        &context,
        usize::MAX,
        |id, visit| spools[(catalog_id_number(id)? - 1) as usize].visit(&context, 0, visit),
    )
    .unwrap()
    {
        crate::index::text::TextCacheBuild::Complete(index) => index,
        other => panic!("complete wire source: {other:?}"),
    };
    assert_eq!(cached.chunk_lengths, actual.chunk_lengths);
    assert_eq!(cached.collection_counts(), actual.collection_counts());
    assert_eq!(cached.avg_len_by_collection, actual.avg_len_by_collection);
    assert_eq!(
        format!("{:?}", cached.postings()),
        format!("{:?}", actual.postings())
    );
    drop(cached);
    assert_eq!(context.available_workspace_bytes(), staged_available);
    let second = &mut spools[1];
    second
        .staging
        .file
        .seek(SeekFrom::Start(second.extent.length - 1))
        .unwrap();
    second
        .staging
        .file
        .write_all(&[second.extent.checksum.to_le_bytes()[3] ^ 1])
        .unwrap();
    let error = TextIndex::try_build_from_visitors(
        segments
            .iter()
            .map(|segment| segment.collection_id.as_str()),
        |id, visit| spools[(catalog_id_number(id)? - 1) as usize].visit(&context, 0, visit),
    )
    .unwrap_err();
    assert_eq!(error.code(), ErrorCode::ChecksumMismatch);
    for cache_limit in [0, usize::MAX] {
        let error = TextIndex::try_cache_from_visitors(
            segments
                .iter()
                .map(|segment| segment.collection_id.as_str()),
            &context,
            cache_limit,
            |id, visit| spools[(catalog_id_number(id)? - 1) as usize].visit(&context, 0, visit),
        )
        .unwrap_err();
        assert_eq!(error.code(), ErrorCode::ChecksumMismatch);
        assert_eq!(context.available_workspace_bytes(), staged_available);
    }
    assert_eq!(context.available_workspace_bytes(), staged_available);
    drop((actual, expected, spools));
    assert_eq!(disk.used(), 0);
    assert_eq!(context.available_workspace_bytes(), available);
}

#[test]
fn copies_preserve_raw_fields_and_df_zero_and_replay_with_new_commit() {
    let directory = Directory::new();
    let context = JsonOperationContext::standalone();
    let available = context.available_workspace_bytes();
    let disk = Arc::new(TemporaryDiskBudget::new(4096));
    let bytes = encoded(&payload());
    let mut body = TextSpool::new(1, 7, &directory.0, disk.clone(), &context).unwrap();
    assert_eq!(
        body.copy_body_from(
            &mut Cursor::new(&bytes[..bytes.len() - 4]),
            bytes.len() - 4,
            7,
            &context,
            0
        )
        .unwrap(),
        1
    );
    body.finish_copy().unwrap();
    assert_eq!(disk.used(), bytes.len() as u64);
    for commit in [7, 8, u64::MAX, 8] {
        let mut actual = Vec::new();
        body.write(&mut actual, commit).unwrap();
        let mut expected = bytes.clone();
        expected[10..18].copy_from_slice(&commit.to_le_bytes());
        let end = expected.len() - 4;
        let crc = crate::hash::crc32c(&expected[..end]);
        expected[end..].copy_from_slice(&crc.to_le_bytes());
        assert_eq!(actual, expected);
    }
    let mut events = Vec::new();
    body.visit(&context, 0, &mut |entry| {
        if let TextSegmentEntry::Term {
            text,
            document_frequency,
        } = entry
        {
            events.push((text.to_owned(), document_frequency));
        }
        Ok(())
    })
    .unwrap();
    assert_eq!(events, [("empty".into(), 0), ("needle".into(), 1)]);
    drop(body);
    assert_eq!(disk.used(), 0);
    assert_eq!(std::fs::read_dir(&directory.0).unwrap().count(), 0);
    assert_eq!(context.available_workspace_bytes(), available);
}

#[test]
fn failed_copy_and_budgeted_output_never_flush_from_drop() {
    struct Fault<'a>(Cursor<&'a [u8]>);
    impl Read for Fault<'_> {
        fn read(&mut self, output: &mut [u8]) -> std::io::Result<usize> {
            if self.0.position() >= 18 {
                return Err(std::io::Error::other("spool read sentinel"));
            }
            let length = output.len().min(18 - self.0.position() as usize);
            self.0.read(&mut output[..length])
        }
    }
    let directory = Directory::new();
    let context = JsonOperationContext::standalone();
    let available = context.available_workspace_bytes();
    let disk = Arc::new(TemporaryDiskBudget::new(4096));
    let bytes = encoded(&payload());
    let mut body = TextSpool::new(1, 7, &directory.0, disk.clone(), &context).unwrap();
    let error = body
        .copy_body_from(
            &mut Fault(Cursor::new(&bytes)),
            bytes.len() - 4,
            7,
            &context,
            4096,
        )
        .err()
        .unwrap();
    assert!(
        matches!(error, IndexReadError::Io(error) if error.to_string() == "spool read sentinel")
    );
    assert_eq!(body.staging.file.metadata().unwrap().len(), 0);
    drop(body);
    assert_eq!(disk.used(), 0);
    let disk = Arc::new(TemporaryDiskBudget::new(3));
    let (mut file, mut cleanup) = create_budgeted_temporary_file_in_context(
        &directory.0,
        "drop",
        "tmp",
        disk.clone(),
        &context,
        "operation_scratch",
    )
    .unwrap();
    let mut reserved = 0;
    {
        let mut writer = StagingWriter {
            output: Some(BufWriter::with_capacity(4096, &mut file)),
            cleanup: &mut cleanup,
            base: 0,
            bytes: 0,
            reserved: &mut reserved,
        };
        writer.append(b"abc").unwrap();
        assert_eq!(
            writer.append(b"d").unwrap_err().code(),
            ErrorCode::ResourceExhausted
        );
    }
    assert_eq!(file.metadata().unwrap().len(), 0);
    drop((file, cleanup));
    assert_eq!(disk.used(), 0);
    assert_eq!(context.available_workspace_bytes(), available);
    assert_eq!(std::fs::read_dir(&directory.0).unwrap().count(), 0);
}

#[test]
fn staging_integrity_precedes_consumer_failure_and_every_truncation_is_rejected() {
    let directory = Directory::new();
    let context = JsonOperationContext::standalone();
    let disk = Arc::new(TemporaryDiskBudget::new(4096));
    let bytes = encoded(&payload());
    let mut body = TextSpool::new(1, 7, &directory.0, disk.clone(), &context).unwrap();
    body.copy_body_from(
        &mut Cursor::new(&bytes[..bytes.len() - 4]),
        bytes.len() - 4,
        7,
        &context,
        0,
    )
    .unwrap();
    body.finish_copy().unwrap();
    let failure = || Error::new(ErrorCode::InvalidSearchParams, "consumer sentinel");
    assert_eq!(
        body.visit(&context, 0, &mut |_| Err(failure()))
            .unwrap_err()
            .message(),
        "consumer sentinel"
    );
    body.staging
        .file
        .seek(SeekFrom::Start(body.extent.length - 1))
        .unwrap();
    body.staging
        .file
        .write_all(&[bytes[bytes.len() - 1] ^ 1])
        .unwrap();
    assert_eq!(
        body.visit(&context, 0, &mut |_| Err(failure()))
            .unwrap_err()
            .code(),
        ErrorCode::ChecksumMismatch
    );
    for length in 0..bytes.len() {
        body.staging.file.set_len(length as u64).unwrap();
        let mut output = Vec::new();
        assert_eq!(
            body.write(&mut output, 8).unwrap_err().code(),
            ErrorCode::ChecksumMismatch
        );
        assert!(output.is_empty());
    }
    drop(body);
    assert_eq!(disk.used(), 0);
}

fn catalog(text: &str) -> Catalog {
    let mut catalog = Catalog::default();
    let collection = CollectionInfo {
        collection_id: "col_1".into(),
        name: "first".into(),
        embedding_dim: 2,
        distance_metric: DistanceMetric::Cosine,
        embedding_model: None,
        vectors_normalized: false,
        vector_index: VectorIndexConfig::Exact,
        tokenizer: Default::default(),
        created_at: String::new(),
    };
    catalog
        .collections
        .insert(collection.collection_id.clone(), collection);
    for id in [9_999_999_999_999_999u64, 10_000_000_000_000_000] {
        let chunk = format!("chk_{id:016}");
        let doc = format!("doc_{id:016}");
        catalog.documents.insert(
            doc.clone(),
            Box::new(Document {
                document_id: doc.clone(),
                collection_id: "col_1".into(),
                external_id: doc.clone(),
                title: None,
                source_uri: None,
                metadata: JsonValue::object(),
                created_at: String::new(),
                updated_at: String::new(),
            }),
        );
        catalog.chunks.insert(
            chunk.clone(),
            Box::new(Chunk {
                chunk_id: chunk.clone(),
                document_id: doc,
                collection_id: "col_1".into(),
                chunk_external_id: chunk,
                text: text.into(),
                text_hash: String::new(),
                token_count: 0,
                ordinal: 0,
                metadata: JsonValue::object(),
                embedding_id: None,
                source_span: None,
                created_at: String::new(),
            }),
        );
    }
    catalog
}

#[test]
fn bounded_external_build_matches_resident_wire_including_real_spill_and_empty_text() {
    let directory = Directory::new();
    for text in [
        "the and".to_owned(),
        "alphaBeta alphaBeta storage_path \u{e9} needle needle".into(),
        (0..5000)
            .map(|i| format!("word{i:05} "))
            .collect::<String>(),
    ] {
        let catalog = catalog(&text);
        let reference = crate::index::text::TextIndex::try_build_matching(
            &catalog.visible_snapshot(),
            |_| true,
            |_| Ok(text.as_str()),
        )
        .unwrap();
        let mut expected = Vec::new();
        TextRecordSegment::new(
            &catalog,
            &reference,
            catalog.collection_by_id("col_1").unwrap(),
            7,
            &JsonOperationContext::standalone(),
        )
        .write(&mut expected)
        .unwrap();
        // The borrowed fixture input is outside this focused sort/spool budget.
        for limit in [8192, 32768, 262144] {
            let context =
                JsonOperationContext::new(Arc::new(WorkLedger::with_limits(limit, limit, limit)));
            // Final merge holds old runs plus one complete output: 2*10000
            // entries of 20-byte header + 18-byte key, plus the 98-byte spool header.
            let disk_limit = if text.len() == 50000 && limit == 32768 {
                2 * 10000 * (20 + 18) + 98
            } else {
                16 * 1024 * 1024
            };
            let disk = Arc::new(TemporaryDiskBudget::new(disk_limit));
            let body = build_collection(
                &catalog,
                catalog.collection_by_id("col_1").unwrap(),
                7,
                &mut |_| Ok(text.as_str()),
                &directory.0,
                &disk,
                &context,
            );
            // File-management capacity varies by platform. At this small quota,
            // either a correct spill or a genuine memory refusal is valid.
            if text.len() == 50000 && limit == 8192 && body.is_err() {
                let error = body.err().unwrap();
                assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                assert!(error
                    .detail()
                    .contains(&("resource".into(), "memory".into())));
                assert_eq!(disk.used(), 0);
                assert_eq!(context.available_workspace_bytes(), limit);
                assert_eq!(std::fs::read_dir(&directory.0).unwrap().count(), 0);
                continue;
            }
            let body = body
                .unwrap_or_else(|error| panic!("input={} limit={limit}: {error:?}", text.len()));
            let mut actual = Vec::new();
            body.write(&mut actual, 7).unwrap();
            assert_eq!(actual, expected, "limit={limit}");
            assert_eq!(disk.used(), actual.len() as u64);
            drop(body);
            assert_eq!(disk.used(), 0);
            assert_eq!(context.available_workspace_bytes(), limit);
            assert_eq!(std::fs::read_dir(&directory.0).unwrap().count(), 0);
        }
    }
}

#[test]
fn budgeted_unicode_token_build_matches_reference_wire_on_both_axes() {
    let directory = Directory::new();
    let text = (0..130).map(|n| format!("token{n} ")).collect::<String>()
        + &"FOO_BAR \u{23a}X_\u{130} \u{39f}\u{3a3}_\u{39f}\u{3c3} Foo_foo ".repeat(200);
    let catalog = catalog(&text);
    let reference = crate::index::text::TextIndex::try_build_matching(
        &catalog.visible_snapshot(),
        |_| true,
        |_| Ok(text.as_str()),
    )
    .unwrap();
    let mut expected = Vec::new();
    TextRecordSegment::new(
        &catalog,
        &reference,
        catalog.collection_by_id("col_1").unwrap(),
        7,
        &JsonOperationContext::standalone(),
    )
    .write(&mut expected)
    .unwrap();
    for parent in [false, true] {
        for limit in [8192, 32768] {
            let context = JsonOperationContext::new(Arc::new(WorkLedger::with_limits(
                if parent { 2 * limit } else { limit },
                0,
                if parent { limit } else { 2 * limit },
            )));
            let disk = Arc::new(TemporaryDiskBudget::new(1024 * 1024));
            let mut reads = 0;
            let body = build_collection(
                &catalog,
                catalog.collection_by_id("col_1").unwrap(),
                7,
                &mut |_| {
                    reads += 1;
                    Ok(text.as_str())
                },
                &directory.0,
                &disk,
                &context,
            )
            .unwrap();
            assert_eq!(reads, 2);
            let mut actual = Vec::new();
            body.write(&mut actual, 7).unwrap();
            assert_eq!(actual, expected, "parent={parent}, limit={limit}");
            assert_eq!(disk.used(), actual.len() as u64);
            drop(body);
            assert_eq!(context.available_workspace_bytes(), limit);
            assert_eq!(disk.used(), 0);
            assert_eq!(std::fs::read_dir(&directory.0).unwrap().count(), 0);
        }
    }
}

#[test]
#[cfg(unix)]
fn normalization_pressure_spills_before_allocating_the_word_or_preserves_disk_error() {
    let directory = Directory::new();
    let word = "PROJECT".repeat(1000);
    let text = (0..130).map(|n| format!("token{n} ")).collect::<String>() + &word;
    let mut catalog = catalog(&text);
    catalog
        .chunks
        .get_mut("chk_10000000000000000")
        .unwrap()
        .text = "shared".into();
    let reference = crate::index::text::TextIndex::try_build_matching(
        &catalog.visible_snapshot(),
        |_| true,
        |visible| Ok(catalog.chunks[&visible.chunk.chunk_id].text.as_str()),
    )
    .unwrap();
    let mut expected = Vec::new();
    TextRecordSegment::new(
        &catalog,
        &reference,
        catalog.collection_by_id("col_1").unwrap(),
        7,
        &JsonOperationContext::standalone(),
    )
    .write(&mut expected)
    .unwrap();
    let total = 65536;
    let management = crate::temporary::test_file_management_bytes(&directory.0);
    let active = 2 * word.len()
        + 2 * management
        + 2 * std::mem::size_of::<(&crate::Chunk, usize)>()
        + 4 * std::mem::size_of::<(Vec<u8>, u64, u64)>()
        + 9;
    let padding_bytes = total - active;
    for parent in [false, true] {
        for denied in [false, true] {
            let context = JsonOperationContext::new(Arc::new(WorkLedger::with_limits(
                if parent { 2 * total } else { total },
                0,
                if parent { total } else { 2 * total },
            )));
            let mut padding = context.workspace_reservation();
            let disk = Arc::new(TemporaryDiskBudget::new(if denied {
                0
            } else {
                1024 * 1024
            }));
            let mut reads = 0;
            let (result, allocations) = crate::test_alloc::key_allocations(word.len(), || {
                build_collection(
                    &catalog,
                    catalog.collection_by_id("col_1").unwrap(),
                    7,
                    &mut |visible| {
                        if reads == 0 {
                            assert_eq!(visible.chunk.chunk_id, "chk_9999999999999999");
                            padding.grow_to(padding_bytes, padding_bytes, "operation_scratch")?;
                        } else {
                            assert_eq!(reads, 1);
                            assert!(disk.used() > 0);
                            padding.retain(0);
                        }
                        reads += 1;
                        Ok(catalog.chunks[&visible.chunk.chunk_id].text.as_str())
                    },
                    &directory.0,
                    &disk,
                    &context,
                )
            });
            if denied {
                let error = result.err().unwrap();
                assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                for (key, value) in [
                    ("resource", "temporary_disk"),
                    ("operation", "temporary_staging"),
                    ("limit_bytes", "0"),
                    ("requested_bytes", "35"),
                ] {
                    assert!(
                        error.detail().contains(&(key.into(), value.into())),
                        "{error:?}"
                    );
                }
                // A push-triggered spill would have already allocated this word.
                assert_eq!(allocations.allocated, 0);
                assert_eq!(reads, 1);
                assert_eq!(context.available_workspace_bytes(), total - padding_bytes);
                padding.retain(0);
            } else {
                let body = result.unwrap();
                assert_eq!(reads, 2);
                let mut actual = Vec::new();
                body.write(&mut actual, 7).unwrap();
                assert_eq!(actual, expected);
                drop(body);
            }
            assert_eq!(allocations.live, 0);
            assert!(!allocations.overflow);
            assert_eq!(context.available_workspace_bytes(), total);
            assert_eq!(disk.used(), 0);
            assert_eq!(std::fs::read_dir(&directory.0).unwrap().count(), 0);
        }
    }
}

#[test]
#[cfg(unix)]
fn token_normalization_refusal_cleans_staging_before_target_allocation() {
    let directory = Directory::new();
    let text = "A".repeat(65537);
    let catalog = catalog(&text);
    let limit = 8192;
    let requested = crate::temporary::test_file_management_bytes(&directory.0)
        + 2 * std::mem::size_of::<(&crate::Chunk, usize)>()
        + text.len();
    for parent in [false, true] {
        let context = JsonOperationContext::new(Arc::new(WorkLedger::with_limits(
            if parent { 2 * requested } else { limit },
            0,
            if parent { limit } else { 2 * requested },
        )));
        let disk = Arc::new(TemporaryDiskBudget::new(4096));
        let mut reads = 0;
        let (result, allocations) = crate::test_alloc::key_allocations(text.len(), || {
            build_collection(
                &catalog,
                catalog.collection_by_id("col_1").unwrap(),
                7,
                &mut |_| {
                    reads += 1;
                    Ok(text.as_str())
                },
                &directory.0,
                &disk,
                &context,
            )
        });
        let error = result.err().unwrap();
        assert_eq!(error.code(), ErrorCode::ResourceExhausted);
        for (key, value) in [
            ("resource", "memory".into()),
            ("operation", "operation_scratch".into()),
            ("limit_bytes", limit.to_string()),
            ("requested_bytes", requested.to_string()),
        ] {
            assert!(error.detail().contains(&(key.into(), value)), "{error:?}");
        }
        assert_eq!(reads, 1);
        assert_eq!(allocations.allocated, 0);
        assert_eq!(allocations.live, 0);
        assert!(!allocations.overflow);
        assert_eq!(context.available_workspace_bytes(), limit);
        assert_eq!(disk.used(), 0);
        assert_eq!(std::fs::read_dir(&directory.0).unwrap().count(), 0);
    }
}

fn append_raw(
    staging: &mut TextStaging,
    bytes: &[u8],
    commit: u64,
    context: &JsonOperationContext,
) -> TextExtent {
    let extent = staging
        .copy_body_from(
            &mut Cursor::new(&bytes[..bytes.len() - 4]),
            bytes.len() - 4,
            commit,
            context,
            0,
        )
        .unwrap();
    staging.finish_copy(&extent).unwrap();
    extent
}

fn assert_raw(staging: &TextStaging, extent: &TextExtent, expected: &[u8]) {
    let mut output = Vec::new();
    StagedText { staging, extent }
        .write(&mut output, extent.commit)
        .unwrap();
    assert_eq!(output, expected);
}

#[test]
fn adjacent_extents_replay_and_reclaim_overlap_preserve_original_bytes_across_100_cases() {
    let directory = Directory::new();
    let context = JsonOperationContext::standalone();
    let available = context.available_workspace_bytes();
    for seed in 0..100 {
        let mut first = payload();
        first.collection_id = "col_10000000000000000".into();
        first.chunks.clear();
        first.terms.clear();
        let second = payload();
        let mut third = payload();
        third.collection_id = "col_9999999999999999".into();
        third.terms.push(TextIndexTermEntry {
            term: format!("zz{}", "x".repeat(9000 + seed)),
            postings: Vec::new(),
        });
        let frames = [
            (10_000_000_000_000_000, 3, first),
            (1, 7, second),
            (9_999_999_999_999_999, 29, third),
        ]
        .map(|(id, commit, mut frame)| {
            frame.covered_commit_sequence_no = commit;
            (id, commit, encoded(&frame))
        });
        let bytes = frames.iter().map(|(_, _, frame)| frame.len() as u64).sum();
        let disk = Arc::new(TemporaryDiskBudget::new(bytes));
        let mut staging = TextStaging::new(&directory.0, disk.clone(), &context).unwrap();
        let mut bodies = Vec::new();
        for (id, commit, frame) in &frames {
            let body = append_raw(&mut staging, frame, *commit, &context);
            assert_eq!(body.collection, *id);
            bodies.push(body);
        }
        assert_eq!(disk.used(), bytes);
        assert_eq!(staging.file.metadata().unwrap().len(), bytes);
        for body in &bodies {
            staging.file.seek(SeekFrom::End(0)).unwrap();
            let frame = frames
                .iter()
                .find(|(id, _, _)| *id == body.collection)
                .unwrap();
            assert_raw(&staging, body, &frame.2);
            let mut calls = 0;
            assert_eq!(
                StagedText {
                    staging: &staging,
                    extent: body
                }
                .visit(&context, 0, &mut |_| {
                    calls += 1;
                    Err(Error::new(ErrorCode::RecoveryError, "consumer sentinel"))
                })
                .unwrap_err()
                .code(),
                ErrorCode::RecoveryError
            );
            assert_eq!(calls, 1);
            assert_raw(&staging, body, &frame.2);
        }
        let removed = if seed % 3 == 0 {
            frames[0].0
        } else {
            frames[1].0
        };
        bodies.retain(|body| {
            body.collection != removed && (seed % 3 != 2 || body.collection == frames[2].0)
        });
        let expected = frames
            .iter()
            .filter(|(id, _, _)| bodies.iter().any(|body| body.collection == *id))
            .flat_map(|(_, _, bytes)| bytes.iter().copied())
            .collect::<Vec<_>>();
        staging.reclaim(&mut bodies).unwrap();
        assert!(bodies
            .windows(2)
            .all(|pair| pair[0].collection < pair[1].collection));
        assert_eq!(std::fs::read(staging.cleanup.path()).unwrap(), expected);
        assert_eq!(disk.used(), expected.len() as u64);
        assert_eq!(staging.bytes, disk.used());
        assert_eq!(staging.reserved, disk.used());
        for body in &bodies {
            let frame = frames
                .iter()
                .find(|(id, _, _)| *id == body.collection)
                .unwrap();
            assert_raw(&staging, body, &frame.2);
        }
        drop((bodies, staging));
        assert_eq!(disk.used(), 0);
        assert_eq!(context.available_workspace_bytes(), available);
        assert_eq!(std::fs::read_dir(&directory.0).unwrap().count(), 0);
    }
}

#[test]
fn adjacent_extent_corruption_and_declared_truncation_never_read_through_next_body() {
    let directory = Directory::new();
    let context = JsonOperationContext::standalone();
    let available = context.available_workspace_bytes();
    for truncated in [false, true] {
        let disk = Arc::new(TemporaryDiskBudget::new(4096));
        let mut staging = TextStaging::new(&directory.0, disk.clone(), &context).unwrap();
        let mut empty = payload();
        empty.chunks.clear();
        empty.terms.clear();
        let first_bytes = encoded(&empty);
        let middle_bytes = encoded(&payload());
        let first = append_raw(&mut staging, &first_bytes, 7, &context);
        let mut middle = append_raw(&mut staging, &middle_bytes, 7, &context);
        let last = append_raw(&mut staging, &first_bytes, 7, &context);
        if truncated {
            middle.length -= 1;
        } else {
            staging
                .file
                .seek(SeekFrom::Start(middle.end().unwrap() - 1))
                .unwrap();
            staging
                .file
                .write_all(&[middle_bytes[middle_bytes.len() - 1] ^ 1])
                .unwrap();
        }
        let failure = || Error::new(ErrorCode::RecoveryError, "consumer sentinel");
        let view = StagedText {
            staging: &staging,
            extent: &middle,
        };
        assert_eq!(
            view.visit(&context, 0, &mut |_| Err(failure()))
                .unwrap_err()
                .code(),
            ErrorCode::ChecksumMismatch
        );
        let mut output = Vec::new();
        assert_eq!(
            view.write(&mut output, 9).unwrap_err().code(),
            ErrorCode::ChecksumMismatch
        );
        assert_eq!(output.len(), (middle.length - 4) as usize);
        assert_raw(&staging, &first, &first_bytes);
        assert_raw(&staging, &last, &first_bytes);
        let used = disk.used();
        let mut bodies = [middle, last];
        assert_eq!(
            staging.reclaim(&mut bodies).unwrap_err().code(),
            ErrorCode::ChecksumMismatch
        );
        assert!(!staging.healthy);
        assert_eq!(disk.used(), used);
        assert_eq!(
            StagedText {
                staging: &staging,
                extent: &first
            }
            .visit(&context, 0, &mut |_| Ok(()))
            .unwrap_err()
            .code(),
            ErrorCode::ChecksumMismatch
        );
        drop(staging);
        assert_eq!(disk.used(), 0);
        assert_eq!(context.available_workspace_bytes(), available);
    }
}

#[test]
#[cfg(unix)]
fn empty_extent_directory_and_single_owner_admit_exact_peak_on_both_budget_axes() {
    let directory = Directory::new();
    let mut empty = payload();
    empty.chunks.clear();
    empty.terms.clear();
    let bytes = encoded(&empty);
    assert_eq!(bytes.len(), 50);
    let width = (4 * 8 + 4usize).next_multiple_of(8);
    assert_eq!(std::mem::size_of::<TextExtent>(), width);
    let management = crate::temporary::test_file_management_bytes(&directory.0);
    for count in [0, 1, 17, 64, 511] {
        let minimum = count * width + management + 128;
        for parent in [false, true] {
            for delta in [-1isize, 0, 1] {
                let limit = (minimum as isize + delta) as usize;
                let context = JsonOperationContext::new(Arc::new(WorkLedger::with_limits(
                    if parent { 2 * minimum } else { limit },
                    0,
                    if parent { limit } else { 2 * minimum },
                )));
                let disk = Arc::new(TemporaryDiskBudget::new(50 * count as u64));
                let result: Result<()> = (|| {
                    let mut bodies =
                        ReservedVec::with_capacity(count, &context, "operation_scratch")?;
                    let mut staging = TextStaging::new(&directory.0, disk.clone(), &context)?;
                    for _ in 0..count {
                        bodies.push(append_raw(&mut staging, &bytes, 7, &context));
                    }
                    assert_eq!(
                        context.available_workspace_bytes(),
                        limit - count * width - management
                    );
                    assert_eq!(disk.used(), 50 * count as u64);
                    Ok(())
                })();
                if delta < 0 {
                    let error = result.unwrap_err();
                    assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                    for (key, value) in [
                        ("resource", "memory".to_owned()),
                        ("operation", "operation_scratch".to_owned()),
                        ("limit_bytes", limit.to_string()),
                        ("requested_bytes", minimum.to_string()),
                    ] {
                        assert!(
                            error.detail().contains(&(key.to_owned(), value)),
                            "{error:?}"
                        );
                    }
                } else {
                    result.unwrap();
                }
                assert_eq!(disk.used(), 0);
                assert_eq!(context.available_workspace_bytes(), limit);
                assert_eq!(std::fs::read_dir(&directory.0).unwrap().count(), 0);
            }
        }
    }
}

#[test]
fn growing_checkpoint_directory_refusal_releases_all_owned_resources() {
    let directory = Directory::new();
    let storage = Storage::create(directory.0.join("fixture.coffer")).unwrap();
    let entries = std::fs::read_dir(&directory.0).unwrap().count();
    let total = 256 * 1024;
    for count in [1usize, 3, 17, 64] {
        let mut catalog = catalog("");
        catalog.documents.clear();
        catalog.chunks.clear();
        let template = catalog.collections.remove("col_1").unwrap();
        for id in 1..=count + 1 {
            let mut collection = template.clone();
            collection.collection_id = format!("col_{id:016}");
            collection.name = format!("empty-{id}");
            catalog
                .collections
                .insert(collection.collection_id.clone(), collection);
        }
        for parent in [false, true] {
            let context =
                JsonOperationContext::new(Arc::new(WorkLedger::with_limits(total, total, total)));
            let disk = Arc::new(TemporaryDiskBudget::new(50 * count as u64));
            let mut staging = TextStaging::new(&directory.0, disk.clone(), &context).unwrap();
            let mut bodies =
                ReservedVec::with_capacity(count, &context, "operation_scratch").unwrap();
            for id in 1..=count {
                let empty = TextIndexSegmentPayload {
                    collection_id: format!("col_{id:016}"),
                    covered_commit_sequence_no: 7,
                    chunks: Vec::new(),
                    terms: Vec::new(),
                };
                bodies.push(append_raw(&mut staging, &encoded(&empty), 7, &context));
            }
            let reusable = TextCheckpoint {
                staging: Some(staging),
                bodies,
                stats: None,
            };
            // Collection references and the larger directory coexist with the old owner.
            let free = (count + 1) * (8 + 40) - 1;
            let occupied = context.available_workspace_bytes() - free;
            let mut blocker = context.workspace_reservation();
            let parent_blocker = if parent {
                Some(
                    context
                        .cache_reservation(occupied as u128, usize::MAX)
                        .unwrap(),
                )
            } else {
                blocker
                    .grow_to(occupied, occupied, "operation_scratch")
                    .unwrap();
                None
            };
            let error = TextCheckpoint::build(
                &storage,
                &catalog,
                99,
                Some((reusable, &BTreeSet::new())),
                |_| -> Result<&str> { panic!("growth refusal must precede primary reads") },
                &disk,
                &context,
            )
            .err()
            .unwrap();
            assert!(is_memory_budget_error(&error), "{error:?}");
            assert!(error
                .detail()
                .contains(&("requested_bytes".into(), (total + 1).to_string())));
            assert_eq!(disk.used(), 0);
            assert_eq!(std::fs::read_dir(&directory.0).unwrap().count(), entries);
            drop((blocker, parent_blocker));
            assert_eq!(context.available_workspace_bytes(), total);
        }
    }
}

#[test]
fn checkpoint_reclamation_keeps_df_zero_source_through_directory_collection_and_stats_failures() {
    use std::cell::{Cell, RefCell};
    let directory = Directory::new();
    let storage = Storage::create(directory.0.join("reclaim.coffer")).unwrap();
    for mode in 0..3 {
        let context = JsonOperationContext::new(Arc::new(WorkLedger::with_limits(
            256 * 1024,
            256 * 1024,
            256 * 1024,
        )));
        let available = context.available_workspace_bytes();
        let disk = Arc::new(TemporaryDiskBudget::new(128 * 1024));
        let mut catalog = catalog("needle");
        if mode == 2 {
            catalog.chunks.clear();
            catalog.documents.clear();
        } else {
            let mut collection = catalog.collections["col_1"].clone();
            collection.collection_id = "col_2".into();
            collection.name = "changed".into();
            catalog.collections.insert("col_2".into(), collection);
            for chunk in catalog.chunks.values_mut() {
                chunk.collection_id = "col_2".into();
            }
            for document in catalog.documents.values_mut() {
                document.collection_id = "col_2".into();
            }
        }
        let mut source = payload();
        source.chunks.clear();
        source.terms.retain(|term| term.term == "empty");
        let bytes = encoded(&source);
        let mut staging = TextStaging::new(&directory.0, disk.clone(), &context).unwrap();
        let body = append_raw(&mut staging, &bytes, 7, &context);
        let path = staging.cleanup.path().to_owned();
        let mut bodies = ReservedVec::with_capacity(1, &context, "operation_scratch").unwrap();
        bodies.push(body);
        let reusable = TextCheckpoint {
            staging: Some(staging),
            bodies,
            stats: None,
        };
        let cached = RefCell::new(None);
        if mode != 1 {
            let free = if mode == 0 {
                2 * std::mem::size_of::<&CollectionInfo>() - 1
            } else {
                std::mem::size_of::<&CollectionInfo>() + 32
            };
            let occupied = context.available_workspace_bytes() - free;
            *cached.borrow_mut() = context.cache_reservation(occupied as u128, usize::MAX);
            assert!(cached.borrow().is_some());
        }
        let reads = Cell::new(0);
        let releases = Cell::new(0);
        let result = TextCheckpoint::build_reclaiming(
            &storage,
            &catalog,
            99,
            Some((reusable, &BTreeSet::new())),
            |visible| {
                assert_eq!(visible.chunk.collection_id, "col_2");
                reads.set(reads.get() + 1);
                if mode == 1 && reads.get() == 1 {
                    *cached.borrow_mut() = context
                        .cache_reservation(context.available_workspace_bytes() as u128, usize::MAX);
                    assert!(cached.borrow().is_some());
                }
                Ok("needle")
            },
            &disk,
            &context,
            || {
                let owner = cached.borrow_mut().take();
                let freed = owner
                    .as_ref()
                    .is_some_and(|owner| owner.reserved_bytes() != 0);
                if freed {
                    releases.set(releases.get() + 1);
                }
                drop(owner);
                freed
            },
        )
        .unwrap();
        assert_eq!(releases.get(), 1);
        assert_eq!(
            reads.get(),
            if mode == 1 {
                3
            } else if mode == 0 {
                2
            } else {
                0
            }
        );
        assert_eq!(result.staging.as_ref().unwrap().cleanup.path(), path);
        let mut actual = Vec::new();
        result.body("col_1").unwrap().write(&mut actual, 7).unwrap();
        assert_eq!(actual, bytes);
        assert_eq!(
            result.stats.as_ref().unwrap().term_count(),
            if mode == 2 { 1 } else { 2 }
        );
        assert_eq!(
            result
                .stats
                .as_ref()
                .unwrap()
                .term_count_in_collection("col_1"),
            0
        );
        assert!(cached.borrow().is_none());
        drop(result);
        assert!(!path.exists());
        assert_eq!(disk.used(), 0);
        assert_eq!(context.available_workspace_bytes(), available);
    }
}

#[test]
fn unchanged_checkpoint_reuses_extent_directory_at_exact_summary_budget() {
    let directory = Directory::new();
    let storage = Storage::create(directory.0.join("fixture.coffer")).unwrap();
    let entries = std::fs::read_dir(&directory.0).unwrap().count();
    let total = 256 * 1024;
    for count in [1usize, 3, 17, 64, 511] {
        let mut catalog = catalog("");
        catalog.documents.clear();
        catalog.chunks.clear();
        let template = catalog.collections.remove("col_1").unwrap();
        for id in 1..=count {
            let mut collection = template.clone();
            collection.collection_id = format!("col_{id:016}");
            collection.name = format!("empty-{id}");
            catalog
                .collections
                .insert(collection.collection_id.clone(), collection);
        }
        // Fixtures are external to this focused preparation workspace.
        for parent in [false, true] {
            for delta in [-1isize, 0, 1] {
                let context = JsonOperationContext::new(Arc::new(WorkLedger::with_limits(
                    total, total, total,
                )));
                let disk = Arc::new(TemporaryDiskBudget::new(50 * count as u64));
                let mut staging = TextStaging::new(&directory.0, disk.clone(), &context).unwrap();
                let mut bodies =
                    ReservedVec::with_capacity(count, &context, "operation_scratch").unwrap();
                for id in 1..=count {
                    let empty = TextIndexSegmentPayload {
                        collection_id: format!("col_{id:016}"),
                        covered_commit_sequence_no: 7,
                        chunks: Vec::new(),
                        terms: Vec::new(),
                    };
                    bodies.push(append_raw(&mut staging, &encoded(&empty), 7, &context));
                }
                let original_pointer = bodies.as_slice().as_ptr();
                let reusable = TextCheckpoint {
                    staging: Some(staging),
                    bodies,
                    stats: None,
                };
                let minimum = count * 16;
                let free = (minimum as isize + delta) as usize;
                let occupied = context.available_workspace_bytes() - free;
                let mut blocker = context.workspace_reservation();
                let parent_blocker = if parent {
                    Some(
                        context
                            .cache_reservation(occupied as u128, usize::MAX)
                            .unwrap(),
                    )
                } else {
                    blocker
                        .grow_to(occupied, occupied, "operation_scratch")
                        .unwrap();
                    None
                };
                assert_eq!(context.available_workspace_bytes(), free);
                let result = TextCheckpoint::build(
                    &storage,
                    &catalog,
                    99,
                    Some((reusable, &BTreeSet::new())),
                    |_| -> Result<&str> { panic!("unchanged empty body must not rebuild") },
                    &disk,
                    &context,
                );
                if delta < 0 {
                    let error = result.err().unwrap();
                    assert!(is_memory_budget_error(&error), "{error:?}");
                    assert!(error
                        .detail()
                        .contains(&("requested_bytes".into(), (total + 1).to_string())));
                } else {
                    let result = result.unwrap();
                    assert_eq!(result.bodies.as_slice().as_ptr(), original_pointer);
                    assert_eq!(result.bodies.as_slice().len(), count);
                    assert_eq!(result.bodies.reserved_bytes(), count * 40);
                    assert_eq!(result.stats.as_ref().unwrap().term_count(), 0);
                    assert_eq!(disk.used(), 50 * count as u64);
                    assert!(result.bodies.as_slice().iter().all(|body| body.commit == 7));
                    drop(result);
                }
                drop((blocker, parent_blocker));
                assert_eq!(context.available_workspace_bytes(), total);
                assert_eq!(disk.used(), 0);
                assert_eq!(std::fs::read_dir(&directory.0).unwrap().count(), entries);
            }
        }
    }
}

#[test]
fn append_refusal_and_partial_read_rollback_keep_completed_prefix_and_credit() {
    struct Fault<'a>(Cursor<&'a [u8]>);
    impl Read for Fault<'_> {
        fn read(&mut self, bytes: &mut [u8]) -> std::io::Result<usize> {
            if self.0.position() >= 18 {
                return Err(std::io::Error::other("append read sentinel"));
            }
            let count = bytes.len().min(18 - self.0.position() as usize);
            self.0.read(&mut bytes[..count])
        }
    }
    let directory = Directory::new();
    let mut empty = payload();
    empty.chunks.clear();
    empty.terms.clear();
    let prefix = encoded(&empty);
    let next = encoded(&payload());
    for mode in ["memory", "disk", "io"] {
        let context = JsonOperationContext::standalone();
        let available = context.available_workspace_bytes();
        let sum = (prefix.len() + next.len()) as u64;
        let limit = if mode == "disk" { sum - 1 } else { sum };
        let disk = Arc::new(TemporaryDiskBudget::new(limit));
        let mut staging = TextStaging::new(&directory.0, disk.clone(), &context).unwrap();
        let first = append_raw(&mut staging, &prefix, 7, &context);
        let mut blocker = context.workspace_reservation();
        if mode == "memory" {
            let free = context.available_workspace_bytes();
            blocker.grow_to(free, free, "operation_scratch").unwrap();
        }
        let mut faulty = Fault(Cursor::new(&next[..next.len() - 4]));
        let mut healthy = Cursor::new(&next[..next.len() - 4]);
        let reader: &mut dyn Read = if mode == "io" {
            &mut faulty
        } else {
            &mut healthy
        };
        let error = staging
            .copy_body_from(reader, next.len() - 4, 7, &context, 0)
            .err()
            .unwrap();
        match mode {
            "memory" => assert!(
                matches!(error, IndexReadError::Resource(error) if crate::json::is_memory_budget_error(&error))
            ),
            "disk" => assert!(
                matches!(error, IndexReadError::Resource(error) if error.code() == ErrorCode::ResourceExhausted
                && error.detail().contains(&("requested_bytes".into(), sum.to_string()))
                && error.detail().contains(&("limit_bytes".into(), limit.to_string())))
            ),
            _ => assert!(
                matches!(error, IndexReadError::Io(error) if error.to_string() == "append read sentinel")
            ),
        }
        assert_eq!(staging.bytes, prefix.len() as u64);
        if mode != "disk" {
            assert!(staging.file.metadata().unwrap().len() > staging.bytes);
            assert_eq!(disk.used(), sum);
        }
        staging.rollback().unwrap();
        assert_eq!(disk.used(), prefix.len() as u64);
        assert_eq!(staging.reserved, staging.bytes);
        assert_eq!(std::fs::read(staging.cleanup.path()).unwrap(), prefix);
        drop(blocker);
        assert_raw(&staging, &first, &prefix);
        let bytes = if mode == "disk" { &prefix } else { &next };
        let second = append_raw(&mut staging, bytes, 7, &context);
        assert_raw(&staging, &first, &prefix);
        assert_raw(&staging, &second, bytes);
        assert_eq!(disk.used(), (prefix.len() + bytes.len()) as u64);
        drop(staging);
        assert_eq!(disk.used(), 0);
        assert_eq!(context.available_workspace_bytes(), available);
        assert_eq!(std::fs::read_dir(&directory.0).unwrap().count(), 0);
    }
}

#[test]
#[cfg(unix)]
fn failed_truncate_or_move_keeps_credit_and_poisoned_owner_until_cleanup() {
    let directory = Directory::new();
    let context = JsonOperationContext::standalone();
    let available = context.available_workspace_bytes();
    let bytes = encoded(&payload());
    for moving in [false, true] {
        let disk = Arc::new(TemporaryDiskBudget::new(4096));
        let mut staging = TextStaging::new(&directory.0, disk.clone(), &context).unwrap();
        let first = append_raw(&mut staging, &bytes, 7, &context);
        let second = append_raw(&mut staging, &bytes, 7, &context);
        let credit = disk.used();
        let path = staging.cleanup.path().to_owned();
        // Same inode, read-only descriptor: both write and truncate fail for real.
        *staging.file = File::open(&path).unwrap();
        let error = if moving {
            staging.reclaim(&mut [second]).unwrap_err()
        } else {
            staging.rollback().unwrap_err()
        };
        assert_eq!(error.code(), ErrorCode::IoError);
        assert!(!staging.healthy);
        assert_eq!(disk.used(), credit);
        assert_eq!(staging.file.metadata().unwrap().len(), credit);
        assert_eq!(
            staging.begin().unwrap_err().code(),
            ErrorCode::ChecksumMismatch
        );
        let view = StagedText {
            staging: &staging,
            extent: &first,
        };
        assert_eq!(
            view.write(&mut Vec::new(), 9).unwrap_err().code(),
            ErrorCode::ChecksumMismatch
        );
        assert_eq!(
            view.visit(&context, 0, &mut |_| Ok(())).unwrap_err().code(),
            ErrorCode::ChecksumMismatch
        );
        drop(staging);
        assert!(!path.exists());
        assert_eq!(disk.used(), 0);
        assert_eq!(context.available_workspace_bytes(), available);
    }
}
