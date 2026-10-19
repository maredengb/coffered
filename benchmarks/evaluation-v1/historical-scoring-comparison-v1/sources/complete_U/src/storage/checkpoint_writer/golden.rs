use super::*;
use crate::catalog::Catalog;
use crate::index::text::TextIndex;
use crate::json::JsonOperationContext;
use crate::model::VectorIndexConfig;
use std::io::Cursor;

const BATCH: u64 = 0x1122_3344_5566_7788;
const COMMIT: u64 = 0x1020_3040_5060_7080;
const CHUNKS: [u64; 2] = [9_999_999_999_999_999, 10_000_000_000_000_000];
const DOCUMENTS: [u64; 2] = [0x1112_1314_1516_1718, 0x2122_2324_2526_2728];
const EMBEDDINGS: [u64; 2] = [0x3132_3334_3536_3738, 0x4142_4344_4546_4748];
const TEXT: &str = "
    0200 0100000000000000 8070605040302010 0100 0100 9a99993f 0000403f
    0200000000000000
    ffffc06ff2862300 1817161514131211 0400000001000000
    0000c16ff2862300 2827262524232221 0200000000000000
    0200000000000000 01000000 61 0200000000000000 0200000000000000
    ffffc06ff2862300 0100000001000000 0000c16ff2862300 0200000000000000
    02000000 c3a9 0100000000000000 0100000000000000
    ffffc06ff2862300 0300000000000000 6a1867e9";
const EMPTY_TEXT: &str = "
    0200 0100000000000000 8070605040302010 0100 0100 9a99993f 0000403f
    0000000000000000 0000000000000000 49d29f4a";
const DF_ZERO_TEXT: &str = "
    0200 0100000000000000 8070605040302010 0100 0100 9a99993f 0000403f
    0000000000000000 0100000000000000
    01000000 7a 0000000000000000 0000000000000000 2d689f82";
const VECTOR: &str = "
    0100 0100000000000000 8070605040302010 0200 03000000
    02 01 02000000 c3a9 00 0200000000000000
    ffffc06ff2862300 3837363534333231 0000803f 00000080 01000000
    0000c16ff2862300 4847464544434241 000000bf 00000000 ffff7f7f
    d618d3e0";
const EMPTY_VECTOR_NONE: &str = "
    0100 0100000000000000 8070605040302010 0200 03000000
    01 00 01 0000000000000000 f75294ab";
const EMPTY_VECTOR_SOME: &str = "
    0100 0100000000000000 8070605040302010 0200 03000000
    03 01 00000000 00 0000000000000000 258c0e74";
const METADATA: &str = "
    0100 8877665544332211 8070605040302010
    0200000000000000 ec75e73a 0200000000000000
    2e00000000000000 5c00000000000000 5dba9827";

fn bytes(hex: &str, length: usize) -> Vec<u8> {
    let hex = hex.split_ascii_whitespace().collect::<String>();
    assert_eq!(hex.len(), length * 2);
    (0..hex.len())
        .step_by(2)
        .map(|offset| u8::from_str_radix(&hex[offset..offset + 2], 16).unwrap())
        .collect()
}

// Bitwise Castagnoli, independent of the engine's table and CRC-combine path.
fn crc(input: &[u8]) -> u32 {
    let mut result = u32::MAX;
    for &byte in input {
        result ^= u32::from(byte);
        for _ in 0..8 {
            result = (result >> 1) ^ (0x82f6_3b78 & 0u32.wrapping_sub(result & 1));
        }
    }
    !result
}

fn segment_bytes(hex: &str, length: usize) -> Vec<u8> {
    assert_eq!(crc(b"123456789"), 0xe306_9283);
    let result = bytes(hex, length);
    assert_eq!(
        crc(&result[..length - 4]).to_le_bytes(),
        result[length - 4..]
    );
    result
}

fn id(prefix: &str, value: u64) -> String {
    format!("{prefix}_{value:016}")
}

fn metadata() -> IndexCheckpointMetadata {
    IndexCheckpointMetadata {
        covered_last_batch_id: BATCH,
        covered_last_commit_sequence_no: COMMIT,
        coverage_hash: 0,
        collection_count: 1,
        live_document_count: 2,
        live_chunk_count: 2,
        live_embedding_count: 2,
        text_segment_count: 1,
        vector_segment_count: 1,
        build_mode: "full_checkpoint".into(),
    }
}

fn text_payload(empty: bool) -> TextIndexSegmentPayload {
    TextIndexSegmentPayload {
        collection_id: id("col", 1),
        covered_commit_sequence_no: COMMIT,
        chunks: if empty {
            Vec::new()
        } else {
            (0..2)
                .rev()
                .map(|i| TextIndexChunkEntry {
                    chunk_id: id("chk", CHUNKS[i]),
                    document_id: id("doc", DOCUMENTS[i]),
                    token_count: [4_294_967_300, 2][i],
                })
                .collect()
        },
        terms: if empty {
            Vec::new()
        } else {
            vec![
                TextIndexTermEntry {
                    term: "\u{e9}".into(),
                    postings: vec![TextIndexPostingEntry {
                        chunk_id: id("chk", CHUNKS[0]),
                        term_frequency: 3,
                    }],
                },
                TextIndexTermEntry {
                    term: "a".into(),
                    postings: (0..2)
                        .rev()
                        .map(|i| TextIndexPostingEntry {
                            chunk_id: id("chk", CHUNKS[i]),
                            term_frequency: [4_294_967_297, 2][i],
                        })
                        .collect(),
                },
            ]
        },
    }
}

fn vector_payload(case: usize) -> VectorIndexSegmentPayload {
    VectorIndexSegmentPayload {
        collection_id: id("col", 1),
        covered_commit_sequence_no: COMMIT,
        dim: 3,
        distance_metric: [
            DistanceMetric::DotProduct,
            DistanceMetric::Cosine,
            DistanceMetric::Euclidean,
        ][case],
        embedding_model: [Some("\u{e9}"), None, Some("")][case].map(str::to_owned),
        vectors_normalized: case == 1,
        vectors: if case == 0 {
            (0..2)
                .rev()
                .map(|i| VectorIndexEntry {
                    chunk_id: id("chk", CHUNKS[i]),
                    embedding_id: id("emb", EMBEDDINGS[i]),
                    vector: [[0x3f80_0000, 0x8000_0000, 1], [0xbf00_0000, 0, 0x7f7f_ffff]][i]
                        .map(f32::from_bits)
                        .to_vec(),
                })
                .collect()
        } else {
            Vec::new()
        },
    }
}

fn collection(vector: &VectorIndexSegmentPayload) -> CollectionInfo {
    CollectionInfo {
        collection_id: vector.collection_id.clone(),
        name: "golden".into(),
        tokenizer: Default::default(),
        embedding_dim: vector.dim,
        distance_metric: vector.distance_metric,
        embedding_model: vector.embedding_model.clone(),
        vectors_normalized: vector.vectors_normalized,
        vector_index: VectorIndexConfig::Exact,
        created_at: "1970-01-01T00:00:00Z".into(),
    }
}

fn text_catalog(payload: &TextIndexSegmentPayload) -> Catalog {
    let mut catalog = Catalog::default();
    let collection = collection(&vector_payload(0));
    catalog
        .collections
        .insert(collection.collection_id.clone(), collection);
    for chunk in &payload.chunks {
        catalog.documents.insert(
            chunk.document_id.clone(),
            Box::new(Document {
                document_id: chunk.document_id.clone(),
                collection_id: payload.collection_id.clone(),
                external_id: chunk.document_id.clone(),
                title: None,
                source_uri: None,
                metadata: JsonValue::object(),
                created_at: String::new(),
                updated_at: String::new(),
            }),
        );
        catalog.chunks.insert(
            chunk.chunk_id.clone(),
            Box::new(Chunk {
                chunk_id: chunk.chunk_id.clone(),
                collection_id: payload.collection_id.clone(),
                document_id: chunk.document_id.clone(),
                chunk_external_id: chunk.chunk_id.clone(),
                text: String::new(),
                text_hash: String::new(),
                ordinal: 0,
                token_count: 1,
                metadata: JsonValue::object(),
                source_span: None,
                embedding_id: None,
                created_at: String::new(),
            }),
        );
    }
    catalog
}

#[test]
fn metadata_fixed_bytes_freeze_widths_order_and_empty_references() {
    let high = bytes(
        "0100 8877665544332211 8070605040302010
         0200000001000000 efcdab89 0200000001000000
         0807060504030201 5c0000002e000000 df9b5713",
        58,
    );
    assert_eq!(
        build_index_commit_metadata_payload(
            &metadata(),
            0x1_0000_0002,
            0x0102_0304_0506_0708,
            0x2e_0000_005c,
            0x1357_9bdf,
            0x89ab_cdef,
        ),
        high.as_slice()
    );
    // Sentinel CRCs above freeze scalar layout, not a giant valid directory.
    let empty = bytes(
        "0100 8877665544332211 8070605040302010
         0000000000000000 1ff426ad 0000000000000000
         0000000000000000 0000000000000000 00000000",
        58,
    );
    assert_eq!(
        index_commit_checksum(BATCH, COMMIT, 0x8877_6655_4433_2211, &[]),
        0xad26_f41f
    );
    assert_eq!(
        build_index_commit_metadata_payload(&metadata(), 0, 0, 0, 0, 0xad26_f41f),
        empty.as_slice()
    );
    let parse = |input: &[u8]| {
        parse_index_commit_metadata_payload(input, BATCH, COMMIT, &[], &BTreeMap::new(), 0)
    };
    let decoded = parse(&empty).unwrap();
    assert_eq!(decoded.index_record_count, 0);
    assert_eq!(decoded.index_checksum, 0xad26_f41f);
    assert_eq!(decoded.extent_range, 0..0);
    for end in 0..empty.len() {
        assert_eq!(
            parse(&empty[..end]).err().unwrap().code,
            "index_commit_metadata_invalid"
        );
    }
    let mut trailing = empty.clone();
    trailing.push(0);
    assert_eq!(
        parse(&trailing).err().unwrap().code,
        "index_commit_metadata_invalid"
    );
    for offset in [0, 2, 10, 18, 30, 38, 46, 54] {
        let mut invalid = empty.clone();
        invalid[offset] ^= 2;
        assert_eq!(
            parse(&invalid).err().unwrap().code,
            "index_commit_metadata_invalid"
        );
    }
}

#[test]
fn text_fixed_bytes_cover_real_record_writer_and_direct_decode() {
    for (empty, hex, length) in [(false, TEXT, 189), (true, EMPTY_TEXT, 50)] {
        let golden = segment_bytes(hex, length);
        let payload = text_payload(empty);
        assert_eq!(encode_text_index_segment_payload(&payload).unwrap(), golden);
        let catalog = text_catalog(&payload);
        let index = TextIndex::try_build_from_segments(vec![payload]).unwrap();
        let context = JsonOperationContext::standalone();
        let record = TextRecordSegment::new(
            &catalog,
            &index,
            catalog.collections.values().next().unwrap(),
            COMMIT,
            &context,
        );
        let mut output = Vec::new();
        record.write(&mut output).unwrap();
        assert_eq!(output, golden);
        let decoded = parse_text_index_segment_payload(&golden, COMMIT, 0)
            .unwrap()
            .payload;
        assert_eq!(decoded.collection_id, "col_0000000000000001");
        assert_eq!(decoded.covered_commit_sequence_no, COMMIT);
        assert_eq!(decoded.chunks.len(), if empty { 0 } else { 2 });
        assert_eq!(decoded.terms.len(), if empty { 0 } else { 2 });
        if !empty {
            for i in 0..2 {
                assert_eq!(decoded.chunks[i].chunk_id, id("chk", CHUNKS[i]));
                assert_eq!(decoded.chunks[i].document_id, id("doc", DOCUMENTS[i]));
                assert_eq!(decoded.chunks[i].token_count, [4_294_967_300, 2][i]);
            }
            assert_eq!(decoded.terms[0].term, "a");
            assert_eq!(decoded.terms[1].term, "\u{e9}");
            assert_eq!(decoded.terms[0].postings.len(), 2);
            assert_eq!(decoded.terms[1].postings.len(), 1);
            for i in 0..2 {
                assert_eq!(decoded.terms[0].postings[i].chunk_id, id("chk", CHUNKS[i]));
                assert_eq!(
                    decoded.terms[0].postings[i].term_frequency,
                    [4_294_967_297, 2][i]
                );
            }
            assert_eq!(decoded.terms[1].postings[0].chunk_id, id("chk", CHUNKS[0]));
            assert_eq!(decoded.terms[1].postings[0].term_frequency, 3);
        }
        reject_body_truncations(&golden, |input, body| {
            decode_text_index_segment(&mut IndexPayloadReader::new(input, body, 0), COMMIT, 0)
                .map(|_| ())
        });
    }
    let df_zero = segment_bytes(DF_ZERO_TEXT, 71);
    let decoded = parse_text_index_segment_payload(&df_zero, COMMIT, 0)
        .unwrap()
        .payload;
    assert!(decoded.chunks.is_empty());
    assert_eq!(decoded.terms.len(), 1);
    assert_eq!(decoded.terms[0].term, "z");
    assert!(decoded.terms[0].postings.is_empty());
}

#[test]
fn vector_fixed_bytes_cover_borrowed_owned_cache_and_direct_decode() {
    for (case, hex, length) in [
        (0, VECTOR, 101),
        (1, EMPTY_VECTOR_NONE, 39),
        (2, EMPTY_VECTOR_SOME, 43),
    ] {
        let golden = segment_bytes(hex, length);
        let mut payload = vector_payload(case);
        assert_eq!(
            encode_vector_index_segment_payload(&payload).unwrap(),
            golden
        );
        assert_eq!(payload.encoded_len().unwrap(), length as u64 + 50);
        payload.vectors.reverse();
        let context = JsonOperationContext::standalone();
        let credit = CachedVectorSegment::reserve(&context, 4096, length as u64 + 50).unwrap();
        let cached = CachedVectorSegment::new(payload, credit);
        let mut output = Vec::new();
        Segment::BorrowedCachedVector(&cached)
            .encode(&mut output)
            .unwrap();
        assert_eq!(output, golden);
        output.clear();
        Segment::CachedVector(cached).encode(&mut output).unwrap();
        assert_eq!(output, golden);
        let decoded = parse_vector_index_segment_payload(&golden, COMMIT, 0)
            .unwrap()
            .payload
            .unwrap();
        let expected = vector_payload(case);
        assert_eq!(decoded.collection_id, "col_0000000000000001");
        assert_eq!(decoded.covered_commit_sequence_no, COMMIT);
        assert_eq!(decoded.dim, 3);
        assert_eq!(decoded.distance_metric, expected.distance_metric);
        assert_eq!(decoded.embedding_model, expected.embedding_model);
        assert_eq!(decoded.vectors_normalized, expected.vectors_normalized);
        assert_eq!(decoded.vectors.len(), expected.vectors.len());
        for (i, vector) in decoded.vectors.iter().enumerate() {
            assert_eq!(vector.chunk_id, id("chk", CHUNKS[i]));
            assert_eq!(vector.embedding_id, id("emb", EMBEDDINGS[i]));
            assert_eq!(
                vector
                    .vector
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                [[0x3f80_0000, 0x8000_0000, 1], [0xbf00_0000, 0, 0x7f7f_ffff]][i]
            );
        }
        reject_body_truncations(&golden, |input, body| {
            decode_vector_index_segment(&mut IndexPayloadReader::new(input, body, 0), COMMIT, 0)
                .map(|_| ())
        });
    }
}

fn reject_body_truncations(
    golden: &[u8],
    decode: impl Fn(&[u8], usize) -> std::result::Result<(), IndexReadError>,
) {
    let body = golden.len() - 4;
    for end in 0..body {
        assert!(decode(&golden[..end], body).is_err(), "body end {end}");
    }
}

fn replace_with_crc(golden: &[u8], offset: usize, replacement: &[u8]) -> Vec<u8> {
    let mut bytes = golden[..golden.len() - 4].to_vec();
    bytes[offset..offset + replacement.len()].copy_from_slice(replacement);
    bytes.extend_from_slice(&crc(&bytes).to_le_bytes());
    bytes
}

fn diagnostic<T>(result: std::result::Result<T, IndexReadError>) -> &'static str {
    match result {
        Err(IndexReadError::Invalid(error)) => error.code,
        Err(error) => panic!("expected index diagnostic, got {error:?}"),
        Ok(_) => panic!("damaged index was accepted"),
    }
}

#[test]
fn fixed_segments_reject_semantic_damage_even_with_valid_independent_crc() {
    let text = segment_bytes(TEXT, 189);
    let text_error = |input: &[u8]| diagnostic(parse_text_index_segment_payload(input, COMMIT, 0));
    let vector_error =
        |input: &[u8]| diagnostic(parse_vector_index_segment_payload(input, COMMIT, 0));
    for (offset, value, code) in [
        (0, 3, "index_segment_header_mismatch"),
        (18, 2, "index_segment_header_mismatch"),
        (20, 2, "index_tokenizer_mismatch"),
        (22, 0, "index_bm25_params_mismatch"),
        (98, 0xff, "index_segment_non_canonical"),
        (99, 1, "index_segment_non_canonical"),
    ] {
        assert_eq!(text_error(&replace_with_crc(&text, offset, &[value])), code);
    }
    for (offset, replacement) in [(62, &text[38..46]), (131, &text[115..123])] {
        assert_eq!(
            text_error(&replace_with_crc(&text, offset, replacement)),
            "index_segment_non_canonical"
        );
    }
    let vector = segment_bytes(VECTOR, 101);
    for (offset, value, code) in [
        (0, 2, "index_segment_header_mismatch"),
        (18, 3, "index_segment_header_mismatch"),
        (24, 0, "index_segment_header_mismatch"),
        (25, 2, "index_segment_non_canonical"),
        (32, 2, "index_segment_header_mismatch"),
    ] {
        assert_eq!(
            vector_error(&replace_with_crc(&vector, offset, &[value])),
            code
        );
    }
    for replacement in [0x7f80_0000u32, 0x7fc0_0001] {
        assert_eq!(
            vector_error(&replace_with_crc(&vector, 57, &replacement.to_le_bytes())),
            "index_segment_non_canonical"
        );
    }
    for offset in [62, 123, 131] {
        assert_eq!(
            text_error(&replace_with_crc(&text, offset, &0u64.to_le_bytes())),
            "index_segment_non_canonical"
        );
    }
    assert_eq!(
        vector_error(&replace_with_crc(&vector, 69, &vector[41..49])),
        "index_segment_non_canonical"
    );
    assert_eq!(
        vector_error(&replace_with_crc(&vector, 69, &0u64.to_le_bytes())),
        "index_segment_non_canonical"
    );
    for (golden, is_text) in [(&text, true), (&vector, false)] {
        let mut trailing = golden[..golden.len() - 4].to_vec();
        trailing.push(0);
        trailing.extend_from_slice(&crc(&trailing).to_le_bytes());
        let code = if is_text {
            text_error(&trailing)
        } else {
            vector_error(&trailing)
        };
        assert_eq!(code, "index_segment_non_canonical");
    }
}

#[test]
fn real_checkpoint_writer_matches_independent_metadata_and_segment_goldens() {
    let text = text_payload(false);
    let catalog = text_catalog(&text);
    let index = TextIndex::try_build_from_segments(vec![text]).unwrap();
    let mut vector = vector_payload(0);
    vector.vectors.reverse();
    let context = JsonOperationContext::standalone();
    let credit = CachedVectorSegment::reserve(&context, 4096, 151).unwrap();
    let cached = CachedVectorSegment::new(vector, credit);
    let mut output = Cursor::new(vec![7; 37]);
    write_checkpoint_segments_in(
        &mut output,
        &metadata(),
        [
            Ok(Segment::TextRecords(TextRecordSegment::new(
                &catalog,
                &index,
                catalog.collections.values().next().unwrap(),
                COMMIT,
                &context,
            ))),
            Ok(Segment::BorrowedCachedVector(&cached)),
        ],
        37,
        |_, _| {},
        reserve_directory(&metadata(), &context).unwrap(),
        false,
    )
    .unwrap();
    assert_eq!(&output.get_ref()[..37], &[7; 37]);
    assert_eq!(&output.get_ref()[192..250], bytes(METADATA, 58));
    assert_eq!(&output.get_ref()[300..489], segment_bytes(TEXT, 189));
    assert_eq!(&output.get_ref()[539..640], segment_bytes(VECTOR, 101));
}

#[test]
fn record_backed_vector_writer_reads_real_payloads_into_fixed_wire_bytes() {
    static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let path = std::env::temp_dir().join(format!(
        "cofferdb-index-golden-vector-{}-{}.coffer",
        std::process::id(),
        NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
    ));
    let mut storage = Storage::create(&path).unwrap();
    let guard = crate::temporary::TempFileGuard::new(path);
    let payload = vector_payload(0);
    let collection = collection(&payload);
    let mut summaries = payload
        .vectors
        .iter()
        .map(|entry| Embedding {
            embedding_id: entry.embedding_id.clone(),
            chunk_id: entry.chunk_id.clone(),
            collection_id: collection.collection_id.clone(),
            chunk_external_id: "not-persisted".into(),
            model_name: collection.embedding_model.clone().unwrap(),
            dim: 3,
            vector: entry.vector.clone(),
            created_at: "1970-01-01T00:00:00Z".into(),
        })
        .collect::<Vec<_>>();
    let records = std::iter::once(RecordPayload::CollectionCreate(collection.clone()))
        .chain(summaries.iter().cloned().map(RecordPayload::EmbeddingPut))
        .collect::<Vec<_>>();
    let append = storage.append_batch(&records, Durability::Fast).unwrap();
    for summary in &mut summaries {
        summary.vector.clear();
    }
    assert!(summaries.iter().all(|summary| summary.vector.is_empty()));
    let context = JsonOperationContext::standalone();
    let mut source = VectorRecordSegment::new(
        &collection,
        COMMIT,
        summaries
            .iter()
            .zip(&append.records[1..])
            .map(|(summary, record)| (summary, Some(&record.record_ref))),
        storage.payload_reader().unwrap(),
        &context,
    )
    .unwrap();
    assert_eq!(source.encoded_len().unwrap(), 151);
    let mut output = Vec::new();
    source.write(&mut output).unwrap();
    assert_eq!(source.encode_calls, 1);
    assert_eq!(output, segment_bytes(VECTOR, 101));
    let cached = source.try_materialize(4096).unwrap().unwrap();
    output.clear();
    Segment::CachedVector(cached).encode(&mut output).unwrap();
    assert_eq!(output, segment_bytes(VECTOR, 101));
    drop(source);
    drop(storage);
    drop(guard);
}
