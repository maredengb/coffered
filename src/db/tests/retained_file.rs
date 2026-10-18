use super::*;
use std::os::unix::fs::MetadataExt;

fn identity(metadata: &fs::Metadata) -> (u64, u64) {
    (metadata.dev(), metadata.ino())
}

fn signature(result: &SearchResults) -> Vec<(String, u64, Option<u64>)> {
    result
        .results
        .iter()
        .map(|hit| {
            (
                hit.chunk_external_id.clone(),
                hit.final_score.to_bits(),
                hit.bm25_score.map(f64::to_bits),
            )
        })
        .collect()
}

#[test]
fn compact_generations_remain_charged_through_writer_close_and_reader_aliases() {
    for durability in [Durability::Fast, Durability::Commit] {
        for rebuild_indexes in [false, true] {
            let directory = temp_path(&format!(
                "retained-compact-{durability:?}-{rebuild_indexes}"
            ))
            .with_extension("dir");
            fs::create_dir(&directory).unwrap();
            let path = directory.join("data.coffer");
            let writer = Database::create(
                &path,
                DatabaseConfig {
                    durability,
                    ..Default::default()
                },
            )
            .unwrap();
            let collection = writer
                .create_collection(CollectionConfig {
                    name: "retained".into(),
                    embedding_dim: 2,
                    distance_metric: DistanceMetric::Cosine,
                    embedding_model: Some("m".into()),
                    vectors_normalized: false,
                    vector_index: VectorIndexConfig::Exact,
                    tokenizer: Default::default(),
                })
                .unwrap();
            let input = |id: &str| DocumentInput {
                external_id: id.into(),
                title: None,
                source_uri: None,
                metadata: JsonValue::object(),
                chunks: vec![ChunkInput {
                    chunk_external_id: id.into(),
                    text: "shared history".into(),
                    ordinal: 0,
                    metadata: JsonValue::object(),
                    source_span: None,
                }],
            };
            collection.insert_document(input("first")).unwrap();
            let first_reader = Database::open_read_only(&path).unwrap();
            let first_collection = first_reader.collection("retained").unwrap();
            let first_query = first_collection.search().text("shared");
            let first_result = first_query.clone().execute().unwrap();
            assert_eq!(first_result.results[0].chunk_external_id, "first");
            assert_eq!(first_result.results.len(), 1);

            collection.insert_document(input("second")).unwrap();
            let first_metadata = fs::metadata(&path).unwrap();
            let first_bytes = first_metadata.len();
            writer.compact(CompactOptions { rebuild_indexes }).unwrap();
            let second_identity = identity(&fs::metadata(&path).unwrap());
            assert_ne!(identity(&first_metadata), second_identity);
            let disk = Arc::clone(&writer.temporary_work);
            assert_eq!(disk.used(), first_bytes);
            let second_reader = Database::open_read_only(&path).unwrap();
            let second_alias = second_reader.clone();
            let second_query = second_reader
                .collection("retained")
                .unwrap()
                .search()
                .text("shared");
            let second_result = second_query.clone().execute().unwrap();
            assert_eq!(second_result.results.len(), 2);
            assert_eq!(second_result.results[1].chunk_external_id, "second");

            let document = collection.get_document_by_external_id("first").unwrap();
            collection.delete_document(&document.document_id).unwrap();
            let second_bytes = fs::metadata(&path).unwrap().len();
            writer.compact(CompactOptions { rebuild_indexes }).unwrap();
            let third_identity = identity(&fs::metadata(&path).unwrap());
            assert_ne!(identity(&first_metadata), third_identity);
            assert_ne!(second_identity, third_identity);
            assert_eq!(disk.used(), first_bytes + second_bytes);
            assert_eq!(
                collection
                    .search()
                    .text("shared")
                    .execute()
                    .unwrap()
                    .results[0]
                    .chunk_external_id,
                "second"
            );

            writer.close().unwrap();
            assert_eq!(disk.used(), first_bytes + second_bytes);
            assert_eq!(
                signature(&first_query.clone().execute().unwrap()),
                signature(&first_result)
            );
            assert_eq!(
                signature(&second_query.clone().execute().unwrap()),
                signature(&second_result)
            );
            drop((first_reader, first_collection));
            assert_eq!(disk.used(), first_bytes + second_bytes);
            drop(first_query);
            assert_eq!(disk.used(), second_bytes);

            // Explicit close releases descriptors even while closed aliases survive.
            second_reader.close().unwrap();
            assert_eq!(disk.used(), 0);
            assert_eq!(second_alias.stats().unwrap_err().code(), ErrorCode::IoError);
            assert_eq!(
                second_query.execute().unwrap_err().code(),
                ErrorCode::IoError
            );
            assert_eq!(first_result.results[0].chunk_external_id, "first");
            assert_eq!(second_result.results.len(), 2);
            let reopened = Database::open_read_only(&path).unwrap();
            let results = reopened
                .collection("retained")
                .unwrap()
                .search()
                .text("shared")
                .execute()
                .unwrap();
            assert_eq!(results.results.len(), 1);
            assert_eq!(results.results[0].chunk_external_id, "second");
            reopened.close().unwrap();
            assert_eq!(disk.used(), 0);
            drop((collection, second_alias));
            fs::remove_dir_all(directory).unwrap();
        }
    }
}
