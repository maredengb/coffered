use super::*;
use crate::json::JsonProcessingPolicyV1;

#[test]
fn embedding_command_streams_past_legacy_file_limit_and_retains_compatible_output() {
    use std::io::Write;
    let path = temp_path("embedding-command-large-source");
    let input = temp_path("embedding-command-large-input");
    let db = Database::create(&path, DatabaseConfig::default()).unwrap();
    let col = db
        .create_collection(CollectionConfig {
            name: "notes".into(),
            embedding_dim: 2,
            distance_metric: DistanceMetric::Cosine,
            embedding_model: Some("m".into()),
            vectors_normalized: false,
            vector_index: VectorIndexConfig::Exact,
            tokenizer: Default::default(),
        })
        .unwrap();
    col.insert_document(DocumentInput {
        external_id: "document".into(),
        title: None,
        source_uri: None,
        metadata: JsonValue::object(),
        chunks: (0..2)
            .map(|ordinal| ChunkInput {
                chunk_external_id: format!("c{ordinal}"),
                ordinal,
                text: "topic".into(),
                metadata: JsonValue::object(),
                source_span: None,
            })
            .collect(),
    })
    .unwrap();
    drop((col, db));
    let mut output = fs::File::create(&input).unwrap();
    output
        .write_all(
            b"{\"chunk_external_id\":\"c0\",\"model_name\":\"m\",\"dim\":2,\"vector\":[1,0]}\n",
        )
        .unwrap();
    let mut padding = [b' '; 65536];
    padding[65535] = b'\n';
    for _ in 0..1025 {
        output.write_all(&padding).unwrap();
    }
    output
        .write_all(
            b"{\"chunk_external_id\":\"c1\",\"model_name\":\"m\",\"dim\":2,\"vector\":[0,1]}\n",
        )
        .unwrap();
    drop(output);
    assert!(fs::metadata(&input).unwrap().len() > crate::import::MAX_JSONL_BYTES);
    let result = run_command(
        "import-embeddings",
        Args::new(vec![
            path.display().to_string(),
            input.display().to_string(),
            "--collection".into(),
            "notes".into(),
            "--durability".into(),
            "fast".into(),
        ]),
    )
    .unwrap();
    let expected = parse(r#"{"documents_inserted":0,"documents_imported":0,"chunks_inserted":0,"chunks_imported":0,"embeddings_inserted":2,"embeddings_imported":2,"documents_skipped":0,"chunks_skipped":0,"embeddings_skipped":0,"skipped":0,"documents_replaced":0,"chunks_replaced":0,"embeddings_replaced":0,"replaced":0,"missing_chunks":0,"model_mismatches":0,"errors":0,"durability":"fast"}"#).unwrap();
    assert_eq!(result.value, expected);
    let context = result.retained.as_ref().unwrap().0.clone();
    let before = fs::read(&path).unwrap();
    assert!(context.available_workspace_bytes() < JsonProcessingPolicyV1::WORK_BUDGET_BYTES);
    assert_eq!(
        success_output("import-embeddings", result, 7).unwrap(),
        cli_envelope("import-embeddings", true, expected, None, 7)
    );
    assert_eq!(
        context.available_workspace_bytes(),
        JsonProcessingPolicyV1::WORK_BUDGET_BYTES
    );
    assert_eq!(fs::read(&path).unwrap(), before);
    assert_eq!(
        Database::open_read_only(&path)
            .unwrap()
            .collection("notes")
            .unwrap()
            .stats()
            .unwrap()
            .embedding_count,
        2
    );
    fs::remove_file(path).unwrap();
    fs::remove_file(input).unwrap();
}
