use cofferdb::{
    ChunkInput, CollectionConfig, Database, DatabaseConfig, DistanceMetric, DocumentInput,
    EmbeddingInput, JsonValue, QueryEmbeddingInput, VectorIndexConfig,
};

fn main() -> cofferdb::Result<()> {
    let path = std::env::temp_dir().join(format!(
        "cofferdb-embedded-demo-{}.coffer",
        std::process::id()
    ));
    let db = Database::create(&path, DatabaseConfig::default())?;
    let collection = db.create_collection(CollectionConfig {
        name: "notes".into(),
        tokenizer: Default::default(),
        embedding_dim: 2,
        distance_metric: DistanceMetric::Cosine,
        embedding_model: Some("demo-v1".into()),
        vectors_normalized: false,
        vector_index: VectorIndexConfig::Exact,
    })?;
    collection.insert_document(DocumentInput {
        external_id: "wal".into(),
        title: Some("Write-ahead log".into()),
        source_uri: None,
        metadata: JsonValue::object(),
        chunks: vec![ChunkInput {
            chunk_external_id: "wal#commit".into(),
            text: "Write ahead logging enables recovery".into(),
            ordinal: 0,
            metadata: JsonValue::object(),
            source_span: None,
        }],
    })?;
    collection.put_embeddings(vec![EmbeddingInput {
        chunk_external_id: "wal#commit".into(),
        model_name: "demo-v1".into(),
        dim: 2,
        vector: vec![1.0, 0.0],
    }])?;
    println!(
        "{}",
        collection
            .search()
            .text("write ahead")
            .query_embedding(QueryEmbeddingInput {
                model_name: Some("demo-v1".into()),
                dim: 2,
                vector: vec![1.0, 0.0]
            })
            .execute()?
            .to_json()
            .to_compact_string()
    );
    drop(collection);
    drop(db);
    std::fs::remove_file(path)?;
    Ok(())
}
