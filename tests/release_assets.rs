use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use cofferdb::{
    parse_json, CompactOptions, Database, DatabaseConfig, ErrorCode, OnConflict, TokenizerConfig,
};

struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "cofferdb-assets-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn published_markdown_and_code_demo_inputs_are_importable() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let temp = Temp::new();
    for (demo, collection, query, filter, expected) in [
        ("markdown", "notes", "write ahead", "{}", "wal#commit"),
        (
            "code",
            "code",
            "page cache",
            r#"{"doc.language":"rust"}"#,
            "storage#cache",
        ),
    ] {
        let db = Database::create(
            temp.0.join(format!("{demo}.coffer")),
            DatabaseConfig::default(),
        )
        .unwrap();
        db.import_manifest(
            root.join("examples")
                .join("data")
                .join(demo)
                .join("import_manifest.json"),
            OnConflict::Error,
        )
        .unwrap();
        let result = db
            .collection(collection)
            .unwrap()
            .search()
            .text(query)
            .filter(parse_json(filter).unwrap())
            .execute()
            .unwrap();
        assert_eq!(result.results[0].chunk_external_id, expected);
        assert_eq!(
            result.results[0].explain.get("tokenizer"),
            Some(&TokenizerConfig::default().to_json())
        );
    }
}

#[test]
fn independently_generated_legacy_file_rebuilds_without_losing_data() {
    let temp = Temp::new();
    let path = temp.0.join("legacy.coffer");
    let hex = include_str!("fixtures/legacy-2d9eb64.hex")
        .split_whitespace()
        .collect::<String>();
    let bytes = hex
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
        .collect::<Vec<_>>();
    fs::write(&path, &bytes).unwrap();
    let reader = Database::open_read_only(&path).unwrap();
    assert_eq!(
        reader.list_collections().unwrap()[0].tokenizer,
        TokenizerConfig::default()
    );
    let collection = reader.collection("code").unwrap();
    assert_eq!(
        collection
            .get_chunk_by_external_id("storage#cache")
            .unwrap()
            .text,
        "struct PageCache { page_cache: Vec<u8> } // WalRecovery reads committed pages"
    );
    assert_eq!(
        collection
            .search()
            .text("page cache")
            .execute()
            .unwrap()
            .results
            .len(),
        1
    );
    assert_eq!(fs::read(&path).unwrap(), bytes);
    drop(collection);
    drop(reader);
    let db = Database::open(&path).unwrap();
    assert_eq!(
        db.drop_collection("code").unwrap_err().code(),
        ErrorCode::UnsupportedVersion
    );
    assert_eq!(fs::read(&path).unwrap(), bytes);
    db.compact(CompactOptions {
        rebuild_indexes: true,
    })
    .unwrap();
    let migrated = fs::read(&path).unwrap();
    assert_eq!(&migrated[..14], b"COFFERDB\x00\x00\x01\x00\x01\x01");
    assert_ne!(migrated, bytes);
    drop(db);
    let db = Database::open_read_only(&path).unwrap();
    assert_eq!(
        db.collection("code")
            .unwrap()
            .search()
            .text("recovery")
            .execute()
            .unwrap()
            .results
            .len(),
        1
    );
}
