use std::collections::BTreeSet;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
use std::time::Instant;

use cofferdb::{
    parse_json, Database, DatabaseOpenOptions, Error, ErrorCode, JsonValue, QueryEmbeddingInput,
    ResourceProfile, Result, SearchResults,
};

struct Trial {
    timing_scope: TimingScope,
    query_id: String,
    text: Option<String>,
    embedding: Option<QueryEmbeddingInput>,
    filter: JsonValue,
    profile: ResourceProfile,
    top_k: usize,
    warmup_queries: usize,
    expected_chunk_ids: Vec<String>,
    expected_index_source: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TimingScope {
    Execute,
    StandardJson,
}

impl TimingScope {
    fn as_str(self) -> &'static str {
        match self {
            Self::Execute => "rust_search_execute",
            Self::StandardJson => "rust_search_standard_json",
        }
    }

    fn render(self, result: &SearchResults) -> Option<(JsonValue, String)> {
        match self {
            Self::Execute => None,
            Self::StandardJson => {
                let value = result.to_json();
                let wire = value.to_pretty_string();
                std::hint::black_box(&wire);
                #[cfg(test)]
                tests::timing_event("serialized");
                Some((value, wire))
            }
        }
    }
}

fn invalid(field: &'static str) -> Error {
    Error::new(
        ErrorCode::InvalidJsonlSchema,
        "invalid search trial configuration",
    )
    .with_detail("field", field)
}

impl Trial {
    fn parse(value: &JsonValue) -> Result<Self> {
        let fields = value.as_object().ok_or_else(|| invalid("root"))?;
        let allowed = [
            "schema_version",
            "query_id",
            "text",
            "embedding",
            "embedding_model",
            "filter",
            "profile",
            "top_k",
            "warmup_queries",
            "expected_chunk_ids",
            "expected_index_source",
        ];
        if fields.keys().any(|key| !allowed.contains(&key.as_str())) {
            return Err(invalid("root"));
        }
        let string = |field| {
            value
                .get(field)
                .and_then(JsonValue::as_str)
                .filter(|value| !value.trim().is_empty())
                .ok_or_else(|| invalid(field))
        };
        let integer = |field, min: usize, max: usize| {
            value
                .get(field)
                .and_then(JsonValue::as_f64)
                .filter(|n| {
                    n.is_finite() && n.fract() == 0.0 && *n >= min as f64 && *n <= max as f64
                })
                .map(|n| n as usize)
                .ok_or_else(|| invalid(field))
        };
        let timing_scope = match string("schema_version")? {
            "cofferdb-rust-search-trial-1" => TimingScope::Execute,
            "cofferdb-rust-search-trial-2" => TimingScope::StandardJson,
            _ => return Err(invalid("schema_version")),
        };
        let text = fields
            .get("text")
            .map(|_| string("text").map(str::to_owned))
            .transpose()?;
        let embedding = fields
            .get("embedding")
            .map(|value| -> Result<_> {
                let values = value.as_array().ok_or_else(|| invalid("embedding"))?;
                let vector = values
                    .iter()
                    .map(|value| {
                        value
                            .as_f64()
                            .map(|n| n as f32)
                            .filter(|n| n.is_finite())
                            .ok_or_else(|| invalid("embedding"))
                    })
                    .collect::<Result<Vec<_>>>()?;
                if vector.is_empty() {
                    return Err(invalid("embedding"));
                }
                let mut embedding = QueryEmbeddingInput::from_vector(vector);
                embedding.model_name = fields
                    .get("embedding_model")
                    .map(|_| string("embedding_model").map(str::to_owned))
                    .transpose()?;
                Ok(embedding)
            })
            .transpose()?;
        if (text.is_none() && embedding.is_none())
            || (embedding.is_none() && fields.contains_key("embedding_model"))
        {
            return Err(invalid("embedding"));
        }
        let filter = fields
            .get("filter")
            .cloned()
            .unwrap_or_else(JsonValue::object);
        if filter.as_object().is_none() {
            return Err(invalid("filter"));
        }
        let profile =
            ResourceProfile::parse(string("profile")?).ok_or_else(|| invalid("profile"))?;
        let top_k = integer("top_k", 1, 100_000)?;
        let expected_chunk_ids = value
            .get("expected_chunk_ids")
            .and_then(JsonValue::as_array)
            .ok_or_else(|| invalid("expected_chunk_ids"))?
            .iter()
            .map(|id| {
                id.as_str()
                    .filter(|id| !id.is_empty())
                    .map(str::to_owned)
                    .ok_or_else(|| invalid("expected_chunk_ids"))
            })
            .collect::<Result<Vec<_>>>()?;
        if expected_chunk_ids.len() > top_k
            || expected_chunk_ids.iter().collect::<BTreeSet<_>>().len() != expected_chunk_ids.len()
        {
            return Err(invalid("expected_chunk_ids"));
        }
        let expected_index_source = string("expected_index_source")?;
        if !["persistent_checkpoint", "transient_rebuild"].contains(&expected_index_source) {
            return Err(invalid("expected_index_source"));
        }
        Ok(Self {
            timing_scope,
            query_id: string("query_id")?.into(),
            text,
            embedding,
            filter,
            profile,
            top_k,
            warmup_queries: integer("warmup_queries", 0, 1000)?,
            expected_chunk_ids,
            expected_index_source: expected_index_source.into(),
        })
    }

    fn verify(&self, result: &SearchResults) -> Result<()> {
        if result.query_context.index_source != self.expected_index_source
            || !result
                .results
                .iter()
                .map(|row| row.chunk_external_id.as_str())
                .eq(self.expected_chunk_ids.iter().map(String::as_str))
        {
            return Err(Error::new(
                ErrorCode::InvalidJsonlSchema,
                "search trial result differs from the frozen expectation",
            ));
        }
        Ok(())
    }

    fn run(&self, database: &Path, collection: &str) -> Result<JsonValue> {
        let db = Database::open_with_options(
            database,
            DatabaseOpenOptions {
                read_only: true,
                resource_profile: Some(self.profile),
                ..DatabaseOpenOptions::default()
            },
        )?;
        let mut query = db
            .collection(collection)?
            .search()
            .filter(self.filter.clone())
            .resource_profile(self.profile)
            .top_k(self.top_k);
        if let Some(text) = &self.text {
            query = query.text(text);
        }
        if let Some(embedding) = &self.embedding {
            query = query.query_embedding(embedding.clone());
        }
        for _ in 0..self.warmup_queries {
            let results = query.clone().execute()?;
            self.verify(&results)?;
            drop(self.timing_scope.render(&results));
        }
        // The protocol fixes whether host DTO copying and pretty JSON are included.
        // Open, input construction, warmup, oracle and stdout IO remain outside.
        let started = Instant::now();
        #[cfg(test)]
        tests::timing_event("started");
        let results = query.execute()?;
        let rendered = self.timing_scope.render(&results);
        let duration_ms = started.elapsed().as_secs_f64() * 1000.0;
        #[cfg(test)]
        tests::timing_event("sampled");
        self.verify(&results)?;
        let search = match rendered {
            Some((value, _wire)) => value,
            None => results.to_json(),
        };
        Ok(object([
            ("schema_version", JsonValue::String("0.1".into())),
            ("command", JsonValue::String("rust-search-trial".into())),
            ("ok", JsonValue::Bool(true)),
            ("duration_ms", JsonValue::Number(duration_ms)),
            ("warnings", JsonValue::Array(Vec::new())),
            (
                "result",
                object([
                    (
                        "timing_scope",
                        JsonValue::String(self.timing_scope.as_str().into()),
                    ),
                    ("query_id", JsonValue::String(self.query_id.clone())),
                    (
                        "warmup_queries",
                        JsonValue::Number(self.warmup_queries as f64),
                    ),
                    ("search", search),
                ]),
            ),
        ]))
    }
}

fn object<const N: usize>(fields: [(&str, JsonValue); N]) -> JsonValue {
    JsonValue::Object(
        fields
            .into_iter()
            .map(|(key, value)| (key.to_owned(), value))
            .collect(),
    )
}

fn load_trial(path: &Path) -> Result<Trial> {
    let file = File::open(path)?;
    if !file.metadata()?.is_file() {
        return Err(invalid("query_config"));
    }
    let mut bytes = Vec::new();
    file.take(1_048_577).read_to_end(&mut bytes)?;
    if bytes.len() > 1_048_576 {
        return Err(invalid("query_config_limit_1_MiB"));
    }
    let text = std::str::from_utf8(&bytes).map_err(|_| invalid("query_config_utf8"))?;
    Trial::parse(&parse_json(text)?)
}

fn run() -> Result<()> {
    let args = std::env::args_os().collect::<Vec<_>>();
    if args.len() != 4 {
        return Err(Error::new(
            ErrorCode::CliUsage,
            "Usage: search_trial <database> <collection> <query-config.json>",
        ));
    }
    let collection = args[2].to_str().ok_or_else(|| invalid("collection"))?;
    let trial = load_trial(Path::new(&args[3]))?;
    let result = trial.run(Path::new(&args[1]), collection)?;
    writeln!(std::io::stdout().lock(), "{}", result.to_pretty_string())?;
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!(
            "{}",
            cofferdb::cli::error_output(Some("rust-search-trial"), &error)
        );
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cofferdb::{
        ChunkInput, CollectionConfig, CompactOptions, DatabaseConfig, DistanceMetric,
        DocumentInput, EmbeddingInput, VectorIndexConfig,
    };
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    thread_local! {
        static TIMING_EVENTS: std::cell::RefCell<Vec<&'static str>> = const { std::cell::RefCell::new(Vec::new()) };
    }

    pub(super) fn timing_event(event: &'static str) {
        TIMING_EVENTS.with(|events| events.borrow_mut().push(event));
    }

    struct Fixture(PathBuf);

    impl Fixture {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "cofferdb-search-trial-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn config() -> JsonValue {
        parse_json(r#"{"schema_version":"cofferdb-rust-search-trial-1","query_id":"q","text":"recovery","profile":"balanced","top_k":1,"warmup_queries":0,"expected_chunk_ids":["c0"],"expected_index_source":"persistent_checkpoint"}"#).unwrap()
    }

    #[test]
    fn trial_profile_configuration_uses_exact_current_names() {
        for (name, expected) in [
            ("expanded", ResourceProfile::Expanded),
            ("balanced", ResourceProfile::Balanced),
            ("constrained", ResourceProfile::Constrained),
        ] {
            let mut value = config();
            value
                .as_object_mut()
                .unwrap()
                .insert("profile".into(), JsonValue::String(name.into()));
            assert_eq!(Trial::parse(&value).unwrap().profile, expected);
        }
        for name in [
            "quality-first",
            "quality_first",
            "resource-first",
            "resource_first",
            "performance",
            "eco",
            "Expanded",
            "Balanced",
            "Constrained",
            "help",
        ] {
            let mut value = config();
            value
                .as_object_mut()
                .unwrap()
                .insert("profile".into(), JsonValue::String(name.into()));
            let error = Trial::parse(&value).err().unwrap();
            assert_eq!(error.code(), ErrorCode::InvalidJsonlSchema);
            assert!(error.detail().contains(&("field".into(), "profile".into())));
        }
    }

    #[test]
    fn configuration_rejects_ambiguous_or_unbounded_work_before_open() {
        assert_eq!(
            Trial::parse(&config()).unwrap().timing_scope,
            TimingScope::Execute
        );
        let mut json_trial = config();
        json_trial.as_object_mut().unwrap().insert(
            "schema_version".into(),
            JsonValue::String("cofferdb-rust-search-trial-2".into()),
        );
        assert_eq!(
            Trial::parse(&json_trial).unwrap().timing_scope,
            TimingScope::StandardJson
        );
        json_trial.as_object_mut().unwrap().insert(
            "timing_scope".into(),
            JsonValue::String("rust_search_execute".into()),
        );
        assert!(Trial::parse(&json_trial).is_err());
        for field in [
            "schema_version",
            "query_id",
            "profile",
            "top_k",
            "warmup_queries",
            "expected_chunk_ids",
            "expected_index_source",
            "text",
        ] {
            let mut value = config();
            value.as_object_mut().unwrap().remove(field);
            assert!(Trial::parse(&value).is_err(), "missing {field}");
        }
        for (field, value) in [
            ("schema_version", r#""unknown""#),
            ("query_id", r#"" ""#),
            ("profile", r#""unknown""#),
            ("top_k", "0"),
            ("top_k", "100001"),
            ("warmup_queries", "-1"),
            ("warmup_queries", "1001"),
            ("filter", "[]"),
            ("embedding", "[]"),
            ("embedding", "[true]"),
            ("embedding", "[null]"),
            ("embedding_model", r#""m""#),
            ("expected_chunk_ids", r#"["c0","c0"]"#),
            ("expected_chunk_ids", "[0]"),
            ("expected_index_source", r#""unknown""#),
            ("private-field", "true"),
        ] {
            let mut config = config();
            config
                .as_object_mut()
                .unwrap()
                .insert(field.into(), parse_json(value).unwrap());
            assert!(Trial::parse(&config).is_err(), "{field}: {value}");
        }
        for seed in 0..100 {
            let mut value = config();
            value.as_object_mut().unwrap().insert(
                "warmup_queries".into(),
                JsonValue::Number(seed as f64 + 0.5),
            );
            assert!(Trial::parse(&value).is_err());
        }
        let mut value = config();
        value.as_object_mut().unwrap().insert(
            "embedding".into(),
            JsonValue::Array(vec![JsonValue::Number(f64::from(f32::MAX) * 2.0)]),
        );
        assert!(Trial::parse(&value).is_err());
    }

    #[test]
    fn trial_input_is_bounded_regular_utf8_json() {
        let fixture = Fixture::new();
        let path = fixture.0.join("query.json");
        for bytes in [vec![b' '; 1_048_577], vec![0xff], b"{invalid".to_vec()] {
            fs::write(&path, bytes).unwrap();
            assert!(load_trial(&path).is_err());
        }
        fs::write(&path, config().to_compact_string()).unwrap();
        assert!(load_trial(&path).is_ok());
        assert!(load_trial(&fixture.0).is_err());
    }

    #[test]
    fn readonly_trials_validate_warmup_and_measured_results_across_query_sources() {
        let fixture = Fixture::new();
        let path = fixture.0.join("data.coffer");
        let db = Database::create(&path, DatabaseConfig::default()).unwrap();
        let collection = db
            .create_collection(CollectionConfig {
                name: "notes".into(),
                embedding_dim: 2,
                tokenizer: Default::default(),
                distance_metric: DistanceMetric::Cosine,
                embedding_model: Some("m".into()),
                vectors_normalized: false,
                vector_index: VectorIndexConfig::Exact,
            })
            .unwrap();
        collection
            .insert_document(DocumentInput {
                external_id: "doc".into(),
                title: None,
                source_uri: None,
                metadata: JsonValue::object(),
                chunks: vec![ChunkInput {
                    chunk_external_id: "c0".into(),
                    text: "recovery log".into(),
                    ordinal: 0,
                    metadata: parse_json(r#"{"lang":"rust"}"#).unwrap(),
                    source_span: None,
                }],
            })
            .unwrap();
        collection
            .put_embeddings(vec![EmbeddingInput {
                chunk_external_id: "c0".into(),
                model_name: "m".into(),
                dim: 2,
                vector: vec![1.0, 0.0],
            }])
            .unwrap();
        drop(collection);
        db.close().unwrap();
        for seed in 0..100 {
            if seed == 50 {
                let db = Database::open(&path).unwrap();
                db.compact(CompactOptions::default()).unwrap();
                db.close().unwrap();
            }
            let before = fs::read(&path).unwrap();
            let mut value = config();
            let object = value.as_object_mut().unwrap();
            if seed % 4 >= 2 {
                object.insert(
                    "schema_version".into(),
                    JsonValue::String("cofferdb-rust-search-trial-2".into()),
                );
            }
            object.insert(
                "profile".into(),
                JsonValue::String(["expanded", "balanced", "constrained"][(seed / 3) % 3].into()),
            );
            object.insert(
                "warmup_queries".into(),
                JsonValue::Number((seed % 3) as f64),
            );
            if seed % 3 != 0 {
                object.insert("embedding".into(), parse_json("[1,0]").unwrap());
                object.insert("embedding_model".into(), JsonValue::String("m".into()));
            }
            if seed % 3 == 1 {
                object.remove("text");
            }
            if seed % 2 != 0 {
                object.insert(
                    "filter".into(),
                    parse_json(r#"{"chunk.lang":"absent"}"#).unwrap(),
                );
                object.insert("expected_chunk_ids".into(), JsonValue::Array(Vec::new()));
            }
            let source = if seed < 50 {
                "persistent_checkpoint"
            } else {
                "transient_rebuild"
            };
            object.insert(
                "expected_index_source".into(),
                JsonValue::String(source.into()),
            );
            let mut trial = Trial::parse(&value).unwrap();
            TIMING_EVENTS.with(|events| events.borrow_mut().clear());
            let report = trial.run(&path, "notes").unwrap();
            let phases = TIMING_EVENTS.with(|events| events.take());
            let mut expected = match trial.timing_scope {
                TimingScope::Execute => Vec::new(),
                TimingScope::StandardJson => vec!["serialized"; trial.warmup_queries],
            };
            expected.push("started");
            if trial.timing_scope == TimingScope::StandardJson {
                expected.push("serialized");
            }
            expected.push("sampled");
            assert_eq!(phases, expected);
            assert!(report.get("duration_ms").unwrap().as_f64().unwrap() >= 0.0);
            let result = report.get("result").unwrap();
            assert_eq!(
                result.get("timing_scope").and_then(JsonValue::as_str),
                Some(trial.timing_scope.as_str())
            );
            let search = result.get("search").unwrap();
            assert_eq!(parse_json(&search.to_pretty_string()).unwrap(), *search);
            assert_eq!(
                result
                    .get("search")
                    .unwrap()
                    .get("query_context")
                    .unwrap()
                    .get("index_source")
                    .and_then(JsonValue::as_str),
                Some(source)
            );
            trial.expected_chunk_ids = vec!["wrong".into()];
            assert!(trial.run(&path, "notes").is_err());
            assert_eq!(fs::read(&path).unwrap(), before);
        }
    }
}
