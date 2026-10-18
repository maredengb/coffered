use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::path::{Path, PathBuf};

use crate::error::{Error, ErrorCode, Result};
use crate::hash::{sha256_reader, Sha256Reader};
use crate::json::{
    parse_reader_in, required_string, required_usize, JsonBuilder, JsonInputKind, JsonLines,
    JsonObject, JsonOperationContext, JsonReservation, JsonSchema, JsonValue,
};
use crate::metadata::FilterExpr;
use crate::model::{json_usize, QueryEmbeddingInput};

mod query_mode;
use query_mode::BenchmarkQueryMode;
mod result;
pub(crate) use result::validate_benchmark_result_contract;
mod runner;
pub(crate) use runner::{run_benchmark, BenchmarkOptions};

const BENCHMARK_DOCUMENT_FIELDS: &[&str] = &["external_id", "title", "source_uri", "metadata"];
const BENCHMARK_CHUNK_FIELDS: &[&str] = &[
    "document_external_id",
    "chunk_external_id",
    "ordinal",
    "text",
    "metadata",
    "source_span",
];
const BENCHMARK_EMBEDDING_FIELDS: &[&str] = &["chunk_external_id", "model_name", "dim", "vector"];
const BENCHMARK_QUERY_FIELDS: &[&str] = &["query_id", "text", "filter", "top_k", "embedding"];
const BENCHMARK_QREL_FIELDS: &[&str] = &["query_id", "chunk_external_id", "relevance"];
const BENCHMARK_QUERY_EMBEDDING_FIELDS: &[&str] = &["query_id", "model_name", "dim", "vector"];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QueryEmbeddingSource {
    Sidecar,
    Inline,
    None,
}

#[derive(Clone, Debug)]
pub struct BenchmarkDatasetManifest {
    pub tokenizer: crate::TokenizerConfig,
    pub dataset_id: String,
    pub version: String,
    pub chunking: JsonValue,
    pub embedding_model: Option<String>,
    pub embedding_dim: usize,
    pub documents_sha256: String,
    pub chunks_sha256: String,
    pub embeddings_sha256: Option<String>,
    pub queries_sha256: String,
    pub qrels_sha256: String,
    pub query_embedding_source: QueryEmbeddingSource,
    pub query_embedding_model: Option<String>,
    pub query_embedding_dim: Option<usize>,
    pub query_embeddings_sha256: Option<String>,
    pub dataset_manifest_sha256: String,
}

#[derive(Clone, Debug)]
pub struct BenchmarkDataset {
    pub manifest: BenchmarkDatasetManifest,
    pub documents_path: PathBuf,
    pub chunks_path: PathBuf,
    pub embeddings_path: Option<PathBuf>,
    pub queries: Vec<BenchmarkQuery>,
    base_dir: PathBuf,
    manifest_path_label: String,
}

#[derive(Clone, Debug)]
pub struct BenchmarkQuery {
    pub query_id: String,
    pub text: Option<String>,
    pub filter: JsonValue,
    pub top_k: usize,
    pub embedding: Option<QueryEmbeddingInput>,
    pub relevant_chunk_ids: BTreeSet<String>,
}

struct BenchmarkManifestInput {
    manifest_path: PathBuf,
    manifest_path_label: String,
    base_dir: PathBuf,
    manifest_sha256: String,
    obj: JsonObject,
    query_embedding_source: QueryEmbeddingSource,
    context: JsonOperationContext,
    work: JsonReservation,
}

#[derive(Clone, Debug)]
struct Artifact {
    raw_path: String,
    path: PathBuf,
    kind: &'static str,
    hash_field: &'static str,
}

#[derive(Clone, Debug)]
struct QueryRow {
    query_id: String,
    line: usize,
    text: Option<String>,
    filter: JsonValue,
    top_k: usize,
    embedding: Option<QueryEmbeddingInput>,
    has_embedding: bool,
}

#[derive(Clone, Debug)]
struct SidecarRow {
    line: usize,
}

impl QueryEmbeddingSource {
    pub fn as_str(self) -> &'static str {
        match self {
            QueryEmbeddingSource::Sidecar => "sidecar",
            QueryEmbeddingSource::Inline => "inline",
            QueryEmbeddingSource::None => "none",
        }
    }
}

impl BenchmarkDatasetManifest {
    #[cfg(test)]
    pub fn load_and_validate(path: impl AsRef<Path>) -> Result<Self> {
        Ok(BenchmarkDataset::load_and_validate(path)?.manifest)
    }

    pub(crate) fn build_result_dataset_json(&self, builder: &JsonBuilder<'_>) -> Result<JsonValue> {
        builder.object([
            ("dataset_id", builder.string(&self.dataset_id)?),
            ("tokenizer", self.tokenizer.build_json(builder)?),
            ("version", builder.string(&self.version)?),
            ("chunking", builder.copy_validated_value(&self.chunking)?),
            (
                "embedding_model",
                builder.optional_str(self.embedding_model.as_deref())?,
            ),
            (
                "embedding_dim",
                JsonValue::Number(self.embedding_dim as f64),
            ),
            (
                "dataset_manifest_sha256",
                builder.string(&self.dataset_manifest_sha256)?,
            ),
            ("documents_sha256", builder.string(&self.documents_sha256)?),
            ("chunks_sha256", builder.string(&self.chunks_sha256)?),
            (
                "embeddings_sha256",
                builder.optional_str(self.embeddings_sha256.as_deref())?,
            ),
            ("queries_sha256", builder.string(&self.queries_sha256)?),
            ("qrels_sha256", builder.string(&self.qrels_sha256)?),
            (
                "query_embedding_source",
                builder.string(self.query_embedding_source.as_str())?,
            ),
            (
                "query_embedding_model",
                builder.optional_str(self.query_embedding_model.as_deref())?,
            ),
            (
                "query_embedding_dim",
                self.query_embedding_dim
                    .map(|dim| JsonValue::Number(dim as f64))
                    .unwrap_or(JsonValue::Null),
            ),
            (
                "query_embeddings_sha256",
                builder.optional_str(self.query_embeddings_sha256.as_deref())?,
            ),
        ])
    }
}

impl BenchmarkDataset {
    fn locate_input_error(&self, error: Error) -> Error {
        let mut error = error.map_path_details(|reported| {
            if reported == self.manifest_path_label {
                return Some(reported.to_owned());
            }
            for path in [
                Some(&self.documents_path),
                Some(&self.chunks_path),
                self.embeddings_path.as_ref(),
            ]
            .into_iter()
            .flatten()
            {
                if Path::new(reported) == path
                    || fs::canonicalize(path).ok().as_deref() == Some(Path::new(reported))
                {
                    return Some(
                        path.strip_prefix(&self.base_dir)
                            .unwrap_or(path)
                            .display()
                            .to_string(),
                    );
                }
            }
            // Generated manifests and private staging paths have no public locator.
            None
        });
        if !error.detail().iter().any(|(key, _)| key == "manifest_path") {
            error = error.with_detail("manifest_path", &self.manifest_path_label);
        }
        error
    }

    #[cfg(test)]
    pub fn load_and_validate(path: impl AsRef<Path>) -> Result<Self> {
        Self::load_and_validate_in(path, &JsonOperationContext::standalone())
            .map(|(dataset, _)| dataset)
    }

    #[cfg(test)]
    pub(crate) fn load_and_validate_in(
        path: impl AsRef<Path>,
        context: &JsonOperationContext,
    ) -> Result<(Self, JsonReservation)> {
        Self::load_with_sources_in(path, context).map(|(dataset, work, _)| (dataset, work))
    }

    pub(crate) fn load_with_sources_in(
        path: impl AsRef<Path>,
        context: &JsonOperationContext,
    ) -> Result<(Self, JsonReservation, Vec<PathBuf>)> {
        BenchmarkManifestInput::load(path, context)?.validate_dataset()
    }
}

impl BenchmarkManifestInput {
    fn load(path: impl AsRef<Path>, context: &JsonOperationContext) -> Result<Self> {
        let requested_path = path.as_ref();
        let manifest_path = if requested_path.is_dir() {
            requested_path.join("dataset_manifest.json")
        } else {
            requested_path.to_path_buf()
        };
        let manifest_path_label = requested_path.display().to_string();
        let mut source = Sha256Reader::new(File::open(&manifest_path)?);
        let (mut value, work) = parse_reader_in(
            &mut source,
            JsonSchema::Value(JsonInputKind::Manifest),
            context,
        )?
        .into_parts();
        let manifest_sha256 = source.finish();
        let obj = value.as_object_mut().map(std::mem::take).ok_or_else(|| {
            Error::new(
                ErrorCode::InvalidJsonlSchema,
                "dataset manifest must be a JSON object",
            )
            .with_detail("manifest_path", &manifest_path_label)
        })?;
        reject_unknown_benchmark_manifest_fields(&obj, &manifest_path_label)?;
        if let Some(hashes) = obj.get("hashes").filter(|value| !value.is_null()) {
            let hashes = hashes.as_object().ok_or_else(|| {
                Error::new(ErrorCode::InvalidJsonlSchema, "hashes must be an object")
                    .with_detail("field", "hashes")
                    .with_detail("manifest_path", &manifest_path_label)
            })?;
            reject_unknown_benchmark_hash_fields(hashes, &manifest_path_label)?;
            validate_benchmark_hash_field_values(hashes, &manifest_path_label)?;
        }
        let source = obj
            .get("query_embedding_source")
            .and_then(JsonValue::as_str)
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::MissingRequiredField,
                    "missing string field query_embedding_source",
                )
                .with_detail("field", "query_embedding_source")
                .with_detail("manifest_path", &manifest_path_label)
            })?;
        let query_embedding_source = match source {
            "sidecar" => QueryEmbeddingSource::Sidecar,
            "inline" => QueryEmbeddingSource::Inline,
            "none" => QueryEmbeddingSource::None,
            other => {
                return Err(Error::new(
                    ErrorCode::InvalidJsonlSchema,
                    "query_embedding_source must be sidecar, inline, or none",
                )
                .with_detail("field", "query_embedding_source")
                .with_detail("query_embedding_source", other)
                .with_detail("manifest_path", &manifest_path_label));
            }
        };
        Ok(Self {
            base_dir: manifest_path
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .to_path_buf(),
            manifest_path,
            manifest_path_label,
            manifest_sha256,
            obj,
            query_embedding_source,
            context: context.clone(),
            work,
        })
    }

    fn validate_dataset(mut self) -> Result<(BenchmarkDataset, JsonReservation, Vec<PathBuf>)> {
        let schema_version = required_string(&self.obj, "schema_version")?;
        if schema_version != "0.1" {
            return Err(Error::new(
                ErrorCode::UnsupportedVersion,
                format!("unsupported dataset manifest schema_version {schema_version}"),
            )
            .with_detail("manifest_path", &self.manifest_path_label));
        }
        let dataset_id = required_string(&self.obj, "dataset_id")?;
        self.require_nonempty_manifest_string(&dataset_id, "dataset_id")?;
        let version = required_string(&self.obj, "version")?;
        self.require_nonempty_manifest_string(&version, "version")?;
        self.validate_chunking()?;
        let hashes = self.hashes()?;
        let documents = self.required_artifact("documents", "documents", "documents_sha256")?;
        let chunks = self.required_artifact("chunks", "chunks", "chunks_sha256")?;
        let queries = self.required_artifact("queries", "queries", "queries_sha256")?;
        let qrels = self.required_artifact("qrels", "qrels", "qrels_sha256")?;
        let embeddings = self.optional_artifact("embeddings", "embeddings", "embeddings_sha256")?;
        let embeddings_hash = optional_string_field(hashes, "embeddings_sha256");
        let query_embeddings = self.optional_artifact(
            "query_embeddings",
            "query_embeddings",
            "query_embeddings_sha256",
        )?;
        let query_embedding_hash = optional_string_field(hashes, "query_embeddings_sha256");
        let embedding_model = optional_string_field(&self.obj, "embedding_model");
        let query_embedding_model = optional_string_field(&self.obj, "query_embedding_model");
        self.reject_empty_optional_manifest_string(embedding_model.as_deref(), "embedding_model")?;
        self.reject_empty_optional_manifest_string(
            query_embedding_model.as_deref(),
            "query_embedding_model",
        )?;
        let query_embedding_dim = self
            .obj
            .get("query_embedding_dim")
            .filter(|value| !value.is_null())
            .map(|value| {
                let dim = json_usize(value).ok_or_else(|| {
                    Error::new(
                        ErrorCode::InvalidJsonlSchema,
                        "query_embedding_dim must be a positive integer",
                    )
                    .with_detail("field", "query_embedding_dim")
                    .with_detail("manifest_path", &self.manifest_path_label)
                })?;
                if dim > 0 {
                    Ok(dim)
                } else {
                    Err(Error::new(
                        ErrorCode::InvalidVectorLength,
                        "query_embedding_dim must be greater than zero",
                    )
                    .with_detail("field", "query_embedding_dim")
                    .with_detail("manifest_path", &self.manifest_path_label))
                }
            })
            .transpose()?;
        self.validate_source_contract(
            query_embeddings.as_ref(),
            query_embedding_hash.as_ref(),
            query_embedding_model.as_ref(),
            query_embedding_dim,
            embeddings.as_ref(),
            embeddings_hash.as_ref(),
        )?;
        let embedding_dim = self.infer_embedding_dim(embeddings.as_ref(), query_embedding_dim)?;

        let documents_sha256 = self.validate_hash(&documents)?;
        let chunks_sha256 = self.validate_hash(&chunks)?;
        let queries_sha256 = self.validate_hash(&queries)?;
        let qrels_sha256 = self.validate_hash(&qrels)?;
        let embeddings_sha256 = embeddings
            .as_ref()
            .filter(|_| embeddings_hash.is_some())
            .map(|artifact| self.validate_hash(artifact))
            .transpose()?;
        let query_embeddings_sha256 = query_embeddings
            .as_ref()
            .map(|artifact| self.validate_hash(artifact))
            .transpose()?;

        let mut retained = self.context.reservation();
        let query_rows = self.read_queries(&queries, query_embedding_dim, &mut retained)?;
        let document_external_ids = self.read_document_external_ids(&documents)?;
        let chunk_external_ids = self.read_chunk_external_ids(&chunks, &document_external_ids)?;
        self.validate_embeddings(
            embeddings.as_ref(),
            embedding_model.as_deref(),
            embedding_dim,
            &chunk_external_ids,
        )?;
        let mut qrels_rows = self.validate_qrels(&qrels, &query_rows, &chunk_external_ids)?;
        let mut sidecar_embeddings = match self.query_embedding_source {
            QueryEmbeddingSource::Sidecar => {
                let query_embeddings = query_embeddings
                    .as_ref()
                    .expect("sidecar contract validates query_embeddings is present");
                self.validate_sidecar(
                    query_embeddings,
                    &query_rows,
                    query_embedding_model.as_deref(),
                    query_embedding_dim,
                    &mut retained,
                )?
            }
            QueryEmbeddingSource::Inline => {
                self.validate_inline_queries(&query_rows)?;
                BTreeMap::new()
            }
            QueryEmbeddingSource::None => {
                self.validate_text_only_queries(&query_rows)?;
                BTreeMap::new()
            }
        };

        let benchmark_queries = query_rows
            .into_iter()
            .map(|row| {
                let embedding = match self.query_embedding_source {
                    QueryEmbeddingSource::Sidecar => sidecar_embeddings.remove(&row.query_id),
                    QueryEmbeddingSource::Inline => row.embedding,
                    QueryEmbeddingSource::None => None,
                };
                let relevant_chunk_ids = qrels_rows.remove(&row.query_id).unwrap_or_default();
                BenchmarkQuery {
                    query_id: row.query_id,
                    text: row.text,
                    filter: row.filter,
                    top_k: row.top_k,
                    embedding,
                    relevant_chunk_ids,
                }
            })
            .collect::<Vec<_>>();

        let tokenizer = self
            .obj
            .get("tokenizer")
            .map(crate::TokenizerConfig::from_json)
            .transpose()?
            .unwrap_or_default();
        let chunking = self.obj.remove("chunking").unwrap_or(JsonValue::Null);
        drop(self.obj);
        chunking.reconcile_retained_in(JsonInputKind::Manifest, &mut self.work)?;
        retained.merge(self.work);
        let input_paths = [
            self.manifest_path,
            documents.path.clone(),
            chunks.path.clone(),
            queries.path,
            qrels.path,
        ]
        .into_iter()
        .chain(embeddings.as_ref().map(|artifact| artifact.path.clone()))
        .chain(query_embeddings.map(|artifact| artifact.path))
        .collect();
        Ok((
            BenchmarkDataset {
                manifest: BenchmarkDatasetManifest {
                    tokenizer,
                    dataset_id,
                    version,
                    chunking,
                    embedding_model,
                    embedding_dim,
                    documents_sha256,
                    chunks_sha256,
                    embeddings_sha256,
                    queries_sha256,
                    qrels_sha256,
                    query_embedding_source: self.query_embedding_source,
                    query_embedding_model,
                    query_embedding_dim,
                    query_embeddings_sha256,
                    dataset_manifest_sha256: self.manifest_sha256,
                },
                documents_path: documents.path,
                chunks_path: chunks.path,
                embeddings_path: embeddings.map(|artifact| artifact.path),
                queries: benchmark_queries,
                base_dir: self.base_dir,
                manifest_path_label: self.manifest_path_label,
            },
            retained,
            input_paths,
        ))
    }

    fn hashes(&self) -> Result<&JsonObject> {
        self.obj
            .get("hashes")
            .and_then(JsonValue::as_object)
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::MissingRequiredField,
                    "missing object field hashes",
                )
                .with_detail("field", "hashes")
                .with_detail("manifest_path", &self.manifest_path_label)
            })
    }

    fn validate_chunking(&self) -> Result<()> {
        let Some(value) = self.obj.get("chunking").filter(|value| !value.is_null()) else {
            return Ok(());
        };
        let obj = value.as_object().ok_or_else(|| {
            Error::new(
                ErrorCode::InvalidJsonlSchema,
                "benchmark manifest chunking must be a JSON object",
            )
            .with_detail("field", "chunking")
            .with_detail("manifest_path", &self.manifest_path_label)
        })?;
        let chunk_size = obj.get("chunk_size").and_then(json_usize).ok_or_else(|| {
            Error::new(
                ErrorCode::MissingRequiredField,
                "benchmark manifest chunking requires positive integer chunk_size",
            )
            .with_detail("field", "chunking.chunk_size")
            .with_detail("manifest_path", &self.manifest_path_label)
        })?;
        if chunk_size == 0 {
            return Err(Error::new(
                ErrorCode::InvalidJsonlSchema,
                "benchmark manifest chunking chunk_size must be greater than zero",
            )
            .with_detail("field", "chunking.chunk_size")
            .with_detail("manifest_path", &self.manifest_path_label));
        }
        let chunk_overlap = obj
            .get("chunk_overlap")
            .and_then(json_usize)
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::MissingRequiredField,
                    "benchmark manifest chunking requires integer chunk_overlap",
                )
                .with_detail("field", "chunking.chunk_overlap")
                .with_detail("manifest_path", &self.manifest_path_label)
            })?;
        if chunk_overlap >= chunk_size {
            return Err(Error::new(
                ErrorCode::InvalidJsonlSchema,
                "benchmark manifest chunk_overlap must be smaller than chunk_size",
            )
            .with_detail("field", "chunking.chunk_overlap")
            .with_detail("manifest_path", &self.manifest_path_label));
        }
        Ok(())
    }

    fn require_nonempty_manifest_string(&self, value: &str, field: &'static str) -> Result<()> {
        if value.trim().is_empty() {
            return Err(Error::new(
                ErrorCode::MissingRequiredField,
                format!("benchmark manifest field {field} must not be empty"),
            )
            .with_detail("field", field)
            .with_detail("manifest_path", &self.manifest_path_label));
        }
        Ok(())
    }

    fn reject_empty_optional_manifest_string(
        &self,
        value: Option<&str>,
        field: &'static str,
    ) -> Result<()> {
        if matches!(value, Some(value) if value.trim().is_empty()) {
            return Err(Error::new(
                ErrorCode::MissingRequiredField,
                format!("benchmark manifest field {field} must not be empty when present"),
            )
            .with_detail("field", field)
            .with_detail("manifest_path", &self.manifest_path_label));
        }
        Ok(())
    }

    fn required_artifact(
        &self,
        path_field: &'static str,
        kind: &'static str,
        hash_field: &'static str,
    ) -> Result<Artifact> {
        let raw_path = required_string(&self.obj, path_field).map_err(|err| {
            Error::new(err.code(), err.message())
                .with_detail("field", path_field)
                .with_detail("artifact_kind", kind)
                .with_detail("manifest_path", &self.manifest_path_label)
        })?;
        if optional_string_field(self.hashes()?, hash_field).is_none() {
            return Err(Error::new(
                ErrorCode::MissingRequiredField,
                format!("missing string field hashes.{hash_field}"),
            )
            .with_detail("field", format!("hashes.{hash_field}"))
            .with_detail("path", &raw_path)
            .with_detail("artifact_kind", kind)
            .with_detail("manifest_path", &self.manifest_path_label));
        }
        Ok(self.artifact(raw_path, kind, hash_field))
    }

    fn optional_artifact(
        &self,
        path_field: &'static str,
        kind: &'static str,
        hash_field: &'static str,
    ) -> Result<Option<Artifact>> {
        Ok(optional_string_field(&self.obj, path_field)
            .map(|raw_path| self.artifact(raw_path, kind, hash_field)))
    }

    fn artifact(&self, raw_path: String, kind: &'static str, hash_field: &'static str) -> Artifact {
        Artifact {
            path: self.base_dir.join(&raw_path),
            raw_path,
            kind,
            hash_field,
        }
    }

    fn validate_source_contract(
        &self,
        query_embeddings: Option<&Artifact>,
        query_embedding_hash: Option<&String>,
        query_embedding_model: Option<&String>,
        query_embedding_dim: Option<usize>,
        embeddings: Option<&Artifact>,
        embeddings_hash: Option<&String>,
    ) -> Result<()> {
        let embedding_model_present = self
            .obj
            .get("embedding_model")
            .and_then(JsonValue::as_str)
            .is_some();
        match self.query_embedding_source {
            QueryEmbeddingSource::Sidecar => {
                if query_embeddings.is_none() {
                    return Err(self.missing_source_field("query_embeddings"));
                }
                if query_embedding_hash.is_none() {
                    return Err(self.missing_source_field("hashes.query_embeddings_sha256"));
                }
                if query_embedding_model.is_none() {
                    return Err(self.missing_source_field("query_embedding_model"));
                }
                if query_embedding_dim.is_none() {
                    return Err(self.missing_source_field("query_embedding_dim"));
                }
                if embeddings.is_none() {
                    return Err(self.missing_source_field("embeddings"));
                }
                if embeddings_hash.is_none() {
                    return Err(self.missing_source_field("hashes.embeddings_sha256"));
                }
                if !embedding_model_present {
                    return Err(self.missing_benchmark_field("embedding_model"));
                }
            }
            QueryEmbeddingSource::Inline => {
                if query_embeddings.is_some() || query_embedding_hash.is_some() {
                    return Err(self.mode_conflict(
                        "inline mode must not declare query_embeddings or hashes.query_embeddings_sha256",
                    ));
                }
                if query_embedding_model.is_none() {
                    return Err(self.missing_source_field("query_embedding_model"));
                }
                if query_embedding_dim.is_none() {
                    return Err(self.missing_source_field("query_embedding_dim"));
                }
                if embeddings.is_none() {
                    return Err(self.missing_source_field("embeddings"));
                }
                if embeddings_hash.is_none() {
                    return Err(self.missing_source_field("hashes.embeddings_sha256"));
                }
                if !embedding_model_present {
                    return Err(self.missing_benchmark_field("embedding_model"));
                }
            }
            QueryEmbeddingSource::None => {
                if query_embeddings.is_some()
                    || query_embedding_hash.is_some()
                    || query_embedding_model.is_some()
                    || query_embedding_dim.is_some()
                    || embeddings.is_some()
                    || embeddings_hash.is_some()
                {
                    return Err(self.mode_conflict(
                        "none mode must not declare embedding artifacts or query embedding fields",
                    ));
                }
            }
        }
        Ok(())
    }

    fn validate_hash(&self, artifact: &Artifact) -> Result<String> {
        let expected = optional_string_field(self.hashes()?, artifact.hash_field)
            .expect("required hash presence was validated before hashing");
        let actual = sha256_reader(File::open(&artifact.path)?)?;
        if actual != expected {
            return Err(artifact_hash_mismatch(
                artifact,
                &self.manifest_path_label,
                &expected,
                &actual,
            ));
        }
        Ok(actual)
    }

    fn infer_embedding_dim(
        &self,
        embeddings: Option<&Artifact>,
        query_embedding_dim: Option<usize>,
    ) -> Result<usize> {
        let Some(embeddings) = embeddings else {
            return Ok(query_embedding_dim.unwrap_or(1));
        };
        let inferred = self
            .read_artifact_objects(embeddings)?
            .next()
            .transpose()?
            .map(
                |ArtifactObject {
                     line, obj, _work, ..
                 }| {
                    required_usize(&obj, "dim").map_err(|err| {
                        Error::new(err.code(), err.message())
                            .with_detail("field", "dim")
                            .with_detail("path", &embeddings.raw_path)
                            .with_detail("artifact_kind", embeddings.kind)
                            .with_detail("line", line)
                            .with_detail("manifest_path", &self.manifest_path_label)
                    })
                },
            )
            .transpose()?;
        let inferred = inferred.or(query_embedding_dim).unwrap_or(1);
        if let Some(query_dim) = query_embedding_dim {
            if inferred != query_dim {
                return Err(Error::new(
                    ErrorCode::InvalidEmbeddingDimension,
                    "embedding dimension does not match query_embedding_dim",
                )
                .with_detail("embedding_dim", inferred)
                .with_detail("query_embedding_dim", query_dim)
                .with_detail("manifest_path", &self.manifest_path_label));
            }
        }
        Ok(inferred)
    }

    fn read_queries(
        &self,
        artifact: &Artifact,
        query_embedding_dim: Option<usize>,
        retained: &mut JsonReservation,
    ) -> Result<Vec<QueryRow>> {
        let mut rows = Vec::new();
        let mut seen_query_ids = BTreeSet::new();
        for row in self.read_artifact_objects(artifact)? {
            let ArtifactObject {
                line,
                mut obj,
                _work: mut work,
                search_top_k,
            } = row?;
            let query_id =
                crate::json::take_required_string(&mut obj, "query_id").map_err(|err| {
                    Error::new(err.code(), err.message())
                        .with_detail("field", "query_id")
                        .with_detail("path", &artifact.raw_path)
                        .with_detail("artifact_kind", artifact.kind)
                        .with_detail("line", line)
                        .with_detail("manifest_path", &self.manifest_path_label)
                })?;
            self.require_nonempty_artifact_string(&query_id, "queries.query_id", artifact, line)?;
            if !seen_query_ids.insert(query_id.clone()) {
                return Err(Error::new(
                    ErrorCode::InvalidJsonlSchema,
                    "duplicate query_id in benchmark queries",
                )
                .with_detail("query_id", &query_id)
                .with_detail("line", line)
                .with_detail("manifest_path", &self.manifest_path_label));
            }
            let text = crate::json::take_required_string(&mut obj, "text").map_err(|err| {
                Error::new(err.code(), err.message())
                    .with_detail("field", "queries.text")
                    .with_detail("query_id", &query_id)
                    .with_detail("path", &artifact.raw_path)
                    .with_detail("artifact_kind", artifact.kind)
                    .with_detail("line", line)
                    .with_detail("manifest_path", &self.manifest_path_label)
            })?;
            if text.trim().is_empty() {
                return Err(Error::new(
                    ErrorCode::MissingRequiredField,
                    "benchmark query requires non-empty text",
                )
                .with_detail("field", "queries.text")
                .with_detail("query_id", &query_id)
                .with_detail("path", &artifact.raw_path)
                .with_detail("artifact_kind", artifact.kind)
                .with_detail("line", line)
                .with_detail("manifest_path", &self.manifest_path_label));
            }
            let filter = obj
                .remove("filter")
                .map(|value| {
                    if matches!(value, JsonValue::Object(_)) {
                        Ok(value)
                    } else {
                        Err(Error::new(
                            ErrorCode::InvalidJsonlSchema,
                            "query filter must be a JSON object",
                        )
                        .with_detail("field", "queries.filter")
                        .with_detail("query_id", &query_id)
                        .with_detail("path", &artifact.raw_path)
                        .with_detail("artifact_kind", artifact.kind)
                        .with_detail("line", line)
                        .with_detail("manifest_path", &self.manifest_path_label))
                    }
                })
                .transpose()?
                .unwrap_or_else(JsonValue::object);
            FilterExpr::parse_validated_in(&filter, &self.context, JsonInputKind::Filter).map_err(
                |err| {
                    if err
                        .detail()
                        .iter()
                        .any(|(key, _)| key == "json_limit_reason")
                    {
                        return err
                            .with_detail("path", &artifact.raw_path)
                            .with_detail("artifact_kind", artifact.kind)
                            .with_detail("line", line)
                            .with_detail("manifest_path", &self.manifest_path_label);
                    }
                    Error::new(
                        ErrorCode::InvalidJsonlSchema,
                        "query filter must be a valid metadata filter",
                    )
                    .with_detail("field", "queries.filter")
                    .with_detail("query_id", &query_id)
                    .with_detail("filter_error_code", err.code().as_str())
                    .with_detail("filter_error", err.message())
                    .with_detail("path", &artifact.raw_path)
                    .with_detail("artifact_kind", artifact.kind)
                    .with_detail("line", line)
                    .with_detail("manifest_path", &self.manifest_path_label)
                },
            )?;
            let top_k_value = obj
                .get("top_k")
                .filter(|value| !value.is_null())
                .ok_or_else(|| {
                    Error::new(
                        ErrorCode::MissingRequiredField,
                        "benchmark query requires top_k",
                    )
                    .with_detail("field", "queries.top_k")
                    .with_detail("query_id", &query_id)
                    .with_detail("path", &artifact.raw_path)
                    .with_detail("artifact_kind", artifact.kind)
                    .with_detail("line", line)
                    .with_detail("manifest_path", &self.manifest_path_label)
                })?;
            let top_k = matches!(top_k_value, JsonValue::Number(_))
                .then_some(search_top_k)
                .flatten()
                .ok_or_else(|| {
                    Error::new(
                        ErrorCode::InvalidJsonlSchema,
                        "query top_k must be a number with validated exact integer value",
                    )
                    .with_detail("field", "queries.top_k")
                    .with_detail("query_id", &query_id)
                    .with_detail("path", &artifact.raw_path)
                    .with_detail("artifact_kind", artifact.kind)
                    .with_detail("line", line)
                    .with_detail("manifest_path", &self.manifest_path_label)
                })?;
            if top_k == 0 || top_k > crate::query::MAX_SEARCH_TOP_K {
                return Err(Error::new(
                    ErrorCode::InvalidJsonlSchema,
                    "query top_k must be in the supported exact JSON range",
                )
                .with_detail("field", "queries.top_k")
                .with_detail("query_id", &query_id)
                .with_detail("path", &artifact.raw_path)
                .with_detail("artifact_kind", artifact.kind)
                .with_detail("line", line)
                .with_detail("manifest_path", &self.manifest_path_label));
            }
            let mut typed_work = self.context.reservation();
            let vector = obj
                .get("embedding")
                .map(|value| self.vector_values(value, &query_id, line, "queries", &mut typed_work))
                .transpose()
                .map_err(|err| self.locate_artifact_error(err, artifact, line))?;
            if let (Some(vector), Some(expected)) = (&vector, query_embedding_dim) {
                let actual = vector.len();
                if actual != expected {
                    return Err(self.locate_artifact_error(
                        invalid_vector_len(
                            &self.manifest_path_label,
                            &query_id,
                            expected,
                            expected,
                            actual,
                        ),
                        artifact,
                        line,
                    ));
                }
            }
            let embedding = vector
                .map(|vector| -> Result<_> {
                    Ok(QueryEmbeddingInput {
                        model_name: self
                            .obj
                            .get("query_embedding_model")
                            .and_then(JsonValue::as_str)
                            .map(|model| {
                                typed_work.copy_string(model, JsonInputKind::QueryEmbedding)
                            })
                            .transpose()?,
                        dim: vector.len(),
                        vector,
                    })
                })
                .transpose()
                .map_err(|err| self.locate_artifact_error(err, artifact, line))?;
            drop(obj);
            filter
                .reconcile_retained_in(JsonInputKind::Filter, &mut work)
                .map_err(|err| self.locate_artifact_error(err, artifact, line))?;
            retained.merge(work);
            retained.merge(typed_work);
            let has_embedding = embedding.is_some();
            rows.push(QueryRow {
                query_id,
                line,
                text: Some(text),
                filter,
                top_k,
                embedding,
                has_embedding,
            });
        }
        if rows.is_empty() {
            return Err(Error::new(
                ErrorCode::MissingRequiredField,
                "benchmark queries must not be empty",
            )
            .with_detail("field", "queries")
            .with_detail("manifest_path", &self.manifest_path_label));
        }
        Ok(rows)
    }

    fn validate_sidecar(
        &self,
        artifact: &Artifact,
        queries: &[QueryRow],
        expected_model: Option<&str>,
        expected_dim: Option<usize>,
        retained: &mut JsonReservation,
    ) -> Result<BTreeMap<String, QueryEmbeddingInput>> {
        if let Some(row) = queries.iter().find(|row| row.has_embedding) {
            return Err(self.mode_conflict(format!(
                "sidecar mode query {} at line {} must not inline embedding",
                row.query_id, row.line
            )));
        }

        let expected_model =
            expected_model.expect("sidecar contract validates query_embedding_model is present");
        let expected_dim = expected_dim.expect("sidecar contract validates query_embedding_dim");
        let query_ids = queries
            .iter()
            .map(|query| query.query_id.as_str())
            .collect::<BTreeSet<_>>();
        let mut seen = BTreeMap::<String, SidecarRow>::new();
        let mut embeddings = BTreeMap::<String, QueryEmbeddingInput>::new();
        for row in self.read_artifact_objects(artifact)? {
            let ArtifactObject {
                line,
                mut obj,
                _work: mut work,
                ..
            } = row?;
            let query_id =
                crate::json::take_required_string(&mut obj, "query_id").map_err(|err| {
                    Error::new(err.code(), err.message())
                        .with_detail("field", "query_id")
                        .with_detail("path", &artifact.raw_path)
                        .with_detail("artifact_kind", artifact.kind)
                        .with_detail("line", line)
                        .with_detail("manifest_path", &self.manifest_path_label)
                })?;
            self.require_nonempty_artifact_string(
                &query_id,
                "query_embeddings.query_id",
                artifact,
                line,
            )?;
            if let Some(first) = seen.get(&query_id) {
                return Err(self.locate_artifact_error(
                    Error::new(
                        ErrorCode::InvalidJsonlSchema,
                        "duplicate query_id in query embeddings sidecar",
                    )
                    .with_detail("query_id", &query_id)
                    .with_u64_detail("first_line", first.line as u64)
                    .with_u64_detail("duplicate_line", line as u64),
                    artifact,
                    line,
                ));
            }
            let row_model =
                crate::json::take_required_string(&mut obj, "model_name").map_err(|err| {
                    self.locate_artifact_error(
                        Error::new(err.code(), err.message())
                            .with_detail("field", "model_name")
                            .with_detail("query_id", &query_id),
                        artifact,
                        line,
                    )
                })?;
            self.require_nonempty_artifact_string(
                &row_model,
                "query_embeddings.model_name",
                artifact,
                line,
            )?;
            if row_model != expected_model {
                return Err(self.locate_artifact_error(
                    Error::new(
                        ErrorCode::EmbeddingModelMismatch,
                        "query embedding model does not match dataset manifest",
                    )
                    .with_detail("query_id", &query_id)
                    .with_detail("manifest_query_embedding_model", expected_model)
                    .with_detail("row_model_name", row_model),
                    artifact,
                    line,
                ));
            }
            let row_dim = required_usize(&obj, "dim").map_err(|err| {
                self.locate_artifact_error(
                    Error::new(err.code(), err.message())
                        .with_detail("field", "dim")
                        .with_detail("query_id", &query_id),
                    artifact,
                    line,
                )
            })?;
            let mut typed_work = self.context.reservation();
            let vector = self
                .vector_values(
                    obj.get("vector").ok_or_else(|| {
                        self.locate_artifact_error(
                            Error::new(
                                ErrorCode::MissingRequiredField,
                                "missing array field vector",
                            )
                            .with_detail("field", "vector")
                            .with_detail("query_id", &query_id),
                            artifact,
                            line,
                        )
                    })?,
                    &query_id,
                    line,
                    artifact.kind,
                    &mut typed_work,
                )
                .map_err(|err| self.locate_artifact_error(err, artifact, line))?;
            let vector_len = vector.len();
            if row_dim != expected_dim || vector_len != expected_dim {
                return Err(self.locate_artifact_error(
                    invalid_vector_len(
                        &self.manifest_path_label,
                        &query_id,
                        expected_dim,
                        row_dim,
                        vector_len,
                    ),
                    artifact,
                    line,
                ));
            }
            if query_ids.contains(query_id.as_str()) {
                drop(obj);
                work.retain(row_model.capacity());
                typed_work.merge(work);
                retained.merge(typed_work);
                embeddings.insert(
                    query_id.clone(),
                    QueryEmbeddingInput {
                        model_name: Some(row_model),
                        dim: row_dim,
                        vector,
                    },
                );
            }
            seen.insert(query_id, SidecarRow { line });
        }

        if let Some(row) = queries
            .iter()
            .find(|query| !seen.contains_key(&query.query_id))
        {
            return Err(Error::new(
                ErrorCode::MissingRequiredField,
                "query embedding sidecar row missing for query_id",
            )
            .with_detail("query_id", &row.query_id)
            .with_detail("field", "query_embeddings[query_id]")
            .with_detail("source", "query_embeddings.jsonl")
            .with_detail("path", &artifact.raw_path)
            .with_detail("artifact_kind", artifact.kind)
            .with_detail("manifest_path", &self.manifest_path_label));
        }
        Ok(embeddings)
    }

    fn validate_inline_queries(&self, queries: &[QueryRow]) -> Result<()> {
        if let Some(row) = queries.iter().find(|row| !row.has_embedding) {
            return Err(Error::new(
                ErrorCode::MissingRequiredField,
                "inline query embedding missing for query_id",
            )
            .with_detail("query_id", &row.query_id)
            .with_detail("field", "queries.embedding")
            .with_detail("line", row.line)
            .with_detail("manifest_path", &self.manifest_path_label));
        }
        Ok(())
    }

    fn validate_qrels(
        &self,
        artifact: &Artifact,
        queries: &[QueryRow],
        chunk_external_ids: &BTreeSet<String>,
    ) -> Result<BTreeMap<String, BTreeSet<String>>> {
        let query_ids = queries
            .iter()
            .map(|query| query.query_id.clone())
            .collect::<BTreeSet<_>>();
        let mut qrels = BTreeMap::<String, BTreeSet<String>>::new();
        let mut seen_pairs = BTreeMap::<(String, String), usize>::new();
        for row in self.read_artifact_objects(artifact)? {
            let ArtifactObject {
                line, obj, _work, ..
            } = row?;
            let query_id = required_string(&obj, "query_id").map_err(|err| {
                Error::new(err.code(), err.message())
                    .with_detail("field", "query_id")
                    .with_detail("path", &artifact.raw_path)
                    .with_detail("artifact_kind", artifact.kind)
                    .with_detail("line", line)
                    .with_detail("manifest_path", &self.manifest_path_label)
            })?;
            self.require_nonempty_artifact_string(&query_id, "qrels.query_id", artifact, line)?;
            let chunk_external_id = required_string(&obj, "chunk_external_id").map_err(|err| {
                Error::new(err.code(), err.message())
                    .with_detail("field", "chunk_external_id")
                    .with_detail("query_id", &query_id)
                    .with_detail("path", &artifact.raw_path)
                    .with_detail("artifact_kind", artifact.kind)
                    .with_detail("line", line)
                    .with_detail("manifest_path", &self.manifest_path_label)
            })?;
            self.require_nonempty_artifact_string(
                &chunk_external_id,
                "qrels.chunk_external_id",
                artifact,
                line,
            )?;
            let relevance = json_usize(obj.get("relevance").ok_or_else(|| {
                Error::new(
                    ErrorCode::MissingRequiredField,
                    "missing numeric field relevance",
                )
                .with_detail("field", "relevance")
                .with_detail("query_id", &query_id)
                .with_detail("path", &artifact.raw_path)
                .with_detail("artifact_kind", artifact.kind)
                .with_detail("line", line)
                .with_detail("manifest_path", &self.manifest_path_label)
            })?)
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::InvalidJsonlSchema,
                    "relevance must be a non-negative integer",
                )
                .with_detail("field", "relevance")
                .with_detail("query_id", &query_id)
                .with_detail("path", &artifact.raw_path)
                .with_detail("artifact_kind", artifact.kind)
                .with_detail("line", line)
                .with_detail("manifest_path", &self.manifest_path_label)
            })?;
            if !query_ids.contains(&query_id) {
                return Err(Error::new(
                    ErrorCode::InvalidJsonlSchema,
                    "qrels row references unknown query_id",
                )
                .with_detail("query_id", &query_id)
                .with_detail("field", "qrels.query_id")
                .with_detail("source", "qrels.jsonl")
                .with_detail("path", &artifact.raw_path)
                .with_detail("artifact_kind", artifact.kind)
                .with_detail("line", line)
                .with_detail("manifest_path", &self.manifest_path_label));
            }
            if !chunk_external_ids.contains(&chunk_external_id) {
                return Err(Error::new(
                    ErrorCode::InvalidJsonlSchema,
                    "qrels row references unknown chunk_external_id",
                )
                .with_detail("query_id", &query_id)
                .with_detail("chunk_external_id", &chunk_external_id)
                .with_detail("field", "qrels.chunk_external_id")
                .with_detail("source", "qrels.jsonl")
                .with_detail("path", &artifact.raw_path)
                .with_detail("artifact_kind", artifact.kind)
                .with_detail("line", line)
                .with_detail("manifest_path", &self.manifest_path_label));
            }
            if let Some(first_line) =
                seen_pairs.insert((query_id.clone(), chunk_external_id.clone()), line)
            {
                return Err(Error::new(
                    ErrorCode::InvalidJsonlSchema,
                    "duplicate qrels row for query_id and chunk_external_id",
                )
                .with_detail("query_id", &query_id)
                .with_detail("chunk_external_id", &chunk_external_id)
                .with_detail("field", "qrels.query_id+chunk_external_id")
                .with_detail("source", "qrels.jsonl")
                .with_detail("path", &artifact.raw_path)
                .with_detail("artifact_kind", artifact.kind)
                .with_detail("first_line", first_line)
                .with_detail("duplicate_line", line)
                .with_detail("manifest_path", &self.manifest_path_label));
            }
            let relevant = qrels.entry(query_id).or_default();
            if relevance > 0 {
                relevant.insert(chunk_external_id);
            }
        }
        if let Some(query) = queries
            .iter()
            .find(|query| !qrels.contains_key(&query.query_id))
        {
            return Err(Error::new(
                ErrorCode::MissingRequiredField,
                "qrels row missing for query_id",
            )
            .with_detail("query_id", &query.query_id)
            .with_detail("field", "qrels[query_id]")
            .with_detail("source", "qrels.jsonl")
            .with_detail("manifest_path", &self.manifest_path_label));
        }
        Ok(qrels)
    }

    fn validate_embeddings(
        &self,
        artifact: Option<&Artifact>,
        expected_model: Option<&str>,
        expected_dim: usize,
        chunk_external_ids: &BTreeSet<String>,
    ) -> Result<()> {
        let Some(artifact) = artifact else {
            return Ok(());
        };
        let mut seen = BTreeMap::<String, usize>::new();
        for row in self.read_artifact_objects(artifact)? {
            let ArtifactObject {
                line, obj, _work, ..
            } = row?;
            let chunk_external_id = required_string(&obj, "chunk_external_id").map_err(|err| {
                Error::new(err.code(), err.message())
                    .with_detail("field", "embeddings.chunk_external_id")
                    .with_detail("path", &artifact.raw_path)
                    .with_detail("artifact_kind", artifact.kind)
                    .with_detail("line", line)
                    .with_detail("manifest_path", &self.manifest_path_label)
            })?;
            self.require_nonempty_artifact_string(
                &chunk_external_id,
                "embeddings.chunk_external_id",
                artifact,
                line,
            )?;
            if !chunk_external_ids.contains(&chunk_external_id) {
                return Err(Error::new(
                    ErrorCode::InvalidJsonlSchema,
                    "embedding row references unknown chunk_external_id",
                )
                .with_detail("chunk_external_id", &chunk_external_id)
                .with_detail("field", "embeddings.chunk_external_id")
                .with_detail("source", "embeddings.jsonl")
                .with_detail("path", &artifact.raw_path)
                .with_detail("artifact_kind", artifact.kind)
                .with_detail("line", line)
                .with_detail("manifest_path", &self.manifest_path_label));
            }
            if let Some(first_line) = seen.insert(chunk_external_id.clone(), line) {
                return Err(Error::new(
                    ErrorCode::DuplicateExternalId,
                    "duplicate chunk_external_id in benchmark embeddings",
                )
                .with_detail("chunk_external_id", &chunk_external_id)
                .with_detail("field", "embeddings.chunk_external_id")
                .with_detail("source", "embeddings.jsonl")
                .with_detail("path", &artifact.raw_path)
                .with_detail("artifact_kind", artifact.kind)
                .with_detail("first_line", first_line)
                .with_detail("duplicate_line", line)
                .with_detail("manifest_path", &self.manifest_path_label));
            }
            let row_model = required_string(&obj, "model_name").map_err(|err| {
                Error::new(err.code(), err.message())
                    .with_detail("field", "embeddings.model_name")
                    .with_detail("chunk_external_id", &chunk_external_id)
                    .with_detail("path", &artifact.raw_path)
                    .with_detail("artifact_kind", artifact.kind)
                    .with_detail("line", line)
                    .with_detail("manifest_path", &self.manifest_path_label)
            })?;
            self.require_nonempty_artifact_string(
                &row_model,
                "embeddings.model_name",
                artifact,
                line,
            )?;
            if let Some(expected_model) = expected_model {
                if row_model != expected_model {
                    return Err(Error::new(
                        ErrorCode::EmbeddingModelMismatch,
                        "embedding model_name does not match dataset manifest",
                    )
                    .with_detail("chunk_external_id", &chunk_external_id)
                    .with_detail("manifest_embedding_model", expected_model)
                    .with_detail("row_model_name", row_model)
                    .with_detail("path", &artifact.raw_path)
                    .with_detail("artifact_kind", artifact.kind)
                    .with_detail("line", line)
                    .with_detail("manifest_path", &self.manifest_path_label));
                }
            }
            let row_dim = required_usize(&obj, "dim").map_err(|err| {
                Error::new(err.code(), err.message())
                    .with_detail("field", "embeddings.dim")
                    .with_detail("chunk_external_id", &chunk_external_id)
                    .with_detail("path", &artifact.raw_path)
                    .with_detail("artifact_kind", artifact.kind)
                    .with_detail("line", line)
                    .with_detail("manifest_path", &self.manifest_path_label)
            })?;
            let vector_len = self.validate_embedding_vector(
                obj.get("vector").ok_or_else(|| {
                    Error::new(
                        ErrorCode::MissingRequiredField,
                        "missing array field vector",
                    )
                    .with_detail("field", "embeddings.vector")
                    .with_detail("chunk_external_id", &chunk_external_id)
                    .with_detail("path", &artifact.raw_path)
                    .with_detail("artifact_kind", artifact.kind)
                    .with_detail("line", line)
                    .with_detail("manifest_path", &self.manifest_path_label)
                })?,
                &chunk_external_id,
                line,
                artifact,
            )?;
            if row_dim != expected_dim || vector_len != expected_dim {
                return Err(Error::new(
                    ErrorCode::InvalidVectorLength,
                    "embedding dimension does not match dataset manifest",
                )
                .with_detail("chunk_external_id", &chunk_external_id)
                .with_detail("expected_dim", expected_dim)
                .with_detail("row_dim", row_dim)
                .with_detail("actual_vector_length", vector_len)
                .with_detail("path", &artifact.raw_path)
                .with_detail("artifact_kind", artifact.kind)
                .with_detail("line", line)
                .with_detail("manifest_path", &self.manifest_path_label));
            }
        }
        if seen.is_empty() {
            return Err(Error::new(
                ErrorCode::MissingRequiredField,
                "benchmark embeddings must not be empty",
            )
            .with_detail("field", "embeddings")
            .with_detail("source", "embeddings.jsonl")
            .with_detail("path", &artifact.raw_path)
            .with_detail("artifact_kind", artifact.kind)
            .with_detail("manifest_path", &self.manifest_path_label));
        }
        if let Some(missing) = chunk_external_ids
            .iter()
            .find(|chunk_id| !seen.contains_key(*chunk_id))
        {
            return Err(Error::new(
                ErrorCode::MissingRequiredField,
                "embedding row missing for chunk_external_id",
            )
            .with_detail("chunk_external_id", missing)
            .with_detail("field", "embeddings[chunk_external_id]")
            .with_detail("source", "embeddings.jsonl")
            .with_detail("path", &artifact.raw_path)
            .with_detail("artifact_kind", artifact.kind)
            .with_detail("manifest_path", &self.manifest_path_label));
        }
        Ok(())
    }

    fn read_document_external_ids(&self, artifact: &Artifact) -> Result<BTreeSet<String>> {
        let mut ids = BTreeSet::new();
        for row in self.read_artifact_objects(artifact)? {
            let ArtifactObject {
                line, obj, _work, ..
            } = row?;
            let external_id = required_string(&obj, "external_id").map_err(|err| {
                Error::new(err.code(), err.message())
                    .with_detail("field", "external_id")
                    .with_detail("path", &artifact.raw_path)
                    .with_detail("artifact_kind", artifact.kind)
                    .with_detail("line", line)
                    .with_detail("manifest_path", &self.manifest_path_label)
            })?;
            self.require_nonempty_artifact_string(
                &external_id,
                "documents.external_id",
                artifact,
                line,
            )?;
            self.require_artifact_object_field(
                &obj,
                "metadata",
                "documents.metadata",
                artifact,
                line,
            )?;
            if !ids.insert(external_id.clone()) {
                return Err(Error::new(
                    ErrorCode::DuplicateExternalId,
                    "duplicate external_id in benchmark documents",
                )
                .with_detail("external_id", &external_id)
                .with_detail("field", "documents.external_id")
                .with_detail("path", &artifact.raw_path)
                .with_detail("artifact_kind", artifact.kind)
                .with_detail("line", line)
                .with_detail("manifest_path", &self.manifest_path_label));
            }
        }
        if ids.is_empty() {
            return Err(Error::new(
                ErrorCode::MissingRequiredField,
                "benchmark documents must not be empty",
            )
            .with_detail("field", "documents")
            .with_detail("path", &artifact.raw_path)
            .with_detail("artifact_kind", artifact.kind)
            .with_detail("manifest_path", &self.manifest_path_label));
        }
        Ok(ids)
    }

    fn read_chunk_external_ids(
        &self,
        artifact: &Artifact,
        document_external_ids: &BTreeSet<String>,
    ) -> Result<BTreeSet<String>> {
        let mut ids = BTreeSet::new();
        let mut ordinals_by_document = BTreeMap::<String, BTreeMap<usize, usize>>::new();
        for row in self.read_artifact_objects(artifact)? {
            let ArtifactObject {
                line, obj, _work, ..
            } = row?;
            let document_external_id =
                required_string(&obj, "document_external_id").map_err(|err| {
                    Error::new(err.code(), err.message())
                        .with_detail("field", "document_external_id")
                        .with_detail("path", &artifact.raw_path)
                        .with_detail("artifact_kind", artifact.kind)
                        .with_detail("line", line)
                        .with_detail("manifest_path", &self.manifest_path_label)
                })?;
            self.require_nonempty_artifact_string(
                &document_external_id,
                "chunks.document_external_id",
                artifact,
                line,
            )?;
            if !document_external_ids.contains(&document_external_id) {
                return Err(Error::new(
                    ErrorCode::InvalidJsonlSchema,
                    "benchmark chunk references unknown document_external_id",
                )
                .with_detail("document_external_id", &document_external_id)
                .with_detail("field", "chunks.document_external_id")
                .with_detail("path", &artifact.raw_path)
                .with_detail("artifact_kind", artifact.kind)
                .with_detail("line", line)
                .with_detail("manifest_path", &self.manifest_path_label));
            }
            let ordinal = required_usize(&obj, "ordinal").map_err(|err| {
                Error::new(err.code(), err.message())
                    .with_detail("field", "chunks.ordinal")
                    .with_detail("path", &artifact.raw_path)
                    .with_detail("artifact_kind", artifact.kind)
                    .with_detail("line", line)
                    .with_detail("manifest_path", &self.manifest_path_label)
            })?;
            if let Some(first_line) = ordinals_by_document
                .entry(document_external_id.clone())
                .or_default()
                .insert(ordinal, line)
            {
                return Err(Error::new(
                    ErrorCode::InvalidJsonlSchema,
                    "duplicate chunk ordinal for document in benchmark chunks",
                )
                .with_detail("document_external_id", &document_external_id)
                .with_detail("ordinal", ordinal)
                .with_detail("field", "chunks.ordinal")
                .with_detail("path", &artifact.raw_path)
                .with_detail("artifact_kind", artifact.kind)
                .with_detail("first_line", first_line)
                .with_detail("duplicate_line", line)
                .with_detail("manifest_path", &self.manifest_path_label));
            }
            let text = required_string(&obj, "text").map_err(|err| {
                Error::new(err.code(), err.message())
                    .with_detail("field", "chunks.text")
                    .with_detail("path", &artifact.raw_path)
                    .with_detail("artifact_kind", artifact.kind)
                    .with_detail("line", line)
                    .with_detail("manifest_path", &self.manifest_path_label)
            })?;
            if text.trim().is_empty() {
                return Err(Error::new(
                    ErrorCode::MissingRequiredField,
                    "benchmark chunk requires non-empty text",
                )
                .with_detail("field", "chunks.text")
                .with_detail("path", &artifact.raw_path)
                .with_detail("artifact_kind", artifact.kind)
                .with_detail("line", line)
                .with_detail("manifest_path", &self.manifest_path_label));
            }
            self.require_artifact_object_field(
                &obj,
                "metadata",
                "chunks.metadata",
                artifact,
                line,
            )?;
            let chunk_external_id = required_string(&obj, "chunk_external_id").map_err(|err| {
                Error::new(err.code(), err.message())
                    .with_detail("field", "chunk_external_id")
                    .with_detail("path", &artifact.raw_path)
                    .with_detail("artifact_kind", artifact.kind)
                    .with_detail("line", line)
                    .with_detail("manifest_path", &self.manifest_path_label)
            })?;
            self.require_nonempty_artifact_string(
                &chunk_external_id,
                "chunks.chunk_external_id",
                artifact,
                line,
            )?;
            if !ids.insert(chunk_external_id.clone()) {
                return Err(Error::new(
                    ErrorCode::DuplicateExternalId,
                    "duplicate chunk_external_id in benchmark chunks",
                )
                .with_detail("chunk_external_id", &chunk_external_id)
                .with_detail("field", "chunks.chunk_external_id")
                .with_detail("path", &artifact.raw_path)
                .with_detail("artifact_kind", artifact.kind)
                .with_detail("line", line)
                .with_detail("manifest_path", &self.manifest_path_label));
            }
        }
        if ids.is_empty() {
            return Err(Error::new(
                ErrorCode::MissingRequiredField,
                "benchmark chunks must not be empty",
            )
            .with_detail("field", "chunks")
            .with_detail("path", &artifact.raw_path)
            .with_detail("artifact_kind", artifact.kind)
            .with_detail("manifest_path", &self.manifest_path_label));
        }
        Ok(ids)
    }

    fn require_artifact_object_field(
        &self,
        obj: &JsonObject,
        field: &'static str,
        detail_field: &'static str,
        artifact: &Artifact,
        line: usize,
    ) -> Result<()> {
        match obj.get(field) {
            Some(JsonValue::Object(_)) => Ok(()),
            Some(_) => Err(Error::new(
                ErrorCode::InvalidJsonlSchema,
                format!("{detail_field} must be a JSON object"),
            )
            .with_detail("field", detail_field)
            .with_detail("path", &artifact.raw_path)
            .with_detail("artifact_kind", artifact.kind)
            .with_detail("line", line)
            .with_detail("manifest_path", &self.manifest_path_label)),
            None => Err(Error::new(
                ErrorCode::MissingRequiredField,
                format!("missing object field {detail_field}"),
            )
            .with_detail("field", detail_field)
            .with_detail("path", &artifact.raw_path)
            .with_detail("artifact_kind", artifact.kind)
            .with_detail("line", line)
            .with_detail("manifest_path", &self.manifest_path_label)),
        }
    }

    fn require_nonempty_artifact_string(
        &self,
        value: &str,
        detail_field: &'static str,
        artifact: &Artifact,
        line: usize,
    ) -> Result<()> {
        if value.trim().is_empty() {
            return Err(Error::new(
                ErrorCode::MissingRequiredField,
                format!("benchmark artifact field {detail_field} must not be empty"),
            )
            .with_detail("field", detail_field)
            .with_detail("path", &artifact.raw_path)
            .with_detail("artifact_kind", artifact.kind)
            .with_detail("line", line)
            .with_detail("manifest_path", &self.manifest_path_label));
        }
        Ok(())
    }

    fn read_artifact_objects<'a>(&'a self, artifact: &'a Artifact) -> Result<ArtifactRows<'a>> {
        let expected = self
            .hashes()?
            .get(artifact.hash_field)
            .and_then(JsonValue::as_str)
            .ok_or_else(|| {
                self.missing_benchmark_field(&format!("hashes.{}", artifact.hash_field))
            })?;
        read_jsonl_objects(artifact, &self.manifest_path_label, &self.context, expected)
    }

    fn locate_artifact_error(&self, error: Error, artifact: &Artifact, line: usize) -> Error {
        locate_artifact_error(error, artifact, &self.manifest_path_label, Some(line))
    }

    fn validate_text_only_queries(&self, queries: &[QueryRow]) -> Result<()> {
        if let Some(row) = queries.iter().find(|row| row.has_embedding) {
            return Err(self.mode_conflict(format!(
                "none mode query {} at line {} must not inline embedding",
                row.query_id, row.line
            )));
        }
        if let Some(row) = queries
            .iter()
            .find(|row| row.text.as_deref().unwrap_or("").trim().is_empty())
        {
            return Err(Error::new(
                ErrorCode::MissingRequiredField,
                "text-only benchmark query requires text",
            )
            .with_detail("field", "queries.text")
            .with_detail("query_id", &row.query_id)
            .with_detail("line", row.line)
            .with_detail("manifest_path", &self.manifest_path_label));
        }
        Ok(())
    }

    fn vector_values(
        &self,
        value: &JsonValue,
        query_id: &str,
        line: usize,
        artifact_kind: &str,
        work: &mut JsonReservation,
    ) -> Result<Vec<f32>> {
        let values = value.as_array().ok_or_else(|| {
            Error::new(
                ErrorCode::MissingRequiredField,
                "missing array field vector",
            )
            .with_detail("field", "vector")
            .with_detail("artifact_kind", artifact_kind)
            .with_detail("query_id", query_id)
            .with_detail("line", line)
            .with_detail("manifest_path", &self.manifest_path_label)
        })?;
        let mut vector = Vec::new();
        work.reserve_vec(&mut vector, values.len(), JsonInputKind::QueryEmbedding)?;
        for entry in values {
            let value = entry.as_f64().ok_or_else(|| {
                Error::new(
                    ErrorCode::InvalidVectorLength,
                    "query embedding vector entries must be finite numbers",
                )
                .with_detail("query_id", query_id)
                .with_detail("line", line)
                .with_detail("manifest_path", &self.manifest_path_label)
            })?;
            let value = value as f32;
            if !value.is_finite() {
                return Err(Error::new(
                    ErrorCode::InvalidVectorLength,
                    "query embedding vector entries must be finite f32 values",
                )
                .with_detail("query_id", query_id)
                .with_detail("line", line)
                .with_detail("manifest_path", &self.manifest_path_label));
            }
            vector.push(value);
        }
        crate::json::validate_query_vector(&vector)?;
        Ok(vector)
    }

    fn validate_embedding_vector(
        &self,
        value: &JsonValue,
        chunk_external_id: &str,
        line: usize,
        artifact: &Artifact,
    ) -> Result<usize> {
        let values = value.as_array().ok_or_else(|| {
            Error::new(
                ErrorCode::MissingRequiredField,
                "missing array field vector",
            )
            .with_detail("field", "embeddings.vector")
            .with_detail("path", &artifact.raw_path)
            .with_detail("artifact_kind", artifact.kind)
            .with_detail("chunk_external_id", chunk_external_id)
            .with_detail("line", line)
            .with_detail("manifest_path", &self.manifest_path_label)
        })?;
        for entry in values {
            let value = entry.as_f64().ok_or_else(|| {
                Error::new(
                    ErrorCode::InvalidVectorLength,
                    "embedding vector entries must be finite numbers",
                )
                .with_detail("field", "embeddings.vector")
                .with_detail("path", &artifact.raw_path)
                .with_detail("artifact_kind", artifact.kind)
                .with_detail("chunk_external_id", chunk_external_id)
                .with_detail("line", line)
                .with_detail("manifest_path", &self.manifest_path_label)
            })?;
            let value = value as f32;
            if !value.is_finite() {
                return Err(Error::new(
                    ErrorCode::InvalidVectorLength,
                    "embedding vector entries must be finite f32 values",
                )
                .with_detail("field", "embeddings.vector")
                .with_detail("path", &artifact.raw_path)
                .with_detail("artifact_kind", artifact.kind)
                .with_detail("chunk_external_id", chunk_external_id)
                .with_detail("line", line)
                .with_detail("manifest_path", &self.manifest_path_label));
            }
        }
        Ok(values.len())
    }

    fn missing_benchmark_field(&self, field: &str) -> Error {
        Error::new(
            ErrorCode::MissingRequiredField,
            format!("missing required benchmark field {field}"),
        )
        .with_detail("field", field)
        .with_detail("manifest_path", &self.manifest_path_label)
    }

    fn missing_source_field(&self, field: &'static str) -> Error {
        Error::new(
            ErrorCode::MissingRequiredField,
            format!("missing required field {field} for query_embedding_source"),
        )
        .with_detail("field", field)
        .with_detail(
            "query_embedding_source",
            self.query_embedding_source.as_str(),
        )
        .with_detail("manifest_path", &self.manifest_path_label)
    }

    fn mode_conflict(&self, message: impl Into<String>) -> Error {
        Error::new(ErrorCode::BenchmarkArtifactModeConflict, message)
            .with_detail(
                "query_embedding_source",
                self.query_embedding_source.as_str(),
            )
            .with_detail("manifest_path", &self.manifest_path_label)
    }
}

fn reject_unknown_benchmark_manifest_fields(
    obj: &JsonObject,
    manifest_path_label: &str,
) -> Result<()> {
    for field in obj.keys() {
        if !matches!(
            field.as_str(),
            "schema_version"
                | "dataset_id"
                | "version"
                | "documents"
                | "chunks"
                | "embeddings"
                | "queries"
                | "qrels"
                | "query_embedding_source"
                | "query_embeddings"
                | "embedding_model"
                | "query_embedding_model"
                | "query_embedding_dim"
                | "chunking"
                | "tokenizer"
                | "hashes"
        ) {
            return Err(Error::new(
                ErrorCode::InvalidJsonlSchema,
                "benchmark dataset manifest contains unsupported field",
            )
            .with_detail("field", "manifest")
            .with_detail("manifest_path", manifest_path_label));
        }
    }
    Ok(())
}

fn reject_unknown_benchmark_hash_fields(obj: &JsonObject, manifest_path_label: &str) -> Result<()> {
    for field in obj.keys() {
        if !matches!(
            field.as_str(),
            "documents_sha256"
                | "chunks_sha256"
                | "embeddings_sha256"
                | "queries_sha256"
                | "qrels_sha256"
                | "query_embeddings_sha256"
        ) {
            return Err(Error::new(
                ErrorCode::InvalidJsonlSchema,
                "benchmark dataset manifest hashes contains unsupported field",
            )
            .with_detail("field", "hashes")
            .with_detail("manifest_path", manifest_path_label));
        }
    }
    Ok(())
}

fn validate_benchmark_hash_field_values(obj: &JsonObject, manifest_path_label: &str) -> Result<()> {
    for (field, value) in obj {
        let Some(hash) = value.as_str() else {
            return Err(Error::new(
                ErrorCode::InvalidJsonlSchema,
                "benchmark manifest hash fields must be strings",
            )
            .with_detail("field", format!("hashes.{field}"))
            .with_detail("manifest_path", manifest_path_label));
        };
        if hash.trim().is_empty() {
            return Err(Error::new(
                ErrorCode::MissingRequiredField,
                "benchmark manifest hash fields must not be empty",
            )
            .with_detail("field", format!("hashes.{field}"))
            .with_detail("manifest_path", manifest_path_label));
        }
    }
    Ok(())
}

fn optional_string_field(obj: &JsonObject, field: &str) -> Option<String> {
    obj.get(field)
        .and_then(JsonValue::as_str)
        .map(ToOwned::to_owned)
}

struct ArtifactObject {
    line: usize,
    obj: JsonObject,
    _work: JsonReservation,
    search_top_k: Option<usize>,
}

// Rows remain provisional until exhaustion verifies the digest of the parsed stream.
struct ArtifactRows<'a> {
    artifact: &'a Artifact,
    manifest_path_label: &'a str,
    rows: Option<JsonLines<Sha256Reader<File>>>,
    expected_sha256: &'a str,
}

fn locate_artifact_error(
    mut error: Error,
    artifact: &Artifact,
    manifest: &str,
    line: Option<usize>,
) -> Error {
    // Location is additive: core diagnostics, numeric types and original IO causes
    // survive unchanged. Only caller-supplied labels may enter public paths.
    for (key, value) in [
        ("path", artifact.raw_path.as_str()),
        ("artifact_kind", artifact.kind),
        ("manifest_path", manifest),
    ] {
        if !error.detail().iter().any(|(field, _)| field == key) {
            error = error.with_detail(key, value);
        }
    }
    if let Some(line) = line {
        if !error.detail().iter().any(|(key, _)| key == "line") {
            error = error.with_detail("line", line);
        }
    }
    error
}

fn artifact_hash_mismatch(
    artifact: &Artifact,
    manifest: &str,
    expected: &str,
    actual: &str,
) -> Error {
    Error::new(
        ErrorCode::ChecksumMismatch,
        format!("{} hash mismatch", artifact.kind),
    )
    .with_detail("field", format!("hashes.{}", artifact.hash_field))
    .with_detail("path", &artifact.raw_path)
    .with_detail("artifact_kind", artifact.kind)
    .with_detail("expected_sha256", expected)
    .with_detail("actual_sha256", actual)
    .with_detail("manifest_path", manifest)
}

fn read_jsonl_objects<'a>(
    artifact: &'a Artifact,
    manifest_path_label: &'a str,
    context: &JsonOperationContext,
    expected_sha256: &'a str,
) -> Result<ArtifactRows<'a>> {
    let file = fs::File::open(&artifact.path)
        .map_err(|err| locate_artifact_error(err.into(), artifact, manifest_path_label, None))?;
    let schema = match artifact.kind {
        "documents" => JsonSchema::DocumentRow,
        "chunks" => JsonSchema::ChunkRow,
        "embeddings" => JsonSchema::EmbeddingRow,
        "queries" => JsonSchema::QueryRow,
        "query_embeddings" => JsonSchema::QueryEmbeddingObject,
        _ => JsonSchema::BenchmarkRecord,
    };
    let rows = JsonLines::new(Sha256Reader::new(file), schema, context, u64::MAX)
        .map_err(|error| locate_artifact_error(error, artifact, manifest_path_label, None))?;
    Ok(ArtifactRows {
        artifact,
        manifest_path_label,
        rows: Some(rows),
        expected_sha256,
    })
}

impl Iterator for ArtifactRows<'_> {
    type Item = Result<ArtifactObject>;

    fn next(&mut self) -> Option<Self::Item> {
        let row = match self.rows.as_mut()?.next() {
            Some(row) => row,
            None => {
                let actual = self.rows.take().unwrap().into_inner().finish();
                return (actual != self.expected_sha256).then(|| {
                    Err(artifact_hash_mismatch(
                        self.artifact,
                        self.manifest_path_label,
                        self.expected_sha256,
                        &actual,
                    ))
                });
            }
        };
        let result = (|| {
            let artifact = self.artifact;
            let manifest_path_label = self.manifest_path_label;
            let (line, parsed) = row.map_err(|error| {
                locate_artifact_error(error, artifact, manifest_path_label, None)
            })?;
            let (mut value, work, search_top_k) = parsed.into_parts_with_search_top_k();
            let obj = value.as_object_mut().map(std::mem::take).ok_or_else(|| {
                Error::new(
                    ErrorCode::InvalidJsonlSchema,
                    format!("{} row must be a JSON object", artifact.kind),
                )
                .with_detail("path", &artifact.raw_path)
                .with_detail("artifact_kind", artifact.kind)
                .with_detail("line", line)
                .with_detail("manifest_path", manifest_path_label)
            })?;
            reject_unknown_benchmark_artifact_fields(&obj, artifact, line, manifest_path_label)?;
            Ok(ArtifactObject {
                line,
                obj,
                _work: work,
                search_top_k,
            })
        })();
        if result.is_err() {
            self.rows = None;
        }
        Some(result)
    }
}

impl std::iter::FusedIterator for ArtifactRows<'_> {}

fn reject_unknown_benchmark_artifact_fields(
    obj: &JsonObject,
    artifact: &Artifact,
    line: usize,
    manifest_path_label: &str,
) -> Result<()> {
    let allowed = match artifact.kind {
        "documents" => BENCHMARK_DOCUMENT_FIELDS,
        "chunks" => BENCHMARK_CHUNK_FIELDS,
        "embeddings" => BENCHMARK_EMBEDDING_FIELDS,
        "queries" => BENCHMARK_QUERY_FIELDS,
        "qrels" => BENCHMARK_QREL_FIELDS,
        "query_embeddings" => BENCHMARK_QUERY_EMBEDDING_FIELDS,
        _ => return Ok(()),
    };
    for field in obj.keys() {
        if !allowed.contains(&field.as_str()) {
            return Err(Error::new(
                ErrorCode::InvalidJsonlSchema,
                "benchmark artifact row contains unsupported field",
            )
            .with_detail("field", artifact.kind)
            .with_detail("path", &artifact.raw_path)
            .with_detail("artifact_kind", artifact.kind)
            .with_detail("line", line)
            .with_detail("manifest_path", manifest_path_label));
        }
    }
    Ok(())
}

fn invalid_vector_len(
    manifest_path: &str,
    query_id: &str,
    expected_dim: usize,
    row_dim: usize,
    actual_vector_length: usize,
) -> Error {
    Error::new(
        ErrorCode::InvalidVectorLength,
        "query embedding dimension does not match dataset manifest",
    )
    .with_detail("query_id", query_id)
    .with_u64_detail("expected_dim", expected_dim as u64)
    .with_u64_detail("row_dim", row_dim as u64)
    .with_u64_detail("actual_vector_length", actual_vector_length as u64)
    .with_detail("manifest_path", manifest_path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash::sha256_hex;

    fn temp_dir(name: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!("cofferdb-benchmark-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap();
        path
    }

    fn write_file(dir: &Path, name: &str, content: &str) -> String {
        let path = dir.join(name);
        fs::write(&path, content).unwrap();
        sha256_hex(&fs::read(path).unwrap())
    }

    #[test]
    fn streamed_manifest_hash_includes_the_exact_original_bytes() {
        let dir = temp_dir("streamed-manifest-hash");
        let path = dir.join("dataset_manifest.json");
        let raw = format!(
            "{}{{\"query_embedding_source\":\"none\"}}\n\t",
            " ".repeat(20_000)
        );
        fs::write(&path, &raw).unwrap();
        let input =
            BenchmarkManifestInput::load(&path, &JsonOperationContext::standalone()).unwrap();
        assert_eq!(input.manifest_sha256, sha256_hex(raw.as_bytes()));
        assert_ne!(
            input.manifest_sha256,
            sha256_hex(JsonValue::Object(input.obj).to_compact_string().as_bytes())
        );
        fs::remove_dir_all(dir).unwrap();
    }

    fn write_sidecar_fixture_with_artifacts(
        dir: &Path,
        documents_content: &str,
        chunks_content: &str,
        embeddings_content: &str,
        queries_content: &str,
        qrels_content: &str,
        query_embeddings_content: &str,
    ) {
        let documents = write_file(dir, "documents.jsonl", documents_content);
        let chunks = write_file(dir, "chunks.jsonl", chunks_content);
        let embeddings = write_file(dir, "embeddings.jsonl", embeddings_content);
        let queries = write_file(dir, "queries.jsonl", queries_content);
        let qrels = write_file(dir, "qrels.jsonl", qrels_content);
        let query_embeddings = write_file(dir, "query_embeddings.jsonl", query_embeddings_content);
        fs::write(
            dir.join("dataset_manifest.json"),
            format!(
                concat!(
                    r#"{{"schema_version":"0.1","dataset_id":"bench","version":"0.1","#,
                    r#""chunking":{{"chunk_size":800,"chunk_overlap":120}},"#,
                    r#""documents":"documents.jsonl","chunks":"chunks.jsonl","#,
                    r#""embeddings":"embeddings.jsonl","queries":"queries.jsonl","#,
                    r#""qrels":"qrels.jsonl","query_embedding_source":"sidecar","#,
                    r#""query_embeddings":"query_embeddings.jsonl","embedding_model":"m","#,
                    r#""query_embedding_model":"m","query_embedding_dim":2,"hashes":{{"#,
                    r#""documents_sha256":"{}","chunks_sha256":"{}","#,
                    r#""embeddings_sha256":"{}","queries_sha256":"{}","#,
                    r#""qrels_sha256":"{}","query_embeddings_sha256":"{}"}}"#,
                    r#"}}"#
                ),
                documents, chunks, embeddings, queries, qrels, query_embeddings
            ),
        )
        .unwrap();
    }

    fn detail_value<'a>(err: &'a Error, key: &str) -> Option<&'a str> {
        err.detail()
            .iter()
            .find(|(field, _)| field == key)
            .map(|(_, value)| value.as_str())
    }

    fn assert_artifact_error_is_additive(error: Error, line: Option<usize>) -> Error {
        let before = error.to_json();
        let had_source = std::error::Error::source(&error).is_some();
        let artifact = Artifact {
            raw_path: "inputs/queries.jsonl".into(),
            path: PathBuf::from("/private/internal-unpublished-artifact/queries.jsonl"),
            kind: "queries",
            hash_field: "queries_sha256",
        };
        let located = locate_artifact_error(error, &artifact, "dataset.json", line);
        let once = located.to_json();
        let located = locate_artifact_error(located, &artifact, "other.json", Some(999));
        let after = located.to_json();
        for key in ["code", "message", "recoverable", "suggested_action"] {
            assert_eq!(before.get(key), after.get(key), "{key}");
        }
        for (key, value) in before.get("detail").unwrap().as_object().unwrap() {
            assert_eq!(after.get("detail").unwrap().get(key), Some(value), "{key}");
        }
        for (key, value) in once.get("detail").unwrap().as_object().unwrap() {
            assert_eq!(after.get("detail").unwrap().get(key), Some(value), "{key}");
        }
        assert_eq!(std::error::Error::source(&located).is_some(), had_source);
        let unique = located
            .detail()
            .iter()
            .map(|(key, _)| key)
            .collect::<BTreeSet<_>>();
        assert_eq!(unique.len(), located.detail().len());
        assert!(!after
            .to_compact_string()
            .contains("internal-unpublished-artifact"));
        located
    }

    #[test]
    fn artifact_location_preserves_resource_contract_and_open_operations_at_100_boundaries() {
        for seed in 0..100_u128 {
            for resource in ["memory", "catalog", "temporary_disk"] {
                for limit in [0, (1_u64 << 53) + 1, u64::MAX] {
                    let requested = if seed % 2 == 0 {
                        u128::from(limit) + seed + 1
                    } else {
                        u128::MAX - seed
                    };
                    let operation = format!("future_bounded_operation_{seed}");
                    let error =
                        Error::new(ErrorCode::ResourceExhausted, "resource budget exhausted")
                            .with_detail("resource", resource)
                            .with_detail("operation", &operation)
                            .with_detail("limit_bytes", limit)
                            .with_detail("requested_bytes", requested);
                    let error = assert_artifact_error_is_additive(error, Some(seed as usize + 1));
                    assert_eq!(detail_value(&error, "operation"), Some(operation.as_str()));
                    assert!(!crate::json::is_limit_error(&error));
                    let json = error.to_json();
                    assert_eq!(json.get("recoverable"), Some(&JsonValue::Bool(true)));
                    let detail = json.get("detail").unwrap();
                    assert_eq!(
                        detail.get("limit_bytes").unwrap().as_str(),
                        Some(limit.to_string().as_str())
                    );
                    assert_eq!(
                        detail.get("requested_bytes").unwrap().as_str(),
                        Some(requested.to_string().as_str())
                    );
                    assert_eq!(
                        detail.get("path").unwrap().as_str(),
                        Some("inputs/queries.jsonl")
                    );
                }
            }
        }
    }

    #[test]
    fn artifact_location_preserves_json_index_io_and_integrity_diagnostics() {
        for kind in [
            JsonInputKind::JsonlRecord,
            JsonInputKind::BenchmarkArtifact,
            JsonInputKind::Filter,
            JsonInputKind::Weights,
            JsonInputKind::MetadataBoost,
            JsonInputKind::StoredJson,
        ] {
            let context = JsonOperationContext::standalone();
            let error = context
                .reservation()
                .grow(
                    crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES + 1,
                    kind,
                )
                .unwrap_err();
            let error = assert_artifact_error_is_additive(error, None);
            assert!(crate::json::is_limit_error(&error));
            assert!(detail_value(&error, "resource").is_none());
        }
        let errors = [
            Error::new(ErrorCode::IndexCorruption, "index unavailable")
                .with_detail(
                    "index_unavailable_reason",
                    "transient_rebuild_budget_exceeded",
                )
                .with_detail("index_source_attempted", "transient_rebuild")
                .with_detail("resource_profile", "constrained")
                .with_u64_detail("budget_mb", 64),
            Error::from(std::io::Error::from(std::io::ErrorKind::PermissionDenied)),
            Error::from(std::io::Error::from(std::io::ErrorKind::UnexpectedEof)),
            Error::new(ErrorCode::ChecksumMismatch, "hash mismatch")
                .with_detail("expected_sha256", "a".repeat(64))
                .with_detail("actual_sha256", "b".repeat(64)),
            Error::new(ErrorCode::FileFormatError, "invalid header"),
        ];
        for error in errors {
            let error = assert_artifact_error_is_additive(error, Some(2));
            assert!(detail_value(&error, "resource").is_none());
        }
    }

    #[test]
    fn artifact_location_never_overwrites_existing_core_locator_or_integer_types() {
        let error = Error::new(ErrorCode::InvalidEmbeddingDimension, "dimension mismatch")
            .with_detail("field", "vector")
            .with_detail("path", "original/sidecar.jsonl")
            .with_detail("artifact_kind", "query_embeddings")
            .with_detail("manifest_path", "original/dataset.json")
            .with_u64_detail("line", 7)
            .with_u64_detail("expected_dim", 768)
            .with_u64_detail("actual_vector_length", 1);
        let before = error.to_json();
        let after = assert_artifact_error_is_additive(error, Some(19)).to_json();
        assert_eq!(before, after);
    }

    #[test]
    fn artifact_parsing_rechecks_the_bytes_after_initial_hash_validation() {
        let dir = temp_dir("artifact-stream-identity");
        write_sidecar_fixture_with_artifacts(
            &dir,
            r#"{"external_id":"doc","metadata":{}}"#,
            r#"{"document_external_id":"doc","chunk_external_id":"c","ordinal":0,"text":"needle","metadata":{}}"#,
            r#"{"chunk_external_id":"c","model_name":"m","dim":2,"vector":[1,0]}"#,
            r#"{"query_id":"q","text":"needle","filter":{},"top_k":1}"#,
            r#"{"query_id":"q","chunk_external_id":"c","relevance":1}"#,
            r#"{"query_id":"q","model_name":"m","dim":2,"vector":[1,0]}"#,
        );
        let kinds = [
            "documents",
            "chunks",
            "embeddings",
            "queries",
            "qrels",
            "query_embeddings",
        ];
        let context = JsonOperationContext::standalone();
        for seed in 0..100 {
            let kind = kinds[seed % kinds.len()];
            let artifact = Artifact {
                raw_path: format!("{kind}.jsonl"),
                path: dir.join(format!("{kind}.jsonl")),
                kind,
                hash_field: match kind {
                    "documents" => "documents_sha256",
                    "chunks" => "chunks_sha256",
                    "embeddings" => "embeddings_sha256",
                    "queries" => "queries_sha256",
                    "qrels" => "qrels_sha256",
                    _ => "query_embeddings_sha256",
                },
            };
            let before = fs::read(&artifact.path).unwrap();
            let input = BenchmarkManifestInput::load(&dir, &context).unwrap();
            input.validate_hash(&artifact).unwrap();
            let mut changed = before.clone();
            changed.extend(vec![b' '; seed * 211 + 1]);
            fs::write(&artifact.path, &changed).unwrap();
            let mut rows = input.read_artifact_objects(&artifact).unwrap();
            let parsed = rows.by_ref().collect::<Result<Vec<_>>>();
            let error = parsed
                .err()
                .expect("parsed artifact must match its frozen input hash");
            assert!(rows.next().is_none());
            assert!(rows.next().is_none());
            drop(rows);
            assert_eq!(error.code(), ErrorCode::ChecksumMismatch);
            assert_eq!(detail_value(&error, "artifact_kind"), Some(kind));
            assert_eq!(
                detail_value(&error, "expected_sha256"),
                Some(sha256_hex(&before).as_str())
            );
            assert_eq!(
                detail_value(&error, "actual_sha256"),
                Some(sha256_hex(&changed).as_str())
            );
            fs::write(&artifact.path, before).unwrap();
            for row in input.read_artifact_objects(&artifact).unwrap() {
                row.unwrap();
            }
            drop(input);
            context
                .reservation()
                .grow(
                    crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
                    JsonInputKind::BenchmarkArtifact,
                )
                .unwrap();
        }
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn artifact_contract_errors_do_not_expose_later_rows() {
        let dir = temp_dir("artifact-contract-fused");
        let artifact = Artifact {
            raw_path: "queries.jsonl".into(),
            path: dir.join("queries.jsonl"),
            kind: "queries",
            hash_field: "queries_sha256",
        };
        for invalid in [r#"{"unexpected":"private_value"}"#, "[]", "{"] {
            let bytes = format!("{invalid}\n{{\"query_id\":\"must_not_be_exposed\"}}\n");
            fs::write(&artifact.path, &bytes).unwrap();
            let hash = sha256_hex(bytes.as_bytes());
            let context = JsonOperationContext::standalone();
            let mut rows =
                read_jsonl_objects(&artifact, "dataset_manifest.json", &context, &hash).unwrap();
            let error = rows.next().unwrap().err().unwrap();
            assert_eq!(error.code(), ErrorCode::InvalidJsonlSchema);
            assert_eq!(detail_value(&error, "line"), Some("1"));
            assert!(!error.to_string().contains("private_value"));
            assert!(rows.next().is_none());
            assert!(rows.next().is_none());
            drop(rows);
            context
                .reservation()
                .grow(
                    crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
                    JsonInputKind::BenchmarkArtifact,
                )
                .unwrap();
        }
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn artifact_rows_are_incremental_and_do_not_hide_later_errors() {
        let dir = temp_dir("incremental-jsonl");
        let path = dir.join("queries.jsonl");
        fs::write(&path, format!("{{\"query_id\":\"first\",\"text\":\"payload\",\"top_k\":1}}\n\n{{\"query_id\":\"second\",\"filter\":{}0{}}}\n", "[".repeat(30_000), "]".repeat(30_000))).unwrap();
        let artifact = Artifact {
            raw_path: "queries.jsonl".into(),
            path,
            kind: "queries",
            hash_field: "queries_sha256",
        };
        let hash = sha256_hex(&fs::read(&artifact.path).unwrap());
        let mut rows = read_jsonl_objects(
            &artifact,
            "dataset_manifest.json",
            &JsonOperationContext::standalone(),
            &hash,
        )
        .unwrap();
        let first = rows.next().unwrap().unwrap();
        assert_eq!(first.line, 1);
        assert_eq!(first.obj.get("query_id").unwrap().as_str(), Some("first"));
        assert!(first._work.reserved_bytes() > 0);
        let error = rows.next().unwrap().err().unwrap();
        assert_eq!(error.code(), ErrorCode::InvalidMetadataFilter);
        assert_eq!(detail_value(&error, "line"), Some("3"));
        assert_eq!(detail_value(&error, "json_input_kind"), Some("filter"));
        assert_eq!(detail_value(&error, "path"), Some("queries.jsonl"));
        assert!(rows.next().is_none());
        drop((first, rows));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn deep_jsonl_artifact_keeps_depth_reason_and_private_locator() {
        let dir = temp_dir("json-depth");
        let path = dir.join("queries.jsonl");
        fs::write(
            &path,
            format!(
                "{{\"filter\":{}0{}}}\n",
                "[".repeat(30_000),
                "]".repeat(30_000)
            ),
        )
        .unwrap();
        let artifact = Artifact {
            raw_path: "queries.jsonl".into(),
            path,
            kind: "queries",
            hash_field: "queries_sha256",
        };
        let err = read_jsonl_objects(
            &artifact,
            "dataset_manifest.json",
            &JsonOperationContext::standalone(),
            &sha256_hex(&fs::read(&artifact.path).unwrap()),
        )
        .unwrap()
        .next()
        .unwrap()
        .err()
        .unwrap();
        assert_eq!(err.code(), ErrorCode::InvalidMetadataFilter);
        for (field, expected) in [
            ("json_limit_reason", "depth_exceeded"),
            ("limit", "128"),
            ("line", "1"),
            ("path", "queries.jsonl"),
            ("artifact_kind", "queries"),
            ("manifest_path", "dataset_manifest.json"),
        ] {
            assert_eq!(detail_value(&err, field), Some(expected));
        }
        assert!(!err.message().contains(dir.to_str().unwrap()));
        fs::remove_dir_all(dir).unwrap();
    }

    fn write_sidecar_fixture(
        dir: &Path,
        queries_content: &str,
        qrels_content: &str,
        query_embeddings_content: &str,
    ) {
        let documents = write_file(
            dir,
            "documents.jsonl",
            r#"{"external_id":"doc","metadata":{}}"#,
        );
        let chunks = write_file(
            dir,
            "chunks.jsonl",
            r#"{"document_external_id":"doc","chunk_external_id":"doc#1","ordinal":1,"text":"wal","metadata":{}}"#,
        );
        let embeddings = write_file(
            dir,
            "embeddings.jsonl",
            r#"{"chunk_external_id":"doc#1","model_name":"m","dim":2,"vector":[1.0,0.0]}"#,
        );
        let queries = write_file(dir, "queries.jsonl", queries_content);
        let qrels = write_file(dir, "qrels.jsonl", qrels_content);
        let query_embeddings = write_file(dir, "query_embeddings.jsonl", query_embeddings_content);
        fs::write(
            dir.join("dataset_manifest.json"),
            format!(
                concat!(
                    r#"{{"schema_version":"0.1","dataset_id":"bench","version":"0.1","#,
                    r#""chunking":{{"chunk_size":800,"chunk_overlap":120}},"#,
                    r#""documents":"documents.jsonl","chunks":"chunks.jsonl","#,
                    r#""embeddings":"embeddings.jsonl","queries":"queries.jsonl","#,
                    r#""qrels":"qrels.jsonl","query_embedding_source":"sidecar","#,
                    r#""query_embeddings":"query_embeddings.jsonl","embedding_model":"m","#,
                    r#""query_embedding_model":"m","query_embedding_dim":2,"hashes":{{"#,
                    r#""documents_sha256":"{}","chunks_sha256":"{}","#,
                    r#""embeddings_sha256":"{}","queries_sha256":"{}","#,
                    r#""qrels_sha256":"{}","query_embeddings_sha256":"{}"}}"#,
                    r#"}}"#
                ),
                documents, chunks, embeddings, queries, qrels, query_embeddings
            ),
        )
        .unwrap();
    }

    fn write_sidecar_fixture_with_chunks_and_embeddings(
        dir: &Path,
        chunks_content: &str,
        embeddings_content: &str,
    ) {
        let documents = write_file(
            dir,
            "documents.jsonl",
            r#"{"external_id":"doc","metadata":{}}"#,
        );
        let chunks = write_file(dir, "chunks.jsonl", chunks_content);
        let embeddings = write_file(dir, "embeddings.jsonl", embeddings_content);
        let queries = write_file(
            dir,
            "queries.jsonl",
            r#"{"query_id":"q1","text":"wal","top_k":10}"#,
        );
        let qrels = write_file(
            dir,
            "qrels.jsonl",
            r#"{"query_id":"q1","chunk_external_id":"doc#1","relevance":2}"#,
        );
        let query_embeddings = write_file(
            dir,
            "query_embeddings.jsonl",
            r#"{"query_id":"q1","model_name":"m","dim":2,"vector":[1.0,0.0]}"#,
        );
        fs::write(
            dir.join("dataset_manifest.json"),
            format!(
                concat!(
                    r#"{{"schema_version":"0.1","dataset_id":"bench","version":"0.1","#,
                    r#""chunking":{{"chunk_size":800,"chunk_overlap":120}},"#,
                    r#""documents":"documents.jsonl","chunks":"chunks.jsonl","#,
                    r#""embeddings":"embeddings.jsonl","queries":"queries.jsonl","#,
                    r#""qrels":"qrels.jsonl","query_embedding_source":"sidecar","#,
                    r#""query_embeddings":"query_embeddings.jsonl","embedding_model":"m","#,
                    r#""query_embedding_model":"m","query_embedding_dim":2,"hashes":{{"#,
                    r#""documents_sha256":"{}","chunks_sha256":"{}","#,
                    r#""embeddings_sha256":"{}","queries_sha256":"{}","#,
                    r#""qrels_sha256":"{}","query_embeddings_sha256":"{}"}}"#,
                    r#"}}"#
                ),
                documents, chunks, embeddings, queries, qrels, query_embeddings
            ),
        )
        .unwrap();
    }

    fn write_none_fixture(dir: &Path, queries_content: &str, qrels_content: &str) {
        let documents = write_file(
            dir,
            "documents.jsonl",
            r#"{"external_id":"doc","metadata":{}}"#,
        );
        let chunks = write_file(
            dir,
            "chunks.jsonl",
            r#"{"document_external_id":"doc","chunk_external_id":"doc#1","ordinal":1,"text":"wal","metadata":{"lang":"rust"}}"#,
        );
        let queries = write_file(dir, "queries.jsonl", queries_content);
        let qrels = write_file(dir, "qrels.jsonl", qrels_content);
        fs::write(
            dir.join("dataset_manifest.json"),
            format!(
                concat!(
                    r#"{{"schema_version":"0.1","dataset_id":"bench","version":"0.1","#,
                    r#""documents":"documents.jsonl","chunks":"chunks.jsonl","#,
                    r#""queries":"queries.jsonl","qrels":"qrels.jsonl","#,
                    r#""query_embedding_source":"none","hashes":{{"#,
                    r#""documents_sha256":"{}","chunks_sha256":"{}","#,
                    r#""queries_sha256":"{}","qrels_sha256":"{}"}}"#,
                    r#"}}"#
                ),
                documents, chunks, queries, qrels
            ),
        )
        .unwrap();
    }

    #[test]
    fn benchmark_query_filters_share_work_across_rows_and_release_on_failure() {
        let dir = temp_dir("query-shared-work");
        let label = "x".repeat(256 * 1024);
        let queries = (0..2).map(|index| format!("{{\"query_id\":\"private_query_{index}\",\"text\":\"wal\",\"filter\":{{\"chunk.private_key\":\"{label}\"}},\"top_k\":10}}\n")).collect::<String>();
        let qrels = (0..2).map(|index| format!("{{\"query_id\":\"private_query_{index}\",\"chunk_external_id\":\"doc#1\",\"relevance\":1}}\n")).collect::<String>();
        write_none_fixture(&dir, &queries, &qrels);
        for available in [1280 * 1024, 2 * 1024 * 1024] {
            let context = JsonOperationContext::standalone();
            let mut busy = context.reservation();
            busy.grow(
                crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES - available,
                JsonInputKind::BenchmarkArtifact,
            )
            .unwrap();
            let result = BenchmarkDataset::load_and_validate_in(&dir, &context);
            if available == 1280 * 1024 {
                let error = result.err().unwrap();
                assert_eq!(error.code(), ErrorCode::InvalidMetadataFilter);
                assert_eq!(
                    detail_value(&error, "json_limit_reason"),
                    Some("work_budget_exceeded")
                );
                assert_eq!(detail_value(&error, "json_input_kind"), Some("filter"));
                assert_eq!(detail_value(&error, "line"), Some("2"));
                assert_eq!(detail_value(&error, "path"), Some("queries.jsonl"));
                assert_eq!(detail_value(&error, "artifact_kind"), Some("queries"));
                assert!(detail_value(&error, "query_id").is_none());
                assert!(!error.to_json().to_compact_string().contains("private_"));
            } else {
                let (dataset, work) = result.unwrap();
                assert_eq!(dataset.queries.len(), 2);
                let expected = dataset
                    .queries
                    .iter()
                    .map(|query| {
                        query
                            .filter
                            .reserve_retained_in(
                                JsonInputKind::Filter,
                                &JsonOperationContext::standalone(),
                            )
                            .unwrap()
                            .reserved_bytes()
                    })
                    .sum::<usize>();
                assert_eq!(work.reserved_bytes(), expected);
                drop(dataset);
                drop(work);
            }
            busy.grow(available, JsonInputKind::BenchmarkArtifact)
                .unwrap();
        }
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn benchmark_query_vectors_and_manifest_values_retain_exact_owned_work() {
        let dir = temp_dir("query-vector-retained-work");
        let queries = (0..100).map(|index| format!("{{\"query_id\":\"q{index}\",\"text\":\"wal\",\"filter\":{{\"chunk.n\":{index}}},\"top_k\":10}}\n")).collect::<String>();
        let qrels = (0..100).map(|index| format!("{{\"query_id\":\"q{index}\",\"chunk_external_id\":\"doc#1\",\"relevance\":1}}\n")).collect::<String>();
        let mut sidecar = (0..100).map(|index| format!("{{\"query_id\":\"q{index}\",\"model_name\":\"m\",\"dim\":2,\"vector\":[{index},0.1]}}\n")).collect::<String>();
        sidecar.push_str(
            "{\"query_id\":\"unused\",\"model_name\":\"m\",\"dim\":2,\"vector\":[1,0]}\n",
        );
        for inline in [false, true] {
            write_sidecar_fixture(&dir, &queries, &qrels, &sidecar);
            if inline {
                let queries = (0..100).map(|index| format!("{{\"query_id\":\"q{index}\",\"text\":\"wal\",\"filter\":{{\"chunk.n\":{index}}},\"top_k\":10,\"embedding\":[{index},0.1]}}\n")).collect::<String>();
                let hash = write_file(&dir, "queries.jsonl", &queries);
                let path = dir.join("dataset_manifest.json");
                let mut manifest = crate::parse_json(&fs::read_to_string(&path).unwrap()).unwrap();
                let obj = manifest.as_object_mut().unwrap();
                obj.insert(
                    "query_embedding_source".into(),
                    JsonValue::String("inline".into()),
                );
                obj.remove("query_embeddings");
                let hashes = obj.get_mut("hashes").unwrap().as_object_mut().unwrap();
                hashes.remove("query_embeddings_sha256");
                hashes.insert("queries_sha256".into(), JsonValue::String(hash));
                fs::write(&path, manifest.to_compact_string()).unwrap();
            }
            let context = JsonOperationContext::standalone();
            let (dataset, work) = BenchmarkDataset::load_and_validate_in(&dir, &context).unwrap();
            let expected_context = JsonOperationContext::standalone();
            let mut expected = dataset
                .manifest
                .chunking
                .reserve_retained_in(JsonInputKind::Manifest, &expected_context)
                .unwrap();
            for (index, query) in dataset.queries.iter().enumerate() {
                expected.merge(
                    query
                        .filter
                        .reserve_retained_in(JsonInputKind::Filter, &expected_context)
                        .unwrap(),
                );
                let embedding = query.embedding.as_ref().unwrap();
                expected.merge(embedding.reserve_json_work_in(&expected_context).unwrap());
                assert_eq!(embedding.vector, vec![index as f32, 0.1]);
                assert_eq!(embedding.model_name.as_deref(), Some("m"));
                assert_eq!(
                    query.filter.get("chunk.n").and_then(JsonValue::as_f64),
                    Some(index as f64)
                );
            }
            assert_eq!(work.reserved_bytes(), expected.reserved_bytes());
            drop(dataset);
            drop(work);
            context
                .reservation()
                .grow(
                    crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
                    JsonInputKind::BenchmarkArtifact,
                )
                .unwrap();
        }
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn benchmark_typed_vector_reserves_before_copy_and_preserves_f32_validation() {
        let dir = temp_dir("query-vector-conversion-budget");
        let path = dir.join("dataset_manifest.json");
        fs::write(&path, r#"{"query_embedding_source":"none"}"#).unwrap();
        let context = JsonOperationContext::standalone();
        let input = BenchmarkManifestInput::load(&path, &context).unwrap();
        let source = JsonValue::Array(vec![JsonValue::Number(1.0); 256]);
        let mut busy = context.reservation();
        let used = input.work.reserved_bytes();
        busy.grow(
            crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES - used - 1023,
            JsonInputKind::BenchmarkArtifact,
        )
        .unwrap();
        let mut conversion = context.reservation();
        let error = input
            .vector_values(&source, "q", 3, "queries", &mut conversion)
            .unwrap_err();
        assert_eq!(
            detail_value(&error, "json_limit_reason"),
            Some("work_budget_exceeded")
        );
        assert_eq!(
            detail_value(&error, "json_input_kind"),
            Some("query_embedding")
        );
        assert_eq!(conversion.reserved_bytes(), 0);
        drop(busy);
        for index in 0..100 {
            let source = JsonValue::Array(vec![
                JsonValue::Number(index as f64 / 10.0),
                JsonValue::Number(-0.0),
            ]);
            let mut conversion = context.reservation();
            let vector = input
                .vector_values(&source, "q", 3, "queries", &mut conversion)
                .unwrap();
            assert_eq!(vector[0].to_bits(), (index as f32 / 10.0).to_bits());
            assert_eq!(vector[1].to_bits(), (-0.0f32).to_bits());
            assert_eq!(conversion.reserved_bytes(), vector.capacity() * 4);
        }
        let mut conversion = context.reservation();
        let error = input
            .vector_values(
                &JsonValue::Array(vec![JsonValue::Number(f64::MAX)]),
                "q",
                3,
                "queries",
                &mut conversion,
            )
            .unwrap_err();
        assert_eq!(error.code(), ErrorCode::InvalidVectorLength);
        drop(conversion);
        drop(input);
        context
            .reservation()
            .grow(
                crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
                JsonInputKind::BenchmarkArtifact,
            )
            .unwrap();
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn sidecar_dataset_manifest_validates_hashes_and_query_embeddings() {
        let dir = temp_dir("sidecar-ok");
        let documents = write_file(
            &dir,
            "documents.jsonl",
            r#"{"external_id":"doc","metadata":{}}"#,
        );
        let chunks = write_file(
            &dir,
            "chunks.jsonl",
            r#"{"document_external_id":"doc","chunk_external_id":"doc#1","ordinal":1,"text":"wal","metadata":{}}"#,
        );
        let embeddings = write_file(
            &dir,
            "embeddings.jsonl",
            r#"{"chunk_external_id":"doc#1","model_name":"m","dim":2,"vector":[1.0,0.0]}"#,
        );
        let queries = write_file(
            &dir,
            "queries.jsonl",
            r#"{"query_id":"q1","text":"wal","top_k":10}"#,
        );
        let qrels = write_file(
            &dir,
            "qrels.jsonl",
            r#"{"query_id":"q1","chunk_external_id":"doc#1","relevance":2}"#,
        );
        let query_embeddings = write_file(
            &dir,
            "query_embeddings.jsonl",
            r#"{"query_id":"q1","model_name":"m","dim":2,"vector":[1.0,0.0]}"#,
        );
        fs::write(
            dir.join("dataset_manifest.json"),
            format!(
                concat!(
                    r#"{{"schema_version":"0.1","dataset_id":"bench","version":"0.1","#,
                    r#""chunking":{{"chunk_size":800,"chunk_overlap":120}},"#,
                    r#""documents":"documents.jsonl","chunks":"chunks.jsonl","#,
                    r#""embeddings":"embeddings.jsonl","queries":"queries.jsonl","#,
                    r#""qrels":"qrels.jsonl","query_embedding_source":"sidecar","#,
                    r#""query_embeddings":"query_embeddings.jsonl","embedding_model":"m","#,
                    r#""query_embedding_model":"m","query_embedding_dim":2,"hashes":{{"#,
                    r#""documents_sha256":"{}","chunks_sha256":"{}","#,
                    r#""embeddings_sha256":"{}","queries_sha256":"{}","#,
                    r#""qrels_sha256":"{}","query_embeddings_sha256":"{}"}}"#,
                    r#"}}"#
                ),
                documents, chunks, embeddings, queries, qrels, query_embeddings
            ),
        )
        .unwrap();

        let (dataset, _work, paths) =
            BenchmarkDataset::load_with_sources_in(&dir, &JsonOperationContext::standalone())
                .unwrap();
        assert_eq!(
            paths.into_iter().collect::<BTreeSet<_>>(),
            [
                "dataset_manifest.json",
                "documents.jsonl",
                "chunks.jsonl",
                "embeddings.jsonl",
                "queries.jsonl",
                "qrels.jsonl",
                "query_embeddings.jsonl",
            ]
            .map(|name| dir.join(name))
            .into_iter()
            .collect()
        );
        let manifest = dataset.manifest;
        assert_eq!(manifest.dataset_id, "bench");
        assert_eq!(
            manifest
                .chunking
                .get("chunk_size")
                .and_then(JsonValue::as_f64),
            Some(800.0)
        );
        assert_eq!(
            manifest.query_embedding_source,
            QueryEmbeddingSource::Sidecar
        );
        assert_eq!(manifest.query_embedding_dim, Some(2));
        assert_eq!(manifest.query_embeddings_sha256, Some(query_embeddings));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn sidecar_row_errors_report_the_actual_line_and_relative_artifact() {
        let dir = temp_dir("sidecar-error-locators");
        let rows = [
            (
                r#"{"query_id":"q1","model_name":"other","dim":2,"vector":[1,0]}"#,
                ErrorCode::EmbeddingModelMismatch,
            ),
            (
                r#"{"query_id":"q1","model_name":"m","dim":3,"vector":[1,0]}"#,
                ErrorCode::InvalidVectorLength,
            ),
            (
                r#"{"query_id":"q1","model_name":"m","dim":"private-dim","vector":[1,0]}"#,
                ErrorCode::MissingRequiredField,
            ),
            (
                concat!(
                    r#"{"query_id":"q1","model_name":"m","dim":2,"vector":[1,0]}"#,
                    "\n",
                    r#"{"query_id":"q1","model_name":"m","dim":2,"vector":[0,1]}"#
                ),
                ErrorCode::InvalidJsonlSchema,
            ),
        ];
        for seed in 0..100 {
            let (row, code) = rows[seed % rows.len()];
            let blank_lines = seed / rows.len();
            write_sidecar_fixture(
                &dir,
                r#"{"query_id":"q1","text":"private query text","top_k":10}"#,
                r#"{"query_id":"q1","chunk_external_id":"doc#1","relevance":2}"#,
                &format!("{}{row}\n", "\n".repeat(blank_lines)),
            );
            let error = BenchmarkDataset::load_and_validate(&dir).err().unwrap();
            assert_eq!(error.code(), code, "case {seed}");
            assert_eq!(detail_value(&error, "path"), Some("query_embeddings.jsonl"));
            assert_eq!(
                detail_value(&error, "artifact_kind"),
                Some("query_embeddings")
            );
            let expected_line = blank_lines + if seed % rows.len() == 3 { 2 } else { 1 };
            assert_eq!(
                detail_value(&error, "line"),
                Some(expected_line.to_string().as_str())
            );
            let wire = error.to_json().to_compact_string();
            assert!(!wire.contains("private-dim"));
            assert!(!wire.contains("private query text"));
            assert_eq!(detail_value(&error, "manifest_path"), dir.to_str());
        }
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn sidecar_rejects_duplicate_and_model_mismatch_rows() {
        let dir = temp_dir("sidecar-duplicate");
        let documents = write_file(
            &dir,
            "documents.jsonl",
            r#"{"external_id":"doc","metadata":{}}"#,
        );
        let chunks = write_file(
            &dir,
            "chunks.jsonl",
            r#"{"document_external_id":"doc","chunk_external_id":"doc#1","ordinal":1,"text":"wal","metadata":{}}"#,
        );
        let embeddings = write_file(
            &dir,
            "embeddings.jsonl",
            r#"{"chunk_external_id":"doc#1","model_name":"m","dim":2,"vector":[1.0,0.0]}"#,
        );
        let queries = write_file(
            &dir,
            "queries.jsonl",
            r#"{"query_id":"q1","text":"wal","top_k":10}"#,
        );
        let qrels = write_file(
            &dir,
            "qrels.jsonl",
            r#"{"query_id":"q1","chunk_external_id":"doc#1","relevance":2}"#,
        );
        let query_embeddings = write_file(
            &dir,
            "query_embeddings.jsonl",
            concat!(
                r#"{"query_id":"q1","model_name":"m","dim":2,"vector":[1.0,0.0]}"#,
                "\n",
                r#"{"query_id":"q1","model_name":"m","dim":2,"vector":[0.9,0.1]}"#
            ),
        );
        fs::write(
            dir.join("dataset_manifest.json"),
            format!(
                concat!(
                    r#"{{"schema_version":"0.1","dataset_id":"bench","version":"0.1","#,
                    r#""documents":"documents.jsonl","chunks":"chunks.jsonl","#,
                    r#""embeddings":"embeddings.jsonl","queries":"queries.jsonl","#,
                    r#""qrels":"qrels.jsonl","query_embedding_source":"sidecar","#,
                    r#""query_embeddings":"query_embeddings.jsonl","embedding_model":"m","#,
                    r#""query_embedding_model":"m","query_embedding_dim":2,"hashes":{{"#,
                    r#""documents_sha256":"{}","chunks_sha256":"{}","#,
                    r#""embeddings_sha256":"{}","queries_sha256":"{}","#,
                    r#""qrels_sha256":"{}","query_embeddings_sha256":"{}"}}"#,
                    r#"}}"#
                ),
                documents, chunks, embeddings, queries, qrels, query_embeddings
            ),
        )
        .unwrap();
        let err = BenchmarkDatasetManifest::load_and_validate(&dir).unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
        assert_eq!(detail_value(&err, "duplicate_line"), Some("2"));
        let _ = fs::remove_dir_all(dir);

        let dir = temp_dir("sidecar-model-mismatch");
        let documents = write_file(
            &dir,
            "documents.jsonl",
            r#"{"external_id":"doc","metadata":{}}"#,
        );
        let chunks = write_file(
            &dir,
            "chunks.jsonl",
            r#"{"document_external_id":"doc","chunk_external_id":"doc#1","ordinal":1,"text":"wal","metadata":{}}"#,
        );
        let embeddings = write_file(
            &dir,
            "embeddings.jsonl",
            r#"{"chunk_external_id":"doc#1","model_name":"m","dim":2,"vector":[1.0,0.0]}"#,
        );
        let queries = write_file(
            &dir,
            "queries.jsonl",
            r#"{"query_id":"q1","text":"wal","top_k":10}"#,
        );
        let qrels = write_file(
            &dir,
            "qrels.jsonl",
            r#"{"query_id":"q1","chunk_external_id":"doc#1","relevance":2}"#,
        );
        let query_embeddings = write_file(
            &dir,
            "query_embeddings.jsonl",
            r#"{"query_id":"q1","model_name":"other","dim":2,"vector":[1.0,0.0]}"#,
        );
        fs::write(
            dir.join("dataset_manifest.json"),
            format!(
                concat!(
                    r#"{{"schema_version":"0.1","dataset_id":"bench","version":"0.1","#,
                    r#""documents":"documents.jsonl","chunks":"chunks.jsonl","#,
                    r#""embeddings":"embeddings.jsonl","queries":"queries.jsonl","#,
                    r#""qrels":"qrels.jsonl","query_embedding_source":"sidecar","#,
                    r#""query_embeddings":"query_embeddings.jsonl","embedding_model":"m","#,
                    r#""query_embedding_model":"m","query_embedding_dim":2,"hashes":{{"#,
                    r#""documents_sha256":"{}","chunks_sha256":"{}","#,
                    r#""embeddings_sha256":"{}","queries_sha256":"{}","#,
                    r#""qrels_sha256":"{}","query_embeddings_sha256":"{}"}}"#,
                    r#"}}"#
                ),
                documents, chunks, embeddings, queries, qrels, query_embeddings
            ),
        )
        .unwrap();
        let err = BenchmarkDatasetManifest::load_and_validate(&dir).unwrap_err();
        assert_eq!(err.code(), ErrorCode::EmbeddingModelMismatch);
        assert_eq!(detail_value(&err, "row_model_name"), Some("other"));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn inline_mode_requires_model_and_valid_embedding_dim() {
        let dir = temp_dir("inline-missing-model");
        let documents = write_file(
            &dir,
            "documents.jsonl",
            r#"{"external_id":"doc","metadata":{}}"#,
        );
        let chunks = write_file(
            &dir,
            "chunks.jsonl",
            r#"{"document_external_id":"doc","chunk_external_id":"doc#1","ordinal":1,"text":"wal","metadata":{}}"#,
        );
        let embeddings = write_file(
            &dir,
            "embeddings.jsonl",
            r#"{"chunk_external_id":"doc#1","model_name":"m","dim":2,"vector":[1.0,0.0]}"#,
        );
        let queries = write_file(
            &dir,
            "queries.jsonl",
            r#"{"query_id":"q1","text":"wal","top_k":10,"embedding":[1.0,0.0]}"#,
        );
        let qrels = write_file(
            &dir,
            "qrels.jsonl",
            r#"{"query_id":"q1","chunk_external_id":"doc#1","relevance":2}"#,
        );
        fs::write(
            dir.join("dataset_manifest.json"),
            format!(
                concat!(
                    r#"{{"schema_version":"0.1","dataset_id":"bench","version":"0.1","#,
                    r#""documents":"documents.jsonl","chunks":"chunks.jsonl","#,
                    r#""embeddings":"embeddings.jsonl","queries":"queries.jsonl","#,
                    r#""qrels":"qrels.jsonl","query_embedding_source":"inline","#,
                    r#""embedding_model":"m","query_embedding_dim":2,"hashes":{{"#,
                    r#""documents_sha256":"{}","chunks_sha256":"{}","#,
                    r#""embeddings_sha256":"{}","queries_sha256":"{}","qrels_sha256":"{}"}}"#,
                    r#"}}"#
                ),
                documents, chunks, embeddings, queries, qrels
            ),
        )
        .unwrap();
        let err = BenchmarkDatasetManifest::load_and_validate(&dir).unwrap_err();
        assert_eq!(err.code(), ErrorCode::MissingRequiredField);
        assert_eq!(detail_value(&err, "field"), Some("query_embedding_model"));
        let _ = fs::remove_dir_all(dir);

        let dir = temp_dir("inline-bad-dim");
        let documents = write_file(
            &dir,
            "documents.jsonl",
            r#"{"external_id":"doc","metadata":{}}"#,
        );
        let chunks = write_file(
            &dir,
            "chunks.jsonl",
            r#"{"document_external_id":"doc","chunk_external_id":"doc#1","ordinal":1,"text":"wal","metadata":{}}"#,
        );
        let embeddings = write_file(
            &dir,
            "embeddings.jsonl",
            r#"{"chunk_external_id":"doc#1","model_name":"m","dim":2,"vector":[1.0,0.0]}"#,
        );
        let queries = write_file(
            &dir,
            "queries.jsonl",
            r#"{"query_id":"q1","text":"wal","top_k":10,"embedding":[1.0]}"#,
        );
        let qrels = write_file(
            &dir,
            "qrels.jsonl",
            r#"{"query_id":"q1","chunk_external_id":"doc#1","relevance":2}"#,
        );
        fs::write(
            dir.join("dataset_manifest.json"),
            format!(
                concat!(
                    r#"{{"schema_version":"0.1","dataset_id":"bench","version":"0.1","#,
                    r#""documents":"documents.jsonl","chunks":"chunks.jsonl","#,
                    r#""embeddings":"embeddings.jsonl","queries":"queries.jsonl","#,
                    r#""qrels":"qrels.jsonl","query_embedding_source":"inline","#,
                    r#""embedding_model":"m","query_embedding_model":"m","#,
                    r#""query_embedding_dim":2,"hashes":{{"#,
                    r#""documents_sha256":"{}","chunks_sha256":"{}","#,
                    r#""embeddings_sha256":"{}","queries_sha256":"{}","qrels_sha256":"{}"}}"#,
                    r#"}}"#
                ),
                documents, chunks, embeddings, queries, qrels
            ),
        )
        .unwrap();
        let err = BenchmarkDatasetManifest::load_and_validate(&dir).unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidVectorLength);
        assert_eq!(detail_value(&err, "actual_vector_length"), Some("1"));
        assert_eq!(detail_value(&err, "path"), Some("queries.jsonl"));
        assert_eq!(detail_value(&err, "artifact_kind"), Some("queries"));
        assert_eq!(detail_value(&err, "line"), Some("1"));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn query_embedding_dim_must_be_positive() {
        let dir = temp_dir("query-embedding-dim-zero");
        write_sidecar_fixture(
            &dir,
            r#"{"query_id":"q1","text":"wal","top_k":10}"#,
            r#"{"query_id":"q1","chunk_external_id":"doc#1","relevance":2}"#,
            r#"{"query_id":"q1","model_name":"m","dim":2,"vector":[1.0,0.0]}"#,
        );
        let manifest_path = dir.join("dataset_manifest.json");
        let manifest = fs::read_to_string(&manifest_path)
            .unwrap()
            .replace(r#""query_embedding_dim":2"#, r#""query_embedding_dim":0"#);
        fs::write(&manifest_path, manifest).unwrap();

        let err = BenchmarkDatasetManifest::load_and_validate(&dir).unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidVectorLength);
        assert_eq!(detail_value(&err, "field"), Some("query_embedding_dim"));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn sidecar_requires_query_embedding_for_each_query() {
        let dir = temp_dir("sidecar-missing-query-id");
        write_sidecar_fixture(
            &dir,
            concat!(
                r#"{"query_id":"q1","text":"wal","top_k":10}"#,
                "\n",
                r#"{"query_id":"q2","text":"log","top_k":10}"#
            ),
            concat!(
                r#"{"query_id":"q1","chunk_external_id":"doc#1","relevance":2}"#,
                "\n",
                r#"{"query_id":"q2","chunk_external_id":"doc#1","relevance":1}"#
            ),
            r#"{"query_id":"q1","model_name":"m","dim":2,"vector":[1.0,0.0]}"#,
        );

        let err = BenchmarkDatasetManifest::load_and_validate(&dir).unwrap_err();
        assert_eq!(err.code(), ErrorCode::MissingRequiredField);
        assert_eq!(detail_value(&err, "query_id"), Some("q2"));
        assert_eq!(
            detail_value(&err, "field"),
            Some("query_embeddings[query_id]")
        );
        assert_eq!(detail_value(&err, "path"), Some("query_embeddings.jsonl"));
        assert_eq!(
            detail_value(&err, "artifact_kind"),
            Some("query_embeddings")
        );
        assert_eq!(detail_value(&err, "line"), None);
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn sidecar_rejects_query_embedding_dim_mismatch() {
        let dir = temp_dir("sidecar-dim-mismatch");
        write_sidecar_fixture(
            &dir,
            r#"{"query_id":"q1","text":"wal","top_k":10}"#,
            r#"{"query_id":"q1","chunk_external_id":"doc#1","relevance":2}"#,
            r#"{"query_id":"q1","model_name":"m","dim":3,"vector":[1.0,0.0]}"#,
        );

        let err = BenchmarkDatasetManifest::load_and_validate(&dir).unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidVectorLength);
        assert_eq!(detail_value(&err, "query_id"), Some("q1"));
        assert_eq!(detail_value(&err, "expected_dim"), Some("2"));
        assert_eq!(detail_value(&err, "row_dim"), Some("3"));
        assert_eq!(detail_value(&err, "actual_vector_length"), Some("2"));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn inline_dataset_manifest_loads_inline_query_embeddings() {
        let dir = temp_dir("inline-ok");
        let documents = write_file(
            &dir,
            "documents.jsonl",
            r#"{"external_id":"doc","metadata":{}}"#,
        );
        let chunks = write_file(
            &dir,
            "chunks.jsonl",
            r#"{"document_external_id":"doc","chunk_external_id":"doc#1","ordinal":1,"text":"wal","metadata":{}}"#,
        );
        let embeddings = write_file(
            &dir,
            "embeddings.jsonl",
            r#"{"chunk_external_id":"doc#1","model_name":"m","dim":2,"vector":[1.0,0.0]}"#,
        );
        let queries = write_file(
            &dir,
            "queries.jsonl",
            r#"{"query_id":"q1","text":"wal","top_k":10,"embedding":[1.0,0.0]}"#,
        );
        let qrels = write_file(
            &dir,
            "qrels.jsonl",
            r#"{"query_id":"q1","chunk_external_id":"doc#1","relevance":2}"#,
        );
        fs::write(
            dir.join("dataset_manifest.json"),
            format!(
                concat!(
                    r#"{{"schema_version":"0.1","dataset_id":"bench","version":"0.1","#,
                    r#""documents":"documents.jsonl","chunks":"chunks.jsonl","#,
                    r#""embeddings":"embeddings.jsonl","queries":"queries.jsonl","#,
                    r#""qrels":"qrels.jsonl","query_embedding_source":"inline","#,
                    r#""embedding_model":"m","query_embedding_model":"m","#,
                    r#""query_embedding_dim":2,"hashes":{{"#,
                    r#""documents_sha256":"{}","chunks_sha256":"{}","#,
                    r#""embeddings_sha256":"{}","queries_sha256":"{}","qrels_sha256":"{}"}}"#,
                    r#"}}"#
                ),
                documents, chunks, embeddings, queries, qrels
            ),
        )
        .unwrap();

        let dataset = BenchmarkDataset::load_and_validate(&dir).unwrap();
        assert_eq!(
            dataset.manifest.query_embedding_source,
            QueryEmbeddingSource::Inline
        );
        assert_eq!(dataset.manifest.query_embeddings_sha256, None);
        assert_eq!(dataset.queries.len(), 1);
        assert_eq!(dataset.queries[0].embedding.as_ref().unwrap().dim, 2);
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn none_mode_rejects_query_embedding_fields() {
        let dir = temp_dir("none-conflict");
        let documents = write_file(
            &dir,
            "documents.jsonl",
            r#"{"external_id":"doc","metadata":{}}"#,
        );
        let chunks = write_file(
            &dir,
            "chunks.jsonl",
            r#"{"document_external_id":"doc","chunk_external_id":"doc#1","ordinal":1,"text":"wal","metadata":{}}"#,
        );
        let queries = write_file(
            &dir,
            "queries.jsonl",
            r#"{"query_id":"q1","text":"wal","top_k":10}"#,
        );
        let qrels = write_file(
            &dir,
            "qrels.jsonl",
            r#"{"query_id":"q1","chunk_external_id":"doc#1","relevance":2}"#,
        );
        fs::write(
            dir.join("dataset_manifest.json"),
            format!(
                concat!(
                    r#"{{"schema_version":"0.1","dataset_id":"bench","version":"0.1","#,
                    r#""documents":"documents.jsonl","chunks":"chunks.jsonl","#,
                    r#""queries":"queries.jsonl","qrels":"qrels.jsonl","#,
                    r#""query_embedding_source":"none","query_embedding_model":"m","#,
                    r#""hashes":{{"documents_sha256":"{}","chunks_sha256":"{}","#,
                    r#""queries_sha256":"{}","qrels_sha256":"{}"}}"#,
                    r#"}}"#
                ),
                documents, chunks, queries, qrels
            ),
        )
        .unwrap();

        let err = BenchmarkDatasetManifest::load_and_validate(&dir).unwrap_err();
        assert_eq!(err.code(), ErrorCode::BenchmarkArtifactModeConflict);
        assert_eq!(detail_value(&err, "query_embedding_source"), Some("none"));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn none_mode_rejects_embedding_artifacts() {
        let dir = temp_dir("none-embedding-conflict");
        let documents = write_file(
            &dir,
            "documents.jsonl",
            r#"{"external_id":"doc","metadata":{}}"#,
        );
        let chunks = write_file(
            &dir,
            "chunks.jsonl",
            r#"{"document_external_id":"doc","chunk_external_id":"doc#1","ordinal":1,"text":"wal","metadata":{}}"#,
        );
        let embeddings = write_file(
            &dir,
            "embeddings.jsonl",
            r#"{"chunk_external_id":"doc#1","model_name":"m","dim":2,"vector":[1.0,0.0]}"#,
        );
        let queries = write_file(
            &dir,
            "queries.jsonl",
            r#"{"query_id":"q1","text":"wal","top_k":10}"#,
        );
        let qrels = write_file(
            &dir,
            "qrels.jsonl",
            r#"{"query_id":"q1","chunk_external_id":"doc#1","relevance":2}"#,
        );
        fs::write(
            dir.join("dataset_manifest.json"),
            format!(
                concat!(
                    r#"{{"schema_version":"0.1","dataset_id":"bench","version":"0.1","#,
                    r#""documents":"documents.jsonl","chunks":"chunks.jsonl","#,
                    r#""embeddings":"embeddings.jsonl","queries":"queries.jsonl","#,
                    r#""qrels":"qrels.jsonl","query_embedding_source":"none","hashes":{{"#,
                    r#""documents_sha256":"{}","chunks_sha256":"{}","#,
                    r#""embeddings_sha256":"{}","queries_sha256":"{}","qrels_sha256":"{}"}}"#,
                    r#"}}"#
                ),
                documents, chunks, embeddings, queries, qrels
            ),
        )
        .unwrap();

        let err = BenchmarkDatasetManifest::load_and_validate(&dir).unwrap_err();
        assert_eq!(err.code(), ErrorCode::BenchmarkArtifactModeConflict);
        assert_eq!(detail_value(&err, "query_embedding_source"), Some("none"));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn sidecar_requires_embeddings_hash() {
        let dir = temp_dir("sidecar-missing-embedding-hash");
        let documents = write_file(
            &dir,
            "documents.jsonl",
            r#"{"external_id":"doc","metadata":{}}"#,
        );
        let chunks = write_file(
            &dir,
            "chunks.jsonl",
            r#"{"document_external_id":"doc","chunk_external_id":"doc#1","ordinal":1,"text":"wal","metadata":{}}"#,
        );
        write_file(
            &dir,
            "embeddings.jsonl",
            r#"{"chunk_external_id":"doc#1","model_name":"m","dim":2,"vector":[1.0,0.0]}"#,
        );
        let queries = write_file(
            &dir,
            "queries.jsonl",
            r#"{"query_id":"q1","text":"wal","top_k":10}"#,
        );
        let qrels = write_file(
            &dir,
            "qrels.jsonl",
            r#"{"query_id":"q1","chunk_external_id":"doc#1","relevance":2}"#,
        );
        let query_embeddings = write_file(
            &dir,
            "query_embeddings.jsonl",
            r#"{"query_id":"q1","model_name":"m","dim":2,"vector":[1.0,0.0]}"#,
        );
        fs::write(
            dir.join("dataset_manifest.json"),
            format!(
                concat!(
                    r#"{{"schema_version":"0.1","dataset_id":"bench","version":"0.1","#,
                    r#""documents":"documents.jsonl","chunks":"chunks.jsonl","#,
                    r#""embeddings":"embeddings.jsonl","queries":"queries.jsonl","#,
                    r#""qrels":"qrels.jsonl","query_embedding_source":"sidecar","#,
                    r#""query_embeddings":"query_embeddings.jsonl","embedding_model":"m","#,
                    r#""query_embedding_model":"m","query_embedding_dim":2,"hashes":{{"#,
                    r#""documents_sha256":"{}","chunks_sha256":"{}","#,
                    r#""queries_sha256":"{}","qrels_sha256":"{}","#,
                    r#""query_embeddings_sha256":"{}"}}"#,
                    r#"}}"#
                ),
                documents, chunks, queries, qrels, query_embeddings
            ),
        )
        .unwrap();

        let err = BenchmarkDatasetManifest::load_and_validate(&dir).unwrap_err();
        assert_eq!(err.code(), ErrorCode::MissingRequiredField);
        assert_eq!(
            detail_value(&err, "field"),
            Some("hashes.embeddings_sha256")
        );
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn benchmark_embeddings_require_manifest_model() {
        let dir = temp_dir("embeddings-missing-model");
        write_sidecar_fixture_with_chunks_and_embeddings(
            &dir,
            r#"{"document_external_id":"doc","chunk_external_id":"doc#1","ordinal":1,"text":"wal","metadata":{}}"#,
            r#"{"chunk_external_id":"doc#1","model_name":"m","dim":2,"vector":[1.0,0.0]}"#,
        );
        let documents = sha256_hex(&fs::read(dir.join("documents.jsonl")).unwrap());
        let chunks = sha256_hex(&fs::read(dir.join("chunks.jsonl")).unwrap());
        let embeddings = sha256_hex(&fs::read(dir.join("embeddings.jsonl")).unwrap());
        let queries = sha256_hex(&fs::read(dir.join("queries.jsonl")).unwrap());
        let qrels = sha256_hex(&fs::read(dir.join("qrels.jsonl")).unwrap());
        let query_embeddings = sha256_hex(&fs::read(dir.join("query_embeddings.jsonl")).unwrap());
        fs::write(
            dir.join("dataset_manifest.json"),
            format!(
                concat!(
                    r#"{{"schema_version":"0.1","dataset_id":"bench","version":"0.1","#,
                    r#""documents":"documents.jsonl","chunks":"chunks.jsonl","#,
                    r#""embeddings":"embeddings.jsonl","queries":"queries.jsonl","#,
                    r#""qrels":"qrels.jsonl","query_embedding_source":"sidecar","#,
                    r#""query_embeddings":"query_embeddings.jsonl","#,
                    r#""query_embedding_model":"m","query_embedding_dim":2,"hashes":{{"#,
                    r#""documents_sha256":"{}","chunks_sha256":"{}","#,
                    r#""embeddings_sha256":"{}","queries_sha256":"{}","#,
                    r#""qrels_sha256":"{}","query_embeddings_sha256":"{}"}}"#,
                    r#"}}"#
                ),
                documents, chunks, embeddings, queries, qrels, query_embeddings
            ),
        )
        .unwrap();

        let err = BenchmarkDatasetManifest::load_and_validate(&dir).unwrap_err();
        assert_eq!(err.code(), ErrorCode::MissingRequiredField);
        assert_eq!(detail_value(&err, "field"), Some("embedding_model"));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn benchmark_embeddings_must_reference_declared_chunks() {
        let dir = temp_dir("embeddings-unknown-chunk");
        write_sidecar_fixture_with_chunks_and_embeddings(
            &dir,
            r#"{"document_external_id":"doc","chunk_external_id":"doc#1","ordinal":1,"text":"wal","metadata":{}}"#,
            r#"{"chunk_external_id":"missing#1","model_name":"m","dim":2,"vector":[1.0,0.0]}"#,
        );

        let err = BenchmarkDatasetManifest::load_and_validate(&dir).unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
        assert_eq!(detail_value(&err, "chunk_external_id"), Some("missing#1"));
        assert_eq!(
            detail_value(&err, "field"),
            Some("embeddings.chunk_external_id")
        );
        assert_eq!(detail_value(&err, "source"), Some("embeddings.jsonl"));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn benchmark_embeddings_reject_duplicate_chunk_rows() {
        let dir = temp_dir("embeddings-duplicate-chunk");
        write_sidecar_fixture_with_chunks_and_embeddings(
            &dir,
            r#"{"document_external_id":"doc","chunk_external_id":"doc#1","ordinal":1,"text":"wal","metadata":{}}"#,
            concat!(
                r#"{"chunk_external_id":"doc#1","model_name":"m","dim":2,"vector":[1.0,0.0]}"#,
                "\n",
                r#"{"chunk_external_id":"doc#1","model_name":"m","dim":2,"vector":[0.0,1.0]}"#
            ),
        );

        let err = BenchmarkDatasetManifest::load_and_validate(&dir).unwrap_err();
        assert_eq!(err.code(), ErrorCode::DuplicateExternalId);
        assert_eq!(detail_value(&err, "chunk_external_id"), Some("doc#1"));
        assert_eq!(detail_value(&err, "first_line"), Some("1"));
        assert_eq!(detail_value(&err, "duplicate_line"), Some("2"));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn benchmark_embeddings_validate_model_dim_and_vector() {
        let dir = temp_dir("embeddings-model-mismatch");
        write_sidecar_fixture_with_chunks_and_embeddings(
            &dir,
            r#"{"document_external_id":"doc","chunk_external_id":"doc#1","ordinal":1,"text":"wal","metadata":{}}"#,
            r#"{"chunk_external_id":"doc#1","model_name":"other","dim":2,"vector":[1.0,0.0]}"#,
        );

        let err = BenchmarkDatasetManifest::load_and_validate(&dir).unwrap_err();
        assert_eq!(err.code(), ErrorCode::EmbeddingModelMismatch);
        assert_eq!(detail_value(&err, "chunk_external_id"), Some("doc#1"));
        assert_eq!(detail_value(&err, "manifest_embedding_model"), Some("m"));
        assert_eq!(detail_value(&err, "row_model_name"), Some("other"));
        let _ = fs::remove_dir_all(dir);

        let dir = temp_dir("embeddings-dim-mismatch");
        write_sidecar_fixture_with_chunks_and_embeddings(
            &dir,
            concat!(
                r#"{"document_external_id":"doc","chunk_external_id":"doc#1","ordinal":1,"text":"wal","metadata":{}}"#,
                "\n",
                r#"{"document_external_id":"doc","chunk_external_id":"doc#2","ordinal":2,"text":"checkpoint","metadata":{}}"#
            ),
            concat!(
                r#"{"chunk_external_id":"doc#1","model_name":"m","dim":2,"vector":[1.0,0.0]}"#,
                "\n",
                r#"{"chunk_external_id":"doc#2","model_name":"m","dim":3,"vector":[0.0,1.0]}"#
            ),
        );

        let err = BenchmarkDatasetManifest::load_and_validate(&dir).unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidVectorLength);
        assert_eq!(detail_value(&err, "chunk_external_id"), Some("doc#2"));
        assert_eq!(detail_value(&err, "expected_dim"), Some("2"));
        assert_eq!(detail_value(&err, "row_dim"), Some("3"));
        assert_eq!(detail_value(&err, "actual_vector_length"), Some("2"));
        let _ = fs::remove_dir_all(dir);

        let dir = temp_dir("embeddings-vector-mismatch");
        write_sidecar_fixture_with_chunks_and_embeddings(
            &dir,
            r#"{"document_external_id":"doc","chunk_external_id":"doc#1","ordinal":1,"text":"wal","metadata":{}}"#,
            r#"{"chunk_external_id":"doc#1","model_name":"m","dim":2,"vector":[1.0]}"#,
        );

        let err = BenchmarkDatasetManifest::load_and_validate(&dir).unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidVectorLength);
        assert_eq!(detail_value(&err, "chunk_external_id"), Some("doc#1"));
        assert_eq!(detail_value(&err, "expected_dim"), Some("2"));
        assert_eq!(detail_value(&err, "row_dim"), Some("2"));
        assert_eq!(detail_value(&err, "actual_vector_length"), Some("1"));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn benchmark_embeddings_require_complete_chunk_coverage() {
        let dir = temp_dir("embeddings-missing-chunk-row");
        write_sidecar_fixture_with_chunks_and_embeddings(
            &dir,
            concat!(
                r#"{"document_external_id":"doc","chunk_external_id":"doc#1","ordinal":1,"text":"wal","metadata":{}}"#,
                "\n",
                r#"{"document_external_id":"doc","chunk_external_id":"doc#2","ordinal":2,"text":"checkpoint","metadata":{}}"#
            ),
            r#"{"chunk_external_id":"doc#1","model_name":"m","dim":2,"vector":[1.0,0.0]}"#,
        );

        let err = BenchmarkDatasetManifest::load_and_validate(&dir).unwrap_err();
        assert_eq!(err.code(), ErrorCode::MissingRequiredField);
        assert_eq!(detail_value(&err, "chunk_external_id"), Some("doc#2"));
        assert_eq!(
            detail_value(&err, "field"),
            Some("embeddings[chunk_external_id]")
        );
        assert_eq!(detail_value(&err, "source"), Some("embeddings.jsonl"));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn hash_mismatch_reports_expected_and_actual_without_payload() {
        let dir = temp_dir("hash-mismatch");
        let documents = write_file(
            &dir,
            "documents.jsonl",
            r#"{"external_id":"doc","metadata":{}}"#,
        );
        let chunks = write_file(
            &dir,
            "chunks.jsonl",
            r#"{"document_external_id":"doc","chunk_external_id":"doc#1","ordinal":1,"text":"wal","metadata":{}}"#,
        );
        let queries = write_file(
            &dir,
            "queries.jsonl",
            r#"{"query_id":"q1","text":"wal","top_k":10}"#,
        );
        let qrels = write_file(
            &dir,
            "qrels.jsonl",
            r#"{"query_id":"q1","chunk_external_id":"doc#1","relevance":2}"#,
        );
        fs::write(
            dir.join("dataset_manifest.json"),
            format!(
                concat!(
                    r#"{{"schema_version":"0.1","dataset_id":"bench","version":"0.1","#,
                    r#""documents":"documents.jsonl","chunks":"chunks.jsonl","#,
                    r#""queries":"queries.jsonl","qrels":"qrels.jsonl","#,
                    r#""query_embedding_source":"none","hashes":{{"#,
                    r#""documents_sha256":"{}","chunks_sha256":"{}","#,
                    r#""queries_sha256":"0000","qrels_sha256":"{}"}}"#,
                    r#"}}"#
                ),
                documents, chunks, qrels
            ),
        )
        .unwrap();

        let err = BenchmarkDatasetManifest::load_and_validate(&dir).unwrap_err();
        assert_eq!(err.code(), ErrorCode::ChecksumMismatch);
        assert_eq!(detail_value(&err, "artifact_kind"), Some("queries"));
        assert_eq!(detail_value(&err, "expected_sha256"), Some("0000"));
        assert_eq!(detail_value(&err, "actual_sha256"), Some(queries.as_str()));
        assert!(err.to_string().contains("hash mismatch"));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn jsonl_parse_errors_report_artifact_locator_without_payload() {
        let dir = temp_dir("query-malformed-json");
        write_none_fixture(
            &dir,
            r#"{"query_id":"q1","text":"do not echo private query text","top_k":10,"embedding":[1.0,"#,
            r#"{"query_id":"q1","chunk_external_id":"doc#1","relevance":2}"#,
        );

        let err = BenchmarkDatasetManifest::load_and_validate(&dir).unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
        assert_eq!(detail_value(&err, "path"), Some("queries.jsonl"));
        assert_eq!(detail_value(&err, "artifact_kind"), Some("queries"));
        assert_eq!(detail_value(&err, "line"), Some("1"));
        let manifest_label = dir.display().to_string();
        assert_eq!(
            detail_value(&err, "manifest_path"),
            Some(manifest_label.as_str())
        );
        let public_json = err.to_json().to_compact_string();
        assert!(!public_json.contains("do not echo private query text"));
        assert!(!public_json.contains("embedding"));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn benchmark_top_k_checks_original_decimal_math_and_target_range() {
        let dir = temp_dir("query-exact-top-k");
        let maximum = crate::query::MAX_SEARCH_TOP_K;
        let max_token = maximum.to_string();
        let accepted = vec![
            ("1".to_owned(), 1),
            ("10.0".to_owned(), 10),
            ("1e1".to_owned(), 10),
            ("0.1e2".to_owned(), 10),
            ("1000e-2".to_owned(), 10),
            ((maximum - 1).to_string(), maximum - 1),
            (max_token.clone(), maximum),
            (format!("{max_token}000e-3"), maximum),
        ];
        for (raw, expected) in accepted {
            let query = format!(r#"{{"query_id":"q1","text":"wal","top_k":{raw}}}"#);
            write_none_fixture(
                &dir,
                &query,
                r#"{"query_id":"q1","chunk_external_id":"doc#1","relevance":1}"#,
            );
            let original = fs::read(dir.join("queries.jsonl")).unwrap();
            let dataset = BenchmarkDataset::load_and_validate(&dir).unwrap();
            assert_eq!(dataset.queries[0].top_k, expected, "{raw}");
            assert_eq!(fs::read(dir.join("queries.jsonl")).unwrap(), original);
        }
        for raw in [
            "0",
            "-1",
            "1.5",
            "1.0000000000000001",
            "1.0000000000000000000000000000000001",
            "9007199254740991.1",
            "9007199254740992",
            "9007199254740993",
            "9.007199254740992e15",
            "1e9999999999999999999999999",
            "1e-9999999999999999999999999",
            "true",
            "[]",
            "[10]",
            "{}",
            "{\"top_k\":10}",
            "\"10\"",
        ] {
            let query = format!(r#"{{"query_id":"q1","text":"wal","top_k":{raw}}}"#);
            write_none_fixture(
                &dir,
                &query,
                r#"{"query_id":"q1","chunk_external_id":"doc#1","relevance":1}"#,
            );
            let original = fs::read(dir.join("queries.jsonl")).unwrap();
            let error = BenchmarkDatasetManifest::load_and_validate(&dir).unwrap_err();
            assert_eq!(error.code(), ErrorCode::InvalidJsonlSchema, "{raw}");
            assert_eq!(detail_value(&error, "artifact_kind"), Some("queries"));
            assert_eq!(detail_value(&error, "line"), Some("1"));
            assert_eq!(fs::read(dir.join("queries.jsonl")).unwrap(), original);
        }
        for field in ["", ",\"top_k\":null"] {
            let query = format!(r#"{{"query_id":"q1","text":"wal"{field}}}"#);
            write_none_fixture(
                &dir,
                &query,
                r#"{"query_id":"q1","chunk_external_id":"doc#1","relevance":1}"#,
            );
            let error = BenchmarkDatasetManifest::load_and_validate(&dir).unwrap_err();
            assert_eq!(error.code(), ErrorCode::MissingRequiredField);
            assert_eq!(detail_value(&error, "field"), Some("queries.top_k"));
        }
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn benchmark_queries_require_text_and_top_k() {
        let dir = temp_dir("query-missing-text");
        let documents = write_file(
            &dir,
            "documents.jsonl",
            r#"{"external_id":"doc","metadata":{}}"#,
        );
        let chunks = write_file(
            &dir,
            "chunks.jsonl",
            r#"{"document_external_id":"doc","chunk_external_id":"doc#1","ordinal":1,"text":"wal","metadata":{}}"#,
        );
        let queries = write_file(&dir, "queries.jsonl", r#"{"query_id":"q1","top_k":10}"#);
        let qrels = write_file(
            &dir,
            "qrels.jsonl",
            r#"{"query_id":"q1","chunk_external_id":"doc#1","relevance":2}"#,
        );
        fs::write(
            dir.join("dataset_manifest.json"),
            format!(
                concat!(
                    r#"{{"schema_version":"0.1","dataset_id":"bench","version":"0.1","#,
                    r#""documents":"documents.jsonl","chunks":"chunks.jsonl","#,
                    r#""queries":"queries.jsonl","qrels":"qrels.jsonl","#,
                    r#""query_embedding_source":"none","hashes":{{"#,
                    r#""documents_sha256":"{}","chunks_sha256":"{}","#,
                    r#""queries_sha256":"{}","qrels_sha256":"{}"}}"#,
                    r#"}}"#
                ),
                documents, chunks, queries, qrels
            ),
        )
        .unwrap();

        let err = BenchmarkDatasetManifest::load_and_validate(&dir).unwrap_err();
        assert_eq!(err.code(), ErrorCode::MissingRequiredField);
        assert_eq!(detail_value(&err, "query_id"), Some("q1"));
        assert_eq!(detail_value(&err, "field"), Some("queries.text"));
        assert_eq!(detail_value(&err, "artifact_kind"), Some("queries"));
        let _ = fs::remove_dir_all(dir);

        let dir = temp_dir("query-missing-top-k");
        let documents = write_file(
            &dir,
            "documents.jsonl",
            r#"{"external_id":"doc","metadata":{}}"#,
        );
        let chunks = write_file(
            &dir,
            "chunks.jsonl",
            r#"{"document_external_id":"doc","chunk_external_id":"doc#1","ordinal":1,"text":"wal","metadata":{}}"#,
        );
        let queries = write_file(&dir, "queries.jsonl", r#"{"query_id":"q1","text":"wal"}"#);
        let qrels = write_file(
            &dir,
            "qrels.jsonl",
            r#"{"query_id":"q1","chunk_external_id":"doc#1","relevance":2}"#,
        );
        fs::write(
            dir.join("dataset_manifest.json"),
            format!(
                concat!(
                    r#"{{"schema_version":"0.1","dataset_id":"bench","version":"0.1","#,
                    r#""documents":"documents.jsonl","chunks":"chunks.jsonl","#,
                    r#""queries":"queries.jsonl","qrels":"qrels.jsonl","#,
                    r#""query_embedding_source":"none","hashes":{{"#,
                    r#""documents_sha256":"{}","chunks_sha256":"{}","#,
                    r#""queries_sha256":"{}","qrels_sha256":"{}"}}"#,
                    r#"}}"#
                ),
                documents, chunks, queries, qrels
            ),
        )
        .unwrap();

        let err = BenchmarkDatasetManifest::load_and_validate(&dir).unwrap_err();
        assert_eq!(err.code(), ErrorCode::MissingRequiredField);
        assert_eq!(detail_value(&err, "query_id"), Some("q1"));
        assert_eq!(detail_value(&err, "field"), Some("queries.top_k"));
        assert_eq!(detail_value(&err, "artifact_kind"), Some("queries"));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn benchmark_artifacts_reject_blank_identifiers_and_models() {
        let dir = temp_dir("blank-query-id");
        write_none_fixture(
            &dir,
            r#"{"query_id":"","text":"wal","top_k":10}"#,
            r#"{"query_id":"","chunk_external_id":"doc#1","relevance":2}"#,
        );

        let err = BenchmarkDatasetManifest::load_and_validate(&dir).unwrap_err();
        assert_eq!(err.code(), ErrorCode::MissingRequiredField);
        assert_eq!(detail_value(&err, "field"), Some("queries.query_id"));
        let _ = fs::remove_dir_all(dir);

        let dir = temp_dir("blank-document-id");
        let documents = write_file(
            &dir,
            "documents.jsonl",
            r#"{"external_id":"","metadata":{}}"#,
        );
        let chunks = write_file(
            &dir,
            "chunks.jsonl",
            r#"{"document_external_id":"","chunk_external_id":"doc#1","ordinal":1,"text":"wal","metadata":{}}"#,
        );
        let queries = write_file(
            &dir,
            "queries.jsonl",
            r#"{"query_id":"q1","text":"wal","top_k":10}"#,
        );
        let qrels = write_file(
            &dir,
            "qrels.jsonl",
            r#"{"query_id":"q1","chunk_external_id":"doc#1","relevance":2}"#,
        );
        fs::write(
            dir.join("dataset_manifest.json"),
            format!(
                concat!(
                    r#"{{"schema_version":"0.1","dataset_id":"bench","version":"0.1","#,
                    r#""documents":"documents.jsonl","chunks":"chunks.jsonl","#,
                    r#""queries":"queries.jsonl","qrels":"qrels.jsonl","#,
                    r#""query_embedding_source":"none","hashes":{{"#,
                    r#""documents_sha256":"{}","chunks_sha256":"{}","#,
                    r#""queries_sha256":"{}","qrels_sha256":"{}"}}"#,
                    r#"}}"#
                ),
                documents, chunks, queries, qrels
            ),
        )
        .unwrap();

        let err = BenchmarkDatasetManifest::load_and_validate(&dir).unwrap_err();
        assert_eq!(err.code(), ErrorCode::MissingRequiredField);
        assert_eq!(detail_value(&err, "field"), Some("documents.external_id"));
        let _ = fs::remove_dir_all(dir);

        let dir = temp_dir("blank-embedding-model");
        write_sidecar_fixture_with_chunks_and_embeddings(
            &dir,
            r#"{"document_external_id":"doc","chunk_external_id":"doc#1","ordinal":1,"text":"wal","metadata":{}}"#,
            r#"{"chunk_external_id":"doc#1","model_name":"","dim":2,"vector":[1.0,0.0]}"#,
        );

        let err = BenchmarkDatasetManifest::load_and_validate(&dir).unwrap_err();
        assert_eq!(err.code(), ErrorCode::MissingRequiredField);
        assert_eq!(detail_value(&err, "field"), Some("embeddings.model_name"));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn unknown_artifact_keys_are_not_reflected_as_public_schema_locations() {
        for seed in 0..100 {
            let secret = format!("private query or metadata value {seed}");
            let obj = [(secret.clone(), JsonValue::Null)]
                .into_iter()
                .collect::<JsonObject>();
            for error in [
                reject_unknown_benchmark_manifest_fields(&obj, "dataset.json").unwrap_err(),
                reject_unknown_benchmark_hash_fields(&obj, "dataset.json").unwrap_err(),
            ] {
                assert_eq!(error.code(), ErrorCode::InvalidJsonlSchema);
                assert!(!error.to_string().contains(&secret));
            }
            for kind in [
                "documents",
                "chunks",
                "embeddings",
                "queries",
                "qrels",
                "query_embeddings",
            ] {
                let artifact = Artifact {
                    raw_path: format!("{kind}.jsonl"),
                    path: PathBuf::from("/private/unused-resolved-path"),
                    kind,
                    hash_field: "unused",
                };
                let error =
                    reject_unknown_benchmark_artifact_fields(&obj, &artifact, 1, "dataset.json")
                        .unwrap_err();
                assert_eq!(detail_value(&error, "field"), Some(kind));
                assert!(!error.to_string().contains(&secret));
                assert!(!error.to_string().contains("/private/unused-resolved-path"));
            }
        }
    }

    #[test]
    fn benchmark_manifest_rejects_unknown_public_fields() {
        let dir = temp_dir("unknown-benchmark-manifest-field");
        fs::write(
            dir.join("dataset_manifest.json"),
            r#"{"schema_version":"0.1","dataset_id":"bench","version":"0.1","documents":"documents.jsonl","chunks":"chunks.jsonl","queries":"queries.jsonl","qrels":"qrels.jsonl","query_embedding_source":"none","hashes":{"documents_sha256":"x","chunks_sha256":"x","queries_sha256":"x","qrels_sha256":"x"},"document":"typo.jsonl"}"#,
        )
        .unwrap();

        let err = BenchmarkDatasetManifest::load_and_validate(&dir).unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
        assert_eq!(detail_value(&err, "field"), Some("manifest"));

        fs::write(
            dir.join("dataset_manifest.json"),
            r#"{"schema_version":"0.1","dataset_id":"bench","version":"0.1","documents":"documents.jsonl","chunks":"chunks.jsonl","queries":"queries.jsonl","qrels":"qrels.jsonl","query_embedding_source":"none","embedding_dim":2,"hashes":{"documents_sha256":"x","chunks_sha256":"x","queries_sha256":"x","qrels_sha256":"x"}}"#,
        )
        .unwrap();

        let err = BenchmarkDatasetManifest::load_and_validate(&dir).unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
        assert_eq!(detail_value(&err, "field"), Some("manifest"));

        fs::write(
            dir.join("dataset_manifest.json"),
            r#"{"schema_version":"0.1","dataset_id":"bench","version":"0.1","documents":"documents.jsonl","chunks":"chunks.jsonl","queries":"queries.jsonl","qrels":"qrels.jsonl","query_embedding_source":"none","hashes":{"documents_hash":"typo","documents_sha256":"x","chunks_sha256":"x","queries_sha256":"x","qrels_sha256":"x"}}"#,
        )
        .unwrap();

        let err = BenchmarkDatasetManifest::load_and_validate(&dir).unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
        assert_eq!(detail_value(&err, "field"), Some("hashes"));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn benchmark_artifact_rows_reject_unknown_public_fields() {
        let valid_documents =
            r#"{"external_id":"doc","title":"Doc","source_uri":"docs/wal.md","metadata":{}}"#;
        let valid_chunks = r#"{"document_external_id":"doc","chunk_external_id":"doc#1","ordinal":1,"text":"wal","metadata":{},"source_span":{"start":0,"end":3}}"#;
        let valid_embeddings =
            r#"{"chunk_external_id":"doc#1","model_name":"m","dim":2,"vector":[1.0,0.0]}"#;
        let valid_queries = r#"{"query_id":"q1","text":"wal","top_k":10}"#;
        let valid_qrels = r#"{"query_id":"q1","chunk_external_id":"doc#1","relevance":2}"#;
        let valid_query_embeddings =
            r#"{"query_id":"q1","model_name":"m","dim":2,"vector":[1.0,0.0]}"#;

        for (
            name,
            expected_field,
            documents_content,
            chunks_content,
            embeddings_content,
            queries_content,
            qrels_content,
            query_embeddings_content,
        ) in [
            (
                "unknown-doc-row-field",
                "documents",
                r#"{"external_id":"doc","metadata":{},"alias":"doc"}"#,
                valid_chunks,
                valid_embeddings,
                valid_queries,
                valid_qrels,
                valid_query_embeddings,
            ),
            (
                "unknown-chunk-row-field",
                "chunks",
                valid_documents,
                r#"{"document_external_id":"doc","chunk_external_id":"doc#1","ordinal":1,"text":"wal","metadata":{},"vector":[1.0]}"#,
                valid_embeddings,
                valid_queries,
                valid_qrels,
                valid_query_embeddings,
            ),
            (
                "unknown-embedding-row-field",
                "embeddings",
                valid_documents,
                valid_chunks,
                r#"{"chunk_external_id":"doc#1","model_name":"m","dim":2,"vector":[1.0,0.0],"extra":true}"#,
                valid_queries,
                valid_qrels,
                valid_query_embeddings,
            ),
            (
                "unknown-query-row-field",
                "queries",
                valid_documents,
                valid_chunks,
                valid_embeddings,
                r#"{"query_id":"q1","text":"wal","top_k":10,"category":"hybrid"}"#,
                valid_qrels,
                valid_query_embeddings,
            ),
            (
                "unknown-qrel-row-field",
                "qrels",
                valid_documents,
                valid_chunks,
                valid_embeddings,
                valid_queries,
                r#"{"query_id":"q1","chunk_external_id":"doc#1","relevance":2,"note":"gold"}"#,
                valid_query_embeddings,
            ),
            (
                "unknown-query-embedding-row-field",
                "query_embeddings",
                valid_documents,
                valid_chunks,
                valid_embeddings,
                valid_queries,
                valid_qrels,
                r#"{"query_id":"q1","model_name":"m","dim":2,"vector":[1.0,0.0],"score":1.0}"#,
            ),
        ] {
            let dir = temp_dir(name);
            write_sidecar_fixture_with_artifacts(
                &dir,
                documents_content,
                chunks_content,
                embeddings_content,
                queries_content,
                qrels_content,
                query_embeddings_content,
            );

            let err = BenchmarkDatasetManifest::load_and_validate(&dir).unwrap_err();
            assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
            assert_eq!(detail_value(&err, "field"), Some(expected_field));
            assert_eq!(detail_value(&err, "line"), Some("1"));
            let _ = fs::remove_dir_all(dir);
        }
    }

    #[test]
    fn benchmark_manifest_rejects_blank_hash_values() {
        let dir = temp_dir("blank-benchmark-hash");
        fs::write(
            dir.join("dataset_manifest.json"),
            r#"{"schema_version":"0.1","dataset_id":"bench","version":"0.1","documents":"documents.jsonl","chunks":"chunks.jsonl","queries":"queries.jsonl","qrels":"qrels.jsonl","query_embedding_source":"none","hashes":{"documents_sha256":"","chunks_sha256":"x","queries_sha256":"x","qrels_sha256":"x"}}"#,
        )
        .unwrap();

        let err = BenchmarkDatasetManifest::load_and_validate(&dir).unwrap_err();
        assert_eq!(err.code(), ErrorCode::MissingRequiredField);
        assert_eq!(detail_value(&err, "field"), Some("hashes.documents_sha256"));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn benchmark_manifest_rejects_blank_identity_fields() {
        let dir = temp_dir("blank-dataset-id");
        write_none_fixture(
            &dir,
            r#"{"query_id":"q1","text":"wal","top_k":10}"#,
            r#"{"query_id":"q1","chunk_external_id":"doc#1","relevance":2}"#,
        );
        let documents = sha256_hex(&fs::read(dir.join("documents.jsonl")).unwrap());
        let chunks = sha256_hex(&fs::read(dir.join("chunks.jsonl")).unwrap());
        let queries = sha256_hex(&fs::read(dir.join("queries.jsonl")).unwrap());
        let qrels = sha256_hex(&fs::read(dir.join("qrels.jsonl")).unwrap());
        fs::write(
            dir.join("dataset_manifest.json"),
            format!(
                concat!(
                    r#"{{"schema_version":"0.1","dataset_id":"","version":"0.1","#,
                    r#""documents":"documents.jsonl","chunks":"chunks.jsonl","#,
                    r#""queries":"queries.jsonl","qrels":"qrels.jsonl","#,
                    r#""query_embedding_source":"none","hashes":{{"#,
                    r#""documents_sha256":"{}","chunks_sha256":"{}","#,
                    r#""queries_sha256":"{}","qrels_sha256":"{}"}}"#,
                    r#"}}"#
                ),
                documents, chunks, queries, qrels
            ),
        )
        .unwrap();

        let err = BenchmarkDatasetManifest::load_and_validate(&dir).unwrap_err();
        assert_eq!(err.code(), ErrorCode::MissingRequiredField);
        assert_eq!(detail_value(&err, "field"), Some("dataset_id"));
        let _ = fs::remove_dir_all(dir);

        let dir = temp_dir("blank-query-embedding-model");
        write_sidecar_fixture(
            &dir,
            r#"{"query_id":"q1","text":"wal","top_k":10}"#,
            r#"{"query_id":"q1","chunk_external_id":"doc#1","relevance":2}"#,
            r#"{"query_id":"q1","model_name":"m","dim":2,"vector":[1.0,0.0]}"#,
        );
        let documents = sha256_hex(&fs::read(dir.join("documents.jsonl")).unwrap());
        let chunks = sha256_hex(&fs::read(dir.join("chunks.jsonl")).unwrap());
        let embeddings = sha256_hex(&fs::read(dir.join("embeddings.jsonl")).unwrap());
        let queries = sha256_hex(&fs::read(dir.join("queries.jsonl")).unwrap());
        let qrels = sha256_hex(&fs::read(dir.join("qrels.jsonl")).unwrap());
        let query_embeddings = sha256_hex(&fs::read(dir.join("query_embeddings.jsonl")).unwrap());
        fs::write(
            dir.join("dataset_manifest.json"),
            format!(
                concat!(
                    r#"{{"schema_version":"0.1","dataset_id":"bench","version":"0.1","#,
                    r#""documents":"documents.jsonl","chunks":"chunks.jsonl","#,
                    r#""embeddings":"embeddings.jsonl","queries":"queries.jsonl","#,
                    r#""qrels":"qrels.jsonl","query_embedding_source":"sidecar","#,
                    r#""query_embeddings":"query_embeddings.jsonl","embedding_model":"m","#,
                    r#""query_embedding_model":"","query_embedding_dim":2,"hashes":{{"#,
                    r#""documents_sha256":"{}","chunks_sha256":"{}","#,
                    r#""embeddings_sha256":"{}","queries_sha256":"{}","#,
                    r#""qrels_sha256":"{}","query_embeddings_sha256":"{}"}}"#,
                    r#"}}"#
                ),
                documents, chunks, embeddings, queries, qrels, query_embeddings
            ),
        )
        .unwrap();

        let err = BenchmarkDatasetManifest::load_and_validate(&dir).unwrap_err();
        assert_eq!(err.code(), ErrorCode::MissingRequiredField);
        assert_eq!(detail_value(&err, "field"), Some("query_embedding_model"));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn benchmark_queries_reject_invalid_metadata_filter() {
        let dir = temp_dir("query-invalid-filter");
        write_none_fixture(
            &dir,
            r#"{"query_id":"q1","text":"wal","filter":{"lang":"rust"},"top_k":10}"#,
            r#"{"query_id":"q1","chunk_external_id":"doc#1","relevance":2}"#,
        );

        let err = BenchmarkDatasetManifest::load_and_validate(&dir).unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
        assert_eq!(detail_value(&err, "query_id"), Some("q1"));
        assert_eq!(detail_value(&err, "field"), Some("queries.filter"));
        assert_eq!(
            detail_value(&err, "filter_error_code"),
            Some("invalid_metadata_filter")
        );
        assert_eq!(detail_value(&err, "artifact_kind"), Some("queries"));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn qrels_must_reference_declared_queries() {
        let dir = temp_dir("qrels-query-id");
        let documents = write_file(
            &dir,
            "documents.jsonl",
            r#"{"external_id":"doc","metadata":{}}"#,
        );
        let chunks = write_file(
            &dir,
            "chunks.jsonl",
            r#"{"document_external_id":"doc","chunk_external_id":"doc#1","ordinal":1,"text":"wal","metadata":{}}"#,
        );
        let queries = write_file(
            &dir,
            "queries.jsonl",
            r#"{"query_id":"q1","text":"wal","top_k":10}"#,
        );
        let qrels = write_file(
            &dir,
            "qrels.jsonl",
            r#"{"query_id":"q-missing","chunk_external_id":"doc#1","relevance":2}"#,
        );
        fs::write(
            dir.join("dataset_manifest.json"),
            format!(
                concat!(
                    r#"{{"schema_version":"0.1","dataset_id":"bench","version":"0.1","#,
                    r#""documents":"documents.jsonl","chunks":"chunks.jsonl","#,
                    r#""queries":"queries.jsonl","qrels":"qrels.jsonl","#,
                    r#""query_embedding_source":"none","hashes":{{"#,
                    r#""documents_sha256":"{}","chunks_sha256":"{}","#,
                    r#""queries_sha256":"{}","qrels_sha256":"{}"}}"#,
                    r#"}}"#
                ),
                documents, chunks, queries, qrels
            ),
        )
        .unwrap();

        let err = BenchmarkDatasetManifest::load_and_validate(&dir).unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
        assert_eq!(detail_value(&err, "query_id"), Some("q-missing"));
        assert_eq!(detail_value(&err, "field"), Some("qrels.query_id"));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn benchmark_chunks_must_reference_declared_documents() {
        let dir = temp_dir("chunks-document-id");
        let documents = write_file(
            &dir,
            "documents.jsonl",
            r#"{"external_id":"doc","metadata":{}}"#,
        );
        let chunks = write_file(
            &dir,
            "chunks.jsonl",
            r#"{"document_external_id":"missing-doc","chunk_external_id":"doc#1","ordinal":1,"text":"wal","metadata":{}}"#,
        );
        let queries = write_file(
            &dir,
            "queries.jsonl",
            r#"{"query_id":"q1","text":"wal","top_k":10}"#,
        );
        let qrels = write_file(
            &dir,
            "qrels.jsonl",
            r#"{"query_id":"q1","chunk_external_id":"doc#1","relevance":2}"#,
        );
        fs::write(
            dir.join("dataset_manifest.json"),
            format!(
                concat!(
                    r#"{{"schema_version":"0.1","dataset_id":"bench","version":"0.1","#,
                    r#""documents":"documents.jsonl","chunks":"chunks.jsonl","#,
                    r#""queries":"queries.jsonl","qrels":"qrels.jsonl","#,
                    r#""query_embedding_source":"none","hashes":{{"#,
                    r#""documents_sha256":"{}","chunks_sha256":"{}","#,
                    r#""queries_sha256":"{}","qrels_sha256":"{}"}}"#,
                    r#"}}"#
                ),
                documents, chunks, queries, qrels
            ),
        )
        .unwrap();

        let err = BenchmarkDatasetManifest::load_and_validate(&dir).unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
        assert_eq!(
            detail_value(&err, "document_external_id"),
            Some("missing-doc")
        );
        assert_eq!(
            detail_value(&err, "field"),
            Some("chunks.document_external_id")
        );
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn benchmark_documents_reject_duplicate_external_ids() {
        let dir = temp_dir("duplicate-document-id");
        let documents = write_file(
            &dir,
            "documents.jsonl",
            concat!(
                r#"{"external_id":"doc","metadata":{}}"#,
                "\n",
                r#"{"external_id":"doc","metadata":{}}"#
            ),
        );
        let chunks = write_file(
            &dir,
            "chunks.jsonl",
            r#"{"document_external_id":"doc","chunk_external_id":"doc#1","ordinal":1,"text":"wal","metadata":{}}"#,
        );
        let queries = write_file(
            &dir,
            "queries.jsonl",
            r#"{"query_id":"q1","text":"wal","top_k":10}"#,
        );
        let qrels = write_file(
            &dir,
            "qrels.jsonl",
            r#"{"query_id":"q1","chunk_external_id":"doc#1","relevance":2}"#,
        );
        fs::write(
            dir.join("dataset_manifest.json"),
            format!(
                concat!(
                    r#"{{"schema_version":"0.1","dataset_id":"bench","version":"0.1","#,
                    r#""documents":"documents.jsonl","chunks":"chunks.jsonl","#,
                    r#""queries":"queries.jsonl","qrels":"qrels.jsonl","#,
                    r#""query_embedding_source":"none","hashes":{{"#,
                    r#""documents_sha256":"{}","chunks_sha256":"{}","#,
                    r#""queries_sha256":"{}","qrels_sha256":"{}"}}"#,
                    r#"}}"#
                ),
                documents, chunks, queries, qrels
            ),
        )
        .unwrap();

        let err = BenchmarkDatasetManifest::load_and_validate(&dir).unwrap_err();
        assert_eq!(err.code(), ErrorCode::DuplicateExternalId);
        assert_eq!(detail_value(&err, "external_id"), Some("doc"));
        assert_eq!(detail_value(&err, "line"), Some("2"));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn benchmark_documents_require_metadata_object() {
        let dir = temp_dir("document-missing-metadata");
        let documents = write_file(&dir, "documents.jsonl", r#"{"external_id":"doc"}"#);
        let chunks = write_file(
            &dir,
            "chunks.jsonl",
            r#"{"document_external_id":"doc","chunk_external_id":"doc#1","ordinal":1,"text":"wal","metadata":{}}"#,
        );
        let queries = write_file(
            &dir,
            "queries.jsonl",
            r#"{"query_id":"q1","text":"wal","top_k":10}"#,
        );
        let qrels = write_file(
            &dir,
            "qrels.jsonl",
            r#"{"query_id":"q1","chunk_external_id":"doc#1","relevance":2}"#,
        );
        fs::write(
            dir.join("dataset_manifest.json"),
            format!(
                concat!(
                    r#"{{"schema_version":"0.1","dataset_id":"bench","version":"0.1","#,
                    r#""documents":"documents.jsonl","chunks":"chunks.jsonl","#,
                    r#""queries":"queries.jsonl","qrels":"qrels.jsonl","#,
                    r#""query_embedding_source":"none","hashes":{{"#,
                    r#""documents_sha256":"{}","chunks_sha256":"{}","#,
                    r#""queries_sha256":"{}","qrels_sha256":"{}"}}"#,
                    r#"}}"#
                ),
                documents, chunks, queries, qrels
            ),
        )
        .unwrap();

        let err = BenchmarkDatasetManifest::load_and_validate(&dir).unwrap_err();
        assert_eq!(err.code(), ErrorCode::MissingRequiredField);
        assert_eq!(detail_value(&err, "field"), Some("documents.metadata"));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn benchmark_chunks_require_ordinal_text_and_metadata() {
        let dir = temp_dir("chunk-missing-ordinal");
        let documents = write_file(
            &dir,
            "documents.jsonl",
            r#"{"external_id":"doc","metadata":{}}"#,
        );
        let chunks = write_file(
            &dir,
            "chunks.jsonl",
            r#"{"document_external_id":"doc","chunk_external_id":"doc#1","text":"wal","metadata":{}}"#,
        );
        let queries = write_file(
            &dir,
            "queries.jsonl",
            r#"{"query_id":"q1","text":"wal","top_k":10}"#,
        );
        let qrels = write_file(
            &dir,
            "qrels.jsonl",
            r#"{"query_id":"q1","chunk_external_id":"doc#1","relevance":2}"#,
        );
        fs::write(
            dir.join("dataset_manifest.json"),
            format!(
                concat!(
                    r#"{{"schema_version":"0.1","dataset_id":"bench","version":"0.1","#,
                    r#""documents":"documents.jsonl","chunks":"chunks.jsonl","#,
                    r#""queries":"queries.jsonl","qrels":"qrels.jsonl","#,
                    r#""query_embedding_source":"none","hashes":{{"#,
                    r#""documents_sha256":"{}","chunks_sha256":"{}","#,
                    r#""queries_sha256":"{}","qrels_sha256":"{}"}}"#,
                    r#"}}"#
                ),
                documents, chunks, queries, qrels
            ),
        )
        .unwrap();

        let err = BenchmarkDatasetManifest::load_and_validate(&dir).unwrap_err();
        assert_eq!(err.code(), ErrorCode::MissingRequiredField);
        assert_eq!(detail_value(&err, "field"), Some("chunks.ordinal"));
        let _ = fs::remove_dir_all(dir);

        let dir = temp_dir("chunk-missing-text");
        let documents = write_file(
            &dir,
            "documents.jsonl",
            r#"{"external_id":"doc","metadata":{}}"#,
        );
        let chunks = write_file(
            &dir,
            "chunks.jsonl",
            r#"{"document_external_id":"doc","chunk_external_id":"doc#1","ordinal":1,"metadata":{}}"#,
        );
        let queries = write_file(
            &dir,
            "queries.jsonl",
            r#"{"query_id":"q1","text":"wal","top_k":10}"#,
        );
        let qrels = write_file(
            &dir,
            "qrels.jsonl",
            r#"{"query_id":"q1","chunk_external_id":"doc#1","relevance":2}"#,
        );
        fs::write(
            dir.join("dataset_manifest.json"),
            format!(
                concat!(
                    r#"{{"schema_version":"0.1","dataset_id":"bench","version":"0.1","#,
                    r#""documents":"documents.jsonl","chunks":"chunks.jsonl","#,
                    r#""queries":"queries.jsonl","qrels":"qrels.jsonl","#,
                    r#""query_embedding_source":"none","hashes":{{"#,
                    r#""documents_sha256":"{}","chunks_sha256":"{}","#,
                    r#""queries_sha256":"{}","qrels_sha256":"{}"}}"#,
                    r#"}}"#
                ),
                documents, chunks, queries, qrels
            ),
        )
        .unwrap();

        let err = BenchmarkDatasetManifest::load_and_validate(&dir).unwrap_err();
        assert_eq!(err.code(), ErrorCode::MissingRequiredField);
        assert_eq!(detail_value(&err, "field"), Some("chunks.text"));
        let _ = fs::remove_dir_all(dir);

        let dir = temp_dir("chunk-missing-metadata");
        let documents = write_file(
            &dir,
            "documents.jsonl",
            r#"{"external_id":"doc","metadata":{}}"#,
        );
        let chunks = write_file(
            &dir,
            "chunks.jsonl",
            r#"{"document_external_id":"doc","chunk_external_id":"doc#1","ordinal":1,"text":"wal"}"#,
        );
        let queries = write_file(
            &dir,
            "queries.jsonl",
            r#"{"query_id":"q1","text":"wal","top_k":10}"#,
        );
        let qrels = write_file(
            &dir,
            "qrels.jsonl",
            r#"{"query_id":"q1","chunk_external_id":"doc#1","relevance":2}"#,
        );
        fs::write(
            dir.join("dataset_manifest.json"),
            format!(
                concat!(
                    r#"{{"schema_version":"0.1","dataset_id":"bench","version":"0.1","#,
                    r#""documents":"documents.jsonl","chunks":"chunks.jsonl","#,
                    r#""queries":"queries.jsonl","qrels":"qrels.jsonl","#,
                    r#""query_embedding_source":"none","hashes":{{"#,
                    r#""documents_sha256":"{}","chunks_sha256":"{}","#,
                    r#""queries_sha256":"{}","qrels_sha256":"{}"}}"#,
                    r#"}}"#
                ),
                documents, chunks, queries, qrels
            ),
        )
        .unwrap();

        let err = BenchmarkDatasetManifest::load_and_validate(&dir).unwrap_err();
        assert_eq!(err.code(), ErrorCode::MissingRequiredField);
        assert_eq!(detail_value(&err, "field"), Some("chunks.metadata"));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn qrels_must_reference_declared_chunks() {
        let dir = temp_dir("qrels-chunk-id");
        let documents = write_file(
            &dir,
            "documents.jsonl",
            r#"{"external_id":"doc","metadata":{}}"#,
        );
        let chunks = write_file(
            &dir,
            "chunks.jsonl",
            r#"{"document_external_id":"doc","chunk_external_id":"doc#1","ordinal":1,"text":"wal","metadata":{}}"#,
        );
        let queries = write_file(
            &dir,
            "queries.jsonl",
            r#"{"query_id":"q1","text":"wal","top_k":10}"#,
        );
        let qrels = write_file(
            &dir,
            "qrels.jsonl",
            r#"{"query_id":"q1","chunk_external_id":"doc#missing","relevance":2}"#,
        );
        fs::write(
            dir.join("dataset_manifest.json"),
            format!(
                concat!(
                    r#"{{"schema_version":"0.1","dataset_id":"bench","version":"0.1","#,
                    r#""documents":"documents.jsonl","chunks":"chunks.jsonl","#,
                    r#""queries":"queries.jsonl","qrels":"qrels.jsonl","#,
                    r#""query_embedding_source":"none","hashes":{{"#,
                    r#""documents_sha256":"{}","chunks_sha256":"{}","#,
                    r#""queries_sha256":"{}","qrels_sha256":"{}"}}"#,
                    r#"}}"#
                ),
                documents, chunks, queries, qrels
            ),
        )
        .unwrap();

        let err = BenchmarkDatasetManifest::load_and_validate(&dir).unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
        assert_eq!(detail_value(&err, "query_id"), Some("q1"));
        assert_eq!(detail_value(&err, "chunk_external_id"), Some("doc#missing"));
        assert_eq!(detail_value(&err, "field"), Some("qrels.chunk_external_id"));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn zero_relevance_queries_are_judged_without_positive_relevant_chunks() {
        let dir = temp_dir("qrels-judged-zero");
        let queries = (0..100)
            .map(|id| format!("{{\"query_id\":\"q-{id}\",\"text\":\"wal\",\"top_k\":10}}\n"))
            .collect::<String>();
        let qrels = (0..100).rev().map(|id| format!("{{\"query_id\":\"q-{id}\",\"chunk_external_id\":\"doc#1\",\"relevance\":{}}}\n", id % 3)).collect::<String>();
        write_none_fixture(&dir, &queries, &qrels);
        let dataset = BenchmarkDataset::load_and_validate(&dir).unwrap();
        assert_eq!(dataset.queries.len(), 100);
        for (id, query) in dataset.queries.iter().enumerate() {
            assert_eq!(query.query_id, format!("q-{id}"));
            assert_eq!(query.relevant_chunk_ids.len(), usize::from(id % 3 != 0));
            if id % 3 != 0 {
                assert!(query.relevant_chunk_ids.contains("doc#1"));
            }
        }
        write_none_fixture(
            &dir,
            &queries,
            &qrels.lines().skip(1).collect::<Vec<_>>().join("\n"),
        );
        let error = BenchmarkDataset::load_and_validate(&dir).unwrap_err();
        assert_eq!(error.code(), ErrorCode::MissingRequiredField);
        assert_eq!(detail_value(&error, "query_id"), Some("q-99"));
        assert_eq!(detail_value(&error, "field"), Some("qrels[query_id]"));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn zero_relevance_qrels_still_validate_identity_and_duplicates() {
        let dir = temp_dir("qrels-zero-unknown-query");
        write_none_fixture(
            &dir,
            r#"{"query_id":"q1","text":"wal","top_k":10}"#,
            r#"{"query_id":"q-missing","chunk_external_id":"doc#1","relevance":0}"#,
        );

        let err = BenchmarkDatasetManifest::load_and_validate(&dir).unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
        assert_eq!(detail_value(&err, "query_id"), Some("q-missing"));
        assert_eq!(detail_value(&err, "field"), Some("qrels.query_id"));
        let _ = fs::remove_dir_all(dir);

        let dir = temp_dir("qrels-zero-duplicate-pair");
        write_none_fixture(
            &dir,
            r#"{"query_id":"q1","text":"wal","top_k":10}"#,
            concat!(
                r#"{"query_id":"q1","chunk_external_id":"doc#1","relevance":0}"#,
                "\n",
                r#"{"query_id":"q1","chunk_external_id":"doc#1","relevance":2}"#
            ),
        );

        let err = BenchmarkDatasetManifest::load_and_validate(&dir).unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
        assert_eq!(detail_value(&err, "query_id"), Some("q1"));
        assert_eq!(detail_value(&err, "chunk_external_id"), Some("doc#1"));
        assert_eq!(detail_value(&err, "first_line"), Some("1"));
        assert_eq!(detail_value(&err, "duplicate_line"), Some("2"));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn qrels_reject_duplicate_query_chunk_pairs() {
        let dir = temp_dir("qrels-duplicate-pair");
        let documents = write_file(
            &dir,
            "documents.jsonl",
            r#"{"external_id":"doc","metadata":{}}"#,
        );
        let chunks = write_file(
            &dir,
            "chunks.jsonl",
            r#"{"document_external_id":"doc","chunk_external_id":"doc#1","ordinal":1,"text":"wal","metadata":{}}"#,
        );
        let queries = write_file(
            &dir,
            "queries.jsonl",
            r#"{"query_id":"q1","text":"wal","top_k":10}"#,
        );
        let qrels = write_file(
            &dir,
            "qrels.jsonl",
            concat!(
                r#"{"query_id":"q1","chunk_external_id":"doc#1","relevance":2}"#,
                "\n",
                r#"{"query_id":"q1","chunk_external_id":"doc#1","relevance":1}"#
            ),
        );
        fs::write(
            dir.join("dataset_manifest.json"),
            format!(
                concat!(
                    r#"{{"schema_version":"0.1","dataset_id":"bench","version":"0.1","#,
                    r#""documents":"documents.jsonl","chunks":"chunks.jsonl","#,
                    r#""queries":"queries.jsonl","qrels":"qrels.jsonl","#,
                    r#""query_embedding_source":"none","hashes":{{"#,
                    r#""documents_sha256":"{}","chunks_sha256":"{}","#,
                    r#""queries_sha256":"{}","qrels_sha256":"{}"}}"#,
                    r#"}}"#
                ),
                documents, chunks, queries, qrels
            ),
        )
        .unwrap();

        let err = BenchmarkDatasetManifest::load_and_validate(&dir).unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
        assert_eq!(detail_value(&err, "query_id"), Some("q1"));
        assert_eq!(detail_value(&err, "chunk_external_id"), Some("doc#1"));
        assert_eq!(detail_value(&err, "first_line"), Some("1"));
        assert_eq!(detail_value(&err, "duplicate_line"), Some("2"));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn benchmark_chunks_reject_duplicate_external_ids_before_qrels() {
        let dir = temp_dir("duplicate-chunk-id");
        let documents = write_file(
            &dir,
            "documents.jsonl",
            r#"{"external_id":"doc","metadata":{}}"#,
        );
        let chunks = write_file(
            &dir,
            "chunks.jsonl",
            concat!(
                r#"{"document_external_id":"doc","chunk_external_id":"doc#1","ordinal":1,"text":"wal","metadata":{}}"#,
                "\n",
                r#"{"document_external_id":"doc","chunk_external_id":"doc#1","ordinal":2,"text":"recovery","metadata":{}}"#
            ),
        );
        let queries = write_file(
            &dir,
            "queries.jsonl",
            r#"{"query_id":"q1","text":"wal","top_k":10}"#,
        );
        let qrels = write_file(
            &dir,
            "qrels.jsonl",
            r#"{"query_id":"q1","chunk_external_id":"doc#1","relevance":2}"#,
        );
        fs::write(
            dir.join("dataset_manifest.json"),
            format!(
                concat!(
                    r#"{{"schema_version":"0.1","dataset_id":"bench","version":"0.1","#,
                    r#""documents":"documents.jsonl","chunks":"chunks.jsonl","#,
                    r#""queries":"queries.jsonl","qrels":"qrels.jsonl","#,
                    r#""query_embedding_source":"none","hashes":{{"#,
                    r#""documents_sha256":"{}","chunks_sha256":"{}","#,
                    r#""queries_sha256":"{}","qrels_sha256":"{}"}}"#,
                    r#"}}"#
                ),
                documents, chunks, queries, qrels
            ),
        )
        .unwrap();

        let err = BenchmarkDatasetManifest::load_and_validate(&dir).unwrap_err();
        assert_eq!(err.code(), ErrorCode::DuplicateExternalId);
        assert_eq!(detail_value(&err, "chunk_external_id"), Some("doc#1"));
        assert_eq!(detail_value(&err, "line"), Some("2"));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn benchmark_chunks_reject_duplicate_ordinals_within_document() {
        let dir = temp_dir("duplicate-chunk-ordinal");
        let documents = write_file(
            &dir,
            "documents.jsonl",
            r#"{"external_id":"doc","metadata":{}}"#,
        );
        let chunks = write_file(
            &dir,
            "chunks.jsonl",
            concat!(
                r#"{"document_external_id":"doc","chunk_external_id":"doc#1","ordinal":1,"text":"wal","metadata":{}}"#,
                "\n",
                r#"{"document_external_id":"doc","chunk_external_id":"doc#2","ordinal":1,"text":"recovery","metadata":{}}"#
            ),
        );
        let queries = write_file(
            &dir,
            "queries.jsonl",
            r#"{"query_id":"q1","text":"wal","top_k":10}"#,
        );
        let qrels = write_file(
            &dir,
            "qrels.jsonl",
            r#"{"query_id":"q1","chunk_external_id":"doc#1","relevance":2}"#,
        );
        fs::write(
            dir.join("dataset_manifest.json"),
            format!(
                concat!(
                    r#"{{"schema_version":"0.1","dataset_id":"bench","version":"0.1","#,
                    r#""documents":"documents.jsonl","chunks":"chunks.jsonl","#,
                    r#""queries":"queries.jsonl","qrels":"qrels.jsonl","#,
                    r#""query_embedding_source":"none","hashes":{{"#,
                    r#""documents_sha256":"{}","chunks_sha256":"{}","#,
                    r#""queries_sha256":"{}","qrels_sha256":"{}"}}"#,
                    r#"}}"#
                ),
                documents, chunks, queries, qrels
            ),
        )
        .unwrap();

        let err = BenchmarkDatasetManifest::load_and_validate(&dir).unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
        assert_eq!(detail_value(&err, "field"), Some("chunks.ordinal"));
        assert_eq!(detail_value(&err, "document_external_id"), Some("doc"));
        assert_eq!(detail_value(&err, "ordinal"), Some("1"));
        assert_eq!(detail_value(&err, "first_line"), Some("1"));
        assert_eq!(detail_value(&err, "duplicate_line"), Some("2"));
        let _ = fs::remove_dir_all(dir);
    }
}
