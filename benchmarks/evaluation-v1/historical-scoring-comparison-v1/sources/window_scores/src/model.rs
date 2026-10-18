use std::collections::BTreeMap;

use crate::error::{Error, ErrorCode, Result};
use crate::hash::sha256_hex;
use crate::json::{
    ensure_object, object_from_pairs, optional_string, required_array, required_string,
    required_usize, JsonBuilder, JsonValue,
};
use crate::profile::ResourceProfile;
use crate::time;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DistanceMetric {
    Cosine,
    DotProduct,
    Euclidean,
}

impl DistanceMetric {
    pub fn as_str(self) -> &'static str {
        match self {
            DistanceMetric::Cosine => "cosine",
            DistanceMetric::DotProduct => "dot_product",
            DistanceMetric::Euclidean => "euclidean",
        }
    }

    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "cosine" => Ok(DistanceMetric::Cosine),
            "dot" | "dot_product" | "dot-product" => Ok(DistanceMetric::DotProduct),
            "euclidean" | "l2" => Ok(DistanceMetric::Euclidean),
            other => Err(Error::new(
                ErrorCode::UnsupportedMetric,
                format!("unsupported distance metric {other}"),
            )),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub enum VectorIndexConfig {
    #[default]
    Exact,
    Ann {
        provider: String,
        params: JsonValue,
    },
}

#[derive(Clone, Debug)]
pub struct CollectionConfig {
    pub tokenizer: crate::TokenizerConfig,
    pub name: String,
    pub embedding_dim: usize,
    pub distance_metric: DistanceMetric,
    pub embedding_model: Option<String>,
    pub vectors_normalized: bool,
    pub vector_index: VectorIndexConfig,
}

#[derive(Clone, Debug)]
pub struct CollectionInfo {
    pub tokenizer: crate::TokenizerConfig,
    pub collection_id: String,
    pub name: String,
    pub embedding_dim: usize,
    pub distance_metric: DistanceMetric,
    pub embedding_model: Option<String>,
    pub vectors_normalized: bool,
    pub vector_index: VectorIndexConfig,
    pub created_at: String,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SourceSpan {
    pub start: Option<u64>,
    pub end: Option<u64>,
    pub line_start: Option<u32>,
    pub line_end: Option<u32>,
}

#[derive(Clone, Debug)]
pub struct DocumentInput {
    pub external_id: String,
    pub title: Option<String>,
    pub source_uri: Option<String>,
    pub metadata: JsonValue,
    pub chunks: Vec<ChunkInput>,
}

#[derive(Clone, Debug)]
pub struct ChunkInput {
    pub chunk_external_id: String,
    pub text: String,
    pub ordinal: usize,
    pub metadata: JsonValue,
    pub source_span: Option<SourceSpan>,
}

#[derive(Clone, Debug)]
pub struct EmbeddingInput {
    pub chunk_external_id: String,
    pub model_name: String,
    pub dim: usize,
    pub vector: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct QueryEmbeddingInput {
    pub model_name: Option<String>,
    pub dim: usize,
    pub vector: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct Document {
    pub document_id: String,
    pub collection_id: String,
    pub external_id: String,
    pub title: Option<String>,
    pub source_uri: Option<String>,
    pub metadata: JsonValue,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Clone, Debug)]
pub struct Chunk {
    pub chunk_id: String,
    pub document_id: String,
    pub collection_id: String,
    pub chunk_external_id: String,
    pub text: String,
    pub text_hash: String,
    pub ordinal: usize,
    pub token_count: usize,
    pub metadata: JsonValue,
    pub embedding_id: Option<String>,
    pub source_span: Option<SourceSpan>,
    pub created_at: String,
}

#[derive(Clone, Debug)]
pub struct Embedding {
    pub embedding_id: String,
    pub chunk_id: String,
    pub collection_id: String,
    pub chunk_external_id: String,
    pub model_name: String,
    pub dim: usize,
    pub vector: Vec<f32>,
    pub created_at: String,
}

#[derive(Clone, Debug)]
pub struct Tombstone {
    pub target_type: String,
    pub target_id: String,
    pub deleted_at: String,
    pub reason: Option<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OnConflict {
    Error,
    Skip,
    Replace,
}

impl OnConflict {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "error" => Ok(OnConflict::Error),
            "skip" => Ok(OnConflict::Skip),
            "replace" => Ok(OnConflict::Replace),
            other => Err(Error::new(
                ErrorCode::CliUsage,
                format!("unsupported conflict strategy {other}"),
            )),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SearchWeights {
    pub bm25: f64,
    pub vector: f64,
    pub metadata: f64,
}

impl SearchWeights {
    const SUM_EPSILON: f64 = 1e-9;

    pub fn new(bm25: f64, vector: f64, metadata: f64) -> Result<Self> {
        let weights = Self {
            bm25,
            vector,
            metadata,
        };
        weights.validate()?;
        Ok(weights)
    }

    pub fn default_for(has_text: bool, has_vector: bool) -> Self {
        match (has_text, has_vector) {
            (true, true) => Self {
                bm25: 0.5,
                vector: 0.5,
                metadata: 0.0,
            },
            (true, false) => Self {
                bm25: 1.0,
                vector: 0.0,
                metadata: 0.0,
            },
            (false, true) => Self {
                bm25: 0.0,
                vector: 1.0,
                metadata: 0.0,
            },
            (false, false) => Self {
                bm25: 0.0,
                vector: 0.0,
                metadata: 0.0,
            },
        }
    }

    pub fn validate(self) -> Result<()> {
        let values = [
            ("bm25", self.bm25),
            ("vector", self.vector),
            ("metadata", self.metadata),
        ];
        if let Some((field, value)) = values
            .iter()
            .find(|(_, value)| !value.is_finite() || *value < 0.0)
        {
            return Err(Error::new(
                ErrorCode::InvalidSearchWeights,
                "search weights must be finite and non-negative",
            )
            .with_detail("field", format!("weights.{field}"))
            .with_detail("reason", "finite_non_negative")
            .with_detail("weights_reason", "finite_non_negative")
            .with_detail("actual", value));
        }
        if values.iter().all(|(_, value)| *value == 0.0) {
            return Err(Error::new(
                ErrorCode::InvalidSearchWeights,
                "at least one search weight must be positive",
            )
            .with_detail("field", "weights")
            .with_detail("reason", "all_zero")
            .with_detail("weights_reason", "all_zero")
            .with_detail("actual_sum", 0.0));
        }
        let sum = self.bm25 + self.vector + self.metadata;
        if (sum - 1.0).abs() > Self::SUM_EPSILON {
            return Err(Error::new(
                ErrorCode::InvalidSearchWeights,
                "search weights must sum to 1.0",
            )
            .with_detail("field", "weights")
            .with_detail("reason", "sum_mismatch")
            .with_detail("weights_reason", "sum_mismatch")
            .with_detail("expected_sum", 1.0)
            .with_detail("actual_sum", sum));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default)]
pub struct InsertSummary {
    pub documents_inserted: usize,
    pub chunks_inserted: usize,
    pub embeddings_inserted: usize,
    pub documents_skipped: usize,
    pub chunks_skipped: usize,
    pub embeddings_skipped: usize,
    pub documents_replaced: usize,
    pub chunks_replaced: usize,
    pub embeddings_replaced: usize,
}

#[derive(Clone, Debug, Default)]
pub struct CollectionStats {
    pub tokenizer: crate::TokenizerConfig,
    pub collection: String,
    pub document_count: usize,
    pub chunk_count: usize,
    pub embedding_count: usize,
    pub tombstone_count: usize,
    pub text_index_terms: usize,
    pub text_index_segments: usize,
    pub vector_count: usize,
    pub index_size_bytes: u64,
    pub live_bytes: u64,
    pub dead_bytes_estimate: u64,
    pub resource_profile: ResourceProfile,
    pub cache_budget_mb: usize,
    pub last_commit_id: u64,
    pub pending_compaction_bytes: u64,
    pub format_version: u16,
    pub background_compaction_enabled: bool,
}

#[derive(Clone, Debug, Default)]
pub struct IndexStats {
    pub text_terms: usize,
    pub text_segments: usize,
    pub vector_count: usize,
    pub vector_segments: usize,
    pub rebuild_needed: bool,
    pub index_size_bytes: u64,
    pub last_index_commit_id: Option<u64>,
}

#[derive(Clone, Debug)]
pub struct DatabaseStats {
    pub collection_count: usize,
    pub document_count: usize,
    pub chunk_count: usize,
    pub embedding_count: usize,
    pub tombstone_count: usize,
    pub file_size: u64,
    pub index_size_bytes: u64,
    pub live_bytes: u64,
    pub dead_bytes_estimate: u64,
    pub active_profile: ResourceProfile,
    pub cache_budget_mb: usize,
    pub last_commit_id: u64,
    pub pending_compaction_bytes: u64,
    pub format_version: u16,
    pub background_compaction_enabled: bool,
}

#[derive(Clone)]
pub struct CollectionHandle {
    pub(crate) db: crate::db::Database,
    pub(crate) collection_id: String,
    pub(crate) name: String,
}

impl CollectionHandle {
    pub(crate) fn new(db: crate::db::Database, collection_id: String, name: String) -> Self {
        Self {
            db,
            collection_id,
            name,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn collection_id(&self) -> &str {
        &self.collection_id
    }

    pub fn insert_document(&self, input: DocumentInput) -> Result<InsertSummary> {
        self.db
            .insert_document_in_collection(&self.collection_id, input, OnConflict::Error)
    }

    pub fn put_embeddings(&self, embeddings: Vec<EmbeddingInput>) -> Result<InsertSummary> {
        self.db
            .put_embeddings_in_collection(&self.collection_id, embeddings, OnConflict::Error)
    }

    pub fn get_document(&self, document_id: &str) -> Result<Document> {
        self.db
            .get_document_in_collection(&self.collection_id, document_id)
    }

    pub fn get_document_by_external_id(&self, external_id: &str) -> Result<Document> {
        self.db
            .get_document_by_external_id_in_collection(&self.collection_id, external_id)
    }

    pub fn list_chunks(&self, document_id: &str) -> Result<Vec<Chunk>> {
        self.db
            .list_chunks_in_collection(&self.collection_id, document_id)
    }

    pub fn list_chunks_by_document_external_id(&self, external_id: &str) -> Result<Vec<Chunk>> {
        self.db
            .list_chunks_by_document_external_id_in_collection(&self.collection_id, external_id)
    }

    pub fn get_chunk(&self, chunk_id: &str) -> Result<Chunk> {
        self.db
            .get_chunk_in_collection(&self.collection_id, chunk_id)
    }

    pub fn get_chunk_by_external_id(&self, chunk_external_id: &str) -> Result<Chunk> {
        self.db
            .get_chunk_by_external_id_in_collection(&self.collection_id, chunk_external_id)
    }

    pub fn delete_document(&self, document_id: &str) -> Result<()> {
        self.db
            .delete_document_in_collection(&self.collection_id, document_id)
    }

    pub fn delete_chunk(&self, chunk_id: &str) -> Result<()> {
        self.db
            .delete_chunk_in_collection(&self.collection_id, chunk_id)
    }

    pub fn stats(&self) -> Result<CollectionStats> {
        self.db.collection_stats_by_id(&self.collection_id)
    }

    pub fn search(&self) -> crate::query::SearchBuilder {
        crate::query::SearchBuilder::new(self.clone())
    }
}

impl CollectionConfig {
    pub fn validate(&self) -> Result<()> {
        self.validate_in(&crate::json::JsonOperationContext::standalone())
            .map(|_| ())
    }

    pub(crate) fn validate_in(
        &self,
        context: &crate::json::JsonOperationContext,
    ) -> Result<crate::json::JsonReservation> {
        if self.name.trim().is_empty() {
            return Err(Error::new(
                ErrorCode::MissingRequiredField,
                "collection name cannot be empty",
            ));
        }
        if self.embedding_dim == 0 || u32::try_from(self.embedding_dim).is_err() {
            return Err(Error::new(
                ErrorCode::InvalidEmbeddingDimension,
                "embedding dimension must be in 1..=u32::MAX",
            ));
        }
        if self
            .embedding_model
            .as_ref()
            .map(|model| model.trim().is_empty())
            .unwrap_or(false)
        {
            return Err(Error::new(
                ErrorCode::MissingRequiredField,
                "embedding_model cannot be empty when provided",
            ));
        }
        if matches!(self.vector_index, VectorIndexConfig::Ann { .. }) {
            return Err(Error::new(
                ErrorCode::UnsupportedFeatureFlag,
                "ANN vector index configuration is reserved for a future version",
            ));
        }
        let mut guard = crate::json::JsonObjectGuard::new(
            crate::json::JsonInputKind::CollectionConfig,
            context,
        );
        guard.string_field("name", &self.name)?;
        match &self.embedding_model {
            Some(model) => guard.string_field("embedding_model", model)?,
            None => guard.value_field("embedding_model", &JsonValue::Null)?,
        }
        guard.value_field(
            "embedding_dim",
            &JsonValue::Number(self.embedding_dim as f64),
        )?;
        guard.value_field(
            "distance_metric",
            &JsonValue::String(self.distance_metric.as_str().into()),
        )?;
        guard.value_field(
            "vectors_normalized",
            &JsonValue::Bool(self.vectors_normalized),
        )?;
        guard.value_field("vector_index", &vector_index_to_json(&self.vector_index))?;
        guard.value_field("tokenizer", &self.tokenizer.to_json())?;
        Ok(guard.finish())
    }
}

impl SourceSpan {
    pub fn to_json(&self) -> JsonValue {
        self.build_json(&JsonBuilder::host())
            .expect("host span construction is infallible")
    }

    pub(crate) fn build_json(&self, builder: &JsonBuilder<'_>) -> Result<JsonValue> {
        builder.object([
            ("start", builder.optional_u64(self.start)?),
            ("end", builder.optional_u64(self.end)?),
            (
                "line_start",
                builder.optional_u64(self.line_start.map(u64::from))?,
            ),
            (
                "line_end",
                builder.optional_u64(self.line_end.map(u64::from))?,
            ),
        ])
    }

    pub fn from_json(value: &JsonValue) -> Result<Self> {
        if value.is_null() {
            return Ok(SourceSpan::default());
        }
        let obj = value.as_object().ok_or_else(|| {
            Error::new(
                ErrorCode::InvalidJsonlSchema,
                "source_span must be an object",
            )
        })?;
        let span = Self {
            start: optional_offset_field(obj, "start", "source_span")?,
            end: optional_offset_field(obj, "end", "source_span")?,
            line_start: optional_line_field(obj, "line_start", "source_span")?,
            line_end: optional_line_field(obj, "line_end", "source_span")?,
        };
        span.validate()?;
        Ok(span)
    }

    pub fn from_metadata(metadata: &JsonValue) -> Option<Self> {
        Self::try_from_metadata(metadata).ok().flatten()
    }

    pub fn try_from_metadata(metadata: &JsonValue) -> Result<Option<Self>> {
        let Some(obj) = metadata.as_object() else {
            return Ok(None);
        };
        let span = Self {
            start: optional_offset_field(obj, "byte_start", "metadata")?,
            end: optional_offset_field(obj, "byte_end", "metadata")?,
            line_start: optional_line_field(obj, "line_start", "metadata")?,
            line_end: optional_line_field(obj, "line_end", "metadata")?,
        };
        if span.start.is_some()
            || span.end.is_some()
            || span.line_start.is_some()
            || span.line_end.is_some()
        {
            span.validate()?;
            Ok(Some(span))
        } else {
            Ok(None)
        }
    }

    pub fn validate(&self) -> Result<()> {
        validate_span_pair(
            self.start,
            self.end,
            "source_span.start",
            "source_span.end",
            0,
        )?;
        validate_span_pair(
            self.line_start.map(u64::from),
            self.line_end.map(u64::from),
            "source_span.line_start",
            "source_span.line_end",
            1,
        )?;
        Ok(())
    }
}

impl CollectionInfo {
    pub fn to_json(&self) -> JsonValue {
        object_from_pairs([
            ("tokenizer", self.tokenizer.to_json()),
            (
                "config",
                object_from_pairs([("tokenizer", self.tokenizer.to_json())]),
            ),
            (
                "collection_id",
                JsonValue::String(self.collection_id.clone()),
            ),
            ("name", JsonValue::String(self.name.clone())),
            (
                "embedding_dim",
                JsonValue::Number(self.embedding_dim as f64),
            ),
            (
                "distance_metric",
                JsonValue::String(self.distance_metric.as_str().into()),
            ),
            ("embedding_model", opt_string(&self.embedding_model)),
            (
                "vectors_normalized",
                JsonValue::Bool(self.vectors_normalized),
            ),
            ("vector_index", vector_index_to_json(&self.vector_index)),
            ("created_at", JsonValue::String(self.created_at.clone())),
        ])
    }

    pub fn from_json(value: &JsonValue) -> Result<Self> {
        let obj = value.as_object().ok_or_else(|| {
            Error::new(
                ErrorCode::FileFormatError,
                "collection record must be an object",
            )
        })?;
        let config_tokenizer = match obj.get("config") {
            Some(config) => config
                .as_object()
                .ok_or_else(|| {
                    Error::new(
                        ErrorCode::FileFormatError,
                        "collection config must be an object",
                    )
                })?
                .get("tokenizer"),
            None => None,
        };
        if let (Some(persisted), Some(public)) = (config_tokenizer, obj.get("tokenizer")) {
            if persisted != public {
                return Err(Error::new(
                    ErrorCode::FileFormatError,
                    "collection tokenizer identity mismatch",
                ));
            }
        }
        Ok(Self {
            collection_id: required_string(obj, "collection_id")?,
            tokenizer: config_tokenizer
                .or_else(|| obj.get("tokenizer"))
                .map(crate::TokenizerConfig::from_json)
                .transpose()?
                .unwrap_or_default(),
            name: required_string(obj, "name")?,
            embedding_dim: required_usize(obj, "embedding_dim")?,
            distance_metric: DistanceMetric::parse(&required_string(obj, "distance_metric")?)?,
            embedding_model: optional_string(obj, "embedding_model"),
            vectors_normalized: obj
                .get("vectors_normalized")
                .and_then(JsonValue::as_bool)
                .unwrap_or(false),
            vector_index: obj
                .get("vector_index")
                .map(vector_index_from_json)
                .transpose()?
                .unwrap_or_default(),
            created_at: required_rfc3339_string(obj, "created_at")?,
        })
    }
}

impl Document {
    pub fn to_json(&self) -> JsonValue {
        self.build_json(self.metadata.clone(), &JsonBuilder::host())
            .expect("host DTO construction is infallible")
    }

    pub(crate) fn to_json_in(
        &self,
        context: &crate::json::JsonOperationContext,
    ) -> Result<(JsonValue, crate::json::JsonReservation)> {
        let (metadata, mut work) = self
            .metadata
            .try_clone_in(crate::json::JsonInputKind::StoredJson, context)?
            .into_parts();
        let value = self.build_json(
            metadata,
            &JsonBuilder::budgeted(&mut work, crate::json::JsonInputKind::StoredJson),
        )?;
        Ok((value, work))
    }

    fn build_json(&self, metadata: JsonValue, builder: &JsonBuilder<'_>) -> Result<JsonValue> {
        builder.object([
            ("document_id", builder.string(&self.document_id)?),
            ("collection_id", builder.string(&self.collection_id)?),
            ("external_id", builder.string(&self.external_id)?),
            ("title", builder.optional_str(self.title.as_deref())?),
            (
                "source_uri",
                builder.optional_str(self.source_uri.as_deref())?,
            ),
            ("metadata", metadata),
            ("created_at", builder.string(&self.created_at)?),
            ("updated_at", builder.string(&self.updated_at)?),
        ])
    }

    pub fn from_json(value: &JsonValue) -> Result<Self> {
        let obj = value.as_object().ok_or_else(|| {
            Error::new(
                ErrorCode::FileFormatError,
                "document record must be an object",
            )
        })?;
        Ok(Self {
            document_id: required_string(obj, "document_id")?,
            collection_id: required_string(obj, "collection_id")?,
            external_id: required_string(obj, "external_id")?,
            title: optional_string(obj, "title"),
            source_uri: optional_string(obj, "source_uri"),
            metadata: obj
                .get("metadata")
                .cloned()
                .map(ensure_object)
                .transpose()?
                .unwrap_or_else(JsonValue::object),
            created_at: required_rfc3339_string(obj, "created_at")?,
            updated_at: required_rfc3339_string(obj, "updated_at")?,
        })
    }
}

impl Chunk {
    pub fn to_json(&self) -> JsonValue {
        self.clone()
            .build_json(&JsonBuilder::host())
            .expect("host DTO construction is infallible")
    }

    pub(crate) fn into_json_retained(
        self,
        work: &mut crate::storage::RecordReservation,
    ) -> Result<JsonValue> {
        let (json, strings) = work.parts_mut();
        self.build_json(&JsonBuilder::budgeted_owned_strings(
            json,
            strings,
            crate::json::JsonInputKind::StoredJson,
        ))
    }

    fn build_json(self, builder: &JsonBuilder<'_>) -> Result<JsonValue> {
        builder.object([
            ("chunk_id", builder.owned_string(self.chunk_id)?),
            ("document_id", builder.owned_string(self.document_id)?),
            ("collection_id", builder.owned_string(self.collection_id)?),
            (
                "chunk_external_id",
                builder.owned_string(self.chunk_external_id)?,
            ),
            ("text", builder.owned_string(self.text)?),
            ("text_hash", builder.owned_string(self.text_hash)?),
            ("ordinal", JsonValue::Number(self.ordinal as f64)),
            ("token_count", JsonValue::Number(self.token_count as f64)),
            ("metadata", self.metadata),
            ("embedding_id", builder.optional_string(self.embedding_id)?),
            (
                "source_span",
                self.source_span
                    .as_ref()
                    .map(|span| span.build_json(builder))
                    .transpose()?
                    .unwrap_or(JsonValue::Null),
            ),
            ("created_at", builder.owned_string(self.created_at)?),
        ])
    }

    pub fn from_json(value: &JsonValue) -> Result<Self> {
        let obj = value.as_object().ok_or_else(|| {
            Error::new(ErrorCode::FileFormatError, "chunk record must be an object")
        })?;
        Ok(Self {
            chunk_id: required_string(obj, "chunk_id")?,
            document_id: required_string(obj, "document_id")?,
            collection_id: required_string(obj, "collection_id")?,
            chunk_external_id: required_string(obj, "chunk_external_id")?,
            text: required_string(obj, "text")?,
            text_hash: required_string(obj, "text_hash")?,
            ordinal: required_usize(obj, "ordinal")?,
            token_count: required_usize(obj, "token_count")?,
            metadata: obj
                .get("metadata")
                .cloned()
                .map(ensure_object)
                .transpose()?
                .unwrap_or_else(JsonValue::object),
            embedding_id: optional_string(obj, "embedding_id"),
            source_span: obj
                .get("source_span")
                .filter(|value| !value.is_null())
                .map(SourceSpan::from_json)
                .transpose()?,
            created_at: required_rfc3339_string(obj, "created_at")?,
        })
    }
}

impl Embedding {
    pub fn to_json(&self) -> JsonValue {
        object_from_pairs([
            ("embedding_id", JsonValue::String(self.embedding_id.clone())),
            ("chunk_id", JsonValue::String(self.chunk_id.clone())),
            (
                "collection_id",
                JsonValue::String(self.collection_id.clone()),
            ),
            (
                "chunk_external_id",
                JsonValue::String(self.chunk_external_id.clone()),
            ),
            ("model_name", JsonValue::String(self.model_name.clone())),
            ("dim", JsonValue::Number(self.dim as f64)),
            (
                "vector",
                JsonValue::Array(
                    self.vector
                        .iter()
                        .map(|value| JsonValue::Number(*value as f64))
                        .collect(),
                ),
            ),
            ("created_at", JsonValue::String(self.created_at.clone())),
        ])
    }

    pub fn from_json(value: &JsonValue) -> Result<Self> {
        let obj = value.as_object().ok_or_else(|| {
            Error::new(
                ErrorCode::FileFormatError,
                "embedding record must be an object",
            )
        })?;
        let vector = required_array(obj, "vector")?
            .iter()
            .map(|value| {
                let value = value.as_f64().ok_or_else(|| {
                    Error::new(
                        ErrorCode::InvalidVectorLength,
                        "embedding vector entries must be numbers",
                    )
                })?;
                let value = value as f32;
                if value.is_finite() {
                    Ok(value)
                } else {
                    Err(Error::new(
                        ErrorCode::InvalidVectorLength,
                        "embedding vector entries must be finite f32 values",
                    ))
                }
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            embedding_id: required_string(obj, "embedding_id")?,
            chunk_id: required_string(obj, "chunk_id")?,
            collection_id: required_string(obj, "collection_id")?,
            chunk_external_id: required_string(obj, "chunk_external_id")?,
            model_name: required_string(obj, "model_name")?,
            dim: required_usize(obj, "dim")?,
            vector,
            created_at: required_rfc3339_string(obj, "created_at")?,
        })
    }
}

impl Tombstone {
    #[cfg(test)]
    pub fn to_json(&self) -> JsonValue {
        object_from_pairs([
            ("target_type", JsonValue::String(self.target_type.clone())),
            ("target_id", JsonValue::String(self.target_id.clone())),
            ("deleted_at", JsonValue::String(self.deleted_at.clone())),
            ("reason", opt_string(&self.reason)),
        ])
    }

    pub fn from_json(value: &JsonValue) -> Result<Self> {
        let obj = value.as_object().ok_or_else(|| {
            Error::new(
                ErrorCode::FileFormatError,
                "tombstone record must be an object",
            )
        })?;
        Ok(Self {
            target_type: required_string(obj, "target_type")?,
            target_id: required_string(obj, "target_id")?,
            deleted_at: required_rfc3339_string(obj, "deleted_at")?,
            reason: optional_string(obj, "reason"),
        })
    }
}

impl DocumentInput {
    pub(crate) fn workspace_bytes(&self) -> usize {
        self.external_id.capacity()
            + self.title.as_ref().map_or(0, String::capacity)
            + self.source_uri.as_ref().map_or(0, String::capacity)
            + self.chunks.capacity() * std::mem::size_of::<ChunkInput>()
            + self
                .chunks
                .iter()
                .map(ChunkInput::workspace_bytes)
                .sum::<usize>()
    }

    pub fn validate(&self) -> Result<()> {
        self.validate_in(&crate::json::JsonOperationContext::standalone())
            .map(|_| ())
    }

    pub(crate) fn validate_in(
        &self,
        context: &crate::json::JsonOperationContext,
    ) -> Result<crate::json::JsonReservation> {
        self.validate_with_workspace(context, None)
    }

    pub(crate) fn validate_owned_in(
        &self,
        context: &crate::json::JsonOperationContext,
        workspace: &mut crate::json::WorkspaceReservation,
    ) -> Result<crate::json::JsonReservation> {
        self.validate_with_workspace(context, Some(workspace))
    }

    fn validate_with_workspace(
        &self,
        context: &crate::json::JsonOperationContext,
        workspace: Option<&mut crate::json::WorkspaceReservation>,
    ) -> Result<crate::json::JsonReservation> {
        let mut reservation = context.reservation();
        self.validate_with(
            context,
            &mut |value: &JsonValue, kind| {
                reservation.merge(value.guard_in(kind, context)?.0);
                Ok(())
            },
            workspace,
        )?;
        Ok(reservation)
    }

    // Only the owned JSONL path may reuse its retained, already guarded metadata.
    pub(crate) fn validate_preprocessed(
        &self,
        context: &crate::json::JsonOperationContext,
    ) -> Result<()> {
        self.validate_with(context, &mut |_, _| Ok(()), None)
    }

    fn validate_with(
        &self,
        context: &crate::json::JsonOperationContext,
        metadata_guard: &mut impl FnMut(&JsonValue, crate::json::JsonInputKind) -> Result<()>,
        workspace: Option<&mut crate::json::WorkspaceReservation>,
    ) -> Result<()> {
        if self.external_id.trim().is_empty() {
            return Err(Error::new(
                ErrorCode::MissingRequiredField,
                "document external_id is required",
            ));
        }
        metadata_guard(&self.metadata, crate::json::JsonInputKind::DocumentMetadata)
            .map_err(|err| err.with_detail("field", "document.metadata"))?;
        if self.metadata.as_object().is_none() {
            return Err(Error::new(
                ErrorCode::InvalidJsonlSchema,
                "metadata/config fields must be JSON objects",
            ));
        }
        if let Some(workspace) = workspace {
            let bytes = self.workspace_bytes();
            workspace.grow_to(bytes, bytes, "batch_prepare")?;
        }
        let duplicate_ordinal = first_duplicate_index_by(&self.chunks, context, |left, right| {
            left.ordinal.cmp(&right.ordinal)
        })?;
        let duplicate_external_id =
            first_duplicate_index_by(&self.chunks, context, |left, right| {
                left.chunk_external_id.cmp(&right.chunk_external_id)
            })?;
        for (index, chunk) in self.chunks.iter().enumerate() {
            chunk.validate_with(metadata_guard)?;
            if duplicate_ordinal == Some(index) {
                return Err(Error::new(
                    ErrorCode::InvalidJsonlSchema,
                    "chunk ordinal must be unique within a document",
                ));
            }
            if duplicate_external_id == Some(index) {
                return Err(Error::new(
                    ErrorCode::DuplicateExternalId,
                    format!(
                        "chunk_external_id {} is duplicated within the document input",
                        chunk.chunk_external_id
                    ),
                ));
            }
        }
        Ok(())
    }
}

pub(crate) fn first_duplicate_index_by<T>(
    values: &[T],
    context: &crate::json::JsonOperationContext,
    compare: impl Fn(&T, &T) -> std::cmp::Ordering,
) -> Result<Option<usize>> {
    if values.len() < 2 {
        return Ok(None);
    }
    let mut order =
        crate::json::ReservedVec::with_capacity(values.len(), context, "catalog_build")?;
    for index in 0..values.len() {
        order.push(index);
    }
    // Tie-break by input position so the second member is its first duplicate.
    // Return the earliest duplicate across groups, preserving sequential validation.
    order.as_mut_slice().sort_unstable_by(|left, right| {
        compare(&values[*left], &values[*right]).then_with(|| left.cmp(right))
    });
    Ok(order
        .as_slice()
        .windows(2)
        .filter(|pair| compare(&values[pair[0]], &values[pair[1]]).is_eq())
        .map(|pair| pair[1])
        .min())
}

impl ChunkInput {
    pub(crate) fn workspace_bytes(&self) -> usize {
        self.chunk_external_id.capacity() + self.text.capacity()
    }

    pub(crate) fn validate_preprocessed(&self) -> Result<()> {
        self.validate_with(&mut |_, _| Ok(()))
    }

    pub fn validate(&self) -> Result<()> {
        let context = crate::json::JsonOperationContext::standalone();
        self.validate_with(&mut |value: &JsonValue, kind| {
            value.guard_in(kind, &context).map(|_| ())
        })
    }

    fn validate_with(
        &self,
        metadata_guard: &mut impl FnMut(&JsonValue, crate::json::JsonInputKind) -> Result<()>,
    ) -> Result<()> {
        if u32::try_from(self.ordinal).is_err() {
            return Err(Error::new(
                ErrorCode::InvalidJsonlSchema,
                "chunk ordinal exceeds u32::MAX",
            )
            .with_detail("field", "ordinal"));
        }
        if self.chunk_external_id.trim().is_empty() {
            return Err(Error::new(
                ErrorCode::MissingRequiredField,
                "chunk_external_id is required",
            ));
        }
        if self.text.is_empty() {
            return Err(Error::new(
                ErrorCode::MissingRequiredField,
                "chunk text cannot be empty",
            ));
        }
        metadata_guard(&self.metadata, crate::json::JsonInputKind::ChunkMetadata)
            .map_err(|err| err.with_detail("field", "chunk.metadata"))?;
        if self.metadata.as_object().is_none() {
            return Err(Error::new(
                ErrorCode::InvalidJsonlSchema,
                "metadata/config fields must be JSON objects",
            ));
        }
        self.resolved_source_span()?;
        Ok(())
    }

    pub(crate) fn resolved_source_span(&self) -> Result<Option<SourceSpan>> {
        let span = match &self.source_span {
            Some(span) => {
                span.validate()?;
                Some(span.clone())
            }
            None => SourceSpan::try_from_metadata(&self.metadata)?,
        };
        Ok(span.filter(|span| span != &SourceSpan::default()))
    }
}

impl EmbeddingInput {
    pub(crate) fn workspace_bytes(&self) -> usize {
        self.chunk_external_id.capacity()
            + self.model_name.capacity()
            + self.vector.capacity() * std::mem::size_of::<f32>()
    }

    pub fn validate(&self, collection: &CollectionInfo) -> Result<()> {
        if self.chunk_external_id.trim().is_empty() {
            return Err(Error::new(
                ErrorCode::MissingRequiredField,
                "embedding chunk_external_id is required",
            ));
        }
        if self.model_name.trim().is_empty() {
            return Err(Error::new(
                ErrorCode::MissingRequiredField,
                "embedding model_name is required",
            ));
        }
        if self.dim != collection.embedding_dim || self.vector.len() != collection.embedding_dim {
            return Err(Error::new(
                ErrorCode::InvalidEmbeddingDimension,
                "embedding dimension does not match collection",
            )
            .with_detail("expected", collection.embedding_dim)
            .with_detail("actual_dim", self.dim)
            .with_detail("actual_len", self.vector.len()));
        }
        if self.vector.iter().any(|value| !value.is_finite()) {
            return Err(Error::new(
                ErrorCode::InvalidVectorLength,
                "embedding vector entries must be finite f32 values",
            ));
        }
        if let Some(expected) = &collection.embedding_model {
            if expected != &self.model_name {
                return Err(Error::new(
                    ErrorCode::EmbeddingModelMismatch,
                    "embedding model does not match collection",
                )
                .with_detail("expected", expected)
                .with_detail("actual", &self.model_name));
            }
        }
        Ok(())
    }
}

impl QueryEmbeddingInput {
    pub(crate) fn reserve_json_work_in(
        &self,
        context: &crate::json::JsonOperationContext,
    ) -> Result<crate::json::JsonReservation> {
        let mut work = context.reservation();
        work.grow(
            self.vector
                .capacity()
                .saturating_mul(std::mem::size_of::<f32>()),
            crate::json::JsonInputKind::QueryEmbedding,
        )?;
        if let Some(model) = &self.model_name {
            work.grow(model.capacity(), crate::json::JsonInputKind::JsonlRecord)?;
        }
        Ok(work)
    }

    pub fn from_vector(vector: Vec<f32>) -> Self {
        Self {
            dim: vector.len(),
            model_name: None,
            vector,
        }
    }

    pub fn validate(&self, collection: &CollectionInfo) -> Result<()> {
        crate::json::validate_query_vector(&self.vector)?;
        if self.dim != collection.embedding_dim || self.vector.len() != collection.embedding_dim {
            return Err(Error::new(
                ErrorCode::InvalidEmbeddingDimension,
                "query embedding dimension does not match collection",
            )
            .with_detail("expected", collection.embedding_dim)
            .with_detail("actual_dim", self.dim)
            .with_detail("actual_len", self.vector.len()));
        }
        if self.vector.iter().any(|value| !value.is_finite()) {
            return Err(Error::new(
                ErrorCode::InvalidVectorLength,
                "query embedding vector entries must be finite f32 values",
            ));
        }
        if self
            .model_name
            .as_ref()
            .map(|model| model.trim().is_empty())
            .unwrap_or(false)
        {
            return Err(Error::new(
                ErrorCode::MissingRequiredField,
                "query embedding model_name cannot be empty when provided",
            ));
        }
        if let Some(expected) = &collection.embedding_model {
            match &self.model_name {
                Some(actual) if actual == expected => {}
                Some(actual) => {
                    return Err(Error::new(
                        ErrorCode::EmbeddingModelMismatch,
                        "query embedding model does not match collection",
                    )
                    .with_detail("expected", expected)
                    .with_detail("actual", actual));
                }
                None => {
                    return Err(Error::new(
                        ErrorCode::MissingRequiredField,
                        "query embedding model_name is required for this collection",
                    )
                    .with_detail("expected", expected));
                }
            }
        }
        Ok(())
    }
}

impl CollectionStats {
    pub fn to_json(&self) -> JsonValue {
        object_from_pairs([
            ("tokenizer", self.tokenizer.to_json()),
            ("collection", JsonValue::String(self.collection.clone())),
            (
                "document_count",
                JsonValue::Number(self.document_count as f64),
            ),
            ("chunk_count", JsonValue::Number(self.chunk_count as f64)),
            (
                "embedding_count",
                JsonValue::Number(self.embedding_count as f64),
            ),
            (
                "tombstone_count",
                JsonValue::Number(self.tombstone_count as f64),
            ),
            (
                "deleted_count",
                JsonValue::Number(self.tombstone_count as f64),
            ),
            (
                "text_index_terms",
                JsonValue::Number(self.text_index_terms as f64),
            ),
            (
                "text_index_segments",
                JsonValue::Number(self.text_index_segments as f64),
            ),
            ("vector_count", JsonValue::Number(self.vector_count as f64)),
            (
                "index_size_bytes",
                JsonValue::Number(self.index_size_bytes as f64),
            ),
            ("live_bytes", JsonValue::Number(self.live_bytes as f64)),
            ("data_size_bytes", JsonValue::Number(self.live_bytes as f64)),
            (
                "dead_bytes_estimate",
                JsonValue::Number(self.dead_bytes_estimate as f64),
            ),
            (
                "resource_profile",
                JsonValue::String(self.resource_profile.as_json_str().into()),
            ),
            (
                "active_profile",
                JsonValue::String(self.resource_profile.as_json_str().into()),
            ),
            (
                "cache_budget_mb",
                JsonValue::Number(self.cache_budget_mb as f64),
            ),
            (
                "last_commit_id",
                JsonValue::Number(self.last_commit_id as f64),
            ),
            (
                "pending_compaction_bytes",
                JsonValue::Number(self.pending_compaction_bytes as f64),
            ),
            (
                "format_version",
                JsonValue::Number(self.format_version as f64),
            ),
            (
                "background_compaction_enabled",
                JsonValue::Bool(self.background_compaction_enabled),
            ),
        ])
    }
}

impl InsertSummary {
    pub fn to_json(&self) -> JsonValue {
        let skipped = self.documents_skipped + self.chunks_skipped + self.embeddings_skipped;
        let replaced = self.documents_replaced + self.chunks_replaced + self.embeddings_replaced;
        object_from_pairs([
            (
                "documents_inserted",
                JsonValue::Number(self.documents_inserted as f64),
            ),
            (
                "documents_imported",
                JsonValue::Number(self.documents_inserted as f64),
            ),
            (
                "chunks_inserted",
                JsonValue::Number(self.chunks_inserted as f64),
            ),
            (
                "chunks_imported",
                JsonValue::Number(self.chunks_inserted as f64),
            ),
            (
                "embeddings_inserted",
                JsonValue::Number(self.embeddings_inserted as f64),
            ),
            (
                "embeddings_imported",
                JsonValue::Number(self.embeddings_inserted as f64),
            ),
            (
                "documents_skipped",
                JsonValue::Number(self.documents_skipped as f64),
            ),
            (
                "chunks_skipped",
                JsonValue::Number(self.chunks_skipped as f64),
            ),
            (
                "embeddings_skipped",
                JsonValue::Number(self.embeddings_skipped as f64),
            ),
            ("skipped", JsonValue::Number(skipped as f64)),
            (
                "documents_replaced",
                JsonValue::Number(self.documents_replaced as f64),
            ),
            (
                "chunks_replaced",
                JsonValue::Number(self.chunks_replaced as f64),
            ),
            (
                "embeddings_replaced",
                JsonValue::Number(self.embeddings_replaced as f64),
            ),
            ("replaced", JsonValue::Number(replaced as f64)),
            ("missing_chunks", JsonValue::Number(0.0)),
            ("model_mismatches", JsonValue::Number(0.0)),
            ("errors", JsonValue::Number(0.0)),
        ])
    }
}

impl IndexStats {
    pub fn to_json(&self) -> JsonValue {
        object_from_pairs([
            ("text_terms", JsonValue::Number(self.text_terms as f64)),
            (
                "text_segments",
                JsonValue::Number(self.text_segments as f64),
            ),
            ("vector_count", JsonValue::Number(self.vector_count as f64)),
            (
                "vector_segments",
                JsonValue::Number(self.vector_segments as f64),
            ),
            ("rebuild_needed", JsonValue::Bool(self.rebuild_needed)),
            (
                "index_size_bytes",
                JsonValue::Number(self.index_size_bytes as f64),
            ),
            (
                "last_index_commit_id",
                self.last_index_commit_id
                    .map(|value| JsonValue::Number(value as f64))
                    .unwrap_or(JsonValue::Null),
            ),
        ])
    }
}

impl DatabaseStats {
    pub fn to_json(&self) -> JsonValue {
        object_from_pairs([
            (
                "collection_count",
                JsonValue::Number(self.collection_count as f64),
            ),
            (
                "document_count",
                JsonValue::Number(self.document_count as f64),
            ),
            ("chunk_count", JsonValue::Number(self.chunk_count as f64)),
            (
                "embedding_count",
                JsonValue::Number(self.embedding_count as f64),
            ),
            (
                "tombstone_count",
                JsonValue::Number(self.tombstone_count as f64),
            ),
            (
                "deleted_count",
                JsonValue::Number(self.tombstone_count as f64),
            ),
            ("file_size", JsonValue::Number(self.file_size as f64)),
            ("file_size_bytes", JsonValue::Number(self.file_size as f64)),
            (
                "index_size_bytes",
                JsonValue::Number(self.index_size_bytes as f64),
            ),
            ("live_bytes", JsonValue::Number(self.live_bytes as f64)),
            ("data_size_bytes", JsonValue::Number(self.live_bytes as f64)),
            (
                "dead_bytes_estimate",
                JsonValue::Number(self.dead_bytes_estimate as f64),
            ),
            (
                "active_profile",
                JsonValue::String(self.active_profile.as_json_str().into()),
            ),
            (
                "cache_budget_mb",
                JsonValue::Number(self.cache_budget_mb as f64),
            ),
            (
                "last_commit_id",
                JsonValue::Number(self.last_commit_id as f64),
            ),
            (
                "pending_compaction_bytes",
                JsonValue::Number(self.pending_compaction_bytes as f64),
            ),
            (
                "format_version",
                JsonValue::Number(self.format_version as f64),
            ),
            (
                "background_compaction_enabled",
                JsonValue::Bool(self.background_compaction_enabled),
            ),
        ])
    }
}

pub(crate) fn chunk_text_hash(text: &str) -> String {
    sha256_hex(text.as_bytes())
}

pub(crate) fn token_count_hint_in(
    text: &str,
    context: &crate::json::JsonOperationContext,
) -> Result<usize> {
    let mut count = 0usize;
    crate::tokenizer::visit_tokens_in(text, context, |_| {
        count = count
            .checked_add(1)
            .ok_or_else(|| Error::new(ErrorCode::FileFormatError, "text token count overflow"))?;
        Ok(())
    })?;
    Ok(count)
}

#[cfg(test)]
pub(crate) fn token_count_hint(text: &str) -> usize {
    crate::tokenizer::tokens(text).count()
}

pub(crate) fn opt_string(value: &Option<String>) -> JsonValue {
    value
        .as_ref()
        .map(|value| JsonValue::String(value.clone()))
        .unwrap_or(JsonValue::Null)
}

pub(crate) fn json_usize(value: &JsonValue) -> Option<usize> {
    value.as_f64().and_then(|value| {
        if value.is_finite()
            && value >= 0.0
            && value < (usize::MAX as u128 + 1) as f64
            && value.fract() == 0.0
        {
            Some(value as usize)
        } else {
            None
        }
    })
}

fn optional_offset_field(
    obj: &BTreeMap<String, JsonValue>,
    key: &str,
    container: &str,
) -> Result<Option<u64>> {
    let Some(value) = obj.get(key) else {
        return Ok(None);
    };
    if value.is_null() {
        return Ok(None);
    }
    let number = match value {
        JsonValue::String(text) if !text.is_empty() && text.bytes().all(|b| b.is_ascii_digit()) => {
            text.parse::<u64>().ok()
        }
        JsonValue::Number(number)
            if number.is_finite()
                && *number >= 0.0
                && *number < 18_446_744_073_709_551_616.0
                && number.fract() == 0.0 =>
        {
            Some(*number as u64)
        }
        _ => None,
    };
    number.map(Some).ok_or_else(|| {
        Error::new(
            ErrorCode::InvalidJsonlSchema,
            format!("{container}.{key} must be a u64 integer or unsigned decimal string"),
        )
        .with_detail("field", format!("{container}.{key}"))
    })
}

fn optional_line_field(
    obj: &BTreeMap<String, JsonValue>,
    key: &str,
    container: &str,
) -> Result<Option<u32>> {
    let Some(value) = obj.get(key).filter(|value| !value.is_null()) else {
        return Ok(None);
    };
    let number = value.as_f64().filter(|value| {
        value.is_finite() && *value >= 0.0 && *value <= u32::MAX as f64 && value.fract() == 0.0
    });
    number.map(|number| Some(number as u32)).ok_or_else(|| {
        Error::new(
            ErrorCode::InvalidJsonlSchema,
            format!("{container}.{key} must be a u32 integer"),
        )
        .with_detail("field", format!("{container}.{key}"))
    })
}

fn validate_span_pair(
    start: Option<u64>,
    end: Option<u64>,
    start_name: &str,
    end_name: &str,
    min_value: u64,
) -> Result<()> {
    match (start, end) {
        (Some(start), Some(end)) => {
            if start < min_value {
                return Err(Error::new(
                    ErrorCode::InvalidJsonlSchema,
                    format!("{start_name} must be at least {min_value}"),
                ));
            }
            if end < min_value {
                return Err(Error::new(
                    ErrorCode::InvalidJsonlSchema,
                    format!("{end_name} must be at least {min_value}"),
                ));
            }
            if start > end {
                return Err(Error::new(
                    ErrorCode::InvalidJsonlSchema,
                    format!("{start_name} must be <= {end_name}"),
                ));
            }
            Ok(())
        }
        (None, None) => Ok(()),
        (Some(_), None) => Err(Error::new(
            ErrorCode::InvalidJsonlSchema,
            format!("{end_name} is required when {start_name} is present"),
        )),
        (None, Some(_)) => Err(Error::new(
            ErrorCode::InvalidJsonlSchema,
            format!("{start_name} is required when {end_name} is present"),
        )),
    }
}

fn required_rfc3339_string(obj: &BTreeMap<String, JsonValue>, key: &str) -> Result<String> {
    let value = required_string(obj, key)?;
    if time::parse_rfc3339_seconds(&value).is_none() {
        return Err(Error::new(
            ErrorCode::FileFormatError,
            format!("{key} must be an RFC3339 timestamp"),
        ));
    }
    Ok(value)
}

pub(crate) fn vector_index_to_json(config: &VectorIndexConfig) -> JsonValue {
    match config {
        VectorIndexConfig::Exact => {
            object_from_pairs([("type", JsonValue::String("exact".into()))])
        }
        VectorIndexConfig::Ann { provider, params } => object_from_pairs([
            ("type", JsonValue::String("ann".into())),
            ("provider", JsonValue::String(provider.clone())),
            ("params", params.clone()),
        ]),
    }
}

pub(crate) fn vector_index_from_json(value: &JsonValue) -> Result<VectorIndexConfig> {
    let obj = value
        .as_object()
        .ok_or_else(|| Error::new(ErrorCode::FileFormatError, "vector_index must be an object"))?;
    match required_string(obj, "type")?.as_str() {
        "exact" => Ok(VectorIndexConfig::Exact),
        "ann" => Ok(VectorIndexConfig::Ann {
            provider: required_string(obj, "provider")?,
            params: obj.get("params").cloned().unwrap_or_else(JsonValue::object),
        }),
        other => Err(Error::new(
            ErrorCode::UnsupportedFeatureFlag,
            format!("unsupported vector index type {other}"),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budgeted_token_hints_preserve_empty_stop_words_and_symbol_counts() {
        let context = crate::json::JsonOperationContext::standalone();
        let available = context.available_workspace_bytes();
        for text in [
            "",
            "the and",
            "THE AND",
            "the_and",
            "fooFoo fooFoo",
            "FOO_BAR Foo_foo",
            "\u{23a}X_\u{130}",
            "\u{39f}\u{3a3}_\u{39f}\u{3c3}_\u{3bf}\u{3c3}",
        ] {
            assert_eq!(
                token_count_hint_in(text, &context).unwrap(),
                token_count_hint(text)
            );
            assert_eq!(context.available_workspace_bytes(), available);
        }
        assert_eq!(token_count_hint_in("the and", &context).unwrap(), 0);
        for text in ["", "the and", "rust rust", "foo_bar foo_bar"] {
            let context = crate::json::JsonOperationContext::new(std::sync::Arc::new(
                crate::json::WorkLedger::with_limits(0, 0, 0),
            ));
            assert_eq!(
                token_count_hint_in(text, &context).unwrap(),
                token_count_hint(text)
            );
            assert_eq!(context.available_workspace_bytes(), 0);
        }
    }

    #[test]
    fn budgeted_token_hints_admit_exact_payload_peak_on_both_axes_and_allow_retry() {
        let minimum = "PROJECT".len();
        for parent in [false, true] {
            for limit in [minimum - 1, minimum, minimum + 1] {
                let context = crate::json::JsonOperationContext::new(std::sync::Arc::new(
                    crate::json::WorkLedger::with_limits(
                        if parent { 2 * minimum } else { limit },
                        0,
                        if parent { limit } else { 2 * minimum },
                    ),
                ));
                let result = token_count_hint_in("rust PROJECT", &context);
                if limit < minimum {
                    let error = result.unwrap_err();
                    assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                    for (key, value) in [
                        ("resource", "memory".into()),
                        ("operation", "operation_scratch".into()),
                        ("limit_bytes", limit.to_string()),
                        ("requested_bytes", minimum.to_string()),
                    ] {
                        assert!(error.detail().contains(&(key.into(), value)), "{error:?}");
                    }
                } else {
                    assert_eq!(result.unwrap(), 2);
                }
                assert_eq!(context.available_workspace_bytes(), limit);
                assert_eq!(token_count_hint_in("rust", &context).unwrap(), 1);
                assert_eq!(context.available_workspace_bytes(), limit);
            }
        }
    }

    #[test]
    fn owned_chunk_dto_moves_payloads_and_matches_public_contract() {
        use crate::json::{JsonInputKind, JsonOperationContext, JsonOutputSchema};
        for seed in 0..100 {
            let context = JsonOperationContext::standalone();
            let metadata =
                object_from_pairs([("value", JsonValue::String(format!("value-{seed}")))]);
            let mut work = metadata
                .guard_in(JsonInputKind::StoredJson, &context)
                .unwrap()
                .0;
            let mut text = String::with_capacity(128 + seed);
            text.push_str("payload\n\u{4e2d}");
            let pointer = text.as_ptr();
            let chunk = Chunk {
                chunk_id: format!("chk_{seed}"),
                document_id: "doc_1".into(),
                collection_id: "col_1".into(),
                chunk_external_id: format!("chunk-{seed}"),
                text,
                text_hash: "hash".into(),
                ordinal: seed,
                token_count: 2,
                metadata,
                embedding_id: (seed % 2 == 0).then(|| "emb_1".into()),
                source_span: Some(SourceSpan {
                    start: Some(u64::MAX - 1),
                    end: Some(u64::MAX),
                    line_start: Some(1),
                    line_end: Some(2),
                }),
                created_at: "1970-01-01T00:00:00Z".into(),
            };
            let expected = object_from_pairs([
                ("chunk_id", JsonValue::String(chunk.chunk_id.clone())),
                ("document_id", JsonValue::String("doc_1".into())),
                ("collection_id", JsonValue::String("col_1".into())),
                (
                    "chunk_external_id",
                    JsonValue::String(chunk.chunk_external_id.clone()),
                ),
                ("text", JsonValue::String(chunk.text.clone())),
                ("text_hash", JsonValue::String("hash".into())),
                ("ordinal", JsonValue::Number(seed as f64)),
                ("token_count", JsonValue::Number(2.0)),
                ("metadata", chunk.metadata.clone()),
                ("embedding_id", opt_string(&chunk.embedding_id)),
                (
                    "source_span",
                    object_from_pairs([
                        ("start", JsonValue::String("18446744073709551614".into())),
                        ("end", JsonValue::String("18446744073709551615".into())),
                        ("line_start", JsonValue::Number(1.0)),
                        ("line_end", JsonValue::Number(2.0)),
                    ]),
                ),
                ("created_at", JsonValue::String(chunk.created_at.clone())),
            ]);
            assert_eq!(chunk.to_json(), expected);
            let value = chunk
                .build_json(&JsonBuilder::budgeted(
                    &mut work,
                    crate::json::JsonInputKind::StoredJson,
                ))
                .unwrap();
            assert_eq!(value, expected);
            assert_eq!(
                value.get("text").unwrap().as_str().unwrap().as_ptr(),
                pointer
            );
            let measured = value
                .reserve_retained_in(JsonInputKind::StoredJson, &context)
                .unwrap();
            assert_eq!(work.reserved_bytes(), measured.reserved_bytes());
            let (output, _) = value
                .pretty_retained_in(
                    JsonOutputSchema::Value(JsonInputKind::StoredJson),
                    &context,
                    &mut work,
                )
                .unwrap();
            assert_eq!(output, expected.to_pretty_string() + "\n");
        }
    }

    #[test]
    fn duplicate_scan_borrows_keys_and_reserves_only_its_index_array() {
        use crate::json::{JsonOperationContext, JsonProcessingPolicyV1};
        use std::cell::Cell;
        use std::collections::BTreeSet;

        let limit = JsonProcessingPolicyV1::WORK_BUDGET_BYTES;
        for seed in 0..100usize {
            let context = JsonOperationContext::standalone();
            let values = (0..seed + 2)
                .map(|index| format!("{}{}", "long-key".repeat(1024), (index * 37 + seed) % 23))
                .collect::<Vec<_>>();
            let mut seen = BTreeSet::new();
            let expected = values.iter().position(|value| !seen.insert(value.as_str()));
            let bytes = values.len() * std::mem::size_of::<usize>();
            let mut held = context.workspace_reservation();
            held.grow_to(limit - bytes + 1, limit - bytes + 1, "operation_scratch")
                .unwrap();
            let error = first_duplicate_index_by(&values, &context, |_, _| {
                panic!("comparison started before reserving the index array")
            })
            .unwrap_err();
            assert_eq!(error.code(), ErrorCode::ResourceExhausted);
            assert_eq!(error_detail_value(&error, "resource"), Some("memory"));
            assert_eq!(
                error_detail_value(&error, "operation"),
                Some("catalog_build")
            );
            assert_eq!(
                error_detail_value(&error, "limit_bytes"),
                Some(limit.to_string().as_str())
            );
            assert_eq!(
                error_detail_value(&error, "requested_bytes"),
                Some((limit + 1).to_string().as_str())
            );
            held.retain(limit - bytes);
            let checked = Cell::new(false);
            let actual = first_duplicate_index_by(&values, &context, |left, right| {
                if !checked.replace(true) {
                    assert!(context
                        .workspace_reservation()
                        .grow_to(1, 1, "operation_scratch")
                        .is_err());
                }
                left.cmp(right)
            })
            .unwrap();
            assert_eq!(actual, expected);
            assert!(checked.get());
            held.grow_to(limit, limit, "operation_scratch").unwrap();
            for short in [&values[..0], &values[..1]] {
                assert_eq!(
                    first_duplicate_index_by(short, &context, String::cmp).unwrap(),
                    None
                );
            }
        }
    }

    #[test]
    fn sorted_duplicate_checks_preserve_sequential_document_error_precedence() {
        use std::collections::BTreeSet;

        for seed in 0..100usize {
            let mut input = DocumentInput {
                external_id: "doc".into(),
                title: None,
                source_uri: None,
                metadata: JsonValue::object(),
                chunks: (0..12)
                    .map(|index| ChunkInput {
                        chunk_external_id: format!("chunk-{index}"),
                        text: "text".into(),
                        ordinal: index,
                        metadata: JsonValue::object(),
                        source_span: None,
                    })
                    .collect(),
            };
            input.chunks[3 + seed % 4].ordinal = 0;
            input.chunks[2 + (seed * 7) % 8].chunk_external_id = "chunk-0".into();
            if seed % 3 != 0 {
                input.chunks[(seed * 11) % 12].text.clear();
            }
            let sequential = (|| {
                let mut ordinals = BTreeSet::new();
                let mut external_ids = BTreeSet::new();
                for chunk in &input.chunks {
                    chunk.validate()?;
                    if !ordinals.insert(chunk.ordinal) {
                        return Err(Error::new(
                            ErrorCode::InvalidJsonlSchema,
                            "chunk ordinal must be unique within a document",
                        ));
                    }
                    if !external_ids.insert(chunk.chunk_external_id.as_str()) {
                        return Err(Error::new(
                            ErrorCode::DuplicateExternalId,
                            format!(
                                "chunk_external_id {} is duplicated within the document input",
                                chunk.chunk_external_id
                            ),
                        ));
                    }
                }
                Ok(())
            })()
            .unwrap_err();
            assert_eq!(
                input.validate().unwrap_err().to_json(),
                sequential.to_json()
            );
            assert_eq!(
                input
                    .validate_preprocessed(&crate::json::JsonOperationContext::standalone())
                    .unwrap_err()
                    .to_json(),
                sequential.to_json()
            );
        }
    }

    #[test]
    fn document_and_chunk_metadata_share_one_validation_work_budget() {
        let metadata =
            || object_from_pairs([("value", JsonValue::String("x".repeat(7 * 1024 * 1024)))]);
        let mut input = DocumentInput {
            external_id: "doc".into(),
            title: None,
            source_uri: None,
            metadata: metadata(),
            chunks: (0..3)
                .map(|ordinal| ChunkInput {
                    chunk_external_id: format!("chunk-{ordinal}"),
                    text: "text".into(),
                    ordinal,
                    metadata: metadata(),
                    source_span: None,
                })
                .collect(),
        };
        assert!(input.validate().is_ok());
        input.chunks.push(ChunkInput {
            chunk_external_id: "chunk-3".into(),
            text: "text".into(),
            ordinal: 3,
            metadata: metadata(),
            source_span: None,
        });
        let error = input.validate().unwrap_err();
        assert!(error
            .detail()
            .iter()
            .any(|(key, value)| key == "json_limit_reason" && value == "work_budget_exceeded"));
        assert!(error
            .detail()
            .iter()
            .any(|(key, value)| key == "json_input_kind" && value == "chunk_metadata"));
        input.chunks.pop();
        assert!(input.validate().is_ok());
    }

    fn collection() -> CollectionInfo {
        CollectionInfo {
            tokenizer: Default::default(),
            collection_id: "col_1".into(),
            name: "notes".into(),
            embedding_dim: 2,
            distance_metric: DistanceMetric::Cosine,
            embedding_model: Some("test-model".into()),
            vectors_normalized: false,
            vector_index: VectorIndexConfig::Exact,
            created_at: "1970-01-01T00:00:00Z".into(),
        }
    }

    fn error_detail_value<'a>(err: &'a Error, key: &str) -> Option<&'a str> {
        err.detail()
            .iter()
            .find(|(field, _)| field == key)
            .map(|(_, value)| value.as_str())
    }

    #[test]
    fn collection_policy_counts_the_complete_normalized_object_before_cloning() {
        let mut config = CollectionConfig {
            tokenizer: Default::default(),
            name: "a".into(),
            embedding_dim: 2,
            distance_metric: DistanceMetric::Cosine,
            embedding_model: Some("m".into()),
            vectors_normalized: false,
            vector_index: VectorIndexConfig::Exact,
        };
        let normalized = object_from_pairs([
            ("name", JsonValue::String(config.name.clone())),
            ("embedding_model", opt_string(&config.embedding_model)),
            ("embedding_dim", JsonValue::Number(2.0)),
            ("distance_metric", JsonValue::String("cosine".into())),
            ("vectors_normalized", JsonValue::Bool(false)),
            ("vector_index", vector_index_to_json(&config.vector_index)),
            ("tokenizer", config.tokenizer.to_json()),
        ]);
        let fixed_bytes = normalized.to_compact_string().len() - 2;
        let limit = crate::json::JsonProcessingPolicyV1::MAX_UNIT_CANONICAL_BYTES;
        config.name = "n".repeat(4 * 1_048_576);
        config.embedding_model = Some("m".repeat(limit - fixed_bytes - config.name.len()));
        config.validate().unwrap();
        config.name.push('n');
        let error = config.validate().unwrap_err();
        assert_eq!(
            error_detail_value(&error, "json_limit_reason"),
            Some("unit_bytes_exceeded")
        );
        assert_eq!(
            error_detail_value(&error, "json_input_kind"),
            Some("collection_config")
        );
        assert_eq!(
            error_detail_value(&error, "actual"),
            Some((limit + 1).to_string().as_str())
        );
        config.name = "\n".repeat(limit / 2);
        config.name.push('n');
        config.embedding_model = None;
        assert_eq!(
            error_detail_value(&config.validate().unwrap_err(), "json_limit_reason"),
            Some("unit_bytes_exceeded")
        );
        config.name = String::with_capacity(crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES);
        config.name.push('n');
        assert_eq!(
            error_detail_value(&config.validate().unwrap_err(), "json_limit_reason"),
            Some("work_budget_exceeded")
        );
    }

    #[test]
    fn document_input_rejects_duplicate_chunk_external_ids() {
        let input = DocumentInput {
            external_id: "doc".into(),
            title: None,
            source_uri: None,
            metadata: JsonValue::object(),
            chunks: vec![
                ChunkInput {
                    chunk_external_id: "doc#1".into(),
                    text: "first".into(),
                    ordinal: 1,
                    metadata: JsonValue::object(),
                    source_span: None,
                },
                ChunkInput {
                    chunk_external_id: "doc#1".into(),
                    text: "second".into(),
                    ordinal: 2,
                    metadata: JsonValue::object(),
                    source_span: None,
                },
            ],
        };
        assert_eq!(
            input.validate().unwrap_err().code(),
            ErrorCode::DuplicateExternalId
        );
    }

    #[test]
    fn source_span_json_is_exact_at_safe_integer_and_u64_boundaries() {
        for value in [
            0,
            9_007_199_254_740_991,
            9_007_199_254_740_992,
            9_007_199_254_740_993,
            u64::MAX,
        ] {
            let span = SourceSpan {
                start: Some(value),
                end: Some(value),
                line_start: Some(u32::MAX),
                line_end: Some(u32::MAX),
            };
            let json = span.to_json();
            assert_eq!(
                json.get("start").unwrap().as_str().is_some(),
                value > 9_007_199_254_740_991
            );
            assert_eq!(
                SourceSpan::from_json(&crate::json::parse(&json.to_compact_string()).unwrap())
                    .unwrap(),
                span
            );
        }
        let numeric =
            crate::json::parse(r#"{"start":9223372036854775808,"end":9223372036854775808}"#)
                .unwrap();
        assert_eq!(
            SourceSpan::from_json(&numeric).unwrap().start,
            Some(1u64 << 63)
        );
        let padded = crate::json::parse(r#"{"start":"0001","end":"0002"}"#).unwrap();
        assert_eq!(
            SourceSpan::from_json(&padded)
                .unwrap()
                .to_json()
                .get("start"),
            Some(&JsonValue::Number(1.0))
        );
    }

    #[test]
    fn source_span_rejects_invalid_decimal_offsets_and_line_overflow() {
        for raw in [
            "",
            "+1",
            "-1",
            "1.0",
            "1e3",
            " 1",
            "1 ",
            "18446744073709551616",
            "not-a-number",
        ] {
            let value = object_from_pairs([
                ("start", JsonValue::String(raw.into())),
                ("end", JsonValue::String(raw.into())),
            ]);
            assert_eq!(
                SourceSpan::from_json(&value).unwrap_err().code(),
                ErrorCode::InvalidJsonlSchema,
                "{raw}"
            );
        }
        for raw in [
            r#"{"start":18446744073709551616,"end":18446744073709551616}"#,
            r#"{"line_start":4294967296,"line_end":4294967296}"#,
            r#"{"line_start":"1","line_end":"2"}"#,
            r#"{"start":1.5,"end":2}"#,
            r#"{"start":"3","end":"2"}"#,
            r#"{"start":"18446744073709551615"}"#,
        ] {
            let value = crate::json::parse(raw).unwrap();
            assert!(SourceSpan::from_json(&value).is_err(), "{raw}");
        }
        assert!(
            crate::json::parse(r#"{"start":9007199254740993,"end":9007199254740993}"#).is_err()
        );
    }

    #[test]
    fn source_span_from_json_rejects_invalid_boundaries() {
        let inverted_bytes = object_from_pairs([
            ("start", JsonValue::Number(20.0)),
            ("end", JsonValue::Number(10.0)),
        ]);
        let err = SourceSpan::from_json(&inverted_bytes).unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
        assert!(err.message().contains("source_span.start"));

        let partial_lines = object_from_pairs([("line_end", JsonValue::Number(10.0))]);
        let err = SourceSpan::from_json(&partial_lines).unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
        assert!(err.message().contains("source_span.line_start"));

        let zero_line = object_from_pairs([
            ("line_start", JsonValue::Number(0.0)),
            ("line_end", JsonValue::Number(10.0)),
        ]);
        let err = SourceSpan::from_json(&zero_line).unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
        assert!(err.message().contains("at least 1"));
    }

    #[test]
    fn chunk_input_rejects_invalid_metadata_source_span() {
        let input = ChunkInput {
            chunk_external_id: "doc#1".into(),
            text: "first".into(),
            ordinal: 1,
            metadata: object_from_pairs([
                ("byte_start", JsonValue::Number(42.0)),
                ("byte_end", JsonValue::Number(7.0)),
            ]),
            source_span: None,
        };

        let err = input.validate().unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
        assert!(err.message().contains("source_span.start"));
    }

    #[test]
    fn source_span_accepts_valid_line_only_metadata() {
        let metadata = object_from_pairs([
            ("line_start", JsonValue::Number(3.0)),
            ("line_end", JsonValue::Number(8.0)),
        ]);

        let span = SourceSpan::try_from_metadata(&metadata)
            .unwrap()
            .expect("line metadata should produce source span");
        assert_eq!(span.start, None);
        assert_eq!(span.end, None);
        assert_eq!(span.line_start, Some(3));
        assert_eq!(span.line_end, Some(8));
    }

    #[test]
    fn embedding_input_rejects_non_finite_vectors() {
        let input = EmbeddingInput {
            chunk_external_id: "doc#1".into(),
            model_name: "test-model".into(),
            dim: 2,
            vector: vec![1.0, f32::NAN],
        };
        assert_eq!(
            input.validate(&collection()).unwrap_err().code(),
            ErrorCode::InvalidVectorLength
        );
    }

    #[test]
    fn embedding_input_rejects_blank_model_name() {
        let input = EmbeddingInput {
            chunk_external_id: "doc#1".into(),
            model_name: " \t ".into(),
            dim: 2,
            vector: vec![1.0, 0.0],
        };
        let mut collection = collection();
        collection.embedding_model = None;

        let err = input.validate(&collection).unwrap_err();
        assert_eq!(err.code(), ErrorCode::MissingRequiredField);
        assert!(err.message().contains("model_name"));
    }

    #[test]
    fn query_embedding_rejects_blank_model_name_when_provided() {
        let input = QueryEmbeddingInput {
            model_name: Some(" ".into()),
            dim: 2,
            vector: vec![1.0, 0.0],
        };
        let mut collection = collection();
        collection.embedding_model = None;

        let err = input.validate(&collection).unwrap_err();
        assert_eq!(err.code(), ErrorCode::MissingRequiredField);
        assert!(err.message().contains("model_name"));
    }

    #[test]
    fn search_weights_must_sum_to_one() {
        let err = SearchWeights::new(2.0, 1.0, 0.0).unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidSearchWeights);
        assert!(err.message().contains("sum to 1.0"));
        assert_eq!(error_detail_value(&err, "field"), Some("weights"));
        assert_eq!(error_detail_value(&err, "reason"), Some("sum_mismatch"));
        assert_eq!(
            error_detail_value(&err, "weights_reason"),
            Some("sum_mismatch")
        );
        assert_eq!(error_detail_value(&err, "expected_sum"), Some("1"));
        assert_eq!(error_detail_value(&err, "actual_sum"), Some("3"));
    }

    #[test]
    fn search_weights_validation_errors_include_machine_readable_detail() {
        let negative = SearchWeights::new(-0.1, 1.1, 0.0).unwrap_err();
        assert_eq!(negative.code(), ErrorCode::InvalidSearchWeights);
        assert_eq!(error_detail_value(&negative, "field"), Some("weights.bm25"));
        assert_eq!(
            error_detail_value(&negative, "reason"),
            Some("finite_non_negative")
        );
        assert_eq!(error_detail_value(&negative, "actual"), Some("-0.1"));

        let zero = SearchWeights::new(0.0, 0.0, 0.0).unwrap_err();
        assert_eq!(zero.code(), ErrorCode::InvalidSearchWeights);
        assert_eq!(error_detail_value(&zero, "field"), Some("weights"));
        assert_eq!(error_detail_value(&zero, "reason"), Some("all_zero"));
        assert_eq!(error_detail_value(&zero, "actual_sum"), Some("0"));
    }

    #[test]
    fn weights_reason_covers_non_finite_negative_zero_and_sum_errors() {
        for seed in 0..100 {
            let mut values = [0.25, 0.25, 0.5];
            let reason = match seed % 6 {
                0..=3 => {
                    values[seed / 6 % 3] = match seed % 6 {
                        0 => f64::NAN,
                        1 => f64::INFINITY,
                        2 => f64::NEG_INFINITY,
                        _ => -(seed as f64 + 1.0),
                    };
                    "finite_non_negative"
                }
                4 => {
                    values = [0.0, -0.0, 0.0];
                    "all_zero"
                }
                _ => {
                    values[seed / 6 % 3] = seed as f64 + 1.0;
                    "sum_mismatch"
                }
            };
            let weights = SearchWeights {
                bm25: values[0],
                vector: values[1],
                metadata: values[2],
            };
            for error in [
                weights.validate().unwrap_err(),
                SearchWeights::new(values[0], values[1], values[2]).unwrap_err(),
            ] {
                assert_eq!(error.code(), ErrorCode::InvalidSearchWeights);
                assert_eq!(error_detail_value(&error, "weights_reason"), Some(reason));
                assert_eq!(error_detail_value(&error, "reason"), Some(reason));
                let json = crate::json::parse(&error.to_json().to_compact_string()).unwrap();
                assert_eq!(
                    json.get("detail")
                        .unwrap()
                        .get("weights_reason")
                        .and_then(JsonValue::as_str),
                    Some(reason)
                );
            }
        }
    }

    #[test]
    fn collection_config_rejects_blank_embedding_model() {
        let config = CollectionConfig {
            tokenizer: Default::default(),
            name: "notes".into(),
            embedding_dim: 2,
            distance_metric: DistanceMetric::Cosine,
            embedding_model: Some("  ".into()),
            vectors_normalized: false,
            vector_index: VectorIndexConfig::Exact,
        };

        let err = config.validate().unwrap_err();
        assert_eq!(err.code(), ErrorCode::MissingRequiredField);
        assert!(err.message().contains("embedding_model"));
    }

    #[test]
    fn collection_config_rejects_reserved_ann_config() {
        let config = CollectionConfig {
            tokenizer: Default::default(),
            name: "notes".into(),
            embedding_dim: 2,
            distance_metric: DistanceMetric::Cosine,
            embedding_model: None,
            vectors_normalized: false,
            vector_index: VectorIndexConfig::Ann {
                provider: "reserved".into(),
                params: JsonValue::object(),
            },
        };
        assert_eq!(
            config.validate().unwrap_err().code(),
            ErrorCode::UnsupportedFeatureFlag
        );
    }

    #[test]
    fn collection_record_rejects_invalid_timestamp() {
        let value = object_from_pairs([
            ("collection_id", JsonValue::String("col_1".into())),
            ("name", JsonValue::String("notes".into())),
            ("embedding_dim", JsonValue::Number(2.0)),
            ("distance_metric", JsonValue::String("cosine".into())),
            ("embedding_model", JsonValue::Null),
            ("vectors_normalized", JsonValue::Bool(false)),
            (
                "vector_index",
                vector_index_to_json(&VectorIndexConfig::Exact),
            ),
            ("created_at", JsonValue::String("not-a-time".into())),
        ]);
        assert_eq!(
            CollectionInfo::from_json(&value).unwrap_err().code(),
            ErrorCode::FileFormatError
        );
    }
}
