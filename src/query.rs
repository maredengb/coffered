use crate::error::{Error, ErrorCode, Result};
#[cfg(test)]
use crate::index::text::Bm25Hit;
use crate::index::text::MatchedTerm;
#[cfg(test)]
use crate::json::object_from_pairs;
use crate::json::{JsonBuilder, JsonInputKind, JsonReservation, JsonValue};
use crate::metadata::MetadataBoostConfig;
use crate::model::{Chunk, CollectionHandle, QueryEmbeddingInput, SearchWeights};
use crate::profile::ResourceProfile;

pub(crate) const MAX_SEARCH_TOP_K: usize = if usize::BITS > 53 {
    9_007_199_254_740_991u64 as usize
} else {
    usize::MAX
};

pub(crate) fn validate_search_top_k(top_k: usize) -> Result<()> {
    if top_k == 0 || top_k > MAX_SEARCH_TOP_K {
        return Err(Error::new(
            ErrorCode::InvalidSearchParams,
            "top_k must be a positive integer in the supported exact JSON range",
        )
        .with_detail("field", "top_k")
        .with_detail("min", 1)
        .with_detail("max", MAX_SEARCH_TOP_K)
        .with_detail("actual", top_k));
    }
    Ok(())
}

#[derive(Clone)]
pub struct SearchBuilder {
    collection: CollectionHandle,
    text: Option<String>,
    vector: Option<QueryEmbeddingInput>,
    filter: JsonValue,
    weights: Option<SearchWeights>,
    metadata_boost_enabled: bool,
    metadata_boost_config: Option<MetadataBoostConfig>,
    top_k: usize,
    resource_profile: Option<ResourceProfile>,
}

#[derive(Clone, Debug)]
pub struct SearchResults {
    pub schema_version: String,
    pub collection: String,
    pub query: Option<String>,
    pub top_k: usize,
    pub profile: ResourceProfile,
    pub query_context: SearchQueryContext,
    pub results: Vec<SearchResult>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SearchExecutionStats {
    pub vector_scan_count: usize,
    pub distance_ops: usize,
    pub fusion_candidate_count: usize,
}

#[derive(Clone, Debug)]
pub struct SearchResult {
    pub rank: usize,
    pub chunk_id: String,
    pub chunk_external_id: String,
    pub document_id: String,
    pub document_external_id: String,
    pub snippet: String,
    pub document_metadata: JsonValue,
    pub chunk_metadata: JsonValue,
    pub final_score: f64,
    pub bm25_score: Option<f64>,
    pub vector_score: Option<f64>,
    pub metadata_boost: f64,
    pub normalized_scores: NormalizedScores,
    pub explain: JsonValue,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct NormalizedScores {
    pub bm25_norm: f64,
    pub vector_norm: Option<f64>,
    pub metadata_boost: f64,
}

#[derive(Clone, Debug)]
pub struct WeightContext {
    pub requested: SearchWeights,
    pub effective: SearchWeights,
    pub source: String,
    pub renormalization_reason: Option<String>,
}

#[derive(Clone, Debug)]
pub struct SearchQueryContext {
    pub text_present: bool,
    pub embedding_present: bool,
    pub filter_present: bool,
    pub weights: WeightContext,
    pub metadata_boost_present: bool,
    pub metadata_boost_enabled: bool,
    pub boost_config_hash: Option<String>,
    pub boost_rule_count: usize,
    pub top_k: usize,
    pub profile: ResourceProfile,
    pub candidate_limits: CandidateLimits,
    pub candidate_counts: CandidateLimits,
    pub execution_stats: SearchExecutionStats,
    pub index_source: String,
    pub index_rebuild_reason: Option<String>,
    pub transient_rebuild: TransientRebuildContext,
}

#[derive(Clone, Debug)]
pub struct SearchQueryContextInput {
    pub text_present: bool,
    pub embedding_present: bool,
    pub filter_present: bool,
    pub weights: WeightContext,
    pub metadata_boost_present: bool,
    pub metadata_boost_enabled: bool,
    pub boost_config_hash: Option<String>,
    pub boost_rule_count: usize,
    pub top_k: usize,
    pub profile: ResourceProfile,
    pub candidate_limits: CandidateLimits,
    pub cache_budget_mb: usize,
    pub chunks_scanned: usize,
    pub bytes_read: Option<u64>,
}

#[derive(Clone, Debug)]
pub struct TransientRebuildContext {
    pub started: bool,
    pub completed: bool,
    pub cancelled: bool,
    pub budget_mb: usize,
    pub chunks_scanned: Option<usize>,
    pub bytes_read: Option<u64>,
    pub budget_exceeded: bool,
}

#[derive(Clone, Debug)]
pub struct ExplainInput<'a> {
    pub collection_name: &'a str,
    pub chunk: &'a Chunk,
    pub document_external_id: &'a str,
    pub document_metadata: &'a JsonValue,
    pub document_source_uri: Option<&'a str>,
    pub bm25_recalled: bool,
    pub vector_recalled: bool,
    pub normalization_candidate_count: usize,
    pub bm25_raw: Option<f64>,
    pub bm25_norm: f64,
    pub bm25_max: Option<f64>,
    pub vector_raw: Option<f64>,
    pub vector_norm: Option<f64>,
    pub metadata_boost: f64,
    pub final_score: f64,
    pub matched_terms: &'a [MatchedTerm],
    pub weights: &'a WeightContext,
    pub resource_profile: ResourceProfile,
    pub candidate_limits: CandidateLimits,
    pub vector_scan_count: usize,
    pub distance_ops: usize,
    pub fusion_candidate_count: usize,
    pub index_source: &'a str,
    pub index_rebuild_reason: Option<&'a str>,
    pub transient_rebuild: TransientRebuildContext,
    pub distance_metric: &'a str,
    pub vectors_normalized: bool,
    pub filter: JsonValue,
    pub metadata_boost_present: bool,
    pub metadata_boost_enabled: bool,
    pub boost_config_hash: Option<&'a str>,
    pub boost_rule_count: usize,
    pub matched_boost_rule_count: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct CandidateLimits {
    pub bm25: usize,
    pub vector: usize,
    pub fusion: usize,
}

#[derive(Clone, Debug)]
pub struct WhyNotDiagnosis {
    pub schema_version: String,
    pub diagnosis_scope: String,
    pub debug_full_scan: bool,
    pub resource_profile: ResourceProfile,
    pub query_text: Option<String>,
    pub query_context: SearchQueryContext,
    pub result: JsonValue,
}

impl SearchBuilder {
    pub fn new(collection: CollectionHandle) -> Self {
        Self {
            collection,
            text: None,
            vector: None,
            filter: JsonValue::object(),
            weights: None,
            metadata_boost_enabled: false,
            metadata_boost_config: None,
            top_k: 10,
            resource_profile: None,
        }
    }

    pub fn text(mut self, text: impl Into<String>) -> Self {
        self.text = Some(text.into());
        self
    }

    pub fn vector(mut self, vector: Vec<f32>) -> Self {
        self.vector = Some(QueryEmbeddingInput::from_vector(vector));
        self
    }

    pub fn query_embedding(mut self, embedding: QueryEmbeddingInput) -> Self {
        self.vector = Some(embedding);
        self
    }

    pub fn filter(mut self, filter: JsonValue) -> Self {
        self.filter = filter;
        self
    }

    pub fn weights(mut self, weights: SearchWeights) -> Self {
        self.weights = Some(weights);
        self
    }

    pub fn metadata_boost_enabled(mut self, enabled: bool) -> Self {
        self.metadata_boost_enabled = enabled;
        self
    }

    pub fn metadata_boost(mut self, config: MetadataBoostConfig) -> Self {
        self.metadata_boost_enabled = true;
        self.metadata_boost_config = Some(config);
        self
    }

    pub fn top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }

    pub fn resource_profile(mut self, profile: ResourceProfile) -> Self {
        self.resource_profile = Some(profile);
        self
    }

    pub fn execute(self) -> Result<SearchResults> {
        self.execute_with_json_context(None)
    }

    pub(crate) fn execute_with_json_context(
        self,
        context: Option<crate::json::JsonOperationContext>,
    ) -> Result<SearchResults> {
        self.execute_retained(context).map(|(results, _)| results)
    }

    pub(crate) fn execute_retained(
        self,
        context: Option<crate::json::JsonOperationContext>,
    ) -> Result<(SearchResults, crate::storage::RecordReservation)> {
        self.collection.db.search_in_collection(
            &self.collection.collection_id,
            self.text.as_deref(),
            self.vector.as_ref(),
            &self.filter,
            self.weights,
            self.metadata_boost_enabled,
            self.metadata_boost_config.as_ref(),
            self.top_k,
            self.resource_profile,
            context,
        )
    }
}

impl SearchResults {
    pub fn to_json(&self) -> JsonValue {
        self.clone().into_json()
    }

    pub(crate) fn into_json(self) -> JsonValue {
        self.build_json(&JsonBuilder::host())
            .expect("host DTO construction is infallible")
    }

    pub(crate) fn into_json_retained(
        mut self,
        work: &mut crate::storage::RecordReservation,
    ) -> Result<JsonValue> {
        let wrapper_bytes = self.wrapper_workspace_bytes();
        let rows = std::mem::take(&mut self.results);
        let results = {
            let (json, strings) = work.parts_mut();
            let builder =
                JsonBuilder::budgeted_owned_strings(json, strings, JsonInputKind::StoredJson);
            builder.array(rows.into_iter().map(|result| result.build_json(&builder)))?
        };
        // The old result array is now freed; its strings moved into JSON without copying.
        work.retain_workspace(wrapper_bytes);
        let value = {
            let (json, strings) = work.parts_mut();
            self.build_json_fields(
                results,
                &JsonBuilder::budgeted_owned_strings(json, strings, JsonInputKind::StoredJson),
            )
        };
        work.retain_workspace(0);
        value
    }

    pub(crate) fn wrapper_workspace_bytes(&self) -> usize {
        self.schema_version.capacity()
            + self.collection.capacity()
            + self.query.as_ref().map_or(0, String::capacity)
            + self.query_context.workspace_bytes()
    }

    pub(crate) fn discard_rows(&mut self, work: &mut crate::storage::RecordReservation) {
        drop(std::mem::take(&mut self.results));
        work.json.retain(0);
        work.retain_workspace(self.wrapper_workspace_bytes());
    }

    fn build_json(mut self, builder: &JsonBuilder<'_>) -> Result<JsonValue> {
        let results = builder.array(
            std::mem::take(&mut self.results)
                .into_iter()
                .map(|result| result.build_json(builder)),
        )?;
        self.build_json_fields(results, builder)
    }

    fn build_json_fields(self, results: JsonValue, builder: &JsonBuilder<'_>) -> Result<JsonValue> {
        builder.object([
            ("schema_version", builder.owned_string(self.schema_version)?),
            ("collection", builder.owned_string(self.collection)?),
            ("query", builder.optional_string(self.query)?),
            ("top_k", JsonValue::Number(self.top_k as f64)),
            ("profile", builder.string(self.profile.as_json_str())?),
            (
                "candidate_counts",
                candidate_limits_json_in(self.query_context.candidate_counts, builder)?,
            ),
            (
                "execution_stats",
                self.query_context.execution_stats.build_json(builder)?,
            ),
            ("query_context", self.query_context.build_json(builder)?),
            ("results", results),
        ])
    }
}

impl SearchResult {
    pub fn to_json(&self) -> JsonValue {
        self.clone()
            .build_json(&JsonBuilder::host())
            .expect("host DTO construction is infallible")
    }

    fn build_json(self, builder: &JsonBuilder<'_>) -> Result<JsonValue> {
        builder.object([
            ("rank", JsonValue::Number(self.rank as f64)),
            ("chunk_id", builder.owned_string(self.chunk_id)?),
            (
                "chunk_external_id",
                builder.owned_string(self.chunk_external_id)?,
            ),
            ("document_id", builder.owned_string(self.document_id)?),
            (
                "document_external_id",
                builder.owned_string(self.document_external_id)?,
            ),
            ("snippet", builder.owned_string(self.snippet)?),
            ("document_metadata", self.document_metadata),
            ("chunk_metadata", self.chunk_metadata),
            ("final_score", JsonValue::Number(self.final_score)),
            (
                "bm25_score",
                self.bm25_score
                    .map(JsonValue::Number)
                    .unwrap_or(JsonValue::Null),
            ),
            (
                "vector_score",
                self.vector_score
                    .map(JsonValue::Number)
                    .unwrap_or(JsonValue::Null),
            ),
            ("metadata_boost", JsonValue::Number(self.metadata_boost)),
            (
                "normalized_scores",
                builder.object([
                    ("bm25", JsonValue::Number(self.normalized_scores.bm25_norm)),
                    (
                        "vector",
                        self.normalized_scores
                            .vector_norm
                            .map(JsonValue::Number)
                            .unwrap_or(JsonValue::Null),
                    ),
                    (
                        "metadata_boost",
                        JsonValue::Number(self.normalized_scores.metadata_boost),
                    ),
                ])?,
            ),
            ("explain", self.explain),
        ])
    }
}

impl SearchQueryContext {
    pub fn transient_from_committed_snapshot(input: SearchQueryContextInput) -> Self {
        Self::from_checkpoint_with(input, false, |value| Ok(value.to_owned()))
            .expect("host DTO construction is infallible")
    }

    pub fn persistent_checkpoint(input: SearchQueryContextInput) -> Self {
        Self::from_checkpoint_with(input, true, |value| Ok(value.to_owned()))
            .expect("host DTO construction is infallible")
    }

    pub(crate) fn from_checkpoint_with(
        input: SearchQueryContextInput,
        persistent: bool,
        mut copy: impl FnMut(&str) -> Result<String>,
    ) -> Result<Self> {
        Ok(Self {
            text_present: input.text_present,
            embedding_present: input.embedding_present,
            filter_present: input.filter_present,
            weights: input.weights,
            metadata_boost_present: input.metadata_boost_present,
            metadata_boost_enabled: input.metadata_boost_enabled,
            boost_config_hash: input.boost_config_hash,
            boost_rule_count: input.boost_rule_count,
            top_k: input.top_k,
            profile: input.profile,
            candidate_limits: input.candidate_limits,
            candidate_counts: CandidateLimits {
                bm25: 0,
                vector: 0,
                fusion: 0,
            },
            execution_stats: SearchExecutionStats::default(),
            index_source: copy(if persistent {
                "persistent_checkpoint"
            } else {
                "transient_rebuild"
            })?,
            index_rebuild_reason: (!persistent)
                .then(|| copy("in_memory_index_from_committed_snapshot"))
                .transpose()?,
            transient_rebuild: TransientRebuildContext {
                started: !persistent,
                completed: !persistent,
                cancelled: false,
                budget_mb: input.cache_budget_mb,
                chunks_scanned: (!persistent).then_some(input.chunks_scanned),
                bytes_read: if persistent { None } else { input.bytes_read },
                budget_exceeded: false,
            },
        })
    }

    pub(crate) fn workspace_bytes(&self) -> usize {
        self.weights.source.capacity()
            + self
                .weights
                .renormalization_reason
                .as_ref()
                .map_or(0, String::capacity)
            + self.boost_config_hash.as_ref().map_or(0, String::capacity)
            + self.index_source.capacity()
            + self
                .index_rebuild_reason
                .as_ref()
                .map_or(0, String::capacity)
    }

    pub fn to_json(&self) -> JsonValue {
        self.clone()
            .build_json(&JsonBuilder::host())
            .expect("host DTO construction is infallible")
    }

    pub(crate) fn build_json(self, builder: &JsonBuilder<'_>) -> Result<JsonValue> {
        builder.object([
            ("text_present", JsonValue::Bool(self.text_present)),
            (
                "tokenizer",
                if self.text_present {
                    crate::TokenizerConfig::default().build_json(builder)?
                } else {
                    JsonValue::Null
                },
            ),
            ("embedding_present", JsonValue::Bool(self.embedding_present)),
            ("filter_present", JsonValue::Bool(self.filter_present)),
            (
                "weights",
                builder.object([
                    (
                        "present",
                        JsonValue::Bool(self.weights.source == "user_override"),
                    ),
                    ("source", builder.string(&self.weights.source)?),
                    (
                        "requested",
                        weights_json_in(self.weights.requested, builder)?,
                    ),
                    (
                        "effective",
                        weights_json_in(self.weights.effective, builder)?,
                    ),
                    (
                        "renormalization_reason",
                        self.weights
                            .renormalization_reason
                            .as_deref()
                            .map(|reason| builder.string(reason))
                            .transpose()?
                            .unwrap_or_default(),
                    ),
                ])?,
            ),
            (
                "requested_weights",
                weights_json_in(self.weights.requested, builder)?,
            ),
            (
                "effective_weights",
                weights_json_in(self.weights.effective, builder)?,
            ),
            ("weights_source", builder.owned_string(self.weights.source)?),
            (
                "renormalization_reason",
                builder.optional_string(self.weights.renormalization_reason)?,
            ),
            (
                "metadata_boost_present",
                JsonValue::Bool(self.metadata_boost_present),
            ),
            (
                "metadata_boost_enabled",
                JsonValue::Bool(self.metadata_boost_enabled),
            ),
            (
                "boost_config_hash",
                builder.optional_string(self.boost_config_hash)?,
            ),
            (
                "boost_rule_count",
                JsonValue::Number(self.boost_rule_count as f64),
            ),
            ("top_k", JsonValue::Number(self.top_k as f64)),
            ("profile", builder.string(self.profile.as_json_str())?),
            (
                "candidate_limits",
                candidate_limits_json_in(self.candidate_limits, builder)?,
            ),
            ("execution_stats", self.execution_stats.build_json(builder)?),
            ("index_source", builder.owned_string(self.index_source)?),
            (
                "index_rebuild_reason",
                builder.optional_string(self.index_rebuild_reason)?,
            ),
            (
                "transient_rebuild",
                self.transient_rebuild.build_json(builder)?,
            ),
        ])
    }
}

impl WhyNotDiagnosis {
    pub fn to_json(&self) -> JsonValue {
        Self {
            schema_version: self.schema_version.clone(),
            diagnosis_scope: self.diagnosis_scope.clone(),
            debug_full_scan: self.debug_full_scan,
            resource_profile: self.resource_profile,
            query_text: self.query_text.clone(),
            query_context: self.query_context.clone(),
            result: self
                .result
                .as_object()
                .map(|_| self.result.clone())
                .unwrap_or_default(),
        }
        .into_json()
    }

    pub(crate) fn into_json(self) -> JsonValue {
        self.build_json(&JsonBuilder::host())
            .expect("host DTO construction is infallible")
    }

    #[cfg(test)]
    pub(crate) fn into_json_in(self, work: &mut JsonReservation) -> Result<JsonValue> {
        self.result
            .reconcile_output_retained_in(crate::json::JsonOutputSchema::WhyNot, work)?;
        self.build_json(&JsonBuilder::budgeted(work, JsonInputKind::StoredJson))
    }

    pub(crate) fn into_json_retained(
        self,
        work: &mut crate::storage::RecordReservation,
    ) -> Result<JsonValue> {
        self.result
            .reconcile_output_retained_in(crate::json::JsonOutputSchema::WhyNot, &mut work.json)?;
        let value = {
            let (json, strings) = work.parts_mut();
            self.build_json(&JsonBuilder::budgeted_owned_strings(
                json,
                strings,
                JsonInputKind::StoredJson,
            ))
        };
        work.retain_workspace(0);
        value
    }

    fn build_json(mut self, builder: &JsonBuilder<'_>) -> Result<JsonValue> {
        let mut object = self
            .result
            .as_object_mut()
            .map(std::mem::take)
            .unwrap_or_default();
        drop(self.result);
        builder.insert_if_absent(&mut object, "schema_version", || {
            builder.owned_string(self.schema_version)
        })?;
        builder.insert_if_absent(&mut object, "diagnosis_scope", || {
            builder.owned_string(self.diagnosis_scope)
        })?;
        builder.insert_if_absent(&mut object, "debug_full_scan", || {
            Ok(JsonValue::Bool(self.debug_full_scan))
        })?;
        builder.insert_if_absent(&mut object, "resource_profile", || {
            builder.string(self.resource_profile.as_json_str())
        })?;
        builder.insert_if_absent(&mut object, "profile", || {
            builder.string(self.resource_profile.as_json_str())
        })?;
        builder.insert_if_absent(&mut object, "query_context", || {
            let mut value = self.query_context.build_json(builder)?;
            builder.insert_if_absent(value.as_object_mut().unwrap(), "text", || {
                builder.optional_string(self.query_text)
            })?;
            Ok(value)
        })?;
        Ok(JsonValue::Object(object))
    }
}

impl SearchExecutionStats {
    pub fn to_json(self) -> JsonValue {
        self.build_json(&JsonBuilder::host())
            .expect("host DTO construction is infallible")
    }

    fn build_json(self, builder: &JsonBuilder<'_>) -> Result<JsonValue> {
        builder.object([
            (
                "vector_scan_count",
                JsonValue::Number(self.vector_scan_count as f64),
            ),
            ("distance_ops", JsonValue::Number(self.distance_ops as f64)),
            (
                "fusion_candidate_count",
                JsonValue::Number(self.fusion_candidate_count as f64),
            ),
        ])
    }
}

impl TransientRebuildContext {
    pub fn to_json(&self) -> JsonValue {
        self.build_json(&JsonBuilder::host())
            .expect("host DTO construction is infallible")
    }

    fn build_json(&self, builder: &JsonBuilder<'_>) -> Result<JsonValue> {
        builder.object([
            ("started", JsonValue::Bool(self.started)),
            ("completed", JsonValue::Bool(self.completed)),
            ("cancelled", JsonValue::Bool(self.cancelled)),
            ("budget_mb", JsonValue::Number(self.budget_mb as f64)),
            (
                "chunks_scanned",
                self.chunks_scanned
                    .map(|count| JsonValue::Number(count as f64))
                    .unwrap_or(JsonValue::Null),
            ),
            (
                "bytes_read",
                self.bytes_read
                    .map(|bytes| JsonValue::Number(bytes as f64))
                    .unwrap_or(JsonValue::Null),
            ),
            ("budget_exceeded", JsonValue::Bool(self.budget_exceeded)),
        ])
    }
}

#[cfg(test)]
fn effective_weights(
    requested: Option<SearchWeights>,
    has_text: bool,
    has_vector: bool,
    metadata_boost_available: bool,
) -> Result<WeightContext> {
    effective_weights_with(
        requested,
        has_text,
        has_vector,
        metadata_boost_available,
        |value| Ok(value.to_owned()),
    )
}

pub(crate) fn effective_weights_with(
    requested: Option<SearchWeights>,
    has_text: bool,
    has_vector: bool,
    metadata_boost_available: bool,
    mut copy: impl FnMut(&str) -> Result<String>,
) -> Result<WeightContext> {
    let source = if requested.is_some() {
        "user_override"
    } else if metadata_boost_available {
        "default_with_boost"
    } else {
        "default_no_boost"
    };
    let requested = requested
        .unwrap_or_else(|| default_weights_for(has_text, has_vector, metadata_boost_available));
    requested.validate()?;
    let mut effective = requested;
    let removed_text = !has_text && effective.bm25 > 0.0;
    let removed_vector = !has_vector && effective.vector > 0.0;
    let removed_metadata = !metadata_boost_available && effective.metadata > 0.0;
    if removed_text {
        effective.bm25 = 0.0;
    }
    if removed_vector {
        effective.vector = 0.0;
    }
    if removed_metadata {
        effective.metadata = 0.0;
    }
    let sum = effective.bm25 + effective.vector + effective.metadata;
    if sum <= 0.0 {
        return Err(Error::new(
            ErrorCode::InvalidSearchWeights,
            "effective search weights sum to zero",
        )
        .with_detail("field", "weights")
        .with_detail("reason", "effective_sum_zero")
        .with_detail("weights_reason", "effective_sum_zero")
        .with_detail("has_text", has_text)
        .with_detail("has_vector", has_vector)
        .with_detail("metadata_boost_available", metadata_boost_available)
        .with_detail("requested_bm25", requested.bm25)
        .with_detail("requested_vector", requested.vector)
        .with_detail("requested_metadata", requested.metadata));
    }
    let normalized = (sum - 1.0).abs() > f64::EPSILON;
    if normalized {
        effective.bm25 /= sum;
        effective.vector /= sum;
        effective.metadata /= sum;
    }
    let reason = match (removed_text, removed_vector, removed_metadata) {
        (true, true, true) => unreachable!("zero effective sum rejected above"),
        (true, true, false) => Some("text branch unavailable; vector branch unavailable"),
        (true, false, true) => Some("text branch unavailable; metadata boost unavailable"),
        (false, true, true) => Some("vector branch unavailable; metadata boost unavailable"),
        (true, false, false) => Some("text branch unavailable"),
        (false, true, false) => Some("vector branch unavailable"),
        (false, false, true) => Some("metadata boost unavailable"),
        (false, false, false) => normalized.then_some("weights normalized to sum 1.0"),
    };
    Ok(WeightContext {
        requested,
        effective,
        source: copy(source)?,
        renormalization_reason: reason.map(&mut copy).transpose()?,
    })
}

pub(crate) fn default_weights_for(
    has_text: bool,
    has_vector: bool,
    metadata_boost_available: bool,
) -> SearchWeights {
    if !metadata_boost_available {
        return SearchWeights::default_for(has_text, has_vector);
    }
    match (has_text, has_vector) {
        (true, true) => SearchWeights {
            bm25: 0.45,
            vector: 0.50,
            metadata: 0.05,
        },
        (true, false) => SearchWeights {
            bm25: 0.95,
            vector: 0.0,
            metadata: 0.05,
        },
        (false, true) => SearchWeights {
            bm25: 0.0,
            vector: 0.95,
            metadata: 0.05,
        },
        (false, false) => SearchWeights::default_for(false, false),
    }
}

#[cfg(test)]
fn build_explain(input: ExplainInput<'_>) -> JsonValue {
    build_explain_with(input, &JsonBuilder::host()).expect("host DTO construction is infallible")
}

pub(crate) fn build_explain_in(
    input: ExplainInput<'_>,
    work: &mut JsonReservation,
) -> Result<JsonValue> {
    build_explain_with(
        input,
        &JsonBuilder::budgeted(work, JsonInputKind::StoredJson),
    )
}

fn build_explain_with(input: ExplainInput<'_>, builder: &JsonBuilder<'_>) -> Result<JsonValue> {
    let candidate_sources: &[&str] = match (input.bm25_recalled, input.vector_recalled) {
        (true, true) => &["bm25", "vector"],
        (true, false) => &["bm25"],
        (false, true) => &["vector"],
        (false, false) => &[],
    };
    builder.object([
        ("schema_version", builder.string("0.1")?),
        ("chunk_id", builder.string(&input.chunk.chunk_id)?),
        (
            "chunk_external_id",
            builder.string(&input.chunk.chunk_external_id)?,
        ),
        ("document_id", builder.string(&input.chunk.document_id)?),
        (
            "document_external_id",
            builder.string(input.document_external_id)?,
        ),
        ("collection", builder.string(input.collection_name)?),
        ("final_score", JsonValue::Number(input.final_score)),
        (
            "scores",
            builder.object([
                (
                    "bm25_raw",
                    input
                        .bm25_raw
                        .map(JsonValue::Number)
                        .unwrap_or(JsonValue::Null),
                ),
                ("bm25_norm", JsonValue::Number(input.bm25_norm)),
                (
                    "vector_raw",
                    input
                        .vector_raw
                        .map(JsonValue::Number)
                        .unwrap_or(JsonValue::Null),
                ),
                (
                    "vector_norm",
                    input
                        .vector_norm
                        .map(JsonValue::Number)
                        .unwrap_or(JsonValue::Null),
                ),
                ("metadata_boost", JsonValue::Number(input.metadata_boost)),
                ("final_score", JsonValue::Number(input.final_score)),
            ])?,
        ),
        (
            "weights",
            builder.object([
                ("bm25", JsonValue::Number(input.weights.effective.bm25)),
                ("vector", JsonValue::Number(input.weights.effective.vector)),
                (
                    "metadata",
                    JsonValue::Number(input.weights.effective.metadata),
                ),
                (
                    "requested",
                    weights_json_in(input.weights.requested, builder)?,
                ),
                (
                    "effective",
                    weights_json_in(input.weights.effective, builder)?,
                ),
                ("source", builder.string(&input.weights.source)?),
                (
                    "renormalization_reason",
                    builder.optional_str(input.weights.renormalization_reason.as_deref())?,
                ),
            ])?,
        ),
        (
            "normalization",
            builder.object([
                ("scope", builder.string("candidate_set")?),
                (
                    "candidate_count",
                    JsonValue::Number(input.normalization_candidate_count as f64),
                ),
                ("bm25_min", JsonValue::Number(0.0)),
                (
                    "bm25_max",
                    input
                        .bm25_max
                        .map(JsonValue::Number)
                        .unwrap_or(JsonValue::Null),
                ),
                (
                    "vector_method",
                    builder.string(vector_normalization_method(
                        input.distance_metric,
                        input.vectors_normalized,
                    ))?,
                ),
            ])?,
        ),
        (
            "matched_terms",
            builder.array(
                input
                    .matched_terms
                    .iter()
                    .map(|term| builder.string(&term.term)),
            )?,
        ),
        (
            "bm25",
            builder.object([
                ("k1", JsonValue::Number(crate::index::text::BM25_K1)),
                ("b", JsonValue::Number(crate::index::text::BM25_B)),
                (
                    "terms",
                    builder.array(input.matched_terms.iter().map(|term| {
                        builder.object([
                            ("term", builder.string(&term.term)?),
                            ("tf", JsonValue::Number(term.frequency as f64)),
                            ("df", JsonValue::Number(term.document_frequency as f64)),
                            ("chunk_length", JsonValue::Number(term.chunk_length as f64)),
                            ("avg_chunk_length", JsonValue::Number(term.avg_chunk_length)),
                            (
                                "raw_contribution",
                                JsonValue::Number(term.bm25_contribution),
                            ),
                        ])
                    }))?,
                ),
            ])?,
        ),
        (
            "matched_term_details",
            builder.array(input.matched_terms.iter().map(|term| {
                builder.object([
                    ("term", builder.string(&term.term)?),
                    ("frequency", JsonValue::Number(term.frequency as f64)),
                    (
                        "bm25_contribution",
                        JsonValue::Number(term.bm25_contribution),
                    ),
                ])
            }))?,
        ),
        (
            "tokenizer",
            crate::TokenizerConfig::default().build_json(builder)?,
        ),
        ("fusion", builder.string("weighted_sum")?),
        (
            "candidate_sources",
            builder.array(
                candidate_sources
                    .iter()
                    .map(|source| builder.string(source)),
            )?,
        ),
        (
            "resource_profile",
            builder.string(input.resource_profile.as_json_str())?,
        ),
        (
            "candidate_limits",
            builder.object([
                (
                    "bm25",
                    JsonValue::Number(input.candidate_limits.bm25 as f64),
                ),
                (
                    "vector",
                    JsonValue::Number(input.candidate_limits.vector as f64),
                ),
                (
                    "fusion",
                    JsonValue::Number(input.candidate_limits.fusion as f64),
                ),
            ])?,
        ),
        (
            "execution_stats",
            builder.object([
                (
                    "vector_scan_count",
                    JsonValue::Number(input.vector_scan_count as f64),
                ),
                ("distance_ops", JsonValue::Number(input.distance_ops as f64)),
                (
                    "fusion_candidate_count",
                    JsonValue::Number(input.fusion_candidate_count as f64),
                ),
            ])?,
        ),
        ("index_source", builder.string(input.index_source)?),
        (
            "index_rebuild_reason",
            builder.optional_str(input.index_rebuild_reason)?,
        ),
        (
            "transient_rebuild",
            input.transient_rebuild.build_json(builder)?,
        ),
        ("distance_metric", builder.string(input.distance_metric)?),
        (
            "vector_raw_kind",
            builder.string(vector_raw_kind(input.distance_metric))?,
        ),
        (
            "normalization_method",
            builder.string(vector_normalization_method(
                input.distance_metric,
                input.vectors_normalized,
            ))?,
        ),
        ("filter", input.filter),
        (
            "metadata_boost_context",
            builder.object([
                ("present", JsonValue::Bool(input.metadata_boost_present)),
                ("enabled", JsonValue::Bool(input.metadata_boost_enabled)),
                (
                    "rule_count",
                    JsonValue::Number(input.boost_rule_count as f64),
                ),
                (
                    "matched_rule_count",
                    JsonValue::Number(input.matched_boost_rule_count as f64),
                ),
                (
                    "config_hash",
                    builder.optional_str(input.boost_config_hash)?,
                ),
            ])?,
        ),
        ("chunk_boundary", {
            let span = input.chunk.source_span.as_ref();
            let start = builder.optional_u64(span.and_then(|span| span.start))?;
            let end = builder.optional_u64(span.and_then(|span| span.end))?;
            if let Some(span) = span.filter(|span| span.line_start.is_some()) {
                builder.object([
                    ("start", start),
                    ("end", end),
                    (
                        "line_start",
                        JsonValue::Number(span.line_start.unwrap() as f64),
                    ),
                    ("line_end", JsonValue::Number(span.line_end.unwrap() as f64)),
                ])?
            } else {
                builder.object([("start", start), ("end", end)])?
            }
        }),
        (
            "source",
            builder.object([
                (
                    "document_path",
                    builder.optional_str(document_path(
                        input.document_metadata,
                        input.document_source_uri,
                    ))?,
                ),
                (
                    "source_uri",
                    builder.optional_str(input.document_source_uri)?,
                ),
            ])?,
        ),
    ])
}

pub(crate) fn weights_json_in(
    weights: SearchWeights,
    builder: &JsonBuilder<'_>,
) -> Result<JsonValue> {
    builder.object([
        ("bm25", JsonValue::Number(weights.bm25)),
        ("vector", JsonValue::Number(weights.vector)),
        ("metadata", JsonValue::Number(weights.metadata)),
    ])
}

fn vector_raw_kind(distance_metric: &str) -> &'static str {
    match distance_metric {
        "dot_product" => "dot_product",
        "euclidean" => "distance",
        _ => "similarity",
    }
}

fn vector_normalization_method(distance_metric: &str, vectors_normalized: bool) -> &'static str {
    match (distance_metric, vectors_normalized) {
        ("cosine", _) => "cosine_to_unit_interval",
        ("dot_product", true) => "normalized_dot_product_cosine_to_unit_interval",
        ("dot_product", false) => "raw_dot_product_no_normalization",
        ("euclidean", _) => "distance_to_inverse_unit_interval",
        _ => "metric_to_unit_interval",
    }
}

pub(crate) fn candidate_limits_json_in(
    limits: CandidateLimits,
    builder: &JsonBuilder<'_>,
) -> Result<JsonValue> {
    builder.object([
        ("bm25", JsonValue::Number(limits.bm25 as f64)),
        ("vector", JsonValue::Number(limits.vector as f64)),
        ("fusion", JsonValue::Number(limits.fusion as f64)),
    ])
}

fn document_path<'a>(
    metadata: &'a JsonValue,
    document_source_uri: Option<&'a str>,
) -> Option<&'a str> {
    metadata
        .get("path")
        .and_then(JsonValue::as_str)
        .or(document_source_uri)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn error_detail_value<'a>(err: &'a Error, key: &str) -> Option<&'a str> {
        err.detail()
            .iter()
            .find(|(field, _)| field == key)
            .map(|(_, value)| value.as_str())
    }

    #[test]
    fn search_result_json_uses_public_normalized_score_keys() {
        let result = SearchResult {
            rank: 1,
            chunk_id: "chunk_1".into(),
            chunk_external_id: "doc#1".into(),
            document_id: "doc_1".into(),
            document_external_id: "doc".into(),
            snippet: "wal recovery".into(),
            document_metadata: JsonValue::object(),
            chunk_metadata: JsonValue::object(),
            final_score: 0.7,
            bm25_score: Some(4.2),
            vector_score: Some(0.8),
            metadata_boost: 0.1,
            normalized_scores: NormalizedScores {
                bm25_norm: 0.6,
                vector_norm: Some(0.8),
                metadata_boost: 0.1,
            },
            explain: object_from_pairs([(
                "scores",
                object_from_pairs([
                    ("bm25_norm", JsonValue::Number(0.6)),
                    ("vector_norm", JsonValue::Number(0.8)),
                    ("metadata_boost", JsonValue::Number(0.1)),
                    ("final_score", JsonValue::Number(0.7)),
                ]),
            )]),
        };

        let json = result.to_json();
        let normalized = json
            .get("normalized_scores")
            .and_then(JsonValue::as_object)
            .expect("normalized_scores object");
        assert!(matches!(
            normalized.get("bm25"),
            Some(JsonValue::Number(0.6))
        ));
        assert!(matches!(
            normalized.get("vector"),
            Some(JsonValue::Number(0.8))
        ));
        assert!(matches!(
            normalized.get("metadata_boost"),
            Some(JsonValue::Number(0.1))
        ));
        assert!(!normalized.contains_key("bm25_norm"));
        assert!(!normalized.contains_key("vector_norm"));
    }

    #[test]
    fn effective_weights_errors_include_unavailable_branch_context() {
        let err = effective_weights(
            Some(SearchWeights::new(0.0, 0.0, 1.0).unwrap()),
            true,
            false,
            false,
        )
        .unwrap_err();

        assert_eq!(err.code(), ErrorCode::InvalidSearchWeights);
        assert_eq!(error_detail_value(&err, "field"), Some("weights"));
        assert_eq!(
            error_detail_value(&err, "weights_reason"),
            Some("effective_sum_zero")
        );
        assert_eq!(
            error_detail_value(&err, "reason"),
            Some("effective_sum_zero")
        );
        assert_eq!(error_detail_value(&err, "has_text"), Some("true"));
        assert_eq!(error_detail_value(&err, "has_vector"), Some("false"));
        assert_eq!(
            error_detail_value(&err, "metadata_boost_available"),
            Some("false")
        );
        assert_eq!(error_detail_value(&err, "requested_metadata"), Some("1"));
    }

    #[test]
    fn weight_context_preflight_preserves_ordered_reasons_and_exact_string_credit() {
        use crate::json::{JsonOperationContext, JsonProcessingPolicyV1};
        use crate::storage::RecordReservation;

        let limit = JsonProcessingPolicyV1::WORK_BUDGET_BYTES;
        for seed in 0..100 {
            for mask in 0..8 {
                let has_text = mask & 1 != 0;
                let has_vector = mask & 2 != 0;
                let boost = mask & 4 != 0;
                let requested = (seed % 3 != 0).then_some(SearchWeights {
                    bm25: 0.1 + seed as f64 / 1000.0,
                    vector: 0.4,
                    metadata: 0.5 - seed as f64 / 1000.0,
                });
                let raw =
                    requested.unwrap_or_else(|| default_weights_for(has_text, has_vector, boost));
                let source = if requested.is_some() {
                    "user_override"
                } else if boost {
                    "default_with_boost"
                } else {
                    "default_no_boost"
                };
                let mut expected = raw;
                let mut reasons = Vec::new();
                for (available, value, reason) in [
                    (has_text, &mut expected.bm25, "text branch unavailable"),
                    (
                        has_vector,
                        &mut expected.vector,
                        "vector branch unavailable",
                    ),
                    (boost, &mut expected.metadata, "metadata boost unavailable"),
                ] {
                    if !available && *value > 0.0 {
                        *value = 0.0;
                        reasons.push(reason);
                    }
                }
                let sum = expected.bm25 + expected.vector + expected.metadata;
                if sum > 0.0 && (sum - 1.0).abs() > f64::EPSILON {
                    expected.bm25 /= sum;
                    expected.vector /= sum;
                    expected.metadata /= sum;
                    if reasons.is_empty() {
                        reasons.push("weights normalized to sum 1.0");
                    }
                }
                let reason = (!reasons.is_empty()).then(|| reasons.join("; "));
                let bytes = source.len() + reason.as_ref().map_or(0, String::len);
                let context = JsonOperationContext::standalone();
                let mut held = context.workspace_reservation();
                held.grow_to(limit - bytes, limit - bytes, "operation_scratch")
                    .unwrap();
                let mut work =
                    RecordReservation::new(context.reservation(), context.workspace_reservation());
                let mut calls = 0;
                let actual =
                    effective_weights_with(requested, has_text, has_vector, boost, |value| {
                        calls += 1;
                        work.copy_string(value, &context)
                    });
                if sum <= 0.0 {
                    assert_eq!(actual.unwrap_err().code(), ErrorCode::InvalidSearchWeights);
                    assert_eq!(calls, 0, "invalid weights allocate no context strings");
                } else {
                    let actual = actual.unwrap();
                    assert_eq!(actual.source, source);
                    assert_eq!(actual.renormalization_reason, reason);
                    assert_eq!(actual.effective.bm25.to_bits(), expected.bm25.to_bits());
                    assert_eq!(actual.effective.vector.to_bits(), expected.vector.to_bits());
                    assert_eq!(
                        actual.effective.metadata.to_bits(),
                        expected.metadata.to_bits()
                    );
                    assert_eq!(work.parts_mut().1.unwrap().reserved_bytes(), bytes);
                    assert_eq!(
                        held.grow_to(limit - bytes + 1, limit - bytes + 1, "operation_scratch")
                            .unwrap_err()
                            .code(),
                        ErrorCode::ResourceExhausted
                    );
                    drop(actual);
                }
                drop(work);
                held.grow_to(limit, limit, "operation_scratch").unwrap();
                let mut work =
                    RecordReservation::new(context.reservation(), context.workspace_reservation());
                let error =
                    effective_weights_with(requested, has_text, has_vector, boost, |value| {
                        work.copy_string(value, &context)
                    })
                    .unwrap_err();
                assert_eq!(
                    error.code(),
                    if sum <= 0.0 {
                        ErrorCode::InvalidSearchWeights
                    } else {
                        ErrorCode::ResourceExhausted
                    }
                );
                assert_eq!(work.parts_mut().1.unwrap().reserved_bytes(), 0);
            }
        }
    }

    #[test]
    fn owned_result_conversion_preserves_json_and_moves_metadata_allocations() {
        let context = SearchQueryContext::persistent_checkpoint(SearchQueryContextInput {
            text_present: true,
            embedding_present: false,
            filter_present: true,
            weights: WeightContext {
                requested: SearchWeights::default_for(true, false),
                effective: SearchWeights::default_for(true, false),
                source: "default_no_boost".into(),
                renormalization_reason: None,
            },
            metadata_boost_present: false,
            metadata_boost_enabled: false,
            boost_config_hash: None,
            boost_rule_count: 0,
            top_k: 1,
            profile: ResourceProfile::Balanced,
            candidate_limits: CandidateLimits {
                bm25: 10,
                vector: 10,
                fusion: 10,
            },
            cache_budget_mb: 256,
            chunks_scanned: 1,
            bytes_read: Some(0),
        });
        let chunk = test_chunk();
        for seed in 0..100 {
            let metadata = object_from_pairs([
                (
                    "payload",
                    JsonValue::String(format!("\u{4e2d}\u{1f600}\n\\\"{seed}")),
                ),
                (
                    "numbers",
                    JsonValue::Array(vec![
                        JsonValue::Number(-0.0),
                        JsonValue::Number(seed as f64 / 7.0),
                    ]),
                ),
            ]);
            let result = SearchResult {
                rank: seed + 1,
                chunk_id: chunk.chunk_id.clone(),
                chunk_external_id: chunk.chunk_external_id.clone(),
                document_id: chunk.document_id.clone(),
                document_external_id: "doc".into(),
                snippet: "alpha".into(),
                document_metadata: metadata.clone(),
                chunk_metadata: metadata.clone(),
                final_score: 0.5,
                bm25_score: Some(1.0),
                vector_score: None,
                metadata_boost: 0.0,
                normalized_scores: NormalizedScores {
                    bm25_norm: 0.5,
                    vector_norm: None,
                    metadata_boost: 0.0,
                },
                explain: build_explain(test_explain_input(&chunk, &metadata)),
            };
            let document_ptr = result
                .document_metadata
                .get("payload")
                .unwrap()
                .as_str()
                .unwrap()
                .as_ptr();
            let chunk_ptr = result
                .chunk_metadata
                .get("payload")
                .unwrap()
                .as_str()
                .unwrap()
                .as_ptr();
            let results = SearchResults {
                schema_version: "0.1".into(),
                collection: "notes".into(),
                query: Some("alpha".into()),
                top_k: 1,
                profile: ResourceProfile::Balanced,
                query_context: context.clone(),
                results: vec![result],
            };
            let expected = results.to_json().to_compact_string();
            let json_context = crate::json::JsonOperationContext::standalone();
            let mut work = json_context.reservation();
            for result in &results.results {
                for value in [
                    &result.document_metadata,
                    &result.chunk_metadata,
                    &result.explain,
                ] {
                    work.merge(
                        value
                            .reserve_retained_in(JsonInputKind::StoredJson, &json_context)
                            .unwrap(),
                    );
                }
            }
            let query_ptr = results.query.as_ref().unwrap().as_ptr();
            let id_ptr = results.results[0].chunk_id.as_ptr();
            let actual = results
                .build_json(&JsonBuilder::budgeted(&mut work, JsonInputKind::StoredJson))
                .unwrap();
            assert_eq!(actual.to_compact_string(), expected);
            let measured = actual
                .reserve_retained_in(JsonInputKind::StoredJson, &json_context)
                .unwrap();
            assert_eq!(work.reserved_bytes(), measured.reserved_bytes());
            assert_eq!(
                actual.get("query").unwrap().as_str().unwrap().as_ptr(),
                query_ptr
            );
            let item = &actual.get("results").unwrap().as_array().unwrap()[0];
            assert_eq!(
                item.get("chunk_id").unwrap().as_str().unwrap().as_ptr(),
                id_ptr
            );
            assert_eq!(
                item.get("document_metadata")
                    .unwrap()
                    .get("payload")
                    .unwrap()
                    .as_str()
                    .unwrap()
                    .as_ptr(),
                document_ptr
            );
            assert_eq!(
                item.get("chunk_metadata")
                    .unwrap()
                    .get("payload")
                    .unwrap()
                    .as_str()
                    .unwrap()
                    .as_ptr(),
                chunk_ptr
            );

            let expression_ptr = metadata.get("payload").unwrap().as_str().unwrap().as_ptr();
            let diagnosis = WhyNotDiagnosis {
                schema_version: "0.1".into(),
                diagnosis_scope: "debug_exact".into(),
                debug_full_scan: true,
                resource_profile: ResourceProfile::Balanced,
                query_text: Some(format!("query-{seed}")),
                query_context: context.clone(),
                result: object_from_pairs([
                    (
                        "filter_result",
                        object_from_pairs([("expression", metadata)]),
                    ),
                    ("profile", JsonValue::String("existing override".into())),
                ]),
            };
            let expected = diagnosis.to_json().to_compact_string();
            let query_ptr = diagnosis.query_text.as_ref().unwrap().as_ptr();
            let mut diagnosis_work = diagnosis
                .result
                .reserve_retained_in(JsonInputKind::Filter, &json_context)
                .unwrap();
            // The CLI input lease also covered a query vector/boost, now released.
            diagnosis_work
                .grow(1024 * 1024, JsonInputKind::Filter)
                .unwrap();
            let actual = diagnosis.into_json_in(&mut diagnosis_work).unwrap();
            assert_eq!(actual.to_compact_string(), expected);
            let measured = actual
                .reserve_retained_in(JsonInputKind::StoredJson, &json_context)
                .unwrap();
            assert_eq!(diagnosis_work.reserved_bytes(), measured.reserved_bytes());
            assert_eq!(
                actual
                    .get("query_context")
                    .unwrap()
                    .get("text")
                    .unwrap()
                    .as_str()
                    .unwrap()
                    .as_ptr(),
                query_ptr
            );
            assert_eq!(
                actual
                    .get("filter_result")
                    .unwrap()
                    .get("expression")
                    .unwrap()
                    .get("payload")
                    .unwrap()
                    .as_str()
                    .unwrap()
                    .as_ptr(),
                expression_ptr
            );
        }
    }

    #[test]
    fn candidate_sources_follow_recall_membership_not_score_sign_or_weight() {
        let chunk = test_chunk();
        let metadata = JsonValue::object();
        for seed in 0..100 {
            for (text, vector, expected) in [
                (false, false, "[]"),
                (true, false, r#"["bm25"]"#),
                (false, true, r#"["vector"]"#),
                (true, true, r#"["bm25","vector"]"#),
            ] {
                let mut input = test_explain_input(&chunk, &metadata);
                input.bm25_recalled = text;
                input.vector_recalled = vector;
                // Complete raw scores do not invent a recall source, even at zero weight.
                input.bm25_raw = Some(seed as f64);
                input.vector_raw = Some(seed as f64 - 50.0);
                input.vector_norm = vector.then_some(0.0);
                input.metadata_boost = 0.5;
                let mut weights = input.weights.clone();
                weights.effective = SearchWeights::new(0.0, 0.0, 1.0).unwrap();
                input.weights = &weights;
                let context = crate::json::JsonOperationContext::standalone();
                let mut work = context.reservation();
                let explain = build_explain_in(input, &mut work).unwrap();
                assert_eq!(
                    explain
                        .get("candidate_sources")
                        .unwrap()
                        .to_compact_string(),
                    expected
                );
                let measured = explain
                    .reserve_retained_in(JsonInputKind::StoredJson, &context)
                    .unwrap();
                assert_eq!(work.reserved_bytes(), measured.reserved_bytes());
            }
        }
    }

    #[test]
    fn explain_weight_aliases_preserve_effective_values_and_tracking() {
        let chunk = test_chunk();
        let metadata = JsonValue::object();
        for seed in 0..100 {
            let has_text = seed % 3 != 0;
            let has_vector = seed % 3 != 1;
            let boost = seed / 3 % 2 == 0;
            let requested = (seed % 4 != 0).then(|| {
                let bm25 = (seed + 1) as f64 / 200.0;
                SearchWeights::new(bm25, 0.25, 0.75 - bm25).unwrap()
            });
            let weights = effective_weights(requested, has_text, has_vector, boost).unwrap();
            let mut input = test_explain_input(&chunk, &metadata);
            input.weights = &weights;
            let context = crate::json::JsonOperationContext::standalone();
            let mut work = context.reservation();
            let explain = build_explain_in(input, &mut work).unwrap();
            let actual = explain.get("weights").unwrap();
            for (field, value) in [
                ("bm25", weights.effective.bm25),
                ("vector", weights.effective.vector),
                ("metadata", weights.effective.metadata),
            ] {
                assert_eq!(
                    actual.get(field),
                    Some(&JsonValue::Number(value)),
                    "seed={seed}"
                );
                assert_eq!(
                    actual.get(field),
                    actual.get("effective").unwrap().get(field)
                );
            }
            assert_eq!(actual.as_object().unwrap().len(), 7);
            assert_eq!(
                actual.get("source").and_then(JsonValue::as_str),
                Some(weights.source.as_str())
            );
            assert_eq!(
                actual.get("requested"),
                Some(&weights_json_in(weights.requested, &JsonBuilder::host()).unwrap())
            );
            assert_eq!(
                actual.get("renormalization_reason"),
                Some(
                    &weights
                        .renormalization_reason
                        .map(JsonValue::String)
                        .unwrap_or(JsonValue::Null)
                )
            );
            let measured = explain
                .reserve_retained_in(JsonInputKind::StoredJson, &context)
                .unwrap();
            assert_eq!(work.reserved_bytes(), measured.reserved_bytes());
        }
    }

    #[test]
    fn explain_scoring_matches_frozen_requirement_oracles() {
        let chunk = test_chunk();
        let metadata = JsonValue::object();
        let requested = SearchWeights::new(0.45, 0.5, 0.05).unwrap();
        // Requirements 8.9/8.10: fixed expected values, not derived from output weights.
        for (text, vector, boost, expected_weights, expected_score, reason) in [
            (true, true, true, [0.45, 0.5, 0.05], 0.80665, None),
            (
                false,
                true,
                true,
                [0.0, 10.0 / 11.0, 1.0 / 11.0],
                (10.0 * 0.9405 + 0.05) / 11.0,
                Some("text branch unavailable"),
            ),
            (
                true,
                false,
                true,
                [0.9, 0.0, 0.1],
                0.6728,
                Some("vector branch unavailable"),
            ),
            (
                true,
                true,
                false,
                [9.0 / 19.0, 10.0 / 19.0, 0.0],
                (9.0 * 0.742 + 10.0 * 0.9405) / 19.0,
                Some("metadata boost unavailable"),
            ),
        ] {
            let weights = effective_weights(Some(requested), text, vector, boost).unwrap();
            let bm25_hits = if text {
                vec![
                    Bm25Hit {
                        chunk_id: chunk.chunk_id.clone(),
                        score: 7.42,
                        ..Default::default()
                    },
                    Bm25Hit {
                        chunk_id: "normalization-reference".into(),
                        score: 10.0,
                        ..Default::default()
                    },
                ]
            } else {
                Vec::new()
            };
            let vector_hits = if vector {
                vec![crate::index::vector::VectorHit {
                    chunk_id: chunk.chunk_id.clone(),
                    raw_score: 0.881,
                    normalized_score: (0.881 + 1.0) / 2.0,
                }]
            } else {
                Vec::new()
            };
            let scores = crate::index::hybrid::fuse(
                &bm25_hits,
                &vector_hits,
                weights.effective,
                10,
                &crate::json::JsonOperationContext::standalone(),
                |id| crate::metadata::MetadataBoostScore {
                    boost: if boost && id == chunk.chunk_id {
                        0.05
                    } else {
                        0.0
                    },
                    matched_rule_count: usize::from(boost && id == chunk.chunk_id),
                },
            )
            .unwrap();
            let score = scores
                .as_slice()
                .iter()
                .find(|score| score.chunk_id == chunk.chunk_id)
                .unwrap();
            let mut input = test_explain_input(&chunk, &metadata);
            input.weights = &weights;
            input.bm25_raw = score.bm25_raw;
            input.bm25_norm = score.bm25_norm;
            input.bm25_max = text.then_some(10.0);
            input.vector_raw = score.vector_raw;
            input.vector_norm = score.vector_norm;
            input.metadata_boost = score.metadata_boost;
            input.final_score = score.final_score;
            let explain = build_explain(input);
            let weights = explain.get("weights").unwrap();
            for (field, expected) in ["bm25", "vector", "metadata"]
                .into_iter()
                .zip(expected_weights)
            {
                assert!((weights.get(field).unwrap().as_f64().unwrap() - expected).abs() < 1e-12);
                assert_eq!(
                    weights.get(field),
                    weights.get("effective").unwrap().get(field)
                );
            }
            assert_eq!(
                weights.get("source").and_then(JsonValue::as_str),
                Some("user_override")
            );
            assert_eq!(
                weights.get("requested").unwrap().to_compact_string(),
                r#"{"bm25":0.45,"metadata":0.05,"vector":0.5}"#
            );
            assert_eq!(
                weights
                    .get("renormalization_reason")
                    .and_then(JsonValue::as_str),
                reason
            );
            for value in [
                explain.get("final_score").unwrap(),
                explain.get("scores").unwrap().get("final_score").unwrap(),
            ] {
                assert!((value.as_f64().unwrap() - expected_score).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn explain_builder_preserves_fields_and_accounts_for_each_string_copy() {
        let chunk = test_chunk();
        let metadata = object_from_pairs([("path", JsonValue::String("\u{4e2d}/notes".into()))]);
        for seed in 0..100 {
            let context = crate::json::JsonOperationContext::standalone();
            let mut work = context.reservation();
            let terms = (0..seed % 7)
                .map(|i| format!("term-{i}"))
                .collect::<Vec<_>>();
            let details = terms
                .iter()
                .map(|term| MatchedTerm {
                    document_frequency: seed + 1,
                    chunk_length: seed + 3,
                    avg_chunk_length: 10.5,
                    term: term.clone(),
                    frequency: 2,
                    bm25_contribution: 0.75,
                })
                .collect::<Vec<_>>();
            let mut input = test_explain_input(&chunk, &metadata);
            input.matched_terms = &details;
            input.document_source_uri = Some("docs/\u{4e2d}.md");
            input.filter =
                object_from_pairs([("expression", JsonValue::String(format!("filter-{seed}")))]);
            work.merge(
                input
                    .filter
                    .reserve_retained_in(JsonInputKind::Filter, &context)
                    .unwrap(),
            );
            let expected = build_explain(input.clone()).to_compact_string();
            let actual = build_explain_in(input, &mut work).unwrap();
            assert_eq!(actual.to_compact_string(), expected);
            assert_eq!(
                actual.get("matched_terms").unwrap(),
                &JsonValue::Array(terms.iter().cloned().map(JsonValue::String).collect())
            );
            let compact = actual
                .get("matched_term_details")
                .unwrap()
                .as_array()
                .unwrap();
            let detailed = actual
                .get("bm25")
                .unwrap()
                .get("terms")
                .unwrap()
                .as_array()
                .unwrap();
            assert_eq!(compact.len(), terms.len());
            assert_eq!(detailed.len(), terms.len());
            for (i, term) in terms.iter().enumerate() {
                assert_eq!(
                    compact[i].get("term").unwrap().as_str(),
                    Some(term.as_str())
                );
                assert_eq!(compact[i].get("frequency").unwrap().as_f64(), Some(2.0));
                assert_eq!(
                    compact[i].get("bm25_contribution").unwrap().as_f64(),
                    Some(0.75)
                );
                assert_eq!(
                    detailed[i].get("term").unwrap().as_str(),
                    Some(term.as_str())
                );
                assert_eq!(detailed[i].get("tf").unwrap().as_f64(), Some(2.0));
                assert_eq!(
                    detailed[i].get("df").unwrap().as_f64(),
                    Some((seed + 1) as f64)
                );
                assert_eq!(
                    detailed[i].get("chunk_length").unwrap().as_f64(),
                    Some((seed + 3) as f64)
                );
                assert_eq!(
                    detailed[i].get("avg_chunk_length").unwrap().as_f64(),
                    Some(10.5)
                );
                assert_eq!(
                    detailed[i].get("raw_contribution").unwrap().as_f64(),
                    Some(0.75)
                );
            }
            let measured = actual
                .reserve_retained_in(JsonInputKind::StoredJson, &context)
                .unwrap();
            assert_eq!(work.reserved_bytes(), measured.reserved_bytes());
        }
    }

    #[test]
    fn explain_builder_rejects_duplicate_source_strings_under_the_same_budget() {
        let context = crate::json::JsonOperationContext::standalone();
        let kind = JsonInputKind::StoredJson;
        let chunk = test_chunk();
        let metadata = JsonValue::object();
        let source = "x".repeat(4 * 1024 * 1024);
        let pointer = source.as_ptr();
        let mut busy = context.reservation();
        busy.grow(
            crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES - 6 * 1024 * 1024,
            kind,
        )
        .unwrap();
        let mut work = context.reservation();
        let mut input = test_explain_input(&chunk, &metadata);
        input.document_source_uri = Some(&source);
        let error = build_explain_in(input, &mut work).unwrap_err();
        assert_eq!(error.code(), ErrorCode::RecoveryError);
        assert_eq!(
            error_detail_value(&error, "json_input_kind"),
            Some("stored_json")
        );
        assert_eq!(source.as_ptr(), pointer);
        assert_eq!(source.len(), 4 * 1024 * 1024);
        drop(work);
        drop(busy);
        context
            .reservation()
            .grow(crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES, kind)
            .unwrap();
    }

    #[test]
    fn explain_reports_euclidean_distance_normalization() {
        let explain = build_explain(ExplainInput {
            collection_name: "notes",
            chunk: &Chunk {
                chunk_id: "chunk_1".into(),
                document_id: "doc_1".into(),
                collection_id: "col_1".into(),
                chunk_external_id: "doc#1".into(),
                text: "alpha beta".into(),
                text_hash: "hash".into(),
                ordinal: 0,
                token_count: 2,
                metadata: JsonValue::object(),
                embedding_id: Some("emb_1".into()),
                source_span: None,
                created_at: "2026-05-31T00:00:00Z".into(),
            },
            document_external_id: "doc",
            document_metadata: &JsonValue::object(),
            document_source_uri: None,
            bm25_recalled: false,
            vector_recalled: true,
            normalization_candidate_count: 1,
            bm25_raw: None,
            bm25_norm: 0.0,
            bm25_max: None,
            vector_raw: Some(5.0),
            vector_norm: Some(1.0 / 6.0),
            metadata_boost: 0.0,
            final_score: 1.0 / 6.0,
            matched_terms: &[],
            weights: &WeightContext {
                requested: SearchWeights::default_for(false, true),
                effective: SearchWeights::default_for(false, true),
                source: "default".into(),
                renormalization_reason: None,
            },
            resource_profile: ResourceProfile::Balanced,
            candidate_limits: CandidateLimits {
                bm25: 0,
                vector: 10,
                fusion: 10,
            },
            vector_scan_count: 1,
            distance_ops: 1,
            fusion_candidate_count: 1,
            index_source: "exact",
            index_rebuild_reason: None,
            transient_rebuild: TransientRebuildContext {
                started: false,
                completed: false,
                cancelled: false,
                budget_mb: 0,
                chunks_scanned: None,
                bytes_read: None,
                budget_exceeded: false,
            },
            distance_metric: "euclidean",
            vectors_normalized: false,
            filter: JsonValue::object(),
            metadata_boost_present: false,
            metadata_boost_enabled: false,
            boost_config_hash: None,
            boost_rule_count: 0,
            matched_boost_rule_count: 0,
        });

        assert_eq!(
            explain.get("vector_raw_kind").and_then(JsonValue::as_str),
            Some("distance")
        );
        assert_eq!(
            explain
                .get("normalization_method")
                .and_then(JsonValue::as_str),
            Some("distance_to_inverse_unit_interval")
        );
        let normalization = explain
            .get("normalization")
            .and_then(JsonValue::as_object)
            .expect("normalization object");
        assert_eq!(
            normalization
                .get("vector_method")
                .and_then(JsonValue::as_str),
            Some("distance_to_inverse_unit_interval")
        );
    }

    #[test]
    fn dot_product_normalization_method_reports_raw_or_normalized_contract() {
        assert_eq!(vector_raw_kind("dot_product"), "dot_product");
        assert_eq!(
            vector_normalization_method("dot_product", false),
            "raw_dot_product_no_normalization"
        );
        assert_eq!(
            vector_normalization_method("dot_product", true),
            "normalized_dot_product_cosine_to_unit_interval"
        );
    }

    #[test]
    fn explain_source_fields_ignore_non_string_metadata_values() {
        let chunk = test_chunk();
        let explain = build_explain(test_explain_input(
            &chunk,
            &object_from_pairs([
                ("path", JsonValue::Array(Vec::new())),
                (
                    "source_uri",
                    object_from_pairs([("nested", JsonValue::String("value".into()))]),
                ),
            ]),
        ));

        let source = explain
            .get("source")
            .and_then(JsonValue::as_object)
            .expect("source object");
        assert!(matches!(source.get("document_path"), Some(JsonValue::Null)));
        assert!(matches!(source.get("source_uri"), Some(JsonValue::Null)));

        let explain = build_explain(test_explain_input(
            &chunk,
            &object_from_pairs([("source_uri", JsonValue::String("metadata/wal.md".into()))]),
        ));
        let source = explain
            .get("source")
            .and_then(JsonValue::as_object)
            .expect("source object");
        assert!(matches!(source.get("document_path"), Some(JsonValue::Null)));
        assert!(matches!(source.get("source_uri"), Some(JsonValue::Null)));

        let metadata =
            object_from_pairs([("source_uri", JsonValue::String("metadata/wal.md".into()))]);
        let mut input = test_explain_input(&chunk, &metadata);
        input.document_source_uri = Some("docs/wal.md");
        let explain = build_explain(input);
        let source = explain
            .get("source")
            .and_then(JsonValue::as_object)
            .expect("source object");
        assert_eq!(
            source.get("document_path").and_then(JsonValue::as_str),
            Some("docs/wal.md")
        );
        assert_eq!(
            source.get("source_uri").and_then(JsonValue::as_str),
            Some("docs/wal.md")
        );

        let metadata = object_from_pairs([("path", JsonValue::String("metadata/path.md".into()))]);
        let mut input = test_explain_input(&chunk, &metadata);
        input.document_source_uri = Some("docs/wal.md");
        let explain = build_explain(input);
        let source = explain
            .get("source")
            .and_then(JsonValue::as_object)
            .expect("source object");
        assert_eq!(
            source.get("document_path").and_then(JsonValue::as_str),
            Some("metadata/path.md")
        );
        assert_eq!(
            source.get("source_uri").and_then(JsonValue::as_str),
            Some("docs/wal.md")
        );
    }

    fn test_chunk() -> Chunk {
        Chunk {
            chunk_id: "chunk_1".into(),
            document_id: "doc_1".into(),
            collection_id: "col_1".into(),
            chunk_external_id: "doc#1".into(),
            text: "alpha beta".into(),
            text_hash: "hash".into(),
            ordinal: 0,
            token_count: 2,
            metadata: JsonValue::object(),
            embedding_id: Some("emb_1".into()),
            source_span: None,
            created_at: "2026-05-31T00:00:00Z".into(),
        }
    }

    fn test_explain_input<'a>(
        chunk: &'a Chunk,
        document_metadata: &'a JsonValue,
    ) -> ExplainInput<'a> {
        static WEIGHTS: std::sync::OnceLock<WeightContext> = std::sync::OnceLock::new();
        ExplainInput {
            collection_name: "notes",
            chunk,
            document_external_id: "doc",
            document_metadata,
            document_source_uri: None,
            bm25_recalled: false,
            vector_recalled: true,
            normalization_candidate_count: 1,
            bm25_raw: None,
            bm25_norm: 0.0,
            bm25_max: None,
            vector_raw: Some(1.0),
            vector_norm: Some(1.0),
            metadata_boost: 0.0,
            final_score: 1.0,
            matched_terms: &[],
            weights: WEIGHTS.get_or_init(|| WeightContext {
                requested: SearchWeights::default_for(false, true),
                effective: SearchWeights::default_for(false, true),
                source: "default_no_boost".into(),
                renormalization_reason: None,
            }),
            resource_profile: ResourceProfile::Balanced,
            candidate_limits: CandidateLimits {
                bm25: 0,
                vector: 10,
                fusion: 10,
            },
            vector_scan_count: 1,
            distance_ops: 1,
            fusion_candidate_count: 1,
            index_source: "persistent_checkpoint",
            index_rebuild_reason: None,
            transient_rebuild: TransientRebuildContext {
                started: false,
                completed: false,
                cancelled: false,
                budget_mb: 256,
                chunks_scanned: None,
                bytes_read: None,
                budget_exceeded: false,
            },
            distance_metric: "cosine",
            vectors_normalized: false,
            filter: JsonValue::object(),
            metadata_boost_present: false,
            metadata_boost_enabled: false,
            boost_config_hash: None,
            boost_rule_count: 0,
            matched_boost_rule_count: 0,
        }
    }
}
