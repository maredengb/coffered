use std::cell::Cell;
#[cfg(test)]
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::ops::{Deref, DerefMut};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Instant;

use crate::catalog::Catalog;
use crate::error::{Error, ErrorCode, Result};
#[cfg(test)]
use crate::hash::crc32c as checksum32;
use crate::index::hybrid::CandidateScore;
use crate::index::text::{
    Bm25Hit, Bm25Hits, MatchedTerm, TextCacheBuild, TextIndex, TextSearchOptions, TextStatsSummary,
};
#[cfg(test)]
use crate::index::vector::raw_score;
use crate::index::vector::{
    normalize_score, ExactVectorIndex, VectorCandidates, VectorHit, VectorHits, VectorScorer,
    VectorSearchOptions, VectorWork,
};
#[cfg(test)]
use crate::json::object_from_pairs;
use crate::json::{JsonBuilder, JsonValue};
use crate::metadata::{FilterExplain, FilterExpr, MetadataBoostConfig};
#[cfg(test)]
use crate::model::{chunk_text_hash, token_count_hint};
use crate::model::{
    token_count_hint_in, Chunk, CollectionConfig, CollectionHandle, CollectionInfo,
    CollectionStats, DatabaseStats, DistanceMetric, Document, DocumentInput, Embedding,
    EmbeddingInput, IndexStats, InsertSummary, OnConflict, QueryEmbeddingInput, SearchWeights,
    Tombstone, VectorIndexConfig,
};
use crate::profile::{DatabaseConfig, Durability, IoMode, ResourceProfile};
use crate::query::{
    build_explain_in, candidate_limits_json_in, effective_weights_with, weights_json_in,
    CandidateLimits, ExplainInput, NormalizedScores, SearchExecutionStats, SearchQueryContext,
    SearchQueryContextInput, SearchResult, SearchResults, WhyNotDiagnosis,
};
use crate::storage::{
    copy_collection_in, copy_document_in, rewrite_database_retained_in,
    scan_file_with_retained_records_in, CachedVectorSegment, CheckIssue, CheckpointSegment,
    CompactMarker, IndexCheckpointMetadata, IndexSegmentSource, IndexSuffixReport,
    OwnedRecordBatch, PayloadReader, RecordPayload, RecordReservation, RetainedChunk,
    RetainedEmbedding, RewritePlan, ScanReport, Storage, TextCheckpoint, VectorCache,
    VectorIndexSegmentPayload, VectorRecordSegment, FORMAT_VERSION,
};
#[cfg(test)]
use crate::storage::{IndexCheckpointPayload, VectorIndexEntry};
use crate::time;

mod coverage;
mod embedding_import;
mod import;
mod models;
use models::ModelAdmission;
mod read;
mod scoring;
use coverage::catalog_index_coverage_hash;
pub(crate) use import::{ImportBatchMetrics, ImportDocument, ImportRow, ManifestImportOptions};
pub(crate) use read::RecordKey;
use read::{read_visible_chunk_text, read_visible_embedding_vector};
#[cfg(test)]
use scoring::fuse_candidates;
use scoring::{filter_candidates, fuse_completed_candidates, supplement_scores};

#[derive(Clone, Debug, Default)]
pub struct DatabaseOpenOptions {
    pub read_only: bool,
    pub resource_profile: Option<ResourceProfile>,
    pub cache_budget_mb: Option<usize>,
    pub background_compaction: Option<bool>,
    pub durability: Option<Durability>,
    pub io_mode: IoMode,
}

#[derive(Clone, Debug, Default)]
pub struct CheckOptions {
    pub verify_indexes: bool,
}

#[derive(Clone, Debug, Default)]
pub struct CompactOptions {
    pub rebuild_indexes: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ImportIndexBuildMode {
    InlineFullCheckpoint,
    DeferredFullCheckpoint,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum IndexSuffixState {
    Missing,
    PersistentCheckpoint,
    MetadataRecordOnly,
    RebuildNeeded,
}

impl IndexSuffixState {
    fn from_report(report: &IndexSuffixReport) -> Self {
        match report.status.as_str() {
            "persistent_checkpoint" => Self::PersistentCheckpoint,
            "metadata_record_only" => Self::MetadataRecordOnly,
            _ => Self::RebuildNeeded,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Missing => "missing",
            Self::PersistentCheckpoint => "persistent_checkpoint",
            Self::MetadataRecordOnly => "metadata_record_only",
            Self::RebuildNeeded => "rebuild_needed",
        }
    }
}

#[derive(Clone, Debug)]
pub struct CheckReport {
    pub schema_version: String,
    pub path: PathBuf,
    pub status: String,
    pub format_version: u16,
    pub last_valid_commit: u64,
    pub critical_end_offset: u64,
    pub index_suffix_status: String,
    pub index_suffix: Option<IndexSuffixReport>,
    pub issues: Vec<CheckIssue>,
    pub repair_available: bool,
    pub index_check: String,
    pub index_check_reason: Option<String>,
    pub live_documents: usize,
    pub live_chunks: usize,
    pub tombstones: usize,
}

#[derive(Clone, Debug, Default)]
pub struct CompactSummary {
    pub before_file_size: u64,
    pub after_file_size: u64,
    pub records_removed: usize,
    pub tombstones_removed: usize,
    pub segments_merged: usize,
    pub index_rebuild: String,
    pub index_rebuild_reason: Option<String>,
    pub duration_ms: u128,
}

#[derive(Clone, Debug)]
pub struct RebuildIndexSummary {
    pub schema_version: String,
    pub collection: Option<String>,
    pub rebuild_scope: String,
    pub index_suffix_status: String,
    pub index_record_count: usize,
    pub index_extent_count: usize,
    pub text_segment_count: usize,
    pub vector_segment_count: usize,
    pub rebuild_needed: bool,
    pub last_index_commit_id: Option<u64>,
    pub duration_ms: u128,
}

#[derive(Clone, Debug)]
pub struct RepairSummary {
    pub schema_version: String,
    pub action: String,
    pub repaired: bool,
    pub before_file_size: u64,
    pub after_file_size: u64,
    pub bytes_removed: u64,
    pub critical_end_offset: u64,
    pub issues_repaired: usize,
}

#[derive(Clone, Debug)]
pub struct SnapshotSummary {
    pub schema_version: String,
    pub source_path: PathBuf,
    pub output_path: PathBuf,
    pub bytes_copied: u64,
    pub last_valid_commit: u64,
    pub critical_end_offset: u64,
    pub index_suffix_status: String,
}

#[derive(Clone)]
pub struct Database {
    inner: Arc<Mutex<Option<Inner>>>,
    json_work: Arc<crate::json::WorkLedger>,
    temporary_work: Arc<crate::temporary::TemporaryDiskBudget>,
}

struct InnerGuard<'a>(MutexGuard<'a, Option<Inner>>);

impl Deref for InnerGuard<'_> {
    type Target = Inner;

    fn deref(&self) -> &Inner {
        self.0.as_ref().expect("open database guard")
    }
}

impl DerefMut for InnerGuard<'_> {
    fn deref_mut(&mut self) -> &mut Inner {
        self.0.as_mut().expect("open database guard")
    }
}

#[cfg(test)]
#[derive(Clone, Copy, Eq, PartialEq)]
enum QueryOutputStage {
    CheckpointCandidates,
    Fusion,
    ResultArray,
    ResultRow(usize),
}

struct Inner {
    work: Arc<crate::json::WorkLedger>,
    temporary_work: Arc<crate::temporary::TemporaryDiskBudget>,
    path: PathBuf,
    storage: Storage,
    catalog: Catalog,
    text_index: TextIndex,
    lazy_text: bool,
    #[cfg(test)]
    before_why_not_ranks: Option<Box<dyn FnOnce() + Send>>,
    #[cfg(test)]
    query_output_hook: Option<(QueryOutputStage, Box<dyn FnOnce() + Send>)>,
    persistent_vector_segments: Option<VectorCache>,
    lazy_vector_source: Option<IndexSegmentSource>,
    lazy_vector_segment: Option<CachedVectorSegment>,
    index_suffix_status: IndexSuffixState,
    last_index_commit_sequence_no: Option<u64>,
    index_size_bytes: u64,
    last_commit_sequence_no: u64,
    #[cfg(test)]
    fail_next_index_checkpoint_refresh: bool,
    #[cfg(test)]
    import_fail_after_batches: Option<usize>,
    #[cfg(test)]
    index_checkpoint_refreshes: usize,
    #[cfg(test)]
    compact_checkpoint_remaining: Option<(usize, bool)>,
    #[cfg(test)]
    compact_file_remaining: Option<usize>,
    #[cfg(test)]
    compact_rewrite_completions: usize,
    config: DatabaseConfig,
}

struct IndexChanges {
    text: BTreeSet<String>,
    vector: BTreeSet<String>,
    work: crate::json::WorkspaceReservation,
}

struct PendingIndexUpdate {
    reusable_checkpoint: bool,
    reusable_vectors: VectorCache,
    reusable_text: Option<TextCheckpoint>,
    changes: IndexChanges,
}

impl IndexChanges {
    fn insert(
        ids: &mut BTreeSet<String>,
        work: &mut crate::json::WorkspaceReservation,
        id: &str,
    ) -> Result<()> {
        if ids.contains(id) {
            return Ok(());
        }
        let nodes = crate::json::btree_workspace::<String, ()>(ids.len() + 1)
            - crate::json::btree_workspace::<String, ()>(ids.len());
        let mut node_work = work.context().workspace_reservation();
        node_work.grow_to(nodes, nodes, "catalog_build")?;
        let (id, string_work) =
            crate::json::ReservedString::copy_in(id, work.context(), "catalog_build")?.into_parts();
        ids.insert(id);
        work.merge(node_work);
        work.merge(string_work);
        Ok(())
    }

    fn extend(&mut self, other: &Self) -> Result<()> {
        for id in &other.text {
            Self::insert(&mut self.text, &mut self.work, id)?;
        }
        for id in &other.vector {
            Self::insert(&mut self.vector, &mut self.work, id)?;
        }
        Ok(())
    }

    fn from_records<'a>(
        catalog: &Catalog,
        records: impl IntoIterator<Item = &'a RecordPayload>,
        context: &crate::json::JsonOperationContext,
    ) -> Result<Self> {
        let mut changes = Self {
            text: BTreeSet::new(),
            vector: BTreeSet::new(),
            work: context.workspace_reservation(),
        };
        for record in records {
            let (collection_id, text_changed) = match record {
                RecordPayload::CollectionCreate(collection) => {
                    (Some(collection.collection_id.as_str()), true)
                }
                RecordPayload::CollectionDrop { collection_id, .. } => {
                    (Some(collection_id.as_str()), true)
                }
                RecordPayload::DocumentPut(document) => {
                    (Some(document.collection_id.as_str()), true)
                }
                RecordPayload::ChunkPut(chunk) => (Some(chunk.collection_id.as_str()), true),
                RecordPayload::EmbeddingPut(embedding) => {
                    (Some(embedding.collection_id.as_str()), false)
                }
                RecordPayload::Tombstone(tombstone) => (
                    match tombstone.target_type.as_str() {
                        "collection" => Some(tombstone.target_id.as_str()),
                        "document" => catalog
                            .documents
                            .get(&tombstone.target_id)
                            .map(|document| document.collection_id.as_str()),
                        "chunk" => catalog
                            .chunks
                            .get(&tombstone.target_id)
                            .map(|chunk| chunk.collection_id.as_str()),
                        _ => None,
                    },
                    true,
                ),
                RecordPayload::CompactMarker(_) | RecordPayload::Commit(_) => (None, true),
            };
            if let Some(id) = collection_id {
                Self::insert(&mut changes.vector, &mut changes.work, id)?;
                if text_changed {
                    Self::insert(&mut changes.text, &mut changes.work, id)?;
                }
            } else {
                // An unrecognized mutation is not proof that any index is unchanged.
                for collection in catalog.live_collections() {
                    Self::insert(
                        &mut changes.text,
                        &mut changes.work,
                        &collection.collection_id,
                    )?;
                    Self::insert(
                        &mut changes.vector,
                        &mut changes.work,
                        &collection.collection_id,
                    )?;
                }
            }
        }
        Ok(changes)
    }
}

impl Database {
    pub fn create(path: impl AsRef<Path>, config: DatabaseConfig) -> Result<Self> {
        config.validate_json_policy()?;
        if config.background_compaction {
            return Err(Error::new(
                ErrorCode::UnsupportedFeatureFlag,
                "background compaction is not implemented; use manual compact",
            ));
        }
        let json_work = Arc::new(crate::json::WorkLedger::default());
        let temporary_work = Arc::new(crate::temporary::TemporaryDiskBudget::default());
        let context = crate::json::JsonOperationContext::new(Arc::clone(&json_work));
        let storage = Storage::create_in(path.as_ref(), &context)?;
        let path = storage.path().to_path_buf();
        let lazy_text = config
            .cache_budget_mb
            .unwrap_or_else(|| config.resource_profile.params().cache_budget_mb)
            == 0;
        Ok(Self {
            json_work: Arc::clone(&json_work),
            temporary_work: Arc::clone(&temporary_work),
            inner: Arc::new(Mutex::new(Some(Inner {
                work: json_work,
                temporary_work,
                path,
                storage,
                catalog: Catalog::new_in(&context),
                text_index: TextIndex::default(),
                lazy_text,
                #[cfg(test)]
                before_why_not_ranks: None,
                #[cfg(test)]
                query_output_hook: None,
                persistent_vector_segments: None,
                lazy_vector_source: None,
                lazy_vector_segment: None,
                index_suffix_status: IndexSuffixState::Missing,
                last_index_commit_sequence_no: None,
                index_size_bytes: 0,
                last_commit_sequence_no: 0,
                #[cfg(test)]
                fail_next_index_checkpoint_refresh: false,
                #[cfg(test)]
                import_fail_after_batches: None,
                #[cfg(test)]
                index_checkpoint_refreshes: 0,
                #[cfg(test)]
                compact_checkpoint_remaining: None,
                #[cfg(test)]
                compact_file_remaining: None,
                #[cfg(test)]
                compact_rewrite_completions: 0,
                config,
            }))),
        })
    }

    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_with_options(path, DatabaseOpenOptions::default())
    }

    pub fn open_read_only(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_with_options(
            path,
            DatabaseOpenOptions {
                read_only: true,
                ..Default::default()
            },
        )
    }

    pub fn open_with_options(path: impl AsRef<Path>, options: DatabaseOpenOptions) -> Result<Self> {
        if options.background_compaction == Some(true) {
            return Err(Error::new(
                ErrorCode::UnsupportedFeatureFlag,
                "background compaction is not implemented; use manual compact",
            ));
        }
        let mut config = DatabaseConfig::default();
        if let Some(profile) = options.resource_profile {
            config.resource_profile = profile;
        }
        config.cache_budget_mb = options.cache_budget_mb;
        if let Some(background) = options.background_compaction {
            config.background_compaction = background;
        }
        if let Some(durability) = options.durability {
            config.durability = durability;
        }
        config.io_mode = options.io_mode;
        config.validate_json_policy()?;
        // Reserve headroom for decoded objects and the parser's temporary copies.
        let cache_limit = config
            .cache_budget_mb
            .unwrap_or_else(|| config.resource_profile.params().cache_budget_mb)
            .saturating_mul(1_048_576);
        let index_budget_bytes = cache_limit as u64 / 8;
        let json_work = Arc::new(crate::json::WorkLedger::default());
        let temporary_work = Arc::new(crate::temporary::TemporaryDiskBudget::default());
        let context = crate::json::JsonOperationContext::new(Arc::clone(&json_work));
        let mut catalog = Catalog::new_in(&context);
        let (storage, mut scan) = Storage::open_with_retained_records(
            path.as_ref(),
            options.read_only,
            index_budget_bytes,
            &context,
            |record, work| catalog.apply_retained(record, work, &context),
        )?;
        if scan.findings.iter().any(|issue| issue.severity == "error") {
            return Err(Error::new(
                ErrorCode::RecoveryError,
                "database scan found recovery errors; run check for diagnostics",
            ));
        }
        catalog.last_commit_id = scan.last_valid_commit;
        let mut index_suffix_status = index_suffix_status_from_scan(&scan, &catalog, &context)?;
        if let Err(error) = catalog.require_consistent_models() {
            if error.code() != ErrorCode::EmbeddingModelMismatch {
                return Err(error);
            }
            index_suffix_status = IndexSuffixState::RebuildNeeded;
        }
        let persistent_checkpoint = scan
            .lazy_vector_checkpoint
            .take()
            .filter(|_| index_suffix_status == IndexSuffixState::PersistentCheckpoint);
        let lazy_text = options.read_only || index_budget_bytes == 0;
        let (text_index, lazy_vector_source) = if let Some(checkpoint) = persistent_checkpoint {
            match TextStatsSummary::from_source(
                &checkpoint.vectors,
                &storage,
                &catalog,
                &temporary_work,
                &context,
            )? {
                Some(stats) => {
                    let cached = if lazy_text {
                        TextCacheBuild::Miss
                    } else {
                        build_text_cache_in(
                            &catalog,
                            |_| true,
                            &context,
                            cache_limit,
                            |id, input, visit| {
                                checkpoint.vectors.visit_text(&storage, id, input, visit)
                            },
                        )?
                    };
                    match cached {
                        TextCacheBuild::Complete(index) => {
                            (index, Some(checkpoint.vectors.with_text_stats(stats)))
                        }
                        TextCacheBuild::Miss => (
                            TextIndex::default(),
                            Some(checkpoint.vectors.with_text_stats(stats)),
                        ),
                        TextCacheBuild::Invalid => {
                            index_suffix_status = IndexSuffixState::RebuildNeeded;
                            (TextIndex::default(), None)
                        }
                    }
                }
                None => {
                    index_suffix_status = IndexSuffixState::RebuildNeeded;
                    (TextIndex::default(), None)
                }
            }
        } else {
            (TextIndex::default(), None)
        };
        let last_index_commit_sequence_no =
            index_commit_sequence_no_from_scan(&scan, index_suffix_status);
        let index_size_bytes = scan
            .index_suffix
            .as_ref()
            .filter(|_| index_suffix_status == IndexSuffixState::PersistentCheckpoint)
            .map(IndexSuffixReport::total_size_bytes)
            .unwrap_or(0);
        let path = storage.path().to_path_buf();
        Ok(Self {
            json_work: Arc::clone(&json_work),
            temporary_work: Arc::clone(&temporary_work),
            inner: Arc::new(Mutex::new(Some(Inner {
                work: json_work,
                temporary_work,
                path,
                storage,
                catalog,
                text_index,
                lazy_text,
                #[cfg(test)]
                before_why_not_ranks: None,
                #[cfg(test)]
                query_output_hook: None,
                persistent_vector_segments: None,
                lazy_vector_source,
                lazy_vector_segment: None,
                index_suffix_status,
                last_index_commit_sequence_no,
                index_size_bytes,
                last_commit_sequence_no: scan.last_valid_commit_sequence_no,
                #[cfg(test)]
                fail_next_index_checkpoint_refresh: false,
                #[cfg(test)]
                import_fail_after_batches: None,
                #[cfg(test)]
                index_checkpoint_refreshes: 0,
                #[cfg(test)]
                compact_checkpoint_remaining: None,
                #[cfg(test)]
                compact_file_remaining: None,
                #[cfg(test)]
                compact_rewrite_completions: 0,
                config,
            }))),
        })
    }

    pub fn create_collection(&self, config: CollectionConfig) -> Result<CollectionHandle> {
        let context = self.json_context();
        let mut config_work = context.reservation();
        let mut input_work = context.workspace_reservation();
        let config = config;
        config_work.merge(self.admit_input_reclaiming(|| config.validate_in(&context))?);
        // Typed config strings were already counted by the JSON validation view.
        let strings =
            config.name.capacity() + config.embedding_model.as_ref().map_or(0, String::capacity);
        input_work.merge(config_work.take_workspace(strings));
        config_work.retain(0);
        let mut inner = self.lock()?;
        let (catalog, mut records) = inner.prepare_write_batch(&context);
        catalog.check_collection_name_available(&config.name)?;
        let collection_id = records.admit(|| catalog.next_id_in("col_", 0, &context))?;
        let handle_name = records.admit(|| {
            crate::json::ReservedString::copy_in(&config.name, &context, "batch_prepare")
        })?;
        let collection = CollectionInfo {
            tokenizer: config.tokenizer,
            collection_id: records.copy_string(collection_id.as_ref())?,
            name: config.name,
            embedding_dim: config.embedding_dim,
            distance_metric: config.distance_metric,
            embedding_model: config.embedding_model,
            vectors_normalized: config.vectors_normalized,
            vector_index: config.vector_index,
            created_at: records.timestamp(time::unix_millis_now())?,
        };
        records.push(RecordPayload::CollectionCreate(collection))?;
        records.retain_workspace(input_work);
        let records = records.into_batch();
        inner.append_and_apply(records, &context)?;
        Ok(CollectionHandle::new(
            self.clone(),
            collection_id.into_parts().0,
            handle_name.into_parts().0,
        ))
    }

    pub fn collection(&self, name: &str) -> Result<CollectionHandle> {
        let context = self.json_context();
        self.lock()?.read_reclaiming(|inner| {
            let collection = inner.catalog.collection_by_name(name)?;
            let mut work = RecordReservation::from(context.reservation());
            Ok(CollectionHandle::new(
                self.clone(),
                work.copy_string(&collection.collection_id, &context)?,
                work.copy_string(&collection.name, &context)?,
            ))
        })
    }

    pub fn list_collections(&self) -> Result<Vec<CollectionInfo>> {
        let context = self.json_context();
        self.lock()?.read_reclaiming(|inner| {
            let mut work = RecordReservation::from(context.reservation());
            let mut collections = crate::json::ReservedVec::with_capacity(
                inner.catalog.live_collections().count(),
                &context,
                "operation_scratch",
            )?;
            for collection in inner.catalog.live_collections() {
                let (collection, reservation) = copy_collection_in(collection, &context)?;
                work.merge(reservation);
                collections.push(collection);
            }
            Ok(collections.into_vec())
        })
    }

    pub fn drop_collection(&self, name: &str) -> Result<()> {
        let context = self.json_context();
        let mut inner = self.lock()?;
        let (catalog, mut records) = inner.prepare_write_batch(&context);
        let collection = catalog.collection_by_name(name)?;
        let record = RecordPayload::CollectionDrop {
            collection_id: records.copy_string(&collection.collection_id)?,
            name: records.copy_string(&collection.name)?,
            dropped_at: records.timestamp(time::unix_millis_now())?,
            reason: None,
        };
        records.push(record)?;
        let records = records.into_batch();
        inner.append_and_apply(records, &context)
    }

    pub fn stats(&self) -> Result<DatabaseStats> {
        self.lock()?.database_stats()
    }

    pub fn index_stats(&self) -> Result<IndexStats> {
        Ok(self.lock()?.index_stats())
    }

    pub fn rebuild_index(&self, collection: Option<&str>) -> Result<RebuildIndexSummary> {
        let started = Instant::now();
        let context = self.json_context();
        let mut inner = self.lock()?;
        inner.storage.require_writer()?;
        let collection = collection
            .map(|name| {
                inner
                    .catalog
                    .collection_by_name(name)
                    .map(|info| info.name.clone())
            })
            .transpose()?;
        if let Some(name) = &collection {
            let id = inner
                .catalog
                .collection_by_name(name)?
                .collection_id
                .clone();
            inner.cache_collection_model(&id)?;
            inner.catalog.visible_model_state(&id)?.validate(None)?;
        } else {
            inner.catalog.require_consistent_models()?;
        }
        if let Err(error) = inner.catalog.require_consistent_models() {
            if error.code() != ErrorCode::EmbeddingModelMismatch {
                return Err(error);
            }
            let name = collection
                .as_deref()
                .expect("full rebuild rejected conflicts");
            let target = inner.catalog.collection_by_name(name)?;
            let entries = inner
                .catalog
                .live_embeddings()
                .filter(|embedding| embedding.collection_id == target.collection_id)
                .map(|embedding| {
                    (
                        embedding,
                        inner
                            .catalog
                            .record_ref_for_embedding(&embedding.embedding_id),
                    )
                });
            VectorRecordSegment::new(
                target,
                inner.last_commit_sequence_no,
                entries,
                inner.storage.payload_reader()?,
                &context,
            )?
            .validate_transient()?;
            return Ok(RebuildIndexSummary {
                schema_version: "0.1".into(),
                collection,
                rebuild_scope: "transient_vector_collection".into(),
                index_suffix_status: inner.index_suffix_status.as_str().into(),
                index_record_count: 0,
                index_extent_count: 0,
                text_segment_count: 0,
                vector_segment_count: 1,
                rebuild_needed: true,
                last_index_commit_id: None,
                duration_ms: started.elapsed().as_millis(),
            });
        }
        let text = inner.read_reclaiming(|inner| {
            inner.prepare_text_checkpoint(None, &BTreeSet::new(), &context)
        })?;
        if !inner.lazy_text {
            inner.text_index.clear_cache();
            inner.text_index = inner.read_reclaiming(|inner| {
                build_text_index_from_checkpoint(
                    text.as_ref(),
                    &inner.catalog,
                    |_| true,
                    &context,
                    inner.active_cache_budget_mb().saturating_mul(1_048_576),
                )
            })?;
        }
        let report = inner.refresh_index_checkpoint_prepared(
            VectorCache::new(&context),
            &BTreeSet::new(),
            text,
            &context,
        )?;
        let index_stats = inner.index_stats();
        let last_index_commit_id = report
            .as_ref()
            .and_then(index_commit_sequence_no_from_report);
        Ok(RebuildIndexSummary {
            schema_version: "0.1".into(),
            collection,
            rebuild_scope: "full_visible_snapshot".into(),
            index_suffix_status: inner.index_suffix_status.as_str().into(),
            index_record_count: report
                .as_ref()
                .map(|suffix| suffix.index_record_count)
                .unwrap_or_default(),
            index_extent_count: report
                .as_ref()
                .map(|suffix| suffix.index_extent_count)
                .unwrap_or_default(),
            text_segment_count: report
                .as_ref()
                .map(|suffix| suffix.text_segment_count)
                .unwrap_or_default(),
            vector_segment_count: report
                .as_ref()
                .map(|suffix| suffix.vector_segment_count)
                .unwrap_or_default(),
            rebuild_needed: index_stats.rebuild_needed,
            last_index_commit_id,
            duration_ms: started.elapsed().as_millis(),
        })
    }

    pub fn check(&self, options: CheckOptions) -> Result<CheckReport> {
        let mut inner = self.lock()?;
        let Inner {
            path,
            text_index,
            persistent_vector_segments,
            lazy_vector_segment,
            ..
        } = &mut *inner;
        build_check_report_reclaiming(path, options, &self.json_context(), || {
            Inner::release_optional_caches(
                text_index,
                persistent_vector_segments,
                lazy_vector_segment,
                std::iter::empty(),
            )
        })
    }

    pub fn check_path(path: impl AsRef<Path>, options: CheckOptions) -> Result<CheckReport> {
        build_check_report(
            path.as_ref(),
            options,
            &crate::json::JsonOperationContext::standalone(),
        )
    }

    pub fn repair_truncate_tail(path: impl AsRef<Path>) -> Result<RepairSummary> {
        let (mut storage, scan) =
            Storage::open_summary_with_records(path.as_ref(), false, |_| Ok(()))?;
        let issues_repaired = scan
            .findings
            .iter()
            .filter(|issue| issue.repair_available || issue.visibility_impact == "tail_only")
            .count();
        let before_file_size = storage.file_size()?;
        let should_truncate = issues_repaired > 0 && before_file_size > scan.critical_end_offset;
        let (before_file_size, after_file_size) = if should_truncate {
            storage.truncate_to_critical_end(Durability::Commit)?
        } else {
            (before_file_size, before_file_size)
        };
        Ok(RepairSummary {
            schema_version: "0.1".into(),
            action: "truncate_tail".into(),
            repaired: after_file_size < before_file_size,
            before_file_size,
            after_file_size,
            bytes_removed: before_file_size.saturating_sub(after_file_size),
            critical_end_offset: scan.critical_end_offset,
            issues_repaired: if after_file_size < before_file_size {
                issues_repaired
            } else {
                0
            },
        })
    }

    pub fn snapshot(
        path: impl AsRef<Path>,
        output_path: impl AsRef<Path>,
    ) -> Result<SnapshotSummary> {
        let context = crate::json::JsonOperationContext::standalone();
        let mut catalog = Catalog::new_in(&context);
        let (storage, scan) = Storage::open_summary_with_retained_records(
            path.as_ref(),
            false,
            &context,
            |record, work| catalog.apply_retained(record, work, &context),
        )?;
        let source_path = storage.path().to_path_buf();
        let output_path = output_path.as_ref().to_path_buf();
        ensure_snapshot_output_path(&source_path, &output_path)?;
        catalog.last_commit_id = scan.last_valid_commit;
        let index_suffix_status = index_suffix_status_from_scan(&scan, &catalog, &context)?;
        let tmp_path = snapshot_temp_path(&output_path)?;
        let bytes_copied = copy_snapshot_file(&source_path, &tmp_path)?;
        if output_path.exists() {
            let _ = fs::remove_file(&tmp_path);
            return Err(Error::new(
                ErrorCode::InvalidSearchParams,
                "snapshot output path already exists",
            ));
        }
        fs::rename(&tmp_path, &output_path)?;
        sync_parent_dir(&output_path)?;
        Ok(SnapshotSummary {
            schema_version: "0.1".into(),
            source_path,
            output_path,
            bytes_copied,
            last_valid_commit: scan.last_valid_commit,
            critical_end_offset: scan.critical_end_offset,
            index_suffix_status: index_suffix_status.as_str().into(),
        })
    }

    pub fn compact(&self, options: CompactOptions) -> Result<CompactSummary> {
        let started = Instant::now();
        let context = self.json_context();
        let mut inner = self.lock()?;
        inner.storage.require_writer()?;
        inner.catalog.require_consistent_models()?;
        let before = inner.storage.file_size()?;
        let tombstones_removed = inner.catalog.tombstone_count();
        let segments_merged = inner.compacted_persistent_segment_count();
        let count_error = || {
            Error::new(
                ErrorCode::FileFormatError,
                "compaction record count overflow",
            )
        };
        let records_rewritten = [
            inner.catalog.live_collections().count(),
            inner.catalog.live_documents().count(),
            inner.catalog.live_chunks().count(),
            inner.catalog.live_embeddings().count(),
        ]
        .into_iter()
        .try_fold(0usize, |count, n| {
            count.checked_add(n).ok_or_else(count_error)
        })?;
        let dropped_count = inner
            .catalog
            .collections
            .values()
            .filter(|collection| {
                inner
                    .catalog
                    .deleted_collections
                    .contains(&collection.collection_id)
            })
            .count();
        let record_count = dropped_count
            .checked_mul(2)
            .and_then(|n| records_rewritten.checked_add(n))
            .and_then(|n| n.checked_add(1))
            .ok_or_else(count_error)?;
        let business_records_after = record_count - 1; // CompactMarker is not a business record.
        let records_removed = inner
            .catalog
            .business_record_count()
            .checked_sub(business_records_after)
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::FileFormatError,
                    "inconsistent compaction record counts",
                )
            })?;
        let marker = RecordPayload::CompactMarker(CompactMarker {
            compact_generation: inner.last_commit_sequence_no,
            source_last_commit_sequence_no: inner.last_commit_sequence_no,
            compacted_at: time::now_rfc3339(),
            records_rewritten,
            id_high_watermarks: inner.catalog.id_high_watermarks.clone(),
            compacted_main_checksum: 0,
        });
        let durability = inner.config.durability;
        let rewrite_plan = RewritePlan {
            records: record_count,
            batch_record_limit: inner.config.resource_profile.params().import_batch_chunks,
            batch_byte_target: batch_byte_target(inner.active_cache_budget_mb()).max(1),
            temporary_disk: self.temporary_disk_budget(),
        };
        // Iterator admission cannot be retried after consuming a source record.
        // Release optional caches before payload/token scratch is needed.
        let Inner {
            text_index,
            persistent_vector_segments,
            lazy_vector_segment,
            ..
        } = &mut *inner;
        Inner::release_optional_caches(
            text_index,
            persistent_vector_segments,
            lazy_vector_segment,
            std::iter::empty(),
        );
        // Build requested indexes before replacement, so rejection leaves the
        // original intact. Disabled rebuilds need no postings or tokenization.
        let text_checkpoint = if options.rebuild_indexes {
            inner.prepare_compact_text_checkpoint(&context)?
        } else {
            None
        };
        let mut text_index = if options.rebuild_indexes && !inner.lazy_text {
            build_text_index_from_checkpoint(
                text_checkpoint.as_ref(),
                &inner.catalog,
                |_| true,
                &context,
                inner.active_cache_budget_mb().saturating_mul(1_048_576),
            )?
        } else {
            TextIndex::default()
        };
        let mut compacted_catalog = Catalog::new_in(&context);
        // Test pressure is admitted through the real shared ledger at the
        // publication boundary; it does not replace a production operation.
        #[cfg(test)]
        let file_pressure = inner.compact_file_remaining.take().map(|remaining| {
            context
                .cache_reservation(
                    (context.available_total_bytes() - remaining) as u128,
                    usize::MAX,
                )
                .unwrap()
        });
        let mut validated_records = 0usize;
        let scan = {
            let Inner {
                storage,
                catalog,
                text_index: previous_text,
                persistent_vector_segments,
                lazy_vector_segment,
                ..
            } = &mut *inner;
            let reader = storage.payload_reader()?;
            let records = compaction_records(catalog, reader, &context)
                .chain(std::iter::once(Ok((marker, context.reservation().into()))));
            rewrite_database_retained_in(
                storage,
                records,
                rewrite_plan,
                durability,
                &context,
                || {
                    let released = text_index.clear_cache();
                    Inner::release_optional_caches(
                        previous_text,
                        persistent_vector_segments,
                        lazy_vector_segment,
                        std::iter::empty(),
                    ) || released
                },
                |record, work| {
                    compacted_catalog.apply_retained(record, work, &context)?;
                    validated_records = validated_records.checked_add(1).ok_or_else(count_error)?;
                    // Storage invokes this during validation, before publishing its private file.
                    if validated_records == record_count
                        && (compacted_catalog.business_record_count() != business_records_after
                            || compacted_catalog.tombstone_count() != 0)
                    {
                        return Err(Error::new(
                            ErrorCode::FileFormatError,
                            "inconsistent compaction record counts",
                        ));
                    }
                    Ok(())
                },
            )?
        };
        inner.mark_index_checkpoint_rebuild_needed();
        compacted_catalog.last_commit_id = scan.last_valid_commit;
        inner.catalog = compacted_catalog;
        inner.last_commit_sequence_no = scan.last_valid_commit_sequence_no;
        inner.text_index = text_index;
        #[cfg(test)]
        let checkpoint_pressure = {
            inner.compact_rewrite_completions += 1;
            inner
                .compact_checkpoint_remaining
                .take()
                .map(|(remaining, parent)| {
                    if parent {
                        (
                            Some(
                                context
                                    .cache_reservation(
                                        (context.available_total_bytes() - remaining) as u128,
                                        usize::MAX,
                                    )
                                    .unwrap(),
                            ),
                            None,
                        )
                    } else {
                        let mut pressure = context.workspace_reservation();
                        let held = context.available_workspace_bytes() - remaining;
                        pressure.grow_to(held, held, "operation_scratch").unwrap();
                        (None, Some(pressure))
                    }
                })
        };
        let checkpoint_error = if options.rebuild_indexes {
            inner
                .refresh_index_checkpoint_prepared(
                    VectorCache::new(&context),
                    &BTreeSet::new(),
                    text_checkpoint,
                    &context,
                )
                .err()
        } else {
            inner.index_suffix_status = IndexSuffixState::Missing;
            inner.last_index_commit_sequence_no = None;
            inner.index_size_bytes = 0;
            inner.persistent_vector_segments = None;
            None
        };
        #[cfg(test)]
        drop((file_pressure, checkpoint_pressure));
        let after = inner.storage.file_size()?;
        let (index_rebuild, index_rebuild_reason) = if let Some(err) = checkpoint_error {
            (
                "failed".into(),
                Some(format!("index_checkpoint_write_failed: {err}")),
            )
        } else if options.rebuild_indexes {
            ("full_checkpoint_written".into(), None)
        } else {
            ("disabled".into(), None)
        };
        Ok(CompactSummary {
            before_file_size: before,
            after_file_size: after,
            records_removed,
            tombstones_removed,
            segments_merged,
            index_rebuild,
            index_rebuild_reason,
            duration_ms: started.elapsed().as_millis(),
        })
    }

    pub fn close(self) -> Result<()> {
        self.close_with(Storage::flush_committed)
    }

    fn close_with(self, finish: impl FnOnce(&mut Storage) -> Result<()>) -> Result<()> {
        let (mut slot, poisoned) = match self.inner.lock() {
            Ok(guard) => (guard, false),
            Err(error) => (error.into_inner(), true),
        };
        let Some(mut inner) = slot.take() else {
            return Ok(());
        };
        // Keep the shared lock until both fallible flushing and resource release finish.
        let result = finish(&mut inner.storage);
        drop(inner);
        drop(slot);
        result.and_then(|()| {
            if poisoned {
                Err(Error::new(
                    ErrorCode::LockContention,
                    "database lock poisoned",
                ))
            } else {
                Ok(())
            }
        })
    }

    pub(crate) fn json_context(&self) -> crate::json::JsonOperationContext {
        crate::json::JsonOperationContext::new(Arc::clone(&self.json_work))
    }

    #[cfg(test)]
    pub(crate) fn index_directory_bytes(&self) -> usize {
        self.lock()
            .unwrap()
            .lazy_vector_source
            .as_ref()
            .map_or(0, IndexSegmentSource::reserved_bytes)
    }

    pub(crate) fn temporary_disk_budget(&self) -> Arc<crate::temporary::TemporaryDiskBudget> {
        Arc::clone(&self.temporary_work)
    }

    pub(crate) fn attach_json_context(
        &self,
        context: &crate::json::JsonOperationContext,
        kind: crate::json::JsonInputKind,
    ) -> Result<()> {
        self.admit_input_reclaiming(|| context.attach(Arc::clone(&self.json_work), kind))
    }

    // Only repeat input admission, before acquiring Inner or performing writes.
    // The attempt must retain any already admitted live input leases.
    pub(crate) fn admit_input_reclaiming<T>(
        &self,
        mut admit: impl FnMut() -> Result<T>,
    ) -> Result<T> {
        match admit() {
            Err(error) if crate::json::is_memory_budget_error(&error) => {
                let mut guard = self.lock()?;
                let inner = &mut *guard;
                let reclaimed = Inner::release_optional_caches(
                    &mut inner.text_index,
                    &mut inner.persistent_vector_segments,
                    &mut inner.lazy_vector_segment,
                    std::iter::empty(),
                );
                drop(guard);
                if reclaimed {
                    admit()
                } else {
                    Err(error)
                }
            }
            result => result,
        }
    }

    pub(crate) fn insert_document_in_collection(
        &self,
        collection_id: &str,
        input: DocumentInput,
        on_conflict: OnConflict,
    ) -> Result<InsertSummary> {
        let context = self.json_context();
        let mut input_strings = context.workspace_reservation();
        let mut input_work = context.reservation();
        // Drop remaining input allocations before their lease on every early return.
        let input = input;
        input_work.merge(
            self.admit_input_reclaiming(|| input.validate_owned_in(&context, &mut input_strings))?,
        );
        let mut inner = self.lock()?;
        let (catalog, mut records) = inner.prepare_write_batch(&context);
        let collection = catalog.collection_by_id(collection_id)?;
        let existing = catalog
            .document_external_ids
            .get(collection_id, &input.external_id)
            .filter(|id| catalog.is_document_live(id));
        let mut documents_replaced = 0;
        if let Some(existing_id) = existing {
            match on_conflict {
                OnConflict::Error => {
                    return Err(Error::new(
                        ErrorCode::DuplicateExternalId,
                        format!("document external_id {} already exists", input.external_id),
                    ));
                }
                OnConflict::Skip => {
                    return Ok(InsertSummary {
                        documents_skipped: 1,
                        chunks_skipped: input.chunks.len(),
                        ..InsertSummary::default()
                    });
                }
                OnConflict::Replace => {
                    documents_replaced += 1;
                    records.push_tombstone("document", existing_id, "replace")?;
                }
            }
        }

        let now = time::unix_millis_now();
        let document_id = records.admit(|| catalog.next_id_in("doc_", 0, &context))?;
        let document = Document {
            document_id: records.copy_string(document_id.as_ref())?,
            collection_id: records.copy_string(&collection.collection_id)?,
            external_id: input.external_id,
            title: input.title,
            source_uri: input.source_uri,
            metadata: input.metadata,
            created_at: records.timestamp(now)?,
            updated_at: records.timestamp(now)?,
        };
        records.push(RecordPayload::DocumentPut(document))?;
        let mut chunks_inserted = 0;
        let mut chunks_skipped = 0;
        let mut chunks_replaced = 0;
        let mut chunk_offset = 0;
        let input_array_bytes = input.chunks.capacity() * std::mem::size_of::<crate::ChunkInput>();
        for chunk_input in input.chunks {
            if let Some(existing_chunk) = catalog
                .chunk_external_ids
                .get(collection_id, &chunk_input.chunk_external_id)
                .filter(|id| catalog.is_chunk_live(id))
            {
                match on_conflict {
                    OnConflict::Error => {
                        return Err(Error::new(
                            ErrorCode::DuplicateExternalId,
                            format!(
                                "chunk_external_id {} already exists",
                                chunk_input.chunk_external_id
                            ),
                        ));
                    }
                    OnConflict::Skip => {
                        chunks_skipped += 1;
                        let bytes = chunk_input.workspace_bytes();
                        let json_bytes = chunk_input.metadata.retained_bytes_in(
                            crate::json::JsonInputKind::ChunkMetadata,
                            &context,
                        )?;
                        drop(chunk_input);
                        input_strings.retain(input_strings.reserved_bytes() - bytes);
                        input_work.retain(input_work.reserved_bytes() - json_bytes);
                        continue;
                    }
                    OnConflict::Replace => {
                        chunks_replaced += 1;
                        records.push_tombstone("chunk", existing_chunk, "replace")?;
                    }
                }
            }
            let source_span = chunk_input.resolved_source_span()?;
            let token_count = records.admit(|| token_count_hint_in(&chunk_input.text, &context))?;
            let chunk_id = records.admit(|| catalog.next_id_in("chk_", chunk_offset, &context))?;
            let chunk = Chunk {
                chunk_id: records.retain_string(chunk_id),
                document_id: records.copy_string(document_id.as_ref())?,
                collection_id: records.copy_string(&collection.collection_id)?,
                chunk_external_id: chunk_input.chunk_external_id,
                text_hash: records.text_hash(&chunk_input.text)?,
                token_count,
                text: chunk_input.text,
                ordinal: chunk_input.ordinal,
                metadata: chunk_input.metadata,
                embedding_id: None,
                source_span,
                created_at: records.timestamp(now)?,
            };
            chunk_offset += 1;
            chunks_inserted += 1;
            records.push(RecordPayload::ChunkPut(chunk))?;
        }
        drop(document_id);
        input_strings.retain(input_strings.reserved_bytes() - input_array_bytes);
        records.retain_json(input_work);
        records.retain_workspace(input_strings);
        let records = records.into_batch();
        inner.append_and_apply(records, &context)?;
        Ok(InsertSummary {
            documents_inserted: 1,
            chunks_inserted,
            chunks_skipped,
            documents_replaced,
            chunks_replaced,
            ..InsertSummary::default()
        })
    }

    pub(crate) fn put_embeddings_in_collection(
        &self,
        collection_id: &str,
        input: Vec<EmbeddingInput>,
        on_conflict: OnConflict,
    ) -> Result<InsertSummary> {
        let context = self.json_context();
        let mut input_work = context.workspace_reservation();
        // The input array must be dropped before its workspace lease on failure.
        let embeddings = input;
        let input_array_bytes = embeddings.capacity() * std::mem::size_of::<EmbeddingInput>();
        let bytes = input_array_bytes
            + embeddings
                .iter()
                .map(EmbeddingInput::workspace_bytes)
                .sum::<usize>();
        self.admit_input_reclaiming(|| input_work.grow_to(bytes, bytes, "batch_prepare"))?;
        let mut inner = self.lock()?;
        let (catalog, mut records) = inner.prepare_write_batch(&context);
        let collection = catalog.collection_by_id(collection_id)?;
        let mut inserted = 0;
        let mut skipped = 0;
        let mut replaced = 0;
        let mut embedding_offset = 0;
        let duplicate = records.admit(|| {
            crate::model::first_duplicate_index_by(&embeddings, &context, |left, right| {
                left.chunk_external_id.cmp(&right.chunk_external_id)
            })
        })?;
        for input in &embeddings {
            input.validate_structure()?;
        }
        if let Some(index) = duplicate {
            return Err(Error::new(
                ErrorCode::DuplicateExternalId,
                format!(
                    "embedding for chunk {} appears more than once in the same batch",
                    embeddings[index].chunk_external_id
                ),
            ));
        }
        let mut models = records.admit(|| ModelAdmission::new(catalog, collection, &context))?;
        for input in embeddings {
            input.validate_target_dimension(collection.embedding_dim)?;
            let chunk = catalog.chunk_by_external_id(collection_id, &input.chunk_external_id)?;
            if catalog.embedding_for_chunk(&chunk.chunk_id).is_some() {
                match on_conflict {
                    OnConflict::Error => {
                        return Err(Error::new(
                            ErrorCode::DuplicateExternalId,
                            format!(
                                "embedding for chunk {} already exists",
                                input.chunk_external_id
                            ),
                        ));
                    }
                    OnConflict::Skip => {
                        skipped += 1;
                        let bytes = input.workspace_bytes();
                        drop(input);
                        input_work.retain(input_work.reserved_bytes() - bytes);
                        continue;
                    }
                    OnConflict::Replace => replaced += 1,
                }
            }
            records.admit(|| models.validate_write(&input.model_name, None, &context))?;
            let embedding_id =
                records.admit(|| catalog.next_id_in("emb_", embedding_offset, &context))?;
            let embedding = Embedding {
                embedding_id: records.retain_string(embedding_id),
                chunk_id: records.copy_string(&chunk.chunk_id)?,
                collection_id: records.copy_string(&collection.collection_id)?,
                chunk_external_id: input.chunk_external_id,
                model_name: input.model_name,
                dim: input.dim,
                vector: input.vector,
                created_at: records.timestamp(time::unix_millis_now())?,
            };
            embedding_offset += 1;
            inserted += 1;
            records.push(RecordPayload::EmbeddingPut(embedding))?;
        }
        input_work.retain(input_work.reserved_bytes() - input_array_bytes);
        if !records.records().is_empty() {
            records.retain_workspace(input_work);
            let records = records.into_batch();
            inner.append_and_apply(records, &context)?;
        }
        Ok(InsertSummary {
            embeddings_inserted: inserted,
            embeddings_skipped: skipped,
            embeddings_replaced: replaced,
            ..InsertSummary::default()
        })
    }

    pub(crate) fn get_document_in_collection(
        &self,
        collection_id: &str,
        document_id: &str,
    ) -> Result<Document> {
        let context = self.json_context();
        self.lock()?.read_reclaiming(|inner| {
            let document = inner
                .catalog
                .document_by_id_ref(collection_id, document_id)?;
            copy_document_in(document, &context).map(|(document, _)| document)
        })
    }

    pub(crate) fn get_document_by_external_id_in_collection(
        &self,
        collection_id: &str,
        external_id: &str,
    ) -> Result<Document> {
        let context = self.json_context();
        self.lock()?.read_reclaiming(|inner| {
            let document = inner
                .catalog
                .document_by_external_id(collection_id, external_id)?;
            copy_document_in(document, &context).map(|(document, _)| document)
        })
    }

    pub(crate) fn list_chunks_in_collection(
        &self,
        collection_id: &str,
        document_id: &str,
    ) -> Result<Vec<Chunk>> {
        let context = self.json_context();
        self.lock()?.read_reclaiming(|inner| {
            inner.materialize_document_chunks_in(collection_id, document_id, &context)
        })
    }

    pub(crate) fn list_chunks_by_document_external_id_in_collection(
        &self,
        collection_id: &str,
        external_id: &str,
    ) -> Result<Vec<Chunk>> {
        let context = self.json_context();
        self.lock()?.read_reclaiming(|inner| {
            let document = inner
                .catalog
                .document_by_external_id(collection_id, external_id)?;
            inner.materialize_document_chunks_in(collection_id, &document.document_id, &context)
        })
    }

    pub(crate) fn get_chunk_in_collection(
        &self,
        collection_id: &str,
        chunk_id: &str,
    ) -> Result<Chunk> {
        let context = self.json_context();
        self.lock()?.read_reclaiming(|inner| {
            inner.catalog.chunk_by_id(collection_id, chunk_id)?;
            inner
                .materialize_chunk_in(chunk_id, &context)
                .map(|(chunk, _)| chunk)
        })
    }

    pub(crate) fn get_chunk_by_external_id_in_collection(
        &self,
        collection_id: &str,
        chunk_external_id: &str,
    ) -> Result<Chunk> {
        let context = self.json_context();
        self.lock()?.read_reclaiming(|inner| {
            let chunk = inner
                .catalog
                .chunk_by_external_id(collection_id, chunk_external_id)?;
            inner
                .materialize_chunk_in(&chunk.chunk_id, &context)
                .map(|(chunk, _)| chunk)
        })
    }

    pub(crate) fn delete_document_in_collection(
        &self,
        collection_id: &str,
        document_id: &str,
    ) -> Result<()> {
        let context = self.json_context();
        let mut inner = self.lock()?;
        let (catalog, mut records) = inner.prepare_write_batch(&context);
        catalog.document_by_id_ref(collection_id, document_id)?;
        records.push_tombstone("document", document_id, "delete_document")?;
        let records = records.into_batch();
        inner.append_and_apply(records, &context)
    }

    pub(crate) fn delete_chunk_in_collection(
        &self,
        collection_id: &str,
        chunk_id: &str,
    ) -> Result<()> {
        let context = self.json_context();
        let mut inner = self.lock()?;
        let (catalog, mut records) = inner.prepare_write_batch(&context);
        catalog.chunk_by_id(collection_id, chunk_id)?;
        records.push_tombstone("chunk", chunk_id, "delete_chunk")?;
        let records = records.into_batch();
        inner.append_and_apply(records, &context)
    }

    pub(crate) fn collection_stats_by_id(&self, collection_id: &str) -> Result<CollectionStats> {
        let context = self.json_context();
        let (stats, _work) = self
            .lock()?
            .read_reclaiming(|inner| inner.collection_stats_retained(collection_id, &context))?;
        Ok(stats)
    }

    pub(crate) fn with_read_snapshot<T>(
        &self,
        read: impl FnOnce(&Catalog, &Storage) -> Result<T>,
    ) -> Result<T> {
        let inner = self.lock()?;
        read(&inner.catalog, &inner.storage)
    }

    pub(crate) fn with_read_snapshot_reclaiming<T>(
        &self,
        read: impl FnOnce(&Catalog, &Storage, &mut dyn FnMut() -> bool) -> Result<T>,
    ) -> Result<T> {
        let mut guard = self.lock()?;
        let Inner {
            catalog,
            storage,
            text_index,
            persistent_vector_segments,
            lazy_vector_segment,
            ..
        } = &mut *guard;
        let mut reclaim = || {
            Inner::release_optional_caches(
                text_index,
                persistent_vector_segments,
                lazy_vector_segment,
                std::iter::empty(),
            )
        };
        read(catalog, storage, &mut reclaim)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn search_in_collection(
        &self,
        collection_id: &str,
        text: Option<&str>,
        vector: Option<&QueryEmbeddingInput>,
        filter_json: &JsonValue,
        requested_weights: Option<SearchWeights>,
        metadata_boost_enabled: bool,
        metadata_boost_config: Option<&MetadataBoostConfig>,
        top_k: usize,
        profile_override: Option<ResourceProfile>,
        prepared_context: Option<crate::json::JsonOperationContext>,
    ) -> Result<(SearchResults, RecordReservation)> {
        self.search_in_collection_traced(
            collection_id,
            text,
            vector,
            filter_json,
            requested_weights,
            metadata_boost_enabled,
            metadata_boost_config,
            top_k,
            profile_override,
            prepared_context,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn search_in_collection_traced(
        &self,
        collection_id: &str,
        text: Option<&str>,
        vector: Option<&QueryEmbeddingInput>,
        filter_json: &JsonValue,
        requested_weights: Option<SearchWeights>,
        metadata_boost_enabled: bool,
        metadata_boost_config: Option<&MetadataBoostConfig>,
        top_k: usize,
        profile_override: Option<ResourceProfile>,
        prepared_context: Option<crate::json::JsonOperationContext>,
        trace: Option<&crate::search_cost::SearchCostTrace>,
    ) -> Result<(SearchResults, RecordReservation)> {
        let _trace_scope = trace.map(crate::search_cost::SearchCostTrace::scope);
        let text = normalize_search_text(text);
        crate::query::validate_search_top_k(top_k)?;
        if text.is_none() && vector.is_none() {
            return Err(Error::new(
                ErrorCode::InvalidSearchParams,
                "search requires text or vector input",
            )
            .with_detail("text_present", false)
            .with_detail("embedding_present", false));
        }

        let preprocessed = prepared_context.is_some();
        let context = prepared_context.unwrap_or_else(|| self.json_context());
        let context = match trace {
            Some(trace) => context.with_search_cost(trace),
            None => context,
        };
        let phase_timer = context.search_cost_timer();
        if preprocessed {
            let kind = if metadata_boost_config.is_some() {
                crate::json::JsonInputKind::MetadataBoost
            } else if vector.is_some()
                && (filter_json.is_null()
                    || filter_json
                        .as_object()
                        .is_some_and(|value| value.is_empty()))
            {
                crate::json::JsonInputKind::QueryEmbedding
            } else {
                crate::json::JsonInputKind::Filter
            };
            self.attach_json_context(&context, kind)?;
        }
        // The CLI retains its parsed-input reservations until this call completes.
        let _input_work = (!preprocessed)
            .then(|| {
                self.admit_input_reclaiming(|| {
                    reserve_query_json_inputs(&context, vector, filter_json, metadata_boost_config)
                })
            })
            .transpose()?;
        let mut guard = self.lock()?;
        let inner = &mut *guard;
        let (mut collection, mut collection_work) = inner.read_reclaiming(|inner| {
            copy_collection_in(inner.catalog.collection_by_id(collection_id)?, &context)
        })?;
        if let Some(query_embedding) = &vector {
            query_embedding.validate_structure(&collection)?;
            inner.cache_collection_model(collection_id)?;
            inner
                .catalog
                .visible_model_state(collection_id)?
                .validate(query_embedding.model_name.as_deref())?;
        }
        let profile = profile_override.unwrap_or(inner.config.resource_profile);
        let params = profile.params();
        let cache_budget_mb = inner
            .config
            .cache_budget_mb
            .unwrap_or(params.cache_budget_mb);
        if inner.optional_cache_bytes() > cache_budget_mb.saturating_mul(1_048_576) {
            Inner::release_optional_caches(
                &mut inner.text_index,
                &mut inner.persistent_vector_segments,
                &mut inner.lazy_vector_segment,
                std::iter::empty(),
            );
        }
        let limits = CandidateLimits {
            bm25: params.bm25_candidates.max(top_k),
            vector: params.vector_candidates.max(top_k),
            fusion: params.fusion_candidates.max(top_k),
        };
        let (filter_expr, _compiled_filter_work) = inner.read_reclaiming(|_| {
            FilterExpr::parse_validated_in(
                filter_json,
                &context,
                crate::json::JsonInputKind::Filter,
            )
        })?;
        let metadata_boost_present = metadata_boost_config.is_some();
        let boost_rule_count = metadata_boost_config
            .map(MetadataBoostConfig::rule_count)
            .unwrap_or(0);
        let mut result_work =
            RecordReservation::new(context.reservation(), context.workspace_reservation());
        let boost_config_hash = metadata_boost_config
            .as_ref()
            .map(|config| result_work.copy_string(config.hash(), &context))
            .transpose()?;
        let metadata_boost_available =
            metadata_boost_present && metadata_boost_enabled && boost_rule_count > 0;
        let weight_context = effective_weights_with(
            requested_weights,
            text.is_some(),
            vector.is_some(),
            metadata_boost_available,
            |value| result_work.copy_string(value, &context),
        )?;
        let checkpoint_ready = inner.index_suffix_status == IndexSuffixState::PersistentCheckpoint;
        if !checkpoint_ready {
            // Queries cannot reuse the text cache retained for a failed or deferred update.
            inner.text_index = TextIndex::default();
        }
        let scratch_bytes = (!checkpoint_ready).then(|| {
            transient_query_scratch_bytes(
                &inner.catalog,
                collection_id,
                text,
                TextSearchOptions::with_explanations(limits.bm25),
                vector.is_some(),
                limits.vector,
            )
        });
        let query_context_input = SearchQueryContextInput {
            text_present: text.is_some(),
            embedding_present: vector.is_some(),
            filter_present: filter_expr.is_some(),
            weights: weight_context,
            metadata_boost_present,
            metadata_boost_enabled: metadata_boost_available,
            boost_config_hash,
            boost_rule_count,
            top_k,
            profile,
            candidate_limits: limits,
            cache_budget_mb,
            chunks_scanned: if checkpoint_ready && inner.text_index.has_resident_cache() {
                inner.text_index.collection_chunk_count(collection_id)
            } else {
                inner
                    .catalog
                    .live_chunks()
                    .filter(|chunk| chunk.collection_id == collection.collection_id)
                    .count()
            },
            bytes_read: Some(0),
        };
        if let Some(scratch_bytes) = scratch_bytes {
            ensure_transient_rebuild_budget(scratch_bytes, cache_budget_mb, profile)?;
        }
        if text.is_some() && checkpoint_ready {
            inner.prepare_text_index(cache_budget_mb, &context)?;
        }
        if vector.is_some() {
            inner.prepare_vector_index_with_context(collection_id, cache_budget_mb, &context)?;
        }
        #[cfg(test)]
        inner.run_query_output_hook(QueryOutputStage::CheckpointCandidates);
        drop(phase_timer);
        let context = context.search_cost_stage(crate::search_cost::Stage::Recall);
        let phase_timer = context.search_cost_timer();
        let cached_hybrid = text.is_some()
            && vector.is_some()
            && (inner
                .persistent_vector_segments
                .as_ref()
                .is_some_and(|segments| segments.contains_key(collection_id))
                || inner
                    .lazy_vector_segment
                    .as_ref()
                    .is_some_and(|segment| segment.collection_id == collection_id));
        let mut streamed_text = if cached_hybrid {
            let query = text.unwrap();
            let hits = inner.search_checkpoint_text_index(
                collection_id,
                query,
                TextSearchOptions::with_explanations(limits.bm25),
                &context,
            )?;
            if hits.is_none() && inner.index_suffix_status == IndexSuffixState::PersistentCheckpoint
            {
                // Preserve cached hybrid's text-first peak: term scratch must
                // finish before vector candidates start retaining their IDs.
                Some(inner.read_reclaiming(|inner| {
                    inner.text_index.search(
                        collection_id,
                        query,
                        TextSearchOptions::with_explanations(limits.bm25),
                        &context,
                    )
                })?)
            } else {
                hits
            }
        } else {
            None
        };
        let mut vector_work = VectorWork::default();
        let mut streamed_vectors = if let Some(query) = vector {
            inner.search_checkpoint_vector_index_with_work(
                &collection,
                &query.vector,
                limits.vector,
                filter_expr.as_ref(),
                &context,
                &mut vector_work,
            )?
        } else {
            None
        };
        if !cached_hybrid {
            streamed_text = if let Some(query) = text {
                inner.search_checkpoint_text_index(
                    collection_id,
                    query,
                    TextSearchOptions::with_explanations(limits.bm25),
                    &context,
                )?
            } else {
                None
            };
        }
        if let Some(query) = vector.filter(|_| streamed_vectors.is_none()) {
            streamed_vectors = inner.search_checkpoint_vector_index_with_work(
                &collection,
                &query.vector,
                limits.vector,
                filter_expr.as_ref(),
                &context,
                &mut vector_work,
            )?;
        }
        if checkpoint_ready && inner.index_suffix_status != IndexSuffixState::PersistentCheckpoint {
            // Lazy vector validation can invalidate the checkpoint after query preflight.
            inner.text_index = TextIndex::default();
        }
        let use_persistent_checkpoint = inner.index_suffix_status
            == IndexSuffixState::PersistentCheckpoint
            && (vector.is_none() || streamed_vectors.is_some());
        if !use_persistent_checkpoint {
            drop(streamed_text.take());
        }
        let mut query_context = SearchQueryContext::from_checkpoint_with(
            query_context_input,
            use_persistent_checkpoint,
            |value| result_work.copy_string(value, &context),
        )?;
        if !use_persistent_checkpoint {
            // Lazy vector validation can invalidate a checkpoint during this query.
            let scratch_bytes = scratch_bytes.unwrap_or_else(|| {
                transient_query_scratch_bytes(
                    &inner.catalog,
                    collection_id,
                    text,
                    TextSearchOptions::with_explanations(limits.bm25),
                    vector.is_some(),
                    limits.vector,
                )
            });
            ensure_transient_rebuild_budget(scratch_bytes, cache_budget_mb, profile)?;
        }

        let mut transient_bytes_read = 0u64;
        let mut bm25_hits = if let Some(query) = &text {
            if use_persistent_checkpoint {
                if let Some(hits) = streamed_text {
                    hits
                } else {
                    inner.text_index.search(
                        collection_id,
                        query,
                        TextSearchOptions::with_explanations(limits.bm25),
                        &context,
                    )?
                }
            } else {
                let mut reader = inner.storage.payload_reader()?;
                TextIndex::search_from_snapshot(
                    &inner.catalog.visible_snapshot(),
                    collection_id,
                    query,
                    TextSearchOptions::with_explanations(limits.bm25),
                    &context,
                    |visible| {
                        read_visible_chunk_text(
                            &mut reader,
                            visible,
                            &context,
                            &mut transient_bytes_read,
                        )
                    },
                )?
            }
        } else {
            Bm25Hits::empty(&context)
        };
        let include_vector = |id: &str| {
            matches_vector_filter(&inner.catalog, collection_id, filter_expr.as_ref(), id)
        };
        let mut vector_hits = if let Some(query_embedding) = &vector {
            if let Some(hits) = streamed_vectors {
                hits
            } else {
                let mut payload_reader = inner.storage.payload_reader()?;
                let vector_index = ExactVectorIndex;
                let (hits, work) = vector_index.search_with_vectors(
                    &inner.catalog.visible_snapshot(),
                    collection_id,
                    &query_embedding.vector,
                    VectorSearchOptions {
                        metric: collection.distance_metric,
                        vectors_normalized: collection.vectors_normalized,
                        limit: limits.vector,
                    },
                    include_vector,
                    &context,
                    |visible| {
                        read_visible_embedding_vector(
                            &mut payload_reader,
                            visible,
                            &context,
                            &mut transient_bytes_read,
                        )
                    },
                )?;
                vector_work.add(work);
                hits
            }
        } else {
            VectorHits::empty(&context)
        };
        let vector_scan_count = vector_work.scans;
        if !use_persistent_checkpoint {
            query_context.transient_rebuild.bytes_read = Some(transient_bytes_read);
            if text.is_none() {
                query_context.transient_rebuild.chunks_scanned = Some(vector_scan_count);
            }
        }
        let distance_ops = vector_work.distance_ops;

        let text_has_matches = !bm25_hits.is_empty();
        filter_candidates(
            &inner.catalog,
            &mut bm25_hits,
            &mut vector_hits,
            filter_expr.as_ref(),
        );
        // Retrieval order is no longer needed; borrow explanations without copying term tables.
        bm25_hits.sort_by_id();
        context.search_cost_count(crate::search_cost::Count::Bm25Recall, bm25_hits.len());
        context.search_cost_count(crate::search_cost::Count::VectorRecall, vector_hits.len());
        if trace.is_some() {
            let union = bm25_hits.len()
                + vector_hits
                    .iter()
                    .filter(|hit| {
                        bm25_hits
                            .binary_search_by(|row| row.chunk_id.cmp(&hit.chunk_id))
                            .is_err()
                    })
                    .count();
            context.search_cost_count(crate::search_cost::Count::RecallUnion, union);
        }
        drop(phase_timer);
        let context = context.search_cost_stage(crate::search_cost::Stage::ScoreCompletion);
        let phase_timer = context.search_cost_timer();
        #[cfg(test)]
        inner.run_query_output_hook(QueryOutputStage::Fusion);
        let mut supplementary_work = VectorWork::default();
        let mut supplementary_bytes_read = 0;
        let mut supplement_fallback = false;
        let supplement = inner.read_reclaiming(|inner| {
            supplement_scores(
                inner,
                &collection,
                text,
                text_has_matches,
                vector,
                &bm25_hits,
                &vector_hits,
                &query_context,
                &context,
                &mut supplementary_work,
                &mut supplementary_bytes_read,
                &mut supplement_fallback,
            )
        })?;
        if supplement.invalidated_text {
            inner.mark_index_checkpoint_rebuild_needed();
            query_context.index_source.clear();
            query_context.index_source.push_str("transient_rebuild");
            query_context.index_rebuild_reason =
                Some(result_work.copy_string("in_memory_index_from_committed_snapshot", &context)?);
            query_context.transient_rebuild.started = true;
            query_context.transient_rebuild.completed = true;
            query_context.transient_rebuild.chunks_scanned = Some(
                inner
                    .catalog
                    .live_chunks()
                    .filter(|chunk| chunk.collection_id == collection_id)
                    .count(),
            );
        }
        drop(phase_timer);
        let context = context.search_cost_stage(crate::search_cost::Stage::NormalizationFusion);
        let phase_timer = context.search_cost_timer();
        let completed = inner.read_reclaiming(|inner| {
            fuse_completed_candidates(
                &inner.catalog,
                &bm25_hits,
                &vector_hits,
                &supplement.text,
                &supplement.vector,
                text.is_some(),
                metadata_boost_config.filter(|_| metadata_boost_available),
                query_context.weights.effective,
                limits.fusion,
                &context,
            )
        })?;
        let normalization_candidate_count = completed.candidate_count;
        let bm25_max = text.map(|_| completed.bm25_max);
        let fused = completed.hits;
        let fused_scores = fused.as_slice();
        let fusion_candidate_count = fused_scores.len();
        context.search_cost_count(
            crate::search_cost::Count::FusionRetained,
            fusion_candidate_count,
        );
        let vector_scan_count = vector_scan_count + supplement.vector_scans;
        let distance_ops = distance_ops + supplement.distance_ops;
        if !use_persistent_checkpoint || supplement.invalidated_text {
            query_context.transient_rebuild.bytes_read =
                Some(transient_bytes_read + supplement.bytes_read);
        }
        query_context.candidate_counts = CandidateLimits {
            bm25: bm25_hits.len(),
            vector: vector_hits.len(),
            fusion: fusion_candidate_count,
        };
        let execution_stats = SearchExecutionStats {
            vector_scan_count,
            distance_ops,
            fusion_candidate_count,
        };
        query_context.execution_stats = execution_stats;
        drop(phase_timer);
        let context = context.search_cost_stage(crate::search_cost::Stage::ResultMaterialization);
        let _phase_timer = context.search_cost_timer();
        #[cfg(test)]
        inner.run_query_output_hook(QueryOutputStage::ResultArray);
        let mut results = inner.read_reclaiming(|_| {
            crate::json::ReservedVec::with_capacity(
                fused_scores.len().min(top_k),
                &context,
                "operation_scratch",
            )
        })?;
        for (rank_index, &score) in fused_scores.iter().take(top_k).enumerate() {
            let matched_terms = bm25_hits
                .binary_search_by(|hit| hit.chunk_id.as_str().cmp(score.chunk_id))
                .map(|index| bm25_hits[index].matched_terms.as_slice())
                .unwrap_or_else(|_| {
                    supplement
                        .text
                        .binary_search_by(|hit| hit.chunk_id.as_str().cmp(score.chunk_id))
                        .map(|index| supplement.text[index].matched_terms.as_slice())
                        .unwrap_or_default()
                });
            #[cfg(test)]
            inner.run_query_output_hook(QueryOutputStage::ResultRow(rank_index));
            // A failed row releases its partial fields without releasing earlier rows.
            let (result, row_work) = inner.read_reclaiming(|inner| {
                let mut row_work =
                    RecordReservation::new(context.reservation(), context.workspace_reservation());
                let result = build_result(
                    inner,
                    &collection,
                    score,
                    matched_terms,
                    filter_expr.as_ref(),
                    filter_json,
                    &query_context,
                    profile,
                    limits,
                    execution_stats,
                    normalization_candidate_count,
                    bm25_max,
                    &context,
                    &mut row_work,
                )?;
                Ok((result, row_work))
            })?;
            if let Some(result) = result {
                results.push(SearchResult {
                    rank: rank_index + 1,
                    ..result
                });
                result_work.merge(row_work);
            }
        }
        drop(fused);
        drop(vector_hits);
        drop(bm25_hits);
        let (results, array_work) = results.into_parts();
        result_work.merge_workspace(array_work);
        let name = std::mem::take(&mut collection.name);
        drop(collection);
        collection_work.json.retain(0);
        collection_work.retain_workspace(name.capacity());
        result_work.merge(collection_work);
        let results = SearchResults {
            schema_version: result_work.copy_string("0.1", &context)?,
            collection: name,
            query: text
                .map(|value| result_work.copy_string(value, &context))
                .transpose()?,
            top_k,
            profile,
            query_context,
            results,
        };
        context.search_cost_count(crate::search_cost::Count::Returned, results.results.len());
        Ok((results, result_work))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn why_not_debug(
        &self,
        collection: &str,
        expected_chunk_id: &str,
        text: Option<String>,
        vector: Option<QueryEmbeddingInput>,
        filter: JsonValue,
        requested_weights: Option<SearchWeights>,
        metadata_boost_enabled: bool,
        top_k: usize,
        profile: Option<ResourceProfile>,
    ) -> Result<WhyNotDiagnosis> {
        self.why_not_debug_with_metadata_boost(
            collection,
            expected_chunk_id,
            text,
            vector,
            filter,
            requested_weights,
            metadata_boost_enabled,
            None,
            top_k,
            profile,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn why_not_debug_with_metadata_boost(
        &self,
        collection: &str,
        expected_chunk_id: &str,
        text: Option<String>,
        vector: Option<QueryEmbeddingInput>,
        filter: JsonValue,
        requested_weights: Option<SearchWeights>,
        metadata_boost_enabled: bool,
        metadata_boost_config: Option<MetadataBoostConfig>,
        top_k: usize,
        profile: Option<ResourceProfile>,
    ) -> Result<WhyNotDiagnosis> {
        self.why_not_debug_with_metadata_boost_in(
            collection,
            expected_chunk_id,
            text,
            vector,
            filter,
            requested_weights,
            metadata_boost_enabled,
            metadata_boost_config,
            top_k,
            profile,
            None,
        )
        .map(|(diagnosis, _)| diagnosis)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn why_not_debug_with_metadata_boost_in(
        &self,
        collection: &str,
        expected_chunk_id: &str,
        text: Option<String>,
        vector: Option<QueryEmbeddingInput>,
        filter: JsonValue,
        requested_weights: Option<SearchWeights>,
        metadata_boost_enabled: bool,
        metadata_boost_config: Option<MetadataBoostConfig>,
        top_k: usize,
        profile: Option<ResourceProfile>,
        prepared_context: Option<crate::json::JsonOperationContext>,
    ) -> Result<(WhyNotDiagnosis, RecordReservation)> {
        let preprocessed = prepared_context.is_some();
        let context = prepared_context.unwrap_or_else(|| self.json_context());
        let _input_work = (!preprocessed)
            .then(|| {
                self.admit_input_reclaiming(|| {
                    reserve_query_json_inputs(
                        &context,
                        vector.as_ref(),
                        &filter,
                        metadata_boost_config.as_ref(),
                    )
                })
            })
            .transpose()?;
        let handle = self.collection(collection)?;
        let (mut results, mut result_work) = self.search_in_collection(
            &handle.collection_id,
            text.as_deref(),
            vector.as_ref(),
            &filter,
            requested_weights,
            metadata_boost_enabled,
            metadata_boost_config.as_ref(),
            top_k,
            profile,
            Some(context.clone()),
        )?;
        let returned_rank = results
            .results
            .iter()
            .find(|result| {
                result.chunk_id == expected_chunk_id
                    || result.chunk_external_id == expected_chunk_id
            })
            .map(|result| result.rank);
        results.discard_rows(&mut result_work);
        #[cfg(test)]
        {
            let hook = self.lock()?.before_why_not_ranks.take();
            if let Some(hook) = hook {
                hook();
            }
        }
        let mut guard = self.lock()?;
        let inner = &mut *guard;
        let cache_limit = results
            .query_context
            .transient_rebuild
            .budget_mb
            .saturating_mul(1_048_576);
        if inner.optional_cache_bytes() > cache_limit {
            Inner::release_optional_caches(
                &mut inner.text_index,
                &mut inner.persistent_vector_segments,
                &mut inner.lazy_vector_segment,
                std::iter::empty(),
            );
        }
        let (collection, _collection_work) = inner.read_reclaiming(|inner| {
            copy_collection_in(
                inner.catalog.collection_by_id(&handle.collection_id)?,
                &context,
            )
        })?;
        if let Some(query) = vector.as_ref() {
            query.validate_structure(&collection)?;
            inner.cache_collection_model(&collection.collection_id)?;
            inner
                .catalog
                .visible_model_state(&collection.collection_id)?
                .validate(query.model_name.as_deref())?;
        }
        let (filter_expr, _compiled_work) = inner.read_reclaiming(|_| {
            FilterExpr::parse_validated_in(&filter, &context, crate::json::JsonInputKind::Filter)
        })?;
        let expected =
            resolve_expected_chunk(&inner.catalog, &collection.collection_id, expected_chunk_id);
        let materialized = if expected.collection_matches && !expected.deleted {
            expected
                .chunk
                .as_ref()
                .map(|chunk| inner.materialize_chunk_in(&chunk.chunk_id, &context))
                .transpose()?
        } else {
            None
        };
        let materialized_chunk = materialized
            .as_ref()
            .map(|(chunk, _)| chunk)
            .or(expected.chunk);
        let missing_embedding = materialized_chunk
            .as_ref()
            .map(|chunk| chunk.embedding_id.is_none())
            .unwrap_or(false);
        let embedding_dim =
            expected_embedding_dim_diagnosis(inner, &collection, materialized_chunk, &context)?;
        let (filter_result, filter_work) = why_not_filter_result(
            &inner.catalog,
            materialized_chunk,
            expected.collection_matches,
            expected.deleted,
            &filter_expr,
            filter,
            &context,
        )?;
        let filter_matched = filter_result.matched;
        let (filter_result, mut filter_work) = filter_result.into_json_retained(filter_work)?;
        let mut invalidated_text = false;
        let ranks = if expected.collection_matches && !expected.deleted && filter_matched {
            if let Some(chunk) = materialized_chunk.as_ref() {
                let mut rank = |inner: &Inner| {
                    why_not_candidate_ranks(
                        inner,
                        &collection,
                        &chunk.chunk_id,
                        results.query.as_deref(),
                        vector.as_ref(),
                        &filter_expr,
                        &results.query_context,
                        metadata_boost_config
                            .as_ref()
                            .filter(|_| results.query_context.metadata_boost_enabled),
                        &context,
                        &mut invalidated_text,
                    )
                };
                match rank(inner) {
                    Err(error) if crate::json::is_memory_budget_error(&error) => {
                        if Inner::release_optional_caches(
                            &mut inner.text_index,
                            &mut inner.persistent_vector_segments,
                            &mut inner.lazy_vector_segment,
                            std::iter::empty(),
                        ) {
                            rank(inner)
                        } else {
                            Err(error)
                        }
                    }
                    result => result,
                }
            } else {
                Ok(WhyNotRanks::default())
            }
        } else {
            Ok(WhyNotRanks::default())
        };
        let diagnosis = (|| {
            let ranks = ranks?;
            let builder =
                JsonBuilder::budgeted(&mut filter_work, crate::json::JsonInputKind::StoredJson);
            let limit_analysis = why_not_limit_analysis(
                &ranks,
                results.query_context.candidate_limits,
                results.query_context.weights.effective,
                returned_rank,
                filter_matched && expected.collection_matches && !expected.deleted,
                &builder,
            )?;
            let missed_due_to_weights = why_not_missed_due_to_weights(
                &ranks,
                returned_rank,
                top_k,
                filter_matched,
                &results,
            );
            let diagnosis_codes = why_not_diagnosis_codes(
                &expected,
                missing_embedding,
                filter_matched,
                returned_rank,
                &limit_analysis,
                missed_due_to_weights,
                results.query_context.embedding_present,
                embedding_dim.matches,
                &ranks,
                top_k,
            );
            let diagnosis = WhyNotDiagnosis {
                schema_version: results.schema_version,
                diagnosis_scope: result_work.copy_string("debug_exact", &context)?,
                debug_full_scan: true,
                resource_profile: results.profile,
                result: builder.object([
                    (
                        "expected_chunk",
                        builder.object([
                            (
                                "chunk_id",
                                builder.string(
                                    materialized_chunk
                                        .as_ref()
                                        .map(|chunk| chunk.chunk_id.as_str())
                                        .unwrap_or(expected_chunk_id),
                                )?,
                            ),
                            (
                                "chunk_external_id",
                                builder.optional_str(
                                    materialized_chunk
                                        .as_ref()
                                        .map(|chunk| chunk.chunk_external_id.as_str()),
                                )?,
                            ),
                            ("exists", JsonValue::Bool(expected.exists)),
                            ("deleted", JsonValue::Bool(expected.deleted)),
                            ("missing_embedding", JsonValue::Bool(missing_embedding)),
                            (
                                "collection_matches",
                                JsonValue::Bool(expected.collection_matches),
                            ),
                            (
                                "embedding_dim_matches",
                                JsonValue::Bool(embedding_dim.matches),
                            ),
                            (
                                "embedding_dim",
                                builder.object([
                                    ("expected", JsonValue::Number(embedding_dim.expected as f64)),
                                    (
                                        "actual_dim",
                                        embedding_dim
                                            .actual_dim
                                            .map(|dim| JsonValue::Number(dim as f64))
                                            .unwrap_or(JsonValue::Null),
                                    ),
                                    (
                                        "actual_vector_len",
                                        embedding_dim
                                            .actual_vector_len
                                            .map(|len| JsonValue::Number(len as f64))
                                            .unwrap_or(JsonValue::Null),
                                    ),
                                    ("matches", JsonValue::Bool(embedding_dim.matches)),
                                ])?,
                            ),
                        ])?,
                    ),
                    ("filter_result", filter_result),
                    ("candidate_ranks", ranks.build_json(&builder)?),
                    (
                        "candidate_limits",
                        candidate_limits_json_in(results.query_context.candidate_limits, &builder)?,
                    ),
                    ("limit_analysis", limit_analysis),
                    (
                        "fusion_analysis",
                        builder.object([
                            (
                                "missed_due_to_weights",
                                JsonValue::Bool(missed_due_to_weights),
                            ),
                            (
                                "weights_source",
                                builder.string(&results.query_context.weights.source)?,
                            ),
                            (
                                "requested_weights",
                                weights_json_in(results.query_context.weights.requested, &builder)?,
                            ),
                            (
                                "effective_weights",
                                weights_json_in(results.query_context.weights.effective, &builder)?,
                            ),
                            (
                                "renormalization_reason",
                                builder.optional_str(
                                    results
                                        .query_context
                                        .weights
                                        .renormalization_reason
                                        .as_deref(),
                                )?,
                            ),
                        ])?,
                    ),
                    (
                        "ann_diagnosis",
                        builder.object([
                            ("supported", JsonValue::Bool(false)),
                            (
                                "unsupported_reason",
                                builder.string(
                                    "ann diagnosis is not part of v0.1 exact/BM25 debug scope",
                                )?,
                            ),
                        ])?,
                    ),
                    (
                        "returned_rank",
                        returned_rank
                            .map(|rank| JsonValue::Number(rank as f64))
                            .unwrap_or(JsonValue::Null),
                    ),
                    (
                        "diagnosis_codes",
                        builder
                            .array(diagnosis_codes.into_iter().map(|code| builder.string(code)))?,
                    ),
                ])?,
                query_text: results.query,
                query_context: results.query_context,
            };
            drop(results.collection);
            result_work.retain_workspace(
                diagnosis.schema_version.capacity()
                    + diagnosis.diagnosis_scope.capacity()
                    + diagnosis.query_text.as_ref().map_or(0, String::capacity)
                    + diagnosis.query_context.workspace_bytes(),
            );
            result_work.merge(filter_work);
            Ok((diagnosis, result_work))
        })();
        if invalidated_text {
            inner.mark_index_checkpoint_rebuild_needed();
        }
        diagnosis
    }

    fn lock(&self) -> Result<InnerGuard<'_>> {
        let guard = self
            .inner
            .lock()
            .map_err(|_| Error::new(ErrorCode::LockContention, "database lock poisoned"))?;
        if guard.is_none() {
            return Err(Error::from(std::io::Error::new(
                std::io::ErrorKind::NotConnected,
                "database handle is closed",
            ))
            .with_detail("reason", "database_closed"));
        }
        Ok(InnerGuard(guard))
    }
}

fn normalize_search_text(text: Option<&str>) -> Option<&str> {
    text.map(str::trim).filter(|value| !value.is_empty())
}

fn reserve_query_json_inputs(
    context: &crate::json::JsonOperationContext,
    vector: Option<&QueryEmbeddingInput>,
    filter: &JsonValue,
    boost: Option<&MetadataBoostConfig>,
) -> Result<crate::json::JsonReservation> {
    let mut work = context.reservation();
    if let Some(embedding) = vector {
        work.merge(embedding.reserve_json_work_in(context)?);
    }
    if let Some(boost) = boost {
        work.merge(boost.reserve_json_work_in(context)?);
    }
    work.merge(
        filter
            .guard_in(crate::json::JsonInputKind::Filter, context)
            .map_err(|error| error.with_detail("field", "filter"))?
            .0,
    );
    Ok(work)
}

#[derive(Clone, Debug, Default)]
struct ExpectedChunkLookup<'a> {
    chunk: Option<&'a Chunk>,
    exists: bool,
    deleted: bool,
    collection_matches: bool,
}

#[derive(Clone, Debug, Default)]
struct WhyNotRanks {
    bm25_rank: Option<usize>,
    bm25_retrieval_rank: Option<usize>,
    vector_rank: Option<usize>,
    fusion_rank: Option<usize>,
    fusion_retrieval_rank: Option<usize>,
    candidate_limit_improves: bool,
    candidate_limit_stage: Option<&'static str>,
    default_weight_rank: Option<usize>,
}

impl WhyNotRanks {
    fn build_json(&self, builder: &JsonBuilder<'_>) -> Result<JsonValue> {
        builder.object([
            ("bm25_rank", rank_to_json(self.bm25_rank)),
            ("vector_rank", rank_to_json(self.vector_rank)),
            ("fusion_rank", rank_to_json(self.fusion_rank)),
        ])
    }
}

fn resolve_expected_chunk<'a>(
    catalog: &'a Catalog,
    collection_id: &str,
    expected_chunk_id: &str,
) -> ExpectedChunkLookup<'a> {
    let collection_external = catalog
        .chunk_external_ids
        .get(collection_id, expected_chunk_id)
        .and_then(|chunk_id| catalog.chunks.get(chunk_id));
    let collection_internal = catalog
        .chunks
        .get(expected_chunk_id)
        .filter(|chunk| chunk.collection_id == collection_id);
    let any_internal = catalog.chunks.get(expected_chunk_id);
    let any_external = catalog
        .chunk_external_ids
        .find_external(expected_chunk_id)
        .and_then(|chunk_id| catalog.chunks.get(chunk_id));
    let chunk = collection_external
        .or(collection_internal)
        .or(any_internal)
        .or(any_external)
        .map(Box::as_ref);
    let exists = chunk.is_some();
    let deleted = chunk
        .as_ref()
        .map(|chunk| !catalog.is_chunk_live(&chunk.chunk_id))
        .unwrap_or(false);
    let collection_matches = chunk
        .as_ref()
        .map(|chunk| chunk.collection_id == collection_id)
        .unwrap_or(false);
    ExpectedChunkLookup {
        chunk,
        exists,
        deleted,
        collection_matches,
    }
}

#[derive(Clone, Copy, Debug)]
struct EmbeddingDimDiagnosis {
    expected: usize,
    actual_dim: Option<usize>,
    actual_vector_len: Option<usize>,
    matches: bool,
}

fn expected_embedding_dim_diagnosis(
    inner: &Inner,
    collection: &CollectionInfo,
    chunk: Option<&Chunk>,
    context: &crate::json::JsonOperationContext,
) -> Result<EmbeddingDimDiagnosis> {
    let Some(embedding_id) = chunk.and_then(|chunk| chunk.embedding_id.as_deref()) else {
        return Ok(EmbeddingDimDiagnosis {
            expected: collection.embedding_dim,
            actual_dim: None,
            actual_vector_len: None,
            matches: true,
        });
    };
    let summary = inner.catalog.embeddings.get(embedding_id).ok_or_else(|| {
        Error::new(
            ErrorCode::EmbeddingMissing,
            format!("embedding {embedding_id} not found"),
        )
    })?;
    let retained = inner
        .catalog
        .record_ref_for_embedding(embedding_id)
        .map(|record_ref| {
            inner
                .storage
                .payload_reader()?
                .read_embedding_in(record_ref, context)
        })
        .transpose()?;
    let embedding = retained
        .as_ref()
        .map(|value| value.embedding())
        .unwrap_or(summary);
    let actual_vector_len = embedding.vector.len();
    Ok(EmbeddingDimDiagnosis {
        expected: collection.embedding_dim,
        actual_dim: Some(embedding.dim),
        actual_vector_len: Some(actual_vector_len),
        matches: embedding.dim == collection.embedding_dim
            && actual_vector_len == collection.embedding_dim,
    })
}

fn why_not_filter_result(
    catalog: &Catalog,
    chunk: Option<&Chunk>,
    collection_matches: bool,
    deleted: bool,
    filter_expr: &Option<FilterExpr>,
    filter_json: JsonValue,
    context: &crate::json::JsonOperationContext,
) -> Result<(FilterExplain, crate::json::JsonReservation)> {
    let Some(expr) = filter_expr else {
        return Ok((FilterExplain::matched(filter_json), context.reservation()));
    };
    let reason = match chunk {
        None => "expected chunk does not exist",
        Some(_) if !collection_matches => {
            "expected chunk collection does not match query collection"
        }
        Some(_) if deleted => "expected chunk is deleted",
        Some(chunk) => return expr.evaluate_for_chunk_in(catalog, chunk, filter_json, context),
    };
    let mut work = context.reservation();
    let mut reasons = Vec::new();
    work.reserve_vec(&mut reasons, 1, crate::json::JsonInputKind::Filter)?;
    reasons.push(work.copy_string(reason, crate::json::JsonInputKind::Filter)?);
    Ok((
        FilterExplain {
            matched: false,
            expression: filter_json,
            reasons,
        },
        work,
    ))
}

#[allow(clippy::too_many_arguments)]
fn why_not_candidate_ranks(
    inner: &Inner,
    collection: &CollectionInfo,
    expected_chunk_id: &str,
    text: Option<&str>,
    query_embedding: Option<&QueryEmbeddingInput>,
    filter_expr: &Option<FilterExpr>,
    query_context: &SearchQueryContext,
    metadata_boost_config: Option<&MetadataBoostConfig>,
    context: &crate::json::JsonOperationContext,
    invalidated_text: &mut bool,
) -> Result<WhyNotRanks> {
    let weights = &query_context.weights;
    let limits = query_context.candidate_limits;
    let live_chunk_count = inner
        .catalog
        .live_chunks()
        .filter(|chunk| chunk.collection_id == collection.collection_id)
        .count()
        .max(1);
    let text_options = TextSearchOptions::scores_only(live_chunk_count);
    if inner.index_suffix_status != IndexSuffixState::PersistentCheckpoint {
        // Debug retains full ranks, so a bounded normal query does not authorize its larger workspace.
        ensure_transient_rebuild_budget(
            transient_query_scratch_bytes(
                &inner.catalog,
                &collection.collection_id,
                text,
                text_options,
                query_embedding.is_some(),
                live_chunk_count,
            ),
            query_context.transient_rebuild.budget_mb,
            query_context.profile,
        )?;
    }
    let mut bm25_hits = text
        .map(|query| -> Result<Bm25Hits> {
            if inner.index_suffix_status != IndexSuffixState::PersistentCheckpoint {
                let mut reader = inner.storage.payload_reader()?;
                let mut bytes_read = 0;
                TextIndex::search_from_snapshot(
                    &inner.catalog.visible_snapshot(),
                    &collection.collection_id,
                    query,
                    text_options,
                    context,
                    |visible| {
                        read_visible_chunk_text(&mut reader, visible, context, &mut bytes_read)
                    },
                )
            } else if !inner.text_index.has_resident_cache() {
                let source = inner
                    .lazy_vector_source
                    .as_ref()
                    .expect("persistent text source");
                if let Some(hits) = TextIndex::search_from_segment(
                    source,
                    &inner.storage,
                    &collection.collection_id,
                    query,
                    text_options,
                    context,
                )? {
                    return Ok(hits);
                }
                *invalidated_text = true;
                ensure_transient_rebuild_budget(
                    transient_query_scratch_bytes(
                        &inner.catalog,
                        &collection.collection_id,
                        Some(query),
                        text_options,
                        false,
                        0,
                    ),
                    query_context.transient_rebuild.budget_mb,
                    query_context.profile,
                )?;
                let mut reader = inner.storage.payload_reader()?;
                let mut bytes_read = 0;
                TextIndex::search_from_snapshot(
                    &inner.catalog.visible_snapshot(),
                    &collection.collection_id,
                    query,
                    text_options,
                    context,
                    |visible| {
                        read_visible_chunk_text(&mut reader, visible, context, &mut bytes_read)
                    },
                )
            } else {
                inner
                    .text_index
                    .search(&collection.collection_id, query, text_options, context)
            }
        })
        .transpose()?
        .unwrap_or_else(|| Bm25Hits::empty(context));
    let mut vector_hits = if let Some(query_embedding) = query_embedding {
        let mut payload_reader = inner.storage.payload_reader()?;
        let mut bytes_read = 0;
        ExactVectorIndex
            .search_with_vectors(
                &inner.catalog.visible_snapshot(),
                &collection.collection_id,
                &query_embedding.vector,
                VectorSearchOptions {
                    metric: collection.distance_metric,
                    vectors_normalized: collection.vectors_normalized,
                    limit: inner
                        .catalog
                        .live_embeddings()
                        .filter(|embedding| embedding.collection_id == collection.collection_id)
                        .count()
                        .max(1),
                },
                |id| {
                    matches_vector_filter(
                        &inner.catalog,
                        &collection.collection_id,
                        filter_expr.as_ref(),
                        id,
                    )
                },
                context,
                |visible| {
                    read_visible_embedding_vector(
                        &mut payload_reader,
                        visible,
                        context,
                        &mut bytes_read,
                    )
                },
            )?
            .0
    } else {
        VectorHits::empty(context)
    };

    // BM25 truncates before metadata filtering; its limit must use the earlier rank.
    let bm25_retrieval_rank = rank_bm25(&bm25_hits, expected_chunk_id);
    let mut limited_bm25 = bm25_hits.copy_prefix(limits.bm25)?;
    let mut limited_vector = vector_hits.copy_prefix(limits.vector)?;
    filter_candidates(
        &inner.catalog,
        &mut limited_bm25,
        &mut limited_vector,
        filter_expr.as_ref(),
    );
    filter_candidates(
        &inner.catalog,
        &mut bm25_hits,
        &mut vector_hits,
        filter_expr.as_ref(),
    );
    let bm25_rank = rank_bm25(&bm25_hits, expected_chunk_id);
    let vector_rank = rank_vector(&vector_hits, expected_chunk_id);
    bm25_hits.sort_by_id();
    vector_hits.sort_by_id();
    let actual = fuse_completed_candidates(
        &inner.catalog,
        &limited_bm25,
        &limited_vector,
        &bm25_hits,
        &vector_hits,
        text.is_some(),
        metadata_boost_config,
        weights.effective,
        live_chunk_count,
        context,
    )?;
    let fusion_rank = rank_fused(actual.hits.as_slice(), expected_chunk_id);
    let default_weight_rank = if weights.source == "user_override" {
        let default = crate::query::default_weights_for(
            text.is_some(),
            query_embedding.is_some(),
            metadata_boost_config.is_some_and(|boost| boost.rule_count() > 0),
        );
        actual
            .hits
            .as_slice()
            .iter()
            .find(|score| score.chunk_id == expected_chunk_id)
            .map(|target| {
                let score = |row: &CandidateScore<'_>| {
                    default.bm25 * row.bm25_norm
                        + default.vector * row.vector_norm.unwrap_or(0.0)
                        + default.metadata * row.metadata_boost
                };
                let target_score = score(target);
                1 + actual
                    .hits
                    .as_slice()
                    .iter()
                    .filter(|row| {
                        score(row) > target_score
                            || (score(row) == target_score && row.chunk_id < target.chunk_id)
                    })
                    .count()
            })
    } else {
        None
    };
    // A window cause is established only by an executed one-window counterfactual.
    // Neither its U nor its M replaces the actual ordinary-query ranks above.
    let mut candidate_limit_stage = None;
    if fusion_rank.is_none() {
        for (eligible, stage) in [
            (
                bm25_retrieval_rank.is_some_and(|rank| rank > limits.bm25),
                "bm25",
            ),
            (
                vector_rank.is_some_and(|rank| rank > limits.vector),
                "vector",
            ),
        ] {
            if !eligible {
                continue;
            }
            let expanded = fuse_completed_candidates(
                &inner.catalog,
                if stage == "bm25" {
                    &bm25_hits
                } else {
                    &limited_bm25
                },
                if stage == "vector" {
                    &vector_hits
                } else {
                    &limited_vector
                },
                &bm25_hits,
                &vector_hits,
                text.is_some(),
                metadata_boost_config,
                weights.effective,
                live_chunk_count,
                context,
            )?;
            if rank_fused(expanded.hits.as_slice(), expected_chunk_id)
                .is_some_and(|rank| rank <= query_context.top_k)
            {
                candidate_limit_stage = Some(stage);
                break;
            }
        }
    }
    let candidate_limit_improves = candidate_limit_stage.is_some();
    Ok(WhyNotRanks {
        bm25_rank,
        bm25_retrieval_rank,
        vector_rank,
        fusion_rank,
        fusion_retrieval_rank: fusion_rank,
        candidate_limit_improves,
        candidate_limit_stage,
        default_weight_rank,
    })
}

fn why_not_limit_analysis(
    ranks: &WhyNotRanks,
    limits: CandidateLimits,
    _weights: SearchWeights,
    returned_rank: Option<usize>,
    can_be_limited: bool,
    builder: &JsonBuilder<'_>,
) -> Result<JsonValue> {
    let limiting = if returned_rank.is_none() && can_be_limited {
        if let Some(rank) = ranks.fusion_retrieval_rank {
            (rank > limits.fusion).then_some((
                Some(rank),
                limits.fusion,
                "fusion",
                "fusion_candidates",
            ))
        } else {
            [
                (
                    ranks.bm25_retrieval_rank,
                    limits.bm25,
                    "bm25",
                    "bm25_candidates",
                ),
                (
                    ranks.vector_rank,
                    limits.vector,
                    "vector",
                    "vector_candidates",
                ),
            ]
            .into_iter()
            .find(|(rank, limit, stage, _)| {
                rank.is_some_and(|rank| rank > *limit)
                    && ranks
                        .candidate_limit_stage
                        .map_or(true, |proved| proved == *stage)
            })
        }
    } else {
        None
    };
    builder.object([
        (
            "missed_due_to_candidate_limit",
            JsonValue::Bool(limiting.is_some() && ranks.candidate_limit_improves),
        ),
        (
            "limiting_stage",
            builder.optional_str(limiting.as_ref().map(|(_, _, stage, _)| *stage))?,
        ),
        (
            "limiting_parameter",
            builder.optional_str(limiting.as_ref().map(|(_, _, _, parameter)| *parameter))?,
        ),
        (
            "effective_candidate_limit",
            limiting
                .map(|(_, limit, _, _)| JsonValue::Number(limit as f64))
                .unwrap_or(JsonValue::Null),
        ),
    ])
}

fn why_not_missed_due_to_weights(
    ranks: &WhyNotRanks,
    returned_rank: Option<usize>,
    top_k: usize,
    filter_matched: bool,
    results: &SearchResults,
) -> bool {
    returned_rank.is_none()
        && filter_matched
        && results.query_context.weights.source == "user_override"
        && ranks.fusion_rank.map(|rank| rank > top_k).unwrap_or(false)
        && ranks.default_weight_rank.is_some_and(|rank| rank <= top_k)
}

#[allow(clippy::too_many_arguments)]
fn why_not_diagnosis_codes(
    expected: &ExpectedChunkLookup<'_>,
    missing_embedding: bool,
    filter_matched: bool,
    returned_rank: Option<usize>,
    limit_analysis: &JsonValue,
    missed_due_to_weights: bool,
    embedding_present: bool,
    embedding_dim_matches: bool,
    ranks: &WhyNotRanks,
    top_k: usize,
) -> Vec<&'static str> {
    let mut codes = Vec::new();
    if !expected.exists {
        codes.push("expected_chunk_not_found");
    } else if !expected.collection_matches {
        codes.push("collection_mismatch");
    } else if expected.deleted {
        codes.push("deleted");
    }
    if missing_embedding && embedding_present {
        codes.push("missing_embedding");
    }
    if !embedding_dim_matches && embedding_present {
        codes.push("embedding_dim_mismatch");
    }
    if expected.exists && expected.collection_matches && !expected.deleted && !filter_matched {
        codes.push("filtered_by_metadata");
    }
    if returned_rank.is_some() {
        codes.push("returned");
    } else if limit_analysis
        .get("missed_due_to_candidate_limit")
        .and_then(JsonValue::as_bool)
        == Some(true)
    {
        codes.push("missed_due_to_candidate_limit");
    } else if missed_due_to_weights {
        codes.push("missed_due_to_weights");
    } else if ranks.fusion_rank.map(|rank| rank > top_k).unwrap_or(false) {
        codes.push("rank_below_top_k");
    } else if codes.is_empty() {
        codes.push("not_returned");
    }
    codes
}

fn rank_bm25(hits: &[Bm25Hit], chunk_id: &str) -> Option<usize> {
    hits.iter()
        .position(|hit| hit.chunk_id == chunk_id)
        .map(|index| index + 1)
}

fn rank_vector(hits: &[VectorHit], chunk_id: &str) -> Option<usize> {
    hits.iter()
        .position(|hit| hit.chunk_id == chunk_id)
        .map(|index| index + 1)
}

fn rank_fused(scores: &[CandidateScore<'_>], chunk_id: &str) -> Option<usize> {
    scores
        .iter()
        .position(|score| score.chunk_id == chunk_id)
        .map(|index| index + 1)
}

fn rank_to_json(rank: Option<usize>) -> JsonValue {
    rank.map(|rank| JsonValue::Number(rank as f64))
        .unwrap_or(JsonValue::Null)
}

fn transient_rebuild_exceeds_budget(bytes_required: u64, budget_mb: usize) -> bool {
    let budget_bytes = (budget_mb as u64).saturating_mul(1_048_576);
    bytes_required > budget_bytes
}

fn ensure_transient_rebuild_budget(
    bytes_required: u64,
    budget_mb: usize,
    profile: ResourceProfile,
) -> Result<()> {
    if transient_rebuild_exceeds_budget(bytes_required, budget_mb) {
        Err(Error::new(
            ErrorCode::IndexCorruption,
            "transient index rebuild exceeds cache budget",
        )
        .with_detail(
            "index_unavailable_reason",
            "transient_rebuild_budget_exceeded",
        )
        .with_detail("index_source_attempted", "transient_rebuild")
        .with_detail("resource_profile", profile.as_json_str())
        .with_detail("bytes_required", bytes_required)
        .with_detail("budget_mb", budget_mb))
    } else {
        Ok(())
    }
}

fn transient_query_scratch_bytes(
    catalog: &Catalog,
    collection_id: &str,
    text: Option<&str>,
    text_options: TextSearchOptions,
    vector: bool,
    vector_limit: usize,
) -> u64 {
    #[cfg(test)]
    tests::record_transient_estimate();
    let mut bytes = 0u64;
    let mut max_text_payload = 0u64;
    let mut chunks = 0usize;
    if text.is_some() {
        for chunk in catalog
            .live_chunks()
            .filter(|chunk| chunk.collection_id == collection_id)
        {
            chunks = chunks.saturating_add(1);
            max_text_payload = max_text_payload.max(
                catalog
                    .record_ref_for_chunk(&chunk.chunk_id)
                    .map_or(chunk.text.len() as u64, |record| record.payload.len),
            );
        }
    }
    if let Some(query) = text {
        // Covers normalized query terms, counters, and bounded candidate term copies.
        // The 256-byte factor covers token splitting/lowercase expansion and per-term overhead.
        let query_workspace = (query.len() as u64).saturating_mul(256);
        let candidate_terms = if text_options.explanations {
            query_workspace
        } else {
            0
        };
        bytes = bytes.saturating_add(query_workspace).saturating_add(
            (chunks.min(text_options.limit) as u64)
                .saturating_mul(candidate_terms.saturating_add(512)),
        );
    }
    bytes = bytes.saturating_add(max_text_payload.saturating_mul(64));
    if vector {
        let mut count = 0usize;
        let mut max_vector_payload = 0u64;
        for embedding in catalog
            .live_embeddings()
            .filter(|embedding| embedding.collection_id == collection_id)
        {
            count = count.saturating_add(1);
            max_vector_payload = max_vector_payload.max(
                catalog
                    .record_ref_for_embedding(&embedding.embedding_id)
                    .map_or(
                        (embedding.vector.len() as u64).saturating_mul(4),
                        |record| record.payload.len,
                    ),
            );
        }
        // Only one decoded vector and bounded top-k candidates are resident during exact scan.
        bytes = bytes
            .saturating_add(max_vector_payload.saturating_mul(3))
            .saturating_add((count.min(vector_limit) as u64).saturating_mul(512));
    }
    bytes
}

#[allow(clippy::too_many_arguments)]
fn build_result(
    inner: &Inner,
    collection: &CollectionInfo,
    score: CandidateScore<'_>,
    matched_terms: &[MatchedTerm],
    filter_expr: Option<&FilterExpr>,
    filter_json: &JsonValue,
    query_context: &SearchQueryContext,
    profile: ResourceProfile,
    limits: CandidateLimits,
    execution_stats: SearchExecutionStats,
    normalization_candidate_count: usize,
    bm25_max: Option<f64>,
    context: &crate::json::JsonOperationContext,
    result_work: &mut RecordReservation,
) -> Result<Option<SearchResult>> {
    let Some(chunk_summary) = inner.catalog.chunks.get(score.chunk_id) else {
        return Ok(None);
    };
    if !inner.catalog.is_chunk_live(&chunk_summary.chunk_id) {
        return Ok(None);
    }
    let (payload, mut chunk_work) = inner.materialize_chunk_in(score.chunk_id, context)?;
    let mut chunk = payload;
    let document = inner
        .catalog
        .document_by_id_ref(&chunk.collection_id, &chunk.document_id)?;
    let (doc_metadata, work) = document
        .metadata
        .try_clone_in(crate::json::JsonInputKind::StoredJson, context)?
        .into_parts();
    result_work.merge(work);
    let (filter_copy, work) = filter_json
        .try_clone_in(crate::json::JsonInputKind::Filter, context)?
        .into_parts();
    result_work.merge(work);
    let (filter, trace_work) = match filter_expr {
        Some(expr) => expr.evaluate_for_chunk_in(&inner.catalog, &chunk, filter_copy, context)?,
        None => (FilterExplain::matched(filter_copy), context.reservation()),
    };
    let (filter, trace_work) = filter.into_json_retained(trace_work)?;
    result_work.merge(trace_work);
    let explain = build_explain_in(
        ExplainInput {
            collection_name: &collection.name,
            chunk: &chunk,
            document_external_id: &document.external_id,
            document_metadata: &doc_metadata,
            document_source_uri: document.source_uri.as_deref(),
            bm25_recalled: score.bm25_recalled,
            vector_recalled: score.vector_recalled,
            normalization_candidate_count,
            bm25_raw: score.bm25_raw,
            bm25_norm: score.bm25_norm,
            bm25_max,
            vector_raw: score.vector_raw,
            vector_norm: score.vector_norm,
            metadata_boost: score.metadata_boost,
            final_score: score.final_score,
            matched_terms,
            weights: &query_context.weights,
            resource_profile: profile,
            candidate_limits: limits,
            vector_scan_count: execution_stats.vector_scan_count,
            distance_ops: execution_stats.distance_ops,
            fusion_candidate_count: execution_stats.fusion_candidate_count,
            index_source: &query_context.index_source,
            index_rebuild_reason: query_context.index_rebuild_reason.as_deref(),
            transient_rebuild: query_context.transient_rebuild.clone(),
            distance_metric: collection.distance_metric.as_str(),
            vectors_normalized: collection.vectors_normalized,
            filter,
            metadata_boost_present: query_context.metadata_boost_present,
            metadata_boost_enabled: query_context.metadata_boost_enabled,
            boost_config_hash: query_context.boost_config_hash.as_deref(),
            boost_rule_count: query_context.boost_rule_count,
            matched_boost_rule_count: score.matched_boost_rule_count,
        },
        &mut result_work.json,
    )?;
    let end = chunk
        .text
        .char_indices()
        .nth(240)
        .map_or(chunk.text.len(), |(offset, _)| offset);
    let [document_id, document_external_id, snippet] = chunk_work.copy_strings(
        [
            &document.document_id,
            &document.external_id,
            &chunk.text[..end],
        ],
        context,
    )?;
    let result = SearchResult {
        rank: 0,
        chunk_id: std::mem::take(&mut chunk.chunk_id),
        chunk_external_id: std::mem::take(&mut chunk.chunk_external_id),
        document_id,
        document_external_id,
        snippet,
        document_metadata: doc_metadata,
        chunk_metadata: std::mem::replace(&mut chunk.metadata, JsonValue::object()),
        final_score: score.final_score,
        bm25_score: score.bm25_raw,
        vector_score: score.vector_raw,
        metadata_boost: score.metadata_boost,
        normalized_scores: NormalizedScores {
            bm25_norm: score.bm25_norm,
            vector_norm: score.vector_norm,
            metadata_boost: score.metadata_boost,
        },
        explain,
    };
    drop(chunk);
    chunk_work.retain_workspace(
        result.chunk_id.capacity()
            + result.chunk_external_id.capacity()
            + result.document_id.capacity()
            + result.document_external_id.capacity()
            + result.snippet.capacity(),
    );
    result_work.merge(chunk_work);
    Ok(Some(result))
}

#[cfg(test)]
fn plan_import_batches(
    records: &[RecordPayload],
    byte_limit: u64,
    max_records: usize,
) -> Result<Vec<usize>> {
    plan_record_batches(
        records.iter().map(|record| {
            crate::storage::encoded_record_len(record)
                .map(|len| (crate::storage::BatchRecordKind::of(record), len))
        }),
        byte_limit,
        max_records,
    )?
    .collect()
}

fn plan_record_batches(
    records: impl Iterator<Item = Result<(crate::storage::BatchRecordKind, u64)>>,
    byte_target: u64,
    max_records: usize,
) -> Result<impl Iterator<Item = Result<usize>>> {
    let commit_bytes = crate::storage::COMMIT_RECORD_BYTES;
    let mut count = 0;
    let mut bytes = u128::from(commit_bytes);
    let mut records = records.peekable();
    let mut finished = false;
    Ok(std::iter::from_fn(move || {
        if finished {
            return None;
        }
        while let Some(record) = records.next() {
            let group = (|| {
                let (record, mut group_bytes) = record?;
                // Replacement tombstone and its new value must never straddle commits.
                let replacement = records
                    .peek()
                    .and_then(|next| next.as_ref().ok())
                    .is_some_and(|(next, _)| record.pairs_with(*next));
                let group_count = if replacement { 2 } else { 1 };
                if replacement {
                    group_bytes = group_bytes
                        .checked_add(records.next().unwrap()?.1)
                        .ok_or_else(|| {
                            Error::new(ErrorCode::FileFormatError, "import record length overflow")
                        })?;
                }
                if group_count > max_records {
                    return Err(Error::new(
                        ErrorCode::FileFormatError,
                        "import batch record limit cannot fit an atomic group",
                    )
                    .with_detail("limit_records", max_records));
                }
                Ok((group_count, group_bytes))
            })();
            let (group_count, group_bytes) = match group {
                Ok(group) => group,
                Err(error) => {
                    finished = true;
                    return Some(Err(error));
                }
            };
            // The byte target is a preference, not an independent memory quota.
            // Keep an oversized atomic group intact; preparation checks its real ledger.
            if count > 0
                && (group_count > max_records - count
                    || bytes + u128::from(group_bytes) > u128::from(byte_target))
            {
                let previous = count;
                count = group_count;
                bytes = u128::from(commit_bytes) + u128::from(group_bytes);
                return Some(Ok(previous));
            }
            count += group_count;
            bytes += u128::from(group_bytes);
        }
        finished = true;
        (count > 0).then_some(Ok(count))
    }))
}

fn batch_byte_target(cache_budget_mb: usize) -> u64 {
    (cache_budget_mb as u64)
        .saturating_mul(1_048_576)
        .saturating_div(8)
        .min(64 * 1_048_576)
}

impl Inner {
    fn prepare_write_batch<'a>(
        &'a mut self,
        context: &crate::json::JsonOperationContext,
    ) -> (
        &'a Catalog,
        crate::storage::ReclaimingRecordBatch<impl FnMut() -> bool + 'a>,
    ) {
        let Self {
            catalog,
            text_index,
            persistent_vector_segments,
            lazy_vector_segment,
            ..
        } = self;
        (
            catalog,
            OwnedRecordBatch::new(context).reclaiming(move || {
                Self::release_optional_caches(
                    text_index,
                    persistent_vector_segments,
                    lazy_vector_segment,
                    std::iter::empty(),
                )
            }),
        )
    }

    // The attempt owns and drops its partial output before retry. Keep Inner
    // locked across both reads so reclamation cannot change the data snapshot.
    fn read_reclaiming<T>(&mut self, mut read: impl FnMut(&Self) -> Result<T>) -> Result<T> {
        match read(self) {
            Err(error) if crate::json::is_memory_budget_error(&error) => {
                if Self::release_optional_caches(
                    &mut self.text_index,
                    &mut self.persistent_vector_segments,
                    &mut self.lazy_vector_segment,
                    std::iter::empty(),
                ) {
                    read(self)
                } else {
                    Err(error)
                }
            }
            result => result,
        }
    }

    #[cfg(test)]
    fn run_query_output_hook(&mut self, stage: QueryOutputStage) {
        if self
            .query_output_hook
            .as_ref()
            .is_some_and(|(target, _)| *target == stage)
        {
            let (_, hook) = self.query_output_hook.take().unwrap();
            hook();
        }
    }

    fn append_import_batches(
        &mut self,
        mut prepared: crate::storage::PreparedRecords,
        mode: ImportIndexBuildMode,
        context: &crate::json::JsonOperationContext,
    ) -> Result<ImportBatchMetrics> {
        let byte_limit = batch_byte_target(self.active_cache_budget_mb());
        let max_records = self.config.resource_profile.params().import_batch_chunks;
        // Validate every atomic group before writing, without retaining a manifest-wide plan.
        plan_record_batches(prepared.records()?, byte_limit, max_records)?
            .try_for_each(|count| count.map(|_| ()))?;
        let mut pending: Option<PendingIndexUpdate> = None;
        let mut metrics = ImportBatchMetrics::default();
        loop {
            let count = plan_record_batches(prepared.records()?, byte_limit, max_records)?
                .next()
                .transpose()?;
            let Some(count) = count else {
                break;
            };
            let mut remaining = count;
            while remaining > 0 {
                #[cfg(test)]
                if let Some(remaining) = self.import_fail_after_batches.as_mut() {
                    if *remaining == 0 {
                        self.import_fail_after_batches = None;
                        return Err(Error::new(
                            ErrorCode::IoError,
                            "injected import batch failure",
                        ));
                    }
                    *remaining -= 1;
                }
                let batch = prepared.take_batch_reclaiming(remaining, || {
                    Self::release_optional_caches(
                        &mut self.text_index,
                        &mut self.persistent_vector_segments,
                        &mut self.lazy_vector_segment,
                        pending
                            .as_mut()
                            .map(|pending| &mut pending.reusable_vectors),
                    )
                })?;
                remaining -= batch.len();
                let chunks = batch
                    .records()
                    .filter(|record| matches!(record, RecordPayload::ChunkPut(_)))
                    .count();
                let started = Instant::now();
                let next = self.commit_prepared_batch_tracking(batch, pending.as_mut())?;
                metrics.commit_time += started.elapsed();
                metrics.committed_batches += 1;
                metrics.max_batch_chunks = metrics.max_batch_chunks.max(chunks);
                match mode {
                    ImportIndexBuildMode::InlineFullCheckpoint => {
                        let started = Instant::now();
                        self.finish_index_update(next, context)?;
                        metrics.checkpoint_time += started.elapsed();
                        metrics.checkpoint_cycles += 1;
                        metrics.checkpoint_ready_cycles += usize::from(
                            self.index_suffix_status == IndexSuffixState::PersistentCheckpoint,
                        );
                    }
                    ImportIndexBuildMode::DeferredFullCheckpoint => {
                        if pending.is_none() {
                            pending = Some(next);
                        }
                    }
                }
            }
        }
        if let Some(pending) = pending {
            let started = Instant::now();
            self.finish_index_update(pending, context)?;
            metrics.checkpoint_time += started.elapsed();
            metrics.checkpoint_cycles += 1;
            metrics.checkpoint_ready_cycles +=
                usize::from(self.index_suffix_status == IndexSuffixState::PersistentCheckpoint);
        }
        metrics.final_checkpoint_ready =
            self.index_suffix_status == IndexSuffixState::PersistentCheckpoint;
        Ok(metrics)
    }

    fn append_and_apply(
        &mut self,
        records: OwnedRecordBatch,
        context: &crate::json::JsonOperationContext,
    ) -> Result<()> {
        let pending = self.commit_records_in(records)?;
        self.finish_index_update(pending, context)
    }

    #[cfg(test)]
    fn commit_records(&mut self, records: Vec<RecordPayload>) -> Result<PendingIndexUpdate> {
        let context = crate::json::JsonOperationContext::standalone();
        let mut batch = OwnedRecordBatch::new(&context);
        for record in records {
            batch.push(record)?;
        }
        self.commit_records_in(batch)
    }

    fn commit_records_in(&mut self, records: OwnedRecordBatch) -> Result<PendingIndexUpdate> {
        self.storage.require_writer()?;
        let reusable_checkpoint =
            self.index_suffix_status == IndexSuffixState::PersistentCheckpoint;
        let context = records.context().clone();
        crate::json::retry_after_memory_reclaim(
            || self.catalog.attach_context(&context),
            || {
                Self::release_optional_caches(
                    &mut self.text_index,
                    &mut self.persistent_vector_segments,
                    &mut self.lazy_vector_segment,
                    [],
                )
            },
        )?;
        let mut records = records.prepare_encoding(|| {
            Self::release_optional_caches(
                &mut self.text_index,
                &mut self.persistent_vector_segments,
                &mut self.lazy_vector_segment,
                [],
            )
        })?;
        let changes = crate::json::retry_after_memory_reclaim(
            || IndexChanges::from_records(&self.catalog, records.records(), &context),
            || {
                Self::release_optional_caches(
                    &mut self.text_index,
                    &mut self.persistent_vector_segments,
                    &mut self.lazy_vector_segment,
                    [],
                )
            },
        )?;
        let mut prefetched = self.prefetch_unchanged_vector_segments(&changes.vector);
        let prefetched_text = match self.prefetch_unchanged_text(&changes.text, &context) {
            Err(error) if crate::json::is_memory_budget_error(&error) => {
                if Self::release_optional_caches(
                    &mut self.text_index,
                    &mut self.persistent_vector_segments,
                    &mut self.lazy_vector_segment,
                    [&mut prefetched],
                ) {
                    self.prefetch_unchanged_text(&changes.text, &context)?
                } else {
                    return Err(error);
                }
            }
            result => result?,
        };
        let mut transaction = self.catalog.transaction(&context);
        let (payloads, json) = records.records_and_json();
        if let Err(error) = transaction.prepare_reclaiming(payloads.iter(), json, || {
            Self::release_optional_caches(
                &mut self.text_index,
                &mut self.persistent_vector_segments,
                &mut self.lazy_vector_segment,
                [&mut prefetched],
            )
        }) {
            // Metadata credit already belongs to the transaction, so its input
            // allocations must be gone before rollback returns that credit.
            drop(records);
            drop(transaction);
            return Err(error);
        }
        let (mut append, _batch_work) =
            self.storage
                .append_owned_batch_in(records, self.config.durability, || {
                    Self::release_optional_caches(
                        &mut self.text_index,
                        &mut self.persistent_vector_segments,
                        &mut self.lazy_vector_segment,
                        [&mut prefetched],
                    )
                })?;
        transaction.finish_references(&mut append.records);
        transaction.commit();
        self.apply_committed_batch(
            append,
            changes,
            reusable_checkpoint,
            prefetched,
            prefetched_text,
        )
    }

    #[cfg(test)]
    fn commit_prepared_batch(
        &mut self,
        batch: crate::storage::PreparedBatch<'_>,
    ) -> Result<PendingIndexUpdate> {
        self.commit_prepared_batch_tracking(batch, None)
    }

    fn commit_prepared_batch_tracking(
        &mut self,
        mut batch: crate::storage::PreparedBatch<'_>,
        mut pending: Option<&mut PendingIndexUpdate>,
    ) -> Result<PendingIndexUpdate> {
        self.storage.require_writer()?;
        let reusable_checkpoint =
            self.index_suffix_status == IndexSuffixState::PersistentCheckpoint;
        let context = batch.context().clone();
        let changes = crate::json::retry_after_memory_reclaim(
            || {
                self.catalog.attach_context(&context)?;
                IndexChanges::from_records(&self.catalog, batch.records(), &context)
            },
            || {
                Self::release_optional_caches(
                    &mut self.text_index,
                    &mut self.persistent_vector_segments,
                    &mut self.lazy_vector_segment,
                    pending
                        .as_mut()
                        .map(|pending| &mut pending.reusable_vectors),
                )
            },
        )?;
        // Deferred imports keep the first reusable checkpoint. Admit their union
        // before this batch writes, while earlier valid commits remain visible.
        if let Some(pending) = pending.as_mut() {
            crate::json::retry_after_memory_reclaim(
                || pending.changes.extend(&changes),
                || {
                    Self::release_optional_caches(
                        &mut self.text_index,
                        &mut self.persistent_vector_segments,
                        &mut self.lazy_vector_segment,
                        [&mut pending.reusable_vectors],
                    )
                },
            )?;
        }
        let mut prefetched = self.prefetch_unchanged_vector_segments(&changes.vector);
        let prefetched_text = match self.prefetch_unchanged_text(&changes.text, &context) {
            Err(error) if crate::json::is_memory_budget_error(&error) => {
                if Self::release_optional_caches(
                    &mut self.text_index,
                    &mut self.persistent_vector_segments,
                    &mut self.lazy_vector_segment,
                    pending
                        .as_mut()
                        .map(|pending| &mut pending.reusable_vectors)
                        .into_iter()
                        .chain(std::iter::once(&mut prefetched)),
                ) {
                    self.prefetch_unchanged_text(&changes.text, &context)?
                } else {
                    return Err(error);
                }
            }
            result => result?,
        };
        let mut transaction = self.catalog.transaction(&context);
        let (payloads, json) = batch.records_and_json();
        if let Err(error) = transaction.prepare_reclaiming(payloads, json, || {
            Self::release_optional_caches(
                &mut self.text_index,
                &mut self.persistent_vector_segments,
                &mut self.lazy_vector_segment,
                pending
                    .as_mut()
                    .map(|pending| &mut pending.reusable_vectors)
                    .into_iter()
                    .chain(std::iter::once(&mut prefetched)),
            )
        }) {
            drop(batch);
            drop(transaction);
            return Err(error);
        }
        let (mut append, _batch_work) = self
            .storage
            .append_prepared_batch(batch, self.config.durability)?;
        transaction.finish_references(&mut append.records);
        transaction.commit();
        self.apply_committed_batch(
            append,
            changes,
            reusable_checkpoint,
            prefetched,
            prefetched_text,
        )
    }

    fn release_optional_caches<'a>(
        text: &mut TextIndex,
        persistent: &mut Option<VectorCache>,
        lazy: &mut Option<CachedVectorSegment>,
        additional: impl IntoIterator<Item = &'a mut VectorCache>,
    ) -> bool {
        let mut bytes = text.reserved_bytes()
            + persistent.as_ref().map_or(0, VectorCache::reserved_bytes)
            + lazy.as_ref().map_or(0, CachedVectorSegment::reserved_bytes);
        text.clear_cache();
        *persistent = None;
        *lazy = None;
        for cache in additional {
            bytes += cache.reserved_bytes();
            cache.clear();
        }
        bytes != 0
    }

    fn optional_cache_bytes(&self) -> usize {
        self.text_index.reserved_bytes()
            + self
                .persistent_vector_segments
                .as_ref()
                .map_or(0, VectorCache::reserved_bytes)
            + self
                .lazy_vector_segment
                .as_ref()
                .map_or(0, CachedVectorSegment::reserved_bytes)
    }

    fn prefetch_unchanged_vector_segments(
        &self,
        changed_collections: &BTreeSet<String>,
    ) -> VectorCache {
        let context = crate::json::JsonOperationContext::new(Arc::clone(&self.work));
        let mut prefetched = VectorCache::new(&context);
        if self.index_suffix_status != IndexSuffixState::PersistentCheckpoint {
            return prefetched;
        }
        let Some(source) = &self.lazy_vector_source else {
            return prefetched;
        };
        let budget = (self.active_cache_budget_mb() as u64).saturating_mul(1_048_576) / 8;
        let cache_limit = self.active_cache_budget_mb().saturating_mul(1_048_576);
        let cached = self
            .persistent_vector_segments
            .iter()
            .flat_map(|segments| segments.keys())
            .chain(
                self.lazy_vector_segment
                    .iter()
                    .map(|segment| &segment.collection_id),
            );
        // Changed segments remain owned until append succeeds, so they still consume budget.
        let retained = cached.fold(0u64, |bytes, id| {
            bytes.saturating_add(source.encoded_len(id).unwrap_or(u64::MAX))
        });
        let mut remaining = budget.saturating_sub(retained);
        for collection in self.catalog.live_collections() {
            let id = &collection.collection_id;
            if changed_collections.contains(id)
                || self
                    .persistent_vector_segments
                    .as_ref()
                    .is_some_and(|segments| segments.contains_key(id))
                || self
                    .lazy_vector_segment
                    .as_ref()
                    .is_some_and(|segment| &segment.collection_id == id)
            {
                continue;
            }
            let Some(bytes) = source.encoded_len(id).filter(|bytes| *bytes <= remaining) else {
                continue;
            };
            // This is an optional reuse path; unreadable suffixes fall back to primary records.
            let Some(reservation) = CachedVectorSegment::reserve(&context, cache_limit, bytes)
            else {
                continue;
            };
            if let Ok(Some(segment)) = source.load_cached(&self.storage, id, reservation, &context)
            {
                if segment.matches_collection(collection) && prefetched.insert(segment, cache_limit)
                {
                    remaining -= bytes;
                }
            }
        }
        prefetched
    }

    fn apply_committed_batch(
        &mut self,
        append: crate::storage::AppendSummary,
        changes: IndexChanges,
        reusable_checkpoint: bool,
        prefetched: VectorCache,
        reusable_text: Option<TextCheckpoint>,
    ) -> Result<PendingIndexUpdate> {
        let context = crate::json::JsonOperationContext::new(Arc::clone(&self.work));
        for id in &changes.vector {
            self.catalog.invalidate_model_state(id);
        }
        let cache_limit = self.active_cache_budget_mb().saturating_mul(1_048_576);
        let reusable_vectors = if reusable_checkpoint {
            let mut segments = self
                .persistent_vector_segments
                .take()
                .unwrap_or_else(|| VectorCache::new(&context));
            if let Some(segment) = self.lazy_vector_segment.take() {
                segments.insert(segment, cache_limit);
            }
            segments.append(prefetched, cache_limit);
            segments
        } else {
            VectorCache::new(&context)
        };
        // The critical commit is visible even if the following rebuild fails.
        self.mark_index_checkpoint_rebuild_needed();
        self.catalog.last_commit_id = append.commit.batch_id;
        self.last_commit_sequence_no = append.commit_sequence_no;
        Ok(PendingIndexUpdate {
            reusable_checkpoint,
            reusable_vectors,
            reusable_text,
            changes,
        })
    }

    fn finish_index_update(
        &mut self,
        pending: PendingIndexUpdate,
        context: &crate::json::JsonOperationContext,
    ) -> Result<()> {
        for id in &pending.changes.vector {
            if self.catalog.is_collection_live(id) {
                self.cache_collection_model(id)?;
            }
        }
        match self.catalog.require_consistent_models() {
            Err(error) if error.code() == ErrorCode::EmbeddingModelMismatch => return Ok(()),
            result => result?,
        }
        let PendingIndexUpdate {
            reusable_checkpoint,
            mut reusable_vectors,
            reusable_text,
            changes,
        } = pending;
        let mut resident = std::mem::take(&mut self.text_index);
        if !reusable_checkpoint {
            resident.clear_cache();
        }
        let text_checkpoint =
            self.prepare_text_checkpoint_reclaiming(reusable_text, &changes.text, context, || {
                let bytes = reusable_vectors.reserved_bytes();
                reusable_vectors.clear();
                resident.clear_cache() || bytes != 0
            });
        let text_checkpoint = match text_checkpoint {
            Ok(text) => text,
            Err(error) => {
                self.text_index = resident;
                return Err(error);
            }
        };
        let cache_limit = self.active_cache_budget_mb().saturating_mul(1_048_576);
        if self.lazy_text {
            resident.clear_cache();
        } else if !changes.text.is_empty() || !resident.has_resident_cache() {
            loop {
                let subset = reusable_checkpoint && resident.has_resident_cache();
                let built = build_text_index_from_checkpoint(
                    text_checkpoint.as_ref(),
                    &self.catalog,
                    |id| !subset || changes.text.contains(id),
                    context,
                    cache_limit,
                );
                let replacement = match built {
                    Err(error) if crate::json::is_memory_budget_error(&error) => {
                        let bytes = reusable_vectors.reserved_bytes();
                        reusable_vectors.clear();
                        if resident.clear_cache() || bytes != 0 {
                            continue;
                        }
                        self.text_index = resident;
                        return Err(error);
                    }
                    Err(error) => {
                        self.text_index = resident;
                        return Err(error);
                    }
                    Ok(replacement) => replacement,
                };
                if !subset {
                    resident = replacement;
                    break;
                }
                let old_chunk_ids = self
                    .catalog
                    .chunks
                    .values()
                    .filter(|chunk| changes.text.contains(&chunk.collection_id))
                    .map(|chunk| chunk.chunk_id.as_str());
                let restore = text_checkpoint
                    .as_ref()
                    .is_some_and(TextCheckpoint::has_zero_frequency_terms);
                if resident.replace_cached_collections(
                    &changes.text,
                    old_chunk_ids,
                    replacement,
                    cache_limit,
                )? && restore
                {
                    let checkpoint = text_checkpoint.as_ref().expect("prepared text checkpoint");
                    let mut restore = || {
                        resident.restore_empty_terms_from_visitors(
                            self.catalog
                                .live_collections()
                                .map(|collection| collection.collection_id.as_str()),
                            cache_limit,
                            |id, visit| checkpoint.body(id)?.visit(context, 0, visit),
                        )
                    };
                    if let Err(error) = restore() {
                        if !crate::json::is_memory_budget_error(&error) {
                            return Err(error);
                        }
                        // The failed optional candidate dropped before this pure validation retry.
                        restore()?;
                    }
                }
                break;
            }
        }
        self.text_index = resident;
        if let Err(_err) = self.refresh_index_checkpoint_prepared(
            reusable_vectors,
            &changes.vector,
            text_checkpoint,
            context,
        ) {
            self.mark_index_checkpoint_rebuild_needed();
        }
        Ok(())
    }

    fn materialize_document_chunks_in(
        &self,
        collection_id: &str,
        document_id: &str,
        context: &crate::json::JsonOperationContext,
    ) -> Result<Vec<Chunk>> {
        let summaries = self
            .catalog
            .chunks_for_document_in(collection_id, document_id, context)?;
        let mut retained =
            RecordReservation::new(context.reservation(), context.workspace_reservation());
        let mut chunks =
            crate::json::ReservedVec::with_capacity(summaries.len(), context, "operation_scratch")?;
        for summary in summaries.iter() {
            let (chunk, work) = self.materialize_chunk_in(&summary.chunk_id, context)?;
            retained.merge(work);
            chunks.push(chunk);
        }
        Ok(chunks.into_vec())
    }

    fn materialize_chunk_in(
        &self,
        chunk_id: &str,
        context: &crate::json::JsonOperationContext,
    ) -> Result<(Chunk, RecordReservation)> {
        let summary = self.catalog.chunks.get(chunk_id).ok_or_else(|| {
            Error::new(
                ErrorCode::ChunkNotFound,
                format!("chunk {chunk_id} not found"),
            )
        })?;
        let record_ref = self.catalog.record_ref_for_chunk(chunk_id);
        let retained = match record_ref {
            Some(record_ref) => self
                .storage
                .payload_reader()?
                .read_chunk_in(record_ref, context)?,
            None => RetainedChunk::copy(summary, context)?,
        };
        let mut retained = retained.with_embedding_id(summary.embedding_id.as_deref())?;
        let token_count = if record_ref.is_some() {
            Some(token_count_hint_in(&retained.parts().0.text, context)?)
        } else {
            None
        };
        let (mut chunk, work) = retained.into_parts();
        if let Some(count) = token_count {
            chunk.token_count = count;
        }
        Ok((chunk, work))
    }

    fn text_stats(&self) -> Option<&TextStatsSummary> {
        (self.index_suffix_status == IndexSuffixState::PersistentCheckpoint)
            .then_some(self.lazy_vector_source.as_ref())
            .flatten()?
            .text_stats(
                self.storage.critical_end_offset(),
                self.last_commit_sequence_no,
            )
    }

    fn index_stats(&self) -> IndexStats {
        let vector_count = self.catalog.live_embeddings().count();
        let collection_count = self.catalog.live_collections().count();
        let text_stats = self.text_stats();
        let persistent_ready = text_stats.is_some();
        let segment_count = if persistent_ready {
            collection_count
        } else {
            0
        };
        IndexStats {
            text_terms: text_stats.map_or(0, TextStatsSummary::term_count),
            text_segments: segment_count,
            vector_count,
            vector_segments: segment_count,
            rebuild_needed: !persistent_ready,
            index_size_bytes: if persistent_ready {
                self.index_size_bytes
            } else {
                0
            },
            last_index_commit_id: persistent_ready
                .then_some(self.last_index_commit_sequence_no)
                .flatten(),
        }
    }

    fn compacted_persistent_segment_count(&self) -> usize {
        if self.index_suffix_status != IndexSuffixState::PersistentCheckpoint {
            return 0;
        }
        let stats = self.index_stats();
        stats.text_segments + stats.vector_segments
    }

    #[cfg(test)]
    fn refresh_index_checkpoint(
        &mut self,
        context: &crate::json::JsonOperationContext,
    ) -> Result<Option<IndexSuffixReport>> {
        self.refresh_index_checkpoint_reusing(VectorCache::new(context), &BTreeSet::new(), context)
    }

    #[cfg(test)]
    fn refresh_index_checkpoint_reusing(
        &mut self,
        reusable_vectors: VectorCache,
        changed_vector_collections: &BTreeSet<String>,
        context: &crate::json::JsonOperationContext,
    ) -> Result<Option<IndexSuffixReport>> {
        let text = self.prepare_text_checkpoint(None, &BTreeSet::new(), context)?;
        self.refresh_index_checkpoint_prepared(
            reusable_vectors,
            changed_vector_collections,
            text,
            context,
        )
    }

    fn prepare_compact_text_checkpoint(
        &self,
        context: &crate::json::JsonOperationContext,
    ) -> Result<Option<TextCheckpoint>> {
        let budget_miss = |error: &Error| {
            error.code() == ErrorCode::ResourceExhausted
                && error.detail().iter().any(|(key, value)| {
                    key == "resource" && matches!(value.as_str(), "memory" | "temporary_disk")
                })
        };
        let changed = BTreeSet::new();
        let reuse = if self.text_stats().is_some() {
            match self.prefetch_unchanged_text(&changed, context) {
                Err(error) if budget_miss(&error) => None,
                result => result?,
            }
        } else {
            None
        };
        let reused = reuse.is_some();
        // Failed preparation drops its staging and credit before the primary fallback.
        match self.prepare_text_checkpoint(reuse, &changed, context) {
            Err(error) if reused && budget_miss(&error) => {
                self.prepare_text_checkpoint(None, &changed, context)
            }
            result => result,
        }
    }

    fn prefetch_unchanged_text(
        &self,
        changed: &BTreeSet<String>,
        context: &crate::json::JsonOperationContext,
    ) -> Result<Option<TextCheckpoint>> {
        if self.index_suffix_status != IndexSuffixState::PersistentCheckpoint {
            return Ok(None);
        }
        TextCheckpoint::prefetch(
            &self.storage,
            &self.catalog,
            self.lazy_vector_source.as_ref(),
            changed,
            &self.temporary_work,
            context,
        )
        .map(Some)
    }

    fn prepare_text_checkpoint(
        &self,
        reuse: Option<TextCheckpoint>,
        changed: &BTreeSet<String>,
        context: &crate::json::JsonOperationContext,
    ) -> Result<Option<TextCheckpoint>> {
        self.prepare_text_checkpoint_reclaiming(reuse, changed, context, || false)
    }

    fn prepare_text_checkpoint_reclaiming(
        &self,
        reuse: Option<TextCheckpoint>,
        changed: &BTreeSet<String>,
        context: &crate::json::JsonOperationContext,
        reclaim: impl FnMut() -> bool,
    ) -> Result<Option<TextCheckpoint>> {
        #[cfg(test)]
        tests::record_text_checkpoint_build();
        let mut reader = self.storage.payload_reader()?;
        let mut bytes_read = 0;
        TextCheckpoint::build_reclaiming(
            &self.storage,
            &self.catalog,
            self.last_commit_sequence_no,
            reuse.map(|reuse| (reuse, changed)),
            |visible| read_visible_chunk_text(&mut reader, visible, context, &mut bytes_read),
            &self.temporary_work,
            context,
            reclaim,
        )
        .map(Some)
    }

    fn refresh_index_checkpoint_prepared(
        &mut self,
        reusable_vectors: VectorCache,
        changed_vector_collections: &BTreeSet<String>,
        mut text: Option<TextCheckpoint>,
        context: &crate::json::JsonOperationContext,
    ) -> Result<Option<IndexSuffixReport>> {
        match self.refresh_index_checkpoint_attempt(
            reusable_vectors,
            changed_vector_collections,
            true,
            text.as_mut(),
            context,
        ) {
            Err(error) if crate::json::is_memory_budget_error(&error) => {
                // Only the rebuildable suffix is retried. Failed attempt caches have
                // already dropped; never repeat a committed primary-data write.
                self.text_index.clear_cache();
                self.refresh_index_checkpoint_attempt(
                    VectorCache::new(context),
                    changed_vector_collections,
                    false,
                    text.as_mut(),
                    context,
                )
            }
            result => result,
        }
    }

    fn refresh_index_checkpoint_attempt(
        &mut self,
        mut reusable_vectors: VectorCache,
        changed_vector_collections: &BTreeSet<String>,
        allow_optional_memory: bool,
        mut text: Option<&mut TextCheckpoint>,
        context: &crate::json::JsonOperationContext,
    ) -> Result<Option<IndexSuffixReport>> {
        if self.catalog.last_commit_id == 0 {
            self.index_suffix_status = IndexSuffixState::Missing;
            self.last_index_commit_sequence_no = None;
            self.index_size_bytes = 0;
            self.persistent_vector_segments = None;
            self.lazy_vector_source = None;
            self.lazy_vector_segment = None;
            return Ok(None);
        }
        let text = text.as_mut().ok_or_else(|| {
            Error::new(
                ErrorCode::IndexCorruption,
                "missing prepared text checkpoint",
            )
        })?;
        self.mark_index_checkpoint_rebuild_needed();
        #[cfg(test)]
        {
            self.index_checkpoint_refreshes += 1;
        }
        #[cfg(test)]
        if self.fail_next_index_checkpoint_refresh {
            self.fail_next_index_checkpoint_refresh = false;
            return Err(Error::new(
                ErrorCode::IoError,
                "injected index checkpoint refresh failure",
            ));
        }
        let budget_mb = self.active_cache_budget_mb();
        let cache_limit = budget_mb.saturating_mul(1_048_576);
        let budget = (budget_mb as u64).saturating_mul(1_048_576) / 8;
        let mut reused_bytes = 0u64;
        reusable_vectors.retain(|id, segment| {
            let eligible = !changed_vector_collections.contains(id)
                && self
                    .catalog
                    .collection_by_id(id)
                    .is_ok_and(|collection| segment.matches_collection(collection));
            eligible
                && segment.encoded_len().is_ok_and(|bytes| {
                    if bytes > budget.saturating_sub(reused_bytes) {
                        return false;
                    }
                    reused_bytes += bytes;
                    true
                })
        });
        if reusable_vectors.is_empty() || reusable_vectors.reserved_bytes() > cache_limit {
            reusable_vectors.clear();
            reused_bytes = 0;
        }
        for segment in reusable_vectors.values_mut() {
            segment.set_covered_commit(self.last_commit_sequence_no);
        }
        let checkpoint = self.index_checkpoint_metadata(context)?;
        // Reused segments remain owned throughout writing, including before their turn.
        let allocated = Cell::new(reused_bytes);
        let mut admitted = VectorCache::new(context);
        let reader = self.storage.payload_reader()?;
        let catalog = &self.catalog;
        let commit = self.last_commit_sequence_no;
        let text_segments = catalog.live_collections().map(|collection| {
            Ok(CheckpointSegment::StagedText {
                body: text.body(&collection.collection_id)?,
                commit,
            })
        });
        let vector_segments = catalog.live_collections().map(|collection| {
            match reusable_vectors.get(&collection.collection_id) {
                Some(segment) => Ok(CheckpointSegment::BorrowedCachedVector(segment)),
                None => {
                    let entries = catalog
                        .live_embeddings()
                        .filter(|embedding| embedding.collection_id == collection.collection_id)
                        .map(|embedding| {
                            (
                                embedding,
                                catalog.record_ref_for_embedding(&embedding.embedding_id),
                            )
                        });
                    let mut source = VectorRecordSegment::new(
                        collection,
                        commit,
                        entries,
                        reader.try_clone()?,
                        context,
                    )?;
                    let encoded_len = source.encoded_len()?;
                    if allow_optional_memory
                        && encoded_len <= budget.saturating_sub(allocated.get())
                    {
                        if let Some(segment) = source.try_materialize(cache_limit)? {
                            return Ok(CheckpointSegment::CachedVector(segment));
                        }
                    }
                    Ok(CheckpointSegment::VectorRecords(source))
                }
            }
        });
        let (report, source) = self.storage.write_index_segments_in(
            &checkpoint,
            text_segments.chain(vector_segments),
            |segment, bytes| {
                if let CheckpointSegment::CachedVector(segment) = segment {
                    if admitted.insert(segment, cache_limit) {
                        allocated.set(allocated.get().saturating_add(bytes));
                    }
                }
            },
            self.config.durability,
            context,
            allow_optional_memory,
        )?;
        admitted.append(reusable_vectors, cache_limit);
        self.persistent_vector_segments = Some(admitted);
        let text_stats = text.take_stats();
        self.lazy_vector_source = Some(source.with_text_stats(text_stats));
        self.index_suffix_status = IndexSuffixState::from_report(&report);
        self.last_index_commit_sequence_no = index_commit_sequence_no_from_report(&report);
        self.index_size_bytes = report.total_size_bytes();
        Ok(Some(report))
    }

    fn prepare_text_index(
        &mut self,
        cache_budget_mb: usize,
        context: &crate::json::JsonOperationContext,
    ) -> Result<()> {
        if self.text_index.has_resident_cache()
            || self.index_suffix_status != IndexSuffixState::PersistentCheckpoint
        {
            return Ok(());
        }
        let Some(source) = &self.lazy_vector_source else {
            return Ok(());
        };
        let limit = cache_budget_mb.saturating_mul(1_048_576);
        let remaining = limit.saturating_sub(self.optional_cache_bytes());
        if source
            .text_encoded_len()
            .map_or(true, |bytes| bytes > (remaining as u64) / 8)
        {
            return Ok(());
        }
        match build_text_cache_in(
            &self.catalog,
            |_| true,
            context,
            limit,
            |id, input, visit| source.visit_text(&self.storage, id, input, visit),
        ) {
            Ok(TextCacheBuild::Complete(index)) => self.text_index = index,
            // Prewarming other collections must not invalidate the target's
            // source. Its normal streaming path still validates required bytes.
            Ok(TextCacheBuild::Miss | TextCacheBuild::Invalid) => {}
            Err(error) if crate::json::is_memory_budget_error(&error) => {}
            Err(error) => return Err(error),
        }
        Ok(())
    }

    #[cfg(test)]
    fn prepare_vector_index(&mut self, collection_id: &str, cache_budget_mb: usize) -> Result<()> {
        self.prepare_vector_index_with_context(
            collection_id,
            cache_budget_mb,
            &crate::json::JsonOperationContext::standalone(),
        )
    }

    fn prepare_vector_index_with_context(
        &mut self,
        collection_id: &str,
        cache_budget_mb: usize,
        caller_context: &crate::json::JsonOperationContext,
    ) -> Result<()> {
        let Some(source) = &self.lazy_vector_source else {
            return Ok(());
        };
        let Some(encoded_len) = source.encoded_len(collection_id) else {
            self.mark_index_checkpoint_rebuild_needed();
            return Ok(());
        };
        // Preserve the existing decoder headroom policy, now applied to one collection.
        let load_budget = (cache_budget_mb as u64).saturating_mul(1_048_576) / 8;
        let mut cached_bytes = self
            .persistent_vector_segments
            .as_ref()
            .map_or(0, |segments| {
                segments.keys().fold(0u64, |total, id| {
                    total.saturating_add(source.encoded_len(id).unwrap_or(u64::MAX))
                })
            });
        let lazy_bytes = self.lazy_vector_segment.as_ref().map_or(0, |segment| {
            source
                .encoded_len(&segment.collection_id)
                .unwrap_or(u64::MAX)
        });
        let actual_bytes = self
            .persistent_vector_segments
            .as_ref()
            .map_or(0, VectorCache::reserved_bytes)
            + self
                .lazy_vector_segment
                .as_ref()
                .map_or(0, CachedVectorSegment::reserved_bytes);
        if cached_bytes.saturating_add(lazy_bytes) > load_budget
            || actual_bytes > cache_budget_mb.saturating_mul(1_048_576)
        {
            self.persistent_vector_segments = None;
            self.lazy_vector_segment = None;
            cached_bytes = 0;
        }
        if self
            .persistent_vector_segments
            .as_ref()
            .is_some_and(|segments| segments.contains_key(collection_id))
        {
            return Ok(());
        }
        if encoded_len > load_budget {
            self.lazy_vector_segment = None;
            return Ok(());
        }
        if self
            .lazy_vector_segment
            .as_ref()
            .is_some_and(|segment| segment.collection_id == collection_id)
        {
            return Ok(());
        }
        self.lazy_vector_segment = None;
        if cached_bytes.saturating_add(encoded_len) > load_budget {
            self.persistent_vector_segments = None;
        }
        let mut context = crate::json::JsonOperationContext::new(Arc::clone(&self.work));
        context.inherit_search_cost(caller_context);
        let cache_limit = cache_budget_mb.saturating_mul(1_048_576);
        let mut reservation = CachedVectorSegment::reserve(&context, cache_limit, encoded_len);
        if reservation.is_none() && self.persistent_vector_segments.is_some() {
            self.persistent_vector_segments = None;
            reservation = CachedVectorSegment::reserve(&context, cache_limit, encoded_len);
        }
        let Some(reservation) = reservation else {
            return Ok(());
        };
        match source.load_cached(&self.storage, collection_id, reservation, &context)? {
            Some(segment) => self.lazy_vector_segment = Some(segment),
            None => self.mark_index_checkpoint_rebuild_needed(),
        }
        Ok(())
    }

    fn search_checkpoint_text_index(
        &mut self,
        collection_id: &str,
        query: &str,
        options: TextSearchOptions,
        context: &crate::json::JsonOperationContext,
    ) -> Result<Option<Bm25Hits>> {
        if self.index_suffix_status != IndexSuffixState::PersistentCheckpoint {
            return Ok(None);
        }
        if self.text_index.has_resident_cache() {
            match self
                .text_index
                .search(collection_id, query, options, context)
            {
                Err(error) if crate::json::is_memory_budget_error(&error) => {
                    if !Self::release_optional_caches(
                        &mut self.text_index,
                        &mut self.persistent_vector_segments,
                        &mut self.lazy_vector_segment,
                        std::iter::empty(),
                    ) {
                        return Err(error);
                    }
                }
                result => return result.map(Some),
            }
        }
        self.search_uncached_text_index(collection_id, query, options, context)
    }

    fn search_uncached_text_index(
        &mut self,
        collection_id: &str,
        query: &str,
        options: TextSearchOptions,
        context: &crate::json::JsonOperationContext,
    ) -> Result<Option<Bm25Hits>> {
        if self.text_index.has_resident_cache()
            || self.index_suffix_status != IndexSuffixState::PersistentCheckpoint
        {
            return Ok(None);
        }
        if !self
            .catalog
            .live_chunks()
            .any(|chunk| chunk.collection_id == collection_id)
        {
            return Ok(Some(Bm25Hits::empty(context)));
        }
        let Some(source) = &self.lazy_vector_source else {
            return Ok(None);
        };
        let result = TextIndex::search_from_segment(
            source,
            &self.storage,
            collection_id,
            query,
            options,
            context,
        )?;
        if result.is_none() {
            self.mark_index_checkpoint_rebuild_needed();
        }
        Ok(result)
    }

    #[cfg(test)]
    fn search_checkpoint_vector_index(
        &mut self,
        collection: &CollectionInfo,
        query: &[f32],
        limit: usize,
        filter: Option<&FilterExpr>,
        context: &crate::json::JsonOperationContext,
    ) -> Result<Option<(VectorHits, usize)>> {
        let mut work = VectorWork::default();
        Ok(self
            .search_checkpoint_vector_index_with_work(
                collection, query, limit, filter, context, &mut work,
            )?
            .map(|hits| (hits, work.scans)))
    }

    #[cfg(test)]
    fn search_uncached_vector_index(
        &mut self,
        collection: &CollectionInfo,
        query: &[f32],
        limit: usize,
        filter: Option<&FilterExpr>,
        context: &crate::json::JsonOperationContext,
    ) -> Result<Option<(VectorHits, usize)>> {
        let mut work = VectorWork::default();
        Ok(self
            .search_uncached_vector_index_with_work(
                collection, query, limit, filter, context, &mut work,
            )?
            .map(|hits| (hits, work.scans)))
    }

    #[allow(clippy::too_many_arguments)]
    fn search_checkpoint_vector_index_with_work(
        &mut self,
        collection: &CollectionInfo,
        query: &[f32],
        limit: usize,
        filter: Option<&FilterExpr>,
        context: &crate::json::JsonOperationContext,
        work: &mut VectorWork,
    ) -> Result<Option<VectorHits>> {
        let cached = self.read_reclaiming(|inner| {
            let segment = inner
                .persistent_vector_segments
                .as_ref()
                .and_then(|segments| segments.get(&collection.collection_id))
                .or_else(|| {
                    inner
                        .lazy_vector_segment
                        .as_ref()
                        .filter(|segment| segment.collection_id == collection.collection_id)
                });
            segment
                .map(|segment| {
                    search_persistent_vector_segment_with_work(
                        segment,
                        query,
                        collection.distance_metric,
                        limit,
                        |id| {
                            matches_vector_filter(
                                &inner.catalog,
                                &collection.collection_id,
                                filter,
                                id,
                            )
                        },
                        context,
                        work,
                    )
                })
                .transpose()
        })?;
        // A failed cached attempt is gone before central reclaim; source rows are
        // visited only after cache ownership and its borrowed segment have ended.
        match cached {
            Some(hits) => Ok(Some(hits)),
            None => self.search_uncached_vector_index_with_work(
                collection, query, limit, filter, context, work,
            ),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn search_uncached_vector_index_with_work(
        &mut self,
        collection: &CollectionInfo,
        query: &[f32],
        limit: usize,
        filter: Option<&FilterExpr>,
        context: &crate::json::JsonOperationContext,
        work: &mut VectorWork,
    ) -> Result<Option<VectorHits>> {
        if self
            .persistent_vector_segments
            .as_ref()
            .is_some_and(|segments| segments.contains_key(&collection.collection_id))
            || self
                .lazy_vector_segment
                .as_ref()
                .is_some_and(|segment| segment.collection_id == collection.collection_id)
        {
            return Ok(None);
        }
        let Some(source) = &self.lazy_vector_source else {
            return Ok(None);
        };
        let mut input = source.reserve_read_buffer(&collection.collection_id, context)?;
        let scorer = VectorScorer::new(query, collection.distance_metric);
        loop {
            let mut candidates = VectorCandidates::new(limit, context);
            let result = source.visit(&self.storage, collection, &input, |chunk_id, values| {
                work.scans += 1;
                context.search_cost_add(crate::search_cost::Counter::VectorChecks, 1);
                if !matches_vector_filter(
                    &self.catalog,
                    &collection.collection_id,
                    filter,
                    chunk_id,
                ) {
                    return Ok(());
                }
                let raw = scorer.score(values);
                work.distance_ops += values.len();
                context.search_cost_add(
                    crate::search_cost::Counter::DistanceComponents,
                    values.len() as u64,
                );
                candidates.push(
                    chunk_id,
                    raw,
                    normalize_score(
                        raw,
                        collection.distance_metric,
                        collection.vectors_normalized,
                    ),
                )?;
                Ok(())
            });
            match result {
                Err(error)
                    if error.code() == ErrorCode::ResourceExhausted
                        && input.reserved_bytes() > 0 =>
                {
                    // Retry a validated frame once without optional buffering.
                    // Discard all provisional candidates before reclaiming its quota.
                    drop(candidates);
                    input.retain(0);
                }
                Err(error) => return Err(error),
                Ok(true) => return Ok(Some(candidates.finish())),
                Ok(false) => {
                    self.mark_index_checkpoint_rebuild_needed();
                    return Ok(None);
                }
            }
        }
    }

    fn mark_index_checkpoint_rebuild_needed(&mut self) {
        self.index_suffix_status = IndexSuffixState::RebuildNeeded;
        self.last_index_commit_sequence_no = None;
        self.index_size_bytes = 0;
        self.persistent_vector_segments = None;
        self.lazy_vector_source = None;
        self.lazy_vector_segment = None;
    }

    #[cfg(test)]
    fn vector_index_segments(
        &self,
        mut reusable: BTreeMap<String, VectorIndexSegmentPayload>,
        changed_collections: &BTreeSet<String>,
    ) -> Result<Vec<VectorIndexSegmentPayload>> {
        reusable.retain(|id, segment| {
            !changed_collections.contains(id)
                && self
                    .catalog
                    .collection_by_id(id)
                    .is_ok_and(|collection| segment.matches_collection(collection))
        });
        let mut reader = self.storage.payload_reader()?;
        self.catalog
            .live_collections()
            .map(|collection| {
                if let Some(mut segment) = reusable.remove(&collection.collection_id) {
                    segment.covered_commit_sequence_no = self.last_commit_sequence_no;
                    return Ok(segment);
                }
                vector_segment_for_collection(
                    &self.catalog,
                    &mut reader,
                    collection,
                    self.last_commit_sequence_no,
                )
            })
            .collect()
    }

    fn index_checkpoint_metadata(
        &self,
        context: &crate::json::JsonOperationContext,
    ) -> Result<IndexCheckpointMetadata> {
        let collection_count = self.catalog.live_collections().count();
        Ok(IndexCheckpointMetadata {
            covered_last_batch_id: self.catalog.last_commit_id,
            covered_last_commit_sequence_no: self.last_commit_sequence_no,
            coverage_hash: catalog_index_coverage_hash(
                &self.catalog,
                self.storage.critical_end_offset(),
                self.last_commit_sequence_no,
                context,
            )?,
            collection_count,
            live_document_count: self.catalog.live_documents().count(),
            live_chunk_count: self.catalog.live_chunks().count(),
            live_embedding_count: self.catalog.live_embeddings().count(),
            text_segment_count: collection_count,
            vector_segment_count: collection_count,
            build_mode: "inline_full_checkpoint".into(),
        })
    }

    fn active_cache_budget_mb(&self) -> usize {
        self.config
            .cache_budget_mb
            .unwrap_or_else(|| self.config.resource_profile.params().cache_budget_mb)
    }
}

fn compaction_records<'a>(
    catalog: &'a Catalog,
    mut reader: PayloadReader,
    context: &'a crate::json::JsonOperationContext,
) -> impl Iterator<Item = Result<(RecordPayload, RecordReservation)>> + 'a {
    enum Source<'a> {
        Collection(&'a CollectionInfo),
        DroppedCollection(&'a CollectionInfo),
        Document(&'a Document),
        Chunk(&'a Chunk),
        Embedding(&'a Embedding),
    }
    let deleted_collections = || {
        catalog.collections.values().filter(|collection| {
            catalog
                .deleted_collections
                .contains(&collection.collection_id)
        })
    };
    catalog
        .live_collections()
        .chain(deleted_collections())
        .map(Source::Collection)
        .chain(deleted_collections().map(Source::DroppedCollection))
        .chain(catalog.live_documents().map(Source::Document))
        .chain(catalog.live_chunks().map(Source::Chunk))
        .chain(catalog.live_embeddings().map(Source::Embedding))
        .map(move |record| match record {
            Source::Collection(collection) => {
                let (collection, work) = copy_collection_in(collection, context)?;
                Ok((RecordPayload::CollectionCreate(collection), work))
            }
            Source::DroppedCollection(collection) => {
                let mut work = RecordReservation::from(context.reservation());
                Ok((
                    RecordPayload::CollectionDrop {
                        collection_id: work.copy_string(&collection.collection_id, context)?,
                        name: work.copy_string(&collection.name, context)?,
                        reason: None,
                        dropped_at: work.copy_string(
                            catalog
                                .collection_drop_times
                                .get(&collection.collection_id)
                                .unwrap_or(&collection.created_at),
                            context,
                        )?,
                    },
                    work,
                ))
            }
            Source::Document(document) => {
                let (document, work) = copy_document_in(document, context)?;
                Ok((RecordPayload::DocumentPut(document), work))
            }
            Source::Chunk(summary) => {
                let mut retained = match catalog.record_ref_for_chunk(&summary.chunk_id) {
                    Some(record_ref) => reader.read_chunk_in(record_ref, context)?,
                    None => RetainedChunk::copy(summary, context)?,
                };
                let token_count = token_count_hint_in(&retained.parts().0.text, context)?;
                let (mut chunk, work) = retained.into_parts();
                chunk.token_count = token_count;
                // Embedding IDs are reconstructed from committed EmbeddingPut records.
                Ok((RecordPayload::ChunkPut(chunk), work))
            }
            Source::Embedding(summary) => {
                let retained = match catalog.record_ref_for_embedding(&summary.embedding_id) {
                    Some(record_ref) => reader.read_embedding_in(record_ref, context)?,
                    None => RetainedEmbedding::copy(summary, context)?,
                };
                let (embedding, workspace) = retained.into_parts();
                // Binary EmbeddingPut omits the alias; recovery derives it from the chunk.
                Ok((
                    RecordPayload::EmbeddingPut(embedding),
                    RecordReservation::new(context.reservation(), workspace),
                ))
            }
        })
}

fn index_suffix_status_from_scan(
    scan: &ScanReport,
    catalog: &Catalog,
    context: &crate::json::JsonOperationContext,
) -> Result<IndexSuffixState> {
    let matches = scan
        .index_suffix
        .as_ref()
        .map(|suffix| index_suffix_matches_catalog(catalog, suffix, context))
        .transpose()?
        .unwrap_or(true);
    Ok(index_suffix_status_with_coverage(scan, matches))
}

fn index_suffix_status_with_coverage(scan: &ScanReport, matches: bool) -> IndexSuffixState {
    let has_rebuild_index_issue = scan.findings.iter().any(index_issue_requires_rebuild);
    if has_rebuild_index_issue || !matches {
        IndexSuffixState::RebuildNeeded
    } else {
        scan.index_suffix
            .as_ref()
            .map(IndexSuffixState::from_report)
            .unwrap_or(IndexSuffixState::Missing)
    }
}

fn index_commit_sequence_no_from_scan(
    scan: &ScanReport,
    index_suffix_status: IndexSuffixState,
) -> Option<u64> {
    if index_suffix_status != IndexSuffixState::PersistentCheckpoint {
        return None;
    }
    scan.index_suffix
        .as_ref()
        .and_then(index_commit_sequence_no_from_report)
}

fn index_commit_sequence_no_from_report(report: &IndexSuffixReport) -> Option<u64> {
    (IndexSuffixState::from_report(report) == IndexSuffixState::PersistentCheckpoint)
        .then_some(report.covered_last_commit_sequence_no)
}

fn index_issue_requires_rebuild(issue: &CheckIssue) -> bool {
    issue.stream == "index" && issue.code != "checkpoint_payload_checksum_mismatch"
}

fn index_suffix_matches_catalog(
    catalog: &Catalog,
    suffix: &IndexSuffixReport,
    context: &crate::json::JsonOperationContext,
) -> Result<bool> {
    let live_collection_count = catalog.live_collections().count();
    let hash = match catalog_index_coverage_hash(
        catalog,
        suffix.covered_critical_end_offset,
        suffix.covered_last_commit_sequence_no,
        context,
    ) {
        Ok(hash) => hash,
        Err(error)
            if matches!(
                error.code(),
                ErrorCode::FileFormatError | ErrorCode::InvalidEmbeddingDimension
            ) =>
        {
            return Ok(false)
        }
        Err(error) => return Err(error),
    };
    Ok(suffix.coverage_hash == hash
        && (IndexSuffixState::from_report(suffix) == IndexSuffixState::MetadataRecordOnly
            || (suffix.text_segment_count == live_collection_count
                && suffix.vector_segment_count == live_collection_count
                && suffix.live_chunk_count == catalog.live_chunks().count()
                && suffix.live_embedding_count == catalog.live_embeddings().count())))
}

#[cfg(test)]
fn vector_segment_for_collection(
    catalog: &Catalog,
    reader: &mut PayloadReader,
    collection: &CollectionInfo,
    commit: u64,
) -> Result<VectorIndexSegmentPayload> {
    let mut vectors = catalog
        .live_embeddings()
        .filter(|embedding| embedding.collection_id == collection.collection_id)
        .map(|summary| {
            let embedding = match catalog.record_ref_for_embedding(&summary.embedding_id) {
                Some(reference) => match reader.read_payload(reference)? {
                    RecordPayload::EmbeddingPut(embedding) => embedding,
                    _ => {
                        return Err(Error::new(
                            ErrorCode::FileFormatError,
                            format!(
                            "record ref for embedding {} did not contain an EmbeddingPut payload",
                            summary.embedding_id
                        ),
                        ))
                    }
                },
                None => summary.clone(),
            };
            Ok(VectorIndexEntry {
                chunk_id: embedding.chunk_id,
                embedding_id: embedding.embedding_id,
                vector: embedding.vector,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    vectors.sort_unstable_by_key(|entry| {
        entry
            .chunk_id
            .strip_prefix("chk_")
            .unwrap()
            .parse::<u64>()
            .unwrap()
    });
    Ok(VectorIndexSegmentPayload {
        collection_id: collection.collection_id.clone(),
        covered_commit_sequence_no: commit,
        dim: collection.embedding_dim,
        distance_metric: collection.distance_metric,
        embedding_model: collection.embedding_model.clone(),
        vectors_normalized: collection.vectors_normalized,
        vectors,
    })
}

#[cfg(test)]
fn vector_segments_by_collection(
    segments: impl IntoIterator<Item = VectorIndexSegmentPayload>,
) -> BTreeMap<String, VectorIndexSegmentPayload> {
    segments
        .into_iter()
        .map(|segment| (segment.collection_id.clone(), segment))
        .collect()
}

fn matches_vector_filter(
    catalog: &Catalog,
    collection_id: &str,
    filter: Option<&FilterExpr>,
    chunk_id: &str,
) -> bool {
    filter.map_or(true, |expr| {
        catalog.chunks.get(chunk_id).is_some_and(|chunk| {
            chunk.collection_id == collection_id
                && catalog.is_chunk_live(chunk_id)
                && expr.matches_for_chunk(catalog, chunk)
        })
    })
}

#[cfg(test)]
fn search_persistent_vector_segment(
    segment: &VectorIndexSegmentPayload,
    query: &[f32],
    metric: DistanceMetric,
    limit: usize,
    include: impl Fn(&str) -> bool,
    context: &crate::json::JsonOperationContext,
) -> Result<(VectorHits, VectorWork)> {
    let mut work = VectorWork::default();
    let hits = search_persistent_vector_segment_with_work(
        segment, query, metric, limit, include, context, &mut work,
    )?;
    Ok((hits, work))
}

#[allow(clippy::too_many_arguments)]
fn search_persistent_vector_segment_with_work(
    segment: &VectorIndexSegmentPayload,
    query: &[f32],
    metric: DistanceMetric,
    limit: usize,
    include: impl Fn(&str) -> bool,
    context: &crate::json::JsonOperationContext,
    work: &mut VectorWork,
) -> Result<VectorHits> {
    let mut candidates = VectorCandidates::new(limit, context);
    let scorer = VectorScorer::new(query, metric);
    for entry in &segment.vectors {
        work.scans += 1;
        context.search_cost_add(crate::search_cost::Counter::VectorChecks, 1);
        if !include(&entry.chunk_id) || entry.vector.len() != query.len() {
            continue;
        }
        let raw = scorer.score(&entry.vector);
        work.distance_ops += entry.vector.len();
        context.search_cost_add(
            crate::search_cost::Counter::DistanceComponents,
            entry.vector.len() as u64,
        );
        candidates.push(
            &entry.chunk_id,
            raw,
            normalize_score(raw, metric, segment.vectors_normalized),
        )?;
    }
    Ok(candidates.finish())
}

fn build_text_index_from_checkpoint(
    checkpoint: Option<&TextCheckpoint>,
    catalog: &Catalog,
    include: impl Fn(&str) -> bool,
    context: &crate::json::JsonOperationContext,
    cache_limit: usize,
) -> Result<TextIndex> {
    let Some(checkpoint) = checkpoint else {
        if catalog.last_commit_id == 0 {
            return Ok(TextIndex::default());
        }
        return Err(Error::new(
            ErrorCode::IndexCorruption,
            "missing prepared text checkpoint",
        ));
    };
    match build_text_cache_in(
        catalog,
        include,
        context,
        cache_limit,
        |id, input, visit| {
            checkpoint
                .body(id)?
                .visit(context, input.reserved_bytes(), visit)
        },
    )? {
        TextCacheBuild::Complete(index) => Ok(index),
        TextCacheBuild::Miss => Ok(TextIndex::default()),
        TextCacheBuild::Invalid => Err(Error::new(
            ErrorCode::IndexCorruption,
            "incomplete prepared text checkpoint",
        )),
    }
}

fn build_text_cache_in(
    catalog: &Catalog,
    include: impl Fn(&str) -> bool,
    context: &crate::json::JsonOperationContext,
    mut cache_limit: usize,
    mut visit: impl FnMut(
        &str,
        &crate::json::WorkspaceReservation,
        &mut crate::storage::TextSegmentVisitor<'_>,
    ) -> Result<bool>,
) -> Result<TextCacheBuild> {
    let mut input = context.workspace_reservation();
    match input.grow_to(0, 64 * 1024, "operation_scratch") {
        Ok(_) => {}
        Err(error) if crate::json::is_memory_budget_error(&error) => {}
        Err(error) => return Err(error),
    }
    loop {
        let result = TextIndex::try_cache_from_visitors(
            catalog
                .live_collections()
                .filter(|collection| include(&collection.collection_id))
                .map(|collection| collection.collection_id.as_str()),
            context,
            cache_limit,
            |id, callback| visit(id, &input, callback),
        );
        match result {
            Err(error)
                if crate::json::is_memory_budget_error(&error) && input.reserved_bytes() > 0 =>
            {
                input.retain(0);
            }
            Err(error) if crate::json::is_memory_budget_error(&error) && cache_limit != 0 => {
                // An optional candidate can crowd later decoder scratch. Validate again without it.
                cache_limit = 0;
            }
            Ok(TextCacheBuild::Miss) if input.reserved_bytes() > 0 => input.retain(0),
            result => return result,
        }
    }
}

#[cfg(test)]
fn build_text_index(
    storage: &Storage,
    catalog: &Catalog,
    disk: &Arc<crate::temporary::TemporaryDiskBudget>,
    context: &crate::json::JsonOperationContext,
) -> Result<TextIndex> {
    build_text_index_matching(storage, catalog, |_| true, disk, context).map(|(index, _)| index)
}

#[cfg(test)]
fn build_text_index_matching(
    storage: &Storage,
    catalog: &Catalog,
    include_collection: impl Fn(&str) -> bool,
    disk: &Arc<crate::temporary::TemporaryDiskBudget>,
    context: &crate::json::JsonOperationContext,
) -> Result<(TextIndex, u64)> {
    #[cfg(test)]
    tests::record_text_build();
    let mut payload_reader = storage.payload_reader()?;
    let mut bytes_read = 0u64;
    let index = TextIndex::try_build_matching_in(
        catalog,
        include_collection,
        |visible| {
            let mut attempt_bytes = 0;
            let text =
                read_visible_chunk_text(&mut payload_reader, visible, context, &mut attempt_bytes)?;
            bytes_read = bytes_read.saturating_add(attempt_bytes);
            Ok(text)
        },
        storage.path().parent().unwrap_or_else(|| Path::new(".")),
        Arc::clone(disk),
        context,
    )?;
    Ok((index, bytes_read))
}

fn build_check_report(
    path: &Path,
    options: CheckOptions,
    context: &crate::json::JsonOperationContext,
) -> Result<CheckReport> {
    build_check_report_reclaiming(path, options, context, || false)
}

fn build_check_report_reclaiming(
    path: &Path,
    options: CheckOptions,
    context: &crate::json::JsonOperationContext,
    reclaim: impl FnOnce() -> bool,
) -> Result<CheckReport> {
    #[cfg(test)]
    tests::record_check_scan();
    let mut catalog = Catalog::new_in(context);
    let scan = scan_file_with_retained_records_in(path, context, |record, work| {
        catalog.apply_retained(record, work, context)
    })?;
    catalog.last_commit_id = scan.last_valid_commit;
    #[cfg(test)]
    let _pressure = tests::hold_check_coverage_pressure(context);
    let matches = if let Some(suffix) = &scan.index_suffix {
        crate::json::retry_after_memory_reclaim(
            || index_suffix_matches_catalog(&catalog, suffix, context),
            reclaim,
        )?
    } else {
        true
    };
    let index_suffix_status = index_suffix_status_with_coverage(&scan, matches);
    let mut issues = scan.findings.clone();
    for collection in catalog.live_collections() {
        if let crate::catalog::ModelState::Conflict(witness) =
            catalog.visible_model_state(&collection.collection_id)?
        {
            let reference = catalog
                .record_refs
                .get(&witness.embedding_id)
                .ok_or_else(|| {
                    Error::new(
                        ErrorCode::FileFormatError,
                        "visible model conflict has no committed record reference",
                    )
                })?;
            issues.push(CheckIssue {
                severity: "error".into(),
                code: "embedding_model_mismatch".into(),
                message: format!("collection {} has conflicting visible embedding models; export and explicitly clean up its data", collection.name),
                stream: "critical".into(),
                visibility_impact: "none".into(),
                offset: Some(reference.offset),
                extent_id: None,
                record_type: Some("EmbeddingPut".into()),
                batch_id: Some(reference.batch_id),
                repair_available: false,
                suggested_command: None,
            });
        }
    }
    if let Some(suffix) = &scan.index_suffix {
        if !matches {
            issues.push(CheckIssue {
                severity: "warning".into(),
                code: "index_coverage_hash_mismatch".into(),
                message: "ignored index checkpoint whose coverage hash does not match the visible catalog snapshot".into(),
                stream: "index".into(),
                visibility_impact: "rebuild_index_only".into(),
                offset: Some(suffix.offset),
                extent_id: None,
                record_type: Some("IndexCheckpointMetadata".into()),
                batch_id: Some(scan.last_valid_commit),
                repair_available: false,
                suggested_command: None,
            });
        }
    }
    attach_check_issue_suggestions(path, &catalog, &mut issues);
    let status = if issues.iter().any(|issue| issue.severity == "error") {
        "error"
    } else if issues.iter().any(|issue| issue.severity == "warning") {
        "warning"
    } else {
        "ok"
    };
    let repair_available = issues.iter().any(|issue| issue.repair_available);
    let has_rebuild_index_issue = issues.iter().any(index_issue_requires_rebuild);
    let (index_check, index_check_reason) = if options.verify_indexes && has_rebuild_index_issue {
        let reason = issues
            .iter()
            .find(|issue| index_issue_requires_rebuild(issue))
            .map(|issue| issue.code.clone())
            .unwrap_or_else(|| "index_suffix_unreadable".into());
        ("rebuild_needed".into(), Some(reason))
    } else if options.verify_indexes
        && index_suffix_status == IndexSuffixState::PersistentCheckpoint
    {
        ("ok".into(), None)
    } else if options.verify_indexes && index_suffix_status == IndexSuffixState::MetadataRecordOnly
    {
        (
            "skipped".into(),
            Some("persistent_index_segments_not_implemented".into()),
        )
    } else if options.verify_indexes {
        (
            "skipped".into(),
            Some("persistent_index_checkpoint_missing".into()),
        )
    } else {
        ("disabled".into(), None)
    };
    Ok(CheckReport {
        schema_version: "0.1".into(),
        path: path.to_path_buf(),
        status: status.into(),
        format_version: FORMAT_VERSION,
        last_valid_commit: scan.last_valid_commit,
        critical_end_offset: scan.critical_end_offset,
        index_suffix_status: index_suffix_status.as_str().into(),
        index_suffix: scan.index_suffix,
        issues,
        repair_available,
        index_check,
        index_check_reason,
        live_documents: catalog.live_documents().count(),
        live_chunks: catalog.live_chunks().count(),
        tombstones: catalog.tombstone_count(),
    })
}

fn attach_check_issue_suggestions(path: &Path, catalog: &Catalog, issues: &mut [CheckIssue]) {
    for issue in issues {
        if issue.suggested_command.is_some() {
            continue;
        }
        issue.suggested_command = check_issue_suggested_command(path, catalog, issue);
    }
}

fn check_issue_suggested_command(
    path: &Path,
    catalog: &Catalog,
    issue: &CheckIssue,
) -> Option<String> {
    if issue.stream == "index" && issue.visibility_impact == "rebuild_index_only" {
        let mut live_collections = catalog.live_collections();
        if let Some(collection) = live_collections.next() {
            if live_collections.next().is_none() {
                return Some(format!(
                    "coffer rebuild-index {} --collection {}",
                    path.display(),
                    collection.name.as_str()
                ));
            }
        }
        return Some(format!("coffer rebuild-index {}", path.display()));
    }
    if issue.repair_available || issue.visibility_impact == "tail_only" {
        return Some(format!("coffer repair {} --truncate-tail", path.display()));
    }
    None
}

impl CheckReport {
    pub fn to_json(&self) -> JsonValue {
        self.build_json(&JsonBuilder::host())
            .expect("host DTO construction is infallible")
    }

    pub(crate) fn build_json(&self, builder: &JsonBuilder<'_>) -> Result<JsonValue> {
        builder.object([
            ("schema_version", builder.string(&self.schema_version)?),
            ("path", builder.display(self.path.display())?),
            ("status", builder.string(&self.status)?),
            (
                "format_version",
                JsonValue::Number(self.format_version as f64),
            ),
            (
                "last_valid_commit",
                JsonValue::Number(self.last_valid_commit as f64),
            ),
            (
                "critical_end_offset",
                JsonValue::Number(self.critical_end_offset as f64),
            ),
            (
                "index_suffix_status",
                builder.string(&self.index_suffix_status)?,
            ),
            (
                "index_suffix",
                self.index_suffix
                    .as_ref()
                    .map(|suffix| suffix.build_json(builder))
                    .transpose()?
                    .unwrap_or(JsonValue::Null),
            ),
            (
                "issues",
                builder.array(self.issues.iter().map(|issue| issue.build_json(builder)))?,
            ),
            ("repair_available", JsonValue::Bool(self.repair_available)),
            ("index_check", builder.string(&self.index_check)?),
            (
                "index_check_reason",
                builder.optional_str(self.index_check_reason.as_deref())?,
            ),
            (
                "result",
                builder.object([
                    (
                        "live_documents",
                        JsonValue::Number(self.live_documents as f64),
                    ),
                    ("live_chunks", JsonValue::Number(self.live_chunks as f64)),
                    ("tombstones", JsonValue::Number(self.tombstones as f64)),
                ])?,
            ),
        ])
    }
}

impl CompactSummary {
    pub fn to_json(&self) -> JsonValue {
        self.build_json(&JsonBuilder::host())
            .expect("host DTO construction is infallible")
    }

    pub(crate) fn build_json(&self, builder: &JsonBuilder<'_>) -> Result<JsonValue> {
        builder.object([
            (
                "before_file_size",
                JsonValue::Number(self.before_file_size as f64),
            ),
            (
                "before_file_size_bytes",
                JsonValue::Number(self.before_file_size as f64),
            ),
            (
                "after_file_size",
                JsonValue::Number(self.after_file_size as f64),
            ),
            (
                "after_file_size_bytes",
                JsonValue::Number(self.after_file_size as f64),
            ),
            (
                "records_removed",
                JsonValue::Number(self.records_removed as f64),
            ),
            (
                "tombstones_removed",
                JsonValue::Number(self.tombstones_removed as f64),
            ),
            (
                "segments_merged",
                JsonValue::Number(self.segments_merged as f64),
            ),
            ("index_rebuild", builder.string(&self.index_rebuild)?),
            (
                "index_rebuild_reason",
                builder.optional_str(self.index_rebuild_reason.as_deref())?,
            ),
            ("duration_ms", JsonValue::Number(self.duration_ms as f64)),
        ])
    }
}

impl RebuildIndexSummary {
    pub fn to_json(&self) -> JsonValue {
        self.build_json(&JsonBuilder::host())
            .expect("host DTO construction is infallible")
    }

    pub(crate) fn build_json(&self, builder: &JsonBuilder<'_>) -> Result<JsonValue> {
        builder.object([
            ("schema_version", builder.string(&self.schema_version)?),
            (
                "collection",
                builder.optional_str(self.collection.as_deref())?,
            ),
            ("rebuild_scope", builder.string(&self.rebuild_scope)?),
            (
                "index_suffix_status",
                builder.string(&self.index_suffix_status)?,
            ),
            (
                "index_record_count",
                JsonValue::Number(self.index_record_count as f64),
            ),
            (
                "index_extent_count",
                JsonValue::Number(self.index_extent_count as f64),
            ),
            (
                "text_segment_count",
                JsonValue::Number(self.text_segment_count as f64),
            ),
            (
                "vector_segment_count",
                JsonValue::Number(self.vector_segment_count as f64),
            ),
            ("rebuild_needed", JsonValue::Bool(self.rebuild_needed)),
            (
                "last_index_commit_id",
                self.last_index_commit_id
                    .map(|commit| JsonValue::Number(commit as f64))
                    .unwrap_or(JsonValue::Null),
            ),
            ("duration_ms", JsonValue::Number(self.duration_ms as f64)),
        ])
    }
}

impl RepairSummary {
    pub fn to_json(&self) -> JsonValue {
        self.build_json(&JsonBuilder::host())
            .expect("host DTO construction is infallible")
    }

    pub(crate) fn build_json(&self, builder: &JsonBuilder<'_>) -> Result<JsonValue> {
        builder.object([
            ("schema_version", builder.string(&self.schema_version)?),
            ("action", builder.string(&self.action)?),
            ("repaired", JsonValue::Bool(self.repaired)),
            (
                "before_file_size",
                JsonValue::Number(self.before_file_size as f64),
            ),
            (
                "before_file_size_bytes",
                JsonValue::Number(self.before_file_size as f64),
            ),
            (
                "after_file_size",
                JsonValue::Number(self.after_file_size as f64),
            ),
            (
                "after_file_size_bytes",
                JsonValue::Number(self.after_file_size as f64),
            ),
            (
                "bytes_removed",
                JsonValue::Number(self.bytes_removed as f64),
            ),
            (
                "critical_end_offset",
                JsonValue::Number(self.critical_end_offset as f64),
            ),
            (
                "issues_repaired",
                JsonValue::Number(self.issues_repaired as f64),
            ),
        ])
    }
}

impl SnapshotSummary {
    pub fn to_json(&self) -> JsonValue {
        self.build_json(&JsonBuilder::host())
            .expect("host DTO construction is infallible")
    }

    pub(crate) fn build_json(&self, builder: &JsonBuilder<'_>) -> Result<JsonValue> {
        builder.object([
            ("schema_version", builder.string(&self.schema_version)?),
            ("source_path", builder.display(self.source_path.display())?),
            ("output_path", builder.display(self.output_path.display())?),
            ("bytes_copied", JsonValue::Number(self.bytes_copied as f64)),
            (
                "last_valid_commit",
                JsonValue::Number(self.last_valid_commit as f64),
            ),
            (
                "critical_end_offset",
                JsonValue::Number(self.critical_end_offset as f64),
            ),
            (
                "index_suffix_status",
                builder.string(&self.index_suffix_status)?,
            ),
        ])
    }
}

fn ensure_snapshot_output_path(source_path: &Path, output_path: &Path) -> Result<()> {
    if output_path.file_name().is_none() {
        return Err(Error::new(
            ErrorCode::InvalidSearchParams,
            "snapshot output path must include a file name",
        ));
    }
    if output_path.exists() {
        return Err(Error::new(
            ErrorCode::InvalidSearchParams,
            "snapshot output path already exists",
        ));
    }
    let source = source_path.canonicalize()?;
    let output = normalize_new_output_path(output_path)?;
    if source == output {
        return Err(Error::new(
            ErrorCode::InvalidSearchParams,
            "snapshot output path must differ from source path",
        ));
    }
    Ok(())
}

fn normalize_new_output_path(path: &Path) -> Result<PathBuf> {
    let file_name = path.file_name().ok_or_else(|| {
        Error::new(
            ErrorCode::InvalidSearchParams,
            "snapshot output path must include a file name",
        )
    })?;
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    Ok(parent.canonicalize()?.join(file_name))
}

fn snapshot_temp_path(output_path: &Path) -> Result<PathBuf> {
    let file_name = output_path.file_name().ok_or_else(|| {
        Error::new(
            ErrorCode::InvalidSearchParams,
            "snapshot output path must include a file name",
        )
    })?;
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let mut tmp_name = file_name.to_os_string();
    tmp_name.push(format!(".{}.{}.tmp", std::process::id(), unique));
    Ok(output_path.with_file_name(tmp_name))
}

fn copy_snapshot_file(source_path: &Path, tmp_path: &Path) -> Result<u64> {
    let mut owned_temporary = false;
    let copy_result = (|| {
        let mut source = File::open(source_path)?;
        let mut output = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(tmp_path)?;
        owned_temporary = true;
        let bytes = std::io::copy(&mut source, &mut output)?;
        output.flush()?;
        output.sync_data()?;
        Ok(bytes)
    })();
    if copy_result.is_err() && owned_temporary {
        let _ = fs::remove_file(tmp_path);
    }
    copy_result
}

fn sync_parent_dir(path: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            let dir = File::open(parent)?;
            dir.sync_all()?;
        }
    }
    #[cfg(not(unix))]
    {
        let _ = path;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    mod close;
    mod collection_stats;
    mod embedding_import;
    mod import_priority;
    mod import_warnings;
    mod models;
    mod query_cost;
    mod query_output;
    mod query_preparation;
    mod query_rows;
    mod readonly_text_cache;
    mod recovery_crc;
    #[cfg(unix)]
    mod retained_file;
    mod token_counts;
    mod vector_reclaim;
    mod why_not;
    mod zero_text_writer;
    use super::*;
    use crate::json::parse;
    use crate::model::{ChunkInput, DistanceMetric, QueryEmbeddingInput, VectorIndexConfig};
    use crate::storage::scan_file;
    use std::io::{Seek, SeekFrom, Write};

    thread_local! {
        static TRANSIENT_ESTIMATES: Cell<usize> = const { Cell::new(0) };
        static TEXT_BUILDS: Cell<usize> = const { Cell::new(0) };
        static TEXT_CHECKPOINT_BUILDS: Cell<usize> = const { Cell::new(0) };
        static COVERAGE_HASH_CALLS: Cell<usize> = const { Cell::new(0) };
        static CHECK_SCANS: Cell<usize> = const { Cell::new(0) };
        static CHECK_COVERAGE_PRESSURE: Cell<Option<CheckCoveragePressure>> = const { Cell::new(None) };
    }

    #[derive(Clone, Copy)]
    enum CheckCoveragePressure {
        WorkspaceHeadroom(usize),
        CacheBytes(usize),
    }

    pub(super) fn record_coverage_hash() {
        COVERAGE_HASH_CALLS.set(COVERAGE_HASH_CALLS.get() + 1);
    }

    pub(super) fn record_check_scan() {
        CHECK_SCANS.set(CHECK_SCANS.get() + 1);
    }

    pub(super) fn hold_check_coverage_pressure(
        context: &crate::json::JsonOperationContext,
    ) -> Option<(
        crate::json::WorkspaceReservation,
        Option<crate::json::CacheReservation>,
    )> {
        let pressure = CHECK_COVERAGE_PRESSURE.take()?;
        let mut work = context.workspace_reservation();
        let cache = match pressure {
            CheckCoveragePressure::WorkspaceHeadroom(bytes) => {
                let held = context.available_workspace_bytes() - bytes;
                work.grow_to(held, held, "operation_scratch").unwrap();
                None
            }
            CheckCoveragePressure::CacheBytes(bytes) => Some(
                context
                    .cache_reservation(bytes as u128, usize::MAX)
                    .unwrap(),
            ),
        };
        Some((work, cache))
    }

    fn with_check_coverage_pressure<T>(
        pressure: CheckCoveragePressure,
        run: impl FnOnce() -> T,
    ) -> T {
        struct Reset;
        impl Drop for Reset {
            fn drop(&mut self) {
                CHECK_COVERAGE_PRESSURE.set(None);
            }
        }
        assert!(CHECK_COVERAGE_PRESSURE.replace(Some(pressure)).is_none());
        let _reset = Reset;
        run()
    }

    pub(super) fn record_transient_estimate() {
        TRANSIENT_ESTIMATES.with(|count| count.set(count.get() + 1));
    }

    pub(super) fn record_text_build() {
        TEXT_BUILDS.with(|count| count.set(count.get() + 1));
    }

    pub(super) fn record_text_checkpoint_build() {
        TEXT_CHECKPOINT_BUILDS.with(|count| count.set(count.get() + 1));
    }

    #[test]
    fn index_suffix_state_matrix_preserves_wire_values_and_commit_validity() {
        let path = temp_path("index-state-matrix");
        let db = Database::create(&path, DatabaseConfig::default()).unwrap();
        db.create_collection(CollectionConfig {
            tokenizer: Default::default(),
            name: "notes".into(),
            embedding_dim: 2,
            distance_metric: DistanceMetric::Cosine,
            embedding_model: None,
            vectors_normalized: false,
            vector_index: VectorIndexConfig::Exact,
        })
        .unwrap();
        let baseline = scan_file(&path).unwrap();
        assert!(baseline.findings.is_empty());
        let mut inner = db.lock().unwrap();
        let suffix_statuses = [
            None,
            Some("persistent_checkpoint"),
            Some("metadata_record_only"),
            Some("rebuild_needed"),
            Some("future_checkpoint"),
        ];
        let findings = [
            None,
            Some(("index", "checkpoint_payload_checksum_mismatch")),
            Some(("index", "index_segment_checksum_mismatch")),
            Some(("critical", "record_checksum_mismatch")),
            Some(("critical", "checkpoint_payload_checksum_mismatch")),
        ];
        let mut cases = 0;
        for status in suffix_statuses {
            for finding in findings {
                for coverage in 0..4 {
                    cases += 1;
                    let mut scan = baseline.clone();
                    if let Some(status) = status {
                        let suffix = scan.index_suffix.as_mut().unwrap();
                        suffix.status = status.into();
                        match coverage {
                            0 => {}
                            1 => suffix.coverage_hash ^= 1,
                            2 => suffix.text_segment_count = 0,
                            3 => suffix.vector_segment_count = 0,
                            _ => unreachable!(),
                        }
                    } else {
                        scan.index_suffix = None;
                    }
                    if let Some((stream, code)) = finding {
                        scan.findings.push(CheckIssue {
                            severity: "warning".into(),
                            code: code.into(),
                            message: "test finding".into(),
                            stream: stream.into(),
                            visibility_impact: "rebuild_index_only".into(),
                            offset: None,
                            extent_id: None,
                            record_type: None,
                            batch_id: None,
                            repair_available: false,
                            suggested_command: None,
                        });
                    }
                    let invalid_coverage = status.is_some()
                        && (coverage == 1
                            || (coverage > 1 && status != Some("metadata_record_only")));
                    let (expected, wire) = if invalid_coverage
                        || finding == Some(("index", "index_segment_checksum_mismatch"))
                    {
                        (IndexSuffixState::RebuildNeeded, "rebuild_needed")
                    } else {
                        match status {
                            None => (IndexSuffixState::Missing, "missing"),
                            Some("persistent_checkpoint") => (
                                IndexSuffixState::PersistentCheckpoint,
                                "persistent_checkpoint",
                            ),
                            Some("metadata_record_only") => {
                                (IndexSuffixState::MetadataRecordOnly, "metadata_record_only")
                            }
                            _ => (IndexSuffixState::RebuildNeeded, "rebuild_needed"),
                        }
                    };
                    let actual =
                        index_suffix_status_from_scan(&scan, &inner.catalog, &db.json_context())
                            .unwrap();
                    assert_eq!(actual, expected, "case {cases}");
                    assert_eq!(actual.as_str(), wire, "case {cases}");
                    assert_eq!(
                        index_commit_sequence_no_from_scan(&scan, actual),
                        (expected == IndexSuffixState::PersistentCheckpoint)
                            .then_some(baseline.last_valid_commit_sequence_no),
                        "case {cases}"
                    );
                    assert_eq!(scan.last_valid_commit, baseline.last_valid_commit);
                    inner.index_suffix_status = actual;
                    let stats = inner.index_stats();
                    let segment_count =
                        usize::from(actual == IndexSuffixState::PersistentCheckpoint);
                    assert_eq!(stats.text_segments, segment_count, "case {cases}");
                    assert_eq!(stats.vector_segments, segment_count, "case {cases}");
                }
            }
        }
        assert_eq!(cases, 100);
        drop(inner);
        db.close().unwrap();
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn diagnostic_fusion_truncation_ignores_counterfactual_full_scan_rank() {
        let builder = JsonBuilder::host();
        let limits = CandidateLimits {
            bm25: 50,
            vector: 50,
            fusion: 50,
        };
        let weights = SearchWeights::new(0.5, 0.5, 0.0).unwrap();
        for rank in 1..=100 {
            let ranks = WhyNotRanks {
                vector_rank: Some(1),
                fusion_rank: Some(101 - rank),
                fusion_retrieval_rank: Some(rank),
                ..Default::default()
            };
            for (returned, allowed) in [(None, true), (Some(1), true), (None, false)] {
                let analysis =
                    why_not_limit_analysis(&ranks, limits, weights, returned, allowed, &builder)
                        .unwrap();
                let limited = rank > 50 && returned.is_none() && allowed;
                assert_eq!(
                    analysis
                        .get("missed_due_to_candidate_limit")
                        .and_then(JsonValue::as_bool),
                    Some(false)
                );
                assert_eq!(
                    analysis.get("limiting_stage").and_then(JsonValue::as_str),
                    limited.then_some("fusion")
                );
                assert_eq!(
                    analysis
                        .get("limiting_parameter")
                        .and_then(JsonValue::as_str),
                    limited.then_some("fusion_candidates")
                );
                assert_eq!(
                    analysis
                        .get("effective_candidate_limit")
                        .and_then(JsonValue::as_f64),
                    limited.then_some(50.0)
                );
            }
            let public = ranks.build_json(&builder).unwrap();
            assert_eq!(public.as_object().unwrap().len(), 3);
            assert_eq!(
                public.get("fusion_rank").and_then(JsonValue::as_f64),
                Some((101 - rank) as f64)
            );
        }
    }

    #[test]
    fn diagnostic_postfilter_rank_does_not_hide_retrieval_limits() {
        let builder = JsonBuilder::host();
        let limits = CandidateLimits {
            bm25: 50,
            vector: 50,
            fusion: 50,
        };
        let weights = SearchWeights::new(0.5, 0.5, 0.0).unwrap();
        for rank in 1..=100 {
            let ranks = WhyNotRanks {
                bm25_rank: Some(1),
                bm25_retrieval_rank: Some(rank),
                vector_rank: None,
                fusion_rank: Some(1),
                fusion_retrieval_rank: None,
                candidate_limit_improves: true,
                ..Default::default()
            };
            for (returned, filter_matched) in [(None, true), (Some(1), true), (None, false)] {
                let analysis = why_not_limit_analysis(
                    &ranks,
                    limits,
                    weights,
                    returned,
                    filter_matched,
                    &builder,
                )
                .unwrap();
                let limited = rank > 50 && returned.is_none() && filter_matched;
                assert_eq!(
                    analysis
                        .get("missed_due_to_candidate_limit")
                        .and_then(JsonValue::as_bool),
                    Some(limited)
                );
                assert_eq!(
                    analysis
                        .get("limiting_parameter")
                        .and_then(JsonValue::as_str),
                    limited.then_some("bm25_candidates")
                );
            }
            let public = ranks.build_json(&builder).unwrap();
            assert_eq!(public.as_object().unwrap().len(), 3);
            assert_eq!(
                public.get("bm25_rank").and_then(JsonValue::as_f64),
                Some(1.0)
            );
            assert!(public.get("bm25_retrieval_rank").is_none());
        }
    }

    fn query_preflight_fixture(name: &str, text: &str) -> PathBuf {
        let path = temp_path(name);
        let db = Database::create(
            &path,
            DatabaseConfig {
                durability: Durability::Fast,
                ..DatabaseConfig::default()
            },
        )
        .unwrap();
        for (name, count) in [("first", 3), ("second", 5), ("empty", 0)] {
            let collection = db
                .create_collection(CollectionConfig {
                    name: name.into(),
                    embedding_dim: 2,
                    tokenizer: Default::default(),
                    distance_metric: DistanceMetric::Cosine,
                    embedding_model: Some("m".into()),
                    vectors_normalized: false,
                    vector_index: VectorIndexConfig::Exact,
                })
                .unwrap();
            if count == 0 {
                continue;
            }
            collection
                .insert_document(DocumentInput {
                    external_id: "doc".into(),
                    title: None,
                    source_uri: None,
                    metadata: JsonValue::object(),
                    chunks: (0..count)
                        .map(|chunk| ChunkInput {
                            chunk_external_id: format!("chunk-{chunk}"),
                            text: text.into(),
                            ordinal: chunk,
                            metadata: parse(r#"{"lang":"rust"}"#).unwrap(),
                            source_span: None,
                        })
                        .collect(),
                })
                .unwrap();
            let deleted = collection.get_chunk_by_external_id("chunk-0").unwrap();
            collection.delete_chunk(&deleted.chunk_id).unwrap();
            collection
                .put_embeddings(
                    (1..count)
                        .map(|chunk| EmbeddingInput {
                            chunk_external_id: format!("chunk-{chunk}"),
                            model_name: "m".into(),
                            dim: 2,
                            vector: vec![1.0, 0.0],
                        })
                        .collect(),
                )
                .unwrap();
        }
        drop(db);
        path
    }

    #[test]
    fn retained_results_release_source_text_and_transfer_only_live_result_ownership() {
        let text = "text \u{4e2d}\n".repeat(20_000);
        let path = query_preflight_fixture("retained-result-strings", &text);
        let db = Database::open(&path).unwrap();
        let before = fs::read(&path).unwrap();
        let limit = db.json_context().available_workspace_bytes();
        let collection = db.collection("first").unwrap();
        for seed in 0..100usize {
            let context = db.json_context();
            let query = || {
                let mut query = collection
                    .search()
                    .query_embedding(QueryEmbeddingInput {
                        model_name: Some("m".into()),
                        dim: 2,
                        vector: vec![1.0, 0.0],
                    })
                    .top_k(1 + seed % 2)
                    .weights(SearchWeights::new(0.4, 0.4, 0.2).unwrap());
                if seed % 3 != 0 {
                    query = query.text(" text \u{4e2d} ");
                }
                if seed % 4 != 0 {
                    query = query.metadata_boost(MetadataBoostConfig::parse(parse(r#"{"schema_version":"0.1","combine":"sum","max_boost":1,"rules":[{"when":{"chunk.lang":"rust"},"boost":0.1}]}"#).unwrap()).unwrap());
                }
                query.execute_retained(Some(context.clone()))
            };
            let mut held = context.workspace_reservation();
            held.grow_to(limit - 128 * 1024, limit - 128 * 1024, "operation_scratch")
                .unwrap();
            assert_eq!(query().err().unwrap().code(), ErrorCode::ResourceExhausted);
            held.grow_to(limit, limit, "operation_scratch").unwrap();
            drop(held);

            let (result, mut work) = query().unwrap();
            assert_eq!(result.results.len(), 1 + seed % 2);
            let bytes = work.json_bytes()
                + result.wrapper_workspace_bytes()
                + result.results.capacity() * std::mem::size_of::<SearchResult>()
                + result
                    .results
                    .iter()
                    .map(|hit| {
                        hit.chunk_id.capacity()
                            + hit.chunk_external_id.capacity()
                            + hit.document_id.capacity()
                            + hit.document_external_id.capacity()
                            + hit.snippet.capacity()
                    })
                    .sum::<usize>();
            assert!(
                bytes < text.len(),
                "source text was retained in the result budget"
            );
            for hit in &result.results {
                assert_eq!(hit.snippet, text.chars().take(240).collect::<String>());
                assert_eq!(hit.snippet.capacity(), hit.snippet.len());
            }
            let expected_value = result.to_json();
            let output_bytes = expected_value
                .reserve_retained_in(crate::json::JsonInputKind::StoredJson, &context)
                .unwrap()
                .reserved_bytes();
            let expected = expected_value.to_compact_string();
            drop(expected_value);
            let mut held = context.workspace_reservation();
            held.grow_to(limit - bytes, limit - bytes, "operation_scratch")
                .unwrap();
            assert!(held
                .grow_to(limit - bytes + 1, limit - bytes + 1, "operation_scratch")
                .is_err());
            let id_pointer = result.results[0].chunk_id.as_ptr();
            let snippet_pointer = result.results[0].snippet.as_ptr();
            let wrapper_pointers = [
                ("schema_version", Some(result.schema_version.as_ptr())),
                ("collection", Some(result.collection.as_ptr())),
                ("query", result.query.as_ref().map(|value| value.as_ptr())),
            ];
            let context_pointers = [
                (
                    "weights_source",
                    Some(result.query_context.weights.source.as_ptr()),
                ),
                (
                    "renormalization_reason",
                    result
                        .query_context
                        .weights
                        .renormalization_reason
                        .as_ref()
                        .map(|value| value.as_ptr()),
                ),
                (
                    "boost_config_hash",
                    result
                        .query_context
                        .boost_config_hash
                        .as_ref()
                        .map(|value| value.as_ptr()),
                ),
                (
                    "index_source",
                    Some(result.query_context.index_source.as_ptr()),
                ),
                (
                    "index_rebuild_reason",
                    result
                        .query_context
                        .index_rebuild_reason
                        .as_ref()
                        .map(|value| value.as_ptr()),
                ),
            ];
            let extra = if seed % 2 == 0 { output_bytes } else { 0 };
            held.retain(limit - bytes - extra);
            let value = result.into_json_retained(&mut work);
            if extra == 0 {
                assert_eq!(value.err().unwrap().code(), ErrorCode::RecoveryError);
            } else {
                let value = value.unwrap();
                held.retain(0);
                let measured = value
                    .reserve_retained_in(crate::json::JsonInputKind::StoredJson, &context)
                    .unwrap();
                assert_eq!(work.json_bytes(), measured.reserved_bytes());
                drop(measured);
                assert_eq!(value.to_compact_string(), expected);
                for (field, pointer) in wrapper_pointers {
                    assert_eq!(value.get(field).unwrap().as_str().map(str::as_ptr), pointer);
                }
                let query_context = value.get("query_context").unwrap();
                for (field, pointer) in context_pointers {
                    assert_eq!(
                        query_context.get(field).unwrap().as_str().map(str::as_ptr),
                        pointer
                    );
                }
                let first = &value.get("results").unwrap().as_array().unwrap()[0];
                assert_eq!(
                    first.get("chunk_id").unwrap().as_str().unwrap().as_ptr(),
                    id_pointer
                );
                assert_eq!(
                    first.get("snippet").unwrap().as_str().unwrap().as_ptr(),
                    snippet_pointer
                );
                drop(value);
            }
            drop(work);
            held.grow_to(limit, limit, "operation_scratch").unwrap();
        }
        assert_eq!(fs::read(&path).unwrap(), before);
        drop((collection, db));
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn diagnostics_keep_context_credit_after_discarding_rows_and_move_it_into_json() {
        use crate::json::JsonInputKind;

        let path = query_preflight_fixture("retained-diagnosis-context", "text \u{4e2d}");
        let before = fs::read(&path).unwrap();
        for persistent in [true, false] {
            let db = Database::open(&path).unwrap();
            if !persistent {
                db.lock().unwrap().index_suffix_status = IndexSuffixState::Missing;
            }
            let limit = db.json_context().available_workspace_bytes();
            for seed in 0..100 {
                let context = db.json_context();
                let expected_chunk = if seed % 3 == 0 { "chunk-0" } else { "chunk-1" };
                let (diagnosis, mut work) = db
                    .why_not_debug_with_metadata_boost_in(
                        if seed % 4 == 0 { "empty" } else { "first" },
                        expected_chunk,
                        Some(" text \u{4e2d} ".into()),
                        None,
                        JsonValue::object(),
                        Some(SearchWeights::new(0.4, 0.4, 0.2).unwrap()),
                        false,
                        None,
                        1 + seed % 2,
                        None,
                        Some(context.clone()),
                    )
                    .unwrap();
                assert_eq!(diagnosis.query_text.as_deref(), Some("text \u{4e2d}"));
                assert_eq!(
                    diagnosis.query_context.index_source,
                    if persistent {
                        "persistent_checkpoint"
                    } else {
                        "transient_rebuild"
                    }
                );
                assert_eq!(
                    diagnosis.query_context.index_rebuild_reason.as_deref(),
                    (!persistent).then_some("in_memory_index_from_committed_snapshot")
                );
                assert_eq!(
                    diagnosis
                        .query_context
                        .weights
                        .renormalization_reason
                        .as_deref(),
                    Some("vector branch unavailable; metadata boost unavailable")
                );
                let wrapper_bytes = diagnosis.schema_version.capacity()
                    + diagnosis.diagnosis_scope.capacity()
                    + diagnosis.query_text.as_ref().unwrap().capacity()
                    + diagnosis.query_context.workspace_bytes();
                assert_eq!(work.parts_mut().1.unwrap().reserved_bytes(), wrapper_bytes);
                let measured = diagnosis
                    .result
                    .reserve_retained_in(JsonInputKind::StoredJson, &context)
                    .unwrap();
                assert_eq!(work.json_bytes(), measured.reserved_bytes());
                drop(measured);
                let occupied = work.json_bytes() + wrapper_bytes;
                let expected = diagnosis.to_json();
                let output_bytes = expected
                    .reserve_retained_in(JsonInputKind::StoredJson, &context)
                    .unwrap()
                    .reserved_bytes();
                let expected = expected.to_compact_string();
                let pointer = diagnosis.query_text.as_ref().unwrap().as_ptr();
                let source_pointer = diagnosis.query_context.index_source.as_ptr();
                let mut held = context.workspace_reservation();
                held.grow_to(limit - occupied, limit - occupied, "operation_scratch")
                    .unwrap();
                assert_eq!(
                    held.grow_to(
                        limit - occupied + 1,
                        limit - occupied + 1,
                        "operation_scratch"
                    )
                    .unwrap_err()
                    .code(),
                    ErrorCode::ResourceExhausted
                );
                let extra = if seed % 2 == 0 { output_bytes } else { 0 };
                held.retain(limit - occupied - extra);
                let actual = diagnosis.into_json_retained(&mut work);
                if extra == 0 {
                    let error = actual.unwrap_err();
                    assert_eq!(error.code(), ErrorCode::RecoveryError);
                    assert!(error
                        .detail()
                        .iter()
                        .any(|(key, value)| key == "json_input_kind" && value == "stored_json"));
                } else {
                    let actual = actual.unwrap();
                    assert_eq!(actual.to_compact_string(), expected);
                    assert_eq!(work.parts_mut().1.unwrap().reserved_bytes(), 0);
                    let query = actual.get("query_context").unwrap();
                    assert_eq!(
                        query.get("text").unwrap().as_str().unwrap().as_ptr(),
                        pointer
                    );
                    assert_eq!(
                        query
                            .get("index_source")
                            .unwrap()
                            .as_str()
                            .unwrap()
                            .as_ptr(),
                        source_pointer
                    );
                    held.retain(0);
                    let measured = actual
                        .reserve_retained_in(JsonInputKind::StoredJson, &context)
                        .unwrap();
                    assert_eq!(work.json_bytes(), measured.reserved_bytes());
                }
                drop(work);
                held.grow_to(limit, limit, "operation_scratch").unwrap();
            }
            assert_eq!(fs::read(&path).unwrap(), before);
        }
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn persistent_queries_skip_transient_estimates_and_preserve_collection_counts() {
        let path = query_preflight_fixture("persistent-preflight", "query text");
        let original = fs::read(&path).unwrap();
        let db = Database::open_with_options(
            &path,
            DatabaseOpenOptions {
                read_only: true,
                cache_budget_mb: Some(1),
                ..DatabaseOpenOptions::default()
            },
        )
        .unwrap();
        let before = TRANSIENT_ESTIMATES.with(Cell::get);
        for seed in 0..100 {
            for (collection, expected) in [("first", 2), ("second", 4), ("empty", 0)] {
                let top_k = 1 + seed % 7;
                let mut query = db
                    .collection(collection)
                    .unwrap()
                    .search()
                    .top_k(top_k)
                    .resource_profile(
                        [
                            ResourceProfile::Balanced,
                            ResourceProfile::Constrained,
                            ResourceProfile::Expanded,
                        ][(seed / 3) % 3],
                    );
                if seed % 3 != 1 {
                    query = query.text("query");
                }
                if seed % 3 != 0 {
                    query = query.query_embedding(QueryEmbeddingInput {
                        model_name: Some("m".into()),
                        dim: 2,
                        vector: vec![1.0, 0.0],
                    });
                }
                if seed % 2 != 0 {
                    query = query.filter(parse(r#"{"chunk.lang":"rust"}"#).unwrap());
                }
                let results = query.execute().unwrap();
                assert_eq!(results.query_context.index_source, "persistent_checkpoint");
                assert_eq!(results.query_context.transient_rebuild.chunks_scanned, None);
                assert_eq!(
                    results.query_context.execution_stats.vector_scan_count,
                    if seed % 3 != 0 { expected } else { 0 }
                );
                assert_eq!(results.results.len(), expected.min(top_k));
            }
        }
        assert_eq!(TRANSIENT_ESTIMATES.with(Cell::get), before);
        assert_eq!(fs::read(&path).unwrap(), original);
        drop(db);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn lazy_checkpoint_failure_still_checks_transient_budget_once_before_rebuild() {
        for budget in [1, 64] {
            let path = query_preflight_fixture(
                &format!("lazy-preflight-{budget}"),
                &"query ".repeat(4096),
            );
            let db = Database::open_with_options(
                &path,
                DatabaseOpenOptions {
                    read_only: true,
                    cache_budget_mb: Some(budget),
                    ..DatabaseOpenOptions::default()
                },
            )
            .unwrap();
            let critical_end = {
                let inner = db.lock().unwrap();
                assert_eq!(
                    inner.index_suffix_status,
                    IndexSuffixState::PersistentCheckpoint
                );
                assert!(inner.lazy_vector_source.is_some());
                assert!(inner.lazy_text);
                assert!(inner.text_index.chunk_lengths.is_empty());
                inner.storage.critical_end_offset()
            };
            std::fs::OpenOptions::new()
                .write(true)
                .open(&path)
                .unwrap()
                .set_len(critical_end)
                .unwrap();
            let original = fs::read(&path).unwrap();
            for _ in 0..2 {
                let before = TRANSIENT_ESTIMATES.with(Cell::get);
                let result = db
                    .collection("first")
                    .unwrap()
                    .search()
                    .text("query")
                    .query_embedding(QueryEmbeddingInput {
                        model_name: Some("m".into()),
                        dim: 2,
                        vector: vec![1.0, 0.0],
                    })
                    .execute();
                assert_eq!(TRANSIENT_ESTIMATES.with(Cell::get), before + 1);
                if budget == 1 {
                    let err = result.unwrap_err();
                    assert_eq!(err.code(), ErrorCode::IndexCorruption);
                    assert!(err
                        .detail()
                        .iter()
                        .any(|(key, value)| key == "index_unavailable_reason"
                            && value == "transient_rebuild_budget_exceeded"));
                } else {
                    let results = result.unwrap();
                    assert_eq!(results.query_context.index_source, "transient_rebuild");
                    assert_eq!(
                        results.query_context.transient_rebuild.chunks_scanned,
                        Some(2)
                    );
                    assert_eq!(results.results.len(), 2);
                }
                assert!(db.lock().unwrap().text_index.chunk_lengths.is_empty());
            }
            assert_eq!(fs::read(&path).unwrap(), original);
            drop(db);
            fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn failed_checkpoint_text_cache_is_released_before_transient_budget_rejection() {
        let path = query_preflight_fixture("failed-checkpoint-cache", &"query ".repeat(4096));
        let db = Database::open_with_options(
            &path,
            DatabaseOpenOptions {
                cache_budget_mb: Some(1),
                ..DatabaseOpenOptions::default()
            },
        )
        .unwrap();
        fail_next_index_checkpoint_refresh(&db);
        assert_eq!(
            db.rebuild_index(None).unwrap_err().code(),
            ErrorCode::IoError
        );
        {
            let inner = db.lock().unwrap();
            assert_eq!(inner.index_suffix_status, IndexSuffixState::RebuildNeeded);
            assert!(!inner.text_index.chunk_lengths.is_empty());
        }
        let before = fs::read(&path).unwrap();
        let error = db
            .collection("first")
            .unwrap()
            .search()
            .text("query")
            .execute()
            .unwrap_err();
        assert_eq!(error.code(), ErrorCode::IndexCorruption);
        assert_eq!(
            error_detail_value(&error, "index_unavailable_reason"),
            Some("transient_rebuild_budget_exceeded")
        );
        {
            let inner = db.lock().unwrap();
            assert!(inner.text_index.chunk_lengths.is_empty());
            assert_eq!(inner.text_index.term_count(), 0);
        }
        assert_eq!(fs::read(&path).unwrap(), before);
        drop(db);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn replayed_updates_keep_one_chunk_membership_before_and_after_compact() {
        let path = temp_path("replayed-current-membership");
        let db = Database::create(&path, DatabaseConfig::default()).unwrap();
        let collection = db
            .create_collection(CollectionConfig {
                name: "versions".into(),
                embedding_dim: 2,
                tokenizer: Default::default(),
                distance_metric: DistanceMetric::Cosine,
                embedding_model: None,
                vectors_normalized: false,
                vector_index: VectorIndexConfig::Exact,
            })
            .unwrap();
        let collection_id = collection.collection_id().to_owned();
        for number in 1..=2 {
            collection
                .insert_document(DocumentInput {
                    external_id: format!("doc-{number}"),
                    title: None,
                    source_uri: None,
                    metadata: JsonValue::object(),
                    chunks: if number == 1 {
                        vec![ChunkInput {
                            chunk_external_id: "initial-chunk".into(),
                            text: "original".into(),
                            ordinal: 0,
                            metadata: JsonValue::object(),
                            source_span: None,
                        }]
                    } else {
                        Vec::new()
                    },
                })
                .unwrap();
        }
        let first = collection.get_document_by_external_id("doc-1").unwrap();
        let second = collection.get_document_by_external_id("doc-2").unwrap();
        let mut latest = collection
            .get_chunk_by_external_id("initial-chunk")
            .unwrap();
        let mut document = first.clone();
        drop((collection, db));
        let (mut storage, _) = Storage::open(&path).unwrap();
        for version in 0..100 {
            document.external_id = format!("document-version-{version}");
            latest.document_id = if version % 3 == 0 {
                &second.document_id
            } else {
                &first.document_id
            }
            .clone();
            latest.chunk_external_id = format!("chunk-version-{version}");
            latest.text = format!("version {version} verified");
            latest.text_hash = chunk_text_hash(&latest.text);
            latest.token_count = token_count_hint(&latest.text);
            storage
                .append_batch(
                    &[
                        RecordPayload::DocumentPut(document.clone()),
                        RecordPayload::ChunkPut(latest.clone()),
                    ],
                    Durability::Fast,
                )
                .unwrap();
        }
        drop(storage);
        let verify = |db: &Database| {
            assert_eq!(db.stats().unwrap().chunk_count, 1);
            let collection = db.collection("versions").unwrap();
            assert_eq!(
                collection
                    .get_document_by_external_id("doc-1")
                    .unwrap_err()
                    .code(),
                ErrorCode::DocumentNotFound
            );
            assert_eq!(
                collection
                    .get_chunk_by_external_id("initial-chunk")
                    .unwrap_err()
                    .code(),
                ErrorCode::ChunkNotFound
            );
            assert_eq!(
                collection
                    .get_document_by_external_id(&document.external_id)
                    .unwrap()
                    .document_id,
                document.document_id
            );
            assert_eq!(
                collection
                    .get_chunk_by_external_id(&latest.chunk_external_id)
                    .unwrap()
                    .text,
                latest.text
            );
            for id in [&first.document_id, &second.document_id] {
                let chunks = db.list_chunks_in_collection(&collection_id, id).unwrap();
                if id == &latest.document_id {
                    assert_eq!(chunks.len(), 1);
                    assert_eq!(chunks[0].chunk_id, latest.chunk_id);
                    assert_eq!(chunks[0].text, latest.text);
                } else {
                    assert!(chunks.is_empty());
                }
            }
        };
        let old_reader = Database::open_read_only(&path).unwrap();
        verify(&old_reader);
        let writer = Database::open(&path).unwrap();
        verify(&writer);
        assert_eq!(
            writer
                .compact(CompactOptions::default())
                .unwrap()
                .records_removed,
            200
        );
        verify(&writer);
        verify(&old_reader);
        drop(writer);
        let reopened = Database::open_read_only(&path).unwrap();
        verify(&reopened);
        drop((reopened, old_reader));
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn streamed_open_check_and_snapshot_preserve_catalog_history_and_record_references() {
        let path = temp_path("streamed-catalog-history");
        let snapshot_path = temp_path("streamed-catalog-history-snapshot");
        let db = Database::create(&path, DatabaseConfig::default()).unwrap();
        let collection = db
            .create_collection(CollectionConfig {
                name: "history".into(),
                embedding_dim: 2,
                tokenizer: Default::default(),
                distance_metric: DistanceMetric::Cosine,
                embedding_model: None,
                vectors_normalized: false,
                vector_index: VectorIndexConfig::Exact,
            })
            .unwrap();
        let collection_id = collection.collection_id().to_owned();
        drop((collection, db));
        let (mut storage, _) = Storage::open(&path).unwrap();
        for seed in 0..100 {
            let mut records = vec![RecordPayload::DocumentPut(Document {
                document_id: format!("doc_{}", seed + 1),
                collection_id: collection_id.clone(),
                external_id: format!("doc-{seed}"),
                title: Some(format!("title-{seed}")),
                source_uri: None,
                metadata: object_from_pairs([("epoch", JsonValue::Number(seed as f64))]),
                created_at: "1970-01-01T00:00:00Z".into(),
                updated_at: "1970-01-01T00:00:00Z".into(),
            })];
            if seed % 3 == 2 {
                records.push(RecordPayload::Tombstone(Tombstone {
                    target_type: "document".into(),
                    target_id: format!("doc_{}", seed),
                    deleted_at: "1970-01-01T00:00:00Z".into(),
                    reason: Some("history-test".into()),
                }));
            }
            storage.append_batch(&records, Durability::Fast).unwrap();
        }
        drop(storage);
        let scan = scan_file(&path).unwrap();
        let mut tombstones = 0;
        for record in &scan.records {
            if let RecordPayload::DocumentPut(document) = &record.payload {
                let seed = document
                    .external_id
                    .strip_prefix("doc-")
                    .unwrap()
                    .parse::<u64>()
                    .unwrap();
                assert_eq!(
                    document.metadata.get("epoch").and_then(JsonValue::as_f64),
                    Some(seed as f64)
                );
            }
            if let RecordPayload::Tombstone(tombstone) = &record.payload {
                tombstones += 1;
                assert_eq!(tombstone.reason.as_deref(), Some("history-test"));
                assert_eq!(tombstone.deleted_at, "1970-01-01T00:00:00Z");
            }
        }
        assert_eq!(tombstones, 33);
        let expected =
            Catalog::from_committed_records(scan.records, scan.last_valid_commit).unwrap();
        assert_eq!(expected.tombstone_count(), 33);
        for document in expected.documents.values() {
            if expected.deleted_documents.contains(&document.document_id) {
                assert_eq!(document.metadata, JsonValue::object());
                assert!(!expected.record_refs.contains_key(&document.document_id));
            } else {
                assert!(document.metadata.get("epoch").is_some());
                assert!(expected.record_refs.contains_key(&document.document_id));
            }
        }
        let signature = expected.state_signature();
        let original = fs::read(&path).unwrap();
        let db = Database::open_read_only(&path).unwrap();
        assert_eq!(db.lock().unwrap().catalog.state_signature(), signature);
        assert_eq!(
            db.stats().unwrap().document_count,
            expected.live_documents().count()
        );
        let check = db
            .check(CheckOptions {
                verify_indexes: true,
            })
            .unwrap();
        assert_eq!(check.live_documents, expected.live_documents().count());
        assert_eq!(check.tombstones, expected.tombstone_count());
        assert_eq!(
            db.collection("history")
                .unwrap()
                .stats()
                .unwrap()
                .tombstone_count,
            33
        );
        assert_eq!(check.last_valid_commit, expected.last_commit_id);
        drop(db);
        let snapshot = Database::snapshot(&path, &snapshot_path).unwrap();
        assert_eq!(snapshot.last_valid_commit, expected.last_commit_id);
        let copy = Database::open_read_only(&snapshot_path).unwrap();
        assert_eq!(copy.lock().unwrap().catalog.state_signature(), signature);
        assert_eq!(fs::read(&path).unwrap(), original);
        assert_eq!(fs::read(&snapshot_path).unwrap(), original);
        drop(copy);
        let writer = Database::open(&path).unwrap();
        let summary = writer.compact(CompactOptions::default()).unwrap();
        assert_eq!(summary.tombstones_removed, 33);
        drop(writer);
        let compacted = Database::open_read_only(&path).unwrap();
        let stats = compacted.stats().unwrap();
        assert_eq!(stats.tombstone_count, 0);
        assert_eq!(stats.document_count, 67);
        assert_eq!(
            compacted.check(CheckOptions::default()).unwrap().tombstones,
            0
        );
        drop(compacted);
        fs::remove_file(path).unwrap();
        fs::remove_file(snapshot_path).unwrap();
    }

    #[test]
    fn deleted_payload_release_preserves_primary_history_old_readers_and_live_exports() {
        for durability in [Durability::Fast, Durability::Commit] {
            let path = temp_path(&format!("deleted-metadata-{durability:?}"));
            let output = path.with_extension("export");
            let db = Database::create(
                &path,
                DatabaseConfig {
                    durability,
                    ..DatabaseConfig::default()
                },
            )
            .unwrap();
            let collection = db
                .create_collection(CollectionConfig {
                    name: "notes".into(),
                    embedding_dim: 2,
                    tokenizer: Default::default(),
                    distance_metric: DistanceMetric::Cosine,
                    embedding_model: None,
                    vectors_normalized: false,
                    vector_index: VectorIndexConfig::Exact,
                })
                .unwrap();
            let metadata =
                object_from_pairs([("payload", JsonValue::String("x".repeat(64 * 1024)))]);
            collection
                .insert_document(DocumentInput {
                    external_id: "doc".into(),
                    title: Some("retained title ".repeat(4096)),
                    source_uri: Some(format!("source/{}", "path".repeat(4096))),
                    metadata: metadata.clone(),
                    chunks: (0..2)
                        .map(|ordinal| ChunkInput {
                            chunk_external_id: format!("chunk-{ordinal}"),
                            text: "searchable content".into(),
                            ordinal,
                            metadata: metadata.clone(),
                            source_span: None,
                        })
                        .collect(),
                })
                .unwrap();
            let document = collection.get_document_by_external_id("doc").unwrap();
            let chunk = collection.get_chunk_by_external_id("chunk-0").unwrap();
            let old = Database::open_read_only(&path).unwrap();
            let old_collection = old.collection("notes").unwrap();
            collection.delete_chunk(&chunk.chunk_id).unwrap();
            {
                let inner = db.lock().unwrap();
                assert_eq!(
                    inner.catalog.chunks[&chunk.chunk_id].metadata,
                    JsonValue::object()
                );
                assert!(!inner.catalog.record_refs.contains_key(&chunk.chunk_id));
                assert_eq!(
                    inner.catalog.chunks[&chunk.chunk_id].text_hash.capacity(),
                    0
                );
                assert_eq!(
                    inner.catalog.chunks[&chunk.chunk_id].created_at.capacity(),
                    0
                );
                assert_eq!(
                    inner.catalog.documents[&document.document_id].metadata,
                    metadata
                );
            }
            assert_eq!(
                old_collection.get_chunk(&chunk.chunk_id).unwrap().metadata,
                metadata
            );
            assert_eq!(
                collection.get_chunk(&chunk.chunk_id).unwrap_err().code(),
                ErrorCode::ChunkNotFound
            );
            let diagnosis = db
                .why_not_debug(
                    "notes",
                    "chunk-0",
                    Some("searchable".into()),
                    None,
                    parse(r#"{"chunk.payload":"absent"}"#).unwrap(),
                    None,
                    false,
                    10,
                    None,
                )
                .unwrap();
            assert!(diagnosis
                .result
                .to_compact_string()
                .contains("expected chunk is deleted"));
            db.export_collection_jsonl("notes", &output).unwrap();
            let exported = fs::read_to_string(output.join("chunks.jsonl")).unwrap();
            assert_eq!(exported.lines().count(), 1);
            assert_eq!(
                parse(exported.trim()).unwrap().get("metadata"),
                Some(&metadata)
            );
            for record in scan_file(&path).unwrap().records {
                if let RecordPayload::DocumentPut(stored) = &record.payload {
                    assert_eq!(stored.title, document.title);
                    assert_eq!(stored.source_uri, document.source_uri);
                    assert_eq!(stored.created_at, document.created_at);
                }
                if let RecordPayload::ChunkPut(stored) = record.payload {
                    assert_eq!(stored.metadata, metadata);
                    assert_eq!(stored.text_hash, chunk.text_hash);
                    assert_eq!(stored.created_at, chunk.created_at);
                }
            }
            collection.delete_document(&document.document_id).unwrap();
            {
                let inner = db.lock().unwrap();
                let deleted = &inner.catalog.documents[&document.document_id];
                assert_eq!(deleted.metadata, JsonValue::object());
                assert!(deleted.title.is_none());
                assert!(deleted.source_uri.is_none());
                assert_eq!(deleted.created_at.capacity(), 0);
                assert_eq!(deleted.updated_at.capacity(), 0);
                assert!(!inner
                    .catalog
                    .record_refs
                    .contains_key(&document.document_id));
            }
            let old_document = old_collection.get_document(&document.document_id).unwrap();
            assert_eq!(old_document.title, document.title);
            assert_eq!(old_document.source_uri, document.source_uri);
            assert_eq!(old_document.created_at, document.created_at);
            assert_eq!(
                old_collection
                    .get_document(&document.document_id)
                    .unwrap()
                    .metadata,
                metadata
            );
            db.compact(CompactOptions::default()).unwrap();
            assert_eq!(
                old_collection.get_chunk(&chunk.chunk_id).unwrap().metadata,
                metadata
            );
            assert_eq!(db.stats().unwrap().document_count, 0);
            assert_eq!(db.stats().unwrap().chunk_count, 0);
            drop((old_collection, old, collection, db));
            let reopened = Database::open_read_only(&path).unwrap();
            assert_eq!(reopened.stats().unwrap().document_count, 0);
            assert_eq!(
                reopened.check(CheckOptions::default()).unwrap().tombstones,
                0
            );
            drop(reopened);
            fs::remove_dir_all(output).unwrap();
            fs::remove_file(path).unwrap();
        }
    }

    fn temp_path(name: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!("cofferdb-db-{name}-{}.coffer", std::process::id()));
        let _ = std::fs::remove_file(&path);
        path
    }

    #[test]
    fn bounded_text_rebuild_shares_runtime_disk_and_does_not_publish_failed_sorting() {
        let directory = temp_path("bounded-text-runtime").with_extension("dir");
        fs::create_dir(&directory).unwrap();
        let path = directory.join("data.coffer");
        let mut db = Database::create(&path, DatabaseConfig::default()).unwrap();
        assert!(Arc::ptr_eq(
            &db.temporary_work,
            &db.lock().unwrap().temporary_work
        ));
        assert!(Arc::ptr_eq(
            &db.temporary_work,
            &db.clone().temporary_disk_budget()
        ));
        let collection = db
            .create_collection(CollectionConfig {
                name: "notes".into(),
                embedding_dim: 2,
                distance_metric: DistanceMetric::Cosine,
                embedding_model: None,
                tokenizer: Default::default(),
                vectors_normalized: false,
                vector_index: VectorIndexConfig::Exact,
            })
            .unwrap();
        let text = (0..4000).map(|n| format!("token{n} ")).collect::<String>() + "shared shared";
        collection
            .insert_document(DocumentInput {
                external_id: "doc".into(),
                title: None,
                source_uri: None,
                metadata: JsonValue::object(),
                chunks: vec![ChunkInput {
                    chunk_external_id: "chunk".into(),
                    text,
                    ordinal: 0,
                    metadata: JsonValue::object(),
                    source_span: None,
                }],
            })
            .unwrap();
        let before = fs::read(&path).unwrap();
        let index_before = format!("{:?}", db.lock().unwrap().text_index);
        let snapshot_before = db.lock().unwrap().catalog.state_signature();
        let original_disk = db.temporary_disk_budget();
        let refusing = Arc::new(crate::temporary::TemporaryDiskBudget::new(0));
        db.temporary_work = refusing.clone();
        db.lock().unwrap().temporary_work = refusing.clone();
        let error = db.rebuild_index(Some("notes")).unwrap_err();
        assert_eq!(error.code(), ErrorCode::ResourceExhausted);
        assert_eq!(
            error_detail_value(&error, "resource"),
            Some("temporary_disk")
        );
        assert_eq!(fs::read(&path).unwrap(), before);
        assert_eq!(format!("{:?}", db.lock().unwrap().text_index), index_before);
        assert_eq!(
            db.lock().unwrap().catalog.state_signature(),
            snapshot_before
        );
        assert_eq!(refusing.used(), 0);
        assert!(!fs::read_dir(&directory).unwrap().any(|entry| entry
            .unwrap()
            .file_name()
            .to_string_lossy()
            .contains("text-sort")));
        db.temporary_work = original_disk.clone();
        db.lock().unwrap().temporary_work = original_disk.clone();
        db.rebuild_index(Some("notes")).unwrap();
        assert_eq!(original_disk.used(), 0);
        let hits = collection.search().text("shared token5").execute().unwrap();
        assert_eq!(hits.results.len(), 1);
        assert_eq!(hits.results[0].chunk_external_id, "chunk");
        drop((collection, db));
        let reopened = Database::open_read_only(&path).unwrap();
        assert!(Arc::ptr_eq(
            &reopened.temporary_work,
            &reopened.lock().unwrap().temporary_work
        ));
        assert_eq!(
            reopened
                .collection("notes")
                .unwrap()
                .search()
                .text("shared token5")
                .execute()
                .unwrap()
                .results
                .len(),
            1
        );
        drop(reopened);
        fs::remove_dir_all(directory).unwrap();
    }

    fn id_number(id: &str, prefix: &str) -> u64 {
        id.strip_prefix(prefix).unwrap().parse::<u64>().unwrap()
    }

    fn snapshot_directory_bytes(db: &Database) -> usize {
        db.index_directory_bytes()
    }

    fn error_detail_value<'a>(err: &'a Error, key: &str) -> Option<&'a str> {
        err.detail()
            .iter()
            .find(|(field, _)| field == key)
            .map(|(_, value)| value.as_str())
    }

    fn append_fake_index_suffix(path: &Path) {
        let mut file = std::fs::OpenOptions::new().append(true).open(path).unwrap();
        file.write_all(b"COFFERIDX").unwrap();
        file.write_all(&[0xa5; 48]).unwrap();
    }

    fn append_overflowing_checkpoint_payload_suffix(path: &Path, scan: &ScanReport) {
        let mut header = Vec::with_capacity(105);
        header.extend_from_slice(b"CFIDXCHK");
        header.extend_from_slice(&1u16.to_le_bytes());
        header.extend_from_slice(&105u16.to_le_bytes());
        header.extend_from_slice(&scan.critical_end_offset.to_le_bytes());
        header.extend_from_slice(&scan.critical_end_offset.to_le_bytes());
        header.extend_from_slice(&scan.critical_end_offset.to_le_bytes());
        header.extend_from_slice(&scan.last_valid_commit.to_le_bytes());
        header.extend_from_slice(&scan.last_valid_commit_sequence_no.to_le_bytes());
        header.push(1);
        header.extend_from_slice(&0u32.to_le_bytes());
        header.extend_from_slice(&1u64.to_le_bytes());
        header.extend_from_slice(&1u64.to_le_bytes());
        header.extend_from_slice(&0u64.to_le_bytes());
        header.extend_from_slice(&46u64.to_le_bytes());
        header.extend_from_slice(&checksum32(&[]).to_le_bytes());
        header.extend_from_slice(&u64::MAX.to_le_bytes());
        header.extend_from_slice(&checksum32(&[]).to_le_bytes());
        assert_eq!(header.len(), 105);
        let mut file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .unwrap();
        file.set_len(scan.critical_end_offset).unwrap();
        file.seek(SeekFrom::Start(scan.critical_end_offset))
            .unwrap();
        file.write_all(&header).unwrap();
    }

    fn fail_next_index_checkpoint_refresh(db: &Database) {
        db.lock().unwrap().fail_next_index_checkpoint_refresh = true;
    }

    fn weight_value(
        context: &std::collections::BTreeMap<String, JsonValue>,
        weights_field: &str,
        weight: &str,
    ) -> Option<f64> {
        context
            .get(weights_field)
            .and_then(JsonValue::as_object)
            .and_then(|weights| weights.get(weight))
            .and_then(JsonValue::as_f64)
    }

    fn json_array_contains(value: Option<&JsonValue>, needle: &str) -> bool {
        value
            .and_then(JsonValue::as_array)
            .map(|items| {
                items
                    .iter()
                    .any(|item| item.as_str().map(|item| item == needle).unwrap_or(false))
            })
            .unwrap_or(false)
    }

    fn assert_stored_read_admission(error: &Error, available: usize, field_prefixes: &[usize]) {
        if let Some(required) = field_prefixes
            .iter()
            .copied()
            .find(|bytes| *bytes > available)
        {
            let limit = crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES;
            assert_eq!(error.code(), ErrorCode::ResourceExhausted, "{error:?}");
            for (key, value) in [
                ("resource", "memory".into()),
                ("operation", "operation_scratch".into()),
                ("limit_bytes", limit.to_string()),
                (
                    "requested_bytes",
                    (limit - available + required).to_string(),
                ),
            ] {
                assert_eq!(
                    error_detail_value(error, key),
                    Some(value.as_str()),
                    "{error:?}"
                );
            }
        } else {
            assert_eq!(error.code(), ErrorCode::RecoveryError, "{error:?}");
            for (key, value) in [
                ("json_input_kind", "stored_json".into()),
                ("json_limit_reason", "work_budget_exceeded".into()),
                ("limit_unit", "bytes".into()),
                (
                    "limit",
                    (available - field_prefixes.last().copied().unwrap_or(0)).to_string(),
                ),
            ] {
                assert_eq!(
                    error_detail_value(error, key),
                    Some(value.as_str()),
                    "{error:?}"
                );
            }
        }
    }

    #[test]
    fn second_database_open_fails_with_lock_contention() {
        let path = temp_path("writer-lock");
        let db = Database::create(&path, DatabaseConfig::default()).unwrap();
        let err = match Database::open(&path) {
            Ok(_) => panic!("second writer should fail while the first handle is open"),
            Err(err) => err,
        };
        assert_eq!(err.code(), ErrorCode::LockContention);
        assert_eq!(
            err.to_json().get("code").and_then(JsonValue::as_str),
            Some("lock_contention")
        );
        drop(db);
        Database::open(&path).unwrap();
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn database_check_shares_runtime_json_capacity_for_current_and_legacy_files() {
        use crate::json::JsonInputKind;
        for legacy in [false, true] {
            let path = if legacy {
                let path = temp_path("legacy-check-json-budget");
                let hex = include_str!("../tests/fixtures/legacy-2d9eb64.hex")
                    .split_whitespace()
                    .collect::<String>();
                let bytes = hex
                    .as_bytes()
                    .chunks_exact(2)
                    .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
                    .collect::<Vec<_>>();
                fs::write(&path, bytes).unwrap();
                path
            } else {
                query_preflight_fixture("check-json-budget", "needle recovery")
            };
            let path = fs::canonicalize(path).unwrap();
            let db = Database::open_read_only(&path).unwrap();
            let capacity = db.json_context().available_workspace_bytes();
            let before = fs::read(&path).unwrap();
            let baseline = db
                .check(CheckOptions {
                    verify_indexes: true,
                })
                .unwrap()
                .to_json();
            let fields = if legacy {
                Vec::new()
            } else {
                let inner = db.lock().unwrap();
                let collection = inner.catalog.collection_by_name("first").unwrap();
                let mut total = 0;
                [
                    collection.collection_id.len(),
                    collection.name.len(),
                    collection.embedding_model.as_ref().unwrap().len(),
                    collection.created_at.len(),
                ]
                .into_iter()
                .map(|bytes| {
                    total += bytes;
                    total
                })
                .collect::<Vec<_>>()
            };
            for available in 0..100 {
                let mut held = db.clone().json_context().reservation();
                held.grow(capacity - available, JsonInputKind::Filter)
                    .unwrap();
                let error = db
                    .check(CheckOptions {
                        verify_indexes: available % 2 == 0,
                    })
                    .unwrap_err();
                // Binary collection fields precede config JSON; legacy records
                // are JSON from the first byte and have no typed-string prefix.
                assert_stored_read_admission(&error, available, &fields);
                held.grow(available, JsonInputKind::Filter).unwrap();
                if available == 0 {
                    assert_eq!(
                        Database::check_path(
                            &path,
                            CheckOptions {
                                verify_indexes: true
                            }
                        )
                        .unwrap()
                        .to_json(),
                        baseline
                    );
                }
            }
            assert_eq!(
                db.check(CheckOptions {
                    verify_indexes: true
                })
                .unwrap()
                .to_json(),
                baseline
            );
            assert_eq!(fs::read(&path).unwrap(), before);
            db.json_context()
                .reservation()
                .grow(capacity, JsonInputKind::Filter)
                .unwrap();
            drop(db);
            Database::open_read_only(&path).unwrap();
            fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn compaction_iterator_keeps_each_embedding_reservation_until_consumed() {
        let path = query_preflight_fixture("compact-vector-ownership", "query text");
        let db = Database::open(&path).unwrap();
        let before = fs::read(&path).unwrap();
        let inner = db.lock().unwrap();
        let prefix = inner.catalog.live_collections().count()
            + inner.catalog.live_documents().count()
            + inner.catalog.live_chunks().count();
        let limit = db.json_context().available_workspace_bytes();
        for stored in [false, true] {
            let mut catalog = inner.catalog.clone();
            if !stored {
                catalog.record_refs.clear();
                for embedding in catalog.embeddings.values_mut() {
                    embedding.vector = vec![1.0, 0.0];
                }
            }
            for seed in 0..100usize {
                let context = db.json_context();
                let mut records =
                    compaction_records(&catalog, inner.storage.payload_reader().unwrap(), &context);
                for _ in 0..prefix {
                    drop(records.next().unwrap().unwrap());
                }
                let mut held = context.workspace_reservation();
                held.grow_to(limit - seed % 32, limit - seed % 32, "operation_scratch")
                    .unwrap();
                assert_eq!(
                    records.next().unwrap().err().unwrap().code(),
                    ErrorCode::ResourceExhausted
                );
                drop(records);
                held.grow_to(limit, limit, "operation_scratch").unwrap();
                drop(held);
                let mut records =
                    compaction_records(&catalog, inner.storage.payload_reader().unwrap(), &context);
                for _ in 0..prefix {
                    drop(records.next().unwrap().unwrap());
                }
                let mut count = 0;
                for record in records {
                    context
                        .workspace_reservation()
                        .grow_to(limit, limit, "operation_scratch")
                        .expect_err("current yielded embedding must be charged");
                    let (record, work) = record.unwrap();
                    let RecordPayload::EmbeddingPut(embedding) = record else {
                        panic!()
                    };
                    assert_eq!(embedding.vector, vec![1.0, 0.0]);
                    let bytes = embedding.vector.capacity() * 4
                        + embedding.embedding_id.capacity()
                        + embedding.chunk_id.capacity()
                        + embedding.collection_id.capacity()
                        + embedding.chunk_external_id.capacity()
                        + embedding.model_name.capacity()
                        + embedding.created_at.capacity();
                    let mut held = context.workspace_reservation();
                    held.grow_to(limit - bytes, limit - bytes, "operation_scratch")
                        .unwrap();
                    assert!(held
                        .grow_to(limit - bytes + 1, limit - bytes + 1, "operation_scratch")
                        .is_err());
                    drop(embedding);
                    drop(work);
                    held.grow_to(limit, limit, "operation_scratch").unwrap();
                    count += 1;
                }
                assert_eq!(count, catalog.live_embeddings().count());
            }
        }
        assert_eq!(fs::read(&path).unwrap(), before);
        drop(inner);
        drop(db);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn compaction_collection_and_document_copies_keep_all_string_credit() {
        let path = query_preflight_fixture("compact-owned-summaries", "needle recovery");
        let db = Database::open(&path).unwrap();
        let inner = db.lock().unwrap();
        let mut catalog = inner.catalog.clone();
        let dropped = catalog
            .collection_by_name("empty")
            .unwrap()
            .collection_id
            .clone();
        catalog.deleted_collections.insert(dropped.clone());
        let limit = db.json_context().available_workspace_bytes();
        for seed in 0..100usize {
            if seed % 2 == 0 {
                catalog
                    .collection_drop_times
                    .insert(dropped.clone(), "2026-10-08T01:02:03Z".into());
            } else {
                catalog.collection_drop_times.remove(&dropped);
            }
            let context = db.json_context();
            let mut records =
                compaction_records(&catalog, inner.storage.payload_reader().unwrap(), &context);
            let mut seen = [0; 3];
            for record in &mut records {
                let (record, work) = record.unwrap();
                let strings = match &record {
                    RecordPayload::CollectionCreate(collection) => {
                        seen[0] += 1;
                        collection.collection_id.capacity()
                            + collection.name.capacity()
                            + collection
                                .embedding_model
                                .as_ref()
                                .map_or(0, String::capacity)
                            + collection.created_at.capacity()
                    }
                    RecordPayload::CollectionDrop {
                        collection_id,
                        name,
                        dropped_at,
                        reason,
                    } => {
                        seen[1] += 1;
                        assert_eq!(collection_id, &dropped);
                        assert!(reason.is_none());
                        assert_eq!(
                            dropped_at,
                            catalog
                                .collection_drop_times
                                .get(&dropped)
                                .unwrap_or(&catalog.collections[&dropped].created_at)
                        );
                        collection_id.capacity() + name.capacity() + dropped_at.capacity()
                    }
                    RecordPayload::DocumentPut(document) => {
                        seen[2] += 1;
                        document.document_id.capacity()
                            + document.collection_id.capacity()
                            + document.external_id.capacity()
                            + document.title.as_ref().map_or(0, String::capacity)
                            + document.source_uri.as_ref().map_or(0, String::capacity)
                            + document.created_at.capacity()
                            + document.updated_at.capacity()
                    }
                    _ => continue,
                };
                let bytes = work.json_bytes() + strings;
                let mut held = context.workspace_reservation();
                held.grow_to(limit - bytes, limit - bytes, "operation_scratch")
                    .unwrap();
                assert!(held
                    .grow_to(limit - bytes + 1, limit - bytes + 1, "operation_scratch")
                    .is_err());
                drop(record);
                drop(work);
                held.grow_to(limit, limit, "operation_scratch").unwrap();
            }
            assert_eq!(seen, [3, 1, 2]);
        }
        drop(inner);
        drop(db);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn compact_payload_budget_failure_preserves_source_and_old_readers() {
        for large_field in ["text", "vector", "document"] {
            let large_vector = large_field == "vector";
            let directory = temp_path("compact-vector-budget");
            fs::create_dir(&directory).unwrap();
            let path = directory.join("data.coffer");
            let db = Database::create(&path, DatabaseConfig::default()).unwrap();
            let dim = if large_vector { 32_769 } else { 2 };
            let collection = db
                .create_collection(CollectionConfig {
                    tokenizer: Default::default(),
                    name: "notes".into(),
                    embedding_dim: dim,
                    distance_metric: DistanceMetric::Cosine,
                    embedding_model: Some("m".into()),
                    vectors_normalized: false,
                    vector_index: VectorIndexConfig::Exact,
                })
                .unwrap();
            collection
                .insert_document(DocumentInput {
                    external_id: "document".into(),
                    title: (large_field == "document").then(|| "title ".repeat(32 * 1024)),
                    source_uri: None,
                    metadata: JsonValue::object(),
                    chunks: vec![ChunkInput {
                        chunk_external_id: "chunk".into(),
                        ordinal: 0,
                        text: if large_field == "text" {
                            "needle recovery ".repeat(8192)
                        } else {
                            "needle recovery".into()
                        },
                        metadata: JsonValue::object(),
                        source_span: None,
                    }],
                })
                .unwrap();
            let mut vector = vec![0.0; dim];
            vector[0] = 1.0;
            collection
                .put_embeddings(vec![EmbeddingInput {
                    chunk_external_id: "chunk".into(),
                    model_name: "m".into(),
                    dim,
                    vector: vector.clone(),
                }])
                .unwrap();
            let query = |database: &Database| {
                database
                    .collection("notes")
                    .unwrap()
                    .search()
                    .query_embedding(QueryEmbeddingInput {
                        model_name: Some("m".into()),
                        dim,
                        vector: vector.clone(),
                    })
                    .execute()
                    .unwrap()
                    .results
                    .into_iter()
                    .map(|hit| (hit.chunk_external_id, hit.final_score.to_bits()))
                    .collect::<Vec<_>>()
            };
            let old_reader = Database::open_read_only(&path).unwrap();
            let expected = query(&old_reader);
            assert_eq!(expected, vec![("chunk".into(), 1.0f64.to_bits())]);
            let entries = fs::read_dir(&directory).unwrap().count();
            let old_bytes = fs::metadata(&path).unwrap().len();
            let mut retained = 0;
            for seed in 0..100 {
                let before = fs::read(&path).unwrap();
                let commit = db.stats().unwrap().last_commit_id;
                let available = 96 * 1024 + seed;
                let context = db.json_context();
                let limit = crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES
                    - db.index_directory_bytes();
                let mut held = context.workspace_reservation();
                held.grow_to(limit - available, limit - available, "operation_scratch")
                    .unwrap();
                let error = db
                    .compact(CompactOptions {
                        rebuild_indexes: seed % 2 == 0,
                    })
                    .unwrap_err();
                assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                assert_eq!(fs::read(&path).unwrap(), before);
                assert_eq!(db.stats().unwrap().last_commit_id, commit);
                assert_eq!(db.temporary_work.used(), retained);
                assert_eq!(fs::read_dir(&directory).unwrap().count(), entries);
                held.grow_to(limit, limit, "operation_scratch").unwrap();
                drop(held);

                db.compact(CompactOptions {
                    rebuild_indexes: seed % 2 == 0,
                })
                .unwrap();
                retained = old_bytes;
                assert_eq!(query(&db), expected);
                assert_eq!(query(&old_reader), expected);
                assert_eq!(query(&Database::open_read_only(&path).unwrap()), expected);
                let capacity = crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES
                    - db.index_directory_bytes();
                context
                    .workspace_reservation()
                    .grow_to(capacity, capacity, "operation_scratch")
                    .unwrap();
                assert_eq!(db.temporary_work.used(), retained);
                assert_eq!(fs::read_dir(&directory).unwrap().count(), entries);
            }
            drop(old_reader);
            assert_eq!(db.temporary_work.used(), 0);
            drop((collection, db));
            fs::remove_dir_all(directory).unwrap();
        }
    }

    #[test]
    fn compaction_iterator_retains_chunk_json_in_the_shared_context() {
        use crate::json::JsonInputKind;
        let path = query_preflight_fixture("compact-reader-json-budget", "needle recovery");
        let db = Database::open(&path).unwrap();
        let inner = db.lock().unwrap();
        let prefix = inner.catalog.collections.len() + inner.catalog.live_documents().count();
        let capacity = db.json_context().available_workspace_bytes();
        for stored in [false, true] {
            let mut catalog = inner.catalog.clone();
            if !stored {
                catalog.record_refs.clear();
            }
            for seed in 0..100 {
                let available = if stored { 1024 + seed } else { seed };
                let context = db.json_context();
                let mut records =
                    compaction_records(&catalog, inner.storage.payload_reader().unwrap(), &context);
                for _ in 0..prefix {
                    drop(records.next().unwrap().unwrap());
                }
                let mut held = db.json_context().reservation();
                held.grow(capacity - available, JsonInputKind::Filter)
                    .unwrap();
                let error = records.next().unwrap().err().unwrap();
                assert_eq!(error.code(), ErrorCode::RecoveryError);
                assert_eq!(
                    error_detail_value(&error, "json_input_kind"),
                    Some("stored_json")
                );
                assert_eq!(
                    error_detail_value(&error, "json_limit_reason"),
                    Some("work_budget_exceeded")
                );
                drop(records);
                held.grow(available, JsonInputKind::Filter).unwrap();
                drop(held);
                let mut records =
                    compaction_records(&catalog, inner.storage.payload_reader().unwrap(), &context);
                for _ in 0..prefix {
                    drop(records.next().unwrap().unwrap());
                }
                let (record, work) = records.next().unwrap().unwrap();
                let RecordPayload::ChunkPut(chunk) = record else {
                    panic!()
                };
                assert_eq!(chunk.metadata, parse(r#"{"lang":"rust"}"#).unwrap());
                assert_eq!(chunk.embedding_id.is_some(), !stored);
                if stored {
                    assert_eq!(chunk.text, "needle recovery");
                }
                assert!(work.json_bytes() > 0);
                let retained = work.json_bytes()
                    + chunk.chunk_id.capacity()
                    + chunk.document_id.capacity()
                    + chunk.collection_id.capacity()
                    + chunk.chunk_external_id.capacity()
                    + chunk.text.capacity()
                    + chunk.text_hash.capacity()
                    + chunk.embedding_id.as_ref().map_or(0, String::capacity)
                    + chunk.created_at.capacity();
                let mut held = db.json_context().reservation();
                held.grow(capacity - retained, JsonInputKind::Filter)
                    .unwrap();
                assert!(held.grow(1, JsonInputKind::Filter).is_err());
                drop(chunk);
                drop(work);
                held.grow(retained, JsonInputKind::Filter).unwrap();
            }
        }
        drop(inner);
        drop(db);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn compact_record_rewrite_shares_json_capacity_and_preserves_original_on_rejection() {
        use crate::json::{JsonInputKind, JsonProcessingPolicyV1};
        let directory = temp_path("compact-record-json-budget");
        fs::create_dir(&directory).unwrap();
        let path = directory.join("data.coffer");
        let db = Database::create(&path, DatabaseConfig::default()).unwrap();
        let collection = db
            .create_collection(CollectionConfig {
                tokenizer: Default::default(),
                name: "notes".into(),
                embedding_dim: 2,
                distance_metric: DistanceMetric::Cosine,
                embedding_model: None,
                vectors_normalized: false,
                vector_index: VectorIndexConfig::Exact,
            })
            .unwrap();
        let metadata = object_from_pairs([("payload", JsonValue::String("x".repeat(256 * 1024)))]);
        collection
            .insert_document(DocumentInput {
                external_id: "document".into(),
                title: None,
                source_uri: None,
                metadata: metadata.clone(),
                chunks: vec![],
            })
            .unwrap();
        let before = fs::read(&path).unwrap();
        let commit = db.stats().unwrap().last_commit_id;
        let files = || {
            let mut files: Vec<_> = fs::read_dir(&directory)
                .unwrap()
                .map(|entry| entry.unwrap().file_name())
                .collect();
            files.sort();
            files
        };
        let original_files = files();
        let capacity = db.json_context().available_workspace_bytes();
        let management = crate::temporary::test_file_management_bytes(
            db.lock().unwrap().storage.path().parent().unwrap(),
        );
        let collection_string_bytes = if cfg!(unix) {
            0
        } else {
            let inner = db.lock().unwrap();
            let info = inner.catalog.collection_by_name("notes").unwrap();
            info.collection_id.len() + info.name.len() + info.created_at.len()
        };
        let minimum = if cfg!(unix) {
            management + 128
        } else {
            time::MAX_TIMESTAMP_BYTES + collection_string_bytes
        };
        for seed in 0..100 {
            let available = if seed < 50 { seed } else { 64 * 1024 + seed };
            let rebuild_indexes = seed % 2 == 0;
            let text_bytes = if rebuild_indexes {
                management
                    + std::mem::size_of::<crate::storage::TextExtent>()
                    + std::mem::size_of::<(u64, usize)>()
            } else {
                0
            };
            let mut held = db.clone().json_context().reservation();
            held.grow(capacity - available, JsonInputKind::Filter)
                .unwrap();
            let error = db.compact(CompactOptions { rebuild_indexes }).unwrap_err();
            if available < minimum {
                assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                assert_eq!(error_detail_value(&error, "resource"), Some("memory"));
                assert_eq!(
                    error_detail_value(&error, "operation"),
                    Some(
                        if !rebuild_indexes && !cfg!(unix) && available < time::MAX_TIMESTAMP_BYTES
                        {
                            "batch_prepare"
                        } else {
                            "operation_scratch"
                        }
                    )
                );
            } else {
                assert_eq!(error.code(), ErrorCode::RecoveryError);
                assert_eq!(
                    error_detail_value(&error, "json_input_kind"),
                    Some("stored_json")
                );
                assert_eq!(
                    error_detail_value(&error, "json_limit_reason"),
                    Some("work_budget_exceeded")
                );
                assert_eq!(
                    error.to_json().get("detail").unwrap().get("limit"),
                    Some(&JsonValue::Number(
                        (available
                            - text_bytes
                            - management
                            - time::MAX_TIMESTAMP_BYTES
                            - if seed < 50 {
                                collection_string_bytes
                            } else {
                                0
                            }) as f64
                    ))
                );
            }
            held.grow(available, JsonInputKind::Filter).unwrap();
            assert_eq!(fs::read(&path).unwrap(), before);
            assert_eq!(db.stats().unwrap().last_commit_id, commit);
            assert!(!db.index_stats().unwrap().rebuild_needed);
            assert_eq!(files(), original_files);
        }
        for rebuild_indexes in [false, true] {
            db.compact(CompactOptions { rebuild_indexes }).unwrap();
            assert_eq!(
                collection
                    .get_document_by_external_id("document")
                    .unwrap()
                    .metadata,
                metadata
            );
        }
        db.json_context()
            .reservation()
            .grow(
                JsonProcessingPolicyV1::WORK_BUDGET_BYTES
                    - db.lock()
                        .unwrap()
                        .lazy_vector_source
                        .as_ref()
                        .unwrap()
                        .reserved_bytes(),
                JsonInputKind::Filter,
            )
            .unwrap();
        drop(collection);
        drop(db);
        let db = Database::open(&path).unwrap();
        assert_eq!(
            db.collection("notes")
                .unwrap()
                .get_document_by_external_id("document")
                .unwrap()
                .metadata,
            metadata
        );
        drop(db);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn text_index_rebuild_observes_shared_json_pressure_before_replacement() {
        let path = query_preflight_fixture("rebuild-shared-json", "needle recovery");
        let db = Database::open(&path).unwrap();
        let before = fs::read(&path).unwrap();
        let commit = db.stats().unwrap().last_commit_id;
        // Required checkpoint directories precede any primary-record JSON decoding.
        let collections = db.lock().unwrap().catalog.live_collections().count();
        let references = collections * std::mem::size_of::<&CollectionInfo>();
        let fields = [
            references,
            references + collections * std::mem::size_of::<crate::storage::TextExtent>(),
        ];
        let capacity = db.json_context().available_workspace_bytes();
        for available in 0..fields[1] {
            let mut held = db.clone().json_context().reservation();
            held.grow(capacity - available, crate::json::JsonInputKind::Filter)
                .unwrap();
            for error in [
                db.rebuild_index(Some("first")).unwrap_err(),
                db.compact(CompactOptions {
                    rebuild_indexes: true,
                })
                .unwrap_err(),
            ] {
                assert_stored_read_admission(&error, available, &fields);
            }
            held.grow(available, crate::json::JsonInputKind::Filter)
                .unwrap();
            assert_eq!(fs::read(&path).unwrap(), before);
            assert_eq!(db.stats().unwrap().last_commit_id, commit);
            assert!(!db.index_stats().unwrap().rebuild_needed);
        }
        db.rebuild_index(None).unwrap();
        db.compact(CompactOptions {
            rebuild_indexes: true,
        })
        .unwrap();
        assert_eq!(
            db.collection("first")
                .unwrap()
                .search()
                .text("needle")
                .execute()
                .unwrap()
                .results
                .len(),
            2
        );
        let mut held = db.json_context().reservation();
        held.grow(
            crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES - snapshot_directory_bytes(&db),
            crate::json::JsonInputKind::Filter,
        )
        .unwrap();
        drop((held, db));
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn automatic_text_rebuild_keeps_operation_quota_and_committed_data_on_rejection() {
        for reusable in [false, true] {
            let path =
                query_preflight_fixture(&format!("automatic-json-{reusable}"), "needle recovery");
            let db = Database::open(&path).unwrap();
            let context = db.json_context();
            let mut inner = db.lock().unwrap();
            let id = inner
                .catalog
                .collection_by_name("first")
                .unwrap()
                .collection_id
                .clone();
            let chunk_id = inner
                .catalog
                .chunk_by_external_id(&id, "chunk-1")
                .unwrap()
                .chunk_id
                .clone();
            let (mut chunk, work) = inner.materialize_chunk_in(&chunk_id, &context).unwrap();
            drop(work);
            chunk.text = "fresh replacement".into();
            chunk.text_hash = chunk_text_hash(&chunk.text);
            chunk.token_count = token_count_hint(&chunk.text);
            if !reusable {
                inner.mark_index_checkpoint_rebuild_needed();
            }
            let pending = inner
                .commit_records(vec![RecordPayload::ChunkPut(chunk)])
                .unwrap();
            assert_eq!(pending.reusable_checkpoint, reusable);
            let committed = inner.catalog.last_commit_id;
            let collections = inner.catalog.live_collections().count();
            let references = collections * std::mem::size_of::<&CollectionInfo>();
            let prepared_bytes = if reusable {
                collections * std::mem::size_of::<crate::storage::TextExtent>()
                    + crate::temporary::test_file_management_bytes(
                        inner.storage.path().parent().unwrap(),
                    )
            } else {
                0
            };
            let bytes = fs::read(&path).unwrap();
            let mut held = context.reservation();
            held.grow(
                crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES
                    - pending.changes.work.reserved_bytes()
                    - prepared_bytes,
                crate::json::JsonInputKind::Filter,
            )
            .unwrap();
            let error = inner.finish_index_update(pending, &context).unwrap_err();
            let limit = crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES;
            assert_eq!(error.code(), ErrorCode::ResourceExhausted);
            for (key, value) in [
                ("resource", "memory".into()),
                ("operation", "operation_scratch".into()),
                ("limit_bytes", limit.to_string()),
                ("requested_bytes", (limit + references).to_string()),
            ] {
                assert_eq!(
                    error_detail_value(&error, key),
                    Some(value.as_str()),
                    "{error:?}"
                );
            }
            assert_eq!(inner.catalog.last_commit_id, committed);
            assert_eq!(inner.index_suffix_status, IndexSuffixState::RebuildNeeded);
            assert_eq!(fs::read(&path).unwrap(), bytes);
            drop((held, inner));
            let results = db
                .collection("first")
                .unwrap()
                .search()
                .text("fresh")
                .execute()
                .unwrap();
            assert_eq!(results.results.len(), 1);
            assert_eq!(results.results[0].chunk_external_id, "chunk-1");
            assert_eq!(results.query_context.index_source, "transient_rebuild");
            drop(db);
            let db = Database::open(&path).unwrap();
            assert_eq!(db.stats().unwrap().last_commit_id, committed);
            db.rebuild_index(None).unwrap();
            assert!(!db.index_stats().unwrap().rebuild_needed);
            assert_eq!(
                db.collection("first")
                    .unwrap()
                    .search()
                    .text("fresh")
                    .execute()
                    .unwrap()
                    .results
                    .len(),
                1
            );
            let mut held = db.json_context().reservation();
            held.grow(
                crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES
                    - snapshot_directory_bytes(&db),
                crate::json::JsonInputKind::Filter,
            )
            .unwrap();
            drop((held, db));
            fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn database_clones_share_json_headroom_and_rejection_keeps_file_unchanged() {
        let path = temp_path("shared-json-work");
        let db = Database::create(&path, DatabaseConfig::default()).unwrap();
        let collection = db
            .create_collection(CollectionConfig {
                name: "notes".into(),
                tokenizer: crate::TokenizerConfig::default(),
                embedding_dim: 2,
                distance_metric: crate::DistanceMetric::Cosine,
                embedding_model: None,
                vectors_normalized: false,
                vector_index: crate::VectorIndexConfig::Exact,
            })
            .unwrap();
        let before = std::fs::read(&path).unwrap();
        let context = db.clone().json_context();
        let mut held = context.reservation();
        held.grow(
            context.available_workspace_bytes(),
            crate::json::JsonInputKind::Filter,
        )
        .unwrap();
        let input = || DocumentInput {
            external_id: "doc".into(),
            title: None,
            source_uri: None,
            metadata: JsonValue::object(),
            chunks: Vec::new(),
        };
        let error = collection.insert_document(input()).unwrap_err();
        assert_eq!(error.code(), ErrorCode::InvalidJsonlSchema);
        let detail = error.to_json();
        assert_eq!(
            detail.get("detail").unwrap().get("limit").unwrap().as_f64(),
            Some(0.0)
        );
        assert!(error
            .detail()
            .iter()
            .any(|(key, value)| key == "json_input_kind" && value == "document_metadata"));
        assert_eq!(std::fs::read(&path).unwrap(), before);
        drop(held);
        assert_eq!(
            collection
                .insert_document(input())
                .unwrap()
                .documents_inserted,
            1
        );
        drop(collection);
        drop(db);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn raw_input_capacity_refusal_preserves_file_and_releases_credit() {
        fn padded(value: &str, capacity: usize) -> String {
            let mut result = String::with_capacity(capacity);
            result.push_str(value);
            result
        }
        let path = temp_path("raw-input-capacity");
        let db = Database::create(&path, DatabaseConfig::default()).unwrap();
        let collection = db
            .create_collection(CollectionConfig {
                name: "notes".into(),
                tokenizer: Default::default(),
                embedding_dim: 2,
                distance_metric: DistanceMetric::Cosine,
                embedding_model: None,
                vectors_normalized: false,
                vector_index: VectorIndexConfig::Exact,
            })
            .unwrap();
        let document = |seed: usize| {
            let mut chunks = Vec::with_capacity(4 + seed);
            chunks.push(ChunkInput {
                chunk_external_id: padded("chunk", 1024 + seed),
                text: padded("searchable", 8192 + seed),
                ordinal: 0,
                metadata: JsonValue::object(),
                source_span: None,
            });
            DocumentInput {
                external_id: padded("doc", 2048 + seed),
                title: Some(padded("title", 512 + seed)),
                source_uri: Some(padded("source", 1024 + seed)),
                metadata: JsonValue::object(),
                chunks,
            }
        };
        let vectors = |seed: usize| {
            let mut values = Vec::with_capacity(4 + seed);
            let mut vector = Vec::with_capacity(1024 + seed);
            vector.extend([1.0, 0.0]);
            values.push(EmbeddingInput {
                chunk_external_id: padded("chunk", 1024 + seed),
                model_name: padded("model", 2048 + seed),
                dim: 2,
                vector,
            });
            values
        };
        let before = fs::read(&path).unwrap();
        let commit = db.stats().unwrap().last_commit_id;
        let limit = crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES;
        let capacity = db.json_context().available_workspace_bytes();
        for seed in 0..100 {
            for is_document in [true, false] {
                let context = db.json_context();
                let document_input = document(seed);
                let vector_input = vectors(seed);
                let bytes = if is_document {
                    // Independently sum capacities, including unused array storage.
                    document_input.external_id.capacity()
                        + document_input.title.as_ref().unwrap().capacity()
                        + document_input.source_uri.as_ref().unwrap().capacity()
                        + document_input.chunks.capacity() * std::mem::size_of::<ChunkInput>()
                        + document_input.chunks[0].chunk_external_id.capacity()
                        + document_input.chunks[0].text.capacity()
                } else {
                    vector_input.capacity() * std::mem::size_of::<EmbeddingInput>()
                        + vector_input[0].chunk_external_id.capacity()
                        + vector_input[0].model_name.capacity()
                        + vector_input[0].vector.capacity() * std::mem::size_of::<f32>()
                };
                let mut held = context.workspace_reservation();
                let occupied = capacity - bytes + 1;
                held.grow_to(occupied, occupied, "operation_scratch")
                    .unwrap();
                let error = if is_document {
                    db.insert_document_in_collection(
                        collection.collection_id(),
                        document_input,
                        OnConflict::Error,
                    )
                } else {
                    db.put_embeddings_in_collection(
                        collection.collection_id(),
                        vector_input,
                        OnConflict::Error,
                    )
                }
                .unwrap_err();
                assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                for (key, expected) in [
                    ("resource", "memory".into()),
                    ("operation", "batch_prepare".into()),
                    ("limit_bytes", limit.to_string()),
                    ("requested_bytes", (limit + 1).to_string()),
                ] {
                    assert_eq!(error_detail_value(&error, key), Some(expected.as_str()));
                }
                held.grow_to(capacity, capacity, "operation_scratch")
                    .unwrap();
                assert_eq!(fs::read(&path).unwrap(), before);
                assert_eq!(db.stats().unwrap().last_commit_id, commit);
            }
        }
        // A fitting large allocation must not require a second copy of its capacity.
        let mut document_input = document(0);
        let text = &mut document_input.chunks[0].text;
        text.reserve_exact(31 * 1024 * 1024 - text.len());
        assert_eq!(
            collection
                .insert_document(document_input)
                .unwrap()
                .chunks_inserted,
            1
        );
        let mut vector_input = vectors(0);
        let vector = &mut vector_input[0].vector;
        vector.reserve_exact(31 * 1024 * 1024 / std::mem::size_of::<f32>() - vector.len());
        assert_eq!(
            db.put_embeddings_in_collection(
                collection.collection_id(),
                vector_input,
                OnConflict::Error
            )
            .unwrap()
            .embeddings_inserted,
            1
        );
        assert_eq!(db.stats().unwrap().last_commit_id, commit + 2);
        drop((collection, db));
        let reopened = Database::open_read_only(&path).unwrap();
        assert_eq!(reopened.stats().unwrap().embedding_count, 1);
        drop(reopened);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn skipped_raw_inputs_return_capacity_before_building_the_remaining_batch() {
        let path = temp_path("raw-input-skip-credit");
        for seed in 0..10 {
            let db = Database::create(&path, DatabaseConfig::default()).unwrap();
            let collection = db
                .create_collection(CollectionConfig {
                    name: "notes".into(),
                    tokenizer: Default::default(),
                    embedding_dim: 2,
                    distance_metric: DistanceMetric::Cosine,
                    embedding_model: None,
                    vectors_normalized: false,
                    vector_index: VectorIndexConfig::Exact,
                })
                .unwrap();
            let chunk = |name: &str, ordinal| ChunkInput {
                chunk_external_id: name.into(),
                text: "searchable".into(),
                ordinal,
                metadata: JsonValue::object(),
                source_span: None,
            };
            collection
                .insert_document(DocumentInput {
                    external_id: "original".into(),
                    title: None,
                    source_uri: None,
                    metadata: JsonValue::object(),
                    chunks: vec![chunk("existing", 0), chunk("next", 1)],
                })
                .unwrap();
            let embedding = |name: &str| EmbeddingInput {
                chunk_external_id: name.into(),
                model_name: "model".into(),
                dim: 2,
                vector: vec![1.0, 0.0],
            };
            db.put_embeddings_in_collection(
                collection.collection_id(),
                vec![embedding("existing")],
                OnConflict::Error,
            )
            .unwrap();
            let limit = crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES;
            for mode in 0..3 {
                let context = db.json_context();
                let capacity = context.available_workspace_bytes();
                let mut held = context.workspace_reservation();
                let summary = if mode == 0 {
                    let mut skipped = chunk("existing", 0);
                    skipped.text.reserve_exact(1024 * 1024 + seed);
                    let mut large = String::with_capacity(1024 * 1024 + seed);
                    large.push('x');
                    skipped.metadata = object_from_pairs([("unused", JsonValue::String(large))]);
                    let json_bytes = skipped
                        .metadata
                        .retained_bytes_in(crate::json::JsonInputKind::ChunkMetadata, &context)
                        .unwrap();
                    let input = DocumentInput {
                        external_id: "new".into(),
                        title: None,
                        source_uri: None,
                        metadata: JsonValue::object(),
                        chunks: vec![skipped, chunk("fresh", 1)],
                    };
                    let occupied = capacity - input.workspace_bytes() - json_bytes - 1024;
                    held.grow_to(occupied, occupied, "operation_scratch")
                        .unwrap();
                    let summary = db
                        .insert_document_in_collection(
                            collection.collection_id(),
                            input,
                            OnConflict::Skip,
                        )
                        .unwrap();
                    assert_eq!((summary.chunks_inserted, summary.chunks_skipped), (1, 1));
                    summary
                } else {
                    let mut skipped = embedding("existing");
                    skipped.vector.reserve_exact(256 * 1024 + seed);
                    let input = vec![skipped, embedding(if mode == 1 { "next" } else { "fresh" })];
                    let bytes = input.capacity() * std::mem::size_of::<EmbeddingInput>()
                        + input
                            .iter()
                            .map(EmbeddingInput::workspace_bytes)
                            .sum::<usize>();
                    let occupied = capacity - bytes - 256;
                    held.grow_to(occupied, occupied, "operation_scratch")
                        .unwrap();
                    let summary = db
                        .put_embeddings_in_collection(
                            collection.collection_id(),
                            input,
                            OnConflict::Skip,
                        )
                        .unwrap();
                    assert_eq!(
                        (summary.embeddings_inserted, summary.embeddings_skipped),
                        (1, 1)
                    );
                    summary
                };
                assert_eq!(
                    summary.documents_replaced
                        + summary.chunks_replaced
                        + summary.embeddings_replaced,
                    0
                );
                let remaining = limit - snapshot_directory_bytes(&db);
                held.grow_to(remaining, remaining, "operation_scratch")
                    .unwrap();
            }
            assert_eq!(db.stats().unwrap().chunk_count, 3);
            assert_eq!(db.stats().unwrap().embedding_count, 3);
            assert_eq!(
                collection
                    .get_chunk_by_external_id("existing")
                    .unwrap()
                    .text,
                "searchable"
            );
            drop((collection, db));
            fs::remove_file(&path).unwrap();
        }
    }

    #[test]
    fn batch_duplicate_workspace_rejection_precedes_writes_and_allows_retry() {
        let path = temp_path("batch-duplicate-workspace");
        let db = Database::create(&path, DatabaseConfig::default()).unwrap();
        let collection = db
            .create_collection(CollectionConfig {
                name: "notes".into(),
                tokenizer: Default::default(),
                embedding_dim: 2,
                distance_metric: DistanceMetric::Cosine,
                embedding_model: None,
                vectors_normalized: false,
                vector_index: VectorIndexConfig::Exact,
            })
            .unwrap();
        let input = |count| DocumentInput {
            external_id: "doc".into(),
            title: None,
            source_uri: None,
            metadata: JsonValue::object(),
            chunks: (0..count)
                .map(|ordinal| ChunkInput {
                    chunk_external_id: format!("chunk-{ordinal}"),
                    text: "searchable text".into(),
                    ordinal,
                    metadata: JsonValue::object(),
                    source_span: None,
                })
                .collect(),
        };
        let embeddings = |count| -> Vec<EmbeddingInput> {
            (0..count)
                .map(|index| EmbeddingInput {
                    chunk_external_id: format!("chunk-{index}"),
                    model_name: "model".into(),
                    dim: 2,
                    vector: vec![1.0, 0.0],
                })
                .collect()
        };
        collection.insert_document(input(256)).unwrap();
        let before = fs::read(&path).unwrap();
        let commit = db.stats().unwrap().last_commit_id;
        let index_commit = db.index_stats().unwrap().last_index_commit_id;
        let limit = crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES;
        let capacity = db.json_context().available_workspace_bytes();
        for seed in 0..100usize {
            let context = db.json_context();
            let available = 512 + seed;
            let count = 128 + seed;
            for document in [true, false] {
                let document_input = input(count);
                let vector_input = embeddings(count);
                let bytes = if document {
                    document_input.workspace_bytes()
                } else {
                    vector_input.capacity() * std::mem::size_of::<EmbeddingInput>()
                        + vector_input
                            .iter()
                            .map(EmbeddingInput::workspace_bytes)
                            .sum::<usize>()
                };
                let mut held = context.workspace_reservation();
                let occupied = capacity - available - bytes;
                held.grow_to(occupied, occupied, "operation_scratch")
                    .unwrap();
                let error = if document {
                    db.insert_document_in_collection(
                        collection.collection_id(),
                        document_input,
                        OnConflict::Replace,
                    )
                } else {
                    db.put_embeddings_in_collection(
                        collection.collection_id(),
                        vector_input,
                        OnConflict::Error,
                    )
                }
                .unwrap_err();
                assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                assert_eq!(error_detail_value(&error, "resource"), Some("memory"));
                assert_eq!(
                    error_detail_value(&error, "operation"),
                    Some("catalog_build")
                );
                let requested = limit - available + count * std::mem::size_of::<usize>();
                assert_eq!(
                    error_detail_value(&error, "requested_bytes"),
                    Some(requested.to_string().as_str())
                );
                held.grow_to(capacity, capacity, "operation_scratch")
                    .unwrap();
            }
            assert_eq!(fs::read(&path).unwrap(), before);
            assert_eq!(db.stats().unwrap().last_commit_id, commit);
            assert_eq!(db.stats().unwrap().chunk_count, 256);
            assert_eq!(db.index_stats().unwrap().last_index_commit_id, index_commit);
            assert!(!db.index_stats().unwrap().rebuild_needed);
        }
        db.put_embeddings_in_collection(
            collection.collection_id(),
            embeddings(256),
            OnConflict::Error,
        )
        .unwrap();
        assert_eq!(db.stats().unwrap().embedding_count, 256);
        db.insert_document_in_collection(
            collection.collection_id(),
            input(256),
            OnConflict::Replace,
        )
        .unwrap();
        assert_eq!(db.stats().unwrap().last_commit_id, commit + 2);
        assert_eq!(db.stats().unwrap().chunk_count, 256);
        assert!(!db.index_stats().unwrap().rebuild_needed);
        drop((collection, db));
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn raw_batch_array_rejection_preserves_commits_indexes_and_writer() {
        let path = temp_path("raw-batch-array-budget");
        let db = Database::create(&path, DatabaseConfig::default()).unwrap();
        let collection = db
            .create_collection(CollectionConfig {
                name: "notes".into(),
                tokenizer: Default::default(),
                embedding_dim: 2,
                distance_metric: DistanceMetric::Cosine,
                embedding_model: None,
                vectors_normalized: false,
                vector_index: VectorIndexConfig::Exact,
            })
            .unwrap();
        let input = || DocumentInput {
            external_id: "doc".into(),
            title: None,
            source_uri: None,
            metadata: JsonValue::object(),
            chunks: (0..2)
                .map(|ordinal| ChunkInput {
                    chunk_external_id: format!("chunk-{ordinal}"),
                    text: "original searchable text".into(),
                    ordinal,
                    metadata: JsonValue::object(),
                    source_span: None,
                })
                .collect(),
        };
        collection.insert_document(input()).unwrap();
        let embeddings = || -> Vec<EmbeddingInput> {
            (0..2)
                .map(|index| EmbeddingInput {
                    chunk_external_id: format!("chunk-{index}"),
                    model_name: "test".into(),
                    dim: 2,
                    vector: vec![1.0, 0.0],
                })
                .collect()
        };
        db.put_embeddings_in_collection(
            collection.collection_id(),
            embeddings(),
            OnConflict::Error,
        )
        .unwrap();
        let before = fs::read(&path).unwrap();
        let commit = db.stats().unwrap().last_commit_id;
        let index_commit = db.index_stats().unwrap().last_index_commit_id;
        let slots = std::mem::size_of::<RecordPayload>()
            + std::mem::size_of::<crate::storage::CommittedRecord>();
        let limit = db.json_context().available_workspace_bytes();
        for seed in 0..100 {
            let context = db.clone().json_context();
            let document_input = input();
            let vector_input = embeddings();
            let input_bytes = if seed % 2 == 0 {
                document_input.workspace_bytes()
            } else {
                vector_input.capacity() * std::mem::size_of::<EmbeddingInput>()
                    + vector_input
                        .iter()
                        .map(EmbeddingInput::workspace_bytes)
                        .sum::<usize>()
            };
            let available = input_bytes + 2 * slots - 1 - seed;
            let mut held = context.workspace_reservation();
            held.grow_to(limit - available, limit - available, "operation_scratch")
                .unwrap();
            let error = if seed % 2 == 0 {
                db.insert_document_in_collection(
                    collection.collection_id(),
                    document_input,
                    OnConflict::Replace,
                )
            } else {
                db.put_embeddings_in_collection(
                    collection.collection_id(),
                    vector_input,
                    OnConflict::Replace,
                )
            }
            .unwrap_err();
            assert_eq!(
                error.code(),
                ErrorCode::ResourceExhausted,
                "seed={seed}: {error}"
            );
            assert_eq!(error_detail_value(&error, "resource"), Some("memory"));
            assert_eq!(
                error_detail_value(&error, "operation"),
                Some("batch_prepare")
            );
            assert_eq!(fs::read(&path).unwrap(), before);
            assert_eq!(db.stats().unwrap().last_commit_id, commit);
            assert_eq!(db.index_stats().unwrap().last_index_commit_id, index_commit);
            assert!(!db.index_stats().unwrap().rebuild_needed);
            context
                .workspace_reservation()
                .grow_to(available, available, "operation_scratch")
                .unwrap();
            drop(held);
            assert_eq!(
                collection
                    .search()
                    .text("original")
                    .execute()
                    .unwrap()
                    .results
                    .len(),
                2
            );
            assert_eq!(db.stats().unwrap().document_count, 1);
            assert_eq!(db.stats().unwrap().chunk_count, 2);
            assert_eq!(db.stats().unwrap().embedding_count, 2);
        }
        let summary = db
            .insert_document_in_collection(collection.collection_id(), input(), OnConflict::Replace)
            .unwrap();
        assert_eq!(
            (summary.documents_replaced, summary.chunks_inserted),
            (1, 2)
        );
        assert_eq!(db.stats().unwrap().last_commit_id, commit + 1);
        assert_eq!(
            collection
                .search()
                .text("original")
                .execute()
                .unwrap()
                .results
                .len(),
            2
        );
        drop((collection, db));
        let reopened = Database::open_read_only(&path).unwrap();
        assert_eq!(reopened.stats().unwrap().last_commit_id, commit + 1);
        assert_eq!(reopened.stats().unwrap().chunk_count, 2);
        drop(reopened);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn chunk_list_array_admission_is_atomic_and_releases_result_ownership() {
        let path = temp_path("chunk-list-array-budget");
        let db = Database::create(&path, DatabaseConfig::default()).unwrap();
        let collection = db
            .create_collection(CollectionConfig {
                name: "notes".into(),
                tokenizer: Default::default(),
                embedding_dim: 2,
                distance_metric: DistanceMetric::Cosine,
                embedding_model: None,
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
                chunks: (0..100)
                    .rev()
                    .map(|ordinal| ChunkInput {
                        chunk_external_id: format!("chunk-{ordinal}"),
                        text: format!("text-{ordinal}"),
                        ordinal,
                        metadata: JsonValue::object(),
                        source_span: None,
                    })
                    .collect(),
            })
            .unwrap();
        let capacity = db.json_context().available_workspace_bytes();
        let document = collection.get_document_by_external_id("doc").unwrap();
        let before = fs::read(&path).unwrap();
        let limit = crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES;
        let minimum = 100 * (std::mem::size_of::<Chunk>() + std::mem::size_of::<(usize, &Chunk)>());
        for seed in 0..100 {
            let available = minimum - 1 - seed;
            let context = db.clone().json_context();
            let mut held = context.workspace_reservation();
            held.grow_to(
                capacity - available,
                capacity - available,
                "operation_scratch",
            )
            .unwrap();
            let error = if seed % 2 == 0 {
                collection.list_chunks(&document.document_id)
            } else {
                collection.list_chunks_by_document_external_id("doc")
            }
            .unwrap_err();
            assert_eq!(error.code(), ErrorCode::ResourceExhausted);
            assert_eq!(error_detail_value(&error, "resource"), Some("memory"));
            assert_eq!(
                error_detail_value(&error, "operation"),
                Some("operation_scratch")
            );
            assert_eq!(
                error_detail_value(&error, "requested_bytes"),
                Some((limit + 1 + seed).to_string().as_str())
            );
            assert_eq!(fs::read(&path).unwrap(), before);
            context
                .workspace_reservation()
                .grow_to(available, available, "operation_scratch")
                .unwrap();
            drop(held);
        }
        let chunks = collection.list_chunks(&document.document_id).unwrap();
        let context = db.json_context();
        let mut all = context.workspace_reservation();
        all.grow_to(capacity, capacity, "operation_scratch")
            .unwrap();
        assert_eq!(chunks.len(), 100);
        for (ordinal, chunk) in chunks.iter().enumerate() {
            assert_eq!(chunk.ordinal, ordinal);
            assert_eq!(chunk.chunk_external_id, format!("chunk-{ordinal}"));
            assert_eq!(chunk.text, format!("text-{ordinal}"));
        }
        assert_eq!(fs::read(&path).unwrap(), before);
        drop((all, chunks, collection, db));
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn index_changes_charge_unique_ids_and_merge_duplicates_without_headroom() {
        use crate::json::{JsonOperationContext, JsonProcessingPolicyV1};
        let catalog = Catalog::default();
        for count in 1..=100 {
            let records = (0..count)
                .map(|index| {
                    let id = format!("col_{index:016}");
                    if index % 3 == 0 {
                        RecordPayload::EmbeddingPut(Embedding {
                            embedding_id: String::new(),
                            chunk_id: String::new(),
                            collection_id: id,
                            chunk_external_id: String::new(),
                            model_name: String::new(),
                            dim: 0,
                            vector: Vec::new(),
                            created_at: String::new(),
                        })
                    } else {
                        RecordPayload::CollectionDrop {
                            collection_id: id,
                            name: String::new(),
                            dropped_at: String::new(),
                            reason: None,
                        }
                    }
                })
                .collect::<Vec<_>>();
            let text = (0..count)
                .filter(|index| index % 3 != 0)
                .map(|index| format!("col_{index:016}"))
                .collect::<BTreeSet<_>>();
            let vector = (0..count)
                .map(|index| format!("col_{index:016}"))
                .collect::<BTreeSet<_>>();
            let footprint = |ids: &BTreeSet<String>| {
                let nodes = ids.len().div_ceil(5);
                let node_bytes =
                    11 * std::mem::size_of::<String>() + 16 * std::mem::size_of::<usize>();
                nodes * node_bytes + ids.iter().map(String::len).sum::<usize>()
            };
            let bytes = footprint(&text) + footprint(&vector);
            let context = JsonOperationContext::standalone();
            let limit = JsonProcessingPolicyV1::WORK_BUDGET_BYTES;
            let mut held = context.workspace_reservation();
            held.grow_to(limit - bytes, limit - bytes, "operation_scratch")
                .unwrap();
            let mut changes = IndexChanges::from_records(
                &catalog,
                records.iter().cycle().take(count * 13),
                &context,
            )
            .unwrap();
            assert_eq!(changes.text, text);
            assert_eq!(changes.vector, vector);
            assert_eq!(changes.work.reserved_bytes(), bytes);
            let pointers = |changes: &IndexChanges| {
                changes
                    .text
                    .iter()
                    .chain(&changes.vector)
                    .map(|id| (id.as_ptr(), id.capacity()))
                    .collect::<Vec<_>>()
            };
            let original = pointers(&changes);
            let other =
                IndexChanges::from_records(&catalog, &records, &JsonOperationContext::standalone())
                    .unwrap();
            changes.extend(&other).unwrap();
            assert_eq!(pointers(&changes), original);
            assert_eq!(changes.work.reserved_bytes(), bytes);
            drop(changes);
            held.grow_to(limit - bytes + 1, limit - bytes + 1, "operation_scratch")
                .unwrap();
            let error = IndexChanges::from_records(&catalog, &records, &context)
                .err()
                .unwrap();
            assert_eq!(error.code(), ErrorCode::ResourceExhausted);
            for (key, value) in [
                ("resource", "memory".to_string()),
                ("operation", "catalog_build".into()),
                ("limit_bytes", limit.to_string()),
                ("requested_bytes", (limit + 1).to_string()),
            ] {
                assert_eq!(error_detail_value(&error, key), Some(value.as_str()));
            }
            held.grow_to(limit, limit, "operation_scratch").unwrap();
        }
    }

    #[test]
    fn raw_encoding_admission_rejects_before_catalog_or_critical_writes() {
        let path = temp_path("raw-encoding-admission");
        let db = Database::create(&path, DatabaseConfig::default()).unwrap();
        let collection = db
            .create_collection(CollectionConfig {
                name: "notes".into(),
                tokenizer: Default::default(),
                embedding_dim: 2,
                distance_metric: DistanceMetric::Cosine,
                embedding_model: None,
                vectors_normalized: false,
                vector_index: VectorIndexConfig::Exact,
            })
            .unwrap();
        let document = |id: usize, metadata| {
            RecordPayload::DocumentPut(Document {
                document_id: format!("doc_{id}"),
                collection_id: collection.collection_id().into(),
                external_id: format!("document-{id}"),
                title: None,
                source_uri: None,
                metadata,
                created_at: "2026-10-09T00:00:00Z".into(),
                updated_at: "2026-10-09T00:00:00Z".into(),
            })
        };
        let context = db.json_context();
        let limit = crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES;
        let capacity = context.available_workspace_bytes();
        let original = fs::read(&path).unwrap();
        let mut inner = db.lock().unwrap();
        let signature = inner.catalog.state_signature();
        let commit = inner.catalog.last_commit_id;
        for seed in 0..100 {
            let value = match seed % 3 {
                0 => JsonValue::String("budgeted".repeat(seed + 1)),
                1 => {
                    let mut value = JsonValue::Null;
                    for _ in 0..128 + seed % 5 {
                        value = JsonValue::Array(vec![value]);
                    }
                    value
                }
                _ => JsonValue::Number(f64::NAN),
            };
            let mut batch = OwnedRecordBatch::new(&context);
            for record in [
                document(1, JsonValue::object()),
                document(2, object_from_pairs([("value", value)])),
            ] {
                if seed % 3 == 0 {
                    let RecordPayload::DocumentPut(value) = &record else {
                        unreachable!()
                    };
                    batch.retain_json(
                        value
                            .metadata
                            .guard_in(crate::json::JsonInputKind::DocumentMetadata, &context)
                            .unwrap()
                            .0,
                    );
                }
                batch.push(record).unwrap();
            }
            let mut held = context.workspace_reservation();
            if seed % 3 == 0 {
                let available = held.grow_to(0, limit, "operation_scratch").unwrap();
                held.retain(available - seed % 64);
            }
            let error = inner.commit_records_in(batch).err().unwrap();
            assert_eq!(error.code(), ErrorCode::InvalidJsonlSchema);
            if seed % 3 != 2 {
                assert_eq!(
                    error_detail_value(&error, "json_input_kind"),
                    Some("document_metadata")
                );
                assert_eq!(
                    error_detail_value(&error, "json_limit_reason"),
                    Some(if seed % 3 == 0 {
                        "work_budget_exceeded"
                    } else {
                        "depth_exceeded"
                    })
                );
            }
            assert_eq!(fs::read(&path).unwrap(), original);
            assert_eq!(inner.catalog.state_signature(), signature);
            assert_eq!(inner.catalog.last_commit_id, commit);
            assert_eq!(
                inner.index_suffix_status,
                IndexSuffixState::PersistentCheckpoint
            );
            inner.storage.require_writer().unwrap();
            held.grow_to(capacity, capacity, "operation_scratch")
                .unwrap();
        }
        let catalog_bytes = inner.catalog.reserved_bytes();
        for seed in 0..100 {
            let mut value = JsonValue::String("owned".repeat(seed + 1));
            for _ in 0..seed {
                value = JsonValue::Array(vec![value]);
            }
            let mut batch = OwnedRecordBatch::new(&context);
            batch
                .push(document(1, object_from_pairs([("value", value)])))
                .unwrap();
            let mut batch = batch.prepare_encoding(|| false).unwrap();
            let mut held = context.workspace_reservation();
            {
                let inner = &mut *inner;
                let mut transaction = inner.catalog.transaction(&context);
                let (records, json) = batch.records_and_json();
                assert!(json.reserved_bytes() > 0);
                transaction.prepare(records.iter(), json).unwrap();
                assert_eq!(json.reserved_bytes(), 0);
                // Exhaust the shared workspace after Catalog has taken source
                // ownership, but before the serializer is admitted for append.
                held.grow_to(0, limit, "operation_scratch").unwrap();
                let error = inner
                    .storage
                    .append_owned_batch_in(batch, inner.config.durability, || false)
                    .err()
                    .unwrap();
                assert_eq!(error.code(), ErrorCode::InvalidJsonlSchema);
                assert_eq!(
                    error_detail_value(&error, "json_input_kind"),
                    Some("document_metadata")
                );
                assert_eq!(
                    error_detail_value(&error, "json_limit_reason"),
                    Some("work_budget_exceeded")
                );
                drop(transaction);
            }
            assert_eq!(inner.catalog.reserved_bytes(), catalog_bytes);
            assert_eq!(inner.catalog.state_signature(), signature);
            assert_eq!(inner.catalog.last_commit_id, commit);
            assert_eq!(fs::read(&path).unwrap(), original);
            assert_eq!(
                inner.index_suffix_status,
                IndexSuffixState::PersistentCheckpoint
            );
            inner.storage.require_writer().unwrap();
            held.grow_to(capacity, capacity, "operation_scratch")
                .unwrap();
        }
        let pending = inner
            .commit_records(vec![
                document(1, object_from_pairs([("value", JsonValue::Bool(true))])),
                document(2, object_from_pairs([("value", JsonValue::Bool(false))])),
            ])
            .unwrap();
        inner.finish_index_update(pending, &context).unwrap();
        assert_eq!(inner.catalog.last_commit_id, commit + 1);
        drop(inner);
        let reopened = Database::open_read_only(&path).unwrap();
        assert_eq!(
            reopened
                .collection("notes")
                .unwrap()
                .stats()
                .unwrap()
                .document_count,
            2
        );
        drop((reopened, collection, db));
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn index_change_admission_precedes_raw_and_prepared_batch_writes() {
        let path = temp_path("index-change-admission");
        let db = Database::create(&path, DatabaseConfig::default()).unwrap();
        let collection = db
            .create_collection(CollectionConfig {
                name: "notes".into(),
                tokenizer: Default::default(),
                embedding_dim: 2,
                distance_metric: DistanceMetric::Cosine,
                embedding_model: None,
                vectors_normalized: false,
                vector_index: VectorIndexConfig::Exact,
            })
            .unwrap();
        let record = || RecordPayload::CollectionDrop {
            collection_id: collection.collection_id().into(),
            name: "notes".into(),
            dropped_at: "2026-10-08T00:00:00Z".into(),
            reason: None,
        };
        let context = db.json_context();
        let limit = crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES;
        let original = fs::read(&path).unwrap();
        let mut inner = db.lock().unwrap();
        let signature = inner.catalog.state_signature();
        let capacity = context.available_workspace_bytes();
        let tracker = IndexChanges::from_records(&inner.catalog, [&record()], &context).unwrap();
        let required = tracker.work.reserved_bytes();
        drop(tracker);
        assert_eq!(context.available_workspace_bytes(), capacity);
        for seed in 0..100 {
            for prepared_input in [false, true] {
                let mut held = context.workspace_reservation();
                let leave_room = |held: &mut crate::json::WorkspaceReservation| {
                    let available = held.grow_to(0, limit, "operation_scratch").unwrap();
                    held.retain(available - (required - 1 - seed));
                };
                let error = if prepared_input {
                    let mut builder = crate::storage::PreparedRecordsBuilder::new(
                        path.parent().unwrap(),
                        1_048_576,
                        context.clone(),
                    )
                    .unwrap();
                    builder.push(record()).unwrap();
                    let mut prepared = builder.finish().unwrap();
                    let batch = prepared.take_batch(1).unwrap();
                    leave_room(&mut held);
                    inner.commit_prepared_batch(batch).err().unwrap()
                } else {
                    let batch = OwnedRecordBatch::single(record(), &context).unwrap();
                    leave_room(&mut held);
                    inner.commit_records_in(batch).err().unwrap()
                };
                assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                assert_eq!(
                    error_detail_value(&error, "operation"),
                    Some("catalog_build")
                );
                assert_eq!(error_detail_value(&error, "resource"), Some("memory"));
                assert_eq!(fs::read(&path).unwrap(), original);
                assert_eq!(inner.catalog.state_signature(), signature);
                assert_eq!(
                    inner.index_suffix_status,
                    IndexSuffixState::PersistentCheckpoint
                );
                held.grow_to(capacity, capacity, "operation_scratch")
                    .unwrap();
            }
        }
        drop(inner);
        db.drop_collection("notes").unwrap();
        assert_eq!(db.stats().unwrap().collection_count, 0);
        drop((collection, db));
        let reopened = Database::open_read_only(&path).unwrap();
        assert_eq!(reopened.stats().unwrap().collection_count, 0);
        drop(reopened);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn deferred_change_union_rejection_preserves_committed_prefix() {
        let path = temp_path("deferred-index-change-admission");
        let db = Database::create(&path, DatabaseConfig::default()).unwrap();
        let mut ids = Vec::new();
        for name in ["first", "second"] {
            let collection = db
                .create_collection(CollectionConfig {
                    name: name.into(),
                    tokenizer: Default::default(),
                    embedding_dim: 2,
                    distance_metric: DistanceMetric::Cosine,
                    embedding_model: None,
                    vectors_normalized: false,
                    vector_index: VectorIndexConfig::Exact,
                })
                .unwrap();
            ids.push(collection.collection_id().to_owned());
        }
        let record = |index: usize| {
            RecordPayload::DocumentPut(Document {
                document_id: format!("doc_{:016}", index + 1),
                collection_id: ids[index].clone(),
                external_id: "new".into(),
                title: None,
                source_uri: None,
                metadata: JsonValue::object(),
                created_at: "2026-10-08T00:00:00Z".into(),
                updated_at: "2026-10-08T00:00:00Z".into(),
            })
        };
        let context = db.json_context();
        let limit = crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES;
        let prepare = || {
            let mut builder = crate::storage::PreparedRecordsBuilder::new(
                path.parent().unwrap(),
                1_048_576,
                context.clone(),
            )
            .unwrap();
            builder.push(record(1)).unwrap();
            builder.finish().unwrap()
        };
        let mut inner = db.lock().unwrap();
        let first = OwnedRecordBatch::single(record(0), &context).unwrap();
        let mut pending = inner.commit_records_in(first).unwrap();
        let prepared_text_bytes = inner.catalog.live_collections().count()
            * std::mem::size_of::<crate::storage::TextExtent>()
            + crate::temporary::test_file_management_bytes(inner.storage.path().parent().unwrap());
        let original = fs::read(&path).unwrap();
        let commit = inner.catalog.last_commit_id;
        let signature = inner.catalog.state_signature();
        let next = IndexChanges::from_records(&inner.catalog, [&record(1)], &context).unwrap();
        let next_bytes = next.work.reserved_bytes();
        drop(next);
        for seed in 0..100 {
            let mut cumulative = PendingIndexUpdate {
                reusable_checkpoint: true,
                reusable_vectors: VectorCache::new(&context),
                reusable_text: None,
                changes: IndexChanges::from_records(&inner.catalog, [&record(0)], &context)
                    .unwrap(),
            };
            let mut prepared = prepare();
            let batch = prepared.take_batch(1).unwrap();
            let mut held = context.workspace_reservation();
            let available = held.grow_to(0, limit, "operation_scratch").unwrap();
            // One more key fits in each existing tree node. Only the two new
            // strings need additional cumulative credit after the current set.
            held.retain(available - next_bytes - 2 * ids[1].len() + 1 + seed % 20);
            let error = inner
                .commit_prepared_batch_tracking(batch, Some(&mut cumulative))
                .err()
                .unwrap();
            assert_eq!(error.code(), ErrorCode::ResourceExhausted);
            assert_eq!(
                error_detail_value(&error, "operation"),
                Some("catalog_build")
            );
            assert!(cumulative.changes.text.contains(&ids[1]));
            assert!(!cumulative.changes.vector.contains(&ids[1]));
            assert_eq!(fs::read(&path).unwrap(), original);
            assert_eq!(inner.catalog.state_signature(), signature);
            assert_eq!(inner.catalog.last_commit_id, commit);
            drop(cumulative);
            assert_eq!(
                held.grow_to(0, limit, "operation_scratch").unwrap(),
                limit
                    - pending.changes.work.reserved_bytes()
                    - prepared_text_bytes
                    - 2 * crate::temporary::test_file_management_bytes(path.parent().unwrap())
            );
            drop(prepared);
            assert_eq!(
                held.grow_to(0, limit, "operation_scratch").unwrap(),
                limit - pending.changes.work.reserved_bytes() - prepared_text_bytes
            );
        }
        let mut prepared = prepare();
        let next = inner
            .commit_prepared_batch_tracking(prepared.take_batch(1).unwrap(), Some(&mut pending))
            .unwrap();
        assert!(!next.reusable_checkpoint);
        drop(next);
        assert_eq!(inner.catalog.last_commit_id, commit + 1);
        inner.finish_index_update(pending, &context).unwrap();
        drop(inner);
        let reopened = Database::open_read_only(&path).unwrap();
        for name in ["first", "second"] {
            assert!(reopened
                .collection(name)
                .unwrap()
                .get_document_by_external_id("new")
                .is_ok());
        }
        drop((reopened, prepared, db));
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn generated_write_field_refusals_precede_commit_and_leave_objects_live() {
        let path = temp_path("generated-write-admission");
        let db = Database::create(&path, DatabaseConfig::default()).unwrap();
        let config = |name| CollectionConfig {
            name,
            tokenizer: Default::default(),
            embedding_dim: 2,
            distance_metric: DistanceMetric::Cosine,
            embedding_model: None,
            vectors_normalized: false,
            vector_index: VectorIndexConfig::Exact,
        };
        let collection = db.create_collection(config("notes".into())).unwrap();
        collection
            .insert_document(DocumentInput {
                external_id: "doc".into(),
                title: None,
                source_uri: None,
                metadata: JsonValue::object(),
                chunks: vec![ChunkInput {
                    chunk_external_id: "chunk".into(),
                    text: "searchable".into(),
                    ordinal: 0,
                    metadata: JsonValue::object(),
                    source_span: None,
                }],
            })
            .unwrap();
        let capacity = db.json_context().available_workspace_bytes();
        let document = collection.get_document_by_external_id("doc").unwrap();
        let chunk = collection.get_chunk_by_external_id("chunk").unwrap();
        let original = fs::read(&path).unwrap();
        let commit = db.stats().unwrap().last_commit_id;
        let limit = crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES;
        for seed in 0..100 {
            let context = db.json_context();
            let mut held = context.workspace_reservation();
            held.grow_to(capacity - seed, capacity - seed, "operation_scratch")
                .unwrap();
            for error in [
                db.drop_collection("notes").unwrap_err(),
                db.delete_document_in_collection(collection.collection_id(), &document.document_id)
                    .unwrap_err(),
                db.delete_chunk_in_collection(collection.collection_id(), &chunk.chunk_id)
                    .unwrap_err(),
            ] {
                assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                assert_eq!(
                    error_detail_value(&error, "operation"),
                    Some("batch_prepare")
                );
                assert_eq!(error_detail_value(&error, "resource"), Some("memory"));
                assert_eq!(
                    error_detail_value(&error, "limit_bytes"),
                    Some(limit.to_string().as_str())
                );
                assert!(
                    error_detail_value(&error, "requested_bytes")
                        .unwrap()
                        .parse::<u128>()
                        .unwrap()
                        > limit as u128
                );
            }
            held.grow_to(capacity, capacity, "operation_scratch")
                .unwrap();
            held.retain(0);
            let input = config("x".repeat(64 * 1024 + seed));
            let validation = input
                .validate_in(&crate::json::JsonOperationContext::standalone())
                .unwrap();
            // Validation also needs its temporary traversal stack, which is not
            // retained by the returned lease. Leave that room before testing the copy.
            let occupied = capacity - validation.reserved_bytes() - 8192;
            held.grow_to(occupied, occupied, "operation_scratch")
                .unwrap();
            let error = db.create_collection(input).err().unwrap();
            assert_eq!(error.code(), ErrorCode::ResourceExhausted, "{error}");
            assert_eq!(
                error_detail_value(&error, "operation"),
                Some("batch_prepare")
            );
            held.grow_to(capacity, capacity, "operation_scratch")
                .unwrap();
            assert_eq!(fs::read(&path).unwrap(), original);
            assert_eq!(db.stats().unwrap().last_commit_id, commit);
        }
        assert_eq!(
            collection
                .search()
                .text("searchable")
                .execute()
                .unwrap()
                .results
                .len(),
            1
        );
        // Original config allocations move once; no second capacity-sized clone is needed.
        let mut name = String::with_capacity(31 * 1024 * 1024);
        name.push_str("large-capacity");
        let created = db.create_collection(config(name)).unwrap();
        assert_eq!(created.collection_id(), "col_0000000000000002");
        db.delete_chunk_in_collection(collection.collection_id(), &chunk.chunk_id)
            .unwrap();
        db.delete_document_in_collection(collection.collection_id(), &document.document_id)
            .unwrap();
        db.drop_collection("notes").unwrap();
        assert_eq!(db.stats().unwrap().collection_count, 1);
        drop((created, collection, db));
        let reopened = Database::open_read_only(&path).unwrap();
        assert_eq!(reopened.stats().unwrap().collection_count, 1);
        assert_eq!(reopened.stats().unwrap().chunk_count, 0);
        drop(reopened);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn collection_creation_uses_shared_json_budget_and_rejects_before_mutation() {
        let path = temp_path("collection-json-policy");
        let db = Database::create(&path, DatabaseConfig::default()).unwrap();
        let before = std::fs::read(&path).unwrap();
        let config = |name| CollectionConfig {
            name,
            tokenizer: Default::default(),
            embedding_dim: 2,
            distance_metric: crate::DistanceMetric::Cosine,
            embedding_model: None,
            vectors_normalized: false,
            vector_index: crate::VectorIndexConfig::Exact,
        };
        let error = db
            .create_collection(config(
                "x".repeat(crate::json::JsonProcessingPolicyV1::MAX_UNIT_CANONICAL_BYTES),
            ))
            .err()
            .unwrap();
        assert!(error
            .detail()
            .iter()
            .any(|(key, value)| key == "json_input_kind" && value == "collection_config"));
        let context = db.json_context();
        let mut held = context.reservation();
        held.grow(
            crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
            crate::json::JsonInputKind::Filter,
        )
        .unwrap();
        let error = db
            .clone()
            .create_collection(config("small".into()))
            .err()
            .unwrap();
        assert_eq!(
            error
                .to_json()
                .get("detail")
                .unwrap()
                .get("limit")
                .unwrap()
                .as_f64(),
            Some(0.0)
        );
        assert_eq!(std::fs::read(&path).unwrap(), before);
        assert!(db.list_collections().unwrap().is_empty());
        drop(held);
        db.create_collection(config("small".into())).unwrap();
        drop(db);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn commit_and_recovery_move_metadata_into_catalog_without_copies() {
        let path = temp_path("commit-owned-metadata");
        let db = Database::create(&path, DatabaseConfig::default()).unwrap();
        let col = db
            .create_collection(CollectionConfig {
                name: "notes".into(),
                embedding_dim: 2,
                distance_metric: DistanceMetric::Cosine,
                embedding_model: None,
                vectors_normalized: false,
                vector_index: VectorIndexConfig::Exact,
                tokenizer: Default::default(),
            })
            .unwrap();
        let metadata = |value: String| object_from_pairs([("value", JsonValue::String(value))]);
        let document_metadata = metadata("document-value".into());
        let document_pointer = document_metadata
            .get("value")
            .unwrap()
            .as_str()
            .unwrap()
            .as_ptr();
        let mut pointers = Vec::new();
        let chunks = (0..100)
            .map(|ordinal| {
                let metadata = metadata(format!("chunk-{ordinal}-{}", "x".repeat(4096 + ordinal)));
                pointers.push(metadata.get("value").unwrap().as_str().unwrap().as_ptr());
                ChunkInput {
                    chunk_external_id: format!("chunk-{ordinal}"),
                    text: "owned payload".into(),
                    ordinal,
                    metadata,
                    source_span: None,
                }
            })
            .collect();
        let summary = col
            .insert_document(DocumentInput {
                external_id: "doc".into(),
                title: None,
                source_uri: None,
                metadata: document_metadata,
                chunks,
            })
            .unwrap();
        assert_eq!(
            (summary.documents_inserted, summary.chunks_inserted),
            (1, 100)
        );
        {
            let inner = db.lock().unwrap();
            let document = inner
                .catalog
                .document_by_external_id(col.collection_id(), "doc")
                .unwrap();
            assert_eq!(
                document
                    .metadata
                    .get("value")
                    .unwrap()
                    .as_str()
                    .unwrap()
                    .as_ptr(),
                document_pointer
            );
            for (ordinal, pointer) in pointers.into_iter().enumerate() {
                let chunk = inner
                    .catalog
                    .chunk_by_external_id(col.collection_id(), &format!("chunk-{ordinal}"))
                    .unwrap();
                assert!(chunk.text.is_empty());
                assert_eq!(
                    chunk
                        .metadata
                        .get("value")
                        .unwrap()
                        .as_str()
                        .unwrap()
                        .as_ptr(),
                    pointer
                );
            }
        }
        drop((col, db));
        let scan = scan_file(&path).unwrap();
        let pointers = scan
            .records
            .iter()
            .filter_map(|record| match &record.payload {
                RecordPayload::ChunkPut(chunk) => Some((
                    chunk.chunk_id.clone(),
                    chunk
                        .metadata
                        .get("value")
                        .unwrap()
                        .as_str()
                        .unwrap()
                        .as_ptr(),
                )),
                _ => None,
            })
            .collect::<Vec<_>>();
        let catalog =
            Catalog::from_committed_records(scan.records, scan.last_valid_commit).unwrap();
        for (id, pointer) in pointers {
            assert_eq!(
                catalog.chunks[&id]
                    .metadata
                    .get("value")
                    .unwrap()
                    .as_str()
                    .unwrap()
                    .as_ptr(),
                pointer
            );
        }
        drop(catalog);
        let db = Database::open_read_only(&path).unwrap();
        let col = db.collection("notes").unwrap();
        assert_eq!(
            col.get_document_by_external_id("doc")
                .unwrap()
                .metadata
                .get("value")
                .unwrap()
                .as_str(),
            Some("document-value")
        );
        let chunks = col.list_chunks_by_document_external_id("doc").unwrap();
        assert_eq!(chunks.len(), 100);
        for (ordinal, chunk) in chunks.iter().enumerate() {
            assert_eq!(chunk.text, "owned payload");
            assert_eq!(
                chunk.metadata.get("value").unwrap().as_str().unwrap(),
                format!("chunk-{ordinal}-{}", "x".repeat(4096 + ordinal))
            );
        }
        drop((col, db));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn catalog_rejection_precedes_publication_and_preserves_retry_check_and_compact() {
        for seed in 0..100 {
            let path = temp_path("catalog-budget-publication");
            let db = Database::create(&path, DatabaseConfig::default()).unwrap();
            let collection = db
                .create_collection(CollectionConfig {
                    name: "notes".into(),
                    embedding_dim: 2,
                    distance_metric: DistanceMetric::Cosine,
                    embedding_model: None,
                    vectors_normalized: false,
                    vector_index: VectorIndexConfig::Exact,
                    tokenizer: Default::default(),
                })
                .unwrap();
            let input = || DocumentInput {
                external_id: "new".into(),
                title: None,
                source_uri: None,
                chunks: Vec::new(),
                metadata: object_from_pairs([("payload", JsonValue::String("x".repeat(seed + 1)))]),
            };
            let context = db.json_context();
            let bytes = input()
                .metadata
                .retained_bytes_in(crate::json::JsonInputKind::StoredJson, &context)
                .unwrap();
            let live = db.lock().unwrap().catalog.reserved_bytes();
            let signature = db.lock().unwrap().catalog.state_signature();
            let original = fs::read(&path).unwrap();
            let limit = crate::json::WorkLedger::CATALOG_LIMIT_BYTES;
            let mut held = context.catalog_reservation();
            held.grow(context.available_catalog_bytes()).unwrap();
            let error = collection.insert_document(input()).unwrap_err();
            assert_eq!(error.code(), ErrorCode::ResourceExhausted);
            for (key, value) in [
                ("resource", "catalog".to_string()),
                ("operation", "catalog_build".into()),
                ("limit_bytes", limit.to_string()),
                ("requested_bytes", (limit + bytes).to_string()),
            ] {
                assert!(
                    error
                        .detail()
                        .iter()
                        .any(|entry| entry == &(key.to_string(), value.clone())),
                    "{error:?}"
                );
            }
            assert_eq!(
                db.check(CheckOptions::default()).unwrap_err().code(),
                ErrorCode::ResourceExhausted
            );
            assert_eq!(
                db.compact(CompactOptions {
                    rebuild_indexes: false
                })
                .unwrap_err()
                .code(),
                ErrorCode::ResourceExhausted
            );
            assert_eq!(fs::read(&path).unwrap(), original);
            assert_eq!(db.lock().unwrap().catalog.state_signature(), signature);
            assert_eq!(db.lock().unwrap().catalog.reserved_bytes(), live);
            drop(held);
            assert_eq!(
                collection
                    .insert_document(input())
                    .unwrap()
                    .documents_inserted,
                1
            );
            let signature = db.lock().unwrap().catalog.state_signature();
            let reopened = Database::open_read_only(&path).unwrap();
            assert_eq!(
                reopened.lock().unwrap().catalog.state_signature(),
                signature
            );
            drop(reopened);
            drop(collection);
            drop(db);
            fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn optional_vector_prefetch_does_not_strand_catalog_commit_memory() {
        let original = query_preflight_fixture("prefetch-commit-source", "stable content");
        let source_bytes = fs::read(&original).unwrap();
        let mut reclaimed = 0;
        for prepared_input in [false, true] {
            for seed in 0..100usize {
                let available = 4096 + seed * 256 + usize::from(prepared_input) * 65536;
                let mut outcomes = Vec::new();
                for prefetch in [false, true] {
                    let path = temp_path("prefetch-commit-budget");
                    fs::copy(&original, &path).unwrap();
                    let db = Database::open_with_options(
                        &path,
                        DatabaseOpenOptions {
                            cache_budget_mb: Some(4096),
                            ..Default::default()
                        },
                    )
                    .unwrap();
                    let context = db.json_context();
                    let mut inner = db.lock().unwrap();
                    if !prefetch {
                        // Disable only optional vectors; both arms retain the same necessary text source.
                        inner.config.cache_budget_mb = Some(0);
                    }
                    let signature = inner.catalog.state_signature();
                    let catalog_bytes = inner.catalog.reserved_bytes();
                    let initial_commit = inner.catalog.last_commit_id;
                    let mut metadata = JsonValue::String("x".repeat(100 + seed));
                    for _ in 0..seed % 32 {
                        metadata = JsonValue::Array(vec![metadata]);
                    }
                    let document = Document {
                        document_id: inner.catalog.next_id("doc_", 0).unwrap(),
                        collection_id: inner
                            .catalog
                            .collection_by_name("first")
                            .unwrap()
                            .collection_id
                            .clone(),
                        external_id: "new-document".into(),
                        title: None,
                        source_uri: None,
                        metadata: object_from_pairs([("payload", metadata)]),
                        created_at: "1970-01-01T00:00:00Z".into(),
                        updated_at: "1970-01-01T00:00:00Z".into(),
                    };
                    let mut prepared = prepared_input.then(|| {
                        crate::storage::PreparedRecords::create(
                            vec![RecordPayload::DocumentPut(document.clone())],
                            path.parent().unwrap(),
                            1_048_576,
                        )
                        .unwrap()
                    });
                    let expected_id = document.document_id.clone();
                    let expected_metadata = document.metadata.clone();
                    let blocker = context
                        .cache_reservation(
                            (context.available_total_bytes() - available) as u128,
                            usize::MAX,
                        )
                        .unwrap();
                    let result = match &mut prepared {
                        Some(prepared) => {
                            inner.commit_prepared_batch(prepared.take_batch(1).unwrap())
                        }
                        None => inner.commit_records(vec![RecordPayload::DocumentPut(document)]),
                    };
                    let success = result.is_ok();
                    let error = result.as_ref().err().map(|error| format!("{error:?}"));
                    if prefetch {
                        if let Ok(pending) = &result {
                            reclaimed += usize::from(pending.reusable_vectors.is_empty());
                            if seed == 99 {
                                assert!(!pending.reusable_vectors.is_empty());
                            }
                        }
                    }
                    drop(result);
                    drop(prepared);
                    if success {
                        assert_eq!(inner.catalog.last_commit_id, initial_commit + 1);
                        assert_eq!(
                            inner.catalog.documents[&expected_id].metadata,
                            expected_metadata
                        );
                    } else {
                        assert_eq!(inner.catalog.state_signature(), signature);
                        assert_eq!(inner.catalog.reserved_bytes(), catalog_bytes);
                        assert_eq!(fs::read(&path).unwrap(), source_bytes);
                    }
                    drop(blocker);
                    drop(inner);
                    drop(db);
                    let reopened = Database::open_read_only(&path).unwrap();
                    assert_eq!(
                        reopened.lock().unwrap().catalog.last_commit_id,
                        initial_commit + u64::from(success)
                    );
                    if success {
                        assert_eq!(
                            reopened.lock().unwrap().catalog.documents[&expected_id].metadata,
                            expected_metadata
                        );
                    }
                    drop(reopened);
                    fs::remove_file(path).unwrap();
                    outcomes.push((success, error));
                }
                assert_eq!(
                    outcomes[0].0, outcomes[1].0,
                    "prepared={prepared_input}, available={available}: {outcomes:?}"
                );
            }
        }
        assert!(reclaimed > 0);
        assert_eq!(fs::read(&original).unwrap(), source_bytes);
        fs::remove_file(original).unwrap();
    }

    #[test]
    #[cfg(unix)]
    fn temporary_file_admission_reclaims_runtime_caches_without_relocking_or_replaying_export() {
        let path = cached_vector_batch_fixture("temporary-runtime-reclaim");
        let original = fs::read(&path).unwrap();
        let db = Database::open_with_options(
            &path,
            DatabaseOpenOptions {
                cache_budget_mb: Some(4096),
                ..Default::default()
            },
        )
        .unwrap();
        let context = db.json_context();
        for mode in 0..3 {
            for gap in [1, 2, 127] {
                let output = temp_path(&format!("temporary-export-reclaim-{gap}"));
                if mode == 2 {
                    fs::create_dir(&output).unwrap();
                }
                let mut inner = db.lock().unwrap();
                inner.persistent_vector_segments = None;
                inner.lazy_vector_segment = None;
                if mode == 1 {
                    let id = inner
                        .catalog
                        .collection_by_name("cached")
                        .unwrap()
                        .collection_id
                        .clone();
                    inner.prepare_vector_index(&id, 4096).unwrap();
                    assert!(inner.lazy_vector_segment.is_some());
                } else {
                    let cached = inner.prefetch_unchanged_vector_segments(&BTreeSet::new());
                    assert!(!cached.is_empty());
                    inner.persistent_vector_segments = Some(cached);
                }
                let cache_bytes = inner.optional_cache_bytes();
                let directory = if mode == 2 {
                    output.as_path()
                } else {
                    inner.storage.path().parent().unwrap()
                };
                let management = crate::temporary::test_file_management_bytes(directory);
                let available = management + 128 - gap;
                let source_bytes = inner.lazy_vector_source.as_ref().unwrap().reserved_bytes();
                let free = context.available_total_bytes();
                let signature = inner.catalog.state_signature();
                let commit = inner.catalog.last_commit_id;
                let directory = directory.to_path_buf();
                let blocker = context
                    .cache_reservation((free - available) as u128, usize::MAX)
                    .unwrap();
                drop(inner);
                let disk = db.temporary_disk_budget();
                match mode {
                    0 => {
                        let owners = db
                            .admit_input_reclaiming(|| {
                                crate::temporary::create_budgeted_temporary_file_in_context(
                                    &directory,
                                    "import-input",
                                    "tmp",
                                    disk.clone(),
                                    &context,
                                    "operation_scratch",
                                )
                            })
                            .unwrap();
                        drop(owners);
                    }
                    1 => {
                        let owners = db
                            .with_read_snapshot_reclaiming(|catalog, storage, reclaim| {
                                assert_eq!(catalog.last_commit_id, commit);
                                assert_eq!(storage.path().parent().unwrap(), directory);
                                crate::temporary::create_budgeted_temporary_file_reclaiming(
                                    &directory,
                                    "export-documents",
                                    "tmp",
                                    disk.clone(),
                                    &context,
                                    "operation_scratch",
                                    reclaim,
                                )
                            })
                            .unwrap();
                        drop(owners);
                    }
                    _ => {
                        let exported = db.export_collection_jsonl("first", &output).unwrap();
                        assert_eq!(exported.documents, 1);
                        assert_eq!(exported.chunks, 2);
                        assert_eq!(exported.embeddings, 2);
                        for (name, count) in [
                            ("documents.jsonl", 1),
                            ("chunks.jsonl", 2),
                            ("embeddings.jsonl", 2),
                        ] {
                            assert_eq!(
                                fs::read_to_string(output.join(name))
                                    .unwrap()
                                    .lines()
                                    .count(),
                                count
                            );
                        }
                        assert_eq!(fs::read_dir(&output).unwrap().count(), 4);
                    }
                }
                let inner = db.lock().unwrap();
                assert!(inner.persistent_vector_segments.is_none());
                assert!(inner.lazy_vector_segment.is_none());
                assert_eq!(inner.catalog.state_signature(), signature);
                assert_eq!(inner.catalog.last_commit_id, commit);
                assert_eq!(
                    inner.index_suffix_status,
                    IndexSuffixState::PersistentCheckpoint
                );
                assert_eq!(
                    inner.lazy_vector_source.as_ref().unwrap().reserved_bytes(),
                    source_bytes
                );
                drop(inner);
                let expected = available + cache_bytes;
                let mut work = context.workspace_reservation();
                work.grow_to(expected, expected, "operation_scratch")
                    .unwrap();
                assert!(work
                    .grow_to(expected + 1, expected + 1, "operation_scratch")
                    .is_err());
                drop(work);
                drop(blocker);
                assert_eq!(fs::read(&path).unwrap(), original);
                if mode == 2 {
                    fs::remove_dir_all(output).unwrap();
                }
            }
        }
        drop(db);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn retained_vector_caches_do_not_strand_batch_preparation() {
        let source = query_preflight_fixture("retained-batch-source", "stable content");
        let original = fs::read(&source).unwrap();
        let mut reclaimed = [0; 3];
        for (mode, reclaimed) in reclaimed.iter_mut().enumerate() {
            for seed in 0..100usize {
                let available = 4096 + seed * 512 + usize::from(mode != 0) * 65536;
                let mut outcomes = Vec::new();
                for keep_cache in [false, true] {
                    let path = temp_path("retained-batch-budget");
                    fs::copy(&source, &path).unwrap();
                    let db = Database::open_with_options(
                        &path,
                        DatabaseOpenOptions {
                            cache_budget_mb: Some(4096),
                            ..Default::default()
                        },
                    )
                    .unwrap();
                    let context = db.json_context();
                    let mut inner = db.lock().unwrap();
                    let record = |inner: &Inner, name: &str, id: &str| {
                        RecordPayload::DocumentPut(Document {
                            document_id: inner.catalog.next_id("doc_", 0).unwrap(),
                            collection_id: inner
                                .catalog
                                .collection_by_name(name)
                                .unwrap()
                                .collection_id
                                .clone(),
                            external_id: id.into(),
                            title: None,
                            source_uri: None,
                            metadata: object_from_pairs([(
                                "payload",
                                JsonValue::String("x".repeat(100 + seed)),
                            )]),
                            created_at: "1970-01-01T00:00:00Z".into(),
                            updated_at: "1970-01-01T00:00:00Z".into(),
                        })
                    };
                    inner.text_index.clear_cache();
                    let mut pending = if mode == 2 {
                        let first = record(&inner, "first", "committed-prefix");
                        let mut pending = inner.commit_records(vec![first]).unwrap();
                        assert!(!pending.reusable_vectors.is_empty());
                        if !keep_cache {
                            pending.reusable_vectors.clear();
                        }
                        Some(pending)
                    } else {
                        let cache = inner.prefetch_unchanged_vector_segments(&BTreeSet::new());
                        assert!(!cache.is_empty());
                        if keep_cache {
                            inner.persistent_vector_segments = Some(cache);
                        }
                        // Only retained caches differ between these two runs.
                        inner.lazy_vector_source = None;
                        None
                    };
                    let signature = inner.catalog.state_signature();
                    let catalog_bytes = inner.catalog.reserved_bytes();
                    let commit = inner.catalog.last_commit_id;
                    let before = fs::read(&path).unwrap();
                    let next = record(&inner, "second", "next-batch");
                    let mut prepared = (mode != 0).then(|| {
                        crate::storage::PreparedRecords::create(
                            vec![next.clone()],
                            path.parent().unwrap(),
                            1_048_576,
                        )
                        .unwrap()
                    });
                    let blocker = context
                        .cache_reservation(
                            (3 * 1024 * 1024 * 1024usize - catalog_bytes - available) as u128,
                            usize::MAX,
                        )
                        .unwrap();
                    let result = match &mut prepared {
                        Some(prepared) => inner.commit_prepared_batch_tracking(
                            prepared.take_batch(1).unwrap(),
                            pending.as_mut(),
                        ),
                        None => inner.commit_records(vec![next]),
                    };
                    let success = result.is_ok();
                    let error = result.as_ref().err().map(|error| format!("{error:?}"));
                    if let Ok(next) = &result {
                        let cache = pending
                            .as_ref()
                            .map_or(&next.reusable_vectors, |pending| &pending.reusable_vectors);
                        if keep_cache {
                            *reclaimed += usize::from(cache.is_empty());
                            if seed == 99 {
                                assert!(!cache.is_empty());
                            }
                        }
                        assert_eq!(inner.catalog.last_commit_id, commit + 1);
                    } else {
                        assert_eq!(fs::read(&path).unwrap(), before);
                        assert_eq!(inner.catalog.state_signature(), signature);
                        assert_eq!(inner.catalog.reserved_bytes(), catalog_bytes);
                    }
                    drop(blocker);
                    if let Ok(next) = result {
                        inner
                            .finish_index_update(pending.take().unwrap_or(next), &context)
                            .unwrap();
                    }
                    let expected = inner.catalog.state_signature();
                    drop((prepared, pending, inner));
                    drop(db);
                    let reopened = Database::open_read_only(&path).unwrap();
                    assert_eq!(reopened.lock().unwrap().catalog.state_signature(), expected);
                    assert_eq!(
                        reopened.lock().unwrap().catalog.last_commit_id,
                        commit + u64::from(success)
                    );
                    if mode == 2 {
                        assert!(reopened
                            .collection("first")
                            .unwrap()
                            .get_document_by_external_id("committed-prefix")
                            .is_ok());
                    }
                    assert_eq!(
                        reopened
                            .collection("second")
                            .unwrap()
                            .get_document_by_external_id("next-batch")
                            .is_ok(),
                        success
                    );
                    drop(reopened);
                    fs::remove_file(path).unwrap();
                    outcomes.push((success, error));
                }
                assert_eq!(
                    outcomes[0].0, outcomes[1].0,
                    "mode={mode}, available={available}: {outcomes:?}"
                );
            }
        }
        assert!(reclaimed.iter().all(|count| *count > 0), "{reclaimed:?}");
        assert_eq!(fs::read(&source).unwrap(), original);
        fs::remove_file(source).unwrap();
    }

    fn assert_text_stats_without_resident_postings(db: &Database) {
        let mut inner = db.lock().unwrap();
        let rebuilt = (!inner.text_index.has_resident_cache()).then(|| {
            let mut reader = inner.storage.payload_reader().unwrap();
            let mut bytes = 0;
            TextIndex::try_build_matching(
                &inner.catalog.visible_snapshot(),
                |_| true,
                |visible| {
                    read_visible_chunk_text(&mut reader, visible, &db.json_context(), &mut bytes)
                },
            )
            .unwrap()
        });
        let expected = rebuilt.as_ref().unwrap_or(&inner.text_index);
        let total = expected.term_count();
        let collections = inner
            .catalog
            .live_collections()
            .map(|collection| {
                let terms = expected
                    .postings()
                    .values()
                    .filter(|postings| {
                        postings.iter().any(|posting| {
                            posting.collection_id.as_ref() == collection.collection_id
                        })
                    })
                    .count();
                (collection.collection_id.clone(), terms)
            })
            .collect::<Vec<_>>();
        let resident = std::mem::take(&mut inner.text_index);
        drop(inner);
        let stats = db.index_stats().unwrap();
        let database_stats = db.stats().unwrap();
        let scoped = collections
            .iter()
            .map(|(id, _)| db.collection_stats_by_id(id).unwrap().text_index_terms)
            .collect::<Vec<_>>();
        db.lock().unwrap().text_index = resident;
        assert!(!stats.rebuild_needed);
        assert_eq!(database_stats.index_size_bytes, stats.index_size_bytes);
        assert_eq!(stats.text_terms, total);
        for ((id, expected), actual) in collections.iter().zip(scoped) {
            assert_eq!(actual, *expected, "{id}");
        }
    }

    #[test]
    fn text_stats_remain_exact_when_postings_are_not_resident() {
        let path = cached_vector_batch_fixture("text-stats-no-postings");
        let db = Database::open(&path).unwrap();
        let reader = Database::open_read_only(&path).unwrap();
        let original = fs::read(&path).unwrap();
        assert_text_stats_without_resident_postings(&db);
        assert_text_stats_without_resident_postings(&reader);
        assert_eq!(fs::read(&path).unwrap(), original);
        let first = db.collection("first").unwrap();
        let input = |external_id: &str, text: &str| DocumentInput {
            external_id: external_id.into(),
            title: None,
            source_uri: None,
            metadata: JsonValue::object(),
            chunks: vec![ChunkInput {
                chunk_external_id: format!("{external_id}-chunk"),
                text: text.into(),
                ordinal: 0,
                metadata: JsonValue::object(),
                source_span: None,
            }],
        };
        first
            .insert_document(input("new", "fresh alpha shared"))
            .unwrap();
        assert_eq!(db.index_stats().unwrap().text_terms, 5);
        assert_text_stats_without_resident_postings(&db);
        assert_text_stats_without_resident_postings(&reader);
        assert_eq!(reader.index_stats().unwrap().text_terms, 2);
        db.insert_document_in_collection(
            first.collection_id(),
            input("new", "beta content"),
            OnConflict::Replace,
        )
        .unwrap();
        assert_eq!(db.index_stats().unwrap().text_terms, 3);
        assert_text_stats_without_resident_postings(&db);
        let commit = db.index_stats().unwrap().last_index_commit_id.unwrap();
        first
            .put_embeddings(vec![EmbeddingInput {
                chunk_external_id: "new-chunk".into(),
                model_name: "m".into(),
                dim: 2,
                vector: vec![0.0, 1.0],
            }])
            .unwrap();
        assert!(db.index_stats().unwrap().last_index_commit_id.unwrap() > commit);
        assert_text_stats_without_resident_postings(&db);
        let document = first.get_document_by_external_id("new").unwrap();
        first.delete_document(&document.document_id).unwrap();
        assert_eq!(db.index_stats().unwrap().text_terms, 2);
        assert_text_stats_without_resident_postings(&db);
        fail_next_index_checkpoint_refresh(&db);
        first
            .insert_document(input("after-failure", "failure visible"))
            .unwrap();
        assert!(db.index_stats().unwrap().rebuild_needed);
        assert_eq!(db.index_stats().unwrap().text_terms, 0);
        assert_eq!(db.index_stats().unwrap().last_index_commit_id, None);
        assert!(db.lock().unwrap().lazy_vector_source.is_none());
        assert!(first.get_document_by_external_id("after-failure").is_ok());
        db.rebuild_index(None).unwrap();
        assert_text_stats_without_resident_postings(&db);
        db.compact(CompactOptions {
            rebuild_indexes: true,
        })
        .unwrap();
        assert_text_stats_without_resident_postings(&db);
        assert_text_stats_without_resident_postings(&reader);
        assert_eq!(reader.index_stats().unwrap().text_terms, 2);
        db.compact(CompactOptions {
            rebuild_indexes: false,
        })
        .unwrap();
        assert!(db.index_stats().unwrap().rebuild_needed);
        assert_eq!(db.index_stats().unwrap().text_terms, 0);
        assert!(db.lock().unwrap().lazy_vector_source.is_none());
        assert_eq!(first.stats().unwrap().text_index_terms, 0);
        db.rebuild_index(None).unwrap();
        assert_text_stats_without_resident_postings(&db);
        drop((first, reader));
        drop(db);
        let reopened = Database::open_read_only(&path).unwrap();
        assert_text_stats_without_resident_postings(&reopened);
        drop(reopened);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn checkpoint_refresh_keeps_caller_operation_for_necessary_snapshot_owners() {
        let path = cached_vector_batch_fixture("checkpoint-original-operation");
        let db = Database::open(&path).unwrap();
        let context = db.json_context();
        let mut input = context.workspace_reservation();
        input.grow_to(17, 17, "operation_scratch").unwrap();
        let mut inner = db.lock().unwrap();
        let primary_end = inner.storage.critical_end_offset() as usize;
        let primary = fs::read(&path).unwrap()[..primary_end].to_vec();
        for reclaim in [0, 1, 2] {
            inner.persistent_vector_segments = None;
            inner.lazy_vector_segment = None;
            let reusable = if reclaim == 2 {
                inner.prefetch_unchanged_vector_segments(&BTreeSet::new())
            } else {
                VectorCache::new(&context)
            };
            let cache = reusable.reserved_bytes() + inner.text_index.reserved_bytes();
            inner.mark_index_checkpoint_rebuild_needed();
            let text = inner
                .prepare_text_checkpoint(None, &BTreeSet::new(), &context)
                .unwrap();
            let required = inner.catalog.live_collections().count()
                * (8 + std::mem::size_of::<&CollectionInfo>() + 3 * std::mem::size_of::<usize>());
            let blocker = if reclaim != 1 {
                assert!(cache > required);
                Some(
                    context
                        .cache_reservation(
                            (context.available_total_bytes() - required + 1) as u128,
                            usize::MAX,
                        )
                        .unwrap(),
                )
            } else {
                None
            };
            let calls = inner.index_checkpoint_refreshes;
            inner
                .refresh_index_checkpoint_prepared(reusable, &BTreeSet::new(), text, &context)
                .unwrap();
            assert_eq!(
                inner.index_checkpoint_refreshes - calls,
                if reclaim != 1 { 2 } else { 1 }
            );
            inner
                .lazy_vector_source
                .as_ref()
                .unwrap()
                .assert_operation_context(&context);
            assert_eq!(fs::read(&path).unwrap()[..primary_end], primary);
            drop(blocker);
        }
        let before = fs::read(&path).unwrap();
        inner.fail_next_index_checkpoint_refresh = true;
        let calls = inner.index_checkpoint_refreshes;
        assert_eq!(
            inner.refresh_index_checkpoint(&context).unwrap_err().code(),
            ErrorCode::IoError
        );
        assert_eq!(inner.index_checkpoint_refreshes - calls, 1);
        assert!(inner.lazy_vector_source.is_none());
        assert_eq!(fs::read(&path).unwrap(), before);
        let mut remaining = context.workspace_reservation();
        let available = crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES - 17;
        remaining
            .grow_to(available, available, "operation_scratch")
            .unwrap();
        assert!(remaining
            .grow_to(available + 1, available + 1, "operation_scratch")
            .is_err());
        drop((remaining, inner, input));
        drop(db);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn checkpoint_statistics_admission_reclaims_only_optional_cache_and_preserves_suffix_on_failure(
    ) {
        let source = cached_vector_batch_fixture("text-summary-admission-source");
        for mode in 0..3 {
            let path = temp_path(&format!("text-summary-admission-{mode}"));
            fs::copy(&source, &path).unwrap();
            let db = Database::open(&path).unwrap();
            let context = db.json_context();
            let mut inner = db.lock().unwrap();
            inner.text_index.clear_cache();
            let original = fs::read(&path).unwrap();
            let signature = inner.catalog.state_signature();
            let commit = inner.catalog.last_commit_id;
            let collections = inner.catalog.live_collections().count();
            let summary = collections * std::mem::size_of::<(u64, usize)>();
            let coverage = collections
                * (8 + std::mem::size_of::<&CollectionInfo>() + 3 * std::mem::size_of::<usize>());
            let read_directory =
                inner.lazy_vector_source.as_ref().unwrap().reserved_bytes() - summary;
            // Reader retains segment slots; writer also admits the metadata slot.
            assert_eq!(read_directory % (2 * collections), 0);
            let directory = read_directory / (2 * collections) * (2 * collections + 1);
            let reusable = if mode == 2 {
                inner.prefetch_unchanged_vector_segments(&BTreeSet::new())
            } else {
                VectorCache::new(&context)
            };
            let cache_bytes = reusable.reserved_bytes();
            if mode == 2 {
                assert!(cache_bytes > 32 * 1024);
            }
            inner.mark_index_checkpoint_rebuild_needed();
            let before_prepare = context.available_workspace_bytes();
            let text = inner
                .prepare_text_checkpoint(None, &BTreeSet::new(), &context)
                .unwrap();
            let prepared_bytes = before_prepare - context.available_workspace_bytes();
            let available = if mode == 1 {
                directory - 1
            } else {
                coverage - 1
            };
            let occupied = context.available_total_bytes() - available;
            let blocker = context
                .cache_reservation(occupied as u128, usize::MAX)
                .unwrap();
            let result =
                inner.refresh_index_checkpoint_prepared(reusable, &BTreeSet::new(), text, &context);
            assert_eq!(inner.catalog.state_signature(), signature);
            assert_eq!(inner.catalog.last_commit_id, commit);
            let free = if mode == 2 {
                result.unwrap().unwrap();
                assert!(!inner.index_stats().rebuild_needed);
                assert!(inner
                    .persistent_vector_segments
                    .as_ref()
                    .unwrap()
                    .is_empty());
                available + cache_bytes + prepared_bytes
                    - inner.lazy_vector_source.as_ref().unwrap().reserved_bytes()
            } else {
                let error = result.unwrap_err();
                assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                for (key, value) in [
                    ("resource", "memory".to_owned()),
                    ("operation", "operation_scratch".to_owned()),
                    ("limit_bytes", (3 * 1024 * 1024 * 1024usize).to_string()),
                    (
                        "requested_bytes",
                        (3 * 1024 * 1024 * 1024usize + 1).to_string(),
                    ),
                ] {
                    assert!(error.detail().contains(&(key.into(), value)), "{error:?}");
                }
                assert!(inner.index_stats().rebuild_needed);
                assert!(inner.lazy_vector_source.is_none());
                assert_eq!(fs::read(&path).unwrap(), original);
                available + prepared_bytes
            };
            let mut work = context.workspace_reservation();
            work.grow_to(free, free, "operation_scratch").unwrap();
            assert!(work
                .grow_to(free + 1, free + 1, "operation_scratch")
                .is_err());
            drop((work, blocker, inner));
            if mode != 2 {
                db.rebuild_index(None).unwrap();
            }
            assert_text_stats_without_resident_postings(&db);
            drop(db);
            fs::remove_file(path).unwrap();
        }
        fs::remove_file(source).unwrap();
    }

    #[test]
    fn checkpoint_write_buffer_yields_to_exact_necessary_streaming_budget() {
        let source = cached_vector_batch_fixture("checkpoint-writer-budget-source");
        for delta in [-1isize, 0, 1] {
            let path = temp_path(&format!("checkpoint-writer-budget-{delta}"));
            fs::copy(&source, &path).unwrap();
            let db = Database::open(&path).unwrap();
            let context = db.json_context();
            let mut inner = db.lock().unwrap();
            let original = fs::read(&path).unwrap();
            let primary_end = inner.storage.critical_end_offset() as usize;
            let signature = inner.catalog.state_signature();
            let commit = inner.catalog.last_commit_id;
            let collections = inner.catalog.live_collections().count();
            let summary = collections * std::mem::size_of::<(u64, usize)>();
            // Five u64 fields, one u32 and one u16, rounded to u64 alignment.
            let extent = (5 * 8 + 4 + 2usize).next_multiple_of(8);
            let directory = (2 * collections + 1) * extent;
            let coverage = collections
                * (8 + std::mem::size_of::<&CollectionInfo>() + 3 * std::mem::size_of::<usize>());
            let segment_peak = inner
                .catalog
                .live_collections()
                .map(|collection| {
                    let text = inner
                        .catalog
                        .live_chunks()
                        .filter(|chunk| chunk.collection_id == collection.collection_id)
                        .count()
                        .max(
                            inner
                                .text_index
                                .terms_in_collection(&collection.collection_id)
                                .map(|(_, postings)| postings.len())
                                .max()
                                .unwrap_or(0),
                        )
                        * std::mem::size_of::<&Chunk>();
                    let embeddings = inner
                        .catalog
                        .live_embeddings()
                        .filter(|embedding| embedding.collection_id == collection.collection_id);
                    let references = embeddings.clone().count()
                        * std::mem::size_of::<(&Embedding, Option<&crate::storage::RecordRef>)>();
                    let decoded = embeddings
                        .map(|embedding| {
                            assert!(inner
                                .catalog
                                .record_ref_for_embedding(&embedding.embedding_id)
                                .is_some());
                            assert!(embedding.vector.is_empty());
                            embedding.dim * std::mem::size_of::<f32>()
                                + embedding.embedding_id.len()
                                + embedding.chunk_id.len()
                                + embedding.collection_id.len()
                                + embedding.model_name.len()
                                + embedding.created_at.len()
                        })
                        .max()
                        .unwrap_or(0);
                    text.max(references + decoded)
                })
                .max()
                .unwrap();
            let minimum = coverage.max(directory + segment_peak);
            let available = (minimum as isize + delta) as usize;
            inner.text_index.clear_cache();
            inner.mark_index_checkpoint_rebuild_needed();
            let before_prepare = context.available_workspace_bytes();
            let text = inner
                .prepare_text_checkpoint(None, &BTreeSet::new(), &context)
                .unwrap();
            let prepared_bytes = before_prepare - context.available_workspace_bytes();
            let total = 3 * 1024 * 1024 * 1024usize;
            let blocker = context
                .cache_reservation(
                    (context.available_total_bytes() - available) as u128,
                    usize::MAX,
                )
                .unwrap();
            let calls = inner.index_checkpoint_refreshes;
            let result = inner.refresh_index_checkpoint_prepared(
                VectorCache::new(&context),
                &BTreeSet::new(),
                text,
                &context,
            );
            assert_eq!(inner.index_checkpoint_refreshes - calls, 2);
            assert_eq!(inner.catalog.state_signature(), signature);
            assert_eq!(inner.catalog.last_commit_id, commit);
            assert_eq!(
                fs::read(&path).unwrap()[..primary_end],
                original[..primary_end]
            );
            let retained = if delta < 0 {
                let error = result.unwrap_err();
                assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                for (key, value) in [
                    ("resource", "memory".to_owned()),
                    ("operation", "operation_scratch".to_owned()),
                    ("limit_bytes", total.to_string()),
                    ("requested_bytes", (total + 1).to_string()),
                ] {
                    assert!(error.detail().contains(&(key.into(), value)), "{error:?}");
                }
                assert!(inner.index_stats().rebuild_needed);
                assert!(inner.lazy_vector_source.is_none());
                0
            } else {
                assert_eq!(result.unwrap().unwrap().status, "persistent_checkpoint");
                assert!(!inner.index_stats().rebuild_needed);
                assert!(inner
                    .persistent_vector_segments
                    .as_ref()
                    .unwrap()
                    .is_empty());
                let source = inner.lazy_vector_source.as_ref().unwrap();
                source.assert_operation_context(&context);
                assert_eq!(source.reserved_bytes(), summary + directory);
                assert_eq!(fs::read(&path).unwrap(), original);
                summary + directory
            };
            let free = available + prepared_bytes - retained;
            let mut work = context.workspace_reservation();
            work.grow_to(free, free, "operation_scratch").unwrap();
            assert!(work
                .grow_to(free + 1, free + 1, "operation_scratch")
                .is_err());
            drop((work, blocker));
            let full = crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES - retained;
            let mut work = context.workspace_reservation();
            work.grow_to(full, full, "operation_scratch").unwrap();
            assert!(work
                .grow_to(full + 1, full + 1, "operation_scratch")
                .is_err());
            drop((work, inner));
            if delta < 0 {
                db.rebuild_index(None).unwrap();
            }
            assert_text_stats_without_resident_postings(&db);
            drop(db);
            let reopened = Database::open_read_only(&path).unwrap();
            assert_text_stats_without_resident_postings(&reopened);
            assert_eq!(
                reopened.lock().unwrap().catalog.state_signature(),
                signature
            );
            assert_eq!(reopened.lock().unwrap().catalog.last_commit_id, commit);
            assert_eq!(
                reopened
                    .collection("cached")
                    .unwrap()
                    .stats()
                    .unwrap()
                    .embedding_count,
                8
            );
            drop(reopened);
            fs::remove_file(path).unwrap();
        }
        fs::remove_file(source).unwrap();
    }

    fn cached_vector_batch_fixture(name: &str) -> PathBuf {
        let source = query_preflight_fixture(name, "stable content");
        {
            let db = Database::open(&source).unwrap();
            let collection = db
                .create_collection(CollectionConfig {
                    name: "cached".into(),
                    embedding_dim: 1024,
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
                    chunks: (0..8)
                        .map(|ordinal| ChunkInput {
                            chunk_external_id: format!("cached-{ordinal}"),
                            text: "stable content".into(),
                            ordinal,
                            metadata: JsonValue::object(),
                            source_span: None,
                        })
                        .collect(),
                })
                .unwrap();
            collection
                .put_embeddings(
                    (0..8)
                        .map(|ordinal| {
                            let mut vector = vec![0.0; 1024];
                            vector[0] = 1.0;
                            EmbeddingInput {
                                chunk_external_id: format!("cached-{ordinal}"),
                                model_name: "m".into(),
                                dim: 1024,
                                vector,
                            }
                        })
                        .collect(),
                )
                .unwrap();
        }
        source
    }

    #[test]
    fn raw_write_preparation_reclaims_optional_vectors_before_mutation() {
        let source = cached_vector_batch_fixture("raw-write-reclaim-source");
        let document_input = || DocumentInput {
            external_id: "new".into(),
            title: None,
            source_uri: None,
            metadata: JsonValue::object(),
            chunks: ["fresh", "chunk-1"]
                .into_iter()
                .enumerate()
                .map(|(ordinal, id)| ChunkInput {
                    chunk_external_id: id.into(),
                    text: "new content".into(),
                    ordinal,
                    metadata: JsonValue::object(),
                    source_span: None,
                })
                .collect(),
        };
        let embedding_input = || {
            ["free", "chunk-1"]
                .into_iter()
                .map(|id| EmbeddingInput {
                    chunk_external_id: id.into(),
                    model_name: "m".into(),
                    dim: 2,
                    vector: vec![1.0, 0.0],
                })
                .collect::<Vec<_>>()
        };
        {
            let db = Database::open(&source).unwrap();
            db.collection("first")
                .unwrap()
                .insert_document(DocumentInput {
                    external_id: "available".into(),
                    title: None,
                    source_uri: None,
                    metadata: JsonValue::object(),
                    chunks: vec![ChunkInput {
                        chunk_external_id: "free".into(),
                        text: "free embedding slot".into(),
                        ordinal: 0,
                        metadata: JsonValue::object(),
                        source_span: None,
                    }],
                })
                .unwrap();
        }
        let original = fs::read(&source).unwrap();
        for seed in 0..16usize {
            for case in 0..6 {
                let mut outcomes = Vec::new();
                for keep_cache in [false, true] {
                    let path = temp_path("raw-write-reclaim");
                    fs::copy(&source, &path).unwrap();
                    let db = Database::open_with_options(
                        &path,
                        DatabaseOpenOptions {
                            cache_budget_mb: Some(4096),
                            ..Default::default()
                        },
                    )
                    .unwrap();
                    let collection = db.collection("first").unwrap();
                    let document = collection.get_document_by_external_id("doc").unwrap();
                    let chunk = collection.get_chunk_by_external_id("chunk-1").unwrap();
                    let before_commit = db.stats().unwrap().last_commit_id;
                    let context = db.json_context();
                    let (catalog_bytes, cached_bytes) = {
                        let mut inner = db.lock().unwrap();
                        inner.text_index.clear_cache();
                        let cached = inner.prefetch_unchanged_vector_segments(&BTreeSet::new());
                        let bytes = cached.reserved_bytes();
                        assert!(bytes > 32 * 1024);
                        if keep_cache {
                            inner.persistent_vector_segments = Some(cached);
                        }
                        (inner.catalog.reserved_bytes(), bytes)
                    };
                    let config = CollectionConfig {
                        name: "x".repeat(64 * 1024 + seed),
                        embedding_dim: 2,
                        tokenizer: Default::default(),
                        distance_metric: DistanceMetric::Cosine,
                        embedding_model: None,
                        vectors_normalized: false,
                        vector_index: VectorIndexConfig::Exact,
                    };
                    let free = match case {
                        3 => {
                            config
                                .validate_in(&crate::json::JsonOperationContext::standalone())
                                .unwrap()
                                .reserved_bytes()
                                + 8192
                        }
                        4 => 2048 + seed * 16,
                        5 => 512 + seed * 8,
                        _ => seed,
                    };
                    let blocker = context
                        .cache_reservation(
                            (context.available_total_bytes()
                                - if keep_cache { 0 } else { cached_bytes }
                                - free) as u128,
                            usize::MAX,
                        )
                        .unwrap();
                    // Admission must already fit; only later preparation may reclaim cache.
                    match case {
                        3 => {
                            config.validate_in(&context).unwrap();
                        }
                        4 => {
                            let mut strings = context.workspace_reservation();
                            document_input()
                                .validate_owned_in(&context, &mut strings)
                                .unwrap();
                        }
                        5 => {
                            let input = embedding_input();
                            let bytes = input.capacity() * std::mem::size_of::<EmbeddingInput>()
                                + input
                                    .iter()
                                    .map(EmbeddingInput::workspace_bytes)
                                    .sum::<usize>();
                            context
                                .workspace_reservation()
                                .grow_to(bytes, bytes, "batch_prepare")
                                .unwrap();
                        }
                        _ => {}
                    }
                    let result = match case {
                        0 => db.drop_collection("first"),
                        1 => collection.delete_document(&document.document_id),
                        2 => collection.delete_chunk(&chunk.chunk_id),
                        3 => db.create_collection(config).map(|_| ()),
                        4 => collection.insert_document(document_input()).map(|_| ()),
                        _ => collection.put_embeddings(embedding_input()).map(|_| ()),
                    };
                    match case {
                        0..=2 => assert!(result.is_ok(), "seed={seed}, case={case}: {result:?}"),
                        3 => assert_eq!(
                            result.as_ref().unwrap_err().code(),
                            ErrorCode::ResourceExhausted
                        ),
                        _ => assert_eq!(
                            result.as_ref().unwrap_err().code(),
                            ErrorCode::DuplicateExternalId
                        ),
                    }
                    if result
                        .as_ref()
                        .is_err_and(crate::json::is_memory_budget_error)
                    {
                        assert!(
                            db.lock().unwrap().persistent_vector_segments.is_none(),
                            "seed={seed}, case={case}, cache={keep_cache}: {result:?}"
                        );
                    }
                    let stats = db.stats().unwrap();
                    if stats.last_commit_id == before_commit {
                        assert_eq!(fs::read(&path).unwrap(), original);
                        assert!(!db.index_stats().unwrap().rebuild_needed);
                        assert_eq!(db.lock().unwrap().catalog.reserved_bytes(), catalog_bytes);
                    }
                    outcomes.push((
                        result.err().map(|error| error.code()),
                        stats.last_commit_id,
                        stats.collection_count,
                        stats.document_count,
                        stats.chunk_count,
                        stats.embedding_count,
                    ));
                    drop(blocker);
                    context
                        .workspace_reservation()
                        .grow_to(
                            crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES
                                - snapshot_directory_bytes(&db),
                            crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES
                                - snapshot_directory_bytes(&db),
                            "operation_scratch",
                        )
                        .unwrap();
                    drop((collection, db));
                    let reopened = Database::open_read_only(&path).unwrap();
                    let actual = reopened.stats().unwrap();
                    assert_eq!(actual.last_commit_id, stats.last_commit_id);
                    assert_eq!(
                        (
                            actual.collection_count,
                            actual.document_count,
                            actual.chunk_count,
                            actual.embedding_count
                        ),
                        (
                            stats.collection_count,
                            stats.document_count,
                            stats.chunk_count,
                            stats.embedding_count
                        )
                    );
                    drop(reopened);
                    fs::remove_file(path).unwrap();
                }
                assert_eq!(outcomes[0], outcomes[1], "seed={seed}, case={case}");
            }
        }
        assert_eq!(fs::read(&source).unwrap(), original);
        fs::remove_file(source).unwrap();
    }

    #[test]
    fn public_reads_reclaim_optional_vectors_without_changing_results() {
        let source = cached_vector_batch_fixture("public-read-cache");
        let original = fs::read(&source).unwrap();
        for seed in 0..16usize {
            for case in 0..8 {
                for keep_cache in [false, true] {
                    let db = Database::open_with_options(
                        &source,
                        DatabaseOpenOptions {
                            read_only: seed & 2 != 0,
                            cache_budget_mb: Some(4096),
                            ..Default::default()
                        },
                    )
                    .unwrap();
                    let collection = db.collection("first").unwrap();
                    let document = collection.get_document_by_external_id("doc").unwrap();
                    let chunk = collection.get_chunk_by_external_id("chunk-1").unwrap();
                    let read = || -> Result<JsonValue> {
                        match case {
                            0 => db.collection("first").map(|handle| {
                                JsonValue::Array(vec![
                                    JsonValue::String(handle.collection_id().into()),
                                    JsonValue::String(handle.name().into()),
                                ])
                            }),
                            1 => db.list_collections().map(|values| {
                                JsonValue::Array(
                                    values.iter().map(CollectionInfo::to_json).collect(),
                                )
                            }),
                            2 => collection
                                .get_document(&document.document_id)
                                .map(|value| value.to_json()),
                            3 => collection
                                .get_document_by_external_id("doc")
                                .map(|value| value.to_json()),
                            4 => collection
                                .get_chunk(&chunk.chunk_id)
                                .map(|value| value.to_json()),
                            5 => collection
                                .get_chunk_by_external_id("chunk-1")
                                .map(|value| value.to_json()),
                            6 => collection.list_chunks(&document.document_id).map(|values| {
                                JsonValue::Array(values.iter().map(Chunk::to_json).collect())
                            }),
                            _ => collection.list_chunks_by_document_external_id("doc").map(
                                |values| {
                                    JsonValue::Array(values.iter().map(Chunk::to_json).collect())
                                },
                            ),
                        }
                    };
                    let expected = read().unwrap();
                    let context = db.json_context();
                    let (cached_bytes, signature) = {
                        let mut inner = db.lock().unwrap();
                        if seed % 2 == 0 {
                            inner.persistent_vector_segments =
                                Some(inner.prefetch_unchanged_vector_segments(&BTreeSet::new()));
                        } else {
                            let id = inner
                                .catalog
                                .collection_by_name("cached")
                                .unwrap()
                                .collection_id
                                .clone();
                            inner.prepare_vector_index(&id, 4096).unwrap();
                        }
                        let bytes = inner
                            .persistent_vector_segments
                            .as_ref()
                            .map_or(0, VectorCache::reserved_bytes)
                            + inner
                                .lazy_vector_segment
                                .as_ref()
                                .map_or(0, CachedVectorSegment::reserved_bytes);
                        assert!(bytes > 32 * 1024);
                        let signature = inner.catalog.state_signature();
                        let bytes = bytes + inner.text_index.reserved_bytes();
                        if !keep_cache {
                            inner.text_index.clear_cache();
                            inner.persistent_vector_segments = None;
                            inner.lazy_vector_segment = None;
                        }
                        (bytes, signature)
                    };
                    assert_eq!(read().unwrap(), expected);
                    if keep_cache {
                        let inner = db.lock().unwrap();
                        assert!(
                            inner.persistent_vector_segments.is_some()
                                || inner.lazy_vector_segment.is_some()
                        );
                    }
                    let available = cached_bytes + seed;
                    let blocker = context
                        .cache_reservation(
                            (context.available_total_bytes()
                                + if keep_cache { cached_bytes } else { 0 }
                                - available) as u128,
                            usize::MAX,
                        )
                        .unwrap();
                    let actual = read().unwrap_or_else(|error| {
                        panic!("seed={seed}, case={case}, cache={keep_cache}: {error:?}")
                    });
                    assert_eq!(actual, expected);
                    {
                        let inner = db.lock().unwrap();
                        assert!(inner.persistent_vector_segments.is_none());
                        assert!(inner.lazy_vector_segment.is_none());
                        assert!(!inner.text_index.has_resident_cache());
                        assert_eq!(inner.catalog.state_signature(), signature);
                        assert_eq!(
                            inner.index_suffix_status,
                            IndexSuffixState::PersistentCheckpoint
                        );
                    }
                    context
                        .workspace_reservation()
                        .grow_to(available, available, "operation_scratch")
                        .unwrap();
                    assert_eq!(fs::read(&source).unwrap(), original);
                    drop(blocker);
                    if seed == 0 && keep_cache {
                        // Reclaiming cache cannot reset another operation's workspace quota.
                        let limit = context.available_workspace_bytes();
                        let mut busy = context.workspace_reservation();
                        busy.grow_to(limit, limit, "operation_scratch").unwrap();
                        let expected_error = read().unwrap_err();
                        assert!(crate::json::is_memory_budget_error(&expected_error));
                        busy.retain(0);
                        {
                            let mut inner = db.lock().unwrap();
                            inner.persistent_vector_segments =
                                Some(inner.prefetch_unchanged_vector_segments(&BTreeSet::new()));
                            assert!(
                                inner
                                    .persistent_vector_segments
                                    .as_ref()
                                    .unwrap()
                                    .reserved_bytes()
                                    > 0
                            );
                        }
                        busy.grow_to(limit, limit, "operation_scratch").unwrap();
                        let error = read().unwrap_err();
                        assert_eq!(error.code(), expected_error.code());
                        assert_eq!(error.detail(), expected_error.detail());
                        assert_eq!(error.to_string(), expected_error.to_string());
                        assert!(db.lock().unwrap().persistent_vector_segments.is_none());
                        assert_eq!(fs::read(&source).unwrap(), original);
                        busy.retain(0);
                        busy.grow_to(limit, limit, "operation_scratch").unwrap();
                    }
                    drop((collection, db));
                }
            }
        }
        fs::remove_file(source).unwrap();
    }

    #[test]
    fn memory_reclamation_preserves_error_contracts_and_lock_boundaries() {
        use crate::json::{JsonInputKind, JsonLimitReason};
        let path = cached_vector_batch_fixture("input-admission-retries");
        let before = fs::read(&path).unwrap();
        let db = Database::open(&path).unwrap();
        let error_for = |case: usize, attempt| {
            let error = match case {
                0 => Error::resource_exhausted("memory", "operation_scratch", 32, 33),
                1 => Error::resource_exhausted("catalog", "catalog_build", 32, 33),
                2 => Error::resource_exhausted("temporary_disk", "temporary_staging", 32, 33),
                3..=5 => JsonInputKind::Filter.limit_error(
                    [
                        JsonLimitReason::WorkBudget,
                        JsonLimitReason::Depth,
                        JsonLimitReason::UnitBytes,
                    ][case - 3],
                    32,
                    Some(33),
                ),
                6 => std::io::Error::from(std::io::ErrorKind::PermissionDenied).into(),
                7 => Error::new(ErrorCode::ChecksumMismatch, "sentinel checksum"),
                _ => Error::new(ErrorCode::FileFormatError, "sentinel format"),
            };
            error.with_detail("attempt", attempt)
        };
        for (case, read) in (0..9).flat_map(|case| [false, true].map(|read| (case, read))) {
            for keep_cache in [false, true] {
                let cached_bytes = {
                    let mut inner = db.lock().unwrap();
                    inner.text_index.clear_cache();
                    inner.lazy_vector_segment = None;
                    inner.persistent_vector_segments = None;
                    let cached = inner.prefetch_unchanged_vector_segments(&BTreeSet::new());
                    let bytes = cached.reserved_bytes();
                    assert!(bytes > 0);
                    if keep_cache {
                        inner.persistent_vector_segments = Some(cached);
                    }
                    bytes
                };
                let mut attempts = 0;
                let result: Result<()> = if read {
                    db.lock().unwrap().read_reclaiming(|_| {
                        assert!(matches!(
                            db.inner.try_lock(),
                            Err(std::sync::TryLockError::WouldBlock)
                        ));
                        attempts += 1;
                        Err(error_for(case, attempts))
                    })
                } else {
                    db.admit_input_reclaiming(|| {
                        drop(db.inner.try_lock().expect("admission must not hold Inner"));
                        attempts += 1;
                        Err(error_for(case, attempts))
                    })
                };
                let reclaimed = keep_cache && matches!(case, 0 | 3);
                assert_eq!(attempts, if reclaimed { 2 } else { 1 });
                let error = result.unwrap_err();
                let expected = error_for(case, attempts);
                assert_eq!(error.code(), expected.code());
                assert_eq!(error.detail(), expected.detail());
                assert_eq!(error.to_string(), expected.to_string());
                let retained = db
                    .lock()
                    .unwrap()
                    .persistent_vector_segments
                    .as_ref()
                    .map_or(0, VectorCache::reserved_bytes);
                assert_eq!(
                    retained,
                    if keep_cache && !reclaimed {
                        cached_bytes
                    } else {
                        0
                    }
                );
            }
        }
        for read in [false, true] {
            let mut attempts = 0;
            let mut succeed = || {
                attempts += 1;
                Ok(17)
            };
            let result = if read {
                db.lock().unwrap().read_reclaiming(|_| succeed())
            } else {
                db.admit_input_reclaiming(succeed)
            };
            assert_eq!(result.unwrap(), 17);
            assert_eq!(attempts, 1);
            assert!(db.lock().unwrap().persistent_vector_segments.is_some());
        }
        assert_eq!(fs::read(&path).unwrap(), before);
        drop(db);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn native_json_allocation_failure_does_not_reclaim_caches_or_repeat_admission() {
        let path = cached_vector_batch_fixture("native-json-no-reclaim");
        let before = fs::read(&path).unwrap();
        let db = Database::open(&path).unwrap();
        let context = db.json_context();
        let cached_bytes = {
            let mut inner = db.lock().unwrap();
            let cached = inner.prefetch_unchanged_vector_segments(&BTreeSet::new());
            let bytes = cached.reserved_bytes();
            assert!(bytes > 0);
            inner.persistent_vector_segments = Some(cached);
            bytes
        };
        let baseline = context.available_workspace_bytes();
        for read in [false, true] {
            let mut attempts = 0;
            let mut allocate = || {
                attempts += 1;
                let mut value = String::new();
                let mut work = context.reservation();
                let result =
                    work.reserve_string(&mut value, 8193, crate::json::JsonInputKind::Filter);
                drop(value);
                drop(work);
                result
            };
            let error = crate::test_alloc::fail_byte_allocation_once(8193, || {
                if read {
                    db.lock().unwrap().read_reclaiming(|_| allocate())
                } else {
                    db.admit_input_reclaiming(allocate)
                }
            })
            .unwrap_err();
            assert_eq!(error.code(), ErrorCode::IoError);
            assert!(error.detail().is_empty());
            assert_eq!(attempts, 1);
            assert_eq!(context.available_workspace_bytes(), baseline);
            assert_eq!(
                db.lock()
                    .unwrap()
                    .persistent_vector_segments
                    .as_ref()
                    .unwrap()
                    .reserved_bytes(),
                cached_bytes
            );
            assert_eq!(fs::read(&path).unwrap(), before);
        }
        drop(db);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn public_read_damage_and_missing_objects_do_not_evict_optional_cache() {
        let source = cached_vector_batch_fixture("public-read-damage-source");
        let original = fs::read(&source).unwrap();
        for seed in 0..8usize {
            for truncated in [false, true] {
                let path = temp_path("public-read-damage");
                fs::copy(&source, &path).unwrap();
                let db = Database::open(&path).unwrap();
                let collection = db.collection("first").unwrap();
                let chunk = collection.get_chunk_by_external_id("chunk-1").unwrap();
                let context = db.json_context();
                let (_, _, cached_bytes, text_bytes, offset, len) = {
                    let mut inner = db.lock().unwrap();
                    let cached = inner.prefetch_unchanged_vector_segments(&BTreeSet::new());
                    let bytes = cached.reserved_bytes();
                    assert!(bytes > 0);
                    inner.persistent_vector_segments = Some(cached);
                    let record = inner.catalog.record_ref_for_chunk(&chunk.chunk_id).unwrap();
                    (
                        inner.catalog.reserved_bytes(),
                        inner.lazy_vector_source.as_ref().unwrap().reserved_bytes(),
                        bytes,
                        inner.text_index.reserved_bytes(),
                        record.payload.offset,
                        record.payload.len,
                    )
                };
                let mut file = OpenOptions::new().write(true).open(&path).unwrap();
                if truncated {
                    file.set_len(offset + len - 1).unwrap();
                } else {
                    file.seek(SeekFrom::Start(offset)).unwrap();
                    file.write_all(&[original[offset as usize] ^ (1 << seed)])
                        .unwrap();
                }
                drop(file);
                let damaged = fs::read(&path).unwrap();
                let baseline = collection.get_chunk(&chunk.chunk_id).unwrap_err();
                assert!(!crate::json::is_memory_budget_error(&baseline));
                if !truncated {
                    assert_eq!(baseline.code(), ErrorCode::ChecksumMismatch);
                }
                let blocker = context
                    .cache_reservation(context.available_total_bytes() as u128, usize::MAX)
                    .unwrap();
                for error in [
                    collection.get_chunk(&chunk.chunk_id).unwrap_err(),
                    collection.get_chunk_by_external_id("chunk-1").unwrap_err(),
                ] {
                    assert_eq!(error.code(), baseline.code());
                    assert_eq!(error.detail(), baseline.detail());
                    assert_eq!(error.to_string(), baseline.to_string());
                }
                for (error, code) in [
                    (
                        db.collection("missing").err().unwrap(),
                        ErrorCode::CollectionNotFound,
                    ),
                    (
                        collection
                            .get_document_by_external_id("missing")
                            .unwrap_err(),
                        ErrorCode::DocumentNotFound,
                    ),
                    (
                        collection.get_chunk_by_external_id("missing").unwrap_err(),
                        ErrorCode::ChunkNotFound,
                    ),
                ] {
                    assert_eq!(error.code(), code);
                }
                assert_eq!(
                    db.lock()
                        .unwrap()
                        .persistent_vector_segments
                        .as_ref()
                        .unwrap()
                        .reserved_bytes(),
                    cached_bytes
                );
                assert_eq!(db.lock().unwrap().text_index.reserved_bytes(), text_bytes);
                assert_eq!(fs::read(&path).unwrap(), damaged);
                drop((blocker, collection, db));
                fs::remove_file(path).unwrap();
            }
        }
        assert_eq!(fs::read(&source).unwrap(), original);
        fs::remove_file(source).unwrap();
    }

    #[test]
    fn public_input_admission_preserves_semantic_errors_under_cache_pressure() {
        let source = cached_vector_batch_fixture("public-input-source");
        let original = fs::read(&source).unwrap();
        let cases = [
            ErrorCode::CollectionAlreadyExists,
            ErrorCode::DuplicateExternalId,
            ErrorCode::EmbeddingModelMismatch,
            ErrorCode::EmbeddingModelMismatch,
            ErrorCode::EmbeddingModelMismatch,
        ];
        for seed in 0..16usize {
            for (case, expected) in cases.iter().enumerate() {
                for keep_cache in [false, true] {
                    let path = temp_path("public-input-budget");
                    fs::copy(&source, &path).unwrap();
                    let db = Database::open_with_options(
                        &path,
                        DatabaseOpenOptions {
                            cache_budget_mb: Some(4096),
                            ..Default::default()
                        },
                    )
                    .unwrap();
                    let collection = db.collection("first").unwrap();
                    let context = db.json_context();
                    let optional = {
                        let mut inner = db.lock().unwrap();
                        let cached = inner.prefetch_unchanged_vector_segments(&BTreeSet::new());
                        assert!(cached.reserved_bytes() > 32 * 1024);
                        if keep_cache {
                            inner.persistent_vector_segments = Some(cached);
                        } else {
                            inner.text_index.clear_cache();
                        }
                        inner.optional_cache_bytes()
                    };
                    let available = 110 * 1024 + seed * 64;
                    let blocker = context
                        .cache_reservation(
                            (context.available_total_bytes() + optional - available) as u128,
                            usize::MAX,
                        )
                        .unwrap();
                    let value = "m".repeat(96 * 1024 + seed);
                    let result = match case {
                        0 => db
                            .create_collection(CollectionConfig {
                                name: "first".into(),
                                embedding_dim: 2,
                                tokenizer: Default::default(),
                                distance_metric: DistanceMetric::Cosine,
                                embedding_model: Some(value),
                                vectors_normalized: false,
                                vector_index: VectorIndexConfig::Exact,
                            })
                            .map(|_| ()),
                        1 => {
                            let (metadata, chunks) = if seed % 2 == 0 {
                                (
                                    object_from_pairs([("payload", JsonValue::String(value))]),
                                    Vec::new(),
                                )
                            } else {
                                (
                                    object_from_pairs([(
                                        "small",
                                        JsonValue::String("first".into()),
                                    )]),
                                    vec![ChunkInput {
                                        chunk_external_id: "new-chunk".into(),
                                        text: "input ownership stays live across admission retry"
                                            .into(),
                                        ordinal: 0,
                                        metadata: object_from_pairs([(
                                            "payload",
                                            JsonValue::String(value),
                                        )]),
                                        source_span: None,
                                    }],
                                )
                            };
                            collection
                                .insert_document(DocumentInput {
                                    external_id: "doc".into(),
                                    title: None,
                                    source_uri: None,
                                    metadata,
                                    chunks,
                                })
                                .map(|_| ())
                        }
                        2 => db
                            .put_embeddings_in_collection(
                                collection.collection_id(),
                                vec![EmbeddingInput {
                                    chunk_external_id: "chunk-1".into(),
                                    model_name: value,
                                    dim: 2,
                                    vector: vec![0.5, 0.5],
                                }],
                                OnConflict::Replace,
                            )
                            .map(|_| ()),
                        3 => collection
                            .search()
                            .query_embedding(QueryEmbeddingInput {
                                model_name: Some(value),
                                dim: 2,
                                vector: vec![0.5, 0.5],
                            })
                            .execute()
                            .map(|_| ()),
                        _ => db
                            .why_not_debug(
                                "first",
                                "chunk-1",
                                None,
                                Some(QueryEmbeddingInput {
                                    model_name: Some(value),
                                    dim: 2,
                                    vector: vec![0.5, 0.5],
                                }),
                                JsonValue::object(),
                                None,
                                false,
                                1,
                                None,
                            )
                            .map(|_| ()),
                    };
                    let error = result.unwrap_err();
                    assert_eq!(
                        error.code(),
                        *expected,
                        "seed={seed}, case={case}, cache={keep_cache}: {error:?}"
                    );
                    assert!(db.lock().unwrap().persistent_vector_segments.is_none());
                    assert!(!db.lock().unwrap().text_index.has_resident_cache());
                    assert_eq!(fs::read(&path).unwrap(), original);
                    context
                        .workspace_reservation()
                        .grow_to(available, available, "operation_scratch")
                        .unwrap();
                    drop((blocker, collection, db));
                    fs::remove_file(path).unwrap();
                }
            }
        }
        assert_eq!(fs::read(&source).unwrap(), original);
        fs::remove_file(source).unwrap();
    }

    #[test]
    fn external_input_admission_reclaims_cache_and_preserves_transfer_ownership() {
        use crate::json::{JsonInputKind, JsonOperationContext, JsonProcessingPolicyV1};
        let source = cached_vector_batch_fixture("input-transfer-source");
        let original = fs::read(&source).unwrap();
        for seed in 1..=32usize {
            for keep_cache in [false, true] {
                let path = temp_path("input-transfer-budget");
                fs::copy(&source, &path).unwrap();
                let db = Database::open_with_options(
                    &path,
                    DatabaseOpenOptions {
                        cache_budget_mb: Some(4096),
                        ..Default::default()
                    },
                )
                .unwrap();
                let runtime = db.json_context();
                let optional = {
                    let mut inner = db.lock().unwrap();
                    let cached = inner.prefetch_unchanged_vector_segments(&BTreeSet::new());
                    assert!(cached.reserved_bytes() > 32 * 1024);
                    if keep_cache {
                        inner.persistent_vector_segments = Some(cached);
                    }
                    inner.optional_cache_bytes()
                };
                let required = 64 * 1024 + seed;
                let enough = seed % 4 != 0;
                let available = if enough {
                    required + seed
                } else {
                    required - 1
                };
                let blocker = runtime
                    .cache_reservation(
                        (runtime.available_total_bytes() + optional - available) as u128,
                        usize::MAX,
                    )
                    .unwrap();
                let input = JsonOperationContext::standalone();
                let json_bytes = match seed % 3 {
                    0 => required,
                    1 => 0,
                    _ => required / 2,
                };
                let mut json = input.reservation();
                json.grow(json_bytes, JsonInputKind::Filter).unwrap();
                let mut workspace = input.workspace_reservation();
                workspace
                    .grow_to(
                        required - json_bytes,
                        required - json_bytes,
                        "operation_scratch",
                    )
                    .unwrap();
                let result = db.attach_json_context(&input, JsonInputKind::Filter);
                assert_eq!(
                    result.is_ok(),
                    enough,
                    "seed={seed}, cache={keep_cache}: {result:?}"
                );
                assert!(db.lock().unwrap().persistent_vector_segments.is_none());
                let mut probe = input.workspace_reservation();
                if enough {
                    // A transferred child must spend the target's remaining budget.
                    probe
                        .grow_to(
                            available - required,
                            available - required,
                            "operation_scratch",
                        )
                        .unwrap();
                    assert!(crate::json::is_memory_budget_error(
                        &probe
                            .grow_to(
                                available - required + 1,
                                available - required + 1,
                                "operation_scratch",
                            )
                            .unwrap_err()
                    ));
                } else {
                    assert!(crate::json::is_memory_budget_error(&result.unwrap_err()));
                    // A rejected transfer keeps every live lease on its original ledger.
                    let free = JsonProcessingPolicyV1::WORK_BUDGET_BYTES - required;
                    probe.grow_to(free, free, "operation_scratch").unwrap();
                }
                drop((probe, json, workspace, input));
                runtime
                    .workspace_reservation()
                    .grow_to(available, available, "operation_scratch")
                    .unwrap();
                assert_eq!(fs::read(&path).unwrap(), original);
                drop((blocker, db));
                fs::remove_file(path).unwrap();
            }
        }
        assert_eq!(fs::read(&source).unwrap(), original);
        fs::remove_file(source).unwrap();
    }

    #[test]
    fn import_batch_read_reclaims_optional_vectors_before_rejecting() {
        let source = cached_vector_batch_fixture("import-read-source");
        let original = fs::read(&source).unwrap();
        let mut successes = 0;
        let mut rejections = 0;
        for mode in [
            ImportIndexBuildMode::InlineFullCheckpoint,
            ImportIndexBuildMode::DeferredFullCheckpoint,
        ] {
            for seed in 0..100usize {
                let available = 65536 + seed * 128;
                let mut outcomes = Vec::new();
                for keep_cache in [false, true] {
                    let path = temp_path("import-read-budget");
                    fs::copy(&source, &path).unwrap();
                    let db = Database::open_with_options(
                        &path,
                        DatabaseOpenOptions {
                            cache_budget_mb: Some(4096),
                            ..Default::default()
                        },
                    )
                    .unwrap();
                    let context = db.json_context();
                    let mut inner = db.lock().unwrap();
                    let cached = inner.prefetch_unchanged_vector_segments(&BTreeSet::new());
                    assert!(!cached.is_empty());
                    if keep_cache {
                        inner.persistent_vector_segments = Some(cached);
                    } else {
                        drop(cached);
                    }
                    let record = RecordPayload::DocumentPut(Document {
                        document_id: inner.catalog.next_id("doc_", 0).unwrap(),
                        collection_id: inner
                            .catalog
                            .collection_by_name("first")
                            .unwrap()
                            .collection_id
                            .clone(),
                        external_id: "imported".into(),
                        title: None,
                        source_uri: None,
                        metadata: object_from_pairs([(
                            "payload",
                            JsonValue::String("x".repeat(100 + seed)),
                        )]),
                        created_at: "1970-01-01T00:00:00Z".into(),
                        updated_at: "1970-01-01T00:00:00Z".into(),
                    });
                    let mut builder = crate::storage::PreparedRecordsBuilder::new(
                        path.parent().unwrap(),
                        1_048_576,
                        context.clone(),
                    )
                    .unwrap();
                    builder.push(record).unwrap();
                    let prepared = builder.finish().unwrap();
                    let catalog_bytes = inner.catalog.reserved_bytes();
                    let signature = inner.catalog.state_signature();
                    let commit = inner.catalog.last_commit_id;
                    let blocker = context
                        .cache_reservation(
                            (3 * 1024 * 1024 * 1024usize - catalog_bytes - available) as u128,
                            usize::MAX,
                        )
                        .unwrap();
                    let result = inner.append_import_batches(prepared, mode, &context);
                    let success = result.is_ok();
                    let error = result.as_ref().err().map(|error| format!("{error:?}"));
                    if success {
                        successes += 1;
                        assert_eq!(inner.catalog.last_commit_id, commit + 1);
                    } else {
                        rejections += 1;
                        assert_eq!(fs::read(&path).unwrap(), original);
                        assert_eq!(inner.catalog.state_signature(), signature);
                        assert_eq!(inner.catalog.reserved_bytes(), catalog_bytes);
                    }
                    drop(blocker);
                    let expected = inner.catalog.state_signature();
                    drop(inner);
                    drop(db);
                    let reopened = Database::open_read_only(&path).unwrap();
                    assert_eq!(reopened.lock().unwrap().catalog.state_signature(), expected);
                    assert_eq!(
                        reopened
                            .collection("first")
                            .unwrap()
                            .get_document_by_external_id("imported")
                            .is_ok(),
                        success
                    );
                    drop(reopened);
                    fs::remove_file(path).unwrap();
                    outcomes.push((success, error));
                }
                assert_eq!(
                    outcomes[0].0, outcomes[1].0,
                    "mode={mode:?}, available={available}: {outcomes:?}"
                );
            }
        }
        assert!(successes > 0 && rejections > 0);
        assert_eq!(fs::read(&source).unwrap(), original);
        fs::remove_file(source).unwrap();
    }

    #[test]
    fn deferred_import_preserves_all_records_under_cache_pressure() {
        let source = cached_vector_batch_fixture("deferred-read-source");
        let original = fs::read(&source).unwrap();
        for seed in 0..32usize {
            for keep_cache in [false, true] {
                let path = temp_path("deferred-read-budget");
                fs::copy(&source, &path).unwrap();
                let db = Database::open_with_options(
                    &path,
                    DatabaseOpenOptions {
                        cache_budget_mb: Some(4096),
                        ..Default::default()
                    },
                )
                .unwrap();
                let context = db.json_context();
                let mut inner = db.lock().unwrap();
                let cache = inner.prefetch_unchanged_vector_segments(&BTreeSet::new());
                assert!(cache.reserved_bytes() > 32 * 1024);
                if keep_cache {
                    inner.persistent_vector_segments = Some(cache);
                } else {
                    drop(cache);
                }
                let mut builder = crate::storage::PreparedRecordsBuilder::new(
                    path.parent().unwrap(),
                    1_048_576,
                    context.clone(),
                )
                .unwrap();
                for (index, bytes) in [1024 + seed, 128 * 1024 + seed].into_iter().enumerate() {
                    builder
                        .push(RecordPayload::DocumentPut(Document {
                            document_id: inner.catalog.next_id("doc_", index as u64).unwrap(),
                            collection_id: inner
                                .catalog
                                .collection_by_name("first")
                                .unwrap()
                                .collection_id
                                .clone(),
                            external_id: format!("imported-{index}"),
                            title: None,
                            source_uri: None,
                            metadata: object_from_pairs([(
                                "payload",
                                JsonValue::String("x".repeat(bytes)),
                            )]),
                            created_at: "1970-01-01T00:00:00Z".into(),
                            updated_at: "1970-01-01T00:00:00Z".into(),
                        }))
                        .unwrap();
                }
                let prepared = builder.finish().unwrap();
                // Leave a bounded decoding window with and without a retained cache.
                let available = 344 * 1024 + seed * 128;
                let blocker = context
                    .cache_reservation(
                        (3 * 1024 * 1024 * 1024usize - inner.catalog.reserved_bytes() - available)
                            as u128,
                        usize::MAX,
                    )
                    .unwrap();
                let commit = inner.catalog.last_commit_id;
                inner
                    .append_import_batches(
                        prepared,
                        ImportIndexBuildMode::DeferredFullCheckpoint,
                        &context,
                    )
                    .unwrap_or_else(|error| panic!("seed={seed}, cache={keep_cache}: {error:?}"));
                let batches = inner.catalog.last_commit_id - commit;
                assert!((1..=2).contains(&batches));
                drop(blocker);
                let expected = inner.catalog.state_signature();
                drop(inner);
                drop(db);
                let reopened = Database::open_read_only(&path).unwrap();
                assert_eq!(reopened.lock().unwrap().catalog.state_signature(), expected);
                let collection = reopened.collection("first").unwrap();
                for (index, bytes) in [1024 + seed, 128 * 1024 + seed].into_iter().enumerate() {
                    let document = collection
                        .get_document_by_external_id(&format!("imported-{index}"))
                        .unwrap();
                    assert_eq!(
                        document
                            .metadata
                            .get("payload")
                            .unwrap()
                            .as_str()
                            .unwrap()
                            .len(),
                        bytes
                    );
                }
                drop((collection, reopened));
                fs::remove_file(path).unwrap();
            }
        }
        assert_eq!(fs::read(&source).unwrap(), original);
        fs::remove_file(source).unwrap();
    }

    #[test]
    fn catalog_batch_keeps_only_the_final_metadata_owner_and_reference() {
        for seed in 0..100 {
            let path = temp_path("catalog-final-batch-owner");
            let db = Database::create(&path, DatabaseConfig::default()).unwrap();
            let collection = db
                .create_collection(CollectionConfig {
                    name: "notes".into(),
                    embedding_dim: 2,
                    distance_metric: DistanceMetric::Cosine,
                    embedding_model: None,
                    vectors_normalized: false,
                    vector_index: VectorIndexConfig::Exact,
                    tokenizer: Default::default(),
                })
                .unwrap();
            let id = format!("doc_{:016}", seed + 1);
            let document = |text: &str| Document {
                document_id: id.clone(),
                collection_id: collection.collection_id().into(),
                external_id: "doc".into(),
                title: None,
                source_uri: None,
                metadata: object_from_pairs([("value", JsonValue::String(text.repeat(seed + 1)))]),
                created_at: "1970-01-01T00:00:00Z".into(),
                updated_at: "1970-01-01T00:00:00Z".into(),
            };
            let newest = document("newest");
            let pointer = newest
                .metadata
                .get("value")
                .unwrap()
                .as_str()
                .unwrap()
                .as_ptr();
            let mut records = vec![
                RecordPayload::DocumentPut(document("old")),
                RecordPayload::DocumentPut(document("middle")),
                RecordPayload::Tombstone(Tombstone {
                    target_type: "document".into(),
                    target_id: id.clone(),
                    deleted_at: "1970-01-01T00:00:00Z".into(),
                    reason: None,
                }),
            ];
            if seed % 2 == 0 {
                records.push(RecordPayload::DocumentPut(newest));
            }
            let mut inner = db.lock().unwrap();
            inner.commit_records(records).unwrap();
            if seed % 2 == 0 {
                let actual = &inner.catalog.documents[&id];
                assert_eq!(
                    actual
                        .metadata
                        .get("value")
                        .unwrap()
                        .as_str()
                        .unwrap()
                        .as_ptr(),
                    pointer
                );
                assert_eq!(
                    inner.catalog.record_refs[&id].sequence_no,
                    inner.last_commit_sequence_no - 1
                );
                assert!(!inner.catalog.deleted_documents.contains(&id));
            } else {
                assert_eq!(inner.catalog.documents[&id].metadata, JsonValue::object());
                assert!(!inner.catalog.record_refs.contains_key(&id));
            }
            let expected = inner.catalog.state_signature();
            drop(inner);
            let reopened = Database::open_read_only(&path).unwrap();
            assert_eq!(reopened.lock().unwrap().catalog.state_signature(), expected);
            drop(reopened);
            drop(collection);
            drop(db);
            fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn loaded_vector_segments_retain_their_owned_allocations() {
        for seed in 0..100 {
            let mut vector = Vec::with_capacity(128 + seed);
            vector.extend([1.0, 0.5, 0.25]);
            let pointer = vector.as_ptr();
            let vectors = vec![VectorIndexEntry {
                chunk_id: "chk_1".into(),
                embedding_id: "emb_1".into(),
                vector,
            }];
            let entries_pointer = vectors.as_ptr();
            let segments = vector_segments_by_collection(vec![VectorIndexSegmentPayload {
                collection_id: "col_1".into(),
                covered_commit_sequence_no: seed as u64,
                dim: 3,
                distance_metric: DistanceMetric::Cosine,
                embedding_model: None,
                vectors_normalized: false,
                vectors,
            }]);
            let segment = &segments["col_1"];
            assert_eq!(segment.vectors.as_ptr(), entries_pointer);
            assert_eq!(segment.vectors[0].vector.as_ptr(), pointer);
            assert_eq!(segment.covered_commit_sequence_no, seed as u64);
            assert_eq!(segment.vectors[0].vector, [1.0, 0.5, 0.25]);
        }
    }

    #[test]
    fn prepared_search_adoption_uses_core_classification_and_current_headroom() {
        let path = temp_path("prepared-query-json-headroom");
        let db = Database::create(&path, DatabaseConfig::default()).unwrap();
        let collection = db
            .create_collection(CollectionConfig {
                tokenizer: Default::default(),
                name: "notes".into(),
                embedding_dim: 2,
                distance_metric: crate::DistanceMetric::Cosine,
                embedding_model: None,
                vectors_normalized: false,
                vector_index: crate::VectorIndexConfig::Exact,
            })
            .unwrap();
        let mut busy = db.json_context().reservation();
        busy.grow(
            db.json_context().available_workspace_bytes(),
            crate::json::JsonInputKind::Filter,
        )
        .unwrap();
        for kind in [
            crate::json::JsonInputKind::Filter,
            crate::json::JsonInputKind::QueryEmbedding,
            crate::json::JsonInputKind::MetadataBoost,
        ] {
            let context = crate::json::JsonOperationContext::standalone();
            let mut input = context.reservation();
            input.grow(4096, kind).unwrap();
            let mut query = collection.search().text("text");
            match kind {
                crate::json::JsonInputKind::Filter => {
                    query = query.filter(crate::json::parse(r#"{"chunk.x":1}"#).unwrap());
                }
                crate::json::JsonInputKind::QueryEmbedding => {
                    query = query.vector(vec![1.0, 0.0]);
                }
                _ => {
                    query = query.metadata_boost(MetadataBoostConfig::parse(crate::json::parse(r#"{"schema_version":"0.1","combine":"sum","max_boost":1,"rules":[{"when":{"chunk.x":1},"boost":0.1}]}"#).unwrap()).unwrap());
                }
            }
            let error = query.execute_with_json_context(Some(context)).unwrap_err();
            assert_eq!(error.code(), kind.error_code());
            assert!(error
                .detail()
                .iter()
                .any(|(key, value)| key == "json_input_kind" && value == kind.as_str()));
            assert_eq!(
                error
                    .to_json()
                    .get("detail")
                    .unwrap()
                    .get("limit")
                    .unwrap()
                    .as_f64(),
                Some(0.0)
            );
        }
        drop(busy);
        drop(collection);
        drop(db);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn why_not_guards_inputs_and_reuses_released_result_json_capacity() {
        let path = temp_path("why-not-json-work");
        let db = Database::create(&path, DatabaseConfig::default()).unwrap();
        let col = db
            .create_collection(CollectionConfig {
                name: "notes".into(),
                embedding_dim: 2,
                distance_metric: DistanceMetric::Cosine,
                embedding_model: None,
                vectors_normalized: false,
                vector_index: VectorIndexConfig::Exact,
                tokenizer: Default::default(),
            })
            .unwrap();
        col.insert_document(DocumentInput {
            external_id: "doc".into(),
            title: None,
            source_uri: None,
            metadata: JsonValue::object(),
            chunks: (0..3)
                .map(|ordinal| ChunkInput {
                    chunk_external_id: format!("doc#{ordinal}"),
                    text: "alpha".into(),
                    ordinal,
                    metadata: object_from_pairs([(
                        "x",
                        JsonValue::Array(vec![JsonValue::Null; 200_000]),
                    )]),
                    source_span: None,
                })
                .collect(),
        })
        .unwrap();
        let before = std::fs::read(&path).unwrap();
        let capacity = db.json_context().available_workspace_bytes();
        let diagnosis = db
            .why_not_debug(
                "notes",
                "doc#0",
                Some("alpha".into()),
                None,
                JsonValue::object(),
                None,
                false,
                3,
                None,
            )
            .unwrap();
        assert!(diagnosis
            .result
            .get("returned_rank")
            .unwrap()
            .as_f64()
            .is_some());
        assert_eq!(
            diagnosis
                .result
                .get("filter_result")
                .unwrap()
                .get("matched"),
            Some(&JsonValue::Bool(true))
        );
        let mut deep = JsonValue::Null;
        for _ in 0..30_000 {
            deep = JsonValue::Array(vec![deep]);
        }
        let error = db
            .why_not_debug(
                "notes",
                "doc#0",
                Some("alpha".into()),
                None,
                deep,
                None,
                false,
                3,
                None,
            )
            .unwrap_err();
        assert_eq!(error.code(), ErrorCode::InvalidMetadataFilter);
        assert!(crate::json::is_depth_error(&error));
        let mut vector =
            Vec::with_capacity(crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES / 4 + 1);
        vector.extend([1.0, 0.0]);
        let error = db
            .why_not_debug(
                "notes",
                "doc#0",
                None,
                Some(QueryEmbeddingInput::from_vector(vector)),
                JsonValue::object(),
                None,
                false,
                3,
                None,
            )
            .unwrap_err();
        assert!(error
            .detail()
            .iter()
            .any(|(key, value)| key == "json_input_kind" && value == "query_embedding"));
        db.json_context()
            .reservation()
            .grow(capacity, crate::json::JsonInputKind::StoredJson)
            .unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), before);
        drop(col);
        drop(db);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn search_result_metadata_and_filter_copies_share_one_operation_budget() {
        for (field, accepted, rejected) in [("payload", 5, 6), ("path", 2, 3)] {
            let path = temp_path(&format!("result-json-budget-{field}"));
            let db = Database::create(&path, DatabaseConfig::default()).unwrap();
            let col = db
                .create_collection(CollectionConfig {
                    name: "notes".into(),
                    embedding_dim: 2,
                    distance_metric: DistanceMetric::Cosine,
                    embedding_model: None,
                    vectors_normalized: false,
                    vector_index: VectorIndexConfig::Exact,
                    tokenizer: Default::default(),
                })
                .unwrap();
            col.insert_document(DocumentInput {
                external_id: "doc".into(),
                title: None,
                source_uri: Some("docs/source.md".into()),
                metadata: object_from_pairs([(
                    field,
                    JsonValue::String("x".repeat(6 * 1024 * 1024)),
                )]),
                chunks: (0..6)
                    .map(|ordinal| ChunkInput {
                        chunk_external_id: format!("doc#{ordinal}"),
                        text: "alpha beta".into(),
                        ordinal,
                        metadata: object_from_pairs([("x", JsonValue::Null)]),
                        source_span: None,
                    })
                    .collect(),
            })
            .unwrap();
            let before = std::fs::read(&path).unwrap();
            let mut results = col
                .search()
                .text("alpha")
                .top_k(accepted)
                .execute()
                .unwrap();
            assert_eq!(results.results.len(), accepted);
            results.results[0]
                .document_metadata
                .as_object_mut()
                .unwrap()
                .insert(field.into(), JsonValue::Null);
            assert_eq!(
                results.results[1]
                    .document_metadata
                    .get(field)
                    .unwrap()
                    .as_str()
                    .unwrap()
                    .len(),
                6 * 1024 * 1024
            );
            let document_path = results.results[0]
                .explain
                .get("source")
                .unwrap()
                .get("document_path")
                .unwrap()
                .as_str()
                .unwrap();
            if field == "payload" {
                assert_eq!(document_path, "docs/source.md");
            } else {
                assert_eq!(document_path.len(), 6 * 1024 * 1024);
            }
            drop(results);
            let error = col
                .search()
                .text("alpha")
                .top_k(rejected)
                .execute()
                .unwrap_err();
            assert_eq!(error.code(), ErrorCode::RecoveryError);
            assert!(error
                .detail()
                .iter()
                .any(|(key, value)| key == "json_input_kind" && value == "stored_json"));
            let filter = object_from_pairs([(
                "chunk.x",
                object_from_pairs([("$in", JsonValue::Array(vec![JsonValue::Null; 350_000]))]),
            )]);
            let error = col
                .search()
                .text("alpha")
                .filter(filter)
                .top_k(1)
                .execute()
                .unwrap_err();
            assert_eq!(error.code(), ErrorCode::InvalidMetadataFilter);
            assert!(error
                .detail()
                .iter()
                .any(|(key, value)| key == "json_limit_reason" && value == "work_budget_exceeded"));
            let results = col.search().text("alpha").top_k(1).execute().unwrap();
            assert_eq!(
                results.results[0]
                    .document_metadata
                    .get(field)
                    .unwrap()
                    .as_str()
                    .unwrap()
                    .len(),
                6 * 1024 * 1024
            );
            assert_eq!(std::fs::read(&path).unwrap(), before);
            drop(col);
            drop(db);
            std::fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn direct_query_inputs_share_retained_capacity_before_core_execution() {
        let path = temp_path("typed-query-json-budget");
        let db = Database::create(&path, DatabaseConfig::default()).unwrap();
        let collection = db
            .create_collection(CollectionConfig {
                tokenizer: Default::default(),
                name: "notes".into(),
                embedding_dim: 2,
                distance_metric: crate::DistanceMetric::Cosine,
                embedding_model: None,
                vectors_normalized: false,
                vector_index: crate::VectorIndexConfig::Exact,
            })
            .unwrap();
        let before = std::fs::read(&path).unwrap();
        let mut vector = Vec::with_capacity(
            crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES / std::mem::size_of::<f32>() + 1,
        );
        vector.extend([1.0, 0.0]);
        let error = collection.search().vector(vector).execute().unwrap_err();
        assert!(error
            .detail()
            .iter()
            .any(|(key, value)| key == "json_input_kind" && value == "query_embedding"));
        let filter = |count| {
            crate::json::object_from_pairs([(
                "chunk.x",
                crate::json::object_from_pairs([(
                    "$in",
                    JsonValue::Array(vec![JsonValue::Null; count]),
                )]),
            )])
        };
        let boost = MetadataBoostConfig::parse(crate::json::object_from_pairs([
            ("schema_version", JsonValue::String("0.1".into())),
            ("combine", JsonValue::String("sum".into())),
            ("max_boost", JsonValue::Number(1.0)),
            (
                "rules",
                JsonValue::Array(vec![crate::json::object_from_pairs([
                    ("when", filter(300_000)),
                    ("boost", JsonValue::Number(0.1)),
                ])]),
            ),
        ]))
        .unwrap();
        collection
            .search()
            .text("q")
            .filter(filter(400_000))
            .execute()
            .unwrap();
        collection
            .search()
            .text("q")
            .metadata_boost(boost.clone())
            .execute()
            .unwrap();
        let error = collection
            .search()
            .text("q")
            .metadata_boost(boost)
            .filter(filter(400_000))
            .execute()
            .unwrap_err();
        assert_eq!(error.code(), ErrorCode::InvalidMetadataFilter);
        assert!(error
            .detail()
            .iter()
            .any(|(key, value)| key == "json_limit_reason" && value == "work_budget_exceeded"));
        assert_eq!(std::fs::read(&path).unwrap(), before);
        drop(collection);
        drop(db);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn document_and_collection_copies_reserve_all_owned_fields_and_release_on_failure() {
        use crate::json::{JsonOperationContext, JsonProcessingPolicyV1};

        let limit = JsonProcessingPolicyV1::WORK_BUDGET_BYTES;
        for seed in 0..100usize {
            let context = JsonOperationContext::standalone();
            let mut source = "title \u{4e2d} ".repeat(8192 + seed);
            source.reserve(4096);
            let document = Document {
                document_id: format!("doc_{:016}", seed + 1),
                collection_id: "col_0000000000000001".into(),
                external_id: format!("external-{seed}"),
                title: Some(source.clone()),
                source_uri: (seed % 2 == 0).then(|| "file:///source".into()),
                metadata: parse(r#"{"source":"fixture","nested":[1,true]}"#).unwrap(),
                created_at: "2026-10-08T00:00:00Z".into(),
                updated_at: "2026-10-08T00:00:01Z".into(),
            };
            let collection = CollectionInfo {
                tokenizer: Default::default(),
                collection_id: document.collection_id.clone(),
                name: "notes".into(),
                embedding_dim: 2,
                distance_metric: DistanceMetric::Cosine,
                embedding_model: Some(source),
                vectors_normalized: false,
                vector_index: if seed % 2 == 0 {
                    VectorIndexConfig::Exact
                } else {
                    VectorIndexConfig::Ann {
                        provider: "fixture-provider".into(),
                        params: parse(r#"{"nested":{"value":10}}"#).unwrap(),
                    }
                },
                created_at: document.created_at.clone(),
            };
            for is_document in [false, true] {
                let copy = || {
                    if is_document {
                        copy_document_in(&document, &context)
                            .map(|(value, work)| (RecordPayload::DocumentPut(value), work))
                    } else {
                        copy_collection_in(&collection, &context)
                            .map(|(value, work)| (RecordPayload::CollectionCreate(value), work))
                    }
                };
                let mut held = context.workspace_reservation();
                let available = 32 * 1024 + seed;
                held.grow_to(limit - available, limit - available, "operation_scratch")
                    .unwrap();
                let error = copy().err().unwrap();
                assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                assert_eq!(error_detail_value(&error, "resource"), Some("memory"));
                assert_eq!(
                    error_detail_value(&error, "operation"),
                    Some("operation_scratch")
                );
                held.grow_to(limit, limit, "operation_scratch").unwrap();
                held.retain(0);

                let (record, work) = copy().unwrap();
                let strings = match &record {
                    RecordPayload::DocumentPut(copied) => {
                        assert_eq!(copied.to_json(), document.to_json());
                        assert_eq!(
                            copied.title.as_ref().unwrap().capacity(),
                            document.title.as_ref().unwrap().len()
                        );
                        copied.document_id.capacity()
                            + copied.collection_id.capacity()
                            + copied.external_id.capacity()
                            + copied.title.as_ref().map_or(0, String::capacity)
                            + copied.source_uri.as_ref().map_or(0, String::capacity)
                            + copied.created_at.capacity()
                            + copied.updated_at.capacity()
                    }
                    RecordPayload::CollectionCreate(copied) => {
                        assert_eq!(copied.to_json(), collection.to_json());
                        assert_eq!(
                            copied.embedding_model.as_ref().unwrap().capacity(),
                            collection.embedding_model.as_ref().unwrap().len()
                        );
                        copied.collection_id.capacity()
                            + copied.name.capacity()
                            + copied.embedding_model.as_ref().map_or(0, String::capacity)
                            + copied.created_at.capacity()
                            + match &copied.vector_index {
                                VectorIndexConfig::Exact => 0,
                                VectorIndexConfig::Ann { provider, .. } => provider.capacity(),
                            }
                    }
                    _ => unreachable!(),
                };
                let bytes = strings + work.json_bytes();
                held.grow_to(limit - bytes, limit - bytes, "operation_scratch")
                    .unwrap();
                assert!(held
                    .grow_to(limit - bytes + 1, limit - bytes + 1, "operation_scratch")
                    .is_err());
                drop(record);
                drop(work);
                held.grow_to(limit, limit, "operation_scratch").unwrap();
            }
        }
    }

    #[test]
    fn public_document_and_collection_reads_keep_budget_failures_read_only() {
        let path = temp_path("owned-document-and-collection-reads");
        let db = Database::create(&path, DatabaseConfig::default()).unwrap();
        let model = "model".repeat(16 * 1024);
        let collection = db
            .create_collection(CollectionConfig {
                tokenizer: Default::default(),
                name: "notes".into(),
                embedding_dim: 2,
                distance_metric: DistanceMetric::Cosine,
                embedding_model: Some(model.clone()),
                vectors_normalized: false,
                vector_index: VectorIndexConfig::Exact,
            })
            .unwrap();
        let title = "title \u{4e2d} ".repeat(16 * 1024);
        collection
            .insert_document(DocumentInput {
                external_id: "document".into(),
                title: Some(title.clone()),
                source_uri: Some("file:///source".into()),
                metadata: JsonValue::object(),
                chunks: vec![ChunkInput {
                    chunk_external_id: "chunk".into(),
                    ordinal: 0,
                    text: "query".into(),
                    metadata: JsonValue::object(),
                    source_span: None,
                }],
            })
            .unwrap();
        let limit = db.json_context().available_workspace_bytes();
        let id = collection
            .get_document_by_external_id("document")
            .unwrap()
            .document_id;
        let before = fs::read(&path).unwrap();
        for seed in 0..100usize {
            let context = db.json_context();
            let available = 32 * 1024 + seed;
            let mut held = context.workspace_reservation();
            held.grow_to(limit - available, limit - available, "operation_scratch")
                .unwrap();
            for error in [
                collection.get_document(&id).unwrap_err(),
                collection
                    .get_document_by_external_id("document")
                    .unwrap_err(),
                db.list_collections().unwrap_err(),
            ] {
                assert_eq!(error.code(), ErrorCode::ResourceExhausted);
            }
            // Acquiring a handle only copies its identity, not its complete configuration.
            drop(db.collection("notes").unwrap());
            held.grow_to(limit, limit, "operation_scratch").unwrap();
            assert_eq!(
                db.collection("notes").err().unwrap().code(),
                ErrorCode::ResourceExhausted
            );
            assert_eq!(
                db.collection("missing").err().unwrap().code(),
                ErrorCode::CollectionNotFound
            );
            held.retain(0);
            let copied = collection.get_document(&id).unwrap();
            assert_eq!(copied.title.as_deref(), Some(title.as_str()));
            let infos = db.list_collections().unwrap();
            assert_eq!(infos.len(), 1);
            assert_eq!(infos[0].embedding_model.as_deref(), Some(model.as_str()));
            // Returned objects belong to the caller and no longer consume runtime credit.
            held.grow_to(limit, limit, "operation_scratch").unwrap();
        }
        assert_eq!(fs::read(&path).unwrap(), before);
        drop((collection, db));
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn get_and_list_share_runtime_json_capacity_and_release_it_on_return() {
        use crate::json::JsonInputKind;
        let path = temp_path("read-json-budget");
        let db = Database::create(&path, DatabaseConfig::default()).unwrap();
        let collection = db
            .create_collection(CollectionConfig {
                tokenizer: Default::default(),
                name: "notes".into(),
                embedding_dim: 2,
                distance_metric: DistanceMetric::Cosine,
                embedding_model: None,
                vectors_normalized: false,
                vector_index: crate::VectorIndexConfig::Exact,
            })
            .unwrap();
        let metadata =
            || object_from_pairs([("payload", JsonValue::String("x".repeat(1024 * 1024)))]);
        collection
            .insert_document(DocumentInput {
                external_id: "document".into(),
                title: None,
                source_uri: None,
                metadata: metadata(),
                chunks: (0..4)
                    .rev()
                    .map(|ordinal| crate::ChunkInput {
                        chunk_external_id: format!("chunk-{ordinal}"),
                        text: "payload".into(),
                        ordinal,
                        metadata: metadata(),
                        source_span: None,
                    })
                    .collect(),
            })
            .unwrap();
        let context = db.json_context();
        let capacity = context.available_workspace_bytes();
        let document = collection.get_document_by_external_id("document").unwrap();
        let chunk = collection.get_chunk_by_external_id("chunk-0").unwrap();
        let before = fs::read(&path).unwrap();
        let mut busy = context.reservation();
        busy.grow(capacity - 3 * 1024 * 1024, JsonInputKind::StoredJson)
            .unwrap();
        let other = db.clone().collection("notes").unwrap();
        assert_eq!(
            other.get_document(&document.document_id).unwrap().metadata,
            document.metadata
        );
        assert_eq!(
            other
                .get_document_by_external_id("document")
                .unwrap()
                .metadata,
            document.metadata
        );
        assert_eq!(
            other.get_chunk(&chunk.chunk_id).unwrap().metadata,
            chunk.metadata
        );
        assert_eq!(
            other.get_chunk_by_external_id("chunk-0").unwrap().metadata,
            chunk.metadata
        );
        let array_bytes =
            4 * (std::mem::size_of::<Chunk>() + std::mem::size_of::<(usize, &Chunk)>());
        for error in [
            other.list_chunks(&document.document_id).unwrap_err(),
            other
                .list_chunks_by_document_external_id("document")
                .unwrap_err(),
        ] {
            assert_eq!(error.code(), ErrorCode::RecoveryError);
            let detail = error.to_json();
            let detail = detail.get("detail").unwrap();
            assert_eq!(
                detail.get("json_input_kind").and_then(JsonValue::as_str),
                Some("stored_json")
            );
            let limit = detail.get("limit").and_then(JsonValue::as_f64).unwrap();
            let actual = detail.get("actual").and_then(JsonValue::as_f64).unwrap();
            assert!(limit > 0.0 && limit <= (3 * 1024 * 1024 - array_bytes) as f64);
            assert!(actual > limit);
        }
        busy.grow(3 * 1024 * 1024, JsonInputKind::StoredJson)
            .unwrap();
        for error in [
            other.get_document(&document.document_id).unwrap_err(),
            other.get_document_by_external_id("document").unwrap_err(),
        ] {
            assert_eq!(error.code(), ErrorCode::RecoveryError);
        }
        for error in [
            other.get_chunk(&chunk.chunk_id).unwrap_err(),
            other.get_chunk_by_external_id("chunk-0").unwrap_err(),
        ] {
            assert_stored_read_admission(&error, 0, &[20]);
        }
        drop(busy);
        let mut chunks = other.list_chunks(&document.document_id).unwrap();
        assert_eq!(
            chunks.iter().map(|chunk| chunk.ordinal).collect::<Vec<_>>(),
            vec![0, 1, 2, 3]
        );
        chunks[0].metadata.as_object_mut().unwrap().clear();
        assert_eq!(
            other.get_chunk(&chunk.chunk_id).unwrap().metadata,
            chunk.metadata
        );
        assert_eq!(
            other
                .list_chunks_by_document_external_id("document")
                .unwrap()
                .len(),
            4
        );
        assert_eq!(fs::read(&path).unwrap(), before);
        context
            .reservation()
            .grow(capacity, JsonInputKind::StoredJson)
            .unwrap();
        drop(other);
        drop(collection);
        drop(db);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn read_snapshot_blocks_mutations_on_cloned_handles() {
        let path = temp_path("read-snapshot-lock");
        let db = Database::create(&path, DatabaseConfig::default()).unwrap();
        let other = db.clone();
        db.with_read_snapshot(|catalog, _| {
            assert_eq!(catalog.live_documents().count(), 0);
            assert!(matches!(
                other.inner.try_lock(),
                Err(std::sync::TryLockError::WouldBlock)
            ));
            Ok(())
        })
        .unwrap();
        assert!(other.inner.try_lock().is_ok());
        drop(other);
        drop(db);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn snapshot_copy_failure_never_removes_preexisting_temporary_file() {
        let source = temp_path("snapshot-copy-source");
        let temporary = temp_path("snapshot-copy-existing-temp");
        fs::write(&source, b"snapshot source").unwrap();
        fs::write(&temporary, b"preexisting unrelated data").unwrap();
        assert!(copy_snapshot_file(&source, &temporary).is_err());
        assert_eq!(fs::read(&temporary).unwrap(), b"preexisting unrelated data");
        fs::remove_file(&source).unwrap();
        assert!(copy_snapshot_file(&source, &temporary).is_err());
        assert_eq!(fs::read(&temporary).unwrap(), b"preexisting unrelated data");
        fs::remove_file(temporary).unwrap();
    }

    #[test]
    fn id_exhaustion_is_rejected_before_any_write() {
        let path = temp_path("id-exhaustion");
        let db = Database::create(&path, DatabaseConfig::default()).unwrap();
        let config = CollectionConfig {
            name: "notes".into(),
            embedding_dim: 2,
            distance_metric: DistanceMetric::Cosine,
            embedding_model: None,
            vectors_normalized: false,
            vector_index: VectorIndexConfig::Exact,
            tokenizer: Default::default(),
        };
        let collection = db.create_collection(config.clone()).unwrap();
        let before = std::fs::read(&path).unwrap();
        db.lock()
            .unwrap()
            .catalog
            .id_high_watermarks
            .insert("doc_".into(), u64::MAX);
        let error = collection
            .insert_document(DocumentInput {
                external_id: "doc".into(),
                title: None,
                source_uri: None,
                metadata: JsonValue::object(),
                chunks: vec![],
            })
            .unwrap_err();
        assert_eq!(error.code(), ErrorCode::FileFormatError);
        assert!(error
            .detail()
            .iter()
            .any(|(key, value)| key == "reason" && value == "id_space_exhausted"));
        db.lock()
            .unwrap()
            .catalog
            .id_high_watermarks
            .insert("col_".into(), u64::MAX);
        let mut config = config;
        config.name = "other".into();
        assert!(db.create_collection(config).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), before);
        let mut catalog = Catalog::default();
        catalog
            .id_high_watermarks
            .insert("chk_".into(), u64::MAX - 1);
        assert_eq!(
            catalog.next_id("chk_", 0).unwrap(),
            format!("chk_{}", u64::MAX)
        );
        assert!(catalog.next_id("chk_", 1).is_err());
        assert!(catalog.next_id("chk_", u64::MAX).is_err());
        drop(collection);
        drop(db);
        let _ = std::fs::remove_file(path);
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn oversized_binary_fields_do_not_poison_writer_or_change_file() {
        let path = temp_path("oversized-fields");
        let db = Database::create(&path, DatabaseConfig::default()).unwrap();
        let mut config = CollectionConfig {
            name: "notes".into(),
            embedding_dim: u32::MAX as usize + 1,
            distance_metric: DistanceMetric::Cosine,
            embedding_model: None,
            vectors_normalized: false,
            vector_index: VectorIndexConfig::Exact,
            tokenizer: Default::default(),
        };
        let before = std::fs::read(&path).unwrap();
        assert_eq!(
            db.create_collection(config.clone()).err().unwrap().code(),
            ErrorCode::InvalidEmbeddingDimension
        );
        assert_eq!(std::fs::read(&path).unwrap(), before);
        config.embedding_dim = 2;
        let collection = db.create_collection(config).unwrap();
        let before = std::fs::read(&path).unwrap();
        let mut document = DocumentInput {
            external_id: "doc".into(),
            title: None,
            source_uri: None,
            metadata: JsonValue::object(),
            chunks: vec![ChunkInput {
                chunk_external_id: "doc#0".into(),
                text: "valid text".into(),
                ordinal: u32::MAX as usize + 1,
                metadata: JsonValue::object(),
                source_span: None,
            }],
        };
        assert_eq!(
            collection
                .insert_document(document.clone())
                .unwrap_err()
                .code(),
            ErrorCode::InvalidJsonlSchema
        );
        document.chunks[0].ordinal = 0;
        document.chunks[0].source_span = Some(crate::SourceSpan {
            start: None,
            end: None,
            line_start: Some(1),
            line_end: None,
        });
        assert_eq!(
            collection
                .insert_document(document.clone())
                .unwrap_err()
                .code(),
            ErrorCode::InvalidJsonlSchema
        );
        assert_eq!(std::fs::read(&path).unwrap(), before);
        document.chunks[0].source_span = None;
        collection.insert_document(document).unwrap();
        drop(collection);
        drop(db);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn close_releases_writer_lock_explicitly() {
        let path = temp_path("explicit-close");
        let db = Database::create(&path, DatabaseConfig::default()).unwrap();
        let err = match Database::open(&path) {
            Ok(_) => panic!("second writer should fail while the first handle is open"),
            Err(err) => err,
        };
        assert_eq!(err.code(), ErrorCode::LockContention);

        db.close().unwrap();
        let reopened = Database::open(&path).unwrap();
        reopened.close().unwrap();
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn check_path_runs_while_writer_lock_is_held() {
        let path = temp_path("check-during-writer");
        let db = Database::create(&path, DatabaseConfig::default()).unwrap();
        let report = Database::check_path(
            &path,
            CheckOptions {
                verify_indexes: false,
            },
        )
        .unwrap();
        assert_eq!(report.status, "ok");
        assert!(report.critical_end_offset > 0);
        assert_eq!(report.index_suffix_status, "missing");
        assert_eq!(report.index_check, "disabled");
        drop(db);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn check_reports_full_checkpoint_after_committed_write() {
        let path = temp_path("metadata-checkpoint");
        let db = Database::create(&path, DatabaseConfig::default()).unwrap();
        db.create_collection(CollectionConfig {
            tokenizer: Default::default(),
            name: "notes".into(),
            embedding_dim: 2,
            distance_metric: DistanceMetric::Cosine,
            embedding_model: None,
            vectors_normalized: false,
            vector_index: VectorIndexConfig::Exact,
        })
        .unwrap();

        let report = Database::check_path(
            &path,
            CheckOptions {
                verify_indexes: true,
            },
        )
        .unwrap();
        assert_eq!(report.status, "ok");
        assert_eq!(report.index_suffix_status, "persistent_checkpoint");
        assert_eq!(
            report.index_suffix.as_ref().map(|suffix| (
                suffix.index_record_count,
                suffix.index_extent_count,
                suffix.text_segment_count,
                suffix.vector_segment_count
            )),
            Some((3, 3, 1, 1))
        );
        assert_eq!(report.index_check, "ok");
        assert_eq!(report.index_check_reason, None);
        drop(db);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn check_reports_aggregate_payload_checksum_mismatch_without_rebuild() {
        let path = temp_path("aggregate-checksum-warning");
        let db = Database::create(&path, DatabaseConfig::default()).unwrap();
        db.create_collection(CollectionConfig {
            tokenizer: Default::default(),
            name: "notes".into(),
            embedding_dim: 2,
            distance_metric: DistanceMetric::Cosine,
            embedding_model: None,
            vectors_normalized: false,
            vector_index: VectorIndexConfig::Exact,
        })
        .unwrap();
        let initial_report = Database::check_path(
            &path,
            CheckOptions {
                verify_indexes: true,
            },
        )
        .unwrap();
        let suffix_offset = initial_report.index_suffix.as_ref().unwrap().offset;
        drop(db);

        let mut file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .unwrap();
        file.seek(SeekFrom::Start(suffix_offset + 101)).unwrap();
        file.write_all(&u32::MAX.to_le_bytes()).unwrap();
        drop(file);

        let report = Database::check_path(
            &path,
            CheckOptions {
                verify_indexes: true,
            },
        )
        .unwrap();
        assert_eq!(report.status, "warning");
        assert_eq!(report.index_suffix_status, "persistent_checkpoint");
        assert_eq!(report.index_check, "ok");
        assert_eq!(report.index_check_reason, None);
        assert!(report.issues.iter().any(|finding| {
            finding.code == "checkpoint_payload_checksum_mismatch"
                && finding.stream == "index"
                && finding.visibility_impact == "rebuild_index_only"
                && finding.repair_available
        }));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn check_reports_full_checkpoint_segments_for_live_data() {
        let path = temp_path("full-checkpoint-live-data");
        let db = Database::create(&path, DatabaseConfig::default()).unwrap();
        let col = db
            .create_collection(CollectionConfig {
                tokenizer: Default::default(),
                name: "notes".into(),
                embedding_dim: 2,
                distance_metric: DistanceMetric::Cosine,
                embedding_model: Some("m".into()),
                vectors_normalized: false,
                vector_index: VectorIndexConfig::Exact,
            })
            .unwrap();
        col.insert_document(DocumentInput {
            external_id: "doc-1".into(),
            title: None,
            source_uri: None,
            metadata: JsonValue::object(),
            chunks: vec![crate::model::ChunkInput {
                chunk_external_id: "doc-1#1".into(),
                text: "wal recovery wal".into(),
                ordinal: 0,
                metadata: JsonValue::object(),
                source_span: None,
            }],
        })
        .unwrap();
        col.put_embeddings(vec![EmbeddingInput {
            chunk_external_id: "doc-1#1".into(),
            model_name: "m".into(),
            dim: 2,
            vector: vec![1.0, 0.0],
        }])
        .unwrap();

        let report = Database::check_path(
            &path,
            CheckOptions {
                verify_indexes: true,
            },
        )
        .unwrap();
        let suffix = report.index_suffix.as_ref().unwrap();
        assert_eq!(report.index_suffix_status, "persistent_checkpoint");
        assert_eq!(report.index_check, "ok");
        assert_eq!(report.index_check_reason, None);
        assert_eq!(suffix.index_record_count, 3);
        assert_eq!(suffix.index_extent_count, 3);
        assert_eq!(suffix.collection_count, 1);
        assert_eq!(suffix.live_chunk_count, 1);
        assert_eq!(suffix.live_embedding_count, 1);
        assert_eq!(suffix.text_segment_count, 1);
        assert_eq!(suffix.vector_segment_count, 1);
        assert!(suffix.total_size_bytes() > suffix.payload_len);
        let index_stats = db.index_stats().unwrap();
        assert!(!index_stats.rebuild_needed);
        assert_eq!(index_stats.index_size_bytes, suffix.total_size_bytes());
        assert_eq!(
            index_stats.last_index_commit_id,
            Some(suffix.covered_last_commit_sequence_no)
        );
        assert!(suffix.covered_last_commit_sequence_no > suffix.covered_last_batch_id);
        let expected_index_commit_sequence_no = suffix.covered_last_commit_sequence_no;
        drop(col);
        drop(db);
        let reopened = Database::open(&path).unwrap();
        let reopened_index_stats = reopened.index_stats().unwrap();
        assert!(!reopened_index_stats.rebuild_needed);
        assert_eq!(
            reopened_index_stats.index_size_bytes,
            suffix.total_size_bytes()
        );
        assert_eq!(
            reopened_index_stats.last_index_commit_id,
            Some(expected_index_commit_sequence_no)
        );
        drop(reopened);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn crash_after_suffix_discard_before_next_commit_preserves_main_data() {
        let path = temp_path("suffix-discard-crash");
        let db = Database::create(&path, DatabaseConfig::default()).unwrap();
        let col = db
            .create_collection(CollectionConfig {
                tokenizer: Default::default(),
                name: "notes".into(),
                embedding_dim: 2,
                distance_metric: DistanceMetric::Cosine,
                embedding_model: Some("m".into()),
                vectors_normalized: false,
                vector_index: VectorIndexConfig::Exact,
            })
            .unwrap();
        col.insert_document(DocumentInput {
            external_id: "doc-1".into(),
            title: Some("durability".into()),
            source_uri: None,
            metadata: JsonValue::object(),
            chunks: vec![ChunkInput {
                chunk_external_id: "doc-1#1".into(),
                text: "suffix discard crash keeps committed main data visible".into(),
                ordinal: 0,
                metadata: JsonValue::object(),
                source_span: None,
            }],
        })
        .unwrap();
        col.put_embeddings(vec![EmbeddingInput {
            chunk_external_id: "doc-1#1".into(),
            model_name: "m".into(),
            dim: 2,
            vector: vec![1.0, 0.0],
        }])
        .unwrap();
        drop(col);
        drop(db);

        let before = Database::check_path(
            &path,
            CheckOptions {
                verify_indexes: true,
            },
        )
        .unwrap();
        assert_eq!(before.index_suffix_status, "persistent_checkpoint");
        assert_eq!(before.index_check, "ok");
        assert!(std::fs::metadata(&path).unwrap().len() > before.critical_end_offset);

        std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(before.critical_end_offset)
            .unwrap();

        let after = Database::check_path(
            &path,
            CheckOptions {
                verify_indexes: true,
            },
        )
        .unwrap();
        assert_eq!(after.status, "ok");
        assert_eq!(after.last_valid_commit, before.last_valid_commit);
        assert_eq!(after.critical_end_offset, before.critical_end_offset);
        assert_eq!(after.index_suffix_status, "missing");
        assert_eq!(after.index_check, "skipped");
        assert_eq!(
            after.index_check_reason.as_deref(),
            Some("persistent_index_checkpoint_missing")
        );
        assert!(after
            .issues
            .iter()
            .all(|issue| issue.visibility_impact != "blocks_main"));

        let reopened = Database::open(&path).unwrap();
        let index_stats = reopened.index_stats().unwrap();
        assert!(index_stats.rebuild_needed);
        assert_eq!(index_stats.index_size_bytes, 0);
        let results = reopened
            .collection("notes")
            .unwrap()
            .search()
            .text("committed main data")
            .top_k(5)
            .execute()
            .unwrap();
        assert_eq!(results.results.len(), 1);
        assert_eq!(results.results[0].chunk_external_id, "doc-1#1");
        let results_json = results.to_json();
        let query_context = results_json
            .get("query_context")
            .and_then(JsonValue::as_object)
            .unwrap();
        assert_eq!(
            query_context
                .get("index_source")
                .and_then(JsonValue::as_str),
            Some("transient_rebuild")
        );
        assert_eq!(
            query_context
                .get("index_rebuild_reason")
                .and_then(JsonValue::as_str),
            Some("in_memory_index_from_committed_snapshot")
        );
        assert!(query_context
            .get("transient_rebuild")
            .and_then(JsonValue::as_object)
            .and_then(|context| context.get("bytes_read"))
            .and_then(JsonValue::as_f64)
            .map(|bytes| bytes > 0.0)
            .unwrap_or(false));
        assert_eq!(
            results.results[0]
                .explain
                .get("index_source")
                .and_then(JsonValue::as_str),
            Some("transient_rebuild")
        );
        assert_eq!(
            results.results[0]
                .explain
                .get("index_rebuild_reason")
                .and_then(JsonValue::as_str),
            Some("in_memory_index_from_committed_snapshot")
        );
        drop(reopened);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn vector_filter_preflight_needs_no_corpus_sized_id_copy() {
        let path = query_preflight_fixture("vector-filter-workspace", "stable content");
        let db = Database::open_read_only(&path).unwrap();
        let inner = db.lock().unwrap();
        let collection = inner.catalog.collection_by_name("first").unwrap();
        let embeddings = inner
            .catalog
            .live_embeddings()
            .filter(|embedding| embedding.collection_id == collection.collection_id)
            .collect::<Vec<_>>();
        let payload = embeddings
            .iter()
            .map(|embedding| {
                inner
                    .catalog
                    .record_ref_for_embedding(&embedding.embedding_id)
                    .unwrap()
                    .payload
                    .len
            })
            .max()
            .unwrap();
        for limit in 0..100 {
            let estimated = transient_query_scratch_bytes(
                &inner.catalog,
                &collection.collection_id,
                None,
                TextSearchOptions::scores_only(limit),
                true,
                limit,
            );
            assert_eq!(
                estimated,
                3 * payload + embeddings.len().min(limit) as u64 * 512
            );
        }
        drop(inner);
        drop(db);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn vector_filter_matches_materialized_reference_without_extra_workspace() {
        let path = query_preflight_fixture("vector-filter-predicate", "stable content");
        let db = Database::open_read_only(&path).unwrap();
        let template = db.lock().unwrap().catalog.clone();
        let context = db.json_context();
        let limit = context.available_workspace_bytes();
        for seed in 0..100 {
            let mut catalog = template.clone();
            for (index, chunk) in catalog.chunks.values_mut().enumerate() {
                chunk.metadata = parse(&format!(
                    r#"{{"score":{},"flag":{},"tags":["rust",{}],"null":null}}"#,
                    (index + seed) % 5,
                    index % 2 == 0,
                    index % 3,
                ))
                .unwrap();
            }
            for document in catalog.documents.values_mut() {
                document.metadata = parse(r#"{"region":"east","score":3}"#).unwrap();
            }
            match seed % 4 {
                1 => {
                    catalog
                        .deleted_documents
                        .insert(catalog.documents.keys().next().unwrap().clone());
                }
                2 => {
                    catalog
                        .deleted_collections
                        .insert(catalog.collections.keys().next().unwrap().clone());
                }
                3 => {
                    catalog
                        .deleted_chunks
                        .extend(catalog.chunks.keys().step_by(2).cloned());
                }
                _ => {}
            }
            for json in [
                r#"{"chunk.score":2}"#,
                r#"{"chunk.score":{"$ne":2}}"#,
                r#"{"chunk.score":{"$gt":2}}"#,
                r#"{"chunk.score":{"$gte":2}}"#,
                r#"{"chunk.score":{"$lt":2}}"#,
                r#"{"chunk.score":{"$lte":2}}"#,
                r#"{"chunk.score":{"$in":[1,3]}}"#,
                r#"{"chunk.tags":{"$contains":"rust"}}"#,
                r#"{"chunk.flag":false}"#,
                r#"{"chunk.null":null}"#,
                r#"{"chunk.missing":{"$ne":null}}"#,
                r#"{"$and":[{"doc.region":"east"},{"chunk.score":{"$gte":2}}]}"#,
            ] {
                let expr = FilterExpr::parse(parse(json).unwrap()).unwrap().unwrap();
                for collection_id in catalog.collections.keys() {
                    let expected = catalog
                        .live_chunks()
                        .filter(|chunk| &chunk.collection_id == collection_id)
                        .filter(|chunk| expr.matches_for_chunk(&catalog, chunk))
                        .map(|chunk| chunk.chunk_id.as_str())
                        .collect::<BTreeSet<_>>();
                    let mut held = context.workspace_reservation();
                    held.grow_to(limit, limit, "operation_scratch").unwrap();
                    for id in catalog.chunks.keys().map(String::as_str).chain(["missing"]) {
                        assert_eq!(
                            matches_vector_filter(&catalog, collection_id, Some(&expr), id),
                            expected.contains(id),
                            "seed={seed}, filter={json}, id={id}"
                        );
                        assert!(matches_vector_filter(&catalog, collection_id, None, id));
                    }
                }
            }
        }
        context
            .workspace_reservation()
            .grow_to(limit, limit, "operation_scratch")
            .unwrap();
        drop(db);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn vector_predicate_prunes_before_loading_and_counts_only_scored_vectors() {
        use std::cell::Cell;
        let path = query_preflight_fixture("vector-filter-scan-count", "stable content");
        let db = Database::open_read_only(&path).unwrap();
        let inner = db.lock().unwrap();
        let collection = inner.catalog.collection_by_name("second").unwrap();
        let base = inner
            .lazy_vector_source
            .as_ref()
            .unwrap()
            .load(&inner.storage, &collection.collection_id)
            .unwrap()
            .unwrap();
        let context = db.json_context();
        let budget = context.available_workspace_bytes();
        for seed in 0..100 {
            let mut segment = base.clone();
            for (index, entry) in segment.vectors.iter_mut().enumerate() {
                entry.vector = vec![((index + seed) % 3) as f32, 1.0];
                if seed % 7 == 0 && index == 0 {
                    entry.vector.push(0.0);
                }
            }
            let included = |id: &str| {
                let index = segment
                    .vectors
                    .iter()
                    .position(|entry| entry.chunk_id == id)
                    .unwrap();
                match seed % 4 {
                    0 => true,
                    1 => false,
                    2 => index % 2 == 0,
                    _ => index == 0,
                }
            };
            let metric = [
                DistanceMetric::Cosine,
                DistanceMetric::DotProduct,
                DistanceMetric::Euclidean,
            ][seed % 3];
            let top_k = seed % 6;
            let mut expected = segment
                .vectors
                .iter()
                .filter(|entry| included(&entry.chunk_id) && entry.vector.len() == 2)
                .map(|entry| {
                    let raw = raw_score(&[1.0, 0.0], &entry.vector, metric);
                    (&entry.chunk_id, raw, normalize_score(raw, metric, false))
                })
                .collect::<Vec<_>>();
            let count = expected.len();
            expected.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap().then_with(|| a.0.cmp(b.0)));
            expected.truncate(top_k);
            let mut held = context.workspace_reservation();
            if seed % 4 == 1 {
                held.grow_to(budget, budget, "operation_scratch").unwrap();
            }
            for cached in [false, true] {
                let predicates = Cell::new(0);
                let loads = Cell::new(0);
                let include = |id: &str| {
                    predicates.set(predicates.get() + 1);
                    included(id)
                };
                let (hits, scanned) = if cached {
                    search_persistent_vector_segment(
                        &segment,
                        &[1.0, 0.0],
                        metric,
                        top_k,
                        include,
                        &context,
                    )
                    .unwrap()
                } else {
                    ExactVectorIndex
                        .search_with_vectors(
                            &inner.catalog.visible_snapshot(),
                            &collection.collection_id,
                            &[1.0, 0.0],
                            VectorSearchOptions {
                                metric,
                                vectors_normalized: false,
                                limit: top_k,
                            },
                            include,
                            &context,
                            |visible| {
                                assert!(included(&visible.embedding.chunk_id));
                                loads.set(loads.get() + 1);
                                Ok(&segment
                                    .vectors
                                    .iter()
                                    .find(|entry| entry.chunk_id == visible.embedding.chunk_id)
                                    .unwrap()
                                    .vector)
                            },
                        )
                        .unwrap()
                };
                assert_eq!(predicates.get(), segment.vectors.len());
                assert_eq!(
                    loads.get(),
                    if cached {
                        0
                    } else {
                        segment
                            .vectors
                            .iter()
                            .filter(|entry| included(&entry.chunk_id))
                            .count()
                    }
                );
                assert_eq!(scanned.scans, segment.vectors.len());
                assert_eq!(scanned.distance_ops, count * 2);
                assert_eq!(hits.len(), expected.len());
                for (hit, (id, raw, normalized)) in hits.iter().zip(&expected) {
                    assert_eq!(&hit.chunk_id, *id);
                    assert_eq!(hit.raw_score.to_bits(), raw.to_bits());
                    assert_eq!(hit.normalized_score.to_bits(), normalized.to_bits());
                }
            }
            drop(held);
            context
                .workspace_reservation()
                .grow_to(budget, budget, "operation_scratch")
                .unwrap();
        }
        drop(inner);
        drop(db);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn vector_candidate_quota_is_shared_by_cached_streamed_and_primary_queries() {
        fn search(
            mode: usize,
            inner: &mut Inner,
            collection: &CollectionInfo,
            segment: &VectorIndexSegmentPayload,
            context: &crate::json::JsonOperationContext,
            limit: usize,
        ) -> Result<VectorHits> {
            match mode {
                0 => search_persistent_vector_segment(
                    segment,
                    &[1.0, 0.0],
                    DistanceMetric::Cosine,
                    limit,
                    |_| true,
                    context,
                )
                .map(|(hits, _)| hits),
                1 => inner
                    .search_uncached_vector_index(collection, &[1.0, 0.0], limit, None, context)
                    .map(|value| value.unwrap().0),
                _ => ExactVectorIndex
                    .search_with_vectors(
                        &inner.catalog.visible_snapshot(),
                        &collection.collection_id,
                        &[1.0, 0.0],
                        VectorSearchOptions {
                            metric: DistanceMetric::Cosine,
                            vectors_normalized: false,
                            limit,
                        },
                        |_| true,
                        context,
                        |visible| {
                            Ok(&segment
                                .vectors
                                .iter()
                                .find(|entry| entry.chunk_id == visible.embedding.chunk_id)
                                .unwrap()
                                .vector)
                        },
                    )
                    .map(|(hits, _)| hits),
            }
        }
        let path = query_preflight_fixture("vector-candidate-quota", "candidate text");
        let db = Database::open_read_only(&path).unwrap();
        let before = fs::read(&path).unwrap();
        let context = db.json_context();
        let limit = crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES;
        let capacity = context.available_workspace_bytes();
        let mut inner = db.lock().unwrap();
        let collection = inner.catalog.collection_by_name("first").unwrap().clone();
        let segment = inner
            .lazy_vector_source
            .as_ref()
            .unwrap()
            .load(&inner.storage, &collection.collection_id)
            .unwrap()
            .unwrap();
        inner.persistent_vector_segments = None;
        inner.lazy_vector_segment = None;
        let signature = inner.catalog.state_signature();
        let id_bytes = segment.vectors[0].chunk_id.len();
        let candidate_bytes = std::mem::size_of::<VectorHit>();
        let mut expected = segment
            .vectors
            .iter()
            .map(|entry| entry.chunk_id.clone())
            .collect::<Vec<_>>();
        expected.sort();
        for seed in 0..100 {
            let available = [0, id_bytes - 1, id_bytes + candidate_bytes - 1][seed % 3];
            let mut held = context.workspace_reservation();
            held.grow_to(
                capacity - available,
                capacity - available,
                "operation_scratch",
            )
            .unwrap();
            for mode in 0..3 {
                let error = search(
                    mode,
                    &mut inner,
                    &collection,
                    &segment,
                    &context,
                    1 + seed % 3,
                )
                .unwrap_err();
                assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                let scratch_bytes = if mode == 1 {
                    collection.embedding_dim * 4
                } else {
                    0
                };
                let needed = if available < scratch_bytes {
                    scratch_bytes
                } else if available < scratch_bytes + id_bytes {
                    scratch_bytes + id_bytes
                } else {
                    scratch_bytes + id_bytes + candidate_bytes
                };
                for (key, value) in [
                    ("resource", "memory".to_string()),
                    ("operation", "operation_scratch".into()),
                    ("limit_bytes", limit.to_string()),
                    ("requested_bytes", (limit - available + needed).to_string()),
                ] {
                    assert_eq!(error_detail_value(&error, key), Some(value.as_str()));
                }
                assert_eq!(inner.catalog.state_signature(), signature);
                assert_eq!(
                    inner.index_suffix_status,
                    IndexSuffixState::PersistentCheckpoint
                );
                held.grow_to(capacity, capacity, "operation_scratch")
                    .unwrap();
                held.retain(capacity - available);
            }
            held.retain(0);
            for mode in 0..3 {
                let hits = search(
                    mode,
                    &mut inner,
                    &collection,
                    &segment,
                    &context,
                    1 + seed % 3,
                )
                .unwrap();
                assert_eq!(
                    hits.iter()
                        .map(|hit| hit.chunk_id.as_str())
                        .collect::<Vec<_>>(),
                    expected
                        .iter()
                        .take(1 + seed % 3)
                        .map(String::as_str)
                        .collect::<Vec<_>>()
                );
                assert!(hits
                    .iter()
                    .all(|hit| hit.raw_score == 1.0 && hit.normalized_score == 1.0));
                drop(hits);
                held.grow_to(capacity, capacity, "operation_scratch")
                    .unwrap();
                held.retain(0);
            }
        }
        assert_eq!(fs::read(&path).unwrap(), before);
        drop(inner);
        let result = db
            .collection("first")
            .unwrap()
            .search()
            .query_embedding(QueryEmbeddingInput {
                model_name: Some("m".into()),
                dim: 2,
                vector: vec![1.0, 0.0],
            })
            .top_k(2)
            .execute()
            .unwrap();
        assert_eq!(result.results.len(), 2);
        drop(db);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn compact_memory_pressure_preserves_the_effective_file_and_retries_only_the_suffix() {
        let source = query_preflight_fixture("compact-memory-boundaries", "shared memory");
        let mut rescued = 0;
        let mut refused = 0;
        for (remaining, parent) in [false, true].into_iter().flat_map(|parent| {
            (32..=3200)
                .step_by(32)
                .map(move |remaining| (remaining, parent))
        }) {
            let path = temp_path("compact-memory-after-publication");
            fs::copy(&source, &path).unwrap();
            let db = Database::open_with_options(
                &path,
                DatabaseOpenOptions {
                    cache_budget_mb: Some(4096),
                    ..Default::default()
                },
            )
            .unwrap();
            let reader = Database::open_read_only(&path).unwrap();
            let old_state = reader.lock().unwrap().catalog.state_signature();
            let before = db.stats().unwrap();
            let calls = db.lock().unwrap().index_checkpoint_refreshes;
            db.lock().unwrap().compact_checkpoint_remaining = Some((remaining, parent));
            let summary = db
                .compact(CompactOptions {
                    rebuild_indexes: true,
                })
                .unwrap();
            let inner = db.lock().unwrap();
            assert_eq!(inner.compact_rewrite_completions, 1);
            let attempts = inner.index_checkpoint_refreshes - calls;
            if summary.index_rebuild == "failed" {
                refused += 1;
                assert_eq!(attempts, 2);
                assert_eq!(inner.index_suffix_status, IndexSuffixState::RebuildNeeded);
                assert!(summary.index_rebuild_reason.is_some());
            } else {
                assert_eq!(summary.index_rebuild, "full_checkpoint_written");
                rescued += usize::from(attempts == 2);
            }
            let published_state = inner.catalog.state_signature();
            let published_commit = inner.catalog.last_commit_id;
            drop(inner);
            let after = db.stats().unwrap();
            assert_eq!(after.document_count, before.document_count);
            assert_eq!(after.chunk_count, before.chunk_count);
            assert_eq!(after.embedding_count, before.embedding_count);
            assert_eq!(reader.lock().unwrap().catalog.state_signature(), old_state);
            assert_eq!(
                reader
                    .collection("first")
                    .unwrap()
                    .search()
                    .text("shared")
                    .execute()
                    .unwrap()
                    .results
                    .len(),
                2
            );
            drop((reader, db));
            let reopened = Database::open_read_only(&path).unwrap();
            assert_eq!(
                reopened.lock().unwrap().catalog.state_signature(),
                published_state
            );
            assert_eq!(
                reopened.lock().unwrap().catalog.last_commit_id,
                published_commit
            );
            assert_eq!(
                reopened
                    .collection("first")
                    .unwrap()
                    .search()
                    .text("shared")
                    .execute()
                    .unwrap()
                    .results
                    .len(),
                2
            );
            drop(reopened);
            fs::remove_file(path).unwrap();
        }
        assert!(
            rescued > 0,
            "must observe a successful second suffix attempt after publication"
        );
        assert!(
            refused > 0,
            "must observe a refused second suffix attempt after publication"
        );

        let path = temp_path("compact-memory-before-publication");
        fs::copy(&source, &path).unwrap();
        let db = Database::open(&path).unwrap();
        let original = fs::read(&path).unwrap();
        let signature = db.lock().unwrap().catalog.state_signature();
        db.lock().unwrap().compact_file_remaining = Some(1);
        let error = db
            .compact(CompactOptions {
                rebuild_indexes: true,
            })
            .unwrap_err();
        assert!(crate::json::is_memory_budget_error(&error));
        assert_eq!(db.lock().unwrap().compact_rewrite_completions, 0);
        assert_eq!(db.lock().unwrap().catalog.state_signature(), signature);
        assert_eq!(fs::read(&path).unwrap(), original);
        drop(db);
        let reopened = Database::open_read_only(&path).unwrap();
        assert_eq!(
            reopened.lock().unwrap().catalog.state_signature(),
            signature
        );
        assert_eq!(
            reopened
                .collection("first")
                .unwrap()
                .search()
                .text("shared")
                .execute()
                .unwrap()
                .results
                .len(),
            2
        );
        drop(reopened);
        fs::remove_file(path).unwrap();
        fs::remove_file(source).unwrap();
    }

    #[test]
    fn checkpoint_scratch_retries_without_optional_cache_and_preserves_primary() {
        let path = query_preflight_fixture("checkpoint-scratch-retry", "shared memory");
        let db = Database::open_with_options(
            &path,
            DatabaseOpenOptions {
                cache_budget_mb: Some(4096),
                ..Default::default()
            },
        )
        .unwrap();
        let context = db.json_context();
        let mut inner = db.lock().unwrap();
        let primary_end = inner.storage.critical_end_offset() as usize;
        let before = fs::read(&path).unwrap()[..primary_end].to_vec();
        let signature = inner.catalog.state_signature();
        let commit = inner.catalog.last_commit_id;
        inner.text_index.clear_cache();
        inner.mark_index_checkpoint_rebuild_needed();
        let available = context.available_total_bytes();
        let mut rescued = 0;
        let mut refused = 0;
        for seed in 0..100 {
            let remaining = 32 + seed * 32;
            let run = |inner: &mut Inner, mode| {
                inner.mark_index_checkpoint_rebuild_needed();
                let mut text = inner
                    .prepare_text_checkpoint(None, &BTreeSet::new(), &context)
                    .unwrap();
                let blocker = context
                    .cache_reservation(
                        (context.available_total_bytes() - remaining) as u128,
                        usize::MAX,
                    )
                    .unwrap();
                let calls = inner.index_checkpoint_refreshes;
                let result = if mode == 2 {
                    inner.refresh_index_checkpoint_prepared(
                        VectorCache::new(&context),
                        &BTreeSet::new(),
                        text.take(),
                        &context,
                    )
                } else {
                    inner.refresh_index_checkpoint_attempt(
                        VectorCache::new(&context),
                        &BTreeSet::new(),
                        mode == 1,
                        text.as_mut(),
                        &context,
                    )
                };
                drop((blocker, text));
                (result, inner.index_checkpoint_refreshes - calls)
            };
            let (first, _) = run(&mut inner, 1);
            let first_failed = first
                .as_ref()
                .is_err_and(crate::json::is_memory_budget_error);
            let (baseline, _) = run(&mut inner, 0);
            let (actual, attempts) = run(&mut inner, 2);
            if first_failed {
                assert_eq!(attempts, 2);
            } else {
                assert_eq!(attempts, 1);
            }
            match (baseline, actual) {
                (Ok(_), Ok(report)) => {
                    rescued += usize::from(first_failed);
                    assert_eq!(report.unwrap().status, "persistent_checkpoint");
                }
                (Err(expected), Err(actual)) => {
                    refused += 1;
                    assert_eq!(actual.code(), expected.code());
                    assert_eq!(actual.detail(), expected.detail());
                    assert_eq!(inner.index_suffix_status, IndexSuffixState::RebuildNeeded);
                    assert!(inner.persistent_vector_segments.is_none());
                    let reader = Database::open_with_options(
                        &path,
                        DatabaseOpenOptions {
                            read_only: true,
                            ..Default::default()
                        },
                    )
                    .unwrap();
                    assert_eq!(reader.lock().unwrap().catalog.state_signature(), signature);
                    assert_eq!(reader.lock().unwrap().catalog.last_commit_id, commit);
                }
                (Err(_), Ok(_)) => {} // A reusable cache can need less scratch than reading primary records.
                (Ok(_), Err(error)) => {
                    panic!("optional cache blocked viable checkpoint: {error:?}")
                }
            }
            assert_eq!(inner.catalog.state_signature(), signature);
            assert_eq!(inner.catalog.last_commit_id, commit);
            assert_eq!(&fs::read(&path).unwrap()[..primary_end], &before);
            inner.mark_index_checkpoint_rebuild_needed();
            assert!(context
                .cache_reservation(available as u128, usize::MAX)
                .is_some());
        }
        assert!(
            rescued > 0,
            "must exercise cache-only failure followed by successful streaming retry"
        );
        assert!(refused > 0, "must exercise genuine minimum-scratch refusal");
        inner.refresh_index_checkpoint(&context).unwrap();
        drop(inner);
        drop(db);
        let reopened = Database::open(&path).unwrap();
        assert_eq!(
            reopened.lock().unwrap().catalog.state_signature(),
            signature
        );
        drop(reopened);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn checkpoint_directory_miss_keeps_valid_indexes_and_primary_data() {
        let path = query_preflight_fixture("vector-directory-miss", "budgeted directory");
        let db = Database::open_with_options(
            &path,
            DatabaseOpenOptions {
                cache_budget_mb: Some(4096),
                ..Default::default()
            },
        )
        .unwrap();
        let context = db.json_context();
        let mut inner = db.lock().unwrap();
        let before = fs::read(&path).unwrap();
        let end = inner.storage.critical_end_offset() as usize;
        let signature = inner.catalog.state_signature();
        let collection = inner.catalog.collection_by_name("second").unwrap().clone();
        inner.text_index.clear_cache();
        let payload_budget = inner
            .catalog
            .live_collections()
            .map(|collection| {
                let mut source = VectorRecordSegment::new(
                    collection,
                    inner.last_commit_sequence_no,
                    inner
                        .catalog
                        .live_embeddings()
                        .filter(|entry| entry.collection_id == collection.collection_id)
                        .map(|entry| {
                            (
                                entry,
                                inner.catalog.record_ref_for_embedding(&entry.embedding_id),
                            )
                        }),
                    inner.storage.payload_reader().unwrap(),
                    &context,
                )
                .unwrap();
                source
                    .try_materialize(usize::MAX)
                    .unwrap()
                    .unwrap()
                    .reserved_bytes()
            })
            .max()
            .unwrap();
        assert!(
            payload_budget + 100 < crate::json::btree_workspace::<String, CachedVectorSegment>(1)
        );
        let summary_bytes =
            inner.catalog.live_collections().count() * std::mem::size_of::<(u64, usize)>();
        inner.mark_index_checkpoint_rebuild_needed();
        let available = context.available_total_bytes();
        for seed in 0..100 {
            inner.mark_index_checkpoint_rebuild_needed();
            let text = inner
                .prepare_text_checkpoint(None, &BTreeSet::new(), &context)
                .unwrap();
            let blocker = context
                .cache_reservation(
                    (context.available_total_bytes() - payload_budget - summary_bytes - seed)
                        as u128,
                    usize::MAX,
                )
                .unwrap();
            let report = inner
                .refresh_index_checkpoint_prepared(
                    VectorCache::new(&context),
                    &BTreeSet::new(),
                    text,
                    &context,
                )
                .unwrap()
                .unwrap();
            assert_eq!(report.status, "persistent_checkpoint");
            assert!(inner
                .persistent_vector_segments
                .as_ref()
                .unwrap()
                .is_empty());
            assert_eq!(
                inner
                    .persistent_vector_segments
                    .as_ref()
                    .unwrap()
                    .reserved_bytes(),
                0
            );
            assert_eq!(&fs::read(&path).unwrap()[..end], &before[..end]);
            assert_eq!(inner.catalog.state_signature(), signature);
            drop(blocker);
            let directory = inner.lazy_vector_source.as_ref().unwrap().reserved_bytes();
            assert!(context
                .cache_reservation((available - directory) as u128, usize::MAX)
                .is_some());
            assert!(context
                .cache_reservation((available - directory + 1) as u128, usize::MAX)
                .is_none());
            let (hits, count) = inner
                .search_uncached_vector_index(&collection, &[1.0, 0.0], 3, None, &context)
                .unwrap()
                .unwrap();
            assert_eq!(count, 4);
            assert_eq!(hits.len(), 3);
            assert!(hits.iter().all(|hit| hit.raw_score == 1.0));
            assert_eq!(
                inner.index_suffix_status,
                IndexSuffixState::PersistentCheckpoint
            );
        }
        inner
            .prepare_vector_index(&collection.collection_id, 4096)
            .unwrap();
        assert_eq!(inner.lazy_vector_segment.as_ref().unwrap().vectors.len(), 4);
        drop(inner);
        drop(db);
        let reopened = Database::open_read_only(&path).unwrap();
        assert!(!reopened.index_stats().unwrap().rebuild_needed);
        drop(reopened);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn shared_memory_cache_miss_streams_without_invalidating_and_releases_on_eviction() {
        let path = query_preflight_fixture("vector-cache-parent", "budgeted cache");
        let db = Database::open_read_only(&path).unwrap();
        let before = fs::read(&path).unwrap();
        let context = db.json_context();
        let mut inner = db.lock().unwrap();
        let collection = inner.catalog.collection_by_name("second").unwrap().clone();
        let signature = inner.catalog.state_signature();
        let available = context.available_total_bytes();
        for seed in 0..100 {
            let blocker = context
                .cache_reservation(
                    (context.available_total_bytes() - 1024 - seed) as u128,
                    usize::MAX,
                )
                .unwrap();
            inner
                .prepare_vector_index(&collection.collection_id, 4096)
                .unwrap();
            assert!(inner.lazy_vector_segment.is_none());
            assert_eq!(
                inner.index_suffix_status,
                IndexSuffixState::PersistentCheckpoint
            );
            let (hits, count) = inner
                .search_uncached_vector_index(&collection, &[1.0, 0.0], 3, None, &context)
                .unwrap()
                .unwrap();
            assert_eq!(count, 4);
            assert_eq!(hits.len(), 3);
            assert!(hits.iter().all(|hit| hit.raw_score == 1.0));
            let expected = hits
                .iter()
                .map(|hit| hit.chunk_id.clone())
                .collect::<Vec<_>>();
            drop(hits);
            drop(blocker);
            inner
                .prepare_vector_index(&collection.collection_id, 4096)
                .unwrap();
            let cached = inner.lazy_vector_segment.as_ref().unwrap();
            assert_eq!(
                cached
                    .vectors
                    .iter()
                    .take(3)
                    .map(|entry| entry.chunk_id.clone())
                    .collect::<Vec<_>>(),
                expected
            );
            assert!(context
                .cache_reservation((context.available_total_bytes() + 1) as u128, usize::MAX)
                .is_none());
            let rest = context
                .cache_reservation(context.available_total_bytes() as u128, usize::MAX)
                .unwrap();
            drop(rest);
            inner
                .prepare_vector_index(&collection.collection_id, 0)
                .unwrap();
            assert!(inner.lazy_vector_segment.is_none());
            assert!(inner.lazy_vector_source.is_some());
            assert!(context
                .cache_reservation(available as u128, usize::MAX)
                .is_some());
            assert_eq!(inner.catalog.state_signature(), signature);
            assert_eq!(
                inner.index_suffix_status,
                IndexSuffixState::PersistentCheckpoint
            );
        }
        assert_eq!(fs::read(&path).unwrap(), before);
        drop(inner);
        drop(db);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn streamed_vectors_retry_without_buffering_and_discard_partial_candidates() {
        let path = query_preflight_fixture("vector-buffer-retry", "budgeted stream");
        let db = Database::open_read_only(&path).unwrap();
        let before = fs::read(&path).unwrap();
        let context = db.json_context();
        let limit = context.available_workspace_bytes();
        let mut inner = db.lock().unwrap();
        let collection = inner.catalog.collection_by_name("second").unwrap().clone();
        let source = inner.lazy_vector_source.as_ref().unwrap().clone();
        let segment = source
            .load(&inner.storage, &collection.collection_id)
            .unwrap()
            .unwrap();
        inner.persistent_vector_segments = None;
        inner.lazy_vector_segment = None;
        let signature = inner.catalog.state_signature();
        let preferred = source
            .encoded_len(&collection.collection_id)
            .unwrap()
            .min(64 * 1024) as usize;
        let array = collection.embedding_dim * 4;
        let candidate = segment.vectors[0].chunk_id.len() + std::mem::size_of::<VectorHit>();
        let expected = segment
            .vectors
            .iter()
            .take(3)
            .map(|entry| entry.chunk_id.as_str())
            .collect::<Vec<_>>();
        for seed in 0..100 {
            let mode = seed % 3;
            let available = if mode == 0 {
                array + 3 * candidate
            } else {
                preferred + array + mode * candidate
            };
            let mut held = context.workspace_reservation();
            held.grow_to(limit - available, limit - available, "operation_scratch")
                .unwrap();
            let input = source
                .reserve_read_buffer(&collection.collection_id, &context)
                .unwrap();
            let mut provisional = VectorCandidates::new(3, &context);
            let mut seen = 0;
            let error = source
                .visit(&inner.storage, &collection, &input, |id, _| {
                    provisional.push(id, 1.0, 1.0)?;
                    seen += 1;
                    Ok(())
                })
                .unwrap_err();
            assert_eq!(error.code(), ErrorCode::ResourceExhausted);
            assert_eq!(seen, mode);
            drop(provisional);
            drop(input);
            let trace = crate::search_cost::SearchCostTrace::new();
            let measured = context
                .with_search_cost(&trace)
                .search_cost_stage(crate::search_cost::Stage::Recall);
            let (hits, count) = inner
                .search_uncached_vector_index(&collection, &[1.0, 0.0], 3, None, &measured)
                .unwrap()
                .unwrap();
            assert_eq!(
                count,
                segment.vectors.len() + if mode == 0 { 0 } else { mode + 1 }
            );
            assert_eq!(
                trace.total(crate::search_cost::Counter::VectorChecks),
                count as u64
            );
            assert_eq!(
                trace.total(crate::search_cost::Counter::DistanceComponents),
                2 * count as u64
            );
            // Even mode 0 reads/drains an initial frame before it can visit a
            // vector; zero provisional vector checks do not imply zero IO.
            let passes = 2;
            assert_eq!(
                trace.total(crate::search_cost::Counter::IndexReadBytes),
                source.encoded_len(&collection.collection_id).unwrap() * passes
            );
            assert_eq!(
                trace.total(crate::search_cost::Counter::IndexRequestedBytes),
                source.encoded_len(&collection.collection_id).unwrap() * passes
            );
            assert_eq!(
                hits.iter()
                    .map(|hit| hit.chunk_id.as_str())
                    .collect::<Vec<_>>(),
                expected
            );
            assert!(hits
                .iter()
                .all(|hit| hit.raw_score == 1.0 && hit.normalized_score == 1.0));
            drop(hits);
            assert_eq!(inner.catalog.state_signature(), signature);
            assert_eq!(
                inner.index_suffix_status,
                IndexSuffixState::PersistentCheckpoint
            );
            held.grow_to(limit, limit, "operation_scratch").unwrap();
        }
        assert_eq!(fs::read(&path).unwrap(), before);
        drop(inner);
        drop(db);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn why_not_score_only_ranking_avoids_unused_explanations_under_shared_pressure() {
        let query = (0..1000)
            .map(|i| format!("word{i}"))
            .collect::<Vec<_>>()
            .join(" ");
        let path = query_preflight_fixture("why-not-score-only-quota", &query);
        let db = Database::open(&path).unwrap();
        let context = db.json_context();
        let limit = context.available_workspace_bytes();
        let mut query_context = db
            .collection("first")
            .unwrap()
            .search()
            .text(&query)
            .top_k(1)
            .execute()
            .unwrap()
            .query_context;
        let before = fs::read(&path).unwrap();
        let mut inner = db.lock().unwrap();
        assert!(!inner.lazy_text);
        assert!(!inner.text_index.chunk_lengths.is_empty());
        let signature = inner.catalog.state_signature();
        let collection = inner.catalog.collection_by_name("first").unwrap().clone();
        let expected_id = inner
            .catalog
            .live_chunks()
            .filter(|chunk| chunk.collection_id == collection.collection_id)
            .map(|chunk| chunk.chunk_id.as_str())
            .max()
            .unwrap()
            .to_owned();
        let embedding = QueryEmbeddingInput {
            model_name: Some("m".into()),
            dim: 2,
            vector: vec![1.0, 0.0],
        };
        for seed in 0..100 {
            let vector = (seed % 2 == 0).then_some(&embedding);
            let included = seed % 3 != 0;
            let filter = match seed % 3 {
                0 => parse(r#"{"chunk.lang":"python"}"#).unwrap(),
                1 => JsonValue::object(),
                _ => parse(r#"{"chunk.lang":"rust"}"#).unwrap(),
            };
            let (filter_expr, filter_work) = FilterExpr::parse_validated_in(
                &filter,
                &context,
                crate::json::JsonInputKind::Filter,
            )
            .unwrap();
            query_context.candidate_limits = CandidateLimits {
                bm25: 1 + (seed / 2) % 2,
                vector: 1 + (seed / 2) % 2,
                fusion: 1 + (seed / 2) % 2,
            };
            query_context.embedding_present = vector.is_some();
            query_context.filter_present = filter_expr.is_some();
            query_context.weights.effective = if vector.is_some() {
                SearchWeights::new(0.5, 0.5, 0.0).unwrap()
            } else {
                SearchWeights::new(1.0, 0.0, 0.0).unwrap()
            };
            let estimate = |options| {
                transient_query_scratch_bytes(
                    &inner.catalog,
                    &collection.collection_id,
                    Some(&query),
                    options,
                    vector.is_some(),
                    2,
                )
            };
            let scores_bytes = estimate(TextSearchOptions::scores_only(2));
            let explained_bytes = estimate(TextSearchOptions::with_explanations(2));
            assert_eq!(explained_bytes - scores_bytes, 2 * query.len() as u64 * 256);
            query_context.transient_rebuild.budget_mb = scores_bytes.div_ceil(1024 * 1024) as usize;
            let old_error = ensure_transient_rebuild_budget(
                explained_bytes,
                query_context.transient_rebuild.budget_mb,
                query_context.profile,
            )
            .unwrap_err();
            assert_eq!(old_error.code(), ErrorCode::IndexCorruption);
            let mut held = context.workspace_reservation();
            let free = held.grow_to(0, limit, "operation_scratch").unwrap();
            for state in [
                IndexSuffixState::PersistentCheckpoint,
                IndexSuffixState::Missing,
            ] {
                let terms = crate::json::btree_workspace::<std::borrow::Cow<'_, str>, usize>(1000);
                let arrays = 1000
                    * if state == IndexSuffixState::PersistentCheckpoint {
                        2 * std::mem::size_of::<&str>() + std::mem::size_of::<f64>()
                    } else {
                        2 * std::mem::size_of::<usize>() + std::mem::size_of::<f64>()
                    };
                let legacy_arrays = 1000
                    * (std::mem::size_of::<std::borrow::Cow<'_, str>>()
                        + 2 * std::mem::size_of::<usize>()
                        + std::mem::size_of::<f64>());
                let available = 98304 + seed + terms + arrays - legacy_arrays;
                held.retain(free - available);
                inner.index_suffix_status = state;
                let error = inner
                    .text_index
                    .search(
                        &collection.collection_id,
                        &query,
                        TextSearchOptions::with_explanations(2),
                        &context,
                    )
                    .unwrap_err();
                assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                let actual = why_not_candidate_ranks(
                    &inner,
                    &collection,
                    &expected_id,
                    Some(&query),
                    vector,
                    &filter_expr,
                    &query_context,
                    None,
                    &context,
                    &mut false,
                )
                .unwrap();
                assert_eq!(actual.bm25_retrieval_rank, Some(2));
                assert_eq!(actual.bm25_rank, included.then_some(2));
                assert_eq!(
                    actual.vector_rank,
                    (included && vector.is_some()).then_some(2)
                );
                assert_eq!(
                    actual.fusion_rank,
                    (included && (seed / 2) % 2 == 1).then_some(2)
                );
                assert_eq!(
                    actual.fusion_retrieval_rank,
                    (included && (seed / 2) % 2 == 1).then_some(2)
                );
                assert_eq!(inner.catalog.state_signature(), signature);
                assert_eq!(inner.index_suffix_status, state);
                held.grow_to(free, free, "operation_scratch").unwrap();
            }
            inner.index_suffix_status = IndexSuffixState::PersistentCheckpoint;
            drop(held);
            drop(filter_work);
            context
                .workspace_reservation()
                .grow_to(limit, limit, "operation_scratch")
                .unwrap();
        }
        assert_eq!(fs::read(&path).unwrap(), before);
        drop(inner);
        let result = db
            .collection("first")
            .unwrap()
            .search()
            .text(&query)
            .top_k(2)
            .execute()
            .unwrap();
        assert_eq!(result.results.len(), 2);
        assert!(result.results.iter().all(|hit| hit
            .explain
            .get("matched_terms")
            .unwrap()
            .as_array()
            .unwrap()
            .len()
            == 1000));
        drop(db);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn bm25_candidate_quota_covers_cached_and_primary_explanations_before_success() {
        let query = (0..1000)
            .map(|i| format!("word{i}"))
            .collect::<Vec<_>>()
            .join(" ");
        let path = query_preflight_fixture("bm25-candidate-quota", &query);
        let db = Database::open(&path).unwrap();
        let context = db.json_context();
        let capacity = context.available_workspace_bytes();
        let before = fs::read(&path).unwrap();
        let inner = db.lock().unwrap();
        assert!(!inner.lazy_text);
        assert!(!inner.text_index.chunk_lengths.is_empty());
        let signature = inner.catalog.state_signature();
        let collection = inner.catalog.collection_by_name("first").unwrap();
        let mut reads = 0;
        let mut search = |cached: bool| -> Result<Bm25Hits> {
            if cached {
                inner.text_index.search(
                    &collection.collection_id,
                    &query,
                    TextSearchOptions::with_explanations(2),
                    &context,
                )
            } else {
                let mut reader = inner.storage.payload_reader()?;
                let mut bytes = 0;
                TextIndex::search_from_snapshot(
                    &inner.catalog.visible_snapshot(),
                    &collection.collection_id,
                    &query,
                    TextSearchOptions::with_explanations(2),
                    &context,
                    |visible| {
                        reads += 1;
                        read_visible_chunk_text(&mut reader, visible, &context, &mut bytes)
                    },
                )
            }
        };
        let expected = search(true).unwrap().as_slice().to_vec();
        assert_eq!(expected.len(), 2);
        assert!(expected.iter().all(|hit| hit.matched_terms.len() == 1000));
        let payload = expected[0].chunk_id.len()
            + 1000 * std::mem::size_of::<MatchedTerm>()
            + expected[0]
                .matched_terms
                .iter()
                .map(|term| term.term.len())
                .sum::<usize>();
        let limit = crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES;
        for seed in 0..100 {
            let first_admitted = seed % 2 != 0;
            let baseline_available = if first_admitted {
                163840 + seed
            } else {
                102400 + seed
            };
            let mut held = db.json_context().workspace_reservation();
            for cached in [true, false] {
                let terms = crate::json::btree_workspace::<std::borrow::Cow<'_, str>, usize>(1000);
                let arrays = 1000
                    * if cached {
                        2 * std::mem::size_of::<&str>() + std::mem::size_of::<f64>()
                    } else {
                        2 * std::mem::size_of::<usize>() + std::mem::size_of::<f64>()
                    };
                let legacy_arrays = 1000
                    * (std::mem::size_of::<std::borrow::Cow<'_, str>>()
                        + 2 * std::mem::size_of::<usize>()
                        + std::mem::size_of::<f64>());
                let available = baseline_available + terms + arrays - legacy_arrays;
                held.grow_to(
                    capacity - available,
                    capacity - available,
                    "operation_scratch",
                )
                .unwrap();
                let error = search(cached).unwrap_err();
                assert_eq!(error.code(), ErrorCode::ResourceExhausted, "{error:?}");
                let candidate_arrays = if cached {
                    2 * (std::mem::size_of::<&str>()
                        + std::mem::size_of::<f64>()
                        + std::mem::size_of::<Bm25Hit>())
                } else if first_admitted {
                    // The unused second heap slot is reclaimed before retrying admission.
                    std::mem::size_of::<Bm25Hit>()
                } else {
                    0
                };
                let requested = limit - available
                    + terms
                    + arrays
                    + candidate_arrays
                    + if first_admitted { payload } else { 0 }
                    + 1000 * std::mem::size_of::<MatchedTerm>();
                for (key, value) in [
                    ("resource", "memory".to_string()),
                    ("operation", "operation_scratch".into()),
                    ("limit_bytes", limit.to_string()),
                    ("requested_bytes", requested.to_string()),
                ] {
                    assert_eq!(error_detail_value(&error, key), Some(value.as_str()));
                }
                assert_eq!(inner.catalog.state_signature(), signature);
                assert_eq!(
                    inner.index_suffix_status,
                    IndexSuffixState::PersistentCheckpoint
                );
                held.grow_to(capacity, capacity, "operation_scratch")
                    .unwrap();
                held.retain(0);
            }
            held.retain(0);
            for cached in [true, false] {
                let hits = search(cached).unwrap();
                assert_eq!(hits.len(), expected.len());
                for (actual, expected) in hits.iter().zip(&expected) {
                    assert_eq!(actual.chunk_id, expected.chunk_id);
                    assert_eq!(actual.score.to_bits(), expected.score.to_bits());
                    assert_eq!(
                        format!("{:?}", actual.matched_terms),
                        format!("{:?}", expected.matched_terms)
                    );
                }
                drop(hits);
                held.grow_to(capacity, capacity, "operation_scratch")
                    .unwrap();
                held.retain(0);
            }
        }
        assert_eq!(reads, 750);
        assert_eq!(fs::read(&path).unwrap(), before);
        drop(inner);
        drop(db);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn borrowed_fusion_preserves_public_results_and_releases_query_scratch() {
        let path = query_preflight_fixture("borrowed-fusion", "shared ranking text");
        let before = fs::read(&path).unwrap();
        let boost = MetadataBoostConfig::parse(parse(r#"{"schema_version":"0.1","combine":"sum","max_boost":2,"rules":[{"when":{"chunk.lang":"rust"},"boost":1.5},{"when":{"chunk.absent":true},"boost":0.5}]}"#).unwrap()).unwrap();
        for persistent in [true, false] {
            let db = Database::open_read_only(&path).unwrap();
            if !persistent {
                db.lock().unwrap().index_suffix_status = IndexSuffixState::Missing;
            }
            let capacity = db.json_context().available_workspace_bytes();
            let collection = db.collection("first").unwrap();
            let mut ids = ["chunk-1", "chunk-2"]
                .map(|id| collection.get_chunk_by_external_id(id).unwrap().chunk_id);
            ids.sort();
            let signature = db.lock().unwrap().catalog.state_signature();
            for seed in 0..100 {
                let context = db.json_context();
                let has_text = seed % 3 != 0;
                let has_vector = seed % 3 != 1;
                let reject_all = seed % 7 == 0;
                let profile = [
                    ResourceProfile::Expanded,
                    ResourceProfile::Balanced,
                    ResourceProfile::Constrained,
                ][seed / 3 % 3];
                let mut query = collection
                    .search()
                    .metadata_boost(boost.clone())
                    .weights(SearchWeights::new(0.4, 0.4, 0.2).unwrap())
                    .top_k(1 + seed % 2)
                    .resource_profile(profile);
                if has_text {
                    query = query.text("shared ranking");
                }
                if has_vector {
                    query = query.query_embedding(QueryEmbeddingInput {
                        model_name: Some("m".into()),
                        dim: 2,
                        vector: vec![1.0, 0.0],
                    });
                }
                if reject_all {
                    query = query.filter(parse(r#"{"chunk.lang":"absent"}"#).unwrap());
                }
                let (result, work) = query.execute_retained(Some(context.clone())).unwrap();
                assert_eq!(
                    result.results.len(),
                    if reject_all { 0 } else { 1 + seed % 2 }
                );
                assert_eq!(
                    result.query_context.execution_stats.fusion_candidate_count,
                    if reject_all { 0 } else { 2 }
                );
                assert_eq!(
                    result.query_context.index_source,
                    if persistent {
                        "persistent_checkpoint"
                    } else {
                        "transient_rebuild"
                    }
                );
                for (rank, hit) in result.results.iter().enumerate() {
                    assert_eq!(hit.chunk_id, ids[rank]);
                    assert_eq!(hit.rank, rank + 1);
                    let expected = 1.0 + result.query_context.weights.effective.metadata * 0.5;
                    assert!((hit.final_score - expected).abs() < 1e-12);
                    assert_eq!(hit.metadata_boost, 1.5);
                    assert_eq!(hit.bm25_score.is_some(), has_text);
                    assert_eq!(hit.vector_score.is_some(), has_vector);
                    assert_eq!(
                        hit.explain
                            .get("metadata_boost_context")
                            .unwrap()
                            .get("matched_rule_count")
                            .and_then(JsonValue::as_f64),
                        Some(1.0)
                    );
                    assert_eq!(hit.snippet, "shared ranking text");
                }
                drop(result);
                drop(work);
                context
                    .workspace_reservation()
                    .grow_to(capacity, capacity, "operation_scratch")
                    .unwrap();
            }
            assert_eq!(db.lock().unwrap().catalog.state_signature(), signature);
        }
        assert_eq!(fs::read(&path).unwrap(), before);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn vector_scan_uses_bounded_scratch_not_total_vector_size() {
        let path = temp_path("streaming-vector-budget");
        let db = Database::create(&path, DatabaseConfig::default()).unwrap();
        let collection = db
            .create_collection(CollectionConfig {
                name: "notes".into(),
                embedding_dim: 256,
                distance_metric: DistanceMetric::Cosine,
                embedding_model: Some("m".into()),
                vectors_normalized: false,
                vector_index: VectorIndexConfig::Exact,
                tokenizer: Default::default(),
            })
            .unwrap();
        collection
            .insert_document(DocumentInput {
                external_id: "doc".into(),
                title: None,
                source_uri: None,
                metadata: JsonValue::object(),
                chunks: (0..2000)
                    .map(|index| ChunkInput {
                        chunk_external_id: format!("doc#{index}"),
                        text: "token".into(),
                        ordinal: index,
                        metadata: JsonValue::object(),
                        source_span: None,
                    })
                    .collect(),
            })
            .unwrap();
        collection
            .put_embeddings(
                (0..2000)
                    .map(|index| EmbeddingInput {
                        chunk_external_id: format!("doc#{index}"),
                        model_name: "m".into(),
                        dim: 256,
                        vector: vec![1.0; 256],
                    })
                    .collect(),
            )
            .unwrap();
        let expected = collection
            .search()
            .query_embedding(QueryEmbeddingInput {
                model_name: Some("m".into()),
                dim: 256,
                vector: vec![1.0; 256],
            })
            .top_k(5)
            .execute()
            .unwrap();
        let expected_ids = expected
            .results
            .iter()
            .map(|row| row.chunk_id.clone())
            .collect::<Vec<_>>();
        let expected_bytes = db
            .with_read_snapshot(|catalog, _| {
                Ok(catalog
                    .live_embeddings()
                    .map(|embedding| {
                        catalog
                            .record_ref_for_embedding(&embedding.embedding_id)
                            .unwrap()
                            .payload
                            .len
                    })
                    .sum::<u64>())
            })
            .unwrap();
        assert!(expected_bytes > 1_048_576);
        drop(collection);
        drop(db);
        let before = fs::read(&path).unwrap();
        let db = Database::open_with_options(
            &path,
            DatabaseOpenOptions {
                read_only: true,
                cache_budget_mb: Some(1),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(db.lock().unwrap().persistent_vector_segments.is_none());
        let result = db
            .collection("notes")
            .unwrap()
            .search()
            .query_embedding(QueryEmbeddingInput {
                model_name: Some("m".into()),
                dim: 256,
                vector: vec![1.0; 256],
            })
            .top_k(5)
            .execute()
            .unwrap();
        assert_eq!(
            result
                .results
                .iter()
                .map(|row| row.chunk_id.clone())
                .collect::<Vec<_>>(),
            expected_ids
        );
        assert_eq!(result.query_context.index_source, "persistent_checkpoint");
        assert!(db.lock().unwrap().lazy_vector_segment.is_none());
        assert_eq!(fs::read(&path).unwrap(), before);
        let critical_end = db.lock().unwrap().storage.critical_end_offset();
        std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(critical_end)
            .unwrap();
        let result = db
            .collection("notes")
            .unwrap()
            .search()
            .query_embedding(QueryEmbeddingInput {
                model_name: Some("m".into()),
                dim: 256,
                vector: vec![1.0; 256],
            })
            .top_k(5)
            .execute()
            .unwrap();
        assert_eq!(
            result
                .results
                .iter()
                .map(|row| row.chunk_id.clone())
                .collect::<Vec<_>>(),
            expected_ids
        );
        assert_eq!(result.query_context.index_source, "transient_rebuild");
        assert_eq!(
            result.query_context.transient_rebuild.bytes_read,
            Some(expected_bytes)
        );
        assert_eq!(
            result.query_context.transient_rebuild.chunks_scanned,
            Some(2000)
        );
        drop(db);
        let _ = fs::remove_file(path);
    }

    #[test]
    fn streamed_vector_queries_match_cached_metrics_filters_and_snapshot_fallback() {
        use crate::json::{object_from_pairs, parse};
        for metric in [
            DistanceMetric::Cosine,
            DistanceMetric::DotProduct,
            DistanceMetric::Euclidean,
        ] {
            let path = temp_path("streamed-vector-ranking");
            let db = Database::create(&path, DatabaseConfig::default()).unwrap();
            let collection = db
                .create_collection(CollectionConfig {
                    name: "notes".into(),
                    embedding_dim: 1025,
                    distance_metric: metric,
                    embedding_model: Some("m".into()),
                    vectors_normalized: false,
                    vector_index: VectorIndexConfig::Exact,
                    tokenizer: Default::default(),
                })
                .unwrap();
            collection
                .insert_document(DocumentInput {
                    external_id: "doc".into(),
                    title: None,
                    source_uri: None,
                    metadata: JsonValue::object(),
                    chunks: (0..64)
                        .map(|index| ChunkInput {
                            chunk_external_id: format!("doc#{index}"),
                            text: "needle repeated token".into(),
                            ordinal: index,
                            metadata: object_from_pairs([(
                                "group",
                                JsonValue::Number((index % 4) as f64),
                            )]),
                            source_span: None,
                        })
                        .collect(),
                })
                .unwrap();
            collection
                .put_embeddings(
                    (0..64)
                        .map(|index| EmbeddingInput {
                            chunk_external_id: format!("doc#{index}"),
                            model_name: "m".into(),
                            dim: 1025,
                            vector: (0..1025)
                                .map(|value| ((index % 7 + value % 3) as f32 - 4.0) / 7.0)
                                .collect(),
                        })
                        .collect(),
                )
                .unwrap();
            let reader = Database::open_with_options(
                &path,
                DatabaseOpenOptions {
                    read_only: true,
                    cache_budget_mb: Some(1),
                    ..Default::default()
                },
            )
            .unwrap();
            let query = |database: &Database, text: bool, filter: &str| {
                let mut builder = database
                    .collection("notes")
                    .unwrap()
                    .search()
                    .query_embedding(QueryEmbeddingInput {
                        model_name: Some("m".into()),
                        dim: 1025,
                        vector: vec![1.0; 1025],
                    })
                    .filter(parse(filter).unwrap())
                    .top_k(8);
                if text {
                    builder = builder.text("needle");
                }
                builder.execute().unwrap()
            };
            let signature = |result: &SearchResults| {
                result
                    .results
                    .iter()
                    .map(|row| {
                        (
                            row.chunk_id.clone(),
                            row.final_score.to_bits(),
                            row.vector_score.map(f64::to_bits),
                            row.normalized_scores.vector_norm.map(f64::to_bits),
                        )
                    })
                    .collect::<Vec<_>>()
            };
            for text in [false, true] {
                for (filter, count) in [
                    ("{}", 64),
                    (r#"{"chunk.group":1}"#, 16),
                    (r#"{"chunk.group":99}"#, 0),
                ] {
                    let expected = query(&db, text, filter);
                    let actual = query(&reader, text, filter);
                    assert_eq!(signature(&actual), signature(&expected));
                    assert_eq!(actual.query_context.index_source, "persistent_checkpoint");
                    assert_eq!(
                        actual.query_context.execution_stats,
                        expected.query_context.execution_stats
                    );
                    assert_eq!(actual.query_context.execution_stats.vector_scan_count, 64);
                    assert_eq!(
                        actual.query_context.execution_stats.distance_ops,
                        count * 1025
                    );
                    let inner = reader.lock().unwrap();
                    assert!(inner.lazy_vector_source.is_some());
                    assert!(inner.lazy_vector_segment.is_none());
                    assert!(inner.persistent_vector_segments.is_none());
                }
            }
            let fallback_cases =
                [r#"{"chunk.group":99}"#, r#"{"chunk.group":1}"#, "{}"].map(|filter| {
                    let reader = Database::open_with_options(
                        &path,
                        DatabaseOpenOptions {
                            read_only: true,
                            cache_budget_mb: Some(1),
                            ..Default::default()
                        },
                    )
                    .unwrap();
                    (filter, query(&db, true, filter), reader)
                });
            let critical_end = reader.lock().unwrap().storage.critical_end_offset();
            let old_len = fs::metadata(&path).unwrap().len();
            // Retain the start of the segment to exercise rejection of partial scan results.
            std::fs::OpenOptions::new()
                .write(true)
                .open(&path)
                .unwrap()
                .set_len(critical_end + (old_len - critical_end) * 3 / 4)
                .unwrap();
            for (filter, expected, reader) in fallback_cases {
                let (visited, matched) = {
                    let inner = reader.lock().unwrap();
                    let source = inner.lazy_vector_source.as_ref().unwrap();
                    let collection = inner.catalog.collection_by_name("notes").unwrap();
                    let expr = FilterExpr::parse(parse(filter).unwrap()).unwrap();
                    let context = reader.json_context();
                    let input = source
                        .reserve_read_buffer(&collection.collection_id, &context)
                        .unwrap();
                    let mut visited = 0;
                    let mut matched = 0;
                    assert!(!source
                        .visit(&inner.storage, collection, &input, |id, _| {
                            visited += 1;
                            matched += usize::from(matches_vector_filter(
                                &inner.catalog,
                                &collection.collection_id,
                                expr.as_ref(),
                                id,
                            ));
                            Ok(())
                        })
                        .unwrap());
                    assert!(visited > 0 && visited < 64);
                    assert_eq!(matched > 0, !expected.results.is_empty());
                    (visited, matched)
                };
                let fallback = query(&reader, true, filter);
                assert_eq!(signature(&fallback), signature(&expected));
                let mut expected_work = expected.query_context.execution_stats;
                expected_work.vector_scan_count += visited;
                expected_work.distance_ops += matched * 1025;
                assert_eq!(fallback.query_context.execution_stats, expected_work);
                assert_eq!(fallback.query_context.index_source, "transient_rebuild");
                assert!(reader.lock().unwrap().lazy_vector_source.is_none());
            }
            drop(reader);
            drop(collection);
            drop(db);
            fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn transient_rebuild_over_cache_budget_returns_index_corruption() {
        let path = temp_path("transient-budget");
        let db = Database::create(&path, DatabaseConfig::default()).unwrap();
        let col = db
            .create_collection(CollectionConfig {
                tokenizer: Default::default(),
                name: "notes".into(),
                embedding_dim: 2,
                distance_metric: DistanceMetric::Cosine,
                embedding_model: Some("m".into()),
                vectors_normalized: false,
                vector_index: VectorIndexConfig::Exact,
            })
            .unwrap();
        col.insert_document(DocumentInput {
            external_id: "doc".into(),
            title: None,
            source_uri: None,
            metadata: JsonValue::object(),
            chunks: vec![ChunkInput {
                chunk_external_id: "doc#1".into(),
                text: "budgeted transient rebuild must fail closed".into(),
                ordinal: 0,
                metadata: JsonValue::object(),
                source_span: None,
            }],
        })
        .unwrap();
        col.put_embeddings(vec![EmbeddingInput {
            chunk_external_id: "doc#1".into(),
            model_name: "m".into(),
            dim: 2,
            vector: vec![1.0, 0.0],
        }])
        .unwrap();
        drop(col);
        drop(db);

        let before = Database::check_path(
            &path,
            CheckOptions {
                verify_indexes: true,
            },
        )
        .unwrap();
        std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(before.critical_end_offset)
            .unwrap();

        let reopened = Database::open_with_options(
            &path,
            DatabaseOpenOptions {
                cache_budget_mb: Some(0),
                ..DatabaseOpenOptions::default()
            },
        )
        .unwrap();
        let err = reopened
            .collection("notes")
            .unwrap()
            .search()
            .text("transient rebuild")
            .top_k(5)
            .execute()
            .unwrap_err();
        assert_eq!(err.code(), ErrorCode::IndexCorruption);
        assert_eq!(
            err.detail()
                .iter()
                .find(|(key, _)| key == "index_unavailable_reason")
                .map(|(_, value)| value.as_str()),
            Some("transient_rebuild_budget_exceeded")
        );
        assert_eq!(
            err.detail()
                .iter()
                .find(|(key, _)| key == "index_source_attempted")
                .map(|(_, value)| value.as_str()),
            Some("transient_rebuild")
        );
        assert_eq!(
            err.detail()
                .iter()
                .find(|(key, _)| key == "resource_profile")
                .map(|(_, value)| value.as_str()),
            Some("balanced")
        );
        assert_eq!(
            err.detail()
                .iter()
                .find(|(key, _)| key == "budget_mb")
                .map(|(_, value)| value.as_str()),
            Some("0")
        );
        let err_json = err.to_json();
        assert_eq!(
            err_json.get("recoverable").and_then(JsonValue::as_bool),
            Some(true)
        );
        assert_eq!(
            err_json.get("suggested_action").and_then(JsonValue::as_str),
            Some("run coffer rebuild-index for the affected collection")
        );
        for profile in [ResourceProfile::Expanded, ResourceProfile::Constrained] {
            let err = reopened
                .collection("notes")
                .unwrap()
                .search()
                .text("transient rebuild")
                .resource_profile(profile)
                .execute()
                .unwrap_err();
            assert_eq!(err.code(), ErrorCode::IndexCorruption);
            assert!(err
                .detail()
                .iter()
                .any(|(key, value)| key == "budget_mb" && value == "0"));
        }
        drop(reopened);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn opened_vectors_load_one_collection_and_fall_back_after_suffix_replacement() {
        let path = temp_path("lazy-vector-snapshot");
        let db = Database::create(&path, DatabaseConfig::default()).unwrap();
        for name in ["alpha", "beta"] {
            let collection = db
                .create_collection(CollectionConfig {
                    tokenizer: Default::default(),
                    name: name.into(),
                    embedding_dim: 2,
                    distance_metric: DistanceMetric::Cosine,
                    embedding_model: Some("m".into()),
                    vectors_normalized: false,
                    vector_index: VectorIndexConfig::Exact,
                })
                .unwrap();
            collection
                .insert_document(DocumentInput {
                    external_id: format!("old-{name}"),
                    title: None,
                    source_uri: None,
                    metadata: JsonValue::object(),
                    chunks: vec![ChunkInput {
                        chunk_external_id: format!("old-{name}#0"),
                        text: "needle snapshot".into(),
                        ordinal: 0,
                        metadata: JsonValue::object(),
                        source_span: None,
                    }],
                })
                .unwrap();
            collection
                .put_embeddings(vec![EmbeddingInput {
                    chunk_external_id: format!("old-{name}#0"),
                    model_name: "m".into(),
                    dim: 2,
                    vector: vec![1.0, 0.0],
                }])
                .unwrap();
        }
        let query = |database: &Database, name: &str| {
            database
                .collection(name)
                .unwrap()
                .search()
                .text("needle")
                .query_embedding(QueryEmbeddingInput {
                    model_name: Some("m".into()),
                    dim: 2,
                    vector: vec![1.0, 0.0],
                })
                .execute()
                .unwrap()
        };
        let vector_query = |database: &Database, name: &str| {
            database
                .collection(name)
                .unwrap()
                .search()
                .query_embedding(QueryEmbeddingInput {
                    model_name: Some("m".into()),
                    dim: 2,
                    vector: vec![1.0, 0.0],
                })
                .execute()
                .unwrap()
        };
        let signature = |results: &SearchResults| {
            results
                .results
                .iter()
                .map(|hit| {
                    (
                        hit.chunk_id.clone(),
                        hit.chunk_external_id.clone(),
                        hit.final_score.to_bits(),
                        hit.bm25_score.map(f64::to_bits),
                        hit.vector_score.map(f64::to_bits),
                    )
                })
                .collect::<Vec<_>>()
        };
        let reader = Database::open_read_only(&path).unwrap();
        let alpha_id = reader
            .collection("alpha")
            .unwrap()
            .collection_id()
            .to_owned();
        let beta_id = reader
            .collection("beta")
            .unwrap()
            .collection_id()
            .to_owned();
        let pointer = || {
            let inner = reader.lock().unwrap();
            let segment = inner.lazy_vector_segment.as_ref().unwrap();
            (
                segment.collection_id.clone(),
                segment.vectors.as_ptr() as usize,
            )
        };
        {
            let inner = reader.lock().unwrap();
            assert!(inner.lazy_vector_source.is_some());
            assert!(inner.lazy_vector_segment.is_none());
            assert!(inner.persistent_vector_segments.is_none());
        }
        reader
            .collection("alpha")
            .unwrap()
            .search()
            .text("needle")
            .execute()
            .unwrap();
        assert!(reader.lock().unwrap().lazy_vector_segment.is_none());
        let alpha = query(&reader, "alpha");
        assert_eq!(alpha.results[0].chunk_external_id, "old-alpha#0");
        let loaded = pointer();
        assert_eq!(loaded.0, alpha_id);
        query(&reader, "alpha");
        assert_eq!(pointer(), loaded);
        let before_beta = query(&reader, "beta");
        assert_eq!(pointer().0, beta_id);
        {
            let mut inner = reader.lock().unwrap();
            inner.prepare_vector_index(&beta_id, 0).unwrap();
            assert!(inner.lazy_vector_segment.is_none());
            assert!(inner.lazy_vector_source.is_some());
            assert_eq!(
                inner.index_suffix_status,
                IndexSuffixState::PersistentCheckpoint
            );
        }
        let before_update = query(&reader, "alpha");
        let vector_before = vector_query(&reader, "alpha");
        let retained_pointer = pointer();
        let before_stats = reader.stats().unwrap();
        db.collection("beta")
            .unwrap()
            .insert_document(DocumentInput {
                external_id: "new-beta".into(),
                title: None,
                source_uri: None,
                metadata: JsonValue::object(),
                chunks: vec![ChunkInput {
                    chunk_external_id: "new-beta#0".into(),
                    text: "needle newer snapshot".into(),
                    ordinal: 0,
                    metadata: JsonValue::object(),
                    source_span: None,
                }],
            })
            .unwrap();
        let cached = vector_query(&reader, "alpha");
        assert_eq!(signature(&cached), signature(&vector_before));
        assert_eq!(pointer(), retained_pointer);
        assert_eq!(
            cached
                .query_context
                .to_json()
                .get("index_source")
                .and_then(JsonValue::as_str),
            Some("persistent_checkpoint")
        );
        let fallback_alpha = query(&reader, "alpha");
        assert_eq!(signature(&fallback_alpha), signature(&before_update));
        assert_eq!(
            fallback_alpha.query_context.index_source,
            "persistent_checkpoint"
        );
        assert!(!fallback_alpha.query_context.transient_rebuild.started);
        reader.lock().unwrap().text_index.clear_cache();
        let fallback_alpha = query(&reader, "alpha");
        assert_eq!(signature(&fallback_alpha), signature(&before_update));
        assert_eq!(
            fallback_alpha.query_context.index_source,
            "transient_rebuild"
        );
        assert!(fallback_alpha.query_context.transient_rebuild.started);
        let after_stats = reader.stats().unwrap();
        assert_eq!(after_stats.document_count, before_stats.document_count);
        assert_eq!(after_stats.chunk_count, before_stats.chunk_count);
        assert_eq!(after_stats.embedding_count, before_stats.embedding_count);
        assert_eq!(after_stats.last_commit_id, before_stats.last_commit_id);
        let old_beta = query(&reader, "beta");
        assert_eq!(signature(&old_beta), signature(&before_beta));
        assert_eq!(old_beta.results.len(), 1);
        assert_eq!(old_beta.results[0].chunk_external_id, "old-beta#0");
        assert_eq!(
            old_beta
                .query_context
                .to_json()
                .get("index_source")
                .and_then(JsonValue::as_str),
            Some("transient_rebuild")
        );
        assert!(reader.lock().unwrap().lazy_vector_source.is_none());
        assert!(reader.lock().unwrap().lazy_vector_segment.is_none());
        assert_eq!(reader.stats().unwrap().chunk_count, 2);
        let current = Database::open_read_only(&path).unwrap();
        assert_eq!(current.stats().unwrap().chunk_count, 3);
        assert_eq!(query(&current, "beta").results.len(), 2);
        drop(current);
        drop(reader);
        drop(db);
        let writer = Database::open(&path).unwrap();
        query(&writer, "alpha");
        let retained_alpha = writer
            .lock()
            .unwrap()
            .lazy_vector_segment
            .as_ref()
            .unwrap()
            .vectors
            .as_ptr() as usize;
        writer
            .collection("beta")
            .unwrap()
            .put_embeddings(vec![EmbeddingInput {
                chunk_external_id: "new-beta#0".into(),
                model_name: "m".into(),
                dim: 2,
                vector: vec![0.0, 1.0],
            }])
            .unwrap();
        {
            let inner = writer.lock().unwrap();
            assert!(inner.lazy_vector_source.is_some());
            assert!(inner.lazy_vector_segment.is_none());
            let segments = inner.persistent_vector_segments.as_ref().unwrap();
            assert_eq!(segments.len(), 2);
            assert_eq!(
                segments[&alpha_id].vectors.as_ptr() as usize,
                retained_alpha
            );
            assert_eq!(segments[&beta_id].vectors.len(), 2);
        }
        assert_eq!(
            query(&writer, "alpha").results[0].final_score.to_bits(),
            alpha.results[0].final_score.to_bits()
        );
        assert_eq!(
            writer
                .check(CheckOptions {
                    verify_indexes: true
                })
                .unwrap()
                .index_check,
            "ok"
        );
        drop(writer);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn vector_cache_numeric_order_survives_collection_local_reuse() {
        let query = |db: &Database| {
            let result = db
                .collection("untouched")
                .unwrap()
                .search()
                .query_embedding(QueryEmbeddingInput {
                    model_name: Some("m".into()),
                    dim: 2,
                    vector: vec![1.0, 0.0],
                })
                .top_k(2)
                .execute()
                .unwrap();
            assert_eq!(
                result
                    .results
                    .iter()
                    .map(|hit| (hit.chunk_external_id.as_str(), hit.vector_score))
                    .collect::<Vec<_>>(),
                vec![("B", Some(2.0)), ("A", Some(1.0))]
            );
            assert_eq!(
                result
                    .query_context
                    .to_json()
                    .get("index_source")
                    .and_then(JsonValue::as_str),
                Some("persistent_checkpoint")
            );
            result
                .results
                .iter()
                .map(|hit| hit.final_score.to_bits())
                .collect::<Vec<_>>()
        };
        let embedding = |id: &str, value| EmbeddingInput {
            chunk_external_id: id.into(),
            model_name: "m".into(),
            dim: 2,
            vector: vec![value, 0.0],
        };
        for disk_lazy in [false, true] {
            let path = temp_path("vector-canonical-cache-reuse");
            let mut db = Database::create(
                &path,
                DatabaseConfig {
                    cache_budget_mb: Some(1),
                    durability: Durability::Fast,
                    ..Default::default()
                },
            )
            .unwrap();
            for name in ["untouched", "changed"] {
                let collection = db
                    .create_collection(CollectionConfig {
                        tokenizer: Default::default(),
                        name: name.into(),
                        embedding_dim: 2,
                        distance_metric: DistanceMetric::DotProduct,
                        embedding_model: Some("m".into()),
                        vectors_normalized: false,
                        vector_index: VectorIndexConfig::Exact,
                    })
                    .unwrap();
                for id in if name == "untouched" {
                    &["A", "B"][..]
                } else {
                    &["C"][..]
                } {
                    collection
                        .insert_document(DocumentInput {
                            external_id: (*id).into(),
                            title: None,
                            source_uri: None,
                            metadata: JsonValue::object(),
                            chunks: vec![ChunkInput {
                                chunk_external_id: (*id).into(),
                                text: "cache order".into(),
                                ordinal: 0,
                                metadata: JsonValue::object(),
                                source_span: None,
                            }],
                        })
                        .unwrap();
                }
            }
            let untouched = db.collection("untouched").unwrap();
            let collection_id = untouched.collection_id().to_owned();
            let ids = ["A", "B"].map(|id| untouched.get_chunk_by_external_id(id).unwrap().chunk_id);
            assert!(id_number(&ids[0], "chk_") < id_number(&ids[1], "chk_"));
            untouched
                .put_embeddings(vec![embedding("B", 2.0), embedding("A", 1.0)])
                .unwrap();
            drop(untouched);
            let observe = |inner: &Inner| {
                assert_eq!(
                    inner.index_suffix_status,
                    IndexSuffixState::PersistentCheckpoint
                );
                let segment = match &inner.persistent_vector_segments {
                    Some(cache) => &cache[&collection_id],
                    None => inner.lazy_vector_segment.as_ref().unwrap(),
                };
                assert_eq!(segment.collection_id, collection_id);
                assert_eq!(segment.vectors.len(), 2);
                for (index, entry) in segment.vectors.iter().enumerate() {
                    assert_eq!(entry.chunk_id, ids[index]);
                    assert_eq!(entry.vector, vec![index as f32 + 1.0, 0.0]);
                }
                (
                    segment.vectors.as_ptr() as usize,
                    [
                        segment.vectors[0].vector.as_ptr() as usize,
                        segment.vectors[1].vector.as_ptr() as usize,
                    ],
                    segment.covered_commit_sequence_no,
                )
            };
            // Check the newly materialized cache before any query or reopen can replace it.
            observe(&db.lock().unwrap());
            if disk_lazy {
                drop(db);
                db = Database::open_with_options(
                    &path,
                    DatabaseOpenOptions {
                        cache_budget_mb: Some(1),
                        durability: Some(Durability::Fast),
                        ..Default::default()
                    },
                )
                .unwrap();
                let inner = db.lock().unwrap();
                assert!(inner.lazy_vector_source.is_some());
                assert!(inner.lazy_vector_segment.is_none());
                assert!(inner.persistent_vector_segments.is_none());
            }
            let before = query(&db);
            let pointers = observe(&db.lock().unwrap());
            db.collection("changed")
                .unwrap()
                .put_embeddings(vec![embedding("C", 3.0)])
                .unwrap();
            {
                let inner = db.lock().unwrap();
                let after = observe(&inner);
                assert!(inner.persistent_vector_segments.is_some());
                assert_eq!((after.0, after.1), (pointers.0, pointers.1));
                assert_eq!(after.2, inner.last_commit_sequence_no);
                assert!(after.2 > pointers.2);
            }
            assert_eq!(query(&db), before);
            assert_eq!(
                db.check(CheckOptions {
                    verify_indexes: true
                })
                .unwrap()
                .index_check,
                "ok"
            );
            drop(db);
            let reopened = Database::open_read_only(&path).unwrap();
            assert_eq!(query(&reopened), before);
            observe(&reopened.lock().unwrap());
            drop(reopened);
            fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn written_vector_cache_is_bounded_and_uncached_collections_remain_queryable() {
        let path = temp_path("written-vector-cache-budget");
        let db = Database::create(
            &path,
            DatabaseConfig {
                cache_budget_mb: Some(1),
                ..Default::default()
            },
        )
        .unwrap();
        let dim = 12288;
        let add = |collection: &CollectionHandle, name: &str, count: usize| {
            collection
                .insert_document(DocumentInput {
                    external_id: name.into(),
                    title: None,
                    source_uri: None,
                    metadata: JsonValue::object(),
                    chunks: (0..count)
                        .map(|index| ChunkInput {
                            chunk_external_id: format!("{name}#{index}"),
                            text: "needle".into(),
                            ordinal: index,
                            metadata: JsonValue::object(),
                            source_span: None,
                        })
                        .collect(),
                })
                .unwrap();
            collection
                .put_embeddings(
                    (0..count)
                        .map(|index| EmbeddingInput {
                            chunk_external_id: format!("{name}#{index}"),
                            model_name: "m".into(),
                            dim,
                            vector: vec![1.0; dim],
                        })
                        .collect(),
                )
                .unwrap();
        };
        for name in ["alpha", "beta", "gamma"] {
            let collection = db
                .create_collection(CollectionConfig {
                    name: name.into(),
                    embedding_dim: dim,
                    distance_metric: DistanceMetric::Cosine,
                    embedding_model: Some("m".into()),
                    vectors_normalized: false,
                    vector_index: VectorIndexConfig::Exact,
                    tokenizer: Default::default(),
                })
                .unwrap();
            add(&collection, name, 1);
        }
        {
            let mut inner = db.lock().unwrap();
            let original_cache = inner.persistent_vector_segments.take();
            let source = inner.lazy_vector_source.clone().unwrap();
            let ids = inner
                .catalog
                .collections
                .keys()
                .cloned()
                .collect::<Vec<_>>();
            for seed in 0..100usize {
                let budget_mb = seed % 3;
                inner.config.cache_budget_mb = Some(budget_mb);
                let mut owned = VectorCache::new(&db.json_context());
                inner.lazy_vector_segment = None;
                let mut occupied = 0u64;
                for (index, id) in ids.iter().enumerate() {
                    if seed & (1 << index) != 0 {
                        let context = db.json_context();
                        let reservation = CachedVectorSegment::reserve(
                            &context,
                            usize::MAX,
                            source.encoded_len(id).unwrap(),
                        )
                        .unwrap();
                        let segment = source
                            .load_cached(&inner.storage, id, reservation, &context)
                            .unwrap()
                            .unwrap();
                        occupied += source.encoded_len(id).unwrap();
                        if seed % 2 == 0 && inner.lazy_vector_segment.is_none() {
                            inner.lazy_vector_segment = Some(segment);
                        } else {
                            assert!(owned.insert(segment, usize::MAX));
                        }
                    }
                }
                inner.persistent_vector_segments = Some(owned);
                let changed = ids
                    .iter()
                    .enumerate()
                    .filter(|(index, _)| seed & (1 << (index + 3)) != 0)
                    .map(|(_, id)| id.clone())
                    .collect::<BTreeSet<_>>();
                let prefetched = inner.prefetch_unchanged_vector_segments(&changed);
                let budget = budget_mb as u64 * 1_048_576 / 8;
                let mut remaining = budget.saturating_sub(occupied);
                for id in &ids {
                    let cached = inner
                        .persistent_vector_segments
                        .as_ref()
                        .unwrap()
                        .contains_key(id)
                        || inner
                            .lazy_vector_segment
                            .as_ref()
                            .is_some_and(|segment| &segment.collection_id == id);
                    let bytes = source.encoded_len(id).unwrap();
                    let expected = !changed.contains(id) && !cached && bytes <= remaining;
                    assert_eq!(prefetched.contains_key(id), expected, "seed {seed}, {id}");
                    if expected {
                        remaining -= bytes;
                        assert_eq!(prefetched[id].vectors.len(), 1);
                    }
                }
            }
            inner.config.cache_budget_mb = Some(1);
            inner.persistent_vector_segments = None;
            inner.lazy_vector_segment = None;
            let changed = BTreeSet::from([ids[0].clone()]);
            let prefetched = inner.prefetch_unchanged_vector_segments(&changed);
            assert_eq!(prefetched.len(), 2);
            inner
                .refresh_index_checkpoint_reusing(prefetched, &changed, &db.json_context())
                .unwrap();
            let retained = inner.persistent_vector_segments.as_ref().unwrap();
            assert_eq!(retained.len(), 2);
            assert!(
                !retained.contains_key(&ids[0]),
                "unwritten reused segments already consume the available headroom"
            );
            inner.persistent_vector_segments = original_cache;
            inner.lazy_vector_segment = None;
        }
        let query = |database: &Database, name: &str| {
            database
                .collection(name)
                .unwrap()
                .search()
                .query_embedding(QueryEmbeddingInput {
                    model_name: Some("m".into()),
                    dim,
                    vector: vec![1.0; dim],
                })
                .execute()
                .unwrap()
        };
        let cached_bytes = |database: &Database| {
            let inner = database.lock().unwrap();
            let source = inner.lazy_vector_source.as_ref().unwrap();
            let mut bytes = inner
                .persistent_vector_segments
                .as_ref()
                .map_or(0, |segments| {
                    segments
                        .keys()
                        .map(|id| source.encoded_len(id).unwrap())
                        .sum::<u64>()
                });
            if let Some(segment) = &inner.lazy_vector_segment {
                bytes += source.encoded_len(&segment.collection_id).unwrap();
            }
            assert!(bytes <= 1_048_576 / 8);
            bytes
        };
        let before = cached_bytes(&db);
        assert!(before > 80_000);
        assert_eq!(
            db.lock()
                .unwrap()
                .persistent_vector_segments
                .as_ref()
                .unwrap()
                .len(),
            2
        );
        let gamma = query(&db, "gamma");
        assert_eq!(gamma.results.len(), 1);
        assert_eq!(gamma.query_context.index_source, "persistent_checkpoint");
        assert!(db.lock().unwrap().persistent_vector_segments.is_none());
        assert!(cached_bytes(&db) < before);
        for name in ["alpha", "beta", "gamma"] {
            assert_eq!(query(&db, name).results.len(), 1);
            cached_bytes(&db);
        }
        let alpha_id = db.collection("alpha").unwrap().collection_id().to_owned();
        db.lock()
            .unwrap()
            .prepare_vector_index(&alpha_id, 0)
            .unwrap();
        assert_eq!(cached_bytes(&db), 0);
        add(&db.collection("gamma").unwrap(), "more", 3);
        let before = cached_bytes(&db);
        assert!(before > 80_000);
        assert_eq!(query(&db, "gamma").results.len(), 4);
        assert_eq!(cached_bytes(&db), before);
        assert!(db.lock().unwrap().lazy_vector_segment.is_none());
        let reader = Database::open_with_options(
            &path,
            DatabaseOpenOptions {
                read_only: true,
                cache_budget_mb: Some(1),
                ..Default::default()
            },
        )
        .unwrap();
        db.compact(CompactOptions {
            rebuild_indexes: true,
        })
        .unwrap();
        cached_bytes(&db);
        assert_eq!(query(&db, "gamma").results.len(), 4);
        assert_eq!(query(&reader, "gamma").results.len(), 4);
        assert_eq!(
            db.check(CheckOptions {
                verify_indexes: true
            })
            .unwrap()
            .index_check,
            "ok"
        );
        drop(reader);
        drop(db);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn checkpoint_build_failure_after_written_segments_keeps_primary_snapshot() {
        let path = temp_path("checkpoint-late-producer-failure");
        let db = Database::create(&path, DatabaseConfig::default()).unwrap();
        for name in ["first", "last"] {
            let collection = db
                .create_collection(CollectionConfig {
                    name: name.into(),
                    embedding_dim: 2,
                    distance_metric: DistanceMetric::Cosine,
                    embedding_model: Some("m".into()),
                    vectors_normalized: false,
                    vector_index: VectorIndexConfig::Exact,
                    tokenizer: Default::default(),
                })
                .unwrap();
            collection
                .insert_document(DocumentInput {
                    external_id: name.into(),
                    title: None,
                    source_uri: None,
                    metadata: JsonValue::object(),
                    chunks: vec![ChunkInput {
                        chunk_external_id: name.into(),
                        text: "needle".into(),
                        ordinal: 0,
                        metadata: JsonValue::object(),
                        source_span: None,
                    }],
                })
                .unwrap();
            collection
                .put_embeddings(vec![EmbeddingInput {
                    chunk_external_id: name.into(),
                    model_name: "m".into(),
                    dim: 2,
                    vector: vec![1.0, 0.0],
                }])
                .unwrap();
        }
        let reader = Database::open_with_options(
            &path,
            DatabaseOpenOptions {
                read_only: true,
                ..Default::default()
            },
        )
        .unwrap();
        let prefix_len = db.lock().unwrap().storage.critical_end_offset() as usize;
        let prefix = fs::read(&path).unwrap()[..prefix_len].to_vec();
        {
            let mut inner = db.lock().unwrap();
            let id = inner
                .catalog
                .live_embeddings()
                .last()
                .unwrap()
                .embedding_id
                .clone();
            let wrong_ref =
                inner.catalog.record_refs[inner.catalog.collections.keys().next().unwrap()].clone();
            let original = inner
                .catalog
                .record_refs
                .insert(id.clone(), wrong_ref)
                .unwrap();
            assert_eq!(
                inner
                    .refresh_index_checkpoint(&db.json_context())
                    .unwrap_err()
                    .code(),
                ErrorCode::FileFormatError
            );
            inner.catalog.record_refs.insert(id, original);
            assert_eq!(inner.index_suffix_status, IndexSuffixState::RebuildNeeded);
            assert!(inner.persistent_vector_segments.is_none());
            assert!(inner.lazy_vector_source.is_none());
            assert!(inner.lazy_vector_segment.is_none());
        }
        let partial = fs::read(&path).unwrap();
        assert_eq!(&partial[..prefix_len], &prefix);
        assert!(partial.len() > prefix_len);
        let scan = scan_file(&path).unwrap();
        assert!(scan.index_checkpoint.is_none());
        assert!(scan
            .findings
            .iter()
            .any(|finding| finding.stream == "index"));
        let fresh = Database::open_with_options(
            &path,
            DatabaseOpenOptions {
                read_only: true,
                ..Default::default()
            },
        )
        .unwrap();
        for database in [&db, &reader, &fresh] {
            for name in ["first", "last"] {
                let result = database
                    .collection(name)
                    .unwrap()
                    .search()
                    .query_embedding(QueryEmbeddingInput {
                        model_name: Some("m".into()),
                        dim: 2,
                        vector: vec![1.0, 0.0],
                    })
                    .execute()
                    .unwrap();
                assert_eq!(result.results.len(), 1);
                assert_eq!(result.results[0].chunk_external_id, name);
            }
        }
        db.rebuild_index(None).unwrap();
        assert_eq!(
            db.check(CheckOptions {
                verify_indexes: true
            })
            .unwrap()
            .index_check,
            "ok"
        );
        assert_eq!(&fs::read(&path).unwrap()[..prefix_len], &prefix);
        drop((fresh, reader, db));
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn opened_reader_keeps_checkpoint_snapshot_after_suffix_truncate_and_new_commit() {
        let path = temp_path("reader-owned-checkpoint");
        let db = Database::create(&path, DatabaseConfig::default()).unwrap();
        let col = db
            .create_collection(CollectionConfig {
                tokenizer: Default::default(),
                name: "notes".into(),
                embedding_dim: 2,
                distance_metric: DistanceMetric::Cosine,
                embedding_model: Some("m".into()),
                vectors_normalized: false,
                vector_index: VectorIndexConfig::Exact,
            })
            .unwrap();
        let collection_id = col.collection_id().to_owned();
        col.insert_document(DocumentInput {
            external_id: "old-doc".into(),
            title: Some("old".into()),
            source_uri: None,
            metadata: JsonValue::object(),
            chunks: vec![ChunkInput {
                chunk_external_id: "old-doc#1".into(),
                text: "alpha snapshot phrase".into(),
                ordinal: 0,
                metadata: JsonValue::object(),
                source_span: None,
            }],
        })
        .unwrap();
        col.put_embeddings(vec![EmbeddingInput {
            chunk_external_id: "old-doc#1".into(),
            model_name: "m".into(),
            dim: 2,
            vector: vec![1.0, 0.0],
        }])
        .unwrap();

        let old_results = col
            .search()
            .text("alpha snapshot")
            .query_embedding(QueryEmbeddingInput {
                model_name: Some("m".into()),
                dim: 2,
                vector: vec![1.0, 0.0],
            })
            .execute()
            .unwrap();
        assert_eq!(old_results.results.len(), 1);
        assert_eq!(old_results.results[0].chunk_external_id, "old-doc#1");
        assert_eq!(
            old_results
                .query_context
                .to_json()
                .get("index_source")
                .and_then(JsonValue::as_str),
            Some("persistent_checkpoint")
        );

        let critical_end_before = {
            let mut inner = db.lock().unwrap();
            assert_eq!(
                inner.index_suffix_status,
                IndexSuffixState::PersistentCheckpoint
            );
            let critical_end_before = inner.storage.critical_end_offset();
            assert!(inner.storage.file_size().unwrap() > critical_end_before);
            let now = time::now_rfc3339();
            let text = "bravo writer phrase".to_owned();
            inner
                .storage
                .append_batch(
                    &[
                        RecordPayload::DocumentPut(Document {
                            document_id: "doc_9999999999999999".into(),
                            collection_id: collection_id.clone(),
                            external_id: "new-doc".into(),
                            title: Some("new".into()),
                            source_uri: None,
                            metadata: JsonValue::object(),
                            created_at: now.clone(),
                            updated_at: now.clone(),
                        }),
                        RecordPayload::ChunkPut(Chunk {
                            chunk_id: "chk_9999999999999999".into(),
                            document_id: "doc_9999999999999999".into(),
                            collection_id: collection_id.clone(),
                            chunk_external_id: "new-doc#1".into(),
                            text_hash: chunk_text_hash(&text),
                            token_count: token_count_hint(&text),
                            text,
                            ordinal: 0,
                            metadata: JsonValue::object(),
                            embedding_id: None,
                            source_span: None,
                            created_at: now,
                        }),
                    ],
                    Durability::Commit,
                )
                .unwrap();
            critical_end_before
        };

        let file_report = Database::check_path(
            &path,
            CheckOptions {
                verify_indexes: true,
            },
        )
        .unwrap();
        assert!(file_report.critical_end_offset > critical_end_before);
        assert_eq!(file_report.live_documents, 2);
        assert_eq!(file_report.index_suffix_status, "missing");

        let old_snapshot_results = col.search().text("alpha snapshot").execute().unwrap();
        assert_eq!(old_snapshot_results.results.len(), 1);
        assert_eq!(
            old_snapshot_results.results[0].chunk_external_id,
            "old-doc#1"
        );
        assert_eq!(
            old_snapshot_results.query_context.index_source,
            "persistent_checkpoint"
        );

        let new_snapshot_results = col.search().text("bravo writer").execute().unwrap();
        assert!(new_snapshot_results.results.is_empty());
        assert_eq!(
            new_snapshot_results.query_context.index_source,
            "persistent_checkpoint"
        );

        drop(col);
        drop(db);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn catalog_visible_snapshot_preserves_payload_locators() {
        let path = temp_path("catalog-locators");
        let db = Database::create(&path, DatabaseConfig::default()).unwrap();
        let col = db
            .create_collection(CollectionConfig {
                tokenizer: Default::default(),
                name: "notes".into(),
                embedding_dim: 2,
                distance_metric: DistanceMetric::Cosine,
                embedding_model: None,
                vectors_normalized: false,
                vector_index: VectorIndexConfig::Exact,
            })
            .unwrap();
        col.insert_document(DocumentInput {
            external_id: "doc".into(),
            title: None,
            source_uri: None,
            metadata: parse("{}").unwrap(),
            chunks: vec![ChunkInput {
                chunk_external_id: "doc#1".into(),
                text: "wal locator search".into(),
                ordinal: 1,
                metadata: parse("{}").unwrap(),
                source_span: None,
            }],
        })
        .unwrap();
        col.put_embeddings(vec![EmbeddingInput {
            chunk_external_id: "doc#1".into(),
            model_name: "m".into(),
            dim: 2,
            vector: vec![1.0, 0.0],
        }])
        .unwrap();
        let chunk = col.get_chunk_by_external_id("doc#1").unwrap();
        let embedding_id = chunk.embedding_id.clone().unwrap();

        let inner = db.lock().unwrap();
        let chunk_ref = inner.catalog.record_ref_for_chunk(&chunk.chunk_id).unwrap();
        assert_eq!(chunk_ref.record_type, "ChunkPut");
        assert!(chunk_ref.payload.len > 0);
        let embedding_ref = inner
            .catalog
            .record_ref_for_embedding(&embedding_id)
            .unwrap();
        assert_eq!(embedding_ref.record_type, "EmbeddingPut");
        assert!(inner
            .catalog
            .visible_snapshot()
            .live_chunks()
            .any(
                |visible| visible.chunk.chunk_id == chunk.chunk_id && visible.record_ref.is_some()
            ));
        assert!(inner
            .catalog
            .visible_snapshot()
            .live_embeddings()
            .any(|visible| visible.embedding.embedding_id == embedding_id
                && visible.record_ref.is_some()));
        drop(inner);

        let scan = scan_file(&path).unwrap();
        let catalog =
            Catalog::from_committed_records(scan.records.clone(), scan.last_valid_commit).unwrap();
        assert!(catalog.record_ref_for_chunk(&chunk.chunk_id).is_some());
        assert!(catalog.record_ref_for_embedding(&embedding_id).is_some());
        drop(col);
        drop(db);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn open_and_check_keep_main_data_visible_with_index_suffix_bytes() {
        let path = temp_path("index-suffix-visible");
        let db = Database::create(&path, DatabaseConfig::default()).unwrap();
        let col = db
            .create_collection(CollectionConfig {
                tokenizer: Default::default(),
                name: "notes".into(),
                embedding_dim: 2,
                distance_metric: DistanceMetric::Cosine,
                embedding_model: None,
                vectors_normalized: false,
                vector_index: VectorIndexConfig::Exact,
            })
            .unwrap();
        col.insert_document(DocumentInput {
            external_id: "doc".into(),
            title: None,
            source_uri: None,
            metadata: parse("{}").unwrap(),
            chunks: vec![ChunkInput {
                chunk_external_id: "doc#1".into(),
                text: "suffix bytes do not hide committed data".into(),
                ordinal: 1,
                metadata: parse("{}").unwrap(),
                source_span: None,
            }],
        })
        .unwrap();
        drop(col);
        drop(db);
        append_fake_index_suffix(&path);

        let db = Database::open(&path).unwrap();
        assert_eq!(db.list_collections().unwrap().len(), 1);
        let col = db.collection("notes").unwrap();
        let chunks = col.list_chunks_by_document_external_id("doc").unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].chunk_external_id, "doc#1");
        drop(col);
        drop(db);

        let report = Database::check_path(
            &path,
            CheckOptions {
                verify_indexes: true,
            },
        )
        .unwrap();
        assert_eq!(report.status, "warning");
        assert_eq!(report.index_suffix_status, "rebuild_needed");
        assert!(report.issues.iter().any(|issue| {
            (issue.code == "index_suffix_unreadable" || issue.code == "index_suffix_trailing_bytes")
                && issue.stream == "index"
                && issue.visibility_impact == "rebuild_index_only"
        }));
        assert!(!report
            .issues
            .iter()
            .any(|issue| issue.visibility_impact == "blocks_main"));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn corrupt_metadata_checkpoint_is_rebuild_index_only() {
        let path = temp_path("corrupt-index-checkpoint");
        let db = Database::create(&path, DatabaseConfig::default()).unwrap();
        db.create_collection(CollectionConfig {
            tokenizer: Default::default(),
            name: "notes".into(),
            embedding_dim: 2,
            distance_metric: DistanceMetric::Cosine,
            embedding_model: None,
            vectors_normalized: false,
            vector_index: VectorIndexConfig::Exact,
        })
        .unwrap();
        drop(db);

        let scan = scan_file(&path).unwrap();
        let suffix_offset = scan.index_suffix.as_ref().unwrap().offset;
        let mut file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .unwrap();
        file.seek(SeekFrom::Start(suffix_offset + 8)).unwrap();
        file.write_all(&[0xff]).unwrap();
        drop(file);

        let db = Database::open(&path).unwrap();
        assert_eq!(db.list_collections().unwrap().len(), 1);
        drop(db);

        let report = Database::check_path(
            &path,
            CheckOptions {
                verify_indexes: true,
            },
        )
        .unwrap();
        assert_eq!(report.status, "warning");
        assert_eq!(report.index_suffix_status, "rebuild_needed");
        assert_eq!(report.index_check, "rebuild_needed");
        assert!(report.issues.iter().any(|issue| {
            issue.code == "unsupported_index_checkpoint"
                && issue.stream == "index"
                && issue.visibility_impact == "rebuild_index_only"
        }));
        assert!(!report
            .issues
            .iter()
            .any(|issue| issue.visibility_impact == "blocks_main"));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn rebuild_index_rewrites_corrupt_suffix() {
        let path = temp_path("rebuild-index-corrupt-suffix");
        let db = Database::create(&path, DatabaseConfig::default()).unwrap();
        let col = db
            .create_collection(CollectionConfig {
                tokenizer: Default::default(),
                name: "notes".into(),
                embedding_dim: 2,
                distance_metric: DistanceMetric::Cosine,
                embedding_model: Some("m".into()),
                vectors_normalized: false,
                vector_index: VectorIndexConfig::Exact,
            })
            .unwrap();
        col.insert_document(DocumentInput {
            external_id: "doc".into(),
            title: None,
            source_uri: None,
            metadata: JsonValue::object(),
            chunks: vec![ChunkInput {
                chunk_external_id: "doc#1".into(),
                text: "explicit rebuild restores index checkpoint".into(),
                ordinal: 0,
                metadata: JsonValue::object(),
                source_span: None,
            }],
        })
        .unwrap();
        col.put_embeddings(vec![EmbeddingInput {
            chunk_external_id: "doc#1".into(),
            model_name: "m".into(),
            dim: 2,
            vector: vec![1.0, 0.0],
        }])
        .unwrap();
        drop(col);
        drop(db);
        append_fake_index_suffix(&path);

        let report = Database::check_path(
            &path,
            CheckOptions {
                verify_indexes: true,
            },
        )
        .unwrap();
        assert_eq!(report.index_suffix_status, "rebuild_needed");
        assert_eq!(report.index_check, "rebuild_needed");

        let db = Database::open(&path).unwrap();
        let summary = db.rebuild_index(Some("notes")).unwrap();
        assert_eq!(summary.collection.as_deref(), Some("notes"));
        assert_eq!(summary.rebuild_scope, "full_visible_snapshot");
        assert_eq!(summary.index_suffix_status, "persistent_checkpoint");
        assert_eq!(summary.index_record_count, 3);
        assert_eq!(summary.index_extent_count, 3);
        assert_eq!(summary.text_segment_count, 1);
        assert_eq!(summary.vector_segment_count, 1);
        assert!(!summary.rebuild_needed);
        drop(db);

        let repaired = Database::check_path(
            &path,
            CheckOptions {
                verify_indexes: true,
            },
        )
        .unwrap();
        assert_eq!(repaired.status, "ok");
        assert_eq!(repaired.index_suffix_status, "persistent_checkpoint");
        assert_eq!(repaired.index_check, "ok");
        let repaired_suffix = repaired.index_suffix.as_ref().unwrap();
        assert_eq!(
            summary.last_index_commit_id,
            Some(repaired_suffix.covered_last_commit_sequence_no)
        );
        assert!(
            repaired_suffix.covered_last_commit_sequence_no > repaired_suffix.covered_last_batch_id
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn rebuild_index_rejects_unknown_collection_scope() {
        let path = temp_path("rebuild-index-unknown-collection");
        let db = Database::create(&path, DatabaseConfig::default()).unwrap();
        db.create_collection(CollectionConfig {
            tokenizer: Default::default(),
            name: "notes".into(),
            embedding_dim: 2,
            distance_metric: DistanceMetric::Cosine,
            embedding_model: None,
            vectors_normalized: false,
            vector_index: VectorIndexConfig::Exact,
        })
        .unwrap();

        let err = db.rebuild_index(Some("missing")).unwrap_err();
        assert_eq!(err.code(), ErrorCode::CollectionNotFound);
        drop(db);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn rebuild_index_failure_clears_persistent_index_state() {
        let path = temp_path("rebuild-index-failure-clears-state");
        let db = Database::create(&path, DatabaseConfig::default()).unwrap();
        db.create_collection(CollectionConfig {
            tokenizer: Default::default(),
            name: "notes".into(),
            embedding_dim: 2,
            distance_metric: DistanceMetric::Cosine,
            embedding_model: None,
            vectors_normalized: false,
            vector_index: VectorIndexConfig::Exact,
        })
        .unwrap();
        let stats_before = db.index_stats().unwrap();
        assert!(!stats_before.rebuild_needed);
        assert!(stats_before.last_index_commit_id.is_some());

        fail_next_index_checkpoint_refresh(&db);
        let err = db.rebuild_index(Some("notes")).unwrap_err();
        assert_eq!(err.code(), ErrorCode::IoError);
        let stats_after = db.index_stats().unwrap();
        assert!(stats_after.rebuild_needed);
        assert_eq!(stats_after.last_index_commit_id, None);

        db.rebuild_index(Some("notes")).unwrap();
        let repaired_stats = db.index_stats().unwrap();
        assert!(!repaired_stats.rebuild_needed);
        assert!(repaired_stats.last_index_commit_id.is_some());
        drop(db);
        let _ = std::fs::remove_file(path);
    }

    fn import_plan_document(id: usize, title_bytes: usize) -> RecordPayload {
        RecordPayload::DocumentPut(Document {
            document_id: format!("doc_{id}"),
            collection_id: "col_1".into(),
            external_id: format!("d-{id}"),
            title: Some("x".repeat(title_bytes)),
            source_uri: None,
            metadata: JsonValue::object(),
            created_at: "1970-01-01T00:00:00Z".into(),
            updated_at: "1970-01-01T00:00:00Z".into(),
        })
    }

    #[test]
    fn import_batch_plan_respects_bytes_and_keeps_replacements_atomic() {
        for seed in 0..100 {
            let records = vec![
                import_plan_document(1, seed * 13),
                RecordPayload::Tombstone(Tombstone {
                    target_type: "document".into(),
                    target_id: "doc_1".into(),
                    deleted_at: "1970-01-01T00:00:00Z".into(),
                    reason: Some("replace".into()),
                }),
                import_plan_document(2, seed * 13),
            ];
            assert_eq!(
                plan_import_batches(&records, u64::MAX, 2).unwrap(),
                vec![1, 2]
            );
            let group_bytes = crate::storage::encoded_record_len(&records[1]).unwrap()
                + crate::storage::encoded_record_len(&records[2]).unwrap();
            // Frozen commit encoding is a 50-byte envelope and a 44-byte payload.
            assert_eq!(
                plan_import_batches(&records, group_bytes + 94, 100).unwrap(),
                vec![1, 2]
            );
            assert_eq!(
                plan_import_batches(&records, group_bytes + 93, 100).unwrap(),
                vec![1, 2]
            );
            assert_eq!(plan_import_batches(&records, 0, 100).unwrap(), vec![1, 2]);
            assert!(plan_import_batches(&records, u64::MAX, 1).is_err());
        }
        assert!(plan_import_batches(&[], 0, 0).unwrap().is_empty());
    }

    #[test]
    fn lazy_import_batch_plan_matches_predefined_groups_across_100_seeds() {
        let document = crate::storage::BatchRecordKind::of(&import_plan_document(1, 0));
        let tombstone = crate::storage::BatchRecordKind::of(&RecordPayload::Tombstone(Tombstone {
            target_type: "document".into(),
            target_id: "doc_1".into(),
            deleted_at: "1970-01-01T00:00:00Z".into(),
            reason: None,
        }));
        for seed in 0..100 {
            let byte_limit = 194 + seed as u64;
            let max_records = 2 + seed % 9;
            let mut records = Vec::new();
            let mut expected = Vec::new();
            let (mut count, mut bytes) = (0, 94);
            for index in 0..100 {
                let group_count = if (index + seed) % 3 == 0 { 2 } else { 1 };
                let group_bytes = 51 + (index + seed) as u64 % 37;
                if count > 0
                    && (count + group_count > max_records || bytes + group_bytes > byte_limit)
                {
                    expected.push(count);
                    count = 0;
                    bytes = 94;
                }
                count += group_count;
                bytes += group_bytes;
                if group_count == 2 {
                    records.push((tombstone, 17));
                    records.push((document, group_bytes - 17));
                } else {
                    records.push((document, group_bytes));
                }
            }
            expected.push(count);
            let observed = Cell::new(0);
            let input = records.iter().copied().map(|record| {
                observed.set(observed.get() + 1);
                Ok(record)
            });
            let mut batches = plan_record_batches(input, byte_limit, max_records).unwrap();
            assert_eq!(observed.get(), 0);
            assert_eq!(batches.next().unwrap().unwrap(), expected[0]);
            assert!(observed.get() <= expected[0] + 3);
            assert_eq!(
                batches.by_ref().collect::<Result<Vec<_>>>().unwrap(),
                expected[1..]
            );
            assert!(batches.next().is_none());
            assert_eq!(observed.get(), records.len());
        }

        for (records, byte_limit, max_records, code) in [
            (vec![Ok((document, 1))], 95, 0, ErrorCode::FileFormatError),
            (
                vec![Ok((tombstone, u64::MAX)), Ok((document, 1))],
                u64::MAX,
                usize::MAX,
                ErrorCode::FileFormatError,
            ),
            (
                vec![
                    Ok((document, 1)),
                    Err(Error::new(ErrorCode::IoError, "injected")),
                ],
                u64::MAX,
                1,
                ErrorCode::IoError,
            ),
        ] {
            let mut batches =
                plan_record_batches(records.into_iter(), byte_limit, max_records).unwrap();
            assert_eq!(batches.next().unwrap().unwrap_err().code(), code);
            assert!(batches.next().is_none());
            assert!(batches.next().is_none());
        }
        assert_eq!(
            plan_record_batches(
                [(document, u64::MAX - 94), (document, 1)]
                    .into_iter()
                    .map(Ok),
                u64::MAX,
                usize::MAX,
            )
            .unwrap()
            .collect::<Result<Vec<_>>>()
            .unwrap(),
            [1, 1]
        );
    }

    #[test]
    fn atomic_batch_byte_preferences_keep_oversized_groups_and_terminal_iterator_state() {
        let document = crate::storage::BatchRecordKind::of(&import_plan_document(1, 0));
        for seed in 0..100_u64 {
            for limit in [0, 93, 94, 95, (1 << 53) + seed, u64::MAX - seed] {
                let minimum_record = limit.saturating_sub(94);
                for extra in [0, 1, 94, 100] {
                    let record_bytes = minimum_record.saturating_add(extra);
                    let mut batches =
                        plan_record_batches([Ok((document, record_bytes))].into_iter(), limit, 1)
                            .unwrap();
                    assert_eq!(batches.next().unwrap().unwrap(), 1);
                    assert!(batches.next().is_none());
                    assert!(batches.next().is_none());
                }
            }
        }
    }

    #[test]
    fn lazy_import_keeps_above_target_tail_in_its_own_bounded_batch() {
        for cache_budget_mb in [1, 2, 4] {
            let dir = temp_path(&format!("lazy-import-preflight-{cache_budget_mb}"));
            fs::create_dir(&dir).unwrap();
            let path = dir.join("data.coffer");
            let db = Database::create(
                &path,
                DatabaseConfig {
                    cache_budget_mb: Some(cache_budget_mb),
                    resource_profile: ResourceProfile::Constrained,
                    ..DatabaseConfig::default()
                },
            )
            .unwrap();
            let collection = db
                .create_collection(CollectionConfig {
                    name: "notes".into(),
                    embedding_dim: 2,
                    distance_metric: DistanceMetric::Cosine,
                    embedding_model: None,
                    vectors_normalized: false,
                    vector_index: VectorIndexConfig::Exact,
                    tokenizer: Default::default(),
                })
                .unwrap();
            let collection_id = collection.collection_id().to_owned();
            drop(collection);
            let before_commit = db.stats().unwrap().last_commit_id;
            let mut records = (1..=251)
                .map(|id| import_plan_document(id, 0))
                .collect::<Vec<_>>();
            records.push(import_plan_document(
                252,
                batch_byte_target(cache_budget_mb) as usize,
            ));
            for record in &mut records {
                let RecordPayload::DocumentPut(document) = record else {
                    unreachable!()
                };
                document.collection_id.clone_from(&collection_id);
            }
            let prepared =
                crate::storage::PreparedRecords::create(records, &dir, 4 * 1_048_576).unwrap();
            let metrics = db
                .lock()
                .unwrap()
                .append_import_batches(
                    prepared,
                    ImportIndexBuildMode::DeferredFullCheckpoint,
                    &db.json_context(),
                )
                .unwrap();
            assert_eq!(metrics.committed_batches, 3);
            assert_eq!(db.stats().unwrap().last_commit_id, before_commit + 3);
            assert_eq!(
                db.collection("notes")
                    .unwrap()
                    .stats()
                    .unwrap()
                    .document_count,
                252
            );
            drop(db);
            assert_eq!(
                Database::open_read_only(&path)
                    .unwrap()
                    .collection("notes")
                    .unwrap()
                    .stats()
                    .unwrap()
                    .document_count,
                252
            );
            assert_eq!(fs::read_dir(&dir).unwrap().count(), 2);
            fs::remove_dir_all(dir).unwrap();
        }
    }

    fn manifest_batch_fixture(dir: &Path) -> PathBuf {
        if dir.exists() {
            fs::remove_dir_all(dir).unwrap();
        }
        fs::create_dir(dir).unwrap();
        let mut documents = String::new();
        let mut chunks = String::new();
        let mut embeddings = String::new();
        for index in 0..100 {
            documents.push_str(&format!(
                "{{\"external_id\":\"d-{index}\",\"metadata\":{{}}}}\n"
            ));
            for ordinal in 0..2 {
                let id = index * 2 + ordinal;
                chunks.push_str(&format!("{{\"document_external_id\":\"d-{index}\",\"chunk_external_id\":\"c-{id:03}\",\"ordinal\":{ordinal},\"text\":\"needle haystack\",\"metadata\":{{}}}}\n"));
                embeddings.push_str(&format!("{{\"chunk_external_id\":\"c-{id:03}\",\"model_name\":\"m\",\"dim\":2,\"vector\":[1,0]}}\n"));
            }
        }
        fs::write(dir.join("documents.jsonl"), documents).unwrap();
        fs::write(dir.join("chunks.jsonl"), chunks).unwrap();
        fs::write(dir.join("embeddings.jsonl"), embeddings).unwrap();
        let manifest = dir.join("manifest.json");
        fs::write(&manifest, r#"{"schema_version":"0.1","collection":"notes","embedding_dim":2,"distance_metric":"cosine","embedding_model":"m","documents":"documents.jsonl","chunks":"chunks.jsonl","embeddings":"embeddings.jsonl"}"#).unwrap();
        manifest
    }

    #[test]
    fn manifest_batches_build_once_or_inline_and_preserve_conflict_semantics() {
        let mut rankings = Vec::new();
        for mode in [
            ImportIndexBuildMode::DeferredFullCheckpoint,
            ImportIndexBuildMode::InlineFullCheckpoint,
        ] {
            let dir = temp_path(&format!("manifest-batches-{mode:?}"));
            let manifest = manifest_batch_fixture(&dir);
            let path = dir.join("data.coffer");
            let db = Database::create(
                &path,
                DatabaseConfig {
                    resource_profile: ResourceProfile::Constrained,
                    ..DatabaseConfig::default()
                },
            )
            .unwrap();
            let measured = db
                .import_manifest_measured_in(&manifest, OnConflict::Error, mode, db.json_context())
                .unwrap();
            assert_eq!(measured.batches.max_batch_chunks, 149);
            assert_eq!(measured.batches.committed_batches, 3);
            let summary = measured.summary;
            assert_eq!(
                (
                    summary.documents_inserted,
                    summary.chunks_inserted,
                    summary.embeddings_inserted
                ),
                (100, 200, 200)
            );
            {
                let inner = db.lock().unwrap();
                assert_eq!(inner.catalog.last_commit_id, 3);
                assert_eq!(
                    inner.index_checkpoint_refreshes,
                    if mode == ImportIndexBuildMode::DeferredFullCheckpoint {
                        1
                    } else {
                        3
                    }
                );
                assert_eq!(
                    measured.batches.checkpoint_cycles,
                    inner.index_checkpoint_refreshes
                );
                assert_eq!(
                    inner.index_suffix_status,
                    IndexSuffixState::PersistentCheckpoint
                );
            }
            let before = fs::read(&path).unwrap();
            let error = db
                .import_manifest_with_index_mode(&manifest, OnConflict::Error, mode)
                .unwrap_err();
            assert_eq!(error.code(), ErrorCode::DuplicateExternalId);
            assert_eq!(fs::read(&path).unwrap(), before);
            let measured = db
                .import_manifest_measured_in(&manifest, OnConflict::Skip, mode, db.json_context())
                .unwrap();
            assert_eq!(measured.batches.max_batch_chunks, 0);
            assert_eq!(measured.batches.committed_batches, 0);
            assert_eq!(measured.batches.checkpoint_cycles, 0);
            assert_eq!(measured.batches.checkpoint_ready_cycles, 0);
            assert!(measured.batches.final_checkpoint_ready);
            assert_eq!(measured.batches.commit_time, std::time::Duration::ZERO);
            assert_eq!(measured.batches.checkpoint_time, std::time::Duration::ZERO);
            let skipped = measured.summary;
            assert_eq!(
                (
                    skipped.documents_skipped,
                    skipped.chunks_skipped,
                    skipped.embeddings_skipped
                ),
                (100, 200, 200)
            );
            assert_eq!(fs::read(&path).unwrap(), before);
            let measured = db
                .import_manifest_measured_in(
                    &manifest,
                    OnConflict::Replace,
                    mode,
                    db.json_context(),
                )
                .unwrap();
            assert_eq!(measured.batches.max_batch_chunks, 125);
            let replaced = measured.summary;
            assert_eq!(
                (replaced.documents_replaced, replaced.chunks_replaced),
                (100, 200)
            );
            assert_eq!(
                (
                    replaced.documents_inserted,
                    replaced.chunks_inserted,
                    replaced.embeddings_inserted
                ),
                (100, 200, 200)
            );
            let results = db
                .collection("notes")
                .unwrap()
                .search()
                .text("needle")
                .query_embedding(QueryEmbeddingInput {
                    model_name: Some("m".into()),
                    dim: 2,
                    vector: vec![1.0, 0.0],
                })
                .top_k(20)
                .execute()
                .unwrap();
            rankings.push(
                results
                    .results
                    .into_iter()
                    .map(|hit| (hit.chunk_external_id, hit.final_score.to_bits()))
                    .collect::<Vec<_>>(),
            );
            drop(db);
            let reopened = Database::open_read_only(&path).unwrap();
            let results = reopened
                .collection("notes")
                .unwrap()
                .search()
                .text("needle")
                .query_embedding(QueryEmbeddingInput {
                    model_name: Some("m".into()),
                    dim: 2,
                    vector: vec![1.0, 0.0],
                })
                .top_k(20)
                .execute()
                .unwrap();
            assert_eq!(
                rankings.last().unwrap(),
                &results
                    .results
                    .into_iter()
                    .map(|hit| (hit.chunk_external_id, hit.final_score.to_bits()))
                    .collect::<Vec<_>>()
            );
            drop(reopened);
            fs::remove_dir_all(dir).unwrap();
        }
        assert_eq!(rankings[0], rankings[1]);
    }

    #[test]
    fn manifest_input_and_encoded_staging_share_one_runtime_disk_budget() {
        let dir = temp_path("manifest-runtime-disk-budget");
        let manifest = manifest_batch_fixture(&dir);
        let path = dir.join("data.coffer");
        let db = Database::create(
            &path,
            DatabaseConfig {
                resource_profile: ResourceProfile::Constrained,
                ..DatabaseConfig::default()
            },
        )
        .unwrap();
        let budget = db.temporary_disk_budget();
        assert!(Arc::ptr_eq(&budget, &db.clone().temporary_disk_budget()));
        let before = fs::read(&path).unwrap();
        let input_bytes = ["documents.jsonl", "chunks.jsonl", "embeddings.jsonl"]
            .iter()
            .map(|name| fs::metadata(dir.join(name)).unwrap().len())
            .sum::<u64>();
        let limit = budget.limit_bytes();
        let (file, mut held) = crate::temporary::create_budgeted_temporary_file_in(
            &dir,
            "busy-input",
            "tmp",
            budget.clone(),
        )
        .unwrap();
        for available in [input_bytes - 1, input_bytes + 128] {
            held.reserve_disk(limit - available).unwrap();
            let error = db
                .import_manifest(&manifest, OnConflict::Error)
                .unwrap_err();
            assert_eq!(error.code(), ErrorCode::ResourceExhausted);
            assert!(error
                .detail()
                .iter()
                .any(|(key, value)| key == "resource" && value == "temporary_disk"));
            assert_eq!(budget.used(), limit - available);
            assert_eq!(fs::read(&path).unwrap(), before);
            assert!(db.list_collections().unwrap().is_empty());
            held.release_disk(limit - available);
            assert_eq!(budget.used(), 0);
        }
        drop(file);
        drop(held);
        db.import_manifest(&manifest, OnConflict::Error).unwrap();
        assert_eq!(budget.used(), 0);
        drop(db);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn manifest_batch_failures_preserve_committed_prefix_and_validate_before_first_write() {
        for fail_after in [0, 1, 2] {
            let dir = temp_path(&format!("manifest-batch-failure-{fail_after}"));
            let manifest = manifest_batch_fixture(&dir);
            let path = dir.join("data.coffer");
            let db = Database::create(
                &path,
                DatabaseConfig {
                    resource_profile: ResourceProfile::Constrained,
                    ..DatabaseConfig::default()
                },
            )
            .unwrap();
            db.lock().unwrap().import_fail_after_batches = Some(fail_after);
            assert_eq!(
                db.import_manifest(&manifest, OnConflict::Error)
                    .unwrap_err()
                    .code(),
                ErrorCode::IoError
            );
            let scan = scan_file(&path).unwrap();
            assert_eq!(scan.commits, fail_after);
            assert_eq!(db.temporary_disk_budget().used(), 0);
            assert_eq!(scan.records.len(), fail_after * 250);
            assert!(scan.findings.is_empty());
            let counts = {
                let inner = db.lock().unwrap();
                assert_eq!(inner.index_checkpoint_refreshes, 0);
                (
                    inner.catalog.live_documents().count(),
                    inner.catalog.live_chunks().count(),
                    inner.catalog.live_embeddings().count(),
                )
            };
            assert_eq!(
                counts,
                match fail_after {
                    0 => (0, 0, 0),
                    1 => (100, 149, 0),
                    _ => (100, 200, 199),
                }
            );
            drop(db);
            let reopened = Database::open_read_only(&path).unwrap();
            let inner = reopened.lock().unwrap();
            assert_eq!(
                (
                    inner.catalog.live_documents().count(),
                    inner.catalog.live_chunks().count(),
                    inner.catalog.live_embeddings().count()
                ),
                counts
            );
            drop(inner);
            drop(reopened);
            fs::remove_dir_all(dir).unwrap();
        }
        let dir = temp_path("manifest-batch-preflight");
        let manifest = manifest_batch_fixture(&dir);
        let path = dir.join("data.coffer");
        let db = Database::create(
            &path,
            DatabaseConfig {
                resource_profile: ResourceProfile::Constrained,
                ..DatabaseConfig::default()
            },
        )
        .unwrap();
        let original = fs::read(&path).unwrap();
        let embeddings = fs::read_to_string(dir.join("embeddings.jsonl")).unwrap();
        for tail in [
            "not json",
            r#"{"chunk_external_id":"c-000","model_name":"m","dim":2,"vector":[1,0]}"#,
            r#"{"chunk_external_id":"c-999","model_name":"m","dim":2,"vector":[1,0]}"#,
        ] {
            fs::write(
                dir.join("embeddings.jsonl"),
                format!("{embeddings}{tail}\n"),
            )
            .unwrap();
            assert!(db.import_manifest(&manifest, OnConflict::Error).is_err());
            assert_eq!(fs::read(&path).unwrap(), original);
            assert!(db.list_collections().unwrap().is_empty());
        }
        fs::write(dir.join("embeddings.jsonl"), embeddings).unwrap();
        db.lock().unwrap().fail_next_index_checkpoint_refresh = true;
        let measured = db
            .import_manifest_measured_in(
                &manifest,
                OnConflict::Error,
                ImportIndexBuildMode::DeferredFullCheckpoint,
                db.json_context(),
            )
            .unwrap();
        assert_eq!(measured.batches.committed_batches, 3);
        assert_eq!(measured.batches.checkpoint_cycles, 1);
        assert_eq!(measured.batches.checkpoint_ready_cycles, 0);
        assert!(!measured.batches.final_checkpoint_ready);
        let summary = measured.summary;
        assert_eq!(
            (
                summary.documents_inserted,
                summary.chunks_inserted,
                summary.embeddings_inserted
            ),
            (100, 200, 200)
        );
        {
            let inner = db.lock().unwrap();
            assert_eq!(inner.catalog.last_commit_id, 3);
            assert_eq!(inner.index_checkpoint_refreshes, 1);
            assert_eq!(inner.index_suffix_status, IndexSuffixState::RebuildNeeded);
        }
        drop(db);
        let reopened = Database::open_read_only(&path).unwrap();
        assert_eq!(
            reopened.lock().unwrap().catalog.live_embeddings().count(),
            200
        );
        drop(reopened);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn streamed_embedding_validation_rejects_late_rows_before_skip_or_replace_commits() {
        let dir = temp_path("manifest-streamed-embedding-validation");
        let manifest = manifest_batch_fixture(&dir);
        let path = dir.join("data.coffer");
        let db = Database::create(&path, DatabaseConfig::default()).unwrap();
        db.import_manifest(&manifest, OnConflict::Error).unwrap();
        let original = fs::read(&path).unwrap();
        let embedding_path = dir.join("embeddings.jsonl");
        let embeddings = fs::read_to_string(&embedding_path).unwrap();
        let prefix = embeddings.lines().take(199).collect::<Vec<_>>().join("\n");
        for (last, code) in [
            (
                r#"{"chunk_external_id":"c-199","model_name":"m","dim":3,"vector":[1,0,0]}"#,
                ErrorCode::InvalidEmbeddingDimension,
            ),
            (
                r#"{"chunk_external_id":"c-199","model_name":"wrong","dim":2,"vector":[1,0]}"#,
                ErrorCode::EmbeddingModelMismatch,
            ),
            (
                r#"{"chunk_external_id":"c-199","model_name":"m","dim":2,"vector":[1]}"#,
                ErrorCode::InvalidEmbeddingDimension,
            ),
            (
                r#"{"chunk_external_id":"c-000","model_name":"m","dim":2,"vector":[1,0]}"#,
                ErrorCode::DuplicateExternalId,
            ),
            (
                r#"{"chunk_external_id":"missing","model_name":"m","dim":2,"vector":[1,0]}"#,
                ErrorCode::InvalidJsonlSchema,
            ),
        ] {
            fs::write(&embedding_path, format!("{prefix}\n{last}\n")).unwrap();
            for conflict in [OnConflict::Skip, OnConflict::Replace] {
                if conflict == OnConflict::Skip && code == ErrorCode::EmbeddingModelMismatch {
                    let skipped = db.import_manifest(&manifest, conflict).unwrap();
                    assert_eq!(skipped.embeddings_skipped, 200);
                    assert_eq!(fs::read(&path).unwrap(), original);
                    continue;
                }
                let error = db.import_manifest(&manifest, conflict).unwrap_err();
                assert_eq!(error.code(), code, "{conflict:?}: {error:?}");
                assert!(error
                    .detail()
                    .iter()
                    .any(|(key, value)| key == "line" && value == "200"));
                assert!(error
                    .detail()
                    .iter()
                    .any(|(key, value)| key == "path"
                        && value == &embedding_path.display().to_string()));
                assert_eq!(fs::read(&path).unwrap(), original);
                assert_eq!(db.lock().unwrap().catalog.live_embeddings().count(), 200);
            }
        }
        fs::write(&embedding_path, embeddings).unwrap();
        let skipped = db.import_manifest(&manifest, OnConflict::Skip).unwrap();
        assert_eq!(
            (skipped.embeddings_seen, skipped.embeddings_skipped),
            (200, 200)
        );
        assert_eq!(fs::read(&path).unwrap(), original);
        drop(db);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn external_embedding_duplicates_fail_before_any_import_batch_for_all_conflict_modes() {
        for conflict in [OnConflict::Error, OnConflict::Skip, OnConflict::Replace] {
            let dir = temp_path(&format!("manifest-external-duplicate-{conflict:?}"));
            let manifest = manifest_batch_fixture(&dir);
            let prefix = "x".repeat(256);
            for name in ["chunks.jsonl", "embeddings.jsonl"] {
                let input = fs::read_to_string(dir.join(name)).unwrap();
                fs::write(dir.join(name), input.replace("c-", &format!("{prefix}c-"))).unwrap();
            }
            let path = dir.join("data.coffer");
            let db = Database::create(
                &path,
                DatabaseConfig {
                    cache_budget_mb: Some(8),
                    ..DatabaseConfig::default()
                },
            )
            .unwrap();
            if conflict != OnConflict::Error {
                db.import_manifest(&manifest, OnConflict::Error).unwrap();
            }
            let original = fs::read(&path).unwrap();
            let embedding_path = dir.join("embeddings.jsonl");
            let embeddings = fs::read_to_string(&embedding_path).unwrap();
            let duplicate = embeddings.lines().next().unwrap();
            fs::write(&embedding_path, format!("{embeddings}\n\n{duplicate}\n")).unwrap();
            let error = db.import_manifest(&manifest, conflict).unwrap_err();
            assert_eq!(error.code(), ErrorCode::DuplicateExternalId, "{error:?}");
            assert_eq!(error_detail_value(&error, "line"), Some("203"));
            assert_eq!(error_detail_value(&error, "first_line"), Some("1"));
            assert_eq!(error_detail_value(&error, "path"), embedding_path.to_str());
            assert_eq!(fs::read(&path).unwrap(), original);
            assert_eq!(db.temporary_disk_budget().used(), 0);
            fs::write(&embedding_path, embeddings).unwrap();
            db.import_manifest(&manifest, conflict).unwrap();
            assert_eq!(db.lock().unwrap().catalog.live_embeddings().count(), 200);
            assert_eq!(db.temporary_disk_budget().used(), 0);
            drop(db);
            fs::remove_dir_all(dir).unwrap();
        }
    }

    #[test]
    fn external_ordinal_validation_preserves_document_scope_and_chunk_locations() {
        for with_embeddings in [false, true] {
            for conflict in [OnConflict::Error, OnConflict::Skip, OnConflict::Replace] {
                let dir = temp_path(&format!(
                    "manifest-ordinal-sort-{with_embeddings}-{conflict:?}"
                ));
                fs::create_dir(&dir).unwrap();
                let manifest = dir.join("manifest.json");
                let embedding_field = if with_embeddings {
                    ",\"embeddings\":\"embeddings.jsonl\""
                } else {
                    ""
                };
                fs::write(&manifest, format!("{{\"schema_version\":\"0.1\",\"collection\":\"notes\",\"embedding_dim\":2,\"distance_metric\":\"cosine\",\"embedding_model\":\"m\",\"documents\":\"documents.jsonl\",\"chunks\":\"chunks.jsonl\"{embedding_field}}}")).unwrap();
                fs::write(
                    dir.join("documents.jsonl"),
                    "{\"external_id\":\"left\",\"metadata\":{}}\n{\"external_id\":\"right\",\"metadata\":{}}\n",
                )
                .unwrap();
                let mut chunks = String::new();
                for index in 0..700 {
                    let ordinal = if index == 699 { u32::MAX } else { index };
                    for document in ["left", "right"] {
                        chunks.push_str(&format!("{{\"document_external_id\":\"{document}\",\"chunk_external_id\":\"{document}-{index}\",\"ordinal\":{ordinal},\"text\":\"needle\",\"metadata\":{{}}}}\n"));
                    }
                }
                let chunks_path = dir.join("chunks.jsonl");
                fs::write(&chunks_path, &chunks).unwrap();
                fs::write(dir.join("embeddings.jsonl"), "{\"chunk_external_id\":\"left-0\",\"model_name\":\"m\",\"dim\":2,\"vector\":[1,0]}\n").unwrap();
                let path = dir.join("data.coffer");
                let db = Database::create(
                    &path,
                    DatabaseConfig {
                        cache_budget_mb: Some(8),
                        ..DatabaseConfig::default()
                    },
                )
                .unwrap();
                if conflict != OnConflict::Error {
                    db.import_manifest(&manifest, OnConflict::Error).unwrap();
                }
                let before = fs::read(&path).unwrap();
                fs::write(&chunks_path, format!("{chunks}\n\n{{\"document_external_id\":\"left\",\"chunk_external_id\":\"late\",\"ordinal\":0,\"text\":\"duplicate\",\"metadata\":{{}}}}\n")).unwrap();
                let error = db.import_manifest(&manifest, conflict).unwrap_err();
                assert_eq!(error.code(), ErrorCode::InvalidJsonlSchema, "{error:?}");
                assert_eq!(
                    error.message(),
                    "chunk ordinal must be unique within a document"
                );
                assert_eq!(error_detail_value(&error, "path"), chunks_path.to_str());
                assert_eq!(error_detail_value(&error, "line"), Some("1403"));
                assert_eq!(error_detail_value(&error, "first_line"), Some("1"));
                assert_eq!(fs::read(&path).unwrap(), before);
                assert_eq!(db.temporary_disk_budget().used(), 0);
                fs::write(&chunks_path, chunks).unwrap();
                db.import_manifest(&manifest, conflict).unwrap();
                let inner = db.lock().unwrap();
                assert_eq!(inner.catalog.live_documents().count(), 2);
                assert_eq!(inner.catalog.live_chunks().count(), 1400);
                assert_eq!(
                    inner.catalog.live_embeddings().count(),
                    usize::from(with_embeddings)
                );
                drop(inner);
                drop(db);
                let reopened = Database::open_read_only(&path).unwrap();
                assert_eq!(reopened.lock().unwrap().catalog.live_chunks().count(), 1400);
                drop(reopened);
                fs::remove_dir_all(dir).unwrap();
            }
        }
    }

    #[test]
    fn metadata_heavy_import_reuses_json_work_and_preserves_admitted_commits_on_failure() {
        for fail_after in [None, Some(1)] {
            let dir = temp_path(&format!("manifest-metadata-admission-{fail_after:?}"));
            fs::create_dir(&dir).unwrap();
            let manifest = dir.join("manifest.json");
            fs::write(&manifest, r#"{"schema_version":"0.1","collection":"notes","embedding_dim":2,"distance_metric":"cosine","documents":"documents.jsonl","chunks":"chunks.jsonl"}"#).unwrap();
            fs::write(dir.join("chunks.jsonl"), "").unwrap();
            let mut output =
                std::io::BufWriter::new(File::create(dir.join("documents.jsonl")).unwrap());
            let payload = "x".repeat(1_048_576);
            for id in 0..12 {
                writeln!(output, "{{\"external_id\":\"d-{id}\",\"metadata\":{{\"payload\":\"{payload}\",\"id\":{id}}}}}").unwrap();
            }
            output.flush().unwrap();
            drop(output);
            let path = dir.join("data.coffer");
            let db = Database::create(
                &path,
                DatabaseConfig {
                    cache_budget_mb: Some(512),
                    ..DatabaseConfig::default()
                },
            )
            .unwrap();
            let other = db.json_context();
            let mut busy = other.reservation();
            busy.grow(24 * 1_048_576, crate::json::JsonInputKind::JsonlRecord)
                .unwrap();
            db.lock().unwrap().import_fail_after_batches = fail_after;
            let imported = db.import_manifest(&manifest, OnConflict::Error);
            let count = db.lock().unwrap().catalog.live_documents().count();
            if fail_after.is_some() {
                assert_eq!(imported.unwrap_err().code(), ErrorCode::IoError);
                assert!(count > 0 && count < 12);
                assert_eq!(db.lock().unwrap().catalog.last_commit_id, 1);
            } else {
                assert_eq!(imported.unwrap().documents_inserted, 12);
                assert_eq!(count, 12);
                assert!(db.lock().unwrap().catalog.last_commit_id > 1);
            }
            other
                .reservation()
                .grow(
                    8 * 1_048_576 - snapshot_directory_bytes(&db),
                    crate::json::JsonInputKind::JsonlRecord,
                )
                .unwrap();
            assert_eq!(db.temporary_disk_budget().used(), 0);
            drop(busy);
            drop(db);
            let reopened = Database::open_read_only(&path).unwrap();
            assert_eq!(
                reopened.lock().unwrap().catalog.live_documents().count(),
                count
            );
            let collection = reopened.collection("notes").unwrap();
            for id in 0..count {
                let document = collection
                    .get_document_by_external_id(&format!("d-{id}"))
                    .unwrap();
                assert_eq!(
                    document.metadata.get("id").and_then(JsonValue::as_f64),
                    Some(id as f64)
                );
                assert_eq!(
                    document
                        .metadata
                        .get("payload")
                        .and_then(JsonValue::as_str)
                        .unwrap(),
                    payload
                );
            }
            drop(collection);
            drop(reopened);
            fs::remove_dir_all(dir).unwrap();
        }
    }

    #[test]
    fn committed_batches_can_defer_indexes_without_losing_visibility_or_reuse() {
        for fail_checkpoint in [false, true] {
            let path = temp_path(if fail_checkpoint {
                "deferred-index-failure"
            } else {
                "deferred-index-success"
            });
            let db = Database::create(&path, DatabaseConfig::default()).unwrap();
            let config = |name: &str| CollectionConfig {
                tokenizer: Default::default(),
                name: name.into(),
                embedding_dim: 2,
                distance_metric: DistanceMetric::Cosine,
                embedding_model: None,
                vectors_normalized: false,
                vector_index: VectorIndexConfig::Exact,
            };
            let changed = db.create_collection(config("changed")).unwrap();
            let other = db.create_collection(config("other")).unwrap();
            other
                .insert_document(DocumentInput {
                    external_id: "stable".into(),
                    title: None,
                    source_uri: None,
                    metadata: JsonValue::object(),
                    chunks: vec![ChunkInput {
                        chunk_external_id: "stable".into(),
                        text: "stable content".into(),
                        ordinal: 0,
                        metadata: JsonValue::object(),
                        source_span: None,
                    }],
                })
                .unwrap();
            other
                .put_embeddings(vec![EmbeddingInput {
                    chunk_external_id: "stable".into(),
                    model_name: "m".into(),
                    dim: 2,
                    vector: vec![1.0, 0.0],
                }])
                .unwrap();
            let old_reader = Database::open_read_only(&path).unwrap();
            let mut inner = db.lock().unwrap();
            let initial_commit = inner.catalog.last_commit_id;
            let initial_term_count = inner.text_index.term_count();
            let vector_pointer = inner.persistent_vector_segments.as_ref().unwrap()
                [other.collection_id()]
            .vectors[0]
                .vector
                .as_ptr();
            let mut pending: Option<PendingIndexUpdate> = None;
            for index in 0..100 {
                let document_id = inner.catalog.next_id("doc_", 0).unwrap();
                let chunk_id = inner.catalog.next_id("chk_", 0).unwrap();
                let text = "fresh content".to_owned();
                let records = vec![
                    RecordPayload::DocumentPut(Document {
                        document_id: document_id.clone(),
                        collection_id: changed.collection_id().into(),
                        external_id: format!("doc-{index}"),
                        title: None,
                        source_uri: None,
                        metadata: JsonValue::object(),
                        created_at: "1970-01-01T00:00:00Z".into(),
                        updated_at: "1970-01-01T00:00:00Z".into(),
                    }),
                    RecordPayload::ChunkPut(Chunk {
                        chunk_id,
                        document_id,
                        collection_id: changed.collection_id().into(),
                        chunk_external_id: format!("chunk-{index:03}"),
                        text_hash: chunk_text_hash(&text),
                        token_count: token_count_hint(&text),
                        text,
                        ordinal: 0,
                        metadata: JsonValue::object(),
                        embedding_id: None,
                        source_span: None,
                        created_at: "1970-01-01T00:00:00Z".into(),
                    }),
                ];
                let next = inner.commit_records(records).unwrap();
                if let Some(pending) = pending.as_mut() {
                    assert!(!next.reusable_checkpoint);
                    assert!(next.reusable_vectors.is_empty());
                    pending.changes.extend(&next.changes).unwrap();
                } else {
                    assert!(next.reusable_checkpoint);
                    pending = Some(next);
                }
                assert_eq!(inner.catalog.last_commit_id, initial_commit + index + 1);
                assert_eq!(inner.index_suffix_status, IndexSuffixState::RebuildNeeded);
                assert_eq!(inner.index_size_bytes, 0);
                assert!(inner.persistent_vector_segments.is_none());
                assert_eq!(inner.text_index.term_count(), initial_term_count);
                assert_eq!(
                    fs::metadata(&path).unwrap().len(),
                    inner.storage.critical_end_offset()
                );
            }
            let pending = pending.unwrap();
            assert_eq!(
                pending.reusable_vectors[other.collection_id()].vectors[0]
                    .vector
                    .as_ptr(),
                vector_pointer
            );
            drop(inner);
            let before = changed
                .search()
                .text("fresh")
                .top_k(100)
                .resource_profile(ResourceProfile::Expanded)
                .execute()
                .unwrap();
            assert_eq!(before.results.len(), 100);
            assert_eq!(before.query_context.index_source, "transient_rebuild");
            assert!(old_reader
                .collection("changed")
                .unwrap()
                .search()
                .text("fresh")
                .execute()
                .unwrap()
                .results
                .is_empty());
            assert_eq!(
                old_reader
                    .collection("other")
                    .unwrap()
                    .search()
                    .text("stable")
                    .execute()
                    .unwrap()
                    .results
                    .len(),
                1
            );
            {
                let mut inner = db.lock().unwrap();
                inner.fail_next_index_checkpoint_refresh = fail_checkpoint;
                inner
                    .finish_index_update(pending, &db.json_context())
                    .unwrap();
                assert_eq!(inner.catalog.last_commit_id, initial_commit + 100);
                assert_eq!(inner.index_stats().rebuild_needed, fail_checkpoint);
                if !fail_checkpoint {
                    assert_eq!(
                        inner.persistent_vector_segments.as_ref().unwrap()[other.collection_id()]
                            .vectors[0]
                            .vector
                            .as_ptr(),
                        vector_pointer
                    );
                }
            }
            let after = changed
                .search()
                .text("fresh")
                .top_k(100)
                .resource_profile(ResourceProfile::Expanded)
                .execute()
                .unwrap();
            let signature = |results: &SearchResults| {
                results
                    .results
                    .iter()
                    .map(|hit| (hit.chunk_external_id.clone(), hit.final_score.to_bits()))
                    .collect::<Vec<_>>()
            };
            assert_eq!(signature(&before), signature(&after));
            let scan = scan_file(&path).unwrap();
            assert_eq!(scan.index_checkpoint.is_none(), fail_checkpoint);
            if let Some(checkpoint) = scan.index_checkpoint {
                assert_eq!(
                    checkpoint.metadata.covered_last_batch_id,
                    initial_commit + 100
                );
                assert_eq!(checkpoint.metadata.live_chunk_count, 101);
            }
            drop((changed, other, old_reader, db));
            let reopened = Database::open(&path).unwrap();
            let recovered = reopened
                .collection("changed")
                .unwrap()
                .search()
                .text("fresh")
                .top_k(100)
                .resource_profile(ResourceProfile::Expanded)
                .execute()
                .unwrap();
            assert_eq!(signature(&after), signature(&recovered));
            if fail_checkpoint {
                reopened.rebuild_index(None).unwrap();
                assert!(!reopened.index_stats().unwrap().rebuild_needed);
            }
            drop(reopened);
            fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn embedding_replacement_bounds_catalog_without_losing_snapshots_or_id_history() {
        let dir = temp_path("embedding-history-catalog");
        fs::create_dir(&dir).unwrap();
        let path = dir.join("data.coffer");
        let db = Database::create(
            &path,
            DatabaseConfig {
                durability: Durability::Fast,
                ..DatabaseConfig::default()
            },
        )
        .unwrap();
        let collection = db
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
        collection
            .insert_document(DocumentInput {
                external_id: "document".into(),
                title: None,
                source_uri: None,
                metadata: JsonValue::object(),
                chunks: ["a", "b"]
                    .into_iter()
                    .enumerate()
                    .map(|(ordinal, id)| ChunkInput {
                        chunk_external_id: id.into(),
                        ordinal,
                        text: "shared retrieval".into(),
                        metadata: JsonValue::object(),
                        source_span: None,
                    })
                    .collect(),
            })
            .unwrap();
        let embedding = |id: &str, vector: Vec<f32>| EmbeddingInput {
            chunk_external_id: id.into(),
            model_name: "m".into(),
            dim: 2,
            vector,
        };
        collection
            .put_embeddings(vec![
                embedding("a", vec![1.0, 0.0]),
                embedding("b", vec![1.0, 1.0]),
            ])
            .unwrap();
        let query = |db: &Database| {
            db.collection("notes")
                .unwrap()
                .search()
                .query_embedding(QueryEmbeddingInput {
                    model_name: Some("m".into()),
                    dim: 2,
                    vector: vec![1.0, 0.0],
                })
                .execute()
                .unwrap()
                .results
                .into_iter()
                .map(|hit| (hit.chunk_external_id, hit.final_score.to_bits()))
                .collect::<Vec<_>>()
        };
        let initial = query(&db);
        let old_reader = Database::open_read_only(&path).unwrap();
        let mut previous = collection
            .get_chunk_by_external_id("a")
            .unwrap()
            .embedding_id
            .unwrap();
        for revision in 0..100 {
            let vector = if revision % 2 == 0 {
                vec![0.0, 1.0]
            } else {
                vec![1.0, 0.0]
            };
            let result = db
                .put_embeddings_in_collection(
                    collection.collection_id(),
                    vec![embedding("a", vector)],
                    OnConflict::Replace,
                )
                .unwrap();
            assert_eq!(result.embeddings_replaced, 1);
            let inner = db.lock().unwrap();
            assert_eq!(inner.catalog.embeddings.len(), 2);
            assert_eq!(inner.catalog.chunk_embeddings.len(), 2);
            assert_eq!(inner.catalog.record_refs.len(), 6);
            assert!(!inner.catalog.embeddings.contains_key(&previous));
            assert!(inner.catalog.record_ref_for_embedding(&previous).is_none());
            assert_eq!(inner.catalog.id_high_watermarks["emb_"], revision + 3);
            drop(inner);
            previous = collection
                .get_chunk_by_external_id("a")
                .unwrap()
                .embedding_id
                .unwrap();
            assert_eq!(query(&db)[0].0, if revision % 2 == 0 { "b" } else { "a" });
        }
        assert_eq!(query(&old_reader), initial);
        assert_eq!(query(&db), initial);
        let scan = scan_file(&path).unwrap();
        assert!(scan.findings.is_empty());
        assert_eq!(
            scan.records
                .iter()
                .filter(|record| matches!(record.payload, RecordPayload::EmbeddingPut(_)))
                .count(),
            102
        );
        assert_eq!(
            db.check(CheckOptions {
                verify_indexes: true
            })
            .unwrap()
            .status,
            "ok"
        );
        drop(collection);
        drop(db);

        let db = Database::open(&path).unwrap();
        assert_eq!(db.lock().unwrap().catalog.embeddings.len(), 2);
        assert_eq!(db.lock().unwrap().catalog.record_refs.len(), 6);
        assert_eq!(query(&db), initial);
        let collection = db.collection("notes").unwrap();
        let deleted = collection.get_chunk_by_external_id("b").unwrap();
        collection.delete_chunk(&deleted.chunk_id).unwrap();
        let diagnosis = db
            .why_not_debug(
                "notes",
                "b",
                None,
                Some(QueryEmbeddingInput {
                    model_name: Some("m".into()),
                    dim: 2,
                    vector: vec![1.0, 0.0],
                }),
                JsonValue::object(),
                None,
                false,
                10,
                None,
            )
            .unwrap();
        let expected = diagnosis.result.get("expected_chunk").unwrap();
        assert_eq!(expected.get("deleted"), Some(&JsonValue::Bool(true)));
        assert_eq!(
            expected.get("missing_embedding"),
            Some(&JsonValue::Bool(false))
        );
        assert_eq!(
            expected.get("embedding_dim_matches"),
            Some(&JsonValue::Bool(true))
        );
        db.compact(CompactOptions {
            rebuild_indexes: true,
        })
        .unwrap();
        assert_eq!(db.lock().unwrap().catalog.embeddings.len(), 1);
        assert_eq!(db.lock().unwrap().catalog.id_high_watermarks["emb_"], 102);
        assert_eq!(query(&old_reader), initial);
        drop(collection);
        drop(db);

        let db = Database::open(&path).unwrap();
        let collection = db.collection("notes").unwrap();
        db.put_embeddings_in_collection(
            collection.collection_id(),
            vec![embedding("a", vec![1.0, 0.0])],
            OnConflict::Replace,
        )
        .unwrap();
        assert_eq!(
            collection
                .get_chunk_by_external_id("a")
                .unwrap()
                .embedding_id
                .as_deref(),
            Some("emb_0000000000000103")
        );
        assert_eq!(db.lock().unwrap().catalog.embeddings.len(), 1);
        assert_eq!(db.stats().unwrap().embedding_count, 1);
        assert_eq!(
            db.check(CheckOptions {
                verify_indexes: true
            })
            .unwrap()
            .status,
            "ok"
        );
        drop((collection, db, old_reader));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn embedding_only_commit_reuses_text_postings_and_keeps_full_checkpoint() {
        let path = temp_path("embedding-only-reuses-text");
        let db = Database::create(&path, DatabaseConfig::default()).unwrap();
        for name in ["notes", "other"] {
            let collection = db
                .create_collection(CollectionConfig {
                    tokenizer: Default::default(),
                    name: name.into(),
                    embedding_dim: 2,
                    distance_metric: DistanceMetric::Cosine,
                    embedding_model: None,
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
                        chunk_external_id: "chunk".into(),
                        text: "consistent retrieval words".into(),
                        ordinal: 0,
                        metadata: JsonValue::object(),
                        source_span: None,
                    }],
                })
                .unwrap();
        }
        drop(db);
        let db = Database::open(&path).unwrap();
        let col = db.collection("notes").unwrap();
        let before = col.search().text("retrieval").execute().unwrap().results[0].clone();
        let posting_pointer = {
            let inner = db.lock().unwrap();
            inner.text_index.postings()["retrieval"].as_ptr() as usize
        };
        col.put_embeddings(vec![EmbeddingInput {
            chunk_external_id: "chunk".into(),
            model_name: "calibration".into(),
            dim: 2,
            vector: vec![1.0, 0.0],
        }])
        .unwrap();
        {
            let inner = db.lock().unwrap();
            assert_eq!(
                inner.text_index.postings()["retrieval"].as_ptr() as usize,
                posting_pointer
            );
            assert_eq!(
                inner.index_suffix_status,
                IndexSuffixState::PersistentCheckpoint
            );
        }
        let after = col.search().text("retrieval").execute().unwrap().results[0].clone();
        assert_eq!(after.chunk_id, before.chunk_id);
        assert_eq!(after.bm25_score, before.bm25_score);
        let scan = scan_file(&path).unwrap();
        let checkpoint = scan.index_checkpoint.unwrap();
        assert_eq!(checkpoint.text_segments.len(), 2);
        assert_eq!(checkpoint.vector_segments.len(), 2);
        assert_eq!(checkpoint.metadata.live_embedding_count, 1);
        assert!(!db.index_stats().unwrap().rebuild_needed);
        drop(col);
        drop(db);
        let db = Database::open(&path).unwrap();
        assert_eq!(
            db.collection("other")
                .unwrap()
                .search()
                .text("retrieval")
                .execute()
                .unwrap()
                .results
                .len(),
            1
        );
        assert_eq!(
            db.collection("notes")
                .unwrap()
                .search()
                .query_embedding(QueryEmbeddingInput {
                    model_name: None,
                    dim: 2,
                    vector: vec![1.0, 0.0]
                })
                .execute()
                .unwrap()
                .results
                .len(),
            1
        );
        drop(db);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn collection_local_changes_reuse_content_and_match_full_rebuild() {
        let path = temp_path("collection-local-index-reuse");
        let db = Database::create(&path, DatabaseConfig::default()).unwrap();
        let config = |name: &str| CollectionConfig {
            tokenizer: Default::default(),
            name: name.into(),
            embedding_dim: 2,
            distance_metric: DistanceMetric::Cosine,
            embedding_model: None,
            vectors_normalized: false,
            vector_index: VectorIndexConfig::Exact,
        };
        let input = |id: &str, text: &str| DocumentInput {
            external_id: id.into(),
            title: None,
            source_uri: None,
            metadata: JsonValue::object(),
            chunks: vec![ChunkInput {
                chunk_external_id: id.into(),
                text: text.into(),
                ordinal: 0,
                metadata: JsonValue::object(),
                source_span: None,
            }],
        };
        let embedding = |id: &str| EmbeddingInput {
            chunk_external_id: id.into(),
            model_name: "local".into(),
            dim: 2,
            vector: vec![1.0, 0.0],
        };
        let changed = db.create_collection(config("changed")).unwrap();
        let untouched = db.create_collection(config("untouched")).unwrap();
        changed
            .insert_document(input("replace", "shared obsolete"))
            .unwrap();
        changed
            .insert_document(input("delete", "shared retained"))
            .unwrap();
        changed
            .insert_document(input("emptytokens", "!!!"))
            .unwrap();
        untouched
            .insert_document(input("untouched", "shared immutable"))
            .unwrap();
        changed
            .put_embeddings(vec![embedding("replace"), embedding("delete")])
            .unwrap();
        untouched
            .put_embeddings(vec![embedding("untouched")])
            .unwrap();
        drop(changed);
        drop(untouched);
        drop(db);

        let db = Database::open(&path).unwrap();
        let changed = db.collection("changed").unwrap();
        let untouched = db.collection("untouched").unwrap();
        let before = untouched.search().text("shared").execute().unwrap();
        untouched
            .search()
            .query_embedding(QueryEmbeddingInput {
                model_name: None,
                dim: 2,
                vector: vec![1.0, 0.0],
            })
            .execute()
            .unwrap();
        let (posting_pointer, vector_pointer) = {
            let inner = db.lock().unwrap();
            (
                inner.text_index.postings()["immutable"].as_ptr() as usize,
                inner.lazy_vector_segment.as_ref().unwrap().vectors[0]
                    .vector
                    .as_ptr() as usize,
            )
        };
        let verify = || {
            let inner = db.lock().unwrap();
            assert_eq!(
                inner.index_suffix_status,
                IndexSuffixState::PersistentCheckpoint
            );
            assert_eq!(
                inner.text_index.postings()["immutable"].as_ptr() as usize,
                posting_pointer
            );
            assert_eq!(
                inner.persistent_vector_segments.as_ref().unwrap()[&untouched.collection_id].vectors
                    [0]
                .vector
                .as_ptr() as usize,
                vector_pointer
            );
            let full = build_text_index(
                &inner.storage,
                &inner.catalog,
                &inner.temporary_work,
                &db.json_context(),
            )
            .unwrap();
            for collection in inner.catalog.live_collections() {
                for query in ["shared", "obsolete", "fresh", "retained", "immutable"] {
                    assert_eq!(
                        format!(
                            "{:?}",
                            inner
                                .text_index
                                .search(
                                    &collection.collection_id,
                                    query,
                                    TextSearchOptions::with_explanations(20),
                                    &db.json_context()
                                )
                                .unwrap()
                                .as_slice()
                        ),
                        format!(
                            "{:?}",
                            full.search(
                                &collection.collection_id,
                                query,
                                TextSearchOptions::with_explanations(20),
                                &db.json_context()
                            )
                            .unwrap()
                            .as_slice()
                        )
                    );
                }
            }
            let rebuilt_vectors = inner
                .vector_index_segments(BTreeMap::new(), &BTreeSet::new())
                .unwrap();
            for segment in rebuilt_vectors {
                assert_eq!(
                    format!("{segment:?}"),
                    format!(
                        "{:?}",
                        inner.persistent_vector_segments.as_ref().unwrap()[&segment.collection_id]
                    )
                );
            }
        };

        {
            let inner = db.lock().unwrap();
            let full = inner
                .vector_index_segments(BTreeMap::new(), &BTreeSet::new())
                .unwrap();
            for invalid_source in ["missing", "dimension", "model", "collection", "changed"] {
                let mut source = vector_segments_by_collection(full.clone());
                let mut changes = BTreeSet::new();
                match invalid_source {
                    "missing" => {
                        source.remove(&untouched.collection_id);
                    }
                    "dimension" => source.get_mut(&untouched.collection_id).unwrap().dim += 1,
                    "model" => {
                        source
                            .get_mut(&untouched.collection_id)
                            .unwrap()
                            .embedding_model = Some("wrong".into())
                    }
                    "collection" => {
                        source
                            .get_mut(&untouched.collection_id)
                            .unwrap()
                            .collection_id = "wrong".into()
                    }
                    "changed" => {
                        source.get_mut(&untouched.collection_id).unwrap().vectors[0].vector[0] =
                            0.0;
                        changes.insert(untouched.collection_id.clone());
                    }
                    _ => unreachable!(),
                }
                assert_eq!(
                    format!(
                        "{:?}",
                        inner.vector_index_segments(source, &changes).unwrap()
                    ),
                    format!("{full:?}"),
                    "{invalid_source}"
                );
            }
        }

        db.insert_document_in_collection(
            &changed.collection_id,
            input("replace", "fresh shared shared"),
            OnConflict::Replace,
        )
        .unwrap();
        verify();
        changed.put_embeddings(vec![embedding("replace")]).unwrap();
        verify();
        let deleted = changed.get_chunk_by_external_id("delete").unwrap();
        changed.delete_chunk(&deleted.chunk_id).unwrap();
        verify();
        let replaced = changed.get_document_by_external_id("replace").unwrap();
        changed.delete_document(&replaced.document_id).unwrap();
        verify();
        db.create_collection(config("empty")).unwrap();
        verify();
        db.drop_collection("changed").unwrap();
        verify();

        let after = untouched.search().text("shared").execute().unwrap();
        assert_eq!(after.results[0].bm25_score, before.results[0].bm25_score);
        assert_eq!(after.results[0].final_score, before.results[0].final_score);
        assert_eq!(
            after.results[0].explain.get("matched_terms"),
            before.results[0].explain.get("matched_terms")
        );
        let scan = scan_file(&path).unwrap();
        assert!(scan.findings.is_empty());
        let checkpoint = scan.index_checkpoint.unwrap();
        assert_eq!(checkpoint.text_segments.len(), 2);
        assert_eq!(checkpoint.vector_segments.len(), 2);
        assert!(checkpoint.text_segments.iter().all(|segment| {
            segment.collection_id != changed.collection_id
                && segment.covered_commit_sequence_no == scan.last_valid_commit_sequence_no
        }));
        assert!(checkpoint.vector_segments.iter().all(|segment| {
            segment.collection_id != changed.collection_id
                && segment.covered_commit_sequence_no == scan.last_valid_commit_sequence_no
        }));
        drop(changed);
        drop(untouched);
        drop(db);
        let db = Database::open(&path).unwrap();
        let reopened = db
            .collection("untouched")
            .unwrap()
            .search()
            .text("shared")
            .execute()
            .unwrap();
        assert_eq!(reopened.results[0].bm25_score, after.results[0].bm25_score);
        assert_eq!(reopened.query_context.index_source, "persistent_checkpoint");
        assert_eq!(db.stats().unwrap().embedding_count, 1);
        assert_eq!(
            db.collection("empty").unwrap().stats().unwrap().chunk_count,
            0
        );
        drop(db);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn local_rebuild_uses_validated_owned_sources_before_suffix_truncate() {
        for (name, budget) in [("warm", None), ("cold", Some(1)), ("disabled", Some(0))] {
            let path = temp_path(&format!("local-rebuild-owned-sources-{name}"));
            let mut db = Database::create(&path, DatabaseConfig::default()).unwrap();
            let mut collections = Vec::new();
            for name in ["changed", "untouched"] {
                let collection = db
                    .create_collection(CollectionConfig {
                        tokenizer: Default::default(),
                        name: name.into(),
                        embedding_dim: 2,
                        distance_metric: DistanceMetric::Cosine,
                        embedding_model: None,
                        vectors_normalized: false,
                        vector_index: VectorIndexConfig::Exact,
                    })
                    .unwrap();
                collection
                    .insert_document(DocumentInput {
                        external_id: name.into(),
                        title: None,
                        source_uri: None,
                        metadata: JsonValue::object(),
                        chunks: vec![ChunkInput {
                            chunk_external_id: name.into(),
                            text: format!("shared {name}"),
                            ordinal: 0,
                            metadata: JsonValue::object(),
                            source_span: None,
                        }],
                    })
                    .unwrap();
                collection
                    .put_embeddings(vec![EmbeddingInput {
                        chunk_external_id: name.into(),
                        model_name: "local".into(),
                        dim: 2,
                        vector: vec![1.0, 0.0],
                    }])
                    .unwrap();
                collections.push(collection);
            }
            if let Some(budget) = budget {
                drop(collections);
                drop(db);
                db = Database::open_with_options(
                    &path,
                    DatabaseOpenOptions {
                        cache_budget_mb: Some(budget.max(1)),
                        ..DatabaseOpenOptions::default()
                    },
                )
                .unwrap();
                collections = ["changed", "untouched"]
                    .into_iter()
                    .map(|name| db.collection(name).unwrap())
                    .collect();
                let mut inner = db.lock().unwrap();
                assert!(inner.persistent_vector_segments.is_none());
                assert!(inner.lazy_vector_segment.is_none());
                assert!(inner.lazy_vector_source.is_some());
                inner.config.cache_budget_mb = Some(budget);
            }
            let offsets = {
                let inner = db.lock().unwrap();
                let chunk = inner
                    .catalog
                    .chunk_by_external_id(&collections[1].collection_id, "untouched")
                    .unwrap();
                let embedding = inner.catalog.embedding_for_chunk(&chunk.chunk_id).unwrap();
                [
                    inner
                        .catalog
                        .record_ref_for_chunk(&chunk.chunk_id)
                        .unwrap()
                        .payload
                        .offset,
                    inner
                        .catalog
                        .record_ref_for_embedding(&embedding.embedding_id)
                        .unwrap()
                        .payload
                        .offset,
                ]
            };
            let original = fs::read(&path).unwrap();
            let mut file = OpenOptions::new().write(true).open(&path).unwrap();
            for offset in offsets {
                file.seek(SeekFrom::Start(offset)).unwrap();
                file.write_all(&[original[offset as usize] ^ 1]).unwrap();
            }
            file.flush().unwrap();
            // Only the affected collection may be reread; old suffix offsets are already gone.
            collections[0]
                .insert_document(DocumentInput {
                    external_id: "new".into(),
                    title: None,
                    source_uri: None,
                    metadata: JsonValue::object(),
                    chunks: vec![ChunkInput {
                        chunk_external_id: "new".into(),
                        text: "new shared".into(),
                        ordinal: 0,
                        metadata: JsonValue::object(),
                        source_span: None,
                    }],
                })
                .unwrap();
            assert_eq!(
                db.index_stats().unwrap().rebuild_needed,
                budget == Some(0),
                "{name}"
            );
            if budget == Some(1) {
                let mut inner = db.lock().unwrap();
                let document = inner
                    .catalog
                    .documents
                    .values()
                    .find(|document| document.collection_id == collections[0].collection_id)
                    .unwrap()
                    .as_ref()
                    .clone();
                let mut prepared = crate::storage::PreparedRecords::create(
                    vec![RecordPayload::DocumentPut(document)],
                    path.parent().unwrap(),
                    1_048_576,
                )
                .unwrap();
                inner.persistent_vector_segments = None;
                inner.lazy_vector_segment = None;
                let pending = inner
                    .commit_prepared_batch(prepared.take_batch(1).unwrap())
                    .unwrap();
                inner
                    .finish_index_update(pending, &db.json_context())
                    .unwrap();
                assert_eq!(
                    inner.index_suffix_status,
                    IndexSuffixState::PersistentCheckpoint
                );
            }
            for offset in offsets {
                file.seek(SeekFrom::Start(offset)).unwrap();
                file.write_all(&[original[offset as usize]]).unwrap();
            }
            file.flush().unwrap();
            drop(file);
            {
                let mut inner = db.lock().unwrap();
                inner.config.cache_budget_mb = Some(1);
                inner.fail_next_index_checkpoint_refresh = true;
            }
            let new_document = collections[0].get_document_by_external_id("new").unwrap();
            collections[0]
                .delete_document(&new_document.document_id)
                .unwrap();
            {
                let inner = db.lock().unwrap();
                assert_eq!(inner.index_suffix_status, IndexSuffixState::RebuildNeeded);
                assert!(inner.persistent_vector_segments.is_none());
            }
            let fallback = collections[1]
                .search()
                .query_embedding(QueryEmbeddingInput {
                    model_name: None,
                    dim: 2,
                    vector: vec![1.0, 0.0],
                })
                .execute()
                .unwrap();
            assert_eq!(fallback.results[0].chunk_external_id, "untouched");
            assert_eq!(fallback.query_context.index_source, "transient_rebuild");
            db.put_embeddings_in_collection(
                &collections[0].collection_id,
                vec![EmbeddingInput {
                    chunk_external_id: "changed".into(),
                    model_name: "local".into(),
                    dim: 2,
                    vector: vec![0.0, 1.0],
                }],
                OnConflict::Replace,
            )
            .unwrap();
            assert!(!db.index_stats().unwrap().rebuild_needed);
            assert!(scan_file(&path).unwrap().findings.is_empty());
            drop(collections);
            drop(db);
            let db = Database::open(&path).unwrap();
            assert_eq!(db.stats().unwrap().chunk_count, 2);
            assert_eq!(db.stats().unwrap().embedding_count, 2);
            drop(db);
            fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn cold_vector_prefetch_is_optional_and_does_not_publish_before_commit() {
        let path = query_preflight_fixture("cold-vector-prefetch-fallback", "query text");
        let original = fs::read(&path).unwrap();
        let db = Database::open_read_only(&path).unwrap();
        {
            let mut inner = db.lock().unwrap();
            let id = inner
                .catalog
                .collection_by_name("first")
                .unwrap()
                .collection_id
                .clone();
            let document = inner
                .catalog
                .documents
                .values()
                .find(|document| document.collection_id == id)
                .unwrap()
                .as_ref()
                .clone();
            assert_eq!(
                inner
                    .prefetch_unchanged_vector_segments(&BTreeSet::new())
                    .len(),
                3
            );
            let commit = inner.last_commit_sequence_no;
            let error = inner
                .commit_records(vec![RecordPayload::DocumentPut(document.clone())])
                .err()
                .unwrap();
            assert_eq!(error.code(), ErrorCode::PermissionDenied);
            let mut prepared = crate::storage::PreparedRecords::create(
                vec![RecordPayload::DocumentPut(document)],
                path.parent().unwrap(),
                1_048_576,
            )
            .unwrap();
            let error = inner
                .commit_prepared_batch(prepared.take_batch(1).unwrap())
                .err()
                .unwrap();
            assert_eq!(error.code(), ErrorCode::PermissionDenied);
            assert_eq!(inner.last_commit_sequence_no, commit);
            assert_eq!(
                inner.index_suffix_status,
                IndexSuffixState::PersistentCheckpoint
            );
            assert!(inner.persistent_vector_segments.is_none());
            assert!(inner.lazy_vector_segment.is_none());
            assert_eq!(fs::read(&path).unwrap(), original);
            inner
                .catalog
                .collections
                .get_mut(&id)
                .unwrap()
                .embedding_dim += 1;
            assert!(!inner
                .prefetch_unchanged_vector_segments(&BTreeSet::new())
                .contains_key(&id));
            inner
                .catalog
                .collections
                .get_mut(&id)
                .unwrap()
                .embedding_dim -= 1;
            inner.index_suffix_status = IndexSuffixState::RebuildNeeded;
            assert!(inner
                .prefetch_unchanged_vector_segments(&BTreeSet::new())
                .is_empty());
        }
        drop(db);
        let db = Database::open(&path).unwrap();
        let record = {
            let inner = db.lock().unwrap();
            OpenOptions::new()
                .write(true)
                .open(&path)
                .unwrap()
                .set_len(inner.storage.critical_end_offset())
                .unwrap();
            assert!(inner
                .prefetch_unchanged_vector_segments(&BTreeSet::new())
                .is_empty());
            RecordPayload::DocumentPut(
                inner
                    .catalog
                    .documents
                    .values()
                    .next()
                    .unwrap()
                    .as_ref()
                    .clone(),
            )
        };
        let mut prepared = crate::storage::PreparedRecords::create(
            vec![record],
            path.parent().unwrap(),
            1_048_576,
        )
        .unwrap();
        {
            let mut inner = db.lock().unwrap();
            let pending = inner
                .commit_prepared_batch(prepared.take_batch(1).unwrap())
                .unwrap();
            inner
                .finish_index_update(pending, &db.json_context())
                .unwrap();
            assert_eq!(
                inner.index_suffix_status,
                IndexSuffixState::PersistentCheckpoint
            );
        }
        assert!(scan_file(&path).unwrap().findings.is_empty());
        drop(db);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn fixed_seed_mutations_match_full_text_and_vector_rebuilds() {
        let path = temp_path("fixed-seed-local-index-reuse");
        let db = Database::create(&path, DatabaseConfig::default()).unwrap();
        let mut collections = Vec::new();
        for name in ["first", "second", "third"] {
            collections.push(
                db.create_collection(CollectionConfig {
                    tokenizer: Default::default(),
                    name: name.into(),
                    embedding_dim: 2,
                    distance_metric: DistanceMetric::Cosine,
                    embedding_model: None,
                    vectors_normalized: false,
                    vector_index: VectorIndexConfig::Exact,
                })
                .unwrap(),
            );
        }
        let mut seed = 0x71a0_936b_54e2_c18du64;
        let mut historical_chunk_seen = false;
        for step in 0..100 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let collection = &collections[(seed >> 32) as usize % collections.len()];
            let external_id = format!("slot-{}", (seed >> 16) % 7);
            let chunk = collection.get_chunk_by_external_id(&external_id).ok();
            match ((seed >> 48) % 4, chunk) {
                (1, Some(_)) => {
                    db.put_embeddings_in_collection(
                        &collection.collection_id,
                        vec![EmbeddingInput {
                            chunk_external_id: external_id,
                            model_name: "local".into(),
                            dim: 2,
                            vector: vec![1.0, (step % 3) as f32],
                        }],
                        OnConflict::Replace,
                    )
                    .unwrap();
                }
                (2, Some(chunk)) => collection.delete_chunk(&chunk.chunk_id).unwrap(),
                (3, Some(chunk)) => collection.delete_document(&chunk.document_id).unwrap(),
                _ => {
                    db.insert_document_in_collection(
                        &collection.collection_id,
                        DocumentInput {
                            external_id: external_id.clone(),
                            title: None,
                            source_uri: None,
                            metadata: object_from_pairs([("step", JsonValue::Number(step as f64))]),
                            chunks: vec![ChunkInput {
                                chunk_external_id: external_id,
                                text: if step % 9 == 0 {
                                    "!!!".into()
                                } else {
                                    format!("shared token{} shared", step % 3)
                                },
                                ordinal: 0,
                                metadata: JsonValue::object(),
                                source_span: None,
                            }],
                        },
                        OnConflict::Replace,
                    )
                    .unwrap();
                }
            }
            let inner = db.lock().unwrap();
            assert_eq!(
                inner.index_suffix_status,
                IndexSuffixState::PersistentCheckpoint,
                "step {step}"
            );
            let full = build_text_index(
                &inner.storage,
                &inner.catalog,
                &inner.temporary_work,
                &db.json_context(),
            )
            .unwrap();
            assert_eq!(
                inner.text_index.chunk_lengths, full.chunk_lengths,
                "step {step}"
            );
            assert_eq!(
                inner.text_index.collection_counts(),
                full.collection_counts(),
                "step {step}"
            );
            assert_eq!(
                inner.text_index.avg_len_by_collection, full.avg_len_by_collection,
                "step {step}"
            );
            for chunk in inner.catalog.chunks.values() {
                if !inner.catalog.is_chunk_live(&chunk.chunk_id) {
                    historical_chunk_seen = true;
                    assert!(!inner.text_index.chunk_lengths.contains_key(&chunk.chunk_id));
                }
            }
            for collection in &collections {
                for query in ["shared", "token0 token1 token2", "absent"] {
                    assert_eq!(
                        format!(
                            "{:?}",
                            inner
                                .text_index
                                .search(
                                    &collection.collection_id,
                                    query,
                                    TextSearchOptions::with_explanations(20),
                                    &db.json_context()
                                )
                                .unwrap()
                                .as_slice()
                        ),
                        format!(
                            "{:?}",
                            full.search(
                                &collection.collection_id,
                                query,
                                TextSearchOptions::with_explanations(20),
                                &db.json_context()
                            )
                            .unwrap()
                            .as_slice()
                        ),
                        "step {step}"
                    );
                }
            }
            for segment in inner
                .vector_index_segments(BTreeMap::new(), &BTreeSet::new())
                .unwrap()
            {
                assert_eq!(
                    format!("{segment:?}"),
                    format!(
                        "{:?}",
                        inner.persistent_vector_segments.as_ref().unwrap()[&segment.collection_id]
                    ),
                    "step {step}"
                );
            }
        }
        assert!(historical_chunk_seen);
        let before = db.stats().unwrap();
        let (lengths, totals, averages, signatures) = {
            let inner = db.lock().unwrap();
            let mut signatures = Vec::new();
            for collection in &collections {
                for query in ["shared", "token0 token1 token2", "absent"] {
                    signatures.push((
                        collection.collection_id.clone(),
                        query,
                        format!(
                            "{:?}",
                            inner
                                .text_index
                                .search(
                                    &collection.collection_id,
                                    query,
                                    TextSearchOptions::with_explanations(20),
                                    &db.json_context(),
                                )
                                .unwrap()
                                .as_slice()
                        ),
                    ));
                }
            }
            (
                inner.text_index.chunk_lengths.clone(),
                inner.text_index.collection_counts(),
                inner.text_index.avg_len_by_collection.clone(),
                signatures,
            )
        };
        assert!(scan_file(&path).unwrap().findings.is_empty());
        drop(collections);
        drop(db);
        let db = Database::open(&path).unwrap();
        assert_eq!(db.stats().unwrap().chunk_count, before.chunk_count);
        assert_eq!(db.stats().unwrap().embedding_count, before.embedding_count);
        assert!(!db.index_stats().unwrap().rebuild_needed);
        {
            let inner = db.lock().unwrap();
            assert_eq!(inner.text_index.chunk_lengths, lengths);
            assert_eq!(inner.text_index.collection_counts(), totals);
            assert_eq!(inner.text_index.avg_len_by_collection, averages);
            for (id, query, expected) in signatures {
                assert_eq!(
                    format!(
                        "{:?}",
                        inner
                            .text_index
                            .search(
                                &id,
                                query,
                                TextSearchOptions::with_explanations(20),
                                &db.json_context(),
                            )
                            .unwrap()
                            .as_slice()
                    ),
                    expected,
                );
            }
        }
        drop(db);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn committed_catalog_invalidates_old_checkpoint_before_text_read_failure() {
        let path = temp_path("commit-before-text-failure");
        let db = Database::create(&path, DatabaseConfig::default()).unwrap();
        let col = db
            .create_collection(CollectionConfig {
                tokenizer: Default::default(),
                name: "notes".into(),
                embedding_dim: 2,
                distance_metric: DistanceMetric::Cosine,
                embedding_model: None,
                vectors_normalized: false,
                vector_index: VectorIndexConfig::Exact,
            })
            .unwrap();
        let input = |id: &str, text: &str| DocumentInput {
            external_id: id.into(),
            title: None,
            source_uri: None,
            metadata: JsonValue::object(),
            chunks: vec![ChunkInput {
                chunk_external_id: id.into(),
                text: text.into(),
                ordinal: 0,
                metadata: JsonValue::object(),
                source_span: None,
            }],
        };
        col.insert_document(input("old", "previous snapshot"))
            .unwrap();
        let (offset, previous_commit) = {
            let inner = db.lock().unwrap();
            let chunk = inner
                .catalog
                .chunk_by_external_id(&col.collection_id, "old")
                .unwrap();
            (
                inner
                    .catalog
                    .record_ref_for_chunk(&chunk.chunk_id)
                    .unwrap()
                    .payload
                    .offset,
                inner.catalog.last_commit_id,
            )
        };
        let original_byte = fs::read(&path).unwrap()[offset as usize];
        let mut file = OpenOptions::new().write(true).open(&path).unwrap();
        file.seek(SeekFrom::Start(offset)).unwrap();
        file.write_all(&[original_byte ^ 1]).unwrap();
        file.flush().unwrap();
        let err = col
            .insert_document(input("new", "newlycommitted searchable"))
            .unwrap_err();
        assert_eq!(err.code(), ErrorCode::ChecksumMismatch);
        {
            let inner = db.lock().unwrap();
            assert_eq!(inner.catalog.last_commit_id, previous_commit + 1);
            assert_eq!(inner.index_suffix_status, IndexSuffixState::RebuildNeeded);
            assert!(inner.persistent_vector_segments.is_none());
            assert!(inner
                .catalog
                .chunk_by_external_id(&col.collection_id, "new")
                .is_ok());
        }
        file.seek(SeekFrom::Start(offset)).unwrap();
        file.write_all(&[original_byte]).unwrap();
        file.flush().unwrap();
        drop(file);
        let result = col.search().text("newlycommitted").execute().unwrap();
        assert_eq!(result.results.len(), 1);
        assert_eq!(result.results[0].chunk_external_id, "new");
        assert_eq!(result.query_context.index_source, "transient_rebuild");
        drop(col);
        drop(db);
        let reopened = Database::open(&path).unwrap();
        assert_eq!(reopened.stats().unwrap().document_count, 2);
        assert!(reopened.index_stats().unwrap().rebuild_needed);
        drop(reopened);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn compact_checkpoint_failure_clears_persistent_index_state() {
        let path = temp_path("compact-checkpoint-failure-clears-state");
        let db = Database::create(&path, DatabaseConfig::default()).unwrap();
        let col = db
            .create_collection(CollectionConfig {
                tokenizer: Default::default(),
                name: "notes".into(),
                embedding_dim: 2,
                distance_metric: DistanceMetric::Cosine,
                embedding_model: None,
                vectors_normalized: false,
                vector_index: VectorIndexConfig::Exact,
            })
            .unwrap();
        col.insert_document(DocumentInput {
            external_id: "doc".into(),
            title: None,
            source_uri: None,
            metadata: JsonValue::object(),
            chunks: vec![ChunkInput {
                chunk_external_id: "doc#1".into(),
                text: "compact rebuild failure clears stale persistent state".into(),
                ordinal: 0,
                metadata: JsonValue::object(),
                source_span: None,
            }],
        })
        .unwrap();
        drop(col);
        let stats_before = db.index_stats().unwrap();
        assert!(!stats_before.rebuild_needed);
        assert!(stats_before.last_index_commit_id.is_some());

        fail_next_index_checkpoint_refresh(&db);
        let summary = db
            .compact(CompactOptions {
                rebuild_indexes: true,
            })
            .unwrap();
        assert_eq!(summary.index_rebuild, "failed");
        assert!(summary
            .index_rebuild_reason
            .as_deref()
            .unwrap_or_default()
            .contains("injected index checkpoint refresh failure"));
        let stats_after = db.index_stats().unwrap();
        assert!(stats_after.rebuild_needed);
        assert_eq!(stats_after.last_index_commit_id, None);
        drop(db);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn compact_reports_discarded_persistent_segment_records() {
        let path = temp_path("compact-segments-merged");
        let db = Database::create(&path, DatabaseConfig::default()).unwrap();
        db.create_collection(CollectionConfig {
            tokenizer: Default::default(),
            name: "notes".into(),
            embedding_dim: 2,
            distance_metric: DistanceMetric::Cosine,
            embedding_model: None,
            vectors_normalized: false,
            vector_index: VectorIndexConfig::Exact,
        })
        .unwrap();
        assert_eq!(db.index_stats().unwrap().text_segments, 1);
        assert_eq!(db.index_stats().unwrap().vector_segments, 1);

        let summary = db
            .compact(CompactOptions {
                rebuild_indexes: true,
            })
            .unwrap();
        assert_eq!(summary.segments_merged, 2);
        assert_eq!(
            summary.to_json().get("segments_merged"),
            Some(&JsonValue::Number(2.0))
        );

        drop(db);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn coverage_hash_admission_shares_original_context_before_counting() {
        let path = cached_vector_batch_fixture("coverage-hash-admission");
        let db = Database::open(&path).unwrap();
        let context = db.json_context();
        let inner = db.lock().unwrap();
        let before = fs::read(&path).unwrap();
        let required = inner.catalog.live_collections().count()
            * (8 + std::mem::size_of::<&CollectionInfo>() + 3 * std::mem::size_of::<usize>());
        let mut held = context.workspace_reservation();
        let occupied = context.available_workspace_bytes() - required + 1;
        held.grow_to(occupied, occupied, "operation_scratch")
            .unwrap();
        let error = catalog_index_coverage_hash(
            &inner.catalog,
            inner.storage.critical_end_offset(),
            inner.last_commit_sequence_no,
            &context,
        )
        .unwrap_err();
        assert_eq!(error.code(), ErrorCode::ResourceExhausted);
        assert_eq!(error_detail_value(&error, "resource"), Some("memory"));
        assert_eq!(
            error_detail_value(&error, "operation"),
            Some("operation_scratch")
        );
        assert_eq!(fs::read(&path).unwrap(), before);
        drop((held, inner));
        drop(db);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn coverage_budget_refusal_is_not_index_damage_or_a_successful_check() {
        let path = cached_vector_batch_fixture("coverage-check-refusal");
        let db = Database::open(&path).unwrap();
        db.lock().unwrap().text_index.clear_cache();
        let context = db.json_context();
        let before = fs::read(&path).unwrap();
        let required = {
            let inner = db.lock().unwrap();
            let required = inner.catalog.live_collections().count()
                * (8 + std::mem::size_of::<&CollectionInfo>() + 3 * std::mem::size_of::<usize>());
            let baseline = scan_file(&path).unwrap();
            let mut held = context.workspace_reservation();
            let occupied = context.available_workspace_bytes() - required + 1;
            held.grow_to(occupied, occupied, "operation_scratch")
                .unwrap();
            let error =
                index_suffix_status_from_scan(&baseline, &inner.catalog, &context).unwrap_err();
            assert_eq!(error.code(), ErrorCode::ResourceExhausted);
            assert_eq!(
                error_detail_value(&error, "operation"),
                Some("operation_scratch")
            );
            required
        };
        let original_stats = db.index_stats().unwrap().to_json();
        for verify_indexes in [false, true] {
            for available in [required - 1, required, required + 1] {
                COVERAGE_HASH_CALLS.set(0);
                CHECK_SCANS.set(0);
                let result = with_check_coverage_pressure(
                    CheckCoveragePressure::WorkspaceHeadroom(available),
                    || db.check(CheckOptions { verify_indexes }),
                );
                if available < required {
                    let error = result.unwrap_err();
                    assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                    assert_eq!(error_detail_value(&error, "resource"), Some("memory"));
                    assert_eq!(
                        error_detail_value(&error, "operation"),
                        Some("operation_scratch")
                    );
                } else {
                    let report = result.unwrap();
                    assert_eq!(report.status, "ok");
                    assert_eq!(report.index_suffix_status, "persistent_checkpoint");
                    assert!(report.issues.is_empty());
                }
                assert_eq!(COVERAGE_HASH_CALLS.get(), 1);
                assert_eq!(CHECK_SCANS.get(), 1);
                assert_eq!(db.index_stats().unwrap().to_json(), original_stats);
                assert_eq!(fs::read(&path).unwrap(), before);
            }
        }
        assert_eq!(
            db.check(CheckOptions {
                verify_indexes: true
            })
            .unwrap()
            .status,
            "ok"
        );
        drop(db);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn coverage_check_reclaims_only_optional_cache_without_rescanning() {
        let path = cached_vector_batch_fixture("coverage-check-cache");
        for lazy in [false, true] {
            for insufficient in [false, true] {
                let db = Database::open_read_only(&path).unwrap();
                let context = db.json_context();
                let before = fs::read(&path).unwrap();
                let (required, catalog, source) = {
                    let mut inner = db.lock().unwrap();
                    if lazy {
                        let id = inner
                            .catalog
                            .collection_by_name("cached")
                            .unwrap()
                            .collection_id
                            .clone();
                        let source = inner.lazy_vector_source.as_ref().unwrap();
                        let reservation = CachedVectorSegment::reserve(
                            &context,
                            usize::MAX,
                            source.encoded_len(&id).unwrap(),
                        )
                        .unwrap();
                        inner.lazy_vector_segment = source
                            .load_cached(&inner.storage, &id, reservation, &context)
                            .unwrap();
                    } else {
                        let cached = inner.prefetch_unchanged_vector_segments(&BTreeSet::new());
                        inner.persistent_vector_segments = Some(cached);
                    }
                    let cache = inner
                        .persistent_vector_segments
                        .as_ref()
                        .map_or(0, VectorCache::reserved_bytes)
                        + inner
                            .lazy_vector_segment
                            .as_ref()
                            .map_or(0, CachedVectorSegment::reserved_bytes);
                    assert!(cache > 0);
                    (
                        inner.catalog.live_collections().count()
                            * (8 + std::mem::size_of::<&CollectionInfo>()
                                + 3 * std::mem::size_of::<usize>()),
                        inner.catalog.reserved_bytes(),
                        inner.lazy_vector_source.as_ref().unwrap().reserved_bytes(),
                    )
                };
                // The second Catalog belongs to check's scan and remains live during hashing.
                let occupied = context.available_total_bytes() - catalog - required + 1;
                let pressure = if insufficient {
                    CheckCoveragePressure::WorkspaceHeadroom(required - 1)
                } else {
                    CheckCoveragePressure::CacheBytes(occupied)
                };
                COVERAGE_HASH_CALLS.set(0);
                CHECK_SCANS.set(0);
                let result = with_check_coverage_pressure(pressure, || {
                    db.check(CheckOptions {
                        verify_indexes: true,
                    })
                });
                if insufficient {
                    let error = result.unwrap_err();
                    assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                    assert_eq!(error_detail_value(&error, "resource"), Some("memory"));
                } else {
                    let report = result.unwrap();
                    assert_eq!(report.status, "ok");
                    assert!(report.issues.is_empty());
                }
                assert_eq!(COVERAGE_HASH_CALLS.get(), 2);
                assert_eq!(CHECK_SCANS.get(), 1);
                let inner = db.lock().unwrap();
                assert!(inner.persistent_vector_segments.is_none());
                assert!(inner.lazy_vector_segment.is_none());
                assert_eq!(
                    inner.lazy_vector_source.as_ref().unwrap().reserved_bytes(),
                    source
                );
                assert!(!inner.index_stats().rebuild_needed);
                drop(inner);
                assert_eq!(fs::read(&path).unwrap(), before);
                let mut free = context.workspace_reservation();
                let available = crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES - source;
                free.grow_to(available, available, "operation_scratch")
                    .unwrap();
                assert!(free
                    .grow_to(available + 1, available + 1, "operation_scratch")
                    .is_err());
                drop(free);
                let parent = context
                    .cache_reservation(context.available_total_bytes() as u128, usize::MAX)
                    .unwrap();
                assert!(context.cache_reservation(1, usize::MAX).is_none());
                drop(parent);
                assert_eq!(
                    db.check(CheckOptions {
                        verify_indexes: true
                    })
                    .unwrap()
                    .status,
                    "ok"
                );
                drop(db);
            }
        }
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn streaming_coverage_hash_preserves_wire_bytes_and_visible_counts() {
        let path = query_preflight_fixture("coverage-streaming", "needle");
        let db = Database::open(&path).unwrap();
        let mut inner = db.lock().unwrap();
        let catalog = &mut inner.catalog;
        let ids = ["first", "second", "empty"].map(|name| {
            catalog
                .collection_by_name(name)
                .unwrap()
                .collection_id
                .clone()
        });
        let document = catalog
            .document_by_external_id(&ids[1], "doc")
            .unwrap()
            .document_id
            .clone();
        let chunk = catalog
            .chunk_by_external_id(&ids[0], "chunk-1")
            .unwrap()
            .chunk_id
            .clone();
        let embedding = catalog.chunk_embeddings[&chunk].clone();
        let originally_deleted = catalog.deleted_chunks.clone();
        for seed in 0..100 {
            catalog.deleted_collections.clear();
            catalog.deleted_documents.clear();
            catalog.deleted_chunks = originally_deleted.clone();
            catalog
                .chunk_embeddings
                .insert(chunk.clone(), embedding.clone());
            let mut counts = [[1u64, 2, 2], [1, 4, 4], [0, 0, 0]];
            match seed % 4 {
                0 => {
                    catalog.deleted_collections.insert(ids[0].clone());
                }
                1 => {
                    catalog.deleted_documents.insert(document.clone());
                    counts[1] = [0, 0, 0];
                }
                2 => {
                    catalog.deleted_chunks.insert(chunk.clone());
                    counts[0] = [1, 1, 1];
                }
                _ => {
                    catalog.chunk_embeddings.remove(&chunk);
                    counts[0] = [1, 2, 1];
                }
            }
            let end = u64::MAX - seed as u64;
            let commit = (1u64 << 60) + seed as u64;
            let mut expected = Vec::new();
            expected.extend_from_slice(&1u16.to_le_bytes());
            expected.extend_from_slice(&end.to_le_bytes());
            expected.extend_from_slice(&commit.to_le_bytes());
            expected.extend_from_slice(&(if seed % 4 == 0 { 2u64 } else { 3 }).to_le_bytes());
            for (index, id) in ids.iter().enumerate() {
                let collection = catalog.collections.get_mut(id).unwrap();
                collection.name = format!("name-{index}-{seed}-\u{4e2d}");
                collection.embedding_dim = [1, 768, u32::MAX as usize][(seed + index) % 3];
                collection.distance_metric = [
                    DistanceMetric::Cosine,
                    DistanceMetric::DotProduct,
                    DistanceMetric::Euclidean,
                ][(seed + index) % 3];
                collection.vectors_normalized = (seed + index) % 2 == 0;
                collection.embedding_model = match (seed + index) % 3 {
                    0 => None,
                    1 => Some(String::new()),
                    _ => Some("\u{3bb}".repeat(if seed == 99 { 32768 } else { seed + 1 })),
                };
                if seed % 4 == 0 && index == 0 {
                    continue;
                }
                expected.extend_from_slice(&(index as u64 + 1).to_le_bytes());
                expected.extend_from_slice(&(collection.name.len() as u32).to_le_bytes());
                expected.extend_from_slice(collection.name.as_bytes());
                expected.extend_from_slice(&(collection.embedding_dim as u32).to_le_bytes());
                expected.push(((seed + index) % 3 + 1) as u8);
                if let Some(model) = &collection.embedding_model {
                    expected.push(1);
                    expected.extend_from_slice(&(model.len() as u32).to_le_bytes());
                    expected.extend_from_slice(model.as_bytes());
                } else {
                    expected.push(0);
                }
                expected.push(u8::from(collection.vectors_normalized));
                for count in counts[index] {
                    expected.extend_from_slice(&count.to_le_bytes());
                }
                expected.extend_from_slice(&[1, 0, 2, 0, 1, 1]);
            }
            let mut crc = u32::MAX;
            for byte in expected {
                crc ^= u32::from(byte);
                for _ in 0..8 {
                    crc = (crc >> 1) ^ (0x82f6_3b78 & 0u32.wrapping_sub(crc & 1));
                }
            }
            let context = db.json_context();
            let required = catalog.live_collections().count()
                * (8 + std::mem::size_of::<&CollectionInfo>() + 3 * std::mem::size_of::<usize>());
            assert!(context.available_workspace_bytes() >= required);
            let (actual, allocated) = crate::test_alloc::allocation_bytes(|| {
                catalog_index_coverage_hash(catalog, end, commit, &context).unwrap()
            });
            assert_eq!(actual, !crc, "seed {seed}");
            assert_eq!(allocated, required as u128, "seed {seed}");
        }
        catalog.collections.get_mut(&ids[1]).unwrap().embedding_dim = u32::MAX as usize + 1;
        assert_eq!(
            catalog_index_coverage_hash(catalog, 0, 0, &db.json_context())
                .unwrap_err()
                .code(),
            ErrorCode::InvalidEmbeddingDimension
        );
        drop(inner);
        drop(db);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn coverage_hash_orders_collection_ids_numerically() {
        let mut catalog = Catalog::default();
        let ids = [9_999_999_999_999_999u64, 10_000_000_000_000_000, u64::MAX];
        for id in ids {
            let collection = CollectionInfo {
                tokenizer: Default::default(),
                collection_id: format!("col_{id:016}"),
                name: "x".into(),
                embedding_dim: 2,
                distance_metric: DistanceMetric::Cosine,
                embedding_model: None,
                vectors_normalized: false,
                vector_index: VectorIndexConfig::Exact,
                created_at: "1970-01-01T00:00:00Z".into(),
            };
            catalog
                .collections
                .insert(collection.collection_id.clone(), collection);
        }
        assert_eq!(
            catalog.live_collections().next().unwrap().collection_id,
            format!("col_{:016}", ids[1])
        );
        let mut expected = Vec::new();
        expected.extend_from_slice(&1u16.to_le_bytes());
        expected.extend_from_slice(&58u64.to_le_bytes());
        expected.extend_from_slice(&7u64.to_le_bytes());
        expected.extend_from_slice(&(ids.len() as u64).to_le_bytes());
        for id in ids {
            expected.extend_from_slice(&id.to_le_bytes());
            expected.extend_from_slice(&1u32.to_le_bytes());
            expected.push(b'x');
            expected.extend_from_slice(&2u32.to_le_bytes());
            expected.extend_from_slice(&[1, 0, 0]);
            expected.extend_from_slice(&[0; 24]);
            expected.extend_from_slice(&[1, 0, 2, 0, 1, 1]);
        }
        let mut crc = u32::MAX;
        for byte in expected {
            crc ^= u32::from(byte);
            for _ in 0..8 {
                crc = (crc >> 1) ^ (0x82f6_3b78 & 0u32.wrapping_sub(crc & 1));
            }
        }
        let context = crate::json::JsonOperationContext::standalone();
        assert_eq!(
            catalog_index_coverage_hash(&catalog, 58, 7, &context).unwrap(),
            !crc
        );
        let mut collection = catalog.collections.pop_first().unwrap().1;
        collection.collection_id = "bad".into();
        catalog.collections.insert("bad".into(), collection);
        assert_eq!(
            catalog_index_coverage_hash(&catalog, 58, 7, &context)
                .unwrap_err()
                .code(),
            ErrorCode::FileFormatError
        );
    }

    #[test]
    fn checkpoint_with_wrong_coverage_hash_is_rebuild_index_only() {
        let path = temp_path("wrong-index-coverage");
        let db = Database::create(&path, DatabaseConfig::default()).unwrap();
        db.create_collection(CollectionConfig {
            tokenizer: Default::default(),
            name: "notes".into(),
            embedding_dim: 2,
            distance_metric: DistanceMetric::Cosine,
            embedding_model: None,
            vectors_normalized: false,
            vector_index: VectorIndexConfig::Exact,
        })
        .unwrap();
        drop(db);

        let (mut storage, scan) = Storage::open(&path).unwrap();
        storage
            .write_index_checkpoint(
                &IndexCheckpointPayload {
                    metadata: IndexCheckpointMetadata {
                        covered_last_batch_id: scan.last_valid_commit,
                        covered_last_commit_sequence_no: scan.last_valid_commit_sequence_no,
                        coverage_hash: u32::MAX,
                        collection_count: 1,
                        live_document_count: 0,
                        live_chunk_count: 0,
                        live_embedding_count: 0,
                        text_segment_count: 0,
                        vector_segment_count: 0,
                        build_mode: "metadata_record_only".into(),
                    },
                    text_segments: Vec::new(),
                    vector_segments: Vec::new(),
                },
                Durability::Commit,
            )
            .unwrap();
        drop(storage);

        let db = Database::open(&path).unwrap();
        assert_eq!(db.list_collections().unwrap().len(), 1);
        drop(db);

        let report = Database::check_path(
            &path,
            CheckOptions {
                verify_indexes: true,
            },
        )
        .unwrap();
        assert_eq!(report.status, "warning");
        assert_eq!(report.index_suffix_status, "rebuild_needed");
        assert_eq!(report.index_check, "rebuild_needed");
        assert!(report.issues.iter().any(|issue| {
            issue.code == "index_coverage_hash_mismatch"
                && issue.stream == "index"
                && issue.visibility_impact == "rebuild_index_only"
        }));
        assert!(!report
            .issues
            .iter()
            .any(|issue| issue.visibility_impact == "blocks_main"));
        let reclaims = Cell::new(0);
        COVERAGE_HASH_CALLS.set(0);
        let same = build_check_report_reclaiming(
            &path,
            CheckOptions {
                verify_indexes: true,
            },
            &crate::json::JsonOperationContext::standalone(),
            || {
                reclaims.set(reclaims.get() + 1);
                true
            },
        )
        .unwrap();
        assert_eq!(same.to_json(), report.to_json());
        assert_eq!(reclaims.get(), 0);
        assert_eq!(COVERAGE_HASH_CALLS.get(), 1);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn open_and_check_preserve_main_data_when_checkpoint_length_overflows() {
        let path = temp_path("checkpoint-length-overflow");
        let db = Database::create(&path, DatabaseConfig::default()).unwrap();
        db.create_collection(CollectionConfig {
            tokenizer: Default::default(),
            name: "notes".into(),
            embedding_dim: 2,
            distance_metric: DistanceMetric::Cosine,
            embedding_model: None,
            vectors_normalized: false,
            vector_index: VectorIndexConfig::Exact,
        })
        .unwrap();
        drop(db);

        let scan = scan_file(&path).unwrap();
        append_overflowing_checkpoint_payload_suffix(&path, &scan);

        let db = Database::open(&path).unwrap();
        assert_eq!(db.list_collections().unwrap().len(), 1);
        drop(db);

        let report = Database::check_path(
            &path,
            CheckOptions {
                verify_indexes: true,
            },
        )
        .unwrap();
        assert_eq!(report.status, "warning");
        assert_eq!(report.index_suffix_status, "rebuild_needed");
        assert!(report.issues.iter().any(|issue| {
            issue.code == "index_checkpoint_length_overflow"
                && issue.stream == "index"
                && issue.visibility_impact == "rebuild_index_only"
        }));
        assert!(!report
            .issues
            .iter()
            .any(|issue| issue.visibility_impact == "blocks_main"));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn reopened_catalog_keeps_large_payloads_lazy_but_materializes_on_demand() {
        let path = temp_path("lazy-payloads");
        let db = Database::create(&path, DatabaseConfig::default()).unwrap();
        let col = db
            .create_collection(CollectionConfig {
                tokenizer: Default::default(),
                name: "notes".into(),
                embedding_dim: 2,
                distance_metric: DistanceMetric::Cosine,
                embedding_model: Some("m".into()),
                vectors_normalized: false,
                vector_index: VectorIndexConfig::Exact,
            })
            .unwrap();
        col.insert_document(DocumentInput {
            external_id: "doc".into(),
            title: None,
            source_uri: None,
            metadata: parse("{}").unwrap(),
            chunks: vec![ChunkInput {
                chunk_external_id: "doc#1".into(),
                text: "lazy payload search keeps catalog small".into(),
                ordinal: 1,
                metadata: parse(r#"{"lang":"rust"}"#).unwrap(),
                source_span: None,
            }],
        })
        .unwrap();
        col.put_embeddings(vec![EmbeddingInput {
            chunk_external_id: "doc#1".into(),
            model_name: "m".into(),
            dim: 2,
            vector: vec![1.0, 0.0],
        }])
        .unwrap();
        drop(col);
        drop(db);

        let db = Database::open(&path).unwrap();
        {
            let inner = db.lock().unwrap();
            let chunk_summary = inner
                .catalog
                .chunks
                .values()
                .find(|chunk| chunk.chunk_external_id == "doc#1")
                .unwrap();
            assert!(chunk_summary.text.is_empty());
            let embedding_summary = inner.catalog.live_embeddings().next().unwrap();
            assert!(embedding_summary.vector.is_empty());
            assert!(inner
                .catalog
                .record_ref_for_chunk(&chunk_summary.chunk_id)
                .is_some());
            assert!(inner
                .catalog
                .record_ref_for_embedding(&embedding_summary.embedding_id)
                .is_some());
        }

        let col = db.collection("notes").unwrap();
        let chunk = col.get_chunk_by_external_id("doc#1").unwrap();
        assert_eq!(chunk.text, "lazy payload search keeps catalog small");
        let results = col
            .search()
            .text("lazy search")
            .query_embedding(QueryEmbeddingInput {
                model_name: Some("m".into()),
                dim: 2,
                vector: vec![1.0, 0.0],
            })
            .filter(parse(r#"{"chunk.lang":"rust"}"#).unwrap())
            .execute()
            .unwrap();
        assert_eq!(results.results.len(), 1);
        assert_eq!(results.results[0].chunk_external_id, "doc#1");

        db.compact(CompactOptions {
            rebuild_indexes: true,
        })
        .unwrap();
        drop(col);
        drop(db);

        let db = Database::open(&path).unwrap();
        let col = db.collection("notes").unwrap();
        assert_eq!(
            col.get_chunk_by_external_id("doc#1").unwrap().text,
            "lazy payload search keeps catalog small"
        );
        let results = col
            .search()
            .query_embedding(QueryEmbeddingInput {
                model_name: Some("m".into()),
                dim: 2,
                vector: vec![1.0, 0.0],
            })
            .execute()
            .unwrap();
        assert_eq!(results.results.len(), 1);
        drop(col);
        drop(db);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn insert_reopen_and_search_hybrid() {
        let path = temp_path("search");
        let db = Database::create(&path, DatabaseConfig::default()).unwrap();
        let col = db
            .create_collection(CollectionConfig {
                tokenizer: Default::default(),
                name: "notes".into(),
                embedding_dim: 2,
                distance_metric: DistanceMetric::Cosine,
                embedding_model: Some("test-model".into()),
                vectors_normalized: false,
                vector_index: VectorIndexConfig::Exact,
            })
            .unwrap();
        col.insert_document(DocumentInput {
            external_id: "doc-1".into(),
            title: Some("WAL".into()),
            source_uri: Some("docs/wal.md".into()),
            metadata: parse(r#"{"path":"docs/wal.md","source":"markdown"}"#).unwrap(),
            chunks: vec![ChunkInput {
                chunk_external_id: "doc-1#1".into(),
                text: "Write ahead log recovery keeps committed records visible".into(),
                ordinal: 1,
                metadata: parse(r#"{"lang":"rust"}"#).unwrap(),
                source_span: None,
            }],
        })
        .unwrap();
        col.put_embeddings(vec![EmbeddingInput {
            chunk_external_id: "doc-1#1".into(),
            model_name: "test-model".into(),
            dim: 2,
            vector: vec![1.0, 0.0],
        }])
        .unwrap();
        drop(col);
        drop(db);

        let db = Database::open(&path).unwrap();
        let results = db
            .collection("notes")
            .unwrap()
            .search()
            .text("wal recovery")
            .query_embedding(QueryEmbeddingInput {
                model_name: Some("test-model".into()),
                dim: 2,
                vector: vec![1.0, 0.0],
            })
            .filter(parse(r#"{"chunk.lang":"rust"}"#).unwrap())
            .top_k(5)
            .execute()
            .unwrap();
        assert_eq!(results.results.len(), 1);
        assert_eq!(results.results[0].chunk_external_id, "doc-1#1");
        let results_json = results.to_json();
        let result_item = results_json
            .get("results")
            .and_then(JsonValue::as_array)
            .and_then(|items| items.first())
            .and_then(JsonValue::as_object)
            .unwrap();
        let document_metadata = result_item
            .get("document_metadata")
            .and_then(JsonValue::as_object)
            .unwrap();
        assert!(!document_metadata.contains_key("source_uri"));
        let explain_source = result_item
            .get("explain")
            .and_then(|explain| explain.get("source"))
            .and_then(JsonValue::as_object)
            .unwrap();
        assert_eq!(
            explain_source.get("source_uri").and_then(JsonValue::as_str),
            Some("docs/wal.md")
        );
        let query_context = results_json
            .get("query_context")
            .and_then(JsonValue::as_object)
            .unwrap();
        let top_execution_stats = results_json
            .get("execution_stats")
            .and_then(JsonValue::as_object)
            .unwrap();
        assert_eq!(
            top_execution_stats
                .get("vector_scan_count")
                .and_then(JsonValue::as_f64),
            Some(1.0)
        );
        assert_eq!(
            query_context
                .get("text_present")
                .and_then(JsonValue::as_bool),
            Some(true)
        );
        assert_eq!(
            query_context
                .get("embedding_present")
                .and_then(JsonValue::as_bool),
            Some(true)
        );
        assert_eq!(
            query_context
                .get("filter_present")
                .and_then(JsonValue::as_bool),
            Some(true)
        );
        assert_eq!(
            query_context
                .get("index_source")
                .and_then(JsonValue::as_str),
            Some("persistent_checkpoint")
        );
        assert!(matches!(
            query_context.get("index_rebuild_reason"),
            Some(JsonValue::Null)
        ));
        let transient_rebuild = query_context
            .get("transient_rebuild")
            .and_then(JsonValue::as_object)
            .unwrap();
        assert_eq!(
            transient_rebuild
                .get("started")
                .and_then(JsonValue::as_bool),
            Some(false)
        );
        assert!(matches!(
            transient_rebuild.get("chunks_scanned"),
            Some(JsonValue::Null)
        ));
        assert!(matches!(
            transient_rebuild.get("bytes_read"),
            Some(JsonValue::Null)
        ));
        let query_execution_stats = query_context
            .get("execution_stats")
            .and_then(JsonValue::as_object)
            .unwrap();
        assert_eq!(
            query_execution_stats
                .get("vector_scan_count")
                .and_then(JsonValue::as_f64),
            Some(1.0)
        );
        assert_eq!(
            query_execution_stats
                .get("distance_ops")
                .and_then(JsonValue::as_f64),
            Some(2.0)
        );
        assert_eq!(
            query_execution_stats
                .get("fusion_candidate_count")
                .and_then(JsonValue::as_f64),
            Some(1.0)
        );
        assert_eq!(
            results.results[0]
                .explain
                .get("document_external_id")
                .and_then(JsonValue::as_str),
            Some("doc-1")
        );
        assert_eq!(
            results.results[0]
                .explain
                .get("index_source")
                .and_then(JsonValue::as_str),
            Some("persistent_checkpoint")
        );
        assert!(matches!(
            results.results[0].explain.get("index_rebuild_reason"),
            Some(JsonValue::Null)
        ));
        assert_eq!(
            results.results[0]
                .explain
                .get("transient_rebuild")
                .and_then(JsonValue::as_object)
                .and_then(|context| context.get("started"))
                .and_then(JsonValue::as_bool),
            Some(false)
        );
        let execution_stats = results.results[0]
            .explain
            .get("execution_stats")
            .and_then(JsonValue::as_object)
            .unwrap();
        assert_eq!(
            execution_stats
                .get("vector_scan_count")
                .and_then(JsonValue::as_f64),
            Some(1.0)
        );
        assert_eq!(
            execution_stats
                .get("distance_ops")
                .and_then(JsonValue::as_f64),
            Some(2.0)
        );
        assert_eq!(
            execution_stats
                .get("fusion_candidate_count")
                .and_then(JsonValue::as_f64),
            Some(1.0)
        );
        let normalization = results.results[0]
            .explain
            .get("normalization")
            .and_then(JsonValue::as_object)
            .unwrap();
        assert_eq!(
            normalization
                .get("candidate_count")
                .and_then(JsonValue::as_f64),
            Some(1.0)
        );
        assert!(normalization
            .get("bm25_max")
            .and_then(JsonValue::as_f64)
            .map(|value| value > 0.0)
            .unwrap_or(false));
        assert_eq!(
            normalization
                .get("vector_method")
                .and_then(JsonValue::as_str),
            Some("cosine_to_unit_interval")
        );
        assert_eq!(
            results.results[0]
                .explain
                .get("normalization_method")
                .and_then(JsonValue::as_str),
            Some("cosine_to_unit_interval")
        );
        assert_eq!(
            results.results[0]
                .explain
                .get("vector_raw_kind")
                .and_then(JsonValue::as_str),
            Some("similarity")
        );
        let term_details = results.results[0]
            .explain
            .get("matched_term_details")
            .and_then(JsonValue::as_array)
            .unwrap();
        assert!(!term_details.is_empty());
        assert!(term_details.iter().any(|value| {
            value
                .get("frequency")
                .and_then(JsonValue::as_f64)
                .map(|frequency| frequency >= 1.0)
                .unwrap_or(false)
                && value
                    .get("bm25_contribution")
                    .and_then(JsonValue::as_f64)
                    .map(|contribution| contribution > 0.0)
                    .unwrap_or(false)
        }));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn vector_search_metadata_filter_prefilter_updates_execution_stats() {
        let path = temp_path("empty-vector-stats");
        let db = Database::create(&path, DatabaseConfig::default()).unwrap();
        let col = db
            .create_collection(CollectionConfig {
                tokenizer: Default::default(),
                name: "notes".into(),
                embedding_dim: 2,
                distance_metric: DistanceMetric::Cosine,
                embedding_model: Some("m".into()),
                vectors_normalized: false,
                vector_index: VectorIndexConfig::Exact,
            })
            .unwrap();
        col.insert_document(DocumentInput {
            external_id: "doc".into(),
            title: None,
            source_uri: None,
            metadata: parse("{}").unwrap(),
            chunks: vec![ChunkInput {
                chunk_external_id: "doc#1".into(),
                text: "vector scan should still be counted".into(),
                ordinal: 1,
                metadata: parse(r#"{"lang":"rust"}"#).unwrap(),
                source_span: None,
            }],
        })
        .unwrap();
        col.put_embeddings(vec![EmbeddingInput {
            chunk_external_id: "doc#1".into(),
            model_name: "m".into(),
            dim: 2,
            vector: vec![1.0, 0.0],
        }])
        .unwrap();
        drop(col);
        drop(db);

        let db = Database::open(&path).unwrap();
        let results = db
            .collection("notes")
            .unwrap()
            .search()
            .query_embedding(QueryEmbeddingInput {
                model_name: Some("m".into()),
                dim: 2,
                vector: vec![1.0, 0.0],
            })
            .filter(parse(r#"{"chunk.lang":"python"}"#).unwrap())
            .top_k(5)
            .execute()
            .unwrap();

        assert!(results.results.is_empty());
        assert_eq!(results.query_context.execution_stats.vector_scan_count, 1);
        assert_eq!(results.query_context.execution_stats.distance_ops, 0);
        assert_eq!(
            results.query_context.execution_stats.fusion_candidate_count,
            0
        );
        let results_json = results.to_json();
        let execution_stats = results_json
            .get("query_context")
            .and_then(|context| context.get("execution_stats"))
            .and_then(JsonValue::as_object)
            .unwrap();
        assert_eq!(
            execution_stats
                .get("vector_scan_count")
                .and_then(JsonValue::as_f64),
            Some(1.0)
        );
        assert_eq!(
            execution_stats
                .get("distance_ops")
                .and_then(JsonValue::as_f64),
            Some(0.0)
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn blank_text_with_embedding_is_normalized_to_vector_only_query_context() {
        let path = temp_path("blank-text-vector");
        let db = Database::create(&path, DatabaseConfig::default()).unwrap();
        let col = db
            .create_collection(CollectionConfig {
                tokenizer: Default::default(),
                name: "notes".into(),
                embedding_dim: 2,
                distance_metric: DistanceMetric::Cosine,
                embedding_model: None,
                vectors_normalized: false,
                vector_index: VectorIndexConfig::Exact,
            })
            .unwrap();
        col.insert_document(DocumentInput {
            external_id: "doc".into(),
            title: None,
            source_uri: None,
            metadata: parse("{}").unwrap(),
            chunks: vec![ChunkInput {
                chunk_external_id: "doc#1".into(),
                text: "vector only query should not call bm25".into(),
                ordinal: 1,
                metadata: parse("{}").unwrap(),
                source_span: None,
            }],
        })
        .unwrap();
        col.put_embeddings(vec![EmbeddingInput {
            chunk_external_id: "doc#1".into(),
            model_name: "m".into(),
            dim: 2,
            vector: vec![1.0, 0.0],
        }])
        .unwrap();

        let default_results = col
            .search()
            .text("   ")
            .vector(vec![1.0, 0.0])
            .top_k(3)
            .execute()
            .unwrap();
        assert!(matches!(
            default_results.to_json().get("query"),
            Some(JsonValue::Null)
        ));
        assert_eq!(default_results.results[0].bm25_score, None);
        let default_json = default_results.to_json();
        let default_context = default_json
            .get("query_context")
            .and_then(JsonValue::as_object)
            .unwrap();
        assert_eq!(
            default_context
                .get("text_present")
                .and_then(JsonValue::as_bool),
            Some(false)
        );
        assert_eq!(
            default_context
                .get("embedding_present")
                .and_then(JsonValue::as_bool),
            Some(true)
        );
        assert_eq!(
            weight_value(default_context, "effective_weights", "bm25"),
            Some(0.0)
        );
        assert_eq!(
            weight_value(default_context, "effective_weights", "vector"),
            Some(1.0)
        );

        let explicit_results = col
            .search()
            .text("\t \n")
            .vector(vec![1.0, 0.0])
            .weights(SearchWeights::new(0.8, 0.2, 0.0).unwrap())
            .top_k(3)
            .execute()
            .unwrap();
        assert_eq!(explicit_results.results[0].bm25_score, None);
        let explicit_json = explicit_results.to_json();
        let explicit_context = explicit_json
            .get("query_context")
            .and_then(JsonValue::as_object)
            .unwrap();
        assert_eq!(
            explicit_context
                .get("text_present")
                .and_then(JsonValue::as_bool),
            Some(false)
        );
        assert_eq!(
            weight_value(explicit_context, "requested_weights", "bm25"),
            Some(0.8)
        );
        assert_eq!(
            weight_value(explicit_context, "effective_weights", "bm25"),
            Some(0.0)
        );
        assert_eq!(
            weight_value(explicit_context, "effective_weights", "vector"),
            Some(1.0)
        );
        assert!(explicit_context
            .get("renormalization_reason")
            .and_then(JsonValue::as_str)
            .map(|reason| reason.contains("text branch unavailable"))
            .unwrap_or(false));

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn search_rejects_invalid_query_options_with_machine_readable_detail() {
        let path = temp_path("invalid-search-options");
        let db = Database::create(&path, DatabaseConfig::default()).unwrap();
        let col = db
            .create_collection(CollectionConfig {
                tokenizer: Default::default(),
                name: "notes".into(),
                embedding_dim: 2,
                distance_metric: DistanceMetric::Cosine,
                embedding_model: None,
                vectors_normalized: false,
                vector_index: VectorIndexConfig::Exact,
            })
            .unwrap();

        let top_k_err = col.search().text("wal").top_k(0).execute().unwrap_err();
        assert_eq!(top_k_err.code(), ErrorCode::InvalidSearchParams);
        assert_eq!(error_detail_value(&top_k_err, "field"), Some("top_k"));
        assert_eq!(error_detail_value(&top_k_err, "min"), Some("1"));
        assert_eq!(error_detail_value(&top_k_err, "actual"), Some("0"));

        let empty_err = col.search().execute().unwrap_err();
        assert_eq!(empty_err.code(), ErrorCode::InvalidSearchParams);
        assert_eq!(
            error_detail_value(&empty_err, "text_present"),
            Some("false")
        );
        assert_eq!(
            error_detail_value(&empty_err, "embedding_present"),
            Some("false")
        );

        drop(db);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn dot_product_search_preserves_raw_score_when_vectors_are_not_normalized() {
        let path = temp_path("dot-product-raw-score");
        let db = Database::create(&path, DatabaseConfig::default()).unwrap();
        let col = db
            .create_collection(CollectionConfig {
                tokenizer: Default::default(),
                name: "notes".into(),
                embedding_dim: 2,
                distance_metric: DistanceMetric::DotProduct,
                embedding_model: Some("m".into()),
                vectors_normalized: false,
                vector_index: VectorIndexConfig::Exact,
            })
            .unwrap();
        col.insert_document(DocumentInput {
            external_id: "doc".into(),
            title: None,
            source_uri: None,
            metadata: JsonValue::object(),
            chunks: vec![ChunkInput {
                chunk_external_id: "doc#1".into(),
                text: "vector raw dot product".into(),
                ordinal: 1,
                metadata: JsonValue::object(),
                source_span: None,
            }],
        })
        .unwrap();
        col.put_embeddings(vec![EmbeddingInput {
            chunk_external_id: "doc#1".into(),
            model_name: "m".into(),
            dim: 2,
            vector: vec![3.0, 0.0],
        }])
        .unwrap();

        let results = col
            .search()
            .query_embedding(QueryEmbeddingInput {
                model_name: Some("m".into()),
                dim: 2,
                vector: vec![2.0, 0.0],
            })
            .execute()
            .unwrap();
        assert_eq!(results.results[0].vector_score, Some(6.0));
        assert_eq!(results.results[0].normalized_scores.vector_norm, Some(6.0));
        assert_eq!(
            results.results[0]
                .explain
                .get("normalization_method")
                .and_then(JsonValue::as_str),
            Some("raw_dot_product_no_normalization")
        );
        assert_eq!(
            results.results[0]
                .explain
                .get("vector_raw_kind")
                .and_then(JsonValue::as_str),
            Some("dot_product")
        );

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn metadata_boost_without_config_is_not_an_effective_weight_branch() {
        let path = temp_path("metadata-boost-unavailable");
        let db = Database::create(&path, DatabaseConfig::default()).unwrap();
        let col = db
            .create_collection(CollectionConfig {
                tokenizer: Default::default(),
                name: "notes".into(),
                embedding_dim: 2,
                distance_metric: DistanceMetric::Cosine,
                embedding_model: None,
                vectors_normalized: false,
                vector_index: VectorIndexConfig::Exact,
            })
            .unwrap();
        col.insert_document(DocumentInput {
            external_id: "doc".into(),
            title: None,
            source_uri: None,
            metadata: parse("{}").unwrap(),
            chunks: vec![ChunkInput {
                chunk_external_id: "doc#1".into(),
                text: "wal recovery metadata boost disabled".into(),
                ordinal: 1,
                metadata: parse("{}").unwrap(),
                source_span: None,
            }],
        })
        .unwrap();

        let results = col
            .search()
            .text("wal recovery")
            .metadata_boost_enabled(true)
            .weights(SearchWeights::new(0.8, 0.0, 0.2).unwrap())
            .top_k(3)
            .execute()
            .unwrap();
        assert_eq!(results.results[0].metadata_boost, 0.0);
        assert_eq!(results.results[0].normalized_scores.metadata_boost, 0.0);
        let json = results.to_json();
        let context = json
            .get("query_context")
            .and_then(JsonValue::as_object)
            .unwrap();
        assert_eq!(
            context
                .get("metadata_boost_present")
                .and_then(JsonValue::as_bool),
            Some(false)
        );
        assert_eq!(
            context
                .get("metadata_boost_enabled")
                .and_then(JsonValue::as_bool),
            Some(false)
        );
        assert!(matches!(
            context.get("boost_config_hash"),
            Some(JsonValue::Null)
        ));
        assert_eq!(
            context.get("boost_rule_count").and_then(JsonValue::as_f64),
            Some(0.0)
        );
        assert_eq!(
            weight_value(context, "requested_weights", "metadata"),
            Some(0.2)
        );
        assert_eq!(
            weight_value(context, "effective_weights", "metadata"),
            Some(0.0)
        );
        assert_eq!(
            weight_value(context, "effective_weights", "bm25"),
            Some(1.0)
        );
        assert!(context
            .get("renormalization_reason")
            .and_then(JsonValue::as_str)
            .map(|reason| reason.contains("metadata boost unavailable"))
            .unwrap_or(false));

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn explicit_metadata_boost_config_scores_candidates_and_explain_context() {
        let path = temp_path("metadata-boost-config");
        let db = Database::create(&path, DatabaseConfig::default()).unwrap();
        let col = db
            .create_collection(CollectionConfig {
                tokenizer: Default::default(),
                name: "notes".into(),
                embedding_dim: 2,
                distance_metric: DistanceMetric::Cosine,
                embedding_model: None,
                vectors_normalized: false,
                vector_index: VectorIndexConfig::Exact,
            })
            .unwrap();
        col.insert_document(DocumentInput {
            external_id: "doc".into(),
            title: None,
            source_uri: None,
            metadata: parse("{}").unwrap(),
            chunks: vec![
                ChunkInput {
                    chunk_external_id: "doc#low".into(),
                    text: "wal recovery metadata boost".into(),
                    ordinal: 1,
                    metadata: parse(r#"{"importance":0.1}"#).unwrap(),
                    source_span: None,
                },
                ChunkInput {
                    chunk_external_id: "doc#high".into(),
                    text: "wal recovery metadata boost".into(),
                    ordinal: 2,
                    metadata: parse(r#"{"importance":0.9}"#).unwrap(),
                    source_span: None,
                },
            ],
        })
        .unwrap();
        let boost = MetadataBoostConfig::parse(
            parse(
                r#"{
                    "schema_version":"0.1",
                    "rules":[{"when":{"chunk.importance":{"$gte":0.8}},"boost":0.3}],
                    "combine":"sum",
                    "max_boost":0.2
                }"#,
            )
            .unwrap(),
        )
        .unwrap();
        let boost_hash = boost.hash().to_owned();

        let results = col
            .search()
            .text("wal recovery")
            .metadata_boost(boost)
            .weights(SearchWeights::new(0.0, 0.0, 1.0).unwrap())
            .top_k(2)
            .execute()
            .unwrap();

        assert_eq!(results.results[0].chunk_external_id, "doc#high");
        assert_eq!(results.results[0].metadata_boost, 0.2);
        assert_eq!(results.results[0].final_score, 0.2);
        let json = results.to_json();
        let context = json
            .get("query_context")
            .and_then(JsonValue::as_object)
            .unwrap();
        assert_eq!(
            context
                .get("metadata_boost_present")
                .and_then(JsonValue::as_bool),
            Some(true)
        );
        assert_eq!(
            context
                .get("metadata_boost_enabled")
                .and_then(JsonValue::as_bool),
            Some(true)
        );
        assert_eq!(
            context.get("boost_config_hash").and_then(JsonValue::as_str),
            Some(boost_hash.as_str())
        );
        assert_eq!(
            context.get("boost_rule_count").and_then(JsonValue::as_f64),
            Some(1.0)
        );
        assert_eq!(
            weight_value(context, "effective_weights", "metadata"),
            Some(1.0)
        );

        let boost_context = results.results[0]
            .explain
            .get("metadata_boost_context")
            .and_then(JsonValue::as_object)
            .unwrap();
        assert_eq!(
            boost_context.get("present").and_then(JsonValue::as_bool),
            Some(true)
        );
        assert_eq!(
            boost_context.get("enabled").and_then(JsonValue::as_bool),
            Some(true)
        );
        assert_eq!(
            boost_context
                .get("matched_rule_count")
                .and_then(JsonValue::as_f64),
            Some(1.0)
        );
        assert_eq!(
            boost_context.get("config_hash").and_then(JsonValue::as_str),
            Some(boost_hash.as_str())
        );

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn explicit_default_weights_are_reported_as_user_override() {
        let path = temp_path("explicit-default-weights-source");
        let db = Database::create(&path, DatabaseConfig::default()).unwrap();
        let col = db
            .create_collection(CollectionConfig {
                tokenizer: Default::default(),
                name: "notes".into(),
                embedding_dim: 2,
                distance_metric: DistanceMetric::Cosine,
                embedding_model: None,
                vectors_normalized: false,
                vector_index: VectorIndexConfig::Exact,
            })
            .unwrap();
        col.insert_document(DocumentInput {
            external_id: "doc".into(),
            title: None,
            source_uri: None,
            metadata: parse("{}").unwrap(),
            chunks: vec![ChunkInput {
                chunk_external_id: "doc#1".into(),
                text: "explicit default weights are still user input".into(),
                ordinal: 1,
                metadata: parse("{}").unwrap(),
                source_span: None,
            }],
        })
        .unwrap();

        let results = col
            .search()
            .text("explicit default weights")
            .weights(SearchWeights::new(1.0, 0.0, 0.0).unwrap())
            .execute()
            .unwrap();
        let json = results.to_json();
        let context = json
            .get("query_context")
            .and_then(JsonValue::as_object)
            .unwrap();
        assert_eq!(
            context.get("weights_source").and_then(JsonValue::as_str),
            Some("user_override")
        );
        assert_eq!(
            weight_value(context, "effective_weights", "bm25"),
            Some(1.0)
        );
        assert!(matches!(
            context.get("renormalization_reason"),
            Some(JsonValue::Null)
        ));

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn why_not_debug_reports_filter_and_query_context() {
        let path = temp_path("why-not-filter");
        let db = Database::create(&path, DatabaseConfig::default()).unwrap();
        let col = db
            .create_collection(CollectionConfig {
                tokenizer: Default::default(),
                name: "notes".into(),
                embedding_dim: 2,
                distance_metric: DistanceMetric::Cosine,
                embedding_model: None,
                vectors_normalized: false,
                vector_index: VectorIndexConfig::Exact,
            })
            .unwrap();
        col.insert_document(DocumentInput {
            external_id: "doc".into(),
            title: None,
            source_uri: None,
            metadata: parse("{}").unwrap(),
            chunks: vec![
                ChunkInput {
                    chunk_external_id: "doc#rust".into(),
                    text: "wal recovery durable index".into(),
                    ordinal: 1,
                    metadata: parse(r#"{"lang":"rust"}"#).unwrap(),
                    source_span: None,
                },
                ChunkInput {
                    chunk_external_id: "doc#python".into(),
                    text: "wal recovery durable index".into(),
                    ordinal: 2,
                    metadata: parse(r#"{"lang":"python"}"#).unwrap(),
                    source_span: None,
                },
            ],
        })
        .unwrap();

        let diagnosis = db
            .why_not_debug(
                "notes",
                "doc#python",
                Some("wal recovery".into()),
                None,
                parse(r#"{"chunk.lang":"rust"}"#).unwrap(),
                Some(SearchWeights::new(1.0, 0.0, 0.0).unwrap()),
                false,
                5,
                Some(ResourceProfile::Constrained),
            )
            .unwrap();
        let json = diagnosis.to_json();
        assert_eq!(
            json.get("schema_version").and_then(JsonValue::as_str),
            Some("0.1")
        );
        assert_eq!(
            json.get("resource_profile").and_then(JsonValue::as_str),
            Some("constrained")
        );
        let context = json
            .get("query_context")
            .and_then(JsonValue::as_object)
            .unwrap();
        assert_eq!(
            context.get("filter_present").and_then(JsonValue::as_bool),
            Some(true)
        );
        assert_eq!(
            context.get("text").and_then(JsonValue::as_str),
            Some("wal recovery")
        );
        assert_eq!(
            context.get("weights_source").and_then(JsonValue::as_str),
            Some("user_override")
        );
        assert_eq!(
            weight_value(context, "requested_weights", "bm25"),
            Some(1.0)
        );
        let nested_weights = context
            .get("weights")
            .and_then(JsonValue::as_object)
            .unwrap();
        assert_eq!(
            nested_weights.get("source").and_then(JsonValue::as_str),
            Some("user_override")
        );
        let expected = json
            .get("expected_chunk")
            .and_then(JsonValue::as_object)
            .unwrap();
        assert_eq!(
            expected
                .get("chunk_external_id")
                .and_then(JsonValue::as_str),
            Some("doc#python")
        );
        assert_eq!(
            expected.get("exists").and_then(JsonValue::as_bool),
            Some(true)
        );
        let filter_result = json
            .get("filter_result")
            .and_then(JsonValue::as_object)
            .unwrap();
        assert_eq!(
            filter_result.get("matched").and_then(JsonValue::as_bool),
            Some(false)
        );
        assert!(json_array_contains(
            json.get("diagnosis_codes"),
            "filtered_by_metadata"
        ));
        assert!(matches!(json.get("returned_rank"), Some(JsonValue::Null)));
        let limit_analysis = json
            .get("limit_analysis")
            .and_then(JsonValue::as_object)
            .unwrap();
        assert_eq!(
            limit_analysis
                .get("missed_due_to_candidate_limit")
                .and_then(JsonValue::as_bool),
            Some(false)
        );

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn why_not_debug_reports_collection_mismatch_and_missing_chunk() {
        let path = temp_path("why-not-collection-mismatch");
        let db = Database::create(&path, DatabaseConfig::default()).unwrap();
        db.create_collection(CollectionConfig {
            tokenizer: Default::default(),
            name: "notes".into(),
            embedding_dim: 2,
            distance_metric: DistanceMetric::Cosine,
            embedding_model: None,
            vectors_normalized: false,
            vector_index: VectorIndexConfig::Exact,
        })
        .unwrap();
        let archive = db
            .create_collection(CollectionConfig {
                tokenizer: Default::default(),
                name: "archive".into(),
                embedding_dim: 2,
                distance_metric: DistanceMetric::Cosine,
                embedding_model: None,
                vectors_normalized: false,
                vector_index: VectorIndexConfig::Exact,
            })
            .unwrap();
        archive
            .insert_document(DocumentInput {
                external_id: "doc".into(),
                title: None,
                source_uri: None,
                metadata: parse("{}").unwrap(),
                chunks: vec![ChunkInput {
                    chunk_external_id: "archive#1".into(),
                    text: "wal recovery archive".into(),
                    ordinal: 1,
                    metadata: parse("{}").unwrap(),
                    source_span: None,
                }],
            })
            .unwrap();

        let mismatch = db
            .why_not_debug(
                "notes",
                "archive#1",
                Some("wal recovery".into()),
                None,
                parse("{}").unwrap(),
                None,
                false,
                5,
                None,
            )
            .unwrap()
            .to_json();
        let expected = mismatch
            .get("expected_chunk")
            .and_then(JsonValue::as_object)
            .unwrap();
        assert_eq!(
            expected
                .get("collection_matches")
                .and_then(JsonValue::as_bool),
            Some(false)
        );
        assert!(json_array_contains(
            mismatch.get("diagnosis_codes"),
            "collection_mismatch"
        ));

        let missing = db
            .why_not_debug(
                "notes",
                "missing#1",
                Some("wal recovery".into()),
                None,
                parse("{}").unwrap(),
                None,
                false,
                5,
                None,
            )
            .unwrap()
            .to_json();
        let expected = missing
            .get("expected_chunk")
            .and_then(JsonValue::as_object)
            .unwrap();
        assert_eq!(
            expected.get("exists").and_then(JsonValue::as_bool),
            Some(false)
        );
        assert!(json_array_contains(
            missing.get("diagnosis_codes"),
            "expected_chunk_not_found"
        ));

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn why_not_debug_reports_embedding_dim_mismatch_without_vector_echo() {
        let path = temp_path("why-not-embedding-dim-mismatch");
        let db = Database::create(&path, DatabaseConfig::default()).unwrap();
        let col = db
            .create_collection(CollectionConfig {
                tokenizer: Default::default(),
                name: "notes".into(),
                embedding_dim: 2,
                distance_metric: DistanceMetric::Cosine,
                embedding_model: None,
                vectors_normalized: false,
                vector_index: VectorIndexConfig::Exact,
            })
            .unwrap();
        col.insert_document(DocumentInput {
            external_id: "doc".into(),
            title: None,
            source_uri: None,
            metadata: parse("{}").unwrap(),
            chunks: vec![ChunkInput {
                chunk_external_id: "doc#1".into(),
                text: "semantic wal recovery".into(),
                ordinal: 1,
                metadata: parse("{}").unwrap(),
                source_span: None,
            }],
        })
        .unwrap();
        col.put_embeddings(vec![EmbeddingInput {
            chunk_external_id: "doc#1".into(),
            model_name: "m".into(),
            dim: 2,
            vector: vec![1.0, 0.0],
        }])
        .unwrap();

        {
            let mut inner = db.lock().unwrap();
            let chunk_id = inner
                .catalog
                .chunk_external_ids
                .get(col.collection_id(), "doc#1")
                .cloned()
                .unwrap();
            let embedding_id = inner
                .catalog
                .chunk_embeddings
                .get(&chunk_id)
                .cloned()
                .unwrap();
            inner.catalog.record_refs.remove(&embedding_id);
            let embedding = inner.catalog.embeddings.get_mut(&embedding_id).unwrap();
            embedding.dim = 3;
            embedding.vector = vec![1.0, 0.0, 0.0];
        }

        let diagnosis = db
            .why_not_debug(
                "notes",
                "doc#1",
                Some("semantic wal".into()),
                Some(QueryEmbeddingInput {
                    model_name: None,
                    dim: 2,
                    vector: vec![1.0, 0.0],
                }),
                parse("{}").unwrap(),
                None,
                false,
                5,
                None,
            )
            .unwrap()
            .to_json();
        let expected = diagnosis
            .get("expected_chunk")
            .and_then(JsonValue::as_object)
            .unwrap();
        assert_eq!(
            expected
                .get("embedding_dim_matches")
                .and_then(JsonValue::as_bool),
            Some(false)
        );
        let embedding_dim = expected
            .get("embedding_dim")
            .and_then(JsonValue::as_object)
            .unwrap();
        assert_eq!(
            embedding_dim.get("expected").and_then(JsonValue::as_f64),
            Some(2.0)
        );
        assert_eq!(
            embedding_dim.get("actual_dim").and_then(JsonValue::as_f64),
            Some(3.0)
        );
        assert_eq!(
            embedding_dim
                .get("actual_vector_len")
                .and_then(JsonValue::as_f64),
            Some(3.0)
        );
        assert!(embedding_dim.get("vector").is_none());
        assert!(json_array_contains(
            diagnosis.get("diagnosis_codes"),
            "embedding_dim_mismatch"
        ));

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn duplicate_embeddings_in_one_batch_are_rejected() {
        let path = temp_path("duplicate-embeddings");
        let db = Database::create(&path, DatabaseConfig::default()).unwrap();
        let col = db
            .create_collection(CollectionConfig {
                tokenizer: Default::default(),
                name: "notes".into(),
                embedding_dim: 2,
                distance_metric: DistanceMetric::Cosine,
                embedding_model: None,
                vectors_normalized: false,
                vector_index: VectorIndexConfig::Exact,
            })
            .unwrap();
        col.insert_document(DocumentInput {
            external_id: "doc".into(),
            title: None,
            source_uri: None,
            metadata: parse("{}").unwrap(),
            chunks: vec![ChunkInput {
                chunk_external_id: "doc#1".into(),
                text: "local retrieval".into(),
                ordinal: 1,
                metadata: parse("{}").unwrap(),
                source_span: None,
            }],
        })
        .unwrap();
        let err = col
            .put_embeddings(vec![
                EmbeddingInput {
                    chunk_external_id: "doc#1".into(),
                    model_name: "m".into(),
                    dim: 2,
                    vector: vec![1.0, 0.0],
                },
                EmbeddingInput {
                    chunk_external_id: "doc#1".into(),
                    model_name: "m".into(),
                    dim: 2,
                    vector: vec![0.0, 1.0],
                },
            ])
            .unwrap_err();
        assert_eq!(err.code(), ErrorCode::DuplicateExternalId);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn query_embedding_model_mismatch_is_rejected() {
        let path = temp_path("query-model");
        let db = Database::create(&path, DatabaseConfig::default()).unwrap();
        let col = db
            .create_collection(CollectionConfig {
                tokenizer: Default::default(),
                name: "notes".into(),
                embedding_dim: 2,
                distance_metric: DistanceMetric::Cosine,
                embedding_model: Some("expected-model".into()),
                vectors_normalized: false,
                vector_index: VectorIndexConfig::Exact,
            })
            .unwrap();
        let err = col
            .search()
            .query_embedding(QueryEmbeddingInput {
                model_name: Some("other-model".into()),
                dim: 2,
                vector: vec![1.0, 0.0],
            })
            .execute()
            .unwrap_err();
        assert_eq!(err.code(), ErrorCode::EmbeddingModelMismatch);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn hybrid_explain_uses_null_vector_norm_for_missing_embedding() {
        let path = temp_path("missing-vector");
        let db = Database::create(&path, DatabaseConfig::default()).unwrap();
        let col = db
            .create_collection(CollectionConfig {
                tokenizer: Default::default(),
                name: "notes".into(),
                embedding_dim: 2,
                distance_metric: DistanceMetric::Cosine,
                embedding_model: None,
                vectors_normalized: false,
                vector_index: VectorIndexConfig::Exact,
            })
            .unwrap();
        col.insert_document(DocumentInput {
            external_id: "doc".into(),
            title: None,
            source_uri: None,
            metadata: parse("{}").unwrap(),
            chunks: vec![
                ChunkInput {
                    chunk_external_id: "doc#no-embedding".into(),
                    text: "wal recovery only text".into(),
                    ordinal: 1,
                    metadata: parse("{}").unwrap(),
                    source_span: None,
                },
                ChunkInput {
                    chunk_external_id: "doc#embedding".into(),
                    text: "unrelated vector chunk".into(),
                    ordinal: 2,
                    metadata: parse("{}").unwrap(),
                    source_span: None,
                },
            ],
        })
        .unwrap();
        col.put_embeddings(vec![EmbeddingInput {
            chunk_external_id: "doc#embedding".into(),
            model_name: "m".into(),
            dim: 2,
            vector: vec![1.0, 0.0],
        }])
        .unwrap();

        let results = col
            .search()
            .text("wal recovery")
            .query_embedding(QueryEmbeddingInput {
                model_name: None,
                dim: 2,
                vector: vec![1.0, 0.0],
            })
            .top_k(10)
            .execute()
            .unwrap();
        let result = results
            .results
            .iter()
            .find(|result| result.chunk_external_id == "doc#no-embedding")
            .unwrap();
        assert_eq!(result.normalized_scores.vector_norm, None);
        let scores = result
            .explain
            .get("scores")
            .and_then(JsonValue::as_object)
            .unwrap();
        assert!(matches!(scores.get("vector_norm"), Some(JsonValue::Null)));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn export_uses_shared_disk_budget_without_overwriting_previous_artifacts_on_failure() {
        let directory = temp_path("export-runtime-disk");
        fs::create_dir(&directory).unwrap();
        let path = directory.join("data.coffer");
        let output = directory.join("output");
        let mut db = Database::create(&path, DatabaseConfig::default()).unwrap();
        let col = db
            .create_collection(CollectionConfig {
                name: "notes".into(),
                embedding_dim: 2,
                distance_metric: DistanceMetric::Cosine,
                embedding_model: Some("model".into()),
                vectors_normalized: false,
                vector_index: VectorIndexConfig::Exact,
                tokenizer: Default::default(),
            })
            .unwrap();
        col.insert_document(DocumentInput {
            external_id: "doc".into(),
            title: None,
            source_uri: None,
            metadata: JsonValue::object(),
            chunks: vec![ChunkInput {
                chunk_external_id: "chunk".into(),
                text: "needle export".into(),
                ordinal: 0,
                metadata: JsonValue::object(),
                source_span: None,
            }],
        })
        .unwrap();
        col.put_embeddings(vec![EmbeddingInput {
            chunk_external_id: "chunk".into(),
            model_name: "model".into(),
            dim: 2,
            vector: vec![1.0, 0.0],
        }])
        .unwrap();
        drop(col);
        let original = fs::read(&path).unwrap();
        db.export_collection_jsonl("notes", &output).unwrap();
        let names = [
            "documents.jsonl",
            "chunks.jsonl",
            "embeddings.jsonl",
            "export_manifest.json",
        ];
        let required: u64 = names
            .iter()
            .map(|name| fs::metadata(output.join(name)).unwrap().len())
            .sum();
        let mut expected_manifest =
            parse(&fs::read_to_string(output.join("export_manifest.json")).unwrap()).unwrap();
        let initial_time = expected_manifest
            .as_object_mut()
            .unwrap()
            .remove("exported_at")
            .unwrap();
        let fixed_bytes = required - initial_time.as_str().unwrap().len() as u64;
        for name in names {
            fs::write(output.join(name), b"previous complete artifact").unwrap();
        }
        for seed in 0..100u64 {
            // Millisecond timestamps use 20 bytes at an exact second, otherwise 24.
            let available = if seed == 99 {
                fixed_bytes + 24
            } else {
                (fixed_bytes + 20 - 1) * seed / 98
            };
            let budget = Arc::new(crate::temporary::TemporaryDiskBudget::new(available + 7));
            db.temporary_work = budget.clone();
            let (mut file, mut held) = crate::temporary::create_budgeted_temporary_file_in(
                &directory,
                "busy",
                "tmp",
                budget.clone(),
            )
            .unwrap();
            held.reserve_disk(7).unwrap();
            file.write_all(b"busy123").unwrap();
            drop(file);
            let result = db.clone().export_collection_jsonl("notes", &output);
            if seed < 99 {
                let error = result.unwrap_err();
                assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                for name in names {
                    assert_eq!(
                        fs::read(output.join(name)).unwrap(),
                        b"previous complete artifact"
                    );
                }
            } else {
                let summary = result.unwrap();
                assert_eq!(
                    (summary.documents, summary.chunks, summary.embeddings),
                    (1, 1, 1)
                );
                let mut manifest =
                    parse(&fs::read_to_string(output.join("export_manifest.json")).unwrap())
                        .unwrap();
                let exported_at = manifest
                    .as_object_mut()
                    .unwrap()
                    .remove("exported_at")
                    .unwrap();
                let timestamp_bytes = exported_at.as_str().unwrap().len() as u64;
                assert!(matches!(timestamp_bytes, 20 | 24));
                assert_eq!(manifest, expected_manifest);
                assert_eq!(
                    names
                        .iter()
                        .map(|name| fs::metadata(output.join(name)).unwrap().len())
                        .sum::<u64>(),
                    fixed_bytes + timestamp_bytes
                );
            }
            assert_eq!(fs::read_dir(&output).unwrap().count(), 4);
            assert_eq!(fs::read(&path).unwrap(), original);
            assert_eq!(budget.used(), 7);
            drop(held);
            assert_eq!(budget.used(), 0);
        }
        let copy =
            Database::create(directory.join("copy.coffer"), DatabaseConfig::default()).unwrap();
        copy.import_manifest(output.join("export_manifest.json"), OnConflict::Error)
            .unwrap();
        let stats = copy.stats().unwrap();
        assert_eq!(
            (
                stats.document_count,
                stats.chunk_count,
                stats.embedding_count
            ),
            (1, 1, 1)
        );
        assert_eq!(
            copy.collection("notes")
                .unwrap()
                .get_chunk_by_external_id("chunk")
                .unwrap()
                .text,
            "needle export"
        );
        drop(copy);
        drop(db);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn compact_uses_the_runtime_disk_ledger_and_preserves_source_on_exhaustion() {
        let directory = temp_path("compact-runtime-disk");
        fs::create_dir(&directory).unwrap();
        let path = directory.join("data.coffer");
        let mut db = Database::create(&path, DatabaseConfig::default()).unwrap();
        db.create_collection(CollectionConfig {
            name: "notes".into(),
            embedding_dim: 2,
            distance_metric: DistanceMetric::Cosine,
            embedding_model: None,
            vectors_normalized: false,
            vector_index: VectorIndexConfig::Exact,
            tokenizer: Default::default(),
        })
        .unwrap();
        let original = fs::read(&path).unwrap();
        let commit = db.stats().unwrap().last_commit_id;
        for available in 0..100u64 {
            let budget = Arc::new(crate::temporary::TemporaryDiskBudget::new(50 + available));
            db.temporary_work = budget.clone();
            let (mut file, mut held) = crate::temporary::create_budgeted_temporary_file_in(
                &directory,
                "held",
                "tmp",
                budget.clone(),
            )
            .unwrap();
            held.reserve_disk(50).unwrap();
            file.write_all(&[0; 50]).unwrap();
            drop(file);
            let entries = fs::read_dir(&directory).unwrap().count();
            let error = db
                .clone()
                .compact(CompactOptions {
                    rebuild_indexes: available % 2 == 0,
                })
                .unwrap_err();
            assert_eq!(error.code(), ErrorCode::ResourceExhausted);
            assert_eq!(budget.used(), 50);
            assert_eq!(fs::read(&path).unwrap(), original);
            assert_eq!(fs::read_dir(&directory).unwrap().count(), entries);
            assert_eq!(db.stats().unwrap().last_commit_id, commit);
            assert_eq!(db.stats().unwrap().collection_count, 1);
            drop(held);
            assert_eq!(budget.used(), 0);
        }
        db.temporary_work = Default::default();
        db.compact(CompactOptions::default()).unwrap();
        assert_eq!(db.temporary_work.used(), 0);
        drop(db);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn compact_counts_committed_business_history_across_100_cases() {
        let counts = |path: &Path| {
            let scan = scan_file(path).unwrap();
            let mut business = 0;
            let mut tombstones = 0;
            for record in scan.records {
                match record.payload {
                    RecordPayload::CollectionCreate(_)
                    | RecordPayload::CollectionDrop { .. }
                    | RecordPayload::DocumentPut(_)
                    | RecordPayload::ChunkPut(_)
                    | RecordPayload::EmbeddingPut(_) => business += 1,
                    RecordPayload::Tombstone(_) => {
                        business += 1;
                        tombstones += 1;
                    }
                    RecordPayload::Commit(_) | RecordPayload::CompactMarker(_) => {}
                }
            }
            (business, tombstones)
        };
        for seed in 0..100 {
            let path = temp_path(&format!("compact-history-{seed}"));
            let mut db = Database::create(
                &path,
                DatabaseConfig {
                    durability: Durability::Fast,
                    ..Default::default()
                },
            )
            .unwrap();
            let config = |name: &str| CollectionConfig {
                name: name.into(),
                embedding_dim: 2,
                distance_metric: DistanceMetric::Cosine,
                embedding_model: Some("model".into()),
                vectors_normalized: false,
                vector_index: VectorIndexConfig::Exact,
                tokenizer: Default::default(),
            };
            let col = db.create_collection(config("notes")).unwrap();
            db.create_collection(config("reserved")).unwrap();
            db.drop_collection("reserved").unwrap();
            for doc in 0..2 {
                col.insert_document(DocumentInput {
                    external_id: format!("doc-{doc}"),
                    title: None,
                    source_uri: None,
                    metadata: JsonValue::object(),
                    chunks: (0..2)
                        .map(|ordinal| ChunkInput {
                            chunk_external_id: format!("doc-{doc}#{ordinal}"),
                            text: "stable searchable text".into(),
                            ordinal,
                            metadata: JsonValue::object(),
                            source_span: None,
                        })
                        .collect(),
                })
                .unwrap();
            }
            let embedding = || EmbeddingInput {
                chunk_external_id: "doc-0#0".into(),
                model_name: "model".into(),
                dim: 2,
                vector: vec![1.0, 0.0],
            };
            col.put_embeddings(vec![embedding()]).unwrap();
            let original_reader = Database::open_read_only(&path).unwrap();
            match seed % 5 {
                0 => {}
                1 => {
                    db.put_embeddings_in_collection(
                        col.collection_id(),
                        vec![embedding()],
                        OnConflict::Replace,
                    )
                    .unwrap();
                }
                2 => {
                    let chunk = col.get_chunk_by_external_id("doc-0#0").unwrap();
                    col.delete_chunk(&chunk.chunk_id).unwrap();
                }
                3 => {
                    let doc = col.get_document_by_external_id("doc-0").unwrap();
                    col.delete_document(&doc.document_id).unwrap();
                }
                4 => db.drop_collection("notes").unwrap(),
                _ => unreachable!(),
            }
            drop(col);
            if seed % 2 == 0 {
                drop(db);
                db = Database::open_with_options(
                    &path,
                    DatabaseOpenOptions {
                        durability: Some(Durability::Fast),
                        ..Default::default()
                    },
                )
                .unwrap();
            }
            let expected = match seed % 5 {
                0 => (0, 0),
                1 => (1, 0),
                2 => (3, 1),
                3 => (5, 1),
                4 => (7, 0),
                _ => unreachable!(),
            };
            let live = db.stats().unwrap();
            for iteration in 0..3 {
                let before = counts(&path);
                assert_eq!(db.lock().unwrap().catalog.business_record_count(), before.0);
                let summary = db
                    .compact(CompactOptions {
                        rebuild_indexes: (seed / 5 + iteration) % 2 == 0,
                    })
                    .unwrap();
                let after = counts(&path);
                let actual = (summary.records_removed, summary.tombstones_removed);
                assert_eq!(
                    actual,
                    (before.0 - after.0, before.1 - after.1),
                    "seed={seed} iteration={iteration}"
                );
                assert_eq!(actual, if iteration == 0 { expected } else { (0, 0) });
                assert_eq!(
                    summary.to_json().get("records_removed"),
                    Some(&JsonValue::Number(actual.0 as f64))
                );
                assert_eq!(db.lock().unwrap().catalog.business_record_count(), after.0);
                let current = db.stats().unwrap();
                assert_eq!(
                    (
                        current.document_count,
                        current.chunk_count,
                        current.embedding_count
                    ),
                    (live.document_count, live.chunk_count, live.embedding_count)
                );
                assert_eq!(original_reader.stats().unwrap().chunk_count, 4);
                assert_eq!(
                    db.create_collection(config("reserved"))
                        .err()
                        .unwrap()
                        .code(),
                    ErrorCode::CollectionDeletedNameReserved
                );
                assert!(db.check(CheckOptions::default()).unwrap().issues.is_empty());
            }
            drop(original_reader);
            drop(db);
            fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn compact_rejects_inconsistent_history_before_replacing_the_source() {
        let path = temp_path("compact-inconsistent-history");
        let db = Database::create(&path, DatabaseConfig::default()).unwrap();
        db.create_collection(CollectionConfig {
            name: "notes".into(),
            embedding_dim: 2,
            distance_metric: DistanceMetric::Cosine,
            embedding_model: None,
            vectors_normalized: false,
            vector_index: VectorIndexConfig::Exact,
            tokenizer: Default::default(),
        })
        .unwrap();
        {
            let mut inner = db.lock().unwrap();
            let mut phantom = inner.catalog.collections.values().next().unwrap().clone();
            phantom.collection_id = "col_uncommitted".into();
            phantom.name = "uncommitted".into();
            inner
                .catalog
                .collections
                .insert(phantom.collection_id.clone(), phantom);
        }
        let original = fs::read(&path).unwrap();
        let error = db.compact(CompactOptions::default()).unwrap_err();
        assert_eq!(error.code(), ErrorCode::FileFormatError);
        assert_eq!(error.message(), "inconsistent compaction record counts");
        assert_eq!(fs::read(&path).unwrap(), original);
        drop(db);
        let reopened = Database::open(&path).unwrap();
        assert_eq!(reopened.stats().unwrap().collection_count, 1);
        let summary = reopened.compact(CompactOptions::default()).unwrap();
        assert_eq!(
            (summary.records_removed, summary.tombstones_removed),
            (0, 0)
        );
        drop(reopened);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn compact_collection_tombstone_preserves_two_business_records() {
        let path = temp_path("compact-collection-tombstone-count");
        let db = Database::create(&path, DatabaseConfig::default()).unwrap();
        let col = db
            .create_collection(CollectionConfig {
                name: "reserved".into(),
                embedding_dim: 2,
                distance_metric: DistanceMetric::Cosine,
                embedding_model: None,
                vectors_normalized: false,
                vector_index: VectorIndexConfig::Exact,
                tokenizer: Default::default(),
            })
            .unwrap();
        let context = db.json_context();
        db.lock()
            .unwrap()
            .append_and_apply(
                OwnedRecordBatch::single(
                    RecordPayload::Tombstone(Tombstone {
                        target_type: "collection".into(),
                        target_id: col.collection_id().into(),
                        deleted_at: time::now_rfc3339(),
                        reason: None,
                    }),
                    &context,
                )
                .unwrap(),
                &context,
            )
            .unwrap();
        let summary = db.compact(CompactOptions::default()).unwrap();
        assert_eq!(summary.records_removed, 0);
        assert_eq!(summary.tombstones_removed, 1);
        assert_eq!(db.lock().unwrap().catalog.business_record_count(), 2);
        assert_eq!(db.stats().unwrap().collection_count, 0);
        assert_eq!(
            scan_file(&path)
                .unwrap()
                .records
                .iter()
                .filter(|r| matches!(r.payload, RecordPayload::CollectionDrop { .. }))
                .count(),
            1
        );
        let repeated = db.compact(CompactOptions::default()).unwrap();
        assert_eq!(
            (repeated.records_removed, repeated.tombstones_removed),
            (0, 0)
        );
        drop(col);
        drop(db);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn compact_never_rewinds_batch_or_sequence_numbers() {
        let path = temp_path("compact-monotonic-counters");
        let mut db = Database::create(&path, DatabaseConfig::default()).unwrap();
        db.create_collection(CollectionConfig {
            name: "notes".into(),
            embedding_dim: 2,
            distance_metric: DistanceMetric::Cosine,
            embedding_model: None,
            vectors_normalized: false,
            vector_index: VectorIndexConfig::Exact,
            tokenizer: Default::default(),
        })
        .unwrap();
        db.drop_collection("notes").unwrap();
        for _ in 0..3 {
            let before = scan_file(&path).unwrap();
            db.compact(CompactOptions {
                rebuild_indexes: true,
            })
            .unwrap();
            let after = scan_file(&path).unwrap();
            assert!(after.findings.is_empty());
            assert!(after.last_valid_commit > before.last_valid_commit);
            assert!(
                after
                    .records
                    .iter()
                    .all(|record| record.record_ref.sequence_no
                        > before.last_valid_commit_sequence_no)
            );
            assert!(after.last_valid_commit_sequence_no > before.last_valid_commit_sequence_no);
            let marker = after
                .records
                .iter()
                .find_map(|record| match &record.payload {
                    RecordPayload::CompactMarker(marker) => Some(marker),
                    _ => None,
                })
                .unwrap();
            assert_eq!(
                marker.source_last_commit_sequence_no,
                before.last_valid_commit_sequence_no
            );
            drop(db);
            db = Database::open(&path).unwrap();
        }
        let before = scan_file(&path).unwrap();
        db.create_collection(CollectionConfig {
            name: "other".into(),
            embedding_dim: 2,
            distance_metric: DistanceMetric::Cosine,
            embedding_model: None,
            vectors_normalized: false,
            vector_index: VectorIndexConfig::Exact,
            tokenizer: Default::default(),
        })
        .unwrap();
        let after = scan_file(&path).unwrap();
        assert_eq!(after.last_valid_commit, before.last_valid_commit + 1);
        assert!(after.last_valid_commit_sequence_no > before.last_valid_commit_sequence_no);
        drop(db);
        let _ = fs::remove_file(path);
    }

    #[test]
    fn compact_preserves_dropped_collection_name_reservation() {
        let path = temp_path("compact-reserved");
        let db = Database::create(&path, DatabaseConfig::default()).unwrap();
        db.create_collection(CollectionConfig {
            tokenizer: Default::default(),
            name: "notes".into(),
            embedding_dim: 2,
            distance_metric: DistanceMetric::Cosine,
            embedding_model: None,
            vectors_normalized: false,
            vector_index: VectorIndexConfig::Exact,
        })
        .unwrap();
        db.drop_collection("notes").unwrap();
        let summary = db
            .compact(CompactOptions {
                rebuild_indexes: true,
            })
            .unwrap();
        assert_eq!(summary.index_rebuild, "full_checkpoint_written");
        assert_eq!(summary.index_rebuild_reason, None);
        assert_eq!(
            summary
                .to_json()
                .get("index_rebuild_reason")
                .and_then(JsonValue::as_str),
            None
        );
        let disabled_summary = db
            .compact(CompactOptions {
                rebuild_indexes: false,
            })
            .unwrap();
        assert_eq!(disabled_summary.index_rebuild, "disabled");
        assert!(disabled_summary.index_rebuild_reason.is_none());
        db.compact(CompactOptions {
            rebuild_indexes: true,
        })
        .unwrap();
        let err = match db.create_collection(CollectionConfig {
            tokenizer: Default::default(),
            name: "notes".into(),
            embedding_dim: 2,
            distance_metric: DistanceMetric::Cosine,
            embedding_model: None,
            vectors_normalized: false,
            vector_index: VectorIndexConfig::Exact,
        }) {
            Ok(_) => panic!("dropped collection name should remain reserved"),
            Err(err) => err,
        };
        assert_eq!(err.code(), ErrorCode::CollectionDeletedNameReserved);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn bounded_compaction_preserves_rank_reserved_names_and_pinned_readers() {
        let path = temp_path("bounded-compaction-lifecycle");
        let db = Database::create(
            &path,
            DatabaseConfig {
                resource_profile: ResourceProfile::Constrained,
                cache_budget_mb: Some(16),
                ..DatabaseConfig::default()
            },
        )
        .unwrap();
        let config = |name: &str| CollectionConfig {
            tokenizer: Default::default(),
            name: name.into(),
            embedding_dim: 2,
            distance_metric: DistanceMetric::Cosine,
            embedding_model: Some("m".into()),
            vectors_normalized: false,
            vector_index: VectorIndexConfig::Exact,
        };
        let collection = db.create_collection(config("notes")).unwrap();
        db.create_collection(config("reserved")).unwrap();
        db.drop_collection("reserved").unwrap();
        collection
            .insert_document(DocumentInput {
                external_id: "doc".into(),
                title: None,
                source_uri: None,
                metadata: JsonValue::object(),
                chunks: (0..256)
                    .map(|index| ChunkInput {
                        chunk_external_id: format!("chunk-{index:04}"),
                        text: "needle stable compaction".into(),
                        ordinal: index,
                        metadata: JsonValue::object(),
                        source_span: None,
                    })
                    .collect(),
            })
            .unwrap();
        collection
            .put_embeddings(
                (0..256)
                    .map(|index| EmbeddingInput {
                        chunk_external_id: format!("chunk-{index:04}"),
                        model_name: "m".into(),
                        dim: 2,
                        vector: vec![1.0, 0.0],
                    })
                    .collect(),
            )
            .unwrap();
        let signature = |database: &Database| {
            database
                .collection("notes")
                .unwrap()
                .search()
                .text("needle")
                .query_embedding(QueryEmbeddingInput {
                    model_name: Some("m".into()),
                    dim: 2,
                    vector: vec![1.0, 0.0],
                })
                .resource_profile(ResourceProfile::Balanced)
                .top_k(10)
                .execute()
                .unwrap()
                .results
                .into_iter()
                .map(|hit| (hit.chunk_external_id, hit.final_score.to_bits()))
                .collect::<Vec<_>>()
        };
        let expected = signature(&db);
        let pinned = Database::open_read_only(&path).unwrap();
        let pinned_commit = pinned.stats().unwrap().last_commit_id;
        for rebuild_indexes in [false, true] {
            db.compact(CompactOptions { rebuild_indexes }).unwrap();
            let scan = scan_file(&path).unwrap();
            assert!(scan.findings.is_empty());
            assert_eq!(scan.commits, 3);
            assert!(scan.last_valid_commit > pinned_commit);
            assert_eq!(scan.index_checkpoint.is_some(), rebuild_indexes);
            assert_eq!(signature(&db), expected);
            assert_eq!(signature(&pinned), expected);
            assert_eq!(pinned.stats().unwrap().last_commit_id, pinned_commit);
            let reopened = Database::open_read_only(&path).unwrap();
            assert_eq!(signature(&reopened), expected);
            assert_eq!(reopened.stats().unwrap().chunk_count, 256);
            assert_eq!(reopened.stats().unwrap().embedding_count, 256);
            assert_eq!(
                db.create_collection(config("reserved"))
                    .err()
                    .unwrap()
                    .code(),
                ErrorCode::CollectionDeletedNameReserved
            );
        }
        drop(pinned);
        drop(collection);
        drop(db);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn compact_skips_disabled_text_rebuilds_and_keeps_catalog_payloads_authoritative() {
        let text = "needle resources retrieval";
        let path = query_preflight_fixture("compact-disabled-text", text);
        let db = Database::open_with_options(
            &path,
            DatabaseOpenOptions {
                durability: Some(Durability::Fast),
                ..Default::default()
            },
        )
        .unwrap();
        let pinned = Database::open_read_only(&path).unwrap();
        let signature = |database: &Database| {
            ["first", "second"].map(|name| {
                let collection = database.collection(name).unwrap();
                let chunk = collection.get_chunk_by_external_id("chunk-1").unwrap();
                assert_eq!(chunk.text, text);
                assert_eq!(
                    chunk.metadata.get("lang").and_then(JsonValue::as_str),
                    Some("rust")
                );
                collection
                    .search()
                    .text("needle retrieval")
                    .query_embedding(QueryEmbeddingInput {
                        model_name: Some("m".into()),
                        dim: 2,
                        vector: vec![1.0, 0.0],
                    })
                    .top_k(10)
                    .execute()
                    .unwrap()
                    .results
                    .into_iter()
                    .map(|hit| {
                        (
                            hit.chunk_external_id,
                            hit.final_score.to_bits(),
                            hit.bm25_score.map(f64::to_bits),
                        )
                    })
                    .collect::<Vec<_>>()
            })
        };
        let expected = signature(&db);
        let pinned_commit = pinned.stats().unwrap().last_commit_id;
        for seed in 0..100 {
            let rebuild_indexes = seed % 2 != 0;
            let old_sequence = db.lock().unwrap().last_commit_sequence_no;
            let old_refreshes = db.lock().unwrap().index_checkpoint_refreshes;
            TEXT_BUILDS.with(|count| count.set(0));
            TEXT_CHECKPOINT_BUILDS.with(|count| count.set(0));
            let summary = db.compact(CompactOptions { rebuild_indexes }).unwrap();
            assert_eq!(TEXT_BUILDS.with(Cell::get), 0);
            assert_eq!(
                TEXT_CHECKPOINT_BUILDS.with(Cell::get),
                usize::from(rebuild_indexes)
            );
            assert_eq!(
                summary.index_rebuild,
                if rebuild_indexes {
                    "full_checkpoint_written"
                } else {
                    "disabled"
                }
            );
            {
                let inner = db.lock().unwrap();
                assert_eq!(
                    inner.index_checkpoint_refreshes,
                    old_refreshes + usize::from(rebuild_indexes)
                );
                assert!(inner
                    .catalog
                    .record_refs
                    .values()
                    .all(|record| record.sequence_no > old_sequence));
                assert_eq!(
                    inner.text_index.chunk_lengths.len(),
                    if rebuild_indexes { 6 } else { 0 }
                );
                if !rebuild_indexes {
                    assert_eq!(inner.text_index.term_count(), 0);
                    assert!(inner.text_index.total_chunks_by_collection.is_empty());
                    assert!(inner.text_index.avg_len_by_collection.is_empty());
                    assert_eq!(inner.index_suffix_status, IndexSuffixState::Missing);
                }
            }
            assert_eq!(signature(&db), expected);
            assert_eq!(signature(&pinned), expected);
            assert_eq!(pinned.stats().unwrap().last_commit_id, pinned_commit);
            if seed % 10 == 0 {
                let reopened = Database::open_read_only(&path).unwrap();
                assert_eq!(signature(&reopened), expected);
            }
        }
        drop(pinned);
        drop(db);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn compact_preserves_id_high_watermarks_for_deleted_records() {
        let path = temp_path("compact-id-watermarks");
        let db = Database::create(&path, DatabaseConfig::default()).unwrap();
        let col = db
            .create_collection(CollectionConfig {
                tokenizer: Default::default(),
                name: "notes".into(),
                embedding_dim: 2,
                distance_metric: DistanceMetric::Cosine,
                embedding_model: Some("m".into()),
                vectors_normalized: false,
                vector_index: VectorIndexConfig::Exact,
            })
            .unwrap();
        let inserted = col
            .insert_document(DocumentInput {
                external_id: "doc-old".into(),
                title: None,
                source_uri: None,
                metadata: parse("{}").unwrap(),
                chunks: vec![ChunkInput {
                    chunk_external_id: "doc-old#1".into(),
                    text: "old compacted payload".into(),
                    ordinal: 1,
                    metadata: parse("{}").unwrap(),
                    source_span: None,
                }],
            })
            .unwrap();
        assert_eq!(inserted.documents_inserted, 1);
        col.put_embeddings(vec![EmbeddingInput {
            chunk_external_id: "doc-old#1".into(),
            model_name: "m".into(),
            dim: 2,
            vector: vec![1.0, 0.0],
        }])
        .unwrap();
        let old_document = col.get_document_by_external_id("doc-old").unwrap();
        let old_chunk = col.get_chunk_by_external_id("doc-old#1").unwrap();
        let old_embedding_id = old_chunk.embedding_id.clone().unwrap();

        col.delete_document(&old_document.document_id).unwrap();
        let summary = db
            .compact(CompactOptions {
                rebuild_indexes: true,
            })
            .unwrap();
        assert_eq!(summary.segments_merged, 2);

        assert_eq!(
            col.get_document(&old_document.document_id)
                .unwrap_err()
                .code(),
            ErrorCode::DocumentNotFound
        );
        assert_eq!(
            col.get_chunk(&old_chunk.chunk_id).unwrap_err().code(),
            ErrorCode::ChunkNotFound
        );

        col.insert_document(DocumentInput {
            external_id: "doc-new".into(),
            title: None,
            source_uri: None,
            metadata: parse("{}").unwrap(),
            chunks: vec![ChunkInput {
                chunk_external_id: "doc-new#1".into(),
                text: "new payload after compact".into(),
                ordinal: 1,
                metadata: parse("{}").unwrap(),
                source_span: None,
            }],
        })
        .unwrap();
        col.put_embeddings(vec![EmbeddingInput {
            chunk_external_id: "doc-new#1".into(),
            model_name: "m".into(),
            dim: 2,
            vector: vec![0.0, 1.0],
        }])
        .unwrap();

        let new_document = col.get_document_by_external_id("doc-new").unwrap();
        let new_chunk = col.get_chunk_by_external_id("doc-new#1").unwrap();
        let new_embedding_id = new_chunk.embedding_id.clone().unwrap();
        assert!(
            id_number(&new_document.document_id, "doc_")
                > id_number(&old_document.document_id, "doc_")
        );
        assert!(id_number(&new_chunk.chunk_id, "chk_") > id_number(&old_chunk.chunk_id, "chk_"));
        assert!(id_number(&new_embedding_id, "emb_") > id_number(&old_embedding_id, "emb_"));
        drop(col);
        drop(db);

        let db = Database::open(&path).unwrap();
        let col = db.collection("notes").unwrap();
        let reopened_chunk = col.get_chunk_by_external_id("doc-new#1").unwrap();
        assert_eq!(
            reopened_chunk.embedding_id.as_deref(),
            Some(new_embedding_id.as_str())
        );
        assert!(
            id_number(&reopened_chunk.chunk_id, "chk_") > id_number(&old_chunk.chunk_id, "chk_")
        );
        drop(col);
        drop(db);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn read_only_storage_never_materializes_text_segments_for_nonzero_cache_budgets() {
        let path = query_preflight_fixture("readonly-storage-text", "query alpha beta");
        let original = fs::read(&path).unwrap();
        for budget in [0, 1_048_576, u64::MAX] {
            let context = crate::json::JsonOperationContext::standalone();
            let available = context.available_workspace_bytes();
            let mut catalog = Catalog::new_in(&context);
            let (storage, mut scan) = Storage::open_with_retained_records(
                &path,
                true,
                budget,
                &context,
                |record, work| catalog.apply_retained(record, work, &context),
            )
            .unwrap();
            assert!(!scan.findings.iter().any(|issue| issue.severity == "error"));
            assert_eq!(catalog.live_chunks().count(), 6);
            let checkpoint = scan.lazy_vector_checkpoint.take().unwrap();
            assert!(checkpoint.text_segments.is_empty(), "budget={budget}");
            let hits = TextIndex::search_from_segment(
                &checkpoint.vectors,
                &storage,
                &catalog.collection_by_name("first").unwrap().collection_id,
                "query",
                TextSearchOptions::with_explanations(2),
                &context,
            )
            .unwrap()
            .unwrap();
            assert_eq!(hits.len(), 2);
            assert!(hits.iter().all(|hit| hit.matched_terms.len() == 1));
            assert_eq!(fs::read(&path).unwrap(), original);
            drop((hits, checkpoint, scan, storage, catalog));
            assert_eq!(context.available_workspace_bytes(), available);
        }
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn read_only_text_matches_cached_scores_explanations_stats_and_debug_for_all_cache_budgets() {
        for seed in 0..20 {
            let path = temp_path("zero-cache-text");
            let db = Database::create(
                &path,
                DatabaseConfig {
                    durability: Durability::Fast,
                    ..Default::default()
                },
            )
            .unwrap();
            for (owner, count) in [("first", 25 + seed % 7), ("second", 17), ("empty", 0)] {
                let collection = db
                    .create_collection(CollectionConfig {
                        name: owner.into(),
                        embedding_dim: 2,
                        tokenizer: Default::default(),
                        distance_metric: DistanceMetric::Cosine,
                        embedding_model: None,
                        vectors_normalized: false,
                        vector_index: VectorIndexConfig::Exact,
                    })
                    .unwrap();
                if count == 0 {
                    continue;
                }
                collection
                    .insert_document(DocumentInput {
                        external_id: "doc".into(),
                        title: None,
                        source_uri: None,
                        metadata: JsonValue::object(),
                        chunks: (0..count)
                            .map(|ordinal| ChunkInput {
                                chunk_external_id: format!("chunk-{ordinal}"),
                                text: if ordinal % 7 == 0 {
                                    "the and".into()
                                } else {
                                    format!(
                                        "shared {} {} term{}",
                                        "alphaBeta ".repeat((ordinal + seed) % 5),
                                        [
                                            "storage_path 42",
                                            "delta delta gamma",
                                            "\u{0130} \u{03a3}"
                                        ][ordinal % 3],
                                        (ordinal + seed) % 13
                                    )
                                },
                                ordinal,
                                metadata: parse(r#"{"lang":"rust"}"#).unwrap(),
                                source_span: None,
                            })
                            .collect(),
                    })
                    .unwrap();
                let deleted = collection.get_chunk_by_external_id("chunk-0").unwrap();
                collection.delete_chunk(&deleted.chunk_id).unwrap();
                collection
                    .put_embeddings(
                        (1..count)
                            .filter(|ordinal| ordinal % 3 != 0)
                            .map(|ordinal| EmbeddingInput {
                                chunk_external_id: format!("chunk-{ordinal}"),
                                model_name: "m".into(),
                                dim: 2,
                                vector: vec![1.0, (ordinal % 7) as f32 / 7.0],
                            })
                            .collect(),
                    )
                    .unwrap();
            }
            for cache_budget_mb in [Some(0), Some(1), Some(256), None] {
                let readonly = Database::open_with_options(
                    &path,
                    DatabaseOpenOptions {
                        read_only: true,
                        cache_budget_mb,
                        ..Default::default()
                    },
                )
                .unwrap();
                {
                    let inner = readonly.lock().unwrap();
                    assert!(inner.lazy_text);
                    assert!(inner.text_index.chunk_lengths.is_empty());
                    assert_eq!(inner.text_index.term_count(), 0);
                    assert_eq!(
                        inner.index_suffix_status,
                        IndexSuffixState::PersistentCheckpoint
                    );
                    assert!(inner.lazy_vector_segment.is_none());
                    assert_eq!(readonly.temporary_work.used(), 0);
                }
                assert_eq!(
                    readonly.index_stats().unwrap().text_terms,
                    db.index_stats().unwrap().text_terms
                );
                for owner in ["first", "second", "empty"] {
                    let actual = readonly.collection(owner).unwrap().stats().unwrap();
                    let expected = db.collection(owner).unwrap().stats().unwrap();
                    assert_eq!(actual.text_index_terms, expected.text_index_terms);
                    assert_eq!(actual.chunk_count, expected.chunk_count);
                    if owner == "empty" {
                        let context = crate::json::JsonOperationContext::new(Arc::new(
                            crate::json::WorkLedger::with_limits(0, 0, 0),
                        ));
                        let mut inner = readonly.lock().unwrap();
                        let collection_id = inner
                            .catalog
                            .collection_by_name(owner)
                            .unwrap()
                            .collection_id
                            .clone();
                        let hits = inner
                            .search_checkpoint_text_index(
                                &collection_id,
                                "actual query",
                                TextSearchOptions::with_explanations(100),
                                &context,
                            )
                            .unwrap()
                            .unwrap();
                        let cached = db
                            .lock()
                            .unwrap()
                            .text_index
                            .search(
                                &collection_id,
                                "actual query",
                                TextSearchOptions::with_explanations(100),
                                &context,
                            )
                            .unwrap();
                        assert!(hits.is_empty() && cached.is_empty());
                        assert_eq!(context.available_workspace_bytes(), 0);
                    }
                    for query in [
                        "shared delta",
                        "alphaBeta storage_path",
                        "term5 term99",
                        "\u{0130} \u{03a3}",
                        "the and",
                    ] {
                        let context = readonly.json_context();
                        let actual = {
                            let inner = readonly.lock().unwrap();
                            TextIndex::search_from_segment(
                                inner.lazy_vector_source.as_ref().unwrap(),
                                &inner.storage,
                                &inner
                                    .catalog
                                    .collection_by_name(owner)
                                    .unwrap()
                                    .collection_id,
                                query,
                                TextSearchOptions::with_explanations(1 + seed % 9),
                                &context,
                            )
                            .unwrap()
                            .unwrap()
                        };
                        let expected = {
                            let inner = db.lock().unwrap();
                            inner
                                .text_index
                                .search(
                                    &inner
                                        .catalog
                                        .collection_by_name(owner)
                                        .unwrap()
                                        .collection_id,
                                    query,
                                    TextSearchOptions::with_explanations(1 + seed % 9),
                                    &context,
                                )
                                .unwrap()
                        };
                        assert_eq!(
                            format!("{:?}", actual.as_slice()),
                            format!("{:?}", expected.as_slice()),
                            "seed {seed}, owner {owner}, query {query}"
                        );
                        for profile in [
                            ResourceProfile::Expanded,
                            ResourceProfile::Balanced,
                            ResourceProfile::Constrained,
                        ] {
                            for hybrid in [false, true] {
                                for &top_k in if seed == 0 && query == "shared delta" {
                                    &[3, 600][..]
                                } else {
                                    &[3][..]
                                } {
                                    let mut actual = readonly
                                        .collection(owner)
                                        .unwrap()
                                        .search()
                                        .text(query)
                                        .resource_profile(profile)
                                        .top_k(top_k);
                                    let mut expected = db
                                        .collection(owner)
                                        .unwrap()
                                        .search()
                                        .text(query)
                                        .resource_profile(profile)
                                        .top_k(top_k);
                                    if hybrid {
                                        actual = actual.vector(vec![1.0, 0.0]);
                                        expected = expected.vector(vec![1.0, 0.0]);
                                    }
                                    let actual = actual.execute().unwrap();
                                    let expected = expected.execute().unwrap();
                                    assert_eq!(
                                        actual.query_context.transient_rebuild.budget_mb,
                                        cache_budget_mb.unwrap_or(profile.params().cache_budget_mb)
                                    );
                                    if hybrid && owner != "empty" {
                                        let inner = readonly.lock().unwrap();
                                        assert_eq!(
                                            inner.lazy_vector_segment.as_ref().is_some_and(
                                                |segment| { !segment.vectors.is_empty() }
                                            ),
                                            cache_budget_mb != Some(0)
                                        );
                                    }
                                    assert_eq!(
                                        actual.query_context.index_source,
                                        "persistent_checkpoint"
                                    );
                                    assert!(!actual.query_context.transient_rebuild.started);
                                    assert_eq!(
                                        actual.query_context.transient_rebuild.chunks_scanned,
                                        expected.query_context.transient_rebuild.chunks_scanned
                                    );
                                    let signature = |result: &SearchResults| {
                                        result
                                            .results
                                            .iter()
                                            .map(|hit| {
                                                (
                                                    hit.chunk_id.clone(),
                                                    hit.final_score.to_bits(),
                                                    hit.bm25_score.map(f64::to_bits),
                                                    hit.vector_score.map(f64::to_bits),
                                                    hit.explain
                                                        .get("bm25")
                                                        .map(JsonValue::to_compact_string),
                                                )
                                            })
                                            .collect::<Vec<_>>()
                                    };
                                    assert_eq!(signature(&actual), signature(&expected));
                                    let limit = profile.params().bm25_candidates.max(top_k);
                                    assert_eq!(actual.query_context.candidate_limits.bm25, limit);
                                    assert_eq!(actual.query_context.candidate_limits.vector, limit);
                                    assert_eq!(actual.query_context.candidate_limits.fusion, limit);
                                    assert_eq!(readonly.temporary_work.used(), 0);
                                }
                            }
                        }
                    }
                    if owner != "empty" {
                        let chunk = readonly
                            .collection(owner)
                            .unwrap()
                            .get_chunk_by_external_id("chunk-1")
                            .unwrap();
                        let actual = readonly
                            .why_not_debug(
                                owner,
                                &chunk.chunk_id,
                                Some("shared delta".into()),
                                None,
                                JsonValue::object(),
                                None,
                                false,
                                3,
                                None,
                            )
                            .unwrap();
                        let expected = db
                            .why_not_debug(
                                owner,
                                &chunk.chunk_id,
                                Some("shared delta".into()),
                                None,
                                JsonValue::object(),
                                None,
                                false,
                                3,
                                None,
                            )
                            .unwrap();
                        assert_eq!(
                            actual.result.to_compact_string(),
                            expected.result.to_compact_string()
                        );
                    }
                }
                let inner = readonly.lock().unwrap();
                if cache_budget_mb == Some(0) {
                    assert!(!inner.text_index.has_resident_cache());
                } else {
                    assert!(inner.text_index.has_resident_cache());
                    assert!(
                        inner.optional_cache_bytes() <= inner.active_cache_budget_mb() * 1_048_576
                    );
                }
                drop(inner);
                drop(readonly);
            }
            drop(db);
            fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn zero_cache_text_discards_a_segment_truncated_between_scoring_and_explanations() {
        let path = query_preflight_fixture("text-second-pass-truncation", "needle alpha beta");
        let db = Database::open_with_options(
            &path,
            DatabaseOpenOptions {
                read_only: true,
                cache_budget_mb: Some(0),
                ..Default::default()
            },
        )
        .unwrap();
        let context = db.json_context();
        let before = context.available_workspace_bytes();
        let prefix = {
            let inner = db.lock().unwrap();
            let end = inner.storage.critical_end_offset();
            let mut calls = 0;
            let result = TextIndex::search_segment_with(
                "needle alpha",
                TextSearchOptions::with_explanations(5),
                &context,
                |visit| {
                    calls += 1;
                    if calls == 2 {
                        std::fs::OpenOptions::new()
                            .write(true)
                            .open(&path)
                            .unwrap()
                            .set_len(end)
                            .unwrap();
                    }
                    let input = context.workspace_reservation();
                    inner.lazy_vector_source.as_ref().unwrap().visit_text(
                        &inner.storage,
                        &inner
                            .catalog
                            .collection_by_name("first")
                            .unwrap()
                            .collection_id,
                        &input,
                        visit,
                    )
                },
            )
            .unwrap();
            assert_eq!(calls, 2);
            assert!(result.is_none());
            end
        };
        assert_eq!(context.available_workspace_bytes(), before);
        let error = db
            .collection("first")
            .unwrap()
            .search()
            .text("needle alpha")
            .execute()
            .unwrap_err();
        assert_eq!(error.code(), ErrorCode::IndexCorruption);
        assert!(error.detail().contains(&(
            "index_unavailable_reason".into(),
            "transient_rebuild_budget_exceeded".into()
        )));
        assert_eq!(
            db.lock().unwrap().index_suffix_status,
            IndexSuffixState::RebuildNeeded
        );
        assert!(db.lock().unwrap().lazy_vector_source.is_none());
        assert_eq!(db.stats().unwrap().chunk_count, 6);
        assert_eq!(std::fs::metadata(&path).unwrap().len(), prefix);
        drop(db);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn zero_cache_why_not_invalidates_the_source_changed_after_normal_search() {
        let path = query_preflight_fixture("why-not-text-source-changed", "needle alpha beta");
        let db = Database::open_with_options(
            &path,
            DatabaseOpenOptions {
                read_only: true,
                cache_budget_mb: Some(0),
                ..Default::default()
            },
        )
        .unwrap();
        let chunk = db
            .collection("first")
            .unwrap()
            .get_chunk_by_external_id("chunk-1")
            .unwrap();
        let context = db.json_context();
        let before = context.available_workspace_bytes();
        let (prefix, source_bytes, signature) = {
            let mut inner = db.lock().unwrap();
            let end = inner.storage.critical_end_offset();
            let changed = path.clone();
            inner.before_why_not_ranks = Some(Box::new(move || {
                std::fs::OpenOptions::new()
                    .write(true)
                    .open(changed)
                    .unwrap()
                    .set_len(end)
                    .unwrap();
            }));
            (
                end,
                inner.lazy_vector_source.as_ref().unwrap().reserved_bytes(),
                inner.catalog.state_signature(),
            )
        };
        let error = db
            .why_not_debug(
                "first",
                &chunk.chunk_id,
                Some("needle alpha".into()),
                None,
                JsonValue::object(),
                None,
                false,
                3,
                None,
            )
            .unwrap_err();
        assert_eq!(error.code(), ErrorCode::IndexCorruption);
        assert!(error.detail().contains(&(
            "index_unavailable_reason".into(),
            "transient_rebuild_budget_exceeded".into()
        )));
        {
            let inner = db.lock().unwrap();
            assert!(inner.before_why_not_ranks.is_none());
            assert_eq!(inner.index_suffix_status, IndexSuffixState::RebuildNeeded);
            assert!(inner.lazy_vector_source.is_none());
            assert_eq!(inner.catalog.state_signature(), signature);
        }
        assert_eq!(context.available_workspace_bytes(), before + source_bytes);
        assert_eq!(std::fs::metadata(&path).unwrap().len(), prefix);
        drop(db);
        fs::remove_file(path).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn check_path_reads_without_write_permission() {
        use std::os::unix::fs::PermissionsExt;

        let path = temp_path("readonly-check");
        {
            let _db = Database::create(&path, DatabaseConfig::default()).unwrap();
        }
        let mut perms = std::fs::metadata(&path).unwrap().permissions();
        perms.set_mode(0o444);
        std::fs::set_permissions(&path, perms).unwrap();

        let report = Database::check_path(
            &path,
            CheckOptions {
                verify_indexes: true,
            },
        )
        .unwrap();
        assert_eq!(report.status, "ok");
        assert!(report.critical_end_offset > 0);
        assert_eq!(report.index_suffix_status, "missing");
        assert_eq!(report.index_check, "skipped");
        assert_eq!(
            report.index_check_reason.as_deref(),
            Some("persistent_index_checkpoint_missing")
        );
        assert_eq!(
            report
                .to_json()
                .get("critical_end_offset")
                .and_then(JsonValue::as_f64)
                .map(|value| value > 0.0),
            Some(true)
        );
        assert_eq!(
            report
                .to_json()
                .get("index_check_reason")
                .and_then(JsonValue::as_str),
            Some("persistent_index_checkpoint_missing")
        );
        let report = Database::check_path(
            &path,
            CheckOptions {
                verify_indexes: false,
            },
        )
        .unwrap();
        assert_eq!(report.index_check, "disabled");
        assert!(report.index_check_reason.is_none());

        let mut perms = std::fs::metadata(&path).unwrap().permissions();
        perms.set_mode(0o644);
        std::fs::set_permissions(&path, perms).unwrap();
        let _ = std::fs::remove_file(path);
    }
    #[test]
    fn supplementary_text_fallback_checks_budget_and_stays_pinned_after_index_recovery() {
        let path = query_preflight_fixture("supplement-fallback-pinned", "query text");
        let original = fs::read(&path).unwrap();
        let db = Database::open_with_options(
            &path,
            DatabaseOpenOptions {
                read_only: true,
                cache_budget_mb: Some(0),
                ..Default::default()
            },
        )
        .unwrap();
        let vector = QueryEmbeddingInput {
            model_name: Some("m".into()),
            dim: 2,
            vector: vec![1.0, 0.0],
        };
        let mut query_context = db
            .collection("first")
            .unwrap()
            .search()
            .text("query")
            .query_embedding(vector.clone())
            .execute()
            .unwrap()
            .query_context;
        let context = db.json_context();
        let mut inner = db.lock().unwrap();
        inner.text_index = TextIndex::default();
        assert_eq!(
            inner.index_suffix_status,
            IndexSuffixState::PersistentCheckpoint
        );
        assert!(inner.lazy_vector_source.is_some());
        let collection = inner.catalog.collection_by_name("first").unwrap();
        let id = inner
            .catalog
            .live_chunks()
            .find(|chunk| chunk.collection_id == collection.collection_id)
            .unwrap()
            .chunk_id
            .clone();
        let expected_bytes = inner
            .catalog
            .live_chunks()
            .filter(|chunk| chunk.collection_id == collection.collection_id)
            .map(|chunk| {
                inner
                    .catalog
                    .record_ref_for_chunk(&chunk.chunk_id)
                    .unwrap()
                    .payload
                    .len
            })
            .sum::<u64>()
            + inner.catalog.record_ref_for_chunk(&id).unwrap().payload.len;
        let mut candidates = VectorCandidates::new(1, &context);
        candidates.push(&id, 1.0, 1.0).unwrap();
        let candidates = candidates.finish();
        let empty_text = Bm25Hits::empty(&context);
        let available = context.available_workspace_bytes();
        fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(inner.storage.critical_end_offset())
            .unwrap();
        let mut work = VectorWork::default();
        let mut bytes = 0;
        let mut pinned = false;
        query_context.transient_rebuild.budget_mb = 0;
        let error = supplement_scores(
            &inner,
            collection,
            Some("query"),
            true,
            Some(&vector),
            &empty_text,
            &candidates,
            &query_context,
            &context,
            &mut work,
            &mut bytes,
            &mut pinned,
        )
        .err()
        .unwrap();
        assert_eq!(error.code(), ErrorCode::IndexCorruption);
        assert!(pinned);
        assert_eq!(bytes, 0); // Reject before any source-data scan.
        assert_eq!(context.available_workspace_bytes(), available);
        query_context.transient_rebuild.budget_mb = 16;
        let first = supplement_scores(
            &inner,
            collection,
            Some("query"),
            true,
            Some(&vector),
            &empty_text,
            &candidates,
            &query_context,
            &context,
            &mut work,
            &mut bytes,
            &mut pinned,
        )
        .unwrap();
        assert!(first.invalidated_text && first.text[0].score > 0.0);
        assert_eq!(bytes, expected_bytes);
        let score = first.text[0].score.to_bits();
        drop(first);
        assert_eq!(context.available_workspace_bytes(), available);
        // Restore exactly the same checkpoint before the next attempt. The query
        // must finish its pinned snapshot path, rather than hide earlier fallback.
        fs::write(&path, &original).unwrap();
        let second = supplement_scores(
            &inner,
            collection,
            Some("query"),
            true,
            Some(&vector),
            &empty_text,
            &candidates,
            &query_context,
            &context,
            &mut work,
            &mut bytes,
            &mut pinned,
        )
        .unwrap();
        assert!(second.invalidated_text && pinned);
        assert_eq!(second.text[0].score.to_bits(), score);
        assert_eq!(bytes, 2 * expected_bytes);
        assert_eq!(work, VectorWork::default());
        drop(second);
        assert_eq!(context.available_workspace_bytes(), available);
        drop((candidates, empty_text, inner));
        db.close().unwrap();
        assert_eq!(fs::read(&path).unwrap(), original);
        fs::remove_file(path).unwrap();
    }
}
