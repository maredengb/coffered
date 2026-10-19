use crate::physical_file::ManagedFile;
use std::collections::{BTreeMap, BTreeSet};
#[cfg(test)]
use std::fs;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::error::{Error, ErrorCode, Result};
use crate::hash::{checksum64, crc32c as checksum32, Crc32c};
#[cfg(test)]
use crate::json::object_from_pairs;
use crate::json::{required_string, required_usize, JsonBuilder, JsonValue};
use crate::lock::{DataFileLock, FileLock};
use crate::model::{Chunk, CollectionInfo, DistanceMetric, Document, Embedding, Tombstone};
use crate::profile::Durability;
use crate::time;

mod checkpoint_writer;
mod codec;
mod index_source;
mod owned;
mod payload;
mod prepared;
mod recovery;
mod vector_cache;
#[cfg(test)]
pub(crate) use checkpoint_writer::text_spool::TextSpool;
pub(crate) use checkpoint_writer::text_spool::{
    StagedText, TextCheckpoint, TextExtent, TextStaging,
};
pub(crate) use checkpoint_writer::{Segment as CheckpointSegment, VectorRecordSegment};
pub(crate) use index_source::{IndexSegmentSource, LazyVectorCheckpoint};
pub(crate) use owned::{
    EncodedRecordBatch, OwnedRecordBatch, ReclaimingRecordBatch, RecordReservation,
};
pub(crate) use payload::{
    copy_collection_in, copy_document_in, record_workspace_bytes, RetainedChunk, RetainedEmbedding,
};
pub(crate) use prepared::{
    BatchRecordKind, PreparedBatch, PreparedRecords, PreparedRecordsBuilder,
};
pub(crate) use vector_cache::{CachedVectorSegment, VectorCache};

const MAGIC: &[u8; 8] = b"COFFERDB";
pub const FORMAT_VERSION: u16 = 1;
const HEADER_LEN: usize = 58;
const RECORD_HEADER_LEN: usize = 50;
pub(crate) const COMMIT_RECORD_BYTES: u64 =
    (RECORD_HEADER_LEN + codec::COMMIT_PAYLOAD_BYTES) as u64;

#[cfg(test)]
pub(crate) fn encoded_record_len(record: &RecordPayload) -> Result<u64> {
    codec::encoded_len(record)?
        .checked_add(RECORD_HEADER_LEN as u64)
        .ok_or_else(|| Error::new(ErrorCode::FileFormatError, "record length overflow"))
}
const LEGACY_HEADER_LEN: usize = 42;
const LEGACY_RECORD_HEADER_LEN: usize = 40;
const INDEX_CHECKPOINT_MAGIC: &[u8; 8] = b"CFIDXCHK";
const INDEX_COMPLETE_MARKER: &[u8; 4] = b"CFOK";
const INDEX_RECORD_MAGIC: &[u8; 4] = b"CFIR";
const INDEX_CHECKPOINT_VERSION: u16 = 1;
const INDEX_CHECKPOINT_HEADER_LEN: usize = 105;
const INDEX_CHECKPOINT_TRAILER_LEN: usize = 10;
const INDEX_COVERAGE_FULL_VISIBLE_SNAPSHOT: u8 = 1;
const INDEX_RECORD_HEADER_LEN: usize = 50;
const INDEX_EXTENT_ENTRY_LEN: usize = 46;
const INDEX_COMMIT_METADATA_LEN: usize = 58;
const INDEX_COMMIT_METADATA: u16 = 0x8001;
const INDEX_TEXT_SEGMENT: u16 = 0x8002;
const INDEX_VECTOR_SEGMENT: u16 = 0x8003;
const TEXT_INDEX_KIND_BM25: u16 = 1;
const VECTOR_INDEX_KIND_EXACT_F32: u16 = 2;
const INDEX_SEGMENT_SCHEMA_VERSION: u16 = 1;
const TEXT_SEGMENT_SCHEMA_VERSION: u16 = 2;
const DEFAULT_EN_TOKENIZER_ID: u16 = 1;
const MAX_INDEX_CHECKPOINT_PAYLOAD_LEN: u64 = 512 * 1024 * 1024;

const COLLECTION_CREATE: u16 = 1;
const COLLECTION_DROP: u16 = 2;
const DOCUMENT_PUT: u16 = 3;
const CHUNK_PUT: u16 = 4;
const EMBEDDING_PUT: u16 = 5;
const TOMBSTONE: u16 = 6;
const COMMIT: u16 = 7;
const COMPACT_MARKER: u16 = 8;

#[derive(Clone, Debug)]
pub enum RecordPayload {
    CollectionCreate(CollectionInfo),
    CollectionDrop {
        collection_id: String,
        name: String,
        dropped_at: String,
        reason: Option<String>,
    },
    DocumentPut(Document),
    ChunkPut(Chunk),
    EmbeddingPut(Embedding),
    Tombstone(Tombstone),
    Commit(CommitPayload),
    CompactMarker(CompactMarker),
}

#[derive(Clone, Debug)]
pub struct CommitPayload {
    pub batch_id: u64,
    pub first_sequence_no: u64,
    pub last_sequence_no: u64,
    pub record_count: usize,
    pub batch_checksum: u64,
    pub committed_at: String,
}

#[derive(Clone, Debug)]
pub struct CompactMarker {
    pub compact_generation: u64,
    pub source_last_commit_sequence_no: u64,
    pub compacted_at: String,
    pub records_rewritten: usize,
    pub id_high_watermarks: BTreeMap<String, u64>,
    pub compacted_main_checksum: u32,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum FileCodec {
    Legacy,
    #[default]
    Binary,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PayloadRef {
    pub offset: u64,
    pub len: u64,
    pub checksum: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RecordRef {
    pub offset: u64,
    pub record_type: &'static str,
    pub batch_id: u64,
    pub sequence_no: u64,
    pub payload: PayloadRef,
}

#[derive(Clone, Debug)]
pub struct CommittedRecord {
    pub payload: RecordPayload,
    pub record_ref: RecordRef,
    payload_checksum: u64,
}

#[derive(Debug)]
pub struct AppendSummary {
    pub commit: CommitPayload,
    pub commit_sequence_no: u64,
    pub records: Vec<CommittedRecord>,
    _commit_work: Option<crate::json::WorkspaceReservation>,
}

#[derive(Debug, Default)]
#[cfg_attr(test, derive(Clone))]
pub struct ScanReport {
    codec: FileCodec,
    #[cfg(test)]
    pub records: Vec<CommittedRecord>,
    pub findings: Vec<CheckIssue>,
    pub index_suffix: Option<IndexSuffixReport>,
    #[cfg(test)]
    pub index_checkpoint: Option<IndexCheckpointPayload>,
    pub lazy_vector_checkpoint: Option<LazyVectorCheckpoint>,
    pub records_scanned: usize,
    pub commits: usize,
    pub last_valid_commit: u64,
    pub last_valid_commit_sequence_no: u64,
    pub critical_end_offset: u64,
    pub max_batch_id: u64,
    pub max_sequence_no: u64,
}

#[derive(Clone, Debug)]
pub struct CheckIssue {
    pub severity: String,
    pub code: String,
    pub message: String,
    pub stream: String,
    pub visibility_impact: String,
    pub offset: Option<u64>,
    pub extent_id: Option<String>,
    pub record_type: Option<String>,
    pub batch_id: Option<u64>,
    pub repair_available: bool,
    pub suggested_command: Option<String>,
}

#[cfg(test)]
#[derive(Clone, Debug)]
pub struct IndexCheckpointPayload {
    pub metadata: IndexCheckpointMetadata,
    pub text_segments: Vec<TextIndexSegmentPayload>,
    pub vector_segments: Vec<VectorIndexSegmentPayload>,
}

#[derive(Clone, Debug)]
pub struct IndexCheckpointMetadata {
    pub covered_last_batch_id: u64,
    pub covered_last_commit_sequence_no: u64,
    pub coverage_hash: u32,
    pub collection_count: usize,
    pub live_document_count: usize,
    pub live_chunk_count: usize,
    pub live_embedding_count: usize,
    pub text_segment_count: usize,
    pub vector_segment_count: usize,
    pub build_mode: String,
}

#[cfg(test)]
#[derive(Clone, Debug)]
pub struct TextIndexSegmentPayload {
    pub collection_id: String,
    #[cfg(test)]
    pub covered_commit_sequence_no: u64,
    pub chunks: Vec<TextIndexChunkEntry>,
    pub terms: Vec<TextIndexTermEntry>,
}

#[cfg(test)]
#[derive(Clone, Debug)]
pub struct TextIndexChunkEntry {
    pub chunk_id: String,
    #[cfg(test)]
    pub document_id: String,
    pub token_count: usize,
}

#[cfg(test)]
#[derive(Clone, Debug)]
pub struct TextIndexTermEntry {
    pub term: String,
    pub postings: Vec<TextIndexPostingEntry>,
}

#[cfg(test)]
#[derive(Clone, Debug)]
pub struct TextIndexPostingEntry {
    pub chunk_id: String,
    pub term_frequency: usize,
}

#[derive(Clone, Debug)]
pub struct VectorIndexSegmentPayload {
    pub collection_id: String,
    pub covered_commit_sequence_no: u64,
    pub dim: usize,
    pub distance_metric: DistanceMetric,
    pub embedding_model: Option<String>,
    pub vectors_normalized: bool,
    pub vectors: Vec<VectorIndexEntry>,
}

impl VectorIndexSegmentPayload {
    pub(crate) fn matches_collection(&self, collection: &CollectionInfo) -> bool {
        self.collection_id == collection.collection_id
            && self.dim == collection.embedding_dim
            && self.distance_metric == collection.distance_metric
            && self.embedding_model == collection.embedding_model
            && self.vectors_normalized == collection.vectors_normalized
    }
}

#[derive(Clone, Debug)]
pub struct VectorIndexEntry {
    pub chunk_id: String,
    pub embedding_id: String,
    pub vector: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct IndexSuffixReport {
    pub status: String,
    pub offset: u64,
    pub checkpoint_generation: u64,
    pub covered_critical_end_offset: u64,
    pub covered_last_batch_id: u64,
    pub covered_last_commit_sequence_no: u64,
    pub coverage_hash: u32,
    pub collection_count: usize,
    pub live_document_count: usize,
    pub live_chunk_count: usize,
    pub live_embedding_count: usize,
    pub text_segment_count: usize,
    pub vector_segment_count: usize,
    pub index_record_count: usize,
    pub index_extent_count: usize,
    pub payload_len: u64,
    pub build_mode: String,
}

pub struct Storage {
    codec: FileCodec,
    write_failed: bool,
    _lock: Option<FileLock>,
    path: PathBuf,
    file: ManagedFile,
    critical_end_offset: u64,
    next_batch_id: u64,
    next_sequence_no: u64,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum RewriteStage {
    BeforeValidate,
    AfterRename,
}

pub(crate) struct RewritePlan {
    pub records: usize,
    pub batch_record_limit: usize,
    pub batch_byte_target: u64,
    pub temporary_disk: std::sync::Arc<crate::temporary::TemporaryDiskBudget>,
}

#[cfg(test)]
impl RewritePlan {
    fn single_batch(records: usize) -> Self {
        Self {
            records,
            batch_record_limit: records.max(1),
            batch_byte_target: u64::MAX,
            temporary_disk: Default::default(),
        }
    }
}

impl Storage {
    #[cfg(test)]
    pub fn create(path: impl AsRef<Path>) -> Result<Self> {
        Self::create_in(path, &crate::json::JsonOperationContext::standalone())
    }

    pub(crate) fn create_in(
        path: impl AsRef<Path>,
        context: &crate::json::JsonOperationContext,
    ) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let mut lock = FileLock::acquire_exclusive(&path)?;
        let file = ManagedFile::create_in(context, || create_storage_file(&path))?;
        drop(lock.replace_data_lock(DataFileLock::acquire(&file, &path)?));
        let path = path.canonicalize()?;
        Ok(Self {
            codec: FileCodec::Binary,
            write_failed: false,
            _lock: Some(lock),
            path,
            file,
            critical_end_offset: HEADER_LEN as u64,
            next_batch_id: 1,
            next_sequence_no: 1,
        })
    }

    #[cfg(test)]
    pub fn open(path: impl AsRef<Path>) -> Result<(Self, ScanReport)> {
        let mut records = Vec::new();
        let (storage, mut report) = Self::open_with_records(path, false, u64::MAX, |record| {
            records.push(record);
            Ok(())
        })?;
        report.records = records;
        Ok((storage, report))
    }

    #[cfg(test)]
    pub(crate) fn open_with_records(
        path: impl AsRef<Path>,
        read_only: bool,
        index_budget_bytes: u64,
        mut visit: impl FnMut(CommittedRecord) -> Result<()>,
    ) -> Result<(Self, ScanReport)> {
        Self::open_scanned(
            path,
            read_only,
            index_budget_bytes.into(),
            &crate::json::JsonOperationContext::standalone(),
            |record, _| visit(record),
        )
    }

    pub(crate) fn open_with_retained_records(
        path: impl AsRef<Path>,
        read_only: bool,
        index_budget_bytes: u64,
        context: &crate::json::JsonOperationContext,
        visit: impl FnMut(CommittedRecord, &mut RecordReservation) -> Result<()>,
    ) -> Result<(Self, ScanReport)> {
        Self::open_scanned(
            path,
            read_only,
            IndexReadOptions {
                budget_bytes: index_budget_bytes,
                mode: IndexReadMode::LazyIndexes,
            },
            context,
            visit,
        )
    }

    pub(crate) fn open_summary_with_records(
        path: impl AsRef<Path>,
        read_only: bool,
        mut visit: impl FnMut(CommittedRecord) -> Result<()>,
    ) -> Result<(Self, ScanReport)> {
        Self::open_summary_with_retained_records(
            path,
            read_only,
            &crate::json::JsonOperationContext::standalone(),
            |record, _| visit(record),
        )
    }

    pub(crate) fn open_summary_with_retained_records(
        path: impl AsRef<Path>,
        read_only: bool,
        context: &crate::json::JsonOperationContext,
        visit: impl FnMut(CommittedRecord, &mut RecordReservation) -> Result<()>,
    ) -> Result<(Self, ScanReport)> {
        Self::open_scanned(
            path,
            read_only,
            IndexReadOptions {
                budget_bytes: u64::MAX,
                mode: IndexReadMode::Summary,
            },
            context,
            visit,
        )
    }

    fn open_scanned(
        path: impl AsRef<Path>,
        read_only: bool,
        index_options: IndexReadOptions,
        context: &crate::json::JsonOperationContext,
        mut visit: impl FnMut(CommittedRecord, &mut RecordReservation) -> Result<()>,
    ) -> Result<(Self, ScanReport)> {
        let path = path.as_ref().canonicalize()?;
        let mut lock = if read_only {
            None
        } else {
            Some(FileLock::acquire_exclusive(&path)?)
        };
        let mut file = ManagedFile::open_in(context, || {
            Ok(OpenOptions::new()
                .read(true)
                .write(!read_only)
                .open(&path)?)
        })?;
        if let Some(lock) = &mut lock {
            drop(lock.replace_data_lock(DataFileLock::acquire(&file, &path)?));
        }
        let scan = scan_open_file_with_records_in(&mut file, index_options, context, &mut visit)?;
        Ok((
            Self {
                codec: scan.codec,
                write_failed: false,
                _lock: lock,
                path,
                file,
                critical_end_offset: scan.critical_end_offset,
                next_batch_id: next_counter(scan.last_valid_commit)?,
                next_sequence_no: next_counter(scan.last_valid_commit_sequence_no)?,
            },
            scan,
        ))
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub(crate) fn require_writer(&self) -> Result<()> {
        if self._lock.is_none() {
            return Err(Error::new(
                ErrorCode::PermissionDenied,
                "database handle is read-only",
            ));
        }
        if self.write_failed {
            return Err(Error::new(ErrorCode::RecoveryError,
                "writer state is uncertain after a failed critical write; close and reopen the database")
                .with_detail("reason", "writer_reopen_required"));
        }
        self.file.require_writer()
    }

    pub fn file_size(&self) -> Result<u64> {
        Ok(self.file.metadata()?.len())
    }

    pub(crate) fn flush_committed(&mut self) -> Result<()> {
        self.flush_committed_with(File::sync_data)
    }

    pub(crate) fn flush_committed_with(
        &mut self,
        sync: impl FnOnce(&File) -> std::io::Result<()>,
    ) -> Result<()> {
        self.file.flush()?;
        if self._lock.is_some() {
            sync(&self.file)?;
        }
        Ok(())
    }

    pub fn critical_end_offset(&self) -> u64 {
        self.critical_end_offset
    }

    pub fn truncate_to_critical_end(&mut self, durability: Durability) -> Result<(u64, u64)> {
        self.require_writer()?;
        let before = self.file.metadata()?.len();
        if before > self.critical_end_offset {
            self.file.set_len(self.critical_end_offset)?;
            self.file.seek(SeekFrom::Start(self.critical_end_offset))?;
            self.file.flush()?;
            if durability == Durability::Commit {
                self.file.sync_data()?;
            }
        }
        let after = self.file.metadata()?.len();
        Ok((before, after))
    }

    #[cfg(test)]
    pub fn read_payload(&self, record_ref: &RecordRef) -> Result<RecordPayload> {
        self.payload_reader()?.read_payload(record_ref)
    }

    pub fn payload_reader(&self) -> Result<PayloadReader> {
        Ok(PayloadReader {
            codec: self.codec,
            file: self.file.try_clone()?,
        })
    }

    #[cfg(test)]
    pub fn append_batch(
        &mut self,
        records: &[RecordPayload],
        durability: Durability,
    ) -> Result<AppendSummary> {
        self.append_owned_batch(records.to_vec(), durability)
    }

    #[cfg(test)]
    pub fn append_owned_batch(
        &mut self,
        records: Vec<RecordPayload>,
        durability: Durability,
    ) -> Result<AppendSummary> {
        let context = crate::json::JsonOperationContext::standalone();
        let mut batch = OwnedRecordBatch::new(&context);
        for record in records {
            batch.push(record)?;
        }
        self.append_owned_batch_in(batch.prepare_encoding(|| false)?, durability, || false)
            .map(|(append, _)| append)
    }

    pub(crate) fn append_owned_batch_in(
        &mut self,
        batch: EncodedRecordBatch,
        durability: Durability,
        reclaim: impl FnOnce() -> bool,
    ) -> Result<(AppendSummary, RecordReservation)> {
        self.require_writer()?;
        let encoding = batch.encoding.admit(reclaim)?;
        let (records, mut work, mut payloads, commit_timestamp) = batch.batch.into_parts();
        let context = work.context().clone();
        let append = self.append_record_batch(
            records.into_iter().map(WriteRecord::planned),
            durability,
            &context,
            commit_timestamp,
            Some(encoding),
        )?;
        work.retain(append.records.capacity() * std::mem::size_of::<CommittedRecord>());
        payloads.merge_workspace(work);
        Ok((append, payloads))
    }

    pub(crate) fn append_prepared_batch(
        &mut self,
        batch: PreparedBatch<'_>,
        durability: Durability,
    ) -> Result<(
        AppendSummary,
        (
            crate::json::JsonReservation,
            crate::json::WorkspaceReservation,
        ),
    )> {
        batch.validate()?;
        let context = batch.records_work.context().clone();
        let append = self.append_record_batch(
            batch.records.into_iter().map(|record| WriteRecord {
                payload: record.payload,
                prepared: Some((batch.source, record.source)),
                encoding: RecordEncoding::Batch,
            }),
            durability,
            &context,
            Some(batch.commit_timestamp),
            None,
        )?;
        let mut records_work = batch.records_work;
        // The input array and copy buffer are gone; the returned summaries
        // remain charged until the caller applies/drops the committed batch.
        records_work.retain(append.records.capacity() * std::mem::size_of::<CommittedRecord>());
        records_work.merge(batch.strings_work);
        Ok((append, (batch.metadata_work, records_work)))
    }

    fn append_record_batch<'a>(
        &mut self,
        records: impl ExactSizeIterator<Item = WriteRecord<'a>>,
        durability: Durability,
        context: &crate::json::JsonOperationContext,
        commit_timestamp: Option<crate::json::ReservedString>,
        encoding: Option<codec::BatchEncoding>,
    ) -> Result<AppendSummary> {
        self.require_writer()?;
        if self.codec == FileCodec::Legacy {
            return Err(Error::new(
                ErrorCode::UnsupportedVersion,
                "legacy experimental files must be explicitly compacted before writing",
            ));
        }
        let record_count = records.len();
        if record_count == 0 {
            return Err(Error::new(
                ErrorCode::FileFormatError,
                "cannot append an empty batch",
            ));
        }
        let mut writer = BatchWriter::new(
            &mut self.file,
            &mut self.next_batch_id,
            &mut self.next_sequence_no,
            record_count,
            true,
            context,
            commit_timestamp,
        )?;
        writer.raw_encoding = encoding;
        let append = match (|| {
            if writer.batch_start > self.critical_end_offset {
                writer.file.set_len(self.critical_end_offset)?;
            }
            writer
                .file
                .seek(SeekFrom::Start(self.critical_end_offset))?;
            writer.batch_start = self.critical_end_offset;
            writer.append_all(records.map(Ok), durability)
        })() {
            Ok(append) => append,
            Err(err) => {
                // A failed flush/fsync can still leave a complete commit on disk.
                self.write_failed = true;
                return Err(err.with_detail("reason", "writer_reopen_required"));
            }
        };
        self.critical_end_offset = self.file.stream_position()?;
        Ok(append)
    }

    #[cfg(test)]
    pub fn write_index_checkpoint(
        &mut self,
        checkpoint: &IndexCheckpointPayload,
        durability: Durability,
    ) -> Result<(IndexSuffixReport, IndexSegmentSource)> {
        use std::borrow::Cow;
        let segments = checkpoint
            .text_segments
            .iter()
            .map(|segment| Ok(CheckpointSegment::Text(Cow::Borrowed(segment))))
            .chain(
                checkpoint
                    .vector_segments
                    .iter()
                    .map(|segment| Ok(CheckpointSegment::Vector(Cow::Borrowed(segment)))),
            );
        self.write_index_segments(&checkpoint.metadata, segments, |_, _| {}, durability)
    }

    #[cfg(test)]
    pub(crate) fn write_index_segments<'a>(
        &mut self,
        checkpoint: &IndexCheckpointMetadata,
        segments: impl IntoIterator<Item = Result<CheckpointSegment<'a>>>,
        completed: impl FnMut(CheckpointSegment<'a>, u64),
        durability: Durability,
    ) -> Result<(IndexSuffixReport, IndexSegmentSource)> {
        self.write_index_segments_in(
            checkpoint,
            segments,
            completed,
            durability,
            &crate::json::JsonOperationContext::standalone(),
            true,
        )
    }

    pub(crate) fn write_index_segments_in<'a>(
        &mut self,
        checkpoint: &IndexCheckpointMetadata,
        segments: impl IntoIterator<Item = Result<CheckpointSegment<'a>>>,
        completed: impl FnMut(CheckpointSegment<'a>, u64),
        durability: Durability,
        context: &crate::json::JsonOperationContext,
        allow_optional_memory: bool,
    ) -> Result<(IndexSuffixReport, IndexSegmentSource)> {
        self.require_writer()?;
        if self.codec == FileCodec::Legacy {
            return Err(Error::new(
                ErrorCode::UnsupportedVersion,
                "legacy experimental files must be explicitly compacted before writing",
            ));
        }
        let directory = checkpoint_writer::reserve_directory(checkpoint, context)?;
        if self.file.metadata()?.len() > self.critical_end_offset {
            self.file.set_len(self.critical_end_offset)?;
        }
        let (payload_len, vectors) = checkpoint_writer::write_checkpoint_segments_in(
            &mut self.file,
            checkpoint,
            segments,
            self.critical_end_offset,
            completed,
            directory,
            allow_optional_memory,
        )?;
        self.file.flush()?;
        #[cfg(test)]
        tests::crash_boundary("index_suffix_flushed")?;
        if durability == Durability::Commit {
            self.file.sync_data()?;
        }
        let segment_record_count = checked_segment_record_count(checkpoint)? as usize;
        let status = if segment_record_count == 0 {
            "metadata_record_only"
        } else {
            "persistent_checkpoint"
        };
        Ok((
            IndexSuffixReport {
                status: status.into(),
                offset: self.critical_end_offset,
                checkpoint_generation: self.critical_end_offset,
                covered_critical_end_offset: self.critical_end_offset,
                covered_last_batch_id: checkpoint.covered_last_batch_id,
                covered_last_commit_sequence_no: checkpoint.covered_last_commit_sequence_no,
                coverage_hash: checkpoint.coverage_hash,
                collection_count: checkpoint.collection_count,
                live_document_count: checkpoint.live_document_count,
                live_chunk_count: checkpoint.live_chunk_count,
                live_embedding_count: checkpoint.live_embedding_count,
                text_segment_count: checkpoint.text_segment_count,
                vector_segment_count: checkpoint.vector_segment_count,
                index_record_count: 1 + segment_record_count,
                index_extent_count: 1 + segment_record_count,
                payload_len,
                build_mode: checkpoint.build_mode.clone(),
            },
            vectors,
        ))
    }

    #[cfg(test)]
    pub fn rewrite_with_records(
        &mut self,
        records: &[RecordPayload],
        durability: Durability,
    ) -> Result<ScanReport> {
        self.rewrite_with_records_with_hook(records, durability, |_| Ok(()))
    }

    #[cfg(test)]
    fn rewrite_with_records_with_hook<F>(
        &mut self,
        records: &[RecordPayload],
        durability: Durability,
        before_validate: F,
    ) -> Result<ScanReport>
    where
        F: FnOnce(&Path) -> Result<()>,
    {
        let mut collected = Vec::new();
        let mut before_validate = Some(before_validate);
        let mut scan = self.rewrite_with_iter_with_hook(
            records.iter().cloned().map(Ok),
            RewritePlan::single_batch(records.len()),
            durability,
            |stage, path| {
                if stage == RewriteStage::BeforeValidate {
                    before_validate.take().unwrap()(path)?;
                }
                Ok(())
            },
            |record| {
                collected.push(record);
                Ok(())
            },
        )?;
        scan.records = collected;
        Ok(scan)
    }

    #[cfg(test)]
    fn rewrite_with_iter_with_hook<I, F>(
        &mut self,
        records: I,
        plan: RewritePlan,
        durability: Durability,
        hook: F,
        mut visit: impl FnMut(CommittedRecord) -> Result<()>,
    ) -> Result<ScanReport>
    where
        I: IntoIterator<Item = Result<RecordPayload>>,
        F: FnMut(RewriteStage, &Path) -> Result<()>,
    {
        self.rewrite_records_with_hook(
            records
                .into_iter()
                .map(|record| record.map(WriteRecord::encoded)),
            plan,
            durability,
            &crate::json::JsonOperationContext::standalone(),
            || false,
            hook,
            |record, _| visit(record),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn rewrite_records_with_hook<'a, I, F>(
        &mut self,
        records: I,
        plan: RewritePlan,
        durability: Durability,
        context: &crate::json::JsonOperationContext,
        reclaim: impl FnOnce() -> bool,
        mut hook: F,
        mut visit: impl FnMut(CommittedRecord, &mut RecordReservation) -> Result<()>,
    ) -> Result<ScanReport>
    where
        I: IntoIterator<Item = Result<WriteRecord<'a>>>,
        F: FnMut(RewriteStage, &Path) -> Result<()>,
    {
        self.require_writer()?;
        if plan.batch_record_limit == 0 || plan.batch_byte_target == 0 {
            return Err(Error::new(
                ErrorCode::FileFormatError,
                "invalid compaction batch limits",
            ));
        }
        let record_count = plan.records;
        let directory = self
            .path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        let (file, mut temporary) = crate::temporary::create_budgeted_temporary_file_reclaiming(
            directory,
            "compact",
            "tmp",
            plan.temporary_disk,
            context,
            "operation_scratch",
            reclaim,
        )?;
        // Keep the file's drop before the guard, including during unwinding.
        let mut file = file;
        temporary.reserve_disk(HEADER_LEN as u64)?;
        write_header(&mut file)?;
        let mut records = records.into_iter();
        let mut lookahead = None;
        let mut remaining = record_count;
        let mut next_batch_id = self.next_batch_id;
        let mut next_sequence_no = self.next_sequence_no;
        let mut compact_crc = Crc32c::default();
        while remaining > 0 {
            let limit = remaining.min(plan.batch_record_limit);
            let mut writer = BatchWriter::new(
                &mut file,
                &mut next_batch_id,
                &mut next_sequence_no,
                limit,
                false,
                context,
                None,
            )?;
            writer.compact_crc = Some(&mut compact_crc);
            writer.temporary_disk = Some(&mut temporary);
            let mut retained_json_bytes = 0usize;
            for _ in 0..limit {
                // The next source decode shares this ledger and must not compete
                // with an optional buffer retained from the preceding record.
                writer.encoding.release_buffer();
                let record = lookahead
                    .take()
                    .map(Ok)
                    .or_else(|| records.next())
                    .ok_or_else(|| {
                        Error::new(
                            ErrorCode::FileFormatError,
                            "compaction record count mismatch",
                        )
                    })??;
                let json_bytes = match &record.encoding {
                    RecordEncoding::Stored(_, work) => work.json_bytes(),
                    _ => 0,
                };
                // Leave half the JSON workspace for input and parser scratch during validation.
                // An indivisible value larger than this batch target is validated on its own.
                if writer.written > 0
                    && retained_json_bytes.saturating_add(json_bytes)
                        > crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES / 2
                {
                    lookahead = Some(record);
                    break;
                }
                retained_json_bytes = retained_json_bytes.saturating_add(json_bytes);
                writer.append(record)?;
                remaining -= 1;
                // A record is indivisible; finish at the first boundary reaching the target.
                if writer.encoded_bytes >= plan.batch_byte_target {
                    break;
                }
            }
            // The private file is synced once, after all complete batches, before publication.
            writer.finish(Durability::Fast)?;
            #[cfg(test)]
            if remaining > 0 {
                tests::crash_boundary("compact_between_batches")?;
            }
        }
        if lookahead.is_some() || records.next().is_some() {
            return Err(Error::new(
                ErrorCode::FileFormatError,
                "compaction record count mismatch",
            ));
        }
        drop(records);
        #[cfg(test)]
        tests::crash_boundary("compact_before_sync")?;
        file.sync_data()?;
        #[cfg(test)]
        tests::crash_boundary("compact_after_sync")?;
        drop(file);
        hook(RewriteStage::BeforeValidate, temporary.path())?;
        let mut validated_count = 0usize;
        let mut validation_file = File::open(temporary.path())?;
        if read_header(&mut validation_file)? != FileCodec::Binary {
            return Err(Error::new(
                ErrorCode::RecoveryError,
                "compaction output has an unexpected format",
            ));
        }
        let scan = scan_binary_file(
            &mut validation_file,
            IndexReadOptions {
                budget_bytes: u64::MAX,
                mode: IndexReadMode::Summary,
            },
            context,
            &mut |record, work| {
                validated_count += 1;
                visit(record, work)
            },
        )?;
        drop(validation_file);
        if validated_count != record_count
            || scan
                .findings
                .iter()
                .any(|issue| issue.visibility_impact == "blocks_main" || issue.stream == "critical")
        {
            return Err(Error::new(
                ErrorCode::RecoveryError,
                "compaction output failed validation",
            ));
        }
        let next_batch_id = next_counter(scan.last_valid_commit)?;
        let next_sequence_no = next_counter(scan.last_valid_commit_sequence_no)?;
        let replacement = ManagedFile::open_in(context, || {
            Ok(OpenOptions::new()
                .read(true)
                .write(true)
                .open(temporary.path())?)
        })?;
        let mut replacement_lock = Some(DataFileLock::acquire(&replacement, &self.path)?);
        #[cfg(test)]
        tests::crash_boundary("compact_before_rename")?;
        let writer_lock = self._lock.as_mut().expect("writer lock required");
        let write_failed = &mut self.write_failed;
        let mut previous_lock = None;
        self.file
            .publish_replacement(&mut temporary, &self.path, &mut || {
                // Publish the new lock and disable old writes before admission resumes.
                *write_failed = true;
                previous_lock = writer_lock
                    .replace_data_lock(replacement_lock.take().expect("prepared replacement lock"));
            })?;
        // Closing a managed descriptor re-enters admission; release it outside the gate.
        drop(previous_lock);
        #[cfg(test)]
        tests::crash_boundary("compact_after_rename")?;
        hook(RewriteStage::AfterRename, &self.path)?;
        if durability == Durability::Commit {
            sync_parent_dir(&self.path)?;
        }
        self.file = replacement;
        self.codec = scan.codec;
        self.critical_end_offset = scan.critical_end_offset;
        self.next_batch_id = next_batch_id;
        self.next_sequence_no = next_sequence_no;
        self.write_failed = false;
        Ok(scan)
    }
}

pub struct PayloadReader {
    codec: FileCodec,
    file: ManagedFile,
}

#[cfg(test)]
pub fn scan_file(path: impl AsRef<Path>) -> Result<ScanReport> {
    let mut file = ManagedFile::open_in(&crate::json::JsonOperationContext::standalone(), || {
        Ok(File::open(path.as_ref())?)
    })?;
    scan_open_file(&mut file, u64::MAX)
}

#[cfg(test)]
pub(crate) fn scan_file_with_records(
    path: impl AsRef<Path>,
    visit: impl FnMut(CommittedRecord) -> Result<()>,
) -> Result<ScanReport> {
    scan_file_with_records_in(
        path,
        &crate::json::JsonOperationContext::standalone(),
        visit,
    )
}

#[cfg(test)]
pub(crate) fn scan_file_with_records_in(
    path: impl AsRef<Path>,
    context: &crate::json::JsonOperationContext,
    mut visit: impl FnMut(CommittedRecord) -> Result<()>,
) -> Result<ScanReport> {
    scan_file_with_retained_records_in(path, context, |record, _| visit(record))
}

pub(crate) fn scan_file_with_retained_records_in(
    path: impl AsRef<Path>,
    context: &crate::json::JsonOperationContext,
    mut visit: impl FnMut(CommittedRecord, &mut RecordReservation) -> Result<()>,
) -> Result<ScanReport> {
    let mut file = ManagedFile::open_in(context, || Ok(File::open(path.as_ref())?))?;
    scan_open_file_with_records_in(
        &mut file,
        IndexReadOptions {
            budget_bytes: u64::MAX,
            mode: IndexReadMode::Summary,
        },
        context,
        &mut visit,
    )
}

#[cfg(test)]
fn scan_open_file(file: &mut File, index_budget_bytes: u64) -> Result<ScanReport> {
    let mut records = Vec::new();
    let mut report = scan_open_file_with_records(file, index_budget_bytes.into(), &mut |record| {
        records.push(record);
        Ok(())
    })?;
    report.records = records;
    Ok(report)
}

#[cfg(test)]
fn scan_open_file_with_records(
    file: &mut File,
    index_options: IndexReadOptions,
    visit: &mut dyn FnMut(CommittedRecord) -> Result<()>,
) -> Result<ScanReport> {
    scan_open_file_with_records_in(
        file,
        index_options,
        &crate::json::JsonOperationContext::standalone(),
        &mut |record, _| visit(record),
    )
}

fn scan_open_file_with_records_in(
    file: &mut File,
    index_options: IndexReadOptions,
    context: &crate::json::JsonOperationContext,
    visit: &mut dyn FnMut(CommittedRecord, &mut RecordReservation) -> Result<()>,
) -> Result<ScanReport> {
    file.seek(SeekFrom::Start(0))?;
    let format = read_header(file)?;
    if format == FileCodec::Binary {
        return scan_binary_file(file, index_options, context, visit);
    }
    let mut report = ScanReport {
        codec: FileCodec::Legacy,
        critical_end_offset: LEGACY_HEADER_LEN as u64,
        ..ScanReport::default()
    };
    let mut pending: BTreeMap<u64, (Vec<CommittedRecord>, RecordReservation)> = BTreeMap::new();

    loop {
        let offset = file.stream_position()?;
        let mut header = [0u8; LEGACY_RECORD_HEADER_LEN];
        match file.read_exact(&mut header) {
            Ok(()) => {}
            Err(err) if err.kind() == std::io::ErrorKind::UnexpectedEof => {
                if file.metadata()?.len() > offset {
                    let (stream, visibility_impact, message) =
                        if is_after_visible_critical_prefix(&report, offset) {
                            (
                                "index",
                                "tail_only",
                                "ignored incomplete rebuildable suffix after last valid commit",
                            )
                        } else {
                            (
                                "critical",
                                "tail_only",
                                "ignored incomplete tail record after last valid commit",
                            )
                        };
                    report.findings.push(issue(
                        "warning",
                        "corrupt_tail",
                        message,
                        stream,
                        visibility_impact,
                        Some(offset),
                        None,
                        None,
                    ));
                }
                break;
            }
            Err(err) => return Err(Error::from(err)),
        }

        if is_after_visible_critical_prefix(&report, offset)
            && &header[0..INDEX_CHECKPOINT_MAGIC.len()] == INDEX_CHECKPOINT_MAGIC
        {
            push_index_suffix_issue(
                &mut report,
                offset,
                "legacy_index_rebuild_needed",
                "legacy experimental indexes require rebuilding",
                None,
                None,
            );
            break;
        }

        let record_type = read_u16(&header[0..2]);
        let version = read_u16(&header[2..4]);
        let batch_id = read_u64(&header[4..12]);
        let sequence_no = read_u64(&header[12..20]);
        let payload_len_u64 = read_u64(&header[20..28]);
        let payload_checksum = read_u64(&header[28..36]);
        let header_checksum = read_u32(&header[36..40]);
        if version != 1 || crate::hash::checksum32(&header[0..36]) != header_checksum {
            push_record_boundary_issue(
                &mut report,
                offset,
                "checksum_mismatch",
                "record header checksum mismatch",
                Some(record_type_name(record_type)),
                Some(batch_id),
            );
            break;
        }

        let payload_start = offset + LEGACY_RECORD_HEADER_LEN as u64;
        let remaining = file.metadata()?.len().saturating_sub(payload_start);
        if payload_len_u64 > remaining {
            report.findings.push(issue(
                "warning",
                "corrupt_tail",
                "ignored incomplete payload after last valid commit",
                "critical",
                "tail_only",
                Some(offset),
                Some(record_type_name(record_type)),
                Some(batch_id),
            ));
            break;
        }
        let payload_len = usize::try_from(payload_len_u64).map_err(|_| {
            Error::new(
                ErrorCode::FileFormatError,
                "record payload length exceeds address space",
            )
        })?;

        let mut input_work = context.reservation();
        let mut payload = Vec::new();
        input_work
            .reserve_vec(
                &mut payload,
                payload_len,
                crate::json::JsonInputKind::StoredJson,
            )
            .map_err(|err| {
                err.with_detail("record_type", record_type_name(record_type))
                    .with_detail("record_offset", offset)
            })?;
        payload.resize(payload_len, 0);
        if let Err(err) = file.read_exact(&mut payload) {
            if err.kind() == std::io::ErrorKind::UnexpectedEof {
                report.findings.push(issue(
                    "warning",
                    "corrupt_tail",
                    "ignored incomplete payload after last valid commit",
                    "critical",
                    "tail_only",
                    Some(offset),
                    Some(record_type_name(record_type)),
                    Some(batch_id),
                ));
                break;
            }
            return Err(Error::from(err));
        }
        if checksum64(&payload) != payload_checksum {
            push_record_boundary_issue(
                &mut report,
                offset,
                "checksum_mismatch",
                "record payload checksum mismatch",
                Some(record_type_name(record_type)),
                Some(batch_id),
            );
            break;
        }

        let text = match std::str::from_utf8(&payload) {
            Ok(text) => text,
            Err(_) => {
                push_record_boundary_issue(
                    &mut report,
                    offset,
                    "file_format_error",
                    "record payload is not utf-8",
                    Some(record_type_name(record_type)),
                    Some(batch_id),
                );
                break;
            }
        };
        let (record, record_work) =
            match payload::decode_legacy_summary_in(record_type, text, context) {
                Ok(record) => record,
                Err(err)
                    if crate::json::is_limit_error(&err)
                        || matches!(
                            err.code(),
                            ErrorCode::IoError | ErrorCode::PermissionDenied
                        ) =>
                {
                    return Err(err
                        .with_detail("record_type", record_type_name(record_type))
                        .with_detail("record_offset", offset));
                }
                Err(_) => {
                    push_record_boundary_issue(
                        &mut report,
                        offset,
                        "file_format_error",
                        "invalid record JSON payload schema or type",
                        Some(record_type_name(record_type)),
                        Some(batch_id),
                    );
                    break;
                }
            };
        report.records_scanned += 1;
        report.max_batch_id = report.max_batch_id.max(batch_id);
        report.max_sequence_no = report.max_sequence_no.max(sequence_no);

        match record {
            RecordPayload::Commit(commit) => {
                let mut batch = pending.remove(&batch_id).unwrap_or_else(|| {
                    (
                        Vec::new(),
                        RecordReservation::new(
                            context.reservation(),
                            context.workspace_reservation(),
                        ),
                    )
                });
                let actual_checksum = batch
                    .0
                    .iter()
                    .fold(0xcbf29ce484222325u64, |acc, rec| acc ^ rec.payload_checksum);
                let sequence_matches =
                    committed_batch_sequence_matches(&batch.0, &commit, batch_id);
                if batch.0.len() == commit.record_count
                    && actual_checksum == commit.batch_checksum
                    && sequence_matches
                    && commit.batch_id == batch_id
                    && Some(sequence_no) == commit.last_sequence_no.checked_add(1)
                {
                    report.last_valid_commit = commit.batch_id;
                    report.last_valid_commit_sequence_no = sequence_no;
                    report.critical_end_offset =
                        offset + LEGACY_RECORD_HEADER_LEN as u64 + payload_len as u64;
                    report.commits += 1;
                    for record in batch.0 {
                        visit(record, &mut batch.1)?;
                    }
                } else {
                    let diagnostic_offset = batch
                        .0
                        .first()
                        .map(|record| record.record_ref.offset)
                        .unwrap_or(offset);
                    report.findings.push(issue(
                        "error",
                        "checksum_mismatch",
                        "commit checksum, sequence range, or record count mismatch; batch ignored",
                        "critical",
                        "blocks_main",
                        Some(diagnostic_offset),
                        Some("Commit"),
                        Some(batch_id),
                    ));
                    break;
                }
            }
            payload => {
                let record_ref = RecordRef {
                    offset,
                    record_type: record_type_name(record_type),
                    batch_id,
                    sequence_no,
                    payload: PayloadRef {
                        offset: offset + LEGACY_RECORD_HEADER_LEN as u64,
                        len: payload_len as u64,
                        checksum: payload_checksum,
                    },
                };
                let batch = pending.entry(batch_id).or_insert_with(|| {
                    (
                        Vec::new(),
                        RecordReservation::new(
                            context.reservation(),
                            context.workspace_reservation(),
                        ),
                    )
                });
                batch.1.merge(record_work);
                batch.0.push(CommittedRecord {
                    payload,
                    record_ref,
                    payload_checksum,
                });
            }
        }
    }

    if !pending.is_empty() {
        report.findings.push(issue(
            "warning",
            "uncommitted_batch",
            "ignored records without a valid commit marker",
            "critical",
            "tail_only",
            None,
            None,
            pending.keys().next().copied(),
        ));
    }

    Ok(report)
}

fn scan_binary_file(
    file: &mut File,
    index_options: IndexReadOptions,
    context: &crate::json::JsonOperationContext,
    visit: &mut dyn FnMut(CommittedRecord, &mut RecordReservation) -> Result<()>,
) -> Result<ScanReport> {
    let file_len = file.metadata()?.len();
    let mut report = ScanReport {
        critical_end_offset: HEADER_LEN as u64,
        ..ScanReport::default()
    };
    let mut pending = recovery::PendingBatch::new(context);
    let mut batch_crc = Crc32c::default();
    let mut data_crc = Crc32c::default();
    let mut id_high_watermarks = BTreeMap::new();
    let mut expected_sequence = 1u64;
    let mut expected_batch = 1u64;
    loop {
        let offset = file.stream_position()?;
        if offset == file_len {
            break;
        }
        let mut header = [0; RECORD_HEADER_LEN];
        if let Err(err) = file.read_exact(&mut header) {
            if err.kind() != std::io::ErrorKind::UnexpectedEof {
                return Err(err.into());
            }
            let stream =
                if pending.is_empty() && report.last_valid_commit > 0 && &header[..4] != b"CFRL" {
                    "index"
                } else {
                    "critical"
                };
            report.findings.push(issue(
                "warning",
                "corrupt_tail",
                "ignored incomplete tail header",
                stream,
                "tail_only",
                Some(offset),
                None,
                None,
            ));
            break;
        }
        if pending.is_empty()
            && offset == report.critical_end_offset
            && &header[..8] == INDEX_CHECKPOINT_MAGIC
        {
            file.seek(SeekFrom::Start(offset))?;
            let parsed = read_index_checkpoint_suffix_in(
                file,
                offset,
                file_len,
                report.critical_end_offset,
                report.last_valid_commit,
                report.last_valid_commit_sequence_no,
                (index_options, context),
            )?;
            match parsed {
                Ok(parsed) => {
                    for diagnostic in parsed.diagnostics {
                        let batch_id = report.last_valid_commit;
                        push_index_suffix_diagnostic_issue(
                            &mut report,
                            &diagnostic,
                            Some("IndexCheckpointMetadata"),
                            Some(batch_id),
                        );
                    }
                    report.index_suffix = Some(parsed.report);
                    #[cfg(test)]
                    {
                        report.index_checkpoint = parsed.checkpoint;
                    }
                    report.lazy_vector_checkpoint = parsed.lazy_vectors;
                }
                Err(diagnostic) => {
                    let batch_id = report.last_valid_commit;
                    push_index_suffix_diagnostic_issue(
                        &mut report,
                        &diagnostic,
                        Some("IndexCheckpointMetadata"),
                        Some(batch_id),
                    );
                }
            }
            break;
        }
        if &header[..4] != b"CFRL" {
            push_record_boundary_issue(
                &mut report,
                offset,
                "file_format_error",
                "record magic mismatch",
                Some("Unknown"),
                None,
            );
            break;
        }
        let record_type = read_u16(&header[4..6]);
        let batch_id = read_u64(&header[16..24]);
        let sequence_no = read_u64(&header[24..32]);
        let payload_len = read_u64(&header[34..42]);
        let payload_checksum = read_u32(&header[42..46]);
        if checksum32(&header[..46]) != read_u32(&header[46..50]) {
            report.findings.push(issue(
                "error",
                "checksum_mismatch",
                "critical record header checksum mismatch",
                "critical",
                "blocks_main",
                Some(offset),
                Some(record_type_name(record_type)),
                Some(batch_id),
            ));
            break;
        }
        let length = payload_len.checked_add(RECORD_HEADER_LEN as u64);
        if offset == HEADER_LEN as u64 {
            expected_batch = batch_id;
            expected_sequence = sequence_no;
        }
        if read_u16(&header[6..8]) != 1
            || read_u16(&header[32..34]) != RECORD_HEADER_LEN as u16
            || length != Some(read_u64(&header[8..16]))
            || !(1..=8).contains(&record_type)
            || sequence_no == 0
            || batch_id == 0
            || sequence_no != expected_sequence
            || batch_id != expected_batch
        {
            report.findings.push(issue(
                "error",
                "file_format_error",
                "invalid critical record length, type, version, or sequence",
                "critical",
                "blocks_main",
                Some(offset),
                Some(record_type_name(record_type)),
                Some(batch_id),
            ));
            break;
        }
        let payload_offset = offset
            .checked_add(RECORD_HEADER_LEN as u64)
            .ok_or_else(|| Error::new(ErrorCode::FileFormatError, "record offset overflow"))?;
        if payload_len > file_len.saturating_sub(payload_offset) {
            report.findings.push(issue(
                "warning",
                "corrupt_tail",
                "ignored incomplete critical payload",
                "critical",
                "tail_only",
                Some(offset),
                Some(record_type_name(record_type)),
                Some(batch_id),
            ));
            break;
        }
        let compact_prefix_checksum = data_crc.finish();
        let data_record = record_type != COMMIT;
        if data_record {
            batch_crc.update(&header);
            data_crc.update(&header);
        }
        let mut strings = context.workspace_reservation();
        let (mut input_work, mut input_error) = match PayloadInputReservation::new(
            context,
            record_type,
            payload_len,
            crate::json::JsonInputKind::StoredJson,
        ) {
            Ok(work) => (Some(work), None),
            Err(error) => (None, Some(error)),
        };
        let initial_batch_crc = batch_crc;
        let initial_data_crc = data_crc;
        let (decoded, actual_checksum, bytes_read) = loop {
            let mut reader = CheckedPayloadReader {
                source: BufReader::with_capacity(
                    input_work
                        .as_ref()
                        .map_or(0, PayloadInputReservation::capacity),
                    (&mut *file).take(payload_len),
                ),
                payload_crc: Crc32c::default(),
                batch_crc: &mut batch_crc,
                data_crc: &mut data_crc,
                data_record,
                bytes_read: 0,
            };
            let decoded = match input_error.take() {
                None => codec::decode_reader_with_kind_in(
                    record_type,
                    &mut reader,
                    payload_len,
                    false,
                    crate::json::JsonInputKind::StoredJson,
                    context,
                    Some((&mut strings, "operation_scratch")),
                ),
                Some(error) => Err(error),
            };
            std::io::copy(&mut reader, &mut std::io::sink())?;
            let checksum = reader.payload_crc.finish();
            let bytes_read = reader.bytes_read;
            drop(reader);
            let budget_error = decoded.as_ref().is_err_and(|error| {
                crate::json::is_limit_error(error) || error.code() == ErrorCode::ResourceExhausted
            });
            if budget_error && checksum == payload_checksum && bytes_read == payload_len {
                let discarded = pending.discard_buffer();
                let unbuffered = input_work
                    .as_mut()
                    .is_some_and(PayloadInputReservation::disable_optional_buffering);
                if discarded || unbuffered {
                    strings.retain(0);
                    if input_work.is_none() {
                        (input_work, input_error) = match PayloadInputReservation::new(
                            context,
                            record_type,
                            payload_len,
                            crate::json::JsonInputKind::StoredJson,
                        ) {
                            Ok(work) => (Some(work), None),
                            Err(error) => (None, Some(error)),
                        };
                    }
                    file.seek(SeekFrom::Start(payload_offset))?;
                    batch_crc = initial_batch_crc;
                    data_crc = initial_data_crc;
                    continue;
                }
            }
            break (decoded, checksum, bytes_read);
        };
        drop(input_work);
        if bytes_read != payload_len {
            report.findings.push(issue(
                "warning",
                "corrupt_tail",
                "ignored incomplete critical payload",
                "critical",
                "tail_only",
                Some(offset),
                Some(record_type_name(record_type)),
                Some(batch_id),
            ));
            break;
        }
        if let Err(err) = &decoded {
            if matches!(err.code(), ErrorCode::IoError | ErrorCode::PermissionDenied) {
                return Err(decoded.err().expect("decode failed"));
            }
        }
        if actual_checksum != payload_checksum {
            report.findings.push(issue(
                "error",
                "checksum_mismatch",
                "critical payload checksum mismatch",
                "critical",
                "blocks_main",
                Some(offset),
                Some(record_type_name(record_type)),
                Some(batch_id),
            ));
            break;
        }
        let (record, json_work) = match decoded {
            Ok(record) => record,
            Err(err)
                if crate::json::is_limit_error(&err)
                    || err.code() == ErrorCode::ResourceExhausted =>
            {
                return Err(err
                    .with_detail("record_type", record_type_name(record_type))
                    .with_detail("record_offset", offset));
            }
            Err(_) => {
                report.findings.push(issue(
                    "error",
                    "file_format_error",
                    "invalid binary critical payload",
                    "critical",
                    "blocks_main",
                    Some(offset),
                    Some(record_type_name(record_type)),
                    Some(batch_id),
                ));
                break;
            }
        };
        if let RecordPayload::CompactMarker(marker) = &record {
            if marker.compacted_main_checksum != compact_prefix_checksum {
                report.findings.push(issue(
                    "error",
                    "checksum_mismatch",
                    "compact prefix checksum mismatch",
                    "critical",
                    "blocks_main",
                    Some(offset),
                    Some("CompactMarker"),
                    Some(batch_id),
                ));
                break;
            }
            if marker.source_last_commit_sequence_no
                >= pending.first_sequence().unwrap_or(sequence_no)
            {
                report.findings.push(issue(
                    "error",
                    "file_format_error",
                    "compact source sequence must precede the output batch",
                    "critical",
                    "blocks_main",
                    Some(offset),
                    Some("CompactMarker"),
                    Some(batch_id),
                ));
                break;
            }
        }
        if let Err(err) = recover_id_high_watermarks(&record, &mut id_high_watermarks) {
            report.findings.push(issue(
                "error",
                "file_format_error",
                err.message(),
                "critical",
                "blocks_main",
                Some(offset),
                Some(record_type_name(record_type)),
                Some(batch_id),
            ));
            break;
        }
        report.records_scanned += 1;
        report.max_sequence_no = sequence_no;
        report.max_batch_id = batch_id;
        if let RecordPayload::Commit(commit) = record {
            if commit.batch_id != batch_id
                || commit.batch_checksum != batch_crc.finish() as u64
                || !pending.matches(&commit)
                || commit.last_sequence_no.checked_add(1) != Some(sequence_no)
            {
                report.findings.push(issue(
                    "error",
                    "checksum_mismatch",
                    "invalid critical commit; later batches ignored",
                    "critical",
                    "blocks_main",
                    Some(offset),
                    Some("Commit"),
                    Some(batch_id),
                ));
                break;
            }
            drop(commit);
            drop(json_work);
            drop(strings);
            pending.visit(file, batch_id, offset, context, visit)?;
            report.commits += 1;
            report.last_valid_commit = batch_id;
            report.last_valid_commit_sequence_no = sequence_no;
            report.critical_end_offset = file.stream_position()?;
            batch_crc = Crc32c::default();
            expected_batch = next_counter(batch_id)?;
        } else {
            pending.push(
                json_work,
                strings,
                CommittedRecord {
                    payload: record,
                    payload_checksum: payload_checksum as u64,
                    record_ref: RecordRef {
                        offset,
                        batch_id,
                        sequence_no,
                        record_type: record_type_name(record_type),
                        payload: PayloadRef {
                            offset: payload_offset,
                            len: payload_len,
                            checksum: payload_checksum as u64,
                        },
                    },
                },
            )?;
        }
        expected_sequence = next_counter(sequence_no)?;
    }
    if !pending.is_empty() {
        report.findings.push(issue(
            "warning",
            "uncommitted_batch",
            "ignored records without a complete commit",
            "critical",
            "tail_only",
            None,
            None,
            Some(expected_batch),
        ));
    }
    Ok(report)
}

fn is_after_visible_critical_prefix(report: &ScanReport, offset: u64) -> bool {
    report.last_valid_commit > 0 && offset >= report.critical_end_offset
}

fn push_record_boundary_issue(
    report: &mut ScanReport,
    offset: u64,
    code: &str,
    message: &str,
    record_type: Option<&str>,
    batch_id: Option<u64>,
) {
    if is_after_visible_critical_prefix(report, offset) && record_type == Some("Unknown") {
        push_index_suffix_issue(
            report,
            offset,
            "index_suffix_unreadable",
            message,
            record_type,
            batch_id,
        );
        return;
    }
    report.findings.push(issue(
        "error",
        code,
        message,
        "critical",
        "blocks_main",
        Some(offset),
        record_type,
        batch_id,
    ));
}

fn push_index_suffix_issue(
    report: &mut ScanReport,
    offset: u64,
    code: &str,
    message: &str,
    record_type: Option<&str>,
    batch_id: Option<u64>,
) {
    let mut finding = issue(
        "warning",
        code,
        message,
        "index",
        "rebuild_index_only",
        Some(offset),
        record_type,
        batch_id,
    );
    finding.repair_available = true;
    report.findings.push(finding);
}

fn push_index_suffix_diagnostic_issue(
    report: &mut ScanReport,
    diagnostic: &IndexSuffixDiagnostic,
    record_type: Option<&str>,
    batch_id: Option<u64>,
) {
    let mut finding = issue(
        "warning",
        diagnostic.code,
        &diagnostic.message,
        "index",
        "rebuild_index_only",
        Some(diagnostic.offset),
        record_type,
        batch_id,
    );
    finding.extent_id = diagnostic.extent_id.clone();
    finding.repair_available = true;
    report.findings.push(finding);
}

fn committed_batch_sequence_matches(
    batch: &[CommittedRecord],
    commit: &CommitPayload,
    batch_id: u64,
) -> bool {
    if batch.is_empty() {
        return false;
    }
    batch
        .iter()
        .all(|record| record.record_ref.batch_id == batch_id)
        && batch.first().map(|record| record.record_ref.sequence_no)
            == Some(commit.first_sequence_no)
        && batch.last().map(|record| record.record_ref.sequence_no) == Some(commit.last_sequence_no)
        && batch.windows(2).all(|records| {
            Some(records[1].record_ref.sequence_no)
                == records[0].record_ref.sequence_no.checked_add(1)
        })
}

#[cfg(test)]
pub(crate) fn rewrite_database_in<I, W>(
    storage: &mut Storage,
    records: I,
    plan: RewritePlan,
    durability: Durability,
    context: &crate::json::JsonOperationContext,
    mut visit: impl FnMut(CommittedRecord) -> Result<()>,
) -> Result<ScanReport>
where
    I: IntoIterator<Item = Result<(RecordPayload, W)>>,
    W: Into<RecordReservation>,
{
    rewrite_database_retained_in(
        storage,
        records,
        plan,
        durability,
        context,
        || false,
        |record, _| visit(record),
    )
}

pub(crate) fn rewrite_database_retained_in<I, W>(
    storage: &mut Storage,
    records: I,
    plan: RewritePlan,
    durability: Durability,
    context: &crate::json::JsonOperationContext,
    reclaim: impl FnOnce() -> bool,
    visit: impl FnMut(CommittedRecord, &mut RecordReservation) -> Result<()>,
) -> Result<ScanReport>
where
    I: IntoIterator<Item = Result<(RecordPayload, W)>>,
    W: Into<RecordReservation>,
{
    storage.rewrite_records_with_hook(
        records.into_iter().map(|record| {
            record.map(|(payload, work)| WriteRecord {
                payload,
                prepared: None,
                encoding: RecordEncoding::Stored(context.clone(), work.into()),
            })
        }),
        plan,
        durability,
        context,
        reclaim,
        |_, _| Ok(()),
        visit,
    )
}

#[cfg(test)]
fn rewrite_database<I>(
    storage: &mut Storage,
    records: I,
    plan: RewritePlan,
    durability: Durability,
    visit: impl FnMut(CommittedRecord) -> Result<()>,
) -> Result<ScanReport>
where
    I: IntoIterator<Item = Result<RecordPayload>>,
{
    storage.rewrite_with_iter_with_hook(records, plan, durability, |_, _| Ok(()), visit)
}

fn create_storage_file(path: &Path) -> Result<File> {
    let mut options = OpenOptions::new();
    options.create_new(true).read(true).write(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options.open(path)?;
    write_header(&mut file)?;
    file.sync_data()?;
    Ok(file)
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

fn next_counter(value: u64) -> Result<u64> {
    value
        .checked_add(1)
        .ok_or_else(|| Error::new(ErrorCode::FileFormatError, "storage counter exhausted"))
}

enum PayloadInputReservation {
    Json(crate::json::JsonReservation),
    Workspace(crate::json::WorkspaceReservation),
}

impl PayloadInputReservation {
    fn new(
        context: &crate::json::JsonOperationContext,
        record_type: u16,
        payload_len: u64,
        kind: crate::json::JsonInputKind,
    ) -> Result<Self> {
        let capacity = payload_len.min(64 * 1024) as usize;
        if matches!(record_type, COLLECTION_CREATE | DOCUMENT_PUT | CHUNK_PUT) {
            let mut work = context.reservation();
            if let Err(error) = work.grow(capacity, kind) {
                if !crate::json::is_memory_budget_error(&error) {
                    return Err(error);
                }
                // The decoder can read directly. Only its required fields and
                // JSON, not an optional buffer, may reject the record.
            }
            Ok(Self::Json(work))
        } else {
            // Buffering is optional: preserve direct reads when other work
            // already owns the runtime's workspace.
            let mut work = context.workspace_reservation();
            work.grow_to(0, capacity, "operation_scratch")?;
            Ok(Self::Workspace(work))
        }
    }

    fn capacity(&self) -> usize {
        match self {
            Self::Json(work) => work.reserved_bytes(),
            Self::Workspace(work) => work.reserved_bytes(),
        }
    }

    fn disable_optional_buffering(&mut self) -> bool {
        match self {
            Self::Json(work) if work.reserved_bytes() > 0 => {
                work.retain(0);
                true
            }
            Self::Workspace(work) if work.reserved_bytes() > 0 => {
                work.retain(0);
                true
            }
            _ => false,
        }
    }
}

struct CheckedPayloadReader<'a, R> {
    source: R,
    payload_crc: Crc32c,
    batch_crc: &'a mut Crc32c,
    data_crc: &'a mut Crc32c,
    data_record: bool,
    bytes_read: u64,
}

impl<R: Read> Read for CheckedPayloadReader<'_, R> {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        let count = self.source.read(buffer)?;
        let bytes = &buffer[..count];
        self.payload_crc.update(bytes);
        if self.data_record {
            self.batch_crc.update(bytes);
            self.data_crc.update(bytes);
        }
        self.bytes_read += count as u64;
        Ok(count)
    }
}

fn recover_id_high_watermarks(
    record: &RecordPayload,
    watermarks: &mut BTreeMap<&'static str, u64>,
) -> Result<()> {
    let id = match record {
        RecordPayload::CollectionCreate(collection) => Some(collection.collection_id.as_str()),
        RecordPayload::CollectionDrop { collection_id, .. } => Some(collection_id.as_str()),
        RecordPayload::DocumentPut(document) => Some(document.document_id.as_str()),
        RecordPayload::ChunkPut(chunk) => Some(chunk.chunk_id.as_str()),
        RecordPayload::EmbeddingPut(embedding) => Some(embedding.embedding_id.as_str()),
        RecordPayload::Tombstone(tombstone) => Some(tombstone.target_id.as_str()),
        RecordPayload::CompactMarker(marker) => {
            for prefix in ["col_", "doc_", "chk_", "emb_"] {
                let value = marker.id_high_watermarks.get(prefix).copied().unwrap_or(0);
                if value < watermarks.get(prefix).copied().unwrap_or(0) {
                    return Err(Error::new(
                        ErrorCode::FileFormatError,
                        "compact marker lowers an allocation high watermark",
                    ));
                }
            }
            for prefix in ["col_", "doc_", "chk_", "emb_"] {
                watermarks.insert(
                    prefix,
                    marker.id_high_watermarks.get(prefix).copied().unwrap_or(0),
                );
            }
            None
        }
        RecordPayload::Commit(_) => None,
    };
    if let Some(id) = id {
        for prefix in ["col_", "doc_", "chk_", "emb_"] {
            if let Some(number) = id.strip_prefix(prefix) {
                let value = number.parse::<u64>().map_err(|_| {
                    Error::new(ErrorCode::FileFormatError, "invalid object allocation ID")
                })?;
                let seen = watermarks.entry(prefix).or_default();
                *seen = (*seen).max(value);
                break;
            }
        }
    }
    Ok(())
}

struct WriteRecord<'a> {
    payload: RecordPayload,
    prepared: Option<(&'a File, PayloadRef)>,
    encoding: RecordEncoding,
}

enum RecordEncoding {
    Batch,
    Stored(crate::json::JsonOperationContext, RecordReservation),
    #[cfg(test)]
    Reference,
}

impl RecordEncoding {
    fn encode(
        &mut self,
        record: &RecordPayload,
        output: &mut impl Write,
        plan: Option<&codec::BatchEncoding>,
        ordinal: usize,
    ) -> Result<()> {
        match self {
            Self::Batch => plan
                .ok_or_else(|| {
                    Error::new(
                        ErrorCode::FileFormatError,
                        "batch encoding was not admitted",
                    )
                })?
                .encode(ordinal, record, output),
            Self::Stored(context, work) => {
                codec::encode_stored_into(record, output, context, &mut work.json)
            }
            #[cfg(test)]
            Self::Reference => codec::encode_into(record, output),
        }
    }
}

impl WriteRecord<'_> {
    fn planned(payload: RecordPayload) -> Self {
        Self {
            payload,
            prepared: None,
            encoding: RecordEncoding::Batch,
        }
    }

    #[cfg(test)]
    fn encoded(payload: RecordPayload) -> Self {
        Self {
            payload,
            prepared: None,
            encoding: RecordEncoding::Reference,
        }
    }
}

#[cfg(test)]
fn write_batch_to_file<I>(
    file: &mut File,
    next_batch_id: &mut u64,
    next_sequence_no: &mut u64,
    records: I,
    record_count: usize,
    durability: Durability,
    retain_records: bool,
) -> Result<AppendSummary>
where
    I: IntoIterator<Item = Result<RecordPayload>>,
{
    BatchWriter::new(
        file,
        next_batch_id,
        next_sequence_no,
        record_count,
        retain_records,
        &crate::json::JsonOperationContext::standalone(),
        None,
    )?
    .append_all(
        records
            .into_iter()
            .map(|record| record.map(WriteRecord::encoded)),
        durability,
    )
}

struct BatchWriter<'a> {
    file: &'a mut File,
    next_sequence_no: &'a mut u64,
    batch_id: u64,
    batch_start: u64,
    first_sequence_no: u64,
    record_limit: usize,
    retain_records: bool,
    written: usize,
    encoded_bytes: u64,
    batch_crc: Crc32c,
    compact_crc: Option<&'a mut Crc32c>,
    temporary_disk: Option<&'a mut crate::temporary::TempFileGuard>,
    committed_records: Vec<CommittedRecord>,
    encoding: EncodingScratch,
    prepared_buffer: Vec<u8>,
    commit_timestamp: crate::json::ReservedString,
    raw_encoding: Option<codec::BatchEncoding>,
}

impl<'a> BatchWriter<'a> {
    fn new(
        file: &'a mut File,
        next_batch_id: &mut u64,
        next_sequence_no: &'a mut u64,
        record_limit: usize,
        retain_records: bool,
        context: &crate::json::JsonOperationContext,
        commit_timestamp: Option<crate::json::ReservedString>,
    ) -> Result<Self> {
        if record_limit == 0 {
            return Err(Error::new(
                ErrorCode::FileFormatError,
                "cannot append an empty batch",
            ));
        }
        let next_batch = next_counter(*next_batch_id)?;
        let sequence_count = u64::try_from(record_limit)
            .ok()
            .and_then(|count| count.checked_add(1))
            .ok_or_else(|| Error::new(ErrorCode::FileFormatError, "batch record count overflow"))?;
        next_sequence_no
            .checked_add(sequence_count)
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::FileFormatError,
                    "storage sequence counter exhausted",
                )
            })?;
        let commit_timestamp = match commit_timestamp {
            Some(timestamp) => timestamp,
            None => crate::json::ReservedString::with_capacity(
                time::MAX_TIMESTAMP_BYTES,
                context,
                "batch_prepare",
            )?,
        };
        let mut committed_records = Vec::new();
        if retain_records {
            committed_records
                .try_reserve_exact(record_limit)
                .map_err(|error| std::io::Error::other(error.to_string()))?;
            if committed_records.capacity() != record_limit {
                return Err(std::io::Error::other(
                    "committed record allocation exceeds requested capacity",
                )
                .into());
            }
        }
        let batch_start = file.seek(SeekFrom::End(0))?;
        let batch_id = *next_batch_id;
        *next_batch_id = next_batch;
        let first_sequence_no = *next_sequence_no;
        Ok(Self {
            file,
            next_sequence_no,
            batch_id,
            batch_start,
            first_sequence_no,
            record_limit,
            retain_records,
            written: 0,
            encoded_bytes: 0,
            batch_crc: Crc32c::default(),
            compact_crc: None,
            temporary_disk: None,
            committed_records,
            encoding: EncodingScratch::new(context),
            prepared_buffer: Vec::new(),
            commit_timestamp,
            raw_encoding: None,
        })
    }

    fn append_all<'b>(
        mut self,
        records: impl IntoIterator<Item = Result<WriteRecord<'b>>>,
        durability: Durability,
    ) -> Result<AppendSummary> {
        for record in records {
            if self.written == self.record_limit {
                return Err(Error::new(
                    ErrorCode::FileFormatError,
                    "batch record count mismatch",
                ));
            }
            self.append(record?)?;
        }
        if self.written != self.record_limit {
            return Err(Error::new(
                ErrorCode::FileFormatError,
                "batch record count mismatch",
            ));
        }
        self.finish(durability)
    }

    fn append(&mut self, record: WriteRecord<'_>) -> Result<()> {
        if self.written == self.record_limit {
            return Err(Error::new(
                ErrorCode::FileFormatError,
                "batch record count mismatch",
            ));
        }
        let WriteRecord {
            payload,
            prepared,
            mut encoding,
        } = record;
        // On every early return, free the payload before returning its reservation.
        let mut record = payload;
        let offset = self.file.stream_position()?;
        let sequence_no = *self.next_sequence_no;
        *self.next_sequence_no += 1;
        if let RecordPayload::CompactMarker(marker) = &mut record {
            marker.compacted_main_checksum = self
                .compact_crc
                .as_deref()
                .map(|crc| crc.finish())
                .unwrap_or_else(|| self.batch_crc.finish());
        }
        let mut stream_work = self.encoding.work.context().workspace_reservation();
        let (payload_len, payload_checksum) = if let Some((_, source)) = &prepared {
            if self.prepared_buffer.is_empty() {
                self.prepared_buffer
                    .try_reserve_exact(prepared::PREPARED_COPY_BYTES)
                    .map_err(|error| std::io::Error::other(error.to_string()))?;
                if self.prepared_buffer.capacity() != prepared::PREPARED_COPY_BYTES {
                    return Err(std::io::Error::other(
                        "prepared copy allocation exceeds requested capacity",
                    )
                    .into());
                }
                self.prepared_buffer
                    .resize(prepared::PREPARED_COPY_BYTES, 0);
            }
            (
                source.len,
                u32::try_from(source.checksum).map_err(|_| {
                    Error::new(
                        ErrorCode::FileFormatError,
                        "invalid prepared payload checksum",
                    )
                })?,
            )
        } else {
            self.encoding.reset();
            stream_work.grow_to(0, 8192, "batch_prepare")?;
            let encoded = encoding.encode(
                &record,
                &mut self.encoding,
                self.raw_encoding.as_ref(),
                self.written,
            );
            if encoded.as_ref().is_err_and(crate::json::is_limit_error)
                && (self.encoding.work.reserved_bytes() > 0 || stream_work.reserved_bytes() > 0)
            {
                // Optional IO buffers must not strand the JSON workspace needed by this record.
                self.encoding.disable_buffering();
                stream_work.retain(0);
                encoding.encode(
                    &record,
                    &mut self.encoding,
                    self.raw_encoding.as_ref(),
                    self.written,
                )?;
            } else {
                encoded?;
            }
            (self.encoding.digest.len, self.encoding.digest.crc.finish())
        };
        let header = record_header_from_digest(
            record_type_id(&record),
            self.batch_id,
            sequence_no,
            payload_len,
            payload_checksum,
        )?;
        let encoded_bytes = self
            .encoded_bytes
            .checked_add(read_u64(&header[8..16]))
            .ok_or_else(|| Error::new(ErrorCode::FileFormatError, "batch byte count overflow"))?;
        if let Some(temporary) = &mut self.temporary_disk {
            temporary.reserve_disk(read_u64(&header[8..16]))?;
        }
        #[cfg(test)]
        tests::write_at_crash_boundary(self.file, &header, "critical_header_partial")?;
        #[cfg(not(test))]
        self.file.write_all(&header)?;
        self.batch_crc.update(&header);
        if let Some(crc) = &mut self.compact_crc {
            crc.update(&header);
        }
        if let Some((source, payload)) = prepared {
            let copied = {
                let mut output = PayloadWriter {
                    file: self.file,
                    batch_crc: &mut self.batch_crc,
                    compact_crc: self.compact_crc.as_deref_mut(),
                    #[cfg(test)]
                    crash_after: payload_len / 2,
                };
                prepared::copy_payload(source, &payload, &mut output, &mut self.prepared_buffer)
            };
            if let Err(error) = copied {
                // No commit has been emitted, so only this batch's suffix can be removed.
                self.file.set_len(self.batch_start)?;
                self.file.seek(SeekFrom::Start(self.batch_start))?;
                return Err(error);
            }
        } else if payload_len == self.encoding.buffer.len() as u64 {
            #[cfg(test)]
            tests::write_at_crash_boundary(
                self.file,
                &self.encoding.buffer,
                "critical_payload_partial",
            )?;
            #[cfg(not(test))]
            self.file.write_all(&self.encoding.buffer)?;
            self.batch_crc.update(&self.encoding.buffer);
            if let Some(crc) = &mut self.compact_crc {
                crc.update(&self.encoding.buffer);
            }
        } else {
            let writer = PayloadWriter {
                file: self.file,
                batch_crc: &mut self.batch_crc,
                compact_crc: self.compact_crc.as_deref_mut(),
                #[cfg(test)]
                crash_after: payload_len / 2,
            };
            let mut output = BufWriter::with_capacity(stream_work.reserved_bytes(), writer);
            if output.capacity() > stream_work.reserved_bytes() {
                let (writer, buffer) = output.into_parts();
                drop(buffer);
                stream_work.retain(0);
                output = BufWriter::with_capacity(0, writer);
            }
            let written = encoding
                .encode(
                    &record,
                    &mut output,
                    self.raw_encoding.as_ref(),
                    self.written,
                )
                .and_then(|()| output.flush().map_err(Error::from));
            // Do not retry a failed buffered write while unwinding the error.
            let _ = output.into_parts();
            written?;
        }
        #[cfg(test)]
        tests::crash_boundary("critical_record_written")?;
        if self.retain_records {
            let record_type = record_type_name(record_type_id(&record));
            self.committed_records.push(CommittedRecord {
                payload: record,
                record_ref: RecordRef {
                    offset,
                    record_type,
                    batch_id: self.batch_id,
                    sequence_no,
                    payload: PayloadRef {
                        offset: offset + RECORD_HEADER_LEN as u64,
                        len: payload_len,
                        checksum: u64::from(payload_checksum),
                    },
                },
                payload_checksum: u64::from(payload_checksum),
            });
        }
        self.written += 1;
        self.encoded_bytes = encoded_bytes;
        Ok(())
    }

    fn finish(mut self, durability: Durability) -> Result<AppendSummary> {
        if self.written == 0 {
            return Err(Error::new(
                ErrorCode::FileFormatError,
                "cannot append an empty batch",
            ));
        }
        self.file.flush()?;
        #[cfg(test)]
        tests::crash_boundary("batch_before_commit")?;
        self.commit_timestamp
            .append(|out| time::write_unix_millis(out, time::unix_millis_now()))?;
        let (committed_at, commit_work) = self.commit_timestamp.into_parts();
        let commit = CommitPayload {
            batch_id: self.batch_id,
            first_sequence_no: self.first_sequence_no,
            last_sequence_no: *self.next_sequence_no - 1,
            record_count: self.written,
            batch_checksum: self.batch_crc.finish() as u64,
            committed_at,
        };
        // Binary v1: four u64 fields, one u32 checksum, and one u64 timestamp.
        let mut payload = [0u8; codec::COMMIT_PAYLOAD_BYTES];
        let mut remaining = payload.as_mut_slice();
        codec::encode_commit_into(&commit, &mut remaining, self.encoding.work.context())?;
        if !remaining.is_empty() {
            return Err(Error::new(
                ErrorCode::FileFormatError,
                "invalid commit payload length",
            ));
        }
        if let Some(temporary) = &mut self.temporary_disk {
            let bytes = (payload.len() as u64)
                .checked_add(RECORD_HEADER_LEN as u64)
                .ok_or_else(|| {
                    Error::new(ErrorCode::FileFormatError, "commit record length overflow")
                })?;
            temporary.reserve_disk(bytes)?;
        }
        write_record(
            self.file,
            COMMIT,
            self.batch_id,
            *self.next_sequence_no,
            &payload,
        )?;
        #[cfg(test)]
        tests::crash_boundary("commit_written")?;
        let commit_sequence_no = *self.next_sequence_no;
        *self.next_sequence_no += 1;
        self.file.flush()?;
        #[cfg(test)]
        tests::crash_boundary("commit_flushed")?;
        if durability == Durability::Commit {
            self.file.sync_data()?;
            #[cfg(test)]
            tests::crash_boundary("commit_synced")?;
        }
        Ok(AppendSummary {
            commit,
            commit_sequence_no,
            records: self.committed_records,
            _commit_work: Some(commit_work),
        })
    }
}

#[derive(Default)]
struct PayloadDigest {
    len: u64,
    crc: Crc32c,
}

const MAX_ENCODE_BUFFER_BYTES: usize = 64 * 1024;

// Small records encode once; larger records retain only their digest and stream
// a second pass. Reuse this bounded allocation across every record in a batch.
struct EncodingScratch {
    digest: PayloadDigest,
    buffer: Vec<u8>,
    work: crate::json::WorkspaceReservation,
    buffering: bool,
}

impl EncodingScratch {
    fn new(context: &crate::json::JsonOperationContext) -> Self {
        Self {
            digest: PayloadDigest::default(),
            buffer: Vec::new(),
            work: context.workspace_reservation(),
            buffering: true,
        }
    }

    fn disable_buffering(&mut self) {
        self.release_buffer();
        self.digest = PayloadDigest::default();
        self.buffering = false;
    }

    fn release_buffer(&mut self) {
        self.buffer = Vec::new();
        self.work.retain(0);
    }

    fn reset(&mut self) {
        self.digest = PayloadDigest::default();
        self.buffer.clear();
    }
}

impl Write for EncodingScratch {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        let written = self.digest.write(bytes)?;
        if self.buffering
            && self.digest.len <= MAX_ENCODE_BUFFER_BYTES as u64
            && self.buffer.len() as u64 + bytes.len() as u64 == self.digest.len
        {
            let length = self.digest.len as usize;
            if length > self.buffer.capacity() {
                let capacity = length.next_power_of_two().min(MAX_ENCODE_BUFFER_BYTES);
                let granted = self
                    .work
                    .grow_to(0, capacity, "batch_prepare")
                    .map_err(|error| std::io::Error::other(error.to_string()))?;
                if granted < length {
                    self.work.retain(self.buffer.capacity());
                    self.buffer.clear();
                    return Ok(written);
                }
                self.buffer
                    .try_reserve_exact(granted - self.buffer.len())
                    .map_err(std::io::Error::other)?;
                if self.buffer.capacity() > granted {
                    self.buffer = Vec::new();
                    self.work.retain(0);
                    return Ok(written);
                }
                self.work.retain(self.buffer.capacity());
            }
            self.buffer.extend_from_slice(bytes);
        } else {
            self.buffer.clear();
        }
        Ok(written)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl Write for PayloadDigest {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.len = self
            .len
            .checked_add(bytes.len() as u64)
            .ok_or_else(|| std::io::Error::other("record payload length overflow"))?;
        self.crc.update(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

struct PayloadWriter<'a> {
    file: &'a mut File,
    batch_crc: &'a mut Crc32c,
    compact_crc: Option<&'a mut Crc32c>,
    #[cfg(test)]
    crash_after: u64,
}

impl Write for PayloadWriter<'_> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        #[cfg(test)]
        let bytes = {
            if self.crash_after == 0 {
                tests::crash_boundary("critical_payload_partial").map_err(std::io::Error::other)?;
                self.crash_after = u64::MAX;
            }
            &bytes[..bytes
                .len()
                .min(usize::try_from(self.crash_after).unwrap_or(usize::MAX))]
        };
        let written = self.file.write(bytes)?;
        self.batch_crc.update(&bytes[..written]);
        if let Some(crc) = &mut self.compact_crc {
            crc.update(&bytes[..written]);
        }
        #[cfg(test)]
        {
            self.crash_after -= written as u64;
        }
        Ok(written)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.file.flush()
    }
}

fn write_header(file: &mut File) -> Result<()> {
    file.write_all(&file_header(time::unix_millis_now()))?;
    Ok(())
}

fn file_header(created_at: u64) -> [u8; HEADER_LEN] {
    let mut header = [0; HEADER_LEN];
    header[..8].copy_from_slice(MAGIC);
    header[10..12].copy_from_slice(&1u16.to_le_bytes());
    header[12] = 1;
    header[13] = 1;
    header[22..30].copy_from_slice(&(HEADER_LEN as u64).to_le_bytes());
    header[30..38].copy_from_slice(&created_at.to_le_bytes());
    header[38..40].copy_from_slice(&44u16.to_le_bytes());
    let checksum = checksum32(&header[..40]);
    header[40..44].copy_from_slice(&checksum.to_le_bytes());
    header[44..46].copy_from_slice(&1u16.to_le_bytes());
    let hint_checksum = checksum32(&header[44..54]);
    header[54..58].copy_from_slice(&hint_checksum.to_le_bytes());
    header
}

fn read_header(file: &mut File) -> Result<FileCodec> {
    let mut prefix = [0u8; 12];
    file.read_exact(&mut prefix)?;
    file.seek(SeekFrom::Start(0))?;
    if &prefix[0..8] != MAGIC {
        return Err(Error::new(
            ErrorCode::FileFormatError,
            "invalid CofferDB magic bytes",
        ));
    }
    if read_u16(&prefix[8..10]) == 0 && read_u16(&prefix[10..12]) == 1 {
        let mut header = [0; HEADER_LEN];
        file.read_exact(&mut header)?;
        if checksum32(&header[..40]) != read_u32(&header[40..44]) {
            return Err(Error::new(
                ErrorCode::ChecksumMismatch,
                "file header checksum mismatch",
            ));
        }
        if header[12] != 1 || read_u64(&header[14..22]) != 0 {
            return Err(Error::new(
                ErrorCode::UnsupportedFeatureFlag,
                "unsupported endianness or feature flags",
            ));
        }
        if header[13] != 1 {
            return Err(Error::new(
                ErrorCode::UnsupportedVersion,
                "unsupported checksum algorithm",
            ));
        }
        if read_u16(&header[38..40]) != 44 || read_u64(&header[22..30]) != HEADER_LEN as u64 {
            return Err(Error::new(
                ErrorCode::FileFormatError,
                "invalid critical header length or stream offset",
            ));
        }
        // Hints are advisory; recovery scans the critical stream even if they are corrupt.
        return Ok(FileCodec::Binary);
    }
    let mut header = [0u8; LEGACY_HEADER_LEN];
    file.read_exact(&mut header).map_err(Error::from)?;
    if &header[0..8] != MAGIC {
        return Err(Error::new(
            ErrorCode::FileFormatError,
            "invalid CofferDB magic bytes",
        ));
    }
    let version = read_u16(&header[8..10]);
    if version != FORMAT_VERSION {
        return Err(Error::new(
            ErrorCode::UnsupportedVersion,
            format!("unsupported file format version {version}"),
        ));
    }
    if header[12] != 1 {
        return Err(Error::new(
            ErrorCode::UnsupportedFeatureFlag,
            "unsupported endianness marker",
        ));
    }
    if read_u16(&header[10..12]) != 0 || header[13] != 0 {
        return Err(Error::new(
            ErrorCode::UnsupportedVersion,
            "unsupported legacy header extension",
        ));
    }
    if read_u64(&header[18..26]) != 0 {
        return Err(Error::new(
            ErrorCode::UnsupportedFeatureFlag,
            "unknown file feature flag",
        ));
    }
    let header_len = read_u32(&header[14..18]) as usize;
    if header_len != LEGACY_HEADER_LEN {
        return Err(Error::new(
            ErrorCode::FileFormatError,
            format!("invalid header length {header_len}"),
        ));
    }
    if read_u64(&header[34..42]) != checksum64(&header[0..34]) {
        return Err(Error::new(
            ErrorCode::ChecksumMismatch,
            "file header checksum mismatch",
        ));
    }
    Ok(FileCodec::Legacy)
}

fn write_record(
    file: &mut File,
    record_type: u16,
    batch_id: u64,
    sequence_no: u64,
    payload: &[u8],
) -> Result<()> {
    let header = record_header(record_type, batch_id, sequence_no, payload)?;
    #[cfg(test)]
    tests::write_at_crash_boundary(file, &header, "commit_header_partial")?;
    #[cfg(not(test))]
    file.write_all(&header)?;
    #[cfg(test)]
    tests::write_at_crash_boundary(file, payload, "commit_payload_partial")?;
    #[cfg(not(test))]
    file.write_all(payload)?;
    Ok(())
}

fn record_header(
    record_type: u16,
    batch_id: u64,
    sequence_no: u64,
    payload: &[u8],
) -> Result<[u8; RECORD_HEADER_LEN]> {
    record_header_from_digest(
        record_type,
        batch_id,
        sequence_no,
        payload.len() as u64,
        checksum32(payload),
    )
}

fn record_header_from_digest(
    record_type: u16,
    batch_id: u64,
    sequence_no: u64,
    payload_len: u64,
    payload_checksum: u32,
) -> Result<[u8; RECORD_HEADER_LEN]> {
    let record_len = payload_len
        .checked_add(RECORD_HEADER_LEN as u64)
        .ok_or_else(|| Error::new(ErrorCode::FileFormatError, "record length overflow"))?;
    let mut header = [0; RECORD_HEADER_LEN];
    header[..4].copy_from_slice(b"CFRL");
    header[4..6].copy_from_slice(&record_type.to_le_bytes());
    header[6..8].copy_from_slice(&1u16.to_le_bytes());
    header[8..16].copy_from_slice(&record_len.to_le_bytes());
    header[16..24].copy_from_slice(&batch_id.to_le_bytes());
    header[24..32].copy_from_slice(&sequence_no.to_le_bytes());
    header[32..34].copy_from_slice(&(RECORD_HEADER_LEN as u16).to_le_bytes());
    header[34..42].copy_from_slice(&payload_len.to_le_bytes());
    header[42..46].copy_from_slice(&payload_checksum.to_le_bytes());
    let header_checksum = checksum32(&header[..46]);
    header[46..50].copy_from_slice(&header_checksum.to_le_bytes());
    Ok(header)
}

#[cfg(test)]
fn build_index_checkpoint_suffix(
    checkpoint: &IndexCheckpointPayload,
    critical_end_offset: u64,
) -> Result<Vec<u8>> {
    build_index_checkpoint_suffix_with_generation(
        checkpoint,
        critical_end_offset,
        critical_end_offset,
    )
}

#[cfg(test)]
fn build_index_checkpoint_suffix_with_generation(
    checkpoint: &IndexCheckpointPayload,
    critical_end_offset: u64,
    checkpoint_generation: u64,
) -> Result<Vec<u8>> {
    let segment_records = build_index_segment_records(checkpoint, checkpoint_generation)?;
    let segment_record_count = u64::try_from(segment_records.len()).map_err(|_| {
        Error::new(
            ErrorCode::FileFormatError,
            "index segment record count exceeds u32",
        )
    })?;
    let metadata_record_len = (INDEX_RECORD_HEADER_LEN + INDEX_COMMIT_METADATA_LEN) as u64;
    let mut segment_extents = Vec::with_capacity(segment_records.len());
    let mut referenced_checksum = index_commit_hasher(
        checkpoint.metadata.covered_last_batch_id,
        checkpoint.metadata.covered_last_commit_sequence_no,
        checkpoint_generation,
    );
    let mut next_payload_offset = checked_writer_add(
        INDEX_CHECKPOINT_HEADER_LEN as u64,
        metadata_record_len,
        "index checkpoint metadata boundary overflow",
    )?;
    for record in &segment_records {
        let extent = IndexExtentEntry {
            index_record_id: record.index_record_id,
            record_type: record.record_type,
            stream_id: record.stream_id,
            suffix_relative_offset: next_payload_offset,
            length: record.bytes.len() as u64,
            checkpoint_generation,
            extent_record_checksum: checksum32(&record.bytes),
        };
        referenced_checksum.update(&encode_extent_entry(&extent));
        referenced_checksum.update(&record.bytes);
        next_payload_offset = checked_writer_add(
            next_payload_offset,
            record.bytes.len() as u64,
            "index payload length overflow",
        )?;
        segment_extents.push(extent);
    }
    let segment_extent_table = encode_extent_table(&segment_extents);
    let metadata_payload = build_index_commit_metadata_payload(
        &checkpoint.metadata,
        segment_record_count,
        INDEX_EXTENT_ENTRY_LEN as u64,
        segment_extent_table.len() as u64,
        checksum32(&segment_extent_table),
        referenced_checksum.finish(),
    );
    let metadata_record = build_index_record(
        INDEX_COMMIT_METADATA,
        checkpoint_generation,
        1,
        &metadata_payload,
    );
    let index_record_count = checked_writer_add(
        segment_record_count,
        1,
        "index checkpoint record count overflow",
    )?;
    let extent_table_len = index_record_count
        .checked_mul(INDEX_EXTENT_ENTRY_LEN as u64)
        .ok_or_else(|| {
            Error::new(
                ErrorCode::FileFormatError,
                "index extent table length overflow",
            )
        })?;
    let suffix_len = checked_writer_add(
        checked_writer_add(
            next_payload_offset,
            extent_table_len,
            "index extent boundary overflow",
        )?,
        INDEX_CHECKPOINT_TRAILER_LEN as u64,
        "index checkpoint length overflow",
    )?;
    let capacity = usize::try_from(suffix_len).map_err(|_| {
        Error::new(
            ErrorCode::FileFormatError,
            "index checkpoint exceeds address space",
        )
    })?;
    let mut suffix = Vec::new();
    suffix
        .try_reserve_exact(capacity)
        .map_err(std::io::Error::other)?;
    // Assemble once in the final buffer, returning each encoded segment as it is consumed.
    suffix.resize(INDEX_CHECKPOINT_HEADER_LEN, 0);
    suffix.extend_from_slice(&metadata_record);
    for record in segment_records {
        suffix.extend_from_slice(&record.bytes);
    }
    let payload_len = suffix.len() as u64 - INDEX_CHECKPOINT_HEADER_LEN as u64;
    let payload_checksum = checksum32(&suffix[INDEX_CHECKPOINT_HEADER_LEN..]);
    let extent_table_offset = INDEX_CHECKPOINT_HEADER_LEN as u64 + payload_len;
    let metadata_extent = IndexExtentEntry {
        index_record_id: 1,
        record_type: INDEX_COMMIT_METADATA,
        stream_id: 0,
        suffix_relative_offset: INDEX_CHECKPOINT_HEADER_LEN as u64,
        length: metadata_record.len() as u64,
        checkpoint_generation,
        extent_record_checksum: checksum32(&metadata_record),
    };
    let mut extents = Vec::with_capacity(1 + segment_extents.len());
    extents.push(metadata_extent);
    extents.extend(segment_extents);
    let extent_table = encode_extent_table(&extents);
    debug_assert_eq!(extent_table.len() as u64, extent_table_len);
    let extent_table_checksum = checksum32(&extent_table);

    let mut header = Vec::with_capacity(INDEX_CHECKPOINT_HEADER_LEN);
    header.extend_from_slice(INDEX_CHECKPOINT_MAGIC);
    header.extend_from_slice(&INDEX_CHECKPOINT_VERSION.to_le_bytes());
    header.extend_from_slice(&(INDEX_CHECKPOINT_HEADER_LEN as u16).to_le_bytes());
    header.extend_from_slice(&checkpoint_generation.to_le_bytes());
    header.extend_from_slice(&critical_end_offset.to_le_bytes());
    header.extend_from_slice(&critical_end_offset.to_le_bytes());
    header.extend_from_slice(&checkpoint.metadata.covered_last_batch_id.to_le_bytes());
    header.extend_from_slice(
        &checkpoint
            .metadata
            .covered_last_commit_sequence_no
            .to_le_bytes(),
    );
    header.push(INDEX_COVERAGE_FULL_VISIBLE_SNAPSHOT);
    header.extend_from_slice(&checkpoint.metadata.coverage_hash.to_le_bytes());
    header.extend_from_slice(&index_record_count.to_le_bytes());
    header.extend_from_slice(&index_record_count.to_le_bytes());
    header.extend_from_slice(&extent_table_offset.to_le_bytes());
    header.extend_from_slice(&extent_table_len.to_le_bytes());
    header.extend_from_slice(&extent_table_checksum.to_le_bytes());
    header.extend_from_slice(&payload_len.to_le_bytes());
    header.extend_from_slice(&payload_checksum.to_le_bytes());
    debug_assert_eq!(header.len(), INDEX_CHECKPOINT_HEADER_LEN);

    let mut trailer = Vec::with_capacity(INDEX_CHECKPOINT_TRAILER_LEN);
    trailer.extend_from_slice(INDEX_COMPLETE_MARKER);
    trailer.extend_from_slice(&(INDEX_CHECKPOINT_TRAILER_LEN as u16).to_le_bytes());
    let trailer_checksum = checksum32(&trailer);
    trailer.extend_from_slice(&trailer_checksum.to_le_bytes());
    debug_assert_eq!(trailer.len(), INDEX_CHECKPOINT_TRAILER_LEN);

    suffix[..INDEX_CHECKPOINT_HEADER_LEN].copy_from_slice(&header);
    suffix.extend_from_slice(&extent_table);
    suffix.extend_from_slice(&trailer);
    debug_assert_eq!(suffix.len(), capacity);
    Ok(suffix)
}

#[cfg(test)]
fn build_index_record(
    record_type: u16,
    checkpoint_generation: u64,
    index_record_id: u64,
    payload: &[u8],
) -> Vec<u8> {
    let record_len = INDEX_RECORD_HEADER_LEN as u64 + payload.len() as u64;
    let mut header = Vec::with_capacity(INDEX_RECORD_HEADER_LEN);
    header.extend_from_slice(INDEX_RECORD_MAGIC);
    header.extend_from_slice(&record_type.to_le_bytes());
    header.extend_from_slice(&1u16.to_le_bytes());
    header.extend_from_slice(&(INDEX_RECORD_HEADER_LEN as u16).to_le_bytes());
    header.extend_from_slice(&record_len.to_le_bytes());
    header.extend_from_slice(&checkpoint_generation.to_le_bytes());
    header.extend_from_slice(&index_record_id.to_le_bytes());
    header.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    header.extend_from_slice(&checksum32(payload).to_le_bytes());
    let header_checksum = checksum32(&header);
    header.extend_from_slice(&header_checksum.to_le_bytes());
    debug_assert_eq!(header.len(), INDEX_RECORD_HEADER_LEN);

    let mut record = Vec::with_capacity(record_len as usize);
    record.extend_from_slice(&header);
    record.extend_from_slice(payload);
    record
}

#[cfg(test)]
struct BuiltIndexRecord {
    index_record_id: u64,
    record_type: u16,
    stream_id: u64,
    bytes: Vec<u8>,
}

#[cfg(test)]
fn build_index_segment_records(
    checkpoint: &IndexCheckpointPayload,
    checkpoint_generation: u64,
) -> Result<Vec<BuiltIndexRecord>> {
    if checkpoint.metadata.text_segment_count != checkpoint.text_segments.len()
        || checkpoint.metadata.vector_segment_count != checkpoint.vector_segments.len()
    {
        return Err(Error::new(
            ErrorCode::FileFormatError,
            "index checkpoint segment counts do not match segment payloads",
        ));
    }
    let checked_count = checked_segment_record_count(&checkpoint.metadata)?;
    let actual_count = u64::try_from(
        checkpoint.text_segments.len() + checkpoint.vector_segments.len(),
    )
    .map_err(|_| {
        Error::new(
            ErrorCode::FileFormatError,
            "index segment count exceeds u32",
        )
    })?;
    if checked_count != actual_count {
        return Err(Error::new(
            ErrorCode::FileFormatError,
            "index checkpoint segment count mismatch",
        ));
    }

    let mut records = Vec::with_capacity(actual_count as usize);
    let mut next_record_id = 2u64;
    for segment in &checkpoint.text_segments {
        let payload = encode_text_index_segment_payload(segment)?;
        let stream_id = catalog_id_number(&segment.collection_id)?;
        records.push(BuiltIndexRecord {
            index_record_id: next_record_id,
            record_type: INDEX_TEXT_SEGMENT,
            stream_id,
            bytes: build_index_record(
                INDEX_TEXT_SEGMENT,
                checkpoint_generation,
                next_record_id,
                &payload,
            ),
        });
        next_record_id += 1;
    }
    for segment in &checkpoint.vector_segments {
        let payload = encode_vector_index_segment_payload(segment)?;
        let stream_id = catalog_id_number(&segment.collection_id)?;
        records.push(BuiltIndexRecord {
            index_record_id: next_record_id,
            record_type: INDEX_VECTOR_SEGMENT,
            stream_id,
            bytes: build_index_record(
                INDEX_VECTOR_SEGMENT,
                checkpoint_generation,
                next_record_id,
                &payload,
            ),
        });
        next_record_id += 1;
    }
    Ok(records)
}

#[cfg(test)]
fn encode_text_index_segment_payload(segment: &TextIndexSegmentPayload) -> Result<Vec<u8>> {
    let mut chunks = segment.chunks.iter().collect::<Vec<_>>();
    chunks.sort_by_key(|chunk| catalog_id_number(&chunk.chunk_id).unwrap_or(u64::MAX));
    let mut terms = segment.terms.iter().collect::<Vec<_>>();
    terms.sort_by(|left, right| left.term.as_bytes().cmp(right.term.as_bytes()));

    let mut payload = Vec::new();
    payload.extend_from_slice(&TEXT_SEGMENT_SCHEMA_VERSION.to_le_bytes());
    payload.extend_from_slice(&catalog_id_number(&segment.collection_id)?.to_le_bytes());
    payload.extend_from_slice(&segment.covered_commit_sequence_no.to_le_bytes());
    payload.extend_from_slice(&TEXT_INDEX_KIND_BM25.to_le_bytes());
    payload.extend_from_slice(&DEFAULT_EN_TOKENIZER_ID.to_le_bytes());
    payload.extend_from_slice(&(crate::index::text::BM25_K1 as f32).to_le_bytes());
    payload.extend_from_slice(&(crate::index::text::BM25_B as f32).to_le_bytes());
    push_u64_len(&mut payload, chunks.len());
    for chunk in chunks {
        payload.extend_from_slice(&catalog_id_number(&chunk.chunk_id)?.to_le_bytes());
        payload.extend_from_slice(&catalog_id_number(&chunk.document_id)?.to_le_bytes());
        push_u64_len(&mut payload, chunk.token_count);
    }
    push_u64_len(&mut payload, terms.len());
    for term in terms {
        let mut postings = term.postings.iter().collect::<Vec<_>>();
        postings.sort_by_key(|posting| catalog_id_number(&posting.chunk_id).unwrap_or(u64::MAX));
        push_index_string(&mut payload, &term.term)?;
        push_u64_len(&mut payload, postings.len());
        push_u64_len(&mut payload, postings.len());
        for posting in postings {
            payload.extend_from_slice(&catalog_id_number(&posting.chunk_id)?.to_le_bytes());
            push_u64_len(&mut payload, posting.term_frequency);
        }
    }
    let segment_checksum = checksum32(&payload);
    payload.extend_from_slice(&segment_checksum.to_le_bytes());
    Ok(payload)
}

#[cfg(test)]
fn encode_vector_index_segment_payload(segment: &VectorIndexSegmentPayload) -> Result<Vec<u8>> {
    let dim = u32::try_from(segment.dim).map_err(|_| {
        Error::new(
            ErrorCode::InvalidVectorLength,
            "vector segment dimension exceeds u32",
        )
    })?;
    let mut vectors = segment.vectors.iter().collect::<Vec<_>>();
    vectors.sort_by_key(|vector| catalog_id_number(&vector.chunk_id).unwrap_or(u64::MAX));

    let mut payload = Vec::new();
    payload.extend_from_slice(&INDEX_SEGMENT_SCHEMA_VERSION.to_le_bytes());
    payload.extend_from_slice(&catalog_id_number(&segment.collection_id)?.to_le_bytes());
    payload.extend_from_slice(&segment.covered_commit_sequence_no.to_le_bytes());
    payload.extend_from_slice(&VECTOR_INDEX_KIND_EXACT_F32.to_le_bytes());
    payload.extend_from_slice(&dim.to_le_bytes());
    payload.push(distance_metric_segment_id(segment.distance_metric));
    push_index_optional_string(&mut payload, segment.embedding_model.as_deref())?;
    payload.push(u8::from(segment.vectors_normalized));
    push_u64_len(&mut payload, vectors.len());
    for entry in vectors {
        if entry.vector.len() != segment.dim {
            return Err(Error::new(
                ErrorCode::InvalidVectorLength,
                "vector segment entry dimension mismatch",
            )
            .with_detail("expected", segment.dim)
            .with_detail("actual", entry.vector.len()));
        }
        if entry.vector.iter().any(|value| !value.is_finite()) {
            return Err(Error::new(
                ErrorCode::InvalidVectorLength,
                "vector segment entries must be finite f32 values",
            ));
        }
        payload.extend_from_slice(&catalog_id_number(&entry.chunk_id)?.to_le_bytes());
        payload.extend_from_slice(&catalog_id_number(&entry.embedding_id)?.to_le_bytes());
        for value in &entry.vector {
            payload.extend_from_slice(&value.to_le_bytes());
        }
    }
    let segment_checksum = checksum32(&payload);
    payload.extend_from_slice(&segment_checksum.to_le_bytes());
    Ok(payload)
}

fn build_index_commit_metadata_payload(
    checkpoint: &IndexCheckpointMetadata,
    segment_record_count: u64,
    index_extent_table_offset: u64,
    index_extent_table_length: u64,
    index_extent_table_checksum: u32,
    index_checksum: u32,
) -> [u8; INDEX_COMMIT_METADATA_LEN] {
    let mut payload = [0; INDEX_COMMIT_METADATA_LEN];
    payload[..2].copy_from_slice(&1u16.to_le_bytes());
    payload[2..10].copy_from_slice(&checkpoint.covered_last_batch_id.to_le_bytes());
    payload[10..18].copy_from_slice(&checkpoint.covered_last_commit_sequence_no.to_le_bytes());
    payload[18..26].copy_from_slice(&segment_record_count.to_le_bytes());
    payload[26..30].copy_from_slice(&index_checksum.to_le_bytes());
    payload[30..38].copy_from_slice(&segment_record_count.to_le_bytes());
    payload[38..46].copy_from_slice(&index_extent_table_offset.to_le_bytes());
    payload[46..54].copy_from_slice(&index_extent_table_length.to_le_bytes());
    payload[54..58].copy_from_slice(&index_extent_table_checksum.to_le_bytes());
    payload
}

fn checked_segment_record_count(checkpoint: &IndexCheckpointMetadata) -> Result<u64> {
    let text_segments = u64::try_from(checkpoint.text_segment_count).map_err(|_| {
        Error::new(
            ErrorCode::FileFormatError,
            "text index segment count exceeds u32",
        )
    })?;
    let vector_segments = u64::try_from(checkpoint.vector_segment_count).map_err(|_| {
        Error::new(
            ErrorCode::FileFormatError,
            "vector index segment count exceeds u32",
        )
    })?;
    text_segments.checked_add(vector_segments).ok_or_else(|| {
        Error::new(
            ErrorCode::FileFormatError,
            "index segment record count overflow",
        )
    })
}

fn checked_writer_add(left: u64, right: u64, message: &'static str) -> Result<u64> {
    left.checked_add(right)
        .ok_or_else(|| Error::new(ErrorCode::FileFormatError, message))
}

#[cfg(test)]
fn push_u32_len(bytes: &mut Vec<u8>, value: usize, field: &'static str) -> Result<()> {
    let value = u32::try_from(value)
        .map_err(|_| Error::new(ErrorCode::FileFormatError, format!("{field} exceeds u32")))?;
    bytes.extend_from_slice(&value.to_le_bytes());
    Ok(())
}

#[cfg(test)]
fn push_u64_len(bytes: &mut Vec<u8>, value: usize) {
    bytes.extend_from_slice(&(value as u64).to_le_bytes());
}

#[cfg(test)]
fn push_index_string(bytes: &mut Vec<u8>, value: &str) -> Result<()> {
    push_u32_len(bytes, value.len(), "index string length")?;
    bytes.extend_from_slice(value.as_bytes());
    Ok(())
}

#[cfg(test)]
fn push_index_optional_string(bytes: &mut Vec<u8>, value: Option<&str>) -> Result<()> {
    match value {
        Some(value) => {
            bytes.push(1);
            push_index_string(bytes, value)?;
        }
        None => bytes.push(0),
    }
    Ok(())
}

pub(crate) fn catalog_id_number(value: &str) -> Result<u64> {
    let suffix = value
        .rsplit_once('_')
        .map(|(_, suffix)| suffix)
        .unwrap_or(value);
    suffix.parse::<u64>().map_err(|_| {
        Error::new(
            ErrorCode::FileFormatError,
            format!("catalog id {value} cannot be encoded in index segment"),
        )
    })
}

#[cfg(test)]
fn catalog_id_string(prefix: &str, value: u64) -> String {
    format!("{prefix}{value:016}")
}

pub(crate) fn catalog_id_text<'a>(
    bytes: &'a mut [u8; 24],
    prefix: &[u8; 4],
    value: u64,
) -> &'a str {
    bytes[..4].copy_from_slice(prefix);
    let mut output = &mut bytes[4..];
    write!(output, "{value:016}").expect("decimal u64 fits in twenty bytes");
    let len = 24 - output.len();
    std::str::from_utf8(&bytes[..len]).expect("catalog ID prefixes are ASCII")
}

fn distance_metric_segment_id(metric: DistanceMetric) -> u8 {
    match metric {
        DistanceMetric::Cosine => 1,
        DistanceMetric::DotProduct => 2,
        DistanceMetric::Euclidean => 3,
    }
}

fn distance_metric_from_segment_id(value: u8) -> Option<DistanceMetric> {
    match value {
        1 => Some(DistanceMetric::Cosine),
        2 => Some(DistanceMetric::DotProduct),
        3 => Some(DistanceMetric::Euclidean),
        _ => None,
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct IndexExtentEntry {
    index_record_id: u64,
    record_type: u16,
    stream_id: u64,
    suffix_relative_offset: u64,
    length: u64,
    checkpoint_generation: u64,
    extent_record_checksum: u32,
}

type IndexDirectory = crate::json::ReservedVec<IndexExtentEntry>;

#[cfg(test)]
fn encode_extent_table(entries: &[IndexExtentEntry]) -> Vec<u8> {
    let mut table = Vec::with_capacity(entries.len() * INDEX_EXTENT_ENTRY_LEN);
    for entry in entries {
        table.extend_from_slice(&encode_extent_entry(entry));
    }
    debug_assert_eq!(table.len(), entries.len() * INDEX_EXTENT_ENTRY_LEN);
    table
}

fn encode_extent_entry(entry: &IndexExtentEntry) -> [u8; INDEX_EXTENT_ENTRY_LEN] {
    let mut bytes = [0u8; INDEX_EXTENT_ENTRY_LEN];
    bytes[0..8].copy_from_slice(&entry.index_record_id.to_le_bytes());
    bytes[8..10].copy_from_slice(&entry.record_type.to_le_bytes());
    bytes[10..18].copy_from_slice(&entry.stream_id.to_le_bytes());
    bytes[18..26].copy_from_slice(&entry.suffix_relative_offset.to_le_bytes());
    bytes[26..34].copy_from_slice(&entry.length.to_le_bytes());
    bytes[34..42].copy_from_slice(&entry.checkpoint_generation.to_le_bytes());
    bytes[42..46].copy_from_slice(&entry.extent_record_checksum.to_le_bytes());
    bytes
}

#[derive(Debug)]
struct IndexSuffixDiagnostic {
    code: &'static str,
    message: String,
    offset: u64,
    extent_id: Option<String>,
}

struct ParsedIndexSuffix {
    report: IndexSuffixReport,
    #[cfg(test)]
    checkpoint: Option<IndexCheckpointPayload>,
    lazy_vectors: Option<LazyVectorCheckpoint>,
    diagnostics: Vec<IndexSuffixDiagnostic>,
}

#[derive(Clone, Copy)]
struct IndexReadOptions {
    budget_bytes: u64,
    mode: IndexReadMode,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum IndexReadMode {
    #[cfg(test)]
    Full,
    Summary,
    #[cfg(test)]
    LazyVectors,
    LazyIndexes,
}

#[cfg(test)]
impl From<u64> for IndexReadOptions {
    fn from(budget_bytes: u64) -> Self {
        Self {
            budget_bytes,
            mode: IndexReadMode::Full,
        }
    }
}

#[cfg(test)]
fn read_index_checkpoint_suffix<R: Read + Seek>(
    reader: &mut R,
    offset: u64,
    file_len: u64,
    expected_critical_end_offset: u64,
    expected_last_batch_id: u64,
    expected_last_commit_sequence_no: u64,
    options: impl Into<IndexReadOptions>,
) -> Result<std::result::Result<ParsedIndexSuffix, IndexSuffixDiagnostic>> {
    read_index_checkpoint_suffix_in(
        reader,
        offset,
        file_len,
        expected_critical_end_offset,
        expected_last_batch_id,
        expected_last_commit_sequence_no,
        (
            options.into(),
            &crate::json::JsonOperationContext::standalone(),
        ),
    )
}

fn read_index_checkpoint_suffix_in<R: Read + Seek>(
    reader: &mut R,
    offset: u64,
    file_len: u64,
    expected_critical_end_offset: u64,
    expected_last_batch_id: u64,
    expected_last_commit_sequence_no: u64,
    (options, context): (IndexReadOptions, &crate::json::JsonOperationContext),
) -> Result<std::result::Result<ParsedIndexSuffix, IndexSuffixDiagnostic>> {
    reader.seek(SeekFrom::Start(offset))?;
    let mut header = [0u8; INDEX_CHECKPOINT_HEADER_LEN];
    if let Err(err) = reader.read_exact(&mut header) {
        if err.kind() == std::io::ErrorKind::UnexpectedEof {
            return Ok(Err(index_suffix_diagnostic(
                "index_checkpoint_incomplete",
                "ignored incomplete index checkpoint header after last valid commit",
                offset,
            )));
        }
        return Err(Error::from(err));
    }

    if &header[0..8] != INDEX_CHECKPOINT_MAGIC {
        return Ok(Err(index_suffix_diagnostic(
            "index_suffix_unreadable",
            "ignored rebuildable suffix with unknown checkpoint magic",
            offset,
        )));
    }
    let version = read_u16(&header[8..10]);
    let header_len = read_u16(&header[10..12]) as usize;
    let checkpoint_generation = read_u64(&header[12..20]);
    let checkpoint_start_offset = read_u64(&header[20..28]);
    let covered_critical_end_offset = read_u64(&header[28..36]);
    let covered_last_batch_id = read_u64(&header[36..44]);
    let covered_last_commit_sequence_no = read_u64(&header[44..52]);
    let coverage_kind = header[52];
    let coverage_hash = read_u32(&header[53..57]);
    let index_record_count = read_u64(&header[57..65]);
    let index_extent_count = read_u64(&header[65..73]);
    let extent_table_offset = read_u64(&header[73..81]);
    let extent_table_len = read_u64(&header[81..89]);
    let extent_table_checksum = read_u32(&header[89..93]);
    let payload_len = read_u64(&header[93..101]);
    let payload_checksum = read_u32(&header[101..105]);

    if version != INDEX_CHECKPOINT_VERSION || header_len != INDEX_CHECKPOINT_HEADER_LEN {
        return Ok(Err(index_suffix_diagnostic(
            "unsupported_index_checkpoint",
            "ignored unsupported index checkpoint version or header length",
            offset,
        )));
    }
    if checkpoint_start_offset != offset
        || covered_critical_end_offset != expected_critical_end_offset
        || covered_last_batch_id != expected_last_batch_id
        || covered_last_commit_sequence_no != expected_last_commit_sequence_no
    {
        return Ok(Err(index_suffix_diagnostic(
            "stale_index_checkpoint",
            "ignored index checkpoint that does not cover the latest visible critical commit",
            offset,
        )));
    }
    if coverage_kind != INDEX_COVERAGE_FULL_VISIBLE_SNAPSHOT {
        return Ok(Err(index_suffix_diagnostic(
            "unsupported_index_checkpoint",
            "ignored index checkpoint with unsupported coverage kind",
            offset,
        )));
    }
    if index_record_count == 0 || index_extent_count != index_record_count {
        return Ok(Err(index_suffix_diagnostic(
            "unsupported_index_checkpoint",
            "ignored index checkpoint with invalid record or extent counts",
            offset,
        )));
    }
    let expected_extent_table_offset = match checked_index_add(
        INDEX_CHECKPOINT_HEADER_LEN as u64,
        payload_len,
        offset,
        "index_checkpoint_length_overflow",
        "ignored index checkpoint with overflowing payload boundary",
    ) {
        Ok(value) => value,
        Err(diagnostic) => return Ok(Err(diagnostic)),
    };
    if extent_table_offset != expected_extent_table_offset {
        return Ok(Err(index_suffix_diagnostic(
            "index_extent_table_mismatch",
            "ignored index checkpoint with invalid extent table boundary",
            offset,
        )));
    }
    let expected_len = match checked_index_add(
        expected_extent_table_offset,
        extent_table_len,
        offset,
        "index_checkpoint_length_overflow",
        "ignored index checkpoint with overflowing extent table boundary",
    )
    .and_then(|value| {
        checked_index_add(
            value,
            INDEX_CHECKPOINT_TRAILER_LEN as u64,
            offset,
            "index_checkpoint_length_overflow",
            "ignored index checkpoint with overflowing trailer boundary",
        )
    }) {
        Ok(value) => value,
        Err(diagnostic) => return Ok(Err(diagnostic)),
    };
    let suffix_len = match file_len.checked_sub(offset) {
        Some(value) => value,
        None => {
            return Ok(Err(index_suffix_diagnostic(
                "index_checkpoint_length_overflow",
                "ignored index checkpoint whose offset is beyond file length",
                offset,
            )));
        }
    };
    if expected_len > suffix_len {
        return Ok(Err(index_suffix_diagnostic(
            "index_checkpoint_incomplete",
            "ignored incomplete index checkpoint suffix after last valid commit",
            offset,
        )));
    }
    if expected_len < suffix_len {
        return Ok(Err(index_suffix_diagnostic(
            "index_suffix_trailing_bytes",
            "ignored index checkpoint with trailing bytes after complete marker",
            checked_diagnostic_offset(offset, expected_len),
        )));
    }
    let expected_extent_table_len =
        match index_extent_count.checked_mul(INDEX_EXTENT_ENTRY_LEN as u64) {
            Some(value) => value,
            None => {
                return Ok(Err(index_suffix_diagnostic(
                    "index_checkpoint_length_overflow",
                    "ignored index checkpoint with overflowing extent table count",
                    offset,
                )));
            }
        };
    if extent_table_len != expected_extent_table_len
        || payload_len > MAX_INDEX_CHECKPOINT_PAYLOAD_LEN
    {
        return Ok(Err(index_suffix_diagnostic(
            "unsupported_index_checkpoint",
            "ignored unsupported index checkpoint payload or extent table size",
            offset,
        )));
    }

    let retained_payload_len = match options.mode {
        IndexReadMode::LazyIndexes => 0,
        #[cfg(test)]
        IndexReadMode::LazyVectors => 0,
        _ => payload_len,
    };
    if options.mode != IndexReadMode::LazyIndexes
        && retained_payload_len
            .checked_add(extent_table_len)
            .map_or(true, |bytes| bytes > options.budget_bytes)
    {
        return Ok(Err(index_suffix_diagnostic(
            "index_checkpoint_load_budget_exceeded",
            "checkpoint decoding deferred because it exceeds the runtime load budget",
            offset,
        )));
    }
    if let Err(diagnostic) = check_index_checkpoint_trailer(reader, file_len)? {
        return Ok(Err(diagnostic));
    }
    reader.seek(SeekFrom::Start(offset + INDEX_CHECKPOINT_HEADER_LEN as u64))?;
    let mut aggregate_checksum = Crc32c::default();
    checksum_index_bytes(reader, payload_len, &mut aggregate_checksum)?;
    // The aggregate payload checksum is advisory; extent and segment checks below
    // decide whether the checkpoint can be used.
    let _checkpoint_payload_checksum_mismatch = aggregate_checksum.finish() != payload_checksum;
    let table_len = match usize::try_from(extent_table_len) {
        Ok(value) => value,
        Err(_) => {
            return Ok(Err(index_suffix_diagnostic(
                "index_checkpoint_length_overflow",
                "ignored index checkpoint whose extent table exceeds address space",
                offset,
            )));
        }
    };
    let record_count = table_len / INDEX_EXTENT_ENTRY_LEN;
    let mut extent_table =
        match crate::json::ReservedVec::zeroed(table_len, context, "operation_scratch") {
            Ok(bytes) => bytes,
            Err(error) if crate::json::is_memory_budget_error(&error) => {
                // A rejected allocation must not hide table damage or a real read error.
                let mut checksum = Crc32c::default();
                checksum_index_bytes(reader, extent_table_len, &mut checksum)?;
                if checksum.finish() != extent_table_checksum {
                    return Ok(Err(index_suffix_diagnostic(
                        "index_extent_table_mismatch",
                        "ignored index checkpoint with extent table checksum mismatch",
                        checked_diagnostic_offset(offset, extent_table_offset),
                    )));
                }
                if let Err(diagnostic) = check_index_checkpoint_trailer(reader, file_len)? {
                    return Ok(Err(diagnostic));
                }
                return Err(error);
            }
            Err(error) => return Err(error),
        };
    for part in extent_table.as_mut_slice().chunks_mut(64 * 1024) {
        reader.read_exact(part)?;
    }
    let extent_table = extent_table.as_slice();
    if checksum32(extent_table) != extent_table_checksum {
        return Ok(Err(index_suffix_diagnostic(
            "index_extent_table_mismatch",
            "ignored index checkpoint with extent table checksum mismatch",
            checked_diagnostic_offset(offset, extent_table_offset),
        )));
    }
    #[cfg(test)]
    if options.mode == IndexReadMode::LazyVectors {
        let retained_bytes = extent_table
            .chunks_exact(INDEX_EXTENT_ENTRY_LEN)
            .map(parse_extent_entry)
            .filter(|extent| extent.record_type != INDEX_VECTOR_SEGMENT)
            .try_fold(extent_table_len, |total, extent| {
                total.checked_add(extent.length)
            });
        if retained_bytes.map_or(true, |bytes| bytes > options.budget_bytes) {
            return Ok(Err(index_suffix_diagnostic(
                "index_checkpoint_load_budget_exceeded",
                "checkpoint text and directory decoding exceeds the runtime load budget",
                offset,
            )));
        }
    }
    let mut records_work = context.workspace_reservation();
    let mut records = BTreeMap::<u64, ParsedIndexRecord>::new();
    for entry in extent_table.chunks_exact(INDEX_EXTENT_ENTRY_LEN) {
        let extent = parse_extent_entry(entry);
        let index_record = match parse_index_record_from_extent(
            reader,
            payload_len as usize,
            &extent,
            checkpoint_generation,
            offset,
        ) {
            Ok(record) => record,
            Err(IndexReadError::Invalid(diagnostic)) => return Ok(Err(diagnostic)),
            Err(IndexReadError::Io(error)) => return Err(error.into()),
            Err(IndexReadError::Resource(error)) => return Err(error),
        };
        if records.contains_key(&index_record.index_record_id) {
            return Ok(Err(index_suffix_diagnostic(
                "index_extent_table_mismatch",
                "ignored index checkpoint with duplicate index record id",
                checked_diagnostic_offset(offset, extent.suffix_relative_offset),
            )));
        }
        let needed = crate::json::btree_workspace::<u64, ParsedIndexRecord>(records.len() + 1);
        if let Err(error) = records_work.grow_to(needed, needed, "operation_scratch") {
            // CRC status is already known for every parsed record, even though
            // normal validation defers the segment-specific diagnostic.
            for record in records.values().chain(std::iter::once(&index_record)) {
                if matches!(
                    record.record_type,
                    INDEX_TEXT_SEGMENT | INDEX_VECTOR_SEGMENT
                ) {
                    if let Err(diagnostic) = validate_record_segment_checksum(record, offset) {
                        return Ok(Err(diagnostic));
                    }
                }
            }
            return Err(error);
        }
        records.insert(index_record.index_record_id, index_record);
    }
    let mut metadata_records = records
        .values()
        .filter(|record| record.record_type == INDEX_COMMIT_METADATA);
    let index_record = match (metadata_records.next(), metadata_records.next()) {
        (Some(record), None) if record.index_record_id == 1 => record,
        _ => {
            return Ok(Err(index_suffix_diagnostic(
                "index_commit_metadata_missing",
                "ignored index checkpoint without a valid IndexCommitMetadata record",
                checked_diagnostic_offset(offset, INDEX_CHECKPOINT_HEADER_LEN as u64),
            )));
        }
    };
    let metadata = match with_checked_index_payload(reader, index_record, context, |source| {
        let mut payload = [0u8; INDEX_COMMIT_METADATA_LEN];
        let len = (index_record.payload.len as usize).min(payload.len());
        source.read_exact(&mut payload[..len])?;
        // Pass a non-canonical length without allocating an untrusted metadata payload.
        let payload = if index_record.payload.len == payload.len() as u64 {
            &payload[..]
        } else {
            &[]
        };
        Ok(parse_index_commit_metadata_payload(
            payload,
            covered_last_batch_id,
            covered_last_commit_sequence_no,
            extent_table,
            &records,
            offset,
        )?)
    }) {
        Ok(metadata) => metadata,
        Err(IndexReadError::Invalid(diagnostic)) => return Ok(Err(diagnostic)),
        Err(IndexReadError::Io(error)) => return Err(error.into()),
        Err(IndexReadError::Resource(error)) => return Err(error),
    };
    let mut referenced_checksum = index_commit_hasher(
        covered_last_batch_id,
        covered_last_commit_sequence_no,
        checkpoint_generation,
    );
    for entry in extent_table[metadata.extent_range.clone()].chunks_exact(INDEX_EXTENT_ENTRY_LEN) {
        let record = &records[&parse_extent_entry(entry).index_record_id];
        referenced_checksum.update(entry);
        reader.seek(SeekFrom::Start(record.offset))?;
        checksum_index_bytes(reader, record.length, &mut referenced_checksum)?;
    }
    if referenced_checksum.finish() != metadata.index_checksum {
        return Ok(Err(index_suffix_diagnostic(
            "index_commit_checksum_mismatch",
            "ignored index checkpoint with IndexCommitMetadata checksum mismatch",
            offset + INDEX_CHECKPOINT_HEADER_LEN as u64 + INDEX_RECORD_HEADER_LEN as u64,
        )));
    }
    if let Err(diagnostic) =
        validate_index_checkpoint_directory(&records, record_count, &metadata, extent_table, offset)
    {
        return Ok(Err(diagnostic));
    }
    let segment_summary = match validate_index_segments(
        reader,
        &records,
        metadata.referenced_record_ids(extent_table),
        covered_last_commit_sequence_no,
        offset,
        options.mode,
        context,
    ) {
        Ok(summary) => summary,
        Err(IndexReadError::Invalid(diagnostic)) => return Ok(Err(diagnostic)),
        Err(IndexReadError::Io(error)) => return Err(error.into()),
        Err(IndexReadError::Resource(error)) => return Err(error),
    };

    drop(records);
    drop(records_work);
    if let Err(diagnostic) = check_index_checkpoint_trailer(reader, file_len)? {
        return Ok(Err(diagnostic));
    }
    let mut diagnostics = Vec::new();
    if _checkpoint_payload_checksum_mismatch {
        diagnostics.push(index_suffix_diagnostic(
            "checkpoint_payload_checksum_mismatch",
            "index checkpoint aggregate payload checksum mismatch; per-record and segment checks passed",
            checked_diagnostic_offset(offset, INDEX_CHECKPOINT_HEADER_LEN as u64),
        ));
    }
    let status = if metadata.index_record_count == 0 {
        "metadata_record_only"
    } else {
        "persistent_checkpoint"
    };
    let build_mode = if metadata.index_record_count == 0 {
        "metadata_record_only"
    } else {
        "inline_full_checkpoint"
    };
    #[cfg(test)]
    let mut checkpoint = None;
    let lazy_vectors = match options.mode {
        #[cfg(test)]
        IndexReadMode::Full => {
            checkpoint = Some(IndexCheckpointPayload {
                metadata: IndexCheckpointMetadata {
                    covered_last_batch_id,
                    covered_last_commit_sequence_no,
                    coverage_hash,
                    collection_count: segment_summary.collection_count,
                    live_document_count: segment_summary.live_document_count,
                    live_chunk_count: segment_summary.live_chunk_count,
                    live_embedding_count: segment_summary.live_embedding_count,
                    text_segment_count: segment_summary.text_segment_count,
                    vector_segment_count: segment_summary.vector_segment_count,
                    build_mode: build_mode.into(),
                },
                text_segments: segment_summary.text_segments,
                vector_segments: segment_summary.vector_segments,
            });
            None
        }
        IndexReadMode::Summary => None,
        _ => {
            let entries = extent_table
                .chunks_exact(INDEX_EXTENT_ENTRY_LEN)
                .map(parse_extent_entry)
                .filter(|extent| {
                    matches!(
                        extent.record_type,
                        INDEX_TEXT_SEGMENT | INDEX_VECTOR_SEGMENT
                    )
                });
            let mut directory = IndexDirectory::with_capacity(
                entries.clone().count(),
                context,
                "operation_scratch",
            )?;
            for extent in entries {
                directory.push(extent);
            }
            Some(LazyVectorCheckpoint {
                #[cfg(test)]
                text_segments: segment_summary.text_segments,
                vectors: IndexSegmentSource::new(
                    offset,
                    payload_len as usize,
                    checkpoint_generation,
                    covered_last_commit_sequence_no,
                    directory,
                )?,
            })
        }
    };
    Ok(Ok(ParsedIndexSuffix {
        report: IndexSuffixReport {
            status: status.into(),
            offset,
            checkpoint_generation,
            covered_critical_end_offset,
            covered_last_batch_id,
            covered_last_commit_sequence_no,
            coverage_hash,
            collection_count: segment_summary.collection_count,
            live_document_count: segment_summary.live_document_count,
            live_chunk_count: segment_summary.live_chunk_count,
            live_embedding_count: segment_summary.live_embedding_count,
            text_segment_count: segment_summary.text_segment_count,
            vector_segment_count: segment_summary.vector_segment_count,
            index_record_count: record_count,
            index_extent_count: record_count,
            payload_len,
            build_mode: build_mode.into(),
        },
        #[cfg(test)]
        checkpoint,
        lazy_vectors,
        diagnostics,
    }))
}

fn check_index_checkpoint_trailer(
    reader: &mut (impl Read + Seek),
    file_len: u64,
) -> Result<std::result::Result<(), IndexSuffixDiagnostic>> {
    let trailer_offset = file_len - INDEX_CHECKPOINT_TRAILER_LEN as u64;
    let mut trailer = [0u8; INDEX_CHECKPOINT_TRAILER_LEN];
    reader.seek(SeekFrom::Start(trailer_offset))?;
    reader.read_exact(&mut trailer)?;
    if &trailer[0..4] != INDEX_COMPLETE_MARKER
        || read_u16(&trailer[4..6]) as usize != INDEX_CHECKPOINT_TRAILER_LEN
        || read_u32(&trailer[6..10]) != checksum32(&trailer[0..6])
    {
        return Ok(Err(index_suffix_diagnostic(
            "index_checkpoint_incomplete",
            "ignored index checkpoint with invalid complete marker or trailer checksum",
            trailer_offset,
        )));
    }
    Ok(Ok(()))
}

fn parse_extent_entry(bytes: &[u8]) -> IndexExtentEntry {
    debug_assert_eq!(bytes.len(), INDEX_EXTENT_ENTRY_LEN);
    IndexExtentEntry {
        index_record_id: read_u64(&bytes[0..8]),
        record_type: read_u16(&bytes[8..10]),
        stream_id: read_u64(&bytes[10..18]),
        suffix_relative_offset: read_u64(&bytes[18..26]),
        length: read_u64(&bytes[26..34]),
        checkpoint_generation: read_u64(&bytes[34..42]),
        extent_record_checksum: read_u32(&bytes[42..46]),
    }
}

#[cfg(test)]
fn parse_extent_table(
    bytes: &[u8],
) -> std::result::Result<Vec<IndexExtentEntry>, IndexSuffixDiagnostic> {
    if bytes.len() % INDEX_EXTENT_ENTRY_LEN != 0 {
        return Err(index_suffix_diagnostic(
            "index_extent_table_mismatch",
            "ignored index checkpoint with non-canonical extent table length",
            0,
        ));
    }
    Ok(bytes
        .chunks_exact(INDEX_EXTENT_ENTRY_LEN)
        .map(parse_extent_entry)
        .collect())
}

fn checked_index_add(
    left: u64,
    right: u64,
    diagnostic_offset: u64,
    code: &'static str,
    message: &'static str,
) -> std::result::Result<u64, IndexSuffixDiagnostic> {
    left.checked_add(right)
        .ok_or_else(|| index_suffix_diagnostic(code, message, diagnostic_offset))
}

fn checked_diagnostic_offset(base: u64, relative: u64) -> u64 {
    base.checked_add(relative).unwrap_or(base)
}

struct ParsedIndexRecord {
    record_type: u16,
    index_record_id: u64,
    stream_id: u64,
    offset: u64,
    length: u64,
    payload: PayloadRef,
    segment_checksum_valid: bool,
}

fn checksum_index_bytes(
    reader: &mut impl Read,
    mut len: u64,
    checksum: &mut Crc32c,
) -> std::io::Result<()> {
    let mut buffer = [0u8; 64 * 1024];
    while len > 0 {
        let count = len.min(buffer.len() as u64) as usize;
        reader.read_exact(&mut buffer[..count])?;
        checksum.update(&buffer[..count]);
        len -= count as u64;
    }
    Ok(())
}

struct IndexChecksumReader<R> {
    source: R,
    checksum: Crc32c,
    bytes_read: u64,
}

impl<R: Read> Read for IndexChecksumReader<R> {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        let count = self.source.read(buffer)?;
        self.checksum.update(&buffer[..count]);
        self.bytes_read += count as u64;
        Ok(count)
    }
}

fn with_checked_index_payload<R: Read + Seek, T>(
    reader: &mut R,
    record: &ParsedIndexRecord,
    context: &crate::json::JsonOperationContext,
    mut decode: impl FnMut(&mut dyn Read) -> std::result::Result<T, IndexReadError>,
) -> std::result::Result<T, IndexReadError> {
    // Metadata is decoded through a fixed array; only segments benefit from read-ahead.
    let mut input_work = if record.record_type == INDEX_COMMIT_METADATA {
        None
    } else {
        Some(
            PayloadInputReservation::new(
                context,
                record.record_type,
                record.payload.len,
                crate::json::JsonInputKind::StoredJson,
            )
            .map_err(IndexReadError::Resource)?,
        )
    };
    loop {
        reader.seek(SeekFrom::Start(record.payload.offset))?;
        let mut source = IndexChecksumReader {
            source: BufReader::with_capacity(
                input_work
                    .as_ref()
                    .map_or(0, PayloadInputReservation::capacity),
                (&mut *reader).take(record.payload.len),
            ),
            checksum: Crc32c::default(),
            bytes_read: 0,
        };
        let result = decode(&mut source);
        if matches!(&result, Err(IndexReadError::Io(_))) {
            return result;
        }
        // Recheck the bytes actually decoded, including the tail after semantic errors.
        // The source may have changed since the record's framing was validated.
        let mut buffer = [0u8; 64 * 1024];
        while source.bytes_read < record.payload.len {
            let count = (record.payload.len - source.bytes_read).min(buffer.len() as u64) as usize;
            source.read_exact(&mut buffer[..count])?;
        }
        if source.checksum.finish() as u64 != record.payload.checksum {
            return Err(index_extent_diagnostic(
                "index_record_payload_checksum_mismatch",
                "ignored index checkpoint with index record payload checksum mismatch",
                record.payload.offset,
                record.index_record_id,
            )
            .into());
        }
        let budget_error = matches!(
            &result,
            Err(IndexReadError::Resource(error)) if crate::json::is_memory_budget_error(error)
        );
        drop(source);
        if budget_error {
            if let Some(work) = input_work.as_mut().filter(|work| work.capacity() > 0) {
                drop(result);
                work.disable_optional_buffering();
                continue;
            }
        }
        return result;
    }
}

fn validate_record_segment_checksum(
    record: &ParsedIndexRecord,
    checkpoint_start_offset: u64,
) -> std::result::Result<(), IndexSuffixDiagnostic> {
    if !record.segment_checksum_valid {
        return Err(index_suffix_diagnostic(
            "index_segment_checksum_mismatch",
            if record.payload.len < 4 {
                "ignored index checkpoint with truncated index segment payload"
            } else {
                "ignored index checkpoint with index segment checksum mismatch"
            },
            checkpoint_start_offset + INDEX_CHECKPOINT_HEADER_LEN as u64,
        ));
    }
    Ok(())
}

fn index_extent_range(
    checkpoint_payload_len: usize,
    extent: &IndexExtentEntry,
    checkpoint_generation: u64,
    checkpoint_start_offset: u64,
) -> std::result::Result<(u64, usize), IndexReadError> {
    let extent_offset = checked_index_add(
        checkpoint_start_offset,
        extent.suffix_relative_offset,
        checkpoint_start_offset,
        "index_record_length_overflow",
        "ignored index checkpoint with overflowing absolute extent offset",
    )?;
    if extent.checkpoint_generation != checkpoint_generation
        || extent.suffix_relative_offset < INDEX_CHECKPOINT_HEADER_LEN as u64
    {
        return Err(index_suffix_diagnostic(
            "index_extent_table_mismatch",
            "ignored index checkpoint with extent outside checkpoint payload",
            checked_diagnostic_offset(checkpoint_start_offset, INDEX_CHECKPOINT_HEADER_LEN as u64),
        )
        .into());
    }
    let start_u64 = extent.suffix_relative_offset - INDEX_CHECKPOINT_HEADER_LEN as u64;
    let start = usize::try_from(start_u64).map_err(|_| {
        index_suffix_diagnostic(
            "index_record_length_overflow",
            "ignored index checkpoint with index record start offset that does not fit usize",
            extent_offset,
        )
    })?;
    let length = usize::try_from(extent.length).map_err(|_| {
        index_suffix_diagnostic(
            "index_record_length_overflow",
            "ignored index checkpoint with index record length that does not fit usize",
            extent_offset,
        )
    })?;
    let Some(end) = start.checked_add(length) else {
        return Err(index_suffix_diagnostic(
            "index_record_length_overflow",
            "ignored index checkpoint with overflowing extent range",
            extent_offset,
        )
        .into());
    };
    if end > checkpoint_payload_len || length < INDEX_RECORD_HEADER_LEN {
        return Err(index_suffix_diagnostic(
            "index_extent_table_mismatch",
            "ignored index checkpoint with extent outside checkpoint payload",
            extent_offset,
        )
        .into());
    }
    checked_index_add(
        extent_offset,
        extent.length,
        extent_offset,
        "index_record_length_overflow",
        "ignored index checkpoint with overflowing absolute extent boundary",
    )?;
    Ok((extent_offset, length))
}

fn parse_index_record_from_extent<R: Read + Seek>(
    reader: &mut R,
    checkpoint_payload_len: usize,
    extent: &IndexExtentEntry,
    checkpoint_generation: u64,
    checkpoint_start_offset: u64,
) -> std::result::Result<ParsedIndexRecord, IndexReadError> {
    let (extent_offset, length) = index_extent_range(
        checkpoint_payload_len,
        extent,
        checkpoint_generation,
        checkpoint_start_offset,
    )?;
    reader.seek(SeekFrom::Start(extent_offset))?;
    let mut record = [0u8; INDEX_RECORD_HEADER_LEN];
    reader.read_exact(&mut record)?;
    let mut record_crc = Crc32c::default();
    record_crc.update(&record);
    let actual_payload_len = length - INDEX_RECORD_HEADER_LEN;
    let mut payload_crc = Crc32c::default();
    let mut remaining = actual_payload_len.saturating_sub(4);
    let mut buffer = [0u8; 64 * 1024];
    while remaining > 0 {
        let count = remaining.min(buffer.len());
        reader.read_exact(&mut buffer[..count])?;
        record_crc.update(&buffer[..count]);
        payload_crc.update(&buffer[..count]);
        remaining -= count;
    }
    let segment_crc = payload_crc.finish();
    let mut tail = [0u8; 4];
    let tail = &mut tail[..actual_payload_len.min(4)];
    reader.read_exact(tail)?;
    record_crc.update(tail);
    payload_crc.update(tail);
    let segment_checksum_valid = tail.len() == 4 && read_u32(tail) == segment_crc;
    if record_crc.finish() != extent.extent_record_checksum {
        return Err(index_extent_diagnostic(
            "index_extent_record_checksum_mismatch",
            "ignored index checkpoint with extent record checksum mismatch",
            extent_offset,
            extent.index_record_id,
        )
        .into());
    }
    let mut parsed = parse_index_record_header(
        &record,
        extent,
        checkpoint_generation,
        extent_offset,
        length,
    )?;
    if payload_crc.finish() as u64 != parsed.payload.checksum {
        return Err(index_extent_diagnostic(
            "index_record_payload_checksum_mismatch",
            "ignored index checkpoint with index record payload checksum mismatch",
            parsed.payload.offset,
            extent.index_record_id,
        )
        .into());
    }
    parsed.segment_checksum_valid = segment_checksum_valid;
    Ok(parsed)
}

fn parse_index_record_header(
    record: &[u8; INDEX_RECORD_HEADER_LEN],
    extent: &IndexExtentEntry,
    checkpoint_generation: u64,
    extent_offset: u64,
    length: usize,
) -> std::result::Result<ParsedIndexRecord, IndexReadError> {
    if &record[0..4] != INDEX_RECORD_MAGIC {
        return Err(index_suffix_diagnostic(
            "index_record_magic_mismatch",
            "ignored index checkpoint with invalid index record magic",
            extent_offset,
        )
        .into());
    }
    let record_type = read_u16(&record[4..6]);
    let version = read_u16(&record[6..8]);
    let header_len = read_u16(&record[8..10]) as usize;
    let record_len = usize::try_from(read_u64(&record[10..18])).map_err(|_| {
        index_suffix_diagnostic(
            "index_record_length_overflow",
            "ignored index checkpoint with index record length that does not fit usize",
            extent_offset,
        )
    })?;
    let generation = read_u64(&record[18..26]);
    let index_record_id = read_u64(&record[26..34]);
    let payload_len = usize::try_from(read_u64(&record[34..42])).map_err(|_| {
        index_suffix_diagnostic(
            "index_record_length_overflow",
            "ignored index checkpoint with index record payload length that does not fit usize",
            extent_offset,
        )
    })?;
    let payload_checksum = read_u32(&record[42..46]);
    let header_checksum = read_u32(&record[46..50]);
    let expected_record_len = INDEX_RECORD_HEADER_LEN
        .checked_add(payload_len)
        .ok_or_else(|| {
            index_suffix_diagnostic(
                "index_record_length_overflow",
                "ignored index checkpoint with overflowing index record payload boundary",
                extent_offset,
            )
        })?;
    if version != 1
        || header_len != INDEX_RECORD_HEADER_LEN
        || record_len != length
        || generation != checkpoint_generation
        || index_record_id != extent.index_record_id
        || record_type != extent.record_type
        || expected_record_len != record_len
        || checksum32(&record[0..46]) != header_checksum
    {
        return Err(index_suffix_diagnostic(
            "index_record_header_mismatch",
            "ignored index checkpoint with invalid index record header",
            extent_offset,
        )
        .into());
    }
    Ok(ParsedIndexRecord {
        record_type,
        index_record_id,
        stream_id: extent.stream_id,
        offset: extent_offset,
        length: extent.length,
        payload: PayloadRef {
            offset: extent_offset + INDEX_RECORD_HEADER_LEN as u64,
            len: payload_len as u64,
            checksum: payload_checksum as u64,
        },
        segment_checksum_valid: false,
    })
}

struct ParsedIndexCommitMetadata {
    index_record_count: usize,
    index_checksum: u32,
    extent_range: std::ops::Range<usize>,
}

impl ParsedIndexCommitMetadata {
    fn referenced_record_ids<'a>(&self, table: &'a [u8]) -> impl Iterator<Item = u64> + 'a {
        table[self.extent_range.clone()]
            .chunks_exact(INDEX_EXTENT_ENTRY_LEN)
            .map(|entry| read_u64(&entry[..8]))
    }
}

fn validate_index_checkpoint_directory(
    records: &BTreeMap<u64, ParsedIndexRecord>,
    checkpoint_record_count: usize,
    metadata: &ParsedIndexCommitMetadata,
    extent_table: &[u8],
    checkpoint_start_offset: u64,
) -> std::result::Result<(), IndexSuffixDiagnostic> {
    if records.len() != checkpoint_record_count
        || checkpoint_record_count != metadata.index_record_count.saturating_add(1)
    {
        return Err(index_suffix_diagnostic(
            "index_checkpoint_directory_mismatch",
            "ignored index checkpoint whose record directory does not match IndexCommitMetadata references",
            checkpoint_start_offset + INDEX_CHECKPOINT_HEADER_LEN as u64,
        ));
    }
    if !records
        .keys()
        .copied()
        .filter(|&id| id != 1)
        .eq(metadata.referenced_record_ids(extent_table))
    {
        return Err(index_suffix_diagnostic(
            "index_checkpoint_directory_mismatch",
            "ignored index checkpoint with unreferenced or missing index records",
            checkpoint_start_offset + INDEX_CHECKPOINT_HEADER_LEN as u64,
        ));
    }
    Ok(())
}

#[derive(Default)]
struct ParsedIndexSegmentSummary {
    collection_count: usize,
    live_document_count: usize,
    live_chunk_count: usize,
    live_embedding_count: usize,
    text_segment_count: usize,
    vector_segment_count: usize,
    #[cfg(test)]
    text_segments: Vec<TextIndexSegmentPayload>,
    #[cfg(test)]
    vector_segments: Vec<VectorIndexSegmentPayload>,
}

fn validate_index_segments<R: Read + Seek>(
    reader: &mut R,
    records: &BTreeMap<u64, ParsedIndexRecord>,
    referenced_index_record_ids: impl Iterator<Item = u64>,
    expected_commit_sequence_no: u64,
    checkpoint_start_offset: u64,
    mode: IndexReadMode,
    context: &crate::json::JsonOperationContext,
) -> std::result::Result<ParsedIndexSegmentSummary, IndexReadError> {
    let mut summary = ParsedIndexSegmentSummary::default();
    let mut directory_work = context.workspace_reservation();
    let mut text_collections = BTreeSet::new();
    let mut vector_collections = BTreeSet::new();
    let mut document_ids = BTreeSet::new();

    for index_record_id in referenced_index_record_ids {
        let Some(record) = records.get(&index_record_id) else {
            return Err(index_suffix_diagnostic(
                "index_commit_metadata_invalid",
                "ignored index checkpoint with metadata reference to missing index segment",
                checkpoint_start_offset + INDEX_CHECKPOINT_HEADER_LEN as u64,
            )
            .into());
        };
        match record.record_type {
            INDEX_TEXT_SEGMENT => {
                validate_record_segment_checksum(record, checkpoint_start_offset)?;
                let segment = with_checked_index_payload(reader, record, context, |source| {
                    decode_text_index_segment_with(
                        &mut IndexPayloadReader::new(
                            source,
                            record.payload.len as usize - 4,
                            checkpoint_start_offset,
                        ),
                        expected_commit_sequence_no,
                        checkpoint_start_offset,
                        match mode {
                            #[cfg(test)]
                            IndexReadMode::Full | IndexReadMode::LazyVectors => {
                                TextDecodeMode::Retain
                            }
                            _ => TextDecodeMode::Summary,
                        },
                        context,
                    )
                })?;
                if record.stream_id != segment.collection_number {
                    return Err(index_suffix_diagnostic(
                        "index_extent_table_mismatch",
                        "ignored index checkpoint with text segment stream id mismatch",
                        checkpoint_start_offset + INDEX_CHECKPOINT_HEADER_LEN as u64,
                    )
                    .into());
                }
                if text_collections.contains(&segment.collection_number) {
                    return Err(index_suffix_diagnostic(
                        "index_segment_cardinality_mismatch",
                        "ignored index checkpoint with duplicate text segment for collection",
                        checkpoint_start_offset + INDEX_CHECKPOINT_HEADER_LEN as u64,
                    )
                    .into());
                }
                reserve_index_validation_directory(
                    &mut directory_work,
                    text_collections.len() + 1,
                    vector_collections.len(),
                    document_ids.len(),
                )?;
                text_collections.insert(segment.collection_number);
                summary.text_segment_count += 1;
                summary.live_chunk_count += segment.chunk_count;
                for &id in segment.document_ids.as_slice() {
                    if !document_ids.contains(&id) {
                        reserve_index_validation_directory(
                            &mut directory_work,
                            text_collections.len(),
                            vector_collections.len(),
                            document_ids.len() + 1,
                        )?;
                        document_ids.insert(id);
                    }
                }
                #[cfg(test)]
                if !matches!(mode, IndexReadMode::Summary | IndexReadMode::LazyIndexes) {
                    summary.text_segments.push(segment.payload);
                }
            }
            INDEX_VECTOR_SEGMENT => {
                validate_record_segment_checksum(record, checkpoint_start_offset)?;
                let segment = with_checked_index_payload(reader, record, context, |source| {
                    decode_vector_index_segment_in(
                        &mut IndexPayloadReader::new(
                            source,
                            record.payload.len as usize - 4,
                            checkpoint_start_offset,
                        ),
                        expected_commit_sequence_no,
                        checkpoint_start_offset,
                        match mode {
                            #[cfg(test)]
                            IndexReadMode::Full => VectorDecodeMode::Retain,
                            _ => VectorDecodeMode::Summary,
                        },
                    )
                })?;
                if record.stream_id != segment.collection_number {
                    return Err(index_suffix_diagnostic(
                        "index_extent_table_mismatch",
                        "ignored index checkpoint with vector segment stream id mismatch",
                        checkpoint_start_offset + INDEX_CHECKPOINT_HEADER_LEN as u64,
                    )
                    .into());
                }
                if vector_collections.contains(&segment.collection_number) {
                    return Err(index_suffix_diagnostic(
                        "index_segment_cardinality_mismatch",
                        "ignored index checkpoint with duplicate vector segment for collection",
                        checkpoint_start_offset + INDEX_CHECKPOINT_HEADER_LEN as u64,
                    )
                    .into());
                }
                reserve_index_validation_directory(
                    &mut directory_work,
                    text_collections.len(),
                    vector_collections.len() + 1,
                    document_ids.len(),
                )?;
                vector_collections.insert(segment.collection_number);
                summary.vector_segment_count += 1;
                summary.live_embedding_count += segment.vector_count;
                #[cfg(test)]
                if mode == IndexReadMode::Full {
                    summary.vector_segments.push(segment.payload.unwrap());
                }
            }
            _ => {
                return Err(index_suffix_diagnostic(
                    "unsupported_index_record",
                    "ignored index checkpoint with unsupported metadata-referenced index record type",
                    checkpoint_start_offset + INDEX_CHECKPOINT_HEADER_LEN as u64,
                ).into());
            }
        }
    }

    if text_collections != vector_collections {
        return Err(index_suffix_diagnostic(
            "index_segment_cardinality_mismatch",
            "ignored index checkpoint without exactly one full text and vector segment per collection",
            checkpoint_start_offset + INDEX_CHECKPOINT_HEADER_LEN as u64,
        ).into());
    }
    summary.collection_count = text_collections.len();
    summary.live_document_count = document_ids.len();
    Ok(summary)
}

fn reserve_index_validation_directory(
    work: &mut crate::json::WorkspaceReservation,
    text_collections: usize,
    vector_collections: usize,
    documents: usize,
) -> std::result::Result<(), IndexReadError> {
    let bytes = [text_collections, vector_collections, documents]
        .into_iter()
        .map(crate::json::btree_workspace::<u64, ()>)
        .map(|bytes| bytes as u128)
        .sum();
    work.grow_to_wide(bytes, bytes, "operation_scratch")
        .map_err(IndexReadError::Resource)?;
    Ok(())
}

struct ParsedTextIndexSegment {
    collection_number: u64,
    chunk_count: usize,
    document_ids: crate::json::ReservedVec<u64>,
    #[cfg(test)]
    payload: TextIndexSegmentPayload,
}

#[cfg(test)]
fn parse_text_index_segment_payload(
    payload: &[u8],
    expected_commit_sequence_no: u64,
    checkpoint_start_offset: u64,
) -> std::result::Result<ParsedTextIndexSegment, IndexReadError> {
    validate_segment_checksum(payload, checkpoint_start_offset)?;
    decode_text_index_segment(
        &mut IndexPayloadReader::new(
            &payload[..payload.len() - 4],
            payload.len() - 4,
            checkpoint_start_offset,
        ),
        expected_commit_sequence_no,
        checkpoint_start_offset,
    )
}

#[cfg(test)]
fn decode_text_index_segment<R: Read>(
    reader: &mut IndexPayloadReader<R>,
    expected_commit_sequence_no: u64,
    checkpoint_start_offset: u64,
) -> std::result::Result<ParsedTextIndexSegment, IndexReadError> {
    decode_text_index_segment_in(
        reader,
        expected_commit_sequence_no,
        checkpoint_start_offset,
        true,
        &crate::json::JsonOperationContext::standalone(),
    )
}

#[cfg(test)]
fn decode_text_index_segment_in<R: Read>(
    reader: &mut IndexPayloadReader<R>,
    expected_commit_sequence_no: u64,
    checkpoint_start_offset: u64,
    retain_entries: bool,
    context: &crate::json::JsonOperationContext,
) -> std::result::Result<ParsedTextIndexSegment, IndexReadError> {
    decode_text_index_segment_with(
        reader,
        expected_commit_sequence_no,
        checkpoint_start_offset,
        if retain_entries {
            TextDecodeMode::Retain
        } else {
            TextDecodeMode::Summary
        },
        context,
    )
}

pub(crate) enum TextSegmentEntry<'a> {
    Header {
        chunks: usize,
    },
    Chunk {
        id: u64,
        length: usize,
    },
    Term {
        text: &'a str,
        document_frequency: usize,
    },
    Posting {
        term: &'a str,
        chunk: u64,
        frequency: usize,
    },
}

pub(crate) type TextSegmentVisitor<'a> =
    dyn for<'entry> FnMut(TextSegmentEntry<'entry>) -> Result<()> + 'a;

enum TextDecodeMode<'a> {
    Summary,
    #[cfg(test)]
    Retain,
    Visit {
        error: &'a mut Option<Error>,
        visit: &'a mut TextSegmentVisitor<'a>,
    },
}

impl TextDecodeMode<'_> {
    fn emit(&mut self, entry: TextSegmentEntry<'_>) {
        if let Self::Visit { error, visit } = self {
            if error.is_none() {
                **error = visit(entry).err();
            }
        }
    }
}

fn decode_text_index_segment_with<R: Read>(
    reader: &mut IndexPayloadReader<R>,
    expected_commit_sequence_no: u64,
    checkpoint_start_offset: u64,
    mut mode: TextDecodeMode<'_>,
    context: &crate::json::JsonOperationContext,
) -> std::result::Result<ParsedTextIndexSegment, IndexReadError> {
    #[cfg(test)]
    let retain_entries = matches!(mode, TextDecodeMode::Retain);
    let schema_version = reader.read_u16("text segment schema version")?;
    let collection_id = reader.read_u64("text segment collection id")?;
    let covered_commit_sequence_no = reader.read_u64("text segment covered commit")?;
    let index_kind = reader.read_u16("text segment index kind")?;
    if schema_version != TEXT_SEGMENT_SCHEMA_VERSION
        || covered_commit_sequence_no != expected_commit_sequence_no
        || index_kind != TEXT_INDEX_KIND_BM25
    {
        return Err(index_suffix_diagnostic(
            "index_segment_header_mismatch",
            "ignored index checkpoint with invalid text segment header",
            checkpoint_start_offset + INDEX_CHECKPOINT_HEADER_LEN as u64,
        )
        .into());
    }
    if reader.read_u16("tokenizer id")? != DEFAULT_EN_TOKENIZER_ID {
        return Err(index_suffix_diagnostic(
            "index_tokenizer_mismatch",
            "ignored text segment with unsupported tokenizer identity",
            checkpoint_start_offset + INDEX_CHECKPOINT_HEADER_LEN as u64,
        )
        .into());
    }
    let k1 = reader.read_f32("bm25 k1")?;
    let b = reader.read_f32("bm25 b")?;
    if k1 != crate::index::text::BM25_K1 as f32 || b != crate::index::text::BM25_B as f32 {
        return Err(index_suffix_diagnostic(
            "index_bm25_params_mismatch",
            "ignored text segment with unsupported BM25 parameters",
            checkpoint_start_offset + INDEX_CHECKPOINT_HEADER_LEN as u64,
        )
        .into());
    }
    let chunk_count = reader.read_count("text segment chunk count", 24)?;
    mode.emit(TextSegmentEntry::Header {
        chunks: chunk_count,
    });
    let mut previous_chunk_id = None;
    let mut chunk_ids =
        crate::json::ReservedVec::with_capacity(chunk_count, context, "operation_scratch")
            .map_err(IndexReadError::Resource)?;
    let mut document_ids =
        crate::json::ReservedVec::with_capacity(chunk_count, context, "operation_scratch")
            .map_err(IndexReadError::Resource)?;
    #[cfg(test)]
    let mut chunks = if retain_entries {
        Vec::with_capacity(chunk_count)
    } else {
        Vec::new()
    };
    for _ in 0..chunk_count {
        let chunk_id = reader.read_u64("text segment chunk id")?;
        let document_id = reader.read_u64("text segment document id")?;
        let token_count = usize::try_from(reader.read_u64("text segment token count")?)
            .map_err(|_| reader.invalid("text segment token count"))?;
        if previous_chunk_id >= Some(chunk_id) {
            return Err(index_suffix_diagnostic(
                "index_segment_non_canonical",
                "ignored index checkpoint with unsorted or duplicate text segment chunks",
                checkpoint_start_offset + INDEX_CHECKPOINT_HEADER_LEN as u64,
            )
            .into());
        }
        previous_chunk_id = Some(chunk_id);
        chunk_ids.push(chunk_id);
        document_ids.push(document_id);
        mode.emit(TextSegmentEntry::Chunk {
            id: chunk_id,
            length: token_count.max(1),
        });
        #[cfg(test)]
        if retain_entries {
            chunks.push(TextIndexChunkEntry {
                chunk_id: catalog_id_string("chk_", chunk_id),
                #[cfg(test)]
                document_id: catalog_id_string("doc_", document_id),
                token_count,
            });
        }
    }
    let term_count = reader.read_count("text segment term count", 20)?;
    #[cfg(test)]
    let mut terms: Vec<TextIndexTermEntry> = if retain_entries {
        Vec::with_capacity(term_count)
    } else {
        Vec::new()
    };
    #[cfg(test)]
    let mut term_order = (!retain_entries).then(|| TextTermOrder::new(context));
    #[cfg(not(test))]
    let mut term_order = TextTermOrder::new(context);
    for _ in 0..term_count {
        #[cfg(test)]
        let term = if let Some(order) = &mut term_order {
            order.read_next(reader)?;
            None
        } else {
            let term = reader.read_string("text segment term")?;
            if terms
                .last()
                .is_some_and(|previous| previous.term.as_bytes() >= term.as_bytes())
            {
                return Err(text_term_order_error(checkpoint_start_offset));
            }
            Some(term)
        };
        #[cfg(not(test))]
        term_order.read_next(reader)?;
        let document_frequency = reader.read_u64("text segment document frequency")?;
        let posting_count = reader.read_count("text segment posting count", 16)?;
        if document_frequency != posting_count as u64 {
            return Err(index_suffix_diagnostic(
                "index_segment_non_canonical",
                "ignored index checkpoint with mismatched text segment posting counts",
                checkpoint_start_offset + INDEX_CHECKPOINT_HEADER_LEN as u64,
            )
            .into());
        }
        #[cfg(test)]
        let term_text = match &term_order {
            Some(order) => std::str::from_utf8(&order.previous).expect("validated text term"),
            None => term.as_deref().expect("retained text term"),
        };
        #[cfg(not(test))]
        let term_text = std::str::from_utf8(&term_order.previous).expect("validated text term");
        mode.emit(TextSegmentEntry::Term {
            text: term_text,
            document_frequency: posting_count,
        });
        let mut previous_posting_chunk_id = None;
        #[cfg(test)]
        let mut postings = if retain_entries {
            Vec::with_capacity(posting_count)
        } else {
            Vec::new()
        };
        for _ in 0..posting_count {
            let posting_chunk_id = reader.read_u64("text segment posting chunk id")?;
            let term_frequency = reader.read_u64("text segment term frequency")?;
            if previous_posting_chunk_id >= Some(posting_chunk_id)
                || chunk_ids
                    .as_slice()
                    .binary_search(&posting_chunk_id)
                    .is_err()
                || term_frequency == 0
            {
                return Err(index_suffix_diagnostic(
                    "index_segment_non_canonical",
                    "ignored index checkpoint with invalid text segment posting",
                    checkpoint_start_offset + INDEX_CHECKPOINT_HEADER_LEN as u64,
                )
                .into());
            }
            previous_posting_chunk_id = Some(posting_chunk_id);
            let term_frequency = usize::try_from(term_frequency)
                .map_err(|_| reader.invalid("text segment term frequency"))?;
            mode.emit(TextSegmentEntry::Posting {
                term: term_text,
                chunk: posting_chunk_id,
                frequency: term_frequency,
            });
            #[cfg(test)]
            if retain_entries {
                postings.push(TextIndexPostingEntry {
                    chunk_id: catalog_id_string("chk_", posting_chunk_id),
                    term_frequency,
                });
            }
        }
        #[cfg(test)]
        if let Some(term) = term {
            terms.push(TextIndexTermEntry { term, postings });
        }
    }
    reader.finish_before_checksum("text segment")?;
    let (mut document_ids, document_work) = document_ids.into_parts();
    document_ids.sort_unstable();
    document_ids.dedup();
    Ok(ParsedTextIndexSegment {
        collection_number: collection_id,
        chunk_count,
        document_ids: crate::json::ReservedVec::from_reserved(document_ids, document_work),
        #[cfg(test)]
        payload: TextIndexSegmentPayload {
            collection_id: if retain_entries {
                catalog_id_string("col_", collection_id)
            } else {
                String::new()
            },
            #[cfg(test)]
            covered_commit_sequence_no,
            chunks,
            terms,
        },
    })
}

fn text_term_order_error(checkpoint_start_offset: u64) -> IndexReadError {
    index_suffix_diagnostic(
        "index_segment_non_canonical",
        "ignored index checkpoint with unsorted or duplicate text segment terms",
        checkpoint_start_offset + INDEX_CHECKPOINT_HEADER_LEN as u64,
    )
    .into()
}

struct TextTermOrder {
    previous: Vec<u8>,
    seen: bool,
    work: crate::json::WorkspaceReservation,
}

impl TextTermOrder {
    fn new(context: &crate::json::JsonOperationContext) -> Self {
        Self {
            previous: Vec::new(),
            seen: false,
            work: context.workspace_reservation(),
        }
    }

    fn read_next<R: Read>(
        &mut self,
        reader: &mut IndexPayloadReader<R>,
    ) -> std::result::Result<(), IndexReadError> {
        let field = "text segment term";
        let len = reader.read_u32(field)? as usize;
        if len > reader.remaining {
            return Err(reader.invalid(field).into());
        }
        if len > self.previous.capacity() {
            let old_capacity = self.previous.capacity();
            let peak = old_capacity as u128 + len as u128;
            self.work
                .grow_to_wide(peak, peak, "operation_scratch")
                .map_err(IndexReadError::Resource)?;
            if let Err(error) = self.previous.try_reserve_exact(len - self.previous.len()) {
                self.work.retain(old_capacity);
                return Err(std::io::Error::other(error).into());
            }
            if self.previous.capacity() != len {
                self.previous = Vec::new();
                self.work.retain(0);
                return Err(std::io::Error::other(
                    "text term allocation exceeds requested capacity",
                )
                .into());
            }
            self.work.retain(len);
        }
        let old_len = self.previous.len();
        self.previous.resize(len, 0);
        let mut ordering = std::cmp::Ordering::Equal;
        let mut position = 0;
        let mut buffer = [0u8; 4096];
        // Compare each block before replacing it; unread previous bytes remain intact.
        // One admitted allocation therefore serves both adjacent terms.
        while position < len {
            let count = (len - position).min(buffer.len());
            reader.read_into(&mut buffer[..count], field)?;
            let end = position + count;
            if self.seen && ordering == std::cmp::Ordering::Equal {
                let common_end = end.min(old_len);
                if position < common_end {
                    ordering =
                        self.previous[position..common_end].cmp(&buffer[..common_end - position]);
                }
                if ordering == std::cmp::Ordering::Equal && end > old_len {
                    ordering = std::cmp::Ordering::Less;
                }
            }
            self.previous[position..end].copy_from_slice(&buffer[..count]);
            position = end;
        }
        std::str::from_utf8(&self.previous).map_err(|_| reader.invalid(field))?;
        if ordering == std::cmp::Ordering::Equal {
            ordering = old_len.cmp(&len);
        }
        if self.seen && ordering != std::cmp::Ordering::Less {
            return Err(text_term_order_error(reader.checkpoint_start_offset));
        }
        self.seen = true;
        Ok(())
    }
}

struct ParsedVectorIndexSegment {
    collection_number: u64,
    vector_count: usize,
    payload: Option<VectorIndexSegmentPayload>,
}

type VectorVisitor<'a> = dyn FnMut(u64, u64, &[f32]) -> Result<()> + 'a;

enum VectorDecodeMode<'a> {
    Summary,
    Retain,
    Visit {
        collection: &'a CollectionInfo,
        input: &'a crate::json::WorkspaceReservation,
        error: &'a mut Option<Error>,
        visit: &'a mut VectorVisitor<'a>,
    },
}

#[cfg(test)]
fn parse_vector_index_segment_payload(
    payload: &[u8],
    expected_commit_sequence_no: u64,
    checkpoint_start_offset: u64,
) -> std::result::Result<ParsedVectorIndexSegment, IndexReadError> {
    validate_segment_checksum(payload, checkpoint_start_offset)?;
    decode_vector_index_segment(
        &mut IndexPayloadReader::new(
            &payload[..payload.len() - 4],
            payload.len() - 4,
            checkpoint_start_offset,
        ),
        expected_commit_sequence_no,
        checkpoint_start_offset,
    )
}

#[cfg(test)]
fn decode_vector_index_segment<R: Read>(
    reader: &mut IndexPayloadReader<R>,
    expected_commit_sequence_no: u64,
    checkpoint_start_offset: u64,
) -> std::result::Result<ParsedVectorIndexSegment, IndexReadError> {
    decode_vector_index_segment_in(
        reader,
        expected_commit_sequence_no,
        checkpoint_start_offset,
        VectorDecodeMode::Retain,
    )
}

fn decode_vector_index_segment_in<R: Read>(
    reader: &mut IndexPayloadReader<R>,
    expected_commit_sequence_no: u64,
    checkpoint_start_offset: u64,
    mut mode: VectorDecodeMode<'_>,
) -> std::result::Result<ParsedVectorIndexSegment, IndexReadError> {
    let schema_version = reader.read_u16("vector segment schema version")?;
    let collection_id = reader.read_u64("vector segment collection id")?;
    let covered_commit_sequence_no = reader.read_u64("vector segment covered commit")?;
    let index_kind = reader.read_u16("vector segment index kind")?;
    let dim = reader.read_u32("vector segment dim")? as usize;
    let distance_metric_id = reader.read_u8("vector segment distance metric")?;
    let embedding_model = reader.read_optional_string(
        "vector segment embedding model",
        match &mode {
            VectorDecodeMode::Retain => IndexStringMode::Retain,
            VectorDecodeMode::Summary => IndexStringMode::Validate,
            VectorDecodeMode::Visit { collection, .. } => {
                IndexStringMode::Match(collection.embedding_model.as_deref())
            }
        },
    )?;
    let vectors_normalized = reader.read_u8("vector segment normalized flag")?;
    let Some(distance_metric) = distance_metric_from_segment_id(distance_metric_id) else {
        return Err(index_suffix_diagnostic(
            "index_segment_header_mismatch",
            "ignored index checkpoint with invalid vector segment header",
            checkpoint_start_offset + INDEX_CHECKPOINT_HEADER_LEN as u64,
        )
        .into());
    };
    if schema_version != INDEX_SEGMENT_SCHEMA_VERSION
        || covered_commit_sequence_no != expected_commit_sequence_no
        || index_kind != VECTOR_INDEX_KIND_EXACT_F32
        || vectors_normalized > 1
    {
        return Err(index_suffix_diagnostic(
            "index_segment_header_mismatch",
            "ignored index checkpoint with invalid vector segment header",
            checkpoint_start_offset + INDEX_CHECKPOINT_HEADER_LEN as u64,
        )
        .into());
    }
    if let VectorDecodeMode::Visit { collection, .. } = &mode {
        let mut id = [0; 24];
        if collection.collection_id != catalog_id_text(&mut id, b"col_", collection_id)
            || collection.embedding_dim != dim
            || collection.distance_metric != distance_metric
            || collection.vectors_normalized != (vectors_normalized == 1)
        {
            return Err(reader
                .invalid("vector segment collection configuration")
                .into());
        }
    }
    let entry_len = dim
        .checked_mul(4)
        .and_then(|len| len.checked_add(16))
        .ok_or_else(|| reader.invalid("vector segment dimension"))?;
    let vector_count = reader.read_count("vector segment vector count", entry_len)?;
    let mut previous_chunk_id = None;
    let mut vectors = if matches!(&mode, VectorDecodeMode::Retain) {
        Vec::with_capacity(vector_count)
    } else {
        Vec::new()
    };
    let mut scratch_work = None;
    let mut scratch = Vec::new();
    for _ in 0..vector_count {
        let chunk_id = reader.read_u64("vector segment chunk id")?;
        let embedding_id = reader.read_u64("vector segment embedding id")?;
        if previous_chunk_id >= Some(chunk_id) {
            return Err(index_suffix_diagnostic(
                "index_segment_non_canonical",
                "ignored index checkpoint with unsorted or duplicate vector segment entries",
                checkpoint_start_offset + INDEX_CHECKPOINT_HEADER_LEN as u64,
            )
            .into());
        }
        previous_chunk_id = Some(chunk_id);
        match &mut mode {
            VectorDecodeMode::Retain => {
                let vector = reader.read_vector(dim)?;
                let mut id = [0; 24];
                vectors.push(VectorIndexEntry {
                    chunk_id: catalog_id_text(&mut id, b"chk_", chunk_id).to_owned(),
                    embedding_id: catalog_id_text(&mut id, b"emb_", embedding_id).to_owned(),
                    vector,
                });
            }
            VectorDecodeMode::Summary => reader.read_vector_values(dim, None)?,
            VectorDecodeMode::Visit {
                input,
                error,
                visit,
                ..
            } => {
                if error.is_none() && scratch_work.is_none() {
                    match crate::json::ReservedVec::<f32>::with_capacity(
                        dim,
                        input.context(),
                        "operation_scratch",
                    ) {
                        Ok(values) => {
                            let (values, work) = values.into_parts();
                            scratch_work = Some(work);
                            scratch = values;
                        }
                        Err(failure) => **error = Some(failure),
                    }
                }
                scratch.clear();
                // A provisional query failure must not hide malformed values or
                // a later I/O/checksum failure in this advisory index frame.
                if error.is_some() {
                    reader.read_vector_values(dim, None)?;
                } else {
                    reader.read_vector_values(dim, Some(&mut scratch))?;
                    **error = visit(chunk_id, embedding_id, &scratch).err();
                }
            }
        }
    }
    reader.finish_before_checksum("vector segment")?;
    Ok(ParsedVectorIndexSegment {
        collection_number: collection_id,
        vector_count,
        payload: matches!(mode, VectorDecodeMode::Retain).then(|| VectorIndexSegmentPayload {
            collection_id: catalog_id_text(&mut [0; 24], b"col_", collection_id).to_owned(),
            covered_commit_sequence_no,
            dim,
            distance_metric,
            embedding_model,
            vectors_normalized: vectors_normalized == 1,
            vectors,
        }),
    })
}

#[cfg(test)]
fn validate_segment_checksum(
    payload: &[u8],
    checkpoint_start_offset: u64,
) -> std::result::Result<(), IndexSuffixDiagnostic> {
    if payload.len() < 4 {
        return Err(index_suffix_diagnostic(
            "index_segment_checksum_mismatch",
            "ignored index checkpoint with truncated index segment payload",
            checkpoint_start_offset + INDEX_CHECKPOINT_HEADER_LEN as u64,
        ));
    }
    let expected = read_u32(&payload[payload.len() - 4..]);
    if checksum32(&payload[..payload.len() - 4]) != expected {
        return Err(index_suffix_diagnostic(
            "index_segment_checksum_mismatch",
            "ignored index checkpoint with index segment checksum mismatch",
            checkpoint_start_offset + INDEX_CHECKPOINT_HEADER_LEN as u64,
        ));
    }
    Ok(())
}

#[derive(Debug)]
enum IndexReadError {
    Invalid(IndexSuffixDiagnostic),
    Io(std::io::Error),
    Resource(Error),
}

impl From<IndexSuffixDiagnostic> for IndexReadError {
    fn from(error: IndexSuffixDiagnostic) -> Self {
        Self::Invalid(error)
    }
}

impl From<std::io::Error> for IndexReadError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

enum IndexStringMode<'a> {
    Retain,
    Validate,
    Match(Option<&'a str>),
}

struct IndexPayloadReader<R> {
    source: R,
    remaining: usize,
    checkpoint_start_offset: u64,
}

impl<R: Read> IndexPayloadReader<R> {
    fn new(source: R, remaining: usize, checkpoint_start_offset: u64) -> Self {
        Self {
            source,
            remaining,
            checkpoint_start_offset,
        }
    }

    fn read_into(
        &mut self,
        output: &mut [u8],
        field: &'static str,
    ) -> std::result::Result<(), IndexReadError> {
        if output.len() > self.remaining {
            return Err(self.invalid(field).into());
        }
        for part in output.chunks_mut(64 * 1024) {
            self.source.read_exact(part)?;
        }
        self.remaining -= output.len();
        Ok(())
    }

    fn fixed<const N: usize>(
        &mut self,
        field: &'static str,
    ) -> std::result::Result<[u8; N], IndexReadError> {
        let mut bytes = [0; N];
        self.read_into(&mut bytes, field)?;
        Ok(bytes)
    }

    fn read_u8(&mut self, field: &'static str) -> std::result::Result<u8, IndexReadError> {
        Ok(self.fixed::<1>(field)?[0])
    }

    fn read_u16(&mut self, field: &'static str) -> std::result::Result<u16, IndexReadError> {
        Ok(u16::from_le_bytes(self.fixed(field)?))
    }

    fn read_u32(&mut self, field: &'static str) -> std::result::Result<u32, IndexReadError> {
        Ok(u32::from_le_bytes(self.fixed(field)?))
    }

    fn read_u64(&mut self, field: &'static str) -> std::result::Result<u64, IndexReadError> {
        Ok(u64::from_le_bytes(self.fixed(field)?))
    }

    fn read_count(
        &mut self,
        field: &'static str,
        min_entry_len: usize,
    ) -> std::result::Result<usize, IndexReadError> {
        let count = usize::try_from(self.read_u64(field)?).map_err(|_| self.invalid(field))?;
        if count > self.remaining / min_entry_len {
            return Err(self.invalid(field).into());
        }
        Ok(count)
    }

    fn read_f32(&mut self, field: &'static str) -> std::result::Result<f32, IndexReadError> {
        Ok(f32::from_le_bytes(self.fixed(field)?))
    }

    fn read_vector(&mut self, dim: usize) -> std::result::Result<Vec<f32>, IndexReadError> {
        let mut vector = Vec::new();
        self.read_vector_values(dim, Some(&mut vector))?;
        Ok(vector)
    }

    fn read_vector_values(
        &mut self,
        dim: usize,
        mut values: Option<&mut Vec<f32>>,
    ) -> std::result::Result<(), IndexReadError> {
        if dim
            .checked_mul(4)
            .filter(|bytes| *bytes <= self.remaining)
            .is_none()
        {
            return Err(self.invalid("vector segment value").into());
        }
        if let Some(vector) = &mut values {
            vector
                .try_reserve_exact(dim)
                .map_err(std::io::Error::other)?;
        }
        let mut bytes = [0; 4096];
        let mut remaining = dim;
        while remaining > 0 {
            let count = remaining.min(bytes.len() / 4);
            let bytes = &mut bytes[..count * 4];
            self.read_into(bytes, "vector segment value")?;
            for value in bytes.chunks_exact(4) {
                let value = f32::from_le_bytes(value.try_into().unwrap());
                if !value.is_finite() {
                    return Err(index_suffix_diagnostic(
                        "index_segment_non_canonical",
                        "ignored index checkpoint with non-finite vector segment value",
                        self.checkpoint_start_offset + INDEX_CHECKPOINT_HEADER_LEN as u64,
                    )
                    .into());
                }
                if let Some(vector) = &mut values {
                    vector.push(value);
                }
            }
            remaining -= count;
        }
        Ok(())
    }

    fn read_string(&mut self, field: &'static str) -> std::result::Result<String, IndexReadError> {
        let len = self.read_u32(field)? as usize;
        if len > self.remaining {
            return Err(self.invalid(field).into());
        }
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(len)
            .map_err(std::io::Error::other)?;
        bytes.resize(len, 0);
        self.read_into(&mut bytes, field)?;
        String::from_utf8(bytes).map_err(|_| self.invalid(field).into())
    }

    fn read_optional_string(
        &mut self,
        field: &'static str,
        mode: IndexStringMode<'_>,
    ) -> std::result::Result<Option<String>, IndexReadError> {
        match (self.read_u8(field)?, mode) {
            (0, IndexStringMode::Match(Some(_))) | (1, IndexStringMode::Match(None)) => {
                Err(self.invalid(field).into())
            }
            (0, _) => Ok(None),
            (1, IndexStringMode::Retain) => self.read_string(field).map(Some),
            (1, mode) => {
                let expected = match mode {
                    IndexStringMode::Match(expected) => expected,
                    _ => None,
                };
                self.check_string(field, expected)?;
                Ok(None)
            }
            _ => Err(self.invalid(field).into()),
        }
    }

    fn check_string(
        &mut self,
        field: &'static str,
        expected: Option<&str>,
    ) -> std::result::Result<(), IndexReadError> {
        let len = self.read_u32(field)? as usize;
        if len > self.remaining || expected.is_some_and(|value| value.len() != len) {
            return Err(self.invalid(field).into());
        }
        let mut bytes = [0; 4096];
        let mut remaining = len;
        let mut carried = 0;
        let mut position = 0;
        while remaining > 0 {
            let count = remaining.min(bytes.len() - carried);
            self.read_into(&mut bytes[carried..carried + count], field)?;
            remaining -= count;
            let filled = carried + count;
            let valid = match std::str::from_utf8(&bytes[..filled]) {
                Ok(_) => filled,
                Err(error) if error.error_len().is_none() && remaining > 0 => error.valid_up_to(),
                Err(_) => return Err(self.invalid(field).into()),
            };
            if expected.is_some_and(|value| {
                value.as_bytes().get(position..position + valid) != Some(&bytes[..valid])
            }) {
                return Err(self.invalid(field).into());
            }
            position += valid;
            // Keep only an incomplete UTF-8 code point across chunk boundaries.
            bytes.copy_within(valid..filled, 0);
            carried = filled - valid;
        }
        Ok(())
    }

    fn finish_before_checksum(
        &self,
        field: &'static str,
    ) -> std::result::Result<(), IndexReadError> {
        if self.remaining == 0 {
            Ok(())
        } else {
            Err(self.invalid(field).into())
        }
    }

    fn invalid(&self, field: &'static str) -> IndexSuffixDiagnostic {
        index_suffix_diagnostic(
            "index_segment_non_canonical",
            format!("ignored index checkpoint with invalid {field} payload"),
            self.checkpoint_start_offset + INDEX_CHECKPOINT_HEADER_LEN as u64,
        )
    }
}

fn parse_index_commit_metadata_payload(
    payload: &[u8],
    expected_batch_id: u64,
    expected_commit_sequence_no: u64,
    extent_table: &[u8],
    records: &BTreeMap<u64, ParsedIndexRecord>,
    checkpoint_start_offset: u64,
) -> std::result::Result<ParsedIndexCommitMetadata, IndexSuffixDiagnostic> {
    if payload.len() != INDEX_COMMIT_METADATA_LEN {
        return Err(index_suffix_diagnostic(
            "index_commit_metadata_invalid",
            "ignored index checkpoint with invalid IndexCommitMetadata payload length",
            checkpoint_start_offset
                + INDEX_CHECKPOINT_HEADER_LEN as u64
                + INDEX_RECORD_HEADER_LEN as u64,
        ));
    }
    let advisory_version = read_u16(&payload[0..2]);
    let batch_id = read_u64(&payload[2..10]);
    let commit_sequence_no = read_u64(&payload[10..18]);
    let index_record_count = read_u64(&payload[18..26]);
    let index_checksum = read_u32(&payload[26..30]);
    let index_extent_count = read_u64(&payload[30..38]);
    let index_extent_table_offset = read_u64(&payload[38..46]);
    let index_extent_table_length = read_u64(&payload[46..54]);
    let index_extent_table_checksum = read_u32(&payload[54..58]);
    if advisory_version != 1
        || batch_id != expected_batch_id
        || commit_sequence_no != expected_commit_sequence_no
        || index_record_count != index_extent_count
    {
        return Err(index_suffix_diagnostic(
            "index_commit_metadata_invalid",
            "ignored index checkpoint with invalid IndexCommitMetadata coverage or extent reference",
            checkpoint_start_offset
                + INDEX_CHECKPOINT_HEADER_LEN as u64
                + INDEX_RECORD_HEADER_LEN as u64,
        ));
    }
    let extent_start = usize::try_from(index_extent_table_offset).map_err(|_| {
        index_suffix_diagnostic(
            "index_commit_metadata_invalid",
            "ignored index checkpoint with IndexCommitMetadata extent offset that does not fit usize",
            checkpoint_start_offset + INDEX_CHECKPOINT_HEADER_LEN as u64,
        )
    })?;
    let extent_len = usize::try_from(index_extent_table_length).map_err(|_| {
        index_suffix_diagnostic(
            "index_commit_metadata_invalid",
            "ignored index checkpoint with IndexCommitMetadata extent length that does not fit usize",
            checkpoint_start_offset + INDEX_CHECKPOINT_HEADER_LEN as u64,
        )
    })?;
    let index_extent_count = usize::try_from(index_extent_count).map_err(|_| {
        index_suffix_diagnostic(
            "index_commit_metadata_invalid",
            "index count exceeds address space",
            checkpoint_start_offset,
        )
    })?;
    let expected_extent_len = index_extent_count
        .checked_mul(INDEX_EXTENT_ENTRY_LEN)
        .ok_or_else(|| {
            index_suffix_diagnostic(
                "index_commit_metadata_invalid",
                "ignored index checkpoint with overflowing IndexCommitMetadata extent count",
                checkpoint_start_offset + INDEX_CHECKPOINT_HEADER_LEN as u64,
            )
        })?;
    let Some(extent_end) = extent_start.checked_add(extent_len) else {
        return Err(index_suffix_diagnostic(
            "index_commit_metadata_invalid",
            "ignored index checkpoint with overflowing IndexCommitMetadata extent range",
            checkpoint_start_offset + INDEX_CHECKPOINT_HEADER_LEN as u64,
        ));
    };
    if extent_len != expected_extent_len || extent_end > extent_table.len() {
        return Err(index_suffix_diagnostic(
            "index_commit_metadata_invalid",
            "ignored index checkpoint with IndexCommitMetadata extent subrange outside table",
            checkpoint_start_offset + INDEX_CHECKPOINT_HEADER_LEN as u64,
        ));
    }
    let referenced_extent_bytes = &extent_table[extent_start..extent_end];
    if checksum32(referenced_extent_bytes) != index_extent_table_checksum {
        return Err(index_suffix_diagnostic(
            "index_commit_metadata_invalid",
            "ignored index checkpoint with IndexCommitMetadata extent subrange checksum mismatch",
            checkpoint_start_offset + INDEX_CHECKPOINT_HEADER_LEN as u64,
        ));
    }
    let mut previous_id = None;
    for entry_bytes in referenced_extent_bytes.chunks_exact(INDEX_EXTENT_ENTRY_LEN) {
        let extent = parse_extent_entry(entry_bytes);
        if extent.record_type == INDEX_COMMIT_METADATA
            || previous_id >= Some(extent.index_record_id)
        {
            return Err(index_suffix_diagnostic(
                "index_commit_metadata_invalid",
                "ignored index checkpoint with non-canonical IndexCommitMetadata extent reference",
                checkpoint_start_offset + INDEX_CHECKPOINT_HEADER_LEN as u64,
            ));
        }
        if !records.contains_key(&extent.index_record_id) {
            return Err(index_suffix_diagnostic(
                "index_commit_metadata_invalid",
                "ignored index checkpoint with IndexCommitMetadata reference to missing record",
                checkpoint_start_offset + INDEX_CHECKPOINT_HEADER_LEN as u64,
            ));
        }
        previous_id = Some(extent.index_record_id);
    }
    Ok(ParsedIndexCommitMetadata {
        index_record_count: index_extent_count,
        index_checksum,
        extent_range: extent_start..extent_end,
    })
}

#[cfg(test)]
fn index_commit_checksum(
    batch_id: u64,
    commit_sequence_no: u64,
    checkpoint_generation: u64,
    referenced_extent_and_record_bytes: &[u8],
) -> u32 {
    let mut checksum = index_commit_hasher(batch_id, commit_sequence_no, checkpoint_generation);
    checksum.update(referenced_extent_and_record_bytes);
    checksum.finish()
}

fn index_commit_hasher(
    batch_id: u64,
    commit_sequence_no: u64,
    checkpoint_generation: u64,
) -> Crc32c {
    let mut checksum = Crc32c::default();
    checksum.update(&1u16.to_le_bytes());
    checksum.update(&batch_id.to_le_bytes());
    checksum.update(&commit_sequence_no.to_le_bytes());
    checksum.update(&checkpoint_generation.to_le_bytes());
    checksum
}

fn index_suffix_diagnostic(
    code: &'static str,
    message: impl Into<String>,
    offset: u64,
) -> IndexSuffixDiagnostic {
    IndexSuffixDiagnostic {
        code,
        message: message.into(),
        offset,
        extent_id: None,
    }
}

fn index_extent_diagnostic(
    code: &'static str,
    message: impl Into<String>,
    offset: u64,
    index_record_id: u64,
) -> IndexSuffixDiagnostic {
    IndexSuffixDiagnostic {
        code,
        message: message.into(),
        offset,
        extent_id: Some(format!("index_record:{index_record_id}")),
    }
}

fn record_from_json(record_type: u16, value: &JsonValue) -> Result<RecordPayload> {
    let obj = value.as_object().ok_or_else(|| {
        Error::new(
            ErrorCode::FileFormatError,
            "record payload must be a JSON object",
        )
    })?;
    match record_type {
        COLLECTION_CREATE => Ok(RecordPayload::CollectionCreate(CollectionInfo::from_json(
            value,
        )?)),
        COLLECTION_DROP => Ok(RecordPayload::CollectionDrop {
            collection_id: required_string(obj, "collection_id")?,
            name: String::new(),
            dropped_at: required_string(obj, "dropped_at")?,
            reason: None,
        }),
        DOCUMENT_PUT => Ok(RecordPayload::DocumentPut(Document::from_json(value)?)),
        CHUNK_PUT => Ok(RecordPayload::ChunkPut(Chunk::from_json(value)?)),
        EMBEDDING_PUT => Ok(RecordPayload::EmbeddingPut(Embedding::from_json(value)?)),
        TOMBSTONE => Ok(RecordPayload::Tombstone(Tombstone::from_json(value)?)),
        COMMIT => Ok(RecordPayload::Commit(CommitPayload {
            batch_id: required_usize(obj, "batch_id")? as u64,
            first_sequence_no: required_usize(obj, "first_sequence_no")? as u64,
            last_sequence_no: required_usize(obj, "last_sequence_no")? as u64,
            record_count: required_usize(obj, "record_count")?,
            batch_checksum: required_u64(obj, "batch_checksum")?,
            committed_at: required_string(obj, "committed_at")?,
        })),
        COMPACT_MARKER => Ok(RecordPayload::CompactMarker(CompactMarker {
            compact_generation: 0,
            source_last_commit_sequence_no: 0,
            compacted_at: required_string(obj, "compacted_at")?,
            records_rewritten: required_usize(obj, "records_rewritten")?,
            id_high_watermarks: optional_u64_map(obj, "id_high_watermarks")?,
            compacted_main_checksum: 0,
        })),
        other => Err(Error::new(
            ErrorCode::FileFormatError,
            format!("unknown record type {other}"),
        )),
    }
}

fn record_type_id(record: &RecordPayload) -> u16 {
    match record {
        RecordPayload::CollectionCreate(_) => COLLECTION_CREATE,
        RecordPayload::CollectionDrop { .. } => COLLECTION_DROP,
        RecordPayload::DocumentPut(_) => DOCUMENT_PUT,
        RecordPayload::ChunkPut(_) => CHUNK_PUT,
        RecordPayload::EmbeddingPut(_) => EMBEDDING_PUT,
        RecordPayload::Tombstone(_) => TOMBSTONE,
        RecordPayload::Commit(_) => COMMIT,
        RecordPayload::CompactMarker(_) => COMPACT_MARKER,
    }
}

fn record_type_name(record_type: u16) -> &'static str {
    match record_type {
        COLLECTION_CREATE => "CollectionCreate",
        COLLECTION_DROP => "CollectionDrop",
        DOCUMENT_PUT => "DocumentPut",
        CHUNK_PUT => "ChunkPut",
        EMBEDDING_PUT => "EmbeddingPut",
        TOMBSTONE => "Tombstone",
        COMMIT => "Commit",
        COMPACT_MARKER => "CompactMarker",
        _ => "Unknown",
    }
}

fn record_type_from_name(record_type: &str) -> Result<u16> {
    match record_type {
        "CollectionCreate" => Ok(COLLECTION_CREATE),
        "CollectionDrop" => Ok(COLLECTION_DROP),
        "DocumentPut" => Ok(DOCUMENT_PUT),
        "ChunkPut" => Ok(CHUNK_PUT),
        "EmbeddingPut" => Ok(EMBEDDING_PUT),
        "Tombstone" => Ok(TOMBSTONE),
        "Commit" => Ok(COMMIT),
        "CompactMarker" => Ok(COMPACT_MARKER),
        other => Err(Error::new(
            ErrorCode::FileFormatError,
            format!("unknown record type {other}"),
        )),
    }
}

#[allow(clippy::too_many_arguments)]
fn issue(
    severity: &str,
    code: &str,
    message: &str,
    stream: &str,
    visibility_impact: &str,
    offset: Option<u64>,
    record_type: Option<&str>,
    batch_id: Option<u64>,
) -> CheckIssue {
    CheckIssue {
        severity: severity.into(),
        code: code.into(),
        message: message.into(),
        stream: stream.into(),
        visibility_impact: visibility_impact.into(),
        offset,
        extent_id: None,
        record_type: record_type.map(ToOwned::to_owned),
        batch_id,
        repair_available: false,
        suggested_command: None,
    }
}

fn read_u16(bytes: &[u8]) -> u16 {
    u16::from_le_bytes([bytes[0], bytes[1]])
}

fn read_u32(bytes: &[u8]) -> u32 {
    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

fn read_u64(bytes: &[u8]) -> u64 {
    u64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
    ])
}

fn required_u64(obj: &std::collections::BTreeMap<String, JsonValue>, key: &str) -> Result<u64> {
    match obj.get(key) {
        Some(JsonValue::String(value)) => value.parse::<u64>().map_err(|_| {
            Error::new(
                ErrorCode::FileFormatError,
                format!("field {key} must be a u64 string"),
            )
        }),
        Some(JsonValue::Number(value)) if *value >= 0.0 && value.fract() == 0.0 => {
            Ok(*value as u64)
        }
        _ => Err(Error::new(
            ErrorCode::FileFormatError,
            format!("missing u64 field {key}"),
        )),
    }
}

fn optional_u64_map(
    obj: &std::collections::BTreeMap<String, JsonValue>,
    key: &str,
) -> Result<BTreeMap<String, u64>> {
    let Some(value) = obj.get(key) else {
        return Ok(BTreeMap::new());
    };
    let map = value.as_object().ok_or_else(|| {
        Error::new(
            ErrorCode::FileFormatError,
            format!("field {key} must be an object"),
        )
    })?;
    let mut parsed = BTreeMap::new();
    for (entry_key, entry_value) in map {
        let value = entry_value.as_f64().ok_or_else(|| {
            Error::new(
                ErrorCode::FileFormatError,
                format!("{key}.{entry_key} must be a non-negative integer"),
            )
        })?;
        if !value.is_finite() || value < 0.0 || value.fract() != 0.0 || value > u64::MAX as f64 {
            return Err(Error::new(
                ErrorCode::FileFormatError,
                format!("{key}.{entry_key} must be a non-negative integer"),
            ));
        }
        parsed.insert(entry_key.clone(), value as u64);
    }
    Ok(parsed)
}

impl CheckIssue {
    pub fn to_json(&self) -> JsonValue {
        self.build_json(&JsonBuilder::host())
            .expect("host DTO construction is infallible")
    }

    pub(crate) fn build_json(&self, builder: &JsonBuilder<'_>) -> Result<JsonValue> {
        builder.object([
            ("severity", builder.string(&self.severity)?),
            ("code", builder.string(&self.code)?),
            ("message", builder.string(&self.message)?),
            ("stream", builder.string(&self.stream)?),
            (
                "visibility_impact",
                builder.string(&self.visibility_impact)?,
            ),
            (
                "offset",
                self.offset
                    .map(|value| JsonValue::Number(value as f64))
                    .unwrap_or(JsonValue::Null),
            ),
            (
                "extent_id",
                builder.optional_str(self.extent_id.as_deref())?,
            ),
            (
                "record_type",
                builder.optional_str(self.record_type.as_deref())?,
            ),
            (
                "batch_id",
                self.batch_id
                    .map(|value| JsonValue::Number(value as f64))
                    .unwrap_or(JsonValue::Null),
            ),
            ("repair_available", JsonValue::Bool(self.repair_available)),
            ("repairable", JsonValue::Bool(self.repair_available)),
            (
                "suggested_command",
                builder.optional_str(self.suggested_command.as_deref())?,
            ),
        ])
    }
}

impl IndexSuffixReport {
    pub fn total_size_bytes(&self) -> u64 {
        (INDEX_CHECKPOINT_HEADER_LEN as u64)
            .saturating_add(self.payload_len)
            .saturating_add(
                (self.index_extent_count as u64).saturating_mul(INDEX_EXTENT_ENTRY_LEN as u64),
            )
            .saturating_add(INDEX_CHECKPOINT_TRAILER_LEN as u64)
    }

    pub fn to_json(&self) -> JsonValue {
        self.build_json(&JsonBuilder::host())
            .expect("host DTO construction is infallible")
    }

    pub(crate) fn build_json(&self, builder: &JsonBuilder<'_>) -> Result<JsonValue> {
        builder.object([
            ("status", builder.string(&self.status)?),
            ("offset", JsonValue::Number(self.offset as f64)),
            (
                "checkpoint_generation",
                JsonValue::Number(self.checkpoint_generation as f64),
            ),
            (
                "covered_critical_end_offset",
                JsonValue::Number(self.covered_critical_end_offset as f64),
            ),
            (
                "covered_last_batch_id",
                JsonValue::Number(self.covered_last_batch_id as f64),
            ),
            (
                "covered_last_commit_sequence_no",
                JsonValue::Number(self.covered_last_commit_sequence_no as f64),
            ),
            (
                "coverage_hash",
                JsonValue::Number(self.coverage_hash as f64),
            ),
            (
                "collection_count",
                JsonValue::Number(self.collection_count as f64),
            ),
            (
                "live_document_count",
                JsonValue::Number(self.live_document_count as f64),
            ),
            (
                "live_chunk_count",
                JsonValue::Number(self.live_chunk_count as f64),
            ),
            (
                "live_embedding_count",
                JsonValue::Number(self.live_embedding_count as f64),
            ),
            (
                "text_segment_count",
                JsonValue::Number(self.text_segment_count as f64),
            ),
            (
                "vector_segment_count",
                JsonValue::Number(self.vector_segment_count as f64),
            ),
            (
                "index_record_count",
                JsonValue::Number(self.index_record_count as f64),
            ),
            (
                "index_extent_count",
                JsonValue::Number(self.index_extent_count as f64),
            ),
            ("payload_len", JsonValue::Number(self.payload_len as f64)),
            (
                "total_size_bytes",
                JsonValue::Number(self.total_size_bytes() as f64),
            ),
            ("build_mode", builder.string(&self.build_mode)?),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::{CheckOptions, Database};
    use crate::model::{DistanceMetric, VectorIndexConfig};
    use std::io::{Seek, SeekFrom, Write};

    #[test]
    fn record_reference_clones_borrow_canonical_kind_names_across_100_generations() {
        for generation in 0..100 {
            for kind in [
                COLLECTION_CREATE,
                COLLECTION_DROP,
                DOCUMENT_PUT,
                CHUNK_PUT,
                EMBEDDING_PUT,
                TOMBSTONE,
                COMMIT,
                COMPACT_MARKER,
            ] {
                let name = record_type_name(kind);
                let original = RecordRef {
                    offset: generation * 100,
                    record_type: name,
                    batch_id: generation + 1,
                    sequence_no: generation * 10 + 1,
                    payload: PayloadRef {
                        offset: generation * 100 + 64,
                        len: 36,
                        checksum: generation,
                    },
                };
                let copy = original.clone();
                assert_eq!(copy, original);
                assert!(std::ptr::eq(copy.record_type, original.record_type));
                assert_eq!(copy.record_type, name);
                assert_eq!(record_type_from_name(copy.record_type).unwrap(), kind);
                assert_eq!(copy.payload.checksum, generation);
            }
        }
    }

    fn append_batch_to_file(
        file: &mut File,
        next_batch_id: &mut u64,
        next_sequence_no: &mut u64,
        records: Vec<RecordPayload>,
        durability: Durability,
    ) -> Result<AppendSummary> {
        let count = records.len();
        write_batch_to_file(
            file,
            next_batch_id,
            next_sequence_no,
            records.into_iter().map(Ok),
            count,
            durability,
            true,
        )
    }

    pub(super) fn crash_boundary(stage: &str) -> Result<()> {
        if std::env::var("COFFERDB_TEST_CRASH_STAGE").ok().as_deref() == Some(stage) {
            let marker = std::env::var_os("COFFERDB_TEST_CRASH_MARKER").unwrap();
            fs::write(marker, stage)?;
            loop {
                std::thread::park();
            }
        }
        Ok(())
    }

    pub(super) fn write_at_crash_boundary(
        file: &mut File,
        bytes: &[u8],
        stage: &str,
    ) -> Result<()> {
        let split = bytes.len() / 2;
        file.write_all(&bytes[..split])?;
        crash_boundary(stage)?;
        file.write_all(&bytes[split..])?;
        Ok(())
    }

    #[test]
    #[ignore = "subprocess worker invoked only by process_kill_preserves_committed_prefix"]
    fn process_crash_worker() {
        let path = std::env::var_os("COFFERDB_TEST_CRASH_DATABASE").unwrap();
        let durability = match std::env::var("COFFERDB_TEST_CRASH_DURABILITY")
            .unwrap()
            .as_str()
        {
            "fast" => Durability::Fast,
            _ => Durability::Commit,
        };
        let (mut storage, scan) = Storage::open(path).unwrap();
        match std::env::var("COFFERDB_TEST_CRASH_OPERATION")
            .unwrap()
            .as_str()
        {
            operation @ ("compact" | "compact-batched") => {
                let records = scan
                    .records
                    .iter()
                    .map(|record| record.payload.clone())
                    .collect::<Vec<_>>();
                if operation == "compact" {
                    storage.rewrite_with_records(&records, durability).unwrap();
                } else {
                    let plan = RewritePlan {
                        records: records.len(),
                        batch_record_limit: 1,
                        batch_byte_target: 1,
                        temporary_disk: Default::default(),
                    };
                    rewrite_database(
                        &mut storage,
                        records.into_iter().map(Ok),
                        plan,
                        durability,
                        |_| Ok(()),
                    )
                    .unwrap();
                }
            }
            "index" => {
                let checkpoint = IndexCheckpointPayload {
                    metadata: IndexCheckpointMetadata {
                        covered_last_batch_id: scan.last_valid_commit,
                        covered_last_commit_sequence_no: scan.last_valid_commit_sequence_no,
                        coverage_hash: 42,
                        collection_count: 2,
                        live_document_count: 0,
                        live_chunk_count: 0,
                        live_embedding_count: 0,
                        text_segment_count: 0,
                        vector_segment_count: 0,
                        build_mode: "metadata_record_only".into(),
                    },
                    text_segments: vec![],
                    vector_segments: vec![],
                };
                storage
                    .write_index_checkpoint(&checkpoint, durability)
                    .unwrap();
            }
            _ => {
                let records = (3..=52)
                    .map(|id| {
                        collection_record(
                            &format!("col_{id}"),
                            &format!("{id}-{}", "x".repeat(64 * 1024)),
                        )
                    })
                    .collect::<Vec<_>>();
                storage.append_batch(&records, durability).unwrap();
            }
        }
        panic!("requested crash boundary was not reached");
    }

    #[test]
    fn process_kill_preserves_committed_prefix() {
        use std::process::{Child, Command, Stdio};
        use std::time::{Duration, Instant};

        struct KillOnDrop(Child);
        impl Drop for KillOnDrop {
            fn drop(&mut self) {
                let _ = self.0.kill();
                let _ = self.0.wait();
            }
        }
        let cases = [
            ("append", "critical_header_partial", false),
            ("append", "critical_payload_partial", false),
            ("append", "critical_record_written", false),
            ("append", "batch_before_commit", false),
            ("append", "commit_header_partial", false),
            ("append", "commit_payload_partial", false),
            ("append", "commit_written", true),
            ("append", "commit_flushed", true),
            ("append", "commit_synced", true),
            ("index", "index_suffix_partial", false),
            ("index", "index_suffix_flushed", false),
            ("compact", "compact_before_sync", false),
            ("compact", "compact_after_sync", false),
            ("compact", "compact_before_rename", false),
            ("compact", "compact_after_rename", false),
            ("compact-batched", "compact_between_batches", false),
            ("compact-batched", "compact_before_rename", false),
            ("compact-batched", "compact_after_rename", false),
        ];
        for durability in ["fast", "commit"] {
            for (operation, stage, new_batch_visible) in cases {
                if durability == "fast" && stage == "commit_synced" {
                    continue;
                }
                let dir = std::env::temp_dir().join(format!(
                    "cofferdb-kill-{stage}-{durability}-{}",
                    std::process::id()
                ));
                let _ = fs::remove_dir_all(&dir);
                fs::create_dir_all(&dir).unwrap();
                let path = dir.join("data.coffer");
                let marker = dir.join("ready");
                let mut storage = Storage::create(&path).unwrap();
                let first = storage
                    .append_batch(&[collection_record("col_1", "first")], Durability::Commit)
                    .unwrap();
                storage
                    .append_batch(&[collection_record("col_2", "second")], Durability::Commit)
                    .unwrap();
                let mut pinned_reader = storage.payload_reader().unwrap();
                drop(storage);
                let mut child = KillOnDrop(
                    Command::new(std::env::current_exe().unwrap())
                        .args([
                            "--ignored",
                            "--exact",
                            "storage::tests::process_crash_worker",
                        ])
                        .env("COFFERDB_TEST_CRASH_DATABASE", &path)
                        .env("COFFERDB_TEST_CRASH_STAGE", stage)
                        .env("COFFERDB_TEST_CRASH_MARKER", &marker)
                        .env("COFFERDB_TEST_CRASH_DURABILITY", durability)
                        .env("COFFERDB_TEST_CRASH_OPERATION", operation)
                        .stdout(Stdio::null())
                        .spawn()
                        .unwrap(),
                );
                let deadline = Instant::now() + Duration::from_secs(15);
                while !marker.exists() {
                    assert!(
                        child.0.try_wait().unwrap().is_none(),
                        "worker exited before {stage}"
                    );
                    assert!(Instant::now() < deadline, "worker did not reach {stage}");
                    std::thread::sleep(Duration::from_millis(5));
                }
                child.0.kill().unwrap();
                assert!(!child.0.wait().unwrap().success());
                let before_open = fs::read(&path).unwrap();
                let (mut reopened, scan) = Storage::open(&path).unwrap();
                assert_eq!(
                    scan.records.len(),
                    if new_batch_visible { 52 } else { 2 },
                    "{stage}/{durability}"
                );
                assert_eq!(
                    fs::read(&path).unwrap(),
                    before_open,
                    "open must not repair implicitly"
                );
                assert!(
                    matches!(pinned_reader.read_payload(&first.records[0].record_ref).unwrap(), RecordPayload::CollectionCreate(collection) if collection.name == "first")
                );
                // The recovered writer discards only the interrupted tail before the next batch.
                reopened
                    .append_batch(
                        &[collection_record("col_53", "after restart")],
                        Durability::Commit,
                    )
                    .unwrap();
                drop(reopened);
                assert_eq!(
                    scan_file(&path).unwrap().records.len(),
                    if new_batch_visible { 53 } else { 3 }
                );
                drop(pinned_reader);
                fs::remove_dir_all(dir).unwrap();
            }
        }
    }

    fn temp_path(name: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "cofferdb-storage-{name}-{}.coffer",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path);
        path
    }

    fn collection_record(id: &str, name: &str) -> RecordPayload {
        RecordPayload::CollectionCreate(CollectionInfo {
            tokenizer: Default::default(),
            collection_id: id.into(),
            name: name.into(),
            embedding_dim: 3,
            distance_metric: DistanceMetric::Cosine,
            embedding_model: None,
            vectors_normalized: false,
            vector_index: VectorIndexConfig::Exact,
            created_at: "1970-01-01T00:00:00Z".into(),
        })
    }

    fn append_fake_index_suffix(path: &Path) {
        let mut file = OpenOptions::new().append(true).open(path).unwrap();
        file.write_all(b"COFFERIDX").unwrap();
        file.write_all(&[0xa5; 48]).unwrap();
    }

    #[test]
    fn commit_workspace_refusal_preserves_suffix_counters_and_writer_retry() {
        use crate::json::{JsonOperationContext, JsonProcessingPolicyV1};
        let path = temp_path("commit-workspace-preflight");
        let mut storage = Storage::create(&path).unwrap();
        storage
            .append_owned_batch(vec![collection_record("col_1", "first")], Durability::Fast)
            .unwrap();
        append_fake_index_suffix(&path);
        let original = fs::read(&path).unwrap();
        let counters = (
            storage.next_batch_id,
            storage.next_sequence_no,
            storage.critical_end_offset,
        );
        let limit = JsonProcessingPolicyV1::WORK_BUDGET_BYTES;
        for seed in 0..100 {
            for prepared_input in [false, true] {
                let context = JsonOperationContext::standalone();
                let mut held = context.workspace_reservation();
                let available = seed % time::MAX_TIMESTAMP_BYTES;
                let leave_room = |held: &mut crate::json::WorkspaceReservation| {
                    let grant = held.grow_to(0, limit, "operation_scratch").unwrap();
                    held.retain(grant - available);
                };
                let record = collection_record("col_2", "second");
                let error = if prepared_input {
                    let mut builder = PreparedRecordsBuilder::new(
                        path.parent().unwrap(),
                        1_048_576,
                        context.clone(),
                    )
                    .unwrap();
                    builder.push(record).unwrap();
                    let mut prepared = builder.finish().unwrap();
                    leave_room(&mut held);
                    let error = prepared.take_batch(1).err().unwrap();
                    assert_eq!(prepared.records().unwrap().count(), 1);
                    error
                } else {
                    leave_room(&mut held);
                    OwnedRecordBatch::single(record, &context).err().unwrap()
                };
                assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                for (key, value) in [
                    ("resource", "memory".to_string()),
                    ("operation", "batch_prepare".into()),
                    ("limit_bytes", limit.to_string()),
                    (
                        "requested_bytes",
                        (limit - available + time::MAX_TIMESTAMP_BYTES).to_string(),
                    ),
                ] {
                    assert!(error.detail().contains(&(key.to_owned(), value)));
                }
                // The prepared fixture has dropped its two file owners here.
                leave_room(&mut held);
                let writer_error = storage
                    .append_record_batch(
                        std::iter::once(WriteRecord::encoded(collection_record("col_2", "second"))),
                        Durability::Fast,
                        &context,
                        None,
                        None,
                    )
                    .unwrap_err();
                assert_eq!(writer_error.code(), error.code());
                assert_eq!(writer_error.detail(), error.detail());
                assert!(!storage.write_failed);
                assert_eq!(fs::read(&path).unwrap(), original);
                assert_eq!(
                    (
                        storage.next_batch_id,
                        storage.next_sequence_no,
                        storage.critical_end_offset
                    ),
                    counters
                );
                held.grow_to(limit, limit, "operation_scratch").unwrap();
            }
        }
        let context = JsonOperationContext::standalone();
        let mut builder =
            PreparedRecordsBuilder::new(path.parent().unwrap(), 1_048_576, context.clone())
                .unwrap();
        builder.push(collection_record("col_2", "second")).unwrap();
        let mut prepared = builder.finish().unwrap();
        let batch = prepared.take_batch(1).unwrap();
        let timestamp_pointer = batch.commit_timestamp.as_ref().as_ptr();
        let mut held = context.workspace_reservation();
        held.grow_to(0, limit, "operation_scratch").unwrap();
        let (append, work) = storage
            .append_prepared_batch(batch, Durability::Commit)
            .unwrap();
        assert_eq!(append.commit.committed_at.as_ptr(), timestamp_pointer);
        assert_eq!(append.commit.batch_id, counters.0);
        assert_eq!(append.commit.first_sequence_no, counters.1);
        assert_eq!(
            append.commit.committed_at.capacity(),
            time::MAX_TIMESTAMP_BYTES
        );
        assert_eq!(
            append._commit_work.as_ref().unwrap().reserved_bytes(),
            time::MAX_TIMESTAMP_BYTES
        );
        drop((append, work));
        drop(held);
        let management = 2 * crate::temporary::test_file_management_bytes(path.parent().unwrap());
        let mut free = context.workspace_reservation();
        free.grow_to(limit - management, limit - management, "operation_scratch")
            .unwrap();
        assert!(free
            .grow_to(
                limit - management + 1,
                limit - management + 1,
                "operation_scratch"
            )
            .is_err());
        drop(free);
        drop(prepared);
        context
            .workspace_reservation()
            .grow_to(limit, limit, "operation_scratch")
            .unwrap();
        drop(storage);
        let (storage, report) = Storage::open(&path).unwrap();
        assert_eq!(report.commits, 2);
        assert!(report.findings.is_empty());
        drop(storage);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn binary_catalog_recovery_shares_tight_workspace_with_pending_cache() {
        use crate::json::{JsonOperationContext, JsonProcessingPolicyV1};

        let path = temp_path("catalog-pending-workspace");
        let mut storage = Storage::create(&path).unwrap();
        let mut records = vec![collection_record("col_1", "notes")];
        records.extend((1..=192).map(|id| {
            RecordPayload::DocumentPut(Document {
                document_id: format!("doc_{id}"),
                collection_id: "col_1".into(),
                external_id: format!("doc-{id}"),
                title: None,
                source_uri: None,
                metadata: object_from_pairs([(
                    "values",
                    JsonValue::Array((0..16).map(|_| JsonValue::String("value".into())).collect()),
                )]),
                created_at: "1970-01-01T00:00:00Z".into(),
                updated_at: "1970-01-01T00:00:00Z".into(),
            })
        }));
        storage.append_batch(&records, Durability::Fast).unwrap();
        drop(storage);
        let original = fs::read(&path).unwrap();
        let mut reference = crate::catalog::Catalog::default();
        for record in scan_file(&path).unwrap().records {
            reference.apply_owned_committed(record).unwrap();
        }
        let mut cached = 0;
        let mut replayed = 0;
        for seed in 0..100 {
            let context = JsonOperationContext::standalone();
            let available = 32 * 1024 + seed * 8191;
            let mut held = context.workspace_reservation();
            let occupied = JsonProcessingPolicyV1::WORK_BUDGET_BYTES - available;
            held.grow_to(occupied, occupied, "operation_scratch")
                .unwrap();
            let mut catalog = crate::catalog::Catalog::new_in(&context);
            let scan = scan_file_with_retained_records_in(&path, &context, |record, work| {
                if matches!(record.payload, RecordPayload::CollectionCreate(_)) {
                    if work.json_bytes() == 0 {
                        replayed += 1;
                    } else {
                        cached += 1;
                    }
                }
                catalog.apply_retained(record, work, &context)
            })
            .unwrap_or_else(|error| panic!("available={available}: {error}"));
            assert!(scan.findings.is_empty());
            assert_eq!(scan.commits, 1);
            assert_eq!(catalog.state_signature(), reference.state_signature());
            held.grow_to(
                JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
                JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
                "operation_scratch",
            )
            .unwrap();
        }
        assert!(cached > 0 && replayed > 0);
        assert_eq!(fs::read(&path).unwrap(), original);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn binary_scan_replays_pending_metadata_and_releases_failed_batches() {
        use crate::json::{JsonInputKind, JsonOperationContext, JsonProcessingPolicyV1};
        let path = temp_path("scan-pending-json");
        let mut storage = Storage::create(&path).unwrap();
        storage
            .append_batch(&[collection_record("col_1", "notes")], Durability::Fast)
            .unwrap();
        let documents = (1..=2)
            .map(|id| {
                RecordPayload::DocumentPut(Document {
                    document_id: format!("doc_{id}"),
                    collection_id: "col_1".into(),
                    external_id: format!("doc-{id}"),
                    title: None,
                    source_uri: None,
                    metadata: object_from_pairs([(
                        "payload",
                        JsonValue::String("x".repeat(128 * 1024)),
                    )]),
                    created_at: "1970-01-01T00:00:00Z".into(),
                    updated_at: "1970-01-01T00:00:00Z".into(),
                })
            })
            .collect::<Vec<_>>();
        let append = storage.append_batch(&documents, Durability::Fast).unwrap();
        let first_offset = append.records[0].record_ref.offset;
        let original = fs::read(&path).unwrap();
        let context = JsonOperationContext::standalone();
        let scan = |visit: &mut dyn FnMut(CommittedRecord) -> Result<()>| {
            let mut file = File::open(&path).unwrap();
            assert_eq!(read_header(&mut file).unwrap(), FileCodec::Binary);
            scan_binary_file(
                &mut file,
                IndexReadOptions {
                    budget_bytes: u64::MAX,
                    mode: IndexReadMode::Summary,
                },
                &context,
                &mut |record, _| visit(record),
            )
        };
        let capacity = context.available_workspace_bytes();
        let mut visits = 0;
        // After {"payload":" the first ASCII run is 8180 bytes; the next
        // 8192-byte run needs 16372 bytes, exceeding a simple capacity doubling.
        let error = crate::test_alloc::fail_byte_allocation_once((8192 - 12) + 8192, || {
            scan(&mut |_| {
                visits += 1;
                Ok(())
            })
        })
        .unwrap_err();
        assert_eq!(error.code(), ErrorCode::IoError);
        assert!(!crate::json::is_limit_error(&error));
        assert_eq!(visits, 1);
        assert_eq!(context.available_workspace_bytes(), capacity);
        assert_eq!(fs::read(&path).unwrap(), original);
        for seed in 0..100 {
            let available = 384 * 1024 + seed;
            let mut held = context.reservation();
            held.grow(
                JsonProcessingPolicyV1::WORK_BUDGET_BYTES - available,
                JsonInputKind::Filter,
            )
            .unwrap();
            let mut visits = 0;
            let report = scan(&mut |_| {
                visits += 1;
                Ok(())
            })
            .unwrap();
            assert_eq!(
                visits, 3,
                "a fitting record must not require its whole batch in memory"
            );
            assert!(report.findings.is_empty());
            held.grow(256 * 1024, JsonInputKind::Filter).unwrap();
            let mut visits = 0;
            let error = scan(&mut |_| {
                visits += 1;
                Ok(())
            })
            .unwrap_err();
            assert_eq!(
                visits, 1,
                "only the earlier committed collection may be visited"
            );
            assert_eq!(error.code(), ErrorCode::RecoveryError);
            assert!(
                error.detail().iter().any(
                    |(key, value)| key == "record_offset" && value == &first_offset.to_string()
                ),
                "{error}"
            );
            assert!(error
                .detail()
                .iter()
                .any(|(key, value)| key == "json_limit_reason" && value == "work_budget_exceeded"));
            held.grow(available - 256 * 1024, JsonInputKind::Filter)
                .unwrap();
        }
        let mut visits = 0;
        let report = scan(&mut |record| {
            if matches!(record.payload, RecordPayload::DocumentPut(_)) {
                assert!(context
                    .reservation()
                    .grow(
                        JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
                        JsonInputKind::Filter
                    )
                    .is_err());
            }
            visits += 1;
            Ok(())
        })
        .unwrap();
        assert_eq!(visits, 3);
        assert!(report.findings.is_empty());
        let error =
            scan(&mut |_| Err(Error::new(ErrorCode::IoError, "visitor failure"))).unwrap_err();
        assert_eq!(error.code(), ErrorCode::IoError);
        context
            .reservation()
            .grow(
                JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
                JsonInputKind::Filter,
            )
            .unwrap();
        assert_eq!(fs::read(&path).unwrap(), original);
        let mut held = context.reservation();
        held.grow(
            JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
            JsonInputKind::Filter,
        )
        .unwrap();
        let mut corrupt = original.clone();
        corrupt[HEADER_LEN + RECORD_HEADER_LEN] ^= 1;
        fs::write(&path, &corrupt).unwrap();
        let report = scan(&mut |_| panic!("corrupt batch must stay invisible")).unwrap();
        assert!(report
            .findings
            .iter()
            .any(|issue| issue.code == "checksum_mismatch"
                && issue.visibility_impact == "blocks_main"));
        assert_eq!(report.last_valid_commit, 0);
        drop(held);
        drop(storage);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn binary_replay_preserves_commit_visibility_under_small_shared_budgets() {
        use crate::json::{JsonInputKind, JsonOperationContext, JsonProcessingPolicyV1};
        let path = temp_path("scan-replay-visibility");
        let mut storage = Storage::create(&path).unwrap();
        let records: Vec<_> = (1..=20)
            .map(|id| RecordPayload::CollectionDrop {
                collection_id: format!("col_{id}"),
                name: format!("name-{id}"),
                dropped_at: "1970-01-01T00:00:00Z".into(),
                reason: Some("retired".into()),
            })
            .collect();
        let first = storage.append_batch(&records, Durability::Fast).unwrap();
        let first_end = storage.file.metadata().unwrap().len();
        storage.append_batch(&records, Durability::Fast).unwrap();
        let original = fs::read(&path).unwrap();
        let last = first.records.last().unwrap();
        let commit_offset = last.record_ref.payload.offset + last.record_ref.payload.len;
        let mut bad_payload = original.clone();
        bad_payload[first.records[3].record_ref.payload.offset as usize] ^= 1;
        let mut bad_commit = original.clone();
        bad_commit[commit_offset as usize + RECORD_HEADER_LEN] ^= 1;
        let context = JsonOperationContext::standalone();
        for seed in 0..100usize {
            let available = 300 + seed;
            let mut held = context.workspace_reservation();
            let occupied = JsonProcessingPolicyV1::WORK_BUDGET_BYTES - available;
            held.grow_to(occupied, occupied, "operation_scratch")
                .unwrap();
            for (bytes, expected, issue_code) in [
                (original.as_slice(), 40, None),
                (bad_payload.as_slice(), 0, Some("checksum_mismatch")),
                (bad_commit.as_slice(), 0, Some("checksum_mismatch")),
                (
                    &original[..commit_offset as usize],
                    0,
                    Some("uncommitted_batch"),
                ),
                (&original[..original.len() - 1], 20, Some("corrupt_tail")),
            ] {
                fs::write(&path, bytes).unwrap();
                let mut file = File::open(&path).unwrap();
                read_header(&mut file).unwrap();
                let mut seen = Vec::new();
                let report = scan_binary_file(
                    &mut file,
                    IndexReadOptions {
                        budget_bytes: 0,
                        mode: IndexReadMode::Summary,
                    },
                    &context,
                    &mut |record, _| {
                        let RecordPayload::CollectionDrop {
                            collection_id,
                            reason,
                            ..
                        } = record.payload
                        else {
                            panic!("unexpected recovery payload")
                        };
                        assert_eq!(reason.as_deref(), Some("retired"));
                        seen.push((collection_id, record.record_ref.sequence_no));
                        Ok(())
                    },
                )
                .unwrap();
                assert_eq!(seen.len(), expected);
                for (index, (id, sequence)) in seen.iter().enumerate() {
                    assert_eq!(*id, format!("col_{:016}", index % 20 + 1));
                    assert_eq!(*sequence, index as u64 + 1 + u64::from(index >= 20));
                }
                match issue_code {
                    Some(code) => assert!(report.findings.iter().any(|issue| issue.code == code)),
                    None => assert!(report.findings.is_empty()),
                }
                if expected == 20 {
                    assert_eq!(report.critical_end_offset, first_end);
                }
                context
                    .reservation()
                    .grow(available, JsonInputKind::Filter)
                    .unwrap();
            }
            fs::write(&path, &original).unwrap();
            let mut file = File::open(&path).unwrap();
            read_header(&mut file).unwrap();
            let mut visits = 0;
            let error = scan_binary_file(
                &mut file,
                IndexReadOptions {
                    budget_bytes: 0,
                    mode: IndexReadMode::Summary,
                },
                &context,
                &mut |_, _| {
                    visits += 1;
                    Err(Error::new(ErrorCode::IoError, "replay visitor failed"))
                },
            )
            .unwrap_err();
            assert_eq!(visits, 1);
            assert_eq!(error.code(), ErrorCode::IoError);
            context
                .reservation()
                .grow(available, JsonInputKind::Filter)
                .unwrap();
        }
        drop(storage);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn binary_scan_non_json_object_refusal_is_resource_error_after_checksum() {
        use crate::json::{JsonOperationContext, JsonProcessingPolicyV1};
        let path = temp_path("scan-string-budget");
        let mut storage = Storage::create(&path).unwrap();
        let record = RecordPayload::CollectionDrop {
            collection_id: "col_1".into(),
            name: "n".repeat(4096),
            dropped_at: "1970-01-01T00:00:00Z".into(),
            reason: None,
        };
        storage.append_batch(&[record], Durability::Fast).unwrap();
        let original = fs::read(&path).unwrap();
        let mut corrupt = original.clone();
        corrupt[HEADER_LEN + RECORD_HEADER_LEN] ^= 1;
        let context = JsonOperationContext::standalone();
        for seed in 0..100 {
            let mut held = context.workspace_reservation();
            let occupied = JsonProcessingPolicyV1::WORK_BUDGET_BYTES - 1000 - seed;
            held.grow_to(occupied, occupied, "operation_scratch")
                .unwrap();
            for (bytes, damaged) in [(&original, false), (&corrupt, true)] {
                fs::write(&path, bytes).unwrap();
                let mut file = File::open(&path).unwrap();
                read_header(&mut file).unwrap();
                let result = scan_binary_file(
                    &mut file,
                    IndexReadOptions {
                        budget_bytes: 0,
                        mode: IndexReadMode::Summary,
                    },
                    &context,
                    &mut |_, _| panic!("rejected batch must not be visible"),
                );
                if damaged {
                    let report = result.unwrap();
                    assert!(report
                        .findings
                        .iter()
                        .any(|issue| issue.code == "checksum_mismatch"));
                } else {
                    let error = result.unwrap_err();
                    assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                    for (key, value) in [
                        ("resource", "memory".into()),
                        ("operation", "operation_scratch".into()),
                        (
                            "limit_bytes",
                            JsonProcessingPolicyV1::WORK_BUDGET_BYTES.to_string(),
                        ),
                        ("requested_bytes", (occupied + 20 + 4096).to_string()),
                    ] {
                        assert!(error.detail().contains(&(key.into(), value)), "{error}");
                    }
                }
                context
                    .workspace_reservation()
                    .grow_to(1000 + seed, 1000 + seed, "operation_scratch")
                    .unwrap();
            }
        }
        drop(storage);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn scan_input_reservations_share_only_optional_headroom() {
        use crate::json::{
            JsonInputKind, JsonOperationContext, JsonProcessingPolicyV1, WorkLedger,
        };
        use std::sync::Arc;
        let limit = JsonProcessingPolicyV1::WORK_BUDGET_BYTES;
        for seed in 0..100usize {
            let ledger = Arc::new(WorkLedger::default());
            let context = JsonOperationContext::new(ledger.clone());
            let other = JsonOperationContext::new(ledger);
            let available = [0, 1, 4095, 4096, 65535, 65536, 70000][seed % 7];
            let mut held = other.workspace_reservation();
            held.grow_to(limit - available, limit - available, "operation_scratch")
                .unwrap();
            for record_type in 1..=8 {
                for payload_len in [0, 1, 4096, 65536, 65537 + seed as u64, u64::MAX] {
                    let preferred = payload_len.min(65536) as usize;
                    let json = matches!(record_type, COLLECTION_CREATE | DOCUMENT_PUT | CHUNK_PUT);
                    let result = PayloadInputReservation::new(
                        &context,
                        record_type,
                        payload_len,
                        JsonInputKind::StoredJson,
                    );
                    let mut work = result.unwrap();
                    let capacity = if json && preferred > available {
                        0
                    } else {
                        preferred.min(available)
                    };
                    assert_eq!(work.capacity(), capacity);
                    let mut probe = other.workspace_reservation();
                    let remaining = available - capacity;
                    probe
                        .grow_to(remaining, remaining, "operation_scratch")
                        .unwrap();
                    let error = probe
                        .grow_to(remaining + 1, remaining + 1, "operation_scratch")
                        .unwrap_err();
                    assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                    assert!(error.detail().iter().any(|(key, value)| {
                        key == "requested_bytes" && value == &(limit + 1).to_string()
                    }));
                    drop(probe);
                    assert_eq!(work.disable_optional_buffering(), capacity != 0);
                    assert_eq!(work.capacity(), 0);
                    assert!(!work.disable_optional_buffering());
                    other
                        .workspace_reservation()
                        .grow_to(available, available, "operation_scratch")
                        .unwrap();
                    drop(work);
                    // Disabling this buffer must not release another operation's lease.
                    context
                        .reservation()
                        .grow(available, JsonInputKind::Filter)
                        .unwrap();
                }
            }
            drop(held);
            context
                .reservation()
                .grow(limit, JsonInputKind::Filter)
                .unwrap();
        }
    }

    #[test]
    fn binary_scan_uses_smaller_non_json_buffers_and_preserves_failure_boundaries() {
        use crate::json::{JsonOperationContext, JsonProcessingPolicyV1, WorkLedger};
        use std::sync::Arc;
        let path = temp_path("scan-non-json-buffers");
        let mut storage = Storage::create(&path).unwrap();
        let append = storage
            .append_batch(
                &[
                    RecordPayload::EmbeddingPut(Embedding {
                        embedding_id: "emb_1".into(),
                        chunk_id: "chk_1".into(),
                        collection_id: "col_1".into(),
                        chunk_external_id: "external".into(),
                        model_name: "model".into(),
                        dim: 16385,
                        vector: vec![0.5; 16385],
                        created_at: "1970-01-01T00:00:00Z".into(),
                    }),
                    RecordPayload::Tombstone(Tombstone {
                        target_type: "chunk".into(),
                        target_id: "chk_1".into(),
                        deleted_at: "1970-01-01T00:00:00Z".into(),
                        reason: Some("x".repeat(65537)),
                    }),
                    RecordPayload::CollectionDrop {
                        collection_id: "col_1".into(),
                        name: "notes".into(),
                        dropped_at: "1970-01-01T00:00:00Z".into(),
                        reason: None,
                    },
                ],
                Durability::Fast,
            )
            .unwrap();
        let original = fs::read(&path).unwrap();
        let ledger = Arc::new(WorkLedger::default());
        let context = JsonOperationContext::new(ledger.clone());
        let other = JsonOperationContext::new(ledger);
        let limit = JsonProcessingPolicyV1::WORK_BUDGET_BYTES;
        let scan = |visit: &mut dyn FnMut(CommittedRecord) -> Result<()>| {
            scan_file_with_records_in(&path, &context, visit)
        };
        for seed in 0..100usize {
            let available = [0, 1, 7, 4096, 65581, 65582, 70000 + seed][seed % 7];
            let mut held = other.workspace_reservation();
            held.grow_to(limit - available, limit - available, "operation_scratch")
                .unwrap();
            let mut visits = 0;
            let result = scan(&mut |record| {
                let mut probe = other.workspace_reservation();
                assert_eq!(
                    probe
                        .grow_to(available, available, "operation_scratch")
                        .unwrap_err()
                        .code(),
                    ErrorCode::ResourceExhausted
                );
                assert_eq!(record.record_ref.sequence_no, visits + 1);
                match record.payload {
                    RecordPayload::EmbeddingPut(e) => {
                        assert_eq!(e.dim, 16385);
                        assert!(e.vector.is_empty());
                    }
                    RecordPayload::Tombstone(t) => assert_eq!(t.reason.unwrap().len(), 65537),
                    RecordPayload::CollectionDrop { name, .. } => assert_eq!(name, "notes"),
                    _ => panic!("unexpected record"),
                }
                visits += 1;
                Ok(())
            });
            if available < 65582 {
                assert_eq!(result.unwrap_err().code(), ErrorCode::ResourceExhausted);
                assert_eq!(visits, 0);
            } else {
                let report = result.unwrap();
                assert_eq!(visits, 3);
                assert_eq!(report.commits, 1);
                assert!(report.findings.is_empty());
                let error = scan(&mut |_| Err(Error::new(ErrorCode::IoError, "visitor failed")))
                    .unwrap_err();
                assert_eq!(error.code(), ErrorCode::IoError);
            }
            held.grow_to(limit, limit, "operation_scratch").unwrap();
        }
        assert_eq!(fs::read(&path).unwrap(), original);
        let mut held = other.workspace_reservation();
        held.grow_to(limit, limit, "operation_scratch").unwrap();
        let first = &append.records[0].record_ref.payload;
        let mut corrupt = original.clone();
        corrupt[(first.offset + first.len - 1) as usize] ^= 1;
        fs::write(&path, corrupt).unwrap();
        let report = scan(&mut |_| panic!("corrupt batch cannot be published")).unwrap();
        assert!(report
            .findings
            .iter()
            .any(|issue| issue.code == "checksum_mismatch"));
        assert_eq!(report.commits, 0);
        fs::write(&path, &original[..(first.offset + first.len - 1) as usize]).unwrap();
        let report = scan(&mut |_| panic!("truncated batch cannot be published")).unwrap();
        assert!(report
            .findings
            .iter()
            .any(|issue| issue.code == "corrupt_tail"));
        assert_eq!(report.commits, 0);
        drop(held);
        drop(storage);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn streamed_committed_records_match_collected_scans_across_one_hundred_tail_cases() {
        let path = temp_path("streamed-committed-records");
        let mut storage = Storage::create(&path).unwrap();
        let mut boundaries = vec![HEADER_LEN];
        let mut first_payloads = Vec::new();
        let mut counts = vec![0];
        let mut id = 0;
        for batch in 0..10 {
            let records = (0..batch % 3 + 1)
                .map(|_| {
                    id += 1;
                    collection_record(&format!("col_{id}"), &format!("collection-{id}"))
                })
                .collect::<Vec<_>>();
            let append = storage.append_batch(&records, Durability::Fast).unwrap();
            first_payloads.push(append.records[0].record_ref.payload.clone());
            boundaries.push(storage.critical_end_offset() as usize);
            counts.push(id);
        }
        drop(storage);
        let original = fs::read(&path).unwrap();
        for seed in 0..100 {
            let prefix = seed % 10;
            let mut bytes = original.clone();
            match (seed / 10) % 5 {
                0 => bytes.truncate(boundaries[prefix]),
                1 => bytes.truncate(
                    (first_payloads[prefix].offset + first_payloads[prefix].len / 2) as usize,
                ),
                2 => bytes.truncate(boundaries[prefix + 1] - 1),
                3 => bytes[first_payloads[prefix].offset as usize] ^= 1,
                _ => {
                    bytes.truncate(boundaries[prefix]);
                    bytes.extend_from_slice(b"COFFERIDX incomplete index tail");
                }
            }
            fs::write(&path, &bytes).unwrap();
            let collected = scan_file(&path).unwrap();
            assert_eq!(collected.records.len(), counts[prefix], "seed {seed}");
            let mut records = Vec::new();
            let mut streamed = scan_file_with_records(&path, |record| {
                assert!(record.record_ref.batch_id <= prefix as u64);
                records.push(record);
                Ok(())
            })
            .unwrap();
            assert!(streamed.records.is_empty());
            streamed.records = records;
            assert_eq!(
                format!("{streamed:?}"),
                format!("{collected:?}"),
                "seed {seed}"
            );
            assert_eq!(fs::read(&path).unwrap(), bytes);
        }
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn streamed_record_consumer_failures_stop_scanning_and_release_writer_locks() {
        let path = temp_path("streamed-record-consumer-failure");
        let mut storage = Storage::create(&path).unwrap();
        for id in 1..=4 {
            storage
                .append_batch(
                    &[collection_record(&format!("col_{id}"), "notes")],
                    Durability::Fast,
                )
                .unwrap();
        }
        drop(storage);
        let original = fs::read(&path).unwrap();
        for fail_at in 1..=4 {
            for panic in [false, true] {
                let mut visited = 0;
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    Storage::open_with_records(&path, false, u64::MAX, |_| {
                        visited += 1;
                        if visited == fail_at {
                            assert!(!panic, "injected consumer panic");
                            return Err(Error::new(
                                ErrorCode::IoError,
                                "injected consumer failure",
                            ));
                        }
                        Ok(())
                    })
                }));
                if panic {
                    assert!(result.is_err());
                } else {
                    assert_eq!(result.unwrap().err().unwrap().code(), ErrorCode::IoError);
                }
                assert_eq!(visited, fail_at);
                assert_eq!(fs::read(&path).unwrap(), original);
                let (_, scan) = Storage::open(&path).unwrap();
                assert_eq!(scan.records.len(), 4);
            }
        }
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn legacy_streamed_records_preserve_validated_payloads_and_diagnostics() {
        let path = temp_path("legacy-streamed-records");
        let hex = include_str!("../tests/fixtures/legacy-2d9eb64.hex")
            .split_whitespace()
            .collect::<String>();
        let original = hex
            .as_bytes()
            .chunks_exact(2)
            .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
            .collect::<Vec<_>>();
        for seed in 0..100 {
            let len = if seed == 0 {
                original.len()
            } else {
                LEGACY_HEADER_LEN + (original.len() - LEGACY_HEADER_LEN) * seed / 100
            };
            fs::write(&path, &original[..len]).unwrap();
            let expected = scan_file(&path).unwrap();
            let mut records = Vec::new();
            let (_, mut actual) = Storage::open_with_records(&path, true, u64::MAX, |record| {
                records.push(record);
                Ok(())
            })
            .unwrap();
            assert!(actual.records.is_empty());
            actual.records = records;
            assert_eq!(format!("{actual:?}"), format!("{expected:?}"));
            assert_eq!(fs::read(&path).unwrap(), original[..len]);
        }
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn unknown_header_flags_and_reserved_fields_are_rejected() {
        let path = temp_path("header-flags");
        drop(Storage::create(&path).unwrap());
        let original = std::fs::read(&path).unwrap();
        for (offset, code) in [
            (14, ErrorCode::UnsupportedFeatureFlag),
            (13, ErrorCode::UnsupportedVersion),
        ] {
            let mut bytes = original.clone();
            bytes[offset] = 2;
            let checksum = checksum32(&bytes[..40]);
            bytes[40..44].copy_from_slice(&checksum.to_le_bytes());
            std::fs::write(&path, bytes).unwrap();
            assert_eq!(scan_file(&path).unwrap_err().code(), code);
        }
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn create_never_overwrites_existing_files() {
        let path = temp_path("create-no-overwrite");
        fs::write(&path, b"existing private data").unwrap();
        assert!(Storage::create(&path).is_err());
        assert_eq!(fs::read(&path).unwrap(), b"existing private data");
        fs::remove_file(&path).unwrap();
        let storage = Storage::create(&path).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                fs::metadata(&path).unwrap().permissions().mode() & 0o777,
                0o600
            );
        }
        drop(storage);
        let before = fs::read(&path).unwrap();
        assert!(Storage::create(&path).is_err());
        assert_eq!(fs::read(&path).unwrap(), before);
        let _ = fs::remove_file(path);
    }

    #[cfg(unix)]
    #[test]
    fn compact_data_lock_is_acquired_before_publish_and_retained_after_publish_failure() {
        for durability in [Durability::Fast, Durability::Commit] {
            let path = temp_path("compact-inode-lock-handoff");
            let mut storage = Storage::create(&path).unwrap();
            let records = [collection_record("col_1", "notes")];
            storage.append_batch(&records, durability).unwrap();
            let before = fs::read(&path).unwrap();
            let mut competing_lock = None;
            let error = storage
                .rewrite_with_records_with_hook(&records, durability, |temporary| {
                    let file = OpenOptions::new().read(true).write(true).open(temporary)?;
                    let file = ManagedFile::open_in(
                        &crate::json::JsonOperationContext::standalone(),
                        || Ok(file),
                    )?;
                    competing_lock = Some(DataFileLock::acquire(&file, temporary)?);
                    Ok(())
                })
                .unwrap_err();
            assert_eq!(error.code(), ErrorCode::LockContention);
            assert_eq!(fs::read(&path).unwrap(), before);
            assert!(storage.require_writer().is_ok());
            drop(competing_lock);

            let old_alias = path.with_extension("retained-old-alias");
            fs::hard_link(&path, &old_alias).unwrap();
            let (old_reader, _) =
                Storage::open_with_records(&old_alias, true, u64::MAX, |_| Ok(())).unwrap();
            assert!(
                matches!(Storage::open(&old_alias), Err(error) if error.code() == ErrorCode::LockContention)
            );

            let alias = path.with_extension("published-alias");
            let error = storage
                .rewrite_with_iter_with_hook(
                    records.iter().cloned().map(Ok),
                    RewritePlan::single_batch(records.len()),
                    durability,
                    |stage, published| {
                        if stage == RewriteStage::AfterRename {
                            assert!(matches!(Storage::open(&old_alias), Err(error) if error.code() == ErrorCode::RecoveryError));
                            assert_eq!(fs::read(&old_alias)?, before);
                            fs::hard_link(published, &alias)?;
                            let error = match Storage::open(&alias) {
                                Ok(_) => panic!("published inode lost writer lock"),
                                Err(error) => error,
                            };
                            assert_eq!(error.code(), ErrorCode::LockContention);
                            return Err(Error::new(
                                ErrorCode::IoError,
                                "injected post-publication failure",
                            ));
                        }
                        Ok(())
                    },
                    |_| Ok(()),
                )
                .unwrap_err();
            assert_eq!(error.code(), ErrorCode::IoError);
            assert_eq!(
                storage.require_writer().unwrap_err().code(),
                ErrorCode::RecoveryError
            );
            assert!(
                matches!(Storage::open(&alias), Err(error) if error.code() == ErrorCode::LockContention)
            );
            drop(storage);
            assert!(
                matches!(Storage::open(&old_alias), Err(error) if error.code() == ErrorCode::RecoveryError)
            );
            drop(Storage::open(&alias).unwrap());
            drop(old_reader);
            drop(Storage::open(&old_alias).unwrap());
            fs::remove_file(&old_alias).unwrap();
            fs::remove_file(crate::lock::lock_path(&old_alias)).unwrap();
            fs::remove_file(&alias).unwrap();
            fs::remove_file(crate::lock::lock_path(&alias)).unwrap();
            fs::remove_file(path).unwrap();
        }
    }

    #[cfg(unix)]
    #[test]
    fn create_and_compact_do_not_follow_existing_temporary_symlinks() {
        use std::os::unix::fs::symlink;
        let dir = std::env::temp_dir().join(format!("cofferdb-safe-temp-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("data.coffer");
        let victim = dir.join("private.txt");
        fs::write(&victim, b"must remain untouched").unwrap();
        symlink(&victim, &path).unwrap();
        assert!(Storage::create(&path).is_err());
        fs::remove_file(&path).unwrap();
        let mut storage = Storage::create(&path).unwrap();
        let records = [collection_record("col_1", "notes")];
        storage.append_batch(&records, Durability::Commit).unwrap();
        let old_tmp = path.with_extension("coffer.tmp");
        symlink(&victim, &old_tmp).unwrap();
        storage
            .rewrite_with_records(&records, Durability::Commit)
            .unwrap();
        assert_eq!(fs::read(&victim).unwrap(), b"must remain untouched");
        assert!(fs::symlink_metadata(&old_tmp)
            .unwrap()
            .file_type()
            .is_symlink());
        let mut actual_temporary = None;
        assert!(storage
            .rewrite_with_records_with_hook(&records, Durability::Commit, |temporary| {
                actual_temporary = Some(temporary.to_path_buf());
                Err(Error::new(
                    ErrorCode::IoError,
                    "injected validation failure",
                ))
            })
            .is_err());
        assert!(!actual_temporary.unwrap().exists());
        assert!(fs::symlink_metadata(&old_tmp)
            .unwrap()
            .file_type()
            .is_symlink());
        drop(storage);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn overflowing_critical_payload_length_is_checked_before_allocation() {
        let path = temp_path("critical-overflow");
        let mut storage = Storage::create(&path).unwrap();
        storage
            .append_batch(&[collection_record("col_1", "notes")], Durability::Commit)
            .unwrap();
        drop(storage);
        let mut bytes = std::fs::read(&path).unwrap();
        bytes[HEADER_LEN + 34..HEADER_LEN + 42].copy_from_slice(&u64::MAX.to_le_bytes());
        let checksum = checksum32(&bytes[HEADER_LEN..HEADER_LEN + 46]);
        bytes[HEADER_LEN + 46..HEADER_LEN + 50].copy_from_slice(&checksum.to_le_bytes());
        std::fs::write(&path, bytes).unwrap();
        let scan = scan_file(&path).unwrap();
        assert_eq!(scan.commits, 0);
        assert!(scan
            .findings
            .iter()
            .any(|issue| issue.code == "file_format_error"));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn decreasing_compact_watermarks_reject_the_batch_and_later_commits() {
        for prefix in ["col_", "doc_", "chk_", "emb_"] {
            let path = temp_path(&format!("decreasing-marker-{prefix}"));
            let mut storage = Storage::create(&path).unwrap();
            let mut marker = CompactMarker {
                compact_generation: 1,
                source_last_commit_sequence_no: 0,
                compacted_at: "1970-01-01T00:00:00Z".into(),
                records_rewritten: 0,
                id_high_watermarks: ["col_", "doc_", "chk_", "emb_"]
                    .into_iter()
                    .map(|prefix| (prefix.into(), 10))
                    .collect(),
                compacted_main_checksum: 0,
            };
            let first = storage
                .append_batch(
                    &[RecordPayload::CompactMarker(marker.clone())],
                    Durability::Commit,
                )
                .unwrap();
            marker.source_last_commit_sequence_no = first.commit_sequence_no;
            marker.id_high_watermarks.insert(prefix.into(), 9);
            let first_ref = &first.records[0].record_ref;
            let first_end = (first_ref.payload.offset + first_ref.payload.len) as usize;
            marker.compacted_main_checksum =
                checksum32(&fs::read(&path).unwrap()[HEADER_LEN..first_end]);
            let payload = codec::encode(&RecordPayload::CompactMarker(marker)).unwrap();
            let mut batch_bytes = record_header(COMPACT_MARKER, 2, 3, &payload)
                .unwrap()
                .to_vec();
            batch_bytes.extend_from_slice(&payload);
            write_record(&mut storage.file, COMPACT_MARKER, 2, 3, &payload).unwrap();
            let commit = RecordPayload::Commit(CommitPayload {
                batch_id: 2,
                first_sequence_no: 3,
                last_sequence_no: 3,
                record_count: 1,
                batch_checksum: checksum32(&batch_bytes) as u64,
                committed_at: "1970-01-01T00:00:00Z".into(),
            });
            write_record(
                &mut storage.file,
                COMMIT,
                2,
                4,
                &codec::encode(&commit).unwrap(),
            )
            .unwrap();
            append_batch_to_file(
                &mut storage.file,
                &mut 3,
                &mut 5,
                vec![collection_record("col_11", "later")],
                Durability::Commit,
            )
            .unwrap();
            drop(storage);
            let before = fs::read(&path).unwrap();
            let (reopened, scan) = Storage::open(&path).unwrap();
            assert_eq!(scan.commits, 1);
            assert_eq!(scan.records.len(), 1);
            assert!(scan.findings.iter().any(|issue| {
                issue.code == "file_format_error"
                    && issue.record_type.as_deref() == Some("CompactMarker")
                    && issue
                        .message
                        .contains("lowers an allocation high watermark")
                    && issue.visibility_impact == "blocks_main"
            }));
            let catalog =
                crate::catalog::Catalog::from_committed_records(scan.records.clone(), 1).unwrap();
            assert_eq!(
                catalog.next_id(prefix, 0).unwrap(),
                format!("{prefix}{:016}", 11)
            );
            assert_eq!(fs::read(&path).unwrap(), before);
            drop(reopened);
            fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn compact_marker_cannot_lower_ids_seen_in_its_output_batch() {
        let path = temp_path("marker-below-output-id");
        let mut storage = Storage::create(&path).unwrap();
        let marker = CompactMarker {
            compact_generation: 1,
            source_last_commit_sequence_no: 0,
            compacted_at: "1970-01-01T00:00:00Z".into(),
            records_rewritten: 1,
            id_high_watermarks: BTreeMap::from([("col_".into(), 8)]),
            compacted_main_checksum: 0,
        };
        storage
            .append_batch(
                &[
                    collection_record("col_9", "output"),
                    RecordPayload::CompactMarker(marker),
                ],
                Durability::Commit,
            )
            .unwrap();
        drop(storage);
        let scan = scan_file(&path).unwrap();
        assert_eq!(scan.commits, 0);
        assert!(scan.records.is_empty());
        assert!(scan.findings.iter().any(|issue| issue
            .message
            .contains("lowers an allocation high watermark")));
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn compact_marker_source_sequence_must_precede_output() {
        let path = temp_path("marker-future-source");
        let mut storage = Storage::create(&path).unwrap();
        let marker = CompactMarker {
            compact_generation: 1,
            source_last_commit_sequence_no: 1,
            compacted_at: "1970-01-01T00:00:00Z".into(),
            records_rewritten: 0,
            id_high_watermarks: BTreeMap::new(),
            compacted_main_checksum: 0,
        };
        storage
            .append_batch(&[RecordPayload::CompactMarker(marker)], Durability::Commit)
            .unwrap();
        drop(storage);
        let scan = scan_file(&path).unwrap();
        assert_eq!(scan.commits, 0);
        assert!(scan
            .findings
            .iter()
            .any(|issue| issue.message.contains("compact source sequence")));
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn invalid_commit_blocks_later_valid_commits() {
        let path = temp_path("commit-prefix");
        let mut storage = Storage::create(&path).unwrap();
        storage
            .append_batch(&[collection_record("col_1", "first")], Durability::Commit)
            .unwrap();
        let payload = codec::encode(&collection_record("col_2", "bad")).unwrap();
        write_record(&mut storage.file, COLLECTION_CREATE, 2, 3, &payload).unwrap();
        let commit = RecordPayload::Commit(CommitPayload {
            batch_id: 2,
            first_sequence_no: 3,
            last_sequence_no: 3,
            record_count: 1,
            batch_checksum: 0,
            committed_at: time::now_rfc3339(),
        });
        write_record(
            &mut storage.file,
            COMMIT,
            2,
            4,
            &codec::encode(&commit).unwrap(),
        )
        .unwrap();
        append_batch_to_file(
            &mut storage.file,
            &mut 3,
            &mut 5,
            vec![collection_record("col_3", "later")],
            Durability::Commit,
        )
        .unwrap();
        drop(storage);
        let scan = scan_file(&path).unwrap();
        assert_eq!(scan.commits, 1);
        assert_eq!(scan.records.len(), 1);
        assert!(scan
            .findings
            .iter()
            .any(|issue| issue.visibility_impact == "blocks_main"));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn damaged_critical_records_after_commit_are_not_index_only() {
        let path = temp_path("critical-diagnostics");
        let mut storage = Storage::create(&path).unwrap();
        storage
            .append_batch(&[collection_record("col_1", "first")], Durability::Commit)
            .unwrap();
        drop(storage);
        let prefix = std::fs::read(&path).unwrap();
        for payload in [b"\xff".as_slice(), b"{".as_slice(), b"{}".as_slice()] {
            std::fs::write(&path, &prefix).unwrap();
            let mut file = OpenOptions::new().append(true).open(&path).unwrap();
            write_record(&mut file, COLLECTION_CREATE, 2, 3, payload).unwrap();
            drop(file);
            let before = std::fs::read(&path).unwrap();
            let scan = scan_file(&path).unwrap();
            assert_eq!(scan.commits, 1);
            assert!(scan.findings.iter().any(
                |issue| issue.stream == "critical" && issue.visibility_impact == "blocks_main"
            ));
            assert_eq!(std::fs::read(&path).unwrap(), before);
        }
        std::fs::write(&path, &prefix).unwrap();
        let (mut storage, scan) = Storage::open(&path).unwrap();
        let before = storage.file_size().unwrap();
        let mut exhausted_sequence = u64::MAX;
        assert!(append_batch_to_file(
            &mut storage.file,
            &mut 2,
            &mut exhausted_sequence,
            vec![collection_record("col_2", "overflow")],
            Durability::Commit
        )
        .is_err());
        assert_eq!(storage.file_size().unwrap(), before);
        let mut records = vec![scan.records[0].clone(), scan.records[0].clone()];
        records[0].record_ref.sequence_no = u64::MAX;
        records[1].record_ref.sequence_no = 0;
        let commit = CommitPayload {
            batch_id: 1,
            first_sequence_no: u64::MAX,
            last_sequence_no: 0,
            record_count: 2,
            batch_checksum: 0,
            committed_at: String::new(),
        };
        assert!(!committed_batch_sequence_matches(&records, &commit, 1));
        drop(storage);
        let mut file = OpenOptions::new().append(true).open(&path).unwrap();
        write_record(
            &mut file,
            COLLECTION_CREATE,
            u64::MAX,
            u64::MAX,
            &codec::encode(&collection_record("col_2", "uncommitted")).unwrap(),
        )
        .unwrap();
        drop(file);
        let (mut storage, scan) = Storage::open(&path).unwrap();
        assert_eq!(scan.commits, 1);
        let append = storage
            .append_batch(
                &[collection_record("col_2", "recovered")],
                Durability::Commit,
            )
            .unwrap();
        assert_eq!(append.commit.batch_id, 2);
        assert_eq!(append.commit.first_sequence_no, 3);
        drop(storage);
        assert_eq!(scan_file(&path).unwrap().commits, 2);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn file_format_constants_are_stable_for_v0_1() {
        assert_eq!(MAGIC, b"COFFERDB");
        assert_eq!(FORMAT_VERSION, 1);
        assert_eq!(HEADER_LEN, 58);
        assert_eq!(RECORD_HEADER_LEN, 50);
        assert_eq!(INDEX_CHECKPOINT_MAGIC, b"CFIDXCHK");
        assert_eq!(INDEX_COMPLETE_MARKER, b"CFOK");
        assert_eq!(INDEX_RECORD_MAGIC, b"CFIR");
        assert_eq!(INDEX_CHECKPOINT_VERSION, 1);
        assert_eq!(INDEX_CHECKPOINT_HEADER_LEN, 105);
        assert_eq!(INDEX_CHECKPOINT_TRAILER_LEN, 10);
        assert_eq!(INDEX_COVERAGE_FULL_VISIBLE_SNAPSHOT, 1);
        assert_eq!(INDEX_RECORD_HEADER_LEN, 50);
        assert_eq!(INDEX_EXTENT_ENTRY_LEN, 46);
        assert_eq!(INDEX_COMMIT_METADATA_LEN, 58);
        assert_eq!(INDEX_COMMIT_METADATA, 0x8001);
        assert_eq!(INDEX_TEXT_SEGMENT, 0x8002);
        assert_eq!(INDEX_VECTOR_SEGMENT, 0x8003);
        assert_eq!(TEXT_INDEX_KIND_BM25, 1);
        assert_eq!(VECTOR_INDEX_KIND_EXACT_F32, 2);
        assert_eq!(INDEX_SEGMENT_SCHEMA_VERSION, 1);
        assert_eq!(MAX_INDEX_CHECKPOINT_PAYLOAD_LEN, 512 * 1024 * 1024);
        assert_eq!(COLLECTION_CREATE, 1);
        assert_eq!(COLLECTION_DROP, 2);
        assert_eq!(DOCUMENT_PUT, 3);
        assert_eq!(CHUNK_PUT, 4);
        assert_eq!(EMBEDDING_PUT, 5);
        assert_eq!(TOMBSTONE, 6);
        assert_eq!(COMMIT, 7);
        assert_eq!(COMPACT_MARKER, 8);
    }

    #[test]
    fn binary_envelope_and_ordered_commit_match_independent_crc() {
        fn independent_crc(bytes: &[u8]) -> u32 {
            let mut crc = u32::MAX;
            for byte in bytes {
                crc ^= *byte as u32;
                for _ in 0..8 {
                    crc = if crc & 1 != 0 {
                        (crc >> 1) ^ 0x82f63b78
                    } else {
                        crc >> 1
                    };
                }
            }
            !crc
        }
        let path = temp_path("independent-envelope");
        let mut storage = Storage::create(&path).unwrap();
        let append = storage
            .append_batch(
                &[
                    collection_record("col_1", "one"),
                    collection_record("col_2", "two"),
                ],
                Durability::Commit,
            )
            .unwrap();
        let bytes = fs::read(&path).unwrap();
        assert_eq!(&bytes[..14], b"COFFERDB\x00\x00\x01\x00\x01\x01");
        assert_eq!(read_u64(&bytes[22..30]), 58);
        assert_eq!(read_u16(&bytes[38..40]), 44);
        assert_eq!(read_u32(&bytes[40..44]), independent_crc(&bytes[..40]));
        for record in &append.records {
            let offset = record.record_ref.offset as usize;
            let end =
                record.record_ref.payload.offset as usize + record.record_ref.payload.len as usize;
            assert_eq!(&bytes[offset..offset + 4], b"CFRL");
            assert_eq!(
                read_u64(&bytes[offset + 8..offset + 16]),
                (end - offset) as u64
            );
            assert_eq!(read_u16(&bytes[offset + 32..offset + 34]), 50);
            assert_eq!(
                read_u32(&bytes[offset + 46..offset + 50]),
                independent_crc(&bytes[offset..offset + 46])
            );
            assert_eq!(
                read_u32(&bytes[offset + 42..offset + 46]),
                independent_crc(&bytes[offset + 50..end])
            );
        }
        let end = append.records.last().unwrap().record_ref.payload.offset as usize
            + append.records.last().unwrap().record_ref.payload.len as usize;
        assert_eq!(
            append.commit.batch_checksum,
            independent_crc(&bytes[HEADER_LEN..end]) as u64
        );
        let mut swapped = bytes.clone();
        let first_end = append.records[1].record_ref.offset as usize;
        let mut reordered = bytes[first_end..end].to_vec();
        reordered.extend_from_slice(&bytes[HEADER_LEN..first_end]);
        swapped[HEADER_LEN..end].copy_from_slice(&reordered);
        drop(storage);
        fs::write(&path, swapped).unwrap();
        assert_eq!(scan_file(&path).unwrap().commits, 0);
        let _ = fs::remove_file(path);
    }

    mod fixed_header_tests {
        use super::*;

        fn crc(bytes: &[u8]) -> u32 {
            let mut state = u32::MAX;
            for byte in bytes {
                state ^= *byte as u32;
                for _ in 0..8 {
                    state = (state >> 1) ^ (0x82f6_3b78 & 0u32.wrapping_sub(state & 1));
                }
            }
            !state
        }

        #[test]
        fn file_header_matches_frozen_wire_without_heap() {
            for seed in 0..100u64 {
                for timestamp in [0, seed * 37 + 7, u64::MAX - seed] {
                    let mut expected = b"COFFERDB\x00\x00\x01\x00\x01\x01".to_vec();
                    expected.extend_from_slice(&0u64.to_le_bytes());
                    expected.extend_from_slice(&58u64.to_le_bytes());
                    expected.extend_from_slice(&timestamp.to_le_bytes());
                    expected.extend_from_slice(&44u16.to_le_bytes());
                    expected.extend_from_slice(&crc(&expected).to_le_bytes());
                    expected.extend_from_slice(&1u16.to_le_bytes());
                    expected.extend_from_slice(&0u64.to_le_bytes());
                    expected.extend_from_slice(&crc(&expected[44..54]).to_le_bytes());
                    let (actual, allocated) =
                        crate::test_alloc::allocation_bytes(|| file_header(timestamp));
                    assert_eq!(actual.as_slice(), expected);
                    assert_eq!(allocated, 0);
                    assert_eq!(read_u32(&actual[40..44]), crc(&actual[..40]));
                    assert_eq!(read_u32(&actual[54..58]), crc(&actual[44..54]));
                }
            }
        }

        fn record_oracle(kind: u16, batch: u64, sequence: u64, len: u64, digest: u32) -> Vec<u8> {
            let mut bytes = Vec::new();
            bytes.extend_from_slice(b"CFRL");
            bytes.extend_from_slice(&kind.to_le_bytes());
            bytes.extend_from_slice(&1u16.to_le_bytes());
            bytes.extend_from_slice(&(len + 50).to_le_bytes());
            bytes.extend_from_slice(&batch.to_le_bytes());
            bytes.extend_from_slice(&sequence.to_le_bytes());
            bytes.extend_from_slice(&50u16.to_le_bytes());
            bytes.extend_from_slice(&len.to_le_bytes());
            bytes.extend_from_slice(&digest.to_le_bytes());
            bytes.extend_from_slice(&crc(&bytes).to_le_bytes());
            bytes
        }

        #[test]
        fn critical_record_header_matches_frozen_wire_without_heap() {
            for (seed, sequence) in (0..100u64).flat_map(|seed| {
                [0, seed * 13 + 7, u64::MAX, 0x0123_4567_89ab_cdef ^ seed]
                    .into_iter()
                    .map(move |sequence| (seed, sequence))
            }) {
                for kind in 1..=8 {
                    let batch = u64::MAX - seed;
                    for len in [0, 1, seed as usize * 13 + 7] {
                        let payload = (0..len)
                            .map(|index| (index as u64 ^ seed) as u8)
                            .collect::<Vec<_>>();
                        let digest = crc(&payload);
                        let expected = record_oracle(kind, batch, sequence, len as u64, digest);
                        let ((from_payload, from_digest), allocated) =
                            crate::test_alloc::allocation_bytes(|| {
                                (
                                    record_header(kind, batch, sequence, &payload).unwrap(),
                                    record_header_from_digest(
                                        kind, batch, sequence, len as u64, digest,
                                    )
                                    .unwrap(),
                                )
                            });
                        assert_eq!(from_payload.as_slice(), expected);
                        assert_eq!(from_digest.as_slice(), expected);
                        assert_eq!(allocated, 0);
                    }
                    let length = u64::MAX - 50;
                    let expected = record_oracle(kind, batch, sequence, length, seed as u32);
                    let (header, allocated) = crate::test_alloc::allocation_bytes(|| {
                        record_header_from_digest(kind, batch, sequence, length, seed as u32)
                            .unwrap()
                    });
                    assert_eq!(header.as_slice(), expected);
                    assert_eq!(allocated, 0);
                    for length in [length + 1, u64::MAX] {
                        let error = record_header_from_digest(kind, batch, sequence, length, 0)
                            .unwrap_err();
                        assert_eq!(error.code(), ErrorCode::FileFormatError);
                        assert_eq!(error.message(), "record length overflow");
                    }
                }
            }
        }
    }

    #[test]
    fn every_byte_truncation_keeps_only_complete_commits_and_does_not_write() {
        let path = temp_path("all-truncations");
        let mut storage = Storage::create(&path).unwrap();
        storage
            .append_batch(&[collection_record("col_1", "first")], Durability::Commit)
            .unwrap();
        let first_end = storage.critical_end_offset() as usize;
        storage
            .append_batch(&[collection_record("col_2", "second")], Durability::Commit)
            .unwrap();
        let bytes = fs::read(&path).unwrap();
        drop(storage);
        assert!(bytes.len() - HEADER_LEN > 100);
        for end in HEADER_LEN..=bytes.len() {
            fs::write(&path, &bytes[..end]).unwrap();
            let scan = scan_file(&path).unwrap();
            let expected = if end == bytes.len() {
                2
            } else if end >= first_end {
                1
            } else {
                0
            };
            assert_eq!(scan.commits, expected, "cut at {end}");
            assert_eq!(scan.records.len(), expected, "cut at {end}");
            assert_eq!(fs::read(&path).unwrap(), bytes[..end]);
        }
        let _ = fs::remove_file(path);
    }

    #[test]
    fn advisory_hint_corruption_does_not_change_visible_data() {
        let path = temp_path("hint-crc");
        let mut storage = Storage::create(&path).unwrap();
        storage
            .append_batch(&[collection_record("col_1", "notes")], Durability::Commit)
            .unwrap();
        drop(storage);
        let original = fs::read(&path).unwrap();
        for offset in 44..HEADER_LEN {
            let mut bytes = original.clone();
            bytes[offset] ^= 0xff;
            fs::write(&path, bytes).unwrap();
            assert_eq!(scan_file(&path).unwrap().commits, 1);
        }
        let _ = fs::remove_file(path);
    }

    #[test]
    fn legacy_codec_is_read_only_until_explicit_compaction() {
        let path = temp_path("legacy-write-policy");
        let hex = include_str!("../tests/fixtures/legacy-2d9eb64.hex")
            .split_whitespace()
            .collect::<String>();
        let bytes = hex
            .as_bytes()
            .chunks_exact(2)
            .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
            .collect::<Vec<_>>();
        fs::write(&path, &bytes).unwrap();
        let (mut storage, scan) = Storage::open(&path).unwrap();
        assert_eq!(scan.codec, FileCodec::Legacy);
        assert_eq!(storage.codec, FileCodec::Legacy);
        assert_eq!(
            storage
                .append_batch(&[collection_record("col_2", "new")], Durability::Commit)
                .unwrap_err()
                .code(),
            ErrorCode::UnsupportedVersion
        );
        assert_eq!(fs::read(&path).unwrap(), bytes);
        drop(storage);
        let _ = fs::remove_file(path);
    }

    #[test]
    fn binary_metadata_depth_limit_is_not_reported_as_data_corruption() {
        let path = temp_path("binary-depth-limit");
        let mut storage = Storage::create(&path).unwrap();
        let raw = format!("{{\"v\":{}0{}}}", "[".repeat(128), "]".repeat(128));
        let mut document = Vec::new();
        document.extend_from_slice(&1u64.to_le_bytes());
        document.extend_from_slice(&1u64.to_le_bytes());
        document.extend_from_slice(&3u32.to_le_bytes());
        document.extend_from_slice(b"doc");
        document.extend_from_slice(&[0, 0]);
        document.extend_from_slice(&(raw.len() as u32).to_le_bytes());
        document.extend_from_slice(raw.as_bytes());
        document.extend_from_slice(&[0; 16]);
        let mut crc = Crc32c::default();
        let records = [
            (
                COLLECTION_CREATE,
                codec::encode(&collection_record("col_1", "notes")).unwrap(),
            ),
            (DOCUMENT_PUT, document),
        ];
        let mut document_offset = 0;
        for (n, (record_type, payload)) in records.iter().enumerate() {
            if *record_type == DOCUMENT_PUT {
                document_offset = storage.file.stream_position().unwrap();
            }
            let header = record_header(*record_type, 1, n as u64 + 1, payload).unwrap();
            crc.update(&header);
            crc.update(payload);
            write_record(&mut storage.file, *record_type, 1, n as u64 + 1, payload).unwrap();
        }
        let commit = RecordPayload::Commit(CommitPayload {
            batch_id: 1,
            first_sequence_no: 1,
            last_sequence_no: 2,
            record_count: 2,
            batch_checksum: crc.finish() as u64,
            committed_at: "1970-01-01T00:00:00Z".into(),
        });
        write_record(
            &mut storage.file,
            COMMIT,
            1,
            3,
            &codec::encode(&commit).unwrap(),
        )
        .unwrap();
        storage.file.flush().unwrap();
        drop(storage);
        let before = fs::read(&path).unwrap();
        for err in [
            scan_file(&path).unwrap_err(),
            Storage::open(&path).err().unwrap(),
        ] {
            assert_eq!(err.code(), ErrorCode::RecoveryError);
            assert!(crate::json::is_depth_error(&err));
            assert!(err
                .detail()
                .iter()
                .any(|(key, value)| key == "record_type" && value == "DocumentPut"));
        }
        assert_eq!(fs::read(&path).unwrap(), before);
        let mut damaged = before.clone();
        damaged[document_offset as usize + RECORD_HEADER_LEN] ^= 1;
        fs::write(&path, damaged).unwrap();
        let scan = scan_file(&path).unwrap();
        assert!(scan
            .findings
            .iter()
            .any(|issue| issue.code == "checksum_mismatch"
                && issue.visibility_impact == "blocks_main"));
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn legacy_scan_retains_batch_json_and_releases_it_on_rejection() {
        use crate::json::{JsonInputKind, JsonOperationContext, JsonProcessingPolicyV1};
        let path = temp_path("legacy-pending-json");
        let hex = include_str!("../tests/fixtures/legacy-2d9eb64.hex")
            .split_whitespace()
            .collect::<String>();
        let mut bytes = hex
            .as_bytes()
            .chunks_exact(2)
            .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
            .collect::<Vec<_>>();
        fs::write(&path, &bytes).unwrap();
        let original = scan_file(&path).unwrap();
        bytes.truncate(original.critical_end_offset as usize);
        let batch = original.max_batch_id + 1;
        let sequence = original.max_sequence_no + 1;
        let append = |bytes: &mut Vec<u8>, record_type: u16, sequence: u64, payload: &str| {
            let mut header = Vec::new();
            header.extend_from_slice(&record_type.to_le_bytes());
            header.extend_from_slice(&1u16.to_le_bytes());
            header.extend_from_slice(&batch.to_le_bytes());
            header.extend_from_slice(&sequence.to_le_bytes());
            header.extend_from_slice(&(payload.len() as u64).to_le_bytes());
            header.extend_from_slice(&checksum64(payload.as_bytes()).to_le_bytes());
            header.extend_from_slice(&crate::hash::checksum32(&header).to_le_bytes());
            bytes.extend_from_slice(&header);
            bytes.extend_from_slice(payload.as_bytes());
        };
        let mut checksum = 0xcbf29ce484222325u64;
        let first_offset = bytes.len();
        let mut second_offset = 0;
        for index in 0..2 {
            let id = 900 + index;
            let payload = format!("{{\"document_id\":\"doc_{id}\",\"collection_id\":\"col_1\",\"external_id\":\"doc-{id}\",\"title\":null,\"source_uri\":null,\"metadata\":{{\"payload\":\"{}\"}},\"created_at\":\"1970-01-01T00:00:00Z\",\"updated_at\":\"1970-01-01T00:00:00Z\"}}", "x".repeat(128 * 1024));
            if index == 1 {
                second_offset = bytes.len();
            }
            checksum ^= checksum64(payload.as_bytes());
            append(&mut bytes, DOCUMENT_PUT, sequence + index, &payload);
        }
        let commit = format!("{{\"batch_id\":{batch},\"first_sequence_no\":{sequence},\"last_sequence_no\":{},\"record_count\":2,\"batch_checksum\":\"{checksum}\",\"committed_at\":\"1970-01-01T00:00:00Z\"}}", sequence + 1);
        append(&mut bytes, COMMIT, sequence + 2, &commit);
        fs::write(&path, &bytes).unwrap();
        let healthy = scan_file(&path).unwrap();
        assert!(healthy.findings.is_empty());
        assert_eq!(healthy.records.len(), original.records.len() + 2);
        let context = JsonOperationContext::standalone();
        let capacity = context.available_workspace_bytes();
        let mut visits = 0;
        let error = crate::test_alloc::fail_byte_allocation_once(128 * 1024, || {
            scan_file_with_records_in(&path, &context, |_| {
                visits += 1;
                Ok(())
            })
        })
        .unwrap_err();
        assert_eq!(error.code(), ErrorCode::IoError);
        assert!(!crate::json::is_limit_error(&error));
        assert!(error
            .detail()
            .iter()
            .any(|(key, value)| key == "record_offset" && value == &first_offset.to_string()));
        assert!(error
            .detail()
            .iter()
            .any(|(key, value)| key == "record_type" && value == "DocumentPut"));
        assert_eq!(visits, original.records.len());
        assert_eq!(context.available_workspace_bytes(), capacity);
        assert_eq!(fs::read(&path).unwrap(), bytes);
        for seed in 0..100 {
            let available = 320 * 1024 + seed;
            let mut held = context.reservation();
            held.grow(
                JsonProcessingPolicyV1::WORK_BUDGET_BYTES - available,
                JsonInputKind::Filter,
            )
            .unwrap();
            let mut visits = 0;
            let error = scan_file_with_records_in(&path, &context, |_| {
                visits += 1;
                Ok(())
            })
            .unwrap_err();
            assert_eq!(visits, original.records.len());
            assert_eq!(error.code(), ErrorCode::RecoveryError);
            assert!(
                error
                    .detail()
                    .iter()
                    .any(|(key, value)| key == "record_offset"
                        && value == &second_offset.to_string()),
                "{error}"
            );
            assert!(error
                .detail()
                .iter()
                .any(|(key, value)| key == "json_input_kind" && value == "stored_json"));
            held.grow(available, JsonInputKind::Filter).unwrap();
        }
        let mut documents = 0;
        let scan = scan_file_with_records_in(&path, &context, |record| {
            if let RecordPayload::DocumentPut(document) = record.payload {
                if document.document_id.starts_with("doc_90") {
                    assert_eq!(
                        document
                            .metadata
                            .get("payload")
                            .unwrap()
                            .as_str()
                            .unwrap()
                            .len(),
                        128 * 1024
                    );
                    documents += 1;
                }
            }
            Ok(())
        })
        .unwrap();
        assert_eq!(documents, 2);
        assert_eq!(scan.commits, original.commits + 1);
        assert!(scan.findings.is_empty());
        context
            .reservation()
            .grow(
                JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
                JsonInputKind::Filter,
            )
            .unwrap();
        assert_eq!(fs::read(&path).unwrap(), bytes);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn legacy_json_depth_limit_is_an_explicit_non_mutating_refusal() {
        let path = temp_path("legacy-depth-limit");
        let hex = include_str!("../tests/fixtures/legacy-2d9eb64.hex")
            .split_whitespace()
            .collect::<String>();
        let mut bytes = hex
            .as_bytes()
            .chunks_exact(2)
            .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
            .collect::<Vec<_>>();
        fs::write(&path, &bytes).unwrap();
        let original = scan_file(&path).unwrap();
        bytes.truncate(original.critical_end_offset as usize);
        let batch = original.max_batch_id + 1;
        let sequence = original.max_sequence_no + 1;
        let raw = format!("{{\"document_id\":\"doc_99\",\"collection_id\":\"col_1\",\"external_id\":\"doc\",\"title\":null,\"source_uri\":null,\"metadata\":{{\"v\":{}0{}}},\"created_at\":\"1970-01-01T00:00:00Z\",\"updated_at\":\"1970-01-01T00:00:00Z\"}}", "[".repeat(128), "]".repeat(128));
        let append = |bytes: &mut Vec<u8>, record_type: u16, sequence: u64, payload: &str| {
            let mut header = Vec::new();
            header.extend_from_slice(&record_type.to_le_bytes());
            header.extend_from_slice(&1u16.to_le_bytes());
            header.extend_from_slice(&batch.to_le_bytes());
            header.extend_from_slice(&sequence.to_le_bytes());
            header.extend_from_slice(&(payload.len() as u64).to_le_bytes());
            header.extend_from_slice(&checksum64(payload.as_bytes()).to_le_bytes());
            header.extend_from_slice(&crate::hash::checksum32(&header).to_le_bytes());
            bytes.extend_from_slice(&header);
            bytes.extend_from_slice(payload.as_bytes());
        };
        append(&mut bytes, DOCUMENT_PUT, sequence, &raw);
        let checksum = 0xcbf29ce484222325u64 ^ checksum64(raw.as_bytes());
        let commit = format!("{{\"batch_id\":{batch},\"first_sequence_no\":{sequence},\"last_sequence_no\":{sequence},\"record_count\":1,\"batch_checksum\":\"{checksum}\",\"committed_at\":\"1970-01-01T00:00:00Z\"}}");
        append(&mut bytes, COMMIT, sequence + 1, &commit);
        fs::write(&path, &bytes).unwrap();
        let err = scan_file(&path).unwrap_err();
        assert_eq!(err.code(), ErrorCode::RecoveryError);
        assert!(crate::json::is_depth_error(&err));
        assert!(Storage::open(&path).is_err());
        assert_eq!(fs::read(&path).unwrap(), bytes);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn failed_critical_write_requires_reopen_before_retry() {
        let path = temp_path("write-failure-retry");
        let mut storage = Storage::create(&path).unwrap();
        storage
            .append_batch(&[collection_record("col_1", "first")], Durability::Commit)
            .unwrap();
        let original = fs::read(&path).unwrap();
        storage.file =
            ManagedFile::open_in(&crate::json::JsonOperationContext::standalone(), || {
                Ok(File::open(&path)?)
            })
            .unwrap();
        assert!(storage
            .append_batch(&[collection_record("col_2", "second")], Durability::Commit)
            .is_err());
        let err = storage
            .append_batch(&[collection_record("col_2", "retry")], Durability::Commit)
            .unwrap_err();
        assert_eq!(err.code(), ErrorCode::RecoveryError);
        assert_eq!(fs::read(&path).unwrap(), original);
        drop(storage);
        let (mut storage, scan) = Storage::open(&path).unwrap();
        assert_eq!(scan.commits, 1);
        storage
            .append_batch(&[collection_record("col_2", "retry")], Durability::Commit)
            .unwrap();
        assert_eq!(scan_file(&path).unwrap().commits, 2);
        drop(storage);
        let _ = fs::remove_file(path);
    }

    #[test]
    fn compaction_rejects_damaged_temporary_commit_without_replacing_source() {
        let path = temp_path("rewrite-critical-crc");
        let mut storage = Storage::create(&path).unwrap();
        let records = [collection_record("col_1", "source")];
        storage.append_batch(&records, Durability::Commit).unwrap();
        let original = fs::read(&path).unwrap();
        let err = storage
            .rewrite_with_records_with_hook(&records, Durability::Commit, |temporary| {
                let mut bytes = fs::read(temporary)?;
                bytes[HEADER_LEN + RECORD_HEADER_LEN] ^= 1;
                fs::write(temporary, bytes)?;
                Ok(())
            })
            .unwrap_err();
        assert_eq!(err.code(), ErrorCode::RecoveryError);
        assert_eq!(fs::read(&path).unwrap(), original);
        assert!(!path.with_extension("coffer.tmp").exists());
        drop(storage);
        let _ = fs::remove_file(path);
    }

    #[test]
    fn streamed_compaction_validation_matches_collected_recovery() {
        let dir = temp_path("stream-compact-validation");
        fs::create_dir(&dir).unwrap();
        let path = dir.join("data.coffer");
        let mut storage = Storage::create(&path).unwrap();
        for seed in 0..100usize {
            let count = seed % 11 + 1;
            let records = (0..count).map(|index| {
                Ok(collection_record(
                    &format!("col_{}", index + 1),
                    &format!("collection-{seed}-{index}"),
                ))
            });
            let mut catalog = crate::catalog::Catalog::default();
            let scan = rewrite_database(
                &mut storage,
                records,
                RewritePlan {
                    records: count,
                    batch_record_limit: seed % 4 + 1,
                    batch_byte_target: 128 + (seed as u64 % 5) * 64,
                    temporary_disk: Default::default(),
                },
                if seed % 2 == 0 {
                    Durability::Fast
                } else {
                    Durability::Commit
                },
                |record| catalog.apply_owned_committed(record),
            )
            .unwrap();
            assert!(scan.records.is_empty());
            assert!(scan.findings.is_empty());
            catalog.last_commit_id = scan.last_valid_commit;
            let collected = scan_file(&path).unwrap();
            let expected = crate::catalog::Catalog::from_committed_records(
                collected.records,
                collected.last_valid_commit,
            )
            .unwrap();
            assert_eq!(format!("{catalog:?}"), format!("{expected:?}"));
            for reference in catalog.record_refs.values() {
                assert!(matches!(
                    storage.read_payload(reference).unwrap(),
                    RecordPayload::CollectionCreate(_)
                ));
            }
            assert_eq!(storage.next_batch_id, scan.last_valid_commit + 1);
            assert_eq!(
                storage.next_sequence_no,
                scan.last_valid_commit_sequence_no + 1
            );
        }
        drop(storage);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    #[cfg(unix)]
    fn compact_file_creation_reclaims_before_consuming_records_and_does_not_retry_disk_failure() {
        use crate::json::{JsonOperationContext, WorkLedger};
        use std::cell::Cell;
        use std::sync::Arc;
        let directory = temp_path("compact-creation-reclaim");
        fs::create_dir(&directory).unwrap();
        let path = directory.join("data.coffer");
        let mut storage = Storage::create(&path).unwrap();
        storage
            .append_batch(&[collection_record("col_1", "original")], Durability::Fast)
            .unwrap();
        let original = fs::read(&path).unwrap();
        let management =
            crate::temporary::test_file_management_bytes(storage.path().parent().unwrap());
        let limit = 64 * 1024;
        for disk_failure in [true, false] {
            let context = JsonOperationContext::new(Arc::new(WorkLedger::with_limits(
                2 * limit,
                limit,
                limit,
            )));
            let occupied = if disk_failure {
                1
            } else {
                limit - management - 128 + 1
            };
            let mut cache = Some(context.cache_reservation(occupied as u128, limit).unwrap());
            let disk = Arc::new(crate::temporary::TemporaryDiskBudget::new(
                if disk_failure {
                    HEADER_LEN as u64 - 1
                } else {
                    4096
                },
            ));
            let mut calls = 0;
            let consumed = Cell::new(0);
            let mut validated = 0;
            let mut stages = Vec::new();
            let records = std::iter::once_with(|| {
                consumed.set(consumed.get() + 1);
                Ok(WriteRecord::encoded(collection_record(
                    "col_1",
                    "replacement",
                )))
            });
            let result = storage.rewrite_records_with_hook(
                records,
                RewritePlan {
                    temporary_disk: disk.clone(),
                    ..RewritePlan::single_batch(1)
                },
                Durability::Commit,
                &context,
                || {
                    calls += 1;
                    assert_eq!(consumed.get(), 0);
                    drop(cache.take());
                    true
                },
                |stage, _| {
                    stages.push(stage);
                    Ok(())
                },
                |record, _| {
                    validated += 1;
                    assert_eq!(
                        codec::encode(&record.payload).unwrap(),
                        codec::encode(&collection_record("col_1", "replacement")).unwrap()
                    );
                    Ok(())
                },
            );
            if disk_failure {
                let error = result.unwrap_err();
                assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                assert!(error
                    .detail()
                    .contains(&("resource".into(), "temporary_disk".into())));
                assert_eq!((calls, consumed.get(), validated), (0, 0, 0));
                assert!(stages.is_empty());
                assert!(cache.is_some());
                assert_eq!(fs::read(&path).unwrap(), original);
            } else {
                let scan = result.unwrap();
                assert_eq!((calls, consumed.get(), validated), (1, 1, 1));
                assert_eq!(scan.commits, 1);
                assert!(scan.findings.is_empty());
                assert!(stages == [RewriteStage::BeforeValidate, RewriteStage::AfterRename]);
                assert!(cache.is_none());
            }
            drop(cache);
            let mut free = context.workspace_reservation();
            let available = context.available_workspace_bytes();
            free.grow_to(available, available, "operation_scratch")
                .unwrap();
            assert!(free
                .grow_to(available + 1, available + 1, "operation_scratch")
                .is_err());
            assert_eq!(disk.used(), 0);
            assert_eq!(fs::read_dir(&directory).unwrap().count(), 2);
        }
        drop(storage);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    #[cfg(unix)]
    fn compact_creation_memory_retry_refusal_preserves_source_and_reopens() {
        use crate::json::{JsonOperationContext, WorkLedger};
        use std::cell::Cell;
        use std::sync::Arc;
        let directory = temp_path("compact-creation-retry-refusal");
        fs::create_dir(&directory).unwrap();
        let path = directory.join("data.coffer");
        let mut storage = Storage::create(&path).unwrap();
        storage
            .append_batch(
                &[collection_record("col_1", "original")],
                Durability::Commit,
            )
            .unwrap();
        let original = fs::read(&path).unwrap();
        let limit = 64 * 1024;
        let context =
            JsonOperationContext::new(Arc::new(WorkLedger::with_limits(2 * limit, limit, limit)));
        let mut cache = context
            .cache_reservation((limit - 1) as u128, limit)
            .unwrap();
        let disk = Arc::new(crate::temporary::TemporaryDiskBudget::new(4096));
        let first_error = crate::temporary::create_budgeted_temporary_file_in_context(
            &directory,
            "compact",
            "tmp",
            disk.clone(),
            &context,
            "operation_scratch",
        )
        .err()
        .unwrap();
        assert!(crate::json::is_memory_budget_error(&first_error));
        let requested = |error: &Error| {
            error
                .detail()
                .iter()
                .find(|(key, _)| key == "requested_bytes")
                .unwrap()
                .1
                .parse::<u128>()
                .unwrap()
        };
        let first_requested = requested(&first_error);
        let consumed = Cell::new(0);
        let mut reclaims = 0;
        let error = storage
            .rewrite_records_with_hook(
                std::iter::once_with(|| {
                    consumed.set(consumed.get() + 1);
                    Ok(WriteRecord::encoded(collection_record(
                        "col_1",
                        "replacement",
                    )))
                }),
                RewritePlan {
                    temporary_disk: disk.clone(),
                    ..RewritePlan::single_batch(1)
                },
                Durability::Commit,
                &context,
                || {
                    reclaims += 1;
                    cache.retain(limit - 2);
                    true
                },
                |_, _| panic!("refused admission must not validate or publish"),
                |_, _| panic!("refused admission must not visit records"),
            )
            .unwrap_err();
        assert!(crate::json::is_memory_budget_error(&error));
        // The changed request proves a second real admission after reclaim,
        // rather than returning the original error without trying again.
        assert_eq!(requested(&error) + 1, first_requested);
        assert_eq!((reclaims, consumed.get()), (1, 0));
        assert_eq!(fs::read(&path).unwrap(), original);
        assert_eq!(disk.used(), 0);
        assert_eq!(fs::read_dir(&directory).unwrap().count(), 2);
        drop((cache, storage));
        let reopened = crate::Database::open_read_only(&path).unwrap();
        assert_eq!(reopened.list_collections().unwrap()[0].name, "original");
        assert_eq!(reopened.stats().unwrap().collection_count, 1);
        reopened.close().unwrap();
        assert_eq!(fs::read(&path).unwrap(), original);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn compact_validation_uses_shared_context_before_publication() {
        use crate::json::{JsonInputKind, JsonOperationContext, JsonProcessingPolicyV1};
        let directory = temp_path("compact-validation-json");
        fs::create_dir(&directory).unwrap();
        let path = directory.join("data.coffer");
        let mut storage = Storage::create(&path).unwrap();
        storage
            .append_batch(&[collection_record("col_1", "original")], Durability::Fast)
            .unwrap();
        let original = fs::read(&path).unwrap();
        let file_count = fs::read_dir(&directory).unwrap().count();
        let context = JsonOperationContext::standalone();
        let management =
            crate::temporary::test_file_management_bytes(storage.path().parent().unwrap());
        for available in 0..100 {
            let mut held = context.reservation();
            let records = std::iter::once(Ok(WriteRecord {
                payload: collection_record("col_1", "replacement"),
                prepared: None,
                encoding: RecordEncoding::Stored(context.clone(), context.reservation().into()),
            }));
            let error = storage
                .rewrite_records_with_hook(
                    records,
                    RewritePlan::single_batch(1),
                    Durability::Commit,
                    &context,
                    || false,
                    |stage, _| {
                        if stage == RewriteStage::BeforeValidate {
                            held.grow(
                                JsonProcessingPolicyV1::WORK_BUDGET_BYTES - available - management,
                                JsonInputKind::Filter,
                            )?;
                        }
                        Ok(())
                    },
                    |_, _| panic!("a rejected batch must not reach the visitor"),
                )
                .unwrap_err();
            // The ID, name and timestamp precede the collection's JSON config.
            let field_bytes = [20, 20 + "replacement".len(), 40 + "replacement".len()];
            if let Some(required) = field_bytes.into_iter().find(|bytes| *bytes > available) {
                assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                for (key, value) in [
                    ("resource", "memory".into()),
                    ("operation", "operation_scratch".into()),
                    (
                        "limit_bytes",
                        JsonProcessingPolicyV1::WORK_BUDGET_BYTES.to_string(),
                    ),
                    (
                        "requested_bytes",
                        (JsonProcessingPolicyV1::WORK_BUDGET_BYTES - available + required)
                            .to_string(),
                    ),
                ] {
                    assert!(error.detail().contains(&(key.into(), value)), "{error:?}");
                }
            } else {
                assert_eq!(error.code(), ErrorCode::RecoveryError);
                for (key, value) in [
                    ("json_input_kind", "stored_json".into()),
                    ("json_limit_reason", "work_budget_exceeded".into()),
                    // The held JSON belongs to this same operation, so the
                    // effective limit includes its existing live credit.
                    (
                        "limit",
                        (JsonProcessingPolicyV1::WORK_BUDGET_BYTES - management - field_bytes[2])
                            .to_string(),
                    ),
                ] {
                    assert!(error.detail().contains(&(key.into(), value)), "{error:?}");
                }
            }
            held.grow(available + management, JsonInputKind::Filter)
                .unwrap();
            assert_eq!(fs::read(&path).unwrap(), original);
            assert_eq!(fs::read_dir(&directory).unwrap().count(), file_count);
            storage.require_writer().unwrap();
        }
        drop(storage);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn writer_keeps_owned_vector_credit_through_encoding_and_returns_it_on_failures() {
        use crate::json::{JsonOperationContext, JsonProcessingPolicyV1};
        let limit = JsonProcessingPolicyV1::WORK_BUDGET_BYTES;
        for seed in 0..100usize {
            let path = temp_path("writer-vector-credit");
            let storage = Storage::create(&path).unwrap();
            let before = fs::read(&path).unwrap();
            let context = JsonOperationContext::standalone();
            let embedding = Embedding {
                embedding_id: "emb_1".into(),
                chunk_id: "chk_1".into(),
                collection_id: "col_1".into(),
                chunk_external_id: "external".into(),
                model_name: "model".into(),
                dim: 2,
                vector: vec![1.0, 0.0],
                created_at: "1970-01-01T00:00:00Z".into(),
            };
            let (mut value, work) = RetainedEmbedding::copy(&embedding, &context)
                .unwrap()
                .into_parts();
            let bytes = work.reserved_bytes();
            let mut held = context.workspace_reservation();
            let occupied = limit - bytes - time::MAX_TIMESTAMP_BYTES;
            held.grow_to(occupied, occupied, "operation_scratch")
                .unwrap();
            if seed % 3 == 1 {
                value.dim = 3;
            }
            let mut file = if seed % 3 == 2 {
                File::open(&path).unwrap()
            } else {
                File::try_clone(&storage.file).unwrap()
            };
            let mut batch = 1;
            let mut sequence = 1;
            let mut writer = BatchWriter::new(
                &mut file,
                &mut batch,
                &mut sequence,
                1,
                false,
                &context,
                None,
            )
            .unwrap();
            let result = writer.append(WriteRecord {
                payload: RecordPayload::EmbeddingPut(value),
                prepared: None,
                encoding: RecordEncoding::Stored(
                    context.clone(),
                    RecordReservation::new(context.reservation(), work),
                ),
            });
            assert_eq!(writer.encoding.buffer.capacity(), 0);
            if seed % 3 == 0 {
                result.unwrap();
                writer.finish(Durability::Fast).unwrap();
            } else {
                assert_eq!(
                    result.unwrap_err().code(),
                    if seed % 3 == 1 {
                        ErrorCode::FileFormatError
                    } else {
                        ErrorCode::IoError
                    }
                );
                drop(writer);
                assert_eq!(fs::read(&path).unwrap(), before);
            }
            held.grow_to(limit, limit, "operation_scratch").unwrap();
            drop(file);
            drop(storage);
            fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn compact_splits_metadata_heavy_batches_before_validation() {
        use crate::json::{JsonInputKind, JsonOperationContext, JsonProcessingPolicyV1};
        let directory = temp_path("compact-json-batches");
        fs::create_dir(&directory).unwrap();
        let path = directory.join("data.coffer");
        let mut storage = Storage::create(&path).unwrap();
        let context = JsonOperationContext::standalone();
        let width = 6 * 1024 * 1024;
        let records = std::iter::once(Ok((
            collection_record("col_1", "notes"),
            context.reservation(),
        )))
        .chain((1..=6).map(|id| {
            let metadata = object_from_pairs([("payload", JsonValue::String("x".repeat(width)))]);
            let work = metadata.reserve_retained_in(JsonInputKind::StoredJson, &context)?;
            Ok((
                RecordPayload::DocumentPut(Document {
                    document_id: format!("doc_{id}"),
                    collection_id: "col_1".into(),
                    external_id: format!("doc-{id}"),
                    title: None,
                    source_uri: None,
                    metadata,
                    created_at: "1970-01-01T00:00:00Z".into(),
                    updated_at: "1970-01-01T00:00:00Z".into(),
                }),
                work,
            ))
        }));
        let mut documents = 0;
        let scan = rewrite_database_in(
            &mut storage,
            records,
            RewritePlan::single_batch(7),
            Durability::Commit,
            &context,
            |record| {
                if let RecordPayload::DocumentPut(document) = record.payload {
                    assert_eq!(
                        document
                            .metadata
                            .get("payload")
                            .unwrap()
                            .as_str()
                            .unwrap()
                            .len(),
                        width
                    );
                    documents += 1;
                }
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(documents, 6);
        assert_eq!(scan.commits, 3);
        assert!(scan.findings.is_empty());
        context
            .reservation()
            .grow(
                JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
                JsonInputKind::Filter,
            )
            .unwrap();
        let mut reread = 0;
        let scan = scan_file_with_records(&path, |_| {
            reread += 1;
            Ok(())
        })
        .unwrap();
        assert_eq!(reread, 7);
        assert!(scan.findings.is_empty());
        drop(storage);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn bounded_compaction_keeps_prefix_checksums_and_respects_batch_boundaries() {
        let dir = temp_path("bounded-compact-checksums");
        fs::create_dir(&dir).unwrap();
        let path = dir.join("data.coffer");
        let mut storage = Storage::create(&path).unwrap();
        for seed in 0..100usize {
            let count = seed % 13;
            let width = [
                10,
                MAX_ENCODE_BUFFER_BYTES - 128,
                MAX_ENCODE_BUFFER_BYTES + 1,
            ][seed % 3];
            let record_limit = [1, 3, 7][seed % 3];
            let byte_target = [1, 1000, 128 * 1024, 1024 * 1024][seed % 4];
            let source_sequence = storage.next_sequence_no - 1;
            let records = (1..=count)
                .map(|index| {
                    Ok(collection_record(
                        &format!("col_{index}"),
                        &format!("collection-{index}-{}", "x".repeat(width)),
                    ))
                })
                .chain(std::iter::once(Ok(RecordPayload::CompactMarker(
                    CompactMarker {
                        compact_generation: seed as u64,
                        source_last_commit_sequence_no: source_sequence,
                        compacted_at: "1970-01-01T00:00:00Z".into(),
                        records_rewritten: count,
                        id_high_watermarks: BTreeMap::from([("col_".into(), 13)]),
                        compacted_main_checksum: 0,
                    },
                ))));
            let context = crate::json::JsonOperationContext::standalone();
            let report = rewrite_database_in(
                &mut storage,
                records.map(|record| record.map(|payload| (payload, context.reservation()))),
                RewritePlan {
                    records: count + 1,
                    batch_record_limit: record_limit,
                    batch_byte_target: byte_target,
                    temporary_disk: Default::default(),
                },
                Durability::Commit,
                &context,
                |_| Ok(()),
            )
            .unwrap();
            assert!(report.findings.is_empty());
            let scan = scan_file(&path).unwrap();
            assert_eq!(scan.records.len(), count + 1);
            let bytes = fs::read(&path).unwrap();
            let mut prefix = Crc32c::default();
            let mut batches = BTreeMap::<u64, (usize, u64, u64)>::new();
            for record in &scan.records {
                let reference = &record.record_ref;
                let length = reference.payload.len + RECORD_HEADER_LEN as u64;
                if let RecordPayload::CompactMarker(marker) = &record.payload {
                    assert_eq!(marker.compacted_main_checksum, prefix.finish());
                    assert_eq!(marker.source_last_commit_sequence_no, source_sequence);
                    assert_eq!(marker.records_rewritten, count);
                }
                prefix.update(
                    &bytes[reference.offset as usize..(reference.offset + length) as usize],
                );
                let batch = batches.entry(reference.batch_id).or_default();
                batch.0 += 1;
                batch.1 += length;
                batch.2 = length;
            }
            assert_eq!(report.commits, batches.len());
            for (count, bytes, last_record_bytes) in batches.values() {
                assert!(*count <= record_limit);
                assert!(*bytes - last_record_bytes < byte_target);
            }
            assert_eq!(
                storage.next_sequence_no,
                scan.last_valid_commit_sequence_no + 1
            );
            assert_eq!(storage.next_batch_id, scan.last_valid_commit + 1);
        }
        drop(storage);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn bounded_compaction_rejects_late_corruption_without_publishing_earlier_batches() {
        let dir = temp_path("bounded-compact-corruption");
        fs::create_dir(&dir).unwrap();
        let path = dir.join("data.coffer");
        let mut storage = Storage::create(&path).unwrap();
        storage
            .append_batch(&[collection_record("col_1", "original")], Durability::Fast)
            .unwrap();
        let original = fs::read(&path).unwrap();
        for damaged_record in 0..4 {
            let mut visited = 0;
            let error = storage
                .rewrite_with_iter_with_hook(
                    (1..=4)
                        .map(|index| Ok(collection_record(&format!("col_{index}"), "replacement"))),
                    RewritePlan {
                        records: 4,
                        batch_record_limit: 1,
                        batch_byte_target: 1,
                        temporary_disk: Default::default(),
                    },
                    Durability::Commit,
                    |stage, temporary| {
                        if stage == RewriteStage::BeforeValidate {
                            let scan = scan_file(temporary)?;
                            let mut bytes = fs::read(temporary)?;
                            bytes[scan.records[damaged_record].record_ref.payload.offset
                                as usize] ^= 1;
                            fs::write(temporary, bytes)?;
                        }
                        Ok(())
                    },
                    |_| {
                        visited += 1;
                        Ok(())
                    },
                )
                .unwrap_err();
            assert_eq!(error.code(), ErrorCode::RecoveryError);
            assert_eq!(visited, damaged_record);
            assert_eq!(fs::read(&path).unwrap(), original);
            assert_eq!(fs::read_dir(&dir).unwrap().count(), 2);
            storage.require_writer().unwrap();
        }
        for (records, target) in [(0, 1), (1, 0)] {
            assert!(rewrite_database(
                &mut storage,
                std::iter::empty(),
                RewritePlan {
                    records: 0,
                    batch_record_limit: records,
                    batch_byte_target: target,
                    temporary_disk: Default::default(),
                },
                Durability::Fast,
                |_| Ok(())
            )
            .is_err());
            assert_eq!(fs::read(&path).unwrap(), original);
            assert_eq!(fs::read_dir(&dir).unwrap().count(), 2);
        }
        drop(storage);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn compaction_reserves_every_byte_and_obeys_shared_disk_boundaries() {
        use crate::temporary::{create_budgeted_temporary_file_in, TemporaryDiskBudget};
        use std::sync::Arc;
        let dir = temp_path("compact-shared-disk");
        fs::create_dir(&dir).unwrap();
        let path = dir.join("data.coffer");
        let mut storage = Storage::create(&path).unwrap();
        let records = || (1..=3).map(|id| Ok(collection_record(&format!("col_{id}"), "target")));
        for batch_limit in [1, 3] {
            let plan = |budget| RewritePlan {
                records: 3,
                batch_record_limit: batch_limit,
                batch_byte_target: u64::MAX,
                temporary_disk: budget,
            };
            rewrite_database(
                &mut storage,
                records(),
                plan(Default::default()),
                Durability::Fast,
                |_| Ok(()),
            )
            .unwrap();
            let required = storage.file_size().unwrap();
            for seed in 0..100u64 {
                let available = required * seed / 99;
                let held_bytes = seed * 7;
                let budget = Arc::new(TemporaryDiskBudget::new(available + held_bytes));
                let (mut file, mut other) =
                    create_budgeted_temporary_file_in(&dir, "held", "tmp", budget.clone()).unwrap();
                other.reserve_disk(held_bytes).unwrap();
                file.write_all(&vec![0; held_bytes as usize]).unwrap();
                drop(file);
                let before = fs::read(&path).unwrap();
                let entries = fs::read_dir(&dir).unwrap().count();
                let mut validation_reached = false;
                let result = storage.rewrite_with_iter_with_hook(
                    records(),
                    plan(budget.clone()),
                    Durability::Fast,
                    |stage, output| {
                        if stage == RewriteStage::BeforeValidate {
                            validation_reached = true;
                            assert_eq!(budget.used(), held_bytes + fs::metadata(output)?.len());
                        }
                        Ok(())
                    },
                    |_| Ok(()),
                );
                if available < required {
                    let error = result.unwrap_err();
                    assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                    assert_eq!(fs::read(&path).unwrap(), before);
                    assert!(!validation_reached);
                    let detail = error
                        .detail()
                        .iter()
                        .map(|(key, value)| (key.as_str(), value.as_str()))
                        .collect::<BTreeMap<_, _>>();
                    assert_eq!(detail["resource"], "temporary_disk");
                    assert_eq!(detail["limit_bytes"], (available + held_bytes).to_string());
                    assert!(
                        detail["requested_bytes"].parse::<u128>().unwrap()
                            > u128::from(available + held_bytes)
                    );
                } else {
                    result.unwrap();
                    assert!(validation_reached);
                    assert_eq!(storage.file_size().unwrap(), required);
                    assert!(scan_file(&path).unwrap().findings.is_empty());
                }
                assert_eq!(budget.used(), held_bytes);
                assert_eq!(fs::read_dir(&dir).unwrap().count(), entries);
                storage.require_writer().unwrap();
                drop(other);
                assert_eq!(budget.used(), 0);
            }
        }
        drop(storage);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn compaction_callback_failures_preserve_source_and_remove_private_output() {
        for phase in ["input", "before_validate", "visit"] {
            for panic in [false, true] {
                for fail_at in 1..=4 {
                    let dir = temp_path("compact-callback-failure");
                    fs::create_dir(&dir).unwrap();
                    let path = dir.join("data.coffer");
                    let mut storage = Storage::create(&path).unwrap();
                    storage
                        .append_batch(&[collection_record("col_1", "original")], Durability::Fast)
                        .unwrap();
                    let original = fs::read(&path).unwrap();
                    let entries = fs::read_dir(&dir).unwrap().count();
                    let fail = || -> Result<()> {
                        assert!(!panic, "injected compaction callback panic");
                        Err(Error::new(
                            ErrorCode::IoError,
                            "injected compaction callback error",
                        ))
                    };
                    let mut visited = 0;
                    let budget =
                        std::sync::Arc::new(crate::temporary::TemporaryDiskBudget::default());
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        storage.rewrite_with_iter_with_hook(
                            (1..=4).map(|index| {
                                if phase == "input" && index == fail_at {
                                    fail()?;
                                }
                                Ok(collection_record(&format!("col_{index}"), "replacement"))
                            }),
                            RewritePlan {
                                records: 4,
                                batch_record_limit: 2,
                                batch_byte_target: 128,
                                temporary_disk: budget.clone(),
                            },
                            Durability::Fast,
                            |stage, _| {
                                if phase == "before_validate"
                                    && stage == RewriteStage::BeforeValidate
                                {
                                    fail()?;
                                }
                                Ok(())
                            },
                            |_| {
                                visited += 1;
                                if phase == "visit" && visited == fail_at {
                                    fail()?;
                                }
                                Ok(())
                            },
                        )
                    }));
                    if panic {
                        assert!(result.is_err());
                    } else {
                        assert_eq!(result.unwrap().unwrap_err().code(), ErrorCode::IoError);
                    }
                    assert_eq!(visited, if phase == "visit" { fail_at } else { 0 });
                    assert_eq!(fs::read(&path).unwrap(), original);
                    assert_eq!(fs::read_dir(&dir).unwrap().count(), entries);
                    assert_eq!(budget.used(), 0);
                    storage.require_writer().unwrap();
                    storage
                        .append_batch(&[collection_record("col_2", "retry")], Durability::Fast)
                        .unwrap();
                    assert_eq!(scan_file(&path).unwrap().commits, 2);
                    drop(storage);
                    fs::remove_dir_all(dir).unwrap();
                }
            }
        }
    }

    #[test]
    fn compaction_post_publish_failure_keeps_old_reads_and_requires_writer_reopen() {
        for panic in [false, true] {
            let dir = temp_path("compact-post-publish-failure");
            fs::create_dir(&dir).unwrap();
            let path = dir.join("data.coffer");
            let mut storage = Storage::create(&path).unwrap();
            let append = storage
                .append_batch(&[collection_record("col_1", "original")], Durability::Fast)
                .unwrap();
            let old_size = storage.file_size().unwrap();
            let old_ref = &append.records[0].record_ref;
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                storage.rewrite_with_iter_with_hook(
                    [Ok(collection_record("col_2", "replacement"))],
                    RewritePlan::single_batch(1),
                    Durability::Commit,
                    |stage, _| {
                        if stage == RewriteStage::AfterRename {
                            assert!(!panic, "injected post-publication panic");
                            return Err(Error::new(ErrorCode::IoError, "injected sync failure"));
                        }
                        Ok(())
                    },
                    |_| Ok(()),
                )
            }));
            if panic {
                assert!(result.is_err());
            } else {
                assert_eq!(result.unwrap().unwrap_err().code(), ErrorCode::IoError);
            }
            assert!(matches!(
                storage.read_payload(old_ref).unwrap(),
                RecordPayload::CollectionCreate(collection) if collection.name == "original"
            ));
            assert_eq!(storage.file_size().unwrap(), old_size);
            assert_eq!(
                storage.require_writer().unwrap_err().code(),
                ErrorCode::RecoveryError
            );
            assert!(storage
                .append_batch(
                    &[collection_record("col_3", "must-not-write")],
                    Durability::Fast
                )
                .is_err());
            let scan = scan_file(&path).unwrap();
            assert!(matches!(
                &scan.records[0].payload,
                RecordPayload::CollectionCreate(collection) if collection.name == "replacement"
            ));
            assert_eq!(fs::read_dir(&dir).unwrap().count(), 2);
            drop(storage);
            let (mut reopened, _) = Storage::open(&path).unwrap();
            reopened
                .append_batch(
                    &[collection_record("col_3", "after-reopen")],
                    Durability::Fast,
                )
                .unwrap();
            assert_eq!(scan_file(&path).unwrap().commits, 2);
            drop(reopened);
            fs::remove_dir_all(dir).unwrap();
        }
    }

    #[test]
    fn streaming_compaction_stops_on_input_error_and_cleans_owned_output() {
        let dir = temp_path("stream-input-error");
        fs::create_dir(&dir).unwrap();
        let path = dir.join("data.coffer");
        let mut storage = Storage::create(&path).unwrap();
        storage
            .append_batch(
                &[collection_record("col_1", "original")],
                Durability::Commit,
            )
            .unwrap();
        let original = fs::read(&path).unwrap();
        let original_entries = fs::read_dir(&dir).unwrap().count();
        let mut requested = 0;
        let records = (0..10).map(|n| {
            requested += 1;
            if n == 5 {
                Err(Error::new(
                    ErrorCode::IoError,
                    "injected input read failure",
                ))
            } else {
                Ok(collection_record(
                    &format!("col_{}", n + 2),
                    &format!("collection-{n}"),
                ))
            }
        });
        let err = rewrite_database(
            &mut storage,
            records,
            RewritePlan {
                records: 10,
                batch_record_limit: 3,
                batch_byte_target: 512,
                temporary_disk: Default::default(),
            },
            Durability::Commit,
            |_| Ok(()),
        )
        .unwrap_err();
        assert_eq!(err.code(), ErrorCode::IoError);
        assert_eq!(requested, 6);
        assert_eq!(fs::read(&path).unwrap(), original);
        assert_eq!(fs::read_dir(&dir).unwrap().count(), original_entries);
        storage
            .append_batch(
                &[collection_record("col_2", "after-failure")],
                Durability::Commit,
            )
            .unwrap();
        assert_eq!(scan_file(&path).unwrap().commits, 2);
        drop(storage);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn streaming_compaction_checks_declared_record_count_before_replacement() {
        let dir = temp_path("stream-count-error");
        fs::create_dir(&dir).unwrap();
        let path = dir.join("data.coffer");
        let mut storage = Storage::create(&path).unwrap();
        storage
            .append_batch(
                &[collection_record("col_1", "original")],
                Durability::Commit,
            )
            .unwrap();
        let original = fs::read(&path).unwrap();
        let original_entries = fs::read_dir(&dir).unwrap().count();
        for (actual, declared) in [(0, 1), (1, 2), (2, 1), (1, 0)] {
            let records = (0..actual)
                .map(|n| Ok(collection_record(&format!("col_{}", n + 2), "replacement")));
            let err = rewrite_database(
                &mut storage,
                records,
                RewritePlan::single_batch(declared),
                Durability::Commit,
                |_| Ok(()),
            )
            .unwrap_err();
            assert_eq!(err.code(), ErrorCode::FileFormatError);
            assert_eq!(fs::read(&path).unwrap(), original);
            assert_eq!(fs::read_dir(&dir).unwrap().count(), original_entries);
        }
        drop(storage);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn incremental_batch_writer_matches_encoded_envelopes_and_commits() {
        for seed in 0..100usize {
            let path = temp_path("incremental-batch-wire");
            let mut storage = Storage::create(&path).unwrap();
            let context = crate::json::JsonOperationContext::standalone();
            let available = [0, 1, 8191, 8192, 9000, 65535, 65536, 73728][seed % 8];
            let mut held = context.workspace_reservation();
            let occupied = crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES
                - available
                - time::MAX_TIMESTAMP_BYTES;
            held.grow_to(occupied, occupied, "operation_scratch")
                .unwrap();
            let count = seed % 7 + 1;
            let width = [
                0,
                MAX_ENCODE_BUFFER_BYTES - 128,
                MAX_ENCODE_BUFFER_BYTES + 1,
            ][seed % 3];
            let mut records = (1..=count)
                .map(|index| {
                    collection_record(
                        &format!("col_{index}"),
                        &format!("collection-{index}-\u{4e2d}{}", "x".repeat(width)),
                    )
                })
                .collect::<Vec<_>>();
            records.push(RecordPayload::CompactMarker(CompactMarker {
                compact_generation: seed as u64,
                source_last_commit_sequence_no: seed as u64 * 13,
                compacted_at: "1970-01-01T00:00:00Z".into(),
                records_rewritten: count,
                id_high_watermarks: BTreeMap::from([("col_".into(), count as u64)]),
                compacted_main_checksum: 0,
            }));
            storage.next_batch_id = seed as u64 + 1;
            storage.next_sequence_no = seed as u64 * 13 + 1;
            let first_sequence = storage.next_sequence_no;
            let batch_id = storage.next_batch_id;
            let mut expected = fs::read(&path).unwrap();
            let mut prefix = Crc32c::default();
            for (index, source) in records.iter().enumerate() {
                let mut record = source.clone();
                if let RecordPayload::CompactMarker(marker) = &mut record {
                    marker.compacted_main_checksum = prefix.finish();
                }
                let payload = codec::encode(&record).unwrap();
                let header = record_header(
                    record_type_id(&record),
                    batch_id,
                    first_sequence + index as u64,
                    &payload,
                )
                .unwrap();
                prefix.update(&header);
                prefix.update(&payload);
                expected.extend_from_slice(&header);
                expected.extend_from_slice(&payload);
            }
            let mut writer = BatchWriter::new(
                &mut storage.file,
                &mut storage.next_batch_id,
                &mut storage.next_sequence_no,
                records.len() + 3,
                seed % 2 == 0,
                &context,
                None,
            )
            .unwrap();
            for record in records {
                writer.append(WriteRecord::encoded(record)).unwrap();
                assert!(writer.encoding.buffer.capacity() <= MAX_ENCODE_BUFFER_BYTES);
            }
            let append = writer.finish(Durability::Fast).unwrap();
            let mut released = context.workspace_reservation();
            assert_eq!(
                released
                    .grow_to(available, available, "operation_scratch")
                    .unwrap(),
                available
            );
            drop(released);
            assert_eq!(append.commit.record_count, count + 1);
            assert_eq!(append.commit.batch_checksum, prefix.finish() as u64);
            assert_eq!(
                append.records.len(),
                if seed % 2 == 0 { count + 1 } else { 0 }
            );
            let commit = codec::encode(&RecordPayload::Commit(append.commit)).unwrap();
            expected.extend_from_slice(
                &record_header(COMMIT, batch_id, append.commit_sequence_no, &commit).unwrap(),
            );
            expected.extend_from_slice(&commit);
            assert_eq!(fs::read(&path).unwrap(), expected);
            let scan = scan_file(&path).unwrap();
            assert!(scan.findings.is_empty());
            assert_eq!(scan.records.len(), count + 1);
            assert_eq!(storage.next_batch_id, batch_id + 1);
            assert_eq!(storage.next_sequence_no, first_sequence + count as u64 + 2);
            drop(storage);
            fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn incremental_batch_writer_checks_admission_and_rejects_empty_finish() {
        let path = temp_path("incremental-batch-admission");
        let mut storage = Storage::create(&path).unwrap();
        let original = fs::read(&path).unwrap();
        for (batch, sequence, limit) in [(1, 1, 0), (u64::MAX, 1, 1), (1, u64::MAX - 1, 1)] {
            let mut next_batch = batch;
            let mut next_sequence = sequence;
            assert!(BatchWriter::new(
                &mut storage.file,
                &mut next_batch,
                &mut next_sequence,
                limit,
                false,
                &crate::json::JsonOperationContext::standalone(),
                None,
            )
            .is_err());
            assert_eq!(next_batch, batch);
            assert_eq!(next_sequence, sequence);
            assert_eq!(fs::read(&path).unwrap(), original);
        }
        let writer = BatchWriter::new(
            &mut storage.file,
            &mut storage.next_batch_id,
            &mut storage.next_sequence_no,
            1,
            false,
            &crate::json::JsonOperationContext::standalone(),
            None,
        )
        .unwrap();
        assert_eq!(
            writer.finish(Durability::Fast).unwrap_err().code(),
            ErrorCode::FileFormatError
        );
        assert_eq!(fs::read(&path).unwrap(), original);
        let mut writer = BatchWriter::new(
            &mut storage.file,
            &mut storage.next_batch_id,
            &mut storage.next_sequence_no,
            1,
            false,
            &crate::json::JsonOperationContext::standalone(),
            None,
        )
        .unwrap();
        writer
            .append(WriteRecord::encoded(collection_record("col_1", "first")))
            .unwrap();
        let length = writer.file.metadata().unwrap().len();
        assert_eq!(
            writer
                .append(WriteRecord::encoded(collection_record("col_2", "extra")))
                .unwrap_err()
                .code(),
            ErrorCode::FileFormatError
        );
        assert_eq!(writer.file.metadata().unwrap().len(), length);
        writer.finish(Durability::Fast).unwrap();
        assert_eq!(scan_file(&path).unwrap().records.len(), 1);
        drop(storage);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn optional_encoding_buffers_yield_to_stored_json_and_release_on_errors() {
        use crate::json::{JsonOperationContext, JsonProcessingPolicyV1};
        for seed in 0..100 {
            for available in [1, 4096, 8192] {
                let context = JsonOperationContext::standalone();
                let mut held = context.workspace_reservation();
                let occupied = JsonProcessingPolicyV1::WORK_BUDGET_BYTES
                    - available
                    - time::MAX_TIMESTAMP_BYTES;
                held.grow_to(occupied, occupied, "operation_scratch")
                    .unwrap();
                let path = temp_path("encoding-json-headroom");
                let mut storage = Storage::create(&path).unwrap();
                let record = RecordPayload::DocumentPut(crate::model::Document {
                    document_id: "doc_1".into(),
                    collection_id: "col_1".into(),
                    external_id: format!("document-{seed}"),
                    title: None,
                    source_uri: None,
                    metadata: crate::json::object_from_pairs([(
                        "value",
                        JsonValue::String("x".repeat(64 + seed)),
                    )]),
                    created_at: "1970-01-01T00:00:00Z".into(),
                    updated_at: "1970-01-01T00:00:00Z".into(),
                });
                let expected = codec::encode(&record).unwrap();
                let mut writer = BatchWriter::new(
                    &mut storage.file,
                    &mut storage.next_batch_id,
                    &mut storage.next_sequence_no,
                    1,
                    false,
                    &context,
                    None,
                )
                .unwrap();
                let result = writer.append(WriteRecord {
                    payload: record,
                    prepared: None,
                    encoding: RecordEncoding::Stored(context.clone(), context.reservation().into()),
                });
                assert!(!writer.encoding.buffering);
                assert_eq!(writer.encoding.buffer.capacity(), 0);
                assert_eq!(writer.encoding.work.reserved_bytes(), 0);
                if available == 1 {
                    let error = result.unwrap_err();
                    assert_eq!(error.code(), ErrorCode::RecoveryError);
                    assert!(crate::json::is_limit_error(&error));
                    assert!(error
                        .detail()
                        .iter()
                        .any(|(key, value)| key == "limit" && value == "1"));
                    assert_eq!(writer.file.metadata().unwrap().len(), HEADER_LEN as u64);
                    drop(writer);
                } else {
                    result.unwrap();
                    writer.finish(Durability::Fast).unwrap();
                    let scan = scan_file(&path).unwrap();
                    assert!(scan.findings.is_empty());
                    assert_eq!(scan.records.len(), 1);
                    assert_eq!(codec::encode(&scan.records[0].payload).unwrap(), expected);
                }
                let mut probe = context.workspace_reservation();
                assert_eq!(
                    probe
                        .grow_to(
                            available + time::MAX_TIMESTAMP_BYTES,
                            available + time::MAX_TIMESTAMP_BYTES,
                            "operation_scratch"
                        )
                        .unwrap(),
                    available + time::MAX_TIMESTAMP_BYTES
                );
                drop(storage);
                fs::remove_file(path).unwrap();
            }
        }
    }

    #[test]
    fn encoding_scratch_respects_shared_headroom_across_chunk_boundaries() {
        use crate::json::{JsonOperationContext, JsonProcessingPolicyV1};
        let bytes = (0..MAX_ENCODE_BUFFER_BYTES + 1)
            .map(|index| (index % 251) as u8)
            .collect::<Vec<_>>();
        for seed in 1..=100 {
            for available in [
                0,
                1,
                31,
                1024,
                MAX_ENCODE_BUFFER_BYTES - 1,
                MAX_ENCODE_BUFFER_BYTES,
            ] {
                let context = JsonOperationContext::standalone();
                let mut held = context.workspace_reservation();
                let occupied = JsonProcessingPolicyV1::WORK_BUDGET_BYTES - available;
                held.grow_to(occupied, occupied, "operation_scratch")
                    .unwrap();
                let mut scratch = EncodingScratch::new(&context);
                for length in [1, 31, 1024, MAX_ENCODE_BUFFER_BYTES, bytes.len()] {
                    scratch.reset();
                    for part in bytes[..length].chunks(seed * 137) {
                        scratch.write_all(part).unwrap();
                        assert_eq!(scratch.buffer.capacity(), scratch.work.reserved_bytes());
                        assert!(scratch.buffer.capacity() <= available);
                    }
                    assert_eq!(scratch.digest.len, length as u64);
                    assert_eq!(scratch.digest.crc.finish(), checksum32(&bytes[..length]));
                    if length <= available {
                        assert_eq!(scratch.buffer, bytes[..length]);
                    } else {
                        assert!(scratch.buffer.is_empty());
                    }
                }
                drop(scratch);
                let mut probe = context.workspace_reservation();
                assert_eq!(
                    probe
                        .grow_to(available, available, "operation_scratch")
                        .unwrap(),
                    available
                );
            }
        }
    }

    #[test]
    fn encoding_scratch_is_bounded_and_reused_at_both_size_boundaries() {
        let mut scratch = EncodingScratch::new(&crate::json::JsonOperationContext::standalone());
        let bytes = (0..MAX_ENCODE_BUFFER_BYTES * 3)
            .map(|index| (index % 251) as u8)
            .collect::<Vec<_>>();
        for seed in 1..=100 {
            for length in [
                MAX_ENCODE_BUFFER_BYTES - 1,
                MAX_ENCODE_BUFFER_BYTES,
                MAX_ENCODE_BUFFER_BYTES + 1,
                bytes.len(),
            ] {
                scratch.reset();
                for part in bytes[..length].chunks(seed * 137) {
                    scratch.write_all(part).unwrap();
                    assert!(scratch.buffer.capacity() <= MAX_ENCODE_BUFFER_BYTES);
                }
                assert_eq!(scratch.digest.len, length as u64);
                assert_eq!(scratch.digest.crc.finish(), checksum32(&bytes[..length]));
                if length <= MAX_ENCODE_BUFFER_BYTES {
                    assert_eq!(scratch.buffer, bytes[..length]);
                } else {
                    assert!(scratch.buffer.is_empty());
                }
                let pointer = scratch.buffer.as_ptr();
                scratch.reset();
                scratch.write_all(b"small").unwrap();
                assert_eq!(scratch.buffer.as_ptr(), pointer);
                assert_eq!(scratch.buffer, b"small");
                assert_eq!(scratch.digest.len, 5);
                assert_eq!(scratch.digest.crc.finish(), checksum32(b"small"));
            }
        }
        assert_eq!(
            record_header_from_digest(COLLECTION_CREATE, 1, 1, u64::MAX, 0)
                .unwrap_err()
                .code(),
            ErrorCode::FileFormatError
        );
    }

    #[test]
    fn streamed_batch_writes_each_record_before_requesting_the_next() {
        let path = temp_path("stream-lazy-write");
        let mut file = create_storage_file(&path).unwrap();
        let mut previous_size = 0;
        let records = (1..=100).map(|n| {
            let size = fs::metadata(&path).unwrap().len();
            assert!(size > previous_size);
            previous_size = size;
            Ok(collection_record(
                &format!("col_{n}"),
                &format!("collection-{n}"),
            ))
        });
        let append = write_batch_to_file(
            &mut file,
            &mut 1,
            &mut 1,
            records,
            100,
            Durability::Commit,
            false,
        )
        .unwrap();
        assert!(append.records.is_empty());
        assert_eq!(append.commit.record_count, 100);
        drop(file);
        let scan = scan_file(&path).unwrap();
        assert_eq!(scan.records.len(), 100);
        assert_eq!(scan.commits, 1);
        assert!(scan.findings.is_empty());
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn borrowed_index_encoding_preserves_canonical_order_and_input_ownership() {
        for seed in 1..=100u64 {
            let mut order = (1..=31u64).collect::<Vec<_>>();
            let mut state = seed;
            for index in (1..order.len()).rev() {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                order.swap(index, state as usize % (index + 1));
            }
            let text = TextIndexSegmentPayload {
                collection_id: "col_1".into(),
                covered_commit_sequence_no: seed,
                chunks: order
                    .iter()
                    .map(|id| TextIndexChunkEntry {
                        chunk_id: format!("chk_{id}"),
                        document_id: "doc_1".into(),
                        token_count: *id as usize + 3,
                    })
                    .collect(),
                terms: ["z", "a", "m"]
                    .into_iter()
                    .map(|term| TextIndexTermEntry {
                        term: term.into(),
                        postings: order
                            .iter()
                            .map(|id| TextIndexPostingEntry {
                                chunk_id: format!("chk_{id}"),
                                term_frequency: *id as usize,
                            })
                            .collect(),
                    })
                    .collect(),
            };
            let text_bytes = encode_text_index_segment_payload(&text).unwrap();
            let mut streamed_text = Vec::new();
            checkpoint_writer::write_text_segment(&text, &mut streamed_text).unwrap();
            assert_eq!(streamed_text, text_bytes);
            let parsed = parse_text_index_segment_payload(&text_bytes, seed, 0)
                .unwrap()
                .payload;
            assert_eq!(
                parsed
                    .terms
                    .iter()
                    .map(|term| term.term.as_str())
                    .collect::<Vec<_>>(),
                ["a", "m", "z"]
            );
            for (index, chunk) in parsed.chunks.iter().enumerate() {
                let id = index as u64 + 1;
                assert_eq!(catalog_id_number(&chunk.chunk_id).unwrap(), id);
                assert_eq!(chunk.token_count, id as usize + 3);
                for term in &parsed.terms {
                    assert_eq!(
                        catalog_id_number(&term.postings[index].chunk_id).unwrap(),
                        id
                    );
                    assert_eq!(term.postings[index].term_frequency, id as usize);
                }
            }
            assert_eq!(
                encode_text_index_segment_payload(&parsed).unwrap(),
                text_bytes
            );
            assert_eq!(text.terms[0].term, "z");
            for (index, id) in order.iter().enumerate() {
                assert_eq!(text.chunks[index].chunk_id, format!("chk_{id}"));
                assert_eq!(text.terms[0].postings[index].chunk_id, format!("chk_{id}"));
            }

            let mut vectors = VectorIndexSegmentPayload {
                collection_id: "col_1".into(),
                covered_commit_sequence_no: seed,
                dim: 3,
                distance_metric: DistanceMetric::DotProduct,
                embedding_model: Some("m".into()),
                vectors_normalized: false,
                vectors: order
                    .iter()
                    .map(|id| VectorIndexEntry {
                        chunk_id: format!("chk_{id}"),
                        embedding_id: format!("emb_{}", id + 100),
                        vector: vec![*id as f32, -0.0, f32::from_bits(1)],
                    })
                    .collect(),
            };
            let pointers = vectors
                .vectors
                .iter()
                .map(|entry| entry.vector.as_ptr())
                .collect::<Vec<_>>();
            let bytes = encode_vector_index_segment_payload(&vectors).unwrap();
            let mut streamed_vectors = Vec::new();
            checkpoint_writer::write_vector_segment(&vectors, &mut streamed_vectors).unwrap();
            assert_eq!(streamed_vectors, bytes);
            assert_eq!(
                vectors.encoded_len().unwrap() as usize,
                bytes.len() + INDEX_RECORD_HEADER_LEN
            );
            let parsed = parse_vector_index_segment_payload(&bytes, seed, 0)
                .unwrap()
                .payload
                .unwrap();
            for (index, entry) in parsed.vectors.iter().enumerate() {
                let id = index as u64 + 1;
                assert_eq!(catalog_id_number(&entry.chunk_id).unwrap(), id);
                assert_eq!(catalog_id_number(&entry.embedding_id).unwrap(), id + 100);
                assert_eq!(
                    entry
                        .vector
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    [(id as f32).to_bits(), 0x8000_0000, 1]
                );
            }
            assert_eq!(encode_vector_index_segment_payload(&parsed).unwrap(), bytes);
            for (index, id) in order.iter().enumerate() {
                assert_eq!(vectors.vectors[index].chunk_id, format!("chk_{id}"));
                assert_eq!(vectors.vectors[index].vector.as_ptr(), pointers[index]);
            }
            vectors.vectors[0].vector.push(1.0);
            assert_eq!(
                encode_vector_index_segment_payload(&vectors)
                    .unwrap_err()
                    .code(),
                ErrorCode::InvalidVectorLength
            );
            vectors.vectors[0].vector.pop();
            vectors.vectors[0].vector[0] = f32::INFINITY;
            assert_eq!(
                encode_vector_index_segment_payload(&vectors)
                    .unwrap_err()
                    .code(),
                ErrorCode::InvalidVectorLength
            );
        }
    }

    #[test]
    fn incremental_index_commit_checksum_matches_flat_wire_bytes() {
        for seed in 0..100u64 {
            let referenced = (0..seed as usize * 137)
                .map(|index| index.wrapping_mul(31) as u8)
                .collect::<Vec<_>>();
            let mut expected = 1u16.to_le_bytes().to_vec();
            expected.extend_from_slice(&seed.to_le_bytes());
            expected.extend_from_slice(&(u64::MAX - seed).to_le_bytes());
            expected.extend_from_slice(&(seed * 97).to_le_bytes());
            expected.extend_from_slice(&referenced);
            assert_eq!(
                index_commit_checksum(seed, u64::MAX - seed, seed * 97, &referenced),
                checksum32(&expected)
            );
        }
    }

    #[test]
    fn incremental_checkpoint_segments_match_wire_and_refuse_incomplete_producers() {
        use checkpoint_writer::{write_checkpoint_segments, Segment};
        use std::borrow::Cow;
        use std::io::Cursor;
        for seed in 0..100u64 {
            let append = AppendSummary {
                commit: CommitPayload {
                    batch_id: seed + 1,
                    first_sequence_no: 1,
                    last_sequence_no: 1,
                    record_count: 1,
                    batch_checksum: 0,
                    committed_at: String::new(),
                },
                commit_sequence_no: seed + 2,
                records: Vec::new(),
                _commit_work: None,
            };
            let mut checkpoint = full_checkpoint_for(&append);
            checkpoint.text_segments[0].terms[0].term =
                "term\u{4e2d}".repeat(seed as usize % 7 + 1);
            let vector = &mut checkpoint.vector_segments[0];
            vector.dim = [1, 17, 1025][seed as usize % 3];
            vector.vectors[0].vector = (0..vector.dim)
                .map(|index| {
                    f32::from_bits(match index % 4 {
                        0 => 0x8000_0000,
                        1 => 1,
                        2 => 0x7f7f_ffff,
                        _ => (index as f32).to_bits(),
                    })
                })
                .collect();
            let expected = build_index_checkpoint_suffix(&checkpoint, 37).unwrap();
            for owned in [false, true] {
                let mut output = Cursor::new(vec![7; 37]);
                let text = &checkpoint.text_segments[0];
                let vector = &checkpoint.vector_segments[0];
                let segments = [
                    Ok(Segment::Text(if owned {
                        Cow::Owned(text.clone())
                    } else {
                        Cow::Borrowed(text)
                    })),
                    Ok(Segment::Vector(if owned {
                        Cow::Owned(vector.clone())
                    } else {
                        Cow::Borrowed(vector)
                    })),
                ];
                write_checkpoint_segments(
                    &mut output,
                    &checkpoint.metadata,
                    segments,
                    37,
                    |_, _| {},
                )
                .unwrap();
                assert_eq!(&output.get_ref()[..37], &[7; 37]);
                assert_eq!(&output.get_ref()[37..], &expected);
            }
            for failure in 0..6 {
                let mut output = Cursor::new(vec![7; 37]);
                let mut metadata = checkpoint.metadata.clone();
                let mut segments = vec![
                    Ok(Segment::Text(Cow::Borrowed(&checkpoint.text_segments[0]))),
                    Ok(Segment::Vector(Cow::Borrowed(
                        &checkpoint.vector_segments[0],
                    ))),
                ];
                match failure {
                    0..=2 => segments.insert(
                        failure,
                        Err(Error::new(
                            ErrorCode::IoError,
                            "injected segment source failure",
                        )),
                    ),
                    3 => {
                        segments.pop();
                    }
                    4 => segments.push(Ok(Segment::Vector(Cow::Borrowed(
                        &checkpoint.vector_segments[0],
                    )))),
                    _ => {
                        metadata.text_segment_count = 0;
                        metadata.vector_segment_count = 2;
                    }
                }
                let error =
                    write_checkpoint_segments(&mut output, &metadata, segments, 37, |_, _| {})
                        .unwrap_err();
                assert_eq!(
                    error.code(),
                    if failure <= 2 {
                        ErrorCode::IoError
                    } else {
                        ErrorCode::FileFormatError
                    }
                );
                assert_eq!(&output.get_ref()[..37], &[7; 37]);
                let len = output.get_ref().len() as u64;
                assert!(!matches!(
                    read_index_checkpoint_suffix(
                        &mut output,
                        37,
                        len,
                        37,
                        seed + 1,
                        seed + 2,
                        u64::MAX
                    ),
                    Ok(Ok(_))
                ));
            }
        }
    }

    #[test]
    fn record_backed_vector_segments_match_owned_wire_and_propagate_source_failures() {
        use std::borrow::Cow;
        for seed in 0..100u64 {
            let path = temp_path("record-backed-vector-checkpoint");
            let mut storage = Storage::create(&path).unwrap();
            let RecordPayload::CollectionCreate(mut collection) =
                collection_record(&catalog_id_string("col_", 1), "notes")
            else {
                panic!()
            };
            collection.embedding_dim = [1, 17, 1025][seed as usize % 3];
            collection.distance_metric = [
                DistanceMetric::Cosine,
                DistanceMetric::DotProduct,
                DistanceMetric::Euclidean,
            ][seed as usize % 3];
            collection.embedding_model =
                (seed % 2 == 0).then(|| "model\u{4e2d}".repeat(seed as usize % 7 + 1));
            collection.vectors_normalized = seed % 2 != 0;
            let embeddings = (0..seed % 8)
                .map(|index| Embedding {
                    embedding_id: catalog_id_string("emb_", u64::MAX - index),
                    chunk_id: catalog_id_string("chk_", u64::MAX - index),
                    collection_id: collection.collection_id.clone(),
                    chunk_external_id: format!("chunk-{index}"),
                    model_name: collection
                        .embedding_model
                        .clone()
                        .unwrap_or_else(|| "m".into()),
                    dim: collection.embedding_dim,
                    vector: (0..collection.embedding_dim)
                        .map(|position| {
                            f32::from_bits(match position % 4 {
                                0 => 0x8000_0000,
                                1 => 1,
                                2 => 0x7f7f_ffff,
                                _ => (position as f32).to_bits(),
                            })
                        })
                        .collect(),
                    created_at: "1970-01-01T00:00:00Z".into(),
                })
                .collect::<Vec<_>>();
            let records = std::iter::once(RecordPayload::CollectionCreate(collection.clone()))
                .chain(embeddings.iter().cloned().map(RecordPayload::EmbeddingPut))
                .collect::<Vec<_>>();
            let append = storage.append_batch(&records, Durability::Fast).unwrap();
            let mut checkpoint = full_checkpoint_for(&append);
            checkpoint.vector_segments = vec![VectorIndexSegmentPayload {
                collection_id: collection.collection_id.clone(),
                covered_commit_sequence_no: append.commit_sequence_no,
                dim: collection.embedding_dim,
                distance_metric: collection.distance_metric,
                embedding_model: collection.embedding_model.clone(),
                vectors_normalized: collection.vectors_normalized,
                vectors: embeddings
                    .iter()
                    .map(|entry| VectorIndexEntry {
                        chunk_id: entry.chunk_id.clone(),
                        embedding_id: entry.embedding_id.clone(),
                        vector: entry.vector.clone(),
                    })
                    .collect(),
            }];
            checkpoint.metadata.live_embedding_count = embeddings.len();
            let offset = storage.critical_end_offset;
            let prefix = fs::read(&path).unwrap();
            let expected = build_index_checkpoint_suffix(&checkpoint, offset).unwrap();
            let summaries = embeddings
                .iter()
                .cloned()
                .map(|mut entry| {
                    entry.vector.clear();
                    entry
                })
                .collect::<Vec<_>>();
            for (stored, cached) in [(false, false), (true, false), (false, true), (true, true)] {
                let context = crate::json::JsonOperationContext::standalone();
                let mut source = VectorRecordSegment::new(
                    &collection,
                    append.commit_sequence_no,
                    embeddings.iter().enumerate().map(|(index, entry)| {
                        if stored {
                            (
                                &summaries[index],
                                Some(&append.records[index + 1].record_ref),
                            )
                        } else {
                            (entry, None)
                        }
                    }),
                    storage.payload_reader().unwrap(),
                    &context,
                )
                .unwrap();
                assert_eq!(
                    source.encoded_len().unwrap() as usize,
                    INDEX_RECORD_HEADER_LEN
                        + encode_vector_index_segment_payload(&checkpoint.vector_segments[0])
                            .unwrap()
                            .len()
                );
                let segment = if cached {
                    let cached = source.try_materialize(1024 * 1024).unwrap().unwrap();
                    assert_eq!(cached.vectors.capacity(), embeddings.len());
                    assert!(cached.vectors.windows(2).all(|pair| {
                        catalog_id_number(&pair[0].chunk_id).unwrap()
                            < catalog_id_number(&pair[1].chunk_id).unwrap()
                    }));
                    CheckpointSegment::CachedVector(cached)
                } else {
                    CheckpointSegment::VectorRecords(source)
                };
                storage
                    .write_index_segments(
                        &checkpoint.metadata,
                        [
                            Ok(CheckpointSegment::Text(Cow::Borrowed(
                                &checkpoint.text_segments[0],
                            ))),
                            Ok(segment),
                        ],
                        |segment, _| {
                            if let CheckpointSegment::VectorRecords(source) = segment {
                                assert_eq!(source.encode_calls, 1);
                            }
                        },
                        Durability::Fast,
                    )
                    .unwrap();
                assert!(context
                    .cache_reservation(1024 * 1024, 1024 * 1024)
                    .is_some());
                let bytes = fs::read(&path).unwrap();
                assert_eq!(&bytes[..offset as usize], &prefix);
                assert_eq!(&bytes[offset as usize..], &expected);
            }
            if let Some(summary) = summaries.last() {
                for invalid in [false, true] {
                    let mut duplicate = summary.clone();
                    if invalid {
                        duplicate.chunk_id = "invalid".into();
                    }
                    let limit = 2 * std::mem::size_of::<(&Embedding, Option<&RecordRef>)>();
                    let context = crate::json::JsonOperationContext::new(std::sync::Arc::new(
                        crate::json::WorkLedger::with_limits(limit, limit, limit),
                    ));
                    assert!(matches!(VectorRecordSegment::new(
                        &collection, append.commit_sequence_no,
                        [(summary, None), (&duplicate, None)],
                        storage.payload_reader().unwrap(), &context,
                    ), Err(error) if error.code() == ErrorCode::FileFormatError));
                    context
                        .workspace_reservation()
                        .grow_to(limit, limit, "operation_scratch")
                        .unwrap();
                }
                for failure in 0..3 {
                    let mut bad = append.records.last().unwrap().record_ref.clone();
                    let mut wrong_summary = summary.clone();
                    let expected_error = match failure {
                        0 => {
                            bad.payload.checksum ^= 1;
                            ErrorCode::ChecksumMismatch
                        }
                        1 => {
                            wrong_summary.chunk_id = "chk_1".into();
                            ErrorCode::FileFormatError
                        }
                        _ => {
                            bad = append.records[0].record_ref.clone();
                            ErrorCode::FileFormatError
                        }
                    };
                    let context = crate::json::JsonOperationContext::standalone();
                    let source = VectorRecordSegment::new(
                        &collection,
                        append.commit_sequence_no,
                        [(&wrong_summary, Some(&bad))],
                        storage.payload_reader().unwrap(),
                        &context,
                    )
                    .unwrap();
                    let mut cached_source = VectorRecordSegment::new(
                        &collection,
                        append.commit_sequence_no,
                        [(&wrong_summary, Some(&bad))],
                        storage.payload_reader().unwrap(),
                        &context,
                    )
                    .unwrap();
                    assert_eq!(
                        cached_source
                            .try_materialize(1024 * 1024)
                            .unwrap_err()
                            .code(),
                        expected_error
                    );
                    assert!(context
                        .cache_reservation(1024 * 1024, 1024 * 1024)
                        .is_some());
                    let error = storage
                        .write_index_segments(
                            &checkpoint.metadata,
                            [
                                Ok(CheckpointSegment::Text(Cow::Borrowed(
                                    &checkpoint.text_segments[0],
                                ))),
                                Ok(CheckpointSegment::VectorRecords(source)),
                            ],
                            |_, _| {},
                            Durability::Fast,
                        )
                        .unwrap_err();
                    assert_eq!(error.code(), expected_error);
                    assert_eq!(&fs::read(&path).unwrap()[..offset as usize], &prefix);
                    assert!(scan_file(&path).unwrap().index_checkpoint.is_none());
                }
            }
            drop(storage);
            fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn vector_cache_materialization_cannot_bypass_shared_decode_workspace() {
        use crate::json::{JsonOperationContext, WorkLedger};
        use std::sync::Arc;

        let path = temp_path("vector-cache-decode-workspace");
        let mut storage = Storage::create(&path).unwrap();
        let RecordPayload::CollectionCreate(collection) =
            collection_record(&catalog_id_string("col_", 1), "notes")
        else {
            panic!()
        };
        let embedding = Embedding {
            embedding_id: catalog_id_string("emb_", 1),
            chunk_id: catalog_id_string("chk_", 1),
            collection_id: collection.collection_id.clone(),
            chunk_external_id: "chunk".into(),
            model_name: "m".repeat(512),
            dim: collection.embedding_dim,
            vector: vec![0.25; collection.embedding_dim],
            created_at: "1970-01-01T00:00:00Z".into(),
        };
        let append = storage
            .append_batch(
                &[
                    RecordPayload::CollectionCreate(collection.clone()),
                    RecordPayload::EmbeddingPut(embedding.clone()),
                ],
                Durability::Fast,
            )
            .unwrap();
        let context = JsonOperationContext::new(Arc::new(WorkLedger::with_limits(64, 4096, 4096)));
        let mut source = VectorRecordSegment::new(
            &collection,
            append.commit_sequence_no,
            [(&embedding, Some(&append.records[1].record_ref))],
            storage.payload_reader().unwrap(),
            &context,
        )
        .unwrap();
        let result = source.try_materialize(4096).unwrap();
        assert!(
            result.is_none(),
            "cached decoding must use the caller's shared workspace"
        );
        drop(result);
        drop(source);
        assert!(context.cache_reservation(4096, 4096).is_some());
        drop(storage);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn record_backed_vector_scratch_is_reserved_and_released_at_each_boundary() {
        use crate::json::{JsonOperationContext, WorkLedger};
        use std::sync::Arc;

        for seed in 0..100usize {
            let path = temp_path("vector-checkpoint-scratch");
            let mut storage = Storage::create(&path).unwrap();
            let RecordPayload::CollectionCreate(mut collection) =
                collection_record(&catalog_id_string("col_", 1), "notes")
            else {
                panic!()
            };
            collection.embedding_dim = 3 + seed;
            let embeddings = (0..1 + seed % 7)
                .rev()
                .map(|index| Embedding {
                    embedding_id: catalog_id_string("emb_", index as u64 + 1),
                    chunk_id: catalog_id_string("chk_", index as u64 + 1),
                    collection_id: collection.collection_id.clone(),
                    chunk_external_id: format!("chunk-{index}"),
                    model_name: "model".repeat(1 + seed),
                    dim: collection.embedding_dim,
                    vector: vec![0.25; collection.embedding_dim],
                    created_at: "1970-01-01T00:00:00Z".into(),
                })
                .collect::<Vec<_>>();
            let records = std::iter::once(RecordPayload::CollectionCreate(collection.clone()))
                .chain(embeddings.iter().cloned().map(RecordPayload::EmbeddingPut))
                .collect::<Vec<_>>();
            let append = storage.append_batch(&records, Durability::Fast).unwrap();
            let mut checkpoint = full_checkpoint_for(&append);
            checkpoint.metadata.text_segment_count = 0;
            checkpoint.metadata.live_embedding_count = embeddings.len();
            checkpoint.text_segments.clear();
            checkpoint.vector_segments[0] = VectorIndexSegmentPayload {
                collection_id: collection.collection_id.clone(),
                covered_commit_sequence_no: append.commit_sequence_no,
                dim: collection.embedding_dim,
                distance_metric: collection.distance_metric,
                embedding_model: collection.embedding_model.clone(),
                vectors_normalized: collection.vectors_normalized,
                vectors: embeddings
                    .iter()
                    .map(|entry| VectorIndexEntry {
                        chunk_id: entry.chunk_id.clone(),
                        embedding_id: entry.embedding_id.clone(),
                        vector: entry.vector.clone(),
                    })
                    .collect(),
            };
            let end = storage.critical_end_offset;
            let primary = fs::read(&path).unwrap();
            let expected = build_index_checkpoint_suffix(&checkpoint, end).unwrap();
            let references =
                embeddings.len() * std::mem::size_of::<(&Embedding, Option<&RecordRef>)>();
            let retained = storage
                .payload_reader()
                .unwrap()
                .read_embedding_in(
                    &append.records[1].record_ref,
                    &JsonOperationContext::standalone(),
                )
                .unwrap();
            let (decoded, work) = retained.into_parts();
            let payload_bytes = work.reserved_bytes();
            drop(decoded);
            drop(work);

            for stored in [false, true] {
                let required = references + if stored { payload_bytes } else { 0 };
                for extra in [0, 1, 2] {
                    let limit = required - 1 + extra;
                    let context = JsonOperationContext::new(Arc::new(WorkLedger::with_limits(
                        limit, limit, limit,
                    )));
                    let source = VectorRecordSegment::new(
                        &collection,
                        append.commit_sequence_no,
                        embeddings.iter().enumerate().map(|(index, embedding)| {
                            (
                                embedding,
                                stored.then_some(&append.records[index + 1].record_ref),
                            )
                        }),
                        storage.payload_reader().unwrap(),
                        &context,
                    );
                    let result = source.and_then(|source| {
                        storage.write_index_segments(
                            &checkpoint.metadata,
                            [Ok(CheckpointSegment::VectorRecords(source))],
                            |_, _| {},
                            Durability::Fast,
                        )
                    });
                    if extra == 0 {
                        let error = result.unwrap_err();
                        assert_eq!(
                            error.code(),
                            ErrorCode::ResourceExhausted,
                            "seed={seed}, stored={stored}"
                        );
                        assert_eq!(
                            error
                                .detail()
                                .iter()
                                .find(|(key, _)| key == "limit_bytes")
                                .map(|(_, value)| value.as_str()),
                            Some(limit.to_string().as_str())
                        );
                        assert_eq!(
                            error
                                .detail()
                                .iter()
                                .find(|(key, _)| key == "requested_bytes")
                                .map(|(_, value)| value.as_str()),
                            Some(required.to_string().as_str())
                        );
                    } else {
                        result.unwrap();
                        assert_eq!(&fs::read(&path).unwrap()[end as usize..], &expected);
                    }
                    assert_eq!(&fs::read(&path).unwrap()[..end as usize], &primary);
                    let mut all = context.workspace_reservation();
                    all.grow_to(limit, limit, "operation_scratch").unwrap();
                }
                let copied_bytes = if stored {
                    payload_bytes
                } else {
                    let (copy, work) = RetainedEmbedding::copy(
                        &embeddings[0],
                        &JsonOperationContext::standalone(),
                    )
                    .unwrap()
                    .into_parts();
                    let bytes = work.reserved_bytes();
                    drop(copy);
                    drop(work);
                    bytes
                };
                let workspace = references + copied_bytes;
                let total = 1024 * 1024;
                let context = JsonOperationContext::new(Arc::new(WorkLedger::with_limits(
                    workspace, total, total,
                )));
                let mut source = VectorRecordSegment::new(
                    &collection,
                    append.commit_sequence_no,
                    embeddings.iter().enumerate().map(|(index, embedding)| {
                        (
                            embedding,
                            stored.then_some(&append.records[index + 1].record_ref),
                        )
                    }),
                    storage.payload_reader().unwrap(),
                    &context,
                )
                .unwrap();
                let cached = source.try_materialize(total).unwrap().unwrap();
                assert_eq!(cached.vectors.len(), embeddings.len());
                let cache_bytes = cached.reserved_bytes();
                if embeddings.len() == 7 {
                    assert!(cache_bytes > workspace);
                }
                drop(cached);
                assert!(source.try_materialize(cache_bytes - 1).unwrap().is_none());
                assert!(context
                    .cache_reservation((total - references) as u128, total)
                    .is_some());
                storage
                    .write_index_segments(
                        &checkpoint.metadata,
                        [Ok(CheckpointSegment::VectorRecords(source))],
                        |_, _| {},
                        Durability::Fast,
                    )
                    .unwrap();
                assert_eq!(&fs::read(&path).unwrap()[end as usize..], &expected);
                assert!(context.cache_reservation(total as u128, total).is_some());
            }
            drop(storage);
            fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn checkpoint_producer_and_completion_may_move_shared_file_cursor() {
        use checkpoint_writer::{reserve_directory, write_checkpoint_segments_in, Segment};
        use std::borrow::Cow;
        use std::cell::{Cell, RefCell};
        use std::io::Cursor;
        use std::rc::Rc;

        #[derive(Clone)]
        struct Shared(Rc<RefCell<Cursor<Vec<u8>>>>);
        impl Read for Shared {
            fn read(&mut self, bytes: &mut [u8]) -> std::io::Result<usize> {
                self.0.borrow_mut().read(bytes)
            }
        }
        impl Write for Shared {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                self.0.borrow_mut().write(bytes)
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        impl Seek for Shared {
            fn seek(&mut self, position: SeekFrom) -> std::io::Result<u64> {
                self.0.borrow_mut().seek(position)
            }
        }
        for (seed, allow_optional_memory) in (0..100u64).flat_map(|seed| {
            [false, true]
                .into_iter()
                .map(move |buffered| (seed, buffered))
        }) {
            let append = AppendSummary {
                commit: CommitPayload {
                    batch_id: seed + 1,
                    first_sequence_no: 1,
                    last_sequence_no: 1,
                    record_count: 1,
                    batch_checksum: 0,
                    committed_at: String::new(),
                },
                commit_sequence_no: seed + 2,
                records: Vec::new(),
                _commit_work: None,
            };
            let checkpoint = full_checkpoint_for(&append);
            let expected = build_index_checkpoint_suffix(&checkpoint, 37).unwrap();
            let mut output = Shared(Rc::new(RefCell::new(Cursor::new(vec![7; 37]))));
            let mut input = output.clone();
            let mut observer = output.clone();
            let completed = Cell::new(0);
            let mut calls = 0;
            let owned = checkpoint.vector_segments[0].clone();
            let pointer = owned.vectors[0].vector.as_ptr();
            let mut segments = [
                Ok(Segment::Text(Cow::Borrowed(&checkpoint.text_segments[0]))),
                Ok(Segment::Vector(Cow::Owned(owned))),
            ]
            .into_iter();
            let producer = std::iter::from_fn(|| {
                assert_eq!(completed.get(), calls);
                calls += 1;
                input.seek(SeekFrom::Start(seed % 37)).unwrap();
                let mut byte = [0];
                input.read_exact(&mut byte).unwrap();
                assert_eq!(byte, [7]);
                segments.next()
            });
            let mut retained = None;
            let context = crate::json::JsonOperationContext::standalone();
            let directory = reserve_directory(&checkpoint.metadata, &context).unwrap();
            write_checkpoint_segments_in(
                &mut output,
                &checkpoint.metadata,
                producer,
                37,
                |segment, len| {
                    assert!(len > INDEX_RECORD_HEADER_LEN as u64);
                    completed.set(completed.get() + 1);
                    observer.seek(SeekFrom::Start(0)).unwrap();
                    match segment {
                        Segment::Text(Cow::Borrowed(_)) => {}
                        Segment::Vector(Cow::Owned(segment)) => {
                            assert_eq!(segment.vectors[0].vector.as_ptr(), pointer);
                            retained = Some(segment);
                        }
                        _ => panic!("segment ownership changed"),
                    }
                },
                directory,
                allow_optional_memory,
            )
            .unwrap();
            assert_eq!(calls, 3);
            assert_eq!(completed.get(), 2);
            assert_eq!(retained.unwrap().vectors[0].vector.as_ptr(), pointer);
            let bytes = output.0.borrow();
            assert_eq!(&bytes.get_ref()[..37], &[7; 37]);
            assert_eq!(&bytes.get_ref()[37..], &expected);
        }
    }

    #[test]
    fn index_checkpoint_assembly_matches_flat_wire_oracle() {
        for seed in 0..100u64 {
            let count = seed as usize % 5;
            let checkpoint = IndexCheckpointPayload {
                metadata: IndexCheckpointMetadata {
                    covered_last_batch_id: seed,
                    covered_last_commit_sequence_no: seed * 97,
                    coverage_hash: seed as u32 * 31,
                    collection_count: count,
                    live_document_count: 0,
                    live_chunk_count: 0,
                    live_embedding_count: 0,
                    text_segment_count: count,
                    vector_segment_count: count,
                    build_mode: "inline_full_checkpoint".into(),
                },
                text_segments: (1..=count)
                    .map(|id| TextIndexSegmentPayload {
                        collection_id: format!("col_{id}"),
                        covered_commit_sequence_no: seed * 97,
                        chunks: Vec::new(),
                        terms: Vec::new(),
                    })
                    .collect(),
                vector_segments: (1..=count)
                    .map(|id| VectorIndexSegmentPayload {
                        collection_id: format!("col_{id}"),
                        covered_commit_sequence_no: seed * 97,
                        dim: 768,
                        distance_metric: DistanceMetric::Cosine,
                        embedding_model: Some("m".into()),
                        vectors_normalized: false,
                        vectors: Vec::new(),
                    })
                    .collect(),
            };
            let generation = u64::MAX - seed;
            let records = build_index_segment_records(&checkpoint, generation).unwrap();
            let mut extents = Vec::new();
            let mut referenced = Vec::new();
            let mut offset = (INDEX_CHECKPOINT_HEADER_LEN
                + INDEX_RECORD_HEADER_LEN
                + INDEX_COMMIT_METADATA_LEN) as u64;
            for record in &records {
                let extent = IndexExtentEntry {
                    index_record_id: record.index_record_id,
                    record_type: record.record_type,
                    stream_id: record.stream_id,
                    suffix_relative_offset: offset,
                    length: record.bytes.len() as u64,
                    checkpoint_generation: generation,
                    extent_record_checksum: checksum32(&record.bytes),
                };
                referenced.extend_from_slice(&encode_extent_entry(&extent));
                referenced.extend_from_slice(&record.bytes);
                offset += record.bytes.len() as u64;
                extents.push(extent);
            }
            let table = encode_extent_table(&extents);
            let metadata = build_index_commit_metadata_payload(
                &checkpoint.metadata,
                records.len() as u64,
                INDEX_EXTENT_ENTRY_LEN as u64,
                table.len() as u64,
                checksum32(&table),
                index_commit_checksum(seed, seed * 97, generation, &referenced),
            );
            let mut payload = build_index_record(INDEX_COMMIT_METADATA, generation, 1, &metadata);
            extents.insert(
                0,
                IndexExtentEntry {
                    index_record_id: 1,
                    record_type: INDEX_COMMIT_METADATA,
                    stream_id: 0,
                    suffix_relative_offset: INDEX_CHECKPOINT_HEADER_LEN as u64,
                    length: payload.len() as u64,
                    checkpoint_generation: generation,
                    extent_record_checksum: checksum32(&payload),
                },
            );
            for record in records {
                payload.extend_from_slice(&record.bytes);
            }
            let expected = checkpoint_suffix_bytes(
                &checkpoint,
                generation,
                generation,
                extents.len() as u64,
                &payload,
                &encode_extent_table(&extents),
            );
            let actual = build_index_checkpoint_suffix(&checkpoint, generation).unwrap();
            assert_eq!(actual, expected);
            let mut streamed = std::io::Cursor::new(vec![7; 37]);
            let (payload_len, written_vectors) =
                checkpoint_writer::write_checkpoint(&mut streamed, &checkpoint, 37).unwrap();
            let expected = build_index_checkpoint_suffix(&checkpoint, 37).unwrap();
            assert_eq!(&streamed.get_ref()[..37], &[7; 37]);
            assert_eq!(&streamed.get_ref()[37..], expected);
            assert_eq!(
                payload_len,
                expected.len() as u64
                    - (INDEX_CHECKPOINT_HEADER_LEN
                        + INDEX_CHECKPOINT_TRAILER_LEN
                        + extents.len() * INDEX_EXTENT_ENTRY_LEN) as u64
            );
            let file_len = streamed.get_ref().len() as u64;
            let parsed = read_index_checkpoint_suffix(
                &mut streamed,
                37,
                file_len,
                37,
                seed,
                seed * 97,
                u64::MAX,
            )
            .unwrap()
            .unwrap();
            assert!(parsed.diagnostics.is_empty());
            assert_eq!(parsed.report.collection_count, count);
            assert_eq!(
                build_index_checkpoint_suffix(parsed.checkpoint.as_ref().unwrap(), 37).unwrap(),
                expected
            );
            let summary = read_index_checkpoint_suffix(
                &mut streamed,
                37,
                file_len,
                37,
                seed,
                seed * 97,
                IndexReadOptions {
                    budget_bytes: u64::MAX,
                    mode: IndexReadMode::Summary,
                },
            )
            .unwrap()
            .unwrap();
            assert!(summary.checkpoint.is_none());
            assert!(summary.diagnostics.is_empty());
            assert_eq!(summary.report.to_json(), parsed.report.to_json());
            for segment in &checkpoint.vector_segments {
                let loaded = written_vectors
                    .read_from(&mut streamed, &segment.collection_id)
                    .unwrap()
                    .unwrap();
                assert_eq!(
                    encode_vector_index_segment_payload(&loaded).unwrap(),
                    encode_vector_index_segment_payload(segment).unwrap()
                );
            }
        }
    }

    #[test]
    fn checkpoint_write_time_checksums_match_reference_for_populated_segments() {
        let mut checkpoint = full_checkpoint_for(&AppendSummary {
            commit: CommitPayload {
                batch_id: 1,
                first_sequence_no: 1,
                last_sequence_no: 3,
                record_count: 3,
                batch_checksum: 0,
                committed_at: "1970-01-01T00:00:00Z".into(),
            },
            commit_sequence_no: 4,
            records: Vec::new(),
            _commit_work: None,
        });
        for seed in 0..100 {
            let dim = [1, 17, 768, 1025][seed % 4];
            checkpoint.vector_segments[0].dim = dim;
            checkpoint.vector_segments[0].vectors[0].vector = (0..dim)
                .map(|index| (seed * 19 + index * 31) as f32 / 1024.0)
                .collect();
            checkpoint.text_segments[0].terms[0].term = "t".repeat(1 + seed * 733);
            let offset = (seed * 37) as u64;
            let expected = build_index_checkpoint_suffix(&checkpoint, offset).unwrap();
            let mut output = std::io::Cursor::new(vec![7; offset as usize]);
            checkpoint_writer::write_checkpoint(&mut output, &checkpoint, offset).unwrap();
            assert_eq!(
                &output.get_ref()[..offset as usize],
                vec![7; offset as usize]
            );
            assert_eq!(
                &output.get_ref()[offset as usize..],
                expected,
                "seed {seed}"
            );
            let file_len = output.get_ref().len() as u64;
            let parsed =
                read_index_checkpoint_suffix(&mut output, offset, file_len, offset, 1, 4, u64::MAX)
                    .unwrap()
                    .unwrap();
            assert!(parsed.diagnostics.is_empty(), "seed {seed}");
            assert_eq!(
                build_index_checkpoint_suffix(parsed.checkpoint.as_ref().unwrap(), offset).unwrap(),
                expected
            );
        }
    }

    #[test]
    fn streaming_index_checkpoint_bounds_io_and_preserves_prefix_on_every_failure() {
        struct Stream {
            bytes: std::io::Cursor<Vec<u8>>,
            writes: usize,
            written: usize,
            seeks: usize,
            flushes: usize,
            write_limit: usize,
            fail_seek: usize,
            fail_flush: usize,
        }
        impl Default for Stream {
            fn default() -> Self {
                Self {
                    bytes: std::io::Cursor::new(vec![7; 37]),
                    writes: 0,
                    written: 0,
                    seeks: 0,
                    flushes: 0,
                    write_limit: usize::MAX,
                    fail_seek: 0,
                    fail_flush: 0,
                }
            }
        }
        impl Write for Stream {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                assert!(bytes.len() <= 64 * 1024);
                self.writes += 1;
                if self.written == self.write_limit {
                    return Err(std::io::Error::other("injected checkpoint write failure"));
                }
                if self.writes % 7 == 0 {
                    return Err(std::io::ErrorKind::Interrupted.into());
                }
                let count = bytes.len().min(4093).min(self.write_limit - self.written);
                let count = self.bytes.write(&bytes[..count])?;
                self.written += count;
                Ok(count)
            }
            fn flush(&mut self) -> std::io::Result<()> {
                self.flushes += 1;
                if self.flushes == self.fail_flush {
                    return Err(std::io::Error::other("injected checkpoint flush failure"));
                }
                Ok(())
            }
        }
        // Deliberately no Read implementation: writing cannot read back its output.
        impl Seek for Stream {
            fn seek(&mut self, position: SeekFrom) -> std::io::Result<u64> {
                self.seeks += 1;
                if self.seeks == self.fail_seek {
                    return Err(std::io::Error::other("injected checkpoint seek failure"));
                }
                self.bytes.seek(position)
            }
        }
        let mut checkpoint = full_checkpoint_for(&AppendSummary {
            commit: CommitPayload {
                batch_id: 1,
                first_sequence_no: 1,
                last_sequence_no: 3,
                record_count: 3,
                batch_checksum: 0,
                committed_at: "1970-01-01T00:00:00Z".into(),
            },
            commit_sequence_no: 4,
            records: Vec::new(),
            _commit_work: None,
        });
        checkpoint.metadata.live_chunk_count = 128;
        checkpoint.metadata.live_embedding_count = 128;
        checkpoint.text_segments[0].terms[0].term = "w".repeat(64 * 1024 + 17);
        checkpoint.text_segments[0].chunks = (1..=128)
            .map(|id| TextIndexChunkEntry {
                chunk_id: format!("chk_{id}"),
                document_id: "doc_1".into(),
                token_count: 2,
            })
            .collect();
        checkpoint.vector_segments[0].dim = 768;
        checkpoint.vector_segments[0].vectors = (1..=128)
            .map(|id| VectorIndexEntry {
                chunk_id: format!("chk_{id}"),
                embedding_id: format!("emb_{id}"),
                vector: vec![1.0; 768],
            })
            .collect();
        let expected = build_index_checkpoint_suffix(&checkpoint, 37).unwrap();
        let mut success = Stream::default();
        checkpoint_writer::write_checkpoint(&mut success, &checkpoint, 37).unwrap();
        assert_eq!(&success.bytes.get_ref()[..37], &[7; 37]);
        assert_eq!(&success.bytes.get_ref()[37..], expected);
        let total_written = success.written;
        let write_limits = (0..100).map(|seed| seed * total_written / 100).chain([
            1,
            104,
            105,
            total_written - 15,
            total_written - 14,
            total_written - 1,
        ]);
        for limit in write_limits {
            let mut stream = Stream {
                write_limit: limit,
                ..Stream::default()
            };
            let error =
                checkpoint_writer::write_checkpoint(&mut stream, &checkpoint, 37).unwrap_err();
            assert_eq!(error.code(), ErrorCode::IoError);
            assert_eq!(&stream.bytes.get_ref()[..37], &[7; 37]);
            assert!(!stream
                .bytes
                .get_ref()
                .ends_with(&expected[expected.len() - INDEX_CHECKPOINT_TRAILER_LEN..]));
        }
        for seek in 1..=success.seeks {
            let mut stream = Stream {
                fail_seek: seek,
                ..Stream::default()
            };
            assert_eq!(
                checkpoint_writer::write_checkpoint(&mut stream, &checkpoint, 37)
                    .unwrap_err()
                    .code(),
                ErrorCode::IoError
            );
            assert_eq!(&stream.bytes.get_ref()[..37], &[7; 37]);
        }
        for flush in 1..=success.flushes {
            let mut stream = Stream {
                fail_flush: flush,
                ..Stream::default()
            };
            assert_eq!(
                checkpoint_writer::write_checkpoint(&mut stream, &checkpoint, 37)
                    .unwrap_err()
                    .code(),
                ErrorCode::IoError
            );
            assert_eq!(&stream.bytes.get_ref()[..37], &[7; 37]);
        }
    }

    #[test]
    fn optional_index_strings_validate_and_match_across_utf8_chunk_boundaries() {
        fn encoded(bytes: &[u8]) -> Vec<u8> {
            let mut result = vec![1];
            result.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
            result.extend_from_slice(bytes);
            result
        }
        fn read(
            bytes: &[u8],
            mode: IndexStringMode<'_>,
        ) -> std::result::Result<Option<String>, IndexReadError> {
            let mut reader = IndexPayloadReader::new(bytes, bytes.len(), 0);
            let result = reader.read_optional_string("model", mode)?;
            reader.finish_before_checksum("model")?;
            Ok(result)
        }
        for seed in 0..100 {
            let value = format!(
                "{}{}",
                "x".repeat(4092 + seed % 9),
                "\u{a2}\u{4e2d}\u{1f642}\0".repeat(900 + seed)
            );
            let bytes = encoded(value.as_bytes());
            assert_eq!(
                read(&bytes, IndexStringMode::Retain).unwrap().as_deref(),
                Some(value.as_str())
            );
            assert!(read(&bytes, IndexStringMode::Validate).unwrap().is_none());
            assert!(read(&bytes, IndexStringMode::Match(Some(&value)))
                .unwrap()
                .is_none());
            let mut different = value.clone();
            different.replace_range(0..1, "y");
            for expected in [None, Some(""), Some(different.as_str())] {
                assert!(matches!(
                    read(&bytes, IndexStringMode::Match(expected)),
                    Err(IndexReadError::Invalid(_))
                ));
            }
            for end in [0, 1, 4, 5, 4096, bytes.len() - 1] {
                let mut reader = IndexPayloadReader::new(&bytes[..end], bytes.len(), 0);
                assert!(
                    matches!(reader.read_optional_string("model", IndexStringMode::Validate), Err(IndexReadError::Io(error)) if error.kind() == std::io::ErrorKind::UnexpectedEof)
                );
            }
            for invalid in [
                &[0x80][..],
                &[0xc0, 0xaf],
                &[0xed, 0xa0, 0x80],
                &[0xf4, 0x90, 0x80, 0x80],
                &[0xe2, 0x82],
                &[0xff],
            ] {
                let mut value = vec![b'x'; 4092 + seed % 9];
                value.extend_from_slice(invalid);
                let bytes = encoded(&value);
                for mode in [IndexStringMode::Retain, IndexStringMode::Validate] {
                    assert!(
                        matches!(read(&bytes, mode), Err(IndexReadError::Invalid(_))),
                        "seed {seed}, invalid {invalid:?}"
                    );
                }
            }
        }
        for mode in [
            IndexStringMode::Retain,
            IndexStringMode::Validate,
            IndexStringMode::Match(None),
        ] {
            assert!(read(&[0], mode).unwrap().is_none());
        }
        assert!(matches!(
            read(&[0], IndexStringMode::Match(Some(""))),
            Err(IndexReadError::Invalid(_))
        ));
        assert!(read(&encoded(b""), IndexStringMode::Match(Some("")))
            .unwrap()
            .is_none());
        assert!(matches!(
            read(&encoded(b""), IndexStringMode::Match(None)),
            Err(IndexReadError::Invalid(_))
        ));
        for mode in [
            IndexStringMode::Retain,
            IndexStringMode::Validate,
            IndexStringMode::Match(Some("")),
        ] {
            assert!(matches!(read(&[2], mode), Err(IndexReadError::Invalid(_))));
        }
        for mode in [IndexStringMode::Retain, IndexStringMode::Validate] {
            let mut bytes = vec![1];
            bytes.extend_from_slice(&u32::MAX.to_le_bytes());
            assert!(matches!(
                read(&bytes, mode),
                Err(IndexReadError::Invalid(_))
            ));
        }
    }

    fn checked_payload_record(record_type: u16, payload: &[u8]) -> ParsedIndexRecord {
        ParsedIndexRecord {
            record_type,
            index_record_id: if record_type == INDEX_COMMIT_METADATA {
                1
            } else {
                2
            },
            stream_id: 1,
            offset: 0,
            length: (INDEX_RECORD_HEADER_LEN + payload.len()) as u64,
            payload: PayloadRef {
                offset: 37,
                len: payload.len() as u64,
                checksum: checksum32(payload) as u64,
            },
            segment_checksum_valid: true,
        }
    }

    #[test]
    fn checked_metadata_payload_does_not_allocate_a_read_buffer() {
        for seed in 0..100 {
            let checkpoint = empty_directory_checkpoint(seed, 0);
            let payload = build_index_commit_metadata_payload(
                &checkpoint.metadata,
                0,
                0,
                0,
                checksum32(&[]),
                seed as u32,
            );
            let record = checked_payload_record(INDEX_COMMIT_METADATA, &payload);
            let mut bytes = vec![7; 37];
            bytes.extend_from_slice(&payload);
            bytes.extend_from_slice(&[9; 11]);
            let records = BTreeMap::new();
            let context = crate::json::JsonOperationContext::new(std::sync::Arc::new(
                crate::json::WorkLedger::with_limits(0, 0, 0),
            ));
            let (parsed, allocated) = crate::test_alloc::allocation_bytes(|| {
                with_checked_index_payload(
                    &mut std::io::Cursor::new(&bytes),
                    &record,
                    &context,
                    |source| {
                        let mut decoded = [0; INDEX_COMMIT_METADATA_LEN];
                        source.read_exact(&mut decoded)?;
                        Ok(parse_index_commit_metadata_payload(
                            &decoded,
                            checkpoint.metadata.covered_last_batch_id,
                            checkpoint.metadata.covered_last_commit_sequence_no,
                            &[],
                            &records,
                            0,
                        )?)
                    },
                )
            });
            let parsed = parsed.unwrap();
            assert_eq!(parsed.index_record_count, 0);
            assert_eq!(parsed.index_checksum, seed as u32);
            assert_eq!(parsed.extent_range, 0..0);
            assert_eq!(allocated, 0);
        }
    }

    #[test]
    fn checked_index_read_ahead_obeys_working_and_parent_credit_and_releases_its_buffer() {
        use crate::json::{JsonOperationContext, WorkLedger};
        use std::sync::Arc;

        for len in [0usize, 1, 65_535, 65_536, 65_537, 131_109] {
            let payload: Vec<_> = (0..len).map(|byte| byte as u8).collect();
            let mut bytes = vec![7; 37];
            bytes.extend_from_slice(&payload);
            bytes.extend_from_slice(&[9; 11]);
            let preferred = len.min(64 * 1024);
            for kind in [INDEX_TEXT_SEGMENT, INDEX_VECTOR_SEGMENT] {
                let record = checked_payload_record(kind, &payload);
                for parent in [false, true] {
                    for available in [0, 1, preferred.saturating_sub(1), preferred, preferred + 1] {
                        let working = if parent { preferred + 19 } else { available };
                        let total = if parent {
                            available + 7
                        } else {
                            preferred + 26
                        };
                        let context = JsonOperationContext::new(Arc::new(WorkLedger::with_limits(
                            working, 0, total,
                        )));
                        let blocker = context.cache_reservation(7, usize::MAX).unwrap();
                        context
                            .workspace_reservation()
                            .grow_to(0, 0, "operation_scratch")
                            .unwrap();
                        let granted = available.min(preferred);
                        let mut input = DirectoryInput::new(&bytes);
                        let mut calls = 0;
                        let (result, allocated) = crate::test_alloc::allocation_bytes(|| {
                            with_checked_index_payload(&mut input, &record, &context, |source| {
                                calls += 1;
                                assert_eq!(
                                    context.available_workspace_bytes(),
                                    available - granted
                                );
                                let mut checksum = Crc32c::default();
                                let mut buffer = [0; 4093];
                                let mut remaining = len;
                                while remaining > 0 {
                                    let count = remaining.min(buffer.len());
                                    source.read_exact(&mut buffer[..count])?;
                                    checksum.update(&buffer[..count]);
                                    remaining -= count;
                                }
                                Ok(checksum.finish())
                            })
                        });
                        assert_eq!(result.unwrap(), checksum32(&payload));
                        assert_eq!(calls, 1);
                        assert_eq!(input.bytes.position(), (37 + len) as u64);
                        assert_eq!(allocated, granted as u128);
                        let mut remaining = context.workspace_reservation();
                        remaining
                            .grow_to(available, available, "operation_scratch")
                            .unwrap();
                        assert!(remaining
                            .grow_to(available + 1, available + 1, "operation_scratch")
                            .is_err());
                        drop(blocker);
                        let complete = if parent { available + 7 } else { available };
                        remaining
                            .grow_to(complete, complete, "operation_scratch")
                            .unwrap();
                        assert!(remaining
                            .grow_to(complete + 1, complete + 1, "operation_scratch")
                            .is_err());
                    }
                }
            }
        }
    }

    #[test]
    fn checked_text_payload_yields_read_ahead_to_necessary_decode_and_retains_only_document_ids() {
        use crate::json::{JsonOperationContext, WorkLedger};
        use std::sync::Arc;

        for seed in 0..100 {
            let segment = summary_term_fixture(seed);
            let payload = encode_text_index_segment_payload(&segment).unwrap();
            let ids = segment.chunks.len() * 16;
            let terms = segment.terms[0].term.len() + segment.terms[1].term.len();
            let needed = ids + terms;
            let preferred = payload.len().min(64 * 1024);
            let record = checked_payload_record(INDEX_TEXT_SEGMENT, &payload);
            let mut bytes = vec![7; 37];
            bytes.extend_from_slice(&payload);
            for available in [needed - 1, needed, needed + 1, needed + preferred] {
                let working = needed + preferred + 31;
                let context = JsonOperationContext::new(Arc::new(WorkLedger::with_limits(
                    working,
                    0,
                    available + 7,
                )));
                let blocker = context.cache_reservation(7, usize::MAX).unwrap();
                let mut input = DirectoryInput::new(&bytes);
                let mut calls = 0;
                let result = with_checked_index_payload(&mut input, &record, &context, |source| {
                    calls += 1;
                    decode_text_index_segment_in(
                        &mut IndexPayloadReader::new(source, payload.len() - 4, 0),
                        11,
                        0,
                        false,
                        &context,
                    )
                });
                assert_eq!(
                    calls,
                    if available == needed + preferred {
                        1
                    } else {
                        2
                    }
                );
                assert_eq!(input.bytes.position(), bytes.len() as u64);
                let mut remaining = context.workspace_reservation();
                if available < needed {
                    let Err(IndexReadError::Resource(error)) = result else {
                        panic!("necessary text scratch must still be refused");
                    };
                    assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                    assert!(error
                        .detail()
                        .contains(&("limit_bytes".into(), (available + 7).to_string())));
                    assert!(error
                        .detail()
                        .contains(&("requested_bytes".into(), (needed + 7).to_string())));
                } else {
                    let parsed = result.unwrap();
                    assert_eq!(parsed.chunk_count, segment.chunks.len());
                    assert_eq!(parsed.document_ids.as_slice(), &[1]);
                    assert_eq!(parsed.payload.terms.capacity(), 0);
                    parsed.document_ids.assert_operation_context(&context);
                    let retained = segment.chunks.len() * 8;
                    remaining
                        .grow_to(
                            available - retained,
                            available - retained,
                            "operation_scratch",
                        )
                        .unwrap();
                    assert!(remaining
                        .grow_to(
                            available - retained + 1,
                            available - retained + 1,
                            "operation_scratch"
                        )
                        .is_err());
                    drop(parsed);
                }
                remaining
                    .grow_to(available, available, "operation_scratch")
                    .unwrap();
                assert!(remaining
                    .grow_to(available + 1, available + 1, "operation_scratch")
                    .is_err());
                drop(blocker);
                remaining
                    .grow_to(available + 7, available + 7, "operation_scratch")
                    .unwrap();
                assert!(remaining
                    .grow_to(available + 8, available + 8, "operation_scratch")
                    .is_err());
            }
        }
    }

    #[test]
    fn checked_metadata_payload_preserves_length_checksum_and_io_priority_without_credit() {
        let context = crate::json::JsonOperationContext::new(std::sync::Arc::new(
            crate::json::WorkLedger::with_limits(0, 0, 0),
        ));
        let records = BTreeMap::new();
        for seed in 0..10 {
            let checkpoint = empty_directory_checkpoint(seed, 0);
            let canonical = build_index_commit_metadata_payload(
                &checkpoint.metadata,
                0,
                0,
                0,
                checksum32(&[]),
                seed as u32,
            );
            for len in [57, 58, 59, 65_537] {
                let mut payload = canonical.to_vec();
                payload.resize(len, 17);
                let record = checked_payload_record(INDEX_COMMIT_METADATA, &payload);
                let mut bytes = vec![7; 37];
                bytes.extend_from_slice(&payload);
                let end = bytes.len() as u64;
                for fault in 0..5 {
                    let mut input = DirectoryInput::new(&bytes);
                    match fault {
                        1 => input.flip_read_at = Some(end - 1),
                        2 => input.eof_at = Some(end - 1),
                        3 => input.fail_read_at = Some(end - 1),
                        4 => input.fail_seek_at = Some(37),
                        _ => {}
                    }
                    let mut calls = 0;
                    let result =
                        with_checked_index_payload(&mut input, &record, &context, |source| {
                            calls += 1;
                            let mut decoded = [0; INDEX_COMMIT_METADATA_LEN];
                            source
                                .read_exact(&mut decoded[..len.min(INDEX_COMMIT_METADATA_LEN)])?;
                            Ok(parse_index_commit_metadata_payload(
                                if len == decoded.len() { &decoded } else { &[] },
                                checkpoint.metadata.covered_last_batch_id,
                                checkpoint.metadata.covered_last_commit_sequence_no,
                                &[],
                                &records,
                                0,
                            )?)
                        });
                    assert_eq!(calls, usize::from(fault != 4));
                    assert_eq!(input.seeks, 1);
                    match (fault, len, result) {
                        (0, INDEX_COMMIT_METADATA_LEN, Ok(parsed)) => assert_eq!(parsed.index_checksum, seed as u32),
                        (0, _, Err(IndexReadError::Invalid(error))) => assert_eq!(error.code, "index_commit_metadata_invalid"),
                        (1, _, Err(IndexReadError::Invalid(error))) => assert_eq!(error.code, "index_record_payload_checksum_mismatch"),
                        (2, _, Err(IndexReadError::Io(error))) => assert_eq!(error.kind(), std::io::ErrorKind::UnexpectedEof),
                        (3 | 4, _, Err(IndexReadError::Io(error))) => assert_eq!(error.kind(), std::io::ErrorKind::Other),
                        _ => panic!("unexpected metadata error priority: seed={seed}, len={len}, fault={fault}"),
                    }
                    if fault <= 1 {
                        assert_eq!(input.bytes.position(), end);
                    }
                    assert_eq!(context.available_workspace_bytes(), 0);
                }
            }
        }
    }

    #[test]
    fn checked_index_payload_retries_only_verified_memory_refusal_and_rechecks_each_attempt() {
        use crate::json::{JsonOperationContext, WorkLedger};
        use std::sync::Arc;

        for seed in 0..20 {
            let payload = vec![seed as u8; 65_537 + seed * 13];
            let mut bytes = vec![7; 37];
            bytes.extend_from_slice(&payload);
            let end = bytes.len() as u64;
            for kind in [INDEX_TEXT_SEGMENT, INDEX_VECTOR_SEGMENT] {
                let record = checked_payload_record(kind, &payload);
                for fault in 0..13 {
                    let limit = if fault == 12 { 0 } else { 17 };
                    let context = JsonOperationContext::new(Arc::new(WorkLedger::with_limits(
                        limit, 0, limit,
                    )));
                    let mut input = DirectoryInput::new(&bytes);
                    if (5..=8).contains(&fault) {
                        input.faults_enabled = false;
                        input.enable_faults_on_seek = Some(2);
                    }
                    match fault {
                        1 | 5 => input.flip_read_at = Some(end - 1),
                        2 | 6 => input.eof_at = Some(end - 1),
                        3 | 7 => input.fail_read_at = Some(end - 1),
                        4 | 8 => input.fail_seek_at = Some(37),
                        _ => {}
                    }
                    let mut calls = 0;
                    let result =
                        with_checked_index_payload(&mut input, &record, &context, |source| {
                            calls += 1;
                            let mut first = [0; 1];
                            source.read_exact(&mut first)?;
                            assert_eq!(first[0], seed as u8);
                            match fault {
                                9 => {
                                    return Err(index_extent_diagnostic(
                                        "index_segment_non_canonical",
                                        "injected semantic failure",
                                        37,
                                        record.index_record_id,
                                    )
                                    .into())
                                }
                                10 => {
                                    return Err(
                                        std::io::Error::other("injected decoder failure").into()
                                    )
                                }
                                11 => {
                                    return Err(IndexReadError::Resource(
                                        Error::resource_exhausted(
                                            "temporary_disk",
                                            "temporary_staging",
                                            17,
                                            18,
                                        ),
                                    ))
                                }
                                _ => {}
                            }
                            let mut required = context.workspace_reservation();
                            required
                                .grow_to(17, 17, "operation_scratch")
                                .map_err(IndexReadError::Resource)?;
                            Ok(required)
                        });
                    let expected_calls = match fault {
                        0 | 5..=7 => 2,
                        4 => 0,
                        _ => 1,
                    };
                    assert_eq!(calls, expected_calls, "fault {fault}");
                    assert_eq!(
                        input.seeks,
                        if (5..=8).contains(&fault) || fault == 0 {
                            2
                        } else {
                            1
                        }
                    );
                    match (fault, result) {
                        (0, Ok(required)) => {
                            assert_eq!(required.reserved_bytes(), 17);
                            assert_eq!(context.available_workspace_bytes(), 0);
                            drop(required);
                        }
                        (1 | 5, Err(IndexReadError::Invalid(error))) => assert_eq!(error.code, "index_record_payload_checksum_mismatch"),
                        (2 | 6, Err(IndexReadError::Io(error))) => assert_eq!(error.kind(), std::io::ErrorKind::UnexpectedEof),
                        (3 | 4 | 7 | 8 | 10, Err(IndexReadError::Io(error))) => assert_eq!(error.kind(), std::io::ErrorKind::Other),
                        (9, Err(IndexReadError::Invalid(error))) => assert_eq!(error.code, "index_segment_non_canonical"),
                        (11, Err(IndexReadError::Resource(error))) => assert_eq!(error.code(), ErrorCode::ResourceExhausted),
                        (12, Err(IndexReadError::Resource(error))) => {
                            assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                            for (key, value) in [
                                ("resource", "memory"),
                                ("operation", "operation_scratch"),
                                ("limit_bytes", "0"),
                                ("requested_bytes", "17"),
                            ] {
                                assert!(error.detail().contains(&(key.into(), value.into())));
                            }
                        }
                        _ => panic!("unexpected checked-payload result: seed={seed}, kind={kind}, fault={fault}"),
                    }
                    let mut free = context.workspace_reservation();
                    free.grow_to(limit, limit, "operation_scratch").unwrap();
                    assert!(free
                        .grow_to(limit + 1, limit + 1, "operation_scratch")
                        .is_err());
                }
            }
        }
    }

    #[test]
    fn parsed_index_records_retain_checked_file_ranges_without_payload_buffers() {
        fn diagnostic(error: IndexReadError) -> &'static str {
            match error {
                IndexReadError::Invalid(error) => error.code,
                IndexReadError::Io(error) => panic!("unexpected IO error: {error}"),
                IndexReadError::Resource(error) => panic!("unexpected resource error: {error}"),
            }
        }
        for seed in 0..100u64 {
            let payload = (0..seed as usize * 193)
                .map(|index| index as u8)
                .collect::<Vec<_>>();
            let record = build_index_record(INDEX_VECTOR_SEGMENT, seed, 2, &payload);
            let prefix = seed as usize % 13;
            let base = 37 + INDEX_CHECKPOINT_HEADER_LEN;
            let mut bytes = vec![7; base + prefix];
            bytes.extend_from_slice(&record);
            bytes.extend_from_slice(&[9; 11]);
            let extent = IndexExtentEntry {
                index_record_id: 2,
                record_type: INDEX_VECTOR_SEGMENT,
                stream_id: 1,
                suffix_relative_offset: (INDEX_CHECKPOINT_HEADER_LEN + prefix) as u64,
                length: record.len() as u64,
                checkpoint_generation: seed,
                extent_record_checksum: checksum32(&record),
            };
            let payload_len = bytes.len() - base;
            let parsed = parse_index_record_from_extent(
                &mut std::io::Cursor::new(&bytes),
                payload_len,
                &extent,
                seed,
                37,
            )
            .unwrap();
            assert_eq!(parsed.offset, (base + prefix) as u64);
            assert_eq!(parsed.length, record.len() as u64);
            assert_eq!(
                parsed.payload.offset,
                (base + prefix + INDEX_RECORD_HEADER_LEN) as u64
            );
            assert_eq!(parsed.payload.len, payload.len() as u64);
            assert_eq!(parsed.payload.checksum, checksum32(&payload) as u64);
            let decoded = with_checked_index_payload(
                &mut std::io::Cursor::new(&bytes),
                &parsed,
                &crate::json::JsonOperationContext::standalone(),
                |source| {
                    let mut output = vec![0; payload.len()];
                    source.read_exact(&mut output)?;
                    Ok(output)
                },
            )
            .unwrap();
            assert_eq!(decoded, payload);
            for damaged in [
                IndexExtentEntry {
                    length: u64::MAX,
                    ..extent
                },
                IndexExtentEntry {
                    suffix_relative_offset: u64::MAX,
                    ..extent
                },
                IndexExtentEntry {
                    length: INDEX_RECORD_HEADER_LEN as u64 - 1,
                    ..extent
                },
                IndexExtentEntry {
                    checkpoint_generation: seed + 1,
                    ..extent
                },
            ] {
                assert!(parse_index_record_from_extent(
                    &mut std::io::Cursor::new(&bytes),
                    payload_len,
                    &damaged,
                    seed,
                    37
                )
                .is_err());
            }
            if !payload.is_empty() {
                bytes[base + prefix + INDEX_RECORD_HEADER_LEN] ^= 1;
                assert_eq!(
                    diagnostic(
                        parse_index_record_from_extent(
                            &mut std::io::Cursor::new(&bytes),
                            payload_len,
                            &extent,
                            seed,
                            37
                        )
                        .err()
                        .unwrap()
                    ),
                    "index_extent_record_checksum_mismatch"
                );
                let repaired_extent = IndexExtentEntry {
                    extent_record_checksum: checksum32(
                        &bytes[base + prefix..base + prefix + record.len()],
                    ),
                    ..extent
                };
                assert_eq!(
                    diagnostic(
                        parse_index_record_from_extent(
                            &mut std::io::Cursor::new(&bytes),
                            payload_len,
                            &repaired_extent,
                            seed,
                            37
                        )
                        .err()
                        .unwrap()
                    ),
                    "index_record_payload_checksum_mismatch"
                );
            }
        }
    }

    #[test]
    fn streamed_segment_decoding_preserves_frames_bits_and_io_errors() {
        struct Fragmented<'a> {
            bytes: &'a [u8],
            fragment: usize,
            calls: usize,
            fail: bool,
        }
        impl Read for Fragmented<'_> {
            fn read(&mut self, output: &mut [u8]) -> std::io::Result<usize> {
                assert!(output.len() <= 4096);
                self.calls += 1;
                if self.fail && self.calls > 3 {
                    return Err(std::io::Error::other("injected index read failure"));
                }
                if self.calls % 7 == 0 {
                    return Err(std::io::ErrorKind::Interrupted.into());
                }
                let count = output.len().min(self.fragment).min(self.bytes.len());
                output[..count].copy_from_slice(&self.bytes[..count]);
                self.bytes = &self.bytes[count..];
                Ok(count)
            }
        }
        for seed in 0..100usize {
            let dim = [1, 2, 1023, 1024, 1025, 4097][seed % 6];
            let segment = VectorIndexSegmentPayload {
                collection_id: "col_1".into(),
                covered_commit_sequence_no: seed as u64,
                dim,
                distance_metric: DistanceMetric::DotProduct,
                embedding_model: Some("m\u{4e2d}\u{6587}".into()),
                vectors_normalized: false,
                vectors: vec![VectorIndexEntry {
                    chunk_id: "chk_1".into(),
                    embedding_id: "emb_2".into(),
                    vector: (0..dim)
                        .map(|index| {
                            f32::from_bits(match index % 4 {
                                0 => 0x8000_0000,
                                1 => 1,
                                2 => 0x7f7f_ffff,
                                _ => (index as f32).to_bits(),
                            })
                        })
                        .collect(),
                }],
            };
            let payload = encode_vector_index_segment_payload(&segment).unwrap();
            let body_len = payload.len() - 4;
            let mut source = Fragmented {
                bytes: &payload,
                fragment: seed % 31 + 1,
                calls: 0,
                fail: false,
            };
            let decoded = decode_vector_index_segment(
                &mut IndexPayloadReader::new(&mut source, body_len, 0),
                seed as u64,
                0,
            )
            .unwrap()
            .payload
            .unwrap();
            assert_eq!(source.bytes, &payload[body_len..]);
            assert_eq!(decoded.embedding_model, segment.embedding_model);
            let mut source = Fragmented {
                bytes: &payload,
                fragment: seed % 31 + 1,
                calls: 0,
                fail: false,
            };
            let summary = decode_vector_index_segment_in(
                &mut IndexPayloadReader::new(&mut source, body_len, 0),
                seed as u64,
                0,
                VectorDecodeMode::Summary,
            )
            .unwrap();
            assert_eq!(summary.vector_count, segment.vectors.len());
            assert!(summary.payload.is_none());
            assert_eq!(source.bytes, &payload[body_len..]);
            assert_eq!(
                decoded.vectors[0]
                    .vector
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                segment.vectors[0]
                    .vector
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>()
            );
            for length in [body_len - 1, body_len + 1] {
                assert!(matches!(
                    decode_vector_index_segment(
                        &mut IndexPayloadReader::new(payload.as_slice(), length, 0),
                        seed as u64,
                        0
                    ),
                    Err(IndexReadError::Invalid(_))
                ));
            }
            let mut truncated = Fragmented {
                bytes: &payload[..body_len - 1],
                fragment: 13,
                calls: 0,
                fail: false,
            };
            assert!(
                matches!(decode_vector_index_segment(&mut IndexPayloadReader::new(&mut truncated, body_len, 0), seed as u64, 0), Err(IndexReadError::Io(error)) if error.kind() == std::io::ErrorKind::UnexpectedEof)
            );
            let mut failed = Fragmented {
                bytes: &payload,
                fragment: 13,
                calls: 0,
                fail: true,
            };
            assert!(
                matches!(decode_vector_index_segment(&mut IndexPayloadReader::new(&mut failed, body_len, 0), seed as u64, 0), Err(IndexReadError::Io(error)) if error.kind() == std::io::ErrorKind::Other)
            );
            let mut oversized = payload.clone();
            oversized[20..24].copy_from_slice(&u32::MAX.to_le_bytes());
            assert!(matches!(
                decode_vector_index_segment(
                    &mut IndexPayloadReader::new(oversized.as_slice(), body_len, 0),
                    seed as u64,
                    0
                ),
                Err(IndexReadError::Invalid(_))
            ));
            let mut nonfinite = payload.clone();
            let vector_start = body_len - dim * 4;
            nonfinite[vector_start..vector_start + 4].copy_from_slice(&f32::NAN.to_le_bytes());
            assert!(
                matches!(decode_vector_index_segment(&mut IndexPayloadReader::new(nonfinite.as_slice(), body_len, 0), seed as u64, 0), Err(IndexReadError::Invalid(error)) if error.code == "index_segment_non_canonical")
            );
            for invalid in [&oversized, &nonfinite] {
                assert!(matches!(
                    decode_vector_index_segment_in(
                        &mut IndexPayloadReader::new(invalid.as_slice(), body_len, 0),
                        seed as u64,
                        0,
                        VectorDecodeMode::Summary,
                    ),
                    Err(IndexReadError::Invalid(_))
                ));
            }

            let text = TextIndexSegmentPayload {
                collection_id: "col_1".into(),
                covered_commit_sequence_no: seed as u64,
                chunks: vec![TextIndexChunkEntry {
                    chunk_id: "chk_1".into(),
                    document_id: "doc_2".into(),
                    token_count: 1,
                }],
                terms: vec![TextIndexTermEntry {
                    term: "\u{4e2d}\u{6587}".into(),
                    postings: vec![TextIndexPostingEntry {
                        chunk_id: "chk_1".into(),
                        term_frequency: 1,
                    }],
                }],
            };
            let payload = encode_text_index_segment_payload(&text).unwrap();
            let body_len = payload.len() - 4;
            let mut source = Fragmented {
                bytes: &payload,
                fragment: seed % 11 + 1,
                calls: 0,
                fail: false,
            };
            let decoded = decode_text_index_segment(
                &mut IndexPayloadReader::new(&mut source, body_len, 0),
                seed as u64,
                0,
            )
            .unwrap();
            assert_eq!(source.bytes, &payload[body_len..]);
            assert_eq!(
                encode_text_index_segment_payload(&decoded.payload).unwrap(),
                payload
            );
            let mut source = Fragmented {
                bytes: &payload,
                fragment: seed % 11 + 1,
                calls: 0,
                fail: false,
            };
            let summary = decode_text_index_segment_in(
                &mut IndexPayloadReader::new(&mut source, body_len, 0),
                seed as u64,
                0,
                false,
                &crate::json::JsonOperationContext::standalone(),
            )
            .unwrap();
            assert_eq!(summary.chunk_count, text.chunks.len());
            assert_eq!(summary.document_ids.as_slice(), &[2]);
            assert_eq!(summary.payload.chunks.capacity(), 0);
            assert_eq!(summary.payload.terms.capacity(), 0);
            assert_eq!(source.bytes, &payload[body_len..]);
            let mut failed = Fragmented {
                bytes: &payload,
                fragment: 13,
                calls: 0,
                fail: true,
            };
            assert!(matches!(
                decode_text_index_segment(
                    &mut IndexPayloadReader::new(&mut failed, body_len, 0),
                    seed as u64,
                    0
                ),
                Err(IndexReadError::Io(_))
            ));
        }
    }

    #[test]
    fn summary_term_allocation_failure_preserves_the_original_buffer_and_credit() {
        use crate::json::{JsonOperationContext, WorkLedger};
        use std::io::Cursor;
        use std::sync::Arc;
        let old = 4097;
        let new = 65_537;
        let terms = ["a".repeat(old), "b".into(), "z".repeat(new)];
        let mut encoded = Vec::new();
        for term in &terms {
            encoded.extend_from_slice(&(term.len() as u32).to_le_bytes());
            encoded.extend_from_slice(term.as_bytes());
        }
        let limit = old + new;
        for parent in [false, true] {
            let context = JsonOperationContext::new(Arc::new(WorkLedger::with_limits(
                if parent { 2 * limit } else { limit },
                2 * limit,
                if parent { limit } else { 2 * limit },
            )));
            let mut reader = IndexPayloadReader::new(Cursor::new(&encoded), encoded.len(), 37);
            let mut order = TextTermOrder::new(&context);
            order.read_next(&mut reader).unwrap();
            order.read_next(&mut reader).unwrap();
            let pointer = order.previous.as_ptr();
            let result =
                crate::test_alloc::fail_byte_allocation_once(new, || order.read_next(&mut reader));
            assert!(
                matches!(result, Err(IndexReadError::Io(error)) if error.kind() == std::io::ErrorKind::Other)
            );
            assert_eq!(reader.source.position() as usize, old + 1 + 3 * 4);
            assert_eq!(order.previous.as_ptr(), pointer);
            assert_eq!(order.previous.as_slice(), b"b");
            assert_eq!(order.previous.capacity(), old);
            assert_eq!(order.work.reserved_bytes(), old);
            assert_eq!(context.available_workspace_bytes(), limit - old);
            drop(order);
            let mut probe = context.workspace_reservation();
            probe.grow_to(limit, limit, "operation_scratch").unwrap();
            assert!(probe
                .grow_to(limit + 1, limit + 1, "operation_scratch")
                .is_err());
        }
    }

    #[test]
    fn summary_term_growth_admits_old_capacity_plus_new_before_reading() {
        use crate::json::{JsonOperationContext, WorkLedger};
        use std::io::Cursor;
        use std::sync::Arc;

        for seed in 0..100usize {
            let old = 4097 + seed;
            let new = 65_537 + seed;
            let terms = ["a".repeat(old), "b".into(), "z".repeat(new)];
            let mut encoded = Vec::new();
            for term in &terms {
                encoded.extend_from_slice(&(term.len() as u32).to_le_bytes());
                encoded.extend_from_slice(term.as_bytes());
            }
            let minimum = old + new;
            for parent in [false, true] {
                for delta in [-1isize, 0, 1] {
                    let limit = (minimum as isize + delta) as usize;
                    let context = JsonOperationContext::new(Arc::new(WorkLedger::with_limits(
                        if parent { 2 * limit } else { limit },
                        2 * limit,
                        if parent { limit } else { 2 * limit },
                    )));
                    let mut reader =
                        IndexPayloadReader::new(Cursor::new(&encoded), encoded.len(), 37);
                    let mut order = TextTermOrder::new(&context);
                    order.read_next(&mut reader).unwrap();
                    order.read_next(&mut reader).unwrap();
                    assert_eq!(order.previous.as_slice(), b"b");
                    assert_eq!(order.previous.capacity(), old);
                    assert_eq!(order.work.reserved_bytes(), old);
                    let pointer = order.previous.as_ptr();
                    let (_, allocations) = crate::test_alloc::key_allocations(new, || {
                        let result = order.read_next(&mut reader);
                        if delta < 0 {
                            let Err(IndexReadError::Resource(error)) = result else {
                                panic!("expected old plus new capacity rejection");
                            };
                            assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                            for (key, expected) in [
                                ("resource", "memory".to_owned()),
                                ("operation", "operation_scratch".to_owned()),
                                ("limit_bytes", limit.to_string()),
                                ("requested_bytes", minimum.to_string()),
                            ] {
                                assert_eq!(
                                    error.detail().iter().find(|item| item.0 == key).unwrap().1,
                                    expected
                                );
                            }
                            assert_eq!(reader.source.position() as usize, old + 1 + 3 * 4);
                            assert_eq!(order.previous.as_ptr(), pointer);
                            assert_eq!(order.previous.as_slice(), b"b");
                            assert_eq!(order.previous.capacity(), old);
                            assert_eq!(order.work.reserved_bytes(), old);
                        } else {
                            result.unwrap();
                            assert_eq!(reader.source.position() as usize, encoded.len());
                            assert_eq!(order.previous.as_slice(), terms[2].as_bytes());
                            assert_eq!(order.previous.capacity(), new);
                            assert_eq!(order.work.reserved_bytes(), new);
                            assert_eq!(context.available_workspace_bytes(), limit - new);
                        }
                        drop(order);
                    });
                    assert_eq!(allocations.allocated, usize::from(delta >= 0));
                    assert_eq!(allocations.peak, usize::from(delta >= 0));
                    assert_eq!(allocations.live, 0);
                    assert!(!allocations.overflow);
                    let mut probe = context.workspace_reservation();
                    probe.grow_to(limit, limit, "operation_scratch").unwrap();
                    assert!(probe
                        .grow_to(limit + 1, limit + 1, "operation_scratch")
                        .is_err());
                }
            }
        }
    }

    #[test]
    fn summary_term_capacity_reuse_needs_no_additional_credit() {
        use crate::json::{JsonOperationContext, WorkLedger};
        use std::io::Cursor;
        use std::sync::Arc;
        for capacity in 4097..4197 {
            let terms = ["a".repeat(capacity), "b".into(), "c".repeat(capacity - 1)];
            let mut encoded = Vec::new();
            for term in &terms {
                encoded.extend_from_slice(&(term.len() as u32).to_le_bytes());
                encoded.extend_from_slice(term.as_bytes());
            }
            let context = JsonOperationContext::new(Arc::new(WorkLedger::with_limits(
                capacity, capacity, capacity,
            )));
            let mut reader = IndexPayloadReader::new(Cursor::new(&encoded), encoded.len(), 37);
            let mut order = TextTermOrder::new(&context);
            order.read_next(&mut reader).unwrap();
            let pointer = order.previous.as_ptr();
            assert_eq!(context.available_workspace_bytes(), 0);
            let (_, allocations) = crate::test_alloc::allocations(|| {
                order.read_next(&mut reader).unwrap();
                order.read_next(&mut reader).unwrap();
            });
            assert_eq!(allocations, 0);
            assert_eq!(order.previous.as_ptr(), pointer);
            assert_eq!(order.previous.as_slice(), terms[2].as_bytes());
            assert_eq!(order.work.reserved_bytes(), capacity);
            drop(order);
            context
                .workspace_reservation()
                .grow_to(capacity, capacity, "operation_scratch")
                .unwrap();
        }
    }

    #[test]
    fn summary_term_order_matches_string_validation_across_block_boundaries() {
        struct ShortReads<'a> {
            bytes: &'a [u8],
            calls: usize,
        }
        impl Read for ShortReads<'_> {
            fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
                self.calls += 1;
                if self.calls % 11 == 0 {
                    return Err(std::io::ErrorKind::Interrupted.into());
                }
                let count = out.len().min(4093);
                self.bytes.read(&mut out[..count])
            }
        }
        for width in [0, 1, 4093, 4094, 4095, 4096, 4097, 8191, 65_535] {
            let prefix = "x".repeat(width);
            let mut terms = vec![
                Vec::new(),
                prefix.as_bytes().to_vec(),
                format!("{prefix}a").into_bytes(),
                format!("{prefix}ab").into_bytes(),
                format!("{prefix}b").into_bytes(),
                format!("{prefix}\u{e9}").into_bytes(),
                format!("{prefix}\u{4e2d}").into_bytes(),
                format!("{prefix}\u{1f4a1}").into_bytes(),
            ];
            for invalid in [
                &[0xc0, 0x80][..],
                &[0xe4, 0x80],
                &[0xf0, 0x80, 0x80, 0x80],
                &[0x80],
            ] {
                let mut bytes = prefix.as_bytes().to_vec();
                bytes.extend_from_slice(invalid);
                terms.push(bytes);
            }
            let mut sequences: Vec<Vec<&[u8]>> = terms
                .iter()
                .flat_map(|first| {
                    terms
                        .iter()
                        .map(move |second| vec![first.as_slice(), second.as_slice()])
                })
                .collect();
            let growing = [
                b"a".to_vec(),
                format!("a{prefix}z").into_bytes(),
                b"b".to_vec(),
                format!("b{prefix}z").into_bytes(),
                b"c".to_vec(),
            ];
            sequences.push(growing.iter().map(Vec::as_slice).collect());
            for sequence in sequences {
                let max = sequence.iter().map(|term| term.len()).max().unwrap();
                let limit = 2 * max;
                let context = crate::json::JsonOperationContext::new(std::sync::Arc::new(
                    crate::json::WorkLedger::with_limits(limit, limit, limit),
                ));
                let mut order = TextTermOrder::new(&context);
                let mut encoded = Vec::new();
                for bytes in &sequence {
                    encoded.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
                    encoded.extend_from_slice(bytes);
                }
                let mut source = IndexPayloadReader::new(
                    ShortReads {
                        bytes: &encoded,
                        calls: 0,
                    },
                    encoded.len(),
                    37,
                );
                let mut reference = IndexPayloadReader::new(encoded.as_slice(), encoded.len(), 37);
                let mut previous: Option<String> = None;
                for _ in &sequence {
                    let expected = reference.read_string("text segment term").and_then(|term| {
                        if previous
                            .as_ref()
                            .is_some_and(|old| old.as_bytes() >= term.as_bytes())
                        {
                            return Err(text_term_order_error(37));
                        }
                        Ok(term)
                    });
                    match (order.read_next(&mut source), expected) {
                        (Ok(()), Ok(term)) => {
                            assert_eq!(order.previous, term.as_bytes());
                            let free = limit - order.previous.capacity();
                            let mut rest = context.workspace_reservation();
                            rest.grow_to(free, free, "operation_scratch").unwrap();
                            assert!(context
                                .workspace_reservation()
                                .grow_to(1, 1, "operation_scratch")
                                .is_err());
                            previous = Some(term);
                        }
                        (
                            Err(IndexReadError::Invalid(actual)),
                            Err(IndexReadError::Invalid(expected)),
                        ) => {
                            assert_eq!(
                                (actual.code, actual.message, actual.offset),
                                (expected.code, expected.message, expected.offset)
                            );
                            break;
                        }
                        _ => panic!("term ordering differed at width {width}"),
                    }
                }
                drop(order);
                context
                    .workspace_reservation()
                    .grow_to(limit, limit, "operation_scratch")
                    .unwrap();
            }
        }
    }

    fn summary_term_fixture(seed: usize) -> TextIndexSegmentPayload {
        let count = seed % 7 + 1;
        let width = if seed % 31 == 0 {
            65_533
        } else {
            seed * 19 + 1
        };
        TextIndexSegmentPayload {
            collection_id: catalog_id_string("col_", 1),
            covered_commit_sequence_no: 11,
            chunks: (1..=count)
                .map(|id| TextIndexChunkEntry {
                    chunk_id: catalog_id_string("chk_", id as u64),
                    document_id: catalog_id_string("doc_", 1),
                    token_count: 4,
                })
                .collect(),
            terms: ["a\u{4e2d}\u{6587}", "a\u{4e2d}\u{6587}x", "b", "z\u{1f4a1}"]
                .into_iter()
                .map(|suffix| TextIndexTermEntry {
                    term: format!("{}{suffix}", "x".repeat(width)),
                    postings: (1..=count)
                        .map(|id| TextIndexPostingEntry {
                            chunk_id: catalog_id_string("chk_", id as u64),
                            term_frequency: 1,
                        })
                        .collect(),
                })
                .collect(),
        }
    }

    #[test]
    fn summary_terms_reuse_one_admitted_buffer_before_reading() {
        use crate::json::{JsonOperationContext, WorkLedger};
        use std::io::Cursor;
        use std::sync::Arc;
        for seed in 0..100 {
            let segment = summary_term_fixture(seed);
            let bytes = encode_text_index_segment_payload(&segment).unwrap();
            let ids = segment.chunks.len() * 16;
            let terms = segment.terms[0].term.len() + segment.terms[1].term.len();
            let needed = ids + terms;
            let limit = needed + 4096;
            for available in [needed - 1, needed, needed + 1] {
                let ledger = Arc::new(WorkLedger::with_limits(limit, limit, limit));
                let context = JsonOperationContext::new(ledger.clone());
                let competing = JsonOperationContext::new(ledger);
                let mut blocker = competing.workspace_reservation();
                blocker
                    .grow_to(limit - available, limit - available, "operation_scratch")
                    .unwrap();
                let mut source = Cursor::new(&bytes);
                let result = decode_text_index_segment_in(
                    &mut IndexPayloadReader::new(&mut source, bytes.len() - 4, 0),
                    11,
                    0,
                    false,
                    &context,
                );
                if available < needed {
                    let Err(IndexReadError::Resource(error)) = result else {
                        panic!("expected shared term workspace rejection");
                    };
                    assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                    assert!(error
                        .detail()
                        .contains(&("limit_bytes".into(), limit.to_string())));
                    assert!(error
                        .detail()
                        .contains(&("requested_bytes".into(), (limit + 1).to_string())));
                    let second_term_data = 38
                        + segment.chunks.len() * 24
                        + 8
                        + 4
                        + segment.terms[0].term.len()
                        + 16
                        + segment.chunks.len() * 16
                        + 4;
                    assert_eq!(source.position() as usize, second_term_data);
                } else {
                    let parsed = result.unwrap();
                    assert_eq!(source.position() as usize, bytes.len() - 4);
                    assert_eq!(parsed.document_ids.as_slice(), &[1]);
                    assert_eq!(parsed.payload.collection_id.capacity(), 0);
                    assert_eq!(parsed.payload.terms.capacity(), 0);
                    let remaining = available - segment.chunks.len() * 8;
                    let mut occupancy = competing.workspace_reservation();
                    occupancy
                        .grow_to(remaining, remaining, "operation_scratch")
                        .unwrap();
                    assert!(context
                        .workspace_reservation()
                        .grow_to(1, 1, "operation_scratch")
                        .is_err());
                    drop(occupancy);
                    drop(parsed);
                }
                context
                    .workspace_reservation()
                    .grow_to(available, available, "operation_scratch")
                    .unwrap();
                drop(blocker);
                context
                    .workspace_reservation()
                    .grow_to(limit, limit, "operation_scratch")
                    .unwrap();
            }
        }
    }

    #[test]
    fn summary_term_budget_rejection_keeps_checksum_and_io_priority() {
        use crate::json::{JsonOperationContext, WorkLedger};
        use std::io::Cursor;
        use std::sync::Arc;
        struct Input {
            bytes: Cursor<Vec<u8>>,
            fail_at: Option<u64>,
        }
        impl Read for Input {
            fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
                let count = if let Some(at) = self.fail_at {
                    if self.bytes.position() >= at {
                        return Err(std::io::Error::other("injected summary input failure"));
                    }
                    out.len().min((at - self.bytes.position()) as usize)
                } else {
                    out.len()
                };
                self.bytes.read(&mut out[..count])
            }
        }
        impl Seek for Input {
            fn seek(&mut self, at: SeekFrom) -> std::io::Result<u64> {
                self.bytes.seek(at)
            }
        }
        for seed in 0..100 {
            let segment = summary_term_fixture(seed);
            let payload = encode_text_index_segment_payload(&segment).unwrap();
            let limit = segment.chunks.len() * 16
                + segment.terms[0].term.len()
                + segment.terms[1].term.len()
                - 1;
            for fault in 0..4 {
                let mut bytes = vec![7; 37];
                bytes.extend_from_slice(&payload);
                if fault == 1 {
                    *bytes.last_mut().unwrap() ^= 1;
                }
                if fault == 2 {
                    bytes.pop();
                }
                let record = ParsedIndexRecord {
                    record_type: INDEX_TEXT_SEGMENT,
                    index_record_id: 2,
                    stream_id: 1,
                    offset: 0,
                    length: (INDEX_RECORD_HEADER_LEN + payload.len()) as u64,
                    payload: PayloadRef {
                        offset: 37,
                        len: payload.len() as u64,
                        checksum: checksum32(&payload) as u64,
                    },
                    segment_checksum_valid: true,
                };
                let fail_at = (fault == 3).then_some(bytes.len() as u64 - 1);
                let mut input = Input {
                    bytes: Cursor::new(bytes),
                    fail_at,
                };
                let context = JsonOperationContext::new(Arc::new(WorkLedger::with_limits(
                    limit, limit, limit,
                )));
                let result = with_checked_index_payload(&mut input, &record, &context, |source| {
                    decode_text_index_segment_in(
                        &mut IndexPayloadReader::new(source, payload.len() - 4, 0),
                        11,
                        0,
                        false,
                        &context,
                    )
                });
                match (fault, result) {
                    (0, Err(IndexReadError::Resource(error))) => {
                        assert_eq!(error.code(), ErrorCode::ResourceExhausted)
                    }
                    (1, Err(IndexReadError::Invalid(error))) => {
                        assert_eq!(error.code, "index_record_payload_checksum_mismatch")
                    }
                    (2, Err(IndexReadError::Io(error))) => {
                        assert_eq!(error.kind(), std::io::ErrorKind::UnexpectedEof)
                    }
                    (3, Err(IndexReadError::Io(error))) => {
                        assert_eq!(error.kind(), std::io::ErrorKind::Other)
                    }
                    _ => panic!("unexpected error priority for seed {seed}, fault {fault}"),
                }
                context
                    .workspace_reservation()
                    .grow_to(limit, limit, "operation_scratch")
                    .unwrap();
            }
        }
    }

    #[test]
    fn summary_term_growth_rejection_rechecks_tail_faults_after_read_ahead_is_released() {
        use crate::json::{JsonOperationContext, WorkLedger};
        use std::sync::Arc;

        let segment = summary_term_fixture(1);
        let payload = encode_text_index_segment_payload(&segment).unwrap();
        let needed =
            segment.chunks.len() * 16 + segment.terms[0].term.len() + segment.terms[1].term.len();
        let limit = needed - 1;
        let record = checked_payload_record(INDEX_TEXT_SEGMENT, &payload);
        let mut bytes = vec![7; 37];
        bytes.extend_from_slice(&payload);
        let second_term_body = 38
            + segment.chunks.len() * 24
            + 8
            + 4
            + segment.terms[0].term.len()
            + 16
            + segment.chunks.len() * 16
            + 4;
        for fault in 0..4 {
            let mut input = DirectoryInput::new(&bytes);
            input.faults_enabled = false;
            input.enable_faults_on_seek = Some(2);
            let tail = bytes.len() as u64 - 1;
            input.flip_read_at = (fault == 1).then_some(tail);
            input.eof_at = (fault == 2).then_some(tail);
            input.fail_read_at = (fault == 3).then_some(tail);
            let context =
                JsonOperationContext::new(Arc::new(WorkLedger::with_limits(limit, 0, limit)));
            let mut positions = Vec::new();
            let mut requests = Vec::new();
            let result = with_checked_index_payload(&mut input, &record, &context, |source| {
                let mut reader = IndexPayloadReader::new(source, payload.len() - 4, 0);
                let result = decode_text_index_segment_in(&mut reader, 11, 0, false, &context);
                positions.push(payload.len() - 4 - reader.remaining);
                let Err(IndexReadError::Resource(error)) = &result else {
                    panic!("each attempt must reach its independent budget refusal");
                };
                requests.push(error.detail().to_vec());
                result
            });
            assert_eq!(positions, [38, second_term_body]);
            assert_eq!(input.seeks, 2);
            for (detail, requested) in requests.iter().zip([limit + 16, needed]) {
                assert!(detail.contains(&("resource".into(), "memory".into())));
                assert!(detail.contains(&("operation".into(), "operation_scratch".into())));
                assert!(detail.contains(&("limit_bytes".into(), limit.to_string())));
                assert!(detail.contains(&("requested_bytes".into(), requested.to_string())));
            }
            match (fault, result) {
                (0, Err(IndexReadError::Resource(error))) => {
                    assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                    assert_eq!(error.detail(), requests[1]);
                }
                (1, Err(IndexReadError::Invalid(error))) => {
                    assert_eq!(error.code, "index_record_payload_checksum_mismatch");
                }
                (2, Err(IndexReadError::Io(error))) => {
                    assert_eq!(error.kind(), std::io::ErrorKind::UnexpectedEof);
                }
                (3, Err(IndexReadError::Io(error))) => {
                    assert_eq!(error.kind(), std::io::ErrorKind::Other);
                    assert_eq!(error.to_string(), "injected directory read failure");
                }
                _ => panic!("unexpected retry error priority for fault {fault}"),
            }
            let mut complete = context.workspace_reservation();
            complete.grow_to(limit, limit, "operation_scratch").unwrap();
            assert!(complete
                .grow_to(limit + 1, limit + 1, "operation_scratch")
                .is_err());
        }
    }

    #[test]
    fn text_validation_arrays_admit_before_reading_and_retain_only_live_capacity() {
        use crate::json::{JsonOperationContext, WorkLedger};
        use std::io::Cursor;
        use std::sync::Arc;

        for seed in 1..=100usize {
            let count = seed * 3;
            let segment = TextIndexSegmentPayload {
                collection_id: "col_1".into(),
                covered_commit_sequence_no: 7,
                chunks: (0..count)
                    .map(|id| TextIndexChunkEntry {
                        chunk_id: catalog_id_string("chk_", u64::MAX - count as u64 + id as u64),
                        document_id: catalog_id_string("doc_", (id % seed) as u64),
                        token_count: 1,
                    })
                    .collect(),
                terms: vec![TextIndexTermEntry {
                    term: "token".into(),
                    postings: (0..count)
                        .map(|id| TextIndexPostingEntry {
                            chunk_id: catalog_id_string(
                                "chk_",
                                u64::MAX - count as u64 + id as u64,
                            ),
                            term_frequency: 1,
                        })
                        .collect(),
                }],
            };
            let bytes = encode_text_index_segment_payload(&segment).unwrap();
            let needed = count * 2 * std::mem::size_of::<u64>();
            for retain in [false, true] {
                let complete = needed + if retain { 0 } else { 5 };
                for limit in [needed - 1, complete, complete + 1] {
                    let ledger = Arc::new(WorkLedger::with_limits(limit, limit, limit));
                    let context = JsonOperationContext::new(ledger.clone());
                    let competing = JsonOperationContext::new(ledger);
                    let mut source = Cursor::new(&bytes);
                    let parsed = decode_text_index_segment_in(
                        &mut IndexPayloadReader::new(&mut source, bytes.len() - 4, 0),
                        7,
                        0,
                        retain,
                        &context,
                    );
                    if limit < needed {
                        let Err(IndexReadError::Resource(error)) = parsed else {
                            panic!("expected shared validation budget rejection");
                        };
                        assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                        assert!(error
                            .detail()
                            .contains(&("requested_bytes".into(), needed.to_string())));
                        assert!(error
                            .detail()
                            .contains(&("limit_bytes".into(), limit.to_string())));
                        assert_eq!(source.position(), 38);
                    } else {
                        let parsed = parsed.unwrap();
                        assert_eq!(
                            parsed.document_ids.as_slice(),
                            (0..seed as u64).collect::<Vec<_>>()
                        );
                        assert_eq!(parsed.chunk_count, count);
                        assert_eq!(source.position() as usize, bytes.len() - 4);
                        if retain {
                            assert_eq!(
                                encode_text_index_segment_payload(&parsed.payload).unwrap(),
                                bytes
                            );
                        }
                        let mut remaining = competing.workspace_reservation();
                        remaining
                            .grow_to(limit - needed / 2, limit - needed / 2, "operation_scratch")
                            .unwrap();
                        assert!(context
                            .workspace_reservation()
                            .grow_to(1, 1, "operation_scratch")
                            .is_err());
                        drop(remaining);
                        drop(parsed);
                    }
                    competing
                        .workspace_reservation()
                        .grow_to(limit, limit, "operation_scratch")
                        .unwrap();
                }
            }
        }
    }

    #[test]
    fn checkpoint_validation_directory_uses_shared_budget_without_corruption_mapping() {
        use crate::json::{btree_workspace, JsonOperationContext, WorkLedger};
        use std::io::Cursor;
        use std::sync::Arc;

        for seed in 1..=100u64 {
            let append = AppendSummary {
                commit: CommitPayload {
                    batch_id: seed,
                    first_sequence_no: 1,
                    last_sequence_no: 1,
                    record_count: 1,
                    batch_checksum: 0,
                    committed_at: String::new(),
                },
                commit_sequence_no: seed + 1,
                records: Vec::new(),
                _commit_work: None,
            };
            let mut checkpoint = full_checkpoint_for(&append);
            let collections = 1 + seed as usize % 17;
            let text = checkpoint.text_segments[0].clone();
            let vector = checkpoint.vector_segments[0].clone();
            checkpoint.text_segments = (1..=collections)
                .map(|id| {
                    let mut segment = text.clone();
                    segment.collection_id = catalog_id_string("col_", id as u64);
                    segment.chunks[0].chunk_id = catalog_id_string("chk_", id as u64);
                    segment.terms[0].postings[0].chunk_id = segment.chunks[0].chunk_id.clone();
                    segment
                })
                .collect();
            checkpoint.vector_segments = (1..=collections)
                .map(|id| {
                    let mut segment = vector.clone();
                    segment.collection_id = catalog_id_string("col_", id as u64);
                    segment.vectors[0].chunk_id = catalog_id_string("chk_", id as u64);
                    segment.vectors[0].embedding_id = catalog_id_string("emb_", id as u64);
                    segment
                })
                .collect();
            checkpoint.metadata.collection_count = collections;
            checkpoint.metadata.text_segment_count = collections;
            checkpoint.metadata.vector_segment_count = collections;
            checkpoint.metadata.live_chunk_count = collections;
            checkpoint.metadata.live_embedding_count = collections;
            let mut bytes = vec![7; 37];
            bytes.extend(build_index_checkpoint_suffix(&checkpoint, 37).unwrap());
            let needed = (collections * 2 + 1) * INDEX_EXTENT_ENTRY_LEN
                + btree_workspace::<u64, ParsedIndexRecord>(collections * 2 + 1)
                + 2 * btree_workspace::<u64, ()>(collections)
                + btree_workspace::<u64, ()>(1);
            for mode in [
                IndexReadMode::Summary,
                IndexReadMode::LazyVectors,
                IndexReadMode::Full,
            ] {
                for available in [needed - 1, needed, needed + 1] {
                    let limit = 64 * 1024;
                    let ledger = Arc::new(WorkLedger::with_limits(limit, limit, limit));
                    let context = JsonOperationContext::new(ledger.clone());
                    let competing = JsonOperationContext::new(ledger);
                    let mut blocker = competing.workspace_reservation();
                    blocker
                        .grow_to(limit - available, limit - available, "operation_scratch")
                        .unwrap();
                    let mut source = Cursor::new(&bytes);
                    let parsed = read_index_checkpoint_suffix_in(
                        &mut source,
                        37,
                        bytes.len() as u64,
                        37,
                        seed,
                        seed + 1,
                        (
                            IndexReadOptions {
                                mode,
                                budget_bytes: u64::MAX,
                            },
                            &context,
                        ),
                    );
                    if available < needed {
                        let Err(error) = parsed else {
                            panic!("validation budget must not become an ignored invalid suffix");
                        };
                        assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                        assert!(error
                            .detail()
                            .contains(&("requested_bytes".into(), (limit + 1).to_string())));
                        assert!(error
                            .detail()
                            .contains(&("limit_bytes".into(), limit.to_string())));
                    } else {
                        let report = parsed.unwrap().unwrap().report;
                        assert_eq!(report.collection_count, collections);
                        assert_eq!(report.live_document_count, 1);
                        assert_eq!(report.live_chunk_count, collections);
                        assert_eq!(report.live_embedding_count, collections);
                    }
                    context
                        .workspace_reservation()
                        .grow_to(available, available, "operation_scratch")
                        .unwrap();
                    drop(blocker);
                    context
                        .workspace_reservation()
                        .grow_to(limit, limit, "operation_scratch")
                        .unwrap();
                }
            }
        }
    }

    #[test]
    fn checkpoint_parser_containers_share_budget_and_release_only_dead_capacity() {
        use crate::json::{btree_workspace, JsonOperationContext, WorkLedger};
        use std::io::Cursor;
        use std::sync::Arc;
        for seed in 0..100u64 {
            let collections = seed as usize % 13;
            let checkpoint = empty_directory_checkpoint(seed, collections);
            let mut bytes = vec![7; 37];
            bytes.extend(build_index_checkpoint_suffix(&checkpoint, 37).unwrap());
            let count = collections * 2 + 1;
            let table_bytes = count * INDEX_EXTENT_ENTRY_LEN;
            let map_bytes = btree_workspace::<u64, ParsedIndexRecord>(count);
            let needed = table_bytes + map_bytes + 2 * btree_workspace::<u64, ()>(collections);
            for mode in [IndexReadMode::Summary, IndexReadMode::LazyVectors] {
                for available in [needed - 1, needed, needed + 1] {
                    let total = 128 * 1024;
                    let ledger = Arc::new(WorkLedger::with_limits(total, total, total));
                    let context = JsonOperationContext::new(ledger.clone());
                    let competing = JsonOperationContext::new(ledger);
                    let mut blocker = competing.workspace_reservation();
                    blocker
                        .grow_to(total - available, total - available, "operation_scratch")
                        .unwrap();
                    let parsed = read_index_checkpoint_suffix_in(
                        &mut Cursor::new(&bytes),
                        37,
                        bytes.len() as u64,
                        37,
                        seed,
                        seed + 1,
                        (
                            IndexReadOptions {
                                mode,
                                budget_bytes: u64::MAX,
                            },
                            &context,
                        ),
                    );
                    if available < needed {
                        let error = parsed.err().expect("directory budget rejection");
                        assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                        assert!(error
                            .detail()
                            .contains(&("limit_bytes".into(), total.to_string())));
                        assert!(error
                            .detail()
                            .contains(&("requested_bytes".into(), (total + 1).to_string())));
                    } else {
                        let parsed = parsed.unwrap().unwrap();
                        assert_eq!(parsed.report.collection_count, collections);
                        assert_eq!(parsed.report.live_chunk_count, 0);
                        let retained = if mode == IndexReadMode::LazyVectors {
                            collections * 2 * std::mem::size_of::<IndexExtentEntry>()
                        } else {
                            0
                        };
                        assert_eq!(context.available_workspace_bytes(), available - retained);
                        let mut remainder = competing.workspace_reservation();
                        remainder
                            .grow_to(
                                available - retained,
                                available - retained,
                                "operation_scratch",
                            )
                            .unwrap();
                        assert!(context
                            .workspace_reservation()
                            .grow_to(1, 1, "operation_scratch")
                            .is_err());
                        drop(parsed);
                        remainder
                            .grow_to(available, available, "operation_scratch")
                            .unwrap();
                    }
                    context
                        .workspace_reservation()
                        .grow_to(available, available, "operation_scratch")
                        .unwrap();
                    drop(blocker);
                    context
                        .workspace_reservation()
                        .grow_to(total, total, "operation_scratch")
                        .unwrap();
                }
            }
        }
    }

    fn empty_directory_checkpoint(seed: u64, collections: usize) -> IndexCheckpointPayload {
        IndexCheckpointPayload {
            metadata: IndexCheckpointMetadata {
                covered_last_batch_id: seed,
                covered_last_commit_sequence_no: seed + 1,
                coverage_hash: seed as u32,
                collection_count: collections,
                live_document_count: 0,
                live_chunk_count: 0,
                live_embedding_count: 0,
                text_segment_count: collections,
                vector_segment_count: collections,
                build_mode: "inline_full_checkpoint".into(),
            },
            text_segments: (1..=collections)
                .map(|id| TextIndexSegmentPayload {
                    collection_id: format!("col_{id}"),
                    covered_commit_sequence_no: seed + 1,
                    chunks: Vec::new(),
                    terms: Vec::new(),
                })
                .collect(),
            vector_segments: (1..=collections)
                .map(|id| VectorIndexSegmentPayload {
                    collection_id: format!("col_{id}"),
                    covered_commit_sequence_no: seed + 1,
                    dim: 3,
                    distance_metric: DistanceMetric::Cosine,
                    embedding_model: None,
                    vectors_normalized: false,
                    vectors: Vec::new(),
                })
                .collect(),
        }
    }

    struct DirectoryInput<'a> {
        bytes: std::io::Cursor<&'a [u8]>,
        fail_read_at: Option<u64>,
        fail_seek_at: Option<u64>,
        fail_after_seek: Option<u64>,
        eof_at: Option<u64>,
        enable_faults_at_seek: Option<u64>,
        enable_faults_on_seek: Option<usize>,
        faults_enabled: bool,
        flip_read_at: Option<u64>,
        calls: usize,
        seeks: usize,
    }

    impl<'a> DirectoryInput<'a> {
        fn new(bytes: &'a [u8]) -> Self {
            Self {
                bytes: std::io::Cursor::new(bytes),
                fail_read_at: None,
                fail_seek_at: None,
                fail_after_seek: None,
                eof_at: None,
                enable_faults_at_seek: None,
                enable_faults_on_seek: None,
                faults_enabled: true,
                flip_read_at: None,
                calls: 0,
                seeks: 0,
            }
        }
    }

    impl Read for DirectoryInput<'_> {
        fn read(&mut self, output: &mut [u8]) -> std::io::Result<usize> {
            self.calls += 1;
            if self.calls % 7 == 0 {
                return Err(std::io::ErrorKind::Interrupted.into());
            }
            let mut count = output.len().min(8191);
            if self.faults_enabled {
                if let Some(at) = self.eof_at {
                    if self.bytes.position() >= at {
                        return Ok(0);
                    }
                    count = count.min((at - self.bytes.position()) as usize);
                }
            }
            let count = if let Some(at) = self.fail_read_at.filter(|_| self.faults_enabled) {
                if self.bytes.position() >= at {
                    return Err(std::io::Error::other("injected directory read failure"));
                }
                count.min((at - self.bytes.position()) as usize)
            } else {
                count
            };
            let start = self.bytes.position();
            let read = self.bytes.read(&mut output[..count])?;
            if let Some(at) = self.flip_read_at.filter(|_| self.faults_enabled) {
                if at >= start && at - start < read as u64 {
                    output[(at - start) as usize] ^= 1;
                }
            }
            Ok(read)
        }
    }

    impl Seek for DirectoryInput<'_> {
        fn seek(&mut self, at: SeekFrom) -> std::io::Result<u64> {
            self.seeks += 1;
            if self.enable_faults_on_seek == Some(self.seeks) {
                self.faults_enabled = true;
            }
            if let SeekFrom::Start(position) = at {
                if self.enable_faults_at_seek == Some(position) {
                    self.faults_enabled = true;
                }
                if self.faults_enabled && self.fail_seek_at == Some(position) {
                    return Err(std::io::Error::other("injected directory seek failure"));
                }
                if self.fail_after_seek == Some(position) {
                    self.fail_read_at = Some(position);
                }
            }
            self.bytes.seek(at)
        }
    }

    #[test]
    fn checkpoint_table_admission_keeps_checksum_trailer_and_io_priority() {
        use crate::json::{JsonOperationContext, WorkLedger};
        use std::sync::Arc;
        for seed in 0..100u64 {
            let collections = if seed == 99 { 2000 } else { seed as usize % 13 };
            let checkpoint = empty_directory_checkpoint(seed, collections);
            let mut original = vec![7; 37];
            original.extend(build_index_checkpoint_suffix(&checkpoint, 37).unwrap());
            let table_start = 37 + read_u64(&original[37 + 73..37 + 81]) as usize;
            let table_len = (collections * 2 + 1) * INDEX_EXTENT_ENTRY_LEN;
            let trailer_start = original.len() - INDEX_CHECKPOINT_TRAILER_LEN;
            for fault in 0..9 {
                let mut bytes = original.clone();
                if fault == 1 {
                    bytes[table_start + table_len - 1] ^= 1;
                }
                if fault == 2 {
                    *bytes.last_mut().unwrap() ^= 1;
                }
                let mut input = DirectoryInput::new(&bytes);
                // Arm after the initial trailer check to exercise the rejected
                // table allocation's bounded verification, not the front check.
                input.faults_enabled = false;
                input.enable_faults_at_seek = Some(37 + INDEX_CHECKPOINT_HEADER_LEN as u64);
                if fault == 3 {
                    input.fail_read_at = Some((table_start + table_len - 1) as u64);
                }
                if fault == 5 {
                    input.fail_read_at = Some((original.len() - 1) as u64);
                }
                if fault == 4 {
                    input.eof_at = Some((table_start + table_len - 1) as u64);
                }
                if fault == 6 {
                    input.eof_at = Some((original.len() - 1) as u64);
                }
                if fault == 7 {
                    input.fail_seek_at = Some(trailer_start as u64);
                }
                if fault == 8 {
                    input.flip_read_at = Some((original.len() - 1) as u64);
                }
                let available = table_len - 1;
                let total = available + 4096;
                let ledger = Arc::new(WorkLedger::with_limits(total, total, total));
                let context = JsonOperationContext::new(ledger.clone());
                let competing = JsonOperationContext::new(ledger);
                let mut blocker = competing.workspace_reservation();
                blocker.grow_to(4096, 4096, "operation_scratch").unwrap();
                let parsed = read_index_checkpoint_suffix_in(
                    &mut input,
                    37,
                    original.len() as u64,
                    37,
                    seed,
                    seed + 1,
                    (
                        IndexReadOptions {
                            mode: IndexReadMode::Summary,
                            budget_bytes: u64::MAX,
                        },
                        &context,
                    ),
                );
                match (fault, parsed) {
                    (0, Err(error)) => {
                        assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                        assert!(error
                            .detail()
                            .contains(&("requested_bytes".into(), (total + 1).to_string())));
                        assert_eq!(input.bytes.position(), original.len() as u64);
                    }
                    (1, Ok(Err(diagnostic))) => {
                        assert_eq!(diagnostic.code, "index_extent_table_mismatch");
                        assert_eq!(diagnostic.offset, table_start as u64);
                    }
                    (2 | 8, Ok(Err(diagnostic))) => {
                        assert_eq!(diagnostic.code, "index_checkpoint_incomplete");
                        assert_eq!(diagnostic.offset, trailer_start as u64);
                    }
                    (3..=7, Err(error)) => assert_eq!(error.code(), ErrorCode::IoError),
                    _ => panic!("unexpected directory result for seed {seed}, fault {fault}"),
                }
                context
                    .workspace_reservation()
                    .grow_to(available, available, "operation_scratch")
                    .unwrap();
                drop(blocker);
                context
                    .workspace_reservation()
                    .grow_to(total, total, "operation_scratch")
                    .unwrap();
            }
        }
    }

    fn directory_suffix_from_records(
        checkpoint: &IndexCheckpointPayload,
        records: &[BuiltIndexRecord],
    ) -> Vec<u8> {
        let mut extents = Vec::new();
        let mut referenced = Vec::new();
        let mut at = (INDEX_CHECKPOINT_HEADER_LEN
            + INDEX_RECORD_HEADER_LEN
            + INDEX_COMMIT_METADATA_LEN) as u64;
        for record in records {
            let extent = IndexExtentEntry {
                index_record_id: record.index_record_id,
                record_type: record.record_type,
                stream_id: record.stream_id,
                suffix_relative_offset: at,
                length: record.bytes.len() as u64,
                checkpoint_generation: 37,
                extent_record_checksum: checksum32(&record.bytes),
            };
            referenced.extend_from_slice(&encode_extent_entry(&extent));
            referenced.extend_from_slice(&record.bytes);
            at += record.bytes.len() as u64;
            extents.push(extent);
        }
        let table = encode_extent_table(&extents);
        let metadata = build_index_commit_metadata_payload(
            &checkpoint.metadata,
            records.len() as u64,
            INDEX_EXTENT_ENTRY_LEN as u64,
            table.len() as u64,
            checksum32(&table),
            index_commit_checksum(
                checkpoint.metadata.covered_last_batch_id,
                checkpoint.metadata.covered_last_commit_sequence_no,
                37,
                &referenced,
            ),
        );
        let mut payload = build_index_record(INDEX_COMMIT_METADATA, 37, 1, &metadata);
        extents.insert(
            0,
            IndexExtentEntry {
                index_record_id: 1,
                record_type: INDEX_COMMIT_METADATA,
                stream_id: 0,
                suffix_relative_offset: INDEX_CHECKPOINT_HEADER_LEN as u64,
                length: payload.len() as u64,
                checkpoint_generation: 37,
                extent_record_checksum: checksum32(&payload),
            },
        );
        for record in records {
            payload.extend_from_slice(&record.bytes);
        }
        let mut bytes = vec![7; 37];
        bytes.extend(checkpoint_suffix_bytes(
            checkpoint,
            37,
            37,
            extents.len() as u64,
            &payload,
            &encode_extent_table(&extents),
        ));
        bytes
    }

    #[test]
    fn checkpoint_record_admission_preserves_current_and_prior_known_damage() {
        use crate::json::{btree_workspace, JsonOperationContext, WorkLedger};
        use std::sync::Arc;
        for seed in 0..100u64 {
            let checkpoint = empty_directory_checkpoint(seed, 3);
            let table_bytes = 7 * INDEX_EXTENT_ENTRY_LEN;
            let first_node = btree_workspace::<u64, ParsedIndexRecord>(5);
            let next_node = btree_workspace::<u64, ParsedIndexRecord>(6);
            assert!(next_node > first_node);
            for fault in 0..7 {
                let mut records = build_index_segment_records(&checkpoint, 37).unwrap();
                for index in if fault == 4 {
                    vec![0, 4]
                } else if fault == 1 {
                    vec![4]
                } else if fault == 2 {
                    vec![0]
                } else {
                    vec![]
                } {
                    let record = &mut records[index];
                    let mut payload = record.bytes[INDEX_RECORD_HEADER_LEN..].to_vec();
                    *payload.last_mut().unwrap() ^= 1;
                    record.bytes = build_index_record(
                        record.record_type,
                        37,
                        record.index_record_id,
                        &payload,
                    );
                }
                if fault == 3 {
                    let old = &records[0];
                    records[4] = BuiltIndexRecord {
                        index_record_id: old.index_record_id,
                        record_type: old.record_type,
                        stream_id: old.stream_id,
                        bytes: old.bytes.clone(),
                    };
                }
                if fault == 5 {
                    *records[4].bytes.last_mut().unwrap() ^= 1;
                }
                let bytes = directory_suffix_from_records(&checkpoint, &records);
                let sixth_start = 37
                    + INDEX_CHECKPOINT_HEADER_LEN
                    + INDEX_RECORD_HEADER_LEN
                    + INDEX_COMMIT_METADATA_LEN
                    + records[..4]
                        .iter()
                        .map(|record| record.bytes.len())
                        .sum::<usize>();
                let mut input = DirectoryInput::new(&bytes);
                if fault == 6 {
                    input.fail_after_seek = Some(sixth_start as u64);
                }
                let limit = table_bytes + first_node;
                let context = JsonOperationContext::new(Arc::new(WorkLedger::with_limits(
                    limit, limit, limit,
                )));
                let parsed = read_index_checkpoint_suffix_in(
                    &mut input,
                    37,
                    bytes.len() as u64,
                    37,
                    seed,
                    seed + 1,
                    (
                        IndexReadOptions {
                            mode: IndexReadMode::Summary,
                            budget_bytes: u64::MAX,
                        },
                        &context,
                    ),
                );
                match (fault, parsed) {
                    (0, Err(error)) => {
                        assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                        assert!(error.detail().contains(&(
                            "requested_bytes".into(),
                            (table_bytes + next_node).to_string()
                        )));
                        assert_eq!(
                            input.bytes.position(),
                            (sixth_start + records[4].bytes.len()) as u64
                        );
                    }
                    (1 | 2 | 4, Ok(Err(diagnostic))) => {
                        assert_eq!(diagnostic.code, "index_segment_checksum_mismatch");
                        assert_eq!(
                            input.bytes.position(),
                            (sixth_start + records[4].bytes.len()) as u64
                        );
                    }
                    (3, Ok(Err(diagnostic))) => {
                        assert_eq!(diagnostic.code, "index_extent_table_mismatch");
                        assert!(diagnostic.message.contains("duplicate index record id"));
                    }
                    (5, Ok(Err(diagnostic))) => {
                        assert_eq!(diagnostic.code, "index_record_payload_checksum_mismatch")
                    }
                    (6, Err(error)) => assert_eq!(error.code(), ErrorCode::IoError),
                    _ => {
                        panic!("unexpected record admission result for seed {seed}, fault {fault}")
                    }
                }
                context
                    .workspace_reservation()
                    .grow_to(limit, limit, "operation_scratch")
                    .unwrap();
            }
        }
    }

    #[test]
    fn checkpoint_reference_iteration_preserves_sparse_ids_and_exact_directory() {
        use std::io::Cursor;
        for ids in [[0, 17], [2, u64::MAX], [17, 18], [u64::MAX - 1, u64::MAX]] {
            let checkpoint = empty_directory_checkpoint(7, 1);
            let mut records = build_index_segment_records(&checkpoint, 37).unwrap();
            for (record, id) in records.iter_mut().zip(ids) {
                record.index_record_id = id;
                let payload = record.bytes[INDEX_RECORD_HEADER_LEN..].to_vec();
                record.bytes = build_index_record(record.record_type, 37, id, &payload);
            }
            let bytes = directory_suffix_from_records(&checkpoint, &records);
            let parsed = read_index_checkpoint_suffix(
                &mut Cursor::new(&bytes),
                37,
                bytes.len() as u64,
                37,
                7,
                8,
                IndexReadOptions {
                    mode: IndexReadMode::Summary,
                    budget_bytes: u64::MAX,
                },
            )
            .unwrap()
            .unwrap();
            assert_eq!(parsed.report.collection_count, 1);
            for order in [[1, 0], [0, 0]] {
                let changed =
                    [&records[order[0]], &records[order[1]]].map(|record| BuiltIndexRecord {
                        index_record_id: record.index_record_id,
                        record_type: record.record_type,
                        stream_id: record.stream_id,
                        bytes: record.bytes.clone(),
                    });
                let bytes = directory_suffix_from_records(&checkpoint, &changed);
                let parsed = read_index_checkpoint_suffix(
                    &mut Cursor::new(&bytes),
                    37,
                    bytes.len() as u64,
                    37,
                    7,
                    8,
                    u64::MAX,
                )
                .unwrap()
                .err()
                .expect("non-canonical index references");
                assert!(matches!(
                    parsed.code,
                    "index_commit_metadata_invalid" | "index_extent_table_mismatch"
                ));
            }
        }
    }

    #[test]
    fn checkpoint_complete_marker_precedes_record_damage_and_resource_refusal() {
        use crate::json::{JsonOperationContext, WorkLedger};
        use std::io::Cursor;
        use std::sync::Arc;
        for seed in 0..100u64 {
            let checkpoint = empty_directory_checkpoint(seed, seed as usize % 5);
            let mut original = vec![7; 37];
            original.extend(build_index_checkpoint_suffix(&checkpoint, 37).unwrap());
            let table_start = 37 + read_u64(&original[37 + 73..37 + 81]) as usize;
            for corrupted_at in [37 + INDEX_CHECKPOINT_HEADER_LEN, table_start] {
                let mut bytes = original.clone();
                bytes[corrupted_at] ^= 1;
                *bytes.last_mut().unwrap() ^= 1;
                for limit in [0, 128 * 1024] {
                    let context = JsonOperationContext::new(Arc::new(WorkLedger::with_limits(
                        limit, limit, limit,
                    )));
                    let diagnostic = read_index_checkpoint_suffix_in(
                        &mut Cursor::new(&bytes),
                        37,
                        bytes.len() as u64,
                        37,
                        seed,
                        seed + 1,
                        (
                            IndexReadOptions {
                                mode: IndexReadMode::Summary,
                                budget_bytes: u64::MAX,
                            },
                            &context,
                        ),
                    )
                    .unwrap()
                    .err()
                    .expect("invalid complete marker");
                    assert_eq!(diagnostic.code, "index_checkpoint_incomplete");
                    assert_eq!(
                        diagnostic.offset,
                        bytes.len() as u64 - INDEX_CHECKPOINT_TRAILER_LEN as u64
                    );
                    context
                        .workspace_reservation()
                        .grow_to(limit, limit, "operation_scratch")
                        .unwrap();
                }
            }
            let mut input = DirectoryInput::new(&original);
            input.faults_enabled = false;
            input.enable_faults_at_seek = Some(37 + INDEX_CHECKPOINT_HEADER_LEN as u64);
            input.flip_read_at = Some(original.len() as u64 - 1);
            let limit = 128 * 1024;
            let context =
                JsonOperationContext::new(Arc::new(WorkLedger::with_limits(limit, limit, limit)));
            let diagnostic = read_index_checkpoint_suffix_in(
                &mut input,
                37,
                original.len() as u64,
                37,
                seed,
                seed + 1,
                (
                    IndexReadOptions {
                        mode: IndexReadMode::Summary,
                        budget_bytes: u64::MAX,
                    },
                    &context,
                ),
            )
            .unwrap()
            .err()
            .expect("trailer changed after its initial validation");
            assert_eq!(diagnostic.code, "index_checkpoint_incomplete");
            assert_eq!(
                diagnostic.offset,
                original.len() as u64 - INDEX_CHECKPOINT_TRAILER_LEN as u64
            );
            assert_eq!(input.bytes.position(), original.len() as u64);
            context
                .workspace_reservation()
                .grow_to(limit, limit, "operation_scratch")
                .unwrap();
        }
    }

    #[test]
    fn summary_text_decoding_matches_retained_validation_across_byte_mutations() {
        fn compare(bytes: &[u8], seed: u64) {
            let context = crate::json::JsonOperationContext::standalone();
            let decode = |retain| {
                decode_text_index_segment_in(
                    &mut IndexPayloadReader::new(bytes, bytes.len(), 37),
                    seed,
                    37,
                    retain,
                    &context,
                )
            };
            match (decode(true), decode(false)) {
                (Ok(retained), Ok(summary)) => {
                    assert_eq!(retained.collection_number, summary.collection_number);
                    assert_eq!(retained.chunk_count, summary.chunk_count);
                    assert_eq!(
                        retained.document_ids.as_slice(),
                        summary.document_ids.as_slice()
                    );
                    assert_eq!(summary.payload.chunks.capacity(), 0);
                    assert_eq!(summary.payload.terms.capacity(), 0);
                    let encoded = encode_text_index_segment_payload(&retained.payload).unwrap();
                    assert_eq!(&encoded[..encoded.len() - 4], bytes);
                }
                (Err(IndexReadError::Invalid(left)), Err(IndexReadError::Invalid(right))) => {
                    assert_eq!(left.code, right.code);
                    assert_eq!(left.message, right.message);
                    assert_eq!(left.offset, right.offset);
                    assert_eq!(left.extent_id, right.extent_id);
                }
                (Err(IndexReadError::Io(left)), Err(IndexReadError::Io(right))) => {
                    assert_eq!(left.kind(), right.kind());
                }
                _ => panic!("text decoder retention changed validation for seed {seed}"),
            }
            let limit = crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES;
            context
                .workspace_reservation()
                .grow_to(limit, limit, "operation_scratch")
                .unwrap();
        }

        for seed in 0..100u64 {
            let count = seed as usize % 7;
            let segment = TextIndexSegmentPayload {
                collection_id: format!("col_{seed}"),
                covered_commit_sequence_no: seed,
                chunks: (0..count)
                    .map(|index| TextIndexChunkEntry {
                        chunk_id: format!("chk_{}", index * 3 + 1),
                        document_id: format!("doc_{}", index % 2),
                        token_count: index + 1,
                    })
                    .collect(),
                terms: (0..seed as usize % 5)
                    .map(|index| TextIndexTermEntry {
                        term: format!("term-{index}-\u{4e2d}\u{6587}"),
                        postings: (0..count)
                            .map(|chunk| TextIndexPostingEntry {
                                chunk_id: format!("chk_{}", chunk * 3 + 1),
                                term_frequency: chunk + index + 1,
                            })
                            .collect(),
                    })
                    .collect(),
            };
            let mut bytes = encode_text_index_segment_payload(&segment).unwrap();
            bytes.truncate(bytes.len() - 4);
            compare(&bytes, seed);
            for index in 0..bytes.len() {
                bytes[index] ^= 0x80;
                compare(&bytes, seed);
                bytes[index] ^= 0x80;
                compare(&bytes[..index], seed);
            }
        }
    }

    #[test]
    fn streamed_index_strings_keep_utf8_and_bound_each_read() {
        struct Limited<'a> {
            bytes: &'a [u8],
            max_request: usize,
            calls: usize,
        }
        impl Read for Limited<'_> {
            fn read(&mut self, output: &mut [u8]) -> std::io::Result<usize> {
                assert!(output.len() <= 64 * 1024);
                self.max_request = self.max_request.max(output.len());
                self.calls += 1;
                if self.calls % 11 == 0 {
                    return Err(std::io::ErrorKind::Interrupted.into());
                }
                let count = output.len().min(4093);
                self.bytes.read(&mut output[..count])
            }
        }
        for prefix in [65_534, 65_535, 65_536] {
            let term = format!("{}\u{4e2d}\u{6587}", "x".repeat(prefix));
            let segment = TextIndexSegmentPayload {
                collection_id: "col_1".into(),
                covered_commit_sequence_no: 2,
                chunks: vec![TextIndexChunkEntry {
                    chunk_id: "chk_1".into(),
                    document_id: "doc_1".into(),
                    token_count: 1,
                }],
                terms: vec![TextIndexTermEntry {
                    term: term.clone(),
                    postings: vec![TextIndexPostingEntry {
                        chunk_id: "chk_1".into(),
                        term_frequency: 1,
                    }],
                }],
            };
            let encoded = encode_text_index_segment_payload(&segment).unwrap();
            let body_len = encoded.len() - 4;
            let mut source = Limited {
                bytes: &encoded,
                max_request: 0,
                calls: 0,
            };
            let decoded = decode_text_index_segment(
                &mut IndexPayloadReader::new(&mut source, body_len, 0),
                2,
                0,
            )
            .unwrap();
            assert_eq!(decoded.payload.terms[0].term, term);
            assert_eq!(source.bytes, &encoded[body_len..]);
            assert_eq!(source.max_request, 64 * 1024);
            let mut streamed = Vec::new();
            checkpoint_writer::write_text_segment(&segment, &mut streamed).unwrap();
            assert_eq!(streamed, encoded);
        }
    }

    #[test]
    fn text_segment_checks_frozen_tokenizer_parameters_and_legacy_schema() {
        let segment = TextIndexSegmentPayload {
            collection_id: "col_1".into(),
            covered_commit_sequence_no: 2,
            chunks: vec![],
            terms: vec![],
        };
        let original = encode_text_index_segment_payload(&segment).unwrap();
        assert!(parse_text_index_segment_payload(&original, 2, 0).is_ok());
        for (offset, value) in [(0, 1u8), (20, 2), (22, 0)] {
            let mut payload = original.clone();
            payload[offset] = value;
            let end = payload.len() - 4;
            let checksum = checksum32(&payload[..end]);
            payload[end..].copy_from_slice(&checksum.to_le_bytes());
            assert!(parse_text_index_segment_payload(&payload, 2, 0).is_err());
        }
    }

    #[test]
    #[cfg(unix)]
    fn payload_reader_does_not_move_shared_writer_cursor() {
        let path = temp_path("positional-reader");
        let mut storage = Storage::create(&path).unwrap();
        let append = storage
            .append_batch(&[collection_record("col_1", "notes")], Durability::Commit)
            .unwrap();
        let before = storage.file.stream_position().unwrap();
        let mut reader = storage.payload_reader().unwrap();
        assert!(matches!(
            reader.read_payload(&append.records[0].record_ref).unwrap(),
            RecordPayload::CollectionCreate(_)
        ));
        assert_eq!(storage.file.stream_position().unwrap(), before);
        drop(storage);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn check_issue_json_exposes_public_repair_aliases() {
        let issue = CheckIssue {
            severity: "warning".into(),
            code: "corrupt_tail".into(),
            message: "ignored incomplete tail".into(),
            stream: "critical".into(),
            visibility_impact: "tail_only".into(),
            offset: Some(42),
            extent_id: None,
            record_type: None,
            batch_id: None,
            repair_available: true,
            suggested_command: Some("coffer repair demo.coffer --truncate-tail".into()),
        };
        let json = issue.to_json();
        assert_eq!(
            json.get("repair_available").and_then(JsonValue::as_bool),
            Some(true)
        );
        assert_eq!(
            json.get("repairable").and_then(JsonValue::as_bool),
            Some(true)
        );
        assert!(matches!(json.get("extent_id"), Some(JsonValue::Null)));
        assert_eq!(
            json.get("suggested_command").and_then(JsonValue::as_str),
            Some("coffer repair demo.coffer --truncate-tail")
        );
    }

    fn checkpoint_for(append: &AppendSummary) -> IndexCheckpointPayload {
        IndexCheckpointPayload {
            metadata: IndexCheckpointMetadata {
                covered_last_batch_id: append.commit.batch_id,
                covered_last_commit_sequence_no: append.commit_sequence_no,
                coverage_hash: 42,
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
        }
    }

    #[test]
    fn checkpoint_budget_is_checked_before_payload_is_read() {
        let path = temp_path("checkpoint-read-budget");
        let mut storage = Storage::create(&path).unwrap();
        let append = storage
            .append_batch(&[collection_record("col_1", "notes")], Durability::Commit)
            .unwrap();
        let offset = storage.critical_end_offset();
        let suffix = build_index_checkpoint_suffix(&full_checkpoint_for(&append), offset).unwrap();
        let mut header_only = std::io::Cursor::new(vec![0; offset as usize]);
        header_only
            .get_mut()
            .extend_from_slice(&suffix[..INDEX_CHECKPOINT_HEADER_LEN]);
        let result = read_index_checkpoint_suffix(
            &mut header_only,
            offset,
            offset + suffix.len() as u64,
            offset,
            append.commit.batch_id,
            append.commit_sequence_no,
            0,
        )
        .unwrap();
        assert!(
            matches!(result, Err(diagnostic) if diagnostic.code == "index_checkpoint_load_budget_exceeded")
        );
        assert_eq!(
            header_only.position(),
            offset + INDEX_CHECKPOINT_HEADER_LEN as u64
        );
        drop(storage);
        let _ = fs::remove_file(path);
    }

    #[test]
    fn streamed_checkpoint_read_bounds_io_and_rechecks_decoded_payloads() {
        struct Stream {
            bytes: std::io::Cursor<Vec<u8>>,
            reads: usize,
            seeks: usize,
            fail_read: usize,
            fail_seek: usize,
            eof: bool,
            mutation: Option<(u64, usize)>,
        }
        impl Stream {
            fn new(bytes: Vec<u8>) -> Self {
                Self {
                    bytes: std::io::Cursor::new(bytes),
                    reads: 0,
                    seeks: 0,
                    fail_read: 0,
                    fail_seek: 0,
                    eof: false,
                    mutation: None,
                }
            }
        }
        impl Read for Stream {
            fn read(&mut self, output: &mut [u8]) -> std::io::Result<usize> {
                assert!(output.len() <= 64 * 1024);
                self.reads += 1;
                if self.reads == self.fail_read {
                    return if self.eof {
                        Ok(0)
                    } else {
                        Err(std::io::Error::other("injected checkpoint read error"))
                    };
                }
                if self.reads % 11 == 0 {
                    return Err(std::io::ErrorKind::Interrupted.into());
                }
                let count = output.len().min(8191);
                self.bytes.read(&mut output[..count])
            }
        }
        impl Seek for Stream {
            fn seek(&mut self, position: SeekFrom) -> std::io::Result<u64> {
                self.seeks += 1;
                if self.seeks == self.fail_seek {
                    return Err(std::io::Error::other("injected checkpoint seek error"));
                }
                if let Some((trigger, byte)) = self.mutation {
                    if position == SeekFrom::Start(trigger) {
                        self.bytes.get_mut()[byte] ^= 1;
                        self.mutation = None;
                    }
                }
                self.bytes.seek(position)
            }
        }
        let append = AppendSummary {
            commit: CommitPayload {
                batch_id: 1,
                first_sequence_no: 1,
                last_sequence_no: 1,
                record_count: 1,
                batch_checksum: 0,
                committed_at: String::new(),
            },
            commit_sequence_no: 2,
            records: Vec::new(),
            _commit_work: None,
        };
        let mut checkpoint = full_checkpoint_for(&append);
        checkpoint.text_segments[0].terms[0].term = "a".repeat(65_537);
        checkpoint.vector_segments[0].dim = 1025;
        checkpoint.vector_segments[0].vectors[0].vector = vec![-0.0; 1025];
        let mut bytes = vec![7; 37];
        bytes.extend_from_slice(&build_index_checkpoint_suffix(&checkpoint, 37).unwrap());
        let file_len = bytes.len() as u64;
        for retain_segments in [true, false] {
            let load = |stream: &mut Stream| {
                read_index_checkpoint_suffix(
                    stream,
                    37,
                    file_len,
                    37,
                    1,
                    2,
                    IndexReadOptions {
                        budget_bytes: u64::MAX,
                        mode: if retain_segments {
                            IndexReadMode::Full
                        } else {
                            IndexReadMode::Summary
                        },
                    },
                )
            };
            let mut success = Stream::new(bytes.clone());
            let parsed = load(&mut success).unwrap().unwrap();
            if let Some(checkpoint) = &parsed.checkpoint {
                assert_eq!(checkpoint.text_segments[0].terms[0].term.len(), 65_537);
                assert!(checkpoint.vector_segments[0].vectors[0]
                    .vector
                    .iter()
                    .all(|value| value.to_bits() == (-0.0f32).to_bits()));
            }
            assert_eq!(parsed.checkpoint.is_some(), retain_segments);
            assert_eq!(success.bytes.position(), file_len);
            for read in 1..=success.reads {
                let mut stream = Stream::new(bytes.clone());
                stream.fail_read = read;
                assert!(
                    matches!(load(&mut stream), Err(error) if error.code() == ErrorCode::IoError),
                    "read {read}"
                );
                if read > 1 {
                    let mut stream = Stream::new(bytes.clone());
                    stream.fail_read = read;
                    stream.eof = true;
                    assert!(
                        matches!(load(&mut stream), Err(error) if error.code() == ErrorCode::IoError),
                        "EOF {read}"
                    );
                }
            }
            for seek in 1..=success.seeks {
                let mut stream = Stream::new(bytes.clone());
                stream.fail_seek = seek;
                assert!(
                    matches!(load(&mut stream), Err(error) if error.code() == ErrorCode::IoError),
                    "seek {seek}"
                );
            }
            let table_offset = 37 + read_u64(&bytes[37 + 73..37 + 81]) as usize;
            let table_len = read_u64(&bytes[37 + 81..37 + 89]) as usize;
            for extent in
                parse_extent_table(&bytes[table_offset..table_offset + table_len]).unwrap()
            {
                let payload_offset =
                    37 + extent.suffix_relative_offset + INDEX_RECORD_HEADER_LEN as u64;
                let payload_len = extent.length - INDEX_RECORD_HEADER_LEN as u64;
                // The header mutation creates a semantic error; the checksum must win.
                // The tail mutation is observed only when draining after decoding.
                for byte in [payload_offset, payload_offset + payload_len - 1] {
                    let mut stream = Stream::new(bytes.clone());
                    stream.mutation = Some((payload_offset, byte as usize));
                    let error = load(&mut stream).unwrap().err().unwrap();
                    assert!(stream.mutation.is_none());
                    assert_eq!(error.code, "index_record_payload_checksum_mismatch");
                    assert_eq!(error.offset, payload_offset);
                    assert_eq!(
                        error.extent_id,
                        Some(format!("index_record:{}", extent.index_record_id))
                    );
                }
            }
            assert_eq!(&bytes[..37], &[7; 37]);
        }
    }

    #[test]
    fn checkpoint_directory_admission_precedes_truncation_and_survives_publication() {
        use crate::json::{JsonOperationContext, WorkLedger};
        use std::borrow::Cow;
        use std::sync::Arc;
        let path = temp_path("checkpoint-directory-admission");
        let mut storage = Storage::create(&path).unwrap();
        let append = storage
            .append_batch(&[collection_record("col_1", "notes")], Durability::Fast)
            .unwrap();
        let checkpoint = full_checkpoint_for(&append);
        storage
            .write_index_checkpoint(&checkpoint, Durability::Fast)
            .unwrap();
        let original = fs::read(&path).unwrap();
        let critical = storage.critical_end_offset;
        let needed = 3 * std::mem::size_of::<IndexExtentEntry>();
        let segments = || {
            [
                Ok(CheckpointSegment::Text(Cow::Borrowed(
                    &checkpoint.text_segments[0],
                ))),
                Ok(CheckpointSegment::Vector(Cow::Borrowed(
                    &checkpoint.vector_segments[0],
                ))),
            ]
        };
        for available in [needed - 1, needed, needed + 1] {
            let context = JsonOperationContext::new(Arc::new(WorkLedger::with_limits(
                available, available, available,
            )));
            if available < needed {
                let error = storage
                    .write_index_segments_in(
                        &checkpoint.metadata,
                        std::iter::from_fn(|| panic!("directory rejection must not poll payloads")),
                        |_, _| {},
                        Durability::Fast,
                        &context,
                        true,
                    )
                    .unwrap_err();
                assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                assert!(error
                    .detail()
                    .contains(&("requested_bytes".into(), needed.to_string())));
            } else {
                let (_, source) = storage
                    .write_index_segments_in(
                        &checkpoint.metadata,
                        segments(),
                        |_, _| {},
                        Durability::Fast,
                        &context,
                        true,
                    )
                    .unwrap();
                assert!(source.encoded_len("col_1").is_some());
                let mut remaining = context.workspace_reservation();
                remaining
                    .grow_to(available - needed, available - needed, "operation_scratch")
                    .unwrap();
                assert!(context
                    .workspace_reservation()
                    .grow_to(1, 1, "operation_scratch")
                    .is_err());
                drop(remaining);
                drop(source);
            }
            assert_eq!(fs::read(&path).unwrap(), original);
            assert_eq!(storage.critical_end_offset, critical);
            context
                .workspace_reservation()
                .grow_to(available, available, "operation_scratch")
                .unwrap();
        }
        let context =
            JsonOperationContext::new(Arc::new(WorkLedger::with_limits(needed, needed, needed)));
        let directory =
            checkpoint_writer::reserve_directory(&checkpoint.metadata, &context).unwrap();
        let mut prefix = [7u8; 37];
        let error = checkpoint_writer::write_checkpoint_segments_in(
            &mut std::io::Cursor::new(&mut prefix[..]),
            &checkpoint.metadata,
            segments(),
            37,
            |_, _| {},
            directory,
            true,
        )
        .unwrap_err();
        assert_eq!(error.code(), ErrorCode::IoError);
        assert_eq!(prefix, [7; 37]);
        context
            .workspace_reservation()
            .grow_to(needed, needed, "operation_scratch")
            .unwrap();
        for missing in 1..=INDEX_CHECKPOINT_TRAILER_LEN {
            let mut bytes = original[..original.len() - missing].to_vec();
            let directory =
                checkpoint_writer::reserve_directory(&checkpoint.metadata, &context).unwrap();
            let mut output = std::io::Cursor::new(&mut bytes[..]);
            let error = checkpoint_writer::write_checkpoint_segments_in(
                &mut output,
                &checkpoint.metadata,
                segments(),
                critical,
                |_, _| {},
                directory,
                true,
            )
            .unwrap_err();
            assert_eq!(error.code(), ErrorCode::IoError);
            assert_eq!(output.position(), (original.len() - missing) as u64);
            assert_eq!(bytes, original[..original.len() - missing]);
            assert_eq!(fs::read(&path).unwrap(), original);
            context
                .workspace_reservation()
                .grow_to(needed, needed, "operation_scratch")
                .unwrap();
        }
        drop(storage);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn read_snapshot_directory_keeps_the_shared_ledger_until_its_owner_drops() {
        use crate::json::{JsonOperationContext, WorkLedger};
        use std::io::Cursor;
        use std::sync::Arc;
        let path = temp_path("read-directory-ownership");
        let mut storage = Storage::create(&path).unwrap();
        let append = storage
            .append_batch(&[collection_record("col_1", "notes")], Durability::Fast)
            .unwrap();
        let checkpoint = full_checkpoint_for(&append);
        let offset = storage.critical_end_offset;
        let bytes = build_index_checkpoint_suffix(&checkpoint, offset).unwrap();
        let mut complete = fs::read(&path).unwrap();
        complete.extend_from_slice(&bytes);
        let needed = 2 * std::mem::size_of::<IndexExtentEntry>();
        let total = 128 * 1024;
        for slack in [0, 1, 4096] {
            let ledger = Arc::new(WorkLedger::with_limits(total, total, total));
            let context = JsonOperationContext::new(ledger.clone());
            let other = JsonOperationContext::new(ledger);
            let mut blocker = other.workspace_reservation();
            blocker.grow_to(slack, slack, "operation_scratch").unwrap();
            let parsed = read_index_checkpoint_suffix_in(
                &mut Cursor::new(&complete),
                offset,
                complete.len() as u64,
                offset,
                append.commit.batch_id,
                append.commit_sequence_no,
                (
                    IndexReadOptions {
                        mode: IndexReadMode::LazyVectors,
                        budget_bytes: u64::MAX,
                    },
                    &context,
                ),
            )
            .unwrap()
            .unwrap();
            let source = parsed.lazy_vectors.unwrap().vectors;
            assert_eq!(source.reserved_bytes(), needed);
            assert!(source.encoded_len("col_1").is_some());
            drop(context);
            let mut remaining = other.workspace_reservation();
            remaining
                .grow_to(
                    total - slack - needed,
                    total - slack - needed,
                    "operation_scratch",
                )
                .unwrap();
            assert_eq!(
                other
                    .workspace_reservation()
                    .grow_to(1, 1, "operation_scratch")
                    .unwrap_err()
                    .code(),
                ErrorCode::ResourceExhausted
            );
            drop(source);
            remaining
                .grow_to(total - slack, total - slack, "operation_scratch")
                .unwrap();
            drop((remaining, blocker));
            other
                .workspace_reservation()
                .grow_to(total, total, "operation_scratch")
                .unwrap();
        }
        drop(storage);
        fs::remove_file(path).unwrap();
    }

    fn full_checkpoint_for(append: &AppendSummary) -> IndexCheckpointPayload {
        IndexCheckpointPayload {
            metadata: IndexCheckpointMetadata {
                covered_last_batch_id: append.commit.batch_id,
                covered_last_commit_sequence_no: append.commit_sequence_no,
                coverage_hash: 42,
                collection_count: 1,
                live_document_count: 1,
                live_chunk_count: 1,
                live_embedding_count: 1,
                text_segment_count: 1,
                vector_segment_count: 1,
                build_mode: "inline_full_checkpoint".into(),
            },
            text_segments: vec![TextIndexSegmentPayload {
                collection_id: "col_1".into(),
                covered_commit_sequence_no: append.commit_sequence_no,
                chunks: vec![TextIndexChunkEntry {
                    chunk_id: "chk_1".into(),
                    document_id: "doc_1".into(),
                    token_count: 2,
                }],
                terms: vec![TextIndexTermEntry {
                    term: "wal".into(),
                    postings: vec![TextIndexPostingEntry {
                        chunk_id: "chk_1".into(),
                        term_frequency: 1,
                    }],
                }],
            }],
            vector_segments: vec![VectorIndexSegmentPayload {
                collection_id: "col_1".into(),
                covered_commit_sequence_no: append.commit_sequence_no,
                dim: 3,
                distance_metric: DistanceMetric::Cosine,
                embedding_model: None,
                vectors_normalized: false,
                vectors: vec![VectorIndexEntry {
                    chunk_id: "chk_1".into(),
                    embedding_id: "emb_1".into(),
                    vector: vec![1.0, 0.0, 0.0],
                }],
            }],
        }
    }

    fn append_malformed_checkpoint_header(
        path: &Path,
        scan: &ScanReport,
        payload_len: u64,
        extent_table_len: u64,
    ) {
        let extent_table_offset = (INDEX_CHECKPOINT_HEADER_LEN as u64)
            .checked_add(payload_len)
            .unwrap_or(0);
        let mut header = Vec::with_capacity(INDEX_CHECKPOINT_HEADER_LEN);
        header.extend_from_slice(INDEX_CHECKPOINT_MAGIC);
        header.extend_from_slice(&INDEX_CHECKPOINT_VERSION.to_le_bytes());
        header.extend_from_slice(&(INDEX_CHECKPOINT_HEADER_LEN as u16).to_le_bytes());
        header.extend_from_slice(&scan.critical_end_offset.to_le_bytes());
        header.extend_from_slice(&scan.critical_end_offset.to_le_bytes());
        header.extend_from_slice(&scan.critical_end_offset.to_le_bytes());
        header.extend_from_slice(&scan.last_valid_commit.to_le_bytes());
        header.extend_from_slice(&scan.last_valid_commit_sequence_no.to_le_bytes());
        header.push(INDEX_COVERAGE_FULL_VISIBLE_SNAPSHOT);
        header.extend_from_slice(&0u32.to_le_bytes());
        header.extend_from_slice(&1u64.to_le_bytes());
        header.extend_from_slice(&1u64.to_le_bytes());
        header.extend_from_slice(&extent_table_offset.to_le_bytes());
        header.extend_from_slice(&extent_table_len.to_le_bytes());
        header.extend_from_slice(&checksum32(&[]).to_le_bytes());
        header.extend_from_slice(&payload_len.to_le_bytes());
        header.extend_from_slice(&checksum32(&[]).to_le_bytes());
        assert_eq!(header.len(), INDEX_CHECKPOINT_HEADER_LEN);
        let mut file = OpenOptions::new().append(true).open(path).unwrap();
        file.write_all(&header).unwrap();
    }

    fn append_checkpoint_with_overflowing_index_record_payload_len(path: &Path, scan: &ScanReport) {
        let mut record_header = Vec::with_capacity(INDEX_RECORD_HEADER_LEN);
        record_header.extend_from_slice(INDEX_RECORD_MAGIC);
        record_header.extend_from_slice(&INDEX_COMMIT_METADATA.to_le_bytes());
        record_header.extend_from_slice(&1u16.to_le_bytes());
        record_header.extend_from_slice(&(INDEX_RECORD_HEADER_LEN as u16).to_le_bytes());
        record_header.extend_from_slice(&(INDEX_RECORD_HEADER_LEN as u64).to_le_bytes());
        record_header.extend_from_slice(&scan.critical_end_offset.to_le_bytes());
        record_header.extend_from_slice(&1u64.to_le_bytes());
        record_header.extend_from_slice(&u64::MAX.to_le_bytes());
        record_header.extend_from_slice(&checksum32(&[]).to_le_bytes());
        let header_checksum = checksum32(&record_header);
        record_header.extend_from_slice(&header_checksum.to_le_bytes());
        assert_eq!(record_header.len(), INDEX_RECORD_HEADER_LEN);

        let extent = IndexExtentEntry {
            index_record_id: 1,
            record_type: INDEX_COMMIT_METADATA,
            stream_id: 0,
            suffix_relative_offset: INDEX_CHECKPOINT_HEADER_LEN as u64,
            length: INDEX_RECORD_HEADER_LEN as u64,
            checkpoint_generation: scan.critical_end_offset,
            extent_record_checksum: checksum32(&record_header),
        };
        let extent_table = encode_extent_table(&[extent]);
        let mut header = Vec::with_capacity(INDEX_CHECKPOINT_HEADER_LEN);
        header.extend_from_slice(INDEX_CHECKPOINT_MAGIC);
        header.extend_from_slice(&INDEX_CHECKPOINT_VERSION.to_le_bytes());
        header.extend_from_slice(&(INDEX_CHECKPOINT_HEADER_LEN as u16).to_le_bytes());
        header.extend_from_slice(&scan.critical_end_offset.to_le_bytes());
        header.extend_from_slice(&scan.critical_end_offset.to_le_bytes());
        header.extend_from_slice(&scan.critical_end_offset.to_le_bytes());
        header.extend_from_slice(&scan.last_valid_commit.to_le_bytes());
        header.extend_from_slice(&scan.last_valid_commit_sequence_no.to_le_bytes());
        header.push(INDEX_COVERAGE_FULL_VISIBLE_SNAPSHOT);
        header.extend_from_slice(&0u32.to_le_bytes());
        header.extend_from_slice(&1u64.to_le_bytes());
        header.extend_from_slice(&1u64.to_le_bytes());
        header.extend_from_slice(
            &(INDEX_CHECKPOINT_HEADER_LEN as u64 + record_header.len() as u64).to_le_bytes(),
        );
        header.extend_from_slice(&(extent_table.len() as u64).to_le_bytes());
        header.extend_from_slice(&checksum32(&extent_table).to_le_bytes());
        header.extend_from_slice(&(record_header.len() as u64).to_le_bytes());
        header.extend_from_slice(&checksum32(&record_header).to_le_bytes());

        let mut trailer = Vec::with_capacity(INDEX_CHECKPOINT_TRAILER_LEN);
        trailer.extend_from_slice(INDEX_COMPLETE_MARKER);
        trailer.extend_from_slice(&(INDEX_CHECKPOINT_TRAILER_LEN as u16).to_le_bytes());
        trailer.extend_from_slice(&checksum32(&trailer).to_le_bytes());

        let mut file = OpenOptions::new().append(true).open(path).unwrap();
        file.write_all(&header).unwrap();
        file.write_all(&record_header).unwrap();
        file.write_all(&extent_table).unwrap();
        file.write_all(&trailer).unwrap();
    }

    fn append_checkpoint_with_unreferenced_unknown_record(
        path: &Path,
        append: &AppendSummary,
        critical_end_offset: u64,
    ) {
        let checkpoint = full_checkpoint_for(append);
        let checkpoint_generation = critical_end_offset;
        let text_payload = encode_text_index_segment_payload(&checkpoint.text_segments[0]).unwrap();
        let vector_payload =
            encode_vector_index_segment_payload(&checkpoint.vector_segments[0]).unwrap();
        let text_record =
            build_index_record(INDEX_TEXT_SEGMENT, checkpoint_generation, 2, &text_payload);
        let vector_record = build_index_record(
            INDEX_VECTOR_SEGMENT,
            checkpoint_generation,
            3,
            &vector_payload,
        );
        let unknown_record = build_index_record(0x8fff, checkpoint_generation, 4, &[]);

        let metadata_record_len = (INDEX_RECORD_HEADER_LEN + INDEX_COMMIT_METADATA_LEN) as u64;
        let text_extent = IndexExtentEntry {
            index_record_id: 2,
            record_type: INDEX_TEXT_SEGMENT,
            stream_id: 1,
            suffix_relative_offset: INDEX_CHECKPOINT_HEADER_LEN as u64 + metadata_record_len,
            length: text_record.len() as u64,
            checkpoint_generation,
            extent_record_checksum: checksum32(&text_record),
        };
        let vector_extent = IndexExtentEntry {
            index_record_id: 3,
            record_type: INDEX_VECTOR_SEGMENT,
            stream_id: 1,
            suffix_relative_offset: text_extent.suffix_relative_offset + text_record.len() as u64,
            length: vector_record.len() as u64,
            checkpoint_generation,
            extent_record_checksum: checksum32(&vector_record),
        };
        let unknown_extent = IndexExtentEntry {
            index_record_id: 4,
            record_type: 0x8fff,
            stream_id: 1,
            suffix_relative_offset: vector_extent.suffix_relative_offset
                + vector_record.len() as u64,
            length: unknown_record.len() as u64,
            checkpoint_generation,
            extent_record_checksum: checksum32(&unknown_record),
        };
        let referenced_extent_table = encode_extent_table(&[text_extent, vector_extent]);
        let mut referenced_extent_and_record_bytes = Vec::new();
        referenced_extent_and_record_bytes.extend_from_slice(&encode_extent_entry(&text_extent));
        referenced_extent_and_record_bytes.extend_from_slice(&text_record);
        referenced_extent_and_record_bytes.extend_from_slice(&encode_extent_entry(&vector_extent));
        referenced_extent_and_record_bytes.extend_from_slice(&vector_record);
        let metadata_payload = build_index_commit_metadata_payload(
            &checkpoint.metadata,
            2,
            INDEX_EXTENT_ENTRY_LEN as u64,
            referenced_extent_table.len() as u64,
            checksum32(&referenced_extent_table),
            index_commit_checksum(
                checkpoint.metadata.covered_last_batch_id,
                checkpoint.metadata.covered_last_commit_sequence_no,
                checkpoint_generation,
                &referenced_extent_and_record_bytes,
            ),
        );
        let metadata_record = build_index_record(
            INDEX_COMMIT_METADATA,
            checkpoint_generation,
            1,
            &metadata_payload,
        );
        let metadata_extent = IndexExtentEntry {
            index_record_id: 1,
            record_type: INDEX_COMMIT_METADATA,
            stream_id: 0,
            suffix_relative_offset: INDEX_CHECKPOINT_HEADER_LEN as u64,
            length: metadata_record.len() as u64,
            checkpoint_generation,
            extent_record_checksum: checksum32(&metadata_record),
        };
        let mut payload = Vec::new();
        payload.extend_from_slice(&metadata_record);
        payload.extend_from_slice(&text_record);
        payload.extend_from_slice(&vector_record);
        payload.extend_from_slice(&unknown_record);
        let extents = [metadata_extent, text_extent, vector_extent, unknown_extent];
        let extent_table = encode_extent_table(&extents);
        let suffix = checkpoint_suffix_bytes(
            &checkpoint,
            checkpoint_generation,
            critical_end_offset,
            4,
            &payload,
            &extent_table,
        );
        let mut file = OpenOptions::new().write(true).open(path).unwrap();
        file.set_len(critical_end_offset).unwrap();
        file.seek(SeekFrom::Start(critical_end_offset)).unwrap();
        file.write_all(&suffix).unwrap();
    }

    fn append_checkpoint_with_text_stream_id_mismatch(
        path: &Path,
        append: &AppendSummary,
        critical_end_offset: u64,
    ) {
        let checkpoint = full_checkpoint_for(append);
        let checkpoint_generation = critical_end_offset;
        let text_payload = encode_text_index_segment_payload(&checkpoint.text_segments[0]).unwrap();
        let vector_payload =
            encode_vector_index_segment_payload(&checkpoint.vector_segments[0]).unwrap();
        let text_record =
            build_index_record(INDEX_TEXT_SEGMENT, checkpoint_generation, 2, &text_payload);
        let vector_record = build_index_record(
            INDEX_VECTOR_SEGMENT,
            checkpoint_generation,
            3,
            &vector_payload,
        );
        let metadata_record_len = (INDEX_RECORD_HEADER_LEN + INDEX_COMMIT_METADATA_LEN) as u64;
        let text_extent = IndexExtentEntry {
            index_record_id: 2,
            record_type: INDEX_TEXT_SEGMENT,
            stream_id: 2,
            suffix_relative_offset: INDEX_CHECKPOINT_HEADER_LEN as u64 + metadata_record_len,
            length: text_record.len() as u64,
            checkpoint_generation,
            extent_record_checksum: checksum32(&text_record),
        };
        let vector_extent = IndexExtentEntry {
            index_record_id: 3,
            record_type: INDEX_VECTOR_SEGMENT,
            stream_id: 1,
            suffix_relative_offset: text_extent.suffix_relative_offset + text_record.len() as u64,
            length: vector_record.len() as u64,
            checkpoint_generation,
            extent_record_checksum: checksum32(&vector_record),
        };
        let referenced_extent_table = encode_extent_table(&[text_extent, vector_extent]);
        let mut referenced_extent_and_record_bytes = Vec::new();
        referenced_extent_and_record_bytes.extend_from_slice(&encode_extent_entry(&text_extent));
        referenced_extent_and_record_bytes.extend_from_slice(&text_record);
        referenced_extent_and_record_bytes.extend_from_slice(&encode_extent_entry(&vector_extent));
        referenced_extent_and_record_bytes.extend_from_slice(&vector_record);
        let metadata_payload = build_index_commit_metadata_payload(
            &checkpoint.metadata,
            2,
            INDEX_EXTENT_ENTRY_LEN as u64,
            referenced_extent_table.len() as u64,
            checksum32(&referenced_extent_table),
            index_commit_checksum(
                checkpoint.metadata.covered_last_batch_id,
                checkpoint.metadata.covered_last_commit_sequence_no,
                checkpoint_generation,
                &referenced_extent_and_record_bytes,
            ),
        );
        let metadata_record = build_index_record(
            INDEX_COMMIT_METADATA,
            checkpoint_generation,
            1,
            &metadata_payload,
        );
        let metadata_extent = IndexExtentEntry {
            index_record_id: 1,
            record_type: INDEX_COMMIT_METADATA,
            stream_id: 0,
            suffix_relative_offset: INDEX_CHECKPOINT_HEADER_LEN as u64,
            length: metadata_record.len() as u64,
            checkpoint_generation,
            extent_record_checksum: checksum32(&metadata_record),
        };
        let mut payload = Vec::new();
        payload.extend_from_slice(&metadata_record);
        payload.extend_from_slice(&text_record);
        payload.extend_from_slice(&vector_record);
        let extents = [metadata_extent, text_extent, vector_extent];
        let extent_table = encode_extent_table(&extents);
        let suffix = checkpoint_suffix_bytes(
            &checkpoint,
            checkpoint_generation,
            critical_end_offset,
            3,
            &payload,
            &extent_table,
        );
        let mut file = OpenOptions::new().write(true).open(path).unwrap();
        file.set_len(critical_end_offset).unwrap();
        file.seek(SeekFrom::Start(critical_end_offset)).unwrap();
        file.write_all(&suffix).unwrap();
    }

    fn checkpoint_suffix_bytes(
        checkpoint: &IndexCheckpointPayload,
        checkpoint_generation: u64,
        critical_end_offset: u64,
        index_record_count: u64,
        payload: &[u8],
        extent_table: &[u8],
    ) -> Vec<u8> {
        let extent_table_offset = INDEX_CHECKPOINT_HEADER_LEN as u64 + payload.len() as u64;
        let mut header = Vec::with_capacity(INDEX_CHECKPOINT_HEADER_LEN);
        header.extend_from_slice(INDEX_CHECKPOINT_MAGIC);
        header.extend_from_slice(&INDEX_CHECKPOINT_VERSION.to_le_bytes());
        header.extend_from_slice(&(INDEX_CHECKPOINT_HEADER_LEN as u16).to_le_bytes());
        header.extend_from_slice(&checkpoint_generation.to_le_bytes());
        header.extend_from_slice(&critical_end_offset.to_le_bytes());
        header.extend_from_slice(&critical_end_offset.to_le_bytes());
        header.extend_from_slice(&checkpoint.metadata.covered_last_batch_id.to_le_bytes());
        header.extend_from_slice(
            &checkpoint
                .metadata
                .covered_last_commit_sequence_no
                .to_le_bytes(),
        );
        header.push(INDEX_COVERAGE_FULL_VISIBLE_SNAPSHOT);
        header.extend_from_slice(&checkpoint.metadata.coverage_hash.to_le_bytes());
        header.extend_from_slice(&index_record_count.to_le_bytes());
        header.extend_from_slice(&index_record_count.to_le_bytes());
        header.extend_from_slice(&extent_table_offset.to_le_bytes());
        header.extend_from_slice(&(extent_table.len() as u64).to_le_bytes());
        header.extend_from_slice(&checksum32(extent_table).to_le_bytes());
        header.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        header.extend_from_slice(&checksum32(payload).to_le_bytes());
        let mut trailer = Vec::with_capacity(INDEX_CHECKPOINT_TRAILER_LEN);
        trailer.extend_from_slice(INDEX_COMPLETE_MARKER);
        trailer.extend_from_slice(&(INDEX_CHECKPOINT_TRAILER_LEN as u16).to_le_bytes());
        trailer.extend_from_slice(&checksum32(&trailer).to_le_bytes());
        let mut suffix = Vec::new();
        suffix.extend_from_slice(&header);
        suffix.extend_from_slice(payload);
        suffix.extend_from_slice(extent_table);
        suffix.extend_from_slice(&trailer);
        suffix
    }

    #[test]
    fn committed_records_survive_reopen() {
        let path = temp_path("commit");
        let mut storage = Storage::create(&path).unwrap();
        storage
            .append_batch(
                &[RecordPayload::CollectionCreate(CollectionInfo {
                    tokenizer: Default::default(),
                    collection_id: "col_1".into(),
                    name: "notes".into(),
                    embedding_dim: 3,
                    distance_metric: DistanceMetric::Cosine,
                    embedding_model: None,
                    vectors_normalized: false,
                    vector_index: VectorIndexConfig::Exact,
                    created_at: "1970-01-01T00:00:00Z".into(),
                })],
                Durability::Commit,
            )
            .unwrap();
        drop(storage);
        let scan = scan_file(&path).unwrap();
        assert_eq!(scan.commits, 1);
        assert_eq!(scan.records.len(), 1);
        assert!(scan.critical_end_offset > 0);
        let record_ref = &scan.records[0].record_ref;
        assert_eq!(record_ref.record_type, "CollectionCreate");
        assert_eq!(record_ref.batch_id, 1);
        assert_eq!(record_ref.sequence_no, 1);
        assert!(record_ref.payload.offset > record_ref.offset);
        assert!(record_ref.payload.len > 0);
        assert!(record_ref.payload.checksum > 0);
        let (storage, _) = Storage::open(&path).unwrap();
        assert!(matches!(
            storage.read_payload(record_ref).unwrap(),
            RecordPayload::CollectionCreate(_)
        ));
        drop(storage);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn rewrite_validates_temp_file_before_rename() {
        let path = temp_path("rewrite-temp-validation");
        let mut storage = Storage::create(&path).unwrap();
        storage
            .append_batch(&[collection_record("col_1", "notes")], Durability::Commit)
            .unwrap();
        let original_critical_end = storage.critical_end_offset();
        let original_file_size = storage.file_size().unwrap();

        let err = storage
            .rewrite_with_records_with_hook(
                &[collection_record("col_2", "replacement")],
                Durability::Commit,
                |tmp_path| {
                    let mut tmp = OpenOptions::new().write(true).open(tmp_path)?;
                    tmp.seek(SeekFrom::Start(0))?;
                    tmp.write_all(b"BADTEMP!")?;
                    tmp.sync_data()?;
                    Ok(())
                },
            )
            .unwrap_err();
        assert_eq!(err.code(), ErrorCode::FileFormatError);
        assert_eq!(storage.critical_end_offset(), original_critical_end);
        assert_eq!(storage.file_size().unwrap(), original_file_size);
        assert!(!path.with_extension("coffer.tmp").exists());

        let scan = scan_file(&path).unwrap();
        assert_eq!(scan.records.len(), 1);
        match &scan.records[0].payload {
            RecordPayload::CollectionCreate(collection) => {
                assert_eq!(collection.collection_id, "col_0000000000000001");
                assert_eq!(collection.name, "notes");
            }
            other => panic!("unexpected compact survivor: {other:?}"),
        }
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn scan_ignores_incomplete_tail_after_commit() {
        let path = temp_path("tail");
        let mut storage = Storage::create(&path).unwrap();
        storage
            .append_batch(
                &[RecordPayload::CollectionCreate(CollectionInfo {
                    tokenizer: Default::default(),
                    collection_id: "col_1".into(),
                    name: "notes".into(),
                    embedding_dim: 3,
                    distance_metric: DistanceMetric::Cosine,
                    embedding_model: None,
                    vectors_normalized: false,
                    vector_index: VectorIndexConfig::Exact,
                    created_at: "1970-01-01T00:00:00Z".into(),
                })],
                Durability::Commit,
            )
            .unwrap();
        drop(storage);
        let mut file = OpenOptions::new().append(true).open(&path).unwrap();
        file.write_all(&[1, 2, 3, 4, 5]).unwrap();
        drop(file);
        let scan = scan_file(&path).unwrap();
        assert_eq!(scan.records.len(), 1);
        assert!(scan
            .findings
            .iter()
            .any(|finding| finding.code == "corrupt_tail"
                && finding.stream == "index"
                && finding.visibility_impact == "tail_only"));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn scan_reports_post_commit_suffix_as_index_rebuild_only() {
        let path = temp_path("index-suffix");
        let mut storage = Storage::create(&path).unwrap();
        storage
            .append_batch(&[collection_record("col_1", "notes")], Durability::Commit)
            .unwrap();
        drop(storage);
        append_fake_index_suffix(&path);

        let scan = scan_file(&path).unwrap();
        assert_eq!(scan.records.len(), 1);
        assert!(scan
            .findings
            .iter()
            .any(|finding| finding.code == "index_suffix_unreadable"
                && finding.stream == "index"
                && finding.visibility_impact == "rebuild_index_only"));
        assert!(!scan
            .findings
            .iter()
            .any(|finding| finding.visibility_impact == "blocks_main"));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn scan_reads_metadata_index_checkpoint_after_critical_prefix() {
        let path = temp_path("index-checkpoint");
        let mut storage = Storage::create(&path).unwrap();
        let append = storage
            .append_batch(&[collection_record("col_1", "notes")], Durability::Commit)
            .unwrap();
        let (suffix, _) = storage
            .write_index_checkpoint(&checkpoint_for(&append), Durability::Commit)
            .unwrap();
        assert_eq!(suffix.status, "metadata_record_only");
        assert_eq!(suffix.index_record_count, 1);
        assert_eq!(suffix.index_extent_count, 1);
        assert!(suffix.offset > HEADER_LEN as u64);
        drop(storage);

        let scan = scan_file(&path).unwrap();
        let suffix = scan.index_suffix.unwrap();
        assert_eq!(scan.records.len(), 1);
        assert_eq!(suffix.status, "metadata_record_only");
        assert_eq!(suffix.index_record_count, 1);
        assert_eq!(suffix.index_extent_count, 1);
        assert_eq!(suffix.covered_last_batch_id, 1);
        assert_eq!(suffix.covered_last_commit_sequence_no, 2);
        assert_eq!(suffix.coverage_hash, 42);
        assert!(!scan
            .findings
            .iter()
            .any(|finding| finding.stream == "index"));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn scan_reads_full_index_checkpoint_segments_after_critical_prefix() {
        let path = temp_path("full-index-checkpoint");
        let mut storage = Storage::create(&path).unwrap();
        let append = storage
            .append_batch(&[collection_record("col_1", "notes")], Durability::Commit)
            .unwrap();
        let (suffix, _) = storage
            .write_index_checkpoint(&full_checkpoint_for(&append), Durability::Commit)
            .unwrap();
        assert_eq!(suffix.status, "persistent_checkpoint");
        assert_eq!(suffix.index_record_count, 3);
        assert_eq!(suffix.index_extent_count, 3);
        assert!(suffix.payload_len > (INDEX_RECORD_HEADER_LEN + INDEX_COMMIT_METADATA_LEN) as u64);
        drop(storage);

        let scan = scan_file(&path).unwrap();
        let suffix = scan.index_suffix.unwrap();
        assert_eq!(scan.records.len(), 1);
        assert_eq!(suffix.status, "persistent_checkpoint");
        assert_eq!(suffix.build_mode, "inline_full_checkpoint");
        assert_eq!(suffix.index_record_count, 3);
        assert_eq!(suffix.index_extent_count, 3);
        assert_eq!(suffix.collection_count, 1);
        assert_eq!(suffix.live_chunk_count, 1);
        assert_eq!(suffix.live_embedding_count, 1);
        assert_eq!(suffix.text_segment_count, 1);
        assert_eq!(suffix.vector_segment_count, 1);
        assert!(!scan
            .findings
            .iter()
            .any(|finding| finding.stream == "index"));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn scan_rejects_damaged_full_index_segment_as_rebuild_only() {
        let path = temp_path("full-index-segment-damage");
        let mut storage = Storage::create(&path).unwrap();
        let append = storage
            .append_batch(&[collection_record("col_1", "notes")], Durability::Commit)
            .unwrap();
        storage
            .write_index_checkpoint(&full_checkpoint_for(&append), Durability::Commit)
            .unwrap();
        drop(storage);

        let scan = scan_file(&path).unwrap();
        let suffix_offset = scan.index_suffix.as_ref().unwrap().offset;
        let text_segment_offset = suffix_offset
            + INDEX_CHECKPOINT_HEADER_LEN as u64
            + (INDEX_RECORD_HEADER_LEN + INDEX_COMMIT_METADATA_LEN) as u64
            + INDEX_RECORD_HEADER_LEN as u64;
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .unwrap();
        file.seek(SeekFrom::Start(text_segment_offset + 2)).unwrap();
        file.write_all(&[0xff]).unwrap();
        drop(file);

        let scan = scan_file(&path).unwrap();
        assert_eq!(scan.records.len(), 1);
        assert!(scan.findings.iter().any(|finding| {
            finding.code == "index_extent_record_checksum_mismatch"
                && finding.stream == "index"
                && finding.visibility_impact == "rebuild_index_only"
                && finding.extent_id.as_deref() == Some("index_record:2")
        }));
        assert!(!scan
            .findings
            .iter()
            .any(|finding| finding.visibility_impact == "blocks_main"));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn scan_rejects_unreferenced_unknown_index_record_as_rebuild_only() {
        let path = temp_path("unreferenced-unknown-index-record");
        let mut storage = Storage::create(&path).unwrap();
        let append = storage
            .append_batch(&[collection_record("col_1", "notes")], Durability::Commit)
            .unwrap();
        storage
            .write_index_checkpoint(&full_checkpoint_for(&append), Durability::Commit)
            .unwrap();
        let critical_end_offset = storage.critical_end_offset();
        drop(storage);
        append_checkpoint_with_unreferenced_unknown_record(&path, &append, critical_end_offset);

        let scan = scan_file(&path).unwrap();
        assert_eq!(scan.records.len(), 1);
        assert!(scan.index_suffix.is_none());
        assert!(scan.findings.iter().any(|finding| {
            finding.code == "index_checkpoint_directory_mismatch"
                && finding.stream == "index"
                && finding.visibility_impact == "rebuild_index_only"
        }));
        let report = Database::check_path(
            &path,
            CheckOptions {
                verify_indexes: true,
            },
        )
        .unwrap();
        assert_eq!(report.index_suffix_status, "rebuild_needed");
        assert_eq!(report.index_check, "rebuild_needed");
        assert_eq!(
            report.index_check_reason.as_deref(),
            Some("index_checkpoint_directory_mismatch")
        );
        assert!(!report
            .issues
            .iter()
            .any(|finding| finding.visibility_impact == "blocks_main"));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn scan_rejects_segment_stream_id_mismatch_as_rebuild_only() {
        let path = temp_path("segment-stream-id-mismatch");
        let mut storage = Storage::create(&path).unwrap();
        let append = storage
            .append_batch(&[collection_record("col_1", "notes")], Durability::Commit)
            .unwrap();
        storage
            .write_index_checkpoint(&full_checkpoint_for(&append), Durability::Commit)
            .unwrap();
        let critical_end_offset = storage.critical_end_offset();
        drop(storage);
        append_checkpoint_with_text_stream_id_mismatch(&path, &append, critical_end_offset);

        let scan = scan_file(&path).unwrap();
        assert_eq!(scan.records.len(), 1);
        assert!(scan.index_suffix.is_none());
        assert!(scan.findings.iter().any(|finding| {
            finding.code == "index_extent_table_mismatch"
                && finding.stream == "index"
                && finding.visibility_impact == "rebuild_index_only"
        }));
        assert!(!scan
            .findings
            .iter()
            .any(|finding| finding.visibility_impact == "blocks_main"));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn scan_accepts_full_segments_when_only_aggregate_payload_checksum_mismatches() {
        let path = temp_path("aggregate-payload-checksum");
        let mut storage = Storage::create(&path).unwrap();
        let append = storage
            .append_batch(&[collection_record("col_1", "notes")], Durability::Commit)
            .unwrap();
        storage
            .write_index_checkpoint(&full_checkpoint_for(&append), Durability::Commit)
            .unwrap();
        drop(storage);

        let scan = scan_file(&path).unwrap();
        let suffix_offset = scan.index_suffix.as_ref().unwrap().offset;
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .unwrap();
        file.seek(SeekFrom::Start(suffix_offset + 101)).unwrap();
        file.write_all(&u32::MAX.to_le_bytes()).unwrap();
        drop(file);

        let scan = scan_file(&path).unwrap();
        let suffix = scan.index_suffix.unwrap();
        assert_eq!(suffix.status, "persistent_checkpoint");
        assert_eq!(suffix.text_segment_count, 1);
        assert_eq!(suffix.vector_segment_count, 1);
        assert!(scan.findings.iter().any(|finding| {
            finding.code == "checkpoint_payload_checksum_mismatch"
                && finding.stream == "index"
                && finding.visibility_impact == "rebuild_index_only"
                && finding.repair_available
        }));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn scan_reports_index_record_envelope_damage_as_rebuild_only() {
        let path = temp_path("index-record-damage");
        let mut storage = Storage::create(&path).unwrap();
        let append = storage
            .append_batch(&[collection_record("col_1", "notes")], Durability::Commit)
            .unwrap();
        storage
            .write_index_checkpoint(&checkpoint_for(&append), Durability::Commit)
            .unwrap();
        drop(storage);

        let scan = scan_file(&path).unwrap();
        let suffix_offset = scan.index_suffix.as_ref().unwrap().offset;
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .unwrap();
        file.seek(SeekFrom::Start(
            suffix_offset + INDEX_CHECKPOINT_HEADER_LEN as u64,
        ))
        .unwrap();
        file.write_all(b"BADC").unwrap();
        drop(file);

        let scan = scan_file(&path).unwrap();
        assert_eq!(scan.records.len(), 1);
        assert!(scan.findings.iter().any(|finding| finding.code
            == "index_extent_record_checksum_mismatch"
            && finding.stream == "index"
            && finding.visibility_impact == "rebuild_index_only"));
        assert!(!scan
            .findings
            .iter()
            .any(|finding| finding.visibility_impact == "blocks_main"));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn scan_rejects_overflowing_checkpoint_payload_len_as_index_only() {
        let path = temp_path("checkpoint-payload-overflow");
        let mut storage = Storage::create(&path).unwrap();
        storage
            .append_batch(&[collection_record("col_1", "notes")], Durability::Commit)
            .unwrap();
        drop(storage);
        let scan = scan_file(&path).unwrap();
        append_malformed_checkpoint_header(&path, &scan, u64::MAX, INDEX_EXTENT_ENTRY_LEN as u64);

        let scan = scan_file(&path).unwrap();
        assert_eq!(scan.records.len(), 1);
        assert!(scan.findings.iter().any(|finding| {
            finding.code == "index_checkpoint_length_overflow"
                && finding.stream == "index"
                && finding.visibility_impact == "rebuild_index_only"
        }));
        assert!(!scan
            .findings
            .iter()
            .any(|finding| finding.visibility_impact == "blocks_main"));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn scan_rejects_overflowing_checkpoint_extent_len_as_index_only() {
        let path = temp_path("checkpoint-extent-overflow");
        let mut storage = Storage::create(&path).unwrap();
        storage
            .append_batch(&[collection_record("col_1", "notes")], Durability::Commit)
            .unwrap();
        drop(storage);
        let scan = scan_file(&path).unwrap();
        append_malformed_checkpoint_header(&path, &scan, 0, u64::MAX);

        let scan = scan_file(&path).unwrap();
        assert_eq!(scan.records.len(), 1);
        assert!(scan.findings.iter().any(|finding| {
            finding.code == "index_checkpoint_length_overflow"
                && finding.stream == "index"
                && finding.visibility_impact == "rebuild_index_only"
        }));
        assert!(!scan
            .findings
            .iter()
            .any(|finding| finding.visibility_impact == "blocks_main"));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn scan_rejects_overflowing_index_record_payload_len_as_index_only() {
        let path = temp_path("index-record-payload-overflow");
        let mut storage = Storage::create(&path).unwrap();
        storage
            .append_batch(&[collection_record("col_1", "notes")], Durability::Commit)
            .unwrap();
        drop(storage);
        let scan = scan_file(&path).unwrap();
        append_checkpoint_with_overflowing_index_record_payload_len(&path, &scan);

        let scan = scan_file(&path).unwrap();
        assert_eq!(scan.records.len(), 1);
        assert!(scan.findings.iter().any(|finding| {
            finding.code == "index_record_length_overflow"
                && finding.stream == "index"
                && finding.visibility_impact == "rebuild_index_only"
        }));
        assert!(!scan
            .findings
            .iter()
            .any(|finding| finding.visibility_impact == "blocks_main"));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn append_truncates_rebuildable_suffix_before_next_batch() {
        let path = temp_path("truncate-suffix");
        let mut storage = Storage::create(&path).unwrap();
        storage
            .append_batch(&[collection_record("col_1", "notes")], Durability::Commit)
            .unwrap();
        drop(storage);
        append_fake_index_suffix(&path);

        let (mut storage, scan) = Storage::open(&path).unwrap();
        assert!(scan
            .findings
            .iter()
            .any(|finding| finding.stream == "index"));
        storage
            .append_batch(&[collection_record("col_2", "tasks")], Durability::Commit)
            .unwrap();
        drop(storage);

        let scan = scan_file(&path).unwrap();
        assert_eq!(scan.records.len(), 2);
        assert!(!scan
            .findings
            .iter()
            .any(|finding| finding.stream == "index"));
        assert_eq!(
            scan.critical_end_offset,
            std::fs::metadata(&path).unwrap().len()
        );
        let _ = std::fs::remove_file(path);
    }
}
