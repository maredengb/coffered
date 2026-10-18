use std::fs::{self, File};
use std::io::{Read, Seek, Write};
use std::path::{Path, PathBuf};

use crate::db::{
    Database, ImportDocument as JsonlDocument, ImportIndexBuildMode, ImportRow,
    ManifestImportOptions,
};
use crate::error::{Error, ErrorCode, Result};
use crate::hash::{digest_hex, Sha256};
#[cfg(test)]
use crate::hash::{sha256_hex, sha256_reader};
use crate::json::{
    ensure_object, object_from_pairs, parse_reader, parse_reader_in, required_array,
    required_string, required_usize, JsonBuilder, JsonFieldValue, JsonInputKind, JsonLines,
    JsonObject, JsonOperationContext, JsonReservation, JsonSchema, JsonValue,
};
use crate::model::{
    json_usize, ChunkInput, DistanceMetric, EmbeddingInput, InsertSummary, OnConflict, SourceSpan,
};
#[cfg(test)]
use crate::temporary::create_budgeted_temporary_file_in;
use crate::temporary::{
    create_budgeted_temporary_file_in_context, create_budgeted_temporary_file_reclaiming,
    TempFileGuard, TemporaryDiskBudget, TemporaryFile,
};

pub(crate) const MAX_JSONL_BYTES: u64 = 64 * 1_048_576;
mod export_writer;
use export_writer::{retry_after_buffer_reclaim, ExportWriter};

struct ExportChunk {
    // The synthesized value shares the retained chunk's JSON reservation.
    source_span: JsonValue,
    retained: crate::storage::RetainedChunk,
}

impl ExportChunk {
    fn read_in(
        reader: &mut crate::storage::PayloadReader,
        record_ref: &crate::storage::RecordRef,
        context: &JsonOperationContext,
    ) -> Result<Self> {
        #[cfg(test)]
        tests::EXPORT_CHUNK_READS.set(tests::EXPORT_CHUNK_READS.get() + 1);
        let mut retained = reader.read_chunk_in(record_ref, context)?;
        #[cfg(test)]
        tests::EXPORT_SPAN_BUILDS.set(tests::EXPORT_SPAN_BUILDS.get() + 1);
        let (chunk, work) = retained.parts();
        let builder = JsonBuilder::budgeted(work, JsonInputKind::StoredJson);
        let source_span = chunk
            .source_span
            .as_ref()
            .map(|span| span.build_json(&builder))
            .transpose()?
            .unwrap_or_default();
        Ok(Self {
            source_span,
            retained,
        })
    }
}
const EXPORT_FILES: [&str; 4] = [
    "documents.jsonl",
    "chunks.jsonl",
    "embeddings.jsonl",
    "export_manifest.json",
];

fn validate_export_destinations(source: &Path, directory: &Path) -> Result<()> {
    let source = fs::canonicalize(source)?;
    let source_metadata = fs::metadata(&source)?;
    for name in EXPORT_FILES {
        let target = directory.join(name);
        let metadata = match fs::metadata(&target) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => return Err(error.into()),
        };
        let same_file = fs::canonicalize(&target)? == source;
        #[cfg(unix)]
        let same_file = {
            use std::os::unix::fs::MetadataExt;
            same_file
                || (metadata.dev() == source_metadata.dev()
                    && metadata.ino() == source_metadata.ino())
        };
        #[cfg(not(unix))]
        let _ = &source_metadata;
        if same_file {
            return Err(Error::new(
                ErrorCode::PermissionDenied,
                "export output cannot replace the source database",
            )
            .with_detail("path", target.display()));
        }
        if !metadata.is_file() {
            return Err(Error::new(
                ErrorCode::IoError,
                "export output destination is not a regular file",
            )
            .with_detail("path", target.display()));
        }
    }
    Ok(())
}

fn input_budget_error(path: &Path, limit: u64) -> Error {
    Error::new(
        ErrorCode::InvalidJsonlSchema,
        "import input exceeds the atomic import budget; split into smaller manifests",
    )
    .with_detail("path", path.display())
    .with_detail("reason", "import_budget_exceeded")
    .with_detail("limit_bytes", limit)
}

fn open_bounded(path: &Path, limit: u64) -> Result<File> {
    let file = File::open(path)?;
    if file.metadata()?.len() > limit {
        return Err(input_budget_error(path, limit));
    }
    Ok(file)
}

fn read_manifest(path: &Path) -> Result<JsonValue> {
    parse_reader(
        File::open(path)?,
        JsonSchema::Value(JsonInputKind::Manifest),
    )
    .map_err(|error| error.with_detail("path", path.display()))
}

#[derive(Clone, Debug)]
pub struct ImportManifest {
    pub tokenizer: crate::TokenizerConfig,
    pub schema_version: String,
    pub dataset_id: Option<String>,
    pub collection: String,
    pub documents: PathBuf,
    pub chunks: PathBuf,
    pub embeddings: Option<PathBuf>,
    pub documents_sha256: Option<String>,
    pub chunks_sha256: Option<String>,
    pub embeddings_sha256: Option<String>,
    pub embedding_model: Option<String>,
    pub embedding_dim: Option<usize>,
    pub distance_metric: Option<DistanceMetric>,
}

#[derive(Clone, Debug, Default)]
pub struct ImportSummary {
    pub tokenizer: crate::TokenizerConfig,
    pub dataset_id: Option<String>,
    pub documents_seen: usize,
    pub chunks_seen: usize,
    pub embeddings_seen: usize,
    pub documents_inserted: usize,
    pub chunks_inserted: usize,
    pub embeddings_inserted: usize,
    pub documents_skipped: usize,
    pub chunks_skipped: usize,
    pub embeddings_skipped: usize,
    pub documents_replaced: usize,
    pub chunks_replaced: usize,
    pub embeddings_replaced: usize,
    pub warnings: Vec<ImportWarning>,
}

pub(crate) struct MeasuredImport {
    pub(crate) summary: ImportSummary,
    pub(crate) batches: crate::db::ImportBatchMetrics,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImportWarning {
    pub code: String,
    pub field: String,
    pub artifact_kind: String,
    pub message: String,
}

#[derive(Clone, Debug, Default)]
pub struct ExportSummary {
    pub documents: usize,
    pub chunks: usize,
    pub embeddings: usize,
    pub manifest: String,
    pub documents_sha256: String,
    pub chunks_sha256: String,
    pub embeddings_sha256: String,
}

impl Database {
    pub fn import_manifest(
        &self,
        manifest_path: impl AsRef<Path>,
        on_conflict: OnConflict,
    ) -> Result<ImportSummary> {
        self.import_manifest_with_index_mode(
            manifest_path.as_ref(),
            on_conflict,
            ImportIndexBuildMode::DeferredFullCheckpoint,
        )
    }

    pub(crate) fn import_manifest_with_index_mode(
        &self,
        manifest_path: &Path,
        on_conflict: OnConflict,
        index_build_mode: ImportIndexBuildMode,
    ) -> Result<ImportSummary> {
        self.import_manifest_in(
            manifest_path,
            on_conflict,
            index_build_mode,
            self.json_context(),
        )
    }

    pub(crate) fn import_manifest_in(
        &self,
        manifest_path: &Path,
        on_conflict: OnConflict,
        index_build_mode: ImportIndexBuildMode,
        context: JsonOperationContext,
    ) -> Result<ImportSummary> {
        self.import_manifest_measured_in(manifest_path, on_conflict, index_build_mode, context)
            .map(|import| import.summary)
    }

    pub(crate) fn import_manifest_measured_in(
        &self,
        manifest_path: &Path,
        on_conflict: OnConflict,
        index_build_mode: ImportIndexBuildMode,
        context: JsonOperationContext,
    ) -> Result<MeasuredImport> {
        self.attach_json_context(&context, JsonInputKind::Manifest)
            .map_err(|error| error.with_detail("path", manifest_path.display()))?;
        let manifest_value = parse_reader_in(
            File::open(manifest_path)?,
            JsonSchema::Value(JsonInputKind::Manifest),
            &context,
        )
        .map_err(|error| error.with_detail("path", manifest_path.display()))?;
        let manifest = ImportManifest::from_value(manifest_path, &manifest_value.value)?;
        let expected_counts =
            parse_export_manifest_counts(manifest_value.value.as_object().unwrap())?;
        drop(manifest_value);
        let disk = self.temporary_disk_budget();
        let mut inputs = manifest.validated_inputs_reclaiming(disk.clone(), || {
            self.admit_input_reclaiming(|| {
                create_budgeted_temporary_file_in_context(
                    &std::env::temp_dir(),
                    "import-input",
                    "tmp",
                    disk.clone(),
                    &context,
                    "operation_scratch",
                )
            })
        })?;
        let (embedding_dim, distance_metric) = self.resolve_import_collection_shape(&manifest)?;
        let mut summary = ImportSummary {
            tokenizer: manifest.tokenizer.clone(),
            dataset_id: manifest.dataset_id.clone(),
            warnings: manifest.reproducibility_warnings(),
            ..ImportSummary::default()
        };
        let (inserted, batches) = self.import_preprocessed_rows(
            ManifestImportOptions {
                tokenizer: manifest.tokenizer,
                collection_name: manifest.collection,
                embedding_dim,
                distance_metric,
                embedding_model: manifest.embedding_model,
                document_path: manifest.documents.clone(),
                chunk_path: manifest.chunks.clone(),
                embedding_path: manifest.embeddings.clone(),
                on_conflict,
                index_build_mode,
            },
            context.clone(),
            |consume| {
                let document_bytes = inputs.documents.len;
                visit_documents_jsonl_in(
                    &manifest.documents,
                    inputs.documents.reader()?,
                    &context,
                    document_bytes,
                    |line, document, work| {
                        summary.documents_seen += 1;
                        consume(
                            ImportRow::Document {
                                input: document,
                                line,
                            },
                            work,
                        )
                    },
                )?;
                let chunk_bytes = inputs.chunks.len;
                visit_chunks_jsonl_in(
                    &manifest.chunks,
                    inputs.chunks.reader()?,
                    &context,
                    chunk_bytes,
                    |line, chunk, work| {
                        summary.chunks_seen += 1;
                        summary.warnings.extend(chunk.warnings);
                        consume(
                            ImportRow::Chunk {
                                document_external_id: chunk.document_external_id,
                                chunk: chunk.chunk,
                                line,
                            },
                            work,
                        )
                    },
                )?;
                if let Some(path) = &manifest.embeddings {
                    let input = inputs.embeddings.as_mut().unwrap();
                    let embedding_bytes = input.len;
                    visit_embeddings_jsonl_in(
                        path,
                        input.reader()?,
                        &context,
                        embedding_bytes,
                        |line, embedding, work| {
                            summary.embeddings_seen += 1;
                            consume(
                                ImportRow::Embedding {
                                    input: embedding,
                                    line,
                                },
                                work,
                            )
                        },
                    )?;
                }
                expected_counts.validate_actual(
                    summary.documents_seen,
                    summary.chunks_seen,
                    summary.embeddings_seen,
                )
            },
        )?;
        merge_insert_summary(&mut summary, inserted);

        Ok(MeasuredImport { summary, batches })
    }

    pub fn export_collection_jsonl(
        &self,
        collection_name: &str,
        output_dir: impl AsRef<Path>,
    ) -> Result<ExportSummary> {
        let context = self.json_context();
        self.with_read_snapshot_reclaiming(|catalog, storage, reclaim| {
            let collection_info = catalog.collection_by_name(collection_name)?;
            let collection_id = &collection_info.collection_id;
            let mut reader = storage.payload_reader()?;
            fs::create_dir_all(output_dir.as_ref())?;
            validate_export_destinations(storage.path(), output_dir.as_ref())?;
            let documents_path = output_dir.as_ref().join("documents.jsonl");
            let chunks_path = output_dir.as_ref().join("chunks.jsonl");
            let embeddings_path = output_dir.as_ref().join("embeddings.jsonl");
            let manifest_path = output_dir.as_ref().join("export_manifest.json");

            const OUTPUT_BUFFER_BYTES: usize = 8192;
            let disk = self.temporary_disk_budget();
            let (documents, mut documents_temp) = create_budgeted_temporary_file_reclaiming(
                output_dir.as_ref(),
                "export-documents",
                "tmp",
                disk.clone(),
                &context,
                "operation_scratch",
                &mut *reclaim,
            )?;
            let (chunks, mut chunks_temp) = create_budgeted_temporary_file_reclaiming(
                output_dir.as_ref(),
                "export-chunks",
                "tmp",
                disk.clone(),
                &context,
                "operation_scratch",
                &mut *reclaim,
            )?;
            let (embeddings, mut embeddings_temp) = create_budgeted_temporary_file_reclaiming(
                output_dir.as_ref(),
                "export-embeddings",
                "tmp",
                disk.clone(),
                &context,
                "operation_scratch",
                &mut *reclaim,
            )?;
            let mut documents_file = ExportWriter::new_in(
                documents,
                &mut documents_temp,
                OUTPUT_BUFFER_BYTES,
                &context,
            )?;
            let mut chunks_file =
                ExportWriter::new_in(chunks, &mut chunks_temp, OUTPUT_BUFFER_BYTES, &context)?;
            let mut embeddings_file = ExportWriter::new_in(
                embeddings,
                &mut embeddings_temp,
                OUTPUT_BUFFER_BYTES,
                &context,
            )?;
            let mut documents_count = 0;
            let mut chunks_count = 0;
            let mut embeddings_count = 0;

            for doc in catalog
                .live_documents()
                .filter(|doc| &doc.collection_id == collection_id)
            {
                let _metadata_work = retry_after_buffer_reclaim(
                    &mut [&mut documents_file, &mut chunks_file, &mut embeddings_file],
                    &mut *reclaim,
                    |_| doc.metadata.guard_in(JsonInputKind::StoredJson, &context),
                )?;
                let fields = [
                    (
                        "external_id",
                        JsonFieldValue::String(Some(&doc.external_id)),
                    ),
                    ("title", JsonFieldValue::String(doc.title.as_deref())),
                    (
                        "source_uri",
                        JsonFieldValue::String(doc.source_uri.as_deref()),
                    ),
                    ("metadata", JsonFieldValue::Json(&doc.metadata)),
                ];
                retry_after_buffer_reclaim(
                    &mut [&mut documents_file, &mut chunks_file, &mut embeddings_file],
                    &mut *reclaim,
                    |writers| writers[0].write_fields(fields, &context),
                )?;
                documents_count += 1;
            }

            for summary in catalog
                .live_chunks()
                .filter(|chunk| &chunk.collection_id == collection_id)
            {
                let doc = catalog.documents.get(&summary.document_id).ok_or_else(|| {
                    Error::new(ErrorCode::FileFormatError, "export chunk has no document")
                })?;
                let record_ref =
                    catalog
                        .record_ref_for_chunk(&summary.chunk_id)
                        .ok_or_else(|| {
                            Error::new(
                                ErrorCode::FileFormatError,
                                "export chunk has no payload locator",
                            )
                        })?;
                let mut prepared = retry_after_buffer_reclaim(
                    &mut [&mut documents_file, &mut chunks_file, &mut embeddings_file],
                    &mut *reclaim,
                    |_| ExportChunk::read_in(&mut reader, record_ref, &context),
                )?;
                let (chunk, _) = prepared.retained.parts();
                let ordinal = JsonValue::Number(chunk.ordinal as f64);
                let fields = [
                    (
                        "document_external_id",
                        JsonFieldValue::String(Some(&doc.external_id)),
                    ),
                    (
                        "chunk_external_id",
                        JsonFieldValue::String(Some(&chunk.chunk_external_id)),
                    ),
                    ("ordinal", JsonFieldValue::Json(&ordinal)),
                    ("text", JsonFieldValue::String(Some(&chunk.text))),
                    ("metadata", JsonFieldValue::Json(&chunk.metadata)),
                    ("source_span", JsonFieldValue::Json(&prepared.source_span)),
                ];
                retry_after_buffer_reclaim(
                    &mut [&mut documents_file, &mut chunks_file, &mut embeddings_file],
                    &mut *reclaim,
                    |writers| writers[1].write_fields(fields, &context),
                )?;
                chunks_count += 1;
            }

            for summary in catalog
                .live_embeddings()
                .filter(|embedding| &embedding.collection_id == collection_id)
            {
                let record_ref = catalog
                    .record_ref_for_embedding(&summary.embedding_id)
                    .ok_or_else(|| {
                        Error::new(
                            ErrorCode::FileFormatError,
                            "export embedding has no payload locator",
                        )
                    })?;
                let retained = retry_after_buffer_reclaim(
                    &mut [&mut documents_file, &mut chunks_file, &mut embeddings_file],
                    &mut *reclaim,
                    |_| reader.read_embedding_in(record_ref, &context),
                )?;
                let embedding = retained.embedding();
                let dim = JsonValue::Number(embedding.dim as f64);
                let fields = [
                    (
                        "chunk_external_id",
                        JsonFieldValue::String(Some(&summary.chunk_external_id)),
                    ),
                    (
                        "model_name",
                        JsonFieldValue::String(Some(&embedding.model_name)),
                    ),
                    ("dim", JsonFieldValue::Json(&dim)),
                    ("vector", JsonFieldValue::Vector(&embedding.vector)),
                ];
                retry_after_buffer_reclaim(
                    &mut [&mut documents_file, &mut chunks_file, &mut embeddings_file],
                    &mut *reclaim,
                    |writers| writers[2].write_fields(fields, &context),
                )?;
                embeddings_count += 1;
            }
            let documents_sha256 = documents_file.finish()?;
            let chunks_sha256 = chunks_file.finish()?;
            let embeddings_sha256 = embeddings_file.finish()?;
            let exported_at = crate::time::unix_millis_now();
            let mut no_writers: [&mut ExportWriter<'_>; 0] = [];
            let mut manifest = retry_after_buffer_reclaim(&mut no_writers, &mut *reclaim, |_| {
                let mut work = context.reservation();
                let builder = JsonBuilder::budgeted(&mut work, JsonInputKind::StoredJson);
                let value = builder.object([
                    ("tokenizer", collection_info.tokenizer.build_json(&builder)?),
                    ("schema_version", builder.string("0.1")?),
                    (
                        "cofferdb_version",
                        builder.string(env!("CARGO_PKG_VERSION"))?,
                    ),
                    (
                        "file_format_version",
                        JsonValue::Number(crate::storage::FORMAT_VERSION as f64),
                    ),
                    ("collection", builder.string(collection_name)?),
                    (
                        "exported_at",
                        builder.owned_string(crate::time::format_unix_millis(exported_at))?,
                    ),
                    (
                        "embedding_model",
                        builder.optional_str(collection_info.embedding_model.as_deref())?,
                    ),
                    (
                        "embedding_dim",
                        JsonValue::Number(collection_info.embedding_dim as f64),
                    ),
                    (
                        "distance_metric",
                        builder.string(collection_info.distance_metric.as_str())?,
                    ),
                    ("documents", builder.string("documents.jsonl")?),
                    ("chunks", builder.string("chunks.jsonl")?),
                    ("embeddings", builder.string("embeddings.jsonl")?),
                    (
                        "counts",
                        builder.object([
                            ("documents", JsonValue::Number(documents_count as f64)),
                            ("chunks", JsonValue::Number(chunks_count as f64)),
                            ("embeddings", JsonValue::Number(embeddings_count as f64)),
                        ])?,
                    ),
                    (
                        "hashes",
                        builder.object([
                            ("documents_sha256", builder.string(&documents_sha256)?),
                            ("chunks_sha256", builder.string(&chunks_sha256)?),
                            ("embeddings_sha256", builder.string(&embeddings_sha256)?),
                        ])?,
                    ),
                ])?;
                Ok((value, work))
            })?;
            let (file, mut manifest_temp) = create_budgeted_temporary_file_reclaiming(
                output_dir.as_ref(),
                "export-manifest",
                "tmp",
                disk,
                &context,
                "operation_scratch",
                &mut *reclaim,
            )?;
            let mut manifest_file =
                ExportWriter::new_in(file, &mut manifest_temp, OUTPUT_BUFFER_BYTES, &context)?;
            retry_after_buffer_reclaim(&mut [&mut manifest_file], &mut *reclaim, |writers| {
                #[cfg(test)]
                tests::EXPORT_MANIFEST_WRITES.set(tests::EXPORT_MANIFEST_WRITES.get() + 1);
                let (value, work) = &mut manifest;
                let written = value.write_value_retained_in(
                    writers[0],
                    JsonInputKind::StoredJson,
                    &context,
                    work,
                    true,
                );
                writers[0].checked(written)
            })?;
            manifest_file.finish()?;
            validate_export_destinations(storage.path(), output_dir.as_ref())?;
            documents_temp.persist(&documents_path)?;
            chunks_temp.persist(&chunks_path)?;
            embeddings_temp.persist(&embeddings_path)?;
            // Publish the manifest only after all complete, hashed artifacts exist.
            manifest_temp.persist(&manifest_path)?;
            Ok(ExportSummary {
                documents: documents_count,
                chunks: chunks_count,
                embeddings: embeddings_count,
                manifest: "export_manifest.json".into(),
                documents_sha256,
                chunks_sha256,
                embeddings_sha256,
            })
        })
    }
}

impl ImportManifest {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let value = read_manifest(path)?;
        Self::from_value(path, &value)
    }

    fn from_value(path: &Path, value: &JsonValue) -> Result<Self> {
        let base = path.parent().unwrap_or_else(|| Path::new("."));
        let obj = value.as_object().ok_or_else(|| {
            Error::new(
                ErrorCode::InvalidJsonlSchema,
                "import manifest must be a JSON object",
            )
        })?;
        let schema_version = required_string(obj, "schema_version")?;
        if schema_version != "0.1" {
            return Err(Error::new(
                ErrorCode::UnsupportedVersion,
                format!("unsupported manifest schema_version {schema_version}"),
            ));
        }
        let hashes = match obj.get("hashes").filter(|value| !value.is_null()) {
            Some(value) => Some(value.as_object().ok_or_else(|| {
                Error::new(ErrorCode::InvalidJsonlSchema, "hashes must be an object")
                    .with_detail("field", "hashes")
            })?),
            None => None,
        };
        if obj.get("metric").is_some() {
            return Err(Error::new(
                ErrorCode::InvalidJsonlSchema,
                "metric is not a stable import manifest field; use distance_metric",
            )
            .with_detail("field", "metric"));
        }
        reject_unknown_import_manifest_fields(obj)?;
        validate_export_manifest_identity_fields(obj)?;
        if let Some(hashes) = hashes {
            reject_unknown_import_hash_fields(hashes)?;
        }
        Ok(Self {
            schema_version,
            tokenizer: obj
                .get("tokenizer")
                .map(crate::TokenizerConfig::from_json)
                .transpose()?
                .unwrap_or_default(),
            dataset_id: optional_manifest_string(obj, "dataset_id")?,
            collection: required_manifest_string(obj, "collection")?,
            documents: base.join(required_manifest_string(obj, "documents")?),
            chunks: base.join(required_manifest_string(obj, "chunks")?),
            embeddings: optional_manifest_string(obj, "embeddings")?.map(|value| base.join(value)),
            documents_sha256: optional_hash(hashes, "documents_sha256")?,
            chunks_sha256: optional_hash(hashes, "chunks_sha256")?,
            embeddings_sha256: optional_hash(hashes, "embeddings_sha256")?,
            embedding_model: optional_manifest_string(obj, "embedding_model")?,
            embedding_dim: optional_usize(obj, "embedding_dim")?,
            distance_metric: optional_distance_metric(obj, "distance_metric")?,
        })
    }

    #[cfg(test)]
    fn validated_inputs(&self, budget: u64) -> Result<ValidatedInputs> {
        self.validated_inputs_in(
            std::sync::Arc::new(TemporaryDiskBudget::new(budget)),
            &JsonOperationContext::standalone(),
        )
    }

    #[cfg(test)]
    fn validated_inputs_in(
        &self,
        disk: std::sync::Arc<TemporaryDiskBudget>,
        context: &JsonOperationContext,
    ) -> Result<ValidatedInputs> {
        self.validated_inputs_reclaiming(disk.clone(), || {
            create_budgeted_temporary_file_in_context(
                &std::env::temp_dir(),
                "import-input",
                "tmp",
                disk.clone(),
                context,
                "operation_scratch",
            )
        })
    }

    fn validated_inputs_reclaiming(
        &self,
        disk: std::sync::Arc<TemporaryDiskBudget>,
        mut create: impl FnMut() -> Result<(TemporaryFile, TempFileGuard)>,
    ) -> Result<ValidatedInputs> {
        if self.embeddings_sha256.is_some() && self.embeddings.is_none() {
            return Err(Error::new(
                ErrorCode::MissingRequiredField,
                "hashes.embeddings_sha256 requires embeddings path",
            )
            .with_detail("field", "embeddings")
            .with_detail("artifact_kind", "embeddings"));
        }
        let mut load = |kind: &str, path: &Path, hash: Option<&str>| -> Result<ValidatedInput> {
            let source = File::open(path)?;
            disk.retry_cleanup()?;
            // This is only a preflight; each actual write still reserves atomically,
            // accounting for concurrent operations and files that grow after opening.
            disk.check_additional(source.metadata()?.len())?;
            let temporary = create()?;
            ValidatedInput::copy_into(source, kind, path, hash, temporary)
        };
        Ok(ValidatedInputs {
            documents: load(
                "documents",
                &self.documents,
                self.documents_sha256.as_deref(),
            )?,
            chunks: load("chunks", &self.chunks, self.chunks_sha256.as_deref())?,
            embeddings: self
                .embeddings
                .as_deref()
                .map(|path| load("embeddings", path, self.embeddings_sha256.as_deref()))
                .transpose()?,
        })
    }

    fn reproducibility_warnings(&self) -> Vec<ImportWarning> {
        let mut warnings = Vec::new();
        if self.documents_sha256.is_none() {
            warnings.push(ImportWarning::missing_hash(
                "documents",
                "hashes.documents_sha256",
            ));
        }
        if self.chunks_sha256.is_none() {
            warnings.push(ImportWarning::missing_hash(
                "chunks",
                "hashes.chunks_sha256",
            ));
        }
        if self.embeddings.is_some() && self.embeddings_sha256.is_none() {
            warnings.push(ImportWarning::missing_hash(
                "embeddings",
                "hashes.embeddings_sha256",
            ));
        }
        warnings
    }
}

#[derive(Debug)]
struct ValidatedInputs {
    documents: ValidatedInput,
    chunks: ValidatedInput,
    embeddings: Option<ValidatedInput>,
}

#[derive(Debug)]
struct ValidatedInput {
    file: TemporaryFile,
    temporary: TempFileGuard,
    len: u64,
}

impl ValidatedInput {
    #[cfg(test)]
    fn copy_from(
        source: impl Read,
        kind: &str,
        path: &Path,
        expected: Option<&str>,
        limit: u64,
        directory: &Path,
    ) -> Result<Self> {
        Self::copy_into(
            source,
            kind,
            path,
            expected,
            create_budgeted_temporary_file_in(
                directory,
                "import-input",
                "tmp",
                std::sync::Arc::new(TemporaryDiskBudget::new(limit)),
            )?,
        )
    }

    fn copy_into(
        mut source: impl Read,
        kind: &str,
        path: &Path,
        expected: Option<&str>,
        (file, temporary): (TemporaryFile, TempFileGuard),
    ) -> Result<Self> {
        let mut input = Self {
            file,
            temporary,
            len: 0,
        };
        let mut hash = expected.map(|_| Sha256::default());
        let mut buffer = [0; 8192];
        loop {
            let count = match source.read(&mut buffer) {
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                result => result?,
            };
            if count == 0 {
                break;
            }
            if let Some(hash) = &mut hash {
                hash.update(&buffer[..count]);
            }
            input
                .temporary
                .reserve_disk(count as u64)
                .map_err(|error| error.with_detail("path", path.display()))?;
            input.len += count as u64;
            input.file.write_all(&buffer[..count])?;
        }
        if let Some(hash) = hash {
            validate_optional_hash(kind, path, expected, &digest_hex(hash.finish()))?;
        }
        Ok(input)
    }

    fn reader(&mut self) -> Result<&mut File> {
        self.file.rewind()?;
        Ok(&mut self.file)
    }
}

fn reject_unknown_import_manifest_fields(obj: &JsonObject) -> Result<()> {
    for field in obj.keys() {
        if !matches!(
            field.as_str(),
            "schema_version"
                | "dataset_id"
                | "collection"
                | "documents"
                | "chunks"
                | "embeddings"
                | "embedding_model"
                | "embedding_dim"
                | "distance_metric"
                | "tokenizer"
                | "cofferdb_version"
                | "file_format_version"
                | "exported_at"
                | "counts"
                | "hashes"
        ) {
            return Err(Error::new(
                ErrorCode::InvalidJsonlSchema,
                "import manifest contains unsupported field",
            )
            .with_detail("field", field));
        }
    }
    Ok(())
}

fn reject_unknown_import_hash_fields(obj: &JsonObject) -> Result<()> {
    for field in obj.keys() {
        if !matches!(
            field.as_str(),
            "documents_sha256" | "chunks_sha256" | "embeddings_sha256"
        ) {
            return Err(Error::new(
                ErrorCode::InvalidJsonlSchema,
                "import manifest hashes contains unsupported field",
            )
            .with_detail("field", format!("hashes.{field}")));
        }
    }
    Ok(())
}

fn validate_export_manifest_identity_fields(obj: &JsonObject) -> Result<()> {
    optional_manifest_string(obj, "cofferdb_version")?;
    optional_manifest_string(obj, "exported_at")?;
    optional_usize(obj, "file_format_version")?;
    parse_export_manifest_counts(obj)?;
    Ok(())
}

#[derive(Clone, Copy, Debug, Default)]
struct ExportManifestCounts {
    documents: Option<usize>,
    chunks: Option<usize>,
    embeddings: Option<usize>,
}

impl ExportManifestCounts {
    fn validate_actual(
        self,
        documents_count: usize,
        chunks_count: usize,
        embeddings_count: usize,
    ) -> Result<()> {
        self.validate_count("documents", documents_count)?;
        self.validate_count("chunks", chunks_count)?;
        self.validate_count("embeddings", embeddings_count)?;
        Ok(())
    }

    fn validate_count(self, field: &str, actual: usize) -> Result<()> {
        let expected = match field {
            "documents" => self.documents,
            "chunks" => self.chunks,
            "embeddings" => self.embeddings,
            _ => None,
        };
        if let Some(expected) = expected {
            if expected != actual {
                return Err(Error::new(
                    ErrorCode::InvalidJsonlSchema,
                    format!("export manifest counts.{field} does not match JSONL row count"),
                )
                .with_detail("field", format!("counts.{field}"))
                .with_detail("expected_count", expected)
                .with_detail("actual_count", actual));
            }
        }
        Ok(())
    }
}

fn parse_export_manifest_counts(obj: &JsonObject) -> Result<ExportManifestCounts> {
    let Some(counts) = obj.get("counts").filter(|value| !value.is_null()) else {
        return Ok(ExportManifestCounts::default());
    };
    let counts = counts.as_object().ok_or_else(|| {
        Error::new(
            ErrorCode::InvalidJsonlSchema,
            "export manifest counts must be an object",
        )
        .with_detail("field", "counts")
    })?;
    for field in counts.keys() {
        if !matches!(field.as_str(), "documents" | "chunks" | "embeddings") {
            return Err(Error::new(
                ErrorCode::InvalidJsonlSchema,
                "export manifest counts contains unsupported field",
            )
            .with_detail("field", format!("counts.{field}")));
        }
    }
    Ok(ExportManifestCounts {
        documents: optional_count_field(counts, "documents")?,
        chunks: optional_count_field(counts, "chunks")?,
        embeddings: optional_count_field(counts, "embeddings")?,
    })
}

fn optional_count_field(obj: &JsonObject, field: &str) -> Result<Option<usize>> {
    obj.get(field)
        .filter(|value| !value.is_null())
        .map(|value| {
            json_usize(value).ok_or_else(|| {
                Error::new(
                    ErrorCode::InvalidJsonlSchema,
                    format!("counts.{field} must be a non-negative integer"),
                )
                .with_detail("field", format!("counts.{field}"))
            })
        })
        .transpose()
}

impl ImportWarning {
    fn missing_hash(artifact_kind: &str, field: &str) -> Self {
        Self {
            code: "missing_reproducibility_hash".into(),
            field: field.into(),
            artifact_kind: artifact_kind.into(),
            message: format!(
                "{field} is missing; import is allowed, but reproducibility is reduced"
            ),
        }
    }

    fn build_json(&self, builder: &JsonBuilder<'_>) -> Result<JsonValue> {
        builder.object([
            ("code", builder.string(&self.code)?),
            ("severity", builder.string("warning")?),
            ("path", JsonValue::Null),
            (
                "details",
                builder.object([
                    ("field", builder.string(&self.field)?),
                    ("artifact_kind", builder.string(&self.artifact_kind)?),
                ])?,
            ),
            ("field", builder.string(&self.field)?),
            ("artifact_kind", builder.string(&self.artifact_kind)?),
            ("message", builder.string(&self.message)?),
        ])
    }

    fn source_span_metadata_conflict() -> Self {
        Self {
            code: "source_span_metadata_conflict".into(),
            field: "source_span".into(),
            artifact_kind: "chunks".into(),
            message: "source_span conflicts with metadata byte/line boundary fields; source_span was used".into(),
        }
    }
}

impl Database {
    fn resolve_import_collection_shape(
        &self,
        manifest: &ImportManifest,
    ) -> Result<(usize, DistanceMetric)> {
        let existing = self
            .list_collections()?
            .into_iter()
            .find(|collection| collection.name == manifest.collection);
        let embedding_dim = match (manifest.embedding_dim, existing.as_ref()) {
            (Some(value), _) => value,
            (None, Some(collection)) => collection.embedding_dim,
            (None, None) => {
                return Err(Error::new(
                    ErrorCode::MissingRequiredField,
                    "embedding_dim is required when importing into a new collection",
                )
                .with_detail("field", "embedding_dim"));
            }
        };
        let distance_metric = match (manifest.distance_metric, existing.as_ref()) {
            (Some(value), _) => value,
            (None, Some(collection)) => collection.distance_metric,
            (None, None) => {
                return Err(Error::new(
                    ErrorCode::MissingRequiredField,
                    "distance_metric is required when importing into a new collection",
                )
                .with_detail("field", "distance_metric"));
            }
        };
        Ok((embedding_dim, distance_metric))
    }
}

fn required_manifest_string(obj: &JsonObject, field: &str) -> Result<String> {
    let value = required_string(obj, field)?;
    if value.trim().is_empty() {
        return Err(Error::new(
            ErrorCode::MissingRequiredField,
            format!("{field} is required"),
        )
        .with_detail("field", field));
    }
    Ok(value)
}

fn optional_manifest_string(obj: &JsonObject, field: &str) -> Result<Option<String>> {
    let Some(value) = obj.get(field).filter(|value| !value.is_null()) else {
        return Ok(None);
    };
    let value = value.as_str().ok_or_else(|| {
        Error::new(
            ErrorCode::InvalidJsonlSchema,
            format!("{field} must be a string"),
        )
        .with_detail("field", field)
    })?;
    if value.trim().is_empty() {
        return Err(Error::new(
            ErrorCode::MissingRequiredField,
            format!("{field} cannot be empty when provided"),
        )
        .with_detail("field", field));
    }
    Ok(Some(value.to_owned()))
}

fn optional_usize(obj: &JsonObject, field: &str) -> Result<Option<usize>> {
    obj.get(field)
        .filter(|value| !value.is_null())
        .map(|value| {
            json_usize(value).ok_or_else(|| {
                Error::new(
                    ErrorCode::InvalidJsonlSchema,
                    format!("{field} must be a non-negative integer"),
                )
                .with_detail("field", field)
            })
        })
        .transpose()
}

fn optional_distance_metric(obj: &JsonObject, field: &str) -> Result<Option<DistanceMetric>> {
    obj.get(field)
        .filter(|value| !value.is_null())
        .map(|value| {
            let value = value.as_str().ok_or_else(|| {
                Error::new(
                    ErrorCode::InvalidJsonlSchema,
                    format!("{field} must be a string"),
                )
                .with_detail("field", field)
            })?;
            DistanceMetric::parse(value)
        })
        .transpose()
}

fn optional_hash(hashes: Option<&JsonObject>, field: &str) -> Result<Option<String>> {
    let Some(hashes) = hashes else {
        return Ok(None);
    };
    hashes
        .get(field)
        .filter(|value| !value.is_null())
        .map(|value| {
            value
                .as_str()
                .map(ToOwned::to_owned)
                .ok_or_else(|| {
                    Error::new(
                        ErrorCode::InvalidJsonlSchema,
                        format!("hashes.{field} must be a string"),
                    )
                    .with_detail("field", format!("hashes.{field}"))
                })
                .and_then(|value| {
                    if value.trim().is_empty() {
                        Err(Error::new(
                            ErrorCode::MissingRequiredField,
                            format!("hashes.{field} cannot be empty when provided"),
                        )
                        .with_detail("field", format!("hashes.{field}")))
                    } else {
                        Ok(value)
                    }
                })
        })
        .transpose()
}

fn validate_optional_hash(
    kind: &str,
    path: &Path,
    expected: Option<&str>,
    actual: &str,
) -> Result<()> {
    let Some(expected) = expected else {
        return Ok(());
    };
    if actual != expected {
        return Err(
            Error::new(ErrorCode::ChecksumMismatch, format!("{kind} hash mismatch"))
                .with_detail("field", format!("hashes.{kind}_sha256"))
                .with_detail("path", path.display())
                .with_detail("artifact_kind", kind)
                .with_detail("expected_sha256", expected)
                .with_detail("actual_sha256", actual),
        );
    }
    Ok(())
}

impl ImportSummary {
    pub fn to_json(&self) -> JsonValue {
        self.build_json(&JsonBuilder::host())
            .expect("host DTO construction is infallible")
    }

    pub(crate) fn build_json(&self, builder: &JsonBuilder<'_>) -> Result<JsonValue> {
        let skipped = self.documents_skipped + self.chunks_skipped + self.embeddings_skipped;
        let replaced = self.documents_replaced + self.chunks_replaced + self.embeddings_replaced;
        builder.object([
            ("tokenizer", self.tokenizer.build_json(builder)?),
            ("skipped_files", JsonValue::Array(Vec::new())),
            (
                "dataset_id",
                builder.optional_str(self.dataset_id.as_deref())?,
            ),
            (
                "documents_seen",
                JsonValue::Number(self.documents_seen as f64),
            ),
            ("chunks_seen", JsonValue::Number(self.chunks_seen as f64)),
            (
                "embeddings_seen",
                JsonValue::Number(self.embeddings_seen as f64),
            ),
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
            (
                "warnings",
                builder.array(
                    self.warnings
                        .iter()
                        .map(|warning| warning.build_json(builder)),
                )?,
            ),
            ("missing_chunks", JsonValue::Number(0.0)),
            ("model_mismatches", JsonValue::Number(0.0)),
            ("errors", JsonValue::Number(0.0)),
        ])
    }
}

impl ExportSummary {
    pub fn to_json(&self) -> JsonValue {
        object_from_pairs([
            ("documents", JsonValue::Number(self.documents as f64)),
            (
                "documents_exported",
                JsonValue::Number(self.documents as f64),
            ),
            ("chunks", JsonValue::Number(self.chunks as f64)),
            ("chunks_exported", JsonValue::Number(self.chunks as f64)),
            ("embeddings", JsonValue::Number(self.embeddings as f64)),
            (
                "embeddings_exported",
                JsonValue::Number(self.embeddings as f64),
            ),
            ("manifest", JsonValue::String(self.manifest.clone())),
            (
                "documents_sha256",
                JsonValue::String(self.documents_sha256.clone()),
            ),
            (
                "chunks_sha256",
                JsonValue::String(self.chunks_sha256.clone()),
            ),
            (
                "embeddings_sha256",
                JsonValue::String(self.embeddings_sha256.clone()),
            ),
        ])
    }
}

pub(crate) struct JsonlChunk {
    pub(crate) document_external_id: String,
    pub(crate) chunk: ChunkInput,
    pub(crate) warnings: Vec<ImportWarning>,
}

#[cfg(test)]
fn read_documents_jsonl(path: &Path) -> Result<Vec<JsonlDocument>> {
    parse_documents_jsonl(path, open_bounded(path, MAX_JSONL_BYTES)?)
}

#[cfg(test)]
fn parse_documents_jsonl(path: &Path, source: impl Read) -> Result<Vec<JsonlDocument>> {
    parse_documents_jsonl_in(path, source, &JsonOperationContext::standalone())
        .map(|rows| rows.values)
}

#[cfg(test)]
fn parse_documents_jsonl_in(
    path: &Path,
    source: impl Read,
    context: &JsonOperationContext,
) -> Result<ParsedRows<JsonlDocument>> {
    let mut rows = ParsedRows::new(context);
    visit_documents_jsonl_in(path, source, context, MAX_JSONL_BYTES, |_, row, work| {
        rows.push(row, work);
        Ok(())
    })?;
    Ok(rows)
}

pub(crate) fn visit_documents_jsonl_in(
    path: &Path,
    source: impl Read,
    context: &JsonOperationContext,
    byte_limit: u64,
    consume: impl FnMut(usize, JsonlDocument, JsonReservation) -> Result<()>,
) -> Result<()> {
    visit_jsonl_rows_in(
        path,
        JsonLines::new(source, JsonSchema::DocumentRow, context, byte_limit),
        |obj| {
            reject_unknown_jsonl_fields(obj, &["external_id", "title", "source_uri", "metadata"])?;
            Ok(JsonlDocument {
                external_id: required_nonempty_jsonl_string(obj, "external_id")?,
                title: optional_jsonl_string(obj, "title")?,
                source_uri: optional_jsonl_string(obj, "source_uri")?,
                metadata: required_metadata(obj)?,
            })
        },
        |row| Some((&row.metadata, JsonInputKind::DocumentMetadata)),
        consume,
    )
}

fn required_metadata(obj: &mut JsonObject) -> Result<JsonValue> {
    ensure_object(obj.remove("metadata").ok_or_else(|| {
        Error::new(
            ErrorCode::MissingRequiredField,
            "metadata is required; use an empty object",
        )
        .with_detail("field", "metadata")
    })?)
}

#[cfg(test)]
fn read_chunks_jsonl(path: &Path) -> Result<Vec<JsonlChunk>> {
    parse_chunks_jsonl(path, open_bounded(path, MAX_JSONL_BYTES)?)
}

#[cfg(test)]
fn parse_chunks_jsonl(path: &Path, source: impl Read) -> Result<Vec<JsonlChunk>> {
    parse_chunks_jsonl_in(path, source, &JsonOperationContext::standalone()).map(|rows| rows.values)
}

#[cfg(test)]
fn parse_chunks_jsonl_in(
    path: &Path,
    source: impl Read,
    context: &JsonOperationContext,
) -> Result<ParsedRows<JsonlChunk>> {
    let mut rows = ParsedRows::new(context);
    visit_chunks_jsonl_in(path, source, context, MAX_JSONL_BYTES, |_, row, work| {
        rows.push(row, work);
        Ok(())
    })?;
    Ok(rows)
}

pub(crate) fn visit_chunks_jsonl_in(
    path: &Path,
    source: impl Read,
    context: &JsonOperationContext,
    byte_limit: u64,
    consume: impl FnMut(usize, JsonlChunk, JsonReservation) -> Result<()>,
) -> Result<()> {
    visit_jsonl_rows_in(
        path,
        JsonLines::new(source, JsonSchema::ChunkRow, context, byte_limit),
        |obj| {
            reject_unknown_jsonl_fields(
                obj,
                &[
                    "document_external_id",
                    "chunk_external_id",
                    "ordinal",
                    "text",
                    "metadata",
                    "source_span",
                ],
            )?;
            let metadata = required_metadata(obj)?;
            let metadata_source_span = SourceSpan::try_from_metadata(&metadata);
            let mut warnings = Vec::new();
            let source_span = if let Some(value) = obj.get("source_span") {
                let explicit = SourceSpan::from_json(value)?;
                if matches!(&metadata_source_span, Ok(Some(span)) if span != &explicit)
                    || metadata_source_span.is_err()
                {
                    warnings.push(ImportWarning::source_span_metadata_conflict());
                }
                // Keep an empty explicit span until write normalization so metadata cannot fill it.
                Some(explicit)
            } else {
                metadata_source_span?
            };
            Ok(JsonlChunk {
                document_external_id: required_nonempty_jsonl_string(obj, "document_external_id")?,
                chunk: ChunkInput {
                    chunk_external_id: required_nonempty_jsonl_string(obj, "chunk_external_id")?,
                    ordinal: required_usize(obj, "ordinal")?,
                    text: required_nonempty_jsonl_string(obj, "text")?,
                    metadata,
                    source_span,
                },
                warnings,
            })
        },
        |row| Some((&row.chunk.metadata, JsonInputKind::ChunkMetadata)),
        consume,
    )
}

pub(crate) fn read_embeddings_jsonl(path: &Path) -> Result<Vec<EmbeddingInput>> {
    parse_embeddings_jsonl_in(
        path,
        open_bounded(path, MAX_JSONL_BYTES)?,
        &JsonOperationContext::standalone(),
    )
    .map(|rows| rows.values)
}

fn parse_embeddings_jsonl_in(
    path: &Path,
    source: impl Read,
    context: &JsonOperationContext,
) -> Result<ParsedRows<EmbeddingInput>> {
    let mut rows = ParsedRows::new(context);
    visit_embeddings_jsonl_in(path, source, context, MAX_JSONL_BYTES, |_, row, work| {
        rows.push(row, work);
        Ok(())
    })?;
    Ok(rows)
}

pub(crate) fn visit_embeddings_jsonl_in(
    path: &Path,
    source: impl Read,
    context: &JsonOperationContext,
    byte_limit: u64,
    consume: impl FnMut(usize, EmbeddingInput, JsonReservation) -> Result<()>,
) -> Result<()> {
    visit_jsonl_rows_in(
        path,
        JsonLines::new(source, JsonSchema::EmbeddingRow, context, byte_limit),
        |obj| {
            reject_unknown_jsonl_fields(
                obj,
                &["chunk_external_id", "model_name", "dim", "vector"],
            )?;
            let values = required_array(obj, "vector")?;
            let mut vector = Vec::new();
            let mut conversion = context.reservation();
            conversion.reserve_vec(&mut vector, values.len(), JsonInputKind::JsonlRecord)?;
            for value in values {
                let value = value.as_f64().ok_or_else(|| {
                    Error::new(
                        ErrorCode::InvalidVectorLength,
                        "embedding vector entries must be numbers",
                    )
                })?;
                let value = value as f32;
                if !value.is_finite() {
                    return Err(Error::new(
                        ErrorCode::InvalidVectorLength,
                        "embedding vector entries must be finite f32 values",
                    ));
                }
                vector.push(value);
            }
            Ok(EmbeddingInput {
                chunk_external_id: required_nonempty_jsonl_string(obj, "chunk_external_id")?,
                model_name: required_nonempty_jsonl_string(obj, "model_name")?,
                dim: required_usize(obj, "dim")?,
                vector,
            })
        },
        |_| None,
        consume,
    )
}

fn reject_unknown_jsonl_fields(obj: &JsonObject, allowed: &[&str]) -> Result<()> {
    for field in obj.keys() {
        if !allowed.contains(&field.as_str()) {
            return Err(Error::new(
                ErrorCode::InvalidJsonlSchema,
                "JSONL row contains unsupported field",
            )
            .with_detail("field", field));
        }
    }
    Ok(())
}

fn optional_jsonl_string(obj: &mut JsonObject, field: &str) -> Result<Option<String>> {
    let Some(value) = obj.get_mut(field).filter(|value| !value.is_null()) else {
        return Ok(None);
    };
    let JsonValue::String(value) = value else {
        return Err(Error::new(
            ErrorCode::InvalidJsonlSchema,
            format!("{field} must be a string or null"),
        )
        .with_detail("field", field));
    };
    if value.trim().is_empty() {
        return Err(Error::new(
            ErrorCode::MissingRequiredField,
            format!("{field} cannot be empty when provided"),
        )
        .with_detail("field", field));
    }
    Ok(Some(std::mem::take(value)))
}

fn required_nonempty_jsonl_string(obj: &mut JsonObject, field: &str) -> Result<String> {
    let Some(JsonValue::String(value)) = obj.get_mut(field) else {
        return Err(Error::new(
            ErrorCode::MissingRequiredField,
            format!("missing string field {field}"),
        ));
    };
    if value.trim().is_empty() {
        return Err(Error::new(
            ErrorCode::MissingRequiredField,
            format!("{field} cannot be empty"),
        )
        .with_detail("field", field));
    }
    Ok(std::mem::take(value))
}

struct ParsedRows<T> {
    values: Vec<T>,
    work: JsonReservation,
}

impl<T> ParsedRows<T> {
    fn new(context: &JsonOperationContext) -> Self {
        Self {
            values: Vec::new(),
            work: context.reservation(),
        }
    }

    fn push(&mut self, row: T, work: JsonReservation) {
        self.values.push(row);
        self.work.merge(work);
    }
}

fn visit_jsonl_rows_in<T>(
    path: &Path,
    reader: Result<JsonLines<impl Read>>,
    mut parse_row: impl FnMut(&mut JsonObject) -> Result<T>,
    retained_json: impl Fn(&T) -> Option<(&JsonValue, JsonInputKind)>,
    mut consume: impl FnMut(usize, T, JsonReservation) -> Result<()>,
) -> Result<()> {
    let reader = reader.map_err(|error| error.with_detail("path", path.display()))?;
    for row in reader {
        let (line_number, parsed) =
            row.map_err(|error| error.with_detail("path", path.display()))?;
        let (mut value, mut work) = parsed.into_parts();
        let obj = value.as_object_mut().ok_or_else(|| {
            Error::new(ErrorCode::InvalidJsonlSchema, "JSONL row must be an object")
                .with_detail("path", path.display())
                .with_detail("line", line_number)
        })?;
        let row = parse_row(obj).map_err(|err| with_jsonl_location(err, path, line_number))?;
        drop(value);
        if let Some((value, kind)) = retained_json(&row) {
            value
                .reconcile_retained_in(kind, &mut work)
                .map_err(|err| with_jsonl_location(err, path, line_number))?;
        } else {
            work.retain(0);
        }
        consume(line_number, row, work)
            .map_err(|err| with_jsonl_location(err, path, line_number))?;
    }
    Ok(())
}

fn with_jsonl_location(err: Error, path: &Path, line: usize) -> Error {
    let has_path = err.detail().iter().any(|(key, _)| key == "path");
    let has_line = err.detail().iter().any(|(key, _)| key == "line");
    let mut located = err;
    if !has_path {
        located = located.with_detail("path", path.display());
    }
    if !has_line {
        located = located.with_detail("line", line);
    }
    located
}

fn merge_insert_summary(summary: &mut ImportSummary, inserted: InsertSummary) {
    summary.documents_inserted += inserted.documents_inserted;
    summary.chunks_inserted += inserted.chunks_inserted;
    summary.embeddings_inserted += inserted.embeddings_inserted;
    summary.documents_skipped += inserted.documents_skipped;
    summary.chunks_skipped += inserted.chunks_skipped;
    summary.embeddings_skipped += inserted.embeddings_skipped;
    summary.documents_replaced += inserted.documents_replaced;
    summary.chunks_replaced += inserted.chunks_replaced;
    summary.embeddings_replaced += inserted.embeddings_replaced;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json::parse;
    use crate::model::{CollectionConfig, DocumentInput, VectorIndexConfig};
    use crate::profile::DatabaseConfig;
    use std::collections::BTreeMap;

    thread_local! {
        pub(super) static EXPORT_CHUNK_READS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
        pub(super) static EXPORT_SPAN_BUILDS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
        pub(super) static EXPORT_MANIFEST_WRITES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    }

    fn json_node_bytes() -> usize {
        11 * std::mem::size_of::<(String, JsonValue)>() + 16 * std::mem::size_of::<usize>()
    }

    fn export_test_collection(db: &Database) -> crate::CollectionHandle {
        db.create_collection(CollectionConfig {
            name: "notes".into(),
            embedding_dim: 2,
            distance_metric: DistanceMetric::Cosine,
            embedding_model: None,
            vectors_normalized: false,
            vector_index: VectorIndexConfig::Exact,
            tokenizer: Default::default(),
        })
        .unwrap()
    }

    #[test]
    fn export_real_chunk_preparation_yields_buffers_at_exact_row_boundary() {
        let dir = temp_dir("export-real-chunk-yield");
        let path = dir.join("data.coffer");
        let db = Database::create(&path, DatabaseConfig::default()).unwrap();
        let collection = export_test_collection(&db);
        let span = SourceSpan {
            start: Some(u64::MAX - 1),
            end: Some(u64::MAX),
            line_start: Some(1),
            line_end: Some(u32::MAX),
        };
        collection
            .insert_document(DocumentInput {
                external_id: "doc".into(),
                title: None,
                source_uri: None,
                metadata: JsonValue::Object(BTreeMap::new()),
                chunks: (0..2)
                    .map(|ordinal| ChunkInput {
                        chunk_external_id: format!("c{ordinal}"),
                        ordinal,
                        text: "x".repeat(65536),
                        metadata: JsonValue::Object(BTreeMap::new()),
                        source_span: Some(span.clone()),
                    })
                    .collect(),
            })
            .unwrap();
        collection
            .put_embeddings(vec![EmbeddingInput {
                chunk_external_id: "c0".into(),
                model_name: "model".into(),
                dim: 2,
                vector: vec![1.0, 0.0],
            }])
            .unwrap();
        let context = db.json_context();
        let capacity = context.available_workspace_bytes();
        let timestamp = db
            .with_read_snapshot(|catalog, storage| {
                let mut maximum = 0;
                for chunk in catalog.live_chunks() {
                    let crate::storage::RecordPayload::ChunkPut(value) = storage
                        .read_payload(catalog.record_ref_for_chunk(&chunk.chunk_id).unwrap())?
                    else {
                        panic!("chunk fixture");
                    };
                    assert!(matches!(value.created_at.len(), 20 | 24));
                    maximum = maximum.max(value.created_at.len());
                }
                Ok(maximum)
            })
            .unwrap();
        let row_bytes = 3 * 20 + 2 + 65536 + 64 + timestamp + json_node_bytes() + 26 + 40;
        let minimum = row_bytes + 129 * crate::json::test_write_frame_bytes();
        let before = fs::read(&path).unwrap();
        for gap in [-1isize, 0, 1] {
            let output = dir.join(format!("out-{gap}"));
            fs::create_dir(&output).unwrap();
            for name in EXPORT_FILES {
                fs::write(output.join(name), b"previous").unwrap();
            }
            let available = 3 * crate::temporary::test_file_management_bytes(&output)
                + minimum.checked_add_signed(gap).unwrap();
            let mut held = context.workspace_reservation();
            held.grow_to(
                capacity - available,
                capacity - available,
                "operation_scratch",
            )
            .unwrap();
            EXPORT_CHUNK_READS.set(0);
            EXPORT_SPAN_BUILDS.set(0);
            let result = db.export_collection_jsonl("notes", &output);
            assert_eq!(EXPORT_CHUNK_READS.get(), if gap < 0 { 2 } else { 3 });
            assert_eq!(EXPORT_SPAN_BUILDS.get(), if gap < 0 { 1 } else { 2 });
            if gap < 0 {
                let error = result.unwrap_err();
                assert_eq!(error.code(), ErrorCode::RecoveryError);
                assert!(crate::json::is_memory_budget_error(&error));
                for name in EXPORT_FILES {
                    assert_eq!(fs::read(output.join(name)).unwrap(), b"previous");
                }
            } else {
                let summary = result.unwrap();
                assert_eq!(
                    (summary.documents, summary.chunks, summary.embeddings),
                    (1, 2, 1)
                );
                for (name, hash, rows) in [
                    ("documents.jsonl", summary.documents_sha256, 1),
                    ("chunks.jsonl", summary.chunks_sha256, 2),
                    ("embeddings.jsonl", summary.embeddings_sha256, 1),
                ] {
                    let bytes = fs::read(output.join(name)).unwrap();
                    assert_eq!(sha256_hex(&bytes), hash);
                    assert_eq!(std::str::from_utf8(&bytes).unwrap().lines().count(), rows);
                }
                let rows = fs::read_to_string(output.join("chunks.jsonl")).unwrap();
                for (ordinal, line) in rows.lines().enumerate() {
                    let value = parse(line).unwrap();
                    assert_eq!(value.get("ordinal").unwrap().as_f64(), Some(ordinal as f64));
                    assert_eq!(
                        value.get("text").unwrap().as_str(),
                        Some("x".repeat(65536).as_str())
                    );
                    assert_eq!(value.get("source_span").unwrap(), &span.to_json());
                }
            }
            assert_eq!(fs::read_dir(&output).unwrap().count(), 4);
            held.grow_to(capacity, capacity, "operation_scratch")
                .unwrap();
            assert!(held
                .grow_to(capacity + 1, capacity + 1, "operation_scratch")
                .is_err());
            assert_eq!(fs::read(&path).unwrap(), before);
        }
        drop(collection);
        drop(db);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn export_span_partial_construction_retry_releases_failed_owner() {
        let dir = temp_dir("export-partial-span-yield");
        let path = dir.join("data.coffer");
        let db = Database::create(&path, DatabaseConfig::default()).unwrap();
        let collection = export_test_collection(&db);
        let span = SourceSpan {
            start: Some(u64::MAX - 1),
            end: Some(u64::MAX),
            line_start: Some(1),
            line_end: Some(u32::MAX),
        };
        collection
            .insert_document(DocumentInput {
                external_id: "doc".into(),
                title: None,
                source_uri: None,
                metadata: JsonValue::Object(BTreeMap::new()),
                chunks: vec![ChunkInput {
                    chunk_external_id: "chunk".into(),
                    ordinal: 0,
                    text: "text".into(),
                    metadata: JsonValue::Object(BTreeMap::new()),
                    source_span: Some(span.clone()),
                }],
            })
            .unwrap();
        db.with_read_snapshot(|catalog, storage| {
            let chunk = catalog.live_chunks().next().unwrap();
            let record_ref = catalog.record_ref_for_chunk(&chunk.chunk_id).unwrap();
            let crate::storage::RecordPayload::ChunkPut(value) =
                storage.read_payload(record_ref)?
            else {
                panic!("chunk fixture");
            };
            let read_bytes = 3 * 20 + 5 + 4 + 64 + value.created_at.len();
            let minimum = read_bytes + json_node_bytes() + 26 + 40;
            drop(value);
            for parent in [false, true] {
                for gap in [-1isize, 0, 1] {
                    let limit = minimum + 7;
                    let context = JsonOperationContext::new(std::sync::Arc::new(
                        crate::json::WorkLedger::with_limits(
                            if parent { usize::MAX } else { limit },
                            usize::MAX,
                            if parent { limit } else { usize::MAX },
                        ),
                    ));
                    let disk = std::sync::Arc::new(TemporaryDiskBudget::new(4096));
                    let (file, mut guard) =
                        create_budgeted_temporary_file_in(&dir, "span-yield", "tmp", disk.clone())
                            .unwrap();
                    let mut writer = ExportWriter::new_in(file, &mut guard, 13, &context).unwrap();
                    let available = minimum.checked_add_signed(gap).unwrap();
                    let mut held = context.workspace_reservation();
                    held.grow_to(limit - available, limit - available, "operation_scratch")
                        .unwrap();
                    let mut reader = storage.payload_reader()?;
                    EXPORT_CHUNK_READS.set(0);
                    EXPORT_SPAN_BUILDS.set(0);
                    let mut attempts = 0;
                    let result =
                        retry_after_buffer_reclaim(&mut [&mut writer], &mut || false, |_| {
                            attempts += 1;
                            ExportChunk::read_in(&mut reader, record_ref, &context)
                        });
                    assert_eq!(attempts, 2);
                    assert_eq!(EXPORT_CHUNK_READS.get(), 2);
                    assert_eq!(EXPORT_SPAN_BUILDS.get(), 2);
                    match result {
                        Ok(prepared) => {
                            assert!(gap >= 0);
                            assert_eq!(prepared.source_span, span.to_json());
                            assert_eq!(context.available_workspace_bytes(), gap as usize);
                            drop(prepared);
                        }
                        Err(error) => {
                            assert_eq!(gap, -1);
                            assert_eq!(error.code(), ErrorCode::RecoveryError);
                            assert!(crate::json::is_memory_budget_error(&error));
                        }
                    }
                    drop(writer);
                    assert_eq!(fs::metadata(guard.path()).unwrap().len(), 0);
                    held.grow_to(limit, limit, "operation_scratch").unwrap();
                    assert!(held
                        .grow_to(limit + 1, limit + 1, "operation_scratch")
                        .is_err());
                    drop(guard);
                    assert_eq!(disk.used(), 0);
                }
            }
            Ok(())
        })
        .unwrap();
        drop(collection);
        drop(db);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn export_real_manifest_yields_buffer_before_first_byte_at_exact_boundary() {
        let dir = temp_dir("export-real-manifest-yield");
        let path = dir.join("data.coffer");
        let db = Database::create(&path, DatabaseConfig::default()).unwrap();
        let collection = export_test_collection(&db);
        let capacity = db.json_context().available_workspace_bytes();
        let keys = |values: &[&str]| values.iter().map(|value| value.len()).sum::<usize>();
        let strings =
            |values: &[&str]| values.iter().map(|value| value.len().max(8)).sum::<usize>();
        let retained = 7 * json_node_bytes()
            + keys(&[
                "tokenizer",
                "schema_version",
                "cofferdb_version",
                "file_format_version",
                "collection",
                "exported_at",
                "embedding_model",
                "embedding_dim",
                "distance_metric",
                "documents",
                "chunks",
                "embeddings",
                "counts",
                "hashes",
            ])
            + keys(&[
                "documents",
                "chunks",
                "embeddings",
                "documents_sha256",
                "chunks_sha256",
                "embeddings_sha256",
            ])
            + keys(&[
                "name",
                "version",
                "profile",
                "language_hint",
                "config",
                "case_fold",
                "split",
                "stop_words",
            ])
            + strings(&[
                "default_en",
                "0.1",
                "latin_code_default",
                "en",
                "lower",
                "unicode_punctuation",
                "default_en",
            ])
            + strings(&[
                "0.1",
                env!("CARGO_PKG_VERSION"),
                "notes",
                "cosine",
                "documents.jsonl",
                "chunks.jsonl",
                "embeddings.jsonl",
            ])
            + 24
            + 3 * 64;
        let minimum = retained + 4 * crate::json::test_write_frame_bytes();
        let before = fs::read(&path).unwrap();
        for gap in [-1isize, 0, 1] {
            let output = dir.join(format!("out-{gap}"));
            fs::create_dir(&output).unwrap();
            for name in EXPORT_FILES {
                fs::write(output.join(name), b"previous").unwrap();
            }
            let available = 4 * crate::temporary::test_file_management_bytes(&output)
                + minimum.checked_add_signed(gap).unwrap();
            let context = db.json_context();
            let mut held = context.workspace_reservation();
            held.grow_to(
                capacity - available,
                capacity - available,
                "operation_scratch",
            )
            .unwrap();
            EXPORT_MANIFEST_WRITES.set(0);
            let result = db.export_collection_jsonl("notes", &output);
            assert_eq!(EXPORT_MANIFEST_WRITES.get(), 2);
            if gap < 0 {
                let error = result.unwrap_err();
                assert_eq!(error.code(), ErrorCode::RecoveryError);
                assert!(crate::json::is_memory_budget_error(&error));
                for name in EXPORT_FILES {
                    assert_eq!(fs::read(output.join(name)).unwrap(), b"previous");
                }
            } else {
                let summary = result.unwrap();
                assert_eq!(
                    (summary.documents, summary.chunks, summary.embeddings),
                    (0, 0, 0)
                );
                let manifest = fs::read_to_string(output.join("export_manifest.json")).unwrap();
                let value = parse(&manifest).unwrap();
                assert_eq!(
                    value.get("tokenizer").unwrap(),
                    &crate::TokenizerConfig::default().to_json()
                );
                assert!(crate::time::parse_rfc3339_millis(
                    value.get("exported_at").unwrap().as_str().unwrap()
                )
                .is_some());
                let hashes = value.get("hashes").unwrap();
                for (name, key) in [
                    ("documents.jsonl", "documents_sha256"),
                    ("chunks.jsonl", "chunks_sha256"),
                    ("embeddings.jsonl", "embeddings_sha256"),
                ] {
                    let bytes = fs::read(output.join(name)).unwrap();
                    assert!(bytes.is_empty());
                    assert_eq!(
                        hashes.get(key).unwrap().as_str(),
                        Some(sha256_hex(&bytes).as_str())
                    );
                }
            }
            assert_eq!(fs::read_dir(&output).unwrap().count(), 4);
            assert_eq!(fs::read(&path).unwrap(), before);
            held.grow_to(capacity, capacity, "operation_scratch")
                .unwrap();
            assert!(held
                .grow_to(capacity + 1, capacity + 1, "operation_scratch")
                .is_err());
        }
        drop(collection);
        drop(db);
        fs::remove_dir_all(dir).unwrap();
    }

    fn temp_dir(name: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!("cofferdb-import-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap();
        path
    }

    fn write_file(dir: &Path, name: &str, content: &str) -> String {
        let path = dir.join(name);
        fs::write(&path, content).unwrap();
        sha256_hex(&fs::read(path).unwrap())
    }

    fn detail_value<'a>(err: &'a Error, key: &str) -> Option<&'a str> {
        err.detail()
            .iter()
            .find(|(field, _)| field == key)
            .map(|(_, value)| value.as_str())
    }

    #[test]
    fn manifest_reader_uses_canonical_unit_policy_not_the_old_file_size_cap() {
        let dir = temp_dir("manifest-streaming-policy");
        let path = dir.join("manifest.json");
        let value =
            object_from_pairs([("dataset_id", JsonValue::String("x".repeat(1_048_576 + 1)))]);
        fs::write(&path, value.to_compact_string()).unwrap();
        assert_eq!(read_manifest(&path).unwrap(), value);
        fs::write(&path, format!("{}{{}}", " ".repeat(9 * 1_048_576))).unwrap();
        assert_eq!(read_manifest(&path).unwrap(), JsonValue::object());
        fs::write(
            &path,
            format!(
                "\"{}\"",
                "x".repeat(crate::json::JsonProcessingPolicyV1::MAX_UNIT_CANONICAL_BYTES)
            ),
        )
        .unwrap();
        let error = read_manifest(&path).unwrap_err();
        assert_eq!(
            detail_value(&error, "json_limit_reason"),
            Some("unit_bytes_exceeded")
        );
        assert_eq!(detail_value(&error, "json_input_kind"), Some("manifest"));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn export_refuses_to_replace_its_database_under_any_output_name() {
        for name in EXPORT_FILES {
            let dir = temp_dir(&format!("export-source-conflict-{name}"));
            let source = dir.join(name);
            let db = Database::create(&source, DatabaseConfig::default()).unwrap();
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
            for other in EXPORT_FILES.into_iter().filter(|other| *other != name) {
                fs::write(dir.join(other), b"existing export").unwrap();
            }
            let before: Vec<_> = EXPORT_FILES
                .map(|name| fs::read(dir.join(name)).unwrap())
                .into();
            let entries = || {
                fs::read_dir(&dir)
                    .unwrap()
                    .map(|entry| entry.unwrap().file_name())
                    .collect::<std::collections::BTreeSet<_>>()
            };
            let before_entries = entries();
            assert_eq!(
                db.export_collection_jsonl("notes", &dir)
                    .unwrap_err()
                    .code(),
                ErrorCode::PermissionDenied
            );
            for (name, expected) in EXPORT_FILES.into_iter().zip(before) {
                assert_eq!(fs::read(dir.join(name)).unwrap(), expected);
            }
            assert_eq!(entries(), before_entries);
            drop(db);
            fs::remove_dir_all(dir).unwrap();
        }
    }

    #[cfg(unix)]
    #[test]
    fn export_rejects_source_links_and_never_follows_other_output_symlinks() {
        use std::os::unix::fs::symlink;
        let dir = temp_dir("export-source-links");
        let source = dir.join("data.coffer");
        let db = Database::create(&source, DatabaseConfig::default()).unwrap();
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
        let before = fs::read(&source).unwrap();
        for hard_link in [false, true] {
            for name in EXPORT_FILES {
                let output = dir.join(format!("{hard_link}-{name}"));
                fs::create_dir(&output).unwrap();
                let target = output.join(name);
                if hard_link {
                    fs::hard_link(&source, &target).unwrap();
                } else {
                    symlink(&source, &target).unwrap();
                }
                assert_eq!(
                    db.export_collection_jsonl("notes", &output)
                        .unwrap_err()
                        .code(),
                    ErrorCode::PermissionDenied
                );
                assert_eq!(fs::read(&source).unwrap(), before);
                assert_eq!(fs::read_dir(&output).unwrap().count(), 1);
            }
        }
        let unrelated = dir.join("unrelated");
        fs::write(&unrelated, b"do not truncate link targets").unwrap();
        let output = dir.join("safe-export");
        fs::create_dir(&output).unwrap();
        symlink(&unrelated, output.join("documents.jsonl")).unwrap();
        db.export_collection_jsonl("notes", &output).unwrap();
        assert_eq!(
            fs::read(&unrelated).unwrap(),
            b"do not truncate link targets"
        );
        assert!(!fs::symlink_metadata(output.join("documents.jsonl"))
            .unwrap()
            .file_type()
            .is_symlink());
        assert_eq!(fs::read(&source).unwrap(), before);
        assert_eq!(fs::read_dir(&output).unwrap().count(), 4);
        drop(db);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn export_payload_budget_failure_preserves_previous_output_and_releases_each_row() {
        for large_vector in [false, true] {
            let dir = temp_dir("export-vector-workspace");
            let path = dir.join("data.coffer");
            let db = Database::create(&path, DatabaseConfig::default()).unwrap();
            let collection = db
                .create_collection(CollectionConfig {
                    name: "notes".into(),
                    embedding_dim: if large_vector { 8192 } else { 2 },
                    distance_metric: DistanceMetric::Cosine,
                    embedding_model: None,
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
                    chunks: (0..4)
                        .map(|ordinal| ChunkInput {
                            chunk_external_id: format!("chunk-{ordinal}"),
                            ordinal,
                            text: if large_vector {
                                "text".into()
                            } else {
                                "text ".repeat(32768)
                            },
                            metadata: JsonValue::object(),
                            source_span: None,
                        })
                        .collect(),
                })
                .unwrap();
            collection
                .put_embeddings(
                    (0..4)
                        .map(|index| EmbeddingInput {
                            chunk_external_id: format!("chunk-{index}"),
                            model_name: "model".into(),
                            dim: if large_vector { 8192 } else { 2 },
                            vector: vec![(index + 1) as f32; if large_vector { 8192 } else { 2 }],
                        })
                        .collect(),
                )
                .unwrap();
            let before = fs::read(&path).unwrap();
            let context = db.json_context();
            let limit = context.available_workspace_bytes();
            let failed = dir.join("previous");
            fs::create_dir(&failed).unwrap();
            for name in EXPORT_FILES {
                fs::write(failed.join(name), b"previous complete export").unwrap();
            }
            let output = dir.join("success");
            let vector_read_bytes = if large_vector {
                db.with_read_snapshot(|catalog, storage| {
                    let first = catalog.live_embeddings().next().unwrap();
                    let record_ref = catalog
                        .record_ref_for_embedding(&first.embedding_id)
                        .unwrap();
                    let retained = storage
                        .payload_reader()?
                        .read_embedding_in(record_ref, &context)?;
                    let timestamp_bytes = retained.embedding().created_at.len();
                    assert!(matches!(timestamp_bytes, 20 | 24));
                    Ok(3 * 20 + "model".len() + timestamp_bytes + 8192 * 4)
                })
                .unwrap()
            } else {
                0
            };
            for seed in 0..100usize {
                let mut held = context.workspace_reservation();
                let available = if large_vector {
                    3 * crate::temporary::test_file_management_bytes(&failed) + vector_read_bytes
                        - 1
                        - seed
                } else {
                    128 * 1024 + seed
                };
                held.grow_to(limit - available, limit - available, "operation_scratch")
                    .unwrap();
                let error = db.export_collection_jsonl("notes", &failed).unwrap_err();
                assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                assert_eq!(detail_value(&error, "resource"), Some("memory"));
                assert_eq!(detail_value(&error, "operation"), Some("operation_scratch"));
                for name in EXPORT_FILES {
                    assert_eq!(
                        fs::read(failed.join(name)).unwrap(),
                        b"previous complete export"
                    );
                }
                assert_eq!(fs::read_dir(&failed).unwrap().count(), 4);
                held.grow_to(limit, limit, "operation_scratch").unwrap();
                held.retain(limit - 1024 * 1024);
                let exported = db.export_collection_jsonl("notes", &output).unwrap();
                assert_eq!(exported.embeddings, 4);
                assert_eq!(exported.chunks, 4);
                let chunks = fs::read_to_string(output.join("chunks.jsonl")).unwrap();
                assert_eq!(chunks.lines().count(), 4);
                for (index, line) in chunks.lines().enumerate() {
                    let value = parse(line).unwrap();
                    assert_eq!(
                        value.get("chunk_external_id").unwrap().as_str(),
                        Some(format!("chunk-{index}").as_str())
                    );
                    assert_eq!(
                        value.get("text").unwrap().as_str().unwrap(),
                        if large_vector {
                            "text".into()
                        } else {
                            "text ".repeat(32768)
                        }
                    );
                }
                let text = fs::read_to_string(output.join("embeddings.jsonl")).unwrap();
                for (index, line) in text.lines().enumerate() {
                    let value = parse(line).unwrap();
                    assert_eq!(
                        value.get("chunk_external_id").unwrap().as_str(),
                        Some(format!("chunk-{index}").as_str())
                    );
                    let vector = value.get("vector").unwrap().as_array().unwrap();
                    assert_eq!(vector.len(), if large_vector { 8192 } else { 2 });
                    assert!(vector
                        .iter()
                        .all(|value| value.as_f64() == Some((index + 1) as f64)));
                }
                held.grow_to(limit, limit, "operation_scratch").unwrap();
            }
            assert_eq!(fs::read(&path).unwrap(), before);
            drop(collection);
            drop(db);
            fs::remove_dir_all(dir).unwrap();
        }
    }

    #[test]
    fn export_reuses_shared_json_capacity_between_rows_and_preserves_source_bytes() {
        let dir = temp_dir("export-shared-json-budget");
        let path = dir.join("data.coffer");
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
        let metadata =
            || object_from_pairs([("payload", JsonValue::String("x".repeat(1024 * 1024)))]);
        collection
            .insert_document(DocumentInput {
                external_id: "doc".into(),
                title: Some("\u{4e2d}\n\"".into()),
                source_uri: None,
                metadata: metadata(),
                chunks: (0..4)
                    .map(|ordinal| ChunkInput {
                        chunk_external_id: format!("doc#{ordinal}"),
                        ordinal,
                        text: "streamed payload".into(),
                        metadata: metadata(),
                        source_span: Some(SourceSpan {
                            start: Some(u64::MAX - 1),
                            end: Some(u64::MAX),
                            line_start: None,
                            line_end: None,
                        }),
                    })
                    .collect(),
            })
            .unwrap();
        collection
            .put_embeddings(vec![EmbeddingInput {
                chunk_external_id: "doc#0".into(),
                model_name: "model".into(),
                dim: 2,
                vector: vec![1.0, -0.0],
            }])
            .unwrap();
        let before = fs::read(&path).unwrap();
        let context = db.json_context();
        let capacity = context.available_workspace_bytes();
        let failed = dir.join("failed");
        let mut busy = context.reservation();
        busy.grow(
            capacity - 512 * 1024 - 3 * crate::temporary::test_file_management_bytes(&failed),
            JsonInputKind::StoredJson,
        )
        .unwrap();
        fs::create_dir(&failed).unwrap();
        for name in EXPORT_FILES {
            fs::write(failed.join(name), b"previous complete export").unwrap();
        }
        let error = db.export_collection_jsonl("notes", &failed).unwrap_err();
        assert_eq!(error.code(), ErrorCode::RecoveryError);
        assert_eq!(detail_value(&error, "json_input_kind"), Some("stored_json"));
        assert_eq!(detail_value(&error, "limit"), Some("524288"));
        for name in EXPORT_FILES {
            assert_eq!(
                fs::read(failed.join(name)).unwrap(),
                b"previous complete export"
            );
        }
        assert_eq!(fs::read_dir(&failed).unwrap().count(), 4);
        drop(busy);
        assert_eq!(context.available_workspace_bytes(), capacity);
        let mut busy = context.reservation();
        busy.grow(capacity - 3 * 1024 * 1024, JsonInputKind::StoredJson)
            .unwrap();
        let output = dir.join("export");
        let summary = db.export_collection_jsonl("notes", &output).unwrap();
        assert_eq!(
            (summary.documents, summary.chunks, summary.embeddings),
            (1, 4, 1)
        );
        let chunks = fs::read(output.join("chunks.jsonl")).unwrap();
        assert_eq!(summary.chunks_sha256, sha256_hex(&chunks));
        assert_eq!(
            summary.documents_sha256,
            sha256_hex(&fs::read(output.join("documents.jsonl")).unwrap()),
        );
        assert_eq!(
            summary.embeddings_sha256,
            sha256_hex(&fs::read(output.join("embeddings.jsonl")).unwrap()),
        );
        assert!(chunks.len() > 4 * 1024 * 1024);
        for line in std::str::from_utf8(&chunks).unwrap().lines() {
            let row = crate::json::parse_with_schema(line, JsonSchema::ChunkRow).unwrap();
            assert_eq!(
                row.get("source_span").unwrap().get("end").unwrap().as_str(),
                Some("18446744073709551615")
            );
            assert_eq!(row.get("metadata"), Some(&metadata()));
        }
        assert_eq!(fs::read(output.join("embeddings.jsonl")).unwrap(), b"{\"chunk_external_id\":\"doc#0\",\"dim\":2,\"model_name\":\"model\",\"vector\":[1,0]}\n");
        assert_eq!(fs::read(&path).unwrap(), before);
        busy.grow(3 * 1024 * 1024, JsonInputKind::StoredJson)
            .unwrap();
        drop(busy);
        drop(collection);
        drop(db);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn export_reads_only_the_selected_collection_payloads() {
        use std::io::{Seek, SeekFrom};

        let dir = temp_dir("export-selected-collection");
        let path = dir.join("data.coffer");
        let db = Database::create(&path, DatabaseConfig::default()).unwrap();
        for name in ["selected", "unrelated"] {
            let collection = db
                .create_collection(CollectionConfig {
                    name: name.into(),
                    embedding_dim: 2,
                    distance_metric: DistanceMetric::Cosine,
                    embedding_model: None,
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
                    metadata: JsonValue::Object(BTreeMap::new()),
                    chunks: vec![ChunkInput {
                        chunk_external_id: "doc#0".into(),
                        ordinal: 0,
                        text: format!("{name} text"),
                        metadata: JsonValue::Object(BTreeMap::new()),
                        source_span: None,
                    }],
                })
                .unwrap();
        }
        let offset = db
            .with_read_snapshot(|catalog, _| {
                let id = &catalog.collection_by_name("unrelated")?.collection_id;
                let chunk = catalog
                    .live_chunks()
                    .find(|chunk| &chunk.collection_id == id)
                    .unwrap();
                Ok(catalog
                    .record_ref_for_chunk(&chunk.chunk_id)
                    .unwrap()
                    .payload
                    .offset)
            })
            .unwrap();
        // Damage a foreign payload after opening: selecting another collection must not read it.
        let mut file = fs::OpenOptions::new().write(true).open(&path).unwrap();
        file.seek(SeekFrom::Start(offset)).unwrap();
        file.write_all(&[0xff]).unwrap();
        let output = dir.join("export");
        let exported = db.export_collection_jsonl("selected", &output).unwrap();
        assert_eq!(
            (exported.documents, exported.chunks, exported.embeddings),
            (1, 1, 0)
        );
        assert_eq!(
            exported.chunks_sha256,
            sha256_hex(&fs::read(output.join("chunks.jsonl")).unwrap())
        );
        assert_eq!(
            exported.embeddings_sha256,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        );
        let restored =
            Database::create(dir.join("restored.coffer"), DatabaseConfig::default()).unwrap();
        restored
            .import_manifest(output.join("export_manifest.json"), OnConflict::Error)
            .unwrap();
        assert_eq!(
            restored
                .collection("selected")
                .unwrap()
                .get_chunk_by_external_id("doc#0")
                .unwrap()
                .text,
            "selected text"
        );
        assert!(db
            .export_collection_jsonl("unrelated", dir.join("bad-export"))
            .is_err());
        drop(restored);
        drop(db);
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn import_manifest_validates_optional_hashes() {
        let dir = temp_dir("hash-ok");
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
        fs::write(
            dir.join("manifest.json"),
            format!(
                concat!(
                    r#"{{"schema_version":"0.1","collection":"notes","#,
                    r#""documents":"documents.jsonl","chunks":"chunks.jsonl","#,
                    r#""embedding_model":null,"embedding_dim":2,"distance_metric":"cosine","#,
                    r#""hashes":{{"documents_sha256":"{}","chunks_sha256":"{}"}}"#,
                    r#"}}"#
                ),
                documents, chunks
            ),
        )
        .unwrap();

        let manifest = ImportManifest::load(dir.join("manifest.json")).unwrap();
        manifest.validated_inputs(MAX_JSONL_BYTES).unwrap();
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn validated_input_snapshot_survives_source_replacement() {
        let dir = temp_dir("fixed-inputs");
        let documents = write_file(
            &dir,
            "documents.jsonl",
            r#"{"external_id":"doc","metadata":{}}"#,
        );
        let chunks = write_file(
            &dir,
            "chunks.jsonl",
            r#"{"document_external_id":"doc","chunk_external_id":"doc#1","ordinal":0,"text":"validated content","metadata":{}}"#,
        );
        let value = parse(&format!(r#"{{"schema_version":"0.1","collection":"notes","documents":"documents.jsonl","chunks":"chunks.jsonl","embedding_dim":2,"distance_metric":"cosine","hashes":{{"documents_sha256":"{documents}","chunks_sha256":"{chunks}"}}}}"#)).unwrap();
        let manifest = ImportManifest::from_value(&dir.join("manifest.json"), &value).unwrap();
        let mut inputs = manifest.validated_inputs(MAX_JSONL_BYTES).unwrap();
        fs::write(&manifest.documents, "not JSON").unwrap();
        fs::write(&manifest.chunks, r#"{"text":"unvalidated replacement"}"#).unwrap();
        let staged_paths = [
            inputs.documents.temporary.path().to_path_buf(),
            inputs.chunks.temporary.path().to_path_buf(),
        ];
        let docs =
            parse_documents_jsonl(&manifest.documents, inputs.documents.reader().unwrap()).unwrap();
        let rows = parse_chunks_jsonl(&manifest.chunks, inputs.chunks.reader().unwrap()).unwrap();
        assert_eq!(docs[0].external_id, "doc");
        assert_eq!(rows[0].chunk.text, "validated content");
        assert_eq!(
            sha256_reader(inputs.documents.reader().unwrap()).unwrap(),
            documents
        );
        assert_eq!(
            sha256_reader(inputs.chunks.reader().unwrap()).unwrap(),
            chunks
        );
        assert!(manifest.validated_inputs(MAX_JSONL_BYTES).is_err());
        drop(inputs);
        assert!(staged_paths.iter().all(|path| !path.exists()));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn manifest_streams_a_file_above_the_legacy_cap_but_obeys_total_staging_budget() {
        let dir = temp_dir("manifest-large-stream");
        let documents = dir.join("documents.jsonl");
        let mut output = std::io::BufWriter::new(File::create(&documents).unwrap());
        let whitespace = vec![b' '; 1_048_576];
        for id in 0..65 {
            output.write_all(&whitespace).unwrap();
            writeln!(output, "{{\"external_id\":\"doc-{id}\",\"metadata\":{{}}}}").unwrap();
        }
        output.flush().unwrap();
        drop(output);
        assert!(fs::metadata(&documents).unwrap().len() > MAX_JSONL_BYTES);
        fs::write(dir.join("chunks.jsonl"), "").unwrap();
        let manifest = dir.join("manifest.json");
        fs::write(&manifest, r#"{"schema_version":"0.1","collection":"notes","documents":"documents.jsonl","chunks":"chunks.jsonl","embedding_dim":2,"distance_metric":"cosine"}"#).unwrap();
        for cache in [1, 256] {
            let path = dir.join(format!("{cache}.coffer"));
            let db = Database::create(
                &path,
                DatabaseConfig {
                    cache_budget_mb: Some(cache),
                    ..DatabaseConfig::default()
                },
            )
            .unwrap();
            let before = fs::read(&path).unwrap();
            let disk = db.temporary_disk_budget();
            assert_eq!(disk.limit_bytes(), 4 * 1024 * 1024 * 1024);
            {
                let (_file, mut busy) =
                    create_budgeted_temporary_file_in(&dir, "busy-disk", "tmp", disk.clone())
                        .unwrap();
                busy.reserve_disk(disk.limit_bytes() - 1024).unwrap();
                let error = db
                    .import_manifest(&manifest, OnConflict::Error)
                    .unwrap_err();
                assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                assert_eq!(detail_value(&error, "operation"), Some("temporary_staging"));
                assert_eq!(detail_value(&error, "limit_bytes"), Some("4294967296"));
                assert_eq!(fs::read(&path).unwrap(), before);
                assert!(db.list_collections().unwrap().is_empty());
                assert_eq!(disk.used(), disk.limit_bytes() - 1024);
            }
            {
                assert_eq!(disk.used(), 0);
                let summary = db.import_manifest(&manifest, OnConflict::Error).unwrap();
                assert_eq!(summary.documents_seen, 65);
                assert_eq!(summary.documents_inserted, 65);
                assert_eq!(summary.chunks_seen, 0);
                let document = db
                    .collection("notes")
                    .unwrap()
                    .get_document_by_external_id("doc-64")
                    .unwrap();
                assert_eq!(document.external_id, "doc-64");
                assert!(document.metadata.as_object().unwrap().is_empty());
            }
            assert_eq!(db.temporary_disk_budget().used(), 0);
            drop(db);
        }
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn staged_inputs_use_bounded_reads_and_clean_up_every_failure() {
        struct Source<'a> {
            bytes: &'a [u8],
            calls: usize,
            fail: bool,
        }
        impl Read for Source<'_> {
            fn read(&mut self, output: &mut [u8]) -> std::io::Result<usize> {
                assert!(output.len() <= 8192);
                self.calls += 1;
                if self.calls % 2 == 1 {
                    return Err(std::io::ErrorKind::Interrupted.into());
                }
                if self.fail && self.calls > 2 {
                    return Err(std::io::ErrorKind::Other.into());
                }
                let count = output.len().min(self.bytes.len()).min(7);
                output[..count].copy_from_slice(&self.bytes[..count]);
                self.bytes = &self.bytes[count..];
                Ok(count)
            }
        }
        let dir = temp_dir("staging-lifecycle");
        let path = Path::new("documents.jsonl");
        let bytes = b"{\"external_id\":\"doc\",\"metadata\":{}}\n";
        let expected = sha256_hex(bytes);
        for case in 0..4 {
            let result = ValidatedInput::copy_from(
                Source {
                    bytes,
                    calls: 0,
                    fail: case == 1,
                },
                "documents",
                path,
                Some(if case == 2 { "wrong hash" } else { &expected }),
                if case == 3 { 8 } else { bytes.len() as u64 },
                &dir,
            );
            if case == 0 {
                let mut input = result.unwrap();
                assert_eq!(input.len, bytes.len() as u64);
                #[cfg(unix)]
                {
                    use std::os::unix::fs::PermissionsExt;
                    assert_eq!(
                        input.file.metadata().unwrap().permissions().mode() & 0o777,
                        0o600
                    );
                }
                for _ in 0..2 {
                    let mut actual = Vec::new();
                    input.reader().unwrap().read_to_end(&mut actual).unwrap();
                    assert_eq!(actual, bytes);
                }
            } else {
                let error = result.unwrap_err();
                assert_eq!(
                    error.code(),
                    match case {
                        1 => ErrorCode::IoError,
                        2 => ErrorCode::ChecksumMismatch,
                        _ => ErrorCode::ResourceExhausted,
                    }
                );
            }
            assert_eq!(fs::read_dir(&dir).unwrap().count(), 0);
        }
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn streamed_jsonl_handles_fragmented_unicode_crlf_and_final_rows() {
        struct Fragmented<'a>(&'a [u8], usize);
        impl Read for Fragmented<'_> {
            fn read(&mut self, output: &mut [u8]) -> std::io::Result<usize> {
                let count = output.len().min(self.0.len()).min(self.1);
                output[..count].copy_from_slice(&self.0[..count]);
                self.0 = &self.0[count..];
                Ok(count)
            }
        }
        for seed in 0..100 {
            let bytes = format!("\r\n {{\"external_id\":\"doc-\u{4e2d}-{seed}\",\"metadata\":{{\"key\":\"\u{1f600}\"}}}}\r\n \n{{\"external_id\":\"last\",\"metadata\":{{}}}}");
            let rows = parse_documents_jsonl(
                Path::new("documents.jsonl"),
                Fragmented(bytes.as_bytes(), seed + 1),
            )
            .unwrap();
            assert_eq!(rows.len(), 2);
            assert_eq!(rows[0].external_id, format!("doc-\u{4e2d}-{seed}"));
            assert_eq!(
                rows[0].metadata.get("key").and_then(JsonValue::as_str),
                Some("\u{1f600}")
            );
            assert_eq!(rows[1].external_id, "last");
        }
        let bytes = b"\n{\"external_id\":\"invalid\xff\",\"metadata\":{}}";
        let error = parse_documents_jsonl(Path::new("documents.jsonl"), Fragmented(bytes, 1))
            .err()
            .unwrap();
        assert_eq!(error.code(), ErrorCode::IoError);
        assert_eq!(detail_value(&error, "line"), Some("2"));
    }

    #[test]
    fn typed_visitors_release_consumed_rows_but_account_for_retained_metadata() {
        use crate::json::JsonProcessingPolicyV1;
        let payload = "x".repeat(256 * 1024);
        for chunks in [false, true] {
            let row = if chunks {
                format!("{{\"document_external_id\":\"doc\",\"chunk_external_id\":\"chunk\",\"ordinal\":0,\"text\":\"text\",\"metadata\":{{\"value\":\"{payload}\"}}}}\n")
            } else {
                format!("{{\"external_id\":\"doc\",\"metadata\":{{\"value\":\"{payload}\"}}}}\n")
            };
            let bytes = row.repeat(100);
            for retain in [false, true] {
                let context = JsonOperationContext::standalone();
                let mut busy = context.reservation();
                busy.grow(
                    JsonProcessingPolicyV1::WORK_BUDGET_BYTES - 2 * 1024 * 1024,
                    JsonInputKind::JsonlRecord,
                )
                .unwrap();
                let mut live = Vec::new();
                let mut count = 0;
                let mut consume = |line, metadata: JsonValue, work: JsonReservation| {
                    count += 1;
                    assert_eq!(line, count);
                    assert_eq!(
                        metadata.get("value").and_then(JsonValue::as_str),
                        Some(payload.as_str())
                    );
                    assert!(work.reserved_bytes() >= payload.len());
                    if retain {
                        live.push((metadata, work));
                    }
                    Ok(())
                };
                let result = if chunks {
                    visit_chunks_jsonl_in(
                        Path::new("rows.jsonl"),
                        bytes.as_bytes(),
                        &context,
                        u64::MAX,
                        |line, row, work| consume(line, row.chunk.metadata, work),
                    )
                } else {
                    visit_documents_jsonl_in(
                        Path::new("rows.jsonl"),
                        bytes.as_bytes(),
                        &context,
                        u64::MAX,
                        |line, row, work| consume(line, row.metadata, work),
                    )
                };
                if retain {
                    let error = result.unwrap_err();
                    assert!(count > 0 && count < 100);
                    assert_eq!(
                        detail_value(&error, "json_limit_reason"),
                        Some("work_budget_exceeded")
                    );
                    assert_eq!(detail_value(&error, "path"), Some("rows.jsonl"));
                    assert_eq!(
                        detail_value(&error, "line"),
                        Some((count + 1).to_string().as_str())
                    );
                } else {
                    result.unwrap();
                    assert_eq!(count, 100);
                }
                drop(live);
                busy.grow(2 * 1024 * 1024, JsonInputKind::JsonlRecord)
                    .unwrap();
            }
        }
    }

    #[test]
    fn native_jsonl_parse_failure_keeps_row_location_and_stops_before_later_rows() {
        let payload = "x".repeat(131_071);
        let valid = format!("{{\"external_id\":\"first\",\"metadata\":{{}}}}\n{{\"external_id\":\"second\",\"title\":\"{payload}\",\"metadata\":{{}}}}\n");
        let rows = format!("{valid}not json\n");
        let context = JsonOperationContext::standalone();
        let capacity = context.available_workspace_bytes();
        let mut visits = 0;
        let error = crate::test_alloc::fail_byte_allocation_once(payload.len(), || {
            visit_documents_jsonl_in(
                Path::new("documents.jsonl"),
                rows.as_bytes(),
                &context,
                u64::MAX,
                |line, row, _work| {
                    assert_eq!(line, 1);
                    assert_eq!(row.external_id, "first");
                    visits += 1;
                    Ok(())
                },
            )
        })
        .unwrap_err();
        assert_eq!(error.code(), ErrorCode::IoError);
        assert_eq!(detail_value(&error, "path"), Some("documents.jsonl"));
        assert_eq!(detail_value(&error, "line"), Some("2"));
        assert_eq!(detail_value(&error, "json_limit_reason"), None);
        assert_eq!(detail_value(&error, "limit"), None);
        assert_eq!(visits, 1);
        assert_eq!(context.available_workspace_bytes(), capacity);
        let mut visits = 0;
        visit_documents_jsonl_in(
            Path::new("documents.jsonl"),
            valid.as_bytes(),
            &context,
            u64::MAX,
            |line, row, _work| {
                visits += 1;
                if line == 2 {
                    assert_eq!(row.title.as_deref(), Some(payload.as_str()));
                }
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(visits, 2);
        assert_eq!(context.available_workspace_bytes(), capacity);
    }

    #[test]
    fn typed_visitor_stops_at_consumer_error_and_preserves_safe_location() {
        let context = JsonOperationContext::standalone();
        let bytes = b"\n{\"external_id\":\"a\",\"metadata\":{}}\r\n{\"external_id\":\"b\",\"metadata\":{}}\nnot json";
        for supplied_location in [false, true] {
            let mut count = 0;
            let error = visit_documents_jsonl_in(
                Path::new("documents.jsonl"),
                &bytes[..],
                &context,
                u64::MAX,
                |line, row, _work| {
                    count += 1;
                    if count == 1 {
                        assert_eq!(line, 2);
                        assert_eq!(row.external_id, "a");
                        return Ok(());
                    }
                    let error = Error::new(ErrorCode::DuplicateExternalId, "consumer rejected row");
                    Err(if supplied_location {
                        error
                            .with_detail("path", "original.jsonl")
                            .with_detail("line", 7)
                    } else {
                        error
                    })
                },
            )
            .unwrap_err();
            assert_eq!(count, 2);
            assert_eq!(error.code(), ErrorCode::DuplicateExternalId);
            assert_eq!(
                detail_value(&error, "line"),
                Some(if supplied_location { "7" } else { "3" })
            );
            assert_eq!(
                detail_value(&error, "path"),
                Some(if supplied_location {
                    "original.jsonl"
                } else {
                    "documents.jsonl"
                })
            );
            context
                .reservation()
                .grow(
                    crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
                    JsonInputKind::JsonlRecord,
                )
                .unwrap();
        }
    }

    #[test]
    fn embedding_visitor_enforces_byte_caps_and_conversion_before_delivery() {
        let context = JsonOperationContext::standalone();
        let bytes =
            b"{\"chunk_external_id\":\"c\",\"model_name\":\"m\",\"dim\":2,\"vector\":[-0.0,1]}\n";
        for limit in [bytes.len() as u64 - 1, bytes.len() as u64] {
            let mut count = 0;
            let result = visit_embeddings_jsonl_in(
                Path::new("vectors.jsonl"),
                &bytes[..],
                &context,
                limit,
                |line, row, work| {
                    count += 1;
                    assert_eq!(line, 1);
                    assert_eq!(row.chunk_external_id, "c");
                    assert_eq!(row.model_name, "m");
                    assert_eq!(row.dim, 2);
                    assert_eq!(row.vector, vec![0.0, 1.0]);
                    assert_eq!(work.reserved_bytes(), 0);
                    Ok(())
                },
            );
            if limit == bytes.len() as u64 {
                result.unwrap();
                assert_eq!(count, 1);
            } else {
                let error = result.unwrap_err();
                assert_eq!(count, 0);
                assert_eq!(
                    detail_value(&error, "reason"),
                    Some("import_budget_exceeded")
                );
                assert_eq!(detail_value(&error, "line"), Some("1"));
            }
        }
        for (vector, code) in [
            ("[1e39]", ErrorCode::InvalidJsonlSchema),
            ("[\"1\"]", ErrorCode::InvalidVectorLength),
            ("null", ErrorCode::MissingRequiredField),
        ] {
            let bytes = format!("\n{{\"chunk_external_id\":\"c\",\"model_name\":\"m\",\"dim\":1,\"vector\":{vector}}}");
            let error = visit_embeddings_jsonl_in(
                Path::new("vectors.jsonl"),
                bytes.as_bytes(),
                &context,
                u64::MAX,
                |_, _, _| panic!("invalid vector delivered"),
            )
            .unwrap_err();
            assert_eq!(error.code(), code, "vector: {vector}");
            assert_eq!(detail_value(&error, "path"), Some("vectors.jsonl"));
            assert_eq!(detail_value(&error, "line"), Some("2"));
        }
        context
            .reservation()
            .grow(
                crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
                JsonInputKind::JsonlRecord,
            )
            .unwrap();
    }

    #[test]
    fn parsed_rows_move_json_allocations_and_keep_exact_live_reservations() {
        let context = JsonOperationContext::standalone();
        let mut bytes = String::new();
        for seed in 0..100 {
            bytes.push_str(&format!("{{\"external_id\":\"doc-{seed}\",\"metadata\":{{\"payload\":\"value-{seed}\"}}}}\n"));
        }
        let mut pointers = Vec::new();
        let mut rows = ParsedRows::new(&context);
        visit_jsonl_rows_in(
            Path::new("documents.jsonl"),
            JsonLines::new(
                bytes.as_bytes(),
                JsonSchema::DocumentRow,
                &context,
                MAX_JSONL_BYTES,
            ),
            |object| {
                pointers.push((
                    object
                        .get("external_id")
                        .unwrap()
                        .as_str()
                        .unwrap()
                        .as_ptr(),
                    object
                        .get("metadata")
                        .unwrap()
                        .get("payload")
                        .unwrap()
                        .as_str()
                        .unwrap()
                        .as_ptr(),
                ));
                Ok((
                    required_nonempty_jsonl_string(object, "external_id")?,
                    required_metadata(object)?,
                ))
            },
            |row| Some((&row.1, JsonInputKind::DocumentMetadata)),
            |_, row, work| {
                rows.push(row, work);
                Ok(())
            },
        )
        .unwrap();
        let mut expected = 0;
        for (index, ((id, metadata), (id_pointer, metadata_pointer))) in
            rows.values.iter().zip(pointers).enumerate()
        {
            assert_eq!(id, &format!("doc-{index}"));
            assert_eq!(id.as_ptr(), id_pointer);
            assert_eq!(
                metadata.get("payload").unwrap().as_str().unwrap().as_ptr(),
                metadata_pointer
            );
            expected += metadata
                .reserve_retained_in(JsonInputKind::DocumentMetadata, &context)
                .unwrap()
                .reserved_bytes();
        }
        assert_eq!(rows.work.reserved_bytes(), expected);
        drop(rows);
        context
            .reservation()
            .grow(
                crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
                JsonInputKind::JsonlRecord,
            )
            .unwrap();
    }

    #[test]
    fn manifest_import_reuses_metadata_work_without_bypassing_shared_admission() {
        use crate::json::JsonProcessingPolicyV1;
        let dir = temp_dir("import-retained-json");
        let path = dir.join("data.coffer");
        let db = Database::create(&path, DatabaseConfig::default()).unwrap();
        let payload = "x".repeat(512 * 1024);
        let documents =
            format!("{{\"external_id\":\"doc\",\"metadata\":{{\"payload\":\"{payload}\"}}}}\n");
        let chunks = (0..3).map(|index| format!("{{\"document_external_id\":\"doc\",\"chunk_external_id\":\"doc#{index}\",\"ordinal\":{index},\"text\":\"payload\",\"metadata\":{{\"payload\":\"{payload}\"}}}}\n")).collect::<String>();
        fs::write(dir.join("documents.jsonl"), &documents).unwrap();
        fs::write(dir.join("chunks.jsonl"), &chunks).unwrap();
        let manifest_path = dir.join("manifest.json");
        fs::write(&manifest_path, r#"{"schema_version":"0.1","collection":"notes","documents":"documents.jsonl","chunks":"chunks.jsonl","embedding_dim":2,"distance_metric":"cosine"}"#).unwrap();
        let before = fs::read(&path).unwrap();
        let other_operation = db.json_context();
        let mut busy = other_operation.reservation();
        busy.grow(
            JsonProcessingPolicyV1::WORK_BUDGET_BYTES - 256 * 1024,
            JsonInputKind::JsonlRecord,
        )
        .unwrap();
        let error = db
            .import_manifest(&manifest_path, OnConflict::Error)
            .unwrap_err();
        assert_eq!(error.code(), ErrorCode::InvalidJsonlSchema);
        assert_eq!(
            detail_value(&error, "json_limit_reason"),
            Some("work_budget_exceeded")
        );
        // Reference directories, two fixed inputs and two staging files coexist.
        let temporary_directory = std::env::temp_dir();
        let remaining = 128 * 1024
            - 2 * temporary_directory.as_os_str().as_encoded_bytes().len()
            - 4 * crate::temporary::test_file_management_bytes(&temporary_directory);
        assert_eq!(
            detail_value(&error, "limit"),
            Some(remaining.to_string().as_str())
        );
        assert!(error.detail().iter().any(|(key, _)| key == "path"));
        assert_eq!(fs::read(&path).unwrap(), before);
        assert!(db.list_collections().unwrap().is_empty());
        drop(busy);
        let mut busy = other_operation.reservation();
        busy.grow(
            JsonProcessingPolicyV1::WORK_BUDGET_BYTES - 3 * 1024 * 1024,
            JsonInputKind::JsonlRecord,
        )
        .unwrap();
        let summary = db
            .import_manifest(&manifest_path, OnConflict::Error)
            .unwrap();
        assert_eq!(
            (summary.documents_inserted, summary.chunks_inserted),
            (1, 3)
        );
        busy.grow(
            3 * 1024 * 1024 - db.index_directory_bytes(),
            JsonInputKind::JsonlRecord,
        )
        .unwrap();
        drop(busy);
        assert_eq!(
            db.collection("notes")
                .unwrap()
                .search()
                .text("payload")
                .execute()
                .unwrap()
                .results
                .len(),
            3
        );
        drop(db);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn preprocessed_manifest_rejects_invalid_batch_before_mutation() {
        let dir = temp_dir("import-preprocessed-validation");
        let path = dir.join("data.coffer");
        let db = Database::create(&path, DatabaseConfig::default()).unwrap();
        let manifest = dir.join("manifest.json");
        fs::write(&manifest, r#"{"schema_version":"0.1","collection":"notes","documents":"documents.jsonl","chunks":"chunks.jsonl","embedding_dim":2,"distance_metric":"cosine"}"#).unwrap();
        let before = fs::read(&path).unwrap();
        for (documents, chunks, expected) in [
            (
                vec!["doc"],
                vec![("doc", "a", 0), ("doc", "b", 0)],
                ErrorCode::InvalidJsonlSchema,
            ),
            (
                vec!["doc"],
                vec![("doc", "a", u64::from(u32::MAX) + 1)],
                ErrorCode::InvalidJsonlSchema,
            ),
            (
                vec!["doc"],
                vec![("doc", "a", 0), ("doc", "a", 1)],
                ErrorCode::DuplicateExternalId,
            ),
            (
                vec!["doc", "other"],
                vec![("doc", "a", 0), ("other", "a", 0)],
                ErrorCode::DuplicateExternalId,
            ),
            (
                vec!["doc", "doc"],
                vec![("doc", "a", 0)],
                ErrorCode::DuplicateExternalId,
            ),
        ] {
            let documents = documents
                .into_iter()
                .map(|id| format!("{{\"external_id\":\"{id}\",\"metadata\":{{}}}}\n"))
                .collect::<String>();
            let chunks = chunks.into_iter().map(|(doc, id, ordinal)| format!("{{\"document_external_id\":\"{doc}\",\"chunk_external_id\":\"{id}\",\"ordinal\":{ordinal},\"text\":\"payload\",\"metadata\":{{}}}}\n")).collect::<String>();
            fs::write(dir.join("documents.jsonl"), documents).unwrap();
            fs::write(dir.join("chunks.jsonl"), chunks).unwrap();
            for conflict in [OnConflict::Error, OnConflict::Skip, OnConflict::Replace] {
                let error = db.import_manifest(&manifest, conflict).unwrap_err();
                assert_eq!(error.code(), expected);
                assert_eq!(fs::read(&path).unwrap(), before);
                assert!(db.list_collections().unwrap().is_empty());
                db.json_context()
                    .reservation()
                    .grow(
                        crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
                        JsonInputKind::JsonlRecord,
                    )
                    .unwrap();
            }
        }
        drop(db);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn prepared_import_context_preserves_usage_across_adoption_failure_and_success() {
        use crate::json::JsonProcessingPolicyV1;
        let dir = temp_dir("import-context-adoption");
        let manifest = dir.join("manifest.json");
        fs::write(&manifest, r#"{"schema_version":"0.1","collection":"notes","documents":"documents.jsonl","chunks":"chunks.jsonl","embedding_dim":2,"distance_metric":"cosine"}"#).unwrap();
        fs::write(
            dir.join("documents.jsonl"),
            "{\"external_id\":\"doc\",\"metadata\":{}}\n",
        )
        .unwrap();
        fs::write(dir.join("chunks.jsonl"), format!("{{\"document_external_id\":\"doc\",\"chunk_external_id\":\"chunk\",\"ordinal\":0,\"text\":\"payload\",\"metadata\":{{\"payload\":\"{}\"}}}}\n", "x".repeat(256 * 1024))).unwrap();
        let path = dir.join("data.coffer");
        let db = Database::create(&path, DatabaseConfig::default()).unwrap();
        let before = fs::read(&path).unwrap();
        let context = JsonOperationContext::standalone();
        let mut preprocessing = context.reservation();
        preprocessing
            .grow(
                JsonProcessingPolicyV1::WORK_BUDGET_BYTES - 128 * 1024,
                JsonInputKind::BenchmarkArtifact,
            )
            .unwrap();
        let mut target_busy = db.json_context().reservation();
        target_busy
            .grow(256 * 1024, JsonInputKind::JsonlRecord)
            .unwrap();
        let error = db
            .import_manifest_in(
                &manifest,
                OnConflict::Error,
                ImportIndexBuildMode::InlineFullCheckpoint,
                context.clone(),
            )
            .unwrap_err();
        assert_eq!(detail_value(&error, "json_input_kind"), Some("manifest"));
        assert_eq!(detail_value(&error, "limit"), Some("33292288"));
        assert_eq!(
            detail_value(&error, "path"),
            Some(manifest.to_str().unwrap())
        );
        context
            .reservation()
            .grow(128 * 1024, JsonInputKind::JsonlRecord)
            .unwrap();
        assert_eq!(fs::read(&path).unwrap(), before);
        drop(target_busy);
        let error = db
            .import_manifest_in(
                &manifest,
                OnConflict::Error,
                ImportIndexBuildMode::InlineFullCheckpoint,
                context.clone(),
            )
            .unwrap_err();
        assert_eq!(error.code(), ErrorCode::ResourceExhausted);
        assert_eq!(detail_value(&error, "resource"), Some("memory"));
        assert_eq!(detail_value(&error, "operation"), Some("catalog_build"));
        assert_eq!(fs::read(&path).unwrap(), before);
        assert_eq!(db.temporary_disk_budget().used(), 0);
        preprocessing.release(128 * 1024);
        let error = db
            .import_manifest_in(
                &manifest,
                OnConflict::Error,
                ImportIndexBuildMode::InlineFullCheckpoint,
                context.clone(),
            )
            .unwrap_err();
        assert_eq!(
            detail_value(&error, "json_limit_reason"),
            Some("work_budget_exceeded")
        );
        let effective = JsonProcessingPolicyV1::WORK_BUDGET_BYTES
            - 128 * 1024
            - 2 * std::env::temp_dir().as_os_str().as_encoded_bytes().len()
            - 4 * crate::temporary::test_file_management_bytes(&std::env::temp_dir());
        assert_eq!(
            detail_value(&error, "limit"),
            Some(effective.to_string().as_str())
        );
        assert_eq!(
            preprocessing.reserved_bytes(),
            JsonProcessingPolicyV1::WORK_BUDGET_BYTES - 256 * 1024
        );
        assert_eq!(fs::read(&path).unwrap(), before);
        preprocessing.release(1024 * 1024);
        let summary = db
            .import_manifest_in(
                &manifest,
                OnConflict::Error,
                ImportIndexBuildMode::InlineFullCheckpoint,
                context.clone(),
            )
            .unwrap();
        assert_eq!(
            (summary.documents_inserted, summary.chunks_inserted),
            (1, 1)
        );
        let capacity = JsonProcessingPolicyV1::WORK_BUDGET_BYTES - db.index_directory_bytes();
        let remaining = capacity - preprocessing.reserved_bytes();
        let mut other = db.json_context().reservation();
        other.grow(remaining, JsonInputKind::JsonlRecord).unwrap();
        assert!(other.grow(1, JsonInputKind::JsonlRecord).is_err());
        drop(other);
        drop(preprocessing);
        db.json_context()
            .reservation()
            .grow(capacity, JsonInputKind::JsonlRecord)
            .unwrap();
        drop(db);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn jsonl_line_buffers_are_reserved_before_growth_with_original_location() {
        use crate::json::JsonProcessingPolicyV1;
        for (schema, path, field) in [
            (JsonSchema::DocumentRow, "documents.jsonl", "title"),
            (JsonSchema::ChunkRow, "chunks.jsonl", "text"),
            (JsonSchema::EmbeddingRow, "embeddings.jsonl", "model_name"),
        ] {
            let context = JsonOperationContext::standalone();
            let mut busy = context.reservation();
            busy.grow(
                JsonProcessingPolicyV1::WORK_BUDGET_BYTES - 16384,
                JsonInputKind::JsonlRecord,
            )
            .unwrap();
            let bytes = format!("\n{{\"{field}\":\"{}\"}}", "x".repeat(32768));
            let error = visit_jsonl_rows_in(
                Path::new(path),
                JsonLines::new(bytes.as_bytes(), schema, &context, MAX_JSONL_BYTES),
                |_| Ok(()),
                |_| None,
                |_, _, _| panic!("line buffer must fail before delivery"),
            )
            .err()
            .unwrap();
            assert_eq!(error.code(), ErrorCode::InvalidJsonlSchema);
            assert_eq!(
                detail_value(&error, "json_input_kind"),
                Some("jsonl_record")
            );
            assert_eq!(
                detail_value(&error, "json_limit_reason"),
                Some("work_budget_exceeded")
            );
            assert_eq!(detail_value(&error, "limit"), Some("33554432"));
            assert_eq!(detail_value(&error, "path"), Some(path));
            assert_eq!(detail_value(&error, "line"), Some("2"));
            busy.grow(16384, JsonInputKind::JsonlRecord).unwrap();
        }
    }

    #[test]
    fn skipped_manifest_documents_do_not_bypass_chunk_validation() {
        let dir = temp_dir("skipped-chunk-stream-validation");
        let manifest = dir.join("manifest.json");
        fs::write(&manifest, r#"{"schema_version":"0.1","collection":"notes","documents":"documents.jsonl","chunks":"chunks.jsonl","embedding_dim":2,"distance_metric":"cosine"}"#).unwrap();
        fs::write(
            dir.join("documents.jsonl"),
            "{\"external_id\":\"doc\",\"metadata\":{}}\n",
        )
        .unwrap();
        let first = "{\"document_external_id\":\"doc\",\"chunk_external_id\":\"first\",\"ordinal\":0,\"text\":\"payload\",\"metadata\":{}}\n";
        let chunks = dir.join("chunks.jsonl");
        fs::write(&chunks, first).unwrap();
        let path = dir.join("data.coffer");
        let db = Database::create(&path, DatabaseConfig::default()).unwrap();
        db.import_manifest(&manifest, OnConflict::Error).unwrap();
        let before = fs::read(&path).unwrap();
        for (document, id, ordinal, code) in [
            ("doc", "second", 0, ErrorCode::InvalidJsonlSchema),
            ("doc", "first", 1, ErrorCode::DuplicateExternalId),
            ("missing", "second", 1, ErrorCode::InvalidJsonlSchema),
            (
                "doc",
                "second",
                u64::from(u32::MAX) + 1,
                ErrorCode::InvalidJsonlSchema,
            ),
        ] {
            fs::write(&chunks, format!("{first}{{\"document_external_id\":\"{document}\",\"chunk_external_id\":\"{id}\",\"ordinal\":{ordinal},\"text\":\"payload\",\"metadata\":{{}}}}\n")).unwrap();
            for conflict in [OnConflict::Skip, OnConflict::Replace] {
                let error = db.import_manifest(&manifest, conflict).unwrap_err();
                assert_eq!(error.code(), code);
                assert_eq!(detail_value(&error, "line"), Some("2"));
                assert_eq!(detail_value(&error, "path"), Some(chunks.to_str().unwrap()));
                assert_eq!(fs::read(&path).unwrap(), before);
            }
        }
        drop(db);
        fs::remove_dir_all(dir).unwrap();
    }

    #[cfg(unix)]
    const FILE_ADMISSION_INPUTS: [(&str, &[u8]); 3] = [
        ("documents.jsonl", b"documents"),
        ("chunks.jsonl", b"chunks"),
        ("embeddings.jsonl", b"embeddings"),
    ];

    #[cfg(unix)]
    fn file_admission_manifest(label: &str) -> (PathBuf, ImportManifest) {
        let dir = temp_dir(label);
        let value = parse(r#"{"schema_version":"0.1","collection":"notes","documents":"documents.jsonl","chunks":"chunks.jsonl","embeddings":"embeddings.jsonl"}"#).unwrap();
        for (name, bytes) in FILE_ADMISSION_INPUTS {
            fs::write(dir.join(name), bytes).unwrap();
        }
        let manifest = ImportManifest::from_value(&dir.join("manifest.json"), &value).unwrap();
        (dir, manifest)
    }

    #[test]
    #[cfg(unix)]
    fn fixed_input_file_creation_reclaims_only_the_third_factory_and_preserves_copied_prefix() {
        let (dir, manifest) = file_admission_manifest("fixed-input-creation-reclaim");
        let management = crate::temporary::test_file_management_bytes(&std::env::temp_dir());
        let limit = 64 * 1024;
        let occupied = limit - 3 * management - 128 + 1;
        for release in [false, true] {
            let ledger = std::sync::Arc::new(crate::json::WorkLedger::with_limits(
                2 * limit,
                limit,
                limit,
            ));
            let context = JsonOperationContext::new(ledger.clone());
            let sibling = JsonOperationContext::new(ledger);
            let mut cache = Some(context.cache_reservation(occupied as u128, limit).unwrap());
            let disk = std::sync::Arc::new(TemporaryDiskBudget::new(4096));
            let mut factories = 0;
            let mut reclaims = 0;
            let result = manifest.validated_inputs_reclaiming(disk.clone(), || {
                factories += 1;
                create_budgeted_temporary_file_reclaiming(
                    &std::env::temp_dir(),
                    "import-input",
                    "tmp",
                    disk.clone(),
                    &context,
                    "operation_scratch",
                    || {
                        reclaims += 1;
                        assert_eq!(factories, 3);
                        assert_eq!(disk.used(), 15);
                        let mut work = sibling.workspace_reservation();
                        work.grow_to(management + 127, management + 127, "operation_scratch")
                            .unwrap();
                        assert!(work
                            .grow_to(management + 128, management + 128, "operation_scratch")
                            .is_err());
                        drop(work);
                        if release {
                            drop(cache.take());
                        }
                        release
                    },
                )
            });
            assert_eq!((factories, reclaims), (3, 1));
            drop(context);
            if release {
                let mut inputs = result.unwrap();
                assert_eq!(disk.used(), 25);
                for (input, (_, expected)) in [
                    &mut inputs.documents,
                    &mut inputs.chunks,
                    inputs.embeddings.as_mut().unwrap(),
                ]
                .into_iter()
                .zip(FILE_ADMISSION_INPUTS)
                {
                    let mut bytes = Vec::new();
                    input.reader().unwrap().read_to_end(&mut bytes).unwrap();
                    assert_eq!(bytes, expected);
                }
                let mut work = sibling.workspace_reservation();
                work.grow_to(
                    limit - 3 * management,
                    limit - 3 * management,
                    "operation_scratch",
                )
                .unwrap();
                assert!(work
                    .grow_to(
                        limit - 3 * management + 1,
                        limit - 3 * management + 1,
                        "operation_scratch"
                    )
                    .is_err());
                drop(work);
                drop(inputs);
            } else {
                let error = result.unwrap_err();
                assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                assert_eq!(
                    detail_value(&error, "requested_bytes"),
                    Some((limit + 1).to_string().as_str())
                );
            }
            drop(cache);
            let mut work = sibling.workspace_reservation();
            work.grow_to(limit, limit, "operation_scratch").unwrap();
            assert!(work
                .grow_to(limit + 1, limit + 1, "operation_scratch")
                .is_err());
            assert_eq!(disk.used(), 0);
            for (name, expected) in FILE_ADMISSION_INPUTS {
                assert_eq!(fs::read(dir.join(name)).unwrap(), expected);
            }
        }
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    #[cfg(unix)]
    fn fixed_input_third_file_admission_returns_prior_owners_and_preserves_sources() {
        let (dir, manifest) = file_admission_manifest("input-management-admission");
        let sources = FILE_ADMISSION_INPUTS;
        let management = 3 * crate::temporary::test_file_management_bytes(&std::env::temp_dir());
        let peak = management + 128;
        for available in [peak - 1, peak, peak + 1] {
            let limit = 2 * peak;
            let ledger = std::sync::Arc::new(crate::json::WorkLedger::with_limits(
                2 * limit,
                limit,
                limit,
            ));
            let context = JsonOperationContext::new(ledger.clone());
            let sibling = JsonOperationContext::new(ledger);
            let mut busy = sibling.workspace_reservation();
            busy.grow_to(limit - available, limit - available, "operation_scratch")
                .unwrap();
            let disk = std::sync::Arc::new(TemporaryDiskBudget::new(4096));
            let result = manifest.validated_inputs_in(disk.clone(), &context);
            drop(context);
            if available < peak {
                let error = result.unwrap_err();
                assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                for (key, expected) in [
                    ("resource", "memory".to_owned()),
                    ("operation", "operation_scratch".to_owned()),
                    ("limit_bytes", limit.to_string()),
                    ("requested_bytes", (limit + 1).to_string()),
                ] {
                    assert!(
                        error.detail().contains(&(key.to_owned(), expected)),
                        "{error:?}"
                    );
                }
            } else {
                let mut inputs = result.unwrap();
                assert_eq!(disk.used(), 25);
                let mut work = sibling.workspace_reservation();
                work.grow_to(
                    available - management,
                    available - management,
                    "operation_scratch",
                )
                .unwrap();
                assert!(work
                    .grow_to(
                        available - management + 1,
                        available - management + 1,
                        "operation_scratch"
                    )
                    .is_err());
                drop(work);
                for (input, (_, expected)) in [
                    &mut inputs.documents,
                    &mut inputs.chunks,
                    inputs.embeddings.as_mut().unwrap(),
                ]
                .into_iter()
                .zip(sources)
                {
                    let mut bytes = Vec::new();
                    input.reader().unwrap().read_to_end(&mut bytes).unwrap();
                    assert_eq!(bytes, expected);
                }
                drop(inputs);
            }
            let mut work = sibling.workspace_reservation();
            work.grow_to(available, available, "operation_scratch")
                .unwrap();
            assert!(work
                .grow_to(available + 1, available + 1, "operation_scratch")
                .is_err());
            drop(work);
            drop(busy);
            sibling
                .workspace_reservation()
                .grow_to(limit, limit, "operation_scratch")
                .unwrap();
            assert_eq!(disk.used(), 0);
            for (name, expected) in sources {
                assert_eq!(fs::read(dir.join(name)).unwrap(), expected);
            }
            assert_eq!(fs::read_dir(&dir).unwrap().count(), 3);
        }
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn input_preflight_accounts_for_existing_runtime_usage_without_leaking_on_failure() {
        let dir = temp_dir("input-shared-preflight");
        let value = parse(r#"{"schema_version":"0.1","collection":"notes","documents":"documents.jsonl","chunks":"chunks.jsonl"}"#).unwrap();
        fs::write(dir.join("documents.jsonl"), b"12345").unwrap();
        fs::write(dir.join("chunks.jsonl"), b"123456").unwrap();
        let manifest = ImportManifest::from_value(&dir.join("manifest.json"), &value).unwrap();
        for held_bytes in 0..100u64 {
            for available in [0, 4, 5, 10, 11, 12] {
                let budget = std::sync::Arc::new(TemporaryDiskBudget::new(held_bytes + available));
                let (mut file, mut held) =
                    create_budgeted_temporary_file_in(&dir, "busy", "tmp", budget.clone()).unwrap();
                held.reserve_disk(held_bytes).unwrap();
                file.write_all(&vec![0; held_bytes as usize]).unwrap();
                drop(file);
                let result = manifest
                    .validated_inputs_in(budget.clone(), &JsonOperationContext::standalone());
                if available < 11 {
                    let error = result.unwrap_err();
                    assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                    assert_eq!(
                        detail_value(&error, "limit_bytes"),
                        Some((held_bytes + available).to_string().as_str())
                    );
                    assert_eq!(
                        detail_value(&error, "requested_bytes"),
                        Some(
                            (held_bytes + if available < 5 { 5 } else { 11 })
                                .to_string()
                                .as_str()
                        )
                    );
                } else {
                    let inputs = result.unwrap();
                    assert_eq!(budget.used(), held_bytes + 11);
                    drop(inputs);
                }
                assert_eq!(budget.used(), held_bytes);
                drop(held);
                assert_eq!(budget.used(), 0);
            }
        }
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn snapshot_enforces_combined_input_budget() {
        let dir = temp_dir("combined-budget");
        let value = parse(r#"{"schema_version":"0.1","collection":"notes","documents":"documents.jsonl","chunks":"chunks.jsonl"}"#).unwrap();
        fs::write(dir.join("documents.jsonl"), "12345").unwrap();
        fs::write(dir.join("chunks.jsonl"), "123456").unwrap();
        let manifest = ImportManifest::from_value(&dir.join("manifest.json"), &value).unwrap();
        let err = manifest.validated_inputs(10).unwrap_err();
        assert_eq!(err.code(), ErrorCode::ResourceExhausted);
        assert_eq!(detail_value(&err, "resource"), Some("temporary_disk"));
        assert_eq!(detail_value(&err, "limit_bytes"), Some("10"));
        assert_eq!(detail_value(&err, "requested_bytes"), Some("11"));
        assert!(manifest.validated_inputs(11).is_ok());
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn import_manifest_rejects_hash_mismatch() {
        let dir = temp_dir("hash-mismatch");
        let documents = write_file(
            &dir,
            "documents.jsonl",
            r#"{"external_id":"doc","metadata":{}}"#,
        );
        write_file(
            &dir,
            "chunks.jsonl",
            r#"{"document_external_id":"doc","chunk_external_id":"doc#1","ordinal":1,"text":"wal","metadata":{}}"#,
        );
        fs::write(
            dir.join("manifest.json"),
            format!(
                concat!(
                    r#"{{"schema_version":"0.1","collection":"notes","#,
                    r#""documents":"documents.jsonl","chunks":"chunks.jsonl","#,
                    r#""embedding_model":null,"embedding_dim":2,"distance_metric":"cosine","#,
                    r#""hashes":{{"documents_sha256":"{}","chunks_sha256":"0000"}}"#,
                    r#"}}"#
                ),
                documents
            ),
        )
        .unwrap();

        let manifest = ImportManifest::load(dir.join("manifest.json")).unwrap();
        let err = manifest.validated_inputs(MAX_JSONL_BYTES).unwrap_err();
        assert_eq!(err.code(), ErrorCode::ChecksumMismatch);
        assert_eq!(detail_value(&err, "artifact_kind"), Some("chunks"));
        assert_eq!(detail_value(&err, "expected_sha256"), Some("0000"));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn import_manifest_rejects_blank_required_strings() {
        let dir = temp_dir("blank-required");
        fs::write(
            dir.join("manifest.json"),
            r#"{"schema_version":"0.1","collection":" ","documents":"documents.jsonl","chunks":"chunks.jsonl"}"#,
        )
        .unwrap();

        let err = ImportManifest::load(dir.join("manifest.json")).unwrap_err();
        assert_eq!(err.code(), ErrorCode::MissingRequiredField);
        assert_eq!(detail_value(&err, "field"), Some("collection"));

        fs::write(
            dir.join("manifest.json"),
            r#"{"schema_version":"0.1","collection":"notes","documents":"","chunks":"chunks.jsonl"}"#,
        )
        .unwrap();

        let err = ImportManifest::load(dir.join("manifest.json")).unwrap_err();
        assert_eq!(err.code(), ErrorCode::MissingRequiredField);
        assert_eq!(detail_value(&err, "field"), Some("documents"));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn import_manifest_rejects_blank_optional_strings_when_provided() {
        let dir = temp_dir("blank-optional");
        fs::write(
            dir.join("manifest.json"),
            r#"{"schema_version":"0.1","collection":"notes","documents":"documents.jsonl","chunks":"chunks.jsonl","embeddings":" "}"#,
        )
        .unwrap();

        let err = ImportManifest::load(dir.join("manifest.json")).unwrap_err();
        assert_eq!(err.code(), ErrorCode::MissingRequiredField);
        assert_eq!(detail_value(&err, "field"), Some("embeddings"));

        fs::write(
            dir.join("manifest.json"),
            r#"{"schema_version":"0.1","collection":"notes","documents":"documents.jsonl","chunks":"chunks.jsonl","embedding_model":""}"#,
        )
        .unwrap();

        let err = ImportManifest::load(dir.join("manifest.json")).unwrap_err();
        assert_eq!(err.code(), ErrorCode::MissingRequiredField);
        assert_eq!(detail_value(&err, "field"), Some("embedding_model"));

        fs::write(
            dir.join("manifest.json"),
            r#"{"schema_version":"0.1","dataset_id":" ","collection":"notes","documents":"documents.jsonl","chunks":"chunks.jsonl"}"#,
        )
        .unwrap();

        let err = ImportManifest::load(dir.join("manifest.json")).unwrap_err();
        assert_eq!(err.code(), ErrorCode::MissingRequiredField);
        assert_eq!(detail_value(&err, "field"), Some("dataset_id"));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn import_manifest_rejects_non_object_hashes() {
        let dir = temp_dir("hashes-schema");
        fs::write(
            dir.join("manifest.json"),
            r#"{"schema_version":"0.1","collection":"notes","documents":"documents.jsonl","chunks":"chunks.jsonl","hashes":"not-an-object"}"#,
        )
        .unwrap();

        let err = ImportManifest::load(dir.join("manifest.json")).unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
        assert_eq!(detail_value(&err, "field"), Some("hashes"));

        fs::write(
            dir.join("manifest.json"),
            r#"{"schema_version":"0.1","collection":"notes","documents":"documents.jsonl","chunks":"chunks.jsonl","hashes":{"documents_sha256":""}}"#,
        )
        .unwrap();

        let err = ImportManifest::load(dir.join("manifest.json")).unwrap_err();
        assert_eq!(err.code(), ErrorCode::MissingRequiredField);
        assert_eq!(detail_value(&err, "field"), Some("hashes.documents_sha256"));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn import_manifest_rejects_legacy_metric_field() {
        let dir = temp_dir("legacy-metric");
        fs::write(
            dir.join("manifest.json"),
            r#"{"schema_version":"0.1","collection":"notes","documents":"documents.jsonl","chunks":"chunks.jsonl","embedding_dim":2,"metric":"cosine"}"#,
        )
        .unwrap();

        let err = ImportManifest::load(dir.join("manifest.json")).unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
        assert_eq!(detail_value(&err, "field"), Some("metric"));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn import_manifest_rejects_unknown_public_fields() {
        let dir = temp_dir("unknown-manifest-field");
        fs::write(
            dir.join("manifest.json"),
            r#"{"schema_version":"0.1","collection":"notes","documents":"documents.jsonl","chunks":"chunks.jsonl","embedding_dim":2,"distance_metric":"cosine","document":"typo.jsonl"}"#,
        )
        .unwrap();

        let err = ImportManifest::load(dir.join("manifest.json")).unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
        assert_eq!(detail_value(&err, "field"), Some("document"));

        fs::write(
            dir.join("manifest.json"),
            r#"{"schema_version":"0.1","collection":"notes","documents":"documents.jsonl","chunks":"chunks.jsonl","embedding_dim":2,"distance_metric":"cosine","hashes":{"documents_hash":"typo"}}"#,
        )
        .unwrap();

        let err = ImportManifest::load(dir.join("manifest.json")).unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
        assert_eq!(detail_value(&err, "field"), Some("hashes.documents_hash"));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn import_manifest_accepts_and_validates_export_identity_fields() {
        let dir = temp_dir("export-identity-fields");
        fs::write(
            dir.join("manifest.json"),
            r#"{"schema_version":"0.1","cofferdb_version":"0.1.0","file_format_version":1,"collection":"notes","exported_at":"2026-05-29T00:00:00Z","documents":"documents.jsonl","chunks":"chunks.jsonl","embedding_dim":2,"distance_metric":"cosine","counts":{"documents":1,"chunks":1,"embeddings":0}}"#,
        )
        .unwrap();

        let manifest = ImportManifest::load(dir.join("manifest.json")).unwrap();
        assert_eq!(manifest.collection, "notes");

        fs::write(
            dir.join("manifest.json"),
            r#"{"schema_version":"0.1","collection":"notes","documents":"documents.jsonl","chunks":"chunks.jsonl","embedding_dim":2,"distance_metric":"cosine","counts":{"documents":"one"}}"#,
        )
        .unwrap();

        let err = ImportManifest::load(dir.join("manifest.json")).unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
        assert_eq!(detail_value(&err, "field"), Some("counts.documents"));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn import_manifest_rejects_export_count_mismatch() {
        let dir = temp_dir("export-count-mismatch");
        let db_path = dir.join("count-mismatch.coffer");
        let db = Database::create(&db_path, DatabaseConfig::default()).unwrap();
        write_file(
            &dir,
            "documents.jsonl",
            r#"{"external_id":"doc","metadata":{}}"#,
        );
        write_file(
            &dir,
            "chunks.jsonl",
            r#"{"document_external_id":"doc","chunk_external_id":"doc#1","ordinal":1,"text":"wal","metadata":{}}"#,
        );
        fs::write(
            dir.join("manifest.json"),
            r#"{"schema_version":"0.1","collection":"notes","documents":"documents.jsonl","chunks":"chunks.jsonl","embedding_dim":2,"distance_metric":"cosine","counts":{"documents":2,"chunks":1,"embeddings":0}}"#,
        )
        .unwrap();

        let err = db
            .import_manifest(dir.join("manifest.json"), OnConflict::Error)
            .unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
        assert_eq!(detail_value(&err, "field"), Some("counts.documents"));
        assert_eq!(detail_value(&err, "expected_count"), Some("2"));
        assert_eq!(detail_value(&err, "actual_count"), Some("1"));

        drop(db);
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn import_manifest_reports_missing_hash_warnings() {
        let dir = temp_dir("hash-warnings");
        let db_path = dir.join("warnings.coffer");
        let db = Database::create(&db_path, DatabaseConfig::default()).unwrap();
        write_file(
            &dir,
            "documents.jsonl",
            r#"{"external_id":"doc","metadata":{}}"#,
        );
        write_file(
            &dir,
            "chunks.jsonl",
            r#"{"document_external_id":"doc","chunk_external_id":"doc#1","ordinal":1,"text":"missing hashes warn","metadata":{}}"#,
        );
        fs::write(
            dir.join("manifest.json"),
            r#"{"schema_version":"0.1","collection":"notes","documents":"documents.jsonl","chunks":"chunks.jsonl","embedding_model":null,"embedding_dim":2,"distance_metric":"cosine"}"#,
        )
        .unwrap();

        let summary = db
            .import_manifest(dir.join("manifest.json"), OnConflict::Error)
            .unwrap();
        assert_eq!(summary.warnings.len(), 2);
        assert_eq!(summary.warnings[0].code, "missing_reproducibility_hash");
        assert_eq!(summary.warnings[0].field, "hashes.documents_sha256");
        assert_eq!(summary.warnings[1].field, "hashes.chunks_sha256");
        let summary_json = summary.to_json();
        let warnings = summary_json
            .get("warnings")
            .and_then(JsonValue::as_array)
            .unwrap();
        assert_eq!(warnings.len(), 2);
        assert_eq!(
            warnings[0].get("artifact_kind").and_then(JsonValue::as_str),
            Some("documents")
        );

        drop(db);
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn import_manifest_warns_when_source_span_conflicts_with_metadata() {
        let dir = temp_dir("source-span-warning");
        let db_path = dir.join("source-span.coffer");
        let db = Database::create(&db_path, DatabaseConfig::default()).unwrap();
        let documents = write_file(
            &dir,
            "documents.jsonl",
            r#"{"external_id":"doc","metadata":{}}"#,
        );
        let chunks = write_file(
            &dir,
            "chunks.jsonl",
            r#"{"document_external_id":"doc","chunk_external_id":"doc#1","ordinal":1,"text":"source span wins","metadata":{"byte_start":0,"byte_end":16,"line_start":1,"line_end":1},"source_span":{"start":5,"end":16,"line_start":2,"line_end":2}}"#,
        );
        fs::write(
            dir.join("manifest.json"),
            format!(
                concat!(
                    r#"{{"schema_version":"0.1","collection":"notes","#,
                    r#""documents":"documents.jsonl","chunks":"chunks.jsonl","#,
                    r#""embedding_model":null,"embedding_dim":2,"distance_metric":"cosine","#,
                    r#""hashes":{{"documents_sha256":"{}","chunks_sha256":"{}"}}"#,
                    r#"}}"#
                ),
                documents, chunks
            ),
        )
        .unwrap();

        let summary = db
            .import_manifest(dir.join("manifest.json"), OnConflict::Error)
            .unwrap();
        assert_eq!(summary.warnings.len(), 1);
        assert_eq!(summary.warnings[0].code, "source_span_metadata_conflict");
        assert_eq!(summary.warnings[0].field, "source_span");
        let chunk = db
            .collection("notes")
            .unwrap()
            .get_chunk_by_external_id("doc#1")
            .unwrap();
        let span = chunk.source_span.unwrap();
        assert_eq!(span.start, Some(5));
        assert_eq!(span.line_start, Some(2));

        drop(db);
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn jsonl_row_schema_errors_include_path_and_line_without_payload() {
        let dir = temp_dir("jsonl-row-location");
        let documents_path = dir.join("documents.jsonl");
        fs::write(&documents_path, r#"{"metadata":{"private":"do not echo"}}"#).unwrap();

        let err = match read_documents_jsonl(&documents_path) {
            Ok(_) => panic!("documents JSONL without external_id must fail"),
            Err(err) => err,
        };
        assert_eq!(err.code(), ErrorCode::MissingRequiredField);
        let path_label = documents_path.display().to_string();
        assert_eq!(detail_value(&err, "path"), Some(path_label.as_str()));
        assert_eq!(detail_value(&err, "line"), Some("1"));
        let public_json = err.to_json().to_compact_string();
        assert!(!public_json.contains("do not echo"));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn import_jsonl_rows_reject_unknown_public_fields() {
        let dir = temp_dir("jsonl-unknown-fields");

        let documents_path = dir.join("documents.jsonl");
        fs::write(
            &documents_path,
            r#"{"external_id":"doc","metadata":{},"embedding":[1.0]}"#,
        )
        .unwrap();
        let err = match read_documents_jsonl(&documents_path) {
            Ok(_) => panic!("documents JSONL with unknown field must fail"),
            Err(err) => err,
        };
        assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
        assert_eq!(detail_value(&err, "field"), Some("embedding"));
        assert_eq!(detail_value(&err, "line"), Some("1"));

        fs::write(
            &documents_path,
            r#"{"external_id":"doc","title":42,"metadata":{}}"#,
        )
        .unwrap();
        let err = match read_documents_jsonl(&documents_path) {
            Ok(_) => panic!("documents JSONL with non-string title must fail"),
            Err(err) => err,
        };
        assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
        assert_eq!(detail_value(&err, "field"), Some("title"));
        assert_eq!(detail_value(&err, "line"), Some("1"));

        fs::write(
            &documents_path,
            r#"{"external_id":"doc","source_uri":"","metadata":{}}"#,
        )
        .unwrap();
        let err = match read_documents_jsonl(&documents_path) {
            Ok(_) => panic!("documents JSONL with blank source_uri must fail"),
            Err(err) => err,
        };
        assert_eq!(err.code(), ErrorCode::MissingRequiredField);
        assert_eq!(detail_value(&err, "field"), Some("source_uri"));
        assert_eq!(detail_value(&err, "line"), Some("1"));

        fs::write(&documents_path, r#"{"external_id":" ","metadata":{}}"#).unwrap();
        let err = match read_documents_jsonl(&documents_path) {
            Ok(_) => panic!("documents JSONL with blank external_id must fail"),
            Err(err) => err,
        };
        assert_eq!(err.code(), ErrorCode::MissingRequiredField);
        assert_eq!(detail_value(&err, "field"), Some("external_id"));
        assert_eq!(detail_value(&err, "line"), Some("1"));

        let chunks_path = dir.join("chunks.jsonl");
        fs::write(
            &chunks_path,
            r#"{"document_external_id":"doc","chunk_external_id":"doc#1","ordinal":1,"text":"hello","metadata":{},"score":1.0}"#,
        )
        .unwrap();
        let err = match read_chunks_jsonl(&chunks_path) {
            Ok(_) => panic!("chunks JSONL with unknown field must fail"),
            Err(err) => err,
        };
        assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
        assert_eq!(detail_value(&err, "field"), Some("score"));
        assert_eq!(detail_value(&err, "line"), Some("1"));

        fs::write(
            &chunks_path,
            r#"{"document_external_id":"doc","chunk_external_id":"doc#1","ordinal":1,"text":"","metadata":{}}"#,
        )
        .unwrap();
        let err = match read_chunks_jsonl(&chunks_path) {
            Ok(_) => panic!("chunks JSONL with blank text must fail"),
            Err(err) => err,
        };
        assert_eq!(err.code(), ErrorCode::MissingRequiredField);
        assert_eq!(detail_value(&err, "field"), Some("text"));
        assert_eq!(detail_value(&err, "line"), Some("1"));

        let embeddings_path = dir.join("embeddings.jsonl");
        fs::write(
            &embeddings_path,
            r#"{"chunk_external_id":"doc#1","model_name":"m","dim":2,"vector":[1.0,0.0],"document_external_id":"doc"}"#,
        )
        .unwrap();
        let err = match read_embeddings_jsonl(&embeddings_path) {
            Ok(_) => panic!("embeddings JSONL with unknown field must fail"),
            Err(err) => err,
        };
        assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
        assert_eq!(detail_value(&err, "field"), Some("document_external_id"));
        assert_eq!(detail_value(&err, "line"), Some("1"));

        fs::write(
            &embeddings_path,
            r#"{"chunk_external_id":"doc#1","model_name":"","dim":2,"vector":[1.0,0.0]}"#,
        )
        .unwrap();
        let err = match read_embeddings_jsonl(&embeddings_path) {
            Ok(_) => panic!("embeddings JSONL with blank model_name must fail"),
            Err(err) => err,
        };
        assert_eq!(err.code(), ErrorCode::MissingRequiredField);
        assert_eq!(detail_value(&err, "field"), Some("model_name"));
        assert_eq!(detail_value(&err, "line"), Some("1"));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn import_manifest_can_omit_shape_for_existing_collection() {
        let dir = temp_dir("existing-shape");
        let db_path = dir.join("existing.coffer");
        let db = Database::create(&db_path, DatabaseConfig::default()).unwrap();
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
        write_file(
            &dir,
            "documents.jsonl",
            r#"{"external_id":"doc","metadata":{}}"#,
        );
        write_file(
            &dir,
            "chunks.jsonl",
            r#"{"document_external_id":"doc","chunk_external_id":"doc#1","ordinal":1,"text":"existing collection shape","metadata":{}}"#,
        );
        fs::write(
            dir.join("manifest.json"),
            r#"{"schema_version":"0.1","dataset_id":"existing-fixture","collection":"notes","documents":"documents.jsonl","chunks":"chunks.jsonl","embedding_model":null}"#,
        )
        .unwrap();

        let summary = db
            .import_manifest(dir.join("manifest.json"), OnConflict::Error)
            .unwrap();
        assert_eq!(summary.dataset_id.as_deref(), Some("existing-fixture"));
        assert_eq!(
            summary
                .to_json()
                .get("dataset_id")
                .and_then(JsonValue::as_str),
            Some("existing-fixture")
        );
        assert_eq!(summary.documents_inserted, 1);
        assert_eq!(summary.chunks_inserted, 1);
        assert_eq!(
            db.collection("notes")
                .unwrap()
                .get_chunk_by_external_id("doc#1")
                .unwrap()
                .text,
            "existing collection shape"
        );
        drop(db);
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn import_manifest_requires_shape_for_new_collection() {
        let dir = temp_dir("new-shape-required");
        let db_path = dir.join("new.coffer");
        let db = Database::create(&db_path, DatabaseConfig::default()).unwrap();
        write_file(
            &dir,
            "documents.jsonl",
            r#"{"external_id":"doc","metadata":{}}"#,
        );
        write_file(
            &dir,
            "chunks.jsonl",
            r#"{"document_external_id":"doc","chunk_external_id":"doc#1","ordinal":1,"text":"new collection needs shape","metadata":{}}"#,
        );
        fs::write(
            dir.join("manifest.json"),
            r#"{"schema_version":"0.1","collection":"notes","documents":"documents.jsonl","chunks":"chunks.jsonl","embedding_model":null}"#,
        )
        .unwrap();

        let err = db
            .import_manifest(dir.join("manifest.json"), OnConflict::Error)
            .unwrap_err();
        assert_eq!(err.code(), ErrorCode::MissingRequiredField);
        assert_eq!(detail_value(&err, "field"), Some("embedding_dim"));
        drop(db);
        let _ = fs::remove_dir_all(dir);
    }
}
