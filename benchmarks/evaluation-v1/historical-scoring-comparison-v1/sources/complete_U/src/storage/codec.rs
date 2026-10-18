use super::*;
#[cfg(test)]
use crate::hash::digest_hex;
use crate::hash::{sha256, Sha256};
use crate::json::{
    JsonEncodingWorkspace, JsonInputKind, JsonOperationContext, JsonReservation, ProcessedJson,
    ReservedVec, WorkspaceReservation,
};
use crate::model::{SourceSpan, VectorIndexConfig};

pub(super) const COMMIT_PAYLOAD_BYTES: usize = 44;

pub(super) fn write_f32_values(output: &mut impl Write, values: &[f32]) -> Result<()> {
    let mut buffer = [0; 4 * 1024];
    for values in values.chunks(buffer.len() / 4) {
        let bytes = &mut buffer[..values.len() * 4];
        for (value, encoded) in values.iter().zip(bytes.chunks_exact_mut(4)) {
            encoded.copy_from_slice(&value.to_le_bytes());
        }
        output.write_all(bytes)?;
    }
    Ok(())
}

fn invalid() -> Error {
    Error::new(ErrorCode::FileFormatError, "invalid binary record payload")
}

#[cfg(test)]
pub(super) fn encode(record: &RecordPayload) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    encode_into(record, &mut bytes)?;
    Ok(bytes)
}

#[cfg(test)]
pub(super) fn encode_into(record: &RecordPayload, output: &mut impl Write) -> Result<()> {
    encode_with(
        record,
        Some(output),
        JsonOperationContext::standalone(),
        None,
        None,
        None,
    )
    .map(|_| ())
}

pub(super) fn encode_commit_into(
    commit: &CommitPayload,
    output: &mut impl Write,
    context: &JsonOperationContext,
) -> Result<()> {
    Writer {
        output: Some(output),
        len: 0,
        context: context.clone(),
        planned_json: None,
    }
    .commit(commit)
}

pub(super) fn encode_retained_into(
    record: &RecordPayload,
    output: &mut impl Write,
    context: &JsonOperationContext,
    metadata_work: &mut JsonReservation,
) -> Result<()> {
    encode_with(
        record,
        Some(output),
        context.clone(),
        Some(metadata_work),
        None,
        None,
    )
    .map(|_| ())
}

pub(super) fn encode_stored_into(
    record: &RecordPayload,
    output: &mut impl Write,
    context: &JsonOperationContext,
    metadata_work: &mut JsonReservation,
) -> Result<()> {
    encode_with(
        record,
        Some(output),
        context.clone(),
        Some(metadata_work),
        Some(JsonInputKind::StoredJson),
        None,
    )
    .map(|_| ())
}

#[cfg(test)]
pub(super) fn encoded_len(record: &RecordPayload) -> Result<u64> {
    encode_with(
        record,
        None::<&mut std::io::Sink>,
        JsonOperationContext::standalone(),
        None,
        None,
        None,
    )
}

fn encode_with(
    record: &RecordPayload,
    output: Option<&mut impl Write>,
    context: JsonOperationContext,
    metadata_work: Option<&mut JsonReservation>,
    json_kind: Option<JsonInputKind>,
    plan: Option<(&JsonEncodingWorkspace, Option<&JsonValue>)>,
) -> Result<u64> {
    let mut w = Writer {
        output,
        len: 0,
        context,
        planned_json: plan.map(|(workspace, _)| workspace),
    };
    match record {
        RecordPayload::CollectionCreate(c) => {
            let kind = json_kind.unwrap_or(JsonInputKind::CollectionConfig);
            w.id(&c.collection_id, "col")?;
            w.string(&c.name)?;
            w.u32(c.embedding_dim)?;
            w.byte(distance_metric_segment_id(c.distance_metric))?;
            w.optional_string(c.embedding_model.as_deref())?;
            w.byte(u8::from(c.vectors_normalized))?;
            w.timestamp(&c.created_at)?;
            if let Some((_, config)) = plan {
                w.json(config.ok_or_else(invalid)?, kind, None)?;
            } else {
                let mut config = collection_config_in(c, kind, &w.context)?;
                w.json(&config.value, kind, Some(&mut config.work))?;
            }
        }
        RecordPayload::CollectionDrop {
            collection_id,
            name,
            dropped_at,
            reason,
        } => {
            w.id(collection_id, "col")?;
            w.string(name)?;
            w.timestamp(dropped_at)?;
            w.optional_string(reason.as_deref())?;
        }
        RecordPayload::DocumentPut(d) => {
            w.id(&d.document_id, "doc")?;
            w.id(&d.collection_id, "col")?;
            w.string(&d.external_id)?;
            w.optional_string(d.title.as_deref())?;
            w.optional_string(d.source_uri.as_deref())?;
            w.json(
                &d.metadata,
                json_kind.unwrap_or(JsonInputKind::DocumentMetadata),
                metadata_work,
            )?;
            w.timestamp(&d.created_at)?;
            w.timestamp(&d.updated_at)?;
        }
        RecordPayload::ChunkPut(c) => {
            let digest = sha256(c.text.as_bytes());
            const HEX: &[u8; 16] = b"0123456789abcdef";
            if c.text_hash.len() != 64
                || !digest
                    .iter()
                    .zip(c.text_hash.as_bytes().chunks_exact(2))
                    .all(|(byte, pair)| {
                        pair == [HEX[usize::from(byte >> 4)], HEX[usize::from(byte & 15)]]
                    })
            {
                return Err(invalid());
            }
            w.id(&c.chunk_id, "chk")?;
            w.id(&c.document_id, "doc")?;
            w.id(&c.collection_id, "col")?;
            w.string(&c.chunk_external_id)?;
            w.string(&c.text)?;
            w.bytes(&digest)?;
            w.u32(c.ordinal)?;
            w.u64(c.token_count as u64)?;
            w.json(
                &c.metadata,
                json_kind.unwrap_or(JsonInputKind::ChunkMetadata),
                metadata_work,
            )?;
            let span = c.source_span.clone().unwrap_or_default();
            span.validate()?;
            w.optional_u64(span.start)?;
            w.optional_u64(span.end)?;
            w.optional_u32(span.line_start)?;
            w.optional_u32(span.line_end)?;
            w.timestamp(&c.created_at)?;
        }
        RecordPayload::EmbeddingPut(e) => {
            if e.vector.len() != e.dim || e.dim == 0 || e.vector.iter().any(|v| !v.is_finite()) {
                return Err(invalid());
            }
            w.id(&e.embedding_id, "emb")?;
            w.id(&e.chunk_id, "chk")?;
            w.id(&e.collection_id, "col")?;
            w.u32(e.dim)?;
            w.byte(1)?;
            w.optional_string(Some(&e.model_name))?;
            w.timestamp(&e.created_at)?;
            w.vector(&e.vector)?;
        }
        RecordPayload::Tombstone(t) => {
            let (tag, prefix) = match t.target_type.as_str() {
                "collection" => (1, "col"),
                "document" => (2, "doc"),
                "chunk" => (3, "chk"),
                "embedding" => (4, "emb"),
                _ => return Err(invalid()),
            };
            w.byte(tag)?;
            w.id(&t.target_id, prefix)?;
            w.timestamp(&t.deleted_at)?;
            w.optional_string(t.reason.as_deref())?;
        }
        RecordPayload::Commit(c) => w.commit(c)?,
        RecordPayload::CompactMarker(m) => {
            w.u64(m.compact_generation)?;
            w.u64(m.source_last_commit_sequence_no)?;
            for prefix in ["col_", "doc_", "chk_", "emb_"] {
                w.u64(m.id_high_watermarks.get(prefix).copied().unwrap_or(0))?;
            }
            w.u64(m.records_rewritten as u64)?;
            w.timestamp(&m.compacted_at)?;
            w.bytes(&m.compacted_main_checksum.to_le_bytes())?;
        }
    }
    Ok(w.len)
}

fn collection_config_in(
    collection: &CollectionInfo,
    kind: JsonInputKind,
    context: &JsonOperationContext,
) -> Result<CollectionEncoding> {
    let mut work = context.reservation();
    let vector_index = match &collection.vector_index {
        VectorIndexConfig::Exact => {
            let builder = crate::json::JsonBuilder::budgeted(&mut work, kind);
            builder.object([("type", builder.string("exact")?)])?
        }
        VectorIndexConfig::Ann { provider, params } => {
            let (params, params_work) = params.try_clone_in(kind, context)?.into_parts();
            work.merge(params_work);
            let builder = crate::json::JsonBuilder::budgeted(&mut work, kind);
            builder.object([
                ("type", builder.string("ann")?),
                ("provider", builder.string(provider)?),
                ("params", params),
            ])?
        }
    };
    let builder = crate::json::JsonBuilder::budgeted(&mut work, kind);
    let config = builder.object([
        ("tokenizer", collection.tokenizer.build_json(&builder)?),
        ("vector_index", vector_index),
    ])?;
    Ok(CollectionEncoding {
        value: config,
        work,
    })
}

struct CollectionEncoding {
    value: JsonValue,
    work: JsonReservation,
}

pub(super) struct BatchEncoding {
    configs: ReservedVec<(usize, CollectionEncoding)>,
    json: JsonEncodingWorkspace,
    context: JsonOperationContext,
}

pub(super) struct PreparedBatchEncoding {
    configs: ReservedVec<(usize, CollectionEncoding)>,
    frames: usize,
    frame_kind: JsonInputKind,
    context: JsonOperationContext,
}

impl PreparedBatchEncoding {
    pub(super) fn prepare(
        records: &[RecordPayload],
        source_work: &mut JsonReservation,
        context: &JsonOperationContext,
    ) -> Result<Self> {
        let collections = records
            .iter()
            .filter(|record| matches!(record, RecordPayload::CollectionCreate(_)))
            .count();
        let mut configs = ReservedVec::with_capacity(collections, context, "batch_prepare")?;
        let mut source_bytes = 0usize;
        let mut frames = 0;
        let mut frame_kind = JsonInputKind::DocumentMetadata;
        for (ordinal, record) in records.iter().enumerate() {
            let (value, kind) = match record {
                RecordPayload::DocumentPut(value) => {
                    if value.metadata.as_object().is_none() {
                        return Err(invalid());
                    }
                    (Some(&value.metadata), JsonInputKind::DocumentMetadata)
                }
                RecordPayload::ChunkPut(value) => {
                    if value.metadata.as_object().is_none() {
                        return Err(invalid());
                    }
                    (Some(&value.metadata), JsonInputKind::ChunkMetadata)
                }
                RecordPayload::CollectionCreate(CollectionInfo {
                    vector_index: VectorIndexConfig::Ann { params, .. },
                    ..
                }) => (Some(params), JsonInputKind::CollectionConfig),
                _ => (None, JsonInputKind::CollectionConfig),
            };
            if let Some(value) = value {
                let (required, bytes) = value.encoding_shape_in(kind, context)?;
                source_bytes = source_bytes
                    .checked_add(bytes)
                    .ok_or_else(|| std::io::Error::other("batch JSON size overflow"))?;
                source_work.grow(
                    source_bytes.saturating_sub(source_work.covered_bytes()),
                    kind,
                )?;
                if required > frames {
                    frames = required;
                    frame_kind = kind;
                }
            }
            if let RecordPayload::CollectionCreate(collection) = record {
                let config = collection_config_in(collection, kind, context)?;
                let required = config.value.encoding_shape_in(kind, context)?.0;
                if required > frames {
                    frames = required;
                    frame_kind = kind;
                }
                configs.push((ordinal, config));
            }
        }
        Ok(Self {
            configs,
            frames,
            frame_kind,
            context: context.clone(),
        })
    }

    pub(super) fn admit(self, reclaim: impl FnOnce() -> bool) -> Result<BatchEncoding> {
        // Catalog may take source JSON ownership between validation and append.
        // Admit the reusable serializer only after that transfer, before any IO.
        let json = crate::json::retry_after_memory_reclaim(
            || JsonEncodingWorkspace::new(self.frames, self.frame_kind, &self.context),
            reclaim,
        )?;
        Ok(BatchEncoding {
            configs: self.configs,
            json,
            context: self.context,
        })
    }
}

impl BatchEncoding {
    pub(super) fn encode(
        &self,
        ordinal: usize,
        record: &RecordPayload,
        output: &mut impl Write,
    ) -> Result<()> {
        let configs = self.configs.as_slice();
        let config = configs
            .binary_search_by_key(&ordinal, |entry| entry.0)
            .ok()
            .map(|index| &configs[index].1.value);
        encode_with(
            record,
            Some(output),
            self.context.clone(),
            None,
            None,
            Some((&self.json, config)),
        )
        .map(|_| ())
    }
}

#[cfg(test)]
pub(super) fn decode(record_type: u16, bytes: &[u8]) -> Result<RecordPayload> {
    decode_reader_in(
        record_type,
        &mut &bytes[..],
        bytes.len() as u64,
        true,
        &JsonOperationContext::standalone(),
    )
    .map(|(record, _)| record)
}

#[cfg(test)]
pub(super) fn decode_summary(
    record_type: u16,
    source: &mut impl Read,
    payload_len: u64,
) -> Result<RecordPayload> {
    decode_reader_in(
        record_type,
        source,
        payload_len,
        false,
        &JsonOperationContext::standalone(),
    )
    .map(|(record, _)| record)
}

#[cfg(test)]
pub(super) fn decode_reader_in(
    record_type: u16,
    source: &mut impl Read,
    payload_len: u64,
    retain_payloads: bool,
    context: &JsonOperationContext,
) -> Result<(RecordPayload, JsonReservation)> {
    decode_reader_with_kind_in(
        record_type,
        source,
        payload_len,
        retain_payloads,
        JsonInputKind::StoredJson,
        context,
        None,
    )
}

pub(super) fn decode_reader_with_kind_in(
    record_type: u16,
    source: &mut impl Read,
    payload_len: u64,
    retain_payloads: bool,
    json_kind: JsonInputKind,
    context: &JsonOperationContext,
    strings: Option<(&mut WorkspaceReservation, &'static str)>,
) -> Result<(RecordPayload, JsonReservation)> {
    let mut r = Reader::new(source, payload_len, context.clone());
    r.json_kind = json_kind;
    r.strings = strings;
    let record = match record_type {
        COLLECTION_CREATE => {
            let collection_id = r.id("col")?;
            let name = r.string()?;
            let embedding_dim = r.u32()? as usize;
            let distance_metric = distance_metric_from_segment_id(r.byte()?).ok_or_else(invalid)?;
            let embedding_model = r.optional_string()?;
            let vectors_normalized = r.boolean()?;
            let created_at = r.timestamp()?;
            let config = r.json_processed()?;
            let config = config.value.as_object().ok_or_else(invalid)?;
            let tokenizer =
                crate::TokenizerConfig::from_json(config.get("tokenizer").ok_or_else(invalid)?)?;
            let vector_index = config
                .get("vector_index")
                .map(crate::model::vector_index_from_json)
                .transpose()?
                .unwrap_or(VectorIndexConfig::Exact);
            if name.is_empty() || embedding_dim == 0 {
                return Err(invalid());
            }
            RecordPayload::CollectionCreate(CollectionInfo {
                collection_id,
                name,
                embedding_dim,
                distance_metric,
                embedding_model,
                vectors_normalized,
                created_at,
                tokenizer,
                vector_index,
            })
        }
        COLLECTION_DROP => RecordPayload::CollectionDrop {
            collection_id: r.id("col")?,
            name: r.string()?,
            dropped_at: r.timestamp()?,
            reason: r.optional_string()?,
        },
        DOCUMENT_PUT => RecordPayload::DocumentPut(Document {
            document_id: r.id("doc")?,
            collection_id: r.id("col")?,
            external_id: r.string()?,
            title: r.optional_string()?,
            source_uri: r.optional_string()?,
            metadata: r.json()?,
            created_at: r.timestamp()?,
            updated_at: r.timestamp()?,
        }),
        CHUNK_PUT => {
            let chunk_id = r.id("chk")?;
            let document_id = r.id("doc")?;
            let collection_id = r.id("col")?;
            let chunk_external_id = r.string()?;
            let (text, digest) = r.text(retain_payloads)?;
            if r.fixed::<32>()? != digest {
                return Err(invalid());
            }
            let mut text_hash = r.string_buffer(64)?;
            crate::hash::append_digest_hex(&mut text_hash, digest);
            let ordinal = r.u32()? as usize;
            let token_count = r.count()?;
            let metadata = r.json()?;
            let span = SourceSpan {
                start: r.optional_u64()?,
                end: r.optional_u64()?,
                line_start: r.optional_u32()?,
                line_end: r.optional_u32()?,
            };
            span.validate()?;
            let source_span = if span == SourceSpan::default() {
                None
            } else {
                Some(span)
            };
            RecordPayload::ChunkPut(Chunk {
                chunk_id,
                document_id,
                collection_id,
                chunk_external_id,
                text,
                text_hash,
                ordinal,
                token_count,
                metadata,
                source_span,
                embedding_id: None,
                created_at: r.timestamp()?,
            })
        }
        EMBEDDING_PUT => {
            let embedding_id = r.id("emb")?;
            let chunk_id = r.id("chk")?;
            let collection_id = r.id("col")?;
            let dim = r.u32()? as usize;
            if r.byte()? != 1 || dim == 0 {
                return Err(invalid());
            }
            let model_name = r.optional_string()?.unwrap_or_default();
            let created_at = r.timestamp()?;
            let vector = r.vector(dim, retain_payloads)?;
            RecordPayload::EmbeddingPut(Embedding {
                embedding_id,
                chunk_id,
                collection_id,
                chunk_external_id: String::new(),
                dim,
                model_name,
                created_at,
                vector,
            })
        }
        TOMBSTONE => {
            let (target_type, prefix) = match r.byte()? {
                1 => ("collection", "col"),
                2 => ("document", "doc"),
                3 => ("chunk", "chk"),
                4 => ("embedding", "emb"),
                _ => return Err(invalid()),
            };
            RecordPayload::Tombstone(Tombstone {
                target_type: r.copy_string(target_type)?,
                target_id: r.id(prefix)?,
                deleted_at: r.timestamp()?,
                reason: r.optional_string()?,
            })
        }
        COMMIT => RecordPayload::Commit(CommitPayload {
            batch_id: r.u64()?,
            first_sequence_no: r.u64()?,
            last_sequence_no: r.u64()?,
            record_count: r.count()?,
            batch_checksum: r.u32()? as u64,
            committed_at: r.timestamp()?,
        }),
        COMPACT_MARKER => {
            let compact_generation = r.u64()?;
            let source_last_commit_sequence_no = r.u64()?;
            r.reserve_owned_bytes(crate::json::btree_workspace::<String, u64>(4))?;
            let mut id_high_watermarks = BTreeMap::new();
            for prefix in ["col_", "doc_", "chk_", "emb_"] {
                id_high_watermarks.insert(r.copy_string(prefix)?, r.u64()?);
            }
            RecordPayload::CompactMarker(CompactMarker {
                compact_generation,
                source_last_commit_sequence_no,
                id_high_watermarks,
                records_rewritten: r.count()?,
                compacted_at: r.timestamp()?,
                compacted_main_checksum: r.u32()?,
            })
        }
        _ => return Err(invalid()),
    };
    if r.remaining != 0 {
        return Err(invalid());
    }
    Ok((record, r.work))
}

struct Writer<'a, W> {
    output: Option<&'a mut W>,
    len: u64,
    context: JsonOperationContext,
    planned_json: Option<&'a JsonEncodingWorkspace>,
}

impl<W: Write> Writer<'_, W> {
    fn commit(&mut self, commit: &CommitPayload) -> Result<()> {
        self.u64(commit.batch_id)?;
        self.u64(commit.first_sequence_no)?;
        self.u64(commit.last_sequence_no)?;
        self.u64(commit.record_count as u64)?;
        self.u32(u32::try_from(commit.batch_checksum).map_err(|_| invalid())? as usize)?;
        self.timestamp(&commit.committed_at)
    }

    fn advance(&mut self, bytes: usize) -> Result<()> {
        self.len = self.len.checked_add(bytes as u64).ok_or_else(invalid)?;
        Ok(())
    }

    fn bytes(&mut self, bytes: &[u8]) -> Result<()> {
        self.advance(bytes.len())?;
        if let Some(output) = self.output.as_mut() {
            output.write_all(bytes)?;
        }
        Ok(())
    }

    fn vector(&mut self, values: &[f32]) -> Result<()> {
        self.advance(values.len().checked_mul(4).ok_or_else(invalid)?)?;
        let Some(output) = self.output.as_mut() else {
            return Ok(());
        };
        write_f32_values(output, values)
    }

    fn byte(&mut self, value: u8) -> Result<()> {
        self.bytes(&[value])
    }
    fn u64(&mut self, value: u64) -> Result<()> {
        self.bytes(&value.to_le_bytes())
    }
    fn u32(&mut self, value: usize) -> Result<()> {
        self.bytes(&u32::try_from(value).map_err(|_| invalid())?.to_le_bytes())
    }
    fn id(&mut self, value: &str, prefix: &str) -> Result<()> {
        let n = value
            .strip_prefix(prefix)
            .map(|s| s.strip_prefix('_').unwrap_or(s))
            .and_then(|s| s.parse::<u64>().ok())
            .filter(|n| *n > 0)
            .ok_or_else(invalid)?;
        self.u64(n)
    }
    fn string(&mut self, value: &str) -> Result<()> {
        self.u32(value.len())?;
        self.bytes(value.as_bytes())
    }
    fn optional_string(&mut self, value: Option<&str>) -> Result<()> {
        self.byte(u8::from(value.is_some()))?;
        if let Some(value) = value {
            self.string(value)?;
        }
        Ok(())
    }
    fn optional_u64(&mut self, value: Option<u64>) -> Result<()> {
        self.byte(u8::from(value.is_some()))?;
        if let Some(value) = value {
            self.u64(value)?;
        }
        Ok(())
    }
    fn optional_u32(&mut self, value: Option<u32>) -> Result<()> {
        self.byte(u8::from(value.is_some()))?;
        if let Some(value) = value {
            self.bytes(&value.to_le_bytes())?;
        }
        Ok(())
    }
    fn timestamp(&mut self, value: &str) -> Result<()> {
        self.u64(time::parse_rfc3339_millis(value).ok_or_else(invalid)?)
    }
    fn json(
        &mut self,
        value: &JsonValue,
        kind: JsonInputKind,
        retained: Option<&mut JsonReservation>,
    ) -> Result<()> {
        if value.as_object().is_none() {
            return Err(invalid());
        }
        if let Some(workspace) = self.planned_json {
            struct Count(usize);
            impl Write for Count {
                fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                    self.0 = self
                        .0
                        .checked_add(bytes.len())
                        .ok_or_else(|| std::io::Error::other("JSON size overflow"))?;
                    Ok(bytes.len())
                }
                fn flush(&mut self) -> std::io::Result<()> {
                    Ok(())
                }
            }
            let mut count = Count(0);
            workspace.write(value, kind, &mut count)?;
            self.u32(count.0)?;
            self.advance(count.0)?;
            if let Some(output) = self.output.as_mut() {
                workspace.write(value, kind, output)?;
            }
            return Ok(());
        }
        let mut borrowed = self.context.reservation();
        let work = retained.unwrap_or(&mut borrowed);
        let bytes = value.canonical_bytes_retained_in(kind, work)?;
        self.u32(bytes)?;
        self.advance(bytes)?;
        if let Some(output) = self.output.as_mut() {
            value.write_value_retained_in(output, kind, &self.context, work, false)?;
        }
        Ok(())
    }
}

struct Reader<'a, R> {
    source: R,
    remaining: u64,
    context: JsonOperationContext,
    work: JsonReservation,
    json_kind: JsonInputKind,
    strings: Option<(&'a mut WorkspaceReservation, &'static str)>,
}

impl<R: Read> Reader<'_, R> {
    fn new(source: R, remaining: u64, context: JsonOperationContext) -> Self {
        let work = context.reservation();
        Self {
            source,
            remaining,
            context,
            work,
            json_kind: JsonInputKind::StoredJson,
            strings: None,
        }
    }
    fn read_into(&mut self, buffer: &mut [u8]) -> Result<()> {
        if buffer.len() as u64 > self.remaining {
            return Err(invalid());
        }
        self.source.read_exact(buffer)?;
        self.remaining -= buffer.len() as u64;
        Ok(())
    }
    fn fixed<const N: usize>(&mut self) -> Result<[u8; N]> {
        let mut bytes = [0; N];
        self.read_into(&mut bytes)?;
        Ok(bytes)
    }
    fn byte(&mut self) -> Result<u8> {
        Ok(self.fixed::<1>()?[0])
    }
    fn boolean(&mut self) -> Result<bool> {
        match self.byte()? {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(invalid()),
        }
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.fixed()?))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.fixed()?))
    }
    fn count(&mut self) -> Result<usize> {
        usize::try_from(self.u64()?).map_err(|_| invalid())
    }
    fn id(&mut self, prefix: &str) -> Result<String> {
        use std::fmt::Write as _;
        let value = self.u64()?;
        if value == 0 {
            return Err(invalid());
        }
        let digits = (value.ilog10() as usize + 1).max(16);
        let mut output = self.string_buffer(prefix.len() + 1 + digits)?;
        write!(output, "{prefix}_{value:016}").expect("writing to String is infallible");
        Ok(output)
    }

    fn reserve_owned_bytes(&mut self, len: usize) -> Result<()> {
        if let Some((work, phase)) = &mut self.strings {
            let required = work
                .reserved_bytes()
                .checked_add(len)
                .ok_or_else(|| std::io::Error::other("binary payload capacity overflow"))?;
            work.grow_to(required, required, phase)?;
        }
        Ok(())
    }

    fn string_buffer(&mut self, len: usize) -> Result<String> {
        self.reserve_owned_bytes(len)?;
        let mut output = String::new();
        output
            .try_reserve_exact(len)
            .map_err(|_| Error::new(ErrorCode::IoError, "binary string allocation failed"))?;
        if output.capacity() != len {
            return Err(std::io::Error::other("binary string allocation exceeded request").into());
        }
        Ok(output)
    }

    fn copy_string(&mut self, value: &str) -> Result<String> {
        let mut output = self.string_buffer(value.len())?;
        output.push_str(value);
        Ok(output)
    }

    fn string(&mut self) -> Result<String> {
        let len = self.u32()? as usize;
        if len as u64 > self.remaining {
            return Err(invalid());
        }
        self.reserve_owned_bytes(len)?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(len)
            .map_err(|_| Error::new(ErrorCode::IoError, "binary string allocation failed"))?;
        if bytes.capacity() != len {
            return Err(std::io::Error::other("binary string allocation exceeded request").into());
        }
        bytes.resize(len, 0);
        self.read_into(&mut bytes)?;
        String::from_utf8(bytes).map_err(|_| invalid())
    }
    fn text(&mut self, retain: bool) -> Result<(String, [u8; 32])> {
        if retain {
            let text = self.string()?;
            let digest = sha256(text.as_bytes());
            return Ok((text, digest));
        }
        let mut remaining = self.u32()? as usize;
        if remaining as u64 > self.remaining {
            return Err(invalid());
        }
        let mut hash = Sha256::default();
        let mut buffer = [0; 8 * 1024 + 3];
        let mut carry = 0;
        while remaining != 0 {
            let count = remaining.min(buffer.len() - 3);
            self.read_into(&mut buffer[carry..carry + count])?;
            hash.update(&buffer[carry..carry + count]);
            remaining -= count;
            let filled = carry + count;
            carry = match std::str::from_utf8(&buffer[..filled]) {
                Ok(_) => 0,
                Err(err) if err.error_len().is_none() && remaining != 0 => {
                    let tail = filled - err.valid_up_to();
                    buffer.copy_within(err.valid_up_to()..filled, 0);
                    tail
                }
                Err(_) => return Err(invalid()),
            };
        }
        Ok((String::new(), hash.finish()))
    }
    fn vector(&mut self, dim: usize, retain: bool) -> Result<Vec<f32>> {
        let mut remaining = dim.checked_mul(4).ok_or_else(invalid)?;
        if remaining as u64 > self.remaining {
            return Err(invalid());
        }
        let mut vector = Vec::new();
        if retain {
            self.reserve_owned_bytes(remaining)?;
            vector
                .try_reserve_exact(dim)
                .map_err(|_| Error::new(ErrorCode::IoError, "binary vector allocation failed"))?;
            if vector.capacity() != dim {
                return Err(
                    std::io::Error::other("binary vector allocation exceeded request").into(),
                );
            }
        }
        let mut buffer = [0; 4 * 1024];
        while remaining != 0 {
            let count = remaining.min(buffer.len());
            self.read_into(&mut buffer[..count])?;
            remaining -= count;
            for bytes in buffer[..count].chunks_exact(4) {
                let value = f32::from_le_bytes(bytes.try_into().unwrap());
                if !value.is_finite() {
                    return Err(invalid());
                }
                if retain {
                    vector.push(value);
                }
            }
        }
        Ok(vector)
    }
    fn optional_string(&mut self) -> Result<Option<String>> {
        if self.boolean()? {
            Ok(Some(self.string()?))
        } else {
            Ok(None)
        }
    }
    fn optional_u64(&mut self) -> Result<Option<u64>> {
        if !self.boolean()? {
            return Ok(None);
        }
        Ok(Some(self.u64()?))
    }
    fn optional_u32(&mut self) -> Result<Option<u32>> {
        if !self.boolean()? {
            return Ok(None);
        }
        Ok(Some(self.u32()?))
    }
    fn timestamp(&mut self) -> Result<String> {
        let value = self.u64()?;
        // Public timestamps support the four-digit RFC3339 year range.
        if value > 253_402_300_799_999 {
            return Err(invalid());
        }
        let mut output = self.string_buffer(if value % 1000 == 0 { 20 } else { 24 })?;
        time::write_unix_millis(&mut output, value).expect("writing to String is infallible");
        Ok(output)
    }
    fn json(&mut self) -> Result<JsonValue> {
        let (value, work) = self.json_processed()?.into_parts();
        self.work.merge(work);
        Ok(value)
    }

    fn json_processed(&mut self) -> Result<ProcessedJson> {
        let len = u64::from(self.u32()?);
        if len > self.remaining {
            return Err(invalid());
        }
        let mut limited = (&mut self.source).take(len);
        let value =
            crate::json::parse_canonical_reader_in(&mut limited, self.json_kind, &self.context)?;
        if limited.limit() != 0 || value.value.as_object().is_none() {
            return Err(invalid());
        }
        self.remaining -= len;
        Ok(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_identifiers_and_timestamps_are_exactly_charged_at_numeric_boundaries() {
        use crate::json::JsonProcessingPolicyV1;
        let mut ids = vec![
            1,
            9,
            10,
            9_999_999_999_999_999,
            10_000_000_000_000_000,
            u64::MAX,
        ];
        ids.extend((0..100_u64).map(|seed| {
            seed.wrapping_mul(6364136223846793005)
                .wrapping_add(1)
                .max(1)
        }));
        for value in ids {
            for prefix in ["col", "doc", "chk", "emb"] {
                let expected = format!("{prefix}_{value:016}");
                let context = JsonOperationContext::standalone();
                let mut busy = context.workspace_reservation();
                let occupied = JsonProcessingPolicyV1::WORK_BUDGET_BYTES - expected.len();
                busy.grow_to(occupied, occupied, "operation_scratch")
                    .unwrap();
                let mut strings = context.workspace_reservation();
                let encoded = value.to_le_bytes();
                let mut reader = Reader::new(&encoded[..], 8, context.clone());
                reader.strings = Some((&mut strings, "batch_prepare"));
                let actual = reader.id(prefix).unwrap();
                drop(reader);
                assert_eq!(actual, expected);
                assert_eq!(strings.reserved_bytes(), actual.capacity());
                assert_eq!(actual.capacity(), actual.len());
                drop(actual);
                strings.retain(0);
                busy.grow_to(occupied + 1, occupied + 1, "operation_scratch")
                    .unwrap();
                let mut reader = Reader::new(&encoded[..], 8, context.clone());
                reader.strings = Some((&mut strings, "batch_prepare"));
                assert_eq!(
                    reader.id(prefix).unwrap_err().code(),
                    ErrorCode::ResourceExhausted
                );
                drop(reader);
                assert_eq!(strings.reserved_bytes(), 0);
            }
        }
        for millis in [0, 1, 999, 1000, 253_402_300_799_000, 253_402_300_799_999] {
            let context = JsonOperationContext::standalone();
            let mut strings = context.workspace_reservation();
            let encoded = u64::to_le_bytes(millis);
            let mut reader = Reader::new(&encoded[..], 8, context);
            reader.strings = Some((&mut strings, "batch_prepare"));
            let actual = reader.timestamp().unwrap();
            drop(reader);
            assert_eq!(actual, time::format_unix_millis(millis));
            assert_eq!(strings.reserved_bytes(), actual.len());
            assert_eq!(actual.capacity(), actual.len());
        }
    }

    #[test]
    fn staged_strings_reserve_before_reading_and_report_shared_pressure_across_100_sizes() {
        use crate::json::JsonProcessingPolicyV1;
        for seed in 0..100 {
            let text = "field-\u{e9}".repeat(seed * 17 + 1);
            let len = text.len();
            let mut encoded = (len as u32).to_le_bytes().to_vec();
            encoded.extend_from_slice(text.as_bytes());
            for short_by in [0, 1] {
                let context = JsonOperationContext::standalone();
                let mut busy = context.workspace_reservation();
                let occupied = JsonProcessingPolicyV1::WORK_BUDGET_BYTES - len + short_by;
                busy.grow_to(occupied, occupied, "operation_scratch")
                    .unwrap();
                let mut strings = context.workspace_reservation();
                let mut source = std::io::Cursor::new(&encoded);
                let mut reader = Reader::new(&mut source, encoded.len() as u64, context.clone());
                reader.strings = Some((&mut strings, "batch_prepare"));
                let result = reader.string();
                drop(reader);
                if short_by == 0 {
                    let decoded = result.unwrap();
                    assert_eq!(decoded, text);
                    assert_eq!(strings.reserved_bytes(), decoded.capacity());
                    assert_eq!(source.position(), encoded.len() as u64);
                    drop(decoded);
                } else {
                    let error = result.unwrap_err();
                    assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                    for (key, expected) in [
                        ("resource", "memory".to_owned()),
                        ("operation", "batch_prepare".to_owned()),
                        (
                            "limit_bytes",
                            JsonProcessingPolicyV1::WORK_BUDGET_BYTES.to_string(),
                        ),
                        (
                            "requested_bytes",
                            (JsonProcessingPolicyV1::WORK_BUDGET_BYTES + 1).to_string(),
                        ),
                    ] {
                        assert!(error
                            .detail()
                            .iter()
                            .any(|(name, value)| name == key && value == &expected));
                    }
                    assert_eq!(strings.reserved_bytes(), 0);
                    assert_eq!(source.position(), 4); // Only the length prefix was read.
                }
                drop(strings);
                drop(busy);
                context
                    .reservation()
                    .grow(
                        JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
                        JsonInputKind::JsonlRecord,
                    )
                    .unwrap();
            }
        }
    }

    struct CappedReader<'a> {
        bytes: &'a [u8],
        fragment: usize,
        max_request: usize,
    }

    impl Read for CappedReader<'_> {
        fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
            assert!(buffer.len() <= 64 * 1024, "unbounded payload read");
            self.max_request = self.max_request.max(buffer.len());
            let count = buffer.len().min(self.bytes.len()).min(self.fragment);
            buffer[..count].copy_from_slice(&self.bytes[..count]);
            self.bytes = &self.bytes[count..];
            Ok(count)
        }
    }

    fn summary(record_type: u16, bytes: &[u8]) -> Result<RecordPayload> {
        let mut reader = CappedReader {
            bytes,
            fragment: 7,
            max_request: 0,
        };
        let result = decode_summary(record_type, &mut reader, bytes.len() as u64)?;
        assert!(reader.bytes.is_empty());
        assert!(reader.max_request <= 64 * 1024);
        Ok(result)
    }

    struct FragmentedWriter {
        bytes: Vec<u8>,
        fragment: usize,
        calls: usize,
        fail_after: usize,
        max_request: usize,
    }

    impl FragmentedWriter {
        fn new(fragment: usize, fail_after: usize) -> Self {
            Self {
                bytes: Vec::new(),
                fragment,
                calls: 0,
                fail_after,
                max_request: 0,
            }
        }
    }

    impl Write for FragmentedWriter {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.calls += 1;
            self.max_request = self.max_request.max(bytes.len());
            if self.calls % 5 == 0 {
                return Err(std::io::ErrorKind::Interrupted.into());
            }
            if self.bytes.len() == self.fail_after {
                return Err(std::io::Error::other("injected payload write failure"));
            }
            let count = bytes
                .len()
                .min(self.fragment)
                .min(self.fail_after - self.bytes.len());
            self.bytes.extend_from_slice(&bytes[..count]);
            Ok(count)
        }
        fn flush(&mut self) -> std::io::Result<()> {
            panic!("encoder must not flush its caller's stream")
        }
    }

    fn assert_planned_encoding(records: &[RecordPayload]) {
        let context = JsonOperationContext::standalone();
        let mut source = context.reservation();
        let plan = PreparedBatchEncoding::prepare(records, &mut source, &context).unwrap();
        let retained = source.reserved_bytes();
        let mut catalog = context.catalog_reservation();
        catalog.absorb(&mut source, None, retained, 0).unwrap();
        assert_eq!(source.reserved_bytes(), 0);
        let plan = plan.admit(|| false).unwrap();
        let limit = crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES;
        let mut held = context.workspace_reservation();
        held.grow_to(0, limit, "operation_scratch").unwrap();
        assert!(context
            .workspace_reservation()
            .grow_to(1, 1, "operation_scratch")
            .is_err());
        for (ordinal, record) in records.iter().enumerate() {
            let expected = encode(record).unwrap();
            for fragment in [1, 7, 1024] {
                let mut output = FragmentedWriter::new(fragment, usize::MAX);
                plan.encode(ordinal, record, &mut output).unwrap();
                assert_eq!(output.bytes, expected);
            }
            for fail_after in [0, 1, expected.len() / 2, expected.len() - 1] {
                let mut output = FragmentedWriter::new(7, fail_after);
                let error = plan.encode(ordinal, record, &mut output).unwrap_err();
                assert_eq!(error.code(), ErrorCode::IoError);
                assert_eq!(output.bytes, expected[..fail_after]);
            }
            assert_eq!(source.reserved_bytes(), 0);
            assert_eq!(catalog.reserved_bytes(), retained);
        }
        drop(plan);
        held.grow_to(limit, limit, "operation_scratch").unwrap();
    }

    fn golden_bytes(hex: &str) -> Vec<u8> {
        let hex: String = hex.split_ascii_whitespace().collect();
        assert_eq!(hex.len() % 2, 0);
        (0..hex.len())
            .step_by(2)
            .map(|start| u8::from_str_radix(&hex[start..start + 2], 16).unwrap())
            .collect()
    }

    fn assert_payload_golden(
        record: &RecordPayload,
        kind: u16,
        golden: &[u8],
        length: usize,
    ) -> RecordPayload {
        assert_eq!(golden.len(), length);
        assert_eq!(encode(record).unwrap(), golden);
        assert_eq!(encoded_len(record).unwrap(), length as u64);
        let context = JsonOperationContext::standalone();
        let mut source = context.reservation();
        let plan =
            PreparedBatchEncoding::prepare(std::slice::from_ref(record), &mut source, &context)
                .unwrap()
                .admit(|| false)
                .unwrap();
        let mut output = FragmentedWriter::new(7, usize::MAX);
        plan.encode(0, record, &mut output).unwrap();
        assert_eq!(output.bytes, golden);
        for end in 0..length {
            assert!(decode(kind, &golden[..end]).is_err(), "end={end}");
        }
        let mut extra = golden.to_vec();
        extra.push(0);
        assert!(decode(kind, &extra).is_err());
        decode(kind, golden).unwrap()
    }

    #[test]
    fn collection_payloads_independent_byte_goldens() {
        let create = RecordPayload::CollectionCreate(CollectionInfo {
            collection_id: "col_1".into(),
            name: "\u{00e9}".into(),
            embedding_dim: 2,
            distance_metric: DistanceMetric::Cosine,
            embedding_model: None,
            vectors_normalized: true,
            created_at: "1970-01-01T00:00:01Z".into(),
            tokenizer: Default::default(),
            vector_index: VectorIndexConfig::Exact,
        });
        let config = br#"{"tokenizer":{"config":{"case_fold":"lower","split":"unicode_punctuation","stop_words":"default_en"},"language_hint":"en","name":"default_en","profile":"latin_code_default","version":"0.1"},"vector_index":{"type":"exact"}}"#;
        assert_eq!(config.len(), 222);
        let mut golden = golden_bytes(
            "0100000000000000 02000000c3a9 02000000 01 00 01
             e803000000000000 de000000",
        );
        golden.extend_from_slice(config);
        let RecordPayload::CollectionCreate(decoded) =
            assert_payload_golden(&create, COLLECTION_CREATE, &golden, 255)
        else {
            panic!()
        };
        assert_eq!(decoded.collection_id, "col_0000000000000001");
        assert_eq!(decoded.name, "\u{00e9}");
        assert_eq!(decoded.embedding_dim, 2);
        assert_eq!(decoded.distance_metric, DistanceMetric::Cosine);
        assert!(decoded.embedding_model.is_none());
        assert!(decoded.vectors_normalized);
        assert_eq!(decoded.created_at, "1970-01-01T00:00:01Z");
        assert_eq!(decoded.tokenizer, crate::TokenizerConfig::default());
        assert_eq!(decoded.vector_index, VectorIndexConfig::Exact);
        for (offset, value) in [(18, 0), (18, 4), (18, 255), (20, 2), (20, 255)] {
            let mut invalid = golden.clone();
            invalid[offset] = value;
            assert_eq!(
                decode(COLLECTION_CREATE, &invalid).unwrap_err().code(),
                ErrorCode::FileFormatError
            );
        }

        let drop = RecordPayload::CollectionDrop {
            collection_id: "col_1".into(),
            name: "\u{00e9}".into(),
            dropped_at: "1970-01-01T00:00:02Z".into(),
            reason: Some("r".into()),
        };
        let golden = golden_bytes("0100000000000000 02000000c3a9 d007000000000000 01 0100000072");
        let RecordPayload::CollectionDrop {
            collection_id,
            name,
            dropped_at,
            reason,
        } = assert_payload_golden(&drop, COLLECTION_DROP, &golden, 28)
        else {
            panic!()
        };
        assert_eq!(collection_id, "col_0000000000000001");
        assert_eq!(name, "\u{00e9}");
        assert_eq!(dropped_at, "1970-01-01T00:00:02Z");
        assert_eq!(reason.as_deref(), Some("r"));
    }

    #[test]
    fn tombstone_payload_independent_byte_goldens() {
        for (tag, target, prefix) in [
            (1, "collection", "col"),
            (2, "document", "doc"),
            (3, "chunk", "chk"),
            (4, "embedding", "emb"),
        ] {
            let tombstone = RecordPayload::Tombstone(Tombstone {
                target_type: target.into(),
                target_id: format!("{prefix}_3"),
                deleted_at: "1970-01-01T00:00:02Z".into(),
                reason: None,
            });
            let mut golden = golden_bytes("03 0300000000000000 d007000000000000 00");
            golden[0] = tag;
            let RecordPayload::Tombstone(decoded) =
                assert_payload_golden(&tombstone, TOMBSTONE, &golden, 18)
            else {
                panic!()
            };
            assert_eq!(decoded.target_type, target);
            assert_eq!(decoded.target_id, format!("{prefix}_0000000000000003"));
            assert_eq!(decoded.deleted_at, "1970-01-01T00:00:02Z");
            assert!(decoded.reason.is_none());
            for invalid_tag in [0, 5, 255] {
                let mut invalid = golden.clone();
                invalid[0] = invalid_tag;
                assert_eq!(
                    decode(TOMBSTONE, &invalid).unwrap_err().code(),
                    ErrorCode::FileFormatError
                );
            }
            for invalid_kind in [0, 9, u16::MAX] {
                assert_eq!(
                    decode(invalid_kind, &golden).unwrap_err().code(),
                    ErrorCode::FileFormatError
                );
            }
        }
    }

    #[test]
    fn chunk_payload_independent_byte_goldens_preserve_optional_span_tags() {
        let prefix = golden_bytes(
            "0300000000000000 0200000000000000 0100000000000000
             010000006b 03000000616263
             ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad
             04030201 0100000000000000 020000007b7d",
        );
        assert_eq!(prefix.len(), 86);
        let timestamp = golden_bytes("e803000000000000");
        let byte_span = SourceSpan {
            start: Some(0x0102_0304_0506_0708),
            end: Some(0x1112_1314_1516_1718),
            ..Default::default()
        };
        let line_span = SourceSpan {
            line_start: Some(0x0102_0304),
            line_end: Some(0x1112_1314),
            ..Default::default()
        };
        let chunk = Chunk {
            chunk_id: "chk_3".into(),
            document_id: "doc_2".into(),
            collection_id: "col_1".into(),
            chunk_external_id: "k".into(),
            text: "abc".into(),
            text_hash: "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad".into(),
            ordinal: 0x0102_0304,
            token_count: 1,
            metadata: JsonValue::object(),
            embedding_id: Some("emb_4".into()),
            source_span: None,
            created_at: "1970-01-01T00:00:01Z".into(),
        };
        for (span, encoded_span, length) in [
            (None, "00 00 00 00", 98),
            (
                Some(byte_span.clone()),
                "01 0807060504030201 01 1817161514131211 00 00",
                114,
            ),
            (
                Some(line_span.clone()),
                "00 00 01 04030201 01 14131211",
                106,
            ),
            (
                Some(SourceSpan {
                    line_start: line_span.line_start,
                    line_end: line_span.line_end,
                    ..byte_span.clone()
                }),
                "01 0807060504030201 01 1817161514131211 01 04030201 01 14131211",
                122,
            ),
            (
                Some(SourceSpan {
                    start: Some(0),
                    end: Some(0),
                    ..Default::default()
                }),
                "01 0000000000000000 01 0000000000000000 00 00",
                114,
            ),
        ] {
            let mut input = chunk.clone();
            input.source_span = span.clone();
            let mut golden = prefix.clone();
            golden.extend(golden_bytes(encoded_span));
            golden.extend_from_slice(&timestamp);
            let RecordPayload::ChunkPut(decoded) = assert_payload_golden(
                &RecordPayload::ChunkPut(input.clone()),
                CHUNK_PUT,
                &golden,
                length,
            ) else {
                panic!()
            };
            assert_eq!(decoded.chunk_id, "chk_0000000000000003");
            assert_eq!(decoded.document_id, "doc_0000000000000002");
            assert_eq!(decoded.collection_id, "col_0000000000000001");
            assert_eq!(decoded.chunk_external_id, "k");
            assert_eq!(decoded.text, "abc");
            assert_eq!(decoded.text_hash, chunk.text_hash);
            assert_eq!(decoded.ordinal, 0x0102_0304);
            assert_eq!(decoded.token_count, 1);
            assert_eq!(decoded.metadata, JsonValue::object());
            assert_eq!(decoded.created_at, "1970-01-01T00:00:01Z");
            assert_eq!(decoded.source_span, span);
            assert!(decoded.embedding_id.is_none());
            if length == 122 {
                for offset in [0, 9, 18, 23] {
                    let mut invalid = golden.clone();
                    invalid[prefix.len() + offset] = 2;
                    assert_eq!(
                        decode(CHUNK_PUT, &invalid).unwrap_err().code(),
                        ErrorCode::FileFormatError
                    );
                }
            }
            if input.source_span.is_none() {
                input.source_span = Some(SourceSpan::default());
                let RecordPayload::ChunkPut(decoded) = assert_payload_golden(
                    &RecordPayload::ChunkPut(input),
                    CHUNK_PUT,
                    &golden,
                    length,
                ) else {
                    panic!()
                };
                assert!(decoded.source_span.is_none());
            }
        }
        let mut input = chunk;
        input.token_count = usize::MAX;
        let mut golden = prefix.clone();
        golden[72..80].fill(0xff);
        golden.extend_from_slice(&[0; 4]);
        golden.extend_from_slice(&timestamp);
        let RecordPayload::ChunkPut(decoded) =
            assert_payload_golden(&RecordPayload::ChunkPut(input), CHUNK_PUT, &golden, 98)
        else {
            panic!()
        };
        assert_eq!(decoded.token_count, usize::MAX);
        // Complete frames with invalid pairs must fail before any trailing-byte check.
        for span in [
            "01 0807060504030201 00 00 00",
            "00 01 1817161514131211 00 00",
            "00 00 01 04030201 00",
            "00 00 00 01 14131211",
            "01 1817161514131211 01 0807060504030201 00 00",
            "00 00 01 00000000 01 01000000",
            "00 00 01 14131211 01 04030201",
        ] {
            let mut invalid = prefix.clone();
            invalid.extend(golden_bytes(span));
            invalid.extend_from_slice(&timestamp);
            assert!(decode(CHUNK_PUT, &invalid).is_err(), "span={span}");
        }
    }

    #[test]
    fn compact_marker_payload_independent_byte_golden_preserves_all_watermarks() {
        let watermarks = BTreeMap::from([
            ("col_".into(), 0x8102_0304_0506_0708),
            ("doc_".into(), 0x920a_0b0c_0d0e_0f10),
            ("chk_".into(), 0xa311_1213_1415_1617),
            ("emb_".into(), 0xb418_191a_1b1c_1d1e),
        ]);
        let count = usize::try_from(0x0000_0001_0000_0005u64).unwrap();
        let marker = RecordPayload::CompactMarker(CompactMarker {
            compact_generation: 0x1122_3344_5566_7788,
            source_last_commit_sequence_no: 0x1020_3040_5060_7080,
            id_high_watermarks: watermarks.clone(),
            records_rewritten: count,
            compacted_at: "1970-01-01T00:00:01Z".into(),
            compacted_main_checksum: 0x89ab_cdef,
        });
        let golden = golden_bytes(
            "8877665544332211 8070605040302010
             0807060504030281 100f0e0d0c0b0a92 17161514131211a3 1e1d1c1b1a1918b4
             0500000001000000 e803000000000000 efcdab89",
        );
        let RecordPayload::CompactMarker(decoded) =
            assert_payload_golden(&marker, COMPACT_MARKER, &golden, 68)
        else {
            panic!()
        };
        assert_eq!(decoded.compact_generation, 0x1122_3344_5566_7788);
        assert_eq!(
            decoded.source_last_commit_sequence_no,
            0x1020_3040_5060_7080
        );
        assert_eq!(decoded.id_high_watermarks, watermarks);
        assert_eq!(decoded.records_rewritten, count);
        assert_eq!(decoded.compacted_at, "1970-01-01T00:00:01Z");
        assert_eq!(decoded.compacted_main_checksum, 0x89ab_cdef);
        let RecordPayload::CompactMarker(mut input) = marker else {
            panic!()
        };
        input.records_rewritten = usize::MAX;
        let mut golden = golden;
        golden[48..56].fill(0xff);
        let RecordPayload::CompactMarker(decoded) = assert_payload_golden(
            &RecordPayload::CompactMarker(input),
            COMPACT_MARKER,
            &golden,
            68,
        ) else {
            panic!()
        };
        assert_eq!(decoded.records_rewritten, usize::MAX);
    }

    #[test]
    fn streamed_encoding_preserves_bytes_with_short_writes_interrupts_and_failures() {
        let record = RecordPayload::DocumentPut(Document {
            document_id: "doc_2".into(),
            collection_id: "col_1".into(),
            external_id: "external".into(),
            title: Some("title".into()),
            source_uri: None,
            metadata: object_from_pairs([
                ("z", JsonValue::Number(-0.0)),
                ("a", JsonValue::String("\u{00e9}\n\"\\".into())),
            ]),
            created_at: "1970-01-01T00:00:00Z".into(),
            updated_at: "1970-01-01T00:00:00.001Z".into(),
        });
        let mut expected = Vec::new();
        expected.extend_from_slice(&2u64.to_le_bytes());
        expected.extend_from_slice(&1u64.to_le_bytes());
        expected.extend_from_slice(&8u32.to_le_bytes());
        expected.extend_from_slice(b"external");
        expected.push(1);
        expected.extend_from_slice(&5u32.to_le_bytes());
        expected.extend_from_slice(b"title");
        expected.push(0);
        let json = "{\"a\":\"\u{00e9}\\n\\\"\\\\\",\"z\":0}";
        expected.extend_from_slice(&(json.len() as u32).to_le_bytes());
        expected.extend_from_slice(json.as_bytes());
        expected.extend_from_slice(&0u64.to_le_bytes());
        expected.extend_from_slice(&1u64.to_le_bytes());
        let RecordPayload::DocumentPut(decoded) =
            assert_payload_golden(&record, DOCUMENT_PUT, &expected, 81)
        else {
            panic!()
        };
        assert_eq!(decoded.document_id, "doc_0000000000000002");
        assert_eq!(decoded.collection_id, "col_0000000000000001");
        assert_eq!(decoded.external_id, "external");
        assert_eq!(decoded.title.as_deref(), Some("title"));
        assert!(decoded.source_uri.is_none());
        assert_eq!(decoded.metadata.to_compact_string(), json);
        assert_eq!(decoded.created_at, "1970-01-01T00:00:00Z");
        assert_eq!(decoded.updated_at, "1970-01-01T00:00:00.001Z");
        assert_planned_encoding(std::slice::from_ref(&record));
        for fragment in 1..=100 {
            let mut output = FragmentedWriter::new(fragment, usize::MAX);
            encode_into(&record, &mut output).unwrap();
            assert_eq!(output.bytes, expected);
        }
        for fail_after in 0..expected.len() {
            let mut output = FragmentedWriter::new(7, fail_after);
            let error = encode_into(&record, &mut output).unwrap_err();
            assert_eq!(error.code(), ErrorCode::IoError);
            assert!(error.message().contains("injected payload write failure"));
            assert_eq!(output.bytes, expected[..fail_after]);
        }
    }

    #[test]
    fn vector_encoding_uses_fixed_blocks_and_preserves_all_finite_f32_bits() {
        let dimensions = [1, 2, 255, 256, 257, 767, 768, 769, 1023, 1024, 1025, 4097];
        for seed in 0..100usize {
            let dim = dimensions[seed % dimensions.len()];
            let mut state = seed as u64 + 1;
            let vector = (0..dim)
                .map(|index| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    f32::from_bits(match index {
                        0 => 0x8000_0000,
                        1 => 1,
                        2 => 0x7f7f_ffff,
                        _ => state as u32 & 0xff7f_ffff,
                    })
                })
                .collect::<Vec<_>>();
            let mut expected = Vec::new();
            for id in [1u64, 2, 3] {
                expected.extend_from_slice(&id.to_le_bytes());
            }
            expected.extend_from_slice(&(dim as u32).to_le_bytes());
            expected.extend_from_slice(&[1, 1]);
            expected.extend_from_slice(&1u32.to_le_bytes());
            expected.push(b'm');
            expected.extend_from_slice(&1u64.to_le_bytes());
            let header_len = expected.len();
            for value in &vector {
                expected.extend_from_slice(&value.to_bits().to_le_bytes());
            }
            let record = RecordPayload::EmbeddingPut(Embedding {
                embedding_id: "emb_1".into(),
                chunk_id: "chk_2".into(),
                collection_id: "col_3".into(),
                chunk_external_id: "external".into(),
                model_name: "m".into(),
                dim,
                vector,
                created_at: "1970-01-01T00:00:00.001Z".into(),
            });
            assert_eq!(encode(&record).unwrap(), expected);
            assert_planned_encoding(std::slice::from_ref(&record));
            assert_eq!(encoded_len(&record).unwrap(), expected.len() as u64);
            for fragment in [1, 3, 17, 4096] {
                let mut output = FragmentedWriter::new(fragment, usize::MAX);
                encode_into(&record, &mut output).unwrap();
                assert_eq!(output.bytes, expected);
                assert!(output.max_request <= 4096);
                if fragment == 4096 {
                    assert!(output.calls < 20 + dim.div_ceil(1024) * 2);
                }
            }
            for fail_after in [
                0,
                header_len - 1,
                header_len,
                header_len + 4095,
                header_len + 4096,
                expected.len() - 1,
            ] {
                if fail_after >= expected.len() {
                    continue;
                }
                let mut output = FragmentedWriter::new(127, fail_after);
                assert_eq!(
                    encode_into(&record, &mut output).unwrap_err().code(),
                    ErrorCode::IoError
                );
                assert_eq!(output.bytes, expected[..fail_after]);
            }
        }
    }

    #[test]
    fn metadata_encoding_needs_no_canonical_string_allocation() {
        let context = JsonOperationContext::standalone();
        let value = object_from_pairs([("payload", JsonValue::String("x".repeat(256 * 1024)))]);
        let retained = value
            .reserve_retained_in(JsonInputKind::DocumentMetadata, &context)
            .unwrap()
            .reserved_bytes();
        let mut busy = context.reservation();
        busy.grow(
            crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES - retained - 8192,
            JsonInputKind::JsonlRecord,
        )
        .unwrap();
        let mut output = PayloadDigest::default();
        let mut writer = Writer {
            output: Some(&mut output),
            len: 0,
            context: context.clone(),
            planned_json: None,
        };
        writer
            .json(&value, JsonInputKind::DocumentMetadata, None)
            .unwrap();
        let canonical = value.to_compact_string();
        let mut expected_crc = Crc32c::default();
        expected_crc.update(&(canonical.len() as u32).to_le_bytes());
        expected_crc.update(canonical.as_bytes());
        assert_eq!(output.len, canonical.len() as u64 + 4);
        assert_eq!(output.crc.finish(), expected_crc.finish());
        busy.grow(retained + 8192, JsonInputKind::JsonlRecord)
            .unwrap();
    }

    #[test]
    fn retained_encoding_reuses_metadata_credit_and_obeys_shared_headroom() {
        use crate::json::{JsonProcessingPolicyV1, WorkLedger};
        for seed in 0..200 {
            let stored = seed >= 100;
            let ledger = std::sync::Arc::new(WorkLedger::default());
            let context = JsonOperationContext::new(ledger.clone());
            let other = JsonOperationContext::new(ledger);
            let mut metadata = JsonValue::String("x".repeat(32768 + seed));
            for _ in 0..seed % 20 {
                metadata = JsonValue::Array(vec![metadata]);
            }
            let record = RecordPayload::DocumentPut(Document {
                document_id: "doc_1".into(),
                collection_id: "col_1".into(),
                external_id: "document".into(),
                title: None,
                source_uri: None,
                metadata: object_from_pairs([("payload", metadata)]),
                created_at: "1970-01-01T00:00:00Z".into(),
                updated_at: "1970-01-01T00:00:00Z".into(),
            });
            let RecordPayload::DocumentPut(document) = record else {
                unreachable!()
            };
            let record = if seed % 2 == 0 {
                RecordPayload::DocumentPut(document)
            } else {
                RecordPayload::ChunkPut(Chunk {
                    chunk_id: "chk_1".into(),
                    document_id: document.document_id,
                    collection_id: document.collection_id,
                    chunk_external_id: "chunk".into(),
                    text: "text".into(),
                    text_hash: digest_hex(sha256(b"text")),
                    ordinal: 0,
                    token_count: 1,
                    metadata: document.metadata,
                    embedding_id: None,
                    source_span: None,
                    created_at: document.created_at,
                })
            };
            let (metadata, kind) = match &record {
                RecordPayload::DocumentPut(document) => {
                    (&document.metadata, JsonInputKind::DocumentMetadata)
                }
                RecordPayload::ChunkPut(chunk) => (&chunk.metadata, JsonInputKind::ChunkMetadata),
                _ => unreachable!(),
            };
            assert_planned_encoding(std::slice::from_ref(&record));
            let mut work = metadata.guard_in(kind, &context).unwrap().0;
            let retained_bytes = work.reserved_bytes();
            let mut busy = other.reservation();
            busy.grow(
                JsonProcessingPolicyV1::WORK_BUDGET_BYTES - retained_bytes - 8192,
                JsonInputKind::JsonlRecord,
            )
            .unwrap();
            let mut actual = Vec::new();
            if stored {
                encode_stored_into(&record, &mut actual, &context, &mut work).unwrap();
            } else {
                encode_retained_into(&record, &mut actual, &context, &mut work).unwrap();
            }
            assert_eq!(actual, encode(&record).unwrap());
            assert_eq!(work.reserved_bytes(), retained_bytes);
            busy.grow(8192, JsonInputKind::JsonlRecord).unwrap();
            let error = if stored {
                encode_stored_into(&record, &mut std::io::sink(), &context, &mut work)
            } else {
                encode_retained_into(&record, &mut std::io::sink(), &context, &mut work)
            }
            .unwrap_err();
            assert_eq!(
                error.code(),
                if stored {
                    ErrorCode::RecoveryError
                } else {
                    ErrorCode::InvalidJsonlSchema
                }
            );
            let expected_kind = if stored {
                "stored_json"
            } else if seed % 2 == 0 {
                "document_metadata"
            } else {
                "chunk_metadata"
            };
            assert!(error
                .detail()
                .iter()
                .any(|(key, value)| key == "json_input_kind" && value == expected_kind));
            assert!(error
                .detail()
                .iter()
                .any(|(key, value)| key == "limit" && value == &retained_bytes.to_string()));
            assert_eq!(work.reserved_bytes(), retained_bytes);
            drop(work);
            drop(busy);
            context
                .reservation()
                .grow(
                    JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
                    JsonInputKind::JsonlRecord,
                )
                .unwrap();
        }
    }

    #[test]
    fn generated_collection_json_reserves_its_copies_in_the_callers_context() {
        let context = JsonOperationContext::standalone();
        let record = RecordPayload::CollectionCreate(CollectionInfo {
            collection_id: "col_1".into(),
            name: "notes".into(),
            embedding_dim: 2,
            distance_metric: crate::model::DistanceMetric::Cosine,
            embedding_model: None,
            vectors_normalized: false,
            created_at: "1970-01-01T00:00:00Z".into(),
            tokenizer: crate::TokenizerConfig::default(),
            vector_index: VectorIndexConfig::Exact,
        });
        let mut busy = context.reservation();
        busy.grow(
            crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES - 128,
            JsonInputKind::JsonlRecord,
        )
        .unwrap();
        let error = encode_retained_into(
            &record,
            &mut std::io::sink(),
            &context,
            &mut context.reservation(),
        )
        .unwrap_err();
        assert_eq!(error.code(), ErrorCode::InvalidJsonlSchema);
        assert!(error
            .detail()
            .iter()
            .any(|(key, value)| key == "json_input_kind" && value == "collection_config"));
        busy.grow(128, JsonInputKind::JsonlRecord).unwrap();
        drop(busy);
        let mut actual = Vec::new();
        encode_retained_into(&record, &mut actual, &context, &mut context.reservation()).unwrap();
        assert_eq!(actual, encode(&record).unwrap());
        context
            .reservation()
            .grow(
                crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
                JsonInputKind::JsonlRecord,
            )
            .unwrap();
    }

    #[test]
    fn sizing_uses_the_same_layout_for_variable_and_optional_record_fields() {
        for seed in 1..=100 {
            let timestamp = "1970-01-01T00:00:00.001Z".to_string();
            let mut records = vec![
                RecordPayload::CollectionCreate(CollectionInfo {
                    collection_id: format!("col_{seed}"),
                    name: "x".repeat(seed),
                    embedding_dim: seed,
                    distance_metric: DistanceMetric::Cosine,
                    embedding_model: (seed % 2 == 0).then(|| "m".repeat(seed)),
                    vectors_normalized: seed % 3 == 0,
                    created_at: timestamp.clone(),
                    tokenizer: crate::TokenizerConfig::default(),
                    vector_index: if seed % 2 == 0 {
                        VectorIndexConfig::Exact
                    } else {
                        VectorIndexConfig::Ann {
                            provider: "test\"\\\n\u{4e2d}".into(),
                            params: object_from_pairs([(
                                "values",
                                JsonValue::Array(vec![JsonValue::Number(seed as f64); seed]),
                            )]),
                        }
                    },
                }),
                RecordPayload::CollectionDrop {
                    collection_id: format!("col_{seed}"),
                    name: "name".repeat(seed),
                    dropped_at: timestamp.clone(),
                    reason: (seed % 2 == 0).then(|| "reason".into()),
                },
                RecordPayload::Tombstone(Tombstone {
                    target_type: "chunk".into(),
                    target_id: format!("chk_{seed}"),
                    deleted_at: timestamp.clone(),
                    reason: (seed % 2 == 0).then(|| "reason".repeat(seed)),
                }),
                RecordPayload::CompactMarker(CompactMarker {
                    compact_generation: seed as u64,
                    source_last_commit_sequence_no: seed as u64 + 1,
                    id_high_watermarks: BTreeMap::from([("doc_".into(), seed as u64)]),
                    records_rewritten: seed,
                    compacted_at: timestamp,
                    compacted_main_checksum: seed as u32,
                }),
            ];
            let mut another = records[0].clone();
            if let RecordPayload::CollectionCreate(value) = &mut another {
                value.collection_id = format!("col_{}", seed + 1);
                value.vector_index = VectorIndexConfig::Exact;
            }
            records.push(another);
            assert_planned_encoding(&records);
            for record in records {
                assert_eq!(
                    encoded_len(&record).unwrap(),
                    encode(&record).unwrap().len() as u64
                );
            }
        }
    }

    #[test]
    fn catalog_text_decode_is_bounded_and_validates_split_utf8_and_digest() {
        for (character, prefix) in [('\u{00e9}', 1), ('\u{20ac}', 2), ('\u{10000}', 3)] {
            let text = format!(
                "{}{character}{}",
                "x".repeat(64 * 1024 - prefix),
                "abc".repeat(70_000)
            );
            let chunk = Chunk {
                chunk_id: "chk_1".into(),
                document_id: "doc_1".into(),
                collection_id: "col_1".into(),
                chunk_external_id: "x".into(),
                text_hash: digest_hex(sha256(text.as_bytes())),
                text,
                ordinal: u32::MAX as usize,
                token_count: 42,
                metadata: object_from_pairs([("source", JsonValue::String("keep".into()))]),
                source_span: Some(SourceSpan {
                    start: Some(u64::MAX - 1),
                    end: Some(u64::MAX),
                    line_start: Some(u32::MAX),
                    line_end: Some(u32::MAX),
                }),
                embedding_id: None,
                created_at: "1970-01-01T00:00:00Z".into(),
            };
            let bytes = encode(&RecordPayload::ChunkPut(chunk.clone())).unwrap();
            assert_eq!(
                encoded_len(&RecordPayload::ChunkPut(chunk.clone())).unwrap(),
                bytes.len() as u64
            );
            let RecordPayload::ChunkPut(decoded) = summary(CHUNK_PUT, &bytes).unwrap() else {
                panic!()
            };
            assert!(decoded.text.is_empty());
            assert_eq!(decoded.text_hash, chunk.text_hash);
            assert_eq!(decoded.metadata, chunk.metadata);
            assert_eq!(decoded.source_span, chunk.source_span);
            assert_eq!(decoded.ordinal, chunk.ordinal);
            assert_eq!(decoded.token_count, chunk.token_count);
            let RecordPayload::ChunkPut(full) = decode(CHUNK_PUT, &bytes).unwrap() else {
                panic!()
            };
            assert_eq!(full.text, chunk.text);
            let mut bad = bytes.clone();
            bad[33 + 64 * 1024 - prefix] = 0xff;
            assert!(summary(CHUNK_PUT, &bad).is_err());
            let mut bad = bytes.clone();
            bad[33 + chunk.text.len()] ^= 1;
            assert!(summary(CHUNK_PUT, &bad).is_err());
            assert!(decode(CHUNK_PUT, &bad).is_err());
            let mut extra = bytes.clone();
            extra.push(0);
            assert!(summary(CHUNK_PUT, &extra).is_err());
        }
        let mut source = &[0xf0, 0x90, 0x80][..];
        let mut bytes = 3u32.to_le_bytes().to_vec();
        bytes.extend_from_slice(source);
        source = &bytes;
        assert!(Reader::new(
            source,
            bytes.len() as u64,
            JsonOperationContext::standalone()
        )
        .text(false)
        .is_err());
    }

    #[test]
    fn catalog_vector_decode_drops_values_but_checks_every_block() {
        let embedding = Embedding {
            embedding_id: "emb_1".into(),
            chunk_id: "chk_1".into(),
            collection_id: "col_1".into(),
            chunk_external_id: "x".into(),
            dim: 70_001,
            model_name: "m".into(),
            created_at: "1970-01-01T00:00:00Z".into(),
            vector: vec![0.125; 70_001],
        };
        let bytes = encode(&RecordPayload::EmbeddingPut(embedding.clone())).unwrap();
        assert_eq!(
            encoded_len(&RecordPayload::EmbeddingPut(embedding.clone())).unwrap(),
            bytes.len() as u64
        );
        let RecordPayload::EmbeddingPut(decoded) = summary(EMBEDDING_PUT, &bytes).unwrap() else {
            panic!()
        };
        assert!(decoded.vector.is_empty());
        assert_eq!(decoded.dim, embedding.dim);
        assert_eq!(decoded.model_name, embedding.model_name);
        let RecordPayload::EmbeddingPut(full) = decode(EMBEDDING_PUT, &bytes).unwrap() else {
            panic!()
        };
        assert_eq!(full.vector, embedding.vector);
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut bad = bytes.clone();
            let end = bad.len();
            bad[end - 4..].copy_from_slice(&value.to_le_bytes());
            assert!(summary(EMBEDDING_PUT, &bad).is_err());
        }
        let mut bad = bytes;
        bad[24..28].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(summary(EMBEDDING_PUT, &bad).is_err());
    }

    #[test]
    fn fixed_commit_encoding_matches_wire_and_needs_no_additional_workspace() {
        assert_eq!(COMMIT_PAYLOAD_BYTES, 44);
        for seed in 0..100u64 {
            let millis = 253_402_300_799_999u64 / 100 * seed;
            let mut commit = CommitPayload {
                batch_id: u64::MAX - seed,
                first_sequence_no: seed + 1,
                last_sequence_no: seed + 17,
                record_count: (seed + 3) as usize,
                batch_checksum: u64::from(u32::MAX) - seed,
                committed_at: time::format_unix_millis(millis),
            };
            let mut expected = [0u8; 44];
            for (part, value) in expected[..32].chunks_exact_mut(8).zip([
                commit.batch_id,
                commit.first_sequence_no,
                commit.last_sequence_no,
                commit.record_count as u64,
            ]) {
                part.copy_from_slice(&value.to_le_bytes());
            }
            expected[32..36].copy_from_slice(&(commit.batch_checksum as u32).to_le_bytes());
            expected[36..44].copy_from_slice(&millis.to_le_bytes());
            let context = JsonOperationContext::standalone();
            let limit = crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES;
            let mut held = context.workspace_reservation();
            held.grow_to(limit, limit, "operation_scratch").unwrap();
            let mut actual = [0u8; 44];
            let mut remaining = actual.as_mut_slice();
            encode_commit_into(&commit, &mut remaining, &context).unwrap();
            assert!(remaining.is_empty());
            assert_eq!(actual, expected);
            assert_eq!(
                encode(&RecordPayload::Commit(commit.clone())).unwrap(),
                expected
            );
            for length in 0..44 {
                let mut short = &mut actual[..length];
                assert_eq!(
                    encode_commit_into(&commit, &mut short, &context)
                        .unwrap_err()
                        .code(),
                    ErrorCode::IoError
                );
            }
            commit.batch_checksum = u64::MAX;
            assert_eq!(
                encode_commit_into(&commit, &mut actual.as_mut_slice(), &context)
                    .unwrap_err()
                    .code(),
                ErrorCode::FileFormatError
            );
            commit.batch_checksum = 0;
            commit.committed_at = time::format_unix_millis(u64::MAX);
            assert_eq!(
                encode_commit_into(&commit, &mut actual.as_mut_slice(), &context)
                    .unwrap_err()
                    .code(),
                ErrorCode::FileFormatError
            );
        }
    }

    #[test]
    fn commit_payload_independent_byte_golden() {
        let golden = [
            1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0,
            0, 0, 0, 0x12, 0x34, 0x56, 0x78, 0xe8, 3, 0, 0, 0, 0, 0, 0,
        ];
        let c = RecordPayload::Commit(CommitPayload {
            batch_id: 1,
            first_sequence_no: 2,
            last_sequence_no: 3,
            record_count: 2,
            batch_checksum: 0x78563412,
            committed_at: "1970-01-01T00:00:01Z".into(),
        });
        assert_eq!(encode(&c).unwrap(), golden);
        assert_eq!(encoded_len(&c).unwrap(), golden.len() as u64);
        let RecordPayload::Commit(c) = decode(COMMIT, &golden).unwrap() else {
            panic!()
        };
        assert_eq!(c.batch_checksum, 0x78563412);
        for len in 0..golden.len() {
            assert!(decode(COMMIT, &golden[..len]).is_err());
        }
        let mut extra = golden.to_vec();
        extra.push(0);
        assert!(decode(COMMIT, &extra).is_err());
        let max_count = RecordPayload::Commit(CommitPayload {
            batch_id: 1,
            first_sequence_no: 1,
            last_sequence_no: u64::MAX - 1,
            record_count: usize::MAX - 1,
            batch_checksum: 0,
            committed_at: "1970-01-01T00:00:01Z".into(),
        });
        let golden = golden_bytes(
            "0100000000000000 0100000000000000 feffffffffffffff
             feffffffffffffff 00000000 e803000000000000",
        );
        let RecordPayload::Commit(decoded) = assert_payload_golden(&max_count, COMMIT, &golden, 44)
        else {
            panic!()
        };
        assert_eq!(decoded.first_sequence_no, 1);
        assert_eq!(decoded.last_sequence_no, u64::MAX - 1);
        assert_eq!(decoded.record_count, usize::MAX - 1);
    }

    #[test]
    fn embedding_uses_binary_f32_and_rejects_nonfinite_dtype_and_lengths() {
        let e = RecordPayload::EmbeddingPut(Embedding {
            embedding_id: "emb_1".into(),
            chunk_id: "chk_2".into(),
            collection_id: "col_3".into(),
            chunk_external_id: "not-persisted".into(),
            model_name: "m".into(),
            dim: 2,
            vector: vec![1.0, -0.5],
            created_at: "1970-01-01T00:00:00.001Z".into(),
        });
        let bytes = golden_bytes(
            "0100000000000000 0200000000000000 0300000000000000
             02000000 01 01 010000006d 0100000000000000 0000803f000000bf",
        );
        let RecordPayload::EmbeddingPut(e) = assert_payload_golden(&e, EMBEDDING_PUT, &bytes, 51)
        else {
            panic!()
        };
        assert_eq!(e.embedding_id, "emb_0000000000000001");
        assert_eq!(e.chunk_id, "chk_0000000000000002");
        assert_eq!(e.collection_id, "col_0000000000000003");
        assert!(e.chunk_external_id.is_empty());
        assert_eq!(e.model_name, "m");
        assert_eq!(e.dim, 2);
        assert_eq!(e.created_at, "1970-01-01T00:00:00.001Z");
        assert_eq!(e.vector, vec![1.0, -0.5]);
        for (offset, value) in [(28, 2), (29, 2)] {
            let mut bad = bytes.clone();
            bad[offset] = value;
            assert!(decode(EMBEDDING_PUT, &bad).is_err());
        }
        let mut bad = bytes.clone();
        let end = bad.len();
        bad[end - 4..].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(decode(EMBEDDING_PUT, &bad).is_err());
    }

    #[test]
    fn streamed_metadata_respects_its_frame_and_never_accepts_truncation() {
        let value = object_from_pairs([("text", JsonValue::String("é\nASCII".repeat(2000)))]);
        let mut bytes = Vec::new();
        let mut w = Writer {
            output: Some(&mut bytes),
            len: 0,
            context: JsonOperationContext::standalone(),
            planned_json: None,
        };
        w.json(&value, JsonInputKind::ChunkMetadata, None).unwrap();
        let frame_len = bytes.len();
        bytes.extend_from_slice(&[0xa5; 17]);
        for fragment in [1, 7, 8192] {
            let mut source = CappedReader {
                bytes: &bytes,
                fragment,
                max_request: 0,
            };
            let mut reader = Reader::new(
                &mut source,
                bytes.len() as u64,
                JsonOperationContext::standalone(),
            );
            assert_eq!(reader.json().unwrap(), value);
            assert_eq!(reader.remaining, 17);
            assert_eq!(source.bytes, &[0xa5; 17]);
            assert!(source.max_request <= 8192);
        }
        for end in [0, 1, 3, 4, 5, frame_len - 1] {
            assert!(Reader::new(
                &bytes[..end],
                frame_len as u64,
                JsonOperationContext::standalone()
            )
            .json()
            .is_err());
        }
        for length in [0, 1, (frame_len - 5) as u32, frame_len as u32, u32::MAX] {
            let mut bad = bytes[..frame_len].to_vec();
            bad[..4].copy_from_slice(&length.to_le_bytes());
            assert!(Reader::new(
                &bad[..],
                bad.len() as u64,
                JsonOperationContext::standalone()
            )
            .json()
            .is_err());
        }
    }

    #[test]
    fn canonical_metadata_and_optional_tags_are_strict() {
        let mut bytes = Vec::new();
        let mut w = Writer {
            output: Some(&mut bytes),
            len: 0,
            context: JsonOperationContext::standalone(),
            planned_json: None,
        };
        w.string("{\"b\":1,\"a\":2}").unwrap();
        assert!(Reader::new(
            &w.output.as_ref().unwrap()[..],
            w.output.as_ref().unwrap().len() as u64,
            JsonOperationContext::standalone()
        )
        .json()
        .is_err());
        w.output.as_mut().unwrap().clear();
        w.json(
            &object_from_pairs([("b", JsonValue::Number(1.0)), ("a", JsonValue::Number(2.0))]),
            JsonInputKind::DocumentMetadata,
            None,
        )
        .unwrap();
        assert!(Reader::new(
            &w.output.as_ref().unwrap()[..],
            w.output.as_ref().unwrap().len() as u64,
            JsonOperationContext::standalone()
        )
        .json()
        .is_ok());
        assert!(Reader::new(&[2][..], 1, JsonOperationContext::standalone())
            .optional_u64()
            .is_err());
        assert!(Reader::new(&[2][..], 1, JsonOperationContext::standalone())
            .optional_string()
            .is_err());
    }
}
