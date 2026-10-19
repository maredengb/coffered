use super::*;
#[cfg(test)]
use std::borrow::Cow;
use std::io::BufWriter;

const BUFFER_BYTES: usize = 64 * 1024;

#[cfg(test)]
mod golden;

pub(super) mod text_spool;

// The caller removes any old suffix first. Publish the complete marker after all backpatches.
#[cfg(test)]
pub(super) fn write_checkpoint<W: Write + Seek>(
    output: &mut W,
    checkpoint: &IndexCheckpointPayload,
    offset: u64,
) -> Result<(u64, IndexSegmentSource)> {
    if checkpoint.metadata.text_segment_count != checkpoint.text_segments.len()
        || checkpoint.metadata.vector_segment_count != checkpoint.vector_segments.len()
    {
        return Err(Error::new(
            ErrorCode::FileFormatError,
            "index checkpoint segment counts do not match segment payloads",
        ));
    }
    let segments = checkpoint
        .text_segments
        .iter()
        .map(|segment| Ok(Segment::Text(Cow::Borrowed(segment))))
        .chain(
            checkpoint
                .vector_segments
                .iter()
                .map(|segment| Ok(Segment::Vector(Cow::Borrowed(segment)))),
        );
    write_checkpoint_segments(output, &checkpoint.metadata, segments, offset, |_, _| {})
}

#[cfg(test)]
pub(super) fn write_checkpoint_segments<'a, W: Write + Seek>(
    output: &mut W,
    checkpoint: &IndexCheckpointMetadata,
    segments: impl IntoIterator<Item = Result<Segment<'a>>>,
    offset: u64,
    completed: impl FnMut(Segment<'a>, u64),
) -> Result<(u64, IndexSegmentSource)> {
    let directory =
        reserve_directory(checkpoint, &crate::json::JsonOperationContext::standalone())?;
    write_checkpoint_segments_in(
        output, checkpoint, segments, offset, completed, directory, true,
    )
}

pub(super) fn reserve_directory(
    checkpoint: &IndexCheckpointMetadata,
    context: &crate::json::JsonOperationContext,
) -> Result<IndexDirectory> {
    let count = checked_writer_add(
        checked_segment_record_count(checkpoint)?,
        1,
        "index checkpoint record count overflow",
    )?;
    let count = usize::try_from(count).map_err(|_| {
        Error::new(
            ErrorCode::FileFormatError,
            "index extent count exceeds address space",
        )
    })?;
    IndexDirectory::with_capacity(count, context, "operation_scratch")
}

pub(super) fn write_checkpoint_segments_in<'a, W: Write + Seek>(
    output: &mut W,
    checkpoint: &IndexCheckpointMetadata,
    segments: impl IntoIterator<Item = Result<Segment<'a>>>,
    offset: u64,
    mut completed: impl FnMut(Segment<'a>, u64),
    mut extents: IndexDirectory,
    allow_optional_memory: bool,
) -> Result<(u64, IndexSegmentSource)> {
    let count = checked_writer_add(
        checked_segment_record_count(checkpoint)?,
        1,
        "index checkpoint record count overflow",
    )?;
    let table_len = count
        .checked_mul(INDEX_EXTENT_ENTRY_LEN as u64)
        .ok_or_else(|| {
            Error::new(
                ErrorCode::FileFormatError,
                "index extent table length overflow",
            )
        })?;
    output.seek(SeekFrom::Start(offset))?;
    output.write_all(&header(checkpoint, offset, count, 0, table_len, 0, 0)?)?;
    #[cfg(test)]
    tests::crash_boundary("index_suffix_partial")?;
    output.write_all(&[0; INDEX_RECORD_HEADER_LEN + INDEX_COMMIT_METADATA_LEN])?;
    extents.push(IndexExtentEntry {
        index_record_id: 1,
        record_type: INDEX_COMMIT_METADATA,
        stream_id: 0,
        suffix_relative_offset: INDEX_CHECKPOINT_HEADER_LEN as u64,
        length: (INDEX_RECORD_HEADER_LEN + INDEX_COMMIT_METADATA_LEN) as u64,
        checkpoint_generation: offset,
        extent_record_checksum: 0,
    });
    let mut text_count = 0usize;
    let mut vector_count = 0usize;
    let mut next_offset = output.stream_position()?;
    for (index, segment) in segments.into_iter().enumerate() {
        let mut segment = segment?;
        match &segment {
            #[cfg(test)]
            Segment::Text(_) => text_count += 1,
            #[cfg(test)]
            Segment::TextRecords(_) => text_count += 1,
            Segment::StagedText { .. } => text_count += 1,
            #[cfg(test)]
            Segment::Vector(_) => vector_count += 1,
            Segment::BorrowedCachedVector(_)
            | Segment::CachedVector(_)
            | Segment::VectorRecords(_) => vector_count += 1,
        }
        if text_count > checkpoint.text_segment_count
            || vector_count > checkpoint.vector_segment_count
        {
            return Err(Error::new(
                ErrorCode::FileFormatError,
                "index checkpoint has more segments than declared",
            ));
        }
        let record_id = index as u64 + 2;
        // Producing a segment may read through a clone of the same file handle.
        let start = next_offset;
        output.seek(SeekFrom::Start(start))?;
        output.write_all(&[0; INDEX_RECORD_HEADER_LEN])?;
        let payload =
            with_write_buffer(output, extents.context(), allow_optional_memory, |buffer| {
                let mut writer = DigestWriter::new(buffer);
                segment.encode(&mut writer)?;
                Ok(writer.digest)
            })?;
        let length = checked_writer_add(
            INDEX_RECORD_HEADER_LEN as u64,
            payload.len,
            "index record length overflow",
        )?;
        let record_header = record_header(segment.record_type(), offset, record_id, &payload)?;
        output.seek(SeekFrom::Start(start))?;
        output.write_all(&record_header)?;
        let mut record_crc = Crc32c::default();
        record_crc.update(&record_header);
        record_crc.append_crc(payload.crc.finish(), payload.len);
        extents.push(IndexExtentEntry {
            index_record_id: record_id,
            record_type: segment.record_type(),
            stream_id: segment.stream_id()?,
            suffix_relative_offset: start - offset,
            length,
            checkpoint_generation: offset,
            extent_record_checksum: record_crc.finish(),
        });
        next_offset = checked_writer_add(start, length, "index record boundary overflow")?;
        completed(segment, length);
    }
    if text_count != checkpoint.text_segment_count
        || vector_count != checkpoint.vector_segment_count
    {
        return Err(Error::new(
            ErrorCode::FileFormatError,
            "index checkpoint has fewer segments than declared",
        ));
    }
    let records_end = next_offset;
    output.seek(SeekFrom::Start(records_end))?;
    let payload_len = records_end - offset - INDEX_CHECKPOINT_HEADER_LEN as u64;
    let addressable_payload_len = usize::try_from(payload_len).map_err(|_| {
        Error::new(
            ErrorCode::FileFormatError,
            "index payload length exceeds address space",
        )
    })?;
    let mut referenced = index_commit_hasher(
        checkpoint.covered_last_batch_id,
        checkpoint.covered_last_commit_sequence_no,
        offset,
    );
    let mut segment_table = Crc32c::default();
    // Combine write-time digests in wire order without reading the new suffix back.
    for extent in &extents.as_slice()[1..] {
        let bytes = encode_extent_entry(extent);
        segment_table.update(&bytes);
        referenced.update(&bytes);
        referenced.append_crc(extent.extent_record_checksum, extent.length);
    }
    let metadata = build_index_commit_metadata_payload(
        checkpoint,
        count - 1,
        INDEX_EXTENT_ENTRY_LEN as u64,
        table_len - INDEX_EXTENT_ENTRY_LEN as u64,
        segment_table.finish(),
        referenced.finish(),
    );
    let metadata_digest = PayloadDigest {
        len: metadata.len() as u64,
        crc: {
            let mut crc = Crc32c::default();
            crc.update(&metadata);
            crc
        },
    };
    let metadata_header = record_header(INDEX_COMMIT_METADATA, offset, 1, &metadata_digest)?;
    output.seek(SeekFrom::Start(offset + INDEX_CHECKPOINT_HEADER_LEN as u64))?;
    let mut metadata_record = DigestWriter::new(&mut *output);
    metadata_record.write_all(&metadata_header)?;
    metadata_record.write_all(&metadata)?;
    extents.as_mut_slice()[0].extent_record_checksum = metadata_record.digest.crc.finish();
    let mut payload_crc = metadata_record.digest.crc;
    for extent in &extents.as_slice()[1..] {
        payload_crc.append_crc(extent.extent_record_checksum, extent.length);
    }
    output.seek(SeekFrom::Start(records_end))?;
    let mut table_crc = Crc32c::default();
    with_write_buffer(output, extents.context(), allow_optional_memory, |writer| {
        for extent in extents.as_slice() {
            let bytes = encode_extent_entry(extent);
            table_crc.update(&bytes);
            writer.write_all(&bytes)?;
        }
        Ok(())
    })?;
    let trailer_start = output.stream_position()?;
    output.seek(SeekFrom::Start(offset))?;
    output.write_all(&header(
        checkpoint,
        offset,
        count,
        payload_len,
        table_len,
        table_crc.finish(),
        payload_crc.finish(),
    )?)?;
    output.seek(SeekFrom::Start(trailer_start))?;
    let mut trailer = [0; INDEX_CHECKPOINT_TRAILER_LEN];
    trailer[..4].copy_from_slice(INDEX_COMPLETE_MARKER);
    trailer[4..6].copy_from_slice(&(INDEX_CHECKPOINT_TRAILER_LEN as u16).to_le_bytes());
    let trailer_checksum = checksum32(&trailer[..6]);
    trailer[6..].copy_from_slice(&trailer_checksum.to_le_bytes());
    let source = IndexSegmentSource::new(
        offset,
        addressable_payload_len,
        offset,
        checkpoint.covered_last_commit_sequence_no,
        extents,
    )?;
    output.write_all(&trailer)?;
    Ok((payload_len, source))
}

fn with_write_buffer<W: Write, T>(
    output: &mut W,
    context: &crate::json::JsonOperationContext,
    allow_optional_memory: bool,
    encode: impl FnOnce(&mut BufWriter<&mut W>) -> Result<T>,
) -> Result<T> {
    let mut work = context.workspace_reservation();
    let capacity = work.grow_to(
        0,
        if allow_optional_memory {
            BUFFER_BYTES
        } else {
            0
        },
        "operation_scratch",
    )?;
    let mut buffer = BufWriter::with_capacity(capacity, output);
    let result = encode(&mut buffer).and_then(|value| {
        buffer.flush()?;
        Ok(value)
    });
    // Do not retry buffered IO from Drop after an encode/flush failure.
    drop(buffer.into_parts().1);
    result
}

fn header(
    checkpoint: &IndexCheckpointMetadata,
    offset: u64,
    count: u64,
    payload_len: u64,
    table_len: u64,
    table_crc: u32,
    payload_crc: u32,
) -> Result<[u8; INDEX_CHECKPOINT_HEADER_LEN]> {
    let mut bytes = [0; INDEX_CHECKPOINT_HEADER_LEN];
    let mut remaining = &mut bytes[..];
    remaining.write_all(INDEX_CHECKPOINT_MAGIC)?;
    remaining.write_all(&INDEX_CHECKPOINT_VERSION.to_le_bytes())?;
    remaining.write_all(&(INDEX_CHECKPOINT_HEADER_LEN as u16).to_le_bytes())?;
    for value in [
        offset,
        offset,
        offset,
        checkpoint.covered_last_batch_id,
        checkpoint.covered_last_commit_sequence_no,
    ] {
        remaining.write_all(&value.to_le_bytes())?;
    }
    remaining.write_all(&[INDEX_COVERAGE_FULL_VISIBLE_SNAPSHOT])?;
    remaining.write_all(&checkpoint.coverage_hash.to_le_bytes())?;
    for value in [
        count,
        count,
        checked_writer_add(
            INDEX_CHECKPOINT_HEADER_LEN as u64,
            payload_len,
            "index payload length overflow",
        )?,
        table_len,
    ] {
        remaining.write_all(&value.to_le_bytes())?;
    }
    remaining.write_all(&table_crc.to_le_bytes())?;
    remaining.write_all(&payload_len.to_le_bytes())?;
    remaining.write_all(&payload_crc.to_le_bytes())?;
    debug_assert!(remaining.is_empty());
    Ok(bytes)
}

fn record_header(
    kind: u16,
    generation: u64,
    id: u64,
    payload: &PayloadDigest,
) -> Result<[u8; INDEX_RECORD_HEADER_LEN]> {
    let mut bytes = [0; INDEX_RECORD_HEADER_LEN];
    let mut remaining = &mut bytes[..];
    remaining.write_all(INDEX_RECORD_MAGIC)?;
    remaining.write_all(&kind.to_le_bytes())?;
    remaining.write_all(&1u16.to_le_bytes())?;
    remaining.write_all(&(INDEX_RECORD_HEADER_LEN as u16).to_le_bytes())?;
    for value in [
        checked_writer_add(
            INDEX_RECORD_HEADER_LEN as u64,
            payload.len,
            "index record length overflow",
        )?,
        generation,
        id,
        payload.len,
    ] {
        remaining.write_all(&value.to_le_bytes())?;
    }
    remaining.write_all(&payload.crc.finish().to_le_bytes())?;
    debug_assert_eq!(remaining.len(), 4);
    let header_checksum = checksum32(&bytes[..INDEX_RECORD_HEADER_LEN - 4]);
    bytes[INDEX_RECORD_HEADER_LEN - 4..].copy_from_slice(&header_checksum.to_le_bytes());
    Ok(bytes)
}

pub(crate) enum Segment<'a> {
    #[cfg(test)]
    Text(Cow<'a, TextIndexSegmentPayload>),
    #[cfg(test)]
    TextRecords(TextRecordSegment<'a>),
    StagedText {
        body: StagedText<'a>,
        commit: u64,
    },
    #[cfg(test)]
    Vector(Cow<'a, VectorIndexSegmentPayload>),
    BorrowedCachedVector(&'a CachedVectorSegment),
    CachedVector(CachedVectorSegment),
    VectorRecords(VectorRecordSegment<'a>),
}

impl Segment<'_> {
    fn record_type(&self) -> u16 {
        match self {
            #[cfg(test)]
            Self::Text(_) => INDEX_TEXT_SEGMENT,
            #[cfg(test)]
            Self::TextRecords(_) => INDEX_TEXT_SEGMENT,
            Self::StagedText { .. } => INDEX_TEXT_SEGMENT,
            #[cfg(test)]
            Self::Vector(_) => INDEX_VECTOR_SEGMENT,
            Self::BorrowedCachedVector(_) | Self::CachedVector(_) | Self::VectorRecords(_) => {
                INDEX_VECTOR_SEGMENT
            }
        }
    }

    fn stream_id(&self) -> Result<u64> {
        if let Self::StagedText { body, .. } = self {
            return Ok(body.collection_number());
        }
        catalog_id_number(match self {
            #[cfg(test)]
            Self::Text(segment) => &segment.collection_id,
            #[cfg(test)]
            Self::TextRecords(segment) => &segment.collection.collection_id,
            Self::StagedText { .. } => unreachable!("handled staged numeric ID"),
            #[cfg(test)]
            Self::Vector(segment) => &segment.collection_id,
            Self::BorrowedCachedVector(segment) => &segment.collection_id,
            Self::CachedVector(segment) => &segment.collection_id,
            Self::VectorRecords(segment) => &segment.collection.collection_id,
        })
    }

    fn encode(&mut self, output: &mut impl Write) -> Result<()> {
        match self {
            #[cfg(test)]
            Self::Text(segment) => write_text_segment(segment, output),
            #[cfg(test)]
            Self::TextRecords(segment) => segment.write(output),
            Self::StagedText { body, commit } => body.write(output, *commit),
            #[cfg(test)]
            Self::Vector(segment) => write_vector_segment(segment, output),
            Self::BorrowedCachedVector(segment) => write_cached_vector_segment(segment, output),
            Self::CachedVector(segment) => write_cached_vector_segment(segment, output),
            Self::VectorRecords(segment) => segment.write(output),
        }
    }
}

#[cfg(test)]
pub(crate) struct TextRecordSegment<'a> {
    catalog: &'a crate::catalog::Catalog,
    index: &'a crate::index::text::TextIndex,
    collection: &'a CollectionInfo,
    commit: u64,
    context: crate::json::JsonOperationContext,
}

#[cfg(test)]
impl<'a> TextRecordSegment<'a> {
    pub(crate) fn new(
        catalog: &'a crate::catalog::Catalog,
        index: &'a crate::index::text::TextIndex,
        collection: &'a CollectionInfo,
        commit: u64,
        context: &crate::json::JsonOperationContext,
    ) -> Self {
        Self {
            catalog,
            index,
            collection,
            commit,
            context: context.clone(),
        }
    }

    fn write(&self, output: &mut impl Write) -> Result<()> {
        let source = self
            .catalog
            .live_chunks()
            .filter(|chunk| chunk.collection_id == self.collection.collection_id);
        let mut chunks = crate::json::ReservedVec::with_capacity(
            source.clone().count(),
            &self.context,
            "operation_scratch",
        )?;
        for chunk in source {
            chunks.push(chunk);
        }
        chunks
            .as_mut_slice()
            .sort_unstable_by_key(|chunk| catalog_id_number(&chunk.chunk_id).unwrap_or(u64::MAX));
        let mut writer = DigestWriter::new(output);
        writer.write_all(&TEXT_SEGMENT_SCHEMA_VERSION.to_le_bytes())?;
        writer.write_all(&catalog_id_number(&self.collection.collection_id)?.to_le_bytes())?;
        writer.write_all(&self.commit.to_le_bytes())?;
        writer.write_all(&TEXT_INDEX_KIND_BM25.to_le_bytes())?;
        writer.write_all(&DEFAULT_EN_TOKENIZER_ID.to_le_bytes())?;
        writer.write_all(&(crate::index::text::BM25_K1 as f32).to_le_bytes())?;
        writer.write_all(&(crate::index::text::BM25_B as f32).to_le_bytes())?;
        writer.write_all(&(chunks.as_slice().len() as u64).to_le_bytes())?;
        for chunk in chunks.as_slice() {
            let count = self
                .index
                .chunk_lengths
                .get(&chunk.chunk_id)
                .copied()
                .unwrap_or(chunk.token_count.max(1));
            writer.write_all(&catalog_id_number(&chunk.chunk_id)?.to_le_bytes())?;
            writer.write_all(&catalog_id_number(&chunk.document_id)?.to_le_bytes())?;
            writer.write_all(&(count as u64).to_le_bytes())?;
        }
        drop(chunks);
        let terms = self
            .index
            .terms_in_collection(&self.collection.collection_id);
        writer.write_all(&(terms.clone().count() as u64).to_le_bytes())?;
        for (term, source) in terms {
            let mut postings = crate::json::ReservedVec::with_capacity(
                source.len(),
                &self.context,
                "operation_scratch",
            )?;
            for posting in source {
                postings.push(posting);
            }
            postings.as_mut_slice().sort_unstable_by_key(|posting| {
                catalog_id_number(&posting.chunk_id).unwrap_or(u64::MAX)
            });
            write_string(&mut writer, term)?;
            writer.write_all(&(source.len() as u64).to_le_bytes())?;
            writer.write_all(&(source.len() as u64).to_le_bytes())?;
            for posting in postings.as_slice() {
                writer.write_all(&catalog_id_number(&posting.chunk_id)?.to_le_bytes())?;
                writer.write_all(&(posting.term_frequency as u64).to_le_bytes())?;
            }
        }
        writer.finish_segment()
    }
}

pub(crate) struct VectorRecordSegment<'a> {
    collection: &'a CollectionInfo,
    commit: u64,
    entries: crate::json::ReservedVec<(&'a Embedding, Option<&'a RecordRef>)>,
    reader: PayloadReader,
    context: crate::json::JsonOperationContext,
    #[cfg(test)]
    pub(super) encode_calls: usize,
}

impl<'a> VectorRecordSegment<'a> {
    pub(crate) fn new<I>(
        collection: &'a CollectionInfo,
        commit: u64,
        entries: I,
        reader: PayloadReader,
        context: &crate::json::JsonOperationContext,
    ) -> Result<Self>
    where
        I: IntoIterator<Item = (&'a Embedding, Option<&'a RecordRef>)>,
        I::IntoIter: Clone,
    {
        let entries = entries.into_iter();
        let mut reserved = crate::json::ReservedVec::with_capacity(
            entries.clone().count(),
            context,
            "operation_scratch",
        )?;
        for entry in entries {
            reserved.push(entry);
        }
        reserved.as_mut_slice().sort_unstable_by_key(|(entry, _)| {
            catalog_id_number(&entry.chunk_id).unwrap_or(u64::MAX)
        });
        validate_vector_order(
            reserved
                .as_slice()
                .iter()
                .map(|(entry, _)| entry.chunk_id.as_str()),
        )?;
        Ok(Self {
            collection,
            commit,
            entries: reserved,
            reader,
            context: context.clone(),
            #[cfg(test)]
            encode_calls: 0,
        })
    }

    fn header(&self) -> VectorSegmentHeader<'_> {
        VectorSegmentHeader {
            collection_id: &self.collection.collection_id,
            commit: self.commit,
            dim: self.collection.embedding_dim,
            metric: self.collection.distance_metric,
            model: self.collection.embedding_model.as_deref(),
            normalized: self.collection.vectors_normalized,
            count: self.entries.as_slice().len(),
        }
    }

    pub(crate) fn encoded_len(&self) -> Result<u64> {
        self.header().encoded_len()
    }

    pub(crate) fn validate_transient(&mut self) -> Result<()> {
        self.write(&mut std::io::sink())
    }

    pub(crate) fn try_materialize(&mut self, limit: usize) -> Result<Option<CachedVectorSegment>> {
        let Some(mut segment) = CachedVectorSegment::with_capacity(
            self.collection,
            self.commit,
            self.entries.as_slice().len(),
            &self.context,
            limit,
        )?
        else {
            return Ok(None);
        };
        for (summary, reference) in self.entries.as_slice() {
            let retained = match reference {
                Some(reference) => self.reader.read_embedding_in(reference, &self.context),
                None => RetainedEmbedding::copy(summary, &self.context),
            };
            let retained = match retained {
                Err(error) if crate::json::is_memory_budget_error(&error) => return Ok(None),
                result => result?,
            };
            validate_embedding_reference(retained.embedding(), summary)?;
            validate_vector_values(&retained.embedding().vector, self.collection.embedding_dim)?;
            if !segment.push_retained(retained, limit) {
                return Ok(None);
            }
        }
        Ok(Some(segment))
    }

    fn write(&mut self, output: &mut impl Write) -> Result<()> {
        #[cfg(test)]
        {
            self.encode_calls += 1;
        }
        let mut writer = DigestWriter::new(output);
        write_vector_header(&self.header(), &mut writer)?;
        for (summary, reference) in self.entries.as_slice() {
            let retained;
            let embedding = match reference {
                Some(reference) => {
                    retained = self.reader.read_embedding_in(reference, &self.context)?;
                    validate_embedding_reference(retained.embedding(), summary)?;
                    retained.embedding()
                }
                None => summary,
            };
            write_vector_values(
                &embedding.chunk_id,
                &embedding.embedding_id,
                &embedding.vector,
                self.collection.embedding_dim,
                &mut writer,
            )?;
        }
        writer.finish_segment()
    }
}

fn validate_embedding_reference(embedding: &Embedding, summary: &Embedding) -> Result<()> {
    if embedding.embedding_id != summary.embedding_id
        || embedding.chunk_id != summary.chunk_id
        || embedding.collection_id != summary.collection_id
    {
        return Err(Error::new(
            ErrorCode::FileFormatError,
            format!(
                "record ref for embedding {} did not contain the expected EmbeddingPut payload",
                summary.embedding_id
            ),
        ));
    }
    Ok(())
}

#[cfg(test)]
pub(super) fn write_text_segment(
    segment: &TextIndexSegmentPayload,
    output: &mut impl Write,
) -> Result<()> {
    let mut chunks = segment.chunks.iter().collect::<Vec<_>>();
    chunks.sort_by_key(|entry| catalog_id_number(&entry.chunk_id).unwrap_or(u64::MAX));
    let mut terms = segment.terms.iter().collect::<Vec<_>>();
    terms.sort_by(|a, b| a.term.as_bytes().cmp(b.term.as_bytes()));
    let mut writer = DigestWriter::new(output);
    writer.write_all(&TEXT_SEGMENT_SCHEMA_VERSION.to_le_bytes())?;
    writer.write_all(&catalog_id_number(&segment.collection_id)?.to_le_bytes())?;
    writer.write_all(&segment.covered_commit_sequence_no.to_le_bytes())?;
    writer.write_all(&TEXT_INDEX_KIND_BM25.to_le_bytes())?;
    writer.write_all(&DEFAULT_EN_TOKENIZER_ID.to_le_bytes())?;
    writer.write_all(&(crate::index::text::BM25_K1 as f32).to_le_bytes())?;
    writer.write_all(&(crate::index::text::BM25_B as f32).to_le_bytes())?;
    writer.write_all(&(chunks.len() as u64).to_le_bytes())?;
    for chunk in chunks {
        writer.write_all(&catalog_id_number(&chunk.chunk_id)?.to_le_bytes())?;
        writer.write_all(&catalog_id_number(&chunk.document_id)?.to_le_bytes())?;
        writer.write_all(&(chunk.token_count as u64).to_le_bytes())?;
    }
    writer.write_all(&(terms.len() as u64).to_le_bytes())?;
    for term in terms {
        let mut postings = term.postings.iter().collect::<Vec<_>>();
        postings.sort_by_key(|entry| catalog_id_number(&entry.chunk_id).unwrap_or(u64::MAX));
        write_string(&mut writer, &term.term)?;
        writer.write_all(&(postings.len() as u64).to_le_bytes())?;
        writer.write_all(&(postings.len() as u64).to_le_bytes())?;
        for posting in postings {
            writer.write_all(&catalog_id_number(&posting.chunk_id)?.to_le_bytes())?;
            writer.write_all(&(posting.term_frequency as u64).to_le_bytes())?;
        }
    }
    writer.finish_segment()
}

fn validate_vector_order<'a>(ids: impl Iterator<Item = &'a str>) -> Result<()> {
    let mut previous = None;
    for id in ids {
        let number = catalog_id_number(id)?;
        if previous >= Some(number) {
            return Err(Error::new(
                ErrorCode::FileFormatError,
                "vector segment chunk IDs must be strictly increasing",
            ));
        }
        previous = Some(number);
    }
    Ok(())
}

fn write_cached_vector_segment(
    segment: &CachedVectorSegment,
    output: &mut impl Write,
) -> Result<()> {
    validate_vector_order(segment.vectors.iter().map(|entry| entry.chunk_id.as_str()))?;
    let mut writer = DigestWriter::new(output);
    write_vector_header(&segment.header(), &mut writer)?;
    for entry in &segment.vectors {
        write_vector_entry(entry, segment.dim, &mut writer)?;
    }
    writer.finish_segment()
}

#[cfg(test)]
pub(super) fn write_vector_segment(
    segment: &VectorIndexSegmentPayload,
    output: &mut impl Write,
) -> Result<()> {
    let mut vectors = segment.vectors.iter().collect::<Vec<_>>();
    vectors.sort_by_key(|entry| catalog_id_number(&entry.chunk_id).unwrap_or(u64::MAX));
    let mut writer = DigestWriter::new(output);
    write_vector_header(&segment.header(), &mut writer)?;
    for entry in vectors {
        write_vector_entry(entry, segment.dim, &mut writer)?;
    }
    writer.finish_segment()
}

impl VectorIndexSegmentPayload {
    fn header(&self) -> VectorSegmentHeader<'_> {
        VectorSegmentHeader {
            collection_id: &self.collection_id,
            commit: self.covered_commit_sequence_no,
            dim: self.dim,
            metric: self.distance_metric,
            model: self.embedding_model.as_deref(),
            normalized: self.vectors_normalized,
            count: self.vectors.len(),
        }
    }

    pub(crate) fn encoded_len(&self) -> Result<u64> {
        self.header().encoded_len()
    }
}

struct VectorSegmentHeader<'a> {
    collection_id: &'a str,
    commit: u64,
    dim: usize,
    metric: DistanceMetric,
    model: Option<&'a str>,
    normalized: bool,
    count: usize,
}

impl VectorSegmentHeader<'_> {
    fn encoded_len(&self) -> Result<u64> {
        let mut digest = PayloadDigest::default();
        write_vector_header(self, &mut digest)?;
        let values = (self.dim as u64)
            .checked_mul(4)
            .and_then(|bytes| bytes.checked_add(16))
            .and_then(|bytes| bytes.checked_mul(self.count as u64))
            .ok_or_else(|| {
                Error::new(ErrorCode::FileFormatError, "vector segment length overflow")
            })?;
        checked_writer_add(digest.len, values, "vector segment length overflow").and_then(|len| {
            checked_writer_add(
                len,
                4 + INDEX_RECORD_HEADER_LEN as u64,
                "vector segment length overflow",
            )
        })
    }
}

fn write_vector_header(header: &VectorSegmentHeader<'_>, writer: &mut impl Write) -> Result<()> {
    let dim = u32::try_from(header.dim).map_err(|_| {
        Error::new(
            ErrorCode::InvalidVectorLength,
            "vector segment dimension exceeds u32",
        )
    })?;
    writer.write_all(&INDEX_SEGMENT_SCHEMA_VERSION.to_le_bytes())?;
    writer.write_all(&catalog_id_number(header.collection_id)?.to_le_bytes())?;
    writer.write_all(&header.commit.to_le_bytes())?;
    writer.write_all(&VECTOR_INDEX_KIND_EXACT_F32.to_le_bytes())?;
    writer.write_all(&dim.to_le_bytes())?;
    writer.write_all(&[distance_metric_segment_id(header.metric)])?;
    writer.write_all(&[u8::from(header.model.is_some())])?;
    if let Some(model) = header.model {
        write_string(writer, model)?;
    }
    writer.write_all(&[u8::from(header.normalized)])?;
    writer.write_all(&(header.count as u64).to_le_bytes())?;
    Ok(())
}

fn write_vector_entry(entry: &VectorIndexEntry, dim: usize, writer: &mut impl Write) -> Result<()> {
    write_vector_values(
        &entry.chunk_id,
        &entry.embedding_id,
        &entry.vector,
        dim,
        writer,
    )
}

fn write_vector_values(
    chunk_id: &str,
    embedding_id: &str,
    vector: &[f32],
    dim: usize,
    writer: &mut impl Write,
) -> Result<()> {
    validate_vector_values(vector, dim)?;
    writer.write_all(&catalog_id_number(chunk_id)?.to_le_bytes())?;
    writer.write_all(&catalog_id_number(embedding_id)?.to_le_bytes())?;
    codec::write_f32_values(writer, vector)?;
    Ok(())
}

fn validate_vector_values(vector: &[f32], dim: usize) -> Result<()> {
    if vector.len() != dim {
        return Err(Error::new(
            ErrorCode::InvalidVectorLength,
            "vector segment entry dimension mismatch",
        )
        .with_detail("expected", dim)
        .with_detail("actual", vector.len()));
    }
    if vector.iter().any(|value| !value.is_finite()) {
        return Err(Error::new(
            ErrorCode::InvalidVectorLength,
            "vector segment entries must be finite f32 values",
        ));
    }
    Ok(())
}

fn write_string(output: &mut impl Write, value: &str) -> Result<()> {
    let len = u32::try_from(value.len()).map_err(|_| {
        Error::new(
            ErrorCode::FileFormatError,
            "index string length exceeds u32",
        )
    })?;
    output.write_all(&len.to_le_bytes())?;
    output.write_all(value.as_bytes())?;
    Ok(())
}

struct DigestWriter<W> {
    output: W,
    digest: PayloadDigest,
}

impl<W: Write> DigestWriter<W> {
    fn new(output: W) -> Self {
        Self {
            output,
            digest: PayloadDigest::default(),
        }
    }

    fn finish_segment(mut self) -> Result<()> {
        self.output
            .write_all(&self.digest.crc.finish().to_le_bytes())?;
        Ok(())
    }
}

impl<W: Write> Write for DigestWriter<W> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        let count = self.output.write(&bytes[..bytes.len().min(BUFFER_BYTES)])?;
        self.digest.write_all(&bytes[..count])?;
        Ok(count)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.output.flush()
    }
}

#[cfg(test)]
mod framing_tests {
    use super::*;

    #[test]
    fn checkpoint_write_buffer_admits_actual_capacity_and_returns_all_credit() {
        use crate::json::{JsonOperationContext, WorkLedger};
        use std::sync::Arc;
        let payload = vec![19; BUFFER_BYTES + 1];
        for parent in [false, true] {
            for available in [0, 1, 13, BUFFER_BYTES - 1, BUFFER_BYTES, BUFFER_BYTES + 1] {
                for buffered in [false, true] {
                    let working = if parent { BUFFER_BYTES + 19 } else { available };
                    let total = if parent {
                        available + 7
                    } else {
                        BUFFER_BYTES + 26
                    };
                    let context = JsonOperationContext::new(Arc::new(WorkLedger::with_limits(
                        working, 0, total,
                    )));
                    let blocker = context.cache_reservation(7, usize::MAX).unwrap();
                    context
                        .workspace_reservation()
                        .grow_to(0, 0, "operation_scratch")
                        .unwrap();
                    let granted = if buffered {
                        available.min(BUFFER_BYTES)
                    } else {
                        0
                    };
                    let mut bytes = vec![7; payload.len()];
                    let mut output = std::io::Cursor::new(bytes.as_mut_slice());
                    let mut calls = 0;
                    let (result, allocated) = crate::test_alloc::allocation_bytes(|| {
                        with_write_buffer(&mut output, &context, buffered, |buffer| {
                            calls += 1;
                            assert_eq!(buffer.capacity(), granted);
                            assert_eq!(context.available_workspace_bytes(), available - granted);
                            let mut digest = DigestWriter::new(buffer);
                            digest.write_all(&payload)?;
                            Ok(digest.digest)
                        })
                    });
                    let digest = result.unwrap();
                    assert_eq!(digest.len, payload.len() as u64);
                    assert_eq!(digest.crc.finish(), checksum32(&payload));
                    assert_eq!(allocated, granted as u128);
                    assert_eq!(calls, 1);
                    assert_eq!(output.position(), payload.len() as u64);
                    assert_eq!(bytes, payload);
                    let mut free = context.workspace_reservation();
                    free.grow_to(available, available, "operation_scratch")
                        .unwrap();
                    assert!(free
                        .grow_to(available + 1, available + 1, "operation_scratch")
                        .is_err());
                    drop(blocker);
                    let complete = if parent { available + 7 } else { available };
                    free.grow_to(complete, complete, "operation_scratch")
                        .unwrap();
                    assert!(free
                        .grow_to(complete + 1, complete + 1, "operation_scratch")
                        .is_err());
                }
            }
        }
    }

    #[test]
    fn checkpoint_write_buffer_never_flushes_or_replays_after_encode_or_io_failure() {
        use crate::json::{JsonOperationContext, WorkLedger};
        use std::sync::Arc;
        struct Output {
            bytes: Vec<u8>,
            writes: usize,
            flushes: usize,
            fail_write: bool,
            fail_flush: bool,
            short_write: bool,
        }
        impl Write for Output {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                self.writes += 1;
                if self.fail_write || (self.short_write && self.writes > 1) {
                    return Err(std::io::Error::other("injected checkpoint write failure"));
                }
                assert!(bytes.len() <= self.bytes.capacity() - self.bytes.len());
                let count = if self.short_write { 2 } else { bytes.len() };
                self.bytes.extend_from_slice(&bytes[..count]);
                Ok(count)
            }
            fn flush(&mut self) -> std::io::Result<()> {
                self.flushes += 1;
                if self.fail_flush {
                    return Err(std::io::Error::other("injected checkpoint flush failure"));
                }
                Ok(())
            }
        }
        for mode in 0..6 {
            let context = JsonOperationContext::new(Arc::new(WorkLedger::with_limits(64, 0, 64)));
            let mut output = Output {
                bytes: Vec::with_capacity(65),
                writes: 0,
                flushes: 0,
                fail_write: mode == 2 || mode == 3,
                fail_flush: mode == 4,
                short_write: mode == 5,
            };
            let mut calls = 0;
            let result = with_write_buffer(&mut output, &context, true, |buffer| {
                calls += 1;
                if mode == 2 {
                    buffer.write_all(&[19; 65])?;
                } else if mode == 5 {
                    buffer.write_all(&[19, 23, 29])?;
                } else {
                    buffer.write_all(&[19])?;
                }
                if mode == 1 {
                    context
                        .workspace_reservation()
                        .grow_to(1, 1, "operation_scratch")?;
                }
                Ok(())
            });
            assert_eq!(calls, 1);
            match (mode, result) {
                (0, Ok(())) => {
                    assert_eq!(output.bytes, [19]);
                    assert_eq!((output.writes, output.flushes), (1, 1));
                }
                (1, Err(error)) => {
                    assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                    assert!(error
                        .detail()
                        .contains(&("limit_bytes".into(), "64".into())));
                    assert!(error
                        .detail()
                        .contains(&("requested_bytes".into(), "65".into())));
                    assert_eq!((output.writes, output.flushes), (0, 0));
                    assert!(output.bytes.is_empty());
                }
                (2 | 3, Err(error)) => {
                    assert_eq!(error.code(), ErrorCode::IoError);
                    assert_eq!((output.writes, output.flushes), (1, 0));
                    assert!(output.bytes.is_empty());
                }
                (4, Err(error)) => {
                    assert_eq!(error.code(), ErrorCode::IoError);
                    assert_eq!((output.writes, output.flushes), (1, 1));
                    assert_eq!(output.bytes, [19]);
                }
                (5, Err(error)) => {
                    assert_eq!(error.code(), ErrorCode::IoError);
                    assert_eq!((output.writes, output.flushes), (2, 0));
                    assert_eq!(output.bytes, [19, 23]);
                }
                _ => panic!("unexpected checkpoint buffer mode {mode}"),
            }
            let mut free = context.workspace_reservation();
            free.grow_to(64, 64, "operation_scratch").unwrap();
            assert!(free.grow_to(65, 65, "operation_scratch").is_err());
        }
    }

    #[test]
    fn complete_metadata_checkpoint_needs_no_heap_after_directory_admission() {
        for seed in 0..100u64 {
            let checkpoint = IndexCheckpointPayload {
                metadata: IndexCheckpointMetadata {
                    covered_last_batch_id: seed + 1,
                    covered_last_commit_sequence_no: seed + 3,
                    coverage_hash: seed as u32 ^ 0xa55a_9e37,
                    collection_count: 0,
                    live_document_count: 0,
                    live_chunk_count: 0,
                    live_embedding_count: 0,
                    text_segment_count: 0,
                    vector_segment_count: 0,
                    build_mode: String::new(),
                },
                text_segments: Vec::new(),
                vector_segments: Vec::new(),
            };
            let expected = build_index_checkpoint_suffix(&checkpoint, 37).unwrap();
            let context = crate::json::JsonOperationContext::standalone();
            let directory = reserve_directory(&checkpoint.metadata, &context).unwrap();
            let mut bytes = [0; 306];
            bytes[..37].fill(7);
            let mut output = std::io::Cursor::new(bytes.as_mut_slice());
            let (result, allocated) = crate::test_alloc::allocation_bytes(|| {
                write_checkpoint_segments_in(
                    &mut output,
                    &checkpoint.metadata,
                    std::iter::empty(),
                    37,
                    |_, _| panic!("metadata-only checkpoint completed a segment"),
                    directory,
                    false,
                )
            });
            let (payload_len, source) = result.unwrap();
            assert_eq!(allocated, 0);
            assert_eq!(payload_len, 108);
            assert_eq!(
                source.reserved_bytes(),
                std::mem::size_of::<IndexExtentEntry>()
            );
            assert_eq!(output.position(), 306);
            assert_eq!(&bytes[..37], &[7; 37]);
            assert_eq!(&bytes[37..], &expected);
            assert_eq!(&bytes[296..300], INDEX_COMPLETE_MARKER);
            assert_eq!(read_u16(&bytes[300..302]), 10);
            assert_eq!(read_u32(&bytes[302..]), checksum32(&bytes[296..302]));
            drop(source);
            assert_eq!(
                context.available_workspace_bytes(),
                crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES
            );
        }
    }

    #[test]
    fn fixed_checkpoint_framing_has_no_heap_allocation_and_keeps_wire_fields() {
        for seed in 0..100u64 {
            let checkpoint = IndexCheckpointMetadata {
                covered_last_batch_id: u64::MAX - seed,
                covered_last_commit_sequence_no: seed * 13 + 3,
                coverage_hash: seed as u32 ^ 0xa55a_9e37,
                collection_count: 0,
                live_document_count: 0,
                live_chunk_count: 0,
                live_embedding_count: 0,
                text_segment_count: 0,
                vector_segment_count: 0,
                build_mode: String::new(),
            };
            for len in [0, seed + 7, u64::MAX - INDEX_CHECKPOINT_HEADER_LEN as u64] {
                let mut payload = PayloadDigest {
                    len,
                    crc: Crc32c::default(),
                };
                payload.crc.update(&seed.to_le_bytes());
                let count = u64::MAX - seed;
                let table_len = seed * INDEX_EXTENT_ENTRY_LEN as u64;
                let ((checkpoint_header, segment_header, metadata), allocated) =
                    crate::test_alloc::allocation_bytes(|| {
                        (
                            header(&checkpoint, seed, count, len, table_len, 17, 29).unwrap(),
                            record_header(INDEX_TEXT_SEGMENT, seed, count, &payload).unwrap(),
                            build_index_commit_metadata_payload(
                                &checkpoint,
                                count,
                                31,
                                table_len,
                                43,
                                47,
                            ),
                        )
                    });
                assert_eq!(allocated, 0);
                assert_eq!(&checkpoint_header[..8], INDEX_CHECKPOINT_MAGIC);
                assert_eq!(
                    read_u16(&checkpoint_header[8..10]),
                    INDEX_CHECKPOINT_VERSION
                );
                assert_eq!(read_u16(&checkpoint_header[10..12]), 105);
                for at in [12, 20, 28] {
                    assert_eq!(read_u64(&checkpoint_header[at..at + 8]), seed);
                }
                assert_eq!(
                    read_u64(&checkpoint_header[36..44]),
                    checkpoint.covered_last_batch_id
                );
                assert_eq!(
                    read_u64(&checkpoint_header[44..52]),
                    checkpoint.covered_last_commit_sequence_no
                );
                assert_eq!(checkpoint_header[52], INDEX_COVERAGE_FULL_VISIBLE_SNAPSHOT);
                assert_eq!(
                    read_u32(&checkpoint_header[53..57]),
                    checkpoint.coverage_hash
                );
                assert_eq!(read_u64(&checkpoint_header[57..65]), count);
                assert_eq!(read_u64(&checkpoint_header[65..73]), count);
                assert_eq!(read_u64(&checkpoint_header[73..81]), len + 105);
                assert_eq!(read_u64(&checkpoint_header[81..89]), table_len);
                assert_eq!(read_u32(&checkpoint_header[89..93]), 17);
                assert_eq!(read_u64(&checkpoint_header[93..101]), len);
                assert_eq!(read_u32(&checkpoint_header[101..105]), 29);
                assert_eq!(&segment_header[..4], INDEX_RECORD_MAGIC);
                assert_eq!(read_u16(&segment_header[4..6]), INDEX_TEXT_SEGMENT);
                assert_eq!(read_u16(&segment_header[6..8]), 1);
                assert_eq!(read_u16(&segment_header[8..10]), 50);
                assert_eq!(read_u64(&segment_header[10..18]), len + 50);
                assert_eq!(read_u64(&segment_header[18..26]), seed);
                assert_eq!(read_u64(&segment_header[26..34]), count);
                assert_eq!(read_u64(&segment_header[34..42]), len);
                assert_eq!(
                    read_u32(&segment_header[42..46]),
                    checksum32(&seed.to_le_bytes())
                );
                assert_eq!(
                    read_u32(&segment_header[46..50]),
                    checksum32(&segment_header[..46])
                );
                assert_eq!(read_u16(&metadata[..2]), 1);
                assert_eq!(read_u64(&metadata[2..10]), checkpoint.covered_last_batch_id);
                assert_eq!(
                    read_u64(&metadata[10..18]),
                    checkpoint.covered_last_commit_sequence_no
                );
                assert_eq!(read_u64(&metadata[18..26]), count);
                assert_eq!(read_u32(&metadata[26..30]), 47);
                assert_eq!(read_u64(&metadata[30..38]), count);
                assert_eq!(read_u64(&metadata[38..46]), 31);
                assert_eq!(read_u64(&metadata[46..54]), table_len);
                assert_eq!(read_u32(&metadata[54..58]), 43);
            }
            let too_large = PayloadDigest {
                len: u64::MAX,
                crc: Crc32c::default(),
            };
            assert_eq!(
                header(&checkpoint, seed, 0, u64::MAX, 0, 0, 0)
                    .unwrap_err()
                    .code(),
                ErrorCode::FileFormatError
            );
            assert_eq!(
                record_header(INDEX_TEXT_SEGMENT, seed, 1, &too_large)
                    .unwrap_err()
                    .code(),
                ErrorCode::FileFormatError
            );
        }
    }
}

#[cfg(test)]
mod cached_vector_tests {
    use super::*;
    use crate::test_alloc::allocations;

    fn payload(count: usize, dim: usize) -> VectorIndexSegmentPayload {
        VectorIndexSegmentPayload {
            collection_id: catalog_id_string("col_", 1),
            covered_commit_sequence_no: 7,
            dim,
            distance_metric: DistanceMetric::DotProduct,
            embedding_model: Some("m".into()),
            vectors_normalized: false,
            vectors: (0..count)
                .map(|index| VectorIndexEntry {
                    chunk_id: catalog_id_string(
                        "chk_",
                        [
                            2,
                            10,
                            65_535,
                            65_536,
                            9_999_999_999_999_999,
                            10_000_000_000_000_000,
                            u64::MAX,
                        ][index % 7],
                    ),
                    embedding_id: catalog_id_string("emb_", index as u64 + 1),
                    vector: (0..dim)
                        .map(|value| if value % 2 == 0 { -0.0 } else { value as f32 })
                        .collect(),
                })
                .collect(),
        }
    }

    #[test]
    fn cached_vector_encoding_allocates_no_sort_buffer_and_preserves_wire() {
        let (_, observed) = allocations(|| std::hint::black_box(vec![7u8; 128]));
        assert!(observed > 0);
        for count in [0, 1, 2, 7, 17, 129] {
            for dim in [1, 4, 1025] {
                let mut original = payload(count, dim);
                if count > 7 {
                    for (index, entry) in original.vectors.iter_mut().enumerate() {
                        entry.chunk_id = catalog_id_string("chk_", index as u64 + 1);
                    }
                }
                let expected = encode_vector_index_segment_payload(&original).unwrap();
                let context = crate::json::JsonOperationContext::standalone();
                let credit = CachedVectorSegment::reserve(
                    &context,
                    usize::MAX,
                    original.encoded_len().unwrap(),
                )
                .unwrap();
                let cached = CachedVectorSegment::new(original, credit);
                let mut borrowed = Segment::BorrowedCachedVector(&cached);
                let mut output = Vec::with_capacity(expected.len());
                let (result, count) = allocations(|| borrowed.encode(&mut output));
                result.unwrap();
                assert_eq!(count, 0);
                assert_eq!(output, expected);
                output.clear();
                let mut owned = Segment::CachedVector(cached);
                let (result, count) = allocations(|| owned.encode(&mut output));
                result.unwrap();
                assert_eq!(count, 0);
                assert_eq!(output, expected);
            }
        }
    }

    #[test]
    fn cached_vector_encoding_rejects_bad_order_before_output() {
        for fault in 0..3 {
            let mut original = payload(2, 2);
            match fault {
                0 => original.vectors.reverse(),
                1 => original.vectors[1].chunk_id = "chk_0002".into(),
                _ => original.vectors[0].chunk_id = "invalid".into(),
            }
            let context = crate::json::JsonOperationContext::standalone();
            let credit = context.cache_reservation(4096, 4096).unwrap();
            let cached = CachedVectorSegment::new(original, credit);
            let mut output = Vec::new();
            assert_eq!(
                write_cached_vector_segment(&cached, &mut output)
                    .unwrap_err()
                    .code(),
                ErrorCode::FileFormatError
            );
            assert!(output.is_empty());
            drop(cached);
            assert!(context.cache_reservation(4096, 4096).is_some());
        }
    }
}

#[cfg(test)]
mod text_tests {
    use super::*;
    use crate::catalog::Catalog;
    use crate::index::text::TextIndex;
    use crate::json::{JsonOperationContext, WorkLedger};
    use crate::model::VectorIndexConfig;
    use std::sync::Arc;

    fn reference_text_segment(
        catalog: &crate::catalog::Catalog,
        index: &crate::index::text::TextIndex,
        collection: &CollectionInfo,
        commit: u64,
    ) -> TextIndexSegmentPayload {
        let chunks = catalog
            .live_chunks()
            .filter(|chunk| chunk.collection_id == collection.collection_id)
            .map(|chunk| TextIndexChunkEntry {
                chunk_id: chunk.chunk_id.clone(),
                document_id: chunk.document_id.clone(),
                token_count: index
                    .chunk_lengths
                    .get(&chunk.chunk_id)
                    .copied()
                    .unwrap_or(chunk.token_count.max(1)),
            })
            .collect();
        let terms = index
            .postings()
            .iter()
            .filter_map(|(term, postings)| {
                let postings = postings
                    .iter()
                    .filter(|posting| posting.collection_id.as_ref() == collection.collection_id)
                    .map(|posting| TextIndexPostingEntry {
                        chunk_id: posting.chunk_id.clone(),
                        term_frequency: posting.term_frequency,
                    })
                    .collect::<Vec<_>>();
                (!postings.is_empty()).then(|| TextIndexTermEntry {
                    term: term.clone(),
                    postings,
                })
            })
            .collect();
        TextIndexSegmentPayload {
            collection_id: collection.collection_id.clone(),
            covered_commit_sequence_no: commit,
            chunks,
            terms,
        }
    }

    fn fixture(seed: usize) -> (Catalog, TextIndex) {
        let mut catalog = Catalog::default();
        for owner in 1..=3 {
            let collection_id = catalog_id_string("col_", owner);
            let document_id = catalog_id_string("doc_", owner);
            catalog.collections.insert(
                collection_id.clone(),
                CollectionInfo {
                    collection_id: collection_id.clone(),
                    name: collection_id.clone(),
                    tokenizer: Default::default(),
                    embedding_dim: 2,
                    distance_metric: DistanceMetric::Cosine,
                    embedding_model: None,
                    vectors_normalized: false,
                    vector_index: VectorIndexConfig::Exact,
                    created_at: String::new(),
                },
            );
            catalog.documents.insert(
                document_id.clone(),
                Box::new(Document {
                    document_id: document_id.clone(),
                    collection_id: collection_id.clone(),
                    external_id: document_id.clone(),
                    title: None,
                    source_uri: None,
                    metadata: JsonValue::object(),
                    created_at: String::new(),
                    updated_at: String::new(),
                }),
            );
            let count = if owner == 3 { 0 } else { seed % 29 + 2 };
            for ordinal in 0..count {
                let number = if ordinal == 0 {
                    u64::MAX - owner
                } else {
                    owner * 1000 + ordinal as u64
                };
                let id = catalog_id_string("chk_", number);
                let text = if ordinal % 5 == 0 {
                    "the and".into()
                } else {
                    format!(
                        "shared symbol_path \u{4e2d}\u{6587} {} {}",
                        "term ".repeat(ordinal % 3 + 1),
                        "long".repeat(if seed == 99 { 16_385 } else { seed + 1 })
                    )
                };
                catalog.chunks.insert(
                    id.clone(),
                    Box::new(Chunk {
                        chunk_id: id.clone(),
                        collection_id: collection_id.clone(),
                        document_id: document_id.clone(),
                        chunk_external_id: id.clone(),
                        text,
                        text_hash: String::new(),
                        ordinal,
                        token_count: seed + 3,
                        metadata: JsonValue::object(),
                        source_span: None,
                        embedding_id: None,
                        created_at: String::new(),
                    }),
                );
                if (ordinal + seed) % 7 == 0 {
                    catalog.deleted_chunks.insert(id);
                }
            }
            if seed % 13 == 0 && owner == 2 {
                catalog.deleted_documents.insert(document_id);
            }
        }
        let mut index = TextIndex::try_build_matching(
            &catalog.visible_snapshot(),
            |_| true,
            |visible| Ok(visible.chunk.text.clone()),
        )
        .unwrap();
        if let Some(chunk) = catalog.live_chunks().next() {
            index.chunk_lengths.remove(&chunk.chunk_id);
        }
        (catalog, index)
    }

    #[test]
    fn borrowed_text_segments_preserve_wire_with_only_one_admitted_sort_array() {
        for seed in 0..100 {
            let (catalog, index) = fixture(seed);
            let signature = catalog.state_signature();
            for collection in catalog.live_collections() {
                let expected = reference_text_segment(&catalog, &index, collection, seed as u64);
                let bytes = encode_text_index_segment_payload(&expected).unwrap();
                let chunks = expected.chunks.len();
                let peak = chunks.max(
                    expected
                        .terms
                        .iter()
                        .map(|term| term.postings.len())
                        .max()
                        .unwrap_or(0),
                ) * std::mem::size_of::<&Chunk>();
                for limit in [peak.saturating_sub(1), peak, peak + 1] {
                    let ledger = Arc::new(WorkLedger::with_limits(limit, limit, limit));
                    let context = JsonOperationContext::new(ledger.clone());
                    let competing = JsonOperationContext::new(ledger);
                    let source =
                        TextRecordSegment::new(&catalog, &index, collection, seed as u64, &context);
                    let mut output = Vec::new();
                    let result = source.write(&mut output);
                    if limit < peak {
                        let error = result.unwrap_err();
                        assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                        assert!(error
                            .detail()
                            .contains(&("requested_bytes".into(), peak.to_string())));
                        assert!(output.is_empty());
                    } else {
                        result.unwrap();
                        assert_eq!(output, bytes);
                    }
                    competing
                        .workspace_reservation()
                        .grow_to(limit, limit, "operation_scratch")
                        .unwrap();
                }
            }
            assert_eq!(catalog.state_signature(), signature);
        }
    }

    #[test]
    fn borrowed_text_segment_io_failures_release_scratch_and_keep_source_reusable() {
        struct FailAfter {
            remaining: usize,
        }
        impl Write for FailAfter {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                if self.remaining == 0 {
                    return Err(std::io::Error::other("injected text output failure"));
                }
                let count = bytes.len().min(self.remaining);
                self.remaining -= count;
                Ok(count)
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        for seed in 0..100 {
            let (catalog, index) = fixture(seed);
            let collection = catalog
                .collections
                .get(&catalog_id_string("col_", 1))
                .unwrap();
            let expected = reference_text_segment(&catalog, &index, collection, seed as u64);
            let bytes = encode_text_index_segment_payload(&expected).unwrap();
            let limit = expected.chunks.len() * std::mem::size_of::<&Chunk>();
            let context =
                JsonOperationContext::new(Arc::new(WorkLedger::with_limits(limit, limit, limit)));
            let source =
                TextRecordSegment::new(&catalog, &index, collection, seed as u64, &context);
            for stop in [0, 38, bytes.len() / 2, bytes.len() - 1] {
                let error = source
                    .write(&mut FailAfter { remaining: stop })
                    .unwrap_err();
                assert_eq!(error.code(), ErrorCode::IoError);
                context
                    .workspace_reservation()
                    .grow_to(limit, limit, "operation_scratch")
                    .unwrap();
                let mut recovered = Vec::new();
                source.write(&mut recovered).unwrap();
                assert_eq!(recovered, bytes);
            }
        }
    }
}
