use super::*;
use crate::catalog::{Catalog, VisibleChunk};
use crate::index::text::TextStatsSummary;
use crate::json::{is_memory_budget_error, JsonOperationContext, ReservedVec};
use crate::temporary::{
    create_budgeted_temporary_file_in_context, ExternalTermFrequencies, TempFileGuard,
    TemporaryDiskBudget, TemporaryFile,
};
use std::sync::Arc;

#[cfg(test)]
mod tests;

// One file owner remains live across all collections and suffix-write attempts.
pub(crate) struct TextStaging {
    file: TemporaryFile,
    cleanup: TempFileGuard,
    disk: Arc<TemporaryDiskBudget>,
    bytes: u64,
    reserved: u64,
    healthy: bool,
}

pub(crate) struct TextExtent {
    collection: u64,
    commit: u64,
    offset: u64,
    length: u64,
    checksum: u32,
}

pub(crate) struct StagedText<'a> {
    staging: &'a TextStaging,
    pub(crate) extent: &'a TextExtent,
}

#[cfg(test)]
pub(crate) struct TextSpool {
    pub(in crate::storage) staging: TextStaging,
    pub(in crate::storage) extent: TextExtent,
}

pub(crate) struct TextCheckpoint {
    staging: Option<TextStaging>,
    bodies: ReservedVec<TextExtent>,
    stats: Option<TextStatsSummary>,
}

impl TextStaging {
    pub(crate) fn new(
        directory: &Path,
        disk: Arc<TemporaryDiskBudget>,
        context: &JsonOperationContext,
    ) -> Result<Self> {
        let (file, cleanup) = create_budgeted_temporary_file_in_context(
            directory,
            "text-checkpoint",
            "tmp",
            disk.clone(),
            context,
            "operation_scratch",
        )?;
        Ok(Self {
            file,
            cleanup,
            disk,
            bytes: 0,
            reserved: 0,
            healthy: true,
        })
    }

    fn begin(&mut self) -> Result<u64> {
        if !self.healthy || self.reserved != self.bytes || self.file.metadata()?.len() != self.bytes
        {
            return Err(Error::new(
                ErrorCode::ChecksumMismatch,
                "text staging length changed",
            ));
        }
        self.file.seek(SeekFrom::Start(self.bytes))?;
        Ok(self.bytes)
    }

    pub(crate) fn rollback(&mut self) -> Result<()> {
        let result = (|| {
            self.file.set_len(self.bytes)?;
            self.file.seek(SeekFrom::Start(self.bytes))?;
            self.cleanup.release_disk(self.reserved - self.bytes);
            self.reserved = self.bytes;
            Ok(())
        })();
        if result.is_err() {
            self.healthy = false;
        }
        result
    }

    pub(in crate::storage) fn copy_body_from(
        &mut self,
        reader: &mut dyn Read,
        length: usize,
        commit: u64,
        context: &JsonOperationContext,
        capacity: usize,
    ) -> std::result::Result<TextExtent, IndexReadError> {
        if !self.healthy || self.reserved != self.bytes || self.file.metadata()?.len() != self.bytes
        {
            return Err(IndexReadError::Resource(staging_checksum_error()));
        }
        let offset = self.bytes;
        self.file.seek(SeekFrom::Start(offset))?;
        let bytes = (length as u64)
            .checked_add(4)
            .ok_or_else(|| IndexReadError::Resource(invalid_staging()))?;
        let end = offset
            .checked_add(bytes)
            .ok_or_else(|| IndexReadError::Resource(invalid_staging()))?;
        self.cleanup
            .reserve_disk(bytes)
            .map_err(IndexReadError::Resource)?;
        self.reserved = end;
        let mut output = BufWriter::with_capacity(capacity, &mut self.file);
        let copied: std::result::Result<_, IndexReadError> = (|| {
            let mut copying = CopyReader {
                source: reader,
                output: &mut output,
                crc: Crc32c::default(),
                bytes: 0,
            };
            let parsed = decode_text_index_segment_with(
                &mut IndexPayloadReader::new(&mut copying, length, 0),
                commit,
                0,
                TextDecodeMode::Summary,
                context,
            )?;
            if copying.bytes != length as u64 {
                return Err(std::io::Error::from(std::io::ErrorKind::UnexpectedEof).into());
            }
            Ok((
                parsed.collection_number,
                copying.bytes,
                copying.crc.finish(),
            ))
        })()
        .and_then(|copied| {
            output.flush()?;
            Ok(copied)
        });
        // Failed reads and writes must not trigger another write from Drop.
        drop(output.into_parts().1);
        let (collection, bytes, checksum) = copied?;
        Ok(TextExtent {
            collection,
            commit,
            offset,
            length: bytes + 4,
            checksum,
        })
    }

    pub(crate) fn finish_copy(&mut self, extent: &TextExtent) -> Result<()> {
        if !self.healthy || extent.offset != self.bytes {
            return Err(invalid_staging());
        }
        self.file.write_all(&extent.checksum.to_le_bytes())?;
        self.file.flush()?;
        self.bytes = extent.end()?;
        Ok(())
    }

    fn reclaim(&mut self, bodies: &mut [TextExtent]) -> Result<()> {
        let result = (|| {
            self.begin()?;
            bodies.sort_unstable_by_key(|body| body.offset);
            let mut end = 0;
            let mut buffer = [0; 8192];
            for body in bodies.iter_mut() {
                let source_end = body.end()?;
                if end > body.offset || source_end > self.bytes || body.length < 22 {
                    return Err(invalid_staging());
                }
                let mut crc = Crc32c::default();
                let body_bytes = body.length - 4;
                let mut copied = 0;
                while copied != body_bytes {
                    let count = (body_bytes - copied).min(buffer.len() as u64) as usize;
                    self.file.seek(SeekFrom::Start(body.offset + copied))?;
                    self.file.read_exact(&mut buffer[..count])?;
                    crc.update(&buffer[..count]);
                    if end != body.offset {
                        self.file.seek(SeekFrom::Start(end + copied))?;
                        self.file.write_all(&buffer[..count])?;
                    }
                    copied += count as u64;
                }
                self.file.seek(SeekFrom::Start(source_end - 4))?;
                let mut tail = [0; 4];
                self.file.read_exact(&mut tail)?;
                if crc.finish() != body.checksum || u32::from_le_bytes(tail) != body.checksum {
                    return Err(staging_checksum_error());
                }
                if end != body.offset {
                    self.file.seek(SeekFrom::Start(end + body_bytes))?;
                    self.file.write_all(&tail)?;
                }
                body.offset = end;
                end = end.checked_add(body.length).ok_or_else(invalid_staging)?;
            }
            self.file.set_len(end)?;
            self.file.seek(SeekFrom::Start(end))?;
            self.cleanup.release_disk(self.reserved - end);
            self.reserved = end;
            self.bytes = end;
            bodies.sort_unstable_by_key(|body| body.collection);
            Ok(())
        })();
        if result.is_err() {
            self.healthy = false;
        }
        result
    }
}

impl TextExtent {
    fn end(&self) -> Result<u64> {
        self.offset
            .checked_add(self.length)
            .ok_or_else(invalid_staging)
    }

    pub(crate) fn collection_number(&self) -> u64 {
        self.collection
    }
}

impl StagedText<'_> {
    #[cfg(all(test, unix))]
    pub(crate) fn staging_identity(&self) -> Result<(u64, u64)> {
        use std::os::unix::fs::MetadataExt;
        let metadata = self.staging.file.metadata()?;
        Ok((metadata.dev(), metadata.ino()))
    }

    fn reader(&self) -> Result<File> {
        let mut file = self.staging.file.try_clone()?;
        if !self.staging.healthy
            || file.metadata()?.len() != self.staging.bytes
            || self.extent.end()? > self.staging.bytes
            || self.extent.length < 22
        {
            return Err(Error::new(
                ErrorCode::ChecksumMismatch,
                "text staging length changed",
            ));
        }
        file.seek(SeekFrom::Start(self.extent.offset))?;
        Ok(file)
    }

    pub(super) fn collection_number(&self) -> u64 {
        self.extent.collection
    }

    pub(super) fn write(&self, output: &mut impl Write, commit: u64) -> Result<()> {
        let mut file = self.reader()?;
        let mut prefix = [0; 18];
        file.read_exact(&mut prefix)?;
        let mut old_crc = Crc32c::default();
        old_crc.update(&prefix);
        let mut writer = DigestWriter::new(output);
        writer.write_all(&prefix[..10])?;
        writer.write_all(&commit.to_le_bytes())?;
        let mut remaining = self
            .extent
            .length
            .checked_sub(22)
            .ok_or_else(invalid_staging)?;
        let mut buffer = [0; 8192];
        while remaining != 0 {
            let count = remaining.min(buffer.len() as u64) as usize;
            file.read_exact(&mut buffer[..count])?;
            old_crc.update(&buffer[..count]);
            writer.write_all(&buffer[..count])?;
            remaining -= count as u64;
        }
        self.check_crc(&mut file, old_crc.finish())?;
        writer.finish_segment()
    }

    fn check_crc(&self, file: &mut File, actual: u32) -> Result<()> {
        let mut tail = [0; 4];
        file.read_exact(&mut tail)?;
        if actual != self.extent.checksum || u32::from_le_bytes(tail) != self.extent.checksum {
            return Err(staging_checksum_error());
        }
        Ok(())
    }

    pub(crate) fn visit(
        &self,
        context: &JsonOperationContext,
        capacity: usize,
        visit: &mut TextSegmentVisitor<'_>,
    ) -> Result<bool> {
        let mut file = self.reader()?;
        let length = usize::try_from(self.extent.length - 4).map_err(|_| invalid_staging())?;
        let mut error = None;
        let (parsed, checksum) = {
            let mut checked = CrcReader {
                source: BufReader::with_capacity(capacity, (&mut file).take(length as u64)),
                crc: Crc32c::default(),
            };
            let parsed = decode_text_index_segment_with(
                &mut IndexPayloadReader::new(&mut checked, length, 0),
                self.extent.commit,
                0,
                TextDecodeMode::Visit {
                    error: &mut error,
                    visit,
                },
                context,
            );
            if matches!(&parsed, Err(IndexReadError::Io(_))) {
                return parsed.map(|_| true).map_err(staging_read_error);
            }
            std::io::copy(&mut checked, &mut std::io::sink())?;
            (parsed, checked.crc.finish())
        };
        self.check_crc(&mut file, checksum)?;
        if matches!(&parsed, Err(IndexReadError::Resource(_))) {
            if let Some(error) = error {
                return Err(error);
            }
        }
        parsed.map_err(staging_read_error)?;
        match error {
            Some(error) => Err(error),
            None => Ok(true),
        }
    }
}

impl TextCheckpoint {
    pub(crate) fn prefetch(
        storage: &Storage,
        catalog: &Catalog,
        source: Option<&IndexSegmentSource>,
        changed: &BTreeSet<String>,
        disk: &Arc<TemporaryDiskBudget>,
        context: &JsonOperationContext,
    ) -> Result<Self> {
        let mut bodies = ReservedVec::with_capacity(
            catalog.live_collections().count(),
            context,
            "operation_scratch",
        )?;
        let directory = storage.path().parent().unwrap_or_else(|| Path::new("."));
        let mut staging = None;
        if let Some(source) = source {
            for collection in catalog.live_collections() {
                if !changed.contains(&collection.collection_id) {
                    if staging.is_none() {
                        staging = Some(TextStaging::new(directory, disk.clone(), context)?);
                    }
                    if let Some(body) = source.stage_text_into(
                        storage,
                        &collection.collection_id,
                        staging.as_mut().expect("created staging"),
                        context,
                    )? {
                        bodies.push(body);
                    }
                }
            }
        }
        bodies
            .as_mut_slice()
            .sort_unstable_by_key(|body| body.collection);
        Ok(Self {
            staging,
            bodies,
            stats: None,
        })
    }

    #[cfg(test)]
    pub(crate) fn build<T: AsRef<str>>(
        storage: &Storage,
        catalog: &Catalog,
        commit: u64,
        reuse: Option<(Self, &BTreeSet<String>)>,
        load: impl FnMut(&VisibleChunk<'_>) -> Result<T>,
        disk: &Arc<TemporaryDiskBudget>,
        context: &JsonOperationContext,
    ) -> Result<Self> {
        Self::build_reclaiming(storage, catalog, commit, reuse, load, disk, context, || {
            false
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn build_reclaiming<T: AsRef<str>>(
        storage: &Storage,
        catalog: &Catalog,
        commit: u64,
        mut reuse: Option<(Self, &BTreeSet<String>)>,
        mut load: impl FnMut(&VisibleChunk<'_>) -> Result<T>,
        disk: &Arc<TemporaryDiskBudget>,
        context: &JsonOperationContext,
        mut reclaim: impl FnMut() -> bool,
    ) -> Result<Self> {
        // Numeric ordering also supports IDs crossing the minimum-width boundary.
        let mut collections = crate::json::retry_after_memory_reclaim(
            || {
                ReservedVec::with_capacity(
                    catalog.live_collections().count(),
                    context,
                    "operation_scratch",
                )
            },
            &mut reclaim,
        )?;
        for collection in catalog.live_collections() {
            collections.push(collection);
        }
        collections
            .as_mut_slice()
            .sort_unstable_by_key(|collection| {
                catalog_id_number(&collection.collection_id).unwrap_or(u64::MAX)
            });
        if let Some((reuse, changed)) = &mut reuse {
            let old_count = reuse.bodies.as_slice().len();
            reuse.bodies.retain(|body| {
                collections
                    .as_slice()
                    .binary_search_by_key(&body.collection, |collection| {
                        catalog_id_number(&collection.collection_id).unwrap_or(u64::MAX)
                    })
                    .is_ok_and(|position| {
                        !changed.contains(&collections.as_slice()[position].collection_id)
                    })
            });
            if old_count != reuse.bodies.as_slice().len() {
                if let Some(staging) = &mut reuse.staging {
                    staging.reclaim(reuse.bodies.as_mut_slice())?;
                }
            }
        }
        let directory = storage.path().parent().unwrap_or_else(|| Path::new("."));
        let (mut staging, old) = match reuse {
            Some((reuse, _)) => (reuse.staging, reuse.bodies),
            None => (
                None,
                ReservedVec::with_capacity(0, context, "operation_scratch")?,
            ),
        };
        let count = collections.as_slice().len();
        let mut bodies = if count != 0 && old.capacity() >= count {
            old
        } else {
            // Keep the old array and its credit together until growth succeeds.
            let mut bodies = crate::json::retry_after_memory_reclaim(
                || ReservedVec::with_capacity(count, context, "operation_scratch"),
                &mut reclaim,
            )?;
            let (old, old_work) = old.into_parts();
            for body in old {
                bodies.push(body);
            }
            drop(old_work);
            bodies
        };
        let retained = bodies.as_slice().len();
        for collection in collections.as_slice() {
            let number = catalog_id_number(&collection.collection_id)?;
            if bodies.as_slice()[..retained]
                .binary_search_by_key(&number, |body| body.collection)
                .is_err()
            {
                if staging.is_none() {
                    staging = Some(crate::json::retry_after_memory_reclaim(
                        || TextStaging::new(directory, disk.clone(), context),
                        &mut reclaim,
                    )?);
                }
                let body = build_collection_into(
                    catalog,
                    collection,
                    commit,
                    &mut load,
                    directory,
                    staging.as_mut().expect("created staging"),
                    context,
                );
                // The failed collection has rolled back; retained bodies stay in the same owner.
                let body = match body {
                    Err(error) if is_memory_budget_error(&error) && reclaim() => {
                        build_collection_into(
                            catalog,
                            collection,
                            commit,
                            &mut load,
                            directory,
                            staging.as_mut().expect("created staging"),
                            context,
                        )
                    }
                    result => result,
                }?;
                bodies.push(body);
            }
        }
        bodies
            .as_mut_slice()
            .sort_unstable_by_key(|body| body.collection);
        if count == 0 {
            staging = None;
        }
        drop(collections);
        let mut result = Self {
            staging,
            bodies,
            stats: None,
        };
        let mut input = context.workspace_reservation();
        if count != 0 {
            crate::json::retry_after_memory_reclaim(
                || input.grow_to(0, BUFFER_BYTES, "operation_scratch"),
                &mut reclaim,
            )?;
        }
        loop {
            let stats =
                TextStatsSummary::from_visitors(catalog, directory, disk, context, |id, visit| {
                    result
                        .body(id)?
                        .visit(context, input.reserved_bytes(), visit)
                });
            match stats {
                Err(error) if is_memory_budget_error(&error) && input.reserved_bytes() > 0 => {
                    input.retain(0)
                }
                Err(error) if is_memory_budget_error(&error) && reclaim() => {}
                stats => {
                    result.stats = stats?;
                    break;
                }
            }
        }
        if result.stats.is_none() {
            return Err(invalid_staging());
        }
        Ok(result)
    }

    pub(crate) fn body(&self, id: &str) -> Result<StagedText<'_>> {
        let number = catalog_id_number(id)?;
        self.bodies
            .as_slice()
            .binary_search_by_key(&number, |body| body.collection)
            .map(|position| StagedText {
                staging: self.staging.as_ref().expect("nonempty staging"),
                extent: &self.bodies.as_slice()[position],
            })
            .map_err(|_| invalid_staging())
    }

    pub(crate) fn take_stats(&mut self) -> TextStatsSummary {
        self.stats.take().expect("prepared text statistics")
    }

    pub(crate) fn has_zero_frequency_terms(&self) -> bool {
        self.stats
            .as_ref()
            .expect("prepared text statistics")
            .has_zero_frequency_terms()
    }
}

#[cfg(test)]
impl TextSpool {
    pub(crate) fn new(
        collection: u64,
        commit: u64,
        directory: &Path,
        disk: Arc<TemporaryDiskBudget>,
        context: &JsonOperationContext,
    ) -> Result<Self> {
        Ok(Self {
            staging: TextStaging::new(directory, disk, context)?,
            extent: TextExtent {
                collection,
                commit,
                offset: 0,
                length: 0,
                checksum: 0,
            },
        })
    }

    fn copy_body_from(
        &mut self,
        reader: &mut dyn Read,
        length: usize,
        commit: u64,
        context: &JsonOperationContext,
        capacity: usize,
    ) -> std::result::Result<u64, IndexReadError> {
        self.extent = self
            .staging
            .copy_body_from(reader, length, commit, context, capacity)?;
        Ok(self.extent.collection)
    }

    fn finish_copy(&mut self) -> Result<()> {
        self.staging.finish_copy(&self.extent)
    }

    fn view(&self) -> StagedText<'_> {
        StagedText {
            staging: &self.staging,
            extent: &self.extent,
        }
    }

    fn write(&self, output: &mut impl Write, commit: u64) -> Result<()> {
        self.view().write(output, commit)
    }

    fn visit(
        &self,
        context: &JsonOperationContext,
        capacity: usize,
        visit: &mut TextSegmentVisitor<'_>,
    ) -> Result<bool> {
        self.view().visit(context, capacity, visit)
    }
}

#[cfg(test)]
fn build_collection<T: AsRef<str>>(
    catalog: &Catalog,
    collection: &CollectionInfo,
    commit: u64,
    load: &mut impl FnMut(&VisibleChunk<'_>) -> Result<T>,
    directory: &Path,
    disk: &Arc<TemporaryDiskBudget>,
    context: &JsonOperationContext,
) -> Result<TextSpool> {
    let mut staging = TextStaging::new(directory, disk.clone(), context)?;
    let extent = build_collection_into(
        catalog,
        collection,
        commit,
        load,
        directory,
        &mut staging,
        context,
    )?;
    Ok(TextSpool { staging, extent })
}

fn build_collection_into<T: AsRef<str>>(
    catalog: &Catalog,
    collection: &CollectionInfo,
    commit: u64,
    load: &mut impl FnMut(&VisibleChunk<'_>) -> Result<T>,
    directory: &Path,
    staging: &mut TextStaging,
    context: &JsonOperationContext,
) -> Result<TextExtent> {
    let result = (|| {
        let offset = staging.begin()?;
        let collection_number = catalog_id_number(&collection.collection_id)?;
        let source = catalog
            .live_chunks()
            .filter(|chunk| chunk.collection_id == collection.collection_id);
        let mut chunks =
            ReservedVec::with_capacity(source.clone().count(), context, "operation_scratch")?;
        for chunk in source {
            chunks.push((chunk, 0usize));
        }
        chunks.as_mut_slice().sort_unstable_by_key(|(chunk, _)| {
            catalog_id_number(&chunk.chunk_id).unwrap_or(u64::MAX)
        });
        for pair in chunks.as_slice().windows(2) {
            if catalog_id_number(&pair[0].0.chunk_id)? == catalog_id_number(&pair[1].0.chunk_id)? {
                return Err(invalid_staging());
            }
        }
        let mut frequencies =
            ExternalTermFrequencies::new(directory, staging.disk.clone(), context);
        let mut max_term = 0;
        for (chunk, count) in chunks.as_mut_slice() {
            let visible = VisibleChunk {
                chunk,
                record_ref: catalog.record_ref_for_chunk(&chunk.chunk_id),
            };
            let text = after_spill(&mut frequencies, || load(&visible))?;
            let id = catalog_id_number(&chunk.chunk_id)?.to_be_bytes();
            crate::tokenizer::visit_tokens_in_reclaiming(
                text.as_ref(),
                context,
                &mut frequencies,
                |frequencies, token| {
                    max_term = max_term.max(token.len());
                    *count = count.checked_add(1).ok_or_else(invalid_staging)?;
                    frequencies.push(&[token.as_bytes(), &[0], &id], 1)
                },
                |frequencies| {
                    Ok(frequencies.release_empty_batch_capacity()? || frequencies.spill()?)
                },
            )?;
            drop(text);
            *count = (*count).max(1);
        }
        let mut previous = after_spill(&mut frequencies, || {
            ReservedVec::zeroed(max_term, context, "operation_scratch")
        })?;
        let mut io = context.workspace_reservation();
        let capacity = io.grow_to(0, BUFFER_BYTES, "operation_scratch")?;
        let mut writer = StagingWriter {
            output: Some(BufWriter::with_capacity(capacity, &mut staging.file)),
            cleanup: &mut staging.cleanup,
            base: offset,
            bytes: 0,
            reserved: &mut staging.reserved,
        };
        writer.append(&TEXT_SEGMENT_SCHEMA_VERSION.to_le_bytes())?;
        writer.u64(collection_number)?;
        writer.u64(commit)?;
        writer.append(&TEXT_INDEX_KIND_BM25.to_le_bytes())?;
        writer.append(&DEFAULT_EN_TOKENIZER_ID.to_le_bytes())?;
        writer.append(&(crate::index::text::BM25_K1 as f32).to_le_bytes())?;
        writer.append(&(crate::index::text::BM25_B as f32).to_le_bytes())?;
        writer.u64(chunks.as_slice().len() as u64)?;
        for (chunk, count) in chunks.as_slice() {
            writer.u64(catalog_id_number(&chunk.chunk_id)?)?;
            writer.u64(catalog_id_number(&chunk.document_id)?)?;
            writer.u64(*count as u64)?;
        }
        drop(chunks);
        let term_count_offset = writer.bytes;
        writer.u64(0)?;
        let mut previous_len = 0;
        let mut count_offset = 0;
        let mut count = 0u64;
        let mut terms = 0u64;
        // Sorting may still admit merge heads. Release the optional writer buffer
        // until the first event, when all merge and previous-term storage is live.
        writer.set_capacity(0)?;
        io.retain(0);
        writer.release_unused_disk();
        frequencies.finish_batch(|key, frequency| {
            let split = key.len().checked_sub(9).ok_or_else(invalid_staging)?;
            if key[split] != 0 || frequency == 0 {
                return Err(invalid_staging());
            }
            let term = std::str::from_utf8(&key[..split]).map_err(|_| invalid_staging())?;
            if terms == 0 || &previous.as_slice()[..previous_len] != term.as_bytes() {
                if terms != 0 {
                    writer.patch_counts(count_offset, count)?;
                } else {
                    let capacity = io.grow_to(0, BUFFER_BYTES, "operation_scratch")?;
                    writer.set_capacity(capacity)?;
                }
                previous.as_mut_slice()[..term.len()].copy_from_slice(term.as_bytes());
                previous_len = term.len();
                writer.string(term)?;
                count_offset = writer.bytes;
                writer.append(&[0; 16])?;
                count = 0;
                terms = terms.checked_add(1).ok_or_else(invalid_staging)?;
            }
            let chunk =
                u64::from_be_bytes(key[split + 1..].try_into().map_err(|_| invalid_staging())?);
            writer.u64(chunk)?;
            writer.u64(frequency)?;
            count = count.checked_add(1).ok_or_else(invalid_staging)?;
            Ok(())
        })?;
        if terms != 0 {
            writer.patch_counts(count_offset, count)?;
        }
        writer.patch(term_count_offset, &terms.to_le_bytes())?;
        writer.output().flush()?;
        let bytes = writer.bytes;
        writer.release_unused_disk();
        drop(writer);
        drop(io);
        staging.file.seek(SeekFrom::Start(offset))?;
        let mut buffer = [0; 8192];
        let mut remaining = bytes;
        let mut crc = Crc32c::default();
        while remaining != 0 {
            let count = remaining.min(buffer.len() as u64) as usize;
            staging.file.read_exact(&mut buffer[..count])?;
            crc.update(&buffer[..count]);
            remaining -= count as u64;
        }
        let length = bytes.checked_add(4).ok_or_else(invalid_staging)?;
        let end = offset.checked_add(length).ok_or_else(invalid_staging)?;
        let reserved = staging
            .reserved
            .checked_add(4)
            .ok_or_else(invalid_staging)?;
        staging.cleanup.reserve_disk(4)?;
        staging.reserved = reserved;
        staging.file.write_all(&crc.finish().to_le_bytes())?;
        let body = TextExtent {
            collection: collection_number,
            commit,
            offset,
            length,
            checksum: crc.finish(),
        };
        staging.bytes = end;
        Ok(body)
    })();
    if result.is_err() {
        staging.rollback()?;
    }
    result
}

struct StagingWriter<'a> {
    output: Option<BufWriter<&'a mut TemporaryFile>>,
    cleanup: &'a mut TempFileGuard,
    base: u64,
    bytes: u64,
    reserved: &'a mut u64,
}

impl<'a> StagingWriter<'a> {
    fn release_unused_disk(&mut self) {
        let end = self.base + self.bytes;
        self.cleanup.release_disk(*self.reserved - end);
        *self.reserved = end;
    }
    fn output(&mut self) -> &mut BufWriter<&'a mut TemporaryFile> {
        self.output.as_mut().expect("live staging writer")
    }
    fn set_capacity(&mut self, capacity: usize) -> Result<()> {
        self.output().flush()?;
        let (file, buffer) = self
            .output
            .take()
            .expect("live staging writer")
            .into_parts();
        drop(buffer);
        self.output = Some(BufWriter::with_capacity(capacity, file));
        Ok(())
    }
    fn append(&mut self, bytes: &[u8]) -> Result<()> {
        let end = self
            .bytes
            .checked_add(bytes.len() as u64)
            .ok_or_else(invalid_staging)?;
        let absolute = self.base.checked_add(end).ok_or_else(invalid_staging)?;
        if absolute > *self.reserved {
            *self.reserved += self
                .cleanup
                .reserve_disk_buffer(absolute - *self.reserved, BUFFER_BYTES as u64)?;
        }
        self.output().write_all(bytes)?;
        self.bytes = end;
        Ok(())
    }
    fn u64(&mut self, value: u64) -> Result<()> {
        self.append(&value.to_le_bytes())
    }
    fn string(&mut self, value: &str) -> Result<()> {
        let length = u32::try_from(value.len()).map_err(|_| invalid_staging())?;
        self.append(&length.to_le_bytes())?;
        self.append(value.as_bytes())
    }
    fn patch(&mut self, offset: u64, bytes: &[u8]) -> Result<()> {
        if offset
            .checked_add(bytes.len() as u64)
            .map_or(true, |end| end > self.bytes)
        {
            return Err(invalid_staging());
        }
        let absolute = self.base.checked_add(offset).ok_or_else(invalid_staging)?;
        self.output().seek(SeekFrom::Start(absolute))?;
        self.output().write_all(bytes)?;
        let end = self.base + self.bytes;
        self.output().seek(SeekFrom::Start(end))?;
        Ok(())
    }
    fn patch_counts(&mut self, offset: u64, count: u64) -> Result<()> {
        let mut bytes = [0; 16];
        bytes[..8].copy_from_slice(&count.to_le_bytes());
        bytes[8..].copy_from_slice(&count.to_le_bytes());
        self.patch(offset, &bytes)
    }
}

impl Drop for StagingWriter<'_> {
    fn drop(&mut self) {
        if let Some(output) = self.output.take() {
            drop(output.into_parts().1);
        }
    }
}

struct CopyReader<'a> {
    source: &'a mut dyn Read,
    output: &'a mut dyn Write,
    crc: Crc32c,
    bytes: u64,
}
impl Read for CopyReader<'_> {
    fn read(&mut self, bytes: &mut [u8]) -> std::io::Result<usize> {
        let count = self.source.read(bytes)?;
        self.output.write_all(&bytes[..count])?;
        self.crc.update(&bytes[..count]);
        self.bytes += count as u64;
        Ok(count)
    }
}
struct CrcReader<R> {
    source: R,
    crc: Crc32c,
}
impl<R: Read> Read for CrcReader<R> {
    fn read(&mut self, bytes: &mut [u8]) -> std::io::Result<usize> {
        let count = self.source.read(bytes)?;
        self.crc.update(&bytes[..count]);
        Ok(count)
    }
}
fn invalid_staging() -> Error {
    Error::new(ErrorCode::FileFormatError, "invalid staged text checkpoint")
}

fn staging_checksum_error() -> Error {
    Error::new(ErrorCode::ChecksumMismatch, "text staging checksum changed")
}

fn after_spill<T>(
    frequencies: &mut ExternalTermFrequencies<'_>,
    mut attempt: impl FnMut() -> Result<T>,
) -> Result<T> {
    match attempt() {
        Err(error) if is_memory_budget_error(&error) => {
            if frequencies.spill()? | frequencies.release_empty_batch_capacity()? {
                attempt()
            } else {
                Err(error)
            }
        }
        result => result,
    }
}
fn staging_read_error(error: IndexReadError) -> Error {
    match error {
        IndexReadError::Io(error) => error.into(),
        IndexReadError::Resource(error) => error,
        IndexReadError::Invalid(_) => invalid_staging(),
    }
}
