use std::collections::{BTreeMap, BTreeSet};

use crate::error::{Error, ErrorCode, Result};
#[cfg(test)]
use crate::model::Tombstone;
use crate::model::{
    Chunk, CollectionInfo, CollectionStats, DatabaseStats, Document, Embedding, IndexStats,
};
use crate::profile::ResourceProfile;
use crate::storage::{CommittedRecord, RecordPayload, RecordRef};

mod aliases;
use aliases::ExternalIds;
mod mutation;

pub(crate) struct DocumentChunks<'a> {
    ordered: crate::json::ReservedVec<(usize, &'a Chunk)>,
}

impl<'a> DocumentChunks<'a> {
    pub(crate) fn len(&self) -> usize {
        self.ordered.as_slice().len()
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = &'a Chunk> + '_ {
        self.ordered.as_slice().iter().map(|(_, chunk)| *chunk)
    }
}

#[cfg_attr(test, derive(Clone))]
#[derive(Debug, Default)]
pub struct Catalog {
    pub collections: BTreeMap<String, CollectionInfo>,
    pub collection_names: BTreeMap<String, String>,
    pub reserved_collection_names: BTreeSet<String>,
    // Box large summaries so spare tree slots only reserve a pointer, not a full model.
    pub documents: BTreeMap<String, Box<Document>>,
    pub(crate) document_external_ids: ExternalIds,
    pub chunks: BTreeMap<String, Box<Chunk>>,
    pub(crate) chunk_external_ids: ExternalIds,
    pub document_chunks: BTreeMap<String, Vec<String>>,
    pub embeddings: BTreeMap<String, Box<Embedding>>,
    pub chunk_embeddings: BTreeMap<String, String>,
    pub record_refs: BTreeMap<String, RecordRef>,
    pub id_high_watermarks: BTreeMap<String, u64>,
    business_record_count: usize,
    tombstone_count: usize,
    collection_tombstones: BTreeMap<String, usize>,
    document_tombstones: BTreeMap<String, usize>,
    chunk_tombstones: BTreeMap<String, usize>,
    pub deleted_collections: BTreeSet<String>,
    pub collection_drop_times: BTreeMap<String, String>,
    pub deleted_documents: BTreeSet<String>,
    pub deleted_chunks: BTreeSet<String>,
    pub last_commit_id: u64,
    memory: Option<mutation::CatalogMemory>,
}

impl Catalog {
    #[cfg(test)]
    pub(crate) fn state_signature(&self) -> String {
        let mut snapshot = self.clone();
        snapshot.memory = None;
        format!("{snapshot:?}")
    }

    #[cfg(test)]
    pub fn from_committed_records(
        records: Vec<CommittedRecord>,
        last_commit_id: u64,
    ) -> Result<Self> {
        let mut catalog = Catalog {
            last_commit_id,
            ..Catalog::default()
        };
        for record in records {
            catalog.apply_owned_committed(record)?;
        }
        Ok(catalog)
    }

    #[cfg(test)]
    pub fn apply_owned_committed(&mut self, record: CommittedRecord) -> Result<()> {
        self.apply_with_ref(record.payload, Some(record.record_ref))
    }

    #[cfg(test)]
    fn apply_with_ref(
        &mut self,
        record: RecordPayload,
        record_ref: Option<RecordRef>,
    ) -> Result<()> {
        // Count committed history, including superseded versions, not just live catalog entries.
        let business_record_count = match &record {
            RecordPayload::CollectionCreate(_)
            | RecordPayload::CollectionDrop { .. }
            | RecordPayload::DocumentPut(_)
            | RecordPayload::ChunkPut(_)
            | RecordPayload::EmbeddingPut(_)
            | RecordPayload::Tombstone(_) => {
                self.business_record_count.checked_add(1).ok_or_else(|| {
                    Error::new(
                        ErrorCode::FileFormatError,
                        "catalog business record count overflow",
                    )
                })?
            }
            RecordPayload::CompactMarker(_) | RecordPayload::Commit(_) => {
                self.business_record_count
            }
        };
        match record {
            RecordPayload::CollectionCreate(collection) => {
                self.track_id(&collection.collection_id);
                if let Some(record_ref) = record_ref {
                    self.record_refs
                        .insert(collection.collection_id.clone(), record_ref);
                }
                self.collection_names
                    .insert(collection.name.clone(), collection.collection_id.clone());
                self.deleted_collections.remove(&collection.collection_id);
                self.collections
                    .insert(collection.collection_id.clone(), collection);
            }
            RecordPayload::CollectionDrop {
                collection_id,
                dropped_at,
                ..
            } => {
                if let Some(collection) = self.collections.get(&collection_id) {
                    self.reserved_collection_names
                        .insert(collection.name.clone());
                }
                self.collection_drop_times
                    .insert(collection_id.clone(), dropped_at);
                self.deleted_collections.insert(collection_id);
            }
            RecordPayload::DocumentPut(document) => {
                self.track_id(&document.document_id);
                if let Some(previous) = self.documents.get(&document.document_id) {
                    if previous.collection_id != document.collection_id
                        || previous.external_id != document.external_id
                    {
                        self.document_external_ids.remove_if_owned(
                            &previous.collection_id,
                            &previous.external_id,
                            &document.document_id,
                        );
                    }
                }
                if let Some(record_ref) = record_ref {
                    self.record_refs
                        .insert(document.document_id.clone(), record_ref);
                }
                self.deleted_documents.remove(&document.document_id);
                self.document_external_ids.insert(
                    &document.collection_id,
                    &document.external_id,
                    &document.document_id,
                );
                self.documents
                    .insert(document.document_id.clone(), Box::new(document));
            }
            RecordPayload::ChunkPut(chunk) => {
                self.track_id(&chunk.chunk_id);
                let previous = self.chunks.get(&chunk.chunk_id);
                if let Some(previous) = previous {
                    if previous.collection_id != chunk.collection_id
                        || previous.chunk_external_id != chunk.chunk_external_id
                    {
                        self.chunk_external_ids.remove_if_owned(
                            &previous.collection_id,
                            &previous.chunk_external_id,
                            &chunk.chunk_id,
                        );
                    }
                }
                if previous.map(|old| &old.document_id) != Some(&chunk.document_id) {
                    if let Some(previous) = previous {
                        if let Some(children) = self.document_chunks.get_mut(&previous.document_id)
                        {
                            children.retain(|id| id != &chunk.chunk_id);
                            if children.is_empty() {
                                self.document_chunks.remove(&previous.document_id);
                            }
                        }
                    }
                    self.document_chunks
                        .entry(chunk.document_id.clone())
                        .or_default()
                        .push(chunk.chunk_id.clone());
                }
                if let Some(record_ref) = record_ref {
                    self.record_refs.insert(chunk.chunk_id.clone(), record_ref);
                }
                let mut chunk_summary = chunk;
                chunk_summary.text = String::new();
                self.deleted_chunks.remove(&chunk_summary.chunk_id);
                self.chunk_external_ids.insert(
                    &chunk_summary.collection_id,
                    &chunk_summary.chunk_external_id,
                    &chunk_summary.chunk_id,
                );
                self.chunks
                    .insert(chunk_summary.chunk_id.clone(), Box::new(chunk_summary));
            }
            RecordPayload::EmbeddingPut(embedding) => {
                self.track_id(&embedding.embedding_id);
                if let Some(record_ref) = record_ref {
                    self.record_refs
                        .insert(embedding.embedding_id.clone(), record_ref);
                }
                let mut embedding_summary = embedding;
                if let Some(chunk) = self.chunks.get(&embedding_summary.chunk_id) {
                    embedding_summary.chunk_external_id = chunk.chunk_external_id.clone();
                }
                embedding_summary.vector = Vec::new();
                if let Some(chunk) = self.chunks.get_mut(&embedding_summary.chunk_id) {
                    chunk.embedding_id = Some(embedding_summary.embedding_id.clone());
                }
                if let Some(previous) = self.chunk_embeddings.insert(
                    embedding_summary.chunk_id.clone(),
                    embedding_summary.embedding_id.clone(),
                ) {
                    // The primary log retains history; no catalog reader uses superseded embeddings.
                    if previous != embedding_summary.embedding_id {
                        self.embeddings.remove(&previous);
                        self.record_refs.remove(&previous);
                    }
                }
                self.embeddings.insert(
                    embedding_summary.embedding_id.clone(),
                    Box::new(embedding_summary),
                );
            }
            RecordPayload::Tombstone(tombstone) => {
                let next_count = self.tombstone_count.checked_add(1).ok_or_else(|| {
                    Error::new(
                        ErrorCode::FileFormatError,
                        "catalog tombstone count overflow",
                    )
                })?;
                self.track_id(&tombstone.target_id);
                match tombstone.target_type.as_str() {
                    "collection" => {
                        *self
                            .collection_tombstones
                            .entry(tombstone.target_id.clone())
                            .or_default() += 1;
                        self.deleted_collections.insert(tombstone.target_id);
                    }
                    "document" => {
                        // Only an explicit Put can revive this object and supplies
                        // its payload again. Keep identity/relations for diagnosis;
                        // the primary log, not this summary, owns deleted history.
                        if let Some(document) = self.documents.get_mut(&tombstone.target_id) {
                            document.title = None;
                            document.source_uri = None;
                            document.metadata = crate::JsonValue::object();
                            document.created_at = String::new();
                            document.updated_at = String::new();
                            self.record_refs.remove(&tombstone.target_id);
                        }
                        *self
                            .document_tombstones
                            .entry(tombstone.target_id.clone())
                            .or_default() += 1;
                        self.deleted_documents.insert(tombstone.target_id);
                    }
                    "chunk" => {
                        if let Some(chunk) = self.chunks.get_mut(&tombstone.target_id) {
                            chunk.metadata = crate::JsonValue::object();
                            chunk.text_hash = String::new();
                            chunk.created_at = String::new();
                            self.record_refs.remove(&tombstone.target_id);
                        }
                        *self
                            .chunk_tombstones
                            .entry(tombstone.target_id.clone())
                            .or_default() += 1;
                        self.deleted_chunks.insert(tombstone.target_id);
                    }
                    _ => {}
                }
                self.tombstone_count = next_count;
            }
            RecordPayload::CompactMarker(marker) => {
                for (prefix, watermark) in marker.id_high_watermarks {
                    self.id_high_watermarks
                        .entry(prefix)
                        .and_modify(|seen| *seen = (*seen).max(watermark))
                        .or_insert(watermark);
                }
            }
            RecordPayload::Commit(_) => {}
        }
        self.business_record_count = business_record_count;
        Ok(())
    }

    pub(crate) fn business_record_count(&self) -> usize {
        self.business_record_count
    }

    pub fn collection_by_name(&self, name: &str) -> Result<&CollectionInfo> {
        let id = self.collection_names.get(name).ok_or_else(|| {
            Error::new(
                ErrorCode::CollectionNotFound,
                format!("collection {name} not found"),
            )
        })?;
        self.collection_by_id(id)
    }

    pub fn collection_by_id(&self, collection_id: &str) -> Result<&CollectionInfo> {
        let collection = self.collections.get(collection_id).ok_or_else(|| {
            Error::new(
                ErrorCode::CollectionNotFound,
                format!("collection {collection_id} not found"),
            )
        })?;
        if self.deleted_collections.contains(collection_id) {
            return Err(Error::new(
                ErrorCode::CollectionNotFound,
                format!("collection {} was deleted", collection.name),
            ));
        }
        Ok(collection)
    }

    pub fn check_collection_name_available(&self, name: &str) -> Result<()> {
        if self
            .collection_names
            .get(name)
            .map(|id| !self.deleted_collections.contains(id))
            .unwrap_or(false)
        {
            return Err(Error::new(
                ErrorCode::CollectionAlreadyExists,
                format!("collection {name} already exists"),
            ));
        }
        if self.reserved_collection_names.contains(name)
            || self
                .collection_names
                .get(name)
                .map(|id| self.deleted_collections.contains(id))
                .unwrap_or(false)
        {
            return Err(Error::new(
                ErrorCode::CollectionDeletedNameReserved,
                format!("deleted collection name {name} is reserved in v0.1"),
            ));
        }
        Ok(())
    }

    pub fn document_by_id_ref(&self, collection_id: &str, document_id: &str) -> Result<&Document> {
        let document = self.documents.get(document_id).ok_or_else(|| {
            Error::new(
                ErrorCode::DocumentNotFound,
                format!("document {document_id} not found"),
            )
        })?;
        if document.collection_id != collection_id || !self.is_document_live(document_id) {
            return Err(Error::new(
                ErrorCode::DocumentNotFound,
                format!("document {document_id} not found"),
            ));
        }
        Ok(document)
    }

    pub fn document_by_external_id(
        &self,
        collection_id: &str,
        external_id: &str,
    ) -> Result<&Document> {
        let id = self
            .document_external_ids
            .get(collection_id, external_id)
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::DocumentNotFound,
                    format!("document external_id {external_id} not found"),
                )
            })?;
        self.document_by_id_ref(collection_id, id)
    }

    pub fn chunk_by_id(&self, collection_id: &str, chunk_id: &str) -> Result<&Chunk> {
        let chunk = self.chunks.get(chunk_id).ok_or_else(|| {
            Error::new(
                ErrorCode::ChunkNotFound,
                format!("chunk {chunk_id} not found"),
            )
        })?;
        if chunk.collection_id != collection_id || !self.is_chunk_live(chunk_id) {
            return Err(Error::new(
                ErrorCode::ChunkNotFound,
                format!("chunk {chunk_id} not found"),
            ));
        }
        Ok(chunk)
    }

    pub fn chunk_by_external_id(
        &self,
        collection_id: &str,
        chunk_external_id: &str,
    ) -> Result<&Chunk> {
        let id = self
            .chunk_external_ids
            .get(collection_id, chunk_external_id)
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::ChunkNotFound,
                    format!("chunk_external_id {chunk_external_id} not found"),
                )
            })?;
        self.chunk_by_id(collection_id, id)
    }

    pub(crate) fn chunks_for_document_in(
        &self,
        collection_id: &str,
        document_id: &str,
        context: &crate::json::JsonOperationContext,
    ) -> Result<DocumentChunks<'_>> {
        self.document_by_id_ref(collection_id, document_id)?;
        let count = self.live_document_chunks(document_id).count();
        let mut ordered =
            crate::json::ReservedVec::with_capacity(count, context, "operation_scratch")?;
        for (position, chunk) in self.live_document_chunks(document_id).enumerate() {
            ordered.push((position, chunk));
        }
        // Remember source order to preserve stable ties without a hidden sort allocation.
        ordered
            .as_mut_slice()
            .sort_unstable_by_key(|(position, chunk)| (chunk.ordinal, *position));
        Ok(DocumentChunks { ordered })
    }

    pub(crate) fn chunk_count_for_document(
        &self,
        collection_id: &str,
        document_id: &str,
    ) -> Result<usize> {
        self.document_by_id_ref(collection_id, document_id)?;
        Ok(self.live_document_chunks(document_id).count())
    }

    fn live_document_chunks(&self, document_id: &str) -> impl Iterator<Item = &Chunk> {
        self.document_chunks
            .get(document_id)
            .into_iter()
            .flat_map(|ids| ids.iter())
            .filter(|id| self.is_chunk_live(id))
            .filter_map(|id| self.chunks.get(id))
            .map(Box::as_ref)
    }

    pub fn embedding_for_chunk(&self, chunk_id: &str) -> Option<&Embedding> {
        self.chunk_embeddings
            .get(chunk_id)
            .and_then(|id| self.embeddings.get(id))
            .map(Box::as_ref)
    }

    pub fn record_ref_for_chunk(&self, chunk_id: &str) -> Option<&RecordRef> {
        self.record_refs.get(chunk_id)
    }

    pub fn record_ref_for_embedding(&self, embedding_id: &str) -> Option<&RecordRef> {
        self.record_refs.get(embedding_id)
    }

    pub fn visible_snapshot(&self) -> VisibleSnapshot<'_> {
        VisibleSnapshot { catalog: self }
    }

    pub fn is_collection_live(&self, collection_id: &str) -> bool {
        self.collections.contains_key(collection_id)
            && !self.deleted_collections.contains(collection_id)
    }

    pub fn is_document_live(&self, document_id: &str) -> bool {
        let Some(document) = self.documents.get(document_id) else {
            return false;
        };
        self.is_collection_live(&document.collection_id)
            && !self.deleted_documents.contains(document_id)
    }

    pub fn is_chunk_live(&self, chunk_id: &str) -> bool {
        let Some(chunk) = self.chunks.get(chunk_id) else {
            return false;
        };
        self.is_document_live(&chunk.document_id) && !self.deleted_chunks.contains(chunk_id)
    }

    pub fn live_collections(&self) -> impl Iterator<Item = &CollectionInfo> {
        self.collections
            .values()
            .filter(|collection| self.is_collection_live(&collection.collection_id))
    }

    pub fn live_documents(&self) -> impl Iterator<Item = &Document> {
        self.documents
            .values()
            .map(Box::as_ref)
            .filter(|document| self.is_document_live(&document.document_id))
    }

    pub fn live_chunks(&self) -> impl Iterator<Item = &Chunk> + Clone {
        self.chunks
            .values()
            .map(Box::as_ref)
            .filter(|chunk| self.is_chunk_live(&chunk.chunk_id))
    }

    pub fn live_embeddings(&self) -> impl Iterator<Item = &Embedding> + Clone {
        self.embeddings
            .values()
            .map(Box::as_ref)
            .filter(|embedding| {
                self.is_chunk_live(&embedding.chunk_id)
                    && self
                        .chunk_embeddings
                        .get(&embedding.chunk_id)
                        .map(|id| id == &embedding.embedding_id)
                        .unwrap_or(false)
            })
    }

    pub fn next_id(&self, prefix: &str, offset: u64) -> Result<String> {
        let next = self.next_id_number(prefix, offset)?;
        Ok(format!("{prefix}{next:016}"))
    }

    pub(crate) fn next_id_in(
        &self,
        prefix: &str,
        offset: u64,
        context: &crate::json::JsonOperationContext,
    ) -> Result<crate::json::ReservedString> {
        let next = self.next_id_number(prefix, offset)?;
        let digits = (next.ilog10() as usize + 1).max(16);
        crate::json::ReservedString::build_exact(
            prefix.len() + digits,
            context,
            "batch_prepare",
            |out| write!(out, "{prefix}{next:016}"),
        )
    }

    fn next_id_number(&self, prefix: &str, offset: u64) -> Result<u64> {
        self.id_high_watermarks
            .get(prefix)
            .copied()
            .unwrap_or(0)
            .checked_add(1)
            .and_then(|base| base.checked_add(offset))
            .ok_or_else(|| {
                Error::new(ErrorCode::FileFormatError, "internal ID space is exhausted")
                    .with_detail("reason", "id_space_exhausted")
                    .with_detail("id_kind", prefix)
            })
    }

    #[cfg(test)]
    fn track_id(&mut self, id: &str) {
        for prefix in ["col_", "doc_", "chk_", "emb_"] {
            if let Some(rest) = id.strip_prefix(prefix) {
                if let Ok(value) = rest.parse::<u64>() {
                    self.id_high_watermarks
                        .entry(prefix.into())
                        .and_modify(|seen| *seen = (*seen).max(value))
                        .or_insert(value);
                }
                break;
            }
        }
    }

    pub fn stats(
        &self,
        file_size: u64,
        profile: ResourceProfile,
        cache_budget_mb: usize,
        index: &IndexStats,
    ) -> DatabaseStats {
        DatabaseStats {
            collection_count: self.live_collections().count(),
            document_count: self.live_documents().count(),
            chunk_count: self.live_chunks().count(),
            embedding_count: self.live_embeddings().count(),
            tombstone_count: self.tombstone_count,
            file_size,
            index_size_bytes: index.index_size_bytes,
            live_bytes: self.estimated_live_bytes(),
            dead_bytes_estimate: self.tombstone_count as u64 * 128,
            active_profile: profile,
            cache_budget_mb,
            last_commit_id: self.last_commit_id,
            pending_compaction_bytes: self.tombstone_count as u64 * 128
                + index.index_size_bytes / 10,
            format_version: crate::storage::FORMAT_VERSION,
            background_compaction_enabled: false,
        }
    }

    pub fn collection_stats(
        &self,
        collection_id: &str,
        profile: ResourceProfile,
        text_terms: usize,
        vector_count: usize,
        copy_name: impl FnOnce(&str) -> Result<String>,
    ) -> Result<CollectionStats> {
        let collection = self.collection_by_id(collection_id)?;
        let name = copy_name(&collection.name)?;
        let document_count = self
            .live_documents()
            .filter(|doc| doc.collection_id == collection_id)
            .count();
        let chunk_count = self
            .live_chunks()
            .filter(|chunk| chunk.collection_id == collection_id)
            .count();
        let embedding_count = self
            .live_embeddings()
            .filter(|embedding| embedding.collection_id == collection_id)
            .count();
        let tombstone_count = self.tombstone_count_for_collection(collection_id);
        Ok(CollectionStats {
            tokenizer: collection.tokenizer.clone(),
            collection: name,
            document_count,
            chunk_count,
            embedding_count,
            tombstone_count,
            text_index_terms: text_terms,
            text_index_segments: 0,
            vector_count,
            index_size_bytes: 0,
            live_bytes: self.estimated_live_bytes_for_collection(collection_id),
            dead_bytes_estimate: tombstone_count as u64 * 128,
            resource_profile: profile,
            cache_budget_mb: 0,
            last_commit_id: self.last_commit_id,
            pending_compaction_bytes: tombstone_count as u64 * 128,
            format_version: crate::storage::FORMAT_VERSION,
            background_compaction_enabled: false,
        })
    }

    pub fn tombstone_count(&self) -> usize {
        self.tombstone_count
    }

    fn tombstone_count_for_collection(&self, collection_id: &str) -> usize {
        // Resolve membership at read time, including targets created after their tombstone.
        self.collection_tombstones
            .get(collection_id)
            .copied()
            .unwrap_or(0)
            + self
                .document_tombstones
                .iter()
                .filter(|(id, _)| {
                    self.documents
                        .get(*id)
                        .is_some_and(|document| document.collection_id == collection_id)
                })
                .map(|(_, count)| count)
                .sum::<usize>()
            + self
                .chunk_tombstones
                .iter()
                .filter(|(id, _)| {
                    self.chunks
                        .get(*id)
                        .is_some_and(|chunk| chunk.collection_id == collection_id)
                })
                .map(|(_, count)| count)
                .sum::<usize>()
    }

    #[cfg(test)]
    fn tombstone_belongs_to_collection(&self, tombstone: &Tombstone, collection_id: &str) -> bool {
        match tombstone.target_type.as_str() {
            "collection" => tombstone.target_id == collection_id,
            "document" => self
                .documents
                .get(&tombstone.target_id)
                .map(|doc| doc.collection_id == collection_id)
                .unwrap_or(false),
            "chunk" => self
                .chunks
                .get(&tombstone.target_id)
                .map(|chunk| chunk.collection_id == collection_id)
                .unwrap_or(false),
            _ => false,
        }
    }

    pub(crate) fn estimated_live_bytes_for_collection(&self, collection_id: &str) -> u64 {
        let text_bytes = self
            .live_chunks()
            .filter(|chunk| chunk.collection_id == collection_id)
            .map(|chunk| {
                self.record_ref_for_chunk(&chunk.chunk_id)
                    .map(|record_ref| record_ref.payload.len)
                    .unwrap_or_else(|| {
                        chunk.text.len() as u64 + chunk.metadata.to_compact_string().len() as u64
                    })
            })
            .sum::<u64>();
        let vector_bytes = self
            .live_embeddings()
            .filter(|embedding| embedding.collection_id == collection_id)
            .map(|embedding| {
                self.record_ref_for_embedding(&embedding.embedding_id)
                    .map(|record_ref| record_ref.payload.len)
                    .unwrap_or_else(|| embedding.vector.len() as u64 * 4)
            })
            .sum::<u64>();
        text_bytes + vector_bytes
    }

    fn estimated_live_bytes(&self) -> u64 {
        self.live_collections()
            .map(|collection| self.estimated_live_bytes_for_collection(&collection.collection_id))
            .sum()
    }
}

#[derive(Clone, Copy)]
pub struct VisibleSnapshot<'a> {
    catalog: &'a Catalog,
}

pub struct VisibleChunk<'a> {
    pub chunk: &'a Chunk,
    pub record_ref: Option<&'a RecordRef>,
}

pub struct VisibleEmbedding<'a> {
    pub embedding: &'a Embedding,
    pub record_ref: Option<&'a RecordRef>,
}

impl<'a> VisibleSnapshot<'a> {
    pub fn live_chunks(&self) -> impl Iterator<Item = VisibleChunk<'a>> + 'a {
        let catalog = self.catalog;
        catalog
            .chunks
            .values()
            .filter(move |chunk| catalog.is_chunk_live(&chunk.chunk_id))
            .map(move |chunk| VisibleChunk {
                chunk,
                record_ref: catalog.record_ref_for_chunk(&chunk.chunk_id),
            })
    }

    pub fn live_embeddings(&self) -> impl Iterator<Item = VisibleEmbedding<'a>> + 'a {
        let catalog = self.catalog;
        catalog
            .embeddings
            .values()
            .filter(move |embedding| {
                catalog.is_chunk_live(&embedding.chunk_id)
                    && catalog
                        .chunk_embeddings
                        .get(&embedding.chunk_id)
                        .map(|id| id == &embedding.embedding_id)
                        .unwrap_or(false)
            })
            .map(move |embedding| VisibleEmbedding {
                embedding,
                record_ref: catalog.record_ref_for_embedding(&embedding.embedding_id),
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{DistanceMetric, VectorIndexConfig};
    use crate::storage::PayloadRef;

    pub(super) fn collection(number: u64) -> CollectionInfo {
        CollectionInfo {
            collection_id: format!("col_{number}"),
            name: format!("collection-{number}"),
            tokenizer: Default::default(),
            embedding_dim: 2,
            distance_metric: DistanceMetric::Cosine,
            embedding_model: None,
            vectors_normalized: false,
            vector_index: VectorIndexConfig::Exact,
            created_at: "1970-01-01T00:00:00Z".into(),
        }
    }
    pub(super) fn document(number: u64, owner: &str) -> Document {
        Document {
            document_id: format!("doc_{number}"),
            collection_id: owner.into(),
            external_id: format!("document-{number}"),
            title: None,
            source_uri: None,
            metadata: crate::JsonValue::object(),
            created_at: "1970-01-01T00:00:00Z".into(),
            updated_at: "1970-01-01T00:00:00Z".into(),
        }
    }
    pub(super) fn chunk(number: u64, owner: &str) -> Chunk {
        Chunk {
            chunk_id: format!("chk_{number}"),
            document_id: format!("doc_{number}"),
            collection_id: owner.into(),
            chunk_external_id: format!("chunk-{number}"),
            text: "stored text".into(),
            text_hash: "hash".into(),
            ordinal: 0,
            token_count: 2,
            metadata: crate::JsonValue::object(),
            embedding_id: None,
            source_span: None,
            created_at: "1970-01-01T00:00:00Z".into(),
        }
    }

    #[test]
    fn document_chunk_ordering_reserves_workspace_and_preserves_stable_ties() {
        use crate::json::{JsonOperationContext, JsonProcessingPolicyV1};

        for seed in 0..100usize {
            let mut catalog = Catalog::default();
            catalog
                .apply_with_ref(RecordPayload::CollectionCreate(collection(1)), None)
                .unwrap();
            catalog
                .apply_with_ref(RecordPayload::DocumentPut(document(1, "col_1")), None)
                .unwrap();
            let mut expected = Vec::new();
            for id in (1..=seed + 8).rev() {
                let mut child = chunk(id as u64, "col_1");
                child.document_id = "doc_1".into();
                child.ordinal = (id * 7 + seed) % 5;
                if id % 7 != 0 {
                    expected.push((child.ordinal, child.chunk_id.clone()));
                }
                let deleted_id = (id % 7 == 0).then(|| child.chunk_id.clone());
                catalog
                    .apply_with_ref(RecordPayload::ChunkPut(child), None)
                    .unwrap();
                if let Some(deleted_id) = deleted_id {
                    catalog.deleted_chunks.insert(deleted_id);
                }
            }
            expected.sort_by_key(|(ordinal, _)| *ordinal);
            let bytes = expected.len() * std::mem::size_of::<(usize, &Chunk)>();
            let limit = JsonProcessingPolicyV1::WORK_BUDGET_BYTES;
            for available in [bytes - 1, bytes, bytes + seed] {
                let context = JsonOperationContext::standalone();
                let mut held = context.workspace_reservation();
                held.grow_to(limit - available, limit - available, "operation_scratch")
                    .unwrap();
                let result = catalog.chunks_for_document_in("col_1", "doc_1", &context);
                if available < bytes {
                    let error = result.err().unwrap();
                    assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                    assert!(error
                        .detail()
                        .iter()
                        .any(|(key, value)| key == "operation" && value == "operation_scratch"));
                } else {
                    let chunks = result.unwrap();
                    assert_eq!(chunks.len(), expected.len());
                    assert_eq!(
                        chunks
                            .iter()
                            .map(|chunk| (chunk.ordinal, chunk.chunk_id.clone()))
                            .collect::<Vec<_>>(),
                        expected
                    );
                    let mut rest = context.workspace_reservation();
                    rest.grow_to(available - bytes, available - bytes, "operation_scratch")
                        .unwrap();
                    assert!(rest
                        .grow_to(
                            available - bytes + 1,
                            available - bytes + 1,
                            "operation_scratch"
                        )
                        .is_err());
                }
                context
                    .workspace_reservation()
                    .grow_to(available, available, "operation_scratch")
                    .unwrap();
                assert_eq!(
                    catalog
                        .chunks_for_document_in("col_2", "doc_1", &context)
                        .err()
                        .unwrap()
                        .code(),
                    ErrorCode::DocumentNotFound
                );
            }
        }
    }

    #[test]
    fn boxed_summaries_keep_identity_during_growth_and_isolate_cloned_catalogs() {
        for seed in 0..100 {
            let mut catalog = Catalog::default();
            catalog
                .apply_with_ref(RecordPayload::CollectionCreate(collection(1)), None)
                .unwrap();
            let mut first_addresses = None;
            for number in 1..=13 + seed {
                let mut doc = document(number, "col_1");
                doc.title = Some(format!("title-{seed}-{number}"));
                doc.metadata = crate::json::object_from_pairs([
                    ("number", crate::JsonValue::Number(number as f64)),
                    (
                        "payload",
                        crate::JsonValue::String("x".repeat(64 + seed as usize)),
                    ),
                ]);
                let child = chunk(number, "col_1");
                let embedding = Embedding {
                    embedding_id: format!("emb_{number}"),
                    chunk_id: child.chunk_id.clone(),
                    collection_id: "col_1".into(),
                    chunk_external_id: child.chunk_external_id.clone(),
                    model_name: "model".into(),
                    dim: 2,
                    vector: vec![1.0, number as f32],
                    created_at: String::new(),
                };
                for record in [
                    RecordPayload::DocumentPut(doc),
                    RecordPayload::ChunkPut(child),
                    RecordPayload::EmbeddingPut(embedding),
                ] {
                    catalog.apply_with_ref(record, None).unwrap();
                }
                let addresses = (
                    catalog.documents["doc_1"].as_ref() as *const Document,
                    catalog.chunks["chk_1"].as_ref() as *const Chunk,
                    catalog.embeddings["emb_1"].as_ref() as *const Embedding,
                );
                assert_eq!(*first_addresses.get_or_insert(addresses), addresses);
            }
            let old = catalog.clone();
            for (id, doc) in &old.documents {
                assert!(!std::ptr::eq(doc.as_ref(), catalog.documents[id].as_ref()));
                assert_eq!(doc.metadata, catalog.documents[id].metadata);
            }
            for (id, child) in &old.chunks {
                assert!(!std::ptr::eq(child.as_ref(), catalog.chunks[id].as_ref()));
                assert!(child.text.is_empty());
            }
            for (id, embedding) in &old.embeddings {
                assert!(!std::ptr::eq(
                    embedding.as_ref(),
                    catalog.embeddings[id].as_ref()
                ));
                assert!(embedding.vector.is_empty());
            }
            for number in 1..=13 + seed {
                if number % 2 == 0 {
                    catalog
                        .apply_with_ref(
                            RecordPayload::Tombstone(Tombstone {
                                target_type: "document".into(),
                                target_id: format!("doc_{number}"),
                                deleted_at: String::new(),
                                reason: None,
                            }),
                            None,
                        )
                        .unwrap();
                    assert!(!catalog.is_chunk_live(&format!("chk_{number}")));
                } else {
                    catalog
                        .apply_with_ref(RecordPayload::DocumentPut(document(number, "col_1")), None)
                        .unwrap();
                }
                let id = format!("doc_{number}");
                assert!(catalog.documents[&id].title.is_none());
                assert_eq!(
                    old.documents[&id].title.as_deref(),
                    Some(format!("title-{seed}-{number}").as_str())
                );
                assert!(old.is_document_live(&id));
                assert!(old.is_chunk_live(&format!("chk_{number}")));
            }
            assert_eq!(
                old.visible_snapshot().live_embeddings().count(),
                13 + seed as usize
            );
            assert_eq!(old.live_documents().count(), 13 + seed as usize);
            assert_eq!(catalog.live_documents().count(), (14 + seed as usize) / 2);
        }
    }

    #[test]
    fn updated_chunk_membership_contains_only_the_latest_version() {
        for seed in 0..100 {
            let mut catalog = Catalog::default();
            catalog
                .apply_with_ref(RecordPayload::CollectionCreate(collection(1)), None)
                .unwrap();
            for id in 1..=3 {
                catalog
                    .apply_with_ref(RecordPayload::DocumentPut(document(id, "col_1")), None)
                    .unwrap();
            }
            let mut children = (1..=7).map(|id| chunk(id, "col_1")).collect::<Vec<_>>();
            for child in &mut children {
                child.document_id = "doc_1".into();
            }
            for step in 0..100 {
                let index = (seed + step) % children.len();
                let child = &mut children[index];
                if step % 3 == 0 {
                    child.document_id = format!("doc_{}", 1 + (seed + step) / 3 % 3);
                }
                child.ordinal = step;
                catalog
                    .apply_with_ref(RecordPayload::ChunkPut(child.clone()), None)
                    .unwrap();
                for id in 1..=3 {
                    let parent = format!("doc_{id}");
                    let mut expected = catalog
                        .chunks
                        .values()
                        .filter(|chunk| chunk.document_id == parent)
                        .map(|chunk| chunk.chunk_id.as_str())
                        .collect::<Vec<_>>();
                    let mut actual = catalog
                        .chunks_for_document_in(
                            "col_1",
                            &parent,
                            &crate::json::JsonOperationContext::standalone(),
                        )
                        .unwrap()
                        .iter()
                        .map(|chunk| chunk.chunk_id.as_str())
                        .collect::<Vec<_>>();
                    expected.sort_unstable();
                    actual.sort_unstable();
                    assert_eq!(actual, expected, "seed {seed}, step {step}");
                    assert_eq!(
                        catalog.chunk_count_for_document("col_1", &parent).unwrap(),
                        expected.len()
                    );
                }
                assert_eq!(
                    catalog
                        .document_chunks
                        .values()
                        .map(Vec::len)
                        .sum::<usize>(),
                    catalog.chunks.len()
                );
                assert!(catalog.document_chunks.values().all(|ids| !ids.is_empty()));
            }
        }
    }

    #[test]
    fn updated_external_ids_remove_old_aliases_without_erasing_new_owners() {
        for seed in 0..100 {
            let mut catalog = Catalog::default();
            for id in 1..=2 {
                catalog
                    .apply_with_ref(RecordPayload::CollectionCreate(collection(id)), None)
                    .unwrap();
            }
            let mut doc = document(1, "col_1");
            let mut child = chunk(1, "col_1");
            for step in 0..100 {
                let previous_doc = (doc.collection_id.clone(), doc.external_id.clone());
                let previous_chunk = (child.collection_id.clone(), child.chunk_external_id.clone());
                doc.collection_id = format!("col_{}", 1 + (seed + step) % 2);
                doc.external_id = format!("version-{step}");
                child.collection_id = doc.collection_id.clone();
                child.chunk_external_id = doc.external_id.clone();
                catalog
                    .apply_with_ref(RecordPayload::DocumentPut(doc.clone()), None)
                    .unwrap();
                catalog
                    .apply_with_ref(RecordPayload::ChunkPut(child.clone()), None)
                    .unwrap();
                assert!(catalog
                    .document_external_ids
                    .get(&previous_doc.0, &previous_doc.1)
                    .is_none());
                assert!(catalog
                    .chunk_external_ids
                    .get(&previous_chunk.0, &previous_chunk.1)
                    .is_none());
                assert_eq!(catalog.document_external_ids.len(), 1);
                assert_eq!(catalog.chunk_external_ids.len(), 1);
                assert_eq!(
                    catalog
                        .document_by_external_id(&doc.collection_id, &doc.external_id)
                        .unwrap()
                        .document_id,
                    doc.document_id
                );
                assert_eq!(
                    catalog
                        .chunk_by_external_id(&child.collection_id, &child.chunk_external_id)
                        .unwrap()
                        .chunk_id,
                    child.chunk_id
                );
            }
            // A newer object can already own an old alias when its former owner changes again.
            let mut replacement_doc = doc.clone();
            replacement_doc.document_id = "doc_2".into();
            let mut replacement_chunk = child.clone();
            replacement_chunk.chunk_id = "chk_2".into();
            replacement_chunk.document_id = "doc_2".into();
            catalog
                .apply_with_ref(RecordPayload::DocumentPut(replacement_doc), None)
                .unwrap();
            catalog
                .apply_with_ref(RecordPayload::ChunkPut(replacement_chunk), None)
                .unwrap();
            let alias = (doc.collection_id.clone(), doc.external_id.clone());
            doc.external_id = "latest".into();
            child.chunk_external_id = "latest".into();
            catalog
                .apply_with_ref(RecordPayload::DocumentPut(doc), None)
                .unwrap();
            catalog
                .apply_with_ref(RecordPayload::ChunkPut(child), None)
                .unwrap();
            assert_eq!(
                catalog
                    .document_external_ids
                    .get(&alias.0, &alias.1)
                    .unwrap(),
                "doc_2"
            );
            assert_eq!(
                catalog.chunk_external_ids.get(&alias.0, &alias.1).unwrap(),
                "chk_2"
            );
        }
    }

    #[test]
    fn explicit_tombstones_release_only_target_payload_across_100_seeds() {
        for seed in 0..100 {
            let mut catalog = Catalog::default();
            let metadata = crate::json::object_from_pairs([
                ("seed", crate::JsonValue::Number(seed as f64)),
                ("payload", crate::JsonValue::String("x".repeat(1024 + seed))),
            ]);
            let mut doc = document(1, "col_1");
            doc.metadata = metadata.clone();
            doc.title = Some("title".repeat(1024 + seed));
            doc.source_uri = Some("source/path".repeat(1024 + seed));
            let mut child = chunk(1, "col_1");
            child.metadata = metadata.clone();
            child.source_span = Some(crate::SourceSpan {
                start: Some(seed as u64),
                end: Some(seed as u64 + child.text.len() as u64),
                ..Default::default()
            });
            let reference = |record_type: &'static str, generation: u64| RecordRef {
                offset: generation * 100,
                record_type,
                batch_id: generation,
                sequence_no: generation,
                payload: PayloadRef {
                    offset: generation * 100 + 50,
                    len: 10,
                    checksum: 0,
                },
            };
            for record in [
                RecordPayload::CollectionCreate(collection(1)),
                RecordPayload::DocumentPut(doc.clone()),
                RecordPayload::ChunkPut(child.clone()),
            ] {
                catalog.apply_with_ref(record, None).unwrap();
            }
            catalog
                .record_refs
                .insert("doc_1".into(), reference("DocumentPut", 1));
            catalog
                .record_refs
                .insert("chk_1".into(), reference("ChunkPut", 2));
            let tombstone = |kind: &str, id: &str| {
                RecordPayload::Tombstone(Tombstone {
                    target_type: kind.into(),
                    target_id: id.into(),
                    deleted_at: "1970-01-01T00:00:00Z".into(),
                    reason: None,
                })
            };
            for record in [
                tombstone("collection", "col_1"),
                RecordPayload::CollectionDrop {
                    collection_id: "col_1".into(),
                    name: "collection-1".into(),
                    dropped_at: "1970-01-01T00:00:00Z".into(),
                    reason: None,
                },
                tombstone("unknown", "chk_1"),
                tombstone("document", "chk_1"),
            ] {
                catalog.apply_with_ref(record, None).unwrap();
                assert_eq!(catalog.documents["doc_1"].metadata, metadata);
                assert_eq!(catalog.documents["doc_1"].title, doc.title);
                assert_eq!(catalog.documents["doc_1"].source_uri, doc.source_uri);
                assert_eq!(catalog.chunks["chk_1"].metadata, metadata);
                assert_eq!(catalog.chunks["chk_1"].text_hash, child.text_hash);
                assert_eq!(catalog.record_refs["doc_1"], reference("DocumentPut", 1));
                assert_eq!(catalog.record_refs["chk_1"], reference("ChunkPut", 2));
            }
            catalog
                .apply_with_ref(RecordPayload::CollectionCreate(collection(1)), None)
                .unwrap();
            assert!(catalog.is_chunk_live("chk_1"));
            for _ in 0..2 {
                catalog
                    .apply_with_ref(tombstone("document", "doc_1"), None)
                    .unwrap();
                assert_eq!(
                    catalog.documents["doc_1"].metadata,
                    crate::JsonValue::object()
                );
                assert_eq!(catalog.chunks["chk_1"].metadata, metadata);
                assert!(catalog.documents["doc_1"].title.is_none());
                assert!(catalog.documents["doc_1"].source_uri.is_none());
                assert_eq!(catalog.documents["doc_1"].created_at.capacity(), 0);
                assert_eq!(catalog.documents["doc_1"].updated_at.capacity(), 0);
                assert!(!catalog.record_refs.contains_key("doc_1"));
                assert_eq!(catalog.record_refs["chk_1"], reference("ChunkPut", 2));
                assert_eq!(catalog.chunks["chk_1"].created_at, child.created_at);
                assert!(!catalog.is_chunk_live("chk_1"));
            }
            doc.metadata =
                crate::json::object_from_pairs([("revived", crate::JsonValue::Bool(true))]);
            catalog
                .apply_with_ref(
                    RecordPayload::DocumentPut(doc.clone()),
                    Some(reference("DocumentPut", 3)),
                )
                .unwrap();
            assert_eq!(catalog.documents["doc_1"].metadata, doc.metadata);
            assert_eq!(catalog.documents["doc_1"].title, doc.title);
            assert_eq!(catalog.documents["doc_1"].source_uri, doc.source_uri);
            assert_eq!(catalog.documents["doc_1"].created_at, doc.created_at);
            assert_eq!(catalog.record_refs["doc_1"], reference("DocumentPut", 3));
            assert_eq!(catalog.chunks["chk_1"].metadata, metadata);
            assert!(catalog.is_chunk_live("chk_1"));
            for _ in 0..2 {
                catalog
                    .apply_with_ref(tombstone("chunk", "chk_1"), None)
                    .unwrap();
                assert_eq!(catalog.chunks["chk_1"].metadata, crate::JsonValue::object());
                assert_eq!(catalog.chunks["chk_1"].text_hash.capacity(), 0);
                assert_eq!(catalog.chunks["chk_1"].created_at.capacity(), 0);
                assert!(!catalog.record_refs.contains_key("chk_1"));
                assert_eq!(
                    catalog.chunks["chk_1"].chunk_external_id,
                    child.chunk_external_id
                );
                assert_eq!(catalog.chunks["chk_1"].document_id, child.document_id);
                assert_eq!(catalog.chunks["chk_1"].source_span, child.source_span);
                assert_eq!(catalog.documents["doc_1"].metadata, doc.metadata);
            }
            child.metadata = doc.metadata.clone();
            catalog
                .apply_with_ref(
                    RecordPayload::ChunkPut(child.clone()),
                    Some(reference("ChunkPut", 4)),
                )
                .unwrap();
            assert_eq!(catalog.chunks["chk_1"].metadata, child.metadata);
            assert_eq!(catalog.chunks["chk_1"].text_hash, child.text_hash);
            assert_eq!(catalog.chunks["chk_1"].created_at, child.created_at);
            assert_eq!(catalog.record_refs["chk_1"], reference("ChunkPut", 4));
            assert!(catalog.is_chunk_live("chk_1"));
            for (kind, id) in [("document", "doc_99"), ("chunk", "chk_99")] {
                catalog.apply_with_ref(tombstone(kind, id), None).unwrap();
            }
            let mut late_doc = document(99, "col_1");
            late_doc.metadata = metadata.clone();
            let mut late_chunk = chunk(99, "col_1");
            late_chunk.metadata = metadata.clone();
            for record in [
                RecordPayload::DocumentPut(late_doc),
                RecordPayload::ChunkPut(late_chunk),
            ] {
                catalog.apply_with_ref(record, None).unwrap();
            }
            assert_eq!(catalog.documents["doc_99"].metadata, metadata);
            assert_eq!(catalog.chunks["chk_99"].metadata, metadata);
            assert!(catalog.is_chunk_live("chk_99"));
            assert_eq!(catalog.next_id("doc_", 0).unwrap(), "doc_0000000000000100");
            assert_eq!(catalog.next_id("chk_", 0).unwrap(), "chk_0000000000000100");
        }
    }

    #[test]
    fn compact_tombstone_counts_match_history_across_100_seeds() {
        fn check(catalog: &Catalog, history: &[Tombstone]) {
            assert_eq!(catalog.tombstone_count(), history.len());
            let stats = catalog.stats(0, ResourceProfile::Balanced, 256, &IndexStats::default());
            assert_eq!(stats.tombstone_count, history.len());
            assert_eq!(stats.dead_bytes_estimate, history.len() as u64 * 128);
            assert_eq!(stats.pending_compaction_bytes, stats.dead_bytes_estimate);
            for owner in ["col_1", "col_2", "col_99", "absent"] {
                let expected = history
                    .iter()
                    .filter(|entry| catalog.tombstone_belongs_to_collection(entry, owner))
                    .count();
                assert_eq!(catalog.tombstone_count_for_collection(owner), expected);
                if catalog.is_collection_live(owner) {
                    let stats = catalog
                        .collection_stats(owner, ResourceProfile::Balanced, 0, 0, |name| {
                            Ok(name.to_owned())
                        })
                        .unwrap();
                    assert_eq!(stats.tombstone_count, expected);
                    assert_eq!(stats.dead_bytes_estimate, expected as u64 * 128);
                }
            }
            for counts in [
                &catalog.collection_tombstones,
                &catalog.document_tombstones,
                &catalog.chunk_tombstones,
            ] {
                assert!(counts.len() <= 3);
            }
        }

        for seed in 0..100 {
            let mut catalog = Catalog::default();
            for number in [1, 2] {
                let owner = format!("col_{number}");
                for record in [
                    RecordPayload::CollectionCreate(collection(number)),
                    RecordPayload::DocumentPut(document(number, &owner)),
                    RecordPayload::ChunkPut(chunk(number, &owner)),
                ] {
                    catalog.apply_with_ref(record, None).unwrap();
                }
            }
            let mut history = Vec::new();
            for step in 0..100 {
                let (kind, prefix) = [
                    ("collection", "col_"),
                    ("document", "doc_"),
                    ("chunk", "chk_"),
                    ("unknown", "emb_"),
                ][(seed + step) % 4];
                let number = [1, 2, 99][step % 3];
                let entry = Tombstone {
                    target_type: kind.into(),
                    target_id: format!("{prefix}{number}"),
                    deleted_at: "1970-01-01T00:00:00Z".into(),
                    reason: Some(format!("history-{seed}-{step}").repeat(8)),
                };
                catalog
                    .apply_with_ref(RecordPayload::Tombstone(entry.clone()), None)
                    .unwrap();
                history.push(entry);
                check(&catalog, &history);
                let owner = if (seed + step) % 2 == 0 {
                    "col_1"
                } else {
                    "col_2"
                };
                if step % 5 == 0 {
                    catalog
                        .apply_with_ref(RecordPayload::DocumentPut(document(1, owner)), None)
                        .unwrap();
                }
                if step % 7 == 0 {
                    catalog
                        .apply_with_ref(RecordPayload::ChunkPut(chunk(1, owner)), None)
                        .unwrap();
                }
                if step == 20 {
                    catalog
                        .apply_with_ref(RecordPayload::DocumentPut(document(99, "col_2")), None)
                        .unwrap();
                }
                if step == 21 {
                    catalog
                        .apply_with_ref(RecordPayload::ChunkPut(chunk(99, "col_2")), None)
                        .unwrap();
                }
                if step % 13 == 0 {
                    for number in [1, 2, 99] {
                        catalog
                            .apply_with_ref(
                                RecordPayload::CollectionCreate(collection(number)),
                                None,
                            )
                            .unwrap();
                    }
                }
                check(&catalog, &history);
            }
            assert_eq!(catalog.next_id("emb_", 0).unwrap(), "emb_0000000000000100");
        }
    }

    #[test]
    fn business_record_count_overflow_does_not_change_catalog_state() {
        let mut catalog = Catalog {
            business_record_count: usize::MAX,
            ..Catalog::default()
        };
        let before = format!("{catalog:?}");
        let error = catalog
            .apply_with_ref(RecordPayload::CollectionCreate(collection(1)), None)
            .unwrap_err();
        assert_eq!(error.code(), ErrorCode::FileFormatError);
        assert_eq!(format!("{catalog:?}"), before);
    }

    #[test]
    fn tombstone_count_overflow_does_not_change_catalog_state() {
        let mut catalog = Catalog {
            tombstone_count: usize::MAX,
            ..Catalog::default()
        };
        let before = format!("{catalog:?}");
        let err = catalog
            .apply_with_ref(
                RecordPayload::Tombstone(Tombstone {
                    target_type: "chunk".into(),
                    target_id: "chk_99".into(),
                    deleted_at: "1970-01-01T00:00:00Z".into(),
                    reason: None,
                }),
                None,
            )
            .unwrap_err();
        assert_eq!(err.code(), ErrorCode::FileFormatError);
        assert_eq!(format!("{catalog:?}"), before);
    }

    #[test]
    fn replaced_embeddings_release_obsolete_summaries_and_refs_across_100_seeds() {
        for seed in 0..100 {
            let mut catalog = Catalog::default();
            let chunk_count = 1 + seed % 17;
            let mut counter = 0;
            for generation in 0..25 {
                for chunk in 0..chunk_count {
                    counter += 1;
                    let id = format!("emb_{counter:016}");
                    let chunk_id = format!("chk_{chunk:016}");
                    let previous = catalog.chunk_embeddings.get(&chunk_id).cloned();
                    let reference = RecordRef {
                        offset: counter * 100,
                        record_type: "EmbeddingPut",
                        batch_id: generation + 1,
                        sequence_no: counter,
                        payload: PayloadRef {
                            offset: counter * 100 + 50,
                            len: 50,
                            checksum: seed,
                        },
                    };
                    let embedding = Embedding {
                        embedding_id: id.clone(),
                        chunk_id: chunk_id.clone(),
                        collection_id: "col_1".into(),
                        chunk_external_id: format!("chunk-{chunk}"),
                        model_name: format!("model-{seed}"),
                        dim: 2,
                        vector: vec![1.0, generation as f32],
                        created_at: "1970-01-01T00:00:00Z".into(),
                    };
                    catalog
                        .apply_with_ref(
                            RecordPayload::EmbeddingPut(embedding.clone()),
                            Some(reference.clone()),
                        )
                        .unwrap();
                    // Reapplying the current ID must not remove its newly installed reference.
                    catalog
                        .apply_with_ref(
                            RecordPayload::EmbeddingPut(embedding),
                            Some(reference.clone()),
                        )
                        .unwrap();
                    let expected_count = counter.min(chunk_count) as usize;
                    assert_eq!(catalog.embeddings.len(), expected_count);
                    assert_eq!(catalog.record_refs.len(), expected_count);
                    assert_eq!(catalog.chunk_embeddings.len(), expected_count);
                    let current = catalog.embedding_for_chunk(&chunk_id).unwrap();
                    assert_eq!(current.embedding_id, id);
                    assert!(current.vector.is_empty());
                    assert_eq!(catalog.record_ref_for_embedding(&id), Some(&reference));
                    if let Some(previous) = previous {
                        assert!(!catalog.embeddings.contains_key(&previous));
                        assert!(catalog.record_ref_for_embedding(&previous).is_none());
                    }
                    assert_eq!(catalog.id_high_watermarks["emb_"], counter);
                    assert_eq!(
                        catalog.next_id("emb_", 0).unwrap(),
                        format!("emb_{:016}", counter + 1)
                    );
                }
            }
        }
    }
}
