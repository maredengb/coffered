use std::mem::size_of;

use super::*;
use crate::json::{
    btree_workspace, CatalogReservation, JsonInputKind, JsonOperationContext, JsonReservation,
    ReservedString, ReservedVec, WorkspaceReservation,
};
use crate::model::VectorIndexConfig;
use crate::storage::RecordReservation;

pub(super) struct CatalogMemory {
    reservation: CatalogReservation,
    live: usize,
}

impl std::fmt::Debug for CatalogMemory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CatalogMemory")
            .field("live", &self.live)
            .field("reserved", &self.reservation.reserved_bytes())
            .finish()
    }
}

#[cfg(test)]
impl Clone for CatalogMemory {
    fn clone(&self) -> Self {
        // Test reference catalogs are independent snapshots, not production copies.
        let mut reservation = JsonOperationContext::standalone().catalog_reservation();
        reservation.grow(self.live).unwrap();
        Self {
            reservation,
            live: self.live,
        }
    }
}

type Map<V> = BTreeMap<String, V>;
type Select<V> = for<'a> fn(&'a mut Catalog, &str) -> &'a mut Map<V>;
type SelectSet = fn(&mut Catalog) -> &mut BTreeSet<String>;

struct MapUndo<V> {
    select: Select<V>,
    scope: String,
    key: String,
    previous: Previous<V>,
}

enum Previous<V> {
    Missing,
    Replaced(V),
    Removed(String, V),
}

impl<V> MapUndo<V> {
    fn rollback(self, catalog: &mut Catalog) {
        let map = (self.select)(catalog, &self.scope);
        match self.previous {
            Previous::Missing => {
                map.remove(&self.key);
            }
            Previous::Replaced(value) => {
                *map.get_mut(&self.key).unwrap() = value;
            }
            Previous::Removed(key, value) => {
                map.insert(key, value);
            }
        }
        if map.is_empty() {
            *map = Map::new();
        }
    }
}

enum Undo {
    Collection(MapUndo<CollectionInfo>),
    Document(MapUndo<Box<Document>>),
    Chunk(MapUndo<Box<Chunk>>),
    Embedding(MapUndo<Box<Embedding>>),
    String(MapUndo<String>),
    Reference(MapUndo<RecordRef>),
    Watermark(MapUndo<u64>),
    Count(MapUndo<usize>),
    Children(MapUndo<Vec<String>>),
    AliasGroup(MapUndo<Map<String>>),
    Set {
        select: SelectSet,
        key: String,
        previous: Option<String>,
    },
    ChildAppend {
        document: String,
    },
    ChildRemove {
        document: String,
        index: usize,
        child: String,
    },
    ChildCapacity {
        document: String,
        previous: Vec<String>,
    },
    ChunkEmbedding {
        chunk: String,
        previous: Option<String>,
    },
}

impl Undo {
    fn rollback(self, catalog: &mut Catalog) {
        match self {
            Self::Collection(edit) => edit.rollback(catalog),
            Self::Document(edit) => edit.rollback(catalog),
            Self::Chunk(edit) => edit.rollback(catalog),
            Self::Embedding(edit) => edit.rollback(catalog),
            Self::String(edit) => edit.rollback(catalog),
            Self::Reference(edit) => edit.rollback(catalog),
            Self::Watermark(edit) => edit.rollback(catalog),
            Self::Count(edit) => edit.rollback(catalog),
            Self::Children(edit) => edit.rollback(catalog),
            Self::AliasGroup(edit) => edit.rollback(catalog),
            Self::Set {
                select,
                key,
                previous,
            } => {
                let set = select(catalog);
                set.remove(&key);
                if let Some(previous) = previous {
                    set.insert(previous);
                }
                if set.is_empty() {
                    *set = BTreeSet::new();
                }
            }
            Self::ChildAppend { document } => {
                catalog.document_chunks.get_mut(&document).unwrap().pop();
            }
            Self::ChildRemove {
                document,
                index,
                child,
            } => {
                catalog
                    .document_chunks
                    .get_mut(&document)
                    .unwrap()
                    .insert(index, child);
            }
            Self::ChildCapacity {
                document,
                mut previous,
            } => {
                let children = catalog.document_chunks.get_mut(&document).unwrap();
                previous.append(children);
                *children = previous;
            }
            Self::ChunkEmbedding { chunk, previous } => {
                catalog.chunks.get_mut(&chunk).unwrap().embedding_id = previous;
            }
        }
    }
}

trait Entry: Sized {
    fn heap(&self, context: &JsonOperationContext) -> Result<usize>;
    fn undo(edit: MapUndo<Self>) -> Undo;
}

macro_rules! scalar_entry {
    ($type:ty, $variant:ident) => {
        impl Entry for $type {
            fn heap(&self, _: &JsonOperationContext) -> Result<usize> {
                Ok(0)
            }
            fn undo(edit: MapUndo<Self>) -> Undo {
                Undo::$variant(edit)
            }
        }
    };
}
scalar_entry!(RecordRef, Reference);
scalar_entry!(u64, Watermark);
scalar_entry!(usize, Count);

impl Entry for String {
    fn heap(&self, _: &JsonOperationContext) -> Result<usize> {
        Ok(self.capacity())
    }
    fn undo(edit: MapUndo<Self>) -> Undo {
        Undo::String(edit)
    }
}
impl Entry for Vec<String> {
    fn heap(&self, _: &JsonOperationContext) -> Result<usize> {
        Ok(
            self.capacity() * size_of::<String>()
                + self.iter().map(String::capacity).sum::<usize>(),
        )
    }
    fn undo(edit: MapUndo<Self>) -> Undo {
        Undo::Children(edit)
    }
}
impl Entry for Map<String> {
    fn heap(&self, _: &JsonOperationContext) -> Result<usize> {
        Ok(btree_workspace::<String, String>(self.len())
            + self
                .iter()
                .map(|(key, value)| key.capacity() + value.capacity())
                .sum::<usize>())
    }
    fn undo(edit: MapUndo<Self>) -> Undo {
        Undo::AliasGroup(edit)
    }
}

fn json_bytes(value: &crate::JsonValue, context: &JsonOperationContext) -> Result<usize> {
    value.retained_bytes_in(JsonInputKind::StoredJson, context)
}
fn optional_bytes(value: &Option<String>) -> usize {
    value.as_ref().map_or(0, String::capacity)
}

impl Entry for CollectionInfo {
    fn heap(&self, context: &JsonOperationContext) -> Result<usize> {
        Ok(self.collection_id.capacity()
            + self.name.capacity()
            + self.created_at.capacity()
            + optional_bytes(&self.embedding_model)
            + match &self.vector_index {
                VectorIndexConfig::Exact => 0,
                VectorIndexConfig::Ann { provider, params } => {
                    provider.capacity() + json_bytes(params, context)?
                }
            })
    }
    fn undo(edit: MapUndo<Self>) -> Undo {
        Undo::Collection(edit)
    }
}
impl Entry for Box<Document> {
    fn heap(&self, context: &JsonOperationContext) -> Result<usize> {
        Ok(size_of::<Document>()
            + self.document_id.capacity()
            + self.collection_id.capacity()
            + self.external_id.capacity()
            + optional_bytes(&self.title)
            + optional_bytes(&self.source_uri)
            + self.created_at.capacity()
            + self.updated_at.capacity()
            + json_bytes(&self.metadata, context)?)
    }
    fn undo(edit: MapUndo<Self>) -> Undo {
        Undo::Document(edit)
    }
}
impl Entry for Box<Chunk> {
    fn heap(&self, context: &JsonOperationContext) -> Result<usize> {
        Ok(size_of::<Chunk>()
            + self.chunk_id.capacity()
            + self.document_id.capacity()
            + self.collection_id.capacity()
            + self.chunk_external_id.capacity()
            + self.text.capacity()
            + self.text_hash.capacity()
            + optional_bytes(&self.embedding_id)
            + self.created_at.capacity()
            + json_bytes(&self.metadata, context)?)
    }
    fn undo(edit: MapUndo<Self>) -> Undo {
        Undo::Chunk(edit)
    }
}
impl Entry for Box<Embedding> {
    fn heap(&self, _: &JsonOperationContext) -> Result<usize> {
        Ok(size_of::<Embedding>()
            + self.embedding_id.capacity()
            + self.chunk_id.capacity()
            + self.collection_id.capacity()
            + self.chunk_external_id.capacity()
            + self.model_name.capacity()
            + self.created_at.capacity()
            + self.vector.capacity() * size_of::<f32>())
    }
    fn undo(edit: MapUndo<Self>) -> Undo {
        Undo::Embedding(edit)
    }
}

// Mutations are private under the database lock until the critical append succeeds.
// The journal owns only changed entries; it never clones an entire catalog or child list.
pub(crate) struct CatalogTransaction<'a> {
    catalog: &'a mut Catalog,
    context: &'a JsonOperationContext,
    undo: Vec<Undo>,
    journal_work: WorkspaceReservation,
    initial_bytes: usize,
    initial_business: usize,
    initial_tombstones: usize,
    committed: bool,
    rollback_on_drop: bool,
    pending_json: Option<ReservedVec<usize>>,
}

impl Catalog {
    #[cfg(test)]
    pub(crate) fn reserved_bytes(&self) -> usize {
        self.memory
            .as_ref()
            .map_or(0, |memory| memory.reservation.reserved_bytes())
    }

    pub(crate) fn attach_context(&self, context: &JsonOperationContext) -> Result<()> {
        match &self.memory {
            Some(memory) => memory.reservation.attach_context(context),
            None => Ok(()),
        }
    }

    pub(crate) fn apply_retained(
        &mut self,
        record: CommittedRecord,
        work: &mut RecordReservation,
        context: &JsonOperationContext,
    ) -> Result<()> {
        let mut transaction = self.transaction(context);
        // Every caller is building a new catalog and discards it on recovery
        // failure. Only mutations of an already published catalog need undo.
        transaction.rollback_on_drop = false;
        transaction.apply(record.payload, Some(record.record_ref), work)?;
        transaction.commit();
        self.clear_model_states();
        Ok(())
    }

    pub(crate) fn new_in(context: &JsonOperationContext) -> Self {
        Self {
            memory: Some(CatalogMemory {
                reservation: context.catalog_reservation(),
                live: 0,
            }),
            ..Self::default()
        }
    }

    pub(crate) fn transaction<'a>(
        &'a mut self,
        context: &'a JsonOperationContext,
    ) -> CatalogTransaction<'a> {
        CatalogTransaction {
            initial_bytes: self.memory.as_ref().map_or(0, |memory| memory.live),
            initial_business: self.business_record_count,
            initial_tombstones: self.tombstone_count,
            catalog: self,
            context,
            undo: Vec::new(),
            journal_work: context.workspace_reservation(),
            committed: false,
            rollback_on_drop: true,
            pending_json: None,
        }
    }
}

impl CatalogTransaction<'_> {
    pub(crate) fn prepare_reclaiming<'b>(
        &mut self,
        records: impl ExactSizeIterator<Item = &'b RecordPayload> + Clone,
        json_work: &mut JsonReservation,
        reclaim: impl FnOnce() -> bool,
    ) -> Result<()> {
        assert!(
            self.rollback_on_drop,
            "reclaiming preparation requires rollback"
        );
        match self.prepare(records.clone(), json_work) {
            Err(error) if crate::json::is_memory_budget_error(&error) && reclaim() => {
                // Input metadata already transferred here must stay charged while
                // the same immutable batch is prepared again, without another IO.
                self.rollback_changes(true);
                self.prepare_records(records, json_work)
            }
            result => result,
        }
    }

    pub(crate) fn prepare<'b>(
        &mut self,
        records: impl ExactSizeIterator<Item = &'b RecordPayload>,
        json_work: &mut JsonReservation,
    ) -> Result<()> {
        assert!(self.pending_json.is_none());
        self.prepare_records(records, json_work)
    }

    fn prepare_records<'b>(
        &mut self,
        records: impl ExactSizeIterator<Item = &'b RecordPayload>,
        json_work: &mut JsonReservation,
    ) -> Result<()> {
        if self.pending_json.is_none() {
            self.pending_json = Some(ReservedVec::with_capacity(
                records.len(),
                self.context,
                "catalog_build",
            )?);
        }
        for (index, record) in records.enumerate() {
            if index >= self.pending_json.as_ref().unwrap().as_slice().len() {
                let bytes = record.metadata_bytes_in(self.context)?;
                json_work.grow(
                    bytes.saturating_sub(json_work.covered_bytes()),
                    JsonInputKind::StoredJson,
                )?;
                if let Some(memory) = &mut self.catalog.memory {
                    memory.reservation.absorb(json_work, None, bytes, 0)?;
                }
                self.pending_json.as_mut().unwrap().push(bytes);
            }
            let (summary, mut work) = record.copy_catalog_fields_in(self.context)?;
            let reference = record_id(record).map(|_| RecordRef {
                offset: 0,
                record_type: "pending",
                batch_id: 0,
                sequence_no: index as u64 + 1,
                payload: crate::storage::PayloadRef {
                    offset: 0,
                    len: 0,
                    checksum: 0,
                },
            });
            self.apply(summary, reference, &mut work)?;
        }
        Ok(())
    }

    fn rollback_changes(&mut self, keep_input_credit: bool) {
        if self.rollback_on_drop {
            while let Some(edit) = self.undo.pop() {
                edit.rollback(self.catalog);
            }
            self.catalog.business_record_count = self.initial_business;
            self.catalog.tombstone_count = self.initial_tombstones;
        }
        if keep_input_credit {
            self.undo = Vec::new();
            self.journal_work.retain(0);
        }
        if let Some(memory) = &mut self.catalog.memory {
            if self.rollback_on_drop {
                memory.live = self.initial_bytes;
            }
            let input_bytes = if keep_input_credit {
                self.pending_json
                    .as_ref()
                    .map_or(0, |bytes| bytes.as_slice().iter().sum())
            } else {
                0
            };
            memory.reservation.retain(memory.live + input_bytes);
        }
    }

    pub(crate) fn finish_references(&mut self, records: &mut Vec<CommittedRecord>) {
        for (index, record) in records.drain(..).enumerate() {
            let current =
                record_id(&record.payload).and_then(|id| self.catalog.record_refs.get_mut(id));
            let Some(reference) = current.filter(|reference| {
                reference.batch_id == 0 && reference.sequence_no == index as u64 + 1
            }) else {
                continue;
            };
            *reference = record.record_ref;
            match record.payload {
                RecordPayload::DocumentPut(value) => {
                    self.catalog
                        .documents
                        .get_mut(&value.document_id)
                        .unwrap()
                        .metadata = value.metadata
                }
                RecordPayload::ChunkPut(value) => {
                    self.catalog
                        .chunks
                        .get_mut(&value.chunk_id)
                        .unwrap()
                        .metadata = value.metadata
                }
                RecordPayload::CollectionCreate(value) => {
                    if let VectorIndexConfig::Ann { params, .. } = value.vector_index {
                        let target = &mut self
                            .catalog
                            .collections
                            .get_mut(&value.collection_id)
                            .unwrap()
                            .vector_index;
                        let VectorIndexConfig::Ann { params: target, .. } = target else {
                            unreachable!()
                        };
                        *target = params;
                    }
                }
                _ => {}
            }
            self.change_live(0, self.pending_json.as_ref().unwrap().as_slice()[index]);
        }
        self.pending_json = None;
    }

    fn catalog_key(
        &mut self,
        select: impl FnOnce(&Catalog) -> Option<&str>,
    ) -> Result<Option<String>> {
        let copied = select(self.catalog)
            .map(|value| ReservedString::copy(value, self.context))
            .transpose()?;
        Ok(copied.map(|copied| {
            let (value, work) = copied.into_parts();
            self.journal_work.merge(work);
            value
        }))
    }

    fn charge(&mut self, bytes: usize) -> Result<()> {
        if let Some(memory) = &mut self.catalog.memory {
            memory.reservation.grow(bytes)?;
        }
        Ok(())
    }

    fn change_live(&mut self, before: usize, after: usize) {
        if let Some(memory) = &mut self.catalog.memory {
            memory.live = memory.live - before + after;
        }
    }

    fn copy(&mut self, value: &str) -> Result<String> {
        self.charge(value.len())?;
        let value = value.to_owned();
        if value.len() != value.capacity() {
            return Err(
                std::io::Error::other("catalog string allocation exceeds reservation").into(),
            );
        }
        Ok(value)
    }

    fn journal_key(&mut self, value: &str) -> Result<String> {
        let (key, work) = ReservedString::copy(value, self.context)?.into_parts();
        self.journal_work.merge(work);
        Ok(key)
    }

    fn undo_key(&mut self, value: &str) -> Result<String> {
        if self.rollback_on_drop {
            self.journal_key(value)
        } else {
            Ok(String::new())
        }
    }

    fn save_undo(&mut self, edit: Undo) {
        if self.rollback_on_drop {
            self.undo.push(edit);
        }
    }

    fn journal_slot(&mut self) -> Result<()> {
        if !self.rollback_on_drop || self.undo.len() < self.undo.capacity() {
            return Ok(());
        }
        let old = self.undo.capacity();
        let minimum = old + 1;
        let preferred = minimum.max(old.saturating_mul(2)).max(8);
        let strings = self.journal_work.reserved_bytes() - old * size_of::<Undo>();
        let granted = self.journal_work.grow_to_wide(
            strings as u128 + minimum as u128 * size_of::<Undo>() as u128,
            strings as u128 + preferred as u128 * size_of::<Undo>() as u128,
            "catalog_build",
        )?;
        let capacity = (granted - strings) / size_of::<Undo>();
        self.undo
            .try_reserve_exact(capacity - self.undo.len())
            .map_err(|error| std::io::Error::other(error.to_string()))?;
        if self.undo.capacity() > capacity {
            return Err(
                std::io::Error::other("catalog journal allocation exceeds reservation").into(),
            );
        }
        self.journal_work
            .retain(strings + self.undo.capacity() * size_of::<Undo>());
        Ok(())
    }

    fn update<V: Entry>(
        &mut self,
        select: Select<V>,
        scope: &str,
        key: &str,
        value: Option<V>,
    ) -> Result<()> {
        let map = select(self.catalog, scope);
        let before_len = map.len();
        let previous_key = map.get_key_value(key).map(|(key, _)| key.capacity());
        if previous_key.is_none() && value.is_none() {
            return Ok(());
        }
        let before = map
            .get_key_value(key)
            .map(|(key, value)| value.heap(self.context).map(|bytes| bytes + key.capacity()))
            .transpose()?
            .unwrap_or(0);
        let after_len =
            before_len - usize::from(map.contains_key(key)) + usize::from(value.is_some());
        let after_value = value
            .as_ref()
            .map(|value| value.heap(self.context))
            .transpose()?
            .unwrap_or(0);
        let before_tree = btree_workspace::<String, V>(before_len);
        let after_tree = btree_workspace::<String, V>(after_len);
        self.charge(after_tree.saturating_sub(before_tree))?;
        self.journal_slot()?;
        let journal_scope = self.undo_key(scope)?;
        let journal_key = self.undo_key(key)?;
        let next_key = (value.is_some() && previous_key.is_none())
            .then(|| self.copy(key))
            .transpose()?;
        let after_key = if value.is_some() {
            previous_key.unwrap_or_else(|| next_key.as_ref().unwrap().capacity())
        } else {
            0
        };
        let after = after_value + after_key;
        let map = select(self.catalog, scope);
        let previous = match value {
            Some(value) => match map.get_mut(key) {
                Some(current) => Previous::Replaced(std::mem::replace(current, value)),
                None => {
                    map.insert(next_key.unwrap(), value);
                    Previous::Missing
                }
            },
            None => match map.remove_entry(key) {
                Some((key, value)) => Previous::Removed(key, value),
                None => Previous::Missing,
            },
        };
        if map.is_empty() {
            *map = Map::new();
        }
        self.save_undo(V::undo(MapUndo {
            select,
            scope: journal_scope,
            key: journal_key,
            previous,
        }));
        self.change_live(before + before_tree, after + after_tree);
        Ok(())
    }

    fn set(&mut self, select: SelectSet, key: &str, present: bool) -> Result<()> {
        let set = select(self.catalog);
        if set.contains(key) == present {
            return Ok(());
        }
        let old_len = set.len();
        let old_bytes = set.get(key).map_or(0, String::capacity);
        let new_len = if present { old_len + 1 } else { old_len - 1 };
        let before_tree = btree_workspace::<String, ()>(old_len);
        let after_tree = btree_workspace::<String, ()>(new_len);
        self.charge(after_tree.saturating_sub(before_tree))?;
        self.journal_slot()?;
        let lookup = self.undo_key(key)?;
        let next = present.then(|| self.copy(key)).transpose()?;
        let new_bytes = next.as_ref().map_or(0, String::capacity);
        let set = select(self.catalog);
        let previous = set.take(key);
        if let Some(next) = next {
            set.insert(next);
        }
        if set.is_empty() {
            *set = BTreeSet::new();
        }
        self.save_undo(Undo::Set {
            select,
            key: lookup,
            previous,
        });
        self.change_live(before_tree + old_bytes, after_tree + new_bytes);
        Ok(())
    }

    fn alias(
        &mut self,
        document: bool,
        collection: &str,
        external: &str,
        id: Option<&str>,
    ) -> Result<()> {
        let groups: Select<Map<String>> = if document {
            |catalog, _| &mut catalog.document_external_ids.collections
        } else {
            |catalog, _| &mut catalog.chunk_external_ids.collections
        };
        let ids: Select<String> = if document {
            |catalog, collection| {
                catalog
                    .document_external_ids
                    .collections
                    .get_mut(collection)
                    .unwrap()
            }
        } else {
            |catalog, collection| {
                catalog
                    .chunk_external_ids
                    .collections
                    .get_mut(collection)
                    .unwrap()
            }
        };
        if !groups(self.catalog, "").contains_key(collection) {
            if id.is_none() {
                return Ok(());
            }
            self.update(groups, "", collection, Some(Map::new()))?;
        }
        if ids(self.catalog, collection)
            .get(external)
            .map(String::as_str)
            == id
        {
            return Ok(());
        }
        let next = id.map(|id| self.copy(id)).transpose()?;
        self.update(ids, collection, external, next)?;
        if ids(self.catalog, collection).is_empty() {
            self.update(groups, "", collection, None)?;
        }
        Ok(())
    }

    fn remove_alias(
        &mut self,
        document: bool,
        collection: &str,
        external: &str,
        id: &str,
    ) -> Result<()> {
        let aliases = if document {
            &self.catalog.document_external_ids
        } else {
            &self.catalog.chunk_external_ids
        };
        if aliases
            .get(collection, external)
            .is_some_and(|owner| owner == id)
        {
            self.alias(document, collection, external, None)?;
        }
        Ok(())
    }

    fn append_child(&mut self, document: &str, chunk: &str) -> Result<()> {
        if !self.catalog.document_chunks.contains_key(document) {
            self.update(
                |catalog, _| &mut catalog.document_chunks,
                "",
                document,
                Some(Vec::new()),
            )?;
        }
        let children = &self.catalog.document_chunks[document];
        if children.len() == children.capacity() {
            let minimum = children.len() + 1;
            let mut capacity = minimum.max(children.capacity().saturating_mul(2)).max(4);
            if self.charge(capacity * size_of::<String>()).is_err() {
                capacity = minimum;
                self.charge(capacity * size_of::<String>())?;
            }
            self.journal_slot()?;
            let key = self.undo_key(document)?;
            let mut replacement = Vec::new();
            replacement
                .try_reserve_exact(capacity)
                .map_err(|error| std::io::Error::other(error.to_string()))?;
            if replacement.capacity() != capacity {
                return Err(
                    std::io::Error::other("catalog child array exceeds reservation").into(),
                );
            }
            let children = self.catalog.document_chunks.get_mut(document).unwrap();
            let old_capacity = children.capacity();
            replacement.append(children);
            let previous = std::mem::replace(children, replacement);
            self.save_undo(Undo::ChildCapacity {
                document: key,
                previous,
            });
            self.change_live(
                old_capacity * size_of::<String>(),
                capacity * size_of::<String>(),
            );
        }
        self.journal_slot()?;
        let key = self.undo_key(document)?;
        let child = self.copy(chunk)?;
        let bytes = child.capacity();
        self.catalog
            .document_chunks
            .get_mut(document)
            .unwrap()
            .push(child);
        self.save_undo(Undo::ChildAppend { document: key });
        self.change_live(0, bytes);
        Ok(())
    }

    fn remove_child(&mut self, document: &str, chunk: &str) -> Result<()> {
        let Some(index) = self
            .catalog
            .document_chunks
            .get(document)
            .and_then(|children| children.iter().position(|child| child == chunk))
        else {
            return Ok(());
        };
        self.journal_slot()?;
        let key = self.undo_key(document)?;
        let child = self
            .catalog
            .document_chunks
            .get_mut(document)
            .unwrap()
            .remove(index);
        let bytes = child.capacity();
        self.save_undo(Undo::ChildRemove {
            document: key,
            index,
            child,
        });
        self.change_live(bytes, 0);
        if self.catalog.document_chunks[document].is_empty() {
            self.update(
                |catalog, _| &mut catalog.document_chunks,
                "",
                document,
                None,
            )?;
        }
        Ok(())
    }

    fn track_id(&mut self, id: &str) -> Result<()> {
        for prefix in ["col_", "doc_", "chk_", "emb_"] {
            if let Some(rest) = id.strip_prefix(prefix) {
                if let Ok(value) = rest.parse::<u64>() {
                    self.watermark(prefix, value)?;
                }
                break;
            }
        }
        Ok(())
    }

    fn watermark(&mut self, prefix: &str, value: u64) -> Result<()> {
        if self
            .catalog
            .id_high_watermarks
            .get(prefix)
            .map_or(true, |old| *old < value)
        {
            self.update(
                |catalog, _| &mut catalog.id_high_watermarks,
                "",
                prefix,
                Some(value),
            )?;
        }
        Ok(())
    }

    fn reference(&mut self, id: &str, reference: Option<RecordRef>) -> Result<()> {
        if let Some(reference) = reference {
            self.update(
                |catalog, _| &mut catalog.record_refs,
                "",
                id,
                Some(reference),
            )?;
        }
        Ok(())
    }

    pub(crate) fn apply(
        &mut self,
        mut record: RecordPayload,
        reference: Option<RecordRef>,
        work: &mut RecordReservation,
    ) -> Result<()> {
        let business = !matches!(
            record,
            RecordPayload::Commit(_) | RecordPayload::CompactMarker(_)
        );
        let next_business = self
            .catalog
            .business_record_count
            .checked_add(usize::from(business))
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::FileFormatError,
                    "catalog business record count overflow",
                )
            })?;
        // Large primary payloads are read lazily, never retained by the catalog.
        match &mut record {
            RecordPayload::ChunkPut(chunk) => chunk.text = String::new(),
            RecordPayload::EmbeddingPut(embedding) => embedding.vector = Vec::new(),
            _ => {}
        }
        let json = match &record {
            RecordPayload::DocumentPut(document) => json_bytes(&document.metadata, self.context)?,
            RecordPayload::ChunkPut(chunk) => json_bytes(&chunk.metadata, self.context)?,
            RecordPayload::CollectionCreate(CollectionInfo {
                vector_index: VectorIndexConfig::Ann { params, .. },
                ..
            }) => json_bytes(params, self.context)?,
            _ => 0,
        };
        if let Some(memory) = &mut self.catalog.memory {
            work.transfer_to_catalog(
                &mut memory.reservation,
                json,
                crate::storage::record_workspace_bytes(&record),
            )?;
        }
        match record {
            RecordPayload::CollectionCreate(collection) => {
                let id = self.journal_key(&collection.collection_id)?;
                self.track_id(&id)?;
                self.reference(&id, reference)?;
                let owner = self.copy(&id)?;
                self.update(
                    |catalog, _| &mut catalog.collection_names,
                    "",
                    &collection.name,
                    Some(owner),
                )?;
                self.set(|catalog| &mut catalog.deleted_collections, &id, false)?;
                self.update(
                    |catalog, _| &mut catalog.collections,
                    "",
                    &id,
                    Some(collection),
                )?;
            }
            RecordPayload::CollectionDrop {
                collection_id,
                dropped_at,
                ..
            } => {
                if let Some(name) = self.catalog_key(|catalog| {
                    catalog
                        .collections
                        .get(&collection_id)
                        .map(|value| value.name.as_str())
                })? {
                    self.set(
                        |catalog| &mut catalog.reserved_collection_names,
                        &name,
                        true,
                    )?;
                }
                self.update(
                    |catalog, _| &mut catalog.collection_drop_times,
                    "",
                    &collection_id,
                    Some(dropped_at),
                )?;
                self.set(
                    |catalog| &mut catalog.deleted_collections,
                    &collection_id,
                    true,
                )?;
            }
            RecordPayload::DocumentPut(document) => {
                let id = self.journal_key(&document.document_id)?;
                self.track_id(&id)?;
                if self.catalog.documents.get(&id).is_some_and(|old| {
                    old.collection_id != document.collection_id
                        || old.external_id != document.external_id
                }) {
                    let collection = self
                        .catalog_key(|catalog| {
                            catalog
                                .documents
                                .get(&id)
                                .map(|old| old.collection_id.as_str())
                        })?
                        .unwrap();
                    let external = self
                        .catalog_key(|catalog| {
                            catalog
                                .documents
                                .get(&id)
                                .map(|old| old.external_id.as_str())
                        })?
                        .unwrap();
                    self.remove_alias(true, &collection, &external, &id)?;
                }
                self.reference(&id, reference)?;
                self.set(|catalog| &mut catalog.deleted_documents, &id, false)?;
                self.alias(
                    true,
                    &document.collection_id,
                    &document.external_id,
                    Some(&id),
                )?;
                self.charge(size_of::<Document>())?;
                self.update(
                    |catalog, _| &mut catalog.documents,
                    "",
                    &id,
                    Some(Box::new(document)),
                )?;
            }
            RecordPayload::ChunkPut(chunk) => {
                let id = self.journal_key(&chunk.chunk_id)?;
                self.track_id(&id)?;
                if self.catalog.chunks.get(&id).is_some_and(|old| {
                    old.collection_id != chunk.collection_id
                        || old.chunk_external_id != chunk.chunk_external_id
                }) {
                    let collection = self
                        .catalog_key(|catalog| {
                            catalog
                                .chunks
                                .get(&id)
                                .map(|old| old.collection_id.as_str())
                        })?
                        .unwrap();
                    let external = self
                        .catalog_key(|catalog| {
                            catalog
                                .chunks
                                .get(&id)
                                .map(|old| old.chunk_external_id.as_str())
                        })?
                        .unwrap();
                    self.remove_alias(false, &collection, &external, &id)?;
                }
                if self.catalog.chunks.get(&id).map(|old| &old.document_id)
                    != Some(&chunk.document_id)
                {
                    if let Some(document) = self.catalog_key(|catalog| {
                        catalog.chunks.get(&id).map(|old| old.document_id.as_str())
                    })? {
                        self.remove_child(&document, &id)?;
                    }
                    self.append_child(&chunk.document_id, &id)?;
                }
                self.reference(&id, reference)?;
                self.set(|catalog| &mut catalog.deleted_chunks, &id, false)?;
                self.alias(
                    false,
                    &chunk.collection_id,
                    &chunk.chunk_external_id,
                    Some(&id),
                )?;
                self.charge(size_of::<Chunk>())?;
                self.update(
                    |catalog, _| &mut catalog.chunks,
                    "",
                    &id,
                    Some(Box::new(chunk)),
                )?;
            }
            RecordPayload::EmbeddingPut(mut embedding) => {
                let id = self.journal_key(&embedding.embedding_id)?;
                self.track_id(&id)?;
                self.reference(&id, reference)?;
                if let Some(external) = self.catalog_key(|catalog| {
                    catalog
                        .chunks
                        .get(&embedding.chunk_id)
                        .map(|chunk| chunk.chunk_external_id.as_str())
                })? {
                    embedding.chunk_external_id = self.copy(&external)?;
                    self.journal_slot()?;
                    let key = self.undo_key(&embedding.chunk_id)?;
                    let next = self.copy(&id)?;
                    let after = next.capacity();
                    let previous = self
                        .catalog
                        .chunks
                        .get_mut(&embedding.chunk_id)
                        .unwrap()
                        .embedding_id
                        .replace(next);
                    let before = optional_bytes(&previous);
                    self.save_undo(Undo::ChunkEmbedding {
                        chunk: key,
                        previous,
                    });
                    self.change_live(before, after);
                }
                if let Some(previous) = self.catalog_key(|catalog| {
                    catalog
                        .chunk_embeddings
                        .get(&embedding.chunk_id)
                        .map(String::as_str)
                })? {
                    if previous != id {
                        self.update(|catalog, _| &mut catalog.embeddings, "", &previous, None)?;
                        self.update(|catalog, _| &mut catalog.record_refs, "", &previous, None)?;
                    }
                }
                let owner = self.copy(&id)?;
                self.update(
                    |catalog, _| &mut catalog.chunk_embeddings,
                    "",
                    &embedding.chunk_id,
                    Some(owner),
                )?;
                self.charge(size_of::<Embedding>())?;
                self.update(
                    |catalog, _| &mut catalog.embeddings,
                    "",
                    &id,
                    Some(Box::new(embedding)),
                )?;
            }
            RecordPayload::Tombstone(tombstone) => {
                self.tombstone(&tombstone.target_type, &tombstone.target_id)?
            }
            RecordPayload::CompactMarker(marker) => {
                for (prefix, value) in marker.id_high_watermarks {
                    self.watermark(&prefix, value)?;
                }
            }
            RecordPayload::Commit(_) => {}
        }
        self.catalog.business_record_count = next_business;
        Ok(())
    }

    fn tombstone(&mut self, target: &str, id: &str) -> Result<()> {
        let count = self.catalog.tombstone_count.checked_add(1).ok_or_else(|| {
            Error::new(
                ErrorCode::FileFormatError,
                "catalog tombstone count overflow",
            )
        })?;
        self.track_id(id)?;
        let counters: Select<usize> = match target {
            "collection" => |catalog, _| &mut catalog.collection_tombstones,
            "document" => |catalog, _| &mut catalog.document_tombstones,
            "chunk" => |catalog, _| &mut catalog.chunk_tombstones,
            _ => {
                self.catalog.tombstone_count = count;
                return Ok(());
            }
        };
        let next = counters(self.catalog, "")
            .get(id)
            .copied()
            .unwrap_or(0)
            .checked_add(1)
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::FileFormatError,
                    "catalog target tombstone count overflow",
                )
            })?;
        self.update(counters, "", id, Some(next))?;
        match target {
            "collection" => self.set(|catalog| &mut catalog.deleted_collections, id, true)?,
            "document" => {
                if self.catalog.documents.contains_key(id) {
                    let collection = self
                        .catalog_key(|catalog| {
                            catalog
                                .documents
                                .get(id)
                                .map(|old| old.collection_id.as_str())
                        })?
                        .unwrap();
                    let external = self
                        .catalog_key(|catalog| {
                            catalog
                                .documents
                                .get(id)
                                .map(|old| old.external_id.as_str())
                        })?
                        .unwrap();
                    self.charge(size_of::<Document>())?;
                    let document = Box::new(Document {
                        document_id: self.copy(id)?,
                        collection_id: self.copy(&collection)?,
                        external_id: self.copy(&external)?,
                        title: None,
                        source_uri: None,
                        metadata: crate::JsonValue::object(),
                        created_at: String::new(),
                        updated_at: String::new(),
                    });
                    self.update(|catalog, _| &mut catalog.documents, "", id, Some(document))?;
                    self.update(|catalog, _| &mut catalog.record_refs, "", id, None)?;
                }
                self.set(|catalog| &mut catalog.deleted_documents, id, true)?;
            }
            "chunk" => {
                if self.catalog.chunks.contains_key(id) {
                    let document = self
                        .catalog_key(|catalog| {
                            catalog.chunks.get(id).map(|old| old.document_id.as_str())
                        })?
                        .unwrap();
                    let collection = self
                        .catalog_key(|catalog| {
                            catalog.chunks.get(id).map(|old| old.collection_id.as_str())
                        })?
                        .unwrap();
                    let external = self
                        .catalog_key(|catalog| {
                            catalog
                                .chunks
                                .get(id)
                                .map(|old| old.chunk_external_id.as_str())
                        })?
                        .unwrap();
                    let embedding = self.catalog_key(|catalog| {
                        catalog
                            .chunks
                            .get(id)
                            .and_then(|old| old.embedding_id.as_deref())
                    })?;
                    let old = &self.catalog.chunks[id];
                    let (ordinal, token_count, source_span) =
                        (old.ordinal, old.token_count, old.source_span.clone());
                    self.charge(size_of::<Chunk>())?;
                    let chunk = Box::new(Chunk {
                        chunk_id: self.copy(id)?,
                        document_id: self.copy(&document)?,
                        collection_id: self.copy(&collection)?,
                        chunk_external_id: self.copy(&external)?,
                        embedding_id: embedding.as_deref().map(|id| self.copy(id)).transpose()?,
                        ordinal,
                        token_count,
                        source_span,
                        text: String::new(),
                        text_hash: String::new(),
                        metadata: crate::JsonValue::object(),
                        created_at: String::new(),
                    });
                    self.update(|catalog, _| &mut catalog.chunks, "", id, Some(chunk))?;
                    self.update(|catalog, _| &mut catalog.record_refs, "", id, None)?;
                }
                self.set(|catalog| &mut catalog.deleted_chunks, id, true)?;
            }
            _ => unreachable!(),
        }
        self.catalog.tombstone_count = count;
        Ok(())
    }

    pub(crate) fn commit(mut self) {
        assert!(
            self.pending_json.is_none(),
            "prepared metadata has not been transferred"
        );
        self.committed = true;
        // Free replaced payloads and old array buffers before returning their credit.
        self.undo.clear();
        if let Some(memory) = &mut self.catalog.memory {
            memory.reservation.retain(memory.live);
        }
    }
}

fn record_id(record: &RecordPayload) -> Option<&str> {
    match record {
        RecordPayload::CollectionCreate(value) => Some(&value.collection_id),
        RecordPayload::DocumentPut(value) => Some(&value.document_id),
        RecordPayload::ChunkPut(value) => Some(&value.chunk_id),
        RecordPayload::EmbeddingPut(value) => Some(&value.embedding_id),
        _ => None,
    }
}

impl Drop for CatalogTransaction<'_> {
    fn drop(&mut self) {
        if self.committed {
            return;
        }
        self.rollback_changes(false);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::tests::{chunk, collection, document};
    use crate::json::{JsonProcessingPolicyV1, WorkLedger};
    use std::sync::Arc;

    fn funded(record: &RecordPayload, context: &JsonOperationContext) -> RecordReservation {
        let mut json = context.reservation();
        json.grow(
            record.metadata_bytes_in(context).unwrap(),
            JsonInputKind::StoredJson,
        )
        .unwrap();
        let mut strings = context.workspace_reservation();
        let bytes = crate::storage::record_workspace_bytes(record);
        strings.grow_to(bytes, bytes, "catalog_build").unwrap();
        RecordReservation::new(json, strings)
    }

    fn apply(
        catalog: &mut Catalog,
        record: RecordPayload,
        context: &JsonOperationContext,
    ) -> Result<()> {
        let mut work = funded(&record, context);
        let mut transaction = catalog.transaction(context);
        transaction.apply(record, None, &mut work)?;
        transaction.commit();
        catalog.clear_model_states();
        Ok(())
    }

    fn replay(
        catalog: &mut Catalog,
        record: RecordPayload,
        context: &JsonOperationContext,
    ) -> Result<()> {
        let mut work = funded(&record, context);
        let mut transaction = catalog.transaction(context);
        transaction.rollback_on_drop = false;
        let result = transaction.apply(record, None, &mut work);
        assert_eq!(transaction.undo.capacity(), 0);
        result?;
        transaction.commit();
        catalog.clear_model_states();
        Ok(())
    }

    fn footprint(catalog: &Catalog, context: &JsonOperationContext) -> usize {
        fn map<V: Entry>(values: &Map<V>, context: &JsonOperationContext) -> usize {
            btree_workspace::<String, V>(values.len())
                + values
                    .iter()
                    .map(|(key, value)| key.capacity() + value.heap(context).unwrap())
                    .sum::<usize>()
        }
        fn set(values: &BTreeSet<String>) -> usize {
            btree_workspace::<String, ()>(values.len())
                + values.iter().map(String::capacity).sum::<usize>()
        }
        map(&catalog.collections, context)
            + map(&catalog.collection_names, context)
            + set(&catalog.reserved_collection_names)
            + map(&catalog.documents, context)
            + map(&catalog.document_external_ids.collections, context)
            + map(&catalog.chunks, context)
            + map(&catalog.chunk_external_ids.collections, context)
            + map(&catalog.document_chunks, context)
            + map(&catalog.embeddings, context)
            + map(&catalog.chunk_embeddings, context)
            + map(&catalog.record_refs, context)
            + map(&catalog.id_high_watermarks, context)
            + map(&catalog.collection_tombstones, context)
            + map(&catalog.document_tombstones, context)
            + map(&catalog.chunk_tombstones, context)
            + set(&catalog.deleted_collections)
            + set(&catalog.deleted_documents)
            + set(&catalog.deleted_chunks)
            + map(&catalog.collection_drop_times, context)
    }

    fn history(seed: usize) -> Vec<RecordPayload> {
        let mut records = vec![
            RecordPayload::CollectionCreate(collection(1)),
            RecordPayload::CollectionCreate(collection(2)),
        ];
        for id in 1..=2 {
            records.push(RecordPayload::DocumentPut(document(id, "col_1")));
        }
        for step in 0..80usize {
            let id = (step * 7 + seed) % 13 + 1;
            let owner = if step % 3 == 0 { "col_2" } else { "col_1" };
            let record = match step % 8 {
                0 | 1 | 2 | 6 => {
                    let mut value = chunk(id as u64, owner);
                    value.document_id = format!("doc_{}", step % 2 + 1);
                    value.chunk_external_id = format!("alias-{}", id % 5);
                    value.metadata = crate::json::object_from_pairs([(
                        "payload",
                        crate::JsonValue::String("x".repeat(seed + 1)),
                    )]);
                    RecordPayload::ChunkPut(value)
                }
                3 | 4 => RecordPayload::EmbeddingPut(Embedding {
                    embedding_id: format!("emb_{}", step + 1),
                    chunk_id: format!("chk_{id}"),
                    collection_id: owner.into(),
                    chunk_external_id: "old-external".into(),
                    model_name: "model".into(),
                    dim: 2,
                    vector: vec![1.0, 2.0],
                    created_at: "time".into(),
                }),
                5 => RecordPayload::Tombstone(crate::model::Tombstone {
                    target_type: "chunk".into(),
                    target_id: format!("chk_{id}"),
                    deleted_at: "time".into(),
                    reason: None,
                }),
                _ => {
                    let mut value = document((id % 2 + 1) as u64, owner);
                    value.external_id = format!("alias-{}", id % 3);
                    RecordPayload::DocumentPut(value)
                }
            };
            records.push(record);
        }
        records.push(RecordPayload::Tombstone(crate::model::Tombstone {
            target_type: "document".into(),
            target_id: "doc_1".into(),
            deleted_at: "time".into(),
            reason: None,
        }));
        records.push(RecordPayload::CollectionDrop {
            collection_id: "col_1".into(),
            name: "collection-1".into(),
            dropped_at: "time".into(),
            reason: None,
        });
        records
    }

    #[test]
    fn retained_catalog_matches_reference_and_exact_live_accounting() {
        for seed in 0..100 {
            let context = JsonOperationContext::standalone();
            let mut catalog = Catalog::new_in(&context);
            let mut recovered = Catalog::new_in(&context);
            let mut reference = Catalog::default();
            for record in history(seed) {
                let previous_key = match &record {
                    RecordPayload::DocumentPut(value) => catalog
                        .documents
                        .get_key_value(&value.document_id)
                        .map(|(key, _)| (value.document_id.clone(), key.as_ptr())),
                    _ => None,
                };
                reference.apply_with_ref(record.clone(), None).unwrap();
                replay(&mut recovered, record.clone(), &context).unwrap();
                apply(&mut catalog, record, &context).unwrap();
                if let Some((id, pointer)) = previous_key {
                    assert_eq!(
                        catalog.documents.get_key_value(&id).unwrap().0.as_ptr(),
                        pointer
                    );
                }
                assert_eq!(
                    catalog.state_signature(),
                    reference.state_signature(),
                    "seed={seed}"
                );
                let memory = catalog.memory.as_ref().unwrap();
                assert_eq!(memory.live, footprint(&catalog, &context));
                assert_eq!(memory.reservation.reserved_bytes(), memory.live);
                assert_eq!(recovered.state_signature(), reference.state_signature());
                assert_eq!(recovered.reserved_bytes(), footprint(&recovered, &context));
            }
            // Persistent allocations must not consume the JSON/operation child quota.
            context
                .workspace_reservation()
                .grow_to(
                    JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
                    JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
                    "operation_scratch",
                )
                .unwrap();
        }
    }

    #[test]
    fn preparation_rollback_keeps_input_credit_until_retry_or_cancel() {
        let mut partial_metadata = 0;
        let mut partial_changes = 0;
        for seed in 0..100 {
            const TOTAL: usize = 2 * 1024 * 1024;
            let context = JsonOperationContext::new(Arc::new(WorkLedger::with_limits(
                JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
                TOTAL,
                TOTAL,
            )));
            let baseline = || {
                vec![
                    RecordPayload::CollectionCreate(collection(1)),
                    RecordPayload::DocumentPut(document(1, "col_1")),
                    RecordPayload::ChunkPut(chunk(1, "col_1")),
                ]
            };
            let mut catalog = Catalog::new_in(&context);
            for record in baseline() {
                apply(&mut catalog, record, &context).unwrap();
            }
            let original = catalog.state_signature();
            let initial = catalog.reserved_bytes();
            let records = history(seed);
            let mut json = context.reservation();
            let metadata = records
                .iter()
                .map(|record| record.metadata_bytes_in(&context).unwrap())
                .sum();
            json.grow(metadata, JsonInputKind::StoredJson).unwrap();
            let mut cache = Some(
                context
                    .cache_reservation(
                        (TOTAL - initial - metadata - seed * 1024) as u128,
                        usize::MAX,
                    )
                    .unwrap(),
            );
            let mut transaction = catalog.transaction(&context);
            if let Err(error) = transaction.prepare(records.iter(), &mut json) {
                assert!(crate::json::is_memory_budget_error(&error), "{error:?}");
                let input: usize = transaction
                    .pending_json
                    .as_ref()
                    .map_or(0, |pending| pending.as_slice().iter().sum());
                partial_metadata += usize::from(input > 0);
                partial_changes += usize::from(!transaction.undo.is_empty());
                drop(cache.take());
                transaction.rollback_changes(true);
                assert_eq!(transaction.catalog.state_signature(), original);
                assert_eq!(transaction.catalog.reserved_bytes(), initial + input);
                assert_eq!(json.reserved_bytes(), metadata - input);
                transaction
                    .prepare_records(records.iter(), &mut json)
                    .unwrap();
            }
            drop(cache);
            assert_eq!(json.reserved_bytes(), 0);
            assert_eq!(
                transaction.pending_json.as_ref().unwrap().as_slice().len(),
                records.len()
            );
            let expected_context = JsonOperationContext::standalone();
            let mut expected = Catalog::new_in(&expected_context);
            for record in baseline() {
                apply(&mut expected, record, &expected_context).unwrap();
            }
            let expected_records = history(seed);
            let mut expected_json = expected_context.reservation();
            expected_json
                .grow(metadata, JsonInputKind::StoredJson)
                .unwrap();
            let mut expected_transaction = expected.transaction(&expected_context);
            expected_transaction
                .prepare(expected_records.iter(), &mut expected_json)
                .unwrap();
            assert_eq!(
                transaction.catalog.state_signature(),
                expected_transaction.catalog.state_signature()
            );
            assert_eq!(
                transaction.catalog.business_record_count,
                expected_transaction.catalog.business_record_count
            );
            assert_eq!(
                transaction.catalog.tombstone_count,
                expected_transaction.catalog.tombstone_count
            );
            drop(expected_records);
            drop(expected_transaction);
            drop(records);
            drop(transaction);
            drop(json);
            assert_eq!(catalog.state_signature(), original);
            assert_eq!(catalog.reserved_bytes(), initial);
            let remainder = context
                .cache_reservation((TOTAL - initial) as u128, usize::MAX)
                .unwrap();
            assert!(context.cache_reservation(1, usize::MAX).is_none());
            drop(remainder);
        }
        assert!(partial_metadata > 0 && partial_changes > 0);
    }

    #[test]
    fn insufficient_cache_reclaim_retries_once_and_returns_all_transaction_credit() {
        for seed in 0..100 {
            const TOTAL: usize = 1024 * 1024;
            let context =
                JsonOperationContext::new(Arc::new(WorkLedger::with_limits(TOTAL, TOTAL, TOTAL)));
            let mut catalog = Catalog::new_in(&context);
            let original = catalog.state_signature();
            let records = history(seed);
            let mut json = context.reservation();
            let metadata: usize = records
                .iter()
                .map(|record| record.metadata_bytes_in(&context).unwrap())
                .sum();
            json.grow(metadata, JsonInputKind::StoredJson).unwrap();
            let mut optional = Some(context.cache_reservation(16, usize::MAX).unwrap());
            let busy = context
                .cache_reservation((TOTAL - metadata - seed * 32 - 16) as u128, usize::MAX)
                .unwrap();
            let mut calls = 0;
            let mut transaction = catalog.transaction(&context);
            let error = transaction
                .prepare_reclaiming(records.iter(), &mut json, || {
                    calls += 1;
                    drop(optional.take());
                    true
                })
                .unwrap_err();
            assert_eq!(calls, 1);
            assert!(crate::json::is_memory_budget_error(&error), "{error:?}");
            drop(records);
            drop(transaction);
            drop(json);
            drop(busy);
            assert_eq!(catalog.state_signature(), original);
            assert_eq!(catalog.reserved_bytes(), 0);
            assert!(context
                .cache_reservation(TOTAL as u128, usize::MAX)
                .is_some());
        }
    }

    #[test]
    fn failed_or_cancelled_batches_restore_catalog_children_and_budget() {
        let baseline = || {
            vec![
                RecordPayload::CollectionCreate(collection(1)),
                RecordPayload::DocumentPut(document(1, "col_1")),
                RecordPayload::ChunkPut(chunk(1, "col_1")),
            ]
        };
        let context = JsonOperationContext::standalone();
        let mut initial = Catalog::new_in(&context);
        for record in baseline() {
            apply(&mut initial, record, &context).unwrap();
        }
        let initial_bytes = initial.memory.as_ref().unwrap().live;
        let mut refusals = 0;
        let mut prepared = 0;
        for seed in 0..100 {
            let catalog_limit = initial_bytes + seed * 4096;
            let context = JsonOperationContext::new(Arc::new(WorkLedger::with_limits(
                JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
                catalog_limit,
                2 * JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
            )));
            let mut catalog = Catalog::new_in(&context);
            for record in baseline() {
                apply(&mut catalog, record, &context).unwrap();
            }
            let expected = catalog.state_signature();
            let pointer = catalog.document_chunks["doc_1"].as_ptr();
            let capacity = catalog.document_chunks["doc_1"].capacity();
            let records = history(seed);
            let mut json = context.reservation();
            for record in &records {
                json.grow(
                    record.metadata_bytes_in(&context).unwrap(),
                    JsonInputKind::StoredJson,
                )
                .unwrap();
            }
            let mut transaction = catalog.transaction(&context);
            match transaction.prepare_reclaiming(records.iter(), &mut json, || {
                panic!("the independent Catalog limit cannot be relieved by cache eviction")
            }) {
                Ok(()) => prepared += 1,
                Err(error) => {
                    refusals += 1;
                    assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                    assert!(error
                        .detail()
                        .iter()
                        .any(|(key, value)| key == "resource" && value == "catalog"));
                }
            }
            drop(records);
            drop(transaction);
            drop(json);
            assert_eq!(catalog.state_signature(), expected);
            assert_eq!(catalog.document_chunks["doc_1"].as_ptr(), pointer);
            assert_eq!(catalog.document_chunks["doc_1"].capacity(), capacity);
            assert_eq!(
                catalog
                    .memory
                    .as_ref()
                    .unwrap()
                    .reservation
                    .reserved_bytes(),
                initial_bytes
            );
            assert_eq!(footprint(&catalog, &context), initial_bytes);
            let mut remainder = context.catalog_reservation();
            remainder.grow(catalog_limit - initial_bytes).unwrap();
            assert_eq!(
                remainder.grow(1).unwrap_err().code(),
                ErrorCode::ResourceExhausted
            );
        }
        assert!(refusals > 0 && prepared > 0);
    }

    #[test]
    fn failed_recovery_drops_partial_state_and_all_catalog_credit() {
        let mut failures = 0;
        for seed in 0..100 {
            let limit = (seed + 1) * 512;
            let context = JsonOperationContext::new(Arc::new(WorkLedger::with_limits(
                JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
                limit,
                2 * JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
            )));
            let mut catalog = Catalog::new_in(&context);
            for record in history(seed) {
                if let Err(error) = replay(&mut catalog, record, &context) {
                    assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                    assert!(error
                        .detail()
                        .iter()
                        .any(|(key, value)| key == "resource" && value == "catalog"));
                    failures += 1;
                    break;
                }
            }
            assert_eq!(catalog.reserved_bytes(), footprint(&catalog, &context));
            drop(catalog);
            context.catalog_reservation().grow(limit).unwrap();
            context
                .workspace_reservation()
                .grow_to(
                    JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
                    JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
                    "operation_scratch",
                )
                .unwrap();
        }
        assert!(failures > 0);
    }
}
