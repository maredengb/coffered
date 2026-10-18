use super::*;

// The optional snapshot cache owns no heap storage; long historical IDs use the live scan.
#[derive(Clone, Copy, Debug)]
struct InlineId {
    bytes: [u8; 32],
    len: u8,
}

impl InlineId {
    fn new(value: &str) -> Option<Self> {
        if value.len() > 32 {
            return None;
        }
        let mut bytes = [0; 32];
        bytes[..value.len()].copy_from_slice(value.as_bytes());
        Some(Self {
            bytes,
            len: value.len() as u8,
        })
    }

    fn as_str(&self) -> &str {
        std::str::from_utf8(&self.bytes[..usize::from(self.len)])
            .expect("inline ID preserves complete UTF-8 bytes")
    }
}

#[derive(Clone, Copy, Debug)]
enum CachedModelState {
    Empty,
    Unique(InlineId),
    Conflict(InlineId),
}

#[derive(Clone, Debug, Default)]
pub(super) struct ModelStates {
    entries: [Option<(InlineId, CachedModelState)>; 8],
    next: usize,
}

impl ModelStates {
    fn get(&self, collection_id: &str) -> Option<&CachedModelState> {
        self.entries
            .iter()
            .flatten()
            .find(|(id, _)| id.as_str() == collection_id)
            .map(|(_, state)| state)
    }

    fn insert(&mut self, collection_id: InlineId, state: CachedModelState) {
        self.entries[self.next] = Some((collection_id, state));
        self.next = (self.next + 1) % self.entries.len();
    }

    fn invalidate(&mut self, collection_id: &str) {
        for entry in &mut self.entries {
            if entry
                .as_ref()
                .is_some_and(|(id, _)| id.as_str() == collection_id)
            {
                *entry = None;
            }
        }
    }

    pub(super) fn clear(&mut self) {
        *self = Self::default();
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum ModelState<'a> {
    Unbound,
    Bound(&'a str),
    Conflict(&'a Embedding),
}

impl ModelState<'_> {
    pub(crate) fn validate(self, actual: Option<&str>) -> Result<()> {
        match self {
            Self::Conflict(witness) => Err(Error::new(
                ErrorCode::EmbeddingModelMismatch,
                "collection has conflicting visible embedding models; export and explicitly clean up its data",
            )
            .with_detail("collection_id", &witness.collection_id)
            .with_detail("embedding_id", &witness.embedding_id)),
            Self::Bound(expected) if actual.is_some_and(|actual| actual != expected) => {
                Err(Error::new(
                    ErrorCode::EmbeddingModelMismatch,
                    "input embedding model does not match the collection snapshot",
                )
                .with_detail("expected", expected)
                .with_detail("actual", actual.unwrap()))
            }
            _ => Ok(()),
        }
    }
}

impl Catalog {
    fn model_witness(&self, collection_id: &str) -> Result<(Option<&Embedding>, bool)> {
        let collection = self.collection_by_id(collection_id)?;
        let mut expected = collection.embedding_model.as_deref();
        let mut witness = None;
        for embedding in self
            .live_embeddings()
            .filter(|embedding| embedding.collection_id == collection_id)
        {
            if expected.is_some_and(|model| model != embedding.model_name) {
                return Ok((Some(embedding), true));
            }
            if witness.is_none() {
                witness = Some(embedding);
                expected = Some(&embedding.model_name);
            }
        }
        Ok((witness, false))
    }

    pub(crate) fn visible_model_state(&self, collection_id: &str) -> Result<ModelState<'_>> {
        let collection = self.collection_by_id(collection_id)?;
        if let Some(cached) = self.model_states.get(collection_id) {
            return Ok(match cached {
                CachedModelState::Empty => collection
                    .embedding_model
                    .as_deref()
                    .map_or(ModelState::Unbound, ModelState::Bound),
                CachedModelState::Unique(id) => {
                    ModelState::Bound(collection.embedding_model.as_deref().unwrap_or_else(|| {
                        &self
                            .embeddings
                            .get(id.as_str())
                            .expect("validated model witness")
                            .model_name
                    }))
                }
                CachedModelState::Conflict(id) => ModelState::Conflict(
                    self.embeddings
                        .get(id.as_str())
                        .expect("validated conflict witness"),
                ),
            });
        }
        let (witness, conflict) = self.model_witness(collection_id)?;
        if conflict {
            return Ok(ModelState::Conflict(
                witness.expect("conflicting model witness"),
            ));
        }
        Ok(collection
            .embedding_model
            .as_deref()
            .or_else(|| witness.map(|embedding| embedding.model_name.as_str()))
            .map_or(ModelState::Unbound, ModelState::Bound))
    }

    pub(crate) fn cache_model_state(&mut self, collection_id: &str) -> Result<()> {
        if self.model_states.get(collection_id).is_some() {
            return Ok(());
        }
        let (witness, conflict) = self.model_witness(collection_id)?;
        let Some(id) = InlineId::new(collection_id) else {
            return Ok(());
        };
        let state = match witness {
            Some(embedding) => {
                let Some(witness_id) = InlineId::new(&embedding.embedding_id) else {
                    return Ok(());
                };
                if conflict {
                    CachedModelState::Conflict(witness_id)
                } else {
                    CachedModelState::Unique(witness_id)
                }
            }
            None => CachedModelState::Empty,
        };
        self.model_states.insert(id, state);
        Ok(())
    }

    pub(crate) fn invalidate_model_state(&mut self, collection_id: &str) {
        self.model_states.invalidate(collection_id);
    }

    pub(super) fn clear_model_states(&mut self) {
        self.model_states.clear();
    }

    pub(crate) fn require_consistent_models(&self) -> Result<()> {
        for collection in self.live_collections() {
            self.visible_model_state(&collection.collection_id)?
                .validate(None)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::tests::{chunk, collection, document};

    fn fixture(explicit: Option<&str>, models: &[&str]) -> Catalog {
        fixture_with_ids(explicit, models, "col_1", "emb_")
    }

    fn fixture_with_ids(
        explicit: Option<&str>,
        models: &[&str],
        collection_id: &str,
        embedding_prefix: &str,
    ) -> Catalog {
        let mut catalog = Catalog::new_in(&crate::json::JsonOperationContext::standalone());
        let mut info = collection(1);
        info.collection_id = collection_id.to_owned();
        info.embedding_model = explicit.map(str::to_owned);
        catalog
            .apply_with_ref(RecordPayload::CollectionCreate(info), None)
            .unwrap();
        for (index, model) in models.iter().enumerate() {
            let number = index as u64 + 1;
            catalog
                .apply_with_ref(
                    RecordPayload::DocumentPut(document(number, collection_id)),
                    None,
                )
                .unwrap();
            catalog
                .apply_with_ref(RecordPayload::ChunkPut(chunk(number, collection_id)), None)
                .unwrap();
            catalog
                .apply_with_ref(
                    RecordPayload::EmbeddingPut(Embedding {
                        embedding_id: format!("{embedding_prefix}{number}"),
                        chunk_id: format!("chk_{number}"),
                        collection_id: collection_id.into(),
                        chunk_external_id: format!("chunk-{number}"),
                        model_name: (*model).into(),
                        dim: 2,
                        vector: Vec::new(),
                        created_at: "1970-01-01T00:00:00Z".into(),
                    }),
                    None,
                )
                .unwrap();
        }
        catalog
    }

    #[test]
    fn model_state_matrix_uses_exact_visible_identity() {
        for (explicit, models, bound, conflict) in [
            (None, vec![], None, false),
            (Some("M"), vec![], Some("M"), false),
            (None, vec!["M", "M"], Some("M"), false),
            (Some("M"), vec!["M"], Some("M"), false),
            (None, vec!["M", "m"], None, true),
            (Some("M"), vec!["m"], None, true),
        ] {
            let mut catalog = fixture(explicit, &models);
            let before = catalog.state_signature();
            catalog.cache_model_state("col_1").unwrap();
            match catalog.visible_model_state("col_1").unwrap() {
                ModelState::Unbound => assert!(bound.is_none() && !conflict),
                ModelState::Bound(actual) => assert_eq!(Some(actual), bound),
                ModelState::Conflict(witness) => {
                    assert!(conflict);
                    assert!(catalog.is_chunk_live(&witness.chunk_id));
                }
            }
            assert_eq!(catalog.state_signature(), before);
            let bytes = catalog.reserved_bytes();
            catalog.cache_model_state("col_1").unwrap();
            assert_eq!(catalog.reserved_bytes(), bytes);
        }
    }

    #[test]
    fn model_cache_has_no_heap_storage_or_catalog_budget_growth() {
        let mut catalog = fixture(None, &["M"]);
        let before = catalog.reserved_bytes();
        let (_, allocations) = crate::test_alloc::allocations(|| {
            catalog.cache_model_state("col_1").unwrap();
            assert!(matches!(
                catalog.visible_model_state("col_1").unwrap(),
                ModelState::Bound("M")
            ));
            catalog.cache_model_state("col_1").unwrap();
            catalog.invalidate_model_state("absent");
            catalog.invalidate_model_state("col_1");
            catalog.cache_model_state("col_1").unwrap();
            catalog.clear_model_states();
        });
        assert_eq!(allocations, 0);
        assert!(std::mem::size_of::<ModelStates>() <= 1024);
        assert_eq!(catalog.reserved_bytes(), before);
        assert!(matches!(
            catalog.visible_model_state("col_1").unwrap(),
            ModelState::Bound("M")
        ));
    }

    #[test]
    fn model_cache_eviction_and_invalidation_preserve_other_collections() {
        let mut catalog = Catalog::new_in(&crate::json::JsonOperationContext::standalone());
        for number in 1..=9 {
            let mut info = collection(number);
            info.embedding_model = Some(format!("M{number}"));
            catalog
                .apply_with_ref(RecordPayload::CollectionCreate(info), None)
                .unwrap();
        }
        let before = catalog.reserved_bytes();
        for number in 1..=9 {
            catalog.cache_model_state(&format!("col_{number}")).unwrap();
        }
        assert!(catalog.model_states.get("col_1").is_none());
        assert!(catalog.model_states.get("col_9").is_some());
        catalog.invalidate_model_state("col_9");
        assert!(catalog.model_states.get("col_9").is_none());
        assert!(catalog.model_states.get("col_8").is_some());
        for number in 1..=9 {
            let id = format!("col_{number}");
            let ModelState::Bound(actual) = catalog.visible_model_state(&id).unwrap() else {
                panic!("configured collection must remain bound");
            };
            assert_eq!(actual, format!("M{number}"));
            catalog.cache_model_state(&id).unwrap();
        }
        assert_eq!(catalog.reserved_bytes(), before);
    }

    #[test]
    fn long_utf8_ids_fall_back_without_restricting_model_identity() {
        let long = "x".repeat(64);
        for (collection_id, prefix) in [
            (long.as_str(), "emb_"),
            ("col_1", long.as_str()),
            (long.as_str(), long.as_str()),
        ] {
            for models in [vec!["Model"], vec!["Model", "model"]] {
                let mut catalog = fixture_with_ids(None, &models, collection_id, prefix);
                let (_, allocations) = crate::test_alloc::allocations(|| {
                    catalog.cache_model_state(collection_id).unwrap();
                });
                assert_eq!(allocations, 0);
                assert!(catalog.model_states.get(collection_id).is_none());
                match catalog.visible_model_state(collection_id).unwrap() {
                    ModelState::Bound(actual) => assert_eq!(actual, "Model"),
                    ModelState::Conflict(witness) => assert_eq!(witness.model_name, "model"),
                    ModelState::Unbound => panic!("live models cannot be unbound"),
                }
            }
        }
        let utf8 = "\u{e9}".repeat(16);
        assert_eq!(InlineId::new(&utf8).unwrap().as_str(), utf8);
        assert!(InlineId::new(&(utf8.clone() + "\u{e9}")).is_none());
    }

    #[test]
    fn deleted_and_replaced_embeddings_do_not_bind() {
        let mut catalog = fixture(None, &["M", "N"]);
        catalog.cache_model_state("col_1").unwrap();
        catalog
            .apply_with_ref(
                RecordPayload::Tombstone(Tombstone {
                    target_type: "document".into(),
                    target_id: "doc_2".into(),
                    deleted_at: "now".into(),
                    reason: None,
                }),
                None,
            )
            .unwrap();
        assert!(matches!(
            catalog.visible_model_state("col_1").unwrap(),
            ModelState::Bound("M")
        ));
        catalog
            .apply_with_ref(
                RecordPayload::EmbeddingPut(Embedding {
                    embedding_id: "emb_3".into(),
                    chunk_id: "chk_1".into(),
                    collection_id: "col_1".into(),
                    chunk_external_id: "chunk-1".into(),
                    model_name: "N".into(),
                    dim: 2,
                    vector: Vec::new(),
                    created_at: "now".into(),
                }),
                None,
            )
            .unwrap();
        assert!(matches!(
            catalog.visible_model_state("col_1").unwrap(),
            ModelState::Bound("N")
        ));
        catalog
            .apply_with_ref(
                RecordPayload::Tombstone(Tombstone {
                    target_type: "chunk".into(),
                    target_id: "chk_1".into(),
                    deleted_at: "now".into(),
                    reason: None,
                }),
                None,
            )
            .unwrap();
        assert!(matches!(
            catalog.visible_model_state("col_1").unwrap(),
            ModelState::Unbound
        ));
    }
}
