use super::*;
use crate::catalog::ModelState;
use crate::json::ReservedString;

impl Inner {
    pub(super) fn cache_collection_model(&mut self, collection_id: &str) -> Result<()> {
        self.catalog.cache_model_state(collection_id)
    }
}

pub(super) struct ModelAdmission {
    expected: Option<ReservedString>,
    historical_conflict: bool,
}

impl ModelAdmission {
    pub(super) fn new(
        catalog: &Catalog,
        collection: &CollectionInfo,
        context: &crate::json::JsonOperationContext,
    ) -> Result<Self> {
        let state = if catalog.collections.contains_key(&collection.collection_id) {
            catalog.visible_model_state(&collection.collection_id)?
        } else {
            collection
                .embedding_model
                .as_deref()
                .map_or(ModelState::Unbound, ModelState::Bound)
        };
        Ok(Self {
            expected: match state {
                ModelState::Bound(model) => Some(ReservedString::copy(model, context)?),
                _ => None,
            },
            historical_conflict: matches!(state, ModelState::Conflict(_)),
        })
    }

    pub(super) fn validate_write(
        &mut self,
        model: &str,
        declared: Option<&str>,
        context: &crate::json::JsonOperationContext,
    ) -> Result<()> {
        if self.historical_conflict {
            return Err(Error::new(
                ErrorCode::EmbeddingModelMismatch,
                "collection has conflicting visible embedding models; export and explicitly clean up its data before adding embeddings",
            ));
        }
        for expected in [declared, self.expected.as_ref().map(AsRef::as_ref)]
            .into_iter()
            .flatten()
        {
            if expected != model {
                return Err(Error::new(
                    ErrorCode::EmbeddingModelMismatch,
                    "input embedding model does not match the logical request",
                )
                .with_detail("expected", expected)
                .with_detail("actual", model));
            }
        }
        if self.expected.is_none() {
            self.expected = Some(ReservedString::copy(model, context)?);
        }
        Ok(())
    }
}
