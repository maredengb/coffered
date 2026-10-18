use std::collections::BTreeMap;

#[derive(Clone, Debug, Default)]
pub(crate) struct ExternalIds {
    pub(super) collections: BTreeMap<String, BTreeMap<String, String>>,
}

impl ExternalIds {
    pub(crate) fn get(&self, collection: &str, external: &str) -> Option<&String> {
        self.collections.get(collection)?.get(external)
    }

    pub(crate) fn find_external(&self, external: &str) -> Option<&String> {
        self.collections.values().find_map(|ids| ids.get(external))
    }

    #[cfg(test)]
    pub(super) fn insert(&mut self, collection: &str, external: &str, id: &str) {
        if let Some(ids) = self.collections.get_mut(collection) {
            ids.insert(external.to_owned(), id.to_owned());
        } else {
            self.collections.insert(
                collection.to_owned(),
                BTreeMap::from([(external.to_owned(), id.to_owned())]),
            );
        }
    }

    #[cfg(test)]
    pub(super) fn remove_if_owned(&mut self, collection: &str, external: &str, id: &str) {
        if let Some(ids) = self.collections.get_mut(collection) {
            // A newer object may already own the alias when its former owner moves.
            if ids.get(external).is_some_and(|owner| owner == id) {
                ids.remove(external);
                if ids.is_empty() {
                    self.collections.remove(collection);
                }
            }
        }
    }

    #[cfg(test)]
    pub(super) fn len(&self) -> usize {
        self.collections.values().map(BTreeMap::len).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grouped_aliases_match_flat_reference_across_reassignments_and_cleanup() {
        for seed in 0..100usize {
            let mut aliases = ExternalIds::default();
            let mut reference = BTreeMap::new();
            for step in 0..200 {
                let collection = format!("col_{}", (step * 7 + seed) % 5);
                let external = format!("external-{}", (step * 11 + seed) % 17);
                let id = format!("id_{}", step % 13);
                let key = (collection.clone(), external.clone());
                if step % 3 != 0 {
                    aliases.insert(&collection, &external, &id);
                    reference.insert(key, id.clone());
                } else {
                    aliases.remove_if_owned(&collection, &external, &id);
                    if reference.get(&key) == Some(&id) {
                        reference.remove(&key);
                    }
                }
                assert_eq!(
                    aliases.get(&collection, &external),
                    reference.get(&(collection.clone(), external.clone()))
                );
                assert_eq!(
                    aliases.find_external(&external),
                    reference
                        .iter()
                        .find(|((_, name), _)| name == &external)
                        .map(|(_, id)| id)
                );
                assert_eq!(aliases.len(), reference.len());
                assert!(aliases.collections.values().all(|ids| !ids.is_empty()));
            }
            let cloned = aliases.clone();
            for ((collection, external), id) in &reference {
                aliases.remove_if_owned(collection, external, "wrong-owner");
                assert_eq!(aliases.get(collection, external), Some(id));
                aliases.remove_if_owned(collection, external, id);
                assert!(aliases.get(collection, external).is_none());
                assert_eq!(cloned.get(collection, external), Some(id));
            }
            assert!(aliases.collections.is_empty());
            assert!(aliases.find_external("external-0").is_none());
            assert_eq!(cloned.len(), reference.len());
        }
    }
}
