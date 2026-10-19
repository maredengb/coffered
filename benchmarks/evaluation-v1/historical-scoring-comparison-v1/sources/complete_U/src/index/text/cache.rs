use super::{CollectionCounts, Posting, TextIndex};
use crate::json::{btree_workspace, CacheReservation, JsonOperationContext};
use crate::{Error, ErrorCode, Result};
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::sync::Arc;

#[cfg(test)]
mod tests;

const TERMS: usize = 0;
const LENGTHS: usize = 1;
const COUNTS: usize = 2;
const AVERAGES: usize = 3;

pub(super) struct TextCacheOwner {
    nodes: [usize; 4],
    payload: usize,
    reservation: CacheReservation,
}

impl TextCacheOwner {
    pub(super) fn new(context: &JsonOperationContext) -> Self {
        Self {
            nodes: [0; 4],
            payload: 0,
            reservation: context
                .cache_reservation(0, usize::MAX)
                .expect("empty cache admission"),
        }
    }

    fn admit_node<K, V>(&mut self, tree: usize, count: usize, key: usize, limit: usize) -> bool {
        let nodes = btree_workspace::<K, V>(count).max(self.nodes[tree]);
        if !self
            .reservation
            .try_grow((nodes - self.nodes[tree]) as u128 + key as u128, limit)
        {
            return false;
        }
        self.nodes[tree] = nodes;
        self.payload += key;
        true
    }

    fn admit_payload(&mut self, bytes: u128, limit: usize) -> bool {
        if !self.reservation.try_grow(bytes, limit) {
            return false;
        }
        self.payload += bytes as usize;
        true
    }

    fn release_payload(&mut self, bytes: usize) {
        self.payload -= bytes;
        self.reservation
            .retain(self.nodes.iter().sum::<usize>() + self.payload);
    }

    fn transfer_payload(&mut self, target: &mut Self, bytes: usize) {
        self.reservation.transfer_to(&mut target.reservation, bytes);
        self.payload -= bytes;
        target.payload += bytes;
    }

    pub(super) fn reserved_bytes(&self) -> usize {
        self.reservation.reserved_bytes()
    }
}

fn arc_bytes(id: &str) -> u128 {
    let align = std::mem::align_of::<usize>() as u128;
    let bytes = 2 * std::mem::size_of::<usize>() as u128 + id.len() as u128;
    bytes.div_ceil(align) * align
}

fn copy_key(value: &str) -> Result<String> {
    let mut key = String::new();
    key.try_reserve_exact(value.len())
        .map_err(std::io::Error::other)?;
    if key.capacity() != value.len() {
        return Err(std::io::Error::other("text cache key capacity differs").into());
    }
    key.push_str(value);
    Ok(key)
}

impl TextIndex {
    pub(crate) fn has_resident_cache(&self) -> bool {
        self.cache.is_some()
    }

    pub(crate) fn reserved_bytes(&self) -> usize {
        self.cache
            .as_ref()
            .map_or(0, TextCacheOwner::reserved_bytes)
    }

    pub(crate) fn clear_cache(&mut self) -> bool {
        let bytes = self.reserved_bytes();
        self.postings = BTreeMap::new();
        self.chunk_lengths = BTreeMap::new();
        self.total_chunks_by_collection = BTreeMap::new();
        self.avg_len_by_collection = BTreeMap::new();
        self.cache = None;
        bytes != 0
    }

    pub(crate) fn collection_chunk_count(&self, id: &str) -> usize {
        self.total_chunks_by_collection
            .get(id)
            .map_or(0, |counts| counts.chunks)
    }

    pub(super) fn cache_chunk(&mut self, id: &str, length: usize, limit: usize) -> Result<bool> {
        if self.chunk_lengths.contains_key(id) {
            return Err(Error::new(
                ErrorCode::IndexCorruption,
                "duplicate text chunk",
            ));
        }
        if !self
            .cache
            .as_mut()
            .expect("resident cache owner")
            .admit_node::<String, usize>(LENGTHS, self.chunk_lengths.len() + 1, id.len(), limit)
        {
            return Ok(false);
        }
        let key = match copy_key(id) {
            Ok(key) => key,
            Err(error) => {
                self.cache.as_mut().unwrap().release_payload(id.len());
                return Err(error);
            }
        };
        self.chunk_lengths.insert(key, length);
        Ok(true)
    }

    pub(super) fn cache_term(
        &mut self,
        term: &str,
        additional: usize,
        limit: usize,
    ) -> Result<bool> {
        let owner = self.cache.as_mut().expect("resident cache owner");
        if !self.postings.contains_key(term) {
            if !owner.admit_node::<String, Vec<Posting>>(
                TERMS,
                self.postings.len() + 1,
                term.len(),
                limit,
            ) {
                return Ok(false);
            }
            let key = match copy_key(term) {
                Ok(key) => key,
                Err(error) => {
                    owner.release_payload(term.len());
                    return Err(error);
                }
            };
            self.postings.insert(key, Vec::new());
        }
        let postings = self.postings.get_mut(term).expect("text term installed");
        let count = postings
            .len()
            .checked_add(additional)
            .ok_or_else(|| Error::new(ErrorCode::IndexCorruption, "text posting count overflow"))?;
        if count <= postings.capacity() {
            return Ok(true);
        }
        // The existing array remains live until the fully admitted new array is installed.
        let preferred = postings
            .capacity()
            .checked_mul(2)
            .unwrap_or(count)
            .max(count);
        let width = std::mem::size_of::<Posting>() as u128;
        let capacity = if owner.admit_payload(preferred as u128 * width, limit) {
            preferred
        } else if preferred != count && owner.admit_payload(count as u128 * width, limit) {
            count
        } else {
            return Ok(false);
        };
        let bytes = capacity as u128 * width;
        let mut grown = Vec::new();
        if let Err(error) = grown.try_reserve_exact(capacity) {
            owner.release_payload(bytes as usize);
            return Err(std::io::Error::other(error).into());
        }
        if grown.capacity() != capacity {
            drop(grown);
            owner.release_payload(bytes as usize);
            return Err(std::io::Error::other("text cache array capacity differs").into());
        }
        let old = postings.capacity() * std::mem::size_of::<Posting>();
        grown.append(postings);
        drop(std::mem::replace(postings, grown));
        owner.release_payload(old);
        Ok(true)
    }

    pub(super) fn cache_collection_owner(&mut self, id: &str, limit: usize) -> Option<Arc<str>> {
        self.cache
            .as_mut()
            .expect("resident cache owner")
            .admit_payload(arc_bytes(id), limit)
            .then(|| Arc::from(id))
    }

    pub(super) fn cache_posting(
        &mut self,
        term: &str,
        id: &str,
        frequency: usize,
        collection: &Arc<str>,
        limit: usize,
    ) -> Result<bool> {
        let postings = self
            .postings
            .get_mut(term)
            .ok_or_else(|| Error::new(ErrorCode::IndexCorruption, "text posting has no term"))?;
        if postings.len() == postings.capacity() || frequency == 0 {
            return Err(Error::new(
                ErrorCode::IndexCorruption,
                "invalid text posting",
            ));
        }
        let owner = self.cache.as_mut().expect("resident cache owner");
        if !owner.admit_payload(id.len() as u128, limit) {
            return Ok(false);
        }
        let key = match copy_key(id) {
            Ok(key) => key,
            Err(error) => {
                owner.release_payload(id.len());
                return Err(error);
            }
        };
        postings.push(Posting {
            collection_id: Arc::clone(collection),
            chunk_id: key,
            term_frequency: frequency,
        });
        Ok(true)
    }

    pub(super) fn cache_collection(
        &mut self,
        id: &str,
        chunks: usize,
        total: usize,
        collection: Option<Arc<str>>,
        limit: usize,
    ) -> Result<bool> {
        if self.total_chunks_by_collection.contains_key(id) {
            return Err(Error::new(
                ErrorCode::IndexCorruption,
                "duplicate text collection",
            ));
        }
        let owner = self.cache.as_mut().expect("resident cache owner");
        if !owner.admit_node::<String, CollectionCounts>(
            COUNTS,
            self.total_chunks_by_collection.len() + 1,
            id.len(),
            limit,
        ) {
            return Ok(false);
        }
        if !owner.admit_node::<String, f64>(
            AVERAGES,
            self.avg_len_by_collection.len() + 1,
            id.len(),
            limit,
        ) {
            owner.release_payload(id.len());
            return Ok(false);
        }
        let keys = (|| -> Result<_> { Ok((copy_key(id)?, copy_key(id)?)) })();
        let (count_key, average_key) = match keys {
            Ok(keys) => keys,
            Err(error) => {
                owner.release_payload(2 * id.len());
                return Err(error);
            }
        };
        self.total_chunks_by_collection.insert(
            count_key,
            CollectionCounts {
                chunks,
                owner: collection,
            },
        );
        self.avg_len_by_collection
            .insert(average_key, total as f64 / chunks as f64);
        Ok(true)
    }

    pub(crate) fn replace_cached_collections<'a>(
        &mut self,
        collections: &BTreeSet<String>,
        old_chunk_ids: impl IntoIterator<Item = &'a str>,
        mut replacement: Self,
        limit: usize,
    ) -> Result<bool> {
        if self.cache.is_none() || replacement.cache.is_none() {
            self.clear_cache();
            return Ok(false);
        }
        self.cache
            .as_ref()
            .unwrap()
            .reservation
            .assert_same_ledger(&replacement.cache.as_ref().unwrap().reservation);
        let result = self.merge_cached(collections, old_chunk_ids, &mut replacement, limit);
        if !matches!(result, Ok(true)) {
            // Moved postings may still share allocations credited by replacement.
            self.clear_cache();
        }
        drop(replacement);
        result
    }

    fn merge_cached<'a>(
        &mut self,
        collections: &BTreeSet<String>,
        old_chunk_ids: impl IntoIterator<Item = &'a str>,
        replacement: &mut Self,
        limit: usize,
    ) -> Result<bool> {
        let mut removed = 0usize;
        self.postings.retain(|key, postings| {
            postings.retain(|posting| {
                if collections.contains(posting.collection_id.as_ref()) {
                    removed += posting.chunk_id.capacity();
                    false
                } else {
                    true
                }
            });
            if postings.is_empty() {
                removed += key.capacity() + postings.capacity() * std::mem::size_of::<Posting>();
                false
            } else {
                true
            }
        });
        for id in old_chunk_ids {
            if let Some((key, _)) = self.chunk_lengths.remove_entry(id) {
                removed += key.capacity();
                drop(key);
            }
        }
        self.total_chunks_by_collection.retain(|key, counts| {
            if collections.contains(key) {
                removed +=
                    key.capacity() + counts.owner.as_ref().map_or(0, |id| arc_bytes(id) as usize);
                false
            } else {
                true
            }
        });
        self.avg_len_by_collection.retain(|key, _| {
            if collections.contains(key) {
                removed += key.capacity();
                false
            } else {
                true
            }
        });
        self.cache.as_mut().unwrap().release_payload(removed);

        while let Some((term, mut postings)) = replacement.postings.pop_first() {
            let array = postings.capacity() * std::mem::size_of::<Posting>();
            let ids = postings
                .iter()
                .map(|posting| posting.chunk_id.capacity())
                .sum::<usize>();
            if self.postings.contains_key(&term) {
                if !self.cache_term(&term, postings.len(), limit)? {
                    return Ok(false);
                }
                let target = self.postings.get_mut(&term).unwrap();
                target.append(&mut postings);
                target.sort_unstable_by(super::posting_order);
                let released = term.capacity() + array;
                drop((term, postings));
                let source = replacement.cache.as_mut().unwrap();
                source.release_payload(released);
                source.transfer_payload(self.cache.as_mut().unwrap(), ids);
            } else {
                if !self
                    .cache
                    .as_mut()
                    .unwrap()
                    .admit_node::<String, Vec<Posting>>(TERMS, self.postings.len() + 1, 0, limit)
                {
                    return Ok(false);
                }
                replacement
                    .cache
                    .as_mut()
                    .unwrap()
                    .transfer_payload(self.cache.as_mut().unwrap(), term.capacity() + array + ids);
                self.postings.insert(term, postings);
            }
        }
        while let Some((key, length)) = replacement.chunk_lengths.pop_first() {
            if !self.cache.as_mut().unwrap().admit_node::<String, usize>(
                LENGTHS,
                self.chunk_lengths.len() + 1,
                0,
                limit,
            ) {
                return Ok(false);
            }
            replacement
                .cache
                .as_mut()
                .unwrap()
                .transfer_payload(self.cache.as_mut().unwrap(), key.capacity());
            self.chunk_lengths.insert(key, length);
        }
        // Collection anchors move only after all their postings and ID storage.
        while let Some((key, counts)) = replacement.total_chunks_by_collection.pop_first() {
            if !self
                .cache
                .as_mut()
                .unwrap()
                .admit_node::<String, CollectionCounts>(
                    COUNTS,
                    self.total_chunks_by_collection.len() + 1,
                    0,
                    limit,
                )
            {
                return Ok(false);
            }
            let bytes =
                key.capacity() + counts.owner.as_ref().map_or(0, |id| arc_bytes(id) as usize);
            replacement
                .cache
                .as_mut()
                .unwrap()
                .transfer_payload(self.cache.as_mut().unwrap(), bytes);
            self.total_chunks_by_collection.insert(key, counts);
        }
        while let Some((key, average)) = replacement.avg_len_by_collection.pop_first() {
            if !self.cache.as_mut().unwrap().admit_node::<String, f64>(
                AVERAGES,
                self.avg_len_by_collection.len() + 1,
                0,
                limit,
            ) {
                return Ok(false);
            }
            replacement
                .cache
                .as_mut()
                .unwrap()
                .transfer_payload(self.cache.as_mut().unwrap(), key.capacity());
            self.avg_len_by_collection.insert(key, average);
        }
        Ok(true)
    }
}
