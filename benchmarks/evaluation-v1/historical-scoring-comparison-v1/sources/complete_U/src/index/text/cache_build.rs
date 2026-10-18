use super::{cache::TextCacheOwner, TextCacheBuild, TextIndex};
use crate::json::JsonOperationContext;
use crate::storage::{catalog_id_text, TextSegmentEntry, TextSegmentVisitor};
use crate::{Error, ErrorCode, Result};
use std::sync::Arc;

impl TextIndex {
    pub(crate) fn try_cache_from_visitors<'a>(
        collections: impl IntoIterator<Item = &'a str>,
        context: &JsonOperationContext,
        limit: usize,
        mut visit: impl FnMut(&str, &mut TextSegmentVisitor<'_>) -> Result<bool>,
    ) -> Result<TextCacheBuild> {
        let mut index = Self {
            cache: Some(TextCacheOwner::new(context)),
            ..Self::default()
        };
        for id in collections {
            let mut owner = None::<Arc<str>>;
            let mut expected = None;
            let mut chunks = 0usize;
            let mut total = 0usize;
            if !visit(id, &mut |entry| {
                match entry {
                    TextSegmentEntry::Header { chunks } => expected = Some(chunks),
                    TextSegmentEntry::Chunk { length, .. } => {
                        chunks = chunks.checked_add(1).ok_or_else(cache_count_overflow)?;
                        total = total
                            .checked_add(length.max(1))
                            .ok_or_else(cache_count_overflow)?;
                    }
                    _ => {}
                }
                if !index.has_resident_cache() {
                    return Ok(());
                }
                let keep = match entry {
                    TextSegmentEntry::Header { .. } => true,
                    TextSegmentEntry::Chunk { id, length } => {
                        let mut bytes = [0; 24];
                        index.cache_chunk(
                            catalog_id_text(&mut bytes, b"chk_", id),
                            length.max(1),
                            limit,
                        )?
                    }
                    TextSegmentEntry::Term {
                        text,
                        document_frequency,
                    } => index.cache_term(text, document_frequency, limit)?,
                    TextSegmentEntry::Posting {
                        term,
                        chunk,
                        frequency,
                    } => {
                        if owner.is_none() {
                            owner = index.cache_collection_owner(id, limit);
                        }
                        if let Some(owner) = &owner {
                            let mut bytes = [0; 24];
                            index.cache_posting(
                                term,
                                catalog_id_text(&mut bytes, b"chk_", chunk),
                                frequency,
                                owner,
                                limit,
                            )?
                        } else {
                            false
                        }
                    }
                };
                if !keep {
                    // A collection-local Arc must drop before the candidate's credit.
                    owner = None;
                    index.clear_cache();
                }
                Ok(())
            })? {
                return Ok(TextCacheBuild::Invalid);
            }
            if expected != Some(chunks) {
                return Err(Error::new(
                    ErrorCode::IndexCorruption,
                    "text segment chunk count mismatch",
                ));
            }
            if chunks != 0
                && index.has_resident_cache()
                && !index.cache_collection(id, chunks, total, owner.take(), limit)?
            {
                index.clear_cache();
            }
        }
        if index.has_resident_cache() {
            index.sort_postings();
            Ok(TextCacheBuild::Complete(index))
        } else {
            Ok(TextCacheBuild::Miss)
        }
    }

    pub(crate) fn restore_empty_terms_from_visitors<'a>(
        &mut self,
        collections: impl IntoIterator<Item = &'a str>,
        limit: usize,
        mut visit: impl FnMut(&str, &mut TextSegmentVisitor<'_>) -> Result<bool>,
    ) -> Result<bool> {
        let result = (|| {
            for id in collections {
                if !visit(id, &mut |entry| {
                    if let TextSegmentEntry::Term {
                        text,
                        document_frequency: 0,
                    } = entry
                    {
                        if self.has_resident_cache() && !self.cache_term(text, 0, limit)? {
                            self.clear_cache();
                        }
                    }
                    Ok(())
                })? {
                    return Err(Error::new(
                        ErrorCode::IndexCorruption,
                        "incomplete prepared text checkpoint",
                    ));
                }
            }
            Ok(self.has_resident_cache())
        })();
        if result.is_err() {
            self.clear_cache();
        }
        result
    }
}

fn cache_count_overflow() -> Error {
    Error::new(ErrorCode::IndexCorruption, "text token count overflow")
}
