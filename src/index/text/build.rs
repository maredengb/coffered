#[cfg(test)]
use super::{CollectionCounts, Posting, TextIndex};
#[cfg(test)]
use crate::catalog::{Catalog, VisibleChunk};
#[cfg(test)]
use crate::json::{is_memory_budget_error, JsonOperationContext, ReservedVec};
#[cfg(test)]
use crate::storage::{catalog_id_text, TextSegmentEntry, TextSegmentVisitor};
#[cfg(test)]
use crate::temporary::{ExternalTermFrequencies, TemporaryDiskBudget};
#[cfg(test)]
use crate::{Error, ErrorCode, Result};
#[cfg(test)]
use std::path::Path;
#[cfg(test)]
use std::sync::Arc;

#[cfg(test)]
impl TextIndex {
    pub(crate) fn try_build_from_visitors<'a>(
        collections: impl IntoIterator<Item = &'a str>,
        mut visit: impl FnMut(&str, &mut TextSegmentVisitor<'_>) -> Result<bool>,
    ) -> Result<Option<Self>> {
        let mut index = Self::default();
        for id in collections {
            let mut owner = None::<Arc<str>>;
            let mut expected = None;
            let mut chunks = 0usize;
            let mut total = 0usize;
            if !visit(id, &mut |entry| {
                match entry {
                    TextSegmentEntry::Header { chunks } => expected = Some(chunks),
                    TextSegmentEntry::Chunk { id, length } => {
                        let mut bytes = [0; 24];
                        let key = catalog_id_text(&mut bytes, b"chk_", id);
                        index.chunk_lengths.insert(key.to_owned(), length.max(1));
                        chunks = chunks.checked_add(1).ok_or_else(count_overflow)?;
                        total = total
                            .checked_add(length.max(1))
                            .ok_or_else(count_overflow)?;
                    }
                    TextSegmentEntry::Term {
                        text,
                        document_frequency,
                    } => {
                        let postings = index.postings.entry(text.to_owned()).or_default();
                        postings
                            .try_reserve_exact(document_frequency)
                            .map_err(|error| std::io::Error::other(error.to_string()))?;
                    }
                    TextSegmentEntry::Posting {
                        term,
                        chunk,
                        frequency,
                    } => {
                        let postings = index.postings.get_mut(term).ok_or_else(|| {
                            Error::new(ErrorCode::IndexCorruption, "text posting has no term")
                        })?;
                        if postings.len() == postings.capacity() || frequency == 0 {
                            return Err(Error::new(
                                ErrorCode::IndexCorruption,
                                "invalid text posting",
                            ));
                        }
                        let mut bytes = [0; 24];
                        postings.push(Posting {
                            collection_id: Arc::clone(owner.get_or_insert_with(|| Arc::from(id))),
                            chunk_id: catalog_id_text(&mut bytes, b"chk_", chunk).to_owned(),
                            term_frequency: frequency,
                        });
                    }
                }
                Ok(())
            })? {
                return Ok(None);
            }
            if expected != Some(chunks) {
                return Err(Error::new(
                    ErrorCode::IndexCorruption,
                    "text segment chunk count mismatch",
                ));
            }
            if chunks != 0 {
                index
                    .total_chunks_by_collection
                    .insert(id.to_owned(), CollectionCounts::count_only(chunks));
                index
                    .avg_len_by_collection
                    .insert(id.to_owned(), total as f64 / chunks as f64);
            }
        }
        index.sort_postings();
        Ok(Some(index))
    }

    #[cfg(test)]
    pub(crate) fn try_build_matching_in<T: AsRef<str>>(
        catalog: &Catalog,
        include: impl Fn(&str) -> bool,
        mut load: impl FnMut(&VisibleChunk<'_>) -> Result<T>,
        directory: &Path,
        disk: Arc<TemporaryDiskBudget>,
        context: &JsonOperationContext,
    ) -> Result<Self> {
        let mut index = Self::default();
        let mut lengths = ReservedVec::with_capacity(
            catalog
                .live_collections()
                .filter(|collection| include(&collection.collection_id))
                .count(),
            context,
            "operation_scratch",
        )?;
        for collection in catalog
            .live_collections()
            .filter(|collection| include(&collection.collection_id))
        {
            lengths.push((
                collection.collection_id.as_str(),
                0usize,
                0usize,
                None::<Arc<str>>,
            ));
        }
        lengths
            .as_mut_slice()
            .sort_unstable_by_key(|&(id, _, _, _)| id);
        let mut frequencies = None;
        for visible in catalog.visible_snapshot().live_chunks() {
            let chunk = visible.chunk;
            if !include(&chunk.collection_id) {
                continue;
            }
            let position = lengths
                .as_slice()
                .binary_search_by_key(&chunk.collection_id.as_str(), |&(id, _, _, _)| id)
                .map_err(|_| {
                    Error::new(
                        ErrorCode::FileFormatError,
                        "text chunk has no live collection",
                    )
                })?;
            let text = match load(&visible) {
                Ok(text) => text,
                Err(error) if is_memory_budget_error(&error) => {
                    let released = match frequencies.as_mut() {
                        Some(sorter) => {
                            ExternalTermFrequencies::release_empty_batch_capacity(sorter)?
                        }
                        None => false,
                    };
                    if !released {
                        return Err(error);
                    }
                    load(&visible)?
                }
                Err(error) => return Err(error),
            };
            let mut count = 0usize;
            crate::tokenizer::visit_tokens_in_reclaiming(
                text.as_ref(),
                context,
                &mut frequencies,
                |frequencies, token| {
                    count = count.checked_add(1).ok_or_else(count_overflow)?;
                    frequencies
                        .get_or_insert_with(|| {
                            ExternalTermFrequencies::new(directory, disk.clone(), context)
                        })
                        .push(&[token.as_bytes()], 1)
                },
                |frequencies| match frequencies.as_mut() {
                    Some(sorter) => Ok(sorter.release_empty_batch_capacity()? || sorter.spill()?),
                    None => Ok(false),
                },
            )?;
            drop(text);
            let count = count.max(1);
            index.chunk_lengths.insert(chunk.chunk_id.clone(), count);
            let (_, total, chunks, collection) = &mut lengths.as_mut_slice()[position];
            *total = total.checked_add(count).ok_or_else(count_overflow)?;
            *chunks = chunks.checked_add(1).ok_or_else(count_overflow)?;
            if let Some(sorter) = frequencies.as_mut() {
                sorter.finish_batch(|key, value| {
                    let term = std::str::from_utf8(key).map_err(|_| {
                        Error::new(ErrorCode::FileFormatError, "invalid text term key")
                    })?;
                    let posting = Posting {
                        collection_id: Arc::clone(
                            collection
                                .get_or_insert_with(|| Arc::from(chunk.collection_id.as_str())),
                        ),
                        chunk_id: chunk.chunk_id.clone(),
                        term_frequency: usize::try_from(value).map_err(|_| count_overflow())?,
                    };
                    if let Some(postings) = index.postings.get_mut(term) {
                        postings.push(posting);
                    } else {
                        index.postings.insert(term.to_owned(), vec![posting]);
                    }
                    Ok(())
                })?;
            }
        }
        for &(id, total, chunks, _) in lengths.as_slice() {
            if chunks != 0 {
                index
                    .total_chunks_by_collection
                    .insert(id.to_owned(), CollectionCounts::count_only(chunks));
                index
                    .avg_len_by_collection
                    .insert(id.to_owned(), total as f64 / chunks as f64);
            }
        }
        index.sort_postings();
        Ok(index)
    }
}

#[cfg(test)]
fn count_overflow() -> Error {
    Error::new(ErrorCode::FileFormatError, "text token count overflow")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::text::{streamed::tests::fixture, TextSearchOptions};
    use crate::json::{ReservedString, WorkLedger};
    use std::fs;

    struct Directory(std::path::PathBuf);
    impl Directory {
        fn new(label: &str) -> Self {
            static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            let path = std::env::temp_dir().join(format!(
                "cofferdb-text-build-{label}-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
        fn clean(&self, disk: &TemporaryDiskBudget) {
            assert_eq!(disk.used(), 0);
            assert_eq!(fs::read_dir(&self.0).unwrap().count(), 0);
        }
    }
    impl Drop for Directory {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.0).unwrap();
        }
    }

    fn assert_same(
        actual: &TextIndex,
        expected: &TextIndex,
        catalog: &Catalog,
        context: &JsonOperationContext,
    ) {
        assert_eq!(actual.chunk_lengths, expected.chunk_lengths);
        assert_eq!(actual.collection_counts(), expected.collection_counts());
        assert_eq!(actual.avg_len_by_collection, expected.avg_len_by_collection);
        let signature = |index: &TextIndex| {
            index
                .postings
                .iter()
                .map(|(term, postings)| {
                    (
                        term.clone(),
                        postings
                            .iter()
                            .map(|posting| {
                                (
                                    posting.collection_id.to_string(),
                                    posting.chunk_id.clone(),
                                    posting.term_frequency,
                                )
                            })
                            .collect::<Vec<_>>(),
                    )
                })
                .collect::<std::collections::BTreeMap<_, _>>()
        };
        assert_eq!(signature(actual), signature(expected));
        for collection in catalog.live_collections() {
            for query in [
                "shared delta",
                "alphaBeta storage_path",
                "the and",
                "\u{0130} \u{03a3}",
                "token5 token99",
            ] {
                let actual = actual
                    .search(
                        &collection.collection_id,
                        query,
                        TextSearchOptions::with_explanations(15),
                        context,
                    )
                    .unwrap();
                let expected = expected
                    .search(
                        &collection.collection_id,
                        query,
                        TextSearchOptions::with_explanations(15),
                        context,
                    )
                    .unwrap();
                assert_eq!(
                    format!("{:?}", actual.as_slice()),
                    format!("{:?}", expected.as_slice())
                );
            }
        }
        for collection in catalog.live_collections() {
            let owners = actual
                .postings
                .values()
                .flatten()
                .filter(|posting| posting.collection_id.as_ref() == collection.collection_id)
                .map(|posting| &posting.collection_id)
                .collect::<Vec<_>>();
            if let Some(first) = owners.first() {
                assert!(owners.iter().all(|owner| Arc::ptr_eq(first, owner)));
                assert_eq!(Arc::strong_count(first), owners.len());
            }
        }
    }

    #[test]
    fn visitor_posting_reservation_preserves_native_errors_and_rejects_incomplete_sources() {
        let build = || {
            TextIndex::try_build_from_visitors(["col_0000000000000001"], |_, visit| {
                visit(TextSegmentEntry::Header { chunks: 0 })?;
                visit(TextSegmentEntry::Term {
                    text: "needle",
                    document_frequency: 257,
                })?;
                Ok(false)
            })
        };
        let (result, allocations) = crate::test_alloc::allocations(build);
        assert!(result.unwrap().is_none());
        // The dictionary key and first BTree node precede the fallible posting array.
        assert_eq!(allocations, 3);
        let error = crate::test_alloc::fail_nth_allocation_once(allocations, build).unwrap_err();
        assert_eq!(error.code(), ErrorCode::IoError);
        assert!(!crate::json::is_memory_budget_error(&error));
        assert!(build().unwrap().is_none());
    }

    #[test]
    fn bounded_tf_build_matches_reference_queries_and_selected_collection_visibility() {
        let directory = Directory::new("oracle");
        let context = JsonOperationContext::standalone();
        let available = context.available_workspace_bytes();
        let disk = Arc::new(TemporaryDiskBudget::new(16 * 1024 * 1024));
        for seed in 0..100 {
            let mut catalog = fixture(seed);
            if seed % 25 == 1 {
                if let Some(chunk) = catalog.chunks.values_mut().next() {
                    chunk.text = (0..4000).map(|n| format!("token{n} ")).collect();
                }
            }
            let signature = catalog.state_signature();
            for selected in [None, Some("first")] {
                let include = |id: &str| selected.map_or(true, |selected| id == selected);
                let expected = TextIndex::try_build_matching(
                    &catalog.visible_snapshot(),
                    include,
                    |visible| Ok(visible.chunk.text.clone()),
                )
                .unwrap();
                let actual = TextIndex::try_build_matching_in(
                    &catalog,
                    include,
                    |visible| ReservedString::copy(&visible.chunk.text, &context),
                    &directory.0,
                    disk.clone(),
                    &context,
                )
                .unwrap();
                assert_same(&actual, &expected, &catalog, &context);
                assert_eq!(context.available_workspace_bytes(), available);
                directory.clean(&disk);
                assert_eq!(catalog.state_signature(), signature);
            }
        }
    }

    #[test]
    fn interleaved_collections_finish_each_chunk_before_the_next_read() {
        let directory = Directory::new("interleaved");
        for large in [false, true] {
            let mut catalog = fixture(1);
            let template = catalog.chunks["first-0"].clone();
            catalog.chunks.clear();
            catalog.deleted_chunks.clear();
            for ordinal in 0..320 {
                for owner in ["first", "second"] {
                    let mut chunk = template.clone();
                    let id = format!("chunk{ordinal:04}-{owner}");
                    chunk.chunk_id = id.clone();
                    chunk.chunk_external_id = id.clone();
                    chunk.collection_id = owner.into();
                    chunk.document_id = owner.into();
                    chunk.ordinal = ordinal;
                    let words = if large && ordinal == 159 && owner == "first" {
                        4000
                    } else {
                        32
                    };
                    chunk.text = (0..words).map(|i| format!("token{i} ")).collect::<String>()
                        + &format!("shared shared shared {owner} alphaBeta storage_path");
                    if ordinal == 103 {
                        chunk.text = "the and".into();
                    }
                    catalog.chunks.insert(id.clone(), chunk);
                    if ordinal % 11 == 0 {
                        catalog.deleted_chunks.insert(id);
                    }
                }
            }
            for selected in [None, Some("first")] {
                let context = JsonOperationContext::standalone();
                let available = context.available_workspace_bytes();
                let disk = Arc::new(TemporaryDiskBudget::new(if large {
                    16 * 1024 * 1024
                } else {
                    0
                }));
                let include = |id: &str| selected.map_or(true, |selected| id == selected);
                let expected = TextIndex::try_build_matching(
                    &catalog.visible_snapshot(),
                    include,
                    |visible| Ok(visible.chunk.text.clone()),
                )
                .unwrap();
                let mut reads = 0;
                let actual = TextIndex::try_build_matching_in(
                    &catalog,
                    include,
                    |visible| {
                        assert_eq!(disk.used(), 0);
                        assert_eq!(fs::read_dir(&directory.0).unwrap().count(), 0);
                        reads += 1;
                        ReservedString::copy(&visible.chunk.text, &context)
                    },
                    &directory.0,
                    disk.clone(),
                    &context,
                )
                .unwrap();
                let chunks = catalog
                    .visible_snapshot()
                    .live_chunks()
                    .filter(|visible| include(&visible.chunk.collection_id))
                    .count();
                assert_eq!(reads, chunks);
                let matched = catalog
                    .visible_snapshot()
                    .live_chunks()
                    .filter(|visible| {
                        include(&visible.chunk.collection_id) && visible.chunk.text != "the and"
                    })
                    .count();
                assert_eq!(actual.postings["shared"].len(), matched);
                assert_eq!(actual.chunk_lengths["chunk0103-first"], 1);
                assert!(actual.postings["shared"]
                    .iter()
                    .all(|posting| posting.term_frequency == 3));
                assert_same(&actual, &expected, &catalog, &context);
                assert_eq!(context.available_workspace_bytes(), available);
                directory.clean(&disk);
            }
        }
    }

    #[test]
    fn stop_words_and_empty_collections_do_not_create_sort_files() {
        let directory = Directory::new("stop-words");
        let mut catalog = fixture(1);
        for chunk in catalog.chunks.values_mut() {
            chunk.text = "the and".into();
        }
        let bytes = catalog.live_collections().count()
            * std::mem::size_of::<(&str, usize, usize, Option<Arc<str>>)>();
        let context =
            JsonOperationContext::new(Arc::new(WorkLedger::with_limits(bytes, bytes, bytes)));
        let disk = Arc::new(TemporaryDiskBudget::new(0));
        let actual = TextIndex::try_build_matching_in(
            &catalog,
            |_| true,
            |_| Ok("the and"),
            &directory.0,
            disk.clone(),
            &context,
        )
        .unwrap();
        assert_eq!(actual.term_count(), 0);
        assert!(actual.chunk_lengths.values().all(|&len| len == 1));
        assert_eq!(context.available_workspace_bytes(), bytes);
        directory.clean(&disk);
    }

    #[test]
    fn single_payload_visits_leave_minimum_sorter_space_without_disk() {
        let directory = Directory::new("token-headroom");
        let mut catalog = fixture(1);
        catalog.chunks.retain(|id, _| id == "first-0");
        catalog.deleted_chunks.clear();
        catalog.chunks.get_mut("first-0").unwrap().text = "FOO_BAR".into();
        let scratch = 3 * std::mem::size_of::<(Vec<u8>, u64, u64)>()
            + "foo_bar".len()
            + "foo".len()
            + "bar".len()
            + "bar".len();
        let minimum = std::mem::size_of::<(&str, usize, usize, Option<Arc<str>>)>() + scratch;
        for parent in [false, true] {
            for limit in [minimum, minimum + 1] {
                let context = JsonOperationContext::new(Arc::new(WorkLedger::with_limits(
                    if parent { 2 * limit } else { limit },
                    0,
                    if parent { limit } else { 2 * limit },
                )));
                let disk = Arc::new(TemporaryDiskBudget::new(0));
                let actual = TextIndex::try_build_matching_in(
                    &catalog,
                    |id| id == "first",
                    |_| Ok("FOO_BAR"),
                    &directory.0,
                    disk.clone(),
                    &context,
                )
                .unwrap();
                assert_eq!(actual.chunk_lengths["first-0"], 3);
                assert_eq!(actual.term_count(), 3);
                for term in ["foo_bar", "foo", "bar"] {
                    assert_eq!(actual.postings[term][0].term_frequency, 1);
                }
                assert_eq!(context.available_workspace_bytes(), limit);
                directory.clean(&disk);
            }
        }
    }

    #[test]
    fn token_normalization_reclaims_completed_array_without_rereading_or_spilling() {
        let directory = Directory::new("token-pressure");
        let mut catalog = fixture(1);
        catalog
            .chunks
            .retain(|id, _| id == "first-0" || id == "first-1");
        catalog.deleted_chunks.clear();
        catalog.chunks.get_mut("first-0").unwrap().text =
            (0..130).map(|n| format!("token{n} ")).collect();
        let raw = "PROJECT".repeat(1000);
        let normalized = raw.to_ascii_lowercase();
        catalog.chunks.get_mut("first-1").unwrap().text = raw;
        let limit = 2 * normalized.len()
            + std::mem::size_of::<(&str, usize, usize, Option<Arc<str>>)>()
            + 4 * std::mem::size_of::<(Vec<u8>, u64, u64)>();
        for parent in [false, true] {
            let context = JsonOperationContext::new(Arc::new(WorkLedger::with_limits(
                if parent { 2 * limit } else { limit },
                0,
                if parent { limit } else { 2 * limit },
            )));
            let disk = Arc::new(TemporaryDiskBudget::new(0));
            let mut reads = [0; 2];
            let actual = TextIndex::try_build_matching_in(
                &catalog,
                |id| id == "first",
                |visible| {
                    let second = visible.chunk.chunk_id == "first-1";
                    reads[usize::from(second)] += 1;
                    if second {
                        assert!(context.available_workspace_bytes() < normalized.len());
                    }
                    Ok(catalog.chunks[&visible.chunk.chunk_id].text.as_str())
                },
                &directory.0,
                disk.clone(),
                &context,
            )
            .unwrap();
            assert_eq!(reads, [1, 1]);
            assert_eq!(actual.term_count(), 131);
            assert_eq!(actual.chunk_lengths["first-0"], 130);
            assert_eq!(actual.chunk_lengths["first-1"], 1);
            assert_eq!(actual.postings[&normalized][0].term_frequency, 1);
            assert_eq!(context.available_workspace_bytes(), limit);
            directory.clean(&disk);
        }
    }

    #[test]
    fn next_chunk_read_reclaims_only_completed_batch_capacity_and_preserves_original_errors() {
        let directory = Directory::new("read-pressure");
        let mut catalog = fixture(1);
        catalog
            .chunks
            .retain(|id, _| id == "first-0" || id == "first-1");
        catalog.deleted_chunks.clear();
        catalog.chunks.get_mut("first-0").unwrap().text =
            (0..64).map(|n| format!("token{n} ")).collect();
        catalog.chunks.get_mut("first-1").unwrap().text = "shared ".repeat(1000);
        let limit = 7000
            + std::mem::size_of::<(&str, usize, usize, Option<Arc<str>>)>()
            + crate::temporary::test_file_management_bytes(&directory.0)
            + 200;
        for parent in [false, true] {
            for mode in 0..4 {
                let context = JsonOperationContext::new(Arc::new(WorkLedger::with_limits(
                    if parent { limit * 2 } else { limit },
                    limit * 2,
                    if parent { limit } else { limit * 2 },
                )));
                let disk = Arc::new(TemporaryDiskBudget::new(if mode == 3 {
                    0
                } else {
                    1024 * 1024
                }));
                let mut calls = [0; 2];
                let result = TextIndex::try_build_matching_in(
                    &catalog,
                    |id| id == "first",
                    |visible| {
                        let second = visible.chunk.chunk_id == "first-1";
                        calls[usize::from(second)] += 1;
                        if second && mode == 1 {
                            return Err(std::io::Error::other("read failed").into());
                        }
                        if second && mode == 2 {
                            return Err(Error::resource_exhausted(
                                "memory",
                                "operation_scratch",
                                limit as u64,
                                (limit + calls[1]) as u128,
                            ));
                        }
                        ReservedString::copy(&visible.chunk.text, &context)
                    },
                    &directory.0,
                    disk.clone(),
                    &context,
                );
                if mode == 0 || mode == 3 {
                    let actual = result.unwrap();
                    assert_eq!(calls, [1, 2]);
                    assert_eq!(actual.postings["shared"][0].term_frequency, 1000);
                    assert_eq!(actual.chunk_lengths["first-1"], 1000);
                    assert_eq!(actual.term_count(), 65);
                } else {
                    let error = result.unwrap_err();
                    assert_eq!(
                        error.code(),
                        if mode == 1 {
                            ErrorCode::IoError
                        } else {
                            ErrorCode::ResourceExhausted
                        }
                    );
                    if mode == 1 {
                        assert_eq!(error.message(), "read failed");
                    }
                    if mode == 2 {
                        for (key, expected) in [
                            ("resource", "memory".to_owned()),
                            ("operation", "operation_scratch".to_owned()),
                            ("limit_bytes", limit.to_string()),
                            ("requested_bytes", (limit + 2).to_string()),
                        ] {
                            assert_eq!(
                                error.detail().iter().find(|item| item.0 == key).unwrap().1,
                                expected
                            );
                        }
                    }
                    assert_eq!(calls, if mode == 2 { [1, 2] } else { [1, 1] });
                }
                directory.clean(&disk);
                let mut probe = context.workspace_reservation();
                probe.grow_to(limit, limit, "operation_scratch").unwrap();
                assert!(probe
                    .grow_to(limit + 1, limit + 1, "operation_scratch")
                    .is_err());
            }
        }
    }
}
