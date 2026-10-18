use super::*;
use crate::index::text::TextCacheBuild;
use crate::json::WorkLedger;
use crate::storage::TextSegmentEntry;

fn context(bytes: usize) -> JsonOperationContext {
    JsonOperationContext::new(Arc::new(WorkLedger::with_limits(bytes, 0, bytes)))
}

fn complete(result: Result<TextCacheBuild>) -> TextIndex {
    match result.unwrap() {
        TextCacheBuild::Complete(index) => index,
        other => panic!("expected complete cache, got {other:?}"),
    }
}

fn fixture(context: &JsonOperationContext, limit: usize) -> Result<TextCacheBuild> {
    TextIndex::try_cache_from_visitors(["first", "second"], context, limit, |id, visit| {
        let offset = if id == "first" { 0 } else { 2 };
        visit(TextSegmentEntry::Header { chunks: 2 })?;
        for (chunk, length) in [(offset + 1, 2), (offset + 2, 1)] {
            visit(TextSegmentEntry::Chunk { id: chunk, length })?;
        }
        visit(TextSegmentEntry::Term {
            text: "alpha",
            document_frequency: 2,
        })?;
        for chunk in [offset + 1, offset + 2] {
            visit(TextSegmentEntry::Posting {
                term: "alpha",
                chunk,
                frequency: 1,
            })?;
        }
        visit(TextSegmentEntry::Term {
            text: "beta",
            document_frequency: 1,
        })?;
        visit(TextSegmentEntry::Posting {
            term: "beta",
            chunk: offset + 1,
            frequency: 1,
        })?;
        visit(TextSegmentEntry::Term {
            text: "void",
            document_frequency: 0,
        })?;
        Ok(true)
    })
}

fn nodes<K, V>(count: usize) -> usize {
    count.div_ceil(5) * (11 * std::mem::size_of::<(K, V)>() + 16 * std::mem::size_of::<usize>())
}

fn measured_payload(index: &TextIndex) -> usize {
    let keys = index.postings.keys().map(String::capacity).sum::<usize>()
        + index
            .chunk_lengths
            .keys()
            .map(String::capacity)
            .sum::<usize>()
        + index
            .total_chunks_by_collection
            .keys()
            .map(String::capacity)
            .sum::<usize>()
        + index
            .avg_len_by_collection
            .keys()
            .map(String::capacity)
            .sum::<usize>();
    let postings = index
        .postings
        .values()
        .map(|entries| {
            entries.capacity() * std::mem::size_of::<Posting>()
                + entries
                    .iter()
                    .map(|posting| posting.chunk_id.capacity())
                    .sum::<usize>()
        })
        .sum::<usize>();
    let arcs = index
        .total_chunks_by_collection
        .values()
        .filter_map(|counts| counts.owner.as_ref())
        .map(|id| {
            let word = std::mem::size_of::<usize>();
            (2 * word + id.len()).div_ceil(word) * word
        })
        .sum::<usize>();
    keys + postings + arcs
}

#[test]
fn four_trees_and_shared_collection_allocations_have_one_persistent_owner() {
    let ledger = Arc::new(WorkLedger::with_limits(1_000_000, 0, 1_000_000));
    let operation = JsonOperationContext::new(ledger.clone());
    let index = complete(fixture(&operation, usize::MAX));
    assert_eq!(index.postings.len(), 3);
    assert_eq!(index.chunk_lengths.len(), 4);
    assert_eq!(
        index.collection_counts(),
        BTreeMap::from([("first".into(), 2), ("second".into(), 2)])
    );
    assert!(index
        .avg_len_by_collection
        .values()
        .all(|&average| average == 1.5));
    for counts in index.total_chunks_by_collection.values() {
        assert_eq!(Arc::strong_count(counts.owner.as_ref().unwrap()), 4);
    }
    for posting in index.postings.values().flatten() {
        assert!(Arc::ptr_eq(
            &posting.collection_id,
            index.total_chunks_by_collection[posting.collection_id.as_ref()]
                .owner
                .as_ref()
                .unwrap()
        ));
    }
    let expected = nodes::<String, Vec<Posting>>(3)
        + nodes::<String, usize>(4)
        + nodes::<String, CollectionCounts>(2)
        + nodes::<String, f64>(2)
        + measured_payload(&index);
    assert_eq!(index.reserved_bytes(), expected);
    drop(operation);
    let next = JsonOperationContext::new(ledger);
    assert_eq!(next.available_workspace_bytes(), 1_000_000 - expected);
    drop(index);
    assert_eq!(next.available_workspace_bytes(), 1_000_000);
}

#[test]
fn tree_growth_admits_the_fifth_to_sixth_node_before_allocation_on_both_axes() {
    for tree in 0..4 {
        let old = match tree {
            TERMS => nodes::<String, Vec<Posting>>(5),
            LENGTHS => nodes::<String, usize>(5),
            COUNTS => nodes::<String, CollectionCounts>(5),
            AVERAGES => nodes::<String, f64>(5),
            _ => unreachable!(),
        };
        for parent in [false, true] {
            for allowance in [2 * old + 17 - 1, 2 * old + 17, 2 * old + 17 + 1] {
                let total = if parent { allowance } else { 1_000_000 };
                let limit = if parent { usize::MAX } else { allowance };
                let operation = context(total);
                let mut owner = TextCacheOwner::new(&operation);
                let admit = |owner: &mut TextCacheOwner, count, key| match tree {
                    TERMS => owner.admit_node::<String, Vec<Posting>>(tree, count, key, limit),
                    LENGTHS => owner.admit_node::<String, usize>(tree, count, key, limit),
                    COUNTS => owner.admit_node::<String, CollectionCounts>(tree, count, key, limit),
                    AVERAGES => owner.admit_node::<String, f64>(tree, count, key, limit),
                    _ => unreachable!(),
                };
                assert!(admit(&mut owner, 5, 0));
                assert_eq!(owner.reserved_bytes(), old);
                assert_eq!(admit(&mut owner, 6, 17), allowance >= 2 * old + 17);
                assert_eq!(
                    owner.reserved_bytes(),
                    if allowance < 2 * old + 17 {
                        old
                    } else {
                        2 * old + 17
                    }
                );
                drop(owner);
                assert_eq!(operation.available_workspace_bytes(), total);
            }
        }
    }
}

#[test]
fn posting_growth_keeps_old_and_full_new_array_credited_and_refunds_native_failure() {
    let size = std::mem::size_of::<Posting>();
    let base = nodes::<String, Vec<Posting>>(1)
        + 5
        + size
        + 7
        + (2 * std::mem::size_of::<usize>() + 10).next_multiple_of(std::mem::size_of::<usize>());
    let peak = base + 2 * size;
    for parent in [false, true] {
        for allowance in [peak - 1, peak, peak + 1] {
            let total = if parent { allowance } else { 1_000_000 };
            let limit = if parent { usize::MAX } else { allowance };
            let operation = context(total);
            let mut index = TextIndex {
                cache: Some(TextCacheOwner::new(&operation)),
                ..Default::default()
            };
            assert!(index.cache_term("alpha", 1, limit).unwrap());
            let collection = index.cache_collection_owner("collection", limit).unwrap();
            assert!(index
                .cache_posting("alpha", "chunk-1", 7, &collection, limit)
                .unwrap());
            let pointer = index.postings["alpha"].as_ptr();
            let (result, allocations) =
                crate::test_alloc::allocations(|| index.cache_term("alpha", 1, limit));
            assert_eq!(result.unwrap(), allowance >= peak);
            assert_eq!(allocations, usize::from(allowance >= peak));
            if allowance < peak {
                assert_eq!(index.postings["alpha"].as_ptr(), pointer);
                assert_eq!(index.reserved_bytes(), base);
            } else {
                assert_eq!(index.postings["alpha"].capacity(), 2);
                assert_eq!(index.reserved_bytes(), base + size);
            }
            assert_eq!(index.postings["alpha"][0].chunk_id, "chunk-1");
            assert_eq!(index.postings["alpha"][0].term_frequency, 7);
            assert!(Arc::ptr_eq(
                &index.postings["alpha"][0].collection_id,
                &collection
            ));
            drop(collection);
            drop(index);
            assert_eq!(operation.available_workspace_bytes(), total);
        }
    }
    let operation = context(1_000_000);
    let mut index = TextIndex {
        cache: Some(TextCacheOwner::new(&operation)),
        ..Default::default()
    };
    assert!(index.cache_term("alpha", 1, usize::MAX).unwrap());
    let collection = index
        .cache_collection_owner("collection", usize::MAX)
        .unwrap();
    assert!(index
        .cache_posting("alpha", "chunk-1", 7, &collection, usize::MAX)
        .unwrap());
    let pointer = index.postings["alpha"].as_ptr();
    let error =
        crate::test_alloc::fail_nth_allocation_once(1, || index.cache_term("alpha", 1, usize::MAX))
            .unwrap_err();
    assert_eq!(error.code(), ErrorCode::IoError);
    assert_eq!(index.postings["alpha"].as_ptr(), pointer);
    assert_eq!(index.postings["alpha"].len(), 1);
    assert_eq!(index.postings["alpha"][0].chunk_id, "chunk-1");
    assert_eq!(index.postings["alpha"][0].term_frequency, 7);
    assert!(Arc::ptr_eq(
        &index.postings["alpha"][0].collection_id,
        &collection
    ));
    assert_eq!(index.reserved_bytes(), base);
    assert!(index.cache_term("alpha", 1, usize::MAX).unwrap());
    drop(collection);
    drop(index);
    assert_eq!(operation.available_workspace_bytes(), 1_000_000);
}

fn growth_fixture(operation: &JsonOperationContext) -> (TextIndex, Arc<str>) {
    let mut index = TextIndex {
        cache: Some(TextCacheOwner::new(operation)),
        ..Default::default()
    };
    assert!(index.cache_term("alpha", 4, usize::MAX).unwrap());
    let collection = index
        .cache_collection_owner("collection", usize::MAX)
        .unwrap();
    for (n, id) in ["chunk-1", "chunk-2", "chunk-3", "chunk-4"]
        .into_iter()
        .enumerate()
    {
        assert!(index
            .cache_posting("alpha", id, n + 1, &collection, usize::MAX)
            .unwrap());
    }
    (index, collection)
}

fn assert_growth_payload(index: &TextIndex, collection: &Arc<str>) {
    assert_eq!(index.postings["alpha"].len(), 4);
    for (n, posting) in index.postings["alpha"].iter().enumerate() {
        assert_eq!(posting.chunk_id, format!("chunk-{}", n + 1));
        assert_eq!(posting.term_frequency, n + 1);
        assert!(Arc::ptr_eq(&posting.collection_id, collection));
    }
}

#[test]
fn posting_growth_prefers_amortization_but_falls_back_to_minimum_on_both_shared_axes() {
    let size = std::mem::size_of::<Posting>();
    let base = nodes::<String, Vec<Posting>>(1)
        + 5
        + 4 * size
        + 4 * 7
        + (2 * std::mem::size_of::<usize>() + 10).next_multiple_of(std::mem::size_of::<usize>());
    for parent in [false, true] {
        let other = 37 + usize::from(parent) * 19;
        let minimum = base + other + 5 * size;
        let preferred = base + other + 8 * size;
        for allowance in [
            minimum - 1,
            minimum,
            minimum + 1,
            preferred - 1,
            preferred,
            preferred + 1,
        ] {
            let total = if parent { allowance } else { 1_000_000 };
            let limit = if parent { usize::MAX } else { allowance };
            let ledger = Arc::new(WorkLedger::with_limits(total, 0, total));
            let operation = JsonOperationContext::new(ledger.clone());
            let other_operation = JsonOperationContext::new(ledger);
            let peer = other_operation.cache_reservation(37, usize::MAX).unwrap();
            let mut scratch = other_operation.workspace_reservation();
            scratch.grow_to(19, 19, "operation_scratch").unwrap();
            let (mut index, collection) = growth_fixture(&operation);
            assert_eq!(index.reserved_bytes(), base);
            let old = index.postings["alpha"].as_ptr();
            let (result, allocations) =
                crate::test_alloc::allocations(|| index.cache_term("alpha", 1, limit));
            let admitted = allowance >= minimum;
            assert_eq!(result.unwrap(), admitted);
            assert_eq!(allocations, usize::from(admitted));
            let capacity = if allowance >= preferred {
                8
            } else if admitted {
                5
            } else {
                4
            };
            assert_eq!(index.postings["alpha"].capacity(), capacity);
            if !admitted {
                assert_eq!(index.postings["alpha"].as_ptr(), old);
            }
            assert_growth_payload(&index, &collection);
            assert_eq!(index.reserved_bytes(), base + (capacity - 4) * size);
            assert_eq!(
                operation.available_workspace_bytes(),
                total - index.reserved_bytes() - 37 - 19
            );
            drop(collection);
            drop(index);
            drop((peer, scratch));
            assert_eq!(operation.available_workspace_bytes(), total);
        }
    }
}

#[test]
fn native_failure_of_preferred_or_minimum_array_does_not_retry_or_change_old_cache() {
    for minimum_only in [false, true] {
        let operation = context(1_000_000);
        let (mut index, collection) = growth_fixture(&operation);
        let base = index.reserved_bytes();
        let old = index.postings["alpha"].as_ptr();
        let limit = if minimum_only {
            base + 5 * std::mem::size_of::<Posting>()
        } else {
            usize::MAX
        };
        let error =
            crate::test_alloc::fail_nth_allocation_once(1, || index.cache_term("alpha", 1, limit))
                .unwrap_err();
        assert_eq!(error.code(), ErrorCode::IoError);
        assert_eq!(index.postings["alpha"].as_ptr(), old);
        assert_eq!(index.postings["alpha"].capacity(), 4);
        assert_eq!(index.reserved_bytes(), base);
        assert_growth_payload(&index, &collection);
        assert!(index.cache_term("alpha", 1, limit).unwrap());
        assert_eq!(
            index.postings["alpha"].capacity(),
            if minimum_only { 5 } else { 8 }
        );
        drop(collection);
        drop(index);
        assert_eq!(operation.available_workspace_bytes(), 1_000_000);
    }
}

#[test]
fn small_posting_appends_have_logarithmic_array_allocations_and_account_for_spare_slots() {
    let operation = context(1_000_000);
    let mut index = TextIndex {
        cache: Some(TextCacheOwner::new(&operation)),
        ..Default::default()
    };
    assert!(index.cache_term("alpha", 0, usize::MAX).unwrap());
    let collection = index
        .cache_collection_owner("collection", usize::MAX)
        .unwrap();
    let mut growths = 0;
    for n in 0..128 {
        let (result, allocations) =
            crate::test_alloc::allocations(|| index.cache_term("alpha", 1, usize::MAX));
        assert!(result.unwrap());
        growths += allocations;
        assert!(index
            .cache_posting(
                "alpha",
                &format!("chunk-{n:03}"),
                n + 1,
                &collection,
                usize::MAX
            )
            .unwrap());
    }
    assert_eq!(growths, 8);
    assert_eq!(index.postings["alpha"].capacity(), 128);
    assert_eq!(
        index.reserved_bytes(),
        nodes::<String, Vec<Posting>>(1)
            + 5
            + 128 * (std::mem::size_of::<Posting>() + 9)
            + arc_bytes("collection") as usize
    );
    for (n, posting) in index.postings["alpha"].iter().enumerate() {
        assert_eq!(posting.chunk_id, format!("chunk-{n:03}"));
        assert_eq!(posting.term_frequency, n + 1);
        assert!(Arc::ptr_eq(&posting.collection_id, &collection));
    }
    drop(collection);
    drop(index);
    assert_eq!(operation.available_workspace_bytes(), 1_000_000);
}

#[test]
fn spare_slots_and_large_batches_use_actual_length_and_overflow_keeps_cache_unchanged() {
    let operation = context(1_000_000);
    let (mut index, collection) = growth_fixture(&operation);
    assert!(index.cache_term("alpha", 1, usize::MAX).unwrap());
    assert_eq!(index.postings["alpha"].capacity(), 8);
    let before = index.reserved_bytes();
    let (result, allocations) = crate::test_alloc::allocations(|| {
        assert!(index.cache_term("alpha", 0, 0).unwrap());
        index.cache_term("alpha", 4, 0)
    });
    assert!(result.unwrap());
    assert_eq!(allocations, 0);
    assert_eq!(index.reserved_bytes(), before);
    assert!(index.cache_term("alpha", 5, usize::MAX).unwrap());
    assert_eq!(index.postings["alpha"].capacity(), 16);
    assert!(index.cache_term("alpha", 29, usize::MAX).unwrap());
    assert_eq!(index.postings["alpha"].capacity(), 33);
    let before = index.reserved_bytes();
    let old = index.postings["alpha"].as_ptr();
    assert_eq!(
        index
            .cache_term("alpha", usize::MAX, usize::MAX)
            .unwrap_err()
            .code(),
        ErrorCode::IndexCorruption
    );
    assert_eq!(index.postings["alpha"].as_ptr(), old);
    assert_eq!(index.reserved_bytes(), before);
    assert_growth_payload(&index, &collection);
    drop(collection);
    drop(index);
    assert_eq!(operation.available_workspace_bytes(), 1_000_000);
}

#[test]
fn shared_term_growth_in_full_visitors_and_incremental_merge_preserves_the_visitor_oracle() {
    let operation = context(1_000_000);
    let collections = [
        "col_9", "col_8", "col_7", "col_6", "col_5", "col_4", "col_3", "col_2", "col_1",
    ];
    let visit = |id: &str, visitor: &mut crate::storage::TextSegmentVisitor<'_>| {
        let n: usize = id[4..].parse().unwrap();
        visitor(TextSegmentEntry::Header { chunks: 1 })?;
        visitor(TextSegmentEntry::Chunk {
            id: n as u64,
            length: n,
        })?;
        visitor(TextSegmentEntry::Term {
            text: "alpha",
            document_frequency: 1,
        })?;
        visitor(TextSegmentEntry::Posting {
            term: "alpha",
            chunk: n as u64,
            frequency: n,
        })?;
        visitor(TextSegmentEntry::Term {
            text: "void",
            document_frequency: 0,
        })?;
        Ok(true)
    };
    let expected = TextIndex::try_build_from_visitors(collections, visit)
        .unwrap()
        .unwrap();
    let full = complete(TextIndex::try_cache_from_visitors(
        collections,
        &operation,
        usize::MAX,
        visit,
    ));
    let mut merged = complete(TextIndex::try_cache_from_visitors(
        collections[..5].iter().copied(),
        &operation,
        usize::MAX,
        visit,
    ));
    let replacement = complete(TextIndex::try_cache_from_visitors(
        collections[5..].iter().copied(),
        &operation,
        usize::MAX,
        visit,
    ));
    assert!(merged
        .replace_cached_collections(&BTreeSet::new(), [], replacement, usize::MAX)
        .unwrap());
    for actual in [&full, &merged] {
        assert_eq!(actual.postings["alpha"].capacity(), 16);
        assert_eq!(
            actual.postings["alpha"].len(),
            expected.postings["alpha"].len()
        );
        assert!(actual.postings.keys().eq(expected.postings.keys()));
        assert!(actual.postings["void"].is_empty());
        for (actual, expected) in actual.postings["alpha"]
            .iter()
            .zip(&expected.postings["alpha"])
        {
            assert_eq!(actual.collection_id, expected.collection_id);
            assert_eq!(actual.chunk_id, expected.chunk_id);
            assert_eq!(actual.term_frequency, expected.term_frequency);
        }
        assert_eq!(actual.chunk_lengths, expected.chunk_lengths);
        assert_eq!(actual.collection_counts(), expected.collection_counts());
        assert_eq!(actual.avg_len_by_collection, expected.avg_len_by_collection);
        for posting in &actual.postings["alpha"] {
            assert!(Arc::ptr_eq(
                &posting.collection_id,
                actual.total_chunks_by_collection[posting.collection_id.as_ref()]
                    .owner
                    .as_ref()
                    .unwrap()
            ));
        }
    }
    drop((full, merged));
    assert_eq!(operation.available_workspace_bytes(), 1_000_000);
}

#[test]
fn shared_term_merge_keeps_source_and_target_arrays_credited_through_both_budget_fallbacks() {
    let build = |operation: &JsonOperationContext, ids: &[&str]| {
        complete(TextIndex::try_cache_from_visitors(
            ids.iter().copied(),
            operation,
            usize::MAX,
            |id, visit| {
                let n: usize = id[4..].parse().unwrap();
                visit(TextSegmentEntry::Header { chunks: 1 })?;
                visit(TextSegmentEntry::Chunk {
                    id: n as u64,
                    length: n,
                })?;
                visit(TextSegmentEntry::Term {
                    text: "alpha",
                    document_frequency: 1,
                })?;
                visit(TextSegmentEntry::Posting {
                    term: "alpha",
                    chunk: n as u64,
                    frequency: n,
                })?;
                Ok(true)
            },
        ))
    };
    let ids = ["col_1", "col_2", "col_3", "col_4", "col_5"];
    let operation = context(1_000_000);
    let old = build(&operation, &ids[..4]);
    let replacement = build(&operation, &ids[4..]);
    let base = old.reserved_bytes() + replacement.reserved_bytes();
    let trees = nodes::<String, Vec<Posting>>(1)
        + nodes::<String, usize>(1)
        + nodes::<String, CollectionCounts>(1)
        + nodes::<String, f64>(1);
    let size = std::mem::size_of::<Posting>();
    let per_collection = 40 + 10 + arc_bytes("col_1") as usize;
    assert_eq!(base, 2 * trees + 10 + 5 * size + 5 * per_collection);
    drop((old, replacement));
    assert_eq!(operation.available_workspace_bytes(), 1_000_000);
    let minimum = base + 5 * size;
    let preferred = base + 8 * size;
    for parent in [false, true] {
        for allowance in [
            minimum - 1,
            minimum,
            minimum + 1,
            preferred - 1,
            preferred,
            preferred + 1,
        ] {
            let total = if parent { allowance } else { 1_000_000 };
            let limit = if parent { usize::MAX } else { allowance };
            let operation = context(total);
            let mut target = build(&operation, &ids[..4]);
            let replacement = build(&operation, &ids[4..]);
            assert_eq!(target.reserved_bytes() + replacement.reserved_bytes(), base);
            assert_eq!(
                target
                    .replace_cached_collections(&BTreeSet::new(), [], replacement, limit)
                    .unwrap(),
                allowance >= minimum
            );
            if allowance < minimum {
                assert!(!target.has_resident_cache());
                assert_eq!(target.reserved_bytes(), 0);
            } else {
                assert_eq!(
                    target.postings["alpha"].capacity(),
                    if allowance >= preferred { 8 } else { 5 }
                );
                assert_eq!(target.postings["alpha"].len(), 5);
                assert_eq!(target.chunk_lengths.len(), 5);
                assert_eq!(target.total_chunks_by_collection.len(), 5);
                let resident =
                    trees + 5 + target.postings["alpha"].capacity() * size + 5 * per_collection;
                assert_eq!(target.reserved_bytes(), resident);
                assert_eq!(operation.available_workspace_bytes(), total - resident);
                for (n, posting) in target.postings["alpha"].iter().enumerate() {
                    assert_eq!(posting.collection_id.as_ref(), ids[n]);
                    assert_eq!(posting.chunk_id, format!("chk_{:016}", n + 1));
                    assert_eq!(posting.term_frequency, n + 1);
                    assert!(Arc::ptr_eq(
                        &posting.collection_id,
                        target.total_chunks_by_collection[ids[n]]
                            .owner
                            .as_ref()
                            .unwrap()
                    ));
                }
            }
            drop(target);
            assert_eq!(operation.available_workspace_bytes(), total);
        }
    }
}

#[test]
fn empty_stopword_and_zero_frequency_sources_have_distinct_cache_shapes() {
    for kind in 0..3 {
        let operation = context(1_000_000);
        let index = complete(TextIndex::try_cache_from_visitors(
            ["first"],
            &operation,
            usize::MAX,
            |_, visit| {
                visit(TextSegmentEntry::Header {
                    chunks: usize::from(kind != 0),
                })?;
                if kind != 0 {
                    visit(TextSegmentEntry::Chunk { id: 1, length: 1 })?;
                }
                if kind == 2 {
                    visit(TextSegmentEntry::Term {
                        text: "void",
                        document_frequency: 0,
                    })?;
                }
                Ok(true)
            },
        ));
        assert!(index.has_resident_cache());
        assert_eq!(index.chunk_lengths.len(), usize::from(kind != 0));
        assert_eq!(index.postings.len(), usize::from(kind == 2));
        assert!(index.postings.values().all(Vec::is_empty));
        assert!(index
            .total_chunks_by_collection
            .values()
            .all(|counts| counts.owner.is_none()));
        if kind == 0 {
            assert_eq!(index.reserved_bytes(), 0);
        }
        drop(index);
        assert_eq!(operation.available_workspace_bytes(), 1_000_000);
    }
}

#[test]
fn late_cache_miss_still_validates_all_sources_and_native_errors_refund_candidate() {
    let operation = context(1_000_000);
    let first = complete(TextIndex::try_cache_from_visitors(
        ["first"],
        &operation,
        usize::MAX,
        |_, visit| {
            visit(TextSegmentEntry::Header { chunks: 1 })?;
            visit(TextSegmentEntry::Chunk { id: 1, length: 1 })?;
            Ok(true)
        },
    ));
    let limit = nodes::<String, usize>(1)
        + nodes::<String, CollectionCounts>(1)
        + nodes::<String, f64>(1)
        + 20
        + 2 * "first".len();
    assert_eq!(first.reserved_bytes(), limit);
    drop(first);
    for invalid in [false, true] {
        let mut visits = Vec::new();
        let result = TextIndex::try_cache_from_visitors(
            ["first", "second", "third"],
            &operation,
            limit,
            |id, visit| {
                visits.push(id.to_owned());
                visit(TextSegmentEntry::Header { chunks: 1 })?;
                visit(TextSegmentEntry::Chunk {
                    id: visits.len() as u64,
                    length: 1,
                })?;
                Ok(!(invalid && id == "third"))
            },
        )
        .unwrap();
        assert!(matches!(
            result,
            TextCacheBuild::Miss | TextCacheBuild::Invalid
        ));
        assert_eq!(matches!(result, TextCacheBuild::Invalid), invalid);
        assert_eq!(visits, ["first", "second", "third"]);
        assert_eq!(operation.available_workspace_bytes(), 1_000_000);
    }
    let error = crate::test_alloc::fail_byte_allocation_once("chk_0000000000000001".len(), || {
        fixture(&operation, usize::MAX)
    })
    .unwrap_err();
    assert_eq!(error.code(), ErrorCode::IoError);
    assert_eq!(operation.available_workspace_bytes(), 1_000_000);
}

#[test]
fn partially_transferred_merge_miss_drops_both_candidates_and_all_payload_credit() {
    let operation = context(1_000_000);
    let build = |id, terms: &[&str], offset| {
        complete(TextIndex::try_cache_from_visitors(
            [id],
            &operation,
            usize::MAX,
            |_, visit| {
                visit(TextSegmentEntry::Header {
                    chunks: terms.len(),
                })?;
                for n in 0..terms.len() {
                    visit(TextSegmentEntry::Chunk {
                        id: offset + n as u64,
                        length: 1,
                    })?;
                }
                for (n, term) in terms.iter().enumerate() {
                    visit(TextSegmentEntry::Term {
                        text: term,
                        document_frequency: 1,
                    })?;
                    visit(TextSegmentEntry::Posting {
                        term,
                        chunk: offset + n as u64,
                        frequency: 9,
                    })?;
                }
                Ok(true)
            },
        ))
    };
    let mut target = build("old", &["a", "b", "c", "d"], 1);
    let replacement = build("new", &["e", "f"], 5);
    let limit = target.reserved_bytes() + replacement.reserved_bytes();
    // The fifth key needs no new node; the sixth needs another node while
    // the first moved posting still shares the source-owned collection Arc.
    assert!(!target
        .replace_cached_collections(&BTreeSet::new(), [], replacement, limit)
        .unwrap());
    assert!(!target.has_resident_cache());
    assert!(target.postings.is_empty());
    assert!(target.chunk_lengths.is_empty());
    assert!(target.total_chunks_by_collection.is_empty());
    assert!(target.avg_len_by_collection.is_empty());
    assert_eq!(target.reserved_bytes(), 0);
    assert_eq!(operation.available_workspace_bytes(), 1_000_000);
}
