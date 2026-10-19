use crate::catalog::Catalog;
use crate::error::{Error, ErrorCode, Result};
use crate::hash::Crc32c;
use crate::json::{JsonOperationContext, ReservedVec};
use crate::model::{CollectionInfo, DistanceMetric};

struct CoverageEntry<'a> {
    id: u64,
    collection: &'a CollectionInfo,
    counts: [usize; 3],
}

fn collection_number(id: &str) -> Option<u64> {
    id.strip_prefix("col_")?.parse().ok()
}

pub(super) fn catalog_index_coverage_hash(
    catalog: &Catalog,
    critical_end_offset: u64,
    last_commit_sequence_no: u64,
    context: &JsonOperationContext,
) -> Result<u32> {
    #[cfg(test)]
    super::tests::record_coverage_hash();
    let mut collections = ReservedVec::with_capacity(
        catalog.live_collections().count(),
        context,
        "operation_scratch",
    )?;
    for collection in catalog.live_collections() {
        let id = collection_number(&collection.collection_id)
            .ok_or_else(|| Error::new(ErrorCode::FileFormatError, "invalid collection ID"))?;
        collections.push(CoverageEntry {
            id,
            collection,
            counts: [0; 3],
        });
    }
    let collections = collections.as_mut_slice();
    collections.sort_unstable_by_key(|entry| entry.id);
    for (id, kind) in catalog
        .live_documents()
        .map(|row| (row.collection_id.as_str(), 0))
        .chain(
            catalog
                .live_chunks()
                .map(|row| (row.collection_id.as_str(), 1)),
        )
        .chain(
            catalog
                .live_embeddings()
                .map(|row| (row.collection_id.as_str(), 2)),
        )
    {
        let Some(number) = collection_number(id) else {
            continue;
        };
        let start = collections.partition_point(|entry| entry.id < number);
        // Preserve exact string ownership even for noncanonical IDs sharing a number.
        for entry in collections[start..]
            .iter_mut()
            .take_while(|entry| entry.id == number)
        {
            if entry.collection.collection_id == id {
                entry.counts[kind] = entry.counts[kind].checked_add(1).ok_or_else(|| {
                    Error::new(ErrorCode::FileFormatError, "coverage count overflow")
                })?;
            }
        }
    }
    let mut checksum = Crc32c::default();
    checksum.update(&1u16.to_le_bytes());
    checksum.update(&critical_end_offset.to_le_bytes());
    checksum.update(&last_commit_sequence_no.to_le_bytes());
    hash_count(&mut checksum, collections.len());
    for entry in collections {
        let collection = entry.collection;
        checksum.update(&entry.id.to_le_bytes());
        hash_string(&mut checksum, &collection.name)?;
        let dim = u32::try_from(collection.embedding_dim).map_err(|_| {
            Error::new(
                ErrorCode::InvalidEmbeddingDimension,
                "dimension exceeds u32",
            )
        })?;
        checksum.update(&dim.to_le_bytes());
        checksum.update(&[distance_metric_id(collection.distance_metric)]);
        hash_optional_string(&mut checksum, collection.embedding_model.as_deref())?;
        checksum.update(&[u8::from(collection.vectors_normalized)]);
        for count in entry.counts {
            hash_count(&mut checksum, count);
        }
        checksum.update(&1u16.to_le_bytes());
        checksum.update(&2u16.to_le_bytes());
        checksum.update(&[1, 1]);
    }
    Ok(checksum.finish())
}

fn hash_string(checksum: &mut Crc32c, value: &str) -> Result<()> {
    let len = u32::try_from(value.len())
        .map_err(|_| Error::new(ErrorCode::FileFormatError, "coverage string exceeds u32"))?;
    checksum.update(&len.to_le_bytes());
    checksum.update(value.as_bytes());
    Ok(())
}

fn hash_optional_string(checksum: &mut Crc32c, value: Option<&str>) -> Result<()> {
    match value {
        Some(value) => {
            checksum.update(&[1]);
            hash_string(checksum, value)?;
        }
        None => checksum.update(&[0]),
    }
    Ok(())
}

fn hash_count(checksum: &mut Crc32c, count: usize) {
    checksum.update(&(count as u64).to_le_bytes());
}

fn distance_metric_id(metric: DistanceMetric) -> u8 {
    match metric {
        DistanceMetric::Cosine => 1,
        DistanceMetric::DotProduct => 2,
        DistanceMetric::Euclidean => 3,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json::WorkLedger;
    use std::sync::Arc;

    fn fixture(seed: usize) -> Catalog {
        let mut catalog = Catalog::default();
        for id in 1..=seed % 17 + 1 {
            let collection = CollectionInfo {
                collection_id: format!("col_{id:016}"),
                name: "\u{4e2d}".repeat(4096 + seed),
                embedding_dim: 2,
                distance_metric: DistanceMetric::Cosine,
                embedding_model: Some("model".repeat(8192 + seed)),
                tokenizer: Default::default(),
                vectors_normalized: false,
                vector_index: Default::default(),
                created_at: String::new(),
            };
            catalog
                .collections
                .insert(collection.collection_id.clone(), collection);
        }
        catalog
    }

    fn assert_available(context: &JsonOperationContext, bytes: usize) {
        let mut work = context.workspace_reservation();
        work.grow_to(bytes, bytes, "operation_scratch").unwrap();
        assert!(work
            .grow_to(bytes + 1, bytes + 1, "operation_scratch")
            .is_err());
    }

    #[test]
    fn coverage_rows_obey_independent_working_and_parent_boundaries_and_release() {
        for seed in 0..100 {
            let catalog = fixture(seed);
            let required = catalog.live_collections().count()
                * (8 + std::mem::size_of::<&CollectionInfo>() + 3 * std::mem::size_of::<usize>());
            let expected = catalog_index_coverage_hash(
                &catalog,
                58,
                seed as u64,
                &JsonOperationContext::standalone(),
            )
            .unwrap();
            for parent in [false, true] {
                for limit in [required - 1, required, required + 1] {
                    let total = if parent { limit + 8 } else { 2 * limit + 8 };
                    let working = if parent { 2 * limit + 8 } else { limit };
                    let context = JsonOperationContext::new(Arc::new(WorkLedger::with_limits(
                        working, total, total,
                    )));
                    let cache = context.cache_reservation(8, 8).unwrap();
                    let result = catalog_index_coverage_hash(&catalog, 58, seed as u64, &context);
                    if limit < required {
                        let error = result.unwrap_err();
                        assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                        for (key, value) in [
                            ("resource", "memory".to_owned()),
                            ("operation", "operation_scratch".to_owned()),
                            (
                                "limit_bytes",
                                if parent { total } else { working }.to_string(),
                            ),
                            (
                                "requested_bytes",
                                if parent { 8 + required } else { required }.to_string(),
                            ),
                        ] {
                            assert!(error.detail().contains(&(key.into(), value)), "{error:?}");
                        }
                    } else {
                        assert_eq!(result.unwrap(), expected);
                    }
                    assert_available(&context, limit);
                    drop(cache);
                }
            }
            let context = JsonOperationContext::new(Arc::new(WorkLedger::with_limits(
                required, required, required,
            )));
            // Initialize the caller's ledger locks outside the array measurement.
            assert_eq!(context.available_workspace_bytes(), required);
            let (actual, allocated) = crate::test_alloc::allocation_bytes(|| {
                catalog_index_coverage_hash(&catalog, 58, seed as u64, &context).unwrap()
            });
            assert_eq!(actual, expected);
            assert_eq!(allocated, required as u128, "seed {seed}");
            assert_available(&context, required);
        }
        let context = JsonOperationContext::new(Arc::new(WorkLedger::with_limits(0, 0, 0)));
        let catalog = Catalog::default();
        assert_eq!(context.available_workspace_bytes(), 0);
        let (_, allocated) = crate::test_alloc::allocation_bytes(|| {
            catalog_index_coverage_hash(&catalog, 58, 0, &context).unwrap()
        });
        assert_eq!(allocated, 0);
        assert_available(&context, 0);
    }

    #[test]
    fn coverage_rows_release_credit_on_id_or_dimension_validation_failure() {
        for invalid_id in [false, true] {
            let mut catalog = fixture(0);
            let (key, mut collection) = catalog.collections.pop_first().unwrap();
            if invalid_id {
                collection.collection_id = "doc_0000000000000001".into();
                catalog
                    .collections
                    .insert(collection.collection_id.clone(), collection);
            } else {
                collection.embedding_dim = u32::MAX as usize + 1;
                catalog.collections.insert(key, collection);
            }
            let required =
                8 + std::mem::size_of::<&CollectionInfo>() + 3 * std::mem::size_of::<usize>();
            let context = JsonOperationContext::new(Arc::new(WorkLedger::with_limits(
                required, required, required,
            )));
            let error = catalog_index_coverage_hash(&catalog, 58, 0, &context).unwrap_err();
            assert_eq!(
                error.code(),
                if invalid_id {
                    ErrorCode::FileFormatError
                } else {
                    ErrorCode::InvalidEmbeddingDimension
                }
            );
            assert_available(&context, required);
        }
    }
}
