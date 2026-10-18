use super::{Bm25Hit, MatchedTerm, RankedChunk};
use crate::error::Result;
use crate::json::{
    reserve_heap_slot, try_shrink_heap, JsonOperationContext, ReservedString, ReservedVec,
    WorkspaceReservation,
};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

impl Bm25Hit {
    fn rank(&self) -> RankedChunk<'_> {
        RankedChunk {
            chunk_id: &self.chunk_id,
            score: self.score,
        }
    }

    fn payload_bytes(&self) -> usize {
        self.chunk_id.capacity()
            + self.matched_terms.capacity() * std::mem::size_of::<MatchedTerm>()
            + self
                .matched_terms
                .iter()
                .map(|term| term.term.capacity())
                .sum::<usize>()
    }
}

impl MatchedTerm {
    fn borrowed(&self) -> MatchedTerm<&str> {
        MatchedTerm {
            term: &self.term,
            document_frequency: self.document_frequency,
            chunk_length: self.chunk_length,
            avg_chunk_length: self.avg_chunk_length,
            frequency: self.frequency,
            bm25_contribution: self.bm25_contribution,
        }
    }
}

// The maximum is the worst retained candidate, including the original signed-zero ties.
impl PartialEq for Bm25Hit {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Bm25Hit {}
impl PartialOrd for Bm25Hit {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Bm25Hit {
    fn cmp(&self, other: &Self) -> Ordering {
        other.rank().cmp(&self.rank())
    }
}

struct OwnedHit {
    hit: Bm25Hit,
    work: WorkspaceReservation,
}

impl OwnedHit {
    fn copy<'a>(
        id: &str,
        score: f64,
        terms: impl Iterator<Item = MatchedTerm<&'a str>> + Clone,
        context: &JsonOperationContext,
    ) -> Result<Self> {
        let mut owned = Self {
            hit: Bm25Hit {
                score,
                ..Bm25Hit::default()
            },
            work: context.workspace_reservation(),
        };
        let (terms_array, work) =
            ReservedVec::with_capacity(terms.clone().count(), context, "operation_scratch")?
                .into_parts();
        owned.hit.matched_terms = terms_array;
        owned.work.merge(work);
        let (id, work) = ReservedString::copy(id, context)?.into_parts();
        owned.hit.chunk_id = id;
        owned.work.merge(work);
        for term in terms {
            let (text, work) = ReservedString::copy(term.term, context)?.into_parts();
            assert!(owned.hit.matched_terms.len() < owned.hit.matched_terms.capacity());
            owned.hit.matched_terms.push(MatchedTerm {
                term: text,
                document_frequency: term.document_frequency,
                chunk_length: term.chunk_length,
                avg_chunk_length: term.avg_chunk_length,
                frequency: term.frequency,
                bm25_contribution: term.bm25_contribution,
            });
            owned.work.merge(work);
        }
        Ok(owned)
    }
}

#[derive(Debug)]
pub(crate) struct Bm25Hits {
    hits: Vec<Bm25Hit>,
    work: WorkspaceReservation,
}

impl std::ops::Deref for Bm25Hits {
    type Target = [Bm25Hit];
    fn deref(&self) -> &Self::Target {
        &self.hits
    }
}

impl Bm25Hits {
    pub(crate) fn empty(context: &JsonOperationContext) -> Self {
        Self {
            hits: Vec::new(),
            work: context.workspace_reservation(),
        }
    }

    pub(crate) fn with_capacity(count: usize, context: &JsonOperationContext) -> Result<Self> {
        let (hits, work) =
            ReservedVec::with_capacity(count, context, "operation_scratch")?.into_parts();
        Ok(Self { hits, work })
    }

    pub(crate) fn push<'a>(
        &mut self,
        id: &str,
        score: f64,
        terms: impl Iterator<Item = MatchedTerm<&'a str>> + Clone,
    ) -> Result<()> {
        assert!(self.hits.len() < self.hits.capacity());
        let owned = OwnedHit::copy(id, score, terms, self.work.context())?;
        self.hits.push(owned.hit);
        self.work.merge(owned.work);
        Ok(())
    }

    pub(crate) fn sort_by_id(&mut self) {
        self.hits
            .sort_unstable_by(|left, right| left.chunk_id.cmp(&right.chunk_id));
    }

    pub(crate) fn as_slice(&self) -> &[Bm25Hit] {
        &self.hits
    }

    pub(crate) fn retain(&mut self, mut include: impl FnMut(&Bm25Hit) -> bool) {
        let mut released = 0;
        self.hits.retain(|hit| {
            let retained = include(hit);
            if !retained {
                released += hit.payload_bytes();
            }
            retained
        });
        self.work.retain(self.work.reserved_bytes() - released);
    }

    pub(crate) fn copy_prefix(&self, limit: usize) -> Result<Self> {
        let mut copy = Self::with_capacity(limit.min(self.len()), self.work.context())?;
        for hit in self.iter().take(limit) {
            copy.push(
                &hit.chunk_id,
                hit.score,
                hit.matched_terms.iter().map(MatchedTerm::borrowed),
            )?;
        }
        Ok(copy)
    }
}

pub(crate) struct TextCandidates {
    limit: usize,
    heap: BinaryHeap<Bm25Hit>,
    work: WorkspaceReservation,
}

impl TextCandidates {
    pub(crate) fn new(limit: usize, context: &JsonOperationContext) -> Self {
        Self {
            limit,
            heap: BinaryHeap::new(),
            work: context.workspace_reservation(),
        }
    }

    pub(crate) fn push<'a>(
        &mut self,
        id: &str,
        score: f64,
        terms: impl Iterator<Item = MatchedTerm<&'a str>> + Clone,
    ) -> Result<()> {
        if self.limit == 0
            || (self.heap.len() == self.limit
                && self.heap.peek().is_some_and(|worst| {
                    RankedChunk {
                        chunk_id: id,
                        score,
                    } <= worst.rank()
                }))
        {
            return Ok(());
        }
        if self.heap.len() == self.limit {
            // The query aborts on admission failure; do not keep the evicted payload alive.
            let removed = self.heap.pop().unwrap();
            let bytes = removed.payload_bytes();
            drop(removed);
            self.work.retain(self.work.reserved_bytes() - bytes);
        }
        let owned = match OwnedHit::copy(id, score, terms.clone(), self.work.context()) {
            Err(error)
                if error.code() == crate::ErrorCode::ResourceExhausted
                    && self.heap.len() < self.heap.capacity() =>
            {
                if !try_shrink_heap(&mut self.heap, &mut self.work)? {
                    return Err(error);
                }
                OwnedHit::copy(id, score, terms, self.work.context())?
            }
            result => result?,
        };
        reserve_heap_slot(&mut self.heap, &mut self.work, self.limit)?;
        self.heap.push(owned.hit);
        self.work.merge(owned.work);
        Ok(())
    }

    pub(crate) fn finish(self) -> Bm25Hits {
        Bm25Hits {
            hits: self.heap.into_sorted_vec(),
            work: self.work,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json::WorkLedger;
    use std::sync::Arc;

    fn terms(seed: usize) -> Vec<MatchedTerm> {
        (0..1 + seed % 5)
            .map(|index| MatchedTerm {
                term: format!("term-{seed}-{index}\u{4e2d}"),
                frequency: index + 1,
                document_frequency: seed + index + 1,
                chunk_length: seed + 11,
                avg_chunk_length: seed as f64 + 0.125,
                bm25_contribution: index as f64 + 0.25,
            })
            .collect()
    }

    fn assert_error(error: &crate::Error, limit: usize, requested: usize) {
        assert_eq!(error.code(), crate::ErrorCode::ResourceExhausted);
        let detail = error
            .detail()
            .iter()
            .cloned()
            .collect::<std::collections::BTreeMap<_, _>>();
        assert_eq!(detail["resource"], "memory");
        assert_eq!(detail["operation"], "operation_scratch");
        assert_eq!(detail["limit_bytes"], limit.to_string());
        assert_eq!(detail["requested_bytes"], requested.to_string());
    }

    #[test]
    fn candidate_second_allocation_accounts_for_both_arrays_and_owned_payloads() {
        let size = std::mem::size_of::<Bm25Hit>();
        for seed in 0..100usize {
            let id = format!("chunk-{seed:04}");
            let next = format!("other-{seed:04}");
            let terms = terms(seed);
            let payload = id.len()
                + terms.len() * std::mem::size_of::<MatchedTerm>()
                + terms.iter().map(|term| term.term.len()).sum::<usize>();
            let active = 13 + seed;
            let needed = active + 2 * payload + 3 * size;
            for parent in [false, true] {
                for delta in [-1isize, 0, 1] {
                    let limit = (needed as isize + delta) as usize;
                    let ledger = Arc::new(WorkLedger::with_limits(
                        if parent { 2 * needed } else { limit },
                        0,
                        if parent { limit } else { 2 * needed },
                    ));
                    let context = JsonOperationContext::new(ledger.clone());
                    let other = JsonOperationContext::new(ledger);
                    let mut outside = other.workspace_reservation();
                    outside
                        .grow_to(active, active, "operation_scratch")
                        .unwrap();
                    let mut blocker = other.workspace_reservation();
                    let blocked = limit - active - payload - size;
                    blocker
                        .grow_to(blocked, blocked, "operation_scratch")
                        .unwrap();
                    let mut candidates = TextCandidates::new(2, &context);
                    candidates
                        .push(&id, 1.0, terms.iter().map(MatchedTerm::borrowed))
                        .unwrap();
                    assert_eq!(candidates.heap.capacity(), 1);
                    let pointer = candidates.heap.as_slice().as_ptr();
                    blocker.retain(0);
                    let result =
                        candidates.push(&next, 2.0, terms.iter().map(MatchedTerm::borrowed));
                    let count = if delta < 0 {
                        assert_error(&result.unwrap_err(), limit, needed);
                        assert_eq!(candidates.heap.capacity(), 1);
                        assert_eq!(candidates.heap.as_slice().as_ptr(), pointer);
                        1
                    } else {
                        result.unwrap();
                        assert_eq!(candidates.heap.capacity(), 2);
                        2
                    };
                    let resident = count * (size + payload);
                    assert_eq!(candidates.work.reserved_bytes(), resident);
                    assert_eq!(
                        context.available_workspace_bytes(),
                        limit - active - resident
                    );
                    let hits = candidates.finish();
                    assert_eq!(hits.as_slice().len(), count);
                    assert_eq!(
                        hits[0].chunk_id.as_str(),
                        if count == 1 {
                            id.as_str()
                        } else {
                            next.as_str()
                        }
                    );
                    assert_eq!(hits[0].score, if count == 1 { 1.0 } else { 2.0 });
                    let expected = [(&next, 2.0f64), (&id, 1.0)];
                    let expected = if count == 1 {
                        &expected[1..]
                    } else {
                        &expected[..]
                    };
                    assert_eq!(
                        hits.iter()
                            .map(|hit| (&hit.chunk_id, hit.score.to_bits()))
                            .collect::<Vec<_>>(),
                        expected
                            .iter()
                            .map(|(id, score)| (*id, score.to_bits()))
                            .collect::<Vec<_>>()
                    );
                    for hit in hits.iter() {
                        assert_eq!(format!("{:?}", hit.matched_terms), format!("{terms:?}"));
                    }
                    assert_eq!(
                        context.available_workspace_bytes(),
                        limit - active - resident
                    );
                    drop(hits);
                    assert_eq!(context.available_workspace_bytes(), limit - active);
                    drop(outside);
                    assert_eq!(context.available_workspace_bytes(), limit);
                }
            }
        }
    }

    #[test]
    fn candidate_spare_recovery_admits_overlap_before_payload_and_heap_growth() {
        let size = std::mem::size_of::<Bm25Hit>();
        for seed in 0..100usize {
            let old = format!("chunk-{seed:04}");
            let terms = terms(seed);
            let payload = old.len()
                + terms.len() * std::mem::size_of::<MatchedTerm>()
                + terms.iter().map(|term| term.term.len()).sum::<usize>();
            let active = 13 + seed;
            let before = 4 * size + payload;
            for (stage, available, length) in [
                (0, size - 1, 2 * size),
                (1, size, 5 * size),
                (2, size, 3 * size + 1),
                (3, size, 2 * size),
            ] {
                let id = "n".repeat(length);
                let limit = active + before + available;
                let context =
                    JsonOperationContext::new(Arc::new(WorkLedger::with_limits(limit, 0, limit)));
                let mut outside = context.workspace_reservation();
                outside
                    .grow_to(active, active, "operation_scratch")
                    .unwrap();
                let mut candidates = TextCandidates::new(usize::MAX, &context);
                candidates
                    .push(&old, 1.25, terms.iter().map(MatchedTerm::borrowed))
                    .unwrap();
                assert_eq!(candidates.heap.capacity(), 4);
                assert_eq!(candidates.work.reserved_bytes(), before);
                let hit = candidates.heap.peek().unwrap();
                let id_pointer = hit.chunk_id.as_ptr();
                let terms_pointer = hit.matched_terms.as_ptr();
                let term_pointers = hit
                    .matched_terms
                    .iter()
                    .map(|term| term.term.as_ptr() as usize)
                    .collect::<Vec<_>>();
                let array = candidates.heap.as_slice().as_ptr();
                let result = candidates.push(&id, 2.25, std::iter::empty());
                let resident = if stage == 3 {
                    result.unwrap();
                    assert_eq!(candidates.heap.capacity(), 2);
                    2 * size + payload + id.len()
                } else {
                    let retained = if stage == 0 { before } else { size + payload };
                    let requested =
                        active + retained + id.len() + if stage == 2 { 2 * size } else { 0 };
                    assert_error(&result.unwrap_err(), limit, requested);
                    assert_eq!(candidates.heap.len(), 1);
                    assert_eq!(candidates.heap.capacity(), if stage == 0 { 4 } else { 1 });
                    if stage == 0 {
                        assert_eq!(candidates.heap.as_slice().as_ptr(), array);
                    }
                    retained
                };
                assert_eq!(candidates.work.reserved_bytes(), resident);
                let hits = candidates.finish();
                let old_hit = hits.last().unwrap();
                assert_eq!(old_hit.chunk_id.as_ptr(), id_pointer);
                assert_eq!(old_hit.matched_terms.as_ptr(), terms_pointer);
                assert_eq!(
                    old_hit
                        .matched_terms
                        .iter()
                        .map(|term| term.term.as_ptr() as usize)
                        .collect::<Vec<_>>(),
                    term_pointers
                );
                let mut expected = vec![Bm25Hit {
                    chunk_id: old.clone(),
                    score: 1.25,
                    matched_terms: terms.clone(),
                }];
                if stage == 3 {
                    expected.insert(
                        0,
                        Bm25Hit {
                            chunk_id: id.clone(),
                            score: 2.25,
                            matched_terms: Vec::new(),
                        },
                    );
                }
                super::super::tests::assert_same_hits(&hits, &expected);
                assert_eq!(
                    context.available_workspace_bytes(),
                    limit - active - resident
                );
                drop(hits);
                drop(outside);
                assert_eq!(context.available_workspace_bytes(), limit);
            }
        }
    }

    #[test]
    fn candidate_owned_fields_and_array_have_exact_admission_and_release() {
        for seed in 0..100 {
            let id = format!("chunk-{seed:04}");
            let terms = terms(seed);
            let stages = std::iter::once(terms.len() * std::mem::size_of::<MatchedTerm>())
                .chain(std::iter::once(id.len()))
                .chain(terms.iter().map(|term| term.term.len()))
                .chain(std::iter::once(std::mem::size_of::<Bm25Hit>()))
                .scan(0, |sum, size| {
                    *sum += size;
                    Some(*sum)
                })
                .collect::<Vec<_>>();
            let required = *stages.last().unwrap();
            let ledger = Arc::new(WorkLedger::with_limits(4096, 4096, 4096));
            let context = JsonOperationContext::new(ledger.clone());
            let other = JsonOperationContext::new(ledger);
            for available in stages
                .iter()
                .map(|bytes| bytes - 1)
                .chain(std::iter::once(required))
            {
                let mut held = other.workspace_reservation();
                held.grow_to(4096 - available, 4096 - available, "operation_scratch")
                    .unwrap();
                let mut candidates = TextCandidates::new(usize::MAX, &context);
                let result = candidates.push(&id, 1.25, terms.iter().map(MatchedTerm::borrowed));
                if available < required {
                    assert_error(&result.unwrap_err(), 4096, 4097);
                    assert!(candidates.heap.is_empty());
                    assert_eq!(candidates.work.reserved_bytes(), 0);
                } else {
                    result.unwrap();
                    assert_eq!(candidates.heap.capacity(), 1);
                    assert_eq!(candidates.work.reserved_bytes(), required);
                    let hits = candidates.finish();
                    assert_eq!(hits[0].chunk_id, id);
                    super::super::tests::assert_same_hits(
                        &hits,
                        &[Bm25Hit {
                            chunk_id: id.clone(),
                            score: 1.25,
                            matched_terms: terms.clone(),
                        }],
                    );
                    assert_error(
                        &hits.copy_prefix(1).unwrap_err(),
                        4096,
                        4096 + std::mem::size_of::<Bm25Hit>(),
                    );
                    drop(hits);
                }
                held.grow_to(4096, 4096, "operation_scratch").unwrap();
            }
        }
    }

    #[test]
    fn candidate_sort_prefix_and_filter_keep_owned_payloads_charged() {
        for seed in 0..100 {
            let ledger = Arc::new(WorkLedger::with_limits(65536, 65536, 65536));
            let context = JsonOperationContext::new(ledger.clone());
            let other = JsonOperationContext::new(ledger);
            let limit = 1 + seed % 11;
            let mut candidates = TextCandidates::new(limit, &context);
            let mut expected = Vec::new();
            for index in 0..31 {
                let id = format!("chunk-{index}");
                let score = if index % 3 == 0 {
                    -0.0
                } else {
                    ((seed + index * 7) % 17) as f64
                };
                let terms = terms(seed + index);
                candidates
                    .push(&id, score, terms.iter().map(MatchedTerm::borrowed))
                    .unwrap();
                expected.push(Bm25Hit {
                    chunk_id: id,
                    score,
                    matched_terms: terms,
                });
            }
            expected.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap()
                    .then_with(|| a.chunk_id.cmp(&b.chunk_id))
            });
            expected.truncate(limit);
            let mut held = other.workspace_reservation();
            held.grow_to(0, 65536, "operation_scratch").unwrap();
            candidates
                .push(
                    "unselected",
                    -1.0,
                    std::iter::once_with(|| -> MatchedTerm<&str> {
                        panic!("unselected explanations must not be built");
                    }),
                )
                .unwrap();
            let pointer = candidates.heap.as_slice().as_ptr();
            let capacity = candidates.heap.capacity();
            let mut hits = candidates.finish();
            assert_eq!(hits.as_ptr(), pointer);
            assert_eq!(hits.hits.capacity(), capacity);
            super::super::tests::assert_same_hits(&hits, &expected);
            let measured = capacity * std::mem::size_of::<Bm25Hit>()
                + hits.iter().map(Bm25Hit::payload_bytes).sum::<usize>();
            assert_eq!(hits.work.reserved_bytes(), measured);
            assert_error(
                &hits.copy_prefix(1).unwrap_err(),
                65536,
                65536 + std::mem::size_of::<Bm25Hit>(),
            );
            assert!(hits.copy_prefix(0).unwrap().is_empty());
            let copy_bytes = std::mem::size_of::<Bm25Hit>() + hits[0].payload_bytes();
            held.retain(held.reserved_bytes() - copy_bytes + 1);
            assert_error(&hits.copy_prefix(1).unwrap_err(), 65536, 65537);
            assert_eq!(hits.work.reserved_bytes(), measured);
            super::super::tests::assert_same_hits(&hits, &expected);
            held.retain(0);
            let copy = hits.copy_prefix(limit.div_ceil(2)).unwrap();
            super::super::tests::assert_same_hits(&copy, &expected[..limit.div_ceil(2)]);
            assert_ne!(copy[0].chunk_id.as_ptr(), hits[0].chunk_id.as_ptr());
            assert_ne!(
                copy[0].matched_terms[0].term.as_ptr(),
                hits[0].matched_terms[0].term.as_ptr()
            );
            drop(copy);
            let released = hits
                .iter()
                .skip(1)
                .map(Bm25Hit::payload_bytes)
                .sum::<usize>();
            let best_id = hits[0].chunk_id.clone();
            hits.retain(|hit| hit.chunk_id == best_id);
            assert_eq!(hits.work.reserved_bytes(), measured - released);
            assert_eq!(hits.as_ptr(), pointer);
            drop(hits);
            held.grow_to(65536, 65536, "operation_scratch").unwrap();
            let mut empty = TextCandidates::new(0, &context);
            empty.push("zero-limit", 1.0, std::iter::empty()).unwrap();
            assert!(empty.finish().is_empty());
        }
    }

    #[test]
    fn competitive_replacement_releases_evicted_payload_before_admission() {
        for seed in 0..100 {
            let terms = terms(seed);
            let bytes = std::mem::size_of::<Bm25Hit>()
                + 4
                + terms.len() * std::mem::size_of::<MatchedTerm>()
                + terms.iter().map(|term| term.term.len()).sum::<usize>();
            let context =
                JsonOperationContext::new(Arc::new(WorkLedger::with_limits(bytes, bytes, bytes)));
            let mut candidates = TextCandidates::new(1, &context);
            candidates
                .push("old!", 0.0, terms.iter().map(MatchedTerm::borrowed))
                .unwrap();
            let array_pointer = candidates.heap.as_slice().as_ptr();
            for rank in 1..=100 {
                candidates
                    .push("new!", rank as f64, terms.iter().map(MatchedTerm::borrowed))
                    .unwrap();
                assert_eq!(candidates.work.reserved_bytes(), bytes);
                assert_eq!(candidates.heap.as_slice().as_ptr(), array_pointer);
            }
            assert_eq!(candidates.finish()[0].score, 100.0);
            context
                .workspace_reservation()
                .grow_to(bytes, bytes, "operation_scratch")
                .unwrap();
        }
    }
}
