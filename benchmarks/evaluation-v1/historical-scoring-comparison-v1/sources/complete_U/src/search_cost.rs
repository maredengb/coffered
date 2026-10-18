//! Fixed, optional diagnostics for one measured search. This owns no resource
//! credit, changes no admission/retry policy, and never enters public query JSON.
use std::io::{Read, Seek, SeekFrom};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use crate::json::{JsonBuilder, JsonOperationContext, JsonValue};
use crate::Result;

pub(crate) const VERSION: &str = "cofferdb-search-cost-v1";

#[derive(Clone, Copy)]
pub(crate) enum Stage {
    Preparation,
    Recall,
    ScoreCompletion,
    NormalizationFusion,
    ResultMaterialization,
}
impl Stage {
    const ALL: [Self; 5] = [
        Self::Preparation,
        Self::Recall,
        Self::ScoreCompletion,
        Self::NormalizationFusion,
        Self::ResultMaterialization,
    ];
    fn name(self) -> &'static str {
        match self {
            Self::Preparation => "preparation",
            Self::Recall => "recall",
            Self::ScoreCompletion => "score_completion",
            Self::NormalizationFusion => "normalization_fusion",
            Self::ResultMaterialization => "result_materialization",
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) enum Counter {
    TextDocumentVisits,
    TextPostingVisits,
    VectorChecks,
    DistanceComponents,
    Bm25ScoreReuse,
    VectorScoreReuse,
    ZeroMatchProofReuse,
    NormalizationScoreVisits,
    FusionCandidateVisits,
    PrimaryRequestedBytes,
    IndexRequestedBytes,
    PrimaryReadCalls,
    IndexReadCalls,
    PrimaryReadBytes,
    IndexReadBytes,
}
impl Counter {
    const ALL: [Self; 15] = [
        Self::TextDocumentVisits,
        Self::TextPostingVisits,
        Self::VectorChecks,
        Self::DistanceComponents,
        Self::Bm25ScoreReuse,
        Self::VectorScoreReuse,
        Self::ZeroMatchProofReuse,
        Self::NormalizationScoreVisits,
        Self::FusionCandidateVisits,
        Self::PrimaryRequestedBytes,
        Self::IndexRequestedBytes,
        Self::PrimaryReadCalls,
        Self::IndexReadCalls,
        Self::PrimaryReadBytes,
        Self::IndexReadBytes,
    ];
    fn name(self) -> &'static str {
        match self {
            Self::TextDocumentVisits => "text_document_visits",
            Self::TextPostingVisits => "text_posting_visits",
            Self::VectorChecks => "vector_checks",
            Self::DistanceComponents => "distance_components",
            Self::Bm25ScoreReuse => "bm25_score_reuse",
            Self::VectorScoreReuse => "vector_score_reuse",
            Self::ZeroMatchProofReuse => "zero_match_proof_reuse",
            Self::NormalizationScoreVisits => "normalization_score_visits",
            Self::FusionCandidateVisits => "fusion_candidate_visits",
            Self::PrimaryRequestedBytes => "primary_requested_bytes",
            Self::IndexRequestedBytes => "index_requested_bytes",
            Self::PrimaryReadCalls => "primary_read_calls",
            Self::IndexReadCalls => "index_read_calls",
            Self::PrimaryReadBytes => "primary_read_bytes",
            Self::IndexReadBytes => "index_read_bytes",
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) enum Count {
    Bm25Recall,
    VectorRecall,
    RecallUnion,
    FusionRetained,
    Returned,
}
impl Count {
    const ALL: [Self; 5] = [
        Self::Bm25Recall,
        Self::VectorRecall,
        Self::RecallUnion,
        Self::FusionRetained,
        Self::Returned,
    ];
    fn name(self) -> &'static str {
        match self {
            Self::Bm25Recall => "bm25_recall_count",
            Self::VectorRecall => "vector_recall_count",
            Self::RecallUnion => "recall_union_count",
            Self::FusionRetained => "fusion_retained_count",
            Self::Returned => "returned_count",
        }
    }
}

#[derive(Default)]
struct Slot {
    counters: [AtomicU64; 15],
    elapsed_ns: AtomicU64,
    entered: AtomicBool,
}
struct State {
    slots: [Slot; 5],
    counts: [AtomicU64; 5],
    count_known: [AtomicBool; 5],
    active: AtomicBool,
    saturated: AtomicBool,
}

pub(crate) struct SearchCostTrace(Arc<State>);
#[derive(Clone)]
pub(crate) struct Handle {
    state: Arc<State>,
    stage: Stage,
}
pub(crate) struct Timer {
    handle: Handle,
    started: Instant,
}
pub(crate) struct Finish<'a>(&'a SearchCostTrace);
impl Drop for Finish<'_> {
    fn drop(&mut self) {
        self.0.finish();
    }
}

impl SearchCostTrace {
    pub(crate) fn new() -> Self {
        Self(Arc::new(State {
            slots: std::array::from_fn(|_| Slot::default()),
            counts: std::array::from_fn(|_| AtomicU64::new(0)),
            count_known: std::array::from_fn(|_| AtomicBool::new(false)),
            active: AtomicBool::new(true),
            saturated: AtomicBool::new(false),
        }))
    }
    pub(crate) fn handle(&self) -> Handle {
        Handle {
            state: Arc::clone(&self.0),
            stage: Stage::Preparation,
        }
    }
    // Cached reservations can outlive this search. Their diagnostic handles must
    // stop recording before verification/journal IO or another operation begins.
    pub(crate) fn finish(&self) {
        self.0.active.store(false, Ordering::Relaxed);
    }
    pub(crate) fn scope(&self) -> Finish<'_> {
        Finish(self)
    }
    pub(crate) fn valid(&self) -> bool {
        for counter in Counter::ALL {
            self.total(counter);
        }
        !self.0.saturated.load(Ordering::Relaxed)
    }
    pub(crate) fn value(&self, stage: Stage, counter: Counter) -> u64 {
        self.0.slots[stage as usize].counters[counter as usize].load(Ordering::Relaxed)
    }
    pub(crate) fn total(&self, counter: Counter) -> u64 {
        Stage::ALL.into_iter().fold(0u64, |n, stage| {
            n.checked_add(self.value(stage, counter))
                .unwrap_or_else(|| {
                    self.0.saturated.store(true, Ordering::Relaxed);
                    u64::MAX
                })
        })
    }
    pub(crate) fn json(&self, builder: &JsonBuilder<'_>) -> Result<JsonValue> {
        let _valid = self.valid();
        let counters = |stage: Option<Stage>| {
            let mut fields = Counter::ALL.map(|counter| (counter.name(), JsonValue::Null));
            for (counter, field) in Counter::ALL.into_iter().zip(&mut fields) {
                let n =
                    stage.map_or_else(|| self.total(counter), |stage| self.value(stage, counter));
                field.1 = builder.optional_u64(Some(n))?;
            }
            builder.object(fields)
        };
        let stages = builder.array(Stage::ALL.into_iter().map(|stage| {
            let slot = &self.0.slots[stage as usize];
            builder.object([
                ("stage", builder.string(stage.name())?),
                (
                    "entered",
                    JsonValue::Bool(slot.entered.load(Ordering::Relaxed)),
                ),
                (
                    "elapsed_ms",
                    JsonValue::Number(slot.elapsed_ns.load(Ordering::Relaxed) as f64 / 1e6),
                ),
                ("work", counters(Some(stage))?),
            ])
        }))?;
        let mut fields = Count::ALL.map(|count| (count.name(), JsonValue::Null));
        for (count, field) in Count::ALL.into_iter().zip(&mut fields) {
            field.1 = builder.optional_u64(
                self.0.count_known[count as usize]
                    .load(Ordering::Relaxed)
                    .then(|| self.0.counts[count as usize].load(Ordering::Relaxed)),
            )?;
        }
        let counts = builder.object(fields)?;
        builder.object([
            ("version", builder.string(VERSION)?),
            ("counter_encoding", builder.string("exact JSON integers through 2^53-1; unsigned decimal strings above it")?),
            ("counter_saturated", JsonValue::Bool(self.0.saturated.load(Ordering::Relaxed))),
            ("stages", stages),
            ("totals", counters(None)?),
            ("candidate_counts", counts),
            ("staging_read_bytes", JsonValue::Number(0.0)),
            ("physical_read_bytes", JsonValue::Null),
            ("physical_read_reason", builder.string("physical disk/cache IO is unavailable; application Read/read_at bytes are not physical IO")?),
        ])
    }
}
impl Drop for SearchCostTrace {
    fn drop(&mut self) {
        self.finish();
    }
}
impl Handle {
    pub(crate) fn stage(&self, stage: Stage) -> Self {
        Self {
            state: Arc::clone(&self.state),
            stage,
        }
    }
    pub(crate) fn add(&self, counter: Counter, amount: u64) {
        if self.state.active.load(Ordering::Relaxed) {
            self.add_atomic(
                &self.state.slots[self.stage as usize].counters[counter as usize],
                amount,
            );
        }
    }
    fn add_atomic(&self, counter: &AtomicU64, amount: u64) {
        let previous = counter
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| {
                Some(n.saturating_add(amount))
            })
            .unwrap();
        if previous.checked_add(amount).is_none() {
            self.state.saturated.store(true, Ordering::Relaxed);
        }
    }
    pub(crate) fn count(&self, count: Count, amount: usize) {
        if self.state.active.load(Ordering::Relaxed) {
            self.state.counts[count as usize].store(amount as u64, Ordering::Relaxed);
            self.state.count_known[count as usize].store(true, Ordering::Relaxed);
        }
    }
    pub(crate) fn timer(&self) -> Timer {
        if self.state.active.load(Ordering::Relaxed) {
            self.state.slots[self.stage as usize]
                .entered
                .store(true, Ordering::Relaxed);
        }
        Timer {
            handle: self.clone(),
            started: Instant::now(),
        }
    }
}
impl Drop for Timer {
    fn drop(&mut self) {
        if self.handle.state.active.load(Ordering::Relaxed) {
            let nanos = self.started.elapsed().as_nanos();
            if nanos > u64::MAX as u128 {
                self.handle.state.saturated.store(true, Ordering::Relaxed);
            }
            self.handle.add_atomic(
                &self.handle.state.slots[self.handle.stage as usize].elapsed_ns,
                nanos.min(u64::MAX as u128) as u64,
            );
        }
    }
}

// Counts every attempted application read and each successfully returned byte,
// including short reads, checksum drains and bytes preceding a later error.
pub(crate) struct IndexReader<'a, R> {
    pub(crate) source: &'a mut R,
    pub(crate) context: Option<&'a JsonOperationContext>,
}
impl<R: Read> Read for IndexReader<'_, R> {
    fn read(&mut self, bytes: &mut [u8]) -> std::io::Result<usize> {
        if let Some(context) = self.context {
            context.search_cost_add(Counter::IndexReadCalls, 1);
        }
        let count = self.source.read(bytes)?;
        if let Some(context) = self.context {
            context.search_cost_add(Counter::IndexReadBytes, count as u64);
        }
        Ok(count)
    }
}
impl<R: Seek> Seek for IndexReader<'_, R> {
    fn seek(&mut self, offset: SeekFrom) -> std::io::Result<u64> {
        self.source.seek(offset)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json::{JsonInputKind, JsonOperationContext};

    #[test]
    fn trace_stage_clones_share_budget_without_relabeling_or_recording_after_finish() {
        fn send_sync<T: Send + Sync>() {}
        send_sync::<JsonOperationContext>();
        send_sync::<Handle>();
        let original = JsonOperationContext::standalone();
        let available = original.available_workspace_bytes();
        let trace = SearchCostTrace::new();
        let recall = original
            .with_search_cost(&trace)
            .search_cost_stage(Stage::Recall);
        let completion = recall.search_cost_stage(Stage::ScoreCompletion);
        let mut reservation = recall.workspace_reservation();
        reservation.grow_to(17, 17, "operation_scratch").unwrap();
        assert_eq!(completion.available_workspace_bytes(), available - 17);
        recall.search_cost_add(Counter::VectorChecks, 3);
        completion.search_cost_add(Counter::VectorChecks, 2);
        assert_eq!(trace.value(Stage::Recall, Counter::VectorChecks), 3);
        assert_eq!(
            trace.value(Stage::ScoreCompletion, Counter::VectorChecks),
            2
        );
        drop(trace.scope());
        let frozen = trace.json(&JsonBuilder::host()).unwrap();
        drop(
            completion
                .search_cost_stage(Stage::ResultMaterialization)
                .search_cost_timer(),
        );
        assert_eq!(trace.json(&JsonBuilder::host()).unwrap(), frozen);
        recall.search_cost_add(Counter::VectorChecks, 100);
        completion.search_cost_count(Count::Returned, 9);
        assert_eq!(trace.total(Counter::VectorChecks), 5);
        drop(reservation);
        assert_eq!(original.available_workspace_bytes(), available);
        let json = trace.json(&JsonBuilder::host()).unwrap();
        assert!(json
            .get("candidate_counts")
            .unwrap()
            .get("returned_count")
            .unwrap()
            .is_null());
        assert!(json.get("physical_read_bytes").unwrap().is_null());
    }

    #[test]
    fn trace_exact_large_integers_and_overflow_invalidation_are_budgeted() {
        let trace = SearchCostTrace::new();
        let context = JsonOperationContext::standalone().with_search_cost(&trace);
        context.search_cost_add(Counter::PrimaryReadBytes, u64::MAX - 1);
        assert!(trace.valid());
        let mut work = context.reservation();
        let builder = JsonBuilder::budgeted(&mut work, JsonInputKind::BenchmarkArtifact);
        let json = trace.json(&builder).unwrap();
        assert_eq!(
            json.get("totals")
                .unwrap()
                .get("primary_read_bytes")
                .unwrap()
                .as_str(),
            Some("18446744073709551614")
        );
        assert!(work.reserved_bytes() > 0);
        context
            .search_cost_stage(Stage::Recall)
            .search_cost_add(Counter::PrimaryReadBytes, 2);
        assert!(!trace.valid()); // Aggregate overflow also invalidates a measurement.
        context.search_cost_add(Counter::PrimaryReadBytes, 2);
        assert!(!trace.valid()); // Per-stage overflow remains explicit.
        let json = trace.json(&JsonBuilder::host()).unwrap();
        assert_eq!(json.get("counter_saturated").unwrap().as_bool(), Some(true));
    }

    #[test]
    fn index_reads_count_partial_success_and_attempts_before_error_without_seek_bytes() {
        struct Input {
            remaining: usize,
        }
        impl Read for Input {
            fn read(&mut self, output: &mut [u8]) -> std::io::Result<usize> {
                if self.remaining == 0 {
                    return Err(std::io::ErrorKind::PermissionDenied.into());
                }
                let n = output.len().min(self.remaining).min(3);
                output[..n].fill(1);
                self.remaining -= n;
                Ok(n)
            }
        }
        impl Seek for Input {
            fn seek(&mut self, _: SeekFrom) -> std::io::Result<u64> {
                Ok(0)
            }
        }
        let trace = SearchCostTrace::new();
        let context = JsonOperationContext::standalone().with_search_cost(&trace);
        let mut source = Input { remaining: 7 };
        let mut reader = IndexReader {
            source: &mut source,
            context: Some(&context),
        };
        reader.seek(SeekFrom::Start(0)).unwrap();
        assert_eq!(
            reader.read_exact(&mut [0; 10]).unwrap_err().kind(),
            std::io::ErrorKind::PermissionDenied
        );
        assert_eq!(trace.total(Counter::IndexReadBytes), 7);
        assert_eq!(trace.total(Counter::IndexReadCalls), 4);
        assert_eq!(trace.total(Counter::PrimaryReadBytes), 0);
    }
}
