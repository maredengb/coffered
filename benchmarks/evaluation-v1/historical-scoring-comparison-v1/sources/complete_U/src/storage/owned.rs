use std::mem::size_of;

use super::{CommittedRecord, RecordPayload};
use crate::json::{
    CatalogReservation, JsonOperationContext, JsonReservation, ReservedString, WorkspaceReservation,
};
use crate::Result;

pub(crate) struct RecordReservation {
    pub(crate) json: JsonReservation,
    workspace: Option<WorkspaceReservation>,
}

impl RecordReservation {
    pub(crate) fn new(json: JsonReservation, workspace: WorkspaceReservation) -> Self {
        Self {
            json,
            workspace: Some(workspace),
        }
    }

    pub(crate) fn json_bytes(&self) -> usize {
        self.json.reserved_bytes()
    }

    pub(crate) fn workspace_bytes(&self) -> usize {
        self.workspace
            .as_ref()
            .map_or(0, WorkspaceReservation::reserved_bytes)
    }

    pub(crate) fn transfer_to_catalog(
        &mut self,
        catalog: &mut CatalogReservation,
        json_bytes: usize,
        workspace_bytes: usize,
    ) -> Result<()> {
        catalog.absorb(
            &mut self.json,
            self.workspace.as_mut(),
            json_bytes,
            workspace_bytes,
        )
    }

    pub(crate) fn merge(&mut self, other: impl Into<Self>) {
        let Self { json, workspace } = other.into();
        self.json.merge(json);
        if let Some(workspace) = workspace {
            self.merge_workspace(workspace);
        }
    }

    pub(crate) fn merge_workspace(&mut self, workspace: WorkspaceReservation) {
        match &mut self.workspace {
            Some(retained) => retained.merge(workspace),
            None => self.workspace = Some(workspace),
        }
    }

    pub(crate) fn copy_string(
        &mut self,
        value: &str,
        context: &JsonOperationContext,
    ) -> Result<String> {
        let (value, work) = ReservedString::copy(value, context)?.into_parts();
        self.merge_workspace(work);
        Ok(value)
    }

    pub(crate) fn copy_strings<const N: usize>(
        &mut self,
        values: [&str; N],
        context: &JsonOperationContext,
    ) -> Result<[String; N]> {
        let bytes = values.iter().map(|value| value.len() as u128).sum();
        let mut work = context.workspace_reservation();
        work.grow_to_wide(bytes, bytes, "operation_scratch")?;
        // Declare copies last so a failed allocation frees them before the lease.
        let mut copies: [String; N] = std::array::from_fn(|_| String::new());
        for (copy, value) in copies.iter_mut().zip(values) {
            *copy = crate::json::copy_precharged_string(value)?;
        }
        self.merge_workspace(work);
        Ok(copies)
    }

    pub(crate) fn retain_workspace(&mut self, bytes: usize) {
        match &mut self.workspace {
            Some(workspace) => workspace.retain(bytes),
            None => assert_eq!(bytes, 0),
        }
    }

    pub(crate) fn parts_mut(
        &mut self,
    ) -> (&mut JsonReservation, Option<&mut WorkspaceReservation>) {
        (&mut self.json, self.workspace.as_mut())
    }

    pub(crate) fn into_json(self) -> JsonReservation {
        self.json
    }
}

impl From<JsonReservation> for RecordReservation {
    fn from(json: JsonReservation) -> Self {
        Self {
            json,
            workspace: None,
        }
    }
}

// Raw API batches are atomic. Reserve both arrays that coexist during commit;
// unlike prepared imports, these batches cannot be split to fit the workspace.
pub(crate) struct OwnedRecordBatch {
    records: Vec<RecordPayload>,
    work: WorkspaceReservation,
    payloads: RecordReservation,
    commit_timestamp: Option<ReservedString>,
}

// The reclaimer is borrowed only while constructing records, never across IO.
pub(crate) struct ReclaimingRecordBatch<F> {
    batch: OwnedRecordBatch,
    reclaim: F,
}

impl<F: FnMut() -> bool> ReclaimingRecordBatch<F> {
    pub(crate) fn into_batch(self) -> OwnedRecordBatch {
        self.batch
    }

    pub(crate) fn records(&self) -> &[RecordPayload] {
        self.batch.records()
    }

    // Admit one independent field or scratch allocation, not an entire write.
    pub(crate) fn admit<T>(&mut self, attempt: impl FnMut() -> Result<T>) -> Result<T> {
        crate::json::retry_after_memory_reclaim(attempt, &mut self.reclaim)
    }

    fn prepare<T>(
        &mut self,
        mut attempt: impl FnMut(&mut OwnedRecordBatch) -> Result<T>,
    ) -> Result<T> {
        crate::json::retry_after_memory_reclaim(|| attempt(&mut self.batch), &mut self.reclaim)
    }

    pub(crate) fn retain_json(&mut self, work: JsonReservation) {
        self.batch.retain_json(work);
    }

    pub(crate) fn retain_workspace(&mut self, work: WorkspaceReservation) {
        self.batch.retain_workspace(work);
    }

    pub(crate) fn retain_string(&mut self, value: ReservedString) -> String {
        self.batch.retain_string(value)
    }

    pub(crate) fn copy_string(&mut self, value: &str) -> Result<String> {
        self.prepare(|batch| batch.copy_string(value))
    }

    pub(crate) fn timestamp(&mut self, millis: u64) -> Result<String> {
        self.prepare(|batch| batch.timestamp(millis))
    }

    pub(crate) fn text_hash(&mut self, text: &str) -> Result<String> {
        self.prepare(|batch| batch.text_hash(text))
    }

    pub(crate) fn push_tombstone(&mut self, kind: &str, id: &str, reason: &str) -> Result<()> {
        let tombstone = crate::model::Tombstone {
            target_type: self.copy_string(kind)?,
            target_id: self.copy_string(id)?,
            deleted_at: self.timestamp(crate::time::unix_millis_now())?,
            reason: Some(self.copy_string(reason)?),
        };
        self.push(RecordPayload::Tombstone(tombstone))
    }

    pub(crate) fn push(&mut self, record: RecordPayload) -> Result<()> {
        self.prepare(OwnedRecordBatch::reserve_slot)?;
        self.batch.records.push(record);
        Ok(())
    }
}

// The encoder is prepared against these immutable payloads before Catalog
// takes their JSON credit. No mutable record access is exposed after admission.
pub(crate) struct EncodedRecordBatch {
    pub(super) batch: OwnedRecordBatch,
    pub(super) encoding: super::codec::PreparedBatchEncoding,
}

impl EncodedRecordBatch {
    pub(crate) fn records(&self) -> &[RecordPayload] {
        self.batch.records()
    }

    pub(crate) fn records_and_json(&mut self) -> (&[RecordPayload], &mut JsonReservation) {
        (&self.batch.records, &mut self.batch.payloads.json)
    }
}

impl OwnedRecordBatch {
    pub(crate) fn new(context: &JsonOperationContext) -> Self {
        Self {
            records: Vec::new(),
            work: context.workspace_reservation(),
            payloads: RecordReservation::from(context.reservation()),
            commit_timestamp: None,
        }
    }

    pub(crate) fn reclaiming<F: FnMut() -> bool>(self, reclaim: F) -> ReclaimingRecordBatch<F> {
        ReclaimingRecordBatch {
            batch: self,
            reclaim,
        }
    }

    #[cfg(test)]
    pub(crate) fn single(record: RecordPayload, context: &JsonOperationContext) -> Result<Self> {
        let mut batch = Self::new(context);
        batch.push(record)?;
        Ok(batch)
    }

    pub(crate) fn records(&self) -> &[RecordPayload] {
        &self.records
    }

    pub(crate) fn context(&self) -> &JsonOperationContext {
        self.work.context()
    }

    pub(crate) fn retain_json(&mut self, work: JsonReservation) {
        self.payloads.merge(work);
    }

    pub(crate) fn retain_workspace(&mut self, work: WorkspaceReservation) {
        self.payloads.merge_workspace(work);
    }

    pub(crate) fn retain_string(&mut self, value: ReservedString) -> String {
        let (value, work) = value.into_parts();
        self.retain_workspace(work);
        value
    }

    pub(crate) fn copy_string(&mut self, value: &str) -> Result<String> {
        let value = ReservedString::copy_in(value, self.context(), "batch_prepare")?;
        Ok(self.retain_string(value))
    }

    pub(crate) fn timestamp(&mut self, millis: u64) -> Result<String> {
        let value = ReservedString::build(self.context(), "batch_prepare", |out| {
            crate::time::write_unix_millis(out, millis)
        })?;
        Ok(self.retain_string(value))
    }

    pub(crate) fn text_hash(&mut self, text: &str) -> Result<String> {
        let value = ReservedString::build_exact(64, self.context(), "batch_prepare", |out| {
            crate::hash::write_digest_hex(out, crate::hash::sha256(text.as_bytes()))
        })?;
        Ok(self.retain_string(value))
    }

    pub(crate) fn prepare_encoding(
        mut self,
        reclaim: impl FnOnce() -> bool,
    ) -> Result<EncodedRecordBatch> {
        let context = self.context().clone();
        let encoding = crate::json::retry_after_memory_reclaim(
            || {
                super::codec::PreparedBatchEncoding::prepare(
                    &self.records,
                    &mut self.payloads.json,
                    &context,
                )
            },
            reclaim,
        )?;
        Ok(EncodedRecordBatch {
            batch: self,
            encoding,
        })
    }

    #[cfg(test)]
    pub(crate) fn push(&mut self, record: RecordPayload) -> Result<()> {
        self.reserve_slot()?;
        self.records.push(record);
        Ok(())
    }

    fn reserve_slot(&mut self) -> Result<()> {
        if self.commit_timestamp.is_none() {
            self.commit_timestamp = Some(ReservedString::with_capacity(
                crate::time::MAX_TIMESTAMP_BYTES,
                self.context(),
                "batch_prepare",
            )?);
        }
        if self.records.len() < self.records.capacity() {
            return Ok(());
        }
        let minimum_capacity = self.records.len() + 1;
        let preferred_capacity = minimum_capacity
            .max(self.records.capacity().saturating_mul(2))
            .max(4);
        let slot_bytes = size_of::<RecordPayload>() + size_of::<CommittedRecord>();
        let bytes = |capacity: usize| {
            capacity
                .checked_mul(slot_bytes)
                .ok_or_else(|| std::io::Error::other("batch array size overflow"))
        };
        let previous = self.work.reserved_bytes();
        let granted = self.work.grow_to(
            bytes(minimum_capacity)?,
            bytes(preferred_capacity)?,
            "batch_prepare",
        )?;
        let capacity = granted / slot_bytes;
        if capacity > self.records.capacity() {
            if let Err(error) = self
                .records
                .try_reserve_exact(capacity - self.records.len())
            {
                self.work.retain(previous);
                return Err(std::io::Error::other(error.to_string()).into());
            }
            if self.records.capacity() > capacity {
                // An allocator overgrant must not survive without a reservation.
                self.records = Vec::new();
                self.work.retain(0);
                return Err(std::io::Error::other(
                    "batch array allocation exceeds requested capacity",
                )
                .into());
            }
        }
        self.work.retain(bytes(self.records.capacity())?);
        Ok(())
    }

    pub(super) fn into_parts(
        mut self,
    ) -> (
        Vec<RecordPayload>,
        WorkspaceReservation,
        RecordReservation,
        Option<ReservedString>,
    ) {
        self.work.retain(
            self.records.capacity() * size_of::<RecordPayload>()
                + self.records.len() * size_of::<CommittedRecord>(),
        );
        (
            self.records,
            self.work,
            self.payloads,
            self.commit_timestamp,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json::JsonProcessingPolicyV1;
    use crate::ErrorCode;

    #[test]
    fn string_group_admits_once_before_copy_and_preserves_prior_owners_on_failure() {
        use crate::json::WorkLedger;
        use std::sync::Arc;

        let first = "first".repeat(53);
        let values = [first.as_str(), "second field", "\u{4e2d}\u{6587}", ""];
        let group: usize = values.iter().map(|value| value.len()).sum();
        for parent in [false, true] {
            for delta in [-1isize, 0, 1] {
                let limit = (group as isize + 5 + 7 + delta) as usize;
                let (working, total) = if parent {
                    (limit + 32, limit)
                } else {
                    (limit, limit + 32)
                };
                let context =
                    JsonOperationContext::new(Arc::new(WorkLedger::with_limits(working, 0, total)));
                let mut held = context.workspace_reservation();
                held.grow_to(7, 7, "operation_scratch").unwrap();
                let mut owner = RecordReservation::from(context.reservation());
                let previous = owner.copy_string("prior", &context).unwrap();
                let pointer = previous.as_ptr();
                let (result, allocations) =
                    crate::test_alloc::key_allocations(values[0].len(), || {
                        owner.copy_strings(values, &context)
                    });
                if delta < 0 {
                    assert_eq!(result.unwrap_err().code(), ErrorCode::ResourceExhausted);
                    assert_eq!(allocations.allocated, 0);
                    assert_eq!(owner.workspace_bytes(), 5);
                } else {
                    let copied = result.unwrap();
                    assert_eq!(copied.each_ref().map(String::as_str), values);
                    assert_eq!(
                        copied.each_ref().map(String::capacity),
                        values.map(str::len)
                    );
                    assert_eq!(allocations.allocated, 1);
                    assert_eq!(owner.workspace_bytes(), group + 5);
                    drop(copied);
                    owner.retain_workspace(5);
                }
                assert_eq!(previous, "prior");
                assert_eq!(previous.as_ptr(), pointer);
                assert_eq!(held.reserved_bytes(), 7);
                assert_eq!(context.available_workspace_bytes(), limit - 12);
                drop((previous, owner, held));
                assert_eq!(context.available_workspace_bytes(), working.min(total));
            }
        }
    }

    #[test]
    fn string_group_native_failure_refunds_partial_copies_without_retrying() {
        let context = JsonOperationContext::standalone();
        let baseline = context.available_workspace_bytes();
        let mut owner = RecordReservation::from(context.reservation());
        let previous = owner.copy_string("prior", &context).unwrap();
        let values = [
            "first owned field",
            "second owned field",
            "last owned field",
        ];
        for failed in [2, 3] {
            let error = crate::test_alloc::fail_nth_allocation_once(failed, || {
                owner.copy_strings(values, &context)
            })
            .unwrap_err();
            assert_eq!(error.code(), ErrorCode::IoError);
            assert_eq!(owner.workspace_bytes(), previous.capacity());
            assert_eq!(
                context.available_workspace_bytes(),
                baseline - previous.capacity()
            );
            let copied = owner.copy_strings(values, &context).unwrap();
            assert_eq!(copied.each_ref().map(String::as_str), values);
            drop(copied);
            owner.retain_workspace(previous.capacity());
        }
        drop((previous, owner));
        assert_eq!(context.available_workspace_bytes(), baseline);
    }

    fn record(index: usize) -> RecordPayload {
        RecordPayload::CollectionDrop {
            collection_id: format!("col_{}", index + 1),
            name: format!("name_{index}"),
            dropped_at: "2026-10-08T00:00:00Z".into(),
            reason: None,
        }
    }

    #[test]
    fn field_reclamation_preserves_completed_records_and_exact_live_credit() {
        use std::cell::Cell;

        fn retain_input(batch: &mut OwnedRecordBatch, record: &RecordPayload) {
            let bytes = super::super::record_workspace_bytes(record);
            let mut work = batch.context().workspace_reservation();
            work.grow_to(bytes, bytes, "batch_prepare").unwrap();
            batch.retain_workspace(work);
        }

        let limit = JsonProcessingPolicyV1::WORK_BUDGET_BYTES;
        for seed in 0..64usize {
            for mode in 0..6 {
                for release in [false, true] {
                    let context = JsonOperationContext::standalone();
                    let mut batch = OwnedRecordBatch::new(&context);
                    for index in 0..4 {
                        let value = record(index);
                        retain_input(&mut batch, &value);
                        batch.push(value).unwrap();
                    }
                    let name_pointer = |record: &RecordPayload| match record {
                        RecordPayload::CollectionDrop { name, .. } => name.as_ptr(),
                        _ => panic!(),
                    };
                    let pointers = batch.records().iter().map(name_pointer).collect::<Vec<_>>();
                    let mut pending = record(99);
                    if mode != 3 {
                        let RecordPayload::CollectionDrop { name, .. } = &mut pending else {
                            panic!()
                        };
                        *name = String::new();
                    }
                    if mode != 4 {
                        retain_input(&mut batch, &pending);
                    }
                    let mut held = context.workspace_reservation();
                    let spare = held.grow_to(0, limit, "operation_scratch").unwrap();
                    let available = if mode == 1 { seed % 20 } else { seed };
                    held.retain(spare - available);
                    let calls = Cell::new(0);
                    let mut prepared = batch.reclaiming(|| {
                        calls.set(calls.get() + 1);
                        if release {
                            held.retain(0);
                        }
                        release
                    });
                    let text = "x".repeat(64 + seed);
                    let result = match mode {
                        3 => prepared.push(pending),
                        4 => prepared.push_tombstone("document", "doc_0000000000000001", "replace"),
                        _ => (|| {
                            let value = match mode {
                                0 => prepared.copy_string(&text)?,
                                1 => prepared.timestamp(1000 + seed as u64)?,
                                2 => prepared.text_hash(&text)?,
                                _ => {
                                    let value = prepared.admit(|| {
                                        ReservedString::copy_in(&text, &context, "batch_prepare")
                                    })?;
                                    prepared.retain_string(value)
                                }
                            };
                            let RecordPayload::CollectionDrop { name, .. } = &mut pending else {
                                panic!()
                            };
                            *name = value;
                            prepared.push(pending)
                        })(),
                    };
                    let batch = prepared.into_batch();
                    assert_eq!(
                        calls.get(),
                        1,
                        "seed={seed}, mode={mode}, release={release}"
                    );
                    assert_eq!(batch.records.len(), if release { 5 } else { 4 });
                    assert_eq!(
                        batch.records[..4]
                            .iter()
                            .map(name_pointer)
                            .collect::<Vec<_>>(),
                        pointers
                    );
                    if release {
                        result.unwrap();
                        let fields = batch
                            .records
                            .iter()
                            .map(super::super::record_workspace_bytes)
                            .sum::<usize>();
                        assert_eq!(batch.payloads.workspace_bytes(), fields);
                        let remaining = limit
                            - fields
                            - batch.work.reserved_bytes()
                            - crate::time::MAX_TIMESTAMP_BYTES;
                        context
                            .workspace_reservation()
                            .grow_to(remaining, remaining, "operation_scratch")
                            .unwrap();
                        match &batch.records[4] {
                            RecordPayload::CollectionDrop { name, .. } => assert_eq!(
                                *name,
                                match mode {
                                    1 => crate::time::format_unix_millis(1000 + seed as u64),
                                    2 => crate::model::chunk_text_hash(&text),
                                    3 => "name_99".into(),
                                    _ => text,
                                }
                            ),
                            RecordPayload::Tombstone(value) => {
                                assert_eq!(value.target_type, "document");
                                assert_eq!(value.target_id, "doc_0000000000000001");
                                assert_eq!(value.reason.as_deref(), Some("replace"));
                            }
                            _ => panic!(),
                        }
                    } else {
                        let error = result.unwrap_err();
                        assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                        for (key, value) in [("resource", "memory"), ("operation", "batch_prepare")]
                        {
                            assert!(error.detail().contains(&(key.into(), value.into())));
                        }
                    }
                    drop((batch, held));
                    context
                        .workspace_reservation()
                        .grow_to(limit, limit, "operation_scratch")
                        .unwrap();
                }
            }
        }
    }

    #[test]
    fn preferred_batch_growth_does_not_reclaim_when_minimum_fits() {
        let limit = JsonProcessingPolicyV1::WORK_BUDGET_BYTES;
        let slot = size_of::<RecordPayload>() + size_of::<CommittedRecord>();
        for spare in 0..64usize {
            let context = JsonOperationContext::standalone();
            let value = record(spare);
            let fields = super::super::record_workspace_bytes(&value);
            let available = fields + slot + crate::time::MAX_TIMESTAMP_BYTES + spare;
            let mut held = context.workspace_reservation();
            held.grow_to(limit - available, limit - available, "operation_scratch")
                .unwrap();
            let mut work = context.workspace_reservation();
            work.grow_to(fields, fields, "batch_prepare").unwrap();
            let mut batch = OwnedRecordBatch::new(&context)
                .reclaiming(|| panic!("preferred growth is optional"));
            batch.retain_workspace(work);
            batch.push(value).unwrap();
            let batch = batch.into_batch();
            assert_eq!(batch.records.capacity(), 1);
            assert_eq!(batch.work.reserved_bytes(), slot);
            drop(batch);
            context
                .workspace_reservation()
                .grow_to(available, available, "operation_scratch")
                .unwrap();
        }
    }

    #[test]
    fn encoding_retry_preserves_metadata_credit_and_input_identity() {
        use crate::json::{object_from_pairs, JsonInputKind, JsonValue};
        use crate::model::Document;

        for seed in 0..100 {
            for release in [0, 1, 8192] {
                let context = JsonOperationContext::standalone();
                let mut batch = OwnedRecordBatch::new(&context);
                let mut expected = 0;
                let mut pointers = Vec::new();
                let mut first_bytes = 0;
                for index in 0..2 {
                    let value = "x".repeat(100 + seed + index * 4096);
                    pointers.push(value.as_ptr());
                    let metadata = object_from_pairs([("payload", JsonValue::String(value))]);
                    let bytes = metadata
                        .retained_bytes_in(JsonInputKind::DocumentMetadata, &context)
                        .unwrap();
                    if index == 0 {
                        first_bytes = bytes;
                    }
                    expected += bytes;
                    batch
                        .push(RecordPayload::DocumentPut(Document {
                            document_id: format!("doc_{index}"),
                            collection_id: "col_1".into(),
                            external_id: format!("external_{index}"),
                            title: None,
                            source_uri: None,
                            metadata,
                            created_at: "1970-01-01T00:00:00Z".into(),
                            updated_at: "1970-01-01T00:00:00Z".into(),
                        }))
                        .unwrap();
                }
                let limit = JsonProcessingPolicyV1::WORK_BUDGET_BYTES;
                let mut optional = context.workspace_reservation();
                optional.grow_to(8192, 8192, "operation_scratch").unwrap();
                let mut held = context.workspace_reservation();
                let available = held.grow_to(0, limit, "operation_scratch").unwrap();
                held.retain(available - first_bytes - 512);
                let mut calls = 0;
                let result = batch.prepare_encoding(|| {
                    calls += 1;
                    optional.retain(8192 - release);
                    release != 0
                });
                assert_eq!(calls, 1);
                match result {
                    Ok(encoded) => {
                        assert_eq!(release, 8192);
                        assert_eq!(encoded.batch.payloads.json_bytes(), expected);
                        for (record, pointer) in encoded.batch.records.iter().zip(pointers) {
                            let RecordPayload::DocumentPut(document) = record else {
                                panic!("unexpected record");
                            };
                            assert_eq!(
                                document
                                    .metadata
                                    .get("payload")
                                    .unwrap()
                                    .as_str()
                                    .unwrap()
                                    .as_ptr(),
                                pointer
                            );
                        }
                    }
                    Err(error) => {
                        assert!(release < 8192);
                        assert!(crate::json::is_memory_budget_error(&error));
                        assert_eq!(error.code(), ErrorCode::InvalidJsonlSchema);
                    }
                }
                drop(optional);
                held.grow_to(limit, limit, "operation_scratch").unwrap();
            }
        }
    }

    #[test]
    fn generated_batch_fields_match_existing_formats_and_keep_exact_credit() {
        for seed in 0..100u64 {
            let context = JsonOperationContext::standalone();
            let limit = JsonProcessingPolicyV1::WORK_BUDGET_BYTES;
            let mut catalog = crate::catalog::Catalog::default();
            catalog
                .id_high_watermarks
                .insert("doc_".into(), u64::MAX - seed - 1);
            let id = format!("doc_{:016}", u64::MAX - seed);
            let text = format!("source text {seed}\n\u{4e2d}");
            let millis = seed.wrapping_mul(6364136223846793005);
            let timestamp = crate::time::format_unix_millis(millis);
            let hash = crate::model::chunk_text_hash(&text);
            let bytes = id.len() + text.len() + timestamp.len() + hash.len();
            let mut held = context.workspace_reservation();
            held.grow_to(limit - bytes, limit - bytes, "operation_scratch")
                .unwrap();
            let mut batch = OwnedRecordBatch::new(&context);
            let actual_id = batch.retain_string(catalog.next_id_in("doc_", 0, &context).unwrap());
            let actual_text = batch.copy_string(&text).unwrap();
            let actual_timestamp = batch.timestamp(millis).unwrap();
            let actual_hash = batch.text_hash(&text).unwrap();
            assert_eq!(
                (&actual_id, &actual_text, &actual_timestamp, &actual_hash),
                (&id, &text, &timestamp, &hash)
            );
            assert_eq!(batch.payloads.workspace_bytes(), bytes);
            assert_eq!(
                actual_id.capacity()
                    + actual_text.capacity()
                    + actual_timestamp.capacity()
                    + actual_hash.capacity(),
                bytes
            );
            let error = batch.copy_string("x").unwrap_err();
            assert_eq!(error.code(), ErrorCode::ResourceExhausted);
            assert_eq!(batch.payloads.workspace_bytes(), bytes);
            drop((actual_id, actual_text, actual_timestamp, actual_hash));
            drop(batch);
            held.grow_to(limit, limit, "operation_scratch").unwrap();
            // Exhaustion keeps its existing format error even when no memory is available.
            catalog.id_high_watermarks.insert("doc_".into(), u64::MAX);
            assert_eq!(
                catalog
                    .next_id_in("doc_", 0, &context)
                    .err()
                    .unwrap()
                    .code(),
                ErrorCode::FileFormatError
            );
        }
    }

    #[test]
    fn owned_batches_fit_minimum_capacity_and_keep_failed_growth_atomic() {
        let slot_bytes = size_of::<RecordPayload>() + size_of::<CommittedRecord>();
        let limit = JsonProcessingPolicyV1::WORK_BUDGET_BYTES;
        for count in 1..=100 {
            let context = JsonOperationContext::standalone();
            let available = count * slot_bytes + count + crate::time::MAX_TIMESTAMP_BYTES;
            let occupied = limit - available;
            let mut held = context.workspace_reservation();
            held.grow_to(occupied, occupied, "operation_scratch")
                .unwrap();
            let mut batch = OwnedRecordBatch::new(&context);
            for index in 0..count {
                batch.push(record(index)).unwrap();
                assert_eq!(
                    batch.work.reserved_bytes(),
                    batch.records.capacity() * slot_bytes
                );
                assert!(batch.records.capacity() <= count);
            }
            let error = batch.push(record(count)).unwrap_err();
            assert_eq!(error.code(), ErrorCode::ResourceExhausted);
            for (key, value) in [
                ("resource", "memory".to_string()),
                ("operation", "batch_prepare".to_string()),
                ("limit_bytes", limit.to_string()),
                (
                    "requested_bytes",
                    (occupied + (count + 1) * slot_bytes + crate::time::MAX_TIMESTAMP_BYTES)
                        .to_string(),
                ),
            ] {
                assert!(error
                    .detail()
                    .iter()
                    .any(|entry| entry == &(key.to_string(), value.clone())));
            }
            assert_eq!(batch.records.len(), count);
            for (index, actual) in batch.records.iter().enumerate() {
                let RecordPayload::CollectionDrop { collection_id, .. } = actual else {
                    panic!("unexpected record");
                };
                assert_eq!(collection_id, &format!("col_{}", index + 1));
            }
            // A failed optional growth has neither discarded the prefix nor leaked credit.
            held.retain(occupied - slot_bytes);
            batch.push(record(count)).unwrap();
            assert_eq!(batch.records.len(), count + 1);
            drop(batch);
            context
                .workspace_reservation()
                .grow_to(
                    available + slot_bytes,
                    available + slot_bytes,
                    "operation_scratch",
                )
                .unwrap();
        }
    }

    #[test]
    fn owned_batch_credit_tracks_live_input_and_output_arrays() {
        let limit = JsonProcessingPolicyV1::WORK_BUDGET_BYTES;
        for count in 1..=100 {
            let context = JsonOperationContext::standalone();
            let mut batch = OwnedRecordBatch::new(&context);
            for index in 0..count {
                batch.push(record(index)).unwrap();
            }
            let (records, mut work, _json, timestamp) = batch.into_parts();
            let expected = records.capacity() * size_of::<RecordPayload>()
                + records.len() * size_of::<CommittedRecord>();
            assert_eq!(work.reserved_bytes(), expected);
            let mut other = context.workspace_reservation();
            let available = limit - expected - crate::time::MAX_TIMESTAMP_BYTES;
            other
                .grow_to(available, available, "operation_scratch")
                .unwrap();
            assert_eq!(
                other
                    .grow_to(available + 1, available + 1, "operation_scratch")
                    .unwrap_err()
                    .code(),
                ErrorCode::ResourceExhausted
            );
            drop(records);
            let output = count * size_of::<CommittedRecord>();
            work.retain(output);
            let available = limit - output - crate::time::MAX_TIMESTAMP_BYTES;
            other
                .grow_to(available, available, "operation_scratch")
                .unwrap();
            drop(work);
            drop(timestamp);
            other.grow_to(limit, limit, "operation_scratch").unwrap();
        }
    }

    #[test]
    fn owned_batch_commit_transfers_array_credit_and_releases_on_storage_errors() {
        use crate::profile::Durability;
        use crate::storage::Storage;

        let path = std::env::temp_dir().join(format!(
            "cofferdb-owned-array-commit-{}.coffer",
            std::process::id()
        ));
        let mut storage = Storage::create(&path).unwrap();
        let limit = JsonProcessingPolicyV1::WORK_BUDGET_BYTES;
        let slot = size_of::<RecordPayload>() + size_of::<CommittedRecord>();
        for count in 1..=100 {
            let context = JsonOperationContext::standalone();
            let mut held = context.workspace_reservation();
            let payloads = (0..count).map(record).collect::<Vec<_>>();
            let payload_bytes = payloads
                .iter()
                .map(super::super::record_workspace_bytes)
                .sum::<usize>();
            let available = count * slot + payload_bytes + crate::time::MAX_TIMESTAMP_BYTES;
            held.grow_to(limit - available, limit - available, "operation_scratch")
                .unwrap();
            let mut batch = OwnedRecordBatch::new(&context);
            let mut payload_work = context.workspace_reservation();
            payload_work
                .grow_to(payload_bytes, payload_bytes, "batch_prepare")
                .unwrap();
            batch.retain_workspace(payload_work);
            for payload in payloads {
                batch.push(payload).unwrap();
            }
            let (append, work) = storage
                .append_owned_batch_in(
                    batch.prepare_encoding(|| false).unwrap(),
                    Durability::Fast,
                    || false,
                )
                .unwrap();
            assert_eq!(append.records.len(), count);
            assert_eq!(append.commit.batch_id, count as u64);
            assert_eq!(
                work.workspace_bytes(),
                count * size_of::<CommittedRecord>() + payload_bytes
            );
            let spare = available - work.workspace_bytes() - crate::time::MAX_TIMESTAMP_BYTES;
            let mut other = context.workspace_reservation();
            other.grow_to(spare, spare, "operation_scratch").unwrap();
            assert!(other
                .grow_to(spare + 1, spare + 1, "operation_scratch")
                .is_err());
            drop(append);
            drop(work);
            other
                .grow_to(available, available, "operation_scratch")
                .unwrap();
            drop(other);
            let before = std::fs::read(&path).unwrap();
            let mut batch = OwnedRecordBatch::new(&context);
            let payload = record(0);
            let bytes = super::super::record_workspace_bytes(&payload);
            let mut payload_work = context.workspace_reservation();
            payload_work.grow_to(bytes, bytes, "batch_prepare").unwrap();
            batch.retain_workspace(payload_work);
            batch.push(payload).unwrap();
            storage.write_failed = true;
            let error = storage
                .append_owned_batch_in(
                    batch.prepare_encoding(|| false).unwrap(),
                    Durability::Fast,
                    || false,
                )
                .err()
                .unwrap();
            storage.write_failed = false;
            assert_eq!(error.code(), ErrorCode::RecoveryError);
            assert_eq!(std::fs::read(&path).unwrap(), before);
            context
                .workspace_reservation()
                .grow_to(available, available, "operation_scratch")
                .unwrap();
        }
        drop(storage);
        let (storage, report) = Storage::open_with_records(&path, true, 0, |_| Ok(())).unwrap();
        assert_eq!(report.last_valid_commit, 100);
        drop(storage);
        std::fs::remove_file(path).unwrap();
    }
}
