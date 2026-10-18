#[cfg(test)]
use std::fs::File;
use std::io::{BufWriter, Write};

use crate::hash::Sha256Writer;
use crate::json::{
    is_memory_budget_error, write_validated_jsonl_fields, JsonFieldValue, JsonOperationContext,
    WorkspaceReservation,
};
use crate::temporary::{TempFileGuard, TemporaryFile};
use crate::{Error, Result};

pub(super) struct ExportWriter<'a, W: Write = TemporaryFile> {
    output: Option<BufWriter<Sha256Writer<W>>>,
    buffer_work: WorkspaceReservation,
    temporary: &'a mut TempFileGuard,
    budget_error: Option<Error>,
    disk_credit: u64,
}

impl<'a, W: Write> ExportWriter<'a, W> {
    pub(super) fn new_in(
        file: W,
        temporary: &'a mut TempFileGuard,
        preferred_buffer_bytes: usize,
        context: &JsonOperationContext,
    ) -> Result<Self> {
        let mut buffer_work = context.workspace_reservation();
        let capacity = buffer_work.grow_to(0, preferred_buffer_bytes, "operation_scratch")?;
        Ok(Self {
            output: Some(BufWriter::with_capacity(capacity, Sha256Writer::new(file))),
            buffer_work,
            temporary,
            budget_error: None,
            disk_credit: 0,
        })
    }

    pub(super) fn write_fields<const N: usize>(
        &mut self,
        fields: [(&str, JsonFieldValue<'_>); N],
        context: &JsonOperationContext,
    ) -> Result<()> {
        let result = write_validated_jsonl_fields(self, fields, context);
        self.checked(result)
    }

    pub(super) fn checked<T>(&mut self, result: Result<T>) -> Result<T> {
        // Return unused preferred capacity at row boundaries so other files can use it.
        self.temporary.release_disk(self.disk_credit);
        self.disk_credit = 0;
        match self.budget_error.take() {
            Some(error) => Err(error),
            None => result,
        }
    }

    pub(super) fn finish(mut self) -> Result<String> {
        self.checked(Ok(()))?;
        Ok(self.take_unbuffered()?.finish()?)
    }

    pub(super) fn disable_buffering(&mut self) -> Result<bool> {
        if self.output.as_ref().unwrap().capacity() == 0 {
            return Ok(false);
        }
        let output = self.take_unbuffered()?;
        self.output = Some(BufWriter::with_capacity(0, output));
        Ok(true)
    }

    fn take_unbuffered(&mut self) -> Result<Sha256Writer<W>> {
        match self.output.take().unwrap().into_inner() {
            Ok(output) => {
                self.buffer_work.retain(0);
                Ok(output)
            }
            Err(error) => {
                let (error, buffer) = error.into_parts();
                // Destroy pending bytes without a second write from BufWriter::drop.
                drop(buffer.into_parts());
                self.buffer_work.retain(0);
                Err(error.into())
            }
        }
    }
}

impl<W: Write> Drop for ExportWriter<'_, W> {
    fn drop(&mut self) {
        if let Some(output) = self.output.take() {
            drop(output.into_parts());
        }
        self.buffer_work.retain(0);
        self.temporary.release_disk(self.disk_credit);
        self.disk_credit = 0;
    }
}

// Current callers only retry local preparation or JSON admission before its
// first output byte, never a row already written or an external iterator.
pub(super) fn retry_after_buffer_reclaim<W: Write, T>(
    writers: &mut [&mut ExportWriter<'_, W>],
    reclaim: &mut dyn FnMut() -> bool,
    mut operation: impl FnMut(&mut [&mut ExportWriter<'_, W>]) -> Result<T>,
) -> Result<T> {
    match operation(writers) {
        Err(error) if is_memory_budget_error(&error) => {
            let mut released = false;
            for writer in writers.iter_mut() {
                released |= writer.disable_buffering()?;
            }
            released |= reclaim();
            if released {
                operation(writers)
            } else {
                Err(error)
            }
        }
        result => result,
    }
}

impl<W: Write> Write for ExportWriter<'_, W> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        let requested = bytes.len() as u64;
        if requested > self.disk_credit {
            let minimum = requested - self.disk_credit;
            match self
                .temporary
                .reserve_disk_buffer(minimum, self.output.as_ref().unwrap().capacity() as u64)
            {
                Ok(granted) => self.disk_credit += granted,
                Err(error) => {
                    self.budget_error = Some(error);
                    return Err(std::io::Error::other("export staging budget exhausted"));
                }
            }
        }
        match self.output.as_mut().unwrap().write(bytes) {
            Ok(written) => {
                self.disk_credit -= written as u64;
                Ok(written)
            }
            Err(error) => {
                // A buffered write can fail after partially writing to the file. Keep
                // its reservation until cleanup rather than guessing the physical length.
                self.disk_credit = 0;
                Err(error)
            }
        }
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.output.as_mut().unwrap().flush()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::temporary::{create_budgeted_temporary_file_in, TemporaryDiskBudget};
    use crate::{ErrorCode, JsonValue};
    use std::sync::Arc;
    use std::{cell::RefCell, rc::Rc};

    struct OutputState {
        bytes: Vec<u8>,
        writes: usize,
        flushes: usize,
        short_first: bool,
        fail_write: Option<usize>,
        fail_flush: bool,
    }

    struct OutputProbe(Rc<RefCell<OutputState>>);

    impl OutputProbe {
        fn new() -> (Self, Rc<RefCell<OutputState>>) {
            let state = Rc::new(RefCell::new(OutputState {
                bytes: Vec::with_capacity(4096),
                writes: 0,
                flushes: 0,
                short_first: false,
                fail_write: None,
                fail_flush: false,
            }));
            (Self(state.clone()), state)
        }
    }

    impl Write for OutputProbe {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            let mut state = self.0.borrow_mut();
            state.writes += 1;
            if state.fail_write == Some(state.writes) {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::PermissionDenied,
                    "injected export write failure",
                ));
            }
            let count = if state.short_first && state.writes == 1 {
                bytes.len().min(2)
            } else {
                bytes.len()
            };
            state.bytes.extend_from_slice(&bytes[..count]);
            Ok(count)
        }

        fn flush(&mut self) -> std::io::Result<()> {
            let mut state = self.0.borrow_mut();
            state.flushes += 1;
            if state.fail_flush {
                Err(std::io::Error::other("injected export flush failure"))
            } else {
                Ok(())
            }
        }
    }

    fn context_with_limit(limit: usize, parent: bool) -> JsonOperationContext {
        use crate::json::WorkLedger;
        JsonOperationContext::new(Arc::new(WorkLedger::with_limits(
            if parent { usize::MAX } else { limit },
            usize::MAX,
            if parent { limit } else { usize::MAX },
        )))
    }

    #[test]
    fn optional_buffer_allocates_only_granted_bytes_and_keeps_original_credit() {
        let directory = std::env::temp_dir();
        for parent in [false, true] {
            for available in [0, 1, 13, 8191, 8192, 8193] {
                let limit = available + 7;
                let context = context_with_limit(limit, parent);
                let mut held = context.workspace_reservation();
                held.grow_to(7, 7, "operation_scratch").unwrap();
                let disk = Arc::new(TemporaryDiskBudget::new(4096));
                let (file, mut guard) = create_budgeted_temporary_file_in(
                    &directory,
                    "export-allocation",
                    "tmp",
                    disk.clone(),
                )
                .unwrap();
                let (result, allocated) = crate::test_alloc::allocation_bytes(|| {
                    ExportWriter::new_in(file, &mut guard, 8192, &context)
                });
                let writer = result.unwrap();
                let granted = available.min(8192);
                assert_eq!(allocated, granted as u128);
                assert_eq!(writer.output.as_ref().unwrap().capacity(), granted);
                assert_eq!(writer.buffer_work.reserved_bytes(), granted);
                assert_eq!(context.available_workspace_bytes(), available - granted);
                let mut identity = writer.buffer_work.context().workspace_reservation();
                identity.merge(context.workspace_reservation());
                drop(identity);
                drop(context);
                held.grow_to(limit - granted, limit - granted, "operation_scratch")
                    .unwrap();
                assert_eq!(
                    held.grow_to(
                        limit - granted + 1,
                        limit - granted + 1,
                        "operation_scratch"
                    )
                    .unwrap_err()
                    .code(),
                    ErrorCode::ResourceExhausted
                );
                drop(writer);
                held.grow_to(limit, limit, "operation_scratch").unwrap();
                drop(guard);
                assert_eq!(disk.used(), 0);
            }
        }
    }

    #[test]
    fn buffer_transition_preserves_hash_and_failures_never_replay_drop_io() {
        let directory = std::env::temp_dir();
        for mode in 0..8 {
            let context = context_with_limit(64, false);
            let disk = Arc::new(TemporaryDiskBudget::new(4096));
            let (_file, mut guard) = create_budgeted_temporary_file_in(
                &directory,
                "export-transition",
                "tmp",
                disk.clone(),
            )
            .unwrap();
            let (probe, state) = OutputProbe::new();
            if mode == 3 || mode == 4 {
                let mut state = state.borrow_mut();
                state.short_first = true;
                state.fail_write = Some(2);
            } else if mode == 5 {
                state.borrow_mut().fail_flush = true;
            } else if mode == 6 {
                state.borrow_mut().fail_write = Some(1);
            }
            let mut writer =
                ExportWriter::new_in(probe, &mut guard, if mode == 6 { 0 } else { 64 }, &context)
                    .unwrap();
            let row = b"{\"old\":true}\n";
            let written = writer.write_all(row).map_err(Error::from);
            if mode == 6 {
                let error = writer.checked(written).unwrap_err();
                assert_eq!(error.code(), ErrorCode::PermissionDenied);
                assert_eq!(error.message(), "injected export write failure");
                drop(writer);
            } else {
                writer.checked(written).unwrap();
                assert_eq!(state.borrow().writes, 0);
                match mode {
                    0 => drop(writer),
                    1 => {
                        assert_eq!(writer.finish().unwrap(), crate::hash::sha256_hex(row));
                    }
                    2 | 7 => {
                        assert!(writer.disable_buffering().unwrap());
                        assert!(!writer.disable_buffering().unwrap());
                        assert_eq!(context.available_workspace_bytes(), 64);
                        assert_eq!(state.borrow().flushes, 0);
                        if mode == 2 {
                            writer.write_all(b"new\n").unwrap();
                            writer.checked(Ok(())).unwrap();
                            assert_eq!(
                                writer.finish().unwrap(),
                                crate::hash::sha256_hex(b"{\"old\":true}\nnew\n")
                            );
                        } else {
                            drop(writer);
                        }
                    }
                    3 => {
                        let error = writer.disable_buffering().unwrap_err();
                        assert_eq!(error.code(), ErrorCode::PermissionDenied);
                        assert_eq!(error.message(), "injected export write failure");
                        drop(writer);
                    }
                    4 => {
                        let error = writer.finish().unwrap_err();
                        assert_eq!(error.code(), ErrorCode::PermissionDenied);
                        assert_eq!(error.message(), "injected export write failure");
                    }
                    5 => {
                        let error = writer.finish().unwrap_err();
                        assert_eq!(error.code(), ErrorCode::IoError);
                        assert_eq!(error.message(), "injected export flush failure");
                    }
                    _ => unreachable!(),
                }
            }
            let observed = state.borrow();
            assert_eq!(
                observed.writes,
                match mode {
                    0 => 0,
                    2..=4 => 2,
                    _ => 1,
                }
            );
            assert_eq!(observed.flushes, usize::from(matches!(mode, 1 | 2 | 5)));
            assert_eq!(
                observed.bytes.as_slice(),
                match mode {
                    0 | 6 => &[][..],
                    2 => b"{\"old\":true}\nnew\n",
                    3 | 4 => &row[..2],
                    _ => row,
                }
            );
            drop(observed);
            let mut held = context.workspace_reservation();
            held.grow_to(64, 64, "operation_scratch").unwrap();
            assert!(held.grow_to(65, 65, "operation_scratch").is_err());
            drop(_file);
            drop(guard);
            assert_eq!(disk.used(), 0);
        }
    }

    #[test]
    fn json_prewrite_admission_yields_all_buffers_without_duplicate_rows_or_hashes() {
        let directory = std::env::temp_dir();
        let minimum = (crate::json::MAX_JSON_DEPTH + 1) * crate::json::test_write_frame_bytes();
        for parent in [false, true] {
            for gap in [-1isize, 0, 1] {
                for preferred in [0, 64] {
                    let limit = minimum + 3 * 64 + 7;
                    let context = context_with_limit(limit, parent);
                    let disk = Arc::new(TemporaryDiskBudget::new(4096));
                    let (f0, mut g0) = create_budgeted_temporary_file_in(
                        &directory,
                        "export-yield0",
                        "tmp",
                        disk.clone(),
                    )
                    .unwrap();
                    let (f1, mut g1) = create_budgeted_temporary_file_in(
                        &directory,
                        "export-yield1",
                        "tmp",
                        disk.clone(),
                    )
                    .unwrap();
                    let (f2, mut g2) = create_budgeted_temporary_file_in(
                        &directory,
                        "export-yield2",
                        "tmp",
                        disk.clone(),
                    )
                    .unwrap();
                    let (p0, s0) = OutputProbe::new();
                    let (p1, s1) = OutputProbe::new();
                    let (p2, s2) = OutputProbe::new();
                    let mut w0 = ExportWriter::new_in(p0, &mut g0, preferred, &context).unwrap();
                    let mut w1 = ExportWriter::new_in(p1, &mut g1, preferred, &context).unwrap();
                    let mut w2 = ExportWriter::new_in(p2, &mut g2, preferred, &context).unwrap();
                    let prior = b"{\"old\":true}\n";
                    for writer in [&mut w0, &mut w1, &mut w2] {
                        writer.write_all(prior).unwrap();
                        writer.checked(Ok(())).unwrap();
                    }
                    let available = minimum.checked_add_signed(gap).unwrap();
                    let mut held = context.workspace_reservation();
                    held.grow_to(limit - available, limit - available, "operation_scratch")
                        .unwrap();
                    let mut attempts = 0;
                    let mut reclaims = 0;
                    let result = retry_after_buffer_reclaim(
                        &mut [&mut w0, &mut w1, &mut w2],
                        &mut || {
                            reclaims += 1;
                            false
                        },
                        |writers| {
                            attempts += 1;
                            writers[1].write_fields(
                                [("value", JsonFieldValue::Json(&JsonValue::Null))],
                                &context,
                            )
                        },
                    );
                    let failed = gap < 0;
                    assert_eq!(result.is_err(), failed);
                    if let Err(error) = result {
                        assert_eq!(error.code(), ErrorCode::RecoveryError);
                        assert!(is_memory_budget_error(&error));
                    }
                    assert_eq!(attempts, if preferred != 0 { 2 } else { 1 });
                    assert_eq!(reclaims, usize::from(preferred != 0 || failed));
                    assert_eq!(context.available_workspace_bytes(), available);
                    let expected = if failed {
                        prior.to_vec()
                    } else {
                        [prior.as_slice(), b"{\"value\":null}\n"].concat()
                    };
                    assert_eq!(w0.finish().unwrap(), crate::hash::sha256_hex(prior));
                    assert_eq!(w1.finish().unwrap(), crate::hash::sha256_hex(&expected));
                    assert_eq!(w2.finish().unwrap(), crate::hash::sha256_hex(prior));
                    assert_eq!(s0.borrow().bytes, prior);
                    assert_eq!(s1.borrow().bytes, expected);
                    assert_eq!(s2.borrow().bytes, prior);
                    for state in [&s0, &s1, &s2] {
                        assert_eq!(state.borrow().flushes, 1);
                    }
                    held.grow_to(limit, limit, "operation_scratch").unwrap();
                    assert!(held
                        .grow_to(limit + 1, limit + 1, "operation_scratch")
                        .is_err());
                    drop((f0, f1, f2));
                    drop((g0, g1, g2));
                    assert_eq!(disk.used(), 0);
                }
            }
        }
    }

    #[test]
    fn reclaim_retries_only_memory_and_preserves_flush_io_priority() {
        let directory = std::env::temp_dir();
        for mode in 0..6 {
            let context = context_with_limit(64, false);
            let disk = Arc::new(TemporaryDiskBudget::new(4096));
            let (file, mut guard) = create_budgeted_temporary_file_in(
                &directory,
                "export-priority",
                "tmp",
                disk.clone(),
            )
            .unwrap();
            let (probe, state) = OutputProbe::new();
            if mode == 5 {
                state.borrow_mut().short_first = true;
                state.borrow_mut().fail_write = Some(2);
            }
            let mut writer = ExportWriter::new_in(probe, &mut guard, 64, &context).unwrap();
            writer.write_all(b"pending\n").unwrap();
            writer.checked(Ok(())).unwrap();
            let original = || match mode {
                0 | 5 => Error::resource_exhausted("memory", "operation_scratch", 64, 65),
                1 => Error::resource_exhausted("temporary_disk", "export_stage", 64, 65),
                2 => Error::resource_exhausted("catalog", "catalog_build", 64, 65),
                3 => crate::json::JsonInputKind::StoredJson.limit_error(
                    crate::json::JsonLimitReason::Depth,
                    128,
                    Some(129),
                ),
                _ => Error::new(ErrorCode::FileFormatError, "payload CRC failure"),
            };
            let expected = original().to_json();
            let mut attempts = 0;
            let mut reclaims = 0;
            let error = retry_after_buffer_reclaim(
                &mut [&mut writer],
                &mut || {
                    reclaims += 1;
                    true
                },
                |_| {
                    attempts += 1;
                    Err::<(), _>(original())
                },
            )
            .unwrap_err();
            if mode == 5 {
                assert_eq!(error.code(), ErrorCode::PermissionDenied);
                assert_eq!(error.message(), "injected export write failure");
                assert_eq!(attempts, 1);
                assert_eq!(reclaims, 0);
                assert_eq!(state.borrow().writes, 2);
                assert_eq!(state.borrow().bytes, b"pe");
            } else {
                assert_eq!(error.to_json(), expected);
                assert_eq!(attempts, if mode == 0 { 2 } else { 1 });
                assert_eq!(reclaims, usize::from(mode == 0));
                assert_eq!(state.borrow().writes, usize::from(mode == 0));
            }
            let writes = state.borrow().writes;
            drop(writer);
            assert_eq!(state.borrow().writes, writes);
            assert_eq!(state.borrow().flushes, 0);
            assert_eq!(context.available_workspace_bytes(), 64);
            drop(file);
            drop(guard);
            assert_eq!(disk.used(), 0);
        }
    }

    #[test]
    fn zero_buffer_can_reclaim_cache_for_one_local_preparation_retry() {
        let directory = std::env::temp_dir();
        for released in [false, true] {
            let context = context_with_limit(1, true);
            let mut cached = context.cache_reservation(1, 1);
            let disk = Arc::new(TemporaryDiskBudget::new(0));
            let (file, mut guard) = create_budgeted_temporary_file_in(
                &directory,
                "export-cache-reclaim",
                "tmp",
                disk.clone(),
            )
            .unwrap();
            let mut writer = ExportWriter::new_in(file, &mut guard, 8192, &context).unwrap();
            assert_eq!(writer.output.as_ref().unwrap().capacity(), 0);
            let mut attempts = 0;
            let mut reclaims = 0;
            let result = retry_after_buffer_reclaim(
                &mut [&mut writer],
                &mut || {
                    reclaims += 1;
                    if released {
                        drop(cached.take());
                    }
                    released
                },
                |_| {
                    attempts += 1;
                    let mut work = context.workspace_reservation();
                    work.grow_to(1, 1, "operation_scratch")?;
                    Ok(work)
                },
            );
            assert_eq!(attempts, if released { 2 } else { 1 });
            assert_eq!(reclaims, 1);
            assert_eq!(result.is_ok(), released);
            assert_eq!(context.available_workspace_bytes(), 0);
            drop(result);
            drop(cached);
            drop(writer);
            assert_eq!(context.available_workspace_bytes(), 1);
            drop(guard);
            assert_eq!(disk.used(), 0);
        }
    }

    #[test]
    fn retained_manifest_prewrite_retry_keeps_timestamp_bytes_and_exact_credit() {
        use crate::json::{object_from_pairs, JsonInputKind};
        let expected = b"{\n  \"counts\": {\n    \"chunks\": 1\n  },\n  \"exported_at\": \"1970-01-01T00:00:00Z\",\n  \"tokenizer\": {\n    \"stop_words\": [\n      \"the\"\n    ]\n  }\n}";
        let node =
            11 * std::mem::size_of::<(String, JsonValue)>() + 16 * std::mem::size_of::<usize>();
        let retained_bytes = 3 * node + std::mem::size_of::<JsonValue>() + 65;
        let minimum = retained_bytes + 4 * crate::json::test_write_frame_bytes();
        for parent in [false, true] {
            for gap in [-1isize, 0, 1] {
                let value = object_from_pairs([
                    (
                        "counts",
                        object_from_pairs([("chunks", JsonValue::Number(1.0))]),
                    ),
                    (
                        "exported_at",
                        JsonValue::String("1970-01-01T00:00:00Z".into()),
                    ),
                    (
                        "tokenizer",
                        object_from_pairs([(
                            "stop_words",
                            JsonValue::Array(vec![JsonValue::String("the".into())]),
                        )]),
                    ),
                ]);
                let limit = minimum + 64 + 7;
                let context = context_with_limit(limit, parent);
                let mut retained = context.reservation();
                retained
                    .grow(retained_bytes, JsonInputKind::StoredJson)
                    .unwrap();
                let disk = Arc::new(TemporaryDiskBudget::new(4096));
                let (file, mut guard) = create_budgeted_temporary_file_in(
                    &std::env::temp_dir(),
                    "export-manifest-yield",
                    "tmp",
                    disk.clone(),
                )
                .unwrap();
                let (probe, state) = OutputProbe::new();
                let mut writer = ExportWriter::new_in(probe, &mut guard, 64, &context).unwrap();
                let available = minimum.checked_add_signed(gap).unwrap();
                let mut held = context.workspace_reservation();
                held.grow_to(limit - available, limit - available, "operation_scratch")
                    .unwrap();
                let mut attempts = 0;
                let mut reclaims = 0;
                let result = retry_after_buffer_reclaim(
                    &mut [&mut writer],
                    &mut || {
                        reclaims += 1;
                        false
                    },
                    |writers| {
                        attempts += 1;
                        let written = value.write_value_retained_in(
                            writers[0],
                            JsonInputKind::StoredJson,
                            &context,
                            &mut retained,
                            true,
                        );
                        writers[0].checked(written)
                    },
                );
                assert_eq!(attempts, 2);
                assert_eq!(reclaims, 1);
                assert_eq!(retained.reserved_bytes(), retained_bytes);
                assert_eq!(
                    context.available_workspace_bytes(),
                    available - retained_bytes
                );
                if gap < 0 {
                    let error = result.unwrap_err();
                    assert_eq!(error.code(), ErrorCode::RecoveryError);
                    assert!(is_memory_budget_error(&error));
                    assert_eq!(state.borrow().writes, 0);
                    assert_eq!(writer.finish().unwrap(), crate::hash::sha256_hex(b""));
                    assert!(state.borrow().bytes.is_empty());
                } else {
                    result.unwrap();
                    assert_eq!(writer.finish().unwrap(), crate::hash::sha256_hex(expected));
                    assert_eq!(state.borrow().bytes, expected);
                }
                assert_eq!(state.borrow().flushes, 1);
                drop(value);
                drop(retained);
                held.grow_to(limit, limit, "operation_scratch").unwrap();
                assert!(held
                    .grow_to(limit + 1, limit + 1, "operation_scratch")
                    .is_err());
                drop(file);
                drop(guard);
                assert_eq!(disk.used(), 0);
            }
        }
    }

    #[test]
    fn row_boundaries_return_unused_credit_and_hash_only_complete_bytes() {
        let directory = std::env::temp_dir();
        for seed in 0..100usize {
            let row = format!("{{\"text\":\"{}\"}}\n", "x".repeat(seed));
            let budget = Arc::new(TemporaryDiskBudget::new(2 * row.len() as u64));
            let mut guards = Vec::new();
            for count in 1..=2 {
                let (file, mut guard) = create_budgeted_temporary_file_in(
                    &directory,
                    "export-budget",
                    "tmp",
                    budget.clone(),
                )
                .unwrap();
                let mut output = ExportWriter::new_in(
                    file,
                    &mut guard,
                    8192,
                    &JsonOperationContext::standalone(),
                )
                .unwrap();
                output
                    .write_fields(
                        [("text", JsonFieldValue::String(Some(&"x".repeat(seed))))],
                        &JsonOperationContext::standalone(),
                    )
                    .unwrap();
                assert_eq!(budget.used(), count * row.len() as u64);
                let hash = output.finish().unwrap();
                assert_eq!(hash, crate::hash::sha256_hex(row.as_bytes()));
                assert_eq!(std::fs::read(guard.path()).unwrap(), row.as_bytes());
                guards.push(guard);
            }
            drop(guards);
            assert_eq!(budget.used(), 0);
        }
    }

    #[test]
    fn export_budget_errors_preserve_the_contract_and_json_and_io_errors_keep_their_codes() {
        let directory = std::env::temp_dir();
        for limit in 0..32 {
            let budget = Arc::new(TemporaryDiskBudget::new(limit));
            let (file, mut guard) = create_budgeted_temporary_file_in(
                &directory,
                "export-reject",
                "tmp",
                budget.clone(),
            )
            .unwrap();
            let mut output =
                ExportWriter::new_in(file, &mut guard, 8192, &JsonOperationContext::standalone())
                    .unwrap();
            let error = output
                .write_fields(
                    [(
                        "text",
                        JsonFieldValue::String(Some("a sufficiently long export value")),
                    )],
                    &JsonOperationContext::standalone(),
                )
                .unwrap_err();
            assert_eq!(error.code(), ErrorCode::ResourceExhausted);
            assert!(budget.used() <= limit);
            drop(output);
            drop(guard);
            assert_eq!(budget.used(), 0);
        }
        let budget = Arc::new(TemporaryDiskBudget::new(0));
        let (file, mut guard) =
            create_budgeted_temporary_file_in(&directory, "export-json", "tmp", budget.clone())
                .unwrap();
        let mut output =
            ExportWriter::new_in(file, &mut guard, 8192, &JsonOperationContext::standalone())
                .unwrap();
        let error = output
            .write_fields(
                [("vector", JsonFieldValue::Vector(&[f32::NAN]))],
                &JsonOperationContext::standalone(),
            )
            .unwrap_err();
        assert_ne!(error.code(), ErrorCode::ResourceExhausted);
        assert_eq!(budget.used(), 0);
        drop(output);
        drop(guard);

        let budget = Arc::new(TemporaryDiskBudget::new(4096));
        let (mut file, mut guard) =
            create_budgeted_temporary_file_in(&directory, "export-io", "tmp", budget.clone())
                .unwrap();
        *file = File::open(guard.path()).unwrap();
        let mut output =
            ExportWriter::new_in(file, &mut guard, 8192, &JsonOperationContext::standalone())
                .unwrap();
        output
            .write_fields(
                [("value", JsonFieldValue::Json(&JsonValue::Null))],
                &JsonOperationContext::standalone(),
            )
            .unwrap();
        let error = output.finish().unwrap_err();
        assert_eq!(error.code(), ErrorCode::IoError);
        assert!(budget.used() > 0);
        drop(guard);
        assert_eq!(budget.used(), 0);
    }
}
