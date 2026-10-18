use super::{
    allocation_error, check_key_capacity, invalid_run, Entry, Run, RunCursor, RunWriter, IO_BYTES,
};
use crate::json::{is_memory_budget_error, JsonOperationContext, WorkspaceReservation};
use crate::temporary::{create_budgeted_temporary_file_in_context, TemporaryDiskBudget};
use crate::{Error, ErrorCode, Result};
use std::cmp::Ordering;
use std::path::Path;
use std::sync::Arc;

const PREFERRED_BATCH_BYTES: usize = 64 * 1024;

// Unlike reference validation, summation needs neither lookup pages nor a fixed
// workspace floor. Runs share the existing codec, file owners and disk ledger.
pub(crate) struct ExternalTermFrequencies<'a> {
    directory: &'a Path,
    disk: Arc<TemporaryDiskBudget>,
    entries: Vec<Entry>,
    key_bytes: usize,
    levels: [Option<Run>; 64],
    failed: bool,
    work: WorkspaceReservation,
}

impl<'a> ExternalTermFrequencies<'a> {
    pub(crate) fn new(
        directory: &'a Path,
        disk: Arc<TemporaryDiskBudget>,
        context: &JsonOperationContext,
    ) -> Self {
        Self {
            directory,
            disk,
            entries: Vec::new(),
            key_bytes: 0,
            levels: std::array::from_fn(|_| None),
            failed: false,
            work: context.workspace_reservation(),
        }
    }

    pub(crate) fn push(&mut self, parts: &[&[u8]], value: u64) -> Result<()> {
        self.check_healthy()?;
        self.failed = true;
        let length = parts.iter().try_fold(0usize, |length, part| {
            length.checked_add(part.len()).ok_or_else(invalid_run)
        })?;
        u32::try_from(length).map_err(|_| invalid_run())?;
        if let Some(entry) = self
            .entries
            .last_mut()
            .filter(|entry| key_matches(&entry.key, parts))
        {
            entry.value = sum(entry.value, value)?;
        } else {
            if !self.entries.is_empty()
                && self.batch_bytes_with(length) > PREFERRED_BATCH_BYTES as u128
            {
                self.combine_batch()?;
                if let Some(entry) = self
                    .entries
                    .iter_mut()
                    .find(|entry| key_matches(&entry.key, parts))
                {
                    entry.value = sum(entry.value, value)?;
                    self.failed = false;
                    return Ok(());
                }
                if self.batch_bytes_with(length) > PREFERRED_BATCH_BYTES as u128 {
                    self.spill_inner()?;
                }
            }
            if let Err(error) = self.insert(parts, length, value) {
                if !is_memory_budget_error(&error) {
                    return Err(error);
                }
                if self.entries.is_empty() {
                    if !self.release_empty_capacity() {
                        return Err(error);
                    }
                    self.insert(parts, length, value)?;
                } else {
                    self.combine_batch()?;
                    if let Some(entry) = self
                        .entries
                        .iter_mut()
                        .find(|entry| key_matches(&entry.key, parts))
                    {
                        entry.value = sum(entry.value, value)?;
                    } else if let Err(error) = self.insert(parts, length, value) {
                        if !is_memory_budget_error(&error) {
                            return Err(error);
                        }
                        self.spill_inner()?;
                        self.insert(parts, length, value)?;
                    }
                }
            }
        }
        self.failed = false;
        Ok(())
    }

    fn batch_bytes_with(&self, length: usize) -> u128 {
        self.key_bytes as u128
            + length as u128
            + self.entries.capacity().max(self.entries.len() + 1) as u128
                * std::mem::size_of::<Entry>() as u128
    }

    fn insert(&mut self, parts: &[&[u8]], length: usize, value: u64) -> Result<()> {
        let slot = std::mem::size_of::<Entry>();
        let old = self.entries.capacity();
        let growing = self.entries.len() == old;
        let minimum_slots = if growing { self.entries.len() + 1 } else { 0 };
        let preferred_slots = if growing {
            old.saturating_mul(2)
                .max(4)
                .min((PREFERRED_BATCH_BYTES / slot).max(minimum_slots))
        } else {
            0
        };
        // During array growth old and new backing allocations coexist.
        let old_bytes = old as u128 * slot as u128;
        let overlap = if growing {
            old_bytes.max(length as u128)
        } else {
            old_bytes + length as u128
        };
        let base = self.key_bytes as u128 + overlap;
        let granted = self.work.grow_to_wide(
            base + minimum_slots as u128 * slot as u128,
            base + preferred_slots as u128 * slot as u128,
            "operation_scratch",
        )?;
        if growing {
            let capacity = ((granted as u128 - base) / slot as u128) as usize;
            let mut replacement = Vec::new();
            replacement
                .try_reserve_exact(capacity)
                .map_err(allocation_error)?;
            if replacement.capacity() != capacity {
                return Err(std::io::Error::other("sort batch exceeds admitted capacity").into());
            }
            replacement.append(&mut self.entries);
            self.entries = replacement;
        }
        let mut key = Vec::new();
        key.try_reserve_exact(length).map_err(allocation_error)?;
        check_key_capacity(&key, length)?;
        for part in parts {
            key.extend_from_slice(part);
        }
        self.key_bytes += key.capacity();
        self.entries.push(Entry {
            key,
            line: 0,
            value,
        });
        self.retain_batch();
        Ok(())
    }

    fn retain_batch(&mut self) {
        self.work
            .retain(self.key_bytes + self.entries.capacity() * std::mem::size_of::<Entry>());
    }

    pub(crate) fn release_empty_batch_capacity(&mut self) -> Result<bool> {
        self.check_healthy()?;
        Ok(self.release_empty_capacity())
    }

    fn release_empty_capacity(&mut self) -> bool {
        if !self.entries.is_empty() || self.entries.capacity() == 0 {
            return false;
        }
        self.entries = Vec::new();
        self.retain_batch();
        true
    }

    fn combine_batch(&mut self) -> Result<()> {
        self.entries
            .sort_unstable_by(|left, right| left.key.cmp(&right.key));
        let mut count = 0;
        for read in 0..self.entries.len() {
            if count != 0 && self.entries[count - 1].key == self.entries[read].key {
                self.entries[count - 1].value =
                    sum(self.entries[count - 1].value, self.entries[read].value)?;
            } else {
                self.entries.swap(count, read);
                count += 1;
            }
        }
        self.entries.truncate(count);
        self.key_bytes = self.entries.iter().map(|entry| entry.key.capacity()).sum();
        self.retain_batch();
        Ok(())
    }

    pub(crate) fn spill(&mut self) -> Result<bool> {
        self.check_healthy()?;
        if self.entries.is_empty() {
            return Ok(false);
        }
        self.failed = true;
        self.spill_inner()?;
        self.failed = false;
        Ok(true)
    }

    fn writer(&self) -> Result<RunWriter> {
        let (file, guard) = create_budgeted_temporary_file_in_context(
            self.directory,
            "text-sort",
            "tmp",
            self.disk.clone(),
            self.work.context(),
            "operation_scratch",
        )?;
        Ok(RunWriter::from_file(file, guard, 0))
    }

    fn spill_inner(&mut self) -> Result<()> {
        self.combine_batch()?;
        let mut io = self.work.context().workspace_reservation();
        let mut output = self.writer()?;
        let capacity = io.grow_to(0, IO_BYTES, "operation_scratch")?;
        output.enable_buffer(capacity)?;
        for entry in &self.entries {
            output.write_entry(entry)?;
        }
        let mut run = output.finish()?;
        io.retain(0);
        self.entries = Vec::new();
        self.key_bytes = 0;
        self.work.retain(0);
        for level in 0..self.levels.len() {
            match self.levels[level].take() {
                Some(previous) => run = self.merge(previous, run)?,
                None => {
                    self.levels[level] = Some(run);
                    return Ok(());
                }
            }
        }
        Err(Error::new(
            ErrorCode::FileFormatError,
            "term frequency run count overflow",
        ))
    }

    fn merge(&self, left: Run, right: Run) -> Result<Run> {
        let left_max = left.max_key;
        let right_max = right.max_key;
        let mut heads = self.work.context().workspace_reservation();
        heads.grow_to_wide(
            left_max as u128 + right_max as u128,
            left_max as u128 + right_max as u128,
            "operation_scratch",
        )?;
        let mut a = empty_head(left_max)?;
        let mut b = empty_head(right_max)?;
        let mut io = self.work.context().workspace_reservation();
        let left = RunCursor::with_capacity(left, left_max, 0)?;
        let right = RunCursor::with_capacity(right, right_max, 0)?;
        let mut output = self.writer()?;
        let capacity = io.grow_to(0, 3 * IO_BYTES, "operation_scratch")?;
        let mut left = left.enable_buffer(capacity.min(IO_BYTES));
        let mut right = right.enable_buffer(capacity.saturating_sub(IO_BYTES).min(IO_BYTES));
        output.enable_buffer(capacity.saturating_sub(2 * IO_BYTES))?;
        let mut x = left.read_next(&mut a.key, true)?;
        let mut y = right.read_next(&mut b.key, true)?;
        while x.is_some() || y.is_some() {
            let order = match (x, y) {
                (Some(_), Some(_)) => a.key.cmp(&b.key),
                (Some(_), None) => Ordering::Less,
                _ => Ordering::Greater,
            };
            if order == Ordering::Equal {
                a.value = sum(x.unwrap().1, y.unwrap().1)?;
                output.write_entry(&a)?;
                x = left.read_next(&mut a.key, true)?;
                y = right.read_next(&mut b.key, true)?;
            } else if order == Ordering::Less {
                a.value = x.unwrap().1;
                output.write_entry(&a)?;
                x = left.read_next(&mut a.key, true)?;
            } else {
                b.value = y.unwrap().1;
                output.write_entry(&b)?;
                y = right.read_next(&mut b.key, true)?;
            }
        }
        output.finish()
    }

    #[cfg(test)]
    pub(crate) fn finish(mut self, visit: impl FnMut(&[u8], u64) -> Result<()>) -> Result<()> {
        self.finish_batch(visit)
    }

    pub(crate) fn finish_batch(
        &mut self,
        mut visit: impl FnMut(&[u8], u64) -> Result<()>,
    ) -> Result<()> {
        self.check_healthy()?;
        // A later checksum or visitor failure must not replay consumed entries.
        self.failed = true;
        if self.levels.iter().all(Option::is_none) {
            self.combine_batch()?;
            for entry in &self.entries {
                visit(&entry.key, entry.value)?;
            }
            self.entries.clear();
            self.key_bytes = 0;
            self.retain_batch();
            self.failed = false;
            return Ok(());
        }
        if !self.entries.is_empty() {
            self.spill_inner()?;
        }
        let mut combined = None;
        for level in 0..self.levels.len() {
            if let Some(run) = self.levels[level].take() {
                combined = Some(match combined {
                    Some(previous) => self.merge(previous, run)?,
                    None => run,
                });
            }
        }
        if let Some(run) = combined {
            let mut key_work = self.work.context().workspace_reservation();
            key_work.grow_to(run.max_key, run.max_key, "operation_scratch")?;
            let mut entry = empty_head(run.max_key)?;
            let mut io = self.work.context().workspace_reservation();
            let capacity = io.grow_to(0, IO_BYTES, "operation_scratch")?;
            let mut cursor = RunCursor::with_capacity(run, entry.key.capacity(), capacity)?;
            while let Some((_, value)) = cursor.read_next(&mut entry.key, true)? {
                visit(&entry.key, value)?;
            }
        }
        self.failed = false;
        Ok(())
    }

    fn check_healthy(&self) -> Result<()> {
        if self.failed {
            Err(Error::new(
                ErrorCode::FileFormatError,
                "cannot reuse failed term frequency sorting",
            ))
        } else {
            Ok(())
        }
    }
}

fn key_matches(key: &[u8], parts: &[&[u8]]) -> bool {
    key.iter()
        .copied()
        .eq(parts.iter().flat_map(|part| part.iter().copied()))
}

fn empty_head(capacity: usize) -> Result<Entry> {
    let mut key = Vec::new();
    key.try_reserve_exact(capacity).map_err(allocation_error)?;
    check_key_capacity(&key, capacity)?;
    Ok(Entry {
        key,
        line: 0,
        value: 0,
    })
}

fn sum(left: u64, right: u64) -> Result<u64> {
    left.checked_add(right)
        .ok_or_else(|| Error::new(ErrorCode::FileFormatError, "text term frequency overflow"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json::WorkLedger;
    use crate::temporary::sort::tests::TestDirectory;
    use std::collections::BTreeMap;
    use std::io::{Seek, SeekFrom, Write};

    fn assert_released(context: &JsonOperationContext, limit: usize) {
        let mut probe = context.workspace_reservation();
        probe.grow_to(limit, limit, "operation_scratch").unwrap();
        assert!(probe
            .grow_to(limit + 1, limit + 1, "operation_scratch")
            .is_err());
    }

    #[test]
    fn completed_batches_reuse_only_the_charged_array_without_carrying_frequencies() {
        let directory = TestDirectory::new("tf-reuse");
        let context = JsonOperationContext::standalone();
        let available = context.available_workspace_bytes();
        let disk = Arc::new(TemporaryDiskBudget::new(0));
        let mut sorter = ExternalTermFrequencies::new(&directory.0, disk.clone(), &context);
        sorter.push(&[b"alpha"], 1).unwrap();
        sorter.push(&[b"beta"], 2).unwrap();
        sorter.finish_batch(|_, _| Ok(())).unwrap();
        let pointer = sorter.entries.as_ptr();
        let array = sorter.entries.capacity() * std::mem::size_of::<Entry>();
        for seed in 0..100u64 {
            let (result, allocated) = crate::test_alloc::allocation_bytes(|| {
                sorter.push(&[b"alpha"], seed + 1)?;
                sorter.push(&[b"beta"], seed + 2)?;
                sorter.push(&[b"alpha"], seed + 3)?;
                let mut count = 0;
                sorter.finish_batch(|key, value| {
                    match count {
                        0 => {
                            assert_eq!(key, b"alpha");
                            assert_eq!(value, 2 * seed + 4);
                        }
                        1 => {
                            assert_eq!(key, b"beta");
                            assert_eq!(value, seed + 2);
                        }
                        _ => panic!("unexpected term"),
                    }
                    count += 1;
                    Ok(())
                })?;
                assert_eq!(count, 2);
                Ok::<(), Error>(())
            });
            result.unwrap();
            assert_eq!(allocated, (2 * "alpha".len() + "beta".len()) as u128);
            assert_eq!(sorter.entries.as_ptr(), pointer);
            assert_eq!(sorter.work.reserved_bytes(), array);
            assert_eq!(context.available_workspace_bytes(), available - array);
            assert_eq!(sorter.key_bytes, 0);
            assert!(sorter.levels.iter().all(Option::is_none));
        }
        assert!(sorter.release_empty_batch_capacity().unwrap());
        assert!(!sorter.release_empty_batch_capacity().unwrap());
        assert_eq!(sorter.work.reserved_bytes(), 0);
        drop(sorter);
        directory.assert_clean(&disk);
        assert_released(&context, available);
    }

    #[test]
    fn empty_reused_array_yields_to_the_first_keys_exact_minimum_on_each_ledger() {
        let directory = TestDirectory::new("tf-first-key-reclaim");
        let slot = std::mem::size_of::<Entry>();
        let limit = 4096;
        for parent in [false, true] {
            for delta in [-1isize, 0, 1] {
                let context = JsonOperationContext::new(Arc::new(WorkLedger::with_limits(
                    if parent { 2 * limit } else { limit },
                    2 * limit,
                    if parent { limit } else { 2 * limit },
                )));
                let disk = Arc::new(TemporaryDiskBudget::new(0));
                let mut sorter = ExternalTermFrequencies::new(&directory.0, disk.clone(), &context);
                sorter.push(&[b"warm"], 1).unwrap();
                sorter.finish_batch(|_, _| Ok(())).unwrap();
                assert_eq!(sorter.entries.capacity(), 4);
                let array = 4 * slot;
                let mut held = context.workspace_reservation();
                held.grow_to(limit - array, limit - array, "operation_scratch")
                    .unwrap();
                assert_eq!(context.available_workspace_bytes(), 0);
                let length = (3 * slot) as isize + delta;
                let key = vec![b'z'; length as usize];
                let result = sorter.push(&[&key], 7);
                if delta > 0 {
                    let error = result.unwrap_err();
                    assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                    for (name, expected) in [
                        ("resource", "memory".to_owned()),
                        ("operation", "operation_scratch".to_owned()),
                        ("limit_bytes", limit.to_string()),
                        ("requested_bytes", (limit + 1).to_string()),
                    ] {
                        assert_eq!(
                            error.detail().iter().find(|item| item.0 == name).unwrap().1,
                            expected
                        );
                    }
                    assert!(sorter.failed);
                } else {
                    result.unwrap();
                    assert_eq!(sorter.entries.capacity(), 1);
                    assert_eq!(sorter.work.reserved_bytes(), slot + key.len());
                    assert_eq!(context.available_workspace_bytes(), (-delta) as usize);
                    sorter
                        .finish_batch(|bytes, value| {
                            assert_eq!(bytes, key);
                            assert_eq!(value, 7);
                            Ok(())
                        })
                        .unwrap();
                }
                drop((sorter, held));
                directory.assert_clean(&disk);
                assert_released(&context, limit);
            }
        }
    }

    #[test]
    fn completed_run_batches_do_not_contaminate_following_in_memory_or_run_batches() {
        let directory = TestDirectory::new("tf-run-batches");
        let context = JsonOperationContext::standalone();
        let available = context.available_workspace_bytes();
        let disk = Arc::new(TemporaryDiskBudget::new(1024 * 1024));
        let mut sorter = ExternalTermFrequencies::new(&directory.0, disk.clone(), &context);
        for seed in 0..100u64 {
            let spill = seed % 3 != 1;
            for (a, b) in [(seed + 1, seed + 2), (seed + 3, seed + 4)] {
                sorter.push(&[b"alpha"], a).unwrap();
                sorter.push(&[b"beta"], b).unwrap();
                if spill {
                    sorter.spill().unwrap();
                }
            }
            assert_eq!(disk.used() != 0, spill);
            let mut count = 0;
            sorter
                .finish_batch(|key, value| {
                    assert_eq!(
                        key,
                        if count == 0 {
                            &b"alpha"[..]
                        } else {
                            &b"beta"[..]
                        }
                    );
                    assert_eq!(value, 2 * seed + 4 + 2 * count);
                    count += 1;
                    Ok(())
                })
                .unwrap();
            assert_eq!(count, 2);
            assert!(sorter.levels.iter().all(Option::is_none));
            assert_eq!(sorter.key_bytes, 0);
            assert!(sorter.entries.is_empty());
            directory.assert_clean(&disk);
        }
        drop(sorter);
        assert_released(&context, available);
    }

    #[test]
    fn batch_finish_failure_poisoning_prevents_replaying_partial_visitors() {
        let directory = TestDirectory::new("tf-batch-poison");
        for mode in 0..4 {
            let context = JsonOperationContext::standalone();
            let available = context.available_workspace_bytes();
            let disk = Arc::new(TemporaryDiskBudget::new(4096));
            let mut sorter = ExternalTermFrequencies::new(&directory.0, disk.clone(), &context);
            sorter
                .push(&[b"alpha"], if mode == 2 { u64::MAX } else { 2 })
                .unwrap();
            sorter.push(&[b"beta"], 3).unwrap();
            if mode == 2 {
                sorter.push(&[b"alpha"], 1).unwrap();
            } else if mode != 0 {
                sorter.spill().unwrap();
                if mode == 3 {
                    let run = sorter.levels.iter_mut().find_map(Option::as_mut).unwrap();
                    run.file.seek(SeekFrom::Start(20)).unwrap();
                    run.file.write_all(b"X").unwrap();
                }
            }
            let mut calls = 0;
            let error = sorter
                .finish_batch(|_, _| {
                    calls += 1;
                    if mode < 2 {
                        Err(std::io::Error::other("visitor failed").into())
                    } else {
                        Ok(())
                    }
                })
                .unwrap_err();
            assert_eq!(
                error.code(),
                match mode {
                    0 | 1 => ErrorCode::IoError,
                    2 => ErrorCode::FileFormatError,
                    _ => ErrorCode::ChecksumMismatch,
                }
            );
            assert_eq!(
                calls,
                match mode {
                    0 | 1 => 1,
                    2 => 0,
                    _ => 2,
                }
            );
            assert_eq!(
                sorter.push(&[b"new"], 1).unwrap_err().code(),
                ErrorCode::FileFormatError
            );
            assert_eq!(
                sorter.spill().unwrap_err().code(),
                ErrorCode::FileFormatError
            );
            assert_eq!(
                sorter.release_empty_batch_capacity().unwrap_err().code(),
                ErrorCode::FileFormatError
            );
            assert_eq!(
                sorter
                    .finish_batch(|_, _| panic!("replayed visitor"))
                    .unwrap_err()
                    .code(),
                ErrorCode::FileFormatError
            );
            drop(sorter);
            directory.assert_clean(&disk);
            assert_released(&context, available);
        }
    }

    #[test]
    fn tiny_frequency_batches_have_no_fixed_floor_or_temporary_files() {
        let directory = TestDirectory::new("tf-minimum");
        for seed in 0..100usize {
            let key = vec![27; seed + 1];
            let minimum = std::mem::size_of::<Entry>() + key.len();
            for parent in [false, true] {
                for limit in [minimum - 1, minimum, minimum + 1] {
                    let context = JsonOperationContext::new(Arc::new(WorkLedger::with_limits(
                        if parent { limit * 2 } else { limit },
                        limit * 2,
                        if parent { limit } else { limit * 2 },
                    )));
                    let disk = Arc::new(TemporaryDiskBudget::new(0));
                    let mut sorter =
                        ExternalTermFrequencies::new(&directory.0, disk.clone(), &context);
                    let result = sorter.push(&[&key[..seed / 2], &key[seed / 2..]], 7);
                    if limit < minimum {
                        let error = result.unwrap_err();
                        assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                        assert_eq!(
                            error
                                .detail()
                                .iter()
                                .find(|item| item.0 == "resource")
                                .unwrap()
                                .1,
                            "memory"
                        );
                        assert_eq!(
                            error
                                .detail()
                                .iter()
                                .find(|item| item.0 == "requested_bytes")
                                .unwrap()
                                .1,
                            minimum.to_string()
                        );
                        assert!(sorter.entries.is_empty());
                        assert_eq!(
                            sorter.push(&[b"x"], 1).unwrap_err().code(),
                            ErrorCode::FileFormatError
                        );
                        drop(sorter);
                    } else {
                        result.unwrap();
                        sorter.push(&[&key], 9).unwrap();
                        assert_eq!(sorter.entries.capacity(), 1);
                        assert_eq!(sorter.work.reserved_bytes(), minimum);
                        let mut visited = 0;
                        sorter
                            .finish(|bytes, value| {
                                assert_eq!(bytes, key);
                                assert_eq!(value, 16);
                                assert_eq!(context.available_workspace_bytes(), limit - minimum);
                                visited += 1;
                                Ok(())
                            })
                            .unwrap();
                        assert_eq!(visited, 1);
                    }
                    directory.assert_clean(&disk);
                    assert_released(&context, limit);
                }
            }
        }
    }

    #[test]
    fn in_memory_and_multilevel_frequency_runs_match_an_independent_sum_oracle() {
        let directory = TestDirectory::new("tf-oracle");
        for seed in 0..100usize {
            let entries = (0..128usize)
                .map(|i| {
                    let id = [9, 10, 255, 256][(i + seed) % 4];
                    let term = ["prefix", "prefixes", "\u{00e9}", "\u{03a3}"][(i / 4 + seed) % 4];
                    (
                        format!("c{}\0k{id}\0{term}", (i / 16 + seed) % 3).into_bytes(),
                        (i % 7 + 1) as u64,
                    )
                })
                .collect::<Vec<_>>();
            let mut expected = BTreeMap::<Vec<u8>, u128>::new();
            for (key, value) in &entries {
                *expected.entry(key.clone()).or_default() += *value as u128;
            }
            for batch in [usize::MAX, 7, 19] {
                let context = JsonOperationContext::standalone();
                let available = context.available_workspace_bytes();
                let disk = Arc::new(TemporaryDiskBudget::new(1024 * 1024));
                let mut sorter = ExternalTermFrequencies::new(&directory.0, disk.clone(), &context);
                for (ordinal, (key, value)) in entries.iter().enumerate() {
                    sorter.push(&[key], *value).unwrap();
                    if (ordinal + 1) % batch == 0 {
                        sorter.spill().unwrap();
                    }
                }
                if batch == usize::MAX {
                    assert_eq!(disk.used(), 0);
                } else {
                    assert!(disk.used() > 0);
                }
                let mut observed = BTreeMap::new();
                let mut previous = None;
                sorter
                    .finish(|key, value| {
                        assert!(previous
                            .as_deref()
                            .map_or(true, |previous: &[u8]| previous < key));
                        previous = Some(key.to_vec());
                        assert!(observed.insert(key.to_vec(), value as u128).is_none());
                        Ok(())
                    })
                    .unwrap();
                assert_eq!(observed, expected, "seed={seed}/batch={batch}");
                directory.assert_clean(&disk);
                assert_released(&context, available);
            }
        }
    }

    #[test]
    fn batch_growth_admits_the_larger_of_old_array_overlap_and_later_key() {
        let directory = TestDirectory::new("tf-growth-minimum");
        let slot = std::mem::size_of::<Entry>();
        for length in [1, slot - 1, slot, slot + 1, 100] {
            let minimum = 3 + 2 * slot + slot.max(length);
            for delta in [-1isize, 0, 1] {
                let limit = (minimum as isize + delta) as usize;
                let context = JsonOperationContext::new(Arc::new(WorkLedger::with_limits(
                    limit,
                    limit * 2,
                    limit * 2,
                )));
                let disk = Arc::new(TemporaryDiskBudget::new(0));
                let mut held = context.workspace_reservation();
                held.grow_to(limit - slot - 3, limit - slot - 3, "operation_scratch")
                    .unwrap();
                let mut sorter = ExternalTermFrequencies::new(&directory.0, disk.clone(), &context);
                sorter.push(&[b"old"], 2).unwrap();
                assert_eq!(sorter.entries.capacity(), 1);
                held.retain(0);
                let key = vec![b'z'; length];
                if delta < 0 {
                    let error = sorter.insert(&[&key], key.len(), 3).unwrap_err();
                    assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                    assert_eq!(
                        error
                            .detail()
                            .iter()
                            .find(|item| item.0 == "requested_bytes")
                            .unwrap()
                            .1,
                        minimum.to_string()
                    );
                    assert_eq!(sorter.entries.len(), 1);
                    drop(sorter);
                } else {
                    sorter.push(&[&key], 3).unwrap();
                    assert_eq!(sorter.entries.capacity(), 2);
                    let expected = BTreeMap::from([(b"old".to_vec(), 2), (key, 3)]);
                    let mut observed = BTreeMap::new();
                    sorter
                        .finish(|key, value| {
                            observed.insert(key.to_vec(), value);
                            Ok(())
                        })
                        .unwrap();
                    assert_eq!(observed, expected);
                }
                directory.assert_clean(&disk);
                assert_released(&context, limit);
            }
        }
    }

    #[test]
    fn frequency_overflow_poisoning_and_failure_cleanup_cover_batch_and_merge() {
        let directory = TestDirectory::new("tf-overflow");
        for mode in 0..4 {
            let context = JsonOperationContext::standalone();
            let available = context.available_workspace_bytes();
            let disk = Arc::new(TemporaryDiskBudget::new(1024 * 1024));
            let mut sorter = ExternalTermFrequencies::new(&directory.0, disk.clone(), &context);
            sorter.push(&[b"shared"], u64::MAX - 1).unwrap();
            if mode != 0 {
                sorter.spill().unwrap();
            }
            sorter.push(&[b"shared"], 1).unwrap();
            if mode == 2 {
                sorter.spill().unwrap();
            }
            if mode == 3 {
                let mut value = None;
                sorter
                    .finish(|_, frequency| {
                        value = Some(frequency);
                        Ok(())
                    })
                    .unwrap();
                assert_eq!(value, Some(u64::MAX));
            } else if mode == 0 {
                let error = sorter.push(&[b"shared"], 1).unwrap_err();
                assert_eq!(error.message(), "text term frequency overflow");
                assert_eq!(
                    sorter.push(&[b"other"], 1).unwrap_err().code(),
                    ErrorCode::FileFormatError
                );
                assert!(sorter
                    .finish(|_, _| panic!("failed sorter must not visit"))
                    .is_err());
            } else {
                sorter.push(&[b"shared"], 1).unwrap();
                let error = sorter
                    .finish(|_, _| panic!("overflow must precede visits"))
                    .unwrap_err();
                assert_eq!(error.code(), ErrorCode::FileFormatError);
                assert_eq!(error.message(), "text term frequency overflow");
            }
            directory.assert_clean(&disk);
            assert_released(&context, available);
        }
        for spill in [false, true] {
            let context = JsonOperationContext::standalone();
            let available = context.available_workspace_bytes();
            let disk = Arc::new(TemporaryDiskBudget::new(4096));
            let mut sorter = ExternalTermFrequencies::new(&directory.0, disk.clone(), &context);
            sorter.push(&[b"shared"], u64::MAX - 1).unwrap();
            sorter.push(&[b"other"], 1).unwrap();
            sorter.push(&[b"shared"], 2).unwrap();
            let error = if spill {
                let error = sorter.spill().unwrap_err();
                assert!(sorter.push(&[b"next"], 1).is_err());
                drop(sorter);
                error
            } else {
                sorter
                    .finish(|_, _| panic!("batch overflow must precede visits"))
                    .unwrap_err()
            };
            assert_eq!(error.message(), "text term frequency overflow");
            directory.assert_clean(&disk);
            assert_released(&context, available);
        }
    }

    #[cfg(unix)]
    #[test]
    fn unequal_merge_heads_admit_exact_file_creation_peak_without_a_fixed_floor() {
        let directory = TestDirectory::new("tf-merge-minimum");
        let left_max = 37;
        let right_max = 201;
        let management = crate::temporary::test_file_management_bytes(&directory.0);
        let minimum = left_max + right_max + 3 * management + 128;
        assert!(minimum < 64 * 1024);
        for parent in [false, true] {
            for delta in [-1isize, 0, 1] {
                let limit = (minimum as isize + delta) as usize;
                let context = JsonOperationContext::new(Arc::new(WorkLedger::with_limits(
                    if parent { 2 * limit } else { limit },
                    2 * limit,
                    if parent { limit } else { 2 * limit },
                )));
                let disk = Arc::new(TemporaryDiskBudget::new(8192));
                let sorter = ExternalTermFrequencies::new(&directory.0, disk.clone(), &context);
                let make_run = |max: usize, side: u64| {
                    let mut writer = sorter.writer().unwrap();
                    for (suffix, length) in [(b'a', max), (b'b', 1), (b'c', max)] {
                        let mut key = vec![b'x'; length];
                        key[0] = suffix;
                        writer
                            .write_entry(&Entry {
                                key,
                                line: 0,
                                value: side,
                            })
                            .unwrap();
                    }
                    writer.finish().unwrap()
                };
                let left = make_run(left_max, 2);
                let right = make_run(right_max, 3);
                let result = sorter.merge(left, right);
                if delta < 0 {
                    let error = result.err().unwrap();
                    assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                    assert_eq!(
                        error
                            .detail()
                            .iter()
                            .find(|item| item.0 == "requested_bytes")
                            .unwrap()
                            .1,
                        minimum.to_string()
                    );
                } else {
                    let run = result.unwrap();
                    assert_eq!(context.available_workspace_bytes(), limit - management);
                    let mut cursor = RunCursor::with_capacity(run, right_max, 0).unwrap();
                    let mut observed = Vec::new();
                    while let Some(entry) = cursor.next().unwrap() {
                        observed.push((entry.key, entry.value));
                    }
                    let mut expected = BTreeMap::new();
                    for (length, value) in [(left_max, 2), (right_max, 3)] {
                        for (suffix, count) in [(b'a', length), (b'b', 1), (b'c', length)] {
                            let mut key = vec![b'x'; count];
                            key[0] = suffix;
                            *expected.entry(key).or_insert(0) += value;
                        }
                    }
                    assert_eq!(observed, expected.into_iter().collect::<Vec<_>>());
                    drop(cursor);
                }
                drop(sorter);
                directory.assert_clean(&disk);
                assert_released(&context, limit);
            }
        }
    }

    #[test]
    fn merge_reuses_two_large_heads_even_for_equal_keys_and_unequal_runs() {
        let directory = TestDirectory::new("tf-heads");
        let size = 65 * 1024;
        let context = JsonOperationContext::standalone();
        let available = context.available_workspace_bytes();
        let disk = Arc::new(TemporaryDiskBudget::new(8 * 1024 * 1024));
        let sorter = ExternalTermFrequencies::new(&directory.0, disk.clone(), &context);
        let make_run = |side: u64| {
            let mut writer = sorter.writer().unwrap();
            for suffix in 0..if side == 0 { 7 } else { 4 } {
                let mut key = vec![b'x'; size];
                key[size - 1] = b'a' + suffix;
                writer
                    .write_entry(&Entry {
                        key,
                        line: 0,
                        value: side + 1,
                    })
                    .unwrap();
            }
            writer.finish().unwrap()
        };
        let left = make_run(0);
        let right = make_run(1);
        let (result, allocations) =
            crate::test_alloc::key_allocations(size, || sorter.merge(left, right));
        let run = result.unwrap();
        assert_eq!(run.max_key, size);
        assert_eq!(allocations.allocated, 2);
        assert_eq!(allocations.peak, 2);
        assert_eq!(allocations.live, 0);
        assert!(!allocations.overflow);
        let mut cursor = RunCursor::new(run, size).unwrap();
        for suffix in 0..7 {
            let entry = cursor.next().unwrap().unwrap();
            assert_eq!(entry.key.len(), size);
            assert_eq!(entry.key[size - 1], b'a' + suffix);
            assert_eq!(entry.value, if suffix < 4 { 3 } else { 1 });
        }
        assert!(cursor.next().unwrap().is_none());
        drop(cursor);
        drop(sorter);
        directory.assert_clean(&disk);
        assert_released(&context, available);
    }

    #[test]
    fn disk_corruption_and_visitor_errors_release_frequency_owners() {
        let directory = TestDirectory::new("tf-failure");
        for mode in 0..5 {
            let context = JsonOperationContext::standalone();
            let available = context.available_workspace_bytes();
            let disk = Arc::new(TemporaryDiskBudget::new(if mode == 0 { 0 } else { 4096 }));
            let mut sorter = ExternalTermFrequencies::new(&directory.0, disk.clone(), &context);
            sorter.push(&[b"term"], 2).unwrap();
            if mode == 0 {
                assert_eq!(
                    sorter.spill().unwrap_err().code(),
                    ErrorCode::ResourceExhausted
                );
                assert_eq!(
                    sorter.spill().unwrap_err().code(),
                    ErrorCode::FileFormatError
                );
                drop(sorter);
            } else {
                sorter.spill().unwrap();
                let run = sorter.levels.iter_mut().find_map(Option::as_mut).unwrap();
                match mode {
                    1 => run.file.set_len(run.bytes - 1).unwrap(),
                    2 => {
                        run.file.seek(SeekFrom::Start(20)).unwrap();
                        run.file.write_all(b"X").unwrap();
                    }
                    3 => run.max_key = 1,
                    _ => (),
                }
                let error = sorter
                    .finish(|_, _| {
                        if mode == 4 {
                            Err(std::io::Error::other("visitor failed").into())
                        } else {
                            Ok(())
                        }
                    })
                    .unwrap_err();
                assert_eq!(
                    error.code(),
                    if mode == 4 {
                        ErrorCode::IoError
                    } else {
                        ErrorCode::ChecksumMismatch
                    }
                );
            }
            directory.assert_clean(&disk);
            assert_released(&context, available);
        }
    }

    #[test]
    fn merge_output_refusal_and_input_crc_or_eof_failures_release_all_owners() {
        let directory = TestDirectory::new("tf-merge-failure");
        for mode in 0..3 {
            let context = JsonOperationContext::standalone();
            let available = context.available_workspace_bytes();
            let disk = Arc::new(TemporaryDiskBudget::new(if mode == 0 { 48 } else { 4096 }));
            let sorter = ExternalTermFrequencies::new(&directory.0, disk.clone(), &context);
            let make_run = |value| {
                let mut writer = sorter.writer().unwrap();
                writer
                    .write_entry(&Entry {
                        key: b"term".to_vec(),
                        line: 0,
                        value,
                    })
                    .unwrap();
                writer.finish().unwrap()
            };
            let mut left = make_run(2);
            let right = make_run(3);
            if mode == 1 {
                left.file.seek(SeekFrom::Start(20)).unwrap();
                left.file.write_all(b"X").unwrap();
            }
            if mode < 2 {
                let error = sorter.merge(left, right).err().unwrap();
                assert_eq!(
                    error.code(),
                    if mode == 0 {
                        ErrorCode::ResourceExhausted
                    } else {
                        ErrorCode::ChecksumMismatch
                    }
                );
            } else {
                // Truncate after metadata validation, before an unbuffered read.
                let mut cursor = RunCursor::with_capacity(left, 4, 0).unwrap();
                cursor.input.get_ref().set_len(22).unwrap();
                let mut head = empty_head(4).unwrap();
                let error = cursor.read_next(&mut head.key, true).unwrap_err();
                assert_eq!(error.code(), ErrorCode::IoError);
                assert_eq!(cursor.count, 1);
                assert_eq!(cursor.bytes, 24);
                drop((cursor, right));
            }
            drop(sorter);
            directory.assert_clean(&disk);
            assert_released(&context, available);
        }
    }
}
