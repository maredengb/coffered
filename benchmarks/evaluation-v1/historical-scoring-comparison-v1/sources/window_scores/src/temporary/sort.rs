use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, Write};
use std::path::Path;
#[cfg(test)]
use std::path::PathBuf;
use std::sync::Arc;

#[cfg(test)]
use super::create_budgeted_temporary_file_in;
use super::{
    create_budgeted_sort_file, TempFileGuard, TemporaryDiskBudget, TemporaryFile, TemporaryPath,
};
use crate::hash::Crc32c;
use crate::json::{JsonOperationContext, WorkspaceReservation};
use crate::{Error, ErrorCode, Result};

mod lookup;
pub(crate) use lookup::ExternalKeyLookup;
mod frequencies;
pub(crate) use frequencies::ExternalTermFrequencies;

const IO_BYTES: usize = 8192;
const ENTRY_HEADER_BYTES: usize = 20;
const MIN_WORKSPACE: usize = 64 * 1024;

fn required_workspace(key: u128) -> u128 {
    let entry = std::mem::size_of::<Entry>() as u128;
    (MIN_WORKSPACE as u128)
        .max(3 * IO_BYTES as u128 + 2 * (key + entry))
        .max(lookup::required_workspace(key))
}

struct Entry {
    key: Vec<u8>,
    line: u64,
    value: u64,
}

impl Entry {
    fn header(&self) -> Result<[u8; ENTRY_HEADER_BYTES]> {
        let length = u32::try_from(self.key.len()).map_err(|_| invalid_run())?;
        let mut header = [0; ENTRY_HEADER_BYTES];
        header[..4].copy_from_slice(&length.to_le_bytes());
        header[4..12].copy_from_slice(&self.line.to_le_bytes());
        header[12..].copy_from_slice(&self.value.to_le_bytes());
        Ok(header)
    }
}

pub(crate) struct ExternalUniqueKeys {
    directory: TemporaryPath,
    disk: Arc<TemporaryDiskBudget>,
    workspace: usize,
    workspace_limit: usize,
    entries: Vec<Entry>,
    key_bytes: usize,
    largest_key: usize,
    levels: [Option<Run>; 64],
    duplicate_code: ErrorCode,
    duplicate_message: &'static str,
    failed: bool,
    #[cfg(test)]
    after_push_flush: Option<Box<dyn FnOnce() + Send>>,
    // Drop after entries and runs; finish_lookup transfers this lease with the data.
    work: WorkspaceReservation,
}

impl ExternalUniqueKeys {
    pub(crate) fn new(
        directory: &Path,
        disk: Arc<TemporaryDiskBudget>,
        workspace: usize,
        context: &JsonOperationContext,
        duplicate_code: ErrorCode,
        duplicate_message: &'static str,
    ) -> Result<Self> {
        if workspace < MIN_WORKSPACE {
            return Err(workspace_error(workspace, MIN_WORKSPACE as u128));
        }
        let mut work = context.workspace_reservation();
        let directory = TemporaryPath::copy_in(directory, context)?;
        // Start with the minimum so empty sorters do not consume the space
        // needed by input parsing and other reference tables.
        let granted = work.grow_to(MIN_WORKSPACE, MIN_WORKSPACE, "catalog_build")?;
        Ok(Self {
            directory,
            disk,
            workspace: granted,
            workspace_limit: workspace,
            entries: Vec::new(),
            key_bytes: 0,
            largest_key: 0,
            levels: std::array::from_fn(|_| None),
            duplicate_code,
            duplicate_message,
            failed: false,
            #[cfg(test)]
            after_push_flush: None,
            work,
        })
    }

    pub(crate) fn push(&mut self, parts: &[&[u8]], line: u64) -> Result<()> {
        self.push_value(parts, line, 0)
    }

    pub(crate) fn push_value(&mut self, parts: &[&[u8]], line: u64, value: u64) -> Result<()> {
        if self.failed {
            return Err(failed_sort());
        }
        self.failed = true;
        self.push_inner(parts, line, value)?;
        self.failed = false;
        Ok(())
    }

    fn push_inner(&mut self, parts: &[&[u8]], line: u64, value: u64) -> Result<()> {
        let length = parts.iter().try_fold(0_u128, |total, part| {
            total
                .checked_add(part.len() as u128)
                .ok_or_else(invalid_run)
        })?;
        let minimum = required_workspace(length.max(self.largest_key as u128));
        if minimum > self.workspace_limit as u128 {
            return Err(workspace_error(self.workspace_limit, minimum));
        }
        let length = length as usize;
        if minimum > self.workspace as u128 {
            // Finish the old batch under its existing lease before growing for
            // a larger key; old batch capacity must not overlap merge heads.
            self.flush_run()?;
            self.workspace = self.work.grow_to_wide(minimum, minimum, "catalog_build")?;
        }
        let entry_bytes = std::mem::size_of::<Entry>();
        let required = self
            .key_bytes
            .saturating_add(length)
            .saturating_add(self.entries.capacity().max(self.entries.len() + 1) * entry_bytes);
        if required > self.workspace / 2 && self.workspace < self.workspace_limit {
            // A larger batch is optional. If the shared ledger is busy, keep
            // the existing lease and spill a smaller run instead of failing.
            self.workspace = self.work.grow_to(
                self.workspace,
                self.workspace.saturating_mul(2).min(self.workspace_limit),
                "catalog_build",
            )?;
        }
        if required > self.workspace / 2 {
            self.flush_run()?;
            #[cfg(test)]
            if let Some(after_flush) = self.after_push_flush.take() {
                after_flush();
            }
            self.workspace = self.work.grow_to_wide(minimum, minimum, "catalog_build")?;
        }
        let maximum_capacity = (self.workspace / 2 - self.key_bytes - length) / entry_bytes;
        if self.entries.len() == self.entries.capacity() {
            let target = self
                .entries
                .capacity()
                .saturating_mul(2)
                .max(4)
                .min(maximum_capacity);
            self.entries
                .try_reserve_exact(target - self.entries.len())
                .map_err(allocation_error)?;
        }
        let mut key = Vec::new();
        key.try_reserve_exact(length).map_err(allocation_error)?;
        check_key_capacity(&key, length)?;
        let allocated = self.key_bytes as u128
            + key.capacity() as u128
            + self.entries.capacity() as u128 * entry_bytes as u128;
        if allocated > (self.workspace / 2) as u128 {
            return Err(workspace_error(self.workspace / 2, allocated));
        }
        for part in parts {
            key.extend_from_slice(part);
        }
        self.key_bytes += key.capacity();
        self.largest_key = self.largest_key.max(length);
        self.entries.push(Entry { key, line, value });
        Ok(())
    }

    fn duplicate(&self, first: u64, second: u64) -> Error {
        Error::new(self.duplicate_code, self.duplicate_message)
            .with_detail("line", first.max(second))
            .with_detail("first_line", first.min(second))
    }

    fn flush_run(&mut self) -> Result<()> {
        if self.entries.is_empty() {
            return Ok(());
        }
        self.entries.sort_unstable_by(|left, right| {
            left.key.cmp(&right.key).then(left.line.cmp(&right.line))
        });
        for pair in self.entries.windows(2) {
            if pair[0].key == pair[1].key {
                return Err(self.duplicate(pair[0].line, pair[1].line));
            }
        }
        let retained = required_workspace(self.largest_key as u128).max(
            (self.key_bytes + self.entries.capacity() * std::mem::size_of::<Entry>() + IO_BYTES)
                as u128,
        ) as usize;
        self.work.retain(retained);
        self.workspace = retained;
        let mut output = RunWriter::new(&self.directory, self.disk.clone(), self.work.context())?;
        for entry in &self.entries {
            output.write_entry(entry)?;
        }
        let mut run = output.finish()?;
        self.entries = Vec::new();
        self.key_bytes = 0;
        // Binary carrying bounds open runs and keeps merge I/O O(n log n).
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
            "reference sort run count exceeds its representable range",
        ))
    }

    fn merge(&mut self, left: Run, right: Run) -> Result<Run> {
        self.workspace = required_workspace(self.largest_key as u128) as usize;
        self.work.retain(self.workspace);
        let mut left = RunCursor::new(left, self.largest_key)?;
        let mut right = RunCursor::new(right, self.largest_key)?;
        let mut a = left.next()?;
        let mut b = right.next()?;
        let mut output = RunWriter::new(&self.directory, self.disk.clone(), self.work.context())?;
        while a.is_some() || b.is_some() {
            let take_left = match (&a, &b) {
                (Some(a), Some(b)) => match a.key.cmp(&b.key) {
                    std::cmp::Ordering::Equal => return Err(self.duplicate(a.line, b.line)),
                    std::cmp::Ordering::Less => true,
                    std::cmp::Ordering::Greater => false,
                },
                (Some(_), None) => true,
                _ => false,
            };
            if take_left {
                let entry = a.take().unwrap();
                output.write_entry(&entry)?;
                drop(entry);
                a = left.next()?;
            } else {
                let entry = b.take().unwrap();
                output.write_entry(&entry)?;
                drop(entry);
                b = right.next()?;
            }
        }
        output.finish()
    }

    fn finish_run(&mut self) -> Result<Option<Run>> {
        if self.failed {
            return Err(failed_sort());
        }
        self.flush_run()?;
        let mut combined = None;
        for level in 0..self.levels.len() {
            if let Some(run) = self.levels[level].take() {
                combined = Some(match combined {
                    Some(previous) => self.merge(previous, run)?,
                    None => run,
                });
            }
        }
        Ok(combined)
    }

    pub(crate) fn finish(mut self) -> Result<()> {
        if let Some(run) = self.finish_run()? {
            let mut cursor = RunCursor::new(run, self.largest_key)?;
            while cursor.next()?.is_some() {}
        }
        Ok(())
    }

    pub(crate) fn finish_lookup(mut self) -> Result<ExternalKeyLookup> {
        let run = self.finish_run()?;
        // Sorting entries are gone. Lookup needs only bounded I/O, key slots
        // and its small page cache, not the preferred sorting batch size.
        let workspace = self
            .work
            .reserved_bytes()
            .min(lookup::required_workspace(self.largest_key as u128) as usize);
        self.work.retain(if run.is_some() { workspace } else { 0 });
        ExternalKeyLookup::new(
            run,
            &self.directory,
            self.disk.clone(),
            self.largest_key,
            self.work,
        )
    }
}

struct Run {
    file: TemporaryFile,
    temporary: TempFileGuard,
    count: u64,
    bytes: u64,
    checksum: u32,
    max_key: usize,
}

struct RunWriter {
    output: Option<BufWriter<TemporaryFile>>,
    temporary: Option<TempFileGuard>,
    count: u64,
    bytes: u64,
    crc: Crc32c,
    max_key: usize,
}

impl RunWriter {
    fn new(
        directory: &Path,
        disk: Arc<TemporaryDiskBudget>,
        context: &JsonOperationContext,
    ) -> Result<Self> {
        let (file, temporary) = create_budgeted_sort_file(directory, "import-sort", disk, context)?;
        Ok(Self::from_file(file, temporary, IO_BYTES))
    }

    fn from_file(file: TemporaryFile, temporary: TempFileGuard, capacity: usize) -> Self {
        Self {
            output: Some(BufWriter::with_capacity(capacity, file)),
            temporary: Some(temporary),
            count: 0,
            bytes: 0,
            crc: Crc32c::default(),
            max_key: 0,
        }
    }

    fn enable_buffer(&mut self, capacity: usize) -> Result<()> {
        let output = self.output.take().unwrap();
        assert_eq!(output.capacity(), 0);
        let file = output.into_inner().map_err(|error| error.into_error())?;
        self.output = Some(BufWriter::with_capacity(capacity, file));
        Ok(())
    }

    fn write_entry(&mut self, entry: &Entry) -> Result<()> {
        let header = entry.header()?;
        let bytes = ENTRY_HEADER_BYTES as u64 + entry.key.len() as u64;
        let next_bytes = self.bytes.checked_add(bytes).ok_or_else(invalid_run)?;
        let next_count = self.count.checked_add(1).ok_or_else(invalid_run)?;
        self.temporary.as_mut().unwrap().reserve_disk(bytes)?;
        let output = self.output.as_mut().unwrap();
        output.write_all(&header)?;
        output.write_all(&entry.key)?;
        self.crc.update(&header);
        self.crc.update(&entry.key);
        self.bytes = next_bytes;
        self.count = next_count;
        self.max_key = self.max_key.max(entry.key.len());
        Ok(())
    }

    fn finish(mut self) -> Result<Run> {
        let file = finish_run_buffer(self.output.take().unwrap())?;
        Ok(Run {
            file,
            temporary: self.temporary.take().unwrap(),
            count: self.count,
            bytes: self.bytes,
            checksum: self.crc.finish(),
            max_key: self.max_key,
        })
    }
}

impl Drop for RunWriter {
    fn drop(&mut self) {
        if let Some(output) = self.output.take() {
            drop(output.into_parts());
        }
    }
}

fn finish_run_buffer<W: Write>(output: BufWriter<W>) -> std::io::Result<W> {
    let mut output = match output.into_inner() {
        Ok(output) => output,
        Err(error) => {
            let (error, output) = error.into_parts();
            drop(output.into_parts());
            return Err(error);
        }
    };
    output.flush()?;
    Ok(output)
}

struct RunCursor {
    input: BufReader<TemporaryFile>,
    _temporary: TempFileGuard,
    count: u64,
    bytes: u64,
    checksum: u32,
    crc: Crc32c,
    max_key: usize,
}

impl RunCursor {
    fn new(run: Run, max_key: usize) -> Result<Self> {
        Self::with_capacity(run, max_key, IO_BYTES)
    }

    fn with_capacity(mut run: Run, max_key: usize, capacity: usize) -> Result<Self> {
        if run.file.metadata()?.len() != run.bytes {
            return Err(invalid_run());
        }
        run.file.rewind()?;
        Ok(Self {
            input: BufReader::with_capacity(capacity, run.file),
            _temporary: run.temporary,
            count: run.count,
            bytes: run.bytes,
            checksum: run.checksum,
            crc: Crc32c::default(),
            max_key,
        })
    }

    fn enable_buffer(mut self, capacity: usize) -> Self {
        assert_eq!(self.input.capacity(), 0);
        self.input = BufReader::with_capacity(capacity, self.input.into_inner());
        self
    }

    fn next(&mut self) -> Result<Option<Entry>> {
        let mut key = Vec::new();
        self.read_next(&mut key, false)
            .map(|entry| entry.map(|(line, value)| Entry { key, line, value }))
    }

    fn read_next(&mut self, key: &mut Vec<u8>, reuse: bool) -> Result<Option<(u64, u64)>> {
        if self.count == 0 {
            if self.bytes != 0 || self.crc.finish() != self.checksum {
                return Err(invalid_run());
            }
            return Ok(None);
        }
        if self.bytes < ENTRY_HEADER_BYTES as u64 {
            return Err(invalid_run());
        }
        let mut header = [0; ENTRY_HEADER_BYTES];
        self.input.read_exact(&mut header)?;
        let length = u32::from_le_bytes(header[..4].try_into().unwrap()) as usize;
        if length > self.max_key || length as u64 > self.bytes - ENTRY_HEADER_BYTES as u64 {
            return Err(invalid_run());
        }
        if reuse {
            if length > key.capacity() {
                return Err(invalid_run());
            }
        } else {
            key.try_reserve_exact(length).map_err(allocation_error)?;
            check_key_capacity(key, length)?;
        }
        key.resize(length, 0);
        self.input.read_exact(key)?;
        self.crc.update(&header);
        self.crc.update(key);
        self.bytes -= ENTRY_HEADER_BYTES as u64 + length as u64;
        self.count -= 1;
        Ok(Some((
            u64::from_le_bytes(header[4..12].try_into().unwrap()),
            u64::from_le_bytes(header[12..].try_into().unwrap()),
        )))
    }
}

fn workspace_error(limit: usize, requested: u128) -> Error {
    Error::resource_exhausted("catalog", "catalog_build", limit as u64, requested)
}

fn allocation_error(error: std::collections::TryReserveError) -> Error {
    // Allocator failure has no known effective budget. Do not invent resource
    // totals or classify it as malformed input.
    std::io::Error::other(error.to_string()).into()
}

fn check_key_capacity(key: &Vec<u8>, maximum: usize) -> Result<()> {
    if key.capacity() > maximum {
        return Err(std::io::Error::other("key allocation exceeds admitted capacity").into());
    }
    Ok(())
}

fn invalid_run() -> Error {
    Error::new(
        ErrorCode::ChecksumMismatch,
        "invalid import reference sort run",
    )
}

fn failed_sort() -> Error {
    Error::new(
        ErrorCode::FileFormatError,
        "cannot reuse failed import reference sorting",
    )
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use std::collections::BTreeSet;
    use std::fs;
    use std::io::SeekFrom;
    use std::sync::atomic::{AtomicU64, Ordering};

    pub(in crate::temporary) struct TestDirectory(pub(in crate::temporary) PathBuf);

    impl TestDirectory {
        pub(in crate::temporary) fn new(label: &str) -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "cofferdb-sort-{label}-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }

        pub(in crate::temporary) fn assert_clean(&self, disk: &TemporaryDiskBudget) {
            assert_eq!(disk.used(), 0);
            assert_eq!(fs::read_dir(&self.0).unwrap().count(), 0);
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.0).unwrap();
        }
    }

    pub(super) fn sorter(
        directory: &TestDirectory,
        disk: Arc<TemporaryDiskBudget>,
    ) -> ExternalUniqueKeys {
        ExternalUniqueKeys::new(
            &directory.0,
            disk,
            64 * 1024,
            &JsonOperationContext::standalone(),
            ErrorCode::DuplicateExternalId,
            "duplicate test key",
        )
        .unwrap()
    }

    fn detail<'a>(error: &'a Error, key: &str) -> &'a str {
        error
            .detail()
            .iter()
            .find(|item| item.0 == key)
            .unwrap()
            .1
            .as_str()
    }

    fn merge_fixed_bytes() -> usize {
        3 * IO_BYTES + 2 * std::mem::size_of::<Entry>()
    }

    #[cfg(unix)]
    fn file_management_bytes(directory: &Path) -> usize {
        let path = directory.as_os_str().as_encoded_bytes().len() + 1 + 128;
        2 * path
            + std::mem::size_of::<super::super::budget::PendingCleanupNode>()
            + std::mem::size_of::<std::sync::Mutex<super::super::Cleanup>>()
            + 2 * std::mem::size_of::<usize>()
            + if cfg!(target_os = "linux") { 0 } else { 256 }
    }

    #[test]
    fn preferred_batch_flush_readmits_a_larger_key_before_allocating() {
        use crate::json::WorkLedger;
        let directory = TestDirectory::new("preferred-flush");
        let disk = Arc::new(TemporaryDiskBudget::new(8 * 1024 * 1024));
        for competition in [false, true] {
            let batch = 128 * 1024;
            let minimum = required_workspace(48 * 1024) as usize;
            let limit = if competition {
                batch
                    + directory.0.as_os_str().as_encoded_bytes().len()
                    + file_management_bytes(&directory.0)
            } else {
                4 * batch
            };
            let context =
                JsonOperationContext::new(Arc::new(WorkLedger::with_limits(limit, limit, limit)));
            let mut keys = ExternalUniqueKeys::new(
                &directory.0,
                disk.clone(),
                batch,
                &context,
                ErrorCode::DuplicateExternalId,
                "duplicate",
            )
            .unwrap();
            for id in 0..256u64 {
                let mut key = [0; 128];
                key[..8].copy_from_slice(&id.to_be_bytes());
                keys.push_value(&[&key], id, id * 3).unwrap();
            }
            assert_eq!(keys.workspace, batch);
            let held = Arc::new(std::sync::Mutex::new(None));
            if competition {
                let context = context.clone();
                let held = held.clone();
                // Occupy exactly one byte beyond the pending key's minimum
                // after the old batch releases its optional preferred lease.
                keys.after_push_flush = Some(Box::new(move || {
                    let bytes = batch - minimum + 1;
                    let mut work = context.workspace_reservation();
                    work.grow_to(bytes, bytes, "catalog_build").unwrap();
                    *held.lock().unwrap() = Some(work);
                }));
            }
            let key = vec![255; 48 * 1024];
            let result = keys.push_value(&[&key], 999, 1001);
            if competition {
                let error = result.unwrap_err();
                assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                assert_eq!(detail(&error, "resource"), "memory");
                assert_eq!(detail(&error, "requested_bytes"), (limit + 1).to_string());
                assert!(keys.entries.is_empty());
                assert_eq!(
                    keys.levels
                        .iter()
                        .flatten()
                        .map(|run| run.count)
                        .sum::<u64>(),
                    256
                );
                assert!(keys.push(&[b"poisoned"], 0).is_err());
                drop(keys);
                drop(held.lock().unwrap().take());
            } else {
                result.unwrap();
                assert_eq!(keys.workspace, minimum);
                let mut lookup = keys.finish_lookup().unwrap();
                assert_eq!(lookup.get(&key).unwrap(), Some((999, 1001)));
                for id in 0..256u64 {
                    let mut key = [0; 128];
                    key[..8].copy_from_slice(&id.to_be_bytes());
                    assert_eq!(lookup.get(&key).unwrap(), Some((id, id * 3)));
                }
                drop(lookup);
            }
            context
                .workspace_reservation()
                .grow_to(limit, limit, "catalog_build")
                .unwrap();
            directory.assert_clean(&disk);
        }
    }

    #[cfg(not(unix))]
    fn file_management_bytes(_: &Path) -> usize {
        0
    }

    #[test]
    fn stage_key_admission_is_exact_with_shared_competition_and_old_runs() {
        use crate::json::{JsonInputKind, JsonProcessingPolicyV1};
        let directory = TestDirectory::new("stage-admission");
        let disk = Arc::new(TemporaryDiskBudget::new(8 * 1024 * 1024));
        let limit = JsonProcessingPolicyV1::WORK_BUDGET_BYTES;
        let directory_bytes = directory.0.as_os_str().as_encoded_bytes().len();
        let file_bytes = file_management_bytes(&directory.0);
        for seed in 0..100usize {
            let size = (MIN_WORKSPACE - merge_fixed_bytes()) / 2 + 1024 + seed;
            let minimum = 2 * size + merge_fixed_bytes();
            let key = vec![17; size];
            for available in [minimum - 1, minimum, minimum + 1] {
                let context = JsonOperationContext::standalone();
                let mut blocker = context.reservation();
                blocker
                    .grow(
                        limit - available - directory_bytes - file_bytes,
                        JsonInputKind::JsonlRecord,
                    )
                    .unwrap();
                let mut keys = ExternalUniqueKeys::new(
                    &directory.0,
                    disk.clone(),
                    limit,
                    &context,
                    ErrorCode::DuplicateExternalId,
                    "duplicate",
                )
                .unwrap();
                keys.push_value(&[b"old"], 3, 5).unwrap();
                let result = keys.push_value(&[&key], 7, 11);
                if available < minimum {
                    let error = result.unwrap_err();
                    assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                    assert_eq!(detail(&error, "resource"), "memory");
                    assert_eq!(detail(&error, "limit_bytes"), limit.to_string());
                    assert_eq!(detail(&error, "requested_bytes"), (limit + 1).to_string());
                    assert!(keys.entries.is_empty());
                    assert_eq!(
                        keys.levels
                            .iter()
                            .flatten()
                            .map(|run| run.count)
                            .sum::<u64>(),
                        1
                    );
                    drop(keys);
                } else {
                    result.unwrap();
                    assert_eq!(keys.workspace, minimum);
                    blocker.release(2 * file_bytes + 128);
                    let mut lookup = keys.finish_lookup().unwrap();
                    assert_eq!(lookup.get(b"old").unwrap(), Some((3, 5)));
                    assert_eq!(lookup.get(&key).unwrap(), Some((7, 11)));
                    assert_eq!(lookup.get(&key[..key.len() - 1]).unwrap(), None);
                    drop(lookup);
                    blocker
                        .grow(2 * file_bytes + 128, JsonInputKind::JsonlRecord)
                        .unwrap();
                }
                blocker
                    .grow(
                        available + directory_bytes + file_bytes,
                        JsonInputKind::JsonlRecord,
                    )
                    .unwrap();
                directory.assert_clean(&disk);
            }
        }
    }

    #[test]
    fn multi_megabyte_keys_merge_and_lookup_without_eight_slot_rejection() {
        let directory = TestDirectory::new("multi-megabyte-keys");
        let disk = Arc::new(TemporaryDiskBudget::new(256 * 1024 * 1024));
        for size in [5 * 1024 * 1024, 5 * 1024 * 1024 + 8191, 8 * 1024 * 1024 - 9] {
            let input = (0..7u64)
                .map(|id| {
                    let mut key = vec![31; size];
                    key[size - 8..].copy_from_slice(&id.to_be_bytes());
                    (key, id + 13, id * 7)
                })
                .collect::<Vec<_>>();
            let context = JsonOperationContext::standalone();
            let mut keys = ExternalUniqueKeys::new(
                &directory.0,
                disk.clone(),
                crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
                &context,
                ErrorCode::DuplicateExternalId,
                "duplicate",
            )
            .unwrap();
            let (mut lookup, observed) = crate::test_alloc::key_allocations(size, || {
                for ordinal in [6, 1, 4, 0, 5, 2, 3] {
                    let (key, line, value) = &input[ordinal];
                    let middle = key.len() / 2;
                    keys.push_value(&[&key[..middle], &key[middle..]], *line, *value)
                        .unwrap();
                    keys.flush_run().unwrap();
                }
                keys.finish_lookup().unwrap()
            });
            assert_eq!(observed.live, 0);
            assert_eq!(observed.peak, 2);
            assert!(observed.allocated > input.len());
            assert!(!observed.overflow);
            for ordinal in [0, 6, 2, 5, 1, 4, 3, 3] {
                let (key, line, value) = &input[ordinal];
                assert_eq!(lookup.get(key).unwrap(), Some((*line, *value)));
                assert_eq!(lookup.get(&key[..key.len() - 1]).unwrap(), None);
            }
            drop(lookup);
            context
                .workspace_reservation()
                .grow_to(
                    crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
                    crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
                    "catalog_build",
                )
                .unwrap();
            directory.assert_clean(&disk);
        }
    }

    #[test]
    fn shared_workspace_falls_back_and_lookup_retains_its_lease_at_100_boundaries() {
        use crate::json::{JsonInputKind, JsonProcessingPolicyV1};
        let directory = TestDirectory::new("shared-workspace");
        let disk = Arc::new(TemporaryDiskBudget::new(4 * 1024 * 1024));
        let file_bytes = file_management_bytes(&directory.0);
        let management = directory.0.as_os_str().as_encoded_bytes().len() + 3 * file_bytes + 128;
        for extra in 0..100 {
            let context = JsonOperationContext::standalone();
            let mut json = context.reservation();
            json.grow(
                JsonProcessingPolicyV1::WORK_BUDGET_BYTES - 64 * 1024 - management - extra,
                JsonInputKind::JsonlRecord,
            )
            .unwrap();
            let mut keys = ExternalUniqueKeys::new(
                &directory.0,
                disk.clone(),
                1024 * 1024,
                &context,
                ErrorCode::DuplicateExternalId,
                "duplicate",
            )
            .unwrap();
            assert_eq!(keys.workspace, 64 * 1024);
            for number in (0..128u64).rev() {
                keys.push_value(&[&number.to_le_bytes(), &[42; 256]], number + 1, number * 7)
                    .unwrap();
            }
            let mut lookup = keys.finish_lookup().unwrap();
            json.grow(
                extra + management - 2 * file_bytes,
                JsonInputKind::JsonlRecord,
            )
            .unwrap();
            assert!(json.grow(1, JsonInputKind::JsonlRecord).is_err());
            for number in 0..128u64 {
                let mut key = number.to_le_bytes().to_vec();
                key.extend_from_slice(&[42; 256]);
                assert_eq!(lookup.get(&key).unwrap(), Some((number + 1, number * 7)));
            }
            drop(lookup);
            json.grow(64 * 1024 + 2 * file_bytes, JsonInputKind::JsonlRecord)
                .unwrap();
            directory.assert_clean(&disk);
        }
    }

    #[test]
    fn shrunken_sort_workspace_rejects_large_keys_as_shared_memory_not_catalog() {
        use crate::json::{JsonInputKind, JsonProcessingPolicyV1};
        let directory = TestDirectory::new("shared-key-minimum");
        let disk = Arc::new(TemporaryDiskBudget::new(4 * 1024 * 1024));
        let directory_bytes = directory.0.as_os_str().as_encoded_bytes().len();
        for extra in 1..=100 {
            let context = JsonOperationContext::standalone();
            let mut json = context.reservation();
            json.grow(
                JsonProcessingPolicyV1::WORK_BUDGET_BYTES - 64 * 1024 - directory_bytes,
                JsonInputKind::JsonlRecord,
            )
            .unwrap();
            let mut keys = ExternalUniqueKeys::new(
                &directory.0,
                disk.clone(),
                1024 * 1024,
                &context,
                ErrorCode::DuplicateExternalId,
                "duplicate",
            )
            .unwrap();
            let key = vec![17; (64 * 1024 - merge_fixed_bytes()) / 2 + extra];
            let error = keys.push(&[&key], 1).unwrap_err();
            assert_eq!(detail(&error, "resource"), "memory");
            assert_eq!(detail(&error, "operation"), "catalog_build");
            assert_eq!(
                detail(&error, "limit_bytes"),
                JsonProcessingPolicyV1::WORK_BUDGET_BYTES.to_string()
            );
            assert_eq!(
                detail(&error, "requested_bytes"),
                (JsonProcessingPolicyV1::WORK_BUDGET_BYTES + extra * 2).to_string()
            );
            assert!(keys.entries.is_empty());
            drop(keys);
            json.grow(64 * 1024 + directory_bytes, JsonInputKind::JsonlRecord)
                .unwrap();
            directory.assert_clean(&disk);
        }
    }

    #[test]
    fn shrunken_sort_can_grow_for_required_keys_and_release_unused_lookup_space() {
        use crate::json::{JsonInputKind, JsonProcessingPolicyV1};
        let directory = TestDirectory::new("shared-key-growth");
        let disk = Arc::new(TemporaryDiskBudget::new(4 * 1024 * 1024));
        let context = JsonOperationContext::standalone();
        let directory_bytes = directory.0.as_os_str().as_encoded_bytes().len();
        let file_bytes = file_management_bytes(&directory.0);
        let mut json = context.reservation();
        json.grow(
            JsonProcessingPolicyV1::WORK_BUDGET_BYTES - 64 * 1024 - directory_bytes,
            JsonInputKind::JsonlRecord,
        )
        .unwrap();
        let mut keys = ExternalUniqueKeys::new(
            &directory.0,
            disk.clone(),
            1024 * 1024,
            &context,
            ErrorCode::DuplicateExternalId,
            "duplicate",
        )
        .unwrap();
        assert_eq!(keys.workspace, 64 * 1024);
        json.release(1024 * 1024);
        let size = (64 * 1024 - merge_fixed_bytes()) / 2 + 1;
        let key = vec![31; size];
        keys.push_value(&[&key], 7, 11).unwrap();
        let required = 2 * size + merge_fixed_bytes();
        assert_eq!(keys.workspace, required);
        let mut lookup = keys.finish_lookup().unwrap();
        assert_eq!(lookup.get(&key).unwrap(), Some((7, 11)));
        json.grow(
            1024 * 1024 + directory_bytes - 2 * file_bytes,
            JsonInputKind::JsonlRecord,
        )
        .unwrap();
        assert!(json.grow(1, JsonInputKind::JsonlRecord).is_err());
        drop(lookup);
        json.grow(64 * 1024 + 2 * file_bytes, JsonInputKind::JsonlRecord)
            .unwrap();
        directory.assert_clean(&disk);
    }

    #[test]
    fn empty_or_failed_sort_returns_workspace_and_disk_ownership() {
        use crate::json::{JsonInputKind, JsonProcessingPolicyV1};
        let directory = TestDirectory::new("shared-sort-cleanup");
        let disk = Arc::new(TemporaryDiskBudget::new(4 * 1024 * 1024));
        let context = JsonOperationContext::standalone();
        for duplicate in [false, true] {
            let mut keys = ExternalUniqueKeys::new(
                &directory.0,
                disk.clone(),
                1024 * 1024,
                &context,
                ErrorCode::DuplicateExternalId,
                "duplicate",
            )
            .unwrap();
            let lookup = if duplicate {
                keys.push(&[b"same"], 1).unwrap();
                keys.push(&[b"same"], 2).unwrap();
                assert_eq!(
                    keys.finish_lookup().err().unwrap().code(),
                    ErrorCode::DuplicateExternalId
                );
                None
            } else {
                Some(keys.finish_lookup().unwrap())
            };
            let mut json = context.reservation();
            json.grow(
                JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
                JsonInputKind::JsonlRecord,
            )
            .unwrap();
            drop(lookup);
            directory.assert_clean(&disk);
        }
    }

    #[test]
    fn bounded_runs_match_unique_key_oracle_for_100_shuffled_inputs() {
        let directory = TestDirectory::new("oracle");
        let disk = Arc::new(TemporaryDiskBudget::new(8 * 1024 * 1024));
        for seed in 0..100u64 {
            let mut state = seed + 1;
            let mut random = || {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state
            };
            let mut input: Vec<_> = (0..1000u64)
                .map(|number| {
                    let mut key = number.to_be_bytes().to_vec();
                    key.resize(32 + (random() % 200) as usize, (random() % 256) as u8);
                    (key, number + 1)
                })
                .collect();
            input.push((Vec::new(), 1001));
            input.push(("UTF-8:\u{4e2d}\u{6587}\0".as_bytes().to_vec(), 1002));
            let expected: BTreeSet<_> = input.iter().cloned().collect();
            for index in (1..input.len()).rev() {
                let other = (random() as usize) % (index + 1);
                input.swap(index, other);
            }
            let mut keys = sorter(&directory, disk.clone());
            for (key, line) in input {
                let middle = key.len() / 2;
                keys.push(&[&key[..middle], &key[middle..]], line).unwrap();
                assert!(
                    keys.key_bytes + keys.entries.capacity() * std::mem::size_of::<Entry>()
                        <= keys.workspace / 2
                );
            }
            keys.flush_run().unwrap();
            assert!(keys.levels.iter().skip(1).any(Option::is_some));
            assert_eq!(
                keys.levels
                    .iter()
                    .flatten()
                    .map(|run| run.count)
                    .sum::<u64>(),
                1002
            );
            let mut combined = None;
            for level in 0..keys.levels.len() {
                if let Some(run) = keys.levels[level].take() {
                    combined = Some(match combined {
                        Some(previous) => keys.merge(previous, run).unwrap(),
                        None => run,
                    });
                }
            }
            let mut cursor = RunCursor::new(combined.unwrap(), keys.largest_key).unwrap();
            for (key, line) in expected {
                let actual = cursor.next().unwrap().unwrap();
                assert_eq!((actual.key, actual.line), (key, line), "seed={seed}");
            }
            assert!(cursor.next().unwrap().is_none());
            drop(cursor);
            keys.finish().unwrap();
            directory.assert_clean(&disk);
        }
    }

    #[test]
    fn merge_releases_written_heads_before_allocating_the_next_key() {
        let directory = TestDirectory::new("two-merge-heads");
        let disk = Arc::new(TemporaryDiskBudget::new(8 * 1024 * 1024));
        for seed in 0..100usize {
            let size = 65537 + seed;
            let workspace = size * 8;
            let context = JsonOperationContext::standalone();
            let mut keys = ExternalUniqueKeys::new(
                &directory.0,
                disk.clone(),
                workspace,
                &context,
                ErrorCode::DuplicateExternalId,
                "duplicate",
            )
            .unwrap();
            keys.workspace = keys
                .work
                .grow_to(workspace, workspace, "catalog_build")
                .unwrap();
            keys.largest_key = size;
            let left_count = 2 + seed % 7;
            let right_count = 1 + seed % 5;
            let make_run = |side: usize, count: usize, disk: &Arc<TemporaryDiskBudget>| {
                let mut output = RunWriter::new(&directory.0, disk.clone(), &context).unwrap();
                for ordinal in 0..count {
                    let id = (ordinal * 2 + side) as u64;
                    let mut key = vec![17; size];
                    key[size - 8..].copy_from_slice(&id.to_be_bytes());
                    output
                        .write_entry(&Entry {
                            key,
                            line: id + 1,
                            value: id * 3,
                        })
                        .unwrap();
                }
                output.finish().unwrap()
            };
            let left = make_run(0, left_count, &disk);
            let right = make_run(1, right_count, &disk);
            let (merged, observed) =
                crate::test_alloc::key_allocations(size, || keys.merge(left, right));
            let merged = merged.unwrap();
            assert_eq!(observed.allocated, left_count + right_count);
            assert_eq!(observed.live, 0);
            assert_eq!(observed.peak, 2, "seed {seed}");
            assert!(!observed.overflow);
            let mut expected = (0..left_count)
                .map(|id| (id * 2) as u64)
                .chain((0..right_count).map(|id| (id * 2 + 1) as u64))
                .collect::<Vec<_>>();
            expected.sort_unstable();
            let mut cursor = RunCursor::new(merged, size).unwrap();
            for id in expected {
                let entry = cursor.next().unwrap().unwrap();
                assert_eq!(entry.key.len(), size);
                assert!(entry.key[..size - 8].iter().all(|&byte| byte == 17));
                assert_eq!(&entry.key[size - 8..], &id.to_be_bytes());
                assert_eq!((entry.line, entry.value), (id + 1, id * 3));
            }
            assert!(cursor.next().unwrap().is_none());
            drop(cursor);
            for corrupt in [false, true] {
                let source_bytes =
                    ((left_count + right_count) * (size + ENTRY_HEADER_BYTES)) as u64;
                let failed_disk = Arc::new(TemporaryDiskBudget::new(if corrupt {
                    source_bytes * 3
                } else {
                    source_bytes
                }));
                let mut left = make_run(0, left_count, &failed_disk);
                let right = make_run(1, right_count, &failed_disk);
                assert_eq!(failed_disk.used(), source_bytes);
                keys.disk = failed_disk.clone();
                if corrupt {
                    left.file.seek(SeekFrom::Start(left.bytes - 2)).unwrap();
                    let mut byte = [0];
                    left.file.read_exact(&mut byte).unwrap();
                    byte[0] ^= 1;
                    left.file.seek(SeekFrom::Start(left.bytes - 2)).unwrap();
                    left.file.write_all(&byte).unwrap();
                }
                let (result, observed) =
                    crate::test_alloc::key_allocations(size, || keys.merge(left, right));
                let error = result.err().expect("failed merge");
                assert_eq!(
                    error.code(),
                    if corrupt {
                        ErrorCode::ChecksumMismatch
                    } else {
                        ErrorCode::ResourceExhausted
                    }
                );
                if !corrupt {
                    assert_eq!(detail(&error, "resource"), "temporary_disk");
                }
                assert!(observed.allocated >= 2);
                assert_eq!(observed.live, 0);
                assert_eq!(observed.peak, 2);
                assert!(!observed.overflow);
                directory.assert_clean(&failed_disk);
            }
            drop(keys);
            context
                .workspace_reservation()
                .grow_to(
                    crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
                    crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
                    "catalog_build",
                )
                .unwrap();
            directory.assert_clean(&disk);
        }
    }

    #[test]
    fn duplicate_keys_across_runs_preserve_locations_and_poison_the_sort() {
        let directory = TestDirectory::new("duplicate");
        let disk = Arc::new(TemporaryDiskBudget::new(1024 * 1024));
        for separate_runs in [false, true] {
            let mut keys = sorter(&directory, disk.clone());
            keys.push(&[b"private key"], 2).unwrap();
            if separate_runs {
                keys.flush_run().unwrap();
            }
            keys.push(&[b"private key"], 90).unwrap();
            let error = keys.finish().unwrap_err();
            assert_eq!(error.code(), ErrorCode::DuplicateExternalId);
            assert_eq!(error.message(), "duplicate test key");
            assert_eq!(detail(&error, "first_line"), "2");
            assert_eq!(detail(&error, "line"), "90");
            assert!(!error.to_string().contains("private key"));
            directory.assert_clean(&disk);
        }

        let mut keys = sorter(&directory, disk.clone());
        keys.push(&[&[1; 8192]], 10).unwrap();
        keys.flush_run().unwrap();
        keys.push(&[&[1; 8192]], 20).unwrap();
        keys.push(&[&[2; 8192]], 21).unwrap();
        keys.push(&[&[3; 8192]], 22).unwrap();
        assert_eq!(
            keys.push(&[&[4; 8192]], 23).unwrap_err().code(),
            ErrorCode::DuplicateExternalId
        );
        assert_eq!(
            keys.push(&[b"valid"], 24).unwrap_err().code(),
            ErrorCode::FileFormatError
        );
        assert_eq!(
            keys.finish().unwrap_err().code(),
            ErrorCode::FileFormatError
        );
        directory.assert_clean(&disk);
    }

    #[test]
    fn workspace_rejections_preserve_exact_totals_across_100_key_boundaries() {
        let directory = TestDirectory::new("workspace-diagnostics");
        let disk = Arc::new(TemporaryDiskBudget::new(4 * 1024 * 1024));
        for workspace in [64 * 1024, 256 * 1024, 1024 * 1024] {
            let key_limit = (workspace - merge_fixed_bytes()) / 2;
            let prefix = vec![7; key_limit];
            for extra in 1..=100 {
                let mut keys = ExternalUniqueKeys::new(
                    &directory.0,
                    disk.clone(),
                    workspace,
                    &JsonOperationContext::standalone(),
                    ErrorCode::DuplicateExternalId,
                    "duplicate",
                )
                .unwrap();
                keys.push(&[b"valid"], 1).unwrap();
                let suffix = vec![8; extra];
                let error = keys.push(&[&prefix, &suffix], 2).unwrap_err();
                assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                assert_eq!(detail(&error, "resource"), "catalog");
                assert_eq!(detail(&error, "operation"), "catalog_build");
                assert_eq!(detail(&error, "limit_bytes"), workspace.to_string());
                assert_eq!(
                    detail(&error, "requested_bytes"),
                    (workspace + extra * 2).to_string()
                );
                assert_eq!(
                    error.to_json().get("recoverable"),
                    Some(&crate::json::JsonValue::Bool(true))
                );
                assert!(!crate::json::is_limit_error(&error));
                assert_eq!(
                    keys.entries.len(),
                    1,
                    "rejection must precede key allocation/insertion"
                );
                assert_eq!(
                    keys.finish().unwrap_err().code(),
                    ErrorCode::FileFormatError
                );
                directory.assert_clean(&disk);
            }
        }
        for workspace in [0, 1, 32 * 1024, 65535] {
            let error = ExternalUniqueKeys::new(
                &directory.0,
                disk.clone(),
                workspace,
                &JsonOperationContext::standalone(),
                ErrorCode::DuplicateExternalId,
                "duplicate",
            )
            .err()
            .unwrap();
            assert_eq!(detail(&error, "limit_bytes"), workspace.to_string());
            assert_eq!(detail(&error, "requested_bytes"), "65536");
        }
    }

    #[test]
    fn allocator_failures_do_not_fabricate_controlled_resource_totals() {
        let error = allocation_error(Vec::<u8>::new().try_reserve_exact(usize::MAX).unwrap_err());
        assert_eq!(error.code(), ErrorCode::IoError);
        assert!(error.detail().is_empty());
        assert!(std::error::Error::source(&error).is_some());
    }

    #[test]
    fn exact_disk_and_key_boundaries_release_all_reservations() {
        let directory = TestDirectory::new("boundaries");
        for limit in [0, 66, 67, 1024] {
            let disk = Arc::new(TemporaryDiskBudget::new(limit));
            let mut keys = sorter(&directory, disk.clone());
            for (line, key) in [b"".as_slice(), b"a", b"abcdef"].into_iter().enumerate() {
                keys.push(&[key], line as u64 + 1).unwrap();
            }
            let result = keys.finish();
            if limit < 67 {
                let error = result.unwrap_err();
                assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                assert_eq!(detail(&error, "resource"), "temporary_disk");
            } else {
                result.unwrap();
            }
            directory.assert_clean(&disk);
        }
        let disk = Arc::new(TemporaryDiskBudget::new(1024 * 1024));
        let mut keys = sorter(&directory, disk.clone());
        keys.push(&[&[7; 8192]], 1).unwrap();
        keys.finish().unwrap();
        directory.assert_clean(&disk);
        let mut keys = sorter(&directory, disk.clone());
        let size = (64 * 1024 - merge_fixed_bytes()) / 2 + 1;
        let key = vec![7; size];
        let error = keys.push(&[&key], 1).unwrap_err();
        assert_eq!(error.code(), ErrorCode::ResourceExhausted);
        assert_eq!(detail(&error, "resource"), "catalog");
        assert_eq!(detail(&error, "limit_bytes"), "65536");
        assert_eq!(detail(&error, "requested_bytes"), "65538");
        assert!(keys.entries.is_empty());
        drop(keys);
        directory.assert_clean(&disk);
        assert!(ExternalUniqueKeys::new(
            &directory.0,
            disk.clone(),
            65535,
            &JsonOperationContext::standalone(),
            ErrorCode::DuplicateExternalId,
            "duplicate"
        )
        .is_err());
        sorter(&directory, disk.clone()).finish().unwrap();
        directory.assert_clean(&disk);
    }

    #[test]
    fn cancellation_and_merge_disk_exhaustion_remove_every_run() {
        let directory = TestDirectory::new("cancel");
        let disk = Arc::new(TemporaryDiskBudget::new(1024 * 1024));
        let mut keys = sorter(&directory, disk.clone());
        for number in 0..1000u64 {
            keys.push(&[&number.to_le_bytes(), &[0; 128]], number)
                .unwrap();
        }
        assert!(disk.used() > 0);
        drop(keys);
        directory.assert_clean(&disk);

        // Two 21-byte input runs coexist with their output during the merge.
        for limit in [83, 84] {
            let disk = Arc::new(TemporaryDiskBudget::new(limit));
            let mut keys = sorter(&directory, disk.clone());
            keys.push(&[b"a"], 1).unwrap();
            keys.flush_run().unwrap();
            keys.push(&[b"b"], 2).unwrap();
            assert_eq!(keys.finish().is_ok(), limit == 84);
            directory.assert_clean(&disk);
        }
    }

    #[cfg(unix)]
    #[test]
    fn run_writer_abort_and_disk_refusal_do_not_flush_pending_bytes_on_drop() {
        let directory = TestDirectory::new("writer-abort");
        for seed in 0..100usize {
            for reject in [false, true] {
                let entry = Entry {
                    key: vec![17; seed + 1],
                    line: seed as u64,
                    value: 7,
                };
                let bytes = ENTRY_HEADER_BYTES as u64 + entry.key.len() as u64;
                let disk = Arc::new(TemporaryDiskBudget::new(bytes));
                let context = JsonOperationContext::standalone();
                let limit = context.available_workspace_bytes();
                let mut lease = context.workspace_reservation();
                lease.grow_to(IO_BYTES, IO_BYTES, "catalog_build").unwrap();
                let mut writer = RunWriter::new(&directory.0, disk.clone(), &context).unwrap();
                let observed = writer
                    .output
                    .as_ref()
                    .unwrap()
                    .get_ref()
                    .try_clone()
                    .unwrap();
                writer.write_entry(&entry).unwrap();
                assert_eq!(disk.used(), bytes);
                assert_eq!(observed.metadata().unwrap().len(), 0);
                if reject {
                    let error = writer.write_entry(&entry).unwrap_err();
                    assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                    assert_eq!(writer.count, 1);
                    assert_eq!(writer.bytes, bytes);
                }
                drop(writer);
                assert_eq!(
                    observed.metadata().unwrap().len(),
                    0,
                    "seed={seed}/reject={reject}"
                );
                drop(observed);
                directory.assert_clean(&disk);
                lease.grow_to(limit, limit, "catalog_build").unwrap();
                assert!(lease
                    .grow_to(limit + 1, limit + 1, "catalog_build")
                    .is_err());
            }
        }
    }

    #[test]
    fn run_buffer_finish_preserves_partial_write_and_flush_failure_without_retry() {
        use std::cell::RefCell;
        use std::rc::Rc;
        #[derive(Default)]
        struct State {
            accepted: Vec<u8>,
            writes: usize,
            flushes: usize,
        }
        struct Output {
            state: Rc<RefCell<State>>,
            mode: usize,
        }
        impl Write for Output {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                let mut state = self.state.borrow_mut();
                state.writes += 1;
                if self.mode == 2 || (self.mode == 3 && state.writes == 2) {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::BrokenPipe,
                        "sort write failure",
                    ));
                }
                let count = if matches!(self.mode, 1 | 3) && state.writes == 1 {
                    bytes.len().min(2)
                } else {
                    bytes.len()
                };
                state.accepted.extend_from_slice(&bytes[..count]);
                Ok(count)
            }
            fn flush(&mut self) -> std::io::Result<()> {
                self.state.borrow_mut().flushes += 1;
                if self.mode == 4 {
                    Err(std::io::Error::other("sort flush failure"))
                } else {
                    Ok(())
                }
            }
        }
        for mode in 0..5 {
            let state = Rc::new(RefCell::new(State::default()));
            let mut buffer = BufWriter::with_capacity(
                64,
                Output {
                    state: state.clone(),
                    mode,
                },
            );
            buffer.write_all(b"pending entry").unwrap();
            assert_eq!(state.borrow().writes, 0);
            let result = finish_run_buffer(buffer);
            if mode < 2 {
                drop(result.unwrap());
            } else {
                let error = result.err().unwrap();
                assert_eq!(
                    error.kind(),
                    if mode == 4 {
                        std::io::ErrorKind::Other
                    } else {
                        std::io::ErrorKind::BrokenPipe
                    }
                );
                assert_eq!(
                    error.to_string(),
                    if mode == 4 {
                        "sort flush failure"
                    } else {
                        "sort write failure"
                    }
                );
            }
            let state = state.borrow();
            assert_eq!(state.writes, if matches!(mode, 1 | 3) { 2 } else { 1 });
            assert_eq!(state.flushes, usize::from(mode < 2 || mode == 4));
            assert_eq!(
                state.accepted.as_slice(),
                match mode {
                    2 => &[][..],
                    3 => b"pe",
                    _ => b"pending entry",
                }
            );
        }
    }

    #[test]
    fn corrupt_runs_are_rejected_before_trusting_lengths_and_cleaned_up() {
        let directory = TestDirectory::new("corrupt");
        let disk = Arc::new(TemporaryDiskBudget::new(1024 * 1024));
        for damage in 0..6 {
            let mut keys = sorter(&directory, disk.clone());
            keys.push(&[b"payload"], 9).unwrap();
            keys.flush_run().unwrap();
            let run = keys.levels.iter_mut().flatten().next().unwrap();
            match damage {
                0 => run.file.set_len(run.bytes - 1).unwrap(),
                1 => run.file.set_len(run.bytes + 1).unwrap(),
                2 => {
                    run.file
                        .seek(SeekFrom::Start(ENTRY_HEADER_BYTES as u64))
                        .unwrap();
                    run.file.write_all(b"X").unwrap();
                }
                3 => {
                    run.file.rewind().unwrap();
                    run.file.write_all(&u32::MAX.to_le_bytes()).unwrap();
                }
                4 => run.count += 1,
                _ => run.count -= 1,
            }
            assert_eq!(
                keys.finish().unwrap_err().code(),
                ErrorCode::ChecksumMismatch,
                "damage={damage}"
            );
            directory.assert_clean(&disk);
        }
    }
}
