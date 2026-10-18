use std::fs::{self, File, OpenOptions};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::Result;

mod budget;
mod path;
mod sort;
pub(crate) use budget::TemporaryDiskBudget;
pub(crate) use path::TemporaryPath;
pub(crate) use sort::{ExternalKeyLookup, ExternalTermFrequencies, ExternalUniqueKeys};

static NEXT_ID: AtomicU64 = AtomicU64::new(0);
#[cfg(unix)]
const MIN_NAME_CAPACITY: usize = 128;
// Linux uses an inline futex. Other supported Unix builds lazily allocate
// native mutex storage; this allowance is conservative, not an ABI size.
#[cfg(unix)]
const CLEANUP_LOCK_WORK_BYTES: usize = if cfg!(target_os = "linux") { 0 } else { 256 };

#[derive(Debug)]
pub(crate) struct TempFileGuard {
    path: TemporaryPath,
    cleanup: SharedCleanup,
}

#[derive(Clone, Debug)]
struct SharedCleanup {
    state: Option<Arc<Mutex<Cleanup>>>,
}

impl std::ops::Deref for SharedCleanup {
    type Target = Mutex<Cleanup>;

    fn deref(&self) -> &Mutex<Cleanup> {
        self.state.as_ref().expect("live cleanup owner")
    }
}

impl Drop for SharedCleanup {
    fn drop(&mut self) {
        // All strong owners use into_inner: even concurrent drops elect one
        // winner and release the Arc allocation before returning its credit.
        if let Some(state) = self.state.take().and_then(Arc::into_inner) {
            let mut cleanup = state
                .into_inner()
                .unwrap_or_else(|error| error.into_inner());
            let work = cleanup._state_work.take();
            drop(cleanup);
            drop(work);
        }
    }
}

#[derive(Debug)]
struct Cleanup {
    path: TemporaryPath,
    file: Option<File>,
    disk: Option<budget::DiskReservation>,
    pending: Option<budget::PendingCleanupOwner>,
    _state_work: Option<crate::json::WorkspaceReservation>,
}

// The file and its guard share cleanup ownership, so either drop order keeps
// the disk reservation until the last handle is closed and cleanup succeeds.
#[derive(Debug)]
pub(crate) struct TemporaryFile {
    file: Option<File>,
    cleanup: SharedCleanup,
}

impl std::ops::Deref for TemporaryFile {
    type Target = File;

    fn deref(&self) -> &File {
        self.file.as_ref().expect("live temporary file")
    }
}

impl std::ops::DerefMut for TemporaryFile {
    fn deref_mut(&mut self) -> &mut File {
        self.file.as_mut().expect("live temporary file")
    }
}

impl std::io::Read for TemporaryFile {
    fn read(&mut self, bytes: &mut [u8]) -> std::io::Result<usize> {
        self.file.as_mut().unwrap().read(bytes)
    }
}

impl std::io::Write for TemporaryFile {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.file.as_mut().unwrap().write(bytes)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.file.as_mut().unwrap().flush()
    }
}

impl std::io::Seek for TemporaryFile {
    fn seek(&mut self, position: std::io::SeekFrom) -> std::io::Result<u64> {
        self.file.as_mut().unwrap().seek(position)
    }
}

impl Drop for TemporaryFile {
    fn drop(&mut self) {
        self.cleanup
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .file = self.file.take();
    }
}

#[cfg(any(test, not(unix)))]
pub(crate) fn create_budgeted_temporary_file_in(
    directory: &Path,
    label: &str,
    extension: &str,
    budget: Arc<TemporaryDiskBudget>,
) -> Result<(TemporaryFile, TempFileGuard)> {
    // Do not accumulate more abandoned files while earlier cleanup still fails.
    budget.retry_cleanup()?;
    // Cleanup runs in Drop: reserve its transfer node before creating a file.
    let pending = budget::PendingCleanupOwner::default();
    let (file, guard) = create_temporary_file_in(directory, label, extension)?;
    {
        let mut cleanup = guard
            .cleanup
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        cleanup.disk = Some(budget.reservation());
        cleanup.pending = Some(pending);
    }
    let file = TemporaryFile {
        file: Some(file),
        cleanup: guard.cleanup.clone(),
    };
    Ok((file, guard))
}

fn create_budgeted_sort_file(
    directory: &Path,
    label: &'static str,
    budget: Arc<TemporaryDiskBudget>,
    context: &crate::json::JsonOperationContext,
) -> Result<(TemporaryFile, TempFileGuard)> {
    create_budgeted_temporary_file_in_context(
        directory,
        label,
        "tmp",
        budget,
        context,
        "catalog_build",
    )
}

pub(crate) fn create_budgeted_temporary_file_in_context(
    directory: &Path,
    label: &str,
    extension: &str,
    budget: Arc<TemporaryDiskBudget>,
    context: &crate::json::JsonOperationContext,
    phase: &'static str,
) -> Result<(TemporaryFile, TempFileGuard)> {
    #[cfg(not(unix))]
    {
        let _ = (context, phase);
        return create_budgeted_temporary_file_in(directory, label, extension, budget);
    }
    #[cfg(unix)]
    {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_nanos())
            .unwrap_or(0);
        create_file_with_names(
            directory,
            budget,
            context,
            phase,
            name_capacity(label, extension)?,
            || {
                let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
                budgeted_filename(
                    label,
                    extension,
                    (std::process::id(), nonce, id),
                    context,
                    phase,
                )
            },
        )
    }
}

pub(crate) fn create_budgeted_temporary_file_reclaiming(
    directory: &Path,
    label: &str,
    extension: &str,
    budget: Arc<TemporaryDiskBudget>,
    context: &crate::json::JsonOperationContext,
    phase: &'static str,
    reclaim: impl FnOnce() -> bool,
) -> Result<(TemporaryFile, TempFileGuard)> {
    let create = || {
        create_budgeted_temporary_file_in_context(
            directory,
            label,
            extension,
            budget.clone(),
            context,
            phase,
        )
    };
    match create() {
        Err(error) if crate::json::is_memory_budget_error(&error) => {
            // Failed admission owners are gone; this file has not been opened.
            if reclaim() {
                create()
            } else {
                Err(error)
            }
        }
        result => result,
    }
}

#[cfg(unix)]
fn name_capacity(label: &str, extension: &str) -> Result<usize> {
    // Prefix/separators plus maximum decimal u32/u128/u64 identifier widths.
    82usize
        .checked_add(label.len())
        .and_then(|bytes| bytes.checked_add(extension.len()))
        .map(|bytes| bytes.max(MIN_NAME_CAPACITY))
        .ok_or_else(|| std::io::Error::other("temporary name capacity overflow").into())
}

#[cfg(unix)]
fn budgeted_filename(
    label: &str,
    extension: &str,
    (pid, nonce, id): (u32, u128, u64),
    context: &crate::json::JsonOperationContext,
    phase: &'static str,
) -> Result<crate::json::ReservedString> {
    let mut name = crate::json::ReservedString::with_capacity(
        name_capacity(label, extension)?,
        context,
        phase,
    )?;
    name.append(|output| write!(output, "cofferdb-{label}-{pid}-{nonce}-{id}.{extension}"))?;
    Ok(name)
}

#[cfg(unix)]
fn create_file_with_names(
    directory: &Path,
    budget: Arc<TemporaryDiskBudget>,
    context: &crate::json::JsonOperationContext,
    phase: &'static str,
    name_capacity: usize,
    mut next_name: impl FnMut() -> Result<crate::json::ReservedString>,
) -> Result<(TemporaryFile, TempFileGuard)> {
    use std::os::unix::fs::OpenOptionsExt;
    budget.retry_cleanup()?;
    let pending = budget::PendingCleanupOwner::new(context, phase)?;
    let capacity = directory
        .as_os_str()
        .as_encoded_bytes()
        .len()
        .checked_add(1)
        .and_then(|bytes| bytes.checked_add(name_capacity))
        .ok_or_else(|| std::io::Error::other("temporary path capacity overflow"))?;
    loop {
        let name = next_name()?;
        let path = TemporaryPath::join_name(directory, name.as_ref(), capacity, context, phase)?;
        let cleanup_path = path.copy_reserved(context, phase)?;
        let bytes = std::mem::size_of::<Mutex<Cleanup>>()
            + 2 * std::mem::size_of::<usize>()
            + CLEANUP_LOCK_WORK_BYTES;
        let mut work = context.workspace_reservation();
        work.grow_to(bytes, bytes, phase)?;
        let mut state = Mutex::new(Cleanup {
            path: TemporaryPath::default(),
            file: None,
            disk: None,
            pending: None,
            _state_work: Some(work),
        });
        // Initialize the native lock before opening a file or sharing owners,
        // so Drop never needs a cold-lock allocation or initialization race.
        drop(state.lock().unwrap_or_else(|error| error.into_inner()));
        let mut options = OpenOptions::new();
        options.read(true).write(true).create_new(true).mode(0o600);
        match options.open(&path) {
            Ok(file) => {
                // Only a successfully created file activates cleanup ownership.
                let owner = state.get_mut().unwrap_or_else(|error| error.into_inner());
                owner.path = cleanup_path;
                owner.disk = Some(budget.reservation());
                owner.pending = Some(pending);
                let cleanup = SharedCleanup {
                    state: Some(Arc::new(state)),
                };
                let guard = TempFileGuard {
                    path,
                    cleanup: cleanup.clone(),
                };
                return Ok((
                    TemporaryFile {
                        file: Some(file),
                        cleanup,
                    },
                    guard,
                ));
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error.into()),
        }
    }
}

pub(crate) fn create_temporary_file_in(
    directory: &Path,
    label: &str,
    extension: &str,
) -> Result<(File, TempFileGuard)> {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    loop {
        let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        let path = directory.join(format!(
            "cofferdb-{label}-{}-{nonce}-{id}.{extension}",
            std::process::id()
        ));
        let mut options = OpenOptions::new();
        options.read(true).write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        match options.open(&path) {
            Ok(file) => return Ok((file, TempFileGuard::new(path))),
            Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(err) => return Err(err.into()),
        }
    }
}

impl TempFileGuard {
    pub(crate) fn new(path: PathBuf) -> Self {
        Self {
            cleanup: SharedCleanup {
                state: Some(Arc::new(Mutex::new(Cleanup {
                    path: TemporaryPath::unreserved(path.clone()),
                    file: None,
                    disk: None,
                    pending: None,
                    _state_work: None,
                }))),
            },
            path: TemporaryPath::unreserved(path),
        }
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    pub(crate) fn reserve_disk(&mut self, bytes: u64) -> Result<()> {
        let mut cleanup = self
            .cleanup
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        match &mut cleanup.disk {
            Some(work) => work.grow(bytes),
            None => Ok(()),
        }
    }

    pub(crate) fn release_disk(&mut self, bytes: u64) {
        let mut cleanup = self
            .cleanup
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if let Some(work) = &mut cleanup.disk {
            work.release(bytes);
        }
    }

    pub(crate) fn transfer_disk_to(&mut self, destination: &mut Self, bytes: u64) -> Result<()> {
        if bytes == 0 {
            return Ok(());
        }
        // Guards are not cloneable, so mutable ownership excludes opposite transfers.
        let mut source = self
            .cleanup
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let mut destination = destination
            .cleanup
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        match (&mut source.disk, &mut destination.disk) {
            (Some(source), Some(destination)) => source.transfer_to(destination, bytes),
            _ => Err(crate::Error::new(
                crate::ErrorCode::FileFormatError,
                "temporary disk transfer requires budgeted files",
            )),
        }
    }

    pub(crate) fn reserve_disk_buffer(&mut self, minimum: u64, preferred: u64) -> Result<u64> {
        let mut cleanup = self
            .cleanup
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        match &mut cleanup.disk {
            Some(work) => work.grow_at_least(minimum, preferred),
            None => Ok(preferred.max(minimum)),
        }
    }

    pub(crate) fn persist(self, destination: &Path) -> Result<()> {
        fs::rename(&self.path, destination)?;
        // Do not remove a different file subsequently created at the temporary name.
        self.cleanup
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .path
            .clear();
        Ok(())
    }
}

impl Cleanup {
    fn cleanup_with(&mut self, remove: impl FnOnce(&Path) -> std::io::Result<()>) {
        if !self.path.as_os_str().is_empty() {
            if let Err(error) = remove(&self.path) {
                if error.kind() != std::io::ErrorKind::NotFound {
                    if let Some(disk) = self.disk.take() {
                        disk.retain_after_cleanup_failure(
                            std::mem::take(&mut self.path),
                            self.file.take().expect("budgeted temporary file owner"),
                            self.pending.take().expect("budgeted cleanup transfer node"),
                        );
                    }
                }
            }
        }
    }
}

impl Drop for Cleanup {
    fn drop(&mut self) {
        if let Some(file) = self.file.take() {
            let result = remove_owned_file(&file, &self.path);
            self.file = Some(file);
            self.cleanup_with(|_| result);
        } else {
            self.cleanup_with(|path| fs::remove_file(path));
        }
    }
}

fn remove_owned_file(file: &File, path: &Path) -> std::io::Result<()> {
    if path.as_os_str().is_empty() {
        return Ok(());
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let owned = file.metadata()?;
        // The pinned descriptor prevents inode reuse. An already unlinked
        // original can be closed without touching a replacement at this name.
        if owned.nlink() == 0 {
            return Ok(());
        }
        let named = match fs::symlink_metadata(path) {
            Ok(named) => named,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                if file.metadata()?.nlink() == 0 {
                    return Ok(());
                }
                return Err(std::io::Error::other(
                    "temporary file was moved but remains linked",
                ));
            }
            Err(error) => return Err(error),
        };
        if owned.dev() != named.dev() || owned.ino() != named.ino() {
            return Err(std::io::Error::other(
                "temporary file name has changed owner",
            ));
        }
    }
    if let Err(error) = fs::remove_file(path) {
        #[cfg(unix)]
        if error.kind() == std::io::ErrorKind::NotFound {
            use std::os::unix::fs::MetadataExt;
            return if file.metadata()?.nlink() == 0 {
                Ok(())
            } else {
                Err(std::io::Error::other(
                    "temporary file was moved but remains linked",
                ))
            };
        }
        return Err(error);
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if file.metadata()?.nlink() != 0 {
            return Err(std::io::Error::other(
                "temporary file still has another link",
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
pub(crate) fn test_file_management_bytes(directory: &Path) -> usize {
    #[cfg(unix)]
    {
        let path = directory.as_os_str().as_encoded_bytes().len() + 1 + 128;
        2 * path
            + std::mem::size_of::<budget::PendingCleanupNode>()
            + std::mem::size_of::<Mutex<Cleanup>>()
            + 2 * std::mem::size_of::<usize>()
            + if cfg!(target_os = "linux") { 0 } else { 256 }
    }
    #[cfg(not(unix))]
    {
        let _ = directory;
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json::{JsonOperationContext, WorkLedger};
    use std::io::Write;

    #[cfg(unix)]
    fn sort_owner_sizes(directory: &Path) -> (usize, usize, usize) {
        (
            directory.as_os_str().as_encoded_bytes().len() + 1 + 128,
            std::mem::size_of::<budget::PendingCleanupNode>(),
            std::mem::size_of::<Mutex<Cleanup>>()
                + 2 * std::mem::size_of::<usize>()
                + if cfg!(target_os = "linux") { 0 } else { 256 },
        )
    }

    fn limited_context(limit: usize) -> JsonOperationContext {
        JsonOperationContext::new(Arc::new(WorkLedger::with_limits(limit, limit, limit)))
    }

    fn assert_available(context: &JsonOperationContext, bytes: usize) {
        let mut spare = context.workspace_reservation();
        spare.grow_to(bytes, bytes, "catalog_build").unwrap();
        assert_eq!(
            spare
                .grow_to(bytes + 1, bytes + 1, "catalog_build")
                .unwrap_err()
                .code(),
            crate::ErrorCode::ResourceExhausted
        );
    }

    #[test]
    #[cfg(unix)]
    fn generic_file_management_uses_original_context_phase_and_parent_total() {
        let directory = sort::tests::TestDirectory::new("generic-management");
        let management = test_file_management_bytes(&directory.0);
        let peak = management + 128;
        for phase in ["catalog_build", "batch_prepare", "operation_scratch"] {
            for label in [
                "compact",
                "import-records",
                "import-input",
                "export-embeddings",
            ] {
                for available in [peak - 1, peak, peak + 1] {
                    let total = available + 7;
                    let ledger = Arc::new(WorkLedger::with_limits(3 * peak, 3 * peak, total));
                    let context = JsonOperationContext::new(ledger.clone());
                    let sibling = JsonOperationContext::new(ledger);
                    let cached = sibling.cache_reservation(7, total).unwrap();
                    let disk = Arc::new(TemporaryDiskBudget::new(4096));
                    let result = create_budgeted_temporary_file_in_context(
                        &directory.0,
                        label,
                        "tmp",
                        disk.clone(),
                        &context,
                        phase,
                    );
                    if available < peak {
                        let error = result.unwrap_err();
                        assert_eq!(error.code(), crate::ErrorCode::ResourceExhausted);
                        for (key, value) in [
                            ("resource", "memory".to_owned()),
                            ("operation", phase.to_owned()),
                            ("limit_bytes", total.to_string()),
                            ("requested_bytes", (peak + 7).to_string()),
                        ] {
                            assert!(
                                error.detail().contains(&(key.to_owned(), value)),
                                "{error:?}"
                            );
                        }
                        drop(context);
                    } else {
                        let owners = result.unwrap();
                        drop(context);
                        assert_available(&sibling, available - management);
                        drop(owners);
                    }
                    assert_available(&sibling, available);
                    drop(cached);
                    assert_available(&sibling, total);
                    directory.assert_clean(&disk);
                }
            }
        }
    }

    #[test]
    #[cfg(unix)]
    fn file_creation_reclaims_once_without_repeating_prior_files() {
        let directory = sort::tests::TestDirectory::new("creation-reclaim");
        let management = test_file_management_bytes(&directory.0);
        let limit = 4 * (management + 128);
        for phase in ["catalog_build", "batch_prepare", "operation_scratch"] {
            for target in 1..=4 {
                let ledger = Arc::new(WorkLedger::with_limits(2 * limit, limit, limit));
                let context = JsonOperationContext::new(ledger.clone());
                let sibling = JsonOperationContext::new(ledger);
                let occupied = limit - target * management - 128 + 1;
                let mut cache = Some(context.cache_reservation(occupied as u128, limit).unwrap());
                let disk = Arc::new(TemporaryDiskBudget::new(4096));
                let mut owners = Vec::new();
                let mut calls = 0;
                for ordinal in 0..target {
                    let (mut file, mut guard) = create_budgeted_temporary_file_reclaiming(
                        &directory.0,
                        "reclaim",
                        "tmp",
                        disk.clone(),
                        &context,
                        phase,
                        || {
                            calls += 1;
                            // Failed name/path/node/state reservations cannot remain live.
                            assert_available(&sibling, management + 127);
                            drop(cache.take());
                            true
                        },
                    )
                    .unwrap();
                    assert_eq!(calls, usize::from(ordinal + 1 == target));
                    guard.reserve_disk(1).unwrap();
                    file.write_all(&[ordinal as u8]).unwrap();
                    owners.push((file, guard));
                }
                assert_eq!(calls, 1);
                assert!(cache.is_none());
                assert_eq!(disk.used(), target as u64);
                drop(context);
                assert_available(&sibling, limit - target * management);
                for (ordinal, (_, guard)) in owners.iter().enumerate() {
                    assert_eq!(fs::read(guard.path()).unwrap(), [ordinal as u8]);
                }
                drop(owners);
                assert_available(&sibling, limit);
                directory.assert_clean(&disk);
            }
        }
    }

    #[test]
    #[cfg(unix)]
    fn file_creation_declined_or_insufficient_reclaim_is_bounded_and_io_does_not_reclaim() {
        let directory = sort::tests::TestDirectory::new("creation-reclaim-failure");
        let management = test_file_management_bytes(&directory.0);
        let peak = management + 128;
        let limit = 2 * peak;
        for release in [0, 1] {
            let ledger = Arc::new(WorkLedger::with_limits(2 * limit, limit, limit));
            let context = JsonOperationContext::new(ledger);
            let occupied = limit - peak + 2;
            let mut cache = context.cache_reservation(occupied as u128, limit).unwrap();
            let disk = Arc::new(TemporaryDiskBudget::new(4096));
            let mut calls = 0;
            let error = create_budgeted_temporary_file_reclaiming(
                &directory.0,
                "reclaim",
                "tmp",
                disk.clone(),
                &context,
                "operation_scratch",
                || {
                    calls += 1;
                    assert_available(&context, peak - 2);
                    cache.retain(occupied - release);
                    release != 0
                },
            )
            .unwrap_err();
            assert_eq!(calls, 1);
            assert_eq!(error.code(), crate::ErrorCode::ResourceExhausted);
            assert!(error
                .detail()
                .contains(&("requested_bytes".into(), (limit + 2 - release).to_string())));
            assert_available(&context, peak - 2 + release);
            drop(cache);
            assert_available(&context, limit);
            directory.assert_clean(&disk);
        }
        let context = limited_context(limit);
        let cache = context.cache_reservation(1, limit).unwrap();
        let disk = Arc::new(TemporaryDiskBudget::new(4096));
        let error = create_budgeted_temporary_file_reclaiming(
            &directory.0.join("missing"),
            "reclaim",
            "tmp",
            disk.clone(),
            &context,
            "operation_scratch",
            || panic!("IO must not reclaim optional cache"),
        )
        .unwrap_err();
        assert_eq!(error.code(), crate::ErrorCode::IoError);
        assert_available(&context, limit - 1);
        drop(cache);
        assert_available(&context, limit);
        directory.assert_clean(&disk);
    }

    #[test]
    #[cfg(unix)]
    fn dynamic_name_capacity_and_open_failure_do_not_leak_or_remove_candidates() {
        let directory = sort::tests::TestDirectory::new("generic-open-failure");
        let context = limited_context(32 * 1024);
        let label = "a".repeat(200);
        let extension = "jsonl";
        let expected = format!(
            "cofferdb-{label}-{}-{}-{}.{extension}",
            u32::MAX,
            u128::MAX,
            u64::MAX
        );
        let name = budgeted_filename(
            &label,
            extension,
            (u32::MAX, u128::MAX, u64::MAX),
            &context,
            "operation_scratch",
        )
        .unwrap();
        assert_eq!(name.as_ref(), expected);
        assert_eq!(name_capacity(&label, extension).unwrap(), expected.len());
        drop(name);
        let disk = Arc::new(TemporaryDiskBudget::new(4096));
        let missing = directory.0.join("missing");
        let error = create_budgeted_temporary_file_in_context(
            &missing,
            "compact",
            "tmp",
            disk.clone(),
            &context,
            "operation_scratch",
        )
        .unwrap_err();
        assert_eq!(error.code(), crate::ErrorCode::IoError);
        assert!(!missing.exists());
        assert_available(&context, 32 * 1024);
        directory.assert_clean(&disk);
    }

    #[test]
    #[cfg(unix)]
    fn native_cleanup_mutex_allocation_fits_the_structure_allowance() {
        let lock = Mutex::new(());
        let (_, bytes) = crate::test_alloc::allocation_bytes(|| drop(lock.lock().unwrap()));
        assert!(
            bytes <= CLEANUP_LOCK_WORK_BYTES as u128,
            "native mutex allocated {bytes} bytes"
        );
        let (_, allocations) = crate::test_alloc::allocations(|| {
            drop(lock.lock().unwrap());
            drop(lock);
        });
        assert_eq!(allocations, 0);
    }

    #[test]
    #[cfg(unix)]
    fn untouched_sort_file_drops_without_cold_lock_allocations() {
        let directory = sort::tests::TestDirectory::new("sort-cold-drop");
        let context = JsonOperationContext::standalone();
        for file_first in [false, true] {
            let disk = Arc::new(TemporaryDiskBudget::new(4096));
            let (file, guard) =
                create_budgeted_sort_file(&directory.0, "import-sort", disk.clone(), &context)
                    .unwrap();
            let (_, allocations) = crate::test_alloc::allocations(|| {
                if file_first {
                    drop(file);
                    drop(guard);
                } else {
                    drop(guard);
                    drop(file);
                }
            });
            assert_eq!(allocations, 0);
            directory.assert_clean(&disk);
        }
    }

    #[test]
    #[cfg(unix)]
    fn sort_file_admits_all_management_before_creation_and_releases_both_drop_orders() {
        let directory = sort::tests::TestDirectory::new("sort-owner-admission");
        let (path_bytes, node_bytes, state_bytes) = sort_owner_sizes(&directory.0);
        let live = 2 * path_bytes + node_bytes + state_bytes;
        let peak = live + 128;
        for limit in [peak - 1, peak, peak + 1] {
            for file_first in [false, true] {
                let context = limited_context(limit);
                let disk = Arc::new(TemporaryDiskBudget::new(4096));
                let result =
                    create_budgeted_sort_file(&directory.0, "import-sort", disk.clone(), &context);
                if limit < peak {
                    let error = result.unwrap_err();
                    assert_eq!(error.code(), crate::ErrorCode::ResourceExhausted);
                    assert!(error
                        .detail()
                        .iter()
                        .any(|item| item.0 == "requested_bytes" && item.1 == peak.to_string()));
                } else {
                    let (mut file, mut guard) = result.unwrap();
                    let path = guard.path().to_path_buf();
                    guard.reserve_disk(13).unwrap();
                    file.write_all(b"owned payload").unwrap();
                    assert_available(&context, limit - live);
                    if file_first {
                        drop(file);
                        assert_available(&context, limit - live);
                        drop(guard);
                    } else {
                        drop(guard);
                        assert_available(&context, limit - live + path_bytes);
                        drop(file);
                    }
                    assert!(!path.exists());
                }
                assert_available(&context, limit);
                directory.assert_clean(&disk);
            }
        }
    }

    #[test]
    #[cfg(unix)]
    fn sort_cleanup_failure_keeps_original_path_and_node_credit_without_allocation() {
        let directory = sort::tests::TestDirectory::new("sort-owner-retry");
        let (path_bytes, node_bytes, state_bytes) = sort_owner_sizes(&directory.0);
        let limit = 128 + 2 * path_bytes + node_bytes + state_bytes;
        for seed in 0..100u64 {
            let context = limited_context(limit);
            let disk = Arc::new(TemporaryDiskBudget::new(4096));
            let (mut file, mut guard) =
                create_budgeted_sort_file(&directory.0, "import-sort", disk.clone(), &context)
                    .unwrap();
            let bytes = seed + 1;
            guard.reserve_disk(bytes).unwrap();
            file.write_all(&vec![17; bytes as usize]).unwrap();
            let path = guard.path().to_path_buf();
            drop(file);
            let (_, allocations) = crate::test_alloc::allocations(|| {
                guard
                    .cleanup
                    .lock()
                    .unwrap()
                    .cleanup_with(|_| Err(std::io::Error::from_raw_os_error(13)));
                drop(guard);
            });
            assert_eq!(allocations, 0);
            assert_available(&context, limit - path_bytes - node_bytes);
            assert_eq!(disk.used(), bytes);
            assert_eq!(Arc::strong_count(&disk), 1);
            assert!(path.exists());
            disk.retry_cleanup().unwrap();
            assert_available(&context, limit);
            directory.assert_clean(&disk);
        }
    }

    #[test]
    #[cfg(unix)]
    fn concurrent_sort_owner_drops_return_state_credit_once_even_when_poisoned() {
        let directory = sort::tests::TestDirectory::new("sort-owner-concurrent");
        let (path_bytes, node_bytes, state_bytes) = sort_owner_sizes(&directory.0);
        let limit = 128 + 2 * path_bytes + node_bytes + state_bytes;
        for poison in [false, true] {
            for _ in 0..100 {
                let context = limited_context(limit);
                let disk = Arc::new(TemporaryDiskBudget::new(4096));
                let (file, mut guard) =
                    create_budgeted_sort_file(&directory.0, "import-sort", disk.clone(), &context)
                        .unwrap();
                guard.reserve_disk(1).unwrap();
                if poison {
                    let state = &guard.cleanup;
                    assert!(std::thread::scope(|scope| scope
                        .spawn(|| {
                            let _lock = state.lock().unwrap();
                            panic!("injected cleanup lock poison");
                        })
                        .join())
                    .is_err());
                }
                let barrier = std::sync::Barrier::new(2);
                std::thread::scope(|scope| {
                    let barrier = &barrier;
                    scope.spawn(move || {
                        barrier.wait();
                        drop(file);
                    });
                    scope.spawn(move || {
                        barrier.wait();
                        drop(guard);
                    });
                });
                assert_available(&context, limit);
                directory.assert_clean(&disk);
            }
        }
    }

    #[test]
    #[cfg(unix)]
    fn sort_name_collision_preserves_existing_file_and_retry_credit() {
        let directory = sort::tests::TestDirectory::new("sort-collision");
        let (path_bytes, node_bytes, state_bytes) = sort_owner_sizes(&directory.0);
        let limit = 128 + 2 * path_bytes + node_bytes + state_bytes;
        let context = limited_context(limit);
        let disk = Arc::new(TemporaryDiskBudget::new(4096));
        let existing =
            budgeted_filename("import-sort", "tmp", (0, 0, 0), &context, "catalog_build").unwrap();
        let existing_path = directory.0.join(existing.as_ref());
        drop(existing);
        fs::write(&existing_path, b"untouched").unwrap();
        let mut id = 0;
        let (file, guard) = create_file_with_names(
            &directory.0,
            disk.clone(),
            &context,
            "catalog_build",
            128,
            || {
                let result =
                    budgeted_filename("import-sort", "tmp", (0, 0, id), &context, "catalog_build");
                id += 1;
                result
            },
        )
        .unwrap();
        assert_eq!(id, 2);
        assert_eq!(fs::read(&existing_path).unwrap(), b"untouched");
        drop((file, guard));
        assert_available(&context, limit);
        assert_eq!(fs::read(&existing_path).unwrap(), b"untouched");
        fs::remove_file(existing_path).unwrap();
        directory.assert_clean(&disk);
        for label in ["import-sort", "import-lookup"] {
            let name = budgeted_filename(
                label,
                "tmp",
                (u32::MAX, u128::MAX, u64::MAX),
                &context,
                "catalog_build",
            )
            .unwrap();
            assert!(name.as_ref().len() <= 128);
            assert_eq!(
                name.as_ref(),
                format!(
                    "cofferdb-{label}-{}-{}-{}.tmp",
                    u32::MAX,
                    u128::MAX,
                    u64::MAX
                )
            );
        }
        assert_available(&context, limit);
    }

    #[test]
    fn cleanup_failure_transfers_existing_owners_without_new_allocations() {
        for seed in 0..100u64 {
            let bytes = seed + 1;
            let budget = Arc::new(TemporaryDiskBudget::new(4096));
            let (mut file, mut guard) = create_budgeted_temporary_file_in(
                &std::env::temp_dir(),
                "cleanup-no-allocation",
                "tmp",
                budget.clone(),
            )
            .unwrap();
            guard.reserve_disk(bytes).unwrap();
            file.write_all(&vec![31; bytes as usize]).unwrap();
            let path = guard.path().to_path_buf();
            drop(file);
            let (_, allocations) = crate::test_alloc::allocations(|| {
                guard
                    .cleanup
                    .lock()
                    .unwrap()
                    .cleanup_with(|_| Err(std::io::Error::from_raw_os_error(13)));
                drop(guard);
            });
            assert_eq!(allocations, 0, "seed {seed}");
            assert_eq!(budget.used(), bytes);
            assert_eq!(Arc::strong_count(&budget), 1);
            assert!(path.exists());
            budget.retry_cleanup().unwrap();
            assert_eq!(budget.used(), 0);
            assert!(!path.exists());
        }
    }

    #[test]
    #[cfg(unix)]
    fn mixed_cleanup_retry_preserves_only_failed_files_and_replacement_names() {
        for seed in 0..10usize {
            let budget = Arc::new(TemporaryDiskBudget::new(4096));
            let mut files = Vec::new();
            for ordinal in 0..8u64 {
                let (mut file, mut guard) = create_budgeted_temporary_file_in(
                    &std::env::temp_dir(),
                    "mixed-cleanup",
                    "tmp",
                    budget.clone(),
                )
                .unwrap();
                let bytes = ordinal + 1;
                guard.reserve_disk(bytes).unwrap();
                file.write_all(&vec![17; bytes as usize]).unwrap();
                files.push((file, guard, bytes));
            }
            let mut retained = Vec::new();
            let mut healthy = Vec::new();
            let mut total = 0;
            for (ordinal, (file, guard, bytes)) in files.into_iter().enumerate() {
                let path = guard.path().to_path_buf();
                drop(file);
                guard
                    .cleanup
                    .lock()
                    .unwrap()
                    .cleanup_with(|_| Err(std::io::Error::from_raw_os_error(13)));
                drop(guard);
                total += bytes;
                if ordinal % 3 == seed % 3 {
                    let original = path.with_extension("owned");
                    fs::rename(&path, &original).unwrap();
                    fs::write(&path, b"replacement").unwrap();
                    retained.push((path, original, bytes));
                } else {
                    healthy.push(path);
                }
            }
            assert_eq!(budget.used(), total);
            assert_eq!(Arc::strong_count(&budget), 1);
            assert_eq!(
                budget.retry_cleanup().unwrap_err().code(),
                crate::ErrorCode::IoError
            );
            assert_eq!(budget.used(), retained.iter().map(|entry| entry.2).sum());
            assert!(healthy.iter().all(|path| !path.exists()));
            let mut restored = Vec::new();
            for (path, original, bytes) in retained {
                assert_eq!(fs::read(&path).unwrap(), b"replacement");
                assert_eq!(fs::read(&original).unwrap(), vec![17; bytes as usize]);
                fs::remove_file(&path).unwrap();
                fs::rename(original, &path).unwrap();
                restored.push(path);
            }
            budget.retry_cleanup().unwrap();
            assert_eq!(budget.used(), 0);
            assert!(restored.iter().all(|path| !path.exists()));
        }
    }

    #[test]
    fn cleanup_failures_keep_ownership_until_retry_or_runtime_shutdown_across_100_seeds() {
        for seed in 0..100_u64 {
            for kind in [
                None,
                Some(std::io::ErrorKind::NotFound),
                Some(std::io::ErrorKind::PermissionDenied),
                Some(std::io::ErrorKind::Interrupted),
                Some(std::io::ErrorKind::Other),
            ] {
                let limit = 4096;
                let bytes = (seed * 127) % (limit + 1);
                let budget = Arc::new(TemporaryDiskBudget::new(limit));
                let weak = Arc::downgrade(&budget);
                let (mut file, mut guard) = create_budgeted_temporary_file_in(
                    &std::env::temp_dir(),
                    "cleanup-retry",
                    "tmp",
                    budget.clone(),
                )
                .unwrap();
                guard.reserve_disk(bytes).unwrap();
                file.write_all(&vec![b'x'; bytes as usize]).unwrap();
                let path = guard.path().to_path_buf();
                drop(file);
                guard.cleanup.lock().unwrap().cleanup_with(|actual| {
                    assert_eq!(actual, path);
                    if matches!(kind, None | Some(std::io::ErrorKind::NotFound)) {
                        fs::remove_file(actual).unwrap();
                    }
                    match kind {
                        None => Ok(()),
                        Some(kind) => Err(std::io::Error::new(kind, "injected cleanup failure")),
                    }
                });
                drop(guard);
                let orphaned = match kind {
                    None | Some(std::io::ErrorKind::NotFound) => 0,
                    _ => bytes,
                };
                assert_eq!(budget.used(), orphaned, "seed {seed}, {kind:?}");
                assert_eq!(Arc::strong_count(&budget), 1);
                let mut next = budget.reservation();
                next.grow(limit - orphaned).unwrap();
                assert!(budget.check_additional(1).is_err());
                drop(next);
                assert_eq!(budget.used(), orphaned);
                if seed % 2 == 0 {
                    let (file, guard) = create_budgeted_temporary_file_in(
                        &std::env::temp_dir(),
                        "after-cleanup",
                        "tmp",
                        budget.clone(),
                    )
                    .unwrap();
                    assert_eq!(budget.used(), 0);
                    assert!(!path.exists());
                    drop((file, guard));
                }
                drop(budget);
                assert!(weak.upgrade().is_none());
                assert!(!path.exists());
            }
        }
    }

    #[test]
    fn disk_and_cleanup_ownership_outlive_either_drop_order() {
        for seed in 0..100 {
            for file_first in [false, true] {
                let bytes = seed + 1;
                let budget = Arc::new(TemporaryDiskBudget::new(bytes));
                let (mut file, mut guard) = create_budgeted_temporary_file_in(
                    &std::env::temp_dir(),
                    "drop-order",
                    "tmp",
                    budget.clone(),
                )
                .unwrap();
                let path = guard.path().to_path_buf();
                guard.reserve_disk(bytes).unwrap();
                file.write_all(&vec![b'x'; bytes as usize]).unwrap();
                if file_first {
                    drop(file);
                    assert!(path.exists());
                    assert_eq!(budget.used(), bytes);
                    drop(guard);
                } else {
                    drop(guard);
                    assert!(path.exists());
                    assert_eq!(budget.used(), bytes);
                    file.write_all(b"").unwrap();
                    drop(file);
                }
                assert!(!path.exists());
                assert_eq!(budget.used(), 0);
            }
        }
    }

    #[test]
    fn live_reservations_retry_cleanup_under_pressure_without_double_release() {
        let budget = Arc::new(TemporaryDiskBudget::new(4096));
        let barrier = std::sync::Barrier::new(8);
        std::thread::scope(|scope| {
            for _ in 0..8 {
                let budget = &budget;
                let barrier = &barrier;
                scope.spawn(move || {
                    let (mut file, mut guard) = create_budgeted_temporary_file_in(
                        &std::env::temp_dir(),
                        "concurrent-cleanup",
                        "tmp",
                        budget.clone(),
                    )
                    .unwrap();
                    guard.reserve_disk(512).unwrap();
                    file.write_all(&[b'x'; 512]).unwrap();
                    let path = guard.path().to_path_buf();
                    drop(file);
                    barrier.wait();
                    guard.cleanup.lock().unwrap().cleanup_with(|_| {
                        Err(std::io::Error::from(std::io::ErrorKind::PermissionDenied))
                    });
                    drop(guard);
                    barrier.wait();
                    // All eight failed files retain their original charge.
                    assert_eq!(budget.used(), 4096);
                    barrier.wait();
                    let mut next = budget.reservation();
                    next.grow(512).unwrap();
                    barrier.wait();
                    assert!(!path.exists());
                    assert_eq!(budget.used(), 4096);
                    barrier.wait();
                    drop(next);
                });
            }
        });
        assert_eq!(budget.used(), 0);
        assert_eq!(Arc::strong_count(&budget), 1);
    }

    #[cfg(unix)]
    #[test]
    fn retry_never_removes_replacement_names_or_releases_still_linked_originals() {
        for seed in 0..100 {
            for hard_link in [false, true] {
                let bytes = seed + 1;
                let budget = Arc::new(TemporaryDiskBudget::new(bytes));
                let (mut file, mut guard) = create_budgeted_temporary_file_in(
                    &std::env::temp_dir(),
                    "cleanup-owner",
                    "tmp",
                    budget.clone(),
                )
                .unwrap();
                let path = guard.path().to_path_buf();
                guard.reserve_disk(bytes).unwrap();
                file.write_all(&vec![b'x'; bytes as usize]).unwrap();
                let (other, moved) =
                    create_temporary_file_in(&std::env::temp_dir(), "moved-owner", "tmp").unwrap();
                drop(other);
                fs::remove_file(moved.path()).unwrap();
                if hard_link {
                    fs::hard_link(&path, moved.path()).unwrap();
                } else {
                    fs::rename(&path, moved.path()).unwrap();
                    fs::write(&path, b"replacement").unwrap();
                }
                drop((file, guard));
                assert_eq!(budget.used(), bytes);
                assert!(budget.retry_cleanup().is_err());
                let error = create_budgeted_temporary_file_in(
                    &std::env::temp_dir(),
                    "refused-after-failure",
                    "tmp",
                    budget.clone(),
                )
                .unwrap_err();
                assert_eq!(error.code(), crate::ErrorCode::IoError);
                assert_eq!(budget.used(), bytes);
                assert_eq!(fs::read(moved.path()).unwrap(), vec![b'x'; bytes as usize]);
                if !hard_link {
                    assert_eq!(fs::read(&path).unwrap(), b"replacement");
                }
                drop(moved);
                budget.retry_cleanup().unwrap();
                assert_eq!(budget.used(), 0);
                if !hard_link {
                    assert_eq!(fs::read(&path).unwrap(), b"replacement");
                    fs::remove_file(&path).unwrap();
                }
            }
        }
    }

    #[cfg(unix)]
    #[test]
    fn already_unlinked_original_does_not_remove_a_replacement_symlink() {
        let budget = Arc::new(TemporaryDiskBudget::new(1));
        let (file, mut guard) = create_budgeted_temporary_file_in(
            &std::env::temp_dir(),
            "replaced-unlinked",
            "tmp",
            budget.clone(),
        )
        .unwrap();
        guard.reserve_disk(1).unwrap();
        let path = guard.path().to_path_buf();
        fs::remove_file(&path).unwrap();
        let (mut target_file, target) =
            create_temporary_file_in(&std::env::temp_dir(), "untouched-target", "tmp").unwrap();
        target_file.write_all(b"untouched").unwrap();
        drop(target_file);
        std::os::unix::fs::symlink(target.path(), &path).unwrap();
        drop((file, guard));
        assert_eq!(budget.used(), 0);
        assert!(fs::symlink_metadata(&path)
            .unwrap()
            .file_type()
            .is_symlink());
        assert_eq!(fs::read(target.path()).unwrap(), b"untouched");
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn disk_charge_is_released_after_actual_cleanup_absence_or_publication() {
        let directory = std::env::temp_dir();
        let budget = Arc::new(TemporaryDiskBudget::new(1024));
        for already_removed in [false, true] {
            let (mut file, mut guard) = create_budgeted_temporary_file_in(
                &directory,
                "cleanup-budget",
                "tmp",
                budget.clone(),
            )
            .unwrap();
            guard.reserve_disk(3).unwrap();
            file.write_all(b"abc").unwrap();
            drop(file);
            let path = guard.path().to_path_buf();
            if already_removed {
                fs::remove_file(&path).unwrap();
            }
            drop(guard);
            assert!(!path.exists());
            assert_eq!(budget.used(), 0);
        }

        let (destination_file, destination) =
            create_temporary_file_in(&directory, "budget-published", "tmp").unwrap();
        drop(destination_file);
        let (mut file, mut guard) =
            create_budgeted_temporary_file_in(&directory, "budget-source", "tmp", budget.clone())
                .unwrap();
        guard.reserve_disk(3).unwrap();
        file.write_all(b"abc").unwrap();
        drop(file);
        guard.persist(destination.path()).unwrap();
        assert_eq!(fs::read(destination.path()).unwrap(), b"abc");
        assert_eq!(budget.used(), 0);
    }

    #[test]
    fn publish_replaces_only_after_complete_writes_and_cleans_up_failed_renames() {
        let directory = std::env::temp_dir();
        let (mut destination_file, destination) =
            create_temporary_file_in(&directory, "publish-destination", "tmp").unwrap();
        destination_file.write_all(b"old").unwrap();
        drop(destination_file);
        let (mut file, staged) =
            create_temporary_file_in(&directory, "publish-source", "tmp").unwrap();
        let staged_path = staged.path().to_path_buf();
        file.write_all(b"new complete output").unwrap();
        file.flush().unwrap();
        drop(file);
        assert_eq!(fs::read(destination.path()).unwrap(), b"old");
        staged.persist(destination.path()).unwrap();
        assert!(!staged_path.exists());
        assert_eq!(
            fs::read(destination.path()).unwrap(),
            b"new complete output"
        );

        let (file, failed) =
            create_temporary_file_in(&directory, "publish-failure", "tmp").unwrap();
        let failed_path = failed.path().to_path_buf();
        drop(file);
        assert!(failed.persist(&directory).is_err());
        assert!(!failed_path.exists());
        assert!(directory.is_dir());
    }
}
