use std::collections::BTreeSet;
use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use crate::error::{Error, ErrorCode, Result};

pub(crate) struct FileLock {
    path: PathBuf,
    file: File,
    data: Option<DataFileLock>,
}

impl FileLock {
    pub(crate) fn acquire_exclusive(data_path: &Path) -> Result<Self> {
        let display_path = lock_path(data_path);
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&display_path)?;
        let registry_path = display_path
            .canonicalize()
            .unwrap_or_else(|_| display_path.clone());
        reserve_process_lock(&registry_path, &display_path)?;
        if let Err(err) = lock_file_exclusive_nonblocking(&file, &display_path) {
            release_process_lock(&registry_path);
            return Err(err);
        }
        Ok(Self {
            path: registry_path,
            file,
            data: None,
        })
    }

    pub(crate) fn replace_data_lock(&mut self, data: DataFileLock) {
        self.data = Some(data);
    }
}

impl Drop for FileLock {
    fn drop(&mut self) {
        self.data.take();
        unlock_file(&self.file);
        release_process_lock(&self.path);
    }
}

pub(crate) struct DataFileLock(File);

impl DataFileLock {
    pub(crate) fn acquire(file: &File, data_path: &Path) -> Result<Self> {
        let file = file.try_clone()?;
        lock_file_exclusive_nonblocking(&file, &lock_path(data_path))?;
        Ok(Self(file))
    }
}

impl Drop for DataFileLock {
    fn drop(&mut self) {
        unlock_file(&self.0);
    }
}

pub(crate) fn lock_path(data_path: &Path) -> PathBuf {
    let mut name = data_path
        .file_name()
        .map(|value| value.to_os_string())
        .unwrap_or_else(|| "cofferdb".into());
    name.push(".lock");
    data_path.with_file_name(name)
}

fn reserve_process_lock(path: &Path, display_path: &Path) -> Result<()> {
    let mut locks = held_locks().lock().map_err(|_| {
        Error::new(
            ErrorCode::LockContention,
            "process lock registry is poisoned",
        )
    })?;
    if !locks.insert(path.to_path_buf()) {
        return Err(lock_contention(display_path));
    }
    Ok(())
}

fn release_process_lock(path: &Path) {
    if let Ok(mut locks) = held_locks().lock() {
        locks.remove(path);
    }
}

fn held_locks() -> &'static Mutex<BTreeSet<PathBuf>> {
    static LOCKS: OnceLock<Mutex<BTreeSet<PathBuf>>> = OnceLock::new();
    LOCKS.get_or_init(|| Mutex::new(BTreeSet::new()))
}

fn lock_contention(path: &Path) -> Error {
    Error::new(
        ErrorCode::LockContention,
        "database writer lock is already held",
    )
    .with_detail("lock_path", path.display())
}

#[cfg(unix)]
fn lock_file_exclusive_nonblocking(file: &File, path: &Path) -> Result<()> {
    use std::os::fd::AsRawFd;
    use std::os::raw::c_int;

    const LOCK_EX: c_int = 2;
    const LOCK_NB: c_int = 4;

    extern "C" {
        fn flock(fd: c_int, operation: c_int) -> c_int;
    }

    if unsafe { flock(file.as_raw_fd(), LOCK_EX | LOCK_NB) } == 0 {
        return Ok(());
    }
    let err = std::io::Error::last_os_error();
    if err.kind() == std::io::ErrorKind::WouldBlock {
        Err(lock_contention(path))
    } else {
        Err(Error::from(err))
    }
}

#[cfg(unix)]
fn unlock_file(file: &File) {
    use std::os::fd::AsRawFd;
    use std::os::raw::c_int;

    const LOCK_UN: c_int = 8;

    extern "C" {
        fn flock(fd: c_int, operation: c_int) -> c_int;
    }

    let _ = unsafe { flock(file.as_raw_fd(), LOCK_UN) };
}

#[cfg(not(unix))]
fn lock_file_exclusive_nonblocking(_file: &File, _path: &Path) -> Result<()> {
    Ok(())
}

#[cfg(not(unix))]
fn unlock_file(_file: &File) {}
