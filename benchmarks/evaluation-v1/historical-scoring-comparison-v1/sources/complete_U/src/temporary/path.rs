use std::ffi::OsString;
use std::path::{Path, PathBuf};

use crate::json::{JsonOperationContext, WorkspaceReservation};
use crate::Result;

#[derive(Debug, Default)]
pub(crate) struct TemporaryPath {
    value: PathBuf,
    _work: Option<WorkspaceReservation>,
}

impl TemporaryPath {
    pub(crate) fn system_temp_in(context: &JsonOperationContext) -> Result<Self> {
        let mut work = context.workspace_reservation();
        // The OS/standard-library discovery allocation precedes adoption; its
        // resident capacity, not just its path length, remains charged here.
        let value = std::env::temp_dir();
        work.grow_to(value.capacity(), value.capacity(), "operation_scratch")?;
        Ok(Self {
            value,
            _work: Some(work),
        })
    }

    pub(super) fn unreserved(value: PathBuf) -> Self {
        Self { value, _work: None }
    }

    pub(super) fn copy_in(value: &Path, context: &JsonOperationContext) -> Result<Self> {
        Self::copy_with_capacity(
            value,
            value.as_os_str().as_encoded_bytes().len(),
            context,
            "catalog_build",
        )
    }

    #[cfg(unix)]
    pub(super) fn copy_reserved(
        &self,
        context: &JsonOperationContext,
        phase: &'static str,
    ) -> Result<Self> {
        Self::copy_with_capacity(&self.value, self.value.capacity(), context, phase)
    }

    fn copy_with_capacity(
        value: &Path,
        capacity: usize,
        context: &JsonOperationContext,
        phase: &'static str,
    ) -> Result<Self> {
        if value.as_os_str().as_encoded_bytes().len() > capacity {
            return Err(std::io::Error::other("path copy exceeds admitted capacity").into());
        }
        let (mut bytes, work) = Self::allocate(capacity, context, phase)?;
        bytes.push(value.as_os_str());
        Self::from_parts(bytes, capacity, work)
    }

    #[cfg(unix)]
    pub(super) fn join_name(
        directory: &Path,
        name: &str,
        capacity: usize,
        context: &JsonOperationContext,
        phase: &'static str,
    ) -> Result<Self> {
        let directory_bytes = directory.as_os_str().as_encoded_bytes();
        let separator = usize::from(directory_bytes.last().is_some_and(|&byte| byte != b'/'));
        let required = directory_bytes
            .len()
            .checked_add(separator)
            .and_then(|bytes| bytes.checked_add(name.len()));
        if required.map_or(true, |bytes| bytes > capacity) {
            return Err(std::io::Error::other("path join exceeds admitted capacity").into());
        }
        let (mut bytes, work) = Self::allocate(capacity, context, phase)?;
        bytes.push(directory.as_os_str());
        if directory
            .as_os_str()
            .as_encoded_bytes()
            .last()
            .is_some_and(|&byte| byte != b'/')
        {
            bytes.push("/");
        }
        bytes.push(name);
        Self::from_parts(bytes, capacity, work)
    }

    fn allocate(
        capacity: usize,
        context: &JsonOperationContext,
        phase: &'static str,
    ) -> Result<(OsString, WorkspaceReservation)> {
        let mut work = context.workspace_reservation();
        work.grow_to(capacity, capacity, phase)?;
        let mut bytes = OsString::new();
        bytes
            .try_reserve_exact(capacity)
            .map_err(|error| std::io::Error::other(error.to_string()))?;
        if bytes.capacity() != capacity {
            return Err(std::io::Error::other("path allocation exceeds admitted capacity").into());
        }
        Ok((bytes, work))
    }

    fn from_parts(bytes: OsString, capacity: usize, work: WorkspaceReservation) -> Result<Self> {
        if bytes.capacity() != capacity {
            drop(bytes);
            drop(work);
            return Err(std::io::Error::other("path growth exceeds admitted capacity").into());
        }
        Ok(Self {
            value: PathBuf::from(bytes),
            _work: Some(work),
        })
    }

    pub(super) fn clear(&mut self) {
        self.value.clear();
    }
}

impl std::ops::Deref for TemporaryPath {
    type Target = Path;

    fn deref(&self) -> &Path {
        &self.value
    }
}

impl AsRef<Path> for TemporaryPath {
    fn as_ref(&self) -> &Path {
        &self.value
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json::WorkLedger;
    use std::sync::Arc;

    fn context(limit: usize) -> JsonOperationContext {
        JsonOperationContext::new(Arc::new(WorkLedger::with_limits(limit, limit, limit)))
    }

    #[test]
    fn system_temp_adoption_retains_actual_capacity_and_returns_it_on_rejection_or_drop() {
        let expected = std::env::temp_dir();
        let needed = expected.capacity();
        for parent in [false, true] {
            for delta in [-1isize, 0, 1] {
                let limit = (needed as isize + delta) as usize;
                let context = JsonOperationContext::new(Arc::new(WorkLedger::with_limits(
                    if parent { 2 * limit } else { limit },
                    0,
                    if parent { limit } else { 2 * limit },
                )));
                let result = TemporaryPath::system_temp_in(&context);
                if delta < 0 {
                    let error = result.unwrap_err();
                    assert_eq!(error.code(), crate::ErrorCode::ResourceExhausted);
                    assert!(error
                        .detail()
                        .contains(&("requested_bytes".into(), needed.to_string())));
                } else {
                    let path = result.unwrap();
                    assert_eq!(path.as_ref(), expected.as_path());
                    assert_eq!(path._work.as_ref().unwrap().reserved_bytes(), needed);
                    assert_eq!(context.available_workspace_bytes(), limit - needed);
                }
                assert_eq!(context.available_workspace_bytes(), limit);
            }
        }
    }

    #[test]
    fn path_copy_admits_exact_capacity_and_clear_keeps_ownership() {
        let input = Path::new("cofferdb-sort-directory");
        let bytes = input.as_os_str().as_encoded_bytes().len();
        for available in [bytes - 1, bytes, bytes + 1] {
            let context = context(available);
            let result = TemporaryPath::copy_in(input, &context);
            if available < bytes {
                assert_eq!(
                    result.unwrap_err().code(),
                    crate::ErrorCode::ResourceExhausted
                );
            } else {
                let mut path = result.unwrap();
                assert_eq!(path.as_ref(), input);
                assert_eq!(path.value.capacity(), bytes);
                path.clear();
                assert!(path.as_os_str().is_empty());
                assert_eq!(path.value.capacity(), bytes);
                let mut spare = context.workspace_reservation();
                spare
                    .grow_to(available - bytes, available - bytes, "catalog_build")
                    .unwrap();
                assert!(spare
                    .grow_to(
                        available - bytes + 1,
                        available - bytes + 1,
                        "catalog_build"
                    )
                    .is_err());
                drop(spare);
                drop(path);
            }
            context
                .workspace_reservation()
                .grow_to(available, available, "catalog_build")
                .unwrap();
        }
    }

    #[test]
    #[cfg(unix)]
    fn native_join_preserves_bytes_and_rejects_undersized_capacity_before_growth() {
        use std::os::unix::ffi::OsStringExt;
        for bytes in [
            b"".as_slice(),
            b"/",
            b"//",
            b"//tmp/",
            b"relative",
            b"/tmp/\xff",
        ] {
            let directory = PathBuf::from(OsString::from_vec(bytes.to_vec()));
            let expected = directory.join("run.tmp");
            let capacity = expected.as_os_str().as_encoded_bytes().len() + 19;
            let context = context(2 * capacity);
            let path = TemporaryPath::join_name(
                &directory,
                "run.tmp",
                capacity,
                &context,
                "catalog_build",
            )
            .unwrap();
            assert_eq!(path.as_ref(), expected);
            assert_eq!(
                path.as_os_str().as_encoded_bytes(),
                expected.as_os_str().as_encoded_bytes()
            );
            let copy = path.copy_reserved(&context, "catalog_build").unwrap();
            assert_eq!(copy.value.capacity(), capacity);
            assert_eq!(copy.as_ref(), path.as_ref());
            assert_eq!(
                copy.as_os_str().as_encoded_bytes(),
                expected.as_os_str().as_encoded_bytes()
            );
            assert!(context
                .workspace_reservation()
                .grow_to(1, 1, "catalog_build")
                .is_err());
            drop((copy, path));
            assert_eq!(
                TemporaryPath::join_name(
                    &directory,
                    "run.tmp",
                    expected.as_os_str().as_encoded_bytes().len() - 1,
                    &context,
                    "catalog_build"
                )
                .unwrap_err()
                .code(),
                crate::ErrorCode::IoError
            );
            context
                .workspace_reservation()
                .grow_to(2 * capacity, 2 * capacity, "catalog_build")
                .unwrap();
        }
    }
}
