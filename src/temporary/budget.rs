use super::TemporaryPath;
use std::fs::File;
use std::sync::{Arc, Mutex};

use crate::{Error, ErrorCode, Result};

#[derive(Debug)]
pub(crate) struct TemporaryDiskBudget {
    limit: u64,
    state: Mutex<DiskState>,
}

#[derive(Debug, Default)]
struct DiskState {
    used: u64,
    #[cfg(test)]
    peak: u64,
    pending: PendingCleanups,
}

#[derive(Debug)]
struct PendingCleanup {
    path: TemporaryPath,
    file: File,
    bytes: u64,
}

#[derive(Default)]
pub(super) struct PendingCleanupNode {
    cleanup: Option<PendingCleanup>,
    next: Option<PendingCleanupOwner>,
}

#[derive(Debug, Default)]
pub(super) struct PendingCleanupOwner {
    node: Box<PendingCleanupNode>,
    _work: Option<crate::json::WorkspaceReservation>,
}

impl PendingCleanupOwner {
    #[cfg(unix)]
    pub(super) fn new(
        context: &crate::json::JsonOperationContext,
        phase: &'static str,
    ) -> Result<Self> {
        let bytes = std::mem::size_of::<PendingCleanupNode>();
        let mut work = context.workspace_reservation();
        work.grow_to(bytes, bytes, phase)?;
        Ok(Self {
            node: Box::default(),
            _work: Some(work),
        })
    }
}

#[derive(Default)]
struct PendingCleanups {
    head: Option<PendingCleanupOwner>,
    len: usize,
}

impl std::fmt::Debug for PendingCleanupNode {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        out.debug_struct("PendingCleanupNode")
            .field("cleanup", &self.cleanup)
            .finish_non_exhaustive()
    }
}

impl std::fmt::Debug for PendingCleanups {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        out.debug_struct("PendingCleanups")
            .field("len", &self.len)
            .finish()
    }
}

impl PendingCleanups {
    fn push(&mut self, mut owner: PendingCleanupOwner) {
        assert!(owner.node.next.is_none());
        owner.node.next = self.head.take();
        self.head = Some(owner);
        self.len += 1;
    }

    fn pop(&mut self) -> Option<PendingCleanupOwner> {
        let mut owner = self.head.take()?;
        self.head = owner.node.next.take();
        self.len -= 1;
        Some(owner)
    }
}

impl Drop for PendingCleanups {
    fn drop(&mut self) {
        while self.pop().is_some() {}
    }
}

impl DiskState {
    fn retry_cleanup(&mut self) -> std::io::Result<()> {
        let mut first_error = None;
        let mut pending = std::mem::take(&mut self.pending);
        let mut failed = PendingCleanups::default();
        while let Some(node) = pending.pop() {
            let owner = node.node.cleanup.as_ref().expect("active cleanup node");
            match super::remove_owned_file(&owner.file, &owner.path) {
                Ok(()) => {
                    let bytes = owner.bytes;
                    drop(node); // Close the final descriptor before returning its charge.
                    self.used -= bytes;
                }
                Err(error) => {
                    first_error.get_or_insert(error);
                    failed.push(node);
                }
            }
        }
        self.pending = failed;
        first_error.map_or(Ok(()), Err)
    }
}

impl Default for TemporaryDiskBudget {
    fn default() -> Self {
        // Disk staging is independent of resident cache; allow input and encoded copies
        // for the v0.1 2-GiB calibration while retaining one finite runtime-wide ceiling.
        Self::new(4 * 1024 * 1024 * 1024)
    }
}

impl TemporaryDiskBudget {
    pub(crate) fn limit_bytes(&self) -> u64 {
        self.limit
    }

    pub(crate) fn new(limit: u64) -> Self {
        Self {
            limit,
            state: Mutex::new(DiskState::default()),
        }
    }

    pub(crate) fn check_additional(&self, bytes: u64) -> Result<()> {
        let state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        self.available_after_check(state.used, bytes).map(|_| ())
    }

    pub(crate) fn retry_cleanup(&self) -> Result<()> {
        self.state
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .retry_cleanup()?;
        Ok(())
    }

    fn available_after_check(&self, used: u64, minimum: u64) -> Result<u64> {
        let available = self.limit - used;
        if minimum > available {
            return Err(Error::resource_exhausted(
                "temporary_disk",
                "temporary_staging",
                self.limit,
                u128::from(used) + u128::from(minimum),
            ));
        }
        Ok(available)
    }

    pub(super) fn reservation(self: &Arc<Self>) -> DiskReservation {
        DiskReservation {
            budget: Arc::clone(self),
            bytes: 0,
        }
    }

    #[cfg(test)]
    pub(crate) fn used(&self) -> u64 {
        self.state
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .used
    }

    #[cfg(test)]
    pub(crate) fn peak_used(&self) -> u64 {
        self.state
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .peak
    }
}

impl Drop for TemporaryDiskBudget {
    fn drop(&mut self) {
        // Final best effort at runtime shutdown; pending entries contain no Arc
        // back to the ledger, so failures cannot keep a runtime alive forever.
        let _ = self
            .state
            .get_mut()
            .unwrap_or_else(|error| error.into_inner())
            .retry_cleanup();
    }
}

#[derive(Debug)]
pub(crate) struct DiskReservation {
    budget: Arc<TemporaryDiskBudget>,
    bytes: u64,
}

impl DiskReservation {
    pub(crate) fn reserved_bytes(&self) -> u64 {
        self.bytes
    }

    pub(crate) fn retain(&mut self, bytes: u64) {
        assert!(bytes <= self.bytes);
        self.release(self.bytes - bytes);
    }

    pub(super) fn grow(&mut self, bytes: u64) -> Result<()> {
        self.grow_at_least(bytes, bytes).map(|_| ())
    }

    pub(super) fn grow_at_least(&mut self, minimum: u64, preferred: u64) -> Result<u64> {
        let mut state = self
            .budget
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if minimum > self.budget.limit - state.used {
            state.retry_cleanup()?;
        }
        let available = self.budget.available_after_check(state.used, minimum)?;
        let granted = preferred.max(minimum).min(available);
        state.used += granted;
        #[cfg(test)]
        {
            state.peak = state.peak.max(state.used);
        }
        self.bytes += granted;
        Ok(granted)
    }

    pub(super) fn release(&mut self, bytes: u64) {
        assert!(bytes <= self.bytes);
        let mut state = self
            .budget
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        state.used -= bytes;
        self.bytes -= bytes;
    }

    pub(super) fn retain_after_cleanup_failure(
        mut self,
        path: TemporaryPath,
        file: File,
        mut node: PendingCleanupOwner,
    ) {
        assert!(node.node.cleanup.is_none());
        node.node.cleanup = Some(PendingCleanup {
            path,
            file,
            bytes: self.bytes,
        });
        self.budget
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .pending
            .push(node);
        self.bytes = 0;
    }

    pub(super) fn transfer_to(&mut self, destination: &mut Self, bytes: u64) -> Result<()> {
        if !Arc::ptr_eq(&self.budget, &destination.budget) || bytes > self.bytes {
            return Err(Error::new(
                ErrorCode::FileFormatError,
                "invalid temporary disk reservation transfer",
            ));
        }
        self.bytes -= bytes;
        destination.bytes += bytes;
        Ok(())
    }
}

impl Drop for DiskReservation {
    fn drop(&mut self) {
        self.release(self.bytes);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn large_pending_lists_drop_iteratively_on_a_small_stack() {
        std::thread::Builder::new()
            .stack_size(64 * 1024)
            .spawn(|| {
                let mut pending = PendingCleanups::default();
                for _ in 0..100_000 {
                    pending.push(PendingCleanupOwner::default());
                }
                assert_eq!(pending.len, 100_000);
                let (_, allocations) = crate::test_alloc::allocations(|| drop(pending));
                assert_eq!(allocations, 0);
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn preflight_is_read_only_and_actual_reservations_recheck_changed_usage() {
        for limit in [0, 1, 1024, u64::MAX] {
            let budget = Arc::new(TemporaryDiskBudget::new(limit));
            budget.check_additional(limit).unwrap();
            assert_eq!(budget.used(), 0);
            let mut other = budget.reservation();
            other.grow(limit).unwrap();
            budget.check_additional(0).unwrap();
            let preflight = budget.check_additional(1).unwrap_err();
            let mut actual = budget.reservation();
            let allocation = actual.grow(1).unwrap_err();
            assert_eq!(preflight.code(), ErrorCode::ResourceExhausted);
            assert_eq!(preflight.to_json(), allocation.to_json());
            assert!(preflight
                .detail()
                .iter()
                .any(|(key, value)| key == "requested_bytes"
                    && value == &(u128::from(limit) + 1).to_string()));
            assert_eq!(budget.used(), limit);
            drop(other);
            budget.check_additional(limit).unwrap();
            actual.grow(limit).unwrap();
            drop(actual);
            assert_eq!(budget.used(), 0);
        }
    }

    #[test]
    fn transfers_preserve_shared_usage_and_reject_invalid_owners_or_amounts() {
        for limit in [0, 1, 1024, u64::MAX] {
            for seed in 0..100 {
                let budget = Arc::new(TemporaryDiskBudget::new(limit));
                let other = Arc::new(TemporaryDiskBudget::new(limit));
                let mut source = budget.reservation();
                let mut destination = budget.reservation();
                let mut foreign = other.reservation();
                source.grow(limit).unwrap();
                let moved = (limit / 100) * seed;
                source.transfer_to(&mut destination, moved).unwrap();
                assert_eq!(source.bytes, limit - moved);
                assert_eq!(destination.bytes, moved);
                assert_eq!(budget.used(), limit);
                assert!(source.transfer_to(&mut foreign, 0).is_err());
                if let Some(excess) = source.bytes.checked_add(1) {
                    assert!(source.transfer_to(&mut destination, excess).is_err());
                }
                assert_eq!(budget.used(), limit);
                assert_eq!(other.used(), 0);
                drop(source);
                assert_eq!(budget.used(), moved);
                drop(destination);
                assert_eq!(budget.used(), 0);
            }
        }
    }

    #[test]
    fn reservations_share_the_limit_and_release_exactly_across_errors_and_overflow() {
        for limit in [0, 1, 1024, u64::MAX] {
            let budget = Arc::new(TemporaryDiskBudget::new(limit));
            let mut first = budget.reservation();
            let mut second = budget.reservation();
            first.grow(limit / 2).unwrap();
            second.grow(limit - limit / 2).unwrap();
            assert_eq!(budget.used(), limit);
            let error = first.grow(1).unwrap_err();
            assert_eq!(error.code(), ErrorCode::ResourceExhausted);
            assert!(error
                .detail()
                .iter()
                .any(|(key, value)| key == "requested_bytes"
                    && value == &(u128::from(limit) + 1).to_string()));
            assert_eq!(budget.used(), limit);
            drop(first);
            assert_eq!(budget.used(), limit - limit / 2);
            second.grow(limit / 2).unwrap();
            assert_eq!(budget.used(), limit);
            drop(second);
            assert_eq!(budget.used(), 0);
        }
    }

    #[test]
    fn concurrent_operations_cannot_multiply_the_runtime_disk_limit() {
        let budget = Arc::new(TemporaryDiskBudget::new(4096));
        let barrier = Arc::new(std::sync::Barrier::new(8));
        std::thread::scope(|scope| {
            for worker in 0..8 {
                let budget = Arc::clone(&budget);
                let barrier = Arc::clone(&barrier);
                scope.spawn(move || {
                    let mut work = budget.reservation();
                    work.grow(512).unwrap();
                    barrier.wait();
                    assert_eq!(budget.used(), 4096);
                    assert!(work.grow(1).is_err());
                    barrier.wait();
                    drop(work);
                    for iteration in 0..100 {
                        let mut work = budget.reservation();
                        let count = 1 + (worker * 17 + iteration * 31) % 512;
                        work.grow(count).unwrap();
                        assert!(budget.used() <= 4096);
                    }
                });
            }
        });
        assert_eq!(budget.used(), 0);
    }
}
