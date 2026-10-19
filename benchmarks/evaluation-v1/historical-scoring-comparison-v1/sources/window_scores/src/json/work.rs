use std::collections::BinaryHeap;
use std::sync::{Arc, Mutex};

use super::{JsonInputKind, JsonLimitReason, JsonProcessingPolicyV1};
use crate::{Error, ErrorCode, Result};

pub(crate) struct WorkLedger {
    state: Mutex<LedgerState>,
}

struct LedgerState {
    limit: usize,
    total_limit: usize,
    catalog_limit: usize,
    used: usize,
    workspace_used: usize,
    catalog_used: usize,
    cache_used: usize,
}

impl Default for WorkLedger {
    fn default() -> Self {
        // Working memory retains its existing ceiling. Persistent state has a
        // separate quota; neither limit is a promise about process RSS.
        Self::with_limits(
            JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
            Self::CATALOG_LIMIT_BYTES,
            Self::TOTAL_LIMIT_BYTES,
        )
    }
}

impl WorkLedger {
    pub(crate) const CATALOG_LIMIT_BYTES: usize = 2 * 1024 * 1024 * 1024;
    const TOTAL_LIMIT_BYTES: usize = 3 * 1024 * 1024 * 1024;

    #[cfg(test)]
    fn new(limit: usize) -> Self {
        Self::with_limits(limit, limit, limit)
    }

    pub(crate) fn with_limits(limit: usize, catalog_limit: usize, total_limit: usize) -> Self {
        Self {
            state: Mutex::new(LedgerState {
                limit,
                total_limit,
                catalog_limit,
                used: 0,
                workspace_used: 0,
                catalog_used: 0,
                cache_used: 0,
            }),
        }
    }
}

impl LedgerState {
    fn try_admit_cache(&mut self, additional: u128, limit: usize) -> bool {
        if additional > limit as u128 {
            return false;
        }
        let cache = self.cache_used as u128 + additional;
        let total =
            cache + self.used as u128 + self.workspace_used as u128 + self.catalog_used as u128;
        if cache > limit as u128 || total > self.total_limit as u128 {
            return false;
        }
        self.cache_used = cache as usize;
        true
    }

    fn working_available(&self) -> usize {
        let working = self.used + self.workspace_used;
        (self.limit - working).min(self.total_limit - working - self.catalog_used - self.cache_used)
    }

    fn admit_workspace(&self, extra: u128, phase: &'static str) -> Result<()> {
        let working = self.used as u128 + self.workspace_used as u128 + extra;
        if working > self.limit as u128 {
            return Err(Error::resource_exhausted(
                "memory",
                phase,
                self.limit as u64,
                working,
            ));
        }
        let total = working + self.catalog_used as u128 + self.cache_used as u128;
        if total > self.total_limit as u128 {
            return Err(Error::resource_exhausted(
                "memory",
                phase,
                self.total_limit as u64,
                total,
            ));
        }
        Ok(())
    }

    fn admit_catalog(&self, additional: u128, transferred: usize) -> Result<()> {
        let catalog = self.catalog_used as u128 + additional;
        if catalog > self.catalog_limit as u128 {
            return Err(Error::resource_exhausted(
                "catalog",
                "catalog_build",
                self.catalog_limit as u64,
                catalog,
            ));
        }
        let total =
            self.used as u128 + self.workspace_used as u128 + catalog + self.cache_used as u128
                - transferred as u128;
        if total > self.total_limit as u128 {
            return Err(Error::resource_exhausted(
                "memory",
                "catalog_build",
                self.total_limit as u64,
                total,
            ));
        }
        Ok(())
    }
}

pub(crate) struct CacheReservation {
    ledger: Arc<WorkLedger>,
    bytes: usize,
}

impl CacheReservation {
    pub(crate) fn assert_same_ledger(&self, other: &Self) {
        assert!(Arc::ptr_eq(&self.ledger, &other.ledger));
    }

    pub(crate) fn try_grow(&mut self, additional: u128, limit: usize) -> bool {
        let mut ledger = self
            .ledger
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if !ledger.try_admit_cache(additional, limit) {
            return false;
        }
        self.bytes += additional as usize;
        true
    }

    pub(crate) fn transfer_to(&mut self, target: &mut Self, bytes: usize) {
        self.assert_same_ledger(target);
        assert!(bytes <= self.bytes);
        // Both guards are exclusively borrowed; the shared total never changes.
        self.bytes -= bytes;
        target.bytes += bytes;
    }

    // Adopt an existing allocation without releasing or charging its shared
    // memory twice. Only the optional cache sub-limit needs new admission.
    pub(crate) fn absorb_workspace(
        &mut self,
        work: &mut WorkspaceReservation,
        bytes: usize,
        limit: usize,
    ) -> bool {
        assert!(bytes <= work.bytes);
        let mut operation = work
            .context
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        assert!(Arc::ptr_eq(&operation.ledger, &self.ledger));
        let mut ledger = self
            .ledger
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if ledger.cache_used as u128 + bytes as u128 > limit as u128 {
            return false;
        }
        ledger.workspace_used -= bytes;
        operation.workspace_used -= bytes;
        ledger.cache_used += bytes;
        work.bytes -= bytes;
        self.bytes += bytes;
        true
    }

    pub(crate) fn reserved_bytes(&self) -> usize {
        self.bytes
    }

    pub(crate) fn retain(&mut self, bytes: usize) {
        assert!(bytes <= self.bytes);
        let mut ledger = self
            .ledger
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        ledger.cache_used -= self.bytes - bytes;
        self.bytes = bytes;
    }
}

impl Drop for CacheReservation {
    fn drop(&mut self) {
        self.retain(0);
    }
}

pub(crate) struct CatalogReservation {
    ledger: Arc<WorkLedger>,
    bytes: usize,
}

impl CatalogReservation {
    pub(crate) fn attach_context(&self, context: &JsonOperationContext) -> Result<()> {
        context.attach(Arc::clone(&self.ledger), JsonInputKind::StoredJson)
    }

    pub(crate) fn reserved_bytes(&self) -> usize {
        self.bytes
    }

    pub(crate) fn grow(&mut self, additional: usize) -> Result<()> {
        let mut ledger = self
            .ledger
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        ledger.admit_catalog(additional as u128, 0)?;
        ledger.catalog_used += additional;
        self.bytes += additional;
        Ok(())
    }

    pub(crate) fn retain(&mut self, bytes: usize) {
        assert!(bytes <= self.bytes);
        let mut ledger = self
            .ledger
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        ledger.catalog_used -= self.bytes - bytes;
        self.bytes = bytes;
    }

    // Move already admitted allocations, including their shared memory credit,
    // without briefly releasing it or charging the same allocation twice.
    pub(crate) fn absorb(
        &mut self,
        json: &mut JsonReservation,
        mut workspace: Option<&mut WorkspaceReservation>,
        json_bytes: usize,
        workspace_bytes: usize,
    ) -> Result<()> {
        assert!(json_bytes <= json.covered);
        assert!(workspace_bytes <= workspace.as_ref().map_or(0, |work| work.bytes));
        if let Some(workspace) = &workspace {
            assert!(Arc::ptr_eq(&json.context.state, &workspace.context.state));
        }
        let mut operation = json
            .context
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        assert!(Arc::ptr_eq(&operation.ledger, &self.ledger));
        let mut ledger = self
            .ledger
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let bytes = json_bytes + workspace_bytes;
        ledger.admit_catalog(bytes as u128, bytes)?;
        ledger.used -= json_bytes;
        operation.used -= json_bytes;
        ledger.workspace_used -= workspace_bytes;
        operation.workspace_used -= workspace_bytes;
        ledger.catalog_used += bytes;
        self.bytes += bytes;
        json.bytes -= json_bytes;
        json.covered -= json_bytes;
        if let Some(workspace) = &mut workspace {
            workspace.bytes -= workspace_bytes;
        }
        Ok(())
    }
}

impl Drop for CatalogReservation {
    fn drop(&mut self) {
        self.retain(0);
    }
}

#[derive(Clone)]
pub(crate) struct JsonOperationContext {
    state: Arc<Mutex<OperationState>>,
}

struct OperationState {
    ledger: Arc<WorkLedger>,
    used: usize,
    workspace_used: usize,
}

impl JsonOperationContext {
    #[cfg(test)]
    pub(crate) fn available_workspace_bytes(&self) -> usize {
        let operation = self.state.lock().unwrap_or_else(|error| error.into_inner());
        let ledger = operation
            .ledger
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        ledger.working_available()
    }

    pub(crate) fn new(ledger: Arc<WorkLedger>) -> Self {
        Self {
            state: Arc::new(Mutex::new(OperationState {
                ledger,
                used: 0,
                workspace_used: 0,
            })),
        }
    }

    pub(crate) fn standalone() -> Self {
        Self::new(Arc::new(WorkLedger::default()))
    }

    pub(crate) fn reservation(&self) -> JsonReservation {
        JsonReservation {
            context: self.clone(),
            bytes: 0,
            covered: 0,
            aggregate: false,
        }
    }

    pub(super) fn aggregate_reservation(&self) -> JsonReservation {
        let mut work = self.reservation();
        work.start_aggregate();
        work
    }

    pub(crate) fn workspace_reservation(&self) -> WorkspaceReservation {
        WorkspaceReservation {
            context: self.clone(),
            bytes: 0,
        }
    }

    pub(crate) fn catalog_reservation(&self) -> CatalogReservation {
        let operation = self.state.lock().unwrap_or_else(|error| error.into_inner());
        CatalogReservation {
            ledger: Arc::clone(&operation.ledger),
            bytes: 0,
        }
    }

    // Cache admission is optional, but its live allocations share the parent
    // memory ceiling with working memory and Catalog state.
    pub(crate) fn cache_reservation(&self, bytes: u128, limit: usize) -> Option<CacheReservation> {
        let operation = self.state.lock().unwrap_or_else(|error| error.into_inner());
        let mut ledger = operation
            .ledger
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if !ledger.try_admit_cache(bytes, limit) {
            return None;
        }
        Some(CacheReservation {
            ledger: Arc::clone(&operation.ledger),
            bytes: bytes as usize,
        })
    }

    pub(crate) fn attach(&self, target: Arc<WorkLedger>, kind: JsonInputKind) -> Result<()> {
        let mut operation = self.state.lock().unwrap_or_else(|error| error.into_inner());
        let previous = Arc::clone(&operation.ledger);
        if Arc::ptr_eq(&previous, &target) {
            return Ok(());
        }
        let transfer = |source: &mut LedgerState, destination: &mut LedgerState| {
            let limit =
                JsonProcessingPolicyV1::WORK_BUDGET_BYTES.min(destination.working_available());
            if operation.used > limit {
                return Err(kind.limit_error(
                    JsonLimitReason::WorkBudget,
                    limit,
                    Some(operation.used),
                ));
            }
            destination.admit_workspace(
                operation.used as u128 + operation.workspace_used as u128,
                "operation_scratch",
            )?;
            source.used -= operation.used;
            destination.used += operation.used;
            source.workspace_used -= operation.workspace_used;
            destination.workspace_used += operation.workspace_used;
            Ok(())
        };
        // A stable lock order also permits simultaneous transfers in opposite directions.
        if Arc::as_ptr(&previous) < Arc::as_ptr(&target) {
            let mut source = previous
                .state
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            let mut destination = target
                .state
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            transfer(&mut source, &mut destination)?;
        } else {
            let mut destination = target
                .state
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            let mut source = previous
                .state
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            transfer(&mut source, &mut destination)?;
        }
        operation.ledger = target;
        Ok(())
    }

    pub(crate) fn limit_error(&self, kind: JsonInputKind) -> crate::Error {
        let operation = self.state.lock().unwrap_or_else(|error| error.into_inner());
        let ledger = operation
            .ledger
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let limit = effective_limit(&operation, &ledger);
        kind.limit_error(JsonLimitReason::WorkBudget, limit, None)
    }

    fn available(&self) -> usize {
        let operation = self.state.lock().unwrap_or_else(|error| error.into_inner());
        let ledger = operation
            .ledger
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        effective_limit(&operation, &ledger) - operation.used
    }
}

fn effective_limit(operation: &OperationState, ledger: &LedgerState) -> usize {
    JsonProcessingPolicyV1::WORK_BUDGET_BYTES.min(ledger.working_available() + operation.used)
}

// This is a lease for bounded working buffers, not an allocator/RSS measurement.
// Keep it alive until the buffers are freed or transfer it with their ownership.
pub(crate) struct WorkspaceReservation {
    context: JsonOperationContext,
    bytes: usize,
}

impl std::fmt::Debug for WorkspaceReservation {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        out.debug_struct("WorkspaceReservation")
            .field("bytes", &self.bytes)
            .finish_non_exhaustive()
    }
}

impl WorkspaceReservation {
    pub(crate) fn take_json(
        &mut self,
        bytes: usize,
        kind: JsonInputKind,
    ) -> Result<JsonReservation> {
        assert!(bytes <= self.bytes);
        let mut operation = self
            .context
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let owner = Arc::clone(&operation.ledger);
        let mut ledger = owner
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let limit = JsonProcessingPolicyV1::WORK_BUDGET_BYTES
            .min(ledger.working_available() + operation.used + bytes);
        let requested = operation.used + bytes;
        if requested > limit {
            return Err(kind.limit_error(JsonLimitReason::WorkBudget, limit, Some(requested)));
        }
        operation.workspace_used -= bytes;
        ledger.workspace_used -= bytes;
        operation.used += bytes;
        ledger.used += bytes;
        self.bytes -= bytes;
        Ok(JsonReservation {
            context: self.context.clone(),
            bytes,
            covered: bytes,
            aggregate: false,
        })
    }

    pub(crate) fn context(&self) -> &JsonOperationContext {
        &self.context
    }

    pub(crate) fn merge(&mut self, mut other: Self) {
        assert!(Arc::ptr_eq(&self.context.state, &other.context.state));
        self.bytes += other.bytes;
        other.bytes = 0;
    }

    pub(crate) fn reserved_bytes(&self) -> usize {
        self.bytes
    }

    pub(crate) fn retain(&mut self, bytes: usize) {
        assert!(bytes <= self.bytes);
        if bytes == self.bytes {
            return;
        }
        let mut operation = self
            .context
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let owner = Arc::clone(&operation.ledger);
        let mut ledger = owner
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let released = self.bytes - bytes;
        operation.workspace_used -= released;
        ledger.workspace_used -= released;
        self.bytes = bytes;
    }

    pub(crate) fn grow_to(
        &mut self,
        minimum: usize,
        preferred: usize,
        phase: &'static str,
    ) -> Result<usize> {
        self.grow_to_wide(minimum as u128, preferred as u128, phase)
    }

    pub(crate) fn grow_to_wide(
        &mut self,
        minimum: u128,
        preferred: u128,
        phase: &'static str,
    ) -> Result<usize> {
        assert!(minimum <= preferred);
        let mut operation = self
            .context
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let owner = Arc::clone(&operation.ledger);
        let mut ledger = owner
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let minimum_extra = minimum.saturating_sub(self.bytes as u128);
        ledger.admit_workspace(minimum_extra, phase)?;
        let additional = preferred
            .saturating_sub(self.bytes as u128)
            .min(ledger.working_available() as u128) as usize;
        ledger.workspace_used += additional;
        operation.workspace_used += additional;
        self.bytes += additional;
        Ok(self.bytes)
    }
}

impl Drop for WorkspaceReservation {
    fn drop(&mut self) {
        self.retain(0);
    }
}

pub(crate) fn reserve_heap_slot<T: Ord>(
    heap: &mut BinaryHeap<T>,
    work: &mut WorkspaceReservation,
    limit: usize,
) -> Result<()> {
    if heap.len() < heap.capacity() {
        return Ok(());
    }
    let size = std::mem::size_of::<T>();
    let old_capacity = heap.capacity();
    let before = work.reserved_bytes();
    let payload = before - old_capacity * size;
    let minimum = heap.len() as u128 + 1;
    let preferred = (old_capacity as u128 * 2).max(4).min(limit as u128);
    let granted = work.grow_to_wide(
        before as u128 + minimum * size as u128,
        before as u128 + preferred * size as u128,
        "operation_scratch",
    )?;
    let capacity = (granted - before) / size;
    // Keep both arrays charged until the old allocation has actually been replaced.
    work.retain(before + capacity * size);
    if let Err(error) = heap.try_reserve_exact(capacity - heap.len()) {
        work.retain(before);
        return Err(std::io::Error::other(error.to_string()).into());
    }
    if heap.capacity() != capacity {
        drop(std::mem::take(heap));
        work.retain(payload);
        return Err(std::io::Error::other("candidate capacity exceeds reservation").into());
    }
    work.retain(payload + capacity * size);
    Ok(())
}

pub(crate) fn try_shrink_heap<T: Ord>(
    heap: &mut BinaryHeap<T>,
    work: &mut WorkspaceReservation,
) -> Result<bool> {
    let capacity = heap.capacity();
    let count = heap.len();
    let size = std::mem::size_of::<T>();
    if count == capacity || size == 0 {
        return Ok(false);
    }
    let before = work.reserved_bytes();
    let retained = before - capacity * size + count * size;
    if count == 0 {
        drop(std::mem::take(heap));
        work.retain(retained);
        return Ok(true);
    }
    let peak = before as u128 + count as u128 * size as u128;
    match work.grow_to_wide(peak, peak, "operation_scratch") {
        Err(error) if error.code() == ErrorCode::ResourceExhausted => return Ok(false),
        result => {
            result?;
        }
    }
    let mut values = Vec::new();
    if let Err(error) = values.try_reserve_exact(count) {
        drop(values);
        work.retain(before);
        return Err(std::io::Error::other(error.to_string()).into());
    }
    if values.capacity() != count {
        drop(values);
        work.retain(before);
        return Err(std::io::Error::other("candidate capacity exceeds reservation").into());
    }
    // Transfer the payloads without cloning; keep the old array charged until freed.
    let mut old = std::mem::take(heap).into_vec();
    values.append(&mut old);
    drop(old);
    *heap = BinaryHeap::from(values);
    work.retain(retained);
    Ok(true)
}

// The array must be freed before its lease. Only an explicit ownership transfer
// to a public result may detach it; growth beyond the admitted capacity is forbidden.
pub(crate) struct ReservedVec<T> {
    values: Vec<T>,
    _work: WorkspaceReservation,
}

impl<T: std::fmt::Debug> std::fmt::Debug for ReservedVec<T> {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.values.fmt(out)
    }
}

impl<T> ReservedVec<T> {
    #[cfg(test)]
    pub(crate) fn assert_operation_context(&self, context: &JsonOperationContext) {
        let mut probe = self._work.context().workspace_reservation();
        probe.merge(context.workspace_reservation());
    }

    #[cfg(test)]
    pub(crate) fn reserved_bytes(&self) -> usize {
        self._work.reserved_bytes()
    }

    pub(crate) fn with_capacity(
        count: usize,
        context: &JsonOperationContext,
        phase: &'static str,
    ) -> Result<Self> {
        let mut work = context.workspace_reservation();
        let bytes = (count as u128) * (std::mem::size_of::<T>() as u128);
        work.grow_to_wide(bytes, bytes, phase)?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(count)
            .map_err(|error| std::io::Error::other(error.to_string()))?;
        if std::mem::size_of::<T>() != 0 && values.capacity() != count {
            return Err(
                std::io::Error::other("array allocation exceeds requested capacity").into(),
            );
        }
        Ok(Self {
            values,
            _work: work,
        })
    }

    pub(crate) fn as_slice(&self) -> &[T] {
        &self.values
    }

    pub(crate) fn capacity(&self) -> usize {
        self.values.capacity()
    }

    pub(crate) fn context(&self) -> &JsonOperationContext {
        self._work.context()
    }

    pub(crate) fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.values
    }

    pub(crate) fn retain(&mut self, keep: impl FnMut(&T) -> bool) {
        self.values.retain(keep);
    }

    pub(crate) fn push(&mut self, value: T) {
        assert!(
            self.values.len() < self.values.capacity(),
            "reserved array capacity exceeded"
        );
        self.values.push(value);
    }

    pub(crate) fn try_push(&mut self, value: T) -> Result<()> {
        if self.values.len() == self.values.capacity() {
            let size = std::mem::size_of::<T>();
            let before = self._work.reserved_bytes();
            let minimum = self
                .values
                .len()
                .checked_add(1)
                .ok_or_else(|| std::io::Error::other("array capacity overflow"))?
                as u128;
            let preferred = (self.values.capacity() as u128 * 2).max(4);
            let granted = self._work.grow_to_wide(
                before as u128 + minimum * size as u128,
                before as u128 + preferred * size as u128,
                "operation_scratch",
            )?;
            let capacity = (granted - before) / size;
            // The old array remains charged until its replacement is allocated.
            self._work.retain(before + capacity * size);
            if let Err(error) = self.values.try_reserve_exact(capacity - self.values.len()) {
                self._work.retain(before);
                return Err(std::io::Error::other(error.to_string()).into());
            }
            if self.values.capacity() != capacity {
                drop(std::mem::take(&mut self.values));
                self._work.retain(0);
                return Err(std::io::Error::other("array capacity exceeds reservation").into());
            }
            self._work.retain(capacity * size);
        }
        self.values.push(value);
        Ok(())
    }

    pub(crate) fn into_vec(self) -> Vec<T> {
        self.values
    }

    pub(crate) fn into_parts(self) -> (Vec<T>, WorkspaceReservation) {
        (self.values, self._work)
    }

    pub(crate) fn from_reserved(values: Vec<T>, mut work: WorkspaceReservation) -> Self {
        work.retain(values.capacity() * std::mem::size_of::<T>());
        Self {
            values,
            _work: work,
        }
    }
}

impl<T> AsRef<[T]> for ReservedVec<T> {
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl ReservedVec<u8> {
    pub(crate) fn zeroed(
        count: usize,
        context: &JsonOperationContext,
        phase: &'static str,
    ) -> Result<Self> {
        let mut bytes = Self::with_capacity(count, context, phase)?;
        bytes.values.resize(count, 0);
        Ok(bytes)
    }
}

pub(crate) struct ReservedString {
    value: String,
    _work: WorkspaceReservation,
}

impl ReservedString {
    pub(crate) fn into_parts(self) -> (String, WorkspaceReservation) {
        (self.value, self._work)
    }

    pub(crate) fn copy(value: &str, context: &JsonOperationContext) -> Result<Self> {
        Self::copy_in(value, context, "operation_scratch")
    }

    pub(crate) fn copy_in(
        value: &str,
        context: &JsonOperationContext,
        phase: &'static str,
    ) -> Result<Self> {
        let mut owned = Self::with_capacity(value.len(), context, phase)?;
        owned.value.push_str(value);
        Ok(owned)
    }

    // Internal renderers must be deterministic. The second pass is bounded even
    // if a renderer changes its output after sizing.
    pub(crate) fn build(
        context: &JsonOperationContext,
        phase: &'static str,
        render: impl Fn(&mut dyn std::fmt::Write) -> std::fmt::Result,
    ) -> Result<Self> {
        struct Length(usize);
        impl std::fmt::Write for Length {
            fn write_str(&mut self, value: &str) -> std::fmt::Result {
                self.0 = self.0.checked_add(value.len()).ok_or(std::fmt::Error)?;
                Ok(())
            }
        }
        let mut length = Length(0);
        render(&mut length).map_err(|_| std::io::Error::other("reserved string sizing failed"))?;
        Self::build_exact(length.0, context, phase, render)
    }

    pub(crate) fn build_exact(
        bytes: usize,
        context: &JsonOperationContext,
        phase: &'static str,
        render: impl FnOnce(&mut dyn std::fmt::Write) -> std::fmt::Result,
    ) -> Result<Self> {
        let mut owned = Self::with_capacity(bytes, context, phase)?;
        owned.append(render)?;
        if owned.value.len() != bytes {
            return Err(std::io::Error::other("reserved string formatting failed").into());
        }
        Ok(owned)
    }

    pub(crate) fn append(
        &mut self,
        render: impl FnOnce(&mut dyn std::fmt::Write) -> std::fmt::Result,
    ) -> Result<()> {
        struct Bounded<'a>(&'a mut String);
        impl std::fmt::Write for Bounded<'_> {
            fn write_str(&mut self, value: &str) -> std::fmt::Result {
                if value.len() > self.0.capacity() - self.0.len() {
                    return Err(std::fmt::Error);
                }
                self.0.push_str(value);
                Ok(())
            }
        }
        render(&mut Bounded(&mut self.value))
            .map_err(|_| std::io::Error::other("reserved string formatting failed").into())
    }

    pub(crate) fn with_capacity(
        bytes: usize,
        context: &JsonOperationContext,
        phase: &'static str,
    ) -> Result<Self> {
        let mut work = context.workspace_reservation();
        work.grow_to(bytes, bytes, phase)?;
        let mut owned = String::new();
        owned
            .try_reserve_exact(bytes)
            .map_err(|error| std::io::Error::other(error.to_string()))?;
        if owned.capacity() != bytes {
            return Err(
                std::io::Error::other("string allocation exceeds requested capacity").into(),
            );
        }
        Ok(Self::from_reserved(owned, work))
    }

    pub(crate) fn from_reserved(value: String, mut work: WorkspaceReservation) -> Self {
        work.retain(value.capacity());
        Self { value, _work: work }
    }
}

impl AsRef<str> for ReservedString {
    fn as_ref(&self) -> &str {
        &self.value
    }
}

pub(crate) struct JsonReservation {
    context: JsonOperationContext,
    // Charged blocks may include spare capacity; coverage only backs live work.
    bytes: usize,
    covered: usize,
    aggregate: bool,
}

impl JsonReservation {
    const BLOCK_BYTES: usize = 4096;

    pub(super) fn start_aggregate(&mut self) {
        self.aggregate = true;
    }

    pub(super) fn finish_aggregate(&mut self) {
        self.retain(self.covered);
        self.aggregate = false;
    }

    pub(super) fn reserve_object_keys<'a>(
        &mut self,
        keys: impl ExactSizeIterator<Item = &'a str>,
        kind: JsonInputKind,
    ) -> Result<Self> {
        let entries = keys.len();
        let bytes = keys.fold(object_workspace(entries), |bytes, key| {
            bytes.saturating_add(key.len())
        });
        let reused = bytes.min(self.bytes - self.covered);
        let mut work = self.scratch();
        self.bytes -= reused;
        work.bytes = reused;
        work.covered = reused;
        work.grow(bytes - reused, kind)?;
        Ok(work)
    }

    pub(crate) fn take_workspace(&mut self, bytes: usize) -> WorkspaceReservation {
        assert!(bytes <= self.covered);
        let mut operation = self
            .context
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let owner = Arc::clone(&operation.ledger);
        let mut ledger = owner
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        operation.used -= bytes;
        ledger.used -= bytes;
        operation.workspace_used += bytes;
        ledger.workspace_used += bytes;
        self.bytes -= bytes;
        self.covered -= bytes;
        WorkspaceReservation {
            context: self.context.clone(),
            bytes,
        }
    }

    pub(super) fn scratch(&self) -> Self {
        self.context.reservation()
    }

    pub(super) fn cover(&mut self, bytes: usize, kind: JsonInputKind) -> Result<()> {
        self.grow_with_granularity(bytes.saturating_sub(self.covered), kind, true)
    }

    pub(crate) fn retain(&mut self, bytes: usize) {
        assert!(bytes <= self.bytes);
        self.refund(self.bytes - bytes);
        self.covered = bytes;
    }

    pub(crate) fn reserved_bytes(&self) -> usize {
        self.bytes
    }

    pub(crate) fn covered_bytes(&self) -> usize {
        self.covered
    }

    pub(crate) fn copy_string(&mut self, value: &str, kind: JsonInputKind) -> Result<String> {
        let mut copy = String::new();
        self.reserve_string(&mut copy, value.len(), kind)?;
        copy.push_str(value);
        Ok(copy)
    }

    pub(super) fn copy_validated_value(
        &mut self,
        value: &super::JsonValue,
        kind: JsonInputKind,
    ) -> Result<super::JsonValue> {
        self.retain(self.covered);
        let (value, work) = value.clone_output_in(kind, &self.context)?.into_parts();
        self.merge(work);
        Ok(value)
    }

    pub(crate) fn merge(&mut self, mut other: Self) {
        assert!(Arc::ptr_eq(&self.context.state, &other.context.state));
        self.bytes += other.bytes;
        self.covered += other.covered;
        other.bytes = 0;
        other.covered = 0;
    }

    pub(crate) fn grow(&mut self, additional: usize, kind: JsonInputKind) -> Result<()> {
        self.grow_with_granularity(additional, kind, self.aggregate)
    }

    fn grow_with_granularity(
        &mut self,
        additional: usize,
        kind: JsonInputKind,
        aggregate: bool,
    ) -> Result<()> {
        if additional == 0 {
            return Ok(());
        }
        let needed = self.covered.checked_add(additional);
        if let Some(needed) = needed.filter(|bytes| *bytes <= self.bytes) {
            self.covered = needed;
            return Ok(());
        }
        let mut operation = self
            .context
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let owner = Arc::clone(&operation.ledger);
        let mut ledger = owner
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let limit = effective_limit(&operation, &ledger);
        let extra = needed.map(|bytes| bytes - self.bytes);
        let requested = extra.and_then(|extra| operation.used.checked_add(extra));
        if !requested.is_some_and(|bytes| bytes <= limit) {
            return Err(kind.limit_error(JsonLimitReason::WorkBudget, limit, requested));
        }
        let needed = needed.unwrap();
        let preferred = if aggregate {
            needed
                .checked_add(Self::BLOCK_BYTES - 1)
                .map_or(needed, |bytes| {
                    bytes / Self::BLOCK_BYTES * Self::BLOCK_BYTES
                })
        } else {
            needed
        };
        let granted = if preferred - self.bytes <= limit - operation.used {
            preferred - self.bytes
        } else {
            needed - self.bytes
        };
        operation.used += granted;
        ledger.used += granted;
        self.bytes += granted;
        self.covered = needed;
        Ok(())
    }

    fn available_for(&self, minimum: usize) -> usize {
        let spare = self.bytes - self.covered;
        if spare >= minimum {
            spare
        } else {
            spare + self.context.available()
        }
    }

    pub(crate) fn release(&mut self, bytes: usize) {
        assert!(bytes <= self.covered);
        self.covered -= bytes;
        if !self.aggregate {
            self.refund(bytes);
        }
    }

    fn refund(&mut self, bytes: usize) {
        if bytes == 0 {
            return;
        }
        assert!(bytes <= self.bytes);
        let mut operation = self
            .context
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let owner = Arc::clone(&operation.ledger);
        let mut ledger = owner
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        self.bytes -= bytes;
        operation.used -= bytes;
        ledger.used -= bytes;
    }

    pub(crate) fn reserve_vec<T>(
        &mut self,
        values: &mut Vec<T>,
        additional: usize,
        kind: JsonInputKind,
    ) -> Result<()> {
        let needed = values
            .len()
            .checked_add(additional)
            .ok_or_else(|| self.context.limit_error(kind))?;
        if needed <= values.capacity() || std::mem::size_of::<T>() == 0 {
            return Ok(());
        }
        let old = values.capacity();
        let mut target = needed
            .max(old.saturating_mul(2))
            .max(4)
            .min(old.saturating_add(
                self.available_for((needed - old).saturating_mul(std::mem::size_of::<T>()))
                    / std::mem::size_of::<T>(),
            ))
            .max(needed);
        let mut bytes = (target - old)
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| self.context.limit_error(kind))?;
        if let Err(error) = self.grow(bytes, kind) {
            if target == needed {
                return Err(error);
            }
            target = needed;
            bytes = (target - old)
                .checked_mul(std::mem::size_of::<T>())
                .ok_or_else(|| self.context.limit_error(kind))?;
            self.grow(bytes, kind)?;
        }
        if let Err(error) = values.try_reserve_exact(target - values.len()) {
            self.release(bytes);
            return Err(std::io::Error::other(error.to_string()).into());
        }
        let extra = (values.capacity() - target) * std::mem::size_of::<T>();
        self.grow(extra, kind)
    }

    pub(crate) fn reserve_string(
        &mut self,
        value: &mut String,
        additional: usize,
        kind: JsonInputKind,
    ) -> Result<()> {
        let needed = value
            .len()
            .checked_add(additional)
            .ok_or_else(|| self.context.limit_error(kind))?;
        if needed <= value.capacity() {
            return Ok(());
        }
        let old = value.capacity();
        let mut target = needed
            .max(old.saturating_mul(2))
            .max(8)
            .min(old.saturating_add(self.available_for(needed - old)))
            .max(needed);
        if let Err(error) = self.grow(target - old, kind) {
            if target == needed {
                return Err(error);
            }
            target = needed;
            self.grow(target - old, kind)?;
        }
        if let Err(error) = value.try_reserve_exact(target - value.len()) {
            self.release(target - old);
            return Err(std::io::Error::other(error.to_string()).into());
        }
        self.grow(value.capacity() - target, kind)
    }
}

// The enclosing aggregate reservation already covers this string's capacity.
pub(crate) fn copy_precharged_string(value: &str) -> Result<String> {
    let mut owned = String::new();
    owned
        .try_reserve_exact(value.len())
        .map_err(|error| std::io::Error::other(error.to_string()))?;
    if owned.capacity() != value.len() {
        return Err(std::io::Error::other("string allocation exceeds requested capacity").into());
    }
    owned.push_str(value);
    Ok(owned)
}

impl Drop for JsonReservation {
    fn drop(&mut self) {
        self.refund(self.bytes);
    }
}

// Reserve a full internal node for every possible minimally occupied B-tree
// node. This is workspace headroom, not an assertion about allocator/RSS bytes.
pub(super) fn object_workspace(entries: usize) -> usize {
    btree_workspace::<String, super::JsonValue>(entries)
}

pub(crate) fn btree_workspace<K, V>(entries: usize) -> usize {
    if entries == 0 {
        return 0;
    }
    let node = 11 * std::mem::size_of::<(K, V)>() + 16 * std::mem::size_of::<usize>();
    (1 + (entries - 1) / 5).saturating_mul(node)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aggregate_growth_reuses_blocks_and_falls_back_to_the_true_minimum() {
        let kind = JsonInputKind::Filter;
        for (limit, charged) in [(4095, 1), (4096, 4096), (4097, 4096)] {
            let context = JsonOperationContext::new(Arc::new(WorkLedger::new(limit)));
            let mut work = context.aggregate_reservation();
            work.grow(1, kind).unwrap();
            assert_eq!((work.reserved_bytes(), work.covered_bytes()), (charged, 1));
            work.finish_aggregate();
            assert_eq!(work.reserved_bytes(), 1);
            drop(work);
            assert_eq!(context.available(), limit);
        }
        let ledger = Arc::new(WorkLedger::new(7000));
        let context = JsonOperationContext::new(Arc::clone(&ledger));
        let mut work = context.aggregate_reservation();
        work.grow(1, kind).unwrap();
        work.grow(4089, kind).unwrap();
        assert_eq!(work.reserved_bytes(), 4096);
        work.release(100);
        assert_eq!((work.reserved_bytes(), work.covered_bytes()), (4096, 3990));
        work.grow(100, kind).unwrap();
        work.grow(16, kind).unwrap();
        assert_eq!((work.reserved_bytes(), work.covered_bytes()), (4106, 4106));
        work.grow(7000 - 4106, kind).unwrap();
        let error = work.grow(1, kind).unwrap_err();
        assert!(error
            .detail()
            .iter()
            .any(|(key, value)| key == "actual" && value == "7001"));
        assert_eq!((work.reserved_bytes(), work.covered_bytes()), (7000, 7000));
        drop(work);
        assert_eq!(ledger.state.lock().unwrap().used, 0);
    }

    #[test]
    fn aggregate_transfers_and_merges_preserve_live_coverage_and_spare_credit() {
        let ledger = Arc::new(WorkLedger::with_limits(12288, 32, 12288));
        let context = JsonOperationContext::new(Arc::clone(&ledger));
        let kind = JsonInputKind::StoredJson;
        let mut work = context.aggregate_reservation();
        work.grow(37, kind).unwrap();
        let mut workspace = work.take_workspace(13);
        assert_eq!((work.reserved_bytes(), work.covered_bytes()), (4083, 24));
        work.merge(workspace.take_json(13, kind).unwrap());
        assert_eq!((work.reserved_bytes(), work.covered_bytes()), (4096, 37));
        let mut catalog = context.catalog_reservation();
        assert!(catalog.absorb(&mut work, None, 37, 0).is_err());
        assert_eq!((work.reserved_bytes(), work.covered_bytes()), (4096, 37));
        catalog.absorb(&mut work, None, 24, 0).unwrap();
        assert_eq!((work.reserved_bytes(), work.covered_bytes()), (4072, 13));
        let mut other = context.aggregate_reservation();
        other.grow(17, kind).unwrap();
        work.merge(other);
        assert_eq!((work.reserved_bytes(), work.covered_bytes()), (8168, 30));
        work.grow(19, kind).unwrap();
        let live = work.take_workspace(49);
        assert_eq!((work.reserved_bytes(), work.covered_bytes()), (8119, 0));
        drop(work);
        let state = ledger.state.lock().unwrap();
        assert_eq!(
            (state.used, state.workspace_used, state.catalog_used),
            (0, 49, 24)
        );
        drop(state);
        drop(live);
        drop(workspace);
        drop(catalog);
        assert_eq!(context.available(), 12288);
    }

    #[test]
    fn aggregate_native_failure_keeps_previous_data_and_retries_with_local_credit() {
        let context = JsonOperationContext::new(Arc::new(WorkLedger::new(8192)));
        let kind = JsonInputKind::StoredJson;
        let mut work = context.aggregate_reservation();
        let previous = work.copy_string("previous payload", kind).unwrap();
        let input = "x".repeat(64);
        let covered = work.covered_bytes();
        let pointer = previous.as_ptr();
        let error =
            crate::test_alloc::fail_byte_allocation_once(64, || work.copy_string(&input, kind))
                .unwrap_err();
        assert_eq!(error.code(), ErrorCode::IoError);
        assert_eq!(previous, "previous payload");
        assert_eq!(previous.as_ptr(), pointer);
        assert_eq!(work.covered_bytes(), covered);
        assert!(work.reserved_bytes() >= covered);
        let value = work.copy_string(&input, kind).unwrap();
        work.finish_aggregate();
        assert_eq!(
            work.reserved_bytes(),
            previous.capacity() + value.capacity()
        );
        drop(value);
        drop(previous);
        drop(work);
        assert_eq!(context.available(), 8192);
    }

    #[test]
    fn object_keys_share_one_reservation_and_preserve_the_existing_owner() {
        let keys = ["alpha", "beta beta", "\u{4e2d}"];
        let bytes = object_workspace(keys.len()) + keys.iter().map(|key| key.len()).sum::<usize>();
        let seed = 13;
        for parent in [false, true] {
            for available in [bytes - 1, bytes, bytes + 1] {
                let limit = seed + available;
                let ledger = Arc::new(WorkLedger::with_limits(
                    if parent { 2 * limit } else { limit },
                    limit,
                    if parent { limit } else { 2 * limit },
                ));
                let context = JsonOperationContext::new(ledger);
                let mut existing = context.reservation();
                existing.grow(seed, JsonInputKind::Filter).unwrap();
                let result = existing.reserve_object_keys(keys.into_iter(), JsonInputKind::Filter);
                if available < bytes {
                    let error = result.err().unwrap();
                    assert_eq!(error.code(), ErrorCode::InvalidMetadataFilter);
                    assert!(error.detail().iter().any(|(key, value)| {
                        key == "actual" && value == &(seed + bytes).to_string()
                    }));
                    assert_eq!(context.available(), available);
                } else {
                    let keys_work = result.unwrap();
                    assert_eq!(keys_work.reserved_bytes(), bytes);
                    assert_eq!(context.available(), available - bytes);
                    existing.merge(keys_work);
                    assert_eq!(existing.reserved_bytes(), seed + bytes);
                }
                drop(existing);
                assert_eq!(context.available(), limit);
            }
        }
    }

    #[test]
    fn heap_shrink_admits_overlap_and_moves_payloads_without_dropping_them() {
        use std::cell::Cell;
        use std::collections::BTreeMap;
        struct Owned<'a> {
            key: String,
            drops: &'a Cell<usize>,
        }
        impl PartialEq for Owned<'_> {
            fn eq(&self, other: &Self) -> bool {
                self.key == other.key
            }
        }
        impl Eq for Owned<'_> {}
        impl PartialOrd for Owned<'_> {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }
        impl Ord for Owned<'_> {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                self.key.cmp(&other.key)
            }
        }
        impl Drop for Owned<'_> {
            fn drop(&mut self) {
                self.drops.set(self.drops.get() + 1);
            }
        }
        let size = std::mem::size_of::<Owned<'_>>();
        for seed in 0..100usize {
            let count = 1 + seed % 13;
            let capacity = count + 1 + seed;
            let active = 13 + seed;
            for parent in [false, true] {
                for delta in [-1isize, 0, 1] {
                    let drops = Cell::new(0);
                    let mut heap = BinaryHeap::new();
                    heap.try_reserve_exact(capacity).unwrap();
                    for index in 0..count {
                        heap.push(Owned {
                            key: format!("key-{seed}-{index:03}"),
                            drops: &drops,
                        });
                    }
                    let payload = heap.iter().map(|item| item.key.capacity()).sum::<usize>();
                    let before = capacity * size + payload;
                    let needed = active + before + count * size;
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
                    let mut work = context.workspace_reservation();
                    work.grow_to(before, before, "operation_scratch").unwrap();
                    assert_eq!(heap.capacity(), capacity);
                    let array = heap.as_slice().as_ptr();
                    let snapshot = || {
                        heap.iter()
                            .map(|item| (item.key.clone(), item.key.as_ptr() as usize))
                            .collect::<BTreeMap<_, _>>()
                    };
                    let previous = snapshot();
                    let (changed, allocations) =
                        crate::test_alloc::allocations(|| try_shrink_heap(&mut heap, &mut work));
                    assert_eq!(changed.unwrap(), delta >= 0);
                    assert_eq!(drops.get(), 0);
                    let actual = heap
                        .iter()
                        .map(|item| (item.key.clone(), item.key.as_ptr() as usize))
                        .collect::<BTreeMap<_, _>>();
                    assert_eq!(actual, previous);
                    let resident = if delta < 0 {
                        assert_eq!(heap.as_slice().as_ptr(), array);
                        assert_eq!(heap.capacity(), capacity);
                        before
                    } else {
                        assert_eq!(allocations, 1);
                        assert_eq!(heap.capacity(), count);
                        count * size + payload
                    };
                    assert_eq!(work.reserved_bytes(), resident);
                    assert_eq!(
                        context.available_workspace_bytes(),
                        limit - active - resident
                    );
                    let sorted = heap.into_sorted_vec();
                    assert_eq!(drops.get(), 0);
                    assert_eq!(
                        sorted
                            .iter()
                            .map(|item| item.key.as_str())
                            .collect::<Vec<_>>(),
                        previous.keys().map(String::as_str).collect::<Vec<_>>()
                    );
                    drop(sorted);
                    assert_eq!(drops.get(), count);
                    drop(work);
                    drop(outside);
                    assert_eq!(context.available_workspace_bytes(), limit);
                }
            }
        }
    }

    #[test]
    fn heap_shrink_allocation_failure_keeps_old_array_and_allows_retry() {
        let size = std::mem::size_of::<[u8; 17]>();
        for count in 1..=100usize {
            let capacity = count + 7;
            let payload = 13;
            let before = payload + capacity * size;
            let needed = before + count * size;
            let context =
                JsonOperationContext::new(Arc::new(WorkLedger::with_limits(needed, 0, needed)));
            let mut work = context.workspace_reservation();
            work.grow_to(before, before, "operation_scratch").unwrap();
            let prefix = vec![12u8; payload];
            let mut heap = BinaryHeap::new();
            heap.try_reserve_exact(capacity).unwrap();
            for value in 0..count {
                heap.push([value as u8; 17]);
            }
            let array = heap.as_slice().as_ptr();
            let previous = heap.as_slice().to_vec();
            let error = crate::test_alloc::fail_byte_allocation_once(count * size, || {
                try_shrink_heap(&mut heap, &mut work)
            })
            .unwrap_err();
            assert_eq!(error.code(), crate::ErrorCode::IoError);
            assert_eq!(heap.as_slice().as_ptr(), array);
            assert_eq!(heap.as_slice(), previous);
            assert_eq!(heap.capacity(), capacity);
            assert_eq!(work.reserved_bytes(), before);
            let (changed, allocations) = crate::test_alloc::key_allocations(count * size, || {
                try_shrink_heap(&mut heap, &mut work)
            });
            assert!(changed.unwrap());
            assert_eq!(allocations.allocated, 1);
            assert!(!allocations.overflow);
            assert_eq!(heap.capacity(), count);
            assert_eq!(work.reserved_bytes(), payload + count * size);
            assert_eq!(
                heap.into_sorted_vec(),
                previous
                    .into_iter()
                    .collect::<BinaryHeap<_>>()
                    .into_sorted_vec()
            );
            drop(prefix);
            drop(work);
            assert_eq!(context.available_workspace_bytes(), needed);
        }
    }

    #[test]
    fn heap_shrink_empty_full_and_zero_sized_paths_do_not_allocate() {
        let size = std::mem::size_of::<[u8; 17]>();
        for count in 0..100usize {
            let payload = 13;
            let before = payload + count * size;
            let context =
                JsonOperationContext::new(Arc::new(WorkLedger::with_limits(before, 0, before)));
            let mut work = context.workspace_reservation();
            work.grow_to(before, before, "operation_scratch").unwrap();
            let prefix = vec![12u8; payload];
            let mut heap = BinaryHeap::new();
            heap.try_reserve_exact(count).unwrap();
            for value in 0..count {
                heap.push([value as u8; 17]);
            }
            let pointer = heap.as_slice().as_ptr();
            let (changed, allocated) =
                crate::test_alloc::allocations(|| try_shrink_heap(&mut heap, &mut work));
            assert!(!changed.unwrap());
            assert_eq!(allocated, 0);
            assert_eq!(heap.as_slice().as_ptr(), pointer);
            assert_eq!(work.reserved_bytes(), before);
            heap.clear();
            let (changed, allocated) =
                crate::test_alloc::allocations(|| try_shrink_heap(&mut heap, &mut work));
            assert_eq!(changed.unwrap(), count != 0);
            assert_eq!(allocated, 0);
            assert_eq!(heap.capacity(), 0);
            assert_eq!(work.reserved_bytes(), payload);
            drop(heap);
            drop(prefix);
            drop(work);
            assert_eq!(context.available_workspace_bytes(), before);
        }
        let context = JsonOperationContext::new(Arc::new(WorkLedger::with_limits(0, 0, 0)));
        let mut work = context.workspace_reservation();
        let mut heap = BinaryHeap::new();
        heap.push(());
        let (changed, allocated) =
            crate::test_alloc::allocations(|| try_shrink_heap(&mut heap, &mut work));
        assert!(!changed.unwrap());
        assert_eq!(allocated, 0);
        assert_eq!(heap.len(), 1);
        assert_eq!(work.reserved_bytes(), 0);
    }

    #[test]
    fn heap_growth_admits_old_and_new_arrays_before_reallocation() {
        let size = std::mem::size_of::<[u8; 17]>();
        for seed in 0..100usize {
            let old = 1 + seed;
            let payload = 13 + seed;
            let active = 7 + seed * 3;
            let needed = active + payload + (old + old + 1) * size;
            for parent in [false, true] {
                for delta in [-1isize, 0, 1] {
                    let limit = (needed as isize + delta) as usize;
                    let ledger = Arc::new(WorkLedger::with_limits(
                        if parent { needed * 2 } else { limit },
                        0,
                        if parent { limit } else { needed * 2 },
                    ));
                    let context = JsonOperationContext::new(ledger.clone());
                    let competing = JsonOperationContext::new(ledger);
                    let mut outside = competing.workspace_reservation();
                    outside
                        .grow_to(active, active, "operation_scratch")
                        .unwrap();
                    let mut work = context.workspace_reservation();
                    work.grow_to(
                        payload + old * size,
                        payload + old * size,
                        "operation_scratch",
                    )
                    .unwrap();
                    let prefix = vec![11u8; payload];
                    let mut heap = BinaryHeap::new();
                    heap.try_reserve_exact(old).unwrap();
                    for value in 0..old {
                        heap.push([value as u8; 17]);
                    }
                    assert_eq!(heap.capacity(), old);
                    let pointer = heap.as_slice().as_ptr();
                    let values = heap.as_slice().to_vec();
                    let (result, allocations) =
                        crate::test_alloc::key_allocations((old + 1) * size, || {
                            reserve_heap_slot(&mut heap, &mut work, usize::MAX)
                        });
                    if delta < 0 {
                        assert_memory_error(
                            &result.unwrap_err(),
                            "operation_scratch",
                            limit,
                            needed as u128,
                        );
                        assert_eq!(allocations.allocated, 0);
                        assert_eq!(heap.capacity(), old);
                        assert_eq!(heap.as_slice().as_ptr(), pointer);
                        assert_eq!(work.reserved_bytes(), payload + old * size);
                    } else {
                        result.unwrap();
                        assert_eq!(allocations.allocated, 1);
                        assert_eq!(heap.capacity(), old + 1);
                        assert_eq!(work.reserved_bytes(), payload + (old + 1) * size);
                    }
                    assert!(!allocations.overflow);
                    assert_eq!(heap.as_slice(), values);
                    if delta >= 0 {
                        heap.push([240; 17]);
                        assert_eq!(heap.peek(), Some(&[240; 17]));
                    }
                    assert_eq!(
                        context.available_workspace_bytes(),
                        limit - active - work.reserved_bytes()
                    );
                    drop(heap);
                    drop(prefix);
                    drop(work);
                    drop(outside);
                    assert_eq!(context.available_workspace_bytes(), limit);
                    let mut all = context.workspace_reservation();
                    all.grow_to(limit, limit, "operation_scratch").unwrap();
                    assert_memory_error(
                        &competing
                            .workspace_reservation()
                            .grow_to(1, 1, "operation_scratch")
                            .unwrap_err(),
                        "operation_scratch",
                        limit,
                        limit as u128 + 1,
                    );
                }
            }
        }
    }

    #[test]
    fn heap_growth_preference_and_spare_slots_keep_only_resident_credit() {
        let size = std::mem::size_of::<[u8; 17]>();
        for old in [0usize, 1, 4, 17, 99] {
            let payload = 23;
            let preferred = (old * 2).max(4);
            for new in [old + 1, (old + 1 + preferred) / 2, preferred] {
                let needed = payload + (old + new) * size;
                for remainder in [0, size - 1] {
                    let limit = needed + remainder;
                    let context = JsonOperationContext::new(Arc::new(WorkLedger::with_limits(
                        limit, 0, limit,
                    )));
                    let mut work = context.workspace_reservation();
                    work.grow_to(
                        payload + old * size,
                        payload + old * size,
                        "operation_scratch",
                    )
                    .unwrap();
                    let prefix = vec![12u8; payload];
                    let mut heap = BinaryHeap::new();
                    heap.try_reserve_exact(old).unwrap();
                    for value in 0..old {
                        heap.push([value as u8; 17]);
                    }
                    reserve_heap_slot(&mut heap, &mut work, usize::MAX).unwrap();
                    assert_eq!(heap.capacity(), new);
                    assert_eq!(work.reserved_bytes(), payload + new * size);
                    let pointer = heap.as_slice().as_ptr();
                    let mut remaining = context.workspace_reservation();
                    let free = limit - work.reserved_bytes();
                    remaining.grow_to(free, free, "operation_scratch").unwrap();
                    let (result, allocated) = crate::test_alloc::allocations(|| {
                        reserve_heap_slot(&mut heap, &mut work, usize::MAX)
                    });
                    result.unwrap();
                    assert_eq!(allocated, 0);
                    assert_eq!(heap.as_slice().as_ptr(), pointer);
                    assert_eq!(work.reserved_bytes(), payload + new * size);
                    heap.push([250; 17]);
                    assert_eq!(heap.peek(), Some(&[250; 17]));
                    drop(heap);
                    drop(prefix);
                    drop(work);
                    drop(remaining);
                    assert_eq!(context.available_workspace_bytes(), limit);
                }
            }
        }
        let context = JsonOperationContext::new(Arc::new(WorkLedger::with_limits(0, 0, 0)));
        let mut work = context.workspace_reservation();
        let mut heap = BinaryHeap::<()>::new();
        let (result, allocated) =
            crate::test_alloc::allocations(|| reserve_heap_slot(&mut heap, &mut work, usize::MAX));
        result.unwrap();
        assert_eq!(allocated, 0);
        heap.push(());
        assert_eq!(heap.len(), 1);
        assert_eq!(work.reserved_bytes(), 0);
    }

    #[test]
    fn heap_reallocation_failure_preserves_old_owner_and_allows_retry() {
        let size = std::mem::size_of::<[u8; 17]>();
        for old in 1..=100usize {
            let payload = 13;
            let needed = payload + (old + old + 1) * size;
            let context =
                JsonOperationContext::new(Arc::new(WorkLedger::with_limits(needed, 0, needed)));
            let mut work = context.workspace_reservation();
            work.grow_to(
                payload + old * size,
                payload + old * size,
                "operation_scratch",
            )
            .unwrap();
            let prefix = vec![11u8; payload];
            let mut heap = BinaryHeap::new();
            heap.try_reserve_exact(old).unwrap();
            for value in 0..old {
                heap.push([value as u8; 17]);
            }
            let pointer = heap.as_slice().as_ptr();
            let values = heap.as_slice().to_vec();
            let error = crate::test_alloc::fail_byte_allocation_once((old + 1) * size, || {
                reserve_heap_slot(&mut heap, &mut work, usize::MAX)
            })
            .unwrap_err();
            assert_eq!(error.code(), crate::ErrorCode::IoError);
            assert_eq!(heap.capacity(), old);
            assert_eq!(heap.as_slice().as_ptr(), pointer);
            assert_eq!(heap.as_slice(), values);
            assert_eq!(work.reserved_bytes(), payload + old * size);
            reserve_heap_slot(&mut heap, &mut work, usize::MAX).unwrap();
            heap.push([250; 17]);
            assert_eq!(heap.peek(), Some(&[250; 17]));
            assert_eq!(heap.capacity(), old + 1);
            assert_eq!(work.reserved_bytes(), payload + (old + 1) * size);
            drop(heap);
            drop(prefix);
            drop(work);
            assert_eq!(context.available_workspace_bytes(), needed);
        }
    }

    #[test]
    fn workspace_cache_transfer_keeps_shared_credit_and_is_atomic_on_refusal() {
        for seed in 1..=100usize {
            let bytes = seed * 17;
            let ledger = Arc::new(WorkLedger::with_limits(bytes, bytes, bytes));
            let context = JsonOperationContext::new(ledger.clone());
            let competing = JsonOperationContext::new(ledger.clone());
            let mut work = context.workspace_reservation();
            work.grow_to(bytes, bytes, "operation_scratch").unwrap();
            let mut cache = context.cache_reservation(0, bytes).unwrap();
            assert!(!cache.absorb_workspace(&mut work, bytes, bytes - 1));
            assert_eq!(work.reserved_bytes(), bytes);
            assert_eq!(cache.reserved_bytes(), 0);
            let partial = bytes / 3;
            assert!(cache.absorb_workspace(&mut work, partial, bytes));
            assert_eq!(work.reserved_bytes(), bytes - partial);
            assert!(competing.cache_reservation(1, bytes + 1).is_none());
            assert!(competing
                .workspace_reservation()
                .grow_to(1, 1, "operation_scratch")
                .is_err());
            assert!(cache.absorb_workspace(&mut work, bytes - partial, bytes));
            assert_eq!(work.reserved_bytes(), 0);
            assert_eq!(cache.reserved_bytes(), bytes);
            drop(work);
            assert!(competing.cache_reservation(1, bytes + 1).is_none());
            drop(cache);
            assert!(competing.cache_reservation(bytes as u128, bytes).is_some());
            let state = ledger.state.lock().unwrap();
            assert_eq!(
                (state.used, state.workspace_used, state.cache_used),
                (0, 0, 0)
            );
        }
    }

    #[test]
    fn workspace_cache_transfer_rejects_a_foreign_or_reattached_ledger() {
        for reattach in [false, true] {
            let first = Arc::new(WorkLedger::with_limits(1024, 1024, 1024));
            let second = Arc::new(WorkLedger::with_limits(1024, 1024, 1024));
            let cache_context = JsonOperationContext::new(first.clone());
            let source = JsonOperationContext::new(if reattach {
                first.clone()
            } else {
                second.clone()
            });
            let mut cache = cache_context.cache_reservation(0, 1024).unwrap();
            let mut work = source.workspace_reservation();
            work.grow_to(512, 512, "operation_scratch").unwrap();
            if reattach {
                source
                    .attach(second.clone(), JsonInputKind::StoredJson)
                    .unwrap();
            }
            assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                cache.absorb_workspace(&mut work, 512, 1024)
            }))
            .is_err());
            assert_eq!(cache.reserved_bytes(), 0);
            assert_eq!(work.reserved_bytes(), 512);
            assert_eq!(second.state.lock().unwrap().workspace_used, 512);
            drop(work);
            assert_eq!(first.state.lock().unwrap().workspace_used, 0);
            assert_eq!(second.state.lock().unwrap().workspace_used, 0);
        }
    }

    #[test]
    fn concurrent_workspace_cache_transfers_never_double_charge_or_release_credit() {
        let limit = 4096;
        let ledger = Arc::new(WorkLedger::with_limits(limit, limit, limit));
        std::thread::scope(|scope| {
            for _ in 0..8 {
                let ledger = ledger.clone();
                scope.spawn(move || {
                    let context = JsonOperationContext::new(ledger.clone());
                    for _ in 0..1000 {
                        let mut work = context.workspace_reservation();
                        work.grow_to(512, 512, "operation_scratch").unwrap();
                        let mut cache = context.cache_reservation(0, limit).unwrap();
                        assert!(cache.absorb_workspace(&mut work, 512, limit));
                        let state = ledger.state.lock().unwrap();
                        assert!(state.workspace_used + state.cache_used <= limit);
                    }
                });
            }
        });
        let state = ledger.state.lock().unwrap();
        assert_eq!((state.workspace_used, state.cache_used), (0, 0));
    }

    #[test]
    fn cache_ownership_shares_parent_without_consuming_work_or_catalog_quota() {
        for seed in 0..100usize {
            let ledger = Arc::new(WorkLedger::with_limits(200, 600, 1000 + seed));
            let context = JsonOperationContext::new(Arc::clone(&ledger));
            let mut catalog = context.catalog_reservation();
            catalog.grow(500).unwrap();
            let mut work = context.workspace_reservation();
            work.grow_to(100, 100, "operation_scratch").unwrap();
            assert!(context.cache_reservation(u128::MAX, usize::MAX).is_none());
            assert!(context
                .cache_reservation(401 + seed as u128, 1000)
                .is_none());
            assert!(context
                .cache_reservation(400 + seed as u128, 399 + seed)
                .is_none());
            let mut cache = context
                .cache_reservation(400 + seed as u128, 400 + seed)
                .unwrap();
            assert_eq!(cache.reserved_bytes(), 400 + seed);
            assert_eq!(ledger.state.lock().unwrap().working_available(), 0);
            assert_memory_error(
                &work.grow_to(101, 101, "operation_scratch").unwrap_err(),
                "operation_scratch",
                1000 + seed,
                1001 + seed as u128,
            );
            assert_memory_error(
                &catalog.grow(1).unwrap_err(),
                "catalog_build",
                1000 + seed,
                1001 + seed as u128,
            );
            let json_error = context
                .reservation()
                .grow(1, JsonInputKind::StoredJson)
                .unwrap_err();
            assert_eq!(json_error.code(), crate::ErrorCode::RecoveryError);
            cache.retain(300 + seed);
            work.grow_to(200, 200, "operation_scratch").unwrap();
            assert!(context.cache_reservation(1, usize::MAX).is_none());
            drop(work);
            let other = JsonOperationContext::new(Arc::clone(&ledger));
            let second = other.cache_reservation(200, 500 + seed).unwrap();
            assert!(context.cache_reservation(1, 500 + seed).is_none());
            let target = Arc::new(WorkLedger::with_limits(200, 600, 1000));
            context
                .attach(Arc::clone(&target), JsonInputKind::StoredJson)
                .unwrap();
            assert_eq!(ledger.state.lock().unwrap().cache_used, 500 + seed);
            assert_eq!(target.state.lock().unwrap().cache_used, 0);
            drop(context);
            drop(cache);
            assert_eq!(ledger.state.lock().unwrap().cache_used, 200);
            drop(second);
            drop(catalog);
            let state = ledger.state.lock().unwrap();
            assert_eq!(
                (
                    state.used,
                    state.workspace_used,
                    state.catalog_used,
                    state.cache_used
                ),
                (0, 0, 0, 0)
            );
        }
    }

    #[test]
    fn simultaneous_cache_admission_cannot_exceed_the_shared_limit() {
        let ledger = Arc::new(WorkLedger::with_limits(1024, 1024, 1024));
        std::thread::scope(|scope| {
            for _ in 0..8 {
                let ledger = Arc::clone(&ledger);
                scope.spawn(move || {
                    let context = JsonOperationContext::new(Arc::clone(&ledger));
                    for _ in 0..1000 {
                        let cache = context.cache_reservation(256, 768);
                        let mut work = context.workspace_reservation();
                        let _ = work.grow_to(256, 256, "operation_scratch");
                        {
                            let state = ledger.state.lock().unwrap();
                            assert!(state.cache_used <= 768);
                            assert!(state.cache_used + state.workspace_used <= 1024);
                        }
                        drop(cache);
                    }
                });
            }
        });
        let state = ledger.state.lock().unwrap();
        assert_eq!((state.cache_used, state.workspace_used), (0, 0));
    }

    #[test]
    fn catalog_quota_and_shared_memory_rejections_identify_the_actual_ledger() {
        for seed in 0..100 {
            let context = JsonOperationContext::new(Arc::new(WorkLedger::with_limits(
                1024,
                600 + seed,
                800 + seed,
            )));
            let mut catalog = context.catalog_reservation();
            catalog.grow(600 + seed).unwrap();
            let error = catalog.grow(1).unwrap_err();
            let detail = error.to_json();
            let detail = detail.get("detail").unwrap();
            assert_eq!(error.code(), crate::ErrorCode::ResourceExhausted);
            assert_eq!(detail.get("resource").unwrap().as_str(), Some("catalog"));
            assert_eq!(
                detail.get("operation").unwrap().as_str(),
                Some("catalog_build")
            );
            assert_eq!(
                detail.get("limit_bytes").unwrap().as_str(),
                Some((600 + seed).to_string().as_str())
            );
            assert_eq!(
                detail.get("requested_bytes").unwrap().as_str(),
                Some((601 + seed).to_string().as_str())
            );
            let mut workspace = context.workspace_reservation();
            assert_eq!(
                workspace.grow_to(200, 1024, "operation_scratch").unwrap(),
                200
            );
            assert_memory_error(
                &workspace
                    .grow_to(201, 1024, "operation_scratch")
                    .unwrap_err(),
                "operation_scratch",
                800 + seed,
                801 + seed as u128,
            );
            catalog.retain(500 + seed);
            workspace.grow_to(300, 300, "operation_scratch").unwrap();
            assert_memory_error(
                &catalog.grow(1).unwrap_err(),
                "catalog_build",
                800 + seed,
                801 + seed as u128,
            );
            let error = context
                .reservation()
                .grow(1, JsonInputKind::StoredJson)
                .unwrap_err();
            assert_eq!(error.code(), crate::ErrorCode::RecoveryError);
            assert!(error
                .detail()
                .iter()
                .any(|(key, value)| key == "limit" && value == "0"));
            drop(workspace);
            drop(catalog);
            context
                .workspace_reservation()
                .grow_to(800 + seed, 800 + seed, "operation_scratch")
                .unwrap();
        }
    }

    #[test]
    fn catalog_transfer_is_atomic_preserves_memory_and_survives_context_reattachment() {
        for seed in 0..100 {
            let ledger = Arc::new(WorkLedger::with_limits(1024, 900, 800));
            let context = JsonOperationContext::new(Arc::clone(&ledger));
            let mut catalog = context.catalog_reservation();
            catalog.grow(400).unwrap();
            let mut json = context.reservation();
            json.grow(200 + seed, JsonInputKind::StoredJson).unwrap();
            let mut workspace = context.workspace_reservation();
            workspace
                .grow_to(200 - seed, 200 - seed, "operation_scratch")
                .unwrap();
            catalog
                .absorb(&mut json, Some(&mut workspace), 100 + seed, 100 - seed)
                .unwrap();
            assert_eq!(catalog.reserved_bytes(), 600);
            assert_eq!(json.reserved_bytes(), 100);
            assert_eq!(workspace.reserved_bytes(), 100);
            {
                let state = ledger.state.lock().unwrap();
                assert_eq!(
                    (state.used, state.workspace_used, state.catalog_used),
                    (100, 100, 600)
                );
            }
            assert_memory_error(&catalog.grow(1).unwrap_err(), "catalog_build", 800, 801);
            let target = Arc::new(WorkLedger::with_limits(1024, 900, 800));
            context
                .attach(Arc::clone(&target), JsonInputKind::StoredJson)
                .unwrap();
            {
                let source = ledger.state.lock().unwrap();
                let target = target.state.lock().unwrap();
                assert_eq!(
                    (source.used, source.workspace_used, source.catalog_used),
                    (0, 0, 600)
                );
                assert_eq!(
                    (target.used, target.workspace_used, target.catalog_used),
                    (100, 100, 0)
                );
            }
            drop(json);
            drop(workspace);
            drop(catalog);
            assert_eq!(ledger.state.lock().unwrap().catalog_used, 0);
            assert_eq!(target.state.lock().unwrap().working_available(), 800);
        }
    }

    #[test]
    fn failed_catalog_transfer_keeps_source_credit_and_shared_json_error_priority() {
        for seed in 0..100 {
            let ledger = Arc::new(WorkLedger::with_limits(1024, 200 + seed, 300 + seed));
            let context = JsonOperationContext::new(Arc::clone(&ledger));
            let mut catalog = context.catalog_reservation();
            catalog.grow(150 + seed).unwrap();
            let mut json = context.reservation();
            json.grow(120, JsonInputKind::StoredJson).unwrap();
            let mut workspace = context.workspace_reservation();
            workspace.grow_to(30, 30, "operation_scratch").unwrap();
            let error = catalog
                .absorb(&mut json, Some(&mut workspace), 120, 30)
                .unwrap_err();
            assert_eq!(error.code(), crate::ErrorCode::ResourceExhausted);
            assert!(error
                .detail()
                .iter()
                .any(|(key, value)| key == "resource" && value == "catalog"));
            assert_eq!(catalog.reserved_bytes(), 150 + seed);
            assert_eq!(json.reserved_bytes(), 120);
            assert_eq!(workspace.reserved_bytes(), 30);
            drop(workspace);
            let error = json.grow(31, JsonInputKind::StoredJson).unwrap_err();
            assert!(error
                .detail()
                .iter()
                .any(|(key, value)| key == "limit" && value == "150"));
            let target = Arc::new(WorkLedger::with_limits(1024, 200, 200));
            let target_context = JsonOperationContext::new(Arc::clone(&target));
            let mut occupied = target_context.catalog_reservation();
            occupied.grow(190).unwrap();
            let error = context
                .attach(Arc::clone(&target), JsonInputKind::StoredJson)
                .unwrap_err();
            assert_eq!(error.code(), crate::ErrorCode::RecoveryError);
            assert!(error
                .detail()
                .iter()
                .any(|(key, value)| key == "limit" && value == "10"));
            assert_eq!(ledger.state.lock().unwrap().used, 120);
            assert_eq!(target.state.lock().unwrap().used, 0);
        }
    }

    fn assert_memory_error(error: &Error, phase: &str, limit: usize, requested: u128) {
        assert_eq!(error.code(), crate::ErrorCode::ResourceExhausted);
        let json = error.to_json();
        let detail = json.get("detail").unwrap();
        assert_eq!(detail.as_object().unwrap().len(), 4);
        for (key, expected) in [
            ("resource", "memory".to_owned()),
            ("operation", phase.to_owned()),
            ("limit_bytes", limit.to_string()),
            ("requested_bytes", requested.to_string()),
        ] {
            assert_eq!(detail.get(key).unwrap().as_str(), Some(expected.as_str()));
        }
    }

    #[test]
    fn reserved_strings_charge_utf8_bytes_before_copying_and_release_on_drop() {
        let limit = 4096;
        for seed in 0..100usize {
            let context = JsonOperationContext::new(Arc::new(WorkLedger::new(limit)));
            let value = "x\u{4e2d}\n".repeat(seed);
            let mut other = context.workspace_reservation();
            other
                .grow_to(
                    limit - value.len(),
                    limit - value.len(),
                    "operation_scratch",
                )
                .unwrap();
            let owned = ReservedString::copy(&value, &context).unwrap();
            assert_eq!(owned.as_ref(), value);
            assert!(other
                .grow_to(
                    limit - value.len() + 1,
                    limit - value.len() + 1,
                    "operation_scratch"
                )
                .is_err());
            drop(owned);
            other.grow_to(limit, limit, "operation_scratch").unwrap();
            if !value.is_empty() {
                other.retain(limit - value.len() + 1);
                let error = ReservedString::copy(&value, &context).err().unwrap();
                assert_memory_error(&error, "operation_scratch", limit, limit as u128 + 1);
                other.grow_to(limit, limit, "operation_scratch").unwrap();
            }
        }
    }

    #[test]
    fn generated_strings_admit_exact_bytes_and_bound_the_rendering_pass() {
        use std::cell::Cell;
        for seed in 0..100usize {
            let limit = 4096;
            let context = JsonOperationContext::new(Arc::new(WorkLedger::new(limit)));
            let text = "x\u{4e2d}".repeat(seed);
            let expected = format!("{text}:{seed:016}");
            let calls = Cell::new(0);
            let render = |out: &mut dyn std::fmt::Write| {
                calls.set(calls.get() + 1);
                write!(out, "{text}:{seed:016}")
            };
            let mut held = context.workspace_reservation();
            let occupied = limit - expected.len();
            held.grow_to(occupied, occupied, "operation_scratch")
                .unwrap();
            let result = ReservedString::build(&context, "batch_prepare", render).unwrap();
            assert_eq!(result.as_ref(), expected);
            assert_eq!(result.value.capacity(), expected.len());
            assert_eq!(calls.get(), 2);
            drop(result);
            calls.set(0);
            held.grow_to(occupied + 1, occupied + 1, "operation_scratch")
                .unwrap();
            let error = ReservedString::build(&context, "batch_prepare", render)
                .err()
                .unwrap();
            assert_memory_error(&error, "batch_prepare", limit, limit as u128 + 1);
            assert_eq!(
                calls.get(),
                1,
                "admission must precede the heap rendering pass"
            );
            held.retain(0);
            for changed in ["", "longer"] {
                let calls = Cell::new(0);
                let error = ReservedString::build(&context, "batch_prepare", |out| {
                    calls.set(calls.get() + 1);
                    out.write_str(if calls.get() == 1 { "x" } else { changed })
                })
                .err()
                .unwrap();
                assert_eq!(error.code(), crate::ErrorCode::IoError);
                assert_eq!(calls.get(), 2);
                held.grow_to(limit, limit, "operation_scratch").unwrap();
                held.retain(0);
            }
            for fail_at in [1, 2] {
                let calls = Cell::new(0);
                let error = ReservedString::build(&context, "batch_prepare", |out| {
                    calls.set(calls.get() + 1);
                    out.write_str("x")?;
                    if calls.get() == fail_at {
                        Err(std::fmt::Error)
                    } else {
                        Ok(())
                    }
                })
                .err()
                .unwrap();
                assert_eq!(error.code(), crate::ErrorCode::IoError);
                held.grow_to(limit, limit, "operation_scratch").unwrap();
                held.retain(0);
            }
            assert!(ReservedString::build(&context, "batch_prepare", |_| Ok(()))
                .unwrap()
                .as_ref()
                .is_empty());
        }
    }

    #[test]
    fn typed_copy_transfer_preserves_shared_usage_without_reacquiring_credit() {
        for seed in 0..100usize {
            let limit = 2048 + seed;
            let ledger = Arc::new(WorkLedger::new(limit));
            let context = JsonOperationContext::new(ledger.clone());
            let mut json = context.reservation();
            json.grow(limit, JsonInputKind::StoredJson).unwrap();
            let moved = 1 + seed * 7;
            let work = json.take_workspace(moved);
            assert_eq!(json.reserved_bytes(), limit - moved);
            assert_eq!(work.reserved_bytes(), moved);
            {
                let state = ledger.state.lock().unwrap();
                assert_eq!((state.used, state.workspace_used), (limit - moved, moved));
            }
            let competing = JsonOperationContext::new(ledger.clone());
            assert_memory_error(
                &competing
                    .workspace_reservation()
                    .grow_to(1, 1, "operation_scratch")
                    .unwrap_err(),
                "operation_scratch",
                limit,
                limit as u128 + 1,
            );
            assert!(competing
                .reservation()
                .grow(1, JsonInputKind::StoredJson)
                .is_err());
            let destination = Arc::new(WorkLedger::new(limit));
            context
                .attach(destination.clone(), JsonInputKind::StoredJson)
                .unwrap();
            {
                let state = ledger.state.lock().unwrap();
                assert_eq!((state.used, state.workspace_used), (0, 0));
            }
            drop(json);
            let mut other = context.reservation();
            other
                .grow(limit - moved, JsonInputKind::StoredJson)
                .unwrap();
            assert!(other.grow(1, JsonInputKind::StoredJson).is_err());
            drop(work);
            other.grow(moved, JsonInputKind::StoredJson).unwrap();
            drop(other);
            let state = destination.state.lock().unwrap();
            assert_eq!((state.used, state.workspace_used), (0, 0));
        }
    }

    #[test]
    fn owned_string_transfer_never_reacquires_credit_and_keeps_json_limit_on_rejection() {
        for seed in 0..100usize {
            let limit = 2048 + seed;
            let ledger = Arc::new(WorkLedger::new(limit));
            let context = JsonOperationContext::new(ledger.clone());
            let mut work = context.workspace_reservation();
            work.grow_to(limit, limit, "operation_scratch").unwrap();
            let moved = 1 + seed * 7;
            let json = work.take_json(moved, JsonInputKind::StoredJson).unwrap();
            assert_eq!(json.reserved_bytes(), moved);
            assert_eq!(work.reserved_bytes(), limit - moved);
            {
                let state = ledger.state.lock().unwrap();
                assert_eq!((state.used, state.workspace_used), (moved, limit - moved));
            }
            let other = JsonOperationContext::new(ledger.clone());
            assert_memory_error(
                &other
                    .workspace_reservation()
                    .grow_to(1, 1, "operation_scratch")
                    .unwrap_err(),
                "operation_scratch",
                limit,
                limit as u128 + 1,
            );
            let destination = Arc::new(WorkLedger::new(limit));
            context
                .attach(destination.clone(), JsonInputKind::StoredJson)
                .unwrap();
            {
                let state = ledger.state.lock().unwrap();
                assert_eq!((state.used, state.workspace_used), (0, 0));
            }
            drop(json);
            drop(work);
            {
                let state = destination.state.lock().unwrap();
                assert_eq!((state.used, state.workspace_used), (0, 0));
            }

            let json_limit = JsonProcessingPolicyV1::WORK_BUDGET_BYTES;
            let ledger = Arc::new(WorkLedger::new(json_limit + 1024));
            let context = JsonOperationContext::new(ledger.clone());
            let mut json = context.reservation();
            json.grow(json_limit - seed, JsonInputKind::StoredJson)
                .unwrap();
            let mut work = context.workspace_reservation();
            work.grow_to(seed + 1, seed + 1, "operation_scratch")
                .unwrap();
            let error = work
                .take_json(seed + 1, JsonInputKind::StoredJson)
                .err()
                .unwrap();
            assert_eq!(error.code(), crate::ErrorCode::RecoveryError);
            assert!(error
                .detail()
                .iter()
                .any(|(key, value)| key == "limit" && value == &json_limit.to_string()));
            assert!(error
                .detail()
                .iter()
                .any(|(key, value)| key == "actual" && value == &(json_limit + 1).to_string()));
            assert_eq!(work.reserved_bytes(), seed + 1);
            {
                let state = ledger.state.lock().unwrap();
                assert_eq!(
                    (state.used, state.workspace_used),
                    (json_limit - seed, seed + 1)
                );
            }
            drop(json);
            let json = work.take_json(seed + 1, JsonInputKind::StoredJson).unwrap();
            assert_eq!(work.reserved_bytes(), 0);
            assert_eq!(json.reserved_bytes(), seed + 1);
            drop(json);
            let state = ledger.state.lock().unwrap();
            assert_eq!((state.used, state.workspace_used), (0, 0));
        }
    }

    #[test]
    fn reserved_arrays_admit_exact_capacity_and_preserve_wide_error_counts() {
        for seed in 0..100usize {
            let count = seed + 1;
            let bytes = count * std::mem::size_of::<u64>();
            let limit = 2048 + seed;
            let context = JsonOperationContext::new(Arc::new(WorkLedger::new(limit)));
            let mut held = context.workspace_reservation();
            held.grow_to(limit - bytes, limit - bytes, "operation_scratch")
                .unwrap();
            let mut values =
                ReservedVec::with_capacity(count, &context, "operation_scratch").unwrap();
            for index in 0..count {
                values.push(index as u64);
            }
            assert_eq!(values._work.bytes, bytes);
            assert_eq!(values.values.capacity(), count);
            let error = ReservedVec::<u8>::with_capacity(1, &context, "operation_scratch")
                .err()
                .unwrap();
            assert_memory_error(&error, "operation_scratch", limit, limit as u128 + 1);
            let public_values = values.into_vec();
            assert_eq!(public_values, (0..count as u64).collect::<Vec<_>>());
            let wide =
                ReservedVec::<[u8; 16]>::with_capacity(usize::MAX, &context, "operation_scratch")
                    .err()
                    .unwrap();
            assert_memory_error(
                &wide,
                "operation_scratch",
                limit,
                (limit - bytes) as u128 + (usize::MAX as u128) * 16,
            );
            held.grow_to(limit, limit, "operation_scratch").unwrap();
            let empty =
                ReservedVec::<u64>::with_capacity(0, &context, "operation_scratch").unwrap();
            assert!(empty.as_slice().is_empty());
            let mut zero_sized =
                ReservedVec::<()>::with_capacity(count, &context, "operation_scratch").unwrap();
            for _ in 0..count {
                zero_sized.push(());
            }
            assert_eq!(zero_sized.as_slice().len(), count);
            assert_eq!(zero_sized._work.bytes, 0);
        }
    }

    #[test]
    fn reserved_arrays_growth_admits_coexisting_arrays_on_both_ledgers() {
        let size = std::mem::size_of::<[u8; 257]>();
        for old in [0usize, 1, 4, 17, 99] {
            let active = 23;
            let minimum = active + (2 * old + 1) * size;
            for parent in [false, true] {
                for delta in [-1isize, 0, 1] {
                    let limit = (minimum as isize + delta) as usize;
                    let context = JsonOperationContext::new(Arc::new(WorkLedger::with_limits(
                        if parent { 2 * minimum } else { limit },
                        0,
                        if parent { limit } else { 2 * minimum },
                    )));
                    let mut held = context.workspace_reservation();
                    held.grow_to(active, active, "operation_scratch").unwrap();
                    let mut values =
                        ReservedVec::with_capacity(old, &context, "operation_scratch").unwrap();
                    for index in 0..old {
                        values.push([index as u8; 257]);
                    }
                    let pointer = values.as_slice().as_ptr();
                    let (result, allocations) =
                        crate::test_alloc::key_allocations((old + 1) * size, || {
                            let result = values.try_push([240; 257]);
                            if let Err(error) = &result {
                                assert_memory_error(
                                    error,
                                    "operation_scratch",
                                    limit,
                                    minimum as u128,
                                );
                                assert_eq!(values.capacity(), old);
                                assert_eq!(values.as_slice().as_ptr(), pointer);
                                assert_eq!(values._work.reserved_bytes(), old * size);
                                assert_eq!(values.as_slice().len(), old);
                            } else {
                                assert_eq!(values.capacity(), old + 1);
                                assert_eq!(values._work.reserved_bytes(), (old + 1) * size);
                                assert_eq!(values.as_slice().last(), Some(&[240; 257]));
                            }
                            for (index, value) in values.as_slice()[..old].iter().enumerate() {
                                assert_eq!(*value, [index as u8; 257]);
                            }
                            drop(values);
                            result.map_err(drop)
                        });
                    assert_eq!(result.is_ok(), delta >= 0);
                    assert_eq!(allocations.allocated, usize::from(delta >= 0));
                    assert_eq!(allocations.live, 0);
                    assert!(!allocations.overflow);
                    assert_eq!(context.available_workspace_bytes(), limit - active);
                    drop(held);
                    assert_eq!(context.available_workspace_bytes(), limit);
                }
            }
        }
    }

    #[test]
    fn reserved_arrays_native_growth_failure_keeps_data_and_credit_for_retry() {
        let old = 17;
        let size = std::mem::size_of::<[u8; 257]>();
        let limit = 3 * old * size;
        let context = JsonOperationContext::new(Arc::new(WorkLedger::new(limit)));
        let mut values = ReservedVec::with_capacity(old, &context, "operation_scratch").unwrap();
        for index in 0..old {
            values.push([index as u8; 257]);
        }
        let pointer = values.as_slice().as_ptr();
        let error = crate::test_alloc::fail_byte_allocation_once(2 * old * size, || {
            values.try_push([240; 257])
        })
        .unwrap_err();
        assert_eq!(error.code(), ErrorCode::IoError);
        assert_eq!(values.capacity(), old);
        assert_eq!(values.as_slice().as_ptr(), pointer);
        assert_eq!(values._work.reserved_bytes(), old * size);
        for (index, value) in values.as_slice().iter().enumerate() {
            assert_eq!(*value, [index as u8; 257]);
        }
        values.try_push([240; 257]).unwrap();
        assert_eq!(values.capacity(), 2 * old);
        assert_eq!(values._work.reserved_bytes(), 2 * old * size);
        assert_eq!(values.as_slice().last(), Some(&[240; 257]));
        drop(values);
        assert_eq!(context.available_workspace_bytes(), limit);

        let context = JsonOperationContext::new(Arc::new(WorkLedger::new(0)));
        let mut units = ReservedVec::with_capacity(0, &context, "operation_scratch").unwrap();
        for _ in 0..100 {
            units.try_push(()).unwrap();
        }
        assert_eq!(units.as_slice().len(), 100);
        assert_eq!(units._work.reserved_bytes(), 0);
    }

    #[test]
    fn reserved_arrays_keep_the_lease_until_all_elements_are_dropped() {
        struct Probe {
            context: JsonOperationContext,
        }
        impl Drop for Probe {
            fn drop(&mut self) {
                assert!(self
                    .context
                    .workspace_reservation()
                    .grow_to(1, 1, "operation_scratch")
                    .is_err());
            }
        }
        for count in 1..=100 {
            let bytes = count * std::mem::size_of::<Probe>();
            let context = JsonOperationContext::new(Arc::new(WorkLedger::new(bytes)));
            let mut values =
                ReservedVec::with_capacity(count, &context, "operation_scratch").unwrap();
            for _ in 0..count {
                values.push(Probe {
                    context: context.clone(),
                });
            }
            drop(values);
            context
                .workspace_reservation()
                .grow_to(bytes, bytes, "operation_scratch")
                .unwrap();
        }
    }

    #[test]
    fn json_and_workspace_share_one_limit_at_100_exact_boundaries() {
        for extra in 0..100 {
            let ledger = Arc::new(WorkLedger::new(1024 + extra));
            let context = JsonOperationContext::new(ledger.clone());
            let mut json = context.reservation();
            json.grow(512, JsonInputKind::JsonlRecord).unwrap();
            let mut work = context.workspace_reservation();
            assert_eq!(
                work.grow_to(512, usize::MAX, "catalog_build").unwrap(),
                512 + extra
            );
            let error = work
                .grow_to(513 + extra, usize::MAX, "catalog_build")
                .unwrap_err();
            assert_memory_error(&error, "catalog_build", 1024 + extra, 1025 + extra as u128);
            assert_eq!(work.bytes, 512 + extra);
            assert_eq!(context.available(), 0);
            let error = json.grow(1, JsonInputKind::JsonlRecord).unwrap_err();
            assert!(super::super::is_limit_error(&error));
            assert_eq!(
                error
                    .to_json()
                    .get("detail")
                    .unwrap()
                    .get("limit")
                    .unwrap()
                    .as_f64(),
                Some(512.0)
            );
            drop(work);
            json.grow(512 + extra, JsonInputKind::JsonlRecord).unwrap();
            drop(json);
            let state = ledger.state.lock().unwrap();
            assert_eq!((state.used, state.workspace_used), (0, 0));
        }
    }

    #[test]
    fn workspace_transfer_is_atomic_and_keeps_both_ownership_views() {
        let source = Arc::new(WorkLedger::new(1024));
        let target = Arc::new(WorkLedger::new(1024));
        let context = JsonOperationContext::new(source.clone());
        let mut json = context.reservation();
        json.grow(200, JsonInputKind::Filter).unwrap();
        let mut work = context.workspace_reservation();
        work.grow_to(500, 500, "catalog_build").unwrap();
        let mut busy = JsonOperationContext::new(target.clone()).reservation();
        busy.grow(400, JsonInputKind::Filter).unwrap();
        let error = context
            .attach(target.clone(), JsonInputKind::Filter)
            .unwrap_err();
        assert_memory_error(&error, "operation_scratch", 1024, 1100);
        assert_eq!(source.state.lock().unwrap().workspace_used, 500);
        assert_eq!(target.state.lock().unwrap().workspace_used, 0);
        busy.release(100);
        context
            .attach(target.clone(), JsonInputKind::Filter)
            .unwrap();
        context
            .attach(target.clone(), JsonInputKind::Filter)
            .unwrap();
        assert_eq!(source.state.lock().unwrap().workspace_used, 0);
        assert_eq!(source.state.lock().unwrap().used, 0);
        let mut later = context.clone().workspace_reservation();
        assert_eq!(later.grow_to(1, 100, "catalog_build").unwrap(), 24);
        drop(work);
        drop(json);
        drop(context);
        assert_eq!(target.state.lock().unwrap().workspace_used, 24);
        drop(later);
        drop(busy);
        let state = target.state.lock().unwrap();
        assert_eq!((state.used, state.workspace_used), (0, 0));
    }

    #[test]
    fn json_transfer_rejection_keeps_priority_with_live_workspace() {
        let source = Arc::new(WorkLedger::new(1024));
        let target = Arc::new(WorkLedger::new(1024));
        let context = JsonOperationContext::new(source.clone());
        let mut json = context.reservation();
        json.grow(600, JsonInputKind::Filter).unwrap();
        let mut work = context.workspace_reservation();
        work.grow_to(100, 100, "catalog_build").unwrap();
        let mut busy = JsonOperationContext::new(target.clone()).workspace_reservation();
        busy.grow_to(500, 500, "catalog_build").unwrap();
        let error = context
            .attach(target.clone(), JsonInputKind::Filter)
            .unwrap_err();
        assert!(super::super::is_limit_error(&error));
        assert_eq!(context.available(), 324);
        drop(busy);
        context
            .attach(target.clone(), JsonInputKind::Filter)
            .unwrap();
        assert_eq!(context.available(), 324);
        drop(json);
        drop(work);
        assert_eq!(target.state.lock().unwrap().workspace_used, 0);
    }

    #[test]
    fn workspace_minimum_totals_are_wider_than_usize_and_preference_never_rejects() {
        for limit in [0, 1, 1024, usize::MAX] {
            let context = JsonOperationContext::new(Arc::new(WorkLedger::new(limit)));
            let mut first = context.workspace_reservation();
            assert_eq!(
                first.grow_to(0, usize::MAX, "operation_scratch").unwrap(),
                limit
            );
            assert_eq!(first.grow_to(0, 0, "operation_scratch").unwrap(), limit);
            let mut second = context.workspace_reservation();
            let error = second
                .grow_to(usize::MAX, usize::MAX, "operation_scratch")
                .unwrap_err();
            assert_memory_error(
                &error,
                "operation_scratch",
                limit,
                limit as u128 + usize::MAX as u128,
            );
            assert_eq!(second.bytes, 0);
            drop(first);
            assert_eq!(
                second.grow_to(0, usize::MAX, "operation_scratch").unwrap(),
                limit
            );
        }
    }

    #[test]
    fn mixed_concurrent_reservations_and_transfers_never_overcommit() {
        let ledgers = [
            Arc::new(WorkLedger::new(1024)),
            Arc::new(WorkLedger::new(1024)),
        ];
        let workers: Vec<_> = (0..8)
            .map(|index| {
                let ledgers = ledgers.clone();
                std::thread::spawn(move || {
                    let context = JsonOperationContext::new(ledgers[index % 2].clone());
                    for step in 0..1000 {
                        let mut json = if step % 2 == 0 {
                            context.aggregate_reservation()
                        } else {
                            context.reservation()
                        };
                        let mut work = context.workspace_reservation();
                        let _ = json.grow(64, JsonInputKind::Filter);
                        let _ = work.grow_to(128, 192, "catalog_build");
                        let _ = context
                            .attach(ledgers[(index + step) % 2].clone(), JsonInputKind::Filter);
                        for ledger in &ledgers {
                            let state = ledger.state.lock().unwrap();
                            assert!(state.used + state.workspace_used <= state.limit);
                        }
                    }
                })
            })
            .collect();
        for worker in workers {
            worker.join().unwrap();
        }
        for ledger in ledgers {
            let state = ledger.state.lock().unwrap();
            assert_eq!((state.used, state.workspace_used), (0, 0));
        }
    }

    #[test]
    fn concurrent_catalog_promotions_share_the_actual_memory_ceiling() {
        let ledger = Arc::new(WorkLedger::with_limits(1024, 1536, 2048));
        let workers: Vec<_> = (0..8)
            .map(|_| {
                let ledger = Arc::clone(&ledger);
                std::thread::spawn(move || {
                    let context = JsonOperationContext::new(Arc::clone(&ledger));
                    for _ in 0..1000 {
                        let mut json = context.reservation();
                        let mut work = context.workspace_reservation();
                        let mut catalog = context.catalog_reservation();
                        let _ = json.grow(80, JsonInputKind::StoredJson);
                        let _ = work.grow_to(96, 144, "catalog_build");
                        let json_bytes = json.reserved_bytes();
                        let work_bytes = work.reserved_bytes();
                        let _ = catalog.absorb(&mut json, Some(&mut work), json_bytes, work_bytes);
                        let _ = catalog.grow(64);
                        let state = ledger.state.lock().unwrap();
                        assert!(state.used + state.workspace_used <= state.limit);
                        assert!(state.catalog_used <= state.catalog_limit);
                        assert!(
                            state.used + state.workspace_used + state.catalog_used
                                <= state.total_limit
                        );
                    }
                })
            })
            .collect();
        for worker in workers {
            worker.join().unwrap();
        }
        let state = ledger.state.lock().unwrap();
        assert_eq!(
            (state.used, state.workspace_used, state.catalog_used),
            (0, 0, 0)
        );
    }

    #[test]
    fn context_transfer_keeps_live_usage_and_rolls_back_a_failed_adoption() {
        let source = Arc::new(WorkLedger::new(1024));
        let target = Arc::new(WorkLedger::new(1024));
        let context = JsonOperationContext::new(Arc::clone(&source));
        let clone = context.clone();
        let mut input = context.reservation();
        input.grow(700, JsonInputKind::Filter).unwrap();
        let mut busy = JsonOperationContext::new(Arc::clone(&target)).reservation();
        busy.grow(400, JsonInputKind::Filter).unwrap();
        let error = context
            .attach(Arc::clone(&target), JsonInputKind::Filter)
            .unwrap_err();
        assert_eq!(
            error
                .to_json()
                .get("detail")
                .unwrap()
                .get("limit")
                .unwrap()
                .as_f64(),
            Some(624.0)
        );
        assert_eq!(source.state.lock().unwrap().used, 700);
        assert_eq!(target.state.lock().unwrap().used, 400);
        busy.release(100);
        context
            .attach(Arc::clone(&target), JsonInputKind::Filter)
            .unwrap();
        assert_eq!(source.state.lock().unwrap().used, 0);
        assert_eq!(target.state.lock().unwrap().used, 1000);
        context
            .attach(Arc::clone(&target), JsonInputKind::Filter)
            .unwrap();
        let mut next = clone.reservation();
        next.grow(24, JsonInputKind::Filter).unwrap();
        assert!(next.grow(1, JsonInputKind::Filter).is_err());
        drop(input);
        assert_eq!(target.state.lock().unwrap().used, 324);
        drop(next);
        drop(busy);
        assert_eq!(target.state.lock().unwrap().used, 0);
    }

    #[test]
    fn opposite_concurrent_context_transfers_preserve_accounting() {
        let ledgers = [
            Arc::new(WorkLedger::new(1024)),
            Arc::new(WorkLedger::new(1024)),
        ];
        let barrier = Arc::new(std::sync::Barrier::new(2));
        let threads: Vec<_> = (0..2)
            .map(|index| {
                let ledgers = ledgers.clone();
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    let context = JsonOperationContext::new(Arc::clone(&ledgers[index]));
                    let mut held = context.reservation();
                    held.grow(64, JsonInputKind::Filter).unwrap();
                    barrier.wait();
                    for step in 0..1000 {
                        context
                            .attach(
                                Arc::clone(&ledgers[(index + step) % 2]),
                                JsonInputKind::Filter,
                            )
                            .unwrap();
                    }
                })
            })
            .collect();
        for thread in threads {
            thread.join().unwrap();
        }
        for ledger in ledgers {
            assert_eq!(ledger.state.lock().unwrap().used, 0);
        }
    }

    #[test]
    fn simultaneous_steps_share_one_quota_and_released_capacity_is_reusable() {
        let ledger = Arc::new(WorkLedger::new(1024));
        let operation = JsonOperationContext::new(Arc::clone(&ledger));
        let mut input = operation.reservation();
        let mut copy = operation.clone().reservation();
        input.grow(700, JsonInputKind::Filter).unwrap();
        copy.grow(324, JsonInputKind::Filter).unwrap();
        let error = copy.grow(1, JsonInputKind::Filter).unwrap_err();
        assert_eq!(
            error
                .to_json()
                .get("detail")
                .unwrap()
                .get("limit")
                .unwrap()
                .as_f64(),
            Some(1024.0)
        );
        assert_eq!(ledger.state.lock().unwrap().used, 1024);
        drop(input);
        copy.grow(700, JsonInputKind::Filter).unwrap();
        drop(copy);
        assert_eq!(ledger.state.lock().unwrap().used, 0);
    }

    #[test]
    fn merging_reservations_preserves_ownership_without_charging_twice() {
        let ledger = Arc::new(WorkLedger::new(1024));
        let operation = JsonOperationContext::new(Arc::clone(&ledger));
        let mut first = operation.reservation();
        let mut second = operation.reservation();
        first.grow(512, JsonInputKind::DocumentMetadata).unwrap();
        second.grow(512, JsonInputKind::ChunkMetadata).unwrap();
        first.merge(second);
        assert_eq!(ledger.state.lock().unwrap().used, 1024);
        drop(operation);
        drop(first);
        assert_eq!(ledger.state.lock().unwrap().used, 0);
    }

    #[test]
    fn different_operations_observe_effective_available_limit_without_double_charging() {
        let ledger = Arc::new(WorkLedger::new(2000));
        let first = JsonOperationContext::new(Arc::clone(&ledger));
        let second = JsonOperationContext::new(Arc::clone(&ledger));
        let mut held = first.reservation();
        held.grow(800, JsonInputKind::DocumentMetadata).unwrap();
        let mut other = second.reservation();
        other.grow(1200, JsonInputKind::StoredJson).unwrap();
        let error = other.grow(1, JsonInputKind::StoredJson).unwrap_err();
        assert_eq!(error.code(), crate::ErrorCode::RecoveryError);
        assert_eq!(
            error
                .to_json()
                .get("detail")
                .unwrap()
                .get("limit")
                .unwrap()
                .as_f64(),
            Some(1200.0)
        );
        drop(held);
        other.grow(800, JsonInputKind::StoredJson).unwrap();
        drop(other);
        assert_eq!(ledger.state.lock().unwrap().used, 0);
    }

    #[test]
    fn operation_limit_applies_even_when_the_shared_ledger_is_larger() {
        let operation = JsonOperationContext::new(Arc::new(WorkLedger::new(usize::MAX)));
        let mut reservation = operation.reservation();
        reservation
            .grow(
                JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
                JsonInputKind::Weights,
            )
            .unwrap();
        let error = reservation
            .grow(usize::MAX, JsonInputKind::Weights)
            .unwrap_err();
        let json = error.to_json();
        let detail = json.get("detail").unwrap();
        assert_eq!(
            detail.get("limit").unwrap().as_f64(),
            Some(JsonProcessingPolicyV1::WORK_BUDGET_BYTES as f64)
        );
        assert!(detail.get("actual").is_none());
        assert_eq!(
            detail.get("weights_reason").unwrap().as_str(),
            Some("json_processing_limit")
        );
    }

    #[test]
    fn rejected_buffer_growth_does_not_allocate_or_consume_a_reservation() {
        let ledger = Arc::new(WorkLedger::new(80));
        let operation = JsonOperationContext::new(Arc::clone(&ledger));
        let mut reservation = operation.reservation();
        let mut values = Vec::<u64>::new();
        reservation
            .reserve_vec(&mut values, 10, JsonInputKind::JsonlRecord)
            .unwrap();
        values.resize(10, 0);
        assert!(reservation
            .reserve_vec(&mut values, 1, JsonInputKind::JsonlRecord)
            .is_err());
        assert_eq!(values.capacity(), 10);
        assert_eq!(ledger.state.lock().unwrap().used, 80);
        drop(reservation);
        assert_eq!(ledger.state.lock().unwrap().used, 0);
    }

    #[test]
    fn json_byte_buffers_native_failure_preserves_data_and_refunds_only_new_credit() {
        for kind in [
            JsonInputKind::JsonlRecord,
            JsonInputKind::Filter,
            JsonInputKind::Weights,
            JsonInputKind::MetadataBoost,
            JsonInputKind::StoredJson,
        ] {
            let context = JsonOperationContext::standalone();
            let capacity = context.available_workspace_bytes();
            let mut other = context.reservation();
            other.grow(17, kind).unwrap();
            let baseline = context.available_workspace_bytes();
            let bytes = 8193;
            let mut work = context.reservation();
            let mut values = Vec::<u8>::new();
            let error = crate::test_alloc::fail_byte_allocation_once(bytes, || {
                work.reserve_vec(&mut values, bytes, kind)
            })
            .unwrap_err();
            assert_eq!(error.code(), ErrorCode::IoError);
            assert!(error.detail().is_empty());
            assert!(!super::super::is_memory_budget_error(&error));
            assert!(values.is_empty());
            assert_eq!(values.capacity(), 0);
            assert_eq!(work.reserved_bytes(), 0);
            assert_eq!(context.available_workspace_bytes(), baseline);
            work.reserve_vec(&mut values, bytes, kind).unwrap();
            values.resize(bytes, 23);
            let pointer = values.as_ptr();
            let error = crate::test_alloc::fail_byte_allocation_once(2 * bytes, || {
                work.reserve_vec(&mut values, 1, kind)
            })
            .unwrap_err();
            assert_eq!(error.code(), ErrorCode::IoError);
            assert!(error.detail().is_empty());
            assert_eq!(values.as_ptr(), pointer);
            assert_eq!(values.capacity(), bytes);
            assert_eq!(values.len(), bytes);
            assert!(values.iter().all(|byte| *byte == 23));
            assert_eq!(work.reserved_bytes(), bytes);
            assert_eq!(context.available_workspace_bytes(), baseline - bytes);
            work.reserve_vec(&mut values, 1, kind).unwrap();
            values.push(24);
            assert_eq!(values.last(), Some(&24));
            drop(values);
            drop(work);
            assert_eq!(context.available_workspace_bytes(), baseline);
            drop(other);
            assert_eq!(context.available_workspace_bytes(), capacity);
        }
    }

    #[test]
    fn json_strings_native_failure_preserves_data_and_refunds_only_new_credit() {
        let payload = "x".repeat(8193);
        for kind in [
            JsonInputKind::JsonlRecord,
            JsonInputKind::Filter,
            JsonInputKind::Weights,
            JsonInputKind::MetadataBoost,
            JsonInputKind::StoredJson,
        ] {
            let context = JsonOperationContext::standalone();
            let capacity = context.available_workspace_bytes();
            let mut other = context.reservation();
            other.grow(17, kind).unwrap();
            let baseline = context.available_workspace_bytes();
            let bytes = payload.len();
            let mut work = context.reservation();
            let mut value = String::new();
            let error = crate::test_alloc::fail_byte_allocation_once(bytes, || {
                work.reserve_string(&mut value, bytes, kind)
            })
            .unwrap_err();
            assert_eq!(error.code(), ErrorCode::IoError);
            assert!(error.detail().is_empty());
            assert!(!super::super::is_memory_budget_error(&error));
            assert!(value.is_empty());
            assert_eq!(value.capacity(), 0);
            assert_eq!(work.reserved_bytes(), 0);
            assert_eq!(context.available_workspace_bytes(), baseline);
            work.reserve_string(&mut value, bytes, kind).unwrap();
            value.push_str(&payload);
            let pointer = value.as_ptr();
            let error = crate::test_alloc::fail_byte_allocation_once(2 * bytes, || {
                work.reserve_string(&mut value, 1, kind)
            })
            .unwrap_err();
            assert_eq!(error.code(), ErrorCode::IoError);
            assert!(error.detail().is_empty());
            assert_eq!(value.as_ptr(), pointer);
            assert_eq!(value.capacity(), bytes);
            assert_eq!(value, payload);
            assert_eq!(work.reserved_bytes(), bytes);
            assert_eq!(context.available_workspace_bytes(), baseline - bytes);
            work.reserve_string(&mut value, 1, kind).unwrap();
            value.push('y');
            assert!(value.ends_with('y'));
            drop(value);
            drop(work);
            assert_eq!(context.available_workspace_bytes(), baseline);
            drop(other);
            assert_eq!(context.available_workspace_bytes(), capacity);
        }
    }

    #[test]
    fn concurrent_reservations_never_exceed_the_shared_ledger() {
        let ledger = Arc::new(WorkLedger::new(1024));
        let mut workers = Vec::new();
        for _ in 0..8 {
            let ledger = Arc::clone(&ledger);
            workers.push(std::thread::spawn(move || {
                let operation = JsonOperationContext::new(Arc::clone(&ledger));
                for _ in 0..1000 {
                    let mut reservation = operation.reservation();
                    let _ = reservation.grow(256, JsonInputKind::ChunkMetadata);
                    assert!(ledger.state.lock().unwrap().used <= 1024);
                }
            }));
        }
        for worker in workers {
            worker.join().unwrap();
        }
        assert_eq!(ledger.state.lock().unwrap().used, 0);
    }
}
